<!-- page 1 -->
1
HPC: Hierarchical Point-based Latent
Representation for Streaming Dynamic Gaussian
Splatting Compression
Yangzhi Ma, Bojun Liu, Wenting Liao, Dong Liu, Senior Member, IEEE
Zhu Li, Senior Member, IEEE, and Li Li†, Senior Member, IEEE,
Abstract—While dynamic Gaussian Splatting has driven signif-
icant advances in free-viewpoint video, maintaining its rendering
quality with a small memory footprint for efficient streaming
transmission still presents an ongoing challenge. Existing stream-
ing dynamic Gaussian Splatting compression methods typically
leverage a latent representation to drive the neural network
for predicting Gaussian residuals between frames. Their core
latent representations can be categorized into structured grid-
based and unstructured point-based paradigms. However, the
former incurs significant parameter redundancy by inevitably
modeling unoccupied space, while the latter suffers from limited
compactness as it fails to exploit local correlations. To relieve
these limitations, we propose HPC, a novel streaming dynamic
Gaussian Splatting compression framework. It employs a hierar-
chical point-based latent representation that operates on a per-
Gaussian basis to avoid parameter redundancy in unoccupied
space. Guided by a tailored aggregation scheme, these latent
points achieve high compactness with low spatial redundancy.
To improve compression efficiency, we further undertake the first
investigation to compress neural networks for streaming dynamic
Gaussian Splatting through mining and exploiting the inter-
frame correlation of parameters. Combined with latent com-
pression, this forms a fully end-to-end compression framework.
Comprehensive experimental evaluations demonstrate that HPC
substantially outperforms state-of-the-art methods. It achieves a
storage reduction of 67% against its baseline while maintaining
high reconstruction fidelity.
Index Terms—Dynamic Gaussian Splatting Compression, Free-
viewpoint Video, Latent Representation
I. INTRODUCTION
W
ITH the advent of immersive media, Free-Viewpoint
Video (FVV) technology, which provides real-time
viewing from arbitrary perspectives, has played a pivotal role
in driving the evolution of next-generation media systems. An
efficient, high-quality FVV framework has the potential to
unlock a variety of applications, such as virtual reality (VR)
and augmented reality (AR). For this purpose, prior works
have investigated various scene representations, which involve
meshes, point clouds, and radiance fields.
Among these candidates, the recently emerged 3D Gaussian
Splatting (3DGS) [1] is opening up substantial new avenues
Y.
Ma,
B.
Liu,
W.
Liao,
D.
Liu
and
L.
Li
are
with
MoE
Key
Laboratory
of
Brain-inspired
Intelligent
Perception
and
Cogni-
tion, University of Science and Technology of China, Hefei 230027,
China (e-mail: mayz@mail.ustc.edu.cn; liubj@mail.ustc.edu.cn; liaowent-
ing@mail.ustc.edu.cn; dongeliu@ustc.edu.cn; lil1@ustc.edu.cn.)
Z.
Li
is
with
the
University
of
Missouri,
Kansas
City.
(e-mail:
lizhu@umkc.edu)
† denotes the corresponding author.
for FVV application. This representation excels at handling
complex visual content while supporting real-time render-
ing. To meet the low-latency demands of real-time video
applications, such as streaming media and immersive live
broadcasting, pioneering works have extended 3DGS into an
online-optimized dynamic framework [2]–[5]. These frame-
works incrementally optimize the inter-frame deformation on a
per-frame basis, thus facilitating the on-the-fly reconstruction.
However, despite their essential advantages, the substantial
size of the dynamic Gaussian Splatting representation poses
significant challenges for streaming transmission. To this end,
integrating compression into dynamic Gaussian Splatting has
become a key research focus.
In the context of compression, a prevalent strategy for
dynamic Gaussians employs a latent representation coupled
with several neural networks for prediction [6]–[9]. The latent
representation and neural networks are jointly optimized, com-
pressed, and transmitted. Within this framework, the design of
the latent representation emerges as one of the central chal-
lenges. Drawing inspiration from its success in earlier radiance
field studies, the structural latent grid [10], [11] has been
introduced to Gaussian Splatting [2], [6]–[8], [12]. Within this
representation, the learnable latent embeddings are allocated to
the vertices of the structural grid to generate a full-space latent
field. During inference, the field is queried at the positions
of individual Gaussians to retrieve their corresponding latent
embeddings. However, assigning latent embeddings across
the entire space incurs a prohibitive parameter explosion,
rendering it impractical for direct optimization and storage.
Therefore, such grid-based methods typically resort to either
low-resolution grids [6] or hashing with collisions [2], [7],
[8] to reduce parameters, which inevitably compromises the
performance.
Alternatively, given that the positions of Gaussians consti-
tute discrete sets occupying a vanishingly small fraction of
the 3D space [13], it is naturally to attach a latent embedding
directly to each point’s position, ignoring unoccupied areas.
However, the existing point-based latent representation [9]
primarily focuses on the isolated point, failing to model corre-
lations within local neighborhoods and thus incurring spatial
redundancy. Furthermore, given the non-uniform distribution
of Gaussians, its single-scale design fails to capture the varied
spatial characteristics, resulting in suboptimal performance.
In addition to the challenges in latent representation, another
critical issue lies in the deficient compression strategy. Existing
arXiv:2602.00671v1  [cs.CV]  31 Jan 2026

<!-- page 2 -->
2
studies [6]–[8] have devoted to compressing the latent repre-
sentation through rate-aware optimization and entropy coding,
but overlook the compression of the neural network parame-
ters. This limitation not only prevents further bitrate reduction
but also results in an imbalanced rate allocation between the
latent and the network parameters. This creates a demand for
an integrated framework that co-optimizes and compresses the
latent representation alongside the neural network parameters.
In response to these limitations, we propose HPC, a novel
streaming dynamic Gaussian Splatting framework that inte-
grates a Hierarchical Point-based latent representation with
a fully end-to-end Compression strategy. On one hand, our
latent representation is built upon discrete points that operates
on a per-Gaussian basis to avoid parameter redundancy in
unoccupied space. To align with the non-uniform distribution
of Gaussians, we equip the representation with multi-scale
receptive fields via progressive downsampling, yielding the
latent hierarchy. We further devise an aggregation scheme
that incorporates both inner-scale and cross-scale processing
to reduce spatial redundancy. It promotes local information
sharing at multiple resolutions, thus achieving compactness.
On the other hand, we undertake the first investigation into
neural network compression for streaming dynamic Gaussian
Splatting. Through an analysis of parameter distributions,
we identify potential temporal redundancy between adjacent
frames. Building on this observation, we derive a temporal
reference strategy that leverages the latest decoded parameters
to guide the optimization and compression of current-frame
parameters. This strategy enables us to encode the energy-
compact inter-frame residual rather than full parameters, sig-
nificantly lowering the bitrate consumption. Unified with the
compression of latent representation, our framework conducts
a fully end-to-end optimization that balances the bitrate be-
tween the latent and network parameters, thereby attaining
superior rate-distortion performance.
In summary, the proposed HPC offers the following key
contributions:
• The HPC framework introduces a novel hierarchical
latent point representation to match the inhomogeneous
distribution of Gaussians while maintaining parameter
efficiency. Spatial redundancy is effectively reduced via
a dedicated aggregation mechanism that operates both
within and across scales, promoting local information
sharing and resulting in a compact representation.
• HPC pioneers the compression of network parameters
in streaming dynamic Gaussian Splatting by exploiting
inter-frame temporal redundancy. The resulting tempo-
ral reference strategy encodes compact residuals and is
combined with latent compression in our fully end-to-
end framework, enabling optimal bitrate allocation.
• Extensive experimental results demonstrate that HPC
achieves a remarkable rate-distortion performance against
state-of-the-art methods. Comprehensive ablations and
analyses substantiate the effectiveness of each proposed
component.
The remainder of this paper is structured as follows.
Section II reviews related work in the relevant fields. The
preliminaries of our framework are provided in Section III.
Sections IV and V detail the proposed HPC methodology.
Experimental configurations and results are presented and
analyzed in Section VI. Finally, Section VII concludes the
paper.
II. RELATED WORK
A. Static Gaussian Splatting Compression
To address the substantial data footprint of 3DGS, Gaussian
Splatting Compression seeks to eliminate redundancy from the
Gaussian representations while preserving rendering fidelity.
This section presents an overview of static Gaussian Splatting
compression, covering two primary branches: post-training
compression and generative compression.
Post-training compression decouples the compression pro-
cess from the 3DGS training [14]–[19]. By directly operating
on a pre-trained 3DGS model, it enables a fast compression
pipeline without introducing additional optimization over-
head. Existing post-training methods often leverage established
transform [14] or encoding tools [15], [20].
For the optimal rate-distortion trade-off, generative com-
pression methods perform joint training of compression and
3DGS representation [17], [21]–[32]. Some methods directly
reduce the number of Gaussians by employing pruning [21],
[22] or masking [23], [24] techniques to discard unimportant
elements. Others apply vector quantization [23], [24], [26],
[27] to reduce attribute-level redundancy. By leveraging the
local spatial redundancy of 3DGS, Scaffold-GS [28] constructs
a more efficient representation where a region of Gaussians is
clustered to a few anchor points with an implicit representa-
tion, which drastically cuts down the storage count. Building
upon Scaffold-GS, several works [29]–[32] take one more step
to introduce dedicated entropy models and end-to-end rate-
distortion optimization to achieve optimal performance.
B. Dynamic Gaussian Splatting Representation
Dynamic Gaussian Splatting extends 3DGS into the tem-
poral domain to meet the demands of FVV applications. To
accommodate diverse application requirements, mainstream
representations can be broadly categorized into two paradigms:
offline optimization and online optimization. The former takes
an entire video sequence or multiple frames as its basic unit
for both training and transmission, allowing it to leverage a
richer temporal context for reference. The latter processes the
individual frame as the basic unit. By modeling inter-frame
changes incrementally on a per-frame basis, this approach
naturally supports streaming applications.
For offline methods, one paradigm extends Gaussians into
a 4D spatiotemporal domain, where each Gaussian captures a
localized region in both space and time [33]–[38]. Another line
of work basically applies the deformation to the canonical-
space Gaussians [39]–[43]. However, these approaches face
limitations in scenarios demanding streaming or involving long
video sequences.
In contrast, online methods process 4D scenes in an it-
erative manner, adapting to changes on a frame-by-frame

<!-- page 3 -->
3
basis to support streaming. Representations for online dy-
namic Gaussians fall into two main paradigms: the structured
grid and the unstructured point representation. Following the
former paradigms, 3DGStream [2] utilizes a hash-grid [10]
to implicitly model inter-frame motion. Similarly, 4DGC [6]
employs a latent grid for motion prediction, supplementing
it with a compensation strategy to reduce information loss.
iFVC [7] builds upon Scaffold-GS [28] and proposes a binary
tri-plane for efficient residual prediction. However, since as-
signing features across the entire space incurs a prohibitive
memory footprint, such representations are inherently limited
by hash collisions [2], [7] or coarse resolution [6], resulting
in suboptimal expressiveness.
In the point representation paradigm, parameters are directly
attached to each Gaussian’s position, thereby eliminating the
parameter waste associated with unoccupied regions. Follow-
ing this principle, HiCoM [3] optimizes the hierarchical coher-
ent motion for efficient prediction. As the extension of HiCoM,
ReCon-GS [4] introduces a dynamic hierarchy reconfiguration
strategy for enhancement. ComGS [5] leverages a key-point
motion model alongside a key-frame strategy, enabling effi-
cient dynamic reconstruction. Despite their advantage of rapid
training, these methods suffer from limited representational
compactness, as they optimize and operate on the motion
directly in the raw parameter space. Instead, another pathway
is optimizing a latent representation, which relies on a neural
network to learn the prediction of the motion or residual.
Within this paradigm, QUEEN [9] operates by assigning a
latent to each Gaussian and employs a jointly optimized latent
decoder for residual inference. However, its per-point model-
ing scheme fails to capture correlations within local neighbor-
hoods, incurring spatial redundancy. Moreover, its single-scale
design is ill-suited to the non-uniformly distributed Gaussians,
unable to adaptively capture spatial characteristics.
C. Compression of Dynamic Gaussian Splatting and Implicit
Neural Video Representation
Following a similar taxonomy to its representation, com-
pression for dynamic Gaussian Splatting can also be divided
into two primary categories: offline and online. In offline
methods, MEGA [37] achieves compression by decoupling
color attributes and employing entropy-constrained deforma-
tion. While both GIFstream [42] and 4DGV [43] leverage
established video codecs to achieve efficient compression.
Regarding online compression methods, QUEEN [9] quan-
tizes the latent representation and then sparsifies the po-
sitional residuals via a learned gating module for storage
saving. iFVC [7] quantizes its tri-plane latent embeddings
in binary and leverages the empirical frequency distribution
for entropy estimation, thereby facilitating end-to-end rate-
distortion optimization. 4DGC [6] adopts a factorized entropy
model [44] for both its motion grid and its compensated
Gaussians. Building upon this, 4DGCPro [8] specifically en-
ables progressive streaming and real-time coding capabilities.
Existing methods primarily focus on compressing the latent
representation. A critical oversight, however, is that none of
them jointly consider the neural network parameters. This
omission not only hinders further bitrate reduction but also
leads to an inefficient bit allocation.
To bridge this gap, relevant insights can be adopted from
the field of Implicit Neural Video Representation [45]–[50],
where compressing the neural network itself has been a central
research focus and offers well-established strategies. As a
pioneer in this field, NeRV [45] applies pruning, quantization,
and entropy coding directly to the neural network parameters.
The follow-up methods [46], [47] integrate a rate estimation
network for rate-aware training, and employ either statistical
frequencies or a manually designed entropy model during
inference. Zhang et al. [48] introduce a network-free entropy
model by leveraging global statistics to ensure consistency be-
tween training and inference. To capture dependencies among
network parameters, subsequent works [49], [50] introduce
context-aware autoregressive modeling, which significantly
enhances compression efficiency. Building on these insights,
our goal is to design a neural network compression strategy
tailored to the characteristics of dynamic Gaussian splatting
and the demands of streaming. Combining with latent repre-
sentation compression, we are aiming to build a fully end-to-
end rate-distortion optimization framework.
III. PRELIMINARIES
A. 3D Gaussian Splatting
3DGS [1] achieves real-time, free-viewpoint rendering by
combining a multitude of Gaussians with differentiable splat-
ting and tile-based rasterization. Each Gaussian is parameter-
ized by a set of attributes. Its spatial extent and shape are
defined by a mean µ ∈R3 and a covariance matrix Σ ∈R3×3,
G(x) = exp

−1
2(x −µ)⊤Σ−1(x −µ)

,
(1)
where x ∈R3 is an arbitrary position in the 3D space, and Σ
is constructed from a scaling matrix S and a rotation matrix
R as Σ = RSS⊤R⊤. Following this, the rendered pixel
color C is synthesized via α-blending, which composites these
overlapping splats as follows:
C =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αj),
(2)
where N is the set of sorted Gaussians contributing to the
pixel, αi is the opacity of the i-th Gaussian after projection,
and ci is its view-dependent color obtained from SH.
B. Neural Gaussian Representation
Scaffold-GS [28] enhances the compactness and fidelity of
3DGS by eliminating its local spatial redundancy through an
implicit representation. It introduces Neural Gaussians NG
as the fundamental representation units that leverage a voxel-
based clustering mechanism to manage the M local Gaussians.
Each Neural Gaussian is defined by a set of attributes:
NG = {X ∈R3, o ∈R3×M, F ∈RD, l ∈R3},
(3)
where X denotes the location of the anchor obtained through
voxelization, o represents the offsets of the M managed local

<!-- page 4 -->
4
Hierarchical Point-based Latent Representation
Up
Up
Prediction 
Heads
Deformation
Cross-scale Latent Aggregation
𝑇𝑡
𝑋
𝑅𝑡
𝑋
∆𝑜𝑡
∆𝐹𝑡
𝒩𝒢𝑡−1
𝑋𝑡−1
𝒩𝒢𝑡
Position 𝑋
Offset 𝑜
Scaling 𝑙
Feature 𝐹
Position 
Offset
Gaussian Primitive
𝒩𝒢
Neural Gaussian
Down
Up
Grid Sampling
𝑋𝑡−1
2
𝑍𝑡
2
ILA
𝑋𝑡−1
1
𝑍𝑡
1
ILA
C
𝐸𝑡
0
ILA
𝑋𝑡−1
0
𝑍𝑡
0
C
Down
Down
𝑃𝑡
𝜖𝑟
Scale 𝑟
Scale 𝑟−1
Bitstream
Encode Process
Decode Process
Concatenate
Parameter Loading
Parameter Sharing
Inner-scale Latent Aggregation
Grid Down-sampling
Grid Up-sampling
෠𝐸𝑡
0
𝐸𝑡
1
෠𝐸𝑡
1
𝐸𝑡
2
෠𝐸𝑡
2
෠𝑃𝑡
Fig. 1: Pipeline of the proposed HPC framework. The framework begins with the latest decoded Neural Gaussians NGt−1
as a reference. It then constructs a hierarchical latent representation {Zr
t }L−1
r=0 (here, L = 3) by progressively down-sampling
their positions into {Xr
t−1}L−1
r=0 and pairing them with the decoded latent embeddings { ˆEr
t }L−1
r=0 . After the Inner-scale Latent
Aggregation (ILA) and Cross-scale Latent Aggregation (CLA), these latent points are fed into the prediction heads to obtain
inter-frame residuals for deformation. In HPC, both the latent embeddings {Er
t }L−1
r=0 and network parameters Pt are compressed
for transmission. We denote the reconstructed elements from the decoder with a hat mark.
Gaussians, and l is a scaling factor that regularizes their spatial
distribution. Leveraging these attributes, the positions of the
M local Gaussians can be computed as:
{µi}M−1
i=0
= X + {oi}M−1
i=0
· l.
(4)
The remaining attributes of the local Gaussians are decoupled
from the implicit feature F through several dedicated MLPs
ΦS, conditioned on the relative viewing information δ:
{ci, Ri, Si, αi}M−1
i=0
= ΦS(F, δ).
(5)
IV. DYNAMIC GAUSSIAN REPRESENTATION
Building upon the success of Neural Gaussians for com-
pressing static 3DGS scenes [29]–[31], we extend this rep-
resentation to dynamic scenes by employing a sequence of
Neural Gaussians {NGt}T −1
t=0 as the scene representation for
each frame, where t denotes the timestep, and T is the total
frame number. Following the previous work [7], we leverage
HAC [29] to generate compact NG0 for the initial frame.
The reconstruction pipeline of succeeding frames is illus-
trated in Fig. 1. Given a timestep t, we leverage the latest
decoded Neural Gaussians NGt−1 as the starting point. To
model the temporal variations efficiently, a hierarchical point-
based latent representation (Sec. IV-A) is constructed by pro-
gressively down-sampling the positions of NGt−1 and pairing
them with the learnable latent embeddings. To achieve com-
pactness, the latent representation undergoes a two-stage ag-
gregation process: inner-scale aggregation followed by cross-
scale aggregation. Subsequently, the aggregated latent rep-
resentation guides the deformation from NGt−1 to NGt
(Sec. IV-B), reconstructing the scene of the current timestep
through a motion affine transformation and feature compensa-
tion.
A. Hierarchical Point-based Latent Representation
To align with the inherently discrete and non-uniform
spatial distribution of Neural Gaussians, we propose using a
hierarchical point-based latent representation for inter-frame
prediction. The construction of hierarchical latent points is
illustrated in Fig. 1. We establish the spatial hierarchy by
first initializing the finest scale from the primary positions
as X0
t−1 = Xt−1, then recursively obtaining coarser scales
through grid sampling [51]:
Xr
t−1 =

Xr−1
t−1 /ϵr + 0.5

· ϵr,
r = 1, 2, . . . , L −1,
(6)
where r, ϵ, and L denote the scale index, grid size, and
total scale count, respectively. For each resulting position
Xr
t−1, we initialize a corresponding latent embedding Er
t .
After optimization, compression, and transmission, the de-
coder receives the reconstructed versions ˆEr
t , which are then
combined with the corresponding Xr
t−1 to construct the the
complete hierarchical latent points {Zr
t = (Xr
t−1, ˆEr
t )}L−1
r=0 .
With this design, the latent points are feasible to capture local
spatial character across different scales, enabling them to better
capture the temporal dynamics of non-uniformly distributed
Neural Gaussians while achieving compactness for efficient
storage and transmission.
To this end, we leverage the established latent representation
by progressively aggregating the latent points from the coarsest

<!-- page 5 -->
5
Targeted Point
kNN
𝑑𝑋𝑘𝑁𝑁
𝑟
𝐸𝑘𝑁𝑁
𝑟
MLP
×
MLP
𝑍𝑘𝑁𝑁
𝑟
𝑍𝑡
𝑟
Softmax
Σ
Fig. 2: Inner-scale Latent Aggregation (ILA). ILA takes a
target point as input and locates its k-nearest neighbors. It
then predicts aggregated weights from the relative positions,
performs a weighted sum of the neighbor embeddings, and
finally passes the result through an MLP to produce the output.
to the finest scale. We factorize this aggregation process into
two complementary components: Inner-scale Latent Aggrega-
tion (ILA) and Cross-scale Latent Aggregation (CLA).
Inner-scale Latent Aggregation. Given the latent points Zr
t =
(Xr
t−1, ˆEr
t ), the ILA module is designed to aggregate the em-
bedding of a point’s neighbors to model local characteristics.
As illustrated in Fig. 2, we first introduce the kNN algorithm to
search for the k nearest neighbors for every single latent point.
Following this, the relative positional offset dXr
kNN between
the target latent point and its neighbors is incorporated as
auxiliary information to guide the aggregation. Specifically,
these relative positions are fed into an MLP to predict per-
neighbor aggregation weights which are normalized via soft-
max. The neighbor embeddings ˆEr
kNN are then adaptively
fused according to these weights. Ultimately, this fusion is
passed through a final MLP to integrate information across
all channels, producing the aggregated result. To reduce the
parameter count, the ILA modules at different scales share
the same set of parameters. Such an aggregation scheme facil-
itates interaction among neighboring latent points, promoting
redundancy reduction and achieving spatial compactness.
Cross-scale Latent Aggregation. Building upon ILA, we next
introduce the CLA process to fuse information across different
resolutions. As illustrated in Fig. 1, CLA operates the inner-
scale-aggregated latent points progressively from coarse to fine
scales. At each step, coarse-scale embeddings are upsampled
by copying them to their corresponding fine-scale positions,
thereby recovering resolution. These upsampled points are
then concatenated with the native fine-scale latent points to
enable cross-scale fusion. Such a process is repeated recur-
sively until the original resolution is reached, ultimately pro-
ducing the cross-scale aggregated latent points for subsequent
prediction. Through this CLA process, the aggregated latent
representation integrates local spatial context captured across
multiple receptive fields. Such a powerful context enables it
to effectively model the underlying characteristics of non-
uniformly distributed Neural Gaussians, thereby providing a
superior foundation for subsequent prediction.
B. Dynamic Neural Gaussian Deformation
Within the aggregated latent points, we proceed to predict
the Neural Gaussians NGt = {Xt, ot, Ft, lt} for the cur-
rent timestep, conditioned on the previous state NGt−1 =
{Xt−1, ot−1, Ft−1, lt−1}. In this deformation, the scaling fac-
tor l is held constant as its variation has been proven to degrade
performance [7], whereas X, o, and F are updated. Given this
setup, the deformation is implemented via two complementary
mechanisms: a residual compensation for the implicit feature
and a motion affine transformation for geometric deformation.
Specifically, the aggregated latent points are first passed
through the MLP-based prediction heads, yielding a set of
deformation parameters: feature residual ∆Ft, anchor transla-
tion T X
t , anchor rotation RX
t , and offset residual ∆ot. For the
implicit features, the compensation is achieved through adding
the predicted residual ∆Ft:
Ft = Ft−1 + ∆Ft.
(7)
Geometric deformation is modeled by a motion affine trans-
formation that combines global and local adjustments. This
transformation updates the anchor position through translation
T X
t . Additionally, each local offset receives an extra individual
adjustment ∆ot. To enable a more expressive geometric trans-
formation, we adopt the strategy from [42] of applying a global
rotation RX
t
to all corresponding offsets. Taken together, the
complete update can be concisely expressed as:
Xt = Xt−1 + T X
t ,
(8)
ot = RX
t (ot−1 + ∆ot).
(9)
Under such a mechanism, the Neural Gaussians gain sufficient
expressive power to model temporal variation, enabling high-
fidelity rendering through Neural Gaussian Splatting [28].
V. COMPRESSION SCHEME
A tailored compression scheme is integrated into HPC,
aiming to fulfill the transmission requirements of streaming
FVV. In our framework, the elements to be compressed
and transmitted are the latent embedding Et = {Er
t }L−1
r=0
(Sec. V-A) and the neural network parameters Pt (Sec. V-B).
Especially, the compression of neural network parameters
has been largely overlooked in streaming dynamic Gaussian
Splatting, where existing methods [2], [6]–[9] simply transmit
full-precision 32-bit floats. To bridge this gap and enhance
rate-distortion efficiency, our work presents a novel scheme for
compressing the streaming neural network parameters. Finally,
we combine the compression of both parts and derive the
overall rate-distortion objective to realize a fully end-to-end
optimization (Sec. V-C).
A. Latent Embedding Compression
By virtue of the ILA and CLA design, the optimized latent
embeddings Et are compact with reduced spatial redundancy,
ensuring they are well-suited for compression. As illustrated
in Fig. 3 (a), the compression pipeline includes a quantization
operation that discretizes the latent embeddings into integers,
followed by an entropy coding module.
To circumvent the non-differentiability of quantization dur-
ing training, we adopt the common practice from prior
works [46]–[48] of employing distinct quantization proxies

<!-- page 6 -->
6
𝐸𝑡
Q
AE
AD
Factorized
Entropy Model
෠𝐸𝑡
Δ𝑃𝑡
෠𝑃𝑡−1
෠𝑃𝑡
Q
AE
AD
DQ
Gaussian
Entropy Model
𝜂𝑡
𝜇Δ ෨𝑃𝑡, 𝜎Δ ෨𝑃𝑡
×
＋
GOP
P-frame
I-frame
…
Reference flow
Q
AE
AD
DQ
෠𝑃𝑡
Gaussian
Entropy Model
𝑃𝑡
(a) Latent Embedding Compression 
(c) I-frame Neural Network Compression 
(d) P-frame Neural Network Compression 
(b) Group of Pictures for Neural Network Compression
Q
Quantization
AE
Arithmetic Encoder
AD
Arithmetic Decoder
DQ
De-quantization
Δ ෠𝑃𝑡
𝜇෨𝑃𝑡, 𝜎෨𝑃𝑡
Fig. 3: HPC’s compression scheme. We incorporate compression for both the latent embeddings and the neural networks.
for distortion calculation and entropy estimation to obtain
the quantized embedding ˆEt. Specifically, the straight-through
estimator (STE) is adopted for distortion calculation:
ˆEt = SG(⌊Et⌉−Et) + Et,
(10)
where SG(·) denotes the gradient-stopping operation. For
entropy estimation, the rounding quantization is substituted
with adding a uniform noise u:
ˆEt = Et + u,
u ∼U(−1
2, 1
2).
(11)
After quantization, the widely used factorized model [44]
is adopted for entropy estimation. The entropy model com-
prises several learnable layers to progressively refine the input
embedding to calculate their probabilities and bitrate:
pPMF( ˆEt) = pCDF( ˆEt + 1
2) −pCDF( ˆEt −1
2),
(12)
R( ˆEt) = E ˆ
Et[−log2 pPMF( ˆEt)],
(13)
where pPMF(·) is the probability mass function, pCDF(·) is the
cumulative distribution function approximated by the factor-
ized model, and R( ˆEt) is the estimated bitrate of ˆEt. Within
the approximated probabilities, we can apply entropy coding
for the latent embeddings to achieve further compression.
B. Neural Network Compression
To explore optimal compression of the neural networks
in HPC, we start with analyzing the characteristics of their
parameters. From the parameter distributions shown in Fig. 4,
we observe that parameters in corresponding layers exhibit
similar distribution between adjacent frames, while showing
considerable differences across layers. This motivates us to
leverage temporal context for redundancy reduction.
However, implementing such a naive strategy may intro-
duce error propagation, where poorly optimized parameters in
one frame can adversely affect subsequent frames, resulting
in cascading performance decline. Drawing inspiration from
traditional video coding, we refer to the concept of Group of
Pictures (GOP) [52] for an optimal reference structure. Specif-
ically, as illustrated in Fig. 3 (b), a GOP typically comprises
an Intra-coded frame (I-frame) and several Predictive-coded
frames (P-frames). The I-frame, positioned at the GOP start,
is independently optimized and compressed, while subsequent
P-frames employ inter-prediction by referencing the preceding
frame. By introducing a GOP structure with periodic insertion
of independent I-frames, we prevent error propagation and
maintain optimization stability over time.
I-frame Neural Network Compression. As illustrated in
Fig. 3 (c), the compression pipeline for I-frames comprises
quantization followed by entropy coding. Since the network
parameters are particularly sensitive to quantization errors, we
introduce a scaling operation prior to quantization to better
preserve their precision:
˜Pt = ⌊
Pt −min(Pt)
max(Pt) −min(Pt) · (2B −1)⌉,
(14)
where ˜Pt denotes the quantized and transmitted parameters,
and B is the bit depth of quantization which controls the
quantization precision. The minimum and maximum values
of Pt are sent to the decoder as side information to obtain the
reconstructed parameters ˆPt:
ˆPt = ˜Pt · [max(Pt) −min(Pt)]/(2B −1) + min(Pt). (15)
During training, we adopt the same proxy strategy used for
latent embeddings: the STE for gradient propagation in the
distortion term, and additive uniform noise for differentiable
entropy estimation.
After quantization, entropy coding is applied to ˜Pt for the
final compression. Following [48], we introduce the network-
free Gaussian distribution for entropy modeling, which is

<!-- page 7 -->
7
0.50
0.25
0.00
0.25
0.50
0
50
100
150
Layer 1
Previous
Current
0.5
0.0
0.5
0
100
200
300
Layer 2
Previous
Current
0.50
0.25
0.00
0.25
0.50
0
25
50
75
100
125
150
Layer 3
Previous
Current
0.5
0.0
0.5
1.0
1.5
0
10
20
30
40
Layer 4
Previous
Current
Fig. 4: Parameter distributions across adjacent frames in different layers.
parameterized solely by the statistical mean µ ˜
Pt and variance
σ2
˜
Pt. Consequently, the approximated probabilities pPMF( ˜Pt)
and bitrate R( ˜Pt) are available as:
pPMF( ˜Pt) =
Y
i
(N(µ ˜
Pt, σ2
˜
Pt) ∗U(−1
2, 1
2))( ˜P i
t ),
(16)
R( ˜Pt) = E ˜
Pt[−log2 pPMF( ˜Pt)],
(17)
where ∗denotes convolution, and P i
t denotes the each element
in Pt. This design enables differentiable entropy estimation
without introducing additional network parameters.
P-frame Neural Network Compression. For P-frames, we aim
to exploit the decoded parameters from the previous frame
ˆPt−1 as a contextual prior, which guides the optimization
and compression of the current parameters. To this end,
we draw upon the classical differential coding to optimize
and encode the residual between consecutive frames, which
exhibits greater energy compaction.
The P-frame neural network compression is illustrated in
Fig. 3 (d). Specifically, inspired by the layer-wise temporal
correlation shown in Fig. 4, we first introduce a learnable, per-
layer scaling factor ηt to adapt ˆPt−1 for a coarse prediction
of the current parameters. To bridge the remaining gap, we
further optimize a per-parameter residual ∆Pt for fine-grained
compensation. Both ηt and ∆Pt are transmitted to the decoder.
Due to its minimal count, the per-layer ηt is preserved at
the original precision without compression. In contrast, the
per-parameter ∆Pt undergoes further compression for bitrate
reduction. Within the received ηt and the decoded residuals
∆ˆPt, the final parameters ˆPt can be reconstructed as follows:
ˆPt = ηt · ˆPt−1 + ∆ˆPt.
(18)
To compress the residuals ∆Pt, we apply a similar quanti-
zation strategy as used for the I-frame parameters. Here, we
normalize the energy-compact residual ∆Pt by introducing the
full parameter dynamic range derived from ˆPt−1, ensuring the
reconstructed parameter ˆPt is represented at a consistent target
precision while achieving bitrate savings:
vmin = min( ˆPt−1, ∆Pt),
vmax = max( ˆPt−1, ∆Pt).
(19)
∆˜Pt = ⌊∆Pt −vmin
vmax −vmin
· (2B −1)⌉,
(20)
∆ˆPt = ∆˜Pt · (vmax −vmin)/(2B −1) + vmin.
(21)
For further entropy calculation, we adopt the same strategy
as the compression of I-frame parameters:
pPMF(∆˜Pt) =
Y
i
(N(µ∆˜
Pt, σ2
∆˜
Pt)∗U(−1
2, 1
2))(∆˜Pt
i), (22)
R(∆˜Pt) = E∆˜
Pt[−log2 pPMF(∆˜Pt)].
(23)
Within the proposed temporal reference strategy, the optimized
residuals exhibit higher energy compactness. This leads to
a lower transmission cost compared to encoding the full
parameters, while still preserving the representational capacity
of the neural network.
C. Rate-distortion Optimization
Finally, we jointly optimize the entire framework by min-
imizing a combined loss of the total estimated bitrate and
the rendering distortion, pursuing the optimal rate-distortion
performance in an end-to-end manner. Given a timestep t, we
calculate the weighted sum of distortion loss L(t)
D and the rate
loss L(t)
R as our overall supervision objective L(t):
L(t) = L(t)
R + λL(t)
D ,
(24)
where λ is the Lagrange multiplier which controls the rate-
distortion balance. Specifically, we adopt the 3DGS rendering
loss [1] as the distortion term:
L(t)
D = L(t)
1
+ λSSIML(t)
SSIM.
(25)
For the rate term, we combined the estimated bitrate of both
latent embeddings and neural network parameters:
L(t)
R =
(
R( ˆEt) + R( ˜Pt),
t mod TGOP = 0,
R( ˆEt) + R(∆˜Pt),
t mod TGOP ̸= 0,
(26)
where TGOP is the predefined size of a GOP.
VI. EXPERIMENTS
A. Experiment Setup
Datasets. We evaluate our method on two widely adopted FVV
datasets: (1) the N3DV dataset [54], captured by a 21-camera
rig, which comprises dynamic scenes at 2704×2028 and 30
FPS. (2) the Meet Room dataset [55], captured by a 13-
camera system, comprising dynamic scenes at 1280×720 and
30 FPS. For each dataset, one camera view is held out for
testing, and the rest are used for training.

<!-- page 8 -->
8
0
200
400
600
Storage (KB/frame)
31.4
31.6
31.8
32.0
32.2
32.4
32.6
PSNR(dB)
N3DV
0
50
100
Storage (KB/frame)
31.8
32.0
32.2
32.4
32.6
N3DV (Zoom in)
HPC (Ours)
HPC w/o NNC (Ours)
iFVC
ComGS
4DGC
ReCon-GS
QUEEN
Ex4DGS
4DGV
0
100
200
300
400
Storage (KB/frame)
28
29
30
31
32
PSNR(dB)
Meet Room
0
20
40
60
80
100
Storage (KB/frame)
31.4
31.6
31.8
32.0
32.2
32.4
32.6
Meet Room (Zoom in)
HPC (Ours)
HPC w/o NNC (Ours)
iFVC
ComGS
4DGC
ReCon-GS
STG
Ex4DGS
4DGV
Fig. 5: Rate-distortion curves on the N3DV and the Meet Room datasets. Methods marked with ⃝and △respectively denote
the online and offline optimized methods. Points and curves closer to the top-left corner indicate better performance.
PSNR: 33.14dB
Storage: 690KB
4DGC
PSNR: 33.09dB
Storage: 89KB
iFVC
PSNR: 33.56dB
Storage: 22KB
HiPoC (Ours)
Ground-truth
PSNR: 32.11dB
Storage: 260KB
PSNR: 32.16dB
Storage: 72KB
PSNR: 32.29dB
Storage: 24KB
Fig. 6: Qualitative comparisons on the flame steak and vrheadset. For a fair comparison, we retrain all methods with the same
initial sparse points, while iFVC [7] and HPC share the same initial-frame model.
Implementation Details. For the latent point hierarchy, we
adopt the approach of ContextGS [30] to construct L = 3
scales, with the ratio of points between adjacent scales set
to 0.2. The latent embeddings contain 16 channels per scale.
In the ILA module, we set k = 4 for the k-NN search. For
neural network compression, parameters are quantized to B =
8 bit depth. The GOP size TGOP is set to 5. For training, λ
is adjusted from 0.048 to 0.002 to achieve variable bitrate.
We train HAC [29] as the initial frame for 15000 iterations
following iFVC [7]. For the subsequent frames, we train our
model for 500 iterations. We employ the torchac library [56]
for arithmetic coding. All experiments are conducted on an
NVIDIA GeForce RTX 3090 GPU.
Baselines. We evaluate the proposed HPC by compar-
ing it with state-of-the-art online frameworks, including
3DGStream [2], HiCoM [3], iFVC [7], 4DGC [6], QUEEN [9],
ComGS [5], 4DGCPro [8], and ReCon-GS [4]. Several offline
methods [34], [37], [42], [43], [53], which are optimized
within multiple frames, are also included in the benchmark
to serve as reference points for a more comprehensive perfor-
mance assessment. As existing methods lack neural network
compression, we introduce a corresponding baseline, HPC w/o
NNC, by disabling this component in our full model. This
allows it to be fairly compared against prior methods while
isolating the contribution of our other components.
Metrics. We evaluate rate-distortion performance using Peak
Signal-to-Noise Ratio (PSNR) and Structural Similarity Index
Measure (SSIM) [57] as objective distortion metrics, with
bitrate measured in kilobytes per frame (KB/frame). For a
precise comparison of rate-distortion efficiency, we compute

<!-- page 9 -->
9
TABLE I: Quantitative comparison. The online methods are
organized, in order, into three categories: grid-based meth-
ods, existing point-based methods, and our methods, with
horizontal lines separating each category. For methods with
multiple rate-distortion points, we report both the lowest (-
l) and highest (-h) quality results. The best and second-best
results are highlighted in Red and Orange cells, respectively.
N3DV dataset
Category
Method
PSNR
(dB)↑
SSIM↑
Storage
(KB/frame)↓
STG [34]
32.05
-
666
GIFStream [42]
31.75
0.938
33
Ex4DGS [53]
32.11
0.94
383
MEGA [37]
31.49
-
83
Offlne
4DGV [43]
32.55
-
70
3DGStream [2]
31.69
0.948
7780
4DGC [6]
31.58
0.943
500
4DGCPro [8]-l
30.68
0.926
210
4DGCPro [8]-h
31.64
0.944
640
iFVC [7]-l
31.84
0.950
63
iFVC [7]-h
32.35
0.953
99
HiCoM [3]
31.17
-
900
QUEEN [9]-l
31.89
0.946
680
QUEEN [9]-h
32.19
0.946
750
ComGS [5]-l
31.87
-
49
ComGS [5]-h
32.12
-
106
ReCon-GS [4]
32.66
0.957
440
HPC w/o NNC-l (Ours)
32.22
0.953
59
HPC w/o NNC-h (Ours)
32.36
0.955
66
HPC-l (Ours)
31.91
0.951
23
Online
HPC-h (Ours)
32.36
0.955
39
Meet Room dataset
Category
Method
PSNR
(dB)↑
SSIM↑
Storage
(KB/frame)↓
STG [34]
29.51
0.932
238
Ex4DGS [53]
31.03
0.946
250
Offline
4DGV [43]
32.31
0.957
64
3DGStream [2]
30.79
-
4100
4DGC [6]
28.08
0.922
420
iFVC [7]-l
32.05
0.956
63
iFVC [7]-h
32.39
0.959
80
HiCoM [3]
26.73
-
600
ComGS [5]
31.49
-
28
ReCon-GS [4]
30.84
0.954
300
HPC w/o NNC-l (Ours)
32.26
0.959
52
HPC w/o NNC-h (Ours)
32.62
0.962
61
HPC-l (Ours)
31.71
0.957
18
Online
HPC-h (Ours)
32.57
0.961
28
the Bjontegaard Delta Bit-Rate (BD-BR) [58], which quanti-
fies the bitrate savings at equivalent PSNR quality. Note that a
negative BD-BR value indicates decreased storage at the same
fidelity compared to the anchor method, which is desirable.
B. Comparison Results
Quantitative Comparisons. The qualitative comparison is
elaborated in Table I. HPC achieves competitive reconstruction
quality while requiring about 20 KB/frame at minimum for
storage on both datasets, which is significantly lower than
other online-optimized methods. By adjusting λ during train-
ing, HPC is able to achieve the best fidelity on the Meet
TABLE II: BD-BR (%) results against iFVC [7]. The results
are tested on the N3DV and Meet Room datasets.
Method
N3DV
Meet Room
Average
iFVC [7]
0.0
0.0
0.0
HPC
-64.9
-70.7
-67.3
HPC w/o NNC
-27.1
-28.9
-28.0
HPC w/o ILA
-60.0
-66.7
-63.4
HPC w/o CLA
-45.9
-51.0
-48.5
Room dataset and the second-best on the N3DV dataset, while
consuming less than 40 KB/frame for storage. Notably, the
reconstruction fidelity is significantly influenced by the initial
frame, which will be discussed in Sec. VI-D. Comprehen-
sively, Fig. 5 shows the overall rate-distortion performance,
where we can observe our HPC significantly surpasses pre-
vious state-of-the-art methods. For a fair comparison, we use
iFVC [7], which shares the same initial frame with ours, as the
anchor to calculate the BD-BR metric. As shown in Table. II,
HPC achieves bitrate savings of 67.3% across the two datasets.
Yet, even with neural network compression disabled, it delivers
a 28% reduction in bitrate, which underscores the substantial
contribution of its representation design.
We give a comprehensive analysis of the significant bitrate
reduction. In contrast to methods like HiCoM [3], ComGS [5],
and ReCon-GS [4], which directly optimize Gaussian attribute
residuals, HPC adopts a different strategy: it optimizes a latent
representation paired with a lightweight network to predict
residuals. This implicit formulation allows entropy constraints
to be applied properly, leading to high compactness and com-
pression efficiency. Compared to approaches with the implicit
dynamic model like iFVC [7], 4DGC [6], 4DGCPro [8], and
QUEEN [9], HPC develops a more compact and efficient
latent representation. When combined with the neural network
compression, this leads to greater overall performance gains.
Qualitative Comparisons. We conduct a qualitative compari-
son with 4DGC [6] and iFVC [7] on the flame steak sequence
and the vrheadset sequence. To ensure a fair comparison,
all methods are initialized with the same sparse point cloud.
Additionally, iFVC [7] and our HPC share an identical first-
frame model. As shown in Fig. 6, HPC achieves the best
reconstruction quality while attaining the smallest model size.
Notably, HPC better preserves fine details, such as facial fea-
tures, animal fur, and flame textures. Furthermore, it demon-
strates superior capability in modeling components with large
motions, including fast-moving arms and hands. These results
indicate that HPC accurately captures dynamic scene elements,
maintains high-fidelity details in complex objects, and achieves
a highly compact representation.
C. Ablation Study
To assess the impact of individual components, we conduct
comprehensive ablation studies. Specifically, our analysis is
structured around the three main contributions: the latent
representation, the latent aggregation scheme, and the neural
network compression scheme.

<!-- page 10 -->
10
Scale 0
10
20
30
40
50
Scale 1
5
10
15
20
25
30
Scale 2
5
10
15
20
25
Ground-truth
Fig. 7: Bit allocation of latent embeddings in different scales. This visualization maps the bitrate consumed by each latent
embedding to a color value, which is then rendered onto a 2D image to show the bit allocation.
TABLE III: Ablation study of different latent representations
on the N3DV dataset. Tri-plane is the anchor for BD-BR.
Tri-plane
Hash-grid
Hierarchical points (Ours)
BD-BR (%)↓
0.0
-6.7
-21.8
TABLE IV: Study on different bit depths of neural network
quantization. The results are based on the Meet Room dataset
with the highest bitrate.
Bit depth
PSNR
(dB)↑
Latent
Storage (KB)↓
Network
Storage (KB)↓
Total
Storage (KB)↓
32
32.62
16
42
58
16
32.60
16
21
37
8
32.57
16
10
26
4
7.31
60
5
65
Different Types of Representation. We first compare the pro-
posed hierarchical latent point with two established structural
representation: the hash-grid and the tri-plane. The hash-grid
adopts the same configuration as in 3DGStream [2], while
the tri-plane adopts the same configuration as in iFVC [7].
As summarized in Table III, the proposed hierarchical points
representation achieves the best performance among the three
candidates. This result demonstrates that our design achieves
more efficient parameter allocation for discrete Gaussians
compared to structural representations.
Latent Aggregation. We evaluate the proposed aggregation
scheme by separately ablating the ILA and CLA modules
as described in Table II. Equipped with the ILA module,
our model yields an extra bitrate saving of 4%. This result
confirms that the component effectively facilitates local infor-
mation sharing, thereby reducing spatial redundancy. When
evaluating the CLA module, it achieves an additional 20%
bitrate saving, underscoring the effectiveness of the aggregated
information from multi-resolution receptive fields. To further
TABLE V: Ablation study of neural network compression on
the Meet Room dataset. M0 is the anchor method for BD-BR
calculation.
Method
M0
M1
M2
Quantization
✓
✓
✓
Entropy constraint
✓
✓
Temporal reference
✓
BD-BR (%)↓
0.0
-6.6
-14.6
demonstrate this, we visualize the bit allocation results in
different scales. As observed in Fig. 7, the allocation exhibit
significant variations across different scales. This indicates
that the multi-scale representation successfully enables HPC
to capture distinct features at different levels of granularity.
The latent embeddings at the coarse scale are oriented toward
capturing motion in large regions (e.g., the moving arm), while
those at finer scales excel at modeling detailed variations. The
aggregation of information across scales empowers HPC to
model scenes more efficiently and comprehensively.
Neural Network Compression. Prior to compression, the
parameters must first be quantized. To investigate its effect, we
reduce the precision from the original 32 bits to several lower
bit depths, including 16, 8, and 4 bits. As shown in Table IV,
the storage cost of network parameters decreases linearly
with the quantization bit depth. Compared to the original
precision, reconstructions at 8 or 16 bits exhibit only marginal
degradation (less than 1 dB). However, at 4-bit quantization,
the training becomes unstable and diverges, leading to a
significant drop in reconstruction quality. Concurrently, the
excessive quantization of the factorized entropy model causes
the increasing bitrate of the latent representation. Based on this
analysis, we select 8-bit quantization as our operating point,
which effectively minimizes storage overhead while preserving
the essential neural network capability.
For further ablation study, we progressively incorporate
other components into the baseline with 8-bit quantization,
which is denoted as M0. As shown in Table. V, to compress
the parameters efficiently, we incorporate an entropy constraint
into the parameter optimization, denoted as M1. This is imple-
mented using the differentiable network-free Gaussian entropy
model [48] to enable bitrate estimation and entropy coding,
which yields a 6.6% performance gain.
Finally, we leverage temporal references across frames
to exploit inter-frame correlations for redundancy reduction,
denoted as M2. As visualized in Fig. 8, our temporal reference
strategy enables the model to capture and transmit the energy-
compact residual. This strategy achieves significant bitrate
savings without compromising the network’s representational
capacity. However, as shown in Fig. 9, a naive integration of
the temporal reference model leads to severe error propagation,
characterized by a continuous degradation in reconstruction
quality with longer reference intervals. We therefore introduce
the GOP reference structure and investigate the impact of its
size. As observed in Fig. 9, compared to no inter-frame refer-

<!-- page 11 -->
11
0
10
20
30
40
0
10
20
30
40
Visualization of Reference
0.5
0.0
0.5
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0
50
100
150
200
250
300
350
400
Distribution of Reference
0
10
20
30
40
0
10
20
30
40
Visualization of Residual
0.5
0.0
0.5
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0
200
400
600
800
1000
1200
1400
1600
Distribution of Residual
0
10
20
30
40
0
10
20
30
40
Visualization of Reconstruction
0.5
0.0
0.5
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0
50
100
150
200
250
300
350
400
Distribution of Reconstruction
Fig. 8: Analysis of neural network parameters and residuals.
The reconstructed parameters are calculated with the reference
parameters and residuals following Eq. (18).
TABLE VI: Complexity comparison. The results are evaluated
on an NVIDIA RTX 3090 GPU, with values averaged on both
the N3DV and Meet Room datasets.
Method
Train
(min)↓
Render
(FPS)↑
Encode
(s)↓
Decode
(s)↓
4DGC [6]
0.83
184
0.70
0.11
iFVC [7]
0.17
140
0.10
0.09
HPC w/o NNC (Ours)
0.60
141
0.13
0.10
HPC (Ours)
1.01
140
0.33
0.18
encing (GOP size=1), larger GOP sizes consistently reduce the
overall storage. In terms of reconstruction quality, GOP sizes
of 10 and 20 still exhibit periodic degradation, compromising
the overall performance. Empirically, a GOP size of 5 is
selected as it not only reduces the bitrate but also yields
a slight improvement in reconstruction quality. This quality
gain stems from a beneficial optimization bias provided by
the well-chosen temporal reference, which steers the network
toward a better optimization direction and thereby strengthens
its expressive power. With the GOP reference structure, our
temporal reference strategy contributes a performance gain of
14.6% over the base model as shown in Table. V.
D. Analysis
Complexity. Table VI compares the complexity of our method
against 4DGC [6] and iFVC [7] in terms of training time,
rendering speed, encoding time, and decoding time. For train-
ing time, the extended duration of HPC is attributed to its
0
50
100
150
200
250
300
Frame index
20
22
24
26
28
30
32
PSNR (dB)
Per-frame PSNR on discussion sequence
w/o GOP, mean PSNR=22.42
GOP 20, mean PSNR=31.36
GOP 10, mean PSNR=32.11
GOP 5, mean PSNR=32.32
GOP 1, mean PSNR=32.28
0
50
100
150
200
250
300
Frame index
0
20
40
60
80
Storage (KB)
Per-frame Storage on discussion sequence
w/o GOP, mean storage=5.11
GOP 20, mean storage=18.91
GOP 10, mean storage=22.20
GOP 5, mean storage=28.16
GOP 1, mean storage=29.89
Fig. 9: Per-frame PSNR and storage on the discussion se-
quence across different GOP size.
rate-aware training scheme for network parameters, which
introduces more complex gradient updates and thus slows the
optimization. When the neural network compression module is
removed, the training speed of HPC surpasses 4DGC [6] but
remains slower than iFVC [7]. This is primarily due to the
additional parameters in the introduced aggregation modules
and entropy model, which in turn require more iterations
for convergence. The rendering is slightly slower for both
HPC and iFVC [7] due to the overhead of Neural Gaussian
Splatting [28], whereas 4DGC [6] uses the faster, non-neural
original splatting [1]. The longer overall coding time of HPC
stems from its need to entropy encode both latent embeddings
and network parameters, whereas the other two methods
compress only the former. When focusing specifically on the
shared task of encoding latent embeddings, HPC matches the
coding speed of iFVC [7] and outperforms 4DGC [6].
Initial Frame Analysis. In streaming FVV, the initial frame
critically influences all subsequent reconstruction. Given its
pivotal role, we comprehensively evaluate our work by ana-
lyzing the operation and results for the initial frame across
different methods as shown in Table VII.
There exists various of initial representations and training
strategies across different methods. 3DGStream [2] adopts the
vanilla 3DGS representation and training strategy for the initial
frame. ReCon-GS [4] regularizes the training of 3DGS by
injecting Gaussian noise into the position attributes to mitigate
overfitting. In contrast to the vanilla 3DGS, both iFVC and
our HPC adopt the HAC framework [29] to obtain a highly
compressed Neural Gaussian [28] representation for the initial

<!-- page 12 -->
12
TABLE VII: Analysis of the initial frame across different methods on N3DV. The average results across all frames are based
on the highest bitrate.
Method
Representation
Init. training strategy
Init. Storage
(KB)↓
Avg. Storage
(KB)↓
Init. PSNR
(dB)↑
Avg. PSNR
(dB)↑
PSNR gain
over Init. (%)↑
3DGStream [2]
Vanilla 3DGS
Vanilla 3DGS training
51660
7780
32.13
31.69
-1.36
ReCon-GS [4]
Vanilla 3DGS
Noise-injected 3DGS training
11120
440
32.75
32.66
-0.27
iFVC [7]
Neural Gaussian
HAC training
2480
99
32.21
32.35
+0.43
HPC (Ours)
Neural Gaussian
HAC training
2480
39
32.21
32.36
+0.46
0
5
10
15
20
25
30
Storage (KB)
0.002
0.016
0.032
0.048
72.6%
27.4%
60.9%
39.1%
60.2%
39.8%
59.5%
40.5%
HiPoC (N3DV)
0
5
10
15
20
25
Storage (KB)
65.8%
34.2%
58.7%
41.3%
56.4%
43.6%
55.0%
45.0%
HiPoC (Meet Room)
0
10
20
30
40
50
60
Storage (KB)
0.002
0.016
0.032
0.048
28.2%
71.8%
21.1%
78.9%
19.1%
80.9%
18.4%
81.6%
HiPoC w/o NNC (N3DV)
0
10
20
30
40
50
60
Storage (KB)
28.2%
71.8%
18.9%
81.1%
15.9%
84.1%
14.9%
85.1%
HiPoC w/o NNC (Meet Room)
Latent embedding
Neural network
Fig. 10: Bit allocation between latent embedding and neural
network.
frame. This approach achieves substantial storage savings
(20× and 4.5× reduction against 3DGStream [2] and ReCon-
GS [4], respectively). In terms of reconstruction fidelity, the
noise-injection strategy effectively prevents ReCon-GS [4]
from overfitting to the training views, thus achieving the
highest PSNR for both the initial and subsequent frames.
However, we observe that using vanilla 3DGS [1] as the
representation leads to a quality drop, whereas adopting Neural
Gaussians [28] yields a consistent gain over the initial re-
construction quality. Among Neural Gaussian-based methods,
our HPC achieves the highest PSNR gain of 0.46%, further
proving its effectiveness.
Bit Allocation. To provide a comprehensive analysis of our
method, we analyze the bit allocation results between the latent
embedding and the neural network across different bitrates. As
observed in Fig. 10, the rate allocation varies across different
bitrate points. At lower bitrates, their allocated rates are com-
parable. As the total bitrate budget increases, the proportion
allocated to the latent embeddings grows consistently and
reaches approximately 70% at the highest bitrate.
When the neural network compression module is disabled,
we observe that the bitrate share of the network parameters
surges to over 80% across low and medium bitrate points.
This high allocation not only impedes further bitrate reduction
but also potentially constrains the model’s overall expressive
capacity, demonstrating the necessity of compressing network
parameters.
VII. CONCLUSION AND FUTURE WORKS
We explore a hierarchical point-based latent representa-
tion tailored for dynamic Gaussian Splatting and introduce
HPC. It ensures efficient parameter allocation, enables the
effective capture of spatial correlations from non-uniformly
distributed Gaussians, and ultimately delivers efficient, high-
quality reconstruction. A fully end-to-end compression frame-
work, which jointly optimizes both the latent representation
and the neural network for an optimal rate-distortion trade-
off, is designed to achieve significant bitrate savings. Extensive
experiments demonstrate that HPC achieves the state-of-the-
art compression performance against existing methods. Com-
prehensive studies analyze and demonstrate the effectiveness
of its technical components. By achieving significant bitrate
savings for dynamic Gaussian Splatting, our work paves the
way for the field of streaming free-viewpoint video.
The primary limitation of HPC is its longer training duration
and higher coding latency compared to competing methods
due to the additional rate-aware optimization and compres-
sion for network parameters. Future work should incorporate
lightweight designs to accelerate both the training and coding
processes, thereby facilitating practical deployment.
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[2] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, “3dgstream: On-
the-fly training of 3d gaussians for efficient streaming of photo-realistic
free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), June 2024, pp.
20 675–20 685.
[3] Q. Gao, J. Meng, C. Wen, J. Chen, and J. Zhang, “Hicom: Hierarchical
coherent motion for dynamic streamable scenes with 3d gaussian splat-
ting,” in Advances in Neural Information Processing Systems (NeurIPS),
2024.
[4] J. Fu, Q. Gao, C. Wen, Y. Wu, S. Ma, J. Zhang, and J. Zhang, “Recon-
GS: Continuum-preserved guassian streaming for fast and compact re-
construction of dynamic scenes,” in The Thirty-ninth Annual Conference
on Neural Information Processing Systems, 2025.
[5] J. Chen, Q. Mao, Y. Bao, X. MENG, F. Meng, R. Wang, and Y. Liang,
“Motion matters: Compact gaussian streaming for free-viewpoint video
reconstruction,” in The Thirty-ninth Annual Conference on Neural In-
formation Processing Systems, 2025.
[6] Q. Hu, Z. Zheng, H. Zhong, S. Fu, L. Song, X. Zhang, G. Zhai,
and Y. Wang, “4dgc: Rate-aware 4d gaussian compression for efficient
streamable free-viewpoint video,” in Proceedings of the Computer Vision
and Pattern Recognition Conference (CVPR), June 2025, pp. 875–885.
[7] L. Tang, J. Yang, R. Peng, Y. Zhai, S. Shen, and R. Wang, “Compressing
streamable free-viewpoint videos to 0.1 mb per frame,” in Proceedings
of the AAAI Conference on Artificial Intelligence, vol. 39, no. 7, 2025,
pp. 7257–7265.

<!-- page 13 -->
13
[8] Z. Zheng, Z. Wu, H. Zhong, Y. Tian, N. Cao, L. Xu, J. Yao, X. Zhang,
Q. Hu, and W. Zhang, “4DGCPro: Efficient hierarchical 4d gaussian
compression for progressive volumetric video streaming,” in The Thirty-
ninth Annual Conference on Neural Information Processing Systems,
2025.
[9] S. Girish, T. Li, A. Mazumdar, A. Shrivastava, D. Luebke, and S. D.
Mello, “QUEEN: QUantized efficient ENcoding for streaming free-
viewpoint videos,” in The Thirty-eighth Annual Conference on Neural
Information Processing Systems, 2024.
[10] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM transactions on
graphics (TOG), vol. 41, no. 4, pp. 1–15, 2022.
[11] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa,
“K-planes: Explicit radiance fields in space, time, and appearance,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 12 479–12 488.
[12] H. Yu, W. Gong, J. Chen, and H. Ma, “Get3dgs: Generate 3d gaussians
based on points deformation fields,” IEEE Transactions on Circuits and
Systems for Video Technology, vol. 35, no. 5, pp. 4437–4449, 2025.
[13] Y. Ma, B. Liu, J. Li, L. Li, and D. Liu, “Hash grid feature pruning,”
arXiv preprint arXiv:2512.22882, 2025.
[14] S. Xie, W. Zhang, C. Tang, Y. Bai, R. Lu, S. Ge, and Z. Wang, “Mesongs:
Post-training compression of 3d gaussians via efficient attribute trans-
formation,” in European Conference on Computer Vision.
Springer,
2024, pp. 434–452.
[15] S. Lee, F. Shu, Y. Sanchez, T. Schierl, and C. Hellge, “Compression of
3d gaussian splatting with optimized feature planes and standard video
codecs,” in Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), October 2025, pp. 25 496–25 505.
[16] S. Girish, K. Gupta, and A. Shrivastava, “Eagles: Efficient accelerated
3d gaussians with lightweight encodings,” in European Conference on
Computer Vision, 2024, pp. 54–71.
[17] S. Li, C. Wu, H. Li, X. Gao, Y. Liao, and L. Yu, “Gscodec studio: A
modular framework for gaussian splat compression,” IEEE Transactions
on Circuits and Systems for Video Technology, pp. 1–1, 2026.
[18] Q. Yang, L. Yang, G. V. der Auwera, and Z. Li, “HybridGS: High-
efficiency gaussian splatting data compression using dual-channel sparse
representation and point cloud encoder,” in Forty-second International
Conference on Machine Learning, 2025.
[19] Y. Wang, M. Liu, Q. Yang, H. Huang, L. Yang, and Y. Xu, “On
the efficient adaptive streaming of 3d gaussian splatting over dynamic
networks,” IEEE Transactions on Circuits and Systems for Video Tech-
nology, pp. 1–1, 2025.
[20] C. Wang, S. N. Sridhara, E. Pavez, A. Ortega, and C. Chang, “Adaptive
voxelization for transform coding of 3d gaussian splatting data,” in 2025
IEEE International Conference on Image Processing (ICIP), 2025, pp.
2414–2419.
[21] M. S. Ali, M. Qamar, S.-H. Bae, and E. Tartaglione, “Trimming the fat:
Efficient compression of 3d gaussian splats through pruning,” in 35th
British Machine Vision Conference 2024, BMVC 2024, Glasgow, UK,
November 25-28, 2024.
BMVA, 2024.
[22] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian:
unbounded 3d gaussian compression with 15x reduction and 200+
fps,” in Proceedings of the 38th International Conference on Neural
Information Processing Systems, ser. NIPS ’24.
Red Hook, NY, USA:
Curran Associates Inc., 2024.
[23] H. Wang, H. Zhu, T. He, R. Feng, J. Deng, J. Bian, and Z. Chen, “End-to-
end rate-distortion optimized 3d gaussian representation,” in European
Conference on Computer Vision.
Springer, 2024, pp. 76–92.
[24] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” in 2024 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 719–
21 728.
[25] X. Liu, X. Wu, P. Zhang, S. Wang, Z. Li, and S. Kwong, “Compgs:
Efficient 3d scene representation via compressed gaussian splatting,” in
Proceedings of the 32nd ACM International Conference on Multimedia,
2024.
[26] K. Navaneet, K. P. Meibodi, S. A. Koohpayegani, and H. Pirsiavash,
“Compgs: Smaller and faster gaussian splatting with vector quantiza-
tion,” ECCV, 2024.
[27] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed
3d gaussian splatting for accelerated novel view synthesis,” in 2024
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2024, pp. 10 349–10 358.
[28] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai,
“Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 654–20 664.
[29] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Hac: Hash-grid
assisted context for 3d gaussian splatting compression,” in European
Conference on Computer Vision.
Springer, 2024, pp. 422–438.
[30] Y. Wang, Z. Li, L. Guo, W. Yang, A. Kot, and B. Wen, “Contextgs: Com-
pact 3d gaussian splatting with anchor level context model,” Advances
in neural information processing systems, vol. 37, pp. 51 532–51 551,
2024.
[31] Y.-T. Zhan, C.-Y. Ho, H. Yang, Y.-H. Chen, J. C. Chiang, Y.-L. Liu, and
W.-H. Peng, “Cat-3dgs: A context-adaptive triplane approach to rate-
distortion-optimized 3dgs compression,” in The Thirteenth International
Conference on Learning Representations, 2025.
[32] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Hac++: Towards 100x
compression of 3d gaussian splatting,” IEEE Transactions on Pattern
Analysis and Machine Intelligence, vol. 47, no. 11, pp. 10 210–10 226,
2025.
[33] Y. Duan, F. Wei, Q. Dai, Y. He, W. Chen, and B. Chen, “4d-rotor
gaussian splatting: Towards efficient novel view synthesis for dynamic
scenes,” in ACM SIGGRAPH 2024 Conference Papers, ser. SIGGRAPH
’24.
New York, NY, USA: Association for Computing Machinery,
2024.
[34] Z. Li, Z. Chen, Z. Li, and Y. Xu, “Spacetime gaussian feature splatting
for real-time dynamic view synthesis,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), June
2024, pp. 8508–8520.
[35] Z. Yang, H. Yang, Z. Pan, and L. Zhang, “Real-time photorealistic
dynamic scene representation and rendering with 4d gaussian splatting,”
in The Twelfth International Conference on Learning Representations,
2024.
[36] Y. Wang, P. Yang, Z. Xu, J. Sun, Z. Zhang, Y. Chen, H. Bao, S. Peng,
and X. Zhou, “Freetimegs: Free gaussian primitives at anytime anywhere
for dynamic scene reconstruction,” in CVPR, 2025.
[37] X. Zhang, Z. Liu, Y. Zhang, X. Ge, D. He, T. Xu, Y. Wang, Z. Lin,
S. Yan, and J. Zhang, “Mega: Memory-efficient 4d gaussian splatting
for dynamic scenes,” arXiv preprint arXiv:2410.13613, 2024.
[38] M. Lee, B. Lee, L. Y. Lee, E. Lee, S. Kim, S. Song, J. C. Lee, J. H. Ko,
J. Park, and E. Park, “Optimized minimal 4d gaussian splatting,” 2025.
[39] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and
X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), June 2024, pp. 20 310–20 320.
[40] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable
3d gaussians for high-fidelity monocular dynamic scene reconstruction,”
in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition (CVPR), 2024, pp. 20 331–20 341.
[41] Z. Guo, W. Zhou, L. Li, M. Wang, and H. Li, “Motion-aware 3d gaussian
splatting for efficient dynamic scene reconstruction,” IEEE Transactions
on Circuits and Systems for Video Technology, vol. 35, no. 4, pp. 3119–
3133, 2025.
[42] H. Li, S. Li, X. Gao, A. Batuer, L. Yu, and Y. Liao, “Gifstream: 4d
gaussian-based immersive video with feature stream,” in Proceedings
of the Computer Vision and Pattern Recognition Conference, 2025, pp.
21 761–21 770.
[43] P. Dai, P. Zhang, Z. Dong, K. Xu, Y. Peng, D. Ding, Y. Shen, Y. Yang,
X. Liu, R. W. H. Lau, and W. Xu, “4d gaussian videos with motion
layering,” ACM Trans. Graph., vol. 44, no. 4, 2025.
[44] J. Ball´e, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational
image compression with a scale hyperprior,” in International Conference
on Learning Representations, 2018.
[45] H. Chen, B. He, H. Wang, Y. Ren, S. N. Lim, and A. Shrivastava, “Nerv:
Neural representations for videos,” Advances in Neural Information
Processing Systems, vol. 34, pp. 21 557–21 568, 2021.
[46] C. Gomes, R. Azevedo, and C. Schroers, “Video compression with
entropy-constrained neural representations,” in 2023 IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition (CVPR), 2023, pp.
18 497–18 506.
[47] S. R. Maiya, S. Girish, M. Ehrlich, H. Wang, K. S. Lee, P. Poirson,
P. Wu, C. Wang, and A. Shrivastava, “Nirvana: Neural implicit repre-
sentations of videos with adaptive networks and autoregressive patch-
wise modeling,” in 2023 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2023, pp. 14 378–14 387.
[48] X. Zhang, R. Yang, D. He, X. Ge, T. Xu, Y. Wang, H. Qin, and
J. Zhang, “Boosting neural representations for videos with a conditional
decoder,” in The IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2024.

<!-- page 14 -->
14
[49] H. M. Kwan, G. Gao, F. Zhang, A. Gower, and D. Bull, “Nvrc: Neural
video representation compression,” Advances in Neural Information
Processing Systems, vol. 37, pp. 132 440–132 462, 2024.
[50] J. Wang, X. Zhang, G. Zhang, J. Zhu, L. Tang, and L. Zhang, “Uar-nvc:
A unified autoregressive framework for memory-efficient neural video
compression,” IEEE Transactions on Circuits and Systems for Video
Technology, pp. 1–1, 2025.
[51] H. Thomas, C. R. Qi, J.-E. Deschaud, B. Marcotegui, F. Goulette, and
L. J. Guibas, “Kpconv: Flexible and deformable convolution for point
clouds,” in Proceedings of the IEEE/CVF international conference on
computer vision, 2019, pp. 6411–6420.
[52] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the
high efficiency video coding (hevc) standard,” IEEE Transactions on
Circuits and Systems for Video Technology, vol. 22, no. 12, pp. 1649–
1668, 2012.
[53] J. Lee, C. Won, H. Jung, I. Bae, and H.-G. Jeon, “Fully explicit dynamic
guassian splatting,” in Proceedings of the Neural Information Processing
Systems, 2024.
[54] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim,
T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe, and Z. Lv,
“Neural 3d video synthesis from multi-view video,” in 2022 IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 2022,
pp. 5511–5521.
[55] L. LI, Z. Shen, Z. Wang, L. Shen, and P. Tan, “Streaming radiance fields
for 3d video synthesis,” in Advances in Neural Information Processing
Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and
A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 13 485–13 498.
[56] F. Mentzer, E. Agustsson, M. Tschannen, R. Timofte, and L. Van Gool,
“Practical full resolution learned lossless image compression,” in Pro-
ceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2019.
[57] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality assess-
ment: from error visibility to structural similarity,” IEEE Transactions
on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.
[58] G. Bjontegaard, “Calculation of average psnr differences between rd-
curves,” ITU SG16 Doc. VCEG-M33, 2001.
