<!-- page 1 -->
MAPo : Motion-Aware Partitioning of Deformable 3D Gaussian Splatting for
High-Fidelity Dynamic Scene Reconstruction
Han Jiao, Jiakai Sun
Zhejiang University
{csjh, csjk}@zju.edu.cn
Yexing Xu
The Shenzhen Campus, Sun Yat-Sen University
xuyx55@mail2.sysu.edu.cn
Lei Zhao, Wei Xing, Huaizhong Lin
Zhejiang University
{cszhl, wxing, linhz}@zju.edu.cn
(a) 4DGaussians
+ Finer Grid
(b) E-D3DGS
+ Bigger MLP
(c) Ours
(d) Ground Truth
Figure 1: Overview. (a-b) Existing deformation-based methods often result in blurriness in areas with intense motion, and even
enhancing the capabilities of the deformation fields does not lead to significant improvements. (c) Our method significantly
improves reconstruction quality in those regions. (d) Ground Truth.
Figure 1. Overview. (a-b) Deformation-based methods often blur details in regions with complex or rapid motion. (c) Our MAPo
significantly improves rendering quality in these areas. (d) Ground Truth.
Abstract
3D Gaussian Splatting, known for enabling high-quality
static scene reconstruction with fast rendering, is increas-
ingly being applied to multi-view dynamic scene recon-
struction. A common strategy involves learning a deforma-
tion field to model the temporal changes of a canonical set
of 3D Gaussians. However, these deformation-based meth-
ods often produce blurred renderings and lose fine motion
details in highly dynamic regions due to the inherent lim-
itations of a single, unified model in representing diverse
motion patterns.
To address these challenges, we intro-
duce Motion-Aware Partitioning of Deformable 3D Gaus-
sian Splatting (MAPo), a novel framework for high-fidelity
dynamic scene reconstruction. Its core is a dynamic score-
based partitioning strategy that distinguishes between high-
and low-dynamic 3D Gaussians.
For high-dynamic 3D
Gaussians, we recursively partition them temporally and
duplicate their deformation networks for each new tempo-
ral segment, enabling specialized modeling to capture in-
tricate motion details. Concurrently, low-dynamic 3DGs
are treated as static to reduce computational costs. How-
ever, this temporal partitioning strategy for high-dynamic
3DGs can introduce visual discontinuities across frames at
the partition boundaries. To address this, we introduce a
cross-frame consistency loss, which not only ensures visual
continuity but also further enhances rendering quality. Ex-
tensive experiments demonstrate that MAPo achieves supe-
rior rendering quality compared to baselines while main-
taining comparable computational costs, particularly in re-
gions with complex or rapid motions.
1. Introduction
Reconstructing high-fidelity dynamic scenes from multi-
view video inputs is a fundamental challenge in computer
vision, with broad applications in virtual reality, visual ef-
fects, and autonomous driving. In recent years, Neural Ra-
diance Fields (NeRF) [19] have demonstrated remarkable
capabilities in representing static scenes for novel view syn-
thesis by leveraging implicit neural representations. Build-
ing upon this, numerous efforts [3, 6, 7, 14, 15, 20–23, 30]
have extended NeRF to dynamic scenes by introducing tem-
poral conditioning. However, the inherent reliance on dense
spatial sampling and costly Multilayer Perceptron (MLP)
querying leads to significant limitations in both training ef-
ficiency and rendering speed.
Recently, 3D Gaussian Splatting (3DGS) [12] has
emerged as a powerful alternative for static scene recon-
arXiv:2508.19786v2  [cs.CV]  25 Nov 2025

<!-- page 2 -->
(a) Average
(b) Rendering 1
(c) Rendering 2
Figure 2. Rendering results of a single unified model. (a) shows
the temporally averaged representation, which is visualized by di-
rectly rendering the canonical 3DGs. The regions highlighted in
blue in (b) and (c) are visually close to this average. The region
highlighted in red in (c) is visually distant from this average.
struction, achieving real-time rendering while maintain-
ing photorealistic rendering quality. This method explic-
itly represents scenes using anisotropic 3D Gaussian (3DG)
primitives. Encouraged by its success in static scenes, re-
searchers have begun adapting Gaussian Splatting to dy-
namic scenes [2, 11, 13, 17, 27–29, 32, 33]. Employing de-
formable 3D Gaussians—where a learned deformation field
maps a canonical set of 3D Gaussians (3DGs) to their time-
varying states—has quickly emerged as a prevailing and ef-
fective strategy for reconstructing dynamic scenes, offering
compelling visual quality from a compact representation.
Despite promising results, these methods suffer from two
critical limitations inherent in their deformation framework:
• Bottleneck in Motion Modeling Capacity: As shown in
Fig. 1, deformation-based methods often produce blurry
reconstructions and lose fine motion details in regions
with complex or rapid motions.
This core limitation
stems from their unified modeling strategy, which relies
on a single canonical set of 3DGs and a single, glob-
ally shared deformation network to represent all spatio-
temporal variations.
Such an approach forces the net-
work to find a single set of parameters that best fits all di-
verse and often conflicting motion patterns across the en-
tire time sequence. Consequently, the learned representa-
tion often converges to a temporally averaged representa-
tion. As shown in Fig. 2, this averaging effect prevents the
model from accurately capturing complex motions that
deviate from this learned average, such as abrupt changes
and fine-grained temporal details. This ultimately leads
to the loss of fine details in dynamic regions.
• Redundant Computation: The 3DGs in static regions
still participate in repeated deformation network compu-
tation, wasting computational resources and potentially
slowing down training and rendering speed.
To tackle these issues, we introduce MAPo, a novel frame-
work for high-fidelity dynamic scene reconstruction.
Its
core is a dynamic score-based partitioning strategy compris-
ing two key components: Temporal Partitioning Based on
Dynamic Scores and Static 3D Gaussian Partitioning. The
former enhances details in dynamic regions through hier-
archical temporal partitioning of high-dynamic 3DGs, and
the latter resolves redundant computation by treating low-
dynamic 3DGs as static 3DGs. To ensure temporal smooth-
ness after partitioning at the partition boundaries, a cross-
frame consistency loss is also employed.
Specifically, MAPo first analyzes each 3DG’s historical
positions to compute a dynamic score, which provides a
practical measure of motion intensity for each 3DG. Dur-
ing temporal partitioning, high-dynamic 3DGs, identified
by their high dynamic scores, are recursively partitioned
along the temporal dimension. The 3DGs that undergo par-
titioning are replicated, creating new 3DG instances for the
resulting temporal sub-segments. Simultaneously, the de-
formation network of the parent segment is replicated, cre-
ating a dedicated sub-network for each new temporal sub-
segment to deform all 3DGs within it. Instead of relying
on a single, unified model, our strategy adaptively enables
multiple sets of networks and their corresponding 3DGs to
model the dynamic scene over distinct temporal segments.
This allows for a more faithful capture of details in highly
dynamic regions and effectively alleviates the “temporal av-
eraging” effect. Meanwhile, 3DGs with low dynamic scores
are identified as static. Their attributes are directly updated
to reflect the deformation results, after which redundant
deformation network computations are skipped to reduce
computational costs.
To further mitigate visual discontinuities introduced by
temporal partitioning, we introduce a cross-frame consis-
tency loss that enforces two constraints: (i) the renderings
of two sets of temporally adjacent 3DGs should remain as
consistent as possible when rendered at the same time and
from the same viewpoint, and (ii) the rendering of the tem-
porally adjacent 3DGs corresponding to the current time
should also align with the ground-truth observation at the
current time. This consistency loss not only reduces visual
discontinuities at partition boundaries but also leverages
temporal context for more faithful reconstructions. Our key
contributions are summarized as follows:
• We propose MAPo, a novel framework for high-fidelity
dynamic scene reconstruction based on a dynamic score-
based partitioning strategy.
Our strategy enhances the
modeling effects of highly dynamic regions by enabling
multiple sets of networks and their corresponding 3DGs
to model the dynamic scene over distinct temporal seg-
ments, and simultaneously reduces computational costs
by identifying low dynamic scores as static.
• We design a cross-frame consistency loss that effectively
mitigates visual discontinuities introduced by temporal
partitioning and further enhances the rendering quality.
• Extensive experiments demonstrate that MAPo achieves
state-of-the-art (SOTA) rendering quality, particularly in
capturing fine details in highly dynamic scenes.

<!-- page 3 -->
2. Related Work
2.1. Dynamic NeRF
Neural Radiance Fields (NeRF) [19] marked a significant
breakthrough in representing static scenes for high-quality
novel view synthesis using implicit MLPs. This success
has spurred numerous efforts to extend its capabilities to
dynamic scenes. One prominent strategy is deformation-
based, where a learned deformation field maps observed
points to a canonical representation.
D-NeRF [22] pio-
neered this concept by directly using spatio-temporal co-
ordinates as the input to the deformation network.
This
was later advanced by methods like Nerfies [20], replac-
ing the explicit time input with a learnable latent code to
govern deformation, and HyperNeRF [21], utilizing higher-
dimensional embeddings to better model complex topolo-
gies. NeRFPlayer [25] introduces a streaming-capable rep-
resentation that separates static and dynamic scene com-
ponents. Other approaches bypass the deformation-based
paradigm.
For instance, DyNeRF [15] directly queries
the appearance attributes of spatio-temporal points using
a 6D spatio-temporal function. To improve efficiency, K-
Planes [7] and HexPlane [3] introduce explicit multi-planar
representations. HyperReel [1] further enhances rendering
speed by employing a more compact dual-plane represen-
tation. Despite these advances, the reliance on dense ray
sampling and per-point MLP queries in NeRF-based meth-
ods remains a barrier to achieving real-time rendering.
2.2. Dynamic Gaussian Splatting
The recent success of 3DGS in achieving real-time, photo-
realistic rendering for static scenes has naturally inspired
its extension to dynamic scenes.
Among these exten-
sions, a significant number of approaches are deformation-
based, where a canonical set of 3DGs is deformed over
time. D3DGS [32] introduces a canonical 3DGs represen-
tation and a deformation network that takes position and
time as input to map canonical 3DGs to the observation
space. Many subsequent methods share similarities with
this paradigm. 4DGaussians [27] decodes features from a
HexPlane representation using spatio-temporal coordinates
to derive per-Gaussian deformations.
E-D3DGS [2] ad-
vances this by replacing direct coordinate inputs with tem-
poral and per-Gaussian embeddings and adopting a dual-
deformation strategy. Similarly, DN-4DGS [17] integrates
spatio-temporal information from neighboring elements
and also employs a dual-deformation approach. To han-
dle long sequences, SWinGS [24] partitions the sequence
into sliding windows and trains an independent model for
each, using 2D optical flow to guide window segmentation
and a post-hoc fine-tuning step to ensure smoothness. To
improve efficiency, Swift4D [29] uses 2D RGB informa-
tion to separate dynamic from static regions, modeling only
the dynamic 3DGs’ deformation with a 4D hash grid and
a small MLP decoder. Inspired by Scaffold-GS [18], Lo-
calDyGS [28] decomposes the scene into seed-based local
spaces and generates time-varying 3DGs by fusing static
and dynamic features, though its overall quality is still lim-
ited. While diverse, these deformation-based methods face
a fundamental trade-off. Approaches relying on a single,
globally shared representation like D3DGS and its direct
successors often suffer from the “temporal averaging” ef-
fect in complex scenes. Conversely, strategies like SWinGS
that partition the sequence into independent, localized mod-
els mitigate this issue, but introduce their own challenges,
including coarse, window-level partitioning; cumbersome
pre- and post-processing steps; and a strong reliance on 2D
priors like optical flow.
Other distinct strategies have also been explored. Curve-
based methods [9, 13, 16] model 3DG attribute changes
over time using parametric curves, but may induce drift
under complex motions and introduce significant storage
overhead. 4D Gaussian-based methods [5, 31, 33] decom-
pose 4D Gaussians into 3DGs and marginal 1D Gaussians,
offering a direct spatio-temporal representation, but incur
significant computational costs. Per-frame training meth-
ods [8, 10, 26] enable online reconstruction but are con-
strained by the inherent trade-off between quality, storage,
and efficiency.
To overcome the limitations of existing methods, we in-
troduce a dynamic score-based partitioning strategy. We re-
cursively partition high-dynamic 3DGs into finer temporal
segments to address the issue of the “temporal average”.
Unlike window-based methods like SWinGS, our partition-
ing operates at a fine-grained, per-3DG level, is guided di-
rectly by 3D motion instead of 2D priors, and is integrated
into a single, end-to-end training framework, thus avoiding
cumbersome pre- and post-processing steps. Furthermore,
we build our method upon a compact deformation field ar-
chitecture and identify low-dynamic 3DGs as static to avoid
the substantial overhead of other categories of methods.
3. Preliminaries
3D Gaussian Splatting
3D Gaussian Splatting introduces
an explicit point-based representation where each point in
the point cloud is equipped with four fundamental proper-
ties: mean µ, covariance matrix Σ, opacity α, and spheri-
cal harmonics sh. The covariance matrix Σ can be decom-
posed as Σ = RSST RT , where S is parameterized by a
3D vector s and R is parameterized by a quaternion q. The
rendering process is fully differentiable. For a given view-
point, all 3DGs are first projected onto the 2D image plane.
Then, these projected 2D Gaussians are sorted by depth and
blended together using alpha-compositing to synthesize the

<!-- page 4 -->
Canonical Field
F
t
Attribute Variations
Deformed 3DGs
(a) Deformation Process
Initial 3DGs and Deformation Field
(c) Dynamic Score-Based
Partitioning of 3D Gaussians
Low Dynamic Score
Partition Level 0
High Dynamic Score
Deformation
Static 3DGs
Partition Level 1
Partition Level N
……
…
…
……
………
Multiple
Deformation
Process
Historical Positions
Maximum
Displacement
Variance
Dynamic
Score
(b) Dynamic Score Calculation
(d) Render Process and Losses
Rasterization
Rasterization
Deformation
Deformation
t
Low Dynamic 3DGs
Dynamic Score
Calculation
:3DG
:Deformation Network
:3DGs with Dynamic Scores Ranging from High to Low
:Fusion of Static and Dynamic 3DGs
Figure 3. An overview of MAPo. (a) 3DGs’ deformation process. (b) Compute the dynamic score of 3DGs from history positions during
training. (c) High-dynamic 3DGs are recursively temporally partitioned, and low-dynamic ones are deformed and treated as static. (d)
Dynamic and static 3DGs are combined for rendering. Losses are computed on the left.
final pixel color C:
C =
N
X
i=1
ciα′
i
i−1
Y
j=1
 1 −α′
j

.
(1)
where ci is the view-dependent color computed from sh and
viewpoint, and α′
i is the 2D evaluation of the 3DG’s opacity.
Embedding-Based Deformation
Our work builds upon
the dual deformation paradigm introduced in E-D3DGS [2].
In this paradigm, each 3DG is associated with a learnable
embedding zg, and each timestamp t is represented by a pair
of temporal embeddings: a coarse embedding ztc capturing
low-frequency motion, and a fine embedding ztf for high-
frequency details. A coarse deformation network F and a
fine deformation network Fθ process these embeddings to
predict deformations for all 3DG attributes. The final defor-
mation is the sum of the coarse and fine predictions:
(∆µ, ∆q, ∆s, ∆α, ∆sh) = F(zg, ztc)+Fθ(zg, ztf ). (2)
4. Method
Our approach consists of two main components: a dynamic
score-based partitioning strategy and a cross-frame consis-
tency loss. The overview of our method is shown in Fig. 3.
First, we elaborate on the dynamic score-based partition-
ing strategy, including how we compute a dynamic score
for each 3DG based on its historical positions, and how
this score guides a partitioning where high-dynamic 3DGs
are recursively temporally partitioned while low-dynamic
3DGs are identified as static. Subsequently, we describe our
cross-frame consistency loss, which is designed to address
the visual discontinuities caused by partitioning.
4.1. Dynamic Score-based Partitioning Strategy
4.1.1. Dynamic Score Calculation
To accurately characterize the motion intensity of each
3DG, we design a comprehensive scoring mechanism that
integrates both maximum displacement and position vari-
ance. For each 3DG Gi, we record its spatial position µij
during training, where i ∈{1, 2, · · · , N} denotes the index
of 3DG and j ∈{1, 2, · · · , m} represents the index of the
recorded historical positions. Here, m is a hyperparame-
ter controlling the number of recorded positions per 3DG.
Initially, we considered using the maximum displacement
as the sole metric for quantifying the motion intensity of
3DGs, which we efficiently compute as the length of the
diagonal of the historical positions’ axis-aligned bounding
box. However, upon closer examination of the positional
records, we realized that relying solely on this metric would
be insufficient. For instance, an object with short-term high-
speed motion but long-term stillness may exhibit a large
maximum displacement but relatively low variance. Con-
versely, an object with continuous small oscillations would
have a small maximum displacement but a relatively high
variance, suggesting a consistent and complex yet less pro-
nounced movement pattern. Therefore, to comprehensively
characterize diverse motion behaviors to accurately identify
more challenging motion, we employ two quantitative met-
rics for each 3DG. For the i-th 3DG, its maximum displace-
ment ri and position variance vi are computed as:
ri =
max
j
µij −min
j
µij
,
vi =
m
X
j=1
∥µij −¯µi∥2
m
, (3)

<!-- page 5 -->
(a)
(b)
(c)
(d)
Figure 4. Effectiveness of temporal partitioning strategy and
consistency loss on a toy example. (a) A 3D curve p(t) simulates
a dynamic trajectory. (b) A single point and a single MLP to fit
p(t) for the entire duration; (c) Two points and two corresponding
MLPs for two partitioned time segments; (d) Apply a consistency
loss to (c) at the partition boundary.
Here, maxj µij and minj µij are the element-wise max-
imum and minimum vectors over the historical positions
{µij}m
j=1 for 3DG i. The term ¯µi = 1
m
Pm
j=1 µij represents
the mean position. The maximum displacement ri captures
the peak amplitude of motion, while the variance vi mea-
sures the dispersion around the mean position. After com-
puting the maximum displacement and variance, we map
all 3DGs’ r and v values to the interval [0, 1] via percentile-
based normalization:
˜ri =
100
X
k=1
1(ri ≥qr(k))
100
,
˜vi =
100
X
k=1
1(vi ≥qv(k))
100
, (4)
where 1(·) denotes the indicator function, and qr(k) and
qv(k) are the k-th percentiles of {ri} and {vi}, respectively.
We use the harmonic mean to fuse ˜ri and ˜vi, as it requires
both inputs to be high for a high output. The final dynamic
score Si of the i-th 3DG is denoted as:
Si =
2
1
˜ri+ε +
1
˜vi+ε
,
(5)
where ε = 10−6 is used for numerical stability.
4.1.2. Temporal Partitioning Based on Dynamic Scores
3DGs with high dynamic scores typically correspond to re-
gions with complex or rapid motion. Since a single 3DG
struggles to effectively model long-term or complex mo-
tion characteristics, an intuitive and effective solution is to
partition these high-dynamic 3DGs along the temporal di-
mension based on their dynamic scores, thereby capturing
motion variations more precisely. Fig. 4 demonstrates the
effectiveness of this partitioning through a simple demo.
Building on this insight, we recursively partition the
3DGs along the temporal dimension, progressively subdi-
viding those with higher dynamic scores into finer temporal
segments to achieve more precise dynamic reconstruction.
Specifically, for the complete time range [0, T], each 3DG
maintains two key properties: its partition level l (initial-
ized to 0) and its temporal segment range [tstart, tend] (ini-
tially [0, T]). For notational convenience, all such ranges
are treated as left-closed and right-open.
Let G[tstart,tend]
denote the set of 3DGs assigned to the temporal segment
range [tstart, tend].
Let F[tstart,tend] represent the deforma-
tion network corresponding to the temporal segment range
[tstart, tend].
It is responsible for deforming the 3DGs in
G[tstart,tend].
When a 3DG at level l exhibits a dynamic
score exceeding the current-level threshold τl within its time
range [tstart, tend], we partition it at the temporal midpoint
tmid = (tstart + tend)/2. The original 3DG retains the first
sub-segment [tstart, tmid] while advancing to level l+1, and a
new replica is created for the second sub-segment [tmid, tend]
with identical attributes.
Correspondingly, the deforma-
tion network F[tstart,tend] is replicated to create F[tstart,tmid] and
F[tmid,tend] to model the distinct spatio-temporal deformation
patterns within each sub-segment. Within each new sub-
segment, this partitioning process is applied recursively.
4.1.3. Static 3D Gaussian Partitioning
3DGs with dynamic scores below a predefined threshold
τstatic are identified as static. These static 3DGs have their
attributes initialized once using the output of their associ-
ated deformation network at a randomly sampled timestep.
Subsequently, they are excluded from computations involv-
ing the deformation network during rendering while their
attributes remain optimizable, significantly reducing com-
putational costs.
4.2. Cross-Frame Consistency Loss
Temporal partitioning, while beneficial for modeling com-
plex motions, can introduce visual discontinuities at the par-
tition boundaries. To ensure temporal smoothness, we intro-
duce the cross-frame consistency loss Lcross, which consists
of two components: Lcurrent and Lgt. The Lcurrent evaluates
rendering consistency at the partition boundary. It compares
two renderings of the same frame at this boundary:
Lcurrent = ∥It(Gt, V ) −It(Gt′ , V )∥1 .
(6)
Here, Gt denotes the set of all 3DGs used for rendering at
a timestamp t. t′ is the timestamp from the nearest neigh-
boring temporal segment closest to t. Gt comprises two dis-
tinct subsets: static 3DGs and dynamic 3DGs. The dynamic
3DGs active at timestamp t, denoted as Gd
t , are governed by
their corresponding deformation networks F d
t . It(Gt, V )
denotes the image rendered at timestamp t from viewpoint
V using Gt. By computing the L1 norm between these two
rendered images, Lcurrent captures the discrepancy between
adjacent temporal segments in rendering the same frame.
Optimizing this term helps reduce visual discontinuities, en-
suring smoother transitions between temporal segments.
Although Lcurrent effectively mitigates visual discontinu-
ities across partition boundaries, we observe that relying on
it alone can compromise rendering quality. Since Lcurrent
only enforces self-consistency between adjacent segments

<!-- page 6 -->
(a) 4DGaussians
(b) Ex4DGS
(c) Swift4D
(d) E-D3DGS
(e) Ours
(f) Ground Truth
Figure 5. Qualitative comparisons against existing SOTA methods on the MeetRoom and N3DV dataset.
without an external reference, continuous optimization can
cause them to converge to a consistent but over-smoothed
state, leading to perceptible blurring in dynamic regions. To
counteract this and anchor the reconstruction to the ground
truth, we introduce Lgt.
This loss is designed to enrich
the adjacent segment’s 3DGs with valuable spatio-temporal
context from the current frame. It achieves this by directly
supervising their rendering against the ground-truth image
of the current view V , which is captured at timestamp t:
Lgt =
It(Gt′ , V ) −IGT
1 ,
(7)
where IGT denotes the corresponding ground-truth image at
timestamp t. This contextual enrichment forces Gt′ to learn
to represent the sharp details of the current frame, thereby
preventing over-smoothing and enhancing overall fidelity.
Finally, the overall cross-frame consistency loss, Lcross,
is defined as a weighted combination of Lcurrent and Lgt:
Lcross = 0.5 · Lcurrent + Lgt,
(8)
By optimizing Lcross, we reduce discontinuities across tem-
poral segments while enhancing rendering quality. We ap-
ply Lcross only for training views whose frame indices are
within 5 frames of any partition boundary. Fig. 4 (d) demon-
strates its effectiveness through a simple demo.
Table 1.
Quantitative comparison on the N3DV dataset.
1
flame salmon was trained on only frag1.
2 only reported results
on the flame salmon frag1 and was trained on 8 GPUs. 3 trained
with 90 frames. 4 trained with 50 frames. Storage, training time,
and FPS are measured on flame salmon frag1.
Method
PSNR↑
SSIM↑
LPIPS↓
Storage↓
Training Time↓
FPS↑
DyNeRF1,2
29.58
-
0.083
56MB
1344 hours
0.01
NeRFPlayer1,3
30.69
0.932
0.111
1654MB
5 hours 36 mins
0.06
Mix Voxels
30.30
0.918
0.127
512 MB
1 hour 28 mins
1.01
K-Planes
30.86
0.939
0.096
309 MB
1 hour 33 mins
0.15
HyperReel4
30.37
0.921
0.106
1362 MB
8 hours 42 mins
1.19
D3DGS
28.27
0.917
0.156
75MB
2 hours 17 mins
20.29
4DGS
30.30
0.933
0.069
3.6GB
7 hours 43 mins
54.36
4DGaussians
30.19
0.917
0.061
53 MB
1 hour 13 mins
78.28
Ex4DGS
30.76
0.939
0.056
205 MB
1 hour 5 mins
51.46
Swift4D
30.05
0.931
0.055
116MB
48 mins
138.00
4DGC
30.78
0.938
0.052
225 MB
5 hours 44 mins
124.61
LocalDyGS
30.75
0.933
0.053
102 MB
42 mins
109.30
E-D3DGS
30.79
0.934
0.051
73 MB
2 hours 41 mins
37.51
E-D3DGS (seg)
30.73
0.935
0.049
215 MB
8 hours 32 mins
37.97
Ours
31.33
0.944
0.044
65 MB
1 hour 52 mins
75.64
5. Experiment
5.1. Dataset and Metrics
We evaluate our method on two real-world dynamic scene
datasets: N3DV [15] and Meet Room [14].
The N3DV
dataset includes videos at 30 FPS captured by 20 cameras.
Following previous work [2], we downsample its images to
1352×1014 and segment the longer flame salmon sequence
into four 10s clips.
The Meet Room dataset consists of

<!-- page 7 -->
Table 2. Quantitative comparison on the Meet Room dataset.
Storage, training time, and FPS are calculated on discussion.
Method
PSNR↑
SSIM↑
LPIPS↓
Storage↓
Training Time↓
FPS↑
D3DGS
25.81
0.890
0.233
36 MB
47 mins
42.51
4DGS
26.12
0.896
0.080
5.4 GB
6 hours 32 mins
70.54
4DGaussians
26.16
0.894
0.081
51 MB
1 hour 3 mins
77.26
Ex4DGS
26.46
0.895
0.083
123 MB
1 hour 6 mins
117.49
Swift4D
25.51
0.882
0.085
76 MB
20 mins
109.58
4DGC
26.56
0.901
0.070
224MB
3 hours 26 mins
160.59
LocalDyGS
25.85
0.888
0.084
98 MB
1 hour 7 mins
130.30
E-D3DGS
26.24
0.896
0.081
28 MB
1 hour 36 mins
90.26
E-D3DGS (seg)
26.31
0.900
0.073
89 MB
4 hour 3 mins
85.20
Ours
26.72
0.903
0.066
49 MB
1 hour 19 mins
92.21
videos at a resolution of 1280×720 and 30 FPS from 13
cameras. We report PSNR, SSIM, and LPIPS for render-
ing quality, alongside computational costs including train-
ing time, rendering speed, and storage, which are calculated
on an NVIDIA RTX A6000 unless otherwise specified.
5.2. Implementation Details
Our implementation builds upon the E-D3DGS codebase.
In our experiments, we set the number of recorded histori-
cal positions, m, to 300 and the maximum partition level to
3. We provide complete implementation details in the ap-
pendix, including training specifics and more hyperparam-
eter settings. Furthermore, we include in-depth analyses of
our method’s mechanics and extensive supplementary ex-
periments in the appendix, covering more baseline compar-
isons and additional ablation studies.
5.3. Comparisons
5.3.1. Quantitative Comparisons.
We compare MAPo against current open-source SOTA
methods. To ensure fairness, all 3DGS-based baselines are
evaluated with identical point cloud initializations. In ad-
dition to these SOTA baselines, we additionally introduce
a simple segmentation baseline, E-D3DGS (seg), for com-
parison to highlight the advantages of our approach. This
baseline applies a naive temporal partitioning strategy by
uniformly splitting the video sequence into three indepen-
dent segments and training a separate E-D3DGS model for
each. We defer direct comparisons with methods that lack
public codebases or employ significantly different proto-
cols (e.g., SWinGS, ST-GS) to the appendix, where full de-
tails on all baselines and experimental setups are provided.
As shown in Tab. 1 and Tab. 2, our method consistently
achieves SOTA rendering quality across both datasets while
avoiding prohibitive computational overhead, thus offering
a compelling balance between high fidelity and practical re-
source usage.
5.3.2. Qualitative Comparisons.
We present qualitative results in Fig. 5. The comparison
highlights that baseline methods often produce degraded re-
sults in areas with complex or rapid motion. For example, in
Table 3. Progressive component ablation on Meet Room. Stor-
age, training time, and FPS are calculated on discussion.
Configuration
PSNR↑
SSIM↑
LPIPS↓
Storage↓
Time↓
FPS↑
tOF (Avg/Bnd)↓
Baseline
26.24
0.896
0.081
28 MB
1h36m
90.26
0.082 / 0.074
Baseline (seg)
26.31
0.900
0.073
89 MB
4h3m
85.20
0.080 / 0.185
+Partition
Temporal Partition
1.1 +Max Dis
26.52
0.901
0.070
65 MB
1h41m
55.21
0.079 / 0.084
1.2 +Var
26.63
0.903
0.067
67 MB
1h42m
54.56
0.079 / 0.082
Static Partition
2.0 +Static
26.60
0.903
0.066
48 MB
1h12m
92.59
0.079 / 0.081
+Cross
3.1 +Lcurrent
26.49
0.899
0.071
48 MB
1h18m
92.88
0.078 / 0.074
3.2 +Lgt
26.72
0.903
0.066
49 MB
1h19m
92.21
0.078 / 0.072
Baseline
(1.1) + Max Dis
(1.2) + Var
Ground Truth
Figure 6. Observation of dynamic partition on Vrheadset.
(1.2) + Var
(2.0) + Static
Static Region
Figure 7. Observation of static partition on Salmon.
cases with fast-moving hands or detailed facial expressions,
baseline methods exhibit severe motion blur and loss of de-
tail. Benefiting from our temporal partitioning strategy, our
approach preserves fine details in highly dynamic regions,
yielding renderings that are significantly sharper and more
faithful to the ground truth.
5.4. Ablation Study and Analysis
To evaluate our method, we present a progressive ablation
study in Tab. 3. We establish two reference points: the E-
D3DGS model, which serves as our Baseline, and a naive
temporal slicing approach of E-D3DGS (Baseline (seg)) for
comparison. Building upon the baseline, we incrementally
add our proposed components to validate their effect. These
components include: 1. Temporal Partition, our method for
Temporal Partitioning Based on Dynamic Scores, where we
use Maximum Displacement (1.1 + Max Dis) for dynamic
score computation, then incorporate Variance (1.2 + Var);
2. Static Partition, for Static 3D Gaussian Partitioning; and
3. Cross, the Cross-Frame Consistency Loss, involving se-
quential integration of Lcurrent (3.1 + Lcurrent) and Lgt (3.2 +
Lgt). Components are added incrementally to evaluate their
collective impact. For clarity, we use consistent numbering
to denote the variants at each stage.
Temporal Partition.
Our temporal partition strategy per-
forms specialized temporal modeling for highly dynamic
3D Gaussians to reconstruct complex motion details that are

<!-- page 8 -->
Annotation
(2.0) + Static
(3.1) + Lcurrent
(3.2) + Lgt
Ground Truth
Figure 8. Observation of Lcross’s effect on visual discontinuity.
The four images on the right are obtained by vertically concate-
nating the horizontal line in the “Annotation” over time.
73
Frame
74
75
76
Partition Boundary
Moving spray gun
Figure 9. Ablation study on the Lcross. We visualize how Lcross
improves temporal consistency and rendering quality across a par-
tition boundary (frames 74-75). The first four columns show the
frame sequence to evaluate smoothness, while the fifth column
provides a magnified view to compare fine detail reconstruction
on the fast-moving spray gun.
averaged out in a single, unified model. Tab. 3 and Fig. 6
clearly demonstrate the quality improvements achieved
through temporal partition. Notably, even our intermedi-
ate partitioning variants (”1.1” and ”1.2”) achieve signifi-
cantly superior results compared to the naive temporal slic-
ing baseline, Baseline (seg). This highlights that our dy-
namic score-based approach is far more effective at im-
proving quality than simply training independent models on
fixed temporal segments.
Static Partition.
Our static partitioning strategy lever-
ages the dynamic score to identify and convert static 3D
Gaussians, aiming to improve training efficiency, rendering
Table 4. Ablation study on the partition level parameter. All
experiments are conducted on the flame salmon frag3.
Method
PSNR↑
SSIM↑
LPIPS↓
Storage↓
Training Time↓
FPS↑
0
29.93
0.923
0.61
44MB
1 hours 13 mins
95.21
1
30.08
0.927
0.56
51MB
1 hours 24 mins
88.13
2
30.21
0.932
0.54
59MB
1 hours 37 mins
82.81
3
30.30
0.934
0.52
70MB
1 hours 56 mins
74.58
4
30.32
0.936
0.50
88MB
2 hours 22 mins
64.25
5
30.36
0.936
0.49
103MB
2 hours 40 mins
57.05
speed, and storage. As shown in Tab.3, static partitioning
significantly reduces computational costs compared to us-
ing temporal partitioning alone.
Furthermore, Tab.3 and
Fig. 7 demonstrate that applying the static partitioning strat-
egy does not cause a decline in reconstruction quality.
Cross-Frame
Consistency
Loss.
We
introduce
the
Lcross to enforce continuity at boundaries and further en-
hance overall rendering quality. We quantitatively evaluate
temporal consistency using the tOF [4] metric, where lower
values indicate better temporal consistency. As shown in
Tab. 3, we measure tOF over both the entire video (Avg)
and at the partition boundaries (Bnd).
The data reveals
that applying only temporal partitioning, while reducing
the average tOF, causes the boundary tOF to rise signifi-
cantly, which confirms the existence of visual discontinuity
at the boundaries. Through the step-by-step integration of
Lcurrent and Lgt, the boundary tOF is progressively sup-
pressed, ultimately dropping below the baseline level in our
full model and completely mitigating the issue. Our quali-
tative results further corroborate the dual efficacy of Lcross.
As shown in Fig. 8, the time-slice visualization intuitively
reveals how our method significantly mitigates the severe
visual discontinuity at the boundaries. Concurrently, the
consecutive frame sequence in Fig. 9 also demonstrates that
our full model (+ Lgt) not only achieves the smoothest tran-
sition but also yields additional quality gains.
Analysis of Maximum Partition Level.
We conduct an
ablation study on the maximum partition level to analyze
its impact on reconstruction quality and computational cost.
As shown in Tab. 4, progressively increasing the partition
level from 0 to 5 yields a general trend of improved ren-
dering quality. However, we observe diminishing returns
in quality gains after level 3, while computational and stor-
age costs continue to increase steadily. Therefore, we select
level 3 as the maximum partition level for our main experi-
ments to strike an optimal balance. Furthermore, the results
highlight that the cost increase is manageable and not ex-
plosive; for instance, the storage at level 4 is merely double
that of level 0, demonstrating the efficiency of our scheme.

<!-- page 9 -->
6. Conclusion
We proposed MAPo, a novel framework for high-fidelity
dynamic scene reconstruction. MAPo employs a hierarchi-
cal partitioning strategy guided by a dynamic score, en-
abling specialized modeling for complex motion regions
while treating static ones efficiently. To ensure temporal
smoothness, we introduce a cross-frame consistency loss to
mitigate visual discontinuities at the partition boundaries.
Extensive experiments demonstrate that MAPo achieves
SOTA rendering quality while maintaining competitive
computational efficiency.
References
[1] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael
Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil
Kim.
Hyperreel:
High-fidelity 6-dof video with ray-
conditioned sampling, 2023. 3
[2] Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun
Bang, and Youngjung Uh. Per-gaussian embedding-based
deformation for deformable 3d gaussian splatting. In Euro-
pean Conference on Computer Vision (ECCV), 2024. 2, 3, 4,
6
[3] Ang Cao and Justin Johnson. Hexplane: A fast representa-
tion for dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 130–141, 2023. 1, 3
[4] Mengyu Chu, You Xie, Jonas Mayer, Laura Leal-Taix´e,
and Nils Thuerey.
Learning temporal coherence via self-
supervision for gan-based video generation. ACM Transac-
tions on Graphics, 39(4), 2020. 8
[5] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wen-
zheng Chen, and Baoquan Chen.
4d gaussian splatting:
Towards efficient novel view synthesis for dynamic scenes,
2024. 3
[6] Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xi-
aopeng Zhang, Wenyu Liu, Matthias Nießner, and Qi Tian.
Fast dynamic radiance fields with time-aware neural vox-
els. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9,
2022. 1
[7] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk
Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes:
Explicit radiance fields in space, time, and appearance. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 12479–12488, 2023. 1,
3
[8] Qiankun Gao, Jiarui Meng, Chengxiang Wen, Jie Chen,
and Jian Zhang. Hicom: Hierarchical coherent motion for
streamable dynamic scene with 3d gaussian splatting, 2024.
3
[9] Quankai Gao, Qiangeng Xu, Zhe Cao, Ben Mildenhall, Wen-
chao Ma, Le Chen, Danhang Tang, and Ulrich Neumann.
Gaussianflow: Splatting gaussian dynamics for 4d content
creation, 2024. 3
[10] Qiang Hu, Zihan Zheng, Houqiang Zhong, Sihua Fu, Li
Song, XiaoyunZhang, Guangtao Zhai, and Yanfeng Wang.
4dgc:
Rate-aware 4d gaussian compression for efficient
streamable free-viewpoint video, 2025. 3
[11] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu,
Yan-Pei Cao, and Xiaojuan Qi.
Sc-gs: Sparse-controlled
gaussian splatting for editable dynamic scenes, 2023. 2
[12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4):1–14, 2023. 1
[13] Junoh Lee, Chang-Yeon Won, Hyunjun Jung, Inhwan Bae,
and Hae-Gon Jeon. Fully explicit dynamic gaussian splat-
ting, 2024. 2, 3
[14] Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and
Ping Tan.
Streaming radiance fields for 3d video synthe-
sis. Advances in Neural Information Processing Systems, 35:
13485–13498, 2022. 1, 6
[15] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon
Green, Christoph Lassner, Changil Kim, Tanner Schmidt,
Steven Lovegrove, Michael Goesele, Richard Newcombe,
et al. Neural 3d video synthesis from multi-view video. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 5521–5531, 2022. 1, 3,
6
[16] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis,
2023. 3
[17] Jiahao Lu, Jiacheng Deng, Ruijie Zhu, Yanzhe Liang, Wenfei
Yang, Tianzhu Zhang, and Xu Zhou. Dn-4dgs: Denoised de-
formable network with temporal-spatial aggregation for dy-
namic scene rendering, 2024. 2, 3
[18] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering, 2023. 3
[19] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
representing scenes as neural radiance fields for view synthe-
sis. Commun. ACM, 65(1):99–106, 2021. 1, 3
[20] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien
Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo
Martin-Brualla. Nerfies: Deformable neural radiance fields.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 5865–5874, 2021. 1, 3
[21] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T
Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-
Brualla, and Steven M Seitz.
Hypernerf:
A higher-
dimensional representation for topologically varying neural
radiance fields. arXiv preprint arXiv:2106.13228, 2021. 3
[22] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer.
D-nerf: Neural radiance fields
for dynamic scenes. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
10318–10327, 2021. 3
[23] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu,
Hongwen Zhang, and Yebin Liu. Tensor4d : Efficient neural
4d decomposition for high-fidelity dynamic reconstruction
and rendering, 2023. 1
[24] Richard Shaw,
Michal Nazarczuk,
Jifei Song,
Arthur
Moreau, Sibi Catley-Chandar, Helisa Dhamo, and Eduardo
Perez-Pellitero. Swings: Sliding windows for dynamic 3d
gaussian splatting, 2024. 3

<!-- page 10 -->
[25] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele
Chen, Junsong Yuan, Yi Xu, and Andreas Geiger.
Nerf-
player: A streamable dynamic scene representation with de-
composed neural radiance fields, 2023. 3
[26] Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei
Zhao, and Wei Xing.
3dgstream: On-the-fly training of
3d gaussians for efficient streaming of photo-realistic free-
viewpoint videos, 2024. 3
[27] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering,
2023. 2, 3
[28] Jiahao Wu, Rui Peng, Jianbo Jiao, Jiayu Yang, Luyang Tang,
Kaiqiang Xiong, Jie Liang, Jinbo Yan, Runling Liu, and
Ronggang Wang.
Localdygs: Multi-view global dynamic
scene modeling via adaptive local implicit feature decou-
pling, 2025. 3
[29] Jiahao Wu, Rui Peng, Zhiyan Wang, Lu Xiao, Luyang
Tang, Jinbo Yan, Kaiqiang Xiong, and Ronggang Wang.
Swift4d:adaptive divide-and-conquer gaussian splatting for
compact and efficient reconstruction of dynamic scene,
2025. 2, 3
[30] Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil
Kim. Space-time neural irradiance fields for free-viewpoint
video. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 9421–9431,
2021. 1
[31] Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun,
Hujun Bao, and Xiaowei Zhou. Representing long volumet-
ric video with temporal gaussian hierarchy. ACM Transac-
tions on Graphics, 43(6):1–18, 2024. 3
[32] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction, 2023. 2, 3
[33] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-
time photorealistic dynamic scene representation and render-
ing with 4d gaussian splatting, 2024. 2, 3
