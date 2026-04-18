<!-- page 1 -->
1
ProGS: Towards Progressive Coding for 3D
Gaussian Splatting
Zhiye Tang, Lingzhuo Liu, Shengjie Jiao, Qiudan Zhang, Member, IEEE, Junhui Hou, Senior Member, IEEE,
You Yang, Senior Member, IEEE, Xu Wang, Member, IEEE
Abstract—With the emergence of 3D Gaussian Splatting
(3DGS), numerous pioneering efforts have been made to address
the effective compression issue of massive 3DGS data. 3DGS
offers an efficient and scalable representation of 3D scenes
by utilizing learnable 3D Gaussians, but the large size of the
generated data has posed significant challenges for storage and
transmission. Existing methods, however, have been limited by
their inability to support progressive coding, a crucial feature
in streaming applications with varying bandwidth. To tackle this
limitation, this paper introduce a novel approach that organizes
3DGS data into an octree structure, enabling efficient progressive
coding. The proposed ProGS is a streaming-friendly codec that
facilitates progressive coding for 3D Gaussian splatting, and
significantly improves both compression efficiency and visual
fidelity. The proposed method incorporates mutual information
enhancement mechanisms to mitigate structural redundancy,
leveraging the relevance between nodes in the octree hierarchy.
By adapting the octree structure and dynamically adjusting
the anchor nodes, ProGS ensures scalable data compression
without compromising the rendering quality. ProGS achieves
a remarkable 45× reduction in file storage compared to the
original 3DGS format, while simultaneously improving visual
performance by over 10%. This demonstrates that ProGS can
provide a robust solution for real-time applications with varying
network conditions.
Index Terms—Volumetric Scene Streaming, 3D Gaussian Splat-
ting, Deep Reinforcement Learning
I. INTRODUCTION
Recently, 3D Gaussian Splatting (3DGS) [1] emerged as a
prominent novel paradigm in the field of novel view synthesis
(NVS) [2]. Unlike neural radiance fields (NeRF) [3], which
rely on dense neural networks to represent a 3D scene,
3DGS explicitly represents each scene as a collection of
learnable 3D Gaussians. These Gaussians are initialized from a
structure-from-motion (SfM) point cloud [4], which provides a
sparse yet efficient representation of the scene’s geometry. The
This work was supported in part by the National Natural Science Foundation
of China under Grant 62371310 and 62501403, in part by the Shenzhen
Science and Technology Program (JCYJ20241202124415021), in part by
the Guangdong Basic and Applied Basic Research Foundation under Grant
2023A1515011236. (Corresponding author: Xu Wang.)
Zhiye Tang, Lingzhuo Liu, Shengjie Jiao, Qiudan Zhang and Xu
Wang
are
with
the
College
of
Computer
Science
and
Software
Engineering,
Shenzhen
University,
Shenzhen,
518060,
China.
Email:
(tangzhiye2022@foxmail.com, karrie019@qq.com, jiaoshengjie@icloud.com,
qiudanzhang@szu.edu.cn, wangxu@szu.edu.cn).
Junhui
Hou is
with Department
of Computer
Science, City Uni-
versity
of
Hong
Kong,
Kowloon,
Hong
Kong
SAR,
China.
Email:
(jh.hou@cityu.edu.hk).
You Yang is with School of Electronic Information and Communications,
Huazhong University of Science and Technology, Wuhan, 430074, China.
Email: (yangyou@hust.edu.cn).
Gaussians are visualized as ellipsoids in the 3D space, each
characterized by learnable parameters such as pose, shape,
and color. In the rendering process, these 3D Gaussians are
“splatted” onto an imaging plane corresponding to a particular
viewpoint, and the final image is synthesized through α-
blending [5]. This splatting-based rendering process simplifies
and accelerates scene rendering by avoiding the frequent and
computationally expensive ray sampling operations required
by methods like NeRF [3]. Furthermore, it facilitates high-
quality, differentiable rasterization. However, this approach
is not without its challenges. While 3DGS enables real-time
rendering, it leads to high memory and storage consumption
due to the large number of 3D Gaussians (often millions)
required to represent a single scene. As a result, there is
a critical need to develop efficient compression strategies
for 3DGS that reduce its data volume without sacrificing
rendering quality. Recent advancements aim to address this
challenge by exploring data structure optimization and the use
of hierarchical encoding techniques, such as octrees, to reduce
storage overhead while maintaining scene fidelity.
With the unorganized and sparse nature of 3DGS, it is chal-
lenging to achieve efficient compression of 3D Gaussians [2],
[6]. These difficulties stem from the inherently high memory
requirements and the lack of structural organization in 3DGS,
which typically results in a large amount of redundant data. In
response, numerous approaches have been proposed to tackle
this issue from various perspectives. Some methods introduce
pruning techniques, such as applying masks to discard 3D
Gaussians with low utility, thereby reducing the overall data
size [7]–[9]. Additionally, codebook-based quantization meth-
ods have been explored, aiming to further compress the data
by grouping similar 3D Gaussians into predefined sets, thus
improving storage efficiency [10]. Despite their effectiveness,
these methods focus primarily on directly compressing the
original 3DGS format, and they often overlook the potential to
leverage the structural relationships between Gaussians. The
lack of such relational awareness limits the opportunity for
creating even more compact representations.
Building on this, Scaffold-GS [11] took inspiration from
grid-guided NeRF [12] and proposed a more structured ap-
proach. This method replaced 3D Gaussians with learnable
anchors, initialized from SfM point clouds and aligned to a
voxel grid. During both training and rendering, these anchors
dynamically generate surrounding neural 3D Gaussians via
multi-layer perceptrons (MLPs), significantly reducing mem-
ory usage. This anchor-based framework has been shown to
provide more efficient representations. More recently, further
arXiv:2603.09703v1  [cs.CV]  10 Mar 2026

<!-- page 2 -->
2
13.14MB
+8.75MB
+4.94MB
+2.19MB
Ground Truth
LoD 1
23.59dB
27.29dB
29.54dB
29.77dB
LoD 2
LoD 3
LoD 4
Fig. 1: ProGS structures a 3D scene into a group of octrees with multiple layers. Each level-of-detail (LoD) l of the whole
scene can be viewed as the first l layers of the octrees. With context-based progressive coding, ProGS can be delivered and
decoded in scalable bitrate, enabling adaptability for bandwidth variations and error conditions.
advancements have incorporated context-based compression
techniques, such as hash-grid-assisted coding, to better capture
the interdependencies between anchor attributes and enhance
storage efficiency [13]. While these techniques achieved sig-
nificant performance improvements, they do not support pro-
gressive coding, which is a crucial feature for scalable 3D
scene streaming. These methods encode 3D scenes into static
files that are decoded in their entirety, failing to adapt dynam-
ically to varying network conditions and bitrates. Progressive
coding, on the other hand, allows for more flexible data
transmission, where coarse-grained data is first transmitted and
refined with finer details, improving channel efficiency and
visual quality as the stream progresses.
To push the boundaries of streaming-friendly 3DGS repre-
sentation, this work draws inspiration from the point cloud
domain, specifically contemplating the idea of representing
3D space using a tree structure [14]. This approach prompts
the essential question: Is it possible to efficiently organize
3DGS scenes into tree structures with varying granularities?
The proposed solution is ProGS, a streaming-friendly codec
designed to facilitate Progressive coding for 3D Gaussian
Splatting. In particular, following the anchor-based design
of Scaffold-GS [11], ProGS mimics the octree structure and
divides the anchor collection into multiple levels of detail
(LoDs), ensuring that each level progressively refines the scene
representation. During the training process, ProGS adaptively
grows and prunes the branches of the octrees, dynamically
adjusting the resolution of each part of the scene, which
reduces memory usage and optimizes the encoding efficiency.
However, such a LoD structure faces inherent challenges.
Lower LoDs represent incomplete 3D scenes, resulting in
visual degradation of the rendered images. To overcome this,
ProGS hypothesize that the representation goals of low and
high LoDs are aligned, and thus, it is affirmative to improve the
performance of lower-level anchors by enhancing the mutual
information (MI) [15] between parent and child nodes within
the octree. By enhancing the common information, lower-
level anchors can more accurately represent the scene’s details,
significantly improving the progressive coding performance.
Furthermore, building on the work of HAC [13], ProGS also
utilizes context-based encoding techniques, where each an-
chor’s context is queried from its location and passed through
arithmetic coding (AE) [16], ensuring the generation of a
compact binary bitstream. This approach not only ensures that
the 3DGS data is compactly represented but also guarantees
efficient, progressive decoding and reconstruction for stream-
ing applications. The main contributions of this paper can be
summarized as follows:
1) A progressive codec for 3DGS: ProGS is a codec that
facilitates progressive coding for 3DGS by constructing
volumetric scenes as hierarchical octree structures. This
novel approach not only allows for efficient encoding
and streaming of 3DGS data but also enables adaptive
bitrate transmission, where the scene’s details can be
progressively refined during the decoding process based
on available bandwidth.
2) MI enhancement for low LoDs: To address the chal-
lenge of visual quality degradation due to the incomplete
representation of 3D scenes at lower LoDs, MI enhance-
ment mechanisms are introduced. By leveraging the
common information shared between parent and child
nodes in the octree, ProGS is able to enhance the ability
of lower-level anchors to more accurately represent the
full 3D scene. This process significantly improves the
rendering quality of lower LoDs, mitigating the loss of
detail and ensuring smoother transitions as the scene is
progressively refined.
3) Coarse-to-fine optimization: To further optimize the
compression efficiency of ProGS, a coarse-to-fine op-
timization scheme is proposed. This method facilitates
the reduction of data redundancy and promotes MI be-
tween anchors, enabling a more compact representation.
By adjusting the anchor placement dynamically during
training, ProGS achieves a balanced representation that
preserves high-fidelity details in areas of interest while
minimizing the storage requirements for less critical
regions of the scene.
4) Extensive experimental validation: Extensive experi-
ments demonstrate that ProGS achieves comparable or
superior fidelity compared to state-of-the-art (SOTA)
methods. Remarkably, ProGS can reduce the number of
anchors used for scene representation by up to 67%,
yet still maintain nearly 80% of the visual perfor-
mance. These results highlight ProGS’s ability to achieve
substantial compression without sacrificing rendering
quality, making it a promising solution for real-time

<!-- page 3 -->
3
applications where bandwidth and storage efficiency are
crucial.
II. RELATED WORK
A. Compression for 3D Gaussian Splatting
3DGS [1] has emerged as an innovative technique for repre-
senting 3D scenes, offering efficient ways to capture complex
details within volumetric spaces by utilizing anisotropic 3D
Gaussians. These Gaussians, which represent scene elements
as ellipsoids with learnable poses, shapes, and colors, provide
a compact representation that enables high-quality rendering
through differentiable rasterization. However, a significant
challenge arises from the large storage cost of explicitly
recording each Gaussian, as each one occupies up to 248 bytes,
necessitating high-efficiency compression techniques to handle
large-scale 3DGS data.
The existing 3DGS compression approaches can be broadly
categorized into two main types: primitive-based methods
and anchor-based methods. Primitive-based methods focus
on directly compressing the 3D Gaussian primitives used in
the original 3DGS format. For example, Lee et al. [7] intro-
duced pruning techniques to remove redundant and less in-
formative Gaussians, thereby reducing memory requirements.
Similarly, approaches like those in [8], [9], [17] utilized
vector quantization methods to group similar Gaussians into
codebooks for more efficient encoding, while LocoGS [18]
decomposed Gaussian attributes into explicit and implicit
parts, which also effectively reduced information redundancy.
To address the non-structural nature of the original 3DGS
format, CompGS [19] proposed a hybrid primitive structure
that leverages the predictive relationships between Gaussians
to improve compression. MesonGS [20] borrowed the oc-
tree voxelization and region-adaptive hierarchical transform
(RAHT) [21] from point cloud compression, also achieved
promising performance. More recently, Chen et al. proposed
FCGS [22], which is a plug-and-play codec for any pre-
trained 3D Gaussian scenes, pushing the boundaries of 3DGS
coding. These methods, while successful in reducing the size
of the 3DGS representation, often fail to exploit the underlying
structural relationships between the Gaussians, which could
lead to even more efficient compressions.
On the other hand, anchor-based approaches aim to re-
place the original 3D Gaussians with anchors, which are more
compact and can represent multiple surrounding Gaussians.
Scaffold-GS [11] was one of the first to introduce this ap-
proach, using anchor-centered features to represent 3D scenes.
Each anchor is initialized from a SfM point cloud and aligned
to a voxel grid. During both training and rendering, anchors
generate a set of surrounding neural 3D Gaussians via MLPs,
thus saving considerable memory and storage. More recently,
HAC [13] further advanced anchor-based compression by
using a context-based hash-grid assisted coding framework,
which incorporates adaptive quantization techniques to en-
hance compression efficiency. This framework achieved SOTA
performance in 3DGS compression, significantly reducing data
size while maintaining high visual fidelity.
Although these approaches have made significant strides
in reducing the storage and memory requirements of 3DGS
representations, they primarily focus on single-bitrate sce-
narios, where a fixed amount of data is compressed and
transmitted. This approach does not effectively handle the
variability in available network bandwidth, especially for real-
time streaming applications where bitrate adaptation is cru-
cial. These methods generally encode 3D scenes into static
files, which limits their flexibility and scalability for dynamic
transmission. Progressive coding, on the other hand, allows
for scenes to be progressively decoded and refined over time,
enabling higher channel efficiency and better quality as more
data becomes available. This paper introduces ProGS, a novel
codec designed specifically for progressive coding of 3DGS.
ProGS organizes scenes into hierarchical octree-like structures,
which are adaptively adjusted during training to support pro-
gressive refinement. This method not only achieves efficient
compression but also ensures that the rendering quality is
not compromised during progressive decoding, marking a
significant advancement over previous methods.
B. Progressive Coding
For now, progressive coding has not yet been extensively
explored for 3DGS, but it has attracted considerable attention
in the fields of image, video, and point cloud compression.
Some progressive codecs, such as traditional JPEG [23],
JPEG2000 [24] and recent [25] for images, as well as
H.264/SVC [26] and SHVC [27] for video, typically construct
enhanced layers for media content. These codecs enable the
adaptive selection and decoding of data based on the current
network conditions and the device capabilities, allowing for
efficient streaming and rendering across varying bandwidths.
This adaptive nature makes progressive coding highly valuable
for multimedia content delivery, where high-quality visuals
are prioritized, but only available bandwidth is used. In par-
allel, learning-based approaches have also gained popularity.
Notably, recurrent neural networks (RNNs) have been em-
ployed in various works [28]–[33], capturing representations
at lower bitrates and gradually refining them to generate high-
resolution images. More recently, the adoption of trit-plane
coding methods, as seen in [34], [35], has yielded comparable
gains, particularly in image and video compression. Mao et
al. [36] leveraged priors from a generative adverseral network
(GAN) [37], also improved upon traditional methods. Further-
more, Dasari et al. [38] and Lin et al. [39] have contributed
novel scalable approaches for video coding, significantly en-
hancing the scalability and efficiency of video codecs for
modern applications.
However, unlike the regular and structured nature of images
and videos, point clouds represent 3D data as discrete point
sets that are unevenly distributed across a three-dimensional
space. This inherent irregularity makes it particularly challeng-
ing to apply progressive coding to point clouds. G-PCC [40]
was a pioneering effort that introduced scalability for point
cloud compression, utilizing the octree structure and LoD
schemes to improve data representation and compression.
Building upon this foundation, works like [41] and [42] have
leveraged learning-based models integrated with G-PCC to
further enhance compression efficiency for LoD-structured

<!-- page 4 -->
4
Ground Truth Images
Ground Truth Images
Adaptive Anchor Adjustment
Multi-Level Rendering
Anchor
Entropy
Model
Decoded Attributes
Octree Construction
Octree Construction
SfM Point Cloud
SfM Point Cloud
Camera Poses
Camera Poses
{
𝜇  
𝜎  
{
𝜇  
𝜎  
Distribution Estimation
{
𝜇  
𝜎  
Distribution Estimation
Neural Gaussian Generation
 𝑥, 𝑦, 𝑧  
𝓛𝐜𝟐𝐟 
𝓛𝐍𝐂𝐄 
𝓛𝐞 
AE/AD
Training
Evaluating
Training
Evaluating
Quantized Attributes
Quantized Attributes
MLP𝑑 
MLP𝑞 
MLP𝑔 
Hash-Grid ℋ 
Attributes 𝒜 
Level 1
Level 2
Level 3
{
{
LoD 3
LoD 2
Fig. 2: Overview of the proposed ProGS. ProGS constructs an initial octree group using the input SfM point cloud. The l-th
LoD of ProGS contains all anchors with levels beneath l. During training, ProGS adaptively adjusts the anchors according to
the optimization gradients. For clear illustration, here uses quadtree of 2D planes to demonstrate the idea of tree-structured
anchors.
point clouds. While these methods have made strides in
improving point cloud compression, they still face challenges
in handling implicit attributes of 3DGS scenes, such as co-
variance and anchor features. These attributes are particularly
sensitive to prediction errors, which can degrade the quality
of the resulting 3D scene representation. Therefore, a more
refined approach for progressive coding should involve the
LoD partitioning strategy during scene training processes and
incorporate context-based entropy coding to ensure lossless
compression.
Among the efforts in this direction, Octree-GS [43] aimed
to incorporate multi-resolution anchors to achieve LoD-
structured 3DGS. However, despite these efforts, the approach
fell short in providing sufficiently refined representations for
3D scenes, primarily due to the lack of entropy coding for
data reduction. This omission resulted in limited performance
when compared to other methods in the field. In contrast, this
paper proposes ProGS, which models 3DGS scenes as octrees
and incorporates MI enhancement to improve compression
efficiency. By utilizing the relationship between parent and
child nodes within the octree, ProGS enables better scene
representation at lower LoDs, facilitating efficient progressive
coding and improving both compression and rendering qual-
ity.
III. METHODOLOGY
The framework of ProGS is illustrated in Fig. 2. Specifically,
ProGS begins by constructing an initial octree structure from
the input SfM point cloud, where the nodes in the octree
correspond to anchors, each associated with a set of attributes
A, including feature vectors, scaling factors, and offsets. These
anchors serve as the foundational elements for predicting
surrounding neural Gaussian clusters, which represent the
scene’s details at varying levels of granularity. The anchors are
distributed at different depths within the octree, and each level
corresponds to a voxel region in the 3D space, where the gran-
ularity of the representation increases with each subsequent
LoD. The l-th LoD of ProGS contains all anchors from level
1 through to level l, with higher LoDs offering progressively
finer resolutions of the scene. During the training process,
ProGS employs an adaptive anchor adjustment strategy that
dynamically grows and prunes the octree branches based on
optimization gradients. This ensures that the octree structure
is both compact and flexible, accurately reflecting the scene’s
most significant features while minimizing redundancy. To
assist with this dynamic adjustment, ProGS utilizes a hash-
grid H, which is queried to estimate the distribution of anchor
attributes and facilitate adaptive quantization. The quantized
attributes, or their decoded counterparts in evaluation mode,
are then used to generate neural Gaussian clusters, which
are further employed in multi-level differentiable rendering.
This process is crucial for ensuring that each level of the
octree contributes to the final rendered image, as shown in
Fig. 2. In addition, to enhance the visual quality of lower-level
anchors, ProGS incorporates the information noise contrastive
estimation (InfoNCE) loss [44], denoted as LNCE, along with
a coarse-to-fine optimization loss Lc2f for MI enhancement.
These mechanisms help improve the representation of the
scene at lower LoDs by promoting consistency between parent
and child nodes in the octree. A more detailed discussion of
the MI enhancement approach is provided in Sec. III-D.
A. Preliminary
3DGS represents a 3D scene as a collection of anisotropic
Gaussians and renders viewpoint images via splatting-based,
differentiable rasterization. Concretely, each Gaussian is pa-
rameterized by a mean (center) µ ∈R3 and a covariance
Σ ∈R3×3,
G(x) = exp
 −1
2(x −µ)⊤Σ−1(x −µ)

,
(1)
where x ∈R3 is an arbitrary 3D location. To guarantee
positive semi-definiteness and enable stable optimization, the
covariance is factorized into a rotation and a non-negative
scaling, Σ = RSS⊤R⊤, where R ∈SO(3) (e.g., parame-
terized by quaternions) defines the Gaussian’s orientation and
S ∈R3×3 is diagonal, controlling the principal-axis scales.

<!-- page 5 -->
5
For image synthesis from a given camera pose, Gaussians are
projected and splatted to the image plane; the per-pixel color
C ∈R3 is then computed with front-to-back α-blending
C =
X
i∈I
ci αi
i−1
Y
j=1
(1 −αj),
(2)
where I denotes the depth-sorted set of Gaussians contributing
to that pixel, αi ∈[0, 1] is the opacity (accumulated transmit-
tance), and ci ∈R3 is the view-dependent color predicted from
spherical harmonics (SH) bases [45]. This rasterization-style
rendering is differentiable w.r.t. Gaussian parameters, making
it amenable to end-to-end learning while naturally handling
occlusion via the transmittance product.
Building on the original framework [1], Scaffold-GS [11]
introduced a more storage-efficient, anchor-based formulation
by replacing explicitly stored Gaussians with sparse anchors.
Each anchor is defined as
ai =

xi, Ai
	
,
xi ∈R3,
(3)
and carries a compact attribute set
Ai =

fi ∈RDf , si ∈RDs, oi ∈RDo×3	
,
(4)
where fi encodes an anchor feature, si parameterizes per-
Gaussian scales/orientations, and oi stores 3D offsets that
place multiple neural Gaussians relative to the anchor center.
In practice, fi is fed to a lightweight generator MLPg to
synthesize a local cluster of Gaussians around anchor i;
their centers are given by xi shifted by rows of oi, while
shapes (and ultimately Σ) are derived from si together with
learned rotation parameterizations. The resulting cluster is
then rendered via the same α-blending rule as above. By
decoupling scene detail from the number of explicitly stored
primitives, the anchor-based design drastically reduces storage
and enables hierarchical organization (e.g., LoDs and octrees),
which ProGS further leverages for progressive coding.
B. Octree Construction
A well-defined LoD partitioning scheme is crucial for en-
abling efficient progressive coding. Octree-GS [43] introduced
multi-resolution anchors to represent voxel regions at varying
granularities, incorporating a gradient-guided anchor adjust-
ment strategy. While this approach allows for multi-resolution
representation, their proposed strategy is position-decoupled,
which leads to a loosely organized structure. This decoupling
fails to fully exploit the interdependencies between anchors
at different levels, resulting in an inefficient representation of
the volumetric scene. To overcome this limitation, this work
presents a more structured approach in ProGS. In the proposed
pipeline, ProGS first estimates an approximate bounding box
based on the inputted SfM point cloud P , which provides a
sparse yet informative representation of the 3D scene. This
bounding box serves as the foundation for constructing the
hierarchical octree structure.
Given the base depth lb and the number of LoDs L, the
entire scene is divided into a set of octrees with varying voxel
𝑐𝑛 
𝑐𝑗 
𝑐𝑞 
𝑐𝑝𝑗 
Pruned
(i)
(iii)
(ii)
Anchors
Anchors
Gaussians
Gaussians
Anchors
Gaussians
Anchors with Low
Accumulated Opacity
Anchor
Candidates
𝐺𝑚 
𝐺𝑖 
𝐺𝑖 
𝐺𝑝 
𝐺𝑝 
Level 𝑙 
Level 𝑙−1 
Level 𝑙 
Level 𝑙+ 1 
Fig. 3: Illustration of the anchor adjustment strategy. This
figure takes 2D quadtree planes as examples. (i) Candidate
cn placed at the same level of Gaussian Gm. (ii) Candidates
cj and cq placed at the upper level of Gaussians Gi and Gq.
With no parent anchor for cj, cpj is queued. (iii) The pruning
operation.
sizes. Each LoD represents a different level of granularity, with
the voxel sizes defined by the equation
vl =
vB
2lb+l ,
(5)
where vB denotes the length of the estimated bounding box,
and l is the current LoD level. The voxel size decreases
exponentially as it progresses through the LoD levels, allowing
for a finer resolution in higher levels of detail. The complete
representation of LoD l consists of the anchors from level
1 to level l, with each level containing increasingly refined
spatial information. Once the initial octrees are constructed,
all points in the SfM point cloud are treated as potential
anchor candidates for the L-th level. These points are then
integrated into the octree structure, ensuring a compact and
efficient representation of the scene.
To further enhance the octree structure, ProGS introduces
a novel anchor adjustment strategy, discussed in detail in
Sec. III-C, which adaptively grows and prunes octree branches
during training based on the optimization gradient. This
adaptive process ensures that the octree structure maintains
its strict hierarchical properties, while also allowing for the
dynamic refinement of the scene representation. The result is
a more compact and efficient octree that accurately reflects
the underlying 3D scene, without unnecessary redundancy.
By adjusting the octree in this manner, ProGS ensures that
only the most relevant data is retained at each level, thus
improving both the compression efficiency and the quality of
the progressive decoding process.
C. Adaptive Anchor Adjustment
The main challenge to guaranteeing octree characteristics
of ProGS is to design an anchor adjustment strategy that
ensures each anchor is a node of the octrees without affecting
the high-fidelity representation of a 3D scene. Similar to [1],

<!-- page 6 -->
6
ProGS grow anchors on the locations where the Gaussians find
significant and eliminate trivial branches.
The growing operation contains three steps:
Step 1: Anchor Candidate Determination. The gradient-
guided anchor candidate determination specifically targets
significant Gaussians defined by their optimization gradients.
For each octree-level l, ProGS computes a threshold:
τ l
g = τg × 2βl,
(6)
where τg and β are pre-set hyperparameters. The average
gradient
∇Gl
i
 of each neural Gaussian at level l over the
latest T iterations is also collected for further processes. As
shown in Fig. 3 (i-ii), Gaussians are categorized into three
types: significant (τ l+1
g
≥
∇Gl
i
 > τ l
g), very significant
(
∇Gl
i
 > τ l+1
g
) and non-significant. For each significant or
very significant Gaussian Gl
i, an anchor candidate cj is then
placed at different levels based on the Gaussian significance,
with each anchor candidate cj can be written as
cj =



(µl
i, l),
τ l+1
g
≥
∇Gl
i
 > τ l
g
(µl
i, l + 1),
∇Gl
i
 > τ l+1
g
,
(7)
where µl
i is the position of the corresponding Gaussian Gl
i of
candidate cj, and vl follows the definition in Eq. (5).
Step 2: Parent-Child Verification. For a level-l candidate
cj determined in the previous step, ProGS check whether its
parent anchor apj at level l −1 exists, which satisfying
round
 xpj/vl−1

= ⌊round (x (cj) /vl) /2⌋,
(8)
while x (cj) refers to cj’s position. If no apj exists, a candidate
cpj = (x (cj) , l −1) will be queued, as in Fig. 3(ii).
Step 3: Anchor Spawn. Finally, queued candidates will be
instantiated as anchors if their target voxels (determined by
x (cj) and level) are unoccupied, strictly preserving octree
sparsity through conflict checking.
To ensure computational efficiency, this paper implements
the entire procedure through a fine-to-coarse traversal that pro-
cesses the finest level first. This approach minimizes redundant
computations when handling parent-child relationships. The
complete pseudo-code detailing this optimized workflow is
provided in Algo. 1.
To eliminate trivial anchors, ProGS tracks the accumulated
Gaussian opacity of each anchor over the past T iterations
and prunes all leaf nodes whose opacity is not higher than a
threshold τo, as marked in Fig. 3 (iii), and those anchors with
no children nodes will be pruned.
D. Mutual Information Enhancement
Under the octree structure, each low LoD is essentially an
incomplete representation of the entire scene, which inevitably
causes noticeable degradation in visual fidelity. Since all LoDs
ultimately pursue the same representation goal, a natural
question arises: Is it possible to enhance MI so that low LoDs
inherit more of the informative content available at higher
LoDs? The answer is affirmative, as prior work in point clouds
and images has repeatedly shown the effectiveness of MI
Algorithm 1 Growing operation of anchors.
Input :
Anchor set {a}, gradients of offsets {∇o}, τg, β.
1: for level l = L →1 do
2:
initialization candidate set Cl = ∅;
3:
Compute τ l
g := τg × 2βl, τ l+1
g
:= τg × 2β(l+1);
4:
for each anchor al
i at level l do
5:
for Gaussian index j = 1 →10 do
6:
∇Gl
j := ∇ox
i,j · ∇oy
i,j · ∇oz
i,j;
7:
if
∇Gl
j
 > τ l+1
g
and l < L then
8:
Cl+1 := Cl+1 ∪
 µl
i,j, l + 1

;
9:
else if
∇Gl
j
 > τ l
g then
10:
Cl := Cl ∪
 µl
i,j, l

;
11:
end if
12:
end for
13:
end for
14:
if l < L then
15:
for each anchor candidate cm ∈Cl+1 do
16:
if not ∃al
n : round

xn
vl

=
j
round

x(cm)
vl+1

/2
k
then
17:
Cl := Cl ∪(x (cm) , l);
18:
end if
19:
end for
20:
end if
21:
instantiate candidates set Cl+1 into {a};
22: end for
23: instantiate candidates set C1 into {a};
maximization for robust representations [46]–[48]. Building
on this insight, we introduce two complementary MI enhance-
ment mechanisms: an InfoNCE-based contrastive objective
that explicitly increases the MI between parent–child anchors
by treating them as positive pairs and unrelated anchors
as negatives, and a coarse-to-fine optimization that provides
cross-level supervision so lower-level anchors better approx-
imate the distributions and rendering behaviors of higher-
level ones. Together, these mechanisms promote information
sharing across levels, stabilize learning at early progressive
stages, and substantially improve ProGS’s rendering quality at
low bitrates by making lower-level anchors more predictive of
fine-grained content present in deeper LoDs.
InfoNCE Loss for Mutual Information. MI is a metric that
measures the amount of information shared between random
variables x ∼X and y ∼Y [49], which is defined as
I(X; Y ) = Ep(x,y)

log

p(x, y)
p(x) · p(y)

=
ZZ
p(x, y) log

p(x, y)
p(x) · p(y)

dxdy
.
(9)
In the context of ProGS, with higher MI between parent and
child nodes comes a stronger ability of low LoDs to represent
the entire scene. However, explicitly estimating the marginal
distributions p(x) and p(y) is hard. Therefore, this paper
introduces InfoNCE loss [44] to facilitate MI maximization
in a contrastive learning way. Note that the MI is modeled as

<!-- page 7 -->
7
0
20
40
60
Size (MB)
0.65
0.70
0.75
0.80
SSIM
(a) Mip-NeRF360 [52]
0
100
200
300
Size (MB)
0.75
0.80
0.85
0.90
(b) BungeeNeRF [53]
0
10
20
30
40
Size (MB)
0.6
0.7
0.8
Lee et al.  (CVPR'24)
Compressed3D (CVPR'24)
EAGLES (ECCV'24)
SOG (ECCV'24)
Compact3D (ECCV'24)
HAC (ECCV'24)
RDO-Gaussian (ECCV'24)
CompGS (ACMMM'24)
R-3DGS (I3D'24)
LightGaussian (NeurIPS'24)
Ours
(c) Tanks and Temples [54]
Fig. 4: RD curves. Notably, the solid lines are the RD curves of different bitrate versions obtained by changing different λe at
the same LoD, while ...
the........
dotted.......
lines are the RD curves of the same bitrate version at different LoDs.
the dot product of the attributes as
LNCE = −log
exp
 Ai · AT
pi/τNCE

PNneg
j=1 exp
 Ai · AT
j /τNCE
,
(10)
in which pi refers to the parent node of node i, with Nneg as
the number of the constructed negative samples, and τNCE as
the temperature factor to control the discrimination of LNCE
for negative samples [50].
Under the explicit supervision of the InfoNCE loss LNCE,
the attributes of the parent and child nodes are highly corre-
lated. Therefore, the context of each child anchor should be
modeled as related to its parent and itself. Following the design
of [13], in which the context of each anchor is modeled as a
hash feature queried from a hash-grid H, in this case, ProGS
models the context as the combination of the hash feature hi
queried by its position and the context feature hpi queried by
its parent pi’s position as
p(Ai, i, H) ∼p(Ai|hi, hpi) × p(H).
(11)
Specifically, for each root node i, with no parent node, the
corresponding hpi is simply set to 0.
Coarse-to-Fine Optimization. To further promote lower-level
anchors representing the entire scene and thereby increase MI
between different levels, ProGS replaces the original L3DGS
by a coarse-to-fine loss Lc2f to activate lower-level anchors in
a macro but implicit way:
Lc2f = 1
L
L
X
l=1
(1 −λSSIM) Ll
1 + λSSIM
 1 −Ll
SSIM

,
(12)
in which Ll
1 and Ll
SSIM represent the absolute error and the
structural similarity index (SSIM) [51] between ground truth
images and the results rendered by anchors with levels beneath
l, respectively. By bridging lower-level anchors to the entire
scene, this loss function can also promote the learning process
of coarse-grained anchors, driving them to represent the whole
scene independently.
E. Context-based Entropy Coding
To facilitate bit reduction via entropy coding such as
AE [16], ProGS explicitly drives the anchor attributes toward
low-entropy, context-predictable distributions. ProGS Follows
HAC [13], querying a mixed 3D–2D hash-grid H at an an-
chor’s spatial location along with its parent’s position to obtain
context features that condition both distribution estimation
and adaptive quantization. Empirically, the attribute channels
in ProGS exhibit approximately Gaussian statistics similar to
HAC; therefore, a same context modeling backbone can be
performed while tailoring it to the octree hierarchy. Concretely,
the context model in Eq. (11) remains aligned with [13]: the
trainable hash-grid H is binarized to {−1, +1} using straight-
through estimation (STE) [55], as in [56], so that its storage
cost can be accounted for explicitly. The bit consumption of
H is estimated by the Bernoulli cross-entropy
Lhash = M+(−log2 f+) + M−(−log2(1 −f+)) ,
(13)
where f+ is the empirical frequency of the symbol “+1”,
and M+/M−are the counts of “+1”/“−1” entries in H.
This encourages balanced yet compressible hash codes and
amortizes the header bits.
Given hash-derived context vectors hi for anchor i and
hpi for its parent, we model the per-dimension distribution
of the quantized attributes ˆ
Ai with a Gaussian prior whose
parameters are predicted by small MLPs:
ˆµi, ˆσi = MLPd(hi, hpi) ,
qi = Q0 ·
 1 + tanh
 MLPq(hi, hpi)

.
(14)
Here, MLPd outputs the mean and scale of the Gaussian
for each attribute channel, while MLPq predicts an adaptive
quantization step qi to balance precision and rate. Under
the common factorization assumption across channels, the
probability of a scalar symbol is given by integrating the
Gaussian over its quantization bin:
p( ˆ
Ai) =
Z
ˆ
Ai+ 1
2 qi
ˆ
Ai−1
2 qi
ϕi(x) dx
= Φi

ˆ
Ai + 1
2qi

−Φi

ˆ
Ai −1
2qi

,
(15)
where ϕi and Φi are the probability density function (PDF)
and cumulative distribution function (CDF) of N(ˆµi, ˆσi). AE
then encodes symbols using these probabilities, while training
minimizes the expected bit cost over anchors and channels:
Lentropy =
N
X
i=1
Df +Ds+Do
X
j=1
−log2 p

ˆ
Ai,j

,
(16)

<!-- page 8 -->
8
with N the number of anchors. Combined with quantization-
aware
training
and
parent–child
conditioning
through
(hi, hpi), this formulation yields differentiable, context-
adaptive priors that reduce symbol entropy and stabilize
bitrate across LoDs, while keeping the hash-grid overhead
measurable and controlled by Lhash.
F. Training and Coding
In the whole training pipeline, ProGS constructs the overall
loss by combining the supervisory and regularization terms
discussed above so as to jointly optimize fidelity, compactness,
and progressive consistency:
L = Lc2f + λvol Lvol + λNCE LNCE + λe Le,
(17)
with the entropy-related term normalized per encoded symbol
as
Le =
Lentropy + Lhash
N(Df + Ds + Do).
(18)
Here, Lc2f enforces cross-level supervision so that lower LoDs
learn to approximate higher-level renderings, Lvol penalizes
the total volume of all generated Gaussians to discourage
oversized primitives and promote sparsity, LNCE enhances
inter-level MI by contrasting parent–child anchors against
negatives, and Le jointly encourages low-entropy, context-
predictable attributes and a well-conditioned hash context.
The scalar weights λvol, λNCE, and λe control the trade-offs
among geometric compactness, MI promotion, and rate; in
particular, sweeping λe yields different operating points along
the rate–distortion curve. The denominator N
 Df +Ds+Do
normalizes by the number of symbols (anchors) and attribute
dimensionality to keep Le scale-invariant across scenes and
settings.
For practical encoding/decoding, we adopt a header–chunk
pipeline that aligns with progressive transmission. First, the
hash-grid H is encoded/decoded once via AE/AD together
with the parameters of the lightweight MLPs that perform
context-driven distribution estimation and quantization; this
header is cached and reused for all subsequent levels. After-
ward, decoding proceeds in a topologically valid order from
root to leaves: at each level, the context queried from H
conditions the probability models used by MLPd and MLPq
to decode (or encode) the anchor attributes. The coordinates
of first-level anchors are stored verbatim to seed the hier-
archy, whereas each subsequent-level anchor is represented
compactly by 23 bit, in which 20 bits record the index of
its unique parent in the previous level and 3 bits encode the
octant offset relative to its parent. Therefore, the absolute
positions are reconstructed deterministically from the voxel
grid. This design keeps the header amortized, enables per-
level independent decoding, and ensures that higher LoDs can
be appended incrementally without reprocessing earlier data,
which is essential for low-latency progressive rendering.
IV. EXPERIMENTS
A. Implementation Details
ProGS is implemented on top of the HAC framework [13]
using the PyTorch library [60], and all models are trained
on an Ubuntu 20.04 LTS server equipped with an NVIDIA
RTX 4090 GPU. Unless otherwise specified, the attribute
dimensions are fixed as Df = 32, Ds = 6, and Do = 10
for the anchor feature, scaling, and offsets, respectively. The
maximum number of LoDs is set to L = 5 with a base
depth lb = 11, which defines the voxelization granularity
for octree construction. For the anchor adjustment policy,
the gradient-based growing threshold uses hyperparameters
β = 0.01 and τg = 5×10−5, while the pruning threshold
on accumulated opacity is τo = 0.5; adjustment is triggered
every T = 100 iterations. For the InfoNCE objective, ProGS
constructs Nneg = 100 negatives per anchor and adopt a
temperature of τNCE = 0.03. To bound memory usage during
contrastive learning, at each iteration we uniformly sample 5%
of anchors to compute LNCE, which preserves diversity while
keeping the GPU footprint manageable.
Training
Started
Training
Terminated
0 1.6K
13K14K
20K
25K
40K
Iterations
Anchor Spawn Paused
Adding Uniform Noise
Applying            ,         and 
ℒ3DGS  ℒvol  
ℒNCE  
Applying            ,         and 
ℒ3DGS  ℒvol  
ℒNCE  
Adding  ℒe 
Adding  ℒe 
Replacing             by
ℒ3DGS  
ℒc2f 
Replacing             by
ℒ3DGS  
ℒc2f 
Fig. 5: The training process of ProGS.
The training schedule is visualized in Fig. 5 and follows a
staged, stabilization-first strategy. Anchor adjustment is acti-
vated at 1,600 iterations, at which point ProGS also replaces
the single-level photometric loss L3DGS with the multi-level
coarse-to-fine supervision Lc2f to explicitly involve lower
LoDs in optimization. Starting from 13,000 iterations, ProGS
adopts quantization-aware training by injecting uniform noise
into anchor attributes A following [13]; to avoid training
instability during this transition, anchor spawning is briefly
paused until 14,000 iterations so that the Gaussians adapt to
the perturbed attributes. From 20,000 iterations onward, the
entropy regularizer Le is added to the total loss to encourage
low-entropy attribute distributions compatible with arithmetic
coding; for memory safety, the computation of Le also uses a
5% random anchor subset per iteration. This phased schedule
consistently stabilizes optimization, leading to smoother LoD
transitions and improved rate–distortion behavior.
For the coding pipeline, we follow [13] and use a mixed
3D–2D structured binary hash-grid as the context source for
AE/AD. The 3D branch contains 12 levels with resolutions
from 16 to 512, and the 2D branch contains 4 levels with
resolutions from 128 to 1024; both branches use embedding
dimension 4. Quantization factors Q0 are set to 1 for f, 0.001
for s, and 0.2 for o to balance precision across attribute types.
In the overall loss Eq. (17), it sets λSSIM = 0.2 and λvol =
0.01 following Scaffold-GS [11], and uses λNCE = 0.005. By
sweeping the entropy weight λe from 5×10−4 to 4×10−3, the
experiments obtain five bitrate variants per scene.
B. Experiment Evaluation
Baselines. Here compares the proposed method with existing
primitive-based and anchor-based 3DGS compression meth-
ods. Specifically, [7]–[10], [17], [57] mainly used parameter

<!-- page 9 -->
9
Mip-NeRF360 [52]
BungeeNeRF [53]
Tanks and Temples [54]
Method/Metric
PSNR (dB)↑
SSIM↑
LPIPS↓
Size (MB)↓
PSNR↑
SSIM↑
LPIPS↓
Size↓
PSNR↑
SSIM↑
LPIPS↓
Size↓
3DGS [1]
27.49
0.970
0.031
68.46
24.87
0.841
0.205
1616
23.69
0.844
0.178
431.0
Scaffold-GS [11]
27.50
0.806
0.252
253.9
26.62
0.865
0.241
183.0
23.96
0.853
0.177
86.50
Lee et al. [7]
27.08
0.798
0.238
48.80
23.36
0.788
0.251
82.60
23.32
0.831
0.201
39.43
Compressed3D [57]
26.98
0.801
0.238
28.80
24.13
0.802
0.245
55.79
23.32
0.832
0.253
25.30
EAGLES [58]
27.23
0.810
0.240
54.00
25.24
0.843
0.221
117.1
23.63
0.850
0.840
29.00
SOG [59]
27.08
0.799
0.230
38.40
24.36
0.807
0.216
50.73
23.56
0.837
0.186
21.70
Compact3D [10]
27.16
0.808
0.228
50.30
24.63
0.823
0.239
104.3
23.47
0.840
0.188
27.97
HAC-highrate [13]
27.77
0.811
0.230
21.87
27.08
0.872
0.209
29.72
24.40
0.853
0.177
11.24
HAC-lowrate [13]
27.53
0.807
0.238
15.26
26.48
0.845
0.250
18.49
24.04
0.846
0.187
8.10
RDO-Gaussian [9]
27.05
0.802
0.239
23.46
26.54
0.907
0.086
25.41
22.09
0.755
0.318
12.02
CompGS [19]
27.26
0.800
0.240
16.50
26.38
0.870
0.177
15.74
23.70
0.840
0.210
9.60
R-3DGS [17]
27.10
0.809
0.226
29.00
26.41
0.888
0.114
127.6
23.57
0.840
0.188
14.00
LightGaussian [8]
27.28
0.805
0.243
42.00
24.52
0.825
0.255
87.28
23.11
0.817
0.231
22.00
Ours-HR (LoD 5)
27.66
0.817
0.210
29.81
28.11
0.923
0.088
47.70
24.47
0.865
0.151
19.43
Ours-HR (LoD 1)
21.88
0.642
0.400
9.85
23.95
0.824
0.178
26.87
16.18
0.556
0.503
4.11
Ours-LR (LoD 5)
27.37
0.811
0.220
20.61
27.75
0.916
0.098
34.67
24.35
0.858
0.162
12.88
Ours-LR (LoD 1)
21.99
0.644
0.403
7.34
24.15
0.828
0.177
20.49
16.45
0.568
0.500
2.99
TABLE I: Quantitative results on Mip-NeRF360 [52], BungeeNeRF [53] and Tanks and Temples [54] datasets. The table
highlights the best and second-best results. Here, “Ours-HR” and “Ours-LR” are the highest and lowest bitrate versions of
ProGS by setting λe as 5e −4 and 4e −3, respectively.
pruning techniques and codebook-based strategies to compress
Gaussian primitives. [9], [58] both utilized entropy constraint
methods while [59] adopted a smooth 2D mesh technique
to compress 3D Gaussians. [19] utilized a hybrid anchor-
coupled primitive structure with residual embeddings and rate-
distortion optimization to achieve a compact 3DGS represen-
tation. In contrast, [11] utilized anchors to distribute local
3D Gaussians, based on which [13] achieved compression by
structuring the relationships between anchors through hash-
grid context.
Datasets. Here evaluates the proposed method across three
real-world datasets, including Mip-NeRF360 [52], BungeeN-
eRF [53], and Tanks and Temples [54] to showcase the
effectiveness and versatility of the proposed approach com-
prehensively. Specifically, the experiments evaluate the en-
tire 9 scenes from Mip-NeRF360 [52] dataset and 6 scenes
from BungeeNeRF dataset following the settings of Scaffold-
GS [11].
Metrics. Here assesses the performance of each method
using several metrics, including peak signal-to-noise ratio
(PSNR), SSIM, and learned perceptual image patch sim-
ilarity (LPIPS) [61] between the synthesized novel views
to the ground truth images. Additionally, the file sizes of
the resulting compact representations are also examined. To
further demonstrate the performance of these methods, 4
presents rate-distortion (RD) curves [62]. For generating these
RD curves, the following experimental results are used: the
“Small” and “Fast” variants of EAGLES [58], the variant
of SOG [59] with SH removed, and the “Low” and “High”
variants of R-3DGS [17]. For HAC [13], RDO-Gaussian [9],
and CompGS [19], the RD curves are constructed based on
the parameter settings provided in their respective papers.
Results. Quantitative results are summarized in Tab. I and the
RD curves are plotted in Fig. 4, with per-scene breakdowns
provided in the supplementary material for completeness.
2.5
5.0
7.5
10.0
12.5
15.0
17.5
20.0
Component Size (MB)
LoD 5
LoD 4
LoD 3
LoD 2
LoD 1
Header
hash-grid
MLP
anchor
feature
scaling
offset
20
40
60
80
Percentage
LoD %
Comp %
Header
LoD 1
LoD 2
LoD 3
LoD 4 LoD 5
Fig. 6: The visualization of the encoded bitstream, evaluated
on dataset BungeeNeRF [53], under λe = 4e −3. Top:
the
storage cost of each component at different LoDs. Bottom:
the percentage of each component and each LoD in the
complete bitstream.
Across the evaluated datasets and metrics, ProGS consistently
delivers strong compression efficiency while maintaining high
rendering fidelity. Notably, at the coarsest operating point
(LoD 1), our codec achieves a size reduction exceeding 144×
relative to vanilla 3DGS [1], yet preserves usable visual qual-
ity. As finer levels are appended, the RD curves exhibit steady,
monotonic gains, and the full progressive representation attains
comparable—or in many cases improved—quality compared
to current state-of-the-art methods at similar bitrates. These
results confirm that ProGS provides a favorable bitrate–quality
trade-off over a wide operating range while enabling progres-
sive, stream-friendly decoding.
C. Bitstream Analysis
Fig. 6 breaks down the storage cost of each component in
the encoded stream and clarifies how bits are allocated across

<!-- page 10 -->
10
Chunk
Header
LoD 1
LoD 2
LoD 3
LoD 4
LoD 5
Enc (s)
Dec (s)
Enc
Dec
FPS
Enc
Dec
FPS
Enc
Dec
FPS
Enc
Dec
FPS
Enc
Dec
FPS
train
0.004
0.378
0.496
1.093
195
0.604
1.408
204
0.992
2.121
148
1.644
3.627
136
1.843
3.787
154
truck
0.005
0.340
1.432
3.159
227
1.268
2.826
216
1.194
2.506
194
1.344
2.520
174
1.414
2.676
188
Avg.
0.004
0.359
0.964
2.126
211
0.936
2.117
210
1.093
2.313
171
1.494
3.074
155
1.628
3.232
171
TABLE II: Coding time and framerates on dataset Tanks and Temples [54], with λe = 4e −3. Facilitating progressive coding,
ProGS only needs to encode/decode each level once. Therefore, the time listed is the time cost of encoding/decoding each
level only.
progressive levels. The “Header” comprises the hash-grid H
and lightweight MLP parameters, and is transmitted once and
amortized over all LoDs; after it is parsed, subsequent chunks
mostly contain structural metadata and entropy-coded anchor
attributes. Because ProGS enforces strict octree properties,
the first decoded level (LoD 1) carries a relatively larger
proportion of structural bits to ensure relatively complete
scene representation; as the hierarchy grows, the fraction of
structural bits decreases and the attribute portion becomes
dominant. This shift reflects the intended rate allocation:
coarse layers invest in scaffolding the hierarchy for reliable
progressive starts, while deeper layers spend bits on fine-
grained appearance and geometry details that most affect
fidelity. The context-based entropy model further reduces
redundancy by correlating attributes with their spatial context,
lowering symbol entropy and stabilizing bitrate. Overall, the
analysis shows that ProGS’s bitstream is front-loaded with
reusable context information, followed by level-wise chunks
that are independently decodable, yielding efficient startup,
predictable growth in quality, and principled storage usage at
each stage of progressive transmission.
D. Qualitative Visualization
To visually evaluate the performance of ProGS, qualitative
results for different scenes at varying LoDs are provided, as
shown in Fig. 7. The rendered images demonstrate how ProGS
progressively refines the 3D scene representation, starting
from coarse-grained, low-quality outputs at lower LoDs and
progressively improving the visual fidelity as higher LoDs
are incorporated. At LoD 1, the scene appears significantly
blurred, with noticeable loss of finer details. However, the
LoD increases, more detailed and accurate representations
emerge, with the visual quality improving progressively. This
gradual refinement is particularly evident in complex areas
with high-frequency features, such as intricate textures and
object boundaries. ProGS effectively reduces the visual degra-
dation typically seen in lower LoDs, thanks to its anchor
adjustment strategy and the incorporation of MI enhancement
mechanisms. These mechanisms ensure that even lower-level
anchors contribute meaningfully to the final scene repre-
sentation. Notably, the progression from LoD 1 to LoD 5
highlights the ability of ProGS to maintain high fidelity and
realistic rendering quality, even with a substantial reduction
in data. This emphasizes ProGS’s effectiveness in achieving
compact and scalable 3D scene representations while retaining
high-quality visual results, which are essential for real-time
applications.
Method/Metric
PSNR (dB)↑
SSIM↑
LPIPS↓
Baseline
26.74
0.883
0.151
Baseline w/ c2f
26.50
0.875
0.163
ProGS raw
27.36
0.912
0.098
ProGS w/o MI
27.77
0.916
0.097
TABLE III: Ablation studies for octree structure and adaptive
anchor adjustment strategy, with λe = 4e −3. The proposed
approaches can provide more refined representation capabili-
ties.
E. Ablation Study
The ablation studies are conducted in two aspects to demon-
strate the effectiveness of the proposed approaches.
Efficacy of Octree Structure and Anchor Adjustment
Strategy. The confucted ablation studies evaluate the efficacy
of the octree structure and the anchor adjustment strategy by
designing controlled variants:
1) “Baseline”: replacing the strict, voxel-aligned octree
with the loosely organized Octree-GS [43]1 and remov-
ing the coarse-to-fine optimization loss Lc2f. In addi-
tion, the MI enhancement is disabled, which eliminates
explicit parent–child dependency during training.
2) “ProGS raw”: to isolate the contribution of the structure
itself, this variant is constructed by starting from the pro-
posed framework but turning off both the MI component
and Lc2f for a fair comparison.
3) “Baseline w/ c2f”: adding only Lc2f on top of Baseline.
4) “ProGS w/o MI”: adding only Lc2f on top of “ProGS
raw”.
As summarized in Tab. III on BungeeNeRF [53], enforcing
a strict octree consistently yields finer spatial organization
and more stable LoD transitions than the position-decoupled
alternative. Moreover, even without MI, introducing Lc2f en-
courages low-level anchors to better approximate full-scene
content, narrowing the gap to the full model. Overall, the ab-
lations indicate that the strict octree layout is the key to refined
representation capacity and robust progressive rendering, and
the anchor adjustment strategy and Lc2f are complementary,
jointly improving compactness and fidelity over [43].
Efficacy of Mutual Information Enhancement Mecha-
nisms. To more clearly assess the contribution of the proposed
MI enhancement in Sec. III-D, here constructs three controlled
1Though named after octree, Octree-GS [43] does not strictly guarantee
that anchors follow octree properties and is essentially a collection of loosely
organized multi-granularity anchors.

<!-- page 11 -->
11
train
pompidou
rome
Groud Truth
LoD 1
LoD 2
LoD 3
LoD 4
LoD 5
22.99 dB
21.12 MB
25.01 dB
25.59 MB
25.94 dB
30.02 MB
26.59 dB
33.65 MB
26.94 dB
36.67 MB
23.03 dB
21.06 MB
24.70 dB
25.23 MB
26.27 dB
29.32 MB
27.23 dB
33.15 MB
27.94 dB
36.67 MB
15.19 dB
1.63 MB
16.88 dB
2.85 MB
18.95 dB
4.82 MB
21.52 dB
7.97 MB
22.75 dB
11.13 MB
truck
stump
17.72 dB
4.18 MB
19.14 dB
9.74 MB
19.89 dB
7.30 MB
20.48 dB
15.53 MB
22.68 dB
19.34 MB
22.44 dB
9.71 MB
25.01 dB
12.11 MB
25.28 dB
22.75 MB
26.67 dB
25.22 MB
25.95 dB
14.45 MB
Fig. 7: Rendering results for different senes at different LoDs.
20
30
40
50
60
70
Size (MB)
0.8
0.9
SSIM
ProGS raw
ProGS w/o MI
ProGS w/o c2f
Ours
Fig. 8: RD curves for ablation study of the three variants
and full ProGS. The solid lines are obtained by varying λe,
while ....
the .......
dotted.......
lines demonstrate the performance at different
LoDs.
variants by selectively disabling components in ProGS so that
each factor can be isolated and contrasted against the full
model.
1) “ProGS raw”: both the direct MI use and the coarse-to-
fine supervision Lc2f are turned off, leaving only the
strict octree with anchor adjustment and the coding
pipeline; this variant serves as a structural baseline
without inter-level guidance.
2) “ProGS w/o MI”: the direct MI term is removed while
keeping Lc2f, isolating the effect of the coarse-to-fine
activation of lower-level anchors on progressive fidelity.
3) “ProGS w/o c2f”: the coarse-to-fine loss is removed
while keeping direct MI, isolating the benefit of explic-
itly strengthening parent–child dependency in the octree.
All variants are trained alongside full ProGS on BungeeN-
eRF [53] under identical settings, and the corresponding RD
curves are reported in Fig. 8. Compared with “ProGS raw”,
enabling either MI or Lc2f consistently improves quality at a
fixed bitrate or reduces bitrate at a fixed quality. In partic-
Ground Truth
ProGS
ProGS w/o MI
ProGS w/o c2f
1.0
0.0
Fig. 9: The visualization of MI↑(normalized and exponen-
tiated). Both proposed approaches contribute to MI enhance-
ment. The rate-distortion factor λe is 4e −3. The selected
scenes are amsterdam and rome from BungeeNeRF [53]
dataset.
ular, direct MI most noticeably benefits mid-to-high bitrates,
whereas Lc2f yields larger gains at low bitrates by driving low-
level anchors to approximate full-scene content, improving
the initial progressive stages. When combined in full ProGS,
the two mechanisms are complementary: the RD curve shifts
further upward/left with smoother LoD transitions, fewer low-
frequency washouts and edge artifacts at early layers, and more
faithful high-frequency recovery as bitrate grows.
To further substantiate these observations, Fig. 9 visualizes
the inter-level MI for “ProGS w/o MI”, “ProGS w/o c2f”, and
the full model. Variant “ProGS w/o MI” exhibits weak and
spatially fragmented parent–child dependence, while “ProGS
w/o c2f” shows MI concentrated only in high-texture regions.
By contrast, full ProGS produces higher and more uniformly
distributed MI across levels, indicating stronger hierarchical
consistency; this aligns with the quantitative RD improvements
and explains the enhanced robustness of progressive decoding
at low bitrates.

<!-- page 12 -->
12
F. Execution Time
With the additional GPU computation required to strictly
maintain the octree properties and the multi-level rendering
induced by Lc2f, the end-to-end training of ProGS is approx-
imately 3× longer than Scaffold-GS [11]. In contrast, the
coding path remains lightweight thanks to the LoD-structured
design: each level is encoded/decoded only once and can be
streamed or cached independently. The per-level encoding and
decoding time measured on GPU is summarized in Tab. II;
here, the “Header” denotes the one-off payload containing the
hash-grid H and the parameters of the small MLPs used for
distribution estimation, quantization, and Gaussian generation,
which is parsed a single time and reused for all subsequent
levels. After the header is loaded, decoding proceeds in a
root-to-leaf order with entropy decoding of anchor attributes
and on-the-fly Gaussian generation, so the marginal latency
of adding an LoD is modest and largely proportional to the
number of active anchors at that depth. Rendering throughput
(FPS) for each LoD is also reported in Tab. II, showing real-
time rates across all levels: coarse LoDs achieve higher FPS
due to fewer anchors, while deeper LoDs introduce a gradual
reduction in speed yet remain in the real-time regime, aided by
anchor pruning and visibility/opacity culling. Overall, despite
the heavier training stage, ProGS delivers short per-level
coding time and consistent real-time rendering; its progressive
chunking enables low-latency scene startup and incremental
refinement without re-encoding, which is crucial for interactive
viewing and adaptive streaming.
V. CONCLUSION
This paper presents ProGS, a novel codec designed to
facilitate progressive coding for 3DGS. ProGS addresses
the challenges associated with efficiently compressing large
3D scene representations while preserving rendering quality,
making it suitable for real-time streaming applications with
varying network conditions. By leveraging hierarchical oc-
tree structures and introducing adaptive anchor adjustments,
ProGS enables scalable and efficient compression of 3DGS
data. Moreover, this work also introduces MI enhancement
techniques, such as the InfoNCE loss and coarse-to-fine opti-
mization, to improve the visual quality of lower-level anchors
and ensure high-fidelity scene representations even at lower
levels of detail. Extensive experiments demonstrate that ProGS
achieves superior performance, with up to 45× reduction in
storage requirements and over 10% improvement in visual
quality compared to existing SOTA methods. ProGS is the
first progressive coding framework for 3DGS and offers a
promising solution for real-time 3D scene rendering, providing
a balance between efficient data compression and high-quality
rendering. Future work could explore further optimization
strategies for ProGS, including real-time adaptive encoding
and decoding mechanisms that can dynamically adjust based
on available resources and bandwidth, further enhancing its
applicability in diverse scenarios.
REFERENCES
[1] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions
on
Graphics,
vol.
42,
no.
4,
July
2023.
[Online].
Available:
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
[2] B. Fei, J. Xu, R. Zhang, Q. Zhou, W. Yang, and Y. He, “3d gaussian
splatting as new era: A survey,” IEEE Transactions on Visualization and
Computer Graphics, 2024.
[3] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[4] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,”
in Proceedings of the IEEE conference on computer vision and pattern
recognition, 2016, pp. 4104–4113.
[5] C. Lassner and M. Zollhofer, “Pulsar: Efficient sphere-based neural
rendering,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2021, pp. 1440–1449.
[6] G. Chen and W. Wang, “A survey on 3d gaussian splatting,” 2024.
[Online]. Available: https://arxiv.org/abs/2401.03890
[7] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp.
21 719–21 728.
[8] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, Z. Wang et al., “Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+
fps,” Advances in neural information processing systems, vol. 37, pp.
140 138–140 158, 2024.
[9] H. Wang, H. Zhu, T. He, R. Feng, J. Deng, J. Bian, and Z. Chen, “End-to-
end rate-distortion optimized 3d gaussian representation,” in European
Conference on Computer Vision, 2024.
[10] K. Navaneet, K. P. Meibodi, S. A. Koohpayegani, and H. Pirsiavash,
“Compgs: smaller and faster gaussian splatting with vector quantization,”
in European Conference on Computer Vision. Springer, 2024, pp. 330–
349.
[11] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai,
“Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 654–20 664.
[12] T. Müller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Trans. Graph.,
vol. 41, no. 4, pp. 102:1–102:15, Jul. 2022. [Online]. Available:
https://doi.org/10.1145/3528223.3530127
[13] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Hac: Hash-grid
assisted context for 3d gaussian splatting compression,” in European
Conference on Computer Vision.
Springer, 2024, pp. 422–438.
[14] R. Schnabel and R. Klein, “Octree-based point-cloud compression.”
PBG@ SIGGRAPH, vol. 3, pp. 111–121, 2006.
[15] C. E. Shannon, “A mathematical theory of communication,” The Bell
system technical journal, vol. 27, no. 3, pp. 379–423, 1948.
[16] J. Rissanen and G. G. Langdon, “Arithmetic coding,” IBM Journal of
research and development, vol. 23, no. 2, pp. 149–162, 1979.
[17] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and G. Drettakis,
“Reducing the memory footprint of 3d gaussian splatting,” Proceedings
of the ACM on Computer Graphics and Interactive Techniques, vol. 7,
no. 1, pp. 1–17, 2024.
[18] S. Shin, J. Park, and S. Cho, “Locality-aware gaussian compression
for fast and high-quality rendering,” in The Thirteenth International
Conference on Learning Representations, 2025. [Online]. Available:
https://openreview.net/forum?id=dHYwfV2KeP
[19] X. Liu, X. Wu, P. Zhang, S. Wang, Z. Li, and S. Kwong, “Compgs:
Efficient 3d scene representation via compressed gaussian splatting,” in
Proceedings of the 32nd ACM International Conference on Multimedia,
2024, pp. 2936–2944.
[20] S. Xie, W. Zhang, C. Tang, Y. Bai, R. Lu, S. Ge, and Z. Wang, “Mesongs:
Post-training compression of 3d gaussians via efficient attribute trans-
formation,” in European Conference on Computer Vision.
Springer,
2025, pp. 434–452.
[21] R. L. de Queiroz and P. A. Chou, “Compression of 3d point clouds
using a region-adaptive hierarchical transform,” IEEE Transactions on
Image Processing, vol. 25, no. 8, pp. 3947–3956, 2016.
[22] Y. Chen, Q. Wu, M. Li, W. Lin, M. Harandi, and J. Cai, “Fast
feedforward 3d gaussian splatting compression,” in The Thirteenth
International Conference on Learning Representations, 2025. [Online].
Available: https://openreview.net/forum?id=DCandSZ2F1
[23] G. K. Wallace, “The jpeg still picture compression standard,” IEEE
transactions on consumer electronics, vol. 38, no. 1, pp. xviii–xxxiv,
1992.

<!-- page 13 -->
13
[24] A. Skodras, C. Christopoulos, and T. Ebrahimi, “The jpeg 2000 still
image compression standard,” IEEE Signal processing magazine, vol. 18,
no. 5, pp. 36–58, 2001.
[25] H. Choi and I. V. Baji´c, “Scalable image coding for humans and
machines,” IEEE Transactions on Image Processing, vol. 31, pp. 2739–
2754, 2022.
[26] T. Stutz and A. Uhl, “A survey of h. 264 avc/svc encryption,” IEEE
Transactions on circuits and systems for video technology, vol. 22, no. 3,
pp. 325–339, 2011.
[27] J. M. Boyce, Y. Ye, J. Chen, and A. K. Ramasubramonian, “Overview of
shvc: Scalable extensions of the high efficiency video coding standard,”
IEEE Transactions on Circuits and Systems for Video Technology,
vol. 26, no. 1, pp. 20–34, 2015.
[28] K. Gregor, F. Besse, D. Jimenez Rezende, I. Danihelka, and D. Wierstra,
“Towards conceptual compression,” Advances in neural information
processing systems, vol. 29, 2016.
[29] N. Johnston, D. Vincent, D. Minnen, M. Covell, S. Singh, T. Chinen,
S. J. Hwang, J. Shor, and G. Toderici, “Improved lossy image compres-
sion with priming and spatially adaptive bit rates for recurrent networks,”
in Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 4385–4393.
[30] G. Toderici, S. M. O’Malley, S. J. Hwang, D. Vincent, D. Minnen,
S. Baluja, M. Covell, and R. Sukthankar, “Variable rate image compres-
sion with recurrent neural networks,” in ICLR (Poster), 2016.
[31] G. Toderici, D. Vincent, N. Johnston, S. Jin Hwang, D. Minnen, J. Shor,
and M. Covell, “Full resolution image compression with recurrent neural
networks,” in Proceedings of the IEEE conference on Computer Vision
and Pattern Recognition, 2017, pp. 5306–5314.
[32] C. Cai, L. Chen, X. Zhang, G. Lu, and Z. Gao, “A novel deep progressive
image compression framework,” in 2019 Picture Coding Symposium
(PCS).
IEEE, 2019, pp. 1–5.
[33] B. Chen and J. Zhang, “Content-aware scalable deep compressed sens-
ing,” IEEE Transactions on Image Processing, vol. 31, pp. 5412–5426,
2022.
[34] J.-H. Lee, S. Jeon, K. P. Choi, Y. Park, and C.-S. Kim, “Dpict: Deep
progressive image compression using trit-planes,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2022,
pp. 16 113–16 122.
[35] S. Jeon, K. P. Choi, Y. Park, and C.-S. Kim, “Context-based trit-
plane coding for progressive image compression,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 14 348–14 357.
[36] Q. Mao, C. Wang, M. Wang, S. Wang, R. Chen, L. Jin, and S. Ma,
“Scalable face image coding via stylegan prior: Toward compression
for human-machine collaborative vision,” IEEE Transactions on Image
Processing, vol. 33, pp. 408–422, 2024.
[37] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley,
S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,”
Communications of the ACM, vol. 63, no. 11, pp. 139–144, 2020.
[38] M. Dasari, K. Kahatapitiya, S. R. Das, A. Balasubramanian, and
D. Samaras, “Swift: Adaptive video streaming with layered neural
codecs,” in 19th USENIX Symposium on Networked Systems Design and
Implementation (NSDI 22), 2022, pp. 103–118.
[39] H. Lin, B. Chen, Z. Zhang, J. Lin, X. Wang, and T. Zhao, “Deepsvc:
Deep scalable video coding for both machine and human vision,” in
Proceedings of the 31st ACM International Conference on Multimedia,
2023, pp. 9205–9214.
[40] I. J. S. W. 7, “G-pcc 2nd edition codec description,” ISO/IEC JTC 1/SC
29/WG 7, 2024, approved, Serial Number: 23290, Available at: https:
//isotc.iso.org/livelink/livelink/open/jtc1sc29wg7.
[41] J. Zhang, J. Wang, D. Ding, and Z. Ma, “Scalable point cloud attribute
compression,” IEEE Transactions on Multimedia, 2023.
[42] X. Mao, H. Yuan, T. Guo, S. Jiang, R. Hamzaoui, and S. Kwong,
“Spac: Sampling-based progressive attribute compression for dense point
clouds,” IEEE Transactions on Image Processing, 2025.
[43] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, “Octree-gs: To-
wards consistent real-time rendering with lod-structured 3d gaussians,”
IEEE Transactions on Pattern Analysis and Machine Intelligence, pp.
1–15, 2025.
[44] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with
contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.
[45] C. Müller, Spherical harmonics.
Springer, 2006, vol. 17.
[46] N. Chen, L. Chu, H. Pan, Y. Lu, and W. Wang, “Self-supervised image
representation learning with geometric set consistency,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, 2022, pp. 19 292–19 302.
[47] A. Sanghi, “Info3d: Representation learning on 3d objects using mu-
tual information maximization and contrastive learning,” in Computer
Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August
23–28, 2020, Proceedings, Part XXIX 16. Springer, 2020, pp. 626–642.
[48] H. Chen, S. Luo, X. Gao, and W. Hu, “Unsupervised learning of
geometric sampling invariant representations for 3d point clouds,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 893–903.
[49] J. Kreer, “A question of terminology,” IRE Transactions on Information
Theory, vol. 3, no. 3, pp. 208–208, 1957.
[50] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast
for unsupervised visual representation learning,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, 2020,
pp. 9729–9738.
[51] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: from error visibility to structural similarity,” IEEE
transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
[52] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman,
“Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern
recognition, 2022, pp. 5470–5479.
[53] Y. Xiangli, L. Xu, X. Pan, N. Zhao, A. Rao, C. Theobalt, B. Dai, and
D. Lin, “Bungeenerf: Progressive neural radiance field for extreme multi-
scale scene rendering,” in European conference on computer vision.
Springer, 2022, pp. 106–122.
[54] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples:
Benchmarking large-scale scene reconstruction,” ACM Transactions on
Graphics (ToG), vol. 36, no. 4, pp. 1–13, 2017.
[55] S. Shin and J. Park, “Binary radiance fields,” Advances in neural
information processing systems, vol. 36, 2024.
[56] Y. Chen, Q. Wu, M. Harandi, and J. Cai, “How far can we compress
instant-ngp-based nerf?” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 20 321–20 330.
[57] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed 3d
gaussian splatting for accelerated novel view synthesis,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, 2024, pp. 10 349–10 358.
[58] S. Girish, K. Gupta, and A. Shrivastava, “Eagles: efficient accelerated
3d gaussians with lightweight encodings,” in European Conference on
Computer Vision.
Springer, 2024, pp. 54–71.
[59] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, “Compact 3d
scene representation via self-organizing gaussian grids,” in European
Conference on Computer Vision.
Springer, 2024, pp. 18–34.
[60] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,
T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An
imperative style, high-performance deep learning library,” Advances in
neural information processing systems, vol. 32, 2019.
[61] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The
unreasonable effectiveness of deep features as a perceptual metric,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 586–595.
[62] G. Bjontegaard, “Calculation of average psnr differences between rd-
curves,” ITU-T SG16 Q, vol. 6, 2001.
Zhiye Tang received the B.E. and M.S. degrees in
the College of Mechatronics and Control Engineer-
ing and the College of Computer Science and Soft-
ware Engineering, Shenzhen University, Shenzhen,
China in 2022 and 2025, respectively. His research
interests include compact representation and efficient
transmission for 3D Gaussian Splatting. He is cur-
rently an image algorithm engineer at vivo Mobile
Communication Co., Ltd..

<!-- page 14 -->
14
Lingzhuo Liu received the B.E. degree in the Col-
lege of Computer Science and Software Engineering
from Shenzhen University, Shenzhen, China in 2023.
She is currently pursuing the M.S. degree at the Col-
lege of Computer Science and Software Engineering
from Shenzhen University, Shenzhen, China. Her
research interests include compact representation of
3D Gaussian Splatting with semantic features.
Shengjie Jiao received the B.E. degree in the Col-
lege of Computer Science and Technology from
Wuyi University, Jiangmen, China in 2023. He is
currently pursuing the M.S. degree at Shenzhen
University. His research interests include point cloud
compression.
Qiudan Zhang received the B.E. and M.S. degrees
in the College of Computer Science and Software
Engineering from Shenzhen University, China in
2015 and 2018, respectively. She received her Ph.D.
degree from the Department of Computer Science,
City University of Hong Kong, China (Hong Kong
SAR) in 2021. She is currently an Assistant Profes-
sor in the College of Computer Science and Soft-
ware Engineering, Shenzhen University, China. Her
research interests include computer vision, visual
attention, 3D vision and deep learning.
Junhui Hou (Senior member, IEEE) is an Asso-
ciate Professor with the Department of Computer
Science, City University of Hong Kong. He holds a
B.Eng. degree in information engineering (Talented
Students Program) from the South China Univer-
sity of Technology, Guangzhou, China (2009), an
M.Eng. degree in signal and information processing
from Northwestern Polytechnical University, Xi’an,
China (2012), and a Ph.D. degree from the School
of Electrical and Electronic Engineering, Nanyang
Technological University, Singapore (2016). His re-
search interests are multi-dimensional visual computing. Dr. Hou received the
Early Career Award (3/381) from the Hong Kong Research Grants Council in
2018 and the NSFC Excellent Young Scientists Fund in 2024. He has served
or is serving as an Associate Editor for IEEE Transactions on Visualization
and Computer Graphics, IEEE Transactions on Image Processing, IEEE
Transactions on Multimedia, and IEEE Transactions on Circuits and Systems
for Video Technology.
You Yang (M’10-SM’16) received the Ph.D. degree
in computer science from the Institute of Computing
Technology, Chinese Academy of Sciences, Beijing,
China, in 2009. He worked as a Postdoctoral Fellow
with the Automation Department, Tsinghua Univer-
sity, from 2009 to 2011. He was a Senior Research
Scientist in Sumavision Research from 2011 to 2013.
He currently heads the Department of Information
Engineering, Huazhong University of Science and
Technology, Wuhan, China. He has authored and
co-authored more than 60 technical papers and au-
thorized 17 patents. His research interests include three-dimensional (3D)
vision system and its applications, including the multi-view imaging system,
3D/VR/AR content processing and broadcasting, human-machine interaction
techniques, and interactive visual applications.
Xu Wang (M’15) received the B.S. degree from
South China Normal University, Guangzhou, China,
in 2007, and M.S. degree from Ningbo University,
Ningbo, China, in 2010. He received his Ph.D.
degree from the Department of Computer Science,
City University of Hong Kong in 2014. In 2015,
he joined the College of Computer Science and
Software Engineering, Shenzhen University, where
he is currently an Associate Professor. His research
interests are video coding and 3D vision.
