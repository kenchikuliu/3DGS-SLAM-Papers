<!-- page 1 -->
AG2aussian: Anchor-Graph Structured Gaussian Splatting for Instance-Level
3D Scene Understanding and Editing
Zhaonan Wang
Manyi Li†
Changhe Tu
Shandong University
dyllanelliia wzn@163.com
manyili@sdu.edu.cn
chtu@sdu.edu.cn
Interactive Click Query
Open-Vocabulary Text Query
Object Removal Editing
Physical Simulation
OpenGaussian
𝐀𝐆𝟐𝐚𝐮𝐬𝐬𝐢𝐚𝐧(Ours)
Feature Map
Selection
Remaining
Feature Map
Selection
Remaining
Stuffed bear
Three cookies
Coffee mug
Figure 1. We propose AG2aussian, an anchor-graph structured Gaussian splatting for instance-level 3D scene understanding and editing
tasks. Compared to existing works that attach semantic features to a collection of free Gaussians, we construct an anchor-graph structure
to organize the semantic features and regulate the associated Gaussians, resulting in a smooth feature distribution and a clean and accurate
instance-level Gaussian selection (top row). Our approach benefits a series of applications, including the interactive click query, open-
vocabulary text query, object removal editing, and physical simulation (bottom row).
Abstract
3D Gaussian Splatting (3DGS) has witnessed exponential
adoption across diverse applications, driving a critical need
for semantic-aware 3D Gaussian representations to enable
scene understanding and editing tasks. Existing approaches
typically attach semantic features to a collection of free
Gaussians and distill the features via differentiable render-
ing, leading to noisy segmentation and a messy selection
of Gaussians. In this paper, we introduce AG2aussian, a
novel framework that leverages an anchor-graph structure
to organize semantic features and regulate Gaussian prim-
itives. Our anchor-graph structure not only promotes com-
pact and instance-aware Gaussian distributions, but also
facilitates graph-based propagation, achieving a clean and
accurate instance-level Gaussian selection. Extensive vali-
dation across four applications, i.e. interactive click-based
†Corresponding author.
query, open-vocabulary text-driven query, object removal
editing, and physics simulation, demonstrates the advan-
tages of our approach and its benefits to various applica-
tions. The experiments and ablation studies further evalu-
ate the effectiveness of the key designs of our approach.
1. Introduction
3D Gaussian Splatting (3DGS) [18] models scenes as a col-
lection of 3D Gaussian primitives and exhibits great utility
in many applications [4, 5, 12, 15, 16, 21, 42, 47, 48, 50].
Recently, the advancements in large Vision-Language Mod-
els (VLMs) [2, 7, 23, 29, 33, 44] have spurred a paradigm
shift toward open-vocabulary scene understanding, where
3DGS serves as a bridge between high-fidelity scene rep-
resentation and language-guided semantic reasoning. By
combining these capabilities, researchers investigate open-
vocabulary scene understanding with 3DGS [27, 34, 46],
which further stimulates a wider range of semantic scene
editing and manipulation tasks [5, 17, 41, 46].
1
arXiv:2508.01740v1  [cs.CV]  3 Aug 2025

<!-- page 2 -->
Existing GS-based scene understanding works [27, 34,
41, 46, 49] distill semantic features from large VLMs [2, 20,
23, 29] to free Gaussians via differentiable rendering, and
use feature similarity to query the object-related Gaussians
for visualization or editing purposes. However, these ap-
proaches face the challenge to obtain an accurate and clean
Gaussian selection. As shown in Figure 1, when query-
ing the objects, the selection often contains extra surround-
ing Gaussians and leaves the inner gaussian in the remain-
ing scene, which apparently hinders the subsequent appli-
cations such as 3D scene editing and physical simulation.
The inaccurate and unclean Gaussian selection is due to
the representation and rendering of 3DGS [18]. First, these
approaches optimize unbounded Gaussians which tend to
expand excessively across different views. It inadvertently
causes semantic artifacts from unrelated objects and cre-
ates redundant features in localized regions. Second, the
α-blending rasterization back-propagates visual attributes
along view rays but suffers from ambiguities when assign-
ing the semantic features to overlapping gaussians, leading
to inconsistent local features. Third, selecting Gaussians
solely based on feature similarity ignores critical 3D spa-
tial constraints, resulting in the inclusion of extra Gaussians
from non-adjacent objects that share similar features.
To address the above issues, we propose AG2aussian,
an anchor-graph structured Gaussian splatting for instance-
level 3D scene understanding and editing tasks. The key
idea is to construct an anchor-graph structure to organize the
semantic features and regulate Gaussian primitives. Specifi-
cally, the anchor-graph acts as a higher-level semantic struc-
ture, where each anchor is attached with a semantic fea-
ture and a small set of Gaussians. The advantages of this
structure are two-fold: First, leveraging the semantic an-
chors to constrain the Gaussians, it promotes a compact and
instance-aware Gaussian distribution. Second, the anchor-
graph enables graph-based propagation to refine the seman-
tic features of anchors, which significantly improves the ac-
curacy and cleanness of object-related Gaussian selection.
Our technical contributions are summarized as follows:
(1) We introduce the anchor-graph structured 3D Gaussian
representation, which constructs the anchor-graph struc-
ture to organize the semantic features and regulate Gaus-
sian primitives for instance-level tasks.
(2) We propose
the anchor-graph feature propagation algorithm to refine
the semantic features and produce accurate Gaussian selec-
tions. (3) We employ the anchor-graph structured 3DGS
to a series of scene understanding and editing applications,
which demonstrate the benefits of our representation for the
processing operations of these applications. We conduct
extensive experiments to validate the effectiveness of our
approach, demonstrating superior performance in terms of
Gaussian selection accuracy and the benefits to the intended
applications. Code released in GitHub/AGGaussian.
2. Related Works
Numerous studies have adapted 3D Gaussian Splatting
(3DGS) [18] for diverse applications [4, 5, 8, 11, 13, 15, 25,
37, 39, 40, 42, 43, 45]. For instance, 2DGS [15] projects
3D volumes into 2D oriented Gaussian disks to maintain
view-consistent geometry reconstruction; Texture-GS [43]
and RefGaussian [45] disentangle geometry, texture, and
reflections to enable flexible scene editing; EVER [25] in-
troduces a physically accurate ellipsoid-based representa-
tion for physics-driven photorealistic rendering; PhyGaus-
sian [42] incorporates physical properties into Gaussian
primitives for high-fidelity simulation, to name a few.
On the other hand, the free-form nature of these Gaus-
sians often leads to redundant components and loose spatial
associations. Some works develop different data structures
to organize the Gaussians to balance the efficiency and high-
fidelity rendering [6, 9, 10, 24, 30, 38]. Scaffold-GS [24]
compresses the Gaussian representation using neural Gaus-
sians to address the memory efficiency issue.
Octree-
GS [30] utilizes the octree structure to enable adaptive detail
control for complex large-scale scenes.
In addition to the geometry and appearance properties,
some recent works study to enhance the 3D representa-
tions with semantic features [3, 27, 28, 34, 41, 46, 49].
Leveraging the 3DGS representation, early works such as
LEGaussian [34] and Feature3DGS [49] laid the ground-
work by integrating semantic information into the Gaus-
sian framework to support semantic segmentation and open-
vocabulary query. Subsequent works, including Gaussian-
Grouping [46], SAGA [3], and OpenGaussian [41], further
advance this field by decoupling semantic cues from seg-
mentation tasks. These methods enable instance-level seg-
mentation and open-vocabulary object selection, but still
face challenges: redundant Gaussian components, inconsis-
tent feature assignments within local regions, and ambigu-
ous object boundaries due to insufficient spatial constraints.
To address the remaining challenges, we construct the
anchor-graph structure to organize the semantic features
and regulate the Gaussian primitives. The closest works to
ours are Scaffold-GS [24] and OpenGaussian [41]. Com-
pared to them, our novel design lies in constructing the
anchor-graph structure to regulate the explicit Gaussian
primitives, and adopt the graph-based propagation to refine
the semantic features. We demonstrate that these designs
play a critical role in producing a clean and accurate Gaus-
sian selection for a series of instance-level tasks.
3. Method
Given the multi-view images of a 3D scene, we optimize the
anchor-graph structured Gaussian representation so that the
renderings match the visual images and their semantic maps
produced by large VLMs [20, 29]. The entire pipeline is di-
2

<!-- page 3 -->
LINK
Stage 1: Anchor-Gaussian Growing
IoU
Matching
Voxelized
Scene Anchors
L_cInA
Anchor-Gaussians
Localized
Rasterizer
Mask-Clip Pairs
Target Masks
GT RGB
ℒ!"
ℒ!"
Anchor
𝑷𝒐𝒔𝒊𝒕𝒊𝒐𝒏
𝒙∈ℝ𝟑
𝒌Gaussians
𝑭𝒆𝒂𝒕𝒖𝒓𝒆
𝒇∈ℝ𝟑
𝑶𝒇𝒇𝒔𝒆𝒕
𝓞∈ℝ𝟑
𝑺𝒊𝒛𝒆
𝒍∈ℝ
𝑷𝒐𝒔𝒊𝒕𝒊𝒐𝒏
𝒙∈ℝ𝟑
𝑷𝒐𝒔𝒊𝒕𝒊𝒐𝒏
𝒙∈ℝ𝟑
Anchor
𝒂, 𝒇, 𝒍
𝒌Gaussians
𝓞, ;𝒔, 𝒒, 𝜶, 𝒄
LINK
Anchor
𝒙, 𝒇, 𝒍
𝒌Gaussians
𝒐, ;𝒔, 𝒒, 𝜶, 𝒄
𝜇= 𝑥+ 𝑜⋅𝑙, 𝑠= 𝑆𝑖𝑔𝑚𝑜𝑖𝑑
̂𝑠⋅𝑙
RGB Map
Feature Map
Distortion Map
ℒ#
ℒ!$
ℒ!%
ℒ&#'$
SAM
Stage 2: Anchor-Graph Propagation
Stage 3: Language Feature Attachment
Graph Construction
Graph Clustering
Anchor Clusters
CLIP
Anchor-Graph
𝒘𝒊𝒋
𝑭𝒊
𝑭𝒋
ℒ()*(
Propagation
Intra-Voxel Edge
Inter-Voxel Edge
…
0.9
Matching
Figure 2. Our AG2aussian consists of three stages. The first stage performs anchor-gaussian growing, where we initialize the anchors based
on multi-resolution voxelization and optimize the anchor-Gaussian via a differentiable rasterizer. The second stage constructs the anchor-
graph structure with sparse inter-voxel edges and intra-voxel edges, and adopts a graph-based propagation to refine the semantic features.
Finally, the third stage localizes the object instances via graph clustering and attaches the matched language features to our representation.
vided into three stages, as shown in Figure 2. The first stage
(Sec. 3.1) performs anchor-Gaussian growing to initialize
the semantic anchors and the associated Gaussians. The
second stage (Sec.3.2) constructs the anchor-graph structure
and adopts a graph-based propagation to refine the semantic
features. The third stage (Sec. 3.3) localizes the object in-
stances and attaches language features to our anchor-graph
structured Gaussian representation.
3.1. Anchor-Gaussian Growing
The vanilla 3D Gaussian splatting [18] represents the 3D
scenes with a collection of Gaussians G = {gi}N
i=1 where
N is the Gaussian number. Each Gaussian can be expressed
as gi = {µi, si, qi, αi, ci}, with mean position µi ∈R3,
scale factors si ∈R3, quaternion qi ∈R4, opacity αi ∈R,
and color properties ci ∈Rd. The scaling and rotation form
the covariance matrix of a Gaussian. These 3D Gaussians
are splatted into a 2D image using differentiable rasteri-
zation via point-based α-blending [51]. Given the camera
viewpoint, the value at pixel v of the rendered image is ob-
tained by summing the Gaussians Nv intersected by its ray:
  I
_
v=\s
um _
{i\
i
n {
\m a thcal {N}_v}}t_i c_i\prod ^{i-1}_{j=1}(1-t_i), \label {eq:gaussian} 
(1)
where ti
=
αie−1
2 (v−ˆµi)ˆΣ−1
i
(v−ˆµi) is the contribution
weight and ci is the color of the i-th Gaussian at pixel v.
Moreover, by replacing the color values with the semantic
feature fi attached to each Gaussian, we can obtain the fea-
ture map following the same rendering process.
However, the vanilla 3DGS [18] allows the Gaussians to
freely extend and split within the 3D scene. It does not only
cause inefficient data storage, but also affects the attribute
back-propagation along the view rays, which becomes the
main obstacle to obtaining a clean Gaussian selection. In-
spired by works such as Scaffold-GS [24] and HAC [6], we
develop a structured 3DGS that uses semantic anchors to
regulate the corresponding Gaussians. The goal is to con-
strain the positions and scales of the Gaussians, in order to
obtain a compact and instance-aware Gaussian distribution.
Anchor-Gaussian Initialization. Given the multi-view im-
ages, we reconstruct the sparse point cloud via Structure-
from-Motion (SfM) [32] as the initial positions of Gaus-
sians. The sparse point cloud initializes a three-layer multi-
resolution voxelization, where the centers of occupied vox-
els are taken as the locations of anchors A. Each anchor
a ∈A is parameterized by the center position x ∈R3, a
voxel size l ∈R, a semantic feature f ∈R3, and k child
Gaussians {gi = {oi, ˆsi, qi, αi, ci}}k
a each with an offset
vector oi ∈R3 and relative scaling ˆsi. Then the mean and
scaling matrix of each Gaussian are:
  \ m u  _ i =x +o _ i\cdot l,\ s _ i=Sigmoid(\hat {s}_i)\cdot l. \label {eq:anchorgaussian} 
(2)
The Gaussians inherit the semantic feature of corresponding
anchors, and can leverage the vanilla 3DGS rasterization to
obtain the rendered images and feature maps.
Anchor-Gaussian Densification. The vanilla 3DGS den-
sifies the Gaussians if they receive large gradients during
training. Intuitively, as for our anchor-Gaussian structure,
we densify the anchors and their associated Gaussians based
3

<!-- page 4 -->
on the gradients of Gaussians. Specifically, for any reso-
lution layer, assuming a Gaussian g is associated with the
anchor a located at the center of voxel v, if the Gaussian g
receives a gradient larger than a pre-defined threshold, we
intend to create a new anchor and its associated Gaussians
at the center of v. If there is no existing anchor positioned at
this voxel center, we consider the new anchor initialization
operation as valid. Otherwise, we cancel the initialization
of this new anchor and continue. Note that the multi-layer
voxelization enables a large resolution to place the anchors,
thus ensuring the expressive ability of the anchor-Gaussian
representation during training.
Structured Spatial Regularization. The anchor-Gaussian
association enables the spatial regularization of Gaussians
in terms of their positions and scaling. As demonstrated
in Eq. 2, we restrict the scale of Gaussian to be within the
voxel at which its corresponding anchor is placed. And we
use a local constraint loss to regularize the positions:
  \ m athcal {
L}_ { in}=\mathb b {E
} _{o\ in \{o_i\}^k_{a\in \mathcal {A}}}\parallel \exp ({\rm ReLU}(\parallel o_i\parallel ^2_2-1))\parallel 
(3)
where {oi}k is the offset set of anchor’s Gaussians.
We adopt the depth distortion loss from 2DGS [15] to
suppress floating Gaussians and enhance the compactness
of object surfaces by minimizing the distances between ray-
Gaussian intersection points. The distortion at pixel v is:
  \m a
t
hcal
 {L
}
_{d
,v}=\su m  _{i\in {\mathcal {N}_v}}\sum ^{i-1}_{j=1}\omega _i\omega _j(z_i-z_j)^2 
(4)
where ωi = ti
Qi−1
j=1(1 −ti) represents the contribution of
the i-th Gaussian to the rendering results, and zi is the depth
of this Gaussian along this ray. The distortion loss is defined
as Ld = E ∥{Ld,v}H×W ∥.
Semantic Contrastive Learning.
We use SAM [20] to
produce multi-view instance masks and distill the semantic
feature for each anchor. The detailed masks that are entirely
contained within other instances are removed.
We employ the contrastive learning strategy proposed by
OpenGaussian [41] to encourage features rendered from the
same object to be close to each other while those from dif-
ferent objects remain distant. For the i-th view of the multi-
view images, given the rendered feature map F and the bi-
nary mask Mj ∈{0, 1}1×H×W of the j-th object produced
by SAM [20], we can obtain the mean feature within the
mask region: F j = (Mj ·F)/ P Mj ∈R3. The intra-mask
smoothing loss Lis and inter-mask contrastive loss Lic are
used to distill semantic features which are rendered to form
the feature maps:
  \ b
egi
n
 {a
lig
n
ed} \
mathca l  {L}_{ i s } &
=\s u
m
 ^{ | M|
}_{
j
=1}
\su
m
 ^{
H,W}
_
{ h , w = 1 } M_{j,h,w}\parallel F_{:,h,w}-\overline {F}_j\parallel \\ \mathcal {L}_{ic}&=\frac {1}{m(m-1)}\sum ^{|M|}_{j=1}\sum ^{|M|}_{\substack {k=1 \\ k\ne j}}\frac {1}{\parallel \overline {F}_j-\overline {F}_k\parallel +1} \end {aligned} 
(5)
where |M| is the number of instance masks in current view,
F j and F k denote the mean features of two different masks.
3.2. Anchor-Graph Propagation
After the first stage, we obtain the anchors and their associ-
ated Gaussians that visually exhibit good semantic features
and appearance, but are far from a clean semantic represen-
tation for accurate object selections. There are two main
reasons: First, due to occlusion between Gaussians, the fea-
tures of their corresponding anchors within the same object
can differ. Second, for objects that never co-appear in the
same view, their features can still be similar. The above rea-
sons make it difficult to distinguish the objects completely.
We introduce the anchor-graph construction and propa-
gation algorithm to address these issues. The main idea is to
leverage local graph structures to refine the anchor features.
We first construct the anchor graph based on their spatial
distribution, then propagate to refine semantic similarities.
Anchor Graph Construction. Since the semantic anchors
are placed at the centers of multi-resolution voxels, we con-
nect the anchors based on the voxel neighborhood to con-
struct the anchor graph. Specifically, for a voxel at the top
layer, i.e. the layer with the smallest resolution, we collect
all the anchors placed within this voxel, no matter which
layer the anchors belong to. We create intra-voxel edges
that connect any two of the anchors within a top-layer voxel,
and inter-voxel edges between the anchors located within
neighboring top-layer voxels. In the following, we only al-
low feature propagation through these edges.
Graph Laplacian Propagation. We introduce a Gaussian-
weighted graph Laplacian propagation to smooth the fea-
tures of connected anchors while preserving sharp feature
variations along object boundaries. Assuming a weighted
adjacent matrix W and the diagonal matrix Dii = P
j Wij,
the Laplacian matrix is defined as
  { \ rm
 L }= { \ rm D}-{\rm W},\quad {\rm L,D,W}\in \mathbb {R}^{|\mathcal {A}|\times |\mathcal {A}|}. 
(6)
Then the graph Laplacian propagation algorithm takes the
Dirichlet Energy term as the propagation loss to enforce the
smoothness between anchor features:
 \lab e l {eq:L_p r
o
p} 
\ma t hc a l {L}_{prop}=2{\rm \bf Tr}({\rm F^\top LF})=\sum _{i,j}w_{ij}\parallel {\rm F}_i-{\rm F}_j\parallel ^2 
(7)
where F = {f}|A| ∈R|A|×3 denotes the feature matrix of
all anchors, and Fi denoting the feature of the i-th anchor.
4

<!-- page 5 -->
We set the weight wij = exp(−∥Fi−Fj∥2
2τ 2
) as a Gaus-
sian kernel function related to the temperature τ = 0.05.
It is worth noting that we don’t need to explicitly construct
the Laplacian matrix, but only the sparse non-zero wij to
compute the propagation loss.
3.3. Language Feature Attachment
After the second stage, we have already obtained an anchor-
graph structured representation with semantic features to
distinguish the objects. To further enable open-vocabulary
query tasks, we perform graph-based clustering to localize
the object instances and score matching to attach the addi-
tional language features to the anchors.
We use Union-Find [36] for the graph-based cluster-
ing to efficiently identify and gather the connected an-
chors with similar features. Each cluster of anchors and
the associate Gaussians form an object instance. By ren-
dering the instance-related Gaussians with white color and
the rest black color, we obtain the binary instance map
Iˆa ∈R1×H×W , with ˆa denoting the cluster of anchors.
The next step is to match the rendered instance maps
and the ground-truth SAM masks for all the views, to ob-
tain CLIP-encoded language features [29] for the instances.
Similar to [41], the matching score between Iˆa and Mi is:
  S_{ \ hat {a}, i}= { \rm  I oU } ( I _{\hat {a}},M_i)\cdot (1-\parallel \overline {F}_{\hat {a}}-\overline {F}_i \parallel ^1) 
(8)
where IoU is used to measure the overlap between Iˆa and
Mi, while the second term estimates the distance between
the mean features. Based on the scores, for the clustered
anchors of each object, we select the most relevant mask
by SAM [20] and attach the feature f clip ∈R512 of corre-
sponding CLIP [29] encodings to these anchors. Notably,
to accelerate this process, we skip the clusters with only
one single anchor, as they are mostly cluttered transparent
points that do not contribute meaningfully to the scene.
3.4. Training
Our three-stage training is set as follows:
Stage 1: Anchor-Gaussian Growing. During the first 30k
iterations, our approach focuses on the growing of anchors
and the associated Gaussians. The loss for this stage is:
  \math c al {L } _{stag e 1}=\ma t hcal { L }_{3dgs}+\lambda _{in}\mathcal {L}_{in}+\lambda _{is}\mathcal {L}_{is}+\lambda _{ic}\mathcal {L}_{ic}+\lambda _{d}\mathcal {L}_{d} (9)
with L3dgs denotes the 3DGS reconstruction loss [18].
Stage 2: Anchor-Graph-based Feature Propagation. For
the next 5k iterations, we first perform the anchor-graph
construction, followed by the graph Laplacian propagation.
The loss function is
  \math c al {L}_ { stage2}=\mathcal {L}_{stage1}+\lambda _{prop}\mathcal {L}_{prop} 
(10)
Stage 3: Language Feature Attachment.
As a post-
processing stage, we sequentially perform the clustering,
matching, and attach the mean language feature f clip of
each object instance to the corresponding anchors.
At the end of the training, we obtain the optimized
anchor-graph structured 3D Gaussian splatting representa-
tion, where each anchor carries its position and correspond-
ing voxel size (x, l), the distilled semantic feature f, the lan-
guage feature f clip, and a small set of associated Gaussians
{gi}k. Note that the anchor-graph structure not only makes
a better scene understanding with the semantic features, but
also ensures a compact set of Gaussians and alleviates the
computation burden.
4. Instance-level Application
The clean semantic Gaussian representation produced by
our approach benefits a series of 3D scene understanding
and editing applications. Below we describe how to deploy
our approach for various applications.
Click Query. We adopt a local region-growing strategy for
the click query application. Given the camera viewpoint
of the image, we render the 3D scene into a depth map.
The depth allows us to transform the clicked pixel into a
3D point p with the camera’s extrinsic and intrinsic matri-
ces. Next, using a Nearest Neighbor search, we identify the
closest anchor a∗as the initial seed anchor. From this seed,
the region grows iteratively: if an anchor ai is connected to
another anchor aj with an edge weight wij > 0.90, then
aj should be included in the selected region. Note that we
double the voxel size as the threshold for creating the graph
edges only during this query.
Text Query. The text query adopts the same growing strat-
egy with click query but a slightly different seed initializa-
tion. Given an open-vocabulary text query, we extract its
text embedding using CLIP encoder [29] and calculate the
cosine similarity between this feature and the language fea-
tures of clustered anchors. Assuming ε to denote the max-
imum feature similarity score among them, we select the
anchor clustering whose language features have a similar-
ity larger than ε −0.1 as the initial seed anchors. By per-
forming the region-growing process, we obtain the anchors
within the region as the query results.
Scene Editing. A clean instance-aware Gaussian represen-
tation enables object selection and local scene editing oper-
ations. Specifically, for 3D object removal, directly deleting
the selected anchors and the associated Gaussians (with the
above query techniques) would leave artifacts in the region.
Therefore, we add a small number of new anchors within
the object region and render the image with a region mask.
The region mask is obtained by rendering the Gaussians of
the newly added anchors and their neighbor anchors. Then
we perform 2D image inpainting with LaMa [35] to obtain
realistic images and optimize our anchor-graph structured
Gaussians w.r.t. the inpainted images.
Physical Simulation. Physical simulation aims to imitate
5

<!-- page 6 -->
Ours
OpenGaussian
GsGrouping
SAGA
Input
Semantic
Feature Map
Click Query
Text Query
plant
stable bread
blue oven gloves
stuffed bear
three cookies
coffee mug
GT Masks
Figure 3. The rendered semantic feature map and object query results of our approach and related methods. The presented GT images are
generated by SAM [20]. By contrast, our approach produces more precise and less noisy instance masks for the queried objects.
the physical interactions between the selected object and the
remaining scene. We adopt PhyGaussian [42], a Gaussian-
based simulator implemented via MLS-MPM [14], as our
physical engine. The selected object is assigned Young’s
modulus E = 2e8 and Poisson’s ratio ν = 0.4 to prevent
deformation during simulation, while the remaining scene
has lower physical coefficients (E = 2e6, ν = 0.3). More
implementation details of the physical simulation applica-
tion is presented are the supplementary material.
5. Experiments
5.1. Setup
Datasets. We evaluate our approach on the LERF-OVS [19]
datasets for the open-vocabulary object query, scene edit-
ing, and physical simulation applications. The four scenes
are manually annotated with SAM to obtain accurate masks
instead of the coarse bounding boxes. We further evaluate
object querying on Mip-NeRF360 [1] and object selection
on LLFF [26], which contain more challenging examples.
Implementation Details. We use the SAM ViT-H model to
generate 2D masks of input images, and extract language
features for each instance using the OpenCLIP ViT-B/32
model. Throughout all experiments, each anchor is assigned
k = 5 Gaussians. For multi-resolution voxels, we develop
three resolution levels, each with a voxel size 4 times larger
than the previous one. The minimum resolution (for the top
layer) is 2003. During training, we use Adam optimizer
and set λin = 0.5, λis = 2.5, λic = 0.25, λd = 50,
λprop = 0.01. For the scene editing task, we use LaMa
Big-Lamm to generate 2D inpainting images.
5.2. Scene Understanding and Object Query
We conduct the scene understanding and open-vocabulary
object query tasks and compare to the existing state-of-
the-art 3DGS-based approaches, i.e. SAGA [3], GsGroup-
ing [46], OpenGaussian [41]. The object query tasks in-
clude click query and text query with object class names.
Figure 3 shows the rendered feature maps and the results
of click query and text query for a variety of scenes. As
shown in Figure 3, our approach produces smooth and clean
semantic feature maps that can easily distinguish the objects
with accurate boundaries. This leads to more precise query
results, compared to the noisy masks and blurry boundaries
with other approaches. Please refer to the supplementary
material for more results.
Table 1 reports the quantitative evaluation. We use aver-
age IoU and BoundaryIoU between the binary images ren-
dered with the selected anchor-Gaussians and the ground-
truth object mask as the evaluation metrics.
Our results
6

<!-- page 7 -->
Table 1. Quantitative evaluation of object query applications on LERF dataset [19].
mIoU ↑
mBIoU. ↑
Query
Methods
figurines
teatime
ramen
kitchen
Mean
figurines
teatime
ramen
kitchen
Mean
Click
SAGA [3]
81.16
92.91
63.08
83.03
80.05
76.21
60.27
54.03
62.07
63.15
GsGrouping [46]
32.54
75.06
39.99
14.70
40.57
28.43
57.20
35.86
9.74
32.81
OpenGaussian [41]
85.15
80.31
48.88
79.48
73.46
81.28
51.34
50.73
49.95
58.33
Ours
88.61
95.87
88.04
80.80
88.33
85.27
87.65
71.54
49.86
73.58
Text
SAGA [3]
16.76
18.90
11.01
5.29
12.99
16.14
17.74
10.20
3.58
11.92
GsGrouping [46]
28.11
64.71
33.47
16.68
35.74
27.19
59.30
31.82
14.82
33.28
OpenGaussian [41]
57.41
62.54
30.77
25.96
44.17
54.83
56.38
26.94
17.32
38.87
Ours
66.98
71.62
47.99
30.82
54.35
65.30
67.83
42.45
22.15
49.43
SAGA
GsGrouping
OpenGaussian
Ours
Selection
Remaining
Selection
Remaining
Figurines
Ramen
Figure 4. Click-based instance-level Gaussian selection and the
remaining scenes. Our approach makes more accurate and com-
plete Gaussian selections, though with visual artifacts of occluded
Gaussians in the remaining scenes.
achieve remarkable improvement over the others in terms
of both the click query and text query, validating the high
accuracy of the mask region and object boundaries.
5.3. 3D Scene Editing and Physical Simulation
An accurate selection of the specified object is the prereq-
uisite for the scene manipulation tasks. Unlike object query
tasks that only require precise object masks for 2D visual-
ization, object selection emphasizes accurately picking the
Gaussians for specific object, including the inner Gaussians.
Object Selection. We present the object selection results of
ours and other existing methods in Figure 4. Their default
Input
Artifact Location
Editing Result
Feature Map
Selection
Removal
Figure 5. Object removal results. After deleting the selected Gaus-
sians, we localize artifact regions and inpaint the remaining scene.
parameters and operation processes are adopted in the ex-
periments. As for our approach, we perform the click query
and take the Gaussians associated with the selected anchors.
In Figure 4, we render both the selected and the remain-
ing Gaussians. In spite of the clean selection shown in the
rendering of the selected Gaussians, we can see that the ob-
jects are more completely removed in the remaining scenes,
e.g. the white hand in the Figurines scene and chopsticks in
the Ramen scene. Although there are still visual artifacts in
the remaining scenes, it is because the remaining Gaussians
of the other objects don’t have the correct appearance due
to the occlusion problem of multi-view reconstruction.
3D Scene Editing. We further present the 3D object re-
moval editing results to validate our claim that our approach
exceeds an accurate and complete object selection. Since
our anchor-graph structure can accurately localize the an-
chors corresponding to the object and its extended bound-
ary, we can easily identify the artifact region to be repaired
without affecting the surrounding objects.
Physical Simulation. The physical simulation task poses a
strict requirement for the object selection, since the object
Gaussians entangled with the remaining scene would cause
unrealistic simulation results. Figure 6 shows the simulation
results of ours and other approaches. Taking the upward-
dragged bag as an example, it remains in the original po-
sition in the SAGA [3] and OpenGaussian [41] results, be-
cause most of the inner-object Gaussians are not selected.
As for GaussianGrouping [46], which uses a convex hull ge-
ometry to select the Gaussians, it moves the bag but leaves
messy surrounding Gaussians. By contrast, our approach
not only successfully drags the bag, but also leaves small
7

<!-- page 8 -->
Ours
OpenGaussian
GsGrouping
SAGA
Ours + Inpainting
Frame 4
Frame 8
Frame 12
Drag and Drop
Drag
Force Direction
Figure 6. Physical simulation by applying the external forces (red arrows) to drag the objects. All methods are simulated with identical
physical coefficients using PhysGaussian. The clean and accurate Gaussian selection enables the realistic simulation with our approach.
𝑤/𝑜𝑎𝑔
𝐹𝑢𝑙𝑙
𝑤/𝑜𝑔𝑟𝑎𝑝ℎ
Selection
Remaining
Figure 7. Visual results of the ablation study for anchor-graph
structure design. With the anchor-graph structure and the graph-
related operations, our full approach produces precise Gaussian
selections and a relatively clean remaining scene.
appearance artifacts that be easily localized and inpainted
with our anchor-graph structure.
5.4. Ablation Study.
Anchor-Graph Structure Design. We emphasize the role
of the anchor-graph structure in organizing the Gaussians
for a clean object selection. The simplest baseline w/o ag
is to remove the anchor-structure from our approach. That
is, the semantic features are attached to the vanilla Gaus-
sians [18] and learned from contrastive learning. Another
baseline w/o graph is to retain the anchor-gaussian struc-
ture but omit all the graph-related operations, i.e. the graph
propagation in the second stage and the region growing in
the query application. The implementation of this baseline
is to remove the loss Lprop and use the semantic feature
similarity for the query applications. The qualitative results
are shown in Figure 7.
Graph-Related Operations. We further analyze the effects
of each graph-related operation separately. The quantita-
Table 2. Quantitative evaluations of the ablation study for the two
graph-related operations.
Case
Lprop
GraphSeg
mIoU ↑
mBIoU. ↑
#1
42.72
40.20
#2
!
45.85
43.51
#3
!
49.10
44.72
Full
!
!
54.35
49.43
tive evaluation is reported in Table 2, where the two vari-
ables Lprop and GraphSeg determine the involvement of
the graph propagation in the second stage and the region
growing in the query application, respectively. Table 2 val-
idates the significant effectiveness of the two graph-related
operations in making accurate object queries.
6. Conclusion
We present AG2aussian, an anchor-graph structured Gaus-
sian representation for instance-level 3D scene understand-
ing and editing tasks.
The key idea is to construct an
anchor-graph structure to organize the semantic features
and regulate the associate Gaussians to produce a com-
pact and instance-aware Gaussian distribution. The graph-
related operations facilitate to make clean and accurate
instance-level Gaussian selection, which, as demonstrated
with our experiments, exhibit great benefits to the scene
understanding and editing applications, including interac-
tive click query, open-vocabulary text query, localized scene
editing, and physical simulation.
Our approach still holds some limitations. Similar to the
existing approaches, our method suffers from the glass and
metal objects due to the feature blending. And sometimes
the over-segmentation of SAM introduces additional diffi-
culty to maintaining the consistent features of the objects.
8

<!-- page 9 -->
7. Acknowledgement
This work is supported by the Joint Funds of the National
Natural Science Foundation of China (U23A20312),
the Excellent Young Scientists Fund Program (Over-
seas)
of
Shandong
Province
(No.2023HWYQ-034),
the
National
Natural
Science
Foundation
of
China
(62302269),
and a grant from the Natural Science
Foundation of Shandong Province (No.ZR2023QF077).
References
[1] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5470–5479, 2022. 6, 1, 2, 3, 4
[2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou,
Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-
ing properties in self-supervised vision transformers.
In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 9650–9660, 2021. 1, 2
[3] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xi-
aopeng Zhang, Wei Shen, and Qi Tian. Segment any 3d gaus-
sians. In Proceedings of the AAAI Conference on Artificial
Intelligence, pages 1971–1979, 2025. 2, 6, 7, 1
[4] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian
Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao,
and Guofeng Zhang. Pgsr: Planar-based gaussian splatting
for efficient and high-fidelity surface reconstruction. IEEE
Transactions on Visualization and Computer Graphics, 2024.
1, 2
[5] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xi-
aofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping
Liu, and Guosheng Lin. Gaussianeditor: Swift and control-
lable 3d editing with gaussian splatting. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 21476–21485, 2024. 1, 2
[6] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi,
and Jianfei Cai.
Hac: Hash-grid assisted context for 3d
gaussian splatting compression. In European Conference on
Computer Vision, 2024. 2, 3
[7] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexan-
der Schwing, and Joon-Young Lee.
Tracking anything
with decoupled video segmentation. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 1316–1326, 2023. 1, 2
[8] Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu,
Huamin Wang, and Weiwei Xu. High-quality surface recon-
struction using gaussian surfels. In ACM SIGGRAPH 2024
Conference Papers, pages 1–11, 2024. 2
[9] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia
Xu, Zhangyang Wang, et al.
Lightgaussian: Unbounded
3d gaussian compression with 15x reduction and 200+ fps.
Advances in neural information processing systems, 37:
140138–140158, 2024. 2
[10] Lin Gao, Jie Yang, Bo-Tao Zhang, Jia-Mu Sun, Yu-Jie Yuan,
Hongbo Fu, and Yu-Kun Lai. Mesh-based gaussian splat-
ting for real-time large-scale deformation.
arXiv preprint
arXiv:2402.04796, 2024. 2
[11] Xinyu Gao, Ziyi Yang, Bingchen Gong, Xiaoguang Han,
Sipeng Yang, and Xiaogang Jin. Towards realistic example-
based modeling via 3d gaussian stitching. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 26597–26607, 2025. 2
[12] Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy
Jatavallabhula,
Bipasha Sen,
Aditya Agarwal,
Corban
Rivera, William Paul, Kirsty Ellis, Rama Chellappa, et al.
Conceptgraphs:
Open-vocabulary 3d scene graphs for
perception and planning.
In 2024 IEEE International
Conference on Robotics and Automation (ICRA), pages
5021–5028. IEEE, 2024. 1
[13] Antoine Gu´edon and Vincent Lepetit.
Sugar:
Surface-
aligned gaussian splatting for efficient 3d mesh reconstruc-
tion and high-quality mesh rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5354–5363, 2024. 2
[14] Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu,
Andre Pradhana, and Chenfanfu Jiang.
A moving least
squares material point method with displacement disconti-
nuity and two-way rigid body coupling. ACM Transactions
on Graphics (TOG), 37(4):150, 2018. 6, 1
[15] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically accu-
rate radiance fields. In ACM SIGGRAPH 2024 conference
papers, pages 1–11, 2024. 1, 2, 4
[16] Tianyu Huang, Haoze Zhang, Yihan Zeng, Zhilu Zhang, Hui
Li, Wangmeng Zuo, and Rynson WH Lau. Dreamphysics:
Learning physical properties of dynamic 3d gaussians with
video diffusion priors.
arXiv preprint arXiv:2406.01476,
2024. 1
[17] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaox-
iao Long, Wenping Wang, and Yuexin Ma. Gaussianshader:
3d gaussian splatting with shading functions for reflective
surfaces.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 5322–
5332, 2024. 1
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 1, 2, 3, 5, 8
[19] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo
Kanazawa, and Matthew Tancik.
Lerf:
Language em-
bedded radiance fields.
In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 19729–
19739, 2023. 6, 7, 2, 3, 4, 5
[20] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al.
Segment
anything.
In Proceedings of the IEEE/CVF international
conference on computer vision, pages 4015–4026, 2023. 2,
4, 5, 6
[21] Xiao-Lei Li, Haodong Li, Hao-Xiang Chen, Tai-Jiang Mu,
and Shi-Min Hu. Discene: Object decoupling and interaction
9

<!-- page 10 -->
modeling for complex scene generation. In SIGGRAPH Asia
2024 Conference Papers, pages 1–12, 2024. 1
[22] Siyun Liang, Sen Wang, Kunyi Li, Michael Niemeyer, Ste-
fano Gasperini, Nassir Navab, and Federico Tombari. Su-
pergseg: Open-vocabulary 3d segmentation with structured
super-gaussians, 2024. 4
[23] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao
Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang,
Hang Su, et al.
Grounding dino:
Marrying dino with
grounded pre-training for open-set object detection.
In
European Conference on Computer Vision, pages 38–55.
Springer, 2024. 1, 2
[24] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 2, 3, 4
[25] Alexander Mai, Peter Hedman, George Kopanas, Dor
Verbin, David Futschik, Qiangeng Xu, Falko Kuester,
Jonathan T Barron, and Yinda Zhang. Ever: Exact volumet-
ric ellipsoid rendering for real-time view synthesis. arXiv
preprint arXiv:2410.01804, 2024. 2
[26] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar.
Local light field fusion:
Practical view
synthesis with prescriptive sampling guidelines.
ACM
Transactions on Graphics (TOG), 2019. 6, 1, 4
[27] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and
Hanspeter Pfister. Langsplat: 3d language gaussian splatting.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 20051–20060, 2024.
1, 2
[28] Yansong Qu, Shaohui Dai, Xinyang Li, Jianghang Lin, Liu-
juan Cao, Shengchuan Zhang, and Rongrong Ji. Goi: Find
3d gaussians of interest with an optimizable open-vocabulary
semantic-space hyperplane.
In Proceedings of the 32nd
ACM International Conference on Multimedia, pages 5328–
5337, 2024. 2
[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning, pages
8748–8763. PmLR, 2021. 1, 2, 5
[30] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. arXiv
preprint arXiv:2403.17898, 2024. 2
[31] Zhongzheng Ren,
Aseem Agarwala†,
Bryan Russell†,
Alexander G. Schwing†, and Oliver Wang†. Neural volumet-
ric object selection. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2022. († alphabetic
ordering). 1
[32] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE
conference on computer vision and pattern recognition,
pages 4104–4113, 2016. 3
[33] Yunhang Shen, Chaoyou Fu, Peixian Chen, Mengdan Zhang,
Ke Li, Xing Sun, Yunsheng Wu, Shaohui Lin, and Rongrong
Ji. Aligning and prompting everything all at once for uni-
versal visual perception. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 13193–13203, 2024. 1
[34] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-
Hua Guan.
Language embedded 3d gaussians for open-
vocabulary scene understanding.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5333–5343, 2024. 1, 2
[35] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin,
Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov,
Naejin Kong, Harshith Goka, Kiwoong Park, and Victor
Lempitsky.
Resolution-robust large mask inpainting with
fourier convolutions.
In Proceedings of the IEEE/CVF
winter conference on applications of computer vision, pages
2149–2159, 2022. 5
[36] Robert Endre Tarjan. Efficiency of a good but not linear set
union algorithm. Journal of the ACM (JACM), 22(2):215–
225, 1975. 5
[37] Joanna Waczy´nska, Piotr Borycki, Sławomir Tadeja, Jacek
Tabor, and Przemysław Spurek. Games: Mesh-based adapt-
ing and modification of gaussian splatting. arXiv preprint
arXiv:2402.01459, 2024. 2
[38] Diwen Wan, Ruijie Lu, and Gang Zeng. Superpoint gaus-
sian splatting for real-time high-fidelity dynamic scene re-
construction. arXiv preprint arXiv:2406.03697, 2024. 2
[39] Tong Wu, Yu-Jie Yuan, Ling-Xiao Zhang, Jie Yang, Yan-
Pei Cao, Ling-Qi Yan, and Lin Gao. Recent advances in 3d
gaussian splatting. Computational Visual Media, 10(4):613–
642, 2024. 2
[40] Tianhao Walter Wu, Jing Yang, Zhilin Guo, Jingyi Wan,
Fangcheng Zhong, and Cengiz Oztireli. Gaussian head &
shoulders: High fidelity neural upper body avatars with an-
chor gaussian guided texture warping.
In The Thirteenth
International Conference on Learning Representations,
2024. 2
[41] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao
Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding,
Jingdong Wang, et al. Opengaussian: Towards point-level 3d
gaussian-based open vocabulary understanding. Advances in
Neural Information Processing Systems, 37:19114–19138,
2024. 1, 2, 4, 5, 6, 7
[42] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao
Feng, Yin Yang, and Chenfanfu Jiang.
Physgaussian:
Physics-integrated 3d gaussians for generative dynamics.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 4389–4398, 2024. 1,
2, 6
[43] Tian-Xing Xu, Wenbo Hu, Yu-Kun Lai, Ying Shan, and
Song-Hai Zhang. Texture-gs: Disentangling the geometry
and texture for 3d gaussian splatting editing. In European
Conference on Computer Vision, pages 37–53. Springer,
2024. 2
[44] Jianing Yang, Xuweiyi Chen, Shengyi Qian, Nikhil Madaan,
Madhavan Iyengar, David F Fouhey, and Joyce Chai. Llm-
grounder: Open-vocabulary 3d visual grounding with large
10

<!-- page 11 -->
language model as an agent.
In 2024 IEEE International
Conference on Robotics and Automation (ICRA), pages
7694–7701. IEEE, 2024. 1
[45] Yuxuan Yao, Zixuan Zeng, Chun Gu, Xiatian Zhu, and
Li Zhang.
Reflective gaussian splatting.
arXiv preprint
arXiv:2412.19282, 2024. 2
[46] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke.
Gaussian grouping: Segment and edit anything in 3d scenes.
In ECCV, 2024. 1, 2, 6, 7
[47] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian
Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao.
Stag4d: Spatial-temporal anchored generative 4d gaussians.
In European Conference on Computer Vision, pages 163–
179. Springer, 2024. 1
[48] Bowen Zhang, Yiji Cheng, Jiaolong Yang, Chunyu Wang,
Feng Zhao, Yansong Tang, Dong Chen, and Baining Guo.
Gaussiancube: Structuring gaussian splatting using optimal
transport for 3d generative modeling. CoRR, 2024. 1
[49] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi.
Feature 3dgs: Supercharg-
ing 3d gaussian splatting to enable distilled feature fields.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 21676–21685, 2024.
2
[50] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li,
Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane
meets gaussian splatting: Fast and generalizable single-view
3d reconstruction with transformers.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10324–10335, 2024. 1
[51] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and
Markus Gross.
Ewa splatting.
IEEE Transactions on
Visualization and Computer Graphics, 8(3):223–238, 2002.
3
11

<!-- page 12 -->
AG2aussian: Anchor-Graph Structured Gaussian Splatting for Instance-Level
3D Scene Understanding and Editing
Supplementary Material
In this supplementary material, we first present the
implementation detail of the physical simulation task in
Sec. A. Then, we present more object query comparisons in
Sec. B and Sec. C. In Sec. D, we demonstrate the robustness
of our editing method and provide additional editing results
on two scenes from the Mip-NeRF360 dataset [1]. Finally,
in Sec. F and Sec. G, we provide more ablation study re-
sults, including both qualitative and quantitative analysis.
A. Application of Physical Simulation
In our experiment, we adopt PhyGaussian [42], a Gaussian-
based simulator implemented via MLS-MPM [14], as our
physical engine. The Gaussians are regarded as particles to
perform the simulation. For computational efficiency pur-
poses, we remove the background using a bounding box and
retain only the foreground particles whose opacity α > 0.02
for simulation. Specifically, in our experiments, we first
use a query operation to select the object to be simulated.
This object is then assigned Young’s modulus E = 2e8 and
Poisson’s ratio ν = 0.4 to prevent deformation during sim-
ulation. The remaining particles within the bounding box,
which serve as sticky boundary conditions with lower phys-
ical coefficients (E = 2e6, ν = 0.3), enable the simulated
object to be easily separated from the surroundings. All of
these particles are subsequently discretized into a grid 643.
For all the physical simulation experiments, we simulate a
total of 30 frames. All particles in this application are as-
signed von Mises Plasticity material.
Frame 8
Frame 0
Input
Figure 8. Physical simulation by applying the external forces (red
arrows) to drag the selected objects.
B. More Text Query Results
We visualize more results of the open-vocabulary text query
task in Figure 9, where our method demonstrates a clear ad-
vantage in selecting the complete 3D objects. By contrast,
OpenGaussian [41], due to its codebook-based clustering
approach, often fails to group an entire object into a sin-
gle cluster, as seen with the ”waldo” in the first row and
the ”stuffed bear” in the second row. Similarly, GsGroup-
ing [46] frequently includes incorrect object IDs for the
query, as seen with the ”stuffed bear” in the second row and
the ”glass of water” in the third row. Meanwhile, SAGA [3]
uses a limited number of clusters and is less aware of spa-
tial information, making it prone to missing matches and
selecting incorrect regions.
In Table 4 and Figure 13, we further report both the quan-
titative and qualitative results of open-vocabulary querying
on Mip-NeRF360 [1], evaluated with the vocabulary pro-
vided by LEGaussian [34]. Our results consistently outper-
form existing approaches, achieving significant improve-
ments in both mIoU and mBIoU. These gains hold across
diverse scenes and object types, and are especially observed
on thin, partially occluded, or clutter-surrounded objects.
Qualitative results further validate that our selected regions
can align well with the entire instance, whereas others al-
ways leave fragmented or jagged boundaries.
C. More Click Query Results
We report more object selection results on LLFF [26] in
Figure 14 and Table 5, using the scribbles provided by
NVOS [31]. As input, we first shrink the scribbles into
skeleton lines and then use the pixels on the skeleton as
click query points. By contrast, our method yields more
accurate segmentation for complex objects like fern and di-
nosaur fossils, benefiting from the use of localized anchor-
Gaussian and our anchor-graph-based strategy.
D. More Object Editing Results
Directly removing the selected Gaussians for the objects
makes artifacts in the remaining scene, due to the miss-
ing observations of the occluded region across all views, as
shown in the left column of Figure 10. Thus an inpainting
operation is necessary to fill the holes.
We compare the two inpainting techniques adopted by
GsGrouping [46] and our approach, which differ in local-
izing the artifact regions to be repaired. GsGrouping uses
Deva Tracking [7]. As shown in the top row of the figure,
due to ambiguous features and the difficulty of precisely
identifying the hole regions, most viewpoints fail to main-
tain a stable artifact mask, resulting in suboptimal editing
outcomes. By contrast, our anchor-graph structure enables
an accurate selection of the object including the inner Gaus-
sians, thus providing a precise localization of the artifact
region by extending the boundary of the selected object,
1

<!-- page 13 -->
Ours
OpenGaussian
GsGrouping
SAGA
GT Masks
Input
Waldo
Red toy chair
Spatula
Stuffed bear
Three cookies
Coffee mug
Spoon
Glass of water
Bowl
Dark cup
Frog cup
Plate
Figure 9. Open-vocabulary 3D object selection on the LERF dataset [19]. AG2aussian outperforms other approaches in accurately identi-
fying the clean and complete 3D objects corresponding to text queries.
Removal Result
DEVA-Tracking
Artifact Location
Editing Result
Graph Boundary
Removal
Figure 10. Object removing editing results with artifact regions
localized and inpainted with different techniques. Compared to the
DEVA-Tracking [7] adopted by GsGrouping [46] (top row), our
anchor-graph structured representation (bottom row) enables an
accurate localization of the artifact regions and thus makes realistic
inpainting results without affecting the surrounding objects.
yielding more reliable and visually coherent editing results.
To further validate the performance of our artifact local-
ization, we performed editing experiments on two scenes
from the Mip-NeRF360 dataset [1]. For the counter scene,
we removed three objects of varying sizes, including a
transparent kettle. As for the kitchen scene, we evaluated
our method’s ability to repair large hole regions resulting
Input
Artifact Location
Editing Result
Figure 11. More editing results on MipNeRF360 [1] using our
graph-based artifact localization technique.
Recoloring & Insertion Insertion & Recoloring
+Recoloring
Figure 12. More object recoloring and insertion editing results on
MipNeRF360 [1].
from object removal. As shown in Figure 11, our approach
accurately identifies and fills the hole regions, resulting in
high-quality and consistent scene editing.
2

<!-- page 14 -->
Ours
OpenGaussian
GsGrouping
SAGA
GT Masks
Input
Plant
Stable bread
Blue oven gloves 
Blue grey chair
Yellow wood floors
Wood desk
Basket Weave Cloth
Old pink striped cloth
LEGO Technic 856 Bulldozer
Figure 13. Open-vocabulary 3D object selection on the Mip-NeRF360 dataset [1].
Ours
OpenGaussian
GsGrouping
SAGA
GT Masks
Input
Figure 14. Scribbles-based 3D object selection on the LLFF dataset [1].
Additionally, we present the results of object recoloring
and insertion of a complex scene in Figure 12, which con-
tains many objects close to each other and has occlusions
across multiple views.
E. Computation Overhead
The maximum reserved memory, training time, and ren-
dering FPS are reported in Table 3. For our anchor-graph
structure, we store the anchors only for the occupied voxels
and the sparse edges between neighbor anchors, incurring
minimal additional memory. On the other hand, this struc-
ture regularizes the Gaussian primitives to lie around the
object surfaces, which largely reduces the number of Gaus-
sians and thus the training time. Notably, we do not intend
to claim a faster rendering speed, since we implemented
a CUDA-based module to render RGB, feature map, and
other outputs in one pass, while SAGA and OpenGaussian
need to invoke the renderer multiple times.
Table 3. Computation Overhead on LERF dataset [19].
Methods
Memory↓
Train Time↓
Rendering FPS↑
SAGA
13.29 GB
33.63 mins
∼252
GsGrouping
20.21 GB
51.21 mins
∼114
OpenGaussian
16.81 GB
74.31 mins
∼96
w/ codebook
12.91 GB
69.97 mins
∼185
Ours
7.56 GB
39.55 mins
∼515
3

<!-- page 15 -->
Table 4. Quantitative evaluation of text querying on Mip-NeRF360 dataset [1].
mIoU. ↑
mBIoU. ↑
Methods
bicycle
bonsai
counter
garden
kitchen
room
Mean
bicycle
bonsai
counter
garden
kitchen
room
Mean
SAGA
1.58
32.38
19.24
19.21
17.26
0.16
14.97
2.13
24.21
15.68
15.36
9.33
0.2
11.15
GsGrouping
10.52
68.73
47.73
34.59
61.7
41.22
44.08
8.89
53.62
44.61
29.52
54.52
36.04
37.86
OpenGaussian
25.97
33.2
47.52
25.87
41.42
41.7
35.94
15.41
26.34
41.59
20.87
21.7
35.84
26.95
Ours
31.15
53.47
61.89
34.46
62.26
50.76
48.99
18.94
48.49
58.85
31.52
41.77
45.03
40.76
Table 5. Quantitative evaluation of click querying on LLFF dataset [26].
mIoU. ↑
mBIoU. ↑
Methods
fern
flower
fortress
horns c
horns l
leaves
orchids
trex
Mean
fern
flower
fortress
horns c
horns l
leaves
orchids
trex
Mean
SAGA
82.53
95.15
98.15
92.83
94.57
92.88
88.82
83.99
91.61
75.12
80.87
78.18
68.44
72.2
77.89
74.76
70.25
75.04
GsGrouping
80.70
57.72
97.75
96.78
94.58
70.5
36.13
51.69
72.73
64.74
35.99
55.02
69.74
73.57
48.68
26.09
49.38
52.56
OpenGaussian
70.74
62.63
94.91
79.81
77.81
87.68
59.88
68.88
75.29
58.81
36.75
67.38
47.24
52.85
57.81
43.49
66.25
53.82
Ours
82.01
95.38
98.59
97.36
96.31
93.89
90.76
87.02
92.66
77.85
81.73
91.06
81.24
83.54
80.71
80.42
85.24
82.64
F. More Ablation Study Results
Table 6 presents the complete ablation study results on the
LERF dataset [19]. Overall, our graph-related operations
significantly improve both mask completeness and bound-
ary quality, as evidenced by notable gains in mIoU and
mBIoU.
To further assess the importance of these operations
for the query task, we demonstrate the selected Gaussians
and the remaining scenes.
Figure 15 provides a full vi-
sualization of all ablation variants.
Our graph-based re-
gion growing effectively prevents the selection of Gaus-
sians outside the target object, as demonstrated by the com-
parison between the w/o GraphSeg variant and our full
method. Moreover, our graph propagation smooths the fea-
ture field within the object and enhances a clean Gaus-
sian selection, effectively eliminating inner Gaussians in
the remaining scenes, as shown by the comparison between
w/o Lprop and our full method. Additionally, our anchor-
Gaussian structure effectively constrains the local distribu-
tion of Gaussians, as demonstrated by the comparison be-
tween w/o ag and w/o Graph. Overall, our full method
not only enables the clean selection of objects but also en-
sures the comprehensive inclusiveness of the inner object
Gaussians.
G. Comparison with Other Structured-GSes
Several recent works explore structured 3DGS, but for dif-
ferent goals and thus framework designs. Scaffold-GS [24]
proposes the Anchor-Gaussian structure to distribute local
3D Gaussians and predicts their view-adaptive attributes.
However, it does not localize the Gaussians to distribute
within the voxel of the corresponding anchor, and elim-
inates the anchor-graph for the feature propagation. Su-
perGSeg [22] proposes to cluster the optimized Gaussians
into Super-Gaussians and distill the semantic features to
comprehensively understand 3D scenes. However, it lacks
anchor-graph-based propagation to further refine the local
feature fields and requires a much larger memory cost dur-
ing training.
Therefore, we perform the ablation study experiments
(w/o localization and w/ codebook) to validate the effec-
tiveness of our design, as shown in Table 7. Specifically, for
w/o localization, we remove the scaling constraint (Eq. 2)
and structured spatial regularization (Eq. 3-4), to evaluate
the effectiveness of our anchor-graph structure compared to
ScaffoldGS and SuperGSeg. For w/ codebook, we preserve
our stage 1 and introduce a learnable codebook to emulate
the Super-Gaussians proposed by SuperGSeg. Our full ap-
proach significantly outperforms both variants in segmenta-
tion accuracy, demonstrating the advantages of our anchor-
graph–based localization and propagation.
4

<!-- page 16 -->
𝐹𝑢𝑙𝑙
GT + Click Points
𝑤/𝑜𝑎𝑔
𝑤/𝑜ℒ!"#!
𝑤/𝑜𝐺𝑟𝑎𝑝ℎ𝑆𝑒𝑔
𝑤/𝑜𝑔𝑟𝑎𝑝ℎ
Selection
Remaining
Selection
Remaining
Figure 15. Ablation study results. We separately validated the importance of our key design for segmentation task, the Anchor-Gaussian
structure, and the Graph-based operation. The advantage of the Anchor-Gaussian is demonstrated by comparing w/o GraphSeg with
a variant that uses 3DGS without our anchor-graph (w/o ag). The effectiveness of our Graph-based Operation respectively adopting
w/o graph, w/o Lprop and w/o GraphSeg.
Table 6. Full ablation studies on the LERF-OVS dataset [19] about the key designed.
mIoU ↑
mBIoU. ↑
Case
w/ Lprop
w/ GraphSeg
figurines
teatime
ramen
kitchen
Mean
figurines
teatime
ramen
kitchen
Mean
#1
57.62
64.72
26.39
22.14
42.72
56.72
61.30
26.05
16.73
40.20
#2
!
55.95
66.54
31.45
29.50
45.85
58.59
63.13
31.02
21.32
43.51
#3
!
65.08
71.16
28.15
32.01
49.10
63.61
67.33
26.46
21.46
44.72
Full
!
!
66.98
71.62
47.99
30.82
54.35
65.30
67.83
42.45
22.15
49.43
Table 7. Ablation Study of Structured-GS Design on LERF-OVS dataset [19]
mIoU ↑
mBIoU. ↑
Methods
figurines
teatime
ramen
kitchen
Mean
figurines
teatime
ramen
kitchen
Mean
w/ codebook
49.83
66.77
17.99
27.48
40.51
35.33
60.67
15.29
20.3
32.89
w/o localized
28.40
53.94
14.51
24.97
30.45
25.94
51.39
13.04
19.73
27.52
Ours
66.98
71.62
47.99
30.82
54.35
65.30
67.83
42.45
22.15
49.43
5
