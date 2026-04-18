<!-- page 1 -->
OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with
Context-aware Cross-view Fusion
Tianyu Huang1
Runnan Chen1
Dongting Hu2
Fengming Huang1
Mingming Gong2
Tongliang Liu1
1University of Sydney
2University of Melbourne
Crop
Clip
“Bed”
“Pikachu”
“Pikachu”
“Paper”
“Knife”
Local Feature Extraction
Mean Feature Aggregration
“Bag”
Clip
Pooling
“Basin”
“Pikachu”
“Pikachu”
“Paper”
“Knife”
Context-Aware Feature Extraction
Attention-Driven Feature Aggregration
“Pikachu”
RGB
Mean
RGB
Attention
Intermediate Feature
Figure 1. We introduce OpenInsGaussian, a simple yet effective approach that improves semantic understanding in 3D Gaussian segmentation.
By leveraging Context-Aware Feature Extraction and Multi-View Attention, OpenInsGaussian addresses two commonly overlooked issues
in existing language understanding pipelines: the loss of contextual information when extracting features from cropped masks and the
inconsistencies introduced by multi-view fusion.
Abstract
Understanding 3D scenes is pivotal for autonomous driving,
robotics, and augmented reality. Recent semantic Gaussian
Splatting approaches leverage large-scale 2D vision models
to project 2D semantic features onto 3D scenes. However,
they suffer from two major limitations: (1) insufficient contex-
tual cues for individual masks during preprocessing and (2)
inconsistencies and missing details when fusing multi-view
features from these 2D models. In this paper, we introduce
OpenInsGaussian, an Open-vocabulary Instance Gaussian
segmentation framework with Context-aware Cross-view Fu-
sion. Our method consists of two modules: Context-Aware
Feature Extraction, which augments each mask with rich
semantic context, and Attention-Driven Feature Aggregation,
which selectively fuses multi-view features to mitigate align-
ment errors and incompleteness. Through extensive exper-
iments on benchmark datasets, OpenInsGaussian achieves
state-of-the-art results in open-vocabulary 3D Gaussian seg-
mentation, outperforming existing baselines by a large mar-
gin. These findings underscore the robustness and generality
of our proposed approach, marking a significant step for-
ward in 3D scene understanding and its practical deploy-
ment across diverse real-world scenarios.
1. Introduction
3D scene understanding has become increasingly crucial for
various real-world applications, such as autonomous driving,
robotics, and augmented reality. As these domains demand
a reliable representation of complex, dynamically chang-
ing environments, researchers have turned to 3D Gaussian
Splatting (3DGS) [14] for high-fidelity scene modelling. In
contrast to traditional point cloud or voxel-based represen-
tations, 3DGS encodes a scene as a collection of learnable
3D Gaussian primitives, achieving impressive visual quality
1
arXiv:2510.18253v1  [cs.CV]  21 Oct 2025

<!-- page 2 -->
and efficient rendering. However, imbuing 3DGS with robust
semantic understanding remains a challenging pursuit.
A natural approach to incorporate semantics into 3DGS
involves harnessing the power of large-scale 2D vision-
language models (VLMs) and projecting 2D VLMs embed-
dings onto the 3D scene [15, 27, 30, 35], thereby augment-
ing each Gaussian primitive with semantic features. While
these works have demonstrated promising results, they face
two critical limitations. First, contextual information loss
occurs when object segments are cropped for feature extrac-
tion, obscuring important cues in the broader image context,
especially for small or partially occluded objects. Second,
multi-view feature inconsistency arises due to variations in
lighting, occlusions, and camera perspectives, causing differ-
ent views of the same object to yield misaligned embeddings.
Simply averaging these features usually propagates noise
and degrades overall performance (Fig. 1).
In this paper, we introduce OpenInsGaussian, an open-
vocabulary instance Gaussian segmentation framework
equipped with a novel Context-aware Cross-view Fusion
mechanism to address these limitations. Our approach is built
around two key components. First, we propose a Context-
Aware Feature Extraction module that pools features directly
from the frozen CLIP backbone using mask-based queries,
preserving spatial context and leveraging the global semantic
knowledge of the model [34]. This design avoids discarding
critical visual cues during preprocessing, enabling richer se-
mantic representations of target objects. Second, we devise
an Attention-Driven Feature Aggregation strategy that se-
lectively weights feature contributions from multiple views
based on their semantic consistency. By down-weighting
noisy or misaligned embeddings, our method produces more
accurate and robust 3D semantic reconstructions.
OpenInsGaussian is evaluated on open-vocabulary 3D
gaussian segmentation benchmarks and demonstrate that it
substantially outperforms prior works, establishing a new
state-of-the-art in 3DGS-based instance segmentation. Our
experiments further show that incorporating spatial context
and adapting attention weights across views significantly
enhances the quality and robustness of the reconstructed
3D scenes. This approach marks an important milestone in
bridging the gap between high-fidelity 3D representations
and semantic understanding, paving the way for broader
deployment of 3D vision models in diverse real-world sce-
narios.
The key contributions of our work are listed as follows:
• We introduce a novel 3D Gaussian Splatting approach,
OpenInsGaussian, designed to handle open-vocabulary
instance-level segmentation by effectively leveraging
large-scale vision-language models.
• To address contextual information loss, we propose a mask-
based feature extraction strategy that pools features di-
rectly from the frozen CLIP backbone.
• We develop an attention-driven feature aggregation module
that adaptively weighs multi-view features based on their
semantic consistency, mitigating feature inconsistency and
enhances overall 3D semantic representations.
• OpenInsGaussian significantly outperforms prior work in
both accuracy and robustness, thus setting a new state of
the art in 3D instance gaussian segmentation.
2. Related Work
2.1. 3D Representation
3D representation is fundamental for novel view synthe-
sis, 3D reconstruction, and scene understanding. Traditional
explicit methods, such as voxels, point clouds, and meshes,
directly encode geometry but suffer from high memory usage
or lack connectivity. Neural implicit representations, particu-
larly Neural Radiance Fields (NeRF) [24], have revolution-
ized novel view synthesis by mapping 3D coordinates to radi-
ance and density values. While NeRF achieves high-quality
rendering, its per-scene optimization is computationally ex-
pensive, leading to slow training and inference. Efforts to
accelerate NeRF include hybrid methods incorporating voxel,
tri-plane or hash grids [12, 26, 29, 31], reducing network
complexity while maintaining visual fidelity. 3DGS [14] fur-
ther advances neural 3D representation by replacing volume
rendering with fast differentiable rasterization. Instead of
ray-marching, 3DGS models scenes as a set of learnable 3D
Gaussians projected onto the image plane and blended via a
splatting operation, enabling real-time rendering while pre-
serving high-quality reconstruction. While 3DGS has been
extended to extensive field like dynamic scenes [23, 36] and
generative modeling [6, 32, 38], these methods focus on vi-
sual quality rather than semantic understanding. In contrast,
our approach focusing on integrates language embeddings
into 3D Gaussian, enabling open-vocabulary 3D point-level
understanding and bridging the gap between high-fidelity
scene representation and semantic reasoning.
2.2. 3D Class Agnostic Segmentation
The Segment Anything Model (SAM) [17] has demon-
strated strong zero-shot segmentation capabilities in 2D
images, making it a foundational tool for various com-
puter vision tasks [7, 13, 41]. Given its success, recent re-
search has explored extending SAM-based segmentation
to 3D [4, 5, 8, 16, 25, 37]. More specifically, research
has explored 3D Gaussian Splatting for scene segmenta-
tion. SAGA [4] applies contrastive learning with SAM-
generated masks, using a trainable MLP to project instance
features into a low-dimensional space, thereby reducing in-
consistencies. Gaussian-Grouping [37] leverages a zero-shot
tracker [7] to improve mask consistency, jointly reconstruct
and segment 3DGS. ClickGaussian [8] introduces Global
Feature-Guided Learning, clustering global feature candi-
2

<!-- page 3 -->
dates derived from noisy 2D segments across multiple views.
More reviews of existing methods that integrate
NeRF [24] with SAM for class-agnostic segmentation is
provided in Appendix A.1.
2.3. 3D Open-Vocabulary Understanding
The emergence of large language models [2, 10] has driven
rapid advancements in vision-language learning. CLIP [28]
demonstrated that contrastively pretraining dual-encoder
models on large-scale image-text pairs enables cross-modal
alignment, achieving strong zero-shot performance on vari-
ous downstream tasks.
Integrating 2D vision-language models into 3D represen-
tation learning has significantly enhanced open-vocabulary
scene understanding. Early efforts in this direction in-
clude Distilled Feature Fields [18] and Neural Feature Fu-
sion Fields [33], which distilled multi-view LSeg [20] or
DINO [3] features into NeRF-based representations. Se-
mantic NeRF [42] further embedded semantic information
into NeRFs to enable novel semantic view synthesis, while
LERF [15] pioneered the integration of CLIP features within
NeRF for open-vocabulary 3D queries. Later approaches,
such as 3D-OVS [21], combined CLIP and DINO features to
improve 3D open-vocabulary segmentation. Nested Neural
Feature Fields [1] extend feature field distillation by introduc-
ing hierarchical supervision that assigns different dimensions
of a single high-dimensional field to encode scene proper-
ties at multiple granularities, leveraging 2D class-agnostic
segmentation and CLIP embeddings for coarse-to-fine open-
vocabulary 3D scene understanding.
Despite their effectiveness in leveraging VLMs for 3D
semantic reasoning, NeRF-based methods still suffer from
computationally expensive volume rendering. To address this
limitation, recent studies have explored 3D Gaussian Splat-
ting as a more efficient alternative for real-time neural scene
representation. LEGaussians [30] introduced uncertainty-
aware semantic attributes to 3D Gaussians, aligning ren-
dered semantic maps with quantized CLIP and DINO fea-
tures. LangSplat [27] employed a scene-specific language
autoencoder to encode object semantics, enhancing feature
separability in rendered images. Feature3DGS [43] pro-
posed a parallel N-dimensional Gaussian rasterizer for high-
dimensional feature distillation. Unlike LangSplat [27] and
LEGaussians [30], which directly learn multi-view quantized
semantic features onto Gaussians, OpenGaussian [35] intro-
duces a novel pipeline. It first applies a contrastive learning
scheme to generate class agnostic 3D masks, then bound
multi-view CLIP features to objects, enabling point-level
open-vocabulary understanding.
These advancements highlight the shift from NeRF-based
methods to more computationally efficient Gaussian splat-
ting approaches, offering new possibilities for scalable and
real-time 3D open-vocabulary scene understanding.
3. Method
3.1. Preliminary: 3D Gaussian Splatting
A 3D scene is represented as a set of 3D Gaussians, each
parameterized by its position µ ∈R3, scale S ∈R3, rotation
R ∈R4, opacity σ, and color features c, which are encoded
using spherical harmonics (SH).
For rendering, 3D Gaussian Splatting [14] follows a
splatting-based pipeline, where each Gaussian is projected
onto the 2D image space according to the camera’s world-
to-frame transformation. Gaussians overlapping at the same
pixel coordinates (x, y) are blended in depth order with
opacity-weighted accumulation to compute the final pixel
color:
  c_ {
x
,y}
 = \
sum
 
_{i
 \ i n N} c_i \alpha _i \prod _{j=1}^{i-1} (1 - \alpha _j), 
(1)
where αi represents the opacity of the i-th Gaussian, and ci
is its associated color. This process enables real-time differ-
entiable rasterization, allowing the optimization of Gaussian
parameters through a reconstruction loss computed against
ground truth images. The flexibility and efficiency of 3D
Gaussians have made it a promising approach for high-
fidelity 3D reconstruction and novel view synthesis.
3.2. Method Overview
Given a set of input images, our approach constructs a 3D
Gaussian representation while integrating language embed-
dings to enable open-vocabulary scene understanding. The
objective is to associate each Gaussian with meaningful lan-
guage features, allowing for robust semantic comprehension.
To achieve this, we design a structured training pipeline
comprising several key stages.
We begin by initializing a set of pretrained 3D Gaussians
and utilizing SAM [17] to generate object masks from the
input images. To extract robust language features, we intro-
duce a context-aware feature extraction method based on
CLIP [28] (Sec. 3.3). This method combines fine-grained,
distinct local features with context-aware global features,
ensuring that each mask is embedded with rich semantic
information.
To achieve class-agnostic 3D instance segmentation, we
follow OpenGaussian [35] to train instance features using
SAM masks within a contrastive learning framework. Then
discretize the pretrained Gaussians based on these learned
instance features using a hierarchical coarse-to-fine cluster-
ing strategy (Sec.3.4). This process generates class-agnostic
3D masks, forming a structured segmentation of the scene.
Finally, we refine the association between 3D Gaussians
and language features by applying a similarity-driven adap-
tive attention mechanism (Sec. 3.5). This mechanism lifts
multi-view 2D language embeddings into the 3D space, en-
suring that each Gaussian is effectively aligned with se-
mantic representations. By integrating these components,
3

<!-- page 4 -->
Clip
SAM
Crop
Mask Pooling
Clip
Contrastive Loss
Discretization
Pretrained 3DGS
context
F
local
F
fuse
F
fuse
F
fuse
F
fuse
F
object
F
Preprocessing
Preprocessing
Segmentation
Segmentation
Instance Feature Learning
Instance Feature Learning
Langauge Feature Aggregation
Langauge Feature Aggregation
Instance Feature
RGB
Intermediate Feature
Object Masks
Class-Agnostic
3D Segmentations
Associated 
2D Masks
Attention
Multi-View Fused
Language Features
Single-View Fused
Language Feature
Single-View Context-Aware
Language Feature
Single-view Local
Language Feature
Object Final
Language Feature
Figure 2. Given a set of images and pretrained 3D Gaussians, our pipeline consists of four key steps: (1) We preprocess the input images
using SAM [4] to obtain object masks and fuse local and context-aware CLIP [28] encoded language features to generate context-aware
language embeddings for each mask. (2) The SAM masks are then used to train class-agnostic instance features for each 3D Gaussian. (3)
We discretize the 3D Gaussians by clustering instance features using a hierarchical approach. (4) Finally, we employ an attention-driven
feature aggregation method to associate language embeddings with the segmented instances, enabling open-vocabulary understanding.
our method enables robust open-vocabulary understanding
within a 3D Gaussian Splatting framework.
3.3. Context-Aware Feature Extraction
In this section, we aim to extract mask-level language fea-
tures from an image batch for open-vocabulary 3D scene un-
derstanding. Given a batch of images  \{ I _ t  \ mi d t = 1, 2, ..., T\} ,
our goal is to extract mask-level language features while
preserving contextual information.
Each image  I _ t \in \mathbb {R}^{3 \times H \times W} is a standard RGB image,
where  H and  W denote the height and width of the image.
We use SAM [17] to generate instance masks, represented
as a mask matrix: Mt ∈R1×H×W , where each pixel value
in  M_t indicates the instance label of that region. The set of
binary masks for each image is: {Bt,1, Bt,2, ..., Bt,Nt} =
{I(Mt = i) | i ∈Mt}. Each binary mask  B_{t,i} is a boolean
indicator map where pixels belonging to instance  i are set to
1, and all others are 0. We extract language embeddings for
all masks in a given image through below two complemen-
tary methods.
3.3.1. Local Feature Extraction
The conventional approach [27] extracts features by cropping
each mask region and passing it through the CLIP image
encoder:
  F_{\tex t  {local}} ( t) = \ { V _{ \t ext {clip}}(I_t \odot B_{t,i}) \mid i = 1, 2, ..., N_t\}, 
(2)
Reference Image
Cropped image
Figure 3. Example of a cropped image patch in local feature extrac-
tion on Scannet dataset.
where  \odot denotes element-wise multiplication, effectively
cropping out object regions.  V_{\text {clip}} is the entire CLIP image
encoding pipeline, which takes an image  I and outputs a
global image-level feature in the shared vision-language em-
bedding space. Cropping isolates objects from their surround-
ings, stripping away crucial spatial context. Many objects,
especially in indoor scenes, rely on background information
for accurate identification. Also, CLIP is trained to process
whole images, and extracting individual object features from
cropped patches disrupts the spatial relationships learned in
the model. This results in a potential loss of semantic fidelity.
Figure 3 illustrates how cropped patches often lack sufficient
contextual cues, leading to ambiguous object recognition.
For example a cropped patch of a “floor” might be indistin-
4

<!-- page 5 -->
guishable without its surrounding environment.
3.3.2. Context-Aware Feature Extraction
CLIP’s image encoder processes entire images and extracts
spatial feature maps at intermediate layers. However, direct
cropping bypasses these feature maps, forcing the model
to rely solely on the global image representation. This mis-
alignment can degrade feature quality, particularly for fine-
grained open-vocabulary understanding.
To overcome these limitations, we propose a context-
aware feature extraction strategy that preserves spatial con-
text while ensuring robust feature alignment. Instead of di-
rectly cropping images, we leverage CLIP’s intermediate
feature maps before the fully connected layer. These feature
maps retain rich contextual information about object sur-
roundings, which is crucial for fine-grained open-vocabulary
understanding.
Rather than encoding each object independently, we ex-
tract spatial feature maps from CLIP’s vision encoder and
aggregate features within each mask. By cropping the feature
maps instead of the raw image, we ensure that object features
are learned within the broader scene context rather than in
isolation. This approach enables the model to capture finer
object details while maintaining surrounding information,
leading to more precise and semantically meaningful feature
representations. Mathematically, the process define as:
  F_{\text {img}}(t) = V_{\text {encoder}}(I_t) \in \mathbb {R}^{D' \times H' \times W'}, g
(3)
where  D' , H ' , W' are the spatial dimensions of the down-
sampled feature map,  V_{\text {encoder}} is the convolutional backbone
or transformer feature extractor within the CLIP model,
which processes an image into a spatial feature map before
the final pooling and projection layers
Since the segmentation mask  M_t is at the original image
resolution, we resize it to match the feature map size:
  
M ' _t = \text  {R e si z e}(M_t,  H', W') \in \mathbb {R}^{1 \times H' \times W'}. 
(4)
For each object instance  i , we generate a binary mask:
  B_ { t,i }
 =  \ m ath bb {I}  (M'_t = i) \in \{0,1\}^{H' \times W'}. 
(5)
We then perform masked average pooling over the feature
map and pass the pooling feature through CLIP’s fully con-
nected layer, producing the final image-text latent space
feature:
  F_{\te xt  
{
mas k}}(t, i) = \frac {\sum _{h,w} B_{t,i}(h,w) F_{\text {img}}(t, :, h, w)}{\sum _{h,w} B_{t,i}(h,w)}, g
(6)
  F_{con t ext} = \{V_{\te xt { p r oj }} (F_{ \text {mask}}(t, i)) \mid i = 1, 2, ..., N_t\}, 
(7)
where  V_{\text {proj}} is the final projection layer of CLIP that maps
extracted image features into the shared vision-language
embedding space. This ensures that the extracted features
retain spatial information while maintaining alignment with
the segmentation mask.
Local: “Bed”
Fuse: “Sink”
Reference Image
Masked Patch Language Embedding
Local: “Desk”
Fuse: “Counter”
Local: “Desk”
Fuse: “Floor”
Local: “Counter”
Fuse: “Desk”
Figure 4. Example of language embedding with different feature
extraction strategies on Scannet dataset.
3.3.3. Feature Fusion Strategy
While our method effectively captures global context, CLIP
models are not pre-trained with precise object masks, leading
to potential misalignment in extracted features. To mitigate
this, we employ a geometric ensemble strategy to fuse local
and context-aware CLIP features, ensuring more accurate
feature alignment.
  F_{\te x t  {fuse}}(t) =  \ a lp h a \cdot F_{\text {context}}(t) + (1 - \alpha ) \cdot F_{\text {local}}(t), 
(8)
where  F_{\text {context}}(t) is derived from context-aware feature ex-
traction, and  F_{\text {local}}(t) comes from local feature extraction.
The parameter  \ alp ha \in [0,1] balances their contributions,
 F_{f u se} \in \mathbb {R}^{D \times N_t} represents the extracted language fea-
ture embeddings for all masks in an image. By integrating
context-aware feature extraction with geometric ensemble
fusion, our approach effectively mitigates the challenges
posed by missing contextual information and multi-view
inconsistencies, leading to a more robust and accurate 3D
language field. Examples in Figure 4 demonstrate that our
fused context-aware language features achieve greater accu-
racy compared to the initial local feature extraction. Here,
we present the query text embeddings from the ScanNet test
classes, selecting the one with the highest similarity to the
masked language features.
3.4. Class-Agnostic Segmentation
After obtaining segmentation masks from SAM and the
corresponding mask-level language embeddings, we follow
5

<!-- page 6 -->
OpenGaussian [35] to segment the scene into class-agnostic
clusters. We first train instance features for 3D Gaussians
using SAM-generated masks and enforce multi-view con-
sistency through contrastive learning, ensuring that Gaus-
sians within the same mask instance are pulled closer while
those from different instances are pushed apart. Next, we
apply two-level codebook feature discretization, a coarse-
to-fine clustering strategy leveraging both spatial positions
and learned instance features to segment objects while pre-
serving geometric integrity. The segmentation is further re-
fined iteratively using pseudo-ground-truth features. Finally,
the 3D scene is segmented into class-agnostic clusters. For
more details on contrastive learning of instance features and
two-level codebook feature discretization, please refer to
(Sec.A.2, Sec.A.3).
3.5. Attention Driven Feature Aggregation
After codebook discretization, the scene is segmented into
multiple class-agnostic objects. We now binding language
features to the objects, started by associating 3d object to
multiview 2d masks. The association between 3D Gaussian
instances and multi-view 2D masks is established by select-
ing the highest IoU mask for each rendered 3D instance and
refining the assignment using instance feature similarity, en-
suring robust alignment. Please refer to (Sec.A.4) for more
details.
Prior work [35] overlooked inconsistencies introduced
by varying camera viewpoints. The same object can exhibit
significant semantic variation across different views, mak-
ing simple feature averaging unreliable due to perspective-
induced discrepancies. We employ a self-attention-based
method for multi-view semantic feature aggregation to asso-
ciate language embeddings with these segments, ensuring
consistent feature binding to each object. Specifically, we
propose a similarity-driven attention mechanism for multi-
view CLIP feature fusion. Our method adaptively weights
features using cosine similarity, enhancing robustness and
interpretability while maintaining computational efficiency.
Instead of learning query-key interactions, we directly com-
pute cosine similarity between each feature and a reference
feature (mean feature) to guide the attention process. This
ensures that semantically similar features receive higher
weights while inconsistent or occluded views are down-
weighted.
Given a set of multi-view CLIP features from section 3.3
Ffuse = {F1, F2, ..., Ft}, we first compute a reference fea-
ture as the mean feature across views:
  F_{ \ t
e
x
t
 {m
ean}} = \frac {1}{N} \sum _{i=1}^{N} F_i. 
(9)
Next, we compute the cosine similarity between each feature
and the mean feature:
  S
_i  = \fr
ac {F_i \cdot F_{\text {mean}}}{\|F_i\| \|F_{\text {mean}}\|}. 
(10)
To obtain attention weights, we apply softmax normalization:
  W _i = \text {softmax}(S_i). 
(11)
Finally, the fused feature is computed as a weighted sum:
  F_{\t e
x
t
 {o
bject}} = \sum _{i=1}^{N} W_i F_i. 
(12)
where Wi ensures that views with higher semantic consis-
tency contribute more to the final feature representation.
Compared to standard self-attention, our method offers
several advantages. It provides robustness to viewpoint vari-
ability by explicitly weighting features based on cosine simi-
larity, ensuring that only semantically consistent views con-
tribute significantly, while occluded or noisy views receive
lower attention. Also, it improves interpretability, as unlike
self-attention, where attention weights are learned implicitly,
our approach uses explicit similarity metrics, making it more
interpretable. Our method enhances computational efficiency
by reducing the complexity from O(N 2D) in self-attention
to O(ND), making it significantly more scalable. Finally,
our approach requires no training, as it is fully unsupervised
and can be applied directly to CLIP features without addi-
tional fine-tuning. This is particularly useful for scenarios
where labeled data is scarce or training a large transformer
model is impractical.
4. Experiment
4.1. Open-Vocabulary Query of Point Cloud in 3D
Space
Task. Given a set of open-vocabulary text queries, the goal of
this task is to find the matching point cloud by computing the
cosine similarity between the text features and the Gaussian
features. Each Gaussian is assigned the category of the text
query with the highest similarity, thereby enabling open-
vocabulary point cloud understanding. While the method
theoretically supports arbitrary text input, for quantitative
evaluation against the annotated ground truth point cloud,
we use pre-defined category names as text queries.
Dataset and Metrics. Following OpenGaussian [35], we
evaluate our method on ScanNetV2 [9], which provides
posed RGB images from video scans, reconstructed point
clouds, and ground truth 3D point-level semantic labels.
We consider 19, 15, and 10 category subsets from Scan-
Net as text queries and assign the closest matching text to
each Gaussian to compute the mean Intersection over Union
(mIoU) and mean Accuracy (mAcc). To ensure consistency,
we use the provided point clouds for initialization and freeze
6

<!-- page 7 -->
Reference Mesh
“Sofa”
“Floor”
Reference Mesh
“Desk”
“Carbinet”
Reference Mesh
“Table”
“Chair”
Reference Mesh
“Table”
“Chair”
Reference Mesh
“Chair”
“Sofa”
Reference Mesh
“Chair”
“Sofa”
Figure 5. Open-vocabulary 3D point cloud understanding on the Scannet dataset.
Methods
19 Classes
15 Classes
10 Classes
mIoU ↑
mAcc. ↑
mIoU ↑
mAcc. ↑
mIoU ↑
mAcc. ↑
LangSplat [27]
3.78
9.11
5.35
13.20
8.40
22.06
LEGaussians [30]
3.84
10.87
9.01
22.22
12.82
28.62
OpenGaussian [35]
24.73
41.54
30.13
48.25
38.29
55.19
Ours
37.50
54.38
38.14
55.30
51.42
69.15
Table 1. Performance of 3D point cloud semantic segmentation on
the ScanNet dataset based on text query.
the coordinates of the Gaussians during training, disabling
the densification process of 3DGS. The training images are
extracted every 20 frames from the video scans, and evalua-
tion is conducted on 10 randomly selected scenes.
Baseline. We primarily compare our method with re-
cent Gaussian-based approaches, including LangSplat [27],
LEGaussians [30], and OpenGaussian [35].
Results. The quantitative results are presented in Table 1,
where our method achieves state-of-the-art performance
across 19, 15, and 10 categories, significantly outperform-
ing Gaussian-based approaches. The poor performance of
LangSplat and LEGaussians can be attributed to ambigu-
ous Gaussian feature learning and suboptimal language fea-
ture distillation. OpenGaussian, despite its improvements,
is constrained by the nature of the ScanNet dataset, where
training images are extracted from video scans. These im-
ages often suffer from motion blur, leading to unclear ob-
ject boundaries and diminished local feature quality. As a
result, OpenGaussian accumulates errors in multi-view se-
mantic fusion, further reducing its effectiveness. In contrast,
our method leverages context-aware preprocessing and an
attention-driven feature fusion module, ensuring more robust
semantic binding even under challenging input conditions.
Figure 5 visualize the effectiveness of our approach.
Methods
mIoU ↑
mAcc. ↑
LangSplat [27]
9.66
12.41
LEGaussians [30]
16.21
23.82
OpenGaussian [35]
38.36
51.43
Ours
42.62
62.11
Table 2. Performance of open vocabulary 3D object selection and
rendering on Lerf dataset. Accuracy is measured by mAcc@0.25
4.2. Open-Vocabulary Selection and Rendering in
3D Space
Task. Given an open-vocabulary text query, we extract its
text feature using CLIP and compute the cosine similar-
ity between this feature and the language features of each
Gaussian. Based on the similarity scores, we select the most
relevant 3D Gaussians and render them into multi-view 2D
images using the rasterization pipeline of 3DGS. This setup
enables us to evaluate the effectiveness of our method in
selecting 3D objects that match the input text queries.
Dataset and Metrics. Following LangSplat [27] and Open-
Gaussian [35], we conduct experiments on the LeRF-OVS
dataset [15]. After rendering the selected 3D objects into
2D images, we compute the mIoU and mAcc by comparing
the rendered images with the corresponding ground truth 2D
object masks.
Baseline. Since only Gaussian-based methods possess both
3D point-level perception and rendering capabilities, we pri-
marily compare our method against LangSplat [27], LEGaus-
sians [30], and OpenGaussian [35].
Results. Quantitative results are presented in Table 2.
LangSplat [27] and LEGaussians [30] fail to accurately
select 3D Gaussians when rendering text-query-relevant
3D Gaussians onto 2D images for evaluation. Compared
to OpenGaussian [35], despite following a similar class-
agnostic segmentation process, our method achieves superior
7

<!-- page 8 -->
Reference Image
OpenGaussian
Ours
Query Text
“Ketchup”
“Pikachu”
“Napkin”
“Corn”
Figure 6. Open-vocabulary 3D object selection and rendering on
the LeRF dataset.
Case
Local Feature
Global Feature
mIoU ↑
mAcc. ↑
#1
✓
40.58
57.93
#2
✓
50.58
67.48
#3
✓
✓
51.21
67.54
Table 3. Performance of 3D point cloud understanding on ScanNet
using different feature extraction strategies, evaluated by mIoU and
mAcc across 10 classes
performance in language understanding, setting a new state-
of-the-art benchmark. The performance advantage of our ap-
proach stems from two key aspects. First, the context-aware
feature extraction process ensures that each mask retains
crucial surrounding information, enhancing semantic con-
sistency. Second, the attention-driven aggregation strategy
mitigates the effects of multi-view inconsistencies, a limita-
tion present in OpenGaussian [35]. The qualitative results in
Figure 6 further illustrate these improvements.
4.3. Ablation Study
Effect of Context-Aware Feature Extraction The ablation
study results in Table 3 highlight the contributions of local
and context-aware feature extraction. Case #1, using only
local features, provides a strong baseline but lacks global
context, leading to semantic inaccuracies. Case #2, relying
solely on context-aware features, improves accuracy by pre-
serving scene-wide contextual relationships. Case #3, which
combines both, achieves the best performance, validating our
geometric ensemble fusion strategy. These results confirm
that local features enhance fine-grained distinctions, while
context-aware features provide contextual awareness, mak-
ing our approach more robust for 3D language field learning.
Effect of Attention-Driven Feature Aggregation The re-
sults in Table 4 demonstrate the effectiveness of our attention-
driven feature aggregation. Case #1, which does not use
Case
Combined
Fea-
ture
Attention Aggre-
gation
mIoU ↑
mAcc. ↑
#1
40.71
57.60
#2
✓
41.98
60.52
#3
✓
41.59
58.50
#4
✓
✓
42.62
62.11
Table 4. Performance of Open Vocabulary query on LeRF-OVS
with Attention-Based Feature Aggregation, evaluated by mIoU and
mAcc. Accuracy is measured by mAcc@0.25
combined features or attention aggregation, provides a base-
line performance. Case #2, which incorporates combined
features, shows an improvement in both mIoU and mAcc, in-
dicating that integrating multi-view CLIP features enhances
feature robustness. Case #3, which applies attention aggrega-
tion alone, also leads to a performance gain, highlighting the
benefits of similarity-driven weighting. Case #4, where both
techniques are combined, achieves the highest scores, con-
firming that attention-driven fusion effectively reduces incon-
sistencies in multi-view feature aggregation. These results
validate that our proposed method enhances semantic coher-
ence while maintaining fine-grained distinctions, making it
a robust approach for 3D open-vocabulary understanding.
5. Limitation
Since our pipeline segments objects first and then binds lan-
guage features, we face a bottleneck at the agnostic segmen-
tation stage. Additionally, using the mean as a reference in
multi-view attention may be suboptimal when the dominant
views contain weak semantic features. Our attention-driven
aggregation serves more as a feature purification process,
rather than a selection process as commonly used in point
cloud-based approaches.
6. Conclusion
Lifting 2D large-model semantic features to 3D remains
challenging due to inconsistencies in multi-view alignment
and semantic drift. Existing methods struggle with feature
degradation and ambiguous 2D-3D associations, limiting
their effectiveness at the 3D point level. To address this, we
introduce a context-aware feature extraction strategy that pre-
serves both local details and context-aware semantics, ensur-
ing richer feature representations. Additionally, we propose
attention-driven feature aggregation, leveraging similarity-
driven adaptive attention to refine multi-view fusion and
improve semantic consistency.
Our method significantly improves open-vocabulary 3D
understanding, achieving state-of-the-art performance across
multiple benchmarks. By strengthening the connection be-
tween 2D and 3D semantic spaces, we enable more robust
and scalable 3D scene interpretation.
8

<!-- page 9 -->
Acknowledgment
Tongliang Liu is partially supported by the following
Australian Research Council projects: FT220100318,
DP220102121,
LP220100527,
LP220200949,
and
IC190100031.
References
[1] Yash Bhalgat, Iro Laina, Joao F Henriques, Andrew Zisser-
man, and Andrea Vedaldi. N2f2: Hierarchical scene under-
standing with nested neural feature fields. In European Con-
ference on Computer Vision, pages 197–214. Springer, 2024.
3
[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah,
Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan,
Pranav Shyam, Girish Sastry, Amanda Askell, et al. Lan-
guage models are few-shot learners. In Advances in Neural
Information Processing Systems (NeurIPS), 2020. 3
[3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou,
Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-
ing properties in self-supervised vision transformers. In Pro-
ceedings of the IEEE/CVF International Conference on Com-
puter Vision (ICCV), pages 9650–9660, 2021. 3, 11
[4] Jingwen Cen, Jiemin Fang, Chuan Yang, Lingxi Xie, Xi-
aopeng Zhang, Wenguan Shen, and Qi Tian. Segment any 3d
gaussians. arXiv preprint arXiv:2312.00860v1, 2024. 2, 4
[5] Jingwen Cen, Ziyao Zhou, Jiemin Fang, Wenguan Shen,
Lingxi Xie, Dongdong Jiang, Xiaopeng Zhang, and Qi Tian.
Segment anything in 3d with nerfs. Advances in Neural Infor-
mation Processing Systems, 36, 2024. 2, 11
[6] Zilong Chen, Feng Wang, and Huaping Liu. Text-to-3d using
gaussian splatting. arXiv preprint, arXiv:2309.16585, 2023.
2
[7] Hongkai Cheng, Seongwon Oh, Brian Price, Alexander
Schwing, and Joon-Young Lee. Tracking anything with de-
coupled video segmentation. In Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pages
1316–1326, 2023. 2
[8] Seokhun Choi, Hyeonseop Song, Jaechul Kim, Taehyeong
Kim, and Hoseok Do. Click-gaussian: Interactive segmenta-
tion to any 3d gaussians. In European Conference on Com-
puter Vision, pages 289–305. Springer, 2024. 2, 11
[9] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 5828–5839, 2017. 6, 12
[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. BERT: Pre-training of deep bidirectional trans-
formers for language understanding. In Proceedings of the
North American Chapter of the Association for Computa-
tional Linguistics (NAACL), 2019. 3
[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale. In Proceedings of
the International Conference on Learning Representations
(ICLR), 2021. 12
[12] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 5501–5510, 2022. 2
[13] Shanghua Gao, Zhijie Lin, Xingyu Xie, Pan Zhou, MingMing
Cheng, and Shuicheng Yan. Editanything: Empowering un-
paralleled flexibility in image editing and generation. In ACM
Multimedia (ACM MM), Demo Track, 2023. 2
[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and
George Drettakis. 3d gaussian splatting for real-time radiance
field rendering. ACM Transactions on Graphics, 42(4):1–14,
2023. 1, 2, 3, 12
[15] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo
Kanazawa, and Matthew Tancik. Lerf: Language embedded
radiance fields. In Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), pages 19729–19739,
2023. 2, 3, 7, 12
[16] Chung Min Kim, Mingxuan Wu, Justin Kerr, Ken Goldberg,
Matthew Tancik, and Angjoo Kanazawa. Garfield: Group any-
thing with radiance fields. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 21530–21539, 2024. 2, 11
[17] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. In Proceedings of the IEEE/CVF international confer-
ence on computer vision, pages 4015–4026, 2023. 2, 3, 4,
12
[18] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitz-
mann. Decomposing nerf for editing via feature field distilla-
tion. In Advances in Neural Information Processing Systems
(NeurIPS), pages 23311–23330, 2022. 3
[19] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick
Haffner. Gradient-based learning applied to document recog-
nition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
12
[20] Boyi Li, Kilian Q. Weinberger, Serge Belongie, Vladlen
Koltun, and Rene Ranftl. Language-driven semantic seg-
mentation. In International Conference on Learning Repre-
sentations (ICLR), 2022. 3
[21] Kunhao Liu, Fangneng Zhan, Jiahui Zhang, Muyu Xu,
Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt,
Eric Xing, and Shijian Lu.
Weakly supervised 3d open-
vocabulary segmentation. In Advances in Neural Information
Processing Systems (NeurIPS), 2023. 3
[22] Stuart Lloyd.
Least squares quantization in pcm.
IEEE
transactions on information theory, 28(2):129–137, 1982.
11
[23] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva
Ramanan. Dynamic 3d gaussians: Tracking by persistent
dynamic view synthesis. arXiv preprint, arXiv:2308.09713,
2023. 2
[24] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
9

<!-- page 10 -->
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021. 2,
3
[25] Arian Mirzaei, Tovi Aumentado-Armstrong, Konstantinos G.
Derpanis, Jonathan Kelly, Marcus A. Brubaker, and Anton
Levinshtein. Spin-nerf: Multiview segmentation and percep-
tual inpainting with neural radiance fields. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 20669–20679, 2023. 2, 11
[26] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander
Keller. Instant neural graphics primitives with a multiresolu-
tion hash encoding. ACM transactions on graphics (TOG),
41(4):1–15, 2022. 2
[27] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and
Hanspeter Pfister. Langsplat: 3d language gaussian splatting.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 20051–20060, 2024. 2,
3, 4, 7, 12
[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning, pages
8748–8763. PmLR, 2021. 3, 4
[29] Christian Reiser, Rick Szeliski, Dor Verbin, Pratul Srinivasan,
Ben Mildenhall, Andreas Geiger, Jon Barron, and Peter Hed-
man. Merf: Memory-efficient radiance fields for real-time
view synthesis in unbounded scenes. ACM Transactions on
Graphics (TOG), 42(4):1–12, 2023. 2
[30] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua
Guan. Language embedded 3d gaussians for open-vocabulary
scene understanding. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
5333–5343, 2024. 2, 3, 7, 12
[31] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel
grid optimization: Super-fast convergence for radiance fields
reconstruction. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
5459–5469, 2022. 2
[32] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang
Zeng. Dreamgaussian: Generative gaussian splatting for effi-
cient 3d content creation. arXiv preprint, arXiv:2309.16653,
2023. 2
[33] Vadim Tschernezki, Iro Laina, Diane Larlus, and Andrea
Vedaldi. Neural feature fusion fields: 3d distillation of self-
supervised 2d image representations. In International Con-
ference on 3D Vision (3DV), pages 443–453. IEEE, 2022.
3
[34] Zhuowen Tu, Zheng Ding, and Jieke Wang. Open-vocabulary
uni-versal image segmentation with maskclip. In ICML, 2023.
2
[35] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao
Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding,
Jingdong Wang, et al. Opengaussian: Towards point-level 3d
gaussian-based open vocabulary understanding. Advances
in Neural Information Processing Systems, 37:19114–19138,
2025. 2, 3, 6, 7, 8, 11, 12
[36] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin.
Deformable 3d gaussians for
high-fidelity monocular dynamic scene reconstruction. arXiv
preprint, arXiv:2309.13101, 2023. 2
[37] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaus-
sian grouping: Segment and edit anything in 3d scenes. In
European Conference on Computer Vision, pages 162–179.
Springer, 2024. 2
[38] Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng
Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussian-
dreamer: Fast generation from text to 3d gaussian splatting
with point cloud priors. arXiv preprint, arXiv:2310.08529,
2023. 2
[39] Haiyang Ying, Yixuan Yin, Jinzhi Zhang, Fan Wang, Tao
Yu, Ruqi Huang, and Lu Fang. Omniseg3d: Omniversal 3d
segmentation via hierarchical contrastive learning. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 20612–20622, 2024. 11
[40] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-
Chieh Chen. Convolutions die hard: Open-vocabulary seg-
mentation with single frozen convolutional clip. Advances
in Neural Information Processing Systems, 36:32215–32234,
2023. 12
[41] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin,
Wenjun Zeng, and Zhibo Chen.
Inpaint anything: Seg-
ment anything meets image inpainting.
arXiv preprint
arXiv:2304.06790, 2023. 2
[42] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and An-
drew J. Davison. In-place scene labelling and understand-
ing with implicit scene representation. In Proceedings of
the IEEE/CVF International Conference on Computer Vision
(ICCV), pages 15838–15847, 2021. 3
[43] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d
gaussian splatting to enable distilled feature fields. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21676–21685, 2024. 3
10

<!-- page 11 -->
A. 3D Class-agnostic Segmentation
A.1. Related Work
SA3D [5] uses user-provided prompts, such as points or
bounding boxs, to generate segmentation masks in refer-
ence views, which are then used to train a neural field
for object segmentation. Similarly, Spin-NeRF [25] em-
ploys a video-based segmenter [3] to generate multi-view
masks. GARField [16] addresses inconsistencies in SAM-
generated masks across different views by incorporating
a scale-conditioned feature field. OmniSeg3D [39] intro-
duces hierarchical contrastive learning to refine 2D SAM
masks into a feature field, achieving fine-grained segmenta-
tion through adaptive cosine similarity thresholds. However,
above methods rely on NeRF-based structures, which im-
pose high computational costs during rendering, limiting
their real-time applicability.
A.2. Contrastive Feature Learning
After obtaining segmentation masks from SAM and the cor-
responding mask-level language embeddings  \{ F _ t
 \ m i d  t = 1, \dots , T\} ,  \{ M _ t  \ m i d  t = 1, \dots , T\} , we learn class-agnostic
instance features by modeling the relationship between 3D
points and 2D pixels. For simplicity, we denote the fused
context-aware language feature  F_{\text {fuse}}(t) as  F_t in the follow-
ing discussion. Following OpenGaussian and other class
agnostic segmentation work [8, 16, 35, 35, 39], we train in-
stance features for 3D Gaussians using segmentation masks.
Each Gaussian is assigned a low-dimensional instance
feature f ∈R6. To enforce multi-view consistency, we ap-
ply contrastive learning, bringing Gaussians within the same
mask instance closer while pushing those from different in-
stances apart. The instance feature map M f ∈R6×H×W is
obtained via alpha-blending, with binary masks Bi defining
object instances:
  \{  B_ 0 ,  B _1,  \dot s  ,  B _ i \} = \{ \mathbb {I} (M = i) \mid i \in M_t \}, 
(13)
To ensure feature consistency within an instance, we com-
pute the mean feature within each mask:
 
 \
b a r { M }
^ f_
i  = \frac {B_i \cdot M^f}{\sum B_i} \in \mathbb {R}^6. 
(14)
The intra-mask smoothing loss encourages all pixels within
an instance to align with their mean feature:
  \
m
a
thc
a
l
 {L
}
_
s =
 \sum _
{i=1 }
^{m} \ s
u m
 
_{h
=
1}^{H} \sum _{w=1}^{W} B_{i,h,w} \cdot \left \| M^f_{:,h,w} - \bar {M}^f_i \right \|^2. 
(15)
To enhance feature distinctiveness across instances, we de-
fine the inter-mask contrastive loss:
  \
m
ath c al
 
{
L}_
c
 
= \frac 
{
1}{ m
( m
+ 1 )
}  
\
sum
 _{i=1}^{m} \sum _{j=1, j \neq i}^{m} \frac {1}{\left \| \bar {M}^f_i - \bar {M}^f_j \right \|^2}, 
(16)
where m is the number of masks, and ¯
M f
i , ¯
M f
j are mean
features of different instances.
These losses ensure cross-view consistency for the same
object while maintaining feature distinctiveness across dif-
ferent objects.
A.3. Two-Level Codebook Feature Discretization
After training instance features on 3D Gaussians, we apply a
two-level coarse-to-fine clustering [35] to segment objects.
At the coarse level, we cluster Gaussians using both 3D
coordinates X  \in \mathbb {R}^{n \times 3} and instance features f  \in \mathbb {R}^{n \times 6},
ensuring spatially aware segmentation:
  \begi n  {spl i t} f \in  \mathbb {R
}^{n \t i mes  6 } ,  X \in
 \ m athbb {R}^{n \times 3} \rightarrow \{C_{\text {coarse}} \in \mathbb {R}^{k_1 \times (6+3)}, \\ I_{\text {coarse}} \in \{1, \dots , k_1\}^n \}, \quad k_1 = 64. \end {split} 
(17)
At the fine level, we further refine clusters using only in-
stance features:
  \beg i n {spl i t} f \in \m
athbb  {R} ^ { n  \times
 6 }  \rightarrow \{C_{\text {fine}} \in \mathbb {R}^{(k_1 \times k_2) \times 6}, \\ I_{\text {fine}} \in \{1, \dots , k_2\}^n \}, \quad k_2 = 10. \end {split} 
(18)
where {C, I} means quantized features and cluster indices at
each level of codebook.
We use K-means clustering [22] at both stages, with
k_1 clusters at the coarse stage and k_ 1  \times k_2 clusters at the
fine stage. This hierarchical approach preserves geometric
integrity, ensuring that spatially unrelated objects are not
grouped together.
During instance feature learning, supervision is limited to
binary SAM masks. In the codebook construction stage, clus-
tered instance features act as pseudo ground truth, replacing
mask-based losses. The new training objective minimizes the
difference between rendered pseudo-ground-truth features
M ^p and quantized features M ^c:
  \ ma t h c al {L}_p = \| M^p - M^c \|_1, 
(19)
This process refines instance segmentation while maintain-
ing feature consistency and geometric structure in the 3D
Gaussian representation.
A.4. Instance-Level 3D-2D Association
To establish a robust link between 3D Gaussian instances
and multi-view 2D masks, we adopt an instance-level 3D-2D
association strategy inspired by OpenGaussian [35]. Unlike
prior methods that require additional networks for compress-
ing language features or depth-based occlusion testing, our
approach retains high-dimensional, lossless linguistic fea-
tures while ensuring reliable associations.
Specifically, given a set of 3D clusters obtained from
the discretization process (Sec. A.3), we render each 3D
instance to individual views, obtaining single-instance maps
11

<!-- page 12 -->
M i ∈R6×H×W . These maps are compared with SAM-
generated 2D masks Bj ∈{0, 1}1×H×W using an Intersec-
tion over Union (IoU) criterion. The SAM mask with the
highest IoU is initially assigned to the corresponding 3D in-
stance. However, to address occlusion-induced ambiguities,
we further refine the association by incorporating feature
similarity.
Instead of relying on depth information for occlusion
testing, we populate the boolean SAM mask Bj with pseudo-
ground truth features, forming a feature-filled mask P j ∈
R6×H×W . We then compute a unified association score:
  S _ {ij} = \te xt { Io U }( \ p i  (M^i), B^j) \cdot (1 - \|M^i - P^j\|_1), 
(20)
where π(·) denotes a binarization operation, ensuring
IoU alignment, while the second term penalizes large feature
discrepancies. The mask with the highest score is then asso-
ciated with the 3D instance, allowing us to bind multi-view
CLIP features effectively to 3D Gaussian objects.
By integrating both geometric alignment and semantic
consistency, our method ensures precise and robust language
embedding associations across multiple views.
B. Implementation Details
B.1. SAM and Clip Backbone
At the preprocessing stage, we utilize SAM-LangSplat,
which is a modified version of SAM [17] for LangSplat [27]
that automatically generates three levels of masks: whole,
part, and sub-part. We select level 3 SAM masks (whole) [17]
and use the ViT-H SAM model checkpoint for segmentation.
For feature extraction, we adopt Convolutional CLIP [19],
a CNN-based variant of CLIP that empirically demonstrates
better generalization than ViT-based CLIP [11] when han-
dling large input resolutions [40] and better intermediate fea-
ture for our global feature extraction. Since competing meth-
ods use the ViT-B/16 checkpoint, we select the ConvNeXt-
Base checkpoint, which has a comparable ImageNet zero-
shot accuracy, ensuring a fair comparison of 2D backbone
architectures.
B.2. Training Strategy
We follow OpenGaussian [35] general training settings. For
the ScanNet dataset [9], that keep point cloud coordinates
fixed and disable 3D Gaussian Splatting (3DGS) densifi-
cation [14]. For the LeRF dataset [15], we optimize point
cloud coordinates and enable 3DGS densification, which is
stopped after 10k training steps.
B.3. Training Time
All experiments are conducted on a single NVIDIA RTX
4090 GPU (24GB). For the LeRF dataset, each scene consists
of approximately 200 images and requires around 60 minutes
Table 5. Ablation study on ScanNet with different feature aggrega-
tion weights. Metrics are reported as mIoU and mAcc for 10, 15,
and 19 class settings.
Context Feature Weight α
mIoU (10, 15, 19)
mAcc (10, 15, 19)
0
0.42, 0.33, 0.33
0.60, 0.50, 0.49
0.2
0.51, 0.38, 0.38
0.69, 0.55, 0.54
0.4
0.51, 0.39, 0.38
0.68, 0.56, 0.55
0.6
0.50, 0.38, 0.38
0.68, 0.56, 0.54
0.8
0.47, 0.36, 0.35
0.65, 0.54, 0.52
1
0.50, 0.37, 0.37
0.68, 0.55, 0.53
for training. For the ScanNet dataset, scenes contain 100–300
images, with an average training time of 30 minutes per
scene.
B.4. ScanNet Dataset Setup
We align our Scannet Dataset test dataset with
[35]
on 10 randomly selected ScanNet scenes, specifically:
scene0000 00,
scene0062 00,
scene0070 00,
scene0097 00,
scene0140 00,
scene0200 00,
scene0347 00,
scene0400 00,
scene0590 00,
scene0645 00.
For text-based queries, we utilize 19 ScanNet-defined
categories:
• 19 categories: wall, floor, cabinet, bed, chair, sofa, table,
door, window, bookshelf, picture, counter, desk, curtain,
refrigerator, shower curtain, toilet, sink, bathtub
• 15 categories: wall, floor, cabinet, bed, chair, sofa, table,
door, window, bookshelf, counter, desk, curtain, toilet, sink
• 10 categories: wall, floor, bed, chair, sofa, table, door,
window, bookshelf, toilet
Training images are downsampled by a factor of 2, and
we use the cleaned point cloud that is processed by Open-
Gaussian.
B.5. Addional Results
We have conducted an ablation study (Tab. 5) to evaluate the
impact of different weighting strategies on performance. This
analysis demonstrates the robustness of our approach and
highlights the sensitivity of the final segmentation quality to
the fusion ratio. We also provide per scene evaluation results
on Scannet (Tab. 6) and Lerf dataset (Tab. 7).
B.6. Efficiency
Regarding inference time, storage memory, feature extrac-
tion time, and training memory cost aspects: previous meth-
ods like LangSplat[27] and LEGaussians[30] perform text-
query localization by rendering a 2D compressed language
embedding map, which is then decoded to match the text
query embeddings—this process is slow. In contrast, our
approach and OpenGaussian[35] can directly localize text
queries in 3D by searching the codebook. Additionally, both
LangSplat[27] and LEGaussians[30] need to maintain the
autoencoder decoder network, which requires larger storage
12

<!-- page 13 -->
Table 6. Per-scene performance of 3D point cloud semantic segmentation on the ScanNet dataset based on text query at different class splits
(10 / 15 / 19 classes).
Scene ID
10-class
15-class
19-class
mIoU ↑
mAcc. ↑
mIoU ↑
mAcc. ↑
mIoU ↑
mAcc. ↑
scene0000 00
0.4744
0.7469
0.4054
0.6208
0.4230
0.6149
scene0062 00
0.4103
0.6476
0.2907
0.5372
0.2923
0.5372
scene0070 00
0.5227
0.6100
0.3899
0.4497
0.3498
0.4086
scene0097 00
0.5607
0.7321
0.3419
0.5689
0.3620
0.5658
scene0140 00
0.5781
0.7134
0.3422
0.4249
0.2985
0.3718
scene0200 00
0.4767
0.6554
0.4336
0.5452
0.4341
0.5452
scene0347 00
0.5587
0.6516
0.4018
0.5599
0.4467
0.5926
scene0400 00
0.5169
0.6865
0.4066
0.5902
0.4067
0.5902
scene0590 00
0.6052
0.7445
0.4517
0.6253
0.3952
0.5609
scene0645 00
0.4388
0.7269
0.3502
0.6075
0.3416
0.6509
Mean
0.5142
0.6915
0.3814
0.5530
0.3750
0.5438
Table 7. Per-scene performance of open vocabulary 3D object selec-
tion and rendering on Lerf dataset with mIoU and mAcc at different
thresholds.
Scene
mIoU ↑mAcc@0.25 ↑mAcc@0.5 ↑
figurines
0.5375
0.7679
0.5893
ramen
0.2638
0.4366
0.1972
teatime
0.5855
0.7797
0.6441
waldo kitchen
0.3178
0.5000
0.4091
Mean
0.4262
0.6211
0.4599
memory. Since our context feature extraction only requires
one additional forward pass of CLIP, we do not introduce
significant additional computation. In our Attention-Driven
Feature Aggregation module, we reuse preloaded multi-view
features, incurring no extra memory cost. All methods are
compatible with the 4090 GPU. Moreover, in previous meth-
ods, the feature extraction process requires running the CLIP
model on each segmentation crop at each granularity level,
which is very time-consuming. Our context-aware feature ex-
traction module can be downgraded to solely global feature
extraction, which significantly improves efficiency while sac-
rificing very little accuracy, as shown in our ablation results.
Therefore, our proposed method can achieve substantial im-
provements without a decrease in efficiency.
13
