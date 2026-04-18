<!-- page 1 -->
The NeRFect Match: Exploring NeRF Features
for Visual Localization
Qunjie Zhou1* , Maxim Maximov2* , Or Litany13 , and Laura Leal-Taixé1
1 NVIDIA
2 Technical University of Munich, Germany
3 Technion, Israel
https://nerfmatch.github.io
Abstract. In this work, we propose the use of Neural Radiance Fields
(NeRF) as a scene representation for visual localization. Recently, NeRF
has been employed to enhance pose regression and scene coordinate re-
gression models by augmenting the training database, providing aux-
iliary supervision through rendered images, or serving as an iterative
refinement module. We extend its recognized advantages – its ability
to provide a compact scene representation with realistic appearances
and accurate geometry – by exploring the potential of NeRF’s internal
features in establishing precise 2D-3D matches for localization. To this
end, we conduct a comprehensive examination of NeRF’s implicit knowl-
edge, acquired through view synthesis, for matching under various condi-
tions. This includes exploring different matching network architectures,
extracting encoder features at multiple layers, and varying training con-
figurations. Significantly, we introduce NeRFMatch, an advanced 2D-3D
matching function that capitalizes on the internal knowledge of NeRF
learned via view synthesis. Our evaluation of NeRFMatch on standard
localization benchmarks, within a structure-based pipeline, achieves com-
petitive results for localization performance on Cambridge Landmarks.
We will release all models and code.
1
Introduction
Visual localization is the task of determining the camera pose of a query image
w.r.t a 3D environment. Such ability to localize an agent in 3D is fundamen-
tal to applications such as robot navigation [21, 68], autonomous driving [27],
and augmented reality [2,65]. Different localization solutions can be categorized
based on their underlying scene representation. Image retrieval [1,4,25,64] rep-
resents a scene as a database of reference images with known camera poses.
A more compact representation utilizes a 3D point cloud, where each point is
triangulated from its 2D projections in multiple views. Structure-based meth-
ods [29,37,51,54,60] rely on such a 3D model with associated keypoint descriptors
to perform accurate localization. Recently, MeshLoc [46] expanded structure-
based localization by integrating dense 3D meshes, allowing to switch between
different descriptors to establish the 2D-3D correspondences.
* These authors contributed equally to this work
arXiv:2403.09577v2  [cs.CV]  21 Aug 2024

<!-- page 2 -->
2
Q. Zhou et al.
Query Image
Reference pose
NeRFMatch
Est. pose
PnP
NeRF
NeRF
Rendered Image
Render
Backprop
Reconstruction 
loss
Pose Reﬁnement
Matching-based Pose Estimation
Render
Synth. DB
Image 
Retrieval
Fig. 1: NeRF-based localization overview. In this work, we propose to use NeRF
as our scene representation for visual localization. Given a query image, we first retrieve
its nearest reference pose using image retrieval, then use NeRFMatch to establish 2D-
3D correspondences between the query image and the NeRF scene points to compute
an initial pose estimate and finally improve its accuracy via pose refinement.
Different from the aforementioned explicit scene representations, absolute
pose regression (APR) [5,11,31,32,56,66] and scene coordinate regression (SCR)
methods [6, 8–10, 36, 57, 72] learn to encode scene information within network
parameters, either directly in the localization network, or as a separate collection
of latent codes as learned scene representation [63]. Despite being more compact
than a sparse point cloud and a 3D mesh, learned implicit scene representations
are less interpretable and are limited to the task of visual localization.
Recently, Neural Radiance Fields (NeRF) [42] have emerged as a powerful
representation of 3D scenes, encoded as continuous mapping of spatial coordi-
nates and viewing angle to density and radiance. NeRF present several benefits:
high interpretability, i.e., one can easily render scene appearance and depth at
any given viewpoint, as well as being highly compact, e.g., a Mip-NeRF [3] model
of 5.28 MB can represent a scene with a spatial extent ranging from 1m2 [57]
to
5km2 [32] (Sec. 5). Owing to its attractive properties, NeRF is emerging
as a prime 3D scene representation [69], alongside meshes, point clouds, and
multi-view images, and has been applied to various other computer vision tasks
such as semantic segmentation [23,35,76], 3D object detection [28,71], Simulta-
neous Localization and Mapping (SLAM) [49,58,75,79] and vision-based local-
ization [15–18,38,40,43,44,73].
NeRF has been leveraged for tackling visual localization in various ways.
iNeRF-style approaches [16, 73] utilize a pre-trained NeRF as an inference-
time pose refinement. Yet such methods commonly suffer from slow convergence
and require pose initialization to be provided. The end-to-end APR and SCR
methods use a pre-trained NeRF only at train-time to augment training sam-
ples [15,44], provide consistency supervision [17,18], and generate proxy depth
ground-truth [15]. In this case, NeRF merely serves as an auxiliary representa-
tion. Orthogonal to the above ways of leveraging NeRF, recent works [38, 43]
propose to use NeRF as a flexible 3D model that can be enriched with volu-

<!-- page 3 -->
NeRFMatch
3
metric descriptors to establish 2D-3D matches between an image and the scene.
CrossFire [43] augments the base NeRF model with an additional 3D feature
prediction branch. This feature is supervised to match 2D features extracted
by an image backbone. NeRF-Loc [38] performs feature matching by utilizing
a combination of features extracted from a generalizable NeRF and projected
multi-view image features as 3D point features. However, both methods require
training NeRF jointly with the matching task, prohibiting the usage of pre-built
NeRF scenes.
In contrast, our work treats NeRF as the primary scene representation in vi-
sual localization without re-training and modifications. Specifically, we focus on
a crucial component – NeRF features – and (i) demonstrate their inherent capa-
bility in effectively supporting feature matching. (ii) We introduce, NeRFMatch,
a matching transformer that aligns 2D image features with 3D NeRF features
and a minimal version of it to facilitate real-time applications. (iii) We present
two options for performing pose refinement on top of NeRFMatch and conduct
detailed analysis on their refinement effectiveness. (iv) We use our NeRFMatch
and pose refinement modules to perform hierarchical NeRF localization which
achieves competitive localization performance on Cambridge Landmarks [32].
Based on our experiments, we point out future work in needs to improve our in-
door localization performance. Our research paves the path towards localization
leveraging NeRF as the sole representation of the scene.
2
Related work
2.1
Visual Localization
Structure-based localization. Such methods [12,29,37,51,54,60] first esti-
mate 2D-3D correspondences between a query image and the 3D points in the
scene and then deploy a Perspective-n-Point (PnP) solver [24,30,34] to compute
the query camera pose. Image retrieval [1,4,25,64] is usually applied in advance
to coarsely localize visible scene structure to a query image [51,60]. Visual fea-
tures [14,20,22,52,59,67,78] are often extracted from a database of scene images
to represent 3D features and matched against the query image features extracted
using the same algorithm to obtain 2D-3D matches. To optimize the inference
runtime, 3D descriptors are cached with the scene model at the cost of high
storage demand and challenging map updates. To relieve the burden coming
from the need of storing visual descriptors, GoMatch [77] performs geometric
feature matching, yet currently being less accuracy than visual feature matching
approaches. Recently, to avoid storing massive amount of scene images as well
as being flexible to switch between different descriptors for 2D-3D matching,
MeshLoc [46] propose to use 3D meshes as a dense 3D model. Compared to
MeshLoc, we also pursuit a dense scene representation via NeRF [42], which not
only store scene images compactly but also provides free per-3D-point NeRF
descriptors for direct 2D-to-3D matching.
End-to-end Learned Localization.
APR methods [5, 11, 31, 32, 56, 66] di-
rectly regress a camera pose from a query image, while being lightweight by

<!-- page 4 -->
4
Q. Zhou et al.
encoding scene information within a single model, they are currently less accu-
rate than approaches based on 2D-3D matching [55]. In contrast, SCR meth-
ods [6,8–10,36,57,72] perform implicit 2D-3D matching via directly regressing
3D scene coordinates from a query image. Similar to APR, they learn to encode
the scene geometry within their own network parameters [6, 8–10, 36] but they
are limited by the model capacity to memorize large-scale scenes. Recently, scene
agnostic SCR methods [63,72] have been proposed to scale up to larger scenes
by decoupling the scene representation from the learned matching function.
2.2
NeRF in Localization
iNeRF [73] directly inverts a NeRF model to refine a camera pose initialization
by iteratively optimizing the photometric difference between the rendered image
and the query image. However, it requires hundreds of iterations to converge and
thus is not directly applicable to real-world visual localization. LENS [44] lever-
ages NeRF as a novel view synthesizer (NVS) to augment the image database for
pose regression training. Similar to LENS [44], NeRF-SCR [15] augments SCR
training samples using RGB-D images rendered from a NeRF model based on
an uncertainty-guided novel view selection. DirectPN [18] incorporates NeRF to
provide photometric consistency supervision for pose regression where it mini-
mizes the color difference between the query image and the image rendered at
a predicted pose. DFNet [17] extends this idea to measure the consistency in
the feature space showing boosted localization performance. NeFes [16] follows
iNeRF to use NeRF as an offline pose refinement module on top of DFNet. It
distills pre-trained DFNet feature into a NeRF model and directly renders it for
pose optimization. Different from NeFes, we directly use 3D viewpoint-invariant
feature learned during a standard NeRF training and validating its potential to
deliver highly-accurate localization performance. In addition, our model is able
to perform localization in a scene-agnostic (multi-scene) setting while DFNet
and NeFes is scene-dependent.
The most relevant works to ours are NeRFLoc [38] and CrossFire [43] as
they also establish explicit 2D-3D matches with features rendered from NeRF.
NeRFLoc proposes a generalizable NeRF that is conditioned on a set of reference
images and reference depths to output descriptors for 3D points by fusing multi-
view image features, while CrossFire lifts an instant-NGP [45] model to directly
outputs feature descriptors for 3D points. In contrast to their methods that both
require to train their customized scene model together with the matching model.
our NeRFMatch directly learn to align image feature with pre-trained NeRF
features, which allows us to directly benefit from the on-going advancement in
the typical NeRF research.
3
NeRF-based Localization
In this work we explore the capability of NeRF features, tasked with view syn-
thesis, to offer precise 2D-3D correspondences for addressing visual localization.

<!-- page 5 -->
NeRFMatch
5
To this end, we introduce our NeRF-based localization pipeline, which adheres
to the general steps of standard structure-based localization approaches [51,60].
We provide an overview of the localization pipeline in Sec. 3.1, followed by an
explanation on how we utilize NeRF as a scene representation for localization
in Sec. 3.2. Afterwards, we detail our iterative pose refinement component in
Sec. 3.3. Finally, we delve into the specific challenge of matching images to
NeRF features using our newly proposed NeRFMatch model in Sec. 4.
3.1
Localization Pipeline
Given a query image and a scene represented by a pre-trained NeRF model
associated with a list of pre-cached reference poses, e.g., the poses used to train
the NeRF, our localization pipeline involves three steps to localize the query as
shown in Fig. 1. (i) We first apply image retrieval [1,64], which extracts visual
descriptors from the query image and the reference images (synthesized by NeRF
at all reference poses, cf . Sec. 5.1) to find the nearest neighbour to the query
based on descriptor distances. This step efficiently narrows down the pose search
space to the vicinity of the reference pose. (ii) We use the reference pose to render
a set of 3D points with associated NeRF descriptors (cf . Sec. 3.2) and feed it
together with the query image into our NeRFMatch (cf . Sec. 4) to predict the
2D-3D matches between the query image and 3D points, from which we estimate
the absolute camera pose of the query image via a PnP solver [24, 30, 34]. (iii)
We further use our pose refinement module (cf . Sec. 3.3) to improve the pose
estimation iteratively. Note that this step is optional considering the trade-off
between accuracy and runtime efficiency.
3.2
NeRF for Localization
NeRF architecture.
NeRF [42] models a 3D scene with a coordinate net-
work. Specifically, it maps a 3D point and a camera viewing direction to their
corresponding volume density and RGB values. As depicted in Fig. 2, a stan-
dard NeRF model consists of two non-learnable positional encodings, Px for 3D
coordinates and Pd for viewing directions, and three trainable components: a 3D
point encoder Θx, a volume density decoder Θσ, and a RGB decoder Θc. More
concretely, the 3D encoder consists of L layers, i.e., Θx = ΘL
x ◦· · · ◦Θ1
x. In this
work, we are particularly interested in exploring the potential of 3D features
extracted within the 3D encoder. Given a 3D point X ∈R3, we define its 3D
feature extracted at j-th 3D encoder layer as f j = Θj
x ◦· · · ◦Θ1
x(Px(X)), where
Θj
x is the j-th layer in the 3D encoder. The last layer 3D feature is next input
into the density decoder to predict the volume density σ = Θσ(f L), while the
color decoder takes a positional-embedded viewing direction d ∈R2 in addition
to the 3D feature f L to compute the view-dependent color c = Θc(f L, Pd(d)).
Volumetric rendering of color. NeRF employs the continuous scene repre-
sentation to render per-ray color using a discretized volumetric rendering pro-
cedure. Given a ray r(t) = o + td emitted from the camera center o ∈R3, along
a viewing direction d which intersects the image plane at pixel x, this ray is

<!-- page 6 -->
6
Q. Zhou et al.
NeRF
NeRF  Features            :
Rendering   _
(Eq. 2)
Fig. 2: An overview of the standard NeRF architecture. The input consists of
a scene coordinate X and ray directions d. The outputs include color c, density σ. We
obtain intermediate features, denoted as f j, using volumetric rendering.
sampled at N 3D points between a near and far planes. The RGB color ˆC(r) at
pixel x is calculated as:
 \lab e
l
 {eq:render_rgb} \hat {C}(r) = \sum _{i=1}^{N} w_i c_i, \indent w_i = T_i (1-exp(-\delta _i \sigma _i)), \
i
(1)
where δ = ti+1 −ti is the sampling interval, ci and σi are the predicted color
and density of i-th sampled 3D point, and Ti = exp(−Pi−1
j=1 σjδj) [42].
Volumetric rendering of 3D points and features. We investigate the ca-
pability of NeRF features to act as descriptors for 3D surface points, facilitating
2D-3D correspondence with a query image. Following the volumetric rendering
process defined in Eq. (1), we define a rendered 3D (surface) point ˆX(r) and its
associated NeRF descriptor ˆF j(r) along the ray r cast through image pixel x as
the weighted sum of sampled 3D points and their respective features:
 \label {eq:render_3d} \hat {X}(r) = \sum _{i=1}^{N} w_i X_i ,\indent \hat {F}^{j}(r) = \sum _{i=1}^{N} w_i f^j_i . r
(2)
Here, Xi is i-th sampled 3D points, f j
i is its 3D feature extracted at j-th 3D
encoder layer and wi is the weight computed from its density prediction Eq. (1).
3.3
Pose Refinement
Following the matching-based pose estimation, we employ two approaches to
refine the estimated camera pose. The first approach, iterative refinement, uses
the estimated camera pose as a new reference for extracting NeRF features. This
process involves repeating the matching procedure with the updated reference
pose, incrementally enhancing the results due to the closer proximity of the ref-
erence pose, which makes the NeRF and image features more similar. While this
process can be repeated multiple times, significant improvements are typically
observed after the first refinement (cf . Sec. 5.4). Inspired by iNeRF [73], our
second refinement option combines optimization and matching where we back-
propagate through the frozen NeRF model to optimize an initial camera pose
estimate by minimizing the photometric difference between the query image and

<!-- page 7 -->
NeRFMatch
7
Query Image
2D 
Encoder
NeRF Feat.
Coarse Match
Feat Sampling
Fine Oﬀset
Fine Matching
NeRF
Ref. Pose
Im. Featc
2D-3D 
Matches
3D Pts
Feature Extraction
NeRFMatch-Mini
2D-3D 
Matches
NeRFMatch
2D Pts 
( H/8*W/8 )
( H*W )
( H/8*W/8*Dc )
Im. Featf
( H/8*W/8*Df )
PE
2D Pts
3D Pts 
PE
SA
SA
shared
CA
SA
Dual Softmax
Dual
Softmax
CA
Proj
Fig. 3: NeRFMatch architecture. We present NeRFMatch as our full matching
model (rightmost) and NeRFMatch-Mini as a light version of it (middle). Both models
share the same feature extraction process, where we use a 2D encoder to extract image
features at two resolutions and render 3D points with associated NeRF features at
sampled 2D pixel locations from the reference viewpoint. The full matching uses self-
attention (SA) and cross-attention (CA) with positional encodings (PE).
the current NeRF rendered image. We then use the optimized camera pose to
render again 3D points and features to perform the 2D-3D matching and com-
pute the final refined camera pose.
4
Image-to-NeRF Matching Network
Overview.
To establish 2D-3D correspondences between a query image and
NeRF scene points for localization, we introduce two variants of our proposed
image-to-NeRF matching network, a full version (NeRFMatch) and a minimal
version (NeRFMatch-Mini), as depicted in Fig. 3. The full version is notably
more expressive, incorporating powerful attention modules [14,52,59] and follows
a coarse-to-fine matching paradigm [14,59,78]. While it delivers more accurate
matches it is computationally more expensive (cf . Sec. 5.3). Consequently, we
also propose a minimal version of our method that focuses on learning good
features for matching, removing the need to learning the matching function itself.
Both models comprise a feature extraction module that encodes both the query
image and 3D scene points into a feature space, and a matching module that
aligns these two feature sets to determine the 2D-3D correspondences. We further
detail their architectural designs and supervision in the subsequent subsections.
4.1
NeRFMatch-Mini
Image encoding. Given a query image I ∈RH×W ×3, we use a CNN encoder
to extract a coarse-level feature map F c
m ∈RNm×Dc with Nm = H
8 × W
8 and a
fine-level feature map F f
m ∈R
H
2 × W
2 ×Df .

<!-- page 8 -->
8
Q. Zhou et al.
NeRF feature encoding. Given the reference pose of the query image found
by image retrieval (cf . Sec. 3.1), we use NeRF to obtain a set of scene 3D points
Xs ∈RNs×3 and their associating NeRF features Fs ∈RNs×Dc (cf . Sec. 3.2).
To make the matching module memory managable, we consider NeRF features
Fs ∈RNs×Dc rendered along rays originated at center pixels of every 8×8 image
patch in the reference view, i.e., Ns = H
8 × W
8 .
Dual-softmax matching. After the feature extraction, NeRFMatch-Mini di-
rectly matches the NeRF feature map Fs against the coarse-level image feature
map F c
m with a non-learnable dual-softmax matching function adopted from
[59]. Specifically, we compute pair-wise cosine similarities between the two fea-
ture maps followed by a dual-softmax operation to obtain the matching score
matrix S ∈RNm×Ns. Finally, we extract mutual matches based on the associa-
tion scores which gives us the predicted matches.
4.2
NeRFMatch
As depicted in Fig. 3, NeRFMatch shares the same feature extraction processes
as NeRFMatch-Mini, but learns an attention-based matching module which fol-
lows the coarse-to-fine paradigm of image-to-image feature matching [14,59,78]:
(i) firstly, we identify 3D-to-image patch matches using a coarse matching mod-
ule, (ii) secondly, we refine to pixel accuracy with a fine matching module. We
describe the detailed matching module architecture in the following paragraphs.
Coarse-level matching.
Equipped with coarse image features F c
m that rep-
resent image local patches and raw NeRF features Fs that represent individual
3D scene points, we first apply 2D positional encoding [13,59] to equip the im-
age features with positional information. We next enrich the global contextual
information within each domain by applying several self-attention blocks. We
share the self-attention weights between image features and NeRF features to
help bring the features from two different domains into a common embedding
space for matching. After self-attention, we enhance the 3D information explic-
itly by concatenating each NeRF feature with its positional-encoded 3D points
using NeRF positional encoding [42]. Afterwards, we feed those features into
a cross-attention layer to enable cross-domain interactions. We use the same
dual-softmax matching function introduced above (cf . Sec. 4.1) to obtain coarse
matches Mc = {(i, j)|i ∈(0, Nm −1), j ∈(0, Ns −1)} where i, j are the indices
of the image and point features.
Fine-level matching. For each coarse match mc = (i, j) that we extract, we
start by gathering its high-resolution image patch feature F f
m(i) ∈Rw×w×Df
from the fine-level feature map, centered around the corresponding location of
the match. Inside each of these image patches, a self-attention block is applied
to spread contextual information throughout the patch. Next, for every 3D point
feature, we take the cross-attended feature obtained from the coarse matching,
and process it through a linear layer to adjust its feature dimension from Dc
to Df. In the subsequent step, we align the 3D feature with its corresponding
local feature map. This alignment [59] produces a heatmap, which depicts the
likelihood of the 3D point j matching with each pixel in the vicinity of image pixel

<!-- page 9 -->
NeRFMatch
9
i. To obtain the exact, fine match, we compute the expected value across this
heatmap’s probability distribution. The final, refined matches obtained through
this process are denoted as Mf.
4.3
Supervision
Ground-truth matches.
We project the rendered 3D scene points Xs onto
the query image using the ground-truth query camera calibration, which gives
their precise 2D projections xs at the resolution in which we want to supervise
the fine matches. To compute the ground-truth coarse association matrix Mgt
which is a binary mask, for each 3D point j, we assign it to its belonging i-th
8 × 8 local patch in the query image. We set the association value Mgt(i, j) = 1
if a 3D point j finds its 2D patch i within the query image boundary. Notice,
one image patch can be assigned to multiple 3D points yet each 3D point has at
most one 2D match.
Losses. To supervise the coarse matching, we apply the log loss [59] to increase
the dual-softmax probability at the ground-truth matching locations in Mgt. The
coarse matching loss Lc is defined as:
 \ l a b
el 
{
eq:coarse
_matchin g_loss} L_c = - \frac {1}{M_{gt}} \sum _{(i, j) \in \mathcal {M}_{gt}} \log (S(i, j)). 
(3)
To compute the fine matching loss, for a 3D point Xj with ground-truth fine
match xj, we supervise its predicted fine match ˜xj by minimizing its pixel dis-
tance to the ground-truth match. Following
[59, 67], we compute the total
variance σ2(j) of the corresponding heatmap and minimize the weighted loss
function:
 \ l
a
be
l
 {eq:fin
e
_matching_ l oss} L_f = \frac {1}{M_{f}} \sum _{(i, j)\in M_f} \frac {1}{\sigma ^2(i)} ||\Tilde {x}_j - x_j||_2. 
(4)
The NeRFMatch-Mini model is supervised with only the coarse loss Lc while
for NeRFMatch model is supervised with the sum of coarse and fine matching
losses Lc + Lf.
5
Experiments
Datasets.
Cambridge Landmarks [32] is a dataset of handheld smartphone
images of 6 outdoor scenes with large exposure variations which is considered
challenging for NeRF techniques. We follow previous work [6, 53] and evaluate
on a subset of 5 scenes whose spatial extent ranges from 875m2 to 5600m2. We
also test our method on 7-Scenes [57], which is composed of RGB-D images
captured in 7 unique indoor scenes whose size ranges from 1m3 to 18m3. Its
images contain large texture-less surfaces, motion blur, and object occlusions.
For both datasets, we follow the original released training and testing splits.
Following recent work [6,7,16], we use the more accurate SfM pose annotations
for 7-Scenes rather than its original pose annotations.

<!-- page 10 -->
10
Q. Zhou et al.
Evaluation metrics. We report median pose errors, i.e., translation error in
centimeters and median rotation error in degrees. We further report localiza-
tion recall that measures the percentage of queries localized with pose errors
below specific thresholds, i.e., (5cm, 5◦) for 7-Scenes. As Cambridge Landmarks
has large variation in scene scales, we follow
[10, 38] to use 5◦rotation er-
ror and variable translation error thresholds, i.e., 38/22/15/35/45cm for King’s
College/Old Hospital/Shop Facade/St. Mary’s Church/Great Court.
Implementation details. We use the first two blocks of ConvFormer [74] as the
image backbone and initialize it with ImageNet-1K [50] pre-trained weights*. We
set feature dimensions for coarse and fine matching as Dc = 256 and Df = 128.
For fine matching, we use local window size w = 5 for image feature cropping.
We resize query images to 480 × 480 for all experiments. We train minimal and
full NeRFMatch models using Adam [33] optimizer with canonical learning rate
clr = 0.0008/0.0004 and batch size cbs = 16 for 30/50 epochs accordingly. We
decay the learning rate based on the cosine annealing scheduling [39]. Our models
are trained on 8 Nvidia V100 GPUs (16/32GB). A MipNeRF model is 5.28 MB
in size. Given a camera pose, it takes 141 milliseconds to render 3600 3D points
with its features on a single 16GB Nividia V100 GPU. Our NeRFMatch-Mini
and NeRFMatch models are 42.8/50.4 MB in size and require 37/157ms to run
a forward pass at 480×480 image resolution. We provide NeRF implementation
details in the supplementary material.
5.1
Localization Evaluation
Baselines.
We first compare our proposed NeRFMatch and NeRFMatch-
Mini against common visual localization approaches on both indoor and out-
door datasets. We split the methods into the end-to-end category including
APR [16, 17, 44, 56] and SCR [6, 10, 36] methods, and the hierarchical cate-
gory [38,43,51,53,60,62,63,72] where methods rely on an extra image retrieval
step to coarsely localize the region in the scene. In addition, we specify the
underlying scene representation used for localization at test time.
Inference settings.
We use top-1/10 reference poses for Mini/Full NeRF-
Match models for outdoor and top-1 for indoor and apply the best pose refine-
ment determined in Sec. 5.4. We provide more details in supplementary.
Results on Cambridge Landmarks. As shown in Tab. 1, our minimal ver-
sion despite being lightweight, is able to achieve comparative results w.r.t. most
of the SCR methods and surpass all APR methods. With a more advanced
attention-based hierarchical matching function, our full model achieves the com-
petitive results among all methods. Our experiments fully demonstrate that the
NeRF inner features learned via view synthesis are discriminative 3D represen-
tations for 2D-3D matching.
Results on 7-Scenes. As shown in Tab. 2, we are on-par with the best APR
method (the upper rows), NeFeS [16], while we are less accurate than SCR and
* The weights can be downloaded from huggingface.co/timm/convformer_b36.sail_
in1k_384

<!-- page 11 -->
NeRFMatch
11
Table 1: Outdoor localization on Cambridge Landmarks [32]. We report per-
scene median rotation and position errors in (cm,◦) and its average across scenes.
Method
Scene
Cambridge Landmarks - Outdoor
Repres.
Kings
Hospital
Shop
StMary
Court
Avg.Med ↓
End-to-End
MS-Trans. [56]
APR Net.
83/1.5
181/2.4
86/3.1
162/4
-
-
DFNet [17]
APR Net.
73/2.4
200 /3
67/2.2
137/4
-
-
LENS [44]
APR Net.
33/0.5
44/0.9
27/1.6
53/1.6
-
-
NeFeS [16]
APR+NeRF
37/0.6
55/0.9
14/0.5
32/1
-
-
DSAC* [10]
SCR Net.
15/0.3
21/0.4
5/0.3
13/0.4
49/0.3
20.6/0.3
HACNet [36]
SCR Net.
18/0.3
19/0.3
6/0.3
9/0.3
28/0.2
16/0.3
ACE [6]
SCR Net.
28/0.4
31/0.6
5/0.3
18/0.6
43/0.2
25/0.4
Hierachical
SANet [72]
3D+RGB
32/0.5
32/0.5
10/0.5
16/0.6
328/2.0
83.6/0.8
DSM [62]
SCR Net.
19/0.4
24/0.4
7/0.4
12/0.4
44/0.2
21.2/0.4
NeuMap [63]
SCode+RGB
14/0.2
19/0.4
6/0.3
17/0.5
6/0.1
12.4/0.3
InLoc [60]
3D+RGB
46/0.8
48/1.0
11/0.5
18/0.6
120/0.6
48.6/0.7
HLoc [51]
3D+RGB
12/0.2
15/0.3
4/0.2
7/0.2
16/0.1
10.8/0.2
PixLoc [53]
3D+RGB
14/0.2
16/0.3
5/0.2
10/0.3
30/0.1
15/0.2
CrossFire [43]
NeRF+RGB
47/0.7
43/0.7
20/1.2
39/1.4
-
-
NeRFLoc [38]
NeRF+RGBD
11/0.2
18/0.4
4/0.2
7/0.2
25/0.1
13/0.2
NeRFMatch-Mini NeRF+RGB
19.0/0.3
30.2/0.6 10.3/0.5 11.3/0.4 29.1/0.2
20.0/0.4
NeRFMatch
NeRF+RGB
13.0/0.2 19.4/0.4
8.5/0.4
7.9/0.3
17.5/0.1
13.3/0.3
NeRFMatch
NeRF
12.7/0.2 20.7/0.4
8.7/0.4
11.3/0.4 19.5/0.1
14.6/0.3
Table 2: Indoor localization on 7-Scenes [57]. We report per-scene median rota-
tion and position errors in (cm,◦) and their average across scenes, along with averaged
localization recall.
Method
Scene
7-Scenes - SfM Poses - Indoor
Repres.
Chess
Fire
Heads
Office
Pump.
Kitchen
Stairs
Avg.Med↓Avg.Recall↑.
MS-Trans. [56]
APR Net.
11/6.4
23/11.5
13/13
18/8.1
17/8.4
16/8.9
29/10.3
18.1/9.5
-
DFNet [17]
APR Net.
3/1.1
6/2.3
4/2.3
6/1.5
7/1.9
7/1.7
12/2.6
6.4/1.9
-
NeFeS [16]
APR+NeRF
2/0.8
2/0.8
2/1.4
2/0.6
2/0.6
2/0.6
5/1.3
2.4/0.9
-
DSAC* [10]
SCR Net.
0.5/0.2
0.8/0.3
0.5/0.3
1.2/0.3
1.2/0.3
0.7/0.2
2.7/0.8
1.1/0.3
97.8
ACE [6]
SCR Net.
0.7/0.5
0.6/0.9
0.5/ 0.5
1.2/0.5
1.1/0.2
0.9/0.5
2.8/1.0
1.1/0.6
97.1
DVLAD+R2D2 [48,64] 3D+RGB
0.4/0.1 0.5/0.2 0.4/0.2 0.7/0.2 0.6/0.1 0.4/0.1 2.4/0.7
0.8/0.2
95.7
HLoc [51]
3D+RGB
0.8/0.1
0.9/0.2
0.6/0.3
1.2/0.2
1.4/0.2
1.1/0.1
2.9/0.8
1.3/0.3
95.7
NeRFMatch-Mini
NeRF+RGB
1.6/0.5
1.5/0.6
1.4/0.9
3.6/1.0
3.5/0.9
1.7/0.5
8.5/2.1
3.1/0.9
74.4
NeRFMatch
NeRF+RGB
0.9/0.3
1.1/0.4
1.4/1.0
3.0/0.8
2.2/0.6
1.0/0.3
9.0/1.5
2.7/0.7
78.2
NeRFMatch
NeRF
0.9/0.3
1.1/0.4
1.5/1.0
3.0/0.8
2.2/0.6
1.0/0.3
10.1/1.7
2.8/0.7
78.4
visual matching methods whose accuracy on 7-Scenes is close to saturation (the
middle rows). Our hypothesis is that those method are able to benefit from
the dense distribution of frames in the sequence. However, we have to limit our
NeRF training to 900 training frames (loaded at once into memory) per scene
for efficient training despite thousands of frames are available.
NeRF-only localization.
We further push our method to using NeRF as
the only scene representation for localization, which means we would no longer
need access to a real image database as part of the scene representation, which
requires a significantly bigger storage than a single NeRF model. For this pur-
pose, we propose to perform image retrieval on synthesized images rendered by
our NeRF model. This entails a small decrease in performance on Cambridge in
translation error. This slight degradation in performance can be attributed to
the increased complexity of these scenes for NeRF. This assertion is supported
by the observation that, when switching to image retrieval on synthesized im-
ages, our model demonstrates almost no change in performance on the indoor

<!-- page 12 -->
12
Q. Zhou et al.
Table 3: NeRF feature ablation on Cambridge [32]. We train NeRFMatch-Mini
with different 3D features and compare their localization performance.
Metrics
Pt3D
Pe3D
f 1
f 2
f 3
f 4
f 5
f 6
f 7
Med. Translation (cm, ↓)
458.0
34.3
28.7
28.4
27.9
28.3
28.3
30.2
61.3
Med. Rotation (◦, ↓)
6.5
0.6
0.5
0.5
0.5
0.5
0.5
0.5
1.3
Localize Recall. (%, ↑)
0.7
51.4
58.6
59.4
59.2
56.9
57.7
53.0
38.8
7-Scenes dataset. Our results open the door to a localization pipeline where only
a NeRF model is needed.
In the following experiments, we conduct ablation studies to thoroughly un-
derstand the different components of our method. We conduct all ablations on
Cambridge Landmarks.
5.2
NeRF Feature Ablation
Experimental setup.
As our first ablation, we investigate the potential of
NeRF features in 2D-3D matching. We consider several different types of 3D
features including the raw 3D point coordinates (Pt3D), positional encoded 3D
points (Pe3D), and NeRF inner features output from all intermediate layers f j
with j ∈[1, 7], as shown in Fig. 2. NeRF features have the same dimension as
the image backbone features, and thus are directly ready for 2D-3D matching.
For 3D point coordinates and positional encoded 3D points as 3D features, we
use a simple linear layer to lift them to the image feature dimension, and we
train them together with the image backbone on feature matching. To fully
focus on the influence of different 3D features, we train NeRFMatch-Mini models
with different 3D features and conduct matching without pose refinement. We
evaluate the quality of 2D-3D matches for localization and report the median
pose errors and localization recall in Tab. 3.
Results. We show that directly matching image features with lifted 3D coor-
dinate features does not yield accurate results. This significantly improves when
using a NeRF positional encoding layer [42] on top of the raw 3D coordinates.
Notably, the Pe3D feature corresponds to the input for NeRF. We further show
that all first six layer NeRF features are better than Pe3D in aligning image
features to 3D features, showing that matching benefits from the view synthesis
training. The only exception is that the last layer feature f 7, which produces
features that are less discriminative for matching and are more focused on the
NeRF goal of predicting density and RGB values. Among layer features, we
found the middle layer f 3 to be slightly better than the others and choose it to
be the default NeRF feature we use for our NeRFMatch models.
5.3
NeRFMatch Ablation
Architecture ablation. Next, we study the influence of different image back-
bones and matching functions on our matching model in Tab. 4. We first compare
two convolution backbones for image feature encoding, i.e., ResNet34 [26] and

<!-- page 13 -->
NeRFMatch
13
Table 4: NeRFMatch architecture ablation on Cambridge Landmarks [32].
We report averaged median pose error in (cm,◦) and localization recall.
Model
Backbone
Pretrain
Matcher
Training
Avg.Med Avg.Recall Model Size Runtime
Name
Backbone
Scenes
(cm/◦) ↓
(%) ↑
(MB) ↓
(ms) ↓
-
ResNet34
✓
Minimal
Per-Scene
32.7/0.6
52.0
32.8
23
-
ConvFormer
✗
Minimal
Per-Scene
34.8/0.6
50.7
42.8
37
NeRFMatch-Mini
ConvFormer
✓
Minimal
Per-Scene
27.9/0.5
59.2
42.8
37
NeRFMatch-Mini (MS) ConvFormer
✓
Minimal Multi-Scene
30.8/0.5
53.6
42.8
37
NeRFMatch
ConvFormer
✓
Full
Per-Scene
16.5/0.3
71.3
50.4
157
NeRFMatch (MS)
ConvFormer
✓
Full
Multi-Scene
22.0/0.4
65.2
50.4
157
ConvFormer [74]. Our method demonstrates improved performance with the lat-
ter, which is consistent with their performance on image classification [50]. We
also observe that large-scale ImageNet [50] pre-training provides a better starting
point for extracting suitable image features for matching, leading to increased
localization accuracy compared to training from scratch. We further show that
for the same backbone, a more advanced attention matching function with a
coarse-to-fine design is crucial for accurate matching and significantly improves
localization accuracy, albeit at the cost of increased runtime.
Training ablation.
In addition to architecture choices, we also examine the
influence of different training settings, i.e., training per-scene (default) and train-
ing multi-scenes (all scenes within a dataset). Despite NeRF features being
trained per-scene, we surprisingly find that both minimal and full NeRFMatch
models can be trained to handle multi-scene (MS) localization with only a slight
decrease in accuracy w.r.t. the per-scene model. This is a similar finding as recent
scene-agnostic SCR methods [62, 63, 72], that extend per-scene SCR to multi-
scenes by conditioning SCR on scene-specific 3D points. While scene-agnostic
SCR learns to regress directly the 3D features in the form of xyz coordinates,
our models learn to find a common ground between image features and NeRF
features for matching.
5.4
Pose Refinement
After conducting matching architecture ablation, we investigate different refine-
ment methods to further increase pose accuracy. We examine two approaches:
(i) an iterative approach, where we re-run matching with the last computed esti-
mation as reference pose, and (ii) an optimization approach, where we optimize
a reference pose through frozen NeRF weights using a photometric loss and then
run the final matching. As shown in Fig. 4, both methods show an improvement
over the initial estimate. We assess both the NeRFMatch model and its minimal
version (Mini) trained per-scene.
Iterative refinement for computational efficiency.
In the NeRFMatch
setting, both refinement approaches show similar early results, since the initial
pose estimate is relatively close to the solution, but the iterative approach is more
stable over time. Additionally, it is worth noting that the optimization approach
incurs a higher computational cost compared to the iterative approach. The
runtime for the optimization refinement excluding the matching step (shown

<!-- page 14 -->
14
Q. Zhou et al.
Model
Best
No Refinement
Refined
Refined
Refinement
(top−1)
(top−1)
(top−10)
Metrics
Avg.Med (cm/◦) ↓/ Avg.Recall (%) ↑
NeRFMatch-Mini Opt+Match
27.9/0.5/59.2
20.5/0.4/70.9 20.5/0.4/70.9
NeRFMatch
Iter.
16.5/0.3/71.3
14.2/0.3/78.2 13.3/0.3/80.8
Fig. 4: Refinement ablation on Cambridge Landmarks [32]. On the left side,
we depict the average recall for optimization-based (Opt+Match) and iterative (Iter)
refinement approaches across multiple iterations. We provide results for both the NeRF-
Match and its minimal setting. On the right side, we report averaged median pose error
in (cm/◦) and localization recall with the best refinement configurations.
in Tab. 4) is 398.4ms for a single optimization step and subsequent rendering.
In contrast, for a single step of iterative refinement, the runtime amounts to
141.2ms. Given the better performance and the quicker computational time, we
opt for the iterative approach as the default refinement for NeRFMatch.
Optimization refinement for large pose corrections. In comparison to the
iterative method, the optimization approach significantly benefits the minimal
matching model. This is because the initial query pose estimation is notably
distant compared to the one given by NeRFMatch, and subsequent iterations
only result in incremental refinements. The optimization approach takes a more
substantial step towards the query pose, and achieving a more optimal refer-
ence pose results in improved query pose estimation. We have already seen that
the optimization refinement incurs higher computational time, but it also de-
pends on a learning rate schedule, which is sensitive to configuration. Through
empirical analysis, we select 1 × 10−3 as the initial learning rate and apply a
cosine annealing learning rate schedule. We set the optimization approach as the
default refinement for NeRFMatch-Mini.
6
Conclusion and Limitations
In this work, we have taken initial steps towards leveraging NeRF as the primary
representation for the task of camera localization. To achieve this, we have thor-
oughly examined the performance of NeRF features in the localization task, con-
sidering various architectural designs, feature extraction from different encoder
layers, and diverse training configurations. Additionally, we have demonstrated
that NeRF can remove the need for the original image set for coarse localization.
Our results suggest that NeRF features are highly effective for 2D-3D matching.
While NeRFMatch marks a significant step towards comprehensive localiza-
tion using NeRF, it also highlights several limitations that necessitate further
research. Specifically, we observe a noticeable performance gap when applying
our method to indoor 7-Scenes dataset.

<!-- page 15 -->
NeRFMatch | Supplementary
In this supplementary document, we provide further details regarding our
proposed method and qualitative results. We describe our implementation details
for NeRF in Appendix A, and for NeRFMatch in Appendix B. Then, we present
additional analysis and discussion of our method in Appendix C.
Fig. 1: Example of masking on Kings College scene. Top images - original images,
bottom - semantic segmentation using [19].
A
NeRF Implementation Details
Handling challenges in outdoor scenes. Outdoor reconstruction in the wild
has a lot of challenges including illumination changes, transient objects, and
distant regions. For the task of localization, we are interested in reconstructing
only the static scene elements, e.g., roads, buildings, and signs.
To properly train NeRF in such a scenario, we use a pre-trained semantic
segmentation model [19] and mask out the sky and transient objects: pedestrians,
bicycles, and vehicles. These objects occupy only a minor part of the captured
images and are excluded from the loss computation during the training process.
Analogous methods for masking in sky regions and/or dynamic object areas
have been implemented in other works focused on the reconstruction of urban
scenes [47,61,70]. We show examples of semantic segmentation in Fig. 1 and its
effect on synthesized views in Fig. 2.

<!-- page 16 -->
16
Fig. 2: Example of masking on the King’s College scene of Cambridge Landmarks [32].
The bottom row are rendered with NeRF, and the top row - ground truth images.
Table 1: NeRF PSNR scores. We present the PSNR scores for our trained MipNeRF
models on each scene of Cambridge Landmarks [32] and 7-Scenes [57].
Cambridge Landmarks - Outdoor
7-Scenes - indoor
Kings Hospital Shop StMary Court Average Chess Fire Heads Office Pump. Kitchen Stairs Average
22.9
22.1
24.0
23.0
23.2
23.1
29.6
30.0
32.5
30.2
31.4
27.9
34.7
30.9
To account for illumination changes, we use an appearance vector that we
concatenate together with the view direction as input, similar to [41]. The ap-
pearance vector changes across sequences but stays the same for all frames in
one sequence since appearance does not drastically change inside a sequence.
NeRF architecture. Our NeRF model consists of a MipNeRF [3] architecture
with both coarse and fine networks. We utilize the final outputs from the fine
network to render RGB, depth maps, and 3D features.
NeRF training. For each scene, we load a subset of 900 training images and 8
validation images and train each model for 15 epochs. From the set of all pixels
in all training samples, we randomly sample a batch of 9216 rays. Subsequently,
for each ray, we sample 128 points for the coarse network and an additional
128 for the fine network. We use the Adam optimizer [33] with a learning rate
1.6×10−3 and cosine annealing schedule [39]. In Tab. 1, we present the per-scene
PSNR scores for our trained models on the training images.
B
NeRFMatch Implementation Details
We summarize average runtime performance for NeRF and both matching mod-
els in Tab. 2.
Training pairs.
We use the same training pairs* generated by PixLoc [53]
which were computed based on image covisibility within the training split. Dur-
ing training, for each train image we load its top-20 covisible pairs. For each
* Image pairs are available from https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/

<!-- page 17 -->
NeRFMatch | Supplementary
17
Table 2: Runtime. We show runtime of NeRFMatch-Mini and NeRFMatch. For pose
refinement we are using optimization refinement for NeRFMatch-Mini and iterative
refinement for NeRFMatch.
NeRF type
NeRFMatch-Mini
NeRFMatch
NeRF render
141ms
141ms
Image-to-NeRF matching
37ms
157ms
Pose refinement
398ms
141ms
training epoch, we then randomly sample 10000 training pairs from those co-
visible pairs for each scene. In the case, we train multiple scenes, we merge
those pairs across scenes which allows us to balance the training samples across
different scenes.
Image retrieval.
We adopt the retrieval pairs pre-computed by PixLoc [53]
using NetVLAD [1] for Cambridge Landmarks [32] and DenseVLAD [64] for 7
Scenes [57] during inference. We use those retrieval pairs for all experiments
by default except for the NeRF-only localization experiment in Sec. 5.1. That
experiment is to confirm the feasibility of NeRF-only localization, therefore we
run NetVLAD [1] to extract retrieval pairs at image resolution 480×480 between
the real query images and the training images synthesized by NeRF.
During inference, we noticed applying top−k retrieval pairs with k > 1 show
evident improvement for NeRFMatch on Cambridge Landmarks. Thus, we set
k = 10 following the common localization practice [51, 53]. For NeRFMatch-
Mini, setting k > 1 did not change much the performance. We suspect this is
due to its less accurate matches, which makes the outlier rejection harder when
merging noisy correspondences from more pairs. For the indoor 7 Scenes dataset,
we use k = 1 which is sufficient for relatively small-size scenes.
Optimization refinement.
Similar to iNeRF [73], we are doing a forward
pass through frozen NeRF MLP layers using an estimated pose as the initial
camera pose. Instead of rendering the entire image, we sample and render 3600
rays, which are equally spread in a grid structure across the image plane. The
we apply a regular photometric loss between the query image and the rendered
image and backpropagate to update the initial camera pose. Instead of using the
raw updated camera pose, we render the NeRF features and match them with
the NeRFMatch to obtain the final camera pose.
C
Additional Details
NeRF backbones. In this section, we evaluate additional NeRF type - Instant
NGP [45] in comparison to MipNeRF [3]. We use MipNeRF for our experiments
in the main paper . As shown in Tab. 3, Instant NGP performs significantly
worse. We hypothesize that this is due to noisy depth reconstruction that is
typical for Instant NGP.
N

<!-- page 18 -->
18
Table 3: NeRF backbone ablation on Cambridge Landmarks. We compare
NeRFMatch-Mini and NeRFMatch performances using Instant NGP.
NeRF type
Avg. Med (cm/◦) ↓/Recall (%) ↑
NeRFMatch-Mini
NeRFMatch
Instant NGP
41.1/0.7/44.4
28.1/0.5/61.3
MipNeRF
20.0/0.4/69.7
13.3/0.3/80.8
Impact of scene sizes. Scene size affects both NeRF and localization perfor-
mance, often coupled with scene content and camera pose distribution. Ranking
scenes by localization errors (lower is better) leads to OldHospital (50 × 40m2)
> KingsCollege (140 × 40m2) > ShopFacade (35 × 25m2) for outdoor and stairs
(2.5 × 2 × 1.5m3) > pumpkin (2.5 × 2 × 1m3) > redkitchen (4 × 4 × 1.5m3) >
chess (3 × 2 × 1m3) for indoor. This suggests that smaller scenes (OldHospital,
stairs) can be more challenging than larger scenes (KingsCollege, redkitchen)
due to challenging contents like repetitive structures and texture-less regions.
Image retrieval on synthesized views. The goal of NeRF-only experiment is
to verify the possibility to use NeRF as the only scene representation removing
the need to maintain the original image collection. Our experiments show a
slight performance decrease due to the domain gap between rendered and real
images. Yet, we did not claim an efficient solution for online image retrieval
and NeRF rendering. Future research is needed to improve its runtime efficiency
either via caching scene reference poses in a hierarchical tree structure to fasten
the searching process or leveraging any available prior information such as GPS
coordinates to quickly find a subset of poses.
Indoor performance bottleneck.
NeRF predicted depth maps are used to
compute pseudo ground-truth for matching supervision. Incorrect depth predic-
tions can lead to misaligned feature correspondences. In contrast, image match-
ing, SCR, and APR methods use more accurate labels like Colmap camera poses
or 3D maps. For small-scale indoor scenes, precise supervision is essential to
achieve centimeter-level errors. Our method based on feature matching, however,
scales better than regression-based approaches in larger outdoor scenes. Intro-
ducing uncertainty measures to ignore inaccurate matches, as in [15], and im-
proved NeRF reconstructions with accurate depth maps will benefit our method.
References
1. Arandjelovic, R., Gronat, P., Torii, A., Pajdla, T., Sivic, J.: Netvlad: Cnn ar-
chitecture for weakly supervised place recognition. In: Proceedings of the IEEE
conference on computer vision and pattern recognition. pp. 5297–5307 (2016)
2. Arth, C., Wagner, D., Klopschitz, M., Irschara, A., Schmalstieg, D.: Wide area
localization on mobile phones. In: 2009 8th ieee international symposium on mixed
and augmented reality. pp. 73–82. IEEE (2009)
3. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srini-
vasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance

<!-- page 19 -->
NeRFMatch | Supplementary
19
fields. In: Proceedings of the IEEE/CVF International Conference on Computer
Vision. pp. 5855–5864 (2021)
4. Berton, G., Masone, C., Caputo, B.: Rethinking visual geo-localization for large-
scale applications. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition. pp. 4878–4888 (2022)
5. Blanton, H., Greenwell, C., Workman, S., Jacobs, N.: Extending absolute pose
regression to multiple scenes. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition Workshops. pp. 38–39 (2020)
6. Brachmann, E., Cavallari, T., Prisacariu, V.A.: Accelerated coordinate encoding:
Learning to relocalize in minutes using rgb and poses. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5044–
5053 (2023)
7. Brachmann, E., Humenberger, M., Rother, C., Sattler, T.: On the limits of pseudo
ground truth in visual camera re-localisation. In: Proceedings of the IEEE/CVF
International Conference on Computer Vision. pp. 6218–6228 (2021)
8. Brachmann, E., Krull, A., Nowozin, S., Shotton, J., Michel, F., Gumhold, S.,
Rother, C.: Dsac-differentiable ransac for camera localization. In: Proceedings of
the IEEE conference on computer vision and pattern recognition. pp. 6684–6692
(2017)
9. Brachmann, E., Rother, C.: Learning less is more-6d camera localization via 3d
surface regression. In: Proceedings of the IEEE conference on computer vision and
pattern recognition. pp. 4654–4662 (2018)
10. Brachmann, E., Rother, C.: Visual camera re-localization from rgb and rgb-d im-
ages using dsac. IEEE transactions on pattern analysis and machine intelligence
44(9), 5847–5865 (2021)
11. Brahmbhatt, S., Gu, J., Kim, K., Hays, J., Kautz, J.: Geometry-aware learning of
maps for camera localization. In: Proceedings of the IEEE conference on computer
vision and pattern recognition. pp. 2616–2625 (2018)
12. Camposeco, F., Cohen, A., Pollefeys, M., Sattler, T.: Hybrid scene compression
for visual localization. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition. pp. 7653–7662 (2019)
13. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-
to-end object detection with transformers. In: European conference on computer
vision. pp. 213–229. Springer (2020)
14. Chen, H., Luo, Z., Zhou, L., Tian, Y., Zhen, M., Fang, T., Mckinnon, D., Tsin, Y.,
Quan, L.: Aspanformer: Detector-free image matching with adaptive span trans-
former. In: European Conference on Computer Vision. pp. 20–36. Springer (2022)
15. Chen, L., Chen, W., Wang, R., Pollefeys, M.: Leveraging neural radiance fields for
uncertainty-aware visual localization. arXiv preprint arXiv:2310.06984 (2023)
16. Chen, S., Bhalgat, Y., Li, X., Bian, J., Li, K., Wang, Z., Prisacariu, V.A.: Re-
finement for absolute pose regression with neural feature synthesis. arXiv preprint
arXiv:2303.10087 (2023)
17. Chen, S., Li, X., Wang, Z., Prisacariu, V.A.: Dfnet: Enhance absolute pose regres-
sion with direct feature matching. In: European Conference on Computer Vision.
pp. 1–17. Springer Nature Switzerland Cham (2022)
18. Chen, S., Wang, Z., Prisacariu, V.: Direct-posenet: Absolute pose regression with
photometric consistency. In: 2021 International Conference on 3D Vision (3DV).
pp. 1175–1185. IEEE (2021)
19. Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention
mask transformer for universal image segmentation. In: Proceedings of the

<!-- page 20 -->
20
IEEE/CVF conference on computer vision and pattern recognition. pp. 1290–1299
(2022)
20. DeTone, D., Malisiewicz, T., Rabinovich, A.: Superpoint: Self-supervised interest
point detection and description. In: Proceedings of the IEEE conference on com-
puter vision and pattern recognition workshops. pp. 224–236 (2018)
21. Dong, Z., Zhang, G., Jia, J., Bao, H.: Keyframe-based real-time camera tracking.
In: 2009 IEEE 12th international conference on computer vision. pp. 1538–1545.
IEEE (2009)
22. Dusmanu, M., Rocco, I., Pajdla, T., Pollefeys, M., Sivic, J., Torii, A., Sattler, T.:
D2-net: A trainable cnn for joint description and detection of local features. In:
Proceedings of the ieee/cvf conference on computer vision and pattern recognition.
pp. 8092–8101 (2019)
23. Fu, X., Zhang, S., Chen, T., Lu, Y., Zhu, L., Zhou, X., Geiger, A., Liao, Y.:
Panoptic nerf: 3d-to-2d label transfer for panoptic urban scene segmentation. In:
2022 International Conference on 3D Vision (3DV). pp. 1–11. IEEE (2022)
24. Gao, X.S., Hou, X.R., Tang, J., Cheng, H.F.: Complete solution classification for
the perspective-three-point problem. IEEE transactions on pattern analysis and
machine intelligence 25(8), 930–943 (2003)
25. Hausler, S., Garg, S., Xu, M., Milford, M., Fischer, T.: Patch-netvlad: Multi-scale
fusion of locally-global descriptors for place recognition. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14141–
14152 (2021)
26. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In:
Proceedings of the IEEE conference on computer vision and pattern recognition.
pp. 770–778 (2016)
27. Heng, L., Choi, B., Cui, Z., Geppert, M., Hu, S., Kuan, B., Liu, P., Nguyen, R., Yeo,
Y.C., Geiger, A., et al.: Project autovision: Localization and 3d scene perception
for an autonomous vehicle with a multi-camera system. In: 2019 International
Conference on Robotics and Automation (ICRA). pp. 4695–4702. IEEE (2019)
28. Hu, B., Huang, J., Liu, Y., Tai, Y.W., Tang, C.K.: Nerf-rpn: A general framework
for object detection in nerfs. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. pp. 23528–23538 (2023)
29. Irschara, A., Zach, C., Frahm, J.M., Bischof, H.: From structure-from-motion point
clouds to fast location recognition. In: 2009 IEEE Conference on Computer Vision
and Pattern Recognition. pp. 2599–2606. IEEE (2009)
30. Ke, T., Roumeliotis, S.I.: An efficient algebraic solution to the perspective-three-
point problem. In: Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition. pp. 7225–7233 (2017)
31. Kendall, A., Cipolla, R.: Geometric loss functions for camera pose regression with
deep learning. In: Proceedings of the IEEE conference on computer vision and
pattern recognition. pp. 5974–5983 (2017)
32. Kendall, A., Grimes, M., Cipolla, R.: Posenet: A convolutional network for real-
time 6-dof camera relocalization. In: Proceedings of the IEEE international con-
ference on computer vision. pp. 2938–2946 (2015)
33. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. 3rd Interna-
tional Conference on Learning Representations, (ICLR) 2015 (2015)
34. Kneip, L., Scaramuzza, D., Siegwart, R.: A novel parametrization of the
perspective-three-point problem for a direct computation of absolute camera posi-
tion and orientation. In: CVPR 2011. pp. 2969–2976. IEEE (2011)

<!-- page 21 -->
NeRFMatch | Supplementary
21
35. Kundu, A., Genova, K., Yin, X., Fathi, A., Pantofaru, C., Guibas, L.J., Tagliasac-
chi, A., Dellaert, F., Funkhouser, T.: Panoptic neural fields: A semantic object-
aware neural scene representation. In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. pp. 12871–12881 (2022)
36. Li, X., Wang, S., Zhao, Y., Verbeek, J., Kannala, J.: Hierarchical scene coordi-
nate classification and regression for visual localization. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11983–
11992 (2020)
37. Li, Y., Snavely, N., Huttenlocher, D.P.: Location recognition using prioritized fea-
ture matching. In: European conference on computer vision. pp. 791–804. Springer
(2010)
38. Liu, J., Nie, Q., Liu, Y., Wang, C.: Nerf-loc: Visual localization with conditional
neural radiance field. 2023 IEEE International Conference on Robotics and Au-
tomation (ICRA) (2023)
39. Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts.
arXiv preprint arXiv:1608.03983 (2016)
40. Maggio, D., Abate, M., Shi, J., Mario, C., Carlone, L.: Loc-nerf: Monte carlo lo-
calization using neural radiance fields. In: 2023 IEEE International Conference on
Robotics and Automation (ICRA). pp. 4018–4025. IEEE (2023)
41. Martin-Brualla, R., Radwan, N., Sajjadi, M.S., Barron, J.T., Dosovitskiy, A., Duck-
worth, D.: Nerf in the wild: Neural radiance fields for unconstrained photo col-
lections. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 7210–7219 (2021)
42. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng,
R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: Eu-
ropean Conference on Computer Vision. pp. 405–421. Springer (2020)
43. Moreau,
A.,
Piasco,
N.,
Bennehar,
M.,
Tsishkou,
D.,
Stanciulescu,
B.,
de La Fortelle, A.: Crossfire: Camera relocalization on self-supervised features from
an implicit representation. arXiv preprint arXiv:2303.04869 (2023)
44. Moreau, A., Piasco, N., Tsishkou, D., Stanciulescu, B., de La Fortelle, A.: Lens:
Localization enhanced by nerf synthesis. In: Conference on Robot Learning. pp.
1347–1356. PMLR (2022)
45. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with
a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1–
15 (2022)
46. Panek, V., Kukelova, Z., Sattler, T.: Meshloc: Mesh-based visual localization. In:
European Conference on Computer Vision. pp. 589–609. Springer (2022)
47. Rematas, K., Liu, A., Srinivasan, P.P., Barron, J.T., Tagliasacchi, A., Funkhouser,
T., Ferrari, V.: Urban radiance fields. In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. pp. 12932–12942 (2022)
48. Revaud, J., De Souza, C., Humenberger, M., Weinzaepfel, P.: R2d2: Reliable and
repeatable detector and descriptor. Advances in neural information processing sys-
tems 32 (2019)
49. Rosinol, A., Leonard, J.J., Carlone, L.: Nerf-slam: Real-time dense monocular slam
with neural radiance fields. arXiv preprint arXiv:2210.13641 (2022)
50. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z.,
Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recog-
nition challenge. International journal of computer vision 115, 211–252 (2015)
51. Sarlin, P.E., Cadena, C., Siegwart, R., Dymczyk, M.: From coarse to fine: Robust
hierarchical localization at large scale. In: Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition. pp. 12716–12725 (2019)

<!-- page 22 -->
22
52. Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A.: Superglue: Learning
feature matching with graph neural networks. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 4938–4947 (2020)
53. Sarlin, P.E., Unagar, A., Larsson, M., Germain, H., Toft, C., Larsson, V., Pollefeys,
M., Lepetit, V., Hammarstrand, L., Kahl, F., et al.: Back to the feature: Learning
robust camera localization from pixels to pose. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 3247–3257 (2021)
54. Sattler, T., Leibe, B., Kobbelt, L.: Efficient & effective prioritized matching for
large-scale image-based localization. IEEE transactions on pattern analysis and
machine intelligence 39(9), 1744–1756 (2016)
55. Sattler, T., Zhou, Q., Pollefeys, M., Leal-Taixe, L.: Understanding the limitations
of cnn-based absolute camera pose regression. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 3302–3312 (2019)
56. Shavit, Y., Ferens, R., Keller, Y.: Learning multi-scene absolute pose regression
with transformers. In: Proceedings of the IEEE/CVF International Conference on
Computer Vision. pp. 2733–2742 (2021)
57. Shotton, J., Glocker, B., Zach, C., Izadi, S., Criminisi, A., Fitzgibbon, A.: Scene
coordinate regression forests for camera relocalization in rgb-d images. In: Pro-
ceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
pp. 2930–2937 (2013)
58. Sucar, E., Liu, S., Ortiz, J., Davison, A.J.: imap: Implicit mapping and position-
ing in real-time. In: Proceedings of the IEEE/CVF International Conference on
Computer Vision. pp. 6229–6238 (2021)
59. Sun, J., Shen, Z., Wang, Y., Bao, H., Zhou, X.: Loftr: Detector-free local fea-
ture matching with transformers. In: Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. pp. 8922–8931 (2021)
60. Taira, H., Okutomi, M., Sattler, T., Cimpoi, M., Pollefeys, M., Sivic, J., Pajdla,
T., Torii, A.: Inloc: Indoor visual localization with dense matching and view syn-
thesis. In: Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition. pp. 7199–7209 (2018)
61. Tancik, M., Casser, V., Yan, X., Pradhan, S., Mildenhall, B., Srinivasan, P.P., Bar-
ron, J.T., Kretzschmar, H.: Block-nerf: Scalable large scene neural view synthesis.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. pp. 8248–8258 (2022)
62. Tang, S., Tang, C., Huang, R., Zhu, S., Tan, P.: Learning camera localization via
dense scene matching. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition. pp. 1831–1841 (2021)
63. Tang, S., Tang, S., Tagliasacchi, A., Tan, P., Furukawa, Y.: Neumap: Neural co-
ordinate mapping by auto-transdecoder for camera localization. In: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp.
929–939 (2023)
64. Torii, A., Arandjelovic, R., Sivic, J., Okutomi, M., Pajdla, T.: 24/7 place recogni-
tion by view synthesis. In: Proceedings of the IEEE conference on computer vision
and pattern recognition. pp. 1808–1817 (2015)
65. Ventura, J., Arth, C., Reitmayr, G., Schmalstieg, D.: Global localization from
monocular slam on a mobile phone. IEEE transactions on visualization and com-
puter graphics 20(4), 531–539 (2014)
66. Walch, F., Hazirbas, C., Leal-Taixe, L., Sattler, T., Hilsenbeck, S., Cremers, D.:
Image-based localization using lstms for structured feature correlation. In: Pro-
ceedings of the IEEE International Conference on Computer Vision. pp. 627–637
(2017)

<!-- page 23 -->
NeRFMatch | Supplementary
23
67. Wang, Q., Zhou, X., Hariharan, B., Snavely, N.: Learning feature descriptors using
camera pose supervision. In: European Conference on Computer Vision. pp. 757–
774. Springer (2020)
68. Wendel, A., Irschara, A., Bischof, H.: Natural landmark-based monocular localiza-
tion for mavs. In: 2011 IEEE International Conference on Robotics and Automa-
tion. pp. 5792–5799. IEEE (2011)
69. Xie, Y., Takikawa, T., Saito, S., Litany, O., Yan, S., Khan, N., Tombari, F., Tomp-
kin, J., Sitzmann, V., Sridhar, S.: Neural fields in visual computing and beyond.
Computer Graphics Forum (2022). https://doi.org/10.1111/cgf.14505
70. Xie, Z., Zhang, J., Li, W., Zhang, F., Zhang, L.: S-nerf: Neural radiance fields for
street views. arXiv preprint arXiv:2303.00749 (2023)
71. Xu, C., Wu, B., Hou, J., Tsai, S., Li, R., Wang, J., Zhan, W., He, Z., Vajda, P.,
Keutzer, K., et al.: Nerf-det: Learning geometry-aware volumetric representation
for multi-view 3d object detection. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision. pp. 23320–23330 (2023)
72. Yang, L., Bai, Z., Tang, C., Li, H., Furukawa, Y., Tan, P.: Sanet: Scene agnostic
network for camera localization. In: Proceedings of the IEEE/CVF international
conference on computer vision. pp. 42–51 (2019)
73. Yen-Chen, L., Florence, P., Barron, J.T., Rodriguez, A., Isola, P., Lin, T.Y.: inerf:
Inverting neural radiance fields for pose estimation. In: 2021 IEEE/RSJ Interna-
tional Conference on Intelligent Robots and Systems (IROS). pp. 1323–1330. IEEE
(2021)
74. Yu, W., Si, C., Zhou, P., Luo, M., Zhou, Y., Feng, J., Yan, S., Wang, X.: Metaformer
baselines for vision. arXiv preprint arXiv:2210.13452 (2022)
75. Zhang, Y., Tosi, F., Mattoccia, S., Poggi, M.: Go-slam: Global optimization for con-
sistent 3d instant reconstruction. In: Proceedings of the IEEE/CVF International
Conference on Computer Vision. pp. 3727–3737 (2023)
76. Zhi, S., Laidlow, T., Leutenegger, S., Davison, A.J.: In-place scene labelling and un-
derstanding with implicit scene representation. In: Proceedings of the IEEE/CVF
International Conference on Computer Vision. pp. 15838–15847 (2021)
77. Zhou, Q., Agostinho, S., Ošep, A., Leal-Taixé, L.: Is geometry enough for matching
in visual localization? In: European Conference on Computer Vision. pp. 407–425.
Springer (2022)
78. Zhou, Q., Sattler, T., Leal-Taixe, L.: Patch2pix: Epipolar-guided pixel-level corre-
spondences. In: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition. pp. 4669–4678 (2021)
79. Zhu, Z., Peng, S., Larsson, V., Xu, W., Bao, H., Cui, Z., Oswald, M.R., Pollefeys,
M.: Nice-slam: Neural implicit scalable encoding for slam. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12786–
12796 (2022)
