<!-- page 1 -->
GSVisLoc: Generalizable Visual Localization for Gaussian Splatting Scene
Representations
Fadi Khatib1∗, Dror Moran1∗, Guy Trostianetsky1, Yoni Kasten2, Meirav Galun1, Ronen Basri1
1Weizmann Institute of Science
2NVIDIA
Project Webpage: https://gsvisloc.github.io/
Abstract
We introduce GSVisLoc, a visual localization method de-
signed for 3D Gaussian Splatting (3DGS) scene repre-
sentations.
Given a 3DGS model of a scene and a
query image, our goal is to estimate the camera’s position
and orientation.
We accomplish this by robustly match-
ing scene features to image features. Scene features are
produced by downsampling and encoding the 3D Gaus-
sians while image features are obtained by encoding im-
age patches. Our algorithm proceeds in three steps, start-
ing with coarse matching, then fine matching, and finally
by applying pose refinement for an accurate final esti-
mate. Importantly, our method leverages the explicit 3DGS
scene representation for visual localization without requir-
ing modifications, retraining, or additional reference im-
ages.
We evaluate GSVisLoc on both indoor and out-
door scenes, demonstrating competitive localization perfor-
mance on standard benchmarks while outperforming exist-
ing 3DGS-based baselines. Moreover, our approach gen-
eralizes effectively to novel scenes without additional train-
ing.
1. Introduction
Visual localization is the problem of estimating the camera
position and orientation in a 3D environment given a query
image. Solving the localization problem is crucial in many
applications such as autonomous driving [30], robot navi-
gation [77], and augmented reality [71].
Visual localization approaches can be categorized by
the underlying scene representation; see review in Sec-
tion 2. Classical, structure-based methods [13, 31, 43, 55,
58, 65] rely on explicit 3D models with keypoint descrip-
tors, achieving accurate localization by matching 2D query
image pixels to 3D model points. End-to-end learned ap-
proaches implicitly encode the scene through the weights of
a neural network. These methods include absolute pose re-
* Equal contributors.
gression (APR) [5, 12, 33, 34, 60, 72] and scene coordinate
regression (SCR) methods [7–9, 11, 42, 61, 73, 78]. These
learned implicit representations are optimized exclusively
for visual localization. [19, 79] use NeRF models [49] to
refine the camera pose at test time through iterative render-
ing and pose adjustments, while [47, 51, 84] work toward
directly estimating 3D-2D correspondences from NeRF.
Recently, 3D Gaussian Splatting (3DGS) [35] has gained
attention as a promising scene representation for novel-
view synthesis (NVS), offering fast training times and high-
quality real-time rendering.
Its growing popularity has
spurred various studies exploring fundamental computer vi-
sion tasks within the 3DGS framework, including 3D seg-
mentation [14], registration [15], and 3D editing [20]. Sev-
eral methods have been proposed to address the localization
challenge for 3DGS. However, they either only refine an ini-
tial pose [46, 64, 70], exhibit lower accuracy compared to
other scene representations (6DGS [6]), or require optimiz-
ing a specific 3DGS model (GSplatLoc [62]).
In this paper, we propose to leverage the explicit scene
representation of 3DGS for visual localization. Specifically,
we introduce GSVisLoc, a generalizable deep-learning-
based network for visual localization by matching 3DGS
with 2D image features (see Figure 1). Given a query image
and a 3D Gaussian Splatting scene representation, we use a
KPConv-based encoder [68] to encode and downsample the
Gaussians, producing features representing regions in 3D
space. Additionally, we use an image encoder to extract
features representing image patches from the query image.
We then establish 3D-2D correspondences between the fea-
tures, first at a coarse scale and then at a fine scale, enabling
us to obtain an estimation for the camera pose. This is fol-
lowed by an additional pose refinement step to achieve the
final camera pose prediction.
Our method achieves accurate pose estimation that
surpasses previous 3DGS-based localization methods.
Moreover,
GSVisLoc’s accuracy is competitive with
state-of-the-art baselines on the 7-scenes dataset [28, 61].
Notably, our model generalizes to novel, unseen scenes
without requiring any modifications or retraining for the
1
arXiv:2508.18242v1  [cs.CV]  25 Aug 2025

<!-- page 2 -->
Figure 1. GSVisLoc. Our method estimates coarse 3D–2D matches (shown in red in the figure) between 3D regions in the 3DGS and
patches in the query image, and then refines them to pixel-level correspondences (yellow). These fine 3D–2D matches are fed into a PnP +
RANSAC pipeline, followed by a pose refinement module, to produce the final camera pose of the query image.
3DGS representation, and it eliminates the need for an
image retrieval step.
In summary, our contributions include:
1. We introduce GSVisLoc, a deep neural network for vi-
sual localization for scenes represented using 3D Gaus-
sian Splatting (3DGS).
2. Our approach requires no retraining or modifications to
the underlying 3DGS representation, simplifying its de-
ployment.
3. In contrast to structure-based and NeRF-based ap-
proaches, GSVisLoc eliminates the need for image re-
trieval during inference, allowing reference images to be
discarded after training.
4. GSVisLoc achieves state-of-the-art performance among
3DGS-based methods across all evaluated datasets, with
accuracy competitive with state-of-the-art methods on
the 7-scenes dataset.
5. Most importantly, GSVisLoc generalizes to novel, un-
seen scenes through learned 3D-2D matching, retaining
accuracy comparable to single-scene trained models.
2. Related work
Visual localization approaches can be grouped according to
their underlying scene representation:
Structure-based Localization. Structure-based localiza-
tion methods [13, 31, 43, 55, 58, 65] operate by establishing
3D-2D correspondences between 3D scene points and fea-
tures in the query image. Camera pose is then calculated
using a Perspective-n-Point (PnP) solver [26, 32, 38]. To
streamline the matching process, an initial image retrieval
step is used [1, 4, 29, 69] to coarsely localize the query im-
age, effectively narrowing the search space [41, 76, 83].
While this approach achieves accurate results, it requires
substantial storage resources. Visual features [16, 22, 23,
56, 63, 75, 82] are typically extracted from a database of
scene images to represent 3D points, and these features
are then matched to features extracted from the query im-
age to establish 3D-2D correspondences. To expedite in-
ference, 3D descriptors are precomputed and stored along-
side the scene model. This design, however, can lead to
large memory footprints, complicating map updates and
scalability [83]. Recent work has shown that storage re-
quirements can be alleviated through compression and de-
scriptor quantization [39, 74]. Complementary to compres-
sion, descriptor-free 2D–3D matching establishes pixel-to-
point correspondences via learned geometric and photomet-
ric reasoning, avoiding persistent high-dimensional 3D de-
scriptors [41, 76, 83].
End-to-End Learned Localization. Absolute Pose Re-
gression (APR) methods [5, 12, 33, 34, 60, 72] train mod-
els to directly predict the camera pose from a query image,
eliminating the need for explicit 3D representations. Al-
though APR methods offer high-speed performance, they
typically lack the accuracy and generalization capabilities
of structure-based approaches [59].
Relative Pose Regression (RPR) methods predict the
transformation between image pairs instead of the absolute
pose of a single query. This often improves generalization
over APR [2, 3, 24, 36, 48], but RPR alone still falls short
of the accuracy achieved by structure-based approaches.
Scene Coordinate Regression (SCR) methods [7–9, 11,
42, 61, 73, 78] perform implicit 3D-2D matching by re-
gressing the 3D scene coordinates directly from the query
image. Like APR, these approaches use network parame-
2

<!-- page 3 -->
ters to encode the geometry of the scene [7–9, 11, 42], but
they are limited in representing large-scale scenes due to
memory constraints. To address this limitation, more re-
cent scene-agnostic SCR methods [67, 78] have decoupled
the scene representation from the learned matching func-
tion, achieving scalability to larger scenes.
NeRF-based Pose Estimation. NeRF-based pose estima-
tion methods [19, 44, 52, 79] rely on iterative rendering and
pose adjustments, resulting in slow convergence and lim-
ited accuracy. NeFeS [19] improves APR pose estimation
but struggles with SCR enhancements and suffers from a
lengthy refinement time. HR-APR [45] accelerates the op-
timization process, yet each query still takes several sec-
onds, even on a high-performance GPU. Other NeRF-based
approaches, such as FQN [27], CrossFire [51], NeRFLoc
[47], and NeRFMatch [84], enhance positioning accuracy
by establishing 3D-2D correspondences. However, these
methods require an image retrieval step during inference.
3DGS-based Pose Estimation. Following the recent intro-
duction of 3D Gaussian-Splatting (3DGS) for novel view
synthesis (NVS), several methods have explored its use
also for pose estimation. 6DGS [6], which is closely re-
lated to our approach, achieves one-shot pose estimation
by projecting rays from an ellipsoid surface, thus avoid-
ing iterative processing. Although 6DGS uses 3DGS for
visual localization, it achieves inferior accuracies compared
to previous methods. GSplatLoc [62], a concurrent work,
trains a Feature-3DGS model [85] in which each Gaus-
sian is assigned a feature through training the Gaussian
splatting model. This approach establishes correspondences
by matching 3D to 2D features through a mutual nearest-
neighbor search. It then uses the PnP algorithm to obtain
an initial coarse pose estimate. The pose is subsequently
refined iteratively to improve accuracy. In contrast to this
method, our method uses the vanilla 3DGS model [35] and
does not require optimizing a specific Gaussian model.
3DGS-based Pose Refinement.
Recent works have uti-
lized 3DGS specifically for pose refinement.
For exam-
ple, iComMa [64], and MCLoc [70], inspired by iNeRF
[79], implements an iterative refinement process for cam-
era pose estimation by inverting 3DGS. Similarly, GS-CPR
[46] combines 3DGS with Mast3R [40], a powerful image
matcher, to apply pose refinement.
Our method establishes precise 3D-2D correspondences
and seamlessly integrates GS-CPR’s pipeline as a plug-and-
play module for pose refinement, achieving superior perfor-
mance over other 3DGS-based methods.
3. Method
3.1. Overview
We aim to establish point correspondences between a 3DGS
scene representation and a 2D query image of the same
scene. With these 3D-2D matches, a Perspective-n-Point
(PnP) solver [26, 32, 38] can efficiently estimate the cam-
era pose.
To accomplish this, we propose GSVisLoc, a
deep neural network for visual localization. Our network
encodes the 3D Gaussians and the query image in latent
feature spaces. It then matches similar features and uses
those matches to estimate the camera parameters. We fur-
ther enhance our results by refining the initial pose estimate.
Details of the architecture, training process, and pose refine-
ment are provided in the following sections.
3.2. GSVisLoc
Our network consists of a 3DGS encoder, a 2D encoder,
and cross-modal matching modules to align features from
both domains and establish 3D-2D correspondences. The
2D encoder encodes two types of features for coarse and
fine matching. Refer to Figure 2 for a detailed schematic of
our network architecture.
2D image encoding. We use a 2D encoder to map image
patches to a latent representation. Each latent vector en-
codes an image patch associated with a 2D location. Given
a query image I ∈RH×W , we use two levels of encodings:
Coarse 2D features F c
m ∈RNc×Cc, where Nc = H
8 × W
8 ,
and fine 2D features F f
m ∈RNf ×Cf , where Nf = H
2 × W
2 .
Cc and Cf denote the encoding dimensions.
3DGS filtering. Since the number of Gaussians can be very
large (typically 700K), it is impractical to match all of them
with the 2D features. Therefore, we first filter Gaussians
with low opacity value and then randomly select a subset of
ˆNg Gaussians.
3DGS encoding.
We aim to match the locations of 3D
Gaussians with image patches by aligning their features.
Since the information in a single Gaussian is not rich
enough to enable accurate matching, we construct an en-
coding incorporating also information from nearby Gaus-
sians.
To this end, we use KPFCNN [68], an encoder
based on Kernel Point Convolution (KPConv) that applies
convolution on a sphere.
Additionally, KPFCNN incor-
porates downsampling layers that operate on a grid to re-
duce the number of 3D points to Ng. The encoder receives
the parameters of the Gaussians as input, including their
opacity, radiance (represented by spherical harmonic coef-
ficients), orientation, and scaling. The outputs are features
Fs ∈RNg×Cc associated with 3D points Q ∈RNg×3, such
that each feature represents a 3D region of Gaussians.
3D-2D features alignment. Next, given the scene and im-
age features, we apply a series of interleaved self- and cross-
attention layers. The self-attention layers operate indepen-
dently on the 3D and 2D features, allowing each domain to
refine its representations and capture local and global de-
pendencies. The cross-attention layers enable interaction
between the 3D and 2D features, aligning the two domains
by focusing on shared information. This interleaved process
3

<!-- page 4 -->
Down 
sampler
3D 
encoder
2D 
encoder
Coarse 3D-2D 
feature 
alignment
Pose 
refinement
෠𝑅, Ƹ𝑡
𝑁𝑔× 𝑁𝑐
PnP
RANSAC
GS scene 
representation
Query image
Matched 
3D feature
Coarse 3D-2D 
matches
Matched 
2D features
Fine 3D-2D 
matches
Fine 3D-2D 
feature 
alignment
Figure 2. GSVisLoc architecture. GSVisLoc uses a 3D Gaussian Splatting (3DGS) scene representation, processed by a 3D encoder,
and a query image processed by a 2D encoder. Coarse 3D-2D matching establishes initial correspondences between the image and the
3D scene, which are refined to pixel-level matches. The final 3D-2D correspondences are passed through PnP with RANSAC for pose
estimation, followed by a pose refinement step, yielding the final pose ( \ha t {R}, \hat {t}) of the query image.
allows for an iterative update of feature representations, fa-
cilitating precise 3D-2D correspondences.
Coarse-level matching. Next, we compute pairwise cosine
similarities between the coarse image features  F
_m^c and the
3DGS features  F_s , after we map them to a shared feature
space, obtaining a matching score matrix  \ mathcal {S} \in \mathbb {R}^{N_{m} \times N_{s}} .
We then select matches whose scores exceed a threshold  \theta _{c} 
while applying the mutual nearest neighbor (MNN) crite-
rion to filter out potential outliers, yielding the final set of
coarse correspondences:
  \ math ca l  {M }_ { c} = \{ (i, j)  \mid (i, j) \in \operatorname {MNN}(\mathcal {S}), \, \mathcal {S}(i, j) \geq \theta _{c} \}. \label {eq:coarse-matches} 
Coarse-to-Fine Module.
For each coarse match  m _
c =  ( i , j) \in \mathcal {M}_{c} , we consider the fine-level feature  F
_m^f (
i) \in \mathbb {R}^{w \times w \times D^f} corresponding to an image patch of size  w
 \times w centered at the match location. We next apply a self-
attention layer individually to each patch  F
_m^f(I) to enhance
contextual information, while in parallel, we apply a linear
layer to the coarse scene features to match the fine-level
dimension  D^f .
Finally, we align the scene features with their corre-
sponding fine image features. To that end, For each scene
point,  X_j , we generate a heatmap expressing the probability
that it aligns with each pixel near the image location  x_i .
The refined match location is then determined by taking
an expectation over this probability distribution, resulting in
the final fine-grained matches, denoted by  \mathcal {M}_f .
3.3. Training
For training, we use query images with known camera poses
extracted from a Structure-from-Motion pipeline. We also
use the output of this SfM pipeline to train our Gaussian
Splatting scene representations. We train our model in three
different pipelines: In Single-scene training, we train a
model on the Gaussians and images of a single scene and
evaluate the model on test images from the same scene (in-
scene generalization). In Multi-scene training, we train a
model on the Gaussians and images of a collection of scenes
and evaluate the model on test images from the same set of
scenes. Lastly, for Cross-scene generalization, we train a
model on the Gaussians and images of a collection of scenes
and evaluate the model on test images from novel unseen
scenes.
Ground-truth matches. We obtain ground-truth matches
for training by projecting the 3DGS points Xs onto the
query image using the ground-truth query camera pose and
use their projected locations as their fine-level matches.
For the coarse ground-truth matches, we compute a binary
coarse association matrix Mgt as follows. For each 3D point
Xj, we use its projected location to assign it to the ith 8×8
patch to which it was matched, letting Mgt(i, j) = 1 if Xj
projects inside the 2D patch i. Notice that a 3D point can
match at most one image patch, yet a single image patch
can match multiple 3D points.
4

<!-- page 5 -->
Losses. The final loss consists of the losses for the coarse-
level and the fine-level: L = Lc + Lf:
Coarse Loss. To guide the coarse matching, we apply the
log loss from [63]. This will work to increase the dual-
softmax probability at the ground-truth matching locations
in Mgt. This loss is defined as
 \ l a
b
el {e
q
:coarse_m
atching_ loss} L_c = - \frac {1}{|\mathcal {M}_{gt}|} \sum _{(i, j) \in \mathcal {M}_{gt}} \log (S(i, j)). 
(1)
where  |\mathcal {M}_{gt}| is the total number of ground-truth coarse
matches, and  S(i ,j) denotes the matching score between
the image patch  i and the 3D point  j .
Fine Loss. Let  X_j and  x_j respectively denote a 3D point
and its ground-truth match, obtained by projecting  X_j using
the ground-truth camera pose. We set the fine matching loss
to minimize the pixel distance between the predicted loca-
tion,  \tilde {x}_j , and the ground-truth location,  x_j , and, following
[63, 75], weigh this distance by the total variance  \sigma ^{2}(j) of
the corresponding heatmap to penalize more heavily devia-
tions in matches that are more certain. The loss is given by
 \ l
a
bel 
{
eq:fine_
m
atching_lo s s} L_f = \frac {1}{|M_{f}|} \sum _{(i, j)\in M_f} \frac {1}{\sigma ^2(j)} ||\tilde {x}_j - x_j||_2. 
(2)
where  |\mathcal {M}_{f}| denotes the total number of predicted fine
matches.
3.4. Pose refinement
We follow GS-CPR [46] for pose refinement. Starting from
an estimated pose of the query image, we use the 3DGS rep-
resentation to render both an image and a depth map. We
next use MASt3R [40] to establish dense 2D-2D correspon-
dences between the query image and the rendered image.
The matched points in the rendered image are lifted to 3D
using the rendered depth, the estimated pose, and camera in-
trinsics, creating 3D-2D correspondences for the query im-
age. Finally, we apply PnP with RANSAC [25] to solve for
the refined pose.
4. Experiments
4.1. Experimental setup
Datasets.
We conduct single-scene training on two well-
established localization datasets.
The 7-Scenes dataset
[28, 61] consists of RGB-D images captured across seven
unique indoor scenes (volumes ranging from 1m3 to 18m3)
that are challenging due to the presence of texture-less sur-
faces, motion blur, and occlusions. We follow the origi-
nal train/test splits and use more accurate SfM pose annota-
tions, as recommended by [10, 11, 18], for both our method
and all baselines.
The Cambridge Landmarks dataset [34] contains
handheld smartphone images from outdoor scenes, each
characterized by significant exposure variations that com-
plicate large-scale localization. Consistent with previous
works [62, 70], we evaluate on King’s College, Old Hos-
pital, Shop facade, and St Mary’s church (spanning 875 -
5600m^{2}), following the original splits.
For cross-scene generalization, we use ScanNet++ [80],
a large-scale indoor dataset that couples high-quality (sub-
millimeter laser scans) and commodity-level (registered im-
ages) geometry and color captures.
Unlike setups that
merely hold out a portion of training images along the same
camera trajectory, ScanNet++ provides novel test views per
scene, leading to more realistic and challenging conditions
for cross-scene visual localization.
The dataset encom-
passes large and varied indoor environments with numerous
glossy and reflective surfaces, making accurate localization
particularly difficult. We randomly sample 130 scenes for
training, 7 for validation, and 15 for testing.
Evaluation metrics. In line with previous works, we report
median pose errors, specifically translation error in centime-
ters and rotation error in degrees. On the 7-Scenes dataset,
we also report localization recall, which measures the per-
centage of query images localized with pose errors below
specified thresholds—namely, 5\text {cm} for the translation and
5^\circ for the rotation.
Baselines.
We compared our method to MS-Trans. [60],
DFNet [17],
LENS [50],
NeFeS [18],
DSAC* [8],
HACNet [42],
ACE [11],
SANet [78],
DSM [66],
NeuMap [67], InLoc[65], HLoc[55], PixLoc[57], Cross-
Fire [51], NeRFLoc [47], NeRFMatch [84], GSplatLoc
[62], DVLAD+R2D2[53, 69] and 6DGS [6].
Following
[84], we categorize the methods into three groups: end-
to-end methods, which include APR and SCR methods;
hierarchical methods, where an initialized camera pose is
estimated using an image retrieval step; and 3DGS-based
methods. Additionally, we include the scene representa-
tion used for localization during testing. For experiments on
Cambridge and 7-scenes, all the evaluations of the baseline
methods were taken from previous works except for 6DGS
[6], which we trained and evaluated using the official im-
plementation. Since ScanNet++ is a relatively new dataset,
there are no publicly available baseline evaluations. There-
fore, we compared our cross-scene generalization capabil-
ities to both GSPlatLoc and our method in a single-scene
training pipeline.
Implementation details.
We use the first two blocks of
ConvFormer [81] as the image backbone, initialized with
ImageNet-1K [54] pre-trained weights1.
The feature di-
mensions are set to  D ^ c = 512 for the coarse matching and
 D ^ f = 128 for the fine matching. For fine matching, a local
window size of  w  = 5 is used for image feature cropping.
θc = 0.3 the score threshold for the coarse matches. Query
1Pre-trained weights can be found here: huggingface.co/timm/
convformer_b36.sail_in1k_384
5

<!-- page 6 -->
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
End-to-End
MS-Trans. [60]
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
NeFeS [18]
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
DSAC* [8]
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
ACE [11]
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
Hierarchical
DVLAD+R2D2[53, 69] 3D+RGB
0.4/0.1
0.5/0.2
0.4/0.2
0.7/0.2
0.6/0.1
0.4/0.1
2.4/0.7
0.8/0.2
95.7
HLoc[55]
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
NeRFLoc [47]
NeRF+RGBD
2/1.1
2/1.1
1/1.9
2/1.1
3/1.3
3/1.5
3/1.3
2.3/1.3
-
NeRFMatch [84]
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
GS-based
6DGS [6]
3DGS
26.8/28.7 33.3/36.8 17.3/33.7 37.6/31.0 22.1/28.0 42.5/35.7 47.5/31.7
32.4/32.2
-
GSplatLoc [62]
3DGS
0.43/0.16 1.03/0.32 1.06/0.62
1.85/0.4
1.8/0.35
2.71/0.55 8.83/2.34
2.53/0.68
-
GSVisLoc (Ours)
3DGS
0.39/0.13 0.58/0.24 0.54/0.34
1.0/0.26
0.90/0.21 0.73/0.18
4.7/0.96
1.3/0.33
88.0
Table 1. Indoor Localization on 7-Scenes [28, 61]. We report per-scene median position errors (in cm) and rotation errors (in degrees),
along with their averages across scenes and the mean localization recall. The best result in each column is highlighted in bold, and the best
result for the GS representation is underlined.
Method
Scene
Cambridge Landmarks - Outdoor
Repres.
Kings
Hospital
Shop
StMary
Avg.Med↓
End-to-End
MS-Trans. [60]
APR Net.
83/1.5
181/2.4
86/3.1
162/4
128/2.8
DFNet [17]
APR Net.
73/2.4
200 /3
67/2.2
137/4
119.3/2.9
LENS [50]
APR Net.
33/0.5
44/0.9
27/1.6
53/1.6
39.3/1.2
NeFeS [18]
APR+NeRF
37/0.6
55/0.9
14/0.5
32/1
34.5/0.8
DSAC* [8]
SCR Net.
15/0.3
21/0.4
5/0.3
13/0.4
13.5/0.4
HACNet [42]
SCR Net.
18/0.3
19/0.3
6/0.3
9/0.3
13/0.3
ACE [11]
SCR Net.
28/0.4
31/0.6
5/0.3
18/0.6
20.5/0.5
Hierarchical
SANet [78]
3D+RGB
32/0.5
32/0.5
10/0.5
16/0.6
22.5/0.5
DSM [66]
SCR Net.
19/0.4
24/0.4
7/0.4
12/0.4
15.5/0.4
NeuMap [67]
SCode+RGB
14/0.2
19/0.4
6/0.3
17/0.5
14/0.3
InLoc[65]
3D+RGB
46/0.8
48/1.0
11/0.5
18/0.6
30.8/0.7
HLoc[55]
3D+RGB
12/0.2
15/0.3
4/0.2
7/0.2
9.5/0.2
PixLoc[57]
3D+RGB
14/0.2
16/0.3
5/0.2
10/0.3
11/0.3
CrossFire [51]
NeRF+RGB
47/0.7
43/0.7
20/1.2
39/1.4
37.3/1
NeRFLoc [47]
NeRF+RGBD
11/0.2
18/0.4
4/0.2
7/0.2
10/0.3
NeRFMatch [84]
NeRF+RGB
13.0/0.2
19.4/0.4
8.5/0.4
7.9/0.3
12.2/0.3
GSplatLoc [62]
3DGS
27/0.46
20/0.71
5/0.36
16/0.61
17/0.53
GSVisLoc (Ours)
3DGS
23/0.3
22/0.42
8/0.29
14/0.45
17/0.36
Table 2. Outdoor Localization on Cambridge Landmarks [34]. We report per-scene median position errors (in cm) and rotation errors
(in degrees), along with the averages across scenes. The best result in each column is highlighted in bold, while the best result for the GS
representation is underlined.
images are resized to  48 0  \times 480 in all experiments. We fil-
ter Gaussians with opacity values lower than 0.9 and then
uniformly subsample 100K Gaussians. For the 3D back-
bone, we use a 3-stage KPFCNN [68] with output channels
of {128, 256, 512} for each stage. We use the output of the
final layer for encoding. The 3D-2D alignment module is
composed of four interleaved self- and cross-attention lay-
ers. Both KPFCN and 3D-2D alignment modules are ini-
tialized with random weights. For pose refinement, we re-
implemented GS-CPR, as the code is not yet publicly avail-
able. Unlike their approach, we use the original Gaussian
splatting model [35] and omit the exposure-adaptive trans-
formation. At inference, we apply three refinement itera-
tions for each query image.
We train our models using the Adam optimizer [37] with
a learning rate of  \ t ext {lr} = 0.0001 and batch size  \ t ext {bs} = 1 for
100 epochs. Our models are trained on a single Nvidia A40
GPU (48 GB). On a Quadro RTX 6000 and ScanNet++ im-
ages at 1752×1168, our average per-query time is 2.82 s
(1.03 s inference + 1.79 s refinement). The current imple-
mentation is not fully optimized and can be further acceler-
ated. Further implementation details of the 3DGS are pro-
vided in the supplementary material.
Hyper-parameter search.
We determined the values of
6

<!-- page 7 -->
Figure 3. GSVisLoc Qualitative Results. Visualization of our model’s pose estimation on indoor scenes (three left columns) and outdoor
scenes (two right columns). Each row, from top to bottom, displays rendered images from the model’s estimated pose, rendered images
from the refined pose, and the corresponding query images.
Method
Chess
Fire
Heads
Office
Pump.
Kitchen
Stairs
Avg.Med↓Avg.Recall↑.
Single scene 0.39/0.13 0.58/0.24 0.54/0.34
1.0/0.26
0.90/0.21
0.73/0.18
4.7/0.96
1.3/0.33
88.0
Multi. scene
0.59/0.15 0.73/0.28 0.57/0.37
1.5/0.36
1.0/0.21
1.1/0.26
4.5/0.9
1.4/0.36
84.4
Table 3. Multi-scene Model on 7-Scenes [28, 61]. We evaluate our multi-scene model against a single-scene model, reporting per-scene
median position errors (cm) and rotation errors (degrees), their averages across scenes, and the mean localization recall. The best result in
each column is shown in bold.
hyper-parameters by running our models on the validation
split from the 7-scenes datasets. These hyper-parameters
include the learning rate, batch size, number of interleaved
self- and cross-attention layers in the 3D-2D alignment
module, and the number of layers used by the 3D encoder.
4.2. Single-scene training
Results on the 7-Scenes Dataset.
As shown in Table 1,
our method outperforms other 3DGS-based approaches and
achieves higher rotation and translation accuracy than APR
and NeRF-based methods. It also performs on par with SCR
methods, though it is slightly outperformed by the structure-
based DVLAD+R2D2 [53, 69].
Results on the Cambridge Landmarks Dataset.
As
shown in Table 2, our approach surpasses GSplatLoc, a
concurrent 3DGS-based method, in rotation accuracy while
achieving comparable translation accuracy. It further out-
performs APR methods by a significant margin in both ro-
tation and translation accuracy. However, it trails behind the
state-of-the-art method HLoc, particularly in translation ac-
curacy. We believe this is due to the relatively low rendering
quality of 3DGS on the Cambridge Landmarks dataset (see
supplementary material). Improved 3DGS representations
optimized for such challenging scenes may help bridge this
gap.
Qualitative Results.
In Figure 3, we provide qualitative
results demonstrating the accuracy of the pose estimation
achieved with our method both before and after the refine-
ment stage. Each example includes the ground-truth (GT)
query image and rendered images from the initially esti-
mated pose and the final refined pose.
The initial pose estimates align reasonably well with the
GT, though some misalignments remain. After refinement,
the final pose rendering shows improved alignment, with
details that closely match the GT image. This refinement
step corrects subtle orientation and translation discrepan-
cies.
4.3. Multi-scene training
A multi-scene model can encode information from multiple
scenes in a single model while retaining accuracy similar to
single-scene training. In addition, a multi-scene model al-
lows for lower resource usage at inference. We tested the
ability of our model to encode several scenes simultane-
ously. To that end, we trained our model on training im-
ages from all the scenes in the 7-scenes dataset and tested
it on test query images from the same scenes. Since the
size of the training data for this experiment is large, we
7

<!-- page 8 -->
Scene/Method
Single-scene
Cross-scene
GSplatLoc [62]
GSVisLoc (Ours)
GSVisLoc (Ours)
ebc200e928
2.02/0.76
0.27 /0.15
0.33/0.22
2a496183e1
1.29/0.31
0.34/0.12
0.22/0.11
b26e64c4b0
0.79/0.20
0.27 /0.09
0.32/0.10
9b74afd2d2
6.18/1.29
0.26 /0.14
0.35/0.13
bc400d86e1
17.57/2.09
0.52 /0.16
0.62/0.36
1204e08f17
1.96/0.53
0.39 /0.10
0.65/0.13
f8f12e4e6b
3.69/0.83
0.39/0.13
0.34/0.13
52599ae063
21.95/5.54
2.31/0.29
0.82/0.18
94ee15e8ba
1.92/0.29
0.33/0.13
0.30/0.09
9f139a318d
1.19/0.22
0.24/0.05
0.22/0.07
30f4a2b44d
4.79/0.74
0.44 /0.18
0.46/0.16
37ea1c52f0
3.59/0.50
0.49 /0.10
0.60/0.10
480ddaadc0
1.51/0.37
0.28 /0.10
0.39/0.12
a24f64f7fb
1.08/0.38
0.54/0.14
0.46/0.13
ab11145646
8.64/0.88
0.38 /0.13
0.95/0.21
Avg.Med↓
5.21/0.99
0.50/0.13
0.47/0.15
Table 4. Generalization on ScanNet++ [80]. We report the me-
dian position errors (in cm) and rotation errors (in degrees) for
each scene, along with the overall averages across scenes. The
best result in each row is shown in bold, and the second-best re-
sult is underlined.
train GSVisLoc without the fine-level matching (coarse-
level only) and compare it to single-scene models with fine-
level matching. The results in Table 3 indicate that, despite
training across multiple scenes at coarse-level, we observe
only a slight reduction in prediction accuracy and even an
improvement in the Stairs scene.
4.4. Cross-scene generalization
We next tested our model for cross-scene generalization.
We trained the model on multiple scenes from the Scan-
Net++ dataset and tested it on novel, unseen scenes from the
test split. We compared the cross-scene model to GSPlat-
Loc and to our method trained in the single-scene train-
ing pipeline. We note that these two baselines, unlike our
cross-scene model, were trained specifically on the tested
scenes. Due to shortage of resources, we trained this model
without the fine-level matching, as in the multi-scene setup.
As shown in Table 4, our cross-scene model outperforms
GSPlatLoc and achieves comparable results to our single-
scene pipeline.
4.5. Ablations
We conducted ablation studies to demonstrate the impor-
tance of different components of our model. We tested our
model without (a) a 3D encoder, (b) the fine-tuning of a
2D encoder, (c) a 3D-2D alignment module, and (d) fine-
level matching. We report the results before pose refine-
ment in Table 5. For (a), when the 3D encoder was omitted,
we sampled  N_g 3D points and used a single linear layer to
align the feature dimensions between the Gaussians and the
Fire
Heads
Pumpkin
No 3D encoder
-
-
-
No 2D encoder fine-tuning
3.7/1.4
4.6/2.7
4.1/0.70
No 3D-2D alignment
2.7/1.0
3.8/2.3
2.3/0.39
No fine-level matching
6.8/2.5
4.9/2.8
3.4/0.56
GSVisLoc
2.6/1.1
3.4/2.1
2.0/0.39
Table 5. Ablation Studies. We evaluate our model’s performance
by removing key components: the 3D encoder, fine-tuning of the
2D encoder, the 3D-2D alignment module, and find-level match-
ing, compared to GSVisLoc. We report per-scene median trans-
lation errors (in cm) and rotation errors (in degrees) on 7-Scenes
without pose refinement. The best result in each column is high-
lighted in bold.
query image. In this setup, the model was unable to extract
a sufficient number of matches for reliable camera pose es-
timation, highlighting the necessity of encoding a 3D re-
gion rather than relying on individual Gaussians. Training a
model without either fine-level matching or fine-tuning the
2D encoder hurts the model’s accuracy, demonstrating their
importance. Surprisingly, removing the 3D-2D alignment
module (i.e., using dual-softmax directly on the encoder
outputs, without applying self and cross-attention layers)
had minimal impact on the results in a single-scene training
setup, indicating that the encoders effectively extract corre-
sponding 3D and 2D features. However, in the cross-scene
generalization setup, omitting the 3D-2D alignment module
led to near-random pose estimates.
5. Conclusion
In this work, we present GSVisLoc, a generalizable visual
localization method for a 3D Gaussian Splatting (3DGS)
scene representation, whose task is to estimate the camera
position and orientation in a 3D environment given a query
image. Our method, which uses deep learning to estab-
lish 3D-2D correspondences enables effective localization
with 3DGS as the sole scene representation, eliminating the
need for reference or training images during inference. Our
method achieves competitive localization accuracy on stan-
dard benchmarks that include images from both indoor and
outdoor scenes. The coarse-to-fine matching strategy, cou-
pled with a 3DGS-based pose refinement step, allows for
precise pose estimation. Most importantly, our method is
trainable, enabling cross-scene generalization with almost
no loss of accuracy. By that, our work opens avenues for
visual localization for 3DGS scene representation, without
reliance on additional reference data. Still, our results high-
light the importance of improving 3DGS scene representa-
tion on outdoor scenes.
8

<!-- page 9 -->
Acknowledgments
Research was partially supported by the Israeli Council for
Higher Education (CHE) via the Weizmann Data Science
Research Center, by the MBZUAI-WIS Joint Program for
Artificial Intelligence Research and by research grants from
the Estates of Tully and Michele Plesser and the Anita
James Rosen and Harry Schutzman Foundations
References
[1] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pa-
jdla, and Josef Sivic. Netvlad: Cnn architecture for weakly
supervised place recognition. In Proceedings of the IEEE
conference on computer vision and pattern recognition,
pages 5297–5307, 2016. 2
[2] Eduardo Arnold, Jamie Wynn, Sara Vicente, Guillermo
Garcia-Hernando, Aron Monszpart, Victor Prisacariu, Dani-
yar Turmukhambetov, and Eric Brachmann. Map-free visual
relocalization: Metric pose relative to a single image.
In
European Conference on Computer Vision, pages 690–708.
Springer, 2022. 2
[3] Vassileios Balntas, Shuda Li, and Victor Prisacariu. Reloc-
net: Continuous metric learning relocalisation using neural
nets. In Proceedings of the European Conference on Com-
puter Vision (ECCV), pages 751–767, 2018. 2
[4] Gabriele Berton, Carlo Masone, and Barbara Caputo. Re-
thinking visual geo-localization for large-scale applications.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 4878–4888, 2022. 2
[5] Hunter Blanton, Connor Greenwell, Scott Workman, and
Nathan Jacobs. Extending absolute pose regression to multi-
ple scenes. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition Workshops, pages
38–39, 2020. 1, 2
[6] Matteo Bortolon, Theodore Tsesmelis, Stuart James, Fabio
Poiesi, and Alessio Del Bue. 6dgs: 6d pose estimation from
a single image and a 3d gaussian splatting model.
arXiv
preprint arXiv:2407.15484, 2024. 1, 3, 5, 6
[7] Eric Brachmann and Carsten Rother. Learning less is more-
6d camera localization via 3d surface regression. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 4654–4662, 2018. 1, 2, 3
[8] Eric Brachmann and Carsten Rother.
Visual camera re-
localization from rgb and rgb-d images using dsac. IEEE
transactions on pattern analysis and machine intelligence,
44(9):5847–5865, 2021. 5, 6
[9] Eric Brachmann, Alexander Krull, Sebastian Nowozin,
Jamie Shotton, Frank Michel, Stefan Gumhold, and Carsten
Rother. Dsac-differentiable ransac for camera localization.
In Proceedings of the IEEE conference on computer vision
and pattern recognition, pages 6684–6692, 2017. 1, 2, 3
[10] Eric Brachmann, Martin Humenberger, Carsten Rother, and
Torsten Sattler. On the limits of pseudo ground truth in vi-
sual camera re-localisation. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 6218–
6228, 2021. 5, 1
[11] Eric Brachmann, Tommaso Cavallari, and Victor Adrian
Prisacariu.
Accelerated coordinate encoding: Learning to
relocalize in minutes using rgb and poses. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5044–5053, 2023. 1, 2, 3, 5, 6
[12] Samarth Brahmbhatt, Jinwei Gu, Kihwan Kim, James Hays,
and Jan Kautz. Geometry-aware learning of maps for cam-
era localization. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 2616–2625,
2018. 1, 2
[13] Federico Camposeco, Andrea Cohen, Marc Pollefeys, and
Torsten Sattler.
Hybrid scene compression for visual lo-
calization.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 7653–
7662, 2019. 1, 2
[14] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xi-
aopeng Zhang, Wei Shen, and Qi Tian.
Segment any 3d
gaussians. arXiv preprint arXiv:2312.00860, 2023. 1
[15] Jiahao Chang, Yinglin Xu, Yihao Li, Yuantao Chen, Wensen
Feng, and Xiaoguang Han. Gaussreg: Fast 3d registration
with gaussian splatting. In European Conference on Com-
puter Vision, pages 407–423. Springer, 2024. 1
[16] Hongkai Chen, Zixin Luo, Lei Zhou, Yurun Tian, Mingmin
Zhen, Tian Fang, David McKinnon, Yanghai Tsin, and Long
Quan.
Aspanformer: Detector-free image matching with
adaptive span transformer. In Computer Vision–ECCV 2022:
17th European Conference, Tel Aviv, Israel, October 23–
27, 2022, Proceedings, Part XXXII, pages 20–36. Springer,
2022. 2
[17] Shuai Chen, Xinghui Li, Zirui Wang, and Victor Adrian
Prisacariu. Dfnet: Enhance absolute pose regression with
direct feature matching. arXiv preprint arXiv:2204.00559,
2022. 5, 6
[18] Shuai Chen, Yash Bhalgat, Xinghui Li, Jiawang Bian, Kejie
Li, Zirui Wang, and V Prisacariu. Refinement for absolute
pose regression with neural feature synthesis. 2023. 5, 6
[19] Shuai Chen, Yash Bhalgat, Xinghui Li, Jia-Wang Bian, Kejie
Li, Zirui Wang, and Victor Adrian Prisacariu. Neural refine-
ment for absolute pose regression with feature synthesis. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 20987–20996, 2024. 1,
3
[20] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xi-
aofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping
Liu, and Guosheng Lin. Gaussianeditor: Swift and control-
lable 3d editing with gaussian splatting. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 21476–21485, 2024. 1
[21] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexan-
der Kirillov, and Rohit Girdhar.
Masked-attention mask
transformer for universal image segmentation. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 1290–1299, 2022. 1
[22] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabi-
novich. Superpoint: Self-supervised interest point detection
and description. In Proceedings of the IEEE conference on
computer vision and pattern recognition workshops, pages
224–236, 2018. 2
9

<!-- page 10 -->
[23] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Polle-
feys, Josef Sivic, Akihiko Torii, and Torsten Sattler.
D2-
net: A trainable cnn for joint description and detection of
local features. In Proceedings of the ieee/cvf conference on
computer vision and pattern recognition, pages 8092–8101,
2019. 2
[24] Sovann En, Alexis Lechervy, and Fr´ed´eric Jurie. Rpnet: An
end-to-end network for relative camera pose estimation. In
Proceedings of the European Conference on Computer Vi-
sion (ECCV) Workshops, pages 0–0, 2018. 2
[25] Martin A Fischler and Robert C Bolles.
Random sample
consensus: a paradigm for model fitting with applications to
image analysis and automated cartography. Communications
of the ACM, 24(6):381–395, 1981. 5
[26] Xiao-Shan Gao, Xiao-Rong Hou, Jianliang Tang, and
Hang-Fei Cheng.
Complete solution classification for the
perspective-three-point problem.
IEEE transactions on
pattern analysis and machine intelligence, 25(8):930–943,
2003. 2, 3
[27] Hugo Germain, Daniel DeTone, Geoffrey Pascoe, Tan-
ner Schmidt, David Novotny, Richard Newcombe, Chris
Sweeney, Richard Szeliski, and Vasileios Balntas. Feature
query networks: Neural surface description for camera pose
refinement.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 5071–
5081, 2022. 3
[28] Ben Glocker, Shahram Izadi, Jamie Shotton, and Antonio
Criminisi. Real-time rgb-d camera relocalization. In 2013
IEEE International Symposium on Mixed and Augmented
Reality (ISMAR), pages 173–179. IEEE, 2013. 1, 5, 6, 7
[29] Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford,
and Tobias Fischer.
Patch-netvlad: Multi-scale fusion of
locally-global descriptors for place recognition. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 14141–14152, 2021. 2
[30] Lionel Heng, Benjamin Choi, Zhaopeng Cui, Marcel Gep-
pert, Sixing Hu, Benson Kuan, Peidong Liu, Rang Nguyen,
Ye Chuan Yeo, Andreas Geiger, et al. Project autovision: Lo-
calization and 3d scene perception for an autonomous vehi-
cle with a multi-camera system. In 2019 International Con-
ference on Robotics and Automation (ICRA), pages 4695–
4702. IEEE, 2019. 1
[31] Arnold Irschara, Christopher Zach, Jan-Michael Frahm, and
Horst Bischof. From structure-from-motion point clouds to
fast location recognition. In 2009 IEEE Conference on Com-
puter Vision and Pattern Recognition, pages 2599–2606.
IEEE, 2009. 1, 2
[32] Tong Ke and Stergios I Roumeliotis. An efficient algebraic
solution to the perspective-three-point problem. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 7225–7233, 2017. 2, 3
[33] Alex Kendall and Roberto Cipolla. Geometric loss functions
for camera pose regression with deep learning. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 5974–5983, 2017. 1, 2
[34] Alex Kendall, Matthew Grimes, and Roberto Cipolla.
Posenet: A convolutional network for real-time 6-dof cam-
era relocalization. In Proceedings of the IEEE international
conference on computer vision, pages 2938–2946, 2015. 1,
2, 5, 6
[35] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 1, 3, 6
[36] Fadi Khatib, Yuval Margalit, Meirav Galun, and Ronen
Basri. Leveraging image matching toward end-to-end rel-
ative camera pose regression. In DAGM German Conference
on Pattern Recognition, pages 185–201. Springer, 2024. 2
[37] Diederik P Kingma. Adam: A method for stochastic opti-
mization. arXiv preprint arXiv:1412.6980, 2014. 6
[38] Laurent Kneip, Davide Scaramuzza, and Roland Siegwart.
A novel parametrization of the perspective-three-point prob-
lem for a direct computation of absolute camera position and
orientation. In CVPR 2011, pages 2969–2976. IEEE, 2011.
2, 3
[39] Zakaria Laskar, Iaroslav Melekhov, Assia Benbihi, Shuzhe
Wang, and Juho Kannala. Differentiable product quantiza-
tion for memory efficient camera relocalization. In European
Conference on Computer Vision, pages 470–489. Springer,
2024. 2
[40] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Ground-
ing image matching in 3d with mast3r.
arXiv preprint
arXiv:2406.09756, 2024. 3, 5
[41] Minhao Li, Zheng Qin, Zhirui Gao, Renjiao Yi, Chenyang
Zhu, Yulan Guo, and Kai Xu. 2d3d-matr: 2d-3d matching
transformer for detection-free registration between images
and point clouds. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 14128–14138,
2023. 2
[42] Xiaotian Li, Shuzhe Wang, Yi Zhao, Jakob Verbeek, and
Juho Kannala. Hierarchical scene coordinate classification
and regression for visual localization.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 11983–11992, 2020. 1, 2, 3, 5, 6
[43] Yunpeng Li, Noah Snavely, and Daniel P Huttenlocher. Lo-
cation recognition using prioritized feature matching.
In
Computer Vision–ECCV 2010: 11th European Conference
on Computer Vision, Heraklion, Crete, Greece, September 5-
11, 2010, Proceedings, Part II 11, pages 791–804. Springer,
2010. 1, 2
[44] Yunzhi Lin, Thomas M¨uller, Jonathan Tremblay, Bowen
Wen, Stephen Tyree, Alex Evans, Patricio A Vela, and Stan
Birchfield. Parallel inversion of neural radiance fields for
robust pose estimation. In 2023 IEEE International Confer-
ence on Robotics and Automation (ICRA), pages 9377–9384.
IEEE, 2023. 3
[45] Changkun Liu, Shuai Chen, Yukun Zhao, Huajian Huang,
Victor Prisacariu, and Tristan Braud.
Hr-apr:
Apr-
agnostic framework with uncertainty estimation and hierar-
chical refinement for camera relocalisation. arXiv preprint
arXiv:2402.14371, 2024. 3
[46] Changkun Liu, Shuai Chen, Yash Sanjay Bhalgat, Siyan HU,
Ming Cheng, Zirui Wang, Victor Adrian Prisacariu, and Tris-
tan Braud. GS-CPR: Efficient camera pose refinement via 3d
gaussian splatting. In The Thirteenth International Confer-
ence on Learning Representations, 2025. 1, 3, 5
10

<!-- page 11 -->
[47] Jianlin Liu, Qiang Nie, Yong Liu, and Chengjie Wang. Nerf-
loc: Visual localization with conditional neural radiance
field. In 2023 IEEE International Conference on Robotics
and Automation (ICRA), pages 9385–9392. IEEE, 2023. 1,
3, 5, 6
[48] Iaroslav Melekhov, Juha Ylioinas, Juho Kannala, and Esa
Rahtu.
Relative camera pose estimation using convolu-
tional neural networks. In International Conference on Ad-
vanced Concepts for Intelligent Vision Systems, pages 675–
687. Springer, 2017. 2
[49] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In European conference on computer vision, pages
405–421. Springer, 2020. 1
[50] Arthur Moreau, Nathan Piasco, Dzmitry Tsishkou, Bogdan
Stanciulescu, and Arnaud de La Fortelle. Lens: Localization
enhanced by nerf synthesis. In Conference on Robot Learn-
ing, pages 1347–1356. PMLR, 2022. 5, 6
[51] Arthur Moreau, Nathan Piasco, Moussab Bennehar, Dzmitry
Tsishkou, Bogdan Stanciulescu, and Arnaud de La Fortelle.
Crossfire: Camera relocalization on self-supervised features
from an implicit representation.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 252–262, 2023. 1, 3, 5, 6
[52] Maxime Pietrantoni, Gabriela Csurka, Martin Humenberger,
and Torsten Sattler. Self-supervised learning of neural im-
plicit feature fields for camera pose refinement. In 2024 In-
ternational Conference on 3D Vision (3DV), pages 484–494.
IEEE, 2024. 3
[53] Jerome Revaud, Cesar De Souza, Martin Humenberger, and
Philippe Weinzaepfel. R2d2: Reliable and repeatable detec-
tor and descriptor. Advances in neural information process-
ing systems, 32, 2019. 5, 6, 7
[54] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, San-
jeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
Aditya Khosla, Michael Bernstein, et al.
Imagenet large
scale visual recognition challenge. International journal of
computer vision, 115:211–252, 2015. 5
[55] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and
Marcin Dymczyk. From coarse to fine: Robust hierarchical
localization at large scale. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 12716–12725, 2019. 1, 2, 5, 6
[56] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,
and Andrew Rabinovich.
Superglue:
Learning feature
matching with graph neural networks.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 4938–4947, 2020. 2
[57] Paul-Edouard Sarlin, Ajaykumar Unagar, Mans Larsson,
Hugo Germain, Carl Toft, Viktor Larsson, Marc Pollefeys,
Vincent Lepetit, Lars Hammarstrand, Fredrik Kahl, et al.
Back to the feature: Learning robust camera localization
from pixels to pose. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
3247–3257, 2021. 5, 6
[58] Torsten Sattler, Bastian Leibe, and Leif Kobbelt. Efficient
& effective prioritized matching for large-scale image-based
localization. IEEE transactions on pattern analysis and ma-
chine intelligence, 39(9):1744–1756, 2016. 1, 2
[59] Torsten Sattler, Qunjie Zhou, Marc Pollefeys, and Laura
Leal-Taixe.
Understanding the limitations of cnn-based
absolute camera pose regression.
In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pages 3302–3312, 2019. 2
[60] Yoli Shavit, Ron Ferens, and Yosi Keller. Learning multi-
scene absolute pose regression with transformers. In Pro-
ceedings of the IEEE/CVF International Conference on
Computer Vision, pages 2733–2742, 2021. 1, 2, 5, 6
[61] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram
Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene co-
ordinate regression forests for camera relocalization in rgb-d
images. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 2930–2937, 2013. 1,
2, 5, 6, 7
[62] Gennady Sidorov, Malik Mohrat, Ksenia Lebedeva, Ruslan
Rakhimov, and Sergey Kolyubin. Gsplatloc: Grounding key-
point descriptors into 3d gaussian splatting for improved vi-
sual localization. arXiv preprint arXiv:2409.16502, 2024. 1,
3, 5, 6, 8
[63] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and
Xiaowei Zhou. Loftr: Detector-free local feature matching
with transformers.
In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
8922–8931, 2021. 2, 5
[64] Yuan Sun, Xuan Wang, Yunfan Zhang, Jie Zhang, Caigui
Jiang, Yu Guo, and Fei Wang. icomma: Inverting 3d gaus-
sians splatting for camera pose estimation via comparing and
matching. arXiv preprint arXiv:2312.09031, 2023. 1, 3
[65] Hajime Taira, Masatoshi Okutomi, Torsten Sattler, Mircea
Cimpoi, Marc Pollefeys, Josef Sivic, Tomas Pajdla, and Ak-
ihiko Torii.
Inloc: Indoor visual localization with dense
matching and view synthesis. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition,
pages 7199–7209, 2018. 1, 2, 5, 6
[66] Shitao Tang, Chengzhou Tang, Rui Huang, Siyu Zhu, and
Ping Tan.
Learning camera localization via dense scene
matching.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 1831–
1841, 2021. 5, 6
[67] Shitao Tang, Sicong Tang, Andrea Tagliasacchi, Ping Tan,
and Yasutaka Furukawa. Neumap: Neural coordinate map-
ping by auto-transdecoder for camera localization. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 929–939, 2023. 3, 5, 6
[68] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud,
Beatriz Marcotegui, Franc¸ois Goulette, and Leonidas J
Guibas. Kpconv: Flexible and deformable convolution for
point clouds. In Proceedings of the IEEE/CVF international
conference on computer vision, pages 6411–6420, 2019. 1,
3, 6
[69] Akihiko Torii, Relja Arandjelovic, Josef Sivic, Masatoshi
Okutomi, and Tomas Pajdla.
24/7 place recognition by
view synthesis. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 1808–1817,
2015. 2, 5, 6, 7
11

<!-- page 12 -->
[70] Gabriele Trivigno, Carlo Masone, Barbara Caputo, and
Torsten Sattler.
The unreasonable effectiveness of pre-
trained features for camera pose refinement. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 12786–12798, 2024. 1, 3, 5
[71] Jonathan Ventura, Clemens Arth, Gerhard Reitmayr, and Di-
eter Schmalstieg. Global localization from monocular slam
on a mobile phone. IEEE transactions on visualization and
computer graphics, 20(4):531–539, 2014. 1
[72] Florian Walch, Caner Hazirbas, Laura Leal-Taixe, Torsten
Sattler, Sebastian Hilsenbeck, and Daniel Cremers. Image-
based localization using lstms for structured feature correla-
tion. In Proceedings of the IEEE international conference on
computer vision, pages 627–637, 2017. 1, 2
[73] Fangjinhua
Wang,
Xudong
Jiang,
Silvano
Galliani,
Christoph Vogel, and Marc Pollefeys. Glace: Global local
accelerated coordinate encoding.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 21562–21571, 2024. 1, 2
[74] Qiang Wang. Mad-dr: Map compression for visual localiza-
tion with matchness aware descriptor dimension reduction.
In European Conference on Computer Vision, pages 261–
278. Springer, 2024. 2
[75] Qianqian Wang, Xiaowei Zhou, Bharath Hariharan, and
Noah Snavely.
Learning feature descriptors using camera
pose supervision.
In Computer Vision–ECCV 2020: 16th
European Conference, Glasgow, UK, August 23–28, 2020,
Proceedings, Part I 16, pages 757–774. Springer, 2020. 2, 5
[76] Shuzhe Wang, Juho Kannala, and Daniel Barath. Dgc-gnn:
Leveraging geometry and color cues for visual descriptor-
free 2d-3d matching. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
20881–20891, 2024. 2
[77] Andreas Wendel, Arnold Irschara, and Horst Bischof. Natu-
ral landmark-based monocular localization for mavs. In 2011
IEEE International Conference on Robotics and Automation,
pages 5792–5799. IEEE, 2011. 1
[78] Luwei Yang, Ziqian Bai, Chengzhou Tang, Honghua Li,
Yasutaka Furukawa, and Ping Tan.
Sanet: Scene agnos-
tic network for camera localization.
In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 42–51, 2019. 1, 2, 3, 5, 6
[79] Lin Yen-Chen, Pete Florence, Jonathan T Barron, Alberto
Rodriguez, Phillip Isola, and Tsung-Yi Lin. inerf: Inverting
neural radiance fields for pose estimation. In 2021 IEEE/RSJ
International Conference on Intelligent Robots and Systems
(IROS), pages 1323–1330. IEEE, 2021. 1, 3
[80] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner,
and Angela Dai. Scannet++: A high-fidelity dataset of 3d
indoor scenes. In Proceedings of the International Confer-
ence on Computer Vision (ICCV), 2023. 5, 8, 2
[81] Weihao Yu, Chenyang Si, Pan Zhou, Mi Luo, Yichen Zhou,
Jiashi Feng, Shuicheng Yan, and Xinchao Wang. Metaformer
baselines for vision. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 2023. 5
[82] Qunjie Zhou,
Torsten Sattler,
and Laura Leal-Taixe.
Patch2pix: Epipolar-guided pixel-level correspondences. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 4669–4678, 2021. 2
[83] Qunjie Zhou, S´ergio Agostinho, Aljoˇsa Oˇsep, and Laura
Leal-Taix´e. Is geometry enough for matching in visual lo-
calization?
In European Conference on Computer Vision,
pages 407–425. Springer, 2022. 2
[84] Qunjie Zhou, Maxim Maximov, Or Litany, and Laura Leal-
Taix´e. The nerfect match: Exploring nerf features for visual
localization. arXiv preprint arXiv:2403.09577, 2024. 1, 3,
5, 6
[85] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi. Feature 3dgs: Supercharging
3d gaussian splatting to enable distilled feature fields. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21676–21685, 2024. 3
12

<!-- page 13 -->
GSVisLoc: Generalizable Visual Localization for Gaussian Splatting Scene
Representations
Supplementary Material
Table 6. Gaussian splatting PSNR scores. We provide the PSNR
scores for our trained GS models across each scene in the Cam-
bridge Landmarks [34] and 7-Scenes [28, 61] datasets.
Cambridge Landmarks - Outdoor
Kings
Hospital
Shop
StMary
Average
20.8
16.9
22.0
21.7
20.4
7-Scenes - Indoor
Chess
Fire
Heads
Office
Pump.
Kitchen
Stairs
Average
28.9
28.8
30.7
28.9
30.0
24.4
30.4
28.9
Below, we provide additional details about our imple-
mentation and examples of challenging scenes from the
ScanNet++ dataset in Figure 4. As shown, the scenes are
large and diverse, including extensive texture-less areas,
which demonstrates the generalization ability of our cross-
scene model.
Our code is still a work in progress. We will publish it
after finishing to clean and refactor it for easy use. All of
the data we used is publicly available. We will also release
our pre-trained models.
6. Gaussain Splatting Implementation Details
We use the pre-built COLMAP reconstructions from [10]
for the 7scenes dataset and the reconstructions provided in
HLoc toolbox [55] for the Cambridge landmarks dataset.
We train all the scenes using the vanilla 3DGS [35], for 30k
iterations using the default parameters, in Table 6 we report
the per-scene PSNR scores for our trained models on the
training images. Notably, the rendering quality of outdoor
scenes is inferior compared to indoor scenes, which might
explain the degradation in our pose estimation accuracy.
Handling challenges in outdoor scenes. To effectively
train a 3DGS model for outdoor scene reconstruction, we
focus on reconstructing static elements such as buildings,
fences, and signs. This approach addresses real-world chal-
lenges like varying lighting conditions, dynamic objects,
and distant regions. To mitigate these issues, we use a pre-
trained semantic segmentation model [21] to mask out sky
regions and moving objects, including pedestrians and vehi-
cles. These elements, which constitute only a small portion
of the captured images, are excluded from the loss function
during training, resulting in more accurate scene reconstruc-
tion. For this purpose, we utilize pre-computed segmen-
tation maps provided by [84], generated using the method
described in [21].
1

<!-- page 14 -->
52599ae063
b26e64c4b0
1204e08f17
480ddaadc0
ab11145646
Train images
Test image
Rendered from
estimated pose
Figure 4. ScanNet++. Examples of qualitative results obtained by our cross-scene model on diverse scenes of ScanNet++ [80]. From left
to right, two images from the training images, a test query image, and the rendered image from our model pose prediction.
2
