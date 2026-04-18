<!-- page 1 -->
VolSplat: Rethinking Feed-Forward 3D Gaussian
Splatting with Voxel-Aligned Prediction
Weijie Wang1,2∗
Yeqing Chen1∗
Zeyu Zhang2
Hengyu Liu2,3
Haoxiao Wang1
Zhiyuan Feng4
Wenkang Qin2
Feng Chen5
Zheng Zhu2†
Donny Y. Chen6
Bohan Zhuang1†
Zhejiang University1
GigaAI2
The Chinese University of Hong Kong3
Tsinghua University4
Adelaide University5
Monash University6
* Equal contribution
† Corresponding authors
Project Page: lhmd.top/volsplat
Abstract. Feed-forward 3D Gaussian Splatting (3DGS) has emerged
as a highly effective solution for novel view synthesis. Existing meth-
ods predominantly rely on a pixel-aligned Gaussian prediction paradigm,
where each 2D pixel is mapped to a 3D Gaussian. We rethink this widely
adopted formulation and identify several inherent limitations: it renders
the reconstructed 3D models heavily dependent on the number of input
views, leads to view-biased density distributions, and introduces align-
ment errors, particularly when source views contain occlusions or low
texture. To address these challenges, we introduce VolSplat, a new multi-
view feed-forward paradigm that replaces pixel alignment with voxel-
aligned Gaussians. By directly predicting Gaussians from a predicted
3D voxel grid, it overcomes pixel alignment’s reliance on error-prone 2D
feature matching, ensuring robust multi-view consistency. Furthermore,
it enables adaptive control over density based on 3D scene complex-
ity, yielding more faithful Gaussians, improved geometric consistency,
and enhanced novel-view rendering quality. Experiments on widely used
benchmarks demonstrate that VolSplat achieves state-of-the-art perfor-
mance, while producing more plausible and view-consistent results.
Keywords: 3D Gaussians · Feed-Forward Reconstruction · View Syn-
thesis
1
Introduction
3D reconstruction is a cornerstone of modern robotics, empowering autonomous
systems with the critical ability to perceive, map, and comprehend their physical
environment, which is fundamental for advanced navigation, object manipula-
tion, and intelligent interaction. Traditional optimization based approaches, in-
cluding Neural Radiance Fields (NeRF) [34] and 3D Gaussian Splatting (3DGS) [22],
obtain high fidelity results by iteratively enforcing photometric or geometric
consistency. These methods achieve excellent accuracy but are computation-
ally intensive and slow to run at inference time. By contrast, feed-forward ap-
proaches [3–6, 19, 40, 42, 44, 47, 51, 53] trade per instance optimization for fast
arXiv:2509.19297v2  [cs.CV]  12 Mar 2026

<!-- page 2 -->
2
W. Wang et al.
Novel
View
x
(a) Pixel-aligned Feed-forward 3DGS
(b) Voxel-aligned Feed-forward 3DGS (Ours)
Cross-view
Matching
Per-pixel
Prediction
Gaussians
Gaussian
Merge
Rendering
Gaussians
Feature Unprojection
and Refinement
Voxel-aligned
Prediction
Rendering
Novel
View
Fig. 1: Comparison between the pixel-aligned feed-forward method and our
approach. Pixel-aligned feed-forward 3DGS methods suffer from two primary limita-
tions: 1) 2D feature matching struggles to effectively resolve the multi-view alignment
problem, and 2) the Gaussian density is constrained and cannot be adaptively con-
trolled according to scene complexity. We propose VolSplat, a framework that directly
regresses Gaussians from 3D features based on a voxel-aligned prediction strategy. This
approach achieves adaptive control over scene complexity and resolves the multi-view
alignment challenge.
learned inference. A single forward pass predicts scene geometry or a 3D rep-
resentation directly from input images. This speed and simplicity make feed-
forward systems attractive for real time applications, large scale datasets, and
downstream tasks that require many reconstructions.
Prior feed-forward 3DGS methods [3, 5, 19, 44, 47] commonly rely on pixel
alignment as their fundamental mechanism for associating image features with
pixel aligned Gaussians. In this design, per-pixel features from precomputed
image feature maps are unprojected to define the corresponding Gaussians. The
prevailing consensus has been to perform fusion directly within the 2D feature
representation. However, pixel aligned designs inherit two primary limitations.
1) Sampling at discrete pixel locations is sensitive to camera calibration and
discretization error, produces inconsistent sampling patterns across views. 2)
The rigid pixel-to-Gaussian association enforces a uniform density distribution
that ignores scene complexity, leading to redundant primitives in simple regions
while failing to capture fine-grained 3D structures.
In this work, we shift the alignment paradigm from pixels to voxels, as il-
lustrated in Fig. 1. Instead of sampling features at projected pixel coordinates,
we align and aggregate image features directly into a 3D voxel grid. Multi-view
image features are aggregated directly into this 3D voxel space, effectively de-
coupling feature fusion from the camera view frustums. Within this unified voxel
space, we employ a 3D U-Net [8] to reason about scene geometry and appearance
volumetrically. Finally, rather than predicting Gaussians per pixel, we predict
primitives directly from the refined voxel features, allowing the distribution of
3D Gaussians to be determined by the volumetric structure itself.
There are practical and conceptual advantages to voxel alignment. Specifi-
cally, volumetric aggregation reduces floaters and view dependent inconsistency
because information from multiple views is fused into a shared 3D container
before Gaussian prediction. Simultaneously, operating in a 3D grid enables the

<!-- page 3 -->
VolSplat
3
use of well studied 3D decoder and regularization strategies, which naturally en-
code locality and geometrical context. Instead of the integration of auxiliary 3D
signals such as depth maps [47] and point clouds [39], our approach naturally re-
solves spatial ambiguities within the unified voxel space, thereby eliminating the
need for such ad hoc priors or auxiliary supervision signals. Furthermore, voxel
representations are amenable to modern acceleration strategies such as sparse
data structures, making the approach practical at the resolutions required for
high quality reconstruction.
In this paper we present a feed-forward three dimensional reconstruction
framework built around voxel alignment. As shown in Fig. 2, we first construct
3D feature grids using the extracted 2D image features, then refine the 3D fea-
tures and use them to predict voxel-aligned Gaussians. We analyze the alignment
errors that arise in pixel aligned pipelines and show how voxel alignment reduces
these errors both conceptually and empirically. Through systematic experiments
on synthetic and real world benchmarks, we demonstrate that voxel aligned feed-
forward models achieve more accurate and robust reconstructions than compara-
ble pixel aligned baselines on large-scale benchmarks such as RealEstate10K [57],
ScanNet [9] and ACID [26]. Our contributions are as follows:
– We introduce voxel alignment as a principled alternative to pixel alignment
for feed-forward 3DGS and present a practical end-to-end framework.
– We provide an analysis of alignment induced errors in pixel aligned systems
and show how volumetric aggregation mitigates these failure modes.
– Experimental results demonstrate that VolSplat achieves state-of-the-art
(SOTA) performance on several large-scale benchmarks.
2
Related Work
Novel view synthesis. Traditional approaches to Novel View Synthesis (NVS)
primarily rely on geometry-based rendering methods that reconstruct explicit
3D scene geometry from images [10], image-based rendering techniques that in-
terpolate between captured views without full 3D reconstruction [17], and light
field rendering that samples and reprojects densely captured rays in space [25].
These methods required either accurate geometric proxies, densely sampled view-
points, or both to produce convincing visual results, limiting their applicability
in real-world scenarios. The emergence of NeRF [34] marked a paradigm shift,
significantly improving both rendering quality and robustness over prior meth-
ods, which learns a continuous, implicit scene representation by utilizing a MLP
to map position and viewing direction to a corresponding color and volume den-
sity. While NeRF-based methods [1, 2] require a long training time due to the
per-ray rendering. 3DGS [22] and its variants [11, 27, 59] have been introduced
to represent the 3D scene using a set of anisotropic 3D Gaussians.
3D voxelization. Voxelization, which discretizes 3D space into regular voxel
grids, has been a foundational representation in 3D reconstruction and model-
ing [32]. Prior methods used dense grids for their simplicity, but suffered from

<!-- page 4 -->
4
W. Wang et al.
high memory costs and poor scalability [21]. To address this, sparse structures
like octrees were introduced for more efficient storage and computation [23]. In
modern applications, voxels are widely used as input to 3D Convolutional Neural
Network (CNN) for tasks such as object detection [58] and semantic segmenta-
tion [38]. More recently, voxels are often used as sparse scaffolding rather than as
the final representation, supporting more advanced rendering techniques. Rep-
resentative methods include Plenoxels [13] and K-Planes [12], which optimize
voxel-based radiance fields for fast, high-quality rendering, as well as structured
strategies such as Scaffold-GS [31] and Octree-GS [37], which leverage voxel grids
to organize and accelerate 3DGS.
Feed-forward 3D Gaussian Splatting. Recent developments in feed-forward
3DGS [3,5, 16, 18,19,44,47,55] offer a compelling alternative that directly pre-
dicts 3D Gaussians from input images in a single forward pass: pixelSplat [3]
proposes a two-view feed-forward pipeline that combines epipolar transform-
ers and depth prediction to generate Gaussians. MVSplat [5] introduces a cost-
volume-based fusion strategy to enhance multi-view consistency. DepthSplat [47]
leverages monocular depth features to improve fine 3D structure reconstruction
from sparse views. Follow-up work extends feed-forward 3DGS to more com-
plex scenarios, including pose-free inputs [20,50], online stream inputs [44], and
more dense inputs [42]. While these works adopt a pixel-aligned strategy to
predict Gaussian primitives, the pixel-wise formulation struggles to handle mul-
tiple input views due to redundancy and inconsistency across pixels. Existing
methods attempt to improve the per-pixel strategy by pruning the number of
Gaussians [55], token merging [42, 60] and voxel-based fusion [18, 28, 43]. How-
ever, these approaches do not fundamentally address the limitations inherent in
per-pixel processing. EVolSplat [33] has explored voxel features in autonomous
driving scenarios, but it has not been generalized to general scenarios and re-
quires explicit 3D point clouds as intermediate representations. In contrast, our
method introduces a voxel-aligned method, which eliminate the need for per-
query 2D prediction patterns. This alignment enables more stable multi-view
fusion, cleaner occlusion handling, and more coherent joint inference of geome-
try and appearance.
3
Method
3.1
Preliminary and Observation
Feed-forward 3D reconstruction aims to learn a mapping from N input images
I = {Ii}N
i=1 where Ii ∈RH×W ×3 and their corresponding camera poses P =
{Pi}N
i=1, to a 3D scene representation. In the context of pixel-aligned 3DGS,
features are extracted from images and refined by cross-view interaction:
  \labe
l { e q:feature_e xtra
ct i o
n
}  \
m athcal {F}=\{\mathbf {F}_i\}_{i=1}^N=h(\Phi _\mathrm {image}(\mathcal {I}, \mathcal {P})), \quad \mathbf {F}_i \in \mathbb {R}^{\frac {H}{p}\times \frac {W}{p}\times C}
(1)
where Φimage is a pretrained image encoder. The function h is responsible for
processing these features from different viewpoints, with its core purpose being

<!-- page 5 -->
VolSplat
5
Mutli-view Transformer for 2D Feature Extraction
Depth
Prediction
Module
Sparse 3D
Decoder
x
Image Features
Cost Volumes
Voxel Features
Refined Voxel Features
Depth Maps
Voxel-aligned 
Prediction
Unproject
Gaussians
Input Images
Fig. 2: Overview of VolSplat. Given multi-view images as input, we first extract
2D features for each image using a Transformer-based network and construct per-view
cost volumes with plane sweeping. Depth Prediction Module then estimates a depth
map for each view, which is used to unproject the 2D features into 3D space to form a
voxel feature grid. Subsequently, we employ a sparse 3D decoder (details in Sec. 3.3) to
refine these features in 3D space and predict the parameters of a 3D Gaussian for each
occupied voxel. Finally, novel views are rendered from the predicted 3D Gaussians.
to perform cross-view feature matching and fusion. For pixel-aligned Gaussian
prediction, the features must be upsampled to the same resolution as the input
image:
  \ma t hcal  {F}_\m athrm {full} = U(\mathcal {F}) \in \mathbb {R}^{N \times H \times W \times C},
(2)
where Ffull denotes the full-resolution feature maps, U is a feature upsampler
such as CNN-based network (in MVSplat [5]) and deconvolution-based net-
work [35] (in DepthSplat [47]). Per-pixel Gaussian predictions are then performed
using the upsampled features:
  \math cal  {G } = \le ft
 \{
( \mathbf {\mu  }_{i}, \mathbf {\Sigma }_{i}, \mathbf {\alpha }_{i}, \mathbf {c}_{i}\right )\}_{i=1}^{H \times W \times N} = \Psi _\mathrm {pred}(\mathcal {F}_\mathrm {full}, \mathcal {P}),
(3)
where the position of the Gaussians are determined by the predicted depth and
pixel location.
While straightforward, this pixel-aligned formulation introduces two critical
limitations. First, the geometric accuracy of the reconstruction is critically de-
pendent on the quality of the predicted depth map. After depth unprojecting
features into 3D space, the lack of interaction with neighboring points within
the 3D space significantly contributes to the generation of floaters. Second, the
structure of the 3D representation is rigidly tied to the 2D image grid. The total
number of Gaussians is fixed at |S| = H × W × N, which is often suboptimal
and cause an over-densification of Gaussians on simple, texture-less surfaces and
an insufficient number for representing complex geometry not captured at the
pixel level. These observations reveal a fundamental bottleneck and motivate our

<!-- page 6 -->
6
W. Wang et al.
proposed voxel-aligned framework, designed to decouple the 3D representation
from the 2D pixel grid.
3.2
3D Feature Construction
Feature extraction and matching. For N input images, we first apply a
weight-sharing ResNet [15] backbone to each RGB image to obtain p× down-
sampled feature maps. These features are then refined with cross-view attention
that exchanges information with the two nearest neighboring views. For effi-
ciency, this cross-attention is implemented with the local window attention [29].
After this stage we obtain cross-view–aware Transformer features {Fi}N
i=1 (F i ∈
R
H
p × W
p ×C) , where C denotes the feature dimension.
Next, we build per-view cost volumes {Ci}N
i=1 using a plane-sweep strat-
egy [48]. For each view i, we sample D candidate depths {dm}D
m=1, warp the
feature from neighboring views to the reference view at each hypothesized depth,
and compute pairwise feature similarities [5].These similarities are aggregated by
dot-product matching and stacked along the depth axis to form {Ci}N
i=1, where
Ci ∈R
H
p × W
p ×D.
To produce robust, multi-view consistent depth estimates, a depth module
fuses the monocular features

F i
mono
	N
i=1 (F i
mono ∈R
H
p × W
p ×C) with the cost
volume Ci and regresses a dense per-pixel depth map Di ∈RH×W , which serves
as a geometric prior for lifting image features into 3D space. These per-view
features F i and depths Di are used in the next stage to construct 3D point
clouds and voxel-based features for volumetric reasoning.
Lifting to 3D feature. Given the predicted depth maps Di and camera pa-
rameters, we conveniently aggregate different depth map views by transforming
the point clouds into a global coordinate system. First each pixel (u, v) in image
space is unprojected to a 3D point in the camera coordinate frame using the
camera intrinsics. Then the 3D point is transformed into the world coordinate
system via the corresponding extrinsic parameters, including the rotation matrix
Ri and translation Ti vector.
 wP_{\ m at hrm { wo r ld
}
} = R_ i\ ,P_
{
\
m
a
t
h
r
m
 { cam}} + T_i = R_i\!\left (D_i(u,v)\,K^{-1}\begin {bmatrix}u\\ v\\ 1\end {bmatrix}\right ) + T_i. \label {eq:unproject} 
(4)
By repeating this process across all views, we obtain a dense |S| = H × W × N
point cloud in world space, where each 3D point is associated with its corre-
sponding image feature.
To convert the unstructured dense point cloud P into a structured volumetric
representation, we voxelize the points [41]. For each 3D point p = (xp, yp, zp) we
compute integer voxel index (i, j, k) by dividing by the voxel size vs and rounding.
  i =
 \o
pe
r
a t o rna
me 
{r
n
d } \ lef
t (
 \
f
rac {x_p}{v_s} \right ),\ j = \operatorname {rnd}\left ( \frac {y_p}{v_s} \right ),\ k = \operatorname {rnd}\left ( \frac {z_p}{v_s} \right ), 
(5)

<!-- page 7 -->
VolSplat
7
x
Sparse Voxel Feature
3D U-Net Refinement
Residual Voxel Feature
x
Refined Voxel Feature
Add
Fig. 3: Architecture of Sparse 3D decoder. Sparse 3D features are fed into a 3D
U-Net for processing, which predicts residual features for each voxel. These residual
features are then added to the original 3D voxel features to obtain the refined features.
where rnd(·) denotes rounding to the nearest integer.
Let Si,j,k be the set of all points falling into voxel (i, j, k) and fpbe the image
feature corresponding to each point p ∈Si,j,k The features within this voxel
are aggregated via average pooling along the channel dimension, resulting in the
voxel feature Vi,j,k:
  V_{i ,
j
,k} &= \
f
rac {1}{
\lvert S_{i,j,k}\rvert }\sum _{p\in S_{i,j,k}} f_p. \label {eq:voxel_feature}
(6)
3.3
Feature Refinement and 3D Gaussians Prediction
Feature refinement. To improve the spatial consistency and structural fidelity
of the voxel representation, we apply an explicit voxel feature refinement stage
as shown in Fig. 3. Given an input voxel grid V (with per-voxel feature vectors),
a sparse convolutional 3D U-Net [8] R predicts a residual voxel field R:
  R & = 
\m a thcal {R}(V), \quad {R}_{i}\in \mathbb {R}^{\mathcal {V}\times C}, \label {eq:refine}
(7)
where V denotes the set of occupied voxels and the refined voxel features are
obtained by a residual update:
  V '  &=
 V
 +  R, \quad {V'}_{i}\in \mathbb {R}^{\mathcal {V}\times C}, \label {eq:voxel_update}
(8)
The refinement network is implemented with hierarchical sparse 3D convolu-
tional blocks, symmetric encoder–decoder stages, and upsampling layers con-
nected by skip connections. This architecture enables multi-scale fusion of local
and global geometric context while keeping computation efficient through spar-
sity. The residual formulation encourages the network to learn correction terms
(fine geometric detail and consistency cues) rather than relearning the entire
feature content, which empirically stabilizes training and preserves the coarse
voxel information supplied by the lifting stage.
3D Gaussians prediction. The output of our network for each voxel v is
a set of learnable Gaussian parameters {[¯µj, ¯αj, Σj, cj] ∈R38}. These include
the offset of the Gaussian center ¯µj, opacity ¯αj, covariance Σj, and spherical

<!-- page 8 -->
8
W. Wang et al.
harmonic color representation cj. To obtain the final rendering parameters, we
apply the following transformations:
  \ m u  _j &= r \cdo t  (\sigma
 ( \ bar {\mu }_j)-0.5) + \mathrm {Center}_j \nonumber ,\\ \alpha _j &= \sigma (\bar {\alpha }_j), \label {eq:gaussian_params}
(9)
where µj is the predicted 3D Gaussian center, and Centerj is the centroid of voxel
v. We utilize the sigmoid activation σ(·) to restrict the learnable offset within
a localized neighborhood. Specifically, the −0.5 shift facilitates symmetrical, bi-
directional movement from the voxel center, while r (set to 3× voxel size) acts
as a scaling factor to control the effective spatial extent of these refinements.
3.4
Training Objectives
Our network predicts a collection of 3D Gaussians {(µv, αv, Σv, cv)}v∈V. These
per-voxel Gaussians are subsequently used to synthesize images at novel cam-
era poses. To ensure a fair comparison and maintain benchmarking consistency
with SOTA feed-forward methods, we follow the training protocol established
by DepthSplat [47]. Specifically, the network is trained end-to-end using ground-
truth RGB images as supervision. For a forward pass that renders M novel views,
we optimize a combined photometric and perceptual loss:
  
\
m
ath
c
al {L} = 
\sum _{ m=1}
^{ M }  \left ( \ma
thcal { L}_{
\m a
t
hrm {MSE}}(I_{\mathrm {render}}^{(m)}, I_{\mathrm {gt}}^{(m)}) + \lambda \mathcal {L}_{\mathrm {LPIPS}}(I_{\mathrm {render}}^{(m)}, I_{\mathrm {gt}}^{(m)}) \right ), \label {eq:gs_loss} 
(10)
where M is the number of novel views rendered in a single pass. Following
DepthSplat [47], the weight λ for the perceptual loss LLPIPS is set to 0.05.
4
Experiments
4.1
Experimental Setup
Datasets. We train our method using two expansive datasets, RealEstate10K [57]
and ScanNet [9], and evaluate its performance on the held-out test splits of both.
For RealEstate10K, we adopt the conventional partition of 67,477 training scenes
and 7,289 test scenes. As for ScanNet, which consists of 1,513 videos of indoor
scenes, we follow past work [14, 44, 56] in using roughly 100 scenes for train-
ing and 8 scenes for evaluation. These datasets span a wide variety of environ-
ments, including indoor and outdoor real-estate walkthroughs (RealEstate10K),
and real-world videos of numerous scenes suitable for indoor robot applications
(ScanNet). We resize training and test images to 256 × 256.
Baselines. We benchmark VolSplat against several recent feed-forward meth-
ods for sparse-view novel view synthesis, including both pixel-aligned and en-
hanced pixel-aligned Gaussian splatting approaches. Pixel-aligned methods pre-
dict Gaussian parameters on a per-pixel basis in image space before unprojecting

<!-- page 9 -->
VolSplat
9
Inputs
VolSplat
Ground Truth
AnySplat WorldMirror
x
MVSplat
DepthSplat
GGN
Fig. 4: Qualitative comparison on RealEstate10K [57]. We compare VolSplat
against SOTA pixel-aligned baselines under sparse-view inputs. While competing meth-
ods often suffer from blurring and geometric distortions in complex environments, Vol-
Splat leverages its voxel-aligned prediction to maintain superior visual fidelity and
structural consistency.
to 3D. These include pixelSplat [3], MVSplat [5], FreeSplat [44], TranSplat [52]
and DepthSplat [47]. Gaussian Graph Network (GGN) [55], AnySplat [18] and
WorldMirror [28] refines the pixel-aligned approach by modeling the relation-
ships between groups of predicted Gaussians across different views while building
upon it. In contrast to both pixel-aligned and enhanced pixel-aligned methods,
our VolSplat employs a voxel-aligned approach, predicting Gaussian primitives
within a 3D voxel grid. This method aggregates multi-view evidence in 3D space,
aligning Gaussian predictions to a voxel structure, which facilitates better geo-
metric consistency and efficient redundancy reduction.
Metrics. For quantitative evaluation, we adopt standard image quality metrics
commonly used in NVS, including pixel-level PSNR, patch-level SSIM [46],and
feature-level LPIPS [54].
Implementation details. We implement VolSplat using PyTorch [36] and op-
timize the model with the AdamW [30] optimizer and a cosine learning rate
schedule. The monocular Vision Transformer backbone is implemented using
the xFormers [24] library. For the pre-trained Depth Anything V2 [49] back-
bone, we use a lower learning rate of 2 × 10−6, while other layers are trained

<!-- page 10 -->
10
W. Wang et al.
Table 1: Quantitative comparisons on RealEstate10K [57]. The top section
are pixel-aligned methods, and the middle section are methods that performs post-
processing on pixel-aligned Gaussians. All baselines are retrained for fair comparison.
Ground truth camera poses are provided for both AnySplat [18] and WorldMirror [28].
“OOM” represents that model cannot infer on a 96G GPU. We compare VolSplat
against pixel-aligned and post-processing baselines under 6, 12, and 24 input views.
VolSplat consistently achieves the best performance across all metrics and view settings.
Method
6v
12v
24v
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
pixelSplat [3]
28.95
0.900
0.163
OOM
OOM
OOM
OOM
OOM
OOM
MVSplat [5]
29.13
0.924
0.091
26.97
0.912
0.101
26.23
0.903
0.108
TranSplat [52]
29.62
0.928
0.084
28.00
0.920
0.089
26.65
0.884
0.115
DepthSplat [47]
30.52
0.931
0.079
28.54
0.919
0.088
26.26
0.880
0.115
GGN [55]
26.68
0.879
0.133
25.83
0.870
0.142
23.86
0.840
0.171
AnySplat [18]
19.05
0.576
0.305
19.86
0.627
0.288
20.21
0.658
0.279
WorldMirror [28] 24.86
0.819
0.079
26.03
0.859
0.072
26.47
0.875
0.071
Ours
31.30 0.941
0.075
29.40 0.928
0.085
27.21 0.896
0.111
FreeSplat
Inputs
VolSplat
Ground Truth
AnySplat
WorldMirror
Fig. 5: Qualitative comparison on ScanNet [9]. Compared to recent baseline
methods (FreeSplat [44], AnySplat [18], and WorldMirror [28]), VolSplat significantly
reduces common floaters and visual artifacts. Our method produces cleaner object
boundaries and a more coherent 3D scene reconstruction.

<!-- page 11 -->
VolSplat
11
Inputs
VolSplat
Ground Truth
MVSplat
DepthSplat
GGN
TranSplat
Fig. 6: Qualitative results on ACID [26]. Despite being trained solely on
RealEstate10K [57], VolSplat generalizes exceptionally well to outdoor scenes. Our
method reconstructs fine-grained details in natural landscapes and structures, produc-
ing photorealistic novel views where baseline methods often fail to maintain visual
coherence.
Table 2: Quantitative comparisons
on ScanNet [9] (6 input views). We
compare VolSplat against SOTA meth-
ods that rely on post-hoc Gaussian fusion
strategies. VolSplat outperforms all base-
lines.
Method
PSNR ↑SSIM ↑LPIPS ↓
FreeSplat [44]
27.45
0.829
0.222
FreeSplat++ [45]
27.45
0.829
0.223
AnySplat [18]
19.45
0.626
0.344
WorldMirror [28]
25.83
0.819
0.136
VolSplat
28.41
0.906
0.127
Table 3: Cross-dataset generaliza-
tion on ACID [26] (6 input views).
Models trained on RealEstate10K [57]
(indoor scenes) are directly evaluated on
ACID [26] (outdoor scenes) without fine-
tuning.
Method
PSNR ↑SSIM ↑LPIPS ↓
MVSplat [5]
28.15
0.841
0.147
TranSplat [52]
28.17
0.842
0.146
DepthSplat [47]
28.37
0.847
0.141
GGN [55]
26.97
0.814
0.196
VolSplat
32.65
0.932
0.092
with a learning rate of 2 × 10−4 following DepthSplat [47]. For experiments on
the RealEstate10K [57] and ScanNet [9] dataset, we train the model for 150,000
iterations using 4× NVIDIA H20 GPUs with a total batch size of 4. Following
the setting of the baseline, we use 256 × 256 as input resolution. In the train-
ing stage, the number of input views is set to 6, and we evaluate the model’s
performance with same numbers of input views. We will make our codes and
pre-trained models publicly available.
4.2
Experimental Results and Analysis
Comparisons with SOTA models. As shown in Tab. 1 and Tab. 2, we re-
port VolSplat’s performance compared to current mainstream pixel-aligned mod-
els [3,5,44,47,52] and their variants [18,28,55]. On both the RealEstate10K [57]
and ScanNet [9] datasets, VolSplat achieves SOTA results. Notably, since AnyS-
plat [18] and WorldMirror [28] are primarily designed for joint pose-geometry
optimization in pose-free settings, they fail to effectively utilize the provided
ground truth camera poses, resulting in lower performance compared to other

<!-- page 12 -->
12
W. Wang et al.
DepthSplat
VolSplat (Ours)
Fig. 7: Visualization of Gaussians and density maps. We compare the rendered
results and the spatial distribution of Gaussian centers between DepthSplat [47] and
VolSplat. DepthSplat is constrained by the pixel grid, resulting in a uniform but re-
dundant distribution regardless of scene content. In contrast, VolSplat adaptively con-
centrates Gaussians on complex geometric structures (e.g., the washbasin boundaries)
while remaining sparse in flat or empty regions, demonstrating a much more efficient
and geometry-aware representation.
baselines. Our experiments reveal a critical distinction between pixel-aligned and
voxel-aligned paradigms. A key observation is that under sparse multi-view set-
tings, all pixel-aligned models exhibit a significant degradation in performance.
In contrast, VolSplat demonstrates promising performance to these challeng-
ing conditions. As illustrated in
Fig. 4 and
Fig. 5, images rendered by our
method are largely free of the common floaters and artifacts that plague com-
peting methods at object boundaries. This visual improvement stems directly
from the ability of our model to resolve multi-view alignment issues within its
3D feature representation, resulting in cleaner edges and a more coherent 3D
scene reconstruction.
Cross-dataset generalization. We assess the generalization capabilities of
our model on unseen outdoor datasets to verify its broad reliability. To this end,
we conducted a cross-dataset generalization experiment by taking our model
pre-trained on the RealEstate10K [57] dataset and evaluating it directly on the
ACID [26] dataset without any fine-tuning. As demonstrated in Tab. 3 and Fig. 6,
VolSplat maintains significantly higher performance in this zero-shot transfer set-
ting. We attribute this superior generalization to the inherent robustness of our
voxel-aligned framework. Pixel-aligned models exhibit a much higher sensitivity
to the variations in data complexity and distribution between different datasets.
In contrast, VolSplat is less susceptible to these domain shifts.
Analysis of Gaussian density. A fundamental principle of 3D reconstruc-
tion is that the complexity of the representation should adapt to the complexity
of the scene. Real-world environments contain a mix of simple, planar surfaces
and intricate, high-frequency geometric details. An ideal model should allocate
its descriptive capacity accordingly. However, pixel-aligned methods are inher-
ently limited in this regard. Their paradigm of predicting one Gaussian per pixel
results in a fixed number of primitives, predetermined by the input image reso-
lution (e.g., H × W Gaussians from a reference view), regardless of whether the
scene is a simple room or a complex outdoor environment.
In stark contrast, our voxel-aligned framework enables adaptive control over
the density of the 3D Gaussians. By predicting primitives based on the occu-

<!-- page 13 -->
VolSplat
13
Table 4: Analysis of voxel size. “PGS” stands for “average number of per-view
Gaussians”. We investigate the impact of voxel resolution on reconstruction quality
and efficiency. A voxel size of 0.1 yields the optimal trade-off, achieving the best per-
formance. Notably, further reducing the voxel size to 0.05 degrades performance due
to the loss of coherent spatial context in overly sparse grids, while larger voxels fail to
capture fine geometric details.
Voxel Size (cm) PSNR ↑SSIM ↑LPIPS ↓PGS Memory(GB) Inference Time(s)
0.05
29.34
0.919
0.092
65415
9.19
0.802
0.1 (default)
29.40
0.928
0.085
60523
9.04
0.768
0.5
27.33
0.899
0.108
59788
8.98
0.744
1
20.78
0.602
0.323
51806
8.74
0.739
Table 5: Ablation of sparse 3D decoder. “w/ 3D CNN” means replacing the 3D
U-Net with a sparse 3D CNN, “w/o residual” means predicting refined voxel feature
without residual design, and “w/o decoder” means removing the refinement stage. We
validate the necessity of our specific refinement module. Replacing our proposed sparse
3D decoder for feature refinement results in performance drop. This demonstrates that
our specific design is essential for refining the voxel features and producing high-quality
3D Gaussians.
Components PSNR ↑SSIM ↑LPIPS ↓Memory(GB) Inference Time(s)
default
29.40
0.928
0.085
9.04
0.768
w/ 3D CNN
28.01
0.919
0.098
9.03
0.705
w/o residual
27.92
0.908
0.101
9.04
0.765
w/o decoder
27.47
0.901
0.102
8.99
0.687
pancy of 3D voxel features, VolSplat naturally allocates a higher concentration of
Gaussians to regions of high geometric detail while using a sparser representation
for simple or empty spaces.
This adaptive capability is quantitatively validated by the results we reported
in Tab. 1, Tab. 2 and Tab. 3. Here, we analyze these findings in greater detail.
The data shows that pixel-aligned methods consistently generate constant den-
sity of Gaussians, irrespective of the scene content. This leads to significant
redundancy, as well as an insufficient representational capacity in areas with
intricate details. Conversely, Gaussians of VolSplat demonstrate significant vari-
ance across different regions, confirming its ability to tailor complexity of the
scene, as shown in Fig. 7. Notably, VolSplat often achieves superior rendering
quality with a non-uniform set of Gaussians compared to the brute-force density
of pixel-aligned approaches.
4.3
Ablation Study
In this section, we study the properties of our key components with 12 input
views on the RealEstate10K [57] dataset.

<!-- page 14 -->
14
W. Wang et al.
Ablation of Voxel Size. The voxel size is a critical hyperparameter in our
framework, as it dictates the resolution of the 3D feature grid. This choice
involves a fundamental trade-off between the fidelity of the geometric repre-
sentation and computational resource consumption. In Tab. 4, we analyze this
trade-off by comparing our default setting against configurations with different
voxels.
Using a small voxel size increases the granularity of the 3D grid, allowing
the model to capture finer geometric details. It comes at a significant cost, sub-
stantially increasing memory usage and processing time due to the cubic growth
of the voxel volume. Conversely, employing a large voxel size reduces the com-
putational footprint but results in a coarser quantization of the 3D space. Our
default configuration strikes an effective balance, achieving SOTA performance
while maintaining manageable computational requirements.
Ablation of Model Architecture. Directly predicting Gaussians from the ini-
tial unprojected 3D features is less effective, particularly for challenging scenes
with complex geometry or sparse viewpoints. To address this, we incorporate a
3D U-Net architecture to refine and enhance this raw feature volume, predict-
ing the residual features. To validate the necessity and efficacy of this design,
we conduct an ablation study with three variants: 1) removing the refinement
module entirely, 2)directly predicting the refined feature, and 3) replacing the
3D U-Net with a standard 3D CNN.
The results, presented in Tab. 5, confirm our architectural choices. Remov-
ing the refinement stage altogether leads to a significant drop in performance,
demonstrating that processing the initial voxel features is critical for producing
a coherent 3D representation. While substituting our module with a sparse 3D
CNN or removing the residual design yields better results than no refinement, it
still falls short of the performance of our full model. The multi-scale feature fu-
sion inherent in the U-Net structure and residual design are crucial for capturing
both fine-grained local details and broader spatial context.
5
Conclusion
We address the fundamental limitations inherent in the prevailing pixel-aligned
paradigm for feed-forward 3D Gaussian Splatting. We identify that existing
methods suffer from a rigid coupling of Gaussian density to input image res-
olution and a high sensitivity to multi-view alignment errors. To overcome these
challenges, we introduce VolSplat, a novel framework that fundamentally shifts
the reconstruction process from 2D pixels to a 3D voxel-aligned space. By con-
structing 3D voxel feature and predicting Gaussians directly from this unified
representation, our method effectively decouples the 3D scene from the con-
straints of the input views. This voxel-centric design enables adaptive control
over Gaussian density according to scene complexity and inherently resolves
alignment ambiguities, leading to more geometrically consistent and faithful re-
constructions for downstream tasks.

<!-- page 15 -->
VolSplat
15
References
1. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srini-
vasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance
fields. In: ICCV. pp. 5855–5864 (2021) 3
2. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Zip-nerf:
Anti-aliased grid-based neural radiance fields. In: ICCV. pp. 19697–19705 (2023)
3
3. Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats
from image pairs for scalable generalizable 3d reconstruction. In: CVPR. pp. 19457–
19467 (2024) 1, 2, 4, 9, 10, 11, 19, 20, 21
4. Chen, A., Xu, Z., Zhao, F., Zhang, X., Xiang, F., Yu, J., Su, H.: Mvsnerf: Fast
generalizable radiance field reconstruction from multi-view stereo. In: ICCV. pp.
14124–14133 (2021) 1
5. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J.,
Cai, J.: Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In:
ECCV. pp. 370–386. Springer (2024) 1, 2, 4, 5, 6, 9, 10, 11, 19, 21
6. Chen, Y., Zheng, C., Xu, H., Zhuang, B., Vedaldi, A., Cham, T.J., Cai, J.: Mvs-
plat360: Feed-forward 360 scene synthesis from sparse views. NeurIPS 37, 107064–
107086 (2024) 1
7. Choy, C., Gwak, J., Savarese, S.: 4d spatio-temporal convnets: Minkowski convo-
lutional neural networks. In: CVPR. pp. 3075–3084 (2019) 19
8. Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O.: 3d u-net:
learning dense volumetric segmentation from sparse annotation. In: International
conference on medical image computing and computer-assisted intervention. pp.
424–432. Springer (2016) 2, 7, 21
9. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet:
Richly-annotated 3d reconstructions of indoor scenes. In: CVPR. pp. 5828–5839
(2017) 3, 8, 10, 11, 19
10. Debevec, P.E., Taylor, C.J., Malik, J.: Modeling and rendering architecture from
photographs: a hybrid geometry- and image-based approach. In: Proceedings of
the 23rd Annual Conference on Computer Graphics and Interactive Techniques.
p. 11–20. SIGGRAPH ’96, Association for Computing Machinery, New York, NY,
USA (1996). https://doi.org/10.1145/237170.237191 3
11. Fan, Z., Wang, K., Wen, K., Zhu, Z., Xu, D., Wang, Z., et al.: Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps. NeurIPS
37, 140138–140158 (2024) 3
12. Fridovich-Keil, S., Meanti, G., Warburg, F.R., Recht, B., Kanazawa, A.: K-planes:
Explicit radiance fields in space, time, and appearance. In: CVPR. pp. 12479–12488
(2023) 4
13. Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenox-
els: Radiance fields without neural networks. In: CVPR. pp. 5501–5510 (2022) 4
14. Gao, Y., Cao, Y.P., Shan, Y.: Surfelnerf: Neural surfel radiance fields for online
photorealistic reconstruction of indoor scenes. In: CVPR. pp. 108–118 (2023) 8
15. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition.
In: CVPR. pp. 770–778 (2016) 6, 19
16. Huang, R., Mikolajczyk, K.: No pose at all: Self-supervised pose-free 3d gaussian
splatting from sparse views. arXiv preprint arXiv: 2508.01171 (2025) 4
17. Ji, D., Kwon, J., McFarland, M., Savarese, S.: Deep view morphing. In: CVPR.
pp. 2155–2163 (2017) 3

<!-- page 16 -->
16
W. Wang et al.
18. Jiang, L., Mao, Y., Xu, L., Lu, T., Ren, K., Jin, Y., Xu, X., Yu, M., Pang, J.,
Zhao, F., Lin, D., Dai, B.: Anysplat: Feed-forward 3d gaussian splatting from
unconstrained views. arXiv preprint arXiv:2505.23716 (2025) 4, 9, 10, 11, 19, 21
19. Kang, G., Nam, S., Sun, X., Khamis, S., Mohamed, A., Park, E.: ilrm: An iterative
large 3d reconstruction model. arXiv preprint arXiv:2507.23277 (2025) 1, 2, 4
20. Kang, G., Yoo, J., Park, J., Nam, S., Im, H., Shin, S., Kim, S., Park, E.: Selfsplat:
Pose-free and 3d prior-free generalizable 3d gaussian splatting. In: CVPR. pp.
22012–22022 (2025) 4
21. Kaufman, A.E., Mueller, K.: Overview of volume rendering. The visualization
handbook 7, 127–174 (2005) 4
22. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM TOG 42(4), 139–1 (2023) 1, 3
23. Koneputugodage, C.H., Ben-Shabat, Y., Gould, S.: Octree guided unoriented sur-
face reconstruction. In: CVPR. pp. 16717–16726 (2023) 4
24. Lefaudeux, B., Massa, F., Liskovich, D., Xiong, W., Caggiano, V., Naren, S., Xu,
M., Hu, J., Tintore, M., Zhang, S., Labatut, P., Haziza, D., Wehrstedt, L., Reizen-
stein, J., Sizov, G.: xformers: A modular and hackable transformer modelling li-
brary. https://github.com/facebookresearch/xformers (2022) 9
25. Levoy, M., Hanrahan, P.: Light field rendering. In: Proceedings of the 23rd An-
nual Conference on Computer Graphics and Interactive Techniques. p. 31–42. SIG-
GRAPH ’96, Association for Computing Machinery, New York, NY, USA (1996).
https://doi.org/10.1145/237170.237199 3
26. Liu, A., Tucker, R., Jampani, V., Makadia, A., Snavely, N., Kanazawa, A.: Infinite
nature: Perpetual view generation of natural scenes from a single image. In: ICCV.
pp. 14458–14467 (2021) 3, 11, 12, 19
27. Liu, H., Wang, Y., Li, C., Cai, R., Wang, K., Li, W., Molchanov, P., Wang, P.,
Wang, Z.: Flexgs: Train once, deploy everywhere with many-in-one flexible 3d
gaussian splatting. In: CVPR. pp. 16336–16345 (2025) 3
28. Liu, Y., Min, Z., Wang, Z., Wu, J., Wang, T., Yuan, Y., Luo, Y., Guo, C.: World-
mirror: Universal 3d world reconstruction with any-prior prompting. arXiv preprint
arXiv:2510.10726 (2025) 4, 9, 10, 11, 19, 20, 21
29. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin
transformer: Hierarchical vision transformer using shifted windows. In: ICCV. pp.
10012–10022 (2021) 6, 19
30. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint
arXiv:1711.05101 (2017) 9, 19
31. Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., Dai, B.: Scaffold-gs: Struc-
tured 3d gaussians for view-adaptive rendering. In: CVPR. pp. 20654–20664 (2024)
4
32. Meagher, D.: Geometric modeling using octree encoding. Computer graphics and
image processing 19(2), 129–147 (1982) 3
33. Miao, S., Huang, J., Bai, D., Yan, X., Zhou, H., Wang, Y., Liu, B., Geiger, A., Liao,
Y.: Evolsplat: Efficient volume-based gaussian splatting for urban view synthesis.
In: CVPR. pp. 11286–11296 (2025) 4
34. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng,
R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Commu-
nications of the ACM 65(1), 99–106 (2021) 1, 3
35. Odena, A., Dumoulin, V., Olah, C.: Deconvolution and checkerboard artifacts. Dis-
till (2016). https://doi.org/10.23915/distill.00003, http://distill.pub/
2016/deconv-checkerboard 5

<!-- page 17 -->
VolSplat
17
36. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen,
T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, high-
performance deep learning library. NeurIPS 32 (2019) 9
37. Ren, K., Jiang, L., Lu, T., Yu, M., Xu, L., Ni, Z., Dai, B.: Octree-gs: Towards
consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint
arXiv:2403.17898 (2024) 4
38. Riegler, G., Osman Ulusoy, A., Geiger, A.: Octnet: Learning deep 3d representa-
tions at high resolutions. In: CVPR. pp. 3577–3586 (2017) 4
39. Shi, D., Wang, W., Chen, D.Y., Zhang, Z., Bian, J., Zhuang, B., Shen, C.: Revis-
iting depth representations for feed-forward 3d gaussian splatting. arXiv preprint
arXiv:2506.05327 (2025) 3
40. Wang, Q., Wang, Z., Genova, K., Srinivasan, P.P., Zhou, H., Barron, J.T., Martin-
Brualla, R., Snavely, N., Funkhouser, T.: Ibrnet: Learning multi-view image-based
rendering. In: CVPR. pp. 4690–4699 (2021) 1
41. Wang, T., Mao, X., Zhu, C., Xu, R., Lyu, R., Li, P., Chen, X., Zhang, W., Chen, K.,
Xue, T., et al.: Embodiedscan: A holistic multi-modal 3d perception suite towards
embodied ai. In: CVPR. pp. 19757–19767 (2024) 6
42. Wang, W., Chen, D.Y., Zhang, Z., Shi, D., Liu, A., Zhuang, B.: Zpressor:
Bottleneck-aware compression for scalable feed-forward 3dgs. arXiv preprint
arXiv:2505.23734 (2025) 1, 4
43. Wang, Y., Chai, L., Luo, X., Niemeyer, M., Lagunas, M., Lombardi, S., Tang, S.,
Sun, T.: Learning efficient fuse-and-refine for feed-forward 3d gaussian splatting.
arXiv preprint arXiv:2503.14698 (2025) 4
44. Wang, Y., Huang, T., Chen, H., Lee, G.H.: Freesplat: Generalizable 3d gaussian
splatting towards free view synthesis of indoor scenes. NeurIPS 37, 107326–107349
(2024) 1, 2, 4, 8, 9, 10, 11
45. Wang, Y., Huang, T., Chen, H., Lee, G.H.: Freesplat++: Generalizable 3d gaussian
splatting for efficient indoor scene reconstruction. arXiv preprint arXiv:2503.22986
(2025) 11
46. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment:
from error visibility to structural similarity. IEEE TIP 13(4), 600–612 (2004) 9
47. Xu, H., Peng, S., Wang, F., Blum, H., Barath, D., Geiger, A., Pollefeys, M.: Depth-
splat: Connecting gaussian splatting and depth. In: CVPR. pp. 16453–16463 (2025)
1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 19, 20, 21
48. Xu, H., Zhang, J., Cai, J., Rezatofighi, H., Yu, F., Tao, D., Geiger, A.: Unifying
flow, stereo and depth estimation. IEEE TPAMI 45(11), 13941–13958 (2023) 6
49. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything
v2. arXiv:2406.09414 (2024) 9
50. Ye, B., Liu, S., Xu, H., Li, X., Pollefeys, M., Yang, M.H., Peng, S.: No pose, no
problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv
preprint arXiv:2410.24207 (2024) 4
51. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from
one or few images. In: CVPR. pp. 4578–4587 (2021) 1
52. Zhang, C., Zou, Y., Li, Z., Yi, M., Wang, H.: Transplat: Generalizable 3d gaussian
splatting from sparse multi-view images with transformers. In: AAAI. pp. 9869–
9877 (2025) 9, 10, 11, 19, 21
53. Zhang, K., Bi, S., Tan, H., Xiangli, Y., Zhao, N., Sunkavalli, K., Xu, Z.: Gs-lrm:
Large reconstruction model for 3d gaussian splatting. In: ECCV. pp. 1–19. Springer
(2024) 1

<!-- page 18 -->
18
W. Wang et al.
54. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable
effectiveness of deep features as a perceptual metric. In: CVPR. pp. 586–595 (2018)
9
55. Zhang, S., Fei, X., Liu, F., Song, H., Duan, Y.: Gaussian graph network: Learn-
ing efficient and generalizable gaussian representations from multi-view images.
NeurIPS 37, 50361–50380 (2024) 4, 9, 10, 11, 19, 21
56. Zhang, X., Bi, S., Sunkavalli, K., Su, H., Xu, Z.: Nerfusion: Fusing radiance fields
for large-scale scene reconstruction. In: CVPR. pp. 5449–5458 (2022) 8
57. Zhou, T., Tucker, R., Flynn, J., Fyffe, G., Snavely, N.: Stereo magnification: Learn-
ing view synthesis using multiplane images. arXiv preprint arXiv:1805.09817 (2018)
3, 8, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 24
58. Zhou, Y., Tuzel, O.: Voxelnet: End-to-end learning for point cloud based 3d object
detection. In: CVPR. pp. 4490–4499 (2018) 4
59. Zhu, Z., Fan, Z., Jiang, Y., Wang, Z.: Fsgs: Real-time few-shot view synthesis using
gaussian splatting. In: ECCV. pp. 145–163. Springer (2024) 3
60. Ziwen, C., Tan, H., Zhang, K., Bi, S., Luan, F., Hong, Y., Fuxin, L., Xu, Z.: Long-
lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats.
arXiv preprint arXiv:2410.12781 (2024) 4

<!-- page 19 -->
VolSplat
19
A
More Implementation Details
Network architecture. Our framework begins with a 2D feature extraction
stage. Following DepthSplat [47], we employ a weight-sharing ResNet [15] back-
bone to extract multi-scale feature maps from each input view. To enhance multi-
view consistency, these features are refined via a cross-view interaction module
implemented with local window attention [29], which efficiently aggregates in-
formation from neighboring views. Following feature extraction, the architecture
proceeds to the multi-view depth prediction module. This module constructs a
cost volume using a plane-sweep strategy with 128 inverse-depth candidates per
reference view. By performing local neighbor matching on the cost volume, the
network predicts robust depth maps that serve as the geometric basis for lift-
ing 2D features into 3D space. The lifted 3D representation is built in world
coordinates and processed as a sparse voxel set. Specifically, we employ a sparse
data structure where only occupied cells are materialized, avoiding the computa-
tional redundancy of a dense global grid. This sparse 3D refinement is efficiently
implemented using MinkowskiEngine [7]. In the default configuration, each occu-
pied voxel predicts one Gaussian primitive with a 38-dimensional parameter vec-
tor, including opacity, center offset, anisotropic scale, quaternion rotation, and
degree-2 spherical-harmonic color coefficients. This design ensures that primitive
allocation remains scene-adaptive while preserving stable sparse 3D decoding.
More training details. Optimization uses AdamW [30] with decoupled weight
decay, together with two learning-rate groups for the pretrained monocular
branch and the remaining trainable parameters, and radient clipping is enabled
for stability. During training, each sample uses 6 input views and 8 target views.
Two input views are first selected as boundary anchors with a randomly sam-
pled frame gap in a predefined range, and the remaining input views are sampled
between these anchors. The anchor-gap range is progressively expanded during
early iterations. All experiments are run at 256 × 256 input resolution. Training
is performed on RealEstate10K [57] and then fine-tuned on ScanNet [9] from the
RealEstate10K checkpoint, while ACID [26] is evaluated in a zero-shot manner.
For details on training objectives and weights, please refer to Sec. 3.4.
Evaluation. Evaluation follows a controlled protocol for fair comparison with
prior feed-forward 3DGS methods [3, 5, 18, 28, 47, 52, 55]. For each scene, input
views are selected using fixed frame-gap rules, and target novel views are chosen
from disjoint camera positions that are not included in the inputs. In our setup,
each sample is evaluated on 8 target novel views.
Open-source. Our source codes are provided in the supplementary material.
We will open-source the complete codebase for VolSplat.

<!-- page 20 -->
20
W. Wang et al.
Table A: Efficiency comparison. All methods are evaluated with 6 input views
on RealEstate10K [57] via a single NVIDIA H20 GPU. Our VolSplat tops all image-
quality metrics while retaining competitive inference efficiency even though utilizing 3D
features. Note that absolute runtimes may differ from original studies due to hardware
discrepancies, but relative rankings ensure fair efficiency comparison.
Method
PSNR↑SSIM↑LPIPS↓Memory (GB) Infer time (s)
pixelSplat
28.95
0.900
0.163
36.82
0.579
MVSplat
29.13
0.924
0.091
4.70
0.369
TranSplat
29.62
0.928
0.084
3.96
1.002
DepthSplat
30.52
0.931
0.079
8.00
0.513
GGN
26.68
0.879
0.133
4.70
0.377
AnySplat
19.05
0.576
0.305
3.57
0.332
World-Mirror
24.86
0.819
0.079
8.05
0.375
Ours
31.30
0.941
0.075
4.65
0.575
B
More Experimental Analysis
Efficiency Analysis. We evaluate the computational efficiency of our method
against state-of-the-art baselines on RealEstate10K [57] as illustrated in Tab. A.
All metrics utilize a single NVIDIA H20 GPU with 6 input views to ensure same
evaluation environment. VolSplat achieves a superior balance between recon-
struction quality and resource consumption despite incorporating explicit 3D
feature processing. Our approach outperforms all competing approaches while
maintaining a competitive inference latency of 0.575s. This runtime remains
comparable to leading pixel-aligned approaches [47], which demonstrates that
our voxel-aligned architecture introduces negligible overhead relative to the sig-
nificant performance gains. Furthermore, VolSplat exhibits competitive memory
consumption by requiring only 4.65GB of VRAM. This footprint is substan-
tially lower than heavy-weight baselines [3,28]. Our method remains on par with
lightweight alternatives such as MVSplat and ensures practicality for deployment
on consumer-grade hardware. While absolute runtime values may vary due to
hardware discrepancies compared to original publications, the relative rankings
presented here reflect a controlled and fair comparison.
C
Limitation and Societal Impacts
Limitation analysis. Our current framework assumes a static scene assump-
tion during the multi-view feature aggregation and voxel construction steps.
Consequently, VolSplat struggle to reconstruct dynamic objects or changing en-
vironments, as the geometric consistency enforced by our cost volume and sparse

<!-- page 21 -->
VolSplat
21
3D U-Net [8] relies on multi-view consistency. Moving elements in the scene can
lead to artifacts such as ghosting or blurring in the rendered novel views.
Potential and negative societal impacts. VolSplat significantly advances the
capability of feed-forward 3D reconstruction by decoupling geometry from input
pixel resolution. It enables the generation of high-fidelity 3D assets from sparse
multi-view images with superior geometric consistency, positioning VolSplat as
a valuable tool for immersive applications such as virtual reality, gaming, and
digital twin creation.
While the ability to produce photorealistic 3D reconstructions from limited
data is beneficial for content creation, it is important to acknowledge the poten-
tial for misuse. The high fidelity of the generated scenes could be exploited to
create deepfakes or unauthorized digital replicas of private spaces. Consequently,
the deployment of VolSplat in sensitive contexts should be accompanied by ro-
bust watermarking techniques or authentication protocols to mitigate the risks
associated with the synthesis of misleading or non-consensual 3D content.
D
More Visual Comparisons
This section provides additional qualitative comparison results. We present fur-
ther visualizations for VolSplat on the RealEstate10K [57] dataset, comparing
against SOTA baselines [3, 5, 18, 28, 47, 52, 55]. To illustrate how VolSplat per-
forms with varying numbers of input views, we showcase comparative results
across different settings. For the standard setting, comparisons with 6 input
views are presented in Fig. A. To demonstrate scalability to denser inputs, vi-
sual comparisons between our method and competing baselines are displayed for
scenarios with 12 and 24 input views in Fig. B and Fig. C, respectively. The cor-
responding quantitative results for these multi-view experiments can be found
in Tab. 1.

<!-- page 22 -->
22
W. Wang et al.
Inputs
VolSplat
Ground Truth
AnySplat
WorldMirror
MVSplat
DepthSplat
GGN
Fig. A: More qualitative comparisons on RealEstate10K [57] under 6 input
views.

<!-- page 23 -->
VolSplat
23
Inputs
VolSplat
Ground Truth
AnySplat
WorldMirror
MVSplat
DepthSplat
GGN
Fig. B: More qualitative comparisons on RealEstate10K [57] under 12 input
views.

<!-- page 24 -->
24
W. Wang et al.
Inputs
VolSplat
Ground Truth
AnySplat
WorldMirror
MVSplat
DepthSplat
GGN
Fig. C: More qualitative comparisons on RealEstate10K [57] under 24 input
views.
