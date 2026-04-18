<!-- page 1 -->
Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting
Arthur Moreau
Richard Shaw
Michal Nazarczuk
Jisu Shin
Thomas Tanay
Zhensong Zhang
Songcen Xu
Eduardo P´erez-Pellitero
Huawei Noah’s Ark Lab
Pixel-aligned Gaussians
Voxel-aligned Gaussians
Off The Grid Gaussians (Ours)
Detection of primitives
#Gaussians: 1 492 696
#Gaussians: 903 878
#Gaussians: 185 389
Pixel-aligned Gaussians
Voxel-aligned Gaussians
Off The Grid Gaussians (Ours)
Detection of primitives
Figure 1. 3D Gaussians placement techniques. Our model learns the position of primitives instead of using regular grids, representing
the scene more accurately with fewer primitives. Voxel-aligned uses AnySplat [18] and Pixel-aligned is an ablated version of our model.
Abstract
Feed-forward 3D Gaussian Splatting (3DGS) models en-
able real-time scene generation but are hindered by sub-
optimal pixel-aligned primitive placement, which relies on
a dense, rigid grid that limits both quality and efficiency.
We introduce a new feed-forward architecture that detects
3D Gaussian primitives at a sub-pixel level, replacing the
pixel grid with an adaptive, “Off-The-Grid” distribution.
Inspired by keypoint detection, our decoder learns to lo-
cally distribute primitives across image patches. We also
provide an Adaptive Density mechanism by assigning vary-
ing number of primitives per patch based on Shannon
entropy.
We combine the proposed decoder with a pre-
trained 3D reconstruction backbone and train them end-
to-end using photometric supervision without any 3D an-
notation. The resulting pose-free model generates photore-
alistic 3DGS scenes in seconds, achieving state-of-the-art
novel view synthesis for feed-forward models. It outper-
forms competitors while using far fewer primitives, demon-
strating a more accurate and efficient allocation that cap-
tures fine details and reduces artifacts.
Project page:
https://arthurmoreau.github.io/OffTheGrid/.
1. Introduction
The recent introduction of 3D Gaussian Splatting [24]
(3DGS) has marked a significant leap forward in the re-
construction of photorealistic 3D scenes. This point-based
scene representation is highly efficient to render, enabling
photorealistic interactive applications [19, 34, 37, 47], al-
though it is generally slower to fit from a set of images
than some of its predecessors [38].
Starting from im-
ages, the standard pipeline first performs 3D reconstruction
with Structure-from-Motion [46] (SfM), and then optimizes
Gaussian primitive parameters by rendering images in an
iterative fashion. Following the initial optimization proce-
dure, this process typically takes tens of minutes to hours.
An alternative approach uses feed-forward models [4,
18] that predict 3D Gaussians directly from images through
neural networks, enabling scene reconstruction in a single
feed-forward step.
Early research has primarily focused
on developing pose-free methods [15, 22, 68] that remove
dependence on pre-computed camera poses. This task is
challenging because the model needs to solve the 3D re-
construction problem before predicting photorealistic prim-
itives. Less attention has been given to designing effective
ways to accurately decode 3D Gaussian primitives from im-
1
arXiv:2512.15508v2  [cs.CV]  30 Mar 2026

<!-- page 2 -->
ages to obtain higher-quality models.
Existing works typically predict pixel-aligned Gaus-
sians [4], unprojecting one primitive per input pixel into
the 3D scene using depth. With as many primitives as in-
put pixels, these models are limited to operating at low res-
olutions with sparse image collections. Beyond primitive
count, we question whether a regular grid-based distribu-
tion is optimal to obtain photorealistic models. As a com-
parison, optimization techniques for Gaussian Splatting use
densification and pruning strategies to distribute primitives
across the scene and ensure a good representation of high-
frequency details. Current feed-forward models lack the
ability to achieve this. Toward more accurate and scalable
solutions, we argue that feed-forward models require more
expressivity in the positioning of primitives.
To address this, we propose to adaptively control the
allocation of primitives at 3 different levels.
First, we
detect 3D Gaussian primitives in image patches at sub-
pixel level, inspired by common keypoint detection tech-
niques [11]. Naturally, there is no actual ellipsoid primi-
tive to be detected in the physical world. However, we see
an opportunity to leverage 2D keypoints as a way to learn
how to optimally distribute primitives across the image in
a self-supervised manner. At a local level, the model can
place primitive centers off the grid by extracting continu-
ous 2D point coordinates from convolutional heatmaps [40].
Then, we define an adaptive density mechanism with a
multi-density decoder, where high entropy patches are as-
signed more primitives than homogeneous ones. Finally,
we learn per-Gaussian confidence values to aggregate prim-
itives from different images, giving the model the ability to
discard primitives when they are not needed.
This decoding strategy is combined with a large 3D re-
construction model, VGGT [56]. Given the geometry pre-
dicted by this model, our decoder learns where and how to
place Gaussians on this geometry to obtain a photorealistic
model. By rendering the 3D Gaussians back to the input
images, we define an end-to-end self-supervised loop that
learns primitive detection and fine-tunes the reconstruction
model without any annotation. We obtain a pose-free 3DGS
model that can generate photorealistic reconstructions of
any scene within seconds. We outperform state-of-the-art
competitors on novel view synthesis while using 7 times
fewer primitives than input pixels.
We observe that our
method fit details accurately, avoid floating artifacts, and
allocate computational resources more effectively.
In summary, we introduce the following contributions:
1. A solution to place primitives at a sub-pixel level, per-
forming better than pixel and voxel aligned approaches,
2. A multi-density adaptive mechanism that allocates prim-
itives dynamically depending on image patch content,
3. A pose-free feed-forward Gaussian Splatting model that
outperforms existing methods on novel view synthesis.
2. Related Work
Neural Rendering of 3D Scenes. Neural Radiance Fields
(NeRF) by Mildenhall et al. [36] introduced a new paradigm
to fit 3D representations to 2D posed images through differ-
entiable volumetric rendering coupled with stochastic gra-
dient descent.
3D Gaussian Splatting (3DGS) [24] rep-
resented a breakthrough in rendering efficiency and over-
came some of NeRF’s inherent computational bottlenecks.
3DGS represents the world with a set of volumetric prim-
itives shaped as 3D Gaussians. This set is initialized from
a sparse point cloud, typically obtained via SfM [46], and
then iteratively optimized to render the input images. Dur-
ing this process, the set of primitives is gradually pruned
or densified, based on e.g. photometric gradient magni-
tude [24], or other heuristics [25, 32, 45], aiming to allo-
cate primitives efficiently where representation capacity is
needed. Despite notable acceleration techniques [7, 33–35],
scene-optimization methodologies often still require pre-
computed 3D camera poses and scene point clouds. Ac-
celerating the fitting stage, and defining improved strategies
for the sampling and distribution of primitives during that
process remain active topics of research [10, 43].
Feed-forward 3DGS. An alternative to optimization is to
train neural networks to efficiently predict the 3D Gaussian
model in a single forward pass. This idea builds on gen-
eralizable view synthesis methods [16, 20, 53] that have
shown great success in learning priors for rendering but
generate novel views directly instead of a 3D model much
faster to render. The pioneer work for Feed-forward 3DGS
is PixelSplat [4], that uses pixel-aligned Gaussians, e.g.
one primitive per input pixel.
This design is popular in
the prior art [52, 60, 61, 65, 66, 70], but the large number
of primitives limits its application to low resolutions (typi-
cally 256 × 256) and sparse image collections. Moreover,
areas observed in several images contain redundant primi-
tives, leading to blurry renderings. 4DGT [66] shows that
such a dense representation is not necessary, as many primi-
tives can be pruned at test time. Alternatively, voxel-aligned
Gaussians have also been proposed [18, 59], using unpro-
jected features to predict one primitive per voxel to address
the multi-view aggregation problem. We observe that these
approaches exhibit noisy geometry, and the regular grid is
often visible when zoomed in (see Figure 6). MVSplat [5]
uses a plane-sweeping cost volume which also discretizes
3D space in a regular grid. Both pixel-aligned and voxel-
aligned strategies distribute primitives in a predefined regu-
lar structure, whereas the best optimization-based methods
do not, limiting the expressivity of the model to place prim-
itives optimally in the scene. Our work proposes Off-The-
Grid Gaussians, which are placed with sub-pixel precision
and do not require explicit 3D aggregation.
2

<!-- page 3 -->
Renderer
N input images
𝐼0
𝐼𝑖
𝐼𝑁−1
Pretrained
3D Reconstruction 
Model
Multi-View
Transformer 
Model
Depth 
Head
Camera 
Params
Head
𝐷𝑖
𝐶𝑖
ሚ𝐶𝑖
෩𝐷𝑖
3D Gaussians 
Decoder
Per Frame Gaussian Model ℳ𝑖
Renderer
Multi-View Fusion
𝛼= 𝛼* 𝑐
ℳ0
ℳ𝑖
ℳ𝑁−1
Geometry Loss
RGB Loss
…
…
Token feature 𝑡𝑖
𝐼𝑖
𝐼𝑖
3D Gaussian Model ℳ
Self-Rendering
Predicted depth and normals
Rendered depth and normals
Figure 2. Overview of our pose-free 3DGS training framework. We process depth and camera parameters from N input images with a
large reconstruction model. Then, our 3D Gaussian decoder predicts primitives for each image, which are rendered and aggregated with
other views. The pipeline is trained end-to-end to reconstruct input images, with geometry consistency and regularization losses.
Pose-free Methods. An important limitation in the scal-
ability of Gaussian models is their dependence on pre-
computed camera poses. One line of research has devel-
oped pose-free approaches [12, 13, 17] that solve the camera
pose estimation problem jointly with Gaussian model re-
construction. In the context of feed-forward methods, sev-
eral works such as FLARE [72], PF3Splat [14], and VicaS-
plat [28] have developed custom pipelines that first predict
camera poses and then reconstruct the Gaussian primitives.
However, SfM remains a notoriously difficult problem for
learning-based methods, which are still outperformed by
classical pipelines based on correspondences [46]. This ob-
servation has been recently challenged by the emergence
of 3D foundation models [23, 27, 56, 57, 62], trained in
a supervised manner on large-scale datasets to reconstruct
3D scene geometry from images, predicting camera param-
eters, depth maps, and/or point maps. These models offer
a great opportunity to build pose-free feed-forward 3DGS
methods by adding a decoder that predicts Gaussian prim-
itives. NoPoSplat [68], Splatt3R [51], and SPFSplat [15]
build on Mast3R [27] and predict Gaussian centers as point
maps, but process only image pairs and lack consistency
with more images. AnySplat [18] fine-tunes VGGT [56] to
predict 3D Gaussians, using pixel-aligned Gaussians aggre-
gated in a voxel grid. We propose a pose-free method that
also fine-tunes VGGT combined with a different decoder
that leads to better visual quality and compactness.
3. Method
We present a feed-forward neural network that predicts a
3DGS model from a set of N unposed and uncalibrated im-
ages Ii<N representing a static scene. An overview of our
pipeline is shown in Figure 2.
3.1. Feed-forward 3D Reconstruction Backbone
We use VGGT [56] as the backbone of our model. It is a
large multi-view transformer that performs 3D reconstruc-
tion from unposed image collections in a single forward
pass.
Each image is first encoded into 14 × 14 patches
through DINOv2 [41].
Then, 24 transformer blocks al-
ternate between global and frame-only attention. Output
tokens ti are decoded into depth maps Di and extrinsics
[Ri|Ti] and intrinsics Ki camera parameters. Instead of us-
ing the point map head, we compute 3D positions by com-
bining the predicted depth and camera parameters. This
model is trained to predict geometry but has no rendering
ability. Similar to AnySplat [18], we fine-tune it to pre-
dict 3D Gaussians. In contrast, we do not use a separate
depth head for rendering. We fine-tune the backbone with
2 purposes: encode multi-view information useful for our
task in the latent features and improve the geometry of the
model from photometric supervision. Although we observe
the 3D geometry provided by VGGT to be fairly accurate,
predicted depth for background pixels is often inaccurately
close, producing floating artifacts. Our method mitigates
this issue by fine-tuning and generates 3D models with sig-
nificantly cleaner geometry.
3.2. 3D Gaussian Decoder
We define a convolutional module, depicted in Figure 3, that
operates after the decoding of camera parameters and depth
maps. The objective of this decoder is to detect 2D loca-
tions of primitives and to describe each primitive to predict
remaining parameters. It does not modify the 3D geometry
provided by the backbone but places primitives on it. The
decoder takes as input VGGT output tokens (ti) but also
input images Ii and predicted depth maps Di. Tokens are
transformed to an image shape by unpatchifying, i.e. project
the 1D vectors to values of a 14 × 14 patch with 8 channels
with a fully-connected layer and reshaping. Inputs are then
concatenated on the channels dimension and fed to the de-
coder. As detection is essentially a low-level vision prob-
lem, we use a simple U-Net [44] convolutional architecture.
It outputs features with the same spatial dimension as inputs
and 32 channels, from which detection and description fea-
3

<!-- page 4 -->
Detected primitives
Heatmaps H
Descriptors
Scaling 𝑠
Orientation 𝑞
Opacity 𝛼
Confidence 𝑐
𝒄𝒙
𝒄𝒚
𝑡𝑖
Unpatchify
UNet
ℎ𝑑𝑒𝑡
ℎ𝑑𝑒𝑠𝑐
Patchify
Unproject
Gaussian Model ℳ𝑖
Colors 𝜎
Centers 𝑚
Eqn. 1
Pixel coordinates
Depths
3D Gaussian parameters G
𝐼𝑖
𝐷𝑖
Descriptor features
Detection features
low
medium
high
low
medium
high
𝐼𝑖
𝐷𝑖
FC layers
Bilinear 
interpolation
Figure 3. Overview of our 3D Gaussian decoder architecture. Images, depth maps, and latent features are concatenated and fed to
a U-Net CNN from which detection and description features are extracted. First, the position of detected primitives is determined from
convolutional heatmaps. Then, image, depths and description features are bilinearly interpolated to decode Gaussian parameters through
depth unprojection and MLP.
tures are extracted separately through heads hdet and hdesc.
2D Detection of Gaussian Primitives. Primitives are not
attached to a pixel but to floating-point 2D coordinates in
image space (x, y). We use a heatmap approach to extract
these coordinates in a differentiable manner. First, we re-
shape detection features back to 14 × 14 patches with P
channels. P is the number of primitives per patch, i.e. each
primitive is assigned a one-channel patch. We perform soft-
max over spatial dimensions to obtain a heatmap for each
primitive, interpreted as the distribution of the primitive
center position in the patch. By using tensors containing
pixel coordinates cx and cy, the expectation of Gaussian po-
sitions (x, y) can be simply computed by :
x =
P
X
i,j=0
cx(i, j)h(i, j)
y =
P
X
i,j=0
cy(i, j)h(i, j)
(1)
This operation can also be seen as a soft-argmax. Nibali
et al. [40] named it DSNT and showed improved perfor-
mance on Human Pose Estimation. One main advantage is
that keypoints are not limited to the pixel grid but are de-
fined in the continuous 2D space. If we want a primitive
to represent 2 neighboring pixels, the best Gaussian center
position is between pixels. Our model can naturally achieve
that by activating the 2 pixels equally in the heatmap. By
performing this operation on all heatmaps of all patches,
we obtain the full set of Gaussian centers.
Adaptive Density of Detection. Instead of assigning a con-
stant number of primitives to each image patch, we want to
allocate more primitives to highly detailed areas. We de-
fine multiple levels of density with increasing numbers of
Gaussians per patch. We follow APT [8] and use Shan-
non entropy as a measure of patch compressibility. Sim-
ilar to their work, we compute per-patch histograms of
grayscale intensities (pk)k∈[0,255] and define per-patch en-
tropy as H = −PK−1
k=0 pk log2(pk +ϵ). We assign the 55%
lowest entropy patches to ‘low density’ (16 primitives), the
following 35% to ‘medium density’ (32) and the highest
15% to ‘high density’ (64). Note that even high-density
patches are allocated many fewer primitives than number of
pixels (196). We learn density-specific convolutional heads
hdet and hdesc to decode detection and description features
at varying levels of detail.
From 2D Points to 3D Gaussians. The decoder predicts
3D Gaussians in the camera coordinate system of each im-
age. From detected 2D pixel coordinates, we extract depth,
RGB colors and descriptors by bilinear interpolation of re-
spectively depth maps, source images and description fea-
tures. The 3D Gaussian centers m are obtained by unpro-
jection of 2D points using interpolated depth and estimated
intrinsics. We simply assign the interpolated color to the
primitive, as we observe that it performs similarly to pre-
dicting it. Finally, remaining parameters are predicted by a
small MLP from the interpolated descriptors. We decode:
• scaling parameters s ∈R3. Instead of direct regression,
we define min and max values and interpolate between
them using a sigmoid activation. We then multiply scal-
ing parameters by the depth of each Gaussian. This way,
the network predicts scale relative to its projected size,
rather than an absolute 3D world scale. This ensures that
primitives with the same 2D footprint—such as an object
n times larger and n times farther away—are represented
by a consistent, depth-independent scale parameter.
• orientations q ∈R4, parametrized as quaternions.
• opacity α ∈[0, 1], obtained from a sigmoid activation.
• confidence c ∈[0, 1], obtained from a sigmoid activation
and used during multi-view fusion to select primitives.
Multi-view Aggregation. We build the final 3D Gaussian
model by transforming the Gaussian centers m and orien-
tations q from camera to world using predicted extrinsics
parameters. Naively gathering primitives from all images
leads to redundant Gaussians representing the same content
multiple times, leading to blurriness. To address this, we
multiply opacities α by confidences c, giving the model the
ability to prune primitives that would deteriorate the model.
4

<!-- page 5 -->
Medium density (32)
High density (64)
14
14
Low density (16)
0.0
Max
Figure 4.
Spatial distribution of detection across image
patches. We observe the distribution of our heatmaps H that are
used to compute the detected Gaussians. For each density level,
we display the average activation of each channel. Most Gaus-
sians appear to operate on a local area of the patch, especially at
low density. At high density, some channels are specialized for
borders or corners, some others have a widespread distribution en-
abling to be allocated dynamically to highly detailed areas.
We observe, as shown in Figure 5, that the model implicitly
learns multi-view reasoning by setting low confidence val-
ues to primitives that are observed better in other images. At
test-time, we prune primitives for which αc < 0.1, enabling
further computational efficiency.
Image Rendering. We use a customized 3DGS rasterizer
able to render depth and normals (direction of primitives
shortest axis), similar to RaDeGS [69], that also supports
backward pass for camera pose parameters. During train-
ing, we render each image i two times, once using only
primitives predicted from i (“self-rendering”), and another
with the full model. Self-rendering helps to guide the detec-
tion process and also enables the model to learn confidence.
3.3. Training Procedure
Our method is trained from images only without any 3D
annotation. Each training step processes a batch of scenes
with a varying number of input images (2 to 12). We de-
scribe the loss function we use below. More details on train-
ing implementation are given in supplementary materials.
Photometric Losses. Our model is trained to render pho-
torealistic images with common photometric supervision
losses L1, SSIM [63] and LPIPS [71]. Notably, our sys-
tem does not require held-out target views as we only ren-
der input images. This design is usually avoided because it
can lead to a collapsed geometry, but we observe that our
teacher geometry losses prevent this problem.
Geometry Consistency Losses. We enforce our 3D Gaus-
sian model and the 3D reconstruction backbone to be con-
sistent with each other. We first define a L1 loss between the
Inputs
Conf
8 images
Conf
4 images
Conf
2 images
Gaussian model (8 images)
Conf
High
Low
Figure 5. Confidence maps depending on number of views. Our
model shows multi-view awareness when predicting confidence,
removing primitives which are better observed in other views. In
the example, one face of the cube is viewed from the side in image
4 and from the front in image 6. When the model sees image 6,
Gaussians from image 4 are discarded. Green is high confidence.
predicted and rendered depth maps Ldepth. Then, we apply
a second-order constraint Lnormal by deriving normal maps
from the predicted depth maps and intrinsics [69]. This nor-
mal map is compared to rendered normals, defined as the
direction of the Gaussians’ shortest axis. This encourages
Gaussian orientations to align with local surface geome-
try. These consistency losses supervise the Gaussian model
geometry, penalizing primitives that reproject wrongly in
other images and provides supervision signal for the depth
head, pushing it towards multi-view consistent depth esti-
mation, necessary to obtain accurate Gaussian models.
Teacher Geometry Losses. Similar to AnySplat [18], we
observe that fine-tuning solely from photometric loss is not
stable and causes the model to diverge. To regularize it,
we use VGGT [56] as a teacher model and constrain depth
maps and camera poses to remain close to VGGT geome-
try. We define a loss on depth maps Lteachdepth, weighted
by VGGT depth confidence. Then, we use a L1 loss on
camera translation Lteacht and minimize the geodesic dis-
tance between camera orientations LteachR. The objective
here is not to distill information because we start from the
same model, but rather to regularize the problem to avoid
diverging to collapsed geometries.
Regularization Losses.
We observed opaque objects to
be often predicted half-transparent by our model due to
the learning of confidence.
Consequently, we regularize
opacities towards either 0 or 1, using the following loss:
Lop = P
i∈G sin(α(i)·c(i)). Finally, because we train with
video data, all images from the same scene share the same
camera intrinsics. We observe VGGT to produce slightly
inconsistent parameters, so we impose consistency as a soft
constraint with the L2 distance to the average intrinsic (we
minimize variance of intrinsics across the scene).
5

<!-- page 6 -->
AnySplat
Ours
Scene 1
Scene 2
Zoomed in
Zoomed in
Figure 6. Extrapolated novel views. We compare AnySplat and our Gaussian models under highly extrapolated views. AnySplat shows
blurriness and oversmoothed geometry while our model shows clean geometries and renderings. When zoomed in, voxel-aligned Gaussians
appear visibly and fail to fit the details, in contrast with our detected Gaussians. Models created using 12 views from DL3DV-benchmark.
Table 1. Novel view synthesis evaluation of pose-free 3DGS methods. Results averaged over 3, 6, 9 and 12 views. Best , runner-up .
Average
7scenes
Charge
DL3DV
SCRREAM [21]
TanksandTemples
mipnerf360
Models
Decoder
#G / pix
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Ours
Ours
0.1431
21.21
0.6470
0.3532
25.13
0.8119
0.2629
21.93
0.6928
0.3997
20.48
0.6489
0.3163
22.00
0.7171
0.3550
19.37
0.5982
0.3348
18.36
0.4131
0.4503
AnySplat [18]
Voxel
0.8141
17.71
0.5075
0.3937
21.39
0.6906
0.3165
18.28
0.6117
0.4320
17.31
0.4710
0.3611
17.56
0.5510
0.4142
16.28
0.4177
0.3851
15.46
0.3028
0.4529
DA3 GS [30]
Pixel
1
18.83
0.5428
0.3834
19.66
0.7141
0.3267
20.35
0.6386
0.4893
18.46
0.5243
0.3209
19.65
0.6156
0.3656
16.94
0.4365
0.3780
17.95
0.3278
0.4203
4. Experiments
Training.
We train our method on a single GPU with
140 GB of VRAM. Similar to VGGT, each training iteration
processes a maximum of 24 images and uses a varying num-
ber of images per scene, ranging from 2 to 12. We train the
model using monocular video sequences and simply sample
frames linearly with a random step size ranging from 5 to
10. We train on subsets of 10 datasets: DL3DV [31], Co3D-
v2 [42], WildRGBD [64], BlendedMVS [67], Unreal-
Stereo4K [54], Real Estate 10k, ARKitScenes [3], DTU [1]
and ScanNet++ [9] and KITTI360 [29]. Because no an-
notation is needed, the process of integrating more train-
ing datasets is facilitated. Most of this data is also used for
training VGGT [56] and AnySplat [18], which is the most
closely related method to ours.
4.1. Comparison with Pose-free Methods
We compare with the most recent feed-forward approaches
for 3DGS. AnySplat [18] is the closest work to ours that also
fine-tunes VGGT to predict Gaussians. The main difference
is the decoder, where they predict voxel-aligned Gaussians.
DepthAnything3 [30] is another large reconstruction model
more recent than VGGT that also include a pixel-aligned
3D Gaussians decoder. We use DA3-GIANT-1.1, a model
with 1.15B parameters (1B for VGGT) as a baseline.
Evaluation Datasets. We use 6 held-out datasets for evalu-
ation, representing a wide variety of captures and imagery.
DL3DV benchmark [31] contains 140 indoor and outdoor
scenes captured from handheld cameras.
Charge [39]
is a high-fidelity synthetic dataset rendered from Blender
movies which includes ground-truth depth maps and camera
poses. SCRREAM [] 7Scenes [49],Tanks and Temples [26]
and MipNeRF360 [2] are common NVS benchmarks. Our
evaluation uses 3, 6, 9, and 12 views uniformly sampled
from the video sequence, with source and target views in-
terleaved. With the exception of Charge, for which we use
the predefined sparse (3, 6, 9) and dense (25) splits. We re-
size images such that their highest dimension is 518 pixels.
Test-time Alignment. Evaluating pose-free methods for
novel view synthesis is challenging because each method
predicts the Gaussian model and camera poses in its own co-
ordinate system. To render a target view for evaluation, test-
time alignment needs to be performed with reference poses.
Aligning with ground-truth poses with standard Umeyama
alignment [55] is often imprecise in sparse settings because
of noise in the predicted poses. We employ the same strat-
egy as AnySplat: Given N context views and T target views,
we first process our Gaussian model using context views.
Then a second independent forward pass is performed us-
ing N+T views, to obtain target camera poses.
Because
VGGT uses the first image as reference, the two models are
6

<!-- page 7 -->
AnySplat [18]
DA3 Giant [30]
Ours
Ground-Truth
9 views
9 views
3 views
3 views
6 views
25 views
Figure 7. Novel view synthesis of Pose-Free 3DGS methods. Zoom in for details.
aligned up to a single scaling factor (if predictions are con-
sistent). This scale is computed by comparing context poses
and used to adjusts the predicted target pose. In contrast
with prior practice, we do not perform test-time optimiza-
tion of camera parameters through rendering but evaluate
the feed-forward performance.
Novel view synthesis.
Rendering results are evaluated
quantitavely in Table 1, our method significantly outper-
forms AnySplat and DepthAnything3 on all datasets. In
Figure 7, we show qualitative results.
DepthAything3
present detailed gaussian models but suffer from bluri-
ness and half-transparency. This common problem is ad-
dressed by our opacity regularization loss. AnySplat suffer
from geometric inaccuracy, leading to camera misalignment
(camera pose) and duplicate geometry (intrinsics) in some
scenes. Our approach represents the scene faithfully, even
if some details with very thin geometry can disappear. Be-
yond NVS improvements on interpolated views, the mod-
els produced by our method appear ‘cleaner’ under extrap-
olated viewpoints, avoiding scan-line artefacts commonly
observed with pixel-aligned primitives. A visualization is
shown in Fig 6. We provide more visualizations in the sup-
plementary video.
Compression ratio In Table 1, we report the ratio #G/pix
between number of primitives and number of input pixels. Pixel-
aligned approaches present the densest models with a ratio of 1
but are not more accurate. AnySplat performs voxel-based ag-
gregation to reduce the number of primitives, but the decrease is
limited to 19%. Our models achieves a decrease of 86% after
confidence-based pruning, which correspond to 7 times less prim-
itives than pixel-aligned approaches. Note that this ratio can be
controlled by changing parameters of the Adaptive Density mech-
anism. Our compression reduces the number of primitives per im-
age, and could potentially be combined with views aggregation
methods [58].
7

<!-- page 8 -->
Table 2. Geometry evaluation on Charge and SCRREAM datasets.
Charge
SCRREAM
Average
depth
cam pose
FoV
depth
cam pose
FoV
depth
cam pose
FoV
Models
AbsRel ↓
AUC@30 ↑
ang err ↓
AbsRel ↓
AUC@30 ↑
ang err ↓
AbsRel ↓
AUC@30 ↑
ang err ↓
OffTheGrid
0.1609
0.9363
1.47
0.1258
0.9194
0.44
0.1433
0.9278
0.96
DA3-Giant
0.1545
0.9177
6.73
0.1133
0.95
0.59
0.1339
0.9339
3.66
AnySplat
0.1767
0.8821
2.22
0.1414
0.7845
4.23
0.159
0.8333
3.225
VGGT
0.178
0.9012
1.61
0.1201
0.8903
1.37
0.14905
0.8957
1.49
Geometry evaluation We evaluate the geometric accuracy of en-
coders in Table 2 using Charge [39] and SCRREAM [21] datasets.
We report average relative depth error AbsRel, AUC@30 for
camera pose estimation, and field of view angular error angerr for
intrinsics evaluation. OffTheGrid and AnySplat both use VGGT
as encoder, such that reported scores measure the impact of fine-
tuning with photometric loss, that enforces multi-view consistent
geometry. We observe that, on average, our method successfully
improves VGGT for both depth and camera parameters estima-
tion. In contrast, AnySplat fine-tuning degrades the encoder. DA3-
Giant is the best encoder for depth and camera pose estimation but
the focal length estimation is inaccurate, which explain the bluri-
ness observed in Figure 7. Our fine-tuned encoder performs the
best for intrinsics estimation, allowing to unproject primitives to
3D space accurately.
4.2. Ablation Study
Primitives placement We study the impact of our primitive de-
tection technique, i.e. the computation of 2D Gaussians positions
from the DSNT operation, against a pixel-aligned baseline that
compute parameters for each pixel from the UNet output (see
Fig 3). One other existing design, proposed by SplatterImage [52],
is to predict a 3D offset applied to pixel-aligned Gaussians. This
way, primitives are also allowed to be positioned anywhere in
space. We also implement this design to compare it against ours.
Results are reported in Table 3 and can be visualized in Figure 8.
Replacing pixel-aligned by Off-The-Grid provides +0.3dB PSNR,
+0.014 SSIM and a 13% reduction of LPIPS. This difference ap-
pears clearly in renderings that appear sharper and less artefacted.
The baseline with the offset from SplatterImage presents a simi-
lar or degraded rendering quality compared to pixel-aligned, and
visible isolated points artefacts, showing that placing primitives
accurately requires more advanced techniques.
Table 3. Ablation study on primitive placement techniques.
Tanks and Temples
MipNeRF360
Average
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Ours (Full)
19.22
0.6044
0.3327
18.03
0.4385
0.4265
18.63
0.5215
0.3796
Ours (No Self Render)
19.22
0.6074
0.3669
17.79
0.4348
0.4614
18.51
0.5211
0.4142
Pixel Aligned + offset
18.96
0.5810
0.4336
17.69
0.4229
0.5151
18.33
0.5020
0.4744
Pixel Aligned
19.01
0.5877
0.3924
17.65
0.4275
0.4757
18.33
0.5076
0.4341
AnySplat
16.92
0.4919
0.3568
15.51
0.3688
0.4162
16.22
0.4304
0.3865
Adaptive density and Confidence-based pruning We evaluate
our model without the adaptive density mechanism (32 gaus-
sians per patch) and without confidence-based pruning on DL3DV
benchmark [31]. Novel view synthesis metrics are given in Ta-
ble 4. Ablated models perform notably worse, especially in LPIPS.
This is because some highly detailed areas can not be fitted accu-
rately with 32 gaussians only. Discarding confidence-based prun-
ing results in duplicated geometry, bluriness and floating artefacts.
AnySplat
Pixel-Aligned
Pixel+Offset
Ours
Ground-Truth
3 views
6 views
9 views
12 views
Figure 8. Qualitative Ablation Study. Zoom in for details.
Table 4. Ablation on adaptive density and confidence.
Adaptive
Confidence
PSNR
SSIM
LPIPS
✓
×
17.80
0.4223
0.4459
×
✓
18.72
0.5652
0.3898
✓
✓
19.09 0.5998 0.3379
5. Limitations and Future Work
A limitation of our method (and of most other feed-forward ap-
proaches) is that it only reconstructs visible areas, leaving holes
in parts of the scene missed during capture and breaking the pho-
torealistic illusion when rendering novel views (for example, the
missing top face of the cube in Figure 5). One solution could be
to use video diffusion models to produce the final renderings, as
proposed by MVSplat360 [6]. Another could be to learn how to
complete Gaussian models in a 3D inpainting manner. We leave
this problem for future work. Our model currently does not model
view-dependent color variations and thus present reduced quality
in scenes with lightning variations. Color harmonization could be
a solution [48]. Humans are not represented in the training data
and thus our model does not perform well on human subjects, but
robustly handles both indoor, outdoor and object centric scenes.
6. Conclusion
We have introduced Off-The-Grid Gaussians, a novel alternative
to pixel-aligned and voxel-aligned strategies that achieves higher
photorealism while using far fewer primitives. By combining de-
tection with adaptive density, confidence-based pruning, photo-
metric rendering ang geometry consistency losses, we obtain a
model that achieves state-of-the-art accuracy for pose-free feed-
forward 3DGS. Our fine-tuning procedure marginally improves
the geometry accuracy of VGGT. By decoupling the number of
primitives from the number of input pixels, our technique should
open the possibility to process 3DGS models from high resolution
images with feed-forward methods.
8

<!-- page 9 -->
References
[1] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis,
Engin Tola, and Anders Bjorholm Dahl. Large-scale data for
multiple-view stereopsis. International Journal of Computer
Vision, pages 1–16, 2016. 6
[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5470–5479, 2022. 6
[3] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry,
Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe,
Daniel Kurz, Arik Schwartz, et al. ARKitScenes: A Diverse
Real-World Dataset For 3D Indoor Scene Understanding Us-
ing Mobile RGB-D Data. arXiv preprint arXiv:2111.08897,
2021. 6
[4] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent
Sitzmann. pixelSplat: 3D Gaussian Splats from Image Pairs
for Scalable Generalizable 3D Reconstruction. In Computer
Vision and Pattern Recognition Conference (CVPR), 2024.
1, 2
[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. MVSplat: Efficient 3D Gaussian Splatting from Sparse
Multi-View Images. In European Conference on Computer
Vision (ECCV), pages 370–386, 2024. 2
[6] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang,
Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai.
MVS-
plat360: Feed-Forward 360 Scene Synthesis from Sparse
Views. In Conference on Neural Information Processing Sys-
tems, 2024. 8
[7] Youyu Chen, Junjun Jiang, Kui Jiang, Xiao Tang, Zhihao Li,
Xianming Liu, and Yinyu Nie. DashGaussian: Optimizing
3D Gaussian Splatting in 200 Seconds. In Computer Vision
and Pattern Recognition Conference (CVPR), 2025. 2
[8] Rohan Choudhury, JungEun Kim, Jinhyung Park, Eunho
Yang, L´aszl´o A Jeni, and Kris M Kitani. Accelerating Vi-
sion Transformers with Adaptive Patch Sizes. arXiv preprint
arXiv:2510.18091, 2025. 4
[9] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Niessner. ScanNet:
Richly-Annotated 3D Reconstructions of Indoor Scenes.
In Computer Vision and Pattern Recognition Conference
(CVPR), 2017. 6
[10] Xiaobin Deng, Changyu Diao, Min Li, Ruohan Yu, and Du-
anqing Xu. Improving Densification in 3D Gaussian Splat-
ting for High-Fidelity Rendering. arXiv:2508.12313, 2025.
2
[11] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabi-
novich. SuperPoint: Self-Supervised Interest Point Detec-
tion and Description. In Computer Vision and Pattern Recog-
nition Conference (CVPR), 2018. 2
[12] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang,
Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic,
Marco Pavone, Georgios Pavlakos, et al.
InstantSplat:
Sparse-view Gaussian Splatting in Seconds. arXiv preprint
arXiv:2403.20309, 2024. 3
[13] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A
Efros, and Xiaolong Wang.
COLMAP-Free 3D Gaussian
Splatting. In Computer Vision and Pattern Recognition Con-
ference (CVPR), 2024. 3
[14] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han,
Jiaolong Yang, Chong Luo, and Seungryong Kim. PF3plat:
Pose-Free Feed-Forward 3D Gaussian Splatting.
arXiv
preprint arXiv:2410.22128, 2024. 3
[15] Ranran Huang and Krystian Mikolajczyk.
No Pose at
All: Self-Supervised Pose-Free 3D Gaussian Splatting from
Sparse Views. In Computer Vision and Pattern Recognition
Conference (CVPR), 2025. 1, 3
[16] Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao,
Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing
Huang, and Georgios Pavlakos. Rayzer: A self-supervised
large view synthesis model. In International Conference on
Computer Vision (ICCV), 2025. 2
[17] Kaiwen Jiang, Yang Fu, Mukund Varma T, Yash Belhe, Xi-
aolong Wang, Hao Su, and Ravi Ramamoorthi. A Construct-
Optimize Approach to Sparse View Synthesis without Cam-
era Pose. In ACM SIGGRAPH Conference Papers, 2024. 3
[18] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui
Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang,
Feng Zhao, et al.
AnySplat:
Feed-forward 3D Gaus-
sian Splatting from Unconstrained Views.
arXiv preprint
arXiv:2505.23716, 2025. 1, 2, 3, 5, 6, 7
[19] Yuheng Jiang, Zhehao Shen, Penghao Wang, Zhuo Su, Yu
Hong, Yingliang Zhang, Jingyi Yu, and Lan Xu. HiFi4G:
High-Fidelity Human Performance Rendering via Compact
Gaussian Splatting. In Computer Vision and Pattern Recog-
nition Conference (CVPR), pages 19734–19745, 2024. 1
[20] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi,
Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang
Xu. LVSM: A Large View Synthesis Model with Minimal
3D Inductive Bias. In International Conference on Learning
Representations (ICLR), 2025. 2
[21] HyunJun Jung, Weihang Li, Shun-Cheng Wu, William Bit-
tner, Nikolas Brasch, Jifei Song, Eduardo P´erez-Pellitero,
Zhensong Zhang, Arthur Moreau, Nassir Navab, et al. Scr-
ream: Scan, register, render and map: A framework for an-
notating accurate and dense 3d indoor scenes with a bench-
mark. Advances in Neural Information Processing Systems,
37:44164–44176, 2024. 6, 8
[22] Gyeongjin Kang, Jisang Yoo, Jihyeon Park, Seungtae Nam,
Hyeonsoo Im, Sangheon Shin, Sangpil Kim, and Eunbyung
Park.
SelfSplat: Pose-Free and 3D Prior-Free Generaliz-
able 3D Gaussian Splatting. In Computer Vision and Pattern
Recognition Conference (CVPR), 2025. 1
[23] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger,
Lorenzo Porzi,
Yuchen Zhang,
Tobias Fischer,
Arno
Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes,
Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota
Bul`o, Christian Richardt, Deva Ramanan, Sebastian Scherer,
and Peter Kontschieder.
MapAnything: Universal feed-
forward metric 3D reconstruction.
In arXiv:2509.13414,
2025. 3
[24] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3D Gaussian Splatting for Real-Time
9

<!-- page 10 -->
Radiance Field Rendering. ACM Transactions on Graphics,
42(4), 2023. 1, 2
[25] Sieun Kim, Kyungjin Lee, and Youngki Lee.
Color-cued
Efficient Densification Method for 3D Gaussian Splatting.
In Computer Vision and Pattern Recognition Conference
(CVPR), 2024. 2
[26] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun. Tanks and temples: Benchmarking large-scale scene
reconstruction. ACM Transactions on Graphics, 36(4), 2017.
6
[27] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Ground-
ing Image Matching in 3D with MASt3R. In European Con-
ference on Computer Vision (ECCV), 2024. 3
[28] Zhiqi Li, Chengrui Dong, Yiming Chen, Zhangchi Huang,
and Peidong Liu. VicaSplat: A Single Run is All You Need
for 3D Gaussian Splatting and Camera Estimation from Un-
posed Video Frames.
arXiv preprint arXiv:2503.10286,
2025. 3
[29] Yiyi Liao, Jun Xie, and Andreas Geiger. KITTI-360: A novel
dataset and benchmarks for urban scene understanding in 2d
and 3d. Pattern Analysis and Machine Intelligence (PAMI),
2022. 6
[30] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen,
Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth
anything 3: Recovering the visual space from any views.
arXiv preprint arXiv:2511.10647, 2025. 6, 7
[31] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin,
Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,
et al. DL3DV-10K: A Large-Scale Scene Dataset for Deep
Learning-based 3D Vision. In Computer Vision and Pattern
Recognition Conference (CVPR), 2024. 6, 8
[32] Yanzhe Lyu, Kai Cheng, Xin Kang, and Xuejin Chen.
ResGS: Residual Densification of 3D Gaussian for Efficient
Detail Recovery. In Computer Vision and Pattern Recogni-
tion Conference (CVPR), 2025. 2
[33] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl,
Markus Steinberger, Francisco Vicente Carrasco, and Fer-
nando De La Torre. Taming 3DGS: High-quality radiance
fields with limited resources. In SIGGRAPH Asia Confer-
ence Papers, 2024. 2
[34] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and An-
drew J. Davison. Gaussian Splatting SLAM. Computer Vi-
sion and Pattern Recognition Conference (CVPR), 2024. 1
[35] Andreas Meuleman, Ishaan Shah, Alexandre Lanvin, Bern-
hard Kerbl, and George Drettakis. On-the-fly Reconstruc-
tion for Large-Scale Novel View Synthesis from Unposed
Images. ACM Transactions on Graphics (TOG), 44(4):1–14,
2025. 2
[36] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
Representing Scenes as Neural Radiance Fields for View
Synthesis.
Communications of the ACM, 65(1):99–106,
2021. 2
[37] Arthur Moreau, Jifei Song, Helisa Dhamo, Richard Shaw,
Yiren Zhou, and Eduardo P´erez-Pellitero.
Human Gaus-
sian Splatting: Real-time Rendering of Animatable Avatars.
In Computer Vision and Pattern Recognition Conference
(CVPR), 2024. 1
[38] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant Neural Graphics Primitives with a Mul-
tiresolution Hash Encoding. ACM Transactions on Graphics
(TOG), 41(4):102:1–102:15, 2022. 1
[39] Michal Nazarczuk, Thomas Tanay, Arthur Moreau, Zhen-
song Zhang, and Eduardo P´erez-Pellitero. Charge: A com-
prehensive novel view synthesis benchmark and dataset to
bind them all. In Computer Vision and Pattern Recognition
Conference (CVPR), 2026. 6, 8
[40] Aiden Nibali, Zhen He, Stuart Morgan, and Luke Prender-
gast. Numerical Coordinate Regression with Convolutional
Neural Networks. arXiv preprint arXiv:1801.07372, 2018.
2, 4
[41] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy V.
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido
Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes,
Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rab-
bat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou,
Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bo-
janowski. DINOv2: Learning Robust Visual Features with-
out Supervision.
Transactions on Machine Learning Re-
search, 2024. 3
[42] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler,
Luca Sbordone, Patrick Labatut, and David Novotny. Com-
mon Objects in 3D: Large-Scale Learning and Evaluation of
Real-life 3D Category Reconstruction. In Computer Vision
and Pattern Recognition Conference (CVPR), 2021. 6
[43] Shiwei Ren, Tianci Wen, Yongchun Fang, and Biao Lu.
FastGS: Training 3D Gaussian Splatting in 100 Seconds.
arXiv:2511.04283, 2025. 2
[44] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-
Net: Convolutional networks for biomedical image segmen-
tation. In International Conference on Medical image com-
puting and computer-assisted intervention, 2015. 3
[45] Samuel Rota Bul`o, Lorenzo Porzi, and Peter Kontschieder.
Revising Densification in Gaussian Splatting. In European
Conference on Computer Vision (ECCV), 2024. 2
[46] Johannes
Lutz
Sch¨onberger
and
Jan-Michael
Frahm.
Structure-from-Motion Revisited. In Computer Vision and
Pattern Recognition Conference (CVPR), 2016. 1, 2, 3
[47] Richard Shaw, Youngkyoon Jang, Athanasios Papaioannou,
Arthur Moreau, Helisa Dhamo, Zhensong Zhang, and Ed-
uardo P´erez-Pellitero. Ico3d: An interactive conversational
3d virtual human. International Journal of Computer Vision,
134(4):161, 2026. 1
[48] Jisu Shin, Richard Shaw, Seunghyun Shin, Zhensong Zhang,
Hae-Gon Jeon, and Eduardo Perez-Pellitero. Chroma: Con-
sistent harmonization of multi-view appearance via bilateral
grid prediction.
In International Conference on Learning
Representations (ICLR), 2026. 8
[49] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram
Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene co-
ordinate regression forests for camera relocalization in rgb-d
images. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 2930–2937, 2013. 6
[50] Mile Sial.
Pytorch-UNet:
PyTorch implementation of
the U-Net for image semantic segmentation.
https:
10

<!-- page 11 -->
//github.com/milesial/Pytorch-UNet/, 2025.
Accessed: 2025-11-20. 1
[51] Brandon Smart, Chuanxia Zheng, Iro Laina, and Vic-
tor Adrian Prisacariu. Splatt3R: Zero-shot Gaussian Splat-
ting from Uncalibrated Image Pairs.
arXiv preprint
arXiv:2408.13912, 2024. 3
[52] Stanislaw Szymanowicz, Christian Rupprecht, and Andrea
Vedaldi. Splatter Image: Ultra-Fast Single-View 3D Recon-
struction. Computer Vision and Pattern Recognition Confer-
ence (CVPR), 2024. 2, 8
[53] Thomas Tanay and Matteo Maggioni. Global Latent Neu-
ral Rendering. In Computer Vision and Pattern Recognition
Conference (CVPR), 2024. 2
[54] Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger.
SMD-Nets: Stereo Mixture Density Networks. In Computer
Vision and Pattern Recognition Conference (CVPR), 2021. 6
[55] Shinji Umeyama. Least-squares estimation of transformation
parameters between two point patterns. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 13(4):376–
380, 1991. 6
[56] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. VGGT:
Visual Geometry Grounded Transformer. In Computer Vi-
sion and Pattern Recognition Conference (CVPR), 2025. 2,
3, 5, 6, 1
[57] Shuzhe Wang,
Vincent Leroy,
Yohann Cabon,
Boris
Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D
Vision Made Easy. In Computer Vision and Pattern Recog-
nition Conference (CVPR), 2024. 3
[58] Weijie Wang, Donny Y Chen, Zeyu Zhang, Duochao Shi,
Akide Liu, and Bohan Zhuang. Zpressor: Bottleneck-aware
compression for scalable feed-forward 3dgs. arXiv preprint
arXiv:2505.23734, 2025. 7
[59] Weijie Wang, Yeqing Chen, Zeyu Zhang, Hengyu Liu,
Haoxiao Wang, Zhiyuan Feng, Wenkang Qin, Zheng Zhu,
Donny Y Chen, and Bohan Zhuang. VolSplat: Rethinking
Feed-Forward 3D Gaussian Splatting with Voxel-Aligned
Prediction. arXiv preprint arXiv:2509.19297, 2025. 2
[60] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee
Lee. FreeSplat: Generalizable 3D Gaussian Splatting To-
wards Free-View Synthesis of Indoor Scenes. In Conference
on Neural Information Processing Systems, 2024. 2
[61] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee
Lee.
FreeSplat++: Generalizable 3D Gaussian Splatting
for Efficient Indoor Scene Reconstruction. arXiv preprint
arXiv:2503.22986, 2025. 2
[62] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang,
Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua
Shen, and Tong He. π3: Permutation-Equivariant Visual Ge-
ometry Learning. arXiv preprint arXiv:2507.13347, 2025.
3
[63] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE Transactions on Image Process-
ing, 13(4):600–612, 2004. 5
[64] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. Rgbd
objects in the wild: Scaling real-world 3d object learning
from rgb-d videos. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
22378–22389, 2024. 6
[65] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Polle-
feys. Depthsplat: Connecting gaussian splatting and depth.
In Computer Vision and Pattern Recognition Conference
(CVPR), 2025. 2, 1
[66] Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard
Newcombe, and Zhaoyang Lv.
4DGT: Learning a 4D
Gaussian Transformer Using Real-World Monocular Videos.
In Conference on Neural Information Processing Systems,
2025. 2
[67] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren,
Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-
scale dataset for generalized multi-view stereo networks. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 1790–1799, 2020. 6
[68] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys,
Ming-Hsuan Yang, and Songyou Peng. No Pose, No Prob-
lem: Surprisingly Simple 3D Gaussian Splats from Sparse
Unposed Images. arXiv preprint arXiv:2410.24207, 2024.
1, 3
[69] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang,
Xiaoxiao Long, and Ping Tan. RaDe-GS: Rasterizing Depth
in Gaussian Splatting.
arXiv preprint arXiv:2406.01467,
2024. 5
[70] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao,
Kalyan Sunkavalli, and Zexiang Xu. GS-LRM: Large Re-
construction Model for 3D Gaussian Splatting.
European
Conference on Computer Vision (ECCV), 2024. 2
[71] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The Unreasonable Effectiveness of Deep
Features as a Perceptual Metric. In Computer Vision and
Pattern Recognition Conference (CVPR), 2018. 5
[72] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue,
Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gor-
don Wetzstein.
FLARE: Feed-forward Geometry, Ap-
pearance and Camera Estimation from Uncalibrated Sparse
Views. In Computer Vision and Pattern Recognition Confer-
ence (CVPR), 2025. 3
11

<!-- page 12 -->
Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting
Supplementary Material
7. Supplementary video
We present novel view synthesis comparison with AnySplat [18]
in the supplementary video. Trajectories include both interpolated
and extrapolated views. We show models that use 6 views and
generate 30 views interpolated between each. In the middle of
the generated video, we extrapolate views with a spiral trajectory.
Please note that despite using the same code to generate trajec-
tories for both methods, views are not exactly aligned due to the
difference in camera poses. Our method exhibits more accurate
geometry and sharper rendering in most scenes.
8. Implementation details on our method
For the 3D reconstruction backbone, we use VGGT-1B [56], start-
ing from official checkpoints released by the authors. We remove
the pointmap head that we don’t use. We tried to use it instead
of depth map but observed significantly inferior accuracy, espe-
cially for large number of images. Regarding the decoder, the
UNet module uses the implementation of Pytorch-UNet [50] with
13 input channels (3 for RGB, 2 for depth and depth confidence,
and 8 for unpatchified latent features and 32 output channels. The
hdet and hdesc heads process features with input image resolution
through 3 convolutional layers with ReLU intermediate activations
and 32 channels. Detection features are transformed into heatmaps
through a softmax with temperature 0.2. Importantly, we remind
that softmax is not applied over channels dimension but over spa-
tial dimension at a patch level.
Bilinear interpolation is done
with Pytorch grid sample function with padding mode set
to border.
9. Depth estimation
We present a qualitative evaluation of depth, shown in Fig. 9. The
first two rows show examples from Charge, whereas next rows are
from DL3DV where ground truth depth is not available.
First, on Charge, we observe accurate and highly similar depth
maps between pre-trained VGGT and our fine-tuned version on
this synthetic dataset. The main difference is observed on back-
ground areas (see second row, where our method is better aligned
with GT for foreground areas but background is predicted too
close, degrading metrics). On DL3DV, we observe more insights
and failure cases from the pre-trained VGGT model. First, this
model is quite sensitive to specularities and create holes on flat but
highly reflective surfaces (see rows 3, 5 and 6). Then, during the
supervised training of this model, sky pixels were masked out, re-
sulting in close depth estimation for these pixels (row 4), which
is not compatible with our rendering task. We also sometimes ob-
serve inaccuracy on some flat surfaces (e.g. the ceiling in row 7)
without clear reasons. All these failures create geometrically in-
accurate models with floating artefacts when we start to train our
method. We observe that self-supervised fine-tuning through ren-
dering is able to address theses issues and obtain more accurate
depth maps without holes, for both our method and AnySplat [18],
to a lesser degree. We observe that AnySplat depth maps are less
accurate than ours, one recurrent artefact being edges appearing
where depth is continuous (see row 4). DepthSplat [65] also claims
to learn a depth estimation module from rendering but its accuracy
is not comparable with VGGT-based models.
1

<!-- page 13 -->
Input Image
Ground Truth
VGGT [56]
AnySplat [18]
DepthSplat [65]
Ours
Figure 9. Qualitative results of our method compared with several SoTA on depth estimation.
2
