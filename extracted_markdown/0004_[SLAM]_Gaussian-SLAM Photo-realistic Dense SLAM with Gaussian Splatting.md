<!-- page 1 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
with Gaussian Splatting
Vladimir Yugay1, Yue Li1, Theo Gevers1, and Martin R. Oswald1
University of Amsterdam, Netherlands
https://vladimiryugay.github.io/gaussian_slam
Abstract. We present a dense simultaneous localization and mapping
(SLAM) method that uses 3D Gaussians as a scene representation. Our
approach enables interactive-time reconstruction and photo-realistic ren-
dering from real-world single-camera RGBD videos. To this end, we pro-
pose a novel effective strategy for seeding new Gaussians for newly ex-
plored areas and their effective online optimization that is independent
of the scene size and thus scalable to larger scenes. This is achieved by
organizing the scene into sub-maps which are independently optimized
and do not need to be kept in memory. We further accomplish frame-to-
model camera tracking by minimizing photometric and geometric losses
between the input and rendered frames. The Gaussian representation
allows for high-quality photo-realistic real-time rendering of real-world
scenes. Evaluation on synthetic and real-world datasets demonstrates
competitive or superior performance in mapping, tracking, and render-
ing compared to existing neural dense SLAM methods.
1
Introduction
Simultaneous localization and mapping (SLAM) has been an active research
topic for the past two decades [15, 22]. A major byproduct of that journey is
the investigation of various scene representations to either push the tracking
performance and mapping capabilities or to adapt it for more complex down-
stream tasks like path planning or semantic understanding. Specifically, earlier
works focus on tracking using various scene representations like feature point
clouds [14, 26, 42], surfels [56, 72], depth maps [45, 61], or implicit representa-
tions [13,44,46]. Later works focus more on the map quality and density. With
the advent of powerful neural scene representations like neural radiance fields [40]
that allow for high fidelity view-synthesis, a rapidly growing body of dense neural
SLAM methods [18,34,53,63,65,66,85,89] has been developed. Despite their im-
pressive gains in scene representation quality, these methods are still limited to
small synthetic scenes and their re-rendering results are far from photo-realistic.
Recently, a novel scene representation based on Gaussian splatting [25] has
been shown to deliver on-par rendering performance with NeRFs while being an
order of magnitude faster in rendering and optimization. Moreover, this scene
representation is directly interpretable and can be directly manipulated which
is desirable for many downstream tasks. With these advantages, the Gaussian
arXiv:2312.10070v2  [cs.CV]  22 Mar 2024

<!-- page 2 -->
2
V. Yugay et al.
splatting representation lends itself to be applied in an online SLAM system
with real-time demands and opens the door to photo-realistic dense SLAM.
ESLAM [34]
Point-SLAM [53]
Gaussian-SLAM (Ours)
Ground Truth
Fig. 1: Rendering Results of Gaussian-SLAM. Embedded into a dense SLAM
pipeline, the 3D Gaussian-based scene representation allows for fast, photo-realistic
rendering of scene views. This leads to high-quality rendering, especially on real-world
data like this TUM-RGBD [62] frame that contains many high-frequency details that
other methods struggle to capture.
In this paper, we introduce Gaussian-SLAM, a dense RGBD SLAM system
using 3D Gaussians to build a scene representation that allows for mapping,
tracking, and photo-realistic re-rendering at interactive runtimes. An example
of the high-fidelity rendering output of Gaussian-SLAM is depicted in Fig. 1. In
summary, our contributions include:
• A dense RGBD SLAM approach that uses 3D Gaussians to construct a scene
representation allowing SOTA rendering results on real-world scenes.
• An extension of Gaussian splatting that better encodes geometry and allows
reconstruction beyond radiance fields in a single-camera setup.
• An online optimization method for Gaussian splats that processes the map as
sub-maps and introduces efficient seeding and optimization strategies.
• A frame-to-model tracker with the Gaussian splatting scene representation via
photometric and geometric error minimization.
All source code and data will be made publicly available.

<!-- page 3 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
3
2
Related Work
Dense Visual SLAM and Online Mapping. The seminal work of Curless
and Levoy [11] set the stage for a variety of 3D reconstruction methods using
truncated signed distance functions (TSDF). A line of works was built upon it
improving speed [44] through efficient implementation and volume integration,
scalability through voxel hashing [20,46,48] and octree data structure [57], as well
as tracking with sparse image features [4] and loop closures [5,13,45,56]. Tackling
the problem of unreliable depth maps, RoutedFusion [70] introduced a learning-
based fusion network for updating the TSDF in volumetric grids. This concept
was further evolved by NeuralFusion [71] and DI-Fusion [18], which adopt im-
plicit learning for scene representation, enhancing their robustness against out-
liers. Recent research has successfully achieved dense online reconstruction using
solely RGB cameras [3,8,27,43,55,58,64], bypassing the need for depth data.
Recently, test-time optimization methods have become popular due to their
ability to adapt to unseen scenes on the fly. Continuous Neural Mapping [77],
for instance, employs a continual mapping strategy from a series of depth maps
to learn scene representation. Inspired by Neural Radiance Fields [40], there has
been immense progress in dense surface reconstruction [47,67] and accurate pose
estimation [2, 29, 52, 69]. These efforts have led to the development of compre-
hensive dense SLAM systems [34, 54, 63, 78, 85, 88, 89], showing a trend in the
pursuit of precise and reliable visual SLAM. A comprehensive survey on online
RGBD reconstruction can be found in [90].
While the latest neural methods show impressive rendering capabilities on
synthetic data, they struggle when applied to real-world data. Further, these
methods are not yet practical for real-world applications due to computation
requirements, slow speed, and the inability to effectively incorporate pose up-
dates, as the neural representations rely on positional encoding. In contrast,
our method shows impressive performance on real-world data, has competitive
tracking and runtime, and uses a scene representation that naturally allows pose
updates.
Scene Representations for SLAM. The majority of dense 3D scene represen-
tations for SLAM are grid-based, point-based, network-based, or hybrid. Among
these, grid-based techniques are perhaps the most extensively researched. They
further divide into methods using dense grids [3,9,11,28,44,64,70–73,86,87,89],
hierarchical octrees [6, 30, 31, 36, 57, 78] and voxel hashing [13, 20, 41, 46, 67] for
efficient memory management. Grids offer the advantage of simple and quick
neighborhood lookups and context integration. However, a key limitation is the
need to predefine grid resolution, which is not easily adjustable during recon-
struction. This can result in inefficient memory usage in empty areas while failing
to capture finer details due to resolution constraints.
Point-based approaches address some of the grid-related challenges and have
been effectively utilized in 3D reconstruction [5, 7, 10, 21, 24, 56, 72, 83]. Unlike
grid resolution, the density of points in these methods does not have to be
predetermined and can naturally vary throughout the scene. Moreover, point

<!-- page 4 -->
4
V. Yugay et al.
sets can be efficiently concentrated around surfaces, not spending memory on
modeling empty space. The trade-off for this adaptability is the complexity of
finding neighboring points, as point sets lack structured connectivity. In dense
SLAM, this challenge can be mitigated by transforming the 3D neighborhood
search into a 2D problem via projection onto keyframes [56,72], or by organizing
points within a grid structure for expedited searching [75].
Network-based methods for dense 3D reconstruction provide a continuous
scene representation by implicitly modeling it with coordinate-based networks [1,
27,38,49,63,66,67,77,79,85]. This representation can capture high-quality maps
and textures. However, they are generally unsuitable for online scene recon-
struction due to their inability to update local scene regions and to scale for
larger scenes. More recently, a hybrid representation combining the advantages
of point-based and neural-based was proposed [53]. While addressing some of
the issues of both representations it struggles with real-world scenes, and cannot
seamlessly integrate trajectory updates in the scene representation.
Outside these three primary categories, some studies have explored alterna-
tive representations like surfels [16,39] and neural planes [34,51]. Parameterized
surface elements are generally not great at modeling a flexible shape template
while feature planes struggle with scene reconstructions containing multiple sur-
faces, due to their overly compressed representation. Recently, Kerbl et al. [25]
proposed to represent a scene with 3D Gaussians. The Gaussian parameters are
optimized via differential rendering with multi-view supervision. While being
very efficient and achieving impressive rendering results, this representation is
tailored for fully-observed multi-view environments and does not encode geome-
try well. Concurrent to our work, [74,80,81] focus on dynamic scene reconstruc-
tion, and [33] on tracking. However, they are all offline methods and do not suit
single-camera dense SLAM setups.
Concurrently, several methods [17, 23, 37, 76] have used Gaussian Splatting
[25] for SLAM. While most splatting-based methods use gradient-based map
densification similar to [25], we follow a more controlled approach with exact
thresholding by utilizing fast nearest-neighbor search and alpha masking. Fur-
ther, unlike all other concurrent work, our mapping pipeline does not require
holding all the 3D Gaussians in the GPU memory, allowing our method to
scale and not slow down as more areas are covered. Moreover, while in con-
current works the 3D Gaussians are very densely seeded, our color gradient and
masking-based seeding strategy allows for sparser seeding while preserving SOTA
rendering quality. Finally, in contrast to [17, 37, 76], our tracking does not rely
on explicitly computed camera pose derivatives and is implemented in PyTorch.
3
Method
The key idea of our approach is to construct a map using 3D Gaussians [25]
as a main building block to make single-camera RGBD neural SLAM scalable,
faster and achieve better rendering on real-world datasets. We introduce a novel
efficient mapping process with bounded computational cost in a sequential single-

<!-- page 5 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
5
Gaussian Point Cloud
Depth and Color Losses
Active Sub-map
Inactive Sub-maps
Active Sub-map Keyframes
Rendered Keyframes
Differential 
Rasterization
Parameters Update
RGBD Input Keyframe
Estimate Camera Pose
New Gaussians
Add New Gaussians
Optimize Active Sub-map
Track Pose and Grow Active Sub-map
Fig. 2: Gaussian-SLAM Architecture. For every input keyframe the camera pose is
estimated using depth and color losses against the active sub-map. Given an estimated
camera pose, the RGBD frame is transformed into 3D and subsampled based on color
gradient and the rendered alpha mask. Points from the subsampled point clouds located
in low-density areas of the active sub-map are used to initialize new 3D Gaussians.
These sparse 3D Gaussians are then added to the Gaussian point cloud of the active
sub-map and are jointly optimized with the depth maps and color images from all
contributing keyframes of this sub-map.
camera setup, a challenging scenario for traditional 3D Gaussian Splatting. To
enable traditional Gaussian splats to render accurate geometry we extend them
by adding a differential depth rendering, explicitly computing gradients for the
Gaussian parameters updates. Finally, we develop a novel frame-to-model track-
ing approach relying on our 3D map representation. Fig. 2 provides an overview
of our method. We now explain our pipeline, starting with an overview of classical
Gaussian splatting [25], and continuing with map construction and optimization,
geometry encoding, and tracking.
3.1
Gaussian Splatting
Gaussian splatting [25] is an effective method for representing 3D scenes with
novel-view synthesis capability. This approach is notable for its speed, without
compromising the rendering quality. In [25], 3D Gaussians are initialized from
a sparse Structure-from-Motion point cloud of a scene. With images observing
the scene from different angles, the Gaussian parameters are optimized using
differentiable rendering. During training, 3D Gaussians are adaptively added or
removed to better render the images based on a set of heuristics.
A single 3D Gaussian is parameterized by mean µ ∈R3, covariance Σ ∈R3×3,
opacity o ∈R, and RGB color C ∈R3. The mean of a projected (splatted) 3D
Gaussian in the 2D image plane µI is computed as follows:
  \ m
u
 ^{I} = \pi \big (
P
(T_{wc} \mu _\text {homogeneous} )\big ) \enspace , 
(1)
where Twc ∈SE(3) is the world-to-camera transformation, P ∈R4×4 is an
OpenGL-style projection matrix, π : R4 →R2 is a projection to pixel coordi-
nates. The 2D covariance ΣI of a splatted Gaussian is computed as:
  \ Sigma ^
{I} = J R_{wc} \Sigma R_{wc}^T J^T \enspace , 
(2)

<!-- page 6 -->
6
V. Yugay et al.
where J ∈R2×3 is an affine transformation from [91], Rwc ∈SO(3) is the
rotation component of world-to-camera transformation Twc. We refer to [91] for
further details about the projection matrices. Color C along one channel ch at
a pixel i influenced by m ordered Gaussians is rendered as:
  C
^
{
c
h}_
{i}
 
=  \ s um  _{j \l e
q
 m}
C^ { ch} _{j} \cdot \alpha _{j} \cdot T_{j} \enspace , \;\text {with }\; T_{j} = \prod _{k < j}(1 - \alpha _{k}) \enspace , \label {eq:color_blending}
(3)
with αj is computed as:
  \ al p ha _{j} 
= o
_{ j }
 \c
d ot \
e
xp  (-\sigma _{j}) \quad \text { and }\quad \sigma _{j} = \frac {1}{2} \Delta _{j}^T \Sigma _{j}^{I-1} \Delta _{j} \enspace , 
(4)
where ∆j ∈R2 is the offset between the pixel coordinates and the 2D mean of a
splatted Gaussian. The parameters of the 3D Gaussians are iteratively optimized
by minimizing the photometric loss between rendered and training images. Dur-
ing optimization, C is encoded with spherical harmonics SH ∈R15 to account
for direction-based color variations. Covariance is decomposed as Σ = RSST RT ,
where R ∈R3×3 and S = diag(s) ∈R3×3 are rotation and scale respectively to
preserve covariance positive semi-definite property during gradient-based opti-
mization.
3.2
3D Gaussian-based Map
To avoid catastrophic forgetting and overfitting and make the mapping compu-
tationally feasible in a single-camera stream scenario we process the input in
chunks (sub-maps). Every sub-map covers several keyframes observing it and
is represented with a separate 3D Gaussian point cloud. Formally, we define a
sub-map Gaussian point cloud P s as a collection of N 3D Gaussians:
  P ^s = 
\{ G(
\ m u 
^{ s}
_ { i}, \Sigma ^{s}_{i}, o^{s}_{i}, C^{s}_{i}) \, | \, i=1,\ldots ,N\} \enspace . i
(5)
Sub-map Initialization. A sub-map starts with the first frame and grows
incrementally with newly incoming keyframes. As the explored area grows, a new
sub-map is needed to cover the unseen regions and avoid storing all the Gaussians
in GPU memory. Instead of using a fixed interval when creating a new sub-
map [9,13,35], an initialization strategy that relies on the camera motion [5,60]
is used. Specifically, a new sub-map is created when the current frame’s estimated
translation relative to the first frame of the active sub-map exceeds a predefined
threshold, dthre, or when the estimated Euler angle surpasses θthre. At any time,
only active sub-map is processed. This approach bounds the compute cost and
ensures that optimization remains fast while exploring larger scenes.
Sub-map Building. Every new keyframe may add 3D Gaussians to the active
sub-map to account for the newly observed parts of the scene. Following the
pose estimation for the current keyframe, a dense point cloud is computed from
keyframe RGBD measurements. At the beginning of each sub-map, we sample
Mu uniformly and Mc points from the keyframe point cloud in high color gradient

<!-- page 7 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
7
regions to add new Gaussians. For the following keyframes of the sub-map, we
sample Mk points uniformly from the regions with the rendered alpha values
lower than a threshold αn. This allows for growing the map in areas sparsely
covered by the 3D Gaussians. New Gaussians are added to the sub-map using
sampled points that have no neighbors within a search radius ρ in the current
sub-map. The new Gaussians are anisotropic and their scales are defined based
on the nearest neighbor distance within the active sub-map. This densification
strategy substantially differs from [25] where new Gaussians were added and
pruned based on the gradient values during optimization and gives fine-grained
control over the number of Gaussians.
Sub-map Optimization. All Gaussians in the active sub-map are jointly op-
timized every time new Gaussians are added to the sub-map for a fixed number
of iterations minimizing the loss (12). We do not clone or prune the Gaussians
as done in [25] during optimization to preserve geometry density obtained from
the depth sensor, decrease computation time, and better control the number of
Gaussians. We optimize the active sub-map to render the depth and color of all
its keyframes. We directly optimize RGB color without using spherical harmonics
to speed up optimization. In Gaussian splatting [25] the scene representation is
optimized for many iterations over all the training views. However, this approach
does not suit the SLAM setup where speed is crucial. Naively optimizing with an
equal number of iterations for all keyframes results in underfitting or excessive
time spent on optimization. We solve this by optimizing only the keyframes in
the active sub-map and spending at least 40% of iterations on the new keyframe.
3.3
Geometry and Color Encoding
While Gaussian Splatting [25] is good at rendering images, the rendered depth
maps are of limited accuracy since there is no direct depth supervision. We tackle
this problem with an additional depth loss. To render the depth Di at pixel i
that is influenced by m ordered Gaussians we compute:
  D
_
{i}
 =
 \ su m  _ {j \leq m}\mu ^{z}_{j} \cdot \alpha _{j} \cdot T_{j} \enspace , 
(6)
where µz
j is a z component of the mean of a 3D Gaussian, αj and Tj are the
same as in Eq. (3). To update the 3D Gaussian parameters based on the observed
depth, we derive the gradients of the depth loss w.r.t. the 3D Gaussians’ means,
covariances, and opacity. Denoting the depth loss as Ldepth, we follow the chain
rule to compute the gradient for the mean update of the Gaussian j:
  \frac
 {\
p artial 
L_\
mat
hrm
 {d
ept
h}}{\partial \mu _j} = \frac {\partial L_\mathrm {depth}}{\partial D_i} \frac {\partial D_i}{\partial \alpha _{j}} \frac {\partial \alpha _j}{\partial \mu _j} \enspace , 
(7)
where
∂Ldepth
∂Di
is computed with PyTorch autograd using Eq. (9) and
∂αj
∂µj is
derived as in [25]. We derive ∂Di
∂αj as:
  \
fra
c  {
\ p ar t
i
al D_
i}{\p
a r ti
al \alpha _{j}} = \mu _j^z \cdot T_j - \frac {\sum _{u > j} \mu _u^z \alpha _u T_u}{1 - \alpha _j} \enspace . 
(8)

<!-- page 8 -->
8
V. Yugay et al.
The gradients for covariance and opacity are computed similarly. Apart from
∂Ldepth
∂Di
, all gradients are explicitly computed in CUDA to preserve the opti-
mization speed of the unified rendering pipeline. For depth supervision, we use
the loss:
  L_\t e x t { dep th} = |\hat {D} - D|_1 \enspace , \label {eq:depth_loss} 
(9)
with D and ˆD being the ground-truth and reconstructed depth maps, respec-
tively. For the color supervision we use a weighted combination of L1 and
SSIM [68] losses:
  L_\t e xt  {c o lor }  = ( 1
 
-  \lambda ) 
\
cdot |\hat {I} - I|_1 + \lambda \big (1 - \mathrm {SSIM}(\hat {I}, I) \big ) \enspace , \label {eq:color_loss} 
(10)
where I is the original image, ˆI is the rendered image, and λ = 0.2.
When seeded sparsely as in our case, a few 3D Gaussians sometimes elongate
too much in scale. To overcome this, we add an isotropic regularization term
Lreg when optimizing a sub-map K:
  L_ \
t
ext  {r e g} =
 \f
rac {\sum _{k \in K}|s_k - \overline {s}_k|_1}{|K|}, 
(11)
where sk ∈R3 is the scale of a 3D Gaussian, sk is the mean sub-map scale,
and |K| is the number of Gaussians in the sub-map. Finally, we optimize color,
depth, and regularization terms together:
  L = \l a mbda _ \ text { c olor} \ cdot  L_\t ext {color} + \lambda _\text {depth} \cdot L_\text {depth} + \lambda _\text {reg} \cdot L_\text {reg} \enspace , \label {eq:joint_loss} 
(12)
where λcolor, λdepth, λreg· ∈R≥0 are weights for the corresponding losses.
3.4
Tracking
We perform frame-to-model tracking based on the mapped scene. We initialize
the current camera pose Ti with a constant speed assumption:
  T _{i}  = T_{ i  - 1}  + (T_{i - 1} - T_{i - 2}) \enspace , 
(13)
where pose Ti = {qi, ti} encodes a quaternion and translation vector. To esti-
mate the camera pose we minimize the tracking loss Ltracking with respect to
relative camera pose Ti−1,i between frames i −1 and i as follows:
  \ und
erset 
{T_{i - 1
,
 i}}{\mathr m {arg\,min }} \, L
_
\text {tracking}\Big (\hat {I}(T_{i - 1, i}), \hat {D}(T_{i - 1, i}), I_{i}, D_{i}, \alpha \Big ) \enspace , 
(14)
where ˆI(Ti−1,i) and ˆD(Ti−1,i) are the rendered color and depth from the sub-
map transformed with the relative transformation Ti−1,i, Ci and Di are the input
color and depth map at frame i.
We introduce soft alpha and error masking to not contaminate the tracking
loss with the pixels from previously unobserved or poorly reconstructed areas.
Soft alpha mask Malpha is a polynomial of the alpha map rendered directly from

<!-- page 9 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
9
the active sub-map. Error boolean mask Minlier discards all the pixels where the
color and depth errors are larger than a frame-relative error threshold:
  L_\text  
{
trackin g } = \s u m M_\t e xt { in l ier}  \ c dot M_\text {alpha} \cdot (\lambda _{c}|\hat {I} - I|_1 + (1 - \lambda _{c})|\hat {D} - D|_1). 
(15)
The weighting ensures the optimization is guided by well-reconstructed regions
where the accumulated alpha values are close to 1 and rendering quality is high.
During optimization, all the 3D Gaussian parameters are frozen.
4
Experiments
We first describe our experimental setup and then evaluate our method against
state-of-the-art dense neural RGBD SLAM methods on synthetic [59] and real-
world datasets [12,62,82]. In addition, we compare our method with concurrent
work with released source code. The reported results are the average of 3 runs
using different seeds. The tables highlight best results as first , second , third .
Implementation Details. We set Mu = 600000 for Replica [59], 100000 for
TUM-RGBD [62] and ScanNet [12], and 400000 for ScanNet++ [82]. Mc is set
to 50000 for all datasets. For the first keyframe in a sub-map, the number of
mapping iterations is set to 1,000 for Replica, 100 for TUM-RGBD and Scan-
Net, and 500 for ScanNet++. For the subsequent keyframes in a sub-map, the
iteration count is set to 100 across all datasets. Every 5th frame is considered
as a keyframe for all the datasets. When selecting point candidates from sub-
sequent keyframes, we use alpha threshold αn = 0.6. We use FAISS [19] GPU
implementation to find nearest neighbors when choosing point candidates to add
as new Gaussians and set the search radius ρ = 0.01 m for all the datasets. For
new sub-map initialization, we set dthre = 0.5 m and θthre = 50◦. For sub-map
optimization, the best results were obtained with λcolor, λreg and λdepth to 1.
We spend at least 40% mapping iterations on the newly added keyframe during
sub-map optimization. To mesh the scene, we render depth and color every fifth
frame over the estimated trajectory and use TSDF Fusion [11] with voxel size
1 cm similar to [53]. Further details are provided in the supplement.
Datasets. The Replica dataset [59] comprises high-quality 3D reconstructions
of a variety of indoor scenes. We utilize the publicly available dataset collected
by Sucar et al. [63], which provides trajectories from an RGBD sensor. Further,
we demonstrate that our framework achieves SOTA results on real-world data
by using the TUM-RGBD [62], ScanNet [12] and ScanNet++ [82] datasets. The
poses for TUM-RGBD were captured using an external motion capture system
while ScanNet uses poses estimated by BundleFusion [13], and ScanNet++ ob-
tains poses by registering the images with a laser scan. Since ScanNet++ is not
specifically designed for benchmarking neural SLAM, it has larger camera move-
ments. Therefore, we choose 5 scenes where the first 250 frames are smooth in
trajectory and use them for benchmarking.
Evaluation Metrics. To assess tracking accuracy, we use ATE RMSE [62],
and for rendering we compute PSNR, SSIM [68] and LPIPS [84]. All rendering

<!-- page 10 -->
10
V. Yugay et al.
metrics are evaluated by rendering full-resolution images along the estimated
trajectory with mapping intervals similar to [53]. We also follow [53] to measure
reconstruction performance on meshes produced by marching cubes [32]. The
reconstructions are also evaluated using the F1-score - the harmonic mean of
the Precision (P) and Recall (R). We use a distance threshold of 1 cm for all
evaluations. We further provide the depth L1 metric for unseen views as in [89].
Baseline Methods. We primarily compare our method to existing state-of-
the-art dense neural RGBD SLAM methods such as NICE-SLAM [89], Vox-
Fusion [78], ESLAM [34], and Point-SLAM [53]. In addition, we compare against
the concurrent work using the released code [23].
Rendering Performance. Tab. 1 compares rendering performance and shows
improvements over all the existing dense neural RGBD SLAM methods on syn-
thetic data. Tab. 2 and Tab. 3 show our state-of-the-art rendering performance
on real-world datasets. Fig. 3 shows exemplary full-resolution renderings where
Gaussian-SLAM yields more accurate details. Qualitative results on novel views
are provided as a video in the supplementary.
Table 1: Rendering Performance on Replica [59]. We outperform all existing
dense neural RGBD methods on the commonly reported rendering metrics. Concurrent
work is marked with an asterisk∗.
Method
Metric
Rm0
Rm1
Rm2
Off0
Off1
Off2
Off3
Off4
Avg.
NICE-SLAM [89]
PSNR↑
22.12 22.47 24.52 29.07 30.34 19.66 22.23 24.94 24.42
SSIM ↑
0.689 0.757 0.814 0.874 0.886 0.797 0.801 0.856 0.809
LPIPS ↓0.330 0.271 0.208 0.229 0.181 0.235 0.209 0.198 0.233
Vox-Fusion [78]
PSNR↑
22.39 22.36 23.92 27.79 29.83 20.33 23.47 25.21 24.41
SSIM↑
0.683 0.751 0.798 0.857 0.876 0.794 0.803 0.847 0.801
LPIPS↓
0.303 0.269 0.234 0.241 0.184 0.243 0.213 0.199 0.236
ESLAM [34]
PSNR↑
25.25 27.39 28.09 30.33 27.04 27.99 29.27 29.15 28.06
SSIM↑
0.874
0.89
0.935 0.934 0.910 0.942 0.953 0.948 0.923
LPIPS↓
0.315 0.296 0.245 0.213 0.254 0.238 0.186 0.210 0.245
Point-SLAM [53]
PSNR↑
32.40 34.08 35.50 38.26 39.16 33.99 33.48 33.49 35.17
SSIM↑
0.974 0.977 0.982 0.983 0.986 0.960 0.960 0.979 0.975
LPIPS↓
0.113 0.116 0.111 0.100 0.118 0.156 0.132 0.142 0.124
SplaTAM∗[23]
PSNR↑
32.86 33.89 35.25 38.26 39.17 31.97 29.70 31.81 34.11
SSIM↑
0.98
0.97
0.98
0.98
0.98
0.97
0.95
0.95
0.97
LPIPS↓
0.07
0.10
0.08
0.09
0.09
0.10
0.12
0.15
0.10
Gaussian-SLAM (ours)
PSNR↑
38.88 41.80 42.44 46.40 45.29 40.10 39.06 42.65 42.08
SSIM↑
0.993 0.996 0.996 0.998 0.997 0.997 0.997 0.997 0.996
LPIPS↓
0.017 0.018 0.019 0.015 0.016 0.020 0.020 0.020 0.018
Tracking Performance. In Tab. 4, Fig. 4, Tab. 5 and Tab. 6 we report the
tracking accuracy. Our method outperforms the nearest competitor by 14% on
[59]. On TUM-RGBD dataset [62], Gaussian-SLAM also performs better than
all baseline methods. On ScanNet dataset, our method exhibits a drift due to
low-quality depth maps and a large amount of motion blur. On ScanNet++,
our Gaussian splatting-based method performs significantly better than NeRF-

<!-- page 11 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
11
Table 2: Rendering Performance on TUM-RGBD [62]. We outperform existing
dense neural RGBD methods on the commonly reported rendering metrics. For quali-
tative results, see Fig. 3. Concurrent work is marked with an asterisk∗.
Method
Metric
fr1/desk
fr2/xyz
fr3/office
Avg.
NICE-SLAM [89]
PSNR↑
13.83
17.87
12.890
14.86
SSIM↑
0.569
0.718
0.554
0.614
LPIPS↓
0.482
0.344
0.498
0.441
Vox-Fusion [78]
PSNR↑
15.79
16.32
17.27
16.46
SSIM↑
0.647
0.706
0.677
0.677
LPIPS↓
0.523
0.433
0.456
0.471
ESLAM [34]
PSNR↑
11.29
17.46
17.02
15.26
SSIM↑
0.666
0.310
0.457
0.478
LPIPS↓
0.358
0.698
0.652
0.569
Point-SLAM [53]
PSNR↑
13.87
17.56
18.43
16.62
SSIM↑
0.627
0.708
0.754
0.696
LPIPS↓
0.544
0.585
0.448
0.526
SplaTAM∗[23]
PSNR↑
22.00
24.50
21.90
22.80
SSIM↑
0.857
0.947
0.876
0.893
LPIPS↓
0.232
0.100
0.202
0.178
Gaussian-SLAM (ours)
PSNR↑
24.01
25.02
26.13
25.05
SSIM↑
0.924
0.924
0.939
0.929
LPIPS↓
0.178
0.186
0.141
0.168
Table 3: Rendering Performance on ScanNet [12]. We outperform existing dense
neural RGBD methods on the commonly reported rendering metrics by a significant
margin. For qualitative results, see Fig. 3. Concurrent work is marked with an asterisk∗.
Method
Metric
0000
0059
0106
0169
0181
0207
Avg.
NICE-SLAM [89]
PSNR↑
18.71
16.55
17.29
18.75
15.56
18.38
17.54
SSIM↑
0.641
0.605
0.646
0.629
0.562
0.646
0.621
LPIPS↓
0.561
0.534
0.510
0.534
0.602
0.552
0.548
Vox-Fusion [78]
PSNR↑
19.06
16.38
18.46
18.69
16.75
19.66
18.17
SSIM↑
0.662
0.615
0.753
0.650
0.666
0.696
0.673
LPIPS↓
0.515
0.528
0.439
0.513
0.532
0.500
0.504
ESLAM [34]
PSNR↑
15.70
14.48
15.44
14.56
14.22
17.32
15.29
SSIM↑
0.687
0.632
0.628
0.656
0.696
0.653
0.658
LPIPS↓
0.449
0.450
0.529
0.486
0.482
0.534
0.488
Point-SLAM [53]
PSNR↑
21.30
19.48
16.80
18.53
22.27
20.56
19.82
SSIM↑
0.806
0.765
0.676
0.686
0.823
0.750
0.751
LPIPS↓
0.485
0.499
0.544
0.542
0.471
0.544
0.514
SplaTAM∗[23]
PSNR↑
19.33
19.27
17.73
21.97
16.76
19.8
19.14
SSIM↑
0.660
0.792
0.690
0.776
0.683
0.696
0.716
LPIPS↓
0.438
0.289
0.376
0.281
0.420
0.341
0.358
Gaussian-SLAM(ours)
PSNR↑
28.539
26.208
26.258
28.604
27.789
28.627
27.67
SSIM↑
0.926
0.9336
0.9259
0.917
0.9223
0.9135
0.923
LPIPS↓
0.271
0.211
0.217
0.226
0.277
0.288
0.248
based methods. In addition, Gaussian-SLAM demonstrates greater robustness
compared to alternative approaches.
Reconstruction Performance. In Tab. 7 we compare our method to NICE-
SLAM [89], Vox-Fusion [78], ESLAM [34], Point-SLAM [53], and concurrent
SplaTAM [23] in terms of the geometric reconstruction accuracy on the Replica
dataset [59]. Our method performs on par with other existing dense SLAM
methods.

<!-- page 12 -->
12
V. Yugay et al.
NICE-SLAM [89]
ESLAM [34]
Point-SLAM [53]
Ours
Ground-truth
scene 0000
scene 0207
fr1-desk
fr3-office
2e74812d00
fb05e13ad1
Fig. 3: Rendering performance on ScanNet [12], TUM-RGBD [61] and Scan-
Net++ [82]. Thanks to 3D Gaussian splatting, Gaussian-SLAM can encode more
high-frequency details and substantially increase the quality of the renderings (please
zoom in for a better view of the details). This is also supported by the quantitative
results in Tab. 2 and Tab. 3.
Table 4: Tracking Performance on Replica [59] (ATE RMSE ↓[cm]). We out-
perform all other methods in on Replica. Concurrent work is marked with an asterisk∗.
Method
Rm0
Rm1
Rm2
Off0
Off1
Off2
Off3
Off4
Avg.
NICE-SLAM [89]
1.69
2.04
1.55
0.99
0.90
1.39
3.97
3.08
1.95
Vox-Fusion [78]
0.27
1.33
0.47
0.70
1.11
0.46
0.26
0.58
0.65
ESLAM [34]
0.71
0.70
0.52
0.57
0.55
0.58
0.72
0.63
0.63
Point-SLAM [53]
0.61
0.41
0.37
0.38
0.48
0.54
0.72
0.63
0.52
SplaTAM∗[23]
0.31
0.40
0.29
0.47
0.27
0.29
0.32
0.55
0.36
Gaussian SLAM (ours) 0.29
0.29
0.22
0.37
0.23
0.41
0.30
0.35
0.31

<!-- page 13 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
13
Method
desk xyz office Avg.
NICE-SLAM [89] 4.3
31.7 3.9
13.3
Vox-Fusion [78]
3.5
1.5
26.0
10.3
Point-SLAM [53] 4.3
1.3
3.5
3.0
SplaTAM∗[23]
3.4
1.2 5.2
3.3
Gaussian SLAM 2.6
1.3
4.6
2.9
Fig. 4: Tracking Performance on TUM-
RGBD
[62]
(ATE
RMSE↓
[cm]).
Our
method outperforms all other methods on
TUM_RGBD. Concurrent work is marked
with an asterisk∗.
Alpha Inlier ATE RMSE PSNR
mask mask
[cm]↓
[dB]↑
✗
✗
12.77
22.96
✓
✗
2.68
24.01
✗
✓
8.96
23.66
✓
✓
2.50
24.32
Fig. 5: Tracking Mask Ablation
on desk. Applying alpha and inlier
masks to the tracking loss improves
tracking leading to better rendering.
Table 5: Tracking Performance on ScanNet [12] (ATE RMSE↓[cm]). Tracking
on ScanNet is especially challenging due to low-quality depth maps and motion blur.
Method
0000
0059
0106
0169
0181
0207
Avg.
NICE-SLAM [89] 12.00
14.00
7.90
10.90
13.40
6.20
10.70
Vox-Fusion [78]
68.84
24.18
8.41
27.28
23.30
9.41
26.90
Point-SLAM [53] 10.24
7.81
8.65
22.16
14.77
9.54
12.19
SplaTAM∗[23]
12.83
10.10
17.72
12.08
11.10
7.46
11.88
Gaussian SLAM
24.75
8.63
11.27
14.59
18.70
14.36
15.38
Table 6: Tracking Performance on ScanNet++ [82] (ATE RMSE ↓[cm]). Our
tracking proves to be robust and competitive in various real-world scenes.
Method
b20a261fdf
8b5caf3398
fb05e13ad1
2e74812d00
281bc17764
Avg.
Point-SLAM [53]
246.16
632.99
830.79
271.42
574.86
511.24
ESLAM [34]
25.15
2.15
27.02
20.89
35.47
22.14
SplaTAM∗[23]
1.50
0.57
0.31
443.10
1.58
89.41
Gaussian SLAM (ours) 1.37
5.97
2.70
2.35
1.02
2.68
Table 7: Reconstruction Performance on Replica [59]. Our method is compa-
rable to the SOTA baseline Point-SLAM [53] which requires ground truth depth maps
for inference while superior to other dense SLAM methods. Concurrent work is marked
with an asterisk∗.
Method
Metric
Rm0
Rm1
Rm2
Off0 Off1 Off2 Off3 Off4 Avg.
NICE-SLAM [89]
Depth L1 [cm]↓
1.81 1.44 2.04 1.39 1.76 8.33 4.99 2.01
2.97
F1 [%]↑
45.0 44.8 43.6 50.0 51.9 39.2 39.9 36.5
43.9
Vox-Fusion [78]
Depth L1 [cm]↓
1.09 1.90 2.21 2.32 3.40 4.19 2.96 1.61
2.46
F1 [%]↑
69.9 34.4 59.7 46.5 40.8 51.0 64.6 50.7
52.2
ESLAM [34]
Depth L1 [cm] ↓0.97 1.07 1.28 0.86 1.26 1.71 1.43 1.06
1.18
F1 [%] ↑
81.0 82.2 83.9 78.4 75.5 77.1 75.5 79.1 79.1
Point-SLAM [53]
Depth L1 [cm]↓
0.53 0.22 0.46 0.30 0.57 0.49 0.51 0.46 0.44
F1 [%]↑
86.9 92.3 90.8 93.8 91.6 89.0 88.2 85.6 89.8
SplaTAM∗[23]
Depth L1 [cm]↓
0.43 0.38 0.54 0.44 0.66 1.05 1.60 0.68
0.72
F1 [%]↑
89.3 88.2 88.0 91.7 90.0 85.1 77.1 80.1
86.1
Gaussian SLAM (ours) Depth L1 [cm]↓
0.61 0.25 0.54 0.50 0.52 0.98 1.63 0.42 0.68
F1 [%]↑
88.8 91.4 90.5 91.7 90.1 87.3 84.2 87.4 88.9

<!-- page 14 -->
14
V. Yugay et al.
Runtime Comparison. In Tab. 8 we compare runtime usage on the Replica
office0 scene. We report both per-iteration and per-frame runtime. The per-
frame runtime is calculated as the optimization time spent on all mapped frames
divided by sequence length, while the per-iteration runtime is the average map-
ping iteration time.
Table 8: Average Mapping, Tracking, and Rendering Speed on Replica
office0. Per-frame runtime is calculated as the total optimization time divided by
sequence length. All metrics are profiled using an NVIDIA RTX A6000 GPU.
Method
Mapping
Mapping
Tracking
Tracking
Rendering
/Iteration(ms)
/Frame(s)
/Iteration(ms)
/Frame(s)
(FPS)
NICE-SLAM [89]
89
1.15
27
1.06
2.64
Vox-Fusion [78]
98
1.47
64
1.92
1.63
ESLAM [34]
30
0.62
18
0.15
0.65
Point-SLAM [53]
57
3.52
27
1.11
2.96
SplaTAM∗[23]
81
4.89
67
2.70
2175
Gaussian-SLAM (ours)
24
0.93
14
0.83
2175
Ablation Study. In Fig. 5 we ablate the effectiveness of soft alpha mask Malpha
and inlier mask Minlier for the tracking performance on TUM-RGBD fr1/desk
scene. It demonstrates the effectiveness of both masks with the soft alpha mask
having more performance impact.
Limitations and Future Work. Although we have effectively used 3D Gaus-
sians for online dense SLAM, tracking a camera trajectory on data with lots of
motion blur and low-quality depth maps remains challenging. We also believe
that some of our empirical hyperparameters like keyframe selection strategy can
be made test time adaptive or learned. Finally, trajectory drift is inevitable in
frame-to-model tracking without additional techniques like loop-closure or bun-
dle adjustment which might be an interesting future work.
5
Conclusion
We introduced Gaussian-SLAM, a dense SLAM system based on 3D Gaussian
Splatting as the scene representation that enables unprecedented re-rendering
capabilities. We proposed effective strategies for efficient seeding and online op-
timization of 3D Gaussians, their organization in sub-maps for better scalability,
and a frame-to-model tracking algorithm. Compared to previous SOTA neural
SLAM systems like Point-SLAM [53] we achieve faster tracking and mapping
while obtaining better rendering results on synthetic and real-world datasets.
We demonstrated that Gaussian-SLAM yields top results in rendering, camera
pose estimation, and scene reconstruction on a variety of datasets.

<!-- page 15 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
15
A
Abstract
This supplementary material accompanies the main paper by providing further
information for better reproducibility as well as additional evaluations and qual-
itative results.
B
Further Implementation Details
The inlier mask Minlier in tracking loss filters out pixels that have depth errors
50 times larger than the median depth error of the current re-rendered depth
map. Pixels without valid depth input are also excluded as the inconsistent re-
rendering in those areas can misguide the pose optimization. For soft alpha mask,
we adopt Malpha = α3 for per-pixel loss weighting. The opacity values for added
Gaussians are initialized as 0.5 and their initial scales are set to the nearest
neighbor distances in the active sub-map. At the middle and the end of mapping
iterations on new keyframes, we prune Gaussians having opacity values lower
than a threshold othre. We set othre = 0.1 for Replica [59] and 0.5 for all other
datasets. Additionally, we use multi-scale RGBD odometry [50] to help initialize
the pose for tracking optimization on Replica dataset, for all other datasets,
we use pose initialization based on constant speed assumption. On Scannet++
dataset [82], if at the initialized pose, the re-rendering loss is 50 times larger
than the running average of the re-rendering loss after tracking optimization, we
use the odometry to re-initialize the pose for the current frame.
Upon completing the pipeline for the input sequence, we merge the saved
sub-maps into a global map. Specifically, we select Gaussians from each sub-
map in sequence as candidates and add them using the same nearest neighbor
checking rule as in the sub-map building. Finally, as a post-processing step, we
perform color refinement on the global map for 10000 iterations.
C
Additional Experiments
Isotropic Regularization Ablation. In Tab. 9 and Tab. 10 we ablate the
isotropic regularization term Lreg in the mapping loss. On Replica dataset [59],
where the RGBD inputs are synthetic and noise-free, the Lreg terms improves
both tracking and rendering performance marginally. While on real-world TUM-
RGBD dataset [61], Lreg proves critical for accurate camera tracking. Fig. 6 fur-
ther examines the underlying Gaussians on fr1/desk scene at a held-out view.
Without isotropic regularization, the elongated Gaussians are evident. While
they slightly enhance rendering by overfitting to training views, they are detri-
mental to pose optimization, which relies on re-rendering at novel views.
Qualitative Reconstruction Results. Fig. 7 shows reconstructed mesh on
Replica dataset with a normal map shader to highlight the difference. Fig. 8 com-
pares colored mesh on ScanNet [12] and TUM-RGBD scenes. Gaussian-SLAM
can recover more geometric and color details in real-world reconstructions.

<!-- page 16 -->
16
V. Yugay et al.
Table 9: Isotropic Regularization Ablation on Replica dataset [59]. The reg-
ularization term Lreg improves both tracking and rendering performance.
Metric
Lreg Rm0
Rm1
Rm2
Off0
Off1
Off2
Off3
Off4
Avg.
ATE ↓
✗
0.25
0.36
0.27
0.52
0.23
0.37
0.30
0.41
0.34
✓
0.29
0.29
0.22
0.37
0.23
0.41
0.30
0.35
0.31
PSNR ↑
✗
38.83
41.71
42.18
46.12
44.72
39.72
38.94
42.58
41.85
✓
38.88 41.80 42.44 46.40 45.29 40.10 39.06 42.65 42.08
Table 10: Isotropic Regularization Ablation on TUM-RGBD dataset [61].
The regularization term Lreg is critical for tracking accuracy and improves the rendering
performance.
Metric
Lreg
fr1/desk
fr2/xyz
fr3/office
Average
ATE ↓
✗
2.7
22.4
4.7
9.9
✓
2.6
1.3
4.6
2.9
PSNR ↑
✗
24.50
23.03
26.60
24.71
✓
24.01
25.02
26.13
25.05
Novel View Synthesis. In Tab. 11 we report the novel view synthesis results
on the selected Scannet++ [82] scenes. The evaluated novel views in this dataset
are not sampled from the input stream, but held-out views, which can better
assess the extrapolation capability of the method. Gaussian-SLAM demonstrates
clear advantage and outperforms concurrent work [23] by an average of 3.6 dB
in PSNR. Qualitative rendering results are provided in Fig. 9.
(a) with isotropic regularization
(b) without isotropic regularization
Fig. 6: Gaussians splatted at a held-out view in the mapped fr1/desk scene.
Notice the highlighted areas in red rectangles, where elongated Gaussians are clearly
noticeable if isotropic regularization is not applied.

<!-- page 17 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
17
ESLAM [34]
Point-SLAM [53]
Gaussian-SLAM
Ground Truth
room 0
room 1
office 0
office 4
Fig. 7: Qualitative Reconstruction Comparison on the Replica dataset [59].
Results are rendered with a normal map shader. Gaussian-SLAM achieves comparable
reconstruction performance with the state-of-the-art dense neural SLAM methods.
Table 11: Novel View Synthesis Performance on ScanNet++ dataset [82]
(PSNR ↑[dB]). Gaussian-SLAM demonstrates a clear advantage, outperforming con-
current work [23] by an average of 3.6 dB in PSNR for held-out views. Our calculation
includes all pixels, regardless of whether they have valid depth input.
Method
b20a261fdf 8b5caf3398 fb05e13ad1 2e74812d00 281bc17764
Average
ESLAM [34]
13.63
11.86
11.83
10.59
10.64
11.71
SplaTAM [23]
23.95
22.66
13.95
8.47
20.06
17.82
Gaussian-SLAM
25.92
24.49
16.36
18.56
22.04
21.47
References
1. Azinović, D., Martin-Brualla, R., Goldman, D.B., Nießner, M., Thies, J.: Neural
rgb-d surface reconstruction. In: IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 6290–6301 (2022) 4
2. Bian, W., Wang, Z., Li, K., Bian, J.W., Prisacariu, V.A.: Nope-nerf: Optimising
neural radiance field with no pose prior. arXiv preprint arXiv:2212.07388 (2022) 3
3. Božič, A., Palafox, P., Thies, J., Dai, A., Nießner, M.: Transformerfusion: Monoc-
ular rgb scene reconstruction using transformers. arXiv preprint arXiv:2107.02191
(2021) 3
4. Bylow, E., Olsson, C., Kahl, F.: Robust online 3d reconstruction combining a depth
sensor and sparse feature points. In: 2016 23rd International Conference on Pattern
Recognition (ICPR). pp. 3709–3714 (2016) 3

<!-- page 18 -->
18
V. Yugay et al.
ESLAM [34]
Point-SLAM [53]
Gaussian-SLAM
Ground Truth
0059
0169
desk1
office
Fig. 8: Qualitative Mesh-based Comparison on ScanNet [12] and TUM-
RGBD [61] datasets. For TUM-RGBD, the ground truth is obtained by TSDF
fusion. NICE-SLAM [89] shows over-smoothed surfaces. Point-SLAM [53] has dupli-
cated geometry. ESLAM [34] improves the reconstruction, while Gaussian-SLAM is
moderately better in recovering geometric details, see the chairs in scene_0059 for ex-
ample.
ESLAM [34]
SplaTAM [23]
Gaussian-SLAM
Ground Truth
281bc17764
8b5caf3398
Fig. 9: Qualitative Novel View Synthesis Comparison on the ScanNet++
dataset [82]. Gaussian-SLAM renders least artifacts at held-out views.
5. Cao, Y.P., Kobbelt, L., Hu, S.M.: Real-time high-accuracy three-dimensional re-
construction with consumer rgb-d cameras. ACM Transactions on Graphics (TOG)
37(5), 1–16 (2018) 3, 6

<!-- page 19 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
19
6. Chen, J., Bautembach, D., Izadi, S.: Scalable real-time volumetric surface recon-
struction. ACM Transactions on Graphics (ToG) 32(4), 1–16 (2013) 3
7. Cho, H.M., Jo, H., Kim, E.: Sp-slam: Surfel-point simultaneous localization and
mapping. IEEE/ASME Transactions on Mechatronics 27(5), 2568–2579 (2021) 3
8. Choe, J., Im, S., Rameau, F., Kang, M., Kweon, I.S.: Volumefusion: Deep depth
fusion for 3d scene reconstruction. In: IEEE/CVF International Conference on
Computer Vision (ICCV). pp. 16086–16095 (October 2021) 3
9. Choi, S., Zhou, Q.Y., Koltun, V.: Robust reconstruction of indoor scenes. In: Pro-
ceedings of the IEEE conference on computer vision and pattern recognition. pp.
5556–5565 (2015) 3, 6
10. Chung, C.M., Tseng, Y.C., Hsu, Y.C., Shi, X.Q., Hua, Y.H., Yeh, J.F., Chen, W.C.,
Chen, Y.T., Hsu, W.H.: Orbeez-slam: A real-time monocular visual slam with orb
features and nerf-realized mapping. arXiv preprint arXiv:2209.13274 (2022) 3
11. Curless, B., Levoy, M.: Volumetric method for building complex models from range
images. In: SIGGRAPH Conference on Computer Graphics. ACM (1996) 3, 9
12. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scan-
Net: Richly-annotated 3D reconstructions of indoor scenes. In: Conference on
Computer Vision and Pattern Recognition (CVPR). IEEE/CVF (2017). https:
//doi.org/10.1109/CVPR.2017.261 9, 11, 12, 13, 15, 18
13. Dai, A., Nießner, M., Zollhöfer, M., Izadi, S., Theobalt, C.: Bundlefusion: Real-
time globally consistent 3d reconstruction using on-the-fly surface reintegration.
ACM Transactions on Graphics (ToG) 36(4), 1 (2017) 1, 3, 6, 9
14. Davison, A.J., Reid, I.D., Molton, N.D., Stasse, O.: Monoslam: Real-time sin-
gle camera slam. IEEE transactions on pattern analysis and machine intelligence
29(6), 1052–1067 (2007) 1
15. Fuentes-Pacheco, J., Ruiz-Ascencio, J., Rendón-Mancha, J.M.: Visual simultaneous
localization and mapping: a survey. Artificial intelligence review 43, 55–81 (2015)
1
16. Gao, Y., Cao, Y.P., Shan, Y.: Surfelnerf: Neural surfel radiance fields for online
photorealistic reconstruction of indoor scenes. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. pp. 108–118 (2023) 4
17. Huang, H., Li, L., Cheng, H., Yeung, S.K.: Photo-slam: Real-time simultaneous
localization and photorealistic mapping for monocular, stereo, and rgb-d cameras
(2023) 4
18. Huang, J., Huang, S.S., Song, H., Hu, S.M.: Di-fusion: Online implicit 3d recon-
struction with deep priors. In: IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 8932–8941 (2021) 1, 3
19. Johnson, J., Douze, M., Jégou, H.: Billion-scale similarity search with GPUs. IEEE
Transactions on Big Data 7(3), 535–547 (2019) 9
20. Kähler, O., Prisacariu, V., Valentin, J., Murray, D.: Hierarchical voxel block hash-
ing for efficient integration of depth images. IEEE Robotics and Automation Let-
ters 1(1), 192–197 (2015) 3
21. Kähler, O., Prisacariu, V.A., Ren, C.Y., Sun, X., Torr, P.H.S., Murray, D.W.:
Very high frame rate volumetric integration of depth images on mobile devices.
IEEE Trans. Vis. Comput. Graph. 21(11), 1241–1250 (2015). https://doi.org/
10.1109/TVCG.2015.2459891, https://doi.org/10.1109/TVCG.2015.2459891 3
22. Kazerouni, I.A., Fitzgerald, L., Dooly, G., Toal, D.: A survey of state-of-the-art on
visual slam. Expert Systems with Applications 205, 117734 (2022) 1
23. Keetha, N., Karhade, J., Jatavallabhula, K.M., Yang, G., Scherer, S., Ramanan,
D., Luiten, J.: Splatam: Splat, track & map 3d gaussians for dense rgb-d slam.
arXiv preprint (2023) 4, 10, 11, 12, 13, 14, 16, 17, 18

<!-- page 20 -->
20
V. Yugay et al.
24. Keller, M., Lefloch, D., Lambers, M., Izadi, S., Weyrich, T., Kolb, A.: Real-time
3d reconstruction in dynamic scenes using point-based fusion. In: International
Conference on 3D Vision (3DV). pp. 1–8. IEEE (2013) 3
25. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July
2023), https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ 1, 4, 5,
7
26. Klein, G., Murray, D.: Parallel tracking and mapping for small ar workspaces.
In: 2007 6th IEEE and ACM international symposium on mixed and augmented
reality. pp. 225–234. IEEE (2007) 1
27. Li, H., Gu, X., Yuan, W., Yang, L., Dong, Z., Tan, P.: Dense rgb slam with neural
implicit maps. arXiv preprint arXiv:2301.08930 (2023) 3, 4
28. Li, K., Tang, Y., Prisacariu, V.A., Torr, P.H.: Bnv-fusion: Dense 3d reconstruction
using bi-level neural volume fusion. In: IEEE/CVF Conference on Computer Vision
and Pattern Recognition. pp. 6166–6175 (2022) 3
29. Lin, C.H., Ma, W.C., Torralba, A., Lucey, S.: BARF: Bundle-Adjusting Neu-
ral Radiance Fields. In: International Conference on Computer Vision (ICCV).
IEEE/CVF (2021). https://doi.org/10.1109/ICCV48922.2021.00569 3
30. Liu, D., Chen, C., Xu, C., Qiu, R.C., Chu, L.: Self-supervised point cloud registra-
tion with deep versatile descriptors for intelligent driving. IEEE Transactions on
Intelligent Transportation Systems (2023) 3
31. Liu, L., Gu, J., Lin, K.Z., Chua, T.S., Theobalt, C.: Neural sparse voxel fields.
In: Advances in Neural Information Processing Systems. vol. 33, pp. 15651–15663
(2020) 3
32. Lorensen, W.E., Cline, H.E.: Marching cubes: A high resolution 3d surface con-
struction algorithm. ACM siggraph computer graphics 21(4), 163–169 (1987) 10
33. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking
by persistent dynamic view synthesis (2023) 4
34. Mahdi Johari, M., Carta, C., Fleuret, F.: Eslam: Efficient dense slam system based
on hybrid representation of signed distance fields. arXiv e-prints pp. arXiv–2211
(2022) 1, 2, 3, 4, 10, 11, 12, 13, 14, 17, 18
35. Maier, R., Sturm, J., Cremers, D.: Submap-based bundle adjustment for 3d re-
construction from rgb-d data. In: Pattern Recognition: 36th German Conference,
GCPR 2014, Münster, Germany, September 2-5, 2014, Proceedings 36. pp. 54–65.
Springer (2014) 6
36. Marniok, N., Johannsen, O., Goldluecke, B.: An efficient octree design for local
variational range image fusion. In: German Conference on Pattern Recognition
(GCPR). pp. 401–412. Springer (2017) 3
37. Matsuki, H., Murai, R., Kelly, P.H.J., Davison, A.J.: Gaussian splatting slam
(2023) 4
38. Mescheder, L., Oechsle, M., Niemeyer, M., Nowozin, S., Geiger, A.: Occupancy
networks: Learning 3d reconstruction in function space. In: IEEE/CVF conference
on computer vision and pattern recognition. pp. 4460–4470 (2019) 4
39. Mihajlovic, M., Weder, S., Pollefeys, M., Oswald, M.R.: Deepsurfels: Learning on-
line appearance fusion. In: IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition. pp. 14524–14535 (2021) 4
40. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng,
R.: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In:
European Conference on Computer Vision (ECCV). CVF (2020). https://doi.
org/10.1007/978-3-030-58452-8{_}24, https://link.springer.com/10.1007/
978-3-030-58452-8_24http://arxiv.org/abs/2003.08934 1, 3

<!-- page 21 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
21
41. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with
a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1–
15 (2022) 3
42. Mur-Artal, R., Tardos, J.D.: ORB-SLAM2: An Open-Source SLAM System for
Monocular, Stereo, and RGB-D Cameras. IEEE Transactions on Robotics 33(5),
1255–1262 (2017). https://doi.org/10.1109/TRO.2017.2705103 1
43. Murez, Z., van As, T., Bartolozzi, J., Sinha, A., Badrinarayanan, V., Rabinovich,
A.: Atlas: End-to-end 3d scene reconstruction from posed images. In: Computer
Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020,
Proceedings, Part VII 16. pp. 414–431. Springer (2020) 3
44. Newcombe, R.A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A.J.,
Kohli, P., Shotton, J., Hodges, S., Fitzgibbon, A.W.: Kinectfusion: Real-time dense
surface mapping and tracking. In: ISMAR. vol. 11, pp. 127–136 (2011) 1, 3
45. Newcombe, R.A., Lovegrove, S.J., Davison, A.J.: Dtam: Dense tracking and map-
ping in real-time. In: International Conference on Computer Vision (ICCV) (2011)
1, 3
46. Nießner, M., Zollhöfer, M., Izadi, S., Stamminger, M.: Real-time 3d reconstruction
at scale using voxel hashing. ACM Transactions on Graphics (TOG) 32 (11 2013).
https://doi.org/10.1145/2508363.2508374 1, 3
47. Oechsle, M., Peng, S., Geiger, A.: UNISURF: Unifying Neural Implicit Surfaces
and Radiance Fields for Multi-View Reconstruction. In: International Conference
on Computer Vision (ICCV). IEEE/CVF (2021). https://doi.org/10.1109/
ICCV48922.2021.00554, https://ieeexplore.ieee.org/document/9709919/ 3
48. Oleynikova, H., Taylor, Z., Fehr, M., Siegwart, R., Nieto, J.I.: Voxblox: Incre-
mental 3d euclidean signed distance fields for on-board MAV planning. In: 2017
IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS
2017, Vancouver, BC, Canada, September 24-28, 2017. pp. 1366–1373. IEEE
(2017). https://doi.org/10.1109/IROS.2017.8202315, https://doi.org/10.
1109/IROS.2017.8202315 3
49. Ortiz, J., Clegg, A., Dong, J., Sucar, E., Novotny, D., Zollhoefer, M., Mukadam, M.:
isdf: Real-time neural signed distance fields for robot perception. arXiv preprint
arXiv:2204.02296 (2022) 4
50. Park, J., Zhou, Q.Y., Koltun, V.: Colored point cloud registration revisited. In:
Proceedings of the IEEE international conference on computer vision. pp. 143–152
(2017) 15
51. Peng, S., Niemeyer, M., Mescheder, L., Pollefeys, M., Geiger, A.: Convolu-
tional Occupancy Networks. In: European Conference Computer Vision (ECCV).
CVF (2020), https://www.microsoft.com/en- us/research/publication/
convolutional-occupancy-networks/ 4
52. Rosinol, A., Leonard, J.J., Carlone, L.: NeRF-SLAM: Real-Time Dense Monocular
SLAM with Neural Radiance Fields. arXiv (2022), http://arxiv.org/abs/2210.
13641 3
53. Sandström, E., Li, Y., Van Gool, L., Oswald, M.R.: Point-slam: Dense neural
point cloud-based slam. In: International Conference on Computer Vision (ICCV).
IEEE/CVF (2023) 1, 2, 4, 9, 10, 11, 12, 13, 14, 17, 18
54. Sandström, E., Ta, K., Van Gool, L., Oswald, M.R.: Uncle-slam: Uncertainty learn-
ing for dense neural slam. In: International Conference on Computer Vision Work-
shops (ICCVW). IEEE/CVF (2023) 3
55. Sayed, M., Gibson, J., Watson, J., Prisacariu, V., Firman, M., Godard, C.: Sim-
plerecon: 3d reconstruction without 3d convolutions. In: European Conference on
Computer Vision. pp. 1–19. Springer (2022) 3

<!-- page 22 -->
22
V. Yugay et al.
56. Schops, T., Sattler, T., Pollefeys, M.: BAD SLAM: Bundle adjusted direct RGB-D
SLAM. In: CVF/IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) (2019) 1, 3, 4
57. Steinbrucker, F., Kerl, C., Cremers, D.: Large-scale multi-resolution surface recon-
struction from rgb-d sequences. In: IEEE International Conference on Computer
Vision. pp. 3264–3271 (2013) 3
58. Stier, N., Rich, A., Sen, P., Höllerer, T.: Vortx: Volumetric 3d reconstruction with
transformers for voxelwise view selection and fusion. In: 2021 International Con-
ference on 3D Vision (3DV). pp. 320–330. IEEE (2021) 3
59. Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., Engel, J.J.,
Mur-Artal, R., Ren, C., Verma, S., et al.: The replica dataset: A digital replica of
indoor spaces. arXiv preprint arXiv:1906.05797 (2019) 9, 10, 11, 12, 13, 15, 16, 17
60. Stückler, J., Behnke, S.: Multi-resolution surfel maps for efficient dense 3d modeling
and tracking. Journal of Visual Communication and Image Representation 25(1),
137–147 (2014) 6
61. Stühmer, J., Gumhold, S., Cremers, D.: Real-time dense geometry from a handheld
camera. In: Joint Pattern Recognition Symposium. pp. 11–20. Springer (2010) 1,
12, 15, 16, 18
62. Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, D.: A benchmark for
the evaluation of RGB-D SLAM systems. In: International Conference on Intelli-
gent Robots and Systems (IROS). IEEE/RSJ (2012). https://doi.org/10.1109/
IROS.2012.6385773, http://ieeexplore.ieee.org/document/6385773/ 2, 9, 10,
11, 13
63. Sucar, E., Liu, S., Ortiz, J., Davison, A.J.: iMAP: Implicit Mapping and Posi-
tioning in Real-Time. In: International Conference on Computer Vision (ICCV).
IEEE/CVF (2021). https://doi.org/10.1109/ICCV48922.2021.00617, https:
//ieeexplore.ieee.org/document/9710431/ 1, 3, 4, 9
64. Sun, J., Xie, Y., Chen, L., Zhou, X., Bao, H.: Neuralrecon: Real-time coherent
3d reconstruction from monocular video. In: IEEE/CVF Conference on Computer
Vision and Pattern Recognition. pp. 15598–15607 (2021) 3
65. Tang, Y., Zhang, J., Yu, Z., Wang, H., Xu, K.: Mips-fusion: Multi-implicit-
submaps for scalable and robust online neural rgb-d reconstruction. arXiv preprint
arXiv:2308.08741 (2023) 1
66. Wang, H., Wang, J., Agapito, L.: Co-slam: Joint coordinate and sparse parametric
encodings for neural real-time slam. In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. pp. 13293–13302 (2023) 1, 4
67. Wang, J., Wang, P., Long, X., Theobalt, C., Komura, T., Liu, L., Wang, W.:
Neuris: Neural reconstruction of indoor scenes using normal priors. In: Computer
Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27,
2022, Proceedings, Part XXXII. pp. 139–155. Springer (2022) 3, 4
68. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment:
from error visibility to structural similarity. IEEE transactions on image processing
13(4), 600–612 (2004) 8, 9
69. Wang, Z., Wu, S., Xie, W., Chen, M., Prisacariu, V.A.: Nerf–: Neural radiance
fields without known camera parameters. arXiv preprint arXiv:2102.07064 (2021)
3
70. Weder, S., Schonberger, J., Pollefeys, M., Oswald, M.R.: Routedfusion: Learning
real-time depth map fusion. In: IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 4887–4897 (2020) 3

<!-- page 23 -->
Gaussian-SLAM: Photo-realistic Dense SLAM
23
71. Weder, S., Schonberger, J.L., Pollefeys, M., Oswald, M.R.: Neuralfusion: Online
depth fusion in latent space. In: IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 3162–3172 (2021) 3
72. Whelan, T., Leutenegger, S., Salas-Moreno, R., Glocker, B., Davison, A.: Elastic-
fusion: Dense slam without a pose graph. In: Robotics: Science and Systems (RSS)
(2015) 1, 3, 4
73. Whelan, T., McDonald, J., Kaess, M., Fallon, M., Johannsson, H., Leonard, J.J.:
Kintinuous: Spatially extended kinectfusion. In: Proceedings of RSS ’12 Workshop
on RGB-D: Advanced Reasoning with Depth Cameras (2012) 3
74. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang,
X.: 4d gaussian splatting for real-time dynamic scene rendering (2023) 4
75. Xu, Q., Xu, Z., Philip, J., Bi, S., Shu, Z., Sunkavalli, K., Neumann, U.: Point-nerf:
Point-based neural radiance fields. In: IEEE/CVF Conference on Computer Vision
and Pattern Recognition. pp. 5438–5448 (2022) 4
76. Yan, C., Qu, D., Wang, D., Xu, D., Wang, Z., Zhao, B., Li, X.: Gs-slam: Dense
visual slam with 3d gaussian splatting (2024) 4
77. Yan, Z., Tian, Y., Shi, X., Guo, P., Wang, P., Zha, H.: Continual neural map-
ping: Learning an implicit scene representation from sequential observations. In:
IEEE/CVF International Conference on Computer Vision (ICCV). pp. 15782–
15792 (October 2021) 3, 4
78. Yang, X., Li, H., Zhai, H., Ming, Y., Liu, Y., Zhang, G.: Vox-fusion: Dense tracking
and mapping with voxel-based neural implicit representation. In: IEEE Interna-
tional Symposium on Mixed and Augmented Reality (ISMAR). pp. 499–507. IEEE
(2022) 3, 10, 11, 12, 13, 14
79. Yang, X., Ming, Y., Cui, Z., Calway, A.: Fd-slam: 3-d reconstruction using features
and dense matching. In: 2022 International Conference on Robotics and Automa-
tion (ICRA). pp. 8040–8046. IEEE (2022) 4
80. Yang, Z., Yang, H., Pan, Z., Zhu, X., Zhang, L.: Real-time photorealistic dynamic
scene representation and rendering with 4d gaussian splatting (2023) 4
81. Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3d gaussians
for high-fidelity monocular dynamic scene reconstruction (2023) 4
82. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset
of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference
on Computer Vision. pp. 12–22 (2023) 9, 12, 13, 15, 16, 17, 18
83. Zhang, H., Chen, G., Wang, Z., Wang, Z., Sun, L.: Dense 3d mapping for indoor
environment based on feature-point slam method. In: 2020 the 4th International
Conference on Innovation in Artificial Intelligence. pp. 42–46 (2020) 3
84. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effec-
tiveness of deep features as a perceptual metric. In: IEEE conference on computer
vision and pattern recognition. pp. 586–595 (2018) 9
85. Zhang, Y., Tosi, F., Mattoccia, S., Poggi, M.: Go-slam: Global optimization for
consistent 3d instant reconstruction. arXiv preprint arXiv:2309.02436 (2023) 1, 3,
4
86. Zhou, Q.Y., Koltun, V.: Dense scene reconstruction with points of interest. ACM
Transactions on Graphics (TOG) 32(4), 120 (2013). https://doi.org/10.1145/
2461912.2461919 3
87. Zhou, Q.Y., Miller, S., Koltun, V.: Elastic fragments for dense scene reconstruction.
Proceedings of the IEEE International Conference on Computer Vision pp. 2726–
2733 (2013) 3

<!-- page 24 -->
24
V. Yugay et al.
88. Zhu, Z., Peng, S., Larsson, V., Cui, Z., Oswald, M.R., Geiger, A., Pollefeys,
M.: Nicer-slam: Neural implicit scene encoding for rgb slam. arXiv preprint
arXiv:2302.03594 (2023) 3
89. Zhu, Z., Peng, S., Larsson, V., Xu, W., Bao, H., Cui, Z., Oswald, M.R., Pollefeys,
M.: Nice-slam: Neural implicit scalable encoding for slam. In: IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition. pp. 12786–12796 (2022) 1, 3,
10, 11, 12, 13, 14, 18
90. Zollhöfer, M., Stotko, P., Görlitz, A., Theobalt, C., Nießner, M., Klein, R., Kolb,
A.: State of the art on 3d reconstruction with rgb-d cameras. In: Computer graphics
forum. vol. 37, pp. 625–652. Wiley Online Library (2018) 3
91. Zwicker, M., Pfister, H., Van Baar, J., Gross, M.: Surface splatting. In: Proceedings
of the 28th annual conference on Computer graphics and interactive techniques.
pp. 371–378 (2001) 6
