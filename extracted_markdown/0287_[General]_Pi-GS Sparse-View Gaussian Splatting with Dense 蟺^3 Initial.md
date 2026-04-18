<!-- page 1 -->
Pi-GS: Sparse-View Gaussian Splatting with Dense π3 Initialization
Manuel Hofer
Markus Steinberger
Thomas K¨ohler
Graz University of Technology
Austria
Figure 1. 3DGS exhibits floaters and view inconsistencies under sparse-view constraints. These artifacts are mostly caused by depth
ambiguities and poor Gaussian alignment with the underlying geometry, as shown in the depth and normal maps. By incorporating
depth supervision, normal supervision, and additional pseudo views, our method significantly reduces these artifacts and produces more
view-consistent novel views with improved Gaussian alignment under sparse-view constraints.
Abstract
Novel view synthesis has evolved rapidly, advancing
from Neural Radiance Fields to 3D Gaussian Splatting
(3DGS), which offers real-time rendering and rapid train-
ing without compromising visual fidelity. However, 3DGS
relies heavily on accurate camera poses and high-quality
point cloud initialization, which are difficult to obtain in
sparse-view scenarios. While traditional Structure from
Motion (SfM) pipelines often fail in these settings, existing
learning-based point estimation alternatives typically re-
quire reliable reference views and remain sensitive to pose
or depth errors. In this work, we propose a robust method
utilizing π3, a reference-free point cloud estimation net-
work. We integrate dense initialization from π3 with a
regularization scheme designed to mitigate geometric in-
accuracies. Specifically, we employ uncertainty-guided
depth supervision, normal consistency loss, and depth
warping. Experimental results demonstrate that our ap-
proach achieves state-of-the-art performance on the Tanks
and Temples, LLFF, DTU, and MipNeRF360 datasets.
1. Introduction
3D scene reconstruction and novel view synthesis (NVS)
are rapidly advancing, with many applications across dif-
ferent domains [32]. These methods can be applied in
fields such as Virtual Reality (VR) for creating immersive
worlds, cinematography to create visually appealing as-
sets efficiently, or robot vision to help robots understand
their physical environment [23]. The foundation of 3D
scene reconstruction was laid by traditional Structure from
Motion pipelines. More recently, significant advances in
NVS were achieved by representing the scene as Neu-
ral Radiance Fields (NeRF) [16]. These methods achieve
state-of-the-art results but suffer from slow training speeds
and are unsuitable for real-time rendering due to high la-
tency.
Newer methods such as 3D Gaussian Splatting
(3DGS) [11] enable high-quality NVS even for real-time
rendering. Additionally, training speed is significantly re-
duced.
A major limitation of these novel view synthesis meth-
ods is the need for dense views, which often is not feasi-
ble for real-world applications. In sparse-view settings,
these methods tend to struggle with bad initialization,
1
arXiv:2602.03327v1  [cs.GR]  3 Feb 2026

<!-- page 2 -->
depth ambiguities and overfitting to training views. To
improve the performance in these settings and counter-
act the depth ambiguities, certain priors are introduced to
better generalize and escape minima throughout the opti-
mization process. Methods such as DNGaussian [14] and
Few-shot Novel View Synthesis using Depth [13] leverage
monocular depth estimators to regularize the model with
the help of the inferred depth. The depth regularization
helps significantly to improve the depth ambiguities and
increase the generalization capability of the models. A
challenge for these models is correct depth scaling, proper
point initialization, and accurate camera poses. The ini-
tial points and camera poses are traditionally generated
using Structure from Motion (SfM) pipelines. However,
these pipelines often struggle with sparse input views and
limited overlap between views. Recent advancements for
sparse-view settings were achieved by leveraging dense
initialization with the help of point cloud estimation net-
works [27, 28, 31].
They replace the traditional SfM
pipeline with models such as MASt3R [7] or DUSt3R [24]
for the point cloud estimation and camera pose estimation.
The resulting models achieve high-fidelity results but re-
quire good initial reference views for accurate predictions.
In addition, a time-consuming iterative camera alignment
process is required, which can take several minutes. In-
accurate camera poses may further reduce reconstruction
quality.
We make the following contributions:
• We discuss a method for leveraging a Permutation-
Equivariant point cloud estimation network for dense
initialization without relying on traditional SfM.
• We introduce confidence aware pearson depth loss, to
counteract uncertain depth estimations.
• We explore the use of PGSR in sparse-view settings for
improved geometry alignment and reduced overfitting.
Our method achieves state-of-the-art results in sparse-
view settings and significantly improves Gaussian surface
alignment, while reducing floaters. Our code is publicly
available at https://github.com/Mango0000/Pi-GS.
2. Related Work
This section reviews prior work on 3D reconstruction,
covering classical geometry-based pipelines, neural radi-
ance fields, and Gaussian splatting approaches, with a fo-
cus on sparse-view and pose-free scenarios.
2.1. Traditional 3D Reconstruction
Classical 3D reconstruction pipelines typically rely on
Structure-from-Motion (SfM) to achieve camera pose es-
timation and to generate a point cloud from a given set of
images taken from various viewpoints. Afterward, Multi-
View Stereo (MVS) and surface reconstruction techniques
such as Poisson reconstruction are used [10, 20, 34].
These methods perform well in textured and opaque
scenes but struggle with transparent materials and sparse
or low-overlap views. Moreover, they are highly sensitive
to SfM failures, which can lead to unstable surface recon-
struction.
2.2. Neural Radiance Fields
Neural Radiance Fields (NeRF) [16] represent scenes by
continuous volumetric functions. This makes them ca-
pable of producing photorealistic novel views and han-
dling view-dependent effects more accurately. However,
a downside is that NeRFs are quite demanding in terms
of computation. As a result, we see more efficient vari-
ants like Instant-NGP [17], PlenOctree [30] and Efficient-
NeRF [9] that drastically shorten the training and render-
ing time by incorporating optimized data structures and
improving the architecture.
2.3. 3D Gaussian Splatting
3D Gaussian Splatting (3DGS) [11] has emerged as a new
method that improves on training and rendering speed by
replacing the implicit radiance field of NeRF-based meth-
ods with an explicit representation.
Its core idea is to
use 3D Gaussians both for optimization and for render-
ing via rasterization, therefore achieving real-time ren-
dering without losing either fine details or transparency.
Advanced 3DGS methods, such as PGSR: Planar-based
Gaussian Splatting for Efficient and High-Fidelity Sur-
face Reconstruction (PGSR) [4], improve Gaussian sur-
face alignment with the help of planar Gaussians and
multi-view consistency losses. However, these methods
generally rely on SfM for initialization and are optimized
for dense and overlapping views.
2.4. Sparse-View Gaussian Splatting
Reconstruction from sparse views remains a major chal-
lenge for 3DGS. Several augmentations exist that address
sparse-view reconstruction by introducing additional con-
straints and regularization terms.
Depth-based super-
vision is explored in Depth-Regularized 3D Gaussian
Splatting [6], Few-shot NVS with Depth-Aware 3D Gaus-
sian Splatting [13], and DNGaussian [14].
This type
of supervision results in fast convergence and reduces
depth ambiguities. Meanwhile, DropGaussian [18] and
DropoutGS [29] deactivate Gaussians at random in order
to counteract overfitting. There are also more advanced
methods like FSGS: Real-Time Few-Shot View Synthesis
using Gaussian Splatting [33] which introduces a pool-
ing strategy and fine-tunes the splitting strategy to im-
prove sparse view reconstruction across different datasets.
While these methods achieve very robust results in sparse-
view scenarios, they typically rely on accurate camera
poses from SfM.
2.5. SfM-Free Methods
Methods such as COLMAP-Free 3D Gaussian Splat-
ting [8] and InstantSplat [31], eliminate the need for SfM
by jointly optimizing the 3D Gaussians as well as the cam-
era poses and using depth estimations for point cloud ini-
tialization. These methods are able to handle sparse-view
2

<!-- page 3 -->
situations more robustly and recover from inaccurate cam-
era poses.
2.6. Diffusion-Based Priors
More recent works incorporate diffusion priors not
only to stabilize the reconstruction, but also to generate
additional views from the limited number of input views.
GenFusion [26], SparseGS [28], Gaussian Scenes [19],
and Intern-GS [27] are some of the methods where these
advantages can also be observed. While these methods
achieve impressive results, they often struggle with
high-frequency textures and view inconsistencies due to
depth ambiguities and inaccurate Gaussian alignment.
Our
method
differs
fundamentally
from
diffusion-
based and optimization-heavy approaches.
Instead of
synthesizing novel views using generative priors, we
improve reconstruction quality through dense geometric
initialization and strong generalizability across datasets.
We leverage depth and normal supervision from esti-
mated depth maps and explicitly model depth uncertainty
through confidence-aware constraints, allowing devi-
ations from noisy estimates.
Camera poses and point
representations are predicted by a feed-forward network,
reducing reliance on iterative optimization and increas-
ing robustness in sparse-view settings.
Consequently,
our approach focuses on geometric consistency and
generalization without relying on view hallucination or
diffusion-based priors.
3. Method
We begin by outlining preliminaries on Gaussian Splatting
and planar depth rendering. Section 3.2 details modifica-
tions to PGSR for sparse settings, followed by our dense
initialization strategy in Section 3.3. We then present our
uncertainty-aware Pearson loss in Section 3.4 and artifact-
free normal supervision in Section 3.5.
Finally, Sec-
tion 3.6 describes our depth warping approach for improv-
ing view consistency.
3.1. Preliminaries
Gaussian Splatting.
3D Gaussian Splatting (3DGS) in-
troduced by Kerbl et al. [11] achieves great novel view
synthesis results with high efficiency by leveraging a
Gaussian scene representation. Another improvement of
this scene representation over NeRF is the real-time ren-
dering speed, as well as much faster training times. Our
approach also builds upon 3DGS. The scene representa-
tion is defined by a set of 3D Gaussians. Each Gaussian
can be defined by a 3D covariance matrix Σ ∈R3×3 and
the 3D center point µ ∈R3 in world space,
G(x) = e−1
2 (x−µi)T Σ−1(x−µi).
(1)
To project this 3D Gaussian onto the 2D image plane
for rendering, the covariance matrix Σ′ in clip space is
defined as the following:
Σ′ = JWΣW T JT ,
(2)
where J is the Jacobian of the affine approximation for
this projection transformation and W is the view transfor-
mation matrix.
For the covariance matrix to be physically meaning-
ful, it needs to be positive semi-definite. To ensure this
throughout the training process, Σ is defined as the fol-
lowing:
Σ = RSST RT ,
(3)
where S ∈R3×3 is the scaling matrix, and R ∈R3×3 is
the rotation matrix. This allows separate optimization of
rotation and scaling and ensures that Σ is positive semi-
definite. For increased memory efficiency, the rotation
matrix is stored as a quaternion, and scaling as 3D vec-
tor.
Furthermore, for rendering the color C, we blend the
colors of each Gaussian along the ray, as follows:
C =
N
X
i=1
Tiαici,
(4)
where N is the number of Gaussians along a ray, ci is the
color of the i-th Gaussian represented by spherical har-
monics (SH) to account for view dependent effects, αi is
the weighted opacity of the i-th Gaussian and Ti is the
transmittance of the i-th Gaussian [11].
Transmittance Ti is defined as:
Ti =
i−1
Y
j=1
(1 −αj).
(5)
By calculating the color for each ray from the camera, we
can render an image. The training of this Gaussian repre-
sentation is done by back propagation with the following
loss function:
L = (1 −λ)L1 + λLD−SSIM,
(6)
where L1 is a simple l1 loss between the rendered and
ground-truth image and LD−SSIM is an image simi-
larity measure between rendered and ground-truth im-
age [2, 11].
3DGS relies on camera poses and points
obtained from structure from motion (SfM). However,
in sparse-view settings, the resulting point cloud can be
highly sparse, and the overlap between the images may be
insufficient to extract reliable structures or accurate cam-
era poses. This leads to a challenging starting point for
3DGS optimization.
Depth and Normal Rendering.
We use Planar-based
Gaussian Splatting for Efficient and High-Fidelity Surface
Reconstruction [4] (PGSR) for normal and depth render-
ing. PGSR builds upon 3DGS, enabling the rendering and
backpropagation of both the depth and normals. A naive
3

<!-- page 4 -->
approach of computing the depth D of a pixel would be to
use depth accumulation defined as:
D =
N
X
i=1
Tiαizi,
(7)
where Ti is the same as in Eq. (5), αi is the weighted
opacity of the i-th Gaussian and zi is its distance from
the camera [5]. PGSR on the other hand compresses the
3D Gaussians to get flat 2D planes, from which unbiased
depth and normal maps can be rendered [4].
To get the 2D planes, PGSR flattens the 3D Gaussians
by minimizing the minimum scale and therefore defining
the scale loss Ls as following:
Ls = || min(s1, s2, s3)||1,
(8)
where si is the i-th scale component of each Gaussian.
The direction of the minimum scale factor corresponds
to the normal ni. Therefore, the normals per ray, N, can
be rendered as following:
N =
N
X
i=1
RT
c niαiTi,
(9)
where Rc is the rotation from the camera to the global
world.
The distance di from the Gaussian plane to the camera
center is defined as:
di = (RT
c (µi −Tc))RT
c nT
i ,
(10)
where Tc is the camera center in the world and µi is the
center of the i-th Gaussian.
The distance D along a ray can now be defined as:
D =
N
X
i=1
diαiTi.
(11)
PGSR extends 3DGS by introducing an Image Edge-
Aware Single-View Loss Lsvgeo, which optimizes the
Gaussian Scene with the Local Plane Assumption. This
assumption states that two neighbouring pixels can be
considered as an approximate local plane, but only if
these pixels do not belong to an edge.
The loss helps
to improve the local depth and normal consistency. They
also propose a Multi-View Geometric Consistency Loss,
Lmvgeom, which enhances geometric smoothness by pro-
jecting the depth and normals from one frame to another.
Finally, they employ a Multi-View Photometric Consis-
tency Loss, Lmvrgb, which projects the grayscale image
from one camera to another camera through depth warp-
ing [4].
3.2. PGSR Sparse-View
Default PGSR does not work well for the sparse-view set-
ting out-of-the-box because of the multi-view observer
(a) Ballroom scene with opacity
reset.
(b) Ballroom scene without
opacity reset.
Figure 2. Comparison of the Ballroom scene from Tanks and
Temples with and without opacity reset [12]. Background details
are lost when opacity reset is executed, and image quality further
degrades over the training process.
trim, which assures that each point is observed by mul-
tiple cameras and this is not guaranteed in sparse-view
settings. Therefore, we deactivate this trimming for our
method. Another parameter that requires adjustment is the
opacity reset interval. When opacity reset happens, fine
details in the background will be lost and artifacts appear,
as can be seen in Fig. 2. The details in Fig. 2a at the back
wall are completely lost and artifacts in the window frame
become visible. By continuing the training process even
further, the artifacts’ strength increases, and they become
even more prominent. When deactivating opacity reset,
the background details are retained and the artifacts van-
ish without sacrificing the overall quality. This can also
be seen in Fig. 2b. The improvement is also reflected in
the PSNR (Peak Signal-to-Noise Ratio), which increases
from 22.76 to 23.73. With these few settings, it is already
possible to run the PGSR framework with acceptable re-
sults. For improved performance, we deactivate the split-
ting strategy as it is not needed for our dense point cloud
initialization. The point cloud is already very detailed and
this setting does not improve the final results (cf. Tab. 1).
3.3. Dense Initialization
Sparse-view settings pose a fundamental challenge for
standard SfM frameworks like COLMAP [21, 22], where
limited image overlap can lead registration to fail. Fur-
thermore, the resulting sparse point clouds serve as a poor
initialization for 3DGS, complicating the optimization of
Gaussian primitives and compromising geometric fidelity.
To mitigate this, we leverage a pre-trained feed-forward
network to predict both depth and camera parameters.
This strategy provides the dense geometric initialization
and accurate poses required for high-quality sparse-view
reconstruction. Figure 3a illustrates the point cloud gen-
erated by the feed-forward model π3 [25], while Fig. 3b
depicts the result from COLMAP [21]. Both methods use
the same 24 input views from the ”bicycle” scene of the
MipNeRF360 dataset [3], rendered here from an identi-
cal viewpoint.
The difference in density is significant:
The COLMAP reconstruction contains only 1,028 points,
whereas π3 yields 1,013,106 points. Note that the π3 out-
put was filtered using the default confidence threshold of
20%.
4

<!-- page 5 -->
(a) Point cloud inferred with π3.
(b) Point cloud created with
COLMAP.
Figure 3. Comparison between π3 point cloud and COLMAP
point cloud, of the bike scene from MipNeRF360 Dataset with
24 training images [3, 21, 25].
3.4. Depth Supervision
From π3, we obtain the per view point clouds which can
be used as a depth map. For depth regularization, we eval-
uated different losses.
Standard L1 and L2 losses often cause the model to
overfit to the limited fidelity of the inferred depth maps.
We also evaluated the Global-Local Depth Normalization
from DNGaussian [14] but found it unnecessary given
the inherent scale consistency of our predictions.
In-
stead, we utilize a Pearson correlation loss, which has
demonstrated superior performance. This approach en-
forces structural consistency while enabling the recovery
of high-frequency details that are missing from the initial
depth estimation.
In addition to the default Pearson correlation loss, we
also integrated the confidence given by π3. As a result,
the final depth can be modeled even more accurately by
assigning low weights to uncertain regions. Our newly
created confidence-aware depth loss, Lpearson, is defined
as:
µp =
PN
i=1 CiDp
i
PN
i=1 Ci
,
µt =
PN
i=1 CiDt
i
PN
i=1 Ci
,
(12)
¯Dp = Dp −µp,
¯Dt = Dt −µt,
(13)
Pconf =
PN
i=1 Ci ¯Dp
i ¯Dt
i
rPN
i=1 Ci
  ¯Dp
i
2 PN
i=1 Ci
  ¯Dt
i
2,
(14)
Lpearson = 1 −Pconf,
(15)
N is the number of pixels, Dp
i is the predicted depth of the
i-th pixel, Ci the confidence of the i-th pixel and Dt
i is the
ground truth of the i-th pixel, which is the depth estimated
by π3, and Pconf is the confidence-aware Pearson corre-
lation. The resulting rendered depth after 7,000 iterations
with the help of confidence-aware Pearson correlation can
be seen in Fig. 4.
3.5. Normal Supervision
Surface Normals can be computed with the help of depth
maps by calculating the pixel-wise partial derivatives ∂z
∂x
and ∂z
∂y, where x and y are the pixel coordinates and z
is the depth value, either rendered or estimated by π3.
(a) Confidence-aware Peason loss.
(b) Pearson loss.
Figure 4. Depth rendering of the Ballroom scene from the Tanks
and Temples dataset, comparing the confidence-aware Pearson
loss with the standard pearson loss [12]. The confidence-aware
loss leverages uncertainty estimates to enhance detail, particu-
larly in the background, and also improves performance with
low-resolution depth estimates.
(a) Default normal map with artifacts.
(b) Masked normal map.
Figure 5. The Normal map generated from the depth map us-
ing partial derivatives, which introduces grid artifacts, and the
masked normal map, which removes grid artifacts introduced by
π3 architecture [25].
Because π3 processes each image in patches of 14 × 14
pixels, the gradient is not continuous between adjacent
patches, leading to grid-like artifacts, as can be seen in
Fig. 5a. To alleviate this problem, we add a mask to ig-
nore these discontinuous regions during loss computation.
The mask is computed by creating a grid with 14 × 14
pixel cells, masking the 1-pixel-wide inner border of each
cell. Therefore, the Gaussians are not regularized in these
border regions, and the grid artifacts do not appear in the
scene representation. The masked normal map can be seen
in Fig. 5b. As supervision, we simply use the L1 loss be-
tween the rendered and ground-truth normal map defined
as:
Lnormal = 1
N
N
X
i=1
||N t
i −N p
i ||1,
(16)
where N is the number of pixels, N t
i is the ground-truth
normal at pixel i and N p
i is the predicted normal at pixel i.
3.6. Depth Warping
To improve generalization of our model further, we in-
clude pseudo-views which are generated with the help of
depth warping. This is achieved by projecting the image
pixels from one camera into 3D space, and then repro-
jecting the 3D points into the 2D image plane of a tar-
get camera. For accurate results, we only project pixels
with high confidence and mask out the rest, including un-
seen regions. To generate high-quality pseudo-cameras,
we use circle interpolation with the camera parameters as
input. A circle can be defined by three points, so we use
5

<!-- page 6 -->
Figure 6. The two Figures show two reprojection examples with
the applied mask for the Barn scene of the Tanks and Temples
dataset [12].
the two nearest cameras to the target camera for pseudo-
view generation. The positions of the three cameras de-
fine our circle. Now then interpolate by a certain amount
between each pair of neighbouring views, which results in
two additional views per camera. We can generate an arbi-
trary number of pseudo-views by adjusting the interpola-
tion step size. However, in our experiments, two pseudo-
views between each pair yielded the best results.
The
nearest cameras are already computed by PGSR, therefore
we can reuse them. A few examples of these generated
pseudo-views can be seen in Fig. 6. These pseudo-views
are then used throughout training for additional supervi-
sion with the help of SSIM and L1 loss, but with a weight
set to 0.1.
4. Evaluation
For testing strategy, we adhere to previous state-of-the-art
models to ensure comparability. The datasets used for the
evaluation are Tanks and Temples [12], MipNeRF360 [3],
LLFF [15] and DTU [1].
Implementation
Details.
The
Tanks
and
Temples
dataset covers real-world indoor and outdoor scenes, but
we only use a subset of 8 scenes, as done by other sparse-
view models like Intern-GS and InstantSplat. We focus
on the 3-view setting and therefore use the same train/test
split. This means the testing set includes 12 images uni-
formly sampled without the first and last frame and the
remaining set is the training set where we again uniformly
sampled the 3 views [31]. For Tanks and Temples, no
downsampling is applied.
The MipNeRF360 dataset contains real-world 360◦in-
door and outdoor scenes. For this dataset, two different
approaches are used. One for the 3-view setting as defined
by Gaussian Scenes [19] and one for the 12-view setting
as defined by SparseGS [28]. For both settings, the 4x
downsampled images are used, to adhere to the evaluation
strategies of state-of-the-art models. For the 3-view set-
ting, we use every 8th image as testing set and uniformly
sample the 3 training views. For the 12-view setting, we
use the split dataset provided by SparseGS [28]. The 12-
view setting uses only 6 of the 9 scenes contained in the
MipNeRF360 dataset, whereas the 3-view setting uses all
9 scenes.
The LLFF dataset contains real-world forward-facing
images. For this dataset, we used the same evaluation
strategy as defined by DNGaussian. A downsampling rate
of 8 is used, and we adhere to the train/test split of the 3-
view setting of DNGaussian [14].
Lastly, we also evaluated on the DTU dataset, which
contains highly calibrated lab captures of object centric
scenes. This dataset also provides bit masks to separate
the background and real camera poses. We used our own
inferred camera poses. We again used the testing strat-
egy defined by DNGaussian. This time we used 4x down-
sampled images and the same train/test split of the 3-view
setting of DNGaussian [14]. Similar to DNGaussian and
other comparable methods, we applied the provided sepa-
ration masks for the evaluation.
We use the exact same settings for all evaluations.
π3 [25] automatically downsamples the images to a cer-
tain pixel size, therefore we counteract the downsampling
by rescaling the cameras to the full size.
To make a
fair comparison, we only project the training views to 3D
space. The testing views are only used to get initial cam-
era positions. We train for 7000 iterations, with depth
loss, normal loss as well as pseudo views. The pseudo
views are generated with a confidence threshold of 20%.
This means that we mask out the projected pixel with
confidence under 20%. Splitting of Gaussians is deacti-
vated. We evaluate our model in terms of PSNR, SSIM
and LPIPS.
4.1. Quantitative Evaluation
Tables 3 and 6 show the comparison between Intern-
GS [27], InstantSplat [31], SparseGS [28], DNGaus-
sian [14], FSGS [33], 3DGS [11] and Our method. On
DTU and Tanks and Temples, our model can reconstruct
the scene accurately, with good Gaussian surface align-
ment and without smoothing out high-frequency textures.
On LLFF our model achieves slightly lower scores, be-
cause of missing information in unseen regions, as our
model optimizes only on seen regions and known infor-
mation. An example of this unseen region is illustrated in
Fig. 7.
Tab. 4 shows the comparison between Gaussian
Scenes, MASt3R Initialization, FSGS and Our method
in the 3-view setting on MipNeRF360 [19, 33].
Our
model achieves the lowest LPIPS score and second high-
est PSNR and SSIM. Compared to FSGS our model does
not rely on accurate camera poses from traditional SfM.
Tab. 5 shows the comparison between 3DGS, DNGaus-
sian, SparseGS and Our method in 12-view setting on
MipNeRF360 [3]. Our model achieves the highest results
with very coherent and view-consistent final scene, as our
model improves the Gaussian surface alignment signifi-
cantly. A comparison can be seen in Fig. 8.
To validate the accuracy of our camera pose estimates, we
evaluate the Absolute Trajectory Error (ATE) on the Tank
and Temples dataset. Our pose estimator, π3, achieves
a mean ATE of 0.0293 and a root mean squared error
(RMSE) of 0.0325, demonstrating that it produces accu-
rate camera poses suitable for fair comparison of photo-
6

<!-- page 7 -->
(a) Ground Truth
(b) Intern-GS
(c) Ours
Figure 7.
The Figures show a comparison between Intern-
GS [27], Ours and the Ground Truth. Our model has very accu-
rate reflections and fewer artifacts, nevertheless our model can
not correctly reconstruct the unseen region at the ceiling.
(a) SparseGS
(b) Ours
Figure
8.
The
Figures
show
a
comparison
between
SparseGS [28] and our method.
Our model reconstructs the
background and ground more accurately, and additionally de-
creases artifacts.
metric metrics in 3D Gaussian splatting.
4.2. Ablation
We evaluate the impact of each individual optimization
on our final result. The evaluation is conducted using the
Barn scene from the Tanks and Temples dataset. It is ev-
ident that all of our optimizations improve the result even
further. Dense point cloud initialization with the help of
π3 significantly improves the result by also reducing the
time required for SfM. Our custom depth loss improves
the score by allowing low confidence depth regions to
optimize more freely. Normal regularization encourages
the Gaussians’ normals to match the ground-truth geom-
etry. Depth warping improves the results by adding more
views, which helps the model generalize better and avoid
overfitting to the training views. Our full model achieves
a PSNR of 22.15 on the Barn scene. We also evaluated
the effect of enabling splitting of Gaussians in our model.
This setting results in a slight decrease in performance and
was therefore deactivated. These results can be seen in
Tab. 1.
Method
PSNR
Original 3DGS
17.53
PGSR
18.05
π3 (dense) initialization
19.66
+ Depth Regularization
20.72
+ Normal Regularization
21.56
+ Depth Warping (Full Model) 22.15
+ Splitting Densification
21.97
Table 1. Ablation study of the regularization techniques intro-
duced in our model on the Barn scene of Tanks and Temples.
Additionally, we evaluated the impact of splitting Gaussians dur-
ing densification.
In addition, we evaluate the impact of using PGSR
compared to standard 3DGS for our sparse view setting
(3-views). Table 2 shows that the planar depth created
by PGSR helps significantly to place the Gaussians more
accurately. Additionally, the losses introduced by PGSR
help to improve the rendering results further. Our model
remains stable even after increased training iterations and
continues to show improved novel view synthesis results.
A visual comparison between 3DGS and PGSR with dif-
ferent number of iterations can be seen in Fig. 9.
Framework Iteration
Tanks and Temples
MipNeRF360
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
PGSR [4]
7000
19.99
0.503
0.355
23.36
0.791
0.156
3DGS [11]
7000
18.00
0.426
0.449
23.07
0.773
0.172
PGSR [4]
15000
20.19
0.517
0.343
23.41
0.795
0.169
3DGS [11]
15000
17.04
0.391
0.465
20.94
0.719
0.244
Table 2. Ablation study on the use of PGSR as base framework
compared to 3DGS. The additional multi-view and single-view
losses introduced by PGSR are activated after iteration 7000.
This comparison shows that the PGSR depth rendering captures
the underlying surface geometry more accurately by also reduc-
ing floaters significantly. With the help of PGSR we achieve
view-consistent surfaces and reduce overfitting significantly.
Method
PSNR↑SSIM↑LPIPS↓
3DGS [11]
15.36
0.572
0.379
DNGaussian [14]
20.69
0.721
0.277
SparseGS [28]
21.20
0.717
0.231
InstantSplat [31]
22.20
0.743
0.199
FSGS [33]
22.31
0.693
0.197
Intern-GS [27]
22.67
0.736
0.191
Ours
22.87
0.764
0.189
Table 3. Evaluation on Tanks and Temples dataset with 3-view
setting. Our model does not oversmooth high-frequency textures
and accurately aligns the Gaussians with the underlying surface
geometry.
7

<!-- page 8 -->
Scene
3DGS [11]
PGSR [4]
7K Iterations
15K Iterations
7K Iterations
15K Iterations
Barn [12]
Ballroom [12]
Kitchen [3]
Garden [3]
Figure 9. Visual comparison between 3DGS and PGSR with different Iterations. The planar depth from PGSR helps significantly to
remove floaters and align the Gaussians accurately to the ground-truth geometry.
Method
PSNR↑SSIM↑LPIPS↓
MASt3R Initialization [19]
12.59
0.231
0.593
Gaussian Scenes [19]
13.81
0.265
0.547
FSGS [33]
14.17
0.318
0.578
Ours
14.14
0.310
0.523
Table 4. Evaluation on MipNeRF360 dataset with 3-view set-
ting. Our model reconstructs seen regions accurately, but can
not introduce geometry in unseen regions.
Method
PSNR↑SSIM↑LPIPS↓
3DGS [11]
17.49
0.490
0.431
DNGaussian [14]
16.28
0.432
0.549
SparseGS [28]
19.37
0.577
0.398
Ours
19.54
0.492
0.362
Table 5. Evaluation on MipNeRF360 dataset with 12-view set-
ting. Our model can reconstruct the scenes with highly accu-
rate surface alignment. The ground is view-consistent, and fewer
floating artifacts compared to SparseGS [28].
5. Conclusion and Limitations
Our model shows strong performance under sparse-view
constraints, specifically when handling between 3 and 12
views. The model demonstrates the importance of accu-
rate dense point cloud initialization. We introduce a mod-
ified depth loss that enables correct scene generalization
by reducing depth ambiguities without introducing arti-
facts in low confidence regions. In addition, we introduce
normal and depth warping loss terms that improve align-
ment with the ground-truth surface geometry. Finally, we
Method
LLFF
DTU
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
3DGS [11]
15.52
0.408
0.405
10.99
0.585
0.313
DNGaussian [14]
19.12
0.591
0.294
18.91
0.790
0.176
SparseGS [28]
19.86
0.668
0.322
18.89
0.834
0.178
InstantSplat [31]
17.67
0.603
0.379
17.55
0.634
0.212
FSGS [33]
20.31
0.652
0.288
19.54
0.732
0.199
Intern-GS [27]
20.49
0.693
0.212
20.34
0.851
0.163
Ours
19.92
0.664
0.254
23.52
0.815
0.145
Table 6. Evaluation on LLFF and DTU dataset with 3-view set-
ting. Following previous work, for evaluation on DTU the back-
ground masks are applied. Our model is able to reconstruct fine-
grained textures accurately, but it underperforms in unobserved
regions compared to methods that generate content for unseen
regions.
relax certain assumptions from PGSR to allow robust op-
timization in sparse-view settings.
Our model faces limitations when dealing with large
datasets, as processing many input views with π3 con-
sumes a large amount of GPU memory, which is infea-
sible on consumer hardware. Additional limitations come
from inaccurate depth estimations in specific scenes, such
as the leaves scene from the LLFF dataset [15]. Future
improvements could include the joint optimization of the
camera poses and the Gaussian scene, which would re-
sult in improved reconstruction quality. Furthermore, the
integration of generative priors could enhance the model’s
ability to maintain photometric and geometric consistency
across occluded or sparse areas.
8

<!-- page 9 -->
References
[1] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vo-
giatzis, Engin Tola, and Anders Bjorholm Dahl. Large-
Scale Data for Multiple-View Stereopsis. IJCV, pages 1–
16, 2016. 6
[2] Allison H. Baker, Alexander Pinard, and Dorit M. Ham-
merling. On a Structural Similarity Index Approach for
Floating-Point Data. IEEE TVCG, 30(9):6261–6274, 2024.
3
[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman.
Mip-NeRF 360:
Un-
bounded Anti-Aliased Neural Radiance Fields. In Proc.
CVPR, pages 5470–5479, 2022. 4, 5, 6, 8
[4] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian
Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao,
and Guofeng Zhang. PGSR: Planar-Based Gaussian Splat-
ting for Efficient and High-Fidelity Surface Reconstruc-
tion. IEEE TVCG, 2024. 2, 3, 4, 7, 8
[5] Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei
Yin, Yuexin Ma, Wenping Wang, and Xuejin Chen. Gaus-
sianPro: 3D Gaussian Splatting with Progressive Propaga-
tion. In Proceedings of the 41st International Conference
on Machine Learning (ICML 2024), 2024. 4
[6] Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee.
Depth-Regularized Optimization for 3D Gaussian Splat-
ting in Few-Shot Images. In Proc. CVPRW, pages 811–
820, 2024. 2
[7] Bardienus Pieter Duisterhof, Lojze Zust, Philippe Wein-
zaepfel, Vincent Leroy, Yohann Cabon, and Jerome Re-
vaud. MASt3R-SfM: a Fully-Integrated Solution for Un-
constrained Structure-from-Motion. In Proc. 3DV, 2025.
2
[8] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A.
Efros, and Xiaolong Wang. COLMAP-Free 3D Gaussian
Splatting. In Proc. CVPR, pages 20796–20805, 2024. 2
[9] Tao Hu, Shu Liu, Yilun Chen, Tiancheng Shen, and Jiaya
Jia. EfficientNeRF: Efficient Neural Radiance Fields. In
Proc. CVPR, pages 12902–12911, 2022. 2
[10] Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe.
Poisson surface reconstruction. In Proc. SGP, 2006. 2
[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis.
3D Gaussian Splatting for Real-
Time Radiance Field Rendering. ACM TOG, 2023. 1, 2, 3,
6, 7, 8
[12] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun. Tanks and Temples: Benchmarking Large-Scale
Scene Reconstruction. ACM TOG, 36(4), 2017. 4, 5, 6, 8
[13] Raja Kumar and Vanshika Vats.
Few-Shot Novel View
Synthesis Using Depth Aware 3D Gaussian Splatting. In
ECCV 2024 Workshops, pages 1–13, Cham, 2025. Springer
Nature Switzerland. 2
[14] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun
Zhou, and Lin Gu. DNGaussian: Optimizing Sparse-View
3D Gaussian Radiance Fields with Global-Local Depth
Normalization. In Proc. CVPR, pages 20775–20785, 2024.
2, 5, 6, 7, 8
[15] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-
Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren
Ng, and Abhishek Kar. Local Light Field Fusion: Practi-
cal View Synthesis with Prescriptive Sampling Guidelines.
ACM TOG, 2019. 6, 8
[16] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng.
NeRF: Representing Scenes as Neural Radiance Fields
for View Synthesis. Communications of the ACM, 65(1):
99–106, 2021. 1, 2
[17] Thomas M¨uller, Alex Evans, Christoph Schied, and
Alexander Keller. Instant Neural Graphics Primitives with
a Multiresolution Hash Encoding.
ACM TOG, 41(4):
102:1–102:15, 2022. 2
[18] Hyunwoo Park, Gun Ryu, and Wonjun Kim. DropGaus-
sian: Structural Regularization for Sparse-view Gaussian
Splatting. In Proc. CVPR, pages 21600–21609, 2025. 2
[19] Soumava Paul, Prakhar Kaushik, and Alan Yuille. Gaus-
sian Scenes: Pose-Free Sparse-View Scene Reconstruction
using Depth-Enhanced Diffusion Priors.
arXiv preprint
arXiv:2411.15966, 2024. 3, 6, 8
[20] Fabio Remondino, Ali Karami, Ziyang Yan, Gabriele Maz-
zacca, Simone Rigon, and Rongjun Qin. A Critical Anal-
ysis of NeRF-Based 3D Reconstruction. Remote Sensing,
15(14), 2023. 2
[21] Johannes Lutz Sch¨onberger and Jan-Michael Frahm.
Structure-from-Motion Revisited. In Proc. CVPR, 2016.
4, 5
[22] Johannes Lutz Sch¨onberger, Enliang Zheng, Marc Polle-
feys, and Jan-Michael Frahm. Pixelwise View Selection
for Unstructured Multi-View Stereo. In Proc. ECCV, 2016.
4
[23] Hamid Taheri and Zhao Chun Xia. SLAM; definition and
evolution. Engineering Applications of Artificial Intelli-
gence, 97:104032, 2021. 1
[24] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D
Vision Made Easy. In Proc. CVPR, 2024. 2
[25] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang,
Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chun-
hua Shen, and Tong He.
π3:
Scalable Permutation-
Equivariant Visual Geometry Learning.
arXiv preprint
arXiv:2507.13347, 2025. 4, 5, 6
[26] Sibo Wu, Congrong Xu, Binbin Huang, Andreas Geiger,
and Anpei Chen. Genfusion: Closing the loop between
reconstruction and generation via videos. In Proc. CVPR,
pages 6078–6088, 2025. 3
[27] Sun Xiangyu, Chen Runnan, Gong Mingming, Xu Dong,
and Liu Tongliang.
Intern-GS: Vision Model Guided
Sparse-View 3D Gaussian Splatting.
arXiv preprint
arXiv:2505.20729, 2025. 2, 3, 6, 7, 8
[28] Haolin Xiong, Sairisheek Muttukuru, Hanyuan Xiao, Rishi
Upadhyay, Pradyumna Chari, Yajie Zhao, and Achuta
Kadambi.
SparseGS: Sparse View Synthesis Using 3D
Gaussian Splatting. In Proc. 3DV, pages 1032–1041, 2025.
2, 3, 6, 7, 8
[29] Yexing Xu, Longguang Wang, Minglin Chen, Sheng Ao,
Li Li, and Yulan Guo. DropoutGS: Dropping Out Gaus-
sians for Better Sparse-view Rendering. In Proc. CVPR,
pages 701–710, 2025. 2
[30] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng,
and Angjoo Kanazawa. PlenOctrees for Real-Time Ren-
dering of Neural Radiance Fields. In Proc. ICCV, pages
5752–5761, 2021. 2
[31] Fan Zhiwen, Wen Kairun, Cong Wenyan, Wang Kevin,
Zhang Jian, Ding Xinghao, Xu Danfei, Ivanovic Boris,
9

<!-- page 10 -->
Pavone Marco, Pavlakos Georgios, Wang Zhangyang, and
Wang Yue. InstantSplat: Sparse-view Gaussian Splatting
in Seconds. arXiv preprint arXiv:2403.20309, 2024. 2, 6,
7, 8
[32] Yiming Zhou, Zixuan Zeng, Andi Chen, Xiaofan Zhou,
Haowei Ni, Shiyao Zhang, Panfeng Li, Liangxi Liu,
Mengyao Zheng, and Xupeng Chen.
Evaluating Mod-
ern Approaches in 3D Scene Reconstruction: NeRF vs
Gaussian-Based Methods. In Proc. DOCS, pages 926–931,
2024. 1
[33] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang
Wang.
Fsgs: Real-time few-shot view synthesis using
gaussian splatting. In Proc. ECCV, page 145–163, Berlin,
Heidelberg, 2024. Springer-Verlag. 2, 6, 7, 8
[34] Onur ¨Ozyes¸il, Vladislav Voroninski, Ronen Basri, and
Amit Singer. A Survey of Structure from Motion. Acta
Numerica, 26:305–364, 2017. 2
10
