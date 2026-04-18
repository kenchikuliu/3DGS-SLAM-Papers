<!-- page 1 -->
SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with
Appearance Embedding
Tianci Wen1,2
Zhiang Liu1,2
Yongchun Fang1,2 ∗
1 IRAIS, tjKLIR, College of Artificial Intelligence, Nankai University
2 IITRS, Shenzhen Research Institute, Nankai University
Figure 1. Our method SEGS-SLAM outperforms SOTA methods (GS-ICP SLAM [9], Photo-SLAM [12], SplaTAM [14], MonoGS [24])
in photorealistic mapping quality across monocular, stereo, and RGB-D cameras. The scenes are from TUM RGB-D dataset [32] (the top
row and the left three images in the bottom row) and the EuRoC MAV dataset [2] (the right three images in the bottom row). Non-obvious
differences in quality are highlighted by arrows/insets.
Abstract
3D Gaussian splatting (3D-GS) has recently revolutionized
novel view synthesis in the simultaneous localization and
mapping (SLAM) problem. However, most existing algo-
rithms fail to fully capture the underlying structure, result-
ing in structural inconsistency.
Additionally, they strug-
gle with abrupt appearance variations, leading to inconsis-
tent visual quality. To address these problems, we propose
SEGS-SLAM, a structure-enhanced 3D Gaussian Splatting
SLAM, which achieves high-quality photorealistic mapping.
Our main contributions are two-fold. First, we propose a
structure-enhanced photorealistic mapping (SEPM) frame-
work that, for the first time, leverages highly structured
point cloud to initialize structured 3D Gaussians, leading
to significant improvements in rendering quality. Second,
we propose Appearance-from-Motion embedding (AfME),
enabling 3D Gaussians to better model image appearance
variations across different camera poses. Extensive exper-
iments on monocular, stereo, and RGB-D datasets demon-
strate that SEGS-SLAM significantly outperforms state-of-
the-art (SOTA) methods in photorealistic mapping quality,
e.g., an improvement of 19.86% in PSNR over MonoGS
∗Corresponding author.
on the TUM RGB-D dataset for monocular cameras. The
project page is available at https://segs-slam.
github.io/.
1. Introduction
Visual simultaneous localization and mapping (SLAM) is
a fundamental problem in 3D computer vision, with wide
applications in autonomous driving, robotics, virtual real-
ity, and augmented reality. SLAM aims to construct dense
or sparse maps to represent the scene. Recently, neural ra-
diance fields (NeRF) [26] has been integrated into SLAM
pipelines, significantly enhancing scene representation ca-
pabilities.
The latest advancement in radiance field ren-
dering is 3D Gaussian splatting (3D-GS) [15], an explicit
scene representation that achieves revolutionary improve-
ments in rendering and training speed.
Recent SLAM
works [9, 11, 12, 14, 17, 24, 27, 39, 40] incorporating
3D-GS have demonstrated that explicit representations pro-
vide more promising rendering performance when com-
pared with implicit ones.
However, most SLAM algorithms based on 3D-GS have
neglected the latent structure in the scene, which constrains
their rendering quality. While some methods [9, 11, 14, 17,
arXiv:2501.05242v3  [cs.CV]  13 Jul 2025

<!-- page 2 -->
24, 27, 39, 40, 43] have explored improvements in accuracy,
efficiency, and semantics, insufficient structural exploita-
tion remains a critical issue.
For instance, as evidenced
by the 2nd row in Fig. 1, MonoGS [24] produces a highly
disorganized reconstruction of the ladder structure due to
this limitation. In contrast, few methods, like Photo-SLAM
[12], leverage scene structure. Photo-SLAM [12] initial-
izes 3D Gaussians with point cloud obtained from indirect
visual SLAM and incorporates a geometry-based densifi-
cation module. In the original 3D-GS, 3D Gaussians are
initialized from COLMAP [30] points. Since indirect visual
SLAM and COLMAP [30] share similar pipeline structures,
the generated point clouds exhibit similar intrinsic proper-
ties. Hence, the 3D Gaussians of Photo-SLAM [12] con-
verge to a relatively optimal result with fewer iterations,
yet it still underutilizes the underlying scene structure. The
blurry reconstruction of the mouse edge by Photo-SLAM
[12] is still apparent as shown in the 2nd row of Fig. 1.
Another partially unresolved challenge in these methods
[9, 11, 12, 14, 17, 24, 27, 39, 40, 43] is the significant ap-
pearance variations within the scene (e.g., exposure, light-
ing). To address this issue, NeRF-W [23] refines the appear-
ance embeddings (AE) in NRW [25] and introduces them
into NeRF. However, AE has a notable limitation: its train-
ing involves each ground-truth image from the test set. In
novel view synthesis tasks, the test set contains 12.5% of
all views, whereas in SLAM tasks, this increases to 80%,
making it more challenging for AE to accurately predict
appearance in novel views. Additionally, these approaches
[9, 11, 12, 14, 17, 21, 24, 27, 39, 40, 43] fail to capture
high-frequency details (e.g., object edges, complex texture
regions). FreGS [45] combines frequency regularization to
model the local details, but its effectiveness is constrained
by the use of a single-scale frequency spectrum.
To address the above limitations, this paper presents
SEGS-SLAM, a novel 3D Gaussian Splatting SLAM sys-
tem. First, we investigate the benefits of leveraging scene
structure for improving rendering accuracy. While point
cloud produced by ORB-SLAM3 [3] preserves strong latent
structure, we observe that the anchor-based 3D Gaussians
in [21] effectively leverage the underlying structure. Moti-
vated by this, we propose a structure-enhanced photoreal-
istic mapping (SEPM) framework, which initializes anchor
points using ORB-SLAM3 [3] point cloud, significantly en-
hancing the utilization of scene structure. Experimental re-
sults validate the effectiveness of this simple yet powerful
strategy, and we hope this insight will inspire further re-
search in this direction. Second, we propose Appearance-
from-Motion embedding (AfME), which takes poses as in-
put and eliminates the need for training on the left half of
each ground-truth image in the test set. We further intro-
duce a frequency pyramid regularization (FPR) technique
to better capture high-frequency details in the scene. The
main contributions of this work are as follows:
1. To our knowledge, structure-enhanced photorealistic
mapping (SEPM) is the first SLAM framework that in-
tializes anchor points with ORB-SLAM3 point cloud to
strengthen the utilization of scene structure, leading to
significant rendering improvements.
2. We
propose
Appearance-from-Motion
embedding
(AfME), which models per-image appearance variations
into a latent space extracted from camera pose.
3. Extensive evaluations on various public datasets demon-
strate that our method significantly surpasses state-of-
the-art (SOTA) methods in photorealistic mapping qual-
ity across monocular, stereo, and RGB-D cameras, while
maintaining competitive tracking accuracy.
2. Related Work
Visual SLAM. Traditional visual SLAM methods can be
classified into two categories: indirect methods and direct
methods. Indirect methods [3, 16] rely on extracting and
tracking features between consecutive frames to estimate
poses and build sparse maps by minimizing a reprojection
error, including ORB-SLAM3 [3]. Direct methods [6, 7]
estimate motion and structure by minimizing a photomet-
ric error, which can build sparse or semi-dense maps. Re-
cently, some methods [1, 18, 33, 35, 36] have integrated
deep learning into visual SLAM systems. Among them,
the current SOTA method is Droid-SLAM [35]. More re-
cently, Lipson et al. [18] combine optical flow prediction
with a pose-solving layer to achieve camera tracking. Our
approach favors traditional indirect visual SLAM.
Implicit Representation based SLAM. iMAP [34] pio-
neers the use of neural implicit representations to achieve
tracking and mapping through reconstruction error. Subse-
quently, many works [4, 13, 22, 37, 41, 42, 49–51] have
explored new representation forms, including voxel-based
neural implicit surface representation [41], multi-scale tri-
planes [13] and point-based neural implicit representation
[28]. Recently, SNI-SLAM [49] and IBD-SLAM [42] in-
troduce a hierarchical semantic representation and an xyz-
map representation, respectively. Some works [5, 10, 19,
29, 44, 47, 48] address other challenges, including loop clo-
sure [19, 48]. However, most efforts focus on scene ge-
ometry reconstruction, with Point-SLAM [28] extending to
novel view synthesis. Moreover, the NeRF models used in
these methods do not account for appearance variations.
3D Gaussian Splatting based SLAM. Recently, an ex-
plicit representation, 3D-GS [15], is introduced into vi-
sual SLAM. Most methods enhance RGB-D SLAM in ren-
dering quality [9, 14, 40, 43], efficiency [9, 27], robust-
ness [11, 39], and semantics [17]. For example, GS-ICP
SLAM[9] achieves photorealistic mapping by fusing 3D-
GS with Generalized ICP. Few methods improve SLAM
rendering accuracy and efficiency across monocular, stereo,

<!-- page 3 -->
Point Cloud
Rendered Image
Ground Truth
Large
Small
AfME
Mono           Stereo            RGB-D
Input
Tracking & Mapping
Structure-Enhanced Photorealistic Mapping
Optimization
Anchor points
Frequency Pyramid
Operation Flow 
Gradient Flow
Mapping results
Loss 
Camera Pose 
Frequency 
Amplitude
Frequency Pyramid
Figure 2. Overview of our method. Our method supports monocular, stereo, and RGB-D cameras. The input image stream is processed by
the localization and geometric mapping modules, generating point cloud and accurate poses. SEPM incrementally initializes anchor points
(middle) based on the point cloud (top left), which preserves the underlying structure. The poses are then fed into the AfME to model
appearance variations in the scene. Additionally, we introduce FPR to improve the reconstruction of high-frequency details in the scene.
and RGB-D cameras, with MonoGS [24] and Photo-SLAM
[12] as exceptions. However, neither of these methods ef-
fectively leverages latent scene structures or enhances the
modeling of scene details, which limits their rendering qual-
ity. To address this, our proposed SEGS-SLAM further re-
inforces the utilization of global scene structure.
3. Preliminaries
In this section, we first introduce the structured 3D Gaussian
splatting in [21]. Subsequently, we review the localization
and mapping process of ORB-SLAM3 [3].
3.1. Structured 3D Gaussian Splatting
Structured 3D Gaussians is a hierarchical representation in
[21]. They construct anchor points by voxelizing the point
cloud obtained from COLMAP [30]. An anchor point is
the center tv of a voxel, equipped with a context feature
ˆfv ∈R32, a scale factor lv ∈R3, and a set of k learnable
offsets Ov = {O0, . . . , Ok−1} ∈Rk×3. For each anchor
point tv, they generate k 3D Gaussians, whose positions are
calculated as:
  \l a b e l  {eq: p os }  \{\ m u  _ 0,\ld o ts ,\mu _{k-1}\} = \mathrm {\mathbf {t}}_v + \{\mathcal {O}_0,\ldots ,\mathcal {O}_{k-1} \} \cdot l_v. 
(1)
Other parameters of k 3D Gaussians are decoded using in-
dividual MLPs, denoted as Mα, Mc, Mq, and Ms, respec-
tively. The colors of the Gaussians are obtained as follows:
  \{ c _ 0 , \dots  ,c_ {k-1 }\} = M_C(\hat {f}_v,\ \delta _{vc},\ \vec {\mathrm {\mathbf {d}}}_{vc}), \label {eq:color_mlp1} 
(2)
where δvc = ∥tv −tc∥2 is the relative distance between
camera position tc and an anchor point and ⃗dvc = (tv −
tc)/δvc is their viewing direction. The opacity {αi}, quater-
nion {qi}, and scale {si} are similarly obtained. After ob-
taining the parameters of each 3D Gaussian G(x) within the
view frustum, it is projected onto the image plane to form
a 2D Gaussian G′
i(x′). Following 3D-GS [15], a tile-based
rasterizer is used to sort the 2D Gaussians, and α−blending
is employed to complete the rendering:
  C(x ^
\
pri
me )
 = 
\
sum
 _ { i\in
 N }  c_i
 \delta _i \prod _{j=1}^{i-1}(1 - \delta _j), \quad \delta _i = \alpha _i G^\prime _i(x^\prime ), \label {eq:rendering} 
(3)
where x′ is the pixel position and N is the number of corre-
sponding 2D Gaussians for each pixel.
3.2. Localization and Geometry Mapping
ORB-SLAM3 [3] can track camera poses and generate
point cloud accurately. The camera pose is represented as
(R, t), where R ∈SO(3) denotes orientation and t ∈R3
represents position. The camera poses (R, t) and the point
cloud {P0, . . . , Pη} ∈Rη×3 of the scene can be solved
through local or global bundle adjustment (BA):
  \{ \ma thr m  {\mat
hbf {P}}
_
m,\math
r
m {\
mathbf  {R} }_l
,\ma th r m { \ mathbf  {t}}_
l\} &= \mathop {\mathrm {argmin}}\limits _{\mathrm {\mathbf {P}}_m,\mathrm {\mathbf {R}}_l,\mathrm {\mathbf {t}}_l} \sum _{\kappa \in \mathcal {K}_L\cup \mathcal {K}_F} \sum _{j\in \mathcal {X}_k} \rho (E(\kappa ,j)), \\ E(\kappa , j) &= \|\mathrm {\mathbf {p}}_j - \pi (\mathrm {\mathbf {R}}_\kappa \mathrm {\mathbf {P}}_j+\mathrm {\mathbf {t}}_\kappa ) \|^2_{\Sigma _g},
(5)
where m ∈PL, l ∈KL, KL is a set of covisible keyframes,
PL are the points seen in KL, KF are other keyframes, Xk is
the set of matched points between a keyframe κ and PL, ρ is
the robust Huber cost function, E(κ, j) is the reprojection
error between the matched 3D points Pj and 2D feature
points pj, π is the projection function, and Σg denotes the
covariance matrix associated with the keypoint’s scale. We
provide more details in Sec. 11.2 of supplementary material.
4. SEGS-SLAM
In this section, we first give an overview of our system.
We then present details of two key innovations: SEPM and
AfME. Finally, we provide an introduction to our training

<!-- page 4 -->
(a) Photo-SLAM
(b) Ours w SEPM only
(c) Ground Truth
Figure 3. Visualization of the Photo-SLAM’s 3D Gaussians and of
our method’s anchor points using only SEPM after 30k iterations.
SEPM enhances the underlying structure of the 3D representation.
loss and FPR. The overview of our SEGS-SLAM is sum-
marized in Fig. 2.
First, the input image stream is pro-
cessed through tracking and geometric mapping process in
Sec. 3.2 to obtain camera poses and point cloud. On one
hand, SEPM voxelizes the point cloud to initialize anchors,
enhancing the exploitation of underlying scene structure.
On the other hand, AfME encodes the camera poses to
model appearance variations. Finally, the rendered images
are supervised by ground-truth images, with the assistance
of FPR, while jointly optimizing the parameters of the 3D
Gaussians and the weights of AfME throughout training.
4.1. Structure-Enhanced Photorealistic Mapping
A common limitation of existing 3DGS-based SLAM sys-
tems is the gradual degradation of the underlying structure
of Gaussians during optimization, limiting the quality of the
rendered results. Our key insight is that preserving strong
scene structure throughout the optimization process is cru-
cial for achieving high-fidelity rendering. Inspired by this,
we make several observations. First, ORB-SLAM3 [3] gen-
erates a point cloud that preserves the scene structure well.
Photo-SLAM [12] initializes 3D Gaussians with the point
cloud, but as shown in the Fig. 3 (a), relying solely on the
latent structure of the point cloud is insufficient, as Gaus-
sians still undergo structural degradation during optimiza-
tion. We further observe that the hierarchical structure in
[21] leverages scene structure, using anchor points to man-
age 3D Gaussians. These anchor points remain fixed during
the optimization process, ensuring that 3D Gaussians retain
the underlying structure of the scene.
Based on this observation, we propose incrementally
voxelizing the point cloud Pk of each keyframe to construct
anchor points, as follows:
  \ labe
l  { e q:voxel} V_k = \{\lfloor \frac {P_k}{\epsilon } \rceil \}\cdot \epsilon , 
(6)
where Vk ∈RN×3 denotes voxel centers, ϵ is the voxel size,
and {·} is the operation to remove redundant points.
Fig. 2 visualizes this process. The ORB-SLAM3 point
cloud (top left) is voxelized into the anchor points (middle),
LV
i: Camera index of training view     j: Camera index of novel view     LV: Learnable Vector      : Embeding vector
1.Training on image of training 
view
1 2 ... n
i
2.1 Training on the left half of each 
image in test set 
2.1 Test on the right half of each 
image in test set 
LV
1
j
2 ... n
j
LV
1
j
2 ... n
j
i
i
(a) Appearance embedding in NeRF-W [23].
MLP
1.Training on image of training view
2.Test on image of novel view w/o additional training
: AfME vector
: Camera pose of training view
MLP
: Camera pose of novel view
: Learned
: Frozen
(b) Our Appearance-from-Motion embedding.
Figure 4. AE [23] and the proposed AfME. The differences be-
tween them are: (1) AE uses image indexes as input, whereas
AfME leverages camera poses. (2) AE adopts learnable vectors
(LV), while AfME utilizes a tiny MLP. (3) Most critically, our
AfME requires no additional training on novel views.
which preserve the underlying structure effectively. In this
way, the structural prior from ORB-SLAM3’s point cloud
and the anchor-based organization are seamlessly fused, en-
hancing the exploitation of the underlying structure. Specif-
ically, the anchor points inherit the strong structural prop-
erties of the point cloud and, due to their fixed positions,
consistently preserve the latent scene structure throughout
training. As shown in the Fig. 3 (b), this effectively pre-
vents structural degradation of Gaussians over training, a
key issue in prior methods. This strategy yields substan-
tial improvements in rendering accuracy, as demonstrated
in our ablation studies. Although conceptually simple, our
approach proves highly effective. After anchor points con-
struction, 3D Gaussians are then generated according to
Eq. (1), Eq. (8), and rendered via Eq. (3).
4.2. Appearance-from-Motion Embedding
To further enhance the rendering quality of SEGS-SLAM,
we observe a drop in the photorealistic mapping quality
when suffering from appearance changes. AE is a proven
solution in [21, 23] for handling such variations.
How-
ever, the limitation of the AE in [23] lies in involving each
ground-truth (left half of the image) in the test set for train-
ing as shown in the mid of Fig. 4 (a). To address this issue,
we propose Appearance-from-Motion embedding (AfME),
which employs a lightweight Multilayer Perceptron (MLP)
Mθa to learn a shared appearance representation. As illus-
trated in Fig. 4 (b), the input of the encoder Mθa is the the
camera pose (R, t). The MLP encodes the pose and out-
puts an embedding vector ℓ(a)
R, t, as shown below:
  \b
m { \ ell }^ {(a)}_{\mathrm {\mathbf {R}},\ \mathrm {\mathbf {t}}} = M_{\theta _a}(\mathrm {\mathbf {R}},\ \mathrm {\mathbf {t}}). \label {eq:embedding} 
(7)
Subsequently, the embedding vector ℓ(a)
R, t is fed into the
color decoder MC. After introducing AfME, the color pre-
diction of 3D Gaussians changes from Eq. (2) to:
  \l a b e l  {eq: c olo r_ml p} \ {c_0, \dot
s ,c_{k-1}\} = M_C(\hat {f}_v,\ \delta _{vc},\ \vec {\mathrm {\mathbf {d}}}_{vc},\ \bm {\ell }^{(a)}_{\mathrm {\mathbf {R}},\ \mathrm {\mathbf {t}}}). 
(8)

<!-- page 5 -->
(a) Same-view renderings with different appearance embedded by AfME.
(b) Renderings from different views correspond to different appearances.
Figure 5. The visualization of AfME controlling appearance. The
rendering viewpoints in the top three images above are same, and
only the input to AfME has been changed. The input poses of the
AfME in the top-row images correspond to those in the bottom-
row images. The results show that only the color and illumination
have changed, while the geometry is fixed.
More details are in Sec. 11.4 of supplementary material. We
choose camera poses as inputs for several reasons: 1) Simi-
lar to image indices, camera poses are unique for each view.
2) Camera poses naturally represent spatial information, en-
abling AfME to predict appearance from spatial context. 3)
Camera poses are more continuous than image indices.
We adopt AfME to encode the scene appearance into the
continuous pose space. Through training on the training set,
AfME learns the mapping between appearance and camera
pose, enabling it to predict the appearance for novel views.
We conduct an experiment to demonstrate this. After train-
ing, we fix ˆfv, δvc, ⃗dvc in the input of Eq. (8) and vary
only the pose {R, t} fed into AfME. As shown in Fig. 5,
the lighting conditions under the same view change consis-
tently with the pose {R, t}, matching the illumination of
corresponding viewpoints. This demonstrates that the illu-
mination conditions can be embedded by AfME effectively.
4.3. Frequency Pyramid Regularization
A minor improvement is the frequency pyramid regulariza-
tion (FPR), which leverages multi-scale frequency repre-
sentation to enhance the reconstruction of high-frequency
details in the scene. To achieve this, we apply bilinear inter-
polation to downsample both the render images Ir and the
ground truth images Ig. Let s ∈S = {s0, s1, . . . , sn}
denote the scale of an image.
We apply a 2D Fast
Fourier Transform (FFT) to obtain the frequency spectra
F(Is
r)(u, v), F(Is
g)(u, v). The loss Lhf is computed as
  & \
m
ath
c
a l 
{
L}_
{hf }
 =\sum _{ s  \
in \mat hc
al  
{S}
}  
\frac { 1} { \mathc al  {N}}
 \lam bda  _ s \ sum _{u, v} \left | F_{hf, r}^{s}(u, v) - F_{hf, g}^{s}(u, v) \right | \label {eq:loss_fre}, \\ &F_{hf, i}^{s}(u, v) = H_{hf}(u, v) \cdot \mathcal {F}(I_i^{s})(u, v),\ i\in \{r, g\} ,
(10)
where F s
hf,r(u, v), F s
hf,g(u, v) is the high-frequency ex-
tracted by a high-pass filter Hhf(u, v), N = HW denotes
the image size, and λs represents the weight of each scale.
More details are in Sec. 8 of supplementary material.
4.4. Losses Design
The optimization of the learnable parameters, the MLP Mα,
Mc, Mq, Ms, and Mθa, are achieved by minimizing the L1
loss L1, SSIM term [38] LSSIM, frequency regularization
Lhf, and volume regularization [20] Lvol between the ren-
dered images and the ground truth images, denoted as
  \mathca l {L} = (1-\lambda ) \mathcal {L}_1 + \lambda (1-\mathcal {L}_{\text {SSIM}}) +\lambda _{\text {vol}}\mathcal {L}_{\text {vol}} + \lambda _{hf} \mathcal {L}_{hf}. \label {eq:loss} (11)
Following [21], we also incorporate Lvol.
5. Experiment
5.1. Experiment Setup
Implementation. Our SEGS-SLAM is fully implemented
using the LibTorch framework with C++ and CUDA. The
training and rendering of 3D Gaussians involves three key
modules: SEPM, AfME, and FPR, operating as a parallel
thread alongside the localization and geometric mapping
process. SEGS-SLAM trains the 3D Gaussians using only
keyframe images, point clouds, and poses, where keyframes
are selected based on co-visibility. In each iteration, SEGS-
SLAM randomly samples a viewpoint from the current set
of keyframes. We use the images and poses of keyframes
as the training set, while the remaining images and poses
serve as the test set. Moreover, following FreGS [45], we
activate FPR once the structure of anchor points stabilizes
and terminate it based on the completion of anchor point
densification. The scale level of FPR is set to 3. Except
for the non-open-source GS-SLAM [40], all methods com-
pared in this paper are run on the same machine using their
official code. The machine is equipped with an NVIDIA
RTX 4090 GPU and a Ryzen 5995WX CPU. By default, our
method runs for 30K iterations. The voxel size ϵ is 0.001 m.
For Eq. (11), we set λ = 0.2, λvol = 0.01, λhf = 0.01.
Baselines. We first list the baseline methods used to eval-
uate photorealistic mapping.
For monocular and stereo
cameras, we compare our method with Photo-SLAM [12],
MonoGS [24], and Photo-SLAM-30K. For RGB-D cam-
eras, we additionally include comparisons with RTG-
SLAM [27], GS-SLAM [40], SplaTAM [14], SGS-SLAM
[17], and GS-ICP SLAM [9], all of which represent SOTA
SLAM methods based on 3D-GS. To ensure fairness, we set
the maximum iteration limit to 30K for all methods, follow-
ing the original 3D-GS [15]. Photo-SLAM-30K refers to
Photo-SLAM [12] with a fixed iteration count of 30K. For
camera pose estimation, we also compare ours with ORB-
SLAM3 [3] and DROID-SLAM [35].

<!-- page 6 -->
Datasets (Camera)
Replica (RGB-D)
TUM RGB-D (RGB-D)
Method
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
MonoGS [24]
36.81
0.964
0.069
24.11
0.800
0.231
Photo-SLAM [12]
35.50
0.949
0.056
20.99
0.736
0.213
Photo-SLAM-30K
36.94
0.952
0.040
21.73
0.757
0.186
RTG-SLAM [27]
32.79
0.918
0.124
16.47
0.574
0.461
GS-SLAM∗[40]
34.27
0.975
0.082
-
-
-
SplaTAM [14]
33.85
0.936
0.099
21.41
0.764
0.265
SGS-SLAM [17]
33.96
0.969
0.099
-
-
-
GS-ICP SLAM [9]
37.14
0.968
0.045
17.81
0.642
0.361
Ours
39.42
0.975
0.021
26.03
0.843
0.107
Table 1. Quantitative evaluation of our method compared to SOTA methods for RGB-D camera on Replica and TUM RGB-D datasets.
Best results are marked as best score , second best score and third best score . GS-SLAM∗denotes the result of GS-SLAM is taken
from [40], all others are obtained in our experiments. ’-’ denotes the system does not provide valid results.
(a) GS-ICP SLAM [9]
(b) SplaTAM [14]
(c) RTG-SLAM [27]
(d) Ours
(e) Ground Truth
Figure 6. We show comparisons of ours to SOTA methods for RGB-D camera. The top scene is office2 from the Replica datasets, and the
bottom is fr3/office from TUM RGB-D datasets. Non-obvious differences in quality are highlighted by insets.
Metrics. We follow the evaluation protocol of MonoGS
[24] to assess both camera pose estimation and novel view
synthesis. For camera pose estimation, we report the root
mean square error (RMSE) of the absolute trajectory error
(ATE) [8] for all frames. For photorealistic mapping, we
report standard rendering quality metrics, including PSNR,
SSIM, and LPIPS [46]. To evaluate the photorealistic map-
ping quality, we only calculate the average metrics over
novel views for all methods. We report the average across
five runs for all methods. To ensure fairness, no training
views are included in the evaluation, and for all RGB-D
SLAM methods, no masks are applied to either the ren-
dered or ground truth images during metric calculation.
As a result, the reported metrics for Photo-SLAM [12] are
slightly lower than those in the original paper, as they aver-
ages both novel and training views. Similarly, the metrics
of SplaTAM [14], SGS-SLAM [17], and GS-ICP SLAM [9]
are slightly lower than reported, as the original methods use
a mask to exclude outliers for both the rendered and ground
truth images based on anomalies in the depth image.
Datasets. Following [9, 11, 12, 14, 17, 24, 27, 28, 34, 40],
we evaluate all methods on all sequences of the Replica
dataset [31] for monocular and RGB-D cameras. Follow-
ing [9, 12, 24, 27, 34, 40, 50], we use the fr1/desk, fr2/xyz,
and fr3/office sequences of the TUM RGB-D dataset [32]
for monocular and RGB-D cameras. Following [12], we
use the MH01, MH02, V101, and V201 sequences of the
EuRoC MAV dataset [2] for stereo cameras.
5.2. Results Analysis
Camera Tracking Accuracy.
As shown in Tab. 3, our
method demonstrates competitive accuracy in tracking for
monocular, stereo, and RGB-D cameras when compared
with SOTA methods. This highlights the advantage of in-
direct visual SLAM in terms of localization accuracy.
Novel View synthesis. The quantitative rendering results
for novel views in RGB-D scenarios are shown in Tab. 1,
where SEGS-SLAM significantly outperforms comparison
methods, achieving the highest average rendering quality
on both TUM RGB-D and Replica datasets.
In the top
of Fig. 6, it is evident that for the Replica dataset, only
our method can accurately recover the contours of edge re-
gions. The TUM RGB-D dataset presents a greater chal-
lenge compared with the Replica dataset, with highly clut-
tered scene structures and substantial lighting variations.
GS-ICP SLAM [9], a leading RGB-D SLAM method based

<!-- page 7 -->
Datasets (Camera)
Replica (Mono)
TUM RGB-D (Mono)
EuRoC (Stereo)
method
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
MonoGS [24]
28.34
0.878
0.256
21.00
0.705
0.393
22.60
0.789
0.274
Photo-SLAM [12]
33.60
0.934
0.077
20.17
0.708
0.224
11.90
0.409
0.439
Photo-SLAM-30K
36.08
0.947
0.054
21.06
0.733
0.186
11.77
0.405
0.430
Ours
37.96
0.964
0.037
25.17
0.825
0.122
23.64
0.791
0.182
Table 2. Quantitative evaluation of our method compared to SOTA methods for Monocular (Mono) and Stereo cameras on Replica, TUM
RGB-D, and EuRoC MAV datasets. Best results are marked as best score and second best score .
(a) MonoGS [24]
(b) Photo-SLAM [12]
(c) Photo-SLAM-30k
(d) Ours
(e) Ground Truth
Figure 7. We show comparisons of ours to SOTA methods for Monocular and Stereo cameras. The top scene is room1 from the Replica
dataset, and the bottom is V201 from the EuRoC MAV dataset. Non-obvious differences in quality are highlighted by insets.
Camera Type
RGB-D
Monocular
Stereo
Datasets
Replica TUM R
Avg.
Replica TUM R
Avg.
EuRoC
Method
RMSE ↓RMSE ↓RMSE ↓RMSE ↓RMSE ↓RMSE ↓RMSE ↓
ORB-SLAM3 [3]
1.780
2.196
1.988
51.744
46.004
48.874
10.907
DRIOD-SLAM [35] 74.264
74.216
74.24
76.600
1.689
39.145
1.926
MonoGS [24]
0.565
1.502
1.033
37.054
4.009
63.437
49.241
Photo-SLAM [12]
0.582
1.870
1.226
0.930
1.539
1.235
11.023
RTG-SLAM [27]
0.191
0.985
0.581
-
-
-
-
GS-SLAM∗[40]
0.500
3.700
2.100
-
-
-
-
SplaTAM [14]
0.343
4.215
2.279
-
-
-
-
SGS-SLAM [17]
0.365
-
-
-
-
-
-
GS-ICP SLAM [9]
0.177
2.921
1.549
-
-
-
-
Ours
0.430
1.528
0.979
0.833
1.505
1.169
7.462
Table 3. Camera tracking result on Replica, TUM RGB-D (TUM
R), and EuRoC MAV datasets for Monocular, stereo, and RGB-D
cameras. RMSE of ATE (cm) is reported. The best results are
marked as best score , second best score and third best score .
’-’ denotes that the system does not provide valid results.
on 3D-GS, achieves the second-highest rendering accuracy
on the Replica dataset. However, it relies heavily on depth
images. As shown in the bottom of Fig. 6, GS-ICP SLAM
[9] and other methods perform poorly on the TUM RGB-D
dataset. Our SEGS-SLAM better reconstructs scene struc-
ture and lighting variations, benefiting from our SEPM and
the AfME.
Tab. 2 presents quantitative rendering results for monoc-
ular scenarios, where SEGS-SLAM surpasses other meth-
ods. Notably, SEGS-SLAM continues to significantly out-
perform comparison methods on the TUM RGB-D dataset.
Importantly, compared with RGB-D scenarios, MonoGS
[24] experiences a sharp decline. The top of Fig. 7 fur-
ther demonstrates that on the Replica dataset, our method
effectively models high-frequency details more realistically
in regions such as the edge of the wall.
Moreover, our method remains effective in stereo sce-
narios. The corresponding quantitative results for realistic
mapping are recorded in Tab. 2, where our method achieves
the highest rendering quality, surpassing the current SOTA
method, MonoGS [24]. As shown in the bottom of Fig. 7,
our approach better reconstructs the global structure and lo-
cal details of the scene. We highlight that a key factor en-
abling our method to achieve superior performance across
different camera types and datasets is the proposed SEPM.
Its enhancement to rendering quality is highly generaliz-
able, as further demonstrated in ablation studies.
5.3. Ablation Studies
Structure-Enhanced photorealistic mapping. To evalu-
ate the impact of SEPM on photorealistic mapping metrics,
we additionally train two variants of our method: one with-
out SEPM, AfME, and FPR, and another without AfME and
FPR. The variant without SEPM, AfME, and FPR directly
uses the original 3D-GS [15]. As shown in Tab. 4, rows
(1) and (2), introducing SEPM consistently yields signifi-
cant improvements in rendering quality across all scenes.
This validates the effectiveness of enhancing the exploita-
tion of latent structure.
Moreover, it demonstrates that
SEPM has strong generalization across diverse camera in-

<!-- page 8 -->
Tum
Replica
EuRoC
Ours w/o FPR, AfME,
SEPM
Ours w/o FPR, AfME
Ground Truth
Figure 8. Ablation of SEPM. The figure presents qualitative re-
sults on the TUM, Replica, and EuRoC datasets. It is evident that,
across all three datasets, incorporating SEPM enables the recon-
struction of more complete and accurate structure.
puts and datasets, inspiring future research. Notably, with
only SEPM, our rendering metrics surpass existing SOTA
methods. To further illustrate the benefits of SEPM, Fig. 8
presents visual comparisons, where a more complete and
accurate scene structure leads to superior rendering quality.
Appearance-from-Motion embedding.
To evaluate the
impact of the proposed AfME, we train an additional model
for our method without AfME. In the rows (4) and (5) of
Tab. 4, our full method (5) outperforms the model without
AfME (4) in terms of PSNR scores. As shown in the top of
Fig. 9, the AfME effectively predicts the lighting conditions
of novel views. On the Replica dataset, the improvements
from AfME are relatively modest.
Replica is an easier
dataset, in which PSNR already exceeds 37 without AfME,
indicating that scene is well-reconstructed. Although incor-
porating AfME still yields a PSNR gain, the benefits are
much more pronounced on the challenging TUM dataset,
as clearly observed in both Fig. 9 and Tab. 4.
Frequency pyramid regularization. To evaluate the ef-
fect of the proposed FPR on photorealistic mapping met-
rics, we train an additional model for our method without
FPR. As shown in Tab. 4, rows (3) and (5), our full method
(5) surpasses the model without FPR (3) in terms of PSNR
scores. Additionally, after applying the FPR, the model ren-
Ours w/o AfME
Ours
Ground Truth
Ours w/o FPR
Ours
Ground Truth
Figure 9. Ablation of AfME (Top) and FPR (Bottom). It is evident
that with the introduction of AfME, the lighting conditions at novel
views are accurately predicted, indicating the effectiveness of our
AfME. The result without AfME is darker than the ground truth.
Camera type
RGB-D
Mono
Stereo
Datasets
Replica
TUM R
Replica
TUM R
EuRoC
#
Method
PSNR ↑
PSNR ↑
PSNR ↑
PSNR ↑
PSNR ↑
(1)
w/o FPR, AfME, SEPM
36.07
21.73
35.46
21.10
11.76
(2)
w/o FPR, AfME
38.98
24.20
36.31
23.54
22.91
(3)
w/o FPR
39.18
25.04
36.44
24.91
23.52
(4)
w/o AfME
39.12
24.66
37.48
23.69
22.99
(5)
Ours
39.42
26.03
37.96
25.17
23.64
Table 4. Ablation Study on the key components (1) - (5). The
best results are marked as best score .
ders finer details in highly textured regions, as demonstrated
by the curtain at the bottom of Fig. 9.
5.4. Limitations
One limitation of our method is that a poorly structured
point cloud leads to a decline in photorealistic mapping
quality. Additionally, while our method achieves real-time
tracking and rendering at 17 and 400 FPS, respectively, it
exhibits reduced rendering speed due to the increased num-
ber of 3D Gaussians used to model high-frequency details.
Currently, AFME is only capable of handling static scenes.
By encoding more complex inputs, it can tackle more so-
phisticated dynamic scenarios.
6. Conclusion
We propose a novel SLAM system with progressively re-
fined 3D-GS, termed SEGS-SLAM. Experimental results
show that our method surpasses SOTA methods in render-
ing quality across monocular, stereo, and RGB-D datasets.
We demonstrate that by enhancing the utilization of the un-
derlying scene structure, SEPM improves the visual quality
of rendering. Furthermore, our proposed AfME and FPR
effectively predict the appearance of novel views and refine
the scene details, respectively.

<!-- page 9 -->
Acknowledgements: This work is supported by the Na-
tional Natural Science Foundation of China under Grant
62233011.
References
[1] Michael Bloesch, Jan Czarnowski, Ronald Clark, Stefan
Leutenegger, and Andrew J. Davison. Codeslam — learning
a compact, optimisable representation for dense visual slam.
In IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2018. 2
[2] Michael Burri, Janosch Nikolic, Pascal Gohl, Thomas
Schneider, Joern Rehder, Sammy Omari, Markus W Achte-
lik, and Roland Siegwart. The euroc micro aerial vehicle
datasets. The International Journal of Robotics Research, 35
(10):1157–1163, 2016. 1, 6
[3] Carlos Campos, Richard Elvira, Juan J. G´omez Rodr´ıguez,
Jos´e M. M. Montiel, and Juan D. Tard´os. Orb-slam3: An
accurate open-source library for visual, visual–inertial, and
multimap slam.
IEEE Transactions on Robotics, 37(6):
1874–1890, 2021. 2, 3, 4, 5, 7
[4] Chi-Ming Chung, Yang-Che Tseng, Ya-Ching Hsu, Xiang-
Qian Shi, et al. Orbeez-slam: A real-time monocular visual
slam with orb features and nerf-realized mapping. In 2023
IEEE International Conference on Robotics and Automation,
pages 9400–9406, 2023. 2
[5] Tianchen Deng, Guole Shen, Tong Qin, Jianyu Wang, Wen-
tao Zhao, Jingchuan Wang, Danwei Wang, and Weidong
Chen. Plgslam: Progressive neural scene represenation with
local to global bundle adjustment. In IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 19657–
19666, 2024. 2
[6] Jakob Engel, Thomas Sch¨ops, and Daniel Cremers.
Lsd-
slam: Large-scale direct monocular slam. In European Con-
ference on Computer Vision, pages 834–849, 2014. 2
[7] Jakob Engel, Vladlen Koltun, and Daniel Cremers. Direct
sparse odometry. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 40(3):611–625, 2018. 2
[8] Michael Grupp.
evo:
Python package for the evalua-
tion of odometry and slam.
https://github.com/
MichaelGrupp/evo, 2017. 6
[9] Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. Rgbd gs-icp
slam. In European Conference on Computer Vision, pages
180–197, 2024. 1, 2, 5, 6, 7, 4
[10] Jiarui Hu, Mao Mao, Hujun Bao, Guofeng Zhang, and
Zhaopeng Cui. Cp-slam: Collaborative neural point-based
slam system. In Advances in Neural Information Processing
Systems, pages 39429–39442, 2023. 2
[11] Jiarui Hu, Xianhao Chen, Boyin Feng, Guanglin Li,
Liangjing Yang, Hujun Bao, Guofeng Zhang, and Zhaopeng
Cui.
Cg-slam: Efficient dense rgb-d slam in a consistent
uncertainty-aware 3d gaussian field. In European Confer-
ence on Computer Vision, pages 93–112, 2024. 1, 2, 6
[12] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung.
Photo-slam: Real-time simultaneous localization and photo-
realistic mapping for monocular stereo and rgb-d cameras.
In IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 21584–21593, 2024. 1, 2, 3, 4, 5, 6, 7, 8,
9, 10
[13] Mohammad Mahdi Johari, Camilla Carta, and Franc¸ois
Fleuret. Eslam: Efficient dense slam system based on hybrid
representation of signed distance fields. In IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
17408–17419, 2023. 2
[14] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat track & map 3d gaussians
for dense rgb-d slam. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 21357–21366, 2024.
1, 2, 5, 6, 7, 4
[15] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3D Gaussian Splatting for Real-Time
Radiance Field Rendering. ACM Transactions on Graphics,
42(4):1–14, 2023. 1, 2, 3, 5, 7
[16] Georg Klein and David Murray. Parallel tracking and map-
ping for small ar workspaces. In 2007 6th IEEE and ACM
International Symposium on Mixed and Augmented Reality,
pages 225–234, 2007. 2
[17] Mingrui Li, Shuhong Liu, Heng Zhou, Guohao Zhu, Na
Cheng, Tianchen Deng, and Hongyu Wang. Sgs-slam: Se-
mantic gaussian splatting for neural dense slam. In European
Conference on Computer Vision, pages 163–179, 2025. 1, 2,
5, 6, 7, 4
[18] Lahav Lipson and Jia Deng. Multi-session slam with dif-
ferentiable wide-baseline pose optimization. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 19626–19635, 2024. 2
[19] Lorenzo Liso,
Erik Sandstr¨om,
Vladimir Yugay,
Luc
Van Gool, and Martin R. Oswald. Loopy-slam: Dense neural
slam with loop closures. In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 20363–20373,
2024. 2
[20] Stephen Lombardi,
Tomas Simon,
Gabriel Schwartz,
Michael Zollhoefer, Yaser Sheikh, and Jason Saragih. Mix-
ture of volumetric primitives for efficient neural rendering.
ACM Transactions on Graphics, 40(4), 2021. 5
[21] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering. In IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
20654–20664, 2024. 2, 3, 4, 5, 1
[22] Yunxuan Mao, Xuan Yu, Zhuqing Zhang, Kai Wang, Yue
Wang, Rong Xiong, and Yiyi Liao.
Ngel-slam: Neural
implicit representation-based global consistent low-latency
slam system.
In 2024 IEEE International Conference on
Robotics and Automation, pages 6952–6958, 2024. 2
[23] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Saj-
jadi, Jonathan T. Barron, Alexey Dosovitskiy, and Daniel
Duckworth.
Nerf in the wild: Neural radiance fields for
unconstrained photo collections. In IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 7210–
7219, 2021. 2, 4
[24] Hidenobu Matsuki, Riku Murai, Paul H.J. Kelly, and An-
drew J. Davison. Gaussian splatting slam.
In IEEE/CVF

<!-- page 10 -->
Conference on Computer Vision and Pattern Recognition,
pages 18039–18048, 2024. 1, 2, 3, 5, 6, 7, 4, 8, 9, 10
[25] Moustafa Meshry, Dan B. Goldman, Sameh Khamis, Hugues
Hoppe, Rohit Pandey, Noah Snavely, and Ricardo Martin-
Brualla. Neural rerendering in the wild. In IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, 2019.
2
[26] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In European Conference on Computer Vision, pages
405–421, 2020. 1
[27] Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang,
Jingdong Wang, and Kun Zhou. Rtg-slam: Real-time 3d re-
construction at scale using gaussian splatting. In ACM SIG-
GRAPH, 2024. 1, 2, 5, 6, 7, 4
[28] Erik Sandstr¨om, Yue Li, Luc Van Gool, and Martin R. Os-
wald.
Point-slam: Dense neural point cloud-based slam.
In IEEE/CVF International Conference on Computer Vision,
pages 18433–18444, 2023. 2, 6
[29] Erik Sandstr¨om, Kevin Ta, Luc Van Gool, and Martin R.
Oswald. Uncle-slam: Uncertainty learning for dense neural
slam. In IEEE/CVF International Conference on Computer
Vision Workshops, pages 4537–4548, 2023. 2
[30] Johannes
Lutz
Sch¨onberger
and
Jan-Michael
Frahm.
Structure-from-motion revisited.
In Conference on Com-
puter Vision and Pattern Recognition, 2016. 2, 3
[31] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J Engel, et al. The replica
dataset: A digital replica of indoor spaces. arXiv preprint
arXiv:1906.05797, 2019. 6
[32] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
Burgard, and Daniel Cremers. A benchmark for the evalua-
tion of rgb-d slam systems. In IEEE/RSJ International Con-
ference on Intelligent Robots and Systems, pages 573–580,
2012. 1, 6
[33] Edgar
Sucar,
Kentaro
Wada,
and
Andrew
Davison.
Nodeslam: Neural object descriptors for multi-view shape
reconstruction. In 2020 International Conference on 3D Vi-
sion, pages 949–958, 2020. 2
[34] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J. Davi-
son. imap: Implicit mapping and positioning in real-time.
In IEEE/CVF International Conference on Computer Vision,
pages 6229–6238, 2021. 2, 6
[35] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam
for monocular, stereo, and rgb-d cameras.
In Advances
in Neural Information Processing Systems, pages 16558–
16569, 2021. 2, 5, 7
[36] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch vi-
sual odometry. In Advances in Neural Information Process-
ing Systems, pages 39033–39051, 2023. 2
[37] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Co-
slam: Joint coordinate and sparse parametric encodings for
neural real-time slam. In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 13293–13302,
2023. 2
[38] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing, 13(4):
600–612, 2004. 5
[39] Yueming Xu, Haochen Jiang, Zhongyang Xiao, Jianfeng
Feng, and Li Zhang. Dg-slam: Robust dynamic gaussian
splatting slam with hybrid pose optimization. In In Advances
in Neural Information Processing Systems, 2024. 1, 2
[40] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d
gaussian splatting. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 19595–19604, 2024.
1, 2, 5, 6, 7, 4
[41] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian
Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and
mapping with voxel-based neural implicit representation. In
2022 IEEE International Symposium on Mixed and Aug-
mented Reality, pages 499–507, 2022. 2
[42] Minghao Yin, Shangzhe Wu, and Kai Han.
Ibd-slam:
Learning image-based depth fusion for generalizable slam.
In IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10563–10573, 2024. 2
[43] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Os-
wald. Gaussian-slam: Photo-realistic dense slam with gaus-
sian splatting. arXiv preprint arXiv:2312.10070, 2023. 2
[44] Hongjia Zhai, Gan Huang, Qirui Hu, Guanglin Li, Hujun
Bao, and Guofeng Zhang.
Nis-slam: Neural implicit se-
mantic rgb-d slam for 3d consistent scene understanding.
IEEE Transactions on Visualization and Computer Graph-
ics, pages 1–11, 2024. 2
[45] Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, and
Eric Xing. Fregs: 3d gaussian splatting with progressive fre-
quency regularization. In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 21424–21433,
2024. 2, 5, 1, 3
[46] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2018. 6
[47] Wei Zhang, Tiecheng Sun, Sen Wang, Qing Cheng, and Nor-
bert Haala. Hi-slam: Monocular real-time dense mapping
with hybrid implicit fields. IEEE Robotics and Automation
Letters, 9(2):1548–1555, 2024. 2
[48] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo
Poggi. Go-slam: Global optimization for consistent 3d in-
stant reconstruction. In IEEE/CVF International Conference
on Computer Vision, pages 3727–3737, 2023. 2
[49] Siting Zhu, Guangming Wang, Hermann Blum, Jiuming Liu,
Liang Song, Marc Pollefeys, and Hesheng Wang. Sni-slam:
Semantic neural implicit slam.
In IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21167–
21177, 2024. 2
[50] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
In IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 12786–12796, 2022. 6
[51] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui,
Martin R. Oswald, Andreas Geiger, and Marc Pollefeys.

<!-- page 11 -->
Nicer-slam: Neural implicit scene encoding for rgb slam. In
International Conference on 3D Vision, pages 42–52, 2024.
2

<!-- page 12 -->
SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with
Appearance Embedding
Supplementary Material
Number of anchor 
points: 61785
Number of anchor 
points: 107152
(a) Ours w/o FPR
(b) Ours
(c) Ground Truth
Figure 10. Visualization of anchor points after 30K iterations. In-
creasing the number of Gaussians along edges improves the ren-
dering quality. In the figure, 3D Gaussians are k = 10 times the
number of anchor points.
7. Overview
The supplementary material is organized as follows: (1)
Sec. 8 introduces more details of FPR. (2) Sec. 9 presents
additional ablation studies.
(3) Sec. 10 provides real-
time performance for all methods.
(4) Sec. 11 pro-
vides additional implementation details, including the de-
tailed pipeline for localization and geometry mapping
(Sec. 11.2), the MLP architecture used for structured 3D
Gaussians (Sec. 11.3), the MLP structure for Appearance-
from-Motion Embedding (Sec. 11.4), and anchor point re-
finement (Sec. 8). (6) Sec. 12 presents quantitative results
for each scene and includes more comparative renderings.
8. Details of FPR
In scenes with simple structures, our structured 3D Gaus-
sians can effectively model both structure and appearance
changes. However, we observe that structured 3D Gaus-
sians perform poorly in rendering high-frequency details,
such as object edges and areas with complex textures.
Hence, we propose the frequency pyramid regularization
(FPR) technique, which effectively leverages multi-scale
frequency spectra. Here, we introduce the frequency pyra-
mid to improve the consistency of rendering details for
the same object across varying viewpoint distances. Un-
like FreGS [45], we leverage only high-frequency infor-
mation, as low-frequency components typically represent
scene structure, which is already effectively captured by our
structured 3D Gaussians as shown in Tab. 6.
The primary effect of FPR is to guide the densifica-
tion of anchor points. Specifically, when the average gra-
dient of all Gaussians within a voxel exceeds a threshold
τg, a new anchor point is added at the center of the voxel.
Consequently, if a high-frequency region in the scene ex-
hibits a substantial discrepancy between output rendering
and ground truth, the total loss includes the frequency reg-
ularization Lhf increases, pushing the average gradient be-
Camera type
RGB-D
Mono
Stereo
Datasets
Replica
TUM R
Replica
TUM R
EuRoC
Scale level
PSNR ↑
PSNR ↑
PSNR ↑
PSNR ↑
PSNR ↑
1
39.14
25.82
37.65
23.83
23.58
2
39.29
25.53
37.55
23.77
23.57
4
39.34
26.02
37.51
24.75
23.42
3 (Ours )
39.42
26.03
37.96
25.17
23.64
Table 5. Ablation Study on the scale level of FPR.
Camera type
RGB-D
Mono
Stereo
Datasets
Replica
TUM R
Replica
TUM R
EuRoC
Metric
PSNR ↑
PSNR ↑
PSNR ↑
PSNR ↑
PSNR ↑
low freq.
38.85
25.76
36.79
24.79
23.37
low & high freq.
39.23
25.84
37.72
24.84
23.27
high freq. (Ours)
39.42
26.03
37.96
25.17
23.64
Table 6. Ablation study on the low frequency component of FPR.
yond τg. Then, more new anchors are added in the region,
and scene edges become sharper, as shown in Fig. 10. Thus,
the high-frequency details in the scene are refined by FPR.
9. Additional Ablation Studies
Scale level of FPR. We think that multiple scales improve
the consistency under varying observation distances. The
scale level of FPR is set to 3. Results are shown in Tab. 5.
The low-frequency component in FPR. In our experi-
ments, we find that the low-frequency component of FPR
conflicts with structured 3D Gaussians, resulting in degra-
dation. The best result is using only the high-frequent com-
ponent in FPR, as shown in Tab. 6.
Replacement of key components. We train two additional
models: one replaces AfME with the appearance embed-
dings (AE) from Scaffold-GS [21], and another replaces
FPR with the single-scale frequency regularization (SFR).
Our AfME differs from AE in two primary ways: 1) AfME
uses camera poses as input, whereas AE uses camera in-
dices; 2) AfME employs an MLP network structure, while
AE utilizes an embedding layer. SFR refers to using only
the original-scale image frequencies in Eq. (9). The dis-
tinction between FPR and SFR lies in the use of multi-scale
image frequencies in FPR. As shown in Tab. 4, rows (1),
(2), and (3), our full method (3) achieves the highest PSNR
scores.
This demonstrates that, compared with AE, our
AfME is more effective in predicting appearance variations
across a wide range of novel views, thus avoiding additional
training on the test set. On the other hand, it also highlights
that by introducing the frequency pyramid, the model main-
tain consistency in scene details across varying viewpoint
distances, leading to superior rendering quality.

<!-- page 13 -->
Camera type
RGB-D
Mono
Stereo
Datasets
Replica
TUM R
Replica
TUM R
EuRoC
#
Method
PSNR ↑
PSNR ↑
PSNR ↑
PSNR ↑
PSNR ↑
(1)
replace AfME with AE
38.22
19.56
36.09
19.33
18.39
(2)
replace FPR with SFR
39.14
25.82
37.65
23.83
23.58
(3)
Ours
39.42
26.03
37.96
25.17
23.64
Table 7. Ablation Study on the replacement of key components.
Metric (RGB-D)
MonoGS Photo-SLAM (-30K) RTG-SLAM SplaTAM SGS-SLAM GS-ICP SLAM
Ours
Rendering FPS ↑
706
1562 (1439)
447
531
486
630
400
Tracking FPS ↑
1.33
30.30 (30.87)
17.24
0.15
0.14
30.32
17.18
Mapping Time ↓
37m40s
1m20s (6m32s)
12m03s
3h45m
4h05m
1m32s
11m14s
Table 8. Real-time performance.
10. Real-time performance
Our method, following Photo-SLAM, employs two paral-
lel threads: Localization & Geometry Mapping and 3D-GS
Mapping. We note that only tracking and rendering are real-
time. The runtime of all methods is provided in Tab. 8.
11. Implementation details
11.1. System Overview
Our system comprises two main modules: localization and
geometry mapping and progressively refined 3D Gaussian
splatting (3D-GS). In our implementation, these two mod-
ules run in separate threads. The localization and geome-
try mapping module focuses on camera pose estimation and
scene point cloud mapping. The progressively refined 3D-
GS module takes the estimated keyframe poses and point
clouds from the localization and geometry mapping mod-
ule. Then the module incrementally completes the photore-
alistic mapping of the scene.
11.2. Localization and Geometry Mapping
In our implementation, the localization and geometric map-
ping module consists of three main threads: tracking, local
mapping, and loop closing, along with an on-demand thread
for global bundle adjustment (BA). Specifically, the track-
ing thread performs a motion-only BA to optimize camera
poses. The local mapping thread optimizes keyframe poses
and map point clouds within a local sliding window via lo-
cal BA. Lastly, the loop closing thread continuously checks
for loop closures. If a loop is detected, a global BA is trig-
gered to jointly optimize the camera poses of all keyframes
and all points of the scene.
Motion-only BA. We optimize the camera orientation R ∈
SO(3) and position t ∈R3 through motion-only BA. The
camera poses (Rι, tι) are optimized by minimizing the re-
projection error between the matched 3D points Pι ∈R3
and 2D feature points pι within a sliding window:
  \{ \ma t
h
rm 
{\math
bf {R
}}_\i o ta ,\m a thrm 
{\mathbf {t}}_\iota \}&= \sum _{\iota \in \mathcal {X}} \mathop {\mathrm {argmin}}\limits _{\mathrm {\mathbf {R}}_\iota ,\mathrm {\mathbf {t}}_\iota } \rho (\|p_\iota - \pi (\mathrm {\mathbf {R}_\iota }\mathrm {\mathbf {P}}_\iota +\mathrm {\mathbf {t}}_\iota ) \|^2_{\Sigma _g}) (12)
where X represent the set of all matches, Σg denote the
covariance matrix associated with the keypoint’s scale, π
is the projection function, and ρ is the robust Huber cost
function.
Local BA. We perform a local BA by optimizing a set of
covisible keyframes KL alone with the set of points PL ob-
served in those keyframes as follows:
  \{ \ma thr m  {\mat
hbf {P}}
_
m,\math
r
m {\
mathbf  {R }}_l
,\ma th r m { \ mathbf  {t}}_
l\} &= \mathop {\mathrm {argmin}}\limits _{\mathrm {\mathbf {P}}_m,\mathrm {\mathbf {R}}_l,\mathrm {\mathbf {t}}_l} \sum _{\kappa \in \mathcal {K}_L\cup \mathcal {K}_F} \sum _{j\in \mathcal {X}_k} \rho (E(\kappa ,j)) \\ E(\kappa , j) &= \|\mathrm {\mathbf {p}}_j - \pi (\mathrm {\mathbf {R}}_\kappa \mathrm {\mathbf {P}}_j+\mathrm {\mathbf {t}}_\kappa ) \|^2_{\Sigma _g}
(14)
where m ∈PL, l ∈KL, KF are all other keyframes, Xk is
the set of matches between keypoints in a keyframe κ and
points in PL.
Global BA. Global BA is a special case of local BA, where
all keyframes and map points are included in the optimiza-
tion, except the origin keyframe, which is kept fixed to pre-
vent gauge freedom.
11.3. Structured 3D Gaussians
MLPs as feature decoders. Following [21], we employ
four MLPs as decoders to derive the parameters of each 3D
Gaussian, including the opacity MLP Mα, the color MLP
MC, and the covariance MLP Mq, Ms. Each MLP adopts a
linear layer followed by ReLU and another linear layer. The
outputs are activated by their respective activation functions
to obtain the final parameters of each 3D Gaussian. The
detailed architecture of these MLPs is illustrated in Fig. 11.
In our implementation, the hidden layer dimensions of all
MLPs are set to 32.
• For opacity, we use Tanh(·) to activate the output of the
final linear layer. Since the opacity values of 3D Gaus-
sians are typically positive, we constrain the value range
to [0, 1) to ensure valid outputs.
• For color, we use Sigmoid function to activate the output
of the final linear layer, which constrains the color value
into a range of [0, 1).
• For rotation, following 3D-GS [15], we employ a nor-
malization to activate the output of the final linear layer,
ensuring the validity of the quaternion representation for
rotation.
• For scaling, a Sigmoid function is applied to activate the
output of the final linear layer. Finally, the scaling of each
3D Gaussian is determined by adjusting the scaling lv of
its associated anchor based on the MLP’s output, as for-
mulated below:
  \{ s _ 0 , \dots  ,s_ {k-1 }\} = M_s ( \hat {f}_v,\ \delta _{vc},\ \vec {\mathrm {\mathbf {d}}}_{vc}) \cdot l_v \label {eq:scaling_mlp1}
(15)
11.4. Appearance-from-Motion Embedding
MLP as feature encoder. For AfME, we employ an MLP
Mθa as the encoder. The input to this MLP is the camera
pose corresponding to each image. The MLP Mθa extracts

<!-- page 14 -->
32
1
1
32
3
1
ReLU
Tanh
vc

vc
d
Opacity
32
3
3
32
3
1
ReLU
Sigmoid
vc

vc
d
Color
32
3
32
3
1
ReLU
vc

vc
d
Scale
7
4
Quaternion
Sigmoid· vl
Normalization
Na
( )
a
R,t
( )
a
R,t
Context feature
Appearance-from-Motion embedding vector
ˆ
vf
ˆ
vf
ˆ
vf
ˆ
vf
vc
d
Viewing direction
vc

Relative distance
Na     Embedding dimension
Input
Figure 11. Structure of the MLPs Mα, MC, Ms, and Mq. For
each anchor, we use these MLPs to predict the opacity, color, scale,
and quaternion of k 3D Gaussian. The inputs to the MLPs include
the relative distance δvc and the viewing direction ⃗dvc between
camera position tc and an anchor point. Since Na is not a fixed
parameter, its specific value is not included in the figure.
pose features and feeds these features to the color decoder
MC. The MLP Mθa adopts a structure of a linear layer
followed by a linear activation function, as illustrated in
Fig. 12. The entire pipeline for obtaining the Gaussian color
is also detailed in Fig. 12. We adopt an encoder-decoder
architecture, where an encoder MLP Mθa extracts features
from the camera poses. Unlike FreGS [45], we leverage
only high-frequency information, as low-frequency compo-
nents typically represent scene structure, which is already
effectively captured by our structured 3D Gaussians.
11.5. Anchor Points Refinement
Our anchor refinement strategy follows [21], and it is in-
cluded here to enhance the completeness of this paper.
Anchor Growing. 3D Gaussians are spatially quantized
into voxels of size ϵg = 0.001.
For all 3D Gaussians
within each voxel, we compute the average gradient af-
ter Nt = 100 training iterations, denoted as ∇g. When
the average gradient ∇g within a voxel exceeds a thresh-
old τg = 0.0002, a new anchor is added at the center of the
voxel. Since the total loss includes the frequency regulariza-
tion Lhf in Eq. (9), anchor points grow toward underrepre-
sented high-frequency regions in the scene. Ultimately, the
local details of the scene are refined. In our implementation,
the scene is quantized into a multi-resolution voxel grid, al-
3
32
3
1
ˆ
vf
vc

vc
d
Color
Na
( )
a
R,t
7
MLP
Linear activation 
function
Encoder
Decoder
C
M
Na
R,t
( )
a
R,t
R,t
Camera pose
Appearance-from-Motion 
embedding vector
a
M
Figure 12. Structure of the MLPs Mθa and MC. We adopt an
encoder-decoder architecture, where an encoder MLP Mθa first
extracts features from the camera poses. For each anchor, the fea-
ture ℓ(a)
R, t, context feature ˆfv, the relative distance δvc between
camera position tc and the anchor point, and their viewing direc-
tion ⃗dvc are then fed into a decoder MC to predict the color of
each Gaussian.
lowing new anchors to be added to regions of varying sizes,
as defined by
  \e
p
s ilon _g^ { (m)
}
 = \ epsilon _g/4^{m-1}, \ \tau _g^{(m)} = \tau _g \cdot 2^{m-1} \label {eq:anchor_growing}
(16)
where m denotes the level of quantization. Additionally, we
adopt a random candidate pruning strategy to moderate the
growth rate of anchors.
To eliminate redundant anchor points, we evaluate their
opacity. Specifically, after Nt training iterations, we accu-
mulate the opacity values of each 3D Gaussian. If the ac-
cumulated value αp falls below a pre-defined threshold, the
Gaussian is removed from the scene.
11.6. Experimental Parameters
For the monocular camera in the Replica dataset, the dimen-
sion of AfME Na is set to 1, while for other configurations,
it is set to 32. Each anchor manages k = 10 3D Gaussians.
Anchors with an opacity value below 0.005 are removed.
The loss weights λ, λvol, and λhf are set to 0.2, 0.01, and
0.01, respectively. For the monocular camera in the Replica
dataset, the weight λhf for frequency regularization is set
to 0.025, and the frequency pyramid consists of 3 levels.
12. Additional Qualitative Results
12.1. Per-scene Results.
Tab. 9, Tab. 10, Tab. 13a, Tab. 11, Tab. 12, and Tab. 13b
present the photorealistic mapping and localization results
of our method across all datasets for each scene. Addition-
ally, Fig. 13, Fig. 14, Fig. 15, Fig. 16, and Fig. 17 show
more rendering comparisons between our method and all
baseline methods for each scene.

<!-- page 15 -->
Datasets
Replica
TUM RGB-D
Method
Metric
R0
R1
R2
Of0
Of1
Of2
Of3
Of4
Avg.
fr1/d
fr2/x
fr3/o
Avg.
MonoGS [24]
PSNR↑
34.29
35.77
36.79
40.87
40.73
35.22
35.89
34.98
36.81
23.59
24.46
24.29
24.11
SSIM↑
0.953
0.957
0.965
0.979
0.977
0.961
0.962
0.955
0.964
0.783
0.789
0.829
0.800
LPIPS↓
0.071
0.078
0.074
0.048
0.052
0.074
0.061
0.092
0.069
0.244
0.227
0.223
0.231
Photo-SLAM [12]
PSNR↑
32.09
34.15
35.91
38.70
39.53
33.13
34.15
36.35
35.50
20.14
22.15
20.68
20.99
SSIM↑
0.920
0.941
0.959
0.967
0.964
0.943
0.943
0.956
0.949
0.722
0.765
0.721
0.736
LPIPS↓
0.069
0.055
0.041
0.048
0.045
0.075
0.064
0.053
0.056
0.258
0.169
0.211
0.213
Photo-SLAM-30K
PSNR↑
31.41
35.84
38.41
40.44
41.06
34.56
35.43
38.36
36.94
21.78
21.57
21.84
21.73
SSIM↑
0.873
0.955
0.971
0.975
0.972
0.952
0.954
0.967
0.952
0.766
0.755
0.751
0.757
LPIPS↓
0.046
0.036
0.026
0.033
0.033
0.059
0.049
0.036
0.040
0.212
0.182
0.165
0.186
RTG-SLAM [27]
PSNR↑
28.49
31.27
32.96
37.32
36.12
31.14
31.19
33.81
32.79
13.62
17.08
18.70
16.47
SSIM↑
0.834
0.902
0.927
0.957
0.943
0.923
0.918
0.937
0.918
0.501
0.573
0.648
0.574
LPIPS↓
0.152
0.119
0.122
0.084
0.103
0.145
0.139
0.125
0.124
0.557
0.403
0.422
0.461
GS-SLAM∗[40]
PSNR↑
31.56
32.86
32.59
38.70
41.17
32.36
32.03
32.92
34.27
-
-
-
-
SSIM↑
0.968
0.973
0.971
0.986
0.993
0.978
0.970
0.968
0.975
-
-
-
-
LPIPS↓
0.094
0.075
0.093
0.050
0.033
0.094
0.110
0.112
0.082
-
-
-
-
SplaTAM [14]
PSNR↑
32.54
33.58
35.03
38.00
38.85
31.71
29.74
31.40
33.85
21.02
23.39
19.81
21.41
SSIM↑
0.938
0.936
0.952
0.963
0.955
0.928
0.902
0.914
0.936
0.753
0.806
0.731
0.764
LPIPS↓
0.068
0.096
0.072
0.087
0.095
0.100
0.119
0.157
0.099
0.341
0.204
0.249
0.265
SGS-SLAM [17]
PSNR↑
32.48
33.50
35.11
38.22
38.91
31.86
30.05
31.53
33.96
-
-
-
-
SSIM↑
0.975
0.968
0.983
0.983
0.982
0.966
0.952
0.946
0.969
-
-
-
-
LPIPS↓
0.071
0.099
0.073
0.083
0.091
0.099
0.118
0.154
0.099
-
-
-
-
GS-ICP SLAM [9]
PSNR↑
34.89
37.15
37.89
41.62
42.86
32.69
31.45
38.54
37.14
15.67
18.49
19.25
17.81
SSIM↑
0.955
0.965
0.970
0.981
0.981
0.965
0.959
0.969
0.968
0.574
0.667
0.692
0.642
LPIPS↓
0.048
0.045
0.047
0.027
0.031
0.057
0.057
0.045
0.045
0.444
0.308
0.329
0.361
Ours
PSNR↑
37.07
39.54
40.33
42.04
43.21
36.38
37.18
39.62
39.42
25.29
26.35
26.46
26.03
SSIM↑
0.968
0.977
0.980
0.982
0.979
0.967
0.969
0.977
0.975
0.839
0.831
0.859
0.843
LPIPS↓
0.023
0.016
0.015
0.020
0.019
0.035
0.026
0.018
0.021
0.136
0.081
0.105
0.107
Table 9. Quantitative evaluation of our method compared to state-of-the-art methods for RGB-D camera on Replica and TUM RGB-D
datasets. The best results are marked as best score , second best score and third best score . GS-SLAM∗denotes the result of GS-
SLAM taken from [40], and all others are obtained in our experiments. ’-’ denotes that the system does not provide valid results.
Datasets (Camera)
Replica (Mono)
TUM RGB-D (Mono)
Method
Metric
R0
R1
R2
Of0
Of1
Of2
Of3
Of4
Avg.
fr1/d
fr2/x
fr3/o
Avg.
MonoGS [24]
PSNR↑
26.19
25.42
27.83
31.90
34.22
26.09
28.56
26.49
28.34
20.38
21.21
21.41
21.00
SSIM↑
0.819
0.798
0.889
0.911
0.930
0.881
0.898
0.897
0.878
0.691
0.690
0.735
0.705
LPIPS↓
0.246
0.368
0.252
0.249
0.192
0.268
0.189
0.284
0.256
0.377
0.377
0.426
0.393
Photo-SLAM [12]
PSNR↑
30.43
32.11
32.89
37.24
38.10
31.60
32.27
34.16
33.60
19.56
20.82
20.12
20.17
SSIM↑
0.890
0.926
0.937
0.960
0.955
0.932
0.928
0.943
0.934
0.705
0.718
0.702
0.708
LPIPS↓
0.099
0.073
0.069
0.062
0.061
0.094
0.084
0.073
0.077
0.281
0.158
0.233
0.224
Photo-SLAM-30K
PSNR↑
32.13
33.14
37.27
38.04
41.73
35.22
34.88
36.22
36.08
22.57
20.54
20.08
21.06
SSIM↑
0.896
0.921
0.965
0.964
0.974
0.952
0.949
0.955
0.947
0.787
0.714
0.697
0.733
LPIPS↓
0.056
0.086
0.035
0.055
0.033
0.061
0.057
0.052
0.054
0.179
0.166
0.213
0.186
Ours
PSNR↑
34.94
37.96
38.28
41.19
42.23
36.30
35.44
37.33
37.96
23.94
25.39
26.17
25.17
SSIM↑
0.949
0.967
0.971
0.978
0.972
0.964
0.952
0.959
0.964
0.804
0.813
0.857
0.825
LPIPS↓
0.039
0.027
0.026
0.027
0.036
0.038
0.055
0.050
0.037
0.135
0.121
0.110
0.122
Table 10. Quantitative evaluation of our method compared to state-of-the-art methods for Monocular (Mono) camera on Replica and
TUM RGB-D datasets. The best results are marked as best score and second best score .

<!-- page 16 -->
Datasets
Replica
TUM RGB-D
Method
Metric
R0
R1
R2
Of0
Of1
Of2
Of3
Of4
Avg.
fr1/d
fr2/x
fr3/o
Avg.
ORB-SLAM3 [3]
RMSE↓0.500
0.537
0.731
0.762
1.338
0.636
0.419
9.319
1.780
5.056
0.390
1.143
2.196
DRIOD-SLAM [35] RMSE↓95.994 52.471 62.908 54.807 36.038 118.191 94.200 79.510 74.264 36.057 16.749 169.844 74.216
MonoGS [24]
RMSE↓0.444
0.273
0.274
0.442
0.469
0.220
0.159
2.237
0.565
1.531
1.440
1.535
1.502
Photo-SLAM [12]
RMSE↓0.529
0.397
0.295
0.501
0.379
1.202
0.768
0.585
0.582
3.578
0.337
1.696
1.870
RTG-SLAM [27]
RMSE↓0.222
0.258
0.248
0.201
0.190
0.115
0.156
0.136
0.191
1.582
0.377
0.996
0.985
GS-SLAM∗[40]
RMSE↓0.480
0.530
0.330
0.520
0.410
0.590
0.460
0.700
0.500
3.300
1.300
6.600
3.700
SplaTAM [14]
RMSE↓0.501
0.220
0.298
0.316
0.582
0.256
0.288
0.279
0.343
5.102
1.339
3.329
4.215
SGS-SLAM [17]
RMSE↓0.463
0.216
0.300
0.339
0.547
0.299
0.451
0.311
0.365
-
-
-
-
GS-ICP SLAM [9]
RMSE↓0.189
0.132
0.216
0.201
0.236
0.160
0.162
0.117
0.177
3.539
2.251
2.972
2.921
Ours
RMSE↓0.296
0.264
0.182
0.429
0.354
1.040
0.434
0.441
0.430
3.187
0.370
1.026
1.528
Table 11. Camera tracking result on Replica and TUM RGB-D datasets for RGB-D camera. RMSE of ATE (cm) is reported. The best
results are marked as best score , second best score and third best score . ’-’ denotes the system does not provide valid results.
Datasets (Camera)
Replica (Mono)
TUM RGB-D (Mono)
Method
Metric
R0
R1
R2
Of0
Of1
Of2
Of3
Of4
Avg.
fr1/d
fr2/x
fr3/o
Avg.
ORB-SLAM3 [3]
RMSE↓51.388 26.384 4.330 110.212 103.948 65.359 51.145 1.188 51.744 4.3269 10.4598 123.226 46.004
DRIOD-SLAM [35] RMSE↓103.892 53.146 66.939 53.267 34.431 119.311 98.089 83.732 76.600 1.769
0.458
2.839
1.689
MonoGS [24]
RMSE↓12.623 56.357 25.350 43.245 19.729 39.148 11.754 88.230 37.054 4.575
4.605
2.847
4.009
Photo-SLAM [12]
RMSE↓
0.336
0.551 0.234
2.703
0.505
2.065
0.399 0.644 0.930
1.633
0.935
2.050
1.539
Ours
RMSE↓
0.288
0.388 0.215
0.579
0.320
3.963
0.307 0.603 0.833
3.187
0.370
1.026
1.505
Table 12. Camera tracking result on Replica and TUM RGB-D datasets for monocular camera. RMSE of ATE (cm) is reported. Best
results are marked as best score , second best score and third best score .
Datasets (Camera)
EuRoC (Stereo)
Method
Metric
MH01
MH02
V101
V201
Avg.
MonoGS
PSNR↑
22.84
25.53
23.39
18.66
22.60
[24]
SSIM↑
0.789
0.850
0.831
0.687
0.789
LPIPS↓
0.243
0.181
0.287
0.384
0.274
Photo-
PSNR↑
11.22
11.14
13.78
11.46
11.90
SLAM
SSIM↑
0.300
0.306
0.520
0.509
0.409
[12]
LPIPS↓
0.469
0.464
0.394
0.427
0.439
Photo-
PSNR↑
11.10
11.04
13.66
11.26
11.77
SLAM-
SSIM↑
0.296
0.300
0.516
0.508
0.405
30K
LPIPS↓
0.466
0.457
0.389
0.409
0.430
Ours
PSNR↑
22.50
22.30
24.90
24.89
23.64
SSIM↑
0.750
0.727
0.843
0.842
0.791
LPIPS↓
0.220
0.269
0.122
0.117
0.182
(a) Quantitative evaluation of our method compared to state-of-the-art meth-
ods for Stereo camera on EuRoC MAV datasets. The best results are marked
as best score and second best score .
Datasets (Camera)
EuRoC (Stereo)
Method
Metric MH01 MH02 V101
V201
Avg.
ORB-SLAM3 [3]
RMSE↓4.806 4.938 8.829 25.057 10.907
DRIOD-SLAM [35] RMSE↓1.177 1.169 3.678
1.680
1.926
MonoGS [24]
RMSE↓11.194 8.327 29.365 148.080 49.241
Photo-SLAM [12]
RMSE↓3.997 4.547 8.882 26.665 11.023
Ours
RMSE↓3.948 3.863 8.823 13.217 7.462
(b) Camera tracking result on EuRoC MAV datasets for stereo camera.
RMSE of ATE (cm) is reported. The best results are marked as best score
and second best score .
Table 13. Quantitative evaluation of our method compared to state-of-the-art methods for Stereo camera on EuRoC MAV datasets.

<!-- page 17 -->
(a) GS-ICP SLAM [9]
(b) SplaTAM [14]
(c) RTG-SLAM [27]
(d) Ours
(e) Ground Truth
Figure 13. We show comparisons of ours to state-of-the-art methods on TUM RGB-D dataset for RGB-D camera. From top to bottom, the
scenes are fr1/desk (rows 1–2), fr2/xyz (rows 3–4), and fr3/office (rows 5–6). Non-obvious differences in quality are highlighted by arrows.

<!-- page 18 -->
(a) GS-ICP SLAM [9]
(b) SplaTAM [14]
(c) RTG-SLAM [27]
(d) Ours
(e) Ground Truth
Figure 14. We show comparisons of ours to state-of-the-art methods on Replica dataset for RGB-D camera. From top to bottom, the
scenes are Office0, Office1, Office2, Office3, Office4, room0, room1, and room2. Non-obvious differences in quality are highlighted by
arrows/insets. Since all methods achieve high-quality rendering results on the Replica dataset, we have highlighted specific regions in the
images. In these annotated areas, our method consistently demonstrates sharper edges or finer textures.

<!-- page 19 -->
(a) MonoGS [24]
(b) Photo-SLAM [12]
(c) Photo-SLAM-30K
(d) Ours
(e) Ground Truth
Figure 15. We show comparisons of ours to state-of-the-art methods on TUM RGB-D dataset for Monocular camera. From top to bottom,
the scenes are fr1/desk (rows 1–2), fr2/xyz (rows 3–4), and fr3/office (rows 5–6). Non-obvious differences in qualityare highlighted by
arrows. Photo-SLAM [12] uses a set of parameters to undistort images as ground truth supervision. Consequently, its rendered images for
fr1/desk and fr2/xyz exhibit black borders.

<!-- page 20 -->
(a) MonoGS [24]
(b) Photo-SLAM [12]
(c) Photo-SLAM-30K
(d) Ours
(e) Ground Truth
Figure 16. We show comparisons of ours to state-of-the-art methods on Replica dataset for Monocular camera. From top to bottom, the
scenes are Office0, Office1, Office2, Office3, Office4, room0, room1, and room2. Non-obvious differences in quality are highlighted by
arrows/insets. Since all methods achieve high-quality rendering results on the Replica dataset, we have highlighted specific regions in the
images. In these annotated areas, our method consistently demonstrates sharper edges or finer textures.

<!-- page 21 -->
(a) MonoGS [24]
(b) Photo-SLAM [12]
(c) Photo-SLAM-30K
(d) Ours
(e) Ground Truth
Figure 17. We show comparisons of ours to state-of-the-art methods on the EuRoC MAV dataset for Stereo camera. From top to bottom,
the scenes are MH01 (rows 1–2), MH02 (rows 3–4), V101 (rows 5–6), and V201 (rows 7–8). Non-obvious differences in quality are
highlighted by arrows.
