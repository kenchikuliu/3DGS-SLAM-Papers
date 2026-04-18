<!-- page 1 -->
ODHSR: Online Dense 3D Reconstruction of Humans and Scenes
from Monocular Videos
Zetong Zhang1
Manuel Kaufmann1
Lixin Xue1
Jie Song1,2,3
Martin R. Oswald4
1ETH Z¨urich
2HKUST(GZ)
3HKUST
4University of Amsterdam
Figure 1. ODHSR takes monocular RGB input videos of humans and jointly reconstructs a photorealistic dense Gaussian representation
of the scene and the moving human as well as camera poses, human poses, and human silhouettes within a SLAM setting.
Abstract
Creating a photorealistic scene and human reconstruction
from a single monocular in-the-wild video figures promi-
nently in the perception of a human-centric 3D world.
Recent neural rendering advances have enabled holis-
tic human-scene reconstruction but require pre-calibrated
camera and human poses, and days of training time. In this
work, we introduce a novel unified framework that simul-
taneously performs camera tracking, human pose estima-
tion and human-scene reconstruction in an online fashion.
3D Gaussian Splatting is utilized to learn Gaussian primi-
tives for humans and scenes efficiently, and reconstruction-
based camera tracking and human pose estimation modules
are designed to enable holistic understanding and effec-
tive disentanglement of pose and appearance. Specifically,
we design a human deformation module to reconstruct the
details and enhance generalizability to out-of-distribution
poses faithfully. Aiming to learn the spatial correlation be-
tween human and scene accurately, we introduce occlusion-
aware human silhouette rendering and monocular geomet-
ric priors, which further improve reconstruction quality.
Experiments on the EMDB and NeuMan datasets demon-
strate superior or on-par performance with existing meth-
ods in camera tracking, human pose estimation, novel view
synthesis and runtime. Our project page is at https://eth-
ait.github.io/ODHSR.
1. Introduction
Building robotic agents for supporting humans in every-
day tasks requires the efficient and holistic understanding
of scenes and dynamic humans in an online manner. Previ-
ous works in this domain do not always meet these criteria:
They either focus on the human reconstruction [13, 52] or
the scene reconstruction alone [43, 58, 71, 99], or if com-
bined, they are rather slow to compute (order of days) [83]
or require pre-calibrated cameras [32].
We present the first unified framework that only needs
a monocular RGB video to simultaneously perform cam-
era tracking, human pose estimation and dense photoreal-
istic reconstruction of both the scene and the human. Our
system, called ODHSR, is orders of magnitude (75x) faster
1
arXiv:2504.13167v2  [cs.CV]  18 Apr 2025

<!-- page 2 -->
than previous work [83] and constitutes an online system.
To achieve this, we carefully design a 3DGS-based [28]
optimization that jointly tracks camera poses and human
poses as well as a consistent dense scene map with both
geometry and appearance information. We model the hu-
man avatar with 3D Gaussians in canonical space, guided
by SMPL-based deformations [41], whereby the deforma-
tion is decomposed into rigid and non-rigid parts to account
for dynamic garments.
The avatar model is designed to
be simple and effective for tracking of the human pose.
The robust and accurate recovery of both scene geometry
and human shape in combination with the decomposition
of human motion and camera motion from only RGB input
is highly challenging. To this end, we design a Gaussian
Splatting-based SLAM pipeline based on [43] with effec-
tive camera tracking and mapping capabilities due to the in-
corporation of a monocular geometric prior [85] as well as
human shape and pose priors from SMPL estimates to ini-
tialize the relative positioning between the human and the
scene. Specifically, we introduce an occlusion-aware hu-
man silhouette term to aid the decomposition of human and
scene reconstruction. To keep the pipeline efficient, only a
small number of keyframes is maintained for instant map-
ping. The 3DGS-based formulation over implicit functions
allows for direct gradient flow to the Gaussians and facili-
tates better generalization to unseen poses.
To assess our method, we compare ODHSR to several
baselines on two datasets: Neuman [21], and EMDB [26]
which feature challenging in-the-wild sequences with ref-
erence SMPL and camera poses. ODHSR outperforms the
state of the art in terms of reconstruction quality for both the
human avatar and scene reconstruction. It does so without
access to any ground-truth depth, with an online algorithm,
and while achieving superior runtime and real-time render-
ing. Further, the experiments show that ODHSR outper-
forms the state of the art in global human pose estimation
and achieves on-par performance in camera tracking.
In summary, we contribute the first method that simulta-
neously performs dense and detailed reconstruction of both
human avatars and the 3D scene in an online manner from
only a monocular RGB video. This is enabled by lever-
aging the direct gradient flow of a holistic 3DGS-based
parametrization and an effective joint optimization.
The
proposed methodology is robust, accurate and efficient in
comparison to existing works.
2. Related Work
3D Scene Reconstruction and SLAM. Modern scene
reconstruction
methods
span
grid-based,
point-based,
network-based, and hybrid approaches. Grid-based meth-
ods [47, 68, 75] are memory-intensive and rely on prede-
fined resolutions, while point-based methods [5, 62] adapt
dynamically to surfaces, reducing memory waste but fac-
ing connectivity issues. Network-based approaches [45, 86]
provide detailed scene representations but struggle with
scalability and efficiency, especially in online tasks. Hybrid
techniques [47] aim to balance speed and quality. Simul-
taneous Localization and Mapping (SLAM) reconstructs
scenes while tracking camera trajectories. Dense SLAM
methods are either frame-centric or map-centric. Frame-
centric approaches [6, 71] minimize photometric errors for
frame-to-frame motion but lack global consistency. Map-
centric methods build unified 3D representations, utilizing
global information for tracking and reconstruction. Classi-
cal techniques use voxel grids [48], point clouds [59, 93],
or neural feature grids [98, 99] for 3D representation.
Recently, 3D Gaussian splatting with differentiable ren-
dering capabilities has emerged as an effective scene rep-
resentation [28]. Its tile-based renderer enables faster train-
ing and rendering than NeRF-based methods [45], making it
well-suited for dense real-time Gaussian-based SLAM [19,
27, 43, 84, 92], see [72] for an overview. Splat-SLAM [60]
augments RGB-only SLAM [43] with a monocular prior, a
deformable Gaussian map to incorporate dense bundle ad-
justment and loop closures for more accurate tracking. We
built our method upon the RGB-only approach [43] and ex-
tend it with a monocular prior as well as human pose and
shape priors for effective joint reconstruction and tracking.
Human Pose Estimation (HPE). With the emergence of
differentiable statistical body models like SMPL [41], and
powerful Deep Learning methods, substantial progress has
been achieved in recent years. Landmark papers include
SMPLify [3], which proposes to optimize the 3D poses to
2D keypoint detections, and HMR [23], which directly re-
gresses 3D poses from an image using 2D keypoint super-
vision and adversarial losses. Many other works have fol-
lowed in these footsteps, achieving impressive results [4, 9,
12, 22, 24, 30, 31, 34–36, 39, 44, 49–51, 66, 67, 69, 94, 97].
Traditionally, these works focus on camera-relative pose es-
timation. More recently, it was proposed to disentangle the
camera from the human motion and estimate the human
pose in global coordinates [33, 64, 65, 70, 87, 91]. This
is similar to our setting as we also estimate human pose and
camera motion in global space. However, we additionally
reconstruct a dense scene and a photo-realistic appearance
of the human. To deal with depth-ambiguities and occlu-
sions in images, others have proposed to use body-worn
sensors for motion capture, e.g., [25, 73, 74, 88].
This
increases instrumentation requirements, but the collected
poses can serve as reference data. We evaluate our method
on the EMDB [26] in-the-wild dataset.
3D Human Reconstruction. The above mentioned works
all estimate the naked human body or assume tight clothing.
It has also been proposed to reconstruct the clothing and ap-
pearance, e.g., by extending explicit SMPL-based represen-
tations [1, 2, 77, 79], by using a personalized pre-scanned
2

<!-- page 3 -->
template of the human [15, 81, 90], articulated NeRFs
[21, 76], or implicit surface fields [13, 20, 55, 57, 78].
All of these works either neglect the scene, reconstruct
low-quality geometry, or impose large computational costs.
With the overwhelming success of 3D Gaussian Splat-
ting [28], the community quickly adopted the representation
for human avatar modeling, both from multi-view [38, 100]
and monocular images [17, 32, 40, 52, 63] to reduce com-
putation times. Animatable 3D Gaussian [40] uses multi-
resolution hash grids to predict the Gaussian attributes and
achieves fast training, but lacks robust to input pose er-
rors because it does not optimize for the poses. [63] in-
troduces hybrid mesh and Gaussian representation, which
leads to a similar problem because there are no direct gradi-
ents to the input pose parameters. [32] and [17] learn LBS
weights to map from canonical to the posed space, regular-
ized by SMPL. However, they do not model any local de-
formations that might be due to non-tight clothing. 3DGS-
Avatar [52] explicitly models the local non-rigid deforma-
tion and pose-dependent color change with two MLPs, but
the large MLP architecture slows down its training com-
pared to other Gaussian-based works.
Human-Scene Modeling. Modeling humans and scenes to-
gether is a topic studied in differing shades. Some methods
do so from an egocentric perspective with the use of body-
worn sensors and cameras [7, 8, 14, 89, 96] but they ei-
ther require a pre-scanned scene or do not reconstruct the
detailed human avatar. Other work focuses on exocentric
views, and solves tasks such es extracting interaction graphs
[61], disambiguating human poses with scene constraints
[16, 37], placing humans into existing scenes [95], or mod-
eling rich contacts using static multi-view setups [18]. Clos-
est to our work are HSR [83] and HUGS [32] who operate
in the same setting. HSR extends the success of neural im-
plicit shape functions to jointly model the human and the
scene. This formulation incurs high computational costs.
Like ours, HUGS models the human avatar and the scene
with 3D Gaussians. However, its triplane formulation is ill-
suited for an online setting as converging to good features
for the tracking is slow.
3. Method
We first describe how we model the human avatar in
Sec. 3.1, and how this ties in with the chosen scene rep-
resentation in Sec. 3.2. Finally, in Sec. 3.3 we describe
how we simultaneously perform camera tracking, human
pose estimation and human-scene reconstruction. For an
overview of our method, please refer to Fig. 2.
3.1. 3D Avatar Representation
We represent the human body in the canonical space with a
set of 3D Gaussians GH = {GH,i|i = 1, . . . , NH}, where
each Gaussian GH,i is parameterized by its own center po-
sition µH, center offset ∆µH, rotation RH, scale SH, opac-
ity oH, RGB color cH and Linear Blend Skinning (LBS)
weights WH ∈RJ with respect to J SMPL joints. Among
these parameters, centers µH are initially sampled around
SMPL vertices and stay fixed at the initial positions, and
we instead optimize the center offsets∆µH, along with all
the other parameters. The skeletal deformation and skin-
ning driven by SMPL can only model the rigid deforma-
tion of human joints, but dynamic garments may not pre-
cisely follow the joint deformations. Therefore, similar to
[46, 52, 76], we decompose the Gaussian deformation into
a rigid part and a non-rigid part.
Time-pose Dependent Non-rigid Deformation and Ap-
pearance.
The deformations of the irregular garment
and hair are dependent on the human pose and time-
accumulating dynamics.
Hence, we model per-Gaussian
local deformation via a time-pose conditioned multi-
resolution hash encoding network [47]. Moreover, shadows
cast on the human surface change with the geometric defor-
mation. A second multiresolution hash encoding network is
introduced to model the ambient occlusion factor to address
the shadow issue. We denote the network parameters as Fϕ
and refer the readers to the supplementary material for the
detailed input encoding and architecture design. Given the
current time step t and human pose θ ∈R3J, we obtain the
local deformation ∆µ′
H, ∆RH and ambient occlusion ∆c
of each Gaussian via:
∆µ′
H, ∆RH, ∆cH = Fϕ(µH, t, θ)
(1)
The canonical Gaussians are then deformed by:
µH,d = µH + ∆µH + ∆µ′
H
(2)
RH,d = RH · ∆RH
(3)
cH,d = ∆cH · cH
(4)
Rigid Transformation. Following SMPL, we use LBS to
deform the model using the joints in the underlying SMPL
model defined by shape β ∈R10 and pose parameters
θ ∈R3J. The transformation Mj ∈SE(3) of each joint
j from the canonical space to the posed space is calculated
using the kinematic tree. Each Gaussian’s skinning trans-
formation P is the weighted average of joint transforma-
tions according to learnable parameter WH, formulated as
P = PJ
j=1 WH,jMj, where WH,j is the j-th element of
LBS weight WH ∈RJ corresponding to the j-th joint. We
then transform the canonical Gaussian positions and rota-
tions calculated in Eq. (2), (3) to the world frame as follows:
µH,w = PµH,d
(5)
RH,w = P:3,:3RH,d
(6)
3

<!-- page 4 -->
Figure 2. System Overview of ODHSR. Given a monocular video featuring a human in the scene, we simultaneously track the camera
and human poses for each frame while training 3D Gaussian primitives. Camera and human pose optimization is achieved through dense
matching for view synthesis and leveraging monocular geometric cues. Mapping is carried out within a small local keyframe window, and
we apply multiple regularizations to enhance reconstruction quality from the sparse set of keyframes.
3.2. Holistic Human-scene Representation
Holistic Representation. We use standard 3D Gaussians to
model the scene. The set of scene Gaussians is denoted as
GS = {GS,i|i = 1, . . . , NS}, where GS,i is the i-th scene
Gaussian and is composed of its own center µS, scale SS,
rotation RS, opacity oS and RGB color cS.
Given scene Gaussians GS and transformed human
Gaussians in the world frame GH, we merge them into a
global Gaussian set G = GS + GH as the holistic human-
scene primitives and feed them into the Gaussian rasterizer
to render the color map ˆI, depth map ˆD, and opacity map
ˆO, respectively.
Occlusion-aware Human Silhouette. 3D Gaussians can,
by design, handle the occlusion between objects since the
Gaussians are sorted by depth along the camera ray in the
rasterizer. The occlusion-aware human opacity (silhouette)
ˆOH can then be retrieved as:
ˆOH =
NH
X
j=1
αj
Nj
Y
k=1
(1 −αk)
(7)
where Nj is the number of all the human and scene Gaus-
sians whose depth along this pixel ray is smaller than that
of GH,j and QNj
k=1(1−αk) represents for the transmittance
at j-th human Gaussian calculated from all the Gaussians
in front. By taking into account the scene Gaussians, which
are closer to the camera, we obtain the human silhouette
rendering where the occlusion is faithfully modeled.
3.3. SLAM
In this section, we present the details of our full SLAM
framework, where camera tracking, human pose estima-
tion, and dense human-scene reconstruction are performed
simultaneously. For each frame, we compute the residu-
als between the input and the rendering from the holistic
Gaussian representation to track both the camera and hu-
man pose. A keyframe check is performed on each tracked
frame, and a local keyframe window is updated, with which
we run mapping to jointly reconstruct the human and the
scene. In the end, we follow the idea of adopting global
bundle adjustment in SLAM approaches[43, 58, 71, 92] to
finetune the holistic representation with all the keyframes.
Our pipeline contains two threads for efficiency. The track-
ing thread takes in new frames, runs camera and human
pose optimization and selects keyframes, while the mapping
thread simultaneously runs mapping and bundle adjustment
over the local keyframe window.
Initialization. The estimation of high-dimensional human
pose and shape parameters θ, β of a person from an RGB
image is challenging without prior knowledge.
We start
4

<!-- page 5 -->
with the poses from an off-the-shelf monocular human pose
estimator, WHAM [65]. For the very first frame of the se-
quence, we carefully refine the human pose estimate θ in a
model-free approach by minimizing the 2D keypoint loss.
Following [26], we extract N = 25 2D keypoints from ViT-
Pose [82] denoted by ˜xi and define a 2D keypoint loss as
Lkp =
25
X
i=1
1[confi > 0.5] · ρ(ˆxi −˜xi)
(8)
where ˆxi = K[R|t]Xi are projection of SMPL joints Xi
onto the image and confi is the confidence of the i-th key-
point predicted by the keypoint detector, 1 is the indicator
function and ρ is the robust Geman-McClure function [10].
In order to faithfully recover the spatial correlation be-
tween the human and the scene and to produce trustworthy
scene geometry for subsequent camera and human pose es-
timation, we utilize a monocular depth estimator [85] to ob-
tain per-frame monocular depth estimation ˜D. We solve for
the scale and shift parameters w, b for the first frame dispar-
ity 1/ ˜D by aligning it with the SMPL mesh disparity cal-
culated from β and θ with a RANSAC estimator to reduce
the effect of outliers. We then initialize the scene Gaussians
at the positions inferred from the re-scaled monocular depth
of the first frame, i.e., 1/(w/ ˜D + b).
Camera and Human Pose Estimation. Given each new
image I, we jointly optimize the camera pose T and human
pose θ of the current frame via the following optimization
constraints while keeping the holistic representation fixed.
▷RGB Loss. We minimize the photometric residual be-
tween the input I and rendered image ˆI as follows.
Lrgb = ∥I −ˆI∥1
(9)
▷Optical Flow Loss. Following [58, 99], to avoid the local
minima introduced by the pixel-wise RGB loss, we estimate
the optical flow ˜pij from the last keyframe i and the current
frame j with a pretrained estimator [80]. Given the rendered
depth ˆD of frame i, we can also compute the flow from
pixels pi in frame i to projected pixel coordinates in frame
j and minimize the optical flow loss Lflow.
Lflow = ∥˜pij −K∆Tij ˆDiK−1[pi, 1]⊤∥1
(10)
where K is the camera intrinsic matrix, ∆Tij is the relative
pose between frames i, j. Since this consistency only holds
for static objects, we mask out the dynamic human via a pre-
estimated human segmentation mask from [29]. We keep
the pose of keyframe i fixed and expect this loss to only
contribute to the camera pose optimization.
▷Monocular Depth Loss. Inspired by [83], we make use
of geometric priors from pre-trained depth estimators and
enforce the depth consistency between our rendered depth
ˆD and the monocular depth ˜D. Since the monocular depth
is usually prone to error in far objects, like sky and build-
ings, in the outdoor scene. During tracking, to stabilize the
pose optimization, we compute inverse depth map d = 1/D
and minimize the geometric residual between the rendering
and monocular ones as:
Ldisp = ∥ˆd −(w ˜d + b)∥1
(11)
where w, b ∈R are the scale and shift used to align ˆd and ˜d,
since ˜d is only known up to an unknown scale. We solve for
w and b per image with least squares at each optimization it-
eration, where both the human and scene pixels are utilized
to ensure the scaled monocular depth map can faithfully re-
flect the human-scene spatial correlations.
▷Human Silhouette Loss. The noisy color and depth ren-
dering from the online mapping could make the optimiza-
tion converge slowly. We also utilize a human silhouette
loss as an auxiliary signal for the human pose optimization.
Given the pre-estimated human segmentation ˜OH and ren-
dered human silhouette ˆOH as in Eq. (7), we formulate the
human silhouette loss as follows.
Lsil = ∥ˆOH −˜OH∥1
(12)
▷2D Keypoint Loss. RGB loss suffers from color ambi-
guities due to the sparse texture on the human, and thus is
not sufficient to accurately align the human joints and learn
the human poses. We additionally use the 2D keypoint loss
formulated in Eq. (8) for each frame as an auxiliary term to
guide the joint alignment.
The final loss for the joint camera and human pose opti-
mization is the weighted sum of all the losses introduced
above:
Lpose =λrgbLrgb + λflowLflow
(13)
+ λdispLdisp + λsilLsil + λkpLkp
Notably, for Lrgb, Lflow , Ldisp and Lsil, we use whole-image
rendered opacity map ˆO as the pixel weights and compute
weighted l1 loss to mitigate the effect from unseen regions.
Keyframing. After tracking, each frame will be checked
for keyframe registration based on multiple criteria, includ-
ing frame interval, camera displacement, human joint dis-
placements, and Gaussian co-visibility. These criteria are
designed to find the most informative frames for the map-
ping. Refer to the supplementary material for the details.
Following
the
strategy
of
Gaussian
Splatting
SLAM [43],
we only maintain a small number of
keyframes in the current window Wk and update the
window constantly to only keep frames that are either the
latest or have the largest visual overlap. By doing this, we
update the Gaussians and networks with the knowledge
from the new keyframes, which can be generalized better
to a subsequent frame.
5

<!-- page 6 -->
For each new keyframe, we insert new Gaussians into the
scene by back-projecting the re-scaled monocular depth of
the static background to 3D space to capture newly visible
scene components.
Mapping.
During mapping, keyframes in local window
Wk along with two random past keyframes are used to re-
construct recently visible regions and avoid forgetting the
global map.
To enforce the consistency between obser-
vation and reconstruction, we minimize the following re-
rendering losses:
▷RGB Loss. We use l1 RGB loss for color reconstruction.
▷Human Silhouette Loss. Separating the human from the
scene is challenging, especially when the views are limited
in our online mapping setting. We also use the human sil-
houette loss formulated in Eq. (12) for reconstruction.
▷Monocular Depth Loss. We use monocular depth loss
Ldepth to stabilize and clean the scene to prevent “floaters”
appearing in free space, which could occlude the human in
the camera view. Different from Eq. (11), we keep w, b
fixed, compute the absolute depth residual, and only con-
strain the depth rendering of scene pixels during mapping.
Due to the limited training poses in the local keyframe
window, we introduce several regularizations on the avatar
representation to better generalize novel human poses and
reconstruct animatable avatars.
▷Local Deformation Loss. We penalize the magnitude of
the local deformation and ambient occlusion factor to stabi-
lize the training with Ldeform.
▷LBS Weights Loss. To prevent the skinning weights WH
from overfitting on the training poses, we supervise the per-
Gaussian skinning weight with the skinning weight in the
SMPL model via LLBS.
▷Canonical Center Loss. We softly regularize the geom-
etry of the reconstructed human with the underlying SMPL
model.
With this regularization, Lcenter, we prevent the
Gaussians from moving too wildly due to limited training
views and poses.
To summarize, we minimize the weighted sum of all
these losses in the mapping thread:
Lmap =λrgbLrgb + λsilLsil + λdepthLdepth
(14)
+ λLBSLLBS + λcenterLcenter + λdeformLdeform
Furthermore, the hash encoding network-based deforma-
tion and appearance modules exhibit sensibility to noise
when exposed to novel pose and time encodings. To mit-
igate this issue, we propose two training strategies aimed at
effectively learning network parameters with robust inter-
polation and extrapolation properties across the spatial and
temporal dimensions, thereby stabilizing the mapping pro-
cess. Please refer to the supplementary material for further
details on these loss and strategy designs.
4. Experiments
4.1. Experimental Setup
Datasets. The following datasets are used for evaluation:
▷EMDB dataset [26] is a recently published large-scale
in-the-wild dataset consisting of versatile sequences cap-
tured in outdoor or indoor scenes. We identify five dis-
tinct sequences that presented various challenges, such as
extended human and camera trajectories, human occlusions,
prominent shadows on the human body, sparse background
texture within the lab setting, and unconventional human
poses (e.g. cartwheels). For consistency concerns, we take
the first 500 frames, i.e., the first 16 seconds, from all these
sequences in our experiments. This is our major dataset for
quantitative evaluation.
▷NeuMan dataset [21] is an in-the-wild dataset with six
sequences, each captured with a moving camera that pans
through the scene. Our keyframe selection is deactivated for
this dataset, and we instead follow the dataset split outlined
in [21] and only run evaluation for reconstruction quality.
Metrics. We report standard photometric rendering quality
metrics (PSNR, SSIM, LPIPS) on the non-keyframes/test
set for the novel view synthesis task. These metrics are
calculated on both whole images and human-only images.
Moreover, we care about the accuracy of our predicted
poses. For camera tracking, we follow conventional monoc-
ular SLAM evaluation protocol to align trajectory and cal-
culate trajectory error (ATE RMSE). The predicted human
poses are evaluated as well via local pose metrics MPJPE,
PA-MPJPE and MVE and global motion metrics W-MPJPE
and WA-MPJPE. Following EMDB [26], we report a jitter
metric to take account of the smoothness of the estimated
joint trajectories.
Reconstruction Baselines. Since, to the best of our knowl-
edge, our work is the first online dense human-scene re-
construction method in the community, we opt for offline
reconstruction works for baseline comparison. Our auto-
selected keyframes are used for training, while the remain-
ing frames are used for evaluation only.
We compare
our method with the holistic human-scene reconstruction
method HSR [83], Vid2Avatar [13] and HUGS [32]. For
3DGS-based [28] scene reconstruction approach HUGS, we
additionally provide captured depth maps to initialize the
scene Gaussians. Our online approach contrarily omits such
demanding preprocessing. In addition, we also compare
our method with 3DGS-based human-only reconstruction
works 3DGS-Avatar [52] and GauHuman [17]. These of-
fline works typically require known camera poses. For fair
comparison on the EMDB dataset, we use the SotA tracker
DROID-SLAM [71] with these baselines, as well as ours
while disabling the camera tracking module. This evalua-
tion strategy also provides insights into the performance of
our camera tracker. On the NeuMan dataset, we evaluate the
6

<!-- page 7 -->
Whole images
Human-only regions
FPS
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Training↑
Rendering↑
GauHuman [17]
-
-
-
25.313
0.943
0.057
0.150
150
3DGS-Avatar [52]
-
-
-
27.952
0.967
0.035
0.112
60
Vid2Avatar [13]
16.656
0.413
0.599
24.258
0.948
0.061
< 10−3
0.02
HUGS [32]
21.605
0.659
0.181
26.165
0.947
0.033
0.042
40
HSR [83]
18.675
0.463
0.632
25.127
0.924
0.054
0.002
0.05
Ours (DROID-SLAM)
23.458
0.756
0.200
29.203
0.965
0.030
0.181
85
Ours (Full)
23.790
0.767
0.197
28.955
0.966
0.031
0.141
85
Table 1. Novel view synthesis evaluation on the EMDB dataset [26]. For the human-only setting, we render the avatar on white
background for all the baselines and compute metrics over the whole image. We report training and rendering FPS, with training FPS
calculated as the inverse of the average training time per image. All experiments were run on a single Nvidia GeForce RTX 4090 GPU.
baselines directly using ground truth camera poses. Consis-
tent with our approach, all baselines take WHAM estimates
as the human pose initialization.
4.2. Evaluation Results
Novel View Synthesis.
In Tab. 1 and Tab. 2, we com-
pare our reconstruction quality over the whole image and
human-only renderings on the EMDB and NeuMan dataset.
We present two variants of our approach, either running the
camera tracking with the optimization scheme or using an
existing SOTA camera tracker as an alternative. We ob-
serve that our camera tracking module achieves comparable
or even better reconstruction while operating in a fully on-
line manner. ODHSR overperforms all the holistic human-
scene reconstruction methods by a large margin. The scene
model in Vid2Avatar is designed to be human-centric, and
the scene geometry and textures are not properly learned
from multi-view correlations.
HSR extends Vid2Avatar
with scene field and holistic representation but still shows
degraded performance for the view synthesis task in out-
door scenes. HUGS achieves moderate performance in the
background, but their overall results are worse than ours as
their human reconstruction is prone to pose noises. We ad-
ditionally compare the human-only reconstruction quality
with two 3DGS-based methods. On the challenging EMDB
dataset where the input poses are noisy and there exists
drastic illumination change and garment deformations, we
show significant advantages over the baselines. On the Neu-
Man dataset, we notice a noticeable drop in performance
when switching from known camera poses.
Even with-
out the known camera trajectory, our holistic reconstruc-
tion achieves the highest performance, and our avatar re-
construction quality is second only to 3DGS-Avatar while
requiring significantly less time.
We further show the qualitative results in Fig. 3.
In
general, our method shows the best reconstruction quality
with complete contours and vivid photorealistic textures,
where the detailed clothes’ deformation and shadows are
better recovered than others. Vid2Avatar struggles with bad
background reconstruction and missing body parts, and pro-
duces a lot of artifacts. HSR handles the background mod-
eling better, but its SDF representation is prone to over-
smooth features.
HUGS fails to produce decent recon-
structions that faithfully recover the detailed human appear-
ance because their joint Gaussian and human pose optimiza-
tion schemes can not effectively disentangle the appearance
and pose. 3DGS-Avatar performs well in approximating
clothes wrinkles and reserving color smoothness but faces
the same problem as HUGS. Gauhuman uses an MLP to
learn the pose correction offset during training and performs
the worst at handling input pose inconsistency. The render-
ings become quite blurry, and the details of the clothes and
faces are almost lost.
Camera Tracking. To provide a straightforward insight
into our camera tracker, we evaluate predicted camera tra-
jectories against the ground truth on the EMDB dataset. Our
method achieves an ATE of 8.4 cm, which is comparable to
the SotA DROID-SLAM, while owns a unique advantage in
scale estimation by leveraging humans as a reference. We
further show that incorporating human information explic-
itly into the tracker facilitates ours to overperforms static
SLAM approaches where humans are masked out.
Human Pose Estimation. We demonstrate that our frame-
work enhances the accuracy of the human trajectory. Using
WHAM’s local (camera) frame estimation for pose initial-
ization, we evaluate our refined human trajectory against
their predicted global trajectories and compare it with raw
WHAM predictions. Our method achieves a WA-MPJPE of
175.215 mm on the EMDB dataset, while WHAM records
636.001 mm.
For detailed evaluation of camera tracking and human
pose estimation, please refer to the supplementary material.
4.3. Ablation Study
We conduct ablation experiments to examine the effect of
removing each of the loss components in Eq. (13). The
2D keypoint loss and human silhouette loss work together
to accelerate human pose optimization, which subsequently
facilitates monocular depth alignment. Clean scene geome-
try, achieved through precise depth scaling, further reduces
7

<!-- page 8 -->
Whole images
Human-only regions
FPS
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
Training↑
Rendering↑
GauHuman [17]
-
-
-
30.731
0.977
0.017
0.038
150
3DGS-Avatar [52]
-
-
-
32.920
0.988
0.015
0.015
60
Vid2Avatar [13]
15.640
0.551
0.572
30.967
0.981
0.018
< 10−3
0.008
HUGS [32]
26.667
0.851
0.126
30.136
0.977
0.017
0.019
40
HSR [83]
21.676
0.669
0.526
29.033
0.971
0.026
< 10−3
0.01
Ours (GT camera)
27.784
0.870
0.153
32.079
0.981
0.016
0.046
85
Ours (Full)
26.470
0.825
0.174
31.729
0.981
0.017
0.040
85
Table 2. Novel view synthesis evaluation on the NeuMan dataset [21]. Compared to recent offline methods, our approach provides
better whole image rendering performance while being comparable on human-only regions despite our more challenging online setting.
(a) GT
(b) Ours
(c) Vid2Avatar
(d) HSR
(e) HUGS
(f) 3DGS-Avatar
(g) GauHuman
Figure 3. Qualitative results on the EMDB dataset [26]. Our online approach is highly competitive when compared to recent offline
methods and outperforms most of them especially with respect to image sharpness and data fidelity.
PSNR↑
ATE RMSE [m]↓
WA-MPJPE [mm]↓
w/o Lflow
22.593
0.214
301.621
w/o Lkeypoint
22.263
0.121
230.875
w/o Ldisp
22.769
0.165
252.547
w/o Lsil
22.648
0.148
240.838
Full model
23.790
0.084
175.215
Table 3. Ablation study of camera and human pose optimiza-
tion on the EMDB dataset [26]. The view synthesis, camera
tracking and human pose estimation results demonstrate consis-
tently superior performance for the full model.
(a) Full model
(b) w/o Lcenter
(c) w/o LLBS
Figure 4. Qualitative ablation of regularizations on avatar. Our
full model comprises the least amount of artifacts.
errors in optical flow, monocular depth, and silhouette ac-
curacy. Together, these four interconnected losses are fun-
damental to the success of our proposed approach.
Alongside tracking, we inspect the impact of our de-
signed objective, particularly its effect on human subjects,
which enables effective generalization to new perspectives
and poses even with a limited training dataset. As illustrated
in Fig. 4, omitting the canonical center loss Lcenter and LBS
loss LLBS allows Gaussian deformation to go unrestrained,
significantly compromising reconstruction accuracy, espe-
cially in challenging areas like the face and arms.
5. Conclusion
We introduce ODHSR, the first unified framework capa-
ble of simultaneously performing camera localization, hu-
man pose estimation, and dense human-scene reconstruc-
tion from monocular RGB videos in a fully online setting.
By integrating monocular geometric priors with explicit 3D
primitives, our approach effectively models human-scene
spatial correlations, enhancing pose optimization and re-
construction accuracy. Our joint optimization demonstrates
improved performance in novel view synthesis and human
pose estimation tasks, marking a significant step forward for
real-time, monocular video-based 3D reconstruction.
8

<!-- page 9 -->
Acknowledgments
This work was partially supported by the Swiss SERI Con-
solidation Grant ”AI-PERCEIVE”.
Computations were
performed on the ETH Z¨urich Euler Cluster. We thank Chen
Guo and TianJian Jiang for their valuable suggestions in this
research project.
References
[1] Thiemo Alldieck, Gerard Pons-Moll, Christian Theobalt,
and Marcus Magnor. Tex2shape: Detailed full human body
geometry from a single image. In IEEE International Con-
ference on Computer Vision (ICCV). IEEE, 2019. 2
[2] Bharat Lal Bhatnagar, Garvita Tiwari, Christian Theobalt,
and Gerard Pons-Moll.
Multi-garment net: Learning to
dress 3d people from images. In IEEE International Con-
ference on Computer Vision (ICCV). IEEE, 2019. 2
[3] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Pe-
ter Gehler, Javier Romero, and Michael J. Black. Keep it
smpl: Automatic estimation of 3d human pose and shape
from a single image, 2016. 2
[4] Junhyeong Cho, Kim Youwang, and Tae-Hyun Oh. Cross-
attention of disentangled modalities for 3d human mesh
recovery with transformers.
In European Conference on
Computer Vision (ECCV), 2022. 2
[5] Chi-Ming Chung, Yang-Che Tseng, Ya-Ching Hsu, Xiang-
Qian Shi, Yun-Hung Hua, Jia-Fong Yeh, Wen-Chin Chen,
Yi-Ting Chen, and Winston H. Hsu. Orbeez-slam: A real-
time monocular visual slam with orb features and nerf-
realized mapping. In 2023 IEEE International Conference
on Robotics and Automation (ICRA), pages 9400–9406,
2023. 2
[6] J Czarnowski, T Laidlow, R Clark, and AJ Davison. Deep-
factors:
Real-time probabilistic dense monocular slam.
IEEE Robotics and Automation Letters, 5:721–728, 2020.
2
[7] Yudi Dai, Yitai Lin, Chenglu Wen, Siqi Shen, Lan Xu,
Jingyi Yu, Yuexin Ma, and Cheng Wang. Hsc4d: Human-
centered 4d scene capture in large-scale indoor-outdoor
space using wearable imus and lidar.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 6792–6802, 2022. 3
[8] Yudi Dai, Yitai Lin, Xiping Lin, Chenglu Wen, Lan Xu,
Hongwei Yi, Siqi Shen, Yuexin Ma, and Cheng Wang.
Sloper4d: A scene-aware dataset for global 4d human pose
estimation in urban environments. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 682–692, 2023. 3
[9] Sai Kumar Dwivedi, Yu Sun, Priyanka Patel, Yao Feng, and
Michael J. Black. TokenHMR: Advancing human mesh re-
covery with a tokenized pose representation. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), 2024. 2
[10] Stuart Geman and Donald E. McClure. Statistical methods
for tomographic image reconstruction. In Bulletin of the
International Statistical Institute, pages 5–21, 1987. 5
[11] Saeed Ghorbani, Kimia Mahdaviani, Anne Thaler, Konrad
Kording, Douglas James Cook, Gunnar Blohm, and Niko-
laus F. Troje. Movi: A large multipurpose motion and video
dataset, 2020. 15
[12] Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran,
Angjoo Kanazawa, and Jitendra Malik.
Humans in 4D:
Reconstructing and tracking humans with transformers. In
ICCV, 2023. 2
[13] Chen Guo, Tianjian Jiang, Xu Chen, Jie Song, and Otmar
Hilliges. Vid2avatar: 3d avatar reconstruction from videos
in the wild via self-supervised scene decomposition. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), 2023. 1, 3, 6, 7, 8, 15
[14] Vladimir Guzov, Aymen Mir, Torsten Sattler, and Gerard
Pons-Moll. Human poseitioning system (hps): 3d human
pose estimation and self-localization in large scenes from
body-mounted sensors. In IEEE Conference on Computer
Vision and Pattern Recognition (CVPR). IEEE, 2021. 3
[15] Marc Habermann, Weipeng Xu, Michael Zollh¨ofer, Gerard
Pons-Moll, and Christian Theobalt. Deepcap: Monocular
human performance capture using weak supervision.
In
2020 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 5051–5062, 2020. 3
[16] Mohamed Hassan, Vasileios Choutas, Dimitrios Tzionas,
and Michael J. Black. Resolving 3D human pose ambigui-
ties with 3D scene constraints. In International Conference
on Computer Vision, 2019. 3
[17] Shoukang Hu and Ziwei Liu.
Gauhuman: Articulated
gaussian splatting from monocular human videos. arXiv
preprint arXiv:, 2023. 3, 6, 7, 8, 15
[18] Chun-Hao P. Huang, Hongwei Yi, Markus H¨oschle, Matvey
Safroshkin, Tsvetelina Alexiadis, Senya Polikovsky, Daniel
Scharstein, and Michael J. Black.
Capturing and infer-
ring dense full-body human-scene contact. In Proceedings
IEEE/CVF Conf. on Computer Vision and Pattern Recogni-
tion (CVPR), pages 13274–13285, 2022. 3
[19] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Ye-
ung.
Photo-slam:
Real-time simultaneous localization
and photorealistic mapping for monocular stereo and rgb-
d cameras. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 21584–
21593, 2024. 2
[20] Boyi Jiang, Yang Hong, Hujun Bao, and Juyong Zhang.
Selfrecon:
Self reconstruction your digital avatar from
monocular video. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2022. 3
[21] Wei Jiang, Kwang Moo Yi, Golnoosh Samei, Oncel Tuzel,
and Anurag Ranjan. Neuman: Neural human radiance field
from a single video. In Proceedings of the European con-
ference on computer vision (ECCV), 2022. 2, 3, 6, 8, 15,
16
[22] Hanbyul Joo, Natalia Neverova, and Andrea Vedaldi. Ex-
emplar fine-tuning for 3d human pose fitting towards in-
the-wild 3d human pose estimation. In 3DV, 2020. 2
[23] Angjoo Kanazawa, Michael J. Black, David W. Jacobs, and
Jitendra Malik. End-to-end recovery of human shape and
pose. In Computer Vision and Pattern Recognition (CVPR),
2018. 2
9

<!-- page 10 -->
[24] Angjoo Kanazawa, Jason Y. Zhang, Panna Felsen, and Ji-
tendra Malik. Learning 3d human dynamics from video. In
Computer Vision and Pattern Recognition (CVPR), 2019. 2
[25] Manuel Kaufmann, Yi Zhao, Chengcheng Tang, Lingling
Tao, Christopher Twigg, Jie Song, Robert Wang, and Ot-
mar Hilliges. Em-pose: 3d human pose estimation from
sparse electromagnetic trackers. In The IEEE International
Conference on Computer Vision (ICCV), 2021. 2
[26] Manuel Kaufmann, Jie Song, Chen Guo, Kaiyue Shen,
Tianjian Jiang, Chengcheng Tang, Juan Jos´e Z´arate, and
Otmar Hilliges. EMDB: The Electromagnetic Database of
Global 3D Human Pose and Shape in the Wild. In Interna-
tional Conference on Computer Vision (ICCV), 2023. 2, 5,
6, 7, 8
[27] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallab-
hula, Gengshan Yang, Sebastian Scherer, Deva Ramanan,
and Jonathon Luiten. Splatam: Splat, track and map 3d
gaussians for dense rgb-d slam.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2024. 2, 17
[28] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics,
42(4), 2023. 2, 3, 6, 15
[29] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi
Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer
Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar,
and Ross Girshick. Segment anything, 2023. 5
[30] Muhammed Kocabas, Nikos Athanasiou, and Michael J.
Black. Vibe: Video inference for human body pose and
shape estimation. In The IEEE Conference on Computer
Vision and Pattern Recognition (CVPR), 2020. 2
[31] Muhammed Kocabas, Chun-Hao P. Huang, Otmar Hilliges,
and Michael J. Black. PARE: Part attention regressor for 3D
human body estimation. In Proc. International Conference
on Computer Vision (ICCV), pages 11127–11137, 2021. 2
[32] Muhammed Kocabas, Rick Chang, James Gabriel, Oncel
Tuzel, and Anurag Ranjan. Hugs: Human gaussian splats,
2023. 1, 3, 6, 7, 8, 14, 15
[33] Muhammed Kocabas, Ye Yuan, Pavlo Molchanov, Yunrong
Guo, Michael J. Black, Otmar Hilliges, Jan Kautz, and
Umar Iqbal. Pace: Human and motion estimation from in-
the-wild videos. In 3DV, 2024. 2
[34] Christoph Lassner, Javier Romero, Martin Kiefel, Federica
Bogo, Michael J. Black, and Peter V. Gehler.
Unite the
people: Closing the loop between 3d and 2d human repre-
sentations, 2017. 2
[35] Jiefeng Li, Chao Xu, Zhicun Chen, Siyuan Bian, Lixin
Yang, and Cewu Lu. Hybrik: A hybrid analytical-neural
inverse kinematics solution for 3d human pose and shape
estimation. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 3383–
3393, 2021.
[36] Zhihao Li, Jianzhuang Liu, Zhensong Zhang, Songcen Xu,
and Youliang Yan.
Cliff: Carrying location information
in full frames into human pose and shape estimation. In
ECCV, 2022. 2
[37] Zhi Li, Soshi Shimada, Bernt Schiele, Christian Theobalt,
and Vladislav Golyanik. Mocapdeform: Monocular 3d hu-
man motion capture in deformable scenes. In International
Conference on 3D Vision (3DV), 2022. 3
[38] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu.
Animatable gaussians: Learning pose-dependent gaussian
maps for high-fidelity human avatar modeling. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2024. 3
[39] Kevin Lin, Lijuan Wang, and Zicheng Liu.
Mesh
graphormer. In ICCV, 2021. 2
[40] Yang Liu, Xiang Huang, Minghan Qin, Qinwei Lin, and
Haoqian Wang. Animatable 3d gaussian: Fast and high-
quality reconstruction of multiple human avatars.
arXiv
preprint arXiv:2311.16482, 2023. 3, 13
[41] Matthew Loper, Naureen Mahmood, Javier Romero, Ger-
ard Pons-Moll, and Michael J. Black. SMPL: A skinned
multi-person linear model. ACM Trans. Graphics (Proc.
SIGGRAPH Asia), 34(6):248:1–248:16, 2015. 2
[42] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje,
Gerard Pons-Moll, and Michael J. Black. AMASS: Archive
of motion capture as surface shapes. In International Con-
ference on Computer Vision, pages 5442–5451, 2019. 15
[43] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and An-
drew J. Davison. Gaussian Splatting SLAM. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024. 1, 2, 4, 5, 13, 16, 17
[44] Dushyant Mehta, Srinath Sridhar, Oleksandr Sotnychenko,
Helge Rhodin, Mohammad Shafiei, Hans-Peter Seidel,
Weipeng Xu, Dan Casas, and Christian Theobalt. Vnect:
Real-time 3d human pose estimation with a single rgb cam-
era.
ACM Transactions on Graphics (TOG), 36(4):44,
2017. 2
[45] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 2, 13
[46] Arthur Moreau, Jifei Song, Helisa Dhamo, Richard Shaw,
Yiren Zhou, and Eduardo P´erez-Pellitero. Human gaussian
splatting: Real-time rendering of animatable avatars, 2024.
3, 13
[47] Thomas M¨uller,
Alex Evans,
Christoph Schied,
and
Alexander Keller. Instant neural graphics primitives with
a multiresolution hash encoding. ACM Trans. Graph., 41
(4):102:1–102:15, 2022. 2, 3
[48] Richard A. Newcombe, Shahram Izadi, Otmar Hilliges,
David Molyneaux, David Kim, Andrew J. Davison, Push-
meet Kohi, Jamie Shotton, Steve Hodges, and Andrew
Fitzgibbon. Kinectfusion: Real-time dense surface map-
ping and tracking. In 2011 10th IEEE International Sym-
posium on Mixed and Augmented Reality, pages 127–136,
2011. 2
[49] Mohamed Omran, Christoph Lassner, Gerard Pons-Moll,
Peter V. Gehler, and Bernt Schiele.
Neural body fitting:
Unifying deep learning and model-based human pose and
shape estimation, 2018. 2
10

<!-- page 11 -->
[50] Georgios Pavlakos, Luyang Zhu, Xiaowei Zhou, and Kostas
Daniilidis. Learning to estimate 3d human pose and shape
from a single color image, 2018.
[51] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani,
Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas,
and Michael J. Black. Expressive body capture: 3D hands,
face, and body from a single image.
In Proceedings
IEEE Conf. on Computer Vision and Pattern Recognition
(CVPR), pages 10975–10985, 2019. 2
[52] Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas
Geiger, and Siyu Tang. 3dgs-avatar: Animatable avatars
via deformable 3d gaussian splatting. In CVPR, 2024. 1, 3,
6, 7, 8, 15
[53] Nikhila Ravi, Jeremy Reizenstein, David Novotny, Tay-
lor Gordon, Wan-Yen Lo, Justin Johnson, and Georgia
Gkioxari. Accelerating 3d deep learning with pytorch3d.
arXiv:2007.08501, 2020. 14
[54] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kun-
chang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen,
Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang,
Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam:
Assembling open-world models for diverse visual tasks,
2024. 17
[55] Shunsuke Saito, Tomas Simon, Jason M. Saragih, and
Hanbyul Joo.
Pifuhd: Multi-level pixel-aligned implicit
function for high-resolution 3d human digitization. 2020
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 81–90, 2020. 3
[56] Shunsuke Saito, Jinlong Yang, Qianli Ma, and Michael J.
Black. SCANimate: Weakly supervised learning of skinned
clothed avatar networks.
In Proceedings IEEE/CVF
Conf. on Computer Vision and Pattern Recognition
(CVPR), 2021. 13
[57] Shunsuke Saito, Jinlong Yang, Qianli Ma, and Michael J.
Black. SCANimate: Weakly supervised learning of skinned
clothed avatar networks.
In Proceedings IEEE/CVF
Conf. on Computer Vision and Pattern Recognition
(CVPR), 2021. 3
[58] Erik
Sandstr¨om,
Keisuke
Tateno,
Michael
Oechsle,
Michael Niemeyer, Luc Van Gool, Martin R Oswald,
and Federico Tombari.
Splat-slam:
Globally opti-
mized rgb-only slam with 3d gaussians.
arXiv preprint
arXiv:2405.16544, 2024. 1, 4, 5
[59] Erik Sandstr¨om, Yue Li, Luc Van Gool, and Martin R. Os-
wald. Point-slam: Dense neural point cloud-based slam. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), 2023. 2
[60] Erik
Sandstr¨om,
Keisuke
Tateno,
Michael
Oechsle,
Michael Niemeyer, Luc Van Gool, Martin R. Oswald, and
Federico Tombari.
Splat-slam: Globally optimized rgb-
only slam with 3d gaussians, 2024. 2
[61] Manolis Savva, Angel X Chang, Pat Hanrahan, Matthew
Fisher, and Matthias Nießner. Pigraphs: learning interac-
tion snapshots from observations. ACM Transactions on
Graphics (TOG), 35(4):1–12, 2016. 3
[62] Thomas Sch¨ops, Torsten Sattler, and Marc Pollefeys.
Bad slam: Bundle adjusted direct rgb-d slam.
In 2019
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 134–144, 2019. 2
[63] Zhijing Shao, Zhaolong Wang, Zhuang Li, Duotun Wang,
Xiangru Lin, Yu Zhang, Mingming Fan, and Zeyu Wang.
SplattingAvatar: Realistic Real-Time Human Avatars with
Mesh-Embedded Gaussian Splatting.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2024. 3
[64] Zehong Shen, Huaijin Pi, Yan Xia, Zhi Cen, Sida Peng,
Zechen Hu, Hujun Bao, Ruizhen Hu, and Xiaowei Zhou.
World-grounded human motion recovery via gravity-view
coordinates. In SIGGRAPH Asia Conference Proceedings,
2024. 2
[65] Soyong Shin, Juyong Kim, Eni Halilaj, and Michael J.
Black.
Wham: Reconstructing world-grounded humans
with accurate 3d motion. In Computer Vision and Pattern
Recognition (CVPR), 2024. 2, 5, 16
[66] Jie Song, Xu Chen, and Otmar Hilliges.
Human body
model fitting by learned gradient descent.
In Computer
Vision–ECCV 2020: 16th European Conference, Glasgow,
UK, August 23–28, 2020, Proceedings, Part XX 16, pages
744–760. Springer, 2020. 2
[67] Anastasis
Stathopoulos,
Ligong
Han,
and
Dimitris
Metaxas. Score-guided diffusion for 3d human recovery.
In CVPR, 2024. 2
[68] Jiaming Sun, Yiming Xie, Linghao Chen, Xiaowei Zhou,
and Hujun Bao. NeuralRecon: Real-time coherent 3D re-
construction from monocular video. CVPR, 2021. 2
[69] Yu Sun, Qian Bao, Wu Liu, Yili Fu, Black Michael J., and
Tao Mei. Monocular, one-stage, regression of multiple 3d
people. In ICCV, 2021. 2
[70] Yu Sun, Qian Bao, Wu Liu, Tao Mei, and Michael J. Black.
TRACE: 5D Temporal Regression of Avatars with Dynamic
Cameras in 3D Environments. In CVPR, 2023. 2
[71] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam
for monocular, stereo, and rgb-d cameras, 2022. 1, 2, 4, 6,
15, 17
[72] Fabio Tosi, Youmin Zhang, Ziren Gong, Erik Sandstr¨om,
Stefano Mattoccia, Martin R. Oswald, and Matteo Poggi.
How nerfs and 3d gaussian splatting are reshaping slam: a
survey, 2024. 2
[73] Matt Trumble, Andrew Gilbert, Charles Malleson, Adrian
Hilton, and John Collomosse. Total capture: 3d human pose
estimation fusing video and inertial sensors. In 2017 British
Machine Vision Conference (BMVC), 2017. 2
[74] Timo von Marcard, Roberto Henschel, Michael Black,
Bodo Rosenhahn, and Gerard Pons-Moll. Recovering ac-
curate 3d human pose in the wild using imus and a mov-
ing camera. In European Conference on Computer Vision
(ECCV), 2018. 2
[75] Silvan Weder, Johannes L. Sch¨onberger, Marc Pollefeys,
and Martin R. Oswald. Routedfusion: Learning real-time
depth map fusion. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2020. 2
[76] Chung-Yi Weng, Brian Curless, Pratul P. Srinivasan,
Jonathan T. Barron, and Ira Kemelmacher-Shlizerman. Hu-
manNeRF: Free-viewpoint rendering of moving people
11

<!-- page 12 -->
from monocular video. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 16210–16220, 2022. 3
[77] Donglai Xiang, Fabi´an Prada, Chenglei Wu, and Jessica K.
Hodgins.
Monoclothcap: Towards temporally coherent
clothing capture from monocular rgb video.
2020 Inter-
national Conference on 3D Vision (3DV), pages 322–332,
2020. 2
[78] Yuliang Xiu,
Jinlong Yang,
Dimitrios Tzionas,
and
Michael J. Black.
ICON: Implicit Clothed humans Ob-
tained from Normals.
In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 13296–13306, 2022. 3
[79] Yuliang Xiu, Jinlong Yang, Xu Cao, Dimitrios Tzionas,
and Michael J. Black.
ECON: Explicit Clothed humans
Optimized via Normal integration. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2023. 2
[80] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and
Dacheng Tao. Gmflow: Learning optical flow via global
matching.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 8121–
8130, 2022. 5
[81] Weipeng Xu, Avishek Chatterjee, Michael Zollhoefer,
Helge Rhodin, Dushyant Mehta, Hans-Peter Seidel, and
Christian Theobalt.
Monoperfcap: Human performance
capture from monocular video.
ACM Transactions on
Graphics, 37(2):1–15, 2018. 3
[82] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao.
ViTPose: Simple vision transformer baselines for human
pose estimation. In Advances in Neural Information Pro-
cessing Systems, 2022. 5
[83] Lixin Xue, Chen Guo, Chengwei Zheng, Fangjinhua Wang,
Tianjian Jiang, Hsuan-I Ho, Manuel Kaufmann, Jie Song,
and Hilliges Otmar. HSR: holistic 3d human-scene recon-
struction from monocular videos. In European Conference
on Computer Vision (ECCV), 2024. 1, 2, 3, 5, 6, 7, 8, 15
[84] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang,
Dong Wang, and Xuelong Li. Gs-slam: Dense visual slam
with 3d gaussian splatting. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 19595–19604, 2024. 2
[85] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth any-
thing v2. arXiv:2406.09414, 2024. 2, 5
[86] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman.
Volume rendering of neural implicit surfaces.
In Thirty-
Fifth Conference on Neural Information Processing Sys-
tems, 2021. 2
[87] Vickie Ye, Georgios Pavlakos, Jitendra Malik, and Angjoo
Kanazawa.
Decoupling human and camera motion from
videos in the wild. In IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 2023. 2
[88] Xinyu Yi, Yuxiao Zhou, Marc Habermann, Soshi Shi-
mada, Vladislav Golyanik, Christian Theobalt, and Feng
Xu. Physical inertial poser (pip): Physics-aware real-time
human motion tracking from sparse inertial sensors.
In
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2022. 2
[89] Xinyu Yi, Yuxiao Zhou, Marc Habermann, Vladislav
Golyanik, Shaohua Pan, Christian Theobalt, and Feng Xu.
Egolocate:
Real-time motion capture, localization, and
mapping with sparse body-mounted sensors. ACM Trans-
actions on Graphics (TOG), 42(4), 2023. 3
[90] Tao Yu, Zerong Zheng, Yuan Zhong, Jianhui Zhao, Qiong-
hai Dai, Gerard Pons-Moll, and Yebin Liu.
Simulcap :
Single-view human performance capture with cloth simu-
lation. In 2019 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 5499–5509, 2019.
3
[91] Ye Yuan, Umar Iqbal, Pavlo Molchanov, Kris Kitani, and
Jan Kautz. Glamr: Global occlusion-aware human mesh
recovery with dynamic cameras.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2022. 2
[92] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Os-
wald. Gaussian-slam: Photo-realistic dense slam with gaus-
sian splatting, 2023. 2, 4
[93] Ganlin Zhang, Erik Sandstr¨om, Youmin Zhang, Manthan
Patel, Luc Van Gool, and Martin R Oswald. Glorie-slam:
Globally optimized rgb-only implicit encoding point cloud
slam. arXiv preprint arXiv:2403.19549, 2024. 2
[94] Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang,
Yebin Liu, Limin Wang, and Zhenan Sun. Pymaf: 3d hu-
man pose and shape regression with pyramidal mesh align-
ment feedback loop. In Proceedings of the IEEE Interna-
tional Conference on Computer Vision, 2021. 2
[95] Siwei Zhang, Yan Zhang, Qianli Ma, Michael J Black, and
Siyu Tang. Place: Proximity learning of articulation and
contact in 3d environments. In 2020 International Confer-
ence on 3D Vision (3DV), pages 642–651. IEEE, 2020. 3
[96] Siwei Zhang, Qianli Ma, Yan Zhang, Zhiyin Qian, Taein
Kwon, Marc Pollefeys, Federica Bogo, and Siyu Tang.
Egobody: Human body shape and motion of interacting
people from head-mounted devices. In European confer-
ence on computer vision (ECCV), 2022. 3
[97] Xiaowei Zhou, Menglong Zhu, Spyridon Leonardos, Kon-
stantinos G Derpanis, and Kostas Daniilidis.
Sparseness
meets deepness: 3d human pose estimation from monocular
video. In Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, pages 4966–4975, 2016. 2
[98] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu,
Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc
Pollefeys.
Nice-slam: Neural implicit scalable encoding
for slam. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2022. 2
[99] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui,
Martin R Oswald, Andreas Geiger, and Marc Pollefeys.
Nicer-slam: Neural implicit scene encoding for rgb slam.
In International Conference on 3D Vision (3DV), 2024. 1,
2, 5
[100] Wojciech Zielonka, Timur Bagautdinov, Shunsuke Saito,
Michael Zollh¨ofer, Justus Thies, and Javier Romero. Driv-
able 3d gaussian avatars, 2023. 3
12

<!-- page 13 -->
A. Time-pose Dependent Deformation Net-
work
Figure 5. Local deformation and ambient occlusion network. Yel-
low: Fixed parameters; Blue: Frozen parameters.
In this section, we present the details of our designed
time-pose dependent non-rigid deformation and appearance
module introduced in Sec. 3.1 in the main paper. 1
Our network design is shown in Fig.5, where two parallel
multiresolution hash encoding networks are utilized to learn
geometric and photometric deformation respectively. Given
time step t, human pose parameter θ and per-Gaussian LBS
weight W, the encodings of time and pose are denoted as
γt(t) and γp(θ; W), respectively. Specifically, we use the
positional encoding as in [45] to encode the normalized in-
put time, where the max degree is set to be 4. For the pose,
we follow the idea of [46] to use an attention-weighting
scheme to encode only the pose parameters of joints that are
close to the Gaussian center. By doing this, the redundant
information in the global pose parameters θ can be removed
so that the local deformation around the input Gaussian will
be better learned without the spurious correlation of irrel-
evant joints. This is inspired by SCANimate [56], which
uses the LBS weight and a predefined attention map V that
limits the propagation of deformation within four neighbor-
ing joints in the kinematic tree. The pose encoding is then
formulated as follows:
γp(θ; W) = (V · W) ⊙θ
(15)
where θ is in quaternion format and ⊙denotes element-
wise multiplication.
Similar to [40], we use the fixed Gaussian centers µ as
the input of the hash grid to compress the size of the hash
1In Sec. A and Sec. C, we omit the subscripts of human Gaussian pa-
rameters for simplicity.
table and prevent optimization from diverging owing to the
unstable Gaussian displacements. The time and pose en-
codings are concatenated with the hash encoding features
queried with the Gaussian center µ as the input of the shal-
low MLP networks to produce deformation ∆µ′, ∆R and
1-channel ambient occlusion ∆c prediction. With the MLP
architecture, we use its smoothness prior and expect good
interpolation properties to be learned to facilitate general-
ization to novel frames and poses. In practice, we param-
eterize the rotation ∆R in the form of a quaternion vector
and limit the ambient occlusion factor within the range of
0-2.
B. Keyframe Management
In this section, we introduce our designed criteria for
keyframe selection selection.
▷Frame interval: Only a frame whose time difference
from the last keyframe is above a threshold τt can be cho-
sen as a new keyframe so that we can avoid registering the
keyframes too frequently and idling the main thread for a
long period.
▷Camera motion: If the displacement of the current cam-
era from that of the last keyframe is larger than a threshold
τc, we will add the current frame to the keyframe set to span
a wide baseline.
▷Human motion: We measure the averaged human joint
displacement from the last keyframe for each frame to esti-
mate the pose change. Large human motion is likely to lead
to unobserved local non-rigid deformation and appearance
change. Thus, we register frames with drastic pose change,
where the joint displacement is above τj, to better model
the garment deformation.
▷Gaussian covisibility: 3D Gaussians respect visibility
ordering since the rasterizer will sort Gaussians along the
camera ray. Similar to [43], we mark Gaussians as visible
if they contribute to the rendering from the camera view. We
then compute the covisibility of all the Gaussians (human +
scene) by computing the IOU value of visible ones between
the current frame and the last keyframe. If the covisibility
is below a threshold τv, the current frame will be selected
as the new keyframe to reduce redundant visual overlap be-
tween keyframes.
When a new keyframe is added or the size of the local
keyframe window is larger than τs, we update the window
with the new keyframe. Previous keyframe whose overlap
with the latest keyframe is below a threshold τr or the frame
whose camera distance from other keyframes is the farthest
will be removed from the current keyframe window. By
doing this, we update the Gaussians and networks with the
knowledge from the new keyframes, which can be general-
ized better to a subsequent frame.
13

<!-- page 14 -->
C. Losses
We describe in this section the detailed formulation of pro-
posed regularizations applied on the avatar representation
that are introduced in Sec. 3.3.
Local Deformation Loss. We constrain the local deforma-
tion to be as small as possible and encourage the frame-
generic model to best learn the average shape and average.
The local deformation loss Ldeform is composed of three
parts that respectively penalize the displacement ∆µ′, push
the rotation offset ∆R to be close to the identity matrix, and
enforce the ambient occlusion factor ∆c to stay close to 1.
Ldeform = ∥∆µ′∥2 + ∥∆R −Rid∥1 + ∥∆c −1∥2
(16)
In practice, we use quaternions to represent the rotations.
LBS Weights Loss.
For each Gaussian in the canoni-
cal space, we use the K-Nearest Neighbor (KNN) algo-
rithm to find the k nearest SMPL vertices vNN and take the
weighted sum of their LBS weights ˜W as the label to super-
vise the Gaussian LBS weights with LLBS = ∥W −˜W∥F .
Inspired by [32], the displacements between the nearest
k SMPL vertices and the Gaussian center, formulated as
∆vNN = µ + ∆µ −vNN, are used to weigh each element
so that SMPL vertices closer to the corresponding Gaussian
will contribute more to the supervision. Differently, we pro-
pose a novel distance-based weighting that takes account of
the shape and scale of each Gaussian and calculate ˜W as
follows.
˜W =
k
X
i=1
wi
w WNN,i
(17)
wi = exp (−1
2∆vNN,i
T Σ−1∆vNN,i)
(18)
w =
k
X
i=1
wi
(19)
where ∆vNN,i and WNN,i are respectively the relative po-
sition and LBS weight of the i-th nearest vertice on the
SMPL mesh from the Gaussian center, and Σ is the Gaus-
sian covariance matrix that is defined as Σ = RSST RT .
In our experiments, we set k = 3.
Canonical Center Loss. In our online pipeline, where the
local window is typically small, each Gaussian is only vis-
ible to limited training views and is thus largely uncon-
strained. To prevent Gaussians from moving and growing
arbitrarily along the camera ray, we softly regularize the
geometry of the reconstructed human with the underlying
SMPL model. Because the garment can lead to large dis-
placements from naked-body SMPL to the reconstructed
avatar, we do not directly regularize the magnitudes of the
Gaussian displacements but instead enforce the nearest ver-
tex on the SMPL mesh from each Gaussian to be the vertex
used to initialize the Gaussian. The regularization is applied
on the canonical Gaussian centers before local deformation
as follows:
Lcenter = ReLU(∥µ + ∆µ −vinit∥2 −∥µ + ∆µ −vNN∥2)
(20)
where vNN is the nearest SMPL vertex from the Gaussian
in the canonical space, and vinit is the corresponding vertex
position initially.
We run the K-Nearest Neighbor algorithm via the effi-
cient CUDA implementation in PyTorch3D [53].
D. Implementation Details
D.1. Model Configurations
We initialize the canonical Gaussian positions by creating r
replicates of each SMPL vertex in the canonical space and
injecting Gaussian noises. r is set to be 5 for the EMDB
dataset and 3 for the NeuMan dataset. The Gaussian opac-
ities are initialized to be 0.9. We use the anisotropic Gaus-
sians for both the scene and human parts.
For the LBS weights per Gaussian, we directly optimize
an offset vector to sum to the original SMPL weights and
use SoftMax as the activation function to apply to each el-
ement of the optimized weights to ensure that their values
are all positive and sum to one.
For the time-pose dependent deformation network, the
canonical points µ are first normalized with a bounding box
that tightly encloses the canonical SMPL mesh. The de-
tailed network hyperparameters are listed in Tab. 4.
Parameter
Value
Number of levels
16
Number of features per level
2
Hash table size
217
Coarsest resolution
4
Per Level Scale
1.5
MLP Width
128
MLP Number of hidden layers
3
Table 4. Local deformation and ambient occlusion network hyper-
parameters
D.2. Training Strategies
Hash Encoding Network Pretraining. Random initial val-
ues of the hash encoding network can produce incorrect out-
put on the fly when there are insufficient training frames
and the training iterations are limited. This is the typical
situation in the online training pipeline. Good interpolation
and extrapolation properties are required to quickly fit the
novel keyframe with the knowledge learned from previous
frames. Otherwise, the Gaussian parameters could also get
optimized in the wrong direction.
14

<!-- page 15 -->
Considering these issues, we propose to pre-train the lo-
cal deformation and ambient occlusion networks introduced
in Sec. A at the very beginning. This is achieved by ran-
domly sampling input time and poses to obtain the defor-
mation outputs from the hash encoding network and mini-
mize the deformation loss Ldeform. We sample the input time
from a uniform distribution between 0 and 1. As for the hu-
man pose, we sample from a combination of the pose of the
first frame and poses stored in a large-scale human database
AMASS [42] so that the network is pre-trained with realistic
poses of large variations. Gaussian noises with a standard
deviation of 0.1 are added to the input pose to augment the
data. In our experiments, the poses in the BMLmovi dataset
[11] are used for sampling. We use Adam optimizer with
learning rate 10−4 to run the optimization for 5000 itera-
tions.
Multi-stage Training.
We evenly divide the mapping
process into two stages and choose not to include the
time-pose-dependent deformation and ambient occlusion in
avatar Gaussians in the first stage while later activate them
in the second stage. This multi-stage training strategy is
employed in both the online mapping and final color refine-
ment steps.
D.3. Training Configurations
We use Adam Optimizer to optimize the camera and human
pose parameters. The learning rates in the tracking thread
are 3 × 10−3 for camera rotation, 10−3 for camera trans-
lation, 10−2 for the human root translation and orientation,
and 10−3 for other local pose parameters. In the mapping
thread where we simultaneously perform local bundle ad-
justment on the keyframe window, the learning rates are re-
duced to 1.5×10−3, 5×10−4, 10−4 and 10−5 respectively.
The learning rates of all the Gaussian parameters are exactly
the same as the original implementation from [28]. For our
additionally designed time-pose dependent network, we set
learning rate of all its parameters to be 10−4.
In the tracking thread, we iteratively run camera and
human pose optimization for 100 iterations with λrgb =
1, λflow = 1, λdisp = 0.001, λsil = 0.1, λkp = 0.0001 in
(13). While for mapping, we set λrgb = 1, λsil = 1, λdepth =
0.001, λLBS = 100, λcenter = 10, λdeform = 0.001 in (14).
Optimized Gaussians in the mapping thread are synchro-
nized with the tracking thread every 20 mapping iterations.
Finally when we iterate over the whole sequence, we fine-
tune the Gaussians with all the selected keyframes for 100
epochs.
For keyframe selection, we set τt = 0.1s, τc = 0.05m,
τj = 0.1m, τv = 0.9 as the thresholds. As for the local
keyframe update, we set τs = 10 and τr = 0.3.
For the scene representation, we periodically perform
Gaussian densification and pruning as originally described
by 3DGS [28]. In contrast, for the fixed-size human, we dis-
able the adaptive seeding during the online mapping since
the complicated topology of the human body and the limited
training viewpoints can lead to noisy gradients, especially
in the occluded human parts. The densification and pruning
module will be later activated for humans in the final color
refinement step to capture richer details.
D.4. Baselines
When assessing the performance of novel view synthesis,
we optimize human poses across all test frames for the base-
line methods to eliminate the impact of pose errors on ren-
dering. In contrast, for our approach, this step is omitted
because the test poses are already optimized dynamically
during the process. By adopting this strategy, we provide
an advantage to the baselines, as their test poses are refined
against the final reconstruction to minimize re-rendering er-
rors. Conversely, our test poses are optimized using the on-
line reconstructed model, which may be incomplete, sub-
optimally refined, and therefore more susceptible to errors.
For consistency, we fix the Gaussian and network param-
eters across all methods and utilize each method’s specific
pose estimation module, applying the same loss functions
used during their training to perform test pose optimization.
This ensures that the evaluation of novel view synthesis re-
flects the robustness of the respective pose optimization de-
signs as well. For direct pose estimation modules, as imple-
mented in [13, 32, 52, 83], we employ a uniform learning
rate of 10−3. For the pose correction MLP network used in
[17], we maintain the same learning rate as during training.
To ensure fairness, pose optimization is conducted for 100
steps on each frame across all baselines.
E. Additional Evaluation Results
E.1. Novel View Synthesis
Qualitative results on the NeuMan dataset [21] are pre-
sented in Fig. 6. Despite performing online tracking and
mapping, our method surpasses most offline reconstruction
approaches in terms of background scene fidelity and clar-
ity, even though those methods leverage ground truth cam-
era poses.
Furthermore, our approach achieves superior
quality in the reconstruction of critical and challenging hu-
man features, such as faces and hands. However, a limita-
tion of our method is that geometry near contact points be-
tween the human and the scene may not always be precisely
recovered, occasionally resulting in blurry reconstructions,
as seen in areas like shoes and the ground.
E.2. Camera Tracking
We demonstrate that our camera tracker achieves on-
par performance with the state-of-the-art SLAM approach
DROID-SLAM[71] in Tab. 5. Without knowing the true
scale, the output from DROID-SLAM cannot be seamlessly
15

<!-- page 16 -->
(a) GT
(b) Ours
(c) Vid2Avatar
(d) HSR
(e) HUGS
(f) 3DGS-Avatar
(g) GauHuman
Figure 6. Qualitative comparison of novel view synthesis task on the NeuMan dataset [21].
ATE RMSE [m]↓
DROID-SLAM
0.079
MonoGS (human masked)
0.459
Ours (human masked)
0.247
Ours (full model)
0.084
Table 5. Camera tracking evaluation on the EMDB dataset.
integrated with human pose estimates unless ground truth
depth or trajectory information is provided, limiting its ap-
plicability in dynamic scenes. However, by explicitly build-
ing the dynamic human and modeling human-scene spatial
correlation, our method handles the scaling well. Moreover,
to further inspect the impact of human on the tracking, we
run MonoGS[43] and our method while using pre-estimated
human masks to completely remove the human in the input
images and the model. As shown in Tab. 5, our method sig-
nificantly enhances the accuracy of predicted camera trajec-
tories by explicitly modeling the human, as it provides ad-
ditional spatial cues and aids in scaling the monocular depth
signal.
E.3. Human Pose Estimation
We evaluate our human pose estimations and compare them
with WHAM[65] in Tab. 6 and Fig. 7. Our reconstruction-
based pose optimization module achieves slightly enhanced
local poses that align more accurately with the 2D image.
For global motion, our holistic human-scene reconstruction
supplies the essential spatial context, enabling the human
tracker to significantly reduce globally aligned joint errors.
In contrast, WHAM, lacking explicit scene awareness, fails
to adapt to terrain changes, resulting in substantial trajec-
tory errors. However, the increased jitter observed in our
method indicates a limitation: the gradient descent opti-
mization approach becomes ineffective for occluded body
parts that are not visible in the 2D image.
F. Ablation Study
F.1. Ablation of Avatar Module Designs
Input and output components of the avatar deformation
module are ablated in Tab. 7. On the challenging EMDB
dataset where drastic garment deformation and illumination
change exist, jointly modeling the per-Gaussian deforma-
16

<!-- page 17 -->
Local Pose
Global Motion
PA-MPJPE↓
MPJPE↓
MVE↓
Jitter↓
WA-MPJPE↓
W-MPJPE↓
WHAM
40.845
72.964
83.254
14.765
636.001
2990.746
Ours
40.571
69.162
79.463
32.183
175.215
449.036
Table 6. Human pose estimation evaluation on the EMDB dataset. Jitter is in the unit of 10m/s−3 and others in mm.
Figure 7. Comparison of global human trajectory estimations on
the EMDB dataset. Left: Human trajectories of GT, WHAM pre-
dictions and our predictions on the x-y and x-z plane. The global
trajectories are globally aligned. Right: Estimated SMPL mesh on
one selected frame.
tion and ambient occlusion significantly improves all the
re-rendering metrics. As for the input, we achieve the best
performance by taking both the pose and time features com-
pared to using either one of them.
PSNR ↑
SSIM ↑
LPIPS ↓
w/o ambient occlusion
28.201
0.958
0.034
w/o deformation
27.927
0.959
0.036
w/o pose encoding
28.741
0.962
0.033
w/o time encoding
27.779
0.957
0.037
w/o HE pretraining
27.801
0.958
0.041
Full model
28.955
0.966
0.031
Table 7. Ablation study on avatar module designs and hash encod-
ing (HE) network pretraining strategy. The performance is evalu-
ated on the human-only rendering on the EMDB dataset.
F.2. Ablation of Hash Encoding Network Pretrain-
ing Strategy
In Tab. 7, we also present the evaluation results without pre-
training the hash encoding network. Due to the random-
ized initial network parameters, the local deformation net-
work produces noisy outputs, resulting in failed learning of
garment deformation and shadows, particularly at unseen
timesteps and poses. The bad interpolation and extrapola-
tion properties lead to an overall degraded performance.
G. Discussions
G.1. Online Training
We follow existing dense SLAM works[27, 43, 71] to per-
form a final refinement step to finetune the Gaussian repre-
sentation with all the selected keyframes. The refinement
process can be seen as a traditional global bundle adjust-
ment (BA) step, in which case it does not conflict with
the online nature of ODHSR. Unlike other approaches, we
do not perform full BA but instead refine only the Gaus-
sians, allowing us to distribute the refinement into the online
optimization rather than applying it as a post-processing
step—though this comes at the cost of lower training FPS.
By distributing refinement into the online pipeline after
each keyframe tracking step and training for ten epochs per
refinement operation, we achieve a final PSNR of 23.013 for
the whole image and 28.814 for human-only regions, which
is slightly worse than the full model and increases runtime
(reducing FPS by 0.06). The final refinement step is de-
signed to prevent catastrophic forgetting, and we showcase
that without this, ODHSR still largely overperforms base-
lines in novel view synthesis and runtime efficiency.
G.2. Challenging Cases
Scene Occlusion.
We demonstrate the impact of our
occlusion-aware human silhouette design in Fig. 8.
For
body parts occluded by scene components, such as legs,
ODHSR consistently generates smooth and precise bound-
ary silhouettes.
In contrast, the state-of-the-art general
segmentation model SAM[54], while capable of predict-
ing occlusions, occasionally produces results with missing
parts. By explicitly modeling occlusions, ODHSR effec-
tively models spatial correlations without losing human fea-
tures.
Figure 8. Results in the scene occlusion scenario. Our generated
human mask is compared against the prediction from the Segment
Anything Model(SAM).
Long Trajectories.
In Fig. 9, we showcase the results
17

<!-- page 18 -->
Figure 9. Results in the long trajectory scenario. Left: Our human-scene reconstruction with tracked cameras. Right: Estimated trajectories
from ours and DROID-SLAM, compared with the ground truth on the EMDB dataset. Colors of the curve segments indicate trajectory
error, ranging from 0. to 1.
of our method in a long-trajectory scenario, where repeti-
tive background patterns pose challenges for camera track-
ing. Overall, ODHSR delivers decent results and effectively
captures camera motion trends with small trajectory errors.
However, the sparse features on the wall and ground in-
crease the challenge of accurate geometric reconstruction,
introducing some surface noise that subsequently leads to
additional errors in the estimated camera poses for certain
frames. DROID-SLAM performs better in such scenarios
by leveraging cleverer bundle adjustment and graph-based
optimization strategy, highlighting a promising direction for
further improvements.
G.3. Limitations
While ODHSR achieves state-of-the-art rendering qual-
ity on the challenging in-the-wild dataset, its performance
heavily depends on single-frame pre-estimations, such as
monocular depth and human keypoints—particularly in the
first frame, which initializes the system. Although we in-
corporate a pairwise flow loss in camera and human pose
optimization, we argue that this alone is insufficient for
constructing a globally consistent scene and pose repre-
sentation. Also, despite producing high-quality renderings,
our method introduces surface noise due to the nonsmooth
depth characteristics of 3D Gaussian Splatting. Addition-
ally, our method could suffer from potential human-scene
interpenetrations around the contact points, such as feet.
Due to the noisy surfaces 3D Gaussians produce, it is not yet
resolved. Finally, our model-based camera and human pose
optimization primarily relies on pixel-level errors, which
can lead to local optima in textureless regions or areas with
uniform features, such as walls and clothing.
18
