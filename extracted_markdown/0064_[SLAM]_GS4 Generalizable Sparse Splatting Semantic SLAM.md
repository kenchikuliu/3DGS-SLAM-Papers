<!-- page 1 -->
GS4: Generalizable Sparse Splatting Semantic SLAM
Mingqi Jiang, Chanho Kim, Chen Ziwen, Li Fuxin
Collaborative Robotics and Intelligent Systems (CoRIS) Institute
Oregon State University
{jiangmi, kimchanh, chenziw, lif}@oregonstate.edu
Project page: https://mingqij.github.io/projects/gs4/
Abstract
Traditional SLAM algorithms excel at camera tracking, but
typically produce incomplete and low-resolution maps that
are not tightly integrated with semantics prediction. Re-
cent work integrates Gaussian Splatting (GS) into SLAM to
enable dense, photorealistic 3D mapping, yet existing GS-
based SLAM methods require per-scene optimization that
is slow and consumes an excessive number of Gaussians.
We present GS4, the first generalizable GS-based semantic
SLAM system. Compared with prior approaches, GS4 runs
10× faster, uses 10× fewer Gaussians, and achieves state-
of-the-art performance across color, depth, semantic map-
ping and camera tracking. From an RGB-D video stream,
GS4 incrementally builds and updates a set of 3D Gaussians
using a feed-forward network. First, the Gaussian Predic-
tion Model estimates a sparse set of Gaussian parameters
from input frame, which integrates both color and semantic
prediction with the same backbone. Then, the Gaussian Re-
finement Network merges new Gaussians with the existing
set while avoiding redundancy. Finally, when significant
pose changes are detected, we perform only 1–5 iterations
of joint Gaussian–pose optimization to correct drift, remove
floaters, and further improve tracking accuracy. Experi-
ments on the real-world ScanNet and ScanNet++ bench-
marks demonstrate state-of-the-art semantic SLAM perfor-
mance, with strong generalization capability shown through
zero-shot transfer to the NYUv2 and TUM RGB-D datasets.
1. Introduction
Simultaneous Localization and Mapping (SLAM) is a long-
standing challenge in computer vision, aiming to recon-
struct a 3D map of an environment while simultaneously
estimating camera poses from a video stream. Semantic
visual SLAM extends this goal by producing dense maps
enriched with semantic labels, enabling applications in au-
tonomous driving, AR/VR, and robotics. By combining ge-
Figure 1. A radar chart comparing rendering and 3d semantic met-
rics. We normalize each metric independently, values closer to the
outer edge indicate better performance.
ometric reconstruction with object-level understanding, se-
mantic SLAM provides rich 3D spatial and semantic infor-
mation that allows robots and other systems to navigate and
interact with their surroundings more effectively.
Traditional visual SLAM systems consist of several in-
dependent components, including keypoint detection, fea-
ture matching, and bundle adjustment [2, 26, 27]. Their
scene representations are typically low-resolution voxels,
which limit geometric detail. Thus, although these systems
generally provide accurate camera localization, they strug-
gle to generate dense, high-quality 3D maps, which are re-
quired for robotics applications such as mobile manipula-
tion. Recent advances in differentiable rendering [15, 25]
introduces new options for scene representation in visual
SLAM. For example, neural scene representations such as
Neural Radiance Fields (NeRF) [25] have been successfully
adopted in SLAM frameworks [12, 37, 52]; however, NeRF
requires hours of per-scene optimization, making it compu-
tationally expensive and forcing a trade-off between recon-
struction quality and training cost.
Recently, 3D Gaussians have emerged as a powerful 3D
scene representation, offering fast, differentiable, and high-
1
arXiv:2506.06517v3  [cs.CV]  3 Dec 2025

<!-- page 2 -->
Figure 2. Comparison of PSNR with respect to number of Gaus-
sians across Gaussian Splatting SLAM algorithms (over an aver-
age of 2, 680 frames in the 6 testing scenes of ScanNet). Our
method achieves state-of-the-art performance with much fewer
Gaussians. GS Num represents the number of 3D Gaussians in
the scene after mapping is complete.
quality rendering capabilities [16]. Leveraging these advan-
tages, Gaussian-based representations have proven highly
effective for SLAM systems [14, 24]. However, existing
approaches still rely on test-time, gradient-based optimiza-
tion to estimate 3D Gaussians for each scene independently,
which is computationally expensive and unsuitable for real-
time applications. In addition, these methods depend on
heuristic Gaussian densification and pruning strategies [16],
often producing overly dense representations that fail to
scale to large, real-world environments.
In this paper, we propose GS4 (Generalizable Sparse
Splatting
Semantic
SLAM),
the
first
generalizable
Gaussian-splatting–based SLAM system, which directly
predicts 3D semantic Gaussians using a learned feed-
forward network,
eliminating the need for expensive
per-scene optimization. By integrating an image recogni-
tion backbone, GS4 jointly reconstructs geometry, color,
and semantic labels of the environment without relying on
any external semantic-segmentation modules.
GS4 begins with the Gaussian Prediction Model that in-
fers a sparse set of 3D semantic Gaussians from each in-
coming RGB-D frame in a feed-forward manner.
Next,
the Gaussian Refinement Network integrates these newly
predicted Gaussians with the evolving 3D map, replacing
the handcrafted heuristics traditionally used for Gaussian
densification and pruning. This learned refinement strategy
yields a compact representation with an order-of-magnitude
fewer Gaussians than competing methods. Finally, after the
global localization (bundle adjustment) step from the cam-
era tracking module updates camera poses and Gaussian lo-
cations, we perform a lightweight few-iteration (only 1∼5)
optimization of Gaussian parameters to preserve rendering
fidelity and mitigate the “floater” artifacts common in feed-
forward GS approaches.
We demonstrate that GS4 achieves state-of-the-art per-
formance across all key metrics in localization, mapping,
and segmentation on the real-world benchmark ScanNet
(Fig. 1), while using only ∼10% of Gaussians compared to
prior GS SLAM methods (Fig. 2). Furthermore, we high-
light the generalization capability of our system via zero-
shot transfer to the NYUv2 and TUM RGB-D datasets,
which, to the best of our knowledge, is the first demon-
stration of zero-shot semantic SLAM generalization in a
modern neural SLAM system.
In summary, our contributions are as follows:
• We propose GS4, the first generalizable Gaussian splat-
ting semantic SLAM approach on RGB-D sequences.
Results showed that GS4 obtains state-of-the-art on real
ScanNet and ScanNet++ scenes, and also zero-shot gen-
eralizes to the real NYUv2 and TUM RGB-D datasets
without any fine-tuning.
• Our proposed Gaussian refinement network effectively
merges Gaussians from different frames into a 3D rep-
resentation, while significantly reducing the number of
Gaussians required to represent a scene to only 10% −
25% of prior work.
• Our proposed few-iteration joint Gaussian-pose optimiza-
tion significantly improves reconstruction quality with a
small additional computational cost, while also slightly
improving tracking accuracy when high-quality ground-
truth RGB-D data are available.
2. Related Work
Traditional SLAM: Early visual SLAM methods [27]
demonstrated robust localization through effective keypoint
detection and matching, which resulted in sparse 3D re-
constructions.
While these approaches provided reliable
localization, the sparse nature of the reconstructed maps
limited their utility in applications requiring detailed 3D
maps. To address this issue, dense visual SLAM [7, 17] fo-
cused on constructing detailed maps to support applications
like augmented reality (AR) and robotics. Prior methods
[1, 3, 9, 30, 32, 42] employ representations based on Signed
Distance Fields (SDF), rather than relying on sparse repre-
sentations such as point clouds or grids. However, these ap-
proaches often suffer from over-smoothed reconstruction,
failing to capture fine details crucial for certain tasks.
NeRF-based SLAM: Neural Radiance Fields (NeRF) [25]
gained popularity as a 3D scene representation due to
its ability to generate accurate and dense reconstruc-
tions.
NeRF employs Multi-Layer Perceptron (MLP) to
encode scene information and performs volume rendering
by querying opacity and color along pixel rays. Methods
such as iMAP [37], NICE-SLAM [52], and ESLAM [12]
incorporate this implicit scene representation into SLAM,
leveraging NeRF’s high-fidelity reconstructions to improve
both localization and mapping. DNS-SLAM [18] further in-
2

<!-- page 3 -->
corporates semantic information into the framework. How-
ever, the volumentric rendering process in NeRF is costly,
often requiring trade-offs such as limiting the number of
pixels during rendering, These trade-offs, while improving
efficiency, may compromise the system’s accuracy in both
localization and mapping.
GS-based SLAM: 3D Gaussian Splatting (3DGS) [16] em-
ploys splatting rasterization instead of ray marching. This
approach iterates over 3D Gaussian primitives rather than
marching along rays, resulting in a more expressive and ef-
ficient representation capable of capturing high-fidelity 3D
scenes with significantly faster rendering speed.
Hence,
GS-based SLAM systems achieve improved accuracy and
speed in dense scene reconstruction. SplaTAM [13] intro-
duces silhouette-guided rendering to support structured map
expansion, enabling efficient dense visual SLAM. Gaus-
sian Splatting SLAM [24] integrates novel Gaussian inser-
tion and pruning strategies, while GS-ICP SLAM [10] and
RTG-SLAM [31] combine ICP with 3DGS to achieve both
higher speed and superior map quality. Expanding upon
these advancements, SGS-SLAM [20], OVO-SLAM [23],
SemGauss-SLAM [50] and GS3LAM [19] extend 3D Gaus-
sian representations to include semantic scene understand-
ing.
However, existing GS-based SLAM methods em-
ploy per-scene optimization, requiring iterative refinement
of Gaussians initialized from keyframes through rendering
supervision. As a result, they all rely on additional segmen-
tation models to predict semantic labels for each image, cre-
ating computational overhead.
Feed-forward Models: Recently, MV-DUSt3R+ [38] and
VGGT [39] have demonstrated multi-view inference by ex-
tending the DUSt3R [40] architecture for multi-view recon-
struction. Meanwhile, several works have introduced feed-
forward approaches for scene-level 3DGS reconstruction
using generalizable models [4, 5, 11, 21, 43, 46]. In par-
ticular, pixelSplat [4] predicts 3D Gaussians directly from
image features, while DepthSplat [43] connects Gaussian
splatting and depth estimation, studying their interaction
to enable feed-forward 3D Gaussian reconstruction from
multi-view images. AnySplat [11] further predicts both 3D
Gaussian primitives and camera intrinsics/extrinsics for un-
calibrated multi-view inputs. However, to the best of our
knowledge, these feed-forward models have so far been
applied only to a relatively small number of input images
and have not been scaled to the thousands of frames typ-
ical in SLAM, nor have they been incorporated into GS-
based semantic SLAM systems operating over such long
sequences.
3. Methods
In this section, we describe our proposed SLAM approach.
We first provide a brief overview of Gaussian Splatting, then
detail our Gaussian prediction network and Gaussian refine-
ment network. Finally, we explain how these networks are
utilized within the entire SLAM system.
3.1. Gaussian Splatting
We represent a 3D map using a set of anisotropic 3D Gaus-
sians. Each Gaussian Gi is characterized by RGB color
ci ∈R3, center position µi ∈R3, scale si ∈R3, quater-
nion ri ∈R4, opacity oi ∈R and semantic class vector
vclass
i
∈RN, where N is the number of classes.
The rendering process is defined as:
Qp =
X
i∈N
qiαi
i−1
Y
j=1
(1 −αj),
where Qp is a quantity of a pixel p to be rendered, which
can be color, depth or semantic label, and qi is that quantity
of the i-th 3D Gaussian, while αi is its visibility, computed
from opacity and covariance parameters (determined by ro-
tation and scale). Following [14], We also render a silhou-
ette image to determine visibility Sp = P
i∈N αi
Qi−1
j=1(1−
αj).
3.2. Gaussian Prediction and Refinement
Our proposed Gaussian prediction network (Fig. 3) takes
RGB-D images as input and predicts 3D Gaussian parame-
ters. Importantly, the backbone generates features that can
predict semantic labels (e.g. trained from 2D segmentation
tasks), enabling the rendering of photometric, geometric,
and semantic views. Next, the Gaussian refinement network
processes Gaussians predicted from a new frame and learns
to merge them with the 3D scene representation computed
from prior frames.
3.2.1. Backbone for Gaussian Prediction
We train a transformer model to regress 3D GS param-
eters from an image with a known camera pose (from
tracking, described in Sec. 3.3.1), while simultaneously as-
signing semantic labels to these 3D Gaussians. We start
with a pre-trained 2D image segmentation model such as
Mask2Former [6] or AutoFocusFormer [53], which encodes
an image into encoder tokens f l
enc and decoder tokens f l
dec
(from their image decoder) at several progressively down-
sampled levels l = 1, . . . , L, with L = 4 usually. We con-
catenate an RGB image I ∈RH×W ×3 and a depth image
D ∈RH×W ×1 resulting in a 4-channel feature map that is
fed into the model:
{f l
enci, f l
deci}i=1:Ntoken = Backbone([I, D])
where Ntoken denotes the total number of prediction tokens
per image.
The variable l represents the network level
at which Gaussians are predicted. If the second level is
chosen, the feature map usually has a spatial resolution of
3

<!-- page 4 -->
Figure 3. Overview of the SLAM System. At each timestep, the system receives an RGB-D frame as input. The tracking system
performs local camera tracking and global localization to determine the current frame’s pose and correct previous pose errors. Our 3D
mapping process comprises three main components: 1) Gaussian Prediction (Sec 3.2.1): Utilizing the current frame’s RGB-D data, the
Gaussian Prediction Model estimates the parameters and semantic labels for all Gaussians in the current frame; 2) Gaussian Refinement
(Sec 3.2.2): Both newly added Gaussians and those in the existing semantic 3D map are refined using the Gaussian Refinement Network to
ensure that the combined set of Gaussians accurately represents the scene. A covisibility check ensures that only non-overlapping Gaussians
are integrated into the existing 3D map. Post-refinement, the transparent Gaussians are pruned; 3) Few-Iteration Joint Gaussian–Pose
Optimization (Sec. 3.3.2): If significant pose corrections are detected, we perform a few iterations of joint Gaussian–pose optimization to
update the Gaussians in the 3D map and further refine the poses; the refined poses are then fed back into the tracking system. This ensures
consistency of the 3D map with the revised camera trajectories and further improves pose accuracy. (Best viewed in color.)
H/8 × W/8, resulting in Ntoken = HW/64 tokens per im-
age.
Processing Prediction Tokens with Transformer. Given
the selected prediction level, we concatenate the encoder
features fenci and decoder features fdeci for each token i,
and process the resulting tokens using local-attention trans-
former layers in the image space to obtain the final fi fea-
tures for the i-th token, integrating information from both
the encoder and decoder.
Decoding Prediction Tokens to Gaussians. Each output
token’s features, fi, from the transformer layers are de-
coded into Gaussian parameters using Multi-Layer Percep-
tron (MLP):
{∆xi, ∆yi, ∆di, ∆ci, si, ri, oi} = MLP(fi),
Here, ∆xi and ∆yi represent the offsets from the 2D posi-
tion (xi, yi) of the token fi in the image space, while ∆di
is the offset for the noisy depth di obtained from the depth
image. These offsets are added to the original values, which
are then backprojected into 3D space using the intrinsic and
extrinsic parameters of the camera, yielding the 3D center
position µi. Similarly, ∆ci represents the offset for the RGB
values, obtained from the downsampled image, where each
token corresponds to a single pixel. Adding the offset to this
value yields the final RGB color for each Gaussian. Besides
Gaussian parameters, the mask decoder head in the segmen-
tation model predicts token-level semantic segmentation la-
bel vector vclass
i
for the input image, which we then assign to
the associated Gaussian. Finally, each Gaussian is assigned
the final feature vector fi of its corresponding token for the
subsequent Gaussian refinement stage.
To supervise the prediction of semantic segmentation,
we follow the setup in Mask2Former [6]. We denote the
corresponding segmentation loss by Lseg. In addition, we
render images at M supervision views—comprising the
current input view and randomly selected novel views that
overlap with the current input—using the predicted Gaus-
sians from the current input, and minimize RGB-D and se-
mantic rendering loss. For novel view supervision, we fo-
cus solely on areas visible in the input view, ensuring that
the optimization process focuses on regions consistently ob-
served across both input and novel views. We explain the
loss functions used during training below.
RGB Rendering Loss.
Following previous work [46,
54],
we
use
a
combination
of
the
Mean
Squared
Error
(MSE)
loss
and
Perceptual
loss:
Lrgb
=
1
M
PM
v=1 (MSE (Igt
v , Ipre
v
) + λ · PER (Igt
v , Ipre
v
)) , where
λ is the weight for the perceptual loss.
Depth Rendering Loss. For depth images, we use L1 loss:
Ld =
1
M
PM
v=1 L1 (Dgt
v , Dpre
v
) .
Semantic
Rendering
Loss.
For
semantic
render-
ing,
we
use
the
cross
entropy
loss:
LSem
=
4

<!-- page 5 -->
1
M
PM
v=1 Cross Entropy
 Semgt
v , Sempre
v

. where the ren-
dered semantic image has N channels, each corresponding
to a different semantic category.
Overall Training Loss. Our total loss comprises multiple
rendering losses and the segmentation loss Lseg: L = λrgb·
Lrgb +λd ·Ld +λSem ·LSem +Lseg, where we use λrgb =
1.0, λd = 1.0 and λSem = 0.1.
3.2.2. Gaussian Refinement Network
The previous subsection predicts Gaussian parameters from
a single frame. In our SLAM system, as new frames ar-
rive, we insert Gaussians from the frame into unmapped
regions of the current 3D reconstruction. We perform co-
visibility, which involves rendering a silhouette image for
the new frame to identify the regions where new Gaus-
sians should be inserted [13]. To ensure that the combined
set of Gaussians accurately represents the scene, we pro-
pose a novel Gaussian Refinement Network to refine both
the existing Gaussians in the 3D map and the newly added
ones, enabling their effective merging. The input to the net-
work includes the features fi and 3D positions µi ∈R3
of all Gaussians from the 3D map that are visible in the
new frame, as well as Gaussians from the new frame. We
process these using several local-attention transformer lay-
ers with 3D neighborhoods in the world coordinate system
to fuse and update the features for each Gaussian. Subse-
quently, MLP layers predict updates ∆ci ∈R3, ∆si ∈R3,
∆ri ∈R4 and ∆oi ∈R for each Gaussian. These updates
refine the Gaussians to accurately render both current and
previous views. To supervise the network, we render the
current view along with previous overlapping views. The
total training loss is:
Lrefine = λrgb · Lrgb + λd · Ld + λSem · LSem
(1)
where we use λrgb = 1.0, λd = 1.0 and λSem = 0.1. After
Gaussian refinement, we prune Gaussians whose updated
opacity falls below 0.005, effectively removing those that
have become unimportant after merging. These merging-
pruning steps lead to a significantly reduced number of
Gaussians in the final 3D map with little performance im-
pact.
During testing time, we introduce a threshold U to man-
age the uncertainty of each Gaussian. Once a Gaussian has
been updated U times by the refinement network, we con-
sider its uncertainty sufficiently reduced and exclude them
from further updates. We set U = 8 in our experiments.
3.3. The SLAM System
An overview of the system is summarized in Fig. 3. The
system always maintains a set of 3D Gaussians representing
the entire scene. For each new RGB-D image, the Gaussian
prediction network predicts 3D Gaussian parameters, which
can be rendered into high-fidelity color, depth, and seman-
tic images. The Gaussian refinement network refines both
the existing Gaussians in the 3D map and the newly added
ones to accurately render both current and previous views.
During testing, we occasionally run few-iteration test-time
optimization and refine 3D Gaussians in the map to reflect
camera pose updates from loop closure and bundle adjust-
ment in the tracking module, while jointly optimizing the
camera poses of these frames.
3.3.1. Tracking and Global Bundle Adjustment
Our main contribution is on the mapping side, hence for
camera tracking in our SLAM system, we can adopt any off-
the-shelf algorithm [28, 48]. A SLAM tracking system usu-
ally consists of two components: local camera tracking and
global localization. In local camera tracking, a keyframe is
initialized when sufficient motion is detected, and loop clo-
sure (LC) is performed. Meanwhile, global localization per-
forms full bundle adjustment (BA) for real-time global re-
finement once the system contains more than 25 keyframes.
Both LC and BA help address the problem of accumulated
errors and drift that can occur during the localization pro-
cess.
3.3.2. Few-Iteration Joint Gaussian–Pose Optimization
Loop closure and bundle adjustment are essential compo-
nents in SLAM systems, employed to correct accumulated
drift and adjust the camera poses of previous frames. How-
ever, these adjustments can cause Gaussians inserted based
on earlier, uncorrected poses to misalign with the scene,
leading to inaccurate rendering and mapping. It is crucial
to implement a mechanism that updates the Gaussians in
the 3D map following pose corrections. To address this is-
sue, we propose using rendering-based optimization to up-
date the Gaussian parameters ci ∈R3, µi ∈R3, S ∈R3,
Q ∈R4 and oi ∈R with only a few iterations. Specif-
ically, we re-render RGB-D images for the top-k frames,
selected based on significant pose changes, and jointly min-
imize a rendering loss with respect to both the Gaussian pa-
rameters and the camera poses of these frames. The op-
timized camera poses are then fed back into the tracking
system, and used for loop closure and bundle adjustment.
This joint optimization keeps the 3D map consistent with
the corrected poses while also leveraging the full 3D map to
further improve pose accuracy. To enhance the efficiency of
this optimization, we employ the batch rendering technique
from [44]. We omit semantic image rendering to improve
system efficiency. For few-iteration optimization, we add a
SSIM term to the RGB loss, following [16]:
Lopt = 1
M
M
X
i′=1
 λrgb ·
 (1 −λ) · L1
 Igt
i′ , Ipre
i′

+
λ
 1 −SSIM(Igt
i′ , Ipre
i′
)

+ λd · L1(Dgt
i′ , Dpre
i′ )

(2)
where λ is set to 0.2 for all experiments. We perform 1–5
iterations in our experiments.
5

<!-- page 6 -->
4. Experiments
4.1. Experimental Setup
Training Settings. We train our Gaussian prediction and
refinement networks entirely on RGB-D videos from the
real ScanNet datasets. We exclude the six standard SLAM
test scenarios and use all remaining training and valida-
tion scenes, supervising with 20 common semantic classes.
For ScanNet++, we exclude the SLAM test scenarios and
use all other available training scenes, supervising with the
100 most common semantic classes. We adopt AutoFocus-
Former [53] as the backbone for both Mask2Former and
Gaussian prediction, using the second stage of the back-
bone as the prediction stage.
Following the low-to-high
resolution curriculum of [54], we train the Gaussian pre-
diction network in three stages with input resolutions of
256×256, 480×480, and 640×480. In the first two stages,
images are resized such that the shorter side is 256 or 480
pixels and then center-cropped to a square. On ScanNet++,
we use resolutions of 256x384. For the refinement network,
which processes multiple consecutive frames, we adopt a
progressive training schedule: beginning with two frames,
then four, and finally eight.
Evaluation Datasets and Settings.
During testing, we
evaluate our method on six real-world scenes from Scan-
Net [8], which are commonly used as SLAM benchmarks,
and six real-world scenes from ScanNet++ [45]. ScanNet++
validation set contains 11 scenes with continuous camera
trajectories, but SplaTAM and SGS-SLAM completely fail
on 5 of them. Hence we only use the 6 scenes they can finish
running, so that their numbers are not infinitely bad. Note
that our approach can successfully run on all 11 scenes. Ad-
ditionally, we perform zero-shot experiments on real scenes
from NYUv2 [29] and TUM RGB-D [36]. For ScanNet,
NYUv2, and TUM RGB-D, we evaluate rendering perfor-
mance on every 5th frame of each sequence. For Scan-
Net++, we evaluate all training views of each RGB-D video
and additionally hold out the novel views provided by the
dataset.
Evaluation Metrics.
We use PSNR, Depth-L1 [51],
SSIM [41], and LPIPS [47] to evaluate the reconstruction
and rendering quality. We additionally report reconstruc-
tion metrics such as Accuracy, Completion, Completion Ra-
tio (<7cm) and F1 (<7cm) in the appendix. For GS-based
SLAM methods, we also report the number of Gaussians.
For semantic segmentation, we report the mean Intersection
over Union (mIoU). To evaluate the accuracy of the camera
pose, we adopt the average absolute trajectory error (ATE
RMSE) [35].
Baselines. We compare our method against several state-
of-the-art approaches: NeRF-based SLAM methods, in-
cluding NICE-SLAM [51], GO-SLAM [48], and Point-
SLAM [34]; 3D Gaussian-based SLAM methods such as
Table 1. Rendering Performance on ScanNet. Values are aver-
aged across the test scenes. Best results are highlighted as first ,
second . GS Num represents the number of 3D Gaussians in-
cluded in the scene after mapping is complete.
Res
Method
PSNR↑
SSIM↑
LPIPS↓
Depth L1↓
GS Num↓
NICE-SLAM
17.54
0.621
0.548
-
-
Point-SLAM
19.82
0.751
0.514
-
-
SplaTAM
18.99
0.702
0.364
7.21
2466k
640 × 480
RTG SLAM
12.75
0.372
0.761
97.56
1229k
GS-ICP SLAM
14.73
0.645
0.684
103.31
2565k
SGS SLAM
15.89
0.594
0.615
11.83
2114k
GS3LAM
20.67
0.796
0.288
11.88
2154k
GS4 (Ours, 1 iter)
22.62
0.851
0.338
6.46
321k
GS4 (Ours, 5 iters)
24.55
0.885
0.299
4.86
224k
GO-SLAM
18.21
0.657
0.553
18.14
-
320 × 240
GS4 (Ours, 1 iter)
22.54
0.885
0.240
5.98
162k
GS4 (Ours, 5 iters)
24.31
0.921
0.196
4.93
124k
SplaTAM [14], RTG-SLAM [31], and GS-ICP SLAM [10];
and semantic 3D Gaussian-based SLAM methods, includ-
ing SGS-SLAM [20], GS3LAM [19] and OVO-Gaussian-
SLAM [23]. SGS-SLAM, GS3LAM and OVO-Gaussian-
SLAM are the only semantic SLAM methods available for
comparison since the code is not available for other se-
mantic SLAM approaches.
Note that SGS-SLAM [20]
and GS3LAM [19] employ test-time optimization using
ground truth semantic labels on the test set. SGS-SLAM
and GS3LAM have been shown to outperform all other
existing semantic SLAM methods [18, 49]. To ensure a
fair comparison and simulating SLAM applications in real-
world scenarios where ground truth semantic labels are un-
available, we trained a 2D segmentation model using a
Swin backbone [22] with Mask2Former [6] on ScanNet
and ScanNet++, following the same training strategy as
our model, and used predicted semantic labels to supervise
SGS-SLAM nad GS3LAM.
4.2. Results
Rendering and Reconstruction Performance. In Table 1,
we evaluate the rendering and reconstruction performance
of our method on ScanNet. This is a difficult task compared
to the synthetic data where neural RGB-D SLAM meth-
ods usually show strong results, because inevitably inaccu-
rate ground truth camera poses and depths make optimiza-
tion much harder than completely clean synthetic datasets.
Compared to existing dense neural RGB-D SLAM meth-
ods, our approach achieves state-of-the-art performance on
PSNR, SSIM, and Depth L1 metrics.
Specifically, our
method surpasses the runner-up, GS3LAM [18], by 3.88
dB in PSNR (a 18.8% percent improvement). and 0.089
in SSIM (a 11.2% percent improvement). Furthermore, our
approach utilizes approximately 10x fewer Gaussians than
the baselines. This efficiency highlights the effectiveness of
our method in achieving high-quality scene representation
with reduced computational complexity.
In Table 1, we ran our method with 240 × 320 input to
6

<!-- page 7 -->
Table 2. Semantic Performance across ScanNet Test Scenes
Methods
SGS SLAM
OVO-Gaussian-SLAM
GS3LAM
GS4 (Ours, 1 iter)
GS4 (Ours, 5 iters)
mIoU(2D)
37.20
%
56.42
63.71
62.10
mIoU(3D)
18.87
32.58
34.42
54.84
53.61
Table 3. Tracking Performance on ScanNet Test Scenes. The
average values are reported. GS4 uses the same tracking algorithm
as GO-SLAM hence the numbers are almost the same.
Metric
NICE-SLAM
Point-SLAM
SplaTAM
RTG SLAM
GS-ICP SLAM
ATE RMSE [cm]↓
10.70
12.19
11.88
144.52
NaN
Metric
SGS SLAM
GS3LAM
GO-SLAM
GS4 (Ours, 1 iter)
GS4 (Ours, 5 iters)
ATE RMSE [cm]↓
40.97
30.88
7.00
6.97
6.95
Table 4. Average Runtime on ScanNet Test Scenes
Methods
Point-SLAM
SplaTAM
RTG-SLAM
GS-ICP SLAM
FPS ↑
0.05
0.23
1.01
3.62
Methods
SGS-SLAM
GS3LAM
GS4 (ours, 1 iter)
GS4 (ours, 5 iters)
FPS ↑
0.17
0.12
2.85
1.82
compare against GO-SLAM which shares the same tracking
method as ours but renders at the same low resolution. GS4
maintains the same PSNR and depth prediction quality as its
high resolution version and significantly outperforms GO-
SLAM across all metrics.
Fig. 4 shows visual results of RGB and depth rendering.
Our method demonstrates superior performance than other
GS-SLAM methods. Notably, sometimes the depth maps
of our approach even turn out to be better than the noisy
ground truth depth inputs. For instance, in the first two
columns, our method delivers a more contiguous and com-
plete rendering of the bicycle tires. Similarly, in the middle
two columns, we reconstruct the chair’s backrest nearly en-
tirely, whereas the GT depth data lacks this detail.
Semantic Performance. In Table 2, we present both 2D
rendering and 3D mean Intersection over Union (mIoU)
scores across the six ScanNet test scenes. For 3D mIoU
evaluation, we first align the reconstructed map with the
ground-truth mesh and then use 3D neighborhood voting
to assign predicted labels. Our method outperforms the pre-
vious runner-up, GS3LAM, by 20.42% in 3D mIoU and by
7.29% in 2D mIoU. Qualitative comparisons are provided
in the appendix.
Tracking Performance.
Table 3 shows the tracking re-
sults. Our method uses the same tracking algorithm as GO-
SLAM, which is significantly better than other GS-based
SLAM methods.
Runtime Comparison. Table 4 presents a runtime compar-
ison of our method against the baselines at the 640 × 480
resolution, conducted on an Nvidia RTX TITAN. FPS is cal-
culated by dividing the total number of frames by the total
time to represent the overall system performance. While
GS-ICP SLAM is faster than ours, its rendering and track-
ing performance is significantly worse (Table 1 and 3).
Our approach is 12x faster than SplaTAM, 17x faster than
SGS-SLAM, and 24x faster than GS3LAM. Notably, even
Table 5. Comparison of averaged performance on ScanNet++
at 256×384 resolution. The 2D rendering metrics are averaged
over both training and novel views. GS Num denotes the number
of 3D Gaussians after mapping is complete, and FPS denotes the
runtime in frames per second.
Method
Rendering Metrics
mIoU (3D)↑
GS Num↓
ATE RMSE↓
FPS↑
PSNR↑
SSIM↑
LPIPS↓
mIoU (2D)↑
SplaTAM
20.34
0.758
0.331
%
%
1899k
1588.95
0.17
SGS-SLAM
15.59
0.533
0.470
15.35
0.58
2374k
1315.89
0.13
GS4 (Ours)
21.06
0.740
0.281
18.92
13.80
85k
5.37
1.60
Table 6.
Zero-shot Rendering Performance on NYUv2 and
TUM-RGBD. Values are averaged across the test scenes. GS Num
represents the number of 3D Gaussians in the scene after mapping.
Dataset
Res
Method
PSNR↑
SSIM↑
LPIPS↓
GS Num↓
SplaTAM
18.86
0.692
0.372
1236k
NYUv2
640 × 480
RTG-SLAM
11.84
0.221
0.703
807k
SGS-SLAM
19.32
0.708
0.357
1108k
GS4 (Ours)
22.24
0.866
0.254
298k
SplaTAM
22.76
0.891
0.182
803k
TUM RGBD
640 × 480
RTG-SLAM
19.75
0.769
0.395
198k
SGS-SLAM
22.44
0.876
0.184
735k
GS4 (Ours)
22.70
0.903
0.191
166k
with this exceptional speed, our approach maintains supe-
rior map quality and outperforms other methods.
ScanNet++ Experiments. Table 5 reports 2D rendering
performance averaged over both training and novel views,
as well as 3D semantic, tracking, and runtime compar-
isons against the baselines at a resolution of 256×384, av-
eraged over six scenes. Our method achieves state-of-the-
art performance on both 2D and 3D tasks, outperforming
the runner-up by 0.72dB in PSNR (3.5% relative improve-
ment), 0.05 in LPIPS (15.1% relative improvement), 3.57%
in 2D mIoU, and 13.22% in 3D mIoU, while using only
3.0%–4.5% of the Gaussians required by other methods.
Compared to SGS-SLAM, although it achieves the second-
best 2D mIoU, its 3D mIoU drops to just 0.58%, indicat-
ing poor generalization to the full 3D scene. In addition
to delivering superior performance across both 2D and 3D
metrics, our method also exhibits substantially higher effi-
ciency, running 9× faster than SplaTAM and 12× faster than
SGS-SLAM.
Zero-shot Experiments. In Table 6, we report quantita-
tive zero-shot results.
For NYUv2, the numbers are av-
eraged over three scenes, and for TUM-RGBD, they are
also averaged over three scenes. Per-scene results are pro-
vided in the appendix. On NYUv2, our method outperforms
all other GS-based SLAM approaches across all rendering
metrics, achieving a 15%–20% relative improvement over
the runner-up while using significantly fewer Gaussians.
Qualitative comparisons are also provided in the appendix.
On TUM-RGBD, our method outperforms the baselines in
terms of SSIM and the number of Gaussians, and closely
matches the best performance in other metrics, despite rely-
ing primarily on a feed-forward model trained on ScanNet.
7

<!-- page 8 -->
GS-ICP SLAM
SplaTAM
SGS SLAM
GS3LAM
GS4 (Ours)
Ground Truth
scene0000
scene0169
scene0207
Figure 4. Renderings on ScanNet. Our method, GS4, renders color & depth for views with fidelity significantly better than all approaches.
Table 7. Ablation on ScanNet (averaged over test scenes)
Design Choice
PSNR [dB]↑
SSIM↑
LPIPS↓
Depth L1↓
mIoU↑
ATE ↓
Gs Num↓
GS Prediction
15.05
0.461
0.662
33.17
40.11
6.99
133k
+ GS Refinement
16.1
0.556
0.584
29.81
44.65
7.00
581k
+ 1-Iter. Optimization
22.72
0.851
0.337
6.24
63.58
6.97
666k
+ GS Pruning (Full SLAM)
22.62
0.851
0.338
6.46
63.71
6.97
321k
Ablation Study. We conduct an ablation study using all
ScanNet test scenes, as shown in Table 7.
The results
demonstrate that both the Gaussian Refinement Network
and the Few-Iteration Joint Gaussian–Pose Optimization
are critical to the performance of GS4. Additionally, Gaus-
sian pruning significantly reduces the number of Gaussians
without sacrificing accuracy.
Ablation Study of Few-Iteration Joint Gaussian–Pose
Optimization. In Sec. 3.3.2, we propose to perform few-
iteration joint Gaussian–pose optimization and feed the re-
fined poses back into the tracking system to further improve
pose accuracy. In Table 8, we ablate (i) Gaussian-only op-
timization and (ii) joint Gaussian–pose optimization with
pose feedback into the tracking system. For the rendering
metrics, joint Gaussian–pose optimization yields slight but
consistent improvements over Gaussian-only optimization
Table 8. Ablation of joint Gaussian–pose optimization on Scan-
Net and ScanNet++ (averaged over test scenes). Results are re-
ported using 5 iterations of the few-iteration joint Gaussian–pose
optimization.
Dataset
Optimization
PSNR↑
SSIM↑
LPIPS↓
GS Num↓
ATE RMSE↓
ScanNet
Gaussian
24.39
0.881
0.302
225k
6.94
Gaussian–Pose
24.55
0.885
0.299
224k
6.95
ScanNet++
Gaussian
20.88
0.736
0.282
90k
6.34
Gaussian–Pose
21.06
0.740
0.281
85k
5.37
on both datasets. For tracking accuracy, we observe that
pose feedback is particularly effective when high-quality
ground truth is available, as in ScanNet++, whose RGB-D
images are of much higher quality than those in ScanNet.
5. Conclusion
We present GS4, a novel SLAM system that incrementally
constructs and updates a 3D semantic scene representation
from a RGB-D video with a learned generalizable network.
Our novel Gaussian refinement network and few-iteration
joint Gaussian-pose optimization significantly improve the
performance of our approach.
Our experiments demon-
strate state-of-the-art semantic SLAM performance on the
8

<!-- page 9 -->
ScanNet benchmark while running 10x faster and using
10x less Gaussians than baselines. The model also showed
strong generalization capabilities through zero-shot transfer
to the NYUv2 and TUM RGB-D datasets. In future work,
we will further improve the computational speed of GS4
and explore options for a pure RGB-based SLAM approach.
References
[1] Erik Bylow, J¨urgen Sturm, Christian Kerl, Fredrik Kahl, and
Daniel Cremers. Real-time camera tracking and 3d recon-
struction using signed distance functions. In Robotics: Sci-
ence and Systems, page 2, 2013. 2
[2] Carlos Campos, Richard Elvira, Juan J G´omez Rodr´ıguez,
Jos´e MM Montiel, and Juan D Tard´os. Orb-slam3: An accu-
rate open-source library for visual, visual–inertial, and mul-
timap slam. IEEE Transactions on Robotics, 37(6):1874–
1890, 2021. 1
[3] Daniel R Canelhas, Todor Stoyanov, and Achim J Lilienthal.
Sdf tracker: A parallel algorithm for on-line pose estima-
tion and scene reconstruction from depth images. In 2013
IEEE/RSJ International Conference on Intelligent Robots
and Systems, pages 3671–3676. IEEE, 2013. 2
[4] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent
Sitzmann. pixelsplat: 3d gaussian splats from image pairs for
scalable generalizable 3d reconstruction. In CVPR, 2024. 3
[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. arXiv preprint arXiv:2403.14627, 2024.
3
[6] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexan-
der Kirillov, and Rohit Girdhar.
Masked-attention mask
transformer for universal image segmentation.
In CVPR,
2022. 3, 4, 6
[7] Jan Czarnowski, Tristan Laidlow, Ronald Clark, and An-
drew J Davison. Deepfactors: Real-time probabilistic dense
monocular slam. IEEE Robotics and Automation Letters, 5
(2):721–728, 2020. 2
[8] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
Richly-annotated 3d reconstructions of indoor scenes.
In
Proc. Computer Vision and Pattern Recognition (CVPR),
IEEE, 2017. 6
[9] Angela Dai, Matthias Nießner, Michael Zollh¨ofer, Shahram
Izadi, and Christian Theobalt.
Bundlefusion: Real-time
globally consistent 3d reconstruction using on-the-fly surface
reintegration. ACM Transactions on Graphics (ToG), 36(4):
1, 2017. 2
[10] Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. Rgbd gs-icp
slam, 2024. 3, 6
[11] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren,
Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng
Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting
from unconstrained views. arXiv preprint arXiv:2505.23716,
2025. 3
[12] Mohammad Mahdi Johari, Camilla Carta, and Franc¸ois
Fleuret. Eslam: Efficient dense slam system based on hybrid
representation of signed distance fields. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 17408–17419, 2023. 1, 2
[13] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallab-
hula, Gengshan Yang, Sebastian Scherer, Deva Ramanan,
and Jonathon Luiten. Splatam: Splat, track & map 3d gaus-
sians for dense rgb-d slam. arXiv preprint arXiv:2312.02126,
2023. 3, 5
[14] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat, track & map 3d gaussians
for dense rgb-d slam. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, 2024.
2, 3, 6
[15] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 1
[16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4), 2023. 2, 3, 5
[17] Christian Kerl, J¨urgen Sturm, and Daniel Cremers. Dense
visual slam for rgb-d cameras. In 2013 IEEE/RSJ Interna-
tional Conference on Intelligent Robots and Systems, pages
2100–2106. IEEE, 2013. 2
[18] Kunyi Li, Michael Niemeyer, Nassir Navab, and Federico
Tombari. Dns slam: Dense neural semantic-informed slam,
2023. 2, 6
[19] Linfei Li, Lin Zhang, Zhong Wang, and Ying Shen. Gs3lam:
Gaussian semantic splatting slam.
In Proceedings of the
32nd ACM International Conference on Multimedia, page
3019–3027, New York, NY, USA, 2024. Association for
Computing Machinery. 3, 6
[20] Mingrui Li, Shuhong Liu, Heng Zhou, Guohao Zhu, Na
Cheng, Tianchen Deng, and Hongyu Wang. Sgs-slam: Se-
mantic gaussian splatting for neural dense slam.
arXiv
preprint arXiv:2402.03246, 2024. 3, 6
[21] Tianqi Liu, Guangcong Wang, Shoukang Hu, Liao Shen,
Xinyi Ye, Yuhang Zang, Zhiguo Cao, Wei Li, and Ziwei Liu.
Mvsgaussian: Fast generalizable gaussian splatting recon-
struction from multi-view stereo. In European Conference
on Computer Vision, pages 37–53. Springer, 2025. 3
[22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng
Zhang, Stephen Lin, and Baining Guo. Swin transformer:
Hierarchical vision transformer using shifted windows. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), 2021. 6
[23] Tomas Berriel Martins, Martin R Oswald, and Javier Civera.
Open-vocabulary online semantic mapping for slam. arXiv
preprint arXiv:2411.15043, 2024. 3, 6, 12
[24] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and An-
drew J. Davison. Gaussian Splatting SLAM. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, 2024. 2, 3, 12
9

<!-- page 10 -->
[25] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 1, 2
[26] Raul Mur-Artal and Juan D Tard´os. Orb-slam2: An open-
source slam system for monocular, stereo, and rgb-d cam-
eras. IEEE transactions on robotics, 33(5):1255–1262, 2017.
1
[27] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D
Tardos. Orb-slam: a versatile and accurate monocular slam
system. IEEE transactions on robotics, 31(5):1147–1163,
2015. 1, 2
[28] Riku Murai, Eric Dexheimer, and Andrew J Davison.
Mast3r-slam: Real-time dense slam with 3d reconstruction
priors. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 16695–16705, 2025. 5, 13
[29] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob
Fergus.
Indoor segmentation and support inference from
rgbd images. In ECCV, 2012. 6
[30] Richard A Newcombe, Shahram Izadi, Otmar Hilliges,
David Molyneaux, David Kim, Andrew J Davison, Pushmeet
Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon.
Kinectfusion: Real-time dense surface mapping and track-
ing. In 2011 10th IEEE international symposium on mixed
and augmented reality, pages 127–136. Ieee, 2011. 2
[31] Zhexi Peng, Tianjia Shao, Liu Yong, Jingke Zhou, Yin Yang,
Jingdong Wang, and Kun Zhou.
Rtg-slam: Real-time 3d
reconstruction at scale using gaussian splatting.
In ACM
SIGGRAPH Conference Proceedings, Denver, CO, United
States, July 28 - August 1, 2024, 2024. 3, 6
[32] Victor Adrian Prisacariu, Olaf K¨ahler, Stuart Golodetz,
Michael Sapienza, Tommaso Cavallari, Philip HS Torr, and
David W Murray.
Infinitam v3: A framework for large-
scale 3d reconstruction with loop closure.
arXiv preprint
arXiv:1708.00783, 2017. 2
[33] Erik Sandstr¨om, Keisuke Tateno, Michael Oechsle, Michael
Niemeyer, Luc Van Gool, Martin R Oswald, and Federico
Tombari. Splat-slam: Globally optimized rgb-only slam with
3d gaussians. arXiv preprint arXiv:2405.16544, 2024. 12
[34] Erik Sandstr¨om, Yue Li, Luc Van Gool, and Martin R. Os-
wald. Point-slam: Dense neural point cloud-based slam. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), 2023. 6
[35] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
Burgard, and Daniel Cremers. A benchmark for the eval-
uation of rgb-d slam systems. In 2012 IEEE/RSJ Interna-
tional Conference on Intelligent Robots and Systems, pages
573–580, 2012. 6
[36] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cre-
mers. A benchmark for the evaluation of rgb-d slam systems.
In Proc. of the International Conference on Intelligent Robot
Systems (IROS), 2012. 6
[37] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davi-
son. imap: Implicit mapping and positioning in real-time. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 6229–6238, 2021. 1, 2
[38] Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu,
Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan.
Mv-dust3r+: Single-stage scene reconstruction from sparse
views in 2 seconds. arXiv preprint arXiv:2412.06974, 2024.
3
[39] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny.
Vggt:
Visual geometry grounded transformer. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2025. 3
[40] Shuzhe Wang,
Vincent Leroy,
Yohann Cabon,
Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In CVPR, 2024. 3
[41] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing, 13(4):
600–612, 2004. 6
[42] Thomas Whelan, Hordur Johannsson, Michael Kaess, John J
Leonard, and John McDonald.
Robust real-time visual
odometry for dense rgb-d mapping. In 2013 IEEE Interna-
tional Conference on Robotics and Automation, pages 5724–
5731. IEEE, 2013. 2
[43] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys.
Depthsplat: Connecting gaussian splatting and depth.
In
CVPR, 2025. 3
[44] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen,
Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey
Hu, Matthew Tancik, and Angjoo Kanazawa.
gsplat: An
open-source library for Gaussian splatting. arXiv preprint
arXiv:2409.06765, 2024. 5
[45] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner,
and Angela Dai. Scannet++: A high-fidelity dataset of 3d in-
door scenes. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 12–22, 2023. 6
[46] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao,
Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large recon-
struction model for 3d gaussian splatting. European Confer-
ence on Computer Vision, 2024. 3, 4
[47] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. In CVPR, 2018. 6
[48] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo
Poggi. Go-slam: Global optimization for consistent 3d in-
stant reconstruction. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision (ICCV), 2023. 5,
6, 13
[49] Siting Zhu, Guangming Wang, Hermann Blum, Jiuming Liu,
Liang Song, Marc Pollefeys, and Hesheng Wang.
Sni-
slam: Semantic neural implicit slam.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 21167–21177, 2024. 6
[50] Siting Zhu, Renjie Qin, Guangming Wang, Jiuming Liu, and
Hesheng Wang. Semgauss-slam: Dense semantic gaussian
splatting slam, 2025. 3
[51] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2022. 6
10

<!-- page 11 -->
[52] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Polle-
feys. Nice-slam: Neural implicit scalable encoding for slam.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 12786–12796, 2022.
1, 2
[53] Chen Ziwen, Kaushik Patnaik, Shuangfei Zhai, Alvin Wan,
Zhile Ren, Alex Schwing, Alex Colburn, and Li Fuxin. Aut-
ofocusformer: Image segmentation off the grid. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2023. 3, 6
[54] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yi-
cong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-
sequence large reconstruction model for wide-coverage
gaussian splats. arXiv preprint 2410.12781, 2024. 4, 6
11

<!-- page 12 -->
Appendix
A. More Experimental Setup
For ScanNet, we evaluate on six scenes—scene0000,
scene0059,
scene0106,
scene0169,
scene0181,
and
scene0207—which
are
commonly
used
by
exist-
ing SLAM methods.
For ScanNet++,
we use six
scenes in total:
8b5caf3398, 3f15a9266d, e7af285f7d,
99fa5c25e1,
09c1414f1b,
and
9071e139d9.
For
NYUv2,
we
use
the
sequences
bedroom 0051 out,
dining room 0031 out,
and
student lounge 0001 out.
For
TUM
RGB-D,
we
evaluate
on
rgbd dataset freiburg1 desk,
rgbd dataset freiburg2 xyz,
and rgbd dataset freiburg3 long office household.
B. Ablation on semantic head
We present the semantic head ablation in Tab. 9. The ver-
sions with and without semantic prediction exhibit simi-
lar rendering performance, suggesting that semantics do
not noticeably change reconstruction quality. This trend is
consistent with the behavior of SGS-SLAM and SplaTAM.
SGS-SLAM is built on top of SplaTAM by adding seman-
tic colors to Gaussians and introducing a semantic loss for
tracking and mapping. However, on the ScanNet, Scan-
Net++ and TUM RGB-D datasets, incorporating seman-
tics actually degrades performance, with SGS-SLAM per-
forming worse than SplaTAM. Taken together, these results
indicate that incorporating semantic information does not
inherently affect SLAM tracking or reconstruction quality.
Nevertheless, we argue that having a unified backbone that
jointly supports both SLAM and semantics remains valu-
able, as it allows a single model to deliver both accurate
geometry and rich semantic understanding of the scene.
Table 9. Ablation of semantic prediction head on ScanNet (av-
eraged over test scenes). Results are reported using 5 iterations
of the few-iteration joint Gaussian–pose optimization.
Semantic Head
PSNR↑
SSIM↑
LPIPS↓
GS Num↓
ATE RMSE↓
False
24.53
0.885
0.292
254k
6.96
True
24.55
0.885
0.299
224k
6.95
C. More Results on GS4 with Different Predic-
tion Stages
In the main paper, we use the second stage of the backbone
as the default prediction stage. Here, we additionally report
results using the third and fourth stages of the backbone as
the prediction stage in Table 10 and Table 11. Using the
fourth stage substantially reduces the number of Gaussians
in the scene—by roughly 10× fewer—while only slightly
degrading the rendering quality and the semantic perfor-
mance. As shown in Table 10, the fourth stage achieves
PSNR that is comparable to the second stage, despite using
only one tenth of the Gaussians, highlighting a favorable
trade-off between reconstruction efficiency and fidelity. We
also observe that, under the same prediction stage, using a
lower input resolution leads to only a small drop in render-
ing performance, but quite significantly degraded seman-
tic performance. This is because our model is ultimately
trained at a higher resolution of 640×480, and downscaling
the input reduces the amount of fine-grained semantic detail
available per frame. Consequently, the per-frame semantic
predictions become less accurate, which accumulates over
the sequence and results in a noticeable drop in semantic
performance at the scene level.
D. Comparison with SLAM Methods Using
Offline Post-Optimization Refinement
We omitted Splat-SLAM [33], MonoGS [24] and OVO-
Gaussian-SLAM [23] from the main paper because these
methods perform an additional 26k optimization iterations
on the full map on the complete video after SLAM has
finished. This heavy post-optimization step substantially
boosts their final reconstruction quality (around +4 dB
PSNR), but is against the mentality of SLAM for generating
online and incremental outputs. It is not directly compara-
ble to our setting, where we report results of online SLAM
without such long global refinement (roughly 5 minutes of
extra optimization per scene). To ensure a fair compari-
son, we therefore disable these final refinement iterations
on the full map and evaluate the baselines under a setting
that matches ours more closely.
In Table 10, we report a comparison between our method
and Splat-SLAM on ScanNet at two different resolutions,
640 × 480 and 320 × 240. The higher resolution is in-
cluded because all other baselines report rendering metrics
at 640 × 480, and this is also the resolution used in our
main experiments. However, Splat-SLAM fails under this
setting on ScanNet, making a direct comparison at full res-
olution impossible. To still provide a fair and informative
Table 10. Rendering Performance on ScanNet. Values are aver-
aged across the test scenes. Results are reported using 5 iterations
of the few-iteration joint Gaussian–pose optimization. GS Num
represents the number of 3D Gaussians included in the scene after
mapping is complete.
Res
Method
PSNR↑
SSIM↑
LPIPS↓
GS Num↓
mIOU(2D)↑
ATE ↓
Splat-SLAM
%
%
%
%
%
%
640 × 480
GS4 (Ours, 2nd stg)
24.55
0.885
0.299
224k
62.10
6.95
GS4 (Ours, 3rd stg)
24.56
0.875
0.344
69k
60.34
6.94
GS4 (Ours, 4th stg)
24.21
0.859
0.397
23k
60.12
6.97
Splat-SLAM
20.67
0.684
0.438
87k
%
7.60
320 × 240
GS4 (Ours, 2nd stg)
22.43
0.921
0.196
124k
52.85
6.94
GS4 (Ours, 3rd stg)
24.24
0.911
0.245
37k
47.55
6.93
GS4 (Ours, 4th stg)
23.65
0.891
0.323
12k
48.71
6.95
12

<!-- page 13 -->
Table 11. Rendering Performance on NYUv2 and TUM-RGBD
using 640 × 480 resolution. Values are averaged across the test
scenes. Results are reported using 5 iterations of the few-iteration
joint Gaussian–pose optimization. GS Num represents the number
of 3D Gaussians included in the scene after mapping is complete.
Dataset
Method
PSNR↑
SSIM↑
LPIPS↓
GS Num↓
ATE ↓
MonoGS
12.88
0.505
0.550
110k
- -
OVO-Gaussian-SLAM
12.88
0.505
0.550
110k
- -
Splat-SLAM
23.30
0.754
0.332
85k
–
NYUv2
GS4 (Ours, 2nd stg)
22.24
0.866
0.254
298k
- -
GS4 (Ours, 3rd stg)
21.86
0.826
0.398
72k
–
GS4 (Ours, 4th stg)
20.94
0.766
0.530
18k
- -
MonoGS
17.78
0.718
0.315
37k
1.52
OVO-Gaussian-SLAM
17.78
0.718
0.315
37k
1.52
Splat-SLAM
22.84
0.780
0.287
62k
1.1
TUM
GS4 (Ours, 2nd stg)
22.70
0.903
0.191
166k
1.41
GS4 (Ours, 3rd stg)
22.32
0.876
0.273
41k
1.42
GS4 (Ours, 4th stg)
21.23
0.833
0.392
12k
1.42
comparison, we additionally evaluate at the lower resolu-
tion of 320 × 240, which is the default configuration for
Splat-SLAM on ScanNet. This allows us to compare per-
formance at the resolution where Splat-SLAM is stable,
while simultaneously highlighting that it does not robustly
handle the higher-resolution setting used by other methods.
At 320 × 240, our method consistently outperforms Splat-
SLAM when using either the second or the fourth backbone
stage for prediction.
With the second stage, we surpass
Splat-SLAM by 1.76 dB in PSNR (8.5%), 0.237 in SSIM
(34.6%), and 0.242 in LPIPS (55.3%). Even when using the
fourth stage, we still outperform Splat-SLAM while using
only 13.8% of its Gaussians, demonstrating a significantly
better trade-off between quality and efficiency.
In Table 11, we report a comparison between our method
and MonoGS, OVO-Gaussian-SLAM, and Splat-SLAM on
the NYUv2 and TUM RGB-D datasets.
MonoGS and
OVO-Gaussian-SLAM share exactly the same reconstruc-
tion metrics because OVO-Gaussian-SLAM is built on top
of the MonoGS system and simply adds semantic labels
using CLIP features to predict a label for each Gaus-
sian; both methods share the same mapping and track-
ing pipeline for RGB-D reconstruction.
It is clear that,
without the long global refinement stage, our method sur-
passes MonoGS, OVO-Gaussian-SLAM, and Splat-SLAM
on most metrics on both NYUv2 and TUM RGB-D. In con-
clusion, our method outperforms these baselines that rely on
post-optimization refinement on most metrics, while using
substantially fewer Gaussians.
E. Do 3DGS-Based Methods Improve with
a Stronger Tracking Module from GO-
SLAM?
We adopt off-the-shelf tracking algorithms [28, 48] in
our system. Specifically, we use the GO-SLAM tracking
module on ScanNet, NYUv2, and TUM RGB-D, and the
Mast3r-SLAM tracking module on ScanNet++. To clarify
whether other 3DGS-based methods could also benefit from
improved tracking when equipped with a similar module,
and to enable a more fair comparison with existing 3DGS-
based SLAM systems, we compare against Splat-SLAM
and MonoGS. Splat-SLAM adopts the MonoGS mapping
pipeline but integrates it into an enhanced DROID-SLAM
tracking framework, which is similar to our use of GO-
SLAM-style tracking (which itself is derived from DROID-
SLAM) on ScanNet, NYUv2, and TUM RGB-D.
We report the results of Splat-SLAM, MonoGS, and
our method on these three datasets in Table 10 and Ta-
ble 11. As expected, with a stronger tracking module, Splat-
SLAM achieves noticeably better rendering performance
than MonoGS, confirming that improved tracking alone
can boost reconstruction quality for 3DGS-based systems.
However, even under this stronger tracking setting, our
method still substantially outperforms Splat-SLAM on most
metrics, indicating that our backbone design and Gaussian
prediction strategy provide additional gains beyond what
can be achieved by improved tracking alone.
To further examine whether stronger tracking could
also benefit semantic 3DGS-based SLAM, we attempted
to inject GO-SLAM poses into GS3LAM by using GO-
SLAM’s local pose estimates to initialize GS3LAM’s track-
ing.
In practice, this integration turned out to be unsta-
ble: GS3LAM’s internal tracking and mapping pipeline is
tightly coupled to its own pose-update dynamics, and the
externally initialized poses conflicted with its optimization
trajectory. This mismatch caused the estimated trajectory to
drift abruptly, after which the system kept spawning large
numbers of Gaussians to compensate for the growing mis-
alignment. As a result, memory usage rapidly escalated and
the run eventually terminated with an out-of-memory fail-
ure.
F. Reconstruction Results on ScanNet
In Table 13, we use the metrics including Accuracy (Acc.),
Completion (Comp.), Completion Ratio[<7cm] and F-
Score[<7cm] to evaluate the scene geometry on ScanNet.
The definitions of the evaluation metrics are detailed in Ta-
ble 12. GS4 outperforms all baselines in terms of comple-
tion and F-score.
Table 12. Metric definitions. p and p∗are the reconstructed and
ground truth point clouds
3D Metric
Formula
Acc
meanp∈P

min
p∗∈P ∗∥p −p∗∥

Comp
meanp∗∈P ∗

min
p∈P ∥p −p∗∥

Completion Ratio[<7cm]
meanp∗∈P ∗

min
p∈P ∥p −p∗∥< 0.07

F-score[<7cm]
2 × meanp∈P [minp∗∈P ∗∥p −p∗∥< 0.07] × meanp∗∈P ∗

min
p∈P ∥p −p∗∥< 0.07

meanp∈P [minp∗∈P ∗∥p −p∗∥< 0.07] + meanp∗∈P ∗

min
p∈P ∥p −p∗∥< 0.07

13

<!-- page 14 -->
Table 13. Reconstruction metrics on ScanNet
Methods
Acc. ↓
Comp. ↓
Comp. Ratio (<7cm) ↑
F-Score (<7cm) ↑
GS Num↓
SplaTAM
8.10
5.58
76.34
75.95
2466k
RTG-SLAM
99.80
47.44
24.61
16.69
1229k
SGS-SLAM
17.11
13.75
55.01
55.26
2114k
GS3LAM
11.83
4.62
82.82
71.92
2154k
GS4
8.48
3.87
89.09
77.44
224k
G. Qualitative Comparison for Semantic Seg-
mentation.
As illustrated in Fig. 5, our approach achieves superior se-
mantic segmentation accuracy compared to the SGS-SLAM
and GS3LAM baselines. For example, in the first column of
Fig.5, our semantic rendering provides a more accurate rep-
resentation of the desks, chairs, and night tables than base-
lines.
SGS SLAM
GS3LAM
GS4 (Ours)
GT (labels)
GT (RGB)
scene0000
scene0169
scene0207
Figure 5. Semantic Renderings on ScanNet. Qualitative com-
parison on semantic synthesis of our method and baseline seman-
tic SLAM method SGS-SLAM. Black areas in GT labels denote
regions that are unannotated.
H. Zero-Shot Results on NYUv2
Fig. 6 illustrates our zero-shot visualization results on the
NYUv2 dataset.
Despite our models being exclusively
trained on the ScanNet dataset, our method demonstrates
superior performance on the NYUv2 dataset compared to
other GS-based SLAM approaches. In Table 14, we present
the quantitative zero-shot results across three scenes from
the NYUv2 dataset. Our method outperforms all other GS-
based SLAM approaches on all rendering metrics, while us-
ing significantly fewer Gaussians.
Table 14. Rendering and Runtime performance on NYUv2 test
scenes with 640 × 480 input. GS Num represents the number of
3D Gaussians included in the scene after mapping is complete.
FPS is conducted on an Nvidia RTX TITAN.
Methods
Metrics
bedroom
student lounge
dining room
Avg
PSNR↑
17.99
20.77
17.82
18.86
SSIM↑
0.692
0.795
0.589
0.692
SplaTAM
LPIPS↓
0.343
0.309
0.465
0.372
GS Num↓
1529k
1116k
1063k
1236k
PSNR↑
10.81
12.94
11.76
11.84
SSIM↑
0.146
0.299
0.217
0.221
RTG-SLAM
LPIPS↓
0.738
0.662
0.709
0.703
GS Num↓
906k
591k
925k
807k
PSNR↑
19.66
20.41
17.90
19.32
SSIM↑
0.754
0.780
0.590
0.708
SGS-SLAM
LPIPS↓
0.289
0.318
0.463
0.357
GS Num↓
1201k
1074k
1049k
1108k
PSNR↑
21.15
22.35
23.21
22.24
SSIM↑
0.885
0.867
0.846
0.866
GS4 (Ours, 2nd stg)
LPIPS↓
0.217
0.238
0.307
0.254
GS Num↓
273k
228k
392k
298k
I. Zero-Shot Results on TUM RGB-D
In Table 15, we present the quantitative zero-shot results
across three scenes from the TUM RGB-D dataset. TUM
RGB-D provides ground truth camera trajectories, so we
also report tracking performance.
Our method achieves
rendering performance comparable to that of all other GS-
based SLAM approaches, while using significantly fewer
Gaussians.
Table 15.
Rendering, Tracking, and Runtime performance on
TUM RGB-D test scenes with 640 × 480 input. GS Num rep-
resents the number of 3D Gaussians included in the scene after
mapping is complete. FPS is conducted on an Nvidia RTX TI-
TAN.
Methods
Metrics
fr1 desk
fr2 xyz
fr3 office
Avg
PSNR↑
22.07
24.66
21.54
22.76
SSIM↑
0.857
0.947
0.870
0.891
SplaTAM
LPIPS↓
0.238
0.099
0.210
0.182
ATE RMSE↓
3.33
1.55
5.28
3.39
GS Num↓
969k
635k
806k
803k
PSNR↑
18.49
20.18
20.59
19.75
SSIM↑
0.715
0.795
0.797
0.769
RTG-SLAM
LPIPS↓
0.438
0.353
0.394
0.395
ATE RMSE↓
1.66
0.38
1.13
1.06
GS Num↓
236k
84k
273k
198k
PSNR↑
22.10
25.61
19.62
22.44
SSIM↑
0.886
0.946
0.796
0.876
SGS-SLAM
LPIPS↓
0.176
0.097
0.280
0.184
ATE RMSE↓
3.57
1.29
9.08
4.65
GS Num↓
808k
695k
701k
735k
PSNR↑
21.71
23.86
22.54
22.70
SSIM↑
0.877
0.904
0.890
0.903
GS4 (Ours, 2nd stg)
LPIPS↓
0.242
0.154
0.226
0.191
ATE RMSE↓
1.86
0.63
1.95
1.48
GS Num↓
175k
87k
190k
166k
14

<!-- page 15 -->
SplaTAM
SGS SLAM
GS4 (Ours)
GT
sudent lounge
dining room
bedroom
Figure 6. Zero-shot Visualization on NYUv2. Qualitative comparison of our method and other GS-based SLAM methods.
15
