<!-- page 1 -->
Snap-Snap: Taking Two Images to Reconstruct
3D Human Gaussians in Milliseconds
Jia Lu1*, Taoran Yi1*, Jiemin Fang2†, Chen Yang3, Chuiyun Wu1, Wei Shen3, Wenyu Liu1,
Qi Tian2, Xinggang Wang1†
1Huazhong University of Science and Technology
2Huawei Inc.
3Shanghai Jiaotong University
{jialu2023, taoranyi, wuchuiyun, liuwy, xgwang}@hust.edu.cn
jaminfong@gmail.com
{ycyangchen, wei.shen}@sjtu.edu.cn
tian.qi1@huawei.com
Abstract
Reconstructing 3D human bodies from sparse views has
been an appealing topic, which is crucial to broader the
related applications. In this paper, we propose a quite chal-
lenging but valuable task to reconstruct the human body
from only two images, i.e., the front and back view, which
can largely lower the barrier for users to create their own
3D digital humans. The main challenges lie in the diffi-
culty of building 3D consistency and recovering missing
information from the highly sparse input. We redesign a
geometry reconstruction model based on foundation recon-
struction models to predict consistent point clouds even in-
put images have scarce overlaps with extensive human data
training. Furthermore, an enhancement algorithm is ap-
plied to supplement the missing color information, and then
the complete human point clouds with colors can be ob-
tained, which are directly transformed into 3D Gaussians
for better rendering quality.
Experiments show that our
method can reconstruct the entire human in 190 ms on a
single NVIDIA RTX 4090, with two images at a resolution
of 1024×1024, demonstrating state-of-the-art performance
on the THuman2.0 and cross-domain datasets. Addition-
ally, our method can complete human reconstruction even
with images captured by low-cost mobile devices, reducing
the requirements for data collection. Demos and code are
available at https://hustvl.github.io/Snap-
Snap/.
1. Introduction
Human reconstruction has always been an important topic
in the 3D field, which can be considered a crucial bridge
between the real and digital worlds. It has broad practi-
*Equal contribution (during internship at Huawei Inc.)
†Corresponding authors
single view
dense views
two views
Snap-Snap
···
misaligned 
human pose
uncontrollable 
textures
expensive 
data collection
SMPL(X) input ❌
fast inference ✅
easy data capture ✅
Figure 1.
Under the setting of two input images, we propose
a feed-forward framework named Snap-Snap which can directly
predict 3D human Gaussians in milliseconds.
cal application prospects, including virtual/augmented real-
ity, games, and Metaverse. Professional/expensive facilities
to capture human data are usually required in previous re-
construction methods [20, 24, 33, 53, 59], e.g., using syn-
chronized cameras to capture the target human from many
views. Loosening the capturing requirements allows more
users to complete their own reconstruction and makes it ap-
plicable to more scenarios.
Reconstructing human bodies from images has always
been a challenging topic and has been studied in many
previous works [3, 4, 18–20, 40, 52]. In single-view hu-
man reconstruction [8, 51, 58], human prior estimations
(SMPL-X [32]) are frequently misaligned with real-world
coordinates (e.g., body inclination), and generative meth-
ods [35, 41] introduced to infer occluded regions often pro-
arXiv:2508.14892v1  [cs.GR]  20 Aug 2025

<!-- page 2 -->
duce results with limited controllability. Meanwhile, hu-
man prior estimations based on limited viewpoints still suf-
fer from inaccuracies in details, such as the hands.
Al-
though human reconstruction methods based on dense input
views [59] achieve high-quality results, they often require
expensive data acquisition setups, making them less acces-
sible to users. Fig. 1 provides a visual comparison of human
reconstruction methods under different settings. Under the
premise of preserving a controllable and consistent human
appearance, we propose to explore an extremely challeng-
ing task - reconstructing human bodies from just two im-
ages, i.e., the front and back view, in milliseconds. This task
makes it quite easy for any user to reconstruct the desired
3D digital human, without needing professional knowledge
to capture redundant images or a long wait to get the result.
One main challenge lies in the lack of overlap between the
front and back views, making it hard to establish geometry
consistency for reconstruction. Additionally, the informa-
tion from the two views is too limited to cover all the details
of the human body.
To tackle the above challenges, we propose to construct
a fast point cloud prediction model which can reconstruct
complete human point clouds from four viewpoints (front,
back, left and right views) even with two input images. With
training upon the human datasets, the reconstruction model
based on geometry reconstruction model [44] adapts the
generalizable geometric prior to the human domain. As the
geometry reconstruction model only predicts point clouds
without color information, we further design an enhance-
ment algorithm to enhance the side-view color by wrapping.
With the above processes, a complete set of colored point
clouds for the target human is obtained. To achieve better
rendering quality, we transform these points into 3D Gaus-
sians [15] by directly inferring corresponding Gaussian at-
tributes. Overall, the framework is efficient enough to com-
plete one human reconstruction at the millisecond level. All
you need to do is take two human images from the front and
back views, so we name our method Snap-Snap to represent
the shutter sound when capturing the human. Our approach
demonstrates superior reconstruction performance on sev-
eral datasets.
Our contributions can be summarized as follows:
• We design a feed-forward reconstruction framework,
which can directly predict 3D human Gaussians from just
two images in milliseconds without human prior.
• We redesign a geometry reconstruction model which can
build human point clouds even with highly sparse input,
adapting the generalizable geometric prior to the human
domain. In addition, a side view enhancement algorithm
is proposed to supplement the unseen information.
• With two-view images at a resolution of 1024×1024, our
method can obtain complete human reconstruction re-
sults in 190 ms and demonstrates state-of-the-art perfor-
mance on the THuman2.0 [55] and cross-domain [6, 45]
datasets. The method is also shown to perform well on
data acquired from low-cost mobile devices.
2. Related Works
3D Gaussian Splatting in Human. 3D Gaussian Splatting
[15] explicitly reconstructs the scene with Gaussian points,
achieving good reconstruction results in an efficient way.
Many methods have introduced 3DGS into the field of hu-
man reconstruction, achieving state-of-the-art reconstruc-
tion results, and even high-quality animated results. Most of
the work uses SMPL [27] or SMPL-X [32] as human prior
for point clouds initialization and accepts videos as input for
scene reconstruction. Most methods [10, 11, 13, 16, 21, 23]
accept monocular video as input for the human reconstruc-
tion, while a few methods based on multi-view video recon-
struction [14, 25, 30] have expanded in other aspects. How-
ever, most of these works lack generalizability, requesting
to be trained separately for each scene.
In contrast, we
propose a feed-forward generalizable human reconstruction
method, achieving good reconstruction quality even under
extremely sparse viewpoints in milliseconds.
Generalizable Human Reconstruction.
Generalizable
Human Reconstruction aims to reconstruct the 3D human
body solely through inference from input images. Many
works have achieved generalizable reconstruction of the 3D
human body under the framework of implicit representa-
tion, such as those based on pixel-aligned features [38, 39],
those based on generalizable neural voxels [53] and those
based on sparse 3D keypoints [29]. Meanwhile, the se-
ries of works [12, 49, 50] have generalized the extraction
of human geometry solely from a single input image. Hu-
manSplat [31] infers explicit human representation from a
single image with semantic cues. Single-view human re-
construction often involves generative models, making it
difficult for the reconstructed results to fully match the tar-
get subject. Under the sparse camera setting, NHP [18]
learns generalizable neural radiance representations with
body motion prior. GHG [20] learns generalizable human
Gaussians on the 2D UV space of a human template. GPS-
Gaussian [59] builds upon a depth estimation model, lever-
aging neural networks to extract Gaussian properties, and
demonstrates strong results in novel view synthesis tasks.
Points Prediction. With the development of neural net-
works [5], image-based scene perception capabilities have
also advanced, where point clouds are extracted solely
based on RGB images. Several methods [2, 34, 42] achieve
scene perception based on monocular depth estimation.
Furthermore, [54] utilizes point cloud neural networks to
improve the estimation of point clouds. Disparity is another
representation of point cloud information in stereo vision.
[26] obtains point cloud information by converting the dis-
parity map to a depth map. Recently, with the progress

<!-- page 3 -->
in general 3D models [47], methods for directly predicting
point clouds [22, 44] have emerged and exhibited excellent
perception capabilities.
3. Method
3.1. Preliminary
3D Gaussian Splatting
3D Gaussians [15] exhibit high
quality during rendering while enjoying the speed of real-
time rendering. 3D Gaussians represent space as ellipses of
different sizes and orientations. Specifically, the 3D Gaus-
sian is defined with its center position µ ∈R3, color c ∈R3,
opacity o ∈R1, and covariance Σ. For optimization pur-
poses, the covariance is decomposed into the scale s ∈R3
and a quaternion q ∈R4 representing rotation. Therefore,
3D Gaussians can be represented as:
  \la be l { e q: def3dgs} \theta =(\mu , c, o, \Sigma =(s, q)) 
(1)
During the projection of 3D Gaussians onto the 2D pixel
plane, 3D Gaussians accumulate along the corresponding
rays. The entire rendering process is differentiable, which
lays the foundation for optimization.
3.2. Overview
Given only the front and back RGB images of human body,
we aim to obtain high-quality human Gaussian in a feed-
forward manner without camera parameters. Our general-
izable human reconstruction algorithm can be divided into
three stages as follows, as shown in Fig. 2:
Point Cloud Prediction. We redesign a human point
cloud prediction model Rp to reconstruct complete human
point clouds from four viewpoints (front, back, left and
right viewpoints). We additionally introduce two heads to
predict point clouds from side views under the condition
of missing side geometry information, while maintaining
alignment with the real-world coordinate system. By train-
ing Rp on human datasets, we adapt the geometric recon-
struction prior to better fit the human domain. Finally, we
concatenate the point clouds from all four views to form the
complete human point clouds.
Side-view Enhancement. Since the point cloud predic-
tion model only predicts geometry for side views without
color information, we construct an enhancement module to
improve the left and right side views Il, Ir with the the front
and back images If, Ib in absence of camera parameters,
thereby enhancing the completeness of the final representa-
tion.
Gaussian Attribute Regression. Similar to point clouds
prediction, we regress Gaussian attributes from four view-
points directly. We input the point clouds of the four per-
spectives, the front and back images of the input, and the
enhanced pseudo-color information into the Gaussian Re-
gression network to obtain the final complete human Gaus-
sian.
3.3. Point Cloud Prediction
We design the geometric reconstruction model Rp to per-
form point cloud prediction from four viewpoints: front,
back, and two sides, ensuring the completeness of the pre-
dicted human point clouds. Specifically, we take the front
and back images If, Ib ∈RW ×W ×3 as input, and pro-
cess them through an encoder Ep and a decoder Dp with
B blocks to obtain intermediate image representations Gf
and Gb.
To fully leverage the priors from the foundation geomet-
ric reconstruction model, we adopt a similar architecture to
DUSt3R [44]. In order to generate point clouds from the
front and back views, two prediction heads Hf, Hb are used
to process Gf and Gb respectively. Benefiting from the in-
formation exchange between front and back tokens within
the decoder, we simply aggregate Gf and Gb to form the
input tokens Gv for the side-view heads Hl and Hr. By
training the foundation geometric reconstruction model Rp
on human data, the model learns to infer plausible geome-
try for two sides, even in the absence of explicit side-view
observations.
  \be
gin  
{
spl i t} \{
G^
v\} _
{ i=1 } ^B
 
= \le
ft 
\
{ ( G^f  + 
G
^b)/2
 \r
i
ght \}_{i=1}^B, \\ P^{l,f} = \mathcal {H}_l\left (\{G^v\}_{i=1}^B\right ), P^{r,f} = \mathcal {H}_r\left (\{G^v\}_{i=1}^B\right ), \end {split} \label {eq:points2} 
(2)
We concatenate the point clouds of the front, back, left,
and right in the perspective of the front viewpoint to obtain
the final complete point clouds of the human body. Further-
more, to align the predicted point cloud with the human in
the real world, we introduce a learnable parameter to esti-
mate the actual human scale, thus obtaining a scaling factor
δ and generate the final human point cloud with the correct
proportions in the real-world coordinate.
  \ b e gi n { s p lit }  P^{ h }  = \delta * (P^{f,f} \oplus P^{b,f}\oplus P^{l,f}\oplus P^{r,f}), \end {split} 
(3)
Based on the priors of the foundation geometry recon-
struction model and training on human datasets, we ob-
tain the complete human point clouds from four views pre-
dicted in the perspective of the front viewpoint, even with
almost no overlapping input. Rp implicitly learns the map-
ping relationships between viewpoints, directly learning the
positional relationships of point clouds between different
viewpoints, enabling the acquisition of the complete human
point clouds even without camera parameters.
3.4. Side-view Enhancement
Since we only use the images from the front and back views,
the predicted side point clouds P l,f, P r,f do not contain any
color information, leading to a severe lack of color informa-
tion on the side views of the human body. To address this,

<!-- page 4 -->
Input Images
Point Clouds 𝑃!
Gaussian Attribute Regression
𝓗𝒇
𝓗𝒃
𝓗𝒍
𝓗𝒓
Concatenate
Point Cloud Prediction with Geometry Reconstruction Prior
Side-view Enhancement
𝓔𝒑𝓓𝒑
Back View
Front View
Side Images
Additional Head
Original Head
Encoder 
&
Decoder
Input Images
Side Images
Human Gaussian 
Splatting
𝑃%,%
𝑃',%
𝑃(,%
𝑃),%
𝑃%,%
𝑃',%
𝑃(,%
𝑃),%
𝓕𝓰
Pseudo View
Project
Point Clouds 𝑃!
Pseudo View
Project
Point Clouds 𝑃!
NNS
𝐼"
𝐼#
𝐼$
𝐼%
Point Clouds 𝑃!
𝐼#
𝐼"
𝐼%
𝐼$
Gaussian attribute 
regression
Figure 2. The framework of Snap-Snap. With the input front and back view images If and Ib, the point cloud prediction model Rp
generate the human point clouds from the front P f,f, back P b,f, left P l,f, and right P r,f views. Side-view color information is supplied
by the side-view enhancement module. With the enhanced images Il, Ir and the input images If, Ib, we obtain fianl human Gaussians
through Gaussian attribute regression F g.
Input Images
ℋ!
ℋ"
ℰ#
𝐺!
Average
𝐺"
𝐺$
ℋ%
ℋ&
𝒟#'
𝒟#(
: information sharing
𝛿
Figure 3. The framework of point cloud prediction network.
we propose a simple nearest neighbor search (NNS) algo-
rithm to warp the color information from the front and back
views to the side views.
In Sec. 3.3, the human front and back point clouds
P f,f, P b,f along with side point clouds P l,f, P r,f are
obtained, but color information is missing.
Since the
point clouds from the geometric reconstruction model Rp
is pixel-wise, we can easily obtain the color informa-
tion Cf,f, Cb,f of front and back point clouds P f,f, P b,f.
Specifically, we project the front and back image pixels to
their corresponding point clouds.
Due to the missing color information on the sides, the fi-
nal reconstruction results are adversely affected. The NNS
algorithm is further used to transfer colors in front and back
point clouds to side point clouds, which are then unpro-
jected to side pseudo-views to obtain the side color informa-
tion. We use NNS to find the nearest neighbors of the side
point clouds P l,f, P r,f in the known colored point clouds
P f,f, P b,f, and then assign their color information to the
side point clouds, thereby obtaining the color information
Cl,f and Cr,f for the side point clouds. Taking the left view
for example. Assume P l,f = {pl,f
1 , pl,f
2 , . . . , pl,f
n }, Cl,f =
{cl,f
1 , cl,f
2 , . . . , cl,f
n }, this process can be formulated as
  \ beg in  {ali g
n e d} \lab el { e q: co m po
i nt
s} \
f
o rall i \in \ { 1,2, ...n \}: \\ &j = F_\text {nns}(\{P^{f,f}, P^{b,f}\}; p^{l,f}_i),\\ & c^{l,f}_i \leftarrow \text {Index}( \{C^{f,f}, C^{b,f}\};j), \end {aligned} 
(4)
where Fnns denotes the process of searching the nearest
neighbor index j in the point cloud set {P f,f, P b,f} for
each element pl,f
i
∈P l,f. Index denotes an indexing pro-
cess which looks up the color set {Cf,f, Cb,f} using index
j and then assigns the fetched value to cl,f
i . Similarly, we
can obtain the color information for P r,f. Since P l,f and
P r,f are also predicted in a pixel-wise manner, we can eas-
ily establish correspondence between the point cloud colors
Cl,f, Cr,f and the corresponding side pseudo-view pixels

<!-- page 5 -->
Il, Ir without known camera parameters.
3.5. Gaussian Attribute Regression
In a feed-forward manner, we predict 3D Gaussians θ =
(µ, c, o, Σ = (s, q)) based on the obtained human point
clouds. The point cloud prior predicted by Sec. 3.3 rep-
resents points upon the human body surface. Considering
the differences between point clouds and 3D Gaussians,
the prediction of absolute 3D coordinates µ is reformu-
lated as predicting the offset ∆µ relative to the point cloud
prior. In addition, we need to predict the scale, opacity,
color, and rotation of the Gaussians. To reconstruct the en-
tire human body, we regress the Gaussian attributes from
the input front and back viewpoints and the left and right
pseudo-viewpoints. With the point clouds predicted from
the four viewpoints P f,f, P b,f, P l,f, P r,f ∈RW ×W ×3 and
the color maps If, Ib, Il, Ir ∈RW ×W ×3, we use a UNet-
like [36] network Fg to predict the Gaussian attributes
θ = {∆µ, c, o, s, q}:
  \ be g in {a lign ed} \t heta  _f, 
\th et a  _b & = \m athc al  {F} ^{g} 
(  \{ P ^{ f ,f } , I^f\}, \{P^{b,f}, I^b\}), \\ \theta _l, \theta _r &= \mathcal {F}^{g} (\{P^{l,f}, I^l\}, \{P^{r,f}, I^r\} ), \\ \theta &= \theta _f \oplus \theta _b \oplus \theta _l \oplus \theta _r , \label {eq: gs_re} \end {aligned} 
(5)
where θf, θb, θl, θr denote the Gaussians predicted from
front, back, left, and right viewpoints, respectively. By con-
catenating them together, we obtain the final Gaussian rep-
resentation θ for the entire human body.
3.6. Training and Inference
The entire framework’s learnable components can be di-
vided into two modules: (1) The point cloud prediction net-
work, and (2) The Gaussian regression network. To ensure
the performance of the algorithm, we train the two modules
separately.
Training Stage 1:
For the point cloud prediction network,
we use 3D point clouds P gt and 2D image mask M gt as
supervision. The L2 loss and cross-entropy loss are denoted
as the regression loss Lreg and the confidence loss Lconf.
  \math c al {
L
} _{ s tag
e 1} = {L }_{re g }\left (P^{h}, P^{gt}\right ) + {L}_{conf}(M^{conf},M^{gt}). 
(6)
Training Stage 2:
For the Gaussian regression network,
we render image Irender from novel views with differen-
tiable splatting, and use the ground-truth image Igt as train-
ing supervision. L1 loss and SSIM loss, denoted as Lrgb
and Lssim, are employed to train the Gaussian regression.
  \labe l  {eq:
 
stage 2 loss
}  \ r esizebo
x
 {.9\hsi ze }
{!}{$ \mathcal {L}_{stage2} = \beta {L}_{rgb}\left (I^{render}, I^{gt}\right ) + (1-\beta ) \mathcal {L}_{ssim}\left (I^{render}, I^{gt}\right ). $} (7)
Inference:
During inference, we first use the input front
and back view images If, Ib to obtain the point clouds of
the human body P f,f, P b,f, P l,f, P t,f through the point
cloud prediction model Rp. Then, we obtain the enhanced
images Il, Ir through the side-view enhancement algo-
rithm. Based on the enhanced images Il, Ir and the in-
put images If, Ib and the point clouds of the human body
P f,f, P b,f, P l,f, P r,f, the human Gaussian are obtained
through Gaussian attribute regression Fg.
4. Experiments
4.1. Implementation Details
We train our model on a single RTX 4090 with 24G mem-
ory and set the batch size as 1. For the training of each
module, we apply the AdamW [28] optimizer with a learn-
ing rate of 1 × 10−4, and a weight decay of 5 × 10−2. The
learning rate is cosine annealed to 1×10−7 during the train-
ing. The training iterations for stage 1 are 100k, while the
training iterations for stages 2 are only 50k. The training
times for the two stages are approximately 13 and 6 hours
respectively. Besides, the β in Eq. 7 for the stage 2 is 0.8.
Datasets.
We train and evaluate on the Thuman2.0
dataset [55] while evaluating on the 2K2K [6] and 4D-
Dress [45] for cross-domain evaluation. The THuman2.0
dataset consists of 526 high-quality human assets. Follow-
ing [20], we select the same 100 subjects to evaluate the
algorithm. Due to the requirement of ground-truth SMPL-
X parameters by GHG [20], we select the first 100 subjects
from the 2K2K training set for evaluation. Furthermore,
to evaluate the reconstruction quality of loose clothing, we
select 50 loose clothing examples from Thuman2.1 [55]
(exclude human scans of Thuman2.0) by calculating the
chamfer distance between the SMPL-X [32] mesh and the
ground-truth human meshes. We sort the Chamfer Distance
from largest to smallest and select the top human bodies as
Loose Clothes val set.
4.2. Comparisons
In Tab. 1, we present the comparison results with GPS-
Gaussian [59] and GHG [20] with three metrics: Peak
Signal-to-Noise Ratio (PSNR), Structure Similarity Index
Measure (SSIM) [46], and Learned Perceptual Image Patch
Similarity (LPIPS) [56] to assess the quality of the rendered
images. LPIPS is calculated with AlexNet [17] model. The
resolution of the rendered images is 1024 × 1024, and we
only use the human regions given by the bounding box of
humans when calculating the metrics following the GHG.
For better comparison, we provide visual results in Fig. 4,
where the resolution of the rendered images for all meth-
ods is 2024 × 2048. GHG under 2 views is trained by the
publicly released code, while we use the released inpainting

<!-- page 6 -->
Thuman2.0 [55]
2K2K [6]
4D-Dress [45]
Method
Views SMPL-X [32] Infer. (ms) PSNR↑SSIM ↑LPIPS ↓PSNR↑SSIM↑LPIPS ↓PSNR↑SSIM↑LPIPS ↓
GPS-Gaussian [59]
5
N/A
144
20.70
88.11
19.15
21.68
88.45
17.17
20.47
84.78
22.43
GHG [20]
3
GT
2858
21.99
88.45
13.79
21.73
87.50
13.88
21.20
86.03
17.24
GHG
2
GT
2853
21.79
87.66
17.54
21.26
86.38
17.54
20.69
85.13
20.58
GHG
2
Estimated
10696
16.71
81.92
26.22
16.72
80.48
26.53
18.34
80.99
26.55
Snap-Snap (Ours)
2
N/A
190
22.44
88.78
13.24
22.38
88.01
13.08
21.62
85.55
17.03
Table 1. We present the comparison results with the GPS-Gaussian [59] and GHG [20]. We find that GHG uses the ground-truth SMPL-
X [32] parameters during inference. For a fair comparison, we estimate the SMPL-X parameters only using two viewpoints though
EasyMocap [1]. Infer. denotes the total inference time from receiving images and masks to the final Gaussians. The detailed time-
consuming analysis is presented in the supplementary materials.
Snap-Snap
(Ours)
GT
GPS-Gaussian
Input
Input
Input 5 Views
Input 2 Views
GHG
(w/ GT SMPL-X)
Normal Clothing
Loose Clothing
Cross Domain
Figure 4. Visual comparisons with GPS-Gaussian [59] and GHG [20].
model weight. GPS-Gaussian under 5 views is completely
trained following the instructions.
The reconstruction quality of GPS-Gaussian is lower
than ours even with five viewpoints. In Fig. 4 we can see

<!-- page 7 -->
Method
Views SMPL-X [32] PSNR↑SSIM ↑LPIPS ↓
GPS-Gaussian [59]
5
N/A
19.49
84.45
22.02
GHG [20]
3
GT
20.02
84.36
17.67
GHG
2
GT
19.74
83.44
21.60
GHG
2
Estimated
15.80
78.77
29.08
Snap-Snap (Ours)
2
N/A
20.98
84.42
17.08
Table 2. The comparison on loose clothes val set.
Method
Views SMPL-X PSNR↑SSIM↑LPIPS ↓
SiTH [8]
2
GT
15.10
76.74
31.68
Snap-Snap (Ours)
2
N/A
22.44
88.78
13.24
Table 3. The comparison between Snap-Snap and mesh-based hu-
man reconstruction SiTH [8].
Side Head
NNS
PSNR ↑
SSIM ↑
LPIPS↓
×
×
22.15
88.20
14.11
✓
×
22.34
88.39
13.75
✓
✓
22.44
88.78
12.47
Table 4. We perform ablation studies on the additional side-view
heads in the point cloud prediction network and the side-view en-
hancement module.
that GPS-Gaussian’s reconstruction shows missing body
parts due to the limitations of the depth estimation module
in GPS-Gaussian, which cannot provide reasonable results
due to the sparsity of the viewpoints. Unlike GPS-Gaussian,
our method completes the side views both in terms of point
clouds and colors, which can directly infer the complete hu-
man point clouds from the front and back views.
GHG completes human body from sparse viewpoints
based on the human prior SMPL-X. We find that GHG uses
the ground-truth SMPL-X parameters during the inference,
which are calculated from far more than two viewpoints. To
fairly evaluate GHG’s reconstruction quality, we use Easy-
Mocap [1] to estimate the SMPL-X parameters from two
viewpoints. Moreover, our method enables human recon-
struction at the millisecond level shown in Tab. 1, in contrast
to GHG which requires time computing SMPL-X parame-
ters and preprocessing data.
In Tab. 2, we evaluate our method on human bodies with
loose clothing, demonstrating the robustness of our method
over different human bodies. In the visual results, we can
see that due to GHG being based on SMPL-X, it cannot re-
construct loose clothing well. Our method directly infers
complete geometric information through the point cloud
prediction model, achieving good modeling even for loose
clothing.
Input images
SiTH*
Snap-Snap (Ours)
GT
SiTH*: use the input two images and ground-truth SMPL-X parameters
Figure 5. Qualitative comparisons with SiTH [8].
front
SIFU
LGM
Ours
GT
GTA
Human3Diffusion
ECON
Figure 6. Visual comparisons of Snap-Snap with single-view re-
construction methods.
Input images
Snap-Snap
SiTH (2views)
Input images
Snap-Snap
SiTH (2views)
Figure 7. Reconstruction results from in-the-wild data.
4.3. Comparison with Single-view Reconstruction
To further evaluate the effectiveness of our method, we
compare it with the mesh-based single-view reconstruction
approaches. To minimize the influence of generative com-
ponents [8] on the human reconstruction, we select SiTH [8]
as the method for quantitative comparison. SiTH allows re-
placing the pseudo back-view with the ground-truth back-
view image, and we use the ground-truth SMPL-X parame-
ters as the input human prior.
Since mesh-based human reconstruction methods are of-
ten not aligned with the world coordinate system, we ap-

<!-- page 8 -->
+ Side Heads
Point clouds
Side-view
Human Gaussians
+ Side Heads
+ NNS
+ Side Heads
+ Side Heads
+ NNS
Figure 8. We present the visual results of ablation studies on the
additional side-view head in the point cloud prediction network
and the side-view enhancement module.
ply ICP registration [37] using the ground-truth mesh and
render the registered meshes for evaluation. We compute
PSNR on the rendered images, as shown in Tab. 3, with
more qualitative results provided in Fig. 5.
The results
demonstrate that our method achieves better reconstruction
quality and aligns more accurately with the human pose in
the input images. As shown in Fig. 5, despite using ac-
curate human prior (SMPL-X parameters), SiTH still pro-
duces distorted poses. In contrast, our method achieves high
consistency with the target human in both pose and texture.
Furthermore, we observe that due to the misalignment be-
tween mesh-based reconstruction results and the real-world
coordinate system, evaluating such methods using PSNR
can be unreliable.
Considering the alignment issue mentioned, we further
provide a qualitative comparison with existing single-view
reconstruction methods [43, 50, 51, 57, 58] in Fig. 11.
These single-image methods exhibit significantly worse re-
construction consistency than ours, even when manually
aligned. Moreover, the results of single-view reconstruc-
tion are often difficult to align with the human body scale in
the world coordinate system, whereas our reconstructions
maintain a reasonable scale due to the point clouds predic-
tion network.
4.4. Reconstruction from In-the-Wild Data
To validate our method on capture from low-cost mobile
phone, we use two mobile phones to build a capture setup
and reconstruct human bodies from the collected data. The
reconstruction results are shown in Fig. 7. Details about the
low-cost data collection device are provided in the supple-
mentary material. In the absence of camera parameters un-
der the simple capture setup, GPS-Gaussian and GHG are
unable to successfully reconstruct human. Consequently,
we provide a qualitative comparison with the two-view
SiTH. Our method achieves superior human reconstruction
in a more robust and end-to-end manner.
Training Scans
PSNR ↑
SSIM ↑
LPIPS ↓
426
22.44
88.78
13.47
2992
22.77
88.98
13.56
Table 5. Evaluation results with more datasets.
4.5. Ablation Studies
We conduct ablation experiments on the heads of the point
cloud prediction model Rp and the side-view enhancement
algorithm to verify the impact of these two modules on the
human reconstruction, as shown in Tab. 4. More ablation
studies are given in the supplementary material.
Heads of Point Cloud Prediction Model.
We perform
ablation experiments on the side heads in the point cloud
prediction model Rp, as shown in the Tab. 4. We find that
the point clouds of the human exhibit obvious gaps without
the side heads, adversely affecting the final human recon-
struction performance. To compensate for these gaps, we
use two heads to predict the side point clouds of the human
body, which greatly improves the completion of the point
clouds and the final reconstruction quality shown in Fig. 8.
Side-view Enhancement.
We conduct ablation experi-
ments on the side-view enhancement algorithm shown in
Tab. 4. Instead of using NNS to obtain side-view images,
we directly feed the front and back images into the Gaus-
sian attribute regression under the condition that camera pa-
rameters are not available. By leveraging the color warping
of NNS which based on the spatial relationships within the
point clouds, our side-enhancement algorithm achieves su-
perior reconstruction results. Fig. 8 illustrates the human
point clouds with colors after wrapping. It can be observed
that the side-view enhancement leads to more consistent
textures, particularly on the side views.
4.6. Scalability
To further explore the relationship between the current
method and the amount of training data, we expand the
dataset by adding Thuman2.1 [55] (1919 scans) and the
CustomHuman dataset [9] (647 scans) to Thuman2.0 [55]
(426 scans), resulting in a total of 2992 human scans
for training. Experimental results in Tab. 5 show that as
the training data increases, the reconstruction performance
of the current method further improves, demonstrating its
strong scalability.
5. Conclusion
In this paper, we propose a feed-forward framework capa-
ble of directly predicting 3D human Gaussians from only

<!-- page 9 -->
two images in 190 ms on one GPU. We redesign a geomet-
ric reconstruction model for human point clouds prediction
from four viewpoints, adapting the robust geometric prior
from recent foundational reconstruction models to the spe-
cific human domain via training on human data. To com-
plete the omitted information from the input two images,
we propose a simple nearest neighbor search algorithm. 3D
human Gaussians can be directly obtained from the com-
pleted point clouds. We expect our method could widen the
applications of human body reconstruction.
References
[1] Easymocap - make human motion capture easier. Github,
2021. 6, 7, 14, 16
[2] Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Tat-Jun
Chin, Chunhua Shen, and Ian Reid. Auto-rectify network
for unsupervised indoor depth estimation. IEEE transactions
on pattern analysis and machine intelligence, 44(12):9802–
9813, 2021. 2
[3] Mingfei Chen, Jianfeng Zhang, Xiangyu Xu, Lijuan Liu, Yu-
jun Cai, Jiashi Feng, and Shuicheng Yan. Geometry-guided
progressive nerf for generalizable and efficient neural human
rendering.
In European Conference on Computer Vision,
pages 222–239. Springer, 2022. 1
[4] Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu,
Kwan-Yee Lin, and Hongsheng Li.
Generalizable neural
performer: Learning robust radiance fields for human novel
view synthesis. arXiv preprint arXiv:2204.11798, 2022. 1
[5] Alexey Dosovitskiy.
An image is worth 16x16 words:
Transformers for image recognition at scale. arXiv preprint
arXiv:2010.11929, 2020. 2
[6] Sang-Hun Han, Min-Gyu Park, Ju Hong Yoon, Ju-Mi Kang,
Young-Jae Park, and Hae-Gon Jeon. High-fidelity 3d human
digitization from single 2k resolution images. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 12869–12879, 2023. 2, 5, 6
[7] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 14
[8] I Ho, Jie Song, Otmar Hilliges, et al. Sith: Single-view tex-
tured human reconstruction with image-conditioned diffu-
sion. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 538–549, 2024.
1, 7, 12
[9] Jie Song Hsuan-I Ho, Lixin Xue and Otmar Hilliges. Learn-
ing locally editable virtual humans. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2023. 8
[10] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao
Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie.
Gaussianavatar: Towards realistic human avatar modeling
from a single video via animatable 3d gaussians. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 634–644, 2024. 2
[11] Shoukang Hu, Tao Hu, and Ziwei Liu. Gauhuman: Articu-
lated gaussian splatting from monocular human videos. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 20418–20431, 2024. 2
[12] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Ji-
axiang Tang, Deng Cai, and Justus Thies. Tech: Text-guided
reconstruction of lifelike clothed humans. In 2024 Interna-
tional Conference on 3D Vision (3DV), pages 1531–1542.
IEEE, 2024. 2
[13] Rohit Jena, Ganesh Subramanian Iyer, Siddharth Choud-
hary, Brandon Smith, Pratik Chaudhari, and James Gee.
Splatarmor:
Articulated gaussian splatting for animat-
able humans from monocular rgb videos.
arXiv preprint
arXiv:2311.10812, 2023. 2
[14] Yuheng Jiang, Zhehao Shen, Penghao Wang, Zhuo Su, Yu
Hong, Yingliang Zhang, Jingyi Yu, and Lan Xu.
Hifi4g:
High-fidelity human performance rendering via compact
gaussian splatting. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
19734–19745, 2024. 2
[15] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 3
[16] Muhammed Kocabas, Jen-Hao Rick Chang, James Gabriel,
Oncel Tuzel, and Anurag Ranjan. Hugs: Human gaussian
splats. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 505–515, 2024.
2
[17] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.
Imagenet classification with deep convolutional neural net-
works. Advances in neural information processing systems,
25, 2012. 5
[18] Youngjoong Kwon, Dahun Kim, Duygu Ceylan, and Henry
Fuchs. Neural human performer: Learning generalizable ra-
diance fields for human performance rendering. Advances
in Neural Information Processing Systems, 34:24741–24752,
2021. 1, 2
[19] Youngjoong Kwon, Dahun Kim, Duygu Ceylan, and Henry
Fuchs.
Neural image-based avatars: Generalizable radi-
ance fields for human avatar modeling.
arXiv preprint
arXiv:2304.04897, 2023.
[20] Youngjoong Kwon, Baole Fang, Yixing Lu, Haoye Dong,
Cheng Zhang, Francisco Vicente Carrasco, Albert Mosella-
Montoro, Jianjin Xu, Shingo Takagi, Daeil Kim, et al. Gen-
eralizable human gaussians for sparse view synthesis. arXiv
preprint arXiv:2407.12777, 2024. 1, 2, 5, 6, 7, 12, 14, 16
[21] Jiahui Lei, Yufu Wang, Georgios Pavlakos, Lingjie Liu, and
Kostas Daniilidis. Gart: Gaussian articulated template mod-
els. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 19876–19887,
2024. 2
[22] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Ground-
ing image matching in 3d with mast3r, 2024. 3
[23] Mengtian Li, Shengxiang Yao, Zhifeng Xie, Keyu Chen,
and Yu-Gang Jiang.
Gaussianbody: Clothed human re-
construction via 3d gaussian splatting.
arXiv preprint
arXiv:2401.09720, 2024. 2

<!-- page 10 -->
[24] Ruilong Li, Julian Tanke, Minh Vo, Michael Zollh¨ofer,
J¨urgen Gall, Angjoo Kanazawa, and Christoph Lassner.
Tava: Template-free animatable volumetric actors. In Eu-
ropean Conference on Computer Vision, pages 419–436.
Springer, 2022. 1
[25] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. Ani-
matable gaussians: Learning pose-dependent gaussian maps
for high-fidelity human avatar modeling. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 19711–19722, 2024. 2
[26] Lahav Lipson, Zachary Teed, and Jia Deng.
Raft-stereo:
Multilevel recurrent field transforms for stereo matching. In
2021 International Conference on 3D Vision (3DV), pages
218–227. IEEE, 2021. 2
[27] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard
Pons-Moll, and Michael J Black. Smpl: A skinned multi-
person linear model. In Seminal Graphics Papers: Pushing
the Boundaries, Volume 2, pages 851–866. 2023. 2
[28] I Loshchilov. Decoupled weight decay regularization. arXiv
preprint arXiv:1711.05101, 2017. 5
[29] Marko Mihajlovic, Aayush Bansal, Michael Zollhoefer, Siyu
Tang, and Shunsuke Saito.
Keypointnerf:
Generalizing
image-based volumetric avatars using relative spatial encod-
ing of keypoints. In European conference on computer vi-
sion, pages 179–197. Springer, 2022. 2
[30] Arthur Moreau, Jifei Song, Helisa Dhamo, Richard Shaw,
Yiren Zhou, and Eduardo P´erez-Pellitero. Human gaussian
splatting: Real-time rendering of animatable avatars. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 788–798, 2024. 2
[31] Panwang Pan, Zhuo Su, Chenguo Lin, Zhen Fan, Yongjie
Zhang, Zeming Li, Tingting Shen, Yadong Mu, and Yebin
Liu.
Humansplat:
Generalizable single-image human
gaussian splatting with structure priors.
arXiv preprint
arXiv:2406.12459, 2024. 2
[32] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani,
Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and
Michael J. Black. Expressive body capture: 3d hands, face,
and body from a single image. In Proceedings IEEE Conf.
on Computer Vision and Pattern Recognition (CVPR), 2019.
1, 2, 5, 6, 7, 14, 16
[33] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang,
Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body:
Implicit neural representations with structured latent codes
for novel view synthesis of dynamic humans. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 9054–9063, 2021. 1
[34] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
sion transformers for dense prediction. In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 12179–12188, 2021. 2
[35] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer.
High-resolution image
synthesis with latent diffusion models.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10684–10695, 2022. 1, 12
[36] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-
net: Convolutional networks for biomedical image segmen-
tation. In Medical image computing and computer-assisted
intervention–MICCAI 2015: 18th international conference,
Munich, Germany, October 5-9, 2015, proceedings, part III
18, pages 234–241. Springer, 2015. 5, 14
[37] Szymon Rusinkiewicz and Marc Levoy. Efficient variants of
the icp algorithm. In Proceedings third international confer-
ence on 3-D digital imaging and modeling, pages 145–152.
IEEE, 2001. 8
[38] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Mor-
ishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned
implicit function for high-resolution clothed human digitiza-
tion. In Proceedings of the IEEE/CVF international confer-
ence on computer vision, pages 2304–2314, 2019. 2
[39] Shunsuke Saito, Tomas Simon, Jason Saragih, and Hanbyul
Joo. Pifuhd: Multi-level pixel-aligned implicit function for
high-resolution 3d human digitization.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 84–93, 2020. 2
[40] Ruizhi Shao, Liliang Chen, Zerong Zheng, Hongwen Zhang,
Yuxiang Zhang, Han Huang, Yandong Guo, and Yebin Liu.
Floren: Real-time high-quality human performance render-
ing via appearance flow using sparse rgb cameras. In SIG-
GRAPH Asia 2022 Conference Papers, pages 1–10, 2022. 1
[41] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li,
and Xiao Yang. Mvdream: Multi-view diffusion for 3d gen-
eration. arXiv:2308.16512, 2023. 1
[42] Libo Sun, Jia-Wang Bian, Huangying Zhan, Wei Yin,
Ian Reid, and Chunhua Shen.
Sc-depthv3: Robust self-
supervised monocular depth estimation for dynamic scenes.
IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, 2023. 2
[43] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang,
Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian
model for high-resolution 3d content creation. In European
Conference on Computer Vision, pages 1–18. Springer, 2025.
8
[44] Shuzhe Wang,
Vincent Leroy,
Yohann Cabon,
Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In CVPR, 2024. 2, 3, 12, 14
[45] Wenbo Wang, Hsuan-I Ho, Chen Guo, Boxiang Rong, Ar-
tur Grigorev, Jie Song, Juan Jose Zarate, and Otmar Hilliges.
4d-dress: A 4d dataset of real-world human clothing with se-
mantic annotations. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
550–560, 2024. 2, 5, 6
[46] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 5
[47] Philippe Weinzaepfel,
Thomas Lucas,
Vincent Leroy,
Yohann Cabon, Vaibhav Arora, Romain Br´egier, Gabriela
Csurka, Leonid Antsfeld, Boris Chidlovskii, and J´erˆome
Revaud. Croco v2: Improved cross-view completion pre-
training for stereo matching and optical flow. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 17969–17980, 2023. 3
[48] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng
Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong

<!-- page 11 -->
Yang. Structured 3d latents for scalable and versatile 3d gen-
eration. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 21469–21480, 2025. 11
[49] Yuliang Xiu, Jinlong Yang, Dimitrios Tzionas, and Michael J
Black. Icon: Implicit clothed humans obtained from nor-
mals.
In 2022 IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition (CVPR), pages 13286–13296.
IEEE, 2022. 2
[50] Yuliang Xiu, Jinlong Yang, Xu Cao, Dimitrios Tzionas, and
Michael J Black. Econ: Explicit clothed humans optimized
via normal integration. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
512–523, 2023. 2, 8
[51] Yuxuan Xue, Xianghui Xie, Riccardo Marin, and Gerard
Pons-Moll. Human-3diffusion: Realistic avatar creation via
explicit 3d consistent diffusion models. Advances in Neural
Information Processing Systems, 37:99601–99645, 2024. 1,
8
[52] Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi
Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Gaussianob-
ject: Just taking four images to get a high-quality 3d object
with gaussian splatting. arXiv preprint arXiv:2402.10259,
2024. 1
[53] Taoran Yi, Jiemin Fang, Xinggang Wang, and Wenyu Liu.
Generalizable neural voxels for fast human radiance fields.
arXiv preprint arXiv:2303.15387, 2023. 1, 2
[54] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus,
Long Mai, Simon Chen, and Chunhua Shen. Learning to
recover 3d scene shape from a single image. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 204–213, 2021. 2
[55] Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qiong-
hai Dai, and Yebin Liu. Function4d: Real-time human vol-
umetric capture from very sparse consumer rgbd sensors. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 5746–5756, 2021. 2, 5,
6, 8
[56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 5
[57] Zechuan Zhang, Li Sun, Zongxin Yang, Ling Chen, and
Yi Yang. Global-correlated 3d-decoupling transformer for
clothed avatar reconstruction. Advances in Neural Informa-
tion Processing Systems, 36:7818–7830, 2023. 8
[58] Zechuan Zhang, Zongxin Yang, and Yi Yang.
Sifu:
Side-view conditioned implicit function for real-world us-
able clothed human reconstruction. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 9936–9947, 2024. 1, 8
[59] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu,
Shengping Zhang, Liqiang Nie, and Yebin Liu.
Gps-
gaussian: Generalizable pixel-wise 3d gaussian splatting for
real-time human novel view synthesis.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 19680–19690, 2024. 1, 2, 5, 6, 7, 12, 13,
15
Figure 9. Data collection setup visualization.
Input Images
TRELLIS (2 view)
Snap-Snap (Ours)
GT
Figure 10. Visual comparisons of Snap-Snap with TRELLIS [48].
A. Appendix
A.1. Data Collection Setup Visualization
We present the low-cost data collection setup in Fig. 9. We
place the two phones on two stands and synchronize the
images captured by the phones with a computer. Directly
reconstructing a human body from images taken in random
poses is quite challenging, so to make the camera poses of
the two mobile phones as similar as possible to those in the
training set, we manually adjust the poses of the two phones
to reduce the reconstruction difficulty.
A.2. Visual Comparison with Two-view 3D Gener-
ation
We provide a visual comparison between our method and
TRELLIS [48], which can reconstruct 3D assets based on
two images (e.g., human bodies).
As shown in Fig. 10,
our method demonstrates superior clarity and texture con-
sistency in a faster inference speed, especially in terms of
skin tone.

<!-- page 12 -->
GHG (3 view images)
GPS-Gaussian (16 view images)
Ours (2 view images)
SMPL-X
Estimator
Human
Reconstrcution
Depth
Estimator
Gaussian
Regression
Point Clods
Predictor
Gaussian
Regression
SMPL-X
Estimator
SiTH (single-view image)
Human
Reconstruction
Generative
Model
···
···
Snap-Snap
Figure 11. Comparison of pipelines with different methods.
A.3. Comparison with Different Methods
To further highlight the advantages of our approach, we
provide a comparative illustration of different representa-
tive frameworks: single-view reconstruction methods (e.g.,
SiTH [8]), sparse-view methods based on SMPL-X estima-
tions (e.g., GHG [20]), and dense-view depth-based meth-
ods (e.g., GPS-Gaussian [59]). Unlike existing approaches,
our method achieves high-quality human reconstruction in
milliseconds, using only a minimal number of input views,
without relying on generative models [35] which often
lead to uncontrollable textures and human prior estimations
which often exist misalignment due to the lacking of views.
Regression Networks
PSNR ↑
SSIM ↑
LPIPS ↓
2
22.41
88.74
13.36
1 (Ours)
22.44
88.78
13.47
Table 6. We perform ablation studies on the number of Gaussian
regressions networks.
Pretraining
PSNR ↑
SSIM ↑
LPIPS ↓
×
21.11
87.49
16.22
✓
22.44
88.78
13.47
Table 7.
We perform ablation studies on the impact of
DUSt3R [44] pretraining weight.
front-view
back-view hallucination
GT back-view
Figure 12. Visualization of different texture introduced by gener-
ative methods.
A.4. Uncontrollability of Generative Models
As shown in Fig. 12, we additionally visualize the texture
uncertainty introduced by generative model [35] in single-
view reconstruction method SiTH [8]. Even with the same
front view is provided, the model may predict inconsistent
back-view textures that significantly deviate from the ex-
pected appearance. This may lead to uncontrollable recon-
struction results and a noticeable misalignment with the tar-
get subject.
A.5. Ablation Studies on Gaussian Regression
Regarding Gaussian attribute regression, considering the
potential inconsistency between the input front/back views
and the side views generated by wrapping based on nearest
neighbor search (NNS), we conduct an ablation study on
the number of regression network, as shown in Tab. 6. The
results show that adding the regression network for the side
views specially brings no significant improvement to the hu-
man reconstruction, which also verifies the effectiveness of
ours proposed side-views enhancement module.
A.6. Foundation Geometry Prior
To validate the importance of DUSt3R’s [44] geometry pri-
ors, we present ablation experiments on loading foundation
geometry reconstruction prior in Tab. 7. The geometry pri-
ors of DUSt3R improve the reconstruction results of our
method, which demonstrates the importance of introducing
general foundation priors into the specific human domain.

<!-- page 13 -->
fusion strategy
PSNR ↑
SSIM ↑
LPIPS ↓
Concat
22.40
88.76
13.29
Average
22.44
88.78
13.47
Table 8. Comparison on fusion strategy of side-view tokens.
Input images
Ours
GT
Input images
Ours
GT
Figure 13. Reconstruction results based on arbitrary image inputs.
A.7. Token Fusion Strategy
In Sec. 3.3 of the main paper, we obtain global side-view
tokens by averaging the front and back image tokens. In
Tab. 8 we present ablation experiments comparing the aver-
aging and concatenation strategies for obtaining side-view
tokens. We observe that both methods yield good recon-
struction quality, which demonstrates the effectiveness of
the point cloud prediction model.
A.8. Reconstruction results from arbitrary inputs.
We provide visualization results of human reconstruction
using two input images from two views apart from front and
back views in Fig. 13. As shown in the figure, our method
is able to produce high-quality reconstructions even from
arbitrary image input, demonstrating strong generalization
capability.
A.9. Multi-view Scenario
To further evaluate our reconstruction performance, we con-
duct experiments in a scenario with five input views and
compare it with GPS-Gaussian [59], as shown in Tab. 9 and
Fig. 18. Considering that it is different from the two-view
scenario, there is no information lacking in the scenario
with 5 input views. We only predict the point clouds in
the two input views, removing the part that predicts the side
view point clouds. Our algorithm achieves high-definition
reconstruction under multiple views. We conduct quanti-
tative and qualitative comparisons with GPS-Gaussian. It
can be observed that our reconstruction method not only
achieves complete human body reconstruction in multi-
view input scenarios but also attains higher reconstruction
quality.
Method
PSNR ↑
SSIM ↑
LPIPS ↓
GPS-Gaussian [59]
20.76
88.04
14.83
Snap-Snap (Ours)
24.73
91.59
10.67
Table 9. We compare the reconstruction results of our method with
GPS-Gaussian [59] in a scenario with five input views.
w/ Side Head
w/o Side Head
Figure 14. We present the visual results of the impact of the addi-
tional side-view heads on the human construction.
Input Images
Rendered Images
Figure 15. Reconstruction results of the same human in different
poses.

<!-- page 14 -->
0
50
75
100
125
175
200
Time (ms)
Inference Time
91ms
87ms
12ms
Inference Time Distribution
DUSt3R Inference
150
Gaussian Regression
25
Side Enhencement
Figure 16. The visual results of the time consumption for different
modules in the inference process.
A.10. Impact of Overlapping Point Clouds.
In the main paper, we adapt additional side-view prediction
heads to complete the geometry of the side views. How-
ever, it is unavoidable that there is some overlap between
the point clouds predicted for the side views and those for
the front view, which may affect the quality of front view re-
construction. Therefore, in Fig. 14, we present visualization
results of front view reconstruction before and after incor-
porating side-view prediction heads. We can observe that
after adding side-view prediction heads, although there are
some additional overlapping point clouds in the front view,
this has almost no impact on the front view reconstruction
quality.
A.11. Consistency of Reconstruction
To validate the consistency of our method in reconstructing
human bodies with different poses, we present reconstruc-
tion results of the same human in various poses in Fig. 15.
We can observe that our method achieves good reconstruc-
tion quality across different poses while maintaining excel-
lent consistency.
A.12. Time-consuming Analysis
In Fig. 16 we show the inference time spent by each module
of our model, with the total inference time around 190 ms.
A.13. Network Architecture
We use a UNet-like [36] network architecture in the Gaus-
sian attribute regression. The specific architecture is shown
in the Fig. 17. For the Gaussian attribute regression net-
work, the inputs are the images from four viewpoints, which
are concatenated together and fed into the network. The
output is the corresponding Gaussian attributes for the cor-
responding view. Considering the resolution of input im-
ages, we only use ResNet blocks [7] to form the UNet-like
network. The downsampling layer number and upsampling
layer number are 3. For Gaussian attribute regression net-
work, we use convolution dimensions of (16, 16, 16).
A.14. Discussion on Snap-Snap
We observe that the reconstruction results of Snap-Snap still
contain some holes, particularly around the armpits or re-
gions occluded by the arms. These missing parts in the hu-
man Gaussians are due to the limitations of point cloud su-
pervision, which is derived from depth maps [44], wherein
certain points are filtered out because of occlusions. The
hollows on the human bodies could potentially be reduced
with the use of geometric generative priors. As the occluded
areas are relatively small, they have little effect on the con-
sistency of the final reconstruction.
A.15. Discussion on GHG [20]
We further visualize the SMPL-X [32] parameters predicted
based on EasyMocap [1] in Fig. 19, and it can be ob-
served that estimating human body parameters using only
two views is very challenging.
The visualization results
show that the predicted SMPL-X parameters have a signifi-
cant impact on reconstruction methods based on the SMPL-
X model, such as GHG [20]. In contrast, our method recon-
structs the human body based on directly predicted point
clouds, effectively avoiding inaccuracies in SMPL-X pa-
rameter estimation.

<!-- page 15 -->
Block
1
Block
7
Block
2
Block
6
Block
3
Block
4
Block
5
Resnet
Block
Resnet
Block
Resnet
Block
Resnet
Block
Resnet
Block
Resnet
Block
Resnet Block
Conv
SiLU
GroupNorm
Downsampling layers
Upsampling layers
Figure 17. The network architecture of the Gaussian attribute regression network.
Input 5 Views
GPS-Gaussian
Snap-Snap (Ours)
GT
Figure 18. We visualize the comparison of reconstruction results between our method and GPS-Gaussian [59] with five input views.

<!-- page 16 -->
GHG
Snap-Snap (Ours)
GT
Keypoints
SMPL-X
Figure 19. We visualize the reconstruction results of GHG [20] based on SMPL-X [32] parameters predicted by EasyMocap [1], as well as
our reconstruction results in the same scenario. The first column shows the input views, the second column shows the detected keypoints,
and the third column shows the corresponding predicted SMPL-X mesh visualization results. Columns 4-9 show the reconstruction results
of GHG, our reconstruction results, and the ground truth, respectively.
