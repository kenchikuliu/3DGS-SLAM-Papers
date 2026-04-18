<!-- page 1 -->
UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering
Yusen Xie1
Zhenmin Huang2
Jianhao Jiao3
Dimitrios Kanoulas3
Jun Ma1,2
1HKUST (GZ)
2HKUST
3UCL
{yxie827@connect.hkust-gz.edu.cn; zhuangdf@connect.ust.hk; ucacjji@ucl.ac.uk;
d.kanoulas@ucl.ac.uk; jun.ma@ust.hk}
Project Page:
https://github.com/xieyuser/UniGS
Figure 1. We propose a unified 3D Gaussian Splatting framework that jointly predicts RGB, depth, normal, and semantic map through a
single forward rasterization process. (a) Our geometry-aware rendering method explicitly incorporates rotation and scaling attribute into
both the rasterization and gradient propagation processes, where we derive an analytical solution for gradient propagation to significantly
accelerate optimization. (b) We present multimodal rendering results across diverse scenes, showcasing RGB images, depth maps, surface
normals, and semantic segmentation map in a four-panel visualization layout. (c) A comprehensive comparison with widely-used baselines
in Replica-Room0 dataset, underscoring the full modality coverage and strong performance of our framework. All scores are normalized
to the [0, 1] range by defining the worst and best reference values for each metric individually, ensuring 1 represents optimal performance.
Incomplete dashed lines indicate that the corresponding baseline does not support specific modalities. Evaluation covers PSNR for RGB
quality, absolute relative error (Abs.Rel.) for depth estimation, cosine similarity (CosSimi) for normal estimation, mean intersection-over-
union (mIoU) for semantic segmentation, as well as system performance metrics including the number of Gaussian primitives (Count) and
rendering speed in frames per second (FPS).
Abstract
In this paper, we propose UniGS, a unified map represen-
tation and differentiable framework for high-fidelity multi-
modal 3D reconstruction based on 3D Gaussian Splatting.
Our framework integrates a CUDA-accelerated rasteriza-
tion pipeline capable of rendering photo-realistic RGB im-
ages, geometrically accurate depth maps, consistent sur-
face normals, and semantic logits simultaneously. We re-
design the rasterization to render depth via differentiable
ray-ellipsoid intersection rather than using Gaussian cen-
ters, enabling effective optimization of rotation and scale
attribute through analytic depth gradients. Furthermore,
we derive the analytic gradient formulation for surface nor-
mal rendering, ensuring geometric consistency among re-
constructed 3D scenes. To improve computational and stor-
age efficiency, we introduce a learnable attribute that en-
ables differentiable pruning of Gaussians with minimal con-
tribution during training. Quantitative and qualitative ex-
periments demonstrate state-of-the-art reconstruction accu-
racy across all modalities, validating the efficacy of our
geometry-aware paradigm. Source code and multimodal
viewer will be available on GitHub.
1. Introduction
High-quality novel view synthesis (NVS) and fast mul-
timodal 3D reconstruction remains a challenging prob-
lem in computer graphics and attracts widespread atten-
tion in fields such as autonomous driving [19, 20, 67, 79]
and robotics [40, 83].
In recent years, neural radiance
fields (NeRF) [1, 2, 43, 56] and 3D Gaussian Splatting
(3DGS) [26, 27] have made big breakthroughs in this field.
NeRF uses coordinate-based networks to store 3D scene in
multi-layer perceptrons (MLPs) via ray sampling, bring-
ing in photo-realistic reconstruction.
3DGS, another ap-
proach, uses explicit 3D Gaussian primitives and modern
GPU-accelerated splatting, and this design has been shown
1
arXiv:2510.12174v2  [cs.CV]  13 Nov 2025

<!-- page 2 -->
to enable high-quality real-time NVS by many 3DGS-based
frameworks [12, 18, 30, 38, 39, 41, 46, 50, 66, 67]. How-
ever, the two types of existing methods still have many is-
sues, which mainly focus on the following aspects. NeRF
requires long training times, lacks easy scene editabil-
ity, and its implicit 3D information storage method ren-
ders geometric reconstruction less intuitive [43, 80]. Ad-
ditionally, it performs poorly when applied to large-scale
scenes [51, 54].
3DGS and its related frameworks have
more challenges: 1) Poor geometric structure and consis-
tency: conventional image-centric 3DGS often ignores ob-
jects’ geometric structure and semantic info. It only uses
Gaussian centers to represent depth, leaving out Gaussian
rotation and scale attributes. This stops gradient propaga-
tion for geometric constraints (e.g. depth and normal), lead-
ing to inconsistent surface normals [26, 34, 41, 62] and mis-
alignment between Gaussians primitives and reconstructed
surfaces.
2) Inefficient densification and pruning: some
3DGS methods create too many redundant Gaussians prim-
itives to fit RGB images well [26, 27, 34, 67].
Though
this improves metrics, it slows down rendering and uses
more GPU memory, losing 3DGS’s real-time edge.
3)
Poor unified multimodal framework: some 3DGS-related
methods use separate frameworks for multimodal render-
ing [15, 33], which causes repeated calculations and in-
consistent 3D scene representations, failing to unify multi-
modal modeling. Existing studies have tried single or mul-
tiple scene modalities [10, 34, 41, 62], but there is still no
unified framework that combines all these modalities while
keeping storage efficient and rendering fast. This gap serves
as the key motivation for our approach.
Against this backdrop, we propose UniGS, a unified
geometry-aware 3D Gaussian Splatting map representation
and framework that establishes a new paradigm for con-
sistent 3D scene reconstruction.
During rendering, each
pixel is shaded via backward ray tracing to determine its
intersection with individual Gaussian ellipsoids (Fig. 1(a)).
This formulation explicitly integrates Gaussian rotation and
scale attributes into the rendering pipeline. By leveraging
analytical gradients derived from depth and normal super-
vision, it enables efficient gradient-based optimization of
these geometric attributes.
Furthermore, we introduce a
learnable attribute to continuously prune Gaussians. Exper-
imental results demonstrate that this attribute significantly
improves computational and storage efficiency without sac-
rificing rendering or geometric accuracy.
Crucially, our
framework achieves high-fidelity rendering of RGB, depth,
normals, and semantic logits through a single differentiable
pipeline simultaneously (Fig. 1(b)).
The loss from each
modality can be back-propagated via analytical gradients to
the shared 3D Gaussian representation, thereby maintain-
ing consistency across all modalities in the reconstructed
environment (Fig. 1(c)). Particularly, our method achieves
a 66.4% improvement in depth estimation accuracy while
reducing the Gaussian primitive count by 17.2%. In this
paper, our contributions are summarized as follows:
• We develop a novel differentiable depth rasterization
method based on ray-ellipsoid intersection, which enables
optimization of Gaussian rotation and scale attributes
via analytic gradients propagation.
This approach en-
sures that Gaussian primitives conform closely to un-
derlying surface geometries, significantly improving geo-
metric consistency in reconstructed scenes.
• We propose a trainable gradient factor in map represen-
tation for dynamically pruning insignificant Gaussians in
a differentiable manner, which enhances rendering speed
and storage efficiency.
• We propose a unified map representation and differen-
tiable rasterization framework that integrates multimodal
data. This integration not only enables mutual enhance-
ment across modalities but also improves rendering ef-
ficiency and strengthens the consistency of the recon-
structed environment.
• Through comprehensive experimental evaluation, we
demonstrate that our approach achieves state-of-the-art
reconstruction quality across all modalities while consis-
tently maintaining real-time performance.
2. Related Works
Photo-Realistic Rendering and Densification. In the do-
main of photo-realistic reconstruction and rendering, NeRF
and 3DGS represent implicit and explicit techniques, re-
spectively.
NeRF [1, 43] and its variants pioneer the
use of implicit neural representations for NVS, achieving
high-fidelity rendering by optimizing a continuous volu-
metric scene network using multi-view images. However,
this approach typically requires extremely long training
times. Subsequent works incorporate geometric prior struc-
tures [2, 55, 74] or efficient encoding [45] to accelerate the
process. Despite these improvements, NeRF still faces in-
herent limitations, such as the difficulty in editing implic-
itly represented scenes. In contrast, 3DGS [26, 68] em-
ploys explicit Gaussian primitives as scene representations.
Through compression [12, 18, 30, 41, 46] and efficient geo-
metric structures [4, 27], several studies extend 3DGS to
large-scale outdoor [35, 50, 66, 67] and urban [38, 39]
environments while maintaining high-quality reconstruc-
tion.
But the aforementioned research primarily focuses
on RGB-based rendering and seldom incorporates explicit
geometric information of the environment. Moreover, im-
pressive efforts have been made to enhance optimization
and storage efficiency. SteepestGS [59] uncovers funda-
mental principles of density control. DashGaussian [5] sig-
nificantly accelerates 3DGS optimization. GeoTexDensi-
fier [21] utilizes geometric information for Gaussian den-
sification, and Pixel-GS [78] introduces pixel-level regu-
larization for density control. These methods are regarded
2

<!-- page 3 -->
Figure 2. Overview of our framework. The yellow arrows (→) indicate the forward rasterization pipeline. (a) First, we predict the
semantic logits o, contribution k, and other 3DGS attributes (i.e., position µ, rotation q, scale s, opacity α, SHs h) in the anchor set through
MLPs. For each anchor, we predict M raw 3DGS data available for rendering. (b) Then, during the rasterization stage, for each pixel, we
use ray tracing to calculate its intersection (×) with the Gaussian ellipsoid, which is closely related to the rotation and scale attributes. We
replace the depth of the Gaussian center point (•) with the depth of the midpoint (▲) between the two intersection points. After splatting
rasterization, we obtain RGB (①), gradient factor map (②), semantic (③), and depth (④). (c) To obtain the surface normal (⑤), we back-
project the depth into three-dimensional space and then use a differential method to obtain the normals in the world coordinate system,
ultimately producing the normal rendering map. The final rendered results are compared with their respective ground truths to calculate
the loss. The green arrows (←) indicate the direction of gradient propagation through a CUDA-accelerated analytical solution.
as post-processing techniques and lack integration within a
unified framework.
Geometry-Aware
Rendering.
Geometry-aware
ap-
proaches enhance reconstruction accuracy by incorporat-
ing surfel [9, 14, 17, 23, 62], normals [51, 62], and re-
lighting [13, 65, 72] capabilities, enabling more physi-
cally realistic renderings and interactions.
Some stud-
ies [47, 51, 58, 69] attempt to achieve accurate geometry
within the NeRF framework, and while significant progress
has been made, these methods still suffer from inherent
limitations that are difficult to overcome with implicit rep-
resentations. 3DGS [26, 41] does not inherently excel in
geometric reconstruction either, as its Gaussian primitives
are independent of each other, making precise geometry re-
covery an open challenge. Several works unitize 3DGS or
2DGS to recover depth [26, 41, 57, 62], surfel [17, 23, 62]
and meshes [9, 14, 17], including normal estimation [62]
and relighting [13, 65, 72] techniques, which can effec-
tively reconstruct smooth object surfaces.
Nevertheless,
these approaches remain largely limited to small-scale ob-
jects. When extended to large environments such as out-
door or urban scenes [44, 64], the required computational
time remains substantial. Some methods attempt to incor-
porate SDF-based surfel constraints [36, 60, 69, 70, 75], in-
verse rendering [52, 71], or restrictions [25, 34] on Gaussian
properties. However, these approaches lack rigorous analyt-
ical derivation for geometric modeling, employ relatively
limited quantitative metrics, and demonstrate constrained
generalizability across diverse scenarios. Consequently, the
improvements achieve thus far remain marginal in terms of
geometric fidelity.
Semantic Rendering Frameworks.
Several works at-
tempt to integrate semantic information into both NeRF [6,
80] and 3DGS [15, 48], including techniques like Open-
Vocabulary Segmentation [10, 29, 76] and Language-
Driven Semantic Segmentation [32, 37, 42, 49], with
promising results. Embedding semantic information facil-
itates downstream tasks such as scene understanding [22,
81], navigation [24, 31], and scene editing [11, 73]. For
instance, SGS-SLAM [33] directly uses semantic ground
truth as supervision signal and rendering output in 3DGS,
though it fails to formulate a unified 3DGS map represen-
tation. Feature-GS [82] extracts semantic features from the
environment using 3DGS, and SEGS-SLAM [63] enhances
structural awareness with appearance embeddings. Never-
theless, such approaches exhibit limitations in reconstruct-
ing accurate environmental geometry and fail to achieve
consistent reconstruction of scene geometry and semantics.
3. Methodology
The methodology section is organized as follows. Firstly,
we define our map representation and objective of this work
(Sec. 3.1).
Then, the subsequent sections detail the key
components of our framework: geometry-aware depth ras-
terization and normal estimation (Sec. 3.2); the novel gra-
dient factor and differentiable pruning strategy (Sec. 3.3);
and finally, the overall multimodal rendering pipeline, loss
functions, and implementation details (Sec. 3.4). The brief
overview of our framework is illustrated in Fig. 2.
3.1. Unified Map Representation and Objectives
Unified Map Representation.
In our framework, each
Gaussian G is defined by position µ ∈R3, quaternion
q ∈R4, scale s ∈R3, opacity α ∈R, and Spherical Har-
monics (SHs) h ∈RCh (Ch is the pre-difined SHs number
per color channel), semantic logits (encode the probability
distribution across Co semantic categories) o ∈RCo and
3

<!-- page 4 -->
gradient factor k ∈R:
G = {µ, q, s, α, h, o, k}.
(1)
Objective. The objective of this framework is to render
RGB color image, geometric depth, surface normals, se-
mantic logits, and a gradient factor map (used in differen-
tiable pruning) simultaneously within a unified rasterization
pipeline. These rendered outputs are then compared against
their corresponding 2D ground truth to compute respective
loss, which guide the optimization of the 3D Gaussian scene
through a differentiable backward process. To accelerate
convergence and enhance computational efficiency, we de-
rive the relevant analytical solutions and implement them
using CUDA-accelerated gradient propagation. Finally, we
can obtain a geometry consistent photo-realistic 3DGS map
representation.
3.2. Geometry-Aware Rasterization
Ray-Gaussian Primitive Intersection Formulation. We
define an ellipsoid E associated with Gaussian G by its cen-
ter µ, rotation matrix R (converted from quaternion q) and
diagonal scaling matrix S = diag(sx, sy, sz), and this ellip-
soid consists of all 3D points x on its surface that satisfy the
following equation:
 S−1R−1(x −µ)
⊤ S−1R−1(x −µ)

= 1.
(2)
For each pixel p = (u, v), the corresponding camera ray
r(t) = tworld
cam + td is calculated through a unprojection pro-
cess:
r(t) = tworld
cam + normalize


Rworld
cam


u−cx
fx
v−cy
fy
1




,
(3)
where tworld
cam
and Rworld
cam
are the translation vector and rota-
tion matrix that transform a point in the camera frame to
the world frame, respectively. The focal lengths fx, fy and
principal point cx, cy are camera parameters.
To find the intersection between the ray r(t) and the
ellipsoid E, we transform the ray r(t) into the local (l)
scaled (s) coordinate system of E:
vs = S−1vl,
vl = R−1(tworld
cam −µ),
ds = S−1dl,
dl = R−1d.
(4)
The intersection problem is then reduced to solving for t in
the unit sphere equation:
∥vs + tds∥2 = 1.
(5)
The solution to this equation can be treated as that of a
quadratic equation: for a light ray intersecting an ellipsoid,
there are two valid solutions (t1, t2) for two intersection
points, t1 = t2 for one point, and no valid solutions for
no intersection. Details are in the supplementary materials.
Depth Rasterization. We take the midpoint tmid = (t1 +
t2)/2 of the two valid solutions (t1, t2) as the intersection
results. The replaced depth d of corresponding 3D primitive
in camera coordinate can be computed by
Pcam = Rcam
world(tworld
cam + tmidd),
d = Pcam, z.
(6)
The final depth value ˜D(u, v) for pixel p = (u, v) is ob-
tained by alpha-rendering the depths d of all overlapping
Gaussians N:
˜D(u, v) =
X
i∈N
αidi
i−1
Y
j=1
(1 −αj).
(7)
The above method allows to directly incorporate the geo-
metric information (rotation q and scale s of the Gaussian
G) into the rasterization pipeline. This enables us to com-
pute the gradients of the loss function with respect to the
rotation q and scale s, thereby effectively optimizing the
Gaussian G. Related gradient backpropagation derivations
are provided in the supplementary materials.
Algorithm 1: Surface Normal Estimation by Depth
1 Input: Depth ˜D (u, v), STEP 1, STEP 2, λ
2 Output: Normal ˜N(u, v)
3 for each pixel p = (u, v) in parallel do
4
1. Backproject to World Coordinate
5
2. Compute Multi-Scale Normals
6
for gi ∈{STEP 1, STEP 2} do
7
Finite differences of neighboring points:
vx,i = P(u + gi, v) −P(u −gi, v),
vy,i = P(u, v + gi) −P(u, v −gi),
8
Estimate normal: ni = vx,i × vy,i
9
3. Fuse and Normalize
10
Ensure consistency: n2 ←n2 · sign(n1 · n2)
11
Fuse estimates: nfused = λn1 + (1 −λ)n2
12
Normalize: ˜N(u, v) = nfused/∥nfused∥
13
4. Orient Towards View Direction
Normal Rasterization from Depth. As mentioned previ-
ously, it is challenging to obtain sufficiently smooth and
accurate surface normals through rasterization-based ren-
dering due to the strong directional information inherent in
normals. Therefore, in this paper, we compute normals in-
directly from depth rasterization result. The focus lies on
deriving the gradient propagation process from the normal
loss back to the depth.
Our approach begins by unprojecting the depth map into
world coordinate using the camera parameters. We then
estimate the normals in world coordinate through a multi-
step finite difference scheme, with step sizes denoted as
STEP 1, STEP 2. The overall procedure is summarized
in Alg. 1. Further details regarding the backprojection of
depth pixels and orientation toward the view direction will
be provided in the supplementary material. Additionally,
the derivation of backpropagation from the normal loss to
depth will also be included in the supplementary material.
4

<!-- page 5 -->
3.3. Differentiable Pruning
Gradient Factor Rasterization and Backpropagation.
We define a new attribute k in (1) for each Gaussian G
to measure its gradient level during the rendering process.
Alongside the rendering of RGB image, we can render a 2D
gradient factor map ˜K by alpha-blending:
˜K(u, v) =
X
i∈N
αiki
i−1
Y
j=1
(1 −αj).
(8)
Our objective is to drive each Gaussian’s k toward 1. Ex-
perimental results show this state corresponds to the optimal
optimization effect of Gaussians. To enforce this, we mini-
mize the L1-loss between ˜K and an all-ones matrix 1:
LK = ∥˜K −1∥1.
(9)
Detailed analysis on why k = 1 is optimal and gradient
backpropagation are provided in the supplementary materi-
als.
Pruning. After a fixed interval of iterations Kp (Kp =
3000 in our experiments), our framework directly elimi-
nates Gaussians with abnormal gradient factor ∥k −1∥1 >
Tk , effectively pruning the set and reducing the number of
Gaussians to enhance overall efficiency. After pruning op-
eration, the k value of all Gaussians will be reset to 0.9 to
maintain the sustainability of updates. Please refer to our
code for details.
3.4. Multimodal Rendering Framework
Architecture Overview. Our system is built upon a mod-
ified Scaffold-GS [41] architecture. For each anchor point,
we maintain a foundational 3D Gaussian representation (1).
During every training iteration, multiple MLPs are used to
predict attribute offsets for each anchor.
Attribute Prediction. For semantic logits o ∈RCo and
gradient factor k ∈R of a Gaussian G, we replicate (de-
noted as Repeat) M times and modulate by the output of an
MLP branch to obtain Gaussian features for rasterization:


ovis
kvis
...

= Repeat(


oah
kah
...

, M) ⊗σ


MLPo(fah)
MLPk(fah)
...

, (10)
where oah ∈RCo is the base semantic logits of the anchor,
kah ∈R is the base gradient factor of the anchor. M is the
replication factor. fah represents the input feature for the an-
chor. MLPo : RF dim →RCo denotes the MLP branch for
semantics. MLPk : RF dim →R denotes the MLP branch
for gradient factor. σ is the sigmoid activation function.
⊗denotes element-wise multiplication. Fdim is the fea-
ture dimension. The resulting modulated attributes are then
combined with other predicted properties (i.e., position µ,
rotation q, scale s, opacity α, SHs h, denoted as ...) to form
the final 3D Gaussians ready for projection and rendering.
Semantic Rendering and Backpropagation.
We render
a 2D semantic logits ˜O ∈RH×W ×Co along with rasteriza-
Algorithm 2: Multimodal Forward Rasterization
Pipeline
Input: G = {µ, q, s, α, h, o, k}, Image
dimensions W, H
Output: ˜C ∈RW ×H×3, ˜O ∈RW ×H×C,
˜D ∈RW ×H, ˜K ∈RW ×H
1 for each pixel = (u, v) do in parallel
// Compute Ray of pixel
2
computeRay (3);
3
Initialize accumulators: C ←0, O ←0,
D ←0, K ←0, T ←0;
4
for each Gaussian i intersecting pixel (u, v) do
5
...
6
hi, oi, αi ←Gaussian i parameters;
// Ray-Ellipsoid Intersection
7
rayEllipsoidIntersection (5) ;
// Use Midpoint for Depth
8
di ←obtainDepth (6) ;
// Alpha-Blending for All
Attributes
9
for each ch in Ch do
10
C[ch] ←C + hi · αi · T;
11
for each ch in Co do
12
O[ch] ←O + oi · αi · T;
13
D ←D + di · αi · T;
14
K ←K + ti · αi · T;
15
...
// Pixel-Wise Assignment
16
˜C[x, y] ←C;
˜O[x, y] ←O;
17
˜D[x, y] ←D;
˜K[x, y] ←K;
18
Save Tfinal = T;
tion pipeline. The forward process for the semantic logits
in pixel p = (u, v) and channel ch is defined as
˜O[ch](u, v) =
X
i∈N
αioi[ch]
i−1
Y
j=1
(1 −αj).
(11)
Then the rendered semantic logits ˜O will be compared to its
ground truth Ogt using Cross-Entropy Loss, resulting in se-
mantic loss Lseg. Let
∂Lseg
∂˜o[ch] be the gradient of the loss with
respect to the rendered ch-th channel of a specific pixel. The
chain rule for the gradient of Lseg with respect to a specific
semantic channel oi[ch] of a Gaussian G is given by
∂Lseg
∂oi[ch]
= ∂Lseg
∂˜O[ch]
· ∂˜O[ch]
∂oi[ch]
,
(12)
where
∂˜O[ch]
∂oi[ch] can be derived from (11).
Tile-based Forward Rasterization. We extend the original
3DGS’s tile-based rendering pipeline [26]. During rasteri-
zation, for each pixel p = (u, v), we calculate the direction
5

<!-- page 6 -->
Table 1. Quantitative comparison experiments on the Replica dataset show that our method significantly outperforms other baselines.
Dataset
O0S1
O1S2
R1S2
R2S2
Method | Metrics
Abs.Rel ↓
RMSE↓
CS ↑
Abs.Rel↓
RMSE↓
CS ↑
Abs.Rel↓
RMSE↓
CS ↑
Abs.Rel↓
RMSE ↓
CS↑
GeoGaussian [34]
0.5828
0.8006
0.3046
2.8959
1.1092
0.1861
0.1336
0.3236
0.5700
0.1082
0.2592
0.5998
3DGS [26]
1.2231
0.9183
0.2304
0.8711
0.6238
0.3201
0.2134
0.4092
0.4963
1.4422
1.0972
0.2905
Scaffold-GS [41]
2.2857
1.2102
0.2195
1.4609
0.9413
0.2854
4.0575
1.4786
0.1555
-
-
0.1286
2DGS [18]
0.0089
0.0679
0.8775
0.0172
0.0399
0.9299
0.0451
0.1400
0.9112
0.0735
0.1759
0.8480
Normal-GS [62]
0.0074
0.0657
0.8755
0.0154
0.0344
0.9234
0.0243
0.0766
0.9234
0.0192
0.0589
0.8987
Ours
0.0062
0.0310
0.8880
0.0015
0.0073
0.9686
0.0136
0.0467
0.9545
0.0144
0.0465
0.9076
Figure 3. Across various dataset tests, our method demonstrates impressive results in geometric reconstruction. Even in textureless areas
and edge regions, it remains capable of recovering fine geometric details.
d of the camera ray r(t) and then compute the intersection
t1/t2 with the Gaussian projected onto this pixel. Finally,
we obtain the corresponding attributes using alpha blending
rendering. The multimodal forward rasterization pipeline
is outlined in Alg. 2. The corresponding backpropagation
pipeline is described in the supplementary materials.
Loss Function. For the RGB image loss, we employ a com-
bination of L1 loss Ll1 and SSIM [61] loss Lssim. For the
normal loss Lnormal, we use cosine similarity, which is cal-
culated as
Lnormal = 1 −Cosine Similarity( ˜N, ˜Ngt).
(13)
Incorporating the gradient factor loss LK and semantic loss
Lseg, we construct the comprehensive loss function as
Lall = λ1Ll1 + λ2
|Ll1|Lssim
|Lssim|
+ λ3
|Ll1|Lnormal
|Lnormal|
λ4
|Ll1|Ldepth
|Ldepth|
+ λ5
|Ll1|Lseg
|Lseg|
+ λ6
|Ll1|LK
|LK|
.
(14)
where weights are applied as λ1 = 1, λ2 = λ3 = λ4 =
λ4 = λ5 = λ6 = 0.1.
Implementation Details. Our framework is implemented
by CUDA using the LibTorch framework [7], incorporating
CUDA code for Gaussian Splatting and trained on a server
with a 2.60 GHz Intel(R) Xeon(R) Platinum 8358P CPU,
1T GB RAM, and a NVIDIA RTX 4090 24 GB GPU. In all
scequences, the hyprparameters used in our experiments are
listed in the supplementary materials.
4. Experiments
In this section, we first introduce the experimental setup in
Sec. 4.1, including datasets, baselines, and parameter set-
tings, etc. Given that RGB rendering is not the central fo-
cus of our study, the corresponding comparison will be pro-
vided in the supplementary material accordingly. Then in
Sec. 4.2, we evaluate the geometry rendering performance
(e.g. depth and normal). In Sec. 4.3, we evaluate the se-
mantic rendering performance. In Sec. 4.4, we provide the
analysis of running time and GPU memory consumption of
the framework. Sec. 4.5 shows ablation experiments of the
proposed framework.
4.1. Experiment Setup
Datasets. We conduct comparative experiments on widely
used datasets, including 7 scenes from Mip-NeRF [1], 2
scenes from Tanks & Temples [28] and 2 scenes from
Deep Blending [16]. All of these datasets do not provide
precise geometric and semantic ground truth. To address
this, we use the results obtained from the depth render-
ing pipeline in Normal-GS [62] as geometric priors for su-
pervision. At the same time, we utilize open vocabulary
queries to pre-process semantic labels with OpenSeed [76].
For accurate quantitative comparison experiments, we use
6

<!-- page 7 -->
Figure 4. A qualitative demonstration showing that the Gaussian
primitives reconstructed by our method are geometrically closer to
the real surface.
the Replica [53] simulation dataset, which provides com-
plete multimodal ground truth. We employ the pre-rendered
Replica dataset from HiSplat [55], which includes 14 mo-
tion sequences across 7 scenes. The initial points in the
Replica [53] dataset are obtained through mesh sampling,
with a sampling interval of 5 in all sequences.
Baselines and Metrics.
We compare our method with
the existing SOTA NeRF-based dense framework Mip-
NeRF [1], iNGP [45] and 3DGS framework [26], Scaffold-
GS [41], GeoGaussian [34], 2DGS [18], Normal-GS [62],
SemanticGaussian [15]. We evaluate the RGB rendering
performance using PSNR, SSIM [61], and LPIPS [77]. The
evaluation metrics for depth estimation we used are Abs.Rel.
and RMSE. The evaluation metric we used for normal es-
timation is cosine similarity (CosSimi), for semantic seg-
mentation, we employed mean mIoU.
4.2. Geometry Rendering Evaluation
We conduct detailed qualitative and quantitative com-
parisons of depth and normal estimation against meth-
ods [26] [41] [23] [34] [62] [18] on the Replica [53] dataset.
The comparison results are shown in Fig. 3 and Tab. 1.
Normal-GS [62] and our method optimize the surface nor-
mals, and both qualitatively and quantitatively outperform
other 3DGS frameworks. Our approach also exhibits supe-
rior geometric accuracy in some extreme scenarios, particu-
larly in texture-less areas, surpassing even Normal-GS [62]
and 2DGS [18]. More comparisons about geometry render-
ing can be found in supplementary materials.
We further compare the geometric impact after modify-
ing our depth rendering method. Firstly, we complete the
reconstruction process of a 3D Gaussian scene and then
identify a detailed area in the reconstructed scene (as shown
Figure 5. Quantitative comparison between our method and SGS-
SLAM [33] in terms of RGB rendering and semantic segmentation
clearly shows that our approach outperforms SGS-SLAM. The
top-left corner of the image displays the PSNR value computed
against the ground truth image.
Table 2. Quantitative geometric comparison between our method
and other methods after sampling the reconstructed Gaussian
model, compared to the initial points or mesh points.
Dataset
MipNeRF-Garden
Replica-Room0
Method | Metrics
Radio↓
Mean↓
Std↓
Hau-Dis↓
Radio ↓
Mean↓
Std ↓
Hau-Dis ↓
Scaffold-GS [41]
0.0390
0.0299
0.0383
0.592
0.0476
0.0120
0.0083
0.141
2DGS [18]
0.0607
0.0984
0.1272
1.091
0.0628
0.0519
0.0244
0.462
Normal-GS [62]
0.0388
0.0303
0.0377
0.936
0.0477
0.0120
0.0082
0.131
Ours
0.0341
0.0272
0.0298
0.405
0.0339
0.0137
0.0081
0.078
in Fig. 4). It is evident that our method, compared to oth-
ers, results in optimized Gaussian primitives that tend to be
closer to the actual surface of the object. This is because our
method integrates rotation and scale attribute into the opti-
mization process, thereby achieving geometric consistency.
Additionally, we perform point sampling [34] of the Gaus-
sian primitives within specific small regions, followed by a
Z-score statistical comparison with the initial ground truth
and a Hausdorff distance comparison, as shown in Tab. 2.
It is evident that our method achieves consistent geometric
surface reconstruction. The calculation method using the Z-
score statistic is explained in the supplementary materials.
Table 3.
Quantitative comparison with the SGS-SLAM [33]
method in terms of RGB rendering, semantic rendering and FPS.
TT represents training time (minutes). Best results are underlined.
Dataset
All
Room0
Room2
Office2
Method | Metrics
TT↓
PSNR ↑
FPS ↑
mIoU ↑
PSNR ↑
FPS ↑
mIoU↑
PSNR ↑
FPS ↑
mIoU ↑
SGS-SLAM [33]
212.6
36.28
96.4
0.976
39.14
72.2
0.978
36.94
112.3
0.944
Ours
32.2
36.66
184.2
0.987
40.06
234.4
0.984
38.16
229.3
0.975
4.3. Semantic Rendering Evalution
We compare our method with the SGS-SLAM [33] frame-
work, which employs a task-specific Gaussian map repre-
sentation and is supervised using semantic result images for
semantic tasks, differing from our unified framework that
7

<!-- page 8 -->
Figure 6. The use of a ray casting-based depth rendering approach
significantly enhances the algorithm’s robustness in textureless re-
gions. Furthermore, the Gaussian splats provide a tighter fit to the
actual object surfaces.
predicts semantic logits. The comparison across multiple
scenes is shown in Fig. 5 and Tab. 3. From the quantitative
results, it can be observed that our method achieves superior
performance in both image rendering quality (PSNR) and
semantic segmentation quality (mIoU) within a very short
training time. Thanks to our unified rendering architecture,
the image rendering FPS is also fast. Additional results can
be found in our supplementary video.
Table 4. Quantitative comparison results in terms of Gaussian splat
count and rendering FPS.
Dataset
Mip-NeRF360
Tanks & Temples
Deep Blending
Replica
Method | Metrics
Count ↓
FPS ↑
Count ↓
FPS ↑
Count ↓
FPS ↑
Count ↓
FPS ↑
GeoGaussian [34]
126021
440.9
94755
441.1
53457
436.2
100877
423.6
3DGS [26]
504167
128.7
421167
136.9
285394
162.4
543581
137.5
Scaffold-GS [41]
301839
240.1
247081
249.1
182304
324.0
235446
346.2
Normal-GS [62]
318064
180.3
226283
178.7
185243
250.7
204358
180.0
Ours
161081
169.8
119947
163.1
48970
296.7
169273
197.8
4.4. Runtime and Memory Cost Analysis
In Tab. 4, we present a comparison of the Count of Gaus-
sians and rendering FPS against other methods. Our ap-
proach employs an effective differentiable pruning module,
which removes a significant number of redundant Gaus-
sians. As a result, the number of Gaussian splats in our
reconstructed scenes is the smallest. However, since our
method requires rendering multiple modalities simultane-
ously, the FPS experiences a certain degree of decrease.
Nevertheless, this does not severely hinder our rendering
speed. In some scenarios, our rendering speed remains su-
perior even compared to Normal-GS [62].
4.5. Ablation Study
Geometry-Aware Rasterization. We compare the perfor-
mance of depth rasterization that considers geometric infor-
mation with that of depth rasterization using only the cen-
Figure 7. Visualization of the distribution of the gradient fac-
tor. To enhance the visualization of the gradient factor, the bright-
ness of all Gaussians is uniformly minimized, resulting in blue
regions as rendered by SuperSplat [8]. (a) Scene with original
brightness. (b) Areas highlighted in yellow indicate regions where
the gradient factor k > 1.5. (c) Areas highlighted in yellow indi-
cate regions where k < 0.5.
ter points in terms of image quality. From the qualitative
results presented in Fig. 6, it can be observed that incorpo-
rating set information forces the Gaussian spheres to align
more closely with the actual surface of the objects. More
quantitative ablation experiments on each modules can be
found in our supplementary materials.
Differentiable Pruning. We conduct continuous optimiza-
tion of the gradient factor without pruning any Gaussians to
examine its distribution, as shown in Fig. 7. It can be ob-
served that regions with anomalous gradient factor, which
deviate significantly from the ideal value of 1, are typi-
cally areas lacking observational data or corresponding to
occluded edges. In these regions, due to insufficient super-
visory information from images, the gradients fail to con-
verge to the desired value. Such areas represent environ-
mental redundancy that can be removed to improve compu-
tational efficiency.
5. Conclusion
In this paper, we present UniGS, a unified and differentiable
framework for high-fidelity multimodal 3D reconstruction
using 3DGS. Our rasterization pipeline effectively renders
photo-realistic RGB images, accurate depth maps, consis-
tent surface normals, and semantic labels simultaneously.
By implementing differentiable ray-ellipsoid intersection,
we optimize rotation and scaling parameters with CUDA-
accelerated analytic gradient backpropagation, ensuring ge-
ometric consistency in surface. We also introduce a learn-
able attribute for efficient pruning of minimally contribut-
ing Gaussians. Extensive experiments demonstrate state-
of-the-art performance in terms of both reconstruction ac-
curacy across all modalities and storage efficiency.
8

<!-- page 9 -->
UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering
Supplementary Material
6. Methodology
In Sec. 6.1, we provide additional details on solving the
intersection points between pixel rays and Gaussians. In
Sec. 6.2, we focuse on deriving the analytical solution for
propagating depth gradients to the Gaussian parameters,
specifically addressing the transition from depth to rotation,
scale and positions. In Sec. 6.3, we include the implemen-
tation details that are omitted in Alg. 1. In Sec. 6.4, we
develop the derivation of the gradient propagation from the
cosine similarity loss of the normal estimation to the ren-
dered depth map, enabling the normal gradients to flow
back to depth gradient. This connection allows the nor-
mal gradients to be backpropagated further, thus linking the
normal consistency loss directly to the Gaussian parame-
ters. In Sec. 6.5, we expand the discussion on differentiable
pruning, with a particular focus on elucidating why k = 1
serves as the optimization objective. Finally, in Sec. 6.6, we
provide the implementation process for the backward ras-
terization gradient propagation corresponding to Alg. 2.
6.1. Ray-Eclipse Intersection Problem
The ray-eclipse intersection problem in (5) aim to solve for
t in the equation of a unit sphere
∥vs + tds∥2 = 1,
(15)
which simplifies to the quadratic equation:
at2 + bt + c = 0,
(16)
where the coefficients are defined as:
a = ds · ds,
b = 2vs · ds,
c = vs · vs −1.
(17)
The solutions are given by t = −b±
√
∆
2a
, where ∆= b2 −
4ac. This is a simple quadratic equation solving problem.
6.2. Backward Propagation for Depth Rendering
The gradient of the loss Ldepth with respect to the a single
Gaussian parameters θ ∈{µ, s, R} is propagated by
∂Ldepth
∂θ
= ∂Ldepth
∂˜D
· ∂˜D
∂di
· ∂di
∂θ ,
(18)
where ∂˜D
∂di can be calculated from (7). The primary chal-
lenge is to compute the gradients of di with respect to the
Gaussian parameters µ, s, and R, which requires backprop-
agating through the ray-ellipsoid intersection. The gradient
flow is structured as
∂di
∂θ = ∂di
∂tmid
·
∂tmid
∂(a, b, c) · ∂(a, b, c)
∂(vs, ds)
· ∂(vs, ds)
∂(vl, dl) · ∂(vl, dl)
∂θ
.
(19)
The key steps in this chain will be derived below.
Gradient of Depth w.r.t. tmid. The depth di of the intersec-
tion point is a function of the midpoint ray parameter tmid.
The gradient is computed as the dot product of the ray di-
rection d with the z-axis of the camera view matrix Rcam
world:
∂di
∂tmid
= Rcam
world[2, 0] · d →x + Rcam
world[2, 1] · d →y
+ Rcam
world[2, 2] · d →z.
(20)
Gradient of tmid w.r.t. Quadratic Coefficients. The mid-
point parameter tmid = (t1 +t2)/2 = −b/(2a) is a function
of the quadratic coefficients a and b. Its derivatives are
∂tmid
∂a
=
b
2a2 , ∂tmid
∂b
= −1
2a, ∂tmid
∂c
= 0.
(21)
Gradients of Coefficients w.r.t. Local Coordinates. The
coefficients a, b, and c are functions of the scaled local ray
origin vs and direction ds from (17). Their gradients are
∂a
∂ds
= 2ds, ∂b
∂vs
= 2ds,
∂b
∂ds
= 2vs, ∂c
∂vs
= 2vs.
(22)
Applying the chain rule through tmid and the coefficients,
the gradients w.r.t. the local (l) scaled (s) coordinates are:
∂tmid
∂vs
= ∂tmid
∂b
· ∂b
∂vs
+ ∂tmid
∂c
· ∂c
∂vs
= −1
a · ds,
∂tmid
∂ds
= ∂tmid
∂a
· ∂a
∂ds
+ ∂tmid
∂b
· ∂b
∂ds
= b
a2 · ds −1
a · vs.
(23)
Gradients w.r.t. Scale s. The local (l) scaled (s) coordi-
nates are defined as vs = vl ⊘s and ds = dl ⊘s, where ⊘
denotes element-wise division. The gradients with respect
to the scale s are computed as:
∂(vs, ds)
∂s
= −vl,k ⊘s −dl,k ⊘s.
(24)
Gradients w.r.t. Position µ and Rotation R. The local co-
ordinates before scaling are obtained by rotating the world-
space vectors into the Gaussian’s local frame: vl = R⊤v
and dl = R⊤d, where v = o −µ. The gradient with
1

<!-- page 10 -->
respect to the position µ is
∂di
∂µ = −R
 ∂di
∂vl
⊘s

.
(25)
The gradient with respect to the rotation matrix R is accu-
mulated from both v and d:
∂di
∂R =
 ∂di
∂vl
⊘s

v⊤+
 ∂di
∂dl
⊘s

d⊤.
(26)
Finally, the gradient
∂di
∂R is converted to a gradient
∂R
∂q
with respect to the quaternion q using the standard adjoint-
sensitive conversion, ensuring the constraint of staying on
the S3 manifold through tangent space projection.
6.3. Normal Estimation from Depth
Following the rendering of the depth map ˜D, we estimate a
corresponding surface normal map ˜N in a post-processing
step. In this supplementary section, we will elaborate on
two key operations of Alg. 1: backprojecting depth pixels
and orienting towards the view direction.
Depth Backprojection to 3D Points. Each pixel p = (u, v)
with depth value ˜D(u, v) is backprojected to a 3D point
Pcam in the camera coordinate system using the intrinsic
camera parameters:
Pworld =
Rworld
cam
tworld
cam
0⊤
1



(u−cx)
fx
· ˜D(u, v)
(v−cy)
fy
· ˜D(u, v)
˜D(u, v)
1

,
(27)
where cx = W/2, cy = H/2 are the coordinates of the
principal point, and fx, fy are the focal lengths.
View Direction Correction. Finally, the normal’s orienta-
tion is corrected to face the camera. The normal is flipped
if it points away from the camera, determined by the sign
of the dot product between the normalized view direction
dview and the normal:
˜N(u, v) ←
( ˜N(u, v),
if ˜N(u, v) · dview ≤0
−˜N(u, v),
otherwise
,
(28)
where dview =
C−Pworld
∥C−Pworld∥is the normalized vector from the
3D point Pworld to the camera center C. This ensures all
normals adhere to a consistent front-facing convention.
6.4. Backward Propagation from Normals to Depth
To enable optimization via losses defined on the rendered
normal map ˜N, the gradients must be backpropagated
through the normal estimation pipeline to the depth map ˜D
and subsequently to the underlying 3D Gaussian parame-
ters. Overall gradient backpropagation pipeline is shown
in Alg. 3. The gradient of the loss Lnormal with respect to
Algorithm 3: Backward Propagation from Nor-
mals to Depth
1 Input: ∂Lnormal
∂˜N
, STEP 1, STEP 2, λ
2 Output: ∂Lnormal
∂˜D
3 for each pixel p = (u, v) in parallel do
4
1. View-Facing Orientation
Flip (32)←∂Lnormal
∂˜N(u,v)
5
2. Grad Normalization (33)
6
3. Grad Fusion (34)←λ
7
4. Compute Multi-Scale Gradient
8
for STEP X ∈{STEP 1, STEP 2} do
9
Gradient Atom Addition (35) (36)
10
5. Backprojection Gradient (37)
11
∂Lnormal
∂˜D(u,v) ←(29)
the depth map ˜D is propagated by
∂Lnormal
∂˜D(u, v)
= ∂Lnormal
∂˜N(u, v)
· ∂˜N(u, v)
∂n(orient)
fused
· ∂n(orient)
fused
∂nfused
· ∂nfused
∂(n1, n2) ·

∂n1
∂Pn1
world
+
∂n2
∂Pn2
world

· ∂Pworld
∂˜D(u, v)
,
(29)
where Pn1
world and Pn2
world represent the sets of 3D points in
the neighborhoods used to compute n1 and n2, respectively.
Cosine Similarity w.r.t. Normal. For encouraging geo-
metric consistency (13) is the negative cosine similarity be-
tween the predicted normal ˜N and a target normal ˜Ntarget:
Lnormal = 1 −1
|Ω|
X
(u,v)∈Ω
˜N(u, v) · ˜Ntarget(u, v),
(30)
where Ωdenotes the set of pixels. The gradient of this loss
with respect to the predicted normal at a specific pixel is
straightforward:
∂Lnormal
∂˜N(u, v)
= −1
|Ω|
˜Ntarget(u, v).
(31)
This gradient ∂Lnormal
∂˜N
is the seed that initiates the back-
ward pass through the normal estimation pipeline.
View-facing Orientation Flip. If the normal is flipped in
the forward pass to face the viewer (defined by (28)), the
incoming gradient is correspondingly flipped:
∂Lnormal
∂n(orient)
fused
=
(
−∂Lnormal
∂˜N(u,v) ,
if flipped
∂Lnormal
∂˜N(u,v) ,
otherwise ,
(32)
Normalization.
The gradient is then backpropagated
through the normalization operation ˜N = nfused/∥nfused∥.
The gradient w.r.t. the unnormalized vector nfused is given
by:
∂n(orient)
fused
∂nfused
=

I −˜N(u, v) ˜N(u, v)⊤
∥nfused∥
.
(33)
Normal Fusion. The gradient is distributed to the two con-
2

<!-- page 11 -->
stituent normal estimates n1 and n2 from which nfused =
λn1 + (1 −λ)n2 is computed. If their directions were in-
consistent in the forward pass (n1 · n2 < 0), the sign of the
gradient for n2 is flipped:
∂nfused
∂n1
= λ, ∂nfused
∂n2
= (1 −λ) · sign(n1 · n2).
(34)
Finite Differences to 3D Points. The core of the back-
ward pass involves computing how the estimated normals
n1 and n2 change with respect to the 3D points Pworld in
their finite-difference neighborhood. For simplicity, we will
take nX ∈{n1, n2}, STEP X ∈{STEP 1, STEP 2} as an
example for the subsequent derivation. For a normal esti-
mate nX calculated from points P in a STEP X × STEP X
window, the gradient w.r.t. each involved point PnX
world is
derived from the cross product operation.
From Alg. 1, vnX
x
=
PnX
world(u + STEP X, v) −
PnX
world(u−STEP X, v) and vnX
y
= PnX
world(u, v+STEP X)−
PnX
world(u, v −STEP X)), the normal is nX = vnX
x
× vnX
y .
The gradients of nX w.r.t. the vectors vnX
x
and vnX
y
are:
∂nX
∂vnX
x
= vnX
y , ∂nX
∂vnX
y
= vnX
x .
(35)
These gradients are then distributed to the specific points
that contributed to vnX
x
and vnX
y :
∂nX
∂PnX
world(u + STEP X, v) += vnX
x ,
∂nX
∂PnX
world(u −STEP X, v) −= vnX
x ,
∂nX
∂PnX
world(u, v + STEP X) += vnX
y ,
∂nX
∂PnX
world(u, v −STEP X) −= vnX
y .
(36)
This process is repeated for both normal estimates n1 and
n2 and their respective neighborhoods Pn1
world and Pn2
world,
and the gradients are accumulated atomically.
3D Points w.r.t Depth. The Jacobian matrix
∂Pworld
∂˜D(u,v) is
derived from the backprojection and transformation chain
Pworld = Rworld
cam Pcam( ˜D(u, v)):
∂Pworld
∂˜D(u, v)
= Rworld
cam
∂Pcam
∂˜D(u, v)
= Rworld
cam


(u −cx)/fx
(v −cy)/fy
1
0

.
(37)
6.5. Differentiable Pruning
Gradient Propagation. During backpropagation of k, the
gradient can be propagated to the gradient factor k through
the rendering equation. The chain rule for the gradient of
the loss with respect to a specific ki can be conceptually
expressed as
∂LK(u, v)
∂ki
= ∂LK(u, v)
∂˜K(u, v)
· ∂˜K(u, v)
∂ki
,
(38)
where ∂˜K(u,v)
∂ki
can be derived from (8) in main text.
Detailed Analysis of optimal state k = 1. The gradient
factor k serves as a core indicator for Gaussian pruning. Its
deviation from 1 directly reflects the optimization state of
Gaussians, and the underlying causes along with pruning
rationale are elaborated as follows.
• Gaussians with k > 1: Anomalies from Gradient Fluc-
tuations Caused by Data Noise. Gaussians with extremely
large k-values are induced by gradient anomalies during
multi-view optimization. During the rendering of specific
views, data noise such as sensor noise and texture incon-
sistencies at scene edges leads to significant fluctuations in
pixel gradients. These anomalous gradients propagate to the
opacity α through backpropagation, causing α to deviate
from its optimal value. This deviation results in unexpected
optimization under specific view. Such abnormal k-values
are usually accompanied by extreme Gaussian parameters
including position, scale as shown in Fig. 7 in the main text.
• Gaussians with k < 1: Redundancy Due to Insuffi-
cient Rendering Participation. Gaussians with small k-
values can be identified based on long-term optimization,
especially considering the periodic k-reset strategy. During
training, we periodically reset the k-value of all Gaussians
to 0.9. This initial value is designed to provide a starting
point for optimization driven by LK. The expectation is
that after multiple optimization iterations, well-optimizing
Gaussians will adjust their k-values from 0.9 toward 1. The
ideal state is when k = 1, which makes the rendering for-
mula of ˜K satisfy
˜K(u, v) =
X
i∈N
αi · 1 ·
i−1
Y
j=1
(1 −αj) = 1.
(39)
If a Gaussian’s k-value remains near 0.9 or changes min-
imally after a long optimization horizon, it indicates the
Gaussian has not deeply participated in the rendering of
any view. Specifically, it lacks sufficient effective alpha-
rendering contributions to pixels across views. This situa-
tion will occur because the Gaussian is located in unobserv-
able regions such as persistent occlusions. Such Gaussians
are redundant, and their exclusion does not affect the accu-
racy of scene reconstruction as they contribute little to the
rendering of ˜K or RGB images.
Pruning Mechanism Based on k.
In summary, well-
optimized Gaussians can adjust their k-values toward 1 un-
der the constraint of LK, and their α and geometric param-
eters can well match the scene without the need for extreme
k-compensation. In contrast, Gaussians with |k −1| > Tk
are either anomalous with k > 1 or redundant with k < 1.
By pruning these Gaussians, we can effectively reduce the
3

<!-- page 12 -->
total number of Gaussians while preserving the quality of
scene reconstruction, thereby improving the overall compu-
tational efficiency.
Algorithm 4: Tile-based Backward Gradient Prop-
agation Pipeline
Input: Forward rendering results: ˜D, ˜N, ˜O, ˜K;
Loss gradients: ∂Ldepth
∂˜D
, ∂Lnormal
∂˜N
, ∂Lseg
∂˜O , ∂LK
∂˜K
Output: Parameter gradients: ∂L
∂µ, ∂L
∂R, ∂L
∂s , ∂L
∂o , ∂L
∂K
1 for each pixel (x, y) do in parallel
// Restore Saved Forward Values
2
Restore: T ←Tfinal;
3
for each Gaussian i intersecting pixel pix (in
reverse depth order) do
4
...
5
T ←T/(1 −αi) ;
6
wα ←αi · T;
// Compute Gradients for
Rendered Attributes
7
for each ch in Co do
8
∂Lall
∂oi[ch] ←∂Lseg
∂˜O[ch]
| {z }
(12)
·wα;
9
∂Lall
∂ki
←∂LK
∂˜K
| {z }
(38)
·wα;
// Depth Gradient Propagation
10
∂Lall
∂di
←(∂Lnormal
∂˜N
|
{z
}
(31)
+ ∂Ldepth
∂˜D
) · wα;
// Propagate Depth Gradient
to Ellipsoid Parameters
11
∂Lall
∂s
= ∂Lall
∂di · ∂di
∂s
|{z}
(24)
;
12
∂Lall
∂µ
= ∂Lall
∂di · ∂di
∂µ
|{z}
(25)
;
13
∂Lall
∂q
= ∂Lall
∂di · ∂di
∂R
|{z}
(26)
· ∂R
∂q ;
14
...
6.6. Tile-Based Gradient Backpropagation
In the previous sections, we obtain the losses for the se-
mantic logits, depth, normal, and gradient factor map with
respect to their respective ground truths, which are given by
∂L
∂˜O, ∂L
∂˜D, ∂L
∂˜N, and ∂L
∂˜K. Corresponding to Alg. 2, we opti-
mize the Gaussian parameters through the backward prop-
agation of gradients via the alpha rendering process. The
backward propagation pipeline is illustrated in Alg. 4. We
omit the gradient propagation process for RGB (indicated
as ...), which is consistent with the original implementa-
tion [26].
7. More Experiments
7.1. RGB Rendering Evaluation
To validate the effectiveness of our method in providing
valuable information for NVS, we compare its performance
on novel view RGB rendering with state-of-the-art frame-
works [1] [45] [74] [26] [41] [34]. All methods utilize their
provided original code and unified original data. The quali-
tative results of the RGB rendering are shown in Fig. 8, and
the quantitative comparisons are presented in Tab. 5. It can
be observed that our method achieves state-of-the-art re-
sults in novel view synthesis for certain scenes. By incorpo-
rating geometric constraints into the framework, our algo-
rithm achieves SOTA geometric performance while main-
taining RGB reconstruction quality and significantly im-
proving memory and rendering efficiency.
7.2. Statistical Outlier Analysis
To quantify the quality of predicted point clouds relative
to ground truth data in Replica [53], we employ a statis-
tical approach based on Z-score analysis of point-to-point
distances. This method allows us to identify and measure
outliers in predicted point clouds with respect to the ground
truth mesh points.
Distance. For each point pi in the predicted point cloud P,
we compute the minimum Euclidean distance to the ground
truth point cloud G:
di = min
g∈G |pi −g|,
(40)
where | · | denotes the Euclidean norm. This computation is
efficiently implemented using a KD-tree data structure for
nearest neighbor search.
Statistical Measures.
From the computed distances
d1, d2, . . . , dn where n is the number of points in the pre-
dicted point cloud, we calculate the following statistical
measures:
µd = 1
n
n
X
i=1
di,
(41)
σd =
v
u
u
t 1
n
n
X
i=1
(di −µd)2,
(42)
where µd represents the Mean distance and σd represents
the standard deviation (Std) of distances.
For each distance di, we compute its Z-score, which
measures how many standard deviations the distance is
4

<!-- page 13 -->
Table 5. Quantitative comparison of RGB rendering on widely used real and simulated datasets. Some metrics (∗) are extracted from the
original data in respective papers. The results ranked from best to worst are highlighted as first , second , and third .
Dataset
Mip-NeRF360
Tanks & Temples
Deep Blending
Replica
Method | Metrics
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
iNPG* [45]
26.43
0.725
0.339
21.72
0.723
0.330
23.62
0.797
0.423
-
-
-
Mip-NeRF* [1]
29.23
0.844
0.207
22.22
0.759
0.257
29.40
0.901
0.245
-
-
-
RaySplats* [3]
27.31
0.846
0.237
22.20
0.829
0.202
29.57
0.900
0.320
-
-
-
GeoGaussian [34]
26.26
0.813
0.219
20.00
0.802
0.261
26.07
0.873
0.310
24.55
0.946
0.114
3DGS [26]
28.69
0.870
0.182
23.14
0.841
0.183
29.41
0.903
0.243
36.50
0.940
0.120
Scaffold-GS [41]
28.84
0.848
0.220
23.96
0.853
0.177
30.21
0.906
0.254
37.45
0.951
0.090
2DGS [18]
27.34
0.814
0.234
22.89
0.833
0.206
27.96
0.873
0.281
36.47
0.943
0.107
Normal-GS [62]
29.34
0.869
0.194
24.22
0.854
0.174
30.19
0.910
0.252
38.19
0.977
0.043
Ours
30.10
0.887
0.148
23.57
0.860
0.168
29.66
0.899
0.231
36.68
0.962
0.083
Figure 8. Qualitative comparison results on NVS demonstrate that our RGB rendering achieves better performance in certain scenarios.
The top-left corner of the image displays the PSNR value computed against the ground truth image.
from the mean:
zi = |di −µd|
σd
.
(43)
Points are classified as outliers based on a threshold param-
eter τ (τ = 0.1 in our experiments):
Fi =
1
if zi > τ
0
otherwise .
(44)
The outlier ratio (Radio) is then calculated as the proportion
of points classified as outliers:
Radio = 1
n
n
X
i=1
Fi.
(45)
Hausdorff Distance. In addition to the statistical outlier
analysis, we compute the Hausdorff distance to measure the
5

<!-- page 14 -->
Table 6. Quantitative comparisons of the ablation experiments on each module.
Datasets
Room0-Sequence1
Office3-Sequence3
Settings | Metrics
PSNR↑
Abs.Rel.↓
CosSimi↑
mIoU↑
Count↓
FPS↑
PSNR↑
Abs.Rel.↓
CosSimi↑
mIoU↑
Count↓
FPS↑
RayCasting-Point (All)
31.341
0.0062
0.855
0.607
170859
183.8
36.892
0.0208
0.931
0.534
178905
184.8
Center Point
31.800
0.0071
0.849
0.607
174177
181.7
36.873
0.0204
0.932
0.534
179128
187.8
w/o prune
31.509
0.0070
0.851
0.606
229487
156.3
37.100
0.0315
0.928
0.534
237914
165.5
w/prune Tk=0.2
31.273
0.0068
0.847
0.605
144185
204.1
36.925
0.0161
0.932
0.534
148167
189.7
w/o seg
31.605
0.0226
0.837
0.021
174230
188.0
36.833
0.0225
0.929
0.024
184952
168.9
w/o normal
32.750
0.0061
0.774
0.608
167494
206.5
37.488
0.00885
0.895
0.535
182019
203.8
w/o depth
32.161
0.0184
0.850
0.607
171462
189.7
37.318
0.0985
0.925
0.539
176346
184.1
w/o depth&normal
33.691
0.0214
0.658
0.609
167198
202.6
37.449
0.1106
0.782
0.539
180747
189.7
maximum deviation between point clouds:
H(G, P) = max

sup
g∈G
inf
p∈P |g −p|, sup
p∈P
inf
g∈G |p −g|

.
(46)
7.3. More Results on Geometric Rendering
We present additional comparative results of depth and sur-
face normal predictions across multiple datasets in Fig. 9,
which demonstrate the superiority of our method in geomet-
ric rendering. Particularly in texture-sparse regions, where
other methods fail to accurately reconstruct object surfaces,
our approach consistently ensures high consistency among
RGB, depth, and normal estimations.
It is worth noting that in the Mip-NeRF [1], Tanks &
Temples [28], and Deep Blending [16] datasets, the depth
and normal information used for supervision is provided by
Normal-GS [62] and may still be inaccurate in certain sce-
narios. Nevertheless, our method achieves competitive re-
sults in both depth and normal estimation.
7.4. Ablation on Algorithm Components
We test the quantitative rendering quality and runtime anal-
ysis with and without the differentiable pruning module, as
shown in Tab. 6. It is evident that our differentiable pruning
module effectively reduces the number of Gaussians used
without significantly compromising image rendering qual-
ity. thereby improving rendering FPS and reducing stray
points in the infinite distance.
Furthermore, Tab. 6 demonstrates the performance gains
achieved by our ray-casting based method on depth and nor-
mal estimation, as well as the impact of integrating various
loss functions on their respective evaluation metrics. It can
be observed that our ray-casting based approach enhances
the algorithm’s performance on geometric metrics.
Unlike methods that optimize for a single modality, our
focus lies in achieving a consistent representation of all
modalities in the scene. We therefore strive for a balanced
performance across all metrics, which ultimately leads to a
more faithful reconstruction of the real world.
8. Multimodal Viewer
We also extend a visualization rendering software based on
the SIBR viewer and a CUDA rendering pipeline we devel-
oped. This tool supports real-time visualization of RGB,
depth, normal, and semantic maps, achieving a rendering
speed of up to 200 FPS. The corresponding viewer will also
been open-sourced on GitHub.
9. Limitations
Real-world environments exhibit complex optical interac-
tions including extensive light reflections and inter reflec-
tions. Image captured by a camera is the product of multi-
path light propagation and global illumination effects. Our
current approach does not explicitly model these radiomet-
ric phenomena such as light reflection and composition.
Furthermore, this work focuses exclusively on supervised
offline reconstruction of static scenes. Extending the frame-
work to incorporate temporal coherence for dynamic scene
modeling remains a direction for future work.
References
[1] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded
Anti-Aliased Neural Radiance Fields. In CVPR, pages 5470–
5479, 2022. 1, 2, 6, 7, 4, 5
[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman.
Zip-NeRF: Anti-Aliased
Grid-Based Neural Radiance Fields. In ICCV, pages 19697–
19705, 2023. 1, 2
[3] Krzysztof Byrski, Marcin Mazur, Jacek Tabor, Tadeusz
Dziarmaga, Marcin Kadziolka, Dawid Baran, and Prze-
mysław Spurek.
RaySplats: Ray Tracing based Gaussian
Splatting. arXiv preprint arXiv:2501.19196, 2025. 5
[4] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi,
and Jianfei Cai. HAC: Hash-Grid Assisted Context for 3D
Gaussian Splatting Compression. In ECCV, pages 422–438,
2024. 2
[5] Youyu Chen, Junjun Jiang, Kui Jiang, Xiao Tang, Zhihao
Li, Xianming Liu, and Yinyu Nie. DashGaussian: Optimiz-
ing 3D Gaussian Splatting in 200 Seconds. In CVPR, pages
11146–11155, 2025. 2
[6] Zi-Ting Chou, Sheng-Yu Huang, I Liu, Yu-Chiang Frank
Wang, et al. GSNeRF: Generalizable Semantic Neural Ra-
6

<!-- page 15 -->
Figure 9. Qualitative comparison of depth and normal estimation accuracy across different methods on several datasets.
diance Fields with Enhanced 3D Scene Understanding. In
CVPR, pages 20806–20815, 2024. 3
[7] PyTorch Contributors. Installing C++ Distributions of Py-
Torch, 2024. 6
[8] SuperSplat Contributors.
SuperSplat - 3D Gaussian Splat
Editor, 2025. 8
[9] Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu,
Huamin Wang, and Weiwei Xu. High-Quality Surface Re-
construction Using Gaussian Surfels. In ACM SIGGRAPH,
pages 1–11, 2024. 3
[10] Yinan Deng, Yufeng Yue, Jianyu Dou, Jingyu Zhao, Jiahui
Wang, Yujie Tang, Yi Yang, and Mengyin Fu. OmniMap: A
General Mapping Framework Integrating Optics, Geometry,
and Semantics. IEEE TRO, 2025. 2, 3
[11] Jiahua Dong and Yu-Xiong Wang. 3DGS-Drag: Dragging
Gaussians for Intuitive Point-Based 3D Editing. In ICLR,
2025. 3
[12] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia
Xu, Zhangyang Wang, et al. LightGaussian: Unbounded 3D
Gaussian Compression with 15x Reduction and 200+ FPS.
7

<!-- page 16 -->
NeurIPS, 37:140138–140158, 2024. 2
[13] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun
Cao, Li Zhang, and Yao Yao. Relightable 3D Gaussians:
Realistic Point Cloud Relighting with BRDF Decomposition
and Ray Tracing. In ECCV, pages 73–89, 2024. 3
[14] Antoine Gu´edon and Vincent Lepetit.
SUGAR: Surface-
Aligned Gaussian Splatting for Efficient 3D Mesh Recon-
struction and High-Quality Mesh Rendering.
In CVPR,
pages 5354–5363, 2024. 3
[15] Jun Guo, Xiaojian Ma, Yue Fan, Huaping Liu, and Qing
Li.
Semantic Gaussians:
Open-Vocabulary Scene Un-
derstanding with 3D Gaussian Splatting.
arXiv preprint
arXiv:2403.15624, 2024. 2, 3, 7
[16] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm,
George Drettakis, and Gabriel Brostow. Deep Blending for
Free-Viewpoint Image-Based Rendering. ACM SIGGRAPH,
37(6):1–15, 2018. 6
[17] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2D Gaussian Splatting for Geometrically Ac-
curate Radiance Fields. ACM SIGGRAPH, 2024. 3
[18] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shengha Gao. 2D Gaussian Splatting for Geometrically Ac-
curate Radiance Fields. In ACM SIGGRAPH, pages 1–11,
2024. 2, 6, 7, 5
[19] Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou,
and Jiwen Lu.
GaussianFormer: Scene as Gaussians for
Vision-Based 3D Semantic Occupancy Prediction. In ECCV,
pages 376–393, 2024. 1
[20] Yuanhui Huang, Amonnut Thammatadatrakoon, Wenzhao
Zheng,
Yunpeng Zhang,
Dalong Du,
and Jiwen Lu.
GaussianFormer-2: Probabilistic Gaussian Superposition for
Efficient 3D Occupancy Prediction. In CVPR, pages 27477–
27486, 2025. 1
[21] Hanqing Jiang, Xiaojun Xiang, Han Sun, Hongjie Li, Liyang
Zhou, Xiaoyu Zhang, and Guofeng Zhang.
GeoTexDen-
sifier:
Geometry-Texture-Aware Densification for High-
Quality Photorealistic 3D Gaussian Splatting. arXiv preprint
arXiv:2412.16809, 2024. 2
[22] Minchao Jiang, Shunyu Jia, Jiaming Gu, Xiaoyuan Lu,
Guangming Zhu, Anqi Dong, and Liang Zhang. VoteSplat:
Hough Voting Gaussian Splatting for 3D Scene Understand-
ing. arXiv preprint arXiv:2506.22799, 2025. 3
[23] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaox-
iao Long, Wenping Wang, and Yuexin Ma. GaussianShader:
3D Gaussian Splatting with Shading Functions for Reflective
Surfaces. In CVPR, pages 5322–5332, 2024. 3, 7
[24] Rui Jin, Yuman Gao, Yingjian Wang, Yuze Wu, Haojian Lu,
Chao Xu, and Fei Gao. GS-Planner: A Gaussian-Splatting-
Based Planning Framework for Active High-Fidelity Recon-
struction. In IROS, pages 11202–11209, 2024. 3
[25] Nikhil Varma Keetha, Jay Karhade, Krishna Murthy Jataval-
labhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan,
and Jonathon Luiten.
SplaTAM: Splat, Track & Map 3D
Gaussians for Dense RGB-D SLAM. CVPR, pages 21357–
21366, 2023. 3
[26] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3D Gaussian Splatting for Real-Time
Radiance Field Rendering. ACM TOG, 42(4):139–1, 2023.
1, 2, 3, 5, 6, 7, 8, 4
[27] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas,
Michael Wimmer, Alexandre Lanvin, and George Drettakis.
A Hierarchical 3D Gaussian Representation for Real-time
Rendering of Very Large Datasets. ACM TOG, 43(4):1–15,
2024. 1, 2
[28] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun.
Tanks and Temples: Benchmarking Large-Scale
Scene Reconstruction. ACM TOG, 36(4):1–13, 2017. 6
[29] Zihang Lai. Exploring Simple Open-Vocabulary Semantic
Segmentation. In CVPR, pages 30221–30230, 2025. 3
[30] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko,
and Eunbyung Park. Compact 3D Gaussian Representation
for Radiance Field. In CVPR, pages 21719–21728, 2024. 2
[31] Xiaohan Lei, Min Wang, Wengang Zhou, and Houqiang Li.
GaussNav: Gaussian Splatting for Visual Navigation. IEEE
TPAMI, 2025. 3
[32] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen
Koltun, and Ren´e Ranftl. Language-Driven Semantic Seg-
mentation. arXiv preprint arXiv:2201.03546, 2022. 3
[33] Mingrui Li, Shuhong Liu, Heng Zhou, Guohao Zhu, Na
Cheng, Tianchen Deng, and Hongyu Wang. SGS-SLAM:
Semantic Gaussian Splatting for Neural Dense SLAM. In
ECCV, pages 163–179, 2024. 2, 3, 7
[34] Yanyan Li, Chenyu Lyu, Yan Di, Guangyao Zhai, Gim Hee
Lee, and Federico Tombari. GeoGaussian: Geometry-Aware
Gaussian Splatting for Scene Rendering. In ECCV, pages
441–457, 2024. 2, 3, 6, 7, 8, 4, 5
[35] Yue Li, Qi Ma, Runyi Yang, Huapeng Li, Mengjiao Ma, Bin
Ren, Nikola Popovic, Nicu Sebe, Ender Konukoglu, Theo
Gevers, et al. SceneSplat: Gaussian Splatting-Based Scene
Understanding with Vision-Language Pretraining.
arXiv
preprint arXiv:2503.18052, 2025. 2
[36] Zhaoshuo Li, Thomas M¨uller, Alex Evans, Russell H Taylor,
Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. Neu-
ralangelo: High-Fidelity Neural Surface Reconstruction. In
CVPR, pages 8456–8455, 2023. 3
[37] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan
Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana
Marculescu. Open-Vocabulary Semantic Segmentation with
Mask-Adapted CLIP. In CVPR, pages 7061–7070, 2023. 3
[38] Yang Liu, Chuanchen Luo, Lue Fan, Naiyan Wang, Jun-
ran Peng, and Zhaoxiang Zhang. CityGaussian: Real-Time
High-Quality Large-Scale Scene Rendering with Gaussians.
In ECCV, pages 265–282, 2024. 2
[39] Yang Liu, Chuanchen Luo, Zhongkai Mao, Junran Peng, and
Zhaoxiang Zhang. CityGaussianV2: Efficient and Geometri-
cally Accurate Reconstruction for Large-Scale Scenes. arXiv
preprint arXiv:2411.00771, 2024. 2
[40] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu, Ji-
wen Lu, and Yansong Tang. ManiGaussian: Dynamic Gaus-
sian Splatting for Multi-Task Robotic Manipulation.
In
ECCV, pages 349–366. Springer, 2024. 1
[41] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai.
Scaffold-GS: Structured
3D Gaussians for View-adaptive Rendering. In CVPR, pages
20654–20664, 2024. 2, 3, 5, 6, 7, 8, 4
8

<!-- page 17 -->
[42] Huaishao Luo, Junwei Bao, Youzheng Wu, Xiaodong He,
and Tianrui Li. SegCLIP: Patch Aggregation with Learn-
able Centers for Open-Vocabulary Semantic Segmentation.
In ICML, pages 23033–23044, 2023. 3
[43] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
Representing Scenes as Neural Radiance Fields for View
Synthesis. In ECCV, 2020. 1, 2
[44] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Ric-
cardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja
Fidler, Nicholas Sharp, and Zan Gojcic. 3D Gaussian Ray
Tracing: Fast Tracing of Particle Scenes. ACM SIGGRAPH
Asia, 2024. 3
[45] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant Neural Graphics Primitives with a Mul-
tiresolution Hash Encoding. ACM TOG, 41(4):1–15, 2022.
2, 7, 4, 5
[46] Simon Niedermayr, Josef Stumpfegger, and R¨udiger Wester-
mann. Compressed 3D Gaussian Splatting for Accelerated
Novel View Synthesis. In CVPR, pages 10349–10358, 2024.
2
[47] Michael Oechsle, Songyou Peng, and Andreas Geiger.
UniSurf: Unifying Neural Implicit Surfaces and Radiance
Fields for Multi-View Reconstruction. In ICCV, pages 5589–
5599, 2021. 3
[48] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and
Hanspeter Pfister. LangSplat: 3D Language Gaussian Splat-
ting. In CVPR, pages 20051–20060, 2024. 3
[49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever. Learning Transferable Visual
Models From Natural Language Supervision.
In ICML,
pages 8748–8763, 2021. 3
[50] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-GS: Towards Consistent
Real-Time Rendering with LOD-Structured 3D Gaussians.
arXiv preprint arXiv:2403.17898, 2024. 2
[51] Ji Shi, Xianghua Ying, Ruohao Guo, Bowei Xing, and Wen-
zhen Yue. Normal-NeRF: Ambiguity-Robust Normal Esti-
mation for Highly Reflective Scenes. In AAAI, pages 6869–
6877, 2025. 2, 3
[52] Yahao Shi, Yanmin Wu, Chenming Wu, Xing Liu, Chen
Zhao, Haocheng Feng, Jian Zhang, Bin Zhou, Errui Ding,
and Jingdong Wang. GIR: 3D Gaussian Inverse Rendering
for Relightable Scene Factorization. IEEE TPAMI, 2025. 3
[53] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl
Ren, Shobhit Verma, et al. The Replica Dataset: A Digital
Replica of Indoor Spaces. arXiv preprint arXiv:1906.05797,
2019. 7, 4
[54] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Prad-
han, Ben P. Mildenhall, Pratul Srinivasan, Jonathan T. Bar-
ron, and Henrik Kretzschmar. Block-NeRF: Scalable Large
Scene Neural View Synthesis. In CVPR, pages 8238–8248,
2022. 2
[55] Shengji Tang, Weicai Ye, Peng Ye, Weihao Lin, Yang Zhou,
Tao Chen, and Wanli Ouyang.
HiSplat: Hierarchical 3D
Gaussian Splatting for Generalizable Sparse-View Recon-
struction. ICLR, 2025. 2, 7
[56] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler,
Jonathan T Barron, and Pratul P Srinivasan.
Ref-NeRF:
Structured View-Dependent Appearance for Neural Radi-
ance Fields. In CVPR, pages 5481–5490, 2022. 1
[57] Evangelos Ververas, Rolandos Alexandros Potamias, Jifei
Song, Jiankang Deng, and Stefanos Zafeiriou.
SAGS:
Structure-Aware 3D Gaussian Splatting.
In ECCV, pages
221–238, 2024. 3
[58] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku
Komura, and Wenping Wang. NeuS: Learning Neural Im-
plicit Surfaces by Volume Rendering for Multi-View Recon-
struction. arXiv preprint arXiv:2106.10689, 2021. 3
[59] Peihao Wang, Yuehao Wang, Dilin Wang, Sreyas Mo-
han, Zhiwen Fan, Lemeng Wu, Ruisi Cai, Yu-Ying Yeh,
Zhangyang Wang, Qiang Liu, et al. Steepest Descent Den-
sity Control for Compact 3D Gaussian Splatting. In CVPR,
pages 26663–26672, 2025. 2
[60] Yiming Wang, Qin Han, Marc Habermann, Kostas Dani-
ilidis, Christian Theobalt, and Lingjie Liu.
NeuS2: Fast
Learning of Neural Implicit Surfaces for Multi-View Recon-
struction. In ICCV, pages 3295–3306, 2023. 3
[61] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P
Simoncelli. Image Quality Sssessment: from Error Visibility
to Structural Similarity. IEEE TIP, 13(4):600–612, 2004. 6,
7
[62] Meng Wei, Qianyi Wu, Jianmin Zheng, Hamid Rezatofighi,
and
Jianfei
Cai.
Normal-GS:
3D
Gaussian
Splat-
ting with Normal-Involved Rendering.
arXiv preprint
arXiv:2410.20593, 2024. 2, 3, 6, 7, 8, 5
[63] Tianci Wen, Zhiang Liu, and Yongchun Fang. SEGS-SLAM:
Structure-Enhanced 3D Gaussian Splatting SLAM with Ap-
pearance Embedding.
arXiv preprint arXiv:2501.05242,
2025. 3
[64] Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas
Moenne-Loccoz, and Zan Gojcic. 3DGUT: Enabling Dis-
torted Cameras and Secondary Rays in Gaussian Splatting.
CVPR, pages 26036–26046, 2025. 3
[65] Tong Wu, Jia-Mu Sun, Yu-Kun Lai, Yuewen Ma, Leif
Kobbelt, and Lin Gao. DeferredGS: Decoupled and Editable
Gaussian Splatting with Deferred Shading. arXiv preprint
arXiv:2404.09412, 2024. 3
[66] Yusen Xie, Zhenmin Huang, Jin Wu, and Jun Ma. GS-LIVM:
Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping
with Gaussian Splatting. ICCV, 2025. 2
[67] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang,
Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou,
and Sida Peng. Street Gaussians: Modeling Dynamic Urban
Scenes with Gaussian Splatting. In ECCV, pages 156–173,
2024. 1, 2
[68] Runyi Yang, Zhenxin Zhu, Zhou Jiang, Baijun Ye, Xiaoxue
Chen, Yifei Zhang, Yuantao Chen, Jian Zhao, and Hao Zhao.
Spectrally Pruned Gaussian Fields with Neural Compensa-
tion. arXiv preprint arXiv:2405.00676, 2024. 2
[69] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Vol-
ume Rendering of Neural Implicit Surfaces. NeurIPS, 34:
4805–4815, 2021. 3
9

<!-- page 18 -->
[70] Lior Yariv, Peter Hedman, Christian Reiser, Dor Verbin,
Pratul P Srinivasan, Richard Szeliski, Jonathan T Barron,
and Ben Mildenhall. BakedSDF: Meshing Neural SDFs for
Real-Time View Synthesis. In ACM SIGGRAPH, pages 1–9,
2023. 3
[71] Kai Ye, Chong Gao, Guanbin Li, Wenzheng Chen, and Bao-
quan Chen. GeoSplatting: Towards Geometry Guided Gaus-
sian Splatting for Physically-Based Inverse Rendering. arXiv
preprint arXiv:2410.24204, 2024. 3
[72] Keyang Ye, Qiming Hou, and Kun Zhou. 3D Gaussian Splat-
ting with Deferred Reflection. In ACM SIGGRAPH, pages
1–10, 2024. 3
[73] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke.
Gaussian Grouping:
Segment and Edit Anything in 3D
Scenes. In ECCV, pages 162–179, 2024. 3
[74] Alex Yu, Sara Fridovich-Keil, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance Fields Without Neural Networks. In CVPR, page 6,
2021. 2, 4
[75] Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xian-
gli, and Bo Dai. GSDF: 3DGS Meets SDF for Improved
Neural Rendering and Reconstruction. NeurIPS, 37:129507–
129530, 2024. 3
[76] Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan
Li, Jianwei Yang, and Lei Zhang.
A Simple Framework
for Open-Vocabulary Segmentation and Detection. In ICCV,
pages 1020–1031, 2023. 3, 6
[77] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The Unreasonable Effectiveness of Deep
Features as a Perceptual Metric. In CVPR, pages 586–595,
2018. 7
[78] Zheng Zhang, Wenbo Hu, Yixing Lao, Tong He, and Heng-
shuang Zhao. Pixel-GS: Density Control with Pixel-Aware
Gradient for 3D Gaussian Splatting. In ECCV, pages 326–
342, 2024. 2
[79] Wenzhao Zheng, Junjie Wu, Yao Zheng, Sicheng Zuo, Zixun
Xie, Longchao Yang, Yong Pan, Zhihui Hao, Peng Jia, Xi-
anpeng Lang, et al. GaussianAD: Gaussian-Centric End-to-
End Autonomous Driving. arXiv preprint arXiv:2412.10371,
2024. 1
[80] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and An-
drew J. Davison. In-Place Scene Labelling and Understand-
ing with Implicit Scene Representation. In ICCV, 2021. 2,
3
[81] Hongyu Zhou, Jiahao Shao, Lu Xu, Dongfeng Bai, Weichao
Qiu, Bingbing Liu, Yue Wang, Andreas Geiger, and Yiyi
Liao. HuGS: Holistic Urban 3D Scene Understanding via
Gaussian Splatting. In CVPR, pages 21336–21345, 2024. 3
[82] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi. Feature 3DGS: Supercharging
3D Gaussian Splatting to Enable Distilled Feature Fields. In
CVPR, pages 21676–21685, 2024. 3
[83] Siting Zhu, Guangming Wang, Xin Kong, Dezhi Kong, and
Hesheng Wang. 3D Gaussian Splatting in Robotics: A Sur-
vey. arXiv preprint arXiv:2410.12262, 2024. 1
10
