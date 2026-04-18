<!-- page 1 -->
Pseudo-View Enhancement via Confidence Fusion for
Unposed Sparse-View Reconstruction
Beizhen Zhao1*
Sicheng Yu1*
Guanzhi Ding1
Yu Hu1
Hao Wang1†
1The Hong Kong University of Science and Technology (Guangzhou)
bzhao610@connect.hkust-gz.edu.cn syu605@connect.hkust-gz.edu.cn haowang@hkust-gz.edu.cn
*Equal Contribution
†Corresponding Author
Bidirectional Pseudo Frame 
Restoration
Very Sparse Frames
Scene Perception Gaussian 
Management
Insert Pseudo Frames
PSNR:20.63
PSNR:26.42
Bidirectional Pseudo Frame 
Restoration
Joint Optimization
Enhanced Reconstruction Quality
BRPO (Ours)
Poor Reconstruction Quality
Figure 1. Overview of our pipeline for 3D Gaussian Splatting from unposed sparse views in outdoor scenes. We utilize a cross-view
inconsistency UNet and a diffusion model for pesudo view completion. Through a confidence mask and scene perception Gaussian
optimization, our method outperforms others in reconstruction quality.
Abstract
3D scene reconstruction under unposed sparse viewpoints
is a highly challenging yet practically important problem,
especially in outdoor scenes due to complex lighting and
scale variation.
With extremely limited input views, di-
rectly utilizing diffusion model to synthesize pseudo frames
will introduce unreasonable geometry, which will harm the
final reconstruction quality.
To address these issues, we
propose a novel framework for sparse-view outdoor recon-
struction that achieves high-quality results through bidi-
rectional pseudo frame restoration and scene perception
Gaussian management. Specifically, we introduce a bidirec-
tional pseudo frame restoration method that restores miss-
ing content by diffusion-based synthesis guided by adjacent
frames with a lightweight pseudo-view deblur model and
confidence mask inference algorithm. Then we propose a
scene perception Gaussian management strategy that opti-
mize Gaussians based on joint depth-density information.
These designs significantly enhance reconstruction com-
pleteness, suppress floating artifacts and improve overall
geometric consistency under extreme view sparsity.
Ex-
periments on outdoor benchmarks demonstrate substantial
gains over existing methods in both fidelity and stability.
1. Introduction
Large-scale 3D outdoor scenes reconstruction from un-
posed and extremely sparse views is of paramount impor-
arXiv:2602.21535v1  [cs.CV]  25 Feb 2026

<!-- page 2 -->
tance for numerous real-world applications, including au-
tonomous driving, augmented reality, and digital twin sys-
tems, which require precise geometry and appearance for
robust long-term localization and environment understand-
ing [13, 16, 19, 35]. Consequently, developing a method
that can achieve reliable, high-fidelity reconstruction from
unposed sparse captures in complex outdoor environments
remains a fundamental and significant challenge [38, 41].
3D Gaussian Splatting (3DGS) reconstruction from un-
posed, sparse-view inputs provides impressive progress for
its real-time rendering speed and quality. Despite recent re-
search, existing methods struggle significantly under these
conditions. Unposed 3DGS approaches [9, 12, 27] often
fail due to insufficient overlap for robust correspondence
alignment. Concurrently, methods incorporating geometric
priors [15, 32, 38] or recent registration techniques [2, 24]
are often limited to refining observed regions or small-scale
indoor scenes, failing to recover unobserved geometry in
large outdoor environments. Furthermore, generative regu-
larization methods [19, 31, 36] typically presuppose known
camera poses or object-centric data, rendering them inap-
plicable to this challenging unposed, large-scale setting.
To make up the severe lack of multi-view constraints in
sparse-view reconstruction, a natural approach is to lever-
age generative models, such as diffusion, to synthesize
pseudo-views and densify the input. However, we find that
a diffusion-based completion model usually generates clear
but actually unreasonable pseudo-view, which will harm
the final reconstruction quality.
These plausible-looking
but geometrically inconsistent syntheses introduce conflict-
ing information during optimization, leading to artifacts
and geometry degradation rather than improvement. There-
fore, how to generate reliable and geometrically-consistent
pseudo-views remains a critical bottleneck for advancing
sparse-view 3D reconstruction.
To address these challenges, we propose a new unposed
sparse-view reconstruction framework that focuses on out-
door view restoration Gaussian management. Our key idea
is a bidirectional pseudo frame restoration method that
enriches visual information for under-constrained regions.
To achieve more realistic pseudo view, we first propose a
lightweight pseudo-view deblur model for more robust ge-
ometric consistency and reliable pseudo frame restoration.
Then, instead of blindly using pseudo frames, we design
an overlap score fusion and confidence mask inference al-
gorithm that fuse bidirectional pseudo frames based on the
viewpoint change and overlap ratio with neighboring inputs
and dynamically decides the confidence of pseudo frame
through feature mapping and reprojection.
While pseudo-view synthesis effectively improves re-
construction completeness, the sparse input view can cause
uneven Gaussians distribution and the difficulty of joint op-
timization. To overcome this issue, we introduce a scene
perception mechanism that adaptively optimize Gaussians
based on perceptual significance. The mechanism leverages
an entropy-based depth-density measure to estimate the im-
portance of each Gaussian in representing the local struc-
ture, thereby enhancing under-optimized regions.
In summary, our main contributions are as follows:
• A bidirectional pseudo frame restoration method with
a pseudo-view deblur network to enhance pseudo view
completion.
• An overlap score fusion with confidence mask inference
algorithm that selectively synthesizes confidence mask
for sparse-view reconstruction.
• A scene perception Gaussian management strategy that
adaptively optimize floating Gaussians through depth and
density control metric.
Extensive experiments demonstrate that our method sub-
stantially outperforms existing 3DGS approaches in recon-
struction accuracy, completeness, and visual consistency,
offering a new solution for high-quality outdoor 3D recon-
struction under sparse-view conditions.
2. Related Work
2.1. Unposed 3D Gaussian splatting
3DGS [13] delivers real-time and high-fidelity novel view
synthesis from explicit Gaussians. Yet reconstruction typ-
ically relies on COLMAP [26] poses and sparse points,
prompting interest in unposed 3DGS, which builds Gaus-
sian scenes directly from raw images.
CF-3DGS [9] initializes the Gaussian field using monoc-
ular depth and progressively refines both the camera param-
eters and Gaussians during optimization. COGS [12] per-
forms incremental scene reconstruction by aligning cameras
through dense 2D correspondences, while Rob-GS [7] in-
troduces a robust pairwise registration strategy to stabilize
pose estimation. Several methods [7, 12, 27] recover cam-
era geometry through dense 2D correspondence alignment,
pairwise registration, or joint optimization , thereby en-
abling incremental, calibration-free reconstruction. In par-
allel, a series of 3DGS-SLAM frameworks [4, 21, 25, 35]
achieve unposed reconstruction by streaming images as in-
put. Recent works leverage pretrained geometric models to
improve unposed 3DGS. InstantSplat [8] employs DUSt3R
[30] for initialization and accelerates convergence through
parallel grid partitioning, though its scalability remains lim-
ited under sparse-view inputs. LongSplat [16] integrates
MASt3R [14] priors with incremental joint optimization
and an efficient octree anchor mechanism, supporting long
sequences for novel view synthesis. [3] utilize sub-graph
alignment and joint optimization based on VGGT [29] to
extend reconstruction to outdoor long sequences.

<!-- page 3 -->
Reprojection Consistency Score
Point Map 
Projection
Joint Optimization
Pose Estimation
Gaussians 
Initialization
k
D
tI
Rendered 
Pseudo Frame
Fixed Frame
Pseudo-view Deblur 
UNet
Confidence Mask
Fixed Frame
Residual
Feature Mapping
Scene Perception Gaussian 
Management
Confidence Mask Inference
Overlap Score Fusion
Fused Image
Adjacent Reference 
Frames
Unposed Sparse 
Views
Bidirectional 
Enhancement
1
k
D
tI
+
Deblurred 
Pseudo Frame
Pseudo Pose
Tracking
Depth Mapping
Figure 2. Framework of BRPO. We begin by the sparse input views and through the cross-view vision to select and restore frame.
We design a lightweight pseudo-view deblur UNet to denoise the Gaussian rendered image and utilize a diffusion-based model for image
completion. After that, we combine the two fixed images by calculating reprojection overlap score and conduct feature mapping to generate
a confidence mask to guide the joint optimization.
2.2. 3D Gaussian Splatting from Sparse Views
Adequate view coverage is vital for high-quality novel view
synthesis [13, 20, 40], while sparse input often causes
floating artifacts and cross-view inconsistency. Some ap-
proaches [5, 15, 22, 32, 38, 41] integrate external geomet-
ric priors to enforce depth alignment. Others [18, 19, 31,
36, 42] use generative priors to regularize view-dependent
effects. Complementary studies [11, 39] introduce opac-
ity attenuation and binocular-guided photometric consis-
tency to suppress floating artifacts and stabilize geometry.
More recently, dropout and stochastic density modulation
have been explored to improve robustness under sparse-
view constraints [23, 33], while [1] explicitly mitigate co-
adaptation effects. However, these methods typically need
COLMAP priors or are limited to object-centric scenes.
RegGS [2] achieves unposed sparse-view reconstruction
by registering locally generated Gaussian models from No-
PoSplat [34] into a globally consistent representation, but
only demonstrated on indoor scenes.
Gscenes [24] de-
velops a diffusion-driven, pose-free sparse-view pipeline
that iteratively refines novel-view renders and depths with
confidence-guided RGB-D priors, but its offline refinement
and reliance on dense RGB-D supervision hinder scalability
and efficiency. SGD [37] regularizes 3DGS for street scenes
using diffusion prior conditioned on adjacent frames and
LiDAR to improve sparse-view synthesis, but it assumes
known poses and LiDAR supervision and remains scene-
specific in its diffusion regularization.
We introduce a novel framework that tackles the chal-
lenge through bidirectional pseudo frame restoration and
ensures robust optimization using a scene perception Gaus-
sian management strategy, significantly enhancing recon-
struction completeness and geometric consistency even un-
der extreme view sparsity.
3. Methodology
Our framework aims to achieve high-quality outdoor 3D re-
construction under extremely sparse unposed viewpoints.
The key idea is to augment the traditional pipeline with
bidirectional pseudo frame restoration and scene percep-
tion Gaussian management. Specifically, we integrate four
essential components:
(1) a bidirectional pseudo frame
restoration module with a pseudo-view deblur network for
robust frame simulation, (2) a pseudo frame fusion algo-
rithm for overlap score calculation and confidence mask in-
ference, (3) a scene perception Gaussian management for
Gaussians optimization, and (4) a joint optimization pro-
cess coupling Gaussian attributes and camera poses. Fig. 2
illustrates the overall framework.
3.1. Bidirectional Pseudo Frame Restoration
Sparse camera sampling yields weak geometric constraints
with missing detail and inconsistent appearance. Directly
using these imperfect frames for Gaussian initialization or
photometric alignment produces floating Gaussians and op-
timization instability. Therefore we first synthesize con-

<!-- page 4 -->
sistent, perceptually faithful pseudo-frames that are used
as reliable supervision for subsequent Gaussian construc-
tion and optimization. We propose a two-stage restoration
pipeline for constructing reliable frames from sparse and
possibly corrupted observations, which leverages geomet-
ric priors and reference information to enforce cross-view
consistency during restoration process.
Pseudo-view Deblur Network.
We observe that directly
leveraging diffusion-based model for parse view completion
will introduce unreasonable information which extremely
harms and misleads the reconstruction quality. To better
handle inter-frame inconsistencies and reduce artifacts be-
fore diffusion-based restoration, we introduce a lightweight
pseudo-view deblur network Uc built on UNet backbone
to integrate complementary cues from adjacent real frames
while preserving the structure and color consistency of the
current view.
Given the Gaussian-rendered current frame Igs
t
and its
two neighboring reference frames Irf
k
and Irf
k+1, the net-
work predicts a refined image ˆIt as:
ˆIt = Uc

Irf
k , Igs
t , Irf
k+1

.
(1)
Uc employs a multi-scale UNet architecture with shared
early encoders for reference alignment and skip connec-
tion for feature fusion. By modeling correlations between
frames, the network explicitly compensates for viewpoint-
induced inconsistencies, effectively removing ghosting and
blending artifacts.
The refined output ˆIt is then forwarded to the reference-
conditioned diffusion model D that produces two candidate
restorations conditioned on the past and future references:
IDk
t
= D(ˆIt, Irf
k ), IDk+1
t
= D(ˆIt, Irf
k+1),
(2)
which produces the final reliable frame from the fused rep-
resentation.
Overlap Score Fusion.
However, through different adja-
cent reference frames, the fixed image is different in details,
which may be unreasonable. So we design a mixture algo-
rithm to generate the final fixed frame. Given two cameras
Ca and Cb, we first estimate their 2D overlap region by pro-
jecting the depth map of Ca into Cb:
pb = Π(K, TbT−1
a Π−1(pa, da)),
(3)
where pa is a pixel in view a, Π and Π−1 denote projection
and back-projection operators. The per-pixel overlap mask
Oab(p) is defined by visibility consistency:
Oab(p) =

pb ∈Ωb

∧

db(pb) > ϵ

.
(4)
The depth-consistency score sd is:
sd(p) = exp

−
|da(p) −db(pb)|
(da(p) + db(pb))/2 + ϵ

.
(5)
We further incorporate a pose-consistency scalar st =
exp(−∥ta −tb∥2) to down-weight references with large
relative translation. We define the combined overlap con-
fidence map:
Cab(p) = sd(p) st.
(6)
Finally, we compute residuals r(i) = ˜I(i)
t
−It and form
the fused repaired image:
Ifix
t
= It + W1 ⊙r(1) + W2 ⊙r(2),
(7)
where
Wi(p) =
Ct,i(p)
Ct,1(p) + Ct,2(p) + ϵ.
(8)
Confidence Mask Inference.
While the fused pseudo-
frame Ifix
t
is visually enhanced, it may still contain ge-
ometric inconsistencies or “hallucinations” introduced by
the diffusion model.
Allowing this unstable information
to propagate would degrade the final reconstruction qual-
ity. We therefore introduce a geometric verification step to
generate a confidence map Cm, based on robust feature cor-
respondences.
The core principle is that pixels in the synthetic frame
Ifix
t
must find geometrically consistent counterparts in the
real reference frames. We leverage a robust correspondence
network [14] to establish two mutual nearest-neighbor cor-
respondence sets: (1) Mk, linking Ifix
t
and last reference
Irf
k , and (2) Mk+1, linking Ifix
t
and next reference Irf
k+1.
Based on this bidirectional geometric constraint, we as-
sign a confidence weight Cm(p) to each pixel p in Ifix
t :
Cm(p) =





1.0
if p ∈Mk ∩Mk+1
0.5
if p ∈Mk ⊕Mk+1
0.0
if p /∈Mk ∪Mk+1
.
(9)
This multi-level weighting strategy effectively suppresses
artifacts by trusting only pixels with strong, bidirectionally-
consistent geometric evidence, guiding the subsequent joint
optimization.
3.2. Scene Perception Gaussian Management
Although the pseudo-views augment the input data, it is
inadequate for the difficulty of joint optimization process.
The optimization instability still corrupts the geometric in-
tegrity and visual fidelity, manifesting as floating artifacts
and poorly-optimized Gaussians.
We use scene percep-
tion Gaussian management that combines a 1D optimal-
transport inspired depth partitioning with a global density-
entropy driven adaptive scoring. The goal is to partition
Gaussians adaptively and compute a stable per-Gaussian
importance score that fuses depth and density cues.

<!-- page 5 -->
Depth partitioning via 1D optimal transport.
We
treat the empirical distribution of depths {zi} as a one-
dimensional probability measure on R+. In one dimension,
the p-Wasserstein distance admits a closed-form represen-
tation in terms of quantile functions (inverse CDF). For two
probability measures µ, ν on R with cumulative distribution
functions Fµ, Fν, the squared 2-Wasserstein distance is:
W 2
2 (µ, ν) =
Z 1
0
 F −1
µ (t) −F −1
ν
(t)
2 dt.
(10)
This relationship implies that for 1D clustering of a sin-
gle empirical measure into K contiguous clusters, quantile-
based splits are a natural approximate minimizer of within-
cluster Wasserstein energy because they align with the
quantile geometry of the measure. Concretely, let FN be the
empirical CDF of depths and QN(t) = F −1
N (t) its quantile
function. For K = 3 clusters we choose percentiles τ1, τ2
and define boundaries b1 = QN(τ1) and b2 = QN(τ2).
The quantile choice yields contiguous depth intervals and
approximates the minimization of intra-cluster Wasserstein
cost in the 1D case. We get the depth scroe by:
ˆs(z)
i
=
1 −d
dmax −dmin + δ ,
(11)
where d denotes the camera viewpoint depth.
Density entropy for global concentration.
We next
characterize the global structure of the density distribution
{ρi} using an information entropy measure. Distributions
with low entropy are concentrated and high entropy indi-
cates a more uniform density. We estimate a histogram-
based discrete probability mass function from {ρi}. Let
a histogram with B bins produce probabilities pb, b =
1, . . . , B, with P
b pb = 1. We normalize the Shannon en-
tropy by the maximum achievable entropy log B to obtain:
H(ρ) = −
1
log B
B
X
b=1
pb log(pb + ε),
(12)
where ε is a small constant for numerical stability. We then
produce an entropy-aware fused density score ˆs(ρ)
i
:
ˆs(ρ)
i
= ˜ρi · (1 −β ¯H) + ¯H · γ,
(13)
where β, γ ∈[0, 1] are hyperparameters (we use β =
0.5, γ = 0.5). This yields ˆs(ρ)
i
∈[0, 1] and smoothly blends
density and entropy priors. Finally, we combine depth and
density into a unified importance score:
Si = α s(z)
i
+ (1 −α) ˆs(ρ)
i
,
(14)
where α ∈[0, 1] balances depth vs. density. In our experi-
ments we set α = 0.5.
Cluster-aware drop probability.
We design cluster-
specific attenuation factors so that the pruning aggressive-
ness is tuned by depth. The drop probability is:
pdrop
i
= r · wcluster(i) · Si,
(15)
where wcluster(i) denotes cluster weights designed for differ-
ent depth levels divided by boundaries and r denotes initial
drop-rate scalar. We sample and apply the binary mask:
αi ←αi · mi,
where
mi ∼Bernoulli(1 −pdrop
i
).
(16)
This stochastic masking preserves a soft expectation of
retained mass and empirically reduces deterministic bound-
ary artifacts versus hard thresholding.
3.3. Joint optimization and Gaussian refinement
We adopt a two-stage optimization strategy: first stabilize
poses and radiometric corrections, then jointly refine Gaus-
sians and poses with geometric and appearance constraints.
Pose
deltas
and
exposure
stabilization.
For
early
frames, we optimize pose offsets and exposure correction
parameters while keeping the Gaussian collection mostly
fixed. Concretely, for each viewpoint we parameterize pose
updates as small rotation and translation deltas ∆Rt, ∆tt
and per-view exposure parameters at, bt that model a simple
affine photometric correction:
I′
t = atIt + bt.
(17)
Early pose stabilization prevents errors from passing to
Gaussian parameters, leading to drifting geometry. Joint
optimization after stable pose initialization enforces tighter
geometry-appearance coupling and more photorealistic re-
constructions. During joint optimization, both the Gaussian
parameters G = {(µi, Σi, ci, αi)} and camera poses {Tt}
are updated through a weighted RGB-D reconstruction loss:
L = β Lrgb + (1 −β) Ld + λsLs,
(18)
where
Lrgb = ∥Cm ⊙(It −ˆIt)∥1
∥Cm∥1
,
(19)
Ld = ∥Cm ⊙(Dt −ˆDt)∥1
∥Cm∥1
.
(20)
Here, Cm is the confidence mask from Eq. 9, Ls is a loss
to regularize the scale of each Gaussian.
The weight β
(typically 0.95) balances the color and depth constraints.
This stage jointly refines camera trajectories and the spa-
tial–radiometric distribution of Gaussians.

<!-- page 6 -->
Method
DL3DV [17]
Waymo [28]
KITTI [10]
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
CF-3DGS [9]
13.01
0.426
0.582
13.67
0.590
0.554
9.79
0.303
0.619
VideoLifter∗[6]
17.84
0.519
0.433
13.53
0.471
0.592
8.356
0.194
0.647
Instantsplat [8]
10.88
0.351
0.615
12.49
0.564
0.645
9.926
0.367
0.654
Longsplat [16]
17.40
0.516
0.428
15.06
0.551
0.522
10.69
0.285
0.617
RegGS∗[2]
19.77
0.612
0.391
-
-
-
-
-
-
OpenGS-SLAM [35]
16.08
0.428
0.403
21.37
0.765
0.457
13.87
0.433
0.537
S3PO-GS [4]
20.67
0.647
0.219
22.03
0.771
0.452
15.58
0.510
0.513
Ours
24.27
0.753
0.183
23.76
0.777
0.348
17.95
0.605
0.472
Table 1. Novel View Synthesis Result on Three Dataset. Note that some methods with a ∗means that this method fails to finish the
reconstruction in some scenes due to the extremely sparse condition and we only shows the successful results.
Figure 3. Visual quality on KITTI dataset. Our approach consistently outperforms other models on different scenes, demonstrating
advantages in challenging scenarios. Best viewed in color.
Gaussian Refinement.
After pose convergence, we per-
form an appearance refinement of G by minimizing
Lrefine = (1 −λssim) LL1 + λssim (1 −LSSIM).
(21)
Here, both the LL1 and LSSIM terms are also weighted by
the confidence map Cm. This refinement encourages struc-
tural consistency and prevents over-expansion of Gaussian
primitives, achieving photorealistic reconstruction.
4. Experiments
4.1. Experiments Setup
Datasets
We evaluate the proposed method on three
widely-used outdoor 3D reconstruction datasets that cap-
ture progressively more challenging imaging and viewpoint
variation regimes.
• DL3DV [17] (easy), a controlled outdoor benchmark
with relatively dense multi-view coverage and moder-
ate viewpoint changes.
Scenes typically contain well-
textured static geometry and limited occlusion.
• Waymo [28] (moderate), a large-scale driving dataset
with realistic city and highway scenes, varying light-
ing and moderately large inter-frame motion, which re-
quires robustness to viewpoint shifts and partial occlu-
sions while still providing sufficient correspondences for
multi-view fusion.
• KITTI [10] (hard), a driving dataset with extreme view-
point changes, large motion parallax, and many regions
with poor texture.
Incorrectly appearance in poorly-
constrained regions tends to produce severe artifacts.
Quantitative metrics reported in this paper include ren-
dering quality measures (PSNR, SSIM and LPIPS) for
novel view synthesis (NVS). For pose estimation evalua-
tion, we use ATE RMSE as a metric. Detailed results are
reported in the supplementary material.
Baselines.
We compare our method with state-of-the-art
methods for unposed reconstruction, including CF-3DGS
[9], VideoLifter [6], Instantsplat [8], Longsplat [16], RegGS
[2], OpenGS-SLAM [35] and S3PO-GS [4].
Implementation details.
All experiments are imple-
mented in PyTorch and run on one NVIDIA A6000 GPU.
The diffusion backbone uses the publicly available Difix3D
[31] pretrained weights as the initial generative prior for
image-space correction. We utilize MASt3R [14] as the
pose estimation model.
We uniformly extract one-tenth
of the frames from the KITTI and Waymo datasets as in-
put, and extract one frame every 30 frames for the DL3DV
dataset.

<!-- page 7 -->
Figure 4. Visual quality on DL3DV dataset. Our approach consistently outperforms other models on different scenes, demonstrating
advantages in challenging scenarios. Best viewed in color.
Figure 5. Pose Estimation Visualization on KITTI and DL3DV
dataset. Our method can reach more accurate pose estimation
compared to other methods.
4.2. Experimental Results and Analysis
In this section we present a focused analysis of the method’s
performance across the three datasets as shown in Fig. 4,
Fig. 3, Tab. 1 and Tab. 2. Across all datasets, we observe
that the full pipeline consistently improves reconstruction
fidelity and reduces view-dependent artifacts relative to
baselines. On DL3DV the scene geometry is relatively well-
Method
DL3DV
Waymo
KITTI
CF-3DGS [9]
3.093
10.84
65.89
VideoLifter∗[6]
3.184
32.71
81.22
Instantsplat [8]
0.333
9.992
65.10
Longsplat [16]
0.919
7.548
67.73
RegGS∗[2]
2.045
-
-
OpenGS-SLAM [35]
1.586
1.784
4.722
S3PO-GS [4]
0.343
3.149
4.490
Ours
0.077
1.352
2.471
Table 2. Pose Estimation Error–ATE RMSE on Three Dataset.
constrained. We observe that in boundary regions, the dif-
fusion model can introduce minor view-dependent inconsis-
tencies. Our pseudo-view deblur module further improves
quality by consolidating evidence from adjacent frames
and suppressing these inconsistencies, while the confidence
mask prevents over correction in predicted regions.
Waymo introduces tougher photometric variation and
occlusions.
The UNet effectively deblurs regions where
single-view diffusion tends to inconsistent content. Gaus-
sian optimization produces improvements by re-distributing
gaussian weights to better match sparse view observations,
which enables more accurate reprojected color priors.
KITTI is the most challenging dataset due to extreme
viewpoint shifts and large textureless areas. The diffusion-
only baseline tends to over-generate in texture-poor re-
gions, producing plausible-looking but inconsistent appear-
ance.
The confidence mask is particularly important as
it suppresses signals from reprojected pixels whose depth
estimates are unreliable, preventing integrating misleading
information.
Although KITTI remains challenging, our
method yields the largest relative improvements, demon-
strating robustness in the face of severe viewpoint variation.

<!-- page 8 -->
Method
Waymo-152706
KITTI-00
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
w/o Mask&Unet
19.76
0.735
0.440
13.63
0.527
0.508
w/o UNet
23.19
0.768
0.357
15.95
0.608
0.396
w/o Mask
24.40
0.779
0.385
16.90
0.638
0.395
w/o Bid-Fusion
24.29
0.776
0.378
17.13
0.636
0.379
w/o SPGM
24.03
0.774
0.381
17.10
0.628
0.382
w/o Depth
24.33
0.779
0.374
17.76
0.641
0.363
w/o Density
24.51
0.780
0.368
17.89
0.655
0.366
Ours
24.98
0.787
0.364
18.34
0.671
0.354
Table 3. Ablation Study on Waymo and KITTI.
4.3. Ablation Study
We conduct a series of controlled ablations to isolate the
contribution of each component as shown in Tab. 3. Each
ablation is evaluated on a held-out validation subset for each
dataset. All other hyper-parameters and training schedules
are kept identical across ablations.
w/o UNet & Mask.
The diffusion prior often produces
visually plausible completions, but in under-constrained re-
gions it will “freely imagine” content that is inconsistent
across views. This leads to large per-pixel and perceptual
errors when the hallucinated content does not align with true
geometry. Qualitatively, this appears as patchy or blurred
seams when reprojected to other views.
w/o UNet.
Introducing the UNet yields a significant re-
duction in these artifacts as shown in Fig. 6. Given the cur-
rent Gaussian-rendered image and neighboring frames, the
pseudo-view deblur UNet learns to keep reliable rendering
information only where they are consistent with multi-view
evidence and to suppress free hallucinations. This stage im-
proves both pixel-wise metrics and perceptual consistency.
w/o Mask.
When the UNet is allowed to fuse all repro-
jected information indiscriminately it can still be misled by
reprojected pixels coming from unreliable depths or occlu-
sions. The reprojection confidence mask gates the contribu-
tion of each reprojected sample. As shown in Fig. 7, using
confidence mask to weight reprojected contributions yields
an additional measurable improvement in both appearance
and geometry metrics.
w/o Bid-Fusion.
We validate the efficacy of bidirectional
fusion.
Compared to relies on information from only a
single adjacent frame, the bidirectional mechanism demon-
strates a clear advantage. By intelligently fusing features
from multi reference frames, this bidirectional flow enables
the compensation and averaging of potential errors inherent
in any single view The result is a significant reduction in
Figure 6. Ablation study on UNet The UNet can prefilter unrea-
sonable artifacts and generate more reliable pseudo frame.
Figure 7. Ablation study on confidence mask The confidence
mask can recognize unreliable regions in pseudo frames.
visual artifacts and a marked improvement in the geometric
consistency of the restored regions.
w/o SPGM.
We test the impact of the scene perception
Gaussian management by disabling it while keeping all
other components. Without this optimization, the recon-
structions are less geometrically consistent and the diffu-
sion stage contains noisier color priors, which results in a
drop in geometry-aware metrics and a modest deterioration
in appearance metrics. The results show that both depth and
density regularization is vital for the reconstruction quality.
4.4. Limitations
While the proposed method demonstrates consistent im-
provements over strong baselines across a range of outdoor
datasets, several limitations remain: Although our pipeline
outperforms baselines on KITTI, scenes with extremely
large viewpoint variation and large textureless regions still
pose challenges. Diffusion priors may still introduce view-
incoherent completions when multi-view constraints are in-
sufficient. Improving robustness in these regimes will likely
require stronger geometric priors. Our current framework
assumes static scene geometry at the scale of fusion. Han-
dling dynamics would require explicit motion modeling or
temporal separation strategies.

<!-- page 9 -->
5. Conclusion
This paper addresses the challenge of large-scale outdoor
3D scene reconstruction from extremely sparse views, a
critical yet unsolved problem. We proposed a novel frame-
work that combines bidirectional pseudo frame restoration
with scene perception Gaussian management to overcome
the limitations of existing methods in sparse, unposed out-
door settings. We introduce a lightweight pseudo-view de-
blur model to guide reliable pseudo-views generation and
a overlap score fusion with confidence mask inference al-
gorithm to fuse informative frames. Extensive experiments
confirm that our method significantly improves reconstruc-
tion quality and visual consistency compared to state-of-
the-art methods. Future work will extend the framework
to 4D scenes.
Acknowledge
This work is supported by the National Natural Sci-
ence Foundation of China (No.
62406267), Guang-
dong
Provincial
Project
(No.
2024QN11X072),
Guangzhou-HKUST(GZ)
Joint
Funding
Pro-
gram
(No.
2025A03J3956)
and
Guangzhou
Mu-
nicipal
Education
Project
(No.
2024312122).
References
[1] Kangjie Chen, Yingji Zhong, Zhihao Li, Jiaqi Lin, Youyu
Chen, Minghan Qin, and Haoqian Wang. Quantifying and
alleviating co-adaptation in sparse-view 3d gaussian splat-
ting. arXiv preprint arXiv:2508.12720, 2025. 3
[2] Chong Cheng, Yu Hu, Sicheng Yu, Beizhen Zhao, Zijian
Wang, and Hao Wang. Reggs: Unposed sparse views gaus-
sian splatting with 3dgs registration. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 8100–8109, 2025. 2, 3, 6, 7
[3] Chong Cheng, Zijian Wang, Sicheng Yu, Yu Hu, Nan-
jie Yao, and Hao Wang.
Unposed 3dgs reconstruction
with probabilistic procrustes mapping.
arXiv preprint
arXiv:2507.18541, 2025. 2
[4] Chong Cheng, Sicheng Yu, Zijian Wang, Yifan Zhou, and
Hao Wang.
Outdoor monocular slam with global scale-
consistent 3d gaussian pointmaps.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 26035–26044, 2025. 2, 6, 7
[5] Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee.
Depth-regularized optimization for 3d gaussian splatting in
few-shot images.
In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
811–820, 2024. 3
[6] Wenyan Cong, Hanqing Zhu, Kevin Wang, Jiahui Lei,
Colton Stearns, Yuanhao Cai, Dilin Wang, Rakesh Ranjan,
Matt Feiszli, Leonidas Guibas, et al.
Videolifter: Lifting
videos to 3d with fast hierarchical stereo alignment. arXiv
preprint arXiv:2501.01949, 2025. 6, 7
[7] Zhen-Hui Dong, Sheng Ye, Yu-Hui Wen, Nannan Li, and
Yong-Jin Liu. Towards better robustness: Progressively joint
pose-3dgs learning for arbitrarily long videos. arXiv e-prints,
pages arXiv–2501, 2025. 2
[8] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang,
Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic,
Marco Pavone, Georgios Pavlakos, et al.
Instantsplat:
Sparse-view gaussian splatting in seconds. arXiv preprint
arXiv:2403.20309, 2024. 2, 6, 7
[9] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A
Efros, and Xiaolong Wang. Colmap-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 20796–20805,
2024. 2, 6, 7
[10] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel
Urtasun. Vision meets robotics: The kitti dataset. The in-
ternational journal of robotics research, 32(11):1231–1237,
2013. 6
[11] Liang Han, Junsheng Zhou, Yu-Shen Liu, and Zhizhong
Han. Binocular-guided 3d gaussian splatting with view con-
sistency for sparse view synthesis. Advances in Neural In-
formation Processing Systems, 37:68595–68621, 2024. 3
[12] Kaiwen Jiang, Yang Fu, Mukund Varma T, Yash Belhe, Xi-
aolong Wang, Hao Su, and Ravi Ramamoorthi. A construct-
optimize approach to sparse view synthesis without camera
pose. In ACM SIGGRAPH 2024 Conference Papers, pages
1–11, 2024. 2
[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 3
[14] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Ground-
ing image matching in 3d with mast3r. In European Confer-
ence on Computer Vision, pages 71–91. Springer, 2024. 2, 4,
6, 1
[15] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun
Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d
gaussian radiance fields with global-local depth normaliza-
tion. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 20775–20785,
2024. 2, 3
[16] Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen,
Yen-Yu Lin, and Yu-Lun Liu. Longsplat: Robust unposed
3d gaussian splatting for casual long videos. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 27412–27422, 2025. 2, 6, 7
[17] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin,
Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,
et al.
Dl3dv-10k: A large-scale scene dataset for deep
learning-based 3d vision. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 22160–22169, 2024. 6
[18] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang,
Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Re-
conx: Reconstruct any scene from sparse views with video
diffusion model. arXiv preprint arXiv:2408.16767, 2024. 3
[19] Xi Liu, Chaoyi Zhou, and Siyu Huang.
3dgs-enhancer:
Enhancing unbounded 3d gaussian splatting with view-

<!-- page 10 -->
consistent 2d diffusion priors. Advances in Neural Informa-
tion Processing Systems, 37:133305–133327, 2024. 2, 3
[20] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 3
[21] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 18039–18048, 2024. 2
[22] Avinash Paliwal, Wei Ye, Jinhui Xiong, Dmytro Kotovenko,
Rakesh Ranjan, Vikas Chandra, and Nima Khademi Kalan-
tari. Coherentgs: Sparse novel view synthesis with coherent
3d gaussians. In European Conference on Computer Vision,
pages 19–37. Springer, 2024. 3
[23] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian:
Structural regularization for sparse-view gaussian splatting.
In Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 21600–21609, 2025. 3
[24] Soumava Paul, Prakhar Kaushik, and Alan Yuille.
Gaus-
sian scenes:
Pose-free sparse-view scene reconstruction
using depth-enhanced diffusion priors.
arXiv preprint
arXiv:2411.15966, 2024. 2, 3
[25] Erik Sandstr¨om, Ganlin Zhang, Keisuke Tateno, Michael
Oechsle, Michael Niemeyer, Youmin Zhang, Manthan Pa-
tel, Luc Van Gool, Martin Oswald, and Federico Tombari.
Splat-slam: Globally optimized rgb-only slam with 3d gaus-
sians. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 1680–1691, 2025. 2
[26] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 2
[27] Dongbo Shi, Shen Cao, Lubin Fan, Bojian Wu, Jinhui Guo,
Renjie Chen, Ligang Liu, and Jieping Ye. Trackgs: Opti-
mizing colmap-free 3d gaussian splatting with global track
constraints. arXiv preprint arXiv:2502.19800, 2025. 2
[28] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien
Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,
Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han,
Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Et-
tinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang,
Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov.
Scalability in perception for autonomous driving: Waymo
open dataset. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2020.
6
[29] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 2
[30] Shuzhe Wang,
Vincent Leroy,
Yohann Cabon,
Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vi-
sion made easy. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 20697–
20709, 2024. 2
[31] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi
Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Goj-
cic, and Huan Ling.
Difix3d+: Improving 3d reconstruc-
tions with single-step diffusion models. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 26024–26035, 2025. 2, 3, 6, 1
[32] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys.
Depthsplat: Connecting gaussian splatting and depth.
In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 16453–16463, 2025. 2, 3
[33] Yexing Xu, Longguang Wang, Minglin Chen, Sheng Ao,
Li Li, and Yulan Guo.
Dropoutgs: Dropping out gaus-
sians for better sparse-view rendering. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
701–710, 2025. 3
[34] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys,
Ming-Hsuan Yang, and Songyou Peng. No pose, no problem:
Surprisingly simple 3d gaussian splats from sparse unposed
images. arXiv preprint arXiv:2410.24207, 2024. 3
[35] Sicheng Yu, Chong Cheng, Yifan Zhou, Xiaojun Yang, and
Hao Wang. Rgb-only gaussian splatting slam for unbounded
outdoor scenes. arXiv preprint arXiv:2502.15633, 2025. 2,
6, 7
[36] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li,
Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan,
and Yonghong Tian. Viewcrafter: Taming video diffusion
models for high-fidelity novel view synthesis. arXiv preprint
arXiv:2409.02048, 2024. 2, 3
[37] Zhongrui Yu, Haoran Wang, Jinze Yang, Hanzhang Wang,
Jiale Cao, Zhong Ji, and Mingming Sun. Sgd: Street view
synthesis with gaussian splatting and diffusion prior. In 2025
IEEE/CVF Winter Conference on Applications of Computer
Vision (WACV), pages 3812–3822. IEEE, 2025. 3
[38] Chuanrui Zhang, Yingshuang Zou, Zhuoling Li, Minmin Yi,
and Haoqian Wang. Transplat: Generalizable 3d gaussian
splatting from sparse multi-view images with transformers.
In Proceedings of the AAAI Conference on Artificial Intelli-
gence, pages 9869–9877, 2025. 2, 3
[39] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu,
Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian
splatting via co-regularization. In European Conference on
Computer Vision, pages 335–352. Springer, 2024. 3
[40] Beizhen Zhao, Yifan Zhou, Sicheng Yu, Zijian Wang, and
Hao Wang. Wavelet-gs: 3d gaussian splatting with wavelet
decomposition. In Proceedings of the 33rd ACM Interna-
tional Conference on Multimedia, pages 8616–8625, 2025.
3
[41] Yulong Zheng, Zicheng Jiang, Shengfeng He, Yandu Sun,
Junyu Dong, Huaidong Zhang, and Yong Du.
Nexusgs:
Sparse view synthesis with epipolar depth priors in 3d gaus-
sian splatting. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 26800–26809, 2025.
2, 3
[42] Yingji Zhong, Zhihao Li, Dave Zhenyu Chen, Lanqing
Hong, and Dan Xu. Taming video diffusion prior with scene-
grounding guidance for 3d gaussian splatting from sparse in-

<!-- page 11 -->
puts.
In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 6133–6143, 2025. 3

<!-- page 12 -->
Pseudo-View Enhancement via Confidence Fusion for
Unposed Sparse-View Reconstruction
Supplementary Material
Figure 8. Structure of Pseudo-view Deblur UNet.
6. Training of Pseudo-view Deblur UNet
For the task of pseudo-view deblurring, we propose a novel
lightweight UNet architecture. The detailed schematic of
this network is illustrated in Fig. 8 and Fig. 9. The model
is fundamentally based on a symmetric Encoder-Decoder
(Enc-Dec) structure.
A critical design choice is the integration of skip connec-
tions, which bridge the corresponding layers between the
encoder and decoder paths. These connections are essen-
tial for enhancing the network’s feature fusion capabilities.
They allow the decoder to directly access low-level, high-
resolution spatial information from the encoder, which is
vital for reconstructing fine details and textures lost during
the down-sampling process.
The lightweight UNet is trained entirely from scratch,
without reliance on any pre-trained weights. To facilitate
this, we generated a comprehensive synthetic dataset. This
dataset contains over 10,000 image pairs, each consisting
of rendered Gaussian images (as the blurred input) and
their corresponding randomly designed neighbor reference
frames.
The training process was performed on a high-
performance computing setup, utilizing four NVIDIA
A6000 GPUs operating in parallel.
We employed the
AdamW optimizer for parameter updates, selected for its
robustness in training deep networks. The learning rate was
dynamically managed using a cosine learning rate scheduler
that included an initial warm-up mechanism to ensure sta-
ble convergence. The model was trained for 1000 epochs,
achieving convergence in approximately 10 hours on this
4-A6000 configuration.
Figure 9.
Detailed structure of Encoder (left) and Decoder
(right).
Scene
1
2
3
Avg.
ATE
0.131
0.036
0.065
0.077
PSNR
27.34
22.70
22.78
24.27
SSIM
0.846
0.747
0.666
0.753
LPIPS
0.117
0.216
0.217
0.183
Table 4. Detailed results on DL3DV dataset.
7. More Experiment Results
All experiments are implemented in PyTorch and run on
one NVIDIA A6000 GPU. The diffusion backbone uses
the publicly available Difix3D [31] pretrained weights as
the initial generative prior for image-space correction. We
utilize Mast3R [14] as the pose estimation model.
We
uniformly extract one-tenth of the frames from the KITTI
and Waymo datasets as input.
Given that the DL3DV
dataset is relatively dense and small, a sparser approach
was necessary to prevent input redundancy.
We there-
fore extracted only one frame every 30 frames from this
dataset. For KITTI dataset, we choose the firse 200 frames
of each scenes. For Waymo dataset, we select the frame
of front camera in nine snenes. Experimental ResultsThe
detailed quantitative performance metrics across the evalu-
ated datasets are systematically presented in the following
tables: Tab. 4 (DL3DV dataset), Tab. 5 (Waymo dataset),
and Tab. 6 (KITTI dataset).

<!-- page 13 -->
Scene
100613
13476
106762
132384
152706
153495
158686
163453
405841
Avg.
ATE
0.719
1.567
1.022
1.465
2.257
2.530
1.154
0.834
0.621
1.352
PSNR
23.28
23.14
24.68
25.10
24.98
23.00
22.97
22.16
24.56
23.76
SSIM
0.779
0.712
0.821
0.855
0.787
0.757
0.746
0.742
0.796
0.777
LPIPS
0.314
0.440
0.272
0.261
0.364
0.354
0.381
0.382
0.365
0.348
Table 5. Detailed results on Waymo dataset.
Scene
00
02
03
05
06
07
08
10
Avg.
ATE
0.413
3.957
5.190
2.030
2.628
0.732
2.278
2.542
2.471
PSNR
18.34
17.39
19.16
18.07
18.18
17.82
18.09
16.53
17.95
SSIM
0.671
0.549
0.559
0.612
0.624
0.654
0.629
0.540
0.605
LPIPS
0.354
0.506
0.532
0.464
0.469
0.441
0.422
0.587
0.472
Table 6. Detailed results on KITTI dataset.
Figure 10. Visual quality on Waymo dataset. Our approach consistently outperforms other models on different scenes, demonstrating
advantages in challenging scenarios. Best viewed in color.

<!-- page 14 -->
Figure 11. Visual quality on KITTI dataset. Our approach consistently outperforms other models on different scenes, demonstrating
advantages in challenging scenarios. Best viewed in color.
Figure 12. Visual quality on DL3DV dataset. Our approach consistently outperforms other models on different scenes, demonstrating
advantages in challenging scenarios. Best viewed in color.
Figure 13. Visual quality on Waymo dataset. Our approach consistently outperforms other models on different scenes, demonstrating
advantages in challenging scenarios. Best viewed in color.
