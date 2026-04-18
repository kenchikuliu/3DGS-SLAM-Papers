<!-- page 1 -->
IDESplat: Iterative Depth Probability Estimation for
Generalizable 3D Gaussian Splatting
Wei Long1, Haifeng Wu1, Shiyin Jiang1, Jinhua Zhang1, Xinchun Ji2, Shuhang Gu1*
1University of Electronic Science and Technology of China
2Aerospace Information Research Institute, Chinese Academy of Sciences
lwsch5940@163.com, shuhanggu@gmail.com
Iterative Warp
Depth
Candidates
Refinement
Multiplicative Boosting
View 1
View 0
Multiplicative Boosting
View 1
View 0
 Single Warp
Depth Probability
Depth Probability
of Iteration N
Depth Probability
of Iteration 1
View 1
View 0
Figure 1. Left: Methods [5, 21, 47] that estimate depth probability via a single warp operation. Middle: Our IDESplat can iteratively
leverage multi-warp operations to boost the depth probability estimate and refine the depth candidates for accurate Gaussian mean predictions.
Right: The experimental results of our IDESplat compared with mainstream methods such as PixelSplat [2], MVSplat [5], MonoSplat [21],
and DepthSplat [47]. The PSNR values are reported for the entire RE10K test set.
Abstract
Generalizable 3D Gaussian Splatting aims to directly pre-
dict Gaussian parameters using a feed-forward network for
scene reconstruction. Among these parameters, Gaussian
means are particularly difficult to predict, so depth is usually
estimated first and then unprojected to obtain the Gaussian
sphere centers. Existing methods typically rely solely on
a single warp to estimate depth probability, which hinders
their ability to fully leverage cross-view geometric cues, re-
sulting in unreliable and coarse depth maps. To address this
limitation, we propose IDESplat, which iteratively applies
warp operations to boost depth probability estimation for
accurate Gaussian mean prediction. First, to eliminate the
inherent unreliability of a single warp, we introduce a Depth
Probability Boosting Unit (DPBU) that integrates epipolar
attention maps produced by cascading warp operations in
a multiplicative manner. Next, we construct an iterative
depth estimation process by stacking multiple DPBUs, pro-
gressively identifying potential depth candidates with high
likelihood. As IDESplat iteratively boosts depth probability
estimates and updates the depth candidates, the depth map is
gradually refined, resulting in accurate Gaussian means. We
conduct experiments on RealEstate10K, ACID and DL3DV.
*corresponding author
IDESplat achieves outstanding reconstruction quality and
state-of-the-art performance with real-time efficiency. On
RE10K, it outperforms DepthSplat by 0.33 dB in PSNR, us-
ing only 10.7% of the parameters and 70% of the memory.
Additionally, our IDESplat improves PSNR by 2.95 dB over
DepthSplat on the DTU dataset in cross-dataset experiments,
demonstrating its strong generalization ability. The code is
available at https://github.com/CVL-UESTC/IDESplat.
1. Introduction
Single-scene 3D Gaussian Splatting (3DGS) benefits from
a rasterization-friendly pipeline, making it well-suited for
real-time scene reconstruction [16, 52], but it suffers from
limited generalization capabilities. Generalizable 3DGS [2,
5, 20, 47, 54] addresses this shortcoming by using a feed-
forward network to directly predict all Gaussian parameters,
enabling it to handle unseen scenes. Among all the Gaussian
sphere parameters, the Gaussian mean is crucial but difficult
to predict directly due to the local support nature of Gaussian
gradients [2], which significantly affects the optimization
of the overall parameters. Existing methods [5, 21, 47, 53]
commonly require first estimating the pixel-wise depth map
and then unprojecting it to obtain the centers of the Gaussian
spheres. This design reduces the difficulty of the network op-
1
arXiv:2601.03824v3  [cs.CV]  26 Mar 2026

<!-- page 2 -->
timization process by decoupling the prediction of Gaussian
mean parameters, while increasing the reliance on accurate
depth estimation.
Efforts have been made to obtain accurate and refined
depth estimation. Early methods [2, 34] directly use differen-
tiable operations to predict the depth probability distribution
from image features. Although these methods can predict
the depth map for scene reconstruction in a generalizable
way, their ability to exploit multi-view feature similarity
is limited, which restricts their performance. Subsequent
approaches [5, 20] introduce cost volumes that use warp
operations to establish feature similarity across views, which
provide valuable geometric cues for depth estimation and
simplify the network learning process. However, these meth-
ods rely solely on a single warp to model feature similarity,
which prevents them from fully exploiting the rich geometric
information, leading to unreliable and coarse depth maps.
Therefore, how to incorporate multiple warps to gradually
leverage rich cross-view geometric details, producing re-
fined and reliable depth maps for accurate Gaussian mean
prediction, remains a key challenge.
In this paper, we propose IDESplat, which iteratively
performs warps to refine depth maps for accurate Gaussian
means prediction. By integrating cascade warp results, we
can progressively boost the feature similarity measure to
identify high-likelihood surface points and suppress low-
probability depth candidates. The iterative warp framework
of our IDESplat is shown in Fig. 1. Firstly, to eliminate
the unreliability of a single warp, we introduce a Depth
Probability Boosting Unit (DPBU) to fuse multiple epipo-
lar attention maps in a multiplicative manner for a reliable
depth map. Next, we gradually update the depth search range
while increasing the feature resolution, enabling warp and
correlation calculations at a finer scale. Feature matching
becomes easier and more precise as the depth search range
is re-centered and image features are enhanced in this pro-
cess. Finally, for other Gaussian parameters, we propose a
Gaussian Focused Module that determines the most relevant
Gaussian tokens to compute attention weights for feature in-
teraction. The experimental results on RealEstate10K show
that IDESplat outperforms DepthSplat by 0.33 dB in PSNR
with only 10.7% of the parameters and 70% of the mem-
ory. Moreover, IDESplat shows outstanding cross-dataset
generalization, achieving 28.79 dB on ACID when trans-
ferred from RE10K, outperforming methods trained directly
on ACID. It also improves PSNR by 2.95 dB over Depth-
Splat on the DTU dataset in cross-dataset experiments. In
summary, the main contributions of this paper are as follows:
• We propose IDESplat, a generalizable feedforward 3DGS
model, which iteratively performs warps to progressively
boost the feature similarity measure and refine depth maps
for accurate Gaussian mean prediction.
• To eliminate the inherent unreliability of a single warp, we
introduce a Depth Probability Boosting Unit that multi-
plicatively integrates multiple epipolar attention maps for
reliable and refined depth map estimation.
• We design a Gaussian Focused Module to identify the most
relevant Gaussian tokens for computing attention scores
and reweight enhanced features.
• Experiments on RealEstate10K ACID, and DL3DV show
that our IDESplat significantly improves reconstruction
quality and generalization capability while maintaining
real-time inference efficiency.
2. Related Work
Generalizable 3D Gaussian Splatting. Single-scene 3D
Gaussian Splatting methods [8, 9, 16, 28, 48] enable more
efficient rendering than neural fields and volume rendering
methods [22, 26, 31, 41] thanks to their rasterization-friendly
formulation, but they still suffer from long optimization time
and limited generalization. To address this issue, general-
izable 3D Gaussian Splatting methods [2, 21, 35, 47] pre-
dict all Gaussian parameters in a single feed-forward pass,
enabling fast reconstruction and novel view synthesis for
unseen scenes. Early works [2, 34] directly regressed Gaus-
sian parameters from image features using feed-forward
networks. With the introduction of cost-volume construc-
tion into generalizable 3DGS, subsequent methods [5, 20]
exploited cross-view geometric cues to improve depth esti-
mation and simplify Gaussian parameter prediction. More
recently, methods [21, 47] further improved performance by
incorporating pre-trained monocular depth models. How-
ever, existing methods still rely on a single warp for depth
estimation, which limits their ability to fully exploit cross-
view feature cues. In contrast, IDESplat integrates feature
similarity from cascaded warps to produce more reliable and
refined depth maps for Gaussian mean prediction.
Iteration-based Optimization Methods. Iteration-based
optimization methods [1, 17] are widely used in various
tasks [3, 12, 56] due to their ability to progressively enhance
and refine the feature learning process. For instance, in opti-
cal flow estimation [13, 14, 33, 36, 49], iterative optimization
is commonly used to build coarse-to-fine pyramidal features
by stacking multiple feature units, resulting in more accurate
and stable flow estimation. In the field of depth estimation,
some works [6, 18, 25, 37, 38, 42, 43] have iteratively re-
trieved multi-view correlation features using structures like
GRUs and LSTMs with shared parameters to update the
disparity field, yielding better depth estimation results. Sim-
ilarly, in monocular depth estimation [10, 30, 57], iterative
methods are often used to gradually refine the depth search
range, leading to more stable and accurate predictions. Ad-
ditionally, in the 3D Gaussian Splatting (3DGS) domain,
recent methods [4, 11, 39, 46, 53] have also attempted to
design iterative optimization processes to reduce the diffi-
culty of 3D Gaussian reconstruction tasks. Unlike existing
2

<!-- page 3 -->
iterative optimization methods, we propose a novel Depth
Probability Estimation Unit that integrates the similarity
results of multiple warps in a multiplicative manner. This ap-
proach produces more reliable and refined depth maps, while
progressively enhancing feature resolution and refining the
depth candidate range throughout the iterative process.
3. Method
3.1. Preliminaries.
Given a sequence of V sparse-view images I = {Ii}V
i=1,
where each image Ii has dimensions H × W × 3, and the
corresponding camera projection matrices are Pi ∈R3×4.
The goal of the generalizable 3DGS task is to learn a network
that maps images to 3D Gaussian parameters. This process
is defined as:
fθ : {(Ii, Pi)}V
i=1 7→{(µj, αj, Σj, cj)}V ×H×W
j=1
,
(1)
where fθ is a feed-forward network with learnable parame-
ters θ. The parameters µj, αj, Σj, and cj are the Gaussian
mean, opacity, covariance, and color, respectively. Since
Gaussian means are difficult to predict directly, most ex-
isting methods estimate them by unprojecting a depth map
into 3D. The depth estimation scheme is as follows: First,
the input image sequence is processed using a multi-view
feature extraction backbone, resulting in a downsampled fea-
ture F ∈RV × H
4 × W
4 ×C. Then, the cost volume is computed
based on F , the camera projection matrices Pi, Pj for differ-
ent views, and the depth candidates G = [d1, d2, · · · , dD] ∈
RD. Specifically, the feature F j from view j is warped to
view i as follows:
F j→i = W(F j, Pi, Pj, G) ∈R
H
4 × W
4 ×D×C,
(2)
where W denotes the warp operator. The correlation is then
computed as the dot product between F i and F j→i
dm :
Ci
dm = (F i · F j→i
dm /
√
C) ∈R
H
4 × W
4 ,
(3)
where m ∈{1, . . . , D} and C denotes the feature channel
dimension. Subsequently, Ci = [Ci
d1, . . . , Ci
dD] is refined
with a U-Net and upsampled to the input resolution as ˜Ci ∈
RH×W ×D. Finally, a softmax function is applied to obtain
the probability of each candidate depth, and the final depth
map is computed as their weighted average.
To achieve high-quality 3D reconstruction, an accurate
depth map D is crucial since it determines the centers of 3D
Gaussians. However, this commonly used method only relies
on a single cross-view warp to compute feature similarity,
underutilizing cross-view geometry and often leading to
unreliable, coarse depth estimates. In addition, this typical
correlation computation method requires storing all the dense
warping features, which incurs significant memory overhead,
especially when the depth candidate size D is large.
3.2. Iterative Multi-View Depth Estimation
In this paper, we propose IDESplat to iteratively boost the
feature similarity measure using multiple warp operations.
As IDESplat progressively mines geometric cues to identify
high-probability potential depth candidates, it can produce
refined and reliable depth probability estimation results for
precise Gaussian mean prediction.
Warp-Index Epipolar Attention. The warp operation is
key to constructing the cost volume or epipolar attention
map, as it establishes pixelwise correspondences between
features in the source and target views. However, as shown in
Eq. (2), this warping operation requires sampling target-view
features for each depth candidate, thereby incurring high
memory cost. To alleviate the inherent memory overhead, we
introduce a Warp-Index Epipolar Attention mechanism that
only stores warp indices for similarity matrix multiplication.
We first compute the warping index map I as follows:
Ij→i = IW(F j, Pi, Pj, G) ∈R
H
4 × W
4 ×D,
(4)
where IW represents the operation that only records the in-
dices obtained during warping, and G is the depth candidate
matrix. Next, we compute the feature correlations in parallel
as follows:
Ci = Ψ(F i, F j, Ij→i) ∈R
H
4 × W
4 ×D,
(5)
where Ψ denotes the Sparse Matrix Multiplication (SMM),
which uses the warp index Ij→i to determine the position in
F j for matrix multiplication with F i. Then, we refine the
correlation map Ci with a lightweight 2D U-Net to obtain
˜Ci ∈RH×W ×D. Finally, a softmax is applied along the
depth dimension to obtain the attention weights:
Ai = softmax( ˜Ci).
(6)
This attention map Ai corresponds to the depth probability
results of view i for different depth candidates in a single
estimation.
Depth Probability Boosting Strategy. In each depth prob-
ability boosting unit, we stack M Warp-Index Epipolar At-
tention layers to produce M depth probability estimation
results. To combine these isolated estimated outputs for
stronger depth estimation capability, we propose a depth
probability boosting strategy. Specifically, we initialize the
depth probability matrix P0 as an all-ones matrix and com-
pute the subsequent updates as follows:
Pm = Norm(Pm−1 ⊙Am),
(7)
where m ∈{1, . . . , M} and Norm(·) denotes row-wise
normalization. Pm represents the depth probability matrix
generated by the m-th Warp-Index Epipolar Attention in the
current depth probability boosting unit. Depth candidates
with consistently high probabilities across layers will be
3

<!-- page 4 -->
Iterative
Refine 
Unproject
3D Gaussians
Render
Gaussian Focused Module
Feature Extraction 
Input Views
Novel View
Boosting
Depth Probability
Boosting Unit (1st)
Warp 0
Warp M
Boosting
Warp 0
Warp M
Depth Probability
Boosting Unit (N-th)
Depth Probability
of Iteration 1
Depth Probability
of Iteration N
Depth Candidate 
of  Iteration 1
Linear
Combination
Depth Candidate
 of  Iteration N
Linear
Combination
Figure 2. The overall architecture of IDESplat. IDESplat consists of three key parts: a feature extraction backbone, an iterative depth
probability estimation process, and a Gaussian Focused Module (GFM). The iterative process is built upon cascaded Depth Probability
Boosting Units (DPBUs). Within each DPBU, depth probabilities are estimated sequentially by stacked Warp-Index Epipolar Attention
(WIEA) blocks, where each block operates on progressively refined features and thus produces distinct probability estimates. Each unit
combines multi-level warp results in a multiplicative manner to mitigate the inherent unreliability of a single warp. As IDESplat iteratively
updates the depth candidates and boosts the probability estimates, the depth map becomes more precise, leading to accurate Gaussian means.
boosted through this cascaded element-wise product process.
As a result, the depth probability produced by the index-
based epipolar attention layer can be gradually enhanced,
becoming more reliable and accurate.
Iterative Depth Estimation Process. Based on the depth
probability boosting strategy, each Depth Unit produces an
enhanced depth-probability map PM,n at iteration n. To
refine depth estimates symmetrically around the current pre-
diction, we formulate the depth update using a relative depth-
candidate offset vector ∆Gn, which allows the network to
predict both positive and negative residuals. Specifically,
for the first iteration (n = 1), we uniformly sample depth
candidates within the initial range [dmin, dmax]. The depth-
candidate vector used for feature matching is defined as
G1 = [dmin + I1, dmin + 2I1, . . . , dmin + D1I1], where
I1 = (dmax −dmin)/(D1 + 1) denotes the sampling inter-
val and D1 is the number of candidates. Since the initial
depth map is set to D0 = 0, the relative offset vector in
the first iteration is defined as ∆G1 = G1. For subse-
quent iterations (n > 1), the depth search range is cen-
tered at the previous depth estimate Dn−1. Accordingly,
the depth-candidate vector is updated as Gn = [Dn−1 −
kIn, . . . , Dn−1, . . . , Dn−1+kIn], and the corresponding
relative offset vector is ∆Gn = [−kIn, . . . , 0, . . . , kIn].
Here, the interval shrinks as In = I1/n, enabling progres-
sively finer refinement over iterations, and the number of
candidates is Dn = 2k + 1. The residual depth map ∆Dn
is then computed as the weighted sum of the relative off-
sets ∆Gn with the probability map PM,n along the depth
dimension:
∆Dn = PM,n∆Gn.
(8)
The refined depth map is updated additively at each iteration:
Dn = Dn−1 + ∆Dn,
(9)
where n ∈{1, . . . , N}. DN is the final depth map of the
iterative depth estimation process, from which the Gaussian
mean parameters are obtained through unprojection. Enabled
by our memory-efficient Warp-Index Epipolar Attention,
IDESplat progressively increases the feature resolution over
iterations. In the final stage, IDESplat performs warping
and similarity computation at the original input resolution of
256 × 256 to produce the refined depth map.
3.3. Gaussian Focused Module
For the remaining Gaussian parameters, we introduce a
window-based Gaussian Focused Module that can filter ir-
relevant Gaussians and retains the most relevant tokens for
attention weight computation. The window-based attention
performs pairwise interactions for each Gaussian token, of-
ten including numerous irrelevant ones. This dense interac-
tion not only slows down the model but also introduces noise
into the attention results. Inspired by [23], we introduce
a Gaussian Focused Layer that reuses the previous layer’s
Gaussian correlation map to guide attention in the current
layer. Formally, for the Gaussian parameters of a given view,
G ∈RC×H×W , we apply three linear layers to obtain Q,
K, and V. We use a matrix IG to record the indices of
Gaussian tokens with high similarity. I0
G is initialized as an
all-ones matrix and is updated after each similarity compu-
tation. Then, we compute the Gaussian similarity map as
follows:
Sl = Ψ(Ql, Kl, Il−1),
(10)
4

<!-- page 5 -->
where Ψ denotes the SMM operation, and Sl is the similar-
ity map of the l-th Gaussian-Focused Layer. Subsequently,
we compute the sparse Gaussian attention map Al for the
current l-th layer as follows:
Al = S
 Norm
 Al−1 ⊙Softmax(Sl)

,
(11)
where S denotes the sparsification operation that retains the
top half of the weights in each row of the attention map. The
positions of the retained weights are recorded as Il
G, which
stores the selected highly similar Gaussian relations. Finally,
the output Gaussian features are reweighted as follows:
Ol = Ψ(Al, Vl, Il).
(12)
The Gaussian Focused Module consists of a series of Gaus-
sian Focused Layers connected in sequence. As the layer in-
dex l increases, Il
G becomes progressively sparser, gradually
identifying the Gaussian positions that are most important to
each query location. This module leverages token similarity
across layers to filter out the influence of irrelevant Gaus-
sian features, achieving relational and sufficiently enriched
Gaussian feature interactions.
4. Experiments
4.1. Experiment Setting
Datasets. Our experiments are conducted on three large-
scale datasets:
RealEstate10K [55], ACID [19] and
DTU [15]. RealEstate10K contains real estate videos down-
loaded from YouTube, which are split into 67,477 training
scenes and 7,289 testing scenes. ACID consists of nature
scenes captured via aerial drones, with 11,075 scenes for
training and 1,972 scenes for testing. Both datasets are
calibrated using the Structure-from-Motion (SfM) [29] al-
gorithm to estimate the camera’s intrinsic and extrinsic pa-
rameters for each frame. Following the settings of previ-
ous works [2, 5, 21, 47], for the RealEstate10K and ACID
datasets, two context images are used as input, and three
novel target views are rendered for each test scene. All input
and target images have a resolution of 256 × 256. In addi-
tion, for the multi-view DTU [15] dataset, which contains
object-centric scenes with known camera poses, we report
results on 16 validation scenes.
Implementation details. For a fair comparison, we followed
the commonly used training setup [5, 21, 47]. Specifically,
our training experiments were conducted on 8 RTX 4090
GPUs with a total batch size of 16, using the AdamW [24]
optimizer for 300,000 iterations. We employed a cosine
learning rate schedule to optimize our model. For the pre-
trained Depth Anything V2 [50] backbone, we used a learn-
ing rate of 2×10−6, while the remaining layers were trained
with a learning rate of 2 × 10−4. Following DepthSplat [47],
we train our model with MSE and LPIPS losses.
4.2. Main Results
We comprehensively compare our proposed method with
several leading approaches in scene-level novel view synthe-
sis, covering three representative categories: light field net-
work methods such as GPNR [32] and AttnRend [7]; NeRF-
based methods including pixelNeRF [51] and MuRF [45];
and 3D Gaussian Splatting-based methods such as pixel-
Splat [2], latentSplat [40], MVSplat [5], eFreeSplat [27],
MonoSplat [21], and DepthSplat [47].
Quantitative results. As shown in Tab. 1, our IDESplat
achieves state-of-the-art performance on all visual quality
metrics in both the RealEstate10K and ACID benchmarks.
Specifically, on the RE10K dataset, compared to DepthSplat,
our method improves PSNR by 0.33 dB, while using only
10.7% of its parameters. Additionally, compared to MonoS-
plat, which has a similar number of parameters, our method
achieves a significant 1.12 dB improvement. For SSIM
and LPIPS metrics, our method also achieves the best re-
sults of 0.893 and 0.108, respectively. On the ACID dataset,
our method achieves a 0.31 dB improvement in PSNR over
MonoSplat, reaching a maximum of 28.94 dB. The superior
performance of IDESplat can be attributed to its iterative
boosting design for depth probability prediction, which grad-
ually refines the depth map. This design allows our method
to achieve reliable and stable depth estimation results with
fewer parameters, enabling accurate Gaussian mean parame-
ter prediction and high-quality scene reconstruction.
Cross-Dataset Generalization. To evaluate the generaliza-
tion ability of our proposed IDESplat, we conducted cross-
dataset generalization tests on unseen datasets. We first
trained the model on the indoor scene dataset RealEstate10K,
and then evaluated it directly on the outdoor scene ACID
dataset and the object-centered DTU dataset. As shown
in Tab. 2, IDESplat demonstrates outstanding cross-dataset
generalization ability, outperforming existing methods on
all metrics. For the DTU dataset, our method improves
PSNR by 2.95 dB over DepthSplat, and achieves the best
LPIPS score of 0.239. On the ACID dataset, our method
also achieves the highest performance with 28.79 dB. These
results show that our iterative depth probability estimation
method effectively enhances the modeling of cross-view
feature similarity, learning strong out-of-distribution scene
reconstruction capabilities. This also demonstrates that our
iterative depth probability estimation method is both effi-
cient and highly generalizable, surpassing dataset-specific
features and avoiding overfitting to the training data.
Visual comparison results. We performed a qualitative com-
parison of scene reconstruction results between our method
and mainstream models, including MVSplat [5], MonoS-
plat [21], and DepthSplat [47]. The visual comparison re-
sults are shown in Fig. 3, which include both indoor and
outdoor scenes. Our proposed IDESplat achieves signifi-
cantly better novel view synthesis results compared to exist-
5

<!-- page 6 -->
Table 1. Quantitative comparisons. We surpass all baseline methods in terms of PSNR, LPIPS, and SSIM for novel view synthesis on the
real-world RealEstate10k [55] and ACID [19] datasets. We highlight first-place results in bold and second-place results with underlines in
each column. ”-” Indicates that the original paper did not contain relevant data.
Method
Params (M) ↓
RealEstate10k [55]
ACID [19]
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Du et al.[7]
125.1
24.78
0.820
0.213
26.88
0.799
0.218
GPNR[32]
9.6
24.11
0.793
0.255
25.28
0.764
0.332
pixelNeRF [51]
28.2
20.43
0.589
0.550
20.97
0.547
0.533
MuRF [45]
5.3
26.10
0.858
0.143
28.09
0.841
0.155
pixelSplat [2] (CVPR 2024)
125.4
26.09
0.863
0.136
28.27
0.843
0.146
latentSplat [40] (ECCV 2024)
187.0
23.07
0.825
0.182
24.95
0.782
0.207
MVSplat [5] (ECCV 2024)
12.0
26.39
0.869
0.128
28.25
0.843
0.144
eFreeSplat [27] (NIPS 2024)
-
26.45
0.865
0.126
28.30
0.851
0.140
MonoSplat [21] (CVPR 2025)
30.3
26.68
0.875
0.123
28.63
0.864
0.138
DepthSplat [47] (CVPR 2025)
354
27.47
0.889
0.114
-
-
-
IDESplat (Ours)
37.6
27.80
0.893
0.108
28.94
0.866
0.130
MVSplat
DepthSplat
MonoSplat
IDESplat (Ours)
Input
Reference
Figure 3. The comparison of visualization results for novel view synthesis on the RealEstate10K dataset. Our IDESplat significantly
outperforms previous state-of-the-art methods in rendering challenging regions.
6

<!-- page 7 -->
Table 2. Quantative comparisons of cross-dataset generalization.
We perform zero-shot tests on ACID [19] and DTU [15] datasets,
using models trained solely on RealEstate10K [55]. Best and
second best results are bolded and underlined.
Method
RE10k→DTU
RE10k→ACID
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
pixelSplat [2]
12.89
0.382
0.560
27.64
0.830
0.160
MVSplat [5]
13.94
0.473
0.385
28.15
0.841
0.147
MonoSplat [21]
15.25
0.605
0.291
28.24
0.848
0.145
Depthsplat [47]
15.38
0.415
0.442
28.37
0.847
0.141
IDESplat (Ours)
18.33
0.719
0.239
28.79
0.853
0.135
ing mainstream open-source models. Specifically, in chal-
lenging areas with rich textures and large position and angle
differences between the input images, existing methods often
produce noticeable artifacts and blurring in the synthesized
views. In contrast, IDESplat is able to generate high-quality
reconstructions in these difficult regions, demonstrating ex-
cellent texture consistency and clearly showing its superior-
ity in reconstructing complex scenes.
Visual comparison of depth maps. We conducted a direct
qualitative comparison of depth maps predicted by IDESplat
and existing state-of-the-art models, including MVSplat [5],
MonoSplat [21], and DepthSplat [47]. It is important to note
that the generalizable 3DGS task typically does not have
depth ground truth, so we analyzed the results using the
corresponding reference RGB images as input. As shown in
Fig. 5, IDESplat consistently outperforms existing methods
in both indoor and outdoor scenes. Even in regions with sim-
ilar foreground and background appearances or rich texture
details, it can better distinguish objects at different depths
and produce more realistic depth estimates with finer details.
This advantage comes from aggregating feature similarity
across multiple warps via the depth probability boosting
strategy, as well as performing warp computation at the orig-
inal scale to better preserve fine textures. More visual results
are provided in the supplementary materials.
Comparison of model efficiency. In Tab. 3, we compare
the model efficiency of IDESplat with several classic meth-
ods, including PixelSplat, MVSplat, and the latest methods
such as MonoSplat and DepthSplat. We measure inference
time and memory usage with two-view inputs at a resolu-
tion of 256 × 256, and the PSNR values are reported on
the commonly used benchmark dataset RE10K. Although
our method shows slightly lower inference efficiency com-
pared to DepthSplat, it outperforms DepthSplat on all other
metrics. Despite using significantly fewer parameters and
computational resources than DepthSplat, IDESplat achieves
better results. The superior performance of IDESplat with
fewer parameters and better memory efficiency is due to
our iterative depth probability estimation architecture, which
performs multiple warp operations to improve the accuracy
of Gaussian mean parameter predictions.
Depth estimation experiment on ScanNet. We report
Table 3. Comparison of model efficiency. Our method demon-
strates relatively low inference costs and reduced memory usage.
Method
Params (M)
Mem. (M)
Time (s)
PSNR↑
pixelSplat [2]
125.4
4108
0.120
26.09
MVSplat [5]
12.0
1940
0.054
26.39
MonoSplat [21]
30.3
1606
0.062
26.68
Depthsplat [47]
354
3342
0.082
27.47
IDESplat (Ours)
37.6
2336
0.110
27.80
DepthSplat
Reference
UniMatch
IDESplat (Ours)
Figure 4. Qualitative comparison of depth error maps on the Scan-
Net dataset.
depth estimation results on the ScanNet dataset in Table 4.
IDESplat consistently outperforms UniMatch and Depth-
Splat across standard quantitative depth estimation metrics.
For qualitative comparison, we further visualize the depth
error maps on the ScanNet dataset in Fig. 4. Compared
with the baseline methods, IDESplat exhibits fewer large-
error regions and produces more accurate depth estimates,
especially around geometrically complex areas.
Table 4. Depth estimation results on the ScanNet dataset.
Method
Abs Rel ↓
RMSE ↓
RMSElog ↓
UniMatch
0.059
0.179
0.082
DepthSplat
0.045
0.125
0.061
IDESplat
0.039
0.116
0.053
4.3. Ablation Study
We conducted detailed ablation experiments on various com-
ponents of the proposed IDESplat. All models were trained
for 20,000 iterations on the RealEstate10K dataset with a
batch size of 8. We analyzed the effectiveness of the Gaus-
sian Focused Module (GFM), Iterative Depth Estimation
process (IDE), and Depth Probability Boosting Strategy
(DPBS) , while keeping all other training settings consis-
tent. It should be noted that IDE(3) indicates the model
iterates the depth probability boosting unit three times. The
detailed experimental results, shown in Tab. 5, demonstrate
that all three designs contribute to improved reconstruction
performance. Specifically, adding GFM and IDE(3) results
in improvements of 0.32 dB and 0.57 dB, respectively, show-
ing that both the iterative depth estimation process and the
Gaussian focused module effectively enhance scene recon-
struction. The performance improves significantly by 0.46
dB when we incorporate the DPBS strategy into the iterative
process, and the best result of 27.56 dB is achieved when
all three strategies are used together, further validating the
7

<!-- page 8 -->
MVSplat
DepthSplat
MonoSplat
IDESplat (Ours)
View 1
View 0
Figure 5. Comparison of depth prediction maps for different models on the RE10K dataset.
Table 5. Ablation study results for each component of IDESplat.
All results are reported on the RealEstate10K dataset. GFM denotes
the Gaussian Focused Module, IDE(3) denotes the Iterative Depth
Estimation process with three iterations, and DPBS denotes the
Depth Probability Boosting Strategy.
Method
PSNR↑
SSIM↑
LPIPS↓
Baseline
26.31
0.866
0.129
+ GFM
26.63
0.875
0.124
+ IDE(3)
26.88
0.878
0.120
+ IDE(3) + GFM
27.07
0.882
0.118
+ IDE(3) + DPBS
27.34
0.887
0.112
Full Model
27.56
0.889
0.110
effectiveness of our proposed method.
We conduct ablation experiments to evaluate the effi-
ciency and reconstruction performance of IDESplat under
different iteration counts. In these experiments, the Gaussian
Focused Module and Depth Probability Boosting Strategy
were used by default. Additionally, during the iterative pro-
cess, we gradually increased the resolution, and the depth
search range was progressively halved. For 3 iterations, the
feature resolutions during warping were 1
4, 1
2, and 1 of the
original resolution. For 4 iterations, the model iterates twice
at the largest feature size. When the number of iterations is
0, it represents performing a single warp for depth probabil-
ity estimation. The results of the experiment, as shown in
Tab. 6, indicate that even with a single iteration, our method
provides a 0.45 dB improvement. With 3 iterations, the
performance improves by 0.93 dB. Increasing the iteration
count adds little parameter overhead and keeps memory us-
age comparable to existing methods. Despite the longer
inference time, 3 iterations already provide a substantial gain
while maintaining real-time inference.
5. Conclusion
We propose IDESplat, which iteratively applies warp oper-
ations and integrates multi-level epipolar attention maps to
Table 6. Ablation results for IDESplat with different numbers
of iterative depth probability boosting units. All results are
reported on the RealEstate10K dataset. Here, an iteration count of
0 denotes using a single warp operation without any DPBUs, while
the other entries correspond to using different numbers of DPBUs.
Iterations Params (M) Mem. (M) Time (s) PSNR↑SSIM↑LPIPS↓
0
35.4
1674
0.056
26.63
0.875
0.124
1
36.8
1734
0.071
27.08
0.882
0.113
2
37.3
1902
0.091
27.31
0.884
0.112
3
37.6
2336
0.110
27.56
0.889
0.110
4
38.0
2745
0.132
27.64
0.890
0.109
enhance depth probability estimation for accurate Gaussian
mean prediction. Specifically, we first design a depth proba-
bility boosting unit to amplify cross-view feature similarity
in a multiplicative manner. Then, we build an iterative depth
estimation process by stacking multiple DPBUs, gradually
identifying the most likely depth locations. Additionally,
during this iterative process, we progressively narrow the
depth search range and increase feature size to achieve more
precise depth estimates. For the other Gaussian parameters,
we design a gaussian focused module to select the most rele-
vant Gaussian tokens for attention-based feature interaction.
Experimental results on large-scale benchmarks clearly show
that IDESplat achieves excellent reconstruction quality and
strong domain generalization ability. Our current model still
has limitations: although it enables real-time 3D reconstruc-
tion, its efficiency could be further improved, and it requires
camera pose input as a prerequisite. Potential directions for
improvement include further enhancing inference speed and
exploring a pose-free framework.
Acknowledgement
This work was supported by the National Natural Science
Foundation of China under Grant No. 62476051.
8

<!-- page 9 -->
References
[1] Jonas Adler and Ozan ¨Oktem. Solving ill-posed inverse prob-
lems using iterative deep neural networks. Inverse Problems,
33(12):124007, 2017. 2
[2] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. pixelsplat: 3d gaussian splats from image
pairs for scalable generalizable 3d reconstruction. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 19457–19467, 2024. 1, 2, 5, 6, 7
[3] Changrui Chen, Jungong Han, and Kurt Debattista. Virtual
category learning: A semi-supervised learning method for
dense prediction with extremely limited labels. IEEE trans-
actions on pattern analysis and machine intelligence, 46(8):
5595–5611, 2024. 2
[4] Yun Chen, Jingkang Wang, Ze Yang, Sivabalan Manivasagam,
and Raquel Urtasun. G3r: Gradient guided generalizable
reconstruction. In European Conference on Computer Vision,
pages 305–323. Springer, 2024. 2
[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In European Conference on Computer
Vision, pages 370–386. Springer, 2024. 1, 2, 5, 6, 7
[6] Ziyang Chen, Wei Long, He Yao, Yongjun Zhang, Bingshu
Wang, Yongbin Qin, and Jia Wu. Mocha-stereo: Motif chan-
nel attention network for stereo matching. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 27768–27777, 2024. 2
[7] Yilun Du, Cameron Smith, Ayush Tewari, and Vincent Sitz-
mann. Learning to render novel views from wide-baseline
stereo pairs. In CVPR, 2023. 5, 6
[8] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang,
Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic,
Marco Pavone, Georgios Pavlakos, et al.
Instantsplat:
Sparse-view gaussian splatting in seconds. arXiv preprint
arXiv:2403.20309, 2024. 2
[9] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang,
Tao Liu, Boni Hu, Linning Xu, Zhilin Pei, Hengjie Li, et al.
Flashgs: Efficient 3d gaussian splatting for large-scale and
high-resolution rendering. In Proceedings of the Computer
Vision and Pattern Recognition Conference, pages 26652–
26662, 2025. 2
[10] Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Bat-
manghelich, and Dacheng Tao. Deep ordinal regression net-
work for monocular depth estimation. In Proceedings of the
IEEE conference on computer vision and pattern recognition,
pages 2002–2011, 2018. 2
[11] James Harrison, Luke Metz, and Jascha Sohl-Dickstein. A
closer look at learned optimization: Stability, robustness, and
inductive biases. Advances in neural information processing
systems, 35:3758–3773, 2022. 2
[12] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li,
Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and Ying-
Cong Chen.
Lotus:
Diffusion-based visual foundation
model for high-quality dense prediction.
arXiv preprint
arXiv:2409.18124, 2024. 2
[13] Tak-Wai Hui, Xiaoou Tang, and Chen Change Loy. Lite-
flownet: A lightweight convolutional neural network for opti-
cal flow estimation. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 8981–8989,
2018. 2
[14] Eddy Ilg, Nikolaus Mayer, Tonmoy Saikia, Margret Keu-
per, Alexey Dosovitskiy, and Thomas Brox. Flownet 2.0:
Evolution of optical flow estimation with deep networks. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 2462–2470, 2017. 2
[15] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola,
and Henrik Aanæs. Large scale multi-view stereopsis evalua-
tion. In CVPR, 2014. 5, 7
[16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and
George Drettakis. 3d gaussian splatting for real-time radiance
field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 1, 2
[17] Yi Li, Gu Wang, Xiangyang Ji, Yu Xiang, and Dieter Fox.
Deepim: Deep iterative matching for 6d pose estimation. In
Proceedings of the European conference on computer vision
(ECCV), pages 683–698, 2018. 2
[18] Lahav Lipson, Zachary Teed, and Jia Deng. Raft-stereo:
Multilevel recurrent field transforms for stereo matching. In
2021 International Conference on 3D Vision (3DV), pages
218–227. IEEE, 2021. 2
[19] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Maka-
dia, Noah Snavely, and Angjoo Kanazawa. Infinite nature:
Perpetual view generation of natural scenes from a single
image. In ICCV, 2021. 5, 6, 7
[20] Tianqi Liu, Guangcong Wang, Shoukang Hu, Liao Shen,
Xinyi Ye, Yuhang Zang, Zhiguo Cao, Wei Li, and Ziwei Liu.
Mvsgaussian: Fast generalizable gaussian splatting recon-
struction from multi-view stereo. In European Conference on
Computer Vision, pages 37–53. Springer, 2024. 1, 2
[21] Yifan Liu, Keyu Fan, Weihao Yu, Chenxin Li, Hao Lu, and
Yixuan Yuan. Monosplat: Generalizable 3d gaussian splatting
from monocular depth foundation models. In Proceedings
of the Computer Vision and Pattern Recognition Conference,
pages 21570–21579, 2025. 1, 2, 5, 6, 7
[22] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel
Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural
volumes: Learning dynamic renderable volumes from images.
arXiv preprint arXiv:1906.07751, 2019. 2
[23] Wei Long, Xingyu Zhou, Leheng Zhang, and Shuhang
Gu. Progressive focused transformer for single image super-
resolution. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 2279–2288, 2025. 4
[24] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization. arXiv preprint arXiv:1711.05101, 2017. 5, 1
[25] Zeyu Ma, Zachary Teed, and Jia Deng. Multiview stereo with
cascaded epipolar raft. In European Conference on Computer
Vision, pages 734–750. Springer, 2022. 2
[26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
9

<!-- page 10 -->
[27] Zhiyuan Min, Yawei Luo, Jianwen Sun, and Yi Yang.
Epipolar-free 3d gaussian splatting for generalizable novel
view synthesis. arXiv preprint arXiv:2410.22817, 2024. 5, 6
[28] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakoto-
saona, Michael Oechsle, Daniel Duckworth, Rama Gosula,
Keisuke Tateno, John Bates, Dominik Kaeser, and Federico
Tombari. Radsplat: Radiance field-informed gaussian splat-
ting for robust real-time rendering with 900+ fps. In 2025
International Conference on 3D Vision (3DV), pages 134–144.
IEEE, 2025. 2
[29] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited. In Proceedings of the IEEE confer-
ence on computer vision and pattern recognition, pages 4104–
4113, 2016. 5
[30] Shuwei Shao, Zhongcai Pei, Xingming Wu, Zhong Liu, Wei-
hai Chen, and Zhengguo Li. Iebins: Iterative elastic bins for
monocular depth estimation. Advances in Neural Information
Processing Systems, 36:53025–53037, 2023. 2
[31] Vincent Sitzmann, Julien Martel, Alexander Bergman, David
Lindell, and Gordon Wetzstein. Implicit neural representa-
tions with periodic activation functions. Advances in neural
information processing systems, 33:7462–7473, 2020. 2
[32] Mohammed Suhail, Carlos Esteves, Leonid Sigal, and
Ameesh Makadia. Generalizable patch-based neural render-
ing. In ECCV, 2022. 5, 6
[33] Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz.
Pwc-net: Cnns for optical flow using pyramid, warping, and
cost volume.
In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 8934–8943,
2018. 2
[34] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea
Vedaldi. Splatter image: Ultra-fast single-view 3d recon-
struction. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 10208–10217,
2024. 2
[35] Shengji Tang, Weicai Ye, Peng Ye, Weihao Lin, Yang Zhou,
Tao Chen, and Wanli Ouyang. Hisplat: Hierarchical 3d gaus-
sian splatting for generalizable sparse-view reconstruction.
arXiv preprint arXiv:2410.06245, 2024. 2
[36] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field
transforms for optical flow. In European conference on com-
puter vision, pages 402–419. Springer, 2020. 2
[37] Fangjinhua Wang, Silvano Galliani, Christoph Vogel, and
Marc Pollefeys. Itermvs: Iterative probability estimation for
efficient multi-view stereo. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages
8606–8615, 2022. 2
[38] Shaoqian Wang, Bo Li, and Yuchao Dai. Efficient multi-view
stereo by iterative dynamic cost volume. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 8655–8664, 2022. 2
[39] Jing Wen, Alex Schwing, and Shenlong Wang. Life-gom:
Generalizable human rendering with learned iterative feed-
back over multi-resolution gaussians-on-mesh. In The Thir-
teenth International Conference on Learning Representations.
2
[40] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, and
Jan Eric Lenssen. latentsplat: Autoencoding variational gaus-
sians for fast generalizable 3d reconstruction. arXiv preprint
arXiv:2403.16292, 2024. 5, 6
[41] Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany,
Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin,
Vincent Sitzmann, and Srinath Sridhar. Neural fields in visual
computing and beyond. In Computer graphics forum, pages
641–676. Wiley Online Library, 2022. 2
[42] Gangwei Xu, Xianqi Wang, Xiaohuan Ding, and Xin Yang.
Iterative geometry encoding volume for stereo matching. In
Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 21919–21928, 2023. 2
[43] Gangwei Xu, Xianqi Wang, Zhaoxing Zhang, Junda Cheng,
Chunyuan Liao, and Xin Yang. Igev++: Iterative multi-range
geometry encoding volumes for stereo matching. IEEE Trans-
actions on Pattern Analysis and Machine Intelligence, 2025.
2
[44] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi,
Fisher Yu, Dacheng Tao, and Andreas Geiger.
Unifying
flow, stereo and depth estimation. IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, 45(11):13941–13958,
2023. 1
[45] Haofei Xu, Anpei Chen, Yuedong Chen, Christos Sakaridis,
Yulun Zhang, Marc Pollefeys, Andreas Geiger, and Fisher Yu.
Murf: Multi-baseline radiance fields. In CVPR, 2024. 5, 6
[46] Haofei Xu, Daniel Barath, Andreas Geiger, and Marc Polle-
feys. Resplat: Learning recurrent gaussian splats. arXiv
preprint arXiv:2510.08575, 2025. 2
[47] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum,
Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depth-
splat: Connecting gaussian splatting and depth. In Proceed-
ings of the Computer Vision and Pattern Recognition Confer-
ence, pages 16453–16463, 2025. 1, 2, 5, 6, 7
[48] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang,
Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou,
and Sida Peng. Street gaussians: Modeling dynamic urban
scenes with gaussian splatting. In European Conference on
Computer Vision, pages 156–173. Springer, 2024. 2
[49] Gengshan Yang and Deva Ramanan. Volumetric correspon-
dence networks for optical flow. Advances in neural informa-
tion processing systems, 32, 2019. 2
[50] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-
gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything
v2. Advances in Neural Information Processing Systems, 37:
21875–21911, 2024. 5, 1
[51] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.
pixelnerf: Neural radiance fields from one or few images. In
Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 4578–4587, 2021. 5, 6
[52] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 19447–19456,
2024. 1
[53] Chuanrui Zhang, Yingshuang Zou, Zhuoling Li, Minmin Yi,
and Haoqian Wang. Transplat: Generalizable 3d gaussian
10

<!-- page 11 -->
splatting from sparse multi-view images with transformers. In
Proceedings of the AAAI Conference on Artificial Intelligence,
pages 9869–9877, 2025. 1, 2
[54] Shengjun Zhang, Xin Fei, Fangfu Liu, Haixu Song, and
Yueqi Duan. Gaussian graph network: Learning efficient
and generalizable gaussian representations from multi-view
images. Advances in Neural Information Processing Systems,
37:50361–50380, 2024. 1
[55] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
and Noah Snavely. Stereo magnification: learning view syn-
thesis using multiplane images. TOG, page 65, 2018. 5, 6,
7
[56] Lanyun Zhu, Tianrun Chen, Jianxiong Yin, Simon See, and
Jun Liu. Addressing background context bias in few-shot
segmentation through iterative modulation. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 3370–3379, 2024. 2
[57] Yiming Zuo and Jia Deng. Ogni-dc: Robust depth comple-
tion with optimization-guided neural iterations. In European
Conference on Computer Vision, pages 78–95. Springer, 2024.
2
11

<!-- page 12 -->
IDESplat: Iterative Depth Probability Estimation for
Generalizable 3D Gaussian Splatting
Supplementary Material
In this supplementary material, we provide additional
details on model training, model architecture, ablation study
on DPBU, experimental results on the DL3DV dataset, and
more visual comparison results. Specifically, in Section
A, we present the training details of the IDESplat model.
In Section B, we provide the model architecture details of
IDESplat and Warp-Index Epipolar Attention. In Section
C, we present the experimental results of IDESplat on the
DL3DV dataset, along with the ablation study on Depth
Probability Boosting Units. Finally, in Section D, we include
more visual comparison results for novel view synthesis and
depth prediction.
A. Training Details
For a fair comparison, we trained our IDESplat using a
standard setup [5, 21, 47]. The training was done on 8 RTX
4090 GPUs with a batch size of 16, using the AdamW [24]
optimizer for 300,000 iterations, which took approximately
3 days. We used a cosine learning rate schedule. For the
pre-trained Depth Anything V2 [50] backbone, the learning
rate was 2 × 10−6, while other layers were trained with a
learning rate of 2 × 10−4. The network was trained with a
combination of MSE and LPIPS losses between the rendered
and ground truth images. Following [47], for the newly
added DL3DV dataset, we trained at a resolution of 256 ×
448. First, we pre-trained on RE10k and then fine-tuned on
the DL3DV dataset for 100K iterations, with a total batch
size of 4, and the number of input views was randomly
sampled from 2 to 6. During inference, we evaluated the
model’s performance on different numbers of input views.
B. Model Architecture Details
We provide a detailed description of the IDESplat network
architecture, as shown in Figure 6. It consists of three main
parts: a feature extraction backbone, an iterative depth esti-
mation process, and a Gaussian focus module. The backbone
has two branches: a multi-view branch using the pre-trained
Unimatch [44] and a monocular branch using the ViT-small
version of DepthAnything V2 [50]. The outputs from both
branches are fused to provide multi-view geometry and tex-
ture information for the next modules. The depth estimation
process includes three Depth Probability Boosting Units
(DPBU) that sequentially generate optimized depth results.
Each DPBU contains two cascaded Warp-Index Epipolar
Attention layers, which use the Hadamard product to en-
hance depth probabilities. The process is repeated for six
transformations at resolutions of 64 × 64, 128 × 128, and
256×256. The GFM has six layers, using a shifting window
strategy with a window size of 16. After each attention cal-
culation, the top half of the most relevant Gaussian positions
are retained. The number of retained Gaussian weights per
layer is [256, 256, 128, 128, 64, 64], and the module uses 6
attention heads with 256 channels in total.
To address the memory issues in existing warp compu-
tations for cross-view similarity, we introduce Warp-Index
Epipolar Attention. Unlike the existing method, which sam-
ples target view features for each depth candidate and con-
sumes a lot of memory, our approach only stores transfor-
mation indices for similarity matrix multiplication and uses
Sparse Matrix Multiplication (SMM) for efficient compu-
tation. This design enables IDESplat to perform multiple
rounds of warp and depth estimation more efficiently.
C. More Experimental Results
We conducted additional experiments on the DL3DV dataset
to further evaluate the proposed IDESplat method. DL3DV
is a large-scale real-world multi-view video dataset, which
helps validate our method’s reconstruction capability in more
complex and larger scenes. The experimental results, shown
in Table 7, demonstrate outstanding performance of our
method compared to existing MVSplat and DepthSplat meth-
ods. IDESplat outperforms DepthSplat by 0.62dB, 0.41dB,
and 0.42dB when using 2, 4, and 6 input views, respectively.
These results clearly show that our IDESplat provides better
reconstruction performance in large, complex scenes with
multiple input views compared to existing methods.
We also conducted ablation experiments on DPBU with
different numbers of Warp-Index Epipolar Attention. The
results in Table 8 show that depth probabilities can only
undergo Multiplicative Boosting when more than one Warp-
Index Epipolar Attention is used. Our IDESplat performs
well when the number of attention layers exceeds one, and
the performance improves as the number of layers increases.
Considering both efficiency and performance, we chose the
model with two Warp-Index Epipolar Attention layers.
D. More Visual Comparison Results
We also provide more visual comparison results in this sup-
plementary material, as shown in Fig. 8 and Fig. 9. Through
these qualitative visual comparisons, it can be observed that
our IDESplat outperforms existing methods in novel view
synthesis. Even in complex lighting and textured regions,
1

<!-- page 13 -->
Multi-View
Transformer
Monocular
Depth model
: Attention map
: Depth probability
: Depth map
: Hadamard product
: Query gaussian
Depth candidate
range of        
Depth candidate
range of        
Depth candidate
range of        
Gaussian
Focused
Module
Unproject
3D Gaussians
Render
SMM
Attention
Warp-Index
 Epipolar
Attention
Warp-Index
 Epipolar
Attention
Depth Probability Boosting Unit
LayerNorm
LayerNorm
MLP
Gaussian Focused Module
Select
SMM
Attention
Previous attention map
Current attention map
: Add
SMM : Sparse matrix multiplication 
Figure 6. The architecture of IDESplat. IDESplat comprises a feature extraction backbone, an iterative depth estimation process with
cascaded Depth Probability Boosting Units (DPBUs), and a Gaussian Focused Module (GFM).
Q
K
V
softmax
O
Q
K
V
softmax
O
SMM
SMM
(a): Epipolar Attention
(b): Warp-Index Epipolar Attention
Depth
Probability
Depth
Probability
Figure 7. The difference between Warp-Index Epipolar Attention
and Epipolar Attention.
our method achieves better reconstruction results. Further-
more, in Fig. 10 and Fig. 11, we provide more qualitative
depth map comparisons. The results show that our method
significantly improves both the consistency and fine texture
details of the depth maps, whether in indoor or outdoor en-
vironments, compared to existing methods. To show how
IDESplat refines depth maps over iterations, we compare
Table 7. Quantitative Experimental Results and Comparisons
on DL3DV. Our IDESplat consistently outperforms MVSplat and
DepthSplat across different numbers of input views.
Method
#Views
PSNR ↑
SSIM ↑
LPIPS ↓
MVSplat [5]
2
17.54
0.529
0.402
DepthSplat [47]
19.31
0.615
0.310
IDESplat
19.93
0.635
0.300
MVSplat [5]
4
21.63
0.721
0.233
DepthSplat [47]
23.12
0.780
0.178
IDESplat
23.53
0.789
0.176
MVSplat [5]
6
22.93
0.775
0.193
DepthSplat [47]
24.19
0.823
0.147
IDESplat
24.61
0.829
0.146
Table 8. Ablation results for DPBU with different numbers of
Warp-Index Epipolar Attention. All results are reported on the
RealEstate10K dataset.
Number of WIEA Params (M) Mem. (M) Time (s) PSNR↑SSIM↑LPIPS↓
1
36.4
2033
0.082
27.11
0.882
0.116
2
37.6
2336
0.110
27.56
0.889
0.110
3
38.9
2642
0.139
27.65
0.890
0.109
4
40.2
2954
0.172
27.72
0.893
0.107
results visually in Fig. 12. Each iteration represents one pass
through the Depth Probability Boosting Unit. We can see that
with more steps, the depth map gets better and more detailed.
After 3 steps, the model can work at the original image size
and the depth map is clearer. With more iterations, the depth
map improves. This leads to more accurate Gaussian centers,
which creates a better scene reconstruction.
2

<!-- page 14 -->
MVSplat
DepthSplat
MonoSplat
IDESplat (Ours)
Input
Reference
Figure 8. The comparison of visualization results for novel view synthesis on the RealEstate10K dataset.
3

<!-- page 15 -->
DepthSplat
IDESplat (Ours)
Reference
Figure 9. The comparison of visualization results for novel view synthesis on the DL3DV dataset.
4

<!-- page 16 -->
MVSplat
DepthSplat
MonoSplat
IDESplat (Ours)
View 0
View 1
Figure 10. Comparison of depth prediction maps for different models on the RE10K dataset.
5

<!-- page 17 -->
DepthSplat
IDESplat (Ours)
Reference
Figure 11. Comparison of depth prediction maps for different models on the DL3DV dataset.
6

<!-- page 18 -->
Iteration 1
Iteration 2
Iteration 3
Reference
Figure 12. Visualization of intermediate depth prediction maps at different iterations in the IDESplat network.
7
