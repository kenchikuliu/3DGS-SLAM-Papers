<!-- page 1 -->
DefenseSplat: Enhancing the Robustness of 3D Gaussian Splatting via
Frequency-Aware Filtering
Yiran Qiao1, Yiren Lu1, Yunlai Zhou1, Rui Yang1, Linlin Hou1, Yu Yin1, Jing Ma1*
1Case Western Reserve University
{yxq350, yxl3538, yxz3057, rxy337, lxh633, yxy1421 jxm1384}@case.edu
Abstract
3D Gaussian Splatting (3DGS) has emerged as a powerful
paradigm for real-time and high-fidelity 3D reconstruction
from posed images. However, recent studies reveal its vul-
nerability to adversarial corruptions in input views, where
imperceptible yet consistent perturbations can drastically
degrade rendering quality, increase training and render-
ing time, and inflate memory usage, even leading to server
denial-of-service. In our work, to mitigate this issue, we be-
gin by analyzing the distinct behaviors of adversarial per-
turbations in the low- and high-frequency components of
input images using wavelet transforms. Based on this ob-
servation, we design a simple yet effective frequency-aware
defense strategy that reconstructs training views by filtering
high-frequency noise while preserving low-frequency con-
tent. This approach effectively suppresses adversarial ar-
tifacts while maintaining the authenticity of the original
scene.
Notably, it does not significantly impair training
on clean data, achieving a desirable trade-off between ro-
bustness and performance on clean inputs. Through exten-
sive experiments under a wide range of attack intensities
on multiple benchmarks, we demonstrate that our method
substantially enhances the robustness of 3DGS without ac-
cess to clean ground-truth supervision. By highlighting and
addressing the overlooked vulnerabilities of 3D Gaussian
Splatting, our work paves the way for more robust and se-
cure 3D reconstructions.
1. Introduction
3D reconstruction, exemplified by Neural Radiance Fields
(NeRF) [29] and the more recent 3D Gaussian Splatting
(3DGS) [14], refers to the process of recovering a scene’s
spatial geometry and structure from multi-view 2D images.
It is a fundamental task in computer vision with wide-
ranging applications in robotics [13, 22, 27, 35, 50], au-
tonomous driving [8, 15, 37, 49], and healthcare [4, 41]. Es-
pecially, 3DGS has recently gained prominence as a dom-
*
Corresponding author.
3.45%
15.25%
5.28%
19.56%
L
H
0.9864
0.0136
L
H
0.9862
0.0138
Frequency
Frequency
Energy(%)
Energy(%)
L
L
H
H
Figure 1.
Effects of Attacks for 3DGS. Top: Matching Rate
(a signal for multi-view consistency) in low-frequency (L) and
high-frequency components (H) of the clean/attacked images.
Medium: Two examples showing the comparison between low-
and high-frequency components on the same image after attack.
Bottom: Energy ratios of different frequency components in clean
images. The left side shows results on the Mip-NeRF 360 dataset,
while the right side shows Tanks-and-Temples. The medium ex-
amples are from the bonsai and Train scenes, respectively.
inant approach in 3D vision [2]. Unlike NeRF, which en-
codes scenes as continuous volumetric radiance fields pa-
rameterized by a multi-layer perceptron, 3DGS explicitly
represents the scene using a collection of 3D Gaussian el-
lipsoids. This explicit modeling confers several advantages
over NeRF, including faster rendering speeds, higher vi-
sual fidelity, greater interpretability, and improved scalabil-
ity. These benefits make 3DGS well-suited for deployment
on remote servers, where it can generate high-quality scene
representations from user-provided images.
1
arXiv:2602.19323v1  [cs.CV]  22 Feb 2026

<!-- page 2 -->
1.1. Vulnerabilities of 3DGS
While 3DGS offers strong performance, its strengths also
reveal critical vulnerabilities. Because it explicitly fits ob-
ject textures and edges, its rendering quality is highly sen-
sitive to input image quality [24]. Moreover, 3DGS does
not use a fixed network architecture; instead, its adaptive
density control dynamically adjusts the number of Gaus-
sian primitives, creating a flexible, data-dependent parame-
ter space [14]. This explicitness and flexibility make 3DGS
particularly vulnerable to adversarial perturbations, which
can degrade rendering and increase computational cost.
Although adversarial threats to 3DGS are only beginning
to be explored, initial work has emerged. Zeybey et al. [46]
introduces a two-stage image-space attack that transfers ad-
versarial noise into 3DGS representations, but it operates
solely in 2D and lacks true 3D-level manipulation. In con-
trast, Poison-Splat [23] performs a genuine 3D-level attack
by training a surrogate model and applying bi-level opti-
mization, achieving stronger disruption at the cost of signif-
icant computational overhead.
1.2. Why is Defense for 3DGS Underexplored?
Given the emergence of 3DGS-specific adversarial attacks
and the increasing awareness of its inherent vulnerabilities,
developing effective defense mechanisms for 3DGS is of
growing importance. However, to the best of our knowl-
edge, defense strategies against adversarial attacks on
3DGS remain largely underexplored.
This gap stems
from several unique obstacles that arise from the intersec-
tion of 3DGS’s design and the nature of adversarial threats:
Challenge 1: Incompatibility with Standard Defense
Frameworks. Most adversarial defenses are tailored for
supervised classification models with fixed neural architec-
tures, where bi-level optimization is used (an inner loop
crafts adversarial perturbations to maximize loss, while the
outer loop minimizes this loss via adversarial training [26]).
In contrast, 3DGS is a self-supervised method without a
fixed network structure, leading to fundamentally different
attack and defense formulations. As a result, standard de-
fense strategies do not apply effectively in this setting.
Challenge 2: Non-Invertible Attack Objectives. Even
when using a quick proxy workaround, such as a pre-trained
3DGS model, reversing the bi-level optimization objective
for defense remains ineffective. This is because adversarial
objectives in 3DGS often target non-invertible image statis-
tics, such as increasing the total variation of rendered im-
ages to inject subtle pixel-level noise. Attempting to reverse
such objectives introduces unintended artifacts like image
blurring or texture distortion, which ultimately degrade re-
construction quality instead of mitigating the attack.
Challenge 3: Absence of Ground-Truth Clean Data. Un-
like supervised settings where clean labels are available,
3DGS lacks a clear notion of “ground-truth” clean input.
This makes it impossible to distinguish clean from adver-
sarially perturbed images during training or inference. As
a result, designing defenses that restore adversarial inputs
without overcorrecting and harming genuinely clean images
becomes highly challenging, especially when performance
degradation cannot be tolerated.
Challenge 4: Limited Effectiveness of Traditional 2D
Image-Space Defenses.
Common 2D image-level or
frequency-based defense techniques often prove insufficient
or harmful in the 3DGS setting. For instance, Gaussian
smoothing and bilinear filtering may reduce adversarial per-
turbations, but they also blur legitimate details critical to
accurate scene reconstruction. Similarly, reducing the num-
ber of Gaussian primitives can limit adversarial effects but
comes at the cost of reconstructive fidelity. Even frequency-
domain filtering (e.g., via Fourier transforms) falls short, as
it neglects the spatial structure essential to 3D rendering,
weakening its ability of meaningful defense for 3DGS.
1.3. A Spark of Opportunity: Frequency
In this work, we present the first comprehensive study of de-
fense mechanisms against adversarial attacks on 3DGS. To
address the fundamental differences from conventional at-
tack and defense (Challenges 1 and 2), we forgo adversarial
training and instead propose a defense strategy that oper-
ates directly on the input multi-view images. We begin by
analyzing different frequency components of the input im-
ages and how adversarial perturbations affect them. Specif-
ically, we employ wavelet transforms to decompose the in-
put into low- and high-frequency components, leveraging
their ability to provide both spatial and frequency represen-
tations. This enables a spatially-aware, frequency-sensitive
consistency analysis across multiple views. Using deep im-
age matching techniques [32], we evaluate the consistency
of these frequency bands and have two important observa-
tions, revealing a new opportunity for defense:
• Observation 1: The adversarial perturbations on 3DGS
inputs often manifest as high-frequency noises (as
shown in the Top and Middle Rows of Fig. 1).
• Observation 2: The low-frequency component retains
the main content of the original scene, as the ratios of
energy (a signal for image structure and content) shown
in the Bottom of Fig. 1.
Building on this observation, we introduce DefenseSplat (as
shown in Fig. 2), a frequency-aware defense method de-
signed to suppress high-frequency adversarial noises while
preserving informative low-frequency content. By doing so,
the 3DGS optimization process naturally blurs out incon-
sistent perturbations during reconstruction, enhancing ro-
bustness without requiring access to clean ground-truth data
(addressing Challenge 3). Importantly, this approach main-
tains high reconstruction quality on clean inputs, address-
ing Challenges 4 by balancing robustness with fidelity, es-
2

<!-- page 3 -->
Filtered
Input
Gaussian Ellipsoid
Attack
DWT
iDWT
Blocking
Gaussian
Scale
Rendering
Wavelet Filtering
3D Optimization
3D Scene
Multi-view Images
Scale Regularization
[1, 1, 10]
[1, 10, 10]
[1, 1, 1]
[10, 10, 10]
Defended Input
View 1
View 2
View 3
View 5
View 4
View n
LL
LH
HL
HH
Figure 2. The overview of our proposed method.
pecially in scenarios where significant performance degra-
dation are not acceptable. Our main contribution can be
summarized as:
• We investigate an important yet previously unexplored
problem of defending 3DGS against adversarial attacks.
We analyze the significance of this problem and identify
the unique challenges it presents.
• We introduce a novel frequency-aware defense strategy
DefenseSplat for 3DGS by leveraging wavelet analysis
to characterize the behavior of adversarial perturbations
across frequency bands. To the best of our knowledge,
this is the first work specifically targeting adversarial
defense for 3DGS.
• Our comprehensive experiments show that the proposed
method enhances the robustness of 3DGS across di-
verse datasets and a range of attack strengths, without
substantially compromising clean-data performance.
2. Related Works
3D Gaussian Splatting. 3DGS [14] is a recently introduced
framework for 3D scene representation, which employs an
explicit set of anisotropic 3D Gaussian primitives parame-
terized by position, scale, orientation, and color to model
the underlying geometry and appearance. Unlike implicit
methods such as NeRF [29], 3DGS supports efficient, real-
time rendering through a differentiable rasterization pro-
cess. Owing to its high rendering quality, interpretability,
and scalability, 3DGS has been rapidly adopted across a
wide spectrum of vision and graphics applications, includ-
ing dynamic scene reconstruction [39, 44], open-vocabulary
3D semantic segmentation [25, 36, 42, 45], and text-to-3D
generation. However, as 3DGS is increasingly deployed on
cloud-based servers and used in commercial applications,
its potential vulnerabilities and susceptibility to adversarial
manipulation have become critical concerns.
Adversarial Attack. Adversarial attacks are typically for-
mulated as a bi-level optimization problem, where imper-
ceptible perturbations are crafted to maximize the model’s
prediction error while remaining constrained in norm. In
2D image classification, such attacks have been extensively
studied under white-box and black-box settings. White-box
methods [7, 26, 30] utilize gradient access to generate per-
turbations, whereas black-box attacks [1, 20] rely on trans-
ferability or query-based optimization. Due to the inher-
ent vulnerabilities of 3D Gaussian Splatting, several recent
works have emerged aiming to exploit these weaknesses
through adversarial attacks at the 3D level. Poison-Splat
[23] is a representative work in this direction, which per-
forms a genuine 3D adversarial attack by pretraining a sur-
rogate 3DGS model and adopting the total variance score of
the rendered images as the attack objective. However, ad-
versarial attacks targeting the domain of 3D Gaussian Splat-
ting remain largely underexplored.
Defense on Images.
The rise of adversarial attacks has
spurred advances in defense strategies in images [33].
Among them, adversarial training [26, 48] is the most
straightforward and widely used approach. It involves aug-
menting the training process with adversarial samples, al-
lowing the model to learn parameters that are robust to spe-
cific perturbation patterns. However, as mentioned in the
challenges discussed in Sec. 1, adversarial training cannot
be directly applied to 3D adversarial attacks. Another line
of defense leverages diffusion models to remove artifacts
and distortions from images [40]. However, such methods
typically rely on large-scale pretraining and are primarily
designed for scenarios where the degradation in rendering
quality arises from sparse-view supervision during 3D re-
construction. As far as we are aware, neither prior work has
addressed defense against 3D adversarial attacks, nor are
existing defense approaches directly applicable to this task.
3

<!-- page 4 -->
Illed Trajectory
Revised Trajectory
TSP
1st
2nd
3rd
4th
5th
6th
1st
2nd
3rd
4th
5th
6th
2
n - 1
n
Matching
Matching
Figure 3. To accurately measure image matching, we order the
camera poses using a Traveling Salesman Problem (TSP) formu-
lation, which yields an optimized traversal path.
Frequency Transform. The Discrete Fourier Transform
(DFT) plays a pivotal role in image processing. Previous
works have leveraged DFT on images to enhance synthe-
sis quality [12] and improve image classification accuracy
[34]. More recent studies have extended the application of
DFT to the domain of 3D reconstruction. FreGS [47] lever-
ages additional frequency spectrum supervision to mitigate
the issue of over-reconstruction in Gaussian Splatting. 3D-
GSW [11] employs DFT to guide the splitting of 3D Gaus-
sians in high-frequency regions, thereby enhancing recon-
struction quality. DFT only captures global frequency in-
formation, whereas the Discrete Wavelet Transform (DWT)
provides a joint representation of both spatial and frequency
domains. DWT can be seamlessly integrated with radiance
field representations [10, 21, 43], significantly enhancing
the quality of 3D reconstruction. Building upon the advan-
tageous properties of DWT and its demonstrated success in
3D reconstruction, we leverage DWT to analyze adversarial
attacks on 3DGS from both spatial and frequency perspec-
tives and further design a wavelet-based defense strategy.
3. Methodology
3.1. Preliminaries
3D Gaussian Splatting. 3DGS represents a 3D scene using
an explicit set of anisotropic 3D Gaussian primitives. Each
primitive is defined as:
G(x; Σ) = exp

−1
2(x)⊤Σ−1(x)

,
(1)
where x is the center of each primitive. Σ is the covariance
matrix, which, together with x determines the spatial distri-
bution of the Gaussian. To enable differentiable optimiza-
tion, Σ is expressed as a combination of a scaling matrix R
and a rotation matrix S, shown as:
Σ = RSS⊤R⊤.
(2)
Given a specific camera pose π, each pixel (x, y) color
Iπ(x, y) rendered onto the 2D image is calculated via alpha-
blending [14], taking into account a set of Gaussians (N)
that are simultaneously visible and contribute to the pixel
through spatial overlap:
Iπ(x, y) =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αj),
(3)
where ci is each projected Gaussian given by spherical
harmonics, and αi is the opacity. During the reconstruction
training process, the parameters of the 3D Gaussian prim-
itives are optimized by jointly minimizing the per-pixel L1
loss and the SSIM loss [38] between the rendered images
and the ground-truth images.
Discrete Wavelet Transform (DWT). DWT decomposes
an image into four frequency subbands: LL (low-low), LH
(low-high), HL (high-low), and HH (high-high). The LL
subband captures the low-frequency components in both
horizontal and vertical directions and typically contains
the majority of the image’s energy, representing its overall
structure and smooth content. This subband can be further
recursively decomposed across multiple levels to extract
hierarchical low-frequency representations. In contrast, the
LH, HL, and HH subbands capture high-frequency infor-
mation along different orientations: LH highlights horizon-
tal edge-like features by encoding low-frequency horizontal
and high-frequency vertical changes;
HL emphasizes
vertical edges by capturing high-frequency horizontal and
low-frequency vertical variations; and HH contains diago-
nal details with high-frequency variations in both directions.
Together, these subbands provide a joint spatial–frequency
representation of the image. The DWT [6] is defined as:
Wφ(j0, m, n) =
1
√
MN
M−1
X
x=0
N−1
X
y=0
I(x, y) φj0,m,n(x, y),
W i
ψ(j, m, n) =
1
√
MN
M−1
X
x=0
N−1
X
y=0
I(x, y) ψi
j,m,n(x, y),
(4)
where Wφ is the LL subband, and W i
ψ represents the LH,
HL, and HH subbands. j0 is the initial approximation
scale, j is the scale index for detail subbands. M and N
are width and height of the RGB image, while m and n
are the spatial location indices in the wavelet coefficient
domain. i = {H, D, V } denotes the horizontal, vertical,
and diagonal coefficients, respectively. φ(x, y) is a scaling
function and ψ(x, y) is a wavelet function. I(x, y) denotes
the pixel intensity of a given poison.
4

<!-- page 5 -->
3.2. Frequency-Aware Analysis of Adversarial Per-
turbations
Frequency Decomposition. To design more targeted de-
fense strategies, it is essential to first conduct a compre-
hensive analysis of the adversarial attacks.
Prior works
[11, 47] have demonstrated the power of frequency-domain
analysis, establishing it as a powerful tool for enhancing
scene reconstruction in 3DGS. For example, it has proven
effective in mitigating issues such as over-reconstruction
[47] and supporting applications like watermark embedding
[11]. Building on these successful precedents, we adopt
DWT to decompose adversarial images. Owing to its joint
representation of both spatial and frequency information,
DWT facilitates comprehensive analysis across 2D and 3D
domains. Moreover, its frequency decomposition enables a
fine-grained dissection of adversarial behavior across differ-
ent frequency subbands, laying the foundation for targeted
defense strategies. Specifically, given an adversarial image
I′ ∈R3×H×W , we apply wavelet decomposition to obtain
a set of subbands {LLl, LHl, HLl, HHl}, where l denotes
the level of DWT. For simplicity and without loss of effec-
tiveness, we set l = 1. To constrain each subband within the
RGB domain, we perform wavelet transform on I′ channel-
wise, normalize each channel to [0, 1], rescale it to [0, 255],
and finally fuse the three channels.
Vulnerability Analysis. Next, we use deep image matching
[9, 31, 32] to assess the 3D consistency of corresponding
subbands between clean and adversarial images, identifying
the most vulnerable regions where perturbations concen-
trate. A key challenge before this step is ensuring efficient
and accurate multi-view matching. Common strategies in-
clude brute-force, retrieval-based, and sequential matching.
Brute-force has quadratic complexity, making it inefficient
at scale. Retrieval-based methods rely on global features,
introducing error without optimal speed. Sequential match-
ing is most efficient but assumes a smooth view sequence,
which is often violated in practice due to randomly ordered
camera poses, leading to failures or reduced accuracy from
large viewpoint differences and occlusions.
To address this issue, we formulate it as a Traveling
Salesman Problem (TSP). We first convert each camera
pose into the form of the Special Euclidean group in 3D,
and then compute the pose loss between every pair of cam-
eras, defined as:
Lpose = wg Lgeo + wt Ltrans,
(5)
where Lgeo is the geodesic loss, Ltrans is the translation loss,
wg and wt are their corresponding weights. Specifically,
these two losses are defined as:
Lgeo = arccos


trace

R⊤
predRgt

−1
2

,
(6)
Ltrans = ∥tpred −tgt∥2
2 ,
(7)
where R(·) and t(·) represent the rotation matrix and the
translation vector, respectively. The subscripts indicate pre-
diction and ground truth, respectively. We treat each cam-
era as a city and the pose loss between cameras as the dis-
tance between cities in TSP. By solving the resulting in-
stance of the TSP using the Lin–Kernighan [18] algorithm,
we obtain a smooth and optimized camera trajectory (as
shown in Fig. 3). Based on the optimized trajectory, we
adopt the state-of-the-art SuperPoint [5] as the feature ex-
tractor and LightGlue [19] as the matcher to perform pair-
wise matching across all wavelet subbands of clean and ad-
versarial images. Each image is only matched with sub-
sequent images along the trajectory. For each image pair,
the matching rate is defined as the ratio of the number of
matched keypoints to the number of extracted keypoints, in-
dicating the consistency of the image across multiple view-
points. The overall multi-view matching rate is computed
by averaging the pairwise matching rates. Since the HH
subband contains the least energy (only < 0.08% in almost
all datasets, with details in appendix) and diagonal details
are relatively rare in natural images, we omit HH and use
LH and HL to represent the high-frequency information.
As shown in Fig. 1, both the bar charts and subband visu-
alizations indicate that the vulnerability is concentrated
in the high-frequency regions. The consistency degrada-
tion in high-frequency subbands is significantly more pro-
nounced than in low-frequency ones, suggesting that 3D ad-
versarial attacks primarily target high-frequency com-
ponents by injecting random noise, while introducing
relatively smooth and consistent pseudo-textures in the
low-frequency regions.
3.3. Defense via Frequency-Based Filtering
Based on the above analysis and insights, we propose a sim-
ple yet effective defense strategy, DefenseSplat. Given ad-
versarial images, we first apply DWT to obtain four sub-
bands. The high-frequency subbands (i.e., LH, HL, and
HH) are then set to zero, and the inverse wavelet transform
is performed using the LL subband together with the zeroed
high-frequency subbands to reconstruct the filtered image
in the spatial domain. This process can be defined as:
If = iDWT(DWT(I′)LL, 0, 0, 0),
(8)
where If represents the filtered images, iDWT is the inverse
Discrete Wavelet Transform, I′ denotes the adversarial im-
ages and DWT(I′)LL is the LL subband extracted from I′.
5

<!-- page 6 -->
Table 1. Quantitative comparison on Mip-NeRF 360 and Tanks-and-Temples datasets. Colors indicate the best and second best results.
#G/M indicates the number of Gaussians (Million).
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
Mip-NeRF 360
3DGS
1:01:01
5.91
20965
25.18
0.6356
0.4481
70
Compact GS
1:22:26
3.00
21072
24.95
0.6374
0.4590
128
Difix
0:45:44
3.65
15894
24.52
0.7326
0.3575
137
Ours
0:34:26
2.24
12567
27.32
0.7968
0.3469
243
Ours+ReLU
0:33:49
2.16
11400
27.47
0.8025
0.3417
258
Tanks-and-Temples
3DGS
0:31:22
2.76
10089
24.88
0.6425
0.4190
165
Compact GS
0:43:29
1.73
10930
24.37
0.6540
0.4278
244
Difix
0:24:12
1.71
7749
24.75
0.8092
0.3031
290
Ours
0:18:56
1.06
6296
26.29
0.8281
0.3199
468
Ours+ReLU
0:19:19
1.01
5858
26.42
0.8373
0.3134
498
Merely relying on the If component is far from sufficient,
as it still contains artificial textures with relatively low
consistency in the low-frequency domain. To mitigate this
problem, we exploit an intrinsic property of 3DGS: when
fitting regions with low cross-view consistency, the Gaus-
sians tend to average the colors across views to minimize
the L1 loss, thereby producing a blurred reconstruction over
these regions, which naturally further suppresses adversar-
ial artifacts in If that lack consistent support across views.
Another common observation is:
For certain artifi-
cially injected textures that exhibit high multi-view con-
sistency, the reconstruction process tends to generate elon-
gated Gaussians to fit these patterns, which increases the
number of Gaussians and memory consumption, thereby
undermining the effectiveness of the defense. To tackle this
issue, we design a ReLU-based scale regularization loss on
the normalized variance of the scales along the three princi-
pal axes of each Gaussian, defined as follows:
Lscale = ReLU(ν −τ),
(9)
where ν denotes the normalized variance, τ is a predefined
threshold that limits the range of scale regularization. For
Gaussians whose normalized variance falls below τ, the
loss does not propagate gradients.
This loss constrains
elongated Gaussians without affecting the other two
types of Gaussians: small spherical Gaussians and large flat
Gaussians (because their normalized variance of scales does
not exceed that of elongated Gaussians). Both of them are
crucial: the former contributes to reconstructing fine details
and alleviates over-reconstruction, while the latter enhances
the reconstruction quality of large smooth regions and
mitigates under-reconstruction. Therefore, this loss can ef-
fectively suppress adversarial textures while preserving the
reconstruction of genuine scene details. Finally, after train-
ing, the 3D Gaussians are rendered on all camera views to
generate the final defended images for downstream tasks.
4. Experiments
In this section, we conduct extensive experiments on multi-
ple datasets. Specifically, we address the following research
questions based on the experimental results. RQ1: How
does DefenseSplat perform compared to other baselines?
Is it an effective defense strategy? RQ2: Can DefenseSplat
consistently demonstrate effective defense capability under
varying levels of attack intensity? RQ3: Given the diversity
of user-submitted inputs, can DefenseSplat maintain a bal-
ance between robustness and performance on clean data?
RQ4: Can the Gaussian scale loss (Eq. (9)) further enhance
the overall performance of DefenseSplat (ablation study)?
4.1. Experiment Settings
Datasets. We use Mip-NeRF 360 [3], Tanks-and-Temples
[16] and LLFF [28], which are commonly used benchmarks
in the field of 3D reconstruction. For the Tanks and Tem-
ples dataset, we exclude scenes containing dynamic objects
and retain 8 static scenes for evaluation. For Mip-NerF 360
and LLFF datasets, we include all available scenes.
Baselines. Due to the absence of prior research dedicated
to the defense of 3D adversarial attacks, no existing method
serves as a directly comparable baseline.
Therefore, in
addition to the original 3DGS as a baseline, we select two
other methods whose research directions are most closely
related to defense as comparative baselines.
The first
one is Difix3D+ [40], which leverages diffusion models
to enhance the quality of reconstructed images while
mitigating 3D inconsistency. Moreover, it shares a similar
6

<!-- page 7 -->
Table 2. Quantitative comparison under different attack levels on Tanks-and-Temples.
ϵ
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
16/255
3DGS
0:27:38
2.17
8558
24.03
0.662
0.375
215
Compact GS
0:36:58
1.47
9466
23.59
0.669
0.386
306
Difix
0:21:39
1.48
7238
23.59
0.787
0.298
295
Ours
0:17:44
0.99
5642
24.89
0.815
0.294
449
Ours + ReLU
0:18:17
0.96
5418
25.03
0.824
0.291
473
32/255
3DGS
0:36:14
3.24
11541
20.36
0.449
0.485
140
Compact GS
0:50:13
2.20
12997
20.26
0.458
0.491
199
Difix
0:23:16
1.69
7417
22.22
0.707
0.348
263
Ours
0:20:02
1.25
6182
23.47
0.690
0.377
362
Ours + ReLU
0:19:43
1.19
5806
23.61
0.703
0.374
388
64/255
3DGS
0:43:00
4.07
12418
15.55
0.272
0.586
119
Compact GS
0:59:48
2.69
15807
15.65
0.285
0.591
178
Difix
0:25:46
1.99
8067
19.18
0.504
0.458
243
Ours
0:23:42
1.79
7651
19.27
0.443
0.509
281
Ours + ReLU
0:23:16
1.71
6958
19.38
0.454
0.505
301
Table 3. Quantitative results of our method on clean inputs.
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
Mip-NeRF 360
Clean
0:38:20
2.62
20965
29.06
0.868
0.248
223
Clean + Ours
0:32:11
1.83
11012
28.58
0.844
0.280
314
Clean + Ours + ReLU
0:32:23
1.79
10945
28.79
0.847
0.271
321
Tansk-and-Temples
Clean
0:22:48
1.56
7328
28.06
0.898
0.212
327
Clean + Ours
0:18:24
0.97
6161
27.18
0.868
0.254
517
Clean + Ours + ReLU
0:18:34
0.94
5857
27.33
0.875
0.245
526
setting with ours, as both methods operate directly on the
input images.
The second is CompactGS [17], which
achieves reduced storage and faster rendering speed while
maintaining high-quality reconstruction by effectively
removing redundant Gaussians that do not significantly
contribute to the overall performance.
This objective
closely aligns with the essence of defense.
Implementation Details. Our method and all baselines are
trained on the same server equipped with four A6000 GPUs.
To ensure fairness, all methods are trained for 30,000 itera-
tions. The remaining hyperparameters of the baselines are
kept unchanged from their original settings. Unless oth-
erwise specified, we use adversarial images with an attack
strength of ϵ = 16/255 as the training input. The weight
for the scale loss λscale is set to 1 × 105.
Evaluation Metrics We evaluate the effectiveness of our
defense strategy along two dimensions: robustness and re-
construction fidelity. For robustness, we evaluate training
time, the number of Gaussians, peak GPU memory usage,
and rendering speed, which together reflect the server’s ca-
pacity to sustain training under the load imposed by the in-
put images. For reconstruction fidelity, we adopt widely
used metrics including peak signal-to-noise ratio (PSNR),
structural similarity (SSIM), and learned perceptual image
patch similarity (LPIPS).
4.2. Experiment Results
Defense Effectiveness of DefenseSplat (RQ1). We con-
duct a comprehensive comparison between our method and
the baselines. As shown in Tab. 1, our method achieves
the best performance in both robustness and rendering
quality on the Mip-NeRF 360 and the Tanks-and-Temples
datasets with the sole exception of the LPIPS on the Tanks-
and-Temples dataset, where our score is only slightly higher
than the best-performing baseline. However, this does not
indicate a performance drawback: our method still achieves
the lowest peak GPU memory usage—an essential metric,
as exceeding hardware limits can lead to denial-of-service
issues and prevent rendering. Fig. 4 presents visualizations
of the rendered images. Since the perturbations introduced
7

<!-- page 8 -->
3DGS
CompactGS
Difix3D+
Ours
Ours+ReLU
Zoom-in GT
GT
room
room LH
Truck
Truck LH
Figure 4. Comparison of reconstruction quality of all methods and ground truth (GT) on Mip-NeRF 360 and Tanks-and-Temples datasets.
by the 3D adversarial attacks are nearly imperceptible to
humans, we zoom in to better highlight the details. As ob-
served in both the room and truck scenes, the rendered im-
ages from the original 3DGS contain numerous adversar-
ial artifacts, which are more clearly revealed in the corre-
sponding LH subband. Although CompactGS reduces the
number of Gaussians, it still fails to effectively constrain the
Gaussians from overfitting to adversarial perturbations. The
rendered images produced by Difix3D+ appear smoother;
however, they tend to introduce new spurious textures or
remove authentic ones. In the room scene, wavy artifacts
emerge on the carpet, while in the Truck scene, genuine de-
tails such as the rust on the hub and the tire tread patterns
are mistakenly eliminated. Our method effectively removes
adversarial artifacts while preserving the original textures
to the greatest extent, resulting in reconstructions that are
closest to the ground truth.
Defense Effectiveness Under Various Attack Strengths
(RQ2). As shown in Tab. 2, our method consistently shows
effective defense under varying attack strengths, due to
the fact that the attack primarily targets high-frequency in-
formation, which our method blocks. Even when the attack
strength increases and low-frequency components are pos-
sibly affected, our defense still remains effective, since our
method consistently filters out high-frequency content.
Robustness and Fidelity Trade-Off (RQ3). As shown in
Tab. 3, even when the input images are clean, our method
does not exhibit any significant performance degrada-
tion. The PSNR and SSIM drop by only 1.75% and 2.5%,
respectively, and LPIPS increases by just 11%. Although
our method removes high-frequency information, the LL
subband retains over 95% of the total energy across all sub-
bands. As a result, even for clean images, the majority of
information is preserved, leading to no significant degrada-
tion in reconstruction quality.
Effectiveness of Scale Loss (RQ4). As evidenced by Tab. 1
and Tab. 3, introducing the scale loss leads to a notice-
able improvement in our method’s performance, yielding
the best results across most evaluation metrics. We provide
additional experimental results in appendix.
5. Disscusion and Conclusion
In this work, we propose a novel and effective defense strat-
egy DefenseSplat against 3DGS adversarial attacks. By ap-
plying filtering on high-frequency wavelet subbands, our
method achieves both robust 3D reconstruction training on
the server side and faithful image rendering quality.
Since only one attack on 3DGS has been proposed so far,
8

<!-- page 9 -->
our evaluation is necessarily limited to this specific method.
However, given our observation that adversarial attacks pri-
marily target vulnerable high-frequency components in the
input, we believe our defense strategy is broadly generaliz-
able. We remain confident and look forward to testing its
effectiveness against future, unseen attacks.
It is worth noting that our defense strategy can also be
flexibly employed as a plug-and-play module, allowing seam-
less integration with existing methods to achieve better de-
fense performance.
Additional comparative implementa-
tions can be found in the appendix.
References
[1] Maksym Andriushchenko, Francesco Croce, Nicolas Flam-
marion, and Matthias Hein. Square attack: a query-efficient
black-box adversarial attack via random search. In European
conference on computer vision, pages 484–501. Springer,
2020. 3
[2] Yanqi Bao, Tianyu Ding, Jing Huo, Yaoli Liu, Yuxin Li,
Wenbin Li, Yang Gao, and Jiebo Luo. 3d gaussian splatting:
Survey, technologies, challenges, and opportunities. IEEE
Transactions on Circuits and Systems for Video Technology,
2025. 1
[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5470–5479, 2022. 6
[4] Yuanhao Cai, Jiahao Wang, Alan Yuille, Zongwei Zhou, and
Angtian Wang.
Structure-aware sparse-view x-ray 3d re-
construction. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 11174–
11183, 2024. 1
[5] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabi-
novich. Superpoint: Self-supervised interest point detection
and description. In Proceedings of the IEEE conference on
computer vision and pattern recognition workshops, pages
224–236, 2018. 5
[6] Rafael C Gonzalez. Digital image processing. Pearson edu-
cation india, 2009. 4
[7] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy.
Explaining and harnessing adversarial examples.
arXiv
preprint arXiv:1412.6572, 2014. 3
[8] Georg Hess, Carl Lindstr¨om, Maryam Fatemi, Christoffer
Petersson, and Lennart Svensson.
Splatad: Real-time li-
dar and camera rendering with 3d gaussian splatting for au-
tonomous driving. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pages 11982–11992,
2025. 1
[9] Francesco
Ioli,
Nicol`o
Dematteis,
Daniele
Giordan,
Francesco Nex, and Pinto Livio.
Deep learning low-cost
photogrammetry for 4d short-term glacier dynamics moni-
toring. PFG – Journal of Photogrammetry, Remote Sensing
and Geoinformation Science, 2024. 5
[10] Youngdong Jang, Dong In Lee, MinHyuk Jang, Jong Wook
Kim, Feng Yang, and Sangpil Kim. Waterf: Robust water-
marks in radiance fields for protection of copyrights. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 12087–12097, 2024. 4
[11] Youngdong Jang, Hyunje Park, Feng Yang, Heeju Ko, Euijin
Choo, and Sangpil Kim. 3d-gsw: 3d gaussian splatting for
robust watermarking. In Proceedings of the Computer Vi-
sion and Pattern Recognition Conference, pages 5938–5948,
2025. 4, 5
[12] Liming Jiang, Bo Dai, Wayne Wu, and Chen Change Loy.
Focal frequency loss for image reconstruction and synthesis.
In Proceedings of the IEEE/CVF international conference on
computer vision, pages 13919–13929, 2021. 4
[13] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. Splatam: Splat track & map 3d gaussians
for dense rgb-d slam. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
21357–21366, 2024. 1
[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 1, 2, 3, 4
[15] Mustafa Khan, Hamidreza Fazlali, Dhruv Sharma, Tongtong
Cao, Dongfeng Bai, Yuan Ren, and Bingbing Liu. Autosplat:
Constrained gaussian splatting for autonomous driving scene
reconstruction. In 2025 IEEE International Conference on
Robotics and Automation (ICRA), pages 8315–8321. IEEE,
2025. 1
[16] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun. Tanks and temples: Benchmarking large-scale scene
reconstruction. ACM Transactions on Graphics, 36(4), 2017.
6
[17] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko,
and Eunbyung Park. Compact 3d gaussian representation for
radiance field. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21719–
21728, 2024. 7
[18] Shen Lin and Brian W Kernighan.
An effective heuristic
algorithm for the traveling-salesman problem. Operations
research, 21(2):498–516, 1973. 5
[19] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Polle-
feys. Lightglue: Local feature matching at light speed. In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 17627–17638, 2023. 5
[20] Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Song.
Delving into transferable adversarial examples and black-
box attacks. arXiv preprint arXiv:1611.02770, 2016. 3
[21] Ange Lou, Benjamin Planche, Zhongpai Gao, Yamin Li,
Tianyu Luan, Hao Ding, Terrence Chen, Jack Noble, and
Ziyan Wu. Darenerf: Direction-aware representation for dy-
namic scenes. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 5031–
5042, 2024. 4
[22] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu, Ji-
wen Lu, and Yansong Tang. Manigaussian: Dynamic gaus-
sian splatting for multi-task robotic manipulation.
In Eu-
ropean Conference on Computer Vision, pages 349–366.
Springer, 2024. 1
9

<!-- page 10 -->
[23] Jiahao Lu, Yifan Zhang, Qiuhong Shen, Xinchao Wang, and
Shuicheng Yan. Poison-splat: Computation cost attack on 3d
gaussian splatting. arXiv preprint arXiv:2410.08190, 2024.
2, 3, 1
[24] Yiren Lu, Yunlai Zhou, Disheng Liu, Tuo Liang, and Yu Yin.
Bard-gs: Blur-aware reconstruction of dynamic scenes via
gaussian splatting. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pages 16532–16542,
2025. 2
[25] Yiren Lu, Yunlai Zhou, Yiran Qiao, Chaoda Song, Tuo
Liang, Jing Ma, and Yu Yin. Segment then splat: A uni-
fied approach for 3d open-vocabulary segmentation based on
gaussian splatting. arXiv preprint arXiv:2503.22204, 2025.
3
[26] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
Dimitris Tsipras, and Adrian Vladu. Towards deep learn-
ing models resistant to adversarial attacks. arXiv preprint
arXiv:1706.06083, 2017. 2, 3
[27] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and An-
drew J Davison. Gaussian splatting slam. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 18039–18048, 2024. 1
[28] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon,
Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and
Abhishek Kar. Local light field fusion: Practical view syn-
thesis with prescriptive sampling guidelines. ACM Transac-
tions on Graphics (ToG), 38(4):1–14, 2019. 6
[29] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021. 1,
3
[30] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and
Pascal Frossard. Deepfool: a simple and accurate method to
fool deep neural networks. In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
2574–2582, 2016. 3
[31] Luca Morelli, Fabio Bellavia, Fabio Menna, and Fabio Re-
mondino. Photogrammetry now and then–from hand-crafted
to deep-learning tie points–. The International Archives of
the Photogrammetry, Remote Sensing and Spatial Informa-
tion Sciences, 48:163–170, 2022. 5
[32] L. Morelli, F. Ioli, F. Maiwald, G. Mazzacca, F. Menna, and
F. Remondino. Deep-image-matching: A toolbox for mul-
tiview image matching of complex scenarios. The Interna-
tional Archives of the Photogrammetry, Remote Sensing and
Spatial Information Sciences, XLVIII-2/W4-2024:309–316,
2024. 2, 5
[33] Yiran Qiao, Disheng Liu, Yiren Lu, Yu Yin, Mengnan
Du, and Jing Ma.
Counterfactual visual explanation
via causally-guided adversarial steering.
arXiv preprint
arXiv:2507.09881, 2025. 3
[34] Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and
Jie Zhou.
Global filter networks for image classification.
Advances in neural information processing systems, 34:980–
993, 2021. 4
[35] Antoni Rosinol, John J Leonard, and Luca Carlone. Nerf-
slam: Real-time dense monocular slam with neural radiance
fields. In 2023 IEEE/RSJ International Conference on Intel-
ligent Robots and Systems (IROS), pages 3437–3444. IEEE,
2023. 1
[36] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-
Hua Guan.
Language embedded 3d gaussians for open-
vocabulary scene understanding.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5333–5343, 2024. 3
[37] Adam Tonderski, Carl Lindstr¨om, Georg Hess, William
Ljungbergh, Lennart Svensson, and Christoffer Petersson.
Neurad: Neural rendering for autonomous driving. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 14895–14904, 2024. 1
[38] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 4
[39] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 20310–20320, 2024. 3
[40] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi
Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Goj-
cic, and Huan Ling.
Difix3d+: Improving 3d reconstruc-
tions with single-step diffusion models. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 26024–26035, 2025. 3, 6
[41] Shaokai Wu, Yuxiang Lu, Yapan Guo, Wei Ji, Suizhi Huang,
Fengyu Yang, Shalayiding Sirejiding, Qichen He, Jing Tong,
Yanbiao Ji, et al. Discretized gaussian representation for to-
mographic reconstruction. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 25073–
25082, 2025. 1
[42] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao
Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding,
Jingdong Wang, et al. Opengaussian: Towards point-level 3d
gaussian-based open vocabulary understanding.
Advances
in Neural Information Processing Systems, 37:19114–19138,
2024. 3
[43] Muyu Xu, Fangneng Zhan, Jiahui Zhang, Yingchen Yu, Xi-
aoqin Zhang, Christian Theobalt, Ling Shao, and Shijian
Lu. Wavenerf: Wavelet-based generalizable neural radiance
fields. In Proceedings of the IEEE/CVF International Con-
ference on Computer Vision, pages 18195–18204, 2023. 4
[44] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction.
In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 20331–20341, 2024. 3
[45] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke.
Gaussian grouping: Segment and edit anything in 3d scenes.
In European conference on computer vision, pages 162–179.
Springer, 2024. 3
[46] Abdurrahman Zeybey,
Mehmet Ergezer,
and Tommy
Nguyen.
Gaussian splatting under attack:
Investigat-
ing adversarial noise in 3d objects.
arXiv preprint
arXiv:2412.02803, 2024. 2
10

<!-- page 11 -->
[47] Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, and
Eric Xing.
Fregs: 3d gaussian splatting with progressive
frequency regularization. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 21424–21433, 2024. 4, 5
[48] Yonggang Zhang, Mingming Gong, Tongliang Liu, Gang
Niu, Xinmei Tian, Bo Han, Bernhard Sch¨olkopf, and Kun
Zhang. Causaladv: Adversarial robustness through the lens
of causality. arXiv preprint arXiv:2106.06196, 2021. 3
[49] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang,
Deqing Sun, and Ming-Hsuan Yang.
Drivinggaussian:
Composite gaussian splatting for surrounding dynamic au-
tonomous driving scenes. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pages 21634–21643, 2024. 1
[50] Xueyan Zou, Yuchen Song, Ri-Zhao Qiu, Xuanbin Peng,
Jianglong Ye, Sifei Liu, and Xiaolong Wang. 3d-spatial mul-
timodal memory. In The Thirteenth International Conference
on Learning Representations, 2025. 1
11

<!-- page 12 -->
DefenseSplat: Enhancing the Robustness of 3D Gaussian Splatting via
Frequency-Aware Filtering
Supplementary Material
A. Additional Implementation Details
Scene Selection. For the Mip-NeRF 360 dataset, we use
all available scenes: bicycle, bonsai, counter, flowers, gar-
den, kitchen, room, stump, and treehill. Eight static scenes
are chosen from the Tanks-and-Temples dataset for evalua-
tion: Auditorium, Ballroom, Caterpillar, Courtroom, M60,
Meetingroom, Train, and Truck. We select the following
scenes from the LLFF dataset: fern, flower, fortress, horns,
orchids, room, and trex.
Poisoned Data Preparation. The perturbed version of the
Mip-NeRF 360 dataset is provided in Poison-Splat [23], and
thus we directly adopt it in our experiments. For the Tanks-
and-Temples dataset, we generate the corresponding per-
turbed version using the script provided by Poison-Splat.
The attack strength is set to ϵ = 16/255, with 36,000 at-
tack iterations. To evaluate the effectiveness of DefenseS-
plat under varying attack intensities, we additionally con-
sider attack strengths of ϵ = 32/255 and ϵ = 64/255. For
the LLFF dataset, to reduce memory consumption, we con-
struct the adversarial data using quarter-resolution images.
The attack strength is still set to ϵ = 16/255, with 30,000
attack iterations. It is important to note that in the Poison-
Splat setting, the attack is applied only to the multi-view
images, while the COLMAP-generated sparse reconstruc-
tion (including camera poses and intrinsics) remains un-
changed. Therefore, during training on the adversarial im-
ages, we still use the camera data obtained from the clean
images.
For the LLFF dataset, the provided camera in-
trinsics are associated with a simple radial camera model,
which is incompatible with 3DGS reconstruction. There-
fore, before generating the adversarial images, we manually
re-run COLMAP to obtain camera intrinsics and poses un-
der the pinhole model, along with the corresponding undis-
torted images.
Training Details. The GPUs used in the experiments, the
weight of the loss terms, and the total number of training
iterations have already been specified in the main text. It
is worth noting that, since we monitor training time and
GPU memory usage, only one task is allowed to run on each
GPU at a time. For GPU memory usage, recording the peak
memory consumption is meaningful. We sample the cur-
rent memory usage every 0.2 seconds during training and
report the maximum value after training completes. As for
the number of Gaussians, we perform the counting on the
.ply file after training is completed. Another important pa-
rameter to set is the threshold τ in the scale loss. To avoid
generating overly elongated Gaussians, we define a refer-
Table 4. Quantitative results of deep image matching on different
frequency components. (CR represents matching rate for clean
input, PR means matching rate for poisoned input.)
Scene
CR(%)
PR(%)
LL
LH&HL
LL
LH&HL
Mip-NeRF 360
bicycle
59.21
54.02
56.18
42.96
bonsai
34.76
26.99
33.17
21.94
counter
60.63
55.33
58.26
46.35
flowers
35.44
24.97
35.34
20.87
garden
52.61
32.46
51.32
24.74
kitchen
54.78
48.47
54.29
38.52
room
53.36
50.02
50.1
40.11
stump
33.00
27.80
25.04
20.83
treehill
30.62
20.50
28.92
17.69
average
46.05
37.84
43.62
30.44
Tanks-and-Temples
Auditorium
49.69
48.38
45.66
35.48
Ballroom
59.37
51.10
57.75
43.68
Caterpillar
59.75
49.45
58.45
46.81
Courtroom
51.30
44.31
48.50
37.27
M60
57.67
52.83
56.41
45.15
Meetingroom
54.43
51.68
52.88
42.66
Train
55.15
48.26
53.81
41.90
Truck
53.72
46.46
52.49
39.71
average
55.14
49.06
53.24
41.58
LLFF
fern
47.95
44.50
46.78
40.43
flower
69.39
63.70
68.19
55.66
fortress
62.51
48.82
57.88
42.59
horns
58.58
51.09
57.94
47.73
orchids
45.74
37.90
45.60
36.87
room
61.68
57.28
56.37
46.54
trex
59.90
52.03
58.61
47.12
average
57.96
50.76
55.91
45.27
ence Gaussian with scales [1, 1, 10] along its three axes. All
Gaussians that are “thinner” than this reference will be pe-
nalized. The normalized variance of this reference Gaussian
is 1.6875, so we conservatively set τ to 1.6.
B. Additional Experimental Results
B.1. Frequency-Aware Analysis of Adversarial Per-
turbations
Since the energy proportion of the HH subband is negligi-
ble and diagonal textures are relatively rare, we omit analy-
1

<!-- page 13 -->
Table 5. Quantitative results of deep image matching on different
frequency components over different attack strengths (Tanks-and-
Templse dataset).
ϵ
CR(%)
PR(%)
LL
LH&HL
LL
LH&HL
16/255
54.44
47.36
53.15
40.81
32/255
54.44
47.36
51.23
37.35
64/255
54.44
47.36
43.93
35.23
sis of the HH component. Instead, we average the results of
the LH and HL subbands to represent the high-frequency
content. As shown in Tab. 4, all three datasets exhibit the
same trend: the decline in image matching after attack is
more pronounced in the high-frequency components than in
the low-frequency components. Specifically, for Mip-NeRF
360, the matching rate in the low-frequency band drops by
5.28%, while the high-frequency band drops by 19.56%;
for Tanks-and-Temples, the drops are 3.45% and 15.25%,
respectively; and for LLFF, the corresponding values are
3.54% and 10.82%.
We also analyze the changes in image matching rates
under different attack intensities. Specifically, we use the
Train and Truck scenes from the Tanks-and-Temples dataset,
as these are among the most commonly used scenes in prior
work. As shown in Tab. 5, when ϵ = 16/255, the low-
frequency matching rate drops by 2.37%, while the high-
frequency matching rate drops by 13.83%. At ϵ = 32/255,
this degradation increases to 5.90% and 21.14%, respec-
tively. When ϵ = 64/255, the corresponding values are
19.31% and 25.61%.
It can be observed that as the at-
tack strength increases, the target of the attack gradually
shifts from high-frequency to low-frequency components.
It is reasonable to predict that as the attack strength further
increases, the degradation in low-frequency matching may
even exceed that of high-frequency. However, this obser-
vation does not undermine the effectiveness of our defense
strategy. The first reason is that adversarial perturbations
are typically expected to remain imperceptible to the hu-
man eye, rendering excessively strong attacks of little prac-
tical relevance in real-world scenarios. The second reason
is that even under stronger attack intensities, high-frequency
components are still targeted; therefore, filtering out high-
frequency information can still effectively remove part of
the perturbation. As a result, our defense strategy remains
effective.
B.2. Additional Quantitative Results
We provide more quantitative results on the LLFF dataset
in this section, as shown in Tab. 6 and Tab. 7. Our method
ranks among the top two across all evaluation metrics. Since
the previous quantitative results only report the average per-
formance on each dataset, we provide detailed per-scene re-
sults for each dataset from Tab. 8 to Tab. 13. In certain
scenes, our method’s LPIPS score is slightly inferior to that
of Difix3D+. This may be attributed to the fact that Di-
fix3D+ employs a diffusion model as its backbone. The de-
noising capability of the pre-trained diffusion model, along
with the rich priors it encodes, may enable it to produce out-
puts that are visually closer to the clean images when the
inputs are adversarially attacked. However, for 3D recon-
struction tasks, pixel-level consistency between the gener-
ated and ground truth images is more critical. Our method
consistently outperforms all baselines across all scenes in
terms of PSNR and SSIM, demonstrating its superiority. It
is important to note that the primary objective of our task is
to enhance the robustness of 3DGS, specifically by reduc-
ing the rate of training failure on the server side. Because
if excessive computational resources are consumed, train-
ing may be interrupted at any time or even fail to start, not
to mention the rendering quality in the subsequent stages.
As shown by the results, our method is capable of com-
pleting training for each scene within a shorter time du-
ration, using a smaller number of Gaussians and minimal
GPU memory. While achieving this objective, it is widely
agreed that higher rendering quality is always preferable.
Our method successfully meets this requirement by deliver-
ing the highest average rendering quality among all meth-
ods, while maintaining high efficiency.
B.3. Additional Qualitative Results
Fig. 5, Fig. 6, and Fig. 7 present additional rendering re-
sults. It is evident that the rendered images from the orig-
inal 3DGS and CompactGS exhibit significant adversarial
artifacts, indicating that these two methods are nearly in-
effective at defending against 3D adversarial attacks. Di-
fix3D+ improves the defense performance to some extent,
but it also introduces several issues. In certain scenes, Di-
fix3D+ excessively smooths the images to enhance quality,
which results in the loss of fine details; in other cases, it
sharpens the entire image or even slightly alters the color
of some objects. Additionally, Difix3D+ sometimes intro-
duces artificial textures that were not present in the original
images. This result is expected, as Difix3D+ was originally
trained on images with poor rendering quality due to sparse
reconstruction. However, the distribution of these images
differs from that of adversarially perturbed ones. Conse-
quently, when applying single-step diffusion inference to
perturbed images, the performance is naturally less satisfac-
tory. The visualization of the LH subband also reflects the
effectiveness of our defense. Since LH highlights the pres-
ence of perturbation textures more clearly, the reduction or
even disappearance of such textures provides additional ev-
idence for the success of our defense strategy.
2

<!-- page 14 -->
Table 6. Quantitative comparison on LLFF. Colors indicate the best and second best results. #G/M indicates the number of Gaussians
(Million).
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
3DGS
0:24:47
2.03
5407
25.04
0.5856
0.4344
196
Compact GS
0:46:01
1.75
8231
24.07
0.5540
0.4453
232
Difix
0:16:11
0.99
3360
25.09
0.7968
0.2654
437
Ours
0:12:01
0.54
2919
27.67
0.8620
0.2480
749
Ours+ReLU
0:11:47
0.53
2144
27.63
0.8635
0.2461
813
Table 7. Quantitative results of our method on clean inputs (LLFF).
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
Clean
0:14:05
0.69
3162
31.29
0.9431
0.1224
650
Clean + Ours
0:11:58
0.50
3130
28.39
0.8919
0.1992
841
Clean + Ours + ReLU
0:11:47
0.48
2161
28.34
0.8910
0.1979
886
Table 8. Detailed quantitative comparison on Mip-NeRF 360. Colors indicate the best and second best results. #G/M indicates the
number of Gaussians (Million).
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
bicycle
3DGS
1:12:03
8.16
23704
23.88
0.6630
0.4038
48
Compact GS
1:41:07
4.22
23190
23.12
0.6305
0.4327
77
Difix
0:57:23
5.98
19984
22.91
0.6753
0.3586
67
Ours
0:43:18
3.92
13902
24.77
0.7606
0.3400
104
Ours+ReLU
0:42:39
3.71
13420
24.66
0.7579
0.3442
112
bonsai
3DGS
0:47:37
4.88
18496
27.38
0.6435
0.5054
82
Compact GS
1:09:08
2.20
18318
27.30
0.6587
0.5041
168
Difix
0:31:58
1.89
12394
25.71
0.8281
0.3440
221
Ours
0:24:19
0.89
11494
30.78
0.8875
0.3384
488
Ours+ReLU
0:24:31
0.86
10498
30.79
0.8910
0.3313
506
counter
3DGS
0:57:48
4.31
23320
26.23
0.6224
0.4821
89
Compact GS
1:08:36
1.79
21608
26.30
0.6523
0.4819
203
Difix
0:40:57
1.86
14720
26.39
0.8264
0.3413
216
Ours
0:31:12
1.05
11512
28.67
0.8470
0.3491
395
Ours+ReLU
0:30:17
0.96
10600
28.89
0.8585
0.3470
437
flowers
3DGS
0:41:57
4.20
13872
22.53
0.6351
0.4284
96
Compact GS
1:04:31
2.40
14412
21.81
0.5982
0.4557
154
Difix
0:35:59
3.10
11572
21.74
0.6406
0.3936
132
Ours
0:29:37
2.17
9500
22.85
0.6857
0.3915
193
Ours+ReLU
0:29:41
2.08
9734
22.92
0.6886
0.3822
202
3

<!-- page 15 -->
Table 9. (Continued Table) Detailed quantitative comparison on Mip-NeRF 360. Colors indicate the best and second best results. #G/M
indicates the number of Gaussians (Million).
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
garden
3DGS
1:05:19
6.20
19246
25.00
0.6900
0.3731
62
Compact GS
1:40:22
3.66
20806
24.93
0.6935
0.3858
97
Difix
0:50:30
4.26
15226
24.88
0.736
0.3267
93
Ours
0:37:50
2.85
11804
26.88
0.8084
0.2958
143
Ours+ReLU
0:37:03
2.79
11450
27.26
0.8239
0.2714
148
kitchen
3DGS
1:13:39
5.89
23010
26.36
0.6620
0.3869
64
Compact GS
1:33:36
3.37
24434
26.52
0.6854
0.3774
108
Difix
0:56:06
3.12
19402
25.74
0.7998
0.2744
123
Ours
0:39:45
1.58
14056
29.80
0.8771
0.2500
253
Ours+ReLU
0:38:10
1.54
12912
30.27
0.8843
0.2388
262
room
3DGS
1:20:18
6.73
31902
26.94
0.5432
0.5143
58
Compact GS
1:31:39
2.96
32784
27.30
0.5818
0.5100
122
Difix
0:43:10
2.01
18130
27.56
0.8264
0.3517
208
Ours
0:32:42
1.23
18649
31.15
0.8329
0.3598
348
Ours+ReLU
0:30:53
1.15
12428
31.37
0.8432
0.3621
379
stump
3DGS
1:00:09
7.97
20388
25.72
0.6535
0.4681
51
Compact GS
1:19:27
3.27
17658
25.16
0.6420
0.4863
112
Difix
0:55:03
7.15
19428
23.84
0.6439
0.3975
57
Ours
0:37:06
3.93
12142
27.66
0.7837
0.3686
107
Ours+ReLU
0:37:32
3.80
11954
27.76
0.7853
0.3666
112
treehill
3DGS
0:50:22
4.87
14744
22.59
0.6079
0.4710
81
Compact GS
1:13:24
3.17
16436
22.14
0.5941
0.4971
115
Difix
0:40:26
3.51
12192
21.90
0.6170
0.4298
116
Ours
0:34:02
2.58
10042
23.28
0.6883
0.4293
160
Ours+ReLU
0:33:33
2.50
9608
23.28
0.6901
0.4319
168
4

<!-- page 16 -->
Table 10. Detailed quantitative comparison on Tanks-and-Temples. Colors indicate the best and second best results. #G/M indicates
the number of Gaussians (Million).
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
Auditorium
3DGS
0:29:45
2.42
11782
26.06
0.5697
0.5125
166
Compact GS
0:43:13
1.79
11842
25.66
0.5924
0.5183
207
Difix
0:17:57
0.76
5710
27.14
0.8713
0.3411
554
Ours
0:15:30
0.50
5120
28.82
0.8682
0.3761
877
Ours+ReLU
0:16:00
0.46
4730
28.90
0.8779
0.3790
951
Ballroom
3DGS
0:38:28
3.53
11362
24.68
0.6896
0.3679
112
Compact GS
0:55:36
2.25
12426
24.26
0.6920
0.3746
164
Difix
0:35:09
2.95
10598
24.12
0.7949
0.2642
135
Ours
0:25:13
1.79
7806
25.59
0.8250
0.2898
228
Ours+ReLU
0:25:23
1.70
7090
25.77
0.8350
0.2747
244
Caterpillar
3DGS
0:22:45
1.76
7598
24.08
0.6793
0.4058
228
Compact GS
0:30:03
1.13
7668
23.56
0.6870
0.4225
345
Difix
0:18:52
1.18
6808
23.45
0.7575
0.3390
347
Ours
0:15:58
0.82
6000
24.97
0.7834
0.3341
513
Ours+ReLU
0:16:22
0.79
5572
25.04
0.7928
0.3277
533
Courtroom
3DGS
0:41:12
4.64
13074
25.09
0.6266
0.4313
87
Compact GS
1:00:36
2.52
14074
24.06
0.6315
0.4433
147
Difix
0:30:24
2.77
9324
24.92
0.7968
0.3003
150
Ours
0:21:45
1.62
6854
26.11
0.8063
0.3448
265
Ours+ReLU
0:22:20
1.55
6428
26.24
0.8152
0.3337
283
M60
3DGS
0:32:19
2.65
9978
25.62
0.6559
0.4030
149
Compact GS
0:39:57
1.41
11830
25.50
0.6780
0.4010
272
Difix
0:24:53
1.60
8000
25.27
0.8364
0.2724
258
Ours
0:20:12
0.95
6994
27.60
0.8635
0.2739
447
Ours+ReLU
0:20:20
0.91
6318
27.80
0.8717
0.2638
467
Meetingroom
3DGS
0:31:10
2.72
9798
25.47
0.5958
0.4822
147
Compact GS
0:44:28
1.81
10670
24.78
0.6134
0.4915
206
Difix
0:23:00
1.45
7072
25.90
0.8435
0.3117
289
Ours
0:17:25
0.83
6308
27.47
0.8474
0.3521
520
Ours+ReLU
0:17:36
0.77
5712
27.58
0.8574
0.3464
564
5

<!-- page 17 -->
Table 11. (Continued Table) Detailed quantitative comparison on Tanks-and-Temples. Colors indicate the best and second best results.
#G/M indicates the number of Gaussians (Million).
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
Train
3DGS
0:21:40
1.30
7596
23.89
0.7231
0.3605
303
Compact GS
0:26:19
0.90
7742
23.13
0.7237
0.3799
434
Difix
0:20:38
1.09
7666
23.02
0.7702
0.3086
372
Ours
0:17:15
0.73
5972
24.21
0.8011
0.3101
562
Ours+ReLU
0:17:19
0.70
5588
24.32
0.8080
0.3056
594
Truck
3DGS
0:33:35
3.04
9520
24.17
0.6001
0.3884
128
Compact GS
0:47:36
2.04
11190
24.04
0.6139
0.3910
179
Difix
0:22:40
1.88
6810
24.16
0.8033
0.2874
219
Ours
0:18:13
1.26
5312
25.56
0.8296
0.2780
337
Ours+ReLU
0:19:14
1.21
5422
25.74
0.8404
0.2753
353
Table 12. Detailed quantitative comparison on LLFF. Colors indicate the best and second best results. #G/M indicates the number of
Gaussians (Million).
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
fern
3DGS
0:27:44
2.22
5588
24.95
0.5988
0.4141
166
Compact GS
0:50:16
2.01
8228
23.01
0.5362
0.4382
189
Difix
0:18:24
1.20
3202
23.76
0.7471
0.2996
314
Ours
0:13:23
0.67
2806
25.80
0.8205
0.2888
560
Ours+ReLU
0:12:44
0.65
2064
25.92
0.8250
0.2872
596
flower
3DGS
0:18:46
1.41
3978
25.27
0.5702
0.4580
279
Compact GS
0:34:54
1.10
5992
24.51
0.5424
0.4564
353
Difix
0:14:17
0.99
2958
25.42
0.7766
0.2840
407
Ours
0:10:14
0.49
1834
29.15
0.8681
0.2179
838
Ours+ReLU
0:10:29
0.49
1916
29.08
0.8695
0.2099
843
fortress
3DGS
0:26:39
2.22
5538
25.51
0.5405
0.4640
172
Compact GS
1:01:14
2.32
10948
25.13
0.5225
0.4696
158
Difix
0:18:25
1.09
3318
25.19
0.7264
0.2867
353
Ours
0:11:52
0.44
2614
30.69
0.8769
0.2408
711
Ours+ReLU
0:12:10
0.44
1902
30.53
0.8771
0.2449
924
horns
3DGS
0:24:30
1.89
5392
24.90
0.6241
0.4049
204
Compact GS
0:42:36
1.43
7358
24.23
0.5974
0.4171
261
Difix
0:17:40
1.03
3408
25.24
0.7989
0.2759
387
Ours
0:13:28
0.58
3552
26.80
0.8464
0.2459
629
Ours+ReLU
0:13:09
0.57
2684
26.79
0.8474
0.2465
709
6

<!-- page 18 -->
Table 13. (Continued Table) Detailed quantitative comparison on LLFF. Colors indicate the best and second best results. #G/M indicates
the number of Gaussians (Million).
Method
Training Time ↓
#G/M ↓
GPU Mem (MB) ↓
PSNR ↑
SSIM ↑
LPIPS ↓
FPS ↑
orchids
3DGS
0:21:14
1.61
4142
23.77
0.7026
0.3478
240
Compact GS
0:38:03
1.28
6194
22.72
0.6716
0.3499
295
Difix
0:19:19
1.34
3594
22.63
0.7584
0.2196
280
Ours
0:13:52
0.84
2582
24.45
0.8282
0.2222
458
Ours+ReLU
0:13:43
0.86
2500
24.15
0.8248
0.2210
466
room
3DGS
0:30:08
2.78
7546
26.03
0.4692
0.5269
135
Compact GS
0:53:43
2.48
10778
24.77
0.4350
0.5456
147
Difix
0:11:41
0.48
3898
28.61
0.9133
0.2322
820
Ours
0:10:24
0.32
3378
30.17
0.9103
0.2691
1154
Ours+ReLU
0:09:44
0.31
1758
30.22
0.9133
0.2666
1230
trex
3DGS
0:24:29
2.11
5662
24.86
0.5936
0.4254
182
Compact GS
0:41:22
1.63
8118
24.13
0.5728
0.4401
223
Difix
0:13:31
0.79
3140
24.75
0.8571
0.2601
502
Ours
0:10:53
0.44
3666
26.64
0.8838
0.2510
895
Ours+ReLU
0:10:30
0.43
2186
26.74
0.8875
0.2464
927
7

<!-- page 19 -->
3DGS
CompactGS
Difix3D+
Ours
Ours+ReLU
Zoom-in GT
GT
bonsai
bonsaiLH
kitchenLH
kitchen
Figure 5. Comparison of reconstruction quality of all methods and ground truth (GT) on Mip-NeRF 360 datasets (bonsai & kitchen).
8

<!-- page 20 -->
3DGS
CompactGS
Difix3D+
Ours
Ours+ReLU
Zoom-in GT
GT
Caterpillar
CaterpillarLH
TrainLH
Train
Figure 6. Comparison of reconstruction quality of all methods and ground truth (GT) on Tanks-and-Temples datasets (Caterpillar & Train).
9

<!-- page 21 -->
3DGS
CompactGS
Difix3D+
Ours
Ours+ReLU
Zoom-in GT
GT
fern
fernLH
hornsLH
horns
Figure 7. Comparison of reconstruction quality of all methods and ground truth (GT) on LLFF datasets (fern & horns).
10
