<!-- page 1 -->
DIP-GS: Deep Image Prior For Gaussian Splatting Sparse View Recovery
Rajaei Khatib and Raja Giryes
Tel Aviv University
rajaeikhatib@mail.tau.ac.il and raja@tauex.tau.ac.il
Abstract
3D Gaussian Splatting (3DGS) is a leading 3D scene recon-
struction method, obtaining high-quality reconstruction with
real-time rendering runtime performance. The main idea be-
hind 3DGS is to represent the scene as a collection of 3D
gaussians, while learning their parameters to fit the given
views of the scene. While achieving superior performance in
the presence of many views, 3DGS struggles with sparse view
reconstruction, where the input views are sparse and do not
fully cover the scene and have low overlaps. In this paper, we
propose DIP-GS, a Deep Image Prior (DIP) 3DGS representa-
tion. By using the DIP prior, which utilizes internal structure
and patterns, with coarse-to-fine manner, DIP-based 3DGS
can operate in scenarios where vanilla 3DGS fails, such as
sparse view recovery. Note that our approach does not use
any pre-trained models such as generative models and depth
estimation, but rather relies only on the input frames. Among
such methods, DIP-GS obtains state-of-the-art (SOTA) com-
petitive results on various sparse-view reconstruction tasks,
demonstrating its capabilities.
Figure 1: DIP-GS general scheme: First, the method starts
by running vanilla 3DGS to get initial Gaussians. Next, DIP
fitting and post-processing are applied sequentially.
Introduction
The advent of 3D Gaussian Splatting (3DGS) (Kerbl et al.
2023) has revolutionized 3D scene representation, enabling
high-quality reconstructions with unprecedented real-time
rendering performance. This efficiency has surpassed many
prior methods that, despite achieving quality, often struggled
with high computational costs, large memory footprints, or
low frame rates (Mildenhall et al. 2021; Barron et al. 2021,
2022; M¨uller et al. 2022; Khatib and Giryes 2024). Conse-
quently, 3DGS has rapidly become a foundational technology
for diverse real-time applications, including AR/VR, 3D gen-
eration, and interactive 3D editing, etc. (Tang et al. 2023; Ren
et al. 2023; Qin et al. 2024; Liu, Zhou, and Huang 2024; Shen
et al. 2024; Kerbl et al. 2024; Wu et al. 2024). At its core,
3DGS models a scene as a collection of 3D Gaussians, whose
parameters are optimized to match a set of input views.
The main concept of 3DGS is to represent the scene as a
group of 3D Gaussians, where each one is characterized by
its center, opacity, scaling, rotation, and color. To render these
Gaussians on a specific view, Kerbl et al. (2023) proposed an
efficient differentiable rendering scheme as well. So, given a
collection of input views of a scene with their known camera
parameters, the Gaussians features are learned in an end-to-
end optimization process, while pushing the rendered views
to be as close as possible to the ground truth ones.
Despite its success with dense input views, vanilla 3DGS
encounters significant challenges in sparse-view reconstruc-
tion. When input views are few and offer limited overlap,
the optimization problem becomes severely ill-posed. This
often leads to over-parameterization, where numerous erro-
neous Gaussian configurations can fit the sparse input views,
resulting in poor generalization to novel viewpoints and un-
convincing 3D geometry. Addressing this sparsity challenge
typically involves two main strategies. The first relies on ex-
ternal, pre-trained models to provide regularization, such as
depth estimators or generative priors (Li et al. 2024; Gu´edon
et al. 2024; Zhu et al. 2024; Fan et al. 2024; Liu et al. 2024;
Sun et al. 2024; Wang et al. 2023). While often effective,
these methods are inherently limited by the diversity, robust-
ness, and potential biases of the pre-trained models, and their
performance can degrade if the pre-trained knowledge does
not align well with the target scene. The second strategy aims
to impose structural regularization directly on the 3DGS rep-
resentation itself (Yin et al. 2024; Yang, Pavone, and Wang
2023). However, the typically unstructured and unordered
nature of 3D Gaussians makes it non-trivial to design and
enforce meaningful structural priors.
In this paper, we introduce DIP-GS, a novel approach that
uniquely leverages DIP (Ulyanov, Vedaldi, and Lempitsky
2018) to instill robust structural regularization for sparse-
view 3DGS, without relying on any pre-trained models. Our
central novelty lies in re-purposing the inherent structure-
arXiv:2508.07372v1  [cs.CV]  10 Aug 2025

<!-- page 2 -->
inducing capabilities of DIP’s convolutional neural network
(CNN) architecture to directly generate a structured 2D grid
of 3D Gaussian parameters. DIP was originally proposed
for image restoration tasks, demonstrating that a randomly
initialized CNN can capture low-level image statistics and in-
ternal patterns from a single degraded image to perform tasks
like denoising or inpainting, effectively using the network
structure itself as a prior. Inspired by this, and by observations
that 3D Gaussian parameters can exhibit ’natural’ structure
when organized into a 2D grid (Morgenstern et al. 2024),
DIP-GS makes two key conceptual leaps:
1. Structured Gaussian Generation via DIP: Instead of
directly optimizing individual, unstructured Gaussians, DIP-
GS learns a DIP network that maps a fixed random noise
input to a 2D grid. Each ”pixel” in this grid’s output channels
corresponds to the parameters (mean, scale, rotation, opacity,
spherical harmonics) of a 3D Gaussian. This fundamentally
changes the representation from an unstructured set to an
implicitly structured one.
2. Learning Network Weights as an Implicit Prior: The
optimization target shifts from the vast number of individ-
ual Gaussian parameters to the comparatively constrained
weights of the DIP network. The CNN architecture itself,
with its spatial biases, enforces a strong prior for local co-
herence and plausible global structure across the generated
Gaussians. This inherent regularization is learned solely from
the internal statistics of the sparse input views.
This structured generation process, driven by optimizing
the DIP network, provides a powerful inductive bias that
helps to disambiguate scene geometry and appearance from
limited data, mitigating the overfitting issues common in
sparse-view 3DGS. Furthermore, we integrate this with a
coarse-to-fine optimization scheme, allowing DIP-GS to pro-
gressively refine the scene details.
By having a strong internally derived prior, DIP-GS can
operate effectively in scenarios where vanilla 3DGS fails and
achieves state-of-the-art (SOTA) competitive results among
pre-training free methods on various sparse-view reconstruc-
tion benchmarks. Our work demonstrates a novel pathway
for robust 3DGS reconstruction by leveraging the implicit
structural biases of neural network architectures, rather than
explicit pre-training on external data.
Related Works and Preliminaries
3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) is a 3D
reconstruction approach, in which the scene is explicitly rep-
resented by a group of optimizable Gaussians. Each Gaussian
is characterized by its mean µ ∈R3, scale s ∈R3, rotation
matrix R ∈R3×3, opacity o ∈R, and spherical harmonics
color coefficients sh ∈R3(l+1)2 with degree l for view-
dependent color. The covariance matrix of each Gaussian
is Σ = RSST RT , in which S is a diagonal matrix with
the scaling s as the diagonal, thus the Gaussian is defined
by G(x) = e−1
2 xT Σ−1x. Given view transform W and the
Jacobian of the affine approximation of the projective trans-
formation J the covariance of the view projected (splatted)
2D Gaussian is Σ′ = JW ΣW T JT . To render the Gaus-
sians, first they are sorted with respect to their distance from
the view origin, then the color of each pixel ˆc is calculated:
ˆc =
N
X
i=1
Ti · αi · ci,
where
Ti =
i−1
Y
j=1
(1 −αi).
(1)
where ci is the color of Gaussian i in order, and αi =
oi · e−1
2 ∆T
i Σ′−1∆i, where ∆i is the offset between the 2D
mean of a splatted Gaussian and the pixel coordinate. The
Gaussian parameters are optimized to minimize the recon-
struction loss ∥c −ˆc∥2, where c is the pixel ground-truth
color. The optimized rendering algorithm enables fast train-
ing and real-time FPS rendering of 3DGS. During training,
Gaussians are added and removed via heuristics as in Kerbl
et al. (2023); Kheradmand et al. (2024).
DIP (Ulyanov, Vedaldi, and Lempitsky 2018), given a
degraded image x0, the method recovers its cleaned version
ˆx by optimizing the parameters of a neural function fθ that
maps a uniform random noise image z to x0 by solving
minθ ∥fθ(˜z) −x0∥2, where ˜z = z + σu, u ∼N(0, 1) and
σ is a scale that is dependent of the recovery task and it
prevents overfitting the degraded image. As demonstrated in
(Ulyanov, Vedaldi, and Lempitsky 2018), by using the prior
bias of a neural network, DIP manages to understand the
internal structure and patterns, and to recover natural images
with respectable performance in tasks such as denoising,
super-resolution, inpainting and etc, without any pre-training.
3DGS Grid (Morgenstern et al. 2024) is a scheme for
organizing Gaussians in a 2d grid such that the features of
the Gaussians form 2D grid channels with ’natural’ structure.
Thus, these channels can be compressed and saved using
any image compression method, such as JPEG compression.
Thus, by mainly re-organizing the Gaussians, this method
managed to integrate image-based compression methods into
3DGS. Inspired by the same idea, we use DIP, which outputs
a naturally structured 2D image, to represent the Gaussians.
Sparse-View 3D Gaussian Splatting. While 3DGS (Kerbl
et al. 2023) excels with dense input views, its performance
significantly degrades in sparse-view scenarios due to the
ill-posed nature of the task. This often leads to overfitting to
the input views and poor novel view generalization. Several
strategies have been proposed to mitigate this.
One common approach involves incorporating external
priors, often derived from pre-trained models. For instance,
DNGaussian (Li et al. 2024) optimizes sparse-view 3D Gaus-
sian radiance fields by introducing global-local depth nor-
malization, leveraging depth information (potentially from
monocular depth estimators) to regularize the geometry.
FSGS (Zhu et al. 2024) focuses on real-time few-shot view
synthesis using Gaussian Splatting, also implicitly or explic-
itly relying on strong priors learned during their specialized
training or initialization to densify the sparse Gaussians effec-
tively for novel view synthesis. Similarly, FewViewGS (Yin
et al. 2024) employs view matching techniques, often using
pre-trained 2D feature extractors, and a multi-stage training
strategy to enhance reconstruction from few views. While
effective, these methods can be constrained by the general-
ization capabilities of the pre-trained models they depend
on (e.g., for depth estimation or feature extraction), or may

<!-- page 3 -->
require specific types of prior information that might not
always be available or perfectly align with the target scene,
which can influence the reconstruction quality.
Another line of work focuses on imposing structural reg-
ularization directly or learning priors from the input data
itself, without relying on extensively pre-trained networks
on external datasets. For example, FreeNeRF (Yang, Pavone,
and Wang 2023), although developed for NeRFs, demon-
strated improvements in few-shot neural rendering using fre-
quency regularization, a self-driven prior that discourages
high-frequency artifacts common in undertrained models.
However, imposing meaningful structural priors directly onto
the typically unordered and unstructured set of 3D Gaussians
can be challenging, as noted in the introduction.
Our proposed DIP-GS method aligns with this latter cat-
egory but offers a distinct approach tailored for 3DGS. By
leveraging DIP (Ulyanov, Vedaldi, and Lempitsky 2018),
DIP-GS learns to generate a structured 2D grid of Gaussian
parameters from a random noise input. This inherently regu-
larizes the Gaussian representation through the architectural
biases of the CNN used in DIP, encouraging spatial coher-
ence and plausible scene structure. Unlike methods reliant on
pre-trained depth estimators or 2D feature extractors, DIP-GS
capitalizes on the internal statistics and self-similarity, guided
by the structure-inducing capabilities of the DIP. This makes
our approach pre-training free and more adaptable to diverse
scenes where external priors might fail, introduce bias, or are
simply unavailable. Our method’s advantage lies in its ability
to extract and enforce regularity from the sparse data alone,
offering robust reconstruction by design without dependence
on external pre-trained knowledge.
DIP Gaussian Splatting (DIP-GS)
In DIP-GS, instead of learning the Gaussians features directly,
we learn a DIP function fθ that maps a uniform random noise
image z to x0, forming an organized Gaussian grid. By using
the prior bias of a neural network, we obtain a regularized
structure for the Gaussians. In case of n2 output Gaussians
and a uniform random noise vector z, the output Gaussian
features are obtained by (µ, o, s, r, sh) = fθ(z), where
µ ∈Rn×n×3, o ∈Rn×n×1, s ∈Rn×n×3, r ∈Rn×n×4,
sh ∈Rn×n×3(l+1)2 with degree l for view-dependent color.
As shown in (Kerbl et al. 2023), the rotation matrices R
are build from the quaternion vector r and scale matrices S
contain s in the diagonal. After obtaining the 2D organized
Gaussian grid as the output of fθ, the grid is splitted to form
the desired Gaussian structure, where each entry in the grid
corresponds to a single Gaussian, and the Gaussian features
are the corresponding parameters of this entry, resulting in n2
Gaussians {(µi,j, oi,j, si,j, ri,j, shi,j)}0≤i,j<n, as in (Mor-
genstern et al. 2024). It is worth mentioning that the Gaus-
sian 2D organized grid is only an internal structure, which is
split to unordered collection to form a regular 3DGS struc-
ture, and from an outside black-box perspective, this method
generates regular unordered set of Gaussians. Thus, these
Gaussians can be passed to any Gaussian related pipeline,
such as rendering. The function fθ is compound of several
DIP networks, where each one of the 5 channels is regressed
by a separate DIP, however, they all operate on the same
input, fθ(z) = (f µ
θµ(z), f o
θo(z), f s
θs(z), f r
θr(z), f sh
θsh(z)), in
which they output the Gaussian features (µ, o, s, r, sh).
Given a sparse collection of views of a given scene, first,
we start by running the vanilla 3DGS on them with ran-
dom initialization to obtain an initial Gaussians estimation
(µinit, oinit, sinit, rinit, shinit), as shown in Figure 1. As
shown in (Yin et al. 2024; Li et al. 2024) and Figure 1, the
initial estimation usually describes inaccurate geometry with
poor novel view performance. DIP-GS starts by fitting the
Gaussian centers µ = f µ
θµ(z) to be close to the initial esti-
mation centers µinit by minimizing the Chamfer Distance
(CD) between these two point clouds:
min
θµ CD(f µ
θµ(˜z), µinit)
(2)
in which ˜z = z + σu, u ∼N(0, 1), and σ is the noise scale.
The operation is illustrated in Figure 2-(a).
After initializing the Gaussian means, the Gaussian scales
s = f s
θs(z) are initialized. To do that, an initial guess of the
scale of each Gaussian sest is calculated to be the average
distance of the 3 closest Gaussians from the previously ini-
tialized means µ, as in vanilla 3DGS. Next, the Gaussian
scales s = f s
θs(z) are fitted to be as close as possible to the
initial scale guess sest (Figure 2-(b)):
min
θs
f s
θs(˜z) −sest

(3)
This initialization process is important for the method’s
convergence, since it is known in the GS literature that a
good initialization is crucial for good recovery, as is demon-
strated in the experiment section. Next, the rendering-based
optimization process starts, where the DIP parameters are
learned to minimize the rendered view loss:
min
θ
X
r∈train
Ph(πr(fθ(˜z)), xr)+
β
(f o
θo(˜z)

1 + γ
(f s
θs(˜z)

1
(4)
where Ph is the rendering photometric loss used in (Kerbl
et al. 2023), πr is the Gaussian rendering operation, xr the
ground-truth image, and β, γ are parameters for the opacity
and scale regularization, see Figure 2-(c).
Since scene recovery from sparse views is an ill-posed
problem, in some cases, floaters close to the camera may ap-
pear in the novel views, as shown in (Yang, Pavone, and Wang
2023). Thus, to prevent that, an occlusion regularization is
added to penalize Gaussians that are close to the cameras, in
the same spirit of the occlusion regularization introduced in
FreeNeRF (Yang, Pavone, and Wang 2023). Given a Gaus-
sian and a view, the bounding box corners of the Gaussian
are estimated, then the depth of the closest corner to the view
origin is calculated - d, and the occlusion regularization for
this Gaussian on this view is o·ReLU(1−d/dmin), where o
is its opacity and dmin is a hyperparameter. Thus, if the depth
d is less than dmin (close to the view) the Gaussian is forced
to either move away from the view or have low opacity. The
final occlusion regularization occ reg is obtained by calcu-
lating the mean of this term over all Gaussians with all views,
and the hyperparameter δ as its weight in the loss term.

<!-- page 4 -->
When the optimization finishes, the output Gaussian are
obtained by using the original noise vector z (and not ˆz)
(µ, o, s, r, sh) = fθ(z). Next, a post-processing stage is
applied using vanilla GS with these Gaussians as the initial
Gaussians, and their test views renderings for test views su-
pervision. At each iteration, the method selects an image from
the estimated test views with
p
1+p probability and an image
from the input sparse views with
1
1+p probability, where p is
a hyperparameter called dominance factor. Since in the DIP
optimization the Gaussians are not directly optimized, this
means that Gaussians cannot be cloned or split, thus, the GS
post-processing operation comes to introduce densification
into this scheme. This operation is shown in Figure 2-(d).
DIP-GS operates in a coarse-to-fine manner, it first starts
with a high noise scale σ, since the initial Gaussians esti-
mation is not accurate. Following that, the output Gaussians
from the GS postprocess step will be the input Gaussian for
the DIP-GS fit process again, followed by another GS post-
process, but this time with equal or smaller noise scale. Thus,
the DIP-GS fit then GS post process stages are applied sev-
eral times with non-increasing noise level values, where the
output Gaussians of each stage are the input for the next one.
This coarse-to-fine approach enables a sequential recovery
of the scene, where at the beginning, when the noise scale is
high, the recovered scene will be blurry, but with good gen-
eralization in the unseen views. Then, when the noise level
drops gradually, more and more details will be recovered.
DIP-GS general scheme is described in Figure 1.
The DIP functions are represented by an Encoder-Decoder
Unet architecture with skip connections, where each layer is
compound of a convolution followed by a normalization and
then activation. Unlike in DIP, where the uniform random
noise vector z was only fed to the Encoder as its input, in this
method z is fed also to the inner intermediate layers. Thus,
in case of n2 output Gaussians, z = z1, z2, ... in which z1 ∈
Rn×n×d1 is the input to the network (The equivalent of DIP),
z2 ∈R
n
2 × n
2 ×d2 which will be concatenated to the output of
the first downscaling layer in the network, and so on. The
injection of the random vector into the intermediate results
reduces the overfitting and ensures a plausible generalization
output to the unseen areas, see Figure 3.
Experiment Results
In the sparse view recovery task, the goal is to recover the
scene given sparse input views with the camera parameters.
The proposed method is applied on several datasets, the first
one is the Blender (Mildenhall et al. 2021) dataset, which
consists of 8 synthetic scenes with complex geometry and
coloring. The second one is LLFF (Mildenhall et al. 2019)
dataset, which consists of 8 real-world scenes, and the third
one is composed of 15 selected scenes from the DTU (Jensen
et al. 2014) dataset. Following the setup in (Li et al. 2024) in
the Blender case, the method is trained on 8 specific images
per scene, with 1
2 of the original resolution. For LLFF and
DTU, the method is trained on 3 specific images per scene
with 1
8 and 1
4 of the original resolution respectively.
Implementation details. We provide below the implementa-
tion details of the various stages of our DIP-GS pipeline:
Method
PSNR↑
SSIM↑
LPIPS↓
3DGS w/o occ reg
14.452
0.734
0.223
3DGS w occ reg
16.299
0.744
0.206
1st stage
18.454
0.797
0.174
2nd stage
19.225
0.816
0.159
3rd stage
19.657
0.828
0.150
4th stage - final ours
19.798
0.836
0.144
Unet with z as only input
18.255
0.808
0.163
W/o opacity reg
19.084
0.813
0.159
W/o scale reg
19.477
0.821
0.154
W/o occ reg
14.106
0.783
0.200
W/o µ init
15.484
0.720
0.233
W/o s init
17.501
0.785
0.182
One Unet
18.497
0.792
0.171
Two Unets
19.070
0.830
0.1485
Dom. factor p = 0.
18.066
0.804
0.160
Table 1: Ablation table on DTU to illustrate the contribu-
tion of DIP-GS design choices. In up-to-down order, first
the contribution of the occ. reg on the inital stage. Then the
contribution of the coarse-to-fine strategy. Next, the method
is applied without Unet intermediate injection, opacity, scale
and occ. regularizations, mean and scale initialization. Fol-
lowing that, the method is tested with one and two Unets.
Finally, it is tested without dominance factor.
1. Initial 3DGS Estimation: The process commences by
running a modified version of vanilla 3DGS. The primary
modification is that the opacity reset operation commonly
used in 3DGS densification is disabled. Also, an opacity βinit
and scaling γinit regularization term are introduced, as in
(Kheradmand et al. 2024), alongside occlusion regularization
δinit for some cases. The initial 3D Gaussians are seeded
from random points within the scene’s bounding box. Let
Ninit be the number of Gaussians obtained from this stage.
2. DIP-GS Network Architecture and Initialization: The
core of our method, the DIP-GS generator fθ, is designed
to produce n2 = 0.75 × Ninit Gaussians. This generator
fθ is comprised of five distinct U-Net architectures (Ron-
neberger, Fischer, and Brox 2015) with 3 down/up scale
stages, each dedicated to regressing a specific component
of the Gaussian parameters: mean µ ∈Rn×n×3, opacity
o ∈Rn×n×1, scale s ∈Rn×n×3, rotation (represented as
a quaternion r ∈Rn×n×4), and spherical harmonics coeffi-
cients sh ∈Rn×n×3 (where the SH degree is chosen l = 0).
All five U-Nets receive the same fixed random noise tensor
z as input, where the input channel dimension d is set to
32, and the intermediate channels dimensions are 16, 32, 64.
Each element of this input noise tensor z is independently
sampled from a uniform distribution U(0, 0.1), as in DIP.
3. DIP-GS Optimization: The optimization of the DIP-GS
network parameters θ proceeds in 3 phases: (i) Chamfer Dis-
tance Fitting (Means): For an initial 3000 iterations, only
the U-Net responsible for generating the Gaussian means
(µ = f µ
θ (z)) is optimized. The objective is to minimize
the point cloud Chamfer Distance between these generated
means and the means of the Gaussians obtained from the
initial 3DGS estimation stage, see Figure 2-(a); and (ii) Scale
Fitting: For an initial 3000 iterations, only the U-Net re-
sponsible for generating the Gaussian scales (s = f s
θ (z)) is
optimized to be as close as possible to the estimates scales,
see Figure 2-(b); and (iii) Rendering Loss Optimization (Full

<!-- page 5 -->
Figure 2: DIP-GS components at a given noise level. (a) - First, the mean’s network f µ
θµ is initialized by minimizing the point
cloud Chamfer Distance between its output µ, which is mapped from the noise ˜z, and the initial Gaussians means µinit. (b) -
Second, the scale’s network f s
θs is initialized by fitting the output scale channel s, which is mapped from the noise ˜z, to the
estimated scale guess sest. (c) - Next, the DIP optimization, in which fθ maps ˜z to the Gaussian features, and θ is learned by
minimizing the render loss alongside other regularizations. (d) - The post-processing stage, where the Gaussians are initialized
by the output of the DIP fθ that was trained in the previous stage. At each step, the method chooses a frame either from the
sparse input views or one of the target views.
Network): Subsequently, for 4000 iterations, the parameters
θ of all five U-Nets are jointly optimized by minimizing a
photometric rendering loss between the rendered views from
DIP-GS and the ground truth input views. During this phase,
the opacity, scale and occlusion regularizations are added
with weights β, γ, and δ respectively.
4. Post-processing with Vanilla 3DGS: After the DIP-GS
optimization, the Gaussians generated by fθ(z) are used to
initialize a standard vanilla 3DGS optimization. This post-
processing step helps to further refine the Gaussians and
introduce densification where needed. During this vanilla
3DGS optimization, a dominance factor p = 0.1 is used
for selecting between training on input views versus self-
supervision from rendered test views. Similar to previous
stages, opacity, scale and occlusion regularizations are also
added with weights βpost, γpost, and δpost, respectively. The
values of these hyper-parameters are attached in supp. mat. It
is worth emphasizing that scale occlusion regularizations are
only used with DTU dataset.
5. Coarse-to-Fine Strategy: The entire procedure, encom-
passing DIP-GS optimization (steps 3a and 3b) followed by
GS post-processing (step 4), is embedded within a coarse-to-
fine framework spanning 4 stages. In each stage k, the input
noise tensor to the DIP U-Nets is perturbed as z′
k = z +
σk · N(0, I), where N(0, I) is standard Gaussian noise and
σk is a stage-dependent noise level. The sequence of noise
levels used is σ = [σ1, σ2, σ3, ...] (e.g., progressively non-
increasing values such as σ = [0.0333, 0.01, 0.005, 0.002]
for our case. This allows the model to first capture coarse
scene structure with higher noise and then refine details as
the noise level decreases.
Ablation Studies. We conduct ablation studies on the DTU
dataset to assess the contribution of each component in DIP-
GS, see Tab. 1. The results, in up-to-down order, show that
occlusion regularization improves performance in the initial
3DGS stage, while the coarse-to-fine strategy with decreasing
noise levels enables the model to progressively refine details.
Injecting the noise vector z into intermediate UNet layers
rather than only at the encoder input yields better results, and
adding opacity, scale, and occlusion regularizations provides

<!-- page 6 -->
Blender
LLFF
DTU
8 Images
3 Images
3 Images
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
DietNeRF*
23.147
0.866
0.109
14.94
0.370
0.496
11.85
0.633
0.314
FewViewGS*
25.550
0.886
0.092
18.96
0.585
0.307
19.13
0.792
0.186
RegNeRF**
23.86
0.852
0.105
19.08
0.587
0.336
18.89
0.745
0.190
SparseNeRF**
24.04
0.876
0.113
19.86
0.624
0.328
19.55
0.769
0.201
DNGaussian**
24.305
0.886
0.088
19.12
0.591
0.294
18.91
0.790
0.176
FSGS**
24.64
0.895
0.095
20.31
0.652
0.288
−
−
−
Mip-NeRF
20.89
0.830
0.168
14.62
0.351
0.495
8.68
0.571
0.353
3DGS
22.226
0.858
0.114
15.52
0.408
0.405
14.45
0.734
0.223
FreeNeRF
24.259
0.883
0.098
19.63
0.612
0.308
19.92
0.787
0.182
DIP-GS (ours)
25.90
0.898
0.087
20.13
0.662
0.221
19.79
0.836
0.144
Table 2: Sparse recovery on Blender, LLFF and DTU datasets. Bold is best, underline is second. First category methods (no
pre-trained used) are not denoted, methods from the second category (non 3D-aware pre-trained used) are denoted with *, and
methods from the third category (3D-aware pre-trained used) are denoted with **. For fair comparison with FewViewGS, we
attach the results with random init for the initial 3DGS stage, such as in DIPGS and the other methods.
LLFF
DTU
6 Images
9 Images
6 Images
9 Images
Method
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
PSNR↑
SSIM↑
LPIPS↓
FewViewGS*
21.33
0.688
0.220
23.09
0.769
0.164
23.51
0.891
0.123
25.75
0.925
0.101
RegNeRF**
23.10
0.760
0.206
24.86
0.820
0.161
19.10
0.757
0.233
22.30
0.823
0.184
DNGaussian**
22.18
0.755
0.198
23.17
0.788
0.180
−
−
−
−
−
−
FSGS**
24.55
0.795
0.177
25.89
0.845
0.143
−
−
−
−
−
−
Mip-NeRF
22.91
0.756
0.213
24.88
0.826
0.160
−
−
−
−
−
−
3DGS
20.36
0.664
0.252
21.49
0.717
0.254
21.85
0.870
0.122
24.60
0.918
0.086
FreeNeRF
23.73
0.779
0.195
25.13
0.827
0.160
22.39
0.779
0.240
24.20
0.833
0.187
DIP-GS (ours)
24.39
0.829
0.125
25.57
0.855
0.108
24.15
0.902
0.104
25.97
0.928
0.087
Table 3: 6 and 9 views scene recovery on LLFF and DTU datasets. Bold is best, underline is second. As noticed, DIP-GS
outperforms other methods in most of the cases. For fair comparison with FewViewGS, we attach the results with random init for
the initial 3DGS stage, such as in DIP-GS and the other methods.
Figure 3: DIP Unet Architecture. The noise vector z (green)
is fed as an input and also injected and concatenated into
intermediate outputs (blue). Dash-lines are the skip cons.
a notable boost. Removing mean and scale initialization sig-
nificantly degrades performance, confirming its importance.
Additionally, using 5 specialized UNets outperforms simpler
configurations such as one UNet for all features or two UNets,
one for the mean feature and the other for the rest. Finally,
the dominance factor in post-processing proves essential for
the final performance.
Comparison with State-of-the-Art. We compare DIP-GS
with both NeRF and 3DGS based sparse recovery methods.
These methods are Mip-NeRF (Barron et al. 2021), Diet-
NeRF (Jain, Tancik, and Abbeel 2021) Reg-NeRF (Niemeyer
et al. 2022), FreeNeRF (Yang, Pavone, and Wang 2023),
SparseNeRF (Wang et al. 2023), 3DGS (Kerbl et al. 2023),
DNGaussian (Li et al. 2024), FSGS (Zhu et al. 2024) and
FewViewGS (Yin et al. 2024). These methods are distributed
between 3 categories, the first category are methods that do
not use any form of pre-trained methods, and these methods
are Mip-NeRF, 3DGS and FreeNeRF. The second category
are methods that use pre-trained methods that are not trained
with 3D-aware data, such as image feature extractors, and
these methods are DietNeRF and FewViewGs. The third cate-
gory are methods that use 3D aware pre-trained methods such
as depth predictors or methods trained on 3D data, and these
methods are RegNeRF, SparseNeRF, DNGaussian and FSGS.
DIP-GS belongs to the first category since it only relies on
the structural regularization that DIP provides.
The quantitative results are presented in Table2. On the
Blender dataset (8 input views), DIP-GS achieves state-of-
the-art performance across all metrics, with a PSNR of 25.90,
SSIM of 0.898, and LPIPS of 0.087. Notably, it outper-
forms methods from all categories, including those relying
on pre-trained models like FewViewGS* (PSNR 25.550) and
FSGS** (PSNR 24.64). This demonstrates the strong regu-
larization capability of DIP-GS in complex synthetic scenes.
On the LLFF dataset (3 input views), which features real-
world scenes and is generally more challenging for sparse-
view reconstruction, DIP-GS remains highly competitive.
It achieves a SOTA competitive PSNR of 20.13 with a mi-
nor difference, and outperforms other methods with SSIM

<!-- page 7 -->
3DGS
DNGaussian
FreeNeRF
DIP-GS
GT
Figure 4: DTU and LLFF qualitative results. The qualitative results clearly illustrate the capabilities of DIP-GS.
of 0.662 and LPIPS of 0.221. More importantly, DIP-GS
significantly outperforms other pre-training-free methods
like FreeNeRF (PSNR 19.63, SSIM 0.612, LPIPS 0.308)
and vanilla 3DGS (PSNR 15.52) with a margin. In the DTU
dataset (3 input views), we obtain similar behaviour in which
our method is SOTA competitive and outperforms it in many
cases, as demonstrated in Table 2 and Figure 4. Overall, DIP-
GS demonstrates SOTA competitive capabilities and results.
Table 3 presents the quantitative results on the LLFF and
DTU datasets with 6 and 9 views respectively. Similar to the
3 views case, our method manages to obtain SOTA results
in most cases, specifically outperforms the pre-training free
methods in these cases. These results further demonstrates
the robustness and superiority of DIP-GS method. Because of
the Unet activation at each iteration, DIP-GS method comes
with a runtime overhead, in which each stage in it (DIP fitting
+ post-processing) takes approximately 20 minutes to run on
a A5000 gpu. Thus, in case of runtime limitation, one may
sacrifice the performance little bit for the sake of runtime by
letting DIP-GS run for fewer stages, or even one.
Conclusion
In this paper, we proposed DIP-GS, a method that gener-
ates Gaussians using DIP network, enabling a strongly regu-
larized structure without pre-trained models. Our approach
achieves competitive results on sparse-view datasets thanks
to its regularized and structured prior. However, DIP-GS has
some limitations: it incurs higher training time compared to
standard GS-based methods due to the need to run a neural
network at each optimization step, and it currently relies on a
separate 3DGS post-processing stage for Gaussian densifica-
tion. We leave integrating densification step directly into the
DIP framework for future work, which could improve both
efficiency and reconstruction quality. More broadly, DIP-GS
paves the way for incorporating other DIP-based techniques,
and single-image-based methods in general, into 3DGS, po-
tentially enabling tasks like super-resolution, inpainting, and
foreground-background separation.

<!-- page 8 -->
References
Barron, J. T.; Mildenhall, B.; Tancik, M.; Hedman, P.; Martin-
Brualla, R.; and Srinivasan, P. P. 2021. Mip-nerf: A multiscale
representation for anti-aliasing neural radiance fields. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, 5855–5864.
Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.;
and Hedman, P. 2022. Mip-nerf 360: Unbounded anti-aliased
neural radiance fields. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
5470–5479.
Fan, Z.; Wen, K.; Cong, W.; Wang, K.; Zhang, J.; Ding, X.;
Xu, D.; Ivanovic, B.; Pavone, M.; Pavlakos, G.; et al. 2024.
InstantSplat: Sparse-view SfM-free Gaussian Splatting in
Seconds. arXiv preprint arXiv:2403.20309.
Gu´edon, A.; Ichikawa, T.; Yamashita, K.; and Nishino, K.
2024. MAtCha Gaussians: Atlas of Charts for High-Quality
Geometry and Photorealism From Sparse Views.
arXiv
preprint arXiv:2412.06767.
Jain, A.; Tancik, M.; and Abbeel, P. 2021. Putting nerf on
a diet: Semantically consistent few-shot view synthesis. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, 5885–5894.
Jensen, R.; Dahl, A.; Vogiatzis, G.; Tola, E.; and Aanæs,
H. 2014. Large scale multi-view stereopsis evaluation. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, 406–413.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3d gaussian splatting for real-time radiance field ren-
dering. ACM Transactions on Graphics (ToG), 42(4): 1–14.
Kerbl, B.; Meuleman, A.; Kopanas, G.; Wimmer, M.; Lanvin,
A.; and Drettakis, G. 2024. A hierarchical 3d gaussian repre-
sentation for real-time rendering of very large datasets. ACM
Transactions on Graphics (TOG), 43(4): 1–15.
Khatib, R.; and Giryes, R. 2024. TriNeRFLet: A Wavelet
Based Triplane NeRF Representation. In European Confer-
ence on Computer Vision, 358–374. Springer.
Kheradmand, S.; Rebain, D.; Sharma, G.; Sun, W.; Tseng, Y.-
C.; Isack, H.; Kar, A.; Tagliasacchi, A.; and Yi, K. M. 2024.
3d gaussian splatting as markov chain monte carlo. Advances
in Neural Information Processing Systems, 37: 80965–80986.
Li, J.; Zhang, J.; Bai, X.; Zheng, J.; Ning, X.; Zhou, J.; and
Gu, L. 2024. Dngaussian: Optimizing sparse-view 3d gaus-
sian radiance fields with global-local depth normalization. In
Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, 20775–20785.
Liu, F.; Sun, W.; Wang, H.; Wang, Y.; Sun, H.; Ye, J.; Zhang,
J.; and Duan, Y. 2024. Reconx: Reconstruct any scene from
sparse views with video diffusion model. arXiv preprint
arXiv:2408.16767.
Liu, X.; Zhou, C.; and Huang, S. 2024.
3dgs-enhancer:
Enhancing unbounded 3d gaussian splatting with view-
consistent 2d diffusion priors. Advances in Neural Infor-
mation Processing Systems, 37: 133305–133327.
Mildenhall, B.; Srinivasan, P. P.; Ortiz-Cayon, R.; Kalantari,
N. K.; Ramamoorthi, R.; Ng, R.; and Kar, A. 2019. Local
light field fusion: Practical view synthesis with prescriptive
sampling guidelines. ACM Transactions on Graphics (ToG),
38(4): 1–14.
Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ra-
mamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes
as neural radiance fields for view synthesis. Communications
of the ACM, 65(1): 99–106.
Morgenstern, W.; Barthel, F.; Hilsmann, A.; and Eisert, P.
2024. Compact 3d scene representation via self-organizing
gaussian grids. In European Conference on Computer Vision,
18–34. Springer.
M¨uller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. Instant
neural graphics primitives with a multiresolution hash encod-
ing. ACM Transactions on Graphics (ToG), 41(4): 1–15.
Niemeyer, M.; Barron, J. T.; Mildenhall, B.; Sajjadi, M. S.;
Geiger, A.; and Radwan, N. 2022. Regnerf: Regularizing
neural radiance fields for view synthesis from sparse inputs.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 5480–5490.
Qin, M.; Li, W.; Zhou, J.; Wang, H.; and Pfister, H. 2024.
Langsplat: 3d language gaussian splatting. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 20051–20060.
Ren, J.; Pan, L.; Tang, J.; Zhang, C.; Cao, A.; Zeng, G.; and
Liu, Z. 2023. Dreamgaussian4d: Generative 4d gaussian
splatting. arXiv preprint arXiv:2312.17142.
Ronneberger, O.; Fischer, P.; and Brox, T. 2015.
U-net:
Convolutional networks for biomedical image segmenta-
tion. In Medical image computing and computer-assisted
intervention–MICCAI 2015: 18th international conference,
Munich, Germany, October 5-9, 2015, proceedings, part III
18, 234–241. Springer.
Shen, Y.; Ceylan, D.; Guerrero, P.; Xu, Z.; Mitra, N. J.; Wang,
S.; and Fr¨uhst¨uck, A. 2024. Supergaussian: Repurposing
video models for 3d super resolution. In European Confer-
ence on Computer Vision, 215–233. Springer.
Sun, W.; Chen, S.; Liu, F.; Chen, Z.; Duan, Y.; Zhang, J.; and
Wang, Y. 2024. Dimensionx: Create any 3d and 4d scenes
from a single image with controllable video diffusion. arXiv
preprint arXiv:2411.04928.
Tang, J.; Ren, J.; Zhou, H.; Liu, Z.; and Zeng, G. 2023.
Dreamgaussian: Generative gaussian splatting for efficient
3d content creation. arXiv preprint arXiv:2309.16653.
Ulyanov, D.; Vedaldi, A.; and Lempitsky, V. 2018. Deep
image prior. In Proceedings of the IEEE conference on com-
puter vision and pattern recognition, 9446–9454.
Wang, G.; Chen, Z.; Loy, C. C.; and Liu, Z. 2023. Sparsenerf:
Distilling depth ranking for few-shot novel view synthesis.
In Proceedings of the IEEE/CVF international conference on
computer vision, 9065–9076.
Wu, J.; Bian, J.-W.; Li, X.; Wang, G.; Reid, I.; Torr, P.; and
Prisacariu, V. A. 2024. Gaussctrl: Multi-view consistent text-
driven 3d gaussian splatting editing. In European Conference
on Computer Vision, 55–71. Springer.

<!-- page 9 -->
Yang, J.; Pavone, M.; and Wang, Y. 2023. Freenerf: Im-
proving few-shot neural rendering with free frequency regu-
larization. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 8254–8263.
Yin, R.; Yugay, V.; Li, Y.; Karaoglu, S.; and Gevers, T. 2024.
FewViewGS: Gaussian Splatting with Few View Matching
and Multi-stage Training. arXiv preprint arXiv:2411.02229.
Zhu, Z.; Fan, Z.; Jiang, Y.; and Wang, Z. 2024. Fsgs: Real-
time few-shot view synthesis using gaussian splatting. In
European conference on computer vision, 145–163. Springer.

<!-- page 10 -->
DIP-GS: Deep Image Prior For Gaussian Splatting Sparse View Recovery
Supplementary Materials
Technical Details
DIP-GS regularization weights are attached in Table 4. It is
worth mentioning that scale and occlusion regularizations are
only used with DTU dataset. For the mean and scale initializa-
tion stage, we use Adam optimizer with lr = 5e−3, 1e−3 re-
spectively, which we run each for 3000 steps as mentioned in
the paper. Before running the initialization process, all input
Gaussians with opacity lower than 0.005 are discarded. For
the DIP optimization, we use AdamW optimizer with weight
decay of 1e −5 and lr = 2e−4, 1e−3, 1e−3, 1e−3, 1e−3 for
(f µ
θµ(z), f o
θo(z), f s
θs(z), f r
θr(z), f sh
θsh(z)) respectively. The
DIP optimization runs for 4000 steps as mentioned in the
paper.
βinit
γinit
δinit
β
γ
δ
βpost
γpost
δpost
Blender
0.05
0
0
0.02
0
0
0.02
0
0
LLFF
0.1
0
0
0.02
0
0
0.05
0
0
DTU
0.1
0.1
20
0.02
0.01
20
0.05
0.01
20
Table 4: Hyperparameters values for the different datasets.
β, γ and δ are the weights for opacity, scale and occlusion
regularization respectively. Init for the initial stage, middle is
for the DIP optimization, and post for the post-processing.
Qualitative Results
Figures 5, 6, 7 include additional qualitative results for DIP-
GS method.

<!-- page 11 -->
3DGS
DNGaussian
FreeNeRF
DIP-GS
GT
Figure 5: LLFF qualitative results

<!-- page 12 -->
3DGS
DIP-GS
GT
Figure 6: Blender qualitative results.

<!-- page 13 -->
3DGS
DNGaussian
FreeNeRF
DIP-GS
GT
Figure 7: DTU qualitative results
