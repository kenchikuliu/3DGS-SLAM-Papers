<!-- page 1 -->
MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate
High-Fidelity Large-Scale Scenes
Kehua Chen1,2, Tianlu Mao1,2, Xinzhu Ma3, Hao Jiang1,2†, Zehao Li1,2, Zihan Liu1,2,
Shuqi Gao1, Honglong Zhao1, Feng Dai1, Yucheng Zhang1, Zhaoqi Wang1,2
1Institute of Computing Technology, Chinese Academy of Sciences, ICT
2University of Chinese Academy of Sciences, UCAS
3Beihang University
Figure 1. Illustration of the superiority of our method. (a) Our method accurately reconstructs the geometric structure of large-scale
urban scenes, faithfully restoring fine details such as buildings, vegetation, and roads. (b) Compared with the SOTA method CityGSV2 [34],
our result are more complete and geometrically precise. (c) Benefiting from a well-designed training framework, our method achieves
superior convergence speed and geometric quality. On four RTX 3090 GPUs, our method reaches better performance with less than 25%
of the training time required by CityGSV2. The quantitative results are reported based on the quality metrics of Modern Building [48].
Abstract
Recently, 3D Gaussian Splatting and its derivatives have
achieved significant breakthroughs in large-scale scene re-
construction. However, how to efficiently and stably achieve
high-quality geometric fidelity remains a core challenge. To
address this issue, we introduce MetroGS, a novel Gaussian
Splatting framework for efficient and robust reconstruction
in complex urban environments. Our method is built upon
a distributed 2D Gaussian Splatting representation as the
core foundation, serving as a unified backbone for sub-
sequent modules.
To handle potential sparse regions in
complex scenes, we propose a structured dense enhance-
ment scheme that utilizes SfM priors and a pointmap model
to achieve a denser initialization, while incorporating a
sparsity compensation mechanism to improve reconstruc-
tion completeness. Furthermore, we design a progressive
hybrid geometric optimization strategy that organically in-
tegrates monocular and multi-view optimization to achieve
efficient and accurate geometric refinement.
Finally, to
address the appearance inconsistency commonly observed
in large-scale scenes, we introduce a depth-guided ap-
pearance modeling approach that learns spatial features
with 3D consistency, facilitating effective decoupling be-
tween geometry and appearance and further enhancing
reconstruction stability.
Experiments on large-scale ur-
ban datasets demonstrate that MetroGS achieves superior
geometric accuracy, rendering quality, offering a unified
solution for high-fidelity large-scale scene reconstruction.
Project page: https://m3phist0.github.io/MetroGS.
1. Introduction
3D scene reconstruction is an essential topic in computer vi-
sion and graphics. Achieving large-scale, high-precision 3D
modeling serves as the foundational support for numerous
applications, such as aerial surveying [11, 17], autonomous
driving [2, 16, 26], immersive AR/VR [7, 8, 46]. Recent
achievements in 3D Gaussian Splatting (3DGS) have no-
1
arXiv:2511.19172v2  [cs.CV]  21 Mar 2026

<!-- page 2 -->
tably accelerated the translation of this goal toward practical
application [3, 9, 23, 51], demonstrating remarkable render-
ing efficiency and visual fidelity. However, current meth-
ods, while excelling in rendering quality [27, 30, 32, 53],
improvements in geometric reconstruction remain relatively
limited, leading to an imbalance between visual fidelity and
geometric accuracy. This imbalance highlights the need for
a scalable reconstruction framework capable of preserving
geometric accuracy under large-scale conditions.
In real-world urban environments, large-scale 3D recon-
struction must cope with multiple challenges, including ob-
jects with diverse structures and scales, varying illumination
conditions, and other complex factors. Most existing meth-
ods primarily focus on scaling up 2DGS [20] or PGSR [5]
frameworks, yet their geometric optimization strategies re-
main underdeveloped. Some methods [29, 34] rely solely
on single-view constraints, making it difficult to maintain
structural consistency. While others [6, 10, 28] adopt multi-
view consistency constraints but typically use single-scale
photometric constraints or simple reprojection errors, re-
sulting in limited adaptability to complex large-scale envi-
ronments. Moreover, illumination and exposure inconsis-
tencies are common in large-scale datasets [36, 55], forcing
models to reconcile appearance variations during optimiza-
tion, which compromises geometric consistency. Conven-
tional multi-view consistency optimization struggles to ef-
fectively address such issues. Meanwhile, we observe that
insufficient initial sampling in weakly textured or sparsely
observed regions is another key factor affecting geometric
quality, often leading to inaccurate recovery of local struc-
tures and resulting in surface holes or structural artifacts.
To overcome these challenges, we propose MetroGS, a
novel Gaussian Splatting framework that focuses on achiev-
ing Multi-view Efficient Tuning for Robust Optimization
in complex urban environments.
Specifically, we adopt
2DGS as the core representation for modeling 3D geometry
and utilize a distributed training strategy [57] to efficiently
support large-scale scene reconstruction.
Building upon
this foundation, we introduce a structured dense enhance-
ment scheme. During initialization, the training images are
partitioned based on SfM-derived priors, and a pre-trained
pointmap model [45] is employed to perform dense en-
hancement on the initial point cloud. In the subsequent den-
sification stage, an additional sparse-compensation mecha-
nism is incorporated to recover the remaining incomplete
regions, thereby improving the overall reconstruction com-
pleteness and quality.
Furthermore, we propose a pro-
gressive hybrid geometric refinement strategy. During the
early stage of training, we perform a lightweight monoc-
ular geometric optimization guided by priors from an off-
the-shelf depth estimator [44]. As training progresses, a
multi-view refinement is introduced. Inspired by [1, 49],
we adopt a carefully designed PatchMatch-based method to
refine the rendered depths, and further complete the refined
depth maps with monocular priors to obtain accurate and
complete depth estimates for subsequent fine-grained ge-
ometric optimization. This progressive design effectively
balances geometric accuracy and computational efficiency.
Finally, we address appearance inconsistencies by decou-
pling geometry and appearance. Specifically, we introduce
a depth-guided appearance modeling module that adopts a
Tri-Mip [19] structure to store spatial features of the scene.
By leveraging the high-quality optimized depth results,
the module queries geometry-aligned 3D-consistent feature
representations, thereby achieving efficient and stable ap-
pearance decoupling. Overall, these components form an
efficient and consistent framework for large-scale scene re-
construction, and extensive experiments on multiple large-
scale datasets validate its effectiveness.
Our main contributions can be summarized as follows:
• We design a structured dense enhancement scheme that
optimizes initialization and densification to compensate
for geometric deficiencies in sparse regions.
• We propose a progressive hybrid geometric refinement
integrating monocular and PatchMatch-based multi-view
optimization for efficient and accurate reconstruction.
• We introduce a depth-guided appearance module that in-
tegrates geometry and appearance to mitigate inter-image
variations and enhance reconstruction stability.
• Comprehensive experiments show that our method deliv-
ers superior reconstruction quality across diverse large-
scale scenes.
2. Related Works
2.1. Novel View Synthesis
Novel view synthesis aims to generate high-fidelity im-
ages from arbitrary viewpoints by learning an underlying
three-dimensional scene representation. The pivotal work
NeRF [37] implicitly models the scene using MLPs to en-
code color and density information for 3D points and view-
ing directions, enabling novel view synthesis of complex
scenes. To address performance limitations, methods exem-
plified by Tri-MipRF [19] employed more advanced feature
encoding techniques [4, 38] to improve both efficiency and
rendering quality. More recently, 3DGS [23] emerged as
another influential framework, which models scenes using
explicit 3D Gaussian primitives and achieves real-time ren-
dering through differentiable rasterization. Following this
breakthrough, a series of works [13, 50, 56] primarily fo-
cus on enhancing rendering quality. These advances collec-
tively inspire the design of our proposed algorithm.
2.2. Surface Reconstruction
The ability of surface reconstruction to generate accurate
3D geometry from diverse inputs is critical for realizing
2

<!-- page 3 -->
practical 3D technology. Recently, many advanced meth-
ods [18, 42, 52] have been developed extended from 3DGS.
PGSR [5] achieves high-fidelity and efficient surface re-
construction by introducing a planar Gaussian representa-
tion combined with unbiased depth rendering and multi-
view geometric regularization. 2DGS [20] enhances geo-
metric accuracy by substituting 3D Gaussians with surface-
oriented 2D surfels, addressing multi-view inconsistency
inherent in 3DGS, and serves as the foundational approach
adopted by the best current surface reconstruction meth-
ods [21, 47, 55]. Nevertheless, these approaches are mainly
optimized for object-level scenes and cannot be directly ap-
plied to large-scale scenes with reliable performance.
2.3. Large Scale Scene Reconstruction
The task of large-scale reconstruction demands coping with
vast amounts of data and more complex scene environ-
ments. Several recent works [30, 32, 39, 53, 57] have ex-
tended 3DGS to large-scale scenes, focusing on rendering
quality and efficiency improvements. In contrast, the explo-
ration dedicated to surface reconstruction remains at a rel-
atively early stage. CityGSV2 [34] continued the strategy
of partitioned parallel training, optimizing 2DGS to adapt
it for large-scale scene reconstruction, and simultaneously
established standard geometric benchmarks for large-scale
scenes. CityGS-X [15] introduced a scalable architecture
supporting multi-GPU parallel rendering, and jointly opti-
mizes the scene’s geometry and appearance through batch-
level multi-task training. While other methods [6, 14] have
extended surface reconstruction algorithms to large-scale
scenes, their simple geometric optimization struggles with
stability in complex large scenes. Our method advances
this field by introducing targeted geometric optimization for
more robust and higher-quality outcomes.
3. Preliminaries
3D Gaussian Splatting [23] models a scene as anisotropic
Gaussian primitives, each defined by its center, covariance,
opacity, and SH coefficients for view-dependent color. Ren-
dering is performed via front-to-back α-blending of the α-
weighted contributions along each ray:
C =
X
i∈N
ciαiTi,
Ti =
i−1
Y
j=1
(1 −αj).
(1)
2D Gaussian Splatting [20] flattens the 3D ellipsoid vol-
ume into 2D planar disks, making the primitives highly suit-
able for explicit 3D surface representation and optimization.
In 2DGS, depth calculation is primarily divided into mean
depth and median depth. The latter is considered more ro-
bust, as it utilizes visibility and treats Ti = 0.5 as the pivot
for surface and free space:
D = max{zi|Ti > 0.5}.
(2)
A regularization term aligning depth gradients with normals
enables 2DGS to achieve geometric optimization.
4. Method
Large-scale surface reconstruction tasks face multiple chal-
lenges, including the vast spatial extent of scenes, the insuf-
ficient quality of initial reconstruction points, the structural
diversity and complexity of objects, and the heterogeneity
of image data caused by inconsistent lighting conditions.
To address these challenges, we propose an efficient
and highly robust framework for large-scale scene recon-
struction. An overview of our method is shown in Fig. 2.
The main components of our method are structured as fol-
lows: Section 4.1 first introduces our fundamental paral-
lel training framework. Following this, the subsequent sec-
tions detail the key mechanisms designed to achieve high-
precision reconstruction. Specifically, Section 4.2 elabo-
rates on our structured dense enhancement scheme. Sec-
tion 4.3 then describes the progressive hybird geometric re-
finement method. Finally, Section 4.4 presents the depth-
guided appearance modeling.
4.1. Scalable Parallel Strategy
We extend 2DGS into a Gaussian-wise distributed training
paradigm inspired by the parallel concepts in [57]. Specif-
ically, the initialization point cloud is uniformly distributed
across multiple GPUs for local Gaussian initialization, and
multi-view batched training is employed to evenly assign
images among devices.
Each worker leverages the spa-
tial locality of Gaussian Splatting to fetch only the re-
quired Gaussian subsets, enabling efficient communication.
During dynamic densification, load balance is maintained
through periodic Gaussian redistribution. This distributed
design maximizes computational resource utilization and
demonstrates excellent scalability, allowing efficient sup-
port for large-scale scene reconstruction.
4.2. Structured Dense Enhancement
Gaussian initialization is based on 3D points from SfM [40].
However, even in large-scale scenes with dense coverage,
the presence of sparse-view or weak-texture regions leads
to an overly sparse initial point cloud. To mitigate this, we
introduce a structured dense enhancement scheme that sep-
arately optimizes initialization and densification.
4.2.1. Pointmap Model Assisted Initialization
We incorporate the pointmap model [45] to obtain auxil-
iary initial dense point clouds for Gaussian initialization,
leveraging its capability for efficient 3D structure predic-
tion. We first construct an undirected image graph G =
(V, E), where each node represents an image and each edge
weight wij corresponds to the number of inter-image fea-
ture matches estimated by SfM. The graph is partitioned
3

<!-- page 4 -->
Figure 2. Overview. Starting with the input image sequences, we first utilize the prior information provided by SfM, combined with a
pointmap model, to generate a high-quality initial point cloud. Next, an additional sparsity compensation optimization is introduced during
the densification process to further refine sparse regions. We then combine monocular depth priors with multi-view consistency optimiza-
tion to achieve progressive hybrid geometric refinement. Simultaneously, a depth-guided appearance modeling module is employed to
decouple geometry and appearance, thereby enhancing reconstruction fidelity.
into N clusters, matching the number of available GPUs,
by minimizing the normalized cut objective:
Ncut(A1, . . . , AN) =
N
X
k=1
Cut(Ak, ¯Ak)
Vol(Ak)
,
(3)
where Cut(Ak, ¯Ak) and Vol(Ak) represent the inter-cluster
and intra-cluster connection weights, respectively. This cri-
terion encourages clusters with strong intra-cluster connec-
tivity and weak inter-cluster links. Subsequently, we apply
the pointmap model to these clusters in parallel for dense
3D prediction.
Within each cluster, images are ordered
according to their matching connectivity and processed in
mini-batches. After each batch, pixel indices provide one-
to-one 3D correspondences between the dense pointmap
and the SfM reconstruction. We then estimate a similar-
ity transformation matrix T∗to align the dense prediction
with the SfM coordinate frame:
T∗= arg
min
T∈Sim(3)
 T ˜X −˜Y

2
F ,
(4)
where ˜X, ˜Y ∈R4×m are the homogeneous representations
of the dense and SfM 3D points. Finally, all aligned cluster
results are sampled and merged into a unified auxiliary point
cloud for Gaussian initialization.
4.2.2. Sparsity Compensation Densification
When the initialized regions are excessively sparse, they
tend to form large, coarse Gaussian primitives. If such re-
gions are observed by only a few effective views, the result-
ing representations are difficult to densify properly. To ad-
dress this issue, we introduce a targeted optimization strat-
egy designed to refine and densify these under-represented
areas. We identify Gaussians Gsplit for splitting based on
a dual criterion combining large contribution area and low
local density:
Gsplit = { Gi | Si > Sth ∧Vi < Vth } .
(5)
Here, Si = P
x∈P δ
 imax(x) = i ∧imid(x) = i

de-
notes the accumulated area where Gaussian Gi simultane-
ously yields the maximum contribution weight and the me-
dian depth along the ray. Vi measures the local voxel den-
sity, defined as the number of Gaussians whose centers fall
within the voxel VGi containing Gi. This criterion favors
splitting Gaussians that dominate large regions yet lie in
sparse neighborhoods, thereby improving geometric cover-
age without over-densification.
4.3. Progressive Hybrid Geometric Refinement
Robust geometric optimization is key to high-quality sur-
face reconstruction. Traditional methods rely on monocu-
4

<!-- page 5 -->
lar depth supervision or multi-view photometric constraints.
However, the former lacks inter-view geometric consis-
tency, while the latter, being single-scale and computation-
ally demanding, is limited in structurally diverse scenes. To
address this, we propose a two-stage progressive hybrid ge-
ometric refinement strategy.
4.3.1. Single-View Optimization
Following [34], we employ a pretrained depth estimation
model [44] to obtain a monocular depth prior. The esti-
mated inverse depth is first aligned with the sparse SfM
depth, and the L1 loss between the rendered and estimated
inverse depths is formulated as Ld to guide depth supervi-
sion. In addition, we preserve the depth–normal consistency
loss Ln from 2DGS [20] to further enhance geometric fi-
delity. In practice, we also observe that large-scale Gaus-
sians often introduce noticeable visual artifacts and blur lo-
cal details, and their extensive coverage on the image plane
leads to heavy GPU memory consumption during training
and slows down the optimization process. To mitigate these
issues, we introduce a scale regularization term defined as:
Ls =
1
|M|
X
i∈M
max(max(si) −τs, ϵ),
(6)
where M denotes the set of visible Gaussians, and τs is
a threshold that limits the maximum allowable Gaussian
scale. The overall geometry optimization loss at this stage
is formulated as:
L(1)
geo = λdLd + λnLn + λsLs.
(7)
4.3.2. Hybird Multi-View Refinement
After sufficient training iterations, the geometric optimiza-
tion transitions into the second stage. For each training im-
age, we predefine its neighboring views based on the prior
information provided by SfM. For each training image, we
refine its rendered depth Dr using the PatchMatch algo-
rithm between the image and its neighboring views. To ef-
fectively handle objects of different scales, we iteratively
apply multi-scale patches for depth refinement. The refined
depth is further filtered based on reprojection errors with
adjacent views, yielding the final reliable depth Df.
Although this filtering effectively mitigates the impact of
incorrect depths, it inevitably removes some valid regions,
resulting in large holes in the refined depth maps. To alle-
viate this issue, we reintroduce the monocular depth prior
Dm, leveraging its relatively accurate relative depth esti-
mation as complementary guidance to recover valid depth
regions that were mistakenly filtered out. Specifically, the
monocular depth map is divided into local patches, and each
patch is locally aligned with its corresponding filtered depth
via least-squares estimation:
s∗, t∗= arg min
s, t
X
p∈Df
∥Df(p) −(s · Dm(p) + t)∥2. (8)
Figure 3. Visualization of hybrid multi-view refinement. (a)
Strict geometric consistency yields reliable PM-refined depth. (b)
and (c) show the restored refined depths, highlighting the effec-
tiveness of patch-based alignment for local restoration.
When the alignment error between the aligned depth and
the filtered depth falls below a predefined threshold, the fil-
tered depth is preserved. The restored depth Dmv is then
used to guide further refinement of the rendered depth Dr.
Specifically, the depth refinement loss is defined as:
Lmv =
1
|Dmv|
X
p∈Dmv
|Dr(p) −Dmv(p)| .
(9)
Unlike direct photometric optimization, we adopt depth-
based supervision for enforcing multi-view consistency.
This design provides two benefits. As the quality of the ren-
dered depth improves during training, the refined depth im-
proves accordingly, and computation is reduced by updating
the refined depth maps only at fixed intervals. These com-
bined mechanisms ensure that the final depth maps achieve
both high geometric accuracy and structural completeness.
The overall geometry optimization loss at this stage is for-
mulated as:
L(2)
geo = λmvLmv + λnLn.
(10)
4.4. Depth-Guided Appearance Modeling
Previous works [30, 55] show that accurate appearance
modeling is crucial for realistic reconstruction, since
geometry-only methods often struggle under complex
imaging conditions. Existing appearance methods typically
do not leverage geometric information, while our method
provides high-quality rendered depth that offer a reliable
structural prior for appearance learning. Based on this, we
design a depth-guided appearance modeling module to en-
sure appearance estimation under precise geometric con-
straints, enabling true geometry–appearance decoupling.
Specifically, we employ a Tri-Mip [19] structure to store
scale-adaptive, multi-resolution 3D features of the scene,
which maintain cross-view consistency in space.
Given
the rendered depth map Dr, we query the Tri-Mip feature
planes using the 3D coordinates of each pixel’s projection,
resulting in structure-aligned representations fTri(x) . These
features provide a stable geometric foundation for appear-
ance estimation, enabling it to focus on color and lighting
5

<!-- page 6 -->
Table 1. Comparison with SOTA reconstruction methods on the GauU-Scene [48] dataset. P and R indicate the Precision and Recall
with respect to the ground-truth point cloud. Results highlighted in red , orange , and yellow correspond to the best, second-best, and
third-best performances, respectively. “NaN” means no results due to NaN error. “FAIL” means fail to extract meaningful mesh.
Methods
Russian Building
Residence
Modern Building
PSNR↑SSIM↑LPIPS↓
P↑
R↑
F1↑
PSNR↑SSIM↑LPIPS↓
P↑
R↑
F1↑
PSNR↑SSIM↑LPIPS↓
P↑
R↑
F1↑
NeuS
13.65 0.202
0.694
FAIL
FAIL
FAIL
15.16 0.244
0.674
FAIL
FAIL
FAIL
14.58 0.236
0.694
FAIL
FAIL
FAIL
Neuralangelo 12.48 0.328
0.698
NaN
NaN
NaN
NaN
NaN
NaN
NaN
NaN
NaN
NaN
NaN
NaN
NaN
NaN
NaN
SuGaR
23.62 0.738
0.332
0.480 0.369 0.417
21.95 0.612
0.452
0.579 0.287 0.384
24.92 0.700
0.381
0.650 0.220 0.329
GOF
21.30 0.713
0.322
0.294 0.394 0.330
20.68 0.652
0.391
0.404 0.418 0.411
25.01 0.749
0.286
0.411 0.357 0.382
2DGS
23.77 0.788
0.189
0.544 0.519 0.531
22.24 0.703
0.306
0.526 0.406 0.458
25.77 0.776
0.202
0.588 0.413 0.485
CityGS
24.37 0.808
0.163
0.459 0.443 0.451
23.59 0.763
0.204
0.524 0.391 0.448
26.29 0.796
0.160
0.582 0.381 0.461
CityGS-X
24.62 0.804
0.155
0.570 0.497 0.531
23.74 0.749
0.238
0.564 0.402 0.470
26.20 0.783
0.171
0.598 0.362 0.451
CityGSV2
24.12 0.784
0.196
0.560 0.530 0.544
23.57 0.742
0.243
0.524 0.421 0.467
25.84 0.770
0.207
0.643 0.398 0.492
Ours
24.94 0.814
0.138
0.610 0.562 0.585
24.51 0.769
0.185
0.566 0.439 0.494
27.07 0.797
0.152
0.662 0.433 0.524
Figure 4.
Qualitative comparison on the MatrixCity [24]
dataset. Image rendering and mesh reconstruction are compared
between our method and CityGSV2 [34].
variations that are independent of geometry. Additionally,
each training image Ii is assigned a learnable appearance
embedding li ∈Rd to capture global illumination and ex-
posure conditions. The queried Tri-Mip feature fTri(x) and
the embedding li are concatenated and passed through a
lightweight MLP tone mapper Fθ:
M(x) = Fθ
 fTri(x); li

,
(11)
where M(x) denotes the tone-mapped appearance at pixel
x. The output is used to modulate the rendered image Ir
i ,
resulting in the final reconstruction It
i with consistent tone
and illumination. The appearance loss is defined as:
Lapp = λL1(It
i, Ii) + (1 −λ)LD−SSIM(Ir
i , Ii).
(12)
Table 2. Comparison on the MatrixCity [24] dataset. “–” indi-
cates the metric was not reported in the original paper.
Methods
MatrixCity-Aerial
PSNR↑
SSIM↑
LPIPS↓
P↑
R↑
F1↑
SuGaR
22.41
0.633
0.493
0.182
0.157
0.169
GOF
17.42
0.374
0.588
FAIL
FAIL
FAIL
2DGS
21.35
0.632
0.562
0.207
0.390
0.270
CityGS
27.46
0.865
0.204
0.362
0.637
0.462
CityGS-X
27.58
−
−
0.444
0.840
0.581
CityGSV2
27.23
0.857
0.169
0.441
0.752
0.556
Ours
27.52
0.854
0.167
0.572
0.828
0.677
Methods
MatrixCity-Street
PSNR↑
SSIM↑
LPIPS↓
P↑
R↑
F1↑
SuGaR
19.82
0.662
0.478
0.053
0.111
0.071
GOF
20.32
0.703
0.440
0.219
0.473
0.300
2DGS
21.50
0.723
0.477
0.334
0.659
0.443
CityGS
22.98
0.808
0.301
0.283
0.689
0.401
CityGS-X
OOM
OOM
OOM
OOM
OOM
OOM
CityGSV2
22.24
0.788
0.347
0.376
0.759
0.503
Ours
23.16
0.798
0.294
0.480
0.828
0.607
4.5. Training Loss
During training, the geometric and appearance optimiza-
tion processes are jointly performed, and the geometric loss
varies with the training stage. The overall loss function is
defined as follows:
Ltotal = Lgeo + Lapp.
(13)
5. Experiments
5.1. Experimental Setup
We conduct comprehensive experiments on the GauU-
Scene [48] dataset and the synthetic MatrixCity [24]
dataset. All experiments are performed on a workstation
equipped with four RTX 3090 GPUs. Both datasets provide
high-precision ground-truth point clouds, making them reli-
6

<!-- page 7 -->
Figure 5. Qualitative results on the GauU-Scene [48] dataset. We present the image and depth rendering results of our method compared
with state-of-the-art methods.
able benchmarks for evaluating geometric quality in large-
scale scene reconstruction. Following the settings in [34],
we evaluate our method on both the aerial-view and street-
view versions. The aerial-view images are downsampled to
a 1600-pixel long side, while the street-view images retain
their original 1000 × 1000 resolution. For mesh extraction,
we adopt the 2DGS methodology, integrating median depth
with TSDF fusion. Detailed training and evaluation config-
urations are provided in the supplementary materials.
5.2. Baselines
We compare our method against a broad set of state-of-
the-art surface reconstruction methods.
For NeRF-based
methods, we include NeuS [43] and Neuralangelo [25].
For GS–based methods, we compare against SuGaR [18],
2DGS [20], GOF [52], CityGS [33], CityGS-X [15], and
CityGSV2 [34]. We assess reconstruction quality from both
visual and geometric perspectives. For large-scale scene re-
construction, we select CityGSV2 as a representative base-
line, as it is among the best-performing open-source meth-
ods in terms of geometric reconstruction quality.
5.3. Main Results
Quantitative Results.
As shown in Tab. 1, we compare
our proposed method with several SOTA methods on the
GauU-Scene [48] dataset, which contains representative
real-world urban scenes. The results show that our method
achieves superior geometric reconstruction and rendering
performance, ranking first on most metrics. Compared to
CityGSV2 [34], our method improves PSNR by 0.88 dB on
average and boosts the F1-score by 0.033, consistently out-
performing it across all metrics. Tab. 2 presents the compar-
ison results on the synthetic MatrixCity [24] dataset. Our
method again achieves the highest F1-score, with an aver-
age improvement of 0.11 over CityGSV2, indicating reli-
able and accurate geometric reconstruction across different
scene types and data settings. In addition, even under the
inherently stable illumination of synthetic data, our method
maintains competitive rendering performance.
Overall,
these results validate the effectiveness of our method for
robust, geometrically accurate, and high-fidelity large-scale
scene reconstruction.
7

<!-- page 8 -->
Qualitative Results.
To further validate the effectiveness
of our method, we provide extensive visual comparisons.
Fig. 4 shows image renderings and mesh reconstructions
from synthetic MatrixCity [24], where our method produces
more accurate and more complete geometric reconstruc-
tions than CityGSV2 [34]. Fig. 5 presents qualitative com-
parisons of rendered images and corresponding depth maps
from GauU-Scene [48]. The first row depicts a scene under
challenging lighting conditions: other methods suffer from
floating artifacts caused by geometric errors, whereas our
method yields more accurate illumination and more consis-
tent geometric structures. The second and third rows con-
tain scenes with rich texture details, where other methods
show blurred or distorted structures in both the rendered im-
ages and depth maps. In contrast, our method preserves fine
geometry and delivers visually coherent rendering results.
5.4. Ablation Studies
We conduct thorough ablation studies on the Russian Build-
ing scene to quantify the effectiveness of our proposed com-
ponents, with results presented in Tab. 3.
Scalable Parallel Strategy.
The first two rows show that
the parallelization strategy substantially improves overall
performance and running efficiency, which is critical for the
high-performance execution of our framework.
Structured Dense Enhancement.
We conduct separate
ablation studies on the pointmap-assisted initialization (w/o
Ini.) and sparsity compensation (w/o Spa.) within the struc-
tured dense enhancement module. Results indicate that re-
moving the former leads to a relatively significant perfor-
mance degradation, whereas the latter only exhibits a slight
drop. The performance change in both cases directly corre-
lates with the reduced number of final reconstructed Gaus-
sians. Fig. 6(a) visually corroborates the efficacy of sparsity
compensation, showing clear improvement in sparsely ob-
served regions of the scene.
Progressive Hybrid Geometric Refinement.
We ablate
the progressive geometric refinement module by removing
the whole module (w/o Geo.), its sub-component, the hy-
brid multi-view refinement (w/o Mul.), and the alignment
& restoration operation (w/o Ali.)
within it.
Removing
the whole module yielded the worst F1-score. Furthermore,
removing the multi-view refinement or the alignment also
significantly impacts geometric metrics. This confirms the
critical role of every component. Notably, Fig. 6(b) further
illustrates that geometric quality also impacts appearance.
Depth-Guided Appearance Modeling.
Finally, we eval-
uate the removal of the entire depth-guided appearance
modeling module (w/o App.) and the Tri-Mip component
(w/o Tri.) within it. Removing appearance modeling caused
Table 3. Ablation on model components. The experiments are
conducted on Russian Building scene of GauU-Scene [48] dataset.
We adopt a customized 2DGS [20] as our base method.
Model
Rendering
Geometry
GS Statistics
PSNR↑
SSIM↑
P↑
R↑
F1↑
#G(M) T(min)
Base
23.88
0.774
0.539
0.509
0.523
4.55
134
Base + Para.
24.35
0.798
0.550
0.515
0.532
7.30
68
w/o Ini.
24.84
0.808
0.598
0.557
0.577
7.51
98
w/o Spa.
24.88
0.811
0.608
0.560
0.583
8.02
104
w/o Geo.
24.83
0.807
0.571
0.557
0.564
8.99
89
w/o Mul.
24.87
0.810
0.586
0.556
0.571
8.17
87
w/o Ali.
24.86
0.811
0.603
0.559
0.580
8.18
101
w/o App.
24.46
0.807
0.581
0.543
0.562
8.29
99
w/o Tri.
23.96
0.807
0.590
0.549
0.569
8.08
95
Full Model
24.94
0.814
0.610
0.562
0.585
8.20
106
Figure 6. Visualization results of ablation study. The top row
shows the results without the corresponding modules, while the
bottom row shows the results with the modules. Further visualiza-
tions are available in the supplementary materials.
a substantial performance drop across all metrics, confirm-
ing the importance of decoupling geometry from appear-
ance in scenes with inconsistent visual conditions. Further
removing the Tri-Mip feature led to an additional decline,
with PSNR dropping even further, highlighting the need for
geometric awareness in appearance modeling. In contrast,
Fig. 6(c) shows that with our appearance modeling, the ren-
dered image become more realistic and natural.
6. Conclusion
In this paper, we present MetroGS, a novel Gaussian Splat-
ting framework designed for large-scale scene reconstruc-
tion. Leveraging the foundation of distributed 2DGS, we
integrate a structured dense enhancement scheme, a pro-
gressive hybrid geometric refinement strategy, and a depth-
guided appearance modeling module. Together, these com-
ponents enable geometrically accurate and training-efficient
reconstruction. Extensive experiments on multiple large-
scale scene datasets validate the efficacy of our method,
demonstrating superior reconstruction performance.
8

<!-- page 9 -->
Acknowledgements
This work was in part supported by the Chinese Academy
of Sciences, the Strategic Priority Research Program of the
Chinese Academy of Sciences (XDA0450402), the Beijing
Natural Science Foundation (L259015), the National Natu-
ral Science Foundation of China (62172392), and the Inno-
vation Research Program of ICT CAS (E261070).
References
[1] Michael Bleyer, Christoph Rhemann, and Carsten Rother.
Patchmatch stereo-stereo matching with slanted support win-
dows. In Bmvc, pages 1–11, 2011. 2
[2] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora,
Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Gi-
ancarlo Baldan, and Oscar Beijbom.
nuscenes: A multi-
modal dataset for autonomous driving. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 11621–11631, 2020. 1
[3] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. pixelsplat: 3d gaussian splats from image
pairs for scalable generalizable 3d reconstruction. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 19457–19467, 2024. 2
[4] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and
Hao Su. Tensorf: Tensorial radiance fields. In European con-
ference on computer vision, pages 333–350. Springer, 2022.
2
[5] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie,
Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and
Guofeng Zhang. Pgsr: Planar-based gaussian splatting for ef-
ficient and high-fidelity surface reconstruction. IEEE Trans-
actions on Visualization and Computer Graphics, 2024. 2,
3
[6] Junyi Chen, Weicai Ye, Yifan Wang, Danpeng Chen, Di
Huang, Wanli Ouyang, Guofeng Zhang, Yu Qiao, and
Tong He.
Gigags:
Scaling up planar-based 3d gaus-
sians for large scene surface reconstruction. arXiv preprint
arXiv:2409.06685, 2024. 2, 3
[7] Kehua Chen, Zhenlong Yuan, Tianlu Mao, and Zhaoqi
Wang. Dual-level precision edges guided multi-view stereo
with accurate planarization. In Proceedings of the AAAI Con-
ference on Artificial Intelligence, pages 2105–2113, 2025. 1
[8] Kehua Chen, Zhenlong Yuan, Haihong Xiao, Tianlu Mao,
and Zhaoqi Wang.
Learning multi-view stereo with
geometry-aware prior. IEEE Transactions on Circuits and
Systems for Video Technology, 2025. 1
[9] Peng Chen, Xiaobao Wei, Qingpo Wuwu, Xinyi Wang,
Xingyu Xiao, and Ming Lu. Mixedgaussianavatar: Realisti-
cally and geometrically accurate head avatar via mixed 2d-3d
gaussian splatting. arXiv preprint arXiv:2412.04955, 2024.
2
[10] Shihan Chen, Zhaojin Li, Zeyu Chen, Qingsong Yan,
Gaoyang Shen, and Ran Duan. 3d gaussian splatting for fine-
detailed surface reconstruction in large-scale scene. arXiv
preprint arXiv:2506.17636, 2025. 2
[11] Alexandre Delplanque, Julie Linchant, Xavier Vincke,
Richard Lamprey, J´erˆome Th´eau, C´edric Vermeulen, Samuel
Foucher, Amara Ouattara, Roger Kouadio, and Philippe
Lejeune. Will artificial intelligence revolutionize aerial sur-
veys? a first large-scale semi-automated survey of african
wildlife using oblique imagery and deep learning. Ecologi-
cal Informatics, 82:102679, 2024. 1
[12] Lue Fan, Yuxue Yang, Minxing Li, Hongsheng Li, and
Zhaoxiang Zhang. Trim 3d gaussian splatting for accurate
geometry representation. arXiv preprint arXiv:2406.07499,
2024. 2
[13] Guangchi Fang and Bing Wang.
Mini-splatting: Repre-
senting scenes with a constrained number of gaussians. In
European Conference on Computer Vision, pages 165–181.
Springer, 2024. 2
[14] Yuanyuan Gao, Yalun Dai, Hao Li, Weicai Ye, Junyi Chen,
Danpeng Chen, Dingwen Zhang, Tong He, Guofeng Zhang,
and Junwei Han. Cosurfgs: Collaborative 3d surface gaus-
sian splatting with distributed learning for large scene recon-
struction. arXiv preprint arXiv:2412.17612, 2024. 3
[15] Yuanyuan Gao, Hao Li, Jiaqi Chen, Zhengyu Zou, Zhihang
Zhong, Dingwen Zhang, Xiao Sun, and Junwei Han. Citygs-
x: A scalable architecture for efficient and geometrically
accurate large-scale scene reconstruction.
arXiv preprint
arXiv:2503.23044, 2025. 3, 7
[16] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we
ready for autonomous driving? the kitti vision benchmark
suite. In 2012 IEEE conference on computer vision and pat-
tern recognition, pages 3354–3361. IEEE, 2012. 1
[17] Jiaming Gu, Minchao Jiang, Hongsheng Li, Xiaoyuan Lu,
Guangming Zhu, Syed Afaq Ali Shah, Liang Zhang, and
Mohammed Bennamoun. Ue4-nerf: Neural radiance field for
real-time rendering of large-scale scene. Advances in Neural
Information Processing Systems, 36:59124–59136, 2023. 1
[18] Antoine Gu´edon and Vincent Lepetit.
Sugar:
Surface-
aligned gaussian splatting for efficient 3d mesh reconstruc-
tion and high-quality mesh rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5354–5363, 2024. 3, 7
[19] Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao,
Xiao Liu, and Yuewen Ma.
Tri-miprf: Tri-mip represen-
tation for efficient anti-aliasing neural radiance fields.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 19774–19783, 2023. 2, 5
[20] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. In ACM SIGGRAPH 2024 conference
papers, pages 1–11, 2024. 2, 3, 5, 7, 8
[21] Han Huang, Yulun Wu, Chao Deng, Ge Gao, Ming Gu,
and Yu-Shen Liu. Fatesgs: Fast and accurate sparse-view
surface reconstruction using gaussian splatting with depth-
feature consistency. In Proceedings of the AAAI Conference
on Artificial Intelligence, pages 3644–3652, 2025. 3
[22] Changjian Jiang, Kerui Ren, Linning Xu, Jiong Chen, Jiang-
miao Pang, Yu Zhang, Bo Dai, and Mulin Yu. Halogs: Loose
coupling of compact geometry and gaussian splats for 3d
scenes. arXiv preprint arXiv:2505.20267, 2025. 3
9

<!-- page 10 -->
[23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 3
[24] Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhen-
zhi Wang, Dahua Lin, and Bo Dai. Matrixcity: A large-scale
city dataset for city-scale neural rendering and beyond. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 3205–3215, 2023. 6, 7, 8, 1
[25] Zhaoshuo Li, Thomas M¨uller, Alex Evans, Russell H Tay-
lor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin.
Neuralangelo: High-fidelity neural surface reconstruction. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 8456–8465, 2023. 7
[26] Zehao Li, Wenwei Han, Yujun Cai, Hao Jiang, Baolong Bi,
Shuqin Gao, Honglong Zhao, and Zhaoqi Wang. Gradiseg:
Gradient-guided gaussian segmentation with enhanced 3d
boundary precision. arXiv preprint arXiv:2412.00392, 2024.
1
[27] Zehao Li, Hao Jiang, Yujun Cai, Jianing Chen, Baolong
Bi, Shuqin Gao, Honglong Zhao, Yiwei Wang, Tianlu
Mao, and Zhaoqi Wang.
Stdr:
Spatio-temporal decou-
pling for real-time dynamic scene rendering. arXiv preprint
arXiv:2505.22400, 2025. 2
[28] Zhuoxiao Li, Shanliang Yao, Taoyu Wu, Yong Yue, Wu-
fan Zhao, Rongjun Qin, ´Angel F Garc´ıa-Fern´andez, Andrew
Levers, Jason Ralph, and Xiaohui Zhu. Ulsr-gs: Urban large-
scale surface reconstruction gaussian splatting with multi-
view geometric consistency. ISPRS Journal of Photogram-
metry and Remote Sensing, 230:861–880, 2025. 2
[29] Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen,
Yen-Yu Lin, and Yu-Lun Liu. Longsplat: Robust unposed
3d gaussian splatting for casual long videos. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 27412–27422, 2025. 2
[30] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong
Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, You-
liang Yan, et al. Vastgaussian: Vast 3d gaussians for large
scene reconstruction. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
5166–5175, 2024. 2, 3, 5
[31] Liqiang Lin, Yilin Liu, Yue Hu, Xingguang Yan, Ke Xie, and
Hui Huang. Capturing, reconstructing, and simulating: the
urbanscene3d dataset. In European Conference on Computer
Vision, pages 93–109. Springer, 2022. 3
[32] Chuandong Liu, Huijiao Wang, Lei Yu, and Gui-Song Xia.
Holistic large-scale scene reconstruction via mixed gaussian
splatting. arXiv preprint arXiv:2505.23280, 2025. 2, 3
[33] Yang Liu, Chuanchen Luo, Lue Fan, Naiyan Wang, Jun-
ran Peng, and Zhaoxiang Zhang. Citygaussian: Real-time
high-quality large-scale scene rendering with gaussians. In
European Conference on Computer Vision, pages 265–282.
Springer, 2024. 7
[34] Yang Liu, Chuanchen Luo, Zhongkai Mao, Junran Peng, and
Zhaoxiang Zhang. Citygaussianv2: Efficient and geometri-
cally accurate reconstruction for large-scale scenes. arXiv
preprint arXiv:2411.00771, 2024. 1, 2, 3, 5, 6, 7, 8
[35] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl,
Markus Steinberger, Francisco Vicente Carrasco, and Fer-
nando De La Torre.
Taming 3dgs: High-quality radiance
fields with limited resources. In SIGGRAPH Asia 2024 Con-
ference Papers, pages 1–11, 2024. 3
[36] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi,
Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duck-
worth. Nerf in the wild: Neural radiance fields for uncon-
strained photo collections. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pages 7210–7219, 2021. 2
[37] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[38] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a mul-
tiresolution hash encoding. ACM transactions on graphics
(TOG), 41(4):1–15, 2022. 2
[39] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. arXiv
preprint arXiv:2403.17898, 2024. 3, 2
[40] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 3
[41] Haithem Turki,
Deva Ramanan,
and Mahadev Satya-
narayanan.
Mega-nerf:
Scalable construction of large-
scale nerfs for virtual fly-throughs.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 12922–12931, 2022. 3
[42] Jiepeng Wang, Yuan Liu, Peng Wang, Cheng Lin, Junhui
Hou, Xin Li, Taku Komura, and Wenping Wang.
Gaus-
surf: Geometry-guided 3d gaussian splatting for surface re-
construction. arXiv preprint arXiv:2411.19454, 2024. 3
[43] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku
Komura, and Wenping Wang. Neus: Learning neural implicit
surfaces by volume rendering for multi-view reconstruction.
arXiv preprint arXiv:2106.10689, 2021. 7
[44] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng
Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong
Yang. Moge-2: Accurate monocular geometry with metric
scale and sharp details. arXiv preprint arXiv:2507.02546,
2025. 2, 5
[45] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang,
Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua
Shen, and Tong He. π3: Scalable permutation-equivariant
visual geometry learning. arXiv preprint arXiv:2507.13347,
2025. 2, 3
[46] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 20310–20320, 2024. 1
[47] Jiang Wu, Rui Li, Yu Zhu, Rong Guo, Jinqiu Sun, and Yan-
ning Zhang.
Sparse2dgs: Geometry-prioritized gaussian
10

<!-- page 11 -->
splatting for surface reconstruction from sparse views. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 11307–11316, 2025. 3
[48] Butian Xiong, Nanjun Zheng, Junhua Liu, and Zhen Li.
Gauu-scene v2: Assessing the reliability of image-based
metrics with expansive lidar image dataset using 3dgs and
nerf. arXiv preprint arXiv:2404.04880, 2024. 1, 6, 7, 8, 2
[49] Qingshan Xu and Wenbing Tao.
Multi-scale geometric
consistency guided multi-view stereo.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5483–5492, 2019. 2
[50] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong
Dou. Absgs: Recovering fine details in 3d gaussian splat-
ting. In Proceedings of the 32nd ACM International Confer-
ence on Multimedia, pages 1053–1061, 2024. 2
[51] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 19447–19456,
2024. 2
[52] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. ACM Transactions on Graphics (ToG),
43(6):1–13, 2024. 3, 7
[53] Zhensheng Yuan, Haozhi Huang, Zhen Xiong, Di Wang,
and Guanghua Yang.
Robust and efficient 3d gaussian
splatting for urban scene reconstruction.
arXiv preprint
arXiv:2507.23006, 2025. 2, 3
[54] Andy Zeng, Shuran Song, Matthias Nießner, Matthew
Fisher, Jianxiong Xiao, and Thomas Funkhouser. 3dmatch:
Learning local geometric descriptors from rgb-d reconstruc-
tions. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 1802–1811, 2017. 1
[55] Youjia Zhang, Anpei Chen, Yumin Wan, Zikai Song, Jun-
qing Yu, Yawei Luo, and Wei Yang.
Ref-gs: Directional
factorization for 2d gaussian splatting. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
26483–26492, 2025. 2, 3, 5
[56] Zheng Zhang, Wenbo Hu, Yixing Lao, Tong He, and Heng-
shuang Zhao.
Pixel-gs: Density control with pixel-aware
gradient for 3d gaussian splatting. In European Conference
on Computer Vision, pages 326–342. Springer, 2024. 2
[57] Hexu Zhao, Haoyang Weng, Daohan Lu, Ang Li, Jinyang Li,
Aurojit Panda, and Saining Xie. On scaling up 3d gaussian
splatting training. In European Conference on Computer Vi-
sion, pages 14–36. Springer, 2024. 2, 3
[58] Hexu Zhao, Xiwen Min, Xiaoteng Liu, Moonjun Gong, Yim-
ing Li, Ang Li, Saining Xie, Jinyang Li, and Aurojit Panda.
Clm: Removing the gpu memory barrier for 3d gaussian
splatting. arXiv preprint arXiv:2511.04951, 2025. 3
11

<!-- page 12 -->
MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate
High-Fidelity Large-Scale Scenes
Supplementary Material
A. Implementation Details
For the GauU-Scene [48] dataset, we conducted parallel
training with a batch size of 4, targeting a total of 60,000
iterations. Subsequently, we train both the single-view and
multi-view geometric refinement stages of Lgeo for 30,000
iterations. During this process, λd decreases from 0.5 to
0.005 as training progresses, while λn is set to 0.0125, λs
to 0.1, and λmv to 2.5. For Lapp, the weight λ is set to
0.8. Densification terminates after the 15,000th iteration,
with sparsity compensation parameters set to Sth = 20 and
Vth = 10. The voxel size is set to 0.1 or 0.01 depending
on the scale of the scene. For evaluation, only the view
embeddings from the training set are available. Since the
image filenames encode temporal information, we first use
it to identify the two training views that are temporally clos-
est to each test view. We select the candidate with the most
similar camera pose to the test view. This nearest-neighbor
assignment provides the interpolated view embedding for
the test view.
For the MatrixCity [24] dataset, the Aerial and Street
scenes were trained for 150,000 and 180,000 iterations, re-
spectively. For Lgeo, single-view optimization is performed
until the 50,000th iteration, followed by the switch to multi-
view refinement.
Densification is also terminated at the
50,000th iteration.
All other training configurations fol-
low those used for the GauU-Scene dataset. For evaluation,
test image filenames lack temporal information, so each test
view selects its most relevant training view solely based on
camera-pose similarity. The corresponding view embed-
ding is then used for image rendering.
For geometric quality evaluation, we follow the parame-
ter settings used in CityGSV2 [34]. Specifically, we render
RGB images and depth maps from the training viewpoints
and fuse them into a projected truncated signed distance
functio (TSDF) volume [54] to extract surface meshes and
point clouds. GauU-Scene uses a voxel size of 0.01, an SDF
truncation of 0.04, and a depth truncation of 2.0. In Matrix-
City, the Aerial split uses 0.01 / 0.04 / 5.0 for voxel size,
SDF truncation, and depth truncation, respectively, whereas
the Street split adopts 1 / 4 / 500.
B. Hyperparameters of Other Methods
For the visualization results of 2DGS, CityGS, and
CityGSV2, we train the models using the default param-
eter settings provided in the CityGSV2 codebase, and for
CityGSV2, we use the provided checkpoints. For the com-
Table 4.
Efficiency performance comparison on the GauU-
Scene [48] dataset. Entries marked with an asterisk (*) represent
the intermediate results obtained after 30,000 training iterations.
Scene
Method
PSNR ↑
F1 ↑
#G(M)
T(min)
Russian
V2-coarse*
23.46
0.509
7.98
110
Ours*
24.60
0.559
8.20
50
CityGSV2
24.12
0.542
7.77
363
Ours
24.94
0.585
8.20
106
Residence
V2-coarse*
22.09
0.437
9.29
103
Ours*
23.96
0.470
11.33
78
CityGSV2
23.55
0.466
8.08
311
Ours
24.51
0.494
11.33
156
Morden
V2-coarse*
25.08
0.479
7.61
98
Ours*
26.68
0.508
9.27
70
CityGSV2
25.79
0.492
7.89
332
Ours
27.07
0.524
9.27
149
Table 5. Efficiency performance comparison on MatrixCity-
Aerial [24]. In CityGS-X, which uses an anchor-based Gaussian
representation, “×10” denotes the Gaussians derived per anchor.
Scene
Method
PSNR ↑
F1 ↑
#G(M)
T(min)
MC-Aerial
CityGS-X
27.53
0.582
2.48×10
716
Ours
27.52
0.677
17.09
415
Figure 7. Supplementary Visualization of ablation study re-
sults. The top row shows results without the modules, and the
bottom row shows results with them. Our components yield a sig-
nificant improvement in depth quality, effectively addressing chal-
lenges across diverse and complex scenes.
parison with CityGS-X, we utilized its provided Mill19
configuration to train the GauU-Scene dataset. Crucially,
we disabled the progressive LOD (Level of Detail) train-
ing within this configuration to ensure better preservation
1

<!-- page 13 -->
Figure 8. Qualitative comparison of meshes on the GauU-Scene [48] dataset. Our method achieves higher-quality results.
Figure 9. Mesh visualization comparison on MatrixCity-Aerial [48]. Our method provides better results than the baselines.
of scene details. For the MatrixCity dataset, we directly ap-
plied the corresponding official configuration provided by
CityGS-X for training.
C. Additional Results
C.1. Training Efficiency Analysis
Using a system with four RTX 3090 GPUs, we con-
ducted a training efficiency comparison between CityGSV2
and CityGS-X on the GauU-Scene and MatrixCity-Aerial
datasets, respectively. As shown in Tab. 4, our method con-
sistently outperforms CityGSV2 in both rendering quality
and geometric fidelity, while also demonstrating a signifi-
cant improvement in training efficiency. Notably, even the
intermediate results of our model at 30k iterations already
surpass the final performance of CityGSV2, while requiring
less than 25% of its training time. Across the GauU-Scene
dataset, our final model achieves an average 2.55× training
speedup relative to CityGSV2. Tab. 5 presents a comparison
of training efficiency between CityGS-X and our method on
the MatrixCity-Aerial dataset. Our approach achieves su-
perior geometric fidelity (F1: 0.677 vs. 0.582) with a 1.7×
reduction in training time, while maintaining comparable
PSNR performance. Overall, these results highlight the re-
markable speed and efficiency of our method. It is worth
noting that CityGSV2 and CityGS-X adopt model-size re-
duction strategies such as trimming [12] and anchor-based
Gaussian compression [39]. Enhancing model-size com-
pactness therefore remains a promising direction for further
improving the efficiency of our method.
C.2. Additional Qualitative Comparison
Fig. 7 presents further visualization results for the ablation
study. Our adopted pointmap assisted initialization effec-
tively supplements sparse point cloud regions, thereby lay-
ing a solid geometric foundation for subsequent reconstruc-
tion. Progressive hybrid geometric refinement and depth-
guided appearance modeling then collaboratively ensure the
final geometric quality exhibits high accuracy and com-
pleteness.
In addition, we include more comprehensive qualitative
comparisons with the baseline methods. Fig. 8 presents the
mesh reconstruction visualization comparison on the GauU-
Scene dataset. Given the relatively small size of the im-
2

<!-- page 14 -->
Table 6. Quantitative results on the Mill19 [41] dataset and UrbanScene3D [31] dataset. The best and second best results are high-
lighted. All missing results are denoted by a “–”.
Methods
Building
Rubble
Residence
Sci-Art
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
NeuS
18.01
0.463
0.611
20.46
0.480
0.618
17.85
0.503
0.533
18.62
0.633
0.472
Neuralangelo
17.89
0.582
0.322
20.18
0.625
0.314
18.03
0.644
0.263
19.10
0.769
0.231
SuGaR
17.76
0.507
0.455
20.69
0.577
0.453
18.74
0.603
0.406
18.60
0.698
0.349
PGSR
16.12
0.480
0.573
23.09
0.728
0.334
20.57
0.746
0.289
19.72
0.799
0.275
PGSR+VastGS
21.63
0.720
0.300
25.32
0.768
0.274
−
−
−
−
−
−
CityGS
21.55
0.778
0.246
25.77
0.813
0.228
22.00
0.813
0.211
21.39
0.837
0.230
CityGS-X
22.76
0.817
0.191
26.15
0.823
0.210
22.44
0.819
0.194
22.77
0.867
0.179
CityGSV2
19.07
0.650
0.397
23.75
0.720
0.322
21.15
0.769
0.234
20.66
0.810
0.266
Ours
23.06
0.787
0.173
27.48
0.826
0.147
23.38
0.824
0.166
25.96
0.872
0.152
age data, we conducted an equivalent comparison in terms
of training time: we trained our method for 30,000 iter-
ations and compared its results with those of CityGSV2-
coarse.
The reconstructed meshes from our method are
much cleaner, containing minimal spurious artifacts or
floating mesh fragments. Fig 9 further presents a compari-
son of our method’s results against CityGSV2 and CityGS-
X on the MatrixCity-Aerial dataset. The results indicate
that our approach achieves a better balance between geo-
metric accuracy and completeness.
C.3. Additional Dataset Evaluation
We have also conducted supplementary evaluations on the
Mill19 [41] and UrbanScene3D [31] datasets, which are
widely used for assessing rendering quality in the field of
large-scale scene reconstruction. Four scenes were selected:
Building, Rubble, Residence, and Sci-Art. The configura-
tion uses 100,000 training iterations, with 50,000 iterations
allocated to each of the two geometric optimization stages.
The densification process is terminated at the 30,000th iter-
ation. The weight λs set to 0.001. The remaining settings
follow those used for GauU-Scene, as detailed in Sec. A.
Quantitative results are presented in Tab. 6, where we
compare against other state-of-the-art surface reconstruc-
tion methods. Our method achieves state-of-the-art perfor-
mance among surface reconstruction approaches in terms
of PSNR and LPIPS, and ranks first in SSIM for most
scenes. In addition, Fig. 10 provides a qualitative compari-
son among our methodd and CityGS (Public Checkpoints),
showing that our approach performs better under challeng-
ing illumination conditions and renders fine-grained details
more faithfully. Overall, our method achieves superior vi-
sual quality and robustness.
D. Discussion
While our method successfully delivers efficient training,
accurate geometry, and high rendering quality for large-
scale scene reconstruction, it still presents the following
Figure 10.
Qualitative results on Mill-19 [41] and Urban-
scene3D [31] datasets. We compare against CityGS.
limitations: Firstly, due to hardware constraints, memory
consumption remains the primary bottleneck limiting the
training scale, which to some extent weakens the model’s
potential performance.
Therefore, it is necessary to in-
troduce techniques such as advanced pruning [35] and
cache management [58] to mitigate memory challenges.
Additionally, our method is based on 2DGS. Although
it achieves excellent geometric reconstruction, its upper
bound for rendering quality may still lag behind 3DGS.
To address this, future work could consider introducing a
new geometry representation similar to [22] for complete
decoupling of geometry and appearance to further realize
improved geometric accuracy and rendering performance.
3
