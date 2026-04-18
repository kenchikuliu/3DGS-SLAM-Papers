<!-- page 1 -->
CUS-GS: A Compact Unified Structured Gaussian Splatting Framework for
Multimodal Scene Representation
Yuhang Ming1, Chenxin Fang1, Xingyuan Yu2, Fan Zhang3, Weichen Dai1,
Wanzeng Kong1∗, Guofeng Zhang2
1School of Computer Science, Hangzhou Dianzi University, 2CAD & CG, Zhejiang University,
3School of Computer Science, University of Bristol
{yuhang.ming, chenxin.fang, daiweichen, kongwanzeng}@hdu.edu.cn,
{rickyyxy, zhangguofeng}@zju.edu.cn, fan.zhang@bristol.ac.uk
∗Corresponding author.
Figure 1. CUS-GS is the first framework to unify structured 3DGS with multimodal semantic modeling. The voxel-anchored structured
design produces a geometry-aware and multimodally aligned 3D feature field, while maintaining high efficiency—achieving competitive
performance with as few as 6M parameters, comparing to 35M of the closest rival.
Abstract
Recent advances in Gaussian Splatting based 3D scene
representation have shown two major trends: semantics-
oriented approaches that focus on high-level understand-
ing but lack explicit 3D geometry modeling, and structure-
oriented approaches that capture spatial structures yet pro-
vide limited semantic abstraction. To bridge this gap, we
present CUS-GS, a compact unified structured Gaussian
Splatting representation, which connects multimodal se-
mantic features with structured 3D geometry. Specifically,
we design a voxelized anchor structure that constructs a
spatial scaffold, while extracting multimodal semantic fea-
tures from a set of foundation models (e.g., CLIP, DINOv2,
SEEM). Moreover, we introduce a multimodal latent fea-
ture allocation mechanism to unify appearance, geometry,
and semantics across heterogeneous feature spaces, ensur-
ing a consistent representation across multiple foundation
models. Finally, we propose a feature-aware significance
evaluation strategy to dynamically guide anchor growing
and pruning, effectively removing redundant or invalid an-
chors while maintaining semantic integrity. Extensive ex-
periments show that CUS-GS achieves competitive perfor-
mance compared to state-of-the-art methods using as few as
6M parameters — an order of magnitude smaller than the
closest rival at 35M — highlighting the excellent trade off
between performance and model efficiency of the proposed
arXiv:2511.17904v1  [cs.CV]  22 Nov 2025

<!-- page 2 -->
framework.
1. Introduction
3D scene representation has long been a central topic in
computer vision and robotics. As a fundamental module, it
bridges perception [29, 33] with various downstream tasks
such as 3D reconstruction [22, 61], SLAM [30, 32], navi-
gation [50, 53], and manipulation [36, 57]. In recent years,
the community has witnessed remarkable progress in neu-
ral rendering–based scene representations (e.g., NeRF [31]
and 3DGS [15]), providing a powerful differentiable frame-
work. Among them, 3DGS is particularly promising for its
high-fidelity real-time rendering capabilities.
To accommodate different downstream tasks, recent
studies on 3DGS–based scene representations have diver-
sified into multiple directions. Among them, two promi-
nent trends are particularly relevant to this work. The first
emphasizes structural hierarchy and rendering efficiency,
forming the structure-oriented branch.
Representative
works, such as Scaffold-GS [27] and Octree-GS [42], en-
hance spatial compactness through voxelization and adap-
tive spatial subdivision. The second line of research fo-
cuses on semantics-oriented extensions, integrating multi-
modal knowledge and high-level understanding into 3DGS.
Examples include Feature 3DGS [63] and M3 [65]. De-
spite these remarkable advancements, a notable gap still
remains between these two paradigms: structure-oriented
methods lack semantic understanding capabilities, whereas
semantics-oriented methods often compromise spatial con-
sistency and structural regularity.
Targeting this gap, we present CUS-GS (shown in Fig-
ure 1), a Compact Unified Structured Gaussian Splatting
framework for multimodal 3D scene representation. It aims
to endow Gaussian splatting based scene representations
with high-level multimodal understanding while maintain-
ing a compact and efficient model structure. In particular,
instead of treating structure and semantics as separate ob-
jectives, our CUS-GS integrates them within a unified rep-
resentation that preserves spatial structure while remaining
semantically expressive.
To this end, we introduce a structured multimodal scene
representation, composed of a 3D voxelized anchor scaf-
fold and a multimodal feature memory. Specifically, we
first construct the multimodal memory by extracting fea-
tures from a set of foundation models and applying redun-
dancy reduction within each feature space. Each voxel an-
chor maintains a learnable latent feature that governs N 3D
Gaussians, which are decoded via a set of shared MLPs
to recover their 3D Gaussian attributes and multimodal se-
mantic features. To ensure robust decoding across heteroge-
neous feature spaces, we design a multimodal latent feature
allocation mechanism together with a hierarchical query
adaptation module, enabling a consistent representation of
appearance, geometry, and semantics. Finally, we propose
a feature-aware anchor pruning strategy that enforces com-
pactness by jointly considering the collective significance
of the Gaussians governed by each anchor, as well as the
capacity and learning dynamics of its latent features. The
primary contributions of our proposed CUS-GS are sum-
marized as follows:
1. This is the first attempt to combine structure and se-
mantics within a unified, compact, and efficient 3DGS
framework for multimodal scene understanding.
2. The novel hierarchical query adaptation mechanism
propagates each anchor’s latent feature to its associ-
ated Gaussian queries through shared MLPs, which
significantly reduces the model size (compared to M3
[65] while achieving spatially coherent and semantically
adaptive rendering.
3. The new feature-aware anchor pruning strategy
jointly considers the rendering-based significance of the
Gaussians governed by each anchor and the capacity and
learning dynamics of its latent features, rather than per-
forming pruning based on Gaussian attributes [8, 27] -
this enables a compact yet semantics-preserving struc-
tured representation.
We conduct extensive experiments on several bench-
marks to evaluate the performance of our proposed CUS-
GS. Results show that CUS-GS achieves competitive im-
age rendering quality and strong multimodal feature align-
ment across multiple foundation models, while maintaining
a model size up to 5× smaller than state-of-the-art meth-
ods. Furthermore, experiments on high-level downstream
tasks demonstrate that CUS-GS effectively balances seman-
tic consistency, structural fidelity, and model compactness,
highlighting its advantage as a unified and efficient multi-
modal scene representation framework.
2. Related Work
In this section, we review related works on neural ren-
dering–based 3D scene representation.
Given the rapid
progress and vast literature in this area, we focus on the
studies most relevant to our problem setting. We begin with
a brief overview of neural rendering–based 3D scene repre-
sentations, followed by two trending directions: structure-
oriented 3DGS and semantic-oriented 3DGS. For a more
comprehensive review, readers are referred to existing sur-
veys [1, 2, 26, 33, 48, 49].
Neural Rendering-based Scene Representation: Neu-
ral rendering, as the integration of traditional computer
graphics and modern deep learning, has established a new
paradigm for 3D scene representation [48, 49]. With the
emergence of NeRF [31], neural rendering has rapidly
become a leading topic in scene representation and re-

<!-- page 3 -->
lated fields, inspiring a wide range of neural implicit ap-
proaches [3, 9, 25, 35, 39, 46]. These methods offer high-
fidelity novel view synthesis, but struggle with real-time
rendering. To address this limitation, many studies have
turned to explicit neural primitives, with 3DGS [15] becom-
ing a central framework. Subsequent work has advanced
3DGS along multiple directions — including expressive ca-
pability [5, 55], robustness [16, 54], efficiency [8, 20, 37],
and generalizability [60, 62].
Structure-oriented 3DGS: Targeting spatial compactness,
a series works have investigated the incorporation of spa-
tial structures and hierarchical organizations.
As one of
the earliest attempts, Scaffold-GS [27] divides the 3D space
into a scaffold of voxels and uses voxel anchors to man-
age nearby 3D Gaussians. Following this idea, various at-
tempts have been explored to make the spatial distribution
of 3D Gaussian more controllable and compact. Examples
include 2D grids in self-organizing gaussian grids [34], 3D
cubes in GaussianCube [59], octrees in OG-Mapping [51],
Octree-GS [42], GS-Octree [24], and hash-grids in HAC [6]
and HAC++ [7]. Alternatively, other works such as Hier-
archical 3DGS [16], HUG [44], and Virtualized 3D Gaus-
sians [56] organize Gaussians in hierarchical blocks or clus-
ters, enabling level-of-detail rendering for large-scale envi-
ronments. Meanwhile, RGBDS-SLAM [4], HiSplat [47],
HRGS [21], and PyGS [52] construct pyramidal or coarse-
to-fine hierarchies, supporting multi-scale reconstruction
and high-resolution rendering.
Semantic-oriented 3DGS: Aiming to make 3DGS mul-
timodally expressive, a parallel line of works augments
3DGS with semantic features extracted from foundation
models. Early approaches distill 2D semantic features from
models like CLIP [41], DiNOv2 [38], and SAM [18] into
the 3DGS representation, as seen by Feature 3DGS [63],
FMGS [66], Semantic Gaussians [11], LEGaussians [43],
and Feature-Splat [40]. Concurrently, other works includ-
ing LUDVIG [28], SLAG [45], and Dr.
Splat [14] di-
rectly project foundation model features onto the Gaussians,
avoiding additional training modules and improving scala-
bility.. In parallel, CLIP-GS [13], UniGS [23], and M3 [65]
introduce trainable latent features to each 3D Gaussian and
align images, text, and 3D into a shared embedding space.
3. Methodology
3DGS is developing rapidly, with numerous works ad-
vancing along two major directions — structure-oriented
and semantic-oriented representations.
Our approach in-
tegrates these two perspectives by constructing a unified
structured multimodal representation that jointly encodes
spatial layout and multimodal features from six foundation
models.
The overall architecture of our proposed CUS-
GS framework is illustrated in Figure 2.
In the follow-
ing, we first briefly review the background of 3DGS and
the principal scene components in Sec. 3.1 to make the pa-
per self-contained. We then present our method in detail —
Sec. 3.2 introduces the structured multimodal representa-
tion, Sec. 3.3 describes the hierarchical query adaptation,
Sec. 3.4 explains the feature-aware growing and pruning
strategy, and Sec. 3.5 provides training details.
3.1. Preliminaries
3D Gaussian Splatting (3DGS) [15] models a 3D scene us-
ing a collection of spatially independent anisotropic Gaus-
sian primitives Gi
N
i=1. Each Gaussian Gi is parameterized
by a mean µi ∈R3, a covariance matrix Σi ∈R3×3, an
opacity αi ∈[0, 1], and view-dependent color coefficients
ci modeled by spherical harmonics.
Given a camera projection matrix P , each Gaussian is
projected onto the image plane as a 2D Gaussian with mean
µ′
i = P µi and covariance
Σ′
i = JiΣiJ⊤
i ,
(1)
where Ji denotes the Jacobian of the projection at µi. The
rendered color at pixel u is then obtained by front-to-back
α-compositing of all projected Gaussians:
C(u) =
N
X
i=1
Ti αi ci exp

−1
2(u −µ′
i)⊤Σ′
i
−1(u −µ′
i)

,
(2)
where Ti = Q
j<i(1 −αj) denotes the accumulated trans-
mittance of all previous Gaussians along the ray.
This differentiable rendering formulation allows 3DGS
to be optimized end-to-end via standard photometric losses,
providing an explicit and efficient representation for high-
quality novel view synthesis.
Principal Scene Component (PSC) [65] is a shared multi-
modal memory bank from multiple foundation models be-
fore linking it to 3D Gaussians. Given a set of scene views
{Iv} and a collection of foundation models {Fj}J
j=1, it first
extracts dense feature maps from each model and granular-
ity, and flattens them into a set of raw feature vectors
R = {rn}Nraw
n=1 ,
∥rn∥2 = 1.
To reduce redundancy while preserving the original em-
bedding spaces, PSC applies a similarity-based reduction.
In particular, it scans R sequentially and selects a vector rn
as a memory entry if its cosine similarity to all previously
selected entries is below a threshold γ. Formally, the result-
ing multimodal memory bank is
M = {mk}K
k=1 ⊂R,
s.t. max
j<k m⊤
k mj < γ,
where each mk is directly taken from the original
foundation-model features and optionally tagged with its

<!-- page 4 -->
Figure 2. Architecture Overview. Our CUS-GS bridges structured 3DGS with multimodal scene understanding through a voxelized
scaffold and a unified multimodal memory bank. Each voxel maintains a latent feature that, together with view-dependent appearance
cues, is decoded into hierarchical queries and Gaussian attributess. Multimodal features extracted from 6 foundation models are first
compressed via PSC, and are then attended by learned queries to retrieve aligned semantic features. The resulting Gaussian attributes drive
differentiable splatting, while the queried semantics enrich the representation, yielding compact, spatially consistent, and multimodally
expressive 3D scene representations.
source model and granularity. This memory bank provides
a compact yet faithful collection of multimodal scene de-
scriptors, which can later be queried by 3D representations.
3.2. Structured Multimodal Representation
To organize both geometry and multimodal semantics in a
unified manner, we divide the 3D space into a voxel scaffold
with cell size l. Each voxel v serves as a spatial anchor that
governs the attributes of N Gaussian primitives {Gv,n}N
n=1
located within its spatial extent. Following the design phi-
losophy of structural 3DGS methods [6, 7, 27, 42], we as-
sociate each voxel with a learnable feature vector zv ∈Rdf
that jointly encodes its local appearance, geometric, and
semantic context, while also interacting with view-related
cues to achieve consistent rendering. Specifically, a view-
dependent appearance embedding ec ∈Rdc is introduced
for each camera c to compensate for illumination and tone
variations across views, and the voxel’s viewing direction
dv is incorporated to maintain angular coherence.
For each Gaussian inside voxel v, its attributes are de-
coded from the combination of (zv, dv, ec) through a set of
shared lightweight MLPs:
αv,n = MLPopacity(zv, dv, ec),
(3)
Σv,n = MLPcov(zv, dv, ec),
(4)
cv,n = MLPcolor(zv, dv, ec),
(5)
where each network is shared across all voxels and jointly
optimized during training. To maintain spatial flexibility,
each Gaussian’s local position is parameterized by a learn-
able offset ∆xv,n relative to the voxel center.
In addition to geometric and appearance attributes, we
further decode per-Gaussian semantic queries for multi-
modal reasoning. A separate network Fθ maps the same
input triplet: (zv, dv, ec) to semantic query vectors:
qv,n = Fθ(zv, dv, ec),
(6)
which serves as the intermediate representation for hierar-
chical query adaptation and multimodal memory retrieval
introduced in Sec. 3.3.
This design allows each voxel to encapsulate local struc-
ture and multimodal context, producing a compact and spa-
tially coherent latent scaffold that can be consistently de-
coded into geometry, appearance, and semantic attributes
across views.
3.3. Hierarchical Query Adaptation
We retrieve multimodal semantic features through a
query–memory mechanism similar to M3 [65], using PSCs
as the memory bank (see Sec. 3.1).
However, flat per-
Gaussian querying often leads to redundant and spatially
inconsistent retrievals, as neighboring Gaussians may cor-
respond to a similar visual context [65]. To overcome this
limitation, we introduce a hierarchical adaptation mecha-
nism that propagates multimodal queries from voxel-level
anchors to individual Gaussians, ensuring both spatial co-
herence and semantic adaptability.

<!-- page 5 -->
Specifically, each voxel feature zv is first mapped to a
voxel-level query qv ∈Rdq through a lightweight MLP:
qv = MLPqv(zv, dv, ec),
(7)
which encodes aggregated appearance, geometric, and mul-
timodal cues within the voxel. For each Gaussian Gv,n gov-
erned by voxel v, a finer query qv,n ∈Rdq is derived from
the voxel query and the Gaussian’s relative offset:
qv,n = Linearqg([qv, ∆xv,n]) ,
(8)
where ∆xv,n denotes the offset. This hierarchical propa-
gation allows local Gaussian queries to inherit the seman-
tic priors of their voxel anchors while capturing position-
specific variation.
The per-Gaussian queries are first passed through stan-
dard rendering pipeline to obtain predicted queries qpred,
which then attend to the multimodal memory bank via dot-
product attention in the adapted gaussian memory attention
module:
apred,k =
exp(q⊤
predmk/
√
d)
P
j exp(q⊤
v,nmj/
√
d)
,
(9)
and the retrieved multimodal feature is obtained by
weighted aggregation:
˜f pred =
X
k
apred,kmk.
(10)
To handle distribution discrepancies among heteroge-
neous foundation-model features, a linear adaptation layer
W adapt is further applied:
f pred = W adapt ˜f pred.
(11)
Through this hierarchical query adaptation, we avoid
storing high-dimensional queries for every Gaussian,
achieving a more compact representation, while the shared
voxel-level queries also encourage smoother semantics
across neighboring Gaussians.
3.4. Feature-aware Pruning
To enhance compactness and better exploit voxel-level in-
formation, we introduce a feature-aware pruning strategy
that evaluates the importance of each voxel using both
voxel-feature statistics and Gaussian-attribute indicators.
For voxel-feature statistics, we use two measures: the
feature norm |zv|2, which reflects the magnitude of the
learned representation, and the gradient norm ∥∇zvL∥2,
which indicates the remaining learning dynamics and thus
the potential for further optimization.
For Gaussian-attribute indicators, we compute per-
Gaussian contribution score Scontrib
v,n
following the global
significance formulation in [8], which accumulates its hit
counts over training rays weighted by opacity and normal-
ized volume. We then aggregate the voxel-level scores as
Scontrib
v
= PN
n=1 Scontrib
v,n
.
The final significance score for voxel v is defined as
Sv = λv,n∥zv∥2 + λv,g∥∇zvL∥2 + Scontrib
v
.
(12)
As voxels with low Sv are pruned, this feature-aware
criterion adaptively refines the voxel scaffold by balanc-
ing representation strength, learning potential, and render-
ing contribution.
3.5. Loss Design
The overall training objective combines image reconstruc-
tion and multimodal feature alignment losses:
L = Limg + λl,e Lfeat.
(13)
The image reconstruction loss Limg enforces photometric
consistency between rendered and ground-truth RGB im-
ages, defined as
Limg = λl,n LL1 + λl,ssim LSSIM + λl,s Lscale,
(14)
where LL1 and LSSIM are pixel-wise ℓ1 and structural sim-
ilarity losses, respectively, and Lscale is an anchor-scale
regularization term to stabilize voxel growth.
The multimodal embedding loss Lfeat aligns the recon-
structed features with their foundation-model counterparts:
Lfeat = Lℓ2 + Lcos,
(15)
where Lℓ2 mmatches feature magnitudes and Lcos enforces
directional consistency.
This combination ensures faith-
ful RGB reconstruction while maintaining semantic consis-
tency across multimodal embeddings.
4. Experiments
4.1. Experimental Setup
Implementation Details.
For the proposed CUS-GS
framework, the voxel size is fixed to l = 0.01 in all experi-
ments. Each voxel governs N = 10 Gaussians, and both the
per-voxel feature vector and the per-camera appearance em-
bedding are set to dimension df = dc = 32. All MLPs con-
sist of two layers with hidden dimensions equal to df and
ReLU activations. For feature-aware pruning, we remove
anchors with the lowest 0.1% importance and set λv,n = 2
and λv,g = 8, as validated in the ablation study.
Regarding multimodal semantics, we follow M3 [65]
and use a 160 −d query vector to encode features extracted
from six foundation models: CLIP [41], SigLIP [58], DI-
NOv2 [38], SEEM [64], and LLaMA-3.1/3.2v [10]. We
train all models for 30,000 iterations using the Adam

<!-- page 6 -->
Table 1. Quantitative Comparison on Image Rendering Quality: Our CUS-GS achieves competitive rendering quality with the smallest
multimodal model size, demonstrating the efficiency of our structured multimodal representation.
Method
Mip-NeRF 360 [3]
Tank and Temple [19]
Size↓
PSNR↑
SSIM↑
LPIPS↓
Size↓
PSNR↑
SSIM↑
LPIPS↓
Plenoxels [9]
2.1GB
23.08
0.626
0.463
2.3GB
21.08
0.719
0.379
Instant NGP [35]
48MB
25.59
0.699
0.331
48MB
21.92
0.745
0.305
VQ-DVGO [25]
63MB
24.23
0.636
0.393
-
-
-
-
3DGS [15]
734MB
27.21
0.815
0.214
411MB
23.14
0.841
0.183
Scaffold-GS [27]
163MB
28.84
0.848
0.220
148MB
23.96
0.853
0.177
Compact 3DGS [20]
48MB
27.08
0.798
0.247
39MB
23.32
0.831
0.201
Compressed 3DGS [37]
28MB
27.03
0.802
0.238
17MB
23.54
0.838
0.189
LightGaussian [8]
45MB
27.13
0.806
0.237
25MB
23.44
0.832
0.202
M3∗[65]
1.1GB
24.14
0.799
0.223
747MB
25.80
0.875
0.171
CUS-GS (Ours)
20MB
26.19
0.822
0.193
19MB
24.79
0.835
0.227
optimizer [17], with the same learning rate schedule as
M3 [65],assigning different rates to appearance, geometry,
and multimodal parameter groups.
Datasets and Metrics. To evaluate CUS-GS, we conduct
experiments on 13 scenes from three public datasets [3, 12,
19]. Following standard practice in 3DGS and multimodal
scene representation [8, 27, 65], we adopt the same evalu-
ation protocol as prior studies. Ourbenchmark includes all
nine real-world scenes from Mip-NeRF360 [3], (five out-
door and four indoor), the “Train” and “Truck” scenes from
the Tank & Temples [19] dataset, and the “Playroom” and
“DrJohnson” scenes from DeepBlending [12].
For evaluating imgage rendering quality, we adopt the
standard metrics used in novel view synthesis:
SSIM,
PSNR, and LPIPS. To assess the quality of multimodal se-
mantic features, we evaluate from both low-level feature
alignment and high-level downstream task performance
perspectives. For low-level feature evaluation, we compute
cosine distance and ℓ2 distance between reconstructed and
reference feature maps. For high-level evaluation, follow-
ing M3 [65] and Feature 3DGS [63], we report mIoU, cIoU,
AP50, and AP60 for semantic retrieval with CLIP [41] fea-
tures, and I2T@1/@5/@10 and T2I@1/@5/@10 for im-
age–text retrieval with SigLIP [58] features. We also report
model size for all the experiments, including both check-
point storage size and total parameter count. All compari-
son results follow the numbers reported in the original pa-
pers; when unavailable, we reproduce them with the au-
thors’ official implementations and mark them with an as-
terisk (∗).
Baselines. Since we evaluate our proposed CUS-GS for
both image rendering quality and semantic feature align-
ment, different sets of comparison methods are selected
for each task.
For color rendering, we include three
NeRF-style methods, Plenoxels [9], Instant NGP [35],
and VQ-DVGO [25] - as well as five 3DGS-based ap-
proaches, Scaffold-GS [27], Compressed 3DGS [37], Com-
pact 3DGS [20], and LightGaussian [8], together with the
original 3DGS [15]. For semantic evaluation, we further
compare against foundation-model–enhanced 3DGS meth-
ods: Feature Splatting (F-Splat) [40], Feature 3DGS (F-
3DGS) [63], and M3 [65], with M3 evaluated for both color
rendering and multimodal semantics.
4.2. Experimental Results
Quantitative Results.
Table 1 presents the quantitative
results on image rendering quality.
Across all datasets,
CUS-GS maintains the smallest model size (20–19 MB),
more than an order of magnitude smaller than M3 (1.1
GB) and comparable to the most compact appearance-only
3DGS variants. On the Mip-NeRF360 dataset, our method
achieves the best LPIPS (0.193) and the second-highest
SSIM (0.822), reflecting strong perceptual and structural
fidelity despite a moderate PSNR drop.
This trend is
consistent with the dataset’s diverse, texture-rich scenes,
where multimodal supervision provides meaningful regu-
larization and reduces sensitivity to pixel-level noise. On
the geometry-dominant Tank and Temples dataset, CUS-GS
attains the second-best PSNR (24.79) while retaining com-
petitive SSIM and LPIPS scores, indicating that the model
preserves high-quality geometric reconstruction even with
multimodal feature integration.
Table 2 summarizes the quantitative comparison on se-
mantic feature alignment. Overall, CUS-GS performs on
par with M3 across most foundation models, while show-
ing a clear and consistent improvement on LLaMA3 fea-
tures. This suggests that the voxel-structured representa-
tion and hierarchical query adaptation offer stronger spatial
support for language-driven semantic embeddings. A slight
decrease is observed for DINOv2, whose texture-oriented
and high-frequency visual cues are partially smoothed
by voxel-level aggregation.
Overall, these results in-
dicate that CUS-GS improves semantic spatial coher-
ence—particularly for high-level, language-oriented rep-
resentations—without compromising overall multimodal

<!-- page 7 -->
Table 2. Quantitative Comparison on Semantic Feature Alignment: Our CUS-GS achieves competitive alignment with 5× fewer
parameters. “D.” denotes Dataset scenes (T.: Train, G.: Garden, D.: DrJohnson, P.: Playroom), and “# P.” the number of parameters.
D.
Method
# P.
DiNOv2
CLIP
SigLIP
SEEM
LLaMA3
LLaMAv
Cosine↓
ℓ2↓
Cosine↓
ℓ2↓
Cosine↓
ℓ2↓
Cosine↓
ℓ2↓
Cosine↓
ℓ2↓
Cosine↓
ℓ2↓
T.
F-Splat [40]
61M
0.6833
1.9835
0.5998
0.4779
0.6346
0.7851
0.4269
11.720
0.5300
0.2900
0.7026
56.23
F-3DGS [63]
61M
0.3790
1.0108
0.3330
0.1540
0.3692
0.3328
0.1063
0.1034
0.4993
0.0150
0.6288
46.48
M3 [65]
35M
0.5321
1.6810
0.3140
0.2800
0.2811
0.5096
0.1389
0.2251
0.4401
0.0253
0.7069
53.43
CUS-GS (Ours)
13M
0.5515
1.7454
0.3254
0.2871
0.2905
0.5237
0.2036
0.2942
0.2331
0.0251
0.7196
53.56
G.
F-Splat[40]
61M
0.7328
1.9567
0.7005
1.3570
0.7247
0.8698
0.4224
9.4675
0.4944
0.3314
0.7443
60.83
F-3DGS [63]
61M
0.2295
0.6033
0.2105
0.0945
0.2697
0.2585
0.1071
0.1424
0.4139
0.0141
0.4913
43.08
M3 [65]
35M
0.5701
1.7279
0.3168
0.2876
0.2927
0.0004
0.1839
0.3469
0.3387
0.0217
0.7235
58.04
CUS-GS (Ours)
13M
0.6060
1.7991
0.3349
0.2980
0.3056
0.5710
0.1965
0.2817
0.1699
0.0181
0.7374
58.03
D.
F-Splat [40]
61M
0.8107
2.0333
0.6689
0.7877
0.6826
0.7744
0.4650
10.411
0.3757
0.0145
0.8184
54.82
F-3DGS [63]
61M
0.4190
1.1279
0.3344
0.1537
0.3846
0.3552
0.1693
0.2169
0.3853
0.0150
0.6669
47.35
M3 [65]
35M
0.5878
1.7553
0.3435
0.2924
0.2975
0.5366
0.2456
0.4179
0.3175
0.0226
0.7224
52.68
CUS-GS (Ours)
8M
0.6063
1.7851
0.3528
0.2987
0.3060
0.5503
0.2063
0.2864
0.1707
0.0182
0.7602
54.18
P.
F-Splat [40]
61M
0.7956
1.9640
0.6458
0.7808
0.6839
0.7678
0.4745
10.873
0.3915
0.0136
0.8185
59.42
F-3DGS [63]
61M
0.4867
1.2193
0.3813
0.1726
0.4571
0.4094
0.1714
0.2103
0.3987
0.0139
0.6922
52.50
M3 [65]
35M
0.6074
1.7545
0.3260
0.2987
0.2951
0.5623
0.2560
0.4584
0.3555
0.0241
0.7288
57.38
CUS-GS (Ours)
6M
0.6156
1.7782
0.3345
0.3050
0.3041
0.5718
0.2126
0.2937
0.1810
0.0188
0.7333
57.10
alignment quality.
Table 3 reports the quantitative results on high-level
downstream tasks. For CLIP-based semantic segmentation,
CUS-GS shows a moderate drop compared with M3 and
F-3DGS, which can be attributed both to the reduced model
capacity (13–18M vs. 35–61M parameters) and the smooth-
ing inherent to voxel-level aggregation. Because CLIP fea-
tures contain substantial global context and high-frequency
variance, voxel aggregation suppresses local fluctuations
and promotes spatial consistency, improving structural co-
herence while slightly weakening fine-grained semantic dis-
crimination. Nevertheless, CUS-GS maintains competitive
segmentation accuracy under a significantly more compact
configuration.
For SigLIP-based image–text retrieval, the results ex-
hibit a complementary trend: On the Train scene, CUS-
GS achieves performance close to the ground truth and
clearly surpasses both baselines, indicating strong mul-
timodal grounding in structured environments.
In the
more cluttered Playroom scene, I2T performance decreases
whereas T2I metrics remain higher than M3 and F-3DGS.
Suggesting that CUS-GS preserves a more stable language-
to-vision alignment, even under complex scene layouts.
Overall, these results highlight that CUS-GS achieves
a favorable balance between semantic consistency, struc-
tural quality, and model compactness, delivering competi-
tive performance with a model size up to 5× smaller than
state-of-the-art counterparts.
Qualitative Results. Figure 3 shows qualitative compar-
isons of multimodal feature maps reconstructed from M3
and our CUS-GS. Overall, CUS-GS produces smoother
and more spatially coherent representations.
For CLIP
and SigLIP, the reconstructed features exhibit more consis-
tent semantic regions, while LLaMA-based features show
clearer spatial organization and improved alignment with
scene structure.
These observations align well with the
quantitative results in Table 2, confirming that the voxel-
structured design effectively preserves spatial hierarchy
for language-driven semantics. For DINOv2, M3 retains
slightly sharper texture cues, whereas CUS-GS appears
more smoothed, which corresponds to the minor drop in
cosine similarity reported in Table 2. Overall, CUS-GS de-
livers more stable and spatially consistent multimodal rep-
resentations across diverse foundation models.
4.3. Ablation Studies
Finally, we conduct three ablation studies in Table 4 to
validate key design choices of CUS-GS. The first row re-
ports the results of the Default Setup, followed by ablations
on Voxel Size (rows 2–3), Query Granularity (row 4), and
Feature-Awareness Ratio (rows 5–7).
Voxel Size. As shown in Table 4, the voxel size primar-
ily influences the trade-off between model compactness and
rendering quality. A smaller voxel size (0.005) leads to finer
details but increases the number of anchors and parameters,
while a larger voxel size (0.050) significantly reduces model
size at the cost of degraded PSNR. We therefore choose
l = 0.01 as a balanced configuration that maintains high-
quality rendering with moderate memory consumption.
Query Granularity.
Switching from per-anchor to per-
Gaussian query substantially increases the model size
(13.4M →245.6M) with only marginal improvement in
rendering quality (PSNR 24.04 →25.27) and feature align-
ment.
This indicates that hierarchical propagation from
voxel anchors is sufficient to capture multimodal semantics
without the need for redundant per-Gaussian queries.
Feature-Awareness Ratio.
The feature-awareness ratio
balances representation magnitude (λv,n) and learning dy-

<!-- page 8 -->
Table 3. Quantitative Comparison on Downstream Task: Our CUS-GS achieves comparable downstream performance to M3 with up
to 5× fewer parameters, demonstrating strong task generalization under compact model size. “# P.” denotes number of parameters.
Dataset
Method
# P.
CLIP
SigLIP
mIoU
cIoU
AP50
AP60
I2T@1
I2T@5
I2T10
T2I@1
T2I@5
T2I@10
Train
Ground Truth
-
25.3
26.3
14.7
3.3
88.7
98.3
100
97.3
100
100
F-3DGS [63]
61M
24.2
24.3
16.3
7.1
2.6
13.2
28.9
0.0
2.6
18.4
M3 [65]
35M
25.4
26.5
19.6
12.5
55.2
84.2
92.1
52.6
84.2
92.1
CUS-GS (Ours)
13M
15.9
16.2
8.4
4.0
85.1
95.0
97.3
77.1
93.7
98.7
Playroom
Ground Truth
-
25.6
24.2
9.6
3.0
92.9
98.7
100
92.0
98.2
98.1
F-3DGS [63]
61M
23.8
21.4
11.9
3.0
79.3
96.6
6.6
31.0
79.3
89.7
M3 [65]
35M
23.1
23.1
11.9
5.9
72.4
96.6
100
41.3
65.5
68.9
CUS-3DGS (Ours)
6M
15.5
16.2
4.8
2.1
37.3
74.2
81.8
58.2
84.8
90.2
Figure 3. Qualitative results: Comparison of example feature maps reconstructed from different foundation models. Our CUS-GS
produces smoother and more spatially coherent representations than M3, with clearer structures in language-driven features (e.g., LLaMA3)
and slightly smoother texture features (e.g., DINOv2).
namics (λv,g) in the pruning process. As shown in rows
5–7 of Table 4, varying the ratio from 8:2 to 4:6 yields only
marginal differences in SSIM and PSNR, indicating that the
model is robust to moderate weighting changes. Larger λv,n
slightly improves PSNR but reduces compactness, while
higher λv,g maintains smaller anchor sets with comparable
quality. We choose 2:8 as a balanced configuration that pro-
vides stable performance and efficient model size.
5. Conclusion
We have presented CUS-GS, the first Compact Unified
Structured Gaussian Splatting framework for multimodal
3D scene representation.
By bridging structural model-
ing and multimodal understanding, CUS-GS unifies spa-
tial organization and semantic expressiveness within a sin-
gle compact architecture. Through voxel-anchored latent
features, hierarchical query adaptation, and feature-aware
pruning, our method achieves competitive rendering qual-
ity and strong multimodal alignment while reducing model
size by over 5× compared with existing approaches. Ex-
tensive experiments across diverse benchmarks demonstrate
that CUS-GS effectively balances visual fidelity, semantic
consistency, and compactness, showing clear advantages for
both low-level reconstruction and high-level downstream
tasks. We believe this work takes an important step toward
unified, scalable, and semantically grounded 3D scene rep-
resentations, and opens new directions for integrating foun-
dation models into structured 3D learning.
References
[1] M. T. Bagdasarian, P. Knoll, Y. Li, F. Barthel, A. Hilsmann,
P. Eisert, and W. Morgenstern. 3dgs.zip: A survey on 3d
gaussian splatting compression methods. Computer Graph-
ics Forum, 44(2):e70078, 2025. 2
[2] Yanqi Bao, Tianyu Ding, Jing Huo, Yaoli Liu, Yuxin Li,
Wenbin Li, Yang Gao, and Jiebo Luo. 3d gaussian splatting:
Survey, technologies, challenges, and opportunities. IEEE
Transactions on Circuits and Systems for Video Technology,
35(7):6832–6852, 2025. 2
[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded

<!-- page 9 -->
Table 4. Ablation Studies: We analyze the effects of voxel size, query granularity, and feature-awareness (F.A.) ratio on model size, image
rendering quality, and semantic feature alignment.
Dataset
Config
Model Size
RBG Image
Semantic Features
Voxel Size
Query Granularity
F.A. Ratio
#Param
Anchor
SSIM↓
PSNR↓
Avg. Cosine↓
Avg. ℓ2↓
Train
0.010
per-anchor
2:8
13.4M
128501
0.820
24.04
0.4006
9.4101
0.050
per-anchor
2:8
8.4M
62639
0.752
23.05
0.4007
9.4114
0.005
per-anchor
2:8
14.3M
141237
0.815
24.17
0.4006
9.4101
0.010
per-gaussian
2:8
245.6M
150624
0.845
25.27
0.3980
9.3958
0.010
per-anchor
8:2
13.4M
129470
0.818
24.37
0.4006
9.4108
0.010
per-anchor
6:4
13.4M
128955
0.830
24.79
0.4006
9.4101
0.010
per-anchor
4:6
13.4M
129465
0.820
24.52
0.4006
9.4105
anti-aliased neural radiance fields.
In 2022 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 5460–5469, 2022. 3, 6
[4] Zhenzhong Cao, Chenyang Zhao, Qianyi Zhang, Jinzheng
Guang, Yinuo Song, and Jingtai Liu. Rgbds-slam: A rgb-d
semantic dense slam based on 3d multi level pyramid gaus-
sian splatting. IEEE Robotics and Automation Letters, 10(5):
4778–4785, 2025. 3
[5] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian
Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao,
and Guofeng Zhang. Pgsr: Planar-based gaussian splatting
for efficient and high-fidelity surface reconstruction. IEEE
Transactions on Visualization and Computer Graphics, 31
(9):6100–6111, 2025. 3
[6] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi,
and Jianfei Cai. Hac: Hash-grid assisted context for 3d gaus-
sian splatting compression.
In Computer Vision – ECCV
2024: 18th European Conference, Milan, Italy, September
29–October 4, 2024, Proceedings, Part VII, page 422–438,
Berlin, Heidelberg, 2024. Springer-Verlag. 3, 4
[7] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi,
and Jianfei Cai. Hac++: Towards 100x compression of 3d
gaussian splatting. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 47(11):10210–10226, 2025. 3, 4
[8] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia
Xu, and Zhangyang Wang. Lightgaussian: Unbounded 3d
gaussian compression with 15x reduction and 200+ FPS. In
The Thirty-eighth Annual Conference on Neural Information
Processing Systems, 2024. 2, 3, 5, 6, 1
[9] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In 2022 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 5491–5500, 2022. 3, 6
[10] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Ab-
hinav Pandey, et al. The llama 3 herd of models, 2024. 5
[11] Jun Guo, Xiaojian Ma, Yue Fan, Huaping Liu, and Qing Li.
Semantic gaussians: Open-vocabulary scene understanding
with 3d gaussian splatting, 2024. 3
[12] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm,
George Drettakis, and Gabriel Brostow. Deep blending for
free-viewpoint image-based rendering. ACM Trans. Graph.,
37(6), 2018. 6
[13] Siyu Jiao, Haoye Dong, Yuyang Yin, Zequn Jie, Yinlong
Qian, Yao Zhao, Humphrey Shi, and Yunchao Wei. Clip-
gs: Unifying vision-language representation with 3d gaus-
sian splatting. In Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), pages 4670–4680,
2025. 3
[14] Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank
Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. splat: Directly
referring 3d gaussian splatting via direct language embed-
ding registration. In 2025 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 14137–
14146, 2025. 3
[15] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4), 2023.
2, 3, 6
[16] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas,
Michael Wimmer, Alexandre Lanvin, and George Drettakis.
A hierarchical 3d gaussian representation for real-time ren-
dering of very large datasets. ACM Trans. Graph., 43(4),
2024. 3
[17] Diederik P. Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. In 3rd International Conference on
Learning Representations, ICLR 2015, San Diego, CA, USA,
May 7-9, 2015, Conference Track Proceedings, 2015. 6
[18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and
Ross Girshick. Segment anything. In 2023 IEEE/CVF In-
ternational Conference on Computer Vision (ICCV), pages
3992–4003, 2023. 3
[19] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun. Tanks and temples: benchmarking large-scale scene
reconstruction. ACM Trans. Graph., 36(4), 2017. 6
[20] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko,
and Eunbyung Park.
Compact 3d gaussian representation
for radiance field. In 2024 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 21719–
21728, 2024. 3, 6
[21] Changbai Li, Haodong Zhu, Hanlin Chen, Juan Zhang,
Tongfei Chen, Shuo Yang, Shuwei Shao, Wenhao Dong, and
Baochang Zhang. Hrgs: Hierarchical gaussian splatting for
memory-efficient high-resolution 3d reconstruction, 2025. 3
[22] Hai Li, Xingrui Yang, Hongjia Zhai, Yuqian Liu, Hujun Bao,
and Guofeng Zhang.
Vox-surf: Voxel-based implicit sur-
face representation. IEEE Transactions on Visualization and
Computer Graphics, 30(3):1743–1755, 2024. 2

<!-- page 10 -->
[23] Haoyuan Li, Zhou Yanpeng, Tao Tang, Jifei Song, Yihan
Zeng, Michael Kampffmeyer, Hang Xu, and Xiaodan Liang.
UniGS: Unified language-image-3d pretraining with gaus-
sian splatting. In The Thirteenth International Conference
on Learning Representations, 2025. 3
[24] Jiaze Li, Zhengyu Wen, Luo Zhang, Jiangbei Hu, Fei Hou,
Zhebin Zhang, and Ying He. Gs-octree: Octree-based 3d
gaussian splatting for robust object-level 3d reconstruction
under strong lighting, 2024. 3
[25] Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and
Liefeng Bo.
Compressing volumetric radiance fields to 1
mb. In 2023 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 4222–4231, 2023. 3, 6
[26] Yun Liao, Yide Di, Hao Zhou, Kaijun Zhu, Mingyu Lu, Qing
Duan, and Junhui Liu. A survey on neural radiance fields.
ACM Comput. Surv., 58(2), 2025. 2
[27] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering. In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 20654–20664, 2024. 2, 3, 4, 6, 1
[28] Juliette Marrie, Romain Menegaux, Michael Arbel, Diane
Larlus, and Julien Mairal.
Ludvig: Learning-free uplift-
ing of 2d visual features to gaussian splatting scenes.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), pages 7440–7450, 2025. 3
[29] Ruben Mascaro and Margarita Chli. Scene representations
for robotic spatial perception.
Annual Review of Control,
Robotics, and Autonomous Systems, 8(Volume 8, 2025):351–
377, 2025. 2
[30] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and
Andrew J. Davison.
Gaussian splatting slam.
In 2024
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 18039–18048, 2024. 2
[31] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
representing scenes as neural radiance fields for view synthe-
sis. Commun. ACM, 65(1):99–106, 2021. 2
[32] Yuhang Ming, Di Ma, Weichen Dai, Han Yang, Rui Fan,
Guofeng Zhang, and Wanzeng Kong. Slc2-slam: Semantic-
guided loop closure using shared latent code for nerf slam.
IEEE Robotics and Automation Letters, 10(5):4978–4985,
2025. 2
[33] Yuhang Ming, Xingrui Yang, Weihan Wang, Zheng Chen,
Jinglun Feng, Yifan Xing, and Guofeng Zhang.
Bench-
marking neural radiance fields for autonomous robots: An
overview.
Engineering Applications of Artificial Intelli-
gence, 140:109685, 2025. 2
[34] Wieland Morgenstern, Florian Barthel, Anna Hilsmann, and
Peter Eisert.
Compact 3d scene representation via self-
organizing gaussian grids.
In Computer Vision – ECCV
2024: 18th European Conference, Milan, Italy, September
29–October 4, 2024, Proceedings, Part LXXXV, page 18–34,
Berlin, Heidelberg, 2024. Springer-Verlag. 3
[35] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a multires-
olution hash encoding. ACM Trans. Graph., 41(4), 2022. 3,
6
[36] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea
Finn, and Abhi Gupta. R3m: A universal visual representa-
tion for robot manipulation. In Conference on Robot Learn-
ing, 2022. 2
[37] Simon Niedermayr, Josef Stumpfegger, and R¨udiger West-
ermann. Compressed 3d gaussian splatting for accelerated
novel view synthesis.
In 2024 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
10349–10358, 2024. 3, 6
[38] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, et al. Dinov2: Learning robust visual features without
supervision, 2024. 3, 5
[39] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
Francesc Moreno-Noguer. D-nerf: Neural radiance fields for
dynamic scenes. In 2021 IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 10313–
10322, 2021. 3
[40] Ri-Zhao Qiu, Ge Yang, Weijia Zeng, and Xiaolong Wang.
Language-driven physics-based scene synthesis and editing
via feature splatting. In Computer Vision – ECCV 2024: 18th
European Conference, Milan, Italy, September 29–October
4, 2024, Proceedings, Part XLI, page 368–383, Berlin, Hei-
delberg, 2024. Springer-Verlag. 3, 6, 7
[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever.
Learning transferable visual
models from natural language supervision. In Proceedings
of the 38th International Conference on Machine Learning,
ICML 2021, 18-24 July 2021, Virtual Event, pages 8748–
8763. PMLR, 2021. 3, 5, 6, 1
[42] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu,
Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent
real-time rendering with lod-structured 3d gaussians. IEEE
Transactions on Pattern Analysis and Machine Intelligence,
pages 1–15, 2025. 2, 3, 4
[43] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-
Hua Guan.
Language embedded 3d gaussians for open-
vocabulary scene understanding. In 2024 IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 5333–5343, 2024. 3
[44] Mai Su, Zhongtao Wang, Huishan Au, Yilong Li, Xizhe Cao,
Chengwei Pan, Yisong Chen, and Guoping Wang.
Hug:
Hierarchical urban gaussian splatting with block-based re-
construction for large-scale aerial scenes. In Proceedings of
the IEEE/CVF International Conference on Computer Vision
(ICCV), pages 28839–28848, 2025. 3
[45] Laszlo Szilagyi, Francis Engelmann, and Jeannette Bohg.
Slag:
Scalable language-augmented gaussian splatting.
IEEE Robotics and Automation Letters, 10(7):6991–6998,
2025. 3
[46] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Prad-
han, Ben P. Mildenhall, Pratul Srinivasan, Jonathan T. Bar-
ron, and Henrik Kretzschmar.
Block-nerf: Scalable large
scene neural view synthesis. In 2022 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
8238–8248, 2022. 3

<!-- page 11 -->
[47] Shengji Tang, Weicai Ye, Peng Ye, Weihao Lin, Yang Zhou,
Tao Chen, and Wanli Ouyang.
Hisplat: Hierarchical 3d
gaussian splatting for generalizable sparse-view reconstruc-
tion. In The Thirteenth International Conference on Learn-
ing Representations, 2025. 3
[48] A. Tewari, O. Fried, J. Thies, V. Sitzmann, S. Lombardi,
K. Sunkavalli, R. Martin-Brualla, T. Simon, J. Saragih, M.
Nießner, R. Pandey, S. Fanello, G. Wetzstein, J.-Y. Zhu, C.
Theobalt, M. Agrawala, E. Shechtman, D. B Goldman, and
M. Zollh¨ofer. State of the art on neural rendering. Computer
Graphics Forum, 39(2):701–727, 2020. 2
[49] A. Tewari, J. Thies, B. Mildenhall, P. Srinivasan, E. Tretschk,
W. Yifan, C. Lassner, V. Sitzmann, R. Martin-Brualla, S.
Lombardi, T. Simon, C. Theobalt, M. Nießner, J. T. Bar-
ron, G. Wetzstein, M. Zollh¨ofer, and V. Golyanik. Advances
in neural rendering. Computer Graphics Forum, 41(2):703–
735, 2022. 2
[50] Hanqing Wang, Wenguan Wang, Wei Liang, Caiming Xiong,
and Jianbing Shen.
Structured scene memory for vision-
language navigation.
In 2021 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
8451–8460, 2021. 2
[51] Meng Wang, Junyi Wang, Changqun Xia, Chen Wang, and
Yue Qi. Og-mapping: Octree-based structured 3d gaussians
for online dense mapping, 2024. 3
[52] Zipeng Wang and Dan Xu. Pygs: Large-scale scene repre-
sentation with pyramidal 3d gaussian splatting, 2024. 3
[53] Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, and
Shuqiang Jiang. Gridmm: Grid memory map for vision-and-
language navigation. In 2023 IEEE/CVF International Con-
ference on Computer Vision (ICCV), pages 15579–15590,
2023. 2
[54] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In 2024 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 20310–20320, 2024. 3
[55] Yusen Xie,
Zhenmin Huang,
Jianhao Jiao,
Dimitrios
Kanoulas, and Jun Ma.
Unigs: Unified geometry-aware
gaussian splatting for multimodal rendering, 2025. 3
[56] Xijie Yang, Linning Xu, Lihan Jiang, Dahua Lin, and Bo
Dai. Virtualized 3d gaussians: Flexible cluster-based level-
of-detail system for real-time rendering of composed scenes.
In Proceedings of the Special Interest Group on Computer
Graphics and Interactive Techniques Conference Conference
Papers, New York, NY, USA, 2025. Association for Comput-
ing Machinery. 3
[57] Qiaojun Yu, Xibin Yuan, Yu jiang, Junting Chen, Dongzhe
Zheng, Ce Hao, Yang You, Yixing Chen, Yao Mu, Liu Liu,
and Cewu Lu.
Artgs:3d gaussian splatting for interactive
visual-physical modeling and manipulation of articulated ob-
jects. In 2025 IEEE/RSJ International Conference on Intel-
ligent Robots and Systems (IROS), 2025. 2
[58] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and
Lucas Beyer. Sigmoid loss for language image pre-training.
In 2023 IEEE/CVF International Conference on Computer
Vision (ICCV), pages 11941–11952, 2023. 5, 6, 1
[59] Bowen Zhang, Yiji Cheng, Jiaolong Yang, Chunyu Wang,
Feng Zhao, Yansong Tang, Dong Chen, and Baining Guo.
Gaussiancube: A structured and explicit radiance represen-
tation for 3d generative modeling. In The Thirty-eighth An-
nual Conference on Neural Information Processing Systems,
2024. 3
[60] Chuanrui Zhang, Yingshuang Zou, Zhuoling Li, Minmin Yi,
and Haoqian Wang. Transplat: Generalizable 3d gaussian
splatting from sparse multi-view images with transformers.
Proceedings of the AAAI Conference on Artificial Intelli-
gence, 39(9):9869–9877, 2025. 3
[61] Shuaifeng Zhi, Michael Bloesch, Stefan Leutenegger, and
Andrew J. Davison. Scenecode: Monocular dense semantic
reconstruction using learned encoded scene representations.
In 2019 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 11768–11777, 2019. 2
[62] Boyao Zhou, Shunyuan Zheng, Hanzhang Tu, Ruizhi Shao,
Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu.
Gps-gaussian+: Generalizable pixel-wise 3d gaussian splat-
ting for real-time human-scene rendering from sparse views.
IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, pages 1–16, 2025. 3
[63] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi. Feature 3dgs: Supercharging
3d gaussian splatting to enable distilled feature fields.
In
2024 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 21676–21685, 2024. 2, 3,
6, 7, 8
[64] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li,
Jianfeng Wang, Lijuan Wang, Jianfeng Gao, and Yong Jae
Lee. Segment everything everywhere all at once. In Thirty-
seventh Conference on Neural Information Processing Sys-
tems, 2023. 5
[65] Xueyan Zou, Yuchen Song, Ri-Zhao Qiu, Xuanbin Peng,
Jianglong Ye, Sifei Liu, and Xiaolong Wang. 3d-spatial mul-
timodal memory. In The Thirteenth International Conference
on Learning Representations, 2025. 2, 3, 4, 5, 6, 7, 8, 1
[66] Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan Di,
and Mingyang Li. FMGS: foundation model embedded 3d
gaussian splatting for holistic 3d scene understanding. Int. J.
Comput. Vis., 133(2):611–627, 2025. 3

<!-- page 12 -->
CUS-GS: A Compact Unified Structured Gaussian Splatting Framework for
Multimodal Scene Representation
Supplementary Material
6. Overview
In this supplementary material, we first provide additional
experimental details in Sec. 7. We then include extended
qualitative results to further illustrate the capabilities of
CUS-GS, including additional image rendering results in
Sec. 8, semantic feature rendering comparisons in Sec. 9,
and visualizations for downstream tasks in Sec. 10.
7. Additional Experimental Details
In this section, we provide additional details on ground
truth generation and evaluation procedure for the quanti-
tative comparison in Table 3. To ensure full consistency
with the evaluation protocol of M3 [65], we adopt the same
ground-truth construction pipeline for both CLIP [41] seg-
mentation and SigLIP [58] retrieval. Since SoM (semantic-
SAM) and GPT-4o used in M3 are not publicly available,
we replace them with fully open-source counterparts: SAM
for mask generation and Qwen-VL for caption generation.
This substitution does not change the mask granularity or
the supervision structure, and all feature extraction and eval-
uation steps remain identical to M3.
For CLIP segmentation, we first apply SAM to the in-
put RGB image to obtain region-level masks, which are
then passed through Qwen-VL to obtain natural-language
descriptions. During evaluation, the image feature maps
are taken from the reconstructed multimodal representation,
while the text embeddings are obtained by encoding the re-
gion descriptions with the CLIP text encoder. Both features
are normalized, and their dot-product similarity is computed
to form similarity maps corresponding to each region de-
scription. These maps are upsampled to the resolution of
the SAM masks and thresholded using Otsu’s method to
produce predicted regions. We then compare the predic-
tions with the SAM masks to compute mIoU, cIoU, and
AP@0.5/0.6.
For SigLIP retrieval, we use Qwen-VL to generate a
global caption for the target image, and then extract the im-
age and text embeddings using the SigLIP vision and text
encoders. We additionally encode COCO images and their
captions with SigLIP to form a negative sample pool, which
is combined with the target features to construct the retrieval
set. During evaluation, we compute cosine similarities be-
tween the target image embedding and all text embeddings
to obtain I2T@k, and likewise compute similarities between
the target caption embedding and all image embeddings to
obtain T2I@k.
8. Image Rendering Results
Figure 4 presents qualitative comparisons of the image ren-
dering results. While the rendered images across different
methods appear visually similar at first glance, the residual
maps reveal a much clearer distinction. Our CUS-GS con-
sistently produces smaller and more uniformly distributed
residuals, indicating more accurate color reconstruction and
better alignment with ground-truth imagery.
In contrast,
both M3 [65] (the only multimodal counterpart) and Light-
Gaussian [8] exhibit noticeable structured errors, particu-
larly around object boundaries and high-frequency regions,
while Scaffold-GS [27] shows moderate deviations.
9. Semantic Feature Rendering Results
Figure 5 presents additional semantic feature rendering
results for the remaining two scenes, “DrJohnson” and
“Train,” complementing the qualitative comparisons shown
in the main text. Across both scenes, CUS-GS consistently
produces cleaner and more spatially coherent multimodal
features, with CLIP and SigLIP exhibiting more stable se-
mantic regions and LLaMA-based features showing clearer
structural organization. As observed previously, DINOv2
features from CUS-GS appear slightly smoother than those
from M3, corresponding to the minor alignment trade-off
discussed in Table 2. Overall, these additional visualiza-
tions further confirm the robustness and spatial consistency
of the multimodal representations rendered by CUS-GS.
10. Downstream Tasks Visualization
In this section, we provide additional visualizations for the
downstream tasks—retrieval, grounding, and caption—as
shown in Figure 6, Figure 7, and Figure 8. These exam-
ples illustrate how the multimodal features reconstructed by
CUS-GS can be directly used in practical semantic reason-
ing tasks through the decoders of their corresponding foun-
dation models. The results show that the rendered features
remain well aligned with the native feature spaces of these
models, enabling them to produce meaningful and coher-
ent outputs without any task-specific adaptation. This fur-
ther verifies that CUS-GS preserves not only semantic con-
sistency but also functional usability across diverse down-
stream scenarios.

<!-- page 13 -->
Figure 4. Examples of the Image Rendering Results. For each scene, the upper row shows the rendered image, and the lower row
presents the residual between the rendered image and the ground truth image.

<!-- page 14 -->
Figure 5. Additional Feature Rendering Results. These results provide complementary evidence that CUS-GS outperforms M3 in
producing cleaner and more structured multimodal feature fields.
Figure 6. Examples of Retrieval Tasks. Given text queries (“Trash Bin”, “Toys”, “Ball”, “Fire Extinguisher”), we show the top-1
retrievals using CUS-GS’s reconstructed features. The top row displays example views from the corresponding scenes, and the bottom row
demonstrates the retrieved target, successfully locating the correct object instances.
Figure 7. Examples of Grounding Tasks. The upper row shows the input view and text prompt, and the lower row presents the object
mask.

<!-- page 15 -->
Figure 8. Examples of Caption Tasks. The lefter column shows the input view and the righter column demonstrates the generated caption.
