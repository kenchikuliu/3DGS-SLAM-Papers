<!-- page 1 -->
Laplacian Analysis Meets Dynamics Modelling: Gaussian Splatting
for 4D Reconstruction
Yifan Zhou * 1, Beizhen Zhao * 1, Pengcheng Wu 2, Hao Wang † 1
1 The Hong Kong University of Science and Technology (Guangzhou)
2 Nanyang Technological University
yifanzhou@hkust-gz.edu.cn, bzhao610@connect.hkust-gz.edu.cn, pengchengwu@ntu.edu.sg, haowang@hkust-gz.edu.cn
Abstract
While 3D Gaussian Splatting (3DGS) excels in static scene
modeling, its extension to dynamic scenes introduces sig-
nificant challenges. Existing dynamic 3DGS methods suffer
from either over-smoothing due to low-rank decomposition
or feature collision from high-dimensional grid sampling.
This is because of the inherent spectral conflicts between
preserving motion details and maintaining deformation con-
sistency at different frequency. To address these challenges,
we propose a novel dynamic 3DGS framework with hybrid
explicit-implicit functions. Our approach contains three key
innovations: a spectral-aware Laplacian encoding architec-
ture which merges Hash encoding and Laplacian-based mod-
ule for flexible frequency motion control, an enhanced Gaus-
sian dynamics attribute that compensates for photometric
distortions caused by geometric deformation, and an adap-
tive Gaussian split strategy guided by KDTree-based primi-
tive control to efficiently query and optimize dynamic areas.
Through extensive experiments, our method demonstrates
state-of-the-art performance in reconstructing complex dy-
namic scenes, achieving better reconstruction fidelity.
Introduction
Dynamic scene reconstruction from monocular videos
presents a critical challenge in computer vision, demand-
ing precise modeling of both persistent geometric structures
and transient deformations (Cai et al. 2022; Du et al. 2021;
Fang et al. 2022; Kratimenos, Lei, and Daniilidis 2024; Li,
Slavcheva et al. 2022). Unlike static environments, dynamic
scenes exhibit heterogeneous motion patterns - rigid com-
ponents maintain temporal consistency while deformable re-
gions require high-frequency trajectory modeling (Xu et al.
2024; Duan et al. 2024; Bae et al. 2024). This inherent com-
plexity creates dual challenges: preserving spatial coherence
across time-varying geometries and capturing transient de-
formation details without over-smoothing artifacts.
While Neural Radiance Fields (NeRF) (Mildenhall et al.
2021; Chan et al. 2022; Yang, Vo et al. 2022; Park et al.
2021b; Martin-Brualla et al. 2021) revolutionized static
scene modeling through continuous volumetric integration,
their dynamic extensions (Choe et al. 2023; Gao et al. 2021;
Liang, Laidlaw et al. 2023; Liu et al. 2023; Wang et al. 2023;
*These authors contributed equally to this work.
†Corresponding author.
Barron et al. 2021; Fridovich-Keil et al. 2023) reveal critical
limitations in handling temporal discontinuities, particularly
the conflicting requirements for spatial fidelity versus tem-
poral coherence arising from uniform spectrum allocation.
Although explicit representations (Barron, Mildenhall et al.
2023; Cao and Johnson 2023; Tancik et al. 2022) improve
efficiency through 4D spacetime factorization, their low-
rank decomposition induces feature collision in overlap-
ping regions. Recent 3D Gaussian Splatting (3DGS) (Kerbl
et al. 2023; Duisterhof et al. 2023; Yang et al. 2023; Liang
et al. 2023; Lin et al. 2024) has achieved impressive effects
for static environments, where discrete volumetric primi-
tives enable both photorealistic rendering and computation-
ally efficient optimization through differentiable rasteriza-
tion (Shao et al. 2023; Wu et al. 2024; Lu et al. 2024; Luiten
et al. 2024; Kratimenos, Lei, and Daniilidis 2024).
However, their direct extension to dynamic scenarios
faces three fundamental limitations: 1) existing deformable
methods suffer from either over-smoothing due to low-rank
decomposition or feature collision from high-dimensional
grid sampling, 2) previous Gaussian-based methods use
a fixed threshold during Gaussian split stage which ig-
nore adaptive split adjustment, and 3) persistent appearance
changes caused by dynamic deformation are often neglected
in current pipelines.
To address the challenges above, our key insight lies in ad-
dressing the anisotropic spatio-temporal sampling nature of
dynamic scenes through hybrid explicit-implicit encoding.
First, we develop a hybrid spectral-aware encoder combin-
ing Hash grids with Laplacian-based module that decouples
spatial and temporal features into different frequency mo-
tion components, overcoming the feature collision of low-
rank assumption while enabling adaptive frequency motion
control. Then, we design an enhanced Gaussian dynamics
attribute to perform individual Gaussian personalized dy-
namic optimization and design an adaptive regularization for
identifying highly dynamic areas. Besides, we propose an
adaptive Gaussian split strategy, which focuses on the opti-
mization trade-off between Gaussian shape and anisotropy
in dynamic scenes and an improved KDTree-based cluster-
ing algorithm was proposed to efficiently query and optimize
dynamic Gaussians.
Our solution rethinks dynamic 3DGS through Laplacian
spectral analysis, which provides a hybrid framework for
arXiv:2508.04966v1  [cs.GR]  7 Aug 2025

<!-- page 2 -->
localized frequency analysis. Meanwhile, we focus on the
dynamics attribute of each Gaussian and the optimization
problem in the derivation process, and propose a novel hy-
brid explicit-implicit algorithm model. We propose three key
innovations. In summary, our contributions are as follows:
• We propose a spectral-aware Laplacian encoding module
combining multi-scale Hash encoding with Laplacian-
based dynamic module that decouples different fre-
quency motion trajectories from complex deformation.
• We design an enhanced Gaussian dynamics attribute that
identify highly dynamic areas for adaptive split and reg-
ularization.
• We design an adaptive Gaussian split strategy that au-
tomatically adjusts the primitive density and anisotropy
using KDTree-guided spectral analysis.
Related Work
NeRF-based Dynamic Modeling
The advent of Neural Radiance Fields (NeRF) has sig-
nificantly transformed the landscape of 3D scene recon-
struction, particularly for static environments. However, ex-
tending NeRF to effectively model dynamic scenes re-
mains a formidable challenge. Initial efforts, such as D-
NeRF (Pumarola et al. 2021) and Nerfies (Park et al. 2021a),
have employed canonical space warping and temporal la-
tent codes to capture motion. Despite their innovative ap-
proaches, these methods often exhibit limitations when deal-
ing with rapid or abrupt movements. Moreover, explicit
spacetime factorization techniques, such as HexPlane (Cao
and Johnson 2023), have been proposed to enhance com-
putational efficiency. However, these methods impose re-
strictive low-rank assumptions that may oversimplify the in-
tricate dynamics present in real-world scenes, particularly
in environments characterized by rapid changes. Further-
more, while segmenting scenes into components with dis-
tinct attributes has been explored to enhance modeling accu-
racy (Gao et al. 2021; Tretschk, Tewari et al. 2021), the im-
plicit representations based on fully connected MLPs often
suffer from over-smoothing and lengthy training processes.
3DGS-based Dynamic Modeling
Recent advances in 3D Gaussian Splatting (3DGS) (Kerbl
et al. 2023; Yu et al. 2024; Huang et al. 2024; Li et al.
2024) have demonstrated remarkable success in static scene
reconstruction, prompting extensions to dynamic scenarios.
While 4D-GS (Wu et al. 2024) employs multi-resolution
HexPlanes with MLPs for deformation modeling, it inherits
the fundamental limitation of plane-based methods: the low-
rank assumption leads to feature collisions and rendering ar-
tifacts in complex motions. Neural deformation fields (Yang
et al. 2024; Huang et al. 2024) address this through MLPs,
but often produce over-smoothed results and struggle with
high-frequency details due to insufficient inductive biases.
Direct optimization of 4D Gaussians (Yang et al. 2023; Duan
et al. 2024) offers greater flexibility but introduces optimiza-
tion challenges including floating artifacts and requires ex-
tensive training with additional regularizers. Grid4D (Xu
et al. 2024) has achieved impressive performance through
combining triplane and Hash-coding while it often lacks
smoothness and works without an explicit method for mod-
eling dynamic processes. While SplineGS (Park et al. 2024)
proposes a pipeline that combine 3DGS and spline func-
tions, however, it requires massive priors such as 2D trajac-
tory and depth estimation to maintain performance. These
limitations collectively highlight the need for a represen-
tation that balances expressiveness with efficient optimiza-
tion for dynamic 3DGS, particularly in handling complex
motions while preserving fine details. Our work addresses
these limitations through a novel Laplacian motion represen-
tation that jointly optimizes for physical plausibility, compu-
tational efficiency, and multi-scale temporal fidelity.
Methodology
In this section, we present our methodology aimed at ad-
dressing the challenges of modeling 4D dynamic scenes
with high-fidelity spatial details and complex temporal vari-
ations. The key innovation lies in a hybrid explicit-implicit
representation that combines multi-scale Hash encoding
with spectral decomposition to capture spatial features and
adaptive temporal dynamics. This framework is structured
into three main components: spectral-aware Laplacian en-
coding module, enhanced Gaussian dynamic attribute and
adaptive Gaussian split strategy. This hybrid approach is
designed to enhance the representation of motion dynam-
ics while maintaining physical consistency across spatial
and temporal domains, significantly outperforming existing
methods in handling complex motion patterns. The overall
pipeline is shown in Fig. 1.
Spectral-Aware Laplacian Encoding Module
This section focuses on the challenge of effectively encod-
ing spatial and temporal information to capture the dynamics
of motion. We employ a spectral-aware Laplacian encoding
module that decomposes the frequency motion trajectories
to accommodate the complexities of 4D spacetime.
Multi-Scale Hash Encoding
To efficiently encode 4D
spacetime information while preserving both spatial and
temporal details, we employ a multi-scale hash encod-
ing strategy that extends traditional methods. Inspired by
Grid4D (Xu et al. 2024), we extend InstantNGP’s Hash
encoding (M¨uller et al. 2022) to 4D spacetime (x, y, z, t)
through anisotropic multi-resolution decomposition.
Hl = {Hl
xyz, Hl
xyt, Hl
yzt, Hl
xzt},
l ∈{1, ..., L}
(1)
Each level l maintains dimension-specific resolutions
computed via geometric progression.
Laplacian-Based Motion Prediction
In the realm of dy-
namic scene reconstruction, accurately predicting motion
dynamics is paramount. Traditional methods often rely on
MLP or linear interpolation, which fails to capture the com-
plex periodic and aperiodic motions present in real-world
dynamic scenes. To overcome these limitations, we pro-
pose a novel hybrid explicit-implicit Laplacian-based mo-

<!-- page 3 -->
Figure 1: Framework of our method. We begin with a multi-scale Hash encoder to extract spatio-temporal features of Gaus-
sians. Then a hybrid laplacian-based dynamic module is designed for adaptive frequency analysis. We design an appearance
adjustment module which combines the Gaussian dynamic attribute with the spatio-temporal feature by attention mechanism.
The whole pipeline benefits from the adaptive dynamic Gaussian split strategy for better performance and efficiency.
tion representation that combines the expressiveness of spec-
tral analysis learnable neural components, allowing for the
effective capture of both low and high-frequency motion dy-
namics. The foundation of our approach lies in the Laplacian
series decomposition of time series:
L(t) =
K
X
k=−K
ck · e2πikt/T ,
(2)
where L(t) represents the Laplacian motion field at time t,
with ck denoting coefficients. Through Euler’s formula, we
substitute this representation into our motion prediction:
L(t) =
K
X
k=−K
ck · cos
2πkt
T

+ i
K
X
k=−K
ck · sin
2πkt
T

.
(3)
This transformation naturally handles periodic motions
common in dynamic scenes and the frequency components
provide interpretable control over motion characteristics. To
further enhance our motion representation, we extend the
equation to a simple formulation:
L(t) =
K−1
X
k=0
[αk cos(2πkt) + βk sin(2πkt)] .
(4)
Here, coefficients (αk, βk) are learnable parameters. This
design aims to automatically adapt to scene-specific motion
frequencies while maintaining end-to-end differentiability.
The Laplacian-based decoding allows our model to learn ap-
propriate frequency compositions directly from data, elimi-
nating the need for manual frequency band selection.
The orthogonal basis properties enable stable gradi-
ent computation during optimization. The incorporation of
learnable frequencies fk through gradient enhances the abil-
ity to capture different frequency motion components:
∂L
∂fk
= 1
σ2
k
X
t
 ∂L
∂L(t) · t · [−αk sin(2πfkt) + βk cos(2πfkt)]

,
(5)
where σk denotes temporal variance. This mechanism auto-
matically balances frequency preservation with motion sta-
bility. Then we introduce an attention mechanism which
combines Laplacian features with Hash spatial features Hs:
AL(t) = L(t) · MLP(Hs).
(6)
Multi-Scale Laplacian Pyramid Supervision
To enforce
consistency across different frequency bands and spatial
scales, we introduce a multi-scale supervision strategy that
enforces consistency across frequency bands, enhancing the
model’s ability to better detail preservation. We supervise
reconstruction using Laplacian pyramid decomposition:

<!-- page 4 -->
Llap =
L
X
l=1
λl∥Ll(Irender) −Ll(Igt)∥1,
(7)
where λl decreasing exponentially to emphasize finer de-
tails. This loss function encourages the model to focus on
both coarse and fine features, ensuring a comprehensive un-
derstanding of motion dynamics.
Enhanced Gaussian Dynamics Attribute with
Adaptive Regularization
To effectively model the dynamic variations inherent in
complex scenes, we augment the standard 3D Gaussian
Splatting representation. Specifically, we associate each 3D
Gaussian Gi with a learnable dynamics attribute, denoted as
di ∈RDd, where Dd represents the dimensionality of this
attribute space. The dynamics attribute di is introduced to
explicitly encapsulate these latent per-Gaussian temporal or
conditional variations, providing a dedicated representation
for dynamic properties.
To further improve the modeling of dynamic scene
changes, we introduce a fusion mechanism that concatenates
the original dynamic attribute vector di with the Hash tem-
poral feature Ht, forming an augmented feature representa-
tion:
˜
Ht = Concatenate(di, Ht).
(8)
This concatenation provides a straightforward yet effec-
tive means of integrating scene-specific temporal informa-
tion with the Gaussian’s intrinsic dynamic attributes, en-
abling the model to leverage both sources for more accurate
deformation prediction. To effectively combine spatial and
temporal information, we introduce an attention mechanism
to aggregate spatio-temporal features through:
Ah(t) = ˜
Ht · Hs.
(9)
Adaptive Dynamic Regularization
To ensure that our
method better model dynamic changes, we implement a
selective regularization mechanism that targets only those
Gaussians exhibiting ”abnormally large” or ”highly dy-
namic” changes. These gaussians are referred to as ”out-
liers”, which need to be increase their gradients and thus
promote their deformation or dynamic transformations.
Instead of using fixed thresholds or applying a regular-
ization on all gaussians, our method employs a data-driven,
adaptive dynamic selection scheme. Specifically, for each
Gaussian, we compute the Euclidean distance between its
dynamic attribute di and a reference mean dynamic attribute
¯d, as well as the associated standard deviation:
di = ∥di −¯d∥2.
(10)
Let µdist and σdist denote the mean and standard deviation
of all d disti across the Gaussian set. We then generate a
mask to identify those points that are significantly deviating
from the typical embedding variation:
maski = di > µdist + σdist.
(11)
Only the Gaussians satisfying this outlier criterion—i.e.,
those with maski = 1—are subjected to the additional regu-
larization loss:
Ldy = 1
N
N
X
i=1
maski · ∥di −¯d∥2
2.
(12)
The purpose of this selective regularization is to inten-
sify the gradients for Gaussians exhibiting large changes,
thereby explicitly promoting their deformation and densi-
fication. By employing this dynamic regularization mech-
anism, the model adaptively concentrates regularization ef-
forts on the most informative and dynamically relevant re-
gions, effectively enhancing the capacity to model com-
plex scene dynamics without imposing uniform constraints
across all Gaussians.
In
addition,
we
use
Normalized
cross-correlation
(NCC) (Yoo and Han 2009) based loss function LNCC to
evaluate the similarity between two images within local re-
gions while maintaining invariance to brightness variations
to enhance the alignment accuracy. The total loss L used
for training is composed of four distinct loss terms, each
weighted by a corresponding hyperparameter λ to control
its relative contribution. The total loss is formulated as:
L = Lorig + λNCCLNCC + λlapLlap + λdyLdy,
(13)
where Lorig denotes original loss function of 3DGS and
consists of L1 and Structural Similarity Index Measure
(SSIM) loss functions (Wang et al. 2004) LSSIM.
Adaptive Gaussian Split Strategy
In this section, we address the challenge of optimizing Gaus-
sian representations of motion dynamics. Our approach uti-
lizes the analysis about the structure of each Gaussian to
adaptively refine Gaussian parameters based on local neigh-
borhood information, enhancing the model’s ability to cap-
ture complex motion patterns.
KDTree-Based Primitive Analysis
To maintain spatial
coherence and prevent overfitting, we analyze Gaussian
primitives through their neighborhood relationships. For
each Gaussian Gi, we find K nearest neighbors {Gj}K
j=1
based on Euclidean distance:
edij =
q
(µi −µj)T (µi −µj).
(14)
By examining the size and anisotropy of each Gaussian
(Xie, Zong et al. 2024), we can determine which Gaussians
exhibit significant differences in their motion characteristics.
Covariance differences are computed through L2 norm:
∆Σij = ∥Σi −Σj∥2.
(15)
This adaptive approach ensures that our models remain re-
sponsive to local variations in motion dynamics. By focusing
on Gaussians with notable differences in size and anisotropy,
we can selectively choose which Gaussian to split, thereby
enhancing the model’s ability to represent complex motion
patterns without introducing unnecessary complexity.

<!-- page 5 -->
KL-Divergence Guided Adaptation
In some cases, we
observed that the KDTree-based partitioning method can not
accurately identify the dynamic Gaussian, which leads to the
instability to capture dynamic motions. This is because the
strategy of splitting Gaussian based on hard threshold will
stop deriving when the shape and size of Gaussian in the
neighborhood are similar. However, it is possible that some
Gaussian in the neighborhood still contribute to the dynamic
modeling. To further refine the Gaussian split process, we
compute the KL-divergence between the neighbor Gaussian
distribution P and a uniform distribution Q:
DKL(P ∥Q) =
K
X
k=1
P(k) log P(k)
Q(k)
(16)
The adaptive splitting threshold τ becomes:
τ = ∆Σ + DKL · τbase
(17)
where τbase is a hyperparameter. This mechanism allows
for dynamic adjustments to the model’s complexity based
on the observed motion patterns. Through adaptive dynamic
Gaussian optimization strategy, we can determine when
a Gaussian should be split more effectively, ensuring the
model captures the nuances of motion dynamics while main-
taining computational efficiency. This strategy enhances the
model’s robustness against overfitting by focusing on the
most relevant Gaussian structures.
Experiment
Experiment Setup
We evaluate our proposed method using three widely rec-
ognized datasets, comprising two real-world datasets and
one synthetic dataset. The Neu3D
(Li, Slavcheva et al.
2022) dataset is a real-world collection that features multi-
ple static cameras and includes between 18 to 21 multi-view
videos. We generate 300 frames for each video and initial
point clouds for each scene following 4DGaussians (Wu
et al. 2024). HyperNeRF (Park et al. 2021b) is a real-world
dataset that captures continuous views with intricate topo-
logical variations at each timestamp within a dynamic scene.
In our experiment, we utilized the “vrig” subset, which was
captured using stereo cameras, training the model with data
from one camera while validating it with data from the other.
The D-NeRF (Pumarola et al. 2021) dataset serves as a syn-
thetic dataset tailored for monocular scenes, with each scene
comprising between 50 to 200 frames. Due to discrepancies
between the training and testing scenarios presented in the
Lego subset of the D-NeRF (Pumarola et al. 2021) dataset,
we excluded it from our experimental analysis.
Comparisons
On the Neu3D dataset, our approach demonstrates excep-
tional proficiency as shown in Tab. 3. The primary chal-
lenge here lies in accurately modeling intricate, often non-
rigid, temporal dynamics while simultaneously reconstruct-
ing high-fidelity static scene geometry from these fixed per-
spectives. Our method excels in generating temporally co-
herent motion representations and preserving sharp geomet-
ric details throughout the sequences, effectively disentan-
gling dynamic elements from the static background. In con-
trast, competing methods frequently struggle to maintain
long-term temporal consistency across the multiple views,
often exhibiting noticeable motion blur, particularly during
complex actions or over extended durations.
The HyperNeRF “vrig” subset introduces a distinct set of
demanding conditions. This dataset tests the model’s abil-
ity to handle complex motion while maintaining consis-
tency across stereo viewpoints and adapt to evolving scene
topology. Our method showcases remarkable resilience and
adaptability in handling these extreme deformations and ef-
fectively leverages the stereo information, generalizing ro-
bustly across the viewpoints even when trained on one and
validated on the other as shown in Tab. 1 and Fig. 3. It
consistently reconstructs intricate topological changes with
greater accuracy and fewer visual artifacts or geometric dis-
tortions compared to existing approaches.
Furthermore, evaluation on the synthetic D-NeRF dataset
underscores our method’s inherent strength in inferring co-
herent 3D structure and plausible motion from limited input
as shown in Tab. 2 and Fig. 2. Reconstructing dynamic 3D
geometry from a single, potentially moving, camera view-
point over time presents profound depth ambiguities and ne-
cessitates strong priors and temporal reasoning. Despite this
inherent ill-posedness and the scarcity of explicit geometric
cues, our approach generates remarkably temporally stable
and geometrically plausible reconstructions. Consequently,
it significantly outperforms baseline methods, which, under
these monocular constraints, often exhibit pronounced depth
inaccuracies that betray instabilities in their representation.
Across this diverse range of evaluated datasets, our
method consistently achieves a marked superiority in per-
formance. This advantage is evident in both the final recon-
struction fidelity and the accurate, coherent capture of dy-
namic motion, ranging from subtle deformations to large-
scale topological changes. This consistent success across
varied and demanding conditions robustly validates the ef-
fectiveness, versatility, and broad applicability of our pro-
posed framework for dynamic scene reconstruction.
Ablation Study and Analysis
To validate the effectiveness of each component within our
framework, we conduct comprehensive ablation studies to
validate the necessity of each component as shown in Tab.
4, Tab. 5 and Fig. 4.
Effect of Laplacian-Based Motion Prediction
Replac-
ing this module with a defrom MLP leads to poorer per-
formance, especially in scenes with diverse motion patterns
or objects of varying sizes evolving over time. Dynamic
scenes inherently possess variations across multiple spatial
and temporal scales. This module is designed to capture
these hierarchies effectively. It allows the model to represent
fine details of motion trajectories while enabling the mod-
eling of slow, gradual changes. By processing information
hierarchically, it ensures consistent and accurate representa-
tion of scene dynamics across different frequency.

<!-- page 6 -->
Table 1: Quantitative comparison to previous methods on HyperNeRF (Park et al. 2021b) dataset. The higher PSNR(↑)
and higher SSIM(↑) denote better rendering quality. The color of each cell shows the best and the second best .
Scene
broom2
vrig-3dprinter
vrig-chicken
vrig-peel-banana
Aveage
Method
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
SSIM↑
PSNR↑
LPIPS↓
HyperNeRF (Park et al. 2021b)
0.210
19.51
—
0.635
20.04
—
0.828
27.46
—
0.719
22.15
—
0.598
22.29
—
D3DGS (Yang et al. 2024)
0.269
19.99
0.700
0.656
20.71
0.277
0.640
22.77
0.363
0.853
25.95
0.155
0.605
22.35
0.374
MotionGS (Zhu et al. 2024)
0.380
22.30
—
0.710
21.80
—
0.790
26.80
—
0.690
28.20
—
0.643
24.78
—
MoDec-GS (Kwak et al. 2025)
0.303
21.04
0.666
0.706
22.00
0.265
0.834
28.77
0.197
0.873
28.25
0.173
0.679
25.02
0.325
4DGaussians (Wu et al. 2024)
0.366
22.01
0.557
0.705
21.98
0.327
0.806
28.49
0.297
0.847
27.73
0.204
0.681
25.05
0.346
ED3DGS (Bae et al. 2024)
0.371
21.84
0.531
0.715
22.34
0.294
0.836
28.75
0.185
0.867
28.80
0.178
0.697
25.43
0.297
Grid4D (Xu et al. 2024)
0.414
21.78
0.423
0.723
22.36
0.245
0.848
29.27
0.199
0.875
28.44
0.167
0.715
25.46
0.259
Ours
0.422
22.36
0.413
0.724
22.56
0.264
0.858
29.57
0.166
0.876
28.81
0.169
0.720
25.83
0.253
Figure 2: Comparison Results. Visual differences are highlighted with red insets for better clarity. Our approach consistently
outperforms other models on D-NeRF (Pumarola et al. 2021) dataset, demonstrating clear advantages in challenging scenarios
such as thin geometries and fine-scale details. Best viewed in color.
Table 2: Quantitative comparison to previous methods on
D-NeRF (Pumarola et al. 2021) dataset. The color of each
cell shows the best and the second best . More detail re-
sults can be found in supplementary material.
Method
SSIM↑
PSNR↑
LPIPS↓
3DGS (Kerbl et al. 2023)
0.930
23.40
0.077
K-Planes (Fridovich-Keil et al. 2023)
0.970
31.41
0.047
HexPlane (Cao and Johnson 2023)
0.972
31.92
0.038
4DGaussians (Wu et al. 2024)
0.985
35.32
0.021
D3DGS (Yang et al. 2024)
0.991
40.08
0.013
SC-GS (Huang et al. 2024)
0.993
41.66
0.009
Grid4D (Xu et al. 2024)
0.994
41.99
0.008
Ours
0.994
42.17
0.007
Effect of Adaptive Gaussian Split Strategy
Removing
this component and reverting to original 3DGS split strategy
results in a drop in reconstruction. Crucially, this component
offers a dual benefit. Firstly, it enhances reconstruction qual-
ity by intelligently allocating Gaussian primitives. It adap-
tively densifies regions with high dynamics while pruning
redundant or insignificant Gaussians. This leads to a more
accurate representation of the scene and results in a more
compact set of Gaussians compared to non-adaptive meth-
ods while achieving similar quality as shown in Tab. 4.
Table 3: Quantitative comparison to previous methods on
Neu3D (Li, Slavcheva et al. 2022) dataset. Color of each
cell shows the best and the second best . We show the av-
erage results of all scenes. More detail results can be found
in supplementary material.
Method
SSIM↑
PSNR↑
LPIPS↓
4DGaussians (Wu et al. 2024)
0.935
30.36
0.152
Grid4D (Xu et al. 2024)
0.934
30.50
0.147
Spacetime (Li et al. 2024)
0.944
31.46
0.142
ED3DGS (Bae et al. 2024)
0.943
31.92
0.139
Ours
0.944
32.12
0.134
Effect of Laplacian Pyramid Loss
When this loss is re-
moved on the rendered image, we observe a noticeable
degradation in reconstruction quality. The Laplacian pyra-
mid loss decomposes the reconstruction error across multi-
ple frequency bands by comparing the Laplacian pyramids
of the rendered and ground truth images. This loss function
proves essential because it enforces structural consistency
across different scales, effectively preserving fine details that
would otherwise be lost in single-scale supervision.
Effect of Gaussian dynamics attribute
Compared with
our full approach, removing the Gaussian dynamics attribute
leads to poorer performance. This performance difference

<!-- page 7 -->
Figure 3: Comparison Results. Our approach consistently outperforms other models on HyperNeRF (Park et al. 2021b) dataset,
demonstrating clear advantages in challenging scenarios. Best viewed in color.
Table 4: Ablation Results on Neu3D (Li, Slavcheva et al.
2022) dataset. The Adaptive Gaussian Split Strategy helps
reduce the number of Gaussians.
Method
Gaussian Counts
w/o adaptive split strategy
589k
Ours
433k
Table 5: Ablation evaluation on Neu3D (Li, Slavcheva
et al. 2022) dataset. The color of each cell shows the best .
Method
SSIM↑
PSNR↑
LPIPS↓
w/o Laplacian module
0.938
31.64
0.149
w/o dynamic attribute
0.938
31.72
0.147
w/o adaptive split strategy
0.943
31.96
0.133
w/o Llap
0.939
31.70
0.148
Ours
0.944
32.12
0.134
underscores the importance of embedding dedicated dy-
namic attributes within the Gaussians themselves. By incor-
porating these dynamic attribute, our method allows each
Gaussian to better adapt its shape and orientation according
to its specific local dynamics, effectively capturing details
and mitigating the feature collision issues inherent in rely-
ing solely on lower-rank spatio-temporal grids.
Figure 4: Ablation Results. Replacing Laplacian-based mo-
tion prediction leads to poorer performance.
Conclusion
In this paper, we present a novel approach for dynamic
3DGS that addresses the challenges of anisotropic spatio-
temporal sampling through a hybrid explicit-implicit encod-
ing framework. We introduced three key innovations: Firstly,
a hybrid motion representation combining multi-scale Hash
encoding with a Laplacian-based dynamic module, effec-
tively decoupling different motion frequencies from com-
plex deformation details. Secondly, an enhanced Gaussian
dynamics attribute that compensates for highly dynamic ar-
eas induced by geometric deformation. Thirdly, an adap-
tive Gaussian split strategy guided by KDTree-based anal-
ysis, which automatically adjusts dynamic primitive density
and anisotropy. This work advance the state-of-the-art in dy-
namic scene modeling by bridging the gap between explicit
representations and spectral analysis, with potential applica-
tions in VR/AR and scene reconstruction.

<!-- page 8 -->
References
Bae, J.; Kim, S.; Yun, Y.; Lee, H.; Bang, G.; and Uh, Y. 2024.
Per-gaussian embedding-based deformation for deformable
3d gaussian splatting. In European Conference on Computer
Vision, 321–335. Springer.
Barron, J. T.; Mildenhall, B.; Tancik, M.; Hedman, P.;
Martin-Brualla, R.; and Srinivasan, P. P. 2021. Mip-nerf:
A multiscale representation for anti-aliasing neural radiance
fields. In Proceedings of the IEEE/CVF international con-
ference on computer vision, 5855–5864.
Barron, J. T.; Mildenhall, B.; et al. 2023. Zip-nerf: Anti-
aliased grid-based neural radiance fields. In Proceedings of
the IEEE/CVF International Conference on Computer Vi-
sion, 19697–19705.
Cai, H.; Feng, W.; Feng, X.; Wang, Y.; and Zhang, J.
2022. Neural surface reconstruction of dynamic scenes with
monocular rgb-d camera. Advances in Neural Information
Processing Systems, 35: 967–981.
Cao, A.; and Johnson, J. 2023. Hexplane: A fast represen-
tation for dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
130–141.
Chan, E. R.; Lin, C. Z.; Chan, M. A.; Nagano, K.; Pan, B.;
De Mello, S.; Gallo, O.; Guibas, L. J.; Tremblay, J.; Khamis,
S.; et al. 2022. Efficient geometry-aware 3d generative ad-
versarial networks. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, 16123–
16133.
Choe, J.; Choy, C.; Park, J.; Kweon, I. S.; and Anandku-
mar, A. 2023.
Spacetime surface regularization for neu-
ral dynamic scene reconstruction.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
17871–17881.
Du, Y.; Zhang, Y.; Yu, H.-X.; Tenenbaum, J. B.; and Wu, J.
2021. Neural radiance flow for 4d view synthesis and video
processing.
In 2021 IEEE/CVF International Conference
on Computer Vision (ICCV), 14304–14314. IEEE Computer
Society.
Duan, Y.; Wei, F.; Dai, Q.; He, Y.; Chen, W.; and Chen, B.
2024. 4d gaussian splatting: Towards efficient novel view
synthesis for dynamic scenes. arXiv e-prints, arXiv–2402.
Duisterhof, B. P.; Mandi, Z.; Yao, Y.; Liu, J.-W.; Shou,
M. Z.; Song, S.; and Ichnowski, J. 2023.
Md-splatting:
Learning metric deformation from 4d gaussians in highly
deformable scenes. arXiv preprint arXiv:2312.00583, 2(3).
Fang, J.; Yi, T.; Wang, X.; Xie, L.; Zhang, X.; Liu, W.;
Nießner, M.; and Tian, Q. 2022.
Fast dynamic radiance
fields with time-aware neural voxels. In SIGGRAPH Asia
2022 Conference Papers, 1–9.
Fridovich-Keil, S.; Meanti, G.; Warburg, F. R.; Recht, B.;
and Kanazawa, A. 2023. K-planes: Explicit radiance fields
in space, time, and appearance.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 12479–12488.
Gao, C.; Saraf, A.; Kopf, J.; and Huang, J.-B. 2021. Dy-
namic view synthesis from dynamic monocular video. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, 5712–5721.
Huang, Y.-H.; Sun, Y.-T.; Yang, Z.; Lyu, X.; Cao, Y.-P.; and
Qi, X. 2024. Sc-gs: Sparse-controlled gaussian splatting for
editable dynamic scenes. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
4220–4230.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3D Gaussian splatting for real-time radiance field ren-
dering. ACM Trans. Graph., 42(4): 139–1.
Kratimenos, A.; Lei, J.; and Daniilidis, K. 2024. Dynmf:
Neural motion factorization for real-time dynamic view syn-
thesis with 3d gaussian splatting. In European Conference
on Computer Vision, 252–269. Springer.
Kwak, S.; Kim, J.; Jeong, J. Y.; Cheong, W.-S.; Oh, J.; and
Kim, M. 2025.
MoDec-GS: Global-to-Local Motion De-
composition and Temporal Interval Adjustment for Com-
pact Dynamic 3D Gaussian Splatting.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR).
Li, T.; Slavcheva, M.; et al. 2022. Neural 3d video synthesis
from multi-view video. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
5521–5531.
Li, Z.; Chen, Z.; Li, Z.; and Xu, Y. 2024. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 8508–8520.
Liang, Y.; Khan, N.; Li, Z.; Nguyen-Phuoc, T.; Lanman, D.;
Tompkin, J.; and Xiao, L. 2023. Gaufre: Gaussian deforma-
tion fields for real-time dynamic novel view synthesis. arXiv
preprint arXiv:2312.11458.
Liang, Y.; Laidlaw, E.; et al. 2023.
Semantic attention
flow fields for monocular dynamic scene decomposition. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, 21797–21806.
Lin, Y.; Dai, Z.; Zhu, S.; and Yao, Y. 2024. Gaussian-flow:
4d reconstruction with dynamic 3d gaussian particle. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 21136–21145.
Liu, Y.-L.; Gao, C.; Meuleman, A.; Tseng, H.-Y.; Saraf,
A.; Kim, C.; Chuang, Y.-Y.; Kopf, J.; and Huang, J.-B.
2023. Robust dynamic radiance fields. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 13–23.
Lu, Z.; Guo, X.; Hui, L.; Chen, T.; Yang, M.; Tang, X.; Zhu,
F.; and Dai, Y. 2024. 3d geometry-aware deformable gaus-
sian splatting for dynamic view synthesis. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 8900–8910.
Luiten, J.; Kopanas, G.; Leibe, B.; and Ramanan, D. 2024.
Dynamic 3d gaussians: Tracking by persistent dynamic view
synthesis. In 2024 International Conference on 3D Vision
(3DV), 800–809. IEEE.
Martin-Brualla, R.; Radwan, N.; Sajjadi, M. S.; Barron, J. T.;
Dosovitskiy, A.; and Duckworth, D. 2021. Nerf in the wild:

<!-- page 9 -->
Neural radiance fields for unconstrained photo collections.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 7210–7219.
Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.;
Ramamoorthi, R.; and Ng, R. 2021.
Nerf: Representing
scenes as neural radiance fields for view synthesis. Com-
munications of the ACM, 65(1): 99–106.
M¨uller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In-
stant neural graphics primitives with a multiresolution hash
encoding. ACM transactions on graphics (TOG), 41(4): 1–
15.
Park, J.; Bui, M.-Q. V.; Bello, J. L. G.; Moon, J.; Oh, J.;
and Kim, M. 2024.
SplineGS: Robust Motion-Adaptive
Spline for Real-Time Dynamic 3D Gaussians from Monoc-
ular Video. arXiv preprint arXiv:2412.09982.
Park, K.; Sinha, U.; Barron, J. T.; Bouaziz, S.; Goldman,
D. B.; Seitz, S. M.; and Martin-Brualla, R. 2021a.
Ner-
fies: Deformable neural radiance fields. In Proceedings of
the IEEE/CVF international conference on computer vision,
5865–5874.
Park, K.; Sinha, U.; Hedman, P.; Barron, J. T.; Bouaziz,
S.; Goldman, D. B.; Martin-Brualla, R.; and Seitz, S. M.
2021b. Hypernerf: A higher-dimensional representation for
topologically varying neural radiance fields. arXiv preprint
arXiv:2106.13228.
Pumarola, A.; Corona, E.; Pons-Moll, G.; and Moreno-
Noguer, F. 2021. D-nerf: Neural radiance fields for dynamic
scenes. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, 10318–10327.
Shao, R.; Zheng, Z.; Tu, H.; Liu, B.; Zhang, H.; and Liu,
Y. 2023. Tensor4d: Efficient neural 4d decomposition for
high-fidelity dynamic reconstruction and rendering. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 16632–16642.
Tancik, M.; Casser, V.; Yan, X.; Pradhan, S.; Mildenhall, B.;
Srinivasan, P. P.; Barron, J. T.; and Kretzschmar, H. 2022.
Block-nerf: Scalable large scene neural view synthesis. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, 8248–8258.
Tretschk, E.; Tewari, A.; et al. 2021. Non-rigid neural ra-
diance fields: Reconstruction and novel view synthesis of a
dynamic scene from monocular video. In Proceedings of
the IEEE/CVF International Conference on Computer Vi-
sion, 12959–12970.
Wang, C.; MacDonald, L. E.; Jeni, L. A.; and Lucey, S.
2023. Flow supervision for deformable nerf. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 21128–21137.
Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P.
2004.
Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image process-
ing, 13(4): 600–612.
Wu, G.; Yi, T.; Fang, J.; Xie, L.; Zhang, X.; Wei, W.; Liu,
W.; Tian, Q.; and Wang, X. 2024.
4d gaussian splatting
for real-time dynamic scene rendering. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, 20310–20320.
Xie, T.; Zong, Z.; et al. 2024.
Physgaussian: Physics-
integrated 3d gaussians for generative dynamics.
In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 4389–4398.
Xu, J.; Fan, Z.; Yang, J.; and Xie, J. 2024. Grid4D: 4D De-
composed Hash Encoding for High-Fidelity Dynamic Gaus-
sian Splatting. arXiv preprint arXiv:2410.20815.
Yang, G.; Vo, M.; et al. 2022. Banmo: Building animatable
3d neural models from many casual videos. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, 2863–2873.
Yang, Z.; Gao, X.; Zhou, W.; Jiao, S.; Zhang, Y.; and
Jin, X. 2024.
Deformable 3d gaussians for high-fidelity
monocular dynamic scene reconstruction. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, 20331–20341.
Yang, Z.; Yang, H.; Pan, Z.; and Zhang, L. 2023.
Real-time photorealistic dynamic scene representation and
rendering with 4d gaussian splatting.
arXiv preprint
arXiv:2310.10642.
Yoo, J.-C.; and Han, T. H. 2009.
Fast normalized cross-
correlation.
Circuits, systems and signal processing, 28:
819–843.
Yu, H.; Julin, J.; Milacski, Z. ´A.; Niinuma, K.; and Jeni,
L. A. 2024. Cogs: Controllable gaussian splatting. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 21624–21633.
Zhu, R.; Liang, Y.; Chang, H.; Deng, J.; Lu, J.; Yang, W.;
Zhang, T.; and Zhang, Y. 2024. Motiongs: Exploring ex-
plicit motion guidance for deformable 3d gaussian splat-
ting. Advances in Neural Information Processing Systems,
37: 101790–101817.
