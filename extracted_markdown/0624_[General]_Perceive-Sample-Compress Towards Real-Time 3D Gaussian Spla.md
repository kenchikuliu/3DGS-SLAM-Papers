<!-- page 1 -->
Perceive-Sample-Compress: Towards Real-Time 3D Gaussian Splatting
Zijian Wang * 1, Beizhen Zhao * 1, Hao Wang † 1
1 The Hong Kong University of Science and Technology (Guangzhou)
zwang886@connect.hkust-gz.edu.cn, bzhao610@connect.hkust-gz.edu.cn, haowang@hkust-gz.edu.cn
Abstract
Recent advances in 3D Gaussian Splatting (3DGS) have
demonstrated remarkable capabilities in real-time and pho-
torealistic novel view synthesis. However, traditional 3DGS
representations often struggle with large-scale scene man-
agement and efficient storage, particularly when dealing
with complex environments or limited computational re-
sources. To address these limitations, we introduce a novel
perceive-sample-compress framework for 3D Gaussian Splat-
ting. Specifically, we propose a scene perception compen-
sation algorithm that intelligently refines Gaussian parame-
ters at each level. This algorithm intelligently prioritizes vi-
sual importance for higher fidelity rendering in critical areas,
while optimizing resource usage and improving overall visi-
ble quality. Furthermore, we propose a pyramid sampling rep-
resentation to manage Gaussian primitives across hierarchical
levels. Finally, to facilitate efficient storage of proposed hier-
archical pyramid representations, we develop a Generalized
Gaussian Mixed model compression algorithm to achieve sig-
nificant compression ratios without sacrificing visual fidelity.
The extensive experiments demonstrate that our method sig-
nificantly improves memory efficiency and high visual qual-
ity while maintaining real-time rendering speed.
Introduction
3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) has rev-
olutionized the field of novel view synthesis (NVS) (Avi-
dan and Shashua 1997; Watson et al. 2022; Choi et al.
2019; Riegler and Koltun 2020; Dalal et al. 2024), offer-
ing a compelling solution to traditional volumetric render-
ing and mesh-based methods (Yariv et al. 2023; Lombardi
et al. 2019; Penner and Zhang 2017; Wizadwongsa et al.
2021; Zhou et al. 2018). By representing scenes as a set of
anisotropic Gaussians, 3DGS achieves remarkable photore-
alism and real-time rendering speeds, quickly becoming a
cornerstone technology for 3D reconstruction.
Despite its impressive capabilities, a key challenge lies
in managing the computational complexity and storage
requirements when scenes involve millions of Gaussian
points. The direct application of 3DGS to large-scale or
memory-constrained scenarios lacks efficient management
representation (Wang and Xu 2024; Liu et al. 2024b; Chen
*These authors contributed equally to this work.
†Corresponding author.
et al. 2024a), resulting in inconsistencies and degradation
when rendering scenes at different scales. When scenes in-
volve millions of Gaussian primitives, the dense collection
of Gaussians can lead to excessive memory consumption
and computational demands, hindering its scalability.
Existing methods struggle to effectively address the chal-
lenges of storage consumption and hierarchical scalabil-
ity simultaneously. Some attempts to address the stor-
age consumption of Gaussian Splatting, such as Scaffold-
GS (Lu et al. 2024), HAC++ (Chen et al. 2025) and
Context-GS (Wang et al. 2024), lack explicit Level-of-Detail
(LOD) (David et al. 2003) mechanisms. This absence re-
sults in inefficient storage and rendering, as the entire scene
representation must be processed regardless of the viewer’s
proximity or the required level of detail. Other approaches,
like Octree-GS (Ren et al. 2024), while introducing hierar-
chical structures, can be overly dependent on camera dis-
tance to optimize the whole structure and may lead to sub-
optimal performance in scenes with complex geometric dis-
tributions, as they might over-prune crucial details and fail
to capture fine-grained information.
To overcome these limitations, we introduce a novel
perceive-sample-compress framework for 3D Gaussian
Splatting. We propose a scene perception compensation al-
gorithm that refines the parameters at each level. Unlike ex-
isting methods that rely solely on camera distance, our al-
gorithm dynamically prioritizes optimization efforts based
on camera coverage area and depth of field. This algorithm
intelligently prioritizes visual importance for higher fidelity
rendering in critical areas, while optimizing resource usage
and improving overall visible quality.
Then, based on the levels of perception areas, we propose
a pyramid sampling representation inspired by the Lapla-
cian pyramid and voxelization. This approach allows us to
group and represent Gaussians at multiple scales for each
level of the pyramid. By voxelizing the scene and applying
pyramid to aggregate Gaussian properties within these vox-
els, we establish a hierarchical multi-scale scene structure.
This pyramid representation significantly enhances memory
efficiency and enables adaptive rendering strategies.
Finally, we propose a generalized Gaussian mixed model
compression algorithm based on the hierarchical pyramid
structure. By analyzing the distribution of Gaussian at-
tributes across hierarchical levels, we can effectively encode
arXiv:2508.04965v1  [cs.GR]  7 Aug 2025

<!-- page 2 -->
redundant information and reduce the model size, making
our framework more practical for storage and transmission.
In summary, our work makes three key contributions:
• We introduce a scene perception compensation algorithm
that intelligently refines Gaussian parameters at each
level based on camera coverage area and depth of field,
leading to improved visual quality.
• We propose a pyramid sampling function for 3D Gaus-
sian Splatting by combining Laplacian pyramid with
voxelization, enabling efficient hierarchical representa-
tion and adaptive processing of large-scale scenes.
• We develop a compression algorithm based on gener-
alized Gaussian distribution for efficient storage of our
Gaussian representations, significantly reducing memory
consumption and facilitating broader applicability.
Related Work
Level-of-Detail 3DGS
Level-of-Detail (LOD) (David et al. 2003) is a classic and
critical technique in computer graphics that effectively man-
ages the rendering budget of 3D scenes to improve rendering
efficiency. Several works have explored LOD mechanisms
in Gaussian-based scene representations (Cui et al. 2024;
Seo et al. 2024; Milef et al. 2025; Ren et al. 2024; Wang
and Xu 2024; Liu et al. 2024b; Kerbl et al. 2024). For ex-
ample, LetsGo (Cui et al. 2024) employs a multi-resolution
Gaussian model and jointly optimizes multiple levels. How-
ever, the rendering quality is highly dependent on multi-
resolution point cloud inputs, which significantly increases
training overhead. CityGaussian (Liu et al. 2024b) improves
efficiency of large-scale scenes by selecting and blending
LOD levels based on distance intervals, but requires man-
ual setting of distance thresholds, which results in a lack of
robustness. While Octree-GS (Ren et al. 2024) significantly
improves rendering speed through its use of an explicit oc-
tree structure and a cumulative LOD strategy, its level selec-
tion strategy relies excessively on camera distance, making
it prone to losing critical visual details due to improper level
switching in complex and heavily occluded scenes.
Compression of 3DGS
Although 3DGS (Kerbl et al. 2023) demonstrates outstand-
ing performance in terms of rendering speed and image fi-
delity, the large number of Gaussians and their associated
parameters result in significant storage overhead. To ad-
dress this issue, researchers have proposed various com-
pression strategies. (Lee et al. 2024; Niedermayr, Stumpfeg-
ger, and Westermann 2024; Girish, Gupta, and Shrivastava
2024; Fan et al. 2024; Navaneet et al. 2023; Papantonakis
et al. 2024; Ali et al. 2024; Wang et al. 2024; Liu et al.
2024a; Chen et al. 2024b, 2025) For example, the work
in (Niedermayr, Stumpfegger, and Westermann 2024; Na-
vaneet et al. 2023) introduces codebooks to cluster Gaussian
parameters, while the studies in (Papantonakis et al. 2024;
Ali et al. 2024) systematically explore pruning-based opti-
mization methods. Building on Scaffold-GS (Lu et al. 2024),
Context-GS (Wang et al. 2024) and CompGS (Liu et al.
Figure 1: Visualization of pyramid level Compared with
Scaffold-GS, our method can achieve higher PSNR scores
with fewer Gaussian primitives.
2024a) adopt context-aware approaches that explicitly incor-
porate hierarchical relationships among anchors and Gaus-
sians to enhance representational capacity. HAC++ (Chen
et al. 2025) further introduces hashed features as priors for
entropy coding and demonstrates its effectiveness in improv-
ing compression efficiency. However, these methods heav-
ily rely on the anchor selection mechanism proposed in
Scaffold-GS, and fail to fully model the distributional differ-
ences among various Gaussian parameters during encoding
stage.
Methodology
Our proposed approach aims to address the limitations of
traditional 3DGS by introducing a robust pyramid structure
framework. This framework is built upon three intercon-
nected components: a scene perception compensation algo-
rithm that leverages camera coverage and depth of field for
improved rendering quality, a hierarchical pyramid sampling
representation with voxelization for efficient scene manage-
ment, and a compression scheme for effective model stor-
age. These components interact seamlessly, with the pyra-
mid sampling establishing the foundation for multi-scale de-
tail, the camera-driven compensation refining pyramid level
partition accuracy, and the compression scheme ensuring
scalability for large scenes. The overall goal is to create a
scalable, efficient, and high-fidelity 3D representation. This
section details the technical implementation of these com-
ponents. The overall pipeline is shown in Fig. 2.

<!-- page 3 -->
Figure 2: Framework of Pyramid-GS. We begin by sampling the input sparse point cloud. We propose a scene perception
compensation algorithm, which take the camera coverage and depth of field accounts for pyramid level partition. Then we
design a hierarchical pyramid sampling representation to set up pyramid structure. Finally, we utilize compression algorithm
for efficient storage while maintaining real-time rendering speed and photorealistic visual effects. An adaptive quantization
module (AQM) is used to generate context feature, while the generalized gaussian mixed model is used for entropy estimation.
Scene Perception Compensation
To refine the pyramid level partition and ensure perceptual
quality, we introduce a compensation algorithm that consid-
ers camera coverage and depth of field (DOF) effects. Our
algorithm aims to dynamically adjust the perceived impor-
tance of anchors based on their visibility and how they are
affected by the camera’s optical properties. We model this
by computing a visibility score for each anchor, which in-
corporating both camera coverage and depth perception.
We approximate the DOF effect by analyzing the distribu-
tion of points along the principal viewing direction from the
camera. For a set of anchors {Ak} with position pk ∈R3,
we can estimate the variance of their projected distances
from the camera. Let dk = pk −cj be the vector from
the camera to the k-th anchor. We identify the main axis of
dispersion mj for these points. The projected depth for Ak
is zk = |dk · mj|. We then compute the standard deviation
of these projected depths, where σz,j = std(zk).
A large σz,j indicates that many points are spread out in
depth, suggesting a shallow DOF or a scene with significant
depth variation. In such cases, we apply a depth compensa-
tion factor fdepth to the distance used for pyramid level par-
tition. This factor is designed such that a larger σz,j leads to
a smaller fdepth, effectively making objects appear ”closer”
in terms of pyramid priority, thus encouraging their repre-
sentation at finer levels. The formulation is as follows:
fdepth = 1 + α · max(0,
σz,j
σz,thresh
−1),
(1)
where α is a superparameter and σz,thresh is a threshold
for DOF significance. The adjusted distance d′
ij for pyra-
mid level partition for anchor Ai from camera cj becomes
d′
ij = dij · fdepth, where dij = ∥pi −cj∥. The pyramid
level Lij for Ai from camera cj is then determined based on
this adjusted distance, typically using a logarithmic scale:
Lij = log2
 
Dstd
d′
ij
!
,
(2)
where Dstd is a standard reference distance. This value is
then mapped to an integer pyramid level.
For each camera cj with center cj and intrinsic param-
eters, we first determine the visible anchors. An anchor is
considered visible to camera cj if its predicted pyramid level
Lij is smaller or equal to the current level. We then count the
number of cameras Nvis,i that render at least one anchor at
current level. The normalized coverage score for Gi is
Ci = β Nvis,i
N
.
(3)
Then we calculate the mean value of the counts of visible
anchors for each camera as the visible threshold τ. The final
visible threshold is updated through the coverage score C.
We can utilize this threshold for camera visibility control
through a mask for scene perception.
τnew = (1 + C)τold.
(4)

<!-- page 4 -->
This scheme ensures that points with larger depth varia-
tion, often corresponding to distant or complex regions, are
allocated appropriate levels to maintain visual quality. The
method enhances the initialization and refinement of pyra-
mid levels, leading to more accurate scene representations.
Pyramid Sampling Representation
To enable efficient management of Gaussian primitives, we
introduce a hierarchical representation inspired by Lapla-
cian pyramid and voxelization. Traditional methods often
lack explicit level control, leading to uniform processing
of all Gaussians, which is inefficient for large scenes. Our
approach tackles this by organizing Gaussians into a multi-
scale structure. To effectively represent the scene at multiple
levels of detail, we employ a Laplacian pyramid framework
that decomposes the scene into several resolution layers. At
each level, scene features are captured with varying degrees
of detail, enabling scalable rendering and processing.
Given an input point cloud P = {pi ∈R3}N
i=1, we con-
struct a Laplacian pyramid with L levels through iterative
voxelization. First, we recursively group voxels into smaller
blocks. At each level of the pyramid, Gaussians within a
block are voxelized into representative anchors. This aggre-
gation is performed using principles akin to the Laplacian
pyramid construction, where each level captures the high-
frequency details relative to the coarser level below:
V
′
l−1 = Rl−1 ∩P,
Rl−1 = Downsample(V
′
l, ρl−1),
(5)
where Vl denotes the voxelized points at level l with res-
olution ρl = 2−lρ0, and Rl stores residuals. Through the
downsample process, we get a simple multi-scale structure
representation. Then we execute an upsample process to
sample detail residuals of each level.
Vl = Rl ∪V
′
l,
Rl = Upsample(V
′
l−1, ρl).
(6)
The core idea is to maintain a set of Gaussians that are
adaptively represented across different spatial resolutions.
The coarsest level represents the overall scene structure,
while finer levels provide increasing detail. This allows us
to selectively process parts of the scene at an appropriate
resolution based on the viewing context. This voxel-based,
pyramidical representation significantly reduces the number
of active Gaussians and improves memory efficiency com-
pared to a flat representation. After voxelizing P into a grid
V, each voxel is parameterized by Gaussian functions pre-
dicted through a multi-layer perceptron (MLP):
Gl(x) = {F σ(xl), F µ(xl), F α(xl), F c(xl)} ,
(7)
where Gl denotes the Gaussian representation at level l. F σ,
F µ, F α and F c represent the MLP network to generate vari-
ance σl, mean µl, opacity α and color c of the gaussians.
This multi-scale decomposition allows us to represent
scene details hierarchically, such that coarse layers cap-
ture the global structure and finer layers preserve de-
tails—enabling efficient rendering and bias towards impor-
tant scene features at different pyramid levels.
Generalized Gaussian Mixed Compression
To ensure the scalability of Pyramid-GS, we propose an ef-
ficient model compression strategy based on context-aware
entropy coding, targeting anchor features across pyramid
levels. Drawing inspiration from HAC++ (Chen et al. 2025),
which employs hash-grid features to assist an Adaptive
Quantization Module (AQM) for Gaussian parameter pre-
diction, we introduce a key modification to the probability
model to better capture the statistical characteristics of our
hierarchical representation.
In image and video compression, the Generalized Gaus-
sian Distribution (GGD) effectively models transform or
prediction residuals. We adopt GGD to uniformly charac-
terize diverse parameter distributions in our framework. No-
tably, high-frequency components in the pyramid exhibit
sparsity, with distributions sharply peaked at zero and heavy
tails, aligning with the Laplace distribution (GGD with β =
1). In contrast, other parameters resemble Gaussian distribu-
tions (GGD with β = 2), as illustrated in Fig. 2. GGD offers
a statistically sound foundation, enabling the application of
tailored priors to different parameters within the framework.
Its probability density function (PDF) is defined as:
p(x|µ, α, β) =
β
2αΓ(1/β) exp
 
−
|x −µ|
α
β!
,
(8)
where µ is the location parameter, α is the scale parameter,
and β is the shape parameter that controls the distribution’s
tail behavior. Following HAC++ (Chen et al. 2025), we use
a lightweight MLP, denoted as MLPc, to predict the proba-
bility distribution parameters for a given Gaussian attribute
ˆf i from its spatial hash-grid feature f i
h. Our MLPc estimates
the location µi and scale αi parameters for the distribution.
The shape parameter βi, however, is pre-set according to
the type of parameter being compressed: for high-frequency
residual features, we set βi = 1 (Laplace), while for other
parameters, we set βi = 2 (Gaussian).
The probability of the quantized attribute ˆf i with a quan-
tization step qi is then calculated by integrating the corre-
sponding GGD’s PDF over the quantization bin:
pGGD( ˆf i) = CDF( ˆf i + qi/2 | µi, αi, βi)
−CDF( ˆf i −qi/2 | µi, αi, βi),
(9)
where CDF(x|µ, α, β) is the cumulative distribution func-
tion of the GGD, with µi, αi = MLPc(f i
h) and βi being the
pre-set value. This design allows the model to leverage the
generality of the GGD framework to impose the most fitting
inductive bias on data with different statistical characteris-
tics, thereby maximizing compression efficiency.
This compression module is applied to the attributes of
Gaussians at each level of our pyramid. The entire model
is trained end-to-end by minimizing a rate-distortion loss,
which balances rendering quality and model size:
L = Lrender + λ
 
1
N
N
X
i=1
−log2 pGGD( ˆf i) + Lhash
!
. (10)
Here, Lrender combines fidelity losses akin to Scaffold-GS,
while the second term denotes the overall rate loss, com-
prising the average entropy of primitive attributes and the

<!-- page 5 -->
hash-grid entropy. With N as the number of primitives and
λ controlling the quality–compression trade-off, this formu-
lation encourages compact yet high-fidelity representations.
Experiment
In this section, we begin by outlining the datasets used for
our evaluation and providing the specific implementation de-
tails of our experiments. Then we present our main compar-
ison results and the ablation study and related discussions.
Dataset
To evaluate the performance and scalability of our model,
we selected 17 scenes from four widely-used datasets.
Small-Scale Datasets.
To evaluate rendering fidelity on
complex, object-centric scenes, we use two standard
datasets. Mip-NeRF 360 (Barron et al. 2022) contains seven
challenging scenes featuring unbounded 360° camera trajec-
tories, with approximately 200 images per scene. Tanks &
Temples (Knapitsch et al. 2017) is used for assessing the
reconstruction of fine-grained geometric details, each scene
containing over 200 images.
Large-Scale Datasets.
To demonstrate the scalability in
large-scale environments, we employ two distinct datasets.
Waymo Open Dataset (Sun et al. 2020) provides real-world
autonomous driving scenarios. We utilize six sequences,
each with around 600 frames, to test our model’s ability
to handle extensive environments and long camera paths.
Matrix City (Li et al. 2023) is a massive, city-scale dataset
that tests the limits of scalability. Following Octree-GS (Ren
et al. 2024), we test on a blocksmall partition which show-
cases the effectiveness of our hierarchical framework in ex-
tremely large-scale reconstruction and rendering tasks.
Implementation Details
We implemented our Pyramid-GS framework in PyTorch.
All experiments were conducted on a single NVIDIA RTX
A6000 GPU with CUDA 11.6, and models were trained for
40,000 iterations using the Adam optimizer. For our Pyra-
mid sampling, the number of levels L is automatically set
according to the dataset. Regarding the Scene Perception
Compensation module, the depth of field standard threshold
σz,thresh is set to 50.0, with a corresponding distance com-
pensation coefficient α of 0.7. The coverage weight β is 0.5.
Finally, for our Laplacian Mixed Model, the rate-distortion
trade-off hyperparameter λ from Eq. 10 is uniformly set to
0.0005 for all reported results
Comparison Experiments
We conducted extensive comparison experiments against
state-of-the-art 3D reconstruction and rendering methods,
including 3DGS (Kerbl et al. 2023), 2DGS (Huang et al.
2024), Scaffold-GS (Lu et al. 2024), Hierarchical-GS (Kerbl
et al. 2024), Octree-GS (Ren et al. 2024), Context-GS (Wang
et al. 2024) and HAC++ (Chen et al. 2025). Our evaluation
focuses on key metrics such as rendering speed, memory
usage, primitive number and reconstruction quality (PSNR,
SSIM (Wang et al. 2004), LPIPS (Zhang et al. 2018)).
As demonstrated in Tab. 1, Tab. 2 and Fig. 3, our Pyramid-
GS consistently achieves competitive rendering quality and
efficiency across a diverse range of datasets. These datasets
span a wide range of scales and complexities, from object-
centric small scenes to large-scale autonomous driving and
massive urban environments, allowing for a comprehensive
evaluation of the capabilities of our method.
Specifically, Tab. 1 highlights the effectiveness of our
Pyramid-GS on large-scale datasets. On Waymo dataset,
our method achieved optimal rendering results with mini-
mal memory consumption, underscoring the scalability and
efficiency of our approach. For Matrix City, we achieved
the best rendering quality, with memory consumption only
slightly increasing compared to HAC++ and Context-GS.
These results confirm the advantage of our hierarchical
scene perception representation in significantly reducing
the number of active Gaussians that need to be processed
and stored, thereby improving memory efficiency on large
scenes with the compression algorithm.
In contrast, Tab. 2 presents results for small-scale scenes,
such as Mip-NeRF360 and Tanks & Temples. On Mip-
NeRF360, our method achieved the most best results, while
on Tanks & Temples, we reached a near-optimal state, no-
tably utilizing the fewest Gaussians to achieve this render-
ing quality. This is because the Tanks & Temples are dense
and involve more complex textures, while our model pays
more attention to large-scale and efficient management than
the rendering quality, so it can achieve suboptimal rendering
quality with the least number of Gaussians. This reinforces
our core contribution: a novel scene perception and com-
pression framework for 3D Gaussian Splatting.
The experimental results reveal that our Pyramid-GS ex-
cels at maintaining high rendering quality across diverse
scales due to our scene perception compensation algorithm.
Furthermore, our pyramid representation and compression
module fosters a flexible and memory-efficient scene repre-
sentation. It allows for adaptive rendering that scales grace-
fully with scene complexity, preserving essential details
even at lower levels of the hierarchy. We present qualitative
results in Fig. 3, which show that our model maintains pho-
torealistic rendering quality and handles complex geometry
effectively across different scales. The performance gains
are more significant in large-scale environments, confirming
the robustness of our perceive-sample-compress approach.
Ablation Studies
To evaluate the contribution of each proposed component,
we performed comprehensive ablation studies as shown in
Tab. 3, and Fig. 4.
Impact of Scene Perception Compensation
First, we
evaluate the importance of the scene perception compensa-
tion algorithm. We compare the Pyramid-GS with a variant
that uses a camera distance-based level partition and mask
without considering the scene compensation. The results in
Tab 3 show that the disabling of scene perception compen-
sation leads to a degradation in rendering quality. In cer-
tain complex scenes, relying solely on distance can cause
important details to be oversimplified. The depth and cov-

<!-- page 6 -->
Table 1: Quantitative comparison on Waymo (Sun et al. 2020) and MatrixCity (Li et al. 2023) datasets. The color of each cell
shows the best and the second best .
Dataset
Method
Waymo (Sun et al. 2020)
MatrixCity (Li et al. 2023)
SSIM↑
PSNR↑
LPIPS↓
#GS(k)↓
Mem(MB)↓
SSIM↑
PSNR↑
LPIPS↓
#GS(k)↓
Mem(MB)↓
3DGS (Kerbl et al. 2023)
0.840
27.19
0.313
1530
382.6
0.823
26.82
0.246
1432
3387.4
2DGS (Huang et al. 2024)
0.801
24.83
0.373
860
187.1
0.818
26.28
0.232
832
1659.1
Hierarchical-GS (Kerbl et al. 2024)
0.808
24.85
0.307
530
1035.7
0.823
27.69
0.276
271
1866.7
Scaffold-GS (Lu et al. 2024)
0.843
27.51
0.310
175
104.2
0.868
29.00
0.210
357
371.2
Octree-GS (Ren et al. 2024)
0.828
26.82
0.321
171
97.5
0.887
29.83
0.192
360
380.3
ContextGS (Wang et al. 2024)
0.847
27.93
0.310
133
11.1
0.878
29.26
0.188
221
30.3
HAC++ (Chen et al. 2025)
0.846
27.68
0.311
124
15.7
0.884
29.29
0.175
206
39.5
Ours
0.851
28.18
0.300
105
9.5
0.896
30.08
0.154
406
44.0
Table 2: Quantitative comparison on Mip-NeRF360 (Barron et al. 2022) and Tanks & Temples datasets (Knapitsch et al. 2017).
The color of each cell shows the best and the second best .
Dataset
Method
Mip-NeRF360 (Barron et al. 2022)
Tanks & Temples (Knapitsch et al. 2017)
SSIM↑
PSNR↑
LPIPS↓
#GS(k)↓
Mem(MB)↓
SSIM↑
PSNR↑
LPIPS↓
#GS(k)↓
Mem(MB)↓
3DGS (Kerbl et al. 2023)
0.870
28.69
0.182
962
754.7
0.844
23.69
0.178
765
430.1
2DGS (Huang et al. 2024)
0.863
28.53
0.201
399
390.4
0.830
23.25
0.212
352
204.4
Scaffold-GS (Lu et al. 2024)
0.870
29.35
0.188
658
182.6
0.853
23.96
0.177
626
167.5
Octree-GS (Ren et al. 2024)
0.867
29.11
0.188
695
140.4
0.866
24.68
0.153
443
88.5
ContextGS (Wang et al. 2024)
0.868
29.30
0.194
685
19.9
0.855
24.29
0.176
469
11.8
HAC++ (Chen et al. 2025)
0.865
29.26
0.207
680
16.0
0.854
24.32
0.178
481
8.6
Ours
0.876
29.39
0.187
377
15.8
0.847
24.32
0.174
317
11.7
Figure 3: Comparison Results. Visual differences are highlighted with yellow insets for better clarity. Our approach consis-
tently outperforms other models on different scenes, demonstrating advantages in challenging scenarios. Best viewed in color.

<!-- page 7 -->
Figure 4: Effect of Scene Perception Compensation.
While relying solely on camera distance can work well in
most situations, in some extreme scenes it may fail and re-
sult in poor rendering results. This is because that too many
key points are ignored by the algorithm. Through our algo-
rithm, the model can correct errors and preserve details.
erage compensation ensures that Gaussians relevant to the
viewer’s perspective are prioritized, leading to better percep-
tual quality and more efficient rendering of visually signif-
icant features. As shown in Fig. 4, in some extreme situa-
tions, the camera distance-based function will lose control
and fail to optimize the scene reconstruction.
Table 3: Ablation study of different components on Waymo
dataset, where SPC means the scene perception compensa-
tion algorithm, PS denotes pyramid sampling.
SSIM↑
PSNR↑
LPIPS↓
#GS(k)↓
Mem(MB)↓
w/o SPC
0.833
27.12
0.312
138
14.5
w/o PS
0.845
27.95
0.301
154
17.2
w/o SPC & PS
0.846
27.90
0.308
115
11.3
w/o Compression
0.847
27.96
0.301
144
81.3
Ours
0.851
28.18
0.300
105
9.5
Impact of Pyramid Sampling
We evaluate the efficacy
of our Laplacian pyramid-based hierarchical representation
against a baseline approach that uses only voxelization with-
out the multi-level aggregation. As shown in Table 3, the full
Pyramid-GS with Laplacian pyramid representation signif-
icantly reduces the number of active Gaussians. This hier-
archical structure not only decreases the number of primi-
tive by about 30% compared to simple voxelization but also
leads to a noticeable improvement in reconstruction qual-
ity. This highlights the benefit of our multi-level aggregation
strategy for efficient scene management.
When we replace the pyramid representation with the
Octree-GS, we can observe a degradation in rendering qual-
ity, this is because we can avoid over-pruning important
points through a backward upsampling process.
Impact of Compression Algorithm
Finally, we evalu-
ate the effectiveness of our Generalized Gaussian Mixed
Table 4: Ablation study on density threshold λ on Waymo
dataset. Increasing λ reduces memory consumption and
number of Gaussian primitive, but slightly affects quality.
λ
SSIM↑
PSNR↑
LPIPS↓
#GS(k)↓
Mem(MB)↓
FPS↑
0.0005
0.851
28.18
0.300
105
9.5
138
0.0014
0.848
28.07
0.310
81
6.9
152
0.0018
0.846
27.95
0.315
76
6.0
159
0.0030
0.842
27.80
0.322
66
5.1
161
0.0040
0.839
27.65
0.329
56
4.1
163
compression algorithm for storing the Gaussian representa-
tions. Tab. 3 demonstrates that our compression algorithm
can achieve significant size reductions with minimal impact
on rendering quality. When replacing it with the Gaussian
Mixed model used in HAC++, we can observe a degradation
in both rendering quality and memory consumption. This
highlights the efficiency of our compression scheme, which
leverages the structure of the pyramid to effectively repre-
sent Gaussian primitives with fewer parameters, making our
Pyramid-GS highly practical for storage and deployment.
Table 5: Comparison of training, encoding, and decoding
time across different methods.
Method
Training Time
Enc. Time(s)
Dec. Time(s)
Octree-GS (Ren et al. 2024)
49min
-
-
Context-GS (Wang et al. 2024)
1h58min
78.39
72.59
HAC++ (Chen et al. 2025)
1h34min
30.81
48.81
Ours
1h38min
16.31
28.95
Discussion
We explore the relationship between the compression rate,
rendering quality, speed, and final storage size by adjust-
ing the λ. Tab. 4 shows that increasing λ results in higher
compression rates, which reduces the final storage size and
increase the FPS. However, this also leads to a decrease in
rendering quality. These findings highlight the trade-offs in-
volved: higher compression leads to more compact storage
and faster rendering but at the expense of visual fidelity.
Additionally, we observe that our method has compara-
ble enc/dec times for compression and overall training times
to HAC++, indicating similar computational efficiency in
Tab. 5. This demonstrates that our approach maintains com-
petitive performance while achieving effective compression.
As for the training time, it is longer than non-compression
methods. This is because joint optimization takes more time.
Conclusion
In this paper, we proposed Pyramid-GS, a novel approach
to 3D Gaussian Splatting that leverages a hierarchical
scene perception and compression representation for effi-
cient primitive management. Through extensive experiments
across diverse datasets, we demonstrated the effectiveness
and scalability of our method in both small-scale and large-
scale environments. Our ablation studies further validated
the contributions of key components, emphasizing the im-
portance of scene perception compensation, pyramid sam-

<!-- page 8 -->
pling and compression algorithm. The results indicate that
Pyramid-GS not only enhances rendering quality but also
significantly improves computational efficiency, making it a
valuable tool for various applications in 3D reconstruction.
References
Ali, M. S.; Qamar, M.; Bae, S.-H.; and Tartaglione, E. 2024.
Trimming the fat: Efficient compression of 3d gaussian
splats through pruning. arXiv preprint arXiv:2406.18214.
Avidan, S.; and Shashua, A. 1997. Novel view synthesis in
tensor space. In Proceedings of IEEE computer society con-
ference on computer vision and pattern recognition, 1034–
1040. IEEE.
Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.;
and Hedman, P. 2022.
Mip-nerf 360: Unbounded anti-
aliased neural radiance fields.
In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, 5470–5479.
Chen, J.; Ye, W.; Wang, Y.; Chen, D.; Huang, D.; Ouyang,
W.; Zhang, G.; Qiao, Y.; and He, T. 2024a. Gigags: Scaling
up planar-based 3d gaussians for large scene surface recon-
struction. arXiv preprint arXiv:2409.06685.
Chen, Y.; Wu, Q.; Lin, W.; Harandi, M.; and Cai, J. 2024b.
Hac: Hash-grid assisted context for 3d gaussian splatting
compression. In European Conference on Computer Vision,
422–438. Springer.
Chen, Y.; Wu, Q.; Lin, W.; Harandi, M.; and Cai, J. 2025.
Hac++: Towards 100x compression of 3d gaussian splatting.
arXiv preprint arXiv:2501.12255.
Choi, I.; Gallo, O.; Troccoli, A.; Kim, M. H.; and Kautz,
J. 2019.
Extreme view synthesis.
In Proceedings of the
IEEE/CVF international conference on computer vision,
7781–7790.
Cui, J.; Cao, J.; Zhao, F.; He, Z.; Chen, Y.; Zhong, Y.; Xu,
L.; Shi, Y.; Zhang, Y.; and Yu, J. 2024. Letsgo: Large-scale
garage modeling and rendering via lidar-assisted gaussian
primitives. ACM Transactions on Graphics (TOG), 43(6):
1–18.
Dalal, A.; Hagen, D.; Robbersmyr, K. G.; and Knausg˚ard,
K. M. 2024. Gaussian splatting: 3D reconstruction and novel
view synthesis: A review. IEEE Access, 12: 96797–96820.
David, L.; Reddy, M.; Cohen, J.; Varshney, A.; Watson, B.;
and Huebner, R. 2003. Level of detail for 3D graphics.
Fan, Z.; Wang, K.; Wen, K.; Zhu, Z.; Xu, D.; Wang, Z.; et al.
2024. Lightgaussian: Unbounded 3d gaussian compression
with 15x reduction and 200+ fps. Advances in neural infor-
mation processing systems, 37: 140138–140158.
Girish, S.; Gupta, K.; and Shrivastava, A. 2024.
Eagles:
Efficient accelerated 3d gaussians with lightweight encod-
ings. In European Conference on Computer Vision, 54–71.
Springer.
Huang, B.; Yu, Z.; Chen, A.; Geiger, A.; and Gao, S. 2024.
2d gaussian splatting for geometrically accurate radiance
fields. In ACM SIGGRAPH 2024 conference papers, 1–11.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3D Gaussian splatting for real-time radiance field ren-
dering. ACM Trans. Graph., 42(4): 139–1.
Kerbl, B.; Meuleman, A.; Kopanas, G.; Wimmer, M.; Lan-
vin, A.; and Drettakis, G. 2024. A hierarchical 3d gaussian
representation for real-time rendering of very large datasets.
ACM Transactions on Graphics (TOG), 43(4): 1–15.
Knapitsch, A.; Park, J.; Zhou, Q.-Y.; and Koltun, V. 2017.
Tanks and temples: Benchmarking large-scale scene recon-
struction. ACM Transactions on Graphics (ToG), 36(4): 1–
13.
Lee, J. C.; Rho, D.; Sun, X.; Ko, J. H.; and Park, E. 2024.
Compact 3d gaussian representation for radiance field. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, 21719–21728.
Li, Y.; Jiang, L.; Xu, L.; Xiangli, Y.; Wang, Z.; Lin, D.; and
Dai, B. 2023. Matrixcity: A large-scale city dataset for city-
scale neural rendering and beyond. In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
3205–3215.
Liu, X.; Wu, X.; Zhang, P.; Wang, S.; Li, Z.; and Kwong, S.
2024a. Compgs: Efficient 3d scene representation via com-
pressed gaussian splatting. In Proceedings of the 32nd ACM
International Conference on Multimedia, 2936–2944.
Liu, Y.; Luo, C.; Fan, L.; Wang, N.; Peng, J.; and Zhang,
Z. 2024b. Citygaussian: Real-time high-quality large-scale
scene rendering with gaussians. In European Conference on
Computer Vision, 265–282. Springer.
Lombardi, S.; Simon, T.; Saragih, J.; Schwartz, G.;
Lehrmann, A.; and Sheikh, Y. 2019.
Neural volumes:
Learning dynamic renderable volumes from images. arXiv
preprint arXiv:1906.07751.
Lu, T.; Yu, M.; Xu, L.; Xiangli, Y.; Wang, L.; Lin, D.;
and Dai, B. 2024. Scaffold-gs: Structured 3d gaussians for
view-adaptive rendering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
20654–20664.
Milef, N.; Seyb, D.; Keeler, T.; Nguyen-Phuoc, T.; Boˇziˇc,
A.; Kondguli, S.; and Marshall, C. 2025. Learning fast 3D
gaussian splatting rendering using continuous level of de-
tail. In Computer Graphics Forum, e70069. Wiley Online
Library.
Navaneet, K.; Meibodi, K. P.; Koohpayegani, S. A.; and
Pirsiavash, H. 2023.
Compact3d: Compressing gaussian
splat radiance field models with vector quantization. arXiv
preprint arXiv:2311.18159, 2(3).
Niedermayr, S.; Stumpfegger, J.; and Westermann, R. 2024.
Compressed 3d gaussian splatting for accelerated novel view
synthesis. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 10349–10358.
Papantonakis, P.; Kopanas, G.; Kerbl, B.; Lanvin, A.; and
Drettakis, G. 2024. Reducing the memory footprint of 3d
gaussian splatting. Proceedings of the ACM on Computer
Graphics and Interactive Techniques, 7(1): 1–17.
Penner, E.; and Zhang, L. 2017.
Soft 3d reconstruction
for view synthesis. ACM Transactions on Graphics (TOG),
36(6): 1–11.

<!-- page 9 -->
Ren, K.; Jiang, L.; Lu, T.; Yu, M.; Xu, L.; Ni, Z.; and
Dai, B. 2024. Octree-gs: Towards consistent real-time ren-
dering with lod-structured 3d gaussians.
arXiv preprint
arXiv:2403.17898.
Riegler, G.; and Koltun, V. 2020. Free view synthesis. In Eu-
ropean conference on computer vision, 623–640. Springer.
Seo, Y.; Choi, Y. S.; Son, H. S.; and Uh, Y. 2024. Flod: Inte-
grating flexible level of detail into 3d gaussian splatting for
customizable rendering. arXiv preprint arXiv:2408.12894.
Sun, P.; Kretzschmar, H.; Dotiwalla, X.; Chouard, A.; Pat-
naik, V.; Tsui, P.; Guo, J.; Zhou, Y.; Chai, Y.; Caine, B.; et al.
2020.
Scalability in perception for autonomous driving:
Waymo open dataset. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, 2446–
2454.
Wang, Y.; Li, Z.; Guo, L.; Yang, W.; Kot, A.; and Wen, B.
2024. Contextgs: Compact 3d gaussian splatting with an-
chor level context model. Advances in neural information
processing systems, 37: 51532–51551.
Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P.
2004.
Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image process-
ing, 13(4): 600–612.
Wang, Z.; and Xu, D. 2024. Pygs: Large-scale scene rep-
resentation with pyramidal 3d gaussian splatting.
arXiv
preprint arXiv:2405.16829.
Watson, D.; Chan, W.; Martin-Brualla, R.; Ho, J.; Tagliasac-
chi, A.; and Norouzi, M. 2022. Novel view synthesis with
diffusion models. arXiv preprint arXiv:2210.04628.
Wizadwongsa, S.; Phongthawee, P.; Yenphraphai, J.; and
Suwajanakorn, S. 2021. Nex: Real-time view synthesis with
neural basis expansion. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
8534–8543.
Yariv, L.; Hedman, P.; Reiser, C.; Verbin, D.; Srinivasan,
P. P.; Szeliski, R.; Barron, J. T.; and Mildenhall, B. 2023.
Bakedsdf: Meshing neural sdfs for real-time view synthesis.
In ACM SIGGRAPH 2023 conference proceedings, 1–9.
Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang,
O. 2018. The unreasonable effectiveness of deep features as
a perceptual metric. In Proceedings of the IEEE conference
on computer vision and pattern recognition, 586–595.
Zhou, T.; Tucker, R.; Flynn, J.; Fyffe, G.; and Snavely, N.
2018. Stereo magnification: Learning view synthesis using
multiplane images. arXiv preprint arXiv:1805.09817.
