<!-- page 1 -->
P-4DGS: Predictive 4D Gaussian Splatting with 90×
Compression
Henan Wang
Hanxin Zhu
Xinliang Gong
Tianyu He
Xin Li∗Zhibo Chen∗
University of Science and Technology of China
{henanwang, hanxinzhu, qq0707}@mail.ustc.edu.cn
{xin.li, chenzhibo}@ustc.edu.cn
Abstract
3D Gaussian Splatting (3DGS) has garnered significant attention due to its supe-
rior scene representation fidelity and real-time rendering performance, especially
for dynamic 3D scene reconstruction (i.e., 4D reconstruction). However, despite
achieving promising results, most existing algorithms overlook the substantial tem-
poral and spatial redundancies inherent in dynamic scenes, leading to prohibitive
memory consumption. To address this, we propose P-4DGS, a novel dynamic
3DGS representation for compact 4D scene modeling. Inspired by intra- and
inter-frame prediction techniques commonly used in video compression, we first
design a 3D anchor point-based spatial-temporal prediction module to fully exploit
the spatial-temporal correlations across different 3D Gaussian primitives. Subse-
quently, we employ an adaptive quantization strategy combined with context-based
entropy coding to further reduce the size of the 3D anchor points, thereby achiev-
ing enhanced compression efficiency. To evaluate the rate-distortion performance
of our proposed P-4DGS in comparison with other dynamic 3DGS representa-
tions, we conduct extensive experiments on both synthetic and real-world datasets.
Experimental results demonstrate that our approach achieves state-of-the-art recon-
struction quality and the fastest rendering speed, with a remarkably low storage
footprint (around 1MB on average), achieving up to 40× and 90× compression on
synthetic and real-world scenes, respectively.
1
Introduction
Recently, 4D Gaussian Splatting (i.e., dynamic 3DGS) [1, 2] has emerged as a powerful paradigm for
reconstructing dynamic 3D scenes, enabling high-fidelity modeling of appearance and motion while
supporting real-time rendering and continuous free-viewpoint exploration.
To achieve this goal, existing methods [1, 3] mainly adopt a static canonical space combined with a
time-varying deformation field to model scene dynamics. In this formulation, the canonical space
encodes the structural information of the scene and serves as a reference for deformation field.
The deformation field captures the temporal evolution of each 3D Gaussian, tracking both spatial
trajectories and attribute changes across frames. For example, D3DGS [1] employs a deformable
MLP to learn the deformation of each canonical Gaussian. However, despite their success, existing
4D reconstruction approaches typically suffer from substantial storage costs, severely limiting their
scalability and real-world deployment.
To better understand and mitigate such inefficiencies, we draw an analogy between dynamic 3D
Gaussian representations and established techniques in video compression [4]. Intuitively, deforma-
tion field-based 4DGS frameworks parallel the predictive structures used in video coding, where the
*Corresponding Authors
Preprint. Under review.
arXiv:2510.10030v1  [cs.CV]  11 Oct 2025

<!-- page 2 -->
canonical space serves as a reference frame and the deformation field functions analogously to motion
vectors that describe inter-frame changes. This observation motivates us to ask: can we design spatial
and temporal prediction structures for dynamic Gaussians, analogous to intra- and inter-prediction
in video compression?
Specifically, video codecs exploit spatial redundancy via intra-frame prediction, temporal redundancy
via inter-frame prediction, and contextual redundancy through entropy models such as CABAC [5].
Inspired by this, we propose an efficient dynamic 3DGS representation (i.e., P-4DGS) that jointly
leverages temporal and spatial prediction. For spatial prediction, we adopt a 3D anchor point-based
predictive structure to exploit the spatial correlations among different 3D Gaussians within the
canonical space, where nearby Gaussians are predicted by a single anchor point to reduce the number
of primitives. For temporal prediction, we utilize the deformation MLP to predict the deformation
vector of each 3D Gaussian from the canonical space to specific time. Furthermore, we introduce
adaptive quantization and a context-aware entropy model to enhance the compression efficiency of
the canonical space’s 3D Gaussians.
We verify our method in multiple benchmarks including both real scenes and synthetic scenes [6, 7].
Results show that our method can achieve better rendering quality while greatly reducing the size
compared to existing 4D compression methods.
The main contributions of this paper can be summarized as:
• We propose P-4DGS, a novel dynamic 3D Gaussian representation designed for compact 4D scene
reconstruction.
• Inspired by intra- and inter-prediction paradigm of video coding, we design a spatial-temporal
prediction module to exploit redundancies within dynamic 3D Gaussians. We further introduce
adaptive quantization and context-based entropy coding to facilitate more efficient coding of these
dynamic 3D Gaussians.
• Experimental results demonstrate that our method achieves high compression efficiency—requiring
less than 1 MB of storage with a 90× compression ratio—while delivering improved rendering
quality.
2
Related Work
2.1
Dynamic 3D Representation
3D Gaussian Splatting [8] provides an innovative representation model for novel view synthesis. With
modest storage needs and after short training sessions, 3DGS has the ability to carry out real-time,
high-fidelity view synthesis in large-scale scenes. Several works [9, 3, 1, 10] have adapted 3DGS to
facilitate the reconstruction of dynamic scenes. On one hand, recent works [2, 11, 12, 13, 14] directly
train a collection of 4D Gaussians to represent static, dynamic, and transient elements in a scene.
Nevertheless, these approaches necessitate a substantial quantity of Gaussians to create high-quality
scene representations, leading to significant storage costs. On the other hand, a range of alternative
works aim to construct geometry and depict dynamics by collaboratively optimizing the Gaussian
and deformation fields in the canonical space. For example, Deformable-3DGS [1] utilizes an MLP
deformation network to represent the motion in dynamic scenes. Similarly, 4DHexPlane [3] leverages
Hexplane [15] to encode sparse data and then outputs the results via a multi-head Gaussian decoder,
successfully enabling real-time rendering at high resolution.
Although both categories achieve decent rendering quality for dynamic scenes, they still demand
substantial storage to handle the multi-dimensional attributes of millions of 3D Gaussians in canonical
Gaussians. Based on this, we propose to perform 4D scene reconstruction on a more compact Gaussian
representation, and simultaneously incorporate joint optimization with entropy encoding to achieve
the compression of the 4D scene.
2.2
Compact 3D Representation
To reduce the substantial memory requirements of 3DGS, researchers have developed two main
strategies. The first category focuses on traditional compression techniques without altering the
original 3DGS representation. These traditional compression methods cover vector quantization [16,
2

<!-- page 3 -->
Anchors
Rendered
Frames
Canonical
Gaussians
Deformed
Gaussians
Anchor MLP
Spatial
Prediction
Deformation MLP
Temporal
Prediction
Rendering
Figure 1: Rendering pipeline of P-4DGS. The pipeline first performs spatial prediction by mapping
anchor points in the canonical space to static Gaussian primitives via an anchor prediction module.
Then, temporal prediction is conducted using a deformation MLP that maps these primitives to a
target time step t, producing dynamic Gaussian primitives for final image rendering.
17, 18, 19, 20, 21], pruning redundant Gaussians [16, 17, 22, 20, 23, 24], implicit encoding of high-
dimensional attributes [25, 17, 26], utilization of standardized compression pipelines [16, 27, 26, 28],
and implementation of entropy constraint [22, 29]. The second category explores more compact
Gaussian representations to mitigate storage challenges [30, 31]. A prominent example is Scaffold-
GS [30], which introduces a unique method by assigning learnable features to a sparse set of anchor
points that predict attributes for a broader set of neighboring 3D Gaussians. Building on Scaffold-GS,
subsequent research has focused on optimizing entropy coding methods by designing various types
of entropy models to more efficiently estimate the distributions of Gaussian attributes [32, 33, 34, 35,
36, 37].
However, adapting methods from both categories to the reconstruction of dynamic scenes may not be
straightforward, as most 4D extensions require substantial architectural modifications to extend 3DGS
for dynamic scene modeling. In our 4D compression work, we conduct dynamic scene reconstruction
on a more compact 3DGS and incorporate an entropy model for joint optimization. This approach
enables us to achieve a high compression ratio while maintaining high rendering quality and speed.
2.3
Video Coding
Video compression aims to reduce storage and transmission costs by exploiting spatial and temporal
redundancies in video data. Standard codecs such as H.264/AVC [38], H.265/HEVC [4], and
H.266/VVC [39] achieve this through intra- and inter-prediction. Intra-prediction operates within a
single frame by predicting a block of pixels based on the values of its neighboring reconstructed blocks.
It uses various directional modes (e.g., vertical, horizontal, angular) to estimate the content, then
encodes only the residual (difference) between the prediction and the actual block. Inter-prediction
predicts the current frame using one or more reference frames. It involves motion estimation to find
matching blocks in reference frames and motion compensation to generate a predicted block using
motion vectors. The encoder then stores the residual and motion data instead of the raw block. These
prediction modules are followed by transform coding, quantization, and entropy coding steps to
further compress the residuals and motion information.
3
Method
3.1
Overview
Fig. 1 and Fig. 2 illustrate the rendering and entropy coding pipeline of our proposed P-4DGS,
respectively. The rendering pipeline is responsible for the generation and rendering of dynamic 3D
Gaussian primitives, while the entropy coding pipeline focuses on data compression.
In the rendering pipeline, anchor points in the canonical space are first processed through an anchor
prediction module to generate static Gaussian primitives in the canonical space. These Gaussian
primitives then serve as inputs to the temporal prediction module, where a deformation MLP is
employed to compute the deformation field from the canonical space to the target time step t. The
resulting deformation vectors are applied to the canonical Gaussian primitives to produce dynamic
Gaussian primitives corresponding to time t, which are subsequently rendered into the final images
through a rendering module.
3

<!-- page 4 -->
In the entropy coding pipeline, the anchor points in the canonical space undergo quantization,
followed by entropy coding to generate a compact bitstream representation. This process enhances
the efficiency of both storage and transmission.
3.2
Spatial-temporal Prediction
Spatial prediction.
The spatial prediction of Gaussian primitives primarily adopts an anchor-based
prediction scheme [30], which exploits spatial correlations among Gaussian primitives. Several
spatially adjacent Gaussian primitives are predicted from a single anchor, significantly reducing
storage overhead. Each anchor is characterized by five key attributes: position xa ∈R3, scale
sa ∈R3, offset scaling la ∈R3, learnable offsets Oa ∈Rk×3 and anchor feature fa ∈Rd.
During rendering, given a camera pose, each visible anchor within the view frustum generates k
nearby Gaussian primitives and predicts their corresponding attributes. Specifically, for an anchor
located at xa, the positions of its k associated Gaussian primitives are computed as:
{x0, ..., xk−1} = xa + {O0, ...Ok−1} · la,
(1)
where {O0, ...Ok−1} ∈Rk×3 are learnable offsets.
The remaining four attributes for the k Gaussian primitives are predicted from the anchor feature fa
by MLPs. These MLPs take as input the anchor feature fa, the relative distance δvc, and the viewing
direction dvc between the anchor and the camera position:
δvc = ∥xa −xc∥2,
dvc =
xa −xc
∥xa −xc∥2
,
(2)
where xc denotes the camera position.
Using these inputs, the opacity of the k Gaussian primitives is predicted as:
{α0, ..., αk−1} = ψα (fa, δvc, dvc) ,
(3)
where ψα denotes the MLP for opacity.
The color c and rotation r attributes are predicted analogously. For the scale attribute, the MLP output
is interpreted as a residual scaling factor relative to the anchor scale sa:
{s0, ..., sk−1} = sa · sigmoid (ψs (fa, δvc, dvc)) ,
(4)
where ψs denotes the MLP for scale attribute.
Like Scaffold-GS [30], to reduce rendering overhead, Gaussian primitives with opacity values below
a threshold (α < τα) are excluded from rendering.
Temporal prediction.
Following D3DGS [1], we adopt a canonical space plus deformation field
model for temporal prediction. The set of Gaussian primitives predicted by anchors serves as
the canonical space. Given a time input t, we query the deformation field for the corresponding
deformation vectors at time t and apply them to the Gaussian primitives in the canonical space,
thereby obtaining the Gaussian primitives at time t.
Temporal prediction consists of two main components: positional encoding and the deformation
MLP. Given the position of a Gaussian primitive and the time t, temporal prediction first encodes the
spatial and temporal information into high-dimensional vectors, enhancing the MLP’s ability to learn
high-frequency variations. These encoded vectors are concatenated and fed into a deformation MLP
to predict deformation vectors for the Gaussian primitive’s position, scale, and rotation, denoted as
(∆x, ∆s, ∆r):
(∆x, ∆s, ∆r) = ψd(concat(E(x), E(t))),
(5)
where concat indicates vector concatenation, E denotes positional encoding and ψd is the deforma-
tion MLP. The deformed attributes of the Gaussian primitive at time t are obtained by adding the
deformation vectors to the canonical attributes:
(x′, s′, r′) = (x + ∆x, s + ∆s, r + ∆r).
(6)
4

<!-- page 5 -->
Hash Grid
Anchor Position
𝑥𝑎
Hash Feature 
ℎ
Anchor Attributes
𝑠𝑎, 𝑙𝑎, 𝑓𝑎, 𝐎𝑎
Bitstream of Anchor 
Attributes
Interpolation
Adaptive
Quantization
Quantized 
Anchor Attributes
Ƹ𝑠𝑎, መ𝑙𝑎, መ𝑓𝑎, ෡𝐎𝑎
Context-based
Entropy Coding
𝜇
𝜎
Context MLP
Context
𝑞, 𝜇, 𝜎
𝑞
𝑞
𝜇
𝜎
Context Generation
Anchor Compression
Figure 2: Entropy coding pipeline of P-4DGS, consisting of context generation and anchor compres-
sion. In context generation, anchor positions xa query a hash grid to produce a feature h. which
an MLP maps to quantization step q, mean µ, and standard deviation σ. In anchor compression,
attributes s, l, f, O are adaptively quantized using q and encoded into a bitstream via context-based
entropy coding with µ, σ.
3.3
Adaptive Quantization and Entropy Coding
Adaptive quantization.
Anchor attributes, including scale sa ∈R3, offset scaling la ∈R3, offsets
of k Gaussian primitives Oa ∈Rk×3, and anchor features fa ∈Rd, are quantized using scalar
quantization after training to reduce storage overhead. To address the undifferentiable problem caused
by quantization, we adopt uniform noise to simulate the quantization error during training [40].
Furthermore, the quantization step is adjusted adaptively for each attribute. Taking the anchor feature
as an example, the adaptive quantization process during training is defined as:
˜fa = fa + U

−1
2, 1
2

· q,
(7)
where ˜fa is the quantized anchor feature with added uniform noise, U(−1
2, 1
2) is a uniform distribution
over (−1
2, 1
2), and q is the quantization step size, computed as:
q = Q0 · (1 + tanh(ψq(h))) ,
(8)
where Q0 is the base quantization step size, and ψq is an MLP that learns a residual offset from Q0
based on the context hash feature h. The quantization step size q lies in the range (0, 2Q0). The hash
feature h is queried from a binary hash grid H using the anchor position xa:
h = H(xa),
(9)
The hash grid consists of one 3D grid (along the x, y, z dimensions) and three 2D grids (along the
xy, yz, and xz planes), where each grid point corresponds to an anchor in the canonical space and
stores its contextual information.
During inference, hard quantization is performed using rounding:
ˆfa = ⌊fa
q ⌉· q,
(10)
where ˆfa is the quantized anchor feature.
Context entropy coding.
To accurately estimate the bitrate and enable rate-distortion optimization,
we design an entropy model to estimate the probability distribution of the quantized anchor attributes.
5

<!-- page 6 -->
Figure 3: Rate-distortion curves on D-NeRF and NeRF-DS datasets. The x-axis shows the bitrate
(log scale) of compressed Gaussian representations, and the y-axes report average PSNR, SSIM,
and LPIPS. Our method achieves high reconstruction quality across bitrates, outperforming D3DGS,
4DHexPlane, and 4DGS, with over 40× and 90× compression on D-NeRF and NeRF-DS, respec-
tively.
We assume a Gaussian prior for the quantized values, following [32, 41]. Taking the anchor feature
as an example, the discrete probability distribution is given by:
p( ˜fa) =
Z
˜
fa+ 1
2 q
˜
fa−1
2 q
ϕµ,σ(x)dx
= Φµ,σ

˜fa + 1
2q

−Φµ,σ

˜fa −1
2q

,
(11)
where ϕµ,σ, Φµ,σ are the probability density function and cumulative distribution function of a
Gaussian distribution with mean µ and standard deviation σ. These parameters are predicted by an
MLP ψp using the hash feature h:
(µ, σ) = ψp(h).
(12)
Using this estimated distribution, the total bitrate for anchor encoding is computed as:
Ranchor =
X
a
(−log2 p( ˜fa) −log2 p( ˜Oa) −log2 p(˜la) −log2 p(˜sa)).
(13)
In addition, we also compute the bitrate required for encoding the binary hash grid. Let p+ be the
probability of a grid value being 1 and N+ be the number of such values. Similarly, p−= 1 −p+ is
the probability of a grid value being -1, and N−is its count. The hash grid bitrate is computed as:
Rhash = N+(−log2(p+)) + N−(−log2(p−)).
(14)
Training objective.
The final objective function combines the rendering loss (Eq. 16) and the
bitrate loss (including both anchor and hash grid bitrates), balanced by a rate weighting factor λrate:
Ltotal = λrateLrate + Lrender,
(15)
Lrender = (1 −λSSIM) L1 + λSSIMLD−SSIM,
(16)
Lrate = Ranchor + Rhash.
(17)
4
Experiments
4.1
Experimental Settings
Datasets.
We evaluate the proposed method on two widely used dynamic 3D scene datasets: the
synthetic D-NeRF dataset [7] and the real-world NeRF-DS dataset [6]. In the D-NeRF dataset,
viewpoints are sampled on a trajectory centered around the object. In contrast, the NeRF-DS dataset
consists of videos captured using a stereo camera setup with two fixed viewpoints. For both datasets,
we follow the official train/test splits provided by the dataset authors.
6

<!-- page 7 -->
Table 1: Rate-distortion performance and rendering FPS on D-NeRF dataset.
Method
PSNR↑
SSIM↑
LPIPS↓
Rate (MB)↓
FPS↑
D3DGS
38.28
0.985
0.016
39.45
149
4DHexPlane
34.02
0.984
0.021
23.45
132
4DGS
32.47
0.976
0.027
375.34
147
Ours
38.10
0.985
0.017
1.039
262
Table 2: Rate-distortion performance and rendering FPS on NeRF-DS dataset.
Method
PSNR↑
SSIM↑
LPIPS↓
Rate (MB)↓
FPS↑
D3DGS
23.75
0.847
0.179
59.38
58
4DHexPlane
21.08
0.729
0.281
66.37
83
4DGS
23.00
0.814
0.259
235.95
208
Ours
24.18
0.855
0.184
0.704
274
Implementation details.
The training of dynamic 3DGS, which combines temporal and spatial
prediction, is divided into four main stages. In the first stage, a static canonical space is trained
using all training images from different time steps, for a total of 3,000 iterations. In the second stage,
quantization-aware training is introduced by injecting uniform noise into anchor point attributes
to simulate quantization noise. Training continues up to 4,000 iterations for real scenes or 5,000
iterations for synthetic scenes. In the third stage, temporal information is incorporated, enabling the
deformation MLP to learn mappings from the canonical space to specific time steps t until 10,000
iterations. In the fourth stage, the entropy model is integrated into the training pipeline to estimate
the bitrate of anchor attributes and enable joint rate-distortion optimization, continuing until 20,000
iterations for real scenes or 40,000 iterations for synthetic scenes.
Baselines.
We compare our method against three representative dynamic 3D Gaussian repre-
sentation approaches, including D3DGS [1], 4DHexPlane [3], and 4DGS [2]. Both D3DGS and
4DHexPlane adopt the canonical space–deformation field framework for temporal modeling. D3DGS
employs the same type of deformation field as ours, consisting solely of a deformation MLP. 4DHex-
Plane, in addition to the deformation MLP, introduces a multi-resolution hex-plane representation to
model spatiotemporal information. In contrast, 4DGS encodes temporal variation by augmenting
each Gaussian with an explicit time attribute, resulting in a time-aware 4D Gaussian point cloud.
4.2
Results
Rate-distortion performance.
By adjusting λrate, we generate a series of compressed dynamic
3DGS representations under different bitrates. Given a test view and time step, we deform the
Gaussian points in the canonical space and render an image, which is then compared to the corre-
sponding ground-truth image. The image quality is evaluated using PSNR, SSIM, and LPIPS metrics.
Meanwhile, the size of the compressed dynamic 3DGS representation is recorded as the bitrate. Using
bitrate as the x-axis and the three quality metrics as the y-axis, we plot the rate-distortion (RD) curves
as shown in Fig. 3. In addition, we compare the performance of our method against other baselines
D3DGS [1], 4DHexPlane [3] and 4DGS [1] on the same graph.
As shown in Fig. 3, on the D-NeRF dataset, our method achieves over 40× compression with minimal
quality degradation compared to D3DGS, mainly benefiting from the efficient spatial prediction
and context-aware entropy coding. 4DHexPlane and 4DGS exhibit relatively poor reconstruction
quality on synthetic data; even at the lowest bitrate, our method consistently achieves higher objective
quality than both. By comparing PSNR between training and testing views, we observe that these
two baselines suffer from severe overfitting, showing poor generalization to unseen views.
On the NeRF-DS dataset, our method surpasses the other three baselines in objective quality under
high-bitrate settings. At equal levels of quality, our approach achieves over 90× compression
compared to D3DGS.
Quantitative comparisons.
To qualitatively assess the rendering quality of our method compared
to other dynamic 3DGS representation approaches, we visualize rendering results from several scenes
in both the D-NeRF and NeRF-DS datasets. The comparisons are illustrated in Fig. 4. As shown,
7

<!-- page 8 -->
Figure 4: Quantitative comparisons on D-NeRF and NeRF-DS datasets. Our method achieves high-
fidelity rendering with significantly lower storage (~1MB) compared to D3DGS, 4DHexPlane, and
4DGS. It faithfully reconstructs dynamic scenes with minimal artifacts, while baseline methods show
visible degradation such as blur or loss of dynamic details.
our method achieves superior rendering quality with minimal storage cost, faithfully reconstructing
dynamic 3D scenes with no noticeable artifacts in most cases. In contrast, the baseline methods suffer
from various degrees of degradation, including blur, streaks, and visual artifacts, and often fail to
reconstruct the dynamic content accurately.
4.3
Ablation Studies
Bitrate savings of each module.
To investigate the contribution of each module to bitrate reduction,
we conduct an ablation study on the T-Rex scene. Starting from the baseline model D3DGS (which
uses only a temporal prediction module), we incrementally add components from our method. The
results are shown in Table 3.
The first step is adding the spatial prediction module, which reduces the bitrate from 56 MB to 7 MB (a
reduction of approximately 90%) with an additional 0.3 dB PSNR increase. This improvement can be
attributed to the ability of anchor-based prediction to effectively capture spatial redundancy between
neighboring Gaussians. Moreover, the number of Gaussians is reduced, enabling the deformation
MLP to learn more efficiently and represent the scene with both higher quality and lower storage cost.
The second step involves applying a compact MLP design, which reduces the number of layers in the
deformation MLP from 8 to 3 and the feature dimension from 256 to 192. The network parameters
are also quantized to float16 precision, significantly reducing the model size. This optimization
yields around 2 MB of bitrate savings.
8

<!-- page 9 -->
Table 3: Bitrate savings achieved through various techniques. Beginning with D3DGS [1], we
incrementally introduce our methodologies to quantify the reduction in bitrate compared to the
preceding step.
Module
Rate (MB)
Savings (%)
PSNR
D3DGS
56.7
-
38.04
+ Spatial Prediction
7.27
−87.1%
38.36
+ Compact Deformation MLP
5.39
−25.8%
38.34
+ Quantization & Entropy Coding
0.85
−84.1%
38.24
Table 4: Composition of our high-rate representation and low-rate one trained on scene T-Rex.
Low rate
High rate
Component
Rate (MB)
Proportion (%)
Rate (MB)
Proportion (%)
Anchor position
0.121
23.9%
0.118
13.2%
Anchor feature
0.021
4.2%
0.191
21.4%
Anchor scale
0.058
11.4%
0.099
11.1%
Gaussian offsets
0.024
4.7%
0.188
21.1%
Hash grid
0.003
0.7%
0.018
2.0%
Deformable MLP
0.241
47.6%
0.241
26.9%
Other MLPs
0.038
7.5%
0.038
4.2%
Total
0.506
-
0.857
-
Finally, the quantization and context entropy coding modules further compress the representation to
under 1MB with negligible loss in quality. This is achieved through quantization-aware training and
a learned entropy model that captures the underlying distributions of anchor attributes.
Representation composition.
We also analyze the component-wise composition of the compressed
representation by examining two bitrate settings from the T-Rex scene. The compressed representation
consists of anchor attributes (including position, features, scale, and Gaussian offsets), the hash grid,
the deformation MLP, and other MLPs such as those for anchor prediction and entropy estimation.
Across different bitrates, we observe that the anchor positions and deformation MLP parameters
remain constant, as they are quantized with fixed 16-bit precision. In contrast, the sizes of the other
components vary with the bitrate, primarily due to changes in the quantization step size introduced
by the adaptive quantization strategy.
4.4
Rendering efficiency
To evaluate the rendering efficiency of our approach, we compare the rendering speed (FPS) of our
method with other baselines on two dynamic 3D scene datasets. The results are shown in the last
columns of Table 1 and Table 2. All FPS measurements are conducted on an NVIDIA RTX 4090
GPU. As observed, our method achieves the highest rendering FPS on both datasets.
5
Conclusion
In this paper, we investigate an efficient representation tailored for dynamic 3D scenes. Inspired
by video encoding techniques, we propose a compact dynamic 3DGS representation framework
that integrates spatial-temporal prediction, adaptive quantization and context-based entropy coding.
Experimental results demonstrate that our method achieves optimal reconstruction quality and the
fastest rendering speed with minimal storage overhead. Compared to the baseline model, our
approach attains up to 40× and 90× compression rates on synthetic and real scenes respectively, while
maintaining comparable visual quality. The main limitation of our method lies in the deformation
MLP, whose storage overhead cannot be adaptively scaled to different bitrates, thereby limiting
compression efficiency under low-rate settings. In future work, we plan to explore more compact and
compressible temporal representations to address this issue and improve rate adaptability.
9

<!-- page 10 -->
A
Supplementary Material
In the supplementary material, we provide a detailed description of our hyperparameters in Sec. A.1
and more detailed quantitative results in Sec. A.2.
A.1
Hyperparameters
Spatial prediction module.
For the spatial prediction module, anchor points are initialized either
using the point cloud estimated by COLMAP (for real-world scenes) or through random sampling in
space (for synthetic scenes). The initialized point cloud is voxelized using a voxel size of ϵ = 0.01 for
real-world scenes or ϵ = 0.001 for synthetic scenes to obtain the initial anchor points. Each anchor
point can produce up to k = 10 Gaussian primitives during rendering. As the training proceeds
and Gaussians are pruned based on opacity, the value of k gradually decreases, averaging 3 to 4
Gaussians per anchor by the end of training. The update init factor for anchor growing is set to 16.
The dimension of anchor attributes is set to d = 16 for synthetic scenes and d = 32 for real-world
scenes. All anchor MLPs ψα, ψc, ψr, and ψs consist of 2 hidden layers with dimensions of 2d.
Temporal prediction module.
The deformation MLP ψd consists of two hidden layers, each with
192 dimensions. The output of the final hidden layer is fed into three independent output layers to
generate deformation vectors for the position, scale, and rotation attributes of k Gaussian primitives.
Quantization and entropy coding module.
For the quantization and entropy coding module, the
base quantization steps for anchor attributes, scale, scaling, and offset are set as Qf = 1, Qs = 0.001,
Ql = 0.001, and Qo = 0.2, respectively. Anchor positions are quantized using 16-bit uniform
quantization. Both MLPs ψp and ψq consist of two hidden layers, each with a hidden dimension of
2d.
Learning rate settings.
For anchor attributes, the learning rate is set to 0.01 for offsets, 0.0075 for
anchor feature, 0.02 for opacity, and 0.007 for scale and offset scaling. For anchor MLPs, the learning
rates are 0.004 for covariance, 0.008 for color, and 0.002 for opacity. The deformation MLP is trained
with a learning rate of 0.00016. The learning rates of hash grid and MLPs related to quantization
and entropy coding are set to 0.0016 and 0.005, respectively. During training, learning rates decay
exponentially with increasing iterations.
A.2
Quantitative Results
In this section, we provide detailed experimental results, including per-scene performance compar-
isons between our method and three baseline approaches (Tab.1 and Tab.2). Additionally, we report
the performance of our method under varying compression rates on two datasets (Tab. 3).
10

<!-- page 11 -->
Table 1: Per-scene quantitative comparison on D-NeRF dataset. We acquire the baseline results by
running their official codes.
Scene
bouncingballs
hellwarrior
hook
jumpingjacks
D3DGS
Rate (MB)
45.27
10.99
38.99
24.04
PSNR
40.56
41.35
36.98
37.57
SSIM
0.995
0.987
0.986
0.990
LPIPS
0.010
0.024
0.016
0.013
4DHexPlane
Rate (MB)
19.70
21.22
20.97
20.78
PSNR
40.74
28.68
32.93
35.34
SSIM
0.994
0.973
0.977
0.986
LPIPS
0.015
0.037
0.027
0.020
4DGS
Rate (MB)
243.19
400.54
322.34
433.92
PSNR
32.73
34.24
31.54
32.24
SSIM
0.983
0.955
0.959
0.971
LPIPS
0.030
0.080
0.047
0.037
Ours
Rate (MB)
1.07
0.63
1.33
0.73
PSNR
42.41
40.79
36.49
37.70
SSIM
0.996
0.987
0.985
0.990
LPIPS
0.008
0.027
0.017
0.015
Scene
lego
mutant
standup
trex
D3DGS
Rate (MB)
73.53
44.44
21.64
56.72
PSNR
24.92
42.63
44.24
38.04
SSIM
0.943
0.995
0.995
0.993
LPIPS
0.044
0.005
0.007
0.010
4DHexPlane
Rate (MB)
33.02
22.45
19.42
31.83
PSNR
25.05
37.71
38.06
33.69
SSIM
0.938
0.988
0.990
0.984
LPIPS
0.056
0.016
0.014
0.022
4DGS
Rate (MB)
423.43
301.36
347.14
751.50
PSNR
24.23
36.62
38.29
29.93
SSIM
0.908
0.982
0.985
0.976
LPIPS
0.096
0.019
0.017
0.029
Ours
Rate (MB)
2.04
0.93
0.70
0.88
PSNR
24.57
41.12
43.53
38.25
SSIM
0.939
0.993
0.994
0.993
LPIPS
0.049
0.009
0.008
0.011
11

<!-- page 12 -->
Table 2: Per-scene quantitative comparison on NeRF-DS dataset. We acquire the baseline results by
running their official codes.
Scene
as
basin
bell
cup
plate
press
sieve
D3DGS
Rate (MB)
47.24
68.67
88.15
52.32
57.22
49.09
52.99
PSNR
25.91
19.64
25.06
24.74
20.14
25.37
25.45
SSIM
0.881
0.788
0.841
0.889
0.806
0.859
0.872
LPIPS
0.183
0.189
0.164
0.155
0.221
0.193
0.150
4DHexPlane
Rate (MB)
52.36
60.16
65.58
52.04
103.00
60.07
50.51
PSNR
23.25
18.65
21.50
22.66
16.38
20.88
24.28
SSIM
0.775
0.704
0.729
0.833
0.613
0.643
0.812
LPIPS
0.258
0.264
0.283
0.187
0.418
0.368
0.195
4DGS
Rate (MB)
204.09
128.75
396.73
280.38
124.93
275.61
164.99
PSNR
24.97
19.20
23.96
23.74
19.74
25.54
23.84
SSIM
0.845
0.750
0.811
0.854
0.786
0.838
0.815
LPIPS
0.248
0.299
0.236
0.219
0.295
0.263
0.260
Ours
Rate (MB)
0.67
0.66
0.90
0.67
0.66
0.75
0.62
PSNR
26.72
19.95
25.38
25.07
21.02
25.76
25.37
SSIM
0.885
0.802
0.845
0.896
0.826
0.864
0.871
LPIPS
0.181
0.204
0.174
0.157
0.211
0.202
0.160
Table 3: Performance of our method under varying bitrates.
Dataset
D-NeRF
NeRF-DS
Rate (MB)
1.03
0.78
0.71
0.57
0.70
0.65
0.57
0.54
PSNR
38.10
37.73
37.38
35.10
24.18
23.93
23.60
23.34
SSIM
0.984
0.984
0.984
0.978
0.856
0.849
0.832
0.822
LPIPS
0.018
0.018625
0.0195
0.031
0.184
0.204
0.246
0.267
References
[1] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable
3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 20331–20341, 2024.
[2] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-time photorealistic dynamic scene
representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642,
2023.
[3] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu,
Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
20310–20320, 2024.
[4] Gary J Sullivan, Jens-Rainer Ohm, Woo-Jin Han, and Thomas Wiegand. Overview of the high
efficiency video coding (hevc) standard. IEEE Transactions on circuits and systems for video
technology, 22(12):1649–1668, 2012.
[5] Detlev Marpe, Heiko Schwarz, and Thomas Wiegand. Context-based adaptive binary arithmetic
coding in the h. 264/avc video compression standard. IEEE Transactions on circuits and systems
for video technology, 13(7):620–636, 2003.
[6] Zhiwen Yan, Chen Li, and Gim Hee Lee. Nerf-ds: Neural radiance fields for dynamic spec-
ular objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 8285–8295, 2023.
12

<!-- page 13 -->
[7] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf:
Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 10318–10327, 2021.
[8] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian
splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
[9] Shizun Wang, Xingyi Yang, Qiuhong Shen, Zhenxiang Jiang, and Xinchao Wang. Gflow:
Recovering 4d world from monocular video. In Proceedings of the AAAI Conference on
Artificial Intelligence, volume 39, pages 7862–7870, 2025.
[10] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi.
Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 4220–4230, 2024.
[11] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen.
4d-rotor gaussian splatting: towards efficient novel view synthesis for dynamic scenes. In ACM
SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.
[12] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time
dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 8508–8520, 2024.
[13] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. Dynmf: Neural motion factorization
for real-time dynamic view synthesis with 3d gaussian splatting. In European Conference on
Computer Vision, pages 252–269. Springer, 2024.
[14] Kai Katsumata, Duc Minh Vo, and Hideki Nakayama. A compact dynamic 3d gaussian
representation for real-time dynamic view synthesis. In European Conference on Computer
Vision, pages 394–412. Springer, 2024.
[15] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 130–141,
2023.
[16] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Light-
gaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in
neural information processing systems, 37:140138–140158, 2024.
[17] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d
gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 21719–21728, 2024.
[18] K Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi Koohpayegani, and Hamed Pirsi-
avash. Compact3d: Compressing gaussian splat radiance field models with vector quantization.
arXiv preprint arXiv:2311.18159, 4, 2023.
[19] Simon Niedermayr, Josef Stumpfegger, and Rüdiger Westermann. Compressed 3d gaussian
splatting for accelerated novel view synthesis. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 10349–10358, 2024.
[20] Panagiotis Papantonakis, Georgios Kopanas, Bernhard Kerbl, Alexandre Lanvin, and George
Drettakis. Reducing the memory footprint of 3d gaussian splatting. Proceedings of the ACM on
Computer Graphics and Interactive Techniques, 7(1):1–17, 2024.
[21] Shuzhao Xie, Weixiang Zhang, Chen Tang, Yunpeng Bai, Rongwei Lu, Shijia Ge, and Zhi Wang.
Mesongs: Post-training compression of 3d gaussians via efficient attribute transformation. In
European Conference on Computer Vision, pages 434–452. Springer, 2024.
[22] Henan Wang, Hanxin Zhu, Tianyu He, Runsen Feng, Jiajun Deng, Jiang Bian, and Zhibo Chen.
End-to-end rate-distortion optimized 3d gaussian representation. In European Conference on
Computer Vision, pages 76–92. Springer, 2024.
13

<!-- page 14 -->
[23] Yangming Zhang, Wenqi Jia, Wei Niu, and Miao Yin. Gaussianspa: An" optimizing-sparsifying"
simplification framework for compact and high-quality 3d gaussian splatting. arXiv preprint
arXiv:2411.06019, 2024.
[24] Yifei Liu, Zhihang Zhong, Yifan Zhan, Sheng Xu, and Xiao Sun. Maskgaussian: Adaptive 3d
gaussian representation from probabilistic masks. arXiv preprint arXiv:2412.20522, 2024.
[25] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. Eagles: Efficient accelerated 3d
gaussians with lightweight encodings. In European Conference on Computer Vision, pages
54–71. Springer, 2024.
[26] Minye Wu and Tinne Tuytelaars. Implicit gaussian splatting with efficient multi-level tri-plane
representation. arXiv preprint arXiv:2408.10041, 2024.
[27] Wieland Morgenstern, Florian Barthel, Anna Hilsmann, and Peter Eisert. Compact 3d scene
representation via self-organizing gaussian grids. In European Conference on Computer Vision,
pages 18–34. Springer, 2024.
[28] Soonbin Lee, Fangwen Shu, Yago Sanchez, Thomas Schierl, and Cornelius Hellge. Compression
of 3d gaussian splatting with optimized feature planes and standard video codecs. arXiv preprint
arXiv:2501.03399, 2025.
[29] Yu-Ting Zhan, Cheng-Yuan Ho, Hebi Yang, Yi-Hsin Chen, Jui Chiu Chiang, Yu-Lun Liu, and
Wen-Hsiao Peng. Cat-3dgs: A context-adaptive triplane approach to rate-distortion-optimized
3dgs compression. arXiv preprint arXiv:2503.00357, 2025.
[30] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-
gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 20654–20664, 2024.
[31] Abdullah Hamdi, Luke Melas-Kyriazi, Jinjie Mai, Guocheng Qian, Ruoshi Liu, Carl Vondrick,
Bernard Ghanem, and Andrea Vedaldi. Ges: Generalized exponential splatting for efficient
radiance field rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 19812–19822, 2024.
[32] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Hac: Hash-grid
assisted context for 3d gaussian splatting compression. In European Conference on Computer
Vision, pages 422–438. Springer, 2024.
[33] Xiangrui Liu, Xinju Wu, Pingping Zhang, Shiqi Wang, Zhu Li, and Sam Kwong. Compgs:
Efficient 3d scene representation via compressed gaussian splatting. In Proceedings of the 32nd
ACM International Conference on Multimedia, pages 2936–2944, 2024.
[34] Yufei Wang, Zhihao Li, Lanqing Guo, Wenhan Yang, Alex Kot, and Bihan Wen. Contextgs:
Compact 3d gaussian splatting with anchor level context model. Advances in neural information
processing systems, 37:51532–51551, 2024.
[35] Yihang Chen, Qianyi Wu, Mengyao Li, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Fast
feedforward 3d gaussian splatting compression. arXiv preprint arXiv:2410.08017, 2024.
[36] Lei Liu, Zhenghao Chen, Wei Jiang, Wei Wang, and Dong Xu. Hemgs: A hybrid entropy model
for 3d gaussian splatting data compression. arXiv preprint arXiv:2411.18473, 2024.
[37] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Hac++: Towards
100x compression of 3d gaussian splatting. arXiv preprint arXiv:2501.12255, 2025.
[38] Thomas Wiegand, Gary J Sullivan, Gisle Bjontegaard, and Ajay Luthra. Overview of the h.
264/avc video coding standard. IEEE Transactions on circuits and systems for video technology,
13(7):560–576, 2003.
[39] Benjamin Bross, Ye-Kui Wang, Yan Ye, Shan Liu, Jianle Chen, Gary J Sullivan, and Jens-
Rainer Ohm. Overview of the versatile video coding (vvc) standard and its applications. IEEE
Transactions on Circuits and Systems for Video Technology, 31(10):3736–3764, 2021.
14

<!-- page 15 -->
[40] Johannes Ballé, Valero Laparra, and Eero P Simoncelli. End-to-end optimized image compres-
sion. arXiv preprint arXiv:1611.01704, 2016.
[41] Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational
image compression with a scale hyperprior. arXiv preprint arXiv:1802.01436, 2018.
15
