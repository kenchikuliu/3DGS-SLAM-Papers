<!-- page 1 -->
PointGauss: Point Cloud-Guided Multi-Object
Segmentation for Gaussian Splatting
Wentao Sun
University of Waterloo
N2L 3G1 Waterloo, Canada
w27sun@uwaterloo.ca
Hanqing Xu
East China Normal University
200050, Shanghai, China
51273901140@stu.ecnu.edu.cn
Quanyun Wu
University of Waterloo
N2L 3G1 Waterloo, Canada
q34wu@uwaterloo.ca
Dedong Zhang
University of Waterloo
N2L 3G1 Waterloo, Canada
dedong.zhang@uwaterloo.ca
Yiping Chen
Sun Yat-Sen University
519082 Zhuhai, China
chenyp79@mail.sysu.edu.cn
Lingfei Ma
East China Normal University
200050, Shanghai, China
l53ma@cufe.edu.cn
John S. Zelek
University of Waterloo
N2L 3G1 Waterloo, Canada
jzelek@uwaterloo.ca
Jonathan Li
University of Waterloo
N2L 3G1 Waterloo, Canada
junli@uwaterloo.ca
Abstract
We introduce PointGauss, a novel point cloud-guided framework for real-time multi-
object segmentation in Gaussian Splatting representations. Unlike existing methods
that suffer from prolonged initialization and limited multi-view consistency, our
approach achieves efficient 3D segmentation by directly parsing Gaussian primi-
tives through a point cloud segmentation-driven pipeline. The key innovation lies
in two aspects: (1) a point cloud-based Gaussian primitive decoder that generates
3D instance masks within 1 minute, and (2) a GPU-accelerated 2D mask rendering
system that ensures multi-view consistency. Extensive experiments demonstrate
significant improvements over previous state-of-the-art methods, achieving perfor-
mance gains of 1.89 to 31.78% in multi-view mIoU, while maintaining superior
computational efficiency. To address the limitations of current benchmarks (single-
object focus, inconsistent 3D evaluation, small scale, and partial coverage), we
present DesktopObjects-360, a novel comprehensive dataset for 3D segmentation in
radiance fields, featuring: (1) complex multi-object scenes, (2) globally consistent
2D annotations, (3) large-scale training data (over 27 thousand 2D masks), (4) full
360° coverage, and (5) 3D evaluation masks. (Code)
Preprint. Under review.
arXiv:2508.00259v1  [cs.CV]  1 Aug 2025

<!-- page 2 -->
1
Introduction
3D Gaussian Splatting (3DGS) [1] and Neural Radiance Fields (NeRF) [2], along with their variants,
have significantly advanced the fast and high-fidelity modeling of 3D scenes, enabling a wide range
of applications in augmented reality [3, 4], autonomous systems [5], and robotic navigation [6–
8]. While these methods demonstrate remarkable performance in scene reconstruction, achieving
accurate and semantically meaningful understanding of 3D environments—especially for downstream
tasks—remains a substantial challenge. Among these tasks, 3D scene segmentation is particularly
critical, as it enables the extraction of instance-level object information, laying the foundation for
comprehensive spatial reasoning and interaction.
In this context, NeRF-based scene segmentation has been extensively studied [9–12]. A represen-
tative example is SA3D [12], which adapts the 2D Segment Anything Model (SAM)[13] to 3D by
leveraging radiance field representations. In contrast, 3DGS-based segmentation remains relatively
underexplored. Recent efforts [14–18] primarily rely on contrastive learning or 2D model distillation.
For example, Feature3DGS [18] proposes a distillation framework that transfers SAM’s semantic
features to Gaussian attributes, followed by 2D reprojection for segmentation. Despite these ad-
vances, existing methods face two major shortcomings: (1) Insufficient utilization of 3D geometry:
By primarily converting 2D masks or features into 3D space, these methods fail to fully leverage
the inherent spatial coherence of 3D structures, often leading to inconsistent segmentation across
viewpoints. (2) Excessive architectural complexity: Multi-stage distillation and cross-dimensional
feature alignment introduce significant training overhead and hinder practical deployment.
Additionally, existing benchmarks for 3D segmentation in Gaussian Splatting scenes, such as
NVOS [19], Spin-NeRF [20], and LERF-Mask [14], exhibit several critical limitations: (1) Single-
object focus: These benchmarks primarily target single-object, multi-view segmentation, lacking the
complexity needed to evaluate multi-object scenarios. (2) The small dataset sizes are inadequate for
training and evaluating supervised learning methods. (3) Limited viewing coverage: Constrained
camera trajectories with partial or no 360° coverage hinder the assessment of segmentation robust-
ness under diverse viewpoints. (4) Lack of 3D ground truth: The absence of comprehensive 3D
segmentation annotations prevents thorough and accurate evaluation.
To address these challenges, we propose PointGauss, a novel point cloud-guided framework for real-
time multi-object segmentation in Gaussian Splatting representations, along with a new benchmark,
DesktopObjects-360, designed for 3D segmentation tasks. Our core innovation lies in directly
operating on Gaussian primitives using point cloud segmentation models. This strategy enables rapid
initialization of 3D instance masks while preserving native spatial coherence – advantages that are
difficult to achieve with 2D foundation model-based methods. In addition, we introduce a GPU-
accelerated rendering pipeline that efficiently propagates segmentation results to novel viewpoints,
ensuring consistent multi-object segmentation across diverse perspectives. The DesktopObjects-360
benchmark advances evaluation standards by providing: complex multi-object scenes with occlusions,
globally consistent 2D ground truth, large-scale supervised data (over 27 thousand masks), complete
360° coverage, and 3D masks for comprehensive assessment. Experimental results demonstrate that
PointGauss achieves 1.89-31.78% gains in multi-view mIoU over state-of-the-art methods, while
offering 200-300× faster initialization and maintaining real-time performance. These results highlight
the effectiveness of integrating point cloud processing with Gaussian Splatting for efficient and
accurate 3D scene understanding.
2
Related Work
2.1
3D Gaussian Representations
Recent progress in 3D Gaussian Splatting (3DGS) has led to diverse methodological improvements,
which we categorize into seven key categories following [21]: (i) sparse view reconstruction [22–25],
(ii) memory-efficient representations [26–28], (iii) photorealistic rendering techniques [29–32], (iv)
optimization strategies [33–36], (v) attribute learning [14, 37, 18], (vi) hybrid architectures [38, 39],
and (vii) rendering innovations [40, 41].
Among these, optimization remains a particularly active area, with recent works addressing the
persistent issue of imbalanced reconstruction quality across different regions of a scene. Geometry-
aware approaches [35, 36] have demonstrated notable success by incorporating structural priors into
2

<!-- page 3 -->
Gaussian Scene + Clicks
Multi-view Consistent Segmentation
Gaussian Scene with Instacne Feature
Coarse 2D Masks
Refined 2D Masks
Gaussian Mean Points
3D Mask
3D Segmentation
< 1 min
DesktopObjects-360:Desk. 5
Figure 1: We propose PointGauss, a novel point cloud-guided framework for real-time multi-object
segmentation in Gaussian Splatting representations, along with a new benchmark, DesktopObjects-
360, designed for 3D segmentation tasks. Our method produces 3D instance segmentation results in
under one minute.
the optimization process. For example, 2DGS[35] introduces planar Gaussian primitives to better
capture surface continuity while reducing computational overhead.
2.2
Segmentation for Gaussian Splatting
Several notable works have advanced this field, such as Feature3DGS [18], Gaussian Grouping [14],
OmniSeg3D [17], SAGA [15], and Click-Gaussian [16]. These methods generally follow a similar
pipeline: they extract 2D masks or features using SAM, lift the information into 3D space via
contrastive learning or distillation, and then project the resulting 3D segmentation features back to
2D to enable segmentation from novel viewpoints. However, each method varies in how it processes
SAM-derived 2D information and faces distinct limitations.
OmniSeg3D addresses 2D masks’ view inconsistency through hierarchical contrastive learning, yet
struggles with scale adaptability. SAGA introduces scale-gated affinity features for multi-level cues
but overlooks 2D mask conflicts. Click-Gaussian improves multi-view consistency with feature
smoothing but is limited in handling multi-instance Gaussian primitives effectively.
Beyond individual shortcomings, these approaches share several common challenges: underutilization
of the inherent 3D spatial structure, increased architectural complexity due to multi-stage pipelines,
and a strong dependency on 2D segmentation models.
These limitations motivate our proposed approach: a point cloud-guided 3D segmentation framework
that leverages the native geometry of Gaussian primitives, eliminates reliance on 2D masks, and
improves both efficiency and multi-view consistency.
2.3
Point Cloud Segmentation
Point cloud segmentation encompasses both semantic segmentation and instance segmentation.
Semantic segmentation methods can be categorized based on their processing strategies. Point-
based approaches, such as PointNet++ [42], operate directly on raw 3D coordinates. Voxel-based
methods [43] discretize point clouds into volumetric grids for structured processing, while multi-view
techniques [44] project 3D data onto 2D image planes to leverage mature 2D CNN architectures.
Hybrid architectures, such as PointTransformerV3 (PTV3) [45], combine these strategies to improve
scalability and efficiency for large-scale scenes.
Instance segmentation builds upon semantic segmentation by distinguishing individual object in-
stances. Proposal-based methods, such as 3D-SIS [46], generate region proposals to localize instances,
while CRSNet [47] incorporate user-provided cues. Clustering-based methods, like SGPN [48], learn
similarity matrices to group points into distinct instances.
3

<!-- page 4 -->
Predicted 
Images
Clicks
Prompts
Prompt 
Encoder
Gaussian 
Decoder
3D Gaussian 
Model
Reconstru
ction
Splatting 
Rendering
Gaussian Splatting Pipeline
Input  Images
Splatting 
Projection
Gaussian Splatting Pipeline
PointGauss Workflow
Visualize
Segmentation 
Results
W
G
G*
inst_label
Concatenate
PointGauss
Figure 2: PointGauss Architecture. The reconstruction branch (top) builds the Gaussian Model G via
differentiable splatting. The segmentation branch (bottom) processes click prompts using a prompt
encoder to obtain the feature W, fuses these features with the Gaussian primitives to generate G∗,
then uses a Gaussian decoder to produce the 3D segmentation labels (inst_label), and finally obtains
2D segmentation masks through splatting projection.
Our approach adopts a point-based segmentation strategy while leveraging the unique properties
of 3D Gaussian Splatting (3DGS) to enhance spatial coherence and enable real-time multi-object
segmentation.
3
Proposed Method
As shown in Fig. 2, PointGauss consists of three modules: a Prompt Encoder to fuse 2D interactions
with 3D Gaussian attributes, a Gaussian Decoder to performs 3D instance segmentation respecting
Gaussian structures, and a Splatting Projection for view-consistent rendering.
3.1
Problem Formulation
Given a pre-trained gaussian scene G = {gi}N
i=1 (trained by 2DGS [35]) where each Gaussian
primitive gi = (µi, Σi, ci) contains position µi ∈R3, anisotropic covariance Σi ∈S3
++, and
appearance attributes ci, and a set of user interactions C = {cj}M
j=1 (clicks) in view Vk, we aim to
learn a mapping function:
F : (G, C) →(R3, R2)
(1)
where R3 = {si}N
i=1 with si ∈{0, 1}K represents the K-instance 3D segmentation, and R2 =
Ψ(R3, Vk) denotes the view-consistent 2D projection through our differentiable renderer Ψ(·). The
mapping must satisfy two key properties: Interaction Faithfulness: R3 should respect all C annotations
in their original view Vk. 3D Consistency: Segmentation labels must remain coherent across arbitrary
novel views.
3.2
Prompt Encoder
Our prompt encoder converts user clicks into meaningful geometric features. When a user clicks on
the screen position p = (u, v), we first cast a viewing ray r(t) = o + td through the camera center o
with direction d computed via perspective projection.
To establish spatial reference, we compute the intersection point between the ray and the Gaussian
primitives using:
t∗= arg min
t


t

X
j
αj(t) > τ


,
(2)
where αj(t) denotes the accumulated opacity of j-th Gaussian along the ray at depth t, and τ = 0.9
is our opacity threshold. The resulting 3D position pclick = r(t∗) serves as the reference anchor
point.
4

<!-- page 5 -->
For each Gaussian primitive Gi with mean µi, we then compute its spatial relevance weight through:
wi = exp

−∥µi −pclick∥2
2σ2

,
(3)
where σ controls the spatial sensitivity (empirically set to 0.15 m). This formulation generates a
smooth attention map over all Gaussians, with values decaying exponentially based on their Euclidean
distance to the interaction point. The computed weights W = {wi}N
i=1 are concatenated as additional
feature channels to the original Gaussian attributes.
3.3
Gaussian Decoder
Given the feature-augmented Gaussians G∗= {g∗
i |g∗
i = [fi; xi, yi, zi]}N
i=1 from the Prompt Encoder,
our Gaussian Decoder performs instance-level segmentation through four-stage processing:
Coarse Region Cropping. We first construct a cylindrical cropping volume Vc ∈R3 centered at the
user’s click position pclick, with radius r = 3.0m and height h = 3m empirically set to encompass
typical interactive segmentation scenarios. The valid Gaussians are selected by:
Gc = {g∗
i |∥(xi, yi) −pxy
click∥2 ≤r ∧|zi −pz
click| ≤h/2}
(4)
Adaptive Point Cloud Batching. To handle varying point densities, we implement dynamic batch
partitioning when processing through the network. For point cloud Pc converted from Gc:
Pbatch
c
=
RandomSplit(Pc, k),
if |Pc| > 8192
Pc,
otherwise
(5)
where k = ⌈|Pc|/8192⌉ensures each batch contains at most 8192 points. The segmentation logits
are aggregated through max-pooling across batches.
Network Backbone. We use a point-based network to classify each Gaussian primitive as foreground
or background. PointTransformerV3 (PTV3) [45], a state-of-the-art point cloud segmentation model,
serves as our backbone due to its strong local and global feature capture. The point cloud Pc is
processed through an encoder-decoder architecture, producing semantic scores for Gaussians in the
region of interest (ROI).
Instance Label Assignment. The final instance labels inst_label ∈{0, 1}N are obtained by
comparing the foreground and background probabilities from semantic segmentation:
inst_labeli = I

pfg
i
> pbg
i

(6)
where pfg
i
and pbg
i denote the foreground and background probabilities of point i, respectively, as
predicted by the semantic segmentation network. I(·) is the indicator function, which outputs 1 if the
condition is true (foreground) and 0 otherwise (background).
3.4
Splatting Projection
Given the instance labels L3D = {lk}K
k=1 generated by the Gaussian Decoder for K Gaussians, our
splatting projection module efficiently renders instance-aware 2D masks Mi ∈RH×W for arbitrary
viewpoint i through rasterization.
Instance-Aware Rendering. We project instance labels from Gaussian primitives to 2D segmentation
masks through geometric-aware rasterization. For each pixel (u, v), we determine its instance label
by analyzing the spatial distribution of Gaussians in the image plane. The projection process follows:
M(u, v) =
ck
if ∃k ∈N(u, v), ρ2
k ≤τ and ck > 0
0
otherwise
(7)
where N(u, v) denotes Gaussians influencing pixel (u, v), ρ2
k = (xk −u)2 + (yk −v)2 measures
the squared distance between the Gaussian center (xk, yk) and target pixel, τ = 4.0 is the distance
threshold, and ck is the instance label of the k-th Gaussian. The rasterization prioritizes the first
valid instance label meeting the spatial constraint during the rendering pass, ensuring efficient label
propagation while maintaining spatial coherence.
5

<!-- page 6 -->
Scene 1
Scene 2
Scene 3
Scene 5
Scene 4
Scene 6
Figure 3: Visualization of the DesktopObjects Benchmark. For each scene (two rows), we show
three representative RGB images (top row), their corresponding 2D instance masks (bottom row),
and 3D instance segmentation visualizations, displayed as Gaussian mean values (left).
Feature3DGS
OmniSeg3D
GARField
SAGA
PointGauss (ours)
Groundtruth
Desk. 5
Figure 4: Segmentation performance on Desk. 5. Our method produces more precise segmentation
masks (color-coded regions) compared to the baselines. (Red dashed circles indicate regions with
segmentation errors.)
This formulation leverages the inherent geometric properties of the Gaussian scene by directly
associating instance labels with Gaussian spatial positions. The distance threshold τ effectively filters
out distant Gaussian primitives while preserving sharp boundaries through hard label assignment.
The complete algorithm (see Alg. 1) is in the supplementary material (see Sec. B).
Post-processing for Mask Refinement. To address common segmentation artifacts such as holes,
fragmentation, and jagged edges, we employ a three-stage refinement pipeline. First, morphology-
based smoothing applies multi-scale binary closing to eliminate discontinuities while preserving
object shapes. Second, edge-constrained hole filling selectively fills enclosed gaps near object
boundaries to maintain structural accuracy. Finally, we retain the largest connected component per
instance to suppress noise and outliers. The full algorithm is detailed in Alg. 2, with additional
implementation details provided in the Appendix (Sec. B).
3.5
Summary
PointGauss establishes a new paradigm for segmentation in Gaussian Splatting scenes by directly
operating on Gaussian primitives, ensuring native compatibility with Gaussian Splatting rendering and
preserving geometric fidelity. The framework comprises three core components: (1) a cross-modal
prompt encoder that bridges 2D user interactions and 3D Gaussian representations, (2) a Gaussian
decoder for generating segmentation outputs, and (3) a rendering module that ensures view-consistent
results. This architecture achieves real-time performance while enabling accurate, geometry-aware
interactive segmentation.
6

<!-- page 7 -->
Magnified Objects
Desk. 5
GT_segmentation
Pr_Segmentation
Scene
Figure 5: 3D Instance Segmentation Visualization. Results on Desk. 5. The columns show: (1)
a scene overview, (2) ground truth 3D instance labels, (3) our PointGauss predictions, and (4-6)
magnified views of objects. (Blue points indicate the background; colored points represent different
instances. Note that the tape appears dark blue but is not part of the background.)
Table 1: Experimental results on Desk.1-3 datasets (%). 3D represents 3D IoU, 2D represents 2D
mIoU, and OA denotes Overall Accuracy.
Method
Desk.1
Desk.2
Desk.3
3D
2D
OA
3D
2D
OA
3D
2D
OA
Feature3DGS [18]
–
38.36
71.49
–
32.72
62.48
–
42.27
71.45
OmniSeg3D [17]
–
53.09
81.11
–
38.85
75.49
–
46.64
74.25
GARField [17]
–
66.22
92.52
–
27.27
82.56
–
71.48
91.86
SAGA [15]
–
68.72
85.67
–
66.55
86.86
–
69.42
87.31
PointGauss(Ours)
69.40
84.33
95.90
82.46
86.91
96.67
73.38
85.85
94.54
4
Dataset
Existing benchmarks for 3D segmentation in Gaussian Splatting scenes, including NVOS [19], Spin-
NeRF [20], and LERF-Mask [14], exhibit several critical limitations: (1) single-object focus, (2)
insufficient scale, (3) constrained viewing range, (4) the absence of 3D ground truth annotations (see
Sec. 1). These constraints are particularly problematic for PointGauss, which requires a robust 3D
point cloud segmentation model. Current NeRF and 3DGS datasets fail to provide sufficient training
data for such models, making existing benchmarks unsuitable for our needs.
To address these limitations, we propose DesktopObjects-360 (see Fig. 3), a new benchmark
featuring: (1) multi-object segmentation in complex scenes, (2) Globally consistent 2D segmentation
ground truth, (3) large-scale data suitable for supervised learning, (4) complete 360° viewpoint
coverage, and (5) 3D masks for global evaluation.
DesktopObjects-360 establishes a rigorous evaluation framework for 3D segmentation algorithms in
Gaussian Splatting environments, overcoming the limitations of previous benchmarks while providing
comprehensive metrics for method comparison. Specifically, the DesktopObjects-360 is a carefully
constructed indoor dataset of six scenes, each containing 7-10 objects, totaling 3,364 multi-view
images with high-quality 2D and 3D annotations. It provides 56 fully annotated 3D instances under
varied layouts and occlusions. Additional details, including the data collection pipeline, are provided
in the Appendix (see Sec. A). The dataset will be hosted on Harvard Dataverse with a DOI, accessible
via a github repository under a CC BY 4.0 license with a simple user agreement.
ID: 1
ID: 1
ID: 1
ID: 1
ID: 2
ID: 2
ID: 2
ID: 2
ID: 3
ID: 3
ID: 3
ID: 3
ID: 4
ID: 4
ID: 4
ID: 4
ID: 5
ID: 5
ID: 5
ID: 5
ID: 6
ID: 6
ID: 6
ID: 6
ID: 7
ID: 7
ID: 7
ID: 7
Figure 6: Viewpoint-Robust Instance IDs. Consistent color/ID assignments (#1-#7) across 4 view-
points validate our 3D-consistent segmentation.
7

<!-- page 8 -->
Table 2: Experimental results on Desk.4-6 datasets (%). 3D represents 3D IoU, 2D represents 2D
mIoU, and OA denotes Overall Accuracy.
Method
Desk.4
Desk.5
Desk.6
3D
2D
OA
3D
2D
OA
3D
2D
OA
Feature3DGS [18]
–
34.49
70.43
–
32.54
66.16
–
19.65
62.42
OmniSeg3D [17]
–
56.35
89.35
–
38.94
77.99
–
32.02
82.02
GARField [11]
–
64.63
93.32
–
58.82
90.61
–
65.21
97.44
SAGA [15]
–
87.47
97.52
–
79.40
95.41
–
59.34
92.83
PointGauss(Ours)
78.05
89.36
97.68
80.94
85.55
96.76
77.67
91.12
99.31
Table 3: 3D preparation (3D Prep.) and per-frame inference time comparison. * indicates the time of
post-processing
Method
3D Prep. (min)
Per-Frame (ms)
GARField [11]
45
3232
OmniSeg3D [17]
37
463
Feature3DGS [18]
35
510
SAGA [15]
32
31
PointGauss (Ours)
0.13
5+388*
5
Experiments
In this section, we evaluate the performance of the proposed method on DesktopObjects-360, focusing
on segmentation accuracy, algorithm deployment and execution efficiency, and overall robustness.
5.1
Experimental Setup
The experiments are conducted on a Linux system equipped with an 11th Gen Intel(R) Core(TM)
i5-11400F CPU running at 2.60GHz, 64 GB of RAM operating at 3200 MHz, and an NVIDIA
GeForce RTX 4090 GPU.
5.2
Segmentation Performance
We evaluate the performance of PointGauss on the DesktopObjects-360 dataset, using five scenes
for training and one scene for testing. Our approach is compared against four baselines: SAGA,
Feature3DGS, OmniSeg3D, and GARField [11]. We assess the methods using three metrics: 2D mean
Intersection over Union (2D mIoU), 3D Intersection over Union (3D IoU), and Overall Accuracy
(OA). The 3D IoU measures segmentation accuracy based on Gaussian primitives, serving as an
effective metric for evaluating performance in 3D space segmentation. For more details on metric
selection, please refer to the Appendix (see Sec. C).
As shown in Table 1 and Table 2, our experimental evaluation demonstrates that PointGauss con-
sistently outperforms existing methods across all six datasets in both 2D segmentation accuracy
(84.33-91.12% 2D mIoU) and overall scene understanding (94.54-99.31% OA). While comparison
methods (Feature3DGS, GARField, SAGA, and OmniSeg3D) are fundamentally constrained to
2D evaluation due to their architectural limitations, our approach uniquely supports comprehensive
3D metric assessment while maintaining superior 2D performance. Notably, PointGauss achieves
substantial improvements over the strongest baseline (SAGA), with gains of 1.89-31.78% in 2D mIoU
and 0.16-6.48% in OA across different datasets. These results validate that our 3D-aware framework
not only provides meaningful 3D understanding (69.40-82.46% 3D IoU) but also delivers more robust
and accurate 2D segmentation than existing methods. Additional experiments are provided in the
Appendix (see Sec. E.2).
Fig. 4 presents multi-view segmentation comparisons, where our method generates more precise
segmentation masks than the baselines. Fig. 5 illustrates PointGauss’s 3D instance segmentation
results, demonstrating its ability to produce accurate 3D masks for Gaussian primitives. The viewpoint
consistency analysis in Fig. 6 highlights PointGauss’s capability to maintain consistent instance IDs
8

<!-- page 9 -->
Without Post-processing
With Post-processing
View
Figure 7: Post-processing Module Analysis. The middle part, without post-processing, exhibits
instance fragmentation (scattered colorful points). The right part, with our post-processing, maintains
instance integrity (solid-colored regions).
Table 4: Effect of the Number of Clicks on Segmentation Performance
Num. of Clicks
3D IoU (%)
2D mIoU (%)
OA (%)
5
68.72
90.01
99.22
10
73.29
90.99
99.29
15
75.17
90.80
99.29
20
77.67
91.12
99.31
25
80.61
91.18
99.30
30
81.83
90.95
99.29
across different viewpoints, ensuring stable segmentation even under occlusion (e.g., ID#4 occluded
by ID#5).
These results confirm PointGauss’s superior balance between 3D geometric consistency and view-
dependent accuracy, essential for applications like robotic manipulation and augmented reality.
5.3
Time Efficiency Analysis
We evaluated the time efficiency of SAGA, Feature3DGS, OmniSeg3D, GARField, and PointGauss
on the Desk. 6 dataset, focusing on 3D information preparation and per-frame processing. Unlike
existing baselines that rely on pre-trained 2D foundation models (e.g., SAM), PointGauss utilizes
point cloud segmentation models. Since the trained model can be reused across multiple Gaussian
scenes, we exclude its training time (approximately 9 minutes) when comparing computational
efficiency on individual scenes.
As shown in Table 3, PointGauss achieves near-instant 3D scene preparation in 0.13 minutes,
compared to 32-45 minutes for other methods—a 200× to 300× speedup. This eliminates the need
for offline preprocessing, enabling real-time interaction. For per-frame inference, SAGA excels
with 31 ms, while PointGauss requires 5 ms (core inference) + 388 ms (post-processing). Although
post-processing introduces some latency, our core inference is highly competitive (e.g., 6× faster than
SAGA).
Our efficiency stems from directly operating on Gaussian primitives, while other methods must
process hundreds of input images during the preparation stage.
5.4
Impacts of Clicks
To assess the impact of click quantity on segmentation performance, we conduct experiments on
Desk.6 dataset with varying number of clicks. Table 4 shows that increasing the number of clicks
improves 3D IoU (from 68.72% to 81.83%), with diminishing returns beyond 25 clicks. In contrast,
2D mIoU remains relatively stable ( 91%), indicating that 2D segmentation requires fewer clicks.
Overall Accuracy (OA) stays consistently high (>99.2%), demonstrating robustness to variations in
click count. Optimal performance is observed with 20-25 clicks.
9

<!-- page 10 -->
5.5
Impacts of Post-processing
To evaluate the impact of post-processing, we conduct an ablation study (Fig. 7). Without post-
processing, the segmentation mask contains holes and artifacts, leading to incomplete and noisy
rendered outputs. In contrast, applying post-processing results in smoother and more complete masks,
producing clearer and more accurate segmentation. These results highlight the essential role of
post-processing in enhancing segmentation quality.
6
Conclusion
We present PointGauss, a novel point cloud-guided framework for real-time multi-object segmen-
tation in Gaussian Splatting representations, along with a new benchmark, DesktopObjects-360,
for 3D segmentation tasks. By directly processing Gaussian primitives, our method eliminates the
need for time-consuming distillation or contrastive learning, while ensuring cross-view consistency.
Experimental results show that PointGauss outperforms state-of-the-art methods in both segmentation
and time efficiency. In future work, we plan to explore more efficient post-processing techniques.
References
[1] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D gaussian splatting for
real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
[2] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren
Ng. Nerf: representing scenes as neural radiance fields for view synthesis. Commun. ACM, 65(1):99–106,
December 2021.
[3] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng
Gao, Yin Yang, and Chenfanfu Jiang. Vr-gs: A physical dynamics-aware interactive gaussian splatting
system in virtual reality. In ACM SIGGRAPH 2024 Conference Papers, 2024.
[4] Hyunjeong Kim and In-Kwon Lee. Is 3dgs useful?: Comparing the effectiveness of recent reconstruction
methods in vr. In 2024 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pages
71–80, 2024.
[5] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Driving-
gaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In 2024
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21634–21643, 2024.
[6] Rui Jin, Yuman Gao, Yingjian Wang, Yuze Wu, Haojian Lu, Chao Xu, and Fei Gao. Gs-planner: A
gaussian-splatting-based planning framework for active high-fidelity reconstruction. In 2024 IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS), pages 11202–11209, 2024.
[7] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. Gs-slam: Dense
visual slam with 3D gaussian splatting. In 2024 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 19595–19604, 2024.
[8] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. Gaussian splatting slam. In 2024
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18039–18048, 2024.
[9] Xavier Timoneda, Markus Herb, Fabian Duerr, Daniel Goehring, and Fisher Yu. Multi-modal nerf self-
supervision for lidar semantic segmentation. In 2024 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), pages 12939–12946, 2024.
[10] Yichen Liu, Benran Hu, Chi-Keung Tang, and Yu-Wing Tai. Sanerf-hq: Segment anything for nerf in
high quality. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages
3216–3226, 2024.
[11] Chung Min Kim, Mingxuan Wu, Justin Kerr, Ken Goldberg, Matthew Tancik, and Angjoo Kanazawa.
Garfield: Group anything with radiance fields. In 2024 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 21530–21539, 2024.
[12] Jiazhong Cen, Zanwei Zhou, Jiemin Fang, chen yang, Wei Shen, Lingxi Xie, Dongsheng Jiang, XIAOPENG
ZHANG, and Qi Tian. Segment anything in 3D with nerfs. In Advances in Neural Information Processing
Systems, volume 36, pages 25971–25990, 2023.
10

<!-- page 11 -->
[13] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao,
Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4015–4026,
October 2023.
[14] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything
in 3D scenes. In Proceedings of the European Conference on Computer Vision (ECCV), pages 162–179,
2025.
[15] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment
any 3d gaussians. Proceedings of the AAAI Conference on Artificial Intelligence, 39(2):1971–1979, Apr.
2025.
[16] Seokhun Choi, Hyeonseop Song, Jaechul Kim, Taehyeong Kim, and Hoseok Do. Click-gaussian: Interac-
tive segmentation to any 3D gaussians. In Proceedings of the European Conference on Computer Vision
(ECCV), page 289–305, 2024.
[17] Haiyang Ying, Yixuan Yin, Jinzhi Zhang, Fan Wang, Tao Yu, Ruqi Huang, and Lu Fang. Omniseg3d:
Omniversal 3D segmentation via hierarchical contrastive learning. In 2024 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages 20612–20622, 2024.
[18] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya
You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3D gaussian splatting to
enable distilled feature fields. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pages 21676–21685, 2024.
[19] Zhongzheng Ren, Aseem Agarwala, Bryan Russell, Alexander G. Schwing, and Oliver Wang. Neural
volumetric object selection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 6123–6132, 2022.
[20] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G. Derpanis, Jonathan Kelly, Marcus A.
Brubaker, Igor Gilitschenski, and Alex Levinshtein. SPIn-NeRF: Multiview segmentation and perceptual
inpainting with neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 20669–20679, 2023.
[21] Guikun Chen and Wenguan Wang. A survey on 3d gaussian splatting. arXiv preprint arXiv:2401.03890,
2025.
[22] Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian.
Gaussianobject: High-quality 3D object reconstruction from four views with gaussian splatting. ACM
Trans. Graph., 43(6), November 2024.
[23] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing
sparse-view 3d gaussian radiance fields with global-local depth normalization.
In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), pages 20775–20785, 2024.
[24] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis
using gaussian splatting. In Computer Vision – ECCV 2024, pages 145–163, Cham, 2025.
[25] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. Pixelsplat: 3d gaussian
splats from image pairs for scalable generalizable 3d reconstruction. In 2024 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages 19457–19467, 2024.
[26] Simon Niedermayr, Josef Stumpfegger, and Rüdiger Westermann. Compressed 3d gaussian splatting
for accelerated novel view synthesis. In 2024 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 10349–10358, 2024.
[27] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3D gaussian rep-
resentation for radiance field. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pages 21719–21728, 2024.
[28] Xinjie Zhang, Xingtong Ge, Tongda Xu, Dailan He, Yan Wang, Hongwei Qin, Guo Lu, Jing Geng, and
Jun Zhang. Gaussianimage: 1000 fps image representation and compression by 2d gaussian splatting. In
Computer Vision – ECCV 2024, pages 327–345, Cham, 2025.
[29] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma.
Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), pages 5322–5332, 2024.
11

<!-- page 12 -->
[30] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee. Multi-scale 3d gaussian splatting for anti-aliased
rendering. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages
20923–20931, 2024.
[31] Luis Bolanos, Shih-Yang Su, and Helge Rhodin. Gaussian shadow casting for neural characters. In 2024
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20997–21006, 2024.
[32] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d
gaussian splatting. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pages 19447–19456, 2024.
[33] Antoine Guédon and Vincent Lepetit. Sugar: Surface-aligned gaussian splatting for efficient 3D mesh
reconstruction and high-quality mesh rendering. In 2024 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 5354–5363, 2024.
[34] Jiahui Zhang, Fangneng Zhan, Muyu Xu, Shijian Lu, and Eric Xing. Fregs: 3d gaussian splatting with
progressive frequency regularization. In 2024 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 21424–21433, 2024.
[35] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D gaussian splatting for
geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, 2024.
[36] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs:
Structured 3d gaussians for view-adaptive rendering. In 2024 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 20654–20664, 2024.
[37] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for
open-vocabulary scene understanding. In 2024 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 5333–5343, 2024.
[38] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian,
and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), pages 20310–20320, 2024.
[39] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d
gaussians for high-fidelity monocular dynamic scene reconstruction. In 2024 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages 20331–20341, 2024.
[40] Jorge Condor, Sebastien Speierer, Lukas Bode, Aljaz Bozic, Simon Green, Piotr Didyk, and Adrian Jarabo.
Don’t splat your gaussians: Volumetric ray-traced primitives for modeling and rendering scattering and
emissive media. ACM Trans. Graph., 44(1), February 2025.
[41] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel
State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray tracing: Fast tracing of particle scenes.
ACM Trans. Graph., 43(6), November 2024.
[42] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. Pointnet++: deep hierarchical feature learning on
point sets in a metric space. In Proceedings of the 31st International Conference on Neural Information
Processing Systems, NIPS’17, page 5105–5114, 2017.
[43] Lin Zhao, Siyuan Xu, Liman Liu, Delie Ming, and Wenbing Tao. Svaseg: Sparse voxel-based attention for
3D lidar point cloud semantic segmentation. Remote Sensing, 14(18), 2022.
[44] Damien Robert, Bruno Vallet, and Loic Landrieu. Learning multi-view aggregation in the wild for
large-scale 3D semantic segmentation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 5565–5574, 2022.
[45] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He,
and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 4840–4851, 2024.
[46] Ji Hou, Angela Dai, and Matthias Nießner. 3d-sis: 3D semantic instance segmentation of rgb-d scans.
In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4416–4425,
2019.
[47] Wentao Sun, Zhipeng Luo, Yiping Chen, Huxiong Li, José Marcato Junior, Wesley Nunes Gonéalves,
and Jonathan Li. A click-based interactive segmentation network for point clouds. IEEE Transactions on
Geoscience and Remote Sensing, 61:1–12, 2023.
12

<!-- page 13 -->
[48] Weiyue Wang, Ronald Yu, Qiangui Huang, and Ulrich Neumann. Sgpn: Similarity group proposal network
for 3D point cloud instance segmentation. In 2018 IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 2569–2578, 2018.
[49] CloudCompare. Cloudcompare (version 2.13.2) [gpl software], 2024. URL http://www.cloudcompare.
org/.
[50] Muhammad Zubair Irshad, Sergey Zakharov, Katherine Liu, Vitor Guizilini, Thomas Kollar, Adrien
Gaidon, Zsolt Kira, and Rares Ambrus. Neo 360: Neural fields for sparse view synthesis of outdoor scenes.
In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9153–9164, 2023.
[51] Johannes L. Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. In 2016 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), pages 4104–4113, 2016.
[52] Johannes L. Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view
selection for unstructured multi-view stereo. In Proceedings of the European Conference on Computer
Vision (ECCV), pages 501–518, 2016.
13

<!-- page 14 -->
Technical Appendices and Supplementary Material
Scene
Tabletop
Images
3D Instances
2D Masks
Test Images
Desk. 1
Green square
324
10
2,417
209
Desk. 2
Green square
373
10
3,449
218
Desk. 3
Green square
957
10
6,997
249
Desk. 4
Brown circular
740
9
5,766
217
Desk. 5
Brown circular
764
10
6,971
210
Desk. 6
Green square
206
7
1,442
205
ALL
3364
56
27042
1308
Table 5: Statistics of the DesktopObjects-360 Dataset. It containing 3,364 images, 56 3D instances,
and 27,042 2D objects masks across 6 scenes. Scene-level distributions are as follows: images
(206-957), 3D instances (7-10), and 2D masks (1,442-6,997), reflecting variations in object density
across different desktop environments.
A
DesktopObjects Dataset
A.1
Overview
We introduce DesktopObjects-360, a benchmark specifically designed for 3D segmentation for
radiance field (3DGS and NeRF). Its key features include: 1. The dataset provides COLMAP format
data, making it directly applicable to 3DGS and NeRF modeling. 2. 3D instance annotations are
available for Gaussian models. 3. Pixel-level instance segmentation annotations are provided for 2D
images. 4. Instance IDs for 2D and 3D objects are consistent across different viewpoints.
The dataset contains a total of 3,364 multi-view images with fine-grained 2D and 3D instance annota-
tions. Each scene includes 7–10 object instances (56 annotated 3D instances in total), representing
common desktop items arranged under varying layouts and occlusions. To support instance segmen-
tation tasks, we provide 26,042 pixel-accurate 2D instance masks across all scenes, averaging 465
masks per 3D instance. This ensures dense and consistent multi-view correspondence. The dataset’s
scale and annotation granularity are further detailed in Table 5.
This dataset uniquely bridges 3D reconstruction and instance segmentation by offering per-instance
3D Gaussian Splatting (3DGS) priors alongside multi-view 2D masks, enabling the joint optimiza-
tion of geometry and segmentation. Scene complexity is intentionally varied, with image counts
ranging from 206 to 957 per scene, challenging algorithms to handle both sparse and dense view
configurations.
Our dataset provides four complementary representations per scene:
• Multi-view RGB streams: capturing real-world scenarios under varying lighting and
occlusions;
• Pixel-precise 2D masks with instance-level consistency across viewpoints;
• 3D geometric priors, represented as per-instance Gaussian primitives, where inst_label
values explicitly encode instance features;
• COLMAP-format data, compatible with common 3D reconstruction pipelines.
A.2
Collection Pipeline
The data collection pipeline involves three key stages:
Multi-view Acquisition: Using an iPhone 15 Pro (capturing 1920×1080 video at 30fps), we
perform 360° surround shooting around a desktop scene containing randomly arranged objects. After
each complete capture sequence, we randomly perturb object positions and orientations to create
configuration variations. This process is repeated to obtain six distinct video sequences per scene.
Sparse Reconstruction: We sample frames at 10–20 fps (yielding 200 to 1,000 images per scene)
and process them using COLMAP to obtain camera poses and sparse point clouds.
14

<!-- page 15 -->
Instance Annotation Pipeline:
• We train Gaussian models using 2DGS [35] for 30,000 iterations to obtain Gaussian splatting
representations.
• Expert annotators manually assign instance labels (inst_label) to each Gaussian primitive
using CloudCompare [49], ensuring consistent labeling across multi-view observations.
• Following the splatting projection method in PointGauss, we generate 2D instance masks
and refine them manually to achieve pixel-accurate annotations.
A.3
Additional Details
Testing Data: Approximately 220 evenly sampled images from the annotated dataset are processed
using COLMAP to generate data compatible with Gaussian Splatting and NeRF. This data is then
used for evaluation within the current scene.
Folder Structure: The DesktopObjects-360 dataset is organized in a hierarchical directory structure,
with a root folder containing six subdirectories (Desk1 through Desk6). Each Desk folder follows a
consistent organization scheme. For example, Desk1 contains subfolders for test data (Desk1_test),
original images (images), segmentation masks (mask), visualized masks (mask_visualize),
pretrained model annotations (annotated_pretrained_model(2dgs)), and a class label file
(class.txt).
Scene Characteristics
• Tabletop Properties:
– Desk. 1–3 and Desk. 6: Bright green square tabletops
– Desk. 4–5: Dark brown circular tabletops
• Object Composition:
– Common objects: ‘bus’, ‘pen’, ‘dog’, ‘stapler’, anthropomorphic figures (‘greenman’,
‘blueman’, ‘yellowman’)
– Desk. 4 omits ’alarmclock’ present in other scenes
– Desk. 6 introduces ‘redman’ while omitting ’alarmclock’, ’tape’, ’bus’, ’blueman’
present in other scenes
B
Splatting Projection
Splatting Projection is a module that, after obtaining 3D instance segmentation results from a network,
projects them into a 2D view to generate 2D instance segmentation masks. The algorithm consists of
two main components: Instance Label Projection during Rasterization and Mask Post-processing.
Alg. 1 presents the code for instance label projection in the rasterization stage. Alg. 2 shows the code
for mask post-processing. The implementation details are as follows:
• Morphological operations use disk-shaped structuring elements.
• The edge margin is set to 7 pixels, based on empirical observations of boundary uncertainty.
• Connected component analysis is implemented using a union-find algorithm.
C
Evaluation Metrics
To comprehensively evaluate the performance of PointGauss on the DesktopObjects-360 dataset, we
adopt a multi-faceted evaluation protocol that covers 3D geometric consistency, 2D view rendering
quality, and segmentation accuracy.
C.1
3D Mean Intersection over Union (3D-IoU)
3D IoU measures the segmentation accuracy of Gaussian primitives, serving as an effective metric for
evaluating an algorithm’s performance in 3D space segmentation. For the 3D instance segmentation
15

<!-- page 16 -->
Dataset
Method
3D-IoU
2D-mIoU
OA
Pr
Recall
F1-score
PQ
AP50
Desk. 1
GARField
–
66.22
92.52
82.89
75.39
78.96
69.32
82.89
Omniseg
–
53.09
81.11
85.75
84.69
85.22
76.98
85.75
Feature3DGS
–
38.36
71.49
78.55
78.55
78.55
67.39
78.55
SAGA
–
68.72
85.67
95.66
95.66
95.66
90.32
95.66
PointGauss (ours)
69.40
84.33
95.90
90.79
87.35
89.04
73.35
90.79
Desk. 2
GARField
–
27.27
82.56
20.24
19.83
20.03
13.84
20.24
Omniseg
–
38.85
75.49
72.47
67.93
70.13
59.22
72.47
Feature3DGS
–
32.72
62.48
75.95
75.88
75.91
62.26
75.95
SAGA
–
66.55
86.86
93.44
93.44
93.44
85.20
93.44
PointGauss (ours)
82.46
86.91
96.67
93.74
92.40
93.07
78.30
93.74
Desk. 3
GARField
–
71.48
91.86
83.32
78.59
80.89
72.47
83.32
Omniseg
–
46.64
74.25
68.48
66.78
67.62
59.10
68.48
Feature3DGS
–
42.27
71.45
62.29
61.98
62.13
50.65
62.29
SAGA
–
69.42
87.31
92.05
92.05
92.05
85.50
92.05
PointGauss (ours)
73.38
85.85
94.54
89.70
87.91
88.80
72.13
89.70
Desk. 4
GARField
–
64.63
93.32
81.64
71.34
76.14
65.17
81.64
Omniseg
–
56.35
89.35
78.98
77.79
78.38
68.88
78.98
Feature3DGS
–
34.49
70.43
77.62
77.44
77.53
62.45
77.62
SAGA
–
87.47
97.52
96.69
96.69
96.69
90.32
96.69
PointGauss (ours)
78.05
89.36
97.68
94.23
93.08
93.65
80.15
94.23
Desk. 5
GARField
–
58.82
90.61
82.60
77.38
79.90
67.93
82.60
Omniseg
–
38.94
77.99
64.82
62.94
63.87
54.76
64.82
Feature3DGS
–
32.54
66.16
68.04
67.75
67.89
53.09
68.04
SAGA
–
79.40
95.41
93.37
93.37
93.37
84.39
93.37
PointGauss (ours)
80.94
85.55
96.76
93.09
92.08
92.58
78.74
93.09
Desk. 6
GARField
–
65.21
97.44
68.60
68.36
68.48
56.84
68.60
Omniseg
–
32.02
82.02
84.83
81.46
83.11
72.60
84.83
Feat3dgsNEW
–
19.65
62.42
79.30
79.30
79.30
60.61
79.30
SAGA
–
59.34
92.83
95.33
95.33
95.33
84.35
95.33
PointGauss (ours)
77.67
91.12
99.31
99.93
99.93
99.93
89.67
99.93
Table 6: Cross-dataset segmentation performance comparison of PointGauss against state-of-the-art
methods. Our approach achieves superior 2D IoU performance across all datasets. All metrics are
reported as percentages.
task with C = 2 classes (0: background, 1: foreground), the Intersection over Union (IoU) for each
class i is computed as:
IoUi =
TPi
TPi + FPi + FNi
,
i ∈{0, 1}
(8)
where:
• TPi (True Positives): The number of samples correctly predicted as class i.
• FPi (False Positives): The number of samples incorrectly predicted as class i (true class is
not i).
• FNi (False Negatives): The number of samples incorrectly predicted as not class i (true
class is i).
The 3D mean Intersection over Union (3D-IoU) is then calculated as the arithmetic mean of the IoU
values for all classes:
3D-IoU = 1
C
C−1
X
i=0
IoUi = IoU0 + IoU1
2
(9)
In the implementation:
16

<!-- page 17 -->
• The confusion matrix is used to accumulate predictions and ground truth labels.
• TPi corresponds to the diagonal elements of the confusion matrix.
• FPi is computed as the column sum minus the diagonal element: FPi = PC−1
j=0 Mj,i−TPi.
• FNi is computed as the row sum minus the diagonal element: FNi = PC−1
j=0 Mi,j −TPi.
This metric effectively evaluates the model’s performance in segmenting foreground objects in 3D
point clouds by accounting for both false positives and false negatives. It provides a more robust
measure than accuracy, particularly in cases of class imbalance.
C.2
Images Semantic Segmentation Metrics
After the splatting projection, we can get a series of 2D masks for instance segmentation. These
are then used to evaluate the pixel-level classification performance of foreground instances. The
2D Intersection over Union (IoU) and Overall Accuracy (OA) metrics assess the segmentation
accuracy of the rendered images, providing an effective measure of the algorithm’s performance in
2D view-dependent image segmentation.
C.2.1
Overall Accuracy
OverallAcc =
PC
c=0 TPc
PC
c=1 (TPc + FPc + FNc)
(10)
C.2.2
2D Mean Intersection over Union (2D-IoU)
2D-IoU =
1
Cvalid
C
X
c=1
TPc
TPc + FPc + FNc
(11)
where:
• C: Total number of instances (excluding background). The background class (ID=0).
• Cvalid: Number of valid classes (excluding classes not present in both prediction and ground
truth).
• TPc: True positive pixels for class c (pixels correctly predicted as class c).
• FPc: False positive pixels for class c (pixels incorrectly predicted as class c).
• FNc: False negative pixels for class c (pixels of class c incorrectly predicted as other classes).
C.3
Images Instance Segmentation Metrics
Instance-level matching based on IoU threshold (default θ = 0.5).
C.3.1
Matching Rule
• Use the Hungarian algorithm to maximize global IoU.
• A match is valid if IoU ≥θ.
• Count true positives (TP), false positives (FP), and false negatives (FN).
C.3.2
Basic Metrics
Precision
P =
TP
TP + FP
(12)
Recall
R =
TP
TP + FN
(13)
17

<!-- page 18 -->
F1-Score
F1 = 2 × P × R
P + R
(14)
where:
• TP: Number of true positive instances (matched instances with IoU ≥θ).
• FP: Number of false positive instances (unmatched predicted instances).
• FN: Number of false negative instances (unmatched ground truth instances).
C.3.3
Panoptic Quality (PQ)
PQ =
P
matched IoUi
TP + 0.5 × FP + 0.5 × FN
(15)
where:
• P
matched IoUi: Sum of IoU for all matched pairs.
• TP + 0.5 × FP + 0.5 × FN: Penalty term for unmatched instances.
C.3.4
AP@50 (Average Precision at IoU=0.5)
AP@50 =
Z 1
0
p(r) dr
(16)
where:
• p(r): Precision at recall level r.
• Calculation steps:
1. Sort all predicted instances by confidence score.
2. Compute the maximum precision p(r) at each recall level r.
3. Integrate p(r) over recall (typically computed via interpolation).
C.4
Implementation Details
• The confusion matrix is dynamically resized to accommodate the maximum class ID in the
dataset.
• The Hungarian algorithm maximizes IoU using 1 −IoU as the cost matrix.
• The PQ formula follows the standard definition in panoptic segmentation, with penalty terms
for FP and FN.
• Background is explicitly excluded in semantic metrics, while instance metrics use binary
masks for separation.
Metric Selection Rationale: Our metrics address three critical aspects: (1) 3D-IoU and PQ validate
the geometric fidelity of Gaussian primitives; (2) 2D-IoU and AP50 reflect the quality of segmentation
in rendered views; and (3) OA and F1-score provide a balanced evaluation under class imbalance. This
multi-faceted evaluation protocol aligns with the hybrid neural-explicit representation characteristics
of 3DGS.
D
Nerds360 Annotation
The NeRDS 360 dataset [50] provides ten scenes, reconstructed from COLMAP [51, 52] inputs,
offering detailed and immersive representations of the urbane environments. These scenes are 360°
panoramic images, each featuring 3-4 vehicles as the primary subjects. To establish instance-level
annotations, we extend the annotation pipeline with vehicle-specific adaptations and get 35 instances
with 3D instance labels on Gaussians.
18

<!-- page 19 -->
Num. of Clicks
3D IoU
2D mIoU
OA
Pr
recall
f1_score
pq
ap_50
5
68.72
90.01
99.22
99.93
99.93
99.93
88.43
99.93
10
73.29
90.99
99.29
99.93
99.93
99.93
89.51
99.93
15
75.17
90.80
99.29
99.93
99.93
99.93
89.41
99.93
20
77.67
91.12
99.31
99.93
99.93
99.93
89.67
99.93
25
80.61
91.18
99.30
99.93
99.93
99.93
89.75
99.93
30
81.83
90.95
99.29
99.93
99.93
99.93
89.56
99.93
Table 7: PointGauss’s interactive segmentation performance under varying user clicks on Desk.
6. The method demonstrates progressive improvements in 3D IoU (81.83% at 30 clicks) while
maintaining high 2D precision. All metrics are reported as percentages.
Figure 8: Viewpoint-robust Instance IDs. Persistent color/ID assignments (#1-#7) across 4 viewpoints
validate our 3D-consistent segmentation.
E
Experiments
E.1
Implementation Detail
Since the algorithms SAGA, OmniSeg3D, Feature3DGS, and GARField were not originally de-
signed for automatic multi-object tracking, obtaining instance-specific segmentation masks requires
additional per-viewpoint prompt information. To address this, we adopt a strategy that leverages
the precise instance location details available in the ground truth masks. Specifically, we randomly
sample a single click within each target instance region to generate the necessary segmentation
prompts.
In contrast, our proposed PointGauss algorithm assigns unique instance labels to each Gaussian
primitive during the 3D initialization stage. As a result, novel-view instance segmentation can be
achieved without requiring additional prompt information, effectively enabling automatic tracking-
like capability. Furthermore, since no directly applicable 3D foundation model exists for point cloud
segmentation, we trained our own segmentation model using the annotated Gaussian models as point
cloud data. For training, we adopted a cross-validation approach in which five annotated Gaussian
models were used for training, and the remaining one was used for testing.
E.2
Segmentation Performance
To provide an in-depth analysis of the experimental results, we systematically evaluate PointGauss
against state-of-the-art approaches on DesktopObjects-360 dataset. The extended comparison results
are shown in Table 6.
19

<!-- page 20 -->
(a) Gaussian Primitives (Mean Value)
(b) Mask
(c) Mask (Post-processing
Figure 9: Void Limitation. (a) The Gaussian primitives with void areas. (b) The mask after projection
rendering. (c) The mask with post-processing. (The red dashed circle points out the flaw).
F
Limitations
F.1
Reconstruction Artifacts and Incomplete Segmentation
Regions with insufficient visual coverage can result in voids within the Gaussian model, particularly
near viewpoint boundaries (Fig. 9(a)). While projection rendering (Fig. 9(b)) and post-processing
(Fig. 9(c)) partially mitigate this issue, significant voids often remain due to the inherent characteristics
of Gaussian primitives. In planar regions with uniform color, low Gaussian density further exacerbates
the problem.
F.2
Lack of 3D Foundation Models
Unlike 2D segmentation, our method lacks access to universal 3D priors, necessitating task-specific
training. This constraint limits generalization to unseen categories and adaptation to alternative
3D representations (e.g., NeRF) without architectural modifications and retraining. Developing
general-purpose 3D segmentation models remains an open challenge.
G
Broader Impacts
G.1
Positive Impacts
• Advancements in 3D Scene Understanding: By integrating point cloud semantic seg-
mentation with 3D Gaussian Splatting, our method may advance dynamic scene instance
segmentation techniques, enabling finer-grained 3D environment comprehension for AR/VR
and robotic navigation applications.
• Low-Cost 3D Modeling: Compared to traditional LiDAR solutions, vision-based 3DGS
instance segmentation reduces the cost of 3D scene analysis, making advanced tools more
accessible to small and medium-sized enterprises.
• Smart City Development: Applicable to urban digital twin construction, enabling auto-
mated identification and management of city infrastructure (e.g., streetlights, traffic signs)
via mobile capture devices.
20

<!-- page 21 -->
G.2
Potential Risks
• Privacy Challenges: High-fidelity scene reconstruction may inadvertently capture sensitive
information (e.g., license plates, faces), necessitating data anonymization in algorithm
design.
• Algorithmic Bias: Training data imbalances may lead to segmentation biases, particularly
for culturally specific scenes.
G.3
Mitigation Strategies
• Integrate privacy-preserving modules (e.g., automatic sensitive area blurring) in open-source
implementations
• Develop domain-specific ethical guidelines (e.g., prohibiting targeted population surveil-
lance)
Algorithm 1 Instance Label Projection in Rasterization
Require:
1: pix: current pixel coordinates (x, y)
2: collected_id: list of Gaussian primitive indices
3: inst_label: array of instance labels per Gaussian
4: inst_image: output instance segmentation map
Ensure:
5: Updated inst_image with projected instance labels
6: ρ2d_threshold ←4.0
▷2D distance threshold
7: procedure PROJECTINSTANCELABELS
8:
for each pixel (x, y) in tile do
9:
inst_image[x, y] ←0
▷Initialize instance label
10:
for each Gaussian j in influencing primitives do
11:
prim_id ←collected_id[j]
12:
prim_center ←points_xy_image[prim_id]
13:
dx ←prim_center.x −x
14:
dy ←prim_center.y −y
15:
ρ2d ←dx2 + dy2
▷Calculate 2D distance
16:
if ρ2d ≤ρ2d_threshold then
17:
current_label ←inst_label[prim_id]
18:
if current_label > 0 and inst_image[x, y] == 0 then
19:
inst_image[x, y] ←current_label
▷Assign instance label
20:
end if
21:
end if
22:
end for
23:
end for
24: end procedure
21

<!-- page 22 -->
Algorithm 2 Mask Post-processing
Input: Raw mask M ∈{0, 1, ..., K}H×W
Output: Refined mask Mref
1: Step 1: Hole Filling & Smoothing
2: for each instance k ∈{1, . . . , K} do
3:
Extract binary mask Bk ←I(M = k)
4:
Apply morphological closing with large kernel (iter=10)
5:
Fill interior holes via flood-fill
6:
Smooth boundaries via small-kernel closing (iter=1)
7: end for
8:
9: Step 2: Edge-aware refinement (safety margin=7px):
• Skip holes touching image borders
• Fill fully enclosed background regions
10:
11: Step 3: Largest Component Selection
12: for each Bk do
13:
Compute connected components via 8-neighbor labeling
14:
Retain only the component with maximum area
15:
Set Mref[Blargest
k
] ←k
16: end for
22
