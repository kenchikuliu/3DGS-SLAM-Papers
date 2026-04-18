<!-- page 1 -->
Holi-Spatial: Evolving Video Streams into Holistic 3D
Spatial Intelligence
Yuanyuan Gao1,2∗, Hao Li2,5∗, Yifei Liu1,6∗, Xinhao Ji1,4∗, Yuning Gong1,7∗, Yuanjun Liao7,
Fangfu Liu8, Manyuan Zhang9, Yuchen Yang10, Dan Xu11, Xue Yang3, Huaxi Huang1,
Hongjie Zhang1, Ziwei Liu5, Xiao Sun1, Dingwen Zhang2,†, Zhihang Zhong3,†
∗Equal Contribution; †Corresponding Authors
1Shanghai AI Lab, 2Northwestern Polytechnical University, 3Shanghai Jiao Tong University,
4Peking University, 5Nanyang Technological University, 6Beihang University, 7Sichuan University,
8Tsinghua University, 9The Chinese University of Hong Kong, 10Fudan University,
11Hong Kong University of Science and Technology
The pursuit of spatial intelligence fundamentally relies on access to large-scale, fine-
grained 3D data. However, existing approaches predominantly construct spatial under-
standing benchmarks by generating question–answer (QA) pairs from a limited number
of manually annotated datasets, rather than systematically annotating new large-scale
3D scenes from raw web data. As a result, their scalability is severely constrained, and
model performance is further hindered by domain gaps inherent in these narrowly
curated datasets. In this work, we propose Holi-Spatial, the first fully automated, large-
scale, spatially-aware multimodal dataset, constructed from raw video inputs without
human intervention, using the proposed data curation pipeline. Holi-Spatial supports
multi-level spatial supervision, ranging from geometrically accurate 3D Gaussian Splat-
ting (3DGS) reconstructions with rendered depth maps to object-level and relational
semantic annotations, together with corresponding spatial Question–Answer (QA) pairs.
Following a principled and systematic pipeline, we further construct Holi-Spatial-4M,
the first large-scale, high-quality 3D semantic dataset, containing 12K optimized 3DGS
scenes, 1.3M 2D masks, 320K 3D bounding boxes, 320K instance captions, 1.2M 3D
grounding instances, and 1.2M spatial QA pairs spanning diverse geometric, relational,
and semantic reasoning tasks. Holi-Spatial demonstrates exceptional performance in
data curation quality, significantly outperforming existing feed-forward and per-scene
optimized methods on datasets such as ScanNet, ScanNet++, and DL3DV. Further-
more, fine-tuning Vision-Language Models (VLMs) on spatial reasoning tasks using this
dataset has also led to substantial improvements in model performance.
Website: https://visionary-laboratory.github.io/holi-spatial/
Code: https://github.com/Visionary-Laboratory/Holi-Spatial
Email: zhongzhihang95@gmail.com
arXiv:2603.07660v1  [cs.CV]  8 Mar 2026

<!-- page 2 -->
3D Object Detection
Spatial Reasoning
3D Grounding
2D Object Segmentation
“Vibrant red fabric sofa with two 
blue patterned throw pillows”
“White vertical panel radiator with 
horizontal fins, smooth surface”
“A poster mounted 
beside the windows 
shows a person’s face”
“Vibrant red fabric sofa with two 
blue patterned throw pillows”
“wooden lantern with square 
frame and light brown color”
Rendered Depth
Recon. Mesh
…
“dark grey monitor with 
silver frame, black screen, 
and visible Dell logo on back”
0 0.2 0.4 0.6 0.8 1
LangSplat
M3
Ours
0
20 40 60 80
SpatialLM
3D-LLaVA
Ours
✅High-Quality 3D Recon.
✅Accurate Semantic Anno.
✅Scalable Anno. Data
Depth F1 Score
3D Det. AP50
0
20
40
60
Accuracy Rate (%)
Ours-8B
Cambrian-S-8B
Qwen3-VL-8B
MindeCube Spatial Reasoning
0
10
20
30
AP50 Acc.  (%)
Ours-8B
VST-7B
Qwen3-VL-8B
Scannet++ 3D Grounding
VLM
3D Grounding
Spatial Reasoning
+64%
+0.5
You are positioned at whiteboard 
and face door. In which direction 
is radiator relative to you?
A. Front
B. Back-Left
C. Front-Right
D. Front-Left
You are positioned at desk and 
face suitcase. In which direction 
is book relative to you?
A. Back-Right
B. Back
C. Front-Left
D. Left
You are positioned at box and 
face clock. In which direction is 
bottle relative to you?
A. Back-Right
B. Left
C. Back
D. Front-Right
When you were taking the photo 
in Image B, where is the chair in 
Image A relative to you?
A. Left-Front
B. Left-Back
C. Right-Back
D. Right-Front
Given view A and view B, Which 
single rotation direction is the 
most prominent? 
A. tile up
B. turn right
C. turn left
D. tilt down
The direction of pillow relative to 
painting is south. What’s direction 
of mirror relative to painting?
A. south-west
B. north-east
C. south
D. east
Figure 1 | We introduce Holi-Spatial, the first fully automated pipeline capable of converting
raw video streams into holistic 3D spatial annotations without human intervention. Com-
pared to state-of-the-art methods, Holi-Spatial achieves a significant leap in annotation quality,
improving multi-view depth estimation by 0.5 F1 and boosting 3D detection AP50 by a remark-
able 64% on ScanNet [28]. Based on this, we introduce Holi-Spatial-4M, a large-scale dataset
that effectively empowers Vision-Language Models. As shown, fine-tuning Qwen3-VL on
Holi-Spatial-4M leads to state-of-the-art performance, with a 15% AP50 gain on ScanNet++ [27]
and a 7.9% accuracy rise on MMSI-Bench [18]. Importantly, because the entire annotation
pipeline is automatic, it can be further scaled up as resources permit.
1. Introduction
Spatial intelligence [20] is a fundamental bridge toward enabling large models to understand
the real 3D world. It requires large multimodal models (LMMs) [47, 13, 15, 14] to move beyond
2D, language-centric perception and develop robust 3D spatial abilities to perceive, ground, and
reason about the 3D world from visual inputs. Such capabilities hold great promise for a wide
range of real-world applications, including robotic manipulation [21, 22] and navigation [16],
scene editing [24], and augmented reality [23].
However, a key limitation is the scarcity and imbalance of raw spatial data. Prior methods [25,
37, 19, 26] typically curate spatial supervision by generating QA pairs from a small set of
manually annotated 3D datasets (e.g., ScanNet [28] and ScanNet++ [27]) or by naively applying
feed-forward perception models [29] to single-image data [30]. While these strategies improve
over general VLMs [13, 14, 15], they are difficult to scale due to reliance on specialized scanning
hardware and human-in-the-loop annotation, and they often provide limited semantic coverage
(e.g., only 50 labeled classes in ScanNet [28]).
To address these limitations, we note that recent advances in relevant AI tools [49, 29, 15,
5, 12, 6, 54] have exceeded expectations; by systematically composing them, we may build an
automated spatial annotation engine that can even outperform human annotations, enabling
a positive data flywheel. Thus, we present Holi-Spatial, a fully automated framework that
2

<!-- page 3 -->
converts raw video streams into high-fidelity 3D geometry together with holistic semantic anno-
tations without requiring any explicit 3D sensors or human-in-the-loop labeling. As summarized
in table 1, Holi-Spatial unifies a broad set of spatial tasks, including 3D reconstruction, novel
view synthesis (NVS), depth rendering, 2D instance segmentation, instance captioning, 3D
bounding boxes, 3D grounding, and spatial QA.
Holi-Spatial is composed of three stages. (i) Geometric Optimization: we initialize from monoc-
ular priors (Depth-Anything-V3 [12]) and optimize a 3D Gaussian Splatting (3DGS) scene under
geometric supervision to sharpen structure and suppress floaters. (ii) Image-level Perception:
we sample keyframes, use a VLM to infer open-vocabulary categories, which guides SAM3 to
produce high-quality open-set masks per-image. (iii) Scene-level Lift and Refinement: We lift the 2D
masks into 3D by back-projecting their pixels using rendered depth and camera intrinsics, and
transforming the resulting points into the world frame using recovered camera poses. The re-
sulting 3D points serve as instance candidates. We merge redundant candidates across views by
checking their bounding box IoU, and use a VLM-based agent to filter low-confidence candidates;
for the merged and most reliable instances, we generate detailed captions and further construct
grounding and QA pairs for training VLM’s 3D Grounding and Spatial Reasoning ability.
Object Num.
Class Num.
Images
Our Refined Label
Vanilla ScanNet GT
Figure 2 | Comparison of our refined annotations with
the official annotations on the ScanNet dataset [28]. Our
method achieves more accurate and sharper segmentation
masks, as well as improved category recognition.
We evaluate the proposed cura-
tion pipeline on established bench-
marks,
including ScanNet [28],
ScanNet++ [27], and DL3DV-10K [31],
where we create open-vocabulary
GT by additionally re-annotating
the scenes with human supervision
to enable reliable evaluation of fine-
grained objects. As shown in fig. 1,
Holi-Spatial consistently outper-
forms both model-based baselines
and 3DGS-based optimization ap-
proaches; on ScanNet++ [27], we
improve multi-view depth estima-
tion by up to 0.5 F1 and boost 3D
detection AP50 by 64%.
Leveraging this pipeline, we construct and release Holi-Spatial-4M from diverse video
sources (ScanNet [28], ScanNet++ [27], and DL3DV-10K [31]). The dataset contains 12K opti-
mized 3DGS scenes with multi-view consistent depth renderings, as well as 1.2M 2D masks,
1.2M 3D Grounding, 320K 3D bounding boxes, 320K instance captions, and 1.3M spatial QA
pairs. Beyond scale, our annotations exhibit higher granularity and improved boundaries
compared to official labels on Scannet [28] ( fig. 2). To verify that Holi-Spatial-4M improves
spatial intelligence, we fine-tune the Qwen3-VL family [13, 14] on our curated data. This yields
consistent gains on public benchmarks, including a 15% AP50 improvement on ScanNet++ [27]
for 3D grounding and a 7.9% accuracy increase on MMSI-Bench [18].
3

<!-- page 4 -->
Table 1 | Overview of Pipeline Capabilities. We compare input modalities and output tasks
across different paradigms, ranging from 2D VLMs and 3D-VLMs to 3D-GS-based understand-
ing methods. Holi-Spatial serves as a unified framework supporting diverse spatial tasks
without relying on 3D priors.
Method
Inputs
Outputs
Images Point Clouds Depth 2D Seg. 3D Det. Grounding Spatial QA
2D-VLM Methods
SAM3
✓
✗
✗
✓
✗
✗
✗
SA2VA
✓
✗
✗
✓
✗
✓
✓
3D-VLM Methods
SpatialLM
✗
✓
✗
✗
✓
✓
✓
LLaVA-3D
✗
✓
✗
✗
✓
✓
✓
SceneScript
✗
✓
✗
✗
✓
✗
✗
3D-GS based Understanding Methods
M3-Spatial
✓
✗
✓
✓
✗
✓
✗
LangSplat
✓
✗
✓
✓
✗
✗
✗
Holi-Spatial
✓
✓
✓
✓
✓
✓
✓
2. Related Work
2.1. Data Scalability in Spatial Intelligence
Recent advancements in Large Multimodal Models (LMMs) have demonstrated remarkable
proficiency in 2D visual understanding and planar reasoning [15, 13, 14], but still exhibit a
significant lag behind human capabilities in Spatial Intelligence [17, 18, 19]. We attribute this gap
primarily to a critical disparity in raw data diversity. Unlike 2D datasets such as LAION-5B [32]
provide billion-scale unique image inputs, the spatial intelligence domain suffers from severe
scene scarcity. Although recent spatial datasets such as SenseNova-SI-800K [37] and VST-4M [26]
boast million-level annotations, these are predominantly derived from a tiny pool of only a
few thousand static 3D scans (e.g., ScanNet). This reliance on a narrow set of environments
fundamentally limits generalization, motivating us to construct a more scalable data curation
framework. Instead of relying on a limited set of labeled 3D assets, we aim to unlock the
potential of abundant web videos, automatically generating dense spatial annotations to fuel
the next generation of spatial intelligence.
2.2. Methods for Spatial Intelligence
Efforts to enhance spatial intelligence in multimodal models primarily follow three approaches:
(1) 3D-native LMMs, a line of works [33, 34, 25, 35, 36] directly consumes explicit 3D observations
(e.g., point clouds, meshes, or multi-view RGB-D) and performs reasoning in 3D space, such
as SpatialLM [35] and LLaVA-3D [34]. (2) 2D-Centric spatial LMMs [26, 37, 3, 19, 58] improve
performance primarily by scaling up training datasets and training recipes on spatial perception
and reasoning. For example, VST [26] adopts 4.1M samples for SFT and RL, while Cambrian-
S [3] develops the VSI-590K dataset to enhance spatial video understanding. (3) 3DGS-based
methods [42, 40, 41] leverage 3DGS [38] as an explicit scene representation and optimize it to
align geometry with language / vision signals [39]. For example, M3-Spatial [40] augments a
4

<!-- page 5 -->
Step 1: Geometric Optimization
Input: Images
Cam Params.
Depth-
Anything-3
Point Cloud
Initialization
Multi-View Consistent
Normal Supervision
3D-GS Training
Render Supervision
Reﬁned
Depth
Step 3: Scene-level Refinement
(1) Sampled 
Key Frames
VLM
Class-Label Memory
Chair
VLM
Class-Label Memory
Chair
VLM
Class-Label Memory
Chair
Step 2: Image-level Perception
Sofa
Sofa
Lamp
SAM3
SAM3
SAM3
(2) VLM-Based Object 
Segmentation
(3) 2D-to-3D 
OBB Generation
(2) Confidence-Based Filtering & Refinement
(1) Multi-View Merge & Post-Process
{
}
Obj 1
2D
3D
Conf.
{
}
Obj 2
2D
3D
Conf.
{
}
Obj 3
2D
3D
Conf.
…
{
}
Obj 1
2D Mask
3D BBox
Conf. 
Img Indx.
table
table
table
Merge
same class & IoU Overlap)
> 0.8
< 0.8
Conf.
> 0.9
Keep
Discard
AI Agent Tool Use
Zoom in
SAM Mask
VLM
{                                }
Obj 1
2D
3D
Caption
(3) Object Caption & QA Generation
{                                
}
Obj 2
2D
3D
Caption
Where is                    lo-
cated on                    ?
Obj 1
Obj 2
                  is on the so-
utheast of
Obj 1
Obj 2
…
Spatial Relation
QA Pairs
VLM
Image
Text
3D Grounding
3D Obj. Detection
Spatial Reasoning
…
…
…
…
…
Figure 3 | Overview of the Holi-Spatial data curation pipeline. The framework operates in
three progressive stages: (1) Geometric Optimization distills high-fidelity 3D structure from
video streams using 3DGS; (2) Image-level Perception lifts 2D VLM and SAM3 predictions
into initial 3D proposals; and (3) Scene-level Refinement employs a coarse-to-fine strategy to
merge, verify, and caption instances, yielding dense, high-quality spatial annotations. Finally,
leveraging the generated Holi-Spatial-4M dataset, we directly fine-tune the Qwen-VL family
for downstream tasks (e.g., 3D grounding and spatial reasoning).
per-scene 3DGS reconstruction with language-aligned features to support open-vocabulary 3D
grounding within the optimized scene.
Despite notable progress, existing paradigms face fundamental scalability bottlenecks. Ap-
proaches (1) and (2) rely heavily on human-annotated 3D data or manually curated scans,
making dataset expansion costly and intrinsically limited by 3D acquisition and annotation
overhead. Approach (3), based on optimization-driven 3DGS, requires per-scene training or
finetuning, which is time-consuming and often unstable, hindering large-scale deployment.
In contrast, Holi-Spatial reframes spatial data curation as a scalable, non-human pipeline that
converts raw videos into annotated high-fidelity 3D scenes, enabling automatic generation of
large-scale spatial-understanding data.
3. Method
Here we introduce the core steps of our curation framework, as illustrated in fig. 3. It is
composed of three stages: (i) Geometric Optimization (section 3.1) distills high-fidelity 3D
structure from raw videos; (ii) Image-level Perception (section 3.2) extracts spatially consistent
object labels from a VLM, produces high-quality 2D masks on keyframes, and lifts them into
floor-aligned 3D proposals; and (iii) Scene-level Refinement (section 3.3) merges, filters, verifies,
and captions instances in 3D to generate dense, reliable spatial annotations.
3.1. Geometric Optimization
The primary objective of this stage is to distill raw video streams into high-fidelity geometric
structures, serving as a robust prerequisite for precise spatial annotations. To achieve this, we
first employ Structure-from-Motion [45] to resolve accurate camera intrinsics and extrinsics,
followed by leveraging a state-of-the-art spatial foundation model [12] to initialize a dense,
5

<!-- page 6 -->
Figure 4 | Pipeline of 2D-to-3D OBB Generation. We transform 2D object masks into initial 3D
OBBs via depth projection, utilizing a four-step strategy to mitigate the impact of depth floaters.
(1) We obtain an initial object depth map by combining 3DGS rendering with SAM3 instance
segmentation. (2) To mitigate 2D boundary errors from SAM3, we erode the object mask near its
contour and keep only the reliable interior region. (3) To remove 3D outliers caused by depth
discontinuities, we use a multi-view-consistent mesh depth as guidance and filter inconsistent
pixels in the 3DGS depth. (4) Finally, we estimate the initial 3D OBB from the refined point
cloud, while preserving the associated 2D mask, confidence score, and source image index.
structurally coherent point cloud. However, feed-forward depth estimations inevitably contain
noise and outliers. To address this, we incorporate 3D Gaussian Splatting (3DGS) [38] for
per-scene optimization. Specifically, we integrate geometric regularization inspired by the
surface reconstruction 3D Gaussian method [44, 41, 48, 51, 52, 50] to enforce multi-view depth
consistency. This process effectively eliminates large-scale floaters that would otherwise interfere
with 3D bounding box generation. These combined efforts yield a clean and consistent scene
representation aligned with physical surfaces.
3.2. Image-level Perception
We uniformly sample a set of keyframes I = {𝐼1, . . . , 𝐼𝑇} from the raw video stream. For
each frame 𝐼𝑡, we employ Gemini3-Pro [15] to generate a caption sequentially. Crucially, we
maintain a dynamic class-label memory M𝑡to ensure semantic consistency. This memory
accumulates recognized categories from previous frames (𝐼1:𝑡−1) and instructs the VLM to
prioritize reusing existing labels, formally updated as M𝑡= M𝑡−1 ∪Extract(𝐼𝑡). Guided by
the prompt derived from M𝑡, SAM3 [29] performs open-vocabulary instance segmentation,
producing a set of predictions O𝑡= {(𝑀𝑘, 𝑠𝑘)}𝑁
𝑘=1, where 𝑀𝑘represents the binary mask and 𝑠𝑘
denotes the confidence score.
Leveraging the refined depth map 𝐷𝑡rendered from our optimized 3DGS, we unproject
each pixel u = (𝑢, 𝑣) in mask 𝑀𝑘into a 3D point P ∈R3 via: P = 𝐷𝑡(u) · K−1 ˜u, where K
is the camera intrinsic matrix and ˜u = [𝑢, 𝑣, 1]⊤is the homogeneous coordinate. However,
directly computing an Oriented Bounding Box (OBB) from these unprojected points often yields
inaccurate bounds due to depth edge floaters around objects. These edge artifacts mainly
arise from two sources: (i) 2D-level errors, where SAM3 segmentation introduces boundary
misalignment near object contours; and (ii) 3D-level noise, where depth discontinuities naturally
lead to unstable measurements and outliers. To suppress these floaters and improve grounding
box precision, we seamlessly integrate a geometry-aware filtering strategy during the 2D-to-3D
lifting process, as detailed in fig. 4. Specifically, this strategy mitigates both the 2D boundary
errors via mask erosion and the 3D outliers through mesh-guided depth filtering, ensuring that
the estimated initial 3D OBBs are derived from a highly reliable geometry subset.
6

<!-- page 7 -->
Figure 5 | Floor-aligned OBB post-processing pipeline. Starting from the input instance OBBs
produced after 2D-to-3D lifting and initial OBB estimation, we (1) detect a floor (or fallback
planar structure) to infer a global up-axis, (2) re-align each instance OBB under a yaw-lock
constraint with optional PCA fallback and update extents, and (3) apply a validation check to
output gravity-/floor-consistent OBBs for downstream scene-level refinement.
3.3. Scene-level Refinement
The core motivation of this stage is a coarse-to-fine refinement strategy designed to distill
high-fidelity annotations from noisy initial proposals. Based on the initial set of proposals
Pinit = {(𝐵𝑖, 𝑐𝑖, 𝑠𝑖)}𝑀
𝑖=1 obtained from 2D-to-3D lifting, where 𝐵𝑖∈R7 denotes the raw 3D box
parameters, 𝑐𝑖is the semantic category, and 𝑠𝑖is the confidence score, our refinement proceeds in
three core parts:
1) Multi-View Merge & Post-Process. We first perform spatial clustering to consolidate redundant
detections. Specifically, we iterate over all pairs of instances 𝑝𝑖, 𝑝𝑗∈Pinit and merge them if they
share the same category and have sufficient 3D overlap:
𝑐𝑖= 𝑐𝑗∧IoU3𝐷(𝐵𝑖, 𝐵𝑗) > 𝜏merge,
(1)
where 𝜏merge is set to 0.2. It mitigates object fragmentation by consolidating disjoint observations
of the same object (e.g., parts of a large sofa) into a single 3D bounding box. For each merged
instance, we update its attributes to preserve the most reliable observation: the confidence score
is set to 𝑠𝑘= max(𝑠𝑖, 𝑠𝑗), and we retain the source image index associated with this maximum.
This selection mechanism grounds subsequent VLM captioning in the most informative visual
perspective. Following the merge, we apply a post-processing module to the resulting 3D OBBs
to address any inconsistent roll and pitch caused by residual depth noise or partial observations.
As detailed in fig. 5, we perform a global gravity-alignment by estimating the global up-axis from
the primary planar structure (e.g., floor) and re-orienting the vertical axis of each instance. This
yields a fully consolidated, floor-aligned, and geometrically reliable set of instances, robustly
prepared for downstream verification, which we denote as Pmerged.
2) Confidence-Based Filtering & Refinement. We then refine the consolidated set Pmerged based
on the updated scores 𝑠𝑘. A tri-level decision rule is applied to each instance 𝑝𝑘:
Action(𝑝𝑘) =


keep,
𝑠𝑘≥𝜏high,
discard,
𝑠𝑘< 𝜏low,
verify,
𝜏low ≤𝑠𝑘< 𝜏high,
(2)
where thresholds are set to 𝜏high = 0.9 and 𝜏low = 0.8. For proposals in the verify band, we invoke
a VLM-based agent equipped with an image zoom-in tool and a SAM3 re-segmentation tool to
reassess the instance, yielding an updated confidence score 𝑠′
𝑘. We keep the proposal if 𝑠′
𝑘≥𝜏high;
otherwise, we discard it.
7

<!-- page 8 -->
(3) Categories Distribution of Spatial QA Pairs
(2) Distribution of the Whole Dataset
(1) Word Cloud of Different Object Classes
Figure 6 | Comprehensive Statistics of Holi-Spatial-4M. (1) Object Diversity: Word cloud
showing the long-tailed distribution of open-vocabulary categories. (2) Dataset Composition:
The inner ring displays source scenes (ScanNet [28], ScanNet++ [27], DL3DV [31]), while the
outer ring details the breakdown of over 4M generated spatial annotations. (3) Spatial QA
Taxonomy: Distribution of 1.25M spatial QA pairs, categorized into Camera-centric tasks (e.g.,
rotation, movement) and Object-centric tasks (e.g., distance, direction).
3) Upon establishing the final set of validated instances Pfinal, we proceed to generate dense
semantic annotations. For each instance 𝑝𝑘∈Pfinal, we retrieve the optimal source image
𝐼∗
𝑘corresponding to the retained high-confidence index from Step 1. Leveraging this most
informative visual perspective, we employ Qwen3-VL-30B to generate a fine-grained caption
for the object and procedurally synthesize a comprehensive suite of spatial QA pairs based
on predefined templates, covering diverse tasks such as 3D grounding, spatial reasoning, and
attribute identification.
4. Dataset
We introduce Holi-Spatial-4M, the first large-scale, multi-spatial-modal dataset constructed
using the proposed automated curation pipeline. As illustrated in fig. 6, this dataset represents a
significant advancement in scale, granularity, and task diversity compared to existing datasets.
Data Composition and Scale. Holi-Spatial-4M is derived from a diverse collection of raw video
streams sourced from ScanNet [28], ScanNet++ [27], and DL3DV-10K [31]. By processing these
streams through our geometric optimization and scene-level refinement stages, we have curated
over 12,000 optimized 3DGS scenes. As shown in fig. 6 (2), the dataset encompasses a total of
4 million+ high-quality spatial annotations. This includes 1.3M 2D instance masks, 1.2M 3D
grounding pairs, 320K 3D bounding boxes, and 320K detailed instance captions, significantly
surpassing the scale of manual annotations in the original datasets.
Open-Vocabulary Diversity. Unlike traditional datasets limited to a closed set of categories,
Holi-Spatial-4M leverages the open-world knowledge of VLMs to annotate a vast array of
objects. fig. 6 (1) presents a word cloud of the object categories, highlighting the dataset’s
coverage of fine-grained indoor items. This semantic richness is crucial for training models
capable of generalized spatial understanding in real-world environments.
Spatial Question-Answering Pairs. To empower Large Multimodal Language Models with
robust spatial reasoning capabilities, we generate 1.25M Spatial QA pairs, structured into a
comprehensive taxonomy. As detailed in fig. 6 (3), these QAs are divided into two primary
categories: 1) Camera-centric Tasks (blue sector), which challenge the model to understand
ego-centric spatial changes such as Camera Rotation and Movement Direction; 2) Object-centric
Tasks (red sector), which focus on allocentric reasoning, including Object-to-Object Distance,
8

<!-- page 9 -->
Table 2 | Quantitative results of 3D spatial understanding on ScanNet, ScanNet++ and DL3DV
datasets. We group methods by their input modality and task definition. 3D Det: 3D Object
Detection (AP@25, AP@50); 2D Seg: 2D Object Segmentation (IoU); Depth: Depth Estimation
(F1-score). Bold indicates best results. ↑: higher is better. ‘—’ denotes that the corresponding
metric is not available.
Method
ScanNet
ScanNet++
DL3DV
Depth (↑)
2D Seg. (↑)
3D Det. (↑)
Depth (↑)
2D Seg. (↑)
3D Det. (↑)
Depth (↑)
2D Seg. (↑)
3D Det. (↑)
F1
IoU
AP25
AP50
F1
IoU
AP25
AP50
F1
IoU
AP25
AP50
2D-VLM Methods
SAM3
–
0.63
–
–
–
0.50
–
–
–
0.66
–
–
SA2VA
–
0.64
–
–
–
0.25
–
–
–
0.44
–
–
3D-VLM Methods
SpatialLM
–
–
11.42
8.19
–
–
9.11
6.23
–
–
7.05
4.38
LLaVA-3D
–
–
9.13
6.86
–
–
12.2
4.80
–
–
6.83
4.11
SceneScript
–
–
8.97
3.54
–
–
9.86
4.42
–
–
5.65
3.98
3DGS-based Methods
M3-Spatial
0.32
0.22
–
–
0.39
0.11
–
–
0.23
0.13
–
–
LangSplat
0.19
0.36
–
–
0.21
0.06
–
–
0.18
0.24
–
–
Holi-Spatial (Ours)
0.98
0.66
76.60
67.00
0.89
0.64
81.06
70.05
0.78
0.71
62.89
52.67
Global/Local Direction, and Size Measurement. This balanced distribution ensures that models
trained on Holi-Spatial-4M develop a holistic understanding of 3D space.
5. Experiment
5.1. Framework Evaluation
Settings. To conduct a fair comparison of our framework, we randomly sampled 10 scenes
from each of the ScanNet, ScanNet++, and DL3DV-10K datasets. For these selected scenes,
we manually annotated 2D instance masks and 3D bounding boxes to serve as the evaluation
ground truth. For depth evaluation, we directly utilized the official ground truth depth maps
provided by the respective scenes.
Results. table 2 highlights Holi-Spatial as the sole framework capable of simultaneously generat-
ing high-quality predictions across 3D object detection, 2D segmentation, and depth estimation,
whereas prior works typically specialize in single modalities. First, in terms of geometric fidelity,
our method significantly outperforms 3DGS-based baselines with a Depth F1-score of 0.89 on
ScanNet++ [27] compared to 0.39 for M3-Spatial [40]. As shown in fig. 7, the quantitative depth
visualization indicates that both DA3 and our method already provide strong relative depth
cues compared with other baselines; to further highlight our multi-view advantage, the right
Image
GT
Ours
LangSplat
M3-spatial
Depth-Anything 3
Ours Refined Point Cloud
Depth-Anything 3 Point cloud
Figure 7 | Qualitative comparison of multi-view depth on ScanNet++. We visualize depth
maps from GT and baselines (LangSplat, M3-Spatial, Depth-Anything-V3) versus Holi-Spatial.
Right: point clouds obtained by multi-view back-projection, where Holi-Spatial produces cleaner
geometry with substantially fewer ghosting artifacts and floaters.
9

<!-- page 10 -->
Image
GT
Ours
Langsplat
M3-spatial
Sam3
Sa2VA
Label
Figure 8 | Qualitative comparison of open-vocabulary 2D instance segmentation on the same
scenes. Holi-Spatial yields sharper boundaries, more complete masks under occlusion, and
more accurate fine-grained categories.
GT 
SceneScript
LLAVA-3D
SpatialLM
Ours
lamp
bed
blanket
laptop
pillow
pillow
pillow
cabinet
sofa
Not detected
bed
lamp
bed
blanket
laptop
pillow
pillow
cabinet
Cloehtes_rack
Cloehtes_rack
Figure 9 | Qualitative comparison of 3D object detection on ScanNet++. We visualize predicted
3D bounding boxes from SceneScript, LLaVA-3D, SpatialLM, and Holi-Spatial. Holi-Spatial
produces tighter boxes and more correct categories.
panel visualizes the point cloud obtained by multi-view projection, where our results exhibit
almost no ghosting and significantly fewer floaters.
Regarding 2D segmentation, our framework demonstrates superior quality, reaching 0.64 IoU
compared to 0.25 for SA2VA [46]. As illustrated in the second row of fig. 8, SAM3 fails to
segment the distant mirror, whereas our method, by leveraging multi-view information, success-
fully segments such challenging and unclear instances. This improvement is attributed to the
synergy between geometric and semantic priors: while the VLM agent provides robust semantic
reasoning at the image level, our explicit integration of multi-view information compensates for
incomplete observations from single images.
Regarding 3D object detection, building on the high-quality geometry and refined semantics
above, Holi-Spatial achieves dominant performance in 3D object detection. On ScanNet++, we
report an AP25 of 81.06, exceeding the state-of-the-art 3D-VLM (LLaVA-3D, 12.2 AP25) by an
order of magnitude. Furthermore, fig. 9 provides a quantitative comparison of the predicted
instances, where Holi-Spatial recovers substantially more objects with accurate labels and tightly
aligned 3D bounding boxes, demonstrating strong semantic fidelity and geometric precision.
5.2. VLM Finetuning Evaluation
Settings. For the spatial reasoning task, we finetune Qwen3-VL [13] families using the 1.2M
spatial QA pairs in our Holi-Spatial-4M dataset for 1 epoch with batch size 1024, and evaluate
its performance on MMSI-Bench [18] and MindCube [19]. For 3D grounding task, we finetune
Qwen3-VL-8B [13] using 1.2M 3D grounding pairs in our Holi-Spatial-4M dataset. All models
10

<!-- page 11 -->
Table 3 | Quantitative results of Spatial Understanding QA tasks. Here we caompare different
models [55, 57, 25] on MMSI-bench [18] and MindCube [19] benchmarks. Bold indicates the
best performance.
Model
MMSI-Bench
MindCube
VST-SFT-3B [26]
30.2
35.9
Cambrian-S-3B [3]
25.2
32.5
VST-SFT-7B [26]
32.0
39.7
Cambrian-S-7B [3]
25.8
39.6
SpaceR-SFT-7B [55]
27.4
37.9
Intern3-VL-8B [57]
28.0
41.5
Spatial-MLLM [25]
27.0
32.1
Qwen3-VL-2B [13]
26.1
33.5
Qwen3-VL-2B + Ours
27.6
44.0
Qwen3-VL-8B [13]
31.1
29.4
Qwen3-VL-8B + Ours
32.6
49.1
Table 4 | Quantitative results of 3D Grounding on ScanNet++ [27] dataset.
Method
AP15
AP25
AP50
VST-7B-SFT [26]
17.29
14.50
11.20
Qwen3-VL-8B [13]
19.82
16.80
13.50
Qwen3-VL-8B + Ours
35.52
31.94
27.98
are trained for a single epoch with a total batch size of 1024. Training is conducted on 32 NVIDIA
H800 GPUs (80GB).
Spatial Reasoning. table 3 shows the QA training results. Our model consistently improves
both 2B and 8B models after finetuning, improving model’s spatial understanding thanks to our
high quality 3D curated data and QA pairs. More QA examples are provided in the Appendix.
3D Grounding. The results are summarized in table 4. In particular, our method achieves an
AP50 of 27.98, exceeding the strongest baseline by 14.48 AP points, which we attribute to fine-
tuning on our curated dataset with stronger 3D grounding supervision. As illustrated in fig. 11,
baseline models such as Qwen3-VL [13], trained primarily on single-view or anchor-view data,
exhibit a clear viewpoint bias and fail to reliably ground objects across different views or at
varying spatial depths.
5.3. Ablation
Geometric Training. In this section, we conduct ablation experiments on geometric training. As
shown in fig. 10, we compare results obtained using the original DA3 depth (ID. 1) and those
using our GS-refined depth (ID. 2) in the ScanNet++ scene. The visualization indicates that
directly using DA3 [12] depth often introduces ghosting artifacts when projecting from multiple
views. For example, in the shelf1curtain case, such ghosting leads to erroneous clustering and
incorrectly splits one instance into multiple instances, which is also clearly observable in the
3D viewer at the bottom right. In contrast, our refined depth is more multi-view consistent,
resulting in more reliable clustering and cleaner instance reconstruction. Moreover, as shown
11

<!-- page 12 -->
ID 2: w/ 3DGS Training
ID 1: w/o 3DGS Training
ID 3: w/o Conf. Filter
ID 5: w/ Conf. Filter
Conf. = 0.84 < 0.9
ID 4: w/o Agent Recall
ID 5: w/ Agent Recall
Shelf
3D viewer
3D viewer
3D viewer
3D viewer
Hair dryer
Curtain
Shelf
Curtain
Vending
Machine
Vending
Machine
Hair dryer
3D Det.
3D Det.
Cart
Cart
Conf. = 0.81 < 0.9
Figure 10 | Stage-wise visualization of scene-level refinement. Detailed discussion is include
in Section 5.3
Lamp
Lamp
Floor
Floor
Lamp
Lamp
Lamp
Lamp
Floor
Floor
Lamp
GT
Pred
GT
Pred
1
2
3
4
1
2
GT
Pred
GT
Pred
Table
Table
Table
Table
Raw Image
Ours-8B
Qwen3-VL-8B
Figure 11 | Given the same input images and queried objects, our predicted boxes align better
with ground truth, indicating improved spatial localization.
Table 5 | Ablation study on depth refinement, confidence filtering, and agent recall. ID. 1–2
(Step 1) compare depth sources, while ID. 3–5 (Step 3) evaluate post-processing modules. 𝑃25
and 𝑅25 denote precision and recall under IoU 25%.
ID
DA3
Depth
3DGS
Training
Conf.
Filter
Agent
Recall
𝑃25
𝑅25
Step 1: Geometric Optimization
1
✓
✗
✓
✓
0.13
0.31
2
✓
✓
✓
✓
0.81
0.89
Step 3: Scene-Level Refinement
3
✓
✓
✗
✗
0.35
0.74
4
✓
✓
✓
✗
0.67
0.69
5
✓
✓
✓
✓
0.81
0.89
in table 5, the GS-refined depth substantially outperforms direct use of DA3 [12] depth.
Confidence Filter. As shown in table 5, we conduct ablation studies on the SAM3 [29] confidence
filter mechanism in the ScanNet++ scene. The confidence filter improves precision (from 0.35 to
0.67) by suppressing false positives: as illustrated in fig. 10 (ID. 3/ID. 4), SAM3 may assign an
12

<!-- page 13 -->
incorrect category (e.g., classifying multiple machines as vending machine), and confidence-based
filtering helps remove such misclassified predictions.
Agent Refinement. According to our tests, as shown in table 5, Confidence filtering also reduces
recall (from 0.74 to 0.69) when comparing ID. 3 and ID. 4, since it tends to discard visually
challenging yet correct instances. For example, in fig. 10, ID. 4, it shows objects such as a hair
dryer among clutter and a cart heavily occluded by buckets and cleaning tools can receive
low confidence and thus be mistakenly filtered out. To address this trade-off, we introduce an
agent-based (VLM) verification step to reconsider borderline cases instead of discarding them
directly, which recovers true positives with low confidence. Overall, the confidence filter and
the agent recall complement each other and together yield the best precision–recall balance.
6. Conclusion
In this paper, we present Holi-Spatial, a fully automated pipeline that converts raw videos
into high-fidelity 3D geometry and holistic spatial annotations. By combining 3DGS-based
geometric optimization, open-vocabulary perception, and scene-level lifting/refinement, it
produces multi-level supervision (rendered depth, 2D masks, 3D boxes, instance captions, and
spatial QA). We further release Holi-Spatial-4M with 12K optimized 3DGS scenes, 1.3M masks,
320K 3D boxes/captions, 1.2M 3D grounding instances, and 1.2M QA pairs, and show on
ScanNet/ScanNet++/DL3DV that fine-tuning VLMs on Holi-Spatial-4M consistently improves
3D grounding and spatial reasoning.
Impact Statement
This work introduces Holi-Spatial, a fully automated data curation pipeline that converts raw
video streams into high-fidelity 3D geometry together with holistic spatial annotations, enabling
the construction of large-scale spatially-aware multimodal datasets.
However, Holi-Spatial still has limitations. The pipeline relies on multiple upstream compo-
nents and per-scene optimization, which can be computationally expensive and may degrade
under challenging videos (e.g., limited viewpoints, motion blur, heavy occlusion, or dynamic
objects). Open-vocabulary semantic labeling may also inherit biases or errors from foundation
models, making robust verification and uncertainty estimation important future directions. We
plan to improve efficiency (e.g., adaptive early stopping and better confidence-based validation),
expand to broader domains and longer video contexts, and build stronger benchmarks for
holistic 3D spatial understanding.
This work uses publicly available data sources and does not require collecting new sen-
sitive personal data. Nonetheless, similar technology could be misused for privacy-invasive
reconstruction of personal spaces; we encourage responsible deployment with consent, data
governance, and appropriate safeguards.
13

<!-- page 14 -->
References
[1] P. Langley, “Crafting papers on machine learning,” in Proceedings of the 17th International Conference
on Machine Learning (ICML 2000), P. Langley, Ed.
Stanford, CA: Morgan Kaufmann, 2000, pp.
1207–1216.
[2] T. M. Mitchell, “The need for biases in learning generalizations,” Computer Science Department,
Rutgers University, New Brunswick, MA, Tech. Rep., 1980.
[3] S. Yang, J. Yang, P. Huang, E. Brown, Z. Yang, Y. Yu, S. Tong, Z. Zheng, Y. Xu, M. Wang, D. Lu,
R. Fergus, Y. LeCun, L. Fei-Fei, and S. Xie, “Cambrian-s: Towards spatial supersensing in video,”
arXiv preprint arXiv:2511.04670, 2025.
[4] M. J. Kearns, “Computational complexity of machine learning,” Ph.D. dissertation, Department of
Computer Science, Harvard University, 1989.
[5] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry
grounded transformer,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025,
pp. 5294–5306.
[6] N. Keetha, N. Müller, J. Schönberger, L. Porzi, Y. Zhang, T. Fischer, A. Knapitsch, D. Zauss, E. Weber,
N. Antunes, J. Luiten, M. Lopez-Antequera, S. R. Bulò, C. Richardt, D. Ramanan, S. Scherer, and
P. Kontschieder, “MapAnything: Universal feed-forward metric 3D reconstruction,” in International
Conference on 3D Vision (3DV).
IEEE, 2026.
[7] R. S. Michalski, J. G. Carbonell, and T. M. Mitchell, Eds., Machine Learning: An Artificial Intelligence
Approach, Vol. I.
Palo Alto, CA: Tioga, 1983.
[8] R. O. Duda, P. E. Hart, and D. G. Stork, Pattern Classification, 2nd ed.
John Wiley and Sons, 2000.
[9] N. N. Author, “Suppressed for anonymity,” 2021.
[10] A. Newell and P. S. Rosenbloom, “Mechanisms of skill acquisition and the law of practice,” in
Cognitive Skills and Their Acquisition, J. R. Anderson, Ed. Hillsdale, NJ: Lawrence Erlbaum Associates,
Inc., 1981, ch. 1, pp. 1–51.
[11] A. L. Samuel, “Some studies in machine learning using the game of checkers,” IBM Journal of Research
and Development, vol. 3, no. 3, pp. 211–229, 1959.
[12] H. Lin, S. Chen, J. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang, “Depth anything 3: Recovering
the visual space from any views,” arXiv preprint arXiv:2511.10647, 2025.
[13] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo,
Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu,
C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun,
J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang,
M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou,
J. Zhou, Y. Zhu, and K. Zhu, “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.
[14] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu,
M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang,
Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.
[15] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth,
K. Millican et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint
arXiv:2312.11805, 2023.
[16] D. He, X. Gao, H. Li, J. Han, and D. Zhang, “Strider: Navigation via instruction-aligned structural
decision space optimization,” arXiv preprint arXiv:2511.00033, 2025.
14

<!-- page 15 -->
[17] Z. Cai, Y. Wang, Q. Sun, R. Wang, C. Gu, W. Yin, Z. Lin, Z. Yang, C. Wei, O. Qian et al., “Holistic
evaluation of multimodal llms on spatial intelligence,” arXiv preprint arXiv:2508.13142, 2025.
[18] S. Yang, R. Xu, Y. Xie, S. Yang, M. Li, J. Lin, C. Zhu, X. Chen, H. Duan, X. Yue et al., “Mmsi-bench: A
benchmark for multi-image spatial intelligence,” arXiv preprint arXiv:2505.23764, 2025.
[19] B. Yin, Q. Wang, P. Zhang, J. Zhang, K. Wang, Z. Wang, J. Zhang, K. Chandrasegaran, H. Liu,
R. Krishna et al., “Spatial mental modeling from limited views,” in Structural Priors for Vision
Workshop at ICCV’25, 2025.
[20] J. Feng, J. Zeng, Q. Long, H. Chen, J. Zhao, Y. Xi, Z. Zhou, Y. Yuan, S. Wang, Q. Zeng et al., “A survey
of large language model-powered spatial intelligence across scales: Advances in embodied agents,
smart cities, and earth science,” arXiv preprint arXiv:2504.09848, 2025.
[21] Z. Zhang, H. Li, Y. Dai, Z. Zhu, L. Zhou, C. Liu, D. Wang, F. E. Tay, S. Chen, Z. Liu et al., “From
spatial to actions: Grounding vision-language-action model in spatial foundation priors,” arXiv
preprint arXiv:2510.17439, 2025.
[22] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang et al., “Spatialvla:
Exploring spatial representations for visual-language-action model,” arXiv preprint arXiv:2501.15830,
2025.
[23] L. Jiang, Y. Mao, L. Xu, T. Lu, K. Ren, Y. Jin, X. Xu, M. Yu, J. Pang, F. Zhao et al., “Anysplat: Feed-
forward 3d gaussian splatting from unconstrained views,” ACM Transactions on Graphics (TOG),
vol. 44, no. 6, pp. 1–16, 2025.
[24] X. Wang, L. Liu, Y. Cao, R. Wu, W. Qin, D. Wang, W. Sui, and Z. Su, “Embodiedgen: Towards a
generative 3d world engine for embodied intelligence,” arXiv preprint arXiv:2506.10600, 2025.
[25] D. Wu, F. Liu, Y.-H. Hung, and Y. Duan, “Spatial-mllm: Boosting mllm capabilities in visual-based
spatial intelligence,” arXiv preprint arXiv:2505.23747, 2025.
[26] R. Yang, Z. Zhu, Y. Li, J. Huang, S. Yan, S. Zhou, Z. Liu, X. Li, S. Li, W. Wang et al., “Visual spatial
tuning,” arXiv preprint arXiv:2511.05491, 2025.
[27] C. Yeshwanth, Y.-C. Liu, M. Nießner, and A. Dai, “Scannet++: A high-fidelity dataset of 3d indoor
scenes,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 12–22.
[28] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-
annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE conference on computer
vision and pattern recognition, 2017, pp. 5828–5839.
[29] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr,
A. Huang et al., “Sam 3: Segment anything with concepts,” arXiv preprint arXiv:2511.16719, 2025.
[30] A. Gupta, P. Dollar, and R. Girshick, “Lvis: A dataset for large vocabulary instance segmentation,” in
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 5356–5364.
[31] L. Ling, Y. Sheng, Z. Tu, W. Zhao, C. Xin, K. Wan, L. Yu, Q. Guo, Z. Yu, Y. Lu et al., “Dl3dv-10k: A
large-scale scene dataset for deep learning-based 3d vision,” in Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2024, pp. 22 160–22 169.
[32] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta,
C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation
image-text models,” Advances in neural information processing systems, vol. 35, pp. 25 278–25 294, 2022.
[33] J. Deng, T. He, L. Jiang, T. Wang, F. Dayoub, and I. Reid, “3d-llava: Towards generalist 3d lmms with
omni superpoint transformer,” in Proceedings of the Computer Vision and Pattern Recognition Conference,
2025, pp. 3772–3782.
15

<!-- page 16 -->
[34] C. Zhu, T. Wang, W. Zhang, J. Pang, and X. Liu, “Llava-3d: A simple yet effective pathway to
empowering lmms with 3d-awareness,” arXiv preprint arXiv:2409.18125, 2024.
[35] Y. Mao, J. Zhong, C. Fang, J. Zheng, R. Tang, H. Zhu, P. Tan, and Z. Zhou, “Spatiallm: Training large
language models for structured indoor modeling,” arXiv preprint arXiv:2506.07491, 2025.
[36] D. Zheng, S. Huang, Y. Li, and L. Wang, “Learning from videos for 3d world: Enhancing mllms with
3d vision geometry priors,” arXiv preprint arXiv:2505.24625, 2025.
[37] Z. Cai, R. Wang, C. Gu, F. Pu, J. Xu, Y. Wang, W. Yin, Z. Yang, C. Wei, Q. Sun et al., “Scaling spatial
intelligence with multimodal foundation models,” arXiv preprint arXiv:2511.13719, 2025.
[38] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3d gaussian splatting for real-time radiance
field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.
[39] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin,
J. Clark et al., “Learning transferable visual models from natural language supervision,” in Interna-
tional conference on machine learning.
PmLR, 2021, pp. 8748–8763.
[40] X. Zou, Y. Song, R.-Z. Qiu, X. Peng, J. Ye, S. Liu, and X. Wang, “3d-spatial multimodal memory,” in
The Thirteenth International Conference on Learning Representations, 2025.
[41] H. Li, R. Qin, Z. Zou, D. He, B. Li, B. Dai, D. Zhang, and J. Han, “Langsurf: Language-embedded
surface gaussians for 3d scene understanding,” arXiv preprint arXiv:2412.17635, 2024.
[42] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, “Langsplat: 3d language gaussian splatting,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 051–
20 060.
[43] Z. Li, S. Yao, T. Wu, Y. Yue, W. Zhao, R. Qin, A. F. Garcia-Fernandez, A. Levers, and X. Zhu, “Ulsr-gs:
Ultra large-scale surface reconstruction gaussian splatting with multi-view geometric consistency,”
2025. [Online]. Available: https://arxiv.org/abs/2412.01402
[44] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao, and G. Zhang, “Pgsr:
Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction,” arXiv preprint
arXiv:2406.06521, 2024.
[45] J. L. Schönberger and J.-M. Frahm, “Structure-from-motion revisited,” in Conference on Computer
Vision and Pattern Recognition (CVPR), 2016.
[46] H. Yuan, X. Li, T. Zhang, Y. Sun, Z. Huang, S. Xu, S. Ji, Y. Tong, L. Qi, J. Feng et al., “Sa2va:
Marrying sam2 with llava for dense grounded understanding of images and videos,” arXiv preprint
arXiv:2501.04001, 2025.
[47] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt,
S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
[48] B. Zhang, C. Fang, R. Shrestha, Y. Liang, X.-X. Long, and P. Tan, “Rade-gs: Rasterizing depth in
gaussian splatting,” ACM Transactions on Graphics.
[49] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson,
E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer,
“Sam 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024. [Online].
Available: https://arxiv.org/abs/2408.00714
[50] Z. Yu, T. Sattler, and A. Geiger, “Gaussian opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes,” ACM Transactions on Graphics, 2024.
[51] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting for geometrically accurate
radiance fields,” in SIGGRAPH 2024 Conference Papers.
Association for Computing Machinery,
2024.
16

<!-- page 17 -->
[52] Y. Gao, H. Li, J. Chen, Z. Zou, Z. Zhong, D. Zhang, X. Sun, and J. Han, “Citygs-x: A scalable
architecture for efficient and geometrically accurate large-scale scene reconstruction,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 27 187–27 196.
[53] A. Avetisyan, C. Xie, H. Howard-Jenkins, T.-Y. Yang, S. Aroudj, S. Patra, F. Zhang, D. Frost, L. Hol-
land, C. Orme et al., “Scenescript: Reconstructing scenes with an autoregressive structured language
model,” in European Conference on Computer Vision.
Springer, 2024, pp. 247–263.
[54] H. Li, Z. Zou, F. Liu, X. Zhang, F. Hong, Y. Cao, Y. Lan, M. Zhang, G. Yu, D. Zhang et al.,
“Iggt: Instance-grounded geometry transformer for semantic 3d reconstruction,” arXiv preprint
arXiv:2510.22706, 2025.
[55] K. Ouyang, Y. Liu, H. Wu, Y. Liu, H. Zhou, J. Zhou, F. Meng, and X. Sun, “Spacer: Reinforcing mllms
in video spatial reasoning,” arXiv preprint arXiv:2504.01805, 2025.
[56] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “Internvl:
Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 24 185–24 198.
[57] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao et al., “Internvl3:
Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv
preprint arXiv:2504.10479, 2025.
[58] R. Zhao, Z. Zhang, J. Xu, J. Chang, D. Chen, L. Li, W. Sun, and Z. Wei, “Spacemind: Camera-guided
modality fusion for spatial reasoning in vision-language models,” arXiv preprint arXiv:2511.23075,
2025.
17

<!-- page 18 -->
Appendix
Table of Contents
A Why 3D Multi-view Merges
19
B
QA Examples
19
C Prompt Templates
19
D Detail of Scene-Level Refinement Algorithm
21
18

<!-- page 19 -->
A. Why 3D Multi-view Merges
Although the masks generated by SAM3 provide instance-level attributes at the image level,
occlusions often cause SAM3 to fragment the same object into multiple instances—for example,
in section A, a complete bed is detected as two separate beds. Thus, directly adopting SAM3’s
instance predictions is unreliable; robust 3D clustering and fusion remain essential.
For effective 3D clustering and merging, high geometric accuracy—namely, precise depth
estimations—is crucial. As demonstrated in fig. 12, without the multi-view geometric constraint,
estimated depth maps suffer from ghosting and aliasing artifacts, causing different objects to
become spatially entangled and ultimately mis-merged as a single instance.
Furthermore, as shown in fig. 15, we provide more qualitative examples of 3D grounding
box predictions across diverse object categories, underscoring the robustness and generalization
capability of our method.
3D Object Clustering
W Geometric Training
W/O Geometric Training
Figure 12 | Ablation study on 3D multi-view merging and depth refinement. (a) 3D geometric
clustering corrects the fragmentation caused by occlusions in SAM3 image-level instance pre-
dictions, merging separated segments into unified object instances. (b) Using GS-refined depth
constraints enables more accurate multi-view lifting, which avoids false merges and preserves
the integrity of spatial object instances.
B. QA Examples
We present examples of each of our curated QA question type in fig. 16, including: camera
rotation, camera movement direction, camera movement distance, camera-object direction,
camera-object distance, camera-object distance (global frame), object-object distance, object
measurement, object-object direction (local frame), object-object direction (global frame) . As
shown in fig. 17, training on the QA dataset can greatly enhance many tasks in MindCube [19]
and MMSI-Bench [18], especially perspective changing and egocentric imagination related
questions. Additionally, we use VLM to describe the obejct’s appearance, and use it as a way
to reference the object in the question. This improves the appearance and grounding ability of
models in spatial question answering, as shown in the bottom-left case in fig. 17.
C. Prompt Templates
We provide the prompt templates for using agent verification in VLM-based confidence filtering
and refinements in fig. 13, and the prompt templates for object grounding in 2D image in fig. 14.
19

<!-- page 20 -->
SYSTEM:
You are a visual label verifier for a single masked object.
USER: 
"You will be given three images: (1) the original image, (2) an overlay image where the target object is highlighted with a 
colored mask, and (3) a zoomed-in pair showing the original object and the highlighted object.\n"
f'The user-provided label/category to verify is: "{category}".\n\n'
"Task:\n"
"Decide whether the given label correctly describes the object indicated by the HIGHLIGHTED region.\n"
"If it is correct, ACCEPT. If it is incorrect (wrong category), REJECT and provide the correct label you believe fits best.\n\n”
"Guidelines (consistency-and-rewrite mindset):\n"
"1) Focus ONLY on the object covered by the HIGHLIGHTED region. Ignore other objects outside the mask.\n"
"2) Identify the core visible noun/object class of the masked object (the best label).\n"
"3) Treat extra modifiers in the label (color/material/size/position/relations) as constraints ONLY if they are clearly visible.\n"
" If modifiers are wrong or unverifiable but the core object class is correct, still ACCEPT.\n"
"4) If the mask covers only a part of an object but the object class is still clear, judge by the most likely full object.\n"
"5) If the masked region is ambiguous, too small, or does not correspond to a recognizable object, REJECT and set 
predicted_label to \"unknown\".\n"
"6) Use common-sense synonyms/hypernyms: accept reasonable equivalents (e.g., \"sofa\" vs \"couch\").\n"
" If the label is too specific and not verifiable (e.g., exact brand/model/species), prefer a more general correct label.\n\n"
"Output format (VERY IMPORTANT):\n"
"Return ONLY a JSON object on a single line with these keys:\n"
" - decision: \"ACCEPT\" or \"REJECT\"\n"
" - predicted_label: a short English noun phrase for the masked object class (e.g., \"chair\", \"person\", \"car\").\n"
"Rules:\n"
" - If decision is ACCEPT, predicted_label should be exactly the provided label (category) or its closest normalized form.\n"
" - If decision is REJECT, predicted_label must be your best guess of the correct label, or \"unknown\" if unclear.\n"
"Do not output any extra text after the JSON object. The JSON should be the final part of your response."
Prompt Template for  Agent Verify
Figure 13 | Prompt template used for Agent verification.
SYSTEM:
You are a helpful assistant.
USER: 
"You are labeling objects in a video frame. Identify ALL visible objects in the image. "
"Use concise, lowercase English nouns as object class names. Be instance-aware when deciding what is present, "
"but output only class names (no instance IDs) and list each class at most once.\n\n"
"Granularity rules (avoid over-fragmentation):\n"
"- Use whole-object categories at a practical level (e.g., sofa, pillow, bed, shoe, broom).\n"
"- Do NOT split a single object into parts or components (e.g., do not use 'broom handle', 'broom head'; use 'broom').\n"
"- Prefer common household object categories; avoid overly fine-grained subtypes.\n\n"
"No skipping rule:\n"
"- You MUST label every distinct object you can see.\n"
"- If you are unsure of an exact subtype, choose the closest reasonable, still-concrete supercategory (e.g., 'chair' instead of a 
specific chair type; "
"'bottle' instead of a specific brand; 'bag' instead of an uncertain bag type). Do not omit objects.\n"
"- Do NOT use non-informative words like 'object', 'thing', 'stuff', 'unknown'.\n\n"
"Consistency with previous frames (canonical vocabulary):\n"
f"- Existing labels from previous frames: {previous_text}\n"
"- If any object in the current image matches a previous label in meaning (including synonyms, singular/plural, or minor 
wording differences), "
"you MUST reuse the previous label exactly and MUST NOT introduce a new synonym.\n"
"- Only create a new label if no existing label can reasonably cover the same object meaning.\n\n"
"Output format:\n"
"- Return a single line of lowercase, comma-separated class names only.\n"
"- Include classes that appeared in previous frames if they also appear here.\n"
"- No explanations, no extra text."
Prompt Template for  Object Grounding
Figure 14 | Prompt template used for object grounding in 2D image.
20

<!-- page 21 -->
Figure 15 | Additional 3D detection visualizations. We show Holi-Spatial predictions across
diverse indoor scenes, demonstrating multi-view consistent localization and robust detection.
D. Detail of Scene-Level Refinement Algorithm
We provide the scene-level refinement algorithm in algorithm 1. The proposed scene-level
refinement algorithm operates in three main stages to lift 2D open-vocabulary predictions into a
globally consistent 3D representation.
First, we project frame-level 2D masks generated by SAM3 into 3D space using depth maps
and camera intrinsics, constructing localized point clouds and 3D bounding boxes for each
semantic label.
Next, to resolve cross-view redundancies, we spatially merge these candidates by clustering
observations that share the same label and exhibit a 3D Intersection-over-Union (IoU) above a
predefined threshold 𝜏iou.
Finally, we employ a hybrid verification mechanism on the canonical view of each merged
instance group: high-confidence instances (𝑠★≥𝜏high) are directly accepted, low-confidence noise
(𝑠★< 𝜏low) is discarded, and ambiguous cases are rigorously arbitrated by a Vision-Language
Model (VLM) using the cropped 2D canonical view.
This pipeline effectively filters out spurious detections, yielding a robust and validated set
of 3D instances O for the scene.
21

<!-- page 22 -->
Given view A and view B, consider the relative rotation from A to B 
expressed in view A's camera frame. Which SINGLE rotation 
direction is the most prominent? 
A. turn right     B. tilt up
C. tilt down       D. turn left
- A
Camera Rotation
Camera Motion Direction
What is the primary camera motion direction from view A to view B 
in view A's coordinate?
A. right   B. backward
C. left     D. forward
- D
Does the camera translation from view A to view B exceed 1.79 
meters.?
-
no
Camera Motion Distance
What is the camera translation distance from view A to view B 
(meters)?
-
1.331
In image A, the object described as "dense green leafy plant with 
fine, feathery foliage, growing in a shallow container, positioned next 
to a glass dome and a red-patterned pot" is the plant. When you were 
taking the photo in Image B, where is the plant area relative to you?
A. Back   B. Right
C. Left   D. Front
- D
Camera Object Direction
Camera Object Distance
Camera Object Direction (Global)
In image A, the object described as "rectangular mirror with white frame, 
mounted on wooden panel wall, reflecting part of white door and hanging coat" 
is the mirror.In image B, the object described as "ceiling-mounted circular 
fluorescent light fixture with bright white illumination and smooth surface 
finish" is the light.The direction of mirror relative to image A is south.
What is the direction of light region relative to image B?
A. South-West   B. North-East
C. West              D. South-East
- B
Object Object Distance
In image A, the object described as "dark brown wooden wardrobe with 
textured panels, one open section revealing stacked clothes, and a visible 
handle on the left door" is the wardrobe. In image B, the object described as 
"dark brown wooden wardrobe door with recessed panel design and vertical 
handle, part of a built-in closet unit" is the door. Estimate the 3D metric 
distance (in meters) between the centers of these two physical objects.
- 0.65m
In image A, the object described as "wooden shelf with light brown 
finish and visible grain, holding various jars and bottles" is the shelf.
Locate the same physical object in image B. Estimate the 3D metric 
distance (in meters) from the camera position of image B (camera 
center) to the "shelf" (center point).
- 1.8m
In image A, the object described as “rectangular mirror with thin frame, 
reflecting white soap dispenser and tiled wall, mounted above sink” is the 
mirror. In image B, the object described as “off-white ceramic bathroom 
sink with oval basin, metallic faucet, and circular drain, mounted against tiled 
wall” is the sink. Which is longer (consider the longest side of the object)? 
mirror or sink?
A. Both are about the same length
B. Cannot tell
C. sink
D. mirror
- C
Object Measurement
Object Object Direction (Local / Global)
In image A, the object described as “Lush green leafy plant with dense foliage, in a 
blue pot, placed on a white windowsill” is the plant. In image A, the object described 
as “white metallic radiator with vertical fins, smooth surface, and small purple cloth 
draped on top edge” is the radiator. In image B, the object described as 
“transparent cylindrical container with metallic silver lid, positioned among other 
kitchen items on a shelf” is the bottle. You are positioned at plant and face radiator.
In which direction is bottle relative to you?
A. Back-Left
B. Front-Left
C. Right
D. Front-Right - D
Figure 16 | Examples of 10 types of curated spatial QA pairs in Holi-Spatial.
Rotation
Among
Around
These three images (image 1, 2, and 3) show 
the same scene from three different 
viewpoints. The image 2 was taken after 
turning the camera 90 degrees to the right 
(clockwise) from the position of image 1. For 
image 3, the camera was turned another 90 
degrees right, so it's basically facing the 
opposite direction of image 1. Based on these 
three images: If I am standing at the same 
spot and facing the same direction as shown in 
image 3, then I turn 90 degrees to the right, 
what is to my left? 
A. Door B. Wooden frame mirror C. Sink
- A
Based on these two views 
showing the same scene: in 
which direction did I move 
from the first view to the 
second view? 
A. Directly left 
B. Directly right 
C. Diagonally forward and left 
D. Diagonally forward and right
- D
Based on these two views 
showing the same scene: in 
which direction did I move 
from the first view to the 
second view? 
A. Directly right 
B. Directly left 
C. Diagonally forward and 
right 
D. Diagonally forward and left
- D
20.0%
63.9%
101.3%
How many sponge strips are on the 
sculpture 
in the picture?
A. 8
B. 9
C. 7
D. 6
- A
Attribute Appearance
62.5%
Motion Camera
The images are taken continuously from a 
first-person perspective. In which 
direction are you moving?
A: Left while moving backward
B: Forward to the left
C: Forward to the right
D: Right while moving backward
- C
48.0%
Pos-Cam-Object
When you took the second photo, where 
was the toilet in relation to you?
A: back right
B: front right
C: front left
D: back left
- D
41.6%
Figure 17 | Training with our curated QA data improves the three types of spatial reasoning
tasks in MindCube [19] (upper row) as well as in MMSI-Bench [18] (lower row).
22

<!-- page 23 -->
Algorithm 1 Scene-level Refinement via 3D Merging and VLM-based Verification
Require: keyframes {𝐼𝑡}𝑇
𝑡=1, camera parameters {Π𝑡}𝑇
𝑡=1, refined depth maps {𝐷𝑡}𝑇
𝑡=1, per-frame
label sets {L𝑡}𝑇
𝑡=1.
Require: SAM3(𝐼𝑡, ℓ) →{(𝑚𝑡,ℓ,𝑘, 𝑠𝑡,ℓ,𝑘)}
𝐾𝑡,ℓ
𝑘=1, where 𝑚is a 2D mask and 𝑠∈[0, 1] is confidence.
Require: thresholds 𝜏iou, 𝜏low < 𝜏high.
Ensure: validated instance set O; each item (ℓ, 𝐵𝑔, 𝑡★, 𝑚★).
1: (1) Lift 2D instances to 3D candidates (per label)
2: Initialize candidate pools Cℓ←∅for all labels ℓ.
3: for 𝑡= 1 to 𝑇do
4:
for all ℓ∈L𝑡
5:
{(𝑚𝑡,ℓ,𝑘, 𝑠𝑡,ℓ,𝑘)}
𝐾𝑡,ℓ
𝑘=1 ←SAM3(𝐼𝑡, ℓ)
6:
for 𝑘= 1 to 𝐾𝑡,ℓdo
7:
𝑃𝑡,ℓ,𝑘←BackProject(𝑚𝑡,ℓ,𝑘, 𝐷𝑡, Π𝑡)
8:
𝐵𝑡,ℓ,𝑘←BBox(𝑃𝑡,ℓ,𝑘)
9:
𝑐𝑡,ℓ,𝑘←(𝑃𝑡,ℓ,𝑘, 𝐵𝑡,ℓ,𝑘, 𝑠𝑡,ℓ,𝑘, 𝑡, 𝑚𝑡,ℓ,𝑘)
10:
Cℓ←Cℓ∪{𝑐𝑡,ℓ,𝑘}
11:
end for
12: end for
13: (2) Multi-view merging within each label (3D IoU)
14: Initialize merged groups G ←∅.
15: for all label ℓsuch that Cℓ≠∅
16:
Gℓ←MergeByIoU3D(Cℓ, 𝜏iou)
17:
for all group 𝑔∈Gℓ
18:
G ←G ∪{(ℓ, 𝑔)}
// each 𝑔is a set of candidates 𝑐𝑡,ℓ,𝑘
19: (3) Confidence gating and VLM-based verification
20: Initialize O ←∅.
21: for all instance group (ℓ, 𝑔) ∈G
22:
𝑐★←arg max𝑐∈𝑔𝑠(𝑐)
// canonical view
23:
(𝑃★, 𝐵★, 𝑠★, 𝑡★, 𝑚★) ←𝑐★
24:
𝑃𝑔←Ð
𝑐∈𝑔𝑃(𝑐)
25:
𝐵𝑔←BBox(𝑃𝑔)
26:
if 𝑠★≥𝜏high then
27:
O ←O ∪{(ℓ, 𝐵𝑔, 𝑡★, 𝑚★)}
28:
else if 𝑠★< 𝜏low then
29:
continue
// discard
30:
else
31:
𝑦←VLMVerify(𝐼𝑡★, 𝑚★, ℓ)
32:
if 𝑦= true then
33:
O ←O ∪{(ℓ, 𝐵𝑔, 𝑡★, 𝑚★)}
34:
end if
35:
end if
36: return O
23
