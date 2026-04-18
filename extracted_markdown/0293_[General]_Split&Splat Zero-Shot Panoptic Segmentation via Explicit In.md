<!-- page 1 -->
Split&Splat: Zero-Shot Panoptic Segmentation via Explicit
Instance Modeling and 3D Gaussian Splatting
Leonardo Monchieri, Elena Camuffo, Francesco Barbato, Pietro Zanuttigh, Simone Milani∗
University of Padova
Padova, Italy
Figure 1: Split&Splat pipeline. Starting from multi-view images, segmentation masks are generated, enabling the split of a scene
into single objects with explicit geometric consistency. These subsets are then independently reconstructed via 3D Gaussian
splatting, merged, and augmented with object-level descriptors, enabling object-level reasoning, retrieval and editing, within a
coherent global scene.
Abstract
3D Gaussian Splatting (GS) enables fast and high-quality scene
reconstruction, but it lacks an object-consistent and semantically
aware structure. We propose Split&Splat, a framework for panoptic
scene reconstruction using 3DGS. Our approach explicitly models
object instances. It first propagates instance masks across views
using depth, thus producing view-consistent 2D masks. Each ob-
ject is then reconstructed independently and merged back into the
scene while refining its boundaries. Finally, instance-level semantic
descriptors are embedded in the reconstructed objects, support-
ing various applications, including panoptic segmentation, object
retrieval, and 3D editing. Unlike existing methods, Split&Splat
tackles the problem by first segmenting the scene and then recon-
structing each object individually. This design naturally supports
downstream tasks and allows Split&Splat to achieve state-of-the-
art performance on the ScanNetv2 segmentation benchmark.
CCS Concepts
• Computing methodologies →Rendering; Volumetric mod-
els; Scene understanding.
Keywords
Scene Understanding, Gaussian Splatting, Panoptic Segmentation
∗Code: https://github.com/LTTM/Split_and_Splat,
Contact: {leonardo.monchieri, elena.camuffo, francesco.barbato, pietro.zanuttigh,
simone.milani}@unipd.it
1
Introduction
3D Gaussian splatting [Kerbl et al. 2023] represents a milestone
advancement in the field of volume rendering, enabling ultra-fast
novel-view synthesis with a photorealistic appearance. The grow-
ing popularity of this approach has paved the way for its application
in various tasks [Qin et al. 2024; Schiavo et al. 2025; Wang et al.
2025b; Ye et al. 2024]. Traditionally, the technique focuses solely on
pixel rendering and does not capture the semantics of scene con-
tent, which is crucial in applications such as object detection, open
vocabulary semantic segmentation, and panoptic segmentation.
Nowadays, this limitation can be effectively addressed by leverag-
ing Vision Foundation Models (VFMs) [Oquab et al. 2023; Radford
et al. 2021; Ravi et al. 2024; Siméoni et al. 2025]. These systems
are capable of producing dense, high-level feature embeddings
that support robust mask generation and region-level classification.
Leveraging such capabilities, recent works [Cheng et al. 2024; Li
et al. 2025a,b; Wang et al. 2025a; Wu et al. 2024; Ye et al. 2024]
have begun integrating GS-based 3D reconstructions with VFM-
derived segmentation cues, thereby enabling the propagation of
object masks and category labels within the reconstructed scene.
However, achieving view-consistent segmentation in a GS repre-
sentation remains a challenging problem. Inconsistencies in mask
assignment or feature aggregation across different viewpoints often
lead to disjointed object representations and degraded segmenta-
tion quality. Such issues propagate to downstream tasks, ultimately
limiting the effectiveness of GS-based approaches for reliable 3D-
aware segmentation and scene understanding.
Compared to approaches that simply fit the semantic features
from the 2D images into an already existing 3DGS, our 2-stage ap-
proach (Split&Splat) integrates reconstruction and segmentation
arXiv:2602.03809v1  [cs.GR]  1 Feb 2026

<!-- page 2 -->
Monchieri et al.
tasks into a single, coherent, and highly-performing framework.
Indeed, explicit modeling of object instances enables a sharper and
more reliable separation of objects in the 3D scene, facilitating the
integration of semantic information as instance-level descriptors.
More specifically, our pipeline reconstructs the scene as follows:
(1) we extract 2D instance masks from the multi-view images of the
scene; (2) the 2D object masks are propagated and refined by exploit-
ing depth cues, ensuring multi-view and geometric consistency; (3)
each object is independently reconstructed using Gaussian splat-
ting; (4) a semantic descriptor is embedded in each segment. Over-
all, our pipeline achieves a segmented and semantically structured
Gaussian splatting representation of the scene that is well-suited
for downstream applications such as panoptic segmentation, object
retrieval, and 3D editing.
2
Related Work
Recent works have extended Gaussian splatting beyond photore-
alistic rendering toward semantic and open vocabulary 3D scene
understanding. Early efforts incorporated per-Gaussian feature em-
beddings to associate each Gaussian with semantic or instance-level
descriptors. Gaussian Grouping [Ye et al. 2024] introduced class-
aware features to enable semantic segmentation directly within
the explicit Gaussian representation. SAGE [Schiavo et al. 2025]
uses semantic-aware Gaussian splatting for real-time optimization.
GAGS [Peng et al. 2024] introduces a granularity-aware 2D to 3D
feature distillation scheme, while [Chacko et al. 2025] integrates
a 2D segmentation step in propagating semantic features to 3D
reconstruction. GOI [Qu et al. 2024] learns an optimizable manifold
in an open vocabulary (CLIP-like) semantic space to identify Gaus-
sians of Interest, enabling text-guided selection and segmentation.
Overall, these methods confirm the 3DGS’s capability to produce
explicit and differentiable models able to generate real-time seman-
tic masks without the heavy computation typically associated with
implicit NeRF-style models.
A more recent line of research focuses on language-driven seg-
mentation by aligning Gaussian features with multimodal embed-
dings. LangSplat [Li et al. 2025b; Qin et al. 2024] and LEGaus-
sians [Shi et al. 2024] encode CLIP-aligned features inside each
Gaussian, enabling text-conditioned segmentation, retrieval, and
editing in 3D space. VALA [Wang et al. 2025a] proposes a visibility-
aware aggregation of language features by integrating text embed-
dings across multiple views while weighting them according to
Gaussian visibility, thereby improving open vocabulary segmenta-
tion under occlusions and partial observations. InstanceGaussian [Li
et al. 2025a] jointly learns appearance and instance semantics within
a unified Gaussian representation. Unlike language-driven or purely
feature-distillation approaches, it focuses on balancing appearance
and semantic attributes through a clustered representation, where
multiple appearance Gaussians share instance features. Addition-
ally, it introduces a bottom-up, category-agnostic instance aggrega-
tion strategy based on over-segmentation and graph connectivity,
enabling adaptive instance discovery without relying on predefined
object counts or category priors.
While the contribution of these approaches cannot be under-
stated, they are still limited by two major issues: (1) they distill
2D features into the 3DGS framework without ensuring that the
per-view embeddings maintain 3D spatial consistency, and (2) the
addition of dense descriptors and features drastically increases the
memory required to store each Gaussian parameter, leading to a
resource-intensive training process. To tackle these issues, our ap-
proach addresses the instance segmentation task in a bottom-up
fashion. Rather than directly learning a unified 3DGS representa-
tion that must be split at test time, we model all object instances
independently before aggregating them into a global representa-
tion. This allows us to achieve higher quality segments and reduce
the memory burden by embedding only sparse (one per segment)
descriptors in the Gaussians, if necessary.
3
Method
We introduce Split&Splat, a novel 3D scene reconstruction frame-
work based on Gaussian splatting that directly embeds semantic de-
scriptors into 3DGS objects, supporting multiple downstream tasks
such as panoptic segmentation. Our method consists of two main
stages: instance segmentation (Split) and reconstruction (Splat).
In the Split stage (Sec. 3.1), a Structure-from-Motion (SfM) al-
gorithm processes the initial multi-view data to create a 3D point
cloud model of the scene and to estimate the camera parameters. At
the same time, a monocular depth estimator associates each view
with its corresponding depth map. Depth and geometry are used
together to create an initial 3D Gaussian splatting representation.
Note that this step is only necessary when depth, pose estimation,
and point cloud are missing; when they are available, it can be
skipped. In parallel, 2D instance segmentation masks for each view
are generated and subsequently refined by exploiting the coarse
reconstruction to achieve multi-view and geometric consistency.
In the Splat stage (Sec. 3.2), the refined masks are used to iden-
tify object instances and reconstruct each of them via 3DGS. The
separate objects are then merged sequentially into the global scene
while embedding semantic information. Finally, visual descriptors
are computed on a set of masked views cropped around the in-
stance. These descriptors are then applied to the instance-level
reconstructions, producing semantically informed 3D representa-
tions. The final output, composed of Gaussian objects associated
with semantic visual descriptors, supports a variety of downstream
tasks in addition to panoptic segmentation, such as object retrieval
and scene editing.
3.1
Split: 3D Instance Segmentation
In this section, we describe the Split stage, where we obtain a set of
segmentation maps from multi-view images. Using the estimated
depth maps along with an initial dense point cloud, we refine the
masks and enforce consistency across the 2D segmentations and
the 3D scene. A schematic overview is shown in Fig. 2.
3.1.1
Mask generation. Given a set of 𝐾multi-view representa-
tions I ={𝐼𝑘}𝐾
𝑘=1 with resolution 𝐻×𝑊, capturing a 3D scene, we
compute per-view instance segmentation masks using a segmenter
𝑆𝐼(·). This yields 𝐾segmentation mask sets M𝑘= {𝑀𝑘,𝑗}, each
containing 𝐽𝑘= |J𝑘| binary masks 𝑀𝑘,𝑗(with 𝑗∈J𝑘) associated
with the object instances 𝑗detected in view 𝑘. The cardinality 𝐽𝑘de-
pends on the number of visible instances in each view 𝑘. The masks
𝑀𝑘,𝑗are automatically initialized using ping points from SAM2’s

<!-- page 3 -->
Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting
Figure 2: During Split, multi-view images are processed to estimate depth and instance masks, which are propagated in 3D to
produce refined, view-consistent segmentations. A, B, and C denote the output of Split, which serves as an input to Splat.
fine-grid policy [Ravi et al. 2024]. The masks are then merged us-
ing a coarse-to-fine approach [Ye et al. 2024] to promote larger
masks (see Fig. 5 for some visual examples). Note that standard 2D
instance segmenters struggle to recover objects that are partially
occluded or missing in a given view; therefore, at this step, the
mask labels are arbitrary and require further processing to ensure
cross-view consistency. These issues are reflected in some of the
failure cases of approaches like DEVA [Cheng et al. 2023] or those
using foundation model descriptors [Oquab et al. 2023; Radford
et al. 2021; Ravi et al. 2024; Siméoni et al. 2025], which directly em-
bed semantic features into Gaussians without particular attention
to the underlying 3D structures. As a result, they cannot ensure
cross-view consistency before performing instance segmentation.
To tackle this issue, we introduce a mask propagation technique,
detailed in Sec. 3.1.3, to produce high-quality, multi-view consistent
instance masks without any manual input.
3.1.2
Initial 3D reconstruction. Concurrently with the mask gener-
ation, we process the images in I using the COLMAP SfM algorithm
[Schonberger and Frahm 2016] to obtain a sparse point cloud re-
construction 𝑃𝑠𝑝𝑎𝑟𝑠𝑒∈R3×𝑛. Similarly, we use a monocular depth
estimator 𝐸𝐷(·) (refer to Sec. 4 for more details) to obtain a set
of 𝐾depth maps 𝐷𝑘. The two are used as initialization for depth-
regularized Gaussian splatting, which will generate a dense point
cloud of the scene 𝑃𝑑𝑒𝑛𝑠𝑒∈R3×𝑛.
3.1.3
Mask propagation. After the reconstruction, we begin iter-
atively assigning the instance labels in J to the points in 𝑃𝑑𝑒𝑛𝑠𝑒,
updating the labels of the points after each iteration to enforce 3D
consistency. The result is a dense and view-consistent point cloud
𝑃𝑙𝑎𝑏𝑒𝑙𝑒𝑑⊆𝑃𝑑𝑒𝑛𝑠𝑒(floaters or inner-object points completely covered
by surface points are removed), constructed as follows.
For each view 𝑘, the points 𝑝∈𝑃𝑑𝑒𝑛𝑠𝑒are mapped onto the
image plane using the camera projection Π𝑘(·), to obtain their
pixel coordinates (𝑤,ℎ) and projected depth ˆ𝑑. Points lying outside
camera 𝑘’s field of view are discarded:
ˆ𝑃𝑘={𝑝|𝑝∈𝑃𝑑𝑒𝑛𝑠𝑒, (𝑤,ℎ, ˆ𝑑) =Π𝑘(𝑝), 0≤𝑤<𝑊, 0≤ℎ<𝐻} .
(1)
In the following, we omit the ranges of ℎand 𝑤for ease of notation.
To ensure the visibility of the selected points in the current view,
we preserve only the points satisfying |𝐷𝑘(𝑤,ℎ) −ˆ𝑑| < 𝜏𝑑𝑒𝑝𝑡ℎ; we
empirically set 𝜏𝑑𝑒𝑝𝑡ℎ= 0.02 (see Sec. 5.4.2 for more details). This
constraint yields the set of surface-consistent points for view 𝑘as:
𝑃𝑘= {𝑝| 𝑝∈ˆ𝑃𝑘, (𝑤,ℎ, ˆ𝑑) = Π𝑘(𝑝), |𝐷𝑘(𝑤,ℎ) −ˆ𝑑| < 𝜏𝑑𝑒𝑝𝑡ℎ} .
(2)
These sets 𝑃𝑘⊆𝑃𝑑𝑒𝑛𝑠𝑒(one for each view 𝑘) are used to progres-
sively update the masks M𝑘starting from frame 𝑘= 1. More in
detail, we extract the subsets 𝑃𝑘,𝑗of points 𝑝∈𝑃𝑘whose projections
fall inside the eroded 𝑀𝑘,𝑗(avoiding border effects):
𝑃𝑘,𝑗= {𝑝| 𝑝∈𝑃𝑘, 𝑀𝑘,𝑗(𝑤,ℎ) = 1} .
(3)
This allows us to compute the sets 𝑃∗
𝑘= Ð
𝑗∈J𝑘𝑃𝑘,𝑗⊂𝑃𝑘of all
labeled points (i.e., non-background) visible by camera 𝑘. To refine
𝑃∗
𝑘, we apply DBSCAN [Ester et al. 1996], removing isolated points.
The sets 𝑃𝑘,𝑗are used to sequentially re-project the semantic labels
𝑗∈J𝑘−1 from 𝑃𝑘−1,𝑗onto the next view 𝑘, thus obtaining virtual
masks 𝑀∗
𝑘,𝑗. If there is no intersection between 𝑀∗
𝑘,𝑗and any mask in
M𝑘, the warped region corresponds to a different object. Otherwise,
labels 𝑖in masks 𝑀𝑘,𝑖are remapped into the label 𝑗that maximizes
the intersection with 𝑀∗
𝑘,𝑗(see Alg. 1 in the Appendix for details).
These labels are then transferred to 𝑃∗
𝑘for 3D-consistent processing.
More specifically, to create a scene-level labeling, we assign a
weight vector 𝑤𝑝∈R| J| to each 3D point 𝑝, where 𝑤𝑝[𝑗] repre-
sents the aggregated scores of label 𝑗∈J from different views.
Whenever 𝑤𝑝is uninitialized, the view score is set to 1 + 𝜆init,
with 0 < 𝜆init < 1, thus ensuring priority to the first view in
case of ties. Otherwise, each view adds a score equal to 1. Af-
ter all views have been processed, the array 𝑤𝑝is normalized,
and the final instance label 𝑙𝑝for 𝑝is assigned via majority vot-
ing, i.e., 𝑙𝑝= argmax𝑗∈J 𝑤𝑝[𝑗]. We discard points whose normal-
ized frequency lies below the threshold 𝜏𝑙𝑎𝑏𝑒𝑙. We empirically set
𝜏𝑙𝑎𝑏𝑒𝑙= 0.7. This strategy ensures that the final labeling is consis-
tent across views and robust to local segmentation errors. Once
all views have been processed, the resulting labeled point cloud
𝑃𝑙𝑎𝑏𝑒𝑙𝑒𝑑is re-projected to update 2D masks by merging overseg-
mented instances, ensuring 3D-consistency across views (Fig. 6).
The output of the Split stage is a set of view-consistent instance
masks
˜
M. The number of masks may differ across views compared
to the original masks M, since the object indices are now global.

<!-- page 4 -->
Monchieri et al.
Figure 3: During Splat, each instance is reconstructed independently using 3DGS, then merged into a global model enriched
with per-instance descriptors. A, B, and C denote the output of Split, which serves as an input to Splat.
This stage resolves over-segmentation when objects share over-
lapping mask support across views, ensuring that multiple masks
corresponding to the same object are merged. However, when ob-
jects do not overlap in any view, residual over-segmentation may
persist and can be addressed with a refinement stage (see Sec. 3.2.2
for details). Any region of the dense point cloud that remains unla-
beled after propagation is treated as background.
3.2
Splat: Object Reconstruction
In the Splat stage (Fig. 3), the refined instance masks are used to
reconstruct each object independently via Gaussian splatting. A
separate 3DGS reconstruction is performed for each object using
the masked multi-view images; then, splats are merged back into
the full scene (as detailed in Section 3.2.3).
3.2.1
Per-instance 3D Gaussian splatting. More in detail, given the
refined instance masks
˜
M ={ ˜
M𝑙}𝐿
𝑙=1, grouped by instance label 𝑙
(independently of the source view 𝑘), we extract a multi-view image
set I𝑙for each instance 𝑙. Note that the images in I𝑙are extracted
from I by masking each instance 𝑙, ensuring that only that instance
remains visible. These images are used to perform instance-specific
3D Gaussian splatting, resulting in a set of G𝑙reconstructions. To
initialize the splatting algorithm, we select a subset of points from
𝑃𝑙𝑎𝑏𝑒𝑙𝑒𝑑that have 𝑙as their label.
3.2.2
Mask reprojection and refinement. After instance reconstruc-
tion, we refine the 2D masks to enhance 3D geometric consistency.
For each view 𝑘and instance 𝑙, we extract the set of Gaussians
G𝑘,𝑙visible from view 𝑘. These sets are used to produce new masks
𝑀𝑔𝑠
𝑘,𝑙by rendering the scene with full-opacity (i.e., 𝛼= 1). Then, we
apply a greedy 2D sampling procedure (similar to KMeans++) to the
projections of G𝑘,𝑙: starting from the point closest to the instance
centroid (the average of the Gaussians’ positions), we iteratively
select the farthest point in the current sample set. This yields a com-
pact and uniformly distributed set of 2D locations that cover the
spatial extent of the visible instance. These sampled points are used
to prompt the segmenter 𝑆𝐼(·), producing a refined mask 𝑀𝑠𝑎𝑚
𝑘,𝑙.
At this stage, we have two candidate segmentation masks: the
propagated mask from the initial segmentation stage ˜𝑀𝑘,𝑙and the
refined mask 𝑀𝑠𝑎𝑚
𝑘,𝑙. We compare these masks by computing the
Intersection-over-Union (IoU) with mask 𝑀𝑔𝑠
𝑘,𝑙, rendered from the
reconstructed Gaussians. If ˜𝑀𝑘,𝑙exists, we prefer it over 𝑀𝑠𝑎𝑚
𝑘,𝑙
when-
ever its IoU is higher. If it is not available (e.g., due to missed detec-
tions or because it has been removed in the geometry consistency
check of Sec. 3.1.3), we keep 𝑀𝑠𝑎𝑚
𝑘,𝑙
only if its IoU with 𝑀𝑔𝑠
𝑘,𝑙exceeds
a threshold 𝜏𝑖𝑜𝑢(empirically set to 0.95). This process yields a fi-
nal refined mask ˆ𝑀𝑘,𝑙that is view-consistent and coherent with
the scene geometry, improving segmentation quality, especially
in cases where objects are small, occluded, or visually ambiguous
(visual examples are provided in Fig. 4).
3.2.3
Instance merging. To assemble a complete scene reconstruc-
tion from the independent instances G𝑙while preserving object
boundaries, we merge them based on spatial overlap. For each
instance 𝑙, we compute an axis-aligned (in the world reference sys-
tem) 3D bounding box 𝐵𝑙around G𝑙and define a collision matrix
𝐶∈R𝐿×𝐿, where each entry 𝐶𝑎𝑏measures the amount of overlap
between the two instances G𝑎and G𝑏:
𝐶𝑎𝑏=
1
|G𝑎|
∑︁
𝑔∈G𝑎
1𝐵𝑏(𝑔) .
(4)
Note that we normalize the entries of 𝐶to allow comparison be-
tween instances of different sizes.
We iteratively select the pair of instances (𝑎,𝑏) with the highest
normalized overlap and merge them until a GS representation of
the complete scene is achieved. The merging is performed directly
at the Gaussian level by merging the sets G𝑎and G𝑏. Since each
instance was reconstructed from masked views, boundary regions
may contain occlusion artifacts, such as dark or oversized Gaussians.
To suppress them, we reinitialize the opacity of all Gaussians in
the merged object to 𝛼= 0 (following [Kerbl et al. 2023]), and run
a short Gaussian splatting refinement with densification disabled.
This stage allows for the smoothing of collision borders and the
removal of Gaussians that correspond to occlusions.
To refine object boundaries during merging, we include a mask
consistency term in the reconstruction objective:
ℓ= ℓ𝑅𝐺𝐵+ 𝑤𝑚𝑎𝑠𝑘· ℓ𝑚𝑎𝑠𝑘,
(5)
where ℓ𝑅𝐺𝐵is the standard GS optimization objective, while ℓ𝑚𝑎𝑠𝑘
encourages alignment between the rendered instance mask 𝑀𝑔𝑠
𝑘,𝑙

<!-- page 5 -->
Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting
and the chosen per-view refined mask ˆ𝑀𝑘,𝑙(Sec. 3.2.2):
ℓ𝑚𝑎𝑠𝑘= 1
𝐾𝐿
𝐾
∑︁
𝑘=1
𝐿
∑︁
𝑙=1
𝑀𝑔𝑠
𝑘,𝑙−ˆ𝑀𝑘,𝑙

1 .
(6)
The weight 𝑤𝑚𝑎𝑠𝑘is gradually increased after each merge, as
detailed in Sec. 4, progressively reinforcing boundary sharpness
and suppressing residual occlusions. We merge instance pairs in
parallel whenever they do not share collisions, update the colli-
sion matrix, and repeat until no further merges are required. This
produces the final full-scene Gaussian representation in which all
instances are jointly reconstructed, and boundary transitions are
geometrically and semantically consistent. The result of the Splat
stage is a fully assembled scene in which object boundaries are ge-
ometrically consistent, and each Gaussian carries its corresponding
instance label. While our approach is not explicitly designed for
open vocabulary segmentation, Sec 5.2 explains how we adapted
it to this task by assigning a semantic descriptor 𝑓𝑜𝑏𝑗to each G𝑙
computed from a multi-view feature extractor 𝐸𝐹(·) (implemented
using CLIP [Radford et al. 2021], see Fig. 3).
4
Implementation Details
The proposed pipeline has been implemented starting from the
original Gaussian splatting [Kerbl et al. 2023] codebase. In the first
stage (Sec. 3.1.1), whenever depth information was not available in
the dataset, the Murre approach [Guo et al. 2025] was employed
for monocular depth estimation. This method was selected over
other approaches [Yang et al. 2024] due to its scale consistency with
respect to the SfM point cloud (obtained via COLMAP [Schonberger
and Frahm 2016]) and consequently, with respect to Gaussian splat-
ting. Masks are instead computed with SAM2 [Ravi et al. 2024],
which ensures accurate mask boundaries (see Fig. 5 for a visual-
ization of different auto-segmentation settings). Finally, semantic
visual descriptors are extracted using CLIP [Radford et al. 2021].
During instance reconstruction, we run Gaussian splatting for 10𝑘
iterations in the LERF dataset, while the denser ScanNetv2 requires
only 1𝑘iterations. During instance merging, we run the splatting
for 1𝑘iterations after each merge. Weight 𝑤𝑚𝑎𝑠𝑘starts from a value
of 0.05 in the first composition and increases by 0.1 after each (par-
allel) merging step until it reaches a maximum of 0.25. We run
our experiments on a single NVIDIA RTX 3090, with a maximum
VRAM usage of ∼10GB (in the largest ScanNetv2 scene 0000_00).
5
Results
The proposed approach produces a set of instance-level Gaussian
splats, each containing a single object along with its semantic de-
scription. This opens the way to downstream tasks in the field
of scene understanding, i.e., instance, panoptic, and open vocabu-
lary segmentation. Additionally, instance-level modeling enables
in-scene editing, e.g., adding, removing, or duplicating objects.
We begin by presenting segmentation and reconstruction results
on the ScanNetv2 dataset [Dai et al. 2017] in Sec. 5.1. Then we
consider a few downstream tasks that our approach is not directly
designed for, i.e., open vocabulary segmentation (Sec. 5.2) and scene
editing (Sec. 5.3), evaluated on the LERF dataset [Kerr et al. 2023].
Finally, Sec. 5.4 presents an ablation study on the components.
Table 1: Per-scene instance segmentation accuracy on Scan-
Netv2 for our approach and InstanceGS [Li et al. 2025a]. Best
in bold.
Instance GS
Split&Splat
Scene
mIoU
mAcc(25)
mIoU
mAcc(25)
0000_00
51.71
85.07
43.97
76.12
0062_00
50.76
85.33
63.46
100.00
0070_00
48.86
82.61
48.70
86.96
0097_00
58.03
82.61
64.98
100.00
0140_00
54.32
91.49
59.76
95.74
0200_00
45.07
68.42
59.98
84.21
0347_00
57.70
89.29
69.81
96.43
0400_00
48.58
73.91
57.93
82.61
0590_00
47.57
78.33
48.58
86.67
0645_00
40.42
67.44
46.70
75.58
mean
50.30
80.45
56.39
88.43
5.1
Instance Segmentation Results
We followed the setup introduced by OpenGaussian [Wu et al. 2024]
and InstanceGS [Li et al. 2025a] to evaluate 3D segmentation on
the ScanNetv2 [Dai et al. 2017] dataset using ground-truth (GT)
instance-level annotations. Specifically, we report instance-level
mIoU and mAcc (with a threshold of 25%) on the 10 scenes selected
in these works. The mIoU is computed over all GT instances in a
scene, while the mAcc(𝑥) measures the fraction of instances iden-
tified with at least 𝑥% IoU. The ScanNetv2 benchmark provides a
3D labeled point cloud to be used as GT for metrics computation.
Following the evaluation pipeline of the competitors, we compute
our labeled point cloud by assigning new instance labels to the 3D
points of the ground-truth point cloud (𝑃𝐺𝑇) using the output of
the Split&Splat pipeline. In detail, the Gaussian means are treated
as points, and labels are assigned to each nearest-neighbor in 𝑃𝐺𝑇.
In Tab. 1, we compare our method with the state-of-the-art
Gaussian-based approach InstanceGS [Li et al. 2025a], which relies
on RGB images and SAM-generated masks as input. Our approach
achieves an average mIoU of 56.39%, improving by more than 6%
over the competitor. Per-scene results confirm this trend, with
Split&Splat outperforming InstanceGS in 8 out of 10 scenes in
terms of mIoU, and in 9 out of 10 when considering mAcc(25).
Qualitative results presented in Fig. 7 further support these find-
ings. Compared to InstanceGS, Split&Splat produces sharper and
more geometrically consistent object boundaries, especially in re-
gions affected by occlusions or clutter. This visual improvement is
a direct consequence of explicitly modeling each object instance
and enforcing multi-view consistency during the reconstruction
process. Regarding limitations, Tab. 1 also shows that our pipeline
degrades slightly in highly object-dense scenes. In particular, scenes
0000_00 and 0070_00 contain 123 and 90 object instances, respec-
tively, compared to an average of approximately 25 in the other
scenes. In these cases, the increased instance density can lead to
missed detections or label conflicts during propagation, which, in
turn, affects the final segmentation accuracy.

<!-- page 6 -->
Monchieri et al.
Image 𝐼
Rendered mask 𝑀𝑔𝑠
SAM2 mask 𝑀𝑠𝑎𝑚
Original mask ˜𝑀
Image 𝐼
Rendered mask 𝑀𝑔𝑠
SAM2 mask 𝑀𝑠𝑎𝑚
LERF/teatime
“old camera"
LERF/figurines
“stuffed bear"
LERF/ramen
“sake bottle"
LERF/waldo
kitchen“spatula"
Figure 4: Examples of mask refinement process. For “old camera” the refined mask is 𝑀𝑠𝑎𝑚while for “stuffed bear” ˜𝑀present
a lower IoU with 𝑀𝑔𝑠. For “sake bottle” and “spatula”, ˜𝑀is not present and is discovered throughout the refinement process.
Selected masks for each object are highlighted with green borders.
Table 2: Open vocabulary results on LERF [Kerr et al. 2023].
Best in bold, second best underlined, third dashed underline.
Method
Average
mIoU
mAcc(25)
LERF [Kerr et al. 2023]
10.35
13.64
LEGaussian [Shi et al. 2024]
16.21
23.82
OpenGaussian [Wu et al. 2024]
38.36
51.43
SuperGSeg [Liang et al. 2024]
35.94
52.02
Dr. Splat [Jun-Seong et al. 2025]
43.29
64.30
InstanceGS [Li et al. 2025a]
43.87
61.09
CAGS [Sun et al. 2026]
50.79
69.62
VoteSplat [Jiang et al. 2025]
50.10
67.38
Occam’s LGS [Cheng et al. 2024]
47.22
74.84
VALA [Wang et al. 2025a]
58.02
82.85
Split&Splat
55.68
73.05
5.2
Open Vocabulary Segmentation Results
As mentioned, while our approach does not natively support the
open vocabulary segmentation task, it can be adapted by assigning
a VLM descriptor to each of the identified instances in a scene.
More specifically, we mask the original views I𝑙using M𝑔𝑠and
crop them around the instance; the remaining part of the images is
either blurred or filled in with one color, as explained in Sec. 5.4.3.
This allows us to compute a VLM description for each image and
average them to obtain a semantically descriptive representation
for each instance that is not influenced by its surroundings. Note
that, as discussed in Sec. 2, recent works that integrate semantic
descriptors into GS [Li et al. 2025b; Shi et al. 2024] typically re-
duce descriptor dimensionality to limit memory usage, which may
compromise semantic expressiveness. In contrast, our sparse for-
mulation allows us to preserve their full dimensionality (in the
Appendix, we propose an additional method to compute descrip-
tors associated with Gaussians rather than with instances, to solve
denser tasks). To perform the evaluation, given an open vocabu-
lary query, we first find the best correlation among the existing
3D Gaussian instances and then extract their corresponding multi-
view 2D segmentation masks from the rasterization pipeline. We
then compute the correlation between each instance descriptor
and the textual query using cosine similarity. The correspondences
with a similarity lower than a threshold 𝜏𝑐𝑜𝑟𝑟are discarded (see
Sec. 5.4.4 for details). This object-centric representation enables
more accurate and coherent mask extraction than InstanceGS (see
Fig. 8), contributing to the improved performance reported in Tab. 2,
where mIoU and mAcc(25) are computed on the LERF [Kerr et al.
2023] dataset, following the protocol of state-of-the-art approaches
such as LangSplat [Li et al. 2025b] and OpenGaussian [Wu et al.
2024] (a more detailed version, including per-scene results, is in the
Appendix). Remarkably, even if the approach has not been designed
for this task, Split&Splat ranks second in terms of mIoU and third
in terms of mAcc(25) among a set of ten very recent works (while
InstanceGS falls short by more than 10 points in both metrics), thus
showing how it can be adapted to achieve competitive performance
alongside state-of-the-art methods explicitly developed for the task.
5.3
Editing Results
Since the output of Split&Splat is an instance-labeled GS recon-
struction, it naturally enables object-level scene editing operations,
including object removal, object duplication, object movement, and
object recoloring. To validate this capability, we apply a variety
of editing operations to the LERF figurines scene and obtain vi-
sually compelling results. Representative examples generated by
combining our approach with external editing tools such as Su-
perSplat [SuperSplat Community 2026] and Splatviz [Barthel et al.
2024] are shown in Fig. 9.
5.4
Ablation Study
We assess the contribution of each component of the proposed
approach through an ablation study conducted on scene 0062_00 of
ScanNetv2, as reported in Tab. 3. Following InstanceGS, we applied
image subsampling (1 : 20). This strategy results in only a 0.15%
performance drop while significantly reducing the computational
cost. This confirms that the proposed approach remains reliable,
even on datasets with a large number of acquisitions. In addition,
we show that mask refinement is fundamental: without it, the mIoU
drops to 48.69% (a decrease of ∼15%).
5.4.1
Exploiting hint-based mask generation. To evaluate our mask
generation and propagation pipeline, we compare its performance
against a hint-based (hb) baseline. Following [Ravi et al. 2024], a
small number of views are provided with hand-crafted inputs in

<!-- page 7 -->
Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting
Table 3: Ablation study for 3D instance segmentation on
ScanNetv2 scene 0062_00.
Ablation
mIoU mAcc(25) mAcc(50)
baseline
63.46
100.00
70.83
w/o subsampling
64,00
100.00
75.00
w/o mask refinement
48.69
50.00
75.00
hb masks w/o propagation
51.84
87.50
54.17
hb masks w/ propagation
58.50
95.83
66.67
Table 4: Propagation performance metrics for scene 0062_00
at different depth thresholds 𝜏𝑑𝑒𝑝𝑡ℎ.
depth threshold 𝜏𝑑𝑒𝑝𝑡ℎ
0.1m
0.02m
0.001m
mIoU
62.33
63.98
51.98
mAcc(25)
66.67
100.00
54.17
mAcc(50)
95.83
70.83
70.83
Table 5: Open vocabulary segmentation accuracy with vary-
ing background masking color on LERF scenes.
Dataset
Mask-black
Mask-blur
Mask-white
mIoU mAcc(25)
mIoU mAcc(25) mIoU mAcc(25)
Figurines
60.28
75.28
61.80
78.22
60.28
75.28
Ramen
58.29
74.27
58.89
75.95
51.53
67.38
Teatime
59.82
75.90
59.43
75.04
62.15
78.93
Waldo Kitchen
32.09
48.81
42.58
62.98
28.61
43.10
Mean
52.62
68.57
55.68
73.05
50.64
66.17
Table 6: Performance metrics and average percentage of in-
stances with label 𝑙for different correlation thresholds 𝜏𝑐𝑜𝑟𝑟.
correlation threshold 𝜏𝑐𝑜𝑟𝑟
0.01
0.02
0.025
0.05
0.075
0.1
0.2
mIoU
50.68
55.30
56.55
57.35
57.20
56.23
56.64
mAcc(25)
65.45
72.31
74.30
76.00
76.03
74.39
74.87
mAcc(50)
61.21
66.83
68.72
69.18
69.20
67.56
68.04
avg % lab. 𝑙
7.69% 12.24% 15.67% 39.24% 67.53% 86.23% 100.00%
the form of arbitrary point prompts, i.e., a set of 2D coordinates
used to approximate the location of object instances in a subset of
views. These prompts are then propagated to the remaining views
by treating them as consecutive video frames. Results in the lower
part of Tab. 3, highlights how the lack of 3D consistency in the
resulting masks propagates to the final reconstruction and leads
to a performance drop of 11.6%. Moreover, since our propagation
pipeline treats these masks as uncorrelated, it partially compensates
for segmentation errors, yielding a mIoU improvement of 6.34%
when applied on top of hint-based masks.
5.4.2
Surface depth threshold. During the propagation step, 𝜏𝑑𝑒𝑝𝑡ℎ
defines the depth threshold used to determine whether a point is
considered part of the surface, which will be projected back onto
the 2D mask. Higher values retain more points, whereas lower
values discard more, resulting in a more conservative policy with
fewer but more reliable surface points. Tab. 4 compares the labeled
point cloud 𝑃𝑙𝑎𝑏𝑒𝑙𝑒𝑑(i.e., the output of the propagation step) with
the ground truth for different values of 𝜏𝑑𝑒𝑝𝑡ℎ. The results show
that an intermediate value yields the best performance, avoiding
the under-segmentation caused by excessively high thresholds and
the over-segmentation introduced by overly low values.
5.4.3
Open vocabulary descriptors ablation. Since our descriptors
are extracted from instance-centric crops defined by the binary
masks produced by Split&Splat, we investigate three masking
strategies for computing CLIP [Radford et al. 2021] descriptors:
background blurring, white background, and black background. As
reported in Tab. 5, background blurring yields the best overall per-
formance, suggesting that preserving some surrounding context
provides useful information about the instance. In contrast, white
and black backgrounds lead to different outcomes depending on the
instance color. In particular, lighter objects tend to produce less in-
formative descriptors when placed on a white background, whereas
a black background can yield more discriminative descriptors.
5.4.4
Open vocabulary correlation threshold. Given the instance-
centric nature of our approach, we use a threshold 𝜏𝑐𝑜𝑟𝑟to associate
textual descriptors (𝑓𝑡𝑒𝑥𝑡) with object-level descriptors (𝑓𝑜𝑏𝑗). This
association is computed using the cosine distance𝑑𝑐𝑜𝑠=𝑑(𝑓𝑡𝑒𝑥𝑡, 𝑓𝑜𝑏𝑗).
For each object, we compute a correlation score 𝛿= |𝑑∗
𝑐𝑜𝑠−𝑑𝑐𝑜𝑠|
where 𝑑∗
𝑐𝑜𝑠denotes the minimum cosine distance for a given query.
A match between an instance and a textual query is established
when 𝛿<𝜏𝑐𝑜𝑟𝑟. As shown in Tab. 6, increasing this threshold im-
proves overall segmentation performance. However, a larger thresh-
old effectively connects each textual query with multiple instances,
resulting in test-time behavior that maximizes the mIoU indepen-
dently for each ground-truth mask. Thus, we adopt a more conser-
vative, albeit suboptimal strategy, setting 𝜏𝑐𝑜𝑟𝑟= 0.02.
6
Conclusion
In this work, we introduced Split&Splat, a novel 3D Gaussian
splatting pipeline for panoptic scene understanding. By explicitly
modeling each object instance prior to reconstructing the full 3D
scene, our method preserves instance-level separation throughout
the reconstruction process, resulting in geometrically and semanti-
cally consistent representations. Extensive quantitative and quali-
tative evaluations on ScanNetv2 demonstrate the effectiveness of
our approach, with Split&Splat consistently outperforming state-
of-the-art competitors. Thanks to its modular design, the proposed
framework is highly flexible and naturally supports a variety of
downstream tasks, including object retrieval and object-level scene
editing (e.g., removal, duplication, and recoloring). Experiments on
the LERF benchmark further confirm the generalizability of our
method, showing that Split&Splat achieves competitive perfor-
mance even in open-vocabulary settings. Despite these encouraging
results, there is still room for improvement, particularly in instance
re-labeling and in the identification of object parts. Addressing
these challenges will further strengthen the applicability of our
framework and constitute an important direction for future work.

<!-- page 8 -->
Monchieri et al.
Coarser ping grid
Coarse ping grid
Medium ping grid
Fine ping grid
Final masks
LERF/Figurines
LERF/Teatime
ScanNetv2/0062_00
ScanNetv2/0347_00
Figure 5: Multi-resolution mask generation. Exploiting SAM2’s multi-scale grid ping point generation, we produce several
masks that are subsequently merged to obtain the final result.
Ground truth mesh
𝑃𝑙𝑎𝑏𝑒𝑙𝑒𝑑
Overlay
ScanNetv2/0062_00
ScanNetv2/0347_00
Figure 6: Labeled point cloud (𝑃𝑙𝑎𝑏𝑒𝑙𝑒𝑑) produced by mask propagation stage, ensuring 3D consistency.

<!-- page 9 -->
Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting
Original mesh
InstanceGS
Split&Splat
GT
ScanNetv2/0062
ScanNetv2/0097
ScanNetv2/0347
ScanNetv2/0400
Figure 7: Visualization comparison of category-agnostic 3D instance segmentation result. Split&Splat outperforms InstanceGS,
accurately distinguishing the different 3D objects.
Figurines
“Miffy, Green apple,
Handle, Old camera"
InstanceGS
Split&Splat
Ramen
“Weavy noodles, Chopstick,
Sake cup, Kamaboko"
Teatime
“Stuffed bear, Napkin,
Plate, Tea in glass"
Figure 8: Open-vocabulary query (retrieval) on LERF.
Removal
“jake” removed
Before
After
Duplication
“red apple”
Recolor
“elephant"
Figure 9: Editing capabilities of Split&Splat.

<!-- page 10 -->
Monchieri et al.
A
Appendix
In this appendix, we report clarifications and additional findings
that was not possible to fit in the main document.
A.1
Mask Propagation Algorithm
Alg. 1 shows a procedural formulation of the mask propagation
and label unification strategy described in Sec. 3.1.3 of the main
document, and is reported here for clarity and reproducibility.
Algorithm 1: Label via Mask–Point Intersection
Input: 𝑃labeled; mask set M𝑘; warped mask 𝑀∗
𝑘,𝑗; bias 𝜆init
Output: label 𝑙for each 𝑀𝑘,𝑗∈M𝑘and updated 𝑃labeled
1 L ←J1 // Initialize the set of global instance indices
2 for 𝑘←2, . . . , 𝐾do
3
for 𝑗∈J𝑘do
// Collect labels overlapping with the virtual mask
4
J𝑘−1∩𝑘←{ 𝑗| 𝑀∗
𝑘,𝑗(𝑥, 𝑦) = 1, 𝑀𝑘,𝑖(𝑥, 𝑦) = 1∀𝑖,𝑥, 𝑦}
5
if J𝑘−1∩𝑘= ∅then
6
𝑙←𝑗
// new instance detected
7
L ←L ∪{𝑗}
8
else
9
𝑙←argmax𝑗∈J𝑘−1
 Í
𝑥,𝑦𝑀∗
𝑘,𝑗· 𝑀𝑘,𝑖

𝑖∈J𝑘
// max overlap
// Update label weights
10
foreach 𝑝∈𝑃𝑗,𝑘do
11
if 𝑤𝑝is uninitialized then
12
𝑤𝑝[𝑖] ←1 + 𝜆init
// first observation
13
else
14
𝑤𝑝[𝑗] ←𝑤𝑝[𝑗] + 1
15
𝑙𝑝←arg max
𝑗∈J 𝑤𝑝[𝑗]
A.2
Dense Semantic Embeddings
In the main document, we assigned a single CLIP-based semantic
descriptor to each object instance. However, some scene recogni-
tion tasks, such as part segmentation and object-level editing, may
require a denser embedding of semantic information embedded
directly in the Gaussians. To achieve this, we propose enriching
object representations by assigning a semantic descriptor 𝑓𝑔𝑖to
each Gaussian 𝑔𝑖∈G𝑙of instance 𝑙. To this end, we extract a new
set of 𝑁𝑠𝑦𝑛𝑡ℎ= 72 (corresponding to 6 different elevations and
12 azimuthal angles) multi-view images I𝑙around instance 𝑙by
sampling camera positions on a semi-spherical trajectory around
the object and rendering it from these viewpoints. The trajectory is
centered at the instance centroid 𝑐𝑎=
1
| G𝑙|
Í
𝑔∈G𝑙, 𝑔∈R3 and has
a radius 𝑟𝑎= 2𝑑𝐵𝑙, where 𝑑𝐵𝑙is the diagonal of its bounding box
𝐵𝑙. This ensures sufficient coverage of the instance while providing
viewpoint diversity.
From these masked views (where the background is blurred), we
compute high-resolution 2D semantic descriptors using a feature
extraction network 𝐸𝐹(·) (e.g., DINOv2), producing descriptor maps
𝑓∈R𝐻×𝑊×𝑛. Each descriptor is then projected back onto the
Gaussian splatting.
For ease of understanding, we align the following mathematical
notation with that of [Kerbl et al. 2023]. For a pixel location q in
Figure 10: PCA on projected descriptors "red apple" (left) and
"ice cream" (right).
view 𝑘, we assign its descriptor to the Gaussian that contributes
most to the alpha blending of the corresponding pixel rasterization:
𝑔𝑖= argmax
𝑖
h
𝛼𝑖· exp

−1
2𝜋⊤
𝑤,ℎΣ−1
𝑖,2𝐷𝜋𝑤,ℎ
i
.
(7)
where 𝜋𝑤,ℎ≔w−Π(ℎ), and Σ−1
𝑖,2𝐷represents the covariance matrix
in the standard Gaussian splatting formulation [Kerbl et al. 2023].
For each Gaussian, the final descriptor 𝑓𝑔𝑖is obtained by averaging
contributions across all the 𝑁𝑣views in which it is visible:
𝑓𝑔𝑖= 1
𝑁𝑣
𝑁𝑣
∑︁
𝑛=1
𝑓𝑛
𝑔𝑖, 𝑁𝑣< 𝑁.
(8)
This produces a final 3D Gaussian reconstruction enriched with
per-Gaussian semantic descriptors. Fig. 10 shows a PCA projection
of the learned object descriptors for two textual queries (“red apple"
and “ice cream"). The learned features capture meaningful semantic
information, showing differences between object parts in the two
instances.
A.3
Additional Results
Tab. 7 extends Tab. 2 of the main document by reporting open-
vocabulary segmentation results across the four main scenes of
LERF, comparing Split&Splat to other methods. Although our
method is not specifically designed for open-vocabulary segmen-
tation, Split&Splat achieves competitive performance, ranking
second in average mIoU (55.68) among all compared approaches
and being outperformed only by the recent VALA method, while
significantly outperforming strong baselines such as InstanceGS,
SuperGSeg, and OpenGaussian. Performance varies across scenes:
the most challenging case is Waldo Kitchen, where the number
of object instances (155) is substantially higher than in the other
scenes (on average ≃25). This highlights a current limitation of our
approach when handling scenes with an extremely large number of
instances. Finally, note that the reported scores underestimate the
actual visual quality of our results. Split&Splat performs object-
level segmentation, whereas the ground truth often splits a single
object into multiple parts. As a result, a visually correct object
mask may be penalized by the evaluation protocol, leading to lower
quantitative scores.
Tab. 8 reports per-instance IoU values on the ScanNetv2 scene
0062_00. The results show a large variability across instances, with
IoU values ranging from 26.9 to 97.8, and a mean IoU of 63.5. This

<!-- page 11 -->
Split&Splat: Zero-Shot Panoptic Segmentation via Explicit Instance Modeling and 3D Gaussian Splatting
Table 7: Open vocabulary results on Lerf dataset [Kerr et al. 2023], competitors data taken from the work of [Wang et al. 2025a].
Best in bold, second best underlined, third dashed underline.
Method
Average
Figurines
Ramen
Teatime
Waldo Kitchen
mIoU
mAcc
mIoU
mAcc
mIoU
mAcc
mIoU
mAcc
mIoU
mAcc
LERF [Kerr et al. 2023]
10.35
13.64
7.27
10.71
10.05
9.86
14.38
20.34
9.71
9.09
LEGaussian [Shi et al. 2024]
16.21
23.82
17.99
23.21
15.79
26.76
19.27
27.12
11.78
18.18
OpenGaussian [Wu et al. 2024]
38.36
51.43
39.29
55.36
31.01
42.25
60.44
76.27
22.70
31.82
SuperGSeg [Liang et al. 2024]
35.94
52.02
43.68
60.71
18.07
23.94
55.31
77.97
26.71
45.45
Dr.Splat [Jun-Seong et al. 2025]
43.29
64.30
54.42
80.36
24.33
35.21
57.35
77.97
37.05
63.64
InstanceGS [Li et al. 2025a]
43.87
61.09
54.87
73.21
25.03
38.03
54.13
69.49
41.47
63.64
CAGS [Sun et al. 2026]
50.79
69.62
60.85
82.14
36.29
46.48
68.40
86.44
37.62
63.64
VoteSplat [Jiang et al. 2025]
50.10
67.38
68.62
85.71
39.24
61.97
66.71
88.14
25.84
33.68
Occam’s LGS [Cheng et al. 2024]
47.22
74.84
52.90
78.57
32.01
54.92
61.02
93.22
42.95
72.72
VALA [Wang et al. 2025a]
58.02
82.85
60.38
89.29
45.41
67.61
70.61
88.14
55.71
86.36
Split&Splat
55.68
73.05
61.80
78.22
58.89
75.95
59.43
75.04
42.58
62.98
Table 8: Per-instance IoU on scene 0062_00.
Instance ID
0
1
2
3
Name
wall
wall
wall
wall
IoU%
40.4
53.3
74.1
56.7
Instance ID
4
5
6
7
Name
wall
mirror
counter
trash can
IoU%
26.9
83.4
81.3
93.6
Instance ID
8
9
10
11
Name
trash can
toilet seat
cover dispenser
paper towel
dispenser
floor
IoU%
75.4
88.2
97.8
90.0
Instance ID
12
13
14
15
Name
toilet
toilet paper
toilet paper
light switch
IoU%
77.9
41.9
33.8
47.4
Instance ID
16
17
18
19
Name
jacket
rail
rail
soap dispenser
IoU%
81.5
57.2
37.3
67.0
Instance ID
20
21
22
23
Name
door
sink
toilet paper
doorframe
IoU%
52.3
72.1
61.9
31.6
mean IoU%
63.5
spread reflects the varying difficulty of different objects, especially
in the presence of clutter, occlusions, and fine-grained structures.
High scores (e.g., instances 7, 10, and 11) correspond to large and
well-isolated objects, while lower values are typically associated
with small or heavily occluded instances.

<!-- page 12 -->
Monchieri et al.
References
Florian Barthel, Arian Beckmann, Wieland Morgenstern, Anna Hilsmann, and Peter
Eisert. 2024. Gaussian Splatting Decoder for 3D-aware Generative Adversarial
Networks. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition
Workshops (CVPRW). IEEE, 7963–7972. doi:10.1109/cvprw63382.2024.00794
Rohan Chacko, Nicolai Hani, Eldar Khaliullin, Lin Sun, and Douglas Lee. 2025. Lifting
by Gaussians: A Simple, Fast and Flexible Method for 3D Instance Segmentation . In
2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE
Computer Society, Los Alamitos, CA, USA, 3497–3507. doi:10.1109/WACV61041.
2025.00345
Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, and Joon-Young Lee.
2023. Tracking Anything with Decoupled Video Segmentation. In Proceedings of
the International Conference on Computer Vision (ICCV). IEEE, Paris Convention
Center, Paris, France.
Jiahuan Cheng, Jan-Nico Zaech, Luc Van Gool, and Danda Pani Paudel. 2024. Oc-
cam’s LGS: A Simple Approach for Language Gaussian Splatting. arXiv preprint
arXiv:2412.01807 (2024).
Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and
Matthias Nießner. 2017. ScanNet: Richly-annotated 3D Reconstructions of Indoor
Scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR). IEEE, Honolulu, Hawaii, USA.
Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 1996. A density-
based algorithm for discovering clusters in large spatial databases with noise. In
Conference on Knowledge Discovery and Data Mining. AAAI Press, Portland, Oregon,
USA., 226–231.
Haoyu Guo, He Zhu, Sida Peng, Haotong Lin, Yunzhi Yan, Tao Xie, Wenguan Wang,
Xiaowei Zhou, and Hujun Bao. 2025. Multi-view Reconstruction via SfM-guided
Monocular Depth Estimation. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR). IEEE, Music City Center, Nashville,
Tennessee, USA.
Minchao Jiang, Shunyu Jia, Jiaming Gu, Xiaoyuan Lu, Guangming Zhu, Anqi Dong,
and Liang Zhang. 2025. VoteSplat: Hough Voting Gaussian Splatting for 3D Scene
Understanding. arXiv preprint arXiv:2506.22799 (2025).
Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank Wang, Jaesung Choe, and
Tae-Hyun Oh. 2025. Dr. splat: Directly referring 3d gaussian splatting via direct
language embedding registration. In Proceedings of the Computer Vision and Pattern
Recognition Conference. IEEE, Music City Center, Nashville, TN, USA, 14137–14146.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.
3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions
on Graphics 42, 4 (July 2023).
https://repo-sam.inria.fr/fungraph/3d-gaussian-
splatting/
Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik.
2023. LERF: Language Embedded Radiance Fields. In Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV). IEEE, Paris Convention Center,
Paris, France.
Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao Zhang, Ronggang Wang,
and Jian Zhang. 2025a. Instancegaussian: Appearance-semantic joint gaussian
representation for 3d instance-level perception. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Music City
Center, Nashville, TN, USA, 14078–14088.
Wanhua Li, Yujie Zhao, Minghan Qin, Yang Liu, Yuanhao Cai, Chuang Gan, and
Hanspeter Pfister. 2025b. LangSplatV2: High-dimensional 3D Language Gaussian
Splatting with 450+ FPS. Advances in Neural Information Processing Systems (2025).
arXiv:2507.07136 [cs.GR] https://arxiv.org/abs/2507.07136
Siyun Liang, Sen Wang, Kunyi Li, Michael Niemeyer, Stefano Gasperini, Nassir Navab,
and Federico Tombari. 2024. Supergseg: Open-vocabulary 3d segmentation with
structured super-gaussians. arXiv preprint arXiv:2412.10231 (2024).
Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil
Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby,
Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech
Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra,
Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski.
2023. DINOv2: Learning Robust Visual Features without Supervision.
Yuning Peng, Haiping Wang, Yuan Liu, Chenglu Wen, Zhen Dong, and Bisheng Yang.
2024. GAGS: Granularity-Aware 3D Feature Distillation for Gaussian Splatting.
arXiv preprint arXiv:2412.13654 (2024).
Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. 2024.
Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR). IEEE, Seattle Convention Center,
Seattle, Washington, USA, 20051–20060.
Yansong Qu, Shaohui Dai, Xinyang Li, Jianghang Lin, Liujuan Cao, Shengchuan Zhang,
and Rongrong Ji. 2024. GOI: Find 3D Gaussians of Interest with an Optimizable
Open-vocabulary Semantic-space Hyperplane. In Proceedings of the 32nd ACM
International Conference on Multimedia. ACM, Melbourne, Australia., 5328–5337.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini
Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From
Natural Language Supervision. In Proceedings of the 38th International Conference
on Machine Learning (Proceedings of Machine Learning Research, Vol. 139), Marina
Meila and Tong Zhang (Eds.). PMLR, 8748–8763. https://proceedings.mlr.press/
v139/radford21a.html
Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu
Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. 2024. Sam
2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024).
Chiara Schiavo, Elena Camuffo, Leonardo Badia, and Simone Milani. 2025. SAGE:
Semantic-Driven Adaptive Gaussian Splatting in Extended Reality. In EUSIPCO.
IEEE, Isola delle Femmine, Palermo, Italy.
Johannes L Schonberger and Jan-Michael Frahm. 2016. Structure-from-motion re-
visited. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR). IEEE, Las Vegas, Nevada, USA., 4104–4113.
Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. 2024. Language
embedded 3d gaussians for open-vocabulary scene understanding. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE,
Seattle Convention Center, Seattle, Washington, USA, 5333–5343.
Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab,
Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa,
et al. 2025. Dinov3. arXiv preprint arXiv:2508.10104 (2025).
Wei Sun, Yuan Li, and Jianbin Jiao. 2026. CAGS: Open-vocabulary 3D scene under-
standing with context-aware Gaussian splatting. Image and Vision Computing 165
(2026), 105830. doi:10.1016/j.imavis.2025.105830
SuperSplat Community. 2026. SuperSplat Editor. https://superspl.at/editor. [Accessed:
2026-01-21].
Sen Wang, Kunyi Li, Siyun Liang, Elena Alegret, Jing Ma, Nassir Navab, and Stefano
Gasperini. 2025a. Visibility-Aware Language Aggregation for Open-Vocabulary
Segmentation in 3D Gaussian Splatting. arXiv:2509.05515 [cs.CV] https://arxiv.
org/abs/2509.05515
Xihan Wang, Dianyi Yang, Yu Gao, Yufeng Yue, Yi Yang, and Mengyin Fu. 2025b.
GaussianGraph: 3D Gaussian-based Scene Graph Generation for Open-world Scene
Understanding. arXiv preprint arXiv:2503.04034 (2025).
Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen
Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, and Jian Zhang. 2024. OpenGaus-
sian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding. In
Proceedings of the Advances in Neural Information Processing Systems (NeurIPS). Cur-
ran Associates, Inc., Vancouver Convention Center, Vancouver, CA, 19114–19138.
Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and
Hengshuang Zhao. 2024. Depth Anything V2. arXiv preprint 2406.09414 (2024).
Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. 2024. Gaussian grouping:
Segment and edit anything in 3d scenes. In Proceedings of the European Conference
on Computer Vision (ECCV). Springer, Milan, Italy., 162–179.
