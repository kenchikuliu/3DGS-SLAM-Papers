# Chorus: Multi-Teacher Pretraining for Holistic 3D Gaussian Scene Encoding

\*Yue Li1, \*Qi Ma2,3, Runyi Yang3, Mengjiao Ma3, Bin Ren4, Nikola Popovic3 Nicu Sebe4, Theo Gevers1, Luc Van Gool3, â Danda Pani Paudel3, â Martin R. Oswald1 1University of Amsterdam 2ETH Zurich Â¨ 3INSAIT, Sofia University âSt. Kliment Ohridskiâ 4University of Trento

<!-- image-->  
(a) Multi-Teacher Pretraining

<!-- image-->  
(b) Example Feature PCA

<!-- image-->  
Figure 1. Chorus Framework. (a) Multi-Teacher Pretraining. A feed-forward 3DGS scene encoder with per-teacher projectors distills complementary signalsâlanguage-aligned, generalist, and object-awareâinto a shared embedding. (b) Example Feature PCA (results on novel scenes). At inference we input the full 3DGS scene; PCA on encoder features presents clear semantic awareness despite domain shift. (c) Evaluation & Data Efficiency. Chorus attains strong results across scene understanding tasks while using noticeably fewer training scenesâ8.32Ã and 39.9Ã less than the SoTA point clouds pretraining baselinesâhighlighting the efficiency of our pretraining.

## Abstract

While 3DGS has emerged as a high-fidelity scene representation, encoding rich, general-purpose features directly from its primitives remains under-explored. We address this gap by introducing Chorus, a multi-teacher pretraining framework that learns a holistic feed-forward 3D Gaussian Splatting (3DGS) scene encoder by distilling complementary signals from 2D foundation models. Chorus employs a shared 3D encoder and teacher-specific projectors to learn from language-aligned, generalist, and object-aware teachers, encouraging a shared embedding space that captures signals from high-level semantics to fine-grained structure.

We evaluate Chorus on a wide range of tasks: openvocabulary semantic and instance segmentation, linear and decoder probing, as well as data-efficient supervision. Besides 3DGS, we also test Chorus on several benchmarks that only support point clouds by pretraining a variant using only Gaussiansâ centers, colors, estimated normals as inputs. Interestingly, this encoder shows strong transfer and outperforms the point clouds baseline while using 39.9Ã fewer training scenes. Finally, we propose a render-anddistill adaptation that facilitates out-of-domain finetuning. Our code and model will be released upon publication.

## 1. Introduction

The community has made rapid progress on scene representations that enable photorealistic rendering, from neural radiance fields (NeRFs) [32] to real-time 3D Gaussian Splatting (3DGS) [22]. In parallel, there is a growing body of work that attaches semantic cues to these representations (e.g., via attaching visionâlanguage features [23, 31, 35, 39, 67]). Yet comparatively minor attention has been paid to treating the 3D representation itself as a modality from which we can directly mine generalpurpose, transferable features at scale. 3DGS is particularly attractive in this regard: it preserves geometryâappearance primitives and supports fast differentiable rendering, which together make it a promising substrate for large-scale pretraining beyond view synthesis [6, 19, 28, 50].

We address the gap in generalizable 3DGS scene encoding by proposing Chorusâa multi-teacher pretraining framework for training a native 3DGS encoder to align with complementary 2D foundation models. Concretely, Chorus uses a shared 3D encoder over Gaussian primitives and lightweight per-teacher projectors to distill (i) language-aligned semantics from the SigLIP2 encoder [49], (ii) generalist visual features from DINOv3 [47], and (iii) object-aware cues from the Perception Encoder variant PE-Spatial [5, 25], which combines self-alignment with SAMlogit alignment to improve spatial locality while preserving semantics. Our multi-teacher design teaches the scene encoder breadth and complementarity, capturing high-level semantics, instance grouping, and fine spatial structure within a single 3D embedding space.

Chorus builds upon the âlift-then-alignâ paradigm established by SceneSplat [28], which lifts dense 2D language features to 3D Gaussians and uses them as pseudo-labels to train a feed-forward 3DGS encoder for open-vocabulary segmentation. However, SceneSplatâs encoder was predominantly aligned with semantic information and demonstrated on semantic segmentation, leaving broader downstream applications (e.g., instance grouping) and reasoning capabilities largely unexplored. Chorus generalizes this paradigm with multi-teacher pretraining to explicitly supervise with diverse signals in order to learn a versatile 3D feature representation. Our framework therefore results in 3D encoding that reaches superior performance across a diverse set of tasks, thereby producing a holistic 3D scene encoder.

We demonstrate the effectiveness of Chorus on the following tasks: semantic segmentation, open-vocabulary semantics, instance segmentation, open-vocabulary instances, and visual question answering. The evaluations are conducted on a comprehensive collection of datasets: Scan-Net200 [44], ScanNet++ [61], Matterport3D [8], and our newly proposed 3DGS-native benchmark (with per-Gaussian labels) built upon InteriorGS [48]. In contrast, prior methods for generalizable 3D scene understanding typically specialize in a limited subset of tasks such as openvocabulary tasks [27â29], semantics and instances [51, 53, 55], and VQA reasoning [9, 12, 14]. Our evaluation demonstrates that pretrained Chorus encoder can simultaneously serve as the most effective solution across this broad spectrum of scene understanding tasks. Furthermore, we carry out probing experiments (linear/decoder probing and full finetuning) for semantic and instance segmentation to assess the feature quality across the same datasets. In addition, we conduct data-efficiency studies that restrict supervision to limited scenes and sparse annotations, thereby stress-testing how much the pretrained encoder alone carries.

Besides 3DGS, we tested Chorus on several benchmarks that only support point clouds. For this purpose, we pretrained a new point-cloud-compatible 3D encoder using the Gaussiansâ centers, colors, estimated normals as the only inputs, while keeping all other training signals and losses identical. To our surprise, this variant is competitive with the recent self-supervised point cloud pretraining method Sonata [55] while using â¼ 39.9Ã fewer training scenes. Chorus also exhibits favorable scaling as we move from subset to joint-dataset pretraining. These observations indicate that our multi-teacher pretraining successfully mines semantics, spatial locality, and instance grouping that carry over when the encoder is evaluated on point clouds tasks, despite the distribution difference. In practice, multi-teacher distillation over 3DGS is a practical, efficient route towards a general feed-forward scene encoder.

To further demonstrate versatility, Chorus facilitates outof-domain adaptation by introducing a render-and-distill strategy that eliminates the need for costly 3D pseudo-label preprocessing. This approach leverages the inherent rendering capability of 3DGS: given a new dataset, we simply render 2D views, perform online teacher inference, and finetune our encoder with teacher knowledge. This makes the adaptation pipeline more lightweight and accessible.

Our contributions can be summarized as follows:

â¢ A multi-teacher pretraining framework that aligns a native 3DGS encoder with diverse 2D teachers (languagealigned, generalist, and object-aware) via a shared backbone and per-teacher projectors.

â¢ A holistic 3D scene encoding that produces highly structured and transferable embeddings for both 3DGS and PC inputs, leading to state-of-the-art performance on a broad range of tasks, while demonstrating data efficiency.

â¢ A lightweight render-and-distill adaptation recipe that enables convenient out-of-domain finetuning without requiring costly 3D pseudo-label preprocessing.

## 2. Related Work

Self-supervised and Cross-modal Distillation for 3D. Self-supervised learning (SSL) has driven strong representation learning for 2D images [7, 15, 16] and has been actively explored for 3D data via contrastive and masked modeling [2, 34, 43, 56, 62]. Recently, Sonata [55] mitigates the geometric shortcut during point clouds self-supervised learning and [64] shows that joint 2Dâ3D SSL can yield more coherent spatial features than using a single modality. In parallel, knowledge distillation [18] has emerged as a powerful paradigm. Cross-modal distillation injects priors from 2D foundation models [26, 33, 40, 49, 63] into 3D, mitigating label scarcity and enabling semantic awareness in 3D representations [23, 36, 57, 58]. This distillation paradigm has progressed from single-teacher to multiteacher aggregation in 2D domain, as shown by [17, 42, 45], which strengthens learning with complementary signals.

<!-- image-->  
Figure 2. Chorus Overview. (a) Multi-Teacher Pretraining. We train a feed-forward 3DGS scene encoder to distill complementary signalsâlanguage-aligned (SigLIP), generalist (DINO), and object-aware (PE)âfrom 2D teachers. This knowledge is transferred into a shared embedding space via lightweight per-teacher projectors and losses. To accelerate out-of-domain adaptation, we support finetuning the encoder with online rendering-based supervision. (b) Task-Specific Transfer. Pretrained Chorus encoder enables diverse downstream tasks, including semantic and instance segmentation, open-vocabulary query, and 3D visual question answering (VQA).

We adopt this perspective and specialize it to 3DGS for the first time: instead of a single teacher or objective, Chorus distills from various 2D teachers (language-aligned, generalist, object-aware) to align embeddings with rich priors, while leveraging the inherent rendering capability of 3DGS. 3D Gaussian Splats Encoders. Unlike 3D point clouds, where representation learning is well explored [37, 38, 51, 53, 65], encoding 3D Gaussian Splats remains underexplored despite their richer parameter space that couples both appearance and geometry. ShapeSplat [30] pioneers object-level masked reconstruction for 3DGS objects, while Can3Tok [13] learns a scene-level VAE that tokenizes 3DGS scenes into latent codes. At the scene level, Scene-Splat [28] lifts 2D semantic priors to train a generalizable 3DGS encoder for open-vocabulary semantics and introduces the SceneSplat-7K dataset. SceneSplat++ [29] further scales scene-level 3DGS data and establishes a comprehensive benchmark for language-aligned 3DGS methods. Building on this trajectory, Chorus proposes consolidating complementary 2D priors into a single feed-forward 3DGS encoder, producing holistic scene embeddings that transfer robustly across diverse tasks (semantic, instance, and question answering [14]).

## 3. Method

Chorus pretrains a general-purpose feed-forward Gaussian scene encoder by distilling knowledge from multiple 2D teachers. We first explain the pretraining data where the 2D feature maps are lifted to the 3DGS (Â§3.1). Then we present the multi-teacher framework, i.e., a shared 3DGS encoder with lightweight per-teacher projectors and losses, including optional contrastive terms that exploit available semantic class/instance structure (Â§3.2). Next, we describe a rendering-based adaptation recipe that shortcuts adaptation via image-plane supervision, accelerating the out-ofdistribution generalization (Â§3.3). Finally, we introduce our 3DGS-aware augmentations to aid pretraining (Â§3.4).

## 3.1. Lifting 2D Teachers for Supervision

3DGS scene rendering. A 3DGS scene is an optimized parameter set of N Gaussians:

$$
\mathcal { G } = \{ ( \mathbf { x } _ { i } , \mathbf { s } _ { i } , \mathbf { q } _ { i } , \alpha _ { i } , \mathbf { c } _ { i } ) \} _ { i = 1 } ^ { N }\tag{1}
$$

to reproduce multi-view images via alpha composition and anisotropic Gaussians [22]. Each tuple contains parameters for a center $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ , scale $\mathbf { s } _ { i } \in \mathbb { R } _ { + } ^ { 3 }$ , orientation $\mathbf { q } _ { i } \in \mathbb { H }$ (unit quaternion), opacity $\alpha _ { i } \in [ 0 , 1 ]$ , and color $\mathbf { c } _ { i } \in [ 0 , 1 ] ^ { 3 }$ . For a viewpoint p and a pixel $ { \mathbf { u } } \in  { \mathbb { N } } ^ { 2 }$ , 3DGS renders colors as

$$
\mathbf { C } ( \mathbf { u } | p ) = \sum _ { i \in \mathcal { S } _ { d , \mathbf { u } } } \underbrace { T _ { i } \alpha _ { i } ( \mathbf { u } | p ) } _ { w _ { i } ( p , \mathbf { u } ) } \ \mathbf { c } _ { i } , T _ { i } = \prod _ { j < i } \big ( 1 - \alpha _ { j } ( \mathbf { u } | p ) \big ) ,\tag{2}
$$

where $\cal { S } _ { d , { \bf { u } } }$ is the depth-sorted set of splats intersecting the viewing ray.

Normalized uplifting. Let $F _ { d , { \mathbf { u } } }$ be a 2D teacher feature at view p, pixel u, and let $f _ { i }$ be the target feature on Gaussian i. Using the same rendering weights as in (2), we obtain uplifted supervision as a weighted average [31]:

$$
\begin{array} { r } { { f } _ { i } = \displaystyle \sum _ { ( p , \mathbf { u } ) \in S _ { i } } \bar { w } _ { i } ( p , \mathbf { u } ) F _ { p , \mathbf { u } } , \bar { w } _ { i } ( p , \mathbf { u } ) = \frac { w _ { i } ( p , \mathbf { u } ) } { \sum _ { ( p ^ { \prime } , \mathbf { u } ^ { \prime } ) \in S _ { i } } w _ { i } ( p ^ { \prime } , \mathbf { u } ^ { \prime } ) } , } \end{array}\tag{3}
$$

where $S _ { i }$ are all view-pixel pairs contributing to feature $f _ { i } .$ Teacher standardization. We supervise with three 2D teachers: SigLIP2 (language-aligned), DINOv3 (generalist features), and PE-Spatial (object-aware). Because teacher activations differ in scale/variance, we apply PHI-S [41], a PCA rotation followed by isotropic Hadamard scaling to achieve unit average per-channel variance while preserving cross-channel relationships. We denote $\widetilde { F } _ { p , { \bf u } } \ =$ $\mathrm { P H I - S } ( F _ { p , \mathbf { u } } )$ and use $\widetilde { f } _ { i }$ analogously when needed.

## 3.2. Multi-Teacher Pretraining

Architecture. A shared 3DGS encoder $g _ { \theta }$ maps Gaussian parameters to latent per-Gaussian features:

$$
Z = g _ { \boldsymbol { \theta } } ( \mathcal { G } ) \in \mathbb { R } ^ { N \times d _ { z } } .\tag{4}
$$

Each teacher â T = {lang, dino, pe} has a lightweight projector head $h _ { t }$ (2-layer MLP with LayerNorm and GELU) producing predictions $\hat { F } ^ { ( t ) } = h _ { t } ( Z ) \in \mathbb { R } ^ { N \times d _ { t } }$ â¢

Per-teacher losses. For teacher t, we denote the uplifted supervision $\widetilde { F } ^ { \left( t \right) }$ and an optional validity mask $M ^ { ( t ) }$ (derived from feature norms / visibility). Our base matching loss combines cosine and smooth- ${ \bf \nabla } \cdot \ell _ { 1 }$ loss:

$$
\begin{array} { l } { { \mathcal { L } _ { \mathrm { m a t c h } } ^ { ( t ) } = \displaystyle \frac { 1 } { | { \cal M } ^ { ( t ) } | } \sum _ { i \in { \cal M } ^ { ( t ) } } \lambda _ { 1 } \Big ( 1 - \cos \big ( \hat { F } _ { i } ^ { ( t ) } , \widetilde { F } _ { i } ^ { ( t ) } \big ) \Big ) } } \\ { { + \lambda _ { 2 } \mathrm { S m o o t h L 1 } \big ( \hat { F } _ { i } ^ { ( t ) } , \widetilde { F } _ { i } ^ { ( t ) } \big ) , } } \end{array}\tag{5}
$$

for preserving both magnitude and angular alignment. We â2-normalize the inputs before calculating cosine terms.

Teacher-specific contrastive loss (optional). We add compact contrastive regularizers [28] when the source dataset provides semantic/instance labels.

â¢ SigLIP2 teacher (semantic): pool class-wise means $\bar { F } _ { c } ^ { ( \bar { t } ) } { = } \mathrm { m e a n } \{ \hat { F } _ { i } ^ { ( t ) } : i \in \mathcal { G } _ { c } \}$ , split each class into two disjoint halves $A / B ,$ , and apply a bidirectional InfoNCE loss over $\ell _ { 2 } \cdot$ -normalized $\bar { F } _ { c , \{ A , B \} } ^ { ( t ) }$ across classes.

â¢ PE-Spatial teacher (instance): pool instance-wise means $\bar { F } _ { k } ^ { ( t ) } { = } \mathrm { m e a n } \{ \hat { F } _ { i } ^ { ( t ) } : i \in \mathcal { T } _ { k } \}$ and similarly apply InfoNCE. We write the loss term succinctly as $\mathcal { L } _ { \mathrm { c o n } } ^ { ( t ) }$ and put equations in the supplement.

Staged pretraining & total optimization objective. Teachers can start at different epochs. Let $A ( e ) \subseteq \tau$ denote the active set at training epoch e $( e . g .$ ., {lang, dino} from the start, then add pe). The total objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } ( e ) = \sum _ { t \in \cal A ( e ) } \lambda _ { t } \Big ( \mathcal { L } _ { \mathrm { m a t c h } } ^ { ( t ) } + \eta _ { t } \mathcal { L } _ { \mathrm { c o n } } ^ { ( t ) } \Big ) , } \end{array}\tag{6}
$$

with simple per-teacher weight $\lambda _ { t }$ and optional $\eta _ { t }$ . We empirically found that PHI-S standardization simplifies loss balancing, i.e., equal weights of $\lambda _ { t }$ suffice across teachers.

## 3.3. Rendering-Based Adaptation

Given a novel data domain, we can adapt our pretrained encoder without precomputing 3D pseudo-labels by online inference. We sample camera poses $\{ p \}$ , conduct visibility culling on the input Gaussians, then run each 2D teacher on the rendered RGB to obtain feature maps $F _ { p , \mathbf { u } } ^ { ( t ) }$ , and obtain per-Gaussian predictions $\hat { F } _ { i } ^ { ( t ) }$ with the current encoder and projector heads. Using the same compositing weights $w _ { i } ( p , { \mathbf { u } } )$ as in Eq. (2), we render an inference feature map for each teacher t:

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->

<!-- image-->  
(c)

<!-- image-->  
(d)

<!-- image-->  
(f)  
Figure 3. Rendering-Based View Sampling and Pairing: (a) Camera Location Sampling: We use Furthest Point Sampling to select camera positions that achieve broad spatial coverage across the entire navigable scene space. (b) Visibility Culling: For each location, we sample view angles and track the visibility of the 3D Gaussians across frames. (c) View Pairing and Selection: We obtain a minimum 2D bounding box covering all visible Gaussians for a given view. Then candidate pairs of poses are calculated and sorted based on the overlap score. $^ \mathrm { ( d , e , f ) }$ Rendered images corresponding to the colored camera viewpoints.

$$
\begin{array} { r } { \hat { F } _ { p , \mathbf { u } } ^ { ( t ) } = \sum _ { i \in { \cal S } _ { p , \mathbf { u } } } w _ { i } ( p , \mathbf { u } ) \hat { F } _ { i } ^ { ( t ) } . } \end{array}\tag{7}
$$

Adaptation objective. Let â¦ be the set of valid pixels with sufficient transmittance. We reuse the same per-teacher matching loss as in Eq. (5) (cosine + SmoothL1, with the same $\lambda _ { 1 } , \lambda _ { 2 } )$ , now applied to the 2D feature maps over â¦:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { i m g } } ^ { ( t ) } = \frac { 1 } { | \Omega | } \sum _ { ( p , { \bf u } ) \in \Omega } \ell _ { \mathrm { m a t c h } } \mathopen { } \mathclose \bgroup \left( \hat { F } _ { p , { \bf u } } ^ { ( t ) } , \widetilde { F } _ { p , { \bf u } } ^ { ( t ) } \aftergroup \egroup \right) . } \end{array}\tag{8}
$$

This render-and-distill loop adapts Chorus using only rendered frames, accelerating the adaptation to new data.

View sampling and pairing pipeline. To enable adaptation on data without provided poses, we select informative views for 2D supervision in two stages: (1) Informative view selection â sample camera positions that are well distributed in the navigable space yet are neither too close to geometry nor heavily occluded; (2) Contextual view pairing â ensure that selected views share sufficient overlap to promote cross-view feature coherence. As illustrated in Fig. 3, we first sample camera positions proportional to the scene size to ensure coverage, and for each position generate eight candidate horizontal viewing directions. Directions whose center ray is too close to scene contents are discarded. For each valid view, we rasterize the Gaussians and record their visibility, then compute the minimum 2D axis-aligned bounding box enclosing all visible splats; only Gaussians that fall inside this region are kept as input for that training view. Finally, for each camera pose we sort the remaining poses by visibility overlap and form training pairs from high-overlap poses, ensuring sufficient multi-view context. Further details are provided in the supplement.

## 3.4. 3DGS-Aware Augmentations

Why point-cloud augmentations are suboptimal for 3DGS? Point clouds augmentations (dropout, elastic distortions, color/geometry jitter, etc.) are designed for i.i.d. sets of points whose attributes (position, color) are direct observations of 3D geometry/appearance. In contrast, a 3D Gaussian Splatting (3DGS) scene is an optimized parameter space. NaÂ¨Ä±ve point-cloud jitter alters $\alpha _ { i }$ and $T _ { i }$ in ways that are not motivated for splat-based rendering, and empirically we observe consistent performance drops when such jitter is applied to our encoder pretraining.

Design principle. We propose two augmentations that are 3DGS-aware: (i) a Rendering-Equivalent perturbation, targeting for the augmented parameters that render approximately the same images, injects small, covariance-aware position noise primarily into low-opacity splats, and (ii) an Immature-Manifold perturbation to mimic earlier (blurrier) stages of optimization, selectively inflates per-splat covariances. Both augmentations are grounded in the rendering equation Eq. (2) and the observed 3DGS optimization dynamics [21, 24]; equations are provided in the supplement.

## 4. Experiments

We evaluate the pretrained Chorus encoder on diverse tasks. First, using the language-aligned projector, we report openvocabulary semantic and instance segmentation. Next, we show that Chorus serves as an effective scene tokenizer for the language model, enabling question answering (Â§4.1). Later, we pretrain a point-cloud variant of Chorus using only Gaussiansâ center, color, estimated normal as inputs, which surprisingly competes with SoTA point clouds encoder consistently (Â§4.2). We analyze this unexpected robustness to gain understanding during ablation study (Â§4.3).

Implementation details. Our pretraining backbone adapts the 5-stage transformer encoder from Sonata [55] with the bottleneck feature dimension of 512. We employ the teachers of SigLIP2-so400m-p16-512, DINOv3-ViTL16, and PE-Spatial-L14-448. For pretraining data, we leverage the collected 3DGS scenes (center, color, opacity, quaternion, scale) from SceneSplat-7K [28], with our newly processed pseudo-labels for each teacher. We set teacher loss weights $\lambda _ { t } = 1 . 0$ and balance the contrastive terms with $\eta _ { t } = 0 . 0 2$ for $\mathcal { L } _ { \mathrm { c o n } } ^ { ( t ) }$ . We pretrain the standard model (denoted â¾ ) using all 3DGS parameters as input, and a point-cloud variant (denoted â¢ ) which instead uses center, color, estimated normal of the Gaussians while keeping all other settings the same. This variant is used for subsequent probing and finetuning to compare against point cloud encoders. For the rendering-based adaptation, we initialize with the pretrained Chorus encoder and for each batch we select 4 overlapping views. By default, the rendered image resolution is 480Ã640, and the rendered feature resolution is 120 Ã 160. We use bilinear interpolation to upsample the online encoded 2D teacher feature to match the rendered feature map. A learning rate of $ { \mathrm { : 2 \times 1 0 ^ { - 4 } } }$ is employed for the adaptation, which runs for 100 epochs. We refer to the supplement for additional training and experiments details.

<!-- image-->  
Figure 4. Inference Feature PCA Visualization. Features from different encoders on a concert hall. Chorus shows the best semantic consistency (see zoomed-in chairs and stairs in the back).

## 4.1. Main Results

Open-vocabulary semantic segmentation. Tab. 1 reports zero-shot semantic segmentation results on fine-grained ScanNet200, Matterport3D, ScanNet++, and InteriorGS benchmarks. Chorus achieves better zero-shot performance, e.g., compared to the previous SoTA SceneSplat, a 3.2% f-mIoU and 9.0% f-mAcc increase on the ScanNet200 benchmark after joint training, and when evaluated on new data, a 5.6% f-mIoU and 5.1% f-mAcc increase on InteriorGS, while using 8.32Ã less training data compared to the point clouds-based pretraining method [27], highlighting the efficiency of our 3DGS-based pretraining framework.

Open-vocabulary instance segmentation. We report open-vocabulary 3D instance segmentation on ScanNet200 in Tab. 2. The results confirm that our encoderâs strong open-vocabulary semantic understanding translates to the instance level. Following the protocol from Mosaic3D [27], we adopt the instance proposals from Mask3D [46] for all baselines. Chorus achieves state-of-the-art performance among the methods that use 3D inputs only, outperforming prior point clouds-based open-vocabulary SoTA [27]. Notably, Chorus reaches a +7.6 mAP gain in recognizing the 66 tail classes, showing ability to recognize rare instances.

Rendering-based adaptation. As shown in Tab. 7, our lightweight adaptation recipe avoids heavy feature I/O during training and adds at most 0.1 s per view for on-thefly feature rasterizationâeliminating the â¼1 TB storage required to precompute teacher features for 800 training scenes. Its effectiveness is evident in Fig. 6: training on an additional 100 scenes from the InteriorGS dataset yields a +2.7% mIoU gain under linear probing over our standard pretraining, indicating better domain adaptation. We also ablate teacher feature resolution during adaptation (Fig. 5), where even low-resolution 30Ã40 DINOv3 features produce a clear improvement, with further gains at higher resolutions and more adaptation scenes.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Training Source</td><td rowspan="2">#Training Scenes</td><td colspan="2">ScanNet200 (200)</td><td colspan="2">Matterport3D (160)</td><td colspan="2">ScanNet++ (100)</td><td colspan="2">InteriorGS (72)</td></tr><tr><td>f-mIoU</td><td>f-mAcc</td><td>f-mIoU</td><td>f-mAcc</td><td>f-mIoU</td><td>f-mAcc</td><td>f-mIoU</td><td>f-mAcc</td></tr><tr><td>OpenSceneâ  [36]</td><td>SN</td><td>Ã1</td><td>6.4</td><td>12.2</td><td>5.7</td><td>10.7</td><td>8.8</td><td>14.7</td><td></td><td></td></tr><tr><td>PLA [11]</td><td>SN</td><td>â</td><td>1.8</td><td>3.1</td><td></td><td>â</td><td>â</td><td></td><td></td><td></td></tr><tr><td>RegionPLC [60]</td><td>SN</td><td></td><td>9.2</td><td>16.4</td><td>6.2</td><td>13.3</td><td>11.3</td><td>20.1</td><td>â</td><td></td></tr><tr><td>OV3D [20]</td><td>SN</td><td>â</td><td>8.7</td><td>â</td><td>â</td><td></td><td>â</td><td></td><td></td><td>â</td></tr><tr><td>Mosaic3D [27]</td><td>SN</td><td>â</td><td>13.0</td><td>24.5</td><td>8.6</td><td>17.8</td><td>16.2</td><td>27.1</td><td>3.8</td><td>8.2</td></tr><tr><td>*SceneSplat [28]</td><td>SN</td><td></td><td>18.9</td><td>31.7</td><td>10.8</td><td>18.7</td><td>14.7</td><td>24.7</td><td>6.1</td><td>8.5</td></tr><tr><td>*Chorus (ours)</td><td>SN</td><td></td><td>22.4</td><td>45.8</td><td>11.4</td><td>16.4</td><td>16.8</td><td>29.1</td><td>9.0</td><td>14.6</td></tr><tr><td>Mosaic3D [27]</td><td>SN, SN++, MP3D, ARKitS, S3D</td><td>Ã24.3</td><td>15.7</td><td>28.3</td><td>13.1</td><td>27.7</td><td>18.0</td><td>29.0</td><td>9.4</td><td>16.0</td></tr><tr><td>SceneSplat [28]</td><td>SN, SN++, MP3D</td><td>Ã2.92</td><td>21.4</td><td>38.7</td><td>13.8</td><td>31.8</td><td>28.4</td><td>50.0</td><td>10.1</td><td>19.0</td></tr><tr><td>Chorus (ours)</td><td>SN, SN++, MP3D</td><td>Ã2.92</td><td>24.6</td><td>47.7</td><td>18.7</td><td>38.5</td><td>29.6</td><td>53.5</td><td>15.7</td><td>24.1</td></tr></table>

Table 1. Zero-Shot 3D Semantic Segmentation on the Fine-Grained ScanNet++ (100 classes) [61], Matterport3D (160 classes) [8], ScanNet200 (200 classes) [10] and InteriorGS (72 classes) [48] Benchmarks. â¾ denotes 3DGS modality input. Chorus and SceneSplat [28] are the only methods that target 3DGS modality pretraining. We report the foreground mean IoU (f-mIoU) and foreground mean accuracy (f-mAcc) excluding wall, floor, ceiling classes, following [36, 60]. â  denotes the official checkpoint and the baseline results are partly taken from [27]. Dataset abbreviations SN, SN++, ARKitS, MP3D, and S3D are short for ScanNet [10], ScanNet++ [61], ARKitScenes [4], Matterport3D [8] and Structured3D [66]. Chorus achieves noticeably better zero-shot performance, e.g., a 3.2% f-mIoU and 9.0% f-mAcc increase on the ScanNet200, and when evaluated on new data, a 5.6% f-mIoU and 5.1% f-mAcc increase on InteriorGS compared to the previous SoTA SceneSplat, while using 8.32Ã less training data compared to the point clouds-based pretraining method [27].

<table><tr><td>Method</td><td>Inputs</td><td>3D Region Proposal Network</td><td>25</td><td>mAP mAP 50</td><td>mAP head</td><td>mAP tail</td></tr><tr><td>Open3DIS SAI3D</td><td>3D+2D 3D+2D</td><td>Superpoints + ISBNet + Grounded-SAM Superpoints + SAM</td><td>32.8 24.1</td><td>29.4 18.8</td><td>27.8 12.1</td><td>21.8 16.2</td></tr><tr><td>OpenScene-3D</td><td>3D</td><td>Mask3D</td><td>7.2</td><td>6.2</td><td>10.6</td><td>0.7</td></tr><tr><td>RegionPLC</td><td>3D</td><td>Mask3D</td><td>9.7</td><td>8.6</td><td>15.6</td><td>1.7</td></tr><tr><td>OpenIns3D</td><td>3D</td><td>Mask3D</td><td>14.4</td><td>10.3</td><td>16.0</td><td>4.2</td></tr><tr><td>Mosaic3D</td><td>3D</td><td>Mask3D</td><td>17.8</td><td>16.0</td><td>21.8</td><td>5.4</td></tr><tr><td>Chorus (ours)</td><td>3D</td><td>Mask3D</td><td>19.6</td><td>18.0</td><td>18.5</td><td>13.0</td></tr></table>

Table 2. Open-Vocabulary 3D Instance Segmentation on Scan-Net200. Methods are grouped by input types, methods using both 3D+2D inputs requires expensive multi-view images processing, whereas Chorus is feed-forward and shows strength, especially on the 66 tail classes.
<table><tr><td rowspan="2">Methods</td><td colspan="3">ScanQA</td><td colspan="3">Nr3D</td></tr><tr><td>EM1</td><td>M</td><td>R</td><td>Sim</td><td>M</td><td>R</td></tr><tr><td>ScanQA [3]</td><td></td><td>13.1</td><td>33.3</td><td></td><td>â</td><td>â</td></tr><tr><td>3D-VLP [59]</td><td></td><td>13.5</td><td>34.5</td><td></td><td></td><td></td></tr><tr><td>Scene-LLM [12]</td><td></td><td>15.8</td><td></td><td>â</td><td>â</td><td>â</td></tr><tr><td>LL3DA [9]</td><td>14.3</td><td>22.8</td><td>34.7</td><td>48.1</td><td>5.8</td><td>9.9</td></tr><tr><td>GaussianVLM [14] *SceneSplat* [28]</td><td>14.4</td><td>22.9</td><td>34.8</td><td>48.2</td><td>20.8</td><td>19.2</td></tr><tr><td>GaussianVLM *Chorus (ours)</td><td>14.8</td><td>22.5</td><td>37.4</td><td>50.6</td><td>22.5</td><td>28.8</td></tr></table>

Table 3. 3D Scene Question and Answering. Comparison across ScanQA (EM@1/M/R) and Nr3D (Sim/M/R).

<!-- image-->

<!-- image-->  
Figure 5. 2D Adaption Ablation. Performance improves with higher teacher render resolution (left) and more adaptation scenes (right). The left x-axis denotes the 2D teacherâs feature resolution, formatted as (feature size) Ã bilinear upsample factor.

Language model-based question answering. We evaluate Chorus as the 3D encoder within an LLM-based pipeline for visual question answering and grounding (see Tab. 3), where swapping in Chorus yields consistent improvements on both benchmarks. Concretely, we follow GaussianVLM [14] and simply replace its 3D backbone: instead of using multi-level features from [28], we feed only the final Chorus encoder stage into the VLM, keeping all other components and training settings unchanged. We train and evaluate both the original GaussianVLM and Chorus-augmented variant on ScanQA [3] (3D-VQA) and Nr3D [1], on the metrics of EM1 (Top-1 Exact Match), M (METEOR), R (ROUGE), and Sim (Sentence Similarity). Fig. 7 provides qualitative VQA examples. As an additional benefit, leveraging only the last encoder stage of Chorus is lighter and faster, achieving approximately a 0.68Ã training time compared to GaussianVLM with SceneSplat.

## 4.2. Chorus on Point Clouds Tasks

Probing & finetuning of semantic segmentation. We evaluate the feature quality of our pretrained encoder via linear/decoder probing and full finetuning on five benchmarks, reported in Tab. 4. With only a learnable linear layer, Chorus can outperform the strong Sonata baseline across five benchmarks, e.g., achieving mIoU gains on ScanNet200 (36.0 vs. 28.8) and ScanNet++ (48.8 vs. 40.7). When fully finetuned, Chorus sets a new state-of-the-art on 4 out of 5 benchmarks, including ScanNet (79.4 mIoU) and ScanNet++ (50.2 mIoU). The advantage is particularly noticeable on ScanNet200, where Chorus achieves 40.9 mIoU, with a gain of +6.5 mIoU. Furthermore, Chorus consistently achieves relatively smaller gaps between linear probing and full finetuning. These results validate that our pretraining produces separable and semantic-aware features.

<table><tr><td>Probing Exp.</td><td colspan="3">ScanNet Val</td><td colspan="3">ScanNet200 Val</td><td colspan="3">ScanNet++ Val</td><td colspan="3">Matterport3D (160)</td><td colspan="3">InteriorGS</td></tr><tr><td>Methods</td><td>mIoU</td><td>mAcc</td><td>allAcc</td><td>mIoU</td><td>mAcc</td><td>allAcc</td><td>mIoU</td><td>mAcc</td><td>allAcc</td><td>mIoU</td><td>mAcc</td><td>allAcc</td><td>mIoU</td><td>mAcc</td><td>allAcc</td></tr><tr><td>MSC [52] (lin.)</td><td>21.8</td><td>32.2</td><td>65.5</td><td>3.3</td><td>5.5</td><td>57.5</td><td>8.1</td><td>11.9</td><td>64.7</td><td>â</td><td>â</td><td>â</td><td>â</td><td>â</td><td>â</td></tr><tr><td>Sonata [55] (lin.)</td><td>73.7</td><td>84.4</td><td>90.3</td><td>28.8</td><td>38.8</td><td>81.8</td><td>40.7</td><td>55.3</td><td>84.8</td><td>18.5</td><td>25.8</td><td>78.8</td><td>24.3</td><td>35.4</td><td>61.4</td></tr><tr><td>*Chorus (lin.)</td><td>75.2</td><td>84.8</td><td>90.5</td><td>36.0</td><td>47.2</td><td>82.8</td><td>48.8</td><td>63.2</td><td>86.4</td><td>20.0</td><td>25.7</td><td>79.4</td><td>27.0</td><td>37.2</td><td>62.6</td></tr><tr><td>Sonata [55] (dec.)</td><td>77.3</td><td>85.9</td><td>92.0</td><td>30.1</td><td>39.4</td><td>83.0</td><td>46.6</td><td>58.9</td><td>86.8</td><td>19.0</td><td>26.2</td><td>79.4</td><td>27.2</td><td>38.6</td><td>65.3</td></tr><tr><td>*/Chorus (dec.)</td><td>75.0</td><td>83.1</td><td>90.6</td><td>32.5</td><td>43.0</td><td>82.2</td><td>48.4</td><td>62.3</td><td>86.7</td><td>19.6</td><td>26.4</td><td>79.6</td><td>29.3</td><td>41.6</td><td>66.8</td></tr><tr><td>PTv3 (sup) [53]</td><td>77.4</td><td>84.8</td><td>92.0</td><td>34.7</td><td>45.4</td><td>83.5</td><td>48.2</td><td>61.6</td><td>87.0</td><td>17.5</td><td>23.3</td><td>78.9</td><td>31.1</td><td>44.0</td><td>67.4</td></tr><tr><td>MSC (f.t.) [52]</td><td>78.2</td><td>85.3</td><td>92.2</td><td>33.4</td><td>43.7</td><td>83.4</td><td>48.7</td><td>61.9</td><td>87.2</td><td>â</td><td></td><td></td><td>â</td><td>â</td><td>â</td></tr><tr><td>Sonata (f.t.) [55]</td><td>78.6</td><td>86.6</td><td>92.3</td><td>34.4</td><td>44.0</td><td>84.0</td><td>49.9</td><td>60.7</td><td>87.4</td><td>21.3</td><td>27.6</td><td>80.2</td><td>30.7</td><td>41.8</td><td>66.2</td></tr><tr><td>*/Chorus (ours)</td><td>79.4</td><td>87.7</td><td>92.4</td><td>40.9</td><td>52.3</td><td>84.1</td><td>52.9</td><td>66.2</td><td>87.1</td><td>23.6</td><td>31.0</td><td>80.5</td><td>31.8</td><td>43.3</td><td>68.9</td></tr></table>

Table 4. Semantic Segmentation Probing & Finetuning Experiments.
<table><tr><td>Data Efficiency</td><td colspan="4">Limited Training Scenes</td><td colspan="4">Limited Annotations</td></tr><tr><td>Methods</td><td>1%</td><td>5%</td><td>10%</td><td>20%</td><td>20</td><td>50</td><td>100</td><td>200</td></tr><tr><td>Sonata [55] (lin.)</td><td>45.3</td><td>62.3</td><td>68.7</td><td>69.8</td><td>68.8</td><td>70.6</td><td>71.2</td><td>71.5</td></tr><tr><td>Chorus (lin.)</td><td>42.0</td><td>60.3</td><td>69.6</td><td>71.3</td><td>70.1</td><td>72.3</td><td>73.3</td><td>73.7</td></tr><tr><td>Sonata [55] (dec)</td><td>43.8</td><td>63.5</td><td>69.5</td><td>72.7</td><td>69.4</td><td>72.9</td><td>74.9</td><td>76.3</td></tr><tr><td>Chorus (dec.)</td><td>43.1</td><td>61.4</td><td>69.7</td><td>72.1</td><td>70.8</td><td>72.4</td><td>74.1</td><td>75.3</td></tr><tr><td>PTv3 [53] (sup.)</td><td>25.8</td><td>48.9</td><td>61.0</td><td>67.0</td><td>60.1</td><td>67.9</td><td>71.4</td><td>72.7</td></tr><tr><td>PPT [54] (sup.)</td><td>31.1</td><td>52.6</td><td>63.3</td><td>68.2</td><td>62.4</td><td>69.1</td><td>74.3</td><td>75.5</td></tr><tr><td>Sonata [55] (f.t.)</td><td>43.5</td><td>63.3</td><td>71.6</td><td>71.5</td><td>68.6</td><td>72.4</td><td>74.9</td><td>75.9</td></tr><tr><td>Chorus (f.t.)</td><td>43.9</td><td>64.0</td><td>73.9</td><td>75.0</td><td>73.1</td><td>76.1</td><td>77.2</td><td>77.4</td></tr></table>

Table 5. ScanNet Data-Efficient Benchmark.

<table><tr><td>Supervise method Preprocess (h)</td><td></td><td>Uplift (h)</td><td>Storage</td><td>Training Overhead</td></tr><tr><td>Uplifting</td><td>3.4</td><td>2.8</td><td>1080 GB</td><td></td></tr><tr><td>Rendering</td><td>0.2</td><td>0</td><td>8 GB</td><td>Rasterization (&lt;0.1s/view)</td></tr></table>

Table 7. Resource and Time Comparison of Uplifting-Based Supervision and Rendering-Based Adaptation. Trade-offs between two approaches for training supervision on InteriorGS (800 scenes): uplifting based (preprocessing heavy) versus renderingbased adaptation (online computation heavy).

Probing & finetuning of instance segmentation. We extend analysis to instance segmentation in Tab. 6. Linear probing again shows the strength of our features; while Sonata leads on ScanNet, Chorus outperforms it on Scan-Net200 (31.6 mAP25) and ScanNet++ (37.0 mAP25). When fully finetuned, Chorus remains competitive, achieving the best results on ScanNet++ $( 4 2 . 9 \ \mathrm { m A P _ { 2 5 } ) }$ and performing comparably to top supervised methods on ScanNet.

<table><tr><td rowspan="2">Methods</td><td colspan="2">ScanNet Val </td><td colspan="2">ScanNet200 Val ScanNet++ Val</td><td colspan="2"></td></tr><tr><td>mAP25 mAP50 mAP25</td><td></td><td></td><td> $\mathrm { \ m A P 5 0 }$ </td><td>mAP25 mAP50</td><td></td></tr><tr><td>MSC [52] (lin.)</td><td>13.3</td><td>5.3</td><td>2.3</td><td>1.0</td><td>4.8</td><td>2.6</td></tr><tr><td>Sonata [55] (lin.)</td><td>72.6</td><td>53.9</td><td>30.0</td><td>20.9</td><td>33.5</td><td>24.5</td></tr><tr><td>Chorus (lin.)</td><td>66.6</td><td>46.9</td><td>31.6</td><td>21.9</td><td>37.0</td><td>27.9</td></tr><tr><td>Sonata [55] (dec.)</td><td>77.3</td><td>62.1</td><td>36.2</td><td>29.3</td><td>39.4</td><td>33.5</td></tr><tr><td>Chorus (dec.)</td><td>76.9</td><td>60.5</td><td>38.8</td><td>31.8</td><td>41.9</td><td>33.8</td></tr><tr><td>PTv3 [53] (sup.)</td><td>74.6</td><td>57.9</td><td>40.1</td><td>32.3</td><td>41.4</td><td>32.5</td></tr><tr><td>Sonata [55] (f.t.)</td><td>77.6</td><td>63.1</td><td>38.3</td><td>31.5</td><td>41.0</td><td>35.3</td></tr><tr><td>Chorus (f.t.)</td><td>78.4</td><td>63.4</td><td>39.3</td><td>33.7</td><td>42.9</td><td>37.2</td></tr></table>

Table 6. Instance Segmentation Probing and Finetuning.

<!-- image-->  
Figure 6. Scaling Trend Together With Rendering-Based Adaptation. Linear probing performance on InteriorGS vs. number of pretraining scenes. We compare our multi-teacher pretraining and the self-supervised pretraining [55] on 3DGS, Chorus scales faster and to higher accuracy. Our adaptation recipe yields a +2.7% mIoU gain on this new dataset using only 100 scenes.

Data efficiency experiments. We validate the benefit of our pretraining under data-scarce conditions on ScanNet in Tab. 5. The results show our encoderâs pretrained features provide advantages over the Sonata baseline. When fully finetuned, Chorus consistently outperforms Sonata across all limited-scene (1%-20%) and limited-annotation (20-200 points/scene) settings. This demonstrates that our pretraining particularly helps in the low-data regime (e.g., +4.5 mAP with 20 labels).

## 4.3. Ablation and Analysis

Why does Chorus work well on point clouds? We examine the Chorus variant that uses only Gaussiansâ centers, colors, normals as inputs while keeping the multi-teacher objectives unchanged. Despite the distribution gap between point clouds (observations) and 3DGS (optimized parameters), we posit two hypotheses: (i) 3DGS pretraining behaves like a strong augmentation of point clouds, inducing stable, noise-robust features; (ii) multi-teacher pretraining is more data-efficient than the self-supervised scheme, yielding better scaling.

<!-- image-->  
Figure 7. VLM Qualitative Results. We visualize a scene in ScanNet and object grounding (left) and QA results (right).

<!-- image-->  
Figure 8. Design Choice Ablation. We validate the choices by evaluating zero-shot segmentation on ScanNet++ Val using a subset of training scenes. SmoothL1 loss, 3DGS-aware augmentations, introducing PE-Spatial in a separate stage, and an instancelevel contrastive term each provide incremental gains.

To test (i), we perform instance-level inference feature retrieval from original point clouds (PC) to perturbed PC (centers with Gaussian noise). We report R@1âfraction whose top-1 nearest feature is from the same instanceâand Same-class@Incorrect top-1âwhen wrong, how often the prediction is at least the correct semantic class. The results are gathered from 684 instances using a subset of 10 scenes in the ScanNet++ Val split. Chorus variant is better on both (Tab. 8), indicating robustness to input noise. To test (ii), we evaluate InteriorGS linear probing as the number of pretraining scenes grows (Fig. 6). When applying on 3DGS, Chorus pretraining scales faster than the self-supervised scheme used in Sonata. Taken together with Tab. 8, this suggests: (1) 3DGS-based pretraining induces noise-robust embeddings that transfer to point clouds, and (2) multiteacher supervision supplies strong signals that keep improving with scale, contributing to the variantâs strong PC performance despite the distribution gap. PCA analysis in Fig. 4 and supplement provide additional visualization.

Teachers ablation. We ablate the three teachersâSigLIP, DINO, and PE-Spatialâin two complementary views.

<table><tr><td>Method</td><td></td><td>R@1 (PCânoisy PC)â Same-class@Incorrect top-1â</td></tr><tr><td>Sonata</td><td>79.8%</td><td>75.0%</td></tr><tr><td>Chorus variant</td><td>85.4%</td><td>78.0%</td></tr></table>

Table 8. Instance Retrieval From PC to Perturbed PC. Averaged over 684 instances from 10 ScanNet++ Val scenes.
<table><tr><td rowspan="2">Training source</td><td colspan="3">Teachers</td><td colspan="2">Val Split</td><td colspan="2">InteriorGS</td></tr><tr><td>Lang</td><td>NO  </td><td></td><td></td><td></td><td>mAccfg mIoUfg mAccfg</td><td></td></tr><tr><td rowspan="2">ScanNet</td><td>â</td><td>â</td><td>â</td><td>21.2</td><td>42.0</td><td>7.3</td><td>8.8</td></tr><tr><td>â</td><td>â</td><td>â</td><td>22.4</td><td>45.8</td><td>9.0</td><td>14.6</td></tr><tr><td rowspan="3">ScanNet++v2</td><td>v</td><td></td><td></td><td>27.1</td><td>45.3</td><td>8.0</td><td>12.8</td></tr><tr><td>â</td><td>v</td><td>â</td><td>29.4</td><td>55.8</td><td>9.3</td><td>16.0</td></tr><tr><td>â</td><td>v</td><td>â</td><td>29.6</td><td>56.4</td><td>11.4</td><td>17.1</td></tr></table>

Table 9. Teachers Ablation with Zero-Shot Semantic Segmentation. The âTeachersâ columns mark included components (â/â). We report foreground metrics likewise.

In Tab. 9 we fix the language teacher and then add DINO, followed by PE; zero-shot semantic segmentation improves consistently on both the training dataset and InteriorGS as teachers are added, indicating complementary semantics (Lang) and general, object-aware structure (DINO/PE). We further ablate SigLIP teacher in the supplement. Together, there are non-redundant gains from each teacher.

Design choice ablation. Fig. 8 evaluates training choices on ScanNet++ Val using a subset. SmoothL1 loss, 3DGSaware augmentations, and an instance-level contrastive term each yield incremental improvements. A key finding is when to introduce PE-Spatial: staging it only in the second half of training outperforms enabling it from the start, suggesting that early PE may over-anchor to local features and compete with teacher alignment, whereas late applying refines spatial awareness after a stable backbone has formed.

<table><tr><td>Method</td><td>SceneSplat</td><td>Chorus</td><td>Sonata</td><td>Mosaic3D</td></tr><tr><td>#Model Params.</td><td>91.7M</td><td>131.3M</td><td>108.5M</td><td>39.1M</td></tr><tr><td>Inference Time/Scene</td><td>0.65s</td><td>0.70s</td><td>0.49s</td><td>0.25s</td></tr></table>

Table 10. Model Size & Runtime. Averaged on 100 scenes.  
Runtime. Tab. 10 compares the model size and average inference time on 100 InteriorGS test scenes (with 965K Gaussians on average). Chorus has the slowest inference, but the runtime of 0.7s per scene is still practical.

## 5. Conclusion

We introduced Chorus, a multi-teacher pretraining framework that learns general-purpose 3D scene representations directly from 3D Gaussian splats. By aligning a native 3DGS encoder with complementary 2D foundation models, Chorus distills language-aligned, generalist, and spatially local cues into a unified 3D embedding that transfers well across scene understanding tasks. Extensive experiments on 3DGS-native and point clouds benchmarks show state-ofthe-art performance and efficient render-and-distill adaptation to new domain. A remaining limitation is the offline cost of precomputing teacher pseudo-labels and an interesting direction is to move toward a unified point-cloudâ3DGS encoder built on our findings.

## References

[1] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. In European conference on computer vision, pages 422â440. Springer, 2020. 6

[2] Mohamed Afham, Isuru Dissanayake, Dinithi Dissanayake, Amaya Dharmasiri, Kanchana Thilakarathna, and Ranga Rodrigo. Crosspoint: Self-supervised cross-modal contrastive learning for 3d point cloud understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9902â9912, 2022. 2

[3] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19129â 19139, 2022. 6

[4] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897, 2021. 6

[5] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, et al. Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181, 2025. 2

[6] Ang Cao, Sergio Arnaud, Oleksandr Maksymets, Jianing Yang, Ayush Jain, Ada Martin, Vincent-Pierre Berges, Paul McVay, Ruslan Partsey, Aravind Rajeswaran, et al. From thousands to billions: 3d visual language grounding via render-supervised distillation from 2d vlms. In Forty-second International Conference on Machine Learning, 2025. 2

[7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J Â´ egou, Â´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650â9660, 2021. 2

[8] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158, 2017. 2, 6

[9] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26428â26438, 2024. 2, 6

[10] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In

Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828â5839, 2017. 6

[11] Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, and Xiaojuan Qi. Pla: Language-driven openvocabulary 3d scene understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7010â7019, 2023. 6

[12] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-llm: Extending language model for 3d visual reasoning. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2195â2206. IEEE, 2025. 2, 6

[13] Quankai Gao, Iliyan Georgiev, Tuanfeng Y Wang, Krishna Kumar Singh, Ulrich Neumann, and Jae Shin Yoon. Can3tok: Canonical 3d tokenization and latent modeling of scene-level 3d gaussians. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9320â 9331, 2025. 3

[14] Anna-Maria Halacheva, Jan-Nico Zaech, Xi Wang, Danda Pani Paudel, and Luc Van Gool. Gaussianvlm: Scene-centric 3d vision-language models using languagealigned gaussian splats for embodied reasoning and beyond. arXiv preprint arXiv:2507.00886, 2025. 2, 3, 6

[15] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729â9738, 2020. 2

[16] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable Â´ vision learners. In CVPR, pages 16000â16009, 2022. 2

[17] Greg Heinrich, Mike Ranzinger, Hongxu Yin, Yao Lu, Jan Kautz, Andrew Tao, Bryan Catanzaro, and Pavlo Molchanov. Radiov2. 5: Improved baselines for agglomerative vision foundation models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22487â22497, 2025. 2

[18] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 2

[19] Haoyi Jiang, Liu Liu, Tianheng Cheng, Xinjie Wang, Tianwei Lin, Zhizhong Su, Wenyu Liu, and Xinggang Wang. Gausstr: Foundation model-aligned gaussian transformer for self-supervised 3d spatial understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11960â11970, 2025. 2

[20] Li Jiang, Shaoshuai Shi, and Bernt Schiele. Open-vocabulary 3d semantic segmentation with foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21284â21294, 2024. 6

[21] Jaewoo Jung, Jisang Han, Honggyu An, Jiwon Kang, Seonghoon Park, and Seungryong Kim. Relaxing accurate initialization constraint for 3d gaussian splatting. 2024. 5

[22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 3

[23] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19729â19739, 2023. 1, 2

[24] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splatting as markov chain monte carlo. Advances in Neural Information Processing Systems, 37:80965â80986, 2024. 5

[25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015â4026, 2023. 2

[26] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015â4026, 2023. 2

[27] Junha Lee, Chunghyun Park, Jaesung Choe, Yu-Chiang Frank Wang, Jan Kautz, Minsu Cho, and Chris Choy. Mosaic3d: Foundation dataset and model for open-vocabulary 3d segmentation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14089â14101, 2025. 2, 5, 6

[28] Yue Li, Qi Ma, Runyi Yang, Huapeng Li, Mengjiao Ma, Bin Ren, Nikola Popovic, Nicu Sebe, Ender Konukoglu, Theo Gevers, et al. Scenesplat: Gaussian splatting-based scene understanding with vision-language pretraining. arXiv preprint arXiv:2503.18052, 2025. 2, 3, 4, 5, 6

[29] Mengjiao Ma, Qi Ma, Yue Li, Jiahuan Cheng, Runyi Yang, Bin Ren, Nikola Popovic, Mingqiang Wei, Nicu Sebe, Luc Van Gool, et al. Scenesplat++: A large dataset and comprehensive benchmark for language gaussian splatting. In NeurIPS, 2025. 2, 3

[30] Qi Ma, Yue Li, Bin Ren, Nicu Sebe, Ender Konukoglu, Theo Gevers, Luc Van Gool, and Danda Pani Paudel. A large-scale dataset of gaussian splats and their self-supervised pretraining. In 2025 International Conference on 3D Vision (3DV), pages 145â155. IEEE, 2025. 3

[31] Juliette Marrie, Romain MenÂ´ egaux, Michael Arbel, Diane Â´ Larlus, and Julien Mairal. Ludvig: Learning-free uplifting of 2d visual features to gaussian splatting scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7440â7450, 2025. 1, 3

[32] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1

[33] Maxime Oquab, Timothee Darcet, Th Â´ eo Moutakanni, Huy Â´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2

[34] Yatian Pang, Eng Hock Francis Tay, Li Yuan, and Zhenghua Chen. Masked autoencoders for 3d point cloud self-

supervised learning. World Scientific Annual Review of Artificial Intelligence, 1:2440001, 2023. 2

[35] Qucheng Peng, Benjamin Planche, Zhongpai Gao, Meng Zheng, Anwesa Choudhuri, Terrence Chen, Chen Chen, and Ziyan Wu. 3d vision-language gaussian splatting. In The Thirteenth International Conference on Learning Representations, 2025. 1

[36] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 815â824, 2023. 2, 6

[37] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652â660, 2017. 3

[38] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017. 3

[39] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051â20060, 2024. 1

[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748â8763. PmLR, 2021. 2

[41] Mike Ranzinger, Jon Barker, Greg Heinrich, Pavlo Molchanov, Bryan Catanzaro, and Andrew Tao. Phi-s: Distribution balancing for label-free multi-teacher distillation. arXiv preprint arXiv:2410.01680, 2024. 3

[42] Mike Ranzinger, Greg Heinrich, Jan Kautz, and Pavlo Molchanov. Am-radio: Agglomerative vision foundation model reduce all domains into one. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12490â12500, 2024. 2

[43] Bin Ren, Guofeng Mei, Danda Pani Paudel, Weijie Wang, Yawei Li, Mengyuan Liu, Rita Cucchiara, Luc Van Gool, and Nicu Sebe. Bringing masked autoencoders explicit contrastive properties for point cloud self-supervised learning. In ACCV, 2024. 2

[44] David Rozenberszki, Or Litany, and Angela Dai. Languagegrounded indoor 3d semantic segmentation in the wild. In European conference on computer vision, pages 125â141. Springer, 2022. 2

[45] Mert Bulent SarÄ±yÄ±ldÄ±z, Philippe Weinzaepfel, Thomas Lu- Â¨ cas, Pau de Jorge, Diane Larlus, and Yannis Kalantidis. Dune: Distilling a universal encoder from heterogeneous 2d and 3d teachers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 30084â30094, 2025. 2

[46] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3d: Mask trans-

former for 3d semantic instance segmentation. arXiv preprint arXiv:2210.03105, 2022. 5

[47] Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico Â´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Â¨ et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 2

[48] Manycore Tech Inc. SpatialVerse Research Team. Interiorgs: A 3d gaussian splatting dataset of semantically labeled indoor scenes. https://huggingface.co/ datasets/spatialverse/InteriorGS, 2025. 2, 6

[49] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 2

[50] Ziyi Wang, Yanran Zhang, Jie Zhou, and Jiwen Lu. Unipre3d: Unified pre-training of 3d point cloud models with cross-modal gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1319â1329, 2025. 2

[51] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point transformer v2: Grouped vector attention and partition-based pooling, 2022. 2, 3

[52] Xiaoyang Wu, Xin Wen, Xihui Liu, and Hengshuang Zhao. Masked scene contrast: A scalable framework for unsupervised 3d representation learning. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 9415â9424, 2023. 7

[53] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4840â4851, 2024. 2, 3, 7

[54] Xiaoyang Wu, Zhuotao Tian, Xin Wen, Bohao Peng, Xihui Liu, Kaicheng Yu, and Hengshuang Zhao. Towards largescale 3d representation learning with multi-dataset point prompt training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19551â19562, 2024. 7

[55] Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, and Julian Straub. Sonata: Selfsupervised learning of reliable point representations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22193â22204, 2025. 2, 5, 7

[56] Saining Xie, Jiatao Gu, Demi Guo, Charles R Qi, Leonidas Guibas, and Or Litany. Pointcontrast: Unsupervised pretraining for 3d point cloud understanding. In European conference on computer vision, pages 574â591. Springer, 2020. 2

[57] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. In European Conference on Computer Vision, pages 131â147. Springer, 2024. 2

[58] Le Xue, Mingfei Gao, Chen Xing, Roberto MartÂ´Ä±n-MartÂ´Ä±n, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1179â1189, 2023. 2

[59] Dejie Yang, Zhu Xu, Wentao Mo, Qingchao Chen, Siyuan Huang, and Yang Liu. 3d vision and language pretraining with large-scale synthetic data. arXiv preprint arXiv:2407.06084, 2024. 6

[60] Jihan Yang, Runyu Ding, Weipeng Deng, Zhe Wang, and Xiaojuan Qi. Regionplc: Regional point-language contrastive learning for open-world 3d scene understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19823â19832, 2024. 6

[61] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias NieÃner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12â22, 2023. 2, 6

[62] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19313â19322, 2022. 2

[63] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975â11986, 2023. 2

[64] Yujia Zhang, Xiaoyang Wu, Yixing Lao, Chengyao Wang, Zhuotao Tian, Naiyan Wang, and Hengshuang Zhao. Concerto: Joint 2d-3d self-supervised learning emerges spatial representations. arXiv preprint arXiv:2510.23607, 2025. 2

[65] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16259â16268, 2021. 3

[66] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3d: A large photo-realistic dataset for structured 3d modeling. In Computer Visionâ ECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part IX 16, pages 519â535. Springer, 2020. 6

[67] Yuhang Zheng, Xiangyu Chen, Yupeng Zheng, Songen Gu, Runyi Yang, Bu Jin, Pengfei Li, Chengliang Zhong, Zengmao Wang, Lina Liu, et al. Gaussiangrasper: 3d language gaussian splatting for open-vocabulary robotic grasping. arXiv preprint arXiv:2403.09637, 2024. 1