# VG3S: Visual Geometry Grounded Gaussian Splatting for Semantic Occupancy Prediction

Xiaoyang Yanâ, Muleilan Peiâ,â , and Shaojie Shen

<!-- image-->  
Fig. 1. Comparison between existing Gaussian-based methods and our proposed VG3S. Existing approaches often produce semantic occupancy with incomplete object coverage due to the lack of accurate 3D geometric priors. In contrast, our VG3S incorporates rich 3D geometric priors embedded in a frozen VFM pre-trained on massive datasets, enabling the decoder to generate more geometrically accurate and consistent semantic occupancy predictions.

Abstractâ 3D semantic occupancy prediction has become a crucial perception task for comprehensive scene understanding in autonomous driving. While recent advances have explored 3D Gaussian splatting for occupancy modeling to substantially reduce computational overhead, the generation of high-quality 3D Gaussians relies heavily on accurate geometric cues, which are often insufficient in purely vision-centric paradigms. To bridge this gap, we advocate for injecting the strong geometric grounding capability from Vision Foundation Models (VFMs) into occupancy prediction. In this regard, we introduce Visual Geometry Grounded Gaussian Splatting (VG3S), a novel framework that empowers Gaussian-based occupancy prediction with cross-view 3D geometric grounding. Specifically, to fully exploit the rich 3D geometric priors from a frozen VFM, we propose a plug-and-play hierarchical geometric feature adapter, which can effectively transform generic VFM tokens via feature aggregation, task-specific alignment, and multi-scale restructuring. Extensive experiments on the nuScenes occupancy benchmark demonstrate that VG3S achieves remarkable improvements of 12.6% in IoU and 7.5% in mIoU over the baseline. Furthermore, we show that VG3S generalizes seamlessly across diverse VFMs, consistently enhancing occupancy prediction accuracy and firmly underscoring the immense value of integrating priors derived from powerful, pre-trained geometry-grounded VFMs.

## I. INTRODUCTION

3D semantic occupancy prediction has emerged as a cornerstone perception module for vision-centric autonomous driving systems [1], [2], providing a dense volumetric representation that jointly encodes scene geometry and semantics. Unlike traditional 3D detection tasks [3], [4], [5], occupancy prediction yields a significantly more comprehensive understanding of the driving environment, which is indispensable for enabling safe navigation in complex urban scenarios.

Existing approaches typically model occupancy leveraging discrete voxels [6], [7] or Birdâs-Eye-View (BEV) representations [8], [9], where multi-view image features are encoded and lifted into 3D space for dense occupancy forecasting. Recently, Gaussian-based scene formulations have become a promising alternative [10], [11]. By explicitly modeling a scene as a set of 3D Gaussian primitives [12] and rendering them via Gaussian-to-voxel splatting, these methods provide more interpretable scene representations while naturally exploiting spatial sparsity. This design avoids dense prediction over large empty regions, significantly improving memory efficiency while preserving fine-grained geometric details.

Despite these advancements, most models rely on feature extractors trained solely with limited occupancy supervision, which prevents them from learning strong 3D geometric priors and explicit cross-view constraints. As a result, these approaches often struggle to maintain structural consistency across views, leading to fragmented object geometries, such as incomplete drivable surface planes and man-made structures, as demonstrated in the top of Fig. 1. Consequently, the performance is heavily bottlenecked by the geometric understanding derived from sparse 3D supervision. Moreover, jointly optimizing image encoders with complex 2D-to-3D lifting modules often results in excessive training cycles and the potential loss of detailed 3D geometric information.

Meanwhile, recent geometry-grounded Visual Foundation Models (VFMs), such as VGGT [13], have exhibited strong cross-view geometric grounding capabilities through largescale pre-training. As shown in the middle of Fig. 1, these models are trained on multi-view images with diverse 3D supervision signals under multi-task objectives. Such a training paradigm encourages the VFM to learn intrinsic correlations and viewpoint constraints among different geometric properties, thereby acquiring highly robust geometric consistency.

Unlike conventional occupancy frameworks that must infer 3D structure from limited supervision, these VFMs encode transferable geometric knowledge in their intermediate representations, including relative depth, structural boundaries, and multi-view correspondence. Such grounding priors offer valuable geometric information that can significantly benefit occupancy prediction, particularly for generalization to unseen environments. However, fully fine-tuning a VFM backbone is extremely resource-intensive, introduces a massive computational burden, and may risk catastrophic forgetting of the universal 3D geometric priors.

In light of the powerful geometric knowledge embedded in VFMs, we explore how to leverage these rich 3D cues from a frozen VFM to empower the Gaussian-based occupancy prediction framework. Nevertheless, directly applying raw features from training-free VFMs for occupancy prediction remains challenging. To bridge this gap, we introduce a plugand-play Hierarchical Geometric Feature Adapter (HGFA) to transform generic VFM embeddings into occupancy-specific visual features, as illustrated at the bottom of Fig. 1. Specifically, we utilize a grouped adaptive token fusion strategy to aggregate VFM tokens into a compact representation across hierarchical layers, suppressing redundant geometric activations. We then perform task-aligned token refinement to filter out task-irrelevant noise and calibrate the extracted geometric priors. Finally, we construct a latent spatial feature pyramid to enhance spatial modeling and enforce local geometric coherence in the feature space, producing structurally consistent multi-scale representations tailored for Gaussian-based decoding. Through this hierarchical adaptation pipeline, the HGFA effectively injects geometry-grounded VFM knowledge into the occupancy prediction pipeline, enabling the Gaussian-based decoder to generate more accurate, coherent, and structurally consistent semantic occupancy predictions.

In summary, our main contributions are as follows:

â¢ We propose Visual Geometry Grounded Gaussian Splatting (VG3S), a flexible framework that empowers semantic occupancy prediction with superior cross-view geometric grounding derived from pre-trained VFMs.

â¢ We design a plug-and-play hierarchical geometric feature adapter that unleashes rich 3D geometric priors of frozen VFMs by injecting the generic VFM embeddings into visual tokens tailored for Gaussian-based decoding.

â¢ Extensive experiments on the nuScenes dataset validate that our VG3S significantly improves occupancy prediction accuracy compared to the baseline and generalizes seamlessly across diverse geometry-grounded VFMs.

## II. RELATED WORK

## A. Scene-Centric Semantic Occupancy Prediction

Semantic occupancy prediction has been receiving increasing attention as a scene-centric alternative to object-centric 3D detection, aiming to model the entire 3D environment with dense semantic labels rather than sparse instance-level bounding boxes. Most existing methods follow a 2D-to-3D lifting paradigm, projecting multi-view image features into a unified 3D representation for occupancy prediction. Voxel-based approaches, such as OccFormer [14] and SurroundOcc [15], construct volumetric grids from camera features to predict semantic occupancy. To reduce the high computational cost of dense voxelization, BEVFormer [8] adopts a BEV representation, while TPVFormer [16] further extends this idea by employing a tri-plane formulation to capture complementary spatial contexts. Despite their effectiveness, these grid-based formulations require exhaustive prediction across all cells in the discretized space, including large volumes of empty regions, resulting in substantial redundancy and limited scalability. In contrast, recent Gaussian-based 3D scene representations provide a more compact and structureaware modeling paradigm by allocating computation only to occupied or relevant regions. This observation motivates our adoption of the Gaussian-based scene representation for semantic occupancy prediction.

## B. Gaussian-Based 3D Scene Modeling

3D Gaussian Splatting (3DGS) [12] has recently advanced 3D scene reconstruction by introducing Gaussian primitives that enable efficient differentiable rendering and explicit spatial modeling. Compared to implicit representations such as Neural Radiance Fields (NeRF) [17], 3DGS offers real-time rendering capability, making it particularly well suited for large-scale outdoor scene understanding, including semantic occupancy prediction in autonomous driving. Building upon this paradigm, GaussianFormer [10] represents driving scenes using a set of sparse semantic 3D Gaussian primitives and performs occupancy prediction via Gaussian-to-voxel splatting. GaussianFormer-2 [11] further enhances representation efficiency through distribution-based initialization and probabilistic Gaussian superposition, effectively reducing redundancy and excessive overlap among Gaussian primitives. Despite these promising efficiency gains, existing Gaussianbased methods still rely heavily on learnable image feature encoders trained with limited 3D supervision. As a result, the learned representations often lack robust geometric awareness, which degrades occupancy prediction performance. To address this limitation, our work incorporates rich geometrygrounded priors from visual foundation models to enhance Gaussian-based semantic occupancy prediction.

## C. Geometry-Grounded Visual Foundation Models

Visual Foundation Models (VFMs) have exhibited strong performance and robust generalization across a broad spectrum of vision tasks. DINO [18] introduces self-distillation for self-supervised representation learning, enabling the discovery of semantic correspondences without human annotations. Subsequent models, including DINOv2 [19] and DINOv3 [20], demonstrate that large-scale pretraining on diverse web-scale datasets yields representations with strong semantic structure and cross-task generalization. Beyond semantic understanding, recent VFMs have further incorporated explicit geometric supervision. VGGT [13] leverages multi-task learning with depth estimation and camera pose prediction to extract geometry-grounded features directly from multi-view images. DVGT [21] and DGGT [22] also validate the effectiveness of such geometry-aware representations in autonomous driving, demonstrating robustness under dynamic and complex scenarios. Collectively, these studies underscore the potential of VFMs to offer consistent and geometry-grounded features across viewpoints. Concurrently, VG3T [23] integrates VGGT into semantic occupancy prediction through full fine-tuning of the VFM backbone, which introduces significant computational overhead and risks undermining the generalization capability of pre-trained geometric priors. To this end, our work is intended to unleash the geometric grounding capability of VFMs in a trainingfree manner, preserving their generalization while effectively enhancing Gaussian-based semantic occupancy prediction.

<!-- image-->  
Fig. 2. Framework overview of VG3S. Our approach leverages a powerful, pre-trained frozen VFM to provide rich 3D geometric priors, empowering the downstream Gaussian-based decoder with cross-view 3D geometric grounding and thereby significantly improving 3D semantic occupancy prediction.

## III. METHODOLOGY

## A. Problem Formulation

The objective of 3D semantic occupancy prediction is to jointly infer the geometric occupancy and semantic category of a scene within a dense volumetric representation. Given a set of S-view images $\mathcal { T } = \{ \mathcal { T } _ { i } \in \mathbb { R } ^ { \hat { H } \times W \times 3 } \} _ { i = 1 } ^ { S }$ , where H and W denote the image height and width, the task is to predict a dense semantic occupancy grid $\boldsymbol { \mathcal { V } } \in \mathcal { C } ^ { X \times Y \times Z } .$ in which each voxel is assigned a semantic label from the category set C. In view of the advantages of 3D Gaussian splatting in scene reconstruction [10], [11], we employ the 3D Gaussian representation for scene modeling. Specifically, the scene is represented by a set of J Gaussian primitives $\mathcal { P } = \{ \mathcal { P } _ { i } \} _ { i = 1 } ^ { J }$ . Each primitive $\mathcal { P } _ { i }$ is parameterized by a 3D position $\mathbf { m } _ { i } \ \in \ \mathbb { R } ^ { 3 }$ and a covariance matrix $\Sigma _ { i }$ which is defined via a scaling vector $\mathbf { s } _ { i } \in \mathbb { R } ^ { 3 }$ and a rotation quaternion $\mathbf { r } _ { i } \in \mathbb { R } ^ { 4 }$ . In addition, each Gaussian is associated with an opacity term $a _ { i } \in [ 0 , 1 ]$ and a semantic logit vector $\mathbf { c } _ { i } \in \mathbb { R } ^ { | \mathcal { C } | }$ . The final semantic occupancy prediction can be obtained via Gaussian-to-voxel splatting.

## B. Framework Overview

The overall framework of VG3S is illustrated in Fig. 2, which empowers semantic occupancy prediction with crossview geometric grounding, thereby improving occupancy prediction performance. We first employ a pre-trained, frozen geometry-grounded VFM to extract visual tokens T from multi-view images I. To effectively incorporate rich 3D geometric priors, we introduce a learnable Hierarchical Geometric Feature Adapter (HGFA) composed of three sequential modules: (i) Grouped Adaptive Token Fusion (GATF) for layer-wise feature aggregation, (ii) Task-Aligned Token Refinement (TATR) for goal-oriented feature calibration, and (iii) Latent Spatial Feature Pyramid (LSFP) for multiscale feature restructuring. Through this hierarchical adaptation process, generic VFM embeddings are progressively transformed into geometry-enhanced tokens F. These visual tokens are then decoded into Gaussian primitives, which are finally rendered into dense semantic occupancy voxels via a Gaussian-to-voxel splatting procedure.

## C. Geometry-Grounded VFM Feature Extraction

Existing occupancy prediction approaches [10], [11], [15] typically rely on task-specific image encoders that are trained jointly with the downstream occupancy objective. However, due to the limited availability of large-scale 3D annotations, such dedicated feature extractors often lack robust crossview geometric grounding capability and fail to generalize beyond the training distribution. Recent advances in VFMs, pre-trained on diverse large-scale 3D datasets, have demonstrated strong 3D scene representation capacity. Motivated by this progress, we aim to inject such solid VFM-derived geometric priors into occupancy prediction to enhance crossview geometric perception.

To preserve the universal geometric grounding capability of VFMs in open-world 3D environments while avoiding the substantial computational overhead of end-to-end VFM training, we employ the frozen VFM to encode surrounding images into visual features equipped with strong 3D geometric grounding consistency. Formally, let E denote the frozen VFM encoder. Given a surround-view image $\mathcal { T } _ { i } .$ , the encoder E first divides it into $h \times w$ patches and encodes them into image tokens $T ^ { \mathrm { i m g } } \in \mathbb { R } ^ { L \times \mathrm { d } }$ using a DINO backbone [19], [20], where $L = h \cdot w$ denotes the sequence length and d represents the embedding dimension of the DINO model:

$$
T ^ { \mathrm { i m g } } = \mathcal { E } ( \mathbb { Z } _ { i } ) .\tag{1}
$$

The image tokens are then augmented with camera tokens $T ^ { \mathrm { c a m } } \in \mathbf { \bar { \mathbb { R } } } ^ { 1 \times \mathrm { d } }$ [24] and register tokens $T ^ { \mathrm { r g t } } \in \mathbb { R } ^ { 4 \times \mathrm { d } }$ [25], yielding the augmented tokens $T \in \mathbb { R } ^ { ( 1 + 4 + L ) \times \mathrm { d } } ;$

$$
T = T ^ { \mathrm { c a m } } \oplus T ^ { \mathrm { r g t } } \oplus T ^ { \mathrm { i m g } } ,\tag{2}
$$

where â denotes concatenation.

The augmented tokens $T$ are subsequently processed via a series of Alternating-Attention (AA) blocks [13]. Let B denote the ordered set of attention layers within each AA block. Herein, B = {frame, global} in VGGT [13] and DGGT [22], while $\boldsymbol { B } =$ {frame, global, temporal} in DVGT [21]. Across N cascaded AA blocks, the initial augmented tokens are sequentially updated through these customized attention operations. Within the j-th block, the intermediate outputs of all |B| attention layers are concatenated to produce the updated tokens, denoted as $\mathrm { T } _ { j } \in \mathbb { R } ^ { ( 1 + 4 + L ) \times \mathcal { D } ^ { \vee } }$ , where ${ \mathcal { D } } ^ { \mathrm { v } } = | { \boldsymbol { B } } | \cdot \mathrm { d } .$ We retain the last L image-relevant tokens as the final visual tokens $\mathcal { T } _ { j } \in \mathbb { R } ^ { L \times D ^ { \vee } }$ . Eventually, collecting the outputs from all blocks yields a set of geometry-grounded visual tokens, $\mathcal { T } = \{ \mathcal { T } _ { j } \} _ { j = 1 } ^ { N }$ which captures multi-level geometric priors of the 3D scene context.

## D. Hierarchical Geometric Feature Adapter

Given the geometry-grounded visual token set T extracted from the frozen VFM, unleashing its rich geometric priors to the downstream Gaussian-based occupancy predictor remains a critical challenge. Direct utilization of these frozen tokens is suboptimal, as they lack the explicit spatial structure and task-specific alignment required for dense 3D forecasting. To bridge this inherent gap, we introduce a learnable Hierarchical Geometric Feature Adapter (HGFA), which effectively aggregates, aligns, and restructures the generic VFM features into geometry-enhanced representations tailored for semantic occupancy prediction.

1) Grouped Adaptive Token Fusion: We first partition the visual token set $\tau$ into K groups $\{ \mathcal { G } _ { k } \in \mathbb { R } ^ { M \times L \times \mathcal { D } ^ { \mathrm { v } } } \} _ { k = 1 } ^ { K } ,$ where each group comprises visual tokens concatenated from $M = N / K$ consecutive layers that exhibit similar semantic granularity, i.e.,

$$
\mathcal { G } _ { k } = \mathcal { T } _ { 1 + ( k - 1 ) \cdot M } \boxplus \mathcal { T } _ { 2 + ( k - 1 ) \cdot M } \boxplus \cdot \cdot \cdot \boxplus \mathcal { T } _ { k \cdot M } .\tag{3}
$$

To effectively synthesize these intra-group tokens, we introduce the Grouped Adaptive Token Fusion (GATF) module, which leverages an adaptive feature fusion network to compute instance-specific combination weights. This design enables the dynamic suppression of redundant geometric responses across layers while selectively preserving the most informative scene features.

Formally, for the k-th group, we derive the layer-wise importance weights $\omega _ { k }$ via:

$$
\begin{array} { r } { \omega _ { k } = \varsigma ( \mathrm { M L P } _ { k } ( \mathcal { G } _ { k } ) ) , } \end{array}\tag{4}
$$

where $\omega _ { k } \in \mathbb { R } ^ { M \times L }$ represents the normalized importance scores across all M layers within the group, $\varsigma ( \cdot )$ denotes the Softmax function applied along the layer dimension, and MLP(Â·) is a multi-layer perceptron block.

The aggregated tokens $\hat { T } _ { k } \in \mathbb { R } ^ { L \times \mathcal { D } ^ { \mathrm { V } } }$ for each group is then obtained through a weighted summation followed by normalization:

$$
\hat { T } _ { k } = \mathrm { L N } _ { k } \left( \sum _ { m = 1 } ^ { M } \omega _ { k , m } \odot \mathcal { G } _ { k , m } \right) ,\tag{5}
$$

where LN(Â·) denotes layer normalization and â represents element-wise multiplication.

2) Task-Aligned Token Refinement: Although the grouped visual tokens are compact, they still remain embedded within the generic latent manifold of the pre-trained VFM, which inevitably contains task-irrelevant activations. To bridge this gap and specialize the representations for the semantic occupancy prediction task, we introduce the Task-Aligned Token Refinement (TATR) module.

Designed as a streamlined residual block, TATR refines the generic VFM features into task-specific visual tokens with minimal computational overhead. Formally, the refined tokens $\tilde { T } _ { k } \in \mathbb { R } ^ { L \times \tilde { D } ^ { \mathrm { v } } }$ can be computed as:

$$
\tilde { T } _ { k } = \hat { T } _ { k } + \Psi _ { k } ^ { \mathrm { F F N } } ( \hat { T } _ { k } ) ,\tag{6}
$$

where $\Psi ^ { \mathrm { F F N } } ( \cdot )$ denotes a Feed-Forward Network (FFN) with GELU activation and dropout.

To balance representational capacity with efficiency, we equip the TATR module with a hierarchical capacity-scaling strategy. Recognizing that different groups contain scene information at varying levels of abstraction, we assign groupdependent hidden dimensions to their respective FFN. For the k-th group, the embedding dimension is expanded to $\rho _ { k } \cdot D ^ { \mathrm { V } }$ , where $\rho _ { k }$ is a group-specific expansion ratio. Larger expansion ratios are allocated to shallow groups to preserve finegrained geometric details, whereas smaller ratios are used for deeper groups to distill compact, high-level semantics.

3) Latent Spatial Feature Pyramid: Having obtained the task-aligned visual tokens $\tilde { T } _ { k }$ , we first restore their spatial structure by reshaping them into grid-shaped features $F _ { k } ^ { \prime } \in$ $\mathbb { R } ^ { h \times w \times \mathcal { D } ^ { \mathrm { v } } }$ , adhering to the original VFM patch layout. To reinforce local geometric coherence and establish robust spatial correspondences in the latent feature space, we introduce the Latent Spatial Feature Pyramid (LSFP) module.

Given $F _ { k } ^ { \prime }$ as input, LSFP first applies depth-wise convolution to capture local spatial context:

$$
F _ { k } ^ { \prime \prime } = \Phi _ { k } ^ { \mathrm { D W } } ( F _ { k } ^ { \prime } ) ,\tag{7}
$$

where $\Phi ^ { \mathrm { D W } } ( \cdot )$ denotes depth-wise convolution.

To further integrate the global contextual information, we incorporate a Squeeze-and-Excitation (SE) mechanism [26], which performs adaptive channel-wise reweighting:

$$
F _ { k } ^ { \prime \prime \prime } = \sigma \big ( \Psi _ { k } ^ { \mathrm { F C } } ( \xi ( F _ { k } ^ { \prime \prime } ) ) \big ) \odot F _ { k } ^ { \prime \prime } ,\tag{8}
$$

where $\xi ( \cdot )$ denotes global average pooling, $\Psi ^ { \mathrm { F C } } ( \cdot )$ represents fully connected layers, and $\sigma ( \cdot )$ is the sigmoid activation.

A point-wise convolution is then utilized to facilitate crosschannel interaction and produce the refined spatial feature $\hat { F } _ { k } \in \mathbb { R } ^ { h \times w \times \mathcal { D } ^ { \vee } }$ with a residual connection:

$$
\begin{array} { r } { \hat { F } _ { k } = F _ { k } ^ { \prime } + \Phi _ { k } ^ { \mathrm { P W } } ( F _ { k } ^ { \prime \prime \prime } ) , } \end{array}\tag{9}
$$

where $\Phi ^ { \mathrm { P W } } ( \cdot )$ denotes point-wise convolution.

Subsequently, to extract multi-scale features, we construct a feature pyramid across token groups, where each group is projected to a distinct expanded hidden dimension $\mathcal { D } _ { k } ^ { \mathcal { H } }$ using pointwise convolution, followed by the injection of explicit spatial priors via a 2D sinusoidal positional embedding [27]:

$$
\begin{array} { r } { \tilde { F } _ { k } = \Phi _ { k } ^ { \mathrm { P W } } ( \hat { F } _ { k } ) + \mathrm { P E } _ { k } , } \end{array}\tag{10}
$$

where $\mathrm { P E } _ { k } \in \mathbb { R } ^ { h \times w \times \mathcal { D } _ { k } ^ { \mathcal { H } } }$ is the 2D positional embedding.

Next, we assign each group a distinct spatial scale factor $\tau _ { k }$ and apply scale-adaptive convolutional resampling to $\tilde { F } _ { k }$ yielding multi-resolution features $\bar { F } _ { k } \in \mathbb { R } ^ { ( \tau _ { k } \cdot h ) \times \mathbf { \bar { ( } } \tau _ { k } \cdot \mathbf { \bar { w } ) } \times \mathcal { D } _ { k } ^ { \mathcal { H } } }$

$$
\begin{array} { r } { \bar { F } _ { k } = \mathcal { R } _ { k } ( \tilde { F } _ { k } ) , } \end{array}\tag{11}
$$

where $\mathcal { R } ( \cdot )$ denotes a scale-specific resampling operator that adjusts the spatial resolution according to the factor $\tau _ { k }$

Each level of the feature pyramid is projected to the target channel dimension D aligned with the Gaussian decoder via point-wise convolution:

$$
F _ { k } = \Phi _ { k } ^ { \mathrm { P W } } \big ( \bar { F } _ { k } \big ) ,\tag{12}
$$

Finally, the grid-shaped spatial features $\{ F _ { k } \} _ { k = 1 } ^ { K }$ are flattened back into token sequences and concatenated together, yielding the ultimate geometry-grounded visual tokens ${ \mathcal { F } } \in$ $\begin{array} { r } { \dot { \mathbb { R } } \big ( L { \cdot } \sum _ { k = 1 } ^ { \overline { { K } } } ( \tau _ { k } ) ^ { 2 } \big ) { \times } \mathcal { D } } \end{array}$ for 3D semantic occupancy prediction.

## E. Gaussian-to-Voxel Splatting

Given the geometry-grounded visual tokens F produced by HGFA, we decode them into a set of semantic 3D Gaussian primitives P. To fully exploit the cross-view geometric features embedded in ${ \mathcal F } ,$ we adopt view-guided deformable attention [28], [29] for reference point sampling. Following the probabilistic Gaussian superposition framework [11], the overall probability of occupancy can be obtained by aggregating the contributions of all Gaussian primitives. Semantic labels are then derived through a normalized expectation over Gaussian-conditioned class prediction. Finally, the geometry and semantics predictions are combined to generate the dense semantic occupancy prediction.

## F. Training Objective

We optimize the network using a weighted combination of the standard cross-entropy loss $\mathcal { L } _ { \mathrm { C E } }$ and the Lovasz-SoftmaxÂ´ loss $\mathcal { L } _ { \mathrm { { L o v } } }$ [30], as commonly adopted in semantic occupancy prediction [10], [16]. The overall training objective $\mathcal { L } _ { \mathrm { t o t a l } }$ is defined as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda \mathcal { L } _ { \mathrm { C E } } + \beta \mathcal { L } _ { \mathrm { L o v } } ,\tag{13}
$$

where Î» and $\beta$ are coefficients for balancing the two losses.

## IV. EXPERIMENTS AND RESULTS

## A. Experimental Setup

1) Dataset: We conduct all experiments on the nuScenes dataset [33], which provides multi-view visual inputs from six synchronized cameras, collectively covering a $3 6 0 ^ { \circ }$ horizontal field of view. The dataset comprises 1,000 urban driving sequences, each approximately 20 seconds long, with annotations provided at 2 Hz. In alignment with established benchmarks [10], [11], we adopt a 700/150/150 sequence split for the training, validation, and testing sets, respectively. For supervision, we utilize dense semantic occupancy labels from SurroundOcc [15] as ground truth. The target 3D space is defined as a $1 0 0 \mathrm { m } \times 1 0 0 \mathrm { m } \times 8 $ m volume centered at the ego vehicle, corresponding to a range of [â50 m, 50 m] along both the X and $\bar { Y }$ axes and $[ - 5 \mathrm { m } , 3 \mathrm { m } ]$ along the Z axis. This volume is uniformly discretized into a $2 0 0 \times 2 0 0 \times 1 6$ voxel grid, resulting in a spatial resolution of 0.5 m per voxel.

2) Metrics: Following established evaluation protocols for semantic occupancy prediction [31], we assess the performance of our proposed approach using two primary metrics based on the Intersection-over-Union (IoU). For the Semantic Scene Completion (SSC) task, which requires jointly predicting both scene geometry and semantic labels, we calculate the mean IoU (mIoU) of SSC across all occupied semantic categories $\mathcal { C } _ { o } \mathrm { : }$

$$
\mathrm { m I o U } = \frac { 1 } { \vert \mathcal { C } _ { o } \vert } \sum _ { i \in \mathcal { C } _ { o } } \frac { \mathrm { T P } _ { i } } { \mathrm { T P } _ { i } + \mathrm { F P } _ { i } + \mathrm { F N } _ { i } } ,\tag{14}
$$

where TP, FP, and FN denote the numbers of true positives, false positives, and false negatives, respectively. In contrast, the Scene Completion (SC) task evaluates geometric reconstruction alone by ignoring semantic distinctions. All nonempty voxels are treated as a single occupied class $^ { O , }$ and the corresponding IoU of SC is computed as:

$$
\mathrm { I o U } = \frac { \mathrm { T P } _ { o } } { \mathrm { T P } _ { o } + \mathrm { F P } _ { o } + \mathrm { F N } _ { o } } .\tag{15}
$$

3) Implementation Details: We integrate the pre-trained geometry-grounded VFM, for example, DVGT [21], into the Gaussian-based occupancy framework [11] for cross-view image feature extraction. The VFM remains entirely frozen throughout training, with its pre-trained weights kept fixed in all experiments. For feature adaptation, the GATF module partitions tokens into $K = 4$ groups, each containing $M = 6$ layers. In TATR, the group-specific expansion ratios are set to $\{ \rho _ { k } \} _ { k = 1 } ^ { 4 } = \{ 4 , 3 , 2 , 1 . 5 \}$ . In LSFP, the expanded hidden dimensions are set to $\{ \mathcal { D } _ { k } ^ { \mathcal { H } } \} _ { k = 1 } ^ { 4 } = \{ 7 6 8 , 5 1 2 , \bar { 3 } 8 4 , 2 5 6 \}$ with corresponding spatial scale factors $\{ \tau _ { k } \} _ { k = 1 } ^ { 4 } = \{ 4 , 2 , 1 , 0 . 5 \}$ Following established Gaussian-based designs [10], [11], the decoder comprises four stacked transformer blocks utilizing J = 25, 600 Gaussian primitives, with the channel dimension D fixed at 128. Training employs a cosine annealing learning rate schedule, preceded by a 500-iteration linear warm-up to a peak learning rate of $2 \times 1 0 ^ { - 4 }$ . To reduce the overfitting issue, we apply standard data augmentations, including image resizing and photometric distortions.

TABLE I  
3D SEMANTIC OCCUPANCY PREDICTION RESULTS ON THE NUSCENES BENCHMARK.
<table><tr><td></td><td></td><td>SC SSC</td><td>er</td><td colspan="7">hhe</td><td>Pespn cone e</td><td>tr</td><td>un</td><td> e</td><td>o e</td><td>Smas</td><td></td><td>hwmuen Rein</td></tr><tr><td>Method</td><td>Venue</td><td>IoU</td><td>mIoU</td><td></td><td></td><td>snq </td><td>eu </td><td>csuos </td><td>wtoooce </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>terrn </td><td></td></tr><tr><td>MonoScene [31]</td><td>CVPR 2022</td><td>| 23.96 7.31</td><td></td><td>4.03</td><td>0.35</td><td>8.00</td><td>8.04</td><td>2.90 0.28</td><td>1.16</td><td>0.67</td><td>4.01</td><td>4.35</td><td>27.72</td><td>5.20</td><td>15.13</td><td>11.29</td><td>9.03</td><td>14.86</td></tr><tr><td>Atlas [32]</td><td>ECCV 2020</td><td>28.66 15.00</td><td></td><td>10.64</td><td>5.68</td><td>19.66</td><td>24.94</td><td>8.90</td><td>8.84</td><td>6.47</td><td>3.28</td><td>10.42</td><td>16.21</td><td>34.86</td><td></td><td>15.46 21.89 20.95</td><td>11.21 20.54</td><td></td></tr><tr><td>BEVFormer [8]</td><td>ECCV 2022</td><td>30.50 16.75</td><td></td><td>14.22</td><td>6.58</td><td>23.46</td><td>28.28</td><td>8.66</td><td>10.77</td><td>6.64</td><td>4.05</td><td>11.20 17.78</td><td>37.28</td><td>18.00</td><td>22.88</td><td>22.17</td><td>13.80</td><td>22.21</td></tr><tr><td>TPVFormer [16]</td><td>CVPR 2023</td><td>11.51</td><td>11.66</td><td>16.14</td><td>7.17</td><td>22.63</td><td>17.13</td><td>8.83</td><td>11.39</td><td>10.46</td><td>8.23</td><td>9.43</td><td>17.02 8.07</td><td>13.64</td><td>13.85</td><td>10.34</td><td>4.90</td><td>7.37</td></tr><tr><td>TPVFormerâ [16]</td><td>CVPR 2023</td><td>30.86</td><td>17.10</td><td>15.96</td><td>5.31</td><td>23.86</td><td>27.32</td><td>9.79</td><td>8.74</td><td>7.09</td><td>5.20</td><td>10.97</td><td>19.22 38.87</td><td>21.25</td><td>24.26</td><td>23.15</td><td>11.73 20.81</td><td></td></tr><tr><td>OccFormer [14]</td><td>ICCV 2023</td><td>31.39</td><td>19.03</td><td>18.65</td><td>10.41</td><td>23.92</td><td>30.29</td><td>10.31</td><td>14.19</td><td>13.59</td><td>10.13</td><td>12.49 20.77</td><td>38.78</td><td>19.79</td><td>24.19</td><td>22.21</td><td>13.48 21.35</td><td></td></tr><tr><td>SurroundOcc [15]</td><td>ICCV 2023</td><td>31.49</td><td>20.30</td><td>20.59</td><td>11.68</td><td>28.06</td><td>30.86</td><td>10.70</td><td>15.14</td><td>14.09</td><td>12.06</td><td>14.38</td><td>22.26 37.29</td><td>23.70</td><td></td><td>24.49 22.77</td><td>14.8921.86</td><td></td></tr><tr><td>GaussianFormer [10]</td><td>ECCV 2024</td><td>29.83</td><td>19.10</td><td>19.52</td><td>11.26</td><td>26.11</td><td>29.78</td><td>10.47</td><td>13.83</td><td>12.58 8.67</td><td></td><td></td><td></td><td></td><td></td><td>12.74 21.57 39.63 23.28 24.46 22.99</td><td>9.59 19.12</td><td></td></tr><tr><td>GaussianFormer-2*[11]</td><td>CVPR 2025</td><td>30.5620.02</td><td></td><td>20.15</td><td>12.99</td><td>27.61</td><td>30.23</td><td>11.19</td><td>15.31</td><td>12.64</td><td>9.63</td><td></td><td></td><td></td><td></td><td>13.31 22.26 39.68 23.47 25.62 23.20</td><td>12.25 20.73</td><td></td></tr><tr><td>VG3S</td><td>Ours</td><td>| 34.4121.52</td><td></td><td>|20.78 12.40 28.09 29.53 11.93 15.59 12.38 10.61 14.65 21.74 42.42 26.39 28.06 26.58 17.46 25.76</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

â  denotes supervision using dense occupancy annotations [15].  
\* indicates results obtained with a 128-channel feature dimension for fair comparison [11].

## B. Quantitative Results

1) Comparison with State-of-the-Art: The results on the nuScenes occupancy prediction benchmark are summarized in Table I, with the best and second-best scores indicated in bold and underlined, respectively. Our proposed VG3S significantly outperforms the baseline, GaussianFormer-2 [11], achieving gains of 12.6% in IoU and 7.5% in mIoU. Furthermore, VG3S surpasses both prior voxel-based and existing Gaussian-based paradigms, demonstrating superior accuracy across most semantic classes. Notably, it yields compelling enhancements in structural categories, such as drivable surfaces, man-made objects, and vegetation. These findings not only validate the effectiveness of our proposed VG3S but also underscore that the rich 3D geometric priors derived from the frozen geometry-grounded VFM (DVGT [21]) fundamentally enhance semantic occupancy prediction.

2) Generalization Across Foundation Models: We further investigate the generalizability and compatibility of our proposed framework by integrating it with multiple pre-trained VFMs. To ensure a comprehensive evaluation, we consider a diverse set of models with different design philosophies. Specifically, we include VGGT [13], which introduces multiple geometry-oriented training objectives built upon the DINOv2 backbone [19], and DGGT [22], which is fine-tuned on autonomous driving datasets. In addition, we incorporate the recent foundation model DINOv3 [20] and its geometryaware variant DVGT [21], which is further adapted to driving scenarios. As presented in Table II, integrating any of these VFMs into our pipeline consistently yields a substantial performance gain over the baseline (GaussianFormer-2), demonstrating the broad applicability of our design. Comparing VG3S equipped with DINOv2 (VG3S-DINOv2) against the geometric variants (VG3S-DGGT and VG3S-VGGT) reveals that although a stronger image backbone provides marginal benefits, the primary performance gains stem directly from the injected 3D geometric priors. Moreover, VG3S-DINOv3 achieves highly competitive semantic scene completion performance (21.36% in mIoU), attributed to its robust, universal semantic representation capabilities. Ultimately, VG3S-DVGT, which synergizes cross-view 3D geometric grounding with domain-specific autonomous driving knowledge, pushes the performance boundaries furthest, securing state-of-the-art scores of 34.41% in IoU and 21.52% in mIoU. These results confirm that our design is highly generalizable across diverse VFMs and exceptionally effective at exploiting their rich 3D geometric priors for 3D semantic occupancy predictions.

TABLE II  
EVALUATION OF GENERALIZATION ACROSS DIVERSE VFMS.
<table><tr><td>Method</td><td>| IoU</td><td>mIoU</td></tr><tr><td>GaussianFormer-2 [11]</td><td>30.56</td><td>20.02</td></tr><tr><td>VG3S-DINOv2 [19] VG3S-VGGT [13]</td><td>32.34 33.29</td><td>20.42 21.10</td></tr><tr><td>VG3S-DGGT [22]</td><td>33.37</td><td>20.81</td></tr><tr><td>VG3S-DINOv3 [20]</td><td>33.20</td><td>21.36</td></tr><tr><td>VG3S-DVGT [21]</td><td>34.41</td><td>21.52</td></tr></table>

## C. Ablation Studies

We conduct ablation studies on the nuScenes validation set to evaluate the effectiveness of the proposed HGFA module. All experiments adopt VGGT [13] as the pre-trained VFM, forming the baseline configuration denoted as VG3S-base.

1) Effect of the HGFA: To assess the necessity of the proposed feature adapter, we replace the entire HGFA module with a standard DPT layer [27]. As reported in Table III, this substitution causes a severe performance drop, yielding only 30.59% in IoU and 19.31% in mIoU. This substantial degradation demonstrates that a naive integration of frozen VFM tokens is insufficient for the downstream task. In contrast, the HGFA effectively adapts the generic VFM embeddings and unlocks their 3D geometric grounding capability, leading to significantly improved semantic occupancy prediction.

<!-- image-->  
Fig. 3. Qualitative comparison between the baseline GaussianFormer-2 [11] and our proposed VG3S. Our approach produces more geometrically accurate and consistent object structures across four challenging scenes compared to the baseline, demonstrating that leveraging strong 3D geometric priors embedded within VFMs significantly improves 3D semantic occupancy predictions.

TABLE III  
EFFECT OF THE HGFA.
<table><tr><td>Method</td><td>HGFA</td><td>IoU</td><td>mIoU</td></tr><tr><td>VG3S-base</td><td>â</td><td>33.29</td><td>21.10</td></tr><tr><td>w/o HGFA</td><td>Ã</td><td>30.59</td><td>19.31</td></tr></table>

TABLE IV

ABLATION STUDY ON THE COMPONENTS OF THE HGFA.
<table><tr><td>Method</td><td>GATF</td><td>TATR</td><td>LSFP</td><td>IoU</td><td>mIoU</td></tr><tr><td>VG3S-base</td><td>â</td><td></td><td></td><td>33.29</td><td>21.10</td></tr><tr><td>w/o GATF</td><td>Ã</td><td>&gt;&gt;</td><td>&gt;&gt;</td><td>32.52</td><td>20.43</td></tr><tr><td>w/o TATR</td><td></td><td></td><td>â</td><td>32.70</td><td>20.38</td></tr><tr><td>w/o LSFP</td><td>&gt;&gt;</td><td>Ã&gt;</td><td>Ã</td><td>32.57</td><td>20.33</td></tr></table>

2) Effect of Components in the HGFA: Table IV presents ablation results demonstrating the individual contributions of the constituent modules within the HGFA pipeline. We systematically disable or modify each module while keeping the rest of the architecture unchanged. First, removing the GATF module causes a performance drop of 2.3% in IoU. This decline indicates that the GATF is indispensable for effectively fusing intra-group features and generating the highly informative representations required for dense occupancy prediction. Second, disabling the TATR module yields decreases of 1.8% in IoU and 3.4% in mIoU. This degradation highlights the importance of the TATR in calibrating and aligning VFM features with the requirements of the downstream task. Finally, to isolate the impact of the LSFP, we replace it with a naive linear interpolation operation. Without the dedicated LSFP design, performance drops to 32.57% in IoU and 20.33% in mIoU, highlighting that the LSFP effectively enhances spatial correspondence and preserves local geometric consistency through its multi-scale reconstruction mechanism. Collectively, these results confirm that each component of the HGFA contributes positively to the exploitation of the cross-view 3D geometric grounding capability embedded in VFMs, ultimately improving overall semantic occupancy prediction performance.

TABLE V  
EFFECT OF THE GROUPING STRATEGY IN THE GATF.
<table><tr><td># of groups (K)</td><td>IoU</td><td>mIoU</td></tr><tr><td>1</td><td>31.45</td><td>19.52</td></tr><tr><td>3</td><td>32.66</td><td>20.25</td></tr><tr><td>4</td><td>33.29</td><td>21.10</td></tr><tr><td>6</td><td>32.57</td><td>20.55</td></tr></table>

3) Effect of the Grouping Strategy in the GATF: Further, we study the impact of the token grouping strategy within the GATF by varying the number of groups K, including $K \in \{ 1 , 3 , 4 , 6 \}$ , where K = 1 corresponds to no grouping. As shown in Table V, the configuration with K = 4 achieves the best performance. This indicates that partitioning into four groups ensures tokens within each group share similar semantic granularity, allowing for effective aggregation and abstraction. This specific configuration strikes an optimal balance between structural fidelity and information compression, maximizing the utility of the frozen VFM tokens and subsequently boosting occupancy prediction accuracy.

## D. Qualitative Results

We present qualitative comparisons between our proposed VG3S and the baseline GaussianFormer-2 [11] across four representative driving scenes, as presented in Fig. 3. For clearer comparison, planar regions (e.g., drivable surfaces) are highlighted with boxes, while structural objects (e.g., buildings in the manmade category) are marked with circles. In the first two scenes, VG3S generates significantly more continuous and coherent road surfaces compared to GaussianFormer-2, resulting in superior reconstruction of the ground plane. In the third scene, the baseline struggles to accurately predict the parking lot on the left and the building structures on the right. Similarly, in the final intersection scenario, it fails to capture the upper building and yields a highly incomplete structure around the lower intersection. In contrast, VG3S consistently preserves structural integrity, faithfully recovering complete object geometries and dense scene layouts across these challenging environments. These visualizations demonstrate that VG3S effectively enables the solid cross-view geometric grounding from VFMs to substantially elevate occupancy prediction performance. More visualization results are provided in the supplementary video.

## V. CONCLUSION

In this paper, we present VG3S, a generic and flexible framework that advances 3D semantic occupancy prediction by leveraging the powerful cross-view geometric grounding capability of VFMs. To this end, we design a plug-and-play hierarchical geometric feature adapter that injects rich 3D geometric priors from frozen VFMs into the Gaussian-based occupancy prediction pipeline. Specifically, VG3S employs the GATF to aggregate hierarchical VFM embeddings, the TATR to perform task-oriented feature alignment, and the LSFP to enable multi-scale spatial restructuring. Extensive experiments on the nuScenes occupancy prediction benchmark demonstrate that VG3S effectively exploits the solid 3D geometric priors embedded within diverse VFMs, yielding substantial improvements over the baseline and highlighting the immense promise of VFM-derived geometric grounding for advancing 3D semantic scene understanding.

## REFERENCES

[1] M. Pan, J. Liu, R. Zhang, P. Huang, X. Li, H. Xie, B. Wang, L. Liu, and S. Zhang, âRenderocc: Vision-centric 3d occupancy prediction with 2d rendering supervision,â in ICRA, 2024.

[2] M. Pei, J. Shan, P. Li, J. Shi, J. Huo, Y. Gao, and S. Shen, âSept: Standard-definition map enhanced scene perception and topology reasoning for autonomous driving,â IEEE Robotics and Automation Letters, 2025.

[3] Y. Wang, V. C. Guizilini, T. Zhang, Y. Wang, H. Zhao, and J. Solomon, âDetr3d: 3d object detection from multi-view images via 3d-to-2d queries,â in CoRL, 2022.

[4] S. Li, Z. Liu, Z. Shen, and K.-T. Cheng, âStereo neural vernier caliper,â in AAAI, 2022.

[5] S. Li, P. Li, Q. Lian, P. Yun, and X. Chen, âLearning better representations for crowded pedestrians in offboard lidar-camera 3d trackingby-detection,â in ICRA, 2025.

[6] Y. Li, Z. Yu, C. Choy, C. Xiao, J. M. Alvarez, S. Fidler, C. Feng, and A. Anandkumar, âVoxformer: Sparse voxel transformer for camerabased 3d semantic scene completion,â in CVPR, 2023.

[7] G. Riegler, A. O. Ulusoy, and A. Geiger, âOctnet: Learning deep 3d representations at high resolutions,â in CVPR, 2017.

[8] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Y. Qiao, and J. Dai, âBevformer: Learning birdâs-eye-view representation from multi-camera images via spatiotemporal transformers,â in ECCV, 2022.

[9] M. Pei, S. Shi, L. Zhang, P. Li, and S. Shen, âGoirl: Graph-oriented inverse reinforcement learning for multimodal trajectory prediction,â in ICML, 2025.

[10] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, âGaussianformer: Scene as gaussians for vision-based 3d semantic occupancy prediction,â in ECCV, 2024.

[11] Y. Huang, A. Thammatadatrakoon, W. Zheng, Y. Zhang, D. Du, and J. Lu, âGaussianformer-2: Probabilistic gaussian superposition for efficient 3d occupancy prediction,â in CVPR, 2025.

[12] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, 2023.

[13] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, âVggt: Visual geometry grounded transformer,â in CVPR, 2025.

[14] Y. Zhang, Z. Zhu, and D. Du, âOccformer: Dual-path transformer for vision-based 3d semantic occupancy prediction,â in ICCV, 2023.

[15] Y. Wei, L. Zhao, W. Zheng, Z. Zhu, J. Zhou, and J. Lu, âSurroundocc: Multi-camera 3d occupancy prediction for autonomous driving,â in ICCV, 2023.

[16] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, âTri-perspective view for vision-based 3d semantic occupancy prediction,â in CVPR, 2023.

[17] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, 2021.

[18] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski, Â´ and A. Joulin, âEmerging properties in self-supervised vision transformers,â in ICCV, 2021.

[19] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., âDinov2: Learning robust visual features without supervision,â arXiv:2304.07193, 2023.

[20] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, Â´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al., âDinov3,â arXiv:2508.10104, 2025.

[21] S. Zuo, Z. Xie, W. Zheng, S. Xu, F. Li, S. Jiang, L. Chen, Z.- X. Yang, and J. Lu, âDvgt: Driving visual geometry transformer,â arXiv:2512.16919, 2025.

[22] X. Chen, Z. Xiong, Y. Chen, G. Li, N. Wang, H. Luo, L. Chen, H. Sun, B. Wang, G. Chen, et al., âDggt: Feedforward 4d reconstruction of dynamic driving scenes using unposed images,â arXiv:2512.03004, 2025.

[23] J. Kim and S. Lee, âVg3t: Visual geometry grounded gaussian transformer,â arXiv:2512.05988, 2025.

[24] T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski, âVision transformers need registers,â in ICLR, 2024.

[25] J. Wang, N. Karaev, C. Rupprecht, and D. Novotny, âVggsfm: Visual geometry grounded deep structure from motion,â in CVPR, 2024.

[26] J. Hu, L. Shen, and G. Sun, âSqueeze-and-excitation networks,â in CVPR, 2018.

[27] R. Ranftl, A. Bochkovskiy, and V. Koltun, âVision transformers for dense prediction,â in ICCV, 2021.

[28] J. Li, X. He, C. Zhou, X. Cheng, Y. Wen, and D. Zhang, âViewformer: Exploring spatiotemporal modeling for multi-view 3d occupancy perception via view-guided transformers,â in ECCV, 2024.

[29] X. Yan, M. Pei, and S. Shen, âSt-gs: Vision-based 3d semantic occupancy prediction with spatial-temporal gaussian splatting,â in ICRA, 2026.

[30] M. Berman, A. Rannen Triki, and M. B. Blaschko, âThe lovasz- Â´ softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks,â in CVPR, 2018.

[31] A.-Q. Cao and R. de Charette, âMonoscene: Monocular 3d semantic scene completion,â in CVPR, 2022.

[32] Z. Murez, T. van As, J. Bartolozzi, A. Sinha, V. Badrinarayanan, and Rabinovich, âAtlas: End-to-end 3d scene reconstruction from posed images,â in ECCV, 2020.

[33] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, ânuscenes: A multimodal dataset for autonomous driving,â in CVPR, 2020.