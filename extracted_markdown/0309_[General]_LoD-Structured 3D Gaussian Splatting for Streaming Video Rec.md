# LoD-Structured 3D Gaussian Splatting for Streaming Video Reconstruction

Xinhui Liu, Can Wang, Lei Liu, Zhenghao Chen, Wei Jiang, Wei Wang, Dong Xu, Fellow, IEEE

AbstractâFree-Viewpoint Video (FVV) reconstruction enables photorealistic and interactive 3D scene visualization; however, real-time streaming is often bottlenecked by sparse-view inputs, prohibitive training costs, and bandwidth constraints. While recent 3D Gaussian Splatting (3DGS) has advanced FVV due to its superior rendering speed, Streaming Free-Viewpoint Video (SFVV) introduces additional demands for rapid optimization, high-fidelity reconstruction under sparse constraints, and minimal storage footprints. To bridge this gap, we propose StreamLoD-GS, an LoD-based Gaussian Splatting framework designed specifically for SFVV. Our approach integrates three core innovations: 1) an Anchor- and Octree-based LoD-structured 3DGS with a hierarchical Gaussian dropout technique to ensure efficient and stable optimization while maintaining high-quality rendering; 2) a GMM-based motion partitioning mechanism that separates dynamic and static content, refining dynamic regions while preserving background stability; and 3) a quantized residual refinement framework that significantly reduces storage requirements without compromising visual fidelity. Extensive experiments demonstrate that StreamLoD-GS achieves competitive or state-of-the-art performance in terms of quality, efficiency, and storage.

Index Termsâ3DGS, Sparse-view, Free-Viewpoint Videos, Video Streaming.

## I. INTRODUCTION

ONSTRUCTING Free-Viewpoint Videos (FVVs) from multi-view images promise to revolutionize visual media by bridging the gap between our perception of the dynamic 3D world and the constraints of conventional 2D video [2]. Recently, 3D Gaussian Splatting (3DGS) [3] has emerged as a cornerstone for FVV reconstruction [2], [4]â[9], primarily due to its capacity for photorealistic and real-time synthesis. Due to the increasing demand for real-time interactivity and immersive experiences, streaming Free-Viewpoint Video (SFVV) [10]â[15] has emerged as a focal point of research, offering dynamic, real-time viewing experiences that transcend traditional media boundaries. However, SFVV introduces new challenges and demands for 3DGS, including: 1) Fast optimization and rendering: the ability to rapidly optimization and render novel views in real-time; 2) High fidelity even with sparse views: ensuring the visual quality of rendered scenes, even when only limited camera setups are available, as capturing and transmitting dense views in real-world environments is often logistically and financially prohibitive; 3) low storage requirements for transport (bandwidth): minimizing the storage of 3DGS points, to optimize bandwidth usage.

To address these challenges, 3DGStream [10] pioneered the extension of 3DGS to SFVV by leveraging Instant-NGP [16], which combines neural network representations and multiresolution hashing to accelerate data querying and rendering. Beyond representation, other contemporary approaches [11]â [13], [15] have shifted focus toward motion field modeling, utilizing residual learning [17] to isolate dynamic regions and reduce optimization costs. Despite these advancements, most 3DGS-based frameworks [10]â[13] still depend on dense, synchronized camera arrays to provide sufficient views for high-quality reconstruction. In practice, however, factors such as bandwidth constraints and spatial limitations often restrict systems to only a few cameras, hindering their applicability in real-world scenarios. When attempting SFVV reconstruction under sparse training views, two primary challenges emerge. First, as the number of input views decreases (e.g., 7, 5, 4, or fewer), weakened multi-view consistency causes the model to overfit to observed views, leading to artifacts in novel-view synthesis. Second, sparse inputs often trigger the production of redundant or misplaced Gaussians during the densification process [18], [19], which inflates storage and transmission overhead that are particularly critical for mobile and low-end devices with limited memory and bandwidth.

Drawing inspiration from Level-of-Detail (LoD) hierarchies [20]â[24], which effectively reduce geometric redundancy while enhancing rendering efficiency, we propose StreamLoD-GS, an LoD-based Gaussian Splatting framework tailored for SFVV. Our approach comprises three core innovations: First, we introduce an LoD-structured 3DGS with Anchor and Octree representations. To prevent overfitting and facilitate stable optimization, we introduce a hierarchical Gaussian dropout technique. This strategy dynamically selects detail levels based on viewing distance and employs levelaware stochastic dropout during training. Ultimately, this design mitigates Sparse-View overfitting while preserving the gradient flow essential for robust convergence. Second, leveraging the inherent temporal coherence in video sequences, we develop a Gaussian Mixture Models [25] (GMM)-based motion partitioning mechanism. This module intelligently identifies dynamic anchors for selective refinement while freezing static regions, thereby exploiting background stability and motion continuity to significantly reduce redundant computations. Finally, to enable bandwidth-efficient streaming, we incorporate a quantized residual refinement framework. This design bridges static frozen anchors and dynamically updated regions through quantized residuals, drastically minimizing storage and transmission footprints without sacrificing visual fidelity.

<!-- image-->  
Fig. 1. StreamLoD-GS enables fast, on-the-fly, high-fidelity reconstruction for Streaming Free-Viewpoint Video. Right: Comparison with existing methods with 5 Training Views on Meet Room [1] dataset.

Our major contributions are summarized as follows.

â¢ We propose an Anchor- and Octree-based LoD-structured 3DGS representation, integrated with a hierarchical Gaussian dropout technique to stabilize training and ensure high-fidelity synthesis in Sparse-View SFVV.

â¢ We introduce a GMM-driven mechanism to decouple dynamic and static content, enabling targeted refinement of moving regions while maintaining the structural stability of the background.

â¢ We develop a quantized residual refinement framework that compresses dynamic updates, facilitating efficient data transmission and storage for low-bandwidth environments.

â¢ Extensive experiments demonstrate that StreamLoD-GS achieves competitive or state-of-the-art performance in terms of quality, efficiency, and storage footprints compared to contemporary SFVV approaches.

## II. RELATED WORKS

## A. Streaming 3DGS

Constructing streamable dynamic scenes via on-the-fly, frame-by-frame training presents significant challenges compared to offline learning from complete multi-view videos. Initial efforts adapted NeRFs [16], [26]â[30] to this task. For instance, StreamRF [1] utilized an incremental learning paradigm by modeling per-frame differences, while ReRF [17] modeled residual information between adjacent frames for long-duration scenes. NeRFPlayer [31], on the other hand, decomposed the 4D spatio-temporal space to optimize for model compactness and reconstruction speed. The advent of 3DGS marked a major leap forward, offering superior training efficiency and rendering speeds. Building on this, streaming

3DGS is designed to reconstruct dynamic scenes through onthe-fly, frame-by-frame training, enabling real-time updates and adjustments as the scene evolves [7], [10]â[13], [15], [32]. 3DGStream [10] proposed a Neural Transformation Cache with multi-resolution hashing to advance 3DGS modeling. Other methods have focused on advanced motion modeling, such as HiCoM [11] with its hierarchical coherent motion mechanism and IGS [13], which uses an anchor-driven network to learn motion residuals in a single step for faster training. To enhance model expressiveness and tackle data size, QUEEN [12] proposed a framework based on Gaussian residuals combined with learned quantization and sparsity for compression. To optimize the rate-distortion trade-off for SFVV, 4DGC [15] introduces a rate-aware compression framework with motion-grid-based representation and differentiable quantization. Despite advancements in fidelity and efficiency, a common bottleneck remains: these methods require dense, multi-view images at every frame for 3D reconstruction. This dependence on synchronized multi-camera setups significantly limits their practical applicability. Furthermore, these approaches often necessitate a large number of Gaussian primitives to model the scene, resulting in substantial storage requirements. In contrast, our method not only enables fast optimization and rendering, but also achieves high fidelity with sparse views, while maintaining low storage requirements.

## B. Novel View Synthesis with Sparse-Views

While both NeRFs [26] and 3DGS [3] achieve exceptional rendering quality with dense imagery, their performance degrades significantly with sparse views due to overfitting. This has spurred extensive research into improving their fewshot performance. 1) Sparse-View NeRF [33]â[37]. Before 3DGS, the NeRF community pioneered several solutions to this problem. These strategies include learning from external priors using pre-trained image encoders (pixelNeRF) [33] or [34] designed a semantic consistency loss using CLIP embeddings [38], enforcing geometric consistency through patchbased appearance and depth regularization (RegNeRF [35],

SparseNeRF [36]), and applying frequency regularization to penalize high-frequency noise and improve generalization (FreeNeRF) [37]. 2) Sparse-View 3DGS [18], [19], [39]â[43]. More recently, a variety of innovative solutions have been developed for the 3DGS framework. Depth Regularization: One line of work focuses on resolving depth ambiguity using learnable parameters (Depth-GS) [39] or local depth normalization (DN-Gaussian) [40]. Consistency and Densification: Other methods improve multi-view consistency using optical flow (CoherentGS) [41] or model disagreement (CoR-GS) [42], while some adaptively densify sparse regions using âGaussian unpoolingâ (FSGS) [43]. Stochastic Regularization: Inspired by dropout, a third approach introduces stochasticity by randomly removing Gaussians during training to effectively mitigate overfitting (DropGaussian [18], DropoutGS [19]). Different from the aforementioned methods, our approach addresses the sparse view issue at the representation level and proposes an LoD-structured 3DGS that enables robust reconstruction even with sparse views.

## C. Level-of-Detail (LoD)

In 3D rendering, LoD refers to the adjustment of the level of detail of 3D models based on their distance from the camera [44], [45]. Several recent works [21], [24], [46]â[48] have explored LoD for 3D graphics systems (3DGS). LetsGo [46] introduces LoD as a multi-resolution Gaussian representation for large-scale scene generation. CityGaussian [47] proposes a block-wise LoD, where during rendering, all Gaussians within the same block share the same level of detail, which is determined by the blockâs distance from the camera. Similarly, Hierarchical-GS [48] divides the large scene into distinct chunks and constructs a LoD for each chunk, facilitating hierarchy generation and consolidation for fast rendering. A more scalable approach was recently introduced by Octree-GS [21], which uses a unified octree with an accumulative LoD strategy to eliminate Gaussian redundancy. Unlike these methods, we propose an anchor- and octree-based LoD-structured 3DGS representation with a hierarchical Gaussian drop technique to prevent overfitting. Additionally, our method specifically addresses the critical challenge of Gaussian storage in a streaming manner, which has not been considered by these approaches.

## III. PRELIMINATRIES

LoD-GS [22] achieves Level-of-Detail (LoD) rendering by organizing 3D Gaussians into hierarchical layers. Each layer l contains Gaussians $\mathcal { G } _ { l }$ with position $\pmb { \mu } _ { l } .$ , covariance matrix $\Sigma _ { l }$ (decomposed into scale $\mathbf { \boldsymbol { s } } _ { l }$ and quaternion $\pmb q _ { l } )$ , opacity $\pmb { \alpha } _ { l } .$ , and spherical harmonic coefficients $\mathbf { } c _ { l } ,$ forming a multi-resolution structure:

$$
\mathcal { G } = \{ \mathcal { G } _ { 0 } , \mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { L - 1 } \} , \quad | \mathcal { G } _ { l } | < | \mathcal { G } _ { l + 1 } |\tag{1}
$$

where coarser layers capture global structure, and finer layers encode details. During rendering, the active layer $l ^ { * }$ is selected based on viewing distance d:

$$
l ^ { * } = \operatorname* { m i n } \left( \left\lfloor \log _ { \beta } \frac { d } { d _ { 0 } } \right\rceil , L \right)\tag{2}
$$

with $d _ { 0 }$ as the base distance and $\beta > 1$ controlling the transition rate. Only Gaussians in active layers are rasterized via tilebased rendering [3], reducing computational cost for distant views while maintaining visual quality through distance-aware training and progressive densification.

Scaffold-GS [49] introduces an anchor-based structured 3D Gaussian representation to reduce redundant Gaussians while maintaining high-quality rendering. The method begins by constructing a sparse voxel grid from Structure-from-Motion [50] (SfM)-derived points, with an anchor placed at the center of each voxel. Each anchor point is associated with a learnable feature Ëf and K neighboring neural Gaussians, generated from the anchor using offsets:

$$
\left\{ \pmb { \mu } _ { 0 } , \dots , \pmb { \mu } _ { K - 1 } \right\} = \pmb { \mu } _ { \mathrm { a n c h o r } } + \left\{ \mathbf { O } _ { 0 } , \dots , \mathbf { O } _ { K - 1 } \right\}\tag{3}
$$

where $\pmb { \mu } _ { \mathrm { a n c h o r } }$ is the anchor center, $\mathbf { O } _ { i }$ is a learnable offset. Other Gaussian attributes are directly decoded from the anchor feature Ëf , the relative viewing distance $\Delta _ { c }$ (the distance between the anchor center and the camera position), and the direction $\vec { d } _ { c }$ (the vector from the camera to the anchor) through individual MLPs, denoted as $\operatorname { F } _ { \alpha } , \operatorname { F } _ { c } , \operatorname { F } _ { q } , \operatorname { F } _ { s }$ . For instance, opacities are:

$$
\{ \alpha _ { 0 } , \dots , \alpha _ { K - 1 } \} =  { \mathrm { F } } _ { \alpha } ( \hat { \mathbf { f } } , \Delta _ { c } , \vec { d } _ { c } )\tag{4}
$$

## IV. METHODOLOGY

Given a sequence of videos captured by N synchronized cameras, denoted as $\{ \mathbf { I } _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ for $t = 0 , 1 , \ldots , T$ , our goal is to reconstruct the video scene in a streaming manner. To achieve this, we propose StreamLoD-GS, a hierarchical LoDstructured Gaussian framework, as shown in Fig. 2. We begin by initializing an anchor- and octree-based LoD-structured 3DGS representation using the multi-view images from the first frame $\{ \mathbf { I } _ { t = 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ (Sec. IV-A); Next, we design a GMMbased motion partitioning method to identify dynamic anchors for selective modeling, while freezing static regions to ensure background stability and reduce training costs (Sec. IV-B); Finally, we propose a quantized residual refinement method to quantize the residuals of the dynamic anchors to reduce storage costs (Sec. IV-C).

## A. LoD-Structured 3DGS with Anchor and Octree

Upon receiving the multi-view images at the first frame $\{ \mathbf { I } _ { t = 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ , we first use them to initialize our LoD-structured representation. We then apply a hierarchical Gaussian dropout to progressively drop Gaussian at each layer to prevent overfitting.

Anchor and LoD Initialization. Standard point cloud initialization typically employs COLMAP-based SfM [51]; however, its reliability diminishes under Sparse-View constraints. To circumvent this, we utilize VGGT [52] to estimate the initial point cloud $\mathcal { P }$ from the first frameâs multi-view images $\{ \mathbf { I } _ { t = 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N } .$ , providing a more accurate starting point for subsequent optimization. Crucially, to ensure a fair experimental comparison, we apply this consistent initialization across all evaluated methods in Sec. V. Following the acquisition of its new layer index can be updated by:

<!-- image-->  
Fig. 2. Illustration of our proposed StreamLoD-GS. It comprises three major components: (a) LoD-Structured 3DGS with Anchor and Octree; (b) GMM-Based Motion Partitioning; (c) Quantized Residual Refinement. Bottom: The streaming training pipeline. Frames are processed sequentially: the initial frame (Time 0) undergoes LoD-structured 3DGS optimization, while subsequent frames apply motion partitioning and residual refinement to ensure temporally coherent RGB rendering throughout the stream.

P, we initialize the Gaussian centers and organize them into an octree structure with an L-layered LoD hierarchy, following Octree-GS [53]. Here L is determined by the scene scale bounds $( d _ { \operatorname* { m a x } } , d _ { \operatorname* { m i n } } )$ as $L = \lfloor \log _ { 2 } ( d _ { \mathrm { m a x } } / d _ { \mathrm { m i n } } ) \rceil + 1$ , where â.â denotes the round operator. Instead of directly placing an anchor in each voxel, we perform a layer-wise placement. At each level $l \in \{ 0 , . . . , L - 1 \}$ , an anchor is placed in a voxel of size $\delta _ { l } \ = \ \delta / 2 ^ { l }$ , where Î´ is the base voxel size at the coarsest level (level 0). Then similar to Scaffold-GS [49], each anchor is assigned a feature Ëf and is associated with K neural Gaussian, each having a learnable offset for Gaussian positions.

To further reduce computational costs, we employ a dynamic anchor selection strategy [21] to select the most important anchors during rendering. For an arbitrary anchor $\mathbf { a } _ { j }$ , its original layer index can be computed as $\begin{array} { r } { L _ { j } = \lfloor \log _ { 2 } \left( \frac { d _ { \operatorname* { m a x } } } { \Delta _ { c _ { i } } } \right) \rceil } \end{array}$ â, where $\Delta c _ { j }$ is the distance between the anchorâs center and the camera position. During training, its gradient regarding to the 2D screen-space projection position is recorded as $\mathbf { g } _ { \pmb { \mu } _ { \mathrm { i } } }$ . Then

$$
L _ { j } ^ { * } = \lfloor \log _ { 2 } \left( \frac { d _ { \operatorname* { m a x } } } { \Delta _ { c _ { j } } } \right) \rceil + \Delta L , i f \ \mathbf { g } _ { \mu _ { \mathrm { j } } } > \mathbf { g } _ { \mathrm { t h r e s h o l d } }\tag{5}
$$

where $\mathrm { \bf g } _ { \mathrm { t h r e s h o l d } }$ is a threshold. During progressive training, we limit the maximum level to $L _ { \mathrm { m a x } }$ . An anchor is selected for rendering if:

$$
L _ { j } \leq \operatorname* { m i n } ( \lfloor L _ { j } ^ { * } \rceil , L _ { \operatorname* { m a x } } )\tag{6}
$$

This function implies that if an anchor moves to a distant coarser layer, it is considered unimportant. As a result, we remove it from the rendering process.

Hierarchical Gaussian Dropout. To mitigate overfitting caused by limited training viewpoints and stabilize the optimization process, we enhance DropGaussian [18] by introducing a level-aware dropout strategy. Since higher LoDs comprise denser Gaussians, we apply a progressively increasing dropout rate defined as $\begin{array} { r } { r _ { m } ^ { ( l ) } = \bar { \gamma ^ { ( l ) } } \cdot \frac { m } { M } } \end{array}$ , where m is the current training step, M is the total number of training steps, and the scale factor $\gamma ^ { ( l ) } = 0 . 1 + 0 . 0 5 \cdot l$ grows with the LoD level l.

Finally, unlike Scaffold-GS [49] that uses separate networks to predict Gaussian attributes, we use a shared MLP network $\mathrm { F _ { m l p } }$ to predict Gaussian attributes of the hierarchical Gaussians ${ \mathcal { G } } = \{ { \mathcal { G } } _ { 0 } , { \mathcal { G } } _ { 1 } , \dots , { \mathcal { G } } _ { L - 1 } \}$ from these anchor features Ëf, viewing distance $\Delta _ { c } ,$ and camera information $\vec { d } _ { c }$ :

$$
\mathcal { G } = \{ \mathcal { G } _ { 0 } , \mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { L - 1 } \} = \mathrm { F } _ { \mathrm { m l p } } ( \hat { \mathbf { f } } , \Delta _ { c } , \vec { d } _ { c } )\tag{7}
$$

We find that using a shared MLP helps avoid the introduction of excessive parameters from multiple networks, thereby accelerating both training and rendering without degrading performance.

## B. GMM-based Motion Partitioning

After initializing the LoD-Structured Gaussian with the first frame (defined as the canonical space $\mathcal { G } _ { \mathrm { c a n o } } )$ as described in Sec. IV-A, we focus on modeling only the dynamic regions in subsequent frames to reduce optimization costs and enable faster training. Our idea is to compare the current frame $\{ \mathbf { I } _ { t = c } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ with the canonical frame $\{ \dot { \bar { \mathbf { I } } } _ { t = 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ rendered from Gcano to calculate the gradient of their differences to estimate the spatial variations. This gradient will allow us to categorize anchors of the scene into two groups: dynamic anchors $\mathbf { a } _ { \mathrm { d y n } }$ and static anchors $\mathbf { a } _ { \mathrm { s t a t i c } }$ . The gradient is calculated by:

$$
\mathbf { g } _ { \mu _ { \mathrm { c a n o } } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla _ { \mu _ { \mathrm { c a n o } } } \left\| \mathbf { I } _ { t = c } ^ { ( i ) } - \hat { \mathbf { I } } _ { t = 0 } ^ { ( i ) } \right\| ^ { 2 }\tag{8}
$$

where $\pmb { \mu } _ { \mathrm { c a n o } }$ denotes the 2D screen-space projection of the Gaussian center $\pmb { \mu }$ in the canonical space. We then apply a GMM [25] to cluster anchors based on the gradient ${ \bf g } _ { \mu _ { \mathrm { c a n o } } } ,$ where the component with the higher mean corresponds to the dynamic regions. Anchors are classified as dynamic anchors $\mathbf { a } _ { \mathrm { d y n } }$ if their GMM posterior probability exceeds the threshold $\rho = 8 0 \%$ , while the remaining anchors are classified as static anchors $\mathbf { a } _ { \mathrm { s t a t i c } }$ . The dynamic anchors $\mathbf { a } _ { \mathrm { d y n } }$ are subsequently utilized for quantized dynamics learning, as described in Sec. IV-C.

## C. Quantized Residual Refinement

After decomposing the dynamic anchors $\mathbf { a } _ { \mathrm { d y n } }$ and static ones $\mathbf { a } _ { \mathrm { s t a t i c } }$ , we aim to efficiently compress the temporal changes in dynamic scenes. This is accomplished by learning a quantized residual Ër for the dynamic anchors $\mathbf { a } _ { \mathrm { d y n } } \colon$

$$
\mathbf { A } _ { \mathrm { d y n } } = \mathbf { A } _ { \mathrm { d y n } } + \tilde { \mathbf { r } } , \quad \tilde { \mathbf { r } } = \left\{ \begin{array} { l l } { \tilde { \mu } _ { \mathrm { a n c h o r } } \quad } & { \mathrm { A n c h o r ~ l o c a t i o n } } \\ { \mathbf { Q } ( \tilde { \mathbf { l } } _ { \mathrm { a t t r } } ) \quad \mathrm { O t h e r ~ a t t r i b u t e s } } \end{array} \right.\tag{9}
$$

where $\mathbf { A } _ { \mathrm { d y n } }$ represents the Anchor attributes of $\mathbf { a } _ { \mathrm { d y n } } . \tilde { \mu } _ { \mathrm { a n c h o r } }$ is a learnable parameter representing the Anchorsâ spatial coordinates, which we do not quantize due to its negligible storage footprint. $\tilde { \mathbf { l } } _ { \mathrm { a t t r } }$ is also a learnable parameter (comprising the anchor feature and offset) that we subject to quantization. We define $\mathbf { Q } ( \widetilde { \mathbf { l } } _ { \mathrm { a t t r } } )$ as the quantized latents, with the quantization optimized through the Straight-Through Estimator [54]. By fixing the static anchors $\mathbf { a } _ { \mathrm { s t a t i c } }$ and modeling the quantized residuals of the dynamic anchors $\mathbf { a } _ { \mathrm { d y n } } ,$ , our method achieves approximately 80% storage reduction while preserving visual quality (see Tab. IV).

## D. Streaming Training

Fig.2-bottom illustrates our streaming training process. In the first step, we initialize the LoD-structured 3D Gaussian representation (Fig.2-(a)) using the first frame $\{ \mathbf { I } _ { t = 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ to train the model. This step determines the structure of the canonical space $\mathcal { G } _ { \mathrm { c a n o } } .$ , including the number of LoD layers, the initial Gaussian attributes, and the centers and quantity of anchors. For subsequent frames $\{ \mathbf { I } _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ where $t = 1 , 2 , \dots , T$ we optimize both GMM-based Motion Partitioning and Quantized Residual Refinement in a frame-by-frame, streaming manner, where only the parameters related to dynamic anchors are optimized. All training is conducted using the same loss function:

$$
\mathcal { L } _ { \mathrm { r e c o n } } = ( 1 - \lambda ) \mathcal { L } _ { 1 } + \lambda \mathcal { L } _ { D \mathrm { - S S I M } }\tag{10}
$$

where Î» is a hyper-parameter. We set $\lambda = 0 . 2$ to prioritize sharp reconstructions while maintaining perceptual coherence [10].

## V. EXPERIMENTS

In this section, we first introduce the experimental setup. Next, we present quantitative and qualitative comparisons against state-of-the-art benchmarks to validate our reconstruction quality across various view densities. Furthermore, we provide an systematic ablation analysis targeting individual components, the number of training views, and the LoD-AO farchitectureâconcluding with an evaluation of the limitations of the proposed StreamLoD-GS.

## A. Experimental Setup

1) Datasets: We conduct experiments on two public datasets: the Neural 3D Video (N3DV) [55] and the Meet Room [1] datasets. N3DV comprises six dynamic indoor scenes, where each video is captured at a resolution of 2704 Ã 2078 and 30 FPS, with 18-21 videos per scene. Following [10]â[13], [15], we downsample the resolution to 1352 Ã 1014. We adopt 3, 4, and 6 views for training and use the remaining views for evaluation.

The Meet Room dataset was recorded using 13 synchronized Azure Kinect cameras across 3 indoor scenes, namely âDiscussionâ, âTrimmingâ, and âVrheadsetâ, at a resolution of 1280 Ã 720 and 30 FPS. We use the original resolution and train with 2, 3, 4, 5, 6, 9, and 12 views, using remaining views for evaluation.

2) Implementation: Following the protocols in [10]â[13], [15], we utilize the first 300 frames (around 10 seconds) for evaluation. Initial point clouds are generated using VGGT [52] based on the provided camera poses. To ensure a rigorous and fair comparison, this initialization is consistently applied across our framework and all competing baselines for every scene. Regarding model hyper-parameters, we set the base voxel size Î´ to 0.001 and and the number of Gaussians per anchor K to 10. Based on empirical observations, the GMM posterior probability threshold $\rho$ is established at 80%. The $F _ { \mathrm { m l p } }$ module (detailed in Fig. 2 and Sec. IV-A) comprises 2- layer MLPs with ReLU activation and a 32-dimensional hidden layer. For attribute quantization, the latent dimensionality for both anchor features and offsets is uniformly set to 12. As specified in Sec. IV-A, the maximum LoD level $L _ { \mathrm { m a x } }$ is incrementally increased during optimization. For anchor point refinement, we compute average gradients over 200 iterations for the initial frame and 30 iterations for subsequent frames. Following each refinement cycle, an anchor is pruned if the accumulated opacity of its associated neural Gaussians falls below a threshold of 0.05. All remaining hyper-parameters adhere to the configurations specified in Octree-GS [21].

TABLE I  
QUANTITATIVE COMPARISON IN MEET ROOM [1] DATASET WITH 3, 4, AND 5 TRAINING VIEWS. THE BEST RESULTS WITHIN EACH CATEGORY IS MARKED IN BOLD. THE SECOND RESULTS ARE HIGHLIGHTED WITH AN UNDERLINE.
<table><tr><td>Views</td><td>Methods</td><td>PSNR (dB)â</td><td>SSIMâ</td><td>LPIPSâ</td><td>Storage (MB)â</td><td>Train (s)â</td><td>Render (FPS)â</td></tr><tr><td rowspan="7">3-views</td><td>StreamRF [1]</td><td>15.82</td><td>0.641</td><td>0.510</td><td>22.81</td><td>7.854</td><td>14.9</td></tr><tr><td>3DGStream [10]</td><td>17.24</td><td>0.687</td><td>0.481</td><td>3.822</td><td>5.526</td><td>588</td></tr><tr><td>HiCoM [11]</td><td>16.92</td><td>0.665</td><td>0.431</td><td>0.913</td><td>1.762</td><td>517</td></tr><tr><td>4DGC [15]</td><td>17.36</td><td>0.713</td><td>0.416</td><td>0.653</td><td>17.41</td><td>418</td></tr><tr><td>QUEEN [12]</td><td>17.01</td><td>0.670</td><td>0.443</td><td>0.860</td><td>0.455</td><td>486</td></tr><tr><td>StreamLoD-GS</td><td>18.50</td><td>0.783</td><td>0.400</td><td>0.245</td><td>0.643</td><td>677</td></tr><tr><td>StreamLoD-GS</td><td>18.85</td><td>0.793</td><td>0.383</td><td>0.253</td><td>1.073</td><td>547</td></tr><tr><td rowspan="7">4-views</td><td>StreamRF [1]</td><td>16.75</td><td>0.748</td><td>0.452</td><td>19.53</td><td>8.210</td><td>17.1</td></tr><tr><td>3DGStream [10]</td><td>18.36</td><td>0.797</td><td>0.413</td><td>3.850</td><td>5.510</td><td>609</td></tr><tr><td>HiCoM [11]</td><td>18.58</td><td>0.809</td><td>0.391</td><td>0.894</td><td>2.012</td><td>524</td></tr><tr><td>4DGC [15]</td><td>19.45</td><td>0.810</td><td>0.385</td><td>0.794</td><td>18.61</td><td>441</td></tr><tr><td>QUEEN [12]</td><td>19.26</td><td>0.753</td><td>0.386</td><td>0.823</td><td>0.581</td><td>484</td></tr><tr><td>StreamLoD-GS</td><td>20.67</td><td>0.823</td><td>0.370</td><td>0.209</td><td>0.823</td><td>689</td></tr><tr><td>StreamLoD-GS</td><td>21.40</td><td>0.832</td><td>0.351</td><td>0.215</td><td>1.581</td><td>625</td></tr><tr><td rowspan="7">5-views</td><td>StreamRF [1]</td><td>17.57</td><td>0.780</td><td>0.432</td><td>16.46</td><td>8.921</td><td>19.3</td></tr><tr><td>3DGStream [10]</td><td>19.53</td><td>0.817</td><td>0.400</td><td>3.841</td><td>5.440</td><td>619</td></tr><tr><td>HiCoM [11]</td><td>19.69</td><td>0.826</td><td>0.391</td><td>0.819</td><td>2.225</td><td>518</td></tr><tr><td>4DGC [15]</td><td>21.28</td><td>0.831</td><td>0.380</td><td>0.639</td><td>19.52</td><td>425</td></tr><tr><td>QUEEN [12]</td><td>21.22</td><td>0.803</td><td>0.343</td><td>0.603</td><td>0.692</td><td>485</td></tr><tr><td>StreamLoD-GS*</td><td>21.95</td><td>0.843</td><td>0.343</td><td>0.198</td><td>0.973</td><td>621</td></tr><tr><td>StreamLoD-GS</td><td>22.73</td><td>0.861</td><td>0.310</td><td>0.205</td><td>1.910</td><td>608</td></tr></table>

The model is trained for 500 epochs during the initial timestep, followed by 10 epochs for each subsequent timestep to facilitate rapid adaptation. Following the evaluation protocols established in [10]â[13], we assess visual quality using frame-wise Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM) [56], and Learned Perceptual Image Patch Similarity (LPIPS) [57]. These metrics are averaged across all views to ensure a comprehensive evaluation. Additionally, we report the average storage footprint, training latency per timestep, and rendering throughput (FPS). All experiments are conducted on a single NVIDIA A800-SXM4- 80GB GPU.

3) Baseline Methods: We evaluate the proposed StreamLoD-GS against several state-of-the-art SFVV frameworks, including StreamRF [1], 3DGStream [10], HiCoM [11], 4DGC [15], and QUEEN [12].

To ensure a rigorous and unbiased evaluation, all baselines are implemented using their official configurations within a standardized experimental environment. Specifically, we enforce a consistent point cloud initialization (utilizing VGGT [52]) across all methods for each scene. For QUEEN, which adopts a similar training protocol to ours, adhere strictly to its official settings. All models are retrained using identical view inputs on the same hardware to eliminate any performance discrepancies arising from computational variances. Additionally, we evaluate a specialized variant,

StreamLoD-GSâ, derived from our proposed StreamLoD-GS. This variant performs static anchor updates and dynamic anchor identification (as described in Sec. IV-B) every four frames to accumulate richer gradient information. Moreover, given the limited number of dynamic anchors within short intervals, StreamLoD-GSâ classifies an anchor as dynamic when its posterior probability, as defined in Sec. IV-B, exceeds $\rho = 9 0 \%$

## B. Quantitative Comparisons under Sparse Views

We conduct a comprehensive quantitative evaluation across varying view densities, comparing our framework against state-of-the-art baselines in Tab. I and Tab. II. Tab. I details the comparative results on the Meeting Room [1] dataset. The results demonstrate that StreamLoD-GS consistently surpasses nearly all baselines across the tested view configurations. Specifically, in the highly challenging 3-view setting, StreamLoD-GS achieves the highest quality metrics, while the StreamLoD-GSâ variant maintains superior rendering speeds, the most compact storage, and competitive training efficiency. At 4 views, StreamLoD-GS reaches a PSNR of 21.40 dB, marking a significant 1.95 dB improvement over 4DGC while requiring only 27% of its storage footprint. For the 5-view configuration, our method again secures the best quality metrics with a mere 0.205 MB of storageâ66% reduction compared to QUEEN, despite the latterâs specialized Gaussian-based learned quantization framework.

Similarly, quantitative results on the N3DV dataset [55] (Tab. II) further validate our modelâs efficiency. With 3 training views, our method achieves a PSNR of 20.54 dB, outperforming the second-best method (4DGC) while consuming only 15% of its storage. At 4 views, StreamLoD-GS reaches 21.30 dB, surpassing the nearest competitor by 1.14 dB. Notably, our storage footprint remains exceptionally lean at

QUANTITATIVE COMPARISON IN N3DV [55] DATASET WITH 3, 4, AND 5 TRAINING VIEWS. THE BEST RESULTS WITHIN EACH CATEGORY IS MARKED IN BOLD. THE SECOND RESULTS ARE HIGHLIGHTED WITH AN UNDERLINE.  
TABLE II
<table><tr><td>Views</td><td>Methods</td><td>PSNR (dB)â</td><td>SSIMâ</td><td>LPIPSâ</td><td>Storage (MB)â</td><td>Train (s)â</td><td>Render (FPS)â</td></tr><tr><td rowspan="7">3-views</td><td>StreamRF [1]</td><td>18.79</td><td>0.732</td><td>0.401</td><td>41.23</td><td>13.614</td><td>19.5</td></tr><tr><td>3DGStream [10]</td><td>19.89</td><td>0.747</td><td>0.373</td><td>6.301</td><td>9.162</td><td>662</td></tr><tr><td>HiCoM [11]</td><td>19.21</td><td>0.741</td><td>0.352</td><td>1.201</td><td>1.83</td><td>512</td></tr><tr><td>4DGC [15]</td><td>19.82</td><td>0.763</td><td>0.339</td><td>0.871</td><td>17.49</td><td>484</td></tr><tr><td>QUEEN [12]</td><td>19.25</td><td>0.743</td><td>0.330</td><td>0.962</td><td>0.452</td><td>460</td></tr><tr><td>StreamLoD-GS</td><td>20.50</td><td>0.785</td><td>0.320</td><td>0.112</td><td>0.793</td><td>651</td></tr><tr><td>StreamLoD-GS</td><td>20.54</td><td>0.790</td><td>0.316</td><td>0.126</td><td>1.336</td><td>530</td></tr><tr><td rowspan="7">4-views</td><td>StreamRF [1]</td><td>19.08</td><td>0.747</td><td>0.390</td><td>39.68</td><td>16.46</td><td>22.3</td></tr><tr><td>3DGStream [10]</td><td>20.16</td><td>0.780</td><td>0.363</td><td>6.610</td><td>8.830</td><td>687</td></tr><tr><td>HiCoM [11]</td><td>19.81</td><td>0.767</td><td>0.326</td><td>1.024</td><td>2.36</td><td>501</td></tr><tr><td>4DGC [15]</td><td>20.12</td><td>0.779</td><td>0.305</td><td>0.837</td><td>18.20</td><td>478</td></tr><tr><td>QUEEN [12]</td><td>19.64</td><td>0.773</td><td>0.293</td><td>0.710</td><td>0.568</td><td>482</td></tr><tr><td>StreamLoD-GS</td><td>21.14</td><td>0.812</td><td>0.283</td><td>0.243</td><td>0.982</td><td>679</td></tr><tr><td>StreamLoD-GS</td><td>21.30</td><td>0.817</td><td>0.280</td><td>0.256</td><td>1.696</td><td>525</td></tr><tr><td rowspan="7">6-views</td><td>StreamRF [1]</td><td>20.95</td><td>0.762</td><td>0.380</td><td>12.78</td><td>17.25</td><td>26.1</td></tr><tr><td>3DGStream [10]</td><td>21.85</td><td>0.800</td><td>0.351</td><td>6.624</td><td>8.965</td><td>666</td></tr><tr><td>HiCoM [11]</td><td>21.17</td><td>0.812</td><td>0.306</td><td>0.847</td><td>4.519</td><td>499</td></tr><tr><td>4DGC [15]</td><td>22.54</td><td>0.833</td><td>0.298</td><td>0.784</td><td>19.07</td><td>472</td></tr><tr><td>QUEEN [12]</td><td>22.88</td><td>0.841</td><td>0.289</td><td>0.718</td><td>0.816</td><td>466</td></tr><tr><td>StreamLoD-GS</td><td>23.36</td><td>0.842</td><td>0.271</td><td>0.211</td><td>1.401</td><td>524</td></tr><tr><td>StreamLoD-GS</td><td>23.85</td><td>0.852</td><td>0.256</td><td>0.223</td><td>2.313</td><td>506</td></tr></table>

<!-- image-->  
Fig. 3. Qualitative Results in N3DV [55] and Meet Room [1] Datasets with 3 training views. We demonstrate novel view results produced by 3DGStream [10], QUEEN [12], and our approach for comparison.

0.256 MB, which is 64% smaller than that of QUEEN. In the 6-view scenario, our approach maintains its lead in quality metrics with a compact 0.223 MB storage. Throughout these tests, the StreamLoD-GSâ variant consistently ranks second in quality while delivering near-optimal rendering speeds and significantly accelerated training times.

TABLE III  
QUANTITATIVE COMPARISON IN MEET ROOM [1] DATASET WITH 12 VIEWS FOR TRAINING AND RESERVED ONE FOR TEST. THE BEST RESULTS WITHIN EACH CATEGORY IS MARKED IN BOLD. THE SECOND RESULTS ARE HIGHLIGHTED WITH AN UNDERLINE.
<table><tr><td>Methods</td><td>PSNR (dB)â</td><td>Storage (MB)â</td><td>Train (s)â</td><td>Render (FPS)â</td></tr><tr><td>3DGStream [10]</td><td>26.36</td><td>4.51</td><td>4.95</td><td>309</td></tr><tr><td>HiCoM [11]</td><td>26.73</td><td>0.60</td><td>3.92</td><td>289</td></tr><tr><td>4DGC [15]</td><td>26.87</td><td>0.63</td><td>21.3</td><td>273</td></tr><tr><td>QUEEN [12]</td><td>27.35</td><td>0.35</td><td>1.31</td><td>254</td></tr><tr><td>StreamLoD-GS</td><td>27.84</td><td>0.19</td><td>2.93</td><td>312</td></tr></table>

## C. Quantitative Comparisons under Dense Views

To further evaluate the robustness of StreamLoD-GS, we conduct a comparative analysis against state-of-the-art baselines under dense-view conditions, utilizing 12 views for training and one for testing. As summarized in Tab. III, StreamLoD-GS consistently outperforms all competitors, achieving the highest PSNR, the fastest rendering throughput, and the most compact storage footprint, while maintaining highly competitive training efficiency. These results underscore the superior scalability and computational efficiency of our framework, demonstrating that StreamLoD-GS not only excels in Sparse-View reconstruction but also maintains a significant performance edge in information-rich, dense-view scenarios.

## D. Qualitative Analysis

Fig. 3 presents a visual comparison of the reconstruction results across various methods. On the N3DV dataset [55] (Scene: Coffee Martini, first row), our LoD-based method reconstructs finer and more structurally complete details than 3DGStream [10] and QUEEN [12], e.g., the window and the transition between the personâs head and the background wall. Notable improvements are visible in the crispness of the window frames and the sharp boundary transition between the subjectâs head and the background wall. Specifically, our approach effectively mitigates the pronounced floaters prevalent in 3DGStream and the structural erosion seen in QUEEN. Similarly, for the Meeting Room dataset [1] (Scene: Discussion, second row), our method preserves superior detail in both the foreground (e.g., the intricate structures of the chair and microphone) and the background (e.g., the reflections on the glass wall), whereas baseline methods exhibit significant blurring or geometric distortion.

TABLE IV  
ABLATION STUDY RESULTS ON MEET ROOM [1] DATASET WITH 3 TRAINING VIEWS AND RESERVED ALL FOR TEST. LOD-AO: LOD-STRUCTURED 3DGS WITH ANCHOR AND OCTREE. GMM-PART: GMM-BASED MOTION PARTITIONING. Q: QUANTIZED RESIDUAL REFINEMENT; THE BEST RESULTS WITHIN EACH CATEGORY IS MARKED IN BOLD.
<table><tr><td>LoD-AO</td><td>GMM-Part</td><td>Q</td><td>PSNR (dB)â</td><td>SSIMâ</td><td>LPIPSâ</td><td>Storage (MB)â</td><td>Train (s)â</td><td>Render (FPS)â</td></tr><tr><td>â</td><td>â</td><td></td><td>22.51</td><td>0.852</td><td>0.320</td><td>1.683</td><td>1.673</td><td>582</td></tr><tr><td>â</td><td></td><td></td><td>23.11</td><td>0.857</td><td>0.326</td><td>0.283</td><td>2.197</td><td>425</td></tr><tr><td></td><td>â</td><td>&gt;&gt;</td><td>20.02</td><td>0.778</td><td>0.337</td><td>0.633</td><td>0.902</td><td>487</td></tr><tr><td>â</td><td>â</td><td>â</td><td>22.73</td><td>0.865</td><td>0.310</td><td>0.205</td><td>1.910</td><td>608</td></tr></table>

<!-- image-->  
Fig. 4. Quantitative comparison on the N3DV (Scene: Coffee Martini) dataset [55] with 3 training views. We compare novel view results on several frames sampled at equal temporal intervals, which are produced by 3DGStream [10], 4DGC [15], QUEEN [12], and our approach.

Additionally, we also compare the image details of the same frame across the current state-of-the-art methods in the âDiscussionâ and âCoffee Martiniâ scenes. To provide a more comprehensive comparison, in Fig.4 and Fig.5, we compare the reconstruction results of different methods across several frames sampled at equal temporal intervals from two scenes, N3DV [55] (Coffee Martini) and Meet Room [1] (Discussion). As shown in Fig.4 and Fig.5, across these sampled frames, our proposed StreamLoD-GS consistently produces higher-quality reconstructions. Compared to the pronounced artifacts in 3DGStream [10], the loss of fine-scale details in QUEEN [12], and the temporal overfitting observed in 4DGC [15], our approach delivers more stable results with richer and more accurate scene details.

<!-- image-->  
Fig. 5. Quantitative comparison on the Meet Room (Scene: Discussion) dataset [1] with 3 training views. We compare novel view results on several frames sampled at equal temporal intervals, which are produced by 3DGStream [10], 4DGC [15], QUEEN [12], and our approach.

<!-- image-->  
Fig. 6. Quantitative comparison in Meet Room [1] dataset with varying numbers of training views.

TABLE V  
ATTRIBUTE-PREDICTION MLP (FMLP) WITH THE SEPARATE NETWORKS FOR ESTIMATING GAUSSIAN ATTRIBUTES.
<table><tr><td>Methods</td><td>PSNR (dB)â</td><td>Storage (MB)â</td><td>Train (s)â</td><td>Render (FPS)â</td></tr><tr><td>w Fseps</td><td>18.68</td><td>0.290</td><td>1.330</td><td>425</td></tr><tr><td>W/o HD-GS</td><td>17.99</td><td>0.256</td><td>1.096</td><td>498</td></tr><tr><td>LoD-AO</td><td>18.85</td><td>0.253</td><td>1.073</td><td>547</td></tr></table>

## E. Ablation Study

1) Analysis of Module Contributions: We evaluate the individual contributions of three core components: LoD-Structured 3DGS with Anchor and Octree (LoD-AO), GMM-Based Motion Partitioning (GMM-Part), and Quantized Residual Refinement (Q). As summarized in Tab. IV, the full configuration of StreamLoD-GS achieves the optimal balance across all metrics, including SSIM, LPIPS, storage efficiency, and rendering throughput. Specifically, removing the Quantized Residual Refinement module leads to a marginal decline in reconstruction quality while inflating the storage footprint to 1.683 MB. The exclusion of GMM-Part results in increased storage and training latency, with rendering speed dropping to 425 FPS. Notably, omitting LoD-AO reduces training time to 0.902 s, but at the cost of a severe 2.71 dB PSNR drop (from 22.73 dB to 20.02 dB) and a more than twofold increase in storage. These results underscore that each component is indispensable and that they work synergistically to maintain high performance.

<!-- image-->  
Fig. 7. Rendering results with dynamic anchors, which are separated by using our GMM-based motion partitioning strategy, on a set of equally spaced frames from the Meet Room (Scene: Discussion) dataset [1].

2) Analysis of LoD-AO with Hierarchical Dropout and Shared MLP : Within the LoD-AO framework, we introduce Hierarchical Gaussian Dropout (HD-GS)âa level-aware strategy designed to mitigate overfitting under Sparse-View constraints and stabilize the optimization process. Furthermore, to avoid the parameter explosion associated with multiple networks, we employ a shared MLP $\mathrm { ( F _ { m l p } ) }$ to predict attributes for hierarchical Gaussians. Unlike Scaffold-GS [49], which utilizes separate networks for this task, our shared architecture accelerates both training and rendering without compromising fidelity. To validate these design choices, we compare our shared $\mathrm { F _ { m l p } }$ against a variant using separate networks $\left( \mathrm { F _ { s e p s } } \right)$ and evaluate the impact of removing the HD-GS module. As shown in Tab. V, the $\mathrm { F _ { s e p s } }$ variant exhibits a noticeable decrease in PSNR and slower rendering throughput, confirming the efficiency and efficacy of the shared network. Most significantly, disabling the HD-GS module results in the most substantial decline in PSNR, demonstrating that hierarchical dropout is critical for maintaining reconstruction fidelity and enabling robust optimization in Sparse-View settings. These findings highlight that the integration of $F _ { \mathrm { m l p } }$ and HD-GS allows LoD-AO to achieve peak rendering speeds (547 FPS) and minimal storage costs while maximizing training efficiency.

3) Analysis of Different Training Views: Fig. 6 illustrates the reconstruction performance of various methods on the Meet Room [1] dataset across a range of training view counts (2, 3, 4, 5, 6, 9, and 12). For each configuration, all available views are reserved for testing to ensure a rigorous assessment of novel-view synthesis. As depicted in the results, all evaluated methods exhibit a positive correlation between PSNR and the number of training views, confirming that additional viewpoints provide essential geometric and photometric constraints for scene reconstruction. Notably, StreamLoD-GS consistently outperforms 3DGStream [10], HiCoM [11], 4DGC [15], and QUEEN [12] across all view densities. This sustained performance gap underscores the superior robustness and generalization of our framework under Sparse-View constraints. These findings highlight StreamLoD-GSâs exceptional ability to effectively reconstruct high-quality dynamic scenes even when provided with severely limited training data.

## F. Limitation

As illustrated in Fig. 7, our GMM-based motion partitioning strategy effectively decouples static and dynamic regions as the video sequence progresses. However, a notable limitation persists when processing static, visually homogeneous surfaces with specular reflections, such as the reflective chair surfaces, the central laptop, and the uniform white desk (indicated by the red and green boxes in Fig. 7). Specifically, the view-dependent appearance changes induced by reflections introduce temporal inconsistencies, a well-recognized challenge in recent literature [58], [59]. These subtle photometric variations across consecutive frames generate phantom gradients that can lead the GMM to misclassify static regions as dynamic. Consequently, enhancing the robustness of motion segmentation for visually uniform areas, particularly under complex lighting and reflective conditions, remains a critical objective. In future work, we aim to address this by integrating both color-consistency constraints and 3D geometric priors to better distinguish true motion from specular artifacts, thereby improving the precision of dynamic anchor identification.

## VI. CONCLUSION

In this work, we introduce StreamLoD-GS, a novel hierarchical Level-of-Detail (LoD) Gaussian Splatting framework designed for efficient streaming Free-Viewpoint Video (SFVV) reconstruction. To mitigate overfitting and stabilize the optimization process, we incorporate a Hierarchical Gaussian Dropout technique, which selectively drops unnecessary Gaussians. StreamLoD-GS also features a GMM-based motion partitioning module that separates the 3DGS into dynamic and static regions, represented by dynamic and static anchors, respectively. During subsequent training, static anchors remain frozen, while dynamic anchors are updated in a streaming manner as new frames arrive. Additionally, a Quantized Residual Refinement module is introduced to quantize the residuals of the dynamic anchors, effectively reducing storage costs without compromising performance. To assess the effectiveness of StreamLoD-GS, we conducted extensive experiments across various view settings. The results show that our method outperforms existing approaches, delivering significant improvements in storage efficiency, rendering speed, and visual quality.

## REFERENCES

[1] L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan, âStreaming radiance fields for 3d video synthesis,â in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 13 485â13 498.

[2] Y. Chen, Q. Wang, H. Chen, X. Song, H. Tang, and M. Tian, âAn overview of augmented reality technology,â in Journal of Physics: Conference Series, vol. 1237, no. 2. IOP Publishing, 2019, p. 022082.

[3] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[4] A. Collet, M. Chuang, P. Sweeney, D. Gillett, D. Evseev, D. Calabrese, H. Hoppe, A. Kirk, and S. Sullivan, âHigh-quality streamable freeviewpoint video,â ACM Transactions on Graphics (ToG), vol. 34, no. 4, pp. 1â13, 2015.

[5] Y. Chen, Y. Liang, Z. Wang, D. Wang, C. Xie, S. Jiao, and L. Zhang, âLivegs: Live free-viewpoint video via high-performance gaussian splatting for mobile devices,â in ACM SIGGRAPH 2025 Emerging Technologies, 2025, pp. 1â2.

[6] X. Zhang, Z. Liu, Y. Zhang, X. Ge, D. He, T. Xu, Y. Wang, Z. Lin, S. Yan, and J. Zhang, âMega: Memory-efficient 4d gaussian splatting for dynamic scenes,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 27 828â27 838.

[7] L. Tang, J. Yang, R. Peng, Y. Zhai, S. Shen, and R. Wang, âCompressing streamable free-viewpoint videos to 0.1 mb per frame,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 7, 2025, pp. 7257â7265.

[8] M. Kim, J. Lim, and B. Han, â4d gaussian splatting in the wild with uncertainty-aware regularization,â Advances in Neural Information Processing Systems, vol. 37, pp. 129 209â129 226, 2024.

[9] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20 310â20 320.

[10] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, â3dgstream: Onthe-fly training of 3d gaussians for efficient streaming of photo-realistic free-viewpoint videos,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 675â20 685.

[11] Q. Gao, J. Meng, C. Wen, J. Chen, and J. Zhang, âHicom: Hierarchical coherent motion for dynamic streamable scenes with 3d gaussian splatting,â in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 80 609â80 633.

[12] S. Girish, T. Li, A. Mazumdar, A. Shrivastava, S. De Mello et al., âQueen: Quantized efficient encoding of dynamic gaussians for streaming free-viewpoint videos,â Advances in Neural Information Processing Systems, vol. 37, pp. 43 435â43 467, 2024.

[13] J. Yan, R. Peng, Z. Wang, L. Tang, J. Yang, J. Liang, J. Wu, and R. Wang, âInstant gaussian stream: Fast and generalizable streaming of dynamic scene reconstruction via gaussian splatting,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 16 520â16 531.

[14] H. Han, J. Li, H. Wei, and X. Ji, âEvent-3dgs: Event-based 3d reconstruction using 3d gaussian splatting,â Advances in Neural Information Processing Systems, vol. 37, pp. 128 139â128 159, 2024.

[15] Q. Hu, Z. Zheng, H. Zhong, S. Fu, L. Song, X. Zhang, G. Zhai, and Y. Wang, â4dgc: Rate-aware 4d gaussian compression for efficient streamable free-viewpoint video,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 875â885.

[16] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM transactions on graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[17] L. Wang, Q. Hu, Q. He, Z. Wang, J. Yu, T. Tuytelaars, L. Xu, and M. Wu, âNeural residual radiance fields for streamably free-viewpoint videos,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 76â87.

[18] H. Park, G. Ryu, and W. Kim, âDropgaussian: Structural regularization for sparse-view gaussian splatting,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21 600â21 609.

[19] Y. Xu, L. Wang, M. Chen, S. Ao, L. Li, and Y. Guo, âDropoutgs: Dropping out gaussians for better sparse-view rendering,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 701â710.

[20] D. Luebke, M. Reddy, J. D. Cohen, A. Varshney, B. Watson, and R. Huebner, Level of detail for 3D graphics. Elsevier, 2002.

[21] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, âOctree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians,â IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[22] J. Shen, Y. Qian, and X. Zhan, âLod-gs: Achieving levels of detail using scalable gaussian soup,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 671â680.

[23] Y. Seo, Y. S. Choi, H. S. Son, and Y. Uh, âFlod: Integrating flexible level of detail into 3d gaussian splatting for customizable rendering,â arXiv preprint arXiv:2408.12894, 2024.

[24] J. Kulhanek, M.-J. Rakotosaona, F. Manhardt, C. Tsalicoglou, M. Niemeyer, T. Sattler, S. Peng, and F. Tombari, âLodge: Levelof-detail large-scale gaussian splatting with efficient rendering,â arXiv preprint arXiv:2505.23158, 2025.

[25] H. Permuter, J. Francos, and I. Jermyn, âA study of gaussian mixture models of color and texture features for image classification and segmentation,â Pattern recognition, vol. 39, no. 4, pp. 695â706, 2006.

[26] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[27] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-nerf: A multiscale representation for antialiasing neural radiance fields,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5855â5864.

[28] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5470â5479.

[29] A. Cao and J. Johnson, âHexplane: A fast representation for dynamic scenes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 130â141.

[30] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, âK-planes: Explicit radiance fields in space, time, and appearance,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 479â12 488.

[31] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y. Xu, and A. Geiger, âNerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields,â IEEE Transactions on Visualization and Computer Graphics, vol. 29, no. 5, pp. 2732â2742, 2023.

[32] L. Liu, Z. Chen, and D. Xu, â3d gaussian splatting data compression with mixture of priors,â in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 8341â8350.

[33] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, âpixelnerf: Neural radiance fields from one or few images,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 4578â 4587.

[34] A. Jain, M. Tancik, and P. Abbeel, âPutting nerf on a diet: Semantically consistent few-shot view synthesis,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5885â5894.

[35] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. Sajjadi, A. Geiger, and N. Radwan, âRegnerf: Regularizing neural radiance fields for view synthesis from sparse inputs,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5480â 5490.

[36] G. Wang, Z. Chen, C. C. Loy, and Z. Liu, âSparsenerf: Distilling depth ranking for few-shot novel view synthesis,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 9065â9076.

[37] J. Yang, M. Pavone, and Y. Wang, âFreenerf: Improving few-shot neural rendering with free frequency regularization,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 8254â8263.

[38] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PmLR, 2021, pp. 8748â8763.

[39] J. Chung, J. Oh, and K. M. Lee, âDepth-regularized optimization for 3d gaussian splatting in few-shot images,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 811â 820.

[40] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, âDngaussian: Optimizing sparse-view 3d gaussian radiance fields with globallocal depth normalization,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20 775â20 785.

[41] A. Paliwal, W. Ye, J. Xiong, D. Kotovenko, R. Ranjan, V. Chandra, and N. K. Kalantari, âCoherentgs: Sparse novel view synthesis with coherent 3d gaussians,â in European Conference on Computer Vision. Springer, 2024, pp. 19â37.

[42] J. Zhang, J. Li, X. Yu, L. Huang, L. Gu, J. Zheng, and X. Bai, âCor-gs: sparse-view 3d gaussian splatting via co-regularization,â in European Conference on Computer Vision. Springer, 2024, pp. 335â352.

[43] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, âFsgs: Real-time few-shot view synthesis using gaussian splatting,â in European conference on computer vision. Springer, 2024, pp. 145â163.

[44] S. Nam, D. Rho, J. H. Ko, and E. Park, âMip-grid: Anti-aliased grid representations for neural radiance fields,â Advances in Neural Information Processing Systems, vol. 36, pp. 2837â2849, 2023.

[45] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âZip-nerf: Anti-aliased grid-based neural radiance fields,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19 697â19 705.

[46] J. Cui, J. Cao, F. Zhao, Z. He, Y. Chen, Y. Zhong, L. Xu, Y. Shi, Y. Zhang, and J. Yu, âLetsgo: Large-scale garage modeling and rendering via lidar-assisted gaussian primitives,â ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp. 1â18, 2024.

[47] Y. Liu, C. Luo, L. Fan, N. Wang, J. Peng, and Z. Zhang, âCitygaussian: Real-time high-quality large-scale scene rendering with gaussians,â in European Conference on Computer Vision. Springer, 2024, pp. 265â 282.

[48] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and G. Drettakis, âA hierarchical 3d gaussian representation for real-time rendering of very large datasets,â ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1â15, 2024.

[49] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffold-gs: Structured 3d gaussians for view-adaptive rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 654â20 664.

[50] N. Snavely, S. M. Seitz, and R. Szeliski, âPhoto tourism: exploring photo collections in 3d,â in ACM siggraph 2006 papers, 2006, pp. 835â846.

[51] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4104â4113.

[52] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, âVggt: Visual geometry grounded transformer,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294â 5306.

[53] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, âOctree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians,â arXiv preprint arXiv:2403.17898, 2024.

[54] Y. Bengio, N. Leonard, and A. Courville, âEstimating or propagating Â´ gradients through stochastic neurons for conditional computation,â arXiv preprint arXiv:1308.3432, 2013.

[55] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim, T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe et al., âNeural 3d video synthesis from multi-view video,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5521â 5531.

[56] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE TIP, vol. 13, no. 4, pp. 600â612, 2004.

[57] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in CVPR, 2018, pp. 586â595.

[58] S. Yao, X. Zhang, X. Liu, M. Liu, and Z. Cui, âStdd: Spatio-temporal dual diffusion for video generation,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 12 575â12 584.

[59] Y. Liu, H. Zhao, K. C. Chan, X. Wang, C. C. Loy, Y. Qiao, and C. Dong, âTemporally consistent video colorization with deep feature propagation and self-regularization learning,â Computational visual media, vol. 10, no. 2, pp. 375â395, 2024.

<!-- image-->

Xinhui Liu received the B.S. and M.S. degrees in electronic information engineering from Xiâan Polytechnic University, Xiâan, China, in 2015 and 2018, respectively. She received the Ph.D. degree from Xiâan Jiaotong University, Xiâan, China, in 2024. From Sep. 2022 to May. 2024, she was a visiting student in the University of Sydney, NSW, Australia, and work with Prof. Luping Zhou. She is currently a Postdoctoral Researcher with the School of Computing and Data Science, the University of Hong Kong, Hong Kong, China. Her current research interests include computer vision and Computer Graphics.

<!-- image-->

Can Wang is currently a research fellow at The University of Hong Kong. He received the Ph.D. degree in the Department of Computer Science, City University of Hong Kong, HK, Peopleâs Republic of China. He got his M.S. in Computer Science and Technology from the University of Science and Technology of China in 2020. His current research interests include computer graphics and computer vision.

<!-- image-->

Lei Liu Lei Liu is a Postdoctoral Fellow at the University of Hong Kong. He received his Ph.D. degree from Beihang University, China, in 2025. He obtained his Bachelorâs and Masterâs degrees from the College of Instrumentation and Electrical Engineering, Jilin University, in 2017 and 2020, respectively. His research interests include visual data compression and 3D Gaussian Splatting editing.

<!-- image-->

Zhenghao Chen Dr. Zhenghao Chen is an Assistant Professor at the University of Newcastle. He obtained his B.IT. H1 and Ph.D. from the University of Sydney, in 2017 and 2022. Previously, he was a Research Engineer at TikTok, a Research Fellow at the University of Sydney, and a Visiting Research Scientist at Microsoft Research and Disney Research. Dr. Chenâs general research interests encompass Computer Vision and Machine Learning.

<!-- image-->

Wei Jiang received her PhD degree in E.E. from Columbia University in 2010, her M.S. and B.E. degrees in Automation from Tsinghua University in 2005 and 2002, respectively. She has published over 40 refereed papers and owned over 40 issued patents in the field of computer vision and machine learning, and holds 10+ standard adoptions in IEEE and JVET/MPEG/JPEG standards. She is a senior member of IEEE and serves as TPC member and reviewers for several top conferences and journals. Wei Jiang is currently a Senior Principal Researcher in Futurewei Technologies. She has broad research interests in computer vision and artificial intelligence, including AI-based image and video compression, image and video restoration and generation, visual computing, and multimedia content analysis. Her current research is focused on research and standardization of next-generation AI-based image/video compression and restoration, image/video generation by vision-language models, and inference acceleration.

<!-- image-->

Wei Wang received his M.S. and B.S. degree from E.E. Department of Fudan University, China, in 1998 and 1995, respectively. He is currently a Principal Researcher in Futurewei Technologies, He was formerly a Senior Staff Researcher in Alibaba Group. He has been involved in international standardization activities, including contributing to ISO/IEC Moving Picture Experts Group for work items on H.265/HEVC SCC and neural network representation, and to IEEE Data Compression Standards Committee for work item on neural image coding. He served as co-Chair of MPEG neural network representation group. His current research interests include high performance computing, image and video compression, neural networks compression and acceleration, and converting algorithms to product on desktop, embedded or ASIC platforms.

<!-- image-->

Dong Xu (Fellow, IEEE) received the B.E. and Ph.D. degrees from the University of Science and Technology of China, Hefei, China, in 2001 and 2005, respectively. While pursuing the Ph.D. degree, he was an Intern with Microsoft Research Asia, Beijing, China, and a Research Assistant with The Chinese University of Hong Kong, Hong Kong, for more than two years. He was a Postdoctoral Research Scientist at Columbia University, New York, NY, USA, for one year. He also worked as a Faculty Member at Nanyang Technological University, Sin-

gapore, and the Chair of computer engineering at The University of Sydney, NSW, Australia. He is currently a Professor with the School of Computing and Data Science, The University of Hong Kong, Hong Kong, China. He was the co-author of a paper that received the Best Student Paper Award from the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) in 2010 and a paper that received the Prize Paper Award in IEEE Transactions on Multimedia in 2014. His current research interests include Artificial Intelligence, Computer Vision, Multimedia and Machine Learning.