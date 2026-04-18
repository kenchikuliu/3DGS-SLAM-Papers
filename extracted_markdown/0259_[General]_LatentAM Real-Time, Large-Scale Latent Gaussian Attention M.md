# LatentAM: Real-Time, Large-Scale Latent Gaussian Attention Mapping via Online Dictionary Learning

Junwoon Lee and Yulun Tian

<!-- image-->  
Fig. 1: Large-scale online latent mapping with LatentAM on a custom multi-floor dataset spanning approximately 530 m of trajectory over 14.6 minutes. The system incrementally constructs a latent 3D Gaussian map from streaming RGB-D observations, while supporting plug-and-play integration with different visualâlanguage models including CLIP [1], DINOv3 [2], and LSeg [3]. The resulting latent map directly supports downstream open-vocabulary perception tasks such as language-driven querying in 3D.

Abstractâ We present LatentAM, an online 3D Gaussian Splatting (3DGS) mapping framework that builds scalable latent feature maps from streaming RGB-D observations for open-vocabulary robotic perception. Instead of distilling highdimensional vision-language Model (VLM) embeddings using model-specific decoders, LatentAM proposes an online dictionary learning approach that is both model-agnostic and pretraining-free, enabling plug-and-play integration with different VLMs at test time. Specifically, our approach associates each Gaussian primitive with a compact query vector that can be converted into approximate VLM embeddings using an attention mechanism with a learnable dictionary. The dictionary is initialized efficiently from streaming observations and optimized online to adapt to evolving scene semantics under trust-region regularization. To scale to long trajectories and large environments, we further propose an efficient map management strategy based on voxel hashing, where optimization is restricted to an active local map on the GPU, while the global map is stored and indexed on the CPU to maintain bounded GPU memory usage. Experiments on public benchmarks and a large-scale custom dataset demonstrate that LatentAM attains significantly better feature reconstruction fidelity compared to state-of-the-art methods, while achieving near-real-time speed (12-35 FPS) on the evaluated datasets. Our project page is at: https://junwoonlee.github.io/ projects/LatentAM

## I. INTRODUCTION

Simultaneous Localization and Mapping (SLAM) is a cornerstone ability that provides situational awareness for mobile robots in unknown environments. Recent advances have significantly improved robustness, scalability, and integration with downstream tasks. However, emerging robotic applications require richer spatial grounding than what is supported by existing geometry-centric or closed-set semantic SLAM approaches, demanding fine-grained and generalpurpose scene understanding. To this end, recent visionlanguage models (VLMs) [1]â[5] offer powerful visual priors learned from internet-scale data. A central challenge, however, is to effectively deploy such representations within onboard SLAM, where perception must be performed incrementally from streaming observations under occlusions, limited viewpoints, and strict runtime and memory constraints.

To address the challenges posed by occlusions and viewpoint variability, recent work has explored fusing 2D VLM embeddings into consistent 3D representations based on point clouds [6], neural radiance field [7], [8], volumetric maps [9], and 3D Gaussian Splatting (3DGS) [10]â[20]. Recently, 3DGS [21] has emerged as an attractive representation due to its superior real-time rendering performance. While the aforementioned approaches typically focus on offline reconstruction, a growing body of work [16]â[18], [22], [23] integrates VLM embeddings with incremental 3DGS mapping to enable 3D visualâlanguage mapping from streaming observations. Despite this progress, existing approaches exhibit two critical limitations. First, they remain computationally expensive, often operating far below sensor rate and limited to small-scale (e.g., single room) environments. Second, many methods rely on feature distillation by pretraining model-specific encoders or decoders. Such a design tightly couples the mapping system to specific choices of VLM, making it costly to incorporate new models or adapt to evolving feature distributions.

Contributions. We present LatentAM, a novel framework for online visualâlanguage mapping based on featureaugmented 3D Gaussian Splatting from streaming RGB-D observations. Rather than relying on feature distillation that requires model-specific encoders or decoders, our core insight is to formulate online feature mapping as a dictionary learning [24] problem, in which a compact dictionary of VLM embeddings are incrementally constructed and updated as the robot navigates and maps the environment. The online reconstructed dictionary enables our approach to continuously adapt to new environments, viewpoints, and semantic content, while a trust-region-style regularization prevents overfitting and catastrophic forgetting during longterm operation. As a direct consequence, our approach is both model-agnostic and pretraining-free; in our experiments, we demonstrate plug-and-play integration with CLIP [1], DINOv3 [2], and LSeg [3] within the same framework. Furthermore, by integrating the proposed dictionary-based feature learning into a scalable localâglobal 3DGS mapping system with voxel hashing, we achieve substantially improved mapping speed and memory efficiency, enabling near-real-time operation (12â35 FPS depending on parameter settings) and scaling to large-scale environments (> 530 m) as shown in Fig. 1. The proposed method improves feature reconstruction accuracy by 72.5% and achieving 12.1Ã speedups compared to the state-of-the-art method [12].

## II. RELATED WORK

## A. SLAM with 3D Gaussian Splatting (3DGS)

3D Gaussian Splatting [21] has recently emerged as a powerful representation for high-fidelity, real-time 3D reconstruction and rendering. Building upon this representation, researchers have proposed 3DGS-based SLAM systems [14], [15], [25]â[27]; see also [28] and the references therein. GS-ICP [27] introduces a 3DGS SLAM framework that achieves real-time performance by using fast Gaussian primitive registration via geometric alignment. OpenGS-SLAM [14] incorporates semantic labels into a GS-ICP backbone to ensure semantic consistency, while OpenGS-Fusion [15] extends this framework to a hybrid representation based on 3DGS and truncated signed distance function.

## B. Offline Feature Mapping

Several papers bridge the gap between 2D VLMs and 3D spatial representations by fusing foundation model embeddings into neural 3D representations. LERF [7] pioneered language-embedded neural representation by volumerendering CLIP features along NeRF rays, trained with multiscale CLIP supervision to capture both global semantics and fine details. Several works extended this paradigm to 3DGS by distilling high-dimensional foundation model embeddings into low-dimensional Gaussian embeddings [10], [11]. Feature-splatting [10] supervises 3D Gaussians using both photometric and feature reconstruction losses, enabling downstream tasks such as semantic segmentation, editing, and retrieval. M3 [12] proposes a different paradigm that avoids direct distillation of high-dimensional embeddings into Gaussians. Instead, it stores observed embeddings in a global dictionary as principal scene components and learns low-dimensional queries for each Gaussian via attention. OpenGaussian [13] associates 3D points with 2D instance-level features stored in a spatial codebook, enabling position-dependent clustering of semantic features. Recently, GOI [20] and CCL-LGS [19] propose learnable codebooks trained with clustering-based objectives, yielding compact and distinctive codebook atoms that improve reconstruction quality by adapting to feature distribution shifts. However, these methods rely on offline optimization for codebook learning, and their clustering-based updates require substantial memory while lacking mechanisms for incremental dictionary maintenance for online SLAM settings.

## C. Incremental and Online Feature Mapping

As an online feature mapping method, Online Language Splatting [22] enables online open-vocabulary mapping within an 3DGS-SLAM pipeline by distilling CLIP features through a lightweight super-resolution decoder. Similarly, LEGO-SLAM [17] learns a scene-adaptive decoder to distill high-dimensional features into low-dimensional Gaussian embeddings, and further employs a latent feature codebook that remains fixed after construction. LEG-SLAM [16] adopts a different strategy by embedding VLM features using a pretrained PCA projection. Similar to [22], FeatureSLAM [29] introduces a pretrained autoencoder with lightweight learnable adaptation layers to efficiently handle streaming data, while improving tracking stability via a feature-based pose tracker. However, distillation-based methods are highly sensitive to model capacity, where insufficient capacity leads to underfitting and excessive capacity increases memory usage and training time.

To address the capacity-accuracy trade-off of decoder-based distillation, recent works have explored codebook-based schemes for efficiently reconstructing high-dimensional embeddings in incremental mapping. OmniMap [23] maintains a global embedding codebook indexed by semantic instance IDs, where each atom stores a instance-level embedding updated over time; the codebook is queried using cosine-similarity scores to associate each instance with embeddings. OpenMonoGS-SLAM [18] similarly maintains a compact codebook of representative embeddings, updates it online using pairwise cosine similarity among atoms, and reconstruct high-dimensional embeddings via memory attention [12]. Nevertheless, by fixing the dictionary atoms, these methods cannot adapt their latent basis to the non-stationary feature distribution encountered over long trajectories, causing stale atoms as new semantics appear and ultimately limiting large-scale mapping scalability.

<!-- image-->  
Fig. 2: Overview of the proposed online dictionary learning and memory attention pipeline.

## III. PROPOSED METHOD

## A. Preliminary: 3DGS Representation

We represent a scene as a collection of 3D Gaussian primitives $\mathcal { G } ~ = ~ \{ G _ { i } \} _ { i = 1 } ^ { N }$ , where each primitive encodes geometric and appearance attributes:

$$
G _ { i } = \left\{ { \bf p } _ { i } , \ { \bf q } _ { i } , \ { \bf c } _ { i } , \ { \bf s } _ { i } , \ \alpha _ { i } \right\} .\tag{1}
$$

Here, $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ denotes the position, $\mathbf { q } _ { i } \in \mathbb { R } ^ { 4 }$ the orientation, $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 }$ the color, $\mathbf { s } _ { i } \in \mathbb { R } ^ { 3 }$ the scale, and $\alpha _ { i } \in [ 0 , 1 ]$ the opacity. We use view-independent color and omit spherical harmonics to enable efficient online optimization like [25].

The renderer $\mathcal { R } ( \cdot )$ maps a set of Gaussian primitives G to an image using depth-ordered alpha blending [21]. Let $\{ G _ { i } \} _ { i = 1 } ^ { N }$ 1 denote the Gaussians intersecting a given pixel ray p, sorted by increasing depth. The rendered color is,

$$
{ \bf C } _ { p } = \sum _ { i = 1 } ^ { N } \alpha _ { i } { \bf c } _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \right) ,\tag{2}
$$

A per-pixel depth value can be rendered using the same alpha-blending operation. Given the camera pose $\mathbf { T } \in \mathrm { S E } ( 3 )$ and known camera intrinsics, we denote the rendering of color and depth by $\widehat { \mathbf { I } } = \mathcal { R } _ { c } ( \mathcal { G } , \mathbf { T } )$ and $\widehat { \mathbf { Z } } = \mathcal { R } _ { d } ( \mathcal { G } , \mathbf { T } )$ , where I and Z are the rendered 2D color image and depth.

## B. Feature Splatting via Dictionary Learning

In this subsection, we extend the standard 3DGS representation to enable scalable latent feature mapping as shown in Fig 2. Instead of directly storing high-dimensional visual features at each Gaussian, we associate each primitive with a low-dimensional query vector and reconstruct full feature embeddings through a dictionary learning formulation.

Feature-augmented Gaussian Representation. In addition to the geometric and photometric attributes introduced in Sec. III-A, each Gaussian primitive stores a latent query feature $\mathbf { f } _ { i } \in \mathbb { R } ^ { d _ { s } }$ , i.e.,

$$
G _ { i } = \left\{ { \bf p } _ { i } , \ { \bf q } _ { i } , \ { \bf c } _ { i } , \ { \bf s } _ { i } , \ \alpha _ { i } , \ { \bf f } _ { i } \right\} .\tag{3}
$$

Here, $d _ { s } \ll d _ { f }$ is the dimension of the query vector (set to 32 by default in our implementation), and $d _ { f }$ is the dimension of the target VLM feature, $\mathbf { e } . \mathbf { g } . , d _ { f } \ = \ 7 6 8$ for CLIP [1]. Given a camera pose T, we can use the same 3DGS rendering equation to produce a per-view query feature map by applying the alpha blending operator (2) over the query feature $\mathbf { f } _ { i }$ instead of the color $\mathbf { c } _ { i }$ . In the following, we denote this operation by

$$
\mathbf { Q } = \mathcal { R } _ { f } ( \mathcal { G } , \mathbf { T } ) \in \mathbb { R } ^ { n \times d _ { s } } ,\tag{4}
$$

where n is the number of feature vectors in a single image. For example, for CLIP, n corresponds to the number of patches extracted using use SAM [4], [30].

Target Embedding Reconstruction with Dictionary Learning. Existing latent mapping techniques typically rely on 3D feature distillation, which compresses highdimensional foundation model embeddings into lowdimensional Gaussian attributes via learned decoders [10], [11]. While effective for reducing memory and computation, such distillation inevitably introduces an information bottleneck, which leads to loss of semantic fidelity and misalignment between the decoded embeddings and the original targets, as confirmed in our experiments (Sec. IV-C).

In contrast, the proposed approach builds on a dictionarybased strategy that is first proposed in M3 [12]. Specifically, we introduce and learn (i) a projection matrix $\mathbf { W } \in \mathbb { R } ^ { d _ { s } \times d _ { f } }$ , and (ii) a memory bank (dictionary) $\mathbf { D } \in \mathbb { R } ^ { K \times d _ { f } }$ that stores K representative embeddings in the target VLM space. Using W and D, we convert the queries Q in (4) into approximate embeddings via,

$$
\widehat { \mathbf { F } } = \operatorname { S o f t m a x } \left( \mathbf { Q } \mathbf { W } \mathbf { D } ^ { \top } \right) \mathbf { D } .\tag{5}
$$

We note that (5) can be viewed as augmenting the standard feature decoder with an attention-based reconstruction mechanism. In particular, the matrix W plays the role of a minimal linear decoder that maps Q to an initial reconstruction $\begin{array} { r } { \widetilde { \textbf { F } } = \textbf { Q W } } \end{array}$ . Then, the softmax operation in (5) computes pairwise similarities between $\widetilde { \mathbf { F } }$ and D, yielding attention weights that reflect the relevance of each dictionary atom. The final reconstructed embedding F is obtained as a weighted combination of dictionary atoms. This approach enables more expressive embeddings and avoids the information bottleneck inherent to direct distillation.

Nevertheless, the original approach in [12] assumes that the full set of latent embeddings is available a priori, and its memory bank is constructed in an offline manner. As a result, the approach does not naturally extend to streaming SLAM settings, where observations arrive sequentially and the feature distribution evolves over time. To address this issue, we take inspiration from the established research field of dictionary learning [24] and proposes a regularized optimization formulation that explicitly handles non-stationary feature distribution while preventing overfitting. Specifically, we jointly optimize the Gaussian primitives G, the projection matrix W, and the dictionary D via,

$$
\begin{array} { l } { \displaystyle \underset { \mathcal { G } , \mathbf { W } , \mathbf { D } } { \operatorname* { m i n } } } & { \displaystyle \mathcal { L } _ { \mathrm { f } } = \lambda _ { \mathrm { c o s } } \big ( 1 - \cos ( \widehat { \mathbf { F } } , \mathbf { F } ) \big ) + \lambda _ { 2 } \| \widehat { \mathbf { F } } - \mathbf { F } \| _ { 2 } ^ { 2 } } \\ { \displaystyle \quad \quad \quad + \lambda _ { D } \displaystyle \sum _ { j = 1 } ^ { K } \mathrm { R e L U } \Big ( \Big \| \mathbf { D } _ { j } ^ { ( 0 ) } - \mathbf { D } _ { j } \Big \| _ { 2 } - \delta \Big ) , } \end{array}\tag{6}
$$

where $\lambda _ { \mathrm { c o s } } , \lambda _ { 2 } , \lambda _ { D } , \delta \ > \ 0$ are constant parameters and $\mathbf { D } ^ { ( 0 ) }$ is the initial dictionary before optimization. In (6), the first two terms correspond to standard reconstruction loss based on the cosine similarity and L2 distance to the true (observed) embeddings F. The third term acts as a $\mathrm { \Delta ^ { 6 6 } s o f t { 7 } }$ trust-region constraint [31] that incurs penalty whenever the jth dictionary atom $\mathbf { D } _ { j }$ deviates outside a neighborhood of its initial value ${ \bf D } _ { j } ^ { ( 0 ) }$ with radius Î´. In Sec. IV-D, we show that the proposed trust-region regularization improves performance in large-scale scenarios by mitigating overfitting and catastrophic forgetting. While similar regularization could also be applied to the projection matrix W, we do not impose such a constraint in practice as we observe that such regularization yields minimal impact in our experiments.

In practice, we combine the feature supervision in (6) with standard photometric and depth supervision. For photometric supervision, we use L1 and SSIM losses between the rendered RGB image I and observed image I,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r g b } } = \lambda _ { I _ { 1 } } \| \widehat { \mathbf { I } } - \mathbf { I } \| _ { 1 } + \lambda _ { I _ { 2 } } \mathcal { L } _ { \mathrm { S S I M } } ( \widehat { \mathbf { I } } , \mathbf { I } ) . } \end{array}\tag{7}
$$

For depth supervision, we compute the L1 loss on the rendered depth image $\hat { \mathbf { Z } }$ and observed depth Z,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d } } = \| \widehat { \mathbf { Z } } - \mathbf { Z } \| _ { 1 } . } \end{array}\tag{8}
$$

Given camera pose T, the overall mapping problem is,

$$
\operatorname* { m i n } _ { \mathcal { G } , \mathbf { W } , \mathbf { D } } \ \lambda _ { \mathrm { r g b } } \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { \mathrm { d } } \mathcal { L } _ { \mathrm { d } } + \lambda _ { \mathrm { f } } \mathcal { L } _ { \mathrm { f } } ,\tag{9}
$$

where $\lambda _ { \mathrm { r g b } } , \lambda _ { \mathrm { d } }$ , and $\lambda _ { \mathrm { f e a t } }$ are scalar coefficients for the RGB, depth, and latent feature loss terms, respectively.

## C. Online Dictionary Learning

We now extend the proposed dictionary-based reconstruction framework to the online setting, where RGB-D observations and their associated embeddings arrive sequentially during real-time mapping. In contrast to offline formulations that assume access to all measurements, our approach introduces a lightweight streaming initialization mechanism that provides high-quality initial dictionary atoms from incoming embeddings. Further, we implement efficient optimization strategies for (9) to satisify the strict computational budgets required by real-time operation. Alg. 1 summarizes the proposed online mapping procedure upon receiving a new keyframe at time t.

Dictionary Initialization from Streaming Data. At each keyframe t, we first udpate dictionary atoms using a streaming K-means [32] procedure (line 1):

$$
\mathbf { D } _ { t } ^ { ( 0 ) } \gets \mathrm { S t r e a m K M e a n s } ( \mathbf { F } _ { t } , \mathbf { D } _ { t - 1 } , K ) ,\tag{10}
$$

where $\mathbf { F } _ { t }$ denotes the newly observed embeddings, $\mathbf { D } _ { t - 1 }$ is the current dictionary, and K is the target dictionary size. In streaming K-means [32], incoming embeddings are first summarized into weighted micro-centers via local K-means, and these summaries are then merged and re-clustered with existing global centers. This enables explicit control over dictionary size and supports fast, lightweight inference, in contrast to pairwise similarity-based methods [12], [18] that lack size control, and SVD-based approaches [33] that are computationally expensive. All steps are implemented using batched computation for GPU, allowing real-time clustering performance.

Algorithm 1 Online Mapping Upon Receiving Keyframe t   
Input: Gaussian primitives G; previous dictionary $\mathbf { D } _ { t - 1 } ;$ current   
camera pose $\mathbf { T } _ { t } ;$ observed and rendered RGB image I, I;   
observed and rendered depth Z, Z; observed embeddings $\mathbf { F } _ { t } ;$   
history buffer $\varkappa ;$ history size B; refinement iterations $R _ { 1 } , R _ { 2 } .$   
1: Update dictionary via streaming K-means   
$\mathbf { D } _ { t } ^ { ( 0 ) }$ â StreamKMeans $( \mathbf { { F } } _ { t } , \mathbf { { D } } _ { t - 1 } , K )$   
2: Augment training data $\mathbf { F } ^ { \mathrm { b } } , \mathbf { T } ^ { \mathrm { b } }$ with history   
$( \mathbf { F } ^ { \mathrm { b } } , \ \mathbf { T } ^ { \mathrm { b } } )  ( \mathbf { F } _ { t } , \mathbf { T } _ { t } ) \cup$ Sample(H, B)   
3: Stage I: jointly optimize G, W, D for $R _ { 1 }$ iterations   
G, W, D â argmin $\lambda _ { \mathrm { r g b } } \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { \mathrm { d } } \mathcal { L } _ { \mathrm { d } } + \lambda _ { \mathrm { f } } \mathcal { L } _ { \mathrm { f } }$   
G, W, D   
4: Stage II: refine W, D for $R _ { 2 }$ iterations   
$\begin{array} { r } { { \bf W } , { \bf D }  \underset { { \bf W } , { \bf D } } { \mathrm { a r g m i n } } \lambda _ { \mathrm { r g b } } \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { \mathrm { d } } \mathcal { L } _ { \mathrm { d } } + \lambda _ { \mathrm { f } } \mathcal { L } _ { \mathrm { f } } } \end{array}$   
5: Update history buffer: $\mathcal { H }  \mathcal { H } \cup \{ ( \mathbf { F } _ { t } , \mathbf { T } _ { t } ) \}$

Two-stage Optimization. Unlike offline methods that allow sufficient optimization time, directly coupling dictionary learning with Gaussian optimization in an online setting can be time consuming. We propose to implement the optimization in (9) using a two-stage strategy. In the first stage, we carry out joint optimization of the Gaussian primitives G, the projection matrix W, and the dictionary D for a small number of $R _ { 1 }$ iterations (1 by default); see line 3. In the second-stage, we perform an lightweight refinement that further optimizes W and D for $R _ { 2 }$ (5 by default) iterations while keeping G fixed; see line 4.

To address the catastrophic forgetting problem, we introduce a historical learning strategy (in lines 3, 4, 5) similar to prior work such as PIN-SLAM [34]. At each time step, we randomly sample B past keyframes from the history buffer, where B is set to 5 by default in our implementation, to augment the newly arrived keyframe, enabling the model to retain previously learned information while adapting to new observations. Together with the trust-region regularization in Eq. (6), this strategy effectively alleviates overfitting or catastrophic forgetting during online mapping.

## D. Large-Scale Mapping via Voxel Hashing

System Overview. Building on the proposed algorithms, we present an online RGB-D 3DGS mapping system that incrementally reconstructs large-scale scenes while jointly modeling geometry, appearance, and latent embeddings, and maintains bounded GPU memory usage by storing the global Gaussian map on the CPU and optimizing only an active local map on the GPU.

First, camera poses are tracked using an external real-time tracker [27], [35]. Depending on the tracker, keyframes are selected based on the mapâscan overlap for [27] and uniform interval sampling for [35]. For each selected keyframe, newly initialized 3D Gaussians are inserted into a voxel-hashed global map stored in CPU memory, discussed in full detail in the next paragraph. At any time, the system maintains a local map around the robot by retrieving Gaussians within a specific radius around the current camera position from the global map and transferring them to GPU memory. Then, these local Gaussians are optimized using photometric, depth, and feature losses, as discussed in Sec. III-B-III-C.

<!-- image-->  
Fig. 3: 3D segmentation results of the proposed method and baselines on Office0 (left), TUM3 (middle), and Floor2 (right) scenes. In the Floor2 scene, Feature-3DGS fails to detect the desk segments (yellow arrow) while our method succeeds (green arrow).

Local-Global Map Management via Voxel Hashing. To avoid duplicate primitives and enable fast spatial indexing, we organize the global map using a voxel-hashing structure. Each Gaussian is assigned to a voxel $\mathbf { v } _ { i } = \lfloor \mathbf { p } _ { i } / s _ { v } \rfloor$ , where $s _ { v }$ denotes the voxel size (0.01 m by default). A hash table $h : \mathbb { Z } ^ { 3 } \to \mathbb { N }$ is used to map integer voxel coordinates to their corresponding indices in CPU memory, storing at most one Gaussian per voxel. We note that hash collisions may occur when distinct voxel coordinates map to the same hash index. In practice, however, collisions occur with low probability and we observe that they have negligible impact on performance. Furthermore, since each rendered pixel aggregates contributions from many 3D Gaussians, alpha blending (2) typically makes the influence of a single collided primitive insignificant.

The global and local maps are synchronized periodically, with synchronization triggered when the traveled distance since the last synchronization exceeds a predefined threshold. Synchronization consists of two ordered steps to maintain consistency. In the GlobalâLocal step, the current local map is pruned to retain only Gaussians within the active radius, after which additional nearby Gaussians are retrieved from the voxel hash table and appended without duplication. In the LocalâGlobal step, locally optimized Gaussians are inserted back to the global map. Gaussians corresponding to existing global primitives, identified via voxel-hash lookup, directly update their associated entries, while newly created Gaussians are inserted by computing their voxel-hash values only if the voxel is unoccupied.

## IV. EXPERIMENTS

In this section, we present a comprehensive evaluation of LatentAM. Our results show that LatentAM achieves stateof-the-art feature reconstruction fidelity and is significantly faster than existing methods. Furthermore, we demonstrate the scalability of our approach on a large-scale multi-floor custom dataset. Lastly, we present detailed ablation studies to evaluate the contributions of key compoenents including the dictionary learning formulation, streaming K-meansâbased dictionary construction, and local map management.

## A. Experimental Setup

We conduct experiments on three public datasets: Replica [37], TUM [36], and the large-scale FastCaMo dataset [38]. In addition, to evaluate scalability and robustness in large-scale and long-term environments, we capture a custom dataset using a Gemini 335Le RGBD camera operating at 30 Hz (Sec. IV-D). For the hyperparameters of the proposed method, we use $K = 2 0 0 0 , \delta = 0 . 1 , \lambda _ { I _ { 1 } } = 0 . 2 \mathrm { { \Omega } }$ , $\lambda _ { I _ { 2 } } = 0 . 8 , \lambda _ { d } = 0 . 1 , \lambda _ { f } = 5 . 0 , \lambda _ { \mathrm { c o s } } = 0 . 5 , \lambda _ { 2 } = 0 . 5 ,$ , and $\lambda _ { D } = 0 . 2$ with the learning rates for W and D set to 0.01.

Baselines. We select state-of-the-art offline feature splatting approaches, including Feature-3DGS [11] and M3 [12], as well as state-of-the-art online methods, including Online Language Splatting [22] for feature splatting SLAM and OmniMap [23] for codebook-based semantic mapping. While OpenMonoGS-SLAM [18] is also related, the code is not available so we present an analysis in the form of an ablation in Sec. IV-C. For offline baselines and OmniMap, we use the same pose tracker [27] and keyframe selection as our method to ensure fair comparison. Each offline model is trained for 20 epochs, which is sufficient for full convergence across all scenes. We set the query dimension $d _ { s } \ = \ 3 2$ for both the offline methods and our method. For Online Language Splatting [22], we follow the original two-stage pipeline, in which both the autoencoder and Gaussian parameters are trained using a pretrained, super-resolution decoder, and MonoGS [25] is used as the tracker, consistent with the original implementation.

Performance Metrics. We use the cosine loss, defined as one minus the cosine similarity between the reconstructed and ground-truth 2D VLM embeddings. Following [10], [11], [22], we also compute the mean Intersection-over-Union (mIoU) over a predefined set of 10 semantic query words. To obtain per-class semantic probabilities, reconstructed and ground-truth 2D embedding vectors are compared with text embeddings via inner-product, followed by a softmax operation. For OmniMap, which performs instance segmentation instead of feature mapping, each 3D instance is projected to the 2D plane and re-grouped by query words to directly compute mIoU. For photometric evaluation, we compute the peak signal to noise ratio (PSNR) on the rendered RGB images. Although our main evaluation uses CLIP [1] and SAM [4] for feature extraction and mask pooling, the proposed method is model-agnostic and can operate with a variety of VLMs. To demonstrate this feature, we also present evaluation using DINOv3 [2] and LSeg [3]. Finally, we report frame per second (FPS) to evaluate real-time performance. All experiments are conducted on a workstation with NVIDIA RTX 5090 GPU, 64 GB RAM, and Intel Core Ultra 9 285K CPU.

TABLE I: Quantitative results on TUM [36], Replica [37], and FastCaMo-Large [38]. The best and second-best are shown in green and yellow. OOM denotes out-of-memory. OmniMap does not report cosine loss since it does not map VLM embeddings.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Metric</td><td colspan="3">TUM</td><td colspan="8">Replica</td><td colspan="4">FastCaMo-Large</td></tr><tr><td>TUM1</td><td>TUM2</td><td>TUM3</td><td>Room0</td><td>Room1</td><td>Room2</td><td>Office0</td><td>Office1</td><td>Office2</td><td>Office3</td><td>Office4 Lab</td><td>Floor1</td><td>Floor2</td><td>Stair1</td><td>Stair2</td></tr><tr><td rowspan="5">Feature3DGS [11]</td><td>Cos-loss â</td><td>0.130</td><td>0.078</td><td>0.100</td><td>0.078 0.072</td><td>0.077</td><td>0.096</td><td>0.058</td><td>0.065</td><td>0.078</td><td>0.064</td><td>0.119</td><td>0.045</td><td>0.074</td><td>0.043</td><td>0.033</td></tr><tr><td>mIoU â</td><td>0.576</td><td>0.949</td><td>0.797</td><td>0.635 0.543</td><td>0.345</td><td>0.447</td><td>0.383</td><td>0.532 22.64</td><td>0.133 24.93</td><td>0.413 23.95</td><td>0.344 1.5</td><td>0.637</td><td>00.493</td><td>0.223 2.71</td><td>0.224</td></tr><tr><td>SNR</td><td>15.000</td><td>19.49</td><td>15.59</td><td>22.63</td><td>25.52</td><td>24.28 0.663</td><td>38.04</td><td>38.02</td><td></td><td></td><td></td><td>22.54</td><td></td><td>24.21</td><td>17.29</td></tr><tr><td>PF PS</td><td>0.77</td><td>0.67</td><td>0.67</td><td>0.54</td><td>0.65</td><td>0.61</td><td></td><td>0.68 0.62</td><td>0.58</td><td>0.62</td><td>0.74</td><td>0.78</td><td>0.79</td><td>0.92</td><td>0.93</td></tr><tr><td>Cos-loss â</td><td>0.136</td><td>0.116</td><td>0.133</td><td>0.091</td><td>0.105</td><td>0.108</td><td>0.141</td><td></td><td></td><td></td><td>0.123</td><td>0.121</td><td>OOM</td><td>OOM</td><td>0.081</td></tr><tr><td rowspan="6">M3 [12]</td><td>OU </td><td>0.674</td><td>0.957</td><td>0.910</td><td>0.883 0.845</td><td>0.752</td><td>0.819</td><td>0.114 0.414</td><td>0.119 0.462</td><td>0.132 0.308</td><td>0.422</td><td>0.588</td><td>OM</td><td>OM</td><td>0.074 0.507</td><td>0.472</td></tr><tr><td>PSNR â</td><td>13.49</td><td>19.10</td><td>16.22</td><td>21.62 25.52</td><td>24.51</td><td>38.04</td><td>38.00</td><td>22.69</td><td>25.20</td><td>24.46</td><td>16.79</td><td>OM</td><td>OM</td><td>119.70</td><td>17.38</td></tr><tr><td>PS </td><td>0.77</td><td>0.67</td><td></td><td>0.65</td><td>0.63</td><td>0.61</td><td>0.68</td><td>0.62</td><td>00.57</td><td>0.62</td><td>00.73</td><td> OM</td><td>OOM</td><td>0.90</td><td>0.93</td></tr><tr><td></td><td></td><td></td><td>0.67</td><td>0.53</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mIoU â</td><td>0.449</td><td>0.405</td><td>0.391</td><td>0.492 0.665</td><td>0.590</td><td>0.282 26..17</td><td>0.207</td><td>0.415 20.39</td><td>0.522 21.77</td><td>0.419</td><td>0.149</td><td>0.189</td><td>0.292</td><td>0.337</td><td>0.453</td></tr><tr><td>PSNR PF PS</td><td>19.21</td><td>20..23 7.80</td><td>21.25 7.97</td><td>22.08 7.4</td><td>23.25 8.63</td><td>21.95</td><td></td><td>27.00</td><td></td><td></td><td>23.13</td><td>16.88</td><td>23.27</td><td>22.45 23.58</td><td>22.76</td></tr><tr><td rowspan="6">OnlineLangSplat [22]</td><td></td><td>6.98</td><td></td><td></td><td></td><td>8.12</td><td>8.87</td><td>9.0</td><td>7.26 0.206</td><td>6.84</td><td>7.22</td><td>6.39</td><td>6.33</td><td>6.46</td><td>7.13</td><td>7.</td></tr><tr><td>Cos-loss â</td><td>0.280</td><td>0.275</td><td>0.274</td><td>0.222 0.269</td><td>0.232</td><td>0.290</td><td>0.206</td><td></td><td>0.218</td><td>0.234</td><td>0.314</td><td>0.287</td><td>0.296</td><td>0.270</td><td>0.264</td></tr><tr><td>OU </td><td>0.714</td><td>0.588</td><td>0.697</td><td>0.609 0.321</td><td>0.484</td><td>0.774</td><td>0.934</td><td>0.884</td><td>0.709</td><td>0.692</td><td>0.504</td><td>0.522</td><td>0.425</td><td>0.241</td><td>0.247</td></tr><tr><td>SNR </td><td>18.02</td><td>15.558</td><td>18.74</td><td>29.77 34.44</td><td>31.65</td><td>37.04</td><td>37.13</td><td>316</td><td>34.85</td><td>3360</td><td>11.83</td><td>12.27</td><td>11.62</td><td>12.00</td><td>13.86</td></tr><tr><td>PS </td><td>0.43</td><td>0.44</td><td>0.37</td><td>1.13 1.08</td><td>1.23</td><td>1.38</td><td>1.49</td><td>1.37</td><td>1.38</td><td>1.12</td><td>1.52</td><td>1.44</td><td>1.57</td><td>1.51</td><td>1.55</td></tr><tr><td>Cos-loss â</td><td>0.110</td><td>0.096</td><td>0.094</td><td>0.069</td><td>0.055 0.061</td><td>0.083</td><td>0.045</td><td>0.058</td><td>0.068</td><td>0.056</td><td>0.090</td><td>0.079</td><td>0.074</td><td>0.043</td><td>0.042</td></tr><tr><td rowspan="3">LatentAM (ours)</td><td>mIOU</td><td>0.792</td><td>0.960</td><td>0.885</td><td>0.686 0.873</td><td>0.844</td><td>0..874</td><td>0.968</td><td>0.687</td><td>0.674</td><td>0453</td><td>00.758</td><td>0.668</td><td>0.613</td><td>0.457</td><td>0.476</td></tr><tr><td>PSNR</td><td>15.09</td><td>19.50</td><td>15.46</td><td>19.97 25.57</td><td>26.35</td><td>38.00</td><td>38.06</td><td>22.77</td><td>26.95</td><td>23.11</td><td>19.67</td><td>25.09</td><td>23.25</td><td>23.93</td><td>22.31</td></tr><tr><td>FPâS </td><td>20.24</td><td>20.17</td><td>17.89</td><td>18.63 117.90</td><td>114.78</td><td>19.82</td><td>22.40</td><td>20.42</td><td>117.41</td><td>18.86</td><td>13.64</td><td>17.07</td><td>15.30</td><td>28.62</td><td>25.34</td></tr></table>

## B. Evaluation of Mapping Performance

In this section, we present quantitative evaluation on benchmark datasets. Table I reports the results on TUM, Replica, and FastCaMo datasets. Qualitative 3D semantic segmentation results are shown in Fig. 3.

On the TUM and Replica datasets, our method consistently outperforms the compared methods in both feature reconstruction and open-vocabulary language tasks, while maintaining real-time performance. In particular, our method outperforms both offline distillation-based and dictionarybased approaches, thanks to the proposed dictionary learning that is designed to effectively adapt to diverse scene variations. Moreover, OmniMap relies on ground-truth poses in its original benchmark [23] and performs instance-level mapping; replacing the poses with GS-ICP [27] leads to unstable instance tracking, whereas our featureâbased mapping remains robust.

On the FastCaMo-Large dataset, our method still achieves state-of-the-art performance. While Feature-3DGS partially outperforms our method on some metrics, it is an offline method and requires an environment- and modeldependent decoder. In contrast, our approach operates online, is pretraining-free, and is agnostic to the specific choice of VLMs. M3 experiences out of memory issues in largescale scenes due to unbounded dictionary growth, while our streaming K-means maintains a fixed dictionary size.

TABLE II: Comparison under different visual foundation models on the TUM dataset. Best results are highlighted in green.
<table><tr><td></td><td></td><td colspan="3">CLIP-SAM</td><td colspan="3">DINOv3-SAM</td><td colspan="3">LSeg</td></tr><tr><td>Seq.</td><td>Method</td><td>Lcos4</td><td>L2â</td><td>FPSâ</td><td>Lcosâ</td><td>L24</td><td>FPSâ</td><td>Lcosâ</td><td>L2â</td><td>FPSâ</td></tr><tr><td rowspan="3">TUM1</td><td>Feat3DGS [11]</td><td>0.130</td><td>0.077</td><td>0.77</td><td>0.231</td><td>0.029</td><td>0.57</td><td>0.069</td><td>0.00028</td><td>0.56</td></tr><tr><td>M3 [12]</td><td>0.136</td><td>0.100</td><td>0.77</td><td>0.192</td><td>0.024</td><td>0.54</td><td>0.134</td><td>0.00051</td><td>0.55</td></tr><tr><td>LatentAM (ours)</td><td>0.110</td><td>0.073</td><td>20.24</td><td>0.185</td><td>0.020</td><td>16.99</td><td>0.064</td><td>0.00024</td><td>17.29</td></tr><tr><td rowspan="3">TUM2</td><td>Feat3DGS [11]</td><td>0.078 0.052</td><td></td><td>0.67</td><td>0.103</td><td>0.013</td><td>0.52</td><td>0.035</td><td>0.00022</td><td>0.55</td></tr><tr><td>M3 [12]</td><td>0.116</td><td>0.121</td><td>0.67</td><td>0.147</td><td>0.021</td><td>0.80</td><td>0.132</td><td>0.00052</td><td>0.55</td></tr><tr><td>LatentAM (ours)</td><td>0.096</td><td>0.061</td><td>20.17</td><td>0.145</td><td>0.014</td><td>23.11</td><td>0.053</td><td>0.00020</td><td>30.57</td></tr><tr><td rowspan="3">TUM3</td><td>Feat3DGS [11]</td><td>0.100</td><td>0.119</td><td>0.67</td><td>0.197</td><td>0.021</td><td>0.48</td><td>0.049</td><td>0.00029</td><td>0.51</td></tr><tr><td>M3 [12]</td><td>0.133</td><td>0.132</td><td>0.67</td><td>0.198</td><td>0.020</td><td>0.50</td><td>.119</td><td>0.00044</td><td>0.54</td></tr><tr><td>LatentAM (ours)</td><td>0.094</td><td>0.060</td><td>17.89</td><td>0.192</td><td>0.017</td><td>20.60</td><td>0.068</td><td>0.00026 26.40</td><td></td></tr></table>

Although our photometric rendering quality (PSNR metric in Table I) is lower than Online Language Splatting, this difference stems from a deliberate tracker choice. We prioritize real-time operation by using a lightweight Gaussian tracker [27]. Online Language Splatting uses a heavier tracker [25], which uses more keyframes for mapping at the expense of significantly slower speed. In comparison, our method achieves 12 FPS with short keyframe intervals (every 4 frames on average), and over 30 FPS with longer intervals (every 10 frames on average). In practice, the keyframe intervals are determined adaptively by the tracker.

Lastly, to demonstrate the model-agnostic mapping capability offered by LatentAM, we evaluate on alternative VLMs including DINOv3 [2] and LSeg [3] in Table II. The proposed method outperforms other offline methods on TUM1 and TUM3 across all visual foundation model settings. In TUM2, the scene is highly repetitive and exhibits minimal camera motion. As a result, the distillation-based method, Feature-3DGS, achieves slightly better performance, although our method still demonstrates comparable feature splatting performance to Feature-3DGS.

## C. Ablation Study

Dictionary Learning vs. Distillation. To demonstrate the efficiency of our online dictionary learning approach, we present a systematic evaluation compared to decoderbased distillation methods such as Feature-3DGS [11]. We report performance with respect to the number of learnable parameters excluding Gaussian primitives. The number of learnable parameters is controlled by the dictionary size in our method and by the decoder depth in the distillation-based method. For Feature-3DGS, we also add skip connections to the decoder to resolve vanishing gradient issues in the deeper layers. Fig. 4 reports the results on TUM1 and TUM3.

<!-- image-->  
Fig. 4: Comparison between dictionary learning and distillation [11] on TUM1 and TUM3. The semantic segmentation results are overlaid, with the ground-truth object (book) highlighted.

TABLE III: Ablation study on online dictionary learning. K-means denotes streaming K-meansâbased dictionary construction, and DL denotes dictionary learning. Best results are highlighted in green.
<table><tr><td rowspan="2">K-means</td><td rowspan="2">DL</td><td colspan="2">TUM1</td><td colspan="2">TUM2</td><td colspan="2">TUM3</td></tr><tr><td>Cos â</td><td>mIoU â</td><td>Cos â</td><td>mIoU â</td><td>Cos â</td><td>mIoU â</td></tr><tr><td>Ã</td><td>Ã</td><td>0.146</td><td>0.506</td><td>0.137</td><td>0.546</td><td>0.124</td><td>0.602</td></tr><tr><td>â</td><td>Ã</td><td>0.127</td><td>0.529</td><td>0.104</td><td>0.780</td><td>0.111</td><td>0.593</td></tr><tr><td>Ã</td><td>â</td><td>0.141</td><td>0.647</td><td>0.124</td><td>0.766</td><td>0.124</td><td>0.765</td></tr><tr><td>â</td><td>â</td><td>0.110</td><td>0.792</td><td>0.096</td><td>0.960</td><td>0.094</td><td>0.885</td></tr></table>

Distillation-based methods remain consistently weaker than our approach despite operating in an offline setting. This observation agrees with prior work [12] and suggests that pure distillation introduces distortion between the recovered and original embeddings. In contrast, our method maintains a compact dictionary of latent embeddings from past observations as explicit memory, which enables higher fidelity reconstruction. Moreover, unlike decoder-based approaches whose performance strongly depends on pretraining data selection, our dictionary is learned directly from streaming data during online mapping.

Online Dictionary Construction. We evaluate four variants of the proposed online dictionary learning method: (i) Random + w/o DL, where the dictionary is randomly initialized from all the past keyframes and kept fixed (without optimization); (ii) Random + with DL, where the randomly initialized dictionary is optimized online; (iii) Kmeans + w/o DL, where the dictionary is constructed using streaming Kmeans and then kept fixed; and (iv) Kmeans + with DL, which is the proposed approach combining streaming Kmeans initialization with dictionary optimization. Quantitative results are reported in Table III. Overall, both streaming K-means and online dictionary learning contribute significantly to the performance of our pipeline. Notably, refining the dictionary via the proposed optimization yields up to 59.4% improvement on the mIoU metric, which clearly demonstrate its advantage compared to static dictionary used by related work such as [18].

TABLE IV: Peak GPU usage (MB) during mapping. The best and second-best results are highlighted in green and yellow, respectively.
<table><tr><td></td><td colspan="2">TUM1</td><td colspan="2">TUM2</td><td colspan="2">TUM3</td></tr><tr><td>Method</td><td>Cos â</td><td>Mem. â</td><td>Cos â</td><td>Mem. â</td><td>Cos â</td><td>Mem. â</td></tr><tr><td>M3 [12]</td><td>0.136</td><td>5188.40</td><td>0.275</td><td>12852.8</td><td>0.274</td><td>10767.2</td></tr><tr><td>LatentAM (w/o local)</td><td>0.105</td><td>6428.77</td><td>0.087</td><td>14133.2</td><td>0.091</td><td>13845.0</td></tr><tr><td>LatentAM (with local)</td><td>0.110</td><td>5180.16</td><td>0.096</td><td>7388.47</td><td>0.094</td><td>8538.66</td></tr></table>

<!-- image-->  
Fig. 5: Comparison of segmentation results with and without trust-region regularization. The target word is monitor. Without regularization, the model misdectects monitors on the wall (yellow arrow), whereas the proposed method suppresses false positives.

Local Mapping. Lastly, we evaluate the efficiency of our local mapping strategy by recording peak GPU memory usage. As shown in Table IV, local mapping substantially reduces memory consumption when compared to M3 and proposed method without local mapping. Thanks to the hashed-voxel representation and the local map formulation, our method significantly improves memory efficiency while incurring minimal performance loss compared to optimizing all Gaussians in the global map.

## D. Large Scale Mapping

To evaluate the scalability of our system, we collect a large-scale, long-term dataset in a campus building environment. The dataset spans 14.6 minutes of continuous operation across two floors connected by a single staircase and includes corridor traversal and multiple room transitions (four rooms in total), covering a total traveled distance of approximately 530 m as shown in Fig. 1. For this experiment, we adopt ORB-SLAM3 [35] with loop-closure disabled as the pose tracker, owing to its tracking stability in large-scale environments. Our method successfully reconstructs both 3D photometric and latent representations over the entire trajectory. By leveraging CLIP embeddings, the reconstructed map enables open-vocabulary semantic queries directly in a largescale 3D map. In contrast, all baseline methods, including M3, Feature3DGS, and Online Language Splatting, fail to complete mapping due to GPU memory exhaustion. This result highlights the memory efficiency of our local-global map management strategy, which enables scalable feature splatting in large-scale environments.

Additionally, we present a qualitative evaluation of the constrained dictionary learning formulation introduced in (6) in Fig. 5. The proposed trust-region regularization yields better separated segmentations, directly contributing to better reconstruction quality in large scenes.

## V. CONCLUSIONS

We presented LatentAM, an online 3DGS-based mapping system that reconstructs geometry, appearance, and VLM embeddings in real time for scalable and versatile 3D perception tasks. By formulating feature splatting as an online dictionary learning problem and using streaming Kmeans initialization and trust-region regularization, LatentAM achieves high-fidelity embedding reconstruction using compact Gaussian queries without pretraining or modelspecific distillation networks. In addition, our local mapping strategy provides memory efficiency that enables long-term, large-scale mapping. Future work will improve photometric rendering quality through hierarchical representations and coarse-to-fine optimization strategies, incorporate languagedriven loop closure using VLM embeddings, and extend the framework to more general sensor configurations, including monocular cameras and LiDAR.

## REFERENCES

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., âLearning transferable visual models from natural language supervision,â in Proc. Int. Conf. Mach. Learn. (ICML), 2021.

[2] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, Â´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., âDINOv3,â arXiv preprint arXiv:2508.10104, 2025.

[3] B. Li, K. Q. Weinberger, S. Belongie, V. Koltun, and R. Ranftl, âLanguage-driven semantic segmentation,â in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[4] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., âSegment anything,â in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023.

[5] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar \` et al., âLLaMA: Open and efficient foundation language models,â arXiv preprint arXiv:2302.13971, 2023.

[6] K. M. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, T. Chen, S. Li, G. Iyer, S. Saryazdi, N. Keetha, A. Tewari, J. B. Tenenbaum, C. M. de Melo, M. Krishna, L. Paull, F. Shkurti, and A. Torralba, âConceptfusion: Open-set multimodal 3D mapping,â in Proc. Robot.: Sci. Syst. (RSS), 2023.

[7] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik, âLERF: Language embedded radiance fields,â in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023.

[8] L. Weijler, S. Koch, F. Poiesi, T. Ropinski, and P. Hermosilla, âOpenhype: Hyperbolic embeddings for hierarchical open-vocabulary radiance fields,â in Adv. Neural Inf. Process. Syst. (NeurIPS), 2025.

[9] J. Wilson, R. Xu, Y. Sun, P. Ewen, M. Zhu, K. Barton, and M. Ghaffari, âLatentBKI: Open-dictionary continuous mapping in visual-language latent spaces with quantifiable uncertainty,â IEEE Robotics and Automation Letters, 2025.

[10] R.-Z. Qiu, G. Yang, W. Zeng, and X. Wang, âLanguage-driven physicsbased scene synthesis and editing via feature splatting,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.

[11] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi, âFeature 3DGS: Supercharging 3D gaussian splatting to enable distilled feature fields,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[12] X. Zou, Y. Song, R.-Z. Qiu, X. Peng, J. Ye, S. Liu, and X. Wang, â3D-Spatial multimodal memory,â in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[13] Y. Wu, J. Meng, H. Li, C. Wu, Y. Shi, X. Cheng, C. Zhao, H. Feng, E. Ding, J. Wang et al., âOpengaussian: Towards point-level 3D gaussian-based open-vocabulary understanding,â in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2024.

[14] D. Yang, Y. Gao, X. Wang, Y. Yue, Y. Yang, and M. Fu, âOpengs-SLAM: Open-set dense semantic SLAM with 3D gaussian splatting for object-level scene understanding,â in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2025.

[15] D. Yang, X. Wang, Y. Gao, S. Liu, B. Ren, Y. Yue, and Y. Yang, âOpengs-fusion: Open-vocabulary dense mapping with hybrid 3D gaussian splatting for refined object-level understanding,â in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2025.

[16] R. Titkov, E. Zubkov, D. Yudin, J. Mahmoud, M. Mohrat, and G. Sidorov, âLEG-SLAM: Real-time language-enhanced gaussian splatting for SLAM,â arXiv preprint arXiv:2506.03073, 2025.

[17] S. Lee, S. Ha, K. Kang, J. Choi, S. Tak, and H. Yu, âLEGO-SLAM: Language-embedded gaussian optimization SLAM,â arXiv preprint arXiv:2511.16144, 2025.

[18] J. Yoo, G. Kang, H.-k. Ko, H. Yu, and E. Park, âOpenmonogs-SLAM: Monocular gaussian splatting SLAM with open-set semantics,â arXiv preprint arXiv:2512.08625, 2025.

[19] L. Tian, X. Li, L. Ma, H. Yin, Z. Zheng, H. Huang, T. Li, H. Lu, and X. Jia, âCCL-LGS: Contrastive codebook learning for 3D language gaussian splatting,â in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025.

[20] Y. Qu, S. Dai, X. Li, J. Lin, L. Cao, S. Zhang, and R. Ji, âGOI: Find 3D gaussians of interest with an optimizable open-vocabulary semanticspace hyperplane,â in Proc. ACM Int. Conf. Multimedia (ACM MM), 2024.

[21] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D gaussian Â¨ splatting for real-time radiance field rendering,â ACM Trans. Graph., 2023.

[22] S. Katragadda, C.-Y. Wu, Y. Guo, X. Huang, G. Huang, and L. Ren, âOnline language splatting,â in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025.

[23] Y. Deng, Y. Yue, J. Dou, J. Zhao, J. Wang, Y. Tang, Y. Yang, and M. Fu, âOmnimap: A general mapping framework integrating optics, geometry, and semantics,â IEEE Trans. Robot., 2025.

[24] J. Mairal, F. Bach, J. Ponce, and G. Sapiro, âOnline dictionary learning for sparse coding,â in Proc. Int. Conf. Mach. Learn. (ICML), 2009.

[25] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, âGaussian splatting SLAM,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[26] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3D gaussians for dense RGB-D SLAM,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[27] S. Ha, J. Yeon, and H. Yu, âRGBD GS-ICP SLAM,â in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024.

[28] F. Tosi, Y. Zhang, Z. Gong, E. Sandstrom, S. Mattoccia, M. R. Oswald, Â¨ and M. Poggi, âHow NeRFs and 3D gaussian splatting are reshaping SLAM: A survey,â arXiv preprint arXiv:2402.13255, 2024.

[29] C. Thirgood, O. Mendez, E. Ling, J. Storey, and S. Hadfield, âFeatureslam: Feature-enriched 3d gaussian splatting slam in real time,â arXiv preprint arXiv:2601.05738, 2026.

[30] C. Zhang, D. Han, Y. Qiao, J. U. Kim, S.-H. Bae, S. Lee, and C. S. Hong, âFaster segment anything: Towards lightweight sam for mobile applications,â arXiv preprint arXiv:2306.14289, 2023.

[31] J. Nocedal and S. J. Wright, Numerical optimization. Springer, 1999.

[32] L. OâCallaghan, N. Mishra, A. Meyerson, S. Guha, and R. Motwani, âStreaming-data algorithms for high-quality clustering,â in Proc. Int. Conf. Data Eng. (ICDE), 2002.

[33] R. Zhang, P. Madumal, T. Miller, K. A. Ehinger, and B. I. P. Rubinstein, âInvertible concept-based explanations for CNN models with non-negative concept activation vectors,â in Proc. AAAI Conf. Artif. Intell. (AAAI), 2021.

[34] Y. Pan, X. Zhong, L. Wiesmann, T. Posewsky, J. Behley, and C. Stachniss, âPIN-SLAM: LiDAR SLAM using a point-based implicit neural representation for achieving global map consistency,â IEEE Trans. Robot., 2024.

[35] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. M. Montiel, and J. D. Tardos, âORB-SLAM3: An accurate open-source library for visual, Â´ visualâinertial, and multi-map SLAM,â IEEE Trans. Robot., 2021.

[36] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of RGB-D SLAM systems,â in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2012.

[37] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[38] Y. Tang, J. Zhang, Z. Yu, H. Wang, and K. Xu, âMips-fusion: Multi-implicit-submaps for scalable and robust online neural RGB-D reconstruction,â ACM Trans. Graph., 2023.