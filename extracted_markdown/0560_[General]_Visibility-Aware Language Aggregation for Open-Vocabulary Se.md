# Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting

Sen Wang1,2 Kunyi Li1,2 Siyun Liang1,5 Elena Alegret 1 Jing Ma 4 Nassir Navab1,2 Stefano Gasperini1,2,3

1Technical University of Munich 2Munich Cental for Machine Learning 3VisualAIs

4Ludwig Maximilian University of Munich 5University of Tubingen Â¨

## Abstract

Recently, distilling open-vocabulary language features from 2D images into 3D Gaussians has attracted significant attention. Although existing methods achieve impressive language-based interactions with 3D scenes, we observe two fundamental issues: background Gaussians, which contribute negligibly to a rendered pixel, receive the same feature as the dominant foreground ones, and multi-view inconsistencies due to view-specific noise in language embeddings. We introduce Visibility-Aware Language Aggregation (VALA), a lightweight yet effective method that computes marginal contributions for each ray and applies a visibility-aware gate to retain only visible Gaussians. Moreover, we propose a streaming weighted geometric median in cosine space to merge noisy multi-view features. Our method yields a robust, view-consistent language feature embedding in a fast and memory-efficient manner. VALA improves open-vocabulary localization and segmentation across reference datasets, consistently surpassing existing works. The source code is available on VALA.

## 1. Introduction

Understanding 3D scenes is essential for interacting with the environment in robotic navigation [2, 23], autonomous driving [8, 32], and augmented reality [10, 17]. Traditional approaches, however, are constrained to a fixed set of object categories defined at training time [4, 27, 35], limiting their applicability to open-world scenarios. Thanks to recent advances in vision-language models [11, 30], openvocabulary methods [9, 24, 42] enable querying and interacting with 3D scenes through natural language, and recognizing unseen object categories without requiring retraining.

While classical 3D understanding methods operate on point clouds or meshes derived from 3D sensors, recent neural scene representations, such as NeRFs [22] and 3D Gaussian Splatting (3DGS) [14], have emerged as a compelling alternative. They not only enable high-quality rendering from novel viewpoints but also facilitate semantic reasoning, as appearance and geometry are encoded jointly. Thus, open-vocabulary reasoning has recently been grounded in neural 3D scene representations [15, 28], enabling new semantic interactions in 3D environments. Initially explored with NeRFs [7, 15], the efficiency and explicit nature of 3DGS simplified the integration of semantic features, contributing to its widespread adoption [3, 13, 28, 38].

<!-- image-->  
Figure 1. Thanks to its feature aggregation that is visibility-aware and multi-view consistent, our proposed VALA is the most accurate and as quick as the fastest [3] to optimize. Comparison in 3D open-vocabulary segmentation on the LeRF-OVS dataset [28].

At the core of these approaches lies the challenge of embedding reliable semantic and language features into the 3D representation. Current methods rely on powerful offthe-shelf 2D foundation models, such as SAM [16] and CLIP [30], which produce 2D feature maps that must be lifted to 3D and aggregated across views. Proper aggregation is critical for accurate 3D segmentation.

Despite numerous recent advances [12, 13, 18, 34], current approaches suffer from an inherent limitation: they assign 2D features indiscriminately to all Gaussians along a camera ray, disregarding scene geometry and occlusion relationships. Consequently, features originating from foreground objects (e.g., a vase) are incorrectly propagated to background structures (e.g., the supporting table or floor), leading to substantial degradation in openvocabulary recognition accuracy.

Furthermore, when lifted into 3D, 2D features exhibit multi-view inconsistencies. The same object may produce divergent feature representations across different viewpoints, a phenomenon known as semantic drift [15]. Current methods address this by promoting cross-view consistency through 3D-consistent clustering and contrastive objectives derived from SAM masks [18, 20, 26, 38]. Nevertheless, such strategies generally require extensive perscene optimization, and their heavy reliance on noisy, viewdependent 2D cues often undermines cluster reliability.

In this paper, we address these fundamental feature aggregation problems with VALA (Visibility-Aware Language Aggregation), a lightweight yet effective framework that combines a two-stage gating mechanism with a robust multi-view feature aggregation strategy. Our gating mechanism leverages the statistical distribution of per-ray Gaussian contributions (termed visibility) to preferentially propagate features to Gaussians with high visibility, thereby ensuring accurate feature assignment. To further mitigate multi-view inconsistencies in 2D language features, we introduce a convex but non-smooth optimization on the unit hypersphere, which we reformulate into a streaming gradient-based procedure that achieves consistent embeddings without additional computational overhead. As shown in Figure 1, VALA strategies are highly effective.

Our contributions can be summarized as follows:

â¢ We identify fundamental issues in the feature aggregation of current works as a bottleneck in open-vocabulary 3D scene understanding.

â¢ We introduce VALA, a visibility-aware feature propagation framework that employs a two-stage gating mechanism to assign features based on Gaussian visibility.

â¢ We propose a robust aggregation strategy for the 2D features using the streaming cosine median, thereby improving multi-view consistency.

â¢ We obtain state-of-the-art performance in 2D and 3D on open-vocabulary segmentation for 3DGS scenes on the reference datasets LeRF-OVS [28] and ScanNet-v2 [5].

## 2. Related works

Open-Vocabulary Feature Distillation. Recent works have embedded 2D vision-language features into 3D scene representations to enable open-vocabulary 3D understanding. Pioneering efforts on NeRFs such as LERF [15] and OpenNeRF [7] used CLIP [30] embeddings and pixelaligned features, enabling open-vocabulary queries. However, due to the computational needs of NeRF [22], they face scalability and efficiency bottlenecks. Thus, subsequent works have embedded language features into

3DGS [31, 41, 43]. LangSplat [28] employs SAM [16] to extract multi-level CLIP features, then compresses dimensionality with an autoencoder to build a compact yet expressive 3D language field. Feature3DGS [41] uses a convolutional neural network (CNN) to lift feature dimensions. Although both approaches aim to compress the supervision signal, this dimensionality reduction inevitably results in information loss. GOI [29] and CCL-LGS [36] employ a single trainable feature codebook to store language embeddings, with an MLP predicting discrete codebook indices for rasterized 2D feature maps, which compress semantics spatially rather than dimensionally and retain semantic richness. However, as these approaches rely on 2D rendered feature maps for perception, their performance in 3D scene understanding is significantly limited.

Other methods first group 3D Gaussians or points into semantically meaningful clusters, typically corresponding to objects or parts, and then assign a language feature to each cluster as a whole [12, 18, 20, 26, 29, 38]. These methods introduce an explicit discrete grouping step as a form of prior semantic structuring: OpenGaussian [38] performs coarse-to-fine clustering based on spatial proximity followed by feature similarity. SuperGSeg [20] and InstanceGaussian [18] both leverage neural Gaussians to model instance-level features: SuperGSeg groups Gaussians into Super-Gaussians to facilitate language assignment, whereas InstanceGaussian directly assigns fused semantic features to each cluster. VoteSplat [12] and Open-Splat3D [26] mitigate the pixel-level ambiguities of the direct distillation. Then, the resulting cluster graph structures support higher-level reasoning [20, 40], which per-Gaussian features cannot easily enable. However, all these methods rely on feature distillation using per-cluster learnable language embeddings. These approaches are computationally expensive and highly sensitive to noise or outliers in the preprocessed feature maps, since the language features are optimized directly in Euclidean space. As a result, even minor errors in the input features can propagate through the model, leading to inconsistent or inaccurate semantic representations, particularly in complex or cluttered scenes.

Open-Vocabulary Feature Aggregation. Beyond cluster-based language features distillation, recent works adopt more efficient strategies for feature aggregation. For instance, Dr.Splat [13] and Occamâs LGS [3] bypass intermediate 2D supervision and clustering by directly injecting language features into 3D Gaussians, achieving fast, accurate results in a training-free regime. While these direct feature aggregation methods deliver strong runtime efficiency and segmentation accuracy, they indiscriminately propagate 2D features to every Gaussian intersected by each camera ray, disregarding scene geometry and occlusion. As a result, features from foreground objects (e.g., a vase) are erroneously assigned to background elements (e.g., the table or floor). Moreover, existing methods share two critical limitations: (i) they assign equal supervision to all Gaussians along a ray, ignoring each Gaussianâs marginal contribution to the rendered pixel, and (ii) they overlook the view-dependent noise and inconsistency in 2D language features. We address these issues with VALA, a robust and efficient training-free framework that improves openvocabulary grounding through visibility-aware gating (for contribution-aligned supervision) and robust multi-view aggregation.

<!-- image-->  
Figure 2. Overview of VALA. The framework is shown on the left, with the orange and green blocks detailed on the right being our key contributions: the visibility-aware feature lifting (orange, Section 4.1), and the robust multi-view aggregation (green, Section 4.2).

## 3. Preliminaries

We briefly recall 3DGS [14] and how the features are assigned to a 3D Gaussian without iterative training.

3D Gaussian Primitives and Projection. A scene is represented by a set of anisotropic Gaussians $\mathcal { G } = \{ g _ { i } \} _ { i = 1 } ^ { N } ,$ with each Gaussian featured with $g _ { i } ~ = ~ ( \pmb { \mu } _ { i } , \pmb { \Sigma } _ { i } , \mathbf { c } _ { i } , o _ { i } )$ , where $\pmb { \mu } _ { i } \in \mathbb { R } ^ { 3 }$ and $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ are the mean position and covariance matrix $\mathbf { c } _ { i }$ encodes appearance (e.g., RGB or spherical harmonics coefficients), and $o _ { i } \in ( 0 , 1 ]$ is a base opacity.

Images are rasterized by splatting the Gaussians from near to far along the camera ray through pixel $u ,$ followed by front-to-back Î±-blending the Gaussian contributions, as:

$$
\alpha _ { i } ( \mathbf { u } ) = 1 - \exp \bigl ( { o _ { i } \rho _ { i } ( \mathbf { u } ) } \bigr ) ,\tag{1}
$$

$$
T _ { i } ( \mathbf { u } ) = \prod _ { j < i } \bigl ( 1 - \alpha _ { j } ( \mathbf { u } ) \bigr ) ,\tag{2}
$$

$$
\mathbf { C } ( \mathbf { u } ) = \sum _ { i } \alpha _ { i } ( \mathbf { u } ) T _ { i } ( \mathbf { u } ) \mathbf { c } _ { i } ( \mathbf { u } ) ,\tag{3}
$$

where $\rho _ { i } ( \mathbf { u } )$ is the projected 2D Gaussian density in screen space, with projected 2D mean $\tilde { \pmb { \mu } } _ { i }$ and covariance $\tilde { \Sigma } _ { i }$ , and

$$
\begin{array} { r } { \rho _ { i } ( { \mathbf { u } } ) = \exp \Big ( - \frac { 1 } { 2 } ( { \mathbf { u } } - \tilde { \pmb { \mu } } _ { i } ) ^ { \top } \tilde { \Sigma } _ { i } ^ { - 1 } ( { \mathbf { u } } - \tilde { \pmb { \mu } } _ { i } ) \Big ) . } \end{array}\tag{4}
$$

We denote the marginal contribution of $g _ { i }$ to pixel u as

$$
w _ { i } ( { \mathbf { u } } ) \ = \ \alpha _ { i } ( { \mathbf { u } } ) T _ { i } ( { \mathbf { u } } ) .\tag{5}
$$

Language Features Assignment via Direct Aggregation. Recent works [3, 13] proposed to directly assign 2D language features to 3D Gaussians via weighted feature aggregation. To obtain training-free 3D language feature embeddings, Kim et al. [13] pool per-pixel weights $w _ { i } ( I , r )$ , defined as in Eq. (5), using segmentation masks $M _ { j } ( I , r ) ;$

$$
w _ { i j } = \sum _ { I \in \mathcal { T } } \sum _ { r \in \Omega _ { I } } M _ { j } ( I , r ) \cdot w _ { i } ( I , r ) ,\tag{6}
$$

where $w _ { i j }$ associates Gaussian i-mask $j ,$ and $\Omega _ { I }$ is the pixel domain of image I. The final CLIP embedding for each i is a weighted average over the mask-level embeddings $f _ { j } ^ { \mathrm { m a p } }$

$$
f _ { i } = \sum _ { j = 1 } ^ { M } \frac { w _ { i j } } { \sum _ { k = 1 } ^ { M } w _ { i k } } f _ { j } ^ { \operatorname* { m a p } } .\tag{7}
$$

Although this mask-based aggregation is a straightforward way to lift CLIP features into 3D, it has a memory footprint that scales quadratically with the scene complexity. To overcome this limitation, we adopt Occamâs LGS [3]âs probabilistic per-view aggregation strategy as our baseline. [3] avoids explicit mask representations and dense weight storage, maintaining semantic consistency across views. So, the 3D feature $f _ { i }$ for Gaussian i becomes:

$$
f _ { i } = \frac { \sum _ { s \in \mathscr { S } _ { i } } w _ { i } ^ { s } f _ { i } ^ { s } } { \sum _ { s \in \mathscr { S } _ { i } } w _ { i } ^ { s } } ,\tag{8}
$$

where $s _ { i }$ is the views set where Gaussian i is visible, $w _ { i } ^ { s }$ is the marginal contribution of i at its center projection in view s, and $f _ { i } ^ { s }$ is the 2D feature at the corresponding pixel.

## 4. Method

We aim to distill language features into 3DGS under visibility constraints, to get semantically rich and view-consistent 3D embeddings. Existing approaches that indiscriminately assign identical 2D features to all Gaussians along a camera ray, which leads to noisy supervision and cross-view inconsistencies. With VALA, we assign only visible features.

Our pipeline is shown in Figure 2. Built on a direct feature assignment method, VALA has two complementary components to improve the assignment of 2D visionlanguage features to the 3D scene. First, we introduce a visibility-aware attribution mechanism to selectively assign language features to Gaussians based on their relevance in the rendered scene (Section 4.1). Second, we propose a robust cross-view consolidation strategy to aggregate perview features while suppressing inconsistent observations, yielding coherent 3D semantic embeddings (Section 4.2).

## 4.1. Visibility-Aware Feature Lifting

Recent works explored lifting 2D language embeddings into 3D space via differentiable rendering pipelines [3, 13]. However, existing approaches assign the same 2D language feature to all Gaussians intersected by a given pixel ray, regardless of each Gaussianâs actual contribution to the rendered pixel. As illustrated in Figure 3, when an object $O _ { 2 }$ is occluded by another object $O _ { 1 }$ , the 2D language embedding at that pixel primarily represents the semantics of $O _ { 1 }$ Nevertheless, a Gaussian $g _ { 2 }$ belonging to $O _ { 2 }$ may still be incorrectly associated with the language feature of $O _ { 1 }$

This erroneous assignment occurs in both alphablending-based language assignment methods [20, 28] and, more prominently, in direct feature assignment methods [3, 13, 38]. As shown in Figure 3 (bâc), even though the transmittance (Eq. (2)) decreases monotonically along the ray from near to farâresulting in a very small transmittance for $g _ { 2 }$ âits alpha value (Eq. (1)) can remain relatively large in the far region. This, yields a non-negligible compositing weight (Eq. (9)) for $g _ { 2 }$ , which, according to Eq. (7) or Eq. (8), contributes substantially to the final aggregated feature of $g _ { 2 }$ . Such unintended contributions introduce ambiguity into the 3D representation.

Recent works have introduced changes that indirectly affect this assignment. Dr.Splat [13] selects the top-k Gaussians along each ray, but this reduces computational costs rather than ensuring the correct semantic allocation. VoteSplat [12] recognizes that distant Gaussians may suffer from occlusion, but discards the compositing weights altogether and instead averages the features of all intersected Gaussians to generate 3D votes for the clustering step. While they may tangentially bring improvements, they leave unsolved the assignment problem described above and continue to propagate wrong features to background regions.

To overcome this limitation, we introduce a visibilityaware gating mechanism, which selectively supervises only the Gaussians along each ray that contribute to the pixel. By leveraging per-ray visibility weights, our method filters out occluded or low-contribution Gaussians before aggre-

<!-- image-->  
a) a camera ray hits 2 objects

<!-- image-->  
b) Opacity from projected densities

<!-- image-->  
c) Front-to-back transmittance

<!-- image-->  
d) Contribution weights along the ray

<!-- image-->

<!-- image-->  
Figure 3. Visibility-aware gating for semantic assignment $( { \mathrm { S e c } } -$ tion 4.1). Simplified representation of a scene with two objects (a) $O _ { 1 } , O _ { 2 }$ and a camera ray r with Gaussians $g _ { 1 } , g _ { 2 }$ . We compute the opacity (b) and compute the transmittance front-to-back (c). Then we calculate the contribution weights for each ray, thresholding with Ï (d). Instead of propagating the features to all Gaussians as prior works do, our gating only propagates to the visible ones (e). gating the features, ensuring that only geometrically and photometrically relevant points receive semantic supervision. First, we clarify how we compute the per-ray weights.

Ray Notation and Marginal Contributions. Let r denote the camera ray through pixel u. For brevity, we write

$$
\begin{array} { r l r l } & { T _ { i } ( r ) \equiv T _ { i } ( \mathbf { u } ) , } & & { \alpha _ { i } ( r ) \equiv \alpha _ { i } ( \mathbf { u } ) , } \\ & { w _ { i } ( r ) \equiv \alpha _ { i } ( r ) T _ { i } ( r ) . } \end{array}\tag{9}
$$

where $\alpha _ { i } ( \boldsymbol { r } )$ encodes coverage $( i . e .$ , how much $g _ { i }$ overlaps the pixel), $T _ { i } ( r )$ represents transmittance $( i . e . $ , how much light reaches $g _ { i }$ after occlusion by nearer Gaussians), and $w _ { i } ( r )$ measures how strongly $g _ { i }$ influences the rendered sample along r. We name this as the Visibility of a Gaussian from a specific view. Instead of assigning this feature to all Gaussians on the ray r, we use a two-stage visibility-aware gate (VAG). We aggregate the weights into a per-view visibility score

$$
S _ { \mathrm { t o t } } ^ { s } = \sum _ { i , r } w _ { i } ( r ) .\tag{10}
$$

Stage A: Mass Coverage on the Thresholded Set. We sort $\{ w _ { i } ( r ) \} .$ i decreasingly, with the indices as $( 1 ) , \ldots , ( k )$ â¢ We then retain the shortest prefix that accounts for a target fraction $\tau _ { \mathrm { v i e w } } \in [ 0 . 5 , 0 . 7 5 ]$ of the total visibility mass:

$$
k _ { \mathrm { m a s s } } ^ { \star } = \displaystyle \operatorname* { m i n } \Big \{ k : \sum _ { j = 1 } ^ { k } w _ { j } \geq \tau _ { \mathrm { v i e w } } S _ { \mathrm { t o t } } ^ { s } \Big \} .\tag{11}
$$

To suppress numerical noise, we apply a small absolute floor $\tau _ { \mathrm { a b s } }$ and define the candidate set as

$$
\mathcal { G } _ { \mathrm { m a s s } } ^ { s } = \Bigl \{ ( 1 ) , \ldots , \bigl ( k _ { \mathrm { m a s s } } ^ { \star } \bigr ) \Bigr \} \cap \bigl \{ i : w _ { i } \geq \tau _ { \mathrm { a b s } } \bigr \} .\tag{12}
$$

Stage B: Quantile-Constrained Truncation. Let $\tau _ { q } ^ { s } =$ Quantile $ _ { 1 - q } ( \{ w _ { i } \} _ { i } )$ , we define $K _ { q } ^ { s } = | \{ i : w _ { i } \geq \tau _ { q } ^ { s } \} |$ and instead of imposing a separate hard limit, we determine the selection cap directly via the q-quantile as

$$
\begin{array} { r l } & { k _ { \mathrm { k e e p } } ^ { \star } = \operatorname* { m i n } \big ( k _ { \mathrm { m a s s } } ^ { \star } , \ K _ { q } ^ { s } \big ) , } \\ & { \mathcal G _ { \mathrm { k e e p } } ^ { s } = \big \{ ( 1 ) , \dots , ( k _ { \mathrm { k e e p } } ^ { \star } ) \big \} . } \end{array}\tag{13}
$$

Why Mass then Quantile? A fixed quantile alone tightly controls cardinality but ignores how visibility mass is distributed, and under heavy tails may discard essential contributors. Conversely, mass coverage secures a target fraction of visible content but can be liberal when scores are flat. Our two-stage rule reconciles both: Stage A guarantees coverage on the relevant (floored) set, while B imposes a quantile-derived cardinality constraint $K _ { q } ^ { s }$ that stabilizes scale across views. Practically, if $K _ { q } ^ { s } \ge k _ { \mathrm { m a s s } } ^ { \star }$ , we keep the mass-coverage set unchanged; otherwise we truncate it to the top- $K _ { q } ^ { s }$ by $w _ { i } .$ . The gate is thus coverage-faithful and scale-adaptive.

## 4.2. Robust Multi-View Aggregation

SAM+CLIP preprocessing pipelines [28] yield crisp mask boundaries and per-pixel open-vocabulary embeddings, but their semantics are often viewpoint-dependent: changes in viewpoint and occlusion induce noticeable drift across views. To enforce multi-view consistency, several 3DGSbased methods first form 3D-consistent clusters, typically supervised with contrastive signals derived from SAM masks, and then assign a language embedding to each cluster [18, 20, 26, 38]. While this decoupled clustering can improve multi-view semantic consistency, it makes the pipelinesâ training multi-stage, thus prolonging the training time. More critically, because clustering is still driven by noisy, view-dependent 2D cues, it does not correct the root cause, namely, upstream semantic drift, which can bias the clusters and ultimately degrade the accuracy of the final language assignments.

To address this multi-view inconsistency at source, we adopt geometric median [1, 21, 37] to robustly aggregate multi-view features by minimizing the cosine distances in feature space. Unlike aggregation by weighted mean, it dampens view-dependent outliers and semantic drift.

Weighted Euclidean Geometric Median. Using the visibility weights defined in Eq. (9), the (weighted) geometric median for $g _ { i }$ is

$$
\mathbf { z } _ { i } ^ { \star } = \operatorname { a r g m i n } _ { \mathbf { z } \in \mathbb { R } ^ { d } } \sum _ { s } w _ { i } ( r ) \left\| \mathbf { z } - \mathbf { f } _ { i } ^ { s } \right\| .\tag{14}
$$

Cosine-loss Median on the Unit Sphere. ${ \bf f } ( I , { \bf u } )$ are â2-normalized embeddings and thus angular consistency is most relevant. Therefore, we constrain $\mathbf { z } _ { i }$ to the unit sphere $\mathbb { S } ^ { d - 1 }$ and minimize a weighted cosine loss:

$$
\mathbf { z } _ { i } ^ { \star } = \mathrm { a r g m i n } _ { \| \mathbf { z } \| _ { 2 } = 1 } \sum _ { s } w _ { i } ( r ) \big ( 1 - \mathbf { f } _ { i } ^ { s \top } \mathbf { z } \big ) ,\tag{15}
$$

Algorithm 1 Streaming cosine-loss median on $\mathbb { S } ^ { d - 1 }$ (Sec  
tion 4.2).   
Require: Stream $\{ ( \mathbf { f } _ { t } , w _ { i } ^ { t } ) \} _ { t = 1 } ^ { T }$ with $\mathbf { f } _ { t } \in \mathbb { R } ^ { d } , \| \mathbf { f } _ { t } \| _ { 2 } = 1$   
and $w _ { i } ^ { t } > 0$   
1: Initialize $\mathbf { z } _ { i , 0 }  \mathbf { f } _ { 1 } , \quad W _ { i , 0 }  0$   
2: for $t = 1 , \dots , T \mathbf { d o }$   
3: $\mathbf { d } _ { t } \gets \mathbf { f } _ { t } - ( \mathbf { f } _ { t } ^ { \top } \mathbf { z } _ { i , t } ) \mathbf { z } _ { i , t }$ â· tangent direction   
4: $\eta _ { t } \gets \frac { w _ { i } ^ { \tau } } { W _ { i , t } + w _ { i } ^ { t } }$ â· streaming step size   
5: $\mathbf { z } _ { i , t + 1 } \gets \mathrm { N o r m } ( \mathbf { z } _ { i , t } + \eta _ { t } \mathbf { d } _ { t } )$   
6: $W _ { i , t + 1 } \gets W _ { i , t } + w _ { i } ^ { t }$   
7: end for   
8: return $\mathbf { z } _ { i }  \mathbf { z } _ { i , T } , ~ W _ { i }  W _ { i , T }$

where $w _ { i } ( r )$ denotes the visibility weight of Gaussian $g _ { i }$ from view $s ,$ since r represents the view s. The gradient of â(f , z) = 1 â f â¤z on Sdâ1 is $\nabla _ { \mathbf { z } } \ell = - \left\lceil \mathbf { f } - ( \mathbf { f } ^ { \top } \mathbf { z } ) \mathbf { z } \right\rceil$ , the projection of f onto the tangent space at z. Compared to the Euclidean formulation in Eq. (14), this objective directly optimizes angular similarity, circumventing the scale sensitivity of Euclidean distances in high dimensions, where norm variations dominate over angular differences, and empirically leads to more stable 3D semantics (Table 3).

Constant-Memory Streaming Update. While effective, solving Eq. (15) with the classical Weiszfeld algorithm [6] requires repeated full-batch updates over all Gaussian features, which scales linearly with the number of views and becomes computationally prohibitive in practice. To address this, we adopt a constant-memory streaming scheme inspired by online optimization [16]. Specifically, as detailed in Algorithm 1, we maintain only the current estimate $( \mathbf { z } _ { i , t } , W _ { i , t } )$ , where $W _ { i , t }$ is the cumulative visibility weight, and incorporate each new observation $( \mathbf { f } _ { t } , w _ { i } ^ { t } )$ via

$$
\mathbf { z } _ { i , t + 1 } = \mathrm { N o r m } \big ( \mathbf { z } _ { i , t } + \eta _ { t } w _ { i } ^ { t } \left[ \mathbf { f } _ { t } - ( \mathbf { f } _ { t } ^ { \top } \mathbf { z } _ { i , t } ) \mathbf { z } _ { i , t } \right] \big ) ,\tag{16}
$$

$$
\eta _ { t } = \frac { w _ { i } ^ { t } } { W _ { i , t } + w _ { i } ^ { t } } , \qquad W _ { i , t + 1 } = W _ { i , t } + w _ { i } ^ { t } ,\tag{17}
$$

where $\mathrm { N o r m } ( \mathbf { x } ) = \mathbf { x } / \| \mathbf { x } \| _ { 2 }$ projects $\mathbf { z } _ { i , t }$ onto the unit sphere $\mathbb { S } ^ { d - 1 }$ . The update direction $\mathbf { f } _ { t } \mathrm { ~ - ~ } ( \mathbf { f } _ { t } ^ { \top } \mathbf { z } _ { i , t } ) \mathbf { z } _ { i , t }$ lies in the tangent space and increases cosine similarity, while the adaptive step size $\eta _ { t }$ weights each sample according to its visibility. Under standard stochastic approximation assumptions, ${ \bf z } _ { i , t }$ converges to a stationary point of Eq. (15) at rate $\mathcal { O } ( 1 / \sqrt { W _ { i , t } } )$

## 5. Experiments

## 5.1. Experimental setup

Datasets. We evaluate on the two reference datasets for this task: LERF-OVS [28] and ScanNet-v2 [5]. LERF-OVS is derived from the LERF dataset of Kerr et al. [15], where we evaluate open-vocabulary object selection in both 2D and 3D. For the 2D evaluation, we follow the protocol of LERF [15]. For the 3D evaluation, we follow Open-Gaussian [38]. On ScanNet, we evaluate 3D semantic segmentation. Previous evaluation protocols [18, 38] freeze the growth of 3D Gaussians, which degrades photometric fidelity. In contrast, we allow full optimization of the 3D Gaussians, resulting in misalignment between the optimized Gaussians and the ground-truth point cloud. We therefore adapt the evaluation protocol in [13] by propagating pseudo ground-truth labels to the Gaussians. Details are provided in the Appendix B.

<table><tr><td colspan="2"></td><td colspan="2">Mean</td><td colspan="2">Figurines</td><td colspan="2">Ramen</td><td colspan="2">Teatime</td><td colspan="2">Waldo_Kitchen</td></tr><tr><td colspan="2">Method</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td rowspan="9">2ae n</td><td>LERF [15]</td><td>37.4</td><td>73.6</td><td>38.6</td><td>75.0</td><td>28.2</td><td>62.0</td><td>45.0</td><td>84.8</td><td>37.9</td><td>72.7</td></tr><tr><td>LEGaussian [31]</td><td>24.6</td><td>67.4</td><td>23.4</td><td>57.1</td><td>20.2</td><td>69.0</td><td>32.3</td><td>79.7</td><td>22.3</td><td>63.6</td></tr><tr><td>GOI [29]</td><td>42.0</td><td>59.2</td><td>23.9</td><td>44.6</td><td>33.7</td><td>56.3</td><td>55.8</td><td>67.8</td><td>54.5</td><td>68.2</td></tr><tr><td>GAGS [25]</td><td>54.1</td><td>81.7</td><td>53.6</td><td>78.6</td><td>46.8</td><td>69.0</td><td>60.3</td><td>88.1</td><td>55.8</td><td>90.9</td></tr><tr><td>LangSplat [28]</td><td>51.4</td><td>84.3</td><td>44.7</td><td>80.4</td><td>51.2</td><td>73.2</td><td>65.1</td><td>88.1</td><td>44.5</td><td>95.5</td></tr><tr><td>LangSplatV2 [19]</td><td>59.9</td><td>84.1</td><td>56.4</td><td>82.1</td><td>51.8</td><td>74.7</td><td>72.2</td><td>93.2</td><td>59.1</td><td>86.4</td></tr><tr><td>Occam&#x27;s LGS [3]</td><td>61.3</td><td>82.5</td><td>58.6</td><td>80.4</td><td>51.0</td><td>74.7</td><td>70.2</td><td>93.2</td><td>65.3</td><td>81.8</td></tr><tr><td>VALA [ours]</td><td>61.7</td><td>86.4</td><td>59.9</td><td>82.1</td><td>51.5</td><td>75.6</td><td>70.2</td><td>91.5</td><td>65.1</td><td>86.4</td></tr><tr><td>LangSplat [28]</td><td>10.35</td><td>13.64</td><td>7.27</td><td>10.71</td><td>10.05</td><td>9.86</td><td>14.38</td><td>20.34</td><td>9.71</td><td>9.09</td></tr><tr><td rowspan="8">an n</td><td>LEGaussians [31]</td><td>16.21</td><td>23.82</td><td>17.99</td><td>23.21</td><td>15.79</td><td>26.76</td><td>19.27</td><td>27.12</td><td>11.78</td><td>18.18</td></tr><tr><td>OpenGaussian [38]</td><td>38.36</td><td>51.43</td><td>39.29</td><td>55.36</td><td>31.01</td><td>42.25</td><td>60.44</td><td>76.27</td><td>22.70</td><td>31.82</td></tr><tr><td>SuperGSeg [20]</td><td>35.94</td><td>52.02</td><td>43.68</td><td>60.71</td><td>18.07</td><td>23.94</td><td>55.31</td><td>77.97</td><td>26.71</td><td>45.45</td></tr><tr><td>Dr.Splat [13]</td><td>43.29</td><td>64.30</td><td>54.42</td><td>80.36</td><td>24.33</td><td>35.21</td><td>57.35</td><td>77.97</td><td>37.05</td><td>63.64</td></tr><tr><td>InstanceGaussian [18]</td><td>43.87</td><td>61.09</td><td>54.87</td><td>73.21</td><td>25.03</td><td>38.03</td><td>54.13</td><td>69.49</td><td>41.47</td><td>63.64</td></tr><tr><td>CAGS [34]</td><td>50.79</td><td>69.62</td><td>60.85</td><td>82.14</td><td>36.29</td><td>46.48</td><td>68.40</td><td>86.44</td><td>37.62</td><td>63.64</td></tr><tr><td>VoteSplat [12]</td><td>50.10</td><td>67.38</td><td>68.62</td><td>85.71</td><td>39.24</td><td>61.97</td><td>66.71</td><td>88.14</td><td>25.84</td><td>33.68</td></tr><tr><td>Occam&#x27;s LGS [3]</td><td>47.22</td><td>74.84</td><td>52.90</td><td>78.57</td><td>32.01</td><td>54.92</td><td>61.02</td><td>93.22</td><td>42.95</td><td>72.72</td></tr><tr><td>VALA [ours]</td><td></td><td>58.02</td><td>82.85</td><td>60.38</td><td>89.29</td><td>45.41</td><td>67.61</td><td>70.61</td><td>88.14</td><td>55.71</td><td>86.36</td></tr></table>

Table 1. Comparison on LERF-OVS (mIoU / mAcc.). In 3D, results are taken from [12, 13, 20, 34, 38] and otherwise evaluated by us.

Implementation Details. We generate SAM [16] masks at subpart, part, and whole object granularities. We use OpenCLIP ViT-B/16 [30] and the gsplat rasterizer [39]. We apply direct feature aggregation in the 512-dimensional space following [3], combined with our proposed trainingfree method. The entire process requires only 10 seconds to one minute per scene (depending on scene scale), thanks to our effective cross-view feature aggregation and streaming updates at constant memory. For all experiments, we used an NVIDIA RTX 4090 GPU.

## 5.2. Analysis on LeRF-OVS dataset

Table 1 compares ours with state-of-the-art works on LERF-OVS in 2D and 3D. In 2D, per-view segmentation quality projected from 3D is checked, while in 3D, we directly assess multi-view consistent semantic reconstruction.

Quantitatives In 2D. Our method achieves the highest scores on both mIoU and mAcc, slightly surpassing the mIoU of Occamâs LGS [3] and outperforming LangSplatV2 [19]. This improvement is consistent across diverse scenes, particularly in Figurines and Ramen, suggesting that our visibility-aware attribution reduces per-ray semantic noise without sacrificing fine-grained per-view accuracy. While GAGS [25] and LangSplat [28] also deliver competitive 2D scores, their performance drops with complex occlusions (e.g., Ramen for GAGS), indicating that their 2D-driven assignments do not fully mitigate crossview inconsistencies.

Quantitatives In 3D. The advantage of our method becomes more pronounced in 3D, with ours exceeding all baselines by a notable margin. The second best, CAGS [34], is a substantial 7.2 absolute mIoU points behind. The scene-level analysis reveals that our approach leads in Ramen, Teatime, and Waldo Kitchen, and ranks second in Figurines, behind VoteSplat [12] due to its specialized multiview voting. The gains are especially significant in large, cluttered environments (Teatime, Waldo Kitchen), where our contribution-aware aggregation better preserves semantics despite severe occlusions.

The strong 3D consistency of our method contrasts with approaches like LangSplat and LEGaussian [31], whose high 2D accuracy does not translate to 3D performance, likely due to their lack of explicit handling of per-ray contribution and occlusion. Similarly, the post-hoc clustering methods OpenGaussian [38] and SuperGSeg [20] exhibit moderate 3D improvements but remain sensitive to upstream semantic drift, thereby limiting their robustness. Our performance relative to Occamâs LGS (baseline) is noteworthy: while both adopt streaming updates, our visibilityguided feature attribution yields much better performance in 3D, highlighting the effectiveness of improving the semantic assignment at the feature aggregation stage rather than solely relying on memory-efficient training.

<!-- image-->  
Figure 4. Qualitative 3D objects selections on LeRF-OVS [28]. We mark as failed those with low or zero IoU with the ground truth (red).

Qualitatives in 3D. We show visual 3D results in Figure 4. Existing approaches, such as InstanceGaussian [18], frequently fail by retrieving incorrect objects across multiple scenes. This can be attributed to their reliance on appearanceâsemantic joint representations, which struggle to distinguish small objects with visually similar appearances. Clustering-based methods struggle with multiple instances that are closely related. For example, querying for âknifeâ, OpenGaussian [38] and InstanceGaussian [18] detect only one out of five knives, whereas Dr.Splat [13] and Occamâs LGS [3] identify all knives but produce indistinct boundaries. In contrast, ours successfully localizes all knives with accurate and sharp delineations. Our approach also demonstrates robustness on challenging small-object queries, such as âKamabokoâ and âeggâ in the Ramen scene. These targets lie within a heavily cluttered context (a bowl of ramen), making them particularly difficult to isolate. Competing methods [13, 18, 38] fail to recognize these objects, while Occamâs LGS correctly retrieves them but with blurred contours. By comparison, ours produces precise boundaries and accurately captures fine object structures. Similar improvements are observed in the âSpatulaâ query, further illustrating that our visibility-aware gating not only mitigates occlusion effects but also enables the recovery of finegrained details in complex scenes.

<table><tr><td rowspan="2">Method</td><td>19 classes</td><td>15 classes</td><td rowspan="2"></td><td colspan="2">10 classes</td></tr><tr><td>mIoU mAcc</td><td>mIoU mAcc</td><td>mIoU mAcc</td><td></td></tr><tr><td>LangSplat [15]</td><td>2.45 8.59</td><td>3.45</td><td>13.21</td><td>6.48</td><td>21.89</td></tr><tr><td>OpenGaussian [38] 27.73</td><td>42.01</td><td>29.67</td><td>46.15</td><td>39.93</td><td>57.34</td></tr><tr><td>Dr. Splat [13]</td><td>29.31 47.68</td><td>33.25</td><td>54.33</td><td>44.19</td><td>65.19</td></tr><tr><td>Occam&#x27;s LGS [3]</td><td>31.93 48.93</td><td>34.25</td><td>53.7145.16</td><td></td><td>64.39</td></tr><tr><td>VALA [ours]</td><td>32.11 50.05</td><td></td><td>35.10 54.77 46.21</td><td></td><td>65.61</td></tr></table>

Table 2. Open-vocabulary 3D semantic segmentation task on the ScanNet-v2 dataset [5] across different amounts of classes.

## 5.3. 3D Semantic Segmentation on ScanNet

Quantitatives. As reported in Table 2, our method achieves the best performance across all evaluation settings, including the most challenging 19-class scenario. Compared to Occamâs LGS [3], our contribution-aware aggregation is advantageous, demonstrating its ability to handle fine-grained class distributions. While Dr.Splat [13] attains competitive accuracy in reduced-category settings, it lags notably in mIoU, indicating weaker spatial consistency. These results confirm that our method achieves robust and precise 3D segmentation across varying label granularities.

<!-- image-->  
Figure 5. Qualitative results of 3D semantic segmentation with 19 classes on the ScanNet-v2 dataset [5].

Qualitatives. Qualitative comparisons are presented in Figure 5. In the large and complex second room, our method accurately predicts the wall behind the bed (bed in orange), a structure often misclassified by others. In the smaller but more occluded third scene, our method also demonstrates superior 3D segmentation, capturing challenging objects such as the central table more effectively. This ability to recover occluded and fine-scale geometry is particularly beneficial for downstream applications such as 3D object localization. Overall, the qualitative results support the quantitative improvements, highlighting both the robustness and effectiveness of our proposed framework.

## 5.4. Ablation Study

We conduct an ablation study on LeRF-OVS [28], averaging the metrics over all scenes. Table 3 disentangles the contributions of our main components, namely visibility-aware gating and cosine-based geometric median. Starting from the baseline Occamâs LGS [3], replacing the naive weighted mean with our cosine median (b) already improves performance, highlighting the advantage of robust aggregation in the embedding space. Incorporating visibility-aware gating further boosts results (c-d), where mass-coverage plus threshold gating (c) yields the strongest individual gain, while quantile pruning (d) provides complementary benefits. We also observe that our gating alone (f) is less effective compared to gating along with our robust median (VALA), showing that the precise aggregation is critical to fully exploit visibility cues. Lastly, we compare cosine and L1 (g) as median, with the former delivering superior results. Our full model (VALA) achieves the best overall performance, validating that both visibility-aware gating and cosine-based median aggregation are important for an accurate and view-consistent 2D-3D language lifting.

<table><tr><td>Ref.</td><td>Stage A</td><td>Stage B</td><td>Median</td><td>mIoU</td><td>mAcc</td></tr><tr><td>O.LGS [3]</td><td></td><td></td><td></td><td>47.22</td><td>74.86</td></tr><tr><td>(b)</td><td></td><td></td><td>cosine</td><td>49.03</td><td>80.08</td></tr><tr><td>(c)</td><td>â</td><td></td><td>cosine</td><td>57.24</td><td>81.25</td></tr><tr><td>(d)</td><td></td><td>â</td><td>cosine</td><td>55.21</td><td>80.37</td></tr><tr><td>VALA</td><td>â</td><td>â</td><td>cosine</td><td>58.02</td><td>82.85</td></tr><tr><td>(f)</td><td>V</td><td>â</td><td></td><td>52.29</td><td>76.17</td></tr><tr><td>(g)</td><td>v</td><td>â</td><td>L1</td><td>56.03</td><td>82.42</td></tr></table>

Table 3. Ablation on LeRF-OVS. First row is Occamâs LGS [3], i.e., our baseline. Stages from Section 4.1, Median from 4.2. All rows share the same data, rasterizer, and hyperparameters.

We refer to the Supplementary Material for additional details and results.

## 6. Conclusion

We introduced VALA, an efficient and effective method to address two fundamental problems in the feature aggregation of open-vocabulary recognition in 3DGS, namely (i) the propagation of 2D features to all Gaussians along a camera ray, and (ii) the multi-view inconsistency of semantic features. VALA tackles (i) with a visibility-aware distillation of language features based on a two-stage gating mechanism, and (ii) with a cosine variant of the geometric median, updating the features via streaming to keep the memory footprint low. These innovations ensure more appropriate features are assigned to the 3D Gaussians, ultimately leading to superior performance in open-vocabulary segmentation. Remarkably, the proposed VALA achieves state-of-the-art performance on 2D and 3D tasks on the reference datasets LeRF-OVS and ScanNet-v2.

## References

[1] Amir Beck and Shoham Sabach. Weiszfeldâs method: Old and new results. Optimization Letters, 9(1):1â18, 2015. See also preprint/PDF for historical notes. 5

[2] Cesar Cadena, Luca Carlone, Henry Carrillo, Yasir Latif, Davide Scaramuzza, Jose Neira, Ian Reid, and John J.Â´ Leonard. Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age. IEEE Transactions on Robotics, 32(6):1309â1332, 2016. 1

[3] Jiahuan Cheng, Jan-Nico Zaech, Luc Van Gool, and Danda Pani Paudel. Occamâs LGS: A simple approach for language Gaussian splatting. arXiv preprint arXiv:2412.01807, 2024. 1, 2, 3, 4, 6, 7, 8

[4] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3075â 3084, 2019. 1

[5] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3D reconstructions of indoor scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017. 2, 5, 7, 8, 1, 3

[6] Ulrich Eckhardt. Weberâs problem and weiszfeldâs algorithm in general spaces. Mathematical Programming, 18(1):186â 196, 1980. 5

[7] Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, and Federico Tombari. OpenNeRF: Open set 3D neural scene segmentation with pixel-wise features and rendered novel views. In The Twelfth International Conference on Learning Representations, 2024. 1, 2

[8] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2443â 2451, 2012. 1

[9] Xiuye Gu, Yen-Chun Kuo, Yin Cui, Zecheng Sun, David Zhang, and Steven C. H. Hoi. Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921, 2021. 1

[10] Shahram Izadi, David Kim, Otmar Hilliges, David Molyneaux, Richard Newcombe, Pushmeet Kohli, Jamie Shotton, Steve Hodges, Daniel Freeman, Andrew Davison, and Andrew Fitzgibbon. Kinectfusion: Real-time 3d reconstruction and interaction using a moving depth camera. In ACM Symposium on User Interface Software and Technology (UIST), pages 559â568, 2011. 1

[11] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Thomas Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning (ICML), pages 4904â4916, 2021. 1

[12] Minchao Jiang, Shunyu Jia, Jiaming Gu, Xiaoyuan Lu, Guangming Zhu, Anqi Dong, and Liang Zhang. Votesplat: Hough voting gaussian splatting for 3d scene understanding. arXiv preprint arXiv:2506.22799, 2025. 1, 2, 4, 6

[13] Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. Splat: Directly referring 3D Gaussian Splatting via direct language embedding registration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 1, 2, 3, 4, 6, 7

[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3D Gaussian Splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4):139â1, 2023. 1, 3

[15] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19729â19739, 2023. 1, 2, 5, 6, 7

[16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015â4026, 2023. 1, 2, 5, 6

[17] Georg Klein and David Murray. Parallel tracking and mapping for small ar workspaces. In IEEE and ACM International Symposium on Mixed and Augmented Reality, pages 225â234, 2007. 1

[18] Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao Zhang, Ronggang Wang, and Jian Zhang. InstanceGaussian: Appearance-semantic joint gaussian representation for 3D instance-level perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14078â14088, 2025. 1, 2, 5, 6, 7

[19] Wanhua Li, Yujie Zhao, Minghan Qin, Yang Liu, Yuanhao Cai, Chuang Gan, and Hanspeter Pfister. Langsplatv2: Highdimensional 3d language gaussian splatting with 450+ fps. arXiv preprint arXiv:2507.07136, 2025. 6

[20] Siyun Liang, Sen Wang, Kunyi Li, Michael Niemeyer, Stefano Gasperini, Nassir Navab, and Federico Tombari. Supergseg: Open-vocabulary 3d segmentation with structured super-gaussians. arXiv preprint arXiv:2412.10231, 2024. 2, 4, 5, 6

[21] Horst Martini, Konrad J. Swanepoel, and Gunter Weiss. OnÂ¨ torricelliâs geometrical solution to a problem of fermat. Elemente der Mathematik, 50(2):93â96, 1995. 5

[22] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In Proceedings of the European Conference on Computer Vision, pages 405â421, 2020. 1, 2

[23] Raul Mur-Artal and Juan D. Tardos. Orb-slam2: An open- Â´ source slam system for monocular, stereo, and rgb-d cameras. IEEE Transactions on Robotics, 33(5):1255â1262, 2017. 1

[24] Songyou Peng, Kyle Genova, Chiyu Max Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, Matthias NieÃner, and Sida Peng Liu. Openscene: 3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1786â1796, 2023. 1

[25] Yuning Peng, Haiping Wang, Yuan Liu, Chenglu Wen, Zhen Dong, and Bisheng Yang. Gags: Granularity-aware feature distillation for language gaussian splatting. arXiv preprint arXiv:2412.13654, 2024. 6

[26] Jens Piekenbrinck, Christian Schmidt, Alexander Hermans, Narunas Vaskevicius, Timm Linder, and Bastian Leibe. Opensplat3d: Open-vocabulary 3d instance segmentation using gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5246â5255, 2025. 2, 5

[27] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. PointNet: Deep learning on point sets for 3D classification and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 652â660, 2017. 1

[28] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3D language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051â20060, 2024. 1, 2, 4, 5, 6, 7, 8

[29] Yansong Qu, Shaohui Dai, Xinyang Li, Jianghang Lin, Liujuan Cao, Shengchuan Zhang, and Rongrong Ji. GOI: Find 3D gaussians of interest with an optimizable openvocabulary semantic-space hyperplane. In Proceedings of the ACM International Conference on Multimedia, pages 5328â5337, 2024. 2, 6

[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, pages 8748â8763. PMLR, 2021. 1, 2, 6

[31] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3D gaussians for openvocabulary scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5333â5343, 2024. 2, 6

[32] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Sergio Casas, Wenjie Lin, Abbas Sadat, Balakrishnan Varadarajan, Jonathon Shlens, Zhifeng Chen, Alan Yuille, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2443â2451, 2020. 1

[33] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 4

[34] Wei Sun, Yanzhao Zhou, Jianbin Jiao, and Yuan Li. Cags: Open-vocabulary 3d scene understanding with contextaware gaussian splatting. arXiv preprint arXiv:2504.11893, 2025. 1, 6

[35] Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, FrancÂ¸ois Goulette, and Leonidas J. Guibas. Kpconv: Flexible and deformable convolution for point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6410â 6419, 2019. 1

[36] Lei Tian, Xiaomin Li, Liqian Ma, Hefei Huang, Zirui Zheng, Hao Yin, Taiqing Li, Huchuan Lu, and Xu Jia. Ccl-lgs: Contrastive codebook learning for 3d language gaussian splatting. arXiv preprint arXiv:2505.20469, 2025. 2

[37] Endre Weiszfeld and Frank Plastria. On the point for which the sum of the distances to n given points is minimum. Annals of Operations Research, 167(1):7â41, 2008. 5

[38] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, et al. OpenGaussian: Towards point-level 3D gaussian-based open vocabulary understanding. Advances in Neural Information Processing Systems, 37:19114â19138, 2024. 1, 2, 4, 5, 6, 7

[39] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1â17, 2025. 6, 1

[40] Chenlu Zhan, Yufei Zhang, Gaoang Wang, and Hongwei Wang. Hi-lsplat: Hierarchical 3d language gaussian splatting. arXiv preprint arXiv:2506.06822, 2025. 2

[41] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3DGS: Supercharging 3D gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676â21685, 2024. 2

[42] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp KrahenbÂ¨ uhl, and Ishan Misra. Detecting twenty-thousandÂ¨ classes using image-level supervision. In European Conference on Computer Vision, pages 350â368, 2022. 1

[43] Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan Di, and Mingyang Li. Fmgs: Foundation model embedded 3D gaussian splatting for holistic 3D scene understanding. International Journal of Computer Vision, pages 1â17, 2024. 2

# Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting

Supplementary Material

In this supplementary material, we provide additional details omitted from the main manuscript. Sec. A describes the implementation details and the 3D tasks under evaluation. Sec. B outlines the experimental setup and the 3D semantic segmentation evaluation protocol on 3D Gaussian Splatting. Sec. C further presents a robustness study, where we stress-test our method under corrupted SAM masks to assess performance degradation in noisy segmentation scenarios. while Sec. D presents qualitative results, annotation analyses, and city-scale evaluations. Finally, Sec. E discusses limitations and future directions.

## A. Implementation Details

Our method operates in two stages. In the pre-training stage, we apply the ViT-H variant of SAM [16] to segment each image. Multi-level language feature maps are then extracted with OpenCLIP ViT-B/16 [30], from which we derive per-patch language embeddings. In parallel, we optimize the 3D Gaussian Splatting parameters [14] using the standard training pipeline with the gsplat rasterizer [39], running 30k iterations. Unlike the original rasterizer, gsplat natively supports rendering high-dimensional Gaussian attributes, which enables evaluation on 2D open-vocabulary tasks.

In the subsequent forward-rendering stage, we adopt the feature aggregation strategy of Occamâs LGS [3]. For each Gaussian within the view frustum, we compute its center-projected pixel location and extract the corresponding 2D language feature $f _ { i } ^ { s }$ . Simultaneously, we record its marginal contribution $w _ { i } ( r )$ as defined in Eq. (9), and retain the most visible Gaussians following the gating strategy in Sec. 4.1. The selected Gaussians are then robustly aligned with multi-view features through our streaming aggregation in cosine space, described in Sec. 4.2.

This entire process completes within 10 seconds to one minute per scene (depending on scene scale) without memory overflow. All experiments are conducted on an NVIDIA RTX 4090 GPU.

## B. Evaluation Protocols

We only compare results following the same evaluation protocol and re-evaluate those prior works that followed other protocols.

Datasets We evaluate our method on two datasets: LERF-OVS [28] and ScanNet [5]. LERF-OVS consists of four scenes (teatime, waldo kitchen, figurines, ramen), each annotated with pixel-wise semantic masks and paired with short text queries. In this dataset, we evaluate openvocabulary object selection in both 2D and 3D settings. To further evaluate 3D semantic segmentation, we adopt a Gaussian-based evaluation protocol on ScanNet, a largescale RGB-D dataset for indoor scene understanding. Each ScanNet sequence is reconstructed into a textured 3D mesh with globally aligned camera poses and semantic annotations. We select eight representative scenes covering diverse indoor environments, including living rooms, bathrooms, kitchens, bedrooms, and meeting rooms.

2D and 3D Evaluation on the LERF-OVS Dataset. For the 2D evaluation, we follow the protocol of LERF [15]: 512-dimensional feature maps are rendered, and a relevancy map is computed with respect to the CLIP-embedded text query. The relevancy map is then thresholded at 0.5 to obtain the predicted binary mask. For the 3D evaluation, we adopt the protocol of OpenGaussian [38], where the relevancy score between each 3D Gaussianâs language embedding and the text query embedding is computed and thresholded at 0.6. The alpha values of the selected Gaussians are subsequently projected onto the image plane to generate the predicted mask. In both cases, the predicted masks are compared against the GT annotations of the LERF-OVS dataset.

3D Semantic Segmentation on the ScanNet-v2 Dataset. Previous works on 3D semantic segmentation [18, 38] typically freeze the input point cloud (derived from ground-truth annotations) during 3D Gaussian Splatting training to cope with the absence of GT labels as the point clouds evolve. However, this strategy degrades the 2D rendering quality of 3DGS. We instead propagate ground-truth (GT) labels from the annotated point cloud to the Gaussians, thereby obtaining pseudo-GT labels at each Gaussianâs 3D mean. Following OpenGaussian [38], we evaluate on subsets of 19, 15, and 10 of the 40 most common classes. For each class, we encode the text label using CLIP [30] to obtain a 512-dimensional embedding, and compute its cosine similarity with the registered language features of each Gaussian. Each Gaussian is then assigned to the class with the highest similarity score. Performance is measured in terms of mIoU and mAcc against the pseudo-GT Gaussian point cloud.

Pseudo-Gaussian Labeling. Given optimized Gaussians $\Theta = \{ \theta _ { i } \} _ { i = { \bar { \iota } } } ^ { N }$ with center $\mu _ { i } ,$ , scale $s _ { i } = ( s _ { i x } , s _ { i y } , s _ { i z } )$ , rotation $R _ { i }$ (hence $\Sigma _ { i } = R _ { i }$ diag( $( s _ { i } ^ { 2 } ) R _ { i } ^ { \top } )$ , and opacity $\alpha _ { i }$ , and a labeled point cloud $\{ ( p _ { k } , s _ { p _ { k } } ) \} _ { k = 1 } ^ { Q }$ , we assign a semantic label to each Gaussian by respecting the true 3DGS geometry and the compositing kernel. In contrast to prior protocols, which (i) maximize the sum of Mahalanobis distances over class points to assign a single label, and (ii) require dense all-pairs computations, our approach assigns semantic labels by respecting the true 3DGS geometry and properties. Specifically, we evaluate the density contribution of a point p to the Gaussian $\mu _ { i } { : }$

$$
\begin{array} { r } { w _ { i } ( p ) = \exp \left( - \frac { 1 } { 2 } d _ { i } ^ { 2 } ( p ) \right) , } \end{array}\tag{18}
$$

where $d _ { i } ^ { 2 } ( p )$ denotes the squared Mahalanobis distance.

Since boundary Gaussians may be partially transparent or occupy negligible volume, we further modulate the votes with a per-Gaussian significance term:

$$
\gamma _ { i } = \alpha _ { i } s _ { i x } s _ { i y } s _ { i z } , \qquad w _ { i } ( p )  \gamma _ { i } w _ { i } ( p ) .\tag{19}
$$

This ensures consistency with the volume-aware IoU metric, which weights Gaussians by both opacity and ellipsoid volume.

Finally, instead of constructing an $N \times Q$ all-pairs distance matrix, we build a per-Gaussian candidate set $K _ { i }$ via spatial culling with an adaptive radius

$$
r a d i u s _ { i } = \tau \cdot \operatorname* { m a x } ( s _ { i } ) ,
$$

with a top-k fallback to handle sparse neighborhoods. We then compute $d _ { i } ^ { 2 } ( \cdot )$ only for $p _ { k } \ \in \ K _ { i }$ , processing Gaussians in GPU-friendly chunks. This reduces the complexity from $O ( N Q )$ to $\begin{array} { r } { \ O ( \sum _ { i } | K _ { i } | ) } \end{array}$ and the memory footprint from $O ( N Q )$ to $O ( | K | )$ ), while retaining only geometrically plausible candidates under each anisotropic ellipsoid. The generated Gaussian point clouds with pseudo GT labels are illustrated in Figure 5 and Figure 7 (the second column from left to right).

## C. Robustness Evaluation with Perturbed Masks

To evaluate robustness against segmentation noise, we perform an experiment on the teatime scene of LERF-OVS by simulating errors in SAM masks.

Stress-Testing Robustness with Corrupted Masks. To stress-test robustness against imperfect proposals, we corrupt each SAM mask by a per-mask morphological perturbation applied at the original image resolution. Let $m _ { k } \in$ $\{ 0 , 1 \} ^ { H \times \mathbf { \bar { W } } }$ denote the binary mask of instance $k ,$ and let

$$
B _ { r } = \{ ( x , y ) \in \mathbb { Z } ^ { 2 } : x ^ { 2 } + y ^ { 2 } \leq r \}
$$

be a disk-shaped structuring element of radius r pixels, where $r \in \{ 5 , 1 0 , 1 5 , 2 0 , 2 5 , 3 0 \}$ , to simulate different perturbation levels.

<!-- image-->  
Figure 6. Robustness under mask boundary corruptions. mIoU/mAcc (%) are shown on the left y-axis; Disp (lower is better) on the right y-axis. We vary the erosion/dilation radius r (pixels). VALA degrades more slowly than Occamâs and its ablation without gating (VALA w/o G), while achieving lower Disp across severities.

For every mask we draw an independent sign variable $\sigma _ { k } \in \{ - 1 , + 1 \}$ with equal probability $P ( \sigma _ { k } = + 1 ) =$ $P ( \sigma _ { k } = - 1 ) = 0 . 5$ . The corrupted mask $\tilde { m } _ { k }$ is then

$$
\tilde { m } _ { k } = \left\{ \begin{array} { l l } { m _ { k } \ominus B _ { r } , } & { \mathrm { i f } \sigma _ { k } = - 1 \quad ( \mathrm { e r o s i o n } ) , } \\ { m _ { k } \oplus B _ { r } , } & { \mathrm { i f } \sigma _ { k } = + 1 \quad ( \mathrm { d i l a t i o n } ) , } \end{array} \right.
$$

where $\ominus$ and â denote morphological erosion and dilation, respectively.

To prevent degenerate outcomes on small objects, we enforce a non-vanishing guard: if erosion yields an empty or tiny region (area below a minimum threshold $\tau _ { \mathrm { m i n } }$ pixels), we fallback to dilation and set $\tilde { m } _ { k } \gets m _ { k } \oplus B _ { r }$ . After corruption, we recompute tight bounding boxes from $\tilde { m } _ { k }$ and propagate them to downstream steps (e.g., cropping and 224 Ã 224 resizing for CLIP feature extraction).

This perturbation stochastically shifts boundaries outward/inward by approximately r pixels while preserving instance identity, thereby simulating over- and undersegmentation errors commonly observed in practice.

Evaluation Protocal. To assess the robustness of the proposed streaming median in the cosine space, we compare three variants: the baseline Occamâs LGS [3], our full model incorporating both visibility-aware gating and robust multi-view aggregation (VALA), and an ablation variant with only the robust multi-view aggregation module (VALA w/o G). In addition to the standard mIoU and mAcc metrics for evaluating the final 3D object selection task, we further introduce the dispersion score, which specifically quantifies the robustness of assigned language features under multiview variations. Given a Gaussian $g _ { i }$ with observed unit features $f _ { i } ^ { s } \in \mathbb { S } ^ { d - 1 }$ , the per-Gaussian dispersion is com-

<!-- image-->  
Figure 7. More qualitative results of 3D semantic segmentation on the ScanNet-v2 dataset [5],

puted as

$$
{ \mathrm { D i s p } } _ { i } = { \frac { 1 } { | S _ { i } | } } \sum _ { ( i , s ) \in S _ { i } } { \Bigl ( } 1 - \langle f _ { i } ^ { s } , z _ { i } ^ { * } \rangle { \Bigr ) } ,\tag{20}
$$

At the scene level, we report the average:

$$
{ \mathrm { D i s p } } _ { \mathrm { s c e n e } } = { \frac { 1 } { | I | } } \sum _ { i \in I } { \mathrm { D i s p } } _ { i } ,\tag{21}
$$

This metric captures the average misalignment between observed features and the aggregated Gaussian feature, where lower values indicate higher consistency.

Results Analysis. The results are presented in Figure 6. As the corruption radius increases from r = 5 to 30 px, all methods show a monotonic decline in mIoU/mAcc and a corresponding rise in Disp, confirming that boundary noise simultaneously degrades semantic accuracy and cross-view consistency. Importantly, the deterioration is substantially slower for our methods than for Occamâs LGS, as reflected by the smaller slope of Disp. In terms of accuracy, VALA achieves the strongest results: at r = 5, it surpasses Occamâs by +12.8 mIoU and +17.0 mAcc, with substantial gains still observed at $r = 1 0$ . Meanwhile, the Disp values reveal a complementary trendâalthough VALAâs Disp is marginally higher than Occamâs at $r = 5 ,$ it drops below Occamâs from r = 10 onwards. This demonstrates that the combination of visibility-aware gating and robust aggregation not only improves accuracy but also enhances multiview consistency in the practically relevant regime of mild mask noise.

When boundary damage becomes severe, however, the picture changes. VALA (w/o G) overtakes the full VALA model in accuracy (e.g., at r = 30, achieving 9.95/15.25 vs. 6.75/1.69 in mIoU/mAcc) and consistently yields the lowest Disp across all radii. This suggests that the fixed gating threshold becomes overly conservative under extreme corruption, discarding too many observations and leaving insufficient evidence for many Gaussians. In contrast, the cosine-median aggregator alone remains robust, preserving both accuracy and consistency in this challenging setting. Overall, these results highlight a clear regime split: visibility-aware gating combined with a cosine median provides the strongest accuracy and consistency under realistic (mild to moderate) noise. However, under extreme boundary corruption, robust aggregation is the key factor, as overly strict gating thresholds reduce coverage and performance.

## D. Additional Results

In this section, we present additional results on the ScanNet dataset and, more importantly, demonstrate that our algorithm can be applied to real-world outdoor datasets, achieving superior open-vocabulary semantic segmentation in autonomous driving scenarios.

More Qualitative Results on the ScanNet Dataset. We provide additional qualitative results on three bedrooms with varying levels of complexity and clutter. Across all scenes, competing methods struggle to correctly recognize the bed (highlighted in orange); the occluded portions near the wall are consistently misclassified as adjacent categories, such as the wall or floor. This issue persists in the third scene, where the bed is fragmented into multiple categories. In contrast, our method preserves the bed as a coherent instance, owing to the proposed gating module that explicitly handles low-visibility Gaussians.

<!-- image-->  
Figure 8. Qualitative results on the Waymo Open Dataset [33]. The colored regions indicate the activation maps corresponding to the given text prompts.

Experiments on the Waymo Open Dataset. To further validate our algorithmâs generalization capability in realworld outdoor environments, we conduct experiments on the Waymo Open Dataset [33]. This dataset is a largescale, high-quality autonomous driving benchmark that provides synchronized LiDAR and multi-camera data collected across diverse urban and suburban geographies, along with comprehensive 2D/3D annotations and tracking identifiers. For evaluation, we select a sequence captured in a residential neighborhood that contains rich semantic elements, such as vehicles, vegetation, street infrastructure, and buildings. We focus on five of the most common outdoor categories, e.g. tree, trash bin, car, streetlight, and house, as well as one tail category, stair. The qualitative results in Figure 8 demonstrate that our method achieves precise open-vocabulary 3D semantic segmentation on outdoor data. Both small-scale objects (e.g., trash bins and streetlights) and large-scale objects (e.g., trees, cars, and houses) are not only correctly retrieved but also segmented with sharp boundaries, reflecting the accurate registration of language features on the 3D Gaussian Splatting representation. Notably, our method remains robust under occlusion owing to the proposed visibility-aware gating module. For example, correctly delineating trees behind metallic structures or houses partially obscured by vegetation.

These findings emphasize the robustness and versatility of our method when transferred from indoor (ScanNet) to challenging outdoor driving scenarios, underscoring its strong potential for real-world autonomous driving applications. A supplementary video is included to further demonstrate the effectiveness and the multi-view consistency of our method.

## E. Limitations

While our approach demonstrates strong performance across multiple tasks, including 2D and 3D object selection as well as 3D semantic segmentation, and exhibits notable generalization to cross-domain settings such as outdoor datasets, certain limitations remain. To assess robustness against noisy SAM masks, we conducted stress tests with multi-scale morphological perturbations. The results show that our visibility-aware gating achieves superior mIoU and mAcc under moderate noise, while the proposed cosine median maintains low dispersion even under severe corruption, indicating the effectiveness of our robust feature aggregator. However, our current framework relies on a fixed threshold to prune Gaussians, which can become overly conservative under extreme noise, resulting in degraded multi-view consistency. Moreover, our method is specifically designed for static scenes and does not naturally extend to dynamic environments. Future work will therefore focus on developing adaptive, scene-aware thresholds and extending our framework to handle dynamic scenes.