# HyGE-Occ: Hybrid View-Transformation with 3D Gaussian and Edge Priors for 3D Panoptic Occupancy Prediction

Jong Wook Kim1 Wonseok Roh1â  Ha Dam Baek1 Pilhyeon Lee3 Jonghyun Choi2 Sangpil Kim1\*

1Korea University 2Hyundai Motor Company 3Inha University

## Abstract

3D Panoptic Occupancy Prediction aims to reconstruct a dense volumetric scene map by predicting the semantic class and instance identity of every occupied region in 3D space. Achieving such fine-grained 3D understanding requires precise geometric reasoning and spatially consistent scene representation across complex environments. However, existing approaches often struggle to maintain precise geometry and capture the precise spatial range of 3D instances critical for robust panoptic separation. To overcome these limitations, we introduce HyGE-Occ, a novel framework that leverages a hybrid view-transformation branch with 3D Gaussian and edge priors to enhance both geometric consistency and boundary awareness in 3D panoptic occupancy prediction. HyGE-Occ employs a hybrid view-transformation branch that fuses a continuous Gaussian-based depth representation with a discretized depth-bin formulation, producing BEV features with improved geometric consistency and structural coherence. In parallel, we extract edge maps from BEV features and use them as auxiliary information to learn edge cues. In our extensive experiments on the Occ3D-nuScenes dataset, HyGE-Occ outperforms existing work, demonstrating superior 3D geometric reasoning.

## 1. Introduction

3D Panoptic Occupancy Prediction (3DPOP) extends 2D perception tasks, such as segmentation [7, 21, 28] and depth estimation [8, 44, 45], into a unified volumetric reasoning framework that predicts both semantic class and instance identity for every occupied region in 3D space. Unlike 2D perception [12], which operates on visible pixels, 3DPOP reconstructs the complete spatial layout of the scene, reasoning about both visible and occluded structures. Such holistic scene understanding is essential for autonomous driving. Estimating the full 3D occupancy is vital for safe motion planning and long-term scene prediction.

<!-- image-->  
Figure 1. HyGE-Occ fuses continuous and discrete depth representations into a hybrid BEV space and integrates an edge prior to enhance boundary cues, yielding robust 3DPOP.

A key line of work in this field focuses on feature lifting from camera views to 3D space using Birdâs-Eye-View (BEV) representations, avoiding the computational burden of full 3D convolutions. The dominant paradigm, Lift-Splat-Shoot (LSS) [36], performs discretized unprojection along predefined depth bins and projects image features into BEV space, providing a simple yet effective mechanism. The BEV representation captures 3D features more efficiently than heavy 3D convolutions. Despite their advantages, discrete depth prediction methods still suffer from depth quantization artifacts, producing coarse geometry with ambiguous object edges in the 3D map [29].

Recent approaches introduce continuous or probabilistic representations, such as Gaussian Splatting [20] and triplane features [9, 16]. These representations model 3D geometry as smooth, differentiable functions rather than discrete fixed bins, enabling them to encode structural features in a more flexible manner. In the case of Gaussian splatting, the method provides a continuous, uncertainty-aware 3D representation by predicting the mean and variance of perpixel depth distributions, effectively characterizing both the expected depth and its surrounding ambiguity. These properties enable smooth geometric interpolation and mitigate quantization artifacts inherent to discretized depth bins. The resulting representation better captures fine-grained spatial variation and provides stable depth reasoning under occlusion or in ambiguous visual conditions, making it particularly effective for dense, complex scenes.

Motivated by this, we explore how continuous 3D

Gaussian-based formulations and discretized depth reasoning can complement each other. While discretized approaches like LSS preserve precise spatial localization, continuous Gaussian representations enhance geometric smoothness and robustness to depth uncertainty. We leverage the complementary strengths of both through a unified hybrid design that balances structural precision and geometric consistency, forming a foundation of our proposed framework for improved geometric reasoning.

However, improved geometric reasoning alone does not guarantee accurate semantic or instance-level occupancy predictions. We observe that many 3D occupancy prediction frameworks [27, 49] struggle to delineate semantic and instance boundaries with precision. These models misrepresent the spatial extent of objects and ambiguously localize their boundaries, leading to mislabeled edge regions or incomplete instance separation. Such errors stem from the lack of explicit supervision of inter-object boundaries, which causes boundaries that appear diffuse or uncertain.

To address these challenges, we propose HyGE-Occ, a hybrid view-transformation method with 3D Gaussian and Edge Priors that enhances both geometric consistency and boundary precision in 3D Panoptic occupancy prediction. HyGE-Occ introduces a Hybrid View-Transformation Branch that integrates dense Gaussian splatting priors with discretized LSS features through alpha blending, combining the geometric consistency of continuous representations with the local spatial precision of discretized depth reasoning. In addition, we introduce an Edge Prediction Module that explicitly enhances boundary awareness within BEV feature representations. Edge maps are generated from ground-truth semantic labels and serve as auxiliary supervision to strengthen boundary cues in the BEV space. This process sharpens structural details before volumetric decoding, enabling the network to produce more accurate and consistent boundaries for both semantic and instance-level occupancy predictions. Both modules are architecturally decoupled, allowing seamless integration into diverse BEVbased occupancy models without architectural redesign.

We validate HyGE-Occ on the Occ3D-nuScenes [41] dataset, where it achieves state-of-the-art performance, demonstrating superior geometric fidelity and sharper panoptic delineation. When we integrate our two modules into existing frameworks such as Panoptic-FlashOcc [49] and ALOcc [5], we surpass their original results, confirming their effectiveness as a scalable enhancement for both panoptic and semantic occupancy prediction models.

In summary, our contributions are threefold:

â¢ We propose a Hybrid View-Transformation Branch that fuses continuous and discrete BEV features for improved geometric fidelity.

â¢ We design an Edge Prediction Module that introduces explicit edge supervision into volumetric decoding, leading to sharper, more precise panoptic delineation.

â¢ We extensively evaluate our method on the Occ3DnuScenes dataset, where quantitative and qualitative results demonstrate state-of-the-art performance, confirming the effectiveness of our approach.

## 2. Related works

3D Occupancy Prediction. Early studies in 3D occupancy prediction primarily focus on estimating voxel-wise occupancy states by discretizing the scene into regular 3D grids. Occ3D [41] established a unified benchmark across Waymo [40] and nuScenes [3], enabling consistent evaluation across sensor modalities and model architectures. Building upon this foundation, research has progressed from LiDAR-centric pipelines [32, 46, 47, 51, 52] toward camera-only methods [4, 13, 16, 30, 33] that offer greater scalability and deployment efficiency. PanoOcc [43] is the first to introduce a unified voxel-based framework for camera-only 3D panoptic occupancy prediction, integrating semantic segmentation and instance-level detection into a single end-to-end model. It learns a coarse-to-fine voxel representation via multi-view and multi-frame fusion, achieving dense, panoptic-level 3D scene understanding without LiDAR input. SparseOcc [27] further advanced the task by introducing ray-based evaluation metrics, RayIoU and RayPQ, which measure geometric and panoptic quality along camera rays and provide a more faithful assessment of long-range scene consistency compared to voxelonly metrics. Following these developments, Panoptic-FlashOcc [49] presents a lightweight camera-only framework that operates directly on BEV features, providing an efficient and scalable 3D panoptic occupancy prediction.

View Transformation for 3DOP. 3D occupancy prediction inherently requires transforming 2D image features into 3D space to infer volumetric scene structure. Numerous strategies have been explored for this 2D-to-3D transformation, ranging from explicit depth-based lifting methods [36] to implicit field modeling approaches [16, 53], and more recently, NeRF [1, 17, 34, 42, 50] or Gaussian Splatting [2, 10, 18, 55]-based scene representations for occupancy prediction. Each formulation offers a distinct trade-off between geometric accuracy and spatial consistency. Among these, Lift-Splat-Shoot (LSS) [36] and its extensions [5, 15, 23, 48, 49] have been widely adopted due to their simplicity and scalability. These methods discretize depth into uniform bins and aggregate voxel-level features to construct BEV representations. However, such discretization inevitably introduces quantization artifacts and limits geometric continuity along the depth axis. To overcome these limitations, GaussianLSS [29] has proposed 3D Gaussian-based uncertainty-aware depth modeling to better handle depth ambiguity. This suggests that the two representations can complement each other when combined, leveraging their respective strengths in spatial localization and geometric continuity. Inspired by this insight, we propose a hybrid framework that integrates discrete with continuous depth reasoning, enhancing geometric consistency and structural coherence for 3DPOP.

<!-- image-->  
Figure 2. Overview of our proposed HyGE-Occ. Our model takes multi-view images as input and first extracts image features through a shared image backbone. The Hybrid View-Transformation Branch fuses a discretized and continuous Gaussian-based depth representation to form hybrid BEV features that combine spatial precision and geometric continuity. The resulting BEV features are further refined by a Edge Prediction Module, which predicts BEV-level edge maps optimised with pseudo edge labels computed from the semantic ground truth. These boundary-enhanced BEV representations are then decoded by the panoptic head, consisting of semantic and instance center branches, to produce the final 3D panoptic occupancy prediction.

Edge Supervision. Precise boundary information provides critical cues for distinguishing semantic regions and capturing fine-grained structural details in scene understanding. In image semantic segmentation [6, 22, 31, 39], edge-aware learning has been extensively explored to enhance boundary precision and mitigate over-smoothing near object edges. Recently, this concept has been extended to 3D perception [11, 38, 54], where boundary priors have also shown effectiveness in improving geometric fidelity and spatial consistency. These findings indicate that boundary-aware cues can enhance spatial consistency in both semantic and panoptic scene reconstruction. Building upon this insight, we introduce an edge head that provides explicit boundaryaware guidance to the model for refined boundaries and improved panoptic consistency.

## 3. Method

## 3.1. Model overview

HyGE-Occ improves BEV-based 3D panoptic occupancy prediction by introducing two novel modules, a Hybrid View-Transformation Branch and an Edge Prediction Module. Given N = 6 multi-view images $\{ I _ { i } \} _ { i = 1 } ^ { N }$ , a 2D backbone extracts feature maps $\mathbf { F } _ { i } \in \mathbf { \bar { \mathbb { R } } } ^ { C \times H \times \overline { { W } } }$ . These features are then lifted into BEV space through our hybrid view-transformation process, which combines discretized $\mathbf { B } ^ { d }$ and continuous $\mathbf { B } ^ { g }$ depth features to generate hybrid BEV features $\mathbf { B } ^ { h }$ that better preserve geometric structure and spatial coherence. To further enhance structural precision, HyGE-Occ employs an edge prediction module trained with pseudo-edge supervision derived from semantic annotations. This module encourages clearer boundarysensitive signals within BEV features, resulting in providing guidance to volumetric decoder towards sharper spatial delineation. The final hybrid BEV representation is fed into a volumetric decoder to produce dense voxel-wise semantic and instance predictions. Hybrid depth fusion and boundary-aware refinement enhance the reliability and coherence of 3D panoptic occupancy predictions The model overall architecture is illustrated in Figure 2.

## 3.2. Discretized Depth Unprojection

For discretized depth modeling, we divide the depth range $[ d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } ]$ into B uniform intervals following the previous work [36], forming a discrete depth set D as:

$$
D = \left\{ d _ { i } = d _ { \operatorname* { m i n } } + i \cdot \frac { d _ { \operatorname* { m a x } } - d _ { \operatorname* { m i n } } } { B } \right\} _ { i = 0 } ^ { B - 1 }\tag{1}
$$

Each image pixel gets a predicted feature vector $\mathbf { c } \in \mathbb { R } ^ { C }$ and a normalized depth probability distribution $\pmb { \alpha } \in \Delta ^ { B }$ , where $\Delta ^ { B }$ is a B-dimensional probability simplex. For a specific depth bin $d _ { i } .$ , the frustum feature $\mathbf { c } _ { d _ { i } } \in \mathbb { R } ^ { C }$ is defined by weighting the context feature by its probability: $\mathbf { c } _ { d _ { i } } = \alpha _ { i } \mathbf { c }$ . These depth-weighted features are then unprojected into 3D frustum space and later aggregated into BEV coordinates. The discretized depth branch supervises voxel occupancy through a binary cross-entropy loss as follows:

$$
\mathcal { L } _ { \mathrm { L S S } } = - { \bf O } _ { g t } \log ( { \bf O } _ { \mathrm { L S S } } ) - ( 1 - { \bf O } _ { g t } ) \log ( 1 - { \bf O } _ { \mathrm { L S S } } ) ,\tag{2}
$$

where $\mathbf { O } _ { g t }$ and $\mathbf { O } _ { \mathrm { L S S } }$ denote the ground-truth and predicted occupancy volumes, respectively. Although the discrete unprojection mechanism efficiently lifts 2D features into 3D, it introduces unstable depth distribution and incomplete spatial coverage [29] due to its fixed bin resolution. In addition, the softmax-based depth probabilities may become unstable, leading to inconsistent geometry in neighboring depths.

<!-- image-->  
Figure 3. Hybrid View-Transformation Branch. The proposed module integrates both discrete and continuous depth representation through a blending module to form hybrid BEV features that combine geometric continuity with spatial precision.

## 3.3. Continuous Gaussian-based Unprojection

To overcome the limitations of discrete binning, we adopt a continuous depth unprojection branch inspired by the 3D Gaussian primitives [20, 29]. Instead of representing depth as a hard depth bin assignment, each pixel predicts a continuous depth distribution parameterized by a mean $\mu$ and variance $\sigma ^ { \bar { 2 } }$ . This introduces an uncertainty-aware spatial range: $( \mu - k \sigma , \mu + k \sigma )$ , where k is a tolerance factor. This formulation mitigates depth quantization by allowing depth to vary smoothly within the predicted distribution, improving robustness under ambiguous conditions such as occlusion or far viewing distances. Our use of Gaussian primitives is motivated by two key properties observed in 3D Gaussian Splatting, smooth spatial support and uncertaintyaware depth modeling. We leverage these properties to enrich the lifted 3D features, soften the influence of incorrect depth estimates, and implicitly capture object extents within the uncertainty region.

To integrate these benefits, we construct Gaussian features by mapping each pixelâs predicted depth distribution to a 3D anisotropic Gaussian centered at Âµ with covariance $\Sigma = \mathrm { d i a g } ( \sigma _ { x } ^ { 2 } , \dot { \sigma _ { y } ^ { 2 } } , \sigma _ { z } ^ { 2 } )$ and opacity $\alpha \in [ 0 , 1 ]$ . Each Gaussian thus defines a volumetric kernel:

$$
G _ { i } ( x ) = \alpha _ { i } \exp \Biggl ( - \frac { 1 } { 2 } ( x - \mu _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( x - \mu _ { i } ) \Biggr ) ,\tag{3}
$$

where $\alpha _ { i }$ modulates the visibility or transparency of the Gaussian primitive. All Gaussians are then splat-rendered into the BEV plane through differentiable accumulation:

$$
F _ { \mathrm { g a u s s } } ( x ) = \sum _ { i } w _ { i } \cdot G _ { i } ( x ) ,\tag{4}
$$

where $w _ { i }$ denotes the learned feature weight derived from the backbone. The resulting splatted representation encodes depth as a smooth spatial field, providing a continuous geometric signal that complements the discrete LSS features within our hybrid design.

In addition to occupancy estimation, the continuous unprojection branch also predicts auxiliary centerness and offset maps. These maps guide the reconstruction of object geometry in the BEV space. The centerness map cË provides spatial weighting that emphasizes object centers and suppresses noisy regions. The offset map oË encodes positional shifts between projected features and their groundtruth centers $c .$ These additional predictions enable the Gaussian-based representation to better capture instance geometry and improve the localization accuracy of reconstructed structures. The continuous depth representation method is optimized with a multi-task loss that jointly supervises segmentation, centerness, and offset prediction. The loss is formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { s e g } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { c e n t e r } _ { - } g } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { o f f s e t } _ { - } g } , } \end{array}\tag{5}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } ,$ , and $\lambda _ { 3 }$ are weighting coefficients that balance each component. The segmentation loss $( \mathcal { L } _ { \mathrm { s e g } } )$ is a focal loss [26] to handle class imbalance:

$$
\mathcal { L } _ { \mathrm { s e g } } = - \alpha _ { t } ( 1 - p _ { t } ) ^ { \gamma } \log ( p _ { t } ) ,\tag{6}
$$

where $p _ { t }$ is the predicted probability for the true class, $\alpha _ { t }$ is a balancing factor, and $\gamma$ is a focusing parameter.

The centerness loss employs an L1 objective, and the offset regression is trained with an L2 loss,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c e n t e r } _ { - } g } = \| \hat { c } - c \| , \qquad \mathcal { L } _ { \mathrm { o f f s e t } _ { - } g } = \| \hat { o } - o \| ^ { 2 } , } \end{array}\tag{7}
$$

where cË and oË are predicted centerness and offset values, and c and o are their respective ground truths.

## 3.4. Hybrid View-Transformation Branch

While the Gaussian formulation provides continuous, uncertainty-aware estimation across depth, the LSS branch preserves local contrast and details through explicit binwise formation. To leverage the complementary properties, as shown in Figure 3, we fuse BEV features from both branches, with a simple yet effective Î±-blending strategy. This hybrid representation captures Gaussian continuity and LSS spatial precision, producing stable, structurally coherent BEV embeddings. For each view i, we obtain the following BEV features as:

$$
\mathbf { B } _ { i } ^ { d } = { \mathcal { U } } _ { \mathrm { L S S } } ( \mathbf { F } _ { i } ) , \qquad \mathbf { B } _ { i } ^ { g } = { \mathcal { U } } _ { \mathrm { G } } ( \mathbf { F } _ { i } ) ,\tag{8}
$$

where $\mathbf { B } _ { i } ^ { d }$ and $\mathbf { B } _ { i } ^ { g }$ denote BEV features from discretized $\boldsymbol { \mathcal { U } } _ { \mathrm { L S S } }$ and continuous $\mathcal { U } _ { \mathrm { G } }$ projections, respectively. Then they are Î±-blended into the hybrid BEV representation $\mathbf { B } _ { i } ^ { h }$

$$
\mathbf { B } _ { i } ^ { h } = \alpha \mathbf { B } _ { i } ^ { g } + \mathbf { B } _ { i } ^ { d } , \quad \alpha \in [ 0 , 1 ] .\tag{9}
$$

All hybrid features from N views are finally fused into $\mathbf { B } _ { \mathrm { a g g } }$ via a standard BEV aggregation Fuse:

$$
{ \bf B } _ { \mathrm { a g g } } = \mathrm { F u s e } \big ( \{ { \bf B } _ { i } ^ { h } \} _ { i = 1 } ^ { N } \big ) .\tag{10}
$$

The obtained geometry-aware BEV embedding serves as input to the volumetric decoder.

Algorithm 1 Edge Prediction Module   
Input: Semantic ground truth $\mathbf { Y } _ { q t } \in \mathbb { R } ^ { H \times W }$   
BEV feature map $\mathbf { B } _ { \mathrm { b e v } } \in \mathbb { R } ^ { C \times H \times W }$   
Output: Edge loss $\mathcal { L } _ { \mathrm { e d g e } }$   
Ground-truth edge extraction   
1: $\mathbf { E } _ { g t }  \sqrt { ( S _ { x } * \mathbf { Y } _ { g t } ) ^ { 2 } + ( S _ { y } * \mathbf { Y } _ { g t } ) ^ { 2 } }$ â· Pseudo Edge label   
BEV edge prediction   
2: $\mathbf { E } _ { { p r e d } }  \mathcal { F } ( \mathbf { B } _ { \mathrm { b e v } } )$ â· Predict edge probability map   
Edge loss calculation   
3: $\mathcal { L } _ { \mathrm { e d g e } }  \mathrm { B C E } ( \mathbf { E } _ { p r e d } , \mathbf { E } _ { g t } )$   
4: return $\mathcal { L } _ { \mathrm { e d g e } }$

## 3.5. Edge Prediction Module

Recent occupancy prediction frameworks primarily optimize volumetric or BEV losses that encourage accurate occupancy predictions, but often overlook explicit boundary reasoning. As a result, they tend to produce blurry or inconsistent object borders, especially where semantic classes or instance identities intersect. To refine structural precision, we introduce the edge prediction module that explicitly encourages BEV features with structural cues and provides boundary-aware bev feature to the volumetric decoder. The pseudo-code showing the steps of boundary supervision by the edge prediction module is presented in Algorithm 1. The edge head ${ \mathcal F } ,$ , a mlp-based projection with activation function Ï, operates on intermediate BEV features to predict a BEV-level edge probability map ${ \bf { E } } _ { p r e d }$

$$
\mathbf { E } _ { p r e d } = \mathcal { F } ( \mathbf { B } _ { b e v } ) , \mathcal { F } = W \sigma ( W \mathbf { x } ) ,\tag{11}
$$

which strengthens structural discontinuities in the reconstructed scene. Ground-truth edges $\mathbf { E } _ { g t }$ are derived from panoptic annotations via Sobel filtering [19] as:

$$
{ \bf E } _ { g t } = \sqrt { ( S _ { x } * { \bf Y } _ { g t } ) ^ { 2 } + ( S _ { y } * { \bf Y } _ { g t } ) ^ { 2 } } ,\tag{12}
$$

where â denotes convolution, $S _ { x }$ and $S _ { y }$ are Sobel kernels, and $\mathrm { \bf Y } _ { g t }$ is the semantic ground truth. The edge head is optimized with a binary cross-entropy loss as follows:

$$
\mathcal { L } _ { \mathrm { e d g e } } = \mathrm { B C E } ( \mathbf { E } _ { p r e d } , \mathbf { E } _ { g t } ) ,\tag{13}
$$

providing auxiliary boundary supervision that guides the volumetric decoder toward sharper contours between adjacent semantic regions and more consistent instance delineation. By explicitly encouraging sharper structural boundaries, the proposed edge head improves instance separation and enhances overall panoptic delineation while introducing negligible computational overhead.

## 3.6. Panoptic Occupancy Prediction Head

Following the convention [49], HyGE-Occ predicts dense voxel-wise semantic and instance-aware occupancy volumes from the hybrid BEV feature representation. The prediction head consists of three branches that infer semantic occupancy, instance center heatmaps, and offset vectors, which jointly enable complete 3D panoptic reconstruction. Semantic Occupancy. The semantic branch estimates pervoxel categorical probabilities $\hat { S } ~ \in ~ \mathbb { R } ^ { H \times W \times D \times C }$ , where C denotes the number of semantic classes. This branch uses a lightweight 3D convolutional decoder that aggregates multi-scale BEV features from the hybrid GaussianâLSS encoder. The semantic loss is computed as the standard cross-entropy objective:

$$
\mathcal { L } _ { \mathrm { s e m } } = - \sum _ { x , y , z } \log \hat { S } _ { x , y , z } ^ { ( C _ { g t } ) } ,\tag{14}
$$

where $c _ { g t }$ is the ground-truth class label for voxel $( x , y , z )$ Instance Center and Offset Prediction. To separate instances within the same semantic category, HyGE-Occ predicts an instance center heatmap CË and a per-voxel offset vector $\hat { O } \in \mathbb { R } ^ { 3 }$ . The center heatmap encodes the likelihood of each voxel being close to an instance centroid, while the offset vector points from each voxel to its corresponding center. These values are supervised by an L1 loss:

$$
\mathcal { L } _ { \mathrm { c e n t e r } } = \Vert \hat { C } - C _ { g t } \Vert , \qquad \mathcal { L } _ { \mathrm { o f f s e t } } = \Vert \hat { O } - O _ { g t } \Vert ,\tag{15}
$$

where $C _ { g t }$ and $O _ { g t }$ are the ground-truth center and offset fields derived from 3D instance annotations.

The overall training objective jointly optimizes semantic and instance-level predictions:

$$
\mathcal { L } _ { \mathrm { o c c } } = \lambda _ { \mathrm { s e m } } \mathcal { L } _ { \mathrm { s e m } } + \lambda _ { \mathrm { c e n t e r } } \mathcal { L } _ { \mathrm { c e n t e r } } + \lambda _ { \mathrm { o f f s e t } } \mathcal { L } _ { \mathrm { o f f s e t } } .\tag{16}
$$

This unified formulation allows HyGE-Occ to achieve both high-fidelity geometric reconstruction and sharp panoptic delineation, while maintaining full compatibility with existing occupancy frameworks.

## 3.7. Training Objective

Our framework is trained end-to-end with four complementary loss components: the original discretized LSS loss $( \mathcal { L } _ { \mathrm { L S S } } )$ , the Gaussian loss $( \mathcal { L } _ { \mathrm { G } } )$ , the panoptic occupancy loss $( \mathcal { L } _ { \mathrm { o c c } } )$ , and the edge loss $( \mathcal { L } _ { \mathrm { e d g e } } )$ . Each term ensures that the hybrid representation maintains both spatial coverage and boundary precision. The overall training objective is as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { L S S } } \mathcal { L } _ { \mathrm { L S S } } + \lambda _ { \mathrm { G } } \mathcal { L } _ { \mathrm { G } } + \lambda _ { \mathrm { e d g e } } \mathcal { L } _ { \mathrm { e d g e } } + \mathcal { L } _ { \mathrm { o c c } } ,\tag{17}
$$

where $\lambda _ { \mathrm { L S S } } , \lambda _ { \mathrm { G } } ,$ , and $\lambda _ { \mathrm { e d g e } }$ control the relative contributions of each auxiliary component. All modules are trained jointly, enabling the hybrid view-transformation branch to learn complementary geometric cues while the edge supervision refines boundary precision.

<table><tr><td>Method</td><td>Backbone</td><td>Input Size</td><td>Vis. Mask</td><td> $\mathbf { R a y I o U }$ </td><td> $\mathbf { R a y I o U } _ { 1 m }$ </td><td> $\mathbf { R a y I o U } _ { 2 m }$ </td><td> $\mathbf { R a y I o U } _ { 4 m }$ </td><td>mIoU</td><td>FPS</td></tr><tr><td>OccFormer [53]</td><td>ResNet-101</td><td>928Ã 1600</td><td>â</td><td></td><td></td><td></td><td></td><td>21.9</td><td></td></tr><tr><td>CTT-0c[41]</td><td>ResNet-101</td><td>928Ã1600</td><td>â</td><td>-</td><td></td><td></td><td></td><td>28.5</td><td> </td></tr><tr><td>BEVFormer [25]</td><td>ResNet-101</td><td>1600Ã900</td><td>X</td><td>33.7</td><td></td><td></td><td>-</td><td>23.7</td><td>2.4</td></tr><tr><td>FB-Occ (16f) [24]</td><td>ResNet-50</td><td>704256</td><td>X</td><td>35.6</td><td>-</td><td>-</td><td>-</td><td>27.9</td><td>9.1</td></tr><tr><td>BEVDetOcc (2f) [14]</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>29.6</td><td>23.6</td><td>30.0</td><td>35.1</td><td>-</td><td>5.5</td></tr><tr><td>SparseOcc (8f) [27]</td><td>ResNet-50</td><td>704Ã26</td><td>X</td><td>34.0</td><td>28.0</td><td>34.7</td><td>39.4</td><td>30.6</td><td>22.6</td></tr><tr><td>Panoptic-FlashOcc-tiny (1f) [49]</td><td>ResNet-50</td><td>704Ã256</td><td>Ã</td><td>34.8</td><td>29.1</td><td>35.7</td><td>39.7</td><td>29.1</td><td>45.8</td></tr><tr><td>HyGE-Occ-tiny (1f) (Ours)</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>36.3</td><td>30.8</td><td>37.2</td><td>41.1</td><td>29.4</td><td>37.7</td></tr><tr><td>Panoptic-FlashOcc (1f)</td><td>ResNet-50</td><td>704Ã256</td><td>Ã</td><td>35.2</td><td>29.4</td><td>36.0</td><td>40.1</td><td>29.4</td><td>41.9</td></tr><tr><td>HyGE-Occ (1f) (Ours)</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>36.8</td><td>31.2</td><td>37.5</td><td>41.6</td><td>29.6</td><td>33.0</td></tr><tr><td>Panoptic-FlashOcc (2f)</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>36.8</td><td>31.2</td><td>37.6</td><td>41.5</td><td>30.3</td><td>49.7</td></tr><tr><td>HyGE-Occ (2f) (Ours)</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>37.8</td><td>32.2</td><td>38.6</td><td>42.6</td><td>30.4</td><td>39.2</td></tr><tr><td>Panoptic-FlashOcc (8f)</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>38.5</td><td>32.8</td><td>39.3</td><td>43.4</td><td>31.6</td><td>48.1</td></tr><tr><td>HyGE-Occ (8f) (Ours)</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>39.9</td><td>34.5</td><td>40.7</td><td>44.5</td><td>32.0</td><td>36.8</td></tr><tr><td>ALOcc-2D-mini (16f) [5]</td><td>ResNet-50</td><td>704Ã256</td><td>Ã</td><td>39.3</td><td>32.9</td><td>40.1</td><td>44.8</td><td>33.4</td><td>23.6</td></tr><tr><td>+HyGE-Occ</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>40.2</td><td>33.9</td><td>40.9</td><td>45.6</td><td>33.9</td><td>22.8</td></tr><tr><td>ALOcc-2D (16f)</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>43.0</td><td>37.1</td><td>43.8</td><td>48.2</td><td>37.4</td><td>9.8</td></tr><tr><td>+HyGE-Occ</td><td>ResNet-50</td><td>704Ã256</td><td>X</td><td>43.2</td><td>37.3</td><td>44.0</td><td>48.3</td><td>37.1</td><td>9.5</td></tr></table>

Table 1. Comparison with occupancy and panoptic occupancy prediction models on the Occ3D-nuScenes validation set. HyGE-Occ achieves the best overall performance across RayIoU and mIoU metrics while maintaining competitive efficiency, demonstrating its effectiveness for accurate 3D occupancy prediction. Vis. Mask refers to camera mask

<table><tr><td>Method</td><td>RayPQ</td><td> $\mathbf { R a y P Q } _ { 1 m }$ </td><td> $\mathbf { R a y P Q } _ { 2 m }$ </td><td> $\mathbf { R a y P Q } _ { 4 m }$ </td></tr><tr><td>SparseOcc (8f) [27]</td><td>14.1</td><td>10.2</td><td>14.5</td><td>17.6</td></tr><tr><td>Panoptic-FlashOcc-tiny (1f) [49]</td><td>12.9</td><td>8.8</td><td>13.4</td><td>16.5</td></tr><tr><td>HyGE-Occ-tiny (1f)</td><td>14.5</td><td>10.5</td><td>14.9</td><td>18.1</td></tr><tr><td>Panoptic-FlashOcc (1f)</td><td>13.2</td><td>9.2</td><td>13.5</td><td>16.8</td></tr><tr><td>HyGE-Occ (1f)</td><td>15.1</td><td>11.1</td><td>15.5</td><td>18.9</td></tr><tr><td>Panoptic-FlashOcc (2f)</td><td>14.5</td><td>10.6</td><td>15.0</td><td>18.0</td></tr><tr><td>HyGE-Occ (2f)</td><td>15.8</td><td>11.7</td><td>16.1</td><td>19.5</td></tr><tr><td>Panoptic-FlashOcc (8f)</td><td>16.0</td><td>11.9</td><td>16.3</td><td>19.7</td></tr><tr><td>HyGE-Occ (8f)</td><td>17.5</td><td>13.1</td><td>17.9</td><td>21.4</td></tr></table>

Table 2. Panoptic quality measured on the Occ3D-nuScenes validation set. HyGE-Occ consistently outperforms prior methods across all RayPQ metrics, demonstrating improved instance coherence and overall panoptic quality in 3D scene understanding.

## 4. Experiments

## 4.1. Experimental Setup

Dataset. We use the Occ3d-nuScenes [41] dataset for all experiments, which provides 700 training, 150 validation and 150 test scenes with multimodal inputs from LiDAR and RGB sensors. The occupancy space spans from -40m to 40m horizontally and from -1 to 5.4m vertically, where each voxel is a cube with 0.4m sides and represents occupancy labels for 17 distinct classes.

Metric. Following previous works, we use mIoU, RayIoU, and RayPQ metrics for model evaluation. The mIoU measures voxel-wise semantic accuracy by averaging the intersection-over-union (IoU) across all semantic categories. Additionally, RayIoU extends mIoU to a ray-based formulation by casting query rays into the predicted 3D volume and evaluating geometric consistency along the viewing direction within depth thresholds (1m, 2m, 4m). Furthermore, RayPQ builds upon the Panoptic Quality (PQ) metric, jointly assessing semantic and instance-level consistency along rays, thus providing a comprehensive evaluation for panoptic 3D occupancy prediction. Note that detailed descriptions of implementation details are provided in the supplementary materials.

## 4.2. Comparison with SOTA Methods

Quantitative Results. We first quantitatively compare our proposed framework with recent state-of-the-art camerabased occupancy prediction methods on the Occ3DnuScenes validation set. While existing approaches have advanced the field through improved geometric reasoning and BEV-based spatial aggregation, most remain limited by discrete depth formulations or insufficient boundary modeling, which can lead to coarse geometry and blurred instance boundaries as seen in Figure 4. Our hybrid design, combining continuous Gaussian reasoning with discretized LSS features, addresses these issues by enhancing geometric consistency and preserving fine structural details. In addition, the edge head explicitly guides the network to recover sharper object contours and more coherent instance separation within the 3D occupancy space.

As shown in Tables 1 and 2, our framework consistently outperforms previous methods in both semantic and panoptic occupancy benchmarks, achieving improvements in geometric fidelity and panoptic quality. Compared to panoptic models such as Panoptic-FlashOcc and SparseOcc, HyGE-Occ demonstrates superior performance across all major metrics. Also, when we applied our method to a semantic model, ALOcc, improvements were shown, validating the effectiveness of the proposed hybrid view-transformation branch and edge head design. These results highlight HyGE-Occ as a new strong baseline for camera-based 3D occupancy prediction, achieving robust and accurate 3D scene understanding in complex environments.

<!-- image-->

<!-- image-->  
Semantic_GT

Panoptic-FlashOcc  
<!-- image-->  
Camera Input

HyGE-Occ  
<!-- image-->

<!-- image-->

<!-- image-->  
Panoptic-FlashOcc  
HyGE-Occ  
Panoptic-FlashOcc

HyGE-Occ  
Figure 4. Qualitative comparison on the Occ3D-nuScenes validation set. Compared to the baseline Panoptic-FlashOcc, the proposed HyGE-Occ produces more accurate and coherent panoptic occupancy predictions. Our method better delineates semantic and instance boundaries, particularly in regions with dense object interactions and occlusions.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Hybrid</td><td rowspan=1 colspan=1>Edge</td><td rowspan=1 colspan=1>RayIoU mIoUFPS</td></tr><tr><td rowspan=3 colspan=1>HyGE-Occ-tiny (1f)</td><td rowspan=3 colspan=1>Ã&gt;Ã&gt;</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>34.8   29.1 45.8</td></tr><tr><td rowspan=2 colspan=1>ÃÃ&gt;&gt;</td><td rowspan=1 colspan=1>35.3   29.3 38.0</td></tr><tr><td rowspan=1 colspan=1>35.4   29.3 45.736.3   29.4 37.7</td></tr><tr><td rowspan=3 colspan=1>ALOcc-2D-mini (16f) [5]+ HyGE-Occ</td><td rowspan=3 colspan=1>Ã&gt;Xâ</td><td rowspan=2 colspan=1>ÃâX</td><td rowspan=1 colspan=1>39.3   33.4 23.6</td></tr><tr><td rowspan=1 colspan=1>39.7   33.6 22.939.7  33.5 23.5</td></tr><tr><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>40.2   33.9 22.8</td></tr></table>

Table 3. Ablation study to investigate the effect of our two modules. The effectiveness of each module in HyGE-Occ is evaluated by selectively enabling the hybrid view-transformation branch and the edge head. Both modules contribute to higher RayIoU and mIoU, with the best performance achieved when combined.

Qualitative Results. We provide qualitative comparisons in Figure. 4 to further demonstrate the effectiveness of our proposed framework. Compared to the Panoptic-FlashOcc model, HyGE-Occ produces more precise object boundaries and more coherent semantic occupancy maps. Also our model shows better performance in distinguishing between different instances. The edge head sharpens inter-instance borders, reducing label bleeding and improving panoptic separation. The hybrid view-transformation branch enhances geometric continuity and spatial coherence by alleviating discretization artifacts commonly observed in depth-based unprojection. This continuous and uncertainty-aware reasoning results in smoother scene geometry, fewer fragmented regions, and more stable semantic predictions across complex environments. Together, these visual improvements confirm that our framework effectively enhances both geometric fidelity and instancelevel delineation within the 3D occupancy space.

<!-- image-->  
LSS

<!-- image-->  
Gaussian LSS

<!-- image-->  
HyGE-Occ

Figure 5. Visualization of BEV features. Comparison between BEV features generated by the discretized LSS, continuous GaussianLSS, and our proposed hybrid representation.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Gating</td><td rowspan=1 colspan=1>Alpha-Blending</td></tr><tr><td rowspan=1 colspan=1>RayIoU / mIoU</td><td rowspan=1 colspan=1>36.3 / 29.1</td><td rowspan=1 colspan=1>36.3 / 29.4</td></tr></table>

Table 4. Ablation study on feature fusion. Comparison between gating and Î±-blending strategies for merging discrete and continuous BEV features. Both achieve comparable results, indicating that simple Î±-blending provides sufficient fusion effectiveness.

<table><tr><td>Method</td><td>0.2</td><td>0.4</td><td>0.6</td><td>0.8</td><td>1.0</td></tr><tr><td>HyGE-Occ-tiny (1f)</td><td>36.2</td><td>36.2</td><td>36.3</td><td>36.2</td><td>36.1</td></tr><tr><td>ALOcc-2D-mini (16f) + HyGE-Occ</td><td>39.5</td><td>40.0</td><td>40.2</td><td>39.7</td><td>39.6</td></tr></table>

Table 5. Ablation study on the Î±-blending coefficient for hybrid feature fusion. The RayIoU remains stable across different Î± values, indicating that HyGE-Occ is robust to the fusion weight.

## 4.3. Ablation Studies

Effect of Individual Modules. Table 3 presents quantitative ablation results on the Occ3D-nuScenes validation set using Panoptic-FlashOcc (1f) and ALOcc-2D-mini as baselines. The hybrid branch consistently improves geometric prediction quality by integrating continuous and discrete depth reasoning, while the edge prior further enhances local boundary precision and instance separation. When the two modules are combined, the model achieves the best overall performance, demonstrating their complementary natureâwhere the hybrid branch strengthens global structural consistency and the edge prior refines fine-grained contours and object boundaries across varied scene conditions.

Effect of the Hybrid View-Transformation Branch. To evaluate the impact of the proposed hybrid viewtransformation design, we compare BEV representations obtained from LSS, GaussianLSS, and our hybrid branch. As shown in Figure 5, the difference between LSS and GaussianLSS is clearly visible: LSS produces discretized, discontinuous BEV features due to its bin-based depth reasoning, leading to aliasing artifacts around object boundaries. In contrast, GaussianLSS models depth as a continuous distribution, producing smoother and geometrically coherent spatial features. By blending the two through our hybrid view-transformation branch, HyGE-Occ achieves both geometric stability and fine-grained spatial fidelity, resulting in a more consistent BEV representation.

<!-- image-->  
Figure 6. Comparison of edge kernels. Edge maps are generated using Sobel, Prewitt, and Laplacian operators. Each are compared with the semantic ground truth. The Sobel operator produces sharper and more consistent boundaries with less noise, making it the most effective choice for our edge supervision.

We further analyze different feature fusion strategies in Table 4. In addition to simple alpha-blending, we evaluate a gating function that predicts a spatially varying fusion weight map, allowing the network to adaptively emphasize either the continuous or discretized features at each BEV location. Although this gating mechanism provides greater flexibility, it introduces additional parameters and increases training sensitivity. In contrast, alpha-blending yields slightly higher overall accuracy while maintaining a simpler and more stable formulation. This indicates that a simple linear fusion is sufficient to integrate the complementary cues from continuous and discretized depth reasoning without requiring complex adaptive weighting schemes.

We also examine the effect of the blending coefficient Î± in Table 5. Across both models, performance remains stable for a wide range of Î± values, demonstrating that the hybrid representation is robust to the choice of blending ratio. The performance gradually improves as Î± grows from 0.2 to 0.6, indicating that incorporating stronger Gaussian cues enhances geometric consistency and feature smoothness. Beyond $\alpha = 0 . 6 $ , the gain saturates or slightly declines, suggesting that excessive reliance on Gaussian features may suppress the fine-grained local details captured by the discretized LSS branch. Overall, Î± = 0.6 provides the best trade-off between geometric fidelity and spatial precision across different frameworks, offering a stable default choice for integrating the two complementary representations. This suggests that the hybrid formulation effectively leverages complementary geometric cues from both representations without requiring fine-grained tuning.

Effect of the Edge Prior Module. To examine the impact of the proposed edge prior, we conduct ablative experiments on the choice of edge extraction kernels and their corresponding hyperparameters. The pseudo edge labels are generated by applying standard gradient-based operators to ground-truth semantic maps, as visualized in Figure 6. Among the tested filters, the Sobel operator provides the most stable and well-localized boundaries, which translate into high-quality pseudo edge supervision. Table 6 quantitatively confirms this observation. Models trained with Sobel-based pseudo edge labels achieve the best performance (36.3 / 29.4) compared to Prewitt [37] (36.3 / 29.1) and Laplacian [35] (36.0 / 29.2), demonstrating that accurate edge localization is critical for effective boundary guidance. We further analyze the effect of kernel size $S _ { x , y }$ and weighting factor $\lambda _ { \mathrm { e d g e } }$ in Table 7. The results indicate that a 3 Ã 3 kernel with $\lambda _ { \mathrm { e d g e } } = 4 . 0$ yields the best trade-off between boundary sharpness and global structural consistency. Larger kernels (5 Ã 5 or 7 Ã 7) introduce redundant edge noise and slightly degrade performance, while smaller or overly strong weights reduce overall stability. Consequently, we adopt the Sobel operator with a $3 \times 3$ kernel and $\lambda _ { \mathrm { e d g e } } = 4 . 0$ as the default experiment configuration.

<table><tr><td>Method</td><td>Sobel</td><td>Prewitt</td><td>Laplacian</td></tr><tr><td>RayIoU / mIoU</td><td>36.3 / 29.4</td><td>36.3 / 29.1</td><td>36.0 / 29.2</td></tr></table>

Table 6. Ablation study on edge detection kernels. Different edge detection kernels are compared for generating pseudo edge labels. The Sobel operator yields the best performance, indicating its effectiveness in capturing clear and consistent structural boundaries.

<table><tr><td>Sx,y</td><td>Î»edge</td><td>2.0</td><td>4.0</td><td>8.0</td></tr><tr><td>3 Ã 3</td><td></td><td>36.3 / 29.3</td><td>36.3 / 29.4</td><td>36.1 / 29.2</td></tr><tr><td>5Ã 5</td><td></td><td>36.3 / 29.3</td><td>36.3 / 29.1</td><td>36.2 / 29.2</td></tr><tr><td>7Ã 7</td><td></td><td>36.3 / 29.1</td><td>36.1 / 29.1</td><td>36.2 / 29.2</td></tr></table>

Table 7. Ablation study on edge loss weight $( \lambda _ { \mathrm { e d g e } } )$ and Sobel kernel size $( S _ { x , y } )$ . The results show that moderate edge supervision $( \lambda _ { \mathrm { e d g e } } = 4 . 0 )$ with a 3Ã3 kernel achieves the best balance between geometric accuracy and boundary precision.

## 5. Conclusion

We presented HyGE-Occ, a hybrid view-transformation framework with 3D Gaussian and edge priors that enhances geometric consistency and boundary precision in 3D panoptic occupancy prediction. By combining continuous 3D Gaussian based reasoning with discretized depth features and introducing an edge-guided BEV refinement, HyGE-Occ produces sharper instance delineation and more coherent 3D geometry. Evaluated on the Occ3D-nuScenes benchmark, it achieves state-of-the-art performance, surpassing strong baselines such as Panoptic-FlashOcc and ALOcc.

Limitations. HyGE-Occ is limited to BEV-based models, and its effectiveness may not directly generalize to other architectures. In addition, the hybrid view-transformation branch introduces modest computational overhead compared to discretized LSS-based methods. Exploring more efficient formulations, broader architectural compatibility, and extending to flow prediction remains as future work.

## References

[1] Simon Boeder and Benjamin Risse. Occflownet: Occupancy estimation via differentiable rendering and occupancy flow. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 306â316. IEEE, 2025. 2

[2] Simon Boeder, Fabian Gigengack, and Benjamin Risse. Gaussianflowocc: Sparse and weakly supervised occupancy estimation using gaussian splatting and temporal flow. arXiv preprint arXiv:2502.17288, 2025. 2

[3] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621â11631, 2020. 2

[4] Anh-Quan Cao and Raoul De Charette. Monoscene: Monocular 3d semantic scene completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3991â4001, 2022. 2

[5] Dubing Chen, Jin Fang, Wencheng Han, Xinjing Cheng, Junbo Yin, Chengzhong Xu, Fahad Shahbaz Khan, and Jianbing Shen. Alocc: Adaptive lifting-based 3d semantic occupancy and cost volume-based flow predictions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4156â4166, 2025. 2, 6, 7

[6] Bowen Cheng, Ross Girshick, Piotr Dollar, Alexander C Â´ Berg, and Alexander Kirillov. Boundary iou: Improving object-centric image segmentation evaluation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15334â15342, 2021. 3

[7] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation. Advances in neural information processing systems, 34:17864â17875, 2021. 1

[8] Arun CS Kumar, Suchendra M Bhandarkar, and Mukta Prasad. Depthnet: A recurrent neural network architecture for monocular depth prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 283â291, 2018. 1

[9] Yubo Cui, Zhiheng Li, Jiaqiang Wang, and Zheng Fang. Loma: Language-assisted semantic occupancy network via triplane mamba. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2609â2617, 2025. 1

[10] Wanshui Gan, Fang Liu, Hongbin Xu, Ningkai Mo, and Naoto Yokoya. Gaussianocc: Fully self-supervised and efficient 3d occupancy estimation with gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 28980â28990, 2025. 2

[11] Jingyu Gong, Jiachen Xu, Xin Tan, Jie Zhou, Yanyun Qu, Yuan Xie, and Lizhuang Ma. Boundary-aware geometric encoding for semantic segmentation of point clouds. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1424â1432, 2021. 3

[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770â778, 2016. 1

[13] Jiawei Hou, Xiaoyan Li, Wenhao Guan, Gang Zhang, Di Feng, Yuheng Du, Xiangyang Xue, and Jian Pu. Fastocc: Accelerating 3d occupancy prediction by fusing the 2d birdâs-eye view and perspective view. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 16425â16431. IEEE, 2024. 2

[14] Junjie Huang and Guan Huang. Bevdet4d: Exploit temporal cues in multi-camera 3d object detection. arXiv preprint arXiv:2203.17054, 2022. 6

[15] Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790, 2021. 2

[16] Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Tri-perspective view for visionbased 3d semantic occupancy prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9223â9232, 2023. 1, 2

[17] Yuanhui Huang, Wenzhao Zheng, Borui Zhang, Jie Zhou, and Jiwen Lu. Selfocc: Self-supervised vision-based 3d occupancy prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19946â19956, 2024. 2

[18] Haoyi Jiang, Liu Liu, Tianheng Cheng, Xinjie Wang, Tianwei Lin, Zhizhong Su, Wenyu Liu, and Xinggang Wang. Gausstr: Foundation model-aligned gaussian transformer for self-supervised 3d spatial understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11960â11970, 2025. 2

[19] Nick Kanopoulos, Nagesh Vasanthavada, and Robert L Baker. Design of an image edge detection filter using the sobel operator. IEEE Journal of solid-state circuits, 23(2): 358â367, 1988. 5

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 4

[21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015â4026, 2023. 1

[22] Hong Joo Lee, Jung Uk Kim, Sangmin Lee, Hak Gu Kim, and Yong Man Ro. Structure boundary preserving segmentation for medical image with ambiguous boundary. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4817â4826, 2020. 3

[23] Yinhao Li, Han Bao, Zheng Ge, Jinrong Yang, Jianjian Sun, and Zeming Li. Bevstereo: Enhancing depth estimation in multi-view 3d object detection with temporal stereo. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1486â1494, 2023. 2

[24] Zhiqi Li, Zhiding Yu, David Austin, Mingsheng Fang, Shiyi Lan, Jan Kautz, and Jose M Alvarez. Fb-occ: 3d occupancy prediction based on forward-backward view transformation. arXiv preprint arXiv:2307.01492, 2023. 6

[25] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer:

learning birdâs-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 6

[26] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection, 2018. Â´ 4

[27] Haisong Liu, Yang Chen, Haiguang Wang, Zetong Yang, Tianyu Li, Jia Zeng, Li Chen, Hongyang Li, and Limin Wang. Fully sparse 3d occupancy prediction. In European Conference on Computer Vision, pages 54â71. Springer, 2024. 2, 6

[28] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012â10022, 2021. 1

[29] Shu-Wei Lu, Yi-Hsuan Tsai, and Yi-Ting Chen. Toward real-world bev perception: Depth uncertainty estimation via gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17124â17133, 2025. 1, 2, 3, 4

[30] Qihang Ma, Xin Tan, Yanyun Qu, Lizhuang Ma, Zhizhong Zhang, and Yuan Xie. Cotr: Compact occupancy transformer for vision-based 3d occupancy prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19936â19945, 2024. 2

[31] Dmitrii Marin, Zijian He, Peter Vajda, Priyam Chatterjee, Sam Tsai, Fei Yang, and Yuri Boykov. Efficient segmentation: Learning downsampling near semantic boundaries. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2131â2141, 2019. 3

[32] Zhenxing Ming, Julie Stephany Berrio, Mao Shan, and Stewart Worrall. Occfusion: Multi-sensor fusion framework for 3d semantic occupancy prediction. IEEE Transactions on Intelligent Vehicles, 2024. 2

[33] Gyeongrok Oh, Sungjune Kim, Heeju Ko, Hyung-gun Chi, Jinkyu Kim, Dongwook Lee, Daehyun Ji, Sungjoon Choi, Sujin Jang, and Sangpil Kim. 3d occupancy prediction with low-resolution queries via prototype-aware view transformation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17134â17144, 2025. 2

[34] Mingjie Pan, Jiaming Liu, Renrui Zhang, Peixiang Huang, Xiaoqi Li, Hongwei Xie, Bing Wang, Li Liu, and Shanghang Zhang. Renderocc: Vision-centric 3d occupancy prediction with 2d rendering supervision. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 12404â12411. IEEE, 2024. 2

[35] Sylvain Paris, Samuel W Hasinoff, and Jan Kautz. Local laplacian filters: Edge-aware image processing with a laplacian pyramid. ACM Trans. Graph., 30(4):68, 2011. 8

[36] Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In European conference on computer vision, pages 194â210. Springer, 2020. 1, 2, 3

[37] Judith MS Prewitt et al. Object enhancement and extraction. Picture processing and Psychopictorics, 10(1):15â19, 1970. 8

[38] Wonseok Roh, Hwanhee Jung, Giljoo Nam, Jinseop Yeom, Hyunje Park, Sang Ho Yoon, and Sangpil Kim. Edge-aware

3d instance segmentation network with intelligent semantic prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20644â20653, 2024. 3

[39] Faraz Saeedan and Stefan Roth. Boosting monocular depth with panoptic segmentation maps. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3853â3862, 2021. 3

[40] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446â2454, 2020. 2

[41] Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, and Hang Zhao. Occ3d: A large-scale 3d occupancy prediction benchmark for autonomous driving. Advances in Neural Information Processing Systems, 36:64318â64330, 2023. 2, 6

[42] Letian Wang, Seung Wook Kim, Jiawei Yang, Cunjun Yu, Boris Ivanovic, Steven Waslander, Yue Wang, Sanja Fidler, Marco Pavone, and Peter Karkus. Distillnerf: Perceiving 3d scenes from single-glance images by distilling neural fields and foundation model features. Advances in Neural Information Processing Systems, 37:62334â62361, 2024. 2

[43] Yuqi Wang, Yuntao Chen, Xingyu Liao, Lue Fan, and Zhaoxiang Zhang. Panoocc: Unified occupancy representation for camera-based 3d panoptic segmentation, 2023. 2

[44] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10371â10381, 2024. 1

[45] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875â21911, 2024. 1

[46] Dongqiangzi Ye, Zixiang Zhou, Weijia Chen, Yufei Xie, Yu Wang, Panqu Wang, and Hassan Foroosh. Lidarmultinet: Towards a unified multi-task network for lidar perception. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3231â3240, 2023. 2

[47] Maosheng Ye, Rui Wan, Shuangjie Xu, Tongyi Cao, and Qifeng Chen. Drinet++: Efficient voxel-as-point point cloud segmentation. arXiv preprint arXiv:2111.08318, 2021. 2

[48] Zichen Yu, Changyong Shu, Jiajun Deng, Kangjie Lu, Zongdai Liu, Jiangyong Yu, Dawei Yang, Hui Li, and Yan Chen. Flashocc: Fast and memory-efficient occupancy prediction via channel-to-height plugin. arXiv preprint arXiv:2311.12058, 2023. 2

[49] Zichen Yu, Changyong Shu, Qianpu Sun, Yifan Bian, Xiaobao Wei, Jiangyong Yu, Zongdai Liu, Dawei Yang, Hui Li, and Yan Chen. Panoptic-flashocc: An efficient baseline to marry semantic occupancy with panoptic via instance center. arXiv preprint arXiv:2406.10527, 2024. 2, 5, 6

[50] Chubin Zhang, Juncheng Yan, Yi Wei, Jiaxin Li, Li Liu, Yansong Tang, Yueqi Duan, and Jiwen Lu. Occnerf: Advancing

3d occupancy prediction in lidar-free environments. IEEE Transactions on Image Processing, 2025. 2

[51] Shuo Zhang, Yupeng Zhai, Jilin Mei, and Yu Hu. Fusionocc: Multi-modal fusion for 3d occupancy prediction. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 787â796, 2024. 2

[52] Yang Zhang, Zixiang Zhou, Philip David, Xiangyu Yue, Zerong Xi, Boqing Gong, and Hassan Foroosh. Polarnet: An improved grid representation for online lidar point clouds semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9601â9610, 2020. 2

[53] Yunpeng Zhang, Zheng Zhu, and Dalong Du. Occformer: Dual-path transformer for vision-based 3d semantic occupancy prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9433â9443, 2023. 2, 6

[54] Weiguang Zhao, Rui Zhang, Qiufeng Wang, Guangliang Cheng, and Kaizhu Huang. Bfanet: Revisiting 3d semantic segmentation with boundary feature analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29395â29405, 2025. 3

[55] Ziyue Zhu, Shenlong Wang, Jin Xie, Jiang-jiang Liu, Jingdong Wang, and Jian Yang. Voxelsplat: Dynamic gaussian splatting as an effective loss for occupancy and flow prediction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6761â6771, 2025. 2