# FLEG: Feed-Forward Language Embedded Gaussian Splatting from Any Views

Qijian Tian1, Xin Tan2,3, Jiayu Ying2, Xuhong Wang3, Yuan Xie2, Lizhuang Ma1 1 Shanghai Jiao Tong University 2 East China Normal University 3 Shanghai Artificial Intelligence Laboratory

<!-- image-->  
Figure 1. FLEG reconstructs language-embedded Gaussians in a single feed-forward pass from any uncalibrated and unposed multi-view images, supporting both sparse and dense views in one model. The reconstructed language-embedded Gaussians simultaneously enable novel-view synthesis, 3D query, and 3D editing, exhibiting potential for efficient data generation and real-time downstream applications.

## Abstract

We present FLEG, a feed-forward network that reconstructs language-embedded 3D Gaussians from any views. Previous straightforward solutions combine feed-forward reconstruction with Gaussian heads but suffer from fixed input views and insufficient 3D training data. In contrast, we propose a 3D-annotation-free training framework for 2D-to-3D lifting from arbitrary uncalibrated and unposed multi-view images. Since the framework does not require 3D annotations, we can leverage large-scale video data with easily obtained 2D instance information to enrich semantic embedding. We also propose an instance-guided contrastive learning to align 2D semantics with the 3D representations. In addition, to mitigate the high memory and computational cost of dense views, we further propose a geometryâsemantic hierarchical sparsification strategy. Our FLEG efficiently reconstructs language-embedded

3D Gaussian representation in a feed-forward manner from arbitrary sparse or dense views, jointly producing accurate geometry, high-fidelity appearance, and languagealigned semantics. Extensive experiments show that it outperforms existing methods on various related tasks. Project page: https://fangzhou2000.github. io/projects/fleg.

## 1. Introduction

Embedding language into the 3D representation is essential for interaction with 3D environments in a human-like manner, which enables query, editing, and other interactions through natural language. It has various applications, such as robotic navigation [1, 8], manipulation [24, 27], and augmented/virtual reality [5]. Constructing such 3D language fields requires lifting 2D observations into coherent 3D geometry and appearance while jointly embedding semantics to achieve open-vocabulary interaction, which involves both 3D reconstruction and visionâlanguage modeling.

With the rapid development of 3D Gaussian Splatting [11] (3DGS), many works [9, 16, 20, 35] embed features from 2D visionâlanguage models into Gaussians and build language fields through per-scene optimization. Although the differentiable rasterization of 3DGS enables fast rendering, the per-scene optimization remains inefficient for language field reconstruction, as the optimization process can take minutes or even hours, making these methods impractical for real-time applications such as robotic navigation and manipulation. Besides, per-scene optimized methods typically rely on dense-view input and perform poorly under sparse-view conditions, while sparse-view input is common in practice. On the other hand, recent feed-forward reconstruction methods [12, 30, 32] have achieved approximately real-time 3D reconstruction from multi-view images to 3D point clouds, while they fail to perform high-fidelity novel-view synthesis (NVS) and semantic embedding. One straightforward solution is to incorporate a Gaussian head into the feed-forward reconstruction model, using 3D data for training and distilling semantics from 2D models. However, these methods are limited to a fixed number of input views, such as only two views [14, 25, 29, 33], or require additional camera parameters [28], which restricts their practical applicability. Moreover, 3D annotation data are also insufficient compared with large-scale 2D video data from the Internet. These limitations in input views and 3D data further hinder semantic distillation, ultimately restricting the accuracy and generalization of embedded semantics.

In this paper, we propose FLEG, a feed-forward network that effectively reconstructs language-embedded Gaussians from any uncalibrated and unposed multi-view images. FLEG simultaneously achieves accurate geometric reconstruction, high-fidelity rendering, embeds language-aligned 3D semantics, and supports both dense and sparse input views. To address the limitations of existing straightforward solutions that rely on fixed input views and require 3D training data, we propose a 3D-annotation-free training framework for 2D-to-3D lifting from arbitrary uncalibrated and unposed multi-view images. Specifically, we leverage the point maps and camera poses from the pretrained 3D foundation model, VGGT [30], to generate view masks for selecting novel views during training and distill geometric priors from selected novel views. Since the framework does not require 3D annotations, we can leverage large-scale video data with easily obtained instance information to enrich semantic embedding. Accordingly, we construct InstanceMV-14K, a large-scale dataset collected from 3D scans and Internet videos, containing 2D multiview images and instance masks for network training without costly 3D annotation. We also propose an instanceguided contrastive learning to align semantics with the

3D representation, which focuses on instance-level cues and reduces the time complexity for feed-forward training. In addition, the pixel-wise 3D representations in typical feed-forward reconstruction methods often lead to redundant Gaussians under dense-view input, while semantic embedding further increases memory and computational costs. To address this issue, we propose a geometryâsemantic hierarchical sparsification strategy, which exploits the inherent sparsity of semantics to achieve more efficient semantic embedding. Through comprehensive design across the training framework, data collection, learning strategy and semantic representation, our FLEG efficiently reconstructs language-embedded 3D Gaussian representations in a single feed-forward pass from arbitrary sparse or dense multi-view inputs, jointly producing high-fidelity geometry, appearance, and language-aligned semantics.

In summary, our key contributions are as follows:

â¢ We propose FLEG, a feed-forward network that effectively reconstructs language-embedded 3D Gaussians in a single feed-forward pass from uncalibrated and unposed images, supporting both sparse and dense views.

â¢ We propose a 3D-annotation-free training framework to train the network with novel view distillation, and construct InstanceMV-14K, a large-scale image dataset with corresponding instance masks from existing scan data and Internet videos to enrich semantic embedding. We further propose an instance-guided contrastive learning to align 2D semantics with 3D representation.

â¢ We propose a geometryâsemantic hierarchical sparsification strategy that embeds semantics with sparse isotropic Gaussians to reduce memory and computational costs, while preserving the quality of reconstruction and semantic embedding.

â¢ Our FLEG simultaneously enables multi-tasks through language-embedded Gaussians, including novel-view synthesis, open-vocabulary object query, and 3D editing. Comprehensive experiments show that FLEG achieves the best performance against previous methods.

## 2. Related Work

## 2.1. Feed-Forward Reconstruction

Typical per-scene optimization reconstruction methods like NeRF [18] and 3DGS [11] achieve high-quality rendering while often taking minutes or even hours to optimize a single scene, making them unsuitable for real-time applications. Recent advances in feed-forward reconstruction have substantially improved the reconstruction efficiency, both for Gaussian-based methods [3, 4, 10] and point-map-based methods [12, 30â32]. However, these methods primarily focus on geometry and appearance reconstruction and lack semantic embedding, preventing natural language interaction with the 3D scene. In contrast, our feed-forward reconstruction method recovers high-quality geometry and appearance from arbitrary input views while simultaneously embedding semantics into the 3D representation, achieving 3D reconstruction and semantic embedding and thereby facilitating the joint perception and understanding of the 3D world.

<!-- image-->  
Figure 2. Overview of FLEG. Our FLEG adopts a large transformer with a DPT-based decoder and corresponding prediction heads to predict language-embedded Gaussians. We propose a 3D-annotation-free training framework to eliminate the reliance on 3D annotation. To embed semantics into 3D representations, we construct InstanceMV-14K to enrich semantic diversity. We also introduce an instance-guided contrastive learning to effectively align 2D instances with 3D representations. We further propose a geometryâsemantic hierarchical sparsification strategy to avoid the cost of per-pixel predictions.

## 2.2. Language Embedded Fields

With recent advances in 3DGS, several methods [20, 26] have explored embedding language into Gaussians to construct language embedded fields. Following works [9, 15] further improve the query efficiency and achieve higher rendering speed. Despite the improvement in query and rendering speed, such per-scene optimization methods remain computationally expensive for reconstruction, limiting their applicability in real-world scenarios. Following the advances in feed-forward reconstruction, several methods [7, 14, 25, 29, 33] attempted to achieve feedforward language-embedded field construction. However, these methods can only process two input views at once and are struggle to handle dense view inputs. Uni3R [28] extends to multi-view settings but remains view-specific, which is less flexible and requires additional camera parameters. Our method overcomes these limitations, enabling feed-forward language field reconstruction from arbitrary uncalibrated and unposed multi-view images.

A related work, EA3D, focuses on incremental reconstruction of language fields, yet still requires per-frame optimization and relies on 2D segmentation models, which may introduce semantic inconsistencies across views. Another recent work, IGGT, enables feed-forward instance segmentation with point map reconstruction. However, it relies on external 2D semantic models to obtain semantic information, which may also lead to semantic inconsistencies, and the point map representation does not support novel-view synthesis. In contrast, FLEG is a fully feed-forward method that does not rely on 2D semantic models during inference. It generates consistent 3D semantic embedding and supports high-fidelity novel-view synthesis through languageembedded Gaussian Splatting.

## 3. Method

## 3.1. Overview

We propose FLEG, a feed-forward network that reconstructs language-embedded Gaussians from any uncalibrated and unposed multi-view images. Our FLEG simultaneously reconstructs accurate geometry (along with the recovered camera parameters), high-fidelity appearance, and language-aligned semantics. It supports both sparse (as few as 2) and dense (more than 30) views. The reconstructed language-embedded Gaussians enable various tasks, including novel-view synthesis, open-vocabulary object query, and 3D editing. Fig. 2 illustrates the overview of our FLEG.

## 3.2. Network Architecture

Our FLEG adopts a large transformer following VGGT [30] that first patchify each image into a set of tokens using DI-NOv2 [19]. Subsequently, a learnable camera token and four register tokens are concatenated to each viewâs token sequences. The combined tokens are processed by 24 layers of global and frame-wise attention. Finally, to decode the processed tokens into language-embedded Gaussians, we employ three heads: camera head, depth head, and semantic Gaussian head. The camera head and depth head predict camera parameters and pixel-wise depths, respectively. The depths are then back-projected into points $\mu$ in the world coordinate through predicted camera parameters. The Gaussian head is designed based on DPT [22] as the depth head, which fuses the DPT feature and image feature to predict the attributions of language-embedded Gaussians, including scale $s \in \ \mathbb { R } ^ { 3 }$ , rotation quaternion $r \in \mathbb { R } ^ { 4 }$ , opacity $\sigma \in \mathbb { R } ^ { + }$ , SH coefficients $c \in \bar { \mathbb { R } } ^ { 3 \times ( k + 1 ) ^ { 2 } }$ that represent colors with spherical-harmonic coefficients of degree $k ,$ and a D-dimensional language-aligned semantic feature f eat $\in \mathbb { R } ^ { D }$ . These attributes, along with the back-projected $\mu \in \mathbb { R } ^ { 3 }$ , constitute the pixel-wise languageembedded Gaussians.

## 3.3. 3D-annotation-free Training Framework

Unlike prior feed-forward methods that rely on 3D annotations such as point maps and camera poses, we introduce a 3D-annotation-free training framework that fully exploits large-scale 2D data, which is significantly easier to obtain. Training on large-scale 2D data expands the diversity of geometric and appearance information for reconstruction, while simultaneously enriching the semantic embeddings.

To train the network without 3D annotations, we leverage the preatrained VGGT [30], a feed-forward reconstruction model with strong geometric ability, to predict point maps and camera parameters as geometric supervision $\mathcal { L } _ { d i s t i l l }$ However, these pseudo-labels contain only geometric information from the input views and cannot guide Gaussian-based representations to learn cross-view consistent appearance for novel-view synthesis. Prior Gaussianbased feed-forward reconstruction methods either rely on 3D information to obtain novel-view photometric supervision [3, 4] or restrict supervision to input views only [10], which limits cross-view consistency and appearance fidelity. To enable 3D annotation-free training with novelview supervision that enhances cross-view consistency, we propose a novel-view-based photometric supervision under the distillation framework.

Specifically, given N input images $\{ I _ { i } \} _ { i = 1 } ^ { N }$ , we obtain the predicted 3D point maps $\{ P _ { i } \} _ { i = 1 } ^ { N }$ and camera parameters $\{ K _ { i } \} _ { i = 1 } ^ { N } , \{ \bar { E } _ { i } \} _ { i = 1 } ^ { N }$ from VGGT. During training, we randomly sample $N _ { c }$ images from $\{ I _ { i } \} _ { i = 1 } ^ { N }$ as context views $\{ I _ { i } \} _ { i = 1 } ^ { N _ { c } }$ for network input, where $2 \leq N _ { c } \leq N$ . For target views used for novel view supervision, we select images from $\{ I _ { i } \} _ { i = 1 } ^ { N }$ 1 that are sufficiently covered by context views to ensure that novel views remain observable and do not introduce many unseen regions that may degrade supervision quality. We project point maps of context views $\{ P _ { i } \} _ { i = 1 } ^ { N _ { c } }$ onto $\{ I _ { i } \} _ { i = 1 } ^ { N }$ to obtain the binary coverage masks $\{ M _ { i } \} _ { i = 1 } ^ { N _ { c } } ;$

$$
\{ M _ { i } \} _ { i = 1 } ^ { N } = \Pi ( \{ P _ { i } \} _ { i = 1 } ^ { N _ { c } } , \{ K _ { i } \} _ { i = 1 } ^ { N } , \{ E _ { i } \} _ { i = 1 } ^ { N } ) ,\tag{1}
$$

where Î  is the projection function that maps 3D points from

world coordinates to camera coordinates, and $\{ M \} _ { i = 1 } ^ { N }$ represent the binary masks of $\{ I \} _ { i = 1 } ^ { N }$

Target views are selected as the images with aggregated coverage masks exceeding a predefined threshold $\boldsymbol { \tau } ;$

$$
\{ I \} _ { i = 1 } ^ { N _ { t } } = \{ I _ { j } | c o v ( M _ { j } ) > \tau \} ,\tag{2}
$$

where cov(Â·) is the coverage ratio for each image:

$$
\operatorname { c o v } ( M _ { j } ) = \frac { 1 } { H \times W } \sum _ { h = 1 } ^ { H } \sum _ { w = 1 } ^ { W } \mathbb { I } \big ( M _ { j } ( h , w ) > 0 \big ) .\tag{3}
$$

With the selected target views, we can apply photometric supervision on novel views to enhance consistency across views:

$$
\mathcal { L } _ { p h o t o } = \eta \frac { 1 - S S I M ( I , \hat { I } ) } { 2 } + ( 1 - \eta ) \| I - \hat { I } \| ,\tag{4}
$$

## 3.4. Language-aligned Semantics Embedding

The 3D-annotation-free training framework enables the network to learn geometry and appearance from $\mathcal { L } _ { d i s t i l l }$ and $\mathcal { L } _ { p h o t o }$ , respectively. To further interpolate languagealigned semantics and predict language-embedded Gaussians, we lift the semantics of the 2D vision-language model [21] from 2D to 3D through feature distillation, also without relying on any 3D annotations under the framework. The embedded semantics further enable languagerelated tasks, such as 3D querying and open-vocabulary segmentation, by computing similarity with text embeddings from the 2D vision-language model.

InstanceMV-14K Dataset. Since our training framework does not require any 3D annotation, we can leverage largescale 2D data with easily obtained instance information to enrich the semantics we embed. To this end, we construct InstanceMV-14K, a large-scale multi-view image dataset with instance masks. The data are collected from existing scan data (ScanNet [6], ScanNet++ [34], ArkitScenes [2]) and Internet video data (RealEstate10K [36]). For Scan-Net and ScanNet++ with 3D annotations, we directly utilize the back-projected 2D instance mask without additional processing. For other data without instance annotation, we employ SAM2 [23] to obtain the instance information. Our InstanceMV-14K contains more than 15 million images and corresponding instance masks, covering a diverse range of approximately 14,000 scenes. Such a large-scale multiview image dataset with instance masks contains abundant semantic information, significantly enriching the semantic embedding for our network.

During training, we leverage instance masks from InstanceMV-14K to extract language-aligned features for each instance using the powerful 2D vision-language model CLIP [21]. The CLIP feature for each instance is applied to the pixels within its corresponding instance mask, generating the pixel-wise language-aligned feature as the target feature $\dot { F } \in \mathbb { R } ^ { N _ { t } \times C \times H \times W }$ The predicted languageembedded Gaussians with D-dimensional features feat are rendered into 2D feature maps $\hat { F } \in \mathbb { R } ^ { N _ { t } \times C \times H \times \tilde { W } }$ , which are passed through a light-weight MLP and compute cosine similarity loss with corresponding target features:

$$
\mathcal { L } _ { f e a t } = 1 - \frac { \hat { F } \cdot F } { \Vert \hat { F } \Vert \Vert F \Vert } ,\tag{5}
$$

In addition, to address the inconsistency of CLIP-extracted features across multi-views, we propose a 3D feature aggregation module to obtain consistent multi-view target features. Specifically, we first use the 3D point maps predicted by VGGT to lift the 2D pixel-wise language-aligned features into 3D space. Based on the assumption that pixels mapped to the same spatial location should share similar features, we voxelize the 3D space and average the features within each voxel, which are subsequently projected back to 2D image, and we further average the features within each instance mask. This process enhances the multi-view consistency of CLIP-extracted features.

Instance-guided Contrastive Learning. The semantic embedding distilled from 2D features lacks instance-level discrimination, resulting in blurred boundaries. To better align the 2D semantics with 3D representations and address this issue, we fully exploit the instance information provided by InstanceMV-14K and propose an instance-guided contrastive learning algorithm. Specifically, given a feature map $F \in \mathbb { R } ^ { N _ { t } \times C \times \mathbf { \breve { H } } \times W }$ and corresponding instance mask $\bar { M } \bar { \mathbf { \Psi } } \in \mathbb { R } ^ { N _ { t } \times H \times W }$ , we first compute mean feature vector for each instance as the anchor feature $\mathbf { f } _ { k } ^ { \mathrm { i n s } } \in \mathbb { R } ^ { C }$ . For each sampled pixel, the pixel feature vector $\mathbf { f } _ { i } ^ { \mathrm { p i x } } \in \mathbb { R } ^ { C }$ is regarded as a positive sample if it belongs to the same instance as the anchor, and as a negative sample otherwise. The instance-guided contrastive loss is formulated as:

$$
\mathcal { L } _ { \mathrm { i n s t } } = - \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \log \frac { \sum _ { j \in \mathcal { P } _ { k } } \exp \left( \sin \left( \mathbf { f } _ { k } ^ { \mathrm { i n s } } , \mathbf { f } _ { j } ^ { \mathrm { p i x } } \right) / \alpha \right) } { \sum _ { j } \exp \left( \sin \left( \mathbf { f } _ { k } ^ { \mathrm { i n s } } , \mathbf { f } _ { j } ^ { \mathrm { p i x } } \right) / \alpha \right) } ,\tag{6}
$$

where $\mathbf { f } _ { k } ^ { \mathrm { i n s } }$ denotes the anchor feature of the k-th instance, $\mathbf { f } _ { j } ^ { \mathrm { p i x } }$ is the feature of the j-th sampled pixel, $\mathcal { P } _ { k }$ is the set of pixels belonging to the same instance as the k-th anchor, and Î± is the temperature parameter.

By leveraging instance masks to distinguish positive and negative samples, the proposed instance-guided contrastive learning effectively aligns the 3D-splat features with 2D instance masks, enabling the network to learn more discriminative language-aligned semantic embedding in 3D representation. In addition, this algorithm significantly reduces the computational cost compared to conventional pixelwise contrastive learning. Traditional pixel-wise contrastive learning requires computing pairwise similarities among all $N _ { p }$ pixels, leading to a complexity of $\mathcal { O } ( N _ { p } ^ { 2 } )$ . In contrast, our algorithm aggregates pixels into K instance-level anchors and computes similarities only between these anchors and $N _ { p }$ pixels, resulting in a complexity of $\mathcal { O } ( N _ { p } K )$ , where typically $K \ \ll \ N _ { p }$ This approach yields a substantial reduction in computational cost while maintaining strong instance-level discrimination. Moreover, if we perform pixel sampling from the $N _ { p }$ pixels, the complexity can be reduced further.

Geometry-Semantic Hierarchical Sparsification Strategy. Typical feed-forward reconstruction methods often perform pixel-wise predictions, resulting in a large number of redundant Gaussians for dense-view input and increasing memory and computational costs. The incorporation of semantic embeddings further exacerbates this issue. Previous methods [10, 17] utilize voxel-based sparsification to reduce redundant Gaussians. However, these methods do not include semantic embedding and therefore do not consider the differing spatial densities required to represent geometric and semantic information. To further mitigate the additional overhead introduced by semantic embeddings, we propose a geometry-semantic hierarchical sparsification strategy based on voxel-based sparsification. First, we voxelize the predicted Gaussians in the 3D coordinate:

$$
\{ V _ { i } \} _ { i = 1 } ^ { S _ { g e o } } = \left\lfloor \frac { \{ \mu _ { i } \} _ { i = 1 } ^ { N _ { c } \times H \times W } } { \epsilon _ { g e o } } \right\rceil ,\tag{7}
$$

where $\{ V _ { i } \} _ { i = 1 } ^ { S _ { g e o } }$ is the voxel index of Gaussian $i , S _ { g e o }$ is the total number of voxels, and $\epsilon _ { g e o }$ is the voxel size. The attributions of pixel-wise Gaussians are averaged within corresponding voxels through softmax:

$$
\bar { x } _ { s } = \sum _ { i : V ^ { i } = s } \frac { e x p ( c o n f _ { i } ) } { \sum _ { j : V ^ { j } = s } e x p ( c o n f _ { j } ) } x _ { i } ,\tag{8}
$$

where $x _ { i } ~ \in ~ \{ \mu _ { i } , s _ { i } , r _ { i } , \sigma _ { i } , c _ { i } , f e a t _ { i } , c o n f _ { i } \} _ { i = 1 } ^ { N _ { c } \times H \times W }$ and con $f _ { i }$ is the predicted Gaussian confidence from Gaussian head.

The voxel-based sparsification reduces the number of Gaussians from pixel-wise prediction by merging adjacent predictions. However, this merging is performed at the Gaussian level and does not take into account that different Gaussian attributes require different spatial densities. For our language-embedded Gaussians, semantic information is inherently sparser than geometry: semantic signals correspond to high-level, instance-level concepts localized in discrete regions, while geometry changes continuously across surfaces. Existing methods that incorporate features into Gaussians typically follow a per-Gaussian embedding paradigm, which results in redundant semantic representations and imposes additional memory and computational costs. To address this issue, we further decouple the geometry and semantic representation and propose a more effective sparsification strategy for semantic embedding.

Building upon the voxel-based sparsification described above, we decouple the voxelized Gaussians $\{ g _ { i } \} _ { i = 1 } ^ { S _ { g e o } }$ into a more sparse set of semantic Gaussians,

$$
\{ g _ { j } \} _ { j = 1 } ^ { S _ { s e m } } = \{ \mu _ { j } , s _ { j } , r _ { j } , \sigma _ { j } , f e a t _ { j } \} _ { j = 1 } ^ { S _ { s e m } }\tag{9}
$$

to represent the semantic embedding while preserving the original voxelized Gaussians without semantic attributes as geometry Gaussians,

$$
\{ g _ { i } \} _ { i = 1 } ^ { S _ { g e o } } = \{ \mu _ { i } , s _ { i } , r _ { i } , \sigma _ { i } , c _ { i } \} _ { i = 1 } ^ { S _ { g e o } } ,\tag{10}
$$

where $S _ { s e m }$ denotes the number of semantic Gaussians obtained via voxelization in Eq. (7) using a smaller voxel size $\epsilon _ { s e m }$ . We further aggregate $\mu _ { i } , \sigma _ { i }$ and feati from $\{ g _ { i } \} _ { i = 1 } ^ { S _ { g e o } }$ through the softmax average in Eq. (8) to derive $\mu _ { j } , \sigma _ { j }$ and f eatj of $\{ g _ { j } \} _ { j = 1 } ^ { S _ { s e m } }$ . The average weight is denoted as wi.

For the scale $s _ { j }$ and rotation $r _ { j }$ that determine the shape of each semantic Gaussian $\{ g _ { j } \} _ { j = 1 } ^ { S _ { s e m } }$ , we do not apply the same averaging used for the aggregation of position, opacity, and feature. Since semantic Gaussians are constructed with higher sparsity, they are encouraged to spatially cover the corresponding geometry Gaussians to maintain semantic consistency across regions. Moreover, as semantic information exhibits weaker anisotropic variation compared to geometric appearance, we employ isotropic semantic Gaussians to achieve a compact and stable representation.

Specifically, for all geometry Gaussians $\{ g _ { i } \} _ { i = 1 } ^ { S _ { g e o } }$ within the same semantic voxel, their covariance matrices $\Sigma _ { i } =$ $R _ { i } \mathrm { d i a g } ( s _ { i } ^ { 2 } ) R _ { i } ^ { \top }$ are fused using a moment-matching scheme with softmax-normalized confidence weights $w _ { i } \colon$

$$
\Sigma _ { j } = \sum _ { i \in j } w _ { i } \left( \Sigma _ { i } + ( \mu _ { i } - \mu _ { j } ) ( \mu _ { i } - \mu _ { j } ) ^ { \top } \right) .\tag{11}
$$

Instead of retaining the full anisotropic covariance, we approximate the fused covariance as isotropic using its trace:

$$
s _ { j } = \sqrt { \frac { \mathrm { T r } ( \tilde { \Sigma } _ { j } ) } { 3 } } \cdot { \bf 1 } , \qquad r _ { j } = [ 1 , 0 , 0 , 0 ] ,\tag{12}
$$

where $s _ { j }$ and $r _ { j }$ denote the scale and rotation of the semantic Gaussian, respectively. This isotropic approximation ensures that each semantic Gaussian effectively spans the corresponding geometric region while avoiding unnecessary anisotropic computation and redundancy.

Through the proposed sparsification strategy, we decouple semantic and geometric densities, reducing the number of semantic Gaussians while preserving representational completeness and maintaining both accurate geometry and consistent semantic coverage.

## 3.5. Training Objectives

During training, our method does not require any 3D annotations, which allows us to leverage the constructed InstanceMV-14K dataset. These large-scale multi-view images provide diverse appearance and semantic information from abundant video data, and the large scale of the data facilitates effective distillation of both geometric structures and feature representations.

Overall, we utilize the following loss function:

$$
\mathcal { L } = \mathcal { L } _ { p h o t o } + \mathcal { L } _ { f e a t } + \lambda _ { 1 } \mathcal { L } _ { d i s t i l l } + \lambda _ { 2 } \mathcal { L } _ { i n s t } .\tag{13}
$$

$\mathcal { L } _ { p h o t o }$ and $\mathcal { L } _ { f e a t }$ enable the basic ability of reconstruction and semantic embedding, while $\mathcal { L } _ { d i s t i l l }$ and $\mathcal { L } _ { i n s t }$ improve the quality of geometry and semantics, respectively.

## 4. Experiments

## 4.1. Implementation details

We initialize the transformer layers, camera head, and depth head with weights from pretrained VGGT [30] and randomly initialize the semantic Gaussian head. The whole network contains approximately 1B parameters. During training, inputs are resized to a maximum resolution of 518 pixels on the longer side with aspect ratios sampled between 0.5 and 1.0 and augmented via random horizontal flipping. For each iteration, we set N = 14 and randomly select $2 \leq N _ { c } \leq N$ as the input views. We apply the AdamW optimizer, using a cosine learning rate scheduler with a peak learning rate at $2 e \mathrm { ~ - ~ } 4$ . The learning rate for transformer layers is scaled by 0.1. In the training objectives, Ldistill includes the MSE loss between the rendered depth and the pseudo-label depth, as well as the Huber loss between the camera pose encoding and its pseudo-label. The corresponding weights $\lambda _ { 1 }$ contain two terms and are set to 0.1 and 10.0, respectively. The weight $\lambda _ { 2 }$ for Linst is set to 0.05.

## 4.2. Evaluation Details

To jointly evaluate the quality of reconstruction and semantic embedding, we evaluate reconstruction quality via novel view synthesis and introduce open-vocabulary semantic segmentation under novel views to evaluate the semantic alignment with language. Reconstruction performance is measured using standard reconstruction metrics, including PSNR, SSIM, and LPIPS, while open-vocabulary segmentation is evaluated with mean Intersection-over-Union (mIoU) and mean Location Accuracy (mAcc) following LangSplat [20]. We conduct the evaluation on the commonly used ScanNet [6] and ScanNet++[34] datasets.

## 4.3. Baselines

We compare our method with previous methods for language-embedded Gaussian reconstruction, including both feed-forward and per-scene optimization methods. The feed-forward methods include LSM [7], Uni3R [28], along with the 2D semantic segmentation model LSeg [13] used in them, which lacks novel-view synthesis capability and thus requires ground-truth novel views as input. These feed-forward methods are mainly designed for sparse-view inputs, typically with fixed and limited numbers of views.

<table><tr><td rowspan="2">Method</td><td colspan="5">2 views</td><td colspan="5">8 views</td></tr><tr><td>mIoUâ</td><td>mAccâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>mIoUâ</td><td>mAccâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>LSeg [13]</td><td>39.88</td><td>58.81</td><td>-</td><td>-</td><td>-</td><td>39.82</td><td>59.93</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LSMM [7]</td><td>38.25</td><td>58.85</td><td>21.45</td><td>0.729</td><td>0.382</td><td>37.22</td><td>60.59</td><td>19.25</td><td>0.680</td><td>0.497</td></tr><tr><td>Uni3R [28]</td><td>38.48</td><td>57.83</td><td>22.79</td><td>0.755</td><td>0.311</td><td>32.35</td><td>58.22</td><td>16.02</td><td>0.607</td><td>0.574</td></tr><tr><td>FLEG (Ours)</td><td>47.12</td><td>72.39</td><td>24.20</td><td>0.776</td><td>0.257</td><td>47.56</td><td>73.90</td><td>23.51</td><td>0.782</td><td>0.293</td></tr><tr><td rowspan="2">Method</td><td colspan="5">16 views</td><td colspan="5">32 views</td></tr><tr><td>mIoUâ</td><td>mAccâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>mIoUâ</td><td>mAccâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>LSeg [13]</td><td>39.71</td><td>58.51</td><td>-</td><td>-</td><td>-</td><td>40.07</td><td>57.68</td><td>-</td><td>-</td><td>-</td></tr><tr><td> LSM [7]</td><td>36.37</td><td>59.65</td><td>18.18</td><td>0.656</td><td>0.532</td><td>35.25</td><td>59.11</td><td>16.85</td><td>0.629</td><td>0.563</td></tr><tr><td>Uni3R [28]</td><td>30.68</td><td>56.97</td><td>15.21</td><td>0.591</td><td>0.590</td><td>20.45</td><td>50.31</td><td>10.55</td><td>0.412</td><td>0.648</td></tr><tr><td>FLEG (Ours)</td><td>47.59</td><td>74.02</td><td>23.36</td><td>0.776</td><td>0.313</td><td>46.91</td><td>74.28</td><td>23.27</td><td>0.773</td><td>0.326</td></tr></table>

Table 1. Comparison with feed-forward methods on the ScanNet dataset. Our FLEG method consistently outperforms baselines in both reconstruction quality and open-vocabulary query accuracy across all input view settings, and maintains stable performance as the number of views increases, demonstrating the robustness and effectiveness of our method.

<!-- image-->  
Figure 3. Qualitative comparisons with feed-forward methods on the ScanNet dataset under sparse-view inputs.

For multi-view inputs, since we aim to evaluate the capabilities of a single model for both 3D reconstruction and semantic embedding under different numbers of input views, we use the same model for each feed-forward baseline across different views. For methods [7, 28] that only accept a fixed number of input views, we adopt a post-processing strategy following DUSt3R [32], which serves as the backbone of these methods, making the comparison reasonable and relatively fair. The per-scene optimization methods include LangSplat [20], which is designed for dense-view input and based on original 3DGS to further embed semantics, and LangScene-X [16], which leverages video frame interpolation to obtain dense-view inputs for optimization and semantic embedding. In contrast, our FLEG requires only a single unified model to perform feed-forward reconstruction from uncalibrated and unposed inputs of arbitrary views, surpassing methods specifically designed for either sparse- or dense-view settings.

## 4.4. Quantitative Comparisons

We compare FLEG against existing feed-forward baselines on novel view synthesis and open-vocabulary segmentation using the ScanNet dataset. As shown in Tab. 1, our method consistently outperforms other feed-forward language-embedded baselines from 2-view to 32-view input settings on both reconstruction quality and segmentation accuracy. Moreover, while current feed-forward reconstruction methods suffer significant degradation as the number of input views increases, our performance remains stable, indicating that FLEG can robustly handle any view inputs ranging from sparse to dense views.

Under relatively dense settings with 16 and 32 input views, we further compare our method against per-scene optimized baselines on the ScanNet dataset in Tab. 2. While these baselines require over ten minutes for per-scene optimization, FLEG reconstructs the language-embedded Gaussians of a scene within 5 seconds, achieving higher accuracy in open-vocabulary querying and demonstrating a marked efficiency advantage.

<table><tr><td rowspan="2">Method</td><td colspan="3">16 views</td><td colspan="3">32 views</td></tr><tr><td>Timesâ</td><td>mIoUâ</td><td>mAccâ</td><td>Timesâ</td><td>mIoUâ</td><td>mAccâ</td></tr><tr><td>LangSplat [20]</td><td>â 10min</td><td>31.24</td><td>63.39</td><td>â 15min</td><td>25.05</td><td>41.90</td></tr><tr><td>LangScene-X [16]</td><td>â 15min</td><td>19.61</td><td>37.89</td><td>â 20min</td><td>17.30</td><td>38.09</td></tr><tr><td>FLEG (Ours)</td><td>1.89s</td><td>44.69</td><td>73.11</td><td>3.20s</td><td>46.56</td><td>79.52</td></tr></table>

Table 2. Comparison with per-scene optimized methods on the ScanNet dataset under dense-view inputs.

## 4.5. Qualitative Comparisons

In Fig. 3, we present the qualitative comparison with feedforward methods under sparse input views (2-view and 8- view) on reconstruction quality and open-vocabulary query accuracy. Existing baselines exhibit a marked decline in reconstruction performance beyond two views and fail to capture fine-grained object details, such as the laptop. In contrast, our method maintains high reconstruction and delivers more accurate language-guided queries.

We also present the qualitative comparison with perscene optimized methods under relatively dense input views (16-view and 32-view) on open-vocabulary query accuracy in Tab. 2. Our method requires only seconds of feedforward inference and still outperforms per-scene optimized baselines in open-vocabulary querying accuracy, demonstrating the potential for large-scale data generation and real-time downstream applications.

<!-- image-->  
Figure 4. Qualitative comparisons with per-scene optimized methods on the ScanNet dataset under dense-view input.

## 4.6. Ablation Studies

We conduct ablation studies on the ScanNet++ [34] dataset in Tab. 3. The results show that the distillation loss substantially improves reconstruction quality, while the semantic embedding components effectively incorporate languagealigned features. The integration of these designs is essential for simultaneously achieving and effectively reconstructing and semantic embedding.

<table><tr><td colspan="4">Ablation for Reconstruction Module</td></tr><tr><td></td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Ours w/o distill loss</td><td>10.99</td><td>0.259</td><td>0.722</td></tr><tr><td>Ours</td><td>21.60</td><td>0.737</td><td>0.309</td></tr><tr><td colspan="4">Ablation for Language Embedding Module</td></tr><tr><td></td><td>mIoUâ</td><td>mAccâ</td><td>Emed. Numâ</td></tr><tr><td>Ours w/o inst loss</td><td>39.01</td><td>68.98</td><td>2.6M</td></tr><tr><td>Ours w/o sparse</td><td>43.43</td><td>72.71</td><td>6.4M</td></tr><tr><td>Ours</td><td>44.58</td><td>73.14</td><td>2.6M</td></tr></table>

Table 3. Ablation studies on ScanNet++ dataset.

## 5. Conclusion

In this paper, we propose FLEG, a feed-forward network to reconstruct language-embedded Gaussians from any uncalibrated and unposed multi-view images. FLEG enables efficient semantic embedding while maintaining high-quality geometric reconstruction and high-fidelity rendering, supporting simultaneous 3D reconstruction and semantic understanding. We propose a 3D-annotation-free training framework to train the network and thereby construct a large-scale multi-view image dataset with easily obtained 2D instance information to enrich semantic embedding.

The 2D semantics are further aligned with the 3D representation through an instance-guided contrastive learning. Our FLEG generates language-embedded Gaussians from any views in a single feed-forward pass and supports both sparse and dense view inputs. Through effectively integrating geometry- and appearance-focused 3D reconstruction with language-aligned semantics in a more practical manner, FLEG enables more effective interaction with the 3D environment and facilitates the perception and understanding of the 3D world via joint 3D representations.

## References

[1] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sunderhauf, Ian Reid, Stephen Gould, and Â¨ Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3674â3683, 2018. 1

[2] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Yuri Feigin, Peter Fu, Thomas Gebauer, Daniel Kurz, Tal Dimry, Brandon Joffe, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. In Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021. 4

[3] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19457â19467, 2024. 2, 4

[4] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. arXiv preprint arXiv:2403.14627, 2024. 2, 4

[5] Alan B Craig. Understanding augmented reality: Concepts and applications. 2013. 1

[6] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 4, 6

[7] Zhiwen Fan, Jian Zhang, Wenyan Cong, Peihao Wang, Renjie Li, Kairun Wen, Shijie Zhou, Achuta Kadambi, Zhangyang Wang, Danfei Xu, et al. Large spatial model: End-to-end unposed images to semantic 3d. Advances in Neural Information Processing Systems (NeurIPS), 37: 40212â40229, 2025. 3, 7

[8] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. In IEEE International Conference on Robotics and Automation (ICRA), pages 10608â10615. IEEE, 2023. 1

[9] Yuzhou Ji, He Zhu, Junshu Tang, Wuyi Liu, Zhizhong Zhang, Xin Tan, and Yuan Xie. Fastlgs: Speeding up lan-

guage embedded gaussians with feature grid mapping. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2025. 2, 3

[10] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. arXiv preprint arXiv:2505.23716, 2025. 2, 4, 5

[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139:1â 139:14, 2023. 2

[12] Vincent Leroy, Yohann Cabon, and JerÂ´ ome Revaud. Ground- Ë ing image matching in 3d with mast3r. In European Conference on Computer Vision (ECCV), pages 71â91. Springer, 2024. 2

[13] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic seg- Â´ mentation. arXiv preprint arXiv:2201.03546, 2022. 7

[14] Qijing Li, Jingxiang Sun, Liang An, Zhaoqi Su, Hongwen Zhang, and Yebin Liu. Semanticsplat: Feed-forward 3d scene understanding with language-aware gaussian fields. arXiv preprint arXiv:2506.09565, 2025. 2, 3

[15] Wanhua Li, Yujie Zhao, Minghan Qin, Yang Liu, Yuanhao Cai, Chuang Gan, and Hanspeter Pfister. Langsplatv2: Highdimensional 3d language gaussian splatting with 450+ fps. arXiv preprint arXiv:2507.07136, 2025. 3

[16] Fangfu Liu, Hao Li, Jiawei Chi, Hanyang Wang, Minghui Yang, Fudong Wang, and Yueqi Duan. Langscene-x: Reconstruct generalizable 3d language-embedded scenes with trimap video diffusion. arXiv preprint arXiv:2507.02813, 2025. 2, 8

[17] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 5

[18] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 2

[19] Maxime Oquab, Timothee Darcet, Th Â´ eo Moutakanni, Huy Â´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3

[20] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20051â 20060, 2024. 2, 3, 6, 8

[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748â8763. PmLR, 2021. 4

[22] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi- Â´ sion transformers for dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12179â12188, 2021. 4

[23] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Seg- Â¨ ment anything in images and videos. In International Conference on Learning Representations (ICLR), 2025. 4

[24] William Shen, Ge Yang, Alan Yu, Jansen Wong, Leslie Pack Kaelbling, and Phillip Isola. Distilled feature fields enable few-shot language-guided manipulation. In Conference on Robot Learning (CoRL), pages 405â424. PMLR, 2023. 1

[25] Yu Sheng, Jiajun Deng, Xinran Zhang, Yu Zhang, Bei Hua, Yanyong Zhang, and Jianmin Ji. Spatialsplat: Efficient semantic 3d from sparse unposed images. arXiv preprint arXiv:2505.23044, 2025. 2, 3

[26] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for openvocabulary scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5333â5343, 2024. 3

[27] Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrishnan, Kuang-Huei Lee, Quan Vuong, Paul Wohlhart, Sean Kirmani, Brianna Zitkovich, Fei Xia, et al. Open-world object manipulation using pre-trained vision-language models. In Conference on Robot Learning (CoRL), pages 3397â3417. PMLR, 2023. 1

[28] Xiangyu Sun, Haoyi Jiang, Liu Liu, Seungtae Nam, Gyeongjin Kang, Xinjie Wang, Wei Sui, Zhizhong Su, Wenyu Liu, Xinggang Wang, et al. Uni3r: Unified 3d reconstruction and semantic understanding via generalizable gaussian splatting from unposed multi-view images. arXiv preprint arXiv:2508.03643, 2025. 2, 3, 7

[29] Qijian Tian, Xin Tan, Jingyu Gong, Yuan Xie, and Lizhuang Ma. Uniforward: Unified 3d scene and semantic field reconstruction via feed-forward gaussian splatting from only sparse-view images. arXiv preprint arXiv:2506.09378, 2025. 2, 3

[30] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 3, 4, 6

[31] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[32] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20697â20709, 2024. 2, 7

[33] Xingrui Wang, Cuiling Lan, Hanxin Zhu, Zhibo Chen, and Yan Lu. Gsemsplat: Generalizable semantic 3d gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2412.16932, 2024. 2, 3

[34] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias NieÃner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12â22, 2023. 4, 6, 8

[35] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21676â21685, 2024. 2

[36] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. ACM Transactions on Graphics (TOG), 37(4):1â12, 2018. 4