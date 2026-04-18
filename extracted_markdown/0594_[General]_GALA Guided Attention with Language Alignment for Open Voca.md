# GALA: Guided Attention with Language Alignment for Open Vocabulary Gaussian Splatting

Elena Alegret1,2,6\* Kunyi Li1,4\* Sen Wang1,4 Siyun Liang5 Michael Niemeyer3 Stefano Gasperini1,4,7 Nassir Navab1,4 Federico Tombari1,3

1Technical University of Munich 2Universitat Politecnica de Catalunya \` 3Google 4Munich Center for Machine Learning 5University of Tubingen 6ETH Zurich 7aser Visualais

<!-- image-->

<!-- image-->  
Figure 1. We present GALA, a novel 3DGSâbased framework for open-vocabulary scene understanding. It delivers strong performance in both 2D and 3D open-vocabulary queries, while preserving high intra-instance feature consistency to boost segmentation quality.

## Abstract

3D scene reconstruction and understanding have gained increasing popularity, yet existing methods still struggle to capture fine-grained, language-aware 3D representations from 2D images. In this paper, we present GALA, a novel framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). GALA distills a scenespecific 3D instance feature field via self-supervised contrastive learning. To extend to generalized language feature fields, we introduce the core contribution of GALA, a crossattention module with two learnable codebooks that encode view-independent semantic embeddings. This design not only ensures intra-instance feature similarity but also supports seamless 2D and 3D open-vocabulary queries. It reduces memory consumption by avoiding per-Gaussian high-dimensional feature learning. Extensive experiments on real-world datasets demonstrate GALAâs remarkable open-vocabulary performance on both 2D and 3D.

## 1. Introduction

Understanding 3D scenes is a central challenge for 3D computer vision (3DV), with wide-ranging applications in autonomous driving [3, 8, 40, 41], robotics [1, 11, 19, 24], and augmented or virtual reality [5, 34]. Open-vocabulary scene understanding not only enables robotics to perceive and reason about the world but also opens new possibilities for intuitive human-robotic interaction, allowing users to explore and query scenes through natural language. This integration of spatial understanding with language grounding represents a promising direction towards more intelligent systems.

Neural Radiance Fields (NeRF) [2, 25, 26, 43] offer the potential to store additional semantic information within the field. Several methods [14, 15, 45] extend NeRF by distilling semantic or language features from 2D images. However, NeRF-based methods suffer from inefficient encoding and high computational costs for training and rendering. 3D Gaussian Splatting (3DGS) [9, 13, 20, 23, 27, 46] provides an explicit and more efficient alternative with a set of 3D Gaussian shape primitives. Subsequent works [12, 21, 31, 32, 47] incorporate feature attributes into these Gaussians, enabling semantic feature rasterization and language-based interactions.

Robotic systems with limited computing resources, for example navigation, is often performed in 2D, even though the robots operate in a 3D world. Therefore, 2D perception remains essential. Recent works [31, 47] address this by distilling high-dimensional 2D language features [16, 33] into 3D Gaussians through compressing high-dimensional language into low-dimensional representations, followed by novel view synthesis to enable arbitrary-view 2D openvocabulary querying. However, such compression inevitably leads to information loss, and their segmentation results exhibit low intra-instance consistency and blurred object boundaries, which hinder accurate semantic segmentation.

Instead of prioritizing efficient 2D open-vocabulary segmentation, some works focus on enhancing 3D scene semantics. However, storing high-dimensional language features for each Gaussian is time- and memory-intensive. Recent methods [18, 42] address this with clustering: Gaussians are grouped into clusters, each assigned a lowdimensional scene-specific feature, and matched to preprocessed per-instance language features via 2Dâ3D associations. Yet, purely KNN-based clustering without explicit supervision can cause one cluster to span multiple instances or split a single instance, leading to misalignment and degraded segmentation performance. Others [6, 12] average multi-view language features without training, achieving strong 3D reconstruction but offering limited or memoryheavy 2D semantic rendering, making them unsuitable for real-time robotics and navigation.

Although reconstructing and perceiving the world in 3D is important and interacting with it in 2D is often the most efficient strategy for robotics [1, 10, 17], existing approaches tend to focus only on one side of the problem. We propose a Guided Attention method with Language Alignment Gaussian Splatting (GALA), a novel framework that enables both 2D and 3D open-vocabulary scene understanding demonstrating the broad applicability to diverse perception tasks, as illustrated in Figure 1. The key idea is to enforce instance-consistent semantics: instead of storing noisy or redundant per-Gaussian language features, GALA learns to associate each Gaussian with a shared instancelevel language embedding, ensuring that the semantics of one instance remain consistent not only across different spatial locations and viewpoints but also in 3D. Our main contributions can be summarized as follows:

â¢ We propose to store per-instance semantics via codebooks, associating each instance with a language embedding and ultimately generating intra-instance consistent semantic features for better segmentation.

â¢ By employing an attention mechanism that maps each Gaussian feature to its corresponding instance, we enable effective 2D and 3D open-vocabulary segmentation.

â¢ We improve the segmentation with an attention-weighted entropy loss, which encourages a clear one-to-one mapping between Gaussian instance features and codebook embeddings.

Extensive experiments on public real-world datasets, LERF-OVS [14] and ScanNet-v2 [7], demonstrate the effectiveness of GALA on both 2D and 3D semantic segmentation and open-vocabulary localization compared to the state-of-the-art. The code and models will be released upon acceptance.

## 2. Related Works

## 2.1. Zero-Shot 2D Scene Understanding

The success of 2D visual foundation models has been demonstrated across a wide range of vision tasks, which enhances both perceptual and reasoning abilities. CLIP [33] aligns image and text features through contrastive learning, enabling robust cross-modal understanding in a shared embedding space. DINO [28], a self-supervised Vision Transformer, learns rich semantic representations from unlabeled images, capturing object boundaries and scene layouts. Building on these models, Grounding DINO [22] extends DINO with open-vocabulary detection capabilities guided by textual queries, through tight visual-language fusion. SAM [16], a promptable segmentation model, enables zero-shot instance segmentation with impressive generalization. Grounded SAM [35] combines SAM with Grounding DINO to support arbitrary text-driven semantic segmentation and detection. APE [36] introduces a unified visual perception framework for tasks like segmentation and grounding, using lightweight visual-language fusion for efficient and generalizable performance. However, these powerful models are inherently limited to 2D image understanding, restricting their applicability in tasks requiring holistic 3D scene understanding.

## 2.2. Open-Vocabulary 3D Scene Understanding

Understanding 3D scenes requires consistent semantic reasoning across multiple views and spatial dimensions. Recent efforts have explored transferring powerful language features from 2D models into 3D representations to allow robots to perceive the world like humans. OpenScene [29] distills CLIP features into 3D point clouds for zero-shot segmentation and language queries, but suffers from limited spatial resolution and reduced generalization due to pointbased representation. More recent methods [4, 14, 15, 45] integrate semantics into continuous neural radiance field by distilling 2D language features, enabling open-vocabulary 3D understanding. However, NeRFs remain slow to render, depend heavily on high-quality 2D masks, and struggle with scalability due to volumetric computation.

In contrast, 3D Gaussian Splatting (3DGS) provides an explicit and efficient representation better suited for realtime 3D understanding. LangSplat [31] applies hierarchical feature distillation by assigning each Gaussian a lowdimensional feature that is rasterized into a 2D feature map. A pretrained autoencoder is used to compress highdimensional language features for supervision. Similarly, Feature3DGS [47] leverages a convolutional neural network (CNN) for feature dimension lifting. While both methods reduce the dimensionality of the supervision signal, this compression inevitably causes information loss. Furthermore, they learn per-Gaussian semantic features without enforcing intra-instance feature consistency, which may lead to ambiguous object representations and hinder robotic interaction and navigation. OpenGaussian [42] and Instance-Gaussian [18] place greater emphasis on 3D awareness by enabling point-level 3D segmentation through hierarchical feature clustering and 3Dâ2D feature association, mapping scene-specific instance features to language features. However, misalignment in this mapping can cause significant performance drops.

Rather than training a semantic feature field per scene, Dr. Splat [12] and Occamâs LGS [6] propose an aggregation method that averages multi-view language features in a single forward pass, greatly improving efficiency. Although these methods improve 3D semantic reconstruction, generating accurate 2D semantic maps remains crucial for robotics, enabling fast and reliable perception from onboard camera images. SuperGSeg [21] clusters thousands of Gaussians into SuperGaussians sharing language embeddings, enabling efficient high-dimensional feature rendering and improving performance. However, its MLP-based cluster update is complex and may lack semantic coherence, sometimes grouping irrelevant or noisy points. Moreover, the K-Nearest Neighbors (KNN)-based initialization depends on point density, so sparse regions can cause a Super-Gaussian to span multiple objects with conflicting semantics, degrading segmentation quality. GOI [32] and CCL-LGS [38] both introduce a single trainable feature codebook to store language embeddings and use a multi-layer perceptron (MLP) to predict discrete codebook indices for the rasterized 2D feature maps. While this approach compresses semantics spatially rather than dimensionally preserving semantic richness, the MLP applies fixed weights uniformly across all input elements, lacking the flexibility to dynamically prioritize important information. This limitation makes it less effective at capturing context-dependent relevance compared to attention mechanisms.

Therefore, we propose a dual-codebook design combined with a guided cross-attention module. Our method computes similarity scores for soft, continuous assignments between Gaussian features and codebook embeddings, enabling instance-level semantics in a differentiable manner.

Despite relying on 2D supervision, the fully linear attention and rasterization modules enhance generalization from 2D tasks to 3D tasks and effectively reduce the multi-view inconsistencies found in prior work.

## 3. Preliminaries

3D Gaussian Splatting (3DGS) [13] employs a set of 3D points to effectively render images from given viewpoints, each characterized by a Gaussian function with 3D mean $\mu _ { i } \in { \mathbb { R } } ^ { 3 }$ , covariance matrix $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ , opacity value $\alpha _ { i } \in$ R, RGB color value $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 }$ , and sometimes with feature value $\mathbf { m } _ { i } \in \mathbb { R } ^ { d }$ :

$$
\sigma _ { i } ( { \bf x } ) = \alpha _ { i } * \exp \left( - \frac { 1 } { 2 } ( { \bf x } - { \mu } _ { i } ) ^ { T } \Sigma _ { i } ^ { - 1 } ( { \bf x } - { \mu } _ { i } ) \right) .\tag{1}
$$

Given a 3D position $\mathbf { x } , \sigma _ { i } ( \mathbf { x } )$ represents current opacity value contributed by the i-th Gaussian. To facilitate optimization, $\Sigma _ { i } = R _ { i } \ : \dot { S _ { i } } S _ { i } ^ { T } R _ { i } ^ { T }$ is factorized into the product of a scaling matrix Si, represented by scale factors $\mathbf { s } _ { i } ~ \in ~ \mathbb { R } ^ { 3 }$ , and a rotation matrix Ri encoded by a quaternion $\mathbf { r } _ { i } \in \mathbb { R } ^ { 4 }$ . Color value $\hat { \mathbf { C } } ( \mathbf { u } )$ and feature value $\hat { \bf M } ( { \bf u } )$ at pixel u are rendered by N projected and ordered Gaussians using point-based Î±-blending:

$$
\{ \hat { \mathbf { C } } , \hat { \mathbf { M } } \} ( \mathbf { u } ) = \sum _ { i \in N } T _ { i } \boldsymbol { \sigma } _ { i } \times \{ \mathbf { c } _ { i } , \mathbf { m } _ { i } \} ,\tag{2}
$$

where $\begin{array} { r } { T _ { i } = \prod _ { i = 1 } ^ { i - 1 } ( 1 - \sigma _ { j } ) } \end{array}$ . Scaffold-GS [23] introduces a neural variant of 3DGS by voxelizing a set of point clouds as anchor points $V \in \mathbb { R } ^ { \bar { N } \times 3 }$ . Each anchor point $\mathbf { v } _ { i } ~ \in ~ V$ is associated with a feature $\mathbf { f } _ { i } \in \mathbb { R } ^ { d }$ , scaling factor $l _ { i } \in \mathbb { R } ^ { 3 }$ and K learnable offsets $\{ \mathcal { O } _ { i , k } \ \in \ \mathbb { R } ^ { 3 } \ | \ k = \ 0 , \ldots , K -$ 1}. Then K neural Gaussians $\left\{ \mu _ { i , 0 } , . . . , \mu _ { i , K - 1 } \right\} = \mathbf { v } _ { i } +$ $\{ \mathcal { O } _ { i , 0 } , . . . , \mathcal { O } _ { i , K - 1 } \} \cdot l _ { i }$ are generated from a given anchor point xv. The remaining attributes of each Gaussian ${ \bf g } _ { i } \in  { }$ $\{ { \alpha } _ { i , k } , { \bf { c } } _ { i , k } , { R } _ { i , k } , { S } _ { i , k } , { \bf { m } } _ { i , k } \}$ are predicted as:

$$
\{ \mathbf { g } _ { i , 0 } , \dots , \mathbf { g } _ { i , K - 1 } \} = \mathcal { F } _ { \mathbf { g } } ( \mathbf { f } _ { i } , \delta _ { \mathbf { i } } , \vec { \mathbf { d } _ { i } } ) ,\tag{3}
$$

where $\begin{array} { r } { \delta _ { i } = \| \mathbf { v } _ { i } - \mathbf { x } _ { c } \| , \vec { \mathbf { d } } _ { i } = \frac { \mathbf { v } _ { i } - \mathbf { x } _ { c } } { \| \mathbf { v } _ { i } - \mathbf { x } _ { c } \| } , \mathbf { x } _ { c } } \end{array}$ is the camera center, and $\mathcal { F } _ { \mathbf { g } }$ is corresponding attribute decoder.

## 4. Methods

As shown in Figure 2, our method builds on neural Gaussian Splatting [23] with two-stage training: (1) self-supervised reconstruction of scene geometry and a scene-specific instance feature field, and (2) rendering these features to 2D and mapping them to generalized language features via guided attention with dual learnable codebooks. The linear attention design enables seamless segmentation in both 2D and 3D using only 2D training, while the per-instance codebooks and attention-weights entropy loss enforce oneto-one mappings, enhancing intra-instance feature consistency.

<!-- image-->  
Figure 2. Overview of GALA. In Stage 1, we reconstruct the 3D scene and distill a scene-specific feature field in a self-supervised manner. In Stage 2, a rasterized instance feature map is used to train a Guided Attention module, which learns to map the scene-specific feature field to a generalized language field via two learnable codebooks. During inference (right), GALA supports open-vocabulary querying and segmentation in both 2D (top) and 3D (bottom).

## 4.1. Scene Reconstruction and Feature Distillation

SuperGSeg [21] builds on Scaffold-GS [23] to perform joint 3D reconstruction and scene-specific instance and hierarchical feature distillation, where these features are used for clustering. In contrast, our method does not rely on hierarchical features for clustering and instead focuses solely on instance learning in a self-supervised manner. Consequently, each anchor point in our method is assigned to two types of features. Using Eq. 3, a geometry feature $\mathbf { f } _ { i } ^ { g } ~ \in ~ \mathbb { R } ^ { d _ { g } }$ is decoded into K Gaussian attributes $\{ \alpha _ { i , k } , \mathbf { c } _ { i , k } , R _ { i , k } , S _ { i , k } \}$ , and a segmentation feature $\mathbf { f } _ { i } ^ { s } \in$ $\mathbb { R } ^ { d _ { \mathrm { s c g } } }$ is decoded into K instance features $\mathbf { m } _ { i , k } ~ \in ~ \mathbb { R } ^ { d _ { \mathrm { i n s } } }$ These attributes and features are then rasterized as a rendered color map $\hat { \mathbf { C } } \in \mathbb { R } ^ { H \times W \times 3 }$ and a instance feature map $\hat { \mathbf { M } } \in \mathbb { R } ^ { H \times W \times \bar { d } _ { \mathrm { i n s } } }$ through Eq. 2.

LangSplat [31] and Feature3DGS [47] learn per-Gaussian semantic features independently, without any instance-level constraints. Our method adopts a selfsupervised method [42, 45] to distill a scene-specific instance field with instance contrastive learning. We first generate a set of instance masks $\{ m _ { i } \in \mathbb { R } \mid i = 0 , \ldots , \mathcal { M } \}$ for each view using Segment Anything Model (SAM) [16]. For a given instance mask $m _ { i }$ , we denote each pixel feature within the mask as $\{ \hat { m } _ { i , j } \in \mathbb { R } ^ { d _ { \mathrm { i n s } } } \mid j = 1 , \ldots , n \}$ , where subscript i denotes the mask index and subscript j denotes the pixel index. We compute the mean feature value within the mask as $\bar { m } _ { i } \in \mathbb { R } ^ { d _ { \mathrm { i n s } } }$ s . To distill the 3D instance field in a self-supervised manner and enhance intra-instance feature similarity, we employ contrastive learning to pull features within the same mask closer together, while pushing features from different masks further apart:

$$
\mathcal { L } _ { \mathrm { i n s } } = \frac { 1 } { \mathcal { M } } \sum _ { i = 1 } \sum _ { j = 1 } - \log \frac { \exp ( \hat { m } _ { i , j } \cdot \bar { m } _ { i } / \tau _ { i } ) } { \sum _ { q \neq i } ^ { \mathcal { M } } \exp ( \hat { m } _ { i , j } \cdot \bar { m } _ { q } / \tau _ { q } ) } .\tag{4}
$$

Therefore, the overall objective function for the first stage is:

$$
\begin{array} { r } { \mathcal { L } _ { 1 } = \mathcal { L } _ { \mathrm { R G B } } + \lambda _ { \mathrm { i n s } } \mathcal { L } _ { \mathrm { i n s } } , } \end{array}\tag{5}
$$

where $\lambda _ { \mathrm { i n s } }$ is the penalty coefficient. And $\begin{array} { r l } { \mathcal { L } _ { \mathrm { R G B } } } & { { } = } \end{array}$ $0 . 8 \times | \mathbf { C } - \hat { \mathbf { C } } | + 0 . 2 \times S S I M ( \mathbf { C } - \hat { \mathbf { C } } )$ is the photometric loss [13] where C is the ground-truth color image.

## 4.2. Semantic Codebooks

Prior works such as OpenGaussian [42], InstanceGaussian [18], and SuperGSeg [21] adopt a bottom-up approach: they first cluster low-level features to form clusters and then learn instance-level segmentation by aggregating these clusters. However, this strategy can lead to several issues, including over-segmentation, one single cluster representing multiple distinct objects, or different parts of the same object being assigned to separate clusters. To address this, we introduce a codebook module designed to represent each instance in the scene with a unique embedding. A codebook consists of $N _ { c }$ learnable embeddings, where $N _ { c }$ approximates the number of instances in the scene. Specifically, we define an Instance Codebook $\mathcal { C } _ { i n s } \in R ^ { N _ { c } \times d _ { i n s } }$ , where each entry captures a distinct instance-level representation. In parallel, we define a Language Codebook $\mathcal { C } _ { l a n g } \in R ^ { N _ { c } \times d _ { c } }$ which stores language embeddings with a one-to-one correspondence to the entries in $\mathcal { C } _ { i n s t } .$ . Each codebook entry is intended to represent a unique instance in the scene. The proposed codebooks decouple semantics from spatial positions and allow for unambiguous, per-instance embedding assignments, ensuring intra-instance feature similarity.

## 4.3. Guided Attention with Codebooks

Perceiving the world through human language is a key goal of 3D scene understanding, for which a purely scenespecific feature field is insufficient. Prior works [18, 42] attempt to align low-dimensional features with highdimensional language semantics via 2Dâ3D associations, while others [31, 47] compress high-dimensional supervision to reduce overhead. However, these approaches are either designed for 3D or 2D segmentation tasks, suffering from information loss, or leading to limited generalization. Our method introduces a guided cross-attention module with codebooks proposed in Section 4.2 that maps scene-specific features to the generalized language field, enabling both 2D and 3D open-vocabulary queries.

Attention with Codebooks. An attention module [39] is adopted with learnable codebooks and residual connections. We use the rasterized instance feature map MË as the query Q, the instance codebook $\mathcal { C } _ { i n s }$ as the key K and the language codebook $\mathcal { C } _ { l a n g }$ as the value V :

$$
\begin{array} { r } { \hat { \bf A } = A ( \hat { \bf M } ) = A t t n ( Q , K , V ) + Q \in \mathbb { R } ^ { H W \times d _ { c } } , } \end{array}\tag{6}
$$

$$
\hat { \mathbf { L } } = \mathcal { F } _ { l i f t } ( \hat { \mathbf { A } } ) \in \mathbb { R } ^ { H W \times d _ { l a n g } } ,\tag{7}
$$

where $Q = { \mathcal { N } } ( { \hat { \mathbf { M } } } \times W ^ { Q } ) , K = { \mathcal { N } } ( C _ { \mathrm { i n s } } ) , V = { \mathcal { N } } ( C _ { \mathrm { l a n g } } ) ,$ N represents layer normalization opertor which is applied to avoid scale discrepancies, W Q is the linear transformation which is applied to project the original input into a same space of instance codebook and $\mathcal { F } _ { l i f t }$ is an MLP to lift the feature dimensionality. During training, we apply only 2D supervision with cosine similarity loss between the predicted language map LË and the preprocessed ground-truth language map L:

$$
\mathcal { L } _ { \mathrm { l a n g } } = 1 - \cos ( \hat { \mathbf { L } } , \mathbf { L } ) .\tag{8}
$$

It is worth noting that the attention operation A defined in Eq. 6 is linear. As the rasterization in Eq. 2 involves a weighted summation, applying A to the 2D rasterized features is mathematically equivalent to applying it directly to the underlying 3D Gaussians:

$$
\mathcal { A } ( \hat { \mathbf { M } } ) = \mathcal { A } \big ( \sum T _ { i } \sigma _ { i } \times \mathbf { m } _ { i } \big ) = \sum T _ { i } \sigma _ { i } \times \mathcal { A } ( \mathbf { m } _ { i } ) .\tag{9}
$$

This property allows us to train the codebooks solely using 2D feature maps as supervision, and during inference, however, the same model can be directly applied to the 3D Gaussians, enabling open-vocabulary semantic queries in both 2D and 3D space. By compacting per-Gaussianâs semantic features into per-instance embeddings, we not only reduce training costs but also enforce intra-instance feature consistency in both 2D and 3D.

Probability Guidance. OpenGaussian [42] adopts twolevel clustering with positional embedding to model instance-level representations. However, without explicit supervision, it struggles to establish a one-to-one correspondence between instances and clusters. Our method leverages attention weights to guide a clear one-to-one mapping between instances and codebook embeddings. The attention weights:

$$
\begin{array} { r } { P = s o f t m a x ( Q K ^ { \top } / \sqrt { d _ { i n s } } ) \in \mathbb { R } ^ { H W \times N _ { c } } , } \end{array}\tag{10}
$$

indicate the relevance probability of each codebook embedding with respect to each feature query. To encourage a one-to-one correspondence, we apply the entropy loss on the attention weights:

$$
\mathcal { L } _ { \mathrm { e n t r o p y } } = - \sum _ { j = 1 } p _ { j } \log ( p _ { j } ) ,\tag{11}
$$

where $p _ { j } ~ \in ~ P$ represents the attention probability distribution over the $N _ { c }$ codebook entries for feature query j. This enforces the probability distribution for each query to be unimodal, meaning each query is associated with a single codebook embedding, ensuring that each instance corresponds to only one embedding in the codebook.

Therefore, the overall objective function of the second stage is:

$$
\begin{array} { r } { \mathcal { L } _ { 2 } = \mathcal { L } _ { \mathrm { l a n g } } + \lambda _ { \mathrm { e n t } } \mathcal { L } _ { \mathrm { e n t r o p y } } , } \end{array}\tag{12}
$$

where $\lambda _ { \mathrm { e n t } }$ is the penalty coefficient.

## 5. Experiments

## 5.1. Experimental Setup

Datasets. We comprehensively evaluate our method on two real-world datasets: ScanNet-v2 [7] and LERF-OVS [14]. Following OpenGaussian [42], 8 scenes are selected from the ScanNet-v2.

Baselines. We compare our method in both 2D and 3D with LERF [14], LangSplat [31], Feature-3DGS [47], GS-Grouping [44], LEGaussians [37], GOI [32], SuperGSeg [21] and OpenGaussian [42].

Metrics. We follow common practice and report openvocabulary segmentation and object selection evaluation with mean Intersection-over-Union (mIoU) for segmentation accuracy and mean accuracy (mAcc) for localization accuracy. While understanding the world in 3D is essential, perceiving it in 2D offers a more efficient pathway for real-time performance in robotics. Therefore, we report our performance both in 2D and 3D to demonstrate the broad applicability of our method to diverse perception tasks.

<!-- image-->  
Figure 3. Qualitative Results of 2D Open-Vocabulary Query. We visualize 2D Open-Vocabulary query results on LERF-OVS dataset [14]. LangSplat fails to localize objects accurately, leading to mismatching or incomplete masks. Our method delivers precise and consistent queries across diverse queries.

<table><tr><td colspan="2"></td><td colspan="2">Mean</td><td colspan="2">Figurines</td><td colspan="2">Teatime</td><td colspan="2">Ramen</td><td colspan="2">Waldo_kitchen</td></tr><tr><td>Eval.</td><td>Method</td><td>mIoU â</td><td>mAcc â</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td rowspan="8">2D</td><td>LERF [14]</td><td>37.40</td><td>73.60</td><td>38.60</td><td>75.00</td><td>45.00</td><td>84.80</td><td>28.20</td><td>62.00</td><td>37.90</td><td>72.70</td></tr><tr><td>LangSplat [31]</td><td>51.40</td><td>84.30</td><td>44.70</td><td>80.40</td><td>65.10</td><td>88.10</td><td>51.20</td><td>73.20</td><td>44.50</td><td>95.50</td></tr><tr><td>Feature-3DGS [47]</td><td>45.70</td><td>77.00</td><td>58.80</td><td>77.20</td><td>40.50</td><td>73.40</td><td>43.70</td><td>69.80</td><td>39.60</td><td>87.60</td></tr><tr><td>GS-Grouping [44]</td><td>46.30</td><td>76.50</td><td>60.90</td><td>75.00</td><td>40.00</td><td>74.30</td><td>45.50</td><td>68.60</td><td>38.70</td><td>88.20</td></tr><tr><td>LEGaussians [37]</td><td>46.90</td><td>77.20</td><td>60.30</td><td>75.60</td><td>40.80</td><td>75.20</td><td>46.00</td><td>67.50</td><td>39.40</td><td>90.30</td></tr><tr><td>GAGS [30]</td><td>54.12</td><td>81.66</td><td>53.59</td><td>78.57</td><td>60.29</td><td>88.14</td><td>46.81</td><td>69.01</td><td>55.80</td><td>90.91</td></tr><tr><td>GOI [32]</td><td>50.60</td><td>84.40</td><td>63.70</td><td>88.60</td><td>44.50</td><td>82.90</td><td>52.60</td><td>75.50</td><td>41.40</td><td>90.40</td></tr><tr><td>Ours</td><td>55.49</td><td>73.43</td><td>59.35</td><td>82.14</td><td>76.73</td><td>88.14</td><td>35.13</td><td>50.70</td><td>50.75</td><td>72.73</td></tr><tr><td rowspan="5">3D</td><td>LangSplat [31]</td><td>9.66</td><td>12.41</td><td>10.16</td><td>8.93</td><td>11.38</td><td>20.34</td><td>7.92</td><td>11.27</td><td>9.18</td><td>9.09</td></tr><tr><td>LEGaussians [37]</td><td>16.21</td><td>23.82</td><td>17.99</td><td>23.22</td><td>19.27</td><td>27.12</td><td>15.79</td><td>26.76</td><td>11.78</td><td>18.18</td></tr><tr><td>OpenGaussian [42]</td><td>38.36</td><td>51.43</td><td>39.29</td><td>55.36</td><td>60.44</td><td>76.27</td><td>31.01</td><td>42.25</td><td>22.70</td><td>31.82</td></tr><tr><td>SuperGSeg [21]</td><td>35.94</td><td>52.02</td><td>43.68</td><td>60.71</td><td>55.31</td><td>77.97</td><td>18.07</td><td>23.94</td><td>26.71</td><td>45.45</td></tr><tr><td>Ours</td><td>36.71</td><td>59.71</td><td>45.25</td><td>69.64</td><td>53.27</td><td>84.75</td><td>17.08</td><td>25.35</td><td>31.22</td><td>59.09</td></tr></table>

Table 1. 2D and 3D Evaluation on LERF-OVS. We report mIoU and mAcc on the LERF-OVS dataset [14]. Note that OpenGaussian [42] and SuperGSeg [21] by default do not report 2D evaluation. LERF [14], Feature-3DGS [47], GS-Grouping [44], GAGS [30] and GOI [32] by default do not support 3D evaluation.

Implementation Details. We perform single-GPU training (NVIDIA RTX 3090). For stage 1, we train 30,000 iterations with $\lambda _ { \mathrm { i n s } } = 0 . 0 0 1$ and for stage 2 we train 15,000 iterations with $\lambda _ { \mathrm { e n t } } = 1 0$ . We set the dimension of both instance codebook and language codebook as $d _ { i n s } = d _ { c } = 1 6$ . We use SAM [16] and CLIP [33] to preprocess the ground-truth language map, and set $d _ { l a n g } = 5 1 2$ . For more implementation details, please refer to the supp. mat..

## 5.2. 2D Evaluation

Table 1 presents the 2D results on the LERF-OVS dataset. We report both per-scene and average evaluations, where our method achieves the highest average mIoU (55.49%) among all existing approaches. In scenes with clearly separated objects, such as Teatime, our method delivers precise performance in both open-vocabulary segmentation and localization, achieving 76.73% mIoU and 88.14% mAcc. Our method also performs robustly in more cluttered environments like Waldo Kitchen, attaining 50.75% mIoU, where it better distinguishes complex domestic objects compared to LangSplat and GOI. Figure 3 demonstrates that our method can accurately distinguish the âGreen Appleâ without ambiguity, whereas LangSplat incorrectly selects both the green and red apples. Overall, our method achieves precise object segmentation with sharp and well-defined boundaries.

<!-- image-->  
Figure 4. Qualitative Results of 3D Open-Vocabulary Segmentation. We visualize the language feature point cloud on LERF-OVS [14] and ScanNet-v2 dataset [7] by compressing the features into the RGB point cloud. Note that the colors for visualization are consistent only within each method and not method-to-method.

## 5.3. 3D Evaluation

3D Evaluation on LERF-OVS. Following the evaluation protocol of LangSplat [31], Table 1 showcases the strong performance of our method in 3D segmentation and localization on the LERF-OVS dataset. Thanks to the linearity of the proposed attention module, our method, trained exclusively with 2D supervision, generalizes seamlessly to 3D tasks without any architectural modifications. Our method even outperforms the 3D-only method OpenGaussian on Figurines and Waldo kitchen, which, however, cannot easily produce 2D segmentation outputs. In contrast, the 2D-only method LangSplat struggles with 3D evaluation, as it is trained solely with 2D supervision and lacks

3D-aware segmentation. We also visualize the feature point cloud in Figure 4. Our method achieves both better geometry reconstruction and 3D segmentation compared with LangSplat [31] and OpenGaussian [42].

3D Evaluation on ScanNet-v2. Table 2 reports the 3D point cloud segmentation results on the ScanNet-v2 dataset, as ScanNet-v2 provides ground-truth semantic point cloud. We present the mean mIoU and mAcc across eight selected scenes containing different numbers of classes. Our method consistently outperforms OpenGaussian on all metrics, delivering strong 3D reconstruction and segmentation accuracy alongside high-quality 3D localization. Figure 4 visualizes the language-featured point clouds. By default, OpenGaussian does not densify Gaussians on ScanNet-v2, resulting in sparse features and lower appearance quality. Our method surpasses OpenGaussian both quantitatively and qualitatively.

## 5.4. Intra-Instance Feature Consistency

Previous methods learn semantic features per Gaussian or cluster, causing variations across positions and viewpoints. Our method addresses this issue in Stage 1 through instance-level contrastive learning, and further reinforces feature consistency via a per-instance codebook design. As shown in Figure 4, on the Teatime and Waldo kitchen scenes, the feature point clouds produced by LangSplat are highly noisy. On Scene0097 00, the door features from LangSplat are difficult to distinguish, and OpenGaussian oversegments the floor. Figure 5 visualizes the results with rendered 2D language feature maps. Our method yields homogeneous feature maps with well-defined boundaries.

<table><tr><td></td><td colspan="2">19 Classes</td><td colspan="2">15 Classes</td><td colspan="2">10 Classes</td></tr><tr><td>Method</td><td>mIoU â</td><td>mAcc â</td><td>mIoU â</td><td>mAcc â</td><td>mIoU â</td><td>mAcc â</td></tr><tr><td>LanSplat [31]</td><td>2.94</td><td>11.63</td><td>3.80</td><td>13.98</td><td>6.60</td><td>22.24</td></tr><tr><td>OpenGaussian [42]</td><td>15.47</td><td>26.04</td><td>17.42</td><td>28.82</td><td>23.46</td><td>37.73</td></tr><tr><td>OOurs</td><td>21.54</td><td>37.47</td><td>25.20</td><td>42.06</td><td>35.85</td><td>57.02</td></tr></table>

Table 2. 3D Evaluation on ScanNet-v2. We report the average 3D mIoU and mAcc on 8 scenes of the ScanNet-v2 dataset [7].

Teatime  
Ramen  
<!-- image-->

<!-- image-->  
Figure 5. Intra-Instance Feature Consistency. We visualize the rendered language feature map and show that our method provides a consistent intra-instance feature map and a clear boundary, which enhances the segmentation performance.

## 5.5. Ablation Study

All ablations are on Teatime of LERF-OVS [14].

Ablation on Codebook. Table 3 shows an ablation on the number of codes. Teatime contains around 64 instances; therefore, the best performance is achieved with 64 codes, matching the number of instances and enabling a near oneto-one mapping between embeddings and objects. Figure 6 shows that too few codes cause semantic ambiguity. The codebook size matching the expected number of instances achieves the best balance of accuracy and efficiency.

Ablation on Attention Module. We also report ablation on the structure of the proposed attention module. In Table 4, we show results that a) with lifting MLP Eq. 7 only, b) with attention Eq. 6 only and set the language codebook as $\mathcal { C } _ { l a n g } \in \ : R ^ { N _ { c } \times 5 \bar { 1 } 2 }$ , c) the attention together with lifting MLP but without residual connection $\mathcal { F } _ { l i f t } \big ( \boldsymbol { A } ( \boldsymbol { Q } , \boldsymbol { K } , \boldsymbol { V } ) \big )$ , e) our full model. We find that our full model achieves the best overall performance. In case b), applying the attention module directly without the MLP leads to high computational cost and convergence difficulties. Comparing cases c) and d), we observe that introducing a residual connection significantly improves training stability.

<!-- image-->  
GT Image

<!-- image-->  
Code 14 (??à¯ àµ 16)

<!-- image-->  
Code 14 (??à¯ àµ 64)

Figure 6. Ablation on the Number of Codes. We visualize an embedding from the codebook as the semantic mask. With codes number $N _ { c } ~ = ~ 1 6$ , code 14 represents both the sheep and plate in the teatime scene of LERF-OVS. With $N _ { c } = 6 4 .$ , our method clearly isolates the sheep, demonstrating improved instance separation.
<table><tr><td> $N _ { c } =$ </td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td>mIoU â</td><td>60.22</td><td>71.65</td><td>76.73</td><td>68.33</td></tr><tr><td>mAcc â</td><td>72.88</td><td>83.05</td><td>88.14</td><td>83.05</td></tr></table>

Table 3. Ablation on Number of Codes.

<table><tr><td>#</td><td>MLP</td><td>Attn</td><td>Res.</td><td>Prob.</td><td>mIoU â</td><td>mAcc â</td></tr><tr><td>a</td><td>â</td><td></td><td></td><td></td><td>72.60</td><td>86.44</td></tr><tr><td>b</td><td></td><td></td><td></td><td></td><td>35.48</td><td>42.37</td></tr><tr><td>c</td><td>â</td><td>&gt;&gt;</td><td></td><td></td><td>74.25</td><td>88.13</td></tr><tr><td>d)</td><td>â</td><td>â</td><td>â</td><td></td><td>75.02</td><td>86.44</td></tr><tr><td>e</td><td>â</td><td>â</td><td>â</td><td>â</td><td>76.73</td><td>88.14</td></tr></table>

Table 4. Ablation on Attention and Probability Guidance.

Ablation on Probability Guidance. In Table 4, we show results that d) without probability guidance. Our guided attention model is better able to assign distinct embeddings to separate object instances, leading to more accurate and interpretable segmentation. The right column of Figure 6 visualizes a selected embedding from our proposed codebook, showing that each embedding indeed captures meaningful instance-level semantics. For more ablations and runtime analysis, please refer to supp. mat..

## 6. Conclusions

We presented GALA, a framework for open-vocabulary 3D scene understanding using 3D Gaussian Splatting. By combining self-supervised instance-level feature distillation with a cross-attention module and learnable codebooks, GALA produces consistent, view-independent semantic embeddings, supports 2D and 3D open-vocabulary queries, and reduces memory usage. Experiments on realworld datasets demonstrate its effectiveness in generating reliable and efficient 3D and 2D feature representations.

## References

[1] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sunderhauf, Ian Reid, Stephen Gould, and Â¨ Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3674â3683, 2018. 1, 2

[2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855â5864, 2021. 1

[3] Holger Caesar, Varun Bankiti, Alex H. Lang, et al. nuscenes: A multimodal dataset for autonomous driving. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 1

[4] Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Wei Shen, Lingxi Xie, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, et al. Segment anything in 3d with nerfs. Advances in Neural Information Processing Systems, 36:25971â25990, 2023. 2

[5] Jiaqi Chen, Ruoxi Zhao, et al. Scenear: Learning to reconstruct 3d indoor scenes for augmented reality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1

[6] Jiahuan Cheng, Jan-Nico Zaech, Luc Van Gool, and Danda Pani Paudel. Occamâs lgs: An efficient approach for language gaussian splatting. arXiv preprint arXiv:2412.01807, 2024. 2, 3

[7] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828â5839, 2017. 2, 5, 7, 8, 1, 4

[8] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012. 1

[9] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1â11, 2024. 1

[10] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. arXiv preprint arXiv:2210.05714, 2022. 2

[11] Nathan Hughes, Yun Chang, Siyi Hu, Rajat Talak, Rumaia Abdulhai, Jared Strader, and Luca Carlone. Foundations of spatial perception for robotics: Hierarchical representations and real-time systems. The International Journal of Robotics Research, 43(10):1457â1505, 2024. 1

[12] Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. splat: Directly referring 3d gaussian splatting via direct language embedding registration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14137â14146, 2025. 1, 2, 3, 4

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 3, 4

[14] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19729â19739, 2023. 1, 2, 5, 6, 7, 8

[15] Chung Min Kim, Mingxuan Wu, Justin Kerr, Ken Goldberg, Matthew Tancik, and Angjoo Kanazawa. Garfield: Group anything with radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21530â21539, 2024. 1, 2

[16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015â4026, 2023. 2, 4, 6, 1

[17] Xiaohan Lei, Min Wang, Wengang Zhou, and Houqiang Li. Gaussnav: Gaussian splatting for visual navigation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 2

[18] Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao Zhang, Ronggang Wang, and Jian Zhang. Instancegaussian: Appearance-semantic joint gaussian representation for 3d instance-level perception. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14078â 14088, 2025. 2, 3, 4, 5

[19] Kunyi Li, Michael Niemeyer, Nassir Navab, and Federico Tombari. Dns-slam: Dense neural semantic-informed slam. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7839â7846. IEEE, 2024. 1

[20] Yanyan Li, Chenyu Lyu, Yan Di, Guangyao Zhai, Gim Hee Lee, and Federico Tombari. Geogaussian: Geometry-aware gaussian splatting for scene rendering. In European conference on computer vision, pages 441â457. Springer, 2024. 1

[21] Siyun Liang, Sen Wang, Kunyi Li, Michael Niemeyer, Stefano Gasperini, Nassir Navab, and Federico Tombari. Supergseg: Open-vocabulary 3d segmentation with structured super-gaussians. arXiv preprint arXiv:2412.10231, 2024. 1, 3, 4, 5, 6

[22] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38â55. Springer, 2024. 2

[23] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654â20664, 2024. 1, 3, 4

[24] Dominic Maggio, Yun Chang, Nathan Hughes, Matthew Trang, Dan Griffith, Carlyn Dougherty, Eric Cristofalo,

Lukas Schmid, and Luca Carlone. Clio: Real-time taskdriven open-set 3d scene graphs. IEEE Robotics and Automation Letters, 2024. 1

[25] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1

[26] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022. 1

[27] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari. Radsplat: Radiance field-informed gaussian splatting for robust real-time rendering with 900+ fps. In International Conference on 3D Vision 2025. 1

[28] Maxime Oquab, Timothee Darcet, Th Â´ eo Moutakanni, Huy Â´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research Journal, 2024. 2

[29] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 815â824, 2023. 2

[30] Yuning Peng, Haiping Wang, Yuan Liu, Chenglu Wen, Zhen Dong, and Bisheng Yang. Gags: Granularity-aware feature distillation for language gaussian splatting. arXiv preprint arXiv:2412.13654, 2024. 6

[31] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051â20060, 2024. 1, 2, 3, 4, 5, 6, 7, 8

[32] Yansong Qu, Shaohui Dai, Xinyang Li, Jianghang Lin, Liujuan Cao, Shengchuan Zhang, and Rongrong Ji. Goi: Find 3d gaussians of interest with an optimizable open-vocabulary semantic-space hyperplane. In Proceedings of the 32nd ACM international conference on multimedia, pages 5328â5337, 2024. 1, 3, 5, 6

[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748â8763. PmLR, 2021. 2, 6, 1

[34] Philipp A Rauschnabel, Reto Felix, Chris Hinsch, Hamza Shahab, and Florian Alt. What is xr? towards a framework for augmented and virtual reality. Computers in human behavior, 133:107289, 2022. 1

[35] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models

for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024. 2

[36] Yunhang Shen, Chaoyou Fu, Peixian Chen, Mengdan Zhang, Ke Li, Xing Sun, Yunsheng Wu, Shaohui Lin, and Rongrong Ji. Aligning and prompting everything all at once for universal visual perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13193â13203, 2024. 2

[37] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for openvocabulary scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5333â5343, 2024. 5, 6

[38] Lei Tian, Xiaomin Li, Liqian Ma, Hefei Huang, Zirui Zheng, Hao Yin, Taiqing Li, Huchuan Lu, and Xu Jia. Ccl-lgs: Contrastive codebook learning for 3d language gaussian splatting. arXiv preprint arXiv:2505.20469, 2025. 3

[39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Åukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 5

[40] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-worlddrive world models for autonomous driving. In European conference on computer vision, pages 55â72. Springer, 2024. 1

[41] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749â14759, 2024. 1

[42] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, et al. Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding. Advances in Neural Information Processing Systems, 37:19114â19138, 2024. 2, 3, 4, 5, 6, 7, 8, 1

[43] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Pointnerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5438â5448, 2022. 1

[44] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In European conference on computer vision, pages 162â179. Springer, 2024. 5, 6

[45] Haiyang Ying, Yixuan Yin, Jinzhi Zhang, Fan Wang, Tao Yu, Ruqi Huang, and Lu Fang. Omniseg3d: Omniversal 3d segmentation via hierarchical contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20612â20622, 2024. 1, 2, 4

[46] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447â19456, 2024. 1

[47] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676â21685, 2024. 1, 2, 3, 4, 5, 6

# GALA: Guided Attention with Language Alignment for Open Vocabulary Gaussian Splatting

Supplementary Material

## A. Implementation Details

## A.1. Hyperparameters

Gaussian Model. We adopt Scaffold-GS [23] as our appearance model. We use segmentation and geometry features of 16 dimensions each. Two separate two-layer MLPs are employed as the segmentation and geometry decoders in Eq.3, both trained with a learning rate of 0.0005. The number of neural Gaussians is set to K = 3 for both ScanNetv2 [7] and LERF-OVS [14]. We further extend the appearance Gaussian model with an instance feature $\mathbf { m } _ { i } \in \mathbb { R } ^ { 1 6 }$ , trained with a learning rate of 0.00005, while the learning rates for other Gaussian attributes follow [23].

Attention and Codebook Module. We set the codebook parameters to $N = 6 4 , d _ { \mathrm { i n s } } = 1 6$ , and $d _ { c } = 1 6 ,$ , and the learning rate to 0.001. A two-layer MLP is employed as the lifting decoder $\mathcal { F } _ { \mathrm { l i f t } } .$ , producing an output with dimension $d _ { \mathrm { l a n g } } = 5 1 2$

Training. By default, we train for 30k iterations in training stage 1 and 15k iterations in training stage 2. During stage 1, we enable the densification and pruning to keep both the appearance and semantic performance.

ScanNet-v2 Dataset. Following OpenGaussian [42], we selected 8 scenes from ScanNet for evaluation. For each scene, we select 1 frame every 20 frames as keyframes for training.

Preprocess Mask and Language Feature. We follow LangSplat [31] to preprocess SAM [16] masks and CLIP [33] language features from ground-truth RGB images. We use the large level of masks from SAM.

## B. Evaluation

2D open-vocabulary semantics provide per-pixel classification of objects or regions in the image plane, making them highly effective for tasks such as image segmentation and navigation. However, they are view-dependent, reflecting only what the camera sees, and may yield inconsistent semantics for the same object across viewpoints due to occlusions. In contrast, 3D open-vocabulary semantics assign per-point classifications in 3D space, preserving object size, location, and spatial relationships. They are particularly useful for applications such as robotic manipulation and AR/VR, but are often sparser. To leverage the advantages of both, we report evaluation metrics in both 2D and 3D, where our method achieves strong performance across both domains. In the following, we show how the 2D and 3D evaluation are performed on different datasets.

## B.1. 2D Evaluation

For 2D evaluation, we first render the 2D language feature maps and then compute mIoU and mACC for evaluation. For LERF-OVS [14], we follow the open-vocabulary query protocol of LangSplat [31].

## B.2. 3D Evaluation

LERF-OVS. Following LangSplat [31], for 3D evaluation on LERF-OVS we first use the text queries to select the corresponding Gaussians based on their language features, and then rasterize the selected Gaussians into 2D for further evaluation.

ScanNet-v2. ScanNet-v2 by default provides semantic point clouds for 3D evaluation. However, during training, the positions and number of points/Gaussians are updated to improve appearance quality. In contrast, OpenGaussian [42] fixes both positions and numbers to facilitate 3D evaluation, which we argue is unfair as it causes a notable drop in appearance performance. Our evaluation is a reproduction of the evaluation protocol proposed in Dr.Splat [12] as their full code is still not accessible. We adopt a shared, volume-aware evaluation protocol that computes per-voxel intersection-over-union (IoU) and accuracy by jointly considering the ScanNet-v2 ground-truth point cloud and the optimized neural Gaussian language features within a common voxel space.

## B.3. Visualization

To visualize semantic Gaussian results, the highdimensional language features need to be converted into 3-dimensional color values. We use the autoencoder from LangSplat [31] to mapping from CLIP features to RGB.

## C. Runtime Analysis

We perform single-GPU training with NVIDIA RTX 3090. Table 5 presents a runtime analysis of the Teatime scene. While our method incurs slightly longer training time due to the Scaffold-GS structure, it achieves significantly faster inference compared to other approaches. The codebook and attention module are extremely lightweight, requiring only 0.6 MB of memory. However, our method generates a fullsized language feature map for each view (requires 2 GB) and applies the cosine similarity loss in Eq. 8 between the full-sized ground-truth and the 512-dimensional rendered feature map (requires 10 GB), which is memory-intensive.

During inference, the cosine similarity loss is not computed, allowing our method to achieve superior runtime efficiency.

Additionally, our training pipeline consists of only two stages, whereas LangSplat and OpenGaussian require three. Regarding memory usage, both our approach and Open-Gaussian generate high-dimensional feature point clouds, while LangSplat uses a compressed feature representation. This accounts for the higher memory demand of our method.

<table><tr><td rowspan="3">Method</td><td colspan="3">Memory (GB)</td><td colspan="4">Train Time (min)</td><td rowspan="3">Inference Time (sec)</td></tr><tr><td>S1</td><td>S2</td><td>S3</td><td>S1</td><td>S2</td><td>S3</td><td>Total</td></tr><tr><td>LangSplat</td><td>5</td><td>2</td><td>6</td><td>18</td><td>7</td><td>84</td><td>109</td><td>96.00</td></tr><tr><td>OpenGaussian</td><td>15</td><td>9</td><td>13</td><td>43</td><td>27</td><td>5</td><td>75</td><td>0.65</td></tr><tr><td>Ours</td><td>12</td><td>14</td><td></td><td>107</td><td>109</td><td></td><td>216</td><td>0.31</td></tr></table>

Table 5. Runtime Analysis. S1âS3 correspond to stage 1, 2 and 3 of training, respectively.

## D. More Ablation Study

## D.1. Ablation on Number of Gaussian

As reported in OpenGaussian [42], densification is disabled during ScanNet-v2 evaluation, and appearance training is conducted at a very low resolution (160\*120). This results in a significant performance drop, as evidenced in Figure 7, where OpenGaussian produces appearance results that lose many fine details compared to ours.

In contrast, we train with the default resolution as in LangSplat [31] (320\*240), and Scaffold-GS introduces a parameter K to control the number of spawned Gaussians, making it unnecessary to disable densification entirely. By setting K = 3 and K = 10 for the same scene, we can flexibly adjust the number of Gaussians. As shown in Figure 7 and Table 6, reducing the number of Gaussians slightly lowers appearance quality, but the degradation is far less severe than in OpenGaussian. More importantly, segmentation performance improves noticeably. This improvement likely arises because semantics carry little or no texture information, so that semantic predictions require fewer Gaussians than appearance modeling. An excessive number of Gaussians may introduce ambiguities that negatively impact segmentation.

## E. More Experiment Results

## E.1. LERF-OVS

Figure 8, presents additional 2D qualitative results on the LERF-OVS dataset [14], while Figure 9 shows the corresponding 3D qualitative results.

## E.2. ScanNet-v2

In Table 7, we report the per-scene 3D open-vocabulary segmentation and localization results on ScanNet-v2 [7]. In

<!-- image-->  
Figure 7. Appearance Rendering. We show the appearance rendering quality on scene0000 00 of ScanNet-v2 dataset. Compared with OpenGaussian, our method achieve much better appearance rendering.

<table><tr><td>Scenes</td><td colspan="2">Teatime</td><td colspan="2">Scene0000_00</td></tr><tr><td>K</td><td>3</td><td>10</td><td>3</td><td>10</td></tr><tr><td>mIoU â</td><td>53.27</td><td>41.68</td><td>23.82</td><td>21.07</td></tr><tr><td>mAcc â</td><td>84.75</td><td>71.19</td><td>46.83</td><td>40.07</td></tr></table>

Table 6. Ablation on Number of Gaussians. We report mIoU and mAcc of our proposed method with different number of Gaussians. We report both 3D evaluation on LERF-OVS and ScanNet-v2.

Figures 10 and Figures 11, we show additional qualitative results.

<!-- image-->

Figure 8. More Qualitative Results of 2D Open-Vocabulary Segmentation on LERF-OVS.  
<!-- image-->  
Figure 9. More Qualitative Results of 3D Open-Vocabulary Segmentation on LERF-OVS. We visualize the language feature point cloud by compressing the features into the RGB point cloud. Note that the colors for visualization are consistent only within each method and not method-to-method.

<table><tr><td>Method</td><td colspan="2">Mean</td><td colspan="2">Scene0000_00</td><td colspan="2">Scene0062_00</td><td colspan="2">Scene0070_00</td><td colspan="2">Scene0097.00</td><td colspan="2">Scene0140_00</td><td colspan="2">Scene0347.00</td><td colspan="2">Scene0590_00</td><td colspan="2">Scene0645_00</td></tr><tr><td></td><td>mIoU â</td><td>mAcc â</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td>Number of Classes: 19</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LanSplat [31]</td><td>2.94</td><td>11.63</td><td>2.72</td><td>7.88</td><td>2.89</td><td>11.92</td><td>1.47</td><td>8.31</td><td>6.67</td><td>14.38</td><td>2.88</td><td>16.59</td><td>2.78</td><td>16.36</td><td>1.42</td><td>8.92</td><td>2.74</td><td>8.76</td></tr><tr><td>OpenGaussian [42]</td><td>15.47</td><td>26.04</td><td>16.12</td><td>27.67</td><td>18.02</td><td>27.42</td><td>19.01</td><td>31.64</td><td>9.01</td><td>13.61</td><td>15.39</td><td>28.49</td><td>23.22</td><td>35.93</td><td>11.84</td><td>18.56</td><td>11.22</td><td>25.06</td></tr><tr><td>Ours</td><td>21.54</td><td>37.47</td><td>19.04</td><td>40.67</td><td>20.14</td><td>41.01</td><td>18.54</td><td>28.88</td><td>18.01</td><td>38.74</td><td>25.55</td><td>38.13</td><td>43.11</td><td>50.13</td><td>11.55</td><td>29.61</td><td>16.40</td><td>32.64</td></tr><tr><td>Number of Classes: 15</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LanSplat [31]</td><td>3.80</td><td>13.98</td><td>3.18</td><td>9.35</td><td>3.73</td><td>16.84</td><td>2.18</td><td>12.64</td><td>8.40</td><td>18.47</td><td>3.40</td><td>13.98</td><td>3.94</td><td>20.04</td><td>3.43</td><td>11.81</td><td>2.13</td><td>8.71</td></tr><tr><td>OpenGaussian [42]</td><td>17.42</td><td>28.82</td><td>19.18</td><td>33.77</td><td>18.65</td><td>28.71</td><td>21.16</td><td>26.89</td><td>10.76</td><td>17.31</td><td>18.64</td><td>29.86</td><td>20.50</td><td>36.85</td><td>15.78</td><td>28.20</td><td>14.73</td><td>29.02</td></tr><tr><td>Ours</td><td>25.20</td><td>42.06</td><td>23.82</td><td>46.83</td><td>18.23</td><td>44.79</td><td>29.43</td><td>34.22</td><td>12.75</td><td>37.79</td><td>30.50</td><td>43.31</td><td>50.85</td><td>59.61</td><td>16.62</td><td>34.99</td><td>19.41</td><td>34.95</td></tr><tr><td>Number of Classes: 10</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LanSplat [31]</td><td>6.60</td><td>22.24</td><td>5.04</td><td>14.35</td><td>5.78</td><td>25.27</td><td>4.34</td><td>16.90</td><td>18.15</td><td>36.69</td><td>4.20</td><td>16.82</td><td>7.13</td><td>38.72</td><td>4.97</td><td>16.59</td><td>3.19</td><td>12.61</td></tr><tr><td>OpenGaussian [42]</td><td>23.46</td><td>37.73</td><td>22.70</td><td>39.32</td><td>28.34</td><td>43.10</td><td>29.09</td><td>37.55</td><td>21.72</td><td>29.88</td><td>21.91</td><td>35.11</td><td>20.34</td><td>46.22</td><td>25.56</td><td>36.17</td><td>18.09</td><td>34.55</td></tr><tr><td> Ours</td><td>35.85</td><td>57.02</td><td>27.61</td><td>57.53</td><td>30.02</td><td>66.91</td><td>51.76</td><td>67.65</td><td>24.65</td><td>52.81</td><td>45.26</td><td>60.62</td><td>56.03</td><td>56.20</td><td>25.17</td><td>43.74</td><td>26.36</td><td>50.74</td></tr></table>

Table 7. 3D Evaluation on ScanNet-v2. We report per scene mIoU and mAcc on the ScanNet-v2 dataset [7], following the evaluation protocol of Dr.Splat [12].

<!-- image-->  
Figure 10. More Qualitative Results of 3D Open-Vocabulary Segmentation on ScanNet-v2. We visualize the language feature point cloud by compressing the features into the RGB point cloud. Note that the colors for visualization are consistent only within each method and not method-to-method. OpenGaussian doesnât enable densification, leads to sparser point cloud.

<!-- image-->  
Figure 11. More Qualitative Results of 3D Open-Vocabulary Segmentation on ScanNet-v2. We visualize the language feature point cloud by compressing the features into the RGB point cloud. Note that the colors for visualization are consistent only within each method and not method-to-method. OpenGaussian doesnât enable densification, leads to sparser point cloud.