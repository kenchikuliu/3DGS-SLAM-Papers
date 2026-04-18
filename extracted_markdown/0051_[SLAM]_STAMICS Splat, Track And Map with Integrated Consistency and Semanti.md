# STAMICS: Splat, Track And Map with Integrated Consistency and Semantics for Dense RGB-D SLAM

Yongxu Wang1â, Xu Cao2â, Weiyun Yi3â,Zhaoxin Fanâ 

Abstractâ Simultaneous Localization and Mapping (SLAM) is a critical task in robotics, enabling systems to autonomously navigate and understand complex environments. Current SLAM approaches predominantly rely on geometric cues for mapping and localization, but they often fail to ensure semantic consistency, particularly in dynamic or densely populated scenes. To address this limitation, we introduce STAMICS, a novel method that integrates semantic information with 3D Gaussian representations to enhance both localization and mapping accuracy. STAMICS consists of three key components: a 3D Gaussian-based scene representation for high-fidelity reconstruction, a graph-based clustering technique that enforces temporal semantic consistency, and an open-vocabulary system that allows for the classification of unseen objects. Extensive experiments show that STAMICS significantly improves camera pose estimation and map quality, outperforming state-of-theart methods while reducing reconstruction errors. Code will be public available.

## I. INTRODUCTION

Simultaneous Localization and Mapping (SLAM) is a crucial technology in fields such as autonomous driving, robotics, and augmented reality, enabling systems to perceive, map, and navigate complex environments in real time [11], [32]. The ability to accurately localize and construct detailed maps is fundamental for autonomous systems to interact safely and effectively with their surroundings. As these systems become more integrated into everyday life, the demand for SLAM solutions [1], [26] that are not only geometrically accurate but also semantically rich has grown, especially in high-density, dynamic environments. [8]

Traditionally, SLAM algorithms [3], [4], [7], [28] have relied heavily on geometric information extracted from RGB and depth data to perform localization and mapping. Recent methods, like Gaussian Splatting and neural implicit representations, have advanced the field by improving the density and fidelity of scene reconstructions. For instance, SplaTAM [12] uses explicit volumetric representations for high-quality scene reconstructions, while SNI-SLAM [34] employs implicit neural models for surface reconstruction and semantic labeling. However, a major limitation of these methods is their inability to maintain semantic consistency over time. In environments with dense or evolving semantics, this results in semantic driftâwhere the same object may be labeled inconsistently across different time framesâundermining both the accuracy of the mapping process and the utility of the reconstructed scene.

<!-- image-->  
Fig. 1: Illustration of our motivation: From left to right, they are the 3D reconstruction map, the semantic reconstruction map, and the depth reconstruction map.

To address this issue, we propose STAMICS, a novel framework that integrates semantic information directly into the SLAM process through Gaussian Splatting, where semantic data acts as a conditional constraint on geometric reconstruction. As shown in Fig. 1, we process semantic information and integrate it into the 3D reconstruction process. Our key idea is to introduce temporal semantic consistency constraints that ensure the same object is labeled consistently across time, thereby preventing semantic drift and improving the overall coherence of the system.

The STAMICS framework integrates three key components to enhance SLAM performance by maintaining semantic consistency. First, Semantic-Enhanced Gaussian Splatting combines semantic information with 3D Gaussian splatting to influence the geometric reconstruction process, ensuring that semantics are preserved and correctly aligned with the geometry of the scene. This improves the systemâs overall scene understanding and ensures semantic consistency within each frame. Second, Temporal Semantic Consistency introduces constraints that ensure consistent semantic labeling of objects across different time frames, preventing semantic drift and maintaining coherence over time. This component enforces the temporal alignment of semantic labels, preventing inconsistencies as the scene evolves. Finally, Open Vocabulary Expansion allows the system to dynamically expand its semantic understanding by classifying and labeling previously unseen objects, ensuring consistent semantics even for novel objects and making the system more adaptable to diverse and complex real-world environments. Through extensive experiments, we demonstrate that STAMICS achieves state-of-the-art performance in both mapping accuracy and camera pose estimation, significantly reducing reconstruction errors and advancing the robustness of dense SLAM systems. Our contributions can be summarized as follows:

â¢ We propose a novel integration of semantic information into the Gaussian Splatting process to improve geometric reconstruction and maintain semantic consistency within each frame.

â¢ We introduce temporal semantic consistency constraints to prevent semantic drift, ensuring robust and coherent mapping over time.

â¢ We incorporate an open vocabulary mechanism to handle previously unseen objects, enhancing the systemâs adaptability and maintaining consistent labeling in realworld scenarios.

## II. RELATED WORKS

Our work is closely related to recent advancements in SLAM technology, particularly in enhancing precision, flexibility, and usability. Therefore, we focus on reviewing two key approaches: SLAM based on Gaussian Splatting and SLAM based on Semantic Injection.

## A. SLAM based on Gaussian Splatting

With the emergence of Neural Radiance Fields (NeRF), there has been a growing interest in NeRF-based SLAM methods [5], [18], [21], [33], [35]. But to address real-time constraints, recent research has shifted towards Gaussian Splatting-based SLAM [13]â[15] methods. 3D Gaussian Splatting (3DGS) represents a promising approach for 3D scene modeling, where scenes are represented as a set of 3D Gaussian points, each characterized by parameters such as position, anisotropic covariance, opacity, and color. SplaTAM [12] was the first to leverage an explicit volumetric representation of 3D Gaussian distributions, achieving high-fidelity scene reconstruction using an unlocalized RGB-D camera. Other methods, such as Photo-SLAM [9], combine explicit geometric features with implicit photometric characteristics, learning multi-level features through a Gaussian pyramidbased training method, thereby enhancing the realism of the reconstructed scenes.

While these methods have significantly advanced the fields of computer vision and SLAMâimproving real-time performance, accuracy, and visual realismâthey primarily focus on geometric reconstruction and often lack semantic integration. This can lead to semantic drift in complex environments, where objects are labeled inconsistently over time. Our work, STAMICS, addresses these limitations by integrating semantic information directly into the Gaussian Splatting process.

## B. SLAM based on Semantic Injection

Semantic-aware SLAM systems go beyond traditional mapping by not only constructing a 3D map of the environment but also recognizing and understanding the semantic information of objects and regions within a scene. For instance, SNI-SLAM [34] leverages neural implicit representations to achieve precise semantic mapping, high-quality surface reconstruction, and robust camera tracking. It introduces hierarchical semantic representations, allowing for multilevel semantic understanding of the scene. Similarly, SGS-SLAM [17] is a semantic visual SLAM system that utilizes Gaussian rendering to address the over-smoothing issues often present in neural implicit SLAM systems.

While these methods have advanced the integration of semantics into SLAM [2], [20], [22], they often face challenges with semantic drift and maintaining temporal semantic consistency in dynamic environments. Our work, STAMICS, tackles these issues by introducing temporal semantic consistency constraints, ensuring stable and coherent semantic labeling over time.

## III. METHOD

## A. Overview

Given a sequence of images, our objective is to achieve accurate 3D scene reconstruction and robust camera tracking with integrated semantic understanding. To this end, we propose STAMICS, a framework that enhances traditional Gaussian-Splatting-based SLAM by incorporating semantic information. The framework is illustrated in Fig.2.

In Gaussian-based SLAM, 3D scene modeling is achieved through parameterized Gaussian distributions. Let $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ denote the Gaussian center position, r the radius, Ï the opacity, and c the color. The 3D Gaussian function is defined as:

$$
f ^ { 3 D } ( { \pmb x } ) = \sigma \exp \left( - \frac { \| { \pmb x } - { \pmb \mu } \| ^ { 2 } } { 2 r ^ { 2 } } \right)\tag{1}
$$

where $\begin{array} { r } { \textbf {  { x } } = ~ ( x , y , z ) ^ { \top } } \end{array}$ denotes spatial coordinates. The projection of these Gaussians onto the 2D rendering plane is governed by the cameraâs pose and intrinsic parameters:

$$
\mu ^ { 2 D } = K \frac { E _ { t } \mu } { d } , \quad r ^ { 2 D } = \frac { f r } { d } , \quad d = ( E _ { t } \mu ) _ { z }\tag{2}
$$

where f denotes the focal length of the camera, d is the depth, $\scriptstyle { E _ { t } }$ is the extrinsic parameters and K represents intrinsic matrix. The rendering process is performed by ordering Gaussians by depth and applying front-to-back volumetric rendering. The RGB and depth reconstruction errors for each pixel $\boldsymbol { p } = \left( u , v \right)$ are computed as:

$$
C ( { \boldsymbol { p } } ) = \sum _ { i = 1 } ^ { n } c _ { i } f _ { i } ( { \boldsymbol { p } } ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - f _ { j } ( { \boldsymbol { p } } ) )\tag{3}
$$

$$
D ( p ) = \sum _ { i = 1 } ^ { n } d _ { i } f _ { i } ( p ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - f _ { j } ( p ) )\tag{4}
$$

which is minimized by using differentiable rendering, then optimizing the camera pose while keeping Gaussian parameters fixed.

STAMICS builds upon this foundation and addresses the limitations of traditional Gaussian-Splatting SLAM by incorporating semantic information through three key components: Semantic-Enhanced Gaussian Splatting, which integrates semantic data into the geometric reconstruction process to ensure alignment between semantics and geometry; Temporal Semantic Consistency, which introduces temporal constraints to maintain consistent semantic labeling across frames, mitigating semantic drift; and Open Vocabulary Expansion, which enables dynamic learning of new objects, expanding the systemâs semantic understanding to handle diverse environments. In the following sections, we will delve into each of these components.

<!-- image-->  
Fig. 2: Overview: The RGB data is processed by the SAM to extract semantic information, which is then fed into the tracking module to localize the camera. Semantic-Enhanced Gaussian Splatting integrates the semantic data into the geometric reconstruction process, ensuring consistency between semantics and geometry. The process is governed by four types of losses, among which the semantic consistency loss originates from the semantic consistency module. The final output features open-vocabulary characteristics, with open vocabulary expansion enabling dynamic learning of new objects and achieving superior reconstruction results.

## B. Semantic-Enhanced Gaussian Splatting

Semantic information is often overlooked in traditional Gaussian-based SLAM systems, limiting the systemâs ability to interpret scenes with nuanced understanding. To address this, we first extract semantic features from the input images and inject them into the Gaussians.

Scene Semantic Extraction: The process begins with an image encoder that translates visual inputs into highdimensional feature representations. These features are then processed by a memory attention module, which leverages a memory bank containing historical data. This module enriches the feature representations by emphasizing relevant details, ensuring that critical information from past observations is retained. Concurrently, a prompt encoder generates task-specific cues, guiding a mask decoder to segment and extract precise semantic regions from the image. Once the image has been segmented, the semantic information is reencoded by a memory encoder and fed back into the memory bank. This continuous feedback loop allows the system to dynamically evolve its understanding of the environment, progressively refining its semantic knowledge over time. The entire process can be encapsulated as:

$$
\begin{array} { r } { M _ { t } = f _ { \mathrm { m e m } } \Big ( W _ { D } \cdot f _ { \mathrm { m a s k } } \Big ( W _ { P } \cdot f _ { \mathrm { p r o m p t } } \Big ( } \\ { A \Big ( W _ { A } \cdot f _ { \mathrm { a t t } } ( E ( I _ { t } ) , M _ { t - 1 } ) \Big ) \Big ) \Big ) \Big ) } \end{array}\tag{5}
$$

where $I _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ is the input image at time $t , F _ { t } =$ $E ( I _ { t } ) \in \mathbb { R } ^ { d _ { f } }$ is the feature embedding from the image encoder, $A ( F _ { t } , M _ { t - 1 } ) \in \mathbb { R } ^ { d _ { f } }$ is the attention mechanism using memory $M _ { t - 1 } , W _ { A } \in \mathbb { R } ^ { d _ { f } \times d _ { a } }$ , $W _ { P } \in \mathbb { R } ^ { d _ { a } \times d _ { p } }$ , and $W _ { D } \in$ $\mathbb { R } ^ { d _ { p } \times d _ { m } ^ { - } }$ are learned weight matrices, fatt, fprompt, fmask, fmem are non-linear activation functions, and $\bar { M _ { t } } \in \mathbb { R } ^ { d _ { m } }$ is the updated memory representation at time t.

Semantic Injection: After the semantic features are extracted, they are integrated into the Gaussian splatting process through Semantic Injection. This allows the system to incorporate both geometric and semantic information into the scene reconstruction and tracking processes. Each Gaussian, traditionally parameterized by position, radius, opacity, and color, is now augmented to carry semantic information. Specifically, each Gaussian is represented as a vector $c _ { i } =$ $[ r _ { i } , g _ { i } , b _ { i } , s e g _ { i } ] ^ { T }$ , where $r _ { i } , \ g _ { i } ,$ , and $b _ { i }$ represent the color channels, and $s e g _ { i }$ encodes the semantic label associated with the Gaussian. This enriched representation ensures that both visual and semantic contexts are accounted for during scene reconstruction. To ensure semantic consistency during rendering, we define the semantic reconstruction error for each pixel $\boldsymbol { p } = \left( u , v \right)$ as:

$$
S ( \boldsymbol { p } ) = \sum _ { i = 1 } ^ { n } s _ { i } f _ { i } ( \boldsymbol { p } ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - f _ { j } ( \boldsymbol { p } ) ) \qquad \tag{6}
$$

Here, $s _ { i }$ represents the semantic value of the i-th Gaussian, and $f _ { i } ( \boldsymbol { p } )$ is the contribution of the i-th Gaussian to the pixel $p ,$ based on its position and depth. This formulation uses front-to-back volumetric rendering to ensure that the semantics are consistently projected onto the 2D image plane.

## C. Temporal Semantic Consistency Processing

Incorporating semantic information from 2D masks improves instance recognition but lacks consistency across frames [25], [30]. To address this, we propose a Temporal Semantic Consistency Processing pipeline to enforce instance consistency over time. Our approach aligns instances across frames using three key components: Semantic Consistency Graph Construction, Graph Clustering, and Semantic Consistency Loss. First, we construct a semantic consistency graph with latent features from 2D masks to find correspondences across frames. Next, graph clustering groups nodes (masks) representing the same instance. Finally, we compute a semantic consistency loss to refine the 3D Gaussian mapping by aligning their projected semantic properties with 2D semantic maps. Next, we detail the process.

Semantic Consistency Graph Construction: We first construct the semantic consistency graph to cluster 2D masks from different frames that likely represent the same instance. Each 2D mask is treated as a node, with latent features $F _ { i }$ extracted and assigned to node i. For each pair of nodes i and j from different frames, we compute the cosine similarity between their latent features, and an edge is formed if the similarity exceeds a predefined threshold Ï . The edge function $E ( i , j )$ is defined as:

$$
E ( i , j ) = \mathbb { I } \left( { \frac { F _ { i } \cdot F _ { j } } { \| F _ { i } \| \| F _ { j } \| } } \geq \tau \right)\tag{7}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function that returns 1 if the condition is true and 0 otherwise. In our case, we set $\tau = 0 . 8$ This process is repeated for all nodes across frames within a sliding window, as shown in Fig. 3, forming the edges of the semantic consistency graph, which captures potential correspondences of instances across frames.

<!-- image-->  
G(k)  
Clustering  
Fig. 3: Illustration of graph clustering. Nodes with high semantic consistency scores are grouped into the same category in the graph G(k). Edges between inconsistent nodes are removed, resulting in a new graph $G ^ { \prime } ( k )$

Graph Clustering: Once the graph is constructed, nodes are clustered to determine whether 2D masks from different frames correspond to the same instance. For each node i, the semantic consistency score $S _ { i }$ is calculated as:

$$
S _ { i } = \frac { \sum _ { j \in N _ { i } } \delta ( L _ { i } , L _ { j } ) } { | N _ { i } | }\tag{8}
$$

where $\delta ( L _ { i } , L _ { j } )$ equals 1 if labels match, and 0 otherwise.

Nodes with a score greater than $\frac { 2 } { 3 }$ are grouped into the same cluster, thereby updating the semantic labels across frames, as illustrated in Fig. 4.

<!-- image-->  
Fig. 4: Illustration of the consistency score. The semantic label for the cabinet in the second frame is inconsistent. For the cabinet node in the first frame, the consistency score is $3 / 4$

Semantic Consistency Loss: Finally, we compute the semantic consistency loss using the updated 2D semantic frames and the initialized 3D Gaussian parameters. Let $S _ { \mathrm { s p l a t } } ^ { ( f ) } ( x , y )$ represent the splatted 2D semantic frame from the initialized Gaussians for frame f at pixel (x, y), and let $S _ { \mathrm { u p d a t e d } } ^ { ( f ) } ( x , y )$ be the updated frame. The semantic consistency loss $L _ { \mathrm { s c } }$ over N frames is given by:

$$
L _ { \mathrm { s c } } = \sum _ { f = 1 } ^ { N } \sum _ { ( x , y ) } \left| S _ { \mathrm { s p l a t } } ^ { ( f ) } ( x , y ) - S _ { \mathrm { u p d a t e d } } ^ { ( f ) } ( x , y ) \right|\tag{9}
$$

This loss ensures that the 3D Gaussians maintain semantic alignment with the updated 2D projections across frames, reinforcing temporal instance consistency.

## D. Open-vocabulary Expansion

To enhance the representation capability in the Temporal Semantic Consistency Processing and dynamically expand the systemâs semantic understanding by classifying and labeling previously unseen objects, we introduce an Openvocabulary Expansion. This ensures consistent semantics even for novel objects, making the system more adaptable to diverse and complex real-world environments. Specifically, we leverage open-vocabulary features to move beyond predefined label sets, enabling the identification and differentiation of a wider variety of instances. This allows the system to flexibly represent diverse objects and maintain semantic consistency across frames.

Our approach employs the Segment Anything Model (SAM) [16] to generate mask regions on the original RGB images, followed by the Contrastive LanguageâImage Pretraining (CLIP) model [19] to extract open-vocabulary features. Formally, given an image I and its corresponding mask M, the open-vocabulary feature $f$ is computed as:

$$
f = \mathcal { F } _ { \mathrm { C L I P } } \left( \mathcal { T } ( \mathcal { F } _ { \mathrm { S A M } } ( I ) \odot M ) \right)\tag{10}
$$

where $\mathcal { F } _ { \mathrm { S A M } } ( I )$ denotes the feature map generated by the SAM model, â represents the element-wise multiplication with the mask M to isolate the region of interest, and $\tau ( \cdot )$ is a transformation operation (e.g., cropping or resizing) applied to the masked region. Finally, $\mathcal { F } _ { \mathrm { C L I P } } ( \cdot )$ extracts the high-dimensional open-vocabulary feature from the transformed image patch.

In practice, we use this open-vocabulary feature as the latent feature in the Temporal Semantic Consistency Processing. These latent features serve as the basis for constructing the semantic consistency graph, where nodes represent the extracted features for each instance across frames. The graph is then clustered to group nodes that likely correspond to the same instance. By using these open-vocabulary latent features, we ensure that our method can dynamically handle diverse and previously unseen objects, maintaining instance consistency and improving the robustness of the graph construction and clustering processes.

## E. Semantic Consistency-Based Gaussian Map Refinement

After achieving semantic consistency through Temporal Semantic Consistency Processing, we focus on optimizing the Gaussian parameters and improving tracking performance. This is achieved via a combination of differential rendering and gradient-based optimization, where the camera poses are fixed, and the Gaussians are updated to refine the scene representation.

The core optimization problem is formulated as minimizing a combined loss function that integrates RGB, depth, and semantic information.

$$
\begin{array} { r } { L _ { t } = \displaystyle \sum _ { p } ( { \bf 1 } ( s ( p ) > 0 . 9 9 ) ) ( } \\ { L _ { 1 } ( D ( p ) ) + 0 . 5 L _ { 1 } ( C ( p ) ) + 1 . 5 L _ { 1 } ( S ( p ) ) ) } \end{array}\tag{11}
$$

where $C ( p ) , D ( p )$ , and $S ( p )$ represent the RGB, depth, and semantic projections at pixel $p ,$ respectively. The weighting factors are set as $w _ { \mathrm { r g b } } = 0 . 5$ , $w _ { \mathrm { d e p t h } } = 1$ , and $w _ { \mathrm { s e m a n t i c } } =$ 1.5 to balance visual details, spatial structure, and semantic understanding.

Given the semantic consistency established across frames, we further optimize the Gaussians by selecting keyframes that maximize overlap. For each current frame $f _ { \mathrm { c u r } } .$ we project its depth map into a 3D point cloud $P _ { \mathrm { c u r } }$ and compute the overlap with previous keyframes $f _ { k } { \mathrm { : } }$

$$
\mathrm { O v e r l a p } ( f _ { \mathrm { c u r } } , f _ { k } ) = \frac { | P _ { \mathrm { c u r } } \cap P _ { k } | } { | P _ { \mathrm { c u r } } | }\tag{12}
$$

We select the top k keyframes with the highest overlap to update the Gaussian parameters.

To ensure temporal consistency in the Gaussian map, we employed the semantic consistency loss $L _ { \mathrm { s c } }$ . The overall optimization problem thus becomes:

$$
L _ { \mathrm { o p t } } = L _ { \mathrm { t o t a l } } + 2 L _ { \mathrm { s c } }\tag{13}
$$

By minimizing $\begin{array} { r } { L _ { \mathrm { o p t } } , } \end{array}$ we achieve precise updates to the Gaussian parameters, ensuring that both geometric and semantic information is accurately captured and optimized in highly dynamic environments.

## IV. EXPERMENT

## A. Implementation Details

We conduct experiments on three benchmark datasets: TUM-RGBD [10], Replica [24], and Scannet [6]. The

Replica dataset consists of high-quality synthetic indoor environments, providing accurate depth maps and minimal interframe camera pose changes, making it suitable for evaluating performance under ideal conditions. In contrast, the TUM-RGBD dataset poses significant challenges for dense SLAM methods due to its low-quality RGB and depth images, sparse and noisy depth maps, and frequent motion blur. The Scannet dataset, similar to TUM-RGBD [10], is used for experiments with evaluations performed every five frames. To assess the influence of semantic consistency on mapping performance, we perform ablation studies on TUM-RGBD [10]. Performance is evaluated using standard metrics. For RGB image rendering, we employ Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS), each examining different aspects of image quality. Depth map accuracy is measured using L1 loss. Camera pose estimation accuracy is assessed using the Root Mean Square Error of the Absolute Trajectory Error (ATE RMSE). Additionally, we introduce a segmentation loss (seg loss) to quantify semantic consistency errors. We compare our method against several state-of-theart approaches. The primary baseline is SplaTAM [12], a leading 3D Gaussian SLAM technique. We also include comparisons with advanced SLAM methods based on dense radiance fields, including Point-SLAM [23], NICE-SLAM [36], and Vox-Fusion [31], to provide a comprehensive evaluation of our approach.

<table><tr><td>Methods</td><td>Avg.</td><td>desk</td><td>desk2</td><td>room</td><td>xyz</td><td>off</td></tr><tr><td>Kintinuous [27]</td><td>4.84</td><td>3.70</td><td>7.08</td><td>7.50</td><td>2.93</td><td>2.98</td></tr><tr><td>ElasticFusion [29]</td><td>6.90</td><td>2.52</td><td>6.81</td><td>21.48</td><td>1.16</td><td>2.53</td></tr><tr><td>NICE-SLAM [36]</td><td>15.86</td><td>4.24</td><td>4.98</td><td>34.48</td><td>31.76</td><td>3.85</td></tr><tr><td>Vox-Fusion [31]</td><td>11.32</td><td>3.53</td><td>6.01</td><td>19.54</td><td>1.48</td><td>26.04</td></tr><tr><td>Point-SLAM [23]</td><td>8.70</td><td>4.34</td><td>4.54</td><td>29.82</td><td>1.33</td><td>3.48</td></tr><tr><td>SplaTAM [12]</td><td>5.53</td><td>3.38</td><td>6.60</td><td>11.13</td><td>1.38</td><td>5.16</td></tr><tr><td>ours</td><td>5.11</td><td>3.22</td><td>5.57</td><td>10.62</td><td>1.18</td><td>4.98</td></tr></table>

TABLE I: ATE comparison using TUM-RGBD [10] dataset.Best results are highlighted in bold as first, and second for the second-best.

## B. Quantitative Comparison

We begin by conducting a quantitative analysis of our SLAM methodâs performance, as shown in Table I, Table II, Table III, Table IV and Table V.

<table><tr><td>Methods</td><td>Avg.</td><td>r0</td><td>r1</td><td>of0</td><td>of1</td><td>of2</td></tr><tr><td>NICE-SLAM [36]</td><td>1.10</td><td>1.02</td><td>1.44</td><td>0.92</td><td>1.06</td><td>1.12</td></tr><tr><td>Vox-Fusion [31]</td><td>3.99</td><td>1.38</td><td>4.84</td><td>8.48</td><td>2.66</td><td>2.58</td></tr><tr><td>Point-SLAM [23]</td><td>0.52</td><td>0.66</td><td>0.43</td><td>0.41</td><td>0.52</td><td>0.56</td></tr><tr><td>SplaTAM [12]</td><td>0.39</td><td>0.38</td><td>0.38</td><td>0.52</td><td>0.34</td><td>0.33</td></tr><tr><td>ours</td><td>0.33</td><td>0.25</td><td>0.38</td><td>0.44</td><td>0.22</td><td>0.26</td></tr></table>

TABLE II: ATE comparison using Replica [24]dataset.Best results are highlighted in bold as first, and second for the second-best.

On the TUM-RGBD [10] dataset, our approach exhibited exceptional stability, achieving an Avg. ATE of 5.11, outperforming other methods in overall consistency. Specifically, in the âxyzâ sequence, our method recorded an error of 1.18, nearly matching the best-performing ElasticFusion at 1.17, indicating strong adaptability across diverse environments. In the more challenging âroomâ sequence, our method significantly outperformed ElasticFusion, with errors of 10.62 and 21.49, respectively, representing a 6% improvement over previous baselines. These results are detailed in Table I. On the Replica [24] dataset, our method achieved the lowest average error of 0.33, outperforming all other methods. In the âr0â sequence, we achieved an error of 0.25, marking a 15% improvement over the closest baseline. This highlights our methodâs robustness in environments characterized by high consistency and repetitive structures, as shown in Table II. On the Scannet [6] dataset, our method matched the performance of NICE-SLAM [36], achieving an average error of 9.68. In the â0000â sequence, our method further distinguished itself with an error of 5.36, significantly outperforming NICE-SLAM (12.68) and Vox-Fusion (59.98), reflecting a 15% improvement over baseline methods (Table III). This demonstrates our methodâs reliability in complex indoor settings.

<!-- image-->  
Fig. 5: Comparison of reconstruction results with existing methods

<table><tr><td>Methods</td><td>Avg.</td><td>0000</td><td>0106</td><td>0169</td><td>0181</td><td>0201</td></tr><tr><td>NICE-SLAM [36]</td><td>9.69</td><td>12.68</td><td>8.02</td><td>10.88</td><td>12.98</td><td>3.87</td></tr><tr><td>Vox-Fusion [31]</td><td>26.00</td><td>59.98</td><td>9.18</td><td>26.69</td><td>24.46</td><td>9.68</td></tr><tr><td>Point-SLAM [23]</td><td>10.70</td><td>10.26</td><td>7.92</td><td>10.88</td><td>14.89</td><td>9.56</td></tr><tr><td>SplaTAM [12]</td><td>12.32</td><td>12.86</td><td>17.86</td><td>12.10</td><td>11.22</td><td>7.56</td></tr><tr><td>ours</td><td>9.68</td><td>5.36</td><td>15.40</td><td>11.49</td><td>8.80</td><td>7.34</td></tr></table>

TABLE III: ATE comparison using Scannet [6] dataset.Best results are highlighted in bold as first, and second for the second-best.

Beyond trajectory accuracy, our method also excels in rendering quality. In the âR1â scenario, we achieved the PSNR of 38.53 and the SSIM of 0.98, substantially outperforming Vox-Fusionâs PSNR of 27.79 and SSIM of 0.86 (Table IV). In dynamic environments, such as âOf0â and âOf1â, our method maintained high PSNR values (38.26 and 39.28) and achieved low LPIPS scores (0.10 and 0.09), demonstrating robustness in challenging, dynamic conditions. Compared to SplaTAM [12], our method consistently delivered superior image quality and depth estimation. As shown in Table V, our method achieved an average PSNR of 23.36, exceeding SplaTAMâs 22.21, with notable improvements in the âxyzâ sequence (26.68 vs. 25.15). These results validate the robustness and generalization capabilities of our method, especially in complex environments.

<table><tr><td>Methods</td><td>Metrics</td><td>Avg.</td><td>R0</td><td>R1</td><td>Of0</td><td>Of1</td></tr><tr><td rowspan="3">Vox-Fusion [31]</td><td>PSNR â</td><td>25.28</td><td>22.41</td><td>22.33</td><td>27.80</td><td>29.81</td></tr><tr><td>SSIM â</td><td>0.85</td><td>0.69</td><td>0.76</td><td>0.84</td><td>0.91</td></tr><tr><td>LPIPS</td><td>0.26</td><td>0.32</td><td>0.28</td><td>0.26</td><td>0.17</td></tr><tr><td rowspan="3">NICE-SLAM [36]</td><td>PSNR â</td><td>26.00</td><td>22.11</td><td>22.46</td><td>29.08</td><td>30.35</td></tr><tr><td>SSIM â</td><td>0.76</td><td>0.68</td><td>0.77</td><td>0.88</td><td>0.90</td></tr><tr><td>LPIPS</td><td>0.34</td><td>0.34</td><td>0.28</td><td>0.21</td><td>0.19</td></tr><tr><td rowspan="3">Point-SLAM [23]</td><td>PSNR â</td><td>35.93</td><td>32.38</td><td>34.07</td><td>38.25</td><td>39.13</td></tr><tr><td>SSIM â</td><td>0.97</td><td>0.98</td><td>0.97</td><td>0.99</td><td>0.99</td></tr><tr><td>LPIPS</td><td>0.12</td><td>0.11</td><td>0.12</td><td>0.10</td><td>0.16</td></tr><tr><td rowspan="3">SplaTAM [12]</td><td>PSNR â</td><td>36.03</td><td>32.84</td><td>33.88</td><td>38.24</td><td>39.16</td></tr><tr><td>SSIM â</td><td>0.97</td><td>0.98</td><td>0.97</td><td>0.99</td><td>0.97</td></tr><tr><td>LPIPSâ</td><td>0.11</td><td>0.07</td><td>0.12</td><td>0.10</td><td>0.09</td></tr><tr><td rowspan="3">Ours</td><td>PSNR â</td><td>38.24</td><td>36.89</td><td>38.53</td><td>38.26</td><td>39.28</td></tr><tr><td>SSIM â</td><td>0.97</td><td>0.96</td><td>0.98</td><td>0.99</td><td>0.98</td></tr><tr><td>LPIPS â</td><td>0.09</td><td>0.08</td><td>0.10</td><td>0.09</td><td>0.09</td></tr></table>

TABLE IV: Comparison of different methods on image quality metrics. he best metrics for PSNR, LPIP, and SSIM are highlighted in bold.

In conclusion, our SLAM method consistently outperforms state-of-the-art approaches across multiple datasets and scenarios, delivering superior accuracy in both trajectory estimation and scene reconstruction. It demonstrates strong adaptability, stability, and precision, making it highly effective in a wide range of challenging environments.

## C. Qualitative Comparison

We provide qualitative visual comparisons to further highlight the superiority of our method over existing approaches. As shown in Fig. 5, these visualizations demonstrate the enhanced detail, accuracy, and consistency of our SLAM system compared to SplaTAM [12] and NICE-SLAM [36].

<table><tr><td>Methods</td><td>Metrics</td><td>Avg.</td><td>D1</td><td>D2</td><td>R1</td><td>xyz</td></tr><tr><td rowspan="4">SplaTAM [12]</td><td>PSNRâ</td><td>22.21</td><td>21.49</td><td>20.98</td><td>21.22</td><td>25.15</td></tr><tr><td>SSIMâ</td><td>0.83</td><td>0.83</td><td>0.79</td><td>0.82</td><td>0.89</td></tr><tr><td>LPIPS â</td><td>0.23</td><td>0.27</td><td>0.27</td><td>0.27</td><td>0.12</td></tr><tr><td>Depth L1 â Seg L1</td><td>3.49 X</td><td>4.96 X</td><td>3.42</td><td>3.30</td><td>2.31 Ã</td></tr><tr><td rowspan="4">Ours</td><td>PSNR â</td><td>23.36</td><td>23.49</td><td>X 21.04</td><td>Ã 22.23</td><td>26.68</td></tr><tr><td>SSIM â</td><td>0.87</td><td>0.90</td><td>0.81</td><td>0.86</td><td>0.90</td></tr><tr><td>LPIPS â</td><td>0.19</td><td>0.19</td><td>0.26</td><td>0.23</td><td>0.09</td></tr><tr><td>Depth L1 â Seg L1</td><td>3.04 0.49</td><td>3.16 0.57</td><td>3.42 0.50</td><td>3.30</td><td>2.28</td></tr></table>

TABLE V: Comparison with baseline methods using TUM-RGBD,The improved indicators are marked in bold.

The improvements in visual fidelityâparticularly in texture and structural reconstructionâreinforce the quantitative gains observed in metrics such as PSNR, SSIM, and LPIPS. Fig. 5 clearly illustrates that our method (OURS) exhibits a distinct advantage in visual reconstruction over other methods (NICE-SLAM [36] and SPLaTAM [12]), with three key areas of improvement: 1) Superior detail preservation: Our approach is more precise in handling intricate details, such as items on a table or frames on a wall, resulting in sharper, more defined edges with minimal blurring or distortion. 2) Higher structural integrity: Our method maintains the geometric shapes and spatial structure of objects with greater accuracy. For instance, the furniture in the scene, such as sofas and chairs, closely resembles the real-world geometry (Ground Truth), whereas other methods exhibit noticeable distortion or deformation. 3) Stronger texture consistency: The textures in our reconstructions are more faithful to the Ground Truth, offering higher resemblance to the real scene.

## D. Ablation Study

Our method is built upon three key components: semanticenhanced Gaussian splatting, temporal semantic consistency processing, and open-vocabulary expansion. To evaluate the impact of each component, we conduct comprehensive ablation studies to analyze their contributions to the overall performance of the SLAM system.

Impact of Semantic-Enhanced Gaussian Splatting: We first investigate the effect of incorporating semantic information into the SLAM system through the loss function. The addition of a semantic module allows the system to process geometric data and capture the semantic structure of the environment. This significantly improves the accuracy of relocalization and loop closure. As shown in Table.VI, our experiments show that the ATE decreases substantially compared to the baseline model, which lacks semantic information. Additionally, the semantic module enhances the modelâs ability to identify and associate environmental features, reducing mismatches and trajectory estimation errors, ultimately optimizing the overall loss function. These results confirm that incorporating semantics into the Gaussian splatting improves both localization accuracy and map quality.

Impact of Temporal Semantic Consistency Processing: Next, we evaluate the effect of introducing a semantic consistency module to the SLAM system. This module enforces the consistency of semantic information during map construction and feature association, particularly in dynamic or complex scenes. Our experiments indicate that this component significantly improves the alignment between point cloud data and the map, and reduces errors in depth estimation. By maintaining high semantic consistency, the system achieves more accurate context-aware depth estimation, directly reducing depth loss and enhancing overall stability. As shown in Table. VII, metrics such as SSIM and LPIPS show marked improvements, with LPIPS improving by 50% and depth errors decreasing by 24% compared to the baseline model. These findings demonstrate the importance of preserving semantic consistency in challenging environments.

<table><tr><td>Methods</td><td>Avg</td><td>desk</td><td>desk2</td><td>room</td><td>off</td></tr><tr><td>Without Seg</td><td>5.81</td><td>3.34</td><td>6.58</td><td>11.49</td><td>5.18</td></tr><tr><td>With Seg</td><td>5.27</td><td>3.28</td><td>5.58</td><td>10.82</td><td>4.69</td></tr></table>

TABLE VI: An ablation study was conducted on the TUM-RGBD dataset to compare the performance before and after incorporating semantic information.

Impact of Open-Vocabulary Expansion: Finally, we assess the contribution of the open-vocabulary module, which extends the systemâs recognition capabilities to include previously unseen objects and scenes. This expansion greatly enhances the systemâs adaptability and flexibility, allowing it to operate effectively in more variable and unknown environments. Experimental results show that the open-vocabulary module improves the systemâs understanding of complex environments and increases the quality and realism of the reconstructed 3D scenes. Broader semantic recognition also enhances the modelâs performance in tasks such as depth estimation and environmental understanding. This component significantly boosts the systemâs overall performance and adaptability, particularly in diverse and dynamic settings.

In summary, the ablation studies clearly demonstrate the contributions of each key component to the systemâs performance. The introduction of open vocabulary aimed to expand the systemâs recognition range, enabling it to identify and understand previously unseen objects and scenes. This capability greatly enhanced the modelâs adaptability and flexibility, allowing it to function effectively in more variable and unknown environments. Experimental results confirmed that the use of open vocabulary not only enhanced the modelâs understanding of complex environments but also, through broader semantic recognition, further improved the quality and realism of the reconstructed 3D scenes. The incorporation of open vocabulary allowed the model to better perform depth estimation and environmental understanding tasks when facing diverse environments, significantly enhancing the modelâs overall performance and adaptability.

<table><tr><td>Methods</td><td>PSNR</td><td>Depth L1</td><td>SSIM</td><td>LPIP</td></tr><tr><td>Without Seg</td><td>21.49</td><td>4.38</td><td>0.83</td><td>0.26</td></tr><tr><td>With Seg</td><td>21.60</td><td>3.34</td><td>0.84</td><td>0.24</td></tr><tr><td>With Seg Consistency</td><td>23.54</td><td>3.30</td><td>0.88</td><td>0.13</td></tr></table>

TABLE VII: Performance comparison using TUM-RGBD dataset.The indicators with significant improvements are marked in bold.

## V. CONCLUSION

We present STAMICS, a novel SLAM method that integrates semantic information with 3D Gaussian representations to improve localization and mapping accuracy. By combining a 3D Gaussian-based scene representation for high-fidelity reconstruction, a graph-based clustering technique to enforce temporal semantic consistency, and an openvocabulary system for classifying unseen objects, STAMICS achieves significant improvements in camera pose estimation and map quality. Extensive experiments demonstrate that our approach outperforms state-of-the-art methods while reducing reconstruction errors.

## REFERENCES

[1] R. Alqobali, M. Alshmrani, R. Alnasser, A. Rashidi, T. Alhmiedat, and O. M. Alia, âA survey on robot semantic navigation systems for indoor environments,â Applied Sciences, vol. 14, no. 1, p. 89, 2023.

[2] M. Bloesch, J. Czarnowski, R. Clark, S. Leutenegger, and A. J. Davison, âCodeslamâlearning a compact, optimisable representation for dense visual slam,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2560â2568.

[3] C. Cadena, L. Carlone, H. Carrillo, Y. Latif, D. Scaramuzza, J. Neira, I. Reid, and J. J. Leonard, âPast, present, and future of simultaneous localization and mapping: Toward the robust-perception age,â IEEE Transactions on robotics, vol. 32, no. 6, pp. 1309â1332, 2016.

[4] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, âOrb-slam3: An accurate open-source library for visual, Â´ visualâinertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[5] Z. Cheng, C. Esteves, V. Jampani, A. Kar, S. Maji, and A. Makadia, âLu-nerf: Scene and pose estimation by synchronizing local unposed nerfs,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 312â18 321.

[6] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. NieÃner, âScannet: Richly-annotated 3d reconstructions of indoor scenes,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828â5839.

[7] J. Engel, T. Schops, and D. Cremers, âLsd-slam: Large-scale di- Â¨ rect monocular slam,â in European conference on computer vision. Springer, 2014, pp. 834â849.

[8] Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, and C. Finn, âHumanplus: Humanoid shadowing and imitation from humans,â arXiv preprint arXiv:2406.10454, 2024.

[9] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Real-time simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 584â 21 593.

[10] A. Kasar, âBenchmarking and comparing popular visual slam algorithms,â arXiv preprint arXiv:1811.09895, 2018.

[11] I. A. Kazerouni, L. Fitzgerald, G. Dooly, and D. Toal, âA survey of state-of-the-art on visual slam,â Expert Systems with Applications, vol. 205, p. 117734, 2022.

[12] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[13] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[14] L. Keselman and M. Hebert, âApproximate differentiable rendering with algebraic surfaces,â in European Conference on Computer Vision. Springer, 2022, pp. 596â614.

[15] ââ , âFlexible techniques for differentiable rendering with 3d gaussians,â arXiv preprint arXiv:2308.14737, 2023.

[16] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., âSegment anything,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015â4026.

[17] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, âSgs-slam: Semantic gaussian splatting for neural dense slam,â in European Conference on Computer Vision. Springer, 2025, pp. 163â 179.

[18] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[19] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PMLR, 2021, pp. 8748â8763.

[20] A. Rosinol, M. Abate, Y. Chang, and L. Carlone, âKimera: an opensource library for real-time metric-semantic localization and mapping,â in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 1689â1696.

[21] A. Rosinol, J. J. Leonard, and L. Carlone, âNerf-slam: Real-time dense monocular slam with neural radiance fields,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 3437â3444.

[22] R. F. Salas-Moreno, R. A. Newcombe, H. Strasdat, P. H. Kelly, and A. J. Davison, âSlam++: Simultaneous localisation and mapping at the level of objects,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2013, pp. 1352â1359.

[23] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald, âPoint- Â¨ slam: Dense neural point cloud-based slam,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 433â18 444.

[24] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[25] A. Takmaz, E. Fedele, R. W. Sumner, M. Pollefeys, F. Tombari, and F. Engelmann, âOpenmask3d: Open-vocabulary 3d instance segmentation,â arXiv preprint arXiv:2306.13631, 2023.

[26] Y. Wang, Y. Tian, J. Chen, K. Xu, and X. Ding, âA survey of visual slam in dynamic environment: the evolution from geometric to semantic approaches,â IEEE Transactions on Instrumentation and Measurement, 2024.

[27] T. Whelan, M. Kaess, M. Fallon, H. Johannsson, J. Leonard, and J. McDonald, âKintinuous: Spatially extended kinectfusion,â 2012.

[28] T. Whelan, M. Kaess, H. Johannsson, M. Fallon, J. J. Leonard, and J. McDonald, âReal-time large-scale dense rgb-d slam with volumetric fusion,â The International Journal of Robotics Research, vol. 34, no. 4-5, pp. 598â626, 2015.

[29] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison, âElasticfusion: Dense slam without a pose graph.â in Robotics: science and systems, vol. 11. Rome, Italy, 2015, p. 3.

[30] M. Yan, J. Zhang, Y. Zhu, and H. Wang, âMaskclustering: View consensus based mask graph clustering for open-vocabulary 3d instance segmentation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 28 274â28 284.

[31] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, âVoxfusion: Dense tracking and mapping with voxel-based neural implicit representation,â in 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2022, pp. 499â507.

[32] A. Yarovoi and Y. K. Cho, âReview of simultaneous localization and mapping (slam) for construction robotics applications,â Automation in Construction, vol. 162, p. 105344, 2024.

[33] M. Yin, S. Wu, and K. Han, âIbd-slam: Learning image-based depth fusion for generalizable slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10 563â10 573.

[34] S. Zhu, G. Wang, H. Blum, J. Liu, L. Song, M. Pollefeys, and H. Wang, âSni-slam: Semantic neural implicit slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 167â21 177.

[35] Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and M. Pollefeys, âNicer-slam: Neural implicit scene encoding for rgb slam,â in 2024 International Conference on 3D Vision (3DV). IEEE, 2024, pp. 42â52.

[36] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 12 786â12 796.