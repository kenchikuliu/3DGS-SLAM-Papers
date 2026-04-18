# MVGSR: Multi-View Consistent 3D Gaussian Super-Resolution via Epipolar Guidance

Kaizhe Zhang Xiâan Jiaotong University zkz1081@stu.xjtu.edu.cn

Weizhan Zhang Xiâan Jiaotong University

Shinan Chen Xiâan Jiaotong University chensn@stu.xjtu.edu.cn

Caixia Yan Xiâan Jiaotong University

Qian Zhao Xiâan Jiaotong University

Yudeng Xin University of Melbourne

## Abstract

Scenes reconstructed by 3D Gaussian Splatting (3DGS) trained on low-resolution (LR) images are unsuitable for high-resolution (HR) rendering. Consequently, a 3DGS super-resolution (SR) method is needed to bridge LR inputs and HR rendering. Early 3DGS SR methods rely on singleimage SR networks, which lack cross-view consistency and fail to fuse complementary information across views. More recent video-based SR approaches attempt to address this limitation but require strictly sequential frames, limiting their applicability to unstructured multi-view datasets. In this work, we introduce Multi-View Consistent 3D Gaussian Splatting Super-Resolution (MVGSR), a framework that focuses on integrating multi-view information for 3DGS rendering with high-frequency details and enhanced consistency. We first propose an Auxiliary View Selection Method based on camera poses, making our method adaptable for arbitrarily organized multi-view datasets without the need of temporal continuity or data reordering. Furthermore, we introduce, for the first time, an epipolar-constrained multiview attention mechanism into 3DGS SR, which serves as the core of our proposed multi-view SR network. This design enables the model to selectively aggregate consistent information from auxiliary views, enhancing the geometric consistency and detail fidelity of 3DGS representations. Extensive experiments demonstrate that our method achieves state-of-the-art performance on both object-centric and scene-level 3DGS SR benchmarks.

## 1. Introduction

Novel view synthesis has been regarded as a fundamental problem in 3D reconstruction, image editing, and virtual scene navigation, aiming to generate images from unseen viewpoints based on existing observations. Recently, Neural Radiance Fields (NeRF) [1] and 3D Gaussian Splatting (3DGS) [2] have achieved remarkable progress in this area. NeRF uses implicit neural representations to generate photorealistic images but struggles with scalability and efficiency, despite recent advances [3â6]. In contrast, 3DGS models scenes with Gaussian primitives, enabling a more efficient and compact representation of complex structures while preserving visual quality comparable to NeRF. As a result, 3DGS has quickly gained traction as one of the leading solutions for view synthesis tasks.

However, real-world photos are often captured with low resolution (LR), making high-resolution Novel view synthesis (HRNVS) particularly challenging. LR images typically lack sufficient detail, limiting the ability to faithfully reconstruct high-frequency information and resulting in noticeable artifacts when synthesizing high-resolution (HR) target views [7]. Therefore, 3DGS super-resolution (SR) methods become crucial for enabling reliable HRNVS.

Mainstream methods leverage image SR priors to guide 3DGS toward HRNVS. These approaches fall into two categories. The first category [8â11] relies heavily on singleimage super-resolution (SISR) models. Although this improves the perceptual quality of individual images, these methods fail to fuse complementary information across views and lack cross-view consistency, resulting in poor recovery of high-frequency details in the synthesized views. The second category [12, 13] leverages video SR techniques to overcome the shortcomings of the aforementioned methods. However, they rely on temporal continuity and alignment, whereas some multi-view datasets cannot be organized into coherent video streams, making direct adoption impractical.

Following prior studies while addressing their limitations, we focus on tackling 3DGS HRNVS problems by leveraging SR priors from multi-view observations.

To this end, we propose a novel Multi-View consistent 3D Gaussian Splatting Super-Resolution (MVGSR) framework. Our approach aims at integrating multi-view information for 3DGS rendering with high-frequency details and enhanced consistency. However, using all views is impractical due to the high computational cost. Thus, we propose an Auxiliary View Selection method based on camera pose to select views most informative for target SR, making it applicable to arbitrarily organized datasets. During auxiliary view selection, we extract and reorganize intrinsic and extrinsic camera parameters for all input views. By computing spatial and directional camera similarity, we identify the most suitable auxiliary views for each target, enabling effective high-frequency detail enhancement.

Furthermore, we design a hierarchical multi-view SR network that utilizes auxiliary image information to enhance the view quality. The target and auxiliary features are first extracted through a Multi-View Feature Extraction Module with an epipolar-constrained attention mechanism to integrate auxiliary image features into the target image feature representation. By identifying geometry-consistent correspondence regions, this mechanism precisely combines relevant features in auxiliary views. The fused multiscale features are progressively decoded and upsampled with SISR prior to reconstruct HR target images with improved geometric consistency and high-frequency details. The enhanced views together with the LR views are further used to optimize 3DGS, with an anti-aliasing subpixel loss.

Importantly, our method is the first to apply epipolarconstrained multi-view attention to 3DGS HRNVS, allowing efficient multi-view feature integration with lower computational cost, thus delivering substantially improved texture detail and structural consistency for 3DGS rendering. Experiments demonstrate that our method consistently achieves state-of-the-art (SOTA) 3DGS performance on both single-object and complex scene datasets.

In summary, our main contributions are as follows:

â¢ We present MVGSR, a novel multi-view consistent SR framework that enables HR 3DGS scene reconstruction and enhances high-frequency details for NVS.

â¢ We propose an auxiliary view selection method based on camera poses from arbitrarily organized multi-view datasets to select views that are more informative for the target image SR.

â¢ We introduce a multi-view SR network equipped with an epipolar-constrained attention mechanism, which enables geometry-aware cross-view fusion, significantly improving spatial consistency and high-frequency detail in 3DGS rendering.

â¢ Experiments show that our approach provides SOTA performance in 3DGS rendering quality over prior methods, improving cross-view consistency and high-frequency details on both single-object and complex scene datasets.

## 2. Related Work

## 2.1. Novel View Synthesis

NVS aims to generate new images from arbitrary target viewpoints given several observed images from known perspectives, which is a key technology in areas such as 3D reconstruction and virtual reality. Classical methods like NeRF [6] implicitly represent the scene as a continuous volumetric field and achieve high-quality novel view generation through neural rendering. However, despite numerous subsequent efforts to enhance its performance [14â17], NeRF-like methods remain constrained in practical applications due to their reliance on high-density sampling and the resulting inefficiency in real-time rendering. More recently, 3DGS [2] has emerged as a prominent explicit scene representation. Unlike NeRF, 3DGS describes the 3D structure with explicit Gaussian primitives and enables real-time, high-quality rendering through differentiable rasterization. Despite its advantages, early 3DGS methods suffer from sparse Gaussian distributions and insufficient texture details, resulting in a noticeable quality drop in HR rendering. Subsequent works [7, 18â20], such as Mip-Splatting, have alleviated rendering artifacts via spatial smoothing in the 3D domain but still struggle to recover rich textures from low-resolution inputs.

## 2.2. 3D Scene Super-Resolution

The objective of 3D scene SR is to reconstruct scenes capable of HR rendering from LR inputs. While SISR methods based on transformer architectures [21â23] have achieved impressive results in 2D vision, they often produce texture inconsistencies across different views and are unable to exploit the complementary information available between multiple views. As a result, their effectiveness for 3D scene reconstruction is limited. Recent efforts have explored 3D SR approaches based on NeRF, such as NeRF-SR [24] and FastSR-NeRF [5]. However, the performance of these methods is still restricted by the inherent modeling capabilities of NeRF and face challenges in recovering high-fidelity details.

With the rise of 3DGS, several studies have integrated SISR and 3DGS SR. SRGS [8], for example, introduces external high-quality SISR models to provide texture priors, thereby enhancing the capability of Gaussian primitives to represent fine-grained details. GaussianSR [9] leverages score distillation sampling to introduce large-scale generative image priors into 3D scene SR. SuperGS [11] adopts a two-stage coarse-to-fine framework, where an LR scene is first optimized and then refined using a pre-trained SISR model to enhance HR details. Although these methods attempt to improve cross-view consistency, their reliance on SISR makes it difficult to maintain texture consistency and recover high-frequency details across views. Another line of work, such as SuperGaussian [12] and SM [13], applies video SR for 3DGS SR. However, SuperGaussian applies video SR to the rendered trajectories of prereconstructed 3DGS scenes, which tends to amplify reconstruction errors present in the LR input. SM relies on a pretrained video SR model and requires an additional viewordering procedure to rearrange unordered multi-view inputs. However, such sorting is computationally intensive and resource-demanding, which limits its applicability in resource-constrained or large-scale multi-view scenarios.

<!-- image-->  
Figure 1. An overview of the proposed MVGSR pipeline. Given a set of LR images and their corresponding camera poses estimated via COLMAP, we first select auxiliary views based on camera pose. The selected auxiliary views, together with the target LR image, are fed into a multi-view SR network. It employs an epipolar-constrained multi-view attention mechanism to extract consistent and complementary high-frequency details from the auxiliary views. The resulting super-resolved images, together with the original LR images, are then used to jointly train the 3DGS.

To address these limitations, we propose MVGSR, a multi-view image-based 3DGS SR framework that avoids the texture inconsistencies and missing cross-view highfrequency details of SISR, while also eliminating error accumulation and poor generalization in video SR methods.

## 3. Methods

## 3.1. Overview

In this work, we propose a novel Multi-View consistent 3D Gaussian Splatting Super-Resolution (MVGSR) framework. The goal of our framework is to leverage multi-view images for enhancing high-frequency details and cross-view consistency, enabling high-quality scene reconstruction and NVS. As illustrated in Fig. 1, we first use COLMAP [25] to estimate the camera poses for a set of input views. For each input LR image LRi, our auxiliary view selection method identifies n auxiliary views and their camera poses. These images and pose information are then fed into our designed multi-view SR network equipped with the epipolar-constrained multi-view attention to generate a super-resolved image $\mathrm { S R } _ { i }$ that incorporates both multi-view consistency and high-frequency details. Using the superresolved images together with the original LR observations as guidance, our framework trains 3DGS to exploit crossview consistency and high-frequency complementary information from auxiliary views, resulting in substantially improved 3D reconstruction quality.

The following sections provide detailed descriptions of each module, including the auxiliary view selection method based on camera poses, the architecture of the multi-view SR network, the epipolar-constrained multi-view attention mechanism, and the loss functions used in our framework.

## 3.2. Auxiliary View Selection Method Based on Camera Poses

For integrating multi-view information for 3DGS rendering, we design an auxiliary view selection method based on camera pose information. Specifically, We first use COLMAP to obtain a set of accurate camera poses from the dataset and store them in a unified camera-to-world format, including all intrinsic and extrinsic parameters, which also facilitates the subsequent invocation of the multi-view SR network. Then, we use the camera poses to select views that provide rich complementary information for the target view. The core principle of the auxiliary view selection method is that, for any target view, its selected auxiliary views should satisfy the following three conditions: 1) The camera position should be closer to the scene center than the target camera so that the auxiliary view may provide finer details; 2) There should be a certain degree of content overlap with the target view, which is the prerequisite for effective information supplementation; 3) The auxiliary camera pose should not be too close to the target camera pose, otherwise the content will be redundant and less informative.

To this end, we compute both the camera position and the direction of each view. We first perform a filtering step based on two geometric constraints, and then rank the remaining candidates according to a mixed distance metric that combines spatial and directional similarity, finally selecting a fixed number of auxiliary views for further feature fusion. Specifically, assuming we have n cameras (the ith camera position and direction are denoted as $P _ { i }$ and $d _ { i } ,$ respectively), the auxiliary view selection can be formulated as follows:

<!-- image-->  
Figure 2. The architecture of the Multi-View SR Network. The whole network consists of the MVFE Module, the SIP Module, and the MSFF Module. The LR target and auxiliary images are taken as input for MVFE to extract multi-view features. The MVFE consists of 3 RET blocks at different scales, each integrated with an EST module employing epipolar-constrained multi-view attention. Combined with the single-image deep prior by the SIP module, the target image is effectively restored by fully fusing the single-image feature with the multi-view feature.

Step 1: Candidate Filtering. For each candidate view j, we retain only those that satisfy:

â¢ Condition 1 (closer to scene center):

$$
d _ { i } ^ { T } ( P _ { j } - P _ { i } ) > 0 .\tag{1}
$$

That is, assuming the target camera i faces the scene center, its direction forms an acute angle with the vector from i to j.

â¢ Condition 2 (viewing cone overlap): Let $\theta _ { j , i  j }$ denotes the angle between $P _ { j } - P _ { i }$ and $d _ { j } \colon$

$$
\sin \theta _ { j , i  j } \geq \frac { 1 } { 2 } ,\tag{2}
$$

indicating sufficient overlap between the view i and j.

Step 2: Distance Computation and Ranking. For the filtered candidates from Step 1, we calculate the position distance $D _ { p o s } ( i j )$ and the camera direction distance $D _ { d i r } ( i j )$ to the target camera i:

$$
D _ { p o s } ( i j ) = \| P _ { i } - P _ { j } \| ,\tag{3}
$$

$$
D _ { d i r } ( i j ) = 1 - \frac { d _ { i } ^ { T } d _ { j } } { \Vert d _ { i } \Vert \cdot \Vert d _ { j } \Vert } .\tag{4}
$$

According to the above, the final distance between the target camera i and the candidate camera j is computed as

$$
D _ { i j } = \{ \begin{array} { l l } { \lambda _ { p o s } D _ { p o s } ( i j ) + ( 1 - \lambda _ { p o s } ) D _ { d i r } ( i j ) , } \\ { \quad \mathrm { i f } \bar { d } _ { i } ^ { \top } ( P _ { j } - P _ { i } ) > 0 \mathrm { a n d } \sin \theta _ { j , i  j } \geq \frac { 1 } { 2 } } \\ { \infty , \qquad \mathrm { o t h e r w i s e , } } \end{array} \tag{5}
$$

where $\lambda _ { p o s }$ is a balancing weight between two distances; empirically, we set $\lambda _ { p o s } = 0 . 5$

Step 3: Final Selection. Finally, we sort the candidate views according to $D _ { i j } , j ~ = ~ 1 , 2 , . . . , n$ and select $N _ { r e f }$ auxiliary views by sampling one view every l positions along the sorted list, rather than simply choosing the $\mathrm { t o p } { - } N _ { r e f }$ closest views. This step ensures the third condition that the selected auxiliary views are both informativeness and diverse. For further explanation and details on our auxiliary view selection, please refer to the Appendix.

## 3.3. Multi-View Super-Resolution Network Architecture

As illustrated in Fig. 2, our Multi-View SR Network consists of three modules: the Multi-View Feature Extraction Module, the Single-Image Prior Module, and the Multi-Scale Feature Fusion Module. These three components cooperate to achieve high-quality reconstruction of the LR target image from auxiliary LR views.

Multi-View Feature Extraction (MVFE) Module. This module is responsible for extracting informative features from multiple auxiliary images and integrating them with the target imageâs feature. The feature extractor consists of three Residual Epipolar Transformer (RET) blocks, operating at different spatial scales. At each block, shallow convolutional layers f extract LR features from both the target and auxiliary images. These features are processed by an Epipolar-Guided Spatial Transformer (EST), which identifies the targetâauxiliary correlation by explicitly exploiting their geometry-aware projection consistency (see Section 3.4 for more details). The propagation process at each stage is formulated as

<!-- image-->  
Figure 3. Epipolar-Constrained Multi-View Attention

$$
\begin{array} { r l r } & { } & { x _ { j } = f ^ { j } ( x _ { j } ) + \mathrm { E S T } ^ { j } \Big ( f ^ { j } ( x _ { j } ) , ~ f ^ { j } ( r _ { 1 } ( x _ { j } ) ) , f ^ { j } ( r _ { 2 } ( x _ { j } ) ) , } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad ( 6 } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad ( 6 } \end{array}
$$

where $x _ { j }$ is the input of Layer $j ,$ , and $f ^ { j } ( \cdot )$ denotes the j-th convolutional layer, and $r _ { i } ( x _ { j } )$ represents the i-th auxiliary view of the $x _ { j }$ . The extracted features are further refined by a residual block: $x _ { j + 1 } = \operatorname { R e s } ^ { \mathrm { j } } ( x _ { j } )$

Single-Image Prior (SIP) Module. Considering that large-scale pre-trained SISR models inherently possess rich priors for single-image detail recovery, we employ a feature extraction module based on SISR prior to obtain deep features from the LR target image. Specifically, we adopt the SwinIR feature extraction module based on Swin-Attention and initialize it with the corresponding pre-trained weights [22]. This module provides a robust foundational representation for subsequent multi-scale feature fusion while reducing training costs.

Multi-Scale Feature Fusion (MSFF) Module. At each scale, the auxiliary feature from the corresponding level of the MVFE module is concatenated with the target feature along the channel dimension and fused via convolutions. The fused features are then upsampled to the next scale. This process is repeated across multiple levels to progressively reconstruct HR features, enabling high-quality restoration from the target image feature and multi-view feature. See the Appendix for more model details.

## 3.4. Epipolar-Constrained Multi-View Attention

The introduced epipolar-constrained multi-view attention mechanism is the core of EST. By identifying geometryconsistent correspondence regions, it precisely locates relevant auxiliary features, enabling the system to aggregate richer information from a larger number of auxiliary views within limited computational budgets. As shown in Fig. 3, the core idea is to leverage multi-view geometry by projecting each query point in the target view onto its corresponding epipolar line in the auxiliary views [26], where the point corresponding to the query must lie. In this way, we restrict attention computation to geometrically valid regions and reduce irrelevant information exchange.

Epipolar Region Computation. In the feature level, given a pixel $x _ { i }$ in the target view and the intrinsic and extrinsic parameters of both target and auxiliary cameras, the corresponding epipolar line $l _ { j }$ in the auxiliary view is computed as

$$
l _ { j } = \mathbf { F } _ { i j } \tilde { x } _ { i } ,\tag{7}
$$

where ${ \tilde { x } } _ { i }$ denotes the homogeneous coordinates of $x _ { i }$ , and $\mathbf { F } _ { i j }$ is the fundamental matrix between the target and reference views [26].

Epipolar Line Sampling and Attention Computation. For $x _ { i }$ , we uniformly sample $K _ { e p i }$ candidate points $\{ y _ { j } ^ { k } \} _ { k = 1 } ^ { K }$ along each epipolar line $l _ { j }$ in the reference view. The features at these locations are extracted as key and value matrices $K _ { j }$ , $V _ { j }$ for the attention computation. Attention is computed over these sampled points:

$$
\alpha _ { i , j } = \mathrm { S o f t m a x } \left( \frac { q _ { i } \cdot K _ { j } } { \sqrt { d } } \right) ,\tag{8}
$$

where $q _ { i }$ is the query feature from the target view.

The aggregated cross-view feature is

$$
f _ { i , j } ^ { \mathrm { e p i } } = \alpha _ { i , j } V _ { j } .\tag{9}
$$

Finally, all aggregated features from auxiliary views are integrated using a self-attention mechanism, resulting in the final enriched representation of $x _ { i }$

## 3.5. Loss Functions

For training the SR network, we adopt a composite loss that includes reconstruction loss $\mathcal { L } _ { \mathrm { r e c } }$ , and perceptual loss $\mathcal { L } _ { \mathrm { p e r } } \mathrm { : }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S R } } = \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { p e r } } \mathcal { L } _ { \mathrm { p e r } } . } \end{array}\tag{10}
$$

Specifically, we use the $\ell _ { 1 }$ -norm for $\mathcal { L } _ { \mathrm { r e c } }$ . The perceptual loss $\mathcal { L } _ { \mathrm { { p e r } } }$ [27] is computed using the VGG19 [28].

For 3DGS training, traditional methods optimize only HR reconstruction against the ground truth, which in SR settings neglects interactions with the LR inputs and can lead to degradation of scene structure. To mitigate this, we introduce a sub-pixel loss into the 3DGS training process, following prior work [8]. Unlike previous methods, we apply an anti-aliased bicubic downsampling scheme to the rendered images, which more faithfully preserves highfrequency details and thus provides more reliable supervision. The sub-pixel loss $\mathcal { L } _ { \mathrm { s p } }$ is computed between these downsampled renders and the LR ground-truth images. The overall objective of 3DGS is

Table 1. Quantitative comparison on NeRF Synthetic Ã4 and Ã2 (8 views). The numbers marked with â  are sourced from their respective paper. The best and second best entries are marked in red and orange, respectively.
<table><tr><td>Methods</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td colspan="4">400Ã400 â 800Ã800</td></tr><tr><td>Bicubic-3DGS((year?)) SwinIR-3DGS((year?))</td><td>30.97</td><td>0.9540</td><td>0.0581</td></tr><tr><td>Mip-Splatting((year?))</td><td>32.57 31.37</td><td>0.9640 0.9572</td><td>0.0374 0.0485</td></tr><tr><td>NeRF-SR((year?))</td><td>30.08</td><td>0.9391</td><td>0.0501</td></tr><tr><td>SRGS((year?))</td><td>32.67</td><td>0.9643</td><td>0.0371</td></tr><tr><td>Ours</td><td>33.01</td><td>0.9655</td><td>0.0368</td></tr><tr><td colspan="4">200Ã200 â 800Ã800</td></tr><tr><td>Bicubic-3DGS((year?)) SwinIR-3DGS((year?))</td><td>27.54 30.76</td><td>0.9149</td><td>0.1158</td></tr><tr><td>Mip-splatting((year?))</td><td>27.54</td><td>0.9498 0.9146</td><td>0.0561</td></tr><tr><td>NeRF-SR((year?))</td><td>28.45</td><td>0.9211</td><td>0.1162</td></tr><tr><td>FastSR-NeRFâ¡((year?))</td><td>30.47</td><td></td><td>0.0758</td></tr><tr><td>SRGS((year?))</td><td></td><td>0.9440</td><td>0.0750</td></tr><tr><td>SuperGS((year?))</td><td>30.83 30.89</td><td>0.9480</td><td>0.0560</td></tr><tr><td>GaussianSR((year?))</td><td>28.37</td><td>0.9490</td><td>0.0560</td></tr><tr><td>SuperGaussian((year?))</td><td>28.44</td><td>0.9240 0.9230</td><td>0.0870</td></tr><tr><td>SM((year?))</td><td></td><td></td><td>0.0670</td></tr><tr><td></td><td>28.05</td><td>0.9377</td><td>0.0627</td></tr><tr><td>Ours</td><td>31.11</td><td>0.9503</td><td>0.0550</td></tr><tr><td>HR-3DGS</td><td>33.32</td><td>0.9749</td><td>0.0239</td></tr></table>

$$
\mathcal { L } _ { \mathrm { 3 D G S } } = \lambda _ { \mathrm { r e n } } \mathcal { L } _ { \mathrm { r e n } } + ( 1 - \lambda _ { \mathrm { r e n } } ) \mathcal { L } _ { \mathrm { s p } } ,\tag{11}
$$

where $\mathcal { L } _ { \mathrm { r e n } }$ denotes the SR 3DGS rendering loss.

## 4. Experiment

## 4.1. Setup

All experiments are conducted on a single NVIDIA RTX 4090 GPU. During the training phase of our SR network, we set the total number of iterations to 200,000, a batch size of 2, and the number of epipolar sampling points $K _ { e p i }$ to 64, 32, and 16 for each block. The number of auxiliary images is fixed at 4, with selecting step size 2. In addition, we decay the learning rate from 1e-4 to 1e-7 in a cosine annealing way. The 3DGS training phase follows the original settings; to ensure fairness, all methods are evaluated under identical data and 3DGS training configurations unless otherwise stated. For more hyperparameter analysis and selection criteria, please see the Appendix. To comprehensively evaluate the robustness of our approach, we conduct experiments on the following datasets:

Tanks & Temples Dataset [30] is a real-world dataset. We select four scenes for testing and use the remaining scenes for training our SR network. The original resolution of the images is 1920Ã1080. For efficient training, we resize the images to 960Ã540 and test the performance at downscaled x2 and x4 resolutions. On the four testing scenes, we select every 8 images for 3DGS testing.

Mip-NeRF 360 Dataset [31] comprises 9 real-world scenes, with 5 outdoors and 4 indoors, each containing a complex central object or area with a detailed background. We downsample the training views by a factor of Ã4 as lowresolution inputs and directly apply our multi-view SR network for testing. Following prior work, we select every 8 images for 3DGS testing.

NeRF Synthetic Dataset [24] is a collection of 8 singleobject scenes, each with images at a resolution of 800Ã800. We use 100 images for 3DGS training and 200 test images for evaluation. All images are downsampled by a factor of 4 to generate the LR inputs.

For quantitative evaluation, we adopt the following metrics: Peak Signal-to-Noise Ratio (PSNR) [32], Structural Similarity Index (SSIM) [33], and Learned Perceptual Image Patch Similarity (LPIPS) [34]. Since our objective is to improve 3DGS reconstruction quality rather than standalone image SR, we treat PSNRâwhich measures deviations from the original sceneâas the primary evaluation metric. All results are obtained through rendering after 3DGS reconstruction, and the metrics are computed by comparing the rendered images with the ground-truth (GT) images.

## 4.2. Quantitative and Qualitative Comparisons

To rigorously validate the effectiveness of our proposed MVGSR, we conduct extensive comparisons against a range of existing approaches, including NeRF-based methods (NeRF-SR [24] and FastSR-NeRF [29]) and 3DGSbased methods (SRGS [8], SuperGS [11], GaussianSR [9], SuperGaussian [12], and SM [13]). Due to the unavailability of source code for some approaches, we directly report results from their original papers under identical configurations for fairness. Additionally, we include comparisons with three baselines: Bicubic-3DGS, SwinIR-3DGS, and Bicubic-Mip-Splatting [7]. For the complete results on the Mip-NeRF 360 Dataset, as well as the additional results on the NeRF Synthetic Dataset and the Tanks & Temples Dataset, please refer to the Appendix.

Quantitative Results. As shown in Tab. 1, our method achieves the best overall balance between quantitative accuracy and perceptual quality, outperforming existing 3DGSbased SR approaches in both Ã4 and Ã2 settings. The improvements in PSNR and LPIPS demonstrate its stronger ability to reconstruct high-frequency details and maintain cross-view consistency. The outstanding results on the object-centric NeRF Synthetic dataset confirm that our method can achieve high-fidelity reconstruction and consistent SR performance at the single-object level.

<!-- image-->

<!-- image-->  
Bic

<!-- image-->  
SwinIR

<!-- image-->  
Mip-Splatting

<!-- image-->  
Nerf-SR

<!-- image-->  
SRGS

<!-- image-->  
SM

<!-- image-->  
Ours

<!-- image-->  
GT

Figure 4. Qualitative comparisons on NeRF Synthetic Ã4 datasets. MVGSR produces more visually appealing results, successfully capturing high-frequency details and textures. Best viewed at screen!  
<!-- image-->

<!-- image-->

<!-- image-->  
SwinIR

<!-- image-->  
SRGS

Ours  
<!-- image-->  
GT  
Figure 5. Qualitative comparisons on Tanks & Temples dataset of 240Ã135 â 960Ã540 task. MVGSR consistently restores coherent structures and intricate details. Best viewed at screen!

As illustrated in Tab. 2, MVGSR consistently surpasses all existing methods across evaluation metrics, indicating its robustness in integrating multi-view information to recover fine-grained textures and high-frequency structures even in complex large-scale scenes.

Qualitative Results. Visual comparisons presented in

Table 2. Quantitative comparison on Tanks & Temples Dataset (4 views) at two resolution scales. The numbers marked with â contain only two scenarios: Truck and Train. The numbers marked with â  are sourced from their respective paper. The best and second best entries are marked in red and orange, respectively.

<table><tr><td>Method</td><td>PSNRâ SSIMâ</td><td>LPIPSâ</td></tr><tr><td>240Ã135</td><td>â 960Ã540</td><td></td></tr><tr><td>Bicubic-3DGS((year?)) Mip-Splatting((year?))</td><td>24.45 0.7699</td><td>0.3580</td></tr><tr><td>SwinIR-3DGS((year?))</td><td>24.42 0.7743</td><td>0.3573</td></tr><tr><td>SRGS ((year?))</td><td>25.57 0.8385</td><td>0.2772</td></tr><tr><td></td><td>25.38 0.8281</td><td>0.2874</td></tr><tr><td>Ours</td><td>25.75 0.8406</td><td>0.2705</td></tr><tr><td>HR-3DGS</td><td>26.63 0.8921</td><td>0.1888</td></tr><tr><td>480x270 â 1920x1080</td><td></td><td></td></tr><tr><td>Bicubic-3DGS(year?)) Mip-Splatting(year?))</td><td>24.23 0.7681 24.22 0.7738</td><td>0.3664 0.3724</td></tr><tr><td>SwinIR-3DGS((year?)) SRGS(year?))</td><td>24.77 0.8102</td><td>0.3217</td></tr><tr><td></td><td>24.79 0.8052</td><td>0.3315</td></tr><tr><td>Ours</td><td>24.90 0.8110</td><td>0.3205</td></tr><tr><td>SuperGS*((year?))</td><td>21.19 0.6950</td><td>0.3640</td></tr><tr><td>Ours*</td><td>23.31 0.8313</td><td>0.3258</td></tr><tr><td>HR-3DGS</td><td>25.30 0.8545</td><td>0.2781</td></tr></table>

Fig. 4 and Fig. 5, highlight MVGSRâs superior capability in recovering high-frequency details and fine textures. Conventional 3DGS-based methods commonly exhibit significant artifacts. Despite SwinIR enhancing local details, independently upscaling each view introduces cross-view inconsistencies, which ultimately cause noticeable detail loss in the rendered 3DGS views. Meanwhile, Mip-Splatting struggles to recover details due to insufficient high-frequency information, and NeRF-SR produces blurry reconstructions lacking precise textures. The SRGS method also suffers from blur and detail loss. Additionally, SM demonstrates reconstruction failures under whitebackground conditions, manifesting as substantial black artifacts. Conversely, MVGSR consistently restores coherent structures and intricate details while significantly mitigating visual artifacts prevalent in competing methods.

Table 3. Ablation studies on Tanks & Temples Dataset (4 views, 240Ã135 â 960Ã540). âAuxiliary â Nearâ: use nearest-neighbor views instead of the auxiliary selection; âAuxiliary â Randomâ: use random views instead; âEpi â Crossâ: replace epipolar attention with cross-attention.
<table><tr><td>Methods</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>MVGSR</td><td>25.72</td><td>0.8404</td><td>0.2710</td></tr><tr><td>Auxiliary â Near</td><td>25.58</td><td>0.8384</td><td>0.2734</td></tr><tr><td>Auxiliary â Random</td><td>25.61</td><td>0.8384</td><td>0.2728</td></tr><tr><td>Epi â Cross</td><td>25.56</td><td>0.8387</td><td>0.2738</td></tr></table>

<!-- image-->  
Figure 6. Attention distribution on the epipolar line. The query point (red dot) in the target view (left) attends to its corresponding regions along epipolar line in the auxiliary view (right), visualized as a heatmap.

Across all tested configurations, MVGSR consistently surpasses existing approaches, delivering superior perceptual quality and quantitative results. By aggregating rich information from auxiliary views, it effectively enhances rendering cross-view consistency and reconstructs fine-grained and realistic details. The results further demonstrate strong generalization across diverse datasets.

## 4.3. Ablation Studies and Analysis

To assess the effectiveness of MVGSR components, comprehensive ablation studies are conducted. To facilitate efficient and fair comparison, each variant is trained for 70000 iterations with consistent training settings, as summarized in Tab. 3. The results verify that both the epipolar attention module and auxiliary view-selection strategy contribute to performance enhancements. Furthermore, replacing conventional cross-attention with epipolar attention lowers the computational cost from $O ( N ^ { 2 } )$ to O(N ). This not only reduces memory consumption, but also enhances performance, since epipolar attention accurately identifies geometry-consistent feature regions for correspondence.

<!-- image-->  
Figure 7. Comparison of cross-view consistency among different methods, along with failure cases caused by inconsistency.

To verify whether the epipolar-constrained multi-view attention can effectively identify corresponding information from auxiliary views, we visualize the attention distribution of the trained Network along the computed epipolar line. As shown in Fig. 6, our approach successfully attends to the corresponding regions of the query points, assigning them the highest attention scores. This demonstrates the capability of our approach to effectively leverage cross-view information for enhanced consistency and complementary image details. We also observe notable attention responses in regions that share similar texture patterns with the query point, suggesting that our epipolar attention also captures non-local similarities across views, further enriching the visual content. The cross view consistency is further demonstrated in Fig. 7. SRGS causes cross-view inconsistencies (e.g., objects appearing in one view but missing in another) because it independently super-resolves each view without geometric guidance. In contrast, our method aggregates geometrically aligned cues from auxiliary views, maintaining coherent structures across viewpoints.

We also compare the memory consumption of our auxiliary-view selection with the view-reordering approach employed in SM [13] in the bicycle scene of Mip-NeRF 360 Dataset. SM requires 25.94 GB of memory to process long video sequences for reordering. In contrast, our method uses only 0.46 GB, as the auxiliary-view selection relies only on camera poses rather than the images. This design enhances the generality and applicability of our method in memory-limited environments where video-based methods are infeasible, and making it suitable for large-scale, largescene image datasets. More comparisons regarding method performance are provided in the Appendix.

## 5. Conclusion

In this paper, we introduce MVGSR, which integrates multi-view information to enhance 3DGS rendering with high-frequency details and improved consistency. We first propose an Auxiliary View Selection method based on camera poses, enabling adaptation to arbitrarily organized datasets without requiring view reordering. We also introduce an epipolar-constrained multi-view attention mechanism for 3DGS SR, which serves as the core of our network and aggregates geometrically consistent information from multiple auxiliary views. This design strengthens both geometric consistency and detail fidelity in 3DGS reconstruction. Experiments show that MVGSR achieves SOTA performance across 3DGS SR benchmarks. We believe MVGSR provides valuable insights for future research.

## References

[1] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1

[2] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 6, 7

[3] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European conference on computer vision, pages 333â350. Springer, 2022. 1

[4] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5501â5510, 2022.

[5] Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14346â 14355, 2021. 2

[6] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5480â5490, 2022. 1, 2

[7] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447â19456, 2024. 1, 2, 6, 7

[8] Xiang Feng, Yongbo He, Yubo Wang, Yan Yang, Wen Li, Yifei Chen, Zhenzhong Kuang, Jianping Fan, Yu Jun, et al. Srgs: Super-resolution 3d gaussian splatting. arXiv preprint arXiv:2404.10318, 2024. 1, 2, 6, 7

[9] Xiqian Yu, Hanxin Zhu, Tianyu He, and Zhibo Chen. Gaussiansr: 3d gaussian super-resolution with 2d diffusion priors. arXiv preprint arXiv:2406.10111, 2024. 2, 6

[10] Yecong Wan, Mingwen Shao, Yuanshuo Cheng, and Wangmeng Zuo. S2gaussian: Sparse-view super-resolution 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 711â721, 2025.

[11] Shiyun Xie, Zhiru Wang, Xu Wang, Yinghao Zhu, Chengwei Pan, and Xiwang Dong. Supergs: Super-resolution 3d gaussian splatting enhanced by variational residual features and uncertainty-augmented learning. arXiv preprint arXiv:2410.02571, 2024. 1, 2, 6, 7

[12] Yuan Shen, Duygu Ceylan, Paul Guerrero, Zexiang Xu, Niloy J Mitra, Shenlong Wang, and Anna Fruhst Â¨ uck. Su- Â¨ pergaussian: Repurposing video models for 3d super resolution. In European Conference on Computer Vision, pages 215â233. Springer, 2024. 1, 3, 6

[13] Hyun-kyu Ko, Dongheok Park, Youngin Park, Byeonghyeon Lee, Juhee Han, and Eunbyung Park. Sequence matters: Harnessing video models in 3d super-resolution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 4356â4364, 2025. 1, 3, 6, 8

[14] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855â5864, 2021. 2

[15] Peter Hedman, Pratul P Srinivasan, Ben Mildenhall, Jonathan T Barron, and Paul Debevec. Baking neural radiance fields for real-time view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5875â5884, 2021.

[16] Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Efficient neural radiance fields for interactive free-viewpoint video. In SIGGRAPH Asia 2022 Conference Papers, pages 1â9, 2022.

[17] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. Advances in Neural Information Processing Systems, 33:15651â15663, 2020. 2

[18] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In European Conference on Computer Vision, pages 165â181. Springer, 2024. 2

[19] Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong Dou. Absgs: Recovering fine details in 3d gaussian splatting. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 1053â1061, 2024.

[20] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654â20664, 2024. 2

[21] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general u-shaped transformer for image restoration. In Proceedings

of the IEEE/CVF conference on computer vision and pattern recognition, pages 17683â17693, 2022. 2

[22] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1833â1844, 2021. 5, 6, 7

[23] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5728â5739, 2022. 2

[24] Chen Wang, Xian Wu, Yuan-Chen Guo, Song-Hai Zhang, Yu-Wing Tai, and Shi-Min Hu. Nerf-sr: High quality neural radiance fields using supersampling. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6445â6454, 2022. 2, 6

[25] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104â4113, 2016. 3

[26] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003. 5

[27] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In European conference on computer vision, pages 694â711. Springer, 2016. 5

[28] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 5

[29] Chien-Yu Lin, Qichen Fu, Thomas Merth, Karren Yang, and Anurag Ranjan. Fastsr-nerf: Improving nerf efficiency on consumer devices with a simple super-resolution pipeline. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 6036â6045, 2024. 6

[30] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36(4):1â13, 2017. 6

[31] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470â5479, 2022. 6

[32] Quan Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics letters, 44(13):800â801, 2008. 6

[33] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004. 6

[34] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 6