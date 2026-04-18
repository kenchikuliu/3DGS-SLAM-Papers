# NEDS-SLAM: A Neural Explicit Dense Semantic SLAM Framework using 3D Gaussian Splatting

Yiming Ji, Yang Liu\*, Guanghu Xie, Boyu Ma, Zongwu Xie, and Hong Liu

AbstractâWe propose NEDS-SLAM, a dense semantic SLAM system based on 3D Gaussian representation, that enables robust 3D semantic mapping, accurate camera tracking, and highquality rendering in real-time. In the system, we propose a Spatially Consistent Feature Fusion model to reduce the effect of erroneous estimates from pre-trained segmentation head on semantic reconstruction, achieving robust 3D semantic Gaussian mapping. Additionally, we employ a lightweight encoderdecoder to compress the high-dimensional semantic features into a compact 3D Gaussian representation, mitigating the burden of excessive memory consumption. Furthermore, we leverage the advantage of 3D Gaussian splatting, which enables efficient and differentiable novel view rendering, and propose a Virtual Camera View Pruning method to eliminate outlier gaussians, thereby effectively enhancing the quality of scene representations. Our NEDS-SLAM method demonstrates competitive performance over existing dense semantic SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in 3D dense semantic mapping.

Index Termsâ3D Gaussian Splatting; Dense Semantic Mapping; Neural SLAM; 3D Reconstruction.

## I. INTRODUCTION

Visual SLAM (Simultaneous Localization and Mapping) is a fundamental research problem in robotics, which involves simultaneously tracking the camera pose and incrementally constructing a map of an unknown environment [1]. Downstream tasks such as autonomous goal navigation, human-computer interaction, mixed reality (MR), and augmented reality (AR) demand not only accurate camera pose tracking from SLAM systems but also robust and dense semantic reconstruction of the environment. This research focuses on semantic RGBD-SLAM, which, in contrast to traditional SLAM, enables the identification, classification, and association of entities within a scene, ultimately generating a semantically-rich map.

Inspired by the success of NERF and 3D Gaussian Splatting (3DGS) in high-fidelity view synthesis, researchers have explored building end-to-end visual SLAM systems based on neural radiance fields. These novel SLAM architectures offer superior solutions compared to traditional algorithms in terms of surface continuity, memory requirements, and scene completion. Specifically, iMAP [2] and NICE-SLAM [3] leverage neural implicit fields for consistent geometry representation, while MonoGS [4] and SplaTAM [5] employ 3DGS to achieve photo-realistic mapping.

Given continuous input of RGB-D frames, dense semantic SLAM aims to create a compact and dense 3D representation of the scene that includes accurate RGB information as well as dense semantic data. However, current state-ofthe-art semantic segmentation models are trained on large amounts of internet images, which are loosely related and time-independent. This leads to estimation errors such as semantic spatial inconsistency, which significantly impairs the density and completeness of semantic reconstruction. The previous 3DGS-based semantic SLAM method [6] overlooked the issue of semantic feature inconsistency, which limits its potential for practical applications.

Furthermore, our research has found that directly embedding semantic category labels into gaussians parameters may not be appropriate. During splatting, overlapping gaussians combine through alpha-blending to form pixel values on the imaging plane. Using RGB color channels as an example, ideally, when 3D gaussians are splatted onto different imaging planes, they create different color blending effects. However, assigning fixed class labels to the gaussians leads to meaningless values in the semantic channels during splatting. Therefore, attempting to embed semantic features instead of semantic category labels into the 3D gaussians parameters would be more promising. However, this approach can cause prohibitive memory requirements and significantly lower the efficiency of both optimization and rendering, as semantic features typically have higher dimensions, whereas category labels are just integer values.

In a 3DGS-based SLAM system, the process of incrementally building a map is often influenced by camera pose estimation errors, object occlusions, and errors in the optimization process. These factors can introduce 3D gaussians that do not align with actual surfaces. When these outlier gaussians are included in the rendering view, they can create visual artifacts, which in turn affect camera pose estimation, creating a vicious cycle. This issue is not addressed in the original 3DGS paper, where the camera poses for each frame are precomputed using an offline SFM method. Therefore, handling outlier gaussians is crucial for 3DGS-based SLAM methods.

Overall, 3DGS based Dense Semantic SLAM can be summarized as facing two key challenges: 1) Providing robust semantic reconstruction results under inconsistent semantic features. 2) Incrementally building a map that can accurately distinguish well-optimized and low-quality regions, while effectively filtering out outliers to improve reconstruction quality.

This paper proposes NEDS-SLAM, with the following key contributions:

â¢ We propose the Spatially Consistent Feature Fusion module (SCFF), which combines semantic features with appearance features. This module addresses the spatial inconsistency of semantic features and provides a more robust semantic SLAM solution.

â¢ We embed semantic features into Gaussian parameters instead of using category labels. We also introduce a lightweight encoder-decoder to prevent memory issues from high-dimensional semantic feature embedding.

â¢ We present the Virtual Camera View Pruning (VCVP) method. VCVP generates multiple virtual camera views to ensure consistency, identifying and removing unstable gaussians caused by occlusions, camera pose errors, and parameter optimization issues, leading to a more accurate 3D Gaussian field.

## II. RELATED WORK

## A. Traditional approaches to dense semantic SLAM

Real-time dense semantic SLAM systems face the challenge of effectively fusing semantic information into underlying 3D geometric representations of the environment. Traditional approaches use voxels, point clouds, and signed distance fields to encode object labels [7], [8]. However, voxel- and point cloud-based approaches struggle with reconstruction speed and high-fidelity model acquisition. Meanwhile, signed distance field representations incur high memory usage that does not scale well to large-scale environments. There remains a need for more efficient and expressive 3D semantic modeling techniques suitable for real-time dense SLAM.

## B. NeRF based SLAM

In recent years, Neural Radiance Fields (NeRF) have sparked significant interest in computer graphics, attracting attention for their high-fidelity novel view synthesis and lightweight scene representation [9]. This enthusiasm has quickly spread to the SLAM field, leading to the development of many innovative SLAM architectures [2] [3]. Zhu et al. introduced SNI-SLAM [10], which employs neural implicit representation and hierarchical semantic encoding for multi-level scene understanding, contributing a cross-attention mechanism for the collaborative integration of appearance, geometry, and semantic features. Due to the limitations of NeRF's volume rendering, NeRF-based dense semantic SLAM struggles to simultaneously model and optimize the semantic and RGB-geometry information of the environment [11] [12]. Additionally, the efficiency of SLAM is constrained by the implicit representation of the map [13].

## C. Gaussian Splatting based SLAM

3DGS representations have emerged as a promising approach for 3D scene modelling using a set of 3D gaussians, each characterized by parameters such as position, anisotropic covariance, opacity, and color [14]. While existing 3DGSbased SLAM methods have primarily focused on RGB reconstruction, exploring end-to-end system architectures, optimization of gaussians parameters, and accurate camera pose tracking through differentiable rendering, less attention has been paid to semantic reconstruction [5], [15], [4], [16]. The few semantic 3DGS-SLAM approaches proposed to date have simply encoded ground truth semantic color labels directly as a second color channel of the gaussians parameters [6], without explicit modeling of semantic information or inference. There is clear potential for more sophisticated integration of semantics within the 3DGS-SLAM framework. The present work conducts a more in-depth exploration of dense semantic SLAM, aiming to simultaneously improve the robustness and reconstruction fidelity of 3DGS-based SLAM systems through more sophisticated modeling and inference of semantic information within the 3DGS representation.

## III. METHODOLOGY

## A. Scene Representation and Semantic embedding

Each 3DGS utilized for representing three-dimensional scenes encompasses mean, covariance, and color information. In this paper, a simplified 3DGS representation of the scene is employed [5], omitting the spherical harmonics functions used for color representation, while assuming gaussians to be isotropic as in Eq 1.

$$
f ^ { g s } \left( { \pmb x } \right) = o \exp \left( - \frac { \left\| { \pmb x } - { \pmb \mu } \right\| ^ { 2 } } { 2 r ^ { 2 } } \right)\tag{1}
$$

Where $\mu \in \mathbb { R } ^ { 3 }$ represents the center position, $r$ is the radius, and $o \in \ [ 0 , 1 ]$ represents the opacity. The rapid and differentiable rendering based on 3DGS serves as the core of mapping and tracking within 3DGS-based SLAM systems. This ability for fast rendering enables the system to directly compute the gradients of the underlying parameters based on the discrepancy between the rendered results and the actual data. Consequently, the gaussians parameters can be updated to achieve an accurate representation of the scene. The differentiable rendering process based on gaussians splatting comprises three steps: Frustum Culling, Splatting, and Rendering by Pixels [17].

$$
C \left( p \right) = \sum _ { i \in N } { c _ { i } f _ { i } ^ { g s } \left( p \right) \prod _ { j = 1 } ^ { i - 1 } { \left( 1 - f _ { j } ^ { g s } \left( p \right) \right) } }\tag{2}
$$

After arranging a collection of 3D gaussians and camera pose, it is imperative to sort the gaussians in a front-toback manner. By employing alpha-compositing, the splatted 2D projection of each gaussian can be efficiently rendered in pixel space, ensuring the generation of RGB images in the desired order, as Eq 2. $\mathbf { c } _ { i }$ represents the color parameters of the gaussians, and $f _ { i } ^ { g s } \left( p \right)$ is computed as in Eq 1 but with the 2D splatted $\mu$ and $r _ { \ast }$ The rendering process is completed by multiplying the opacity of each gaussian with the color and accumulating the results. The depth map is rendered in a similar manner, as shown in Eq 3.

$$
D \left( p \right) = \sum _ { i \in N } d _ { i } f _ { i } ^ { g s } \left( p \right) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ^ { g s } \left( p \right) \right)\tag{3}
$$

The most notable distinction between semantic features and color and geometric features lies in their high-dimensional attributes. The semantic features do not refer to the per-pixel class labels generated by the segmentation head. Instead, it pertains to the high-dimensional semantic features extracted by the pre-trained model at each pixel. Taking DINO [18] as an example, the ViT-S model produces latent feature encodings of 384 dimensions, while the ViT-G model produces encodings of 1536 dimensions.

<!-- image-->

A simple way to combine 3DGS with semantic features is to add trainable feature vectors to each gaussian. These parameters can be learned during the differentiable rendering process, which allows end-to-end training. However, for dense semantic SLAM, adding a high dimensional semantic feature vector to each 3DGS is memory-inefficient. Inspired by LangSplat [19], we propose using a simple MLP as an encoder to compact semantic features into a low-dimensional vector. The compressed semantic features are then added to the 3D gaussians and can be rendered as in Eq 4.

$$
{ \cal S } \left( p \right) = \sum _ { i \in N } f _ { i } f _ { i } ^ { g s } \left( p \right) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ^ { g s } \left( p \right) \right)\tag{4}
$$

## B. Adaptive 3D Gaussian Expansion Mapping

1) Spatially Consistent Feature Fusion (SCFF) : Pervious semantic SLAM approaches typically use pretrained segmentation models to compute pixel-level labels from each RGB frame,but these class labels lack environmental specificity. Pretrained models may produce inconsistent semantic estimates, where the same object is predicted with different semantic labels in images from different camera views.

To address this issue, SNI-SLAM [10] computes a fused feature by combining geometry, appearance, and semantic features. CoSSegGaussians [20] incorporates DINO [18] features with superior multi-view semantic scale consistency into the gaussians parameters. Subsequently, the semantic encoding of each gaussians is fused with spatial coordinates to render semantic features, thereby enhancing robustness.

In this paper, we propose a simplified fusion mechanism. It combines the appearance features with the semantic features extracted from pretrained model. The resulting mixed feature, obtained through MLP encoder, is then embedded as the final semantic encoding in the 3DGS representations.

As shown in Fig 1, the pretrained semantic feature extractor extracts an $H \times W \times D _ { f }$ feature map $F _ { d f }$ from an $H \times W \times 3$ RGB frame. At the same time, the spatial feature extractor extracts $H \times W \times D _ { s }$ features $F _ { d s }$ from RGB data. After three layers of convolution, the feature channels of $F _ { d f }$ are reduced to 256, 128, and 16, respectively. Similarly, the feature channels of $F _ { d s }$ are increased to 16 through one layer of CNN. After concatenation and the final convolution, we obtain the spatially consistent feature $F _ { s c f f } ^ { 3 2 }$ with 32 channels. Using the external parameters of the camera, we can convert an input frame of RGBD into a series of points in 3D space. Each point includes xyz coordinates, RGB information, and 32-channel SCFF features.

To reduce the number of 3D gaussian parameters, we need to use an information encoding method to compress $F _ { s c f f } ^ { 3 2 }$ to a lower dimension, such as using hash encoding [21], GPR [22], etc. In this paper, we use a simple MLP to compress $F _ { s c f f } ^ { 3 2 }$ to three dimensions, resulting in $F _ { s c f f } ^ { 3 }$

It is important to clarify that the semantic category labels are numerical IDs from a predefined category library (e.g., 0 represents a person), while the $F _ { s c f f } ^ { 3 }$ values range between 0 and 1. These semantic features can be decoded back into semantic category labels by a subsequent decoder.

We use the pre-trained DINO [18] model as a semantic feature extractor, obtaining features $F _ { d f }$ with 384 channels $( D _ { f } = 3 8 4 )$ . We use DepthAnything [23] as the spatial feature extractor, resulting in $D _ { s } ~ = ~ 1$ . Relative depth output from DepthAnything is used as the appearance feature because changes in the camera viewpoint do not affect the relative position of surfaces on the object. The spatial consistency of appearance features helps SCFF achieve stable semantic feature estimation.

The relative depth between pixels can reflect the geometric structure of observed surfaces. The SCFF module dynamically adjusts the weights of semantic features according to the spatially consistent relationships. It thereby reduces the impact of segmentation errors on the spatial consistency of semantic features.

2) Updating 3D Gaussians: During the mapping process, we assume that the camera pose for the current frame is known. We need to use the current keyframe's RGBD data to update the gaussians representation of the scene. Updating has two meanings: optimizing existing scene parameters and generating a new 3DGS distribution for the scene.

Following the processes used in Splatam [5] and GS-SLAM [15], we use Eq.5 to calculate the silhouette value per pixel. The silhouette images are rendered to determine the contribution of each gaussian to the map.

$$
\displaystyle { S i l \left( p \right) = \sum _ { i \in N } f _ { i } ^ { g s } \left( p \right) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ^ { g s } \left( p \right) \right) }\tag{5}
$$

At the same time, the difference between the projected depth value and the ground truth value of pixels corresponding to newly added gaussian is checked when they are projected back onto the image plane.

$$
M \left( p \right) = \left[ S i l \left( p \right) < T _ { s } \right] + \left[ \left( D _ { g t } \left( p \right) - D \left( p \right) \right) < T _ { d } \right]\tag{6}
$$

The densification mask M (p) is calculated according to Eq 6, where $D \left( p \right)$ represents the depth value of pixel p. M (p) represents a Boolean mask for pixel p. The optimization of 3DGS and the addition of new gaussians will be confined to areas where the mask value is True, thereby avoiding the densification of gaussians in well-reconstructed areas. This differs from the approach in [14], which splits gaussians in overreconstructed regions. Due to the high real-time requirements of SLAM systems, setting threshold parameters in M (p) allows the system to avoid the heavy computation associated with the gaussians densification method in [14].

After the process discussed in Section III-A, the scene representations contains three feature channels: spatial position, surface color, and potential semantics. The spatial position and surface color are directly obtained from the RGBD data stream. Meanwhile, the fusion of semantic encoding is supervised by the mask output from a pretrained segmentation model.

$$
\begin{array} { r } { L _ { c } = \lambda L _ { 1 } \left( I _ { r } , I _ { g t } \right) + \left( 1 - \lambda \right) \left[ 1 - s s i m \left( I _ { r } , I _ { g t } \right) \right] } \end{array}\tag{7}
$$

The color loss $L _ { c }$ is represented as a weighted combination of SSIM [14] and L1 loss as in Eq 7.

$$
L _ { d } = \sum _ { p i x } \left| D _ { p i x } ^ { r e n d e r } - D _ { p i x } ^ { g t } \right|\tag{8}
$$

The depth loss $L _ { d }$ is calculated as in Eq 8. During the mapping stage, the multi-channel loss is as shown in Eq 9, where $S _ { r e n d e r }$ represents the semantic labels after decoding

<!-- image-->  
Figure 2. The concept of virtual view pruning for identifying outlier gaussians. We analyze only the gaussians visible in the current ground-truth view (points A, B, C in the figure). Point A is not visible from either of the two virtual views, thus identified as an outlier gaussians, and its opacity is degraded during subsequent optimization. While the figure depicts two virtual views in a planar scenario, our approach creates four virtual cameras by rotating the camera pose from the focal point of each GT view frame along four directions: up, down, left, and right.

the semantic features and $S _ { h e a d }$ represents the class labels computed by the pretrained model. We use the cross-entropy loss $L _ { C E }$ to supervise the semantic channel.

$$
L _ { m a p p i n g } = \lambda _ { c } L _ { c } + \lambda _ { d } L _ { d } + \lambda _ { s } L _ { C E } \left( S _ { r e n d e r } , S _ { h e a d } \right)\tag{9}
$$

In Eq $9 , \lambda _ { d } , \lambda _ { s } .$ ,and $\lambda _ { c }$ are predefined hyperparameters used to assign weighted values to the depth, semantic, and color channels respectively.

3) Vitrual Camera View Pruning 3D Gaussians (VCVP): The key aspects of GS-based SLAM are: 1) Distinguishing well-established areas from areas requiring further optimization, and 2) Identifying and removing outlier points. The former resolves where to add gaussians, and also plays a key role in camera tracking. Areas of low quality can severely affect the accuracy of pose tracking. The second key aspect resolves where to delete gaussians. Outlier points will cause holes and defects during image rendering, and these flaws can also affect the accuracy of camera tracking.

The distinction between well-optimized and areas with low quality is implemented through Eq 6. This section discusses issues related to gaussians pruning.

Multi-view consistency constraints have been proven effective in identifying geometrically unstable gaussians. Previous methods [4] check whether gaussians inserted within the latest three frames of a keyframe window are recorded by other keyframes, thereby determining outlier gaussians. This method improves mapping accuracy by using collaborative constraints among multiple keyframes, but it increases computational costs and reduces real-time performance. Drastic viewpoint changes during SLAM cause significant overlap variations between keyframes, leading to errors in outlier detection.

In contrast to this method, the VCVP method proposed in this paper does not perform comparison between keyframes. Instead, it compares the viewpoint between a real camera frame and the corresponding virtual accompanying camera frame, as depicted in Fig 2. The virtual cameras $( V C )$ are created by rotating the real camera Â±Î¸ around the focal point:

<!-- image-->  
Figure 3. Rendered virtual camera views on the ScanNet dataset. The middle images provide a zoomed-in illustration of the effectiveness of Virtual Camera Pruning, where 'vcvp' denotes virtual camera view. Eliminating outlier gaussians not only improves rendering quality but also reduces the storage footprint of the map representation.

$V C _ { 1 }$ and $V C _ { 2 }$ by rotating on the horizontal plane (xz plane), and $V C _ { 3 }$ and $V C _ { 4 }$ by rotating on the vertical plane (yz plane).

The points A and B represent outlier gaussians, while the GT view denotes the camera pose estimated within the RGBD stream. In the current keyframe, both A and B are visible. However, in the $V C _ { 1 }$ , neither of these outlier points is visible, and in the $V C _ { 2 }$ ,only B is visible while A is not. The virtual camera operates alongside the real camera. If a gaussians is invisible in all virtual views but visible in the real view, it is then considered an outlier.

The virtual multi-view consistency check method takes advantage of the fast rendering capabilities of the Gaussian Splatting, enabling the marking of gaussians that significantly deviate from the object surface. The VCVP method eliminates the dependence on historical keyframes, allowing it to remain unaffected by drastic changes in camera views. This enables the identification of single-view outlier gaussians.

In subsequent optimization processes, the involvement of outlier gaussians in the scene is diminished by degrading their opacity. Consistent with [14], gaussians with near-zero opacity or excessive radius are removed in the mapping process. As illustrated in Fig 3, we render virtual views and further optimize the 3D gaussians parameters only for keyframes. The specific approach for generating virtual views is not fixed. Although Gaussian splatting enables extremely fast virtual view synthesis (nearly 300 FPS), introducing too many viewpoints can compromise the system's real-time performance. We conducted detailed tests in Section IV-C to evaluate how the generation and function of the virtual camera impact the performance of the SLAM system. We choose four virtual views along the up, down, left, and right directions, which achieves a desirable balance between effectiveness and efficiency.

4) Camera tracking: The camera tracking phase involves estimating the relative pose of the camera for each new frame, based on the already established map model. The camera pose for the new frame is initialized under the assumption of constant velocity, which includes both a constant linear and angular velocity.

<!-- image-->  
Figure 4. The first row shows the RGB reconstruction results. The second row shows the semantic labels predicted directly on the current frame using M2F [24]. The third row shows the semantic reconstruction results using the SGS-SLAM [6] method based on SplaTAM [5]. The fourth row shows the reconstruction results of our proposed model.

Table I  
COMPARISON EXPERIMENTS WITH OTHER METHODS ON MAP RECONSTRUCTION AND LOCALIZATION ACCURACY
<table><tr><td>Methods</td><td>Depth L1[cm]â</td><td></td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>ATE RMSE[cm] â</td></tr><tr><td>NICE-SLAM [3]</td><td>I</td><td>1.903</td><td>0.23</td><td>0.81</td><td>24.22</td><td>2.503</td></tr><tr><td>Vox-Fusion [25]</td><td>I</td><td>2.913</td><td>0.24</td><td>0.80</td><td>24.41</td><td>1.473</td></tr><tr><td>Co-SLAM [26]</td><td>I</td><td>1.513</td><td>0.336</td><td>0.94</td><td>30.24</td><td>1.059</td></tr><tr><td>ESLAM [27]</td><td>I</td><td>0.945</td><td>0.34</td><td>0.929</td><td>29.08</td><td>0.678</td></tr><tr><td>SplaTAM [5]</td><td>I</td><td>0.49</td><td>0.10</td><td>0.97</td><td>34.11</td><td>0.36</td></tr><tr><td>NEDS-SLAM(Ours)</td><td></td><td>0.47</td><td>0.088</td><td>0.962</td><td>34.76 I</td><td>0.354</td></tr></table>

The camera pose is subsequently refined iteratively by minimizing the tracking loss between the ground truth of the color, depth, and semantic channels and the gaussian rendered results from the camera's perspective.

$$
L _ { \mathrm { t r a c k i n g } } = ( \lambda _ { c } L _ { c } + \lambda _ { d } L _ { d } + \lambda _ { s } L _ { \mathrm { C E } } \left( S _ { \mathrm { r e n d e r } } , S _ { \mathrm { h e a d } } \right) ) \cdot M\tag{10}
$$

M in Eq 10 is computed as Eq 6. Artifacts and flaws such as holes and spurious effects caused by outlier gaussians significantly impact the precision of camera tracking. Subsequent experiments demonstrate that the incorporation of semantic loss improve the tracking accuracy. This improvement is attributed to the enriched understanding of the geometric information of objects, facilitated by the integration of semantic features.

## IV. EXPERIMENT

## A. Experimental Setup

Dataset. We evaluate our method on both synthetic and realworld datasets with semantic maps. Following other nerf-based and gaussian-based SLAM methods, for the reconstruction quality, we evaluate quantitatively on 8 synthetic scenes from Replica [29] and qualitatively on 6 scenes from ScanNet [30].

Table â¡I  
COMPARISON EXPERIMENT ON THE ATE RMSE METRIC
<table><tr><td>Methods</td><td></td><td>scene0000</td><td>scene0169</td><td>scene0181</td><td>scene0207</td><td>Avg.</td></tr><tr><td>NICE-SLAM [3]</td><td>I</td><td>12.00</td><td>10.90</td><td>13.40</td><td>6.20</td><td>10.63</td></tr><tr><td>Vox-Fusion [25]</td><td>I</td><td>68.84</td><td>27.28</td><td>23.30</td><td>9.41</td><td>I 32.21</td></tr><tr><td>Point-SLAM [28]</td><td>I</td><td>10.24</td><td>22.16</td><td>14.77</td><td>9.54</td><td>I 14.18</td></tr><tr><td>SplaTAM [5]</td><td>I</td><td>12.56</td><td>11.09</td><td>11.07</td><td>7.46</td><td>10.54</td></tr><tr><td>NEDS-SLAM(Ours)</td><td>I</td><td>12.34</td><td>11.21</td><td>10.35</td><td>6.56</td><td>10.12</td></tr></table>

Table III

<!-- image-->  
COMPARISON EXPERIMENT ON THE MIOU METRIC  
Rasterize sihouete our approach is ablstrve more details.  
struction quality in our study. These include the peak signal-tonoise ratio (PSNR), Depth-L1 (on 2D depth maps), Structural Similarity (SSIM [31]), and Learned Perceptual Image Patch Similarity (LPIPS [32]). Additionally, we assess the accuracy of camera pose estimation using the average absolute trajectory error (ATE RMSE [33]). To evaluate the performance of semantic segmentation, we calculate the mIoU (mean Intersection over Union) score.

Baselines. We compare the tracking and mapping with stateof-the-art methods NICE-SLAM [3], Co-SLAM [26], ESLAM [27], and SplaTAM [5]. For semantic segmentation accuracy, we compare with NIDS-SLAM [12], DNS-SLAM [11], and SNI-SLAM [10].

Implementation Details. We conducted experiments using a single NVIDIA RTX 4090 and an Intel Xeon Platinum 8358P, validating on the REPLICA dataset with the mapping iteration set to 40, tracking iteration set to 60, and SCFF iteration set to 50. After obtaining 384 feature channels through the DINO model, we derived 64-dimensional fused features by applying 2D convolutions separately to the Spatial Features. Finally, we obtained three-dimensional features by passing them through an encoder and embedding them into the gaussians parameters. We use a learning rate of 0.005 and 0.001 respectively for all learnable parameters on Replica and ScanNet datasets. For camera poses, we only employ a learning rate of 0.0005 in tracking.

## B. Experiment result

Quantitative measures of reconstruction quality using the Replica dataset are presented in Table I. The experiments on the ScanNet dataset can be found in Table II. The data shows that our method achieves the highest camera pose tracking accuracy. Our method demonstrates competitive performance when compared to other approaches. As shown in Fig 5, due to the VCVP method removing geometrically unstable gaussians,

<table><tr><td>Methods</td><td>AVG.mIoU[%] â</td><td>Room0</td><td>Room1</td><td>Office0</td></tr><tr><td>NIDS-SLAM [12]</td><td>82.37</td><td>82.45</td><td>84.08</td><td>85.94</td></tr><tr><td>DNS-SLAM [11]</td><td>84.77</td><td>88.32</td><td>84.90</td><td>84.66</td></tr><tr><td>SNI-SLAM [10]</td><td>87.41</td><td>I 88.42</td><td>87.43</td><td>87.63</td></tr><tr><td>Ours</td><td>90.78</td><td>I 90.73</td><td>91.20</td><td>90.42</td></tr></table>

the foundation of 3DGS, n and semantic reconstrucdes a comparison between cit approaches in terms of pe.

Due to the precise representation of object edges offered by the 3DGS, our methougerring about significant improvements in semantic reconstruction. Other methods have not considered the issue of spatially inconsistent semantic estimation by pretrained semantic segmentation models on consecutive RGBD frame inputs. Therefore, for the sake of fair performance comparisonin Table III, we usedthe ground truth per-pixel semantic class labels as input. More detailed experiments on the SCFF module are conducted in Table V in Section IV-C.

When testing the Mask2Former model on the replica room0 scene, as shown in Fig 4, there are noticeable inconsistencies in the predictions for the floor and chairs. This affects the semantic reconstruction quality. As shown in Fig 4, NEDS-SLAM effectively filters out the negative impact of spatial semantic inconsistencies, generating robust semantic estimates and providing more accurate semantic reconstruction.

## C. Ablation Study

## Effectiveness of VCVP Module.

The VCVP method involves two subproblems: (1) determining the number of virtual camera views to generate and how to generate them, and (2) deciding on which frame or frames to perform VCVP operation. The solutions to these subproblems will impact the computational costs of the VCVP modules. In Table IV, the data in the third and fourth rows labeled 'A/B/C' indicates that we used three configurations for the calculations. Configuration A and B represent generating two virtual views in the horizontal and vertical directions, respectively. Configuration C represents generating four virtual views simultaneously in both horizontal and vertical directions.

The VCVP module significantly enhances scene modeling accuracy and camera pose tracking precision. Increasing the number of virtual camera views and their usage within the keyframes window can achieve the best camera localization accuracy, but this also increases computational overhead. In our most extreme test case, VCVP detection was performed on 10 keyframes during each mapping iteration, with four virtual camera viewpoints rendered for each detection, resulting in nearly a 50% improvement in pose accuracy

frame 379  
frame 389  
frame 419  
<!-- image-->  
Figure 6. The validation results on the Scannet scene0000_00 dataset. The first row indicates the RGB reconstruction results of NEDS-SLAM, the second row indicates the semantic features predicted by M2F, the third row is the semantic reconstruction results without the Spatially Consistent Feature Fusion (SCFF) module, and the fourth row is the results with the SCFF module.

Table IV  
ABLATION EXPERIMENTS ON THE VCVP MODULE CONDUCTED IN REPLICA ROOMO.
<table><tr><td>Settings</td><td>ATE RMSE â</td><td>AVG SSIM â</td><td>Scene Embedding â</td><td>Mapping /iteration â</td></tr><tr><td>Base1</td><td>0.42</td><td>0.90</td><td>100.16 MB</td><td>14 ms</td></tr><tr><td>Base + VCVP_w5*</td><td>0.30/0.34/0.28</td><td>0.91/0.93/0.96</td><td>90.45 MB</td><td>16/16/20 ms</td></tr><tr><td>Base + VCVP_w10</td><td>0.26/0.27/0.22</td><td>0.92/0.92/0.97</td><td>88.93 MB</td><td>18/18/26 ms</td></tr><tr><td>Base + RCVp2</td><td>0.36</td><td>0.95</td><td>95.27 MB</td><td>20 ms</td></tr><tr><td>SpaTAM [5]</td><td>0.36</td><td>0.98</td><td>T00.00 MB</td><td>24 ms</td></tr><tr><td>Co-SLAM [26]</td><td>0.97</td><td>0.91</td><td></td><td>13 ms</td></tr><tr><td>NICE-SLAM [3]</td><td>0.99</td><td>0.69</td><td>48.48 MB</td><td>66 ms</td></tr></table>

1 Base refers to the configuration without SCFF, without lightweight encoder, and without VCVP, implementing only the 3DGS dense SLAM functionality.  
\*VCVP1d ec 510 orent e VCVP operations.  
2 RCVP involves using real camera views for consistency checks and removing outlier gaussians.

We conducted another experiment comparing our method to the density control method (denote as DC method) from the original 3DGS paper, as in Table.VII. In experiments on three scenes from the TUM RGBD dataset, the DC method achieved ATE RMSE values of 3.62, 1.41, and 6.63, which are higher than those of GS-SLAM, which also uses partial DC operations (3.3, 1.3, and 6.6 respectively). Our VCVP method demonstrated even higher performance.

## Effectiveness of SCFF Module.

Following SGS-SLAM, we directly incorporated semantic parameters into the 3D gaussians by calling a pre-trained M2F segmentation model [24] on each RGB frame. As shown in the third row in Fig 4, corresponding to the Base_S settings in Table V. SCFF_wo_SFE represents configurations includes the SCFF module, but does not use SFE. For the ScanNet scene0000 dataset, the M2F model achieved a semantic segmentation mIoU of 52.4. Using M2F for segmentation head gave an average mIoU of 26.52, serving as the baseline.

Table V  
ABLatIoN Study of THE SCFF MODULE on ScANNET DatAsET.
<table><tr><td>Settings</td><td>mIoU â</td><td>Mapping /iteration â</td><td>Scene Embedding â</td></tr><tr><td>Base_S</td><td>26.52%</td><td>28 ms</td><td>123.08 MB</td></tr><tr><td>Base_S + SCFF_wo_SFE</td><td>30.24%</td><td>86 </td><td>405.64 MB</td></tr><tr><td>Base_S+SCFF_w_SFE</td><td>42.18%</td><td>6 ms</td><td>410.38 MB</td></tr><tr><td>Base_S + SCFF_w_SFE + encoder</td><td>40.81%</td><td>5ms</td><td>141.93 MB</td></tr></table>

Table VI

RUNTIME PERFORMANCE COMPARISON OF NEDS-SLAM ON TWO DIFFERENT HARDWARE PLATFORMS.
<table><tr><td rowspan="3">Hardware Settings</td><td colspan="2"># replica room0</td><td colspan="2"># TUM RGBD fr1/desk</td></tr><tr><td>Tracking/it â</td><td>Mapping/it â</td><td>Tracking/it â</td><td>Mapping/it â</td></tr><tr><td>Platform A</td><td>28 ms</td><td>35 ms</td><td>26 ms</td><td>34 ms</td></tr><tr><td>Platform B</td><td>42 ms</td><td>76 ms</td><td>42 ms</td><td>75 ms</td></tr></table>

As can be seen in Fig 6, the semantic features calculated by the M2F model were inconsistent (such as the partitions and books on the table). After processing with the SCFF module, the inconsistencies were resolved and NEDS-SLAM output a more complete semantic reconstruction. The SCFF module filters out unstable semantic estimations between frames, resulting in more accurate semantic reconstruction. Our designed SCFF features a lightweight network structure, which does not significantly increase inference time. In fact, the time consumption in the SLAM process (Mapping/Iteration in the table) mainly arises from optimizing a large number of 3D gaussians. Therefore, our specially designed encoder compresses the semantic features and embeds them into the gaussian parameters, reducing the number of parameters and thereby increasing the mapping speed.

## Runtime Comparison.

As shown in the last column of Table VI, our lightweight configuration of NEDS-SLAM achieves faster mapping speeds than SplaTAM while maintaining more accurate camera pose tracking precision. With higher configurations, NEDS-SLAM offers better performance, though the computation speed decreases. We ran NEDS-SLAM on different hardware platforms and datasets. Platform A is as in section IV-A. Platform B consists of an Intel i9-13900K and a single NVIDIA RTX 4060Ti. The results show that both hardware platforms achieve similar camera pose tracking accuracy and reconstruction accuracy with NEDS-SLAM, but the model takes more time to run on Platform B compared to Platform A.

Table VII  
COMPARISON OF THE VCVP MODULE WITH THE ORIGINAL DENSITY CONTROL METHOD ON TUM-RGBD DATASET.
<table><tr><td rowspan="2">DATASETS</td><td colspan="2">with VCVP</td><td colspan="2">Original density control method as in [14]</td></tr><tr><td>ATE RMSE â</td><td>AVG SSIM â</td><td>ATE RMSEâ</td><td>AVG SSIM â</td></tr><tr><td>Fr1/desk1</td><td>3.30</td><td>0.91</td><td>3.62</td><td>0.93</td></tr><tr><td>Fr2/xyz</td><td>1.13</td><td>0.95</td><td>1.41</td><td>0.95</td></tr><tr><td>Fr3/off</td><td>4.94</td><td>0.90</td><td>6.63</td><td>0.92</td></tr></table>

Table VIII

VERIFICATION OF THE EFFECTIVENESS OF THE SCFF MODULE
<table><tr><td>Model Settings</td><td>M2F [24]</td><td>M2F+SCFF</td><td>MRCNN [34]</td><td>MRCNN+SCFF</td></tr><tr><td>AVG mIoU</td><td>25.89%</td><td>36.25%</td><td>24.34%</td><td>34.07%</td></tr></table>

## V. Conclusion and LimitationS

The proposed NEDS-SLAM is an end-to-end semantic SLAM system based on 3DGS. By integrating a Spatially Consistent feature fusion model, NEDS-SLAM effectively addresses the challenges of robustly estimating semantic labels with pre-trained models, significantly enhancing semantic reconstruction performance. The Virtual Camera View Pruning method uses differentiable Gaussian splatting for quick and realistic novel view synthesis. It removes outlier gaussians during SLAM, significantly improving the reconstruction quality of neural radiance fields.

The experiment with public datasets confirmed NEDS-SLAM's effectiveness but revealed some shortcomings. The virtual camera view pruning method improves mapping speed by removing more gaussians. However, increasing the frequency of VCVP usage also raises computational load, indicating room for further optimization. Future plans include optimizing and incorporating semantic reconstruction for dynamic scenes.

## REFERENCES

[1] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon, "Kinectfusion: Real-time dense surface mapping and tracking," in 2011 10th IEEE international symposium on mixed and augmented reality. Ieee, 2011, pp. 127136.

[2] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, "imap: Implicit mapping and positioning in real-time," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 62296238.

[3] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, "Nice-slam: Neural implicit scalable encoding for slam," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 78612 796.

[4] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, "Gaussian Splatting SLAM," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[5] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, "Splatam: Splat, track & map 3d gaussians for dense rgb-d slam," arXiv preprint arXiv:2312.02126, 2023.

[6] M. Li, S. Liu, and H. Zhou, "Sgs-slam: Semantic gaussian splatting for neural dense slam," arXiv preprint arXiv:2402.03246, 2024.

[7] A. Hermans, G. Floros, and B. Leibe, "Dense 3d semantic mapping of indoor scenes from rgb-d images," in 2014 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2014, pp. 26312638.

[8] G. Narita, T. Seno, T. Ishikawa, and Y. Kaji, "Panopticfusion: Online volumetric semantic mapping at the level of stuff and things," in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019, pp. 42054212.

[9] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, "Nerf: Representing scenes as neural radiance fields for view synthesis," Communications of the ACM, vol. 65, no. 1, pp. 99106, 2021.

[10] S. Zhu, G. Wang, H. Blum, J. Liu, L. Song, M. Pollefeys, and H. Wang, "Sni-slam: Semantic neural implicit slam," arXiv preprint arXiv:2311.11016, 2023.

[11] K. Li, M. Niemeyer, N. Navab, and F. Tombari, "Dns slam: Dense neural semantic-informed slam," arXiv preprint arXiv:2312.00204, 2023.

[12] Y. Haghighi, S. Kumar, J.-P. Thiran, and L. Van Gool, "Neural implicit dense semantic slam," arXiv preprint arXiv:2304.14560, 2023.

[13] F. Tosi, Y. Zhang, Z. Gong, E. SandstrÃ¶m, S. Mattoccia, M. R. Oswald, and M. Poggi, "How nerfs and 3d gaussian splatting are reshaping slam: a survey," arXiv preprint arXiv:2402.13255, vol. 4, 2024.

[14] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, "3d gaussian splatting for real-time radiance field rendering," ACM Transactions on Graphics, vol. 42, no. 4, 2023.

[15] C. Yan, D. Qu, D. Wang, D. Xu, Z. Wang, B. Zhao, and X. Li, "Gsslam: Dense visual slam with 3d gaussian splatting," arXiv preprint arXiv:2311.11700, 2023.

[16] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, "Gaussian-slam: Photo-realistic dense slam with gaussian splatting," arXiv preprint arXiv:2312.10070, 2023.

[17] G. Chen and W. Wang, "A survey on 3d gaussian splatting," arXiv preprint arXiv:2401.03890, 2024.

[18] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, "Dinov2: Learning robust visual features without supervision," arXiv:2304.07193, 2023.

[19] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, "Langsplat: 3d language gaussian splatting," arXiv preprint arXiv:2312.16084, 2023.

[20] B. Dou, T. Zhang, Y. Ma, Z. Wang, and Z. Yuan, "Cosseggaussians: Compact and swift scene segmenting 3d gaussians," arXiv preprint arXiv:2401.05925, 2024.

[21] X. Zuo, P. Samangouei, Y. Zhou, Y. Di, and M. Li, "Fmgs: Foundation model embedded 3d gaussian splatting for holistic 3d scene understanding," arXiv preprint arXiv:2401.01970, 2024.

[22] Y. Yuan and A. NÃ¼chter, "Uni-fusion: Universal continuous mapping," IEEE Transactions on Robotics, 2024.

[23] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, "Depth anything: Unleashing the power of large-scale unlabeled data," arXiv preprint arXiv:2401.10891, 2024.

[24] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, Masked-attention mask transformer for universal image segmentation, 2022.

[25] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, "Voxfusion: Dense tracking and mapping with voxel-based neural implicit representation," in 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2022, pp. 499507.

[26] H. Wang, J. Wang, and L. Agapito, "Co-slam: Joint coordinate and sparse parametric encodings for neural real-time slam," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 29313 302.

[27] M. M. Johari, C. Carta, and F. Fleuret, "Eslam: Efficient dense slam system based on hybrid representation of signed distance fields," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 40817 419.

[28] E. SandstrÃ¶m, Y. Li, L. Van Gool, and M. R. Oswald, "Point-slam: Dense neural point cloud-based slam," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 43318 444.

[29] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, et al., "The replica dataset: A digital replica of indoor spaces," arXiv preprint arXiv:1906.05797, 2019.

[30] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. NieBner, "Scannet: Richly-annotated 3d reconstructions of indoor scenes," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 58285839.

[31] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE transactions on image processing, vol. 13, no. 4, pp. 600612, 2004.

[32] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586595.

[33] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, "A benchmark for the evaluation of rgb-d slam systems," in 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012, pp. 573580.

[34] W. Abdulla, "Mask r-cnn for object detection and instance segmentation on keras and tensorflow," https://github.com/matterport/Mask_RCNN, 2017.