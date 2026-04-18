# PanoSLAM: Panoptic 3D Scene Reconstruction via Gaussian SLAM

Runnan Chen1 Zhaoqing Wang1 Jiepeng Wang2

Yuexin Ma3 Mingming Gong4 Wenping Wang5 Tongliang Liu1

1The University of Sydney 2The University of Hong Kong

3ShanghaiTech University 4The University of Melbourne 5Texas A&M University

## Abstract

Understanding geometric, semantic, and instance information in 3D scenes from sequential video data is essential for applications in robotics and augmented reality. However, existing Simultaneous Localization and Mapping (SLAM) methods generally focus on either geometric or semantic reconstruction. In this paper, we introduce PanoSLAM, the first SLAM system to integrate geometric reconstruction, 3D semantic segmentation, and 3D instance segmentation within a unified framework. Our approach builds upon 3D Gaussian Splatting, modified with several critical components to enable efficient rendering of depth, color, semantic, and instance information from arbitrary viewpoints. To achieve panoptic 3D scene reconstruction from sequential RGB-D videos, we propose an online Spatial-Temporal Lifting (STL) module that transfers 2D panoptic predictions from vision models into 3D Gaussian representations. This STL module addresses the challenges of label noise and inconsistencies in 2D predictions by refining the pseudo labels across multi-view inputs, creating a coherent 3D representation that enhances segmentation accuracy. Our experiments show that PanoSLAM outperforms recent semantic SLAM methods in both mapping and tracking accuracy. For the first time, it achieves panoptic 3D reconstruction of open-world environments directly from the RGB-D video.

## 1. Introduction

Semantic Simultaneous Localization and Mapping (SLAM) combines scene reconstruction and camera pose estimation with semantic scene understanding, providing a more comprehensive interpretation of the environment than traditional SLAM. By generating 3D semantic maps, semantic SLAM facilitates applications in diverse fields, including autonomous driving, robotic navigation, and digital city planning. These semantic maps contribute to advanced decision-making and environment interaction, making semantic SLAM a cornerstone technology in intelligent systems.

<!-- image-->  
Figure 1. We present PanoSLAM, a SLAM system based on 3D Gaussian Splatting, capable of 3D geometric, semantic and instance reconstruction from unlabeled RGB-D videos.

However, existing semantic SLAM methods [14, 29, 30, 68, 69] face limitations in fully capturing the panoptic nature of 3D scenes, an essential aspect that includes both instance and semantic-level details. Additionally, these methods generally rely on densely labeled scenes for semantic mapping, a requirement that is labor-intensive, timeconsuming, and cost-prohibitive, especially in open-world environments where scene diversity makes manual labeling impractical. This challenge underscores a fundamental question: How can we reconstruct a panoptic 3D scene from sequential video data without the need for manual semantic annotation?

Our approach is inspired by recent breakthroughs in vision foundation models such as CLIP [44] and SAM [24], which demonstrate remarkable zero-shot perception capabilities across diverse environments. These foundation models have opened up new possibilities for transferring 2D vision knowledge to 3D representations, which include point clouds [3, 6, 39], Neural Fields [20, 51], and 3D Gaussians [43, 50], enabling label-free 3D scene understanding. However, the application of these models in SLAM has been hindered by substantial offline optimization requirements, which conflict with the online demands of SLAM.

Recent advancements in Gaussian Splatting [19] have shown promising results for scene reconstruction by leveraging 3D Gaussian representations, which allow for highquality and efficient rendering through a splatting-based approach. Some SLAM systems [18, 36, 62, 66] have adopted 3D Gaussian Splatting to achieve photorealistic scene mapping. Yet, integrating vision foundation models into SLAM systems for open-world 3D scene understanding remains an unexplored area, leaving a gap in fully panoptic SLAM reconstruction methods.

In this work, we address this gap by introducing PanoSLAM, a novel SLAM framework that enables panoptic 3D scene reconstruction from unlabeled RGB-D video input. Our method builds upon the Gaussian Splatting technique, enhanced with critical modifications for panoptic rendering, including semantic Gaussian initialization, densification, and panoptic segmentation formulation. A primary challenge in our approach is dealing with label noise, as we rely on pseudo-labels from 2D panoptic predictions provided by vision models. These pseudo-labels are susceptible to noise, such as inconsistencies in mask predictions and class labels across different views, which can create conflicts during optimization and degrade semantic map quality. To tackle this challenge, we introduce a Spatial-Temporal Lifting (STL) module that refines the noisy pseudo-labels by projecting them into 3D space, leveraging multi-view consistency to enhance the reliability of labels in 3D. Our STL module integrates multi-view 2D panoptic predictions to create a cohesive 3D representation, addressing label noise and facilitating high-quality panoptic scene reconstruction.

We evaluate PanoSLAM on benchmark datasets, including Replica [52] and ScanNet++ [64], where our method significantly outperforms recent semantic SLAM approaches in mapping and tracking accuracy. Notably, PanoSLAM is the first method to achieve panoptic 3D scene reconstruction without manual labels. Our work combines efficient Gaussian Splatting with vision foundation models, to extend the possibilities of panoptic 3D reconstruction in diverse open-world environments.

In summary, our contributions are as follows:

â¢ We introduce the first panoptic 3D scene reconstruction method based on Gaussian Splatting within a SLAM framework.

â¢ We propose an innovative Spatial-Temporal Lifting module for consistent 2D-to-3D knowledge distillation across multiple views, addressing challenges with noisy labels in panoptic reconstruction.

â¢ Our experimental results demonstrate that PanoSLAM achieves state-of-the-art performance, pioneering labelfree panoptic 3D scene reconstruction.

## 2. Related Work

Scene Understanding. Scene understanding, which focuses on recognizing objects and their relationships, is essential in fields like robotics, autonomous driving, and smart cities. While supervised methods have significantly advanced both 2D and 3D scene understanding [7â11, 15, 26, 27, 33, 42, 53, 55, 58, 59, 61, 65, 70, 71], they rely on extensive annotations, limiting their adaptability to new object categories outside the training data. To address these limitations, some approaches focus on open-world scene understanding [1, 2, 4, 4, 5, 13, 16, 28, 31, 34, 35, 37, 45, 60, 67], while others [3, 6, 40, 41, 49, 63] aim to reduce 3D annotation demands by leveraging knowledge from 2D networks. However, these methods often face challenges in open-world scenarios where label-free, real-time understanding is crucial. Vision foundation models like CLIP [44] and SAM [24] have shown strong potential in open-world tasks. Efforts like CLIP2Scene [3] and CNS [6] transfer 2D vision model knowledge to 3D representations (e.g., point clouds, Neural Fields, and 3D Gaussians) for label-free 3D understanding, often using a 2D-3D calibration matrix. However, these approaches typically require extensive offline optimization, which restricts their online application in SLAM. Our work integrates online SLAM with knowledge from 2D vision models, enabling efficient reconstruction of 3D panoptic semantic maps from unlabeled RGB-D videos.

Semantic SLAM. Traditional SLAM methods follow frameworks like MonoSLAM [12], PTAM [25], and ORB SLAM [38], which separate tasks into mapping and tracking. Recent approaches leverage neural implicit representation and rendering for dense SLAM [17, 48, 54, 57, 68, 72]. Gaussian Splatting [19], based on 3D Gaussians, has emerged as a more efficient alternative for 3D scene reconstruction [19, 21, 22, 32, 56], inspiring Gaussian-based SLAM methods [18, 36, 62, 66]. For example, SplaTAM [18] uses silhouette-guided optimization for dense mapping, while Gaussian SLAM [36] incorporates Gaussian insertion and pruning for monocular SLAM.

Semantic SLAM methods [14, 29, 30, 68, 69] enhance environmental understanding by integrating semantic information into SLAM. Object-aware systems like SLAM++ [47] and Kimera [46] capture object-level or semantic mesh information, while recent methods such as SGS-SLAM [30] and SemGauss-SLAM [69] apply 3D Gaussian Splatting for semantic SLAM. However, these methods depend on manual labels and struggle with openworld generalization, limiting their scalability and adaptability to diverse, unstructured environments. In contrast, this paper proposes a Gaussian-based SLAM system that reconstructs 3D panoptic semantic maps in open-world environments without any input labels, achieving robust, labelfree scene understanding suitable for online applications.

<!-- image-->  
Figure 2. Overview of the PanoSLAM framework for panoptic 3D scene reconstruction from unlabeled RGB-D videos. The system comprises four main components: (1) Camera Tracking: Estimating camera poses from input RGB-D videos to facilitate 3D mapping. (2) Panoptic Information Inference: Utilizing 2D vision models to generate multi-view 2D panoptic pseudo-labels, which are re-projected into 3D space for label refinement. (3) 3D Gaussians Updating: Constructing and updating a 3D Gaussian map for efficient rendering and mapping. (4) Spatial-Temporal Lifting: Refining 3D panoptic pseudo-labels through consistent optimization across multiple views to address label noise, enabling high-quality panoptic scene reconstruction.

## 3. Methodology

Current semantic SLAM methods fall short in capturing both instance-level and semantic details, collectively known as panoptic information. In this section, we introduce a new method, called PanoSLAM, designed to efficiently reconstruct 3D panoptic semantic maps from unlabeled RGB-D videos. PanoSLAM achieves this by transferring the knowledge of a 2D vision model into an online SLAM system. Specifically, we enhance a Gaussian-based SLAM framework with the capability to render panoptic information through targeted modifications to the Gaussian Splattingbased SLAM. Additionally, we develop a novel Spatial-Temporal Lifting module to handle noisy labels effectively. The following sections provide a brief overview of the Gaussian-based SLAM system, detail our modifications, and present the Spatial-Temporal Lifting module.

## 3.1. Preliminaries about Gaussian-based SLAM

Our system builds upon SplaTAM [18], a state-of-the-art dense RGB-D SLAM solution that leverages 3D Gaussian Splatting. SplaTAM represents the environment as a collection of 3D Gaussians, which can be rendered into highquality color and depth images. By utilizing differentiable rendering and gradient-based optimization, it jointly optimizes the camera pose for each frame and constructs a volumetric map of the scene. In the following sections, we provide a brief overview of 3D Gaussian Splatting, followed by a detailed description of each module in SplaTAM.

Gaussian Representation. SplaTAM simplifies the Gaussian representation by adopting view-independent color and ensuring isotropy of the Gaussians. Each Gaussian is thus characterized by only eight parameters: three values for its RGB color $\mathbf { c } \in \mathbb { R } ^ { 3 }$ , three for its center position $\mathbf { u } \in \mathbb { R } ^ { 3 }$ , one for the radius (standard deviation) r, and one for opacity o. The contribution of each Gaussian to a point in 3D space $\textbf { x } \in { \mathbb { R } } ^ { 3 }$ is determined by the standard (unnormalized) Gaussian function, modulated by its opacity:

$$
f ( x ) = o \exp \left( - \frac { \| \textbf { x } - \textbf { u } \| ^ { 2 } } { 2 r ^ { 2 } } \right) .\tag{1}
$$

Differentiable Rendering via Splatting. SplaTAM renders an RGB image by sorting a collection of 3D Gaussians from front to back and then efficiently alpha-compositing the splatted 2D projection of each Gaussian in pixel space. The color of a rendered pixel $P = ( u , v )$ can be expressed as:

$$
C ( P ) = \sum _ { i = 1 } ^ { n } c _ { i } f _ { i } ( P ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - f _ { i } ( P ) ) ,\tag{2}
$$

where $f _ { i } ( P )$ is calculated according to Eq. 1, using the u and r values of the splatted 2D Gaussians in pixel space:

$$
{ \bf u } ^ { 2 D } = G \frac { E _ { t } { \bf u } } { d } , ~ r ^ { 2 D } = \frac { f r } { d } , ~ d = ( E _ { t } { \bf u } ) _ { z } ,\tag{3}
$$

where the variables G, $E _ { t } , f ,$ and d represent the camera intrinsic matrix, extrinsic matrix for the rotation and translation of the camera at frame t, focal length (known), and the depth of the i-th Gaussian in camera coordinates, respectively.

## 3.2. PanoSLAM

Our PanoSLAM method, built upon Gaussian-based SLAM, integrates 2D vision model knowledge into a Gaussian SLAM system to efficiently reconstruct 3D panoptic maps from unlabeled RGB-D videos. To enable this, we implement key modifications for Splatting-based SLAM, allowing it to render panoptic information through semantic Gaussian initialization, densification, and panoptic segmentation formulation. The system leverages the 2D vision modelâs panoptic predictions, including instance and semantic labels, as pseudo-labels to guide 3D panoptic map reconstruction. However, these pseudo-labels are susceptible to label noise, such as inconsistent mask and class predictions across views, which can create optimization conflicts and degrade the quality of the semantic map. To mitigate the effects of noisy pseudo-labels, we introduce a Spatial-Temporal Lifting module. In the following, we explain the steps and insights of PanoSLAM in detail.

Semantic Gaussian Representation, Initialization, and Densification. In contrast to SplaTAM, where each Gaussian is parameterized with 8 values, PanoSLAM assigns 13 values per Gaussian to facilitate semantic rendering. These include three values for its semantic embedding $\textbf { s } \in \ \mathbb { R } ^ { 3 }$ one for its semantic radius rË (standard deviation), and one for semantic opacity $\hat { o } ,$ along with the same 8 parameters as in SplaTAM. The semantic embedding of a pixel $P = ( u , v )$ is thus rendered as:

$$
S ( P ) = \sum _ { i = 1 } ^ { n } \mathbf { s } _ { i } { \hat { f } } _ { i } ( P ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - { \hat { f } } _ { i } ( P ) ) ,\tag{4}
$$

where $\hat { f } _ { i } ( P )$ is calculated using Eqs. 1 and 3 with rË and oË values.

To initialize the Gaussians in the first frame, we set the camera pose to identity and add a new Gaussian for each pixel. The center u is determined by unprojecting the pixel depth, and both the radius r and semantic radius rË are set to a one-pixel radius in the 2D image, i.e., $\begin{array} { r } { \hat { r } = r = \frac { D _ { G T } } { f } } \end{array}$ . The color c and semantic embedding s are initialized to the pixel color, while both optical opacity o and semantic opacity oË are initialized to 0.5.

For densification in subsequent frames, we add Gaussian kernels at locations where existing ones fail to adequately capture the sceneâs geometry or semantics. To identify these areas, we define a densification mask as follows:

$$
\begin{array} { r } { M ( P ) = ( F ( P ) < 0 . 5 ) ~ \mathrm { o r } ~ ( \hat { F } ( P ) < 0 . 5 ) ~ \mathrm { o r } } \\ { ( L ( D ( P ) ) > T ) , } \end{array}\tag{5}
$$

where $D ( P )$ is the differentiably rendered depth:

$$
D ( P ) = \sum _ { i = 1 } ^ { n } d _ { i } f _ { i } ( P ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - f _ { i } ( P ) ) ,\tag{6}
$$

and $\hat { F } ( P )$ is the silhouette map for visibility determination, calculated using oË and rË:

$$
\hat { F } ( P ) = \sum _ { i = 1 } ^ { n } \hat { f } _ { i } ( P ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - \hat { f } _ { i } ( P ) ) ,\tag{7}
$$

where $T$ is a threshold set to 50 times the median depth error. The densification mask serves a dual purpose: identifying areas where the map lacks density $( S < 0 . 5 )$ in geometry or semantics and detecting discrepancies between the ground-truth depth and the rendered depth. For each pixel flagged by this mask, we introduce a new Gaussian using the same initialization as in the first frame.

Camera Tracking. Camera tracking involves estimating the camera pose of the current incoming online RGB-D image in the input stream. The camera pose is initialized for a new timestep through a constant velocity forward projection of the pose parameters in the camera centre + quaternion space:

$$
E _ { t + 1 } = E _ { t } + ( E _ { t } - E _ { t - 1 } ) .\tag{8}
$$

The camera pose $E _ { t }$ is then iteratively refined through gradient-based optimization by differentiably rendering RGB, depth, and silhouette maps and adjusting the camera parameters to minimize the loss, while fixing Gaussian parameters.

Panoptic Segmentation Formulation. Panoptic segmentation is achieved by combining semantic and instance segmentation. Inspired by Maskformer [8], we decompose the panoptic segmentation task into two steps: 1) segmenting the image into N regions using binary masks, and 2) assigning each region a probability distribution over K categories. Specifically, for each pixel $\boldsymbol { P } = \left( u , v \right)$ , we first generate region predictions:

$$
R ( P ) = \Gamma ( S ( P ) ) \otimes \mathbb { M } ,\tag{9}
$$

where $R ( P ) \in \mathbb { R } ^ { N }$ represents the predicted distribution over N regions, $\mathbb { M } \in \mathbb { R } ^ { N \times H }$ is a set of N region embeddings with H dimensions, Î(â¢) is an MLP decoder that elevates the semantic embedding dimension from 3 to H, and â denotes matrix multiplication. The N regions are then classified into K categories as follows:

$$
O ( { \mathbb { M } } ) = { \mathbb { M } } \otimes { \mathbb { C } } ,\tag{10}
$$

where $\mathbb { C } \in \mathbb { R } ^ { N \times K }$ serves as the classifier, and $O ( \mathbb { M } ) \ \in$ $\mathbb { R } ^ { K }$ represents the predicted distribution over K categories. These definitions effectively assign a pixel at location $\boldsymbol { P } ~ = ~ ( u , v )$ to the i-th probability-region and $j \mathrm { - t h }$ probability-category only if both the region prediction probability $R ^ { i } ( P )$ and the highest class probability $O ^ { j } ( \mathbb { M } )$ are sufficiently high [8]. Low-confidence predictions are discarded before inference, and segments with substantial parts of their binary masks occluded by other predictions are also removed.

Spatial-Temporally Lifting. Given the absence of ground truth labels, for each image frame of the input video sequences, we utilize the off-the-shelf 2D vision model to perform panoptic prediction for each pixel $P = ( u , v )$ including both instances $\{ \hat { R } _ { t } ( P ) \} _ { t = 1 } ^ { T }$ and semantics predictions $\{ \hat { O } _ { t } ( P ) \} _ { t = 1 } ^ { T }$ . Here, T is the number of time steps $( \mathrm { i . e . , }$ , image frames) to be processed. These predictions are then utilized as the pseudo labels to optimize the panoptic segmentation results from PanoSLAM. However, these predictions usually contain noisy labels that may corrupt the optimization process.

To address this issue, we unify the panoptic prediction across different time steps over the shared 3D shape through multi-view correspondence. To facilitate this, given two pixels $P ^ { * 1 }$ and $P ^ { * 2 }$ belonging to different views, if their respective unprojected 3D points $\mathbf { P } ^ { * 1 }$ and $\mathbf { P } ^ { * 2 }$ are located in the same local small voxel space, we regard these two pixels as being in correspondence. Specifically, to efficiently identify corresponding pixels across the T frames, we first re-project the pixel depths of each frame to the world coordinates according to their corresponding camera poses and intrinsic (Eq. 11). These reprojected 3D points are then fused to form a point cloud. The 3D space occupied by this point cloud is uniformly split into voxels with a specified size $S _ { n }$ , and we quantize all these points into the centers of voxels directly, allowing us to quickly obtain the corresponding voxel indices. Finally, the points sharing the same voxel index indicate that they are located in the same local small voxel and inherently should have consistent semantic labels for the corresponding pixels across the T frames. This approach allows us to efficiently determine the correspondence of pixels and unify the semantic labels of these pixels.

Specifically, we unprojected the 2D pixel $P _ { t }$ to the 3D location $\mathbf { P } _ { t } \in \mathbb { R } ^ { 3 }$ by following:

$$
{ \bf P } _ { t } = E _ { t } ^ { - 1 } G ^ { - 1 } d P _ { t } ,\tag{11}
$$

where $E _ { t } ^ { - 1 }$ is the inverse camera pose at the t time step, and d and G are the ground truth depth and camera intrinsic matrix, respectively.

In the next, we set all region predictions to be identical within the same local voxel $g _ { n } , i . e .$

$$
\hat { R } ( P ^ { * } ) = \frac { 1 } { | g _ { n } | } \sum _ { * \in g _ { n } } \hat { R } ( P ^ { * } ) ,\tag{12}
$$

To this end, we obtain the spatial-temporal refined region prediction $\{ \hat { R } _ { t } ( P ^ { * } ) \} _ { t = 1 } ^ { T }$ Notably, in our panoptic segmentation formulation Eq. 10, the semantics predictions $\{ \hat { O } _ { t } ( \mathbb { M } ) \} _ { t = 1 } ^ { T }$ of each pixel are adjusted accordingly when their region prediction is changed.

We optimize the Gaussian SLAM system by minimizing a color rendering loss $L _ { 1 } \left( C _ { t } ( P ) , C _ { G T } \right)$ , a depth rendering loss $L _ { 1 } ( D _ { t } ( P ) , D _ { G T } )$ , and panoptic prediction loss for $R _ { t } ( P )$ and $O _ { t } ( \mathbb { M } )$ . The objective function is as follows:

$$
\begin{array} { r } { \mathbb { L } = \displaystyle \frac { 1 } { T } \sum _ { t \in T } \sum _ { P } \lambda _ { 1 } L _ { 1 } \left( C _ { t } ( P ) , C _ { G T } \right) } \\ { + \lambda _ { 2 } L _ { 1 } ( D _ { t } ( P ) , D _ { G T } ) } \\ { + \lambda _ { 3 } C E ( O _ { t } ( \mathbb { M } ) , \hat { O } _ { t } ( \mathbb { M } ) ) } \\ { + \lambda _ { 4 } D I C E ( R _ { t } ( P ) , \hat { R } _ { t } ( P ^ { * } ) ) } \\ { + \lambda _ { 5 } S i g _ { F } ( R _ { t } ( P ) , \hat { R } _ { t } ( P ^ { * } ) ) , } \end{array}\tag{13}
$$

where $\lambda _ { 1 , \ldots , 5 }$ are the loss weights. $L _ { 1 }$ indicates the $L _ { 1 }$ loss, $C E$ is the cross-entropy loss, and $D I C E$ and $S i g _ { F }$ are the dice loss and sigmoid focal loss, respectively. Following SplaTAM, we do not optimize over all previous frames, but instead select frames that are most likely to impact the newly added Gaussians. We designate every u-th frame as a keyframe and choose T frames for optimization, including the keyframes with the greatest overlap with the current frame. Overlap is determined by analyzing the point cloud of the current frameâs depth map and counting the number of points within the frustum of each keyframe.

SLAM System. Our SLAM system is built upon the Gaussian representation and differentiable rendering framework described above. Assuming an existing map represented by a set of 3D Gaussians created from a series of camera frames up to time t, when a new RGB-D frame at time t+1 is introduced, the SLAM system proceeds through three main steps: Camera Tracking, Gaussian Densification, and Spatial-Temporal Lifting.

First, in the Camera Tracking step, we minimize the image and depth reconstruction errors of the RGB-D frame to optimize the camera pose parameters at time t + 1, focusing on errors only within the visible silhouette.

Next, the Gaussian Densification step incorporates additional Gaussians into the map based on the rendered silhouette and input depth to enhance the scene representation.

Table 1. Quantitative comparison of PanoSLAM with other semantic SLAM methods for semantic segmentation performance on Replica (mIoU(%). We report the PQ, RQ, and SQ for each scene.
<table><tr><td>W GT</td><td>Methods</td><td>Metrics</td><td>room0</td><td>room1</td><td>room2</td><td>office0</td><td>office1</td><td>office2</td><td>office3</td><td>office4</td></tr><tr><td rowspan="3">Yes</td><td>NIDS-SLAM</td><td>mIoU</td><td>82.45</td><td>84.08</td><td>76.99</td><td>85.94</td><td>â</td><td></td><td>â</td><td></td></tr><tr><td>DNS SLAM</td><td>mIoU</td><td>88.32</td><td>84.90</td><td>81.20</td><td>84.66</td><td>â</td><td>â</td><td>â</td><td>â</td></tr><tr><td>SNI-SLAM</td><td>mIoU</td><td>88.42</td><td>87.43</td><td>86.16</td><td>87.63</td><td>78.63</td><td>86.49</td><td>74.01</td><td>80.22</td></tr><tr><td rowspan="5">No</td><td rowspan="5">Baseline</td><td>PQ</td><td>15.2</td><td>13.3</td><td>12.2</td><td>5.7</td><td>16.2</td><td>11.6</td><td>14.0</td><td>16.2</td></tr><tr><td>SQ</td><td>27.0</td><td>25.3</td><td>23.9</td><td>16.6</td><td>30.8</td><td>25.2</td><td>24.4</td><td>30.8</td></tr><tr><td>RQ</td><td>19.5</td><td>17.1</td><td>14.7</td><td>7.1</td><td>18.4</td><td>15.1</td><td>17.0</td><td>18.4</td></tr><tr><td>mIoU</td><td>49.07</td><td>49.80</td><td>39.96</td><td>40.06</td><td>67.45</td><td>46.95</td><td>29.04</td><td>67.45</td></tr><tr><td>PQ</td><td>19.9</td><td>14.1</td><td>18.3</td><td>10.6</td><td>16.7</td><td>12.8</td><td>14.1</td><td>16.3</td></tr><tr><td rowspan="4"></td><td></td><td>46.0</td><td>26.3</td><td>45.9</td><td>56.3</td><td>33.2</td><td>26.3</td><td>38.1</td><td></td><td>41.0</td></tr><tr><td>PanoSLAM</td><td>SQ RQ</td><td>26.6</td><td>18.5</td><td>21.9</td><td>14.2</td><td>17.6</td><td>16.2</td><td>18.3</td><td>13.0</td></tr><tr><td>mIoU</td><td></td><td></td><td>44.14</td><td></td><td></td><td>65.2</td><td>47.4</td><td>37.40</td><td></td></tr><tr><td></td><td>50.32</td><td>50.24</td><td></td><td>42.34</td><td></td><td></td><td></td><td></td><td>68.3</td></tr></table>

Table 2. Quantitative comparison of tracking accuracy for our PanoSLAM with other SLAM methods on Replica dataset. We utilize the RMSE (cm) metric as the evaluation metric.
<table><tr><td>Type</td><td>Methods</td><td>room0</td><td>room1</td><td>room2</td><td>office0</td><td>office1</td><td>office2</td><td>office3</td><td>office4</td><td>Avg.</td></tr><tr><td rowspan="5">Visual</td><td>NICE-SLAM</td><td>1.86</td><td>2.37</td><td>2.26</td><td>1.50</td><td>1.01</td><td>1.85</td><td>5.67</td><td>3.53</td><td>2.51</td></tr><tr><td>Co-SLAM</td><td>0.72</td><td>0.85</td><td>1.02</td><td>0.69</td><td>0.56</td><td>2.12</td><td>1.62</td><td>0.87</td><td>1.06</td></tr><tr><td>ESLAM</td><td>0.76</td><td>0.71</td><td>0.56</td><td>0.53</td><td>0.49</td><td>0.58</td><td>0.74</td><td>0.64</td><td>0.62</td></tr><tr><td>Point-SLAM</td><td>0.61</td><td>0.41</td><td>0.37</td><td>0.38</td><td>0.48</td><td>0.54</td><td>0.69</td><td>0.72</td><td>0.52</td></tr><tr><td>SplaTAM</td><td>0.31</td><td>0.40</td><td>0.29</td><td>0.47</td><td>0.27</td><td>0.29</td><td>0.32</td><td>0.55</td><td>0.36</td></tr><tr><td rowspan="3">Semantic</td><td>SNI-SLAM</td><td>0.50</td><td>0.55</td><td>0.45</td><td>0.35</td><td>0.41</td><td>0.33</td><td>0.62</td><td>0.50</td><td>0.46</td></tr><tr><td>DNS SLAM</td><td>0.49</td><td>0.46</td><td>0.38</td><td>0.34</td><td>0.35</td><td>0.39</td><td>0.62</td><td>0.60</td><td>0.45</td></tr><tr><td>Ours</td><td>0.34</td><td>0.44</td><td>0.24</td><td>0.48</td><td>0.29</td><td>0.33</td><td>0.52</td><td>0.52</td><td>0.39</td></tr></table>

Table 3. Evaluation on ScanNet++ (Scene 8b5caf3398 and b20a261fdf) dataset. The experiment result indicates our methodâs efficiency in real-world scenes.
<table><tr><td>Scenes</td><td>Methods</td><td>PQ</td><td>sQ</td><td>RQ</td><td>mIoU</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td rowspan="2">8b5caf3398</td><td>base</td><td>16.1</td><td>26.4</td><td>18.3</td><td>51.16</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ours</td><td>20.3</td><td>39.5</td><td>33.1</td><td>53.26</td><td>27.74</td><td>0.92</td><td>0.13</td></tr><tr><td rowspan="2">b20a261fdf</td><td>base</td><td>11.9</td><td>18.7</td><td>20.4</td><td>46.51</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ours</td><td>19.2</td><td>42.8</td><td>28.3</td><td>48.20</td><td>28.05</td><td>0.91</td><td>0.15</td></tr></table>

Finally, in the Spatial-Temporal Lifting step, we refine the parameters of all Gaussians in the scene by minimizing RGB, depth and panoptic prediction errors across all images up to frame t + 1, given the camera poses from frame 1 to t + 1. This is achieved by optimizing a selected subset of keyframes that overlap with the most recent frame, keeping the batch size manageable for efficient computation.

## 4. Experiments

Datasets. We evaluate the performance of PanoSLAM on the Replica [52] and ScanNet++ [64] datasets, both of which provide ground truth annotations for comprehensive benchmarking. In our experiments, we use eight scenes from the Replica dataset and two scenes from the Scan-Net++ dataset to assess PanoSLAMâs mapping, tracking, and segmentation capabilities in diverse environments.

Metrics. To measure the rendering quality and mapping performance, we follow the evaluation metrics defined in [48]. For mapping accuracy, we use Depth L1 (in cm), and for tracking accuracy, we employ ATE RMSE (in cm). Additionally, RGB image rendering quality is evaluated using PSNR (dB), SSIM, and LPIPS. For semantic segmentation performance, we utilize the widely-used mIoU (mean Intersection-over-Union), which provides a per-pixel accuracy measure consistent with per-pixel classification. For panoptic segmentation, we adopt the standard PQ (Panoptic Quality), SQ (Segmentation Quality), and RQ (Recognition Quality) metrics [23], reporting single-run results due to the high computational cost associated with training.

Baselines. We compare PanoSLAM with current stateof-the-art dense visual SLAM methods, including NeRFbased SLAM approaches [17, 48, 54, 57, 72] and 3D Gaussian SLAM, specifically SplaTAM [18]. For evaluating dense semantic SLAM performance, we benchmark against NeRF-based semantic SLAM methods, including SNI SLAM [68], DNS SLAM [29], and NIDS-SLAM [14], which serve as the baselines for comparison in semantic and panoptic segmentation tasks.

<!-- image-->  
Figure 3. Qualitative results of PanoSLAM in rendering appearance, panoptic information and depth. We present four scenes in the replica dataset, named room0, room1, office0 and office2. Note that different colour in panoptic infomation indicates different instances.

Table 4. Running time comparison with SplaTam. STL indicates the Spatial-Temporally Lifting. /I and /F indicate running time (ms) per iteration and frame, respectively.
<table><tr><td>Methods</td><td>Tracking/I</td><td>Mapping/I</td><td>STL/I</td><td>Tracking/F</td><td>Mapping/F</td><td>STL/F</td></tr><tr><td>SplaTam</td><td>21</td><td>22</td><td>-</td><td>890</td><td>1210</td><td>-</td></tr><tr><td>Ours</td><td>23</td><td>24</td><td>64</td><td>930</td><td>1345</td><td>642</td></tr></table>

Implementation Details. We employ the SEEM [73] 2D vision model to make predictions about the panoptic information. Our framework is built using PyTorch and trained on a RTX 4090 GPU. Throughout the training process, the 2D vision models remain frozen. We conduct warmup training 100 times for the initial five frames in order to initialize the Gaussian semantic rendering. When working with the Replica dataset, the image number set to 4. The loss weights $\lambda _ { 1 , \ldots , 5 }$ are defined as 1, 1, 1, 1, and 20, respectively. To train the region mask, we utilize the Hungarian algorithm to match the rendering and pseudo masks provided by the vision models.

## 4.1. Results and Discussion

We performed a comprehensive quantitative comparison of our method against state-of-the-art approaches across multiple dimensions, including segmentation (Tab. 1), tracking (Tab. 2), rendering (Tab. 5), and reconstruction (Tab. 6) results. Besides, we present qualitative results in Fig. 3.

Panoptic and Semantic Segmentation. The panoptic and semantic segmentation results are shown in Tables 1 and 3. As the first label-free semantic SLAM, we compare our method with the baseline SEEMâs predictions. Our method demonstrates improved performance on both the Replica and ScanNet++ datasets, highlighting its effectiveness in label-free panoptic and semantic segmentation.

Tracking. As shown in Tab. 2, our approach achieves superior tracking accuracy compared to other semantic SLAM methods. This improvement is largely due to the Spatial-Temporal Lifting technique, which ensures consistent semantic information over time and across viewpoints, effectively reducing accumulated drift in the tracking process.

Rendering. Table 5 presents the rendering quality on input views from the Replica dataset. Our method achieves the highest performance across PSNR, SSIM, and LPIPS metrics compared to other dense semantic SLAM methods, indicating superior visual fidelity.

Reconstruction. As shown in Tab. 6, our method outperforms other semantic SLAM methods in reconstruction accuracy, demonstrating its superior effectiveness and efficiency in the 3D mapping process.

Qualitative Results. Fig. 3 shows qualitative evaluations of PanoSLAM, highlighting its performance in rendering appearance, semantics, and depth. Notably, PanoSLAM achieves impressive panoptic and semantic segmentation results without any manual labels, underscoring its robustness in label-free environments.

Running Time Analysis. Table 4 provides a running time analysis, showing that our method incurs slightly higher per-iteration times for tracking (19.9 ms vs. 15.2 ms) and mapping (46.0 ms vs. 27.0 ms) compared to SplaTAM, due to the added panoptic processing. Nevertheless, the perframe times remain efficient, with minimal overhead from the STL module (0.979 ms), confirming the overall efficiency of our approach.

Table 5. Quantitative comparison of training view rendering performance on Replica dataset. Our work outperforms other semantic SLAM methods on all three metrics across all scenes.
<table><tr><td>Type</td><td>Methods</td><td>Metrics</td><td>room0</td><td>room1</td><td>room2</td><td>office0</td><td>office1</td><td>office2</td><td>office3</td><td>office4</td><td>Avg.</td></tr><tr><td rowspan="8">Visual</td><td rowspan="2">NICE-SLAM</td><td>PSNRâ</td><td>22.12</td><td>22.47</td><td>24.52</td><td>29.07</td><td>30.34</td><td>19.66</td><td>22.23</td><td>24.94</td><td>24.42</td></tr><tr><td>SSIMâ</td><td>0.689</td><td>0.757</td><td>0.814</td><td>0.874</td><td>0.886</td><td>0.797</td><td>0.801</td><td>0.856</td><td>0.809</td></tr><tr><td rowspan="2"></td><td>LPIPSâ</td><td>0.330</td><td>0.271</td><td>0.208</td><td>0.229</td><td>0.181</td><td>0.235</td><td>0.209</td><td>0.198</td><td>0.233</td></tr><tr><td>PSNRâ</td><td>27.27</td><td>28.45</td><td>29.06</td><td>34.14</td><td>34.87</td><td>28.43</td><td>28.76</td><td>30.91</td><td>30.24</td></tr><tr><td rowspan="2">Co-SLAM</td><td>SSIMâ</td><td>0.910</td><td>0.909</td><td>0.932</td><td>0.961</td><td>0.969</td><td>0.938</td><td>0.941</td><td>0.955</td><td>0.939</td></tr><tr><td>LPIPSâ</td><td>0.324</td><td>0.294</td><td>0.266</td><td>0.209</td><td>0.196</td><td>0.258</td><td>0.229</td><td>0.236</td><td>0.252</td></tr><tr><td rowspan="2">ESLAM</td><td>PSNRâ</td><td>25.32</td><td>27.77</td><td>29.08</td><td>33.71</td><td>30.20</td><td>28.09</td><td>28.77</td><td>29.71</td><td>29.08</td></tr><tr><td>SSIMâ</td><td>0.875</td><td>0.902</td><td>0.932</td><td>0.960</td><td>0.923</td><td>0.943</td><td>0.948</td><td>0.945</td><td>0.929</td></tr><tr><td rowspan="2">SplaTAM</td><td>LPIPSâ</td><td>0.313</td><td>0.298</td><td>0.248</td><td>0.184</td><td>0.228</td><td>0.241</td><td>0.196</td><td>0.204</td><td></td><td>0.239</td></tr><tr><td>PSNRâ</td><td>32.86</td><td>33.89</td><td>35.25</td><td>38.26</td><td>39.17</td><td>31.97</td><td>29.70</td><td>31.81</td><td></td><td>34.11</td></tr><tr><td rowspan="2"></td><td>SSIMâ</td><td>0.978</td><td>0.969</td><td>0.979</td><td>0.977</td><td>0.978</td><td></td><td>0.968</td><td>0.949</td><td>0.949</td><td>0.968</td></tr><tr><td>LPIPSâ</td><td>0.072</td><td>0.103</td><td>0.081</td><td>0.092</td><td>0.093</td><td>0.102</td><td></td><td>0.121</td><td>0.152</td><td>0.102</td></tr><tr><td rowspan="4">Semantic</td><td rowspan="2">SNI-SLAM</td><td>PSNRâ</td><td>25.91</td><td>28.17</td><td>29.15</td><td>33.86</td><td>30.34</td><td>29.10</td><td>29.02</td><td>29.87</td><td>29.43</td></tr><tr><td>SSIMâ LPIPSâ</td><td>0.885</td><td>0.910</td><td>0.938</td><td>0.965</td><td>0.927</td><td>0.950</td><td>0.950</td><td>0.952</td><td>0.935</td></tr><tr><td rowspan="2"></td><td></td><td>0.307</td><td>0.292</td><td>0.245</td><td>0.182</td><td>0.225</td><td>0.238</td><td>0.192</td><td>0.198</td><td>0.235</td></tr><tr><td>PSNRâ</td><td>32.89</td><td>33.05</td><td>34.24</td><td>37.90</td><td>38.29</td><td>29.61</td><td>29.82</td><td>31.02</td><td>33.35</td></tr><tr><td rowspan="2"></td><td>Ours SSIMâ</td><td>0.979</td><td>0.960</td><td></td><td>0.982</td><td>0.980</td><td>0.980</td><td>0.952</td><td>0.943</td><td>0.939</td><td>0.964</td></tr><tr><td>LPIPSâ</td><td>0.071</td><td>0.115</td><td>0.082</td><td>0.092</td><td>0.095</td><td></td><td>0.121</td><td>0.127</td><td>0.166</td><td>0.108</td></tr></table>

Table 6. Comparion of our method with existing radiance field-based SLAM methods on Replica for reconstruction metric Depth L1 (cm).
<table><tr><td>Type</td><td>Methods</td><td>room0</td><td>room1</td><td>room2</td><td>office0</td><td>office1</td><td>office2</td><td>office3</td><td>office4</td><td>Avg.</td></tr><tr><td rowspan="4">Visual</td><td>NICE-SLAM</td><td>1.81</td><td>1.44</td><td>2.04</td><td>1.39</td><td>1.76</td><td>8.33</td><td>4.99</td><td>2.01</td><td>2.97</td></tr><tr><td>Vox-Fusion</td><td>1.09</td><td>1.90</td><td>2.21</td><td>2.32</td><td>3.40</td><td>4.19</td><td>2.96</td><td>1.61</td><td>2.46</td></tr><tr><td>Co-SLAM</td><td>1.05</td><td>0.85</td><td>2.37</td><td>1.24</td><td>1.48</td><td>1.86</td><td>1.66</td><td>1.54</td><td>1.51</td></tr><tr><td>ESLAM</td><td>0.73</td><td>0.74</td><td>1.26</td><td>0.71</td><td>1.02</td><td>0.93</td><td>1.03</td><td>1.18</td><td>0.95</td></tr><tr><td rowspan="2">Semantic</td><td>SNI-SLAM</td><td>0.55</td><td>0.58</td><td>0.87</td><td>0.55</td><td>0.97</td><td>0.89</td><td>0.75</td><td>0.97</td><td>0.77</td></tr><tr><td>Ours</td><td>0.52</td><td>0.51</td><td>0.53</td><td>0.38</td><td>0.29</td><td>0.85</td><td>0.70</td><td>1.15</td><td>0.61</td></tr></table>

Table 7. Ablation experiments on room0. We report the panoptic segmentation performance.
<table><tr><td>Setting</td><td>PQ</td><td>SQ</td><td>RQ</td><td>mIoU</td><td>PSNR â</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>base</td><td>15.2</td><td>27.0</td><td>19.5</td><td>49.07</td><td>-</td><td>-</td><td>-</td></tr><tr><td>w/o STL</td><td>7.3</td><td>18.2</td><td>10.3</td><td>40.05</td><td>29.29</td><td>0.918</td><td>0.113</td></tr><tr><td>STL-2</td><td>11.6</td><td>36.1</td><td>17.1</td><td>46.87</td><td>30.31</td><td>0.945</td><td>0.101</td></tr><tr><td>Ours</td><td>19.9</td><td>46.0</td><td>26.6</td><td>50.32</td><td>32.89</td><td>0.979</td><td>0.071</td></tr></table>

## 4.2. Ablation Study

We performed a series of ablation experiments on the room0 scene to evaluate the effectiveness of various components in our framework. As shown in Tab. 7, âBaseâ represents the predictions from the 2D vision model alone, while âw/o STLâ indicates that the Spatial-Temporal Lifting (STL) module was not used during training. âSTL-(n)â represents the inclusion of STL with n time-steps of images, with n set to 4 in our full method. The results of these ablation studies highlight the critical role of STL in enhancing the ability of 3D Gaussians to effectively learn from the noisy pseudo-labels generated by the 2D vision model. This emphasizes the significant impact of STL within our framework.

## 5. Conclusions

We introduce PanoSLAM, the first Gaussian-based SLAM method capable of reconstructing panoptic 3D scene from unlabeled RGB-D videos. To effectively distill knowledge from 2D vision foundation models into a 3D Gaussian splatting SLAM framework, we propose a novel Spatial-Temporal Lifting module. Experimental results demonstrate that our method significantly outperforms state-ofthe-art approaches. Furthermore, for the first time, we successfully recover panoptic information in 3D open-world scenes without any manual labels.

Limitations and Future Work. Currently, our approach relies on 2D vision foundation models to generate pseudolabels for guiding semantic reconstruction. However, these labels can be noisy, particularly in areas with fine and intricate details, such as flower leaves in large, complex rooms. Despite the improvements provided by our Spatial-Temporal Lifting module, achieving precise semantic reconstruction in these regions remains challenging. In future work, we aim to explore ways to integrate multi-view information into 2D vision foundation models to produce more accurate and detailed semantic labels, ultimately enhancing the quality of semantic scene reconstruction.

## References

[1] Maxime Bucher, Tuan-Hung Vu, Matthieu Cord, and Patrick Perez. Zero-shot semantic segmentation. Â´ Advances in Neural Information Processing Systems, 32, 2019. 2

[2] Runnan Chen, Xinge Zhu, Nenglun Chen, Wei Li, Yuexin Ma, Ruigang Yang, and Wenping Wang. Zero-shot point cloud segmentation by transferring geometric primitives. arXiv preprint arXiv:2210.09923, 2022. 2

[3] Runnan Chen, Youquan Liu, Lingdong Kong, Xinge Zhu, Yuexin Ma, Yikang Li, Yuenan Hou, Yu Qiao, and Wenping Wang. Clip2scene: Towards label-efficient 3d scene understanding by clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7020â7030, 2023. 1, 2

[4] Runnan Chen, Xinge Zhu, Nenglun Chen, Wei Li, Yuexin Ma, Ruigang Yang, and Wenping Wang. Bridging language and geometric primitives for zero-shot point cloud segmentation. In Proceedings of the 31st ACM International Conference on Multimedia, pages 5380â5388, 2023. 2

[5] Runnan Chen, Xinge Zhu, Nenglun Chen, Dawei Wang, Wei Li, Yuexin Ma, Ruigang Yang, Tongliang Liu, and Wenping Wang. Model2scene: Learning 3d scene representation via contrastive language-cad models pre-training. arXiv preprint arXiv:2309.16956, 2023. 2

[6] Runnan Chen, Youquan Liu, Lingdong Kong, Nenglun Chen, Xinge Zhu, Yuexin Ma, Tongliang Liu, and Wenping Wang. Towards label-free scene understanding by vision foundation models. Advances in Neural Information Processing Systems, 36, 2024. 1, 2

[7] Bowen Cheng, Maxwell D Collins, Yukun Zhu, Ting Liu, Thomas S Huang, Hartwig Adam, and Liang-Chieh Chen. Panoptic-deeplab: A simple, strong, and fast baseline for bottom-up panoptic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12475â12485, 2020. 2

[8] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation. Advances in neural information processing systems, 34:17864â17875, 2021. 4, 5

[9] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1290â1299, 2022.

[10] Ran Cheng, Ryan Razani, Ehsan Taghavi, Enxu Li, and Bingbing Liu. (af)2-s3net: Attentive feature fusion with adaptive feature selection for sparse semantic segmentation network. In IEEE Conference on Computer Vision and Pattern Recognition, pages 12547â12556, 2021.

[11] MMDetection3D Contributors. Mmdetection3d: Openmmlab next-generation platform for general 3d object detection, 2020. 2

[12] Andrew J Davison, Ian D Reid, Nicholas D Molton, and Olivier Stasse. Monoslam: Real-time single camera slam. IEEE transactions on pattern analysis and machine intelligence, 29(6):1052â1067, 2007. 2

[13] Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, and Xiaojuan Qi. Language-driven openvocabulary 3d scene understanding. arXiv preprint arXiv:2211.16312, 2022. 2

[14] Yasaman Haghighi, Suryansh Kumar, Jean-Philippe Thiran, and Luc Van Gool. Neural implicit dense semantic slam. arXiv preprint arXiv:2304.14560, 2023. 1, 2, 6

[15] Fangzhou Hong, Lingdong Kong, Hui Zhou, Xinge Zhu, Hongsheng Li, and Ziwei Liu. Unified 3d and 4d panoptic segmentation via dynamic shifting network. arXiv preprint arXiv:2203.07186, 2022. 2

[16] Ping Hu, Stan Sclaroff, and Kate Saenko. Uncertainty-aware learning for zero-shot semantic segmentation. Advances in Neural Information Processing Systems, 33:21713â21724, 2020. 2

[17] Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret. Eslam: Efficient dense slam system based on hybrid representation of signed distance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17408â17419, 2023. 2, 6

[18] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. arXiv preprint arXiv:2312.02126, 2023. 2, 3, 6

[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4):1â14, 2023. 2

[20] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19729â19739, 2023. 1

[21] Leonid Keselman and Martial Hebert. Approximate differentiable rendering with algebraic surfaces. In European Conference on Computer Vision, pages 596â614. Springer, 2022. 2

[22] Leonid Keselman and Martial Hebert. Flexible techniques for differentiable rendering with 3d gaussians. arXiv preprint arXiv:2308.14737, 2023. 2

[23] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollar. Panoptic segmentation. In Â´ Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9404â9413, 2019. 6

[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015â4026, 2023. 1, 2

[25] Georg Klein and David Murray. Parallel tracking and mapping for small ar workspaces. In 2007 6th IEEE and ACM international symposium on mixed and augmented reality, pages 225â234. IEEE, 2007. 2

[26] Lingdong Kong, Youquan Liu, Runnan Chen, Yuexin Ma, Xinge Zhu, Yikang Li, Yuenan Hou, Yu Qiao, and Ziwei Liu. Rethinking range view representation for lidar segmentation.

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 228â240, 2023. 2

[27] Lingdong Kong, Youquan Liu, Xin Li, Runnan Chen, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Robo3d: Towards robust and reliable 3d perception against corruptions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19994â20006, 2023. 2

[28] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic segmentation. In International Conference on Learning Representations, 2022. 2

[29] Kunyi Li, Michael Niemeyer, Nassir Navab, and Federico Tombari. Dns slam: Dense neural semantic-informed slam. arXiv preprint arXiv:2312.00204, 2023. 1, 2, 6

[30] Mingrui Li, Shuhong Liu, and Heng Zhou. Sgs-slam: Semantic gaussian splatting for neural dense slam. arXiv preprint arXiv:2402.03246, 2024. 1, 2

[31] Peike Li, Yunchao Wei, and Yi Yang. Consistent structural relation learning for zero-shot segmentation. Advances in Neural Information Processing Systems, 33:10317â10327, 2020. 2

[32] Ziwen Li, Jiaxin Huang, Runnan Chen, Yunlong Che, Yandong Guo, Tongliang Liu, Fakhri Karray, and Mingming Gong. Urban4d: Semantic-guided 4d gaussian splatting for urban scene reconstruction. arXiv preprint arXiv:2412.03473, 2024. 2

[33] Youquan Liu, Runnan Chen, Xin Li, Lingdong Kong, Yuchen Yang, Zhaoyang Xia, Yeqi Bai, Xinge Zhu, Yuexin Ma, Yikang Li, et al. Uniseg: A unified multi-modal lidar segmentation network and the openpcseg codebase. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21662â21673, 2023. 2

[34] Youquan Liu, Lingdong Kong, Xiaoyang Wu, Runnan Chen, Xin Li, Liang Pan, Ziwei Liu, and Yuexin Ma. Multi-space alignments towards universal lidar segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14648â14661, 2024. 2

[35] Yuhang Lu, Qi Jiang, Runnan Chen, Yuenan Hou, Xinge Zhu, and Yuexin Ma. See more and know more: Zero-shot point cloud segmentation via multi-modal visual data. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21674â21684, 2023. 2

[36] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. arXiv preprint arXiv:2312.06741, 2023. 2

[37] Bjorn Michele, Alexandre Boulch, Gilles Puy, Maxime Â¨ Bucher, and Renaud Marlet. Generative zero-shot learning for semantic segmentation of 3d point clouds. In International Conference on 3D Vision, pages 992â1002, 2021. 2

[38] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: a versatile and accurate monocular slam system. IEEE transactions on robotics, 31(5):1147â1163, 2015. 2

[39] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. arXiv preprint arXiv:2211.15654, 2022. 1

[40] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 815â824, 2023. 2

[41] Xidong Peng, Runnan Chen, Feng Qiao, Lingdong Kong, Youquan Liu, Yujing Sun, Tai Wang, Xinge Zhu, and Yuexin Ma. Learning to adapt sam for segmenting cross-domain point clouds. In European Conference on Computer Vision, pages 54â71. Springer, 2025. 2

[42] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 652â660, 2017. 2

[43] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051â20060, 2024. 1

[44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748â8763. PMLR, 2021. 1, 2

[45] Luigi Riz, Cristiano Saltori, Elisa Ricci, and Fabio Poiesi. Novel class discovery for 3d point cloud semantic segmentation. arXiv preprint arXiv:2303.11610, 2023. 2

[46] Antoni Rosinol, Marcus Abate, Yun Chang, and Luca Carlone. Kimera: an open-source library for real-time metricsemantic localization and mapping. In IEEE International Conference on Robotics and Automation, pages 1689â1696. IEEE, 2020. 2

[47] Renato F Salas-Moreno, Richard A Newcombe, Hauke Strasdat, Paul HJ Kelly, and Andrew J Davison. Slam++: Simultaneous localisation and mapping at the level of objects. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1352â1359, 2013. 2

[48] Erik Sandstrom, Yue Li, Luc Van Gool, and Martin R Os-Â¨ wald. Point-slam: Dense neural point cloud-based slam. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18433â18444, 2023. 2, 6

[49] Corentin Sautier, Gilles Puy, Spyros Gidaris, Alexandre Boulch, Andrei Bursuc, and Renaud Marlet. Image-to-lidar self-supervised distillation for autonomous driving data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9891â9901, 2022. 2

[50] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for open-vocabulary scene understanding. arXiv preprint arXiv:2311.18482, 2023. 1

[51] Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Nor- Â´ man Muller, Matthias NieÃner, Angela Dai, and PeterÂ¨ Kontschieder. Panoptic lifting for 3d scene understanding with neural fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9043â9052, 2023. 1

[52] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019. 2, 6

[53] Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. Segmenter: Transformer for semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7262â7272, 2021. 2

[54] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davison. imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6229â6238, 2021. 2, 6

[55] Jiahao Sun, Chunmei Qing, Xiang Xu, Lingdong Kong, Youquan Liu, Li Li, Chenming Zhu, Jingwei Zhang, Zeqi Xiao, Runnan Chen, et al. An empirical study of training state-of-the-art lidar segmentation models. arXiv preprint arXiv:2405.14870, 2024. 2

[56] Angtian Wang, Peng Wang, Jian Sun, Adam Kortylewski, and Alan Yuille. Voge: a differentiable volume renderer using gaussian ellipsoids for analysis-by-synthesis. arXiv preprint arXiv:2205.15401, 2022. 2

[57] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Coslam: Joint coordinate and sparse parametric encodings for neural real-time slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13293â13302, 2023. 2, 6

[58] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point transformer v2: Grouped vector attention and partition-based pooling. arXiv preprint arXiv:2210.05666, 2022. 2

[59] Jianyun Xu, Ruixiang Zhang, Jian Dou, Yushi Zhu, Jie Sun, and Shiliang Pu. Rpvnet: A deep and efficient range-pointvoxel fusion network for lidar point cloud segmentation. In IEEE/CVF International Conference on Computer Vision, pages 16024â16033, 2021. 2

[60] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for zeroshot semantic segmentation with pre-trained vision-language model. arXiv preprint arXiv:2112.14757, 2021. 2

[61] Yiteng Xu, Peishan Cong, Yichen Yao, Runnan Chen, Yuenan Hou, Xinge Zhu, Xuming He, Jingyi Yu, and Yuexin Ma. Human-centric scene understanding for 3d large-scale scenarios. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20349â20359, 2023. 2

[62] Chi Yan, Delin Qu, Dong Wang, Dan Xu, Zhigang Wang, Bin Zhao, and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting. arXiv preprint arXiv:2311.11700, 2023. 2

[63] Xu Yan, Jiantao Gao, Chaoda Zheng, Chaoda Zheng, Ruimao Zhang, Shenghui Cui, and Zhen Li. 2dpass: 2d priors assisted semantic segmentation on lidar point clouds. In ECCV, 2022. 2

[64] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias NieÃner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12â22, 2023. 2, 6

[65] Junbo Yin, Jianbing Shen, Runnan Chen, Wei Li, Ruigang Yang, Pascal Frossard, and Wenguan Wang. Is-fusion: Instance-scene collaborative fusion for multimodal 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14905â 14915, 2024. 2

[66] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Oswald. Gaussian-slam: Photo-realistic dense slam with gaussian splatting. arXiv preprint arXiv:2312.10070, 2023. 2

[67] Hui Zhang and Henghui Ding. Prototypical matching and open set rejection for zero-shot semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6974â6983, 2021. 2

[68] Siting Zhu, Guangming Wang, Hermann Blum, Jiuming Liu, Liang Song, Marc Pollefeys, and Hesheng Wang. Sni-slam: Semantic neural implicit slam. arXiv preprint arXiv:2311.11016, 2023. 1, 2, 6

[69] Siting Zhu, Renjie Qin, Guangming Wang, Jiuming Liu, and Hesheng Wang. Semgauss-slam: Dense semantic gaussian splatting slam. arXiv preprint arXiv:2403.07494, 2024. 1, 2

[70] Xinge Zhu, Hui Zhou, Tai Wang, Fangzhou Hong, Yuexin Ma, Wei Li, Hongsheng Li, and Dahua Lin. Cylindrical and asymmetrical 3d convolution networks for lidar segmentation. arXiv preprint arXiv:2011.10033, 2020. 2

[71] Xinge Zhu, Hui Zhou, Tai Wang, Fangzhou Hong, Yuexin Ma, Wei Li, Hongsheng Li, and Dahua Lin. Cylindrical and asymmetrical 3d convolution networks for lidar segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9939â9948, 2021. 2

[72] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12786â12796, 2022. 2, 6

[73] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. Advances in Neural Information Processing Systems, 36, 2024. 7