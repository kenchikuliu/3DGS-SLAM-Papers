# UDGS-SLAM: UniDepth Assisted Gaussian Splatting for Monocular SLAM

Mostafa Mansour a ,â, Ahmed Abdelsalam b ,â, Ari Happonen b, Jari Porras b,c, Esa Rahtu d

a Faculty of Engineering and Natural Sciences, Tampere University, Finland

b School of Engineering Science, LUT University, Finland

c School of Electrical Engineering, Aalto University, Finland

d Faculty of Information Technology and Communication Sciences, Tampere University, Finland

## A R T I C L E I N F O

Keywords:   
UniDepth   
Gaussian splitting   
Monocular SLAM   
Dense SLAM   
Mapping   
Scene representation

## A B S T R A C T

Recent advancements in monocular neural depth estimation, particularly those achieved by the UniDepth network, have prompted the investigation of integrating UniDepth within a Gaussian splatting framework for monocular SLAM. This study presents UDGS-SLAM, a novel approach that eliminates the necessity of RGB-D sensors for depth estimation within Gaussian splatting framework. UDGS-SLAM employs statistical filtering to ensure local consistency of the estimated depth and jointly optimizes camera trajectory and Gaussian scene representation parameters. The proposed method achieves high-fidelity rendered images and low ATE-RMSE of the camera trajectory. The performance of UDGS-SLAM is rigorously evaluated using the TUM RGB-D dataset and benchmarked against several baseline methods, demonstrating superior performance across various scenarios. Additionally, an ablation study is conducted to validate design choices and investigate the impact of different network backbone encoders on system performance.

## 1. Introduction

Visual Simultaneous Localization and Mapping (VSLAM) is essential for estimating the pose of a moving vision sensor while simultaneously constructing a map of the environment. VSLAM plays a crucial role in various applications, such as robotics, virtual reality, and augmented reality [1â3]. The choice of map (or scene) representation significantly affects the performance of the SLAM system, influencing both its internal subsystems and external systems that rely on its outputs.

The representation of the map has been a focal point of extensive research [4]. Approaches have focused on explicit handcrafted sparse [5â 8] and dense representations [9â12], utilizing points, voxels, surfels, and signed distance fields for map construction. However, despite their maturity, these representations have notable limitations. They heavily depend on the availability of 3D geometric features and are limited to representing only observed parts of the environment. Furthermore, they lack the ability to generate or synthesize photorealistic, high-fidelity novel scenes from different camera viewpoints, which is critical in virtual and augmented reality applications.

To address these limitations, recent research has focused on implicit volumetric photorealistic representations, such as Neural Radiance Fields (NeRF) [13] and Gaussian Splatting (GS) [14], to enable unified, high-fidelity scene representation [15]. NeRF-based methods enhance scene representation by minimizing photometric losses through differentiable rendering, encoding the scene into the weight space of a neural network using multi-layer perceptrons and ray marching [13,15â18]. However, NeRF methods face several challenges, including high computational requirements, long training times, overfitting to specific objects or scenes, and susceptibility to catastrophic forgetting [15]. In contrast, GS uses tile-based rasterization for efficient rendering and represents the scene using 3D Gaussians [14], optimized with each new input. This approach is computationally more efficient, adapts well to large scenes, and avoids catastrophic forgetting. Additionally, GS-based SLAM integrates both photometric information and depth maps, making it better suited for explicit spatial geometry modeling. The differentiable rendering of Gaussians, combined with rapid GPU-based implementation, allows for fast and joint optimization of scene parameters and camera trajectory.

Due to these advantages, GS has emerged as a leading method for photorealistic 3D scene reconstruction, unifying representations for tracking, mapping, and rendering. Recent research predominantly focuses on utilizing RGB-D inputs to benefit from depth sensors [19â25]. However, there remains a gap in exploring monocular methods due to the lack of depth information [19].

Inspired by recent advancements in monocular depth estimation through neural networks [26], we investigate the use of neural depth estimation within the GS framework for monocular SLAM. This approach eliminates the need for RGB-D sensors while retaining the benefits of GS for scene representation. Specifically, we explore the use of the UniDepth network [27] for depth estimation in the GS-based monocular SLAM framework. Our proposed method emphasizes the joint optimization of camera trajectory and 3D Gaussian map representation, utilizing the depth estimation from the UniDepth network. Additionally, we introduce a statistical filtering technique to enhance the local consistency of the estimated depth, improving the quality of photorealistic reconstruction.

In summary, our contributions are as follows:

â¢ Integrating UniDepth network for depth estimation within Gaussian Splatting: We leverage UniDepth for depth estimation from RGB images within the Gaussian Splatting framework, facilitating the joint optimization of camera trajectory and 3D photorealistic reconstruction.

â¢ Introducing Statistical Filtering: We implement a straightforward yet effective statistical filtering stage that ensures local consistency of the estimated depth map, enhancing the overall performance of the framework.

â¢ Evaluation on real dataset benchmark: Our method is rigorously tested on TUM RGB-D dataset, demonstrating superior performance compared to baseline methods in various scenarios.

â¢ Ablation studies: We conduct comprehensive ablation studies using different backbone encoders of the UniDepth network, both with and without the implementation of statistical filtering, to evaluate their impact on performance.

## 2. Related work

## 2.1. Monocular neural depth estimation

Monocular depth estimation (MDE) is a classic research area in computer vision that involves determining the precise depth of each pixel in an image. This capability enables the reconstruction of 3D scenes from 2D images, which is crucial for a wide range of applications in computer vision and robotics, such as autonomous navigation, augmented reality, object detection, and 3D modeling. Early methods relied heavily on geometric principles and handcrafted features to estimate depth [28â30].

The advent of deep learning revolutionized the field of computer vision, including MDE. Neural networks offer a data-driven approach to learning complex features directly from images, allowing depth estimation from a single image. One of the earliest neural network-based methods for MDE was introduced by Eigen et al. [31]. This method leveraged the ability of CNNs to capture hierarchical features, achieving reasonable accuracy on the NYU [32] and KITTI [33] datasets. As neural networks continued to evolve, two main branches of monocular depth estimation from a single image emerged: Monocular Metric Depth Estimation (MMDE) and Monocular Relative (Scale-Agnostic) Depth Estimation (MRDE).

Focusing on MMDE [31,34â41], it aims to predict absolute values in physical units (e.g., meters), which is necessary to perform 3D reconstruction effectively. Most of the existing MMDE methods have shown great accuracy across several benchmarks, but they fail to generalize to real-world scenarios and tend to overfit specific datasets [42]. Some methods have attempted to solve this by training a single metric depth estimation model across multiple datasets, but it has been reported that this often deteriorates performance, especially when the collection includes images with large differences in depth scale, such as indoor and outdoor images [42].

Few methods [43,44] have tackled the challenging problem of generalization, but these methods still rely on controlled testing conditions, including fixed camera intrinsics. Hu et al. relaxed some of these conditions; however, their solution still requires known camera intrinsics [45]. Unlike other methods, UniDepth [46] addresses the generalization problem without the limitation of fixed camera intrinsics. UniDepth consistently sets new state-of-the-art benchmarks, even compared with non-zero-shot methods. Therefore, the UniDepth network is chosen for depth estimation as a first step in our pipeline.

## 2.2. NeRF based SLAM

Mildenhall et al. introduced NeRF as an implicit volumetric scene representation [47]. Originally, NeRF required known camera poses to construct its scene representation. To accommodate this, many studies have employed the COLMAP structure-from-motion package [48] to estimate camera poses for use in NeRF implementations. iMAP [49] was the first to relax the requirement for known camera poses by simultaneously performing tracking and mapping using NeRF representation. Despite its innovation, iMAP faced scalability issues, which were subsequently addressed by NICE-SLAM [17] through the introduction of hierarchical multi-feature grids. Vox-Fusion [18] proposed a hybrid solution that combines NeRF with traditional volumetric fusion methods to enhance scene representation. Recently, Point-SLAM [50] enhanced 3D reconstruction by employing neural point clouds and feature interpolation for volumetric rendering. Other additional improvements are discussed in Tosi et al. [15]. Despite these advancements, NeRF-based methods still fundamentally grapple with long training times due to the computational demands of ray marching rendering, and issues with catastrophic forgetting. In contrast, Gaussian-based methods avoid these pitfalls by incorporating dynamic insertion and pruning techniques to manage newly visible scenes, and by utilizing fast rasterization instead of ray marching to boost rendering efficiency.

## 2.3. 3D Gaussians based SLAM

Since its introduction as a promising 3D scene representation [14], 3D Gaussian splatting has emerged as a prominent technology for SLAM due to its fast rasterization rendering via splatting and its ability to overcome the catastrophic forgetting problem through Gaussian insertion and pruning management [15]. Given the importance of depth maps in Gaussian representation, most studies employ RGB-D cameras within the Gaussian framework, benefiting from the integration of depth sensors [15,19,20,25,51â55]. Li et al. focused on generating novel syntheses from sparse inputs by optimizing radiance fields with depth regularization, without optimizing the camera trajectory [56]. Unlike them, UDGS-SLAM optimizes both the camera trajectory and scene representation. Additionally, UDGS-SLAM proposes a statistical filter to enforce local consistency and geometric stability. Some works have integrated RGB-D cameras with IMUs within the Gaussian splatting framework [54], while others have incorporated various depth and normal prior cues with RGB-D measurements [51]. However, there is a notable lack of investigation into using monocular camera measurements within the Gaussian splatting framework, primarily, due to the absence of direct depth measurement.

Lee et al. proposed a monocular-based 3D Gaussian representation for scenes without estimating camera trajectory [57]. They utilized IMU and LiDAR data to provide odometry and estimate the trajectory. In contrast, UDGS-SLAM does not rely on any complementary sensors for trajectory estimation and uses only a monocular camera for this purpose. Matsuki et al. introduced Gaussian splatting for SLAM using a monocular camera [19]. Their approach utilized prior knowledge about scene depth, initializing the 3D Gaussians with depths normally distributed around the mean scene depth. To address the lack of direct depth sensor data, they optimized the Gaussian parameters and camera trajectory by minimizing the photometric error. In contrast, UDGS-SLAM does not rely on any prior knowledge about scene depth. It leverages statistically filtered depth maps from the UniDepth network for initialization. Furthermore, it optimizes the Gaussian parameters and camera trajectory by minimizing a weighted sum of photometric and geometric errors. This approach enables UDGS-SLAM to outperform the monocular SLAM proposed by Matsuki et al. in most scenarios of the TUM dataset, as presented in Section 5.

<!-- image-->  
Fig. 1. UDGS-SLAM utilizes 3D Gaussian splats for scene representation, enabling high-fidelity photorealistic reconstruction for dense SLAM using a monocular camera. It employs the Unidepth network to estimate scene depth from a single RGB image (a, b). The estimated depth is subsequently filtered for local consistency (c). Through differential rendering rasterization, it generates rendered RGB and depth images for a given camera pose. The system then achieves 3D scene representation by jointly optimizing 3D Gaussian splats and camera trajectory through the minimization of photometric error between the input and rendered RGB images, as well as the minimization of geometric error between the estimated and rendered depths (d). This approach enables the reconstruction of a dense scene (e) and allows for photorealistic rendering of the scene from any given camera pose (f).

## 3. Methodology

The proposed approach estimates the camera poses for each frame $\{ P _ { i } \} _ { i = 1 } ^ { N }$ and reconstruct a 3D volumetric map representation of the scene from a sequential RGB image stream $\{ I _ { i } \} _ { i = 1 } ^ { N }$ obtained from a monocular camera with known camera intrinsic $\mathbf { \bar { K } } \in \mathbb { R } ^ { 3 \times 3 }$ 3. The map is represented by a collection of 3D Gaussians, which can be rendered into a photorealstic image for a given view point of a camera pose. This representation is achieved by using differentiable rendering through 3D Gaussian splatting and gradient-based optimization, facilitating the optimization of the camera pose for each frame as well as the volumetric representation of the scene.

## 3.1. 3D Gaussian scene representation

The proposed approach optimizes the scene representation to effectively capture both geometrical and appearance features, enabling it to be rendered into high-fidelity color and depth images. We represent the scene as a set of 3D Gassians coupled with view-independent color, opacity, and a covariance matrix.

$$
\mathbf { G } = \{ G _ { i } : ( \mu _ { i } ^ { W } , \mathbf { c } _ { i } , o _ { i } , { \boldsymbol { \Sigma } } _ { i } ^ { W } ) \mid i = 1 , \ldots , N \} .\tag{1}
$$

Each 3D Gaussian $G _ { i } ,$ , in the world coordinate frame ?? is defined by its center position $\mu _ { i } ^ { \mathrm { W } } \in \mathbb { R } ^ { 3 }$ , its RGB color $\mathbf { c } _ { i } ,$ , a covariance matrix $\boldsymbol { \itSigma } _ { i } ^ { \boldsymbol { W } }$ , and its opacity $o \in [ 0 , 1 ] . \mathrm { A }$ Gaussian $G _ { i }$ affects a 3D point $X \in \mathbb { R } ^ { 3 }$ according to the Gaussian equation weighted by its opacity as follows:

$$
f ( X ) = o _ { i } ( \frac { \exp ( - \frac { 1 } { 2 } ( X - \mu _ { i } ^ { W } ) ^ { T } \varSigma _ { i } ^ { W ^ { - 1 } } ( X - \mu _ { i } ^ { W } ) ) } { ( 2 \pi ) ^ { 3 / 2 } \left| \varSigma _ { i } ^ { W } \right| ^ { 1 / 2 } } ) .\tag{2}
$$

## 3.2. Color and depth differentiable rendering via splatting

The objective of Gaussian splatting [20] is to render high-fidelity RGB and depth images from the 3D volumetric Gaussian scene representation given a camera pose. Importantly, the rendering should be differentiable allowing the gradient to be calculated for the underlying Gaussiansâ map parameters and camera poses with respect to the photometric and geometric discrepancies between the rendered and the provided RGB and depth images, respectively. The gradient is used to minimize the discrepancies by updating both the parameters of 3D Gaussian splats and camera poses. According to [14], an RGB image is rendered from a set of 3D Gaussians by, first, sorting all the Gaussians from front to back with respect to a given camera pose. Then, the 3D Gaussians within the camera frustum are splatted (projected) into 2D pixel space using the camera pose and the camera intrinsic matrix ??. Finally, an RGB image can be rendered by alpha-blending each 2D splatted Gaussian in order in pixel space. The rendered color of a $\boldsymbol { p } = ( u , v )$ pixel can be written as:

$$
C ( \boldsymbol { p } ) = \sum _ { i = 1 } ^ { n } \mathbf { c } _ { i } f ( \boldsymbol { p } ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - f ( \boldsymbol { p } ) ) ,\tag{3}
$$

where ?? is the number of pixels per image and $f ( p )$ is calculated according to (2) after projecting each 3D Gaussian $\mathcal { N } ( \mu ^ { W } , \Sigma ^ { W } )$ into 2D image space Gaussian $\mathcal { N } ( \mu ^ { I } , \Sigma ^ { I } )$ as follows:

$$
\begin{array} { l } { { \displaystyle \mu ^ { I } = \pi ( T _ { C W } \mu ^ { W } ) , } } \\ { { \displaystyle \Sigma ^ { I } = \mathbf { J } \mathbf { R } \Sigma ^ { W } ( \mathbf { J } \mathbf { R } ) ^ { T } , } } \end{array}\tag{4}
$$

where ?? is the camera perspective projection function, $T _ { C W } \in { \bf S } { \bf E } ( 3 )$ is the camera pose of a viewpoint in the world coordinate system. J is the Jacobian of the perspective projection function and ?? â ????(3) is the rotation component of the camera pose $T _ { C W }$ . Similar to color, a rendered depth for a pixel ?? can be written as:

$$
D ( p ) = \sum _ { i = 1 } ^ { n } d _ { i } ^ { C } f ( p ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - f ( p ) ) ,\tag{5}
$$

where $d _ { i } ^ { C } = \left[ T _ { C W } \mu _ { i } ^ { W } \right] _ { Z }$ is the depth, i.e. ?? coordinate, of a Gaussian ?? in camera coordinate frame. This formulation ensures that the rendered Gaussian splats are differentiable with respect to their 3D Gaussian splat parameters. By employing gradient descent optimization, Gaussian splats iteratively refine their optical and geometric parameters, thereby enabling an accurate representation of the scene with high fidelity.

## 3.3. Differentiable camera pose estimation

The formulation of the projected 2D Gaussian splats in (4) ensures that they are differentiable with respect to the camera pose $T _ { C W }$ as well. Applying the chain rule to (4):

$$
\frac { \partial \mu ^ { I } } { \partial T _ { C W } } = \frac { \partial \mu ^ { I } } { \partial \mu ^ { C } } \frac { \partial \mu ^ { C } } { \partial T _ { C W } } ,
$$

<!-- image-->  
Fig. 2. UDGS-SLAM pipeline consists of three phases: neural depth estimation and local consistency enforcement (left), image rendering and loss computation (middle), and camera pose estimation and map parameter updates (right).

$$
\frac { \partial \varSigma ^ { I } } { \partial T _ { C W } } = \frac { \partial \varSigma ^ { I } } { \partial \mathbf { J } } \frac { \partial \mathbf { J } } { \partial \mu ^ { C } } \frac { \partial \mu ^ { C } } { \partial T _ { C W } } + \frac { \partial \varSigma ^ { I } } { \partial \mathbf { R } } \frac { \partial \mathbf { R } } { \partial T _ { C W } } ,\tag{6}
$$

where $\mu ^ { C }$ represents the 3D position of a Gaussian splat in the camera coordinate frame. Following [19], the derivatives with respect to the camera pose $T _ { C W }$ are derived using the exponential and logarithmic mapping between Lie algebra and the Lie group as follows,

$$
\frac { \partial \mu ^ { C } } { \partial T _ { C W } } = I - \varOmega ^ { + } ,
$$

$$
\frac { \partial { \bf R } } { \partial T _ { C W } } = \left[ 0 \mathrm { ~ } - { \bf R } _ { 1 } ^ { + } \right] ,\tag{7}
$$

where $\varOmega ^ { + }$ and $\mathbf { R } _ { i } ^ { + }$ represent the skew matrices of $\mu ^ { c }$ and the ??th column of $\mathbf { R } ,$ respectively.

## 4. SLAM pipeline

This section presents the details of the UDGS-SLAM pipeline. An overview of the system is summarized in Fig. 2.

## 4.1. Neural depth estimation

The UniDepth network is utilized to estimate the scene depth from a single RGB image captured by a monocular camera [27]. The network extracts color features from the RGB image and predicts the depth values based on these features. UniDepth employs different backbone encoders, with our findings showing that the ViT-style large-size model encoder delivers the highest accuracy. A performance comparison among different backbones is presented as an ablation study in Section 6. Regardless of the backbone, the estimated depth image is not locally consistent. Similar to stereo-depth estimation [58], the UniDepth estimated values exhibit a left-skewed pattern with heavy right tails, as shown in Fig. 3.c. These heavy tail patterns are particularly noticeable at transitions between proximal and distal objects (see Fig. 3.a and b). In the Gaussian Splatting framework, the color features (from the RGB image) and the predicted depth image from the Unidepth network are merged by associating each pixelâs color with its corresponding depth estimate. This combined input is then used to optimize the parameters of the 3D Gaussians, ensuring consistency in both spatial geometry (depth) and appearance (color). Empirical observations have indicated that filtering out these extreme values and ensuring local consistency in the depth map can enhance trajectory and map estimation accuracy. To assure local consistency, We introduce a straightforward yet effective statistical filtering method. The method retains only those depth values falling within the Interquartile Range (IQR), and marks any outliers beyond this range as invalid. Subsequently, only the valid depth values are utilized for geometric error computation. The use of (IQR) for filtering depth estimates is justified by its robustness to outliers without assuming data normality, making it particularly suitable for depth data with non-Gaussian noise distributions. IQR-based filtering offers computational efficiency critical for real-time SLAM systems while providing comparable performance to more complex methods [59]. Due to its practicality, IQR has been effectively applied for outliers rejection in similar application [60,61], where its balance of robustness and efficiency outweighs the theoretical advantages of computationally intensive alternatives like deep learning-based outlier rejection. After applying statistical filtering to the depth image, a local consistent depth image is obtained, with outliers marked as invalid values, as presented in Fig. 3.d. The ablation studies highlight the importance of the statistical filter, as the performance degrades when it is not used, as shown in Table 4.

## 4.2. Rendering and loss computation

The 3D Gaussians ?? can be rendered for a viewpoint of given camera pose $T _ { C W }$ via differentiable rasterization. Rasterization involves sorting and alpha-blending of the Gaussians as outlined in Section 3.2. The derivatives for all parameters are calculated explicitly. We have adopted the implementations from [19,62] for rendering RGB and depth images, respectively. Once an RGB image is rendered, the photometric error is calculated by comparing the rendered image to the captured image as follows,

$$
E _ { p h o } = \left\| { \cal I } - \hat { \mathrm { I } } ( { \bf G } , T _ { C W } ) \right\| _ { 1 } ,\tag{8}
$$

where ?? is the input frame and $\hat { \mathrm { I } } ( \mathbf { G } , T _ { C W } )$ is the rendered rgb image of the Gaussians G at a view point of a given camera pose $T _ { C W }$ . The rendered RGB image can be computed as explained in Eq. (3). Similarly, the geometric error can be computed as

$$
E _ { g e o } = \left\| D - \hat { \mathbf { D } } ( \mathbf { G } , T _ { C W } ) \right\| _ { 1 } ,\tag{9}
$$

where ?? is the filtered neural depth calculated as presented in Section 4.1 and $\hat { \mathrm { D } } ( \mathbf { G } , T _ { C W } )$ is the rendered depth image. The rendered depth image can be computed as explained in $\operatorname { E q . }$ (5). A total loss function can then be formulated from a weighted combination of the photometric and geometric errors as follows,

$$
\begin{array} { r } { \mathcal { L } ( \mathbf { G } , T _ { C W } ) = \lambda E _ { p h o } + ( 1 - \lambda ) E _ { g e o } , } \end{array}\tag{10}
$$

where ?? is a weighting factor that balances the contribution of the photometric error $E _ { p h o }$ and the geometric error $E _ { g e o }$ in the total loss.

## 4.3. Tracking and mapping

This section introduces the different steps used for refining 3D Gaussian splats (map) and camera pose.

<!-- image-->  
Fig. 3. The UniDepth network is used to estimate the scene depth from a single RGB image(a). The depths at transitions between proximal and distal objects exhibit nonconsistency (b). This inconsistency makes a lift-skewed distribution with right heavy tails (c). By applying statistical filtering, local consistency is achieved by marking outliers as invalid values (d).

## 4.3.1. Keyframes management

Although it is theoretically possible to use all previously obtained RGB and depth images for refining the map parameters and camera poses, this method is practically infeasible due to computational constraints. Instead of using all the images, carefully selected keyframes within a small window $\mathcal { W } _ { k }$ are used. The keyframes should not be redundant, should observe the same area [63], and should maintain a wide baseline among them to provide robust multiview constraints [64, 65]. following Matsuki et al. [19] and DSO [63], a small window $\mathcal { W } _ { k }$ of keyframes is maintained. A new frame is considered to be a keyframe by assessing its covisibility, which is calculated by determining the intersection over union of the observed Gaussians between the current frame and the previous keyframe. A new keyframe is added to the window if the covisibility falls below a certain threshold or if the translation (baseline) between the current frame and the previous keyframe is significantly large relative to the median depth.

## 4.3.2. Gaussians insertion and pruning management

As the camera moves, newly unobserved areas come into view, requiring the insertion of new Gaussians to capture their optical and geometric properties. New Gaussians are inserted at each keyframe for these newly observed regions, with their means $\mu ^ { w }$ initialized by backprojecting the filtered UniDepth-estimated depth values. Their optical properties are derived from the corresponding RGB input image, and an initial opacity of 0.5 is assigned to ensure stable initialization for gradient-based optimization, preventing any single splat from dominating the rendering process. Instead of inserting one Gaussian per pixel, a structured placement strategy is employed to reduce redundancy while preserving fine details. An adaptive density control mechanism prioritizes insertions in high-gradient areas while minimizing unnecessary additions in low-texture regions, optimizing computational efficiency without sacrificing accuracy. The newly inserted and existing Gaussians are refined through sequential optimization, adjusting their properties dynamically. In addition to Gaussian insertion, excess Gaussians are pruned; if a Gaussian within a keyframe is not observed in at least three subsequent frames, it is considered geometrically unstable and removed from the scene.

## 4.3.3. Tracking and mapping

The purpose of the tracking and mapping module is to maintain a 3D Gaussian map of the scene where each Gaussian is defined as explained in Eq. (1) and to estimate the camera pose for each obtained RGB frame. In addition, the map should be coherent and consistent enough to allow rendering RGB images with high fidelity. To achieve this consistency, a window of previously obtained keyframes $\mathcal { W } _ { r }$ is used along with the current window of keyframes $\mathcal { W } _ { k }$ for map and poses refinement. Similar to [19], two past keyframes are selected randomly to form $w _ { r }$ . The 3D Gaussian parameters (map parameters) and the camera pose estimation are formulated as an optimization problem and their parameters can be estimated by minimizing the loss function in (10) as follows,

$$
\operatorname* { m i n } _ { T _ { C W } ^ { k } , \mathbf { G } , \forall k \in \mathcal { W } } \sum _ { \forall k \in \mathcal { W } } ( \lambda E _ { p h o } ^ { k } + ( 1 - \lambda ) E _ { g e o } ^ { k } ) ,\tag{11}
$$

where ?? is a keyframe and $\mathcal { W } = \mathcal { W } _ { r } \cup \mathcal { W } _ { k }$ is an optimization window of keyframes, calculated as the union of the randomly selected previous keyframes î?? and the keyframes in the current window $\mathcal { W } _ { k }$

## 4.4. Pipeline initialization

Unlike Matsuki et al. in their monocular camera pipeline [19], UDGS-SLAM does not use any prior information about scene depth in the initialization step. Instead, the Gaussians are initialized at the depths of the filtered UniDepth estimated depth image. Their color proprieties are obtained from the corresponding pixels in the input RGB image. Then, The Gaussians parameters are refined further by minimizing the loss function in (10) using gradient descent for the map parameters solely. During initialization, the initial camera pose $T _ { C W }$ is set to $[ \mathbf { I } _ { 3 \mathbf { x } 3 } | \mathbf { 0 } _ { 3 \mathbf { x } 1 } ]$ or to the pose of the camera in the world coordinate system if it is known.

## 5. Experiments and results

An evaluation of the proposed system is conducted on real-world dataset. Additionally, an ablation study is proposed to justify the design choices and to investigate the impact of different UniDepth backbone encoders on the results. This section presents the experiment setup and the results while the ablation study is presented in the subsequent section (Section 6).

## 5.1. Experiment setup

## 5.1.1. Dataset

The proposed approach is evaluated on TUM RGB-D dataset [66] (3 sequences). Although the dataset includes depth images, RGB images are only used in the proposed approach. Camera pose estimates are compared with the provided ground truth poses. For the ablation study, only one sequence (fr1-desk) from the dataset is used to assess the performance variations among different backbone encoders.

## 5.1.2. Implementation details

UDGS-SLAM is tested on a laptop with Intel Core i7-13700H, 5.0 GHz, 32 GB RAM, and a single Nvidia GeForce RTX 4070 GPU. 3D Gaussian rendering relies on CUDA C++ implementation proposed at [14,19]. The rest of the pipeline is developed with Pytorch. UDGS-SLAM was able to achieve 5 FPS during the experiments.

## 5.1.3. Metrics

For camera pose estimation, the pipeline reports the Root Mean Square Error of Absolute Trajectory Error (ATE RMSE â) of the estimated keyframes. To evaluate map and rendering quality, the pipeline reports standard metrics: Peak Signal-to-Noise Ration (PSNR â), Structural Similarity Index Measurement (SSIM â), and Learned Perceptual Image Patch Similarity (LPIPS â) [50].

## 5.1.4. BaseLine methods

Since UDGS-SLAM does not incorporate loop closure, it is compared with similar SLAM methods that also lack explicit loop closure routines, which include NeRF and Gaussian splatting-based methods that achieve photorealistic map representation. Given the proposed solution reliance on monocular images, it is benchmarked against other monocular-based Gaussian splatting solutions. Due to the scarcity of monocular-based Gaussian splatting SLAM solutions, RGB-D Gaussian splatting methods are also considered for a more comprehensive performance comparison. Specifically, for RGB based methods the proposed solution is compared with DROID-VO [67]. DepthCov-VO [16], and MonoGS [19] using its monocular implementation. For RGB-D based methods, the proposed solution is compared with NICE-SLAM [17], Vox-Fusion [18], and SplaTAM [20].

## 5.2. Evaluation

This section presents the results of UDGS-SLAM. The results discussed herein utilize the ViT large model encoder of the UniDepth network and a statistical filter to ensure local consistency. The ablation study section discusses other UniDepth encoder backbones with/ without statistical filtering (see Section 6).

## 5.2.1. Camera tracking accuracy

Fig. 4 presents camera trajectory estimation for 3 sequences. Despite poor RGB image quality (resolution is 640 Ã 480) and high motion blur, UDGS-SLAM gives small ATE RMSE. In Table 1, UDGS-SLAMâs camera pose estimation is benchmarked against various baselines using the TUM RGB-D dataset. A comprehensive quantitative analysis reveals that the proposed method performs well against both Gaussian splatting and non-Gaussian splatting-based methods. Additionally, The comparisons also include methods utilizing both monocular and RGB-D inputs. Remarkably, the proposed approach not only outperforms other monocular-based methods but also exceeds the performance of RGB-D-based methods. It reports the best (lowest) ATE RMSE, achieving the minimum trajectory error compared to all baselines in the fr1- desk sequence - reducing the error by more than 10% from 3.5 cm to 3.0 cm compared to SplaTAM [20]. In the fr2-xyz sequence, it achieves the second-best performance among monocular-based methods, trailing only behind DepthCov-VP [16], and outperforms the RGB-D based method NICE-SLAM [17]. In the fr3-office sequence, although the method surpasses some baselines (DepthCov-VO [16] and Vox-Fusion [18]), its performance is not as strong compared to other sequences. This may be attributed to high motion blur caused by fast camera motion and the larger covered area in the fr3-office sequence, which leads to drift in the estimated trajectory. This highlights potential areas for future improvement.

## 5.2.2. Rendering results

In addition to camera pose tracking estimation, the rendering performance is also analyzed for high photorealistic reconstruction. In UDGS-SLAM, the scene/map is represented by a number of Gaussians as explained in Section 3.1 similar to the depiction in Fig. 1.d. For a given viewpoint of camera pose, the scene can be rendered to produce a photorealistic image similar to the one presented in Fig. 1.f. Using the metrics in Section 5.1.3, Table 2 reports the rendering performance of UDGS-SLAM on TUM dataset showing good rendering metrics across all sequences. The rendering metrics for the fr2-xyz and fr3-office scenarios are higher than those for fr1-desk. This is because the former scenarios feature distinct, non-occluded objects with sufficient separation and varying depths, whereas fr1-desk contains many occluded objects at similar distances.

UDGS-SLAM rendering metrics are compared with several baselines. The average metrics are reported in Table 3. It is worth noting that the rendering metrics of SplaTAM for TUM RGB-D dataset were not reported by its authors [20].

These results clearly demonstrate that the proposed approach not only surpasses monocular-based methods but also outperforms the RGB-D-based Point-SLAM. Furthermore, it achieves comparable performance, closely matching that of MonoGS in its RGB-D configuration, while using only a monocular camera.

## 6. Ablation study

In Table 4, an ablative analysis is conducted to validate the design choices of the UniDepth network. The study examines the performance of different encoder backbones for the UniDepth network, both with (w) and without (w/o) statistical filtering to ensure local consistency. This analysis includes both version 1 (V1) and version 2 (V2) architectures of UniDepth [46,68]. For V1, the ViT Large model and ConvNext are used as backbone encoders. For V2, the ViT Large model, the only available encoder at the time of testing, is utilized. The evaluation is conducted on the fr1-desk sequence of the TUM dataset. The results indicate that the UniDepth V1 network, when combined with the ViT Large model and statistical filtering, achieves the lowest ATE-RMSE and the highest rendering metrics.

<!-- image-->

<!-- image-->  
[b]

<!-- image-->  
Fig. 4. ATE RMSE (âm) of Camera pose estimation using UDGS-SLAM. (a) Fr1-desk sequence trajectory estimation. (b) Fr2-xyz sequence trajectory estimation. (c) Fr3-office sequence trajectory estimation.

Table 1  
Camera tracking results on TUM for monocular and RGB-D. ATE RMSE in (âcm) is reported.
<table><tr><td>Methods</td><td>Input</td><td>Based on</td><td>fr1-desk</td><td>fr2-xyz</td><td>fr3-office</td></tr><tr><td>DROID-VO [67]a</td><td>Monocular</td><td>ConvGRU</td><td>5.2</td><td>10.7</td><td>7.3</td></tr><tr><td>DepthCov-VO [16]a</td><td>Monocular</td><td>Gaussian Process</td><td>5.6</td><td>1.2</td><td>68.8</td></tr><tr><td>MonoGS [19]a</td><td>Monocular</td><td>Gaussian Splatting</td><td>3.78</td><td>4.6</td><td>3.5</td></tr><tr><td>NICE-SLAM [17]a</td><td>RGB-D</td><td>NERF</td><td>4.26</td><td>6.19</td><td>3.87</td></tr><tr><td>Vox-Fusion [18]a</td><td>RGB-D</td><td>NERF</td><td>3.52</td><td>1.49</td><td>26.01</td></tr><tr><td>SplaTAM [20]</td><td>RGB-D</td><td>Gaussian Splatting</td><td>3.35</td><td>1.24</td><td>5.16</td></tr><tr><td>UDGS-SLAM(ours)</td><td>Monocular</td><td>Gaussian Splatting</td><td>3.0</td><td>2.2</td><td>11.3</td></tr></table>

a The results are adapted from [19].

Table 2  
UDGS-SLAM rendering metrics for TUM dataset.
<table><tr><td>Metric</td><td>fr1-desk</td><td>fr2-xyz</td><td>fr3-office</td><td>Average</td></tr><tr><td>PSNR â</td><td>23.3</td><td>24.9</td><td>23.6</td><td>24</td></tr><tr><td>SSIM â</td><td>0.79</td><td>0.8</td><td>0.806</td><td>0.8</td></tr><tr><td>LPIPS â</td><td>0.26</td><td>0.22</td><td>0.26</td><td>0.246</td></tr></table>

Table 3

Average rendering metrics for TUM dataset.
<table><tr><td>Method</td><td>Input</td><td>PSNR â</td><td>SSIM â</td><td>LPIPSâ</td></tr><tr><td>MonoGS [19]</td><td>Monocular</td><td>21</td><td>0.7</td><td>0.3</td></tr><tr><td>MonoGS [19]a</td><td>RGB-D</td><td>24.37</td><td>0.804</td><td>0.225</td></tr><tr><td>Point-SLAM [50]a</td><td>RGB-D</td><td>21.39</td><td>0.727</td><td>0.463</td></tr><tr><td>SplaTAM [20]</td><td>RGB-D</td><td></td><td></td><td></td></tr><tr><td>UDGS-SLAM (ours)</td><td>Monocular</td><td>24</td><td>0.8</td><td>0.246</td></tr></table>

a The results are adapted from [19].

## 7. Conclusion

This work presents UDGS-SLAM, a system that adapts 3D Gaussians as its underlying map representation, enabling photorealistic rendering, dense mapping, and camera trajectory optimization without the need for explicit prior knowledge about the scene or camera motion. UDGS-SLAM leverages advances in neural depth estimation from a single RGB image by utilizing depth maps generated by the UniDepth network. Additionally, it employs a straightforward yet effective statistical filtering method to ensure local consistency and enhance estimation and rendering accuracy. The effectiveness of UDGS-SLAM is demonstrated through testing on the TUM RGB-D dataset, where it exhibits competitive performance, achieving results comparable to or better than existing baselines. This work highlights the potential of integrating neural depth estimation from monocular cameras with Gaussian splatting to develop more sophisticated and efficient SLAM methods. Nonetheless, potential improvements in the proposed approach remain. For example, due to their complementary nature, integrating image-IMU depth estimation with neural depth could yield more accurate depth maps, thereby enhancing overall performance. Furthermore, exploring the incorporation of loop closure could increase the global consistency of the map. These aspects will be investigated in future work. Additionally, dynamic scenes can be handled by incorporating motion segmentation to identify dynamic objects. This can be followed by selective filtering or separate modeling of static and dynamic components, allowing for the exclusion of dynamic Gaussians from the mapping process. Alternatively, dynamic objects can be explicitly modeled using separate sets of Gaussians that incorporate velocity parameters.

Table 4  
Ablation analysis on the UniDepth backbone encoders with/without local consistency. The analysis confirms that using a V1-ViT large model encoder with statistical filtering to assure local consistency gives the best performance.
<table><tr><td>Backbone encoder</td><td>Statistical filtering</td><td>ATE-RMSE âcm</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td rowspan="2">V1-ViT large</td><td>w</td><td>3</td><td>23.3</td><td>0.79</td><td>0.26</td></tr><tr><td>w/o</td><td>3.6</td><td>23</td><td>0.75</td><td>0.27</td></tr><tr><td>V1-ConvNext large</td><td>W</td><td>3.7</td><td>21</td><td>0.7</td><td>0.28</td></tr><tr><td rowspan="2">V2-ViT large</td><td>w</td><td>6.2</td><td>23</td><td>0.75</td><td>0.28</td></tr><tr><td>w/o</td><td>5.8</td><td>23</td><td>0.7</td><td>0.25</td></tr></table>

## CRediT authorship contribution statement

Mostafa Mansour: Writing â original draft, Visualization, Validation, Software, Resources, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. Ahmed Abdelsalam: Writing â review & editing, Writing â original draft, Visualization, Validation, Software, Methodology, Investigation, Formal analysis, Data curation. Ari Happonen: Supervision, Project administration, Funding acquisition. Jari Porras: Supervision, Project administration, Funding acquisition. Esa Rahtu: Writing â review & editing, Supervision, Project administration.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

Data will be made available on request.

## References

[1] Durrant-Whyte HF, Bailey T. Simultaneous localization and mapping: part I. IEEE Robot Autom Mag 2006;13:99â110, URL https://api.semanticscholar.org/ CorpusID:8061430.

[2] Jiang X, Zhu L, Liu J, Song A. A SLAM-based 6DoF controller with smooth auto-calibration for virtual reality. Vis Comput 2022;39(9):3873â86.

[3] Reitmayr G, Langlotz T, Wagner D, Mulloni A, Schall G, Schmalstieg D, Pan Q. Simultaneous localization and mapping for augmented reality. In: Proceedings of international symposium on ubiquitous virtual reality. 2010, International Symposium on Ubiquitous Virtual Reality : ISUVR 2010 ; Conference date: 07-07-2010 Through 10-07-2010.

[4] Macario Barros A, Michel M, Moline Y, Corre G, Carrel F. A comprehensive survey of visual SLAM algorithms. Robotics 2022;11(1). URL https://www.mdpi. com/2218-6581/11/1/24.

[5] Cadena C, Carlone L, Carrillo H, Latif Y, Scaramuzza D, Neira J, Reid I, Leonard JJ. Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age. IEEE Trans Robot 2016;32(6):1309â32.

[6] Campos C, Elvira R, Rodriguez JJG, M. Montiel JM, D. Tardos J. ORB-SLAM3: An accurate open-source library for visual, visualâInertial, and multimap SLAM. IEEE Trans Robot 2021;37(6):1874â90.

[7] Davison AJ, Reid ID, Molton ND, Stasse O. MonoSLAM: Real-time single camera SLAM. IEEE Trans Pattern Anal Mach Intell 2007;29(6):1052â67.

[8] Mur-Artal R, TardÃ³s JD. ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras. IEEE Trans Robot 2017;33(5):1255â62.

[9] Whelan T, Salas-Moreno RF, Glocker B, Davison AJ, Leutenegger S. ElasticFusion. Int J Robot Res 2016;35(14):1697â716.

[10] Engel J, SchÃ¶ps T, Cremers D. LSD-SLAM: Large-scale direct monocular SLAM. In: Fleet D, Pajdla T, Schiele B, Tuytelaars T, editors. Computer vision â ECCV 2014. Cham: Springer International Publishing; 2014, p. 834â49.

[11] Whelan T, Kaess M, Johannsson H, Fallon M, Leonard JJ, McDonald J. Realtime large-scale dense RGB-D SLAM with volumetric fusion. Int J Robot Res 2015;34(4â5):598â626.

[12] Kerl C, Sturm J, Cremers D. Robust odometry estimation for RGB-D cameras. In: 2013 IEEE international conference on robotics and automation. 2013, p. 3748â54.

[13] Mildenhall B, Srinivasan PP, Tancik M, Barron JT, Ramamoorthi R, Ng R. NeRF: Representing scenes as neural radiance fields for view synthesis. 2020, arXiv:2003.08934. URL https://arxiv.org/abs/2003.08934.

[14] Kerbl B, Kopanas G, LeimkÃ¼hler T, Drettakis G. 3D Gaussian splatting for realtime radiance field rendering. ACM Trans Graph 2023;42(4). URL https://reposam.inria.fr/fungraph/3d-gaussian-splatting/.

[15] Tosi F, Zhang Y, Gong Z, SandstrÃ¶m E, Mattoccia S, Oswald MR, Poggi M. How NeRFs and 3D Gaussian splatting are reshaping SLAM: a survey. 2024, arXiv:2402.13255. URL https://arxiv.org/abs/2402.13255.

[16] Dexheimer E, Davison AJ. Learning a depth covariance function. 2024, arXiv: 2303.12157. URL https://arxiv.org/abs/2303.12157.

[17] Zhu Z, Peng S, Larsson V, Xu W, Bao H, Cui Z, Oswald MR, Pollefeys M. NICE-SLAM: Neural implicit scalable encoding for SLAM. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. CVPR, 2022.

[18] Yang X, Li H, Zhai H, Ming Y, Liu Y, Zhang G. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In: 2022 IEEE international symposium on mixed and augmented reality. ISMAR, 2022, p. 499â507.

[19] Matsuki H, Murai R, Kelly PHJ, Davison AJ. Gaussian splatting SLAM. 2024, arXiv:2312.06741. URL https://arxiv.org/abs/2312.06741.

[20] Keetha N, Karhade J, Jatavallabhula KM, Yang G, Scherer S, Ramanan D, Luiten J. SplaTAM: Splat, track & map 3D Gaussians for dense RGB-D SLAM. 2023, arXiv Preprint.

[21] Ha S, Yeon J, Yu H. RGBD GS-ICP SLAM. 2024, arXiv:2403.12550. URL https: //arxiv.org/abs/2403.12550.

[22] Hu J, Chen X, Feng B, Li G, Yang L, Bao H, Zhang G, Cui Z. CG-SLAM: Efficient dense RGB-D SLAM in a consistent uncertainty-aware 3D Gaussian field. 2024, arXiv:2403.16095. URL https://arxiv.org/abs/2403.16095.

[23] Sun LC, Bhatt NP, Liu JC, Fan Z, Wang Z, Humphreys TE, Topcu U. MM3DGS SLAM: Multi-modal 3D Gaussian splatting for SLAM using vision, depth, and inertial measurements. 2024, arXiv:2404.00923. URL https://arxiv.org/abs/2404. 00923.

[24] Yan C, Qu D, Xu D, Zhao B, Wang Z, Wang D, Li X. GS-SLAM: Dense visual SLAM with 3D Gaussian splatting. 2024, arXiv:2311.11700. URL https://arxiv. org/abs/2311.11700.

[25] Yugay V, Li Y, Gevers T, Oswald MR. Gaussian-SLAM: Photo-realistic dense SLAM with Gaussian splatting. 2023, arXiv:2312.10070.

[26] Masoumian A, Rashwan HA, Cristiano J, Asif MS, Puig D. Monocular depth estimation using deep learning: A review. Sensors 2022;22(14). URL https: //www.mdpi.com/1424-8220/22/14/5353.

[27] Piccinelli L, Yang Y-H, Sakaridis C, Segu M, Li S, Gool LV, Yu F. UniDepth: Universal monocular metric depth estimation. 2024, arXiv:2403.18913. URL https://arxiv.org/abs/2403.18913.

[28] ÃzyeÅil: A survey of structure from motion\*. - Google Scholar.

[29] Scharstein D, Szeliski R, Zabih R. A taxonomy and evaluation of dense two-frame stereo correspondence algorithms. In: Proceedings IEEE workshop on stereo and multi-baseline vision. SMBV 2001, 2001, p. 131â40, URL https://ieeexplore.ieee. org/document/988771.

[30] Kundu A, Krishna KM, Sivaswamy J. Moving object detection by multi-view geometric techniques from a single camera mounted robot. In: 2009 IEEE/RSJ international conference on intelligent robots and systems. 2009, p. 4306â12, URL https://ieeexplore.ieee.org/abstract/document/5354227. ISSN: 2153-0866.

[31] Eigen D, Puhrsch C, Fergus R. Depth map prediction from a single image using a multi-scale deep network. In: Proceedings of the 27th international conference on neural information processing systems - volume 2. NIPS â14, Cambridge, MA, USA: MIT Press; 2014, p. 2366â74.

[32] Silberman N, Hoiem D, Kohli P, Fergus R. Indoor segmentation and support inference from RGBD images. In: Fitzgibbon A, Lazebnik S, Perona P, Sato Y, Schmid C, editors. Computer vision â ECCV 2012. Berlin, Heidelberg: Springer; 2012, p. 746â60.

[33] Menze M, Geiger A. Object scene flow for autonomous vehicles. In: 2015 IEEE conference on computer vision and pattern recognition. CVPR, 2015, p. 3061â70, URL https://ieeexplore.ieee.org/document/7298925. ISSN: 1063-6919.

[34] Bhat SF, Alhashim I, Wonka P. AdaBins: Depth estimation using adaptive bins. 2021, p. 4009â18, URL https://openaccess.thecvf.com/content/CVPR2021/html/ Bhat_AdaBins_Depth_Estimation_Using_Adaptive_Bins_CVPR_2021_paper.html.

[35] Bhat SF, Alhashim I, Wonka P. LocalBins: Improving depth estimation by learning local distributions. In: Avidan S, Brostow G, CissÃ© M, Farinella GM, Hassner T, editors. Computer vision â ECCV 2022. Cham: Springer Nature Switzerland; 2022, p. 480â96.

[36] Li Z, Wang X, Liu X, Jiang J. BinsFormer: Revisiting adaptive bins for monocular depth estimation. IEEE Trans Image Process 2024;33:3964â76, URL https:// ieeexplore.ieee.org/document/10570231. Conference Name: IEEE Transactions on Image Processing.

[37] Yuan W, Gu X, Dai Z, Zhu S, Tan P. Neural window fully-connected CRFs for monocular depth estimation. In: 2022 IEEE/CVF conference on computer vision and pattern recognition. CVPR, 2022, p. 3906â15, URL https://ieeexplore.ieee. org/document/9879975. ISSN: 2575-7075.

[38] Fu H, Gong M, Wang C, Batmanghelich K, Tao D. Deep ordinal regression network for monocular depth estimation. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2018, p. 2002â11, URL https://ieeexplore. ieee.org/document/8578312/. Conference Name: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) ISBN: 9781538664209 Place: Salt Lake City, UT Publisher: IEEE.

[39] Patil V, Sakaridis C, Liniger A, Van Gool L. P3Depth: Monocular depth estimation with a piecewise planarity prior. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition. CVPR, 2022, p. 1600â11, URL https://ieeexplore. ieee.org/document/9880373/. Conference Name: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) ISBN: 9781665469463 Place: New Orleans, LA, USA Publisher: IEEE.

[40] Piccinelli L, Sakaridis C, Yu F. iDisc: Internal discretization for monocular depth estimation. In: 2023 IEEE/CVF conference on computer vision and pattern recognition. CVPR, Vancouver, BC, Canada: IEEE; 2023, p. 21477â87, URL https://ieeexplore.ieee.org/document/10204996/.

[41] Ranftl R, Bochkovskiy A, Koltun V. Vision transformers for dense prediction. In: 2021 IEEE/CVF international conference on computer vision. ICCV, Montreal, QC, Canada: IEEE; 2021, p. 12159â68, URL https://ieeexplore.ieee.org/ document/9711226/.

[42] Wang Y, Chen X, You Y, Li LE, Hariharan B, Campbell M, Weinberger KQ, Chao W-L. Train in Germany, test in the USA: Making 3D object detectors generalize. In: 2020 IEEE/CVF conference on computer vision and pattern recognition. CVPR, 2020, p. 11710â20, URL https://ieeexplore.ieee.org/document/9156543. ISSN: 2575-7075.

[43] Guizilini V, Vasiljevic I, Chen D, AmbruÅ R, Gaidon A. Towards zero-shot scaleaware monocular depth estimation. In: 2023 IEEE/CVF international conference on computer vision. ICCV, Paris, France: IEEE; 2023, p. 9199â209, URL https: //ieeexplore.ieee.org/document/10377512/.

[44] Yin W, Zhang C, Chen H, Cai Z, Yu G, Wang K, Chen X, Shen C. Metric3D: Towards zero-shot metric 3D prediction from a single image. 2023, http://dx.doi. org/10.48550/arXiv.2307.10984, URL http://arxiv.org/abs/2307.10984. arXiv: 2307.10984 [cs].

[45] Hu M, Yin W, Zhang C, Cai Z, Long X, Chen H, Wang K, Yu G, Shen C, Shen S. Metric3D v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Trans Pattern Anal Mach Intell 2024;46(12):10579â96.

[46] Piccinelli L, Yang Y-H, Sakaridis C, Segu M, Li S, Van Gool L, Yu F. UniDepth: Universal Monocular Metric Depth Estimation. 2024, p. 10106â16, URL https://openaccess.thecvf.com/content/CVPR2024/html/Piccinelli_UniDepth_ Universal_Monocular_Metric_Depth_Estimation_CVPR_2024_paper.html.

[47] Mildenhall B, Srinivasan PP, Tancik M, Barron JT, Ramamoorthi R, Ng R. NeRF: Representing scenes as neural radiance fields for view synthesis. 2020, arXiv:2003.08934. URL https://arxiv.org/abs/2003.08934.

[48] SchÃ¶nberger JL, Zheng E, Frahm J-M, Pollefeys M. Pixelwise view selection for unstructured multi-view stereo. In: Leibe B, Matas J, Sebe N, Welling M, editors. Computer vision â ECCV 2016. Cham: Springer International Publishing; 2016, p. 501â18.

[49] Sucar E, Liu S, Ortiz J, Davison AJ. iMAP: Implicit mapping and positioning in real-time. 2021, arXiv:2103.12352. URL https://arxiv.org/abs/2103.12352.

[50] SandstrÃ¶m E, Li Y, Gool LV, Oswald MR. Point-SLAM: Dense neural point cloud-based SLAM. 2023, arXiv:2304.04278. URL https://arxiv.org/abs/2304. 04278.

[51] Turkulainen M, Ren X, Melekhov I, Seiskari O, Rahtu E, Kannala J. DNsplatter: Depth and normal priors for Gaussian splatting and meshing. 2024, arXiv:2403.17822. URL https://arxiv.org/abs/2403.17822.

[52] Ha S, Yeon J, Yu H. RGBD GS-ICP SLAM. 2024, arXiv:2403.12550. URL https: //arxiv.org/abs/2403.12550.

[53] Hu J, Chen X, Feng B, Li G, Yang L, Bao H, Zhang G, Cui Z. CG-SLAM: Efficient dense RGB-D SLAM in a consistent uncertainty-aware 3D Gaussian field. 2024, arXiv:2403.16095. URL https://arxiv.org/abs/2403.16095.

[54] Sun LC, Bhatt NP, Liu JC, Fan Z, Wang Z, Humphreys TE, Topcu U. MM3Dgs SLAM: Multi-modal 3D Gaussian splatting for SLAM using vision, depth, and inertial measurements. 2024, arXiv:2404.00923. URL https://arxiv.org/abs/2404. 00923.

[55] Yan C, Qu D, Xu D, Zhao B, Wang Z, Wang D, Li X. GS-SLAM: Dense visual SLAM with 3D Gaussian splatting. 2024, arXiv:2311.11700. URL https://arxiv. org/abs/2311.11700.

[56] Li J, Zhang J, Bai X, Zheng J, Ning X, Zhou J, Gu L. DNGaussian: Optimizing sparse-view 3D Gaussian radiance fields with global-local depth normalization. 2024, arXiv:2403.06912. URL https://arxiv.org/abs/2403.06912.

[57] Lee Y, Choi J, Jung D, Yun J, Ryu S, Manocha D, Yeon S. Mode-GS: Monocular depth guided anchored 3D Gaussian splatting for robust ground-view scene rendering. 2024, arXiv:2410.04646. URL https://arxiv.org/abs/2410.04646.

[58] Abdelsalam A, Mansour M, Porras J, Happonen A. Depth accuracy analysis of the ZED 2i stereo camera in an indoor environment. Robot Auton Syst 2024;179:104753, URL https://www.sciencedirect.com/science/article/pii/ S0921889024001374.

[59] Rousseeuw P, Hubert M. Robust statistics for outlier detection. Wiley Interdiscip Rev: Data Min Knowl Discov 2011;1:73â9.

[60] Cadena C, Carlone L, Carrillo H, Latif Y, Scaramuzza D, Neira J, Reid I, Leonard JJ. Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age. IEEE Trans Robot 2016;32(6):1309â32.

[61] Mur-Artal R, Tardos JD. ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras. IEEE Trans Robot 2017;33(5):1255â62.

[62] Kaehler A, Bradski G. Learning OpenCV 3: computer vision in C++ with the OpenCV library. " OâReilly Media, Inc."; 2016.

[63] Engel JJ, Koltun V, Cremers D. Direct sparse odometry. IEEE Trans Pattern Anal Mach Intell 2016;40:611â25, URL https://api.semanticscholar.org/ CorpusID:3299195.

[64] Vishnyakov BV, Vizilter YV, Knyaz VA, Malin IK, Vygolov OV, Zheltov SY. Stereo sequences analysis for dynamic scene understanding in a driver assistance system. In: Proc. automated visual inspection and machine vision, Munich, Germany. Vol. 9530, SPIE; 2015.

[65] Mansour M, Davidson P, Stepanov O, PichÃ© R. Relative importance of binocular disparity and motion parallax for depth estimation: A computer vision approach. Remote Sens 2019;11(17). URL https://www.mdpi.com/2072-4292/11/17/1990.

[66] Sturm J, Engelhard N, Endres F, Burgard W, Cremers D. A benchmark for the evaluation of RGB-d SLAM systems. In: Proc. of the international conference on intelligent robot systems. IROS, 2012.

[67] Teed Z, Deng J. DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras. Adv Neural Inf Process Syst 2021.

[68] lpiccinelli-eth. Unidepth. 2024, https://github.com/lpiccinelli-eth/UniDepth. (Accessed 16 July 2024).