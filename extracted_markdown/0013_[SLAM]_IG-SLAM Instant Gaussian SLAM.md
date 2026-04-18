# IG-SLAM: Instant Gaussian SLAM

F. Aykut SarÄ±kamÄ±sÂ¸ A. AydÄ±n Alatan Center for Image Analysis (OGAM), EEE Department, METU, Turkey

<!-- image-->  
Photo-SLAM

<!-- image-->  
IG-SLAM

<!-- image-->  
Ground Truth

Figure 1. Qualitative rendering results from Photo-SLAM [11] and IG-SLAM. We compare the visual quality of the methods on the large-scale EuRoC dataset [3].

## Abstract

3D Gaussian Splatting has recently shown promising results as an alternative scene representation in SLAM systems to neural implicit representations. However, current methods either lack dense depth maps to supervise the mapping process or detailed training designs that consider the scale of the environment. To address these drawbacks, we present IG-SLAM, a dense RGB-only SLAM system that employs robust dense SLAM methods for tracking and combines them with Gaussian Splatting. A 3D map of the environment is constructed using accurate pose and dense depth provided by tracking. Additionally, we utilize depth uncertainty in map optimization to improve 3D reconstruction. Our decay strategy in map optimization enhances convergence and allows the system to run at 10 fps in a single process. We demonstrate competitive performance with state-of-the-art RGB-only SLAM systems while achieving faster operation speeds. We present our experiments on the Replica, TUM-RGBD, ScanNet, and EuRoC datasets. The system achieves photo-realistic 3D reconstruction in largescale sequences, particularly in the EuRoC dataset.

## 1. Introduction

Dense Simultaneous Localization and Mapping (SLAM) is a fundamental problem in computer vision with numerous applications in robotics, augmented reality, virtual reality, and more. Any SLAM system must operate in real-time and scale to large scenes for all these real-world applications. Additionally, the system must be robust against noisy visual sensor measurements.

The prominent scene representation is a 3D point cloud in traditional Dense SLAM systems. However, point clouds are an impoverished representation of the world. As a sparse representation, the point clouds do not provide watertight, photo-realistic depictions of the environment. Recently, two promising scene representations have been introduced and studied in the SLAM literature: Neural Radiance Fields (NeRF) [17] and Gaussian Splatting [14].

Earlier dense SLAM studies that equip NeRF as an only-scene representation [31, 49] achieved 3D reconstruction without camera poses in real-time. Several following studies [5, 26, 47] integrate classical SLAM methods such as tracking by feature matching, dense-bundle adjustment, loop closure, and global bundle adjustment. Several performance improvements are made in later studies [12, 36, 45, 48, 50] by incorporating additional data structures along with NeRF [17], by employing off-the-shelf tracking modules [19, 34] and monocular depth estimation [7]. However, NeRF suffers from slow rendering speed [24]; since the real-time operation is crucial for a SLAM system, slow rendering speed puts NeRF into a disadvantageous position as a scene representation.

Later the following studies incorporate Gaussian Splatting as scene representation: Early works [16, 39, 43] adopt Gaussian Splatting as an only-scene representation and simultaneously track and map the environment in real-time. However, utilizing novel view synthesis methods as both tracking and mapping tools is compelling yet challenging. The difficulty arises because pose and map optimizations are performed jointly. To decouple these two daunting tasks, [11, 27] utilize traditional SLAM methods demonstrating superior performance over only-scene representation methods in terms of reconstruction. However, these studies either lack dense depth supervision or a high frame rate.

Purposely, we introduce IG-SLAM, a deep-learningbased dense SLAM system that achieves photo-realistic 3D reconstruction in real-time. The proposed system features robust pose estimation, refined dense depth maps, and Gaussian Splatting representation. The proposed system frequently performs global dense bundle adjustment to reduce drift. Since the pose and depth maps optimized by a dense SLAM system are often noisy, we utilize depth uncertainty to make the mapping process robust to noise. Our efficient mapping algorithm is optimized specifically to work with dense depth maps enabling our system to operate at high frame rates. We perform extensive experiments on various indoor RGB sequences, demonstrating the robustness, fast operation speed, and scalability of our method. In summary, we make the following contributions:

â¢ We present IG-SLAM, an efficient dense RGB SLAM system that performs at high frame rates, offering scalability and robustness even in challenging conditions.

â¢ A novel 3D reconstruction algorithm that accounts for depth uncertainty, making the 3D reconstruction robust to noise.

â¢ A training procedure to make dense depth supervision for the mapping process as efficient as possible.

## 2. Related Work

## 2.1. Dense Visual SLAM

Pioneering dense SLAM algorithms, DTAM [21] and KinectFusion [20], show that dense SLAM can be performed in real-time despite its computational complexity. DTAM aims to produce dense depth maps associated with the keyframes, known as the view-centric approach. Later research adopted a similar approach but with a crucial distinction. While these traditional approaches generally decouple the optimization of dense maps and poses, some recent works focus on joint optimization. However, optimization of the full-resolution depth map is not feasible due to the high number of independent variables. Therefore, the following research focuses on reducing the computational complexity of joint optimization. For this purpose, BA-Net [32] includes a depth map into the bundle adjustment layer utilizing a basis of depth maps and optimizing the linear combination coefficients. Code-SLAM [2] reduces the dimension of dense maps by an autoencoder-inspired architecture. DROID-SLAM [34] optimizes down-sampled dense maps in a bundle-adjustment layer with a reprojection error, aided by optical flow revisions [33]. A recent work, FlowMap [28], estimates a dense depth map with a convolutional neural network and calculates the pose analytically using the optical flow. As world-centric alternatives to this approach, Neural Radiance Fields [17] and Gaussian Splatting [14] are utilized in the literature.

## 2.2. Neural Radiance Field Scene Representation

NeRF [17] encodes the scene as radiance fields utilizing a simple multi-layer perceptron (MLP). The original NeRF formulation exhibits slow training and rendering speeds. However, several improvements have been proposed on this initial formulation. The cone-shaped rendering [1] is utilized to address anti-aliasing, additional data structures are also employed, such as voxel grid [8, 15, 25], plenoctree [9, 37, 42], hash tables [18] and many more achieve orders of magnitude faster rendering and training compared to the original NeRF [17]. Surface-based methods [22, 38, 40] also unify surface and volume rendering.

The landmark work iNeRF [41] calculates camera poses given a NeRF representation by fixing the NeRF representation and minimizing rendering error by optimizing the camera pose around an initial guess. iMAP [31], as the first representation-only work, optimizes the pose by fixing the NeRF representation and optimizes the map based on the calculated pose. NICE-SLAM [49] introduces a hierarchical coarse-to-fine mapping approach. To decouple map and pose optimization, Orbeez-SLAM [5] leverages robust visual SLAM methods [19] and multi-resolution hash encoding [18]. NeRF-SLAM [26] introduces dense depth maps with covariance and poses generated by the robust dense-SLAM algorithm DROID-SLAM [34]. GO-SLAM employs loop closing and global dense bundle adjustment to achieve globally consistent reconstruction. NICER-SLAM [50] extends NICE-SLAM [49] incorporating off-the-shelf monocular depth and normal estimators. Recently, MoD-SLAM [48] utilizes cone-shaped projection in rendering [1]. GlORIE-SLAM [45] utilizes monocular depth estimation for mapping supervision.

## 2.3. 3D Gaussian Splatting Scene Representation

3D Gaussian Splatting represents the scene as a set of Gaussians of varying colors, shapes, and opacity. Several improvements are proposed for consistency and reconstruction quality. For example, 2D counterpart [10] is also proposed to enhance multi-view consistency. Moreover, the rendering depth with alpha-blending as in the original 3D Gaussian Splatting causes noisy surfaces; hence, more rigorous methods address this issue by utilizing varying depths per Gaussian according to the viewpoint [4, 44].

Due to its fast rendering speed and being an explicit scene representation as opposed to NeRF [17],

<!-- image-->  
Figure 2. System Overview. Our system takes an RGB image stream as input and outputs the camera pose and scene representation in the form of a set of Gaussians. We decouple this objective into two parts: tracking and mapping. Tracking: Keyframes are created and added to the frame graph based on average optical flow. Pretrained GRU refines optical flow between keyframes. Dense bundle adjustment (DBA) is performed on the frame graph, minimizing reprojection error while optimizing the dense depth map and camera pose, and calculating depth map covariance simultaneously. After several iterations, depth maps and camera poses are expected to converge. Mapping: Keyframesâ pose, depth, and covariance obtained from tracking are used for 3D reconstruction. We initialize Gaussians from low covariance regions utilizing the camera pose and depth map. 3D Gaussians are then projected onto the image plane and rendered utilizing a differentiable tile rasterizer. The loss function is a combination of depth and color loss. The depth loss is weighted by covariance. Finally, the loss is backpropagated to optimize Gaussians orientation, scaling, opacity, position, and color designated by orange arrows in the figure. Moreover, Gaussians are split, cloned, and pruned based on the local gradients.

Gaussian Splatting [14] has also quickly gained attention in the SLAM literature. MonoGS [16], GS-SLAM [39], and SplaTAM [13] are pioneering Gaussian-Splatting representation-only SLAM algorithms that jointly optimize Gaussians and the pose. Gaussian-SLAM [43] introduces sub-maps to mitigate neural forgetting. Photo-SLAM [11] decouples tracking and mapping by employing a traditional visual SLAM algorithm [19] as its tracking module and introduces a coarse-to-fine map optimization approach. RTG-SLAM [23] renders depth by considering only the foremost opaque Gaussians. Recent work, Splat-SLAM [27] uses proxy depth maps to supervise map optimization.

## 3. Proposed Method

We provide an overview of the proposed method in Fig. 2. Our tracking algorithm (Sec. 3.1) generates a dense depth map, depth uncertainty, and the camera pose for each keyframe. These outputs are then used to supervise our mapping algorithm (Sec. 3.2). The Gaussians are initialized based on the camera pose and dense depth and are optimized using color and weighted depth loss. Real-time operation is achieved through a sliding window of keyframes.

## 3.1. Tracking

We mainly employ DROID-SLAM [34] as our tracking module. DROID-SLAM maintains two state variables:

camera pose $\mathbf { G } _ { t }$ and inverse depth $\mathbf { d } _ { t }$ for each camera frame t. DROID-SLAM constructs a frame graph $( \nu , \mathcal { E } )$ of keyframes based on co-visibility. Keyframes are selected from all camera frames when the average magnitude of the optical flow for a frame is higher than a certain threshold. If there is a visual overlap between frames i and frame j, an edge is created between the $i ^ { t h }$ and $j ^ { t h }$ vertex in V. This graph is updated during inference. Given the initial pose and depth estimates $( \mathbf { G } _ { i } , \mathbf { d } _ { i } )$ and $( \mathbf { G } _ { j } , \mathbf { d } _ { j } )$ for frame i and j, the optical flow field is estimated by unprojecting the pixels from frame i, projecting them into frame j, and taking the pixel-wise position difference. In other words, the reprojected pixel locations $p _ { i j }$ is calculated as in Eq. (1)

$$
p _ { i j } = \Pi ( \mathbf { G } _ { i j } \circ \Pi ^ { - 1 } ( \mathbf { p } _ { i } , \mathbf { d } _ { i } ) ) , \mathbf { p } _ { i j } \in \mathbb { R } ^ { H \times W \times 2 }\tag{1}
$$

where $\mathbf { G } _ { i j } = \mathbf { G } _ { i } ^ { - 1 } \circ \mathbf { G } _ { i }$ . Then, the optical flow is initially calculated as $p _ { i j } - p _ { j }$ . This estimate is fed into GRU along with a correlation vector which is an inner product between features of the frames. The GRU produces flow revisions $\mathbf { r } _ { i j }$ and confidence weights $\mathbf { w } _ { i j }$ . the refined reprojected pixel locations $\mathbf { p } _ { i j } ^ { * }$ are computed similarly to Eq. (1) incorporating the flow correction from the GRU. Then, the dense bundle adjustment layer minimizes the cost function in Eq. (2).

$$
\begin{array} { r l } & { \mathbf { E } ( \mathbf { G } ^ { \prime } , \mathbf { d } ^ { \prime } ) = \displaystyle \sum _ { i , j \in \mathcal { E } } \big \| \mathbf { p } _ { i j } ^ { * } - \mathbf { p } ^ { \prime } { } _ { i j } \big \| _ { \Sigma _ { i j } } ^ { 2 } } \\ & { \mathbf { p } ^ { \prime } { } _ { i j } = \Pi ( \mathbf { G } ^ { \prime } { } _ { i j } \circ \Pi ^ { - 1 } ( \mathbf { p } _ { i } , \mathbf { d } ^ { \prime } { } _ { i } ) ) } \end{array}\tag{2}
$$

where $\begin{array} { r l } { \Sigma _ { i j } } & { { } = ~ \mathrm { d i a g } ( \mathbf { w } _ { i j } ) } \end{array}$ and $\| . \| _ { \Sigma }$ Mahalanobis norm weighted according to the weights $\mathbf { w } _ { i j }$ . Linearizing Eq. (2) around $( \mathbf { G } ^ { \prime } , \mathbf { d } ^ { \prime } )$ and solve for pose and depth updates $( \Delta \xi , \Delta \mathbf { d } )$ using Gauss-Newton algorithm. The linearized system of equations becomes

$$
H \mathbf { x } = \mathbf { b } , H = \left[ { \begin{array} { c c } { C } & { E } \\ { E ^ { T } } & { P } \end{array} } \right] , \mathbf { x } = \left[ { \begin{array} { c } { \Delta \xi } \\ { \Delta \mathbf { d } } \end{array} } \right] , \mathbf { b } = \left[ { \begin{array} { c } { \mathbf { v } } \\ { \mathbf { w } } \end{array} } \right]\tag{3}
$$

where H is the Hessian matrix, $\mathbf { x } = [ \Delta \xi , \Delta \mathbf { d } ]$ is the pose and depth updates, $\mathbf b = \left[ \mathbf v , \mathbf w \right]$ is the pose and depth residuals, C is the block camera matrix. E is the camera/depth off-diagonal block matrices, and $P$ is the diagonal matrix corresponding to disparities per pixel per keyframe. The bundle adjustment layer operates on the initial flow estimates and updates the keyframesâ pose and depth map. Optical flow is then recalculated by refined poses and depth maps which are subsequently fed back into the dense bundle adjustment layer. After successive iterative refinements on the keyframe graph, the poses and depth maps are expected to converge.

After the dense bundle adjustment step, we compute the covariance for depth estimates. As shown in NeRF-SLAM [26], the same Hessian structure in Eq. (3) can be used to calculate covariance for depth estimates $\Sigma _ { d }$ and poses $\Sigma _ { G }$ as shown in Eq. (4). The depth covariance is used both as a mask for initializing Gaussians and as weights in the depth component of the loss function.

$$
\begin{array} { l } { { \Sigma _ { d } = P ^ { - 1 } + P ^ { - T } E ^ { T } \Sigma _ { G } E P ^ { - 1 } } } \\ { { \Sigma _ { G } = ( L L ^ { T } ) ^ { - 1 } } } \end{array}\tag{4}
$$

Keyframing We utilize all the keyframes that are actively optimized in the tracking process without any filtering. Each keyframe that participates in mapping contains its camera image I, depth map d, depth covariance $\Sigma _ { d } ,$ and pose G. The mapping process accepts a keyframe only if it is not already in the sliding window. Note that, we do not send all the keyframes created in a mapping cycle, but only the most recent one. Therefore, this approach may result in some keyframes being missed during optimization. However, this design choice prevents abrupt changes in the sliding window caused by a sharp camera movement.

Global BA After the number of total keyframes exceeds the sliding window length for the Dense Bundle Adjustment, we regularly perform Global Bundle Adjustment for all existing keyframes on a separate graph as described in GO-SLAM [47]. The graph is constructed utilizing a distance metric, where the distance between frame pairs is the average optical flow magnitude. Graph edges are established between consecutive keyframes and those that are close according to the distance metric. Dense bundle adjustment is then applied based on this graph every 10 keyframes. The pose and depth maps are updated at the start of every mapping cycle, along with their covariances. We perform one last global BA at the end of tracking.

## 3.2. Mapping

The mapping process is responsible for 3D reconstruction with keyframes equipped with pose, image, depth, and covariance acquired from the tracking process.

Representation We adopt Gaussian Splatting [14] as scene representation. A Gaussian function is described by Eq. (5)

$$
G ( \mathbf { x } ) = \exp \left( \frac { 1 } { 2 } ( { \mathbf { x } } - { \boldsymbol { \mu } } ) ^ { T } { \Sigma ^ { - 1 } } ( { \mathbf { x } } - { \boldsymbol { \mu } } ) \right)\tag{5}
$$

where $\mu$ and Î£ are the mean and covariance which define the position and shape of this Gaussian. To ensure that the covariance remains semi-definite during optimization, covariance Î£ is decomposed into $R S S ^ { T } \bar { R } ^ { T }$ where R is the rotation matrix and S is the scaling matrix. In addition to position, rotation, and scaling, opacity Î± and color c are also optimized. Although the original implementation parameterizes color as spherical harmonic coefficients, our algorithm optimizes the color directly. The projection of a 3D covariance is formulated as $\Sigma ^ { \prime } \ = \ J \bar { R } \bar { \Sigma } R ^ { T } J ^ { T }$ where R is the rotation component of the world-to-camera transformation $T _ { c w }$ and J is the Jacobian of the affine approximation of the projective transformation $P \left[ 5 1 \right]$ . The position is projected directly as $\mu ^ { \prime } = P T _ { c w } \mu$

Rendering A set of Gaussians $\mathcal { N }$ visible from a viewpoint, is first projected onto the image plane. 2D Gaussians are then sorted according to their depths and are rasterized via Î±-blending as described in Eq. (6) for color and depth.

$$
\hat { C } = \sum _ { i \in \cal N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , ~ \hat { D } = \sum _ { i \in \cal N } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } )\tag{6}
$$

Hierarchical Optimization Since dense depth maps for keyframes are available, we adopt a training strategy similar to RGB-D MonoGS [16] but utilizing a coarse-to-fine training strategy inspired by Photo-SLAM [11] and Instant-NGP [18].

For each keyframe, an image pyramid is constructed by downsampling image, depth, and covariance by a factor of s using bilinear interpolation, as in Eq. (7)

$$
\begin{array} { l } { { \bf K F } _ { i } ^ { l } = \{ I _ { i } ^ { l } , { \bf d } _ { i } ^ { l } , \Sigma _ { d i } ^ { l } \} } \\ { { \cal I } _ { i } ^ { l } = I _ { i } ^ { 0 } \downarrow s ^ { l } , ~ { \bf d } _ { i } ^ { l } = { \bf d } _ { i } ^ { 0 } \downarrow s ^ { l } , ~ \Sigma _ { d i } ^ { l } = \Sigma _ { d i } ^ { 0 } \downarrow s ^ { l } } \end{array}\tag{7}
$$

where â denotes the downsampling operation with linear interpolation and l is the pyramid level and $I _ { i } ^ { 0 } , \mathbf { d } _ { i } ^ { 0 } , \Sigma _ { d i } ^ { 0 }$ are the full resolution image, depth, and covariance respectively. In Photo-SLAM [11], the authors utilize a sharp downsampling factor s of 0.5 and a 2-level pyramid. In contrast, we employ a smoother downsampling factor $s = 0 . 8$ similar to Instant-NGP [18] and a 3-level pyramid.

In each pyramid level, Gaussians are initialized by unprojection as follows: The points are sampled randomly from the most recent keyframe by using a downsampling factor Î¸. The sampled points are then unprojected according to depth maps. To account for the noise in depth maps, regions with high covariance are masked out to make the Gaussian initialization more robust to noise. Eq. (8) describes a mask for a given normalized depth covariance.

$$
M = \{ ( i , j ) \mid \sigma _ { i j } < 0 . 2 \}\tag{8}
$$

where M represents the binary mask matrix and i and j represent pixel location. The mask is created by normalizing the covariance Î£ between 0 and 1 and identifying the pixel values below 0.2 normalized covariance Ï. The mask is then smoothed using thresholding operation as described in Eq. (8) with a maximum filter followed by a majority filter. An example of a mask for a given covariance is shown in figure Fig. 3.

<!-- image-->  
Figure 3. An example of normalized covariance(left) and corresponding mask(right). The mask is created by thresholding normalized covariance with a maximum filter and smoothing with a majority filter. The white region on the mask is left out and not used during Gaussian initialization.

The map optimization is performed on a sliding window in a coarse-to-fine fashion. We maintain the last N keyframes within the sliding window to meet the real-time requirements. As the number of iterations increases, we switch to training with higher resolutions in the image pyramid. At the beginning of optimization at each level l, Gaussians are unprotected according to its depth map $\mathbf { d } _ { i } ^ { l } .$ . We render the Gaussians from keyframesâ viewpoints in the sliding window, and the loss function is calculated based on the rendered image and depth. Camera images and dense depth maps are utilized as ground truth in mapping supervision.

We employ a loss function that combines weighted depth loss $L _ { \mathrm { d e p t h } }$ and color loss $L _ { \mathrm { c o l o r } }$ which are defined as below

$$
L _ { \mathrm { d e p t h } } = \left\| D - \hat { D } \right\| _ { \Sigma _ { d } ^ { - 1 } } ^ { 1 } , L _ { \mathrm { c o l o r } } = \left\| C - \hat { C } \right\| ^ { 1 }\tag{9}
$$

where D and C are the ground truth depth and image, respectively, and DË and $\hat { C }$ are the rendered depth and image according to Eq. (6). The depth loss $L _ { \mathrm { d e p t h } }$ is weighted by the inverse covariance to ensure that the pixels with high uncertainty are weighted less. The combined loss is given by $L ~ = ~ \alpha L _ { \mathrm { c o l o r } } + ( 1 - \alpha ) L _ { \mathrm { d e p t h } }$ We set $\alpha = 0 . 5$ throughout all of our experiments. The loss is then backpropagated through a differentiable rendering pipeline where the position, opacity, covariance, maps, and color of the Gaussians are optimized.

Post Processing We refine the mapping results by optimizing the map for several iterations following the conventions established in MonoGS [16], GlORIE-SLAM [45] and Splat-SLAM [27]. For this purpose, we randomly select single frames and optimize the map with the same loss function used in the mapping. We perform the same number of iterations in MonoGS [16] and Splat-SLAM [27] for fairness.

## 3.3. Training Strategy

A subtle yet crucial point regarding our training strategy is that dense depth maps may be noisy; however, they are unlikely to disrupt depth order. In other words, having a position learning rate such that Gaussians switch positions during training is redundant and hinders optimization convergence. This effect is illustrated in Fig. 4. It should be noted that this is never the case for standard Gaussian Splatting training where the method typically starts with a sparse SfM point cloud. However, since Gaussians are initialized from a dense depth map, they are quite close to each other.

As illustrated in Fig. 4, case A) high learning rates cause the optimization to bounce Gaussians around the desired position. Conversely, the polar opposite in C) also hinders the convergence. Since setting a perfect learning rate for each iteration is neither feasible nor practical, we choose a learning rate that decays during training according to Eq. (10) to reduce this TV static noise during training. We initialize the learning rate to cover the full range needed to detail the model from coarse to fine while allowing for gradual decay.

$$
\mathbf { \boldsymbol { \mathbf { \mathit { I r } } } } ( t ) = \exp ( ( 1 - t ) \ln ( \mathbf { \boldsymbol { \mathbf { \mathit { I r } } } } _ { i } ) + t \ln ( \mathbf { \boldsymbol { \mathbf { \mathit { I r } } } } _ { f } ) )\tag{10}
$$

where $t = n / \tau$ is the iteration number n over decay constant $\tau ,$ and $\operatorname { l r } _ { i } , \ \operatorname { l r } _ { f }$ are the initial and final learning rate, respectively. The impact of learning rate and its decay in training performance are examined in Sec. 4.

<!-- image-->  
Figure 4. Three hypothetical cases to encounter in training. Dashed lines pass through ground truth Gaussian positions from the camera center. The faded Gaussians represent their previous positions. Red lines are the position update steps along the gradient direction. In A), a large position update causes the order of Gaussians to change, creating TV-static-like noise in training. In B), multiple iterations are needed to move Gaussians to the correct place because of small position updates. C) represents the ideal case where position update is exactly the position error.

We densify Gaussians in high loss gradient regions at every 150 iterations. Densification is achieved by cloning small Gaussians and by splitting large ones. The occluded Gaussians are also pruned at the end of each sliding window optimization to ensure that only the necessary Gaussians for accurate reconstruction are retained.

## 4. Experiments

We evaluate our system on various synthetic and real-world datasets. The ablation studies and hyperparameter analyses are also demonstrated to justify our design choices.

## 4.1. Experimental Setup

Datasets We evaluate the system in Replica [29], TUM RGB-D [30], ScanNet [6], and EuRoC MAV [3] datasets. Replica is a dataset of synthetic indoor scenes. The TUM RGB-D dataset consists of sequences that are recorded in small indoor office environments. The ScanNet dataset consists of 6 sequences of real-world indoor environments. The EuRoC is a dataset collected on board a Micro Aerial Vehicle (MAV) containing stereo images of relatively large-scale indoor environments. All datasets are evaluated without clipping except EuRoC. We clip from the start of the sequences to skip typical pauses at the beginning. We run all sequences 3 times and report the average results to mitigate the effect of the non-deterministic nature of multi-processing.

Metrics Following the view synthesis SLAM literature convention, we evaluate our system using PSNR, SSIM, and LPIPS [46]. We also provide depth L1[cm] metric compared to the ground truth depth in the Replica dataset. The evaluation is performed after post-processing every 5 frames in sequences skipping the keyframes used for mapping. This approach aligns with the evaluation methods used in MonoGS [16] and Splat-SLAM [27].

Implementation Details Our system runs on a PC with a 3.6GHz AMD Ryzen Threadripper PRO 5975WX and an NVIDIA RTX 4090 GPU. In all our experiments, we set $l = 0 . 8 , \theta = 1 2 8 , \alpha = 0 . 5 , \mathrm { l r } _ { i } = 1 . 6 \times 1 0 ^ { - 4 }$ , ${ \bf l r } _ { f } ~ = ~ 1 . 6 ~ \times ~ 1 0 ^ { - 6 } , ~ \tau ~ = ~ 3 0 0 0$ for hyperparameters in mapping. We set $\beta = 2 0 0 0$ for the EuRoC [3] and Replica [29] datasets and $\beta = 2 6 0 0 0$ for the TUM RGB-D [30] and the ScanNet [6] datasets. These values are consistent with those used in MonoGS [43] and Splat-SLAM [27]. For tracking, pre-trained GRU weights from DROID-SLAM [34] are utilized. We set the mean optical flow threshold for keyframe selection to 4.0 pixels, and the local dense bundle adjustment window to 16. Optimizations in the tracking module are performed in LieTorch [35] framework. The mapping process accepts only the latest keyframe created after finishing its optimization step if the latest keyframe is not already in the sliding window.

Baselines We compare our system to state-of-the-art RGBonly Gaussian Splatting and NeRF SLAM algorithms, including MonoGS [16], Photo-SLAM [11], GlORIE-SLAM [45], and Splat-SLAM [27].

MonoGS [16] is the state-of-the-art representation-only SLAM algorithm that utilizes the Gaussian scene representation for tracking and mapping. Photo-SLAM, like GlORIE-SLAM [45], Splat-SLAM [27], and our system, features a decoupled design for tracking and mapping. One key difference is that Photo-SLAM lacks dense depth maps while mapping. GlORIE-SLAM and Splat-SLAM utilize monocular depth estimation [7] and the dense bundle adjustment layer. The most important difference between them is that GlORIE-SLAM [45] models the scene with NeRF [17] and Splat-SLAM [27] does so with 3D Gaussian Splatting [14].

## 4.2. Evaluation

We compare our system with state-of-the-art algorithms based on rendering quality, 3D reconstruction accuracy, and runtime performance.

Rendering and Reconstruction Accuracy We evaluate rendering and reconstruction accuracy for the Replica [29] in Tab. 1. Our algorithmâs performance is quite similar to Splat-SLAM [27] in Replica [29]. In Tab. 2, we compare head-to-head with GlORIE-SLAM [45] on the ScanNet [6], where we trail behind Splat-SLAM [27]. In Tab. 3, we rank just behind Splat-SLAM, outperforming other algorithms on the TUM RGB-D [30] dataset. However, we are superior in terms of on-the-fly map optimization to Splat-SLAM as shown in Tab. 6. We place the first in the the EuRoC [3] dataset demonstrating a significant margin over Photo-SLAM [11]. A qualitative comparison

is shown in Fig. 1. Our experiments reveal that sequences focusing on a centered object in an unbounded scene, such as TUM-RGBD f3/off, are particularly challenging.
<table><tr><td>Metrics</td><td>Mono- GS [16]</td><td>GIORIE- SLAM [45]</td><td>Photo- SLAM [11]</td><td>Splat - SLAM [27]</td><td>Ours</td></tr><tr><td>PSNRâ</td><td>31.22</td><td>31.04</td><td>33.30</td><td>36.45</td><td>36.21</td></tr><tr><td>SSIMâ</td><td>0.91</td><td>0.91</td><td>0.93</td><td>0.95</td><td>0.96</td></tr><tr><td>LPIPSâ</td><td>0.21</td><td>0.12</td><td>-</td><td>0.06</td><td>0.05</td></tr><tr><td>Depth L1â</td><td>-</td><td>-</td><td>-</td><td>2.41</td><td>4.34</td></tr></table>

Table 1. Rendering and Tracking Results on Replica [29] for RGB-Methods. The results are averaged over 8 scenes and each scene result is the average of 3 runs. We take the numbers from [27] except for ours. The best results are highlighted as first , second . Our method shows similar performance to Splat-SLAM [27] and outperforms all the other methods.

<table><tr><td>Method</td><td>Metric</td><td>0000</td><td>0059</td><td>0106</td><td>0169</td><td>0181</td><td>0207</td><td>Avg.</td></tr><tr><td rowspan="3">MonoGS [16]</td><td>PSNRâ</td><td>16.91</td><td>19.15</td><td>18.57</td><td>20.21</td><td>19.51</td><td>18.37</td><td>18.79</td></tr><tr><td>SSIM â</td><td>0.62</td><td>0.69</td><td>0.74</td><td>0.74</td><td>0.75</td><td>0.70</td><td>0.71</td></tr><tr><td>LPIPSâ</td><td>0.70</td><td>0.51</td><td>0.55</td><td>0.54</td><td>0.63</td><td>0.58</td><td>0.59</td></tr><tr><td>GlORIE-</td><td>PSNR</td><td>23.42</td><td>20.66</td><td>20.41</td><td>25.23</td><td>21.28</td><td>23.68</td><td>22.45</td></tr><tr><td rowspan="3">SLAM [45]</td><td>SSIM â</td><td>0.87</td><td>0.87</td><td>0.83</td><td>0.84</td><td>0.91</td><td>0.76</td><td>0.85</td></tr><tr><td>LPIPSâ</td><td>0.26</td><td>0.31</td><td>0.31</td><td>0.21</td><td>0.44</td><td>0.29</td><td>0.30</td></tr><tr><td>PSNR</td><td>28.68</td><td>27.69</td><td>27.70</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Splat- SLAM [27]</td><td>SSIMâ</td><td>0.83</td><td>0.87</td><td>0.86</td><td>31.14 0.87</td><td>31.15</td><td>30.49 0.84</td><td>29.48 0.85</td></tr><tr><td>LPIPS â</td><td>0.19</td><td>0.15</td><td>0.18</td><td>0.15</td><td>0.84 0.23</td><td>0.19</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.18</td></tr><tr><td rowspan="3">IG-SLAM (Ours)</td><td>PSNR</td><td>24.68</td><td>20.09</td><td>25.30</td><td>27.85</td><td>25.80</td><td>26.69</td><td>25.07</td></tr><tr><td>SSIMâ</td><td>0.74</td><td>0.68</td><td>0.83</td><td>0.82</td><td>0.83</td><td>0.78</td><td>0.78</td></tr><tr><td>LPIPS â</td><td>0.29</td><td>0.39</td><td>0.22</td><td>0.19</td><td>0.27</td><td>0.27</td><td>0.27</td></tr></table>

Table 2. Rendering Performance on ScanNet [6]. Each scene result is the average of 3 runs. We take the numbers from [27] except for ours. Our method shows competitive performance to the state-of-the-art methods exhibiting the second high visual quality results.

Runtime Analysis We assess real-time performance of our algorithm in Tab. 5. We benchmark the runtime on a 3.6GHz AMD Ryzen Threadripper PRO 5975WX and an NVIDIA GeForce RTX 4090 with 24 GB of memory. Our system operates at 9.94 fps, making it 8 times faster than Splat-SLAM [27] in a single-process implementation. Our method outperforms other algorithms without compromising visual quality. The reference multi-process implementation of our method achieves a frame rate of 16 fps. Our methodâs peak memory consumption and map size are comparable to existing methods.

## 4.3. Ablations

Post-processing, decay, and weighted depth loss are our system design choices. We present ablation studies to validate and support each of these design decisions.

Post Processing We show post processing ablation results in Tab. 6. PSNR and Depth L1 metrics are recalculated for every 500 post-processing iterations. Our method exhibits a relatively small visual quality degradation when post-processing is skipped (indicated as 0K in Tab. 6) whereas visual quality significantly drops with no postprocessing for Splat-SLAM [27]. Our system exhibits diminishing returns with increased post-processing iterations. We attribute the fast convergence of our map and the minimal reliance on post-processing to our training strategy.

<table><tr><td>Method</td><td>Metric</td><td>f1/desk</td><td>f2/xyz</td><td>f3/off</td><td>Avg.</td></tr><tr><td rowspan="3">Photo-SLAM [11]</td><td>PSNRâ</td><td>20.97</td><td>21.07</td><td>19.59</td><td>20.54</td></tr><tr><td>SSIM â</td><td>0.74</td><td>0.73</td><td>0.69</td><td>0.72</td></tr><tr><td>LPIPS â</td><td>0.23</td><td>0.17</td><td>0.24</td><td>0.21</td></tr><tr><td></td><td>PSNRâ</td><td>19.67</td><td>16.17</td><td>20.63</td><td>18.82</td></tr><tr><td rowspan="2">MonoGS [16]</td><td>SSIMâ</td><td>0.73</td><td>0.72</td><td>0.77</td><td>0.74</td></tr><tr><td>LPIPS â</td><td>0.33</td><td>0.31</td><td>0.34</td><td>0.33</td></tr><tr><td>GlORIE-</td><td>PSNRâ</td><td>20.26</td><td>25.62</td><td>21.21</td><td>22.36</td></tr><tr><td>SLAM [45]</td><td>SSIMâ</td><td>0.79</td><td>0.72</td><td>0.72</td><td>0.74</td></tr><tr><td></td><td>LPIPS â</td><td>0.31</td><td>0.09</td><td>0.32</td><td>0.24</td></tr><tr><td>Splat-</td><td>PSNRâ</td><td>25.61</td><td>29.53</td><td>26.05</td><td>27.06</td></tr><tr><td>SLAM [27]</td><td>SSIMâ</td><td>0.84</td><td>0.90</td><td>0.84</td><td>0.86</td></tr><tr><td></td><td>LPIPS â</td><td>0.18</td><td>0.08</td><td>0.20</td><td>0.15</td></tr><tr><td rowspan="3">IG-SLAM (Ours)</td><td>PSNRâ</td><td>24.45</td><td>26.35</td><td>25.27</td><td>25.36</td></tr><tr><td>SSIM â</td><td>0.80</td><td>0.85</td><td>0.83</td><td>0.83</td></tr><tr><td>LPIPS â</td><td>0.20</td><td>0.10</td><td>0.17</td><td>0.16</td></tr></table>

Table 3. Rendering Performance on TUM-RGBD [30]. Each scene result is the average of 3 runs. We take the numbers from [27] except for ours. Our method demonstrates similar performance to Splat-SLAM [27] in challenging indoor environments showing a clear performance margin to the other methods.

<table><tr><td>Method</td><td>Metric</td><td>MH-01</td><td>MH-02</td><td>V1-01</td><td>V2-01</td><td>Avg.</td></tr><tr><td></td><td>PSNRâ</td><td>13.95</td><td>14.20</td><td>17.07</td><td>15.68</td><td>15.23</td></tr><tr><td>Photo-SLAM [11]</td><td>SSIM â</td><td>0.42</td><td>0.43</td><td>0.62</td><td>0.62</td><td>0.52</td></tr><tr><td></td><td>LPIPS â</td><td>0.37</td><td>0.36</td><td>0.27</td><td>0.32</td><td>0.33</td></tr><tr><td>IG-SLAM</td><td>PSNRâ</td><td>22.33</td><td>22.31</td><td>20.55</td><td>24.59</td><td>22.44</td></tr><tr><td>(Ours)</td><td>SSIMâ</td><td>0.78</td><td>0.77</td><td>0.79</td><td>0.85</td><td>0.80</td></tr><tr><td></td><td>LPIPS â</td><td>0.22</td><td>0.23</td><td>0.29</td><td>0.18</td><td>0.23</td></tr></table>

Table 4. Rendering Performance on EuRoC [3]. Each scene result is the average of 3 runs. We take the numbers for Photo-SLAM [11] from their work. We successfully show the scalability of our system. Photorealistic 3D reconstruction comparison of large indoor environment EuRoC [3] MH-01 is shown in Fig. 1

<table><tr><td></td><td>GO-SLAM [47]</td><td>GIORIE-SLAM [45]</td><td>MonoGS [16]</td><td>Splat-SLAM [27]</td><td>Ours</td></tr><tr><td>GPU Usage [GiB]</td><td>18.50</td><td>15.22</td><td>14.62</td><td>17.57</td><td>16.20</td></tr><tr><td>Map Size [MB]</td><td></td><td>114.0</td><td>6.8</td><td>6.5</td><td>14.8</td></tr><tr><td>Avg. FPS</td><td>8.36</td><td>0.23</td><td>0.32</td><td>1.24</td><td>9.94</td></tr></table>

Table 5. Memory and Running Time Evaluation on Replica [29] room0. We measure the runtime statistics on the single process implementation of our method. We take the numbers from [27] except for ours. Our peak memory usage and map size are comparable to existing works. Our method achieves to exhibit state-ofthe-art 3D reconstruction in higher frame rates compared to other methods.

<table><tr><td>Nbr of Final Iterations Î²</td><td>Metric</td><td>OK</td><td>0.5K</td><td>1K</td><td>2K</td></tr><tr><td>Splat-</td><td>PSNR â</td><td>30.50</td><td>39.87</td><td>40.59</td><td>41.20</td></tr><tr><td>SLAM [27]</td><td>Depth L1 â</td><td>6.55</td><td>2.37</td><td>2.34</td><td>2.40</td></tr><tr><td>Ours</td><td>PSNR â</td><td>38.30</td><td>40.92</td><td>41.53</td><td>41.68</td></tr><tr><td></td><td>Depth L1 â</td><td>2.63</td><td>2.18</td><td>2.17</td><td>2.30</td></tr></table>

Table 6. Post-processing iterations ablation on Replica [29] office0. The numbers for Splat-SLAM [27] are taken from their work. Due to the fast convergence of mapping during tracking, we do not heavily rely on post-processing. The reconstruction benefits only a little from post-processing.

Decay We demonstrate learning rate decay ablation in Tab. 7. We compare 3 learning rates without decay with decaying learning rates. The selected 3 learning rates are $\mathrm { l r } _ { f } ) = 1 . 6 \times 1 0 ^ { - 6 }$ for lower bound, $\operatorname { l r } _ { i } ) = 1 . 6 \times 1 0 ^ { - 4 }$ for upper bound, and the mean learning rate value $5 \times 1 0 ^ { - 5 }$ calculated according to Eq. (10). We conduct this experiment with and without post-processing. As seen in no postprocessing experiment in Tab. 7, learning with decay greatly enhances the visual quality compared to other non-decaying learning rate setups. Qualitative results are shown in Fig. 5. As observed, the fine details are not captured with nondecaying learning rates. Moreover, a post-processing step completely shadows the convergence problems of constant learning rate as seen in the experiment with post-processing in Tab. 7.

<table><tr><td>Metric</td><td>Learning Rate</td><td>1.6 Ã 10â6</td><td>5 Ã 10â5</td><td></td><td>1.6 Ã 10â4 1.6 Ã 10â4 w/ decay</td></tr><tr><td colspan="6">w/o Post Processing</td></tr><tr><td>PSNR â</td><td></td><td>31.92</td><td>35.84</td><td>34.71</td><td>38.30</td></tr><tr><td>Depth L1 â</td><td></td><td>5.37</td><td>2.71</td><td>2.76</td><td>2.63</td></tr><tr><td colspan="6">w/ Post Processing</td></tr><tr><td>PSNR â</td><td></td><td>39.71</td><td>39.91</td><td>40.85</td><td>41.68</td></tr><tr><td>Depth L1 â</td><td></td><td>2.73</td><td>2.17</td><td>2.20</td><td>2.30</td></tr></table>

Table 7. Learning Rate Hyperparameter Search on Replica [29] office0. Our system benefits greatly from a slow learning rate combined with decay. In the presence of reliable depth maps, a high learning rate contributes to TV-static noise and slows down map convergence.

Depth Loss The weighted depth loss ablation results are shown in Tab. 8. The weighted depth loss that is given in Eq. (9) is compared to the scenarios with no depth loss in the overall loss function (Î± = 1) and with raw depth values without weighting them by depth covariance. Postprocessing is disabled to ensure the results are not obscured.

<table><tr><td>Metric</td><td>Weighted</td><td>No Depth</td><td>Raw Depth</td></tr><tr><td>PSNRâ</td><td>31.91</td><td>31.56</td><td>30.81</td></tr><tr><td>Depth L1 â</td><td>6.33</td><td>13.16</td><td>6.39</td></tr></table>

Table 8. Weighted Depth Loss Ablation on Replica [29] office2. Weighted depth loss enables better reconstruction without decreasing visual quality.

The weighted loss is superior to other choices as observed in Tab. 8. A pure color loss performs well in terms of visual quality but deteriorates reconstruction quality. Using raw depth values in the loss function performs worse than the weighted loss regarding visual quality. Therefore, weighting the depth prevents visual quality from decreasing due to high uncertainty regions while keeping the reconstruction quality up by supervising depth. We speculate visual quality differences are not dramatic because our system initializes Gaussians according to depth maps regardless of the loss function. Therefore, initialized Gaussians are already in the vicinity of the corresponding depth value.

<!-- image-->

<!-- image-->  
Figure 5. Qualitative results for learning rate decay ablation study. The four cases studied in Tab. 7 are shown in the figure. The results are given as constant learning rates of $1 . 6 \times 1 0 ^ { - 4 }$ at top-left, $5 \times 1 0 ^ { - 5 }$ at top-right, $1 . 6 \times 1 0 ^ { - 6 }$ at bottom-left and the decaying $1 . 6 \times 1 0 ^ { - 4 }$ learning rate at bottom-left as reference.

## 5. Limitations

The dense bundle adjustment is not feasible in full resolution. Therefore, dense depth maps are optimized at a lower resolution and upsampled back to the original resolution. We observe that this upsampling operation results in blurry edges. Therefore, utilizing upsampled dense depth maps to supervise the system results in poor performance at locations where sharp changes in depth occur.

## 6. Conclusion

We showed that the depth supervision from a robust dense-SLAM method greatly enhances 3D reconstruction performance. Additionally, utilizing depth uncertainty as a mask for Gaussian initialization and as weights for depth loss aids the mapping process. We also highlighted the nuance between sparse and dense Gaussian initialization and its implications on mapping optimization. Our experiments demonstrated that dense SLAM-based 3D reconstruction can provide both state-of-the-art visual quality and a high frame rate even in relatively large scenes.

## References

[1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855â5864, 2021. 2

[2] Michael Bloesch, Jan Czarnowski, Ronald Clark, Stefan Leutenegger, and Andrew J. Davison. Codeslam - learning a compact, optimisable representation for dense visual SLAM. CoRR, abs/1804.00874, 2018. 2

[3] Michael Burri, Janosch Nikolic, Pascal Gohl, Thomas Schneider, Joern Rehder, Sammy Omari, Markus W Achtelik, and Roland Siegwart. The euroc micro aerial vehicle datasets. The International Journal of Robotics Research, 35 (10):1157â1163, 2016. 1, 6, 7, 2

[4] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. arXiv preprint arXiv:2406.06521, 2024. 2

[5] Chi-Ming Chung, Yang-Che Tseng, Ya-Ching Hsu, Xiang-Qian Shi, Yun-Hung Hua, Jia-Fong Yeh, Wen-Chin Chen, Yi-Ting Chen, and Winston H Hsu. Orbeez-slam: A realtime monocular visual slam with orb features and nerfrealized mapping. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9400â9406. IEEE, 2023. 1, 2

[6] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas A. Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. CoRR, abs/1702.04405, 2017. 6, 7, 1

[7] Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multitask mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10786â10796, 2021. 1, 6

[8] Peter Hedman, Pratul P. Srinivasan, Ben Mildenhall, Jonathan T. Barron, and Paul Debevec. Baking neural radiance fields for real-time view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 5875â5884, 2021. 2

[9] Tao Hu, Shu Liu, Yilun Chen, Tiancheng Shen, and Jiaya Jia. Efficientnerf efficient neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12902â12911, 2022. 2

[10] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, pages 1â11, 2024. 2

[11] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung. Photo-slam: Real-time simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21584â21593, 2024. 1, 2, 3, 4, 5, 6, 7

[12] Mohammad Mahdi Johari, Camilla Carta, and FrancÂ¸ois Fleuret. Eslam: Efficient dense slam system based on hybrid representation of signed distance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17408â17419, 2023. 1

[13] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21357â21366, 2024. 3

[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 1, 2, 3, 4, 6

[15] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. Advances in Neural Information Processing Systems, 33:15651â15663, 2020. 2

[16] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18039â18048, 2024. 1, 3, 4, 5, 6, 7

[17] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. CoRR, abs/2003.08934, 2020. 1, 2, 6

[18] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022. 2, 4, 5

[19] Raul Mur-Artal and Juan D Tardos. Orb-slam2: An open- Â´ source slam system for monocular, stereo, and rgb-d cameras. IEEE transactions on robotics, 33(5):1255â1262, 2017. 1, 2, 3

[20] Richard A. Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J. Davison, Pushmeet Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon. Kinectfusion: Real-time dense surface mapping and tracking. In 2011 10th IEEE International Symposium on Mixed and Augmented Reality, pages 127â136, 2011. 2

[21] Richard A. Newcombe, Steven J. Lovegrove, and Andrew J. Davison. Dtam: Dense tracking and mapping in real-time. In 2011 International Conference on Computer Vision, pages 2320â2327, 2011. 2

[22] Michael Oechsle, Songyou Peng, and Andreas Geiger. Unisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5589â5599, 2021. 2

[23] Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang, Jingdong Wang, and Kun Zhou. Rtg-slam: Real-time 3d reconstruction at scale using gaussian splatting. In ACM SIG-GRAPH 2024 Conference Papers, pages 1â11, 2024. 3

[24] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. Kilonerf: Speeding up neural radiance fields with

thousands of tiny mlps. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14335â 14345, 2021. 1

[25] Konstantinos Rematas and Vittorio Ferrari. Neural voxel renderer: Learning an accurate and controllable rendering tool. In CVPR, 2020. 2

[26] Antoni Rosinol, John Leonard, and Luca Carlone. Nerfslam: Real-time dense monocular slam with neural radiance fields. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3437â3444. IEEE, 2023. 1, 2, 4

[27] Erik Sandstrom, Keisuke Tateno, Michael Oechsle, Michael Â¨ Niemeyer, Luc Van Gool, Martin R Oswald, and Federico Tombari. Splat-slam: Globally optimized rgb-only slam with 3d gaussians. arXiv preprint arXiv:2405.16544, 2024. 2, 3, 5, 6, 7, 8

[28] Cameron Smith, David Charatan, Ayush Tewari, and Vincent Sitzmann. Flowmap: High-quality camera poses, intrinsics, and depth via gradient descent. arXiv preprint arXiv:2404.15259, 2024. 2

[29] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Yuheng Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard A. Newcombe. The replica dataset: A digital replica of indoor spaces. CoRR, abs/1906.05797, 2019. 6, 7, 8, 1

[30] Jurgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Â¨ Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 573â580. IEEE, 2012. 6, 7, 1

[31] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davison. imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6229â6238, 2021. 1, 2

[32] Chengzhou Tang and Ping Tan. Ba-net: Dense bundle adjustment network. CoRR, abs/1806.04807, 2018. 2

[33] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â 28, 2020, Proceedings, Part II 16, pages 402â419. Springer, 2020. 2

[34] Zachary Teed and Jia Deng. DROID-SLAM: deep visual SLAM for monocular, stereo, and RGB-D cameras. CoRR, abs/2108.10869, 2021. 1, 2, 3, 6

[35] Zachary Teed and Jia Deng. Tangent space backpropagation for 3d transformation groups. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10338â10347, 2021. 6

[36] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Coslam: Joint coordinate and sparse parametric encodings for neural real-time slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13293â13302, 2023. 1

[37] Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang, Minye Wu, Jingyi Yu, and Lan Xu. Fourier plenoctrees for dynamic radiance field rendering in real-time. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13524â13534, 2022. 2

[38] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021. 2

[39] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19595â19604, 2024. 1, 3

[40] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. Advances in Neural Information Processing Systems, 34:4805â4815, 2021. 2

[41] Lin Yen-Chen, Pete Florence, Jonathan T Barron, Alberto Rodriguez, Phillip Isola, and Tsung-Yi Lin. inerf: Inverting neural radiance fields for pose estimation. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1323â1330. IEEE, 2021. 2

[42] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5752â 5761, 2021. 2

[43] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Oswald. Gaussian-slam: Photo-realistic dense slam with gaussian splatting. arXiv preprint arXiv:2312.10070, 2023. 1, 3, 6

[44] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. Rade-gs: Rasterizing depth in gaussian splatting. arXiv preprint arXiv:2406.01467, 2024. 2

[45] Ganlin Zhang, Erik Sandstrom, Youmin Zhang, Manthan Pa- Â¨ tel, Luc Van Gool, and Martin R Oswald. Glorie-slam: Globally optimized rgb-only implicit encoding point cloud slam. arXiv preprint arXiv:2403.19549, 2024. 1, 2, 5, 6, 7

[46] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 6

[47] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo Poggi. Go-slam: Global optimization for consistent 3d instant reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3727â3737, 2023. 1, 4, 7

[48] Heng Zhou, Zhetao Guo, Shuhong Liu, Lechen Zhang, Qihao Wang, Yuxiang Ren, and Mingrui Li. Mod-slam: Monocular dense mapping for unbounded 3d scene reconstruction. arXiv preprint arXiv:2402.03762, 2024. 1, 2

[49] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12786â12796, 2022. 1, 2

[50] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui, Martin R Oswald, Andreas Geiger, and Marc Pollefeys. Nicer-slam: Neural implicit scene encoding for rgb slam. In 2024 International Conference on 3D Vision (3DV), pages 42â52. IEEE, 2024. 1, 2

[51] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Ewa volume splatting. In Proceedings Visualization, 2001. VISâ01., pages 29â538. IEEE, 2001. 4

# IG-SLAM: Instant Gaussian SLAM

Supplementary Material

IG-SLAM is a dense SLAM system capable of photorealistic 3D reconstruction, while simultaneously running at high frame rates. In this supplementary material, we provide additional results.

## 7. Method

We describe additional details about our method.

## 7.1. Covariance Mask

Assume the covariance for a depth map is given by Î£, we normalize covariance between [0,1] by Eq. (11)

$$
\tilde { \Sigma } ( u , v ) = \frac { \Sigma ( u , v ) - \operatorname* { m i n } \left( \Sigma ( u , v ) \right) } { \operatorname* { m a x } \left( \Sigma ( u , v ) \right) - \operatorname* { m i n } \left( \Sigma ( u , v ) \right) }\tag{11}
$$

where (u, v) are the pixel coordinates. A Maximum filter with a kernel size of 32 is applied to normalized covariance. Pixels with normalized covariance less than 0.2 are selected. Additionally, a majority filter with a kernel size of 32 is applied to obtain smooth valid regions in the mask.

## 7.2. Pruning and Densification

We follow the same procedure for pruning and identification in MonoGS [16] n. Pruning is based on occlusion-aware visibility: if new Gaussians initialized in the last keyframes are not visible from this keyframe at the end of the optimization, they are removed. Additionally, for every 150 mapping iterations, Gaussians with opacity lower than 0.1 are removed. Densification is performed by splitting large Gaussians and cloning small ones in regions with high loss gradients, also every 150 mapping iterations.

## 8. Additional Results

We provide additional tracking and mapping results.

## 9. Tracking

We do not improve over GO-SLAM [47] in terms of tracking performance, as it is outside the scope of our work. However, we include the tracking results of Replica [29], TUM-RGB-D [30], and ScanNet [6] in Tab. 9, Tab. 10, and Tab. 11 for reference.

<table><tr><td>Metric</td><td>R-O</td><td>R-1</td><td>R-2</td><td>0-0</td><td>0-1</td><td>0-2</td><td>0-3</td><td>0-4</td></tr><tr><td>ATE(cm)</td><td>0.45</td><td>0.39</td><td>0.31</td><td>0.33</td><td>0.50</td><td>0.39</td><td>0.47</td><td>0.68</td></tr></table>

Table 9. Tracking Accuracy ATE RMSE [cm] â on Replica [29]. Each scene result is the average of 3 runs.

<table><tr><td>Metric</td><td>f1/desk</td><td>f2/xyz</td><td>f3/off</td></tr><tr><td>ATE(cm)</td><td>2.73</td><td>0.35</td><td>2.08</td></tr></table>

Table 10. Tracking Accuracy ATE RMSE [cm] â on TUM-RGBD [30]. Each scene result is the average of 3 runs.

<table><tr><td>Metric</td><td>0000</td><td>0059</td><td>0106</td><td>0169</td><td>0181</td><td>0207</td></tr><tr><td>ATE(cm)</td><td>6.16</td><td>71.46</td><td>7.38</td><td>8.46</td><td>8.60</td><td>9.55</td></tr></table>

Table 11. Tracking Accuracy ATE RMSE [cm] â on Scan-Net [6]. Each scene result is the average of 3 runs.

## 9.1. Mapping

The results of each scene of the Replica [29] are given in Tab. 12. Full evaluations on EuRoC [3] Machine Hall and Vicon Room are given in Tab. 13 and Tab. 14. Moreover, additional qualitative results of EuRoC [3] are exhibited in Fig. 6

<table><tr><td>Metric</td><td>R-0</td><td>R-1</td><td>R-2</td><td>O-0</td><td>0-1</td><td>0-2</td><td>0-3</td><td>0-4</td></tr><tr><td>PSNRâ</td><td>32.33</td><td>34.64</td><td>35.29</td><td>41.68</td><td>41.30</td><td>34.68</td><td>34.92</td><td>34.80</td></tr><tr><td>SSIM â</td><td>0.93</td><td>0.95</td><td>0.96</td><td>0.98</td><td>0.98</td><td>0.95</td><td>0.96</td><td>0.96</td></tr><tr><td>LPIPSâ</td><td>0.07</td><td>0.06</td><td>0.05</td><td>0.02</td><td>0.03</td><td>0.06</td><td>0.05</td><td>0.07</td></tr><tr><td>Depth L1â</td><td>4.79</td><td>3.04</td><td>4.15</td><td>2.23</td><td>1.94</td><td>6.40</td><td>7.67</td><td>4.45</td></tr></table>

Table 12. Full evaluation on Replica [29]. Each scene result is the average of 3 runs.

<table><tr><td>Metric</td><td>MH-01</td><td>MH-02</td><td>MH-03</td><td>MH-04</td><td>MH-05</td></tr><tr><td>PSNRâ</td><td>22.33</td><td>22.31</td><td>20.78</td><td>23.62</td><td>19.85</td></tr><tr><td>SSIM â</td><td>0.78</td><td>0.77</td><td>0.71</td><td>0.82</td><td>0.70</td></tr><tr><td>LPIPSâ</td><td>0.22</td><td>0.23</td><td>0.28</td><td>0.19</td><td>0.35</td></tr></table>

Table 13. Full evaluation on EuRoC [3] Machine Hall. Each scene result is the average of 3 runs.

<table><tr><td>Metric</td><td>V1-01</td><td>V1-02</td><td>V1-03</td><td>V2-01</td><td>V2-02</td><td>V2-03</td></tr><tr><td>PSNRâ</td><td>20.55</td><td>22.86</td><td>20.11</td><td>24.59</td><td>23.70</td><td>21.62</td></tr><tr><td>SSIM â</td><td>0.79</td><td>0.84</td><td>0.74</td><td>0.85</td><td>0.83</td><td>0.74</td></tr><tr><td>LPIPSâ</td><td>0.29</td><td>0.26</td><td>0.42</td><td>0.18</td><td>0.23</td><td>0.41</td></tr></table>

Table 14. Full evaluation on EuRoC [3] Vicon Room Each scene result is the average of 3 runs.

<!-- image-->  
Figure 6. Qualitative results of IG-SLAM on EuRoC [3]. The results in the top row, middle row, and bottom row are from MH-02, MH-03, V1-01 respectively.