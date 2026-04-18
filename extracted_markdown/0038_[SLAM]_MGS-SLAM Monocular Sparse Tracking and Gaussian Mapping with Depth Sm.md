# MGS-SLAM: Monocular Sparse Tracking and Gaussian Mapping with Depth Smooth Regularization

Pengcheng Zhu, Yaoming Zhuang, Baoquan Chen, Li Li, Chengdong Wu, and Zhanlin Liu

AbstractâThis letter introduces a novel framework for dense Visual Simultaneous Localization and Mapping (VSLAM) based on Gaussian Splatting. Recently, SLAM based on Gaussian Splatting has shown promising results. However, in monocular scenarios, the Gaussian maps reconstructed lack geometric accuracy and exhibit weaker tracking capability. To address these limitations, we jointly optimize sparse visual odometry tracking and 3D Gaussian Splatting scene representation for the first time. We obtain depth maps on visual odometry keyframe windows using a fast Multi-View Stereo (MVS) network for the geometric supervision of Gaussian maps. Furthermore, we propose a depth smooth loss and Sparse-Dense Adjustment Ring (SDAR) to reduce the negative effect of estimated depth maps and preserve the consistency in scale between the visual odometry and Gaussian maps. We have evaluated our system across various synthetic and real-world datasets. The accuracy of our pose estimation surpasses existing methods and achieves state-of-theart. Additionally, it outperforms previous monocular methods in terms of novel view synthesis and geometric reconstruction fidelities.

Index TermsâSLAM; Mapping; 3D Gaussian Splatting

## I. INTRODUCTION

IMULTANEOUS Localization and Mapping (SLAM) is a key technology in robotics and autonomous driving. It aims to solve the problem of how robots determine their location and reconstruct maps of the environment in unknown scenes. The development of SLAM technology has gone through multiple stages, starting with the initial filter-based method [1], advancing to graph optimization-based method [2], and more recently, integrating deep learning. This integration has significantly improved the accuracy and robustness of SLAM systems. With the rapid development of deep learning technology, a new approach to SLAM technology has emerged, utilizing differentiable rendering. The initial applications of differentiable rendering-based SLAM utilized Neural Radiance Fields (NeRF) as their foundational construction method. NeRF, as detailed in [3], employs neural networks to represent 3D scenes, enabling the synthesis of high-quality images and the recovery of dense geometric structures from multiple views. NeRF-based SLAM systems preserve detailed scene information during mapping, which enhances support for subsequent navigation and path planning. However, NeRFâs approach requires multiple forward predictions for each pixel during image rendering, leading to significant computational redundancy. Consequently, this inefficiency prevents NeRFbased SLAM from operating in real-time, thus limiting its practicality for immediate downstream tasks.

<!-- image-->  
Fig. 1. Map reconstruction process by the proposed system. The prior depth map is estimated from the keyframes of sparse visual odometry and optimized by a sparse point cloud map, and the optimized depth map is used to construct a dense Gaussian map.

Recently, a novel scene representation framework called 3D Gaussian Splatting [4] has demonstrated superior performance compared to NeRF. It features a more concise scene representation method and real-time rendering capability. This method not only delivers an accurate description of the scene but also offers a differentiable approach for optimizing the scene and camera poses. This opens up a new research direction for differentiable rendering-based SLAM. However, current Gaussian Splatting-based SLAM systems rely on the depth maps input to achieve precise geometric reconstruction, which constrains the scope of their application.

This letter presents MGS-SLAM, a novel monocular Gaussian Splatting-based SLAM system. This work introduces several groundbreaking advancements in the field of SLAM, which include integrating Gaussian Splatting techniques with sparse visual odometry, employing a pre-trained Multi-View Stereo (MVS) depth estimation network, pioneering a geometric smooth depth loss, and developing the SDAR strategy to ensure scale consistency. Together, these innovations significantly improve the accuracy and functionality of SLAM systems that rely solely on RGB image input. Fig. 1 illustrates the map construction process: initially, sparse visual odometry constructs the sparse maps; subsequently, the MVS depth estimation network generates priori depth maps; these depth maps, along with the sparse point maps, are then refined through depth optimization in the SDAR; and finally, the Gaussian map is constructed using the optimized depth maps and depth smooth regularization loss.

<!-- image-->  
Fig. 2. System pipeline. The system inputs an RGB stream and operates frontend and backend processes in parallel. In the frontend, sparse visual odometry extracts patch features from images to estimate poses. These estimated poses and images are inputs to a pre-trained Multi-View Stereo (MVS) network, which estimates priori depth maps. In the backend, the estimated priori depth maps and images, coupled with poses from the frontend, are utilized as supervisory information to construct a Gaussian map. The frontend and backend maintain scale consistency through the SDAR strategy.

The key contributions of the proposed system are summarized as follows:

â¢ Introducing the first SLAM system that jointly optimizes sparse visual odometry poses and 3D Gaussian Splatting to achieve the accurate geometric reconstruction of Gaussian maps and pose tracking.

â¢ Developing a pre-trained Multi-View Stereo (MVS) depth estimation network that utilizes sparse odometry keyframes and their poses to estimate prior depth maps, thus providing crucial geometric constraints for Gaussian map reconstruction with only RGB image input.

â¢ Proposing a geometric depth smooth loss method to minimize the adverse impacts of inaccuracies in estimated prior depth maps on the Gaussian map and guide its alignment to correct geometric positions.

â¢ Proposing a Sparse-Dense Adjustment Ring (SDAR) strategy to unify the scale consistency of sparse visual odometry and dense Gaussian map.

## II. RELATED WORKS

Monocular Dense SLAM. Over the past few decades, monocular dense SLAM technology has seen significant advancements. DTAM [5] pioneered one of the earliest realtime dense SLAM systems by performing parallel depth computations on GPU. To balance computational costs and accuracy, there are also semi-dense methods such as [6], but these methods struggle to capture areas with poor texture. In the era of deep learning, DROID-SLAM [7] utilizes optical flow networks to establish dense pixel correspondences and achieve precise pose estimation. Another study [8], combines a real-time VO system with a Multi-View Stereo (MVS) network for parallel tracking and dense depth estimation, and then the Truncated Signed Distance Function (TSDF) is used to fuse depth maps and extract mesh. Codemapping [9] and Rosinol et al. [10] incorporate sparse point cloud correction and volumetric fusion strategy on the estimated depth map to mitigate the impact of errors in the estimated depth map. We have also adopted a strategy for correcting the estimated depth map, but the difference is that we use a linear variance correction as depth optimization, which is less computations.

Differentiable Rendering SLAM. With the emergence of Neural Radiance Fields (NeRF) in 2020, numerous NeRFbased SLAM works have been proposed. iMAP [11] represented the pioneering work in NeRF-based SLAM, utilizing a dual-threading mode to track camera poses and execute mapping simultaneously. NICE-SLAM [12] introduced feature grids based on iMAP, enabling NeRF-based SLAM to represent larger scenes. Subsequent works such as GO-SLAM [13] and Loopy-SLAM [14] incorporated global bundle adjustment (BA) and loop closure correction, further enhancing pose estimation accuracy and mapping performance. PLGSLAM [15] proposes a progressive scene representation method to improve reconstruction and localization accuracy in large scenarios. Recently, 3D Gaussian Splatting has shown superior performance in 3D scene representation. It has fast rendering capability and is more suitable for online systems like SLAM. SplaTAM [16] and GS-SLAM [17] combine 3D Gaussian Splatting with SLAM, leveraging the realistic scene reconstruction ability of 3D Gaussian Splatting to surpass NeRF-based SLAM methods in rendering quality. Compact-SLAM [18] proposes a compact 3D Gaussian Splatting SLAM system that reduces the number and the parameter size of Gaussian ellipsoids. NGM-SLAM [19] utilizes neural radiance field submaps for progressive scene expression, achieving effective loop closure detection. MonoGS [20] and Photo-SLAM [21] achieve monocular map reconstruction of Gaussian Splatting-based SLAM. However, existing Gaussian Splatting-based SLAM implementations typically require depth map input from RGB-D sensors to obtain accurate geometry reconstruction.

<!-- image-->  
Fig. 3. The fast Multi-View Stereo network. The inputs of the network are images with poses from sparse visual odometry, image features are extracted by Feature Pyramid Network (FPN) and warped to the 2D cost volume. Finally, encoded and decoded to depth maps using coarse-to-fine strategy.

## III. METHODS

Our approach utilizes RGB image as input, parallelly performing camera pose estimation and photorealistic dense mapping. As depicted in Fig. 2, the core idea of the approach is to use a pre-trained Multi-View Stereo (MVS) network to couple sparse VO and dense Gaussian Splatting mapping. Specifically, in the frontend part, tracking RGB image provides the backend with coarse camera poses and priori depth maps (Sec. III-A). In the backend part, we represent the dense map using 3D Gaussian Splatting, and jointly optimize the dense map and the coarse poses from the frontend (Sec. III-B). In the system components part, system initialization, selecting the keyframes for the system and correcting the scale between sparse point cloud map and dense Gaussian map by SDAR strategy are reported (Sec. III-C).

## A. Sparse Visual Odometry Frontend

To achieve more accurate camera pose tracking and provide dense depth geometry before backend mapping, the frontend of our framework is built on the Deep Patch Visual Odometry (DPVO) [22] algorithm. DPVO is a learning-based sparse monocular visual odometry method. Given an input RGB stream, the scene is represented as a collection of camera poses $\mathbf { T } \in S E ( 3 ) ^ { N }$ and a series of square image patches P extracted from the images. The reprojection of a square patch k taken from frame i in frame j can be formulated as:

$$
\mathbf { P } _ { k } ^ { i j } \sim \mathbf { K } \mathbf { T } _ { j } \mathbf { T } _ { i } ^ { - 1 } \mathbf { K } ^ { - 1 } \mathbf { P } _ { k } ^ { i }\tag{1}
$$

where K refers to camera intrinsic matrix, $\mathbf { P } _ { k } ^ { i } = [ u , v , 1 , d ] ^ { T }$ denotes patch k in frame i, and [u, v] denote the pixel coordinates in images, d denotes the inverse depth.

The core of DPVO is an update operator that computes the hidden state for each edge $( k , i , j ) \in \ \varepsilon .$ . It optimizes the reprojection errors on the patch graph to predict a 2D correction vector $\delta _ { k } ^ { i j } \in \mathbb { R } ^ { 2 }$ and confidence weight $\psi _ { k } ^ { i j } \in \mathbb { R } ^ { 2 }$ Bundle Adjustment (BA) is performed using optical flow correction as a constraint, with iterative updates and refinement of camera poses and patch depths achieved through the nonlinear least squares method. The cost function for bundle adjustment is as follows:

$$
\sum _ { ( k , i , j ) \in \varepsilon } \| \mathbf { K } \mathbf { T } _ { j } \mathbf { T } _ { i } ^ { - 1 } \mathbf { K } ^ { - 1 } \mathbf { P } _ { k } ^ { i } - [ \bar { \mathbf { P } } _ { k } ^ { i j } + \delta _ { k } ^ { i j } ] \| _ { \psi _ { k } ^ { i j } } ^ { 2 }\tag{2}
$$

where $\| \cdot \| _ { \psi }$ represents Mahalanobis distance, PÂ¯ denotes the centre of patch.

Multi-view priori depth estimation. The backend dense Gaussian mapping requires the geometric supervision of depth maps to obtain the accurate geometric positions of Gaussians. In order to make monocular SLAM have the ability of geometric supervision, unlike the previous method [23], we use a pre-trained Multi-View Stereo (MVS) network to estimate priori depth maps on the keyframes window of DPVO, the network is shown in Fig. 3. This method utilizes the geometric consistency of the MVS network to achieve the supervision of the geometric positions of Gaussians through only monocular RGB image input. Furthermore, our MVS network consists entirely of 2D convolutions with a coarse-to-fine structure that progressively refines the estimated priori depth map to reduce the runtime of the MVS network. Tab. IV and Tab. V show that this method achieves better rendering and reconstruction performance.

To be more specific, the frame currently tracked by the sparse visual odometry is used as the reference image $\mathbf { I } ^ { 0 } .$ Additionally, we employ the previous N keyframes as a series of original images $\mathbf { I } ^ { n \in \mathbf { \breve { 1 } } , \dots , N }$ . These images and their corresponding camera poses, serve as inputs to the MVS network. Utilizing the Feature Pyramid Network (FPN) module, we extract three layers of image features $\mathbf { F } _ { i } ^ { s }$ for each image, with s denoting the layer index and i representing the image index. In each layer, the original image features dot the reference image features by a differentiable warping operation to obtain a cost volume with dimensions $\mathbf { D } \times \mathbf { H } ^ { s } \times \mathbf { W } ^ { s }$ , and the priori depth map of each layer is obtained by 2D convolutions encoding and decoding. The estimated depth map of the previous layer is upsampled as the reference depth map of the next layer. The final depth map is estimated after three layers to achieve the coarse-to-fine effect.

Our MVS depth estimation network is trained on the Scan-Net dataset [24]. We train with the AdamW optimizer for 100k steps with a weight decay of $1 0 ^ { - 4 }$ , and a learning rate of $1 0 ^ { - 4 }$ for 70k steps, $1 0 ^ { - 5 }$ until 80k, then dropped to $1 0 ^ { - 6 }$ for remainder, which takes approximately 84 hours on two 24GB RTX3090 GPUs. We use a scale-invariant loss function to accommodate the relative poses of the sparse visual odometers:

$$
\mathcal { L } _ { s i } ^ { s } = \sqrt { \frac { 1 } { H ^ { s } W ^ { s } } \sum _ { i , j } ( g _ { i , j } ^ { s } ) ^ { 2 } - \frac { \lambda } { ( H ^ { s } W ^ { s } ) ^ { 2 } } ( \sum _ { i , j } g _ { i , j } ^ { s } ) ^ { 2 } }\tag{3}
$$

where $g _ { i , j } ^ { s } = \uparrow _ { g t }$ log $\hat { D } _ { i , j } ^ { s } - \log D _ { i , j } ^ { g t } . ~ D _ { i , j } ^ { g t }$ denotes a ground truth depth map, which is aligned to the size of predicted depth $D _ { i , j } ^ { s }$ by an upsampling operation $\uparrow _ { g t } . \mathrm { ~ \ \lambda ~ }$ is a constant 0.85.

In addition, the multi-view loss and the normal loss are added to the loss function to maintain the geometric consistency of depth estimation. The multi-view loss average absolute error on log depth over all valid points:

$$
\mathcal { L } _ { m v } ^ { s } = \frac { 1 } { N H ^ { s } W ^ { s } } \sum _ { n , i , j } \left| \uparrow _ { g t } \log \mathbf { T } _ { 0 n } ( \hat { D } _ { i , j } ^ { s } ) - \log D _ { n , i , j } ^ { g t } \right|\tag{4}
$$

$$
\mathcal { L } _ { n o r m a l } ^ { s } = \frac { 1 } { 2 H ^ { s } W ^ { s } } \sum _ { i , j } ( 1 - \hat { N } _ { i , j } ^ { s } \cdot N _ { i , j } ^ { s } )\tag{5}
$$

where $\mathbf { T } _ { 0 n }$ denotes the transformation matrix from the reference image to the original image n. $\hat { N } _ { i , j } ^ { s }$ and $N _ { i , j } ^ { s }$ respectively denote the prediction normals and ground truth normals. The final MVS depth estimation network loss is as follows:

$$
\mathcal { L } = \sum _ { s = 0 } ^ { l } \frac { 1 } { 2 ^ { s } } ( \lambda _ { s i } \mathcal { L } _ { s i } ^ { s } + \lambda _ { m v } \mathcal { L } _ { m v } ^ { s } + \lambda _ { n o r m a l } \mathcal { L } _ { n o r m a l } ^ { s } )\tag{6}
$$

where l is 2, and we assign the loss weights $\lambda _ { s i } , \lambda _ { m v }$ and $\lambda _ { n o r m a l }$ to 1.0, 0.2 and 1.0 respectively.

## B. 3D Gaussian Splatting Mapping Backend

The main responsibility of the backend is to further optimize the coarse poses from the frontend and map a Gaussian scene. The key to this thread is differentiable rendering and depth smooth regularisation loss, computing the loss between the renderings and the ground truth, and adjusting the coarse poses and Gaussian map by backward gradient propagation.

Differentiable Gaussian map representation. We use 3D Gaussian Splatting as a dense representation of the scene. The influence of a single 3D Gaussian $p _ { i } \in \mathbb { R } ^ { 3 }$ in 3D scene is as follows:

$$
f ( p _ { i } ) = \sigma ( o _ { i } ) \cdot \exp ( - \frac { 1 } { 2 } ( p _ { i } - \mu _ { i } ) ^ { T } \Sigma ^ { - 1 } ( p _ { i } - \mu _ { i } ) )\tag{7}
$$

where $o _ { i } \in \mathbb { R }$ denotes the opacity of the Gaussian, $\mu _ { i } \in \mathbb { R } ^ { 3 }$ is the centre of the Gaussian, $\dot { \Sigma ^ { } } = R S S ^ { T } R ^ { T } \in \mathbb { R } ^ { 3 , 3 }$ is the covariance matrix computed with $S \in \mathbb { R } ^ { 3 }$ scaling and $R \in$ $\mathbb { R } ^ { 3 , 3 }$ components. The expression for the projection of a 3D Gaussian onto the image plane is as follows:

$$
\mu _ { I } = \pi ( \mathbf { T } _ { C W } \cdot \mu _ { W } )\tag{8}
$$

$$
\Sigma _ { I } = J W \Sigma _ { W } W ^ { T } J ^ { T }\tag{9}
$$

where $\pi ( \cdot )$ denotes the projection of the 3D Gaussian center, $\mathbf { T } _ { C W } \ \in \ S E { ( 3 ) }$ is the the transformation matrix from world coordinate to camera coordinate in 3D space, J is a linear approximation to the Jacobian matrix of the projective transformation, W is the rotational component of $\mathbf { T } _ { C W }$ . The Eq. (8) and Eq. (9) are differentiable, which ensures that the Gaussian map can be used with first-order gradient descent to continuously optimize the geometric and photometric of the map, allowing the map to be rendered as photo-realistic images. A single pixel color $C _ { p }$ is rendered from N Gaussians by splatting and blending:

$$
C _ { p } = \sum _ { i \in N } c _ { i } o _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - o _ { j } )\tag{10}
$$

<!-- image-->  
Fig. 4. Depth smooth regularization loss. Comparing the effect of having no depth smooth loss, there is better photometry and geometry with depth smooth loss, and bad photometry and geometry without depth smooth loss.

where $c _ { i }$ is the color of Gaussian i, and $o _ { i }$ is the opacity of Gaussian i.

Mapping Optimization Losses. We changed the loss function of the vanilla 3D Gaussian splatting and added more geometric constraints to make it more suitable for online mapping systems like SLAM. Specifically, our loss function consists of four components: photometric loss, depth geometric loss, depth smooth regularization loss and isotropic loss. In the photometric loss, the L1 loss is calculated between the rendered color image and the ground truth color image in the current camera pose TCW :

$$
\mathcal { L } _ { p h o } = \| I ( \mathcal { G } , \mathbf { T } _ { C W } ) - I _ { g t } \| _ { 1 }\tag{11}
$$

where $I ( { \mathcal { G } } , \mathbf { T } _ { C W } )$ is the rendered color image from Gaussians G, and $I _ { g t }$ is ground truth color image.

To improve the geometric accuracy of the Gaussian map, similar to Eq. (10), We also rendered the depth:

$$
D _ { p } = \sum _ { i \in N } z _ { i } o _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - o _ { j } )\tag{12}
$$

where $z _ { i }$ is the distance along the camera ray to the center $\mu _ { W }$ of Gaussian i. Therefore, the depth geometric loss is as follows:

$$
\mathcal { L } _ { g e o } = \| D ( \mathcal { G } , \mathbf { T } _ { C W } ) - \bar { D } _ { d } \| _ { 1 }\tag{13}
$$

where $D ( \mathcal { G } , \mathbf { T } _ { C W } )$ is the rendered depth map from Gaussians G, $\bar { D } _ { d }$ is the optimized priori depth map by SDAR strategy. The optimization process is in Sec. III-C.

The prior depth maps obtained from the MVS network may not be entirely accurate. As depicted in Fig. 4, direct utilization of these depth maps leads to erroneous guidance in the geometric reconstruction of the Gaussian map. Similar to NeSLAM [25], we introduce the depth smooth regularization loss to reduce this erroneous guidance:

$$
\mathcal { L } _ { s m o o t h } = \| d _ { i , j - 1 } - d _ { i , j } \| _ { 2 } + \| d _ { i + 1 , j } - d _ { i , j } \| _ { 2 }\tag{14}
$$

where $d _ { i , j }$ denotes the depth value of the pixel coordinate at $( i , j )$ in the rendering depth map. However, NeSLAM is an RGB-D SLAM system, which optimizes the noisy depth from RGB-D sensors by the denoising network and constraining the standard variance of depth to obtain better depth input. In contrast, we regularize the adjacent pixels between depth maps rendered from the Gaussian map, enabling the Gaussians to have better geometric positions.

The vanilla 3D Gaussian Splatting algorithm places no constraints on the Gaussians in the ray direction along the viewpoint. This has no effect on 3D reconstruction with fixed viewpoints. However, SLAM is an online mapping system, so this causes the Gaussians to elongate along the direction of the view ray, leading to the appearance of artifacts. To solve this problem, as well as [20], we also introduce isotropic loss:

<!-- image-->  
Fig. 5. Priori depth optimization. this optimization strategy in the SDAR is to correct the geometry of the priori depth map from the MVS network and align the scale with the sparse point cloud map.

$$
\mathcal { L } _ { i s o } = \sum _ { i = 1 } ^ { | \mathcal { G } | } \| s _ { i } - \bar { s } _ { i } \cdot 1 \| _ { 1 }\tag{15}
$$

where $s _ { i }$ is the scaling of Gaussians, suppressing the elongation of the Gaussians by regularizing both the scaling and mean $\bar { s } _ { i } .$ . The final mapping optimization loss function is as follows:

$$
\mathcal { L } = \lambda _ { p } \mathcal { L } _ { p h o } + \lambda _ { g } \mathcal { L } _ { g e o } + \lambda _ { s } \mathcal { L } _ { s m o o t h } + \lambda _ { i } \mathcal { L } _ { i s o }\tag{16}
$$

where we assign the loss weights $\lambda _ { p } , \lambda _ { g } , \lambda _ { s }$ and $\lambda _ { i }$ to 0.99, 0.01, 1.0 and 1.0 respectively.

Camera poses optimization from the Gaussian map. We use the camera poses $\mathbf { T } _ { C W } ^ { i }$ obtained from sparse visual odometry tracking in the frontend as the initial poses for Gaussian mapping in the backend. As in Eq. (10) and $\operatorname { E q . }$ (12), we render the color image and depth map from the Gaussian map at the viewpoint of the current initial poses and compute the loss of renderings and the ground truth. Since this process is differentiable, the loss gradient is propagated to both the Gaussian map and the initial poses during the gradient backward process. The equation of the initial poses optimization update is as follows:

$$
\underset { { \bf { T } } _ { C W } ^ { i } , \mathcal { G } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { n } \mathcal { L } _ { m a p p i n g } ( \mathcal { G } , { \bf { T } } _ { C W } ^ { i } , I _ { g t } ^ { i } , \bar { D } _ { d } ^ { i } )\tag{17}
$$

where $\mathcal { L } _ { m a p p i n g }$ is the Eq. (16), $I _ { g t } ^ { i }$ and $\hat { D } _ { d } ^ { i }$ are ith ground truth color image and optimized priori depth map from the viewpoint of $\mathbf { T } _ { C W } ^ { i }$ in mapping window. n is mapping window size. Minimize the mapping loss to optimize both Gaussians $\mathcal { G }$ and initial poses $\mathbf { T } _ { C W } ^ { i }$ simultaneously.

## C. System Components

System initialization. Similar to DPVO, The system uses 8 frames for initialization. The pose of the new frame is initialized using a constant velocity motion model. We add new patches and frames until 8 frames have been accumulated, and then run 12 iterations of the update operator. The 8 frames in the initialization are used as MVS network inputs to estimate the priori depth of the first frame. The backend uses the first priori depth as the foundation to initialize the Gaussian map.

Keyframe selection. In the frontend tracking process, we always consider the 3 most recent frames as keyframes to fulfill the constant velocity motion model requirement. However, these 3 frames are not utilized for Gaussian mapping. Instead, we assess whether 4th frame satisfies Gaussian co-visibility criteria. If it does, we add it to the mapping process in the backend; otherwise, we discard this frame. This method can determine whether the tracked frame has new information exceeding a threshold, improve the efficiency of keyframe usage, and reduce memory consumption. Between two keyframes $i , j ,$ we define the co-visibility using Intersection of Union (IOU):

$$
I O U _ { c o v } ( i , j ) = \frac { | \mathcal { G } _ { i } \cap \mathcal { G } _ { j } | } { | \mathcal { G } _ { i } \cup \mathcal { G } _ { j } | }\tag{18}
$$

where $\mathcal { G } _ { i } , \mathcal { G } _ { j }$ are visible Gaussians in the viewpoints of frame i and frame $j .$ . If IOU is less than a threshold, the system will create a new keyframe.

Sparse-Dense Adjustment Ring. We propose the Sparse-Dense Adjustment Ring (SDAR) strategy to achieve scale unification of the system. The method consists of three parts is as follows:

Firstly, We use a sparse point cloud map with better geometric accuracy to correct the priori depth map from the MVS network estimate. The priori depth map and the sparse depth map conform to the normal distribution of $\hat { D } _ { d } \sim \mathcal N ( \mu _ { d } , \sigma _ { d } ^ { 2 } )$ and $D _ { s } \sim \mathcal N ( \mu _ { s } , \sigma _ { s } ^ { 2 } )$ . Align the priori depth map with the sparse depth map using the following equation:

$$
\bar { D } _ { d } = \frac { \sigma _ { s } } { \hat { \sigma } _ { d } } \hat { D } _ { d } + \mu _ { d } \big ( \frac { \mu _ { s } } { \hat { \mu } _ { d } } - \frac { \sigma _ { s } } { \hat { \sigma } _ { d } } \big )\tag{19}
$$

where $\hat { \mu } _ { d }$ and $\hat { \sigma } _ { d }$ are the mean and standard deviation statustics of the sparsified priori depth map extracted from $\hat { D } _ { d }$ at the pixel coordinates of $D _ { s }$ . This strategy corrects the prior depth errors, as shown in Fig. 5.

Secondly, we backproject the optimized prior depth map with RGB color into space, generating a new point cloud. Subsequently, downsampling is performed on this new point cloud. New Gaussians are then initialized with the downsampled point cloud and added to the Gaussian map.

Finally, to achieve scale closure, we leverage the real-time rendering capability of the Gaussian map to generate the depth map of the frame being tracked at the frontend. We then initialize the depth of the tracking frameâs point cloud using this depth map. This strategy ensures that the frontend track aligns with the scale of the backend Gaussian map.

## IV. EXPERIMENTS

We evaluate our proposed system on a series of real and synthetic datasets, including the TUM dataset [26], Replica dataset [27] and ICL-NUIM dataset [28]. We compare the pose estimation accuracy (ATE), novel view rendering quality and geometric reconstruction quality with previous works, utilizing experimental results from papers or open-source code of these works. The experimental data from the source code represents the average of three runs. Additionally, we conduct some ablation studies to demonstrate the effectiveness of our systemâs components. Finally, we analyze the system runtime and memory.

NICE-SLAM  
Ground Truth  
Ours  
<!-- image-->  
Fig. 6. The results of novel view rendering demonstrate the visualization outcomes on the Replica dataset for the proposed MGS-SLAM and other methods. Our system consistently generates significantly higher-quality and more realistic images than other monocular and RGB-D methods. This observation is further supported by quantitative results in Tab. IV.

TABLE I  
ATE [CM] RESULTS ON TUM DATASET.
<table><tr><td>Input</td><td>Method</td><td>fr1/desk</td><td>fr1/desk2</td><td>fr1/plant</td><td>fr2/xyz</td><td>fr3/office</td><td>Avg.</td></tr><tr><td rowspan="4">FRCR</td><td>SplaTAM</td><td>3.35</td><td>6.54</td><td>2.74</td><td>1.24</td><td>5.16</td><td>3.81</td></tr><tr><td>Co-SLAM</td><td>2.70</td><td>4.31</td><td>4.74</td><td>1.90</td><td>2.60</td><td>3.25</td></tr><tr><td>ESLAM</td><td>2.30</td><td>3.78</td><td>2.11</td><td>1.10</td><td>2.40</td><td>2.34</td></tr><tr><td>DSO</td><td>22.40</td><td>91.60</td><td>12.10</td><td>1.10</td><td>9.50</td><td>27.34</td></tr><tr><td rowspan="5">&#x27;OuoW</td><td>DROID-VO</td><td>5.20</td><td>9.90</td><td>2.80</td><td>10.70</td><td>7.30</td><td>7.18</td></tr><tr><td>MonoGS</td><td>4.15</td><td>7.16</td><td>7.82</td><td>4.79</td><td>4.39</td><td>5.66</td></tr><tr><td>Photo-SLAM</td><td>1.54</td><td>21.00</td><td>3.67</td><td>0.98</td><td>1.26</td><td>5.69</td></tr><tr><td>DPVO</td><td>3.80</td><td>6.40</td><td>4.70</td><td>0.54</td><td>7.00</td><td>4.49</td></tr><tr><td>Ours</td><td>2.33</td><td>5.32</td><td>3.55</td><td>0.44</td><td>3.00</td><td>2.93</td></tr></table>

TABLE II

ATE [CM] RESULTS ON REPLICA DATASET
<table><tr><td>Input</td><td>Method</td><td>R0</td><td>R1</td><td>R2</td><td>O0</td><td>O1</td><td>O2</td><td>O3</td><td>O4</td><td>Avg.</td></tr><tr><td rowspan="5">FCP</td><td>Co-SLAM</td><td>0.70</td><td>0.95</td><td>1.35</td><td>0.59</td><td>0.55</td><td>2.03</td><td>1.56</td><td>0.72</td><td>1.06</td></tr><tr><td>ESLAM</td><td>0.71</td><td>0.70</td><td>0.52</td><td>0.57</td><td>0.55</td><td>0.58</td><td>0.72</td><td>0.63</td><td>0.62</td></tr><tr><td>NeSLAM</td><td>0.60</td><td>0.93</td><td>0.52</td><td>0.41</td><td>0.43</td><td>0.57</td><td>0.96</td><td>0.83</td><td>0.66</td></tr><tr><td>GS-SLAM</td><td>0.48</td><td>0.53</td><td>0.33</td><td>0.52</td><td>0.41</td><td>0.59</td><td>0.46</td><td>0.70</td><td>0.50</td></tr><tr><td>SplaTAM</td><td>0.31</td><td>0.40</td><td>0.29</td><td>0.47</td><td>0.27</td><td>0.29</td><td>0.32</td><td>0.55</td><td>0.36</td></tr><tr><td rowspan="6">&#x27;ouow</td><td>DROID-VO</td><td>0.50</td><td>0.70</td><td>0.30</td><td>0.98</td><td>0.29</td><td>0.84</td><td>0.45</td><td>1.53</td><td>0.70</td></tr><tr><td>NICER-SLAM</td><td>1.36</td><td>1.60</td><td>1.14</td><td>2.12</td><td>3.23</td><td>2.12</td><td>1.42</td><td>2.01</td><td>1.88</td></tr><tr><td>MonoGS</td><td>9.94</td><td>66.22</td><td>43.94</td><td>62.09</td><td>19.09</td><td>45.60</td><td>11.58</td><td>58.75</td><td>39.65</td></tr><tr><td>Photo-SLAM</td><td>0.35</td><td>1.18</td><td>0.23</td><td>0.58</td><td>0.32</td><td>5.03</td><td>0.47</td><td>0.58</td><td>1.09</td></tr><tr><td>DPVO</td><td>0.49</td><td>0.54</td><td>0.54</td><td>0.77</td><td>0.36</td><td>0.57</td><td>0.46</td><td>0.57</td><td>0.54</td></tr><tr><td>Ours</td><td>0.36</td><td>0.35</td><td>0.32</td><td>0.35</td><td>0.28</td><td>0.26</td><td>0.32</td><td>0.34</td><td>0.32</td></tr></table>

## A. Implementation Details

We evaluate our proposed system and other methods on a desktop with an Intel Core i7 12700 processor running at 3.60GHz and a single NVIDIA GeForce RTX 3090. The size of input images is consistent with the dataset size in our system. Similar to Gaussian Splatting, mapping rasterization and gradient computations are implemented using CUDA. The remainder of our system pipeline is developed with PyTorch. For map optimization, we set the maximum gradient threshold to 0.0002 and the minimum opacity threshold to 0.65 for the Gaussians in the densify and prune operation.

## B. Camera Tracking Accuracy

For camera tracking accuracy, we report the Root Mean Square Error (RMSE) of the keyframeâs Absolute Trajectory Error (ATE). We benchmark our system against other approaches. The comparative works are very comprehensive including traditional visual odometry DSO [29], learningbased visual odometry DROID-VO [7], neural implicit-based NICER-SLAM [30], NeSLAM [25], ESLAM [31], Co-SLAM [32] and more recently Gaussian Splatting-based SplaTAM [16], MonoGS [20], GS-SLAM [17], Photo-SLAM [21].

TABLE III  
ATE [CM] RESULTS ON ICL-NUIM DATASET
<table><tr><td>Input</td><td>Method</td><td>L0</td><td>L1</td><td>L2</td><td>L3</td><td>O0</td><td>O1</td><td>O2</td><td>O3</td><td>Avg.</td></tr><tr><td rowspan="3">F-CP</td><td>Co-SLAM</td><td>1.15</td><td>0.85</td><td>1.03</td><td>16.46</td><td>52.46</td><td>3.60</td><td>1.76</td><td>39.15</td><td>14.56</td></tr><tr><td>ESLAM</td><td>0.45</td><td>0.49</td><td>1.61</td><td>5.84</td><td>0.42</td><td>1.37</td><td>1.01</td><td>0.46</td><td>1.46</td></tr><tr><td>SplaTAM</td><td>0.53</td><td>0.70</td><td>1.13</td><td>4.63</td><td>0.42</td><td>1.03</td><td>0.92</td><td>1.16</td><td>1.32</td></tr><tr><td rowspan="6">&#x27;OuoW</td><td>DSO</td><td>1.00</td><td>2.00</td><td>6.00</td><td>3.00</td><td>21.00</td><td>83.00</td><td>36.00</td><td>64.00</td><td>27.00</td></tr><tr><td>DROID-VO</td><td>1.00</td><td>12.30</td><td>7.20</td><td>3.20</td><td>9.50</td><td>4.10</td><td>84.20</td><td>50.40</td><td>21.49</td></tr><tr><td>MonoGS</td><td>6.40</td><td>21.21</td><td>31.40</td><td>100.76</td><td>13.87</td><td>35.76</td><td>24.73</td><td>73.42</td><td>38.44</td></tr><tr><td>Photo-SLAM</td><td>0.54</td><td>4.52</td><td>0.72</td><td>0.98</td><td>3.41</td><td>18.19</td><td>1.54</td><td>4.71</td><td>4.33</td></tr><tr><td>DPVO</td><td>0.60</td><td>0.60</td><td>2.30</td><td>1.00</td><td>6.70</td><td>1.20</td><td>1.70</td><td>63.50</td><td>9.70</td></tr><tr><td>Ours</td><td>0.58</td><td>0.50</td><td>1.82</td><td>0.77</td><td>1.46</td><td>1.01</td><td>1.19</td><td>1.49</td><td>1.10</td></tr></table>

Tab. I shows the tracking results on the TUM dataset. The tracking accuracy of our system outperforms other monocular methods by 35% and is comparable to ESLAM using RGB-D input. Tab. II and Tab. III show that our system achieved the best tracking performance compared to other systems including monocular and RGB-D. In addition, The experimental data from the tables show that our tracking performance is superior to the DPVO on which the frontend is based. This demonstrates the effectiveness of our combination of sparse visual odometry and Gaussian mapping in achieving a more robust and accurate SLAM system.

## C. Novel View Rendering

We evaluated the methods for novel view rendering on Replica. To evaluate map quality, we report standard photometric rendering quality metrics (PSNR, SSIM and LPIPS). The methods we are comparing have RGB-D input and monocular input. NICE-SLAM [12], Vox-Fusion [33], ESLAM [31] and Co-SLAM [32] are neural implicit-based RGB-D input and the rest are monocular input. We take the average of frames other than keyframes to evaluate rendering quality. Tab. IV shows the results, our proposed system performs state-of-theart in most scenes. The visualization of the rendering is shown in Fig. 6, where the quality of our rendered image is higher than the other methods and almost indistinguishable from the ground truth.

TABLE IV  
RENDERING PERFORMANCE ON REPLICA DATASET. BEST RESULTS ARE HIGHLIGHTED AS FIRST SECOND , AND THIRD
<table><tr><td>Input Method</td><td>PSNR[dB]â</td><td>Metric</td><td>R0</td><td>R1</td><td>R2</td><td>O0</td><td>01</td><td>O2</td><td>O3</td><td>O4</td><td>Avg.</td></tr><tr><td colspan="16" rowspan="19">NICE- SLAM FRCR</td></tr><tr><td>22.12 22.47</td><td>24.52 0.814</td><td>29.07 0.874</td><td>30.34 0.886</td><td>19.66 0.797</td><td>22.23 0.801</td><td>24.94</td><td>24.42</td></tr><tr><td>SSIMâ LPIPSâ</td><td>0.689 0.330</td><td>0.757 0.271</td><td>0.208</td><td>0.229</td><td>0.181</td><td>0.235</td><td>0.209</td><td>0.856 0.198</td><td>0.809 0.233</td></tr><tr><td>PSNR[B]*</td><td></td><td>22.36</td><td>23.92</td><td>27.79</td><td>29.83</td><td>20.33</td><td>23.47</td><td>25.21</td><td>24.41</td></tr><tr><td>Vox- Fusion</td><td>SSIMâ</td><td>22.39 0.683</td><td>0.751</td><td>0.798</td><td>0.857</td><td>0.876</td><td>0.794</td><td>0.803</td><td>0.847 0.801</td></tr><tr><td></td><td>LPIPSâ PSSNR(dB]*</td><td>0.303 25.32</td><td>0.269 27.77</td><td>0.234 29.08</td><td>0.241 33.71</td><td>0.184 30.20</td><td>0.243 28.09</td><td>0.213 28.77</td><td>0.199 0.236 29.71 29.08</td></tr><tr><td>ESLAM</td><td>SSIMâ LPIPSâ</td><td>0.875</td><td>0.902</td><td>0..932</td><td>0.960</td><td>0.923</td><td>0.943</td><td>0.948</td><td>0.945 0.928</td></tr><tr><td></td><td>PSNR[DB]â </td><td>0.313</td><td>0.298</td><td>0.248</td><td>0.184</td><td>0.228</td><td>0.241</td><td>0.196</td><td>0.204 0.239</td></tr><tr><td>Co- SLAM</td><td>SSIMâ</td><td>27.27</td><td>28.45</td><td>29.06</td><td>34.14</td><td>34.87</td><td>28.43</td><td>28.76</td><td>30.91 30.24</td></tr><tr><td></td><td>LPIPSâ</td><td>0.910 0.324</td><td>0.909</td><td>0.932</td><td>0.961</td><td>0.969</td><td>0.938</td><td>0.941</td><td>0.955 0.939</td></tr><tr><td rowspan="7">SLAM NICER- SLAM</td><td>GO-</td><td></td><td>0.294</td><td>0.266</td><td>0.209</td><td>0.196</td><td>0.258</td><td>0.229</td><td>0.236</td><td>0.252</td></tr><tr><td>PSNR[dB]â SSIMâ</td><td>23.25</td><td>20.70</td><td>21.08</td><td>21.44</td><td>22.59</td><td>22.33</td><td>22.19</td><td>22.76</td><td>22.04</td></tr><tr><td>LPIPSâ</td><td>0.712 0.222</td><td>0.739 0.492</td><td>0.708 0.317</td><td>0.761 0.319</td><td>0.726 0.269</td><td>0.740 0.434</td><td>0.752 0.396</td><td>0.722 0.385</td><td>0.733 0.354</td></tr><tr><td>PSNR(B]</td><td>25.33</td><td></td><td></td><td></td><td>25.86</td><td>21.95</td><td>26.13</td><td>25.47</td><td></td></tr><tr><td>SSIMâ</td><td>0.751</td><td>23.92</td><td>26.12 0.831</td><td>28.54 0.866</td><td>0.852</td><td>0.820</td><td>0.856</td><td>0.865</td><td>25.41</td></tr><tr><td>LPIPSâ</td><td>0.250</td><td>0.771 0.215</td><td>0.176</td><td>0.172</td><td>0.178</td><td>0.195</td><td>0.162</td><td>0.177</td><td>0.827 0.191</td></tr><tr><td>Mono SSIMâ GS</td><td>PSNR[dB]â</td><td>25.11 24.66</td><td></td><td>22.30</td><td>28.76</td><td>29.17</td><td>23.74 - - - -</td><td>23.66 23.99</td><td>25.17</td></tr><tr><td rowspan="6">&#x27;ouoW Photo- SLAM</td><td></td><td>0.790</td><td>0.790</td><td>0.843</td><td>0.884</td><td>0.852</td><td>0.840</td><td>0.855</td><td>0.863</td><td>0.840</td></tr><tr><td>LPIPSâ</td><td>0.260</td><td>0.360</td><td>0.351</td><td>0.293</td><td>0.274</td><td>0.290</td><td>0.216</td><td>0.340</td><td>0.298</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PSNR[dB]â SIM</td><td>29.07</td><td>31.02</td><td>31.22</td><td>35.23</td><td>5.11</td><td>29.70</td><td>31.20</td><td>31.27</td><td>31.73</td></tr><tr><td>LPIPSâ</td><td>0.845 0.186</td><td>0.902</td><td>0.923 0.127</td><td>0.948 0.109</td><td>0.942 0.121</td><td>0.907</td><td>0.915</td><td>0.930</td><td>0.914</td></tr><tr><td></td><td></td><td></td><td>0.125</td><td></td><td></td><td></td><td>0.173</td><td>0.137</td><td>0.120</td><td>0.137</td></tr><tr><td rowspan="2"></td><td>PSNR[dB]â</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>29.91</td><td>3106</td><td>31.9</td><td>35.51</td><td>34.25</td><td>30.83</td><td>31.86</td><td>34.38</td><td>32.41</td></tr><tr><td></td><td>SSIMâ LPIPSâ</td><td>0.894 0.084</td><td>0.895</td><td>0.913</td><td>0.941</td><td>0.930</td><td>0.906</td><td>0.919</td><td>0.945</td><td>0.918</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>0.086</td><td>0.081</td><td></td><td>0.070 0.114</td><td>0.120</td><td>0.074</td><td>0.077</td><td>0.088</td></tr></table>

TABLE V

RECONSTRUCTION PERFORMANCE ON REPLICA DATASET. BEST RESULTSARE HIGHLIGHTED AS FIRST SECOND , AND THIRD
<table><tr><td>Input</td><td>Method</td><td>Depth L1[cm]â</td><td>Acc.[cm]â</td><td>Comp.[cm]â</td><td>Comp. Ratio[&lt;5cm]â</td></tr><tr><td rowspan="5">&#x27;ouoW</td><td>MonoGS</td><td>36.58</td><td>74.02</td><td>19.30</td><td>37.51</td></tr><tr><td>Photo-SLAM</td><td>19.73</td><td>53.70</td><td>8.08</td><td>49.46</td></tr><tr><td>GO-SLAM</td><td>4.39</td><td>3.81</td><td>4.79</td><td>78.00</td></tr><tr><td>NICER-SLAM</td><td></td><td>3.65</td><td>4.16</td><td>79.37</td></tr><tr><td>Ours</td><td>7.77</td><td>7.51</td><td>3.64</td><td>82.71</td></tr></table>

## D. Geometric Reconstruction

We evaluated the methods for geometric reconstruction on Replica. The methods evaluated are all monocular differentiable rendering SLAM approaches. We report standard mesh geometric reconstruction metrics (Depth L1, Accuracy, Completion, Completion Ratio). Tab. V shows the results, our method achieved the best results in terms of Completion and Completion Ratio metrics. It is worth noting that our geometric reconstruction performance is 50% higher than other monocular 3D Gaussian Splatting-based SLAM, which proves the effectiveness of our method in utilizing the MVS network to promote geometric reconstruction. Furthermore, this better geometric reconstruction also improves the rendering.

## E. Ablative Analysis

Mapping losses ablation. We changed the loss function of the vanilla 3D Gaussian Splatting by introducing depth loss, smooth loss, and isotropic loss. As shown in Tab. VI, we did an ablation study of these losses. The results show that all these losses contribute to the accuracy improvement of the system. It is worth noting that the incorrect geometric guidance caused by the depth loss using the prior depth maps was corrected after adding depth smoothing loss.

TABLE VI  
MAPPING LOSSES ABLATION ON OFFICE 0
<table><tr><td> $\mathcal { L } _ { g e o }$ </td><td>smooth</td><td> $\mathcal { L } _ { i s o }$ </td><td>ATE[cm]â</td><td>PSNR[dB]â</td><td>Depth L1[cm]â</td></tr><tr><td>X</td><td>X</td><td></td><td>0.53</td><td>31.21</td><td>25.46</td></tr><tr><td></td><td></td><td></td><td>0.45</td><td>33.80</td><td>11.09</td></tr><tr><td></td><td></td><td>xxÃ&gt;</td><td>0.40</td><td>33.88</td><td>7.21</td></tr><tr><td></td><td></td><td></td><td>0.35</td><td>34.85</td><td>5.37</td></tr></table>

TABLE VII

SPARSE-DENSE ADJUSTMENT RING ABLATION ON OFFICE 0
<table><tr><td>Comp. 1</td><td>Comp. 2</td><td>Comp. 3</td><td>ATE[cm]â</td><td>PSNR[dB]â</td><td>Depth L1[cm]â</td></tr><tr><td>X</td><td>X</td><td></td><td>0.61</td><td>28.66</td><td>15.55</td></tr><tr><td></td><td>X</td><td></td><td>0.49</td><td>29.53</td><td>11.01</td></tr><tr><td></td><td></td><td></td><td>0.41</td><td>33.22</td><td>5.56</td></tr><tr><td></td><td></td><td>xÃÃ&gt;</td><td>0.35</td><td>34.85</td><td>5.37</td></tr></table>

TABLE VIII

RUNTIME AND MEMORY ANALYSIS ON TUM AND REPLICA DATASETS
<table><tr><td>Dataset</td><td>Method</td><td>Tra/It.â</td><td>Map/It.â</td><td>Tra/Fr.â</td><td>Map/Fr.â</td><td>Ren. FPSâ</td><td>Mem.â</td></tr><tr><td rowspan="4">Nâ©L</td><td>SplaTAM</td><td>14.28ms</td><td>16.77ms</td><td>2.85s</td><td>0.50s</td><td>526.32</td><td>42.31MB</td></tr><tr><td>MonoGS</td><td>6.78ms</td><td>12.67ms</td><td>0.65s</td><td>1.90s</td><td>1126.10</td><td>2.80MB</td></tr><tr><td>Photo-SLAM</td><td>-</td><td>8.91ms</td><td>33.33ms</td><td>-</td><td>1648.20</td><td>12.77MB</td></tr><tr><td>Ours</td><td>-</td><td>11.90ms</td><td>35.17ms</td><td>1.85s</td><td>1173.21</td><td>1.96MB</td></tr><tr><td rowspan="4">Bda</td><td>SplaTAM</td><td>25.43ms</td><td>23.80ms</td><td>2.25s</td><td>1.43s</td><td>125.64</td><td>273.09MB</td></tr><tr><td>MonoGS</td><td>10.78ms</td><td>20.50ms</td><td>1.10s</td><td>3.07s</td><td>769.00</td><td>24.50MB</td></tr><tr><td>Photo-SLAM</td><td>-</td><td>15.18ms</td><td>37.45ms</td><td>-</td><td>911.26</td><td>22.21MB</td></tr><tr><td>Ours</td><td></td><td>18.98ms</td><td>38.41ms</td><td>2.97s</td><td>776.50</td><td>20.90MB</td></tr></table>

Sparse-Dense Adjustment Ring ablation. We propose the Sparse-Dense Adjustment Ring (SDAR) strategy to unify the frontend and backend scales. This strategy comprises three components (Sec. III-C). We conducted an ablation study of these three components to demonstrate their effect on the system. As shown in Tab. VII, the contribution of SDAR to the system is mainly in the tracking accuracy ATE. The tracking accuracy of the system is similar to DPVO without the SDAR strategy.

MVS window analysis. As depicted in Fig. 7, we have analyzed the effect of different window sizes of MVS on the accuracy and speed of the system. Since our MVS network consists of 2D convolutions, increasing the window size has little effect on inference time. However, increasing the window size improves the systemâs tracking accuracy and mapping quality. This is because more keyframes with different views provide additional geometrical cues.

## F. Runtime and Memory Analysis

As shown in Tab. VIII, We quoted the method [18] to analyze the runtime and memory of our system and compare it to other methods on the TUM and Replica datasets. The memory is the memory usage of the checkpoint. Some methods do not use this metric and are represented by shorter lines. The metric of tracking each frame contains the inference time of the MVS network in our method, and the MVS network runs on keyframes. The results show that our tracking speed is similar to Photo-SLAM. However, our method achieved better geometry at the expense of tracking time, resulting in more compact checkpoint and the best memory utilization.

<!-- image-->  
Fig. 7. MVS window analysis on Office 0. The MVS window size is a hyperparameter that allows for finding a balance between speed, tracking, and rendering quality. PSNR is divided by 50.

## V. CONCLUSIONS

This letter introduces MGS-SLAM, a novel Gaussian Splatting-based SLAM framework. For the first time, our framework jointly optimizes sparse visual odometry tracking and 3D Gaussian mapping, enhancing tracking accuracy and geometric reconstruction precision of Gaussian maps when only given RGB image input. We develop a lightweight MVS depth estimation network to facilitate this integration. Additionally, we propose the Sparse-Dense Adjustment Ring (SDAR) strategy to adjust the scale between the sparse map and the Gaussian map. Comparative evaluations demonstrate that our approach achieves state-of-the-art accuracy compared to previous methods. We believe that this innovative method will bring some inspiration to future works.

## REFERENCES

[1] A. J. Davison, I. D. Reid, N. D. Molton, and O. Stasse, âMonoslam: Real-time single camera slam,â IEEE transactions on pattern analysis and machine intelligence, vol. 29, no. 6, pp. 1052â1067, 2007.

[2] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, âOrb-slam: a versatile and accurate monocular slam system,â IEEE transactions on robotics, vol. 31, no. 5, pp. 1147â1163, 2015.

[3] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[4] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, pp. 1â14, 2023.

[5] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, âDtam: Dense tracking and mapping in real-time,â in 2011 international conference on computer vision. IEEE, 2011, pp. 2320â2327.

[6] J. Engel, T. Schops, and D. Cremers, âLsd-slam: Large-scale direct Â¨ monocular slam,â in European conference on computer vision. Springer, 2014, pp. 834â849.

[7] Z. Teed and J. Deng, âDroid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras,â Advances in neural information processing systems, vol. 34, pp. 16 558â16 569, 2021.

[8] R. Craig and R. C. Beavis, âTandem: matching proteins with tandem mass spectra,â Bioinformatics, vol. 20, no. 9, pp. 1466â1467, 2004.

[9] H. Matsuki, R. Scona, J. Czarnowski, and A. J. Davison, âCodemapping: Real-time dense mapping for sparse slam using compact scene representations,â IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 7105â7112, 2021.

[10] A. Rosinol, J. J. Leonard, and L. Carlone, âProbabilistic volumetric fusion for dense monocular slam,â in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 3097â 3105.

[11] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, âimap: Implicit mapping and positioning in real-time,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 6229â6238.

[12] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 786â12 796.

[13] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, âGo-slam: Global optimization for consistent 3d instant reconstruction,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3727â3737.

[14] L. Liso, E. Sandstrom, V. Yugay, L. Van Gool, and M. R. Oswald, Â¨ âLoopy-slam: Dense neural slam with loop closures,â arXiv preprint arXiv:2402.09944, 2024.

[15] T. Deng, G. Shen, T. Qin, J. Wang, W. Zhao, J. Wang, D. Wang, and W. Chen, âPlgslam: Progressive neural scene represenation with local to global bundle adjustment,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 657â19 666.

[16] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat, track & map 3d gaussians for dense rgb-d slam,â arXiv preprint arXiv:2312.02126, 2023.

[17] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, âGsslam: Dense visual slam with 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 595â19 604.

[18] T. Deng, Y. Chen, L. Zhang, J. Yang, S. Yuan, D. Wang, and W. Chen, âCompact 3d gaussian splatting for dense visual slam,â arXiv preprint arXiv:2403.11247, 2024.

[19] M. Li, J. Huang, L. Sun, A. X. Tian, T. Deng, and H. Wang, âNgm-slam: Gaussian splatting slam with radiance field submap,â arXiv preprint arXiv:2405.05702, 2024.

[20] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â arXiv preprint arXiv:2312.06741, 2023.

[21] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Realtime simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 584â21 593.

[22] Z. Teed, L. Lipson, and J. Deng, âDeep patch visual odometry,â Advances in Neural Information Processing Systems, vol. 36, 2024.

[23] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, âColmapfree 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 20 796â20 805.

[24] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. NieÃner, âScannet: Richly-annotated 3d reconstructions of indoor scenes,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828â5839.

[25] T. Deng, Y. Wang, H. Xie, H. Wang, J. Wang, D. Wang, and W. Chen, âNeslam: Neural implicit mapping and self-supervised feature tracking with depth completion and denoising,â arXiv preprint arXiv:2403.20034, 2024.

[26] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of rgb-d slam systems,â in 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012, pp. 573â580.

[27] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[28] A. Handa, T. Whelan, J. McDonald, and A. Davison, âA benchmark for RGB-D visual odometry, 3D reconstruction and SLAM,â in IEEE Intl. Conf. on Robotics and Automation, ICRA, Hong Kong, China, May 2014.

[29] J. Engel, V. Koltun, and D. Cremers, âDirect sparse odometry,â IEEE transactions on pattern analysis and machine intelligence, vol. 40, no. 3, pp. 611â625, 2017.

[30] Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and M. Pollefeys, âNicer-slam: Neural implicit scene encoding for rgb slam,â arXiv preprint arXiv:2302.03594, 2023.

[31] M. M. Johari, C. Carta, and F. Fleuret, âEslam: Efficient dense slam system based on hybrid representation of signed distance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 408â17 419.

[32] H. Wang, J. Wang, and L. Agapito, âCo-slam: Joint coordinate and sparse parametric encodings for neural real-time slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 293â13 302.

[33] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, âVoxfusion: Dense tracking and mapping with voxel-based neural implicit representation,â in 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2022, pp. 499â507.