# Rig-Aware 3D Reconstruction of Vehicle Undercarriages using Gaussian Splatting

Nitin Kulkarniââ , Akhil Devarashettiâ, Charlie Clussâ, Livio Forteâ,

Dan Buckmasterâ, Philip Schneiderâ, Chunming Qiaoâ , Alina Vereshchakaâ 

â University at Buffalo, Buffalo, NY, USA

âACV Auctions, Buffalo, NY, USA

{nitinvis, qiao, avereshc}@buffalo.edu, {adevarashetti, ccluss, lforte, dbuckmaster, pschneider}@acvauctions.com

AbstractâInspecting the undercarriage of used vehicles is a labor-intensive task that requires inspectors to crouch or crawl underneath each vehicle to thoroughly examine it. Additionally, online buyers rarely see undercarriage photos. We present an end-to-end pipeline that utilizes a three-camera rig to capture videos of the undercarriage as the vehicle drives over it, and produces an interactive 3D model of the undercarriage. The 3D model enables inspectors and customers to rotate, zoom, and slice through the undercarriage, allowing them to detect rust, leaks, or impact damage in seconds, thereby improving both workplace safety and buyer confidence. Our primary contribution is a rigaware Structure-from-Motion (SfM) pipeline specifically designed to overcome the challenges of wide-angle lens distortion and lowparallax scenes. Our method overcomes the challenges of wideangle lens distortion and low-parallax scenes by integrating precise camera calibration, synchronized video streams, and strong geometric priors from the camera rig. We use a constrained matching strategy with learned components, the DISK feature extractor, and the attention-based LightGlue matcher to generate high-quality sparse point clouds that are often unattainable with standard SfM pipelines. These point clouds seed the Gaussian splatting process to generate photorealistic undercarriage models that render in real-time. Our experiments and ablation studies demonstrate that our design choices are essential to achieve stateof-the-art quality.

Index TermsâGaussian splatting, structure-from-motion, LightGlue, NeRFs, 3D reconstruction, photogrammetry, vehicle inspection

## I. INTRODUCTION

Used vehicles represent a multi-billion-dollar global market [1], with digital auctions capturing an increasingly large share. Yet, assessing a vehicleâs underbody condition, one of the key factors in determining its value, remains challenging. Signs of rust, fluid leaks, or frame damage are often obscured from standard imaging and inspection routines. Vehicle inspectors may miss issues due to cramped working conditions, and buyers, especially in rural or remote areas, often lack access to trusted inspection services. This challenge is particularly pronounced for individuals in rural or remote areas, who often lack access to specialized inspection services. Without clear visibility into the vehicleâs underside, buyers make decisions under uncertainty, risking costly repairs and reducing trust in online transactions.

C. Qiao is supported in part by the National Science Foundation under Grant Nos. CNS-2413876 and CNS-2120369. The authors thank ACV Auctions for providing computing resources, vehicle data, and the camera rig.

To bridge this information gap, we propose a solution that delivers high-resolution, photorealistic, and interactive 3D models of vehicle undercarriages. Such 3D models enable buyers to remotely inspect a vehicle with fine-grained detail. For sellers and inspectors, this pipeline provides increased transparency and accountability, ultimately improving trust and setting a new standard in the resale marketplace.

Our system captures high-resolution videos from three laterally spaced wide-angle cameras as a vehicle drives over a custom rig. Wide-angle lenses (160Â° FOV) are required to capture the undercarriage as the distance between the cameras and the undercarriage is short (12 â 30 cm). However, such wideangle lenses introduce significant distortion, which makes feature matching difficult. A single camera lens only provides parallax along the length of the vehicle as the vehicle moves in a straight line over the cameras. To generate width-wise parallax for depth estimation, we use three laterally spaced cameras with wide-angle lenses. Traditional Structure-from-Motion (SfM) pipelines often struggle under these conditions, and Neural Radiance Fields (NeRFs) [2] are too slow for production-level vehicle inspection pipelines.

We introduce a novel, modular pipeline that integrates traditional multi-view geometry with 3D Gaussian splatting (3D-GS) [3] to efficiently reconstruct detailed and interactive undercarriage models. First, we perform precise camera calibration using ChArUco boards and select a diverse and sharp set of images to model lens distortion and minimize projection error (Sec. III-A). Next, we synchronize multi-camera streams at sub-frame precision by estimating global vertical shifts using phase correlation, followed by refining the timing offsets through adaptive cross-correlation (Sec. III-B). This two-stage alignment achieves millisecond-level synchronization, which is needed for our downstream processing.

We then apply our rig-aware SfM approach that uses stateof-the-art learned components. After selecting the sharpest image triplets, we use our calibrated intrinsics to undistort the images. From these undistorted images, we extract dense local features using DISK [4], a learned keypoint detector designed to work well in challenging conditions. We match features using the attention-based matcher LightGlue [5] on a constrained set of spatiotemporally proximate image pairs. The resulting raw matches are geometrically verified within COLMAP [6]. We then use GLOMAP [7] to run global bundle adjustment, using priors from the known camera rig geometry, to refine the camera poses and generate a clean and accurate point cloud of the undercarriage (Sec. III-C). The point cloud and optimized camera poses serve as the seed for Gaussian splatting to generate the interactive 3D model (Sec. III-D).

Our proposed pipeline efficiently generates inspection-grade 3D reconstructions, allowing vehicle inspectors and buyers to inspect a carâs undercarriage interactively in real-time. We demonstrate that precise calibration, synchronization, and geometry-aware SfM can achieve NeRF-level quality on challenging wide-angle, low-parallax sequences, making this a practical and scalable solution for real-world deployment. This technology has the potential to standardize vehicle inspections across online marketplaces, enhance trust, and support applications such as insurance claims and fleet maintenance.

## II. BACKGROUND

Accurate novel-view synthesis from 2D images is fundamental to applications such as virtual reality, cultural-heritage digitization, and industrial inspection [8]. Our work builds on key strands of 3D reconstruction: classical multi-view geometry, deep learning models for feature extraction and matching, and recent advances in radiance fields.

## A. Structure-from-Motion

Structure-from-Motion (SfM) estimates 3D scene geometry and camera poses from overlapping images by detecting and matching keypoints across views. Starting from an initial image pair, systems like COLMAP incrementally add new views through resectioning and triangulation, followed by bundle adjustment (BA) to jointly refine camera poses and 3D point locations. The result is a sparse 3D point cloud and a globally consistent camera trajectory.

However, standard SfM pipelines face challenges with distortion and low-parallax scenes, and the quality of the reconstruction is highly dependent on the quality and distribution of feature matches. Scenes captured with wide-angle lenses, as required in our undercarriage inspection, introduce severe nonlinear distortions that can confound feature matching if not accurately modeled [9]. Additionally, SfM struggles with camera configurations with low parallax; differences between views that are crucial for depth estimation [10]. This scenario inherent to our drive-over camera rig can lead to geometric degeneracies and drift in the estimated camera poses, ultimately degrading the quality of the final 3D point cloud.

## B. Learned Local Features and Matchers

Traditional SfM pipelines have relied on handcrafted features like SIFT [11], but recent progress in deep learning has led to more powerful alternatives. Learned local features, such as DISK, are trained on large datasets of images to detect keypoints and generate descriptors that are more robust to extreme viewpoint and illumination changes than classical methods.

Similarly, deep learning-based matchers have enhanced the ability to match features across images. Instead of relying on nearest-neighbor matching, modern approaches, such as LoFTR [12], SuperGlue [13], and LightGlue [5], utilize attention-based graph neural networks to consider the global context of all features in both images simultaneously. This helps them find accurate matches even in low-texture or repetitive scenes that are difficult for traditional methods. Our work integrates these learned feature extractors and matchers into a geometrically constrained SfM pipeline to overcome the challenges of our setup.

## C. Radiance Fields and Novel View Synthesis

While SfM captures sparse geometry, it does not provide photorealistic renderings. Earlier pipelines filled this gap with Multi-View Stereo (MVS) [14], surface reconstruction, and texture mapping. More recently, Neural Radiance Fields (NeRFs) [2] have shown exceptional capability in synthesizing novel views by learning a volumetric scene representation with a neural network.

NeRFs can produce realistic 3D scenes that capture fine details, such as reflections and translucency. However, they are computationally expensive, often requiring many hours to train and rendering at only a few frames per minute. Subsequent NeRF variants reduced per-scene training time; for example, Instant-NGP uses a multi-resolution hash encoding and optimized kernels for faster training and interactive rendering [15]. 3D Gaussian splatting (3D-GS), by contrast, is an explicit SfM-seeded representation that models scenes as anisotropic Gaussians and uses visibility-aware splatting to enable fast optimization and real-time rendering, a trade-off well suited to our drive-over undercarriage workflow [3], [16].

## D. Gaussian Splatting

3D Gaussian splatting (3D-GS) [3] offers a real-time alternative to NeRFs, achieving comparable or better visual fidelity with faster training and rendering. Instead of an implicit neural representation, 3D-GS models the scene explicitly as a large collection of 3D Gaussians.

Each Gaussian is defined by a set of optimizable attributes: its position (mean Âµ), shape and orientation (a 3Ã3 covariance matrix Î£), color (represented by spherical harmonics), and opacity (Î±). These Gaussians are initialized from the sparse point cloud generated from SfM. During training, their attributes are optimized to reconstruct the training images.

Unlike NeRF, 3D-GS does not need to trace rays through the scene. Instead, it projects each Gaussian directly onto the images, like soft 2D dots (âsplatsâ) and combines them using alpha blending. This is done efficiently using a fast rasterizer. This results in fast training time and the ability to render in real-time.

As 3D-GS relies on an initial SfM point cloud, the quality of the initial reconstruction is critical. This directly motivates our rig-aware SfM pipeline, designed to handle low-parallax, wide-angle scenes like vehicle undercarriages.

## III. METHODOLOGY

Our pipeline, illustrated in Fig. 1, converts raw videos from multiple camera views into a high-quality, interactive 3D model of a vehicleâs undercarriage. There are four major steps in our pipeline. The first step (Sec. III-A) is precise camera calibration to correct severe lens distortion. This calibration only needs to be performed once; the calibrated camera parameters can be used to reconstruct 3D models of any vehicle undercarriage. The second step (Sec. III-B) is to synchronize the videos from the three cameras to ensure spatial-temporal correspondence across the three cameras. The third step (Sec. III-C) is rig-aware Structure-from-Motion (SfM) to estimate the 3D geometry and camera poses. The last step (Sec. III-D) is 3D Gaussian splatting to generate a photorealistic, interactive 3D model of the vehicle undercarriage. Each step is designed to address the challenges posed by wide-angle lenses or lowparallax views.

<!-- image-->  
Fig. 1: 3D reconstruction pipeline overview. (1) We perform a one-time camera calibration using a ChArUco board to model and correct for severe wide-angle lens distortion. (2) For each vehicle, we synchronize the raw videos from the three cameras on our rig to ensure spatial-temporal alignment. (3) From the synchronized videos, we uniformly sample the sharpest frames triplets, undistort the frames, extract DISK features, apply a constrained feature matching strategy to find matches across different frames via LightGlue, and generate the 3D sparse point cloud of the undercarriage via bundle adjustment. (4) Finally, the point cloud is used to initialize the 3D Gaussians, which are optimized to produce the interactive 3D undercarriage model.

## A. Camera Calibration

Wide-angle lenses, while necessary for capturing the undercarriage from a short distance, introduce severe radial and tangential distortion; without accurate camera intrinsic parameters, these non-linear distortions propagate into the SfM stage and warp the sparse point cloud. While SfM frameworks such as COLMAP can estimate camera intrinsic parameters, using a dedicated calibration pipeline produces noticeably lower projection error and reduces bundle-adjustment drift.

1) ChArUco Board: We use a ChArUco board (Fig. 2), a chessboard pattern in which every second square is replaced by an ArUco marker. ChArUco boards provide precise sub-pixel corner detection of chessboard corners and robust ID-based detection of ArUco tags, resulting in lower projection error compared to marker-only or plain chessboard patterns [17]. Our board contains 53Ã37 inner squares of side length 22 mm, with each ArUco marker being 16 mm wide. We capture the calibration videos by sweeping the board through the three wide-angle cameras, varying roll, pitch, yaw, and distance (â 30â120 cm), ensuring each camera observes large baseline changes.

<!-- image-->  
Fig. 2: Sample of ChArUco board [18].

2) Frame Curation: We sample a diverse and sharp set of calibration images by sliding through a window of ten consecutive images from each video. Within each sliding window, the images are converted to grayscale and scored by the variance of the Laplacian (Eq. 1). The Laplacian highlights regions of rapid intensity change, making its variance an effective metric for edge acuity and overall image sharpness.

$$
\mathrm { V a r } \big ( \nabla ^ { 2 } I \big ) = \frac { 1 } { M N } \sum _ { x = 1 } ^ { M } \sum _ { y = 1 } ^ { N } \big ( \nabla ^ { 2 } I ( x , y ) - \mu _ { \nabla ^ { 2 } } \big ) ^ { 2 }\tag{1}
$$

$$
{ \mathrm { w h e r e } } , \mu _ { \nabla ^ { 2 } } = { \frac { 1 } { M N } } \sum _ { x = 1 } ^ { M } \sum _ { y = 1 } ^ { N } \nabla ^ { 2 } I ( x , y )
$$

Higher values indicate crisper edges and less blur. From each ten-frame window, we sample the frame with the highest Laplacian variance, ensuring we sample the sharpest frames with minimal motion blur.

3) Camera Models and Optimization: The projection of a 3D point $P = [ X , Y , Z ] ^ { \top }$ to a 2D pixel coordinate $\boldsymbol { p } = [ u , v ] ^ { \top }$ is modeled in two steps. First, the 3D point is projected to normalized, undistorted image coordinates $p ^ { \prime } = [ x ^ { \prime } , y ^ { \prime } ] ^ { \intercal }$ , where $x ^ { \prime } = X / Z$ and $y ^ { \prime } = Y / Z$ . Second, a distortion function is applied to $p ^ { \prime }$ to get distorted coordinates $P _ { d } = [ x _ { d } , y _ { d } ] ^ { \top }$ , which are then mapped to pixel coordinates using the camera intrinsic matrix K (Eq. 2).

$$
\mathbf { K } = { \left[ \begin{array} { l l l } { f _ { x } } & { ~ 0 } & { ~ c _ { x } } \\ { 0 } & { ~ f _ { y } } & { ~ c _ { y } } \\ { 0 } & { ~ 0 } & { ~ 1 } \end{array} \right] }\tag{2}
$$

We fit the eight-parameter Full OpenCV camera model (Eq. 3) using the Levenberg-Marquardt optimizer [19] to find the distortion parameters for our lenses. This model combines radial distortion (terms with $k _ { i } )$ and tangential distortion (terms with $p _ { i } )$ , which accounts for decentering and non-parallel lens elements. It extends the standard OpenCV model with three additional radial distortion coefficients $( k _ { 4 } , k _ { 5 } , k _ { 6 } )$ to better fit the complex distortion profiles of extreme wide-angle lenses.

$$
\begin{array} { l } { { x _ { d } = x ^ { \prime } \displaystyle \frac { 1 + k _ { 1 } r ^ { 2 } + k _ { 2 } r ^ { 4 } + k _ { 3 } r ^ { 6 } } { 1 + k _ { 4 } r ^ { 2 } + k _ { 5 } r ^ { 4 } + k _ { 6 } r ^ { 6 } } + 2 p _ { 1 } x ^ { \prime } y ^ { \prime } + p _ { 2 } ( r ^ { 2 } + 2 x ^ { \prime 2 } ) } } \\ { { y _ { d } = y ^ { \prime } \displaystyle \frac { 1 + k _ { 1 } r ^ { 2 } + k _ { 2 } r ^ { 4 } + k _ { 3 } r ^ { 6 } } { 1 + k _ { 4 } r ^ { 2 } + k _ { 5 } r ^ { 4 } + k _ { 6 } r ^ { 6 } } + p _ { 1 } ( r ^ { 2 } + 2 y ^ { \prime 2 } ) + 2 p _ { 2 } x ^ { \prime } y ^ { \prime } } } \end{array}\tag{3}
$$

4) Error Metric: The optimization minimizes the Root-Mean-Square (RMS) reprojection error, which measures the Euclidean distance in pixels between the observed corner locations $p _ { i }$ on the ChArUco board and their corresponding projected locations $\hat { p _ { i } } ,$ as computed by the model: $\begin{array} { r } { \mathrm { R M S } \ = \ \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lVert { \bf p } _ { i } - \hat { \bf p } _ { i } \rVert ^ { 2 } } } \end{array}$

This quantitative metric is supplemented by a qualitative evaluation, where we visually inspect the undistorted images to confirm that straight lines in the scene appear straight (Fig. 4). The resulting intrinsic matrix K and the distortion coefficients are then fixed and passed to our SfM pipeline (Sec. III-C).

## B. Video Synchronization

Our camera rig is programmed to trigger video recording simultaneously on all three cameras. However, due to signal and encoding latencies, the resulting videos may be slightly out of sync. We address this issue in two stages: first, we estimate global vertical motion, then we align the motion profiles by minimizing the L1 loss between them.

1) Global Motion Estimation: For every pair of consecutive frames, we compute the number of pixels that moved in the vertical direction using phase correlation. This yields a sequence of vertical displacements, or shifts, denoted $S \in \mathbb { Z } ^ { N - 1 }$ for a video with N frames. We preprocess each frame by applying a Gaussian blur, contrast-limited adaptive histogram equalization (CLAHE), and a Laplacian filter to highlight its key features.

For the three videos captured by the left, center, and right cameras $\{ l , c , r \}$ , we obtain three corresponding shift sequences $\{ S _ { l } , S _ { c } , S _ { r } \}$ of lengths $\left\{ N _ { l } - 1 , N _ { c } - 1 , N _ { r } - 1 \right\}$ , respectively. These shifts provide a compressed signal summarizing global vertical motion over time, which we use for temporal alignment.

2) Offset Search via L1 Loss Minimization: We synchronize the videos by identifying the temporal offsets that minimize the average absolute difference between their shift signals. For any two videos a and b, we define the L1 alignment loss as:

$$
{ \mathcal { L } } ( a , b ) = { \frac { 1 } { \overline { { { N } } } } } \sum _ { i = 1 } ^ { \overline { { { N } } } } | S _ { i } ^ { a } - S _ { i } ^ { b } | , \quad { \mathrm { w h e r e ~ } } \overline { { { N } } } = \operatorname* { m i n } ( N ^ { a } - 1 , N ^ { b } - 1 )\tag{4}
$$

As we increase or decrease the offset, the error moves monotonically towards the minimum (Fig. 3c), thus, we treat this as a convex optimization problem. To optimize the error, we follow an iterative method. First, we explore uniformly within a fixed boundary of offsets shown in the blue dots in Fig. 3c. Then, we pick the offset with the minimum error and explore within a smaller boundary shown in the orange dots. We repeat this process until we reach the minimum with no unexplored offsets within the boundary for that iteration.

For three videos, we use the center camera as the temporal reference and compute offsets for the left and right videos independently. Once aligned, we trim the excess frames to equalize video lengths, resulting in synchronized sequences of length $N _ { \mathrm { s y n c e d } } .$

## C. Structure-from-Motion (SfM)

We pass the synchronized videos and camera intrinsic parameters to our SfM pipeline to generate a point cloud that represents the vehicleâs undercarriage and compute precise camera poses. This sparse point cloud will then serve as the seed for Gaussian splatting (Sec. III-D). Our SfM pipeline is specifically tailored to address the challenges of a lowparallax, wide-angle capture by integrating a robust, learningbased feature matching workflow with strong geometric priors from our camera rig. Key algorithmic choices are detailed below:

1) Frame Selection and Image Undistortion: Given the vehicleâs speed and proximity to the cameras, motion blur can degrade image quality. To reduce computation and improve quality, we uniformly sample the sharpest $k ~ = ~ 2 5 0$ frame triplets from the synchronized left, center, and right camera videos. Sharpness is scored by the average Laplacian variance of the frame triplets (Eq. 5).

<!-- image-->

(a) Shifts for 2 videos before syncing.  
<!-- image-->  
(b) Shift alignment after synchronization.

<!-- image-->  
(c) L1 loss at various offsets. The minimum point shows optimal synchronization.  
Fig. 3: Video synchronization of two videos using phase correlation and L1 loss minimization.

$$
\mathrm { S c o r e } ( I ) = \frac { 1 } { 3 } ( \mathrm { V a r } \big ( \nabla ^ { 2 } I _ { l } \big ) + \mathrm { V a r } \big ( \nabla ^ { 2 } I _ { c } \big ) + \mathrm { V a r } \big ( \nabla ^ { 2 } I _ { r } \big ) )\tag{5}
$$

where $I _ { l } , \ I _ { c } ,$ and $I _ { r }$ are the gray-scale images from the left, center, and right cameras, respectively, and $\nabla ^ { 2 } I$ is the discrete Laplacian of the gray-scale image $I .$

A key step in our pipeline is the initial correction of severe lens distortion. We undistort each selected frame using the calibrated camera parameters obtained from our one-time ChArUco calibration process (Sec. III-A). This pre-processing step rectifies the images, ensuring that features are detected in a geometrically correct space.

2) Feature Extraction: To enhance local contrast and reveal features in darker, underexposed areas of the undercarriage, we apply CLAHE, which adds 15 â 25 % more features.

We utilize DISK, a learned local feature descriptor, for its robustness in challenging conditions. It is trained on a massive dataset of images to detect and describe features that are highly repeatable and reliable across significant viewpoint and illumination changes, which are common in our undercarriage capture environment. We extract up to 8192 DISK features from each pre-processed frame.

3) Constrained Feature Matching: Standard SfM approaches often rely on exhaustive or sequential matching, which can be computationally expensive and prone to error in scenes with repetitive structures or challenging motion from multi-camera rigs. We leverage the known spatiotemporal structure of our capture process to implement a highly efficient and robust constrained matching strategy.

For each image triplet at index i, we consider a temporal window $\mathcal { W } _ { 5 } ( i ) = \{ i - 5 , \dots , i + 5 \}$ and create three classes of pairs:

$$
\begin{array} { r l r } & { \mathrm { ( i ) ~ i n t r a - c a m e r a : ~ } ( c , i )  ( c , j ) , \quad c \in \{ \mathrm { L } , \mathrm { C } , \mathrm { R } \} , \quad j \in \mathcal { W } _ { 5 } ( i ) , } & \\ & { \mathrm { ( i i ) ~ c r o s s ~ } \mathrm { L } \to \mathrm { C } : ( \mathrm { L } , i )  ( \mathrm { C } , j ) , } & { j \in \mathcal { W } _ { 5 } ( i ) , } \\ & { \mathrm { ( i i i ) ~ c r o s s ~ } \mathrm { C } \to \mathrm { R } : ( \mathrm { C } , i )  ( \mathrm { R } , j ) , } & { j \in \mathcal { W } _ { 5 } ( i ) . } \end{array}
$$

We intentionally omit direct matching between the left (L) and right (R) camera views because the relatively large horizontal distance (62 cm baseline) between these cameras, compared to the short vertical distance (12 â 30 cm) from the camera to the vehicle undercarriage, creates extreme perspective differences, making accurate feature matching difficult. The intra-camera matches help track camera motion over time, whereas the cross-camera matches provide the stereo baseline necessary for robust triangulation. The feature descriptors from these constrained pairs are matched using LightGlue, which uses an attention-based graph neural network to find correspondences. LightGlue considers the global context of all features in both images, allowing it to produce a highly accurate set of raw matches with few outliers. These raw matches are then imported into COLMAP, where a final geometric verification is performed using RANSAC [20] to filter any remaining outliers and ensure consistency with our calibrated camera models. This two-stage approach, first constraining the search space spatiotemporally, then employing a high-fidelity learned matcher, synergistically accelerates the matching process while minimizing erroneous correspondences.

4) Rig-Aware Sparse Point Cloud Generation: We use an incremental SfM method that starts with a strong image pair and progressively adds more views. A key part of our methodology is integrating the known camera rig geometry as a prior within the bundle adjustment (BA) optimization.

After triangulating 3D points, BA refines the camera poses and 3D structure by minimizing the total reprojection error (Eq. 6). The objective function is:

$$
J = \operatorname* { m i n } _ { \mathbf { C } _ { i } , \mathbf { P } _ { j } } \sum _ { i , j } w _ { i j } \left\| \pi ( \mathbf { C } _ { i } , \mathbf { P } _ { j } ) - \mathbf { p } _ { i j } \right\| ^ { 2 }\tag{6}
$$

where $\mathbf { C } _ { i }$ is the pose of camera $i , \mathbf { P } _ { j }$ is the 3D position of point $j , \mathbf { p } _ { i j }$ is the observed projection of point j in camera $i ,$ and $w _ { i j }$ is a binary term indicating visibility.

To enforce the known camera rig geometry, we augment this objective function with a regularization term based on the relative camera positions and orientations on our camera rig. As per our camera rig schematics, we provide a pose prior for the left and right cameras relative to the center camera.

$$
\mathbf { t } _ { \mathrm { L - C } } = [ - 0 . 3 1 , 0 , 0 ] ^ { \top } , \quad \mathbf { t } _ { \mathrm { C - R } } = [ + 0 . 3 1 , 0 , 0 ] ^ { \top }
$$

This rig-aware BA prevents inter-camera drift, resulting in an accurate sparse point cloud.

## D. Gaussian Splatting

We pass the sparse point cloud generated from our SfM pipeline (Sec. III-C) along with the undistorted images to the 3D Gaussian splatting framework introduced by Kerbl et al. [3] to transform the discrete set of 3D points into a dense radiance representation that can be rendered from any viewpoint.

Each point in the sparse point cloud is initialized as a Gaussian:

$$
G _ { i } ( { \bf x } ) = \alpha _ { i } \exp \Bigl [ - \frac { 1 } { 2 } ( { \bf x } - { \pmb \mu } _ { i } ) ^ { \sf T } { \pmb \Sigma } _ { i } ^ { - 1 } ( { \bf x } - { \pmb \mu } _ { i } ) \Bigr ]\tag{7}
$$

where $\pmb { \mu } _ { i } \in \mathbb { R } ^ { 3 }$ is the mean position, $\alpha _ { i }$ is its initial opacity, and $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ is a diagonal covariance matrix scaled by the local point density.

The 3D Gaussian parameters are then optimized, yielding a dense radiance representation of the scene. For rendering, each 3D Gaussian $G _ { i }$ is projected onto the image space as a 2D Gaussian $G _ { i } ( \mathbf { u } )$ , where, $\mathbf { u } = ( u , v )$ . Visibility and color are composited via front-to-back alpha blending:

$$
\mathbf { C } ( \mathbf { u } ) = \sum _ { i = 1 } ^ { N } w _ { i } ( \mathbf { u } ) \mathbf { c } _ { i } , \quad w _ { i } ( \mathbf { u } ) = \tilde { \alpha } _ { i } ( \mathbf { u } ) \prod _ { j < i } \left[ 1 - \tilde { \alpha } _ { j } ( \mathbf { u } ) \right]\tag{8}
$$

The SfM points are used to initialize the means $\mu _ { i } ,$ colors $c _ { i }$ are initialized based on the color of the associated image pixel, and covariances $\Sigma _ { i }$ are initialized as isotropic $\sigma ^ { 2 } \mathbf { I }$

## IV. EXPERIMENTS

This section evaluates each stage of our pipeline and highlights the design choices that enable high-fidelity 3D Gaussian splatting.

We tested our system using ten vehicles of varying makes, models, and undercarriage conditions (e.g., new, rusted, damaged), recorded with our wide-angle camera rig. Each vehicleâs drive-through produces three videos (left, center, right) with a resolution of 1920 Ã 1080 at 120 frames per second (fps). The videos are â 8 seconds long, giving us â 960 frames per camera.

All experiments were conducted on a workstation equipped with an AMD EPYC 7763 CPU, 2 TB of RAM, and eight NVIDIA RTX A6000 GPUs.

## A. Camera Calibration

Correcting the extreme distortion introduced by wide-angle lenses requires accurate camera calibration. An inaccurate camera model would propagate errors through the entire reconstruction pipeline, leading to warped point clouds and distorted renders. We evaluated the Full OpenCV camera model by fitting it to our curated set of sharp ChArUco board images. The eight-parameter Full OpenCV model, which accounts for higher-order radial distortion terms and tangential distortion, achieved a low RMS projection error of 0.74 pixels.

Fig. 4 shows how the Full OpenCV camera model corrects severe distortion in calibration images. Fig. 5 shows the same correction on a real undercarriage image. Accurate intrinsic parameters minimize bundle-adjustment drift in the SfM pipeline, resulting in geometrically accurate point clouds.

<!-- image-->  
Fig. 4: Qualitative comparison of original calibration image (left) and undistorted image using Full OpenCV model (right).

<!-- image-->  
Fig. 5: Qualitative comparison of original undercarriage image (left) and undistorted image using Full OpenCV model (right).

## B. Video Synchronization

Even with a hardware trigger, minor latencies can lead to temporal misalignment between the three video streams. Our two-stage synchronization algorithm is designed to correct these offsets with sub-frame precision.

The videos from the three cameras could be out of sync by up to Â± 35 frames. After synchronization, the average difference in vertical motion between camera pairs drops from 774 pixels to â 22 pixels.

The algorithm uses a convex loss function, which helps reliably find the best alignment. Precise synchronization ensures spatial-temporal correspondence for triplet selection, thereby preventing ghost artifacts in the Gaussian splat.

## C. Structure-from-Motion

The SfM stage generates the geometric foundation for our model. It outputs a sparse 3D point cloud and camera poses, which set an upper bound on the quality of the Gaussian splat. To validate our methodology, we perform a comprehensive evaluation comparing our proposed pipeline against two key baselines: a vanilla SfM approach and a version of our rigaware pipeline using classic SIFT features and COLMAPâs matcher instead of DISK+LightGlue. We then conduct a series of ablation studies on our pipeline to demonstrate the impact of each of our core contributions.

The quantitative results are detailed in Table I. The vanilla Baseline SfM, which does not include our calibration, synchronization, custom matching strategy, and rig-based pose priors, fails to produce a coherent reconstruction. Our rig-aware pipeline using classic SIFT features generates a strong baseline, significantly improving all metrics. By integrating learned features (DISK) and an attention-based matcher (LightGlue), our proposed methodology achieves the best overall performance, yielding the densest and most accurate sparse point cloud. The ablation studies highlight the importance of each component in our final pipeline. Removing either the dedicated camera calibration (and thus pre-undistortion), the custom matching strategy, or the rig-based pose priors results in degenerate point clouds that do not represent the vehicle undercarriage, and thereby result in degenerate Gaussian splats as shown in Fig. 7.

<!-- image-->  
Fig. 6: Sparse point cloud for a vehicleâs undercarriage.

TABLE I: SfM baseline comparison and ablation study. Each ablation removes a specific component from our SfM pipeline.
<table><tr><td>Metric</td><td>Baseline SfM*</td><td>Rig-Aware SfM (SIFT)</td><td>- Camera Calibration*</td><td>- Video Sync</td><td>- Custom Matching*</td><td>. Pose Priors*</td><td>Our SfM DISK+LG</td></tr><tr><td>Registered Images â</td><td>749</td><td>750</td><td>750</td><td>750</td><td>745</td><td>673</td><td>750</td></tr><tr><td>Sparse 3D Points â</td><td>126,496</td><td>323,005</td><td>189,121</td><td>339,209</td><td>368,751</td><td>313,395</td><td>427,299</td></tr><tr><td>Mean Track Length â</td><td>10.0868</td><td>8.6509</td><td>6.8651</td><td>8.1906</td><td>14.0779</td><td>8.0244</td><td>8.6674</td></tr><tr><td>Reprojection Error (px) â</td><td>0.7817</td><td>0.5376</td><td>0.7058</td><td>0.4995</td><td>0.5279</td><td>0.5478</td><td>0.4909</td></tr></table>

\* These SfM approaches yield a degenerate sparse point cloud, which doesnât represent a vehicle undercarriage. Consequently, these sparse point clouds produce a degenerate Gaussian splat.

As shown in Fig. 6, our final method generates a dense, coherent representation of the undercarriage, providing a solid geometric base for Gaussian splatting.

## D. Gaussian Splatting

Finally, we evaluate the Gaussian splatting results of the 3D vehicle undercarriage. To evaluate our results, we use standard image quality metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and the perceptual Learned Perceptual Image Patch Similarity (LPIPS) metric.

As detailed in Table II, our pipeline produces good results, with a mean PSNR of 30.66 dB, a mean SSIM of 0.92, and a mean LPIPS of 0.19, indicating that our rendered views are visually and perceptually closer to the real camera images. This demonstrates a clear improvement over a version of our pipeline that uses classic SIFT features and matcher, indicating that the denser and more accurate point cloud from the DISK+LightGlue approach provides a superior seed for the Gaussian splatting optimization. As visualized in Fig. 7, the baseline and several ablated SfM initializations result in degenerate, noisy, or incomplete renders, confirming that our pipeline is crucial for achieving a high-quality 3D model.

Our method achieves real-time rendering performance, averaging over 130 frames per second (fps), making it suitable for interactive inspection. Fig. 9 shows that our models are photorealistic and geometrically coherent. The high-quality SfM initialization allows the 3D-GS optimization to converge to a sharp representation, capturing fine details such as bolt heads, rust patterns, fluid marks, and the metallic sheen of exhaust systems (Fig. 8). Notably, the models are largely free of the âfloaterâ artifacts and hazy regions that often plague radiance fields trained from less accurate camera poses.

<!-- image-->  
Without Custom Matching

<!-- image-->  
Without Pose Priors  
Fig. 7: The baseline SfM and ablations for camera calibration, custom matching, and pose priors all result in degenerate, noisy, or incomplete renders. These results visually confirm the quantitative findings in Table II and demonstrate that each component of our proposed pipeline is essential for generating a coherent and high-fidelity 3D model.

Training takes â 8 â 10 minutes on an RTX A6000 GPU. Efficient training and a high-fidelity 3D model enable us to view the vehicleâs undercarriage from different angles, identifying potential issues such as corrosion, leaks, or damage, making it practical for production use.

TABLE II: Gaussian splatting baseline comparison and ablation study. Each ablation removes a specific component from our SfM pipeline.
<table><tr><td>Metric</td><td>Baseline SfM*</td><td>Rig-Aware SfM (SIFT)</td><td>- Camera Calibration*</td><td>- Video Sync</td><td>Custom Matching*</td><td>Pose Priors*</td><td>Our SfM DISK+LG</td></tr><tr><td>PSNR (dB) â</td><td>23.46</td><td>29.68</td><td>21.40</td><td>29.47</td><td>20.78</td><td>16.96</td><td>30.66</td></tr><tr><td>SSIM â</td><td>0.78</td><td>0.91</td><td>0.75</td><td>0.89</td><td>0.76</td><td>0.68</td><td>0.92</td></tr><tr><td>LPIPS â</td><td>0.42</td><td>0.21</td><td>0.57</td><td>0.26</td><td>0.43</td><td>0.66</td><td>0.19</td></tr></table>

## V. CONCLUSION

In this paper, we presented a complete pipeline for generating high-quality, interactive 3D models of vehicle undercarriages using multi-view video. Our system is designed to address the practical challenges of under-vehicle inspection, which is often tedious, physically demanding, and especially difficult to scale for online marketplaces. By converting raw videos into detailed, photorealistic 3D models, our method enables faster, safer, and more transparent vehicle assessments.

<!-- image-->  
Fig. 8: Gaussian splat render highlighting the ability of our pipeline to capture fine-grained textures and diagnostic details. The render shows areas of significant rust on the frame. This level of detail enables accurate vehicle condition assessments by inspectors and increases buyer confidence in online marketplaces.

<!-- image-->  
Fig. 9: Photorealistic 3D undercarriage models produced by our pipeline. The top image shows a comprehensive overhead view of the vehicleâs undercarriage, while the bottom images provide different perspectives of the rear.

The core of our approach is a rig-aware SfM pipeline, built specifically to support 3D Gaussian splatting. We found that accurately handling wide-angle lens distortion and utilizing physical priors from the camera rig are key to recovering reliable geometry in challenging settings, such as low parallax and linear motion. This solid geometric foundation enabled the generation of sharp, artifact-free radiance fields.

Our experiments show that the pipeline performs well, producing high-quality renderings (PSNR > 30, SSIM > 0.90)

at real-time speeds (> 130 FPS). The resulting 3D models reveal diagnostic features like rust, leaks, and surface wear, making the tool valuable for vehicle inspectors and consumers. We see this work as a foundation for future research in automated 3D damage detection, with potential applications in other industrial inspection settings.

## REFERENCES

[1] S. Chauhan and L. J. Katare, âUsed cars market size, share, competitive landscape and trend analysis report, by vehicle type, by fuel type, by distribution channel: Global opportunity analysis and industry forecast, 2023â2033,â Allied Market Research, Tech. Rep. A06429, Feb. 2025, [Online; accessed 19-Jul-2025]. [Online]. Available: https://www.alliedmarketresearch.com/used-cars-market-A06429

[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[3] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[4] M. Tyszkiewicz, P. Fua, and E. Trulls, âDisk: Learning local features with policy gradient,â Advances in neural information processing systems, vol. 33, pp. 14 254â14 265, 2020.

[5] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, âLightglue: Local feature matching at light speed,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 17 627â17 638.

[6] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,âÂ¨ in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[7] L. Pan, D. Barath, M. Pollefeys, and J. L. Sch Â´ onberger, âGlobal Â¨ structure-from-motion revisited,â in European Conference on Computer Vision. Springer, 2024, pp. 58â77.

[8] N. Snavely, S. M. Seitz, and R. Szeliski, âPhoto tourism: exploring photo collections in 3d,â in ACM siggraph 2006 papers, 2006, pp. 835â846.

[9] C. Mei and P. Rives, âSingle view point omnidirectional camera calibration from planar grids,â in Proceedings 2007 IEEE International Conference on Robotics and Automation. IEEE, 2007, pp. 3945â3950.

[10] R. Hartley and A. Zisserman, Multiple view geometry in computer vision. Cambridge university press, 2003.

[11] D. G. Lowe, âDistinctive image features from scale-invariant keypoints,â International journal of computer vision, vol. 60, no. 2, pp. 91â110, 2004.

[12] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, âLoftr: Detectorfree local feature matching with transformers,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 8922â8931.

[13] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperglue: Learning feature matching with graph neural networks,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4938â4947.

[14] M. Goesele, B. Curless, and S. M. Seitz, âMulti-view stereo revisited,â in 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPRâ06), vol. 2. IEEE, 2006, pp. 2402â2409.

[15] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM transactions on graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[16] B. Fei, J. Xu, R. Zhang, Q. Zhou, W. Yang, and Y. He, â3d gaussian splatting as new era: A survey,â IEEE Transactions on Visualization and Computer Graphics, 2024.

[17] G. H. An, S. Lee, M.-W. Seo, K. Yun, W.-S. Cheong, and S.-J. Kang, âCharuco board-based omnidirectional camera calibration method,â Electronics, vol. 7, no. 12, p. 421, 2018.

[18] OpenCV Development Team, âDetection of ChArUco Boards,â Online. [Online]. Available: https://docs.opencv.org/3.4/df/d4a/tutorial charuco detection.html

[19] A. Ranganathan, âThe levenberg-marquardt algorithm,â Tutoral on LM algorithm, vol. 11, no. 1, pp. 101â110, 2004.

[20] R. Schnabel, R. Wahl, and R. Klein, âEfficient ransac for point-cloud shape detection,â in Computer graphics forum, vol. 26, no. 2. Wiley Online Library, 2007, pp. 214â226.