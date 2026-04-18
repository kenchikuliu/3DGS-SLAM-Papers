# 3DGS-ReLoc: 3D Gaussian Splatting for Map Representation and Visual ReLocalization

Peng Jiang1, Gaurav Pandey 2 and Srikanth Saripalli1

Abstractâ This paper presents a novel system designed for 3D mapping and visual relocalization using 3D Gaussian Splatting. Our proposed method uses LiDAR and camera data to create accurate and visually plausible representations of the environment. By leveraging LiDAR data to initiate the training of the 3D Gaussian Splatting map, our system constructs maps that are both detailed and geometrically accurate. To mitigate excessive GPU memory usage and facilitate rapid spatial queries, we employ a combination of a 2D voxel map and a KD-tree. This preparation makes our method well-suited for visual localization tasks, enabling efficient identification of correspondences between the query image and the rendered image from the Gaussian Splatting map via normalized crosscorrelation (NCC). Additionally, we refine the camera pose of the query image using feature-based matching and the Perspective-n-Point (PnP) technique. The effectiveness, adaptability, and precision of our system are demonstrated through extensive evaluation on the KITTI360 dataset.

## I. INTRODUCTION

The rapid evolution of autonomous driving and robotic navigation technologies has underscored the critical importance of advanced scene reconstruction methodologies. These technologies rely heavily on the integration of data from diverse sensor modalities to create describable and accurate representations of the environment. Among various sensor fusion techniques, the combination of LiDAR and camera data is particularly noteworthy. This fusion harnesses LiDARâs precise depth sensing capabilities alongside the rich visual details captured by cameras, a synergy crucial for achieving the level of environmental understanding necessary for autonomous systems to navigate safely and efficiently. However, the challenge lies in harmonizing these different types of data into a unified, detailed, and geometrically accurate representation of the scene, a task that is both complex and essential for traversing intricate urban landscapes.

This paper introduces 3DGS-ReLoc, a novel system tailored for visual relocalization in autonomous navigation, employing 3D Gaussian Splatting (3DGS) as its primary map representation technique[1]. Utilizing LiDAR data, our method initiates the training of the 3D Gaussian Splatting representation, enabling the generation of large-scale, geometry-accurate maps. This initial training with LiDAR significantly improves our systemâs ability to create detailed and precise environmental models, which is essential for advanced perception systems in autonomous vehicles. Moreover, to address the high GPU memory consumption challenge, we adopt a strategy of dividing 3D Gaussian Splatting maps into 2D voxels and utilizing a KD-tree for efficient spatial querying.

3D Gaussian Splatting representation can generate highfidelity images and depth data in association with known camera poses within the mapâs coordinates. This capability simplifies our method by facilitating the straightforward identification of correspondences between the query image and the Gaussian Splatting map through image feature detection and matching techniques. Additionally, by leveraging the depth information and its corresponding camera pose, we can accurately determine the camera pose of the query image. Implementing 3D Gaussian Splatting for visual relocalization not only showcases the adaptability of our method but also effectively tackles the complexities involved in fusing sensor data, contributing to the development of more precise and efficient scene representation techniques.

We conducted an extensive evaluation of our methodology with the KITTI360 dataset [2]. This dataset was chosen for its comprehensive annotations, which aid in creating accurate maps in diverse urban landscapes. Our results highlight our systemâs effectiveness, versatility, and precision. Specifically, we showcase the utility of 3D Gaussian Splatting for scene representation in visual relocalization tasks.

## II. RELATED WORK

## A. Map Representation

Maps are crucial for robot navigation and autonomous driving, with traditional representations including voxel grids, point clouds, and meshes, as highlighted in recent literature [3]. The advent of neural rendering techniques has introduced a new avenue for constructing maps with high fidelity. These models capture and depict 3D scenes by utilizing images and corresponding poses for guidance. This approach enables synthesizing high-fidelity images from novel views of the scene. Among these, Neural Radiance Fields (NeRF)[4] has gained prominence. It encodes the radiance fields of complex 3-D scenes into the weights of multilayer perceptrons (MLPs), demonstrating exceptional realism in rendering 3-D environments through volume rendering under 2-D supervision. This innovation has significantly contributed to the development of mapping systems and the enhancement of SLAM (Simultaneous Localization and Mapping) systems, including iMAP[5] and NICE-SLAM[6]. iMAP, for instance, employs an MLP for realtime scene representation within a SLAM framework, while

NICE-SLAM introduces a dense, efficient, and robust SLAM approach by integrating multilevel local scene information and optimizing with geometric priors for better detail in large indoor scenes.

However, the scene-specific nature of networks trained with NeRF, where each 3-D sceneâs representation is encoded in an MLPâs weights, restricts their generalizability across different environments. Furthermore, the computational intensity of NeRF-based methods results in slow rendering times. 3D Gaussian Splatting [1] has emerged as a viable alternative, providing an explicit representation more in line with traditional mapping approaches and enabling easier integration of conventional methods with minimal adjustments. This approach not only accelerates training times but also maintains high-quality visuals akin to NeRF. Recent efforts to apply 3D Gaussian Splatting to SLAM [7], [8], [9], [10] have shown promise. SplaTAM [10] represents an innovative application of 3D Gaussian splatting in SLAM, offering dense SLAM capabilities with monocular RGB-D cameras and enabling online camera pose tracking through singular 3D Gaussian Splatting map. Building on these efforts, [9] introduced SGS-SLAM, which incorporates 3D semantic segmentation into the GS-SLAM system. This method uses multi-channel optimization during mapping to combine appearance, geometric, and semantic constraints with key-frame optimization, enhancing the quality of reconstruction. Despite these advancements, the focus of research remains predominantly on indoor scenes of limited size, utilizing RGB-D cameras to generate dense point clouds. Several studies have been conducted on outdoor large-scale 3D Gaussian Splatting reconstruction [11], [12], [13], [14]. However, these studies primarily focus on generating highquality images [11] or handling dynamic scenarios in street data [14], [13] for simulation purposes, and do not explore their potential for map representation and relocalization.

## B. Visual Relocalization

Visual relocalization aims to estimate a cameraâs position and orientation from a single query image. Approaches to visual relocalization vary, including feature-based methods, scene coordinate regression, pose regression, and direct image alignment. DSAC [15] exemplifies the scene coordinate regression method, circumventing the need for explicit 3D mapping by mastering a pixel-to-point transformation through differentiable RANSAC for seamless end-to-end learning. In the realm of pose regression methods, the notable work by Laskar et al. [16] stands out. They leverage Convolutional Neural Networks (CNNs) to identify similar images within a database and calculate relative poses, employing RANSAC to enhance accuracy. Meanwhile, PixLoc [17] serves as a prime example of direct image alignment, utilizing deep multiscale features. PixLoc redefines localization as a metric learning challenge, facilitating comprehensive endto-end training.

Despite the variety of methods, 2D-3D feature-based approaches remain predominant. 2D-3D feature-based approaches aims to estimate a cameraâs position and orientation (pose) from a 2D image within a previously mapped 3D scene. The construction of these 3D models typically involves Structure-from-Motion (SfM) with color images [18], Truncated Signed-Distance Function (TSDF) from range images [19], or LiDAR-based mapping techniques [20]. They compute the camera pose by matching 2D-3D correspondences through local feature descriptors. Since these descriptors often depend on the original imaging angle, research has focused on creating viewpoint-independent features [21] or learning across different modalities, such as with P2- Netâs unified descriptor for pixel-point matching [22]. Contrastive learning has been explored to bridge the gap between camera images and LiDAR point clouds [23]. Additionally, approaches like that of Wolcott et al.[20] propose localizing a camera within a 3D LiDAR-generated prior ground map by maximizing normalized mutual information between real camera measurements and generated synthetic LiDAR intensity image. Compared to traditional map representations, the 3D Gaussian Splatting representation has a more direct linkage between images, as it enables the rendering of new images and depth maps from novel viewpoints. This capability facilitates mitigating the challenges associated with view dependence, enhancing our ability to manage perspectiverelated difficulties more effectively.

## III. METHOD

This section will first revisit the concept of 3D Gaussian Splatting. Then, we will detail our system, which consists of two main components: the 3D Gaussian Splatting (3DGS) Map Representation and visual relocalization using the 3DGS Map. The complete system is illustrated in Fig.1.

## A. Revisit 3D Gaussian Splatting

The 3D Gaussian Splatting (3DGS) [1] is a rasterization technique designed for real-time rendering of photorealistic scenes using a group of 3D Gaussians for modeling. The original approach unfolds in three steps: a) employing the Structure from Motion (SfM) technique to estimate the poses of a collection of images from the same scene and a sparse point cloud of the scene; b) transforming each point in the cloud into a 3D Gaussian; c) applying Stochastic Gradient Descent (SGD) to refine the Gaussians, allowing for adaptive densification and pruning of the Gaussians based on the gradients and predefined criteria. The following parameters characterize each Gaussian in the model:

â¢ Center of the Gaussian $\mu _ { i } = [ x _ { 1 } , x _ { 2 } , x _ { 3 } ] \in \mathbb { R } ^ { 3 }$ , (usually initialize using sparse point cloud from SfM)

â¢ Covariance matrix of the Gaussian $\Sigma _ { i } = R _ { i } S _ { i } S _ { i } ^ { \top } R _ { i } ^ { \top 1 }$ , comprised of a scaling matrix Si = diag $\left( \left[ s _ { x } , s _ { y } , s _ { z } \right] \right)$ and a rotation matrix ${ \cal R } _ { i } = \mathrm { q } 2 { \bf R } \left( \left[ r _ { w } , r _ { x } , r _ { y } , r _ { z } \right] \right)$ , with q2R converting a quaternion to a rotation matrix.

<!-- image-->  
Fig. 1. Pipeline of 3D Gaussian Splatting for Map Representation and Visual ReLocalization: The process starts by creating a colorized point cloud map from LiDAR scans, images, and poses. This map serves as the initialization for the 3D Gaussian Splatting (3DGS) map, which is incrementally trained on submaps. The 3DGS map is stored as a 2D voxel map, with a KD-tree enabling rapid spatial queries. For relocalization, a submap proximate to the query imageâs coarse pose is selected to render a series of images and depths. The query image is then subjected to a brute-force search against this image sequence to find the closest rendered image and depth. Subsequently, feature-based matching and the Perspective-n-Point (PnP) method are employed to iteratively refine the query imageâs pose, achieving precise localization within the global map.

â¢ RGB color $c _ { i } \in \mathbb { R } ^ { 3 }$ or spherical harmonics (SH) coefficients $c _ { i } \in \mathbb { R } ^ { k }$ , facilitating view-dependent colors with k representing the degrees of freedom;

â¢ Opacity $o _ { i } \in \mathbb { R } .$

Accordingly, a 3D Gaussian is defined as $g _ { i } = [ \mu _ { i } , S _ { i } , R _ { i } , c _ { i } , o _ { i } ]$ and a full 3DGS Map is a set of the Gaussian representation $G = \{ g _ { 0 } , . . . , g _ { N } \}$

To render an image for a camera characterized by the intrinsic matrix K and pose Wt (world-to-camera transformation), the Gaussians are first transformed into camera coordinates. They are then sorted by depth and rendered in a front-to-back sequence using Maxâs volume rendering formula [24]:

$$
C \left( \hat { x } \right) = \sum _ { i \in \mathcal { S } } c _ { i } q _ { i } \left( \hat { x } \right) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - q _ { j } \left( \hat { x } \right) \right)\tag{1}
$$

Here, the final rendered color $C \left( \hat { x } \right)$ at the camera projection plane for pixel Ëx is the weighted sum of each Gaussianâs color $c _ { i } .$ . The weight is calculated using the footprint function $q _ { i }$ derived from the Gaussian kernel [25] (see Eq.2), and is modulated by an occlusion (transmittance) term that accounts for all Gaussians preceding the current one.

$$
q _ { i } ( \hat { x } ) = o _ { i } \frac { 1 } { \left| J ^ { - 1 } \right| \left| W ^ { - 1 } \right| } G _ { \hat { \Sigma } _ { i } ^ { c } } \left( \hat { x } - \hat { \mu } _ { i } \right)\tag{2}
$$

where $G _ { \hat { \Sigma } _ { i } ^ { c } }$ is a Gaussian function with covariance matrix $\hat { \Sigma } _ { i } ^ { c } .$ a $2 \times 2$ matrix obtained by excluding the last row and column from the matrix computed using Eq.3, and $\hat { \mu } = [ x _ { 1 } , x _ { 2 } ]$ is the first two value of mean $\mu$ of this Gaussian.

$$
\Sigma _ { i } ^ { c } = J W \Sigma _ { i } W ^ { \top } J ^ { \top }\tag{3}
$$

Where $J = \partial m ( \mu ) / \partial \mu$ is the Jacobian of the projection formula Eq. 4:

$$
m \left( \mu \right) = K \left( \frac { W \mu } { \left( W \mu \right) _ { z } } \right)\tag{4}
$$

For a comprehensive derivation of the footprint function, readers are directed to [25].

For rendering depth, we can simply replace the color $c _ { i }$ with the $z _ { i } = x _ { 3 }$ of the Gauassian transformed in the camera

coordinate:

$$
D \left( \hat { x } \right) = \sum _ { i \in \mathcal { S } } z _ { i } q _ { i } \left( \hat { x } \right) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - q _ { j } \left( \hat { x } \right) \right)\tag{5}
$$

## B. 3D Gaussian Splatting Map Representation

1) 3D Map Construction and Initialization: Contrary to the original methods that begin with Structure from Motion (SfM) as outlined in [1], our approach initiates the 3D Gaussian process by utilizing the 3D map generated from LiDAR point cloud data and corresponding images. This ensures that our foundational representation possesses accurate geometric information. However, the depth information derived from the LiDAR point cloud is sparse, presenting challenges for subsequent visual localization tasks due to the possibility that keypoints detected at this stage may lack corresponding LiDAR depth. Nevertheless, by leveraging the densification scheme of the 3D Gaussian Splatting method [1], our method can increase the number of underlying Gaussians during the training process. This enhancement allows rendering dense depth from various viewpoints using 3DGS representation. This dense depth can provide precise depth information for our visual localization tasks.

2) Map Storage and Management: The 3D Gaussian Splatting method is known for its high GPU memory consumption, making the representation of large outdoor scenes challenging. To mitigate this, we have opted to use only RGB color information, foregoing the use of Spectral Harmonics (SH) decomposition. While Spherical Harmonics (SH) decomposition aids in capturing lighting and view-dependent effects, it significantly raises memory requirements. Our choice reduces the mapâs memory footprint, but it constrains direct comparisons of rendering quality with methods encoding lighting informationâkey in outdoor settings due to complex light and shadow dynamics (see Section V-A). Our primary aim is establishing a dependable mapping system for visual relocalization, making detailed rendering quality comparisons, particularly regarding lighting effects, beyond this workâs scope.

To efficiently manage and train large-scale 3DGS maps, weâve organized the 3D environment into a 2D voxel grid. This method divides the map into smaller voxels, each storing 3DGS parameters based on the Âµ, and assigns a unique hash ID to every voxel for quick querying. For better efficiency in spatial querying and updating 3DGS parameters in voxels according to the camera pose of each image, we utilize a KD tree. Constructed from the voxelsâ center points, the KD tree swiftly identifies voxels within a certain range of the cameras. This approach not only improves our systemâs scalability but also minimizes the use of computational resources, allowing for the detailed reconstruction of large environments without excessive GPU memory demands.

3) Loss Function: The original 3D Gaussian Splatting technique was developed primarily for novel view synthesis, focusing on producing high-quality images. Consequently, it utilizes a balanced combination $( L _ { r g b } )$ of the Mean Absolute Error loss $\left( L _ { 1 } \right)$ and the Structural Similarity Index Measure (SSIM) loss $\left( L _ { D - S S I M } \right)$ to evaluate the difference between rendered $( ( \mathcal { R } ( G _ { t } ) ) )$ image and actual ground truth image (It ) (see to Eq. 7). This singular focus allows Gaussian Splatting to compromise the precision of rendered depth in favor of enhancing the visual quality of RGB images. Additionally, this leads the densification process to introduce new Gaussians that do not adhere to the underlying geometry.

$$
L _ { r g b } ( I _ { 1 } , I _ { 2 } ) = ( 1 - \lambda ) L _ { 1 } ( I _ { 1 } , I _ { 2 } ) + \lambda L _ { \mathrm { D - S S I M } } ( I _ { 1 } , I _ { 2 } )\tag{6}
$$

$$
L _ { p h o t o } = L _ { r g b } \left( \mathcal { R } \left( G _ { t } \right) , I _ { t } \right)\tag{7}
$$

To address this limitation, we incorporate a re-projection error loss aimed at maintaining both the geometric accuracy of the scene representation and the fidelity of rendered depth. We acquire depth information $( \mathcal { D } ( G _ { t } ) )$ at pose Wt through Gaussian Splatting $( G _ { t } )$ . This depth information is then used to re-project the ground truth image (It ) from pose Wt to pose $W _ { t + 1 }$ through transformation matrix $T _ { t } ^ { t + 1 }$ , and the re-projected image $\dot { ( \mathcal { P } } \left( \mathcal { D } \left( G _ { t } \right) , I _ { t } , T _ { t } ^ { t + 1 } \right) )$ is subsequently compared with the actual ground truth image $\left( I _ { t + 1 } \right)$ at pose $W _ { t + 1 }$ (see Eq. 8).

$$
L _ { r e p r o j } = L _ { r g b } \left( \mathcal { P } \left( \mathcal { D } \left( G _ { t } \right) , I _ { t } , T _ { t } ^ { t + 1 } \right) , I _ { t + 1 } \right)\tag{8}
$$

As a result, our comprehensive loss function is outlined in Eq. 9, with Ï1 and Ï2 serving as weights to balance the contributions of the two loss functions effectively.

$$
L = \omega _ { 1 } L _ { p h o t o } + \omega _ { 2 } L _ { r e p r o j }\tag{9}
$$

## C. Visual ReLocalization Method Using 3DGS Map

1) Initial ReLocalization: Our approach starts with leveraging raw pose data to pinpoint the queryâs location on the global map. This data may come from various sources, including GPS systems. Utilizing the raw pose as a reference, we retrieve a segment of the global 3DGS map most likely to contain the query imageâs precise location.

After selecting the nearby submap, we refine location accuracy through a brute-force search. This involves generating and comparing multiple images from the 3DGS submap with the query image to find the most visually similar one, assuming that similarity indicates closeness in location. This method improves upon GPS accuracy, providing a precise starting point for feature-based matching.

We employ normalized cross-correlation (NCC) [26] for this image comparison, a metric frequently applied in medical image registration to evaluate similarity, as defined below:

$$
\mathrm { N C C } ( I _ { q } , I _ { G _ { t } } ) = \frac { \sum _ { ( i , j ) } ( I _ { q } - \bar { I } _ { q } ) ( I _ { G _ { t } } - \bar { I } _ { G t } ) } { \sqrt { \sum _ { ( i , j ) } ( I _ { q } - \bar { I } _ { q } ) ^ { 2 } } \sqrt { \sum { ( i , j ) } ( I _ { G _ { t } } - \bar { I } _ { G t } ) ^ { 2 } } }\tag{10}
$$

Where $I _ { q }$ represents the query image, $I _ { G _ { t } }$ denotes the image rendered from the 3DGS submap at pose $W _ { t }$ , and Â¯I indicates the mean intensity of the images.

Another rationale behind selecting normalized crosscorrelation is its differentiable nature, which aligns well with the differentiable characteristics of the 3D Gaussian representation. This compatibility has the potential to facilitate a fully differentiable relocalization pipeline, creating avenues for seamless integration and optimization within the overall system. (For a more discussion, please see Section V-B).

2) ReLocalization Refinement: After pinpointing the closest rendered image, we adopt a feature-based matching technique. This phase entails identifying matching points between the query image and the closest rendered counterpart. By harnessing the known camera pose associated with the rendered image, along with the depth rendered from the 3DGS map, we employ the Perspective-n-Point (PnP) algorithm to refine the pose of the query image within the selected submap.

To further enhance the precision of localization, we employ an iterative refinement process on the initially estimated pose. This involves repeatedly performing the feature-based matching step with images newly rendered using the pose estimated from the preceding step. Each cycle is designed to progressively refine the pose estimation, capitalizing on the increased accuracy with each iteration to achieve the most precise localization achievable.

Considering the broad spectrum of available feature detection and matching algorithms[18], we opted for Superpoint [21] for keypoint detection and feature extraction, coupled with LightGlue [27] for the matching process. These choices were driven by their proven effectiveness and compatibility with our localization framework, enabling us to achieve high-quality feature matching and efficient pose recovery, as shown in Section IV.

3) Live Relocalization: In live relocalization task, the system must continuously track a cameraâs pose using streaming images. For the initial query image, we conduct initialization and refinement as outlined in Sections III-C.1 and III-C.2. For images that follow, we adopt a constant velocity model for predicting the cameraâs next pose, further refining the pose with the feature-matching technique described in Section III-C.2. This streamlined approach eliminates the necessity of brute-force searches for every query image, significantly boosting the efficiency of ongoing relocalization efforts. Such efficiency is crucial for real-time applications in fields like autonomous navigation and robotics, providing smoother and more reliable tracking. To enhance pose estimation accuracy, we can incorporate more sophisticated methods, such as filter-based techniques [28].

<!-- image-->

(a) X Error-Normalized Cross Correlation  
<!-- image-->

(b) Query Image  
<!-- image-->

<!-- image-->

(e) Yaw Error-Normalized Cross Correlation  
<!-- image-->

(c) Best Matched Rendered Image Along X  
<!-- image-->  
(f) Query Image

(d) Worst Matched Rendered Image Along X  
<!-- image-->

(g) Best Matched Rendered Image Along Yaw  
<!-- image-->  
(h) Worst Matched Rendered Image Along Yaw  
Fig. 2. (a)/(e) Illustrating the Relationship between X/Yaw Error and Normalized Cross-Correlation in Localization Initialization; (b)/(f) Query Image for Localization; (c)/(g) Best Matches in Rendered Image Sequences; (d)/(h) Worst Matches in Rendered Image Sequences.

## IV. EXPERIMENT

Our experimental evaluation was conducted using the KITTI360 dataset, which includes LiDAR data, camera images, ground truth poses, and semantic/instance labels. We focus on the datasetâs first route (2013 05 28 drive 0000 sync), dividing it into two segments: the initial drive and the revisit one.

The 3D Gaussian Map was constructed using data solely from the first drive, which provided the necessary inputs for initializing and training our 3DGS map. This map aimed to establish a reliable reference for our relocalization tasks. Data from the second drive were then used to test our relocalization algorithm, allowing us to measure our approachâs effectiveness in a real-world setting. Specifically, we selected two subsequences for our experiments:

â¢ Seq0: frames 4200 to 4550 were used for map creation, and frames 7779 to 8002 for visual relocalization;

â¢ Seq1: frames 7120 to 7450 were used for constructing the map, and frames 9754 to 10062 for visual relocalization;

During map construction, we utilized the datasetâs semantic annotations to exclude the sky and instance labels corresponding to moving objects like vehicles and pedestrians. This approach concentrated our efforts on static environmental features, eliminating the necessity for dynamic reconstruction within the 3DGS model [29]. While our mapping system is capable of processing the entire route, our relocalization experiments focused on selected subsequences. All training and experimental activities were conducted on an NVIDIA RTX A4000 equipped with 16 GB of memory. The submap size was set to 120 meters for training and 150 meters for relocalization, with a voxel resolution of 1 meter.

## A. Initial ReLocalization

In the initial phase of relocalization, we employ the normalized cross-correlation (NCC) metric to evaluate the similarity between pairs of images as mentioned in Section III-C.1. To illustrate the utility of NCC, we present two examples demonstrating its effectiveness in overcoming common localization challenges.

In Fig. 2 (a-d), we examine the scenario where there is an error in the yaw angle. Fig. 2(b) displays the query image used for relocalization. Despite the presence of two new bicyclists in the query image, the NCC metric successfully identifies the correct match (indicated by the yellow point), demonstrating robustness to changes in scene composition and minor orientation errors.

Further, Fig. 2 (e-h) explores the impact of errors in the x position on the localization process. Remarkably, the NCC metric maintains its effectiveness even with an error margin of up to 10 meters, accurately localizing the correct position. This scenario reveals a notably negative relationship between the x position error and the NCC metricâs performance, underscoring the metricâs capacity to guide correct localization under significant positional discrepancies.

These examples highlight the NCC metricâs effectiveness in handling orientation (yaw) and positional (x) errors during initial relocalization. Utilizing the NCC metric enhances our methodâs ability to withstand scene variations and inaccuracies in starting positions, laying a robust groundwork for accurate localization in complex settings. Itâs important to note that the closely matched examples presented here benefit from the use of a very fine grid size during the search process. However, in practical applications and in our implementation, we opt for a coarser grid size to expedite the initialization phase. For achieving precise relocalization, we subsequently apply a feature-based matching method, as detailed in Section III-C.2 and illustrated in the subsequent section.

<table><tr><td>Seq</td><td>Success</td><td>Stage</td><td>X Error</td><td>Y Error</td><td>Yaw Error</td></tr><tr><td rowspan="2">0</td><td rowspan="2">219/223</td><td>Init</td><td>3.513 (3.080)</td><td>2.381 (1.807)</td><td>14.007 (10.685)</td></tr><tr><td>Refine</td><td>0.185 (0.189)</td><td>0.117 (0.168)</td><td>0.535 (0.498)</td></tr><tr><td rowspan="2">1</td><td rowspan="2">304/308</td><td>Init</td><td>3.212 (2.535)</td><td>3.148 (2.450)</td><td>12.001 (13.388)</td></tr><tr><td>Refine</td><td>0.098 (0.076)</td><td>0.114 (0.103)</td><td>0.247 (0.239)</td></tr></table>

TABLE I  
EVALUATION FOR SINGLE IMAGE QUERY RE-LOCALIZATION ERROR ININITIALIZATION AND REFINEMENT STAGE

To evaluate our methodâs effectiveness more thoroughly, we used all query images for the initial relocalization analysis. We introduced noise into $( x , y , y a w )$ of the ground truth pose of each query image. The noise was uniformly sampled within a range of (â10, 10) meters for the x and y translations and $\left( - 9 0 ^ { \circ } , 9 0 ^ { \circ } \right)$ for the yaw rotation. A brute-force search was conducted with a grid size of 2 meters and 10â¦ within a search space of (â15, 15) meters and (â360â¦, 360â¦).We explored the initial 20% of the search space using a random search and applied an early stop criterion. This criterion was based on whether the Normalized Cross-Correlation (NCC) dropped below a set threshold and whether we could successfully obtain sufficient matching points with the second-stage method. The outcomes of this evaluation are detailed in Table I and illustrated in Figure 3. As indicated in Table I, both sequences exhibit high success rates, with Seq 0 achieving a 98.2% success rate (219 out of 223 attempts) and Seq 1 achieving a 98.7% success rate (304 out of 308 attempts). Following the exclusion of unsuccessful matches, we calculated the mean and standard deviation of the errors in $( x , y , y a w )$ . The distribution of translation errors, predominantly within the (â5, 5) meter range yet occasionally exceeding this, is depicted in Figure 3. Despite the presence of significant translation errors initially, the refinement stage markedly enhanced localization accuracy.

## B. ReLocalization Refinement

To thoroughly assess the accuracy of our final fine-pose estimations, we started with the initial poses that were successfully obtained, as outlined in Section IV-A. These poses underwent processing via keypoint detection and feature extraction utilizing Superpoint [21], followed by matching through LightGlue [27]. This cycle of detection, description, and matching was iteratively performed to enhance the accuracy of our estimations. The outcomes of these iterative enhancements are succinctly summarized in Table I and illustrated in Figure 4.

Following the refinement process, we observed significant improvements in the results. For instance, in Seq 0, initial positioning errors decreased from 3.513 meters to 0.185 meters in the X-axis, from 2.381 meters to 0.117 meters in the Y-axis, and from 14.007â¦ to 0.535â¦ in Yaw. Similarly, in Seq 1, errors were reduced from 3.212 meters to 0.098 meters in X, from 3.148 meters to 0.114 meters in Y , and from $1 2 . 0 0 1 ^ { \circ }$ to 0.247â¦ in Yaw. Beyond the reduction in errors, the consistency of our methodology is also evident from the diminished standard deviation values, showcasing the reliability and precision of our approach.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
(a) Seq 0

<!-- image-->  
(b) Seq 1

Fig. 3. Evaluation of Initial Localization X, Y, Yaw Error Histogram  
<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

(a) Seq 0  
<!-- image-->  
(b) Seq 1  
Fig. 4. Evaluation of Re-Localization X, Y, Yaw Error Histogram

## C. Live ReLocalization

For live relocalization evaluation, we randomly initialized the pose of the first query image, which corresponds to the first frame in each sequence. We then streamed subsequent images for live relocalization. As detailed in Table II, we utilize Absolute Pose Error (APE) and Relative Pose Error (RPE) to evaluate our systemâs performance on live relocalization task. The table presents comprehensive statistics for both metrics across two sequences, including the Root Mean Square Error (RMSE), Mean, Median, Standard

<table><tr><td>Metric</td><td>Seq</td><td>RMSE</td><td>Mean</td><td>Median</td><td>Std</td><td>Min</td><td>Max</td><td>SSE</td></tr><tr><td rowspan="2">APE</td><td>0</td><td>0.103</td><td>0.092</td><td>0.088</td><td>0.047</td><td>0.013</td><td>0.349</td><td>2.387</td></tr><tr><td>1</td><td>0.099</td><td>0.087</td><td>0.079</td><td>0.047</td><td>0.013</td><td>0.311</td><td>3.032</td></tr><tr><td rowspan="2">RPE</td><td>0</td><td>0.083</td><td>0.070</td><td>0.060</td><td>0.046</td><td>0.008</td><td>0.252</td><td>1.543</td></tr><tr><td>1</td><td>0.083</td><td>0.070</td><td>0.060</td><td>0.045</td><td>0.008</td><td>0.292</td><td>2.140</td></tr></table>

TABLE II

EVALUATION FOR LIVE RE-LOCALIZATION USING ABSOLUTE POSE ERROR (APE) AND RELATIVE POSE ERROR (RPE)  
<!-- image-->

<!-- image-->  
Fig. 5. Comparison of Ground Truth and Predicted Trajectories from Six Views: Roll, Pitch, Yaw, X, Y, Z for Sequence 0

Deviation (Std), Minimum (Min), Maximum (Max), and Sum of Squared Errors (SSE).

For APE, RMSE is around 0.1, with an average error near 0.09 and a median of 0.08, indicating high accuracy with low variability (standard deviation of 0.047). SSE values highlight precise pose estimations over time.

RPE shows consistent metrics for sequences 0 and 1, with an RMSE of 0.083 and mean and median errors of 0.070 and 0.060, respectively, showing stable relative pose accuracy. Standard deviations are minimal (0.046 for Sequence 0 and 0.045 for Sequence 1), with error ranges of 0.008 to 0.252 for Sequence 0 and 0.008 to 0.292 for Sequence 1, and SSE values of 1.543 for Sequence 0 and 2.140 for Sequence 1, indicating robust relative pose estimation.

Visual analysis of roll-pitch-yaw and XYZ trajectories shows close alignment with ground truth (see Fig. 5-6), but pitch and Z-axis estimates have more noise. The noise may stem from inaccuracies in key points extracted from ground features, which are less precisely depicted in 3D Gaussian plots. To enhance accuracy and reduce noise, employing more sophisticated trajectory estimation techniques, like filter-based methods, could offer smoother and more accurate results.

## V. LIMITATION AND DISCUSSION

A. Balancing Visual Quality with Memory and Geometric Fidelity

To minimize the mapâs footprint, we opted against using Spectral Harmonics (SH) to encode lighting and viewdependent information. While effective in reducing memory usage, this decision has its trade-offs, particularly in outdoor environments where dynamic lighting plays a significant role.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

Fig. 6. Comparison of Ground Truth and Predicted Trajectories from Six Views: Roll, Pitch, Yaw, X, Y, Z for Sequence 1  
<!-- image-->

(a) Ground with Reflection  
<!-- image-->

(b) Ground without Reflection after Moving forward  
<!-- image-->  
(c) Rendered Image  
Fig. 7. Without encoding lighting information in Gaussian Map, lighting changes cause rendering artifacts

For instance, as illustrated in Fig.7, changes in lighting direction can result in varying ground colors, leading to artifacts in our rendered images. Despite this challenge, which was particularly noticeable in different Seq 0 due to varying lighting conditions, the localization accuracy between Seq 0 and Seq 1 remained consistent in our experiments. This resilience is primarily attributed to the robustness of the Normalized Cross Correlation (NCC) metrics, as well as the feature detection and matching capabilities of Superpoint[21] and LightGlue[27]. This observation prompts a reevaluation of the necessity to encode lighting information within the map for relocalization tasks. Our findings suggest incorporating dynamic lighting and shadows may not be essential for achieving accurate localization. Moreover, an ideal map might benefit from eliminating lighting and shadow effects to focus more on the environmentâs geometric and structural aspects, further streamlining the localization process without compromising accuracy. This finding suggests a potential direction for future research, exploring the balance between visual fidelity, memory efficiency, and geometric accuracy in the context of map representation and localization.

## B. Towards a Fully Differentiable Localization Pipeline

The 3D Gaussian Splatting representationâs differentiability is an interesting feature, which might offer the possibility of creating a fully differentiable pipeline to perform localization on a 3D Gaussian Splatting submap. This capability might enable us to bypass the traditional detectiondescription-matching approach, removing the need to train separate models for feature detection and extraction. Additionally, a fully differentiable pipeline can facilitate integration with other differentiable methods for navigation and planning systems. We have initially evaluated several metrics to perform direct localization on a 3D Gaussian Splatting map. These metrics include Gradient Correlation (GC), Normalized Cross Correlation (NCC) [26], and Mutual Information (MI) [30]. However, our preliminary experiment indicates that these metrics are particularly sensitive to initial pose estimates and prone to settling into local minima during gradient descent optimization. These challenges suggest the need to explore alternative optimization techniques or strategies to address these issues.

## VI. CONCLUSION

This paper has explored the integration of LiDAR and camera data through the novel application of 3D Gaussian Splatting, addressing the crucial need for advanced map representation methodologies in the rapidly evolving domains of autonomous driving and robotic navigation. By leveraging the strengths of both LiDARâs depth sensing and the detailed imagery provided by cameras, we have demonstrated a robust approach to creating detailed and geometrically accurate environmental representations, crucial for the safe and efficient navigation of autonomous systems. Our methodology, which begins with LiDAR data to initiate the training of the 3D Gaussian Splatting representation, facilitates the construction of comprehensive maps while addressing common challenges such as high memory usage and inaccuracies in underlying geometry.

The implementation of our technique in visual relocalization task showcases its capacity to enhance the precision of feature identification and positioning, contributing significantly to the field by enabling more sophisticated perception systems for autonomous vehicles. Through our evaluation with the KITTI360 dataset, we have validated the effectiveness, adaptability, and precision of our approach, underscoring its potential to advance environmental perception and system reliability. Ultimately, our work contributes to the broader conversation on sensor data integration and map representation, offering a pathway toward more efficient, accurate, and reliable localization and navigation in complex urban landscapes.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D Gaussian Â¨ Splatting for Real-Time Radiance Field Rendering,â vol. 42, no. 4, p. 1.

[2] Y. Liao, J. Xie, and A. Geiger, âKITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D,â vol. 45, no. 3, pp. 3292â3310.

[3] C. Chen, B. Wang, C. X. Lu, N. Trigoni, and A. Markham, âDeep Learning for Visual Localization and Mapping: A Survey,â pp. 1â21.

[4] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.

[5] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, âiMAP: Implicit Mapping and Positioning in Real-Time,â pp. 6229â6238.

[6] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNICE-SLAM: Neural Implicit Scalable Encoding for SLAM.â

[7] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald. Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting.

[8] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison. Gaussian Splatting SLAM.

[9] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, and H. Wang. SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM.

[10] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten. SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.

[11] J. Lin, Z. Li, X. Tang, J. Liu, S. Liu, J. Liu, Y. Lu, X. Wu, S. Xu, Y. Yan, and W. Yang. VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction.

[12] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang. Periodic Vibration Gaussian: Dynamic Urban Scene Reconstruction and Real-time Rendering.

[13] Y. Yan, H. Lin, C. Zhou, W. Wang, H. Sun, K. Zhan, X. Lang, X. Zhou, and S. Peng. Street Gaussians for Modeling Dynamic Urban Scenes.

[14] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang. DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes.

[15] E. Brachmann, A. Krull, S. Nowozin, J. Shotton, F. Michel, S. Gumhold, and C. Rother, âDSAC - Differentiable RANSAC for Camera Localization,â pp. 6684â6692.

[16] Z. Laskar, I. Melekhov, S. Kalia, and J. Kannala, âCamera Relocalization by Computing Pairwise Relative Poses Using Convolutional Neural Network,â pp. 929â938.

[17] P.-E. Sarlin, A. Unagar, M. Larsson, H. Germain, C. Toft, V. Larsson, M. Pollefeys, V. Lepetit, L. Hammarstrand, F. Kahl, and T. Sattler, âBack to the Feature: Learning Robust Camera Localization From Pixels To Pose,â pp. 3247â3257.

[18] M. R. U. Saputra, A. Markham, and N. Trigoni, âVisual SLAM and Structure from Motion in Dynamic Environments: A Survey,â vol. 51, no. 2, pp. 37:1â37:36.

[19] D. Werner, A. Al-Hamadi, and P. Werner, âTruncated Signed Distance Function: Experiments on Voxel Size,â in Image Analysis and Recognition, ser. Lecture Notes in Computer Science, A. Campilho and M. Kamel, Eds. Springer International Publishing, pp. 357â364.

[20] R. W. Wolcott and R. M. Eustice, âVisual localization within LIDAR maps for automated urban driving,â in 2014 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 176â183.

[21] D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperPoint: Self-Supervised Interest Point Detection and Description,â pp. 224â236.

[22] B. Wang, C. Chen, Z. Cui, J. Qin, C. X. Lu, Z. Yu, P. Zhao, Z. Dong, F. Zhu, N. Trigoni, and A. Markham, âP2-Net: Joint Description and Detection of Local Features for Pixel and Point Matching.â

[23] P. Jiang and S. Saripalli, âContrastive Learning of Features between Images and LiDAR,â in 2022 IEEE 18th International Conference on Automation Science and Engineering (CASE), pp. 411â417.

[24] N. Max, âOptical models for direct volume rendering,â vol. 1, no. 2, pp. 99â108.

[25] M. Zwicker, J. Pfister, H.Baar, and M. Gross, âEWA volume splatting,â in Proceedings Visualization, 2001. VIS â01., pp. 29â538.

[26] Y. Hiasa, Y. Otake, M. Takao, T. Matsuoka, K. Takashima, A. Carass, J. L. Prince, N. Sugano, and Y. Sato, âCross-modality image synthesis from unpaired data using cyclegan,â in International Workshop on Simulation and Synthesis in Medical Imaging. Springer, 2018, pp. 31â41.

[27] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, âLightGlue: Local Feature Matching at Light Speed,â pp. 17 627â17 638.

[28] S. Thrun, W. Burgard, and D. Fox, Probabilistic Robotics (Intelligent Robotics and Autonomous Agents). The MIT Press.

[29] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan. Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis.

[30] P. Jiang, P. Osteen, and S. Saripalli, âSemCal: Semantic LiDAR-Camera Calibration using Neural Mutual Information Estimator,â in 2021 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI). IEEE, pp. 1â7.