# DGS-SLAM: Gaussian Splatting SLAM in Dynamic Environment

Mangyu Kong1, Jaewon Lee1, Seongwon Lee2 and Euntai Kim1

Abstractâ We introduce Dynamic Gaussian Splatting SLAM (DGS-SLAM), the first dynamic SLAM framework built on the foundation of Gaussian Splatting. While recent advancements in dense SLAM have leveraged Gaussian Splatting to enhance scene representation, most approaches assume a static environment, making them vulnerable to photometric and geometric inconsistencies caused by dynamic objects. To address these challenges, we integrate Gaussian Splatting SLAM with a robust filtering process to handle dynamic objects throughout the entire pipeline, including Gaussian insertion and keyframe selection. Within this framework, to further improve the accuracy of dynamic object removal, we introduce a robust mask generation method that enforces photometric consistency across keyframes, reducing noise from inaccurate segmentation and artifacts such as shadows. Additionally, we propose the loop-aware window selection mechanism, which utilizes unique keyframe IDs of 3D Gaussians to detect loops between the current and past frames, facilitating joint optimization of the current camera poses and the Gaussian map. DGS-SLAM achieves state-of-the-art performance in both camera tracking and novel view synthesis on various dynamic SLAM benchmarks, proving its effectiveness in handling real-world dynamic scenes. The codes are available at this url.

## I. INTRODUCTION

Dense Simultaneous Localization and Mapping (SLAM) has been a fundamental task in robotics and computer vision for decades. The primary goal of dense SLAM is to simultaneously estimate the cameraâs position and reconstruct a dense map in an unknown environment. This task is crucial for many real-world applications such as robot navigation, autonomous driving, and VR/AR. Since the specific algorithm used in dense SLAM is inherently linked to the type of map representation, the choice of representation has a significant impact on SLAM performance. As a result, various approaches have been explored, including surfels [1], [2], meshes [3], and signed distance fields (SDF) [4], [5].

Recently, NeRF-based volumetric representation [6], [7] has emerged as a promising map representation, offering detailed scene reconstruction. These methods capture dense photometric information from the environment, enabling high-quality image synthesis from novel viewpoints. Following these advancements, Gaussian Splatting [8] stands out as a novel volumetric technique. Unlike ray-matching methods like NeRF [6], Gaussian Splatting employs differential rasterization of 3D primitives, significantly enhancing reconstruction quality and rendering speed.

<!-- image-->

Input RGB-D data  
<!-- image-->

<!-- image-->

3D Gaussian Splatting  
<!-- image-->

<!-- image-->

<!-- image-->  
Rendered Images

<!-- image-->  
Fig. 1. Result of our framework. Top: RGB-D frames as input. Center: Reconstructed Gaussian map model without dynamics. Bottom: Rendered images from tracked camera pose.

With these capabilities, various SLAM methods utilizing Gaussian Splatting [9], [10], [11], [12], [13] have been introduced, demonstrating remarkable improvements in SLAM performance. Gaussian Splatting enables highly efficient and accurate map representations with reduced computational overhead, making it a compelling choice for SLAM applications. Despite its advantages, however, most existing methods assume static environments, leading to inaccuracies and degraded performance in dynamic scenarios. Therefore, addressing this constraint is essential for expanding the applicability of Gaussian Splatting SLAM to real-world scenarios.

On the other hand, several traditional SLAM methods have been developed to address the challenges posed by complex dynamic environments. These approaches utilize techniques such as semantic segmentation priors [14], [15], optical flow [16], [17], and residual optimization [5] to filter out moving objects. While these approaches have shown some success in mitigating the effects of dynamic elements, they also come with limitations. Methods that rely on semantic priors often struggle with segmentation errors and artifacts in real-world scenarios. Although residual optimization can be effective, it tends to fail when confronted with large object movements. Moreover, traditional dynamic SLAM systems are generally limited in their ability to generate detailed scene representations.

To address both the challenges of dynamic environments and the limitations of traditional dynamic SLAM methods, we introduce Dynamic Gaussian Splatting SLAM (DGS-SLAM) in this work. Our DGS-SLAM is designed to integrate Gaussian Splatting SLAM with a robust dynamic filtering process to handle dynamic objects throughout the entire SLAM pipeline, from Gaussian initialization to joint optimization. We fully exploit the inherent properties of Gaussian Splatting to ensure both robustness and efficiency in dynamic environments.

In this framework, we additionally propose a robust mask generation method that improves the accuracy of dynamic object removal by ensuring photometric consistency across keyframes. This method helps to reduce noise from inaccurate segmentation and minimize the impact of artifacts such as shadows, reliably separating dynamic and static elements in the scene. We also propose a Gaussian-based loop-aware window selection strategy utilizing unique keyframe IDs associated with each Gaussian. This enables joint optimization across previous keyframes, further enhancing the frameworkâs ability to maintain accurate localization over time while revisiting scenes in dynamic environments.

Our proposed DGS-SLAM achieves state-of-the-art performance in both camera tracking and novel view synthesis on various dynamic SLAM benchmarks, proving its effectiveness in handling real-world dynamic scenes.

## II. RELATED WORKS

## A. Dynamic SLAM

Dynamic SLAM focuses on accurately estimating poses and reconstructing static scenes in environments where objects are moving or changes occur. Various dynamic SLAM methods have been developed to address these challenges. Some approaches use deep learning segmentation modules to remove dynamic elements from the scene, enabling accurate pose prediction and mapping [14], [15], [18]. Others utilize dense optical flow module [19], [20] to detect motion and identify dynamic regions [16], [17], [21]. Additionally, some methods leverage residuals obtained after initial registration to filter out outliers [5]. However, these methods often suffer from noise due to domain gaps in real-world environments and struggle with large movements of dynamic objects. Moreover, in existing dense dynamic SLAM, achieving photorealistic rendering is challenging. In this work, we employ Gaussian Splatting as a map representation in dynamic SLAM, enabling high-quality rendering from a novel view. Furthermore, we propose a robust mask generation methods to complement inaccurate segmentations and artifacts from dynamic objects.

## B. Neural Implicit SLAM

Neural implicit representations like NeRF [6] and SDF [7], have gained significant attention for its remarkable ability to represent scenes densely and continuously. iMap [22], the first approach to apply neural implicit representation in SLAM, achieved real-time mapping and tracking. After iMAP, Nice-SLAM [23] introduced a hierarchical multifeature representation to improve scalability and address over-smoothed scene reconstruction. Following these, several studies [24], [25], [26], [27], [28] propose more practical scene representations to address memory limitations and enhance reconstruction accuracy. While most neural implicit SLAM systems demonstrate impressive performance under the assumption of static scenes, they show limitations in dynamic environments. Our DGS-SLAM addresses challenges in dynamic environments and demonstrates outstanding performance by utilizing a novel map representation.

## C. 3D Gaussian Splatting SLAM

Gaussian Splatting [8] has recently emerged as an effective method for representing scenes through a set of Gaussians. Unlike neural implicit representations that rely on ray marching, 3D Gaussian Splatting utilizes differential rasterization, achieving both fast rendering speeds and accurate scene reconstruction. Leveraging these advantages, this approach has expanded into various fields, including more accurate 3D scene reconstruction [29], [30], [31], dynamic scene modeling [32], [33], [34], [35], and scene editing [36], [37]. One of the most developed areas is Dense SLAM utilizing Gaussian Splatting as map representation. SplaTAM [9] introduced silhouette-guided optimization for progressive map reconstruction in dense SLAM. Gaussian Splatting SLAM [10] proposed a novel Gaussian insertion and pruning strategy, enabling it to work not only with RGB-D but also with monocular cases. Additionally, Photo-SLAM [12] leveraged explicit geometric features like ORB for localization and introduced a hyper primitives map for photometric feature mapping. However, most of the Gaussian Splatting SLAM approaches have focused on static scenes, overlooking dynamic environments. To address this issue, we propose Dynamic Gaussian Splatting SLAM (DGS-SLAM), which filters out dynamic elements across the entire Gaussian Splatting SLAM system.

## III. METHOD

The overview of our DGS-SLAM is illustrated in Figure 2. Similar to existing Gaussian Splatting SLAM [10], [12], our system is composed of a frontend tracking and a backend mapping process after Gaussian map initialization. DGS-SLAM simultaneously optimizes the camera pose $\{ \mathbf { T } \} _ { k = 1 } ^ { N } , \mathbf { T } _ { k } \in \mathbb { S } \mathbb { E } ( 3 )$ and reconstructs 3D Gaussian splatting with dynamic elements removed from a sequence of RGB-D frame inputs $\{ I _ { k } , D _ { k } \} _ { k = 1 } ^ { N }$

## A. Gaussian Splatting

Our dynamic SLAM system utilizes Gaussian splatting as the map representation. Each anisotropic Gaussian $\mathcal { G } ^ { i }$ is parameterized by its RGB color $c ^ { i } .$ , mean position $\mu ^ { i } \in$ $\mathbb { R } ^ { 3 }$ , covariance $\Sigma ^ { i }$ , and opacity $o ^ { i } \in [ 0 , 1 ]$ . We simplify each Gaussian to view-independent color, removing spherical harmonics (SHs). The Gaussian equation for a 3D point $x \in \mathbb { R } ^ { 3 }$ is as follows:

$$
g ( x ) = o \exp \left( - \frac { \| x - \mu \| ^ { 2 } } { 2 r ^ { 2 } } \right) .\tag{1}
$$

<!-- image-->  
Fig. 2. Framework Overview Our framework simultaneously estimates the camera pose while reconstructing a 3D gaussian splatting map with a sequence of RGB-D frames in a dynamic environment. DGS-SLAM consists of three main components: initialization, frontend tracking, and backend mapping. During initialization, the Gaussians are optimized based on the first frame. In the frontend, DGS-SLAM estimates the camera pose while filtering out dynamic elements. The backend then performs joint optimization to refine the pose and update the 3D Gaussian Splatting map.

Following 3D Gaussian Splatting [8], each 3D Gaussian rasterizes into 2D splats, enabling gradient flow for scene reconstruction and pose estimation. We project m Gaussians which are sorted in order of depth, and blend them into the color of pixel $p \mathrm { : }$

$$
C ( \boldsymbol { p } ) = \sum _ { i = 1 } ^ { m } g _ { i } ( \boldsymbol { p } ) c _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - g _ { i } ( \boldsymbol { p } ) ) ,\tag{2}
$$

During rasterization, the mean position $\mu ^ { 2 D }$ and covariance $\Sigma ^ { 2 \bar { D } }$ projected from 3D gaussian in pixel-space are as follows:

$$
\mu ^ { 2 D } = K \mathbf { T } \mu d ^ { - 1 } , \Sigma ^ { 2 D } = J \mathbf { T } _ { \mathrm { r o t } } \Sigma \mathbf { T } _ { \mathrm { r o t } } ^ { T } J ^ { T } ,\tag{3}
$$

where $d = \left( \mathbf { T } \mu \right)$ z is the distance from the camera to the Gaussian and K is the calibrated camera intrinsic parameter.

For the depth rendering of pixel $p ,$ we follow a similar to equation 2:

$$
D ( \boldsymbol { p } ) = \sum _ { i = 1 } ^ { m } g _ { i } ( \boldsymbol { p } ) d _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - g _ { j } ( \boldsymbol { p } ) ) ,\tag{4}
$$

## B. Pose Tracking

During the tracking process in the frontend, the pose $T _ { k }$ is optimized by minimizing the difference between each input frame $C _ { k } , D _ { k }$ and the rendered result $\hat { C } ( T _ { k } ) , \hat { D } ( T _ { k } )$ from the predicted pose $T _ { k }$ . However, it is necessary to filter out dynamic elements to accurately optimize pose estimation. For each input frame $k ,$ we utilize an off-theshelf online instance video module [38] to obtain mask $M _ { \mathrm { s e g } } ^ { k }$ of dynamic objects. Additionally, we generate opacity mask $M _ { \mathrm { o p a c i t y } } ^ { k }$ through opacity checks to filter out empty spaces before mapping. We combine the opacity mask and the segmentation mask to generate the overall tracking mask $\bar { M _ { \mathrm { t r a c k i n g } } ^ { k } }$ as:

$$
\widehat { M } _ { \mathrm { t r a c k i n g } } ^ { k } = \widehat { M } _ { \mathrm { s e g } } ^ { k } \otimes \left( \sum _ { i = 1 } ^ { m } g _ { i } ( p ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - g _ { j } ( p ) ) > \tau _ { \mathrm { o p a c i t y } } \right) .\tag{5}
$$

where $\tau _ { \mathrm { o p a c i t y } }$ is the threshold value to determine the unmapped region, which is set as 0.95.

We track the camera pose of frame k by minimizing the following loss:

$$
L _ { \mathrm { t r a c k i n g } } ^ { k } = \widehat { M } _ { \mathrm { t r a c k i n g } } ^ { k } \left( \alpha L _ { \mathrm { c o l o r } } ^ { k } + ( 1 - \alpha ) L _ { \mathrm { d e p t h } } ^ { k } \right) ,\tag{6}
$$

where $L _ { \mathrm { c o l o r } } ^ { k }$ is the photometric residual $\| \hat { C } _ { k } ( \mathcal { G } , T _ { k } ) - C _ { k }$ â¥1 between rendered image and the input image, and $L _ { \mathrm { d e p t h } } ^ { k }$ is the depth residual $\| \hat { D } _ { k } ( \mathcal { G } , T _ { k } ) - D _ { k } \| _ { 1 }$

## C. Loop-aware Keyframe Management

Keyframe selection After optimizing the camera pose $\mathbf { T } _ { k }$ of the input frame k, we form a window W with keyframes to jointly optimize the Gaussians $\mathcal { G }$ and camera poses $\mathbf { T } _ { k } , k \in \ \mathcal { W }$ The criteria for keyframe selection are Gaussian covisibility, relative pose $\mathbf { T } _ { i j }$ from the last keyframe j, and the unique keyframe ID of the Gaussians. For Gaussian covisibility, we measure the intersection over union (IoU) of visible Gaussians between the current frame i and the last keyframe j. A frame is registered if the IOU of covisibility $\frac { | \check { \mathcal { G } } _ { i } ^ { v } \cap \mathcal { G } _ { j } ^ { v } | } { | \mathcal { G } _ { i } ^ { v } \cup \mathcal { G } _ { j } ^ { v } | }$ falls below a certain threshold $\sigma _ { \mathrm { { I o U } } }$ And for the relative pose $\mathbf { T } _ { i j }$ , when not only the transition but rotation difference of the relative pose exceeds a defined value, the current frame is the keyframe. Additionally, to maintain a consistent window length, keyframes are removed based on similar criteria.

Loop-aware keyframe insertion When the window W for joint optimization is managed solely based on covisibility and the relative pose with the last keyframe, removed keyframes can no longer affect the global map. This disrupts the consistency of the global Gaussian map. To address this, we introduce Loop-aware keyframe insertion, assigning the ID of the frame where it is generated to each Gaussian as $k \mapsto \mathrm { I D } ( \mathcal G _ { i } )$ . The unique keyframe ID of the visible Gaussian $\mathcal { G } _ { k _ { c } } ^ { v }$ in the current frame $k _ { c }$ is key to identifying loops in past keyframes. The portion of these identified loop keyframes $k _ { l }$ is re-included in the window and jointly optimized with the current keyframes as:

$$
\mathcal { W } = \mathcal { W } \cup \{ k _ { l } | k _ { l } = \mathrm { I D } ( \mathcal { G } ^ { v } ) \mathrm { ~ f o r ~ } \mathcal { G } ^ { v } \mathrm { ~ i n ~ } \mathcal { G } _ { k _ { c } } ^ { v } \} .\tag{7}
$$

## D. Mapping process

## Gaussian Insertion & Pruning

Before the camera tracking and mapping process begins, we insert and initialize the 3D Gaussians using the first frame input, $C _ { k = 0 } , D _ { k = 0 }$ . To initialize the scene more accurately, we align the estimated depth $D _ { \mathrm { a l i g n } } ~ = ~ a D _ { \mathrm { e s t } } ( C _ { k = 0 } ) + b$ with the ground truth depth $D _ { t = 0 }$ . Filling depth holes aid in completing the initial scene densely. Additionally, we define the position $\mu _ { 3 D }$ of 3D Gaussian by unprojecting the pixels p which are not on the invalid region of dynamic elements:

$$
\mu _ { 3 D } = \mathbf { T } ^ { - 1 } \cdot ( K ^ { - 1 } p \cdot D ( p ) ) , \mathrm { w h e r e } p \in M _ { \mathrm { s e g } } .\tag{8}
$$

Initial Gaussians $\mathcal { G } _ { i n i t }$ are optimized by the corresponding loss $L _ { \mathrm { i n i t } } = M _ { \mathrm { s e g , } k = 0 } ( \alpha L _ { \mathrm { c o l o r } } + ( 1 - \alpha ) L _ { \mathrm { d e p t h } } )$ . Afterward, every time a new keyframe is added, 3D Gaussians are inserted according to Equation 8. For memory efficiency, Gaussians are generated more sparsely than initialization, and the scale of each Gaussian is defined in proportion to the median depth, excluding invalid depths and dynamic regions. We primarily follow the Gaussian densification and pruning strategies as [8], [10].

## Robust Mask Generation

Online segmentation in real-world environments is not perfect. Therefore, to ensure photometric and geometric consistency, we additionally generate masks for outliers during the window optimization process. Similar to the approach proposed in [5], [39], we sort the phometric residuals $R _ { k } =$ $\Vert \hat { C } ( \mathcal { G } , T _ { k } ) - \bar { C } _ { k } \Vert _ { 1 }$ 1 between the rendered image and the input image of the keyframe $k ,$ and represent them as a histogram $\mathcal { H } ( R _ { k } )$ . We assume that pixels with photometric residuals below a certain percentile $\tau _ { \mathrm { r o b u s t } }$ of the residual histogram are inliers. The residual threshold $\epsilon _ { \mathrm { r o b u s t } }$ , which distinguishes inlier pixels from outlier pixels, is defined as follows:

$$
\epsilon _ { \mathrm { r o b u s t } } = \operatorname* { m i n } \left\{ \epsilon \mid \sum _ { i = 0 } ^ { \epsilon } \mathcal { H } _ { i } \geq \tau _ { \mathrm { r o b u s t } } \cdot \sum _ { i = 0 } ^ { \infty } \mathcal { H } _ { i } \right\}\tag{9}
$$

To further avoid incorrect masking in high-frequency regions, we apply smoothing using a normalized kernel W.

$$
M _ { \mathrm { r o b u s t } } ^ { k } = 1 \{ ( 1 \{ R _ { k } > \epsilon _ { \mathrm { r o b u s t } } \} \circledast \mathbf { W } ) > 0 . 5 \}\tag{10}
$$

To gradually update the photometric residual histogram, The histogram used to determine outlier pixels is iteratively updated, as follows:

$$
\mathcal { H } _ { t } ^ { k } = ( 1 - \gamma ) \cdot \mathcal { H } _ { t - 1 } ^ { k } + \mathcal { H } ( R _ { k } ) ) .\tag{11}
$$

Window Optimization In the backend mapping process, we optimize the 3D Gaussians using the keyframes selected through both window management and loop-aware selection. During the window optimization, we jointly optimize both the scene representation and the camera poses. Ultimately, we minimize the following loss,

$$
\operatorname* { m i n } _ { \substack { \mathbb { Y } ^ { k } \in \mathcal { W } } } \sum _ { \forall k \in \mathcal { W } } M _ { \mathrm { w i n d o w } } ^ { k } ( \alpha L _ { p h o } ^ { k } + ( 1 - \alpha ) L _ { D e p t h } ^ { k } ) + \lambda _ { i s o } L _ { i s o } ,\tag{12}
$$

where $M _ { \mathrm { w i n d o w } } ^ { k }$ is the mask for window optimization obtained as $M _ { \mathrm { s e g } } ^ { k } \otimes ( 1 - M _ { \mathrm { r o b u s t } } ^ { k } )$ , and $\boldsymbol { L } _ { i s o }$ is an isotropic regularization term. This isotropic regularization aids in reducing artifacts in mapping 3D Gaussian Splatting.

## IV. EXPERIMENTS

Datasets We evaluated our approach on two prominent dynamic datasets: the TUM RGB-D dataset [40] and the Bonn RGB-D dataset [5]. Both datasets were captured in indoor environments using a handheld device, providing RGB images, depth maps, and ground truth trajectories.

Metrics To evaluate the quality of novel view synthesis for the map, we report the representative photometric metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). For assessing the rendering quality of dynamic scenes, we masked out the pixels corresponding to dynamic objects in black during evaluation. To measure camera tracking accuracy, we adopted the RMSE and standard deviation (STD) of the Absolute Trajectory Error (ATE) [40] of the keyframes.

Implementation details Our DGS-SLAM experiments were conducted on a workspace equipped with a TITAN RTX and a 3.4GHz AMD Ryzen 7 3800XT. For the experiments, the loss parameters were set as follows: $\alpha = 0 . 9 , \delta _ { i s o } = 0 . 1$ the robust threshold $\sigma _ { r o b u s t }$ for robust mask estimation is 0.9, and the smoothing kernel size is 7. Additionally, the opacity threshold $\sigma _ { o p a c i t y }$ for tracking is set to 0.95 and IoU threshold $\sigma _ { \mathrm { { I o U } } }$ to 0.9. We use the Adam optimizer [41] for camera optimization, with a learning rate of 0.003 for rotation and 0.001 for translation. The hyperparameters for the 3D Gaussians followed the same settings as [8]. For the segmentation mask, we leverage Track Anything [38], [42], [43], an open-vocabulary video segmentation module that operates online. In particular, we implemented a lightweight tiny version to save computation time during experiments.

## A. Evaluation of Mapping

To demonstrate the mapping performance of our DGS-SLAM in dynamic environments, we evaluated its novel view synthesis performance as a qualitative result. We evaluated rendering quality by averaging the differences between the rendered images and the ground truth images across all frames. We show a comparison of rendering quality with Gaussian Splatting-based SLAM methods, SplaTAM [9] and MonoGS [10]. As shown in Table I, our proposed DGS-SLAM achieves the best performance among Gaussian Splatting SLAMs on the dynamic scenes of the TUM dataset. In Figure 3 , we qualitatively compare the rendered image from reconstructed Gaussian maps, showing how our method is robust in a dynamic environment compared to other GS-SLAM approaches. In dynamic scenes, existing GS-based SLAM systems generate Gaussians for moving elements, leading to photometric inconsistencies and failing to accurately reconstruct the scene.

TABLE I  
NOVEL VIEW SYNTHESIS RESULTS AS A MAPPING QUALITY ON SEVERAL DYNAMIC SEQUENCES IN THE TUM RGB-D DATASET.
<table><tr><td rowspan="2"></td><td colspan="3">f3/wk_xyz</td><td colspan="3">f3/wk_hf</td><td colspan="3"> $\pm 3 / \mathrm { w k } _ { - } \mathrm { s t }$ </td><td colspan="3"> $\pm 3 / \mathsf { s t \mathrm { _ { - } h f } }$ </td><td rowspan="2">PSNRâ</td><td rowspan="2">Avg. SSIM</td><td rowspan="2">LPIPSâ</td></tr><tr><td>PSNRâ</td><td></td><td>SSIMâLPIPSâ</td><td></td><td>PSNRâ</td><td>SSIMâ LPIPSâ</td><td>PSNRâ</td><td>SSIM</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>SplaTAM [9]</td><td>14.54</td><td>0.539</td><td>0.480</td><td>13.52</td><td>0.507</td><td>0.517</td><td>17.90</td><td>0.802</td><td>0.270</td><td>16.25</td><td>0.680</td><td>0.384</td><td>15.55</td><td>0.633</td><td>0.413</td></tr><tr><td>Monos [10]</td><td>14.41</td><td>0.535</td><td>0.391</td><td>14.23</td><td>0.542</td><td>0.457</td><td>16.81</td><td>0.714</td><td>0.252</td><td>18.11</td><td>0.692</td><td>0.333</td><td>15.89</td><td>0.621</td><td>0.358</td></tr><tr><td>GS-CP[11]</td><td>16.92</td><td>0.694</td><td>0.356</td><td>16.36</td><td>0.671</td><td>0.392</td><td>18.30</td><td>0.727</td><td>0.311</td><td>18.12</td><td>0.728</td><td>0.293</td><td>17.42</td><td>0.705</td><td>0.338</td></tr><tr><td>DGS-SLAM (Ours)</td><td>20.48</td><td>0.798</td><td>0.173</td><td>20.00</td><td>0.774</td><td>0.229</td><td>22.89</td><td>0.897</td><td>0.089</td><td>19.16</td><td>0.763</td><td>0.251</td><td>20.63</td><td>0.807</td><td>0.186</td></tr></table>

TABLE II

CAMERA TRACKING RESULTS ON DYNAMIC AND STATIC SCENES IN THE TUM RGB-D DATASET. THE UNITS FOR ATE AND S.D ARE IN CM.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Dense</td><td colspan="7">Dynamic</td><td colspan="4">Static</td><td rowspan="2" colspan="2">Avg.</td></tr><tr><td>f3/wk_xyz</td><td colspan="2">f3/wk_hf</td><td colspan="2">f3/wk_st</td><td colspan="2"> $\pm 3 / \mathrm { s t \mathrm { \_ h f } }$ </td><td colspan="2">f1/xyz</td><td colspan="2"> $\pm 1 / \tt r p y$ </td></tr><tr><td>Traditional SLAM methods</td><td rowspan="5">â</td><td>ATE S.D. 12.2</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td></tr><tr><td>ORB-SLAM3 [44]</td><td>28.1</td><td>30.5</td><td>9.0</td><td>2.0</td><td>1.1</td><td>2.6</td><td>1.6</td><td>1.1</td><td>0.6</td><td>2.2</td><td>1.3</td><td>11.1</td><td>4.3</td></tr><tr><td>DVO-SLAM [?]</td><td>59.7</td><td>-</td><td>52.9</td><td></td><td>21.2</td><td>6.2</td><td></td><td>1.1</td><td></td><td>2.0</td><td></td><td>22.9</td><td>-</td></tr><tr><td>DynaSLAM [15]</td><td>1.7</td><td>2.6</td><td></td><td>0.7</td><td>-</td><td>2.8</td><td></td><td></td><td></td><td></td><td></td><td>2.0</td><td></td></tr><tr><td>ReFusion [5]</td><td>9.9</td><td>10.4</td><td>-</td><td>1.7</td><td>-</td><td>11.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td></td><td>8.3</td><td>-</td></tr><tr><td>Radiance-Field SLAM methods</td><td></td><td>ATE S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td></tr><tr><td>NICE-SLAM [23]</td><td></td><td>113.8 42.9</td><td>X</td><td>X</td><td>88.2</td><td>27.8</td><td>45.0</td><td>14.4</td><td>4.6</td><td>3.8</td><td>3.4</td><td>2.5</td><td>51</td><td>18.3</td></tr><tr><td>Vox-Fusion [24]</td><td></td><td>146.6 32.1</td><td>X</td><td>X</td><td>109.9</td><td>25.5</td><td>89.1</td><td>28.5</td><td>1.8</td><td>0.9</td><td>4.3</td><td>3.0</td><td>70.4</td><td>18</td></tr><tr><td>Co-SLAM [26]</td><td></td><td>51.8 25.3</td><td>105.1</td><td>42.0</td><td>49.5</td><td>10.8</td><td>4.7</td><td>2.2</td><td>2.3</td><td>1.2</td><td>3.9</td><td>2.8</td><td>36.3</td><td>14.1</td></tr><tr><td>ESLAM [25]</td><td>1</td><td>45.7 28.5</td><td>60.8</td><td>27.9</td><td>93.6</td><td>20.7</td><td>3.6</td><td>1.6</td><td>1.1</td><td>0.6</td><td>2.2</td><td>1.2</td><td>34.5</td><td>13.5</td></tr><tr><td>SplaTAM [9]</td><td></td><td>134.4 32.1</td><td>746.1</td><td>250.5</td><td>97.8</td><td>26.9</td><td>14.1</td><td>6.8</td><td>1.0</td><td>0.5</td><td>2.6</td><td>1.3</td><td>166.0</td><td>52.9</td></tr><tr><td>MonoGS [10]</td><td></td><td>73.4 20.1</td><td>65.6</td><td>24.8</td><td>5.5</td><td>3.0</td><td>2.7</td><td>1.5</td><td>1.0</td><td>0.4</td><td>2.5</td><td>1.3</td><td>37.7</td><td>25.1</td></tr><tr><td>GS-ICP SLAM [11]</td><td></td><td>70.5 45.1</td><td>73.9</td><td>34.1</td><td>98.2</td><td>24.1</td><td>9.9</td><td>3.7</td><td>1.4</td><td>0.7</td><td>3.2</td><td>2.8</td><td>42.9</td><td>18.4</td></tr><tr><td>RoDyn-SLAM [45]</td><td></td><td>8.3 5.5</td><td>5.6</td><td>2.8</td><td>1.7</td><td>0.9</td><td>4.4</td><td>2.2</td><td>1.5</td><td>0.8</td><td>2.8</td><td>1.5</td><td>4.1</td><td>2.3</td></tr><tr><td>DGS-SLAM (ours)</td><td>â</td><td>4.1 2.2</td><td>5.5</td><td>2.8</td><td>0.6</td><td>0.2</td><td>4.1</td><td>1.6</td><td>1.2</td><td>0.6</td><td>2.4</td><td>1.3</td><td>3.0</td><td>1.5</td></tr></table>

Input Frame

SplaTAM  
MonoGS  
<!-- image-->  
Fig. 3. Comparison of rendered results from state-of-the-art Gaussian Splatting SLAM approaches based on the estimated input frame poses.

## B. Evaluation of Camera Tracking

To evaluate the camera tracking performance in dynamic environments, we compared our method with radiance field based SLAM methods [23], [26], [9], [10], [11] and traditional SLAM methods [44], [?], [5], including SLAM methods [15], [46], [45] specifically designed for dynamic environments. In Table II, we present the results for four dynamic scenes from the TUM dataset. We achieved superior results through the advantages of using Gaussian splatting as a map representation, along with a novel robust mask and a novel keyframe management strategy. While existing Gaussian Splatting-based SLAMs often fail in dynamic scenes, our method demonstrates accurate camera tracking. Furthermore, compared to RoDyn-SLAM [45], a NeRFbased dynamic SLAM, our approach achieves more accurate results in most of the scenes. Our approach even outperforms traditional dynamic SLAMs in walking xyz scene. Table III presents the camera tracking results on the Bonn RGB-D dataset [5]. The Bonn dataset is more complex and captured in larger scenes with various dynamic movements. In Bonn dataset, our DGS-SLAM demonstrates superior performance compared to other radiance-field based SLAM methods, outperforming some traditional sparse SLAM approaches.

## C. Ablation study

Ablative Analysis By evaluating our framework by removing each component individually, we demonstrate the contribution of each proposed method. Table IV presents the RMSE ATE and the mean STD of camera tracking results across six scenes from the Bonn dataset. When loop-aware keyframe selection was removed, the performance in largescale environments declined, and the robust mask contributed to stability in pose tracking.

<!-- image-->  
Fig. 4. Visualization of robust mask generation. From right to left: the input image, rendered image, robust mask, and full mask. In the full mask, blue represents the semantic segmentation mask, and red indicates the robust mask.

TABLE III  
CAMERA TRACKING RESULTS ON DYNAMIC SCENES IN THE BONN RGB-D DATASET. THE UNITS FOR ATE AND S.D ARE IN CENTIMETERS (CM).
<table><tr><td>Methods</td><td>Dense</td><td>balloon</td><td colspan="2"></td><td colspan="2">balloon2</td><td colspan="2">ps_track</td><td colspan="2">ps_track2</td><td colspan="2">ball_track</td><td colspan="2">mv_box2</td><td colspan="2">Avg.</td></tr><tr><td>Traditional SLAM methods ORB-SLAM3 [44]</td><td rowspan="6">â</td><td>ATE</td><td>S.D. 2.8</td><td>ATE 17.7</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td></td><td>S.D.</td><td>ATE 3.1</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td></tr><tr><td></td><td>5.8 5.4</td><td></td><td></td><td>8.6</td><td>70.7 21.34</td><td>32.6</td><td>77.9</td><td>43.8</td><td></td><td></td><td>1.6</td><td>3.5</td><td>1.5</td><td>29.8</td><td>15.2</td></tr><tr><td>Droid-VO [46]</td><td></td><td>-</td><td>4.6</td><td>-</td><td></td><td></td><td></td><td>46.0</td><td></td><td>8.9</td><td>-</td><td>5.9</td><td>-</td><td>15.4</td><td>-</td></tr><tr><td>DynaSLAM [15]</td><td>3.0</td><td>-</td><td>2.9</td><td>-</td><td>6.1</td><td></td><td>7.8</td><td></td><td></td><td>4.9</td><td>-</td><td>3.9</td><td></td><td>4.8</td><td></td></tr><tr><td>ReFusion [5]</td><td>17.5</td><td>-</td><td>25.4</td><td>-</td><td>28.9</td><td></td><td>46.3</td><td></td><td>-</td><td>30.2</td><td>-</td><td>17.9</td><td>-</td><td>27.7</td><td>-</td></tr><tr><td>Radiance-Field SLAM methods</td><td></td><td>ATE S.D.</td><td></td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td>ATE</td><td>S.D.</td><td></td><td></td><td>S.D.</td></tr><tr><td>NICE-SLAM [23]</td><td></td><td>X</td><td>X</td><td>66.8</td><td>20.0</td><td>54.9</td><td>27.5</td><td>45.3</td><td></td><td>17.5</td><td>21.2</td><td>13.1</td><td>31.9</td><td>13.6</td><td>ATE 44.1</td><td>18.4</td></tr><tr><td>Vox-Fusion [24]</td><td></td><td>65.7</td><td>30.9</td><td>82.1</td><td>52.0</td><td>128.6</td><td>52.5</td><td>162.2</td><td></td><td>46.2</td><td>43.9</td><td>16.5</td><td>47.5</td><td>19.5</td><td>88.4</td><td>36.3</td></tr><tr><td>Co-SLAM [26]</td><td></td><td>28.8</td><td>9.6</td><td>20.6</td><td>8.1</td><td>61.0</td><td>22.2</td><td>59.1</td><td></td><td>24.0</td><td>38.3</td><td>17.4</td><td>70.0</td><td>25.5</td><td>46.3</td><td>17.8</td></tr><tr><td>ESLAM [25]</td><td></td><td>22.6</td><td>12.2</td><td>36.2</td><td>19.9</td><td>48.0</td><td>18.7</td><td>51.4</td><td></td><td>23.2</td><td>12.4</td><td>6.6</td><td>17.7</td><td>7.5</td><td>31.4</td><td>14.7</td></tr><tr><td>SplaTAM [9]</td><td></td><td>35.7</td><td>14.1</td><td>36.4</td><td>17.4</td><td>124.8</td><td>36.5</td><td>163.0</td><td></td><td>51.3</td><td>12.8</td><td>16.8</td><td>17.9</td><td>9.3</td><td>65.1</td><td>24.2</td></tr><tr><td>MonoGS [10]</td><td></td><td>33.2</td><td>16.4</td><td>26.5</td><td>14.2</td><td>63.2</td><td>29.0</td><td>47.2</td><td></td><td>15.4</td><td>4.3</td><td>2.2</td><td>22.9</td><td>12.4</td><td>32.9</td><td>14.2</td></tr><tr><td>GS-ICP SLAM [11]</td><td></td><td>43.8</td><td>16.0</td><td>42.1</td><td>19.1</td><td>92.8</td><td>42.3</td><td></td><td>44.7</td><td>20.3</td><td>27.9</td><td>17.4</td><td>24.8</td><td>11.5</td><td>31.3</td><td>14.2</td></tr><tr><td>RoDyn-SLAM [45]</td><td>1</td><td>7.9</td><td>2.7</td><td>11.5</td><td>6.1</td><td>14.5</td><td>4.6</td><td>13.8</td><td></td><td>3.5</td><td>13.3</td><td>4.7</td><td>12.6</td><td>4.7</td><td>12.3</td><td>4.4</td></tr><tr><td>DGS-SLAM (Ours)</td><td></td><td>2.9</td><td>0.8</td><td>6.0</td><td>2.8</td><td>9.8</td><td>4.1</td><td>11.1</td><td></td><td>3.9</td><td>5.6</td><td>2.8</td><td>8.8</td><td>3.8</td><td>7.3</td><td>3.0</td></tr></table>

TABLE IV

ABLATION STUDY OF OUR PROPOSED METHODS.
<table><tr><td></td><td>w/o Both</td><td>w/o Robust</td><td>w/o Loop</td><td>DGS-SLAM</td></tr><tr><td>ATE (cm) â</td><td>9.036</td><td>8.343</td><td>8.008</td><td>7.322</td></tr><tr><td>STD (cm) </td><td>3.557</td><td>3.802</td><td>3.164</td><td>3.015</td></tr></table>

Robust Mask Visualization We visualize the robust mask generated during the window optimization in Figure 4. The results of the generated mask demonstrate that our method not only extracts masks corresponding to semantic information but also artifacts like shadows created by dynamic objects. Our robust mask generation is achieved through iteratively updating photometric residuals, leveraging the high-level rendering capabilities of Gaussian splatting.

TABLE V  
TIME ANALYSIS
<table><tr><td></td><td>Total Time (Sec)</td><td>FPS (Hz)</td></tr><tr><td>MonoGS [10]</td><td>776.4</td><td>1.07</td></tr><tr><td>Ours</td><td>519.0</td><td>1.60</td></tr></table>

Time Analysis We measured the total SLAM process time for the fr3/wk xyz sequence from the TUM RGB-D dataset in our workspace, excluding the time spent on semantic segmentation. Our framework took 519 seconds for the total processing time, and when divided by the total number of frames, it achieved a performance of 1.60 FPS as shown in Table V. In comparison, the baseline Gaussian Splatting SLAM [10] showed only 1.07 FPS due to the presence of dynamic elements. Note that it reaches 1.60 FPS in static conditions, similar to our framework. This demonstrates that our framework effectively processes the dynamic SLAM pipeline with minimal additional computation.

## V. CONCLUSION

We present DGS-SLAM, the first dynamic Gaussian Splatting SLAM. Our approach handles dynamic elements not only during the tracking and mapping stages but also across the entire Gaussian Splatting SLAM system, enabling robust pose tracking and reconstruction of the Gaussian splatting. Additionally, to achieve precise localization and mapping performance, we introduce a robust mask generation method that leverages photometric residuals. Furthermore, our loopaware keyframe management accounts for loops with past frames, ensuring consistency to the Gaussian map. Our method demonstrates state-of-the-art performance among radiance-field-based SLAM approaches on two representative dynamic datasets.

[1] Z. Yan, M. Ye, and L. Ren, âDense visual slam with probabilistic surfel map,â IEEE transactions on visualization and computer graphics, vol. 23, no. 11, pp. 2389â2398, 2017.

[2] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison, âElasticfusion: Dense slam without a pose graph.â in Robotics: science and systems, vol. 11. Rome, Italy, 2015, p. 3.

[3] T. Schops, T. Sattler, and M. Pollefeys, âSurfelmeshing: Online surfel- Â¨ based mesh reconstruction,â IEEE transactions on pattern analysis and machine intelligence, vol. 42, no. 10, pp. 2494â2507, 2019.

[4] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon, âKinectfusion: Real-time dense surface mapping and tracking,â in 2011 10th IEEE international symposium on mixed and augmented reality. Ieee, 2011, pp. 127â136.

[5] E. Palazzolo, J. Behley, P. Lottes, P. Giguere, and C. Stachniss, âRefusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals,â in IROS, 2019.

[6] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[7] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM Trans. Graph., 2022.

[8] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[9] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 21 357â21 366.

[10] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 18 039â18 048.

[11] S. Ha, J. Yeon, and H. Yu, âRgbd gs-icp slam,â arXiv preprint arXiv:2403.12550, 2024.

[12] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Real-time simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 584â 21 593.

[13] T. Deng, Y. Chen, L. Zhang, J. Yang, S. Yuan, D. Wang, and W. Chen, âCompact 3d gaussian splatting for dense visual slam,â arXiv preprint arXiv:2403.11247, 2024.

[14] C. Yu, Z. Liu, X.-J. Liu, F. Xie, Y. Yang, Q. Wei, and Q. Fei, âDs-slam: A semantic visual slam towards dynamic environments,â in IROS, 2018.

[15] B. Bescos, J. M. Facil, J. Civera, and J. Neira, âDynaslam: Tracking, Â´ mapping, and inpainting in dynamic scenes,â IEEE RAL, 2018.

[16] T. Zhang, H. Zhang, Y. Li, Y. Nakamura, and L. Zhang, âFlowfusion: Dynamic dense rgb-d slam based on optical flow,â in ICRA, 2020.

[17] Y. Sun, M. Liu, and M. Q.-H. Meng, âMotion removal for reliable rgb-d slam in dynamic environments,â IEEE RAS, 2018.

[18] J. Zhang, M. Henein, R. Mahony, and V. Ila, âVdo-slam: a visual dynamic object-aware slam system,â arXiv preprint, 2020.

[19] Z. Teed and J. Deng, âRaft: Recurrent all-pairs field transforms for optical flow,â in Eur. Conf. Comput. Vis., 2020.

[20] H. Xu, J. Zhang, J. Cai, H. Rezatofighi, and D. Tao, âGmflow: Learning optical flow via global matching,â in IEEE Conf. Comput. Vis. Pattern Recog., 2022.

[21] J. Cheng, Y. Sun, and M. Q.-H. Meng, âImproving monocular visual slam in dynamic environments: An optical-flow-based approach,â Advanced Robotics, 2019.

[22] E. Sucar, S. Liu, J. Ortiz, and A. Davison, âiMAP: Implicit mapping and positioning in real-time,â in Int. Conf. Comput. Vis., 2021.

[23] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in IEEE Conf. Comput. Vis. Pattern Recog., 2022.

[24] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, âVoxfusion: Dense tracking and mapping with voxel-based neural implicit representation,â in ISMAR, 2022.

[25] M. M. Johari, C. Carta, and F. Fleuret, âEslam: Efficient dense slam system based on hybrid representation of signed distance fields,â in IEEE Conf. Comput. Vis. Pattern Recog., 2023.

[26] H. Wang, J. Wang, and L. Agapito, âCo-slam: Joint coordinate and sparse parametric encodings for neural real-time slam,â in IEEE Conf. Comput. Vis. Pattern Recog., 2023.

[27] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald, âPoint- Â¨ slam: Dense neural point cloud-based slam,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 433â18 444.

[28] X. Kong, S. Liu, M. Taher, and A. J. Davison, âvmap: Vectorised object mapping for neural field slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 952â961.

[29] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2d gaussian splatting for geometrically accurate radiance fields,â in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1â11.

[30] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffold-gs: Structured 3d gaussians for view-adaptive rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 654â20 664.

[31] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, âMip-splatting: Alias-free 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 447â19 456.

[32] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 310â20 320.

[33] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, âDeformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 331â20 341.

[34] Q. Gao, Q. Xu, Z. Cao, B. Mildenhall, W. Ma, L. Chen, D. Tang, and U. Neumann, âGaussianflow: Splatting gaussian dynamics for 4d content creation,â arXiv preprint arXiv:2403.12365, 2024.

[35] Z. Li, Z. Chen, Z. Li, and Y. Xu, âSpacetime gaussian feature splatting for real-time dynamic view synthesis,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8508â8520.

[36] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi, âFeature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 676â21 685.

[37] Y. Chen, Z. Chen, C. Zhang, F. Wang, X. Yang, Y. Wang, Z. Cai, L. Yang, H. Liu, and G. Lin, âGaussianeditor: Swift and controllable 3d editing with gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 476â21 485.

[38] H. K. Cheng, S. W. Oh, B. Price, A. Schwing, and J.-Y. Lee, âTracking anything with decoupled video segmentation,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 1316â1326.

[39] S. Sabour, S. Vora, D. Duckworth, I. Krasin, D. J. Fleet, and A. Tagliasacchi, âRobustnerf: Ignoring distractors with robust losses,â in IEEE Conf. Comput. Vis. Pattern Recog., 2023.

[40] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of rgb-d slam systems,â in IROS, 2012.

[41] D. P. Kingma and J. Ba, âAdam: A method for stochastic optimization,â arXiv preprint, 2014.

[42] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., âSegment anything,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015â4026.

[43] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu et al., âGrounding dino: Marrying dino with grounded pre-training for open-set object detection,â arXiv preprint arXiv:2303.05499, 2023.

[44] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, âOrb-slam3: An accurate open-source library for visual, Â´ visualâinertial, and multimap slam,â IEEE TRO, 2021.

[45] H. Jiang, Y. Xu, K. Li, J. Feng, and L. Zhang, âRodyn-slam: Robust dynamic dense rgb-d slam with neural radiance fields,â IEEE RAL, 2024.

[46] Z. Teed and J. Deng, âDROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras,â in Adv. Neural Inform. Process. Syst., 2021.