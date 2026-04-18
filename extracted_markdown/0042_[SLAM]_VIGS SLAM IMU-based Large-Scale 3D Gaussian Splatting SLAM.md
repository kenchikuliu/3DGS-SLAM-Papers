# VIGS SLAM: IMU-based Large-Scale 3D Gaussian Splatting SLAM

Gyuhyeon Pak1 and Euntai Kim1,â

Abstractâ Recently, map representations based on radiance fields such as 3D Gaussian Splatting and NeRF, which excellent for realistic depiction, have attracted considerable attention, leading to attempts to combine them with SLAM. While these approaches can build highly realistic maps, large-scale SLAM still remains a challenge because they require a large number of Gaussian images for mapping and adjacent images as keyframes for tracking. We propose a novel 3D Gaussian Splatting SLAM method, VIGS SLAM, that utilizes sensor fusion of RGB-D and IMU sensors for large-scale indoor environments. To reduce the computational load of 3DGSbased tracking, we adopt an ICP-based tracking framework that combines IMU preintegration to provide a good initial guess for accurate pose estimation. Our proposed method is the first to propose that Gaussian Splatting-based SLAM can be effectively performed in large-scale environments by integrating IMU sensor measurements. This proposal not only enhances the performance of Gaussian Splatting SLAM beyond room-scale scenarios but also achieves SLAM performance comparable to state-of-the-art methods in large-scale indoor environments.

## I. INTRODUCTION

Simultaneous Localization and Mapping (SLAM) is a critical technology in the fields of robotics and autonomous systems, aiming to simultaneously solve the problems of estimating a robotâs location and constructing a map of the surrounding environment. This task is essentially challenging due to the interdependence between localization and mapping. Accurate map construction requires precise localization, and conversely, precise localization depends on an accurate map.

Recently, studies that utilize the concept of radiance fields, such as NeRF [1] or 3D Gaussian Splatting (3DGS) [2], to represent 3D spaces have received significant attention due to their advantages in realistic depiction and fast rendering speeds. These advantages have led to new approaches that combine radiance field-based methods with SLAM [3]â[10], to improve the accuracy and efficiency of both localization and mapping. In particular, 3DGS processes data by considering the uncertainty at each point, allowing for the construction of highly accurate and realistic maps. As a result, 3DGS-based SLAM [5]â[10] (breifly, 3DGS SLAM) has emerged as a promising technology for efficiently handling high-resolution visual data, thereby improving robot localization and environmental understanding.

Basically, 3DGS SLAM [5]â[10] consists of a front-end and a back-end: The front-end is responsible for tracking and localization while the back-end is responsible for optimization and map construction. This paper focuses on the frontend of 3DGS SLAM. Reflecting broader trends in classical SLAM research, there are generally two main approaches in the front-end of 3DGS SLAM: the direct method and the feature-based method.

<!-- image-->  
GroundTruth

<!-- image-->  
Ours (VIGS SLAM)

<!-- image-->  
GS-ICP SLAM  
Fig. 1. Comparison of rendering image quality according to initial estimates on Humans12 (upper 2 rows) and Humans24 (lower 2 rows) of uHumansV1 dataset

The first approach, the direct method [5]â[7], conducts tracking by comparing the raw sensor image with the reconstructed images using dense photometric loss. Tracking in the direct method is easily combined with back-end optimization (i.e., optimization for 3DGS reconstruction), leading to high accuracy. However, this method requires that keyframes are relatively closely spaced due to the sensitivity of the dense photometric loss, thereby storing a large number of keyframes. As a result, the direct method has difficulty in being applied to large-scale environments due to memory limitations.

The second approach is the feature-based method [8]. Since it stores only a limited number of features, it requires significantly less memory to store the map, making it suitable for large-scale environments. However, in featurebased 3DGS SLAM, the front-end tracking and back-end optimization are mostly decoupled, which results in a notable degradation in accuracy. In summary, neither method performs well in terms of both accuracy and scalability to large environments.

To address this challenge, we propose a novel 3DGS SLAM approach called Visual-Inertial Gaussian Splatting (VIGS) SLAM. Our SLAM leverages sensor fusion between an RGB-D sensor and an Inertial Measurement Unit (IMU) for being applied to large-scale indoor environments. The RGB-D sensor provides both visual and depth information, enabling more accurate 3D environment perception, while the IMU captures data related to the robotâs motion by measuring acceleration and angular velocity. By effectively fusing data from these two sensors, the performance of the SLAM system is significantly enhanced.

Combining IMU and 3DGS SLAM is not completely new, but a paper which applies IMU to direct method was very recently reported [9]. However, we believe that combining dense direct tracking with IMU is not the optimal solution. This approach can only slightly increase the distance between keyframes (i.e., it can slightly reduce the number of keyframes) due to the dense nature of photometric loss, making it difficult to apply to large-scale environments.

Instead, we propose applying IMU data not to raw image matching, but to the matching of the point cloud generated from the depth image. This new combination overcomes the limitations of dense photometric loss, significantly increasing the distance between keyframes (thereby significantly reducing the number of keyframes and memory usage) while maintaining the connection between the front-end and backend, thus preserving high accuracy. Thus, we believe that our method can achieve both high accuracy and large-scale coverage. Our approach uses [10] as a base line and has the following contributions.

â We propose a large-scale visual-inertial SLAM framework, VIGS SLAM, leveraging sensor fusion between RGB-D and IMU sensor, inspired by GS-ICP SLAM [10].

â By utilizing the IMU preintegration values between consecutive frames, we improve the accuracy of the initial guesses. These good initial guesses play a crucial role in improving the accuracy of both pose estimation and mapping vy provinding more precise inputs to the Iterative Closest Point (ICP) algorithm.

â We have developed a SLAM system that outperforms existing room-scale 3DGS SLAM, enabling efficient operation in large-scale indoor environments. Our approach represents a significant advance in large-scale indoor SLAM, demonstrating both scalability and robustness comparable to state-of-the-art SLAM systems.

The remainder of this paper is organized as follows: In Section II, we review related works on SLAM leveraging multiple sensor fusion and 3D GS SLAM. In Section III, we provide a detailed description of the proposed method, including Generalized ICP tracking, IMU preintegration, 3D Gaussian Splatting mapping. Section IV presents experimental results demonstrating the qualitative and quantitative performance of the proposed method in large-scale indoor environments. Finally, in Section V, we discuss the conclusions of the proposed method and potential directions for

future research.

## II. RELATED WORKS

## A. Sensor Fusion for SLAM

Although visual perception through camera sensors is effective, it faces limitations such as motion blur, changes in exposure, and sensitivity to lighting conditions. One way to overcome these limitations is through the fusion of complementary sensors, which compensates for the weaknesses of each sensor. In this regard, previous approaches have enhanced SLAM performance by integrating depth sensors [11], [12], LiDAR [13]â[15], and IMU sensors [16]â[19] with visual SLAM, thereby addressing issues like scale ambiguity and improving the robustness of SLAM systems. In this paper, we focus on the fusion with IMU sensors, which are particularly effective for their ability to track rapid movements within short time frames and their high data acquisition rate, making them ideal for integrating inertial measurements.

## B. 3DGS SLAM

3D Gaussian Splatting (3DGS) based methods [5]â[7] have detailed the significant advantages of 3DGS over traditional map representations in SLAM tasks for online photorealistic mapping. Additionally, they highlight that the vanilla 3DGS must be appropriately adapted for efficient application in SLAM. SplaTAM [5] and GS-SLAM [7] describe the benefits of 3DGS map representations over conventional SLAM map representations. MonoGS [6], using a single RGB sensor, addresses the ambiguity of incremental reconstruction by introducing geometric regularization, making it work well with monocular images.

In contrast, methods such as GS-ICP SLAM [10] and Photo-SLAM [8] estimate the necessary poses for mapping using classical visual odometry to estimate camera movement, thereby enabling sufficiently fast SLAM operations. Photo-SLAM [8] leverages the classical visual odometry method, ORB-SLAM3 [20], for accurate pose estimation and reconstructs a hybrid Gaussian map that incorporates ORB features. GS-ICP SLAM [10] performs pose tracking between successive point clouds using ICP-based tracking [21] results and incorporates the covariance of each point, obtained during tracking, into 3DGS mapping to achieve real-time SLAM.

## III. METHOD

The VIGS SLAM framework consists of three main stages: 1) Generalized ICP tracking, 2) IMU preintegration, 3) 3D Gaussian Splatting mapping. An overview of the framework is described in Fig. 2.

## A. Generalized ICP Tracking

Let us suppose that the RGB image $I _ { k }$ and depth image $D _ { k }$ are presented at the kth frame. The corresponding point cloud $\pmb { \mathcal { X } } _ { k } = \{ \boldsymbol { x } _ { m } \} _ { m = 1 , \dots , M _ { k } }$ is obtained from $D _ { k }$ , and the associated covariance set $\pmb { \Sigma } _ { k } = \{ \Sigma _ { m } \} _ { m = 1 , \dots , M _ { k } }$ is given by computing the covariance matrix of k-nearest neighbors of each point $x _ { m }$ , where $x _ { m } \in \mathbb { R } ^ { 3 }$ and $\Sigma _ { m } \in \mathbb { R } ^ { 3 \times 3 }$ are a 3D point and the corresponding covariance matrix, respectively, and m is an index for $\scriptstyle { \mathcal { X } } _ { k }$ and $\Sigma _ { k }$ . Following the process, we can obtain the set of Gaussians $\boldsymbol { G } _ { k } = \{ \boldsymbol { x } _ { k } , \Sigma _ { k } \}$ corresponding to the point cloud at each frame.

<!-- image-->  
Fig. 2. Overview of VIGS SLAM. The input of VIGS SLAM is RGB-D image and IMU meausrements. The system generates a point cloud from RGB and depth inputs, followed by GICP tracking. The IMU preintegration values are used as a good initial guess to enhance tracking performance, and these values are updated after tracking each frame. Keyframes are identified for 3D Gaussian Splatting-based mapping, which efficiently updates the 3DGS map with detailed environmental representations.

The relative transformation $\mathbfit { T } _ { k }$ between the source Gaussians $G _ { k } ^ { s r c } \ = \ \{ \pmb { \mathcal { X } } _ { k } ^ { s r c } , \pmb { \Sigma } _ { k } ^ { s r c } \}$ generated from current RGB-D image $\left\{ I _ { k } , ~ D _ { k } \right\}$ and corresponding target Gaussian set $G _ { k } ^ { t g t } \ = \ \{ \dot { \pmb { { \mathscr { X } } } } _ { k } ^ { t g t } , { \pmb { \Sigma } } _ { k } ^ { t \dot { g } t } \}$ that constitutes the map M can be estimated using Generalized ICP (GICP) [21]. By modeling each point $x _ { m }$ as a Gaussian distribution $N ( { \hat { x } } _ { m } , \Sigma _ { m } )$ , the distance $d _ { m } ~ = ~ x _ { m } ^ { t g t } - \pmb { T } _ { k } x _ { m } ^ { s r c }$ between the corresponding distributions pair in the GICP framework is given by

$$
d _ { m } \sim N ( \hat { d } _ { m } , \Sigma _ { m } ^ { t g t } + \pmb { T } _ { k } \Sigma _ { m } ^ { s r c } ( \pmb { T } _ { k } ) ^ { T } )
$$

$$
= N \big ( \hat { x } _ { m } ^ { t g t } - \pmb { T } _ { k } \hat { x } _ { m } ^ { s r c } , \pmb { \Sigma } _ { m } ^ { t g t } + \pmb { T } _ { k } \pmb { \Sigma } _ { m } ^ { s r c } ( \pmb { T } _ { k } ) ^ { T } \big ) .\tag{1}
$$

(2)

We derive the optimal transformation $\boldsymbol { \mathbf { \mathit { T } } } _ { k } ^ { * }$ by applying the maximum likelihood estimation method to the previously expressed distance function. The optimal transformation $\boldsymbol { T } _ { k } ^ { * }$ is obtained by maximizing the likelihood function:

$$
\pmb { T } _ { k } ^ { * } = \underset { \pmb { T } _ { k } } { \operatorname { a r g m a x } } \prod _ { m = 1 } ^ { N } p ( d _ { m } ) = \underset { \pmb { T } _ { k } } { \operatorname { a r g m a x } } \sum _ { m = 1 } ^ { N } \log p ( d _ { m } )\tag{3}
$$

$$
= \underset { { \pmb T } _ { k } } { \arg \operatorname* { m a x } } \sum _ { m = 1 } ^ { N } d _ { m } ^ { T } \big ( \Sigma _ { m } ^ { t g t } + { \pmb T } _ { k } \Sigma _ { m } ^ { s r c } ( { \pmb T } _ { k } ) ^ { T } \big ) ^ { - 1 } d _ { m } .\tag{4}
$$

GICP performs camera tracking by alternating between (1) finding the correspondences between the closest points, and (2) minimizing the Euclidean distance between these point pairs. In each iteration, the correspondences are recalculated, and the tracnsformation is updated through the optimization process. If the initial correspondences are accurate, even when the camera moves rapidly, GICP enables fast and reliable tracking, reducing mapping errors. The key idea of this paper is to use IMU preintegration as the intial estimate for the correspondences, allowing for efficient and reliable tracking. An example result is shown in Fig. 3. As illustrated, the mapping results of our baseline method are significanty skewed, whereas our VIGS SLAM demonstrates excellent mapping accuracy.

<!-- image-->  
Fig. 3. Comparison of map reconstruction on uHumansV1. GS-ICP SLAM (left), VIGS SLAM (right)

## B. IMU Preintegration

The IMU sensor outputs linear acceleration $\begin{array} { r l } { \hat { a } } & { { } = } \end{array}$ $\left[ \hat { a } _ { x } , \hat { a } _ { y } , \hat { a } _ { z } \right] ^ { T } \in \mathbb { R } ^ { 3 }$ and angular velocity $\hat { \omega } = [ \hat { \omega } _ { x } , \hat { \omega } _ { y } , \hat { \omega } _ { z } ] ^ { T } \in$ $\mathbb { R } ^ { 3 }$ and helps measure the sensorâs motion along these 6 degrees of freedom (DoF). Since the IMU operates at a much higer frequency than the camera, we pre-integrate the IMU sensorâs measurements between RGB-D frames and use the pre-integration results as the initial guess for the sensor motion in GICP tracking part. In this paper we use k as the time index for RGB-D frames, whereas use t as the time index for the IMU sensor.

Since the measurements include inherent noise and bias of the sensor, the corrected linear acceleration $a _ { t }$ and angular velocity $\omega _ { t }$ are given by the following equations:

$$
\hat { a } _ { t } = a _ { t } + \infty R ^ { w } g + b _ { a } + n _ { a }\tag{5}
$$

$$
\hat { \omega } _ { t } = \omega _ { t } + b _ { \omega } + n _ { \omega } ,\tag{6}
$$

where $_ w ^ { R }$ represents transpose of the sensorâs rotation, $^ w g$ represents gravity vector, and $b _ { a } , b _ { g } , n _ { a }$ , and $n _ { g }$ denote the biases and noises of the accelerometer and gyroscope, respectively.

Using the kinematic model [18], the current frameâs relative position can be predicted based on the IMU measure-

ments.

$$
p _ { t + 1 } ^ { k } = p _ { t } ^ { k } + v _ { t } ^ { k } \Delta t + \frac { 1 } { 2 } \big ( \hat { a } _ { t } - _ { w } R ^ { w } g - b _ { a } - n _ { a } \big ) \Delta t ^ { 2 }\tag{7}
$$

$$
v _ { t + 1 } ^ { k } = v _ { t } ^ { k } + \big ( \hat { a } _ { t } - _ { w } R ^ { w } g - b _ { a } - n _ { a } \big ) \Delta t\tag{8}
$$

$$
R _ { t + 1 } ^ { k } = R _ { t } ^ { k } \mathrm { E x p } \{ ( \hat { \omega } _ { t } - b _ { \omega } - n _ { \omega } ) \Delta t \} ,\tag{9}
$$

where IMU sensorâs position, velocity, and rotation are denoted as $p _ { t } ^ { k } , v _ { t } ^ { k } , R _ { t } ^ { k }$ , and ât is time difference between time t and $t + 1$

To avoid repetitive computations for the motion between consecutive frames, the IMU sensorâs preintegration [16] values can be expressed as follows:

$$
\alpha _ { k + 1 } ^ { k } = \iint _ { t \in [ t _ { k } , t _ { k + 1 } ] } R _ { t } ^ { k } \big ( \hat { a } _ { t } - b _ { a } \big ) d t ^ { 2 }\tag{10}
$$

$$
\beta _ { k + 1 } ^ { k } = \int _ { t \in \left[ t _ { k } , t _ { k + 1 } \right] } R _ { t } ^ { k } ( \widehat { a } _ { t } - b _ { a } ) d t\tag{11}
$$

$$
\gamma _ { k + 1 } ^ { k } = \int _ { t \in \left[ t _ { k } , t _ { k + 1 } \right] } \frac 1 2 \Omega ( \hat { \omega } _ { t } - b _ { \omega } ) \gamma _ { t } ^ { k } d t ,\tag{12}
$$

where $\Omega ( \omega ) = \left[ \begin{array} { c c } { - \lfloor \omega \rfloor _ { \times } } & { \omega } \\ { - \omega ^ { T } } & { 0 } \end{array} \right] , \ \lfloor \omega \rfloor _ { \times } = \left[ \begin{array} { c c c } { 0 } & { - \omega _ { z } } & { \omega _ { y } } \\ { \omega _ { z } } & { 0 } & { - \omega _ { x } } \\ { - \omega _ { y } } & { \omega _ { x } } & { 0 } \end{array} \right]$ â£ â¦represents the skew-symmetric matrix associated with the angular velocity Ï.

Using (7), (8), and (9), the relative transformation between consecutive frames, $\mathbf { \Lambda } ^ { I } \mathbfcal { T } _ { k } ^ { k - 1 } = \big [ R _ { k } ^ { k - 1 } | p _ { k } ^ { k - 1 } \big ]$ , can be computed. We transform the relative transformation into camera coordinate using an extrinsic parameter, ${ } _ { I } ^ { C } { \pmb T } ,$ , between camera and the IMU sensor. We can obtain the good intial guess, ${ } ^ { C } { \pmb T } _ { k } ,$ for GICP tracking from the IMU preintegration.

$$
{ } ^ { C } { \pmb T } _ { k } ^ { k - 1 } = { } _ { I } ^ { C } { \pmb T } ^ { I } { \pmb T } _ { k } ^ { k - 1 }\tag{13}
$$

$$
{ } ^ { C } { \pmb T } _ { k } = { } ^ { C } { \pmb T } _ { k } ^ { k - 1 } { \pmb C } _ { { \pmb T } _ { k - 1 } . }\tag{14}
$$

## C. Update Preintegration

IMU-based GICP tracking allows for faster and more reliable performance compared to the original GICP method. This is because it leverages the accurate initial guess provided by the IMU, rather than estimating the initial guess based on the previous frameâs position or using a constant velocity model. IMU preintegration continuously integrates accelerations and angular velocities over time. However, due to the fact that the IMU sensor captures continuous measurements, drift accumulates at low speeds, which results in a degradation of the tracking performance over time caused by the accumulation of integration errors. To address such degradation, it is necessary to update the initial values of the IMU preintegration parametersâposition $p _ { t } ^ { k }$ , velocity $v _ { t } ^ { k }$ , and rotation $R _ { t } ^ { k }$ âusing the optimized transformation $^ { C } T _ { k } ^ { * } = [ ^ { C } R _ { k } ^ { * } | ^ { C } p _ { k } ^ { * } ]$ obtained from GICP tracking.

$$
{ } ^ { I } T _ { k } ^ { * } = { } _ { C } { \cal T } ^ { C } T _ { k } ^ { * } ,\tag{15}
$$

$$
p _ { t } ^ { k }  { ^ I } p _ { k } ^ { * } , R _ { t } ^ { k }  { ^ I } R _ { k } ^ { * } , v _ { t } ^ { k }  \frac { { ^ I } p _ { k } ^ { * } - { ^ I } p _ { k - 1 } ^ { * } } { \Delta t } ,\tag{16}
$$

where $\Delta t$ represents the time interval between consecutive RGB-D frames, and $^ C p _ { k } ^ { * }$ and $^ C { \cal R } _ { k } ^ { * }$ denote the results obtained from GICP tracking for position and rotation, respectively.

## D. 3D Gaussian Splatting Mapping

In this paper, the map representation is based on 3D Gaussian Splatting [2], using a set of Gaussian models G to represent the 3D space. Each Gaussian $G _ { m }$ is defined by its RGB color $c _ { m }$ , opacity $\sigma _ { m } .$ , center position $\mu _ { m } ,$ and scale $s _ { i } .$ . As mentioned in the section III-A, the input point cloud for the GICP tracking part is expressed as a Gaussian distribution $G _ { m }$ , with its center $\mu _ { m }$ and covariance $\Sigma _ { m }$ calculated accordingly. These values are then reused in the mapping process, eliminating the need for redundant computations.

$$
G _ { m } ( x ) = \operatorname { E x p } ( - \frac 1 2 ( x - \mu _ { m } ) \Sigma _ { m } ^ { - 1 } ( x - \mu _ { m } ) ^ { T } )\tag{17}
$$

The covariance can be expressed through singular value decomposition as $\Sigma _ { m } ~ = ~ R \Lambda ^ { 2 } R ^ { T }$ , where R represents the orientation of the Gaussian, and $\Lambda = d i a g ( s _ { 2 } , s _ { 1 } , s _ { 0 } )$ denotes the scale matrix. The map represented using the 3DGS method is rendered into a 2D image through volume rendering, where the color values of the pixels in the image are determined by the contributions of the  Gaussians that make up the pixel p.

The color of a pixel $C _ { p }$ can be computed using the following equation:

$$
C _ { p } = \sum _ { m \in \mathcal { N } } c _ { m } \alpha _ { m } \prod _ { n = 1 } ^ { m - 1 } \big ( 1 - \alpha _ { n } \big ) ,\tag{18}
$$

where $c _ { i }$ denotes the color of the ith Gaussian, and $\alpha _ { i }$ represents the opacity sampled from the ith Gaussian distribution at the pixel position. Similarly, the opacity at a pixel position can be calculated as follows:

$$
O _ { p } = \sum _ { m \in \mathcal { N } } \alpha _ { m } \prod _ { n = 1 } ^ { m - 1 } \left( 1 - \alpha _ { n } \right)\tag{19}
$$

To optimize the Gaussians representing the rendered images, we utilize the dense photometric loss $L _ { p h o t o }$ , which is the L1 loss between the rendered and observed images, the SSIM image loss $\mathcal { L } _ { S S I M }$ , and the depth loss $\mathcal { L } _ { d e p t h }$ , which is the L1 loss between the rendered and observed depth images. The combined mapping loss function is given by:

$$
\mathcal { L } _ { m a p p i n g } = ( 1 - \lambda _ { I } ) \mathcal { L } _ { p h o t o } + \lambda _ { I } \mathcal { L } _ { S S I M } + \lambda _ { D } \mathcal { L } _ { d e p t h }\tag{20}
$$

While we employ the same loss functions as the original 3DGS for mapping, unlike traditional 3DGS, where all Gaussians are optimized simultaneously, SLAM tasks progressively introduce new Gaussians. To prevent scale imbalance between Gaussians introduced early and those added later in the process, we apply scale normalization [10].

## IV. EXPERIMENTS

The proposed method was evaluated using the photorealistic and large-scale visual-inertial datasets uHumansV1 [22] and uHumansV2 [19]. The uHumansV1 dataset was collected in a 65m  65m office space, with each scenario containing varying numbers of people, specifically 12, 24, and 60 individuals. This dataset is well-suited for visual inertial odometry and visual inertial SLAM, although it has the limitation of providing only grayscale image data. We also conducted experiments on the uHumansV2 dataset ranging from small apartment scenes to large office scenes, to evaluate the mapping and rendering performance of the proposed algorithm given RGB images.

TABLE I  
TRACKING AND RENDERING PERFORMANCE COMPARED TO EXISTING METHODS ON UHUMANSV1 DATASET
<table><tr><td rowspan="2" colspan="2">uHumansV1 [22]</td><td colspan="4">Humans12</td><td colspan="4">Humans24</td><td colspan="4">Humans60</td></tr><tr><td>ATEâ(cm) PSNR SSIM âLPIPSâATE â (cm) PSNR SSIM âLPIPS âATE â (cm) PSNR âSSIM âLPIPS â</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Traditional VIO</td><td>ORB-SLAM3 [12]</td><td>191.76</td><td>-</td><td>-</td><td>-</td><td>15.82</td><td>-</td><td></td><td>-</td><td>17.15</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VINS-Mono [16]</td><td>18.85</td><td></td><td></td><td></td><td>25.60</td><td></td><td></td><td>-</td><td>22.29</td><td>-</td><td></td><td>-</td></tr><tr><td rowspan="3">3DGS</td><td>MonoGS [6]</td><td>603.50</td><td>20.99</td><td>0.714</td><td>0.503</td><td>1138.15</td><td>21.35</td><td>0.740</td><td>0.499</td><td>970.68</td><td>17.49</td><td>0.637</td><td>0.612</td></tr><tr><td>PhotoSLAM [8]</td><td>650.97</td><td>14.27</td><td>0.596</td><td>0.598</td><td>1573.21</td><td>14.23</td><td>0.609</td><td>0.592</td><td>1315.79</td><td>14.78</td><td>0.596</td><td>0.587</td></tr><tr><td>GS-ICP SLAM [10]</td><td>677.87 35.35</td><td>22.89</td><td>0.788 0.849</td><td>0.345 0.205</td><td>776.79 25.03</td><td>23.49 25.89</td><td>0.794 0.853</td><td>0.334</td><td>973.49</td><td>20.85</td><td>0.741</td><td>0.422</td></tr><tr><td colspan="10">Ours (VIGS SLAM)</td><td colspan="5">46.86 23.91</td></tr><tr><td colspan="9"></td><td colspan="5"></td></tr></table>

The algorithm with the highest performance is indicated in bold, and the second-best is underlined. Traditional VIO methods are excepted in best algorithm.

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 4. Trajectory in Humans12(left), Humans24(middle), Humans60(right) of uHumansV1, compared with ORB-SLAM3, VINS-Mono, GS-ICP SLAM, and VIGS SLAM(ours).

## A. Experiments Setup

All experiments were conducted on a desktop equipped with an Intel i5-12500 CPU, 32GB of RAM, and an NVIDIA RTX A5000 GPU with 24GB of memory. The accuracy of tracking was evaluated using the Root Mean Square Error (RMSE) of the Absolute Trajectory Error (ATE) [23]. To assess the mapping and rendering processes, metrics such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) were utilized. The methods used for comparison include the recently proposed 3DGS SLAM approaches such as MonoGS [6] and PhotoSLAM [8] as well as ORB-SLAM3 [20] and VINS-Mono [16]. In addition, GS-ICP SLAM [10] was included as a baseline.

## B. Performance on the uHumansV1

To evaluate the performance of our method compared to state-of-the-art method, we conducted experiments on uHumansV1, as detailed results are shown in Tab. I.

Tracking Performance. VIGS SLAM demonstrates superior tracking performance compared to existing 3DGS SLAM approaches, achieving tracking performance on par with traditional VIO methods, as shown in Fig.4. These differences stem from the lower similarity between consecutive images in large-scale environments with dynamic objects. While existing 3DGS SLAM methods are effective in room-scale static datasets, the uHumansV1 dataset presents a greater challenge due to its larger spatial coverage over the same time frame, reducing the similarity between consecutive images. Additionally, 3DGS methods that rely on dense photometric loss for tracking are more vulnerable to dynamic objects, thus limiting their performance. Compared with GS-ICP SLAM, VIGS SLAM outperforms in all scenarios and significantly reduces the tracking error from a few meters to several tens of centimeters. These results indicate that by employing IMU preintegration values as the initial guess, our method achieves a notable tracking accuracy.

Mapping & Rendering Performance. As demonstrated in Tab. I, the VIGS SLAM achieves the highest rendering performance. Since MonoGS optimize the dense photometric loss for tracking and PhotoSLAM utilize the sparse feature based tracking, poor tracking performance of MonoGS and PhotoSLAM leads to poor rendering performance. GS-ICP SLAM shows relatively high rendering performance, but the mapping performance deteriorates due to poor tracking performance in the revisited area, as shown in Fig. 3 and Fig. 4. Fig. 1 and 3 highlight the superior rendering and mapping performance of the proposed method compared to GS-ICP SLAM. In particular, VIGS SLAM successfully renders images with humans cleanly excluded. This result indicates that the integration of IMU sensor enables robust correspondences even in the presence of dynamic objects.

<!-- image-->  
GroundTruth

<!-- image-->  
Ours (VIGS SLAM)

<!-- image-->  
GS-ICP SLAM

<!-- image-->  
MonoGS

<!-- image-->  
PhotoSLAM  
Fig. 5. Qualitative rendering results on the apartment scene of uHumansV2 dataset, compared with VIGS SLAM(ours), GS-ICP SLAM, MonoGS, and PhotoSLAM.

TABLE II  
TRACKING PERFORMANCE COMPARED TO EXISTING METHODS ON UHUMANSV2 DATASET (ATE RMSE IS IN CM)
<table><tr><td colspan="2">uHumansV2 [19]</td><td>Apartment</td><td>Office</td></tr><tr><td rowspan="3">3DGS</td><td>MonoGS [6]</td><td>16.64</td><td>802.86</td></tr><tr><td>PhotoSLAM [8] GS-ICP SLAM [10]</td><td>285.99 29.21</td><td>34.74 674.38</td></tr><tr><td>Ours (VIGS SLAM)</td><td>16.17</td><td>144.72</td></tr></table>

The best-performing algorithms are shown in bold, and second best is underlined

TABLE III  
RENDERING PERFORMANCE COMPARED TO EXISTING METHODS ON UHUMANSV2 DATASET
<table><tr><td rowspan="2" colspan="2">uHumansV2 [19]</td><td colspan="3">Apartment</td><td colspan="3">Office</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPSâ</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td rowspan="3">3DGS</td><td>MonoGS [6]</td><td>29.68</td><td>0.883</td><td>0.196</td><td>20.56</td><td>0.696</td><td>0.596</td></tr><tr><td>PhotoSLAM [8]</td><td>19.13</td><td>0.691</td><td>0.481</td><td>14.35</td><td>0.585</td><td>0.646</td></tr><tr><td>GS-ICP SLAM [10]</td><td>26.54</td><td>0.813</td><td>0.333</td><td>22.92</td><td>0.773</td><td>0.378</td></tr><tr><td colspan="2">Ours (VIGS SLAM)</td><td>27.26</td><td>0.826</td><td>0.289</td><td>23.88</td><td>0.793</td><td>0.330</td></tr></table>

The best-performing algorithms are shown in bold, and second best is underlined

## C. Performance on the uHumansV2

As reported in Tab. II and Tab. III, the results on the uHumansV2 dataset demonstrate that our method also performs well in environments where RGB images are provided. While MonoGS and GS-ICP SLAM demonstrate high tracking accuracy in the small-scale apartment scene, they notably degrade in the large-scale office scene. Although PhotoSLAM achieves low translation errors in the office environment, it suffers from decoupling between front-end and back-end, leading to poor overall mapping performance in both scenes. In contrast, our method shows consistently robust tracking and rendering performance across both environments.

Fig. 5 illustrates a qualitative rendering results from our method and those of the comparison methods. Most methods fail to accurately render the decoration on the right shelf because they are too thin in the image in the first row, but our method successfully captures these details. In the third row, methods with poor tracking performance produce maps with incorrect poses, resulting in low-quality rendered image. In contrast, our method generates high-quality image from accurate poses. The last row corresponds to the end part of the sequence, where less observations were made around the door. Despite strong tracking performance of MonoGS, it renders a low-quality image in this area. On the other hand, VIGS SLAM, which utilizes the dense point cloud matching, maintains robust rendering quality.

## V. CONCLUSIONS & FUTURE WORK

In this paper, we proposed VIGS SLAM, a novel framework that leverages visual, depth, and inertial measurements to achieve improved tracking performance and 3D map reconstruction in large-scale environments. To evaluate the effectiveness of our approach, we conducted experiments on the uHumansV1 and uHumansV2 datasets, which provide large-scale visual-inertial data. As a result, we demonstrate that our method outperforms SOTA methods through tracking and mapping performance comparisons in most datasets.

As future work, we intend to develop a tightly-coupled visual-inertial framework. Furthermore, we will integrate a loop-closing module to address long-term drift, thereby ensuring more robust and accurate SLAM performance.

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[3] A. Rosinol, J. J. Leonard, and L. Carlone, âNerf-slam: Real-time dense monocular slam with neural radiance fields,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 3437â3444.

[4] M. M. Johari, C. Carta, and F. Fleuret, âEslam: Efficient dense slam system based on hybrid representation of signed distance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 408â17 419.

[5] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[6] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[7] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, âGsslam: Dense visual slam with 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 595â19 604.

[8] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Real-time simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 584â 21 593.

[9] L. C. Sun, N. P. Bhatt, J. C. Liu, Z. Fan, Z. Wang, T. E. Humphreys, and U. Topcu, âMm3dgs slam: Multi-modal 3d gaussian splatting for slam using vision, depth, and inertial measurements,â arXiv preprint arXiv:2404.00923, 2024.

[10] S. Ha, J. Yeon, and H. Yu, âRgbd gs-icp slam,â arXiv preprint arXiv:2403.12550, 2024.

[11] T. Whelan, H. Johannsson, M. Kaess, J. J. Leonard, and J. McDonald, âRobust real-time visual odometry for dense rgb-d mapping,â in 2013 IEEE International Conference on Robotics and Automation. IEEE, 2013, pp. 5724â5731.

[12] R. Mur-Artal and J. D. Tardos, âOrb-slam2: An open-source slam Â´ system for monocular, stereo, and rgb-d cameras,â IEEE transactions on robotics, vol. 33, no. 5, pp. 1255â1262, 2017.

[13] S. Hong, J. He, X. Zheng, C. Zheng, and S. Shen, âLiv-gaussmap: Lidar-inertial-visual fusion for real-time 3d radiance field map rendering,â IEEE Robotics and Automation Letters, 2024.

[14] T. Shan, B. Englot, C. Ratti, and D. Rus, âLvi-sam: Tightly-coupled lidar-visual-inertial odometry via smoothing and mapping,â in 2021 IEEE international conference on robotics and automation (ICRA). IEEE, 2021, pp. 5692â5698.

[15] X. Lang, L. Li, H. Zhang, F. Xiong, M. Xu, Y. Liu, X. Zuo, and J. Lv, âGaussian-lic: Photo-realistic lidar-inertial-camera slam with 3d gaussian splatting,â arXiv preprint arXiv:2404.06926, 2024.

[16] T. Qin, P. Li, and S. Shen, âVins-mono: A robust and versatile monocular visual-inertial state estimator,â IEEE transactions on robotics, vol. 34, no. 4, pp. 1004â1020, 2018.

[17] P. Geneva, K. Eckenhoff, W. Lee, Y. Yang, and G. Huang, âOpenvins: A research platform for visual-inertial estimation,â in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 4666â4672.

[18] C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza, âImu preintegration on manifold for efficient visual-inertial maximum-a-posteriori estimation,â in Robotics: Science and Systems XI, 2015.

[19] A. Rosinol, A. Violette, M. Abate, N. Hughes, Y. Chang, J. Shi, A. Gupta, and L. Carlone, âKimera: from SLAM to spatial perception with 3D dynamic scene graphs,â Intl. J. of Robotics Research, vol. 40, no. 12â14, pp. 1510â1546, 2021, arXiv preprint: 2101.06894, PDF.

[20] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, âOrb-slam3: An accurate open-source library for visual, Â´ visualâinertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[21] K. Koide, M. Yokozuka, S. Oishi, and A. Banno, âVoxelized gicp for fast and accurate 3d point cloud registration,â in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021, pp. 11 054â11 059.

[22] A. Rosinol, A. Gupta, M. Abate, J. Shi, and L. Carlone, â3d dynamic scene graphs: Actionable spatial perception with places, objects, and humans,â arXiv preprint arXiv:2002.06289, 2020.

[23] M. Grupp, âevo: Python package for the evaluation of odometry and slam.â https://github.com/MichaelGrupp/evo, 2017.

[24] C. Wang, D. Gao, K. Xu, J. Geng, Y. Hu, Y. Qiu, B. Li, F. Yang, B. Moon, A. Pandey, Aryan, J. Xu, T. Wu, H. He, D. Huang, Z. Ren, S. Zhao, T. Fu, P. Reddy, X. Lin, W. Wang, J. Shi, R. Talak, K. Cao, Y. Du, H. Wang, H. Yu, S. Wang, S. Chen, A. Kashyap, R. Bandaru, K. Dantu, J. Wu, L. Xie, L. Carlone, M. Hutter, and S. Scherer, âPyPose: A library for robot learning with physics-based optimization,â in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.