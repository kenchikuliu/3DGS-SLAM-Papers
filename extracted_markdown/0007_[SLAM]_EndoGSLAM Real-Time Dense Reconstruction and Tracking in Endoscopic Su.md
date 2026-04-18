# EndoGSLAM: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries using Gaussian Splatting

Kailing Wang\*1, Chen Yang\*1, Yuehao Wang2, Sikuang Li1, Yan Wang3, Qi Dou2, Xiaokang Yang1, Wei Shen1â 

1 MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University 2 Dept. of Computer Science and Engineering, The Chinese University of Hong Kong 3 Shanghai Key Laboratory of Multidimensional Information Processing, East China Normal University

Abstract. Precise camera tracking, high-fidelity 3D tissue reconstruction, and real-time online visualization are critical for intrabody medical imaging devices such as endoscopes and capsule robots. However, existing SLAM (Simultaneous Localization and Mapping) methods often struggle to achieve both complete high-quality surgical field reconstruction and efficient computation, restricting their intraoperative applications among endoscopic surgeries. In this paper, we introduce EndoGSLAM, an efficient SLAM approach for endoscopic surgeries, which integrates streamlined Gaussian representation and differentiable rasterization to facilitate over 100 fps rendering speed during online camera tracking and tissue reconstructing. Extensive experiments show that EndoGSLAM achieves a better trade-off between intraoperative availability and reconstruction quality than traditional or neural SLAM approaches, showing tremendous potential for endoscopic surgeries. The project page is at https://EndoGSLAM.loping151.com

Keywords: Endoscopic surgeries Â· SLAM Â· Real-time rendering Â· Tissue reconstruction.

## 1 Introduction

Endoscopy, a minimally invasive technique for examining and treating internal organs and passages, relies heavily on the skill and precision of operators, especially during complex surgical procedures. This reliance underscores the vital need for advanced visualization systems that enhance the surgeonâs field of view, aid in pinpointing critical areas, and facilitate safer and more efficacious surgical interventions. Key technologies such as endoscopic reconstruction and tracking play a pivotal role in surgical visualization, with Simultaneous Localization and Mapping (SLAM) being a common choice for them [2,1].

One ideal SLAM approach for surgeries should support online tracking and reconstruction. More importantly, it should enable real-time online visualization of reconstruction, which means it can simultaneously perform tracking, reconstructing, and rendering, allowing surgeons to review any area of interest among previously observed regions at any time. Additionally, the method should achieve precise localization and induce complete and high-quality reconstructions.

<!-- image-->  
Fig. 1: Comparative Visualization of Novel View Synthesis. From left to right, we show the holistic rendering from EndoGSLAM, the ground truth of one given viewpoint, renderings of EndoGSLAM, NICESLAM [31] and Endo-Depth [17]. These comparisons highlight EndoGSLAMâs superior fidelity.

Traditional SLAM approaches [5,11,22] often yield sparse geometric representations, primarily serving to facilitate endoscope tracking since geometric features are scarce and unreliable among endoscopic procedures [13]. To address this, some approaches [19,9,10,26,15,17,16,6] have adopted appearance-based optimization for dense mapping and enhanced tracking precision. However, these methods struggle to achieve fine-grained dense reconstructions, impacting novel view rendering and limiting their effectiveness in real-world surgical applications.

Recent advancements in neural rendering, especially Neural Radiance Fields (NeRF) [12] and 3D Gaussian Splatting [7], have shown promise for high-fidelity surgical reconstruction [28,27,24]. Several methods [31,21,18,8,23,30] are proposed to integrate NeRF with SLAM. Implicit neural representations, despite offering detailed global maps and photometric capture via differentiable rendering, incur high computational costs, which necessitate pixel sampling for efficiency. This hinders their intraoperative viability in endoscopic contexts.

In this paper, we propose a novel SLAM approach designed for endoscopic surgeries, EndoGSLAM, which simultaneously performs online precise camera tracking, high-quality dense reconstruction, and real-time novel view synthesis. Specifically, EndoGSLAM designs a simplified Gaussian representation and uses differentiable rasterization to facilitate fast optimization and rendering. Unlike traditional or implicit SLAM representations that depend on sparse geometric features or are limited by inadequate pixel sampling strategies, EndoGSLAM can use dense photometric loss for real-time tracking and reconstruction, making it robust among complex surgical fields. Besides, EndoGSLAM iteratively expands 3D Gaussians on those previously unobserved regions and partially refines the reconstructed surgical field, significantly reducing computational costs. Extensive evaluations demonstrate EndoGSLAMâs advantages in terms of optimization speed, rendering quality, and overall system efficiency, showing its huge potential for advanced surgical navigation.

<!-- image-->  
Fig. 2: Overview. EndoGSLAM aims to track the camera and reconstruct tissues among endoscopic surgeries while enabling online visualization.

## 2 Method

EndoGSLAM is an efficient dense RGB-D SLAM method for endoscopic procedures utilizing 3D Gaussians as the core representation. It begins with an innovative modification to the standard 3D Gaussian representation, initializing it to adapt to the complex environments encountered in endoscopy (Sec. 2.1). After the initialization, we leverage differentiable rasterization to enable gradientbased optimization for optimizing the camera pose in each incoming frame (Sec. 2.2). We then proceed to expand our 3D Gaussian representation into areas previously unobserved, thus complementing the scene (Sec. 2.3). Finally, we propose a partial refinement strategy for efficiently optimizing the expanded 3D Gaussians (Sec. 2.4). The overall framework is illustrated in Fig. 2.

## 2.1 Preliminaries and Initialization

To efficiently handle the highly localized illumination characteristic of endoscopic procedures, we propose a streamlined 3D Gaussian representation. 3D Gaussian Splatting [7] represents complex scenes with collections of 3D Gaussians, each defined by a set of parameters including center location $\mu ,$ rotation quaternion, scaling vector, opacity $\sigma ,$ and spherical harmonic (SH) coefficients. We first replace SH coefficients with a color attribute c based on the fact that lighting primarily moves with the camera in endoscopy, reducing the need for complex view-dependent effects modeling. Besides, we employ a uniform scaling factor for all three dimensions to accelerate optimization. In this way, a surgical field is parameterized as a set of isotropic Gaussians: $\mathcal { G } = \{ G _ { i } : \mu _ { i } , c _ { i } , r _ { i } , \sigma _ { i } \} _ { i = 1 } ^ { N } ,$ where $r _ { i }$ represents the radius of the i-th Gaussian. Our simplification significantly reduces the number of parameters to optimize, leading to a significant computational cost reduction of approximately 86% (59 to 8 parameters).

We utilize the efficient differentiable 3D Gaussian Splatting algorithm [7] to render our simplified Gaussian representation. Given a collection of 3D Gaussians ${ \mathcal { G } } ,$ along with camera pose and intrinsic parameters, our rendering process begins by sorting all Gaussians from near end to far end. Subsequently, we efficiently render an RGB image by alpha-compositing the splatted 2D projection of each Gaussian in the pixel space, determining the color of a pixel u as:

$$
\hat { C } ( u ) = \sum _ { i \in N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , \quad \alpha _ { i } = \sigma _ { i } \exp \left( - \frac { \| u - \mu _ { i } ^ { 2 D } \| ^ { 2 } } { 2 ( r _ { i } ^ { 2 D } ) ^ { 2 } } \right)\tag{1}
$$

where $\mu _ { i } ^ { 2 D }$ and $r _ { i } ^ { 2 D }$ are the 2D projection of $\mu _ { i }$ and $r _ { i } ,$ , respectively. We estimate the depth $D ( u )$ at a pixel u similar to color rendering as the sum of z coordinates of the Gaussians affecting this pixel weighted by the transmittance factor:

$$
\hat { D } ( u ) = \sum _ { i \in { \cal N } } z _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $z _ { i }$ is the z coordinate of $\mu _ { i }$ . Since $D ( u )$ is a weighted sum of $z _ { i } ,$ , we can simply accumulate the weights to represent the visibility of u:

$$
V ( u ) = \sum _ { i \in { \cal N } } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) .\tag{3}
$$

This differentiable rendering process enables us to optimize camera pose and Gaussian parameters via gradient-based optimization.

Initiating from an initial frame, we conduct pixel reprojection into 3D space to construct a point cloud using a known intrinsic matrix and an identityinitialized pose matrix. Subsequently, we convert the point cloud into a set of 3D Gaussians denoted as $\mathscr { G } _ { t = 0 }$ . Each point within this Gaussian ensemble is assigned positional coordinates represented by $\mu _ { i }$ and its color converted to $c _ { i }$ . The radius $r _ { i }$ is determined as equivalent to a one-pixel radius upon projection into the 2D image, calculated by dividing the depth by the focal length. The opacity parameter $\sigma _ { i }$ is initialized as a constant value (0.5).

## 2.2 Camera Tracking

We employ gradient descent for camera tracking using sequential RGB-D frames. The current pose $\scriptstyle { E _ { t } }$ is initialized based on the previous pose $\mathbf { \delta E } _ { t - 1 }$ and constant velocity $\varDelta ( E _ { t - 1 } , E _ { t - 2 } )$ . We then render the current image $\hat { C } _ { t }$ , depth $\hat { D } _ { t }$ , and visibility $V _ { t }$ using splatting based on $\scriptstyle { \pmb { E } } _ { t }$ , optimizing the pose $\scriptstyle { E _ { t } }$ by minimizing a re-rendering loss. Recognizing that not all pixels contribute equally to accurate tracking, we employ a pre-filter $M _ { t }$ defined on the gray-scale pixel intensities of the current image to exclude pixels with unreliable brightness, i.e., if $\delta \leq$ $G _ { t } ( u ) \leq 1 - \delta , M _ { t } ( u ) = 1 ;$ otherwise, $M _ { t } ( u ) = 0$ , where $G _ { t } ( u )$ is the gray-scale intensity at pixel u in the image $C _ { t }$ and $\delta = 0 . 1$ is a fixed intensity threshold. This approach is necessitated by the unique lighting conditions in the surgical field, where the light source moves with the camera, leading to variability in tissue brightness across frames, and causing insufficient brightness and color in areas further away from the camera. We also utilize the visibility map to identify accurately reconstructed tissues ensuring optimization focuses on these areas, thereby enhancing tracking accuracy. The loss function for camera tracking is:

$$
\mathcal { L } _ { t r } = \sum _ { n } M _ { t } \left( u \right) \cdot V _ { t } ( u ; \rho _ { t } ) \cdot \left( \left| \hat { C } _ { t } ( u ) - C _ { t } ( u ) \right| _ { 1 } + \left| \hat { D } _ { t } ( u ) - D _ { t } ( u ) \right| _ { 1 } \right) ,\tag{4}
$$

where $C _ { t } ( u )$ and $D _ { t } ( u )$ are the ground truth color and depth of pixel u in the current frame; $V _ { t } ( u ; \rho _ { t } )$ is an indicator function to indicate whether $V _ { t } ( u )$ is greater than a visibility threshold $\rho _ { t } . ~ \rho _ { t }$ is fixed to 0.99 through all the experiments.

## 2.3 Gaussian Expanding

Following the camera tracking, we update the 3D Gaussian representation to incorporate newly observed tissues. To update the Gaussians $\mathcal { G } _ { t }$ , the expansion process adheres to three key principles: 1) Areas that fail to represent the current surgical field accurately, typically for areas with visibility $V _ { t } ( u )$ lower than a visibility threshold $\rho _ { e }$ . 2) Regions identified as containing new geometric details in front of the existing tissue reconstruction surface are also added to $\mathcal { G } _ { t } . \ 3 )$ Pixels that with unreliable color are excluded from expansion. Pixels satisfying these criteria are added to $\mathcal { G } _ { t }$ using the same method employed during initialization. This involves reprojection of pixels, and conversion to 3D Gaussians with corresponding color, center location, and other parameters, as detailed in Sec. 2.1.

## 2.4 Partial Refining

After applying Gaussian expansion, we obtain the updated Gaussians $\mathcal { G } _ { t }$ . However, newly expanded Gaussians require further optimization for better novel view synthesis. We design a partial refining strategy that focuses on those newly expanded Gaussians and recently added sub-optimal Gaussians simultaneously, leading to stable and efficient reconstruction. Specifically, we designate every k-th frame as a keyframe and then cache them into a keyframe list. To improve efficiency, we assign higher sampling probabilities to keyframes that are temporally or spatially closer to the current frame. The probability is derived by:

$$
P ( f _ { l } ) = \log _ { 2 } \left( 1 + \frac { 1 } { d _ { l } + s } \right) + \log _ { 2 } \left( 1 + \frac { 1 } { t _ { l } + s } \right) ,\tag{5}
$$

Ground Truth  
EndoGSLAM  
ORB-SLAM3  
NICESLAM  
Endo-Depth  
<!-- image-->  
Fig. 3: Qualitative results on sequence cecum_t2_b and sigmoid_ $t 2 \_ a .$

where $f _ { l }$ is the l-th keyframe in the list; $d _ { l }$ and $t _ { l }$ are the L2 distance and time index of $f _ { l } ,$ scaled down by the distance and time index of the current frame from 0-th frame; s is a constant to limit the scale of the probability, and is set to 0.2 through all the experiments. We assign a certain probability $p _ { c }$ to the current frame, normalize $P ( f _ { l } )$ so that $\begin{array} { r } { \sum _ { l } P ( f _ { l } ) = 1 - p _ { c } } \end{array}$ , and utilize this normalized probability distribution to sample keyframes. In each iteration, we select a frame from the list according to its probability and refine $\mathcal { G } _ { t }$ using the following loss function:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { r e } = \sum _ { u } M \left( { \boldsymbol { u } } \right) \cdot \left( \left( 1 - \lambda _ { s s i m } \right) \Big | \hat { C } ( { \boldsymbol { u } } ) - C ( { \boldsymbol { u } } ) \Big | _ { 1 } \right. } \\ { \displaystyle \left. + \lambda _ { s s i m } \left( 1 - \mathrm { S S I M } ( \hat { C } ( { \boldsymbol { u } } ) , C ( { \boldsymbol { u } } ) ) \right) + \Big | \hat { D } ( { \boldsymbol { u } } ) - D ( { \boldsymbol { u } } ) \Big | _ { 1 } \right) , } \end{array}\tag{6}
$$

where $\hat { C } ( u ) , C ( u ) , \hat { D } ( u )$ , and $D ( u )$ are the rendered color, ground truth color, render depth, and ground truth depth of pixel u in the selected frame, respectively. SSIM means SSIM loss and $\lambda _ { s s i m } = 0 . 2$ across all experiments.

Table 1: Quantitative results on the C3VD dataset.
<table><tr><td>Methods</td><td>PSNRâ</td><td>SSIMâ</td><td> $\overline { { \mathrm { { L P I P S } \downarrow } } }$ </td><td> $\overline { { \mathrm { R M S E } ( \mathrm { m m } ) \downarrow } }$ </td><td>ATE (mm)â</td></tr><tr><td>ORB-SLAM3[4]</td><td> $\overline { { 1 7 . 8 9 \pm 2 . 3 1 } }$ </td><td> $\overline { { 0 . 6 4 \pm 0 . 1 0 } }$ </td><td> $\overline { { 0 . 3 5 \pm 0 . 0 6 } }$ </td><td> $\overline { { 7 . 7 2 \pm 2 . 6 5 } }$ </td><td> $\overline { { 0 . 3 2 \pm 0 . 0 9 } }$ </td></tr><tr><td>NICESLAM[ [31]</td><td> $2 2 . 0 7 \pm 4 . 1 2$ </td><td> $0 . 7 3 \pm 0 . 1 3$ </td><td> $0 . 3 3 \pm 0 . 0 7$ </td><td> $1 . 8 8 \pm 1 . 0 4$ </td><td> $0 . 4 8 \pm 0 . 3 3$ </td></tr><tr><td>Endo-Depth[17]</td><td> $1 8 . 1 3 \pm 2 . 4 3$ </td><td> $0 . 6 4 \pm 0 . 0 9$ </td><td> $0 . 3 3 \pm 0 . 0 6$ </td><td> $5 . 1 0 \pm 2 . 3 9$ </td><td> $1 . 2 5 \pm 0 . 9 8$ </td></tr><tr><td>EndoGSLAM-H</td><td> $\overline { { 2 2 . 1 6 \pm 2 . 6 6 } }$ </td><td> $\overline { { 0 . 7 7 \pm 0 . 0 8 } }$ </td><td> $\overline { { 0 . 2 2 \pm 0 . 0 5 } }$ </td><td> $\overline { { 2 . 1 7 \pm 1 . 2 6 } }$ </td><td> $\overline { { 0 . 3 4 \pm 0 . 2 1 } }$ </td></tr><tr><td>EndoGSLAM-R</td><td> $1 8 . 3 7 \pm 2 . 1 7$ </td><td> $0 . 6 7 \pm 0 . 1 0$ </td><td> $0 . 3 0 \pm 0 . 0 7$ </td><td> $4 . 3 3 \pm 2 . 3 9$ </td><td> $1 . 2 3 \pm 0 . 9 0$ </td></tr><tr><td>w.o. Pre-filter</td><td> $\overline { { 1 7 . 7 9 \pm 2 . 5 7 } }$ </td><td> $\overline { { 0 . 6 3 \pm 0 . 1 4 } }$ </td><td> $\overline { { 0 . 3 2 \pm 0 . 0 8 } }$ </td><td> $\overline { { 4 . 1 1 \pm 2 . 0 7 } }$ </td><td> $\overline { { 2 . 1 4 \pm 2 . 3 3 } }$ </td></tr><tr><td>w.o. Partial Refining</td><td> $1 7 . 6 4 \pm 2 . 4 9$ </td><td> $0 . 6 3 \pm 0 . 1 2$ </td><td> $0 . 3 1 \pm 0 . 0 8$ </td><td> $4 . 3 9 \pm 2 . 0 2$ </td><td> $1 . 3 4 \pm 1 . 1 3$ </td></tr><tr><td>w.o. Simplification</td><td> $1 7 . 2 3 \pm 2 . 4 5$ </td><td> $0 . 6 5 \pm 0 . 1 4$ </td><td> $0 . 3 7 \pm 0 . 0 8$ </td><td> $4 . 2 3 \pm 2 . 4 2$ </td><td> $2 . 2 6 \pm 3 . 4 2$ </td></tr></table>

## 3 Experiments

## 3.1 Dataset and Evaluation Metrics

We evaluate our proposed method on the Colonoscopy 3D Video Dataset (C3VD) [3]. This dataset provides ground-truth RGB images, depths, and camera poses for both photometric and geometric evaluation. We choose 10 clips of highdefinition clinical colonoscopic videos. Each lasts for 21 seconds and contains 638 frames on average. We pre-undistort the images and the resolution is $6 7 5 \times 5 4 0$

For reconstruction, we use the RMSE [20] (mm) on depth for geometric evaluation. As for camera tracking, we use the absolute trajectory (ATE, mm) error to evaluate. We further demonstrate our superior rendering performance using the peak signal-to-noise ratio (PSNR), SSIM [25], and LPIPS [29].

## 3.2 Implementation Details

We implement EndoGSLAM mainly with PyTorch [14] and CUDA and provide two versions, i.e. EndoGSLAM-H (high-quality) and EndoGSLAM-R (realtime). For EndoGSLAM-R, we use $\rho _ { e } = 0 . 3$ to reproject fewer pixels during expansion, optimize camera poses for 5 iterations/frame at half resolution, and refine for 6 iterations every 2 frames. Keyframes are selected every 4 frames, and we set $p _ { c } = 0 . 9 5$ to emphasize the current frame. As for EndoGSLAM-H, we set $\rho _ { e } = 0 . 5$ , optimize camera poses for 15 iterations/frame, and refine for 25 iterations/frame. Keyframes are selected every 8 frames, and $p _ { c } = 0 . 1$ prioritizes keyframes for quality improvement. All the experiments are done on a machine with Core 13700K CPU and RTX 4090 GPU running Ubuntu 22.04.

## 3.3 Evaluation

We primarily compare EndoGSLAM to three representative methods: A wellknown traditional SLAM with robust visual tracking and sparse mapping, ORB-SLAM3 [4]; A state-of-the-art dense SLAM based on NeRF [12] that introduces a hierarchical neural implicit representation, NICESLAM [31]; An endoscopic SLAM that employs photometric constraints to achieve accurate reconstruction and tracking, Endo-Depth [17]. For a fair comparison, all these methods are provided with RGB-D frames.

Table 2: Speed on the C3VD dataset.
<table><tr><td>Methods</td><td>tracking time/frame</td><td>reconstruction time/frame</td><td>online reconstruction</td><td>online rendering speed</td></tr><tr><td>ORB-SLAM3[4]</td><td>8.5ms</td><td>32.3ms</td><td>Ã</td><td>Ã</td></tr><tr><td>NICESLAM[31]</td><td>140.29ms</td><td>2558.0ms</td><td>â</td><td>0.27 fps</td></tr><tr><td>Endo-Depth[17]</td><td>194.52ms</td><td>93.7ms</td><td>Ã</td><td>Ã</td></tr><tr><td>EndoGSLAM-H</td><td>151.4ms</td><td>268.0ms</td><td>â</td><td>100+ fps</td></tr><tr><td>EndoGSLAM-R</td><td>62.4ms</td><td>65.1ms</td><td>â</td><td>100+ fps</td></tr><tr><td>w.o. Simplification</td><td>90.0ms</td><td>98.0ms</td><td>â</td><td>100+ fps</td></tr></table>

In Table. 1, we compare two versions of EndoGSLAM with other methods in terms of novel view rendering, reconstruction, and camera localization performance. We also show the average runtime in Tabel 2 and qualitative results in Fig. 3. Only EndoGSLAM achieves online precise tracking, high-quality reconstruction, and real-time online visualization simultaneously, demonstrating its huge potential for intraoperative navigation in endoscopic surgery. Traditional systems, i.e. ORB-SLAM3 and Endo-Depth, excel in localization but depend on post-process volumetric fusion for dense reconstruction. This fusion process is sensitive to pose shifts and depth noise, leading to massive fragments in space. NICESLAM shows competitive performance but struggles with efficiency, only achieving online rendering speed at 0.27 fps, which is unacceptable for surgeries. Besides, NICESLAM often synthesizes blurred renderings due to its implicit representation. In contrast, EndoGSLAM-H utilizes an explicit 3D Gaussian representation to process RGB-D streams at 3 fps and shows better localization, reconstruction, and rendering performance. Moreover, it supports online rendering at over 100 fps, providing robust assistance for surgical procedures. To further support time-sensitive surgical settings, we introduce a real-time variant, EndoGSLAM-R. It prioritizes immediate processing capabilities by making a deliberate trade-off, accepting a slight reduction in performance to achieve real-time process, thus addressing the critical balance between speed and quality necessary for intraoperative assistance.

## 3.4 Ablation Study

We also report our ablation on the pre-filter M, the keyframe-based refining strategy and the simplification of Gaussians. in Table. 1. Metrics are tested on EndoGSALM-R since EndoGSLAM-H is more robust to these variations due to its more training iterations on wider data. Results show that our pre-filter M effectively reduces the influence of unreliable information. Omitting this module leads to artifacts in the reconstruction and instability in the tracking process. The keyframe-based refining strategy, which uses previous keyframes to assist training, improves overall performance, particularly in real-time scenarios where efficient training is crucial. The simplification of Gaussians results in enhanced optimization speeds, as demonstrated in Table 2. Additionally, the simplification of SH coefficients contributes to color stability. In the absence of such simplifications, the color becomes contingent upon the viewing direction, leading to pronounced artifacts when observed from a novel view.

## 4 Conclusion and Future Work

In this work, we introduce EndoGSLAM, an advanced dense SLAM framework that enables accurate localization, high-quality reconstruction, and more importantly, online real-time visualization, owing to a streamlined 3D Gaussians representation, differentiable rasterization, and efficient optimization strategy. Experiments prove the superior performance of EndoGSLAM compared to traditional and neural SLAM methods, demonstrating its tremendous potential to enhance endoscopic surgical procedures. Future work aims to eliminate the reliance on depth information, consider minor deformation, and seamlessly integrate it into surgical navigation systems.

## References

1. Ali, S.: Where do we stand in ai for endoscopic image analysis? deciphering gaps and future directions. npj Digital Medicine 5(1), 184 (2022) 1

2. Azagra, P., Sostres, C., FerrÃ¡ndez, Ã., Riazuelo, L., Tomasini, C., Barbed, O.L., Morlana, J., Recasens, D., Batlle, V.M., GÃ³mez-RodrÃ­guez, J.J., et al.: Endomapper dataset of complete calibrated endoscopy procedures. Scientific Data 10(1), 671 (2023) 1

3. Bobrow, T.L., Golhar, M., Vijayan, R., Akshintala, V.S., Garcia, J.R., Durr, N.J.: Colonoscopy 3d video dataset with paired depth from 2d-3d registration. Medical Image Analysis p. 102956 (2023) 7

4. Campos, C., Elvira, R., RodrÃ­guez, J.J.G., Montiel, J.M., TardÃ³s, J.D.: Orb-slam3: An accurate open-source library for visual, visualâinertial, and multimap slam. IEEE Transactions on Robotics 37(6), 1874â1890 (2021) 7, 8

5. Grasa, O.G., Bernal, E., Casado, S., Gil, I., Montiel, J.: Visual slam for handheld monocular endoscope. IEEE transactions on medical imaging 33(1), 135â146 (2013) 2

6. Gu, Y., Gu, C., Yang, J., Sun, J., Yang, G.Z.: Visionâkinematics interaction for robotic-assisted bronchoscopy navigation. IEEE Transactions on Medical Imaging 41(12), 3600â3610 (2022) 2

7. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. TOG 42(4) (2023) 2, 3, 4

8. Li, H., Gu, X., Yuan, W., Yang, L., Dong, Z., Tan, P.: Dense rgb slam with neural implicit maps. arXiv preprint arXiv:2301.08930 (2023) 2

9. Liu, X., Li, Z., Ishii, M., Hager, G.D., Taylor, R.H., Unberath, M.: Sage: slam with appearance and geometry prior for endoscopy. In: ICRA. pp. 5587â5593. IEEE (2022) 2

10. Ma, R., Wang, R., Zhang, Y., Pizer, S., McGill, S.K., Rosenman, J., Frahm, J.M.: Rnnslam: Reconstructing the 3d colon to visualize missing regions during a colonoscopy. Medical image analysis 72, 102100 (2021) 2

11. Mahmoud, N., Hostettler, A., Collins, T., Soler, L., Doignon, C., Montiel, J.M.M.: Slam based quasi dense reconstruction for minimally invasive surgery scenes. arXiv preprint arXiv:1705.09107 (2017) 2

12. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99â106 (2021) 2, 7

13. Ozyoruk, K.B., Gokceler, G.I., Bobrow, T.L., Coskun, G., Incetan, K., Almalioglu, Y., Mahmood, F., Curto, E., Perdigoto, L., Oliveira, M., et al.: Endoslam dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic videos. Medical image analysis 71, 102058 (2021) 2

14. Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., Lerer, A.: Automatic differentiation in pytorch (2017) 7

15. Posner, E., Zholkover, A., Frank, N., Bouhnik, M.: C 3 fusion: consistent contrastive colon fusion, towards deep slam in colonoscopy. In: International Workshop on Shape in Medical Imaging. pp. 15â34. Springer (2023) 2

16. Rau, A., Bhattarai, B., Agapito, L., Stoyanov, D.: Bimodal camera pose prediction for endoscopy. IEEE Transactions on Medical Robotics and Bionics (2023) 2

17. Recasens, D., Lamarca, J., FÃ¡cil, J.M., Montiel, J., Civera, J.: Endo-depth-andmotion: Reconstruction and tracking in endoscopic videos using depth networks and photometric constraints. RAL 6(4), 7225â7232 (2021) 2, 7, 8

18. SandstrÃ¶m, E., Li, Y., Van Gool, L., Oswald, M.R.: Point-slam: Dense neural point cloud-based slam. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 18433â18444 (2023) 2

19. Shao, S., Pei, Z., Chen, W., Zhu, W., Wu, X., Sun, D., Zhang, B.: Self-supervised monocular depth and ego-motion estimation in endoscopy: Appearance flow to the rescue. Medical image analysis 77, 102338 (2022) 2

20. Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, D.: A benchmark for the evaluation of rgb-d slam systems. In: 2012 IEEE/RSJ international conference on intelligent robots and systems. pp. 573â580. IEEE (2012) 7

21. Sucar, E., Liu, S., Ortiz, J., Davison, A.J.: imap: Implicit mapping and positioning in real-time. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6229â6238 (2021) 2

22. Wang, C., Oda, M., Hayashi, Y., Kitasaka, T., Honma, H., Takabatake, H., Mori, M., Natori, H., Mori, K.: Visual slam for bronchoscope tracking and bronchus reconstruction in bronchoscopic navigation. In: Medical Imaging 2019. vol. 10951, pp. 51â57. SPIE (2019) 2

23. Wang, H., Wang, J., Agapito, L.: Co-slam: Joint coordinate and sparse parametric encodings for neural real-time slam. In: CVPR. pp. 13293â13302 (2023) 2

24. Wang, Y., Long, Y., Fan, S.H., Dou, Q.: Neural rendering for stereo 3d reconstruction of deformable tissues in robotic surgery. In: MICCAI. pp. 431â441. Springer (2022) 2

25. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600â612 (2004) 7

26. Wei, R., Li, B., Mo, H., Lu, B., Long, Y., Yang, B., Dou, Q., Liu, Y., Sun, D.: Stereo dense scene reconstruction and accurate localization for learning-based navigation of laparoscope in minimally invasive surgery. IEEE Transactions on Biomedical Engineering 70(2), 488â500 (2022) 2

27. Yang, C., Wang, K., Wang, Y., Dou, Q., Yang, X., Shen, W.: Efficient deformable tissue reconstruction via orthogonal neural plane. arXiv preprint arXiv:2312.15253 (2023) 2

28. Yang, C., Wang, K., Wang, Y., Yang, X., Shen, W.: Neural lerplane representations for fast 4d reconstruction of deformable tissues. arXiv preprint arXiv:2305.19906 (2023) 2

29. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586â595 (2018) 7

30. Zhu, Z., Peng, S., Larsson, V., Cui, Z., Oswald, M.R., Geiger, A., Pollefeys, M.: Nicer-slam: Neural implicit scene encoding for rgb slam. arXiv preprint arXiv:2302.03594 (2023) 2

31. Zhu, Z., Peng, S., Larsson, V., Xu, W., Bao, H., Cui, Z., Oswald, M.R., Pollefeys, M.: Nice-slam: Neural implicit scalable encoding for slam. In: CVPR. pp. 12786â 12796 (2022) 2, 7, 8