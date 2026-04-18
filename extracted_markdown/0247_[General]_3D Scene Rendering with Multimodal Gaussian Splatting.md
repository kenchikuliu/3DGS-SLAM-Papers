# 3D Scene Rendering with Multimodal Gaussian Splatting

Chi-Shiang Gau1, Konstantinos D. Polyzos1, Athanasios Bacharis2, Saketh Madhuvarasu1, and Tara Javidi1

Department of Electrical and Computer Engineering, University of California San Diego, USA1 NVIDIA2

Abstractâ3D scene reconstruction and rendering are core tasks in computer vision, with applications spanning industrial monitoring, robotics, and autonomous driving. Recent advances in 3D Gaussian Splatting (GS) and its variants have achieved impressive rendering fidelity while maintaining high computational and memory efficiency. However, conventional visionbased GS pipelines typically rely on a sufficient number of camera views to initialize the Gaussian primitives and train their parameters, typically incurring additional processing cost during initialization while falling short in conditions where visual cues are unreliable, such as adverse weather, low illumination, or partial occlusions. To cope with these challenges, and motivated by the robustness of radio-frequency (RF) signals to weather, lighting, and occlusions, we introduce a multimodal framework that integrates RF sensing, such as automotive radar, with GS-based rendering as a more efficient and robust alternative to vision-only GS rendering. The proposed approach enables efficient depth prediction from only sparse RF-based depth measurements, yielding a high-quality 3D point cloud for initializing Gaussian functions across diverse GS architectures. Numerical tests demonstrate the merits of judiciously incorporating RF sensing into GS pipelines, achieving high-fidelity 3D scene rendering driven by RF-informed structural accuracy.

Index Termsâ3D reconstruction, 3D scene rendering, RF sensing, Gaussian splatting, depth prediction, 3D point cloud prediction, multimodal sensing

## I. INTRODUCTION

Lying at the crossroads of computer vision and robotics, 3D scene reconstruction and the ability to generate accurate 2D estimates from novel/unseen viewpoints, collectively referred to as 3D scene rendering, have become fundamental tasks due to their wide-ranging applications in autonomous driving, robotics, and surveillance, among others. Early remarkable advances in this domain were driven by neural radiance field (NeRF) methods [10], [4], which demonstrated remarkable reconstruction and rendering fidelity but at the cost of substantial computational and memory demands. More recently, 3D Gaussian Splatting (GS) [6] and its subsequent extensions [8], [3] have emerged as efficient and lightweight alternatives. By representing the 3D scene with a set of anisotropic

Gaussian functions, GS achieves high-quality rendering while significantly reducing computational and memory overhead.

Given a fixed budget of available training camera views, the original and subsequent GS architectures rely on these images to (i) initialize the Gaussian primitives, typically by predicting a 3D point cloud (PC) to ensure proper alignment with the underlying 3D scene structure; and (ii) optimize the Gaussian parameters so that the rendered outputs match the groundtruth training images. Nonetheless, GS pipelines that depend on a large number of training views often incur substantial preprocessing overhead to generate this initial 3D PC, commonly via the traditional âstructure-from-motionâ process [16] or through pre-trained depth and 2D-3D correspondence models [17], [7]. To mitigate this burden and reduce redundancy, view-planning methods in robotics and active vision aim to identify a compact set of camera viewpoints for efficient PC generation; see, e.g., [2], [1]. Motivated by the same objective in the context of GS and 3D rendering, the recently proposed ActiveInitSplat framework [13] introduced an efficient activeview-selection strategy that identifies a small yet informative set of images for GS initialization and training, while being compatible with diverse GS architectures. Although effective and significantly more computationally efficient than passive GS pipelines, ActiveInitSplat still requires a non-negligible time to collect and process the actively selected views. More critically, ActiveInitSplat along with all other vision-only GS methods, is vulnerable in challenging real-world conditions where visual sensing degrades, including adverse weather conditions, low illumination, reduced image resolution, or partial occlusions.

While visual sensing can degrade under adverse conditions, complementary sensing modalities that use radio-frequency (RF) signals, such as automotive radar, offer robust alternatives for predicting depth and generating the corresponding 3D PC for reliable 3D reconstruction and rendering. In particular, RF signals exhibit strong robustness to weather, lighting conditions, and partial occlusions, making them wellsuited for depth prediction when visual information becomes unreliable; see e.g., [11], [5]. Even in scenarios where vision sensing quality is not degraded, obtaining a high-quality 3D PC from images often incurs a non-negligible runtime, restricting the practicality of such vision-only pipelines in real-time applications.

Driven by these insights, we introduce a multimodal approach for efficient 3D scene rendering whose contributions can be summarized in the following aspects:

C1. We introduce an efficient RF-based depth prediction module that serves as a time- and computationallyefficient alternative to vision-based approaches for generating a reliable 3D PC for GS, while remaining robust under adverse conditions where visual cues become unreliable.

C2. Using only sparse RF-based depth measurements, we introduce an efficient depth-map reconstruction approach that adapts conventional Gaussian Processes (GPs) through a principled localization scheme. By modeling different spatial regions with distinct local GPs, the proposed method provides more detailed uncertainty estimates and improves both computational efficiency and prediction accuracy at unobserved locations.

C3. Numerical tests on a real-world setting demonstrate the effectiveness of the proposed approach in combining RF and vision sensing modalities for efficient GS-based rendering.

## II. PRELIMINARIES AND PROBLEM FORMULATION

GS has attracted significant attention from the research community due to its efficient representation of 3D scenes using a set of N anisotropic Gaussian functions $\{ G _ { i } \} _ { i = 1 } ^ { N }$ [6]. To initialize these Gaussian functions, the original GS formulation and subsequent variants typically rely on a given budget of $T _ { \mathrm { t r a i n } }$ training images or camera views $\{ { \bf { I } } _ { m } \} _ { m = 1 } ^ { T _ { \mathrm { { t r a i n } } } }$ , to generate a set of 3D points called 3D point cloud $\mathrm { ( P C ) }$ of the scene of interest. This PC is commonly obtained through a structure-from-motion (SfM) pipeline [16] or estimated using pre-trained depth and 2Dâ3D correspondence models such as [7], [17].

With the PC at hand, each point $\mathbf { p } _ { i } , i \in \{ 1 , \ldots , N \}$ in the PC is associated with an anisotropic Gaussian function defined as

$$
G _ { i } ( \mathbf { z } ) = \alpha _ { i } \mathrm { e x p } ( - \frac { 1 } { 2 } ( \mathbf { z } - \mathbf { m } _ { i } ) ^ { \top } \mathbf { \Sigma } \mathbf { \Sigma } _ { i } ^ { - 1 } ( \mathbf { z } - \mathbf { m } _ { i } ) )\tag{1}
$$

where z denotes any location in the 3D space at which the Gaussian is evaluated, $\mathbf { m } _ { i }$ is the mean of $G _ { i } ,$ , initially placed at the position of the PC point $\mathbf { p } _ { i }$ , and $\Sigma _ { i }$ is the $3 \times 3$ covariance matrix that determines the shape, size and orientation of $G _ { i }$ The opacity parameter $\alpha _ { i }$ controls the contribution of $G _ { i }$ to the final rendered images.

Given these Gaussian functions, the color at any pixel p of the rendered 2D image is computed as [6]

$$
\mathbf { c } ( \mathbf { p } ) = \sum _ { i = 1 } ^ { N } \mathbf { c } _ { i } G _ { i } ^ { 2 D } ( \mathbf { p } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - G _ { j } ^ { 2 D } ( \mathbf { p } ) \right)\tag{2}
$$

where $\mathbf { c } ( \mathbf { p } )$ denotes the color at pixel p, and $\mathbf { c } _ { i }$ is the color associated with the ith Gaussian, parameterized either directly in RGB or as spherical harmonics (SH) to model view-dependent appearance. The term $G _ { i } ^ { 2 D } ( \cdot )$ denotes the 2D projection of the corresponding 3D Gaussian function $G _ { i } ,$ whose mean and covariance are given by

$$
\mathbf { m } _ { i } ^ { 2 D } = [ u / b , l / b ] ^ { \top } , ~ [ u , l , b ] ^ { \top } = \mathbf { L W } [ \mathbf { m } _ { i } , 1 ] ^ { \top }\tag{3a}
$$

$$
\pmb { \Sigma } _ { i } ^ { 2 D } = \mathbf { J } \mathbf { W } \pmb { \Sigma } _ { i } \mathbf { W } ^ { \top } \mathbf { J } ^ { \top }\tag{3b}
$$

where matrices W, L denote the extrinsic and intrinsic camera parameter matrices, respectively, and J is the corresponding Jacobian; additional details are available in [6].

All Gaussian parameters are optimized so that the rendered images match the ground-truth training views $\{ \mathbf { I } _ { m } \} _ { m = 1 } ^ { M }$ . To guarantee that each covariance matrix $\Sigma _ { i }$ remains positive semi-definite during optimization, it is parameterized using the factorization $\pmb { \Sigma } _ { i } = \mathbf { \bar { R } } _ { i } \mathbf { \bar { S } } _ { i } \mathbf { S } _ { i } ^ { \top } \mathbf { R } _ { i } ^ { \top }$ where $\mathbf { S } _ { i }$ is the $3 \times 3$ diagonal scaling matrix and $\mathbf { R } _ { i }$ is the rotation matrix represented analytically via quaternions.

In the vision domain, obtaining a high-quality PC that faithfully reflects the underlying 3D structure from training images $\{ \mathbf { I } _ { m } \} _ { m = 1 } ^ { T _ { \mathrm { t r a i n } } }$ often requires (i) a sufficiently large number $T _ { \mathrm { t r a i n } }$ of views, whose processing, whether through SfM pipeline or pre-trained models, can be computationally expensive and unsuitable for time-critical applications; and (ii) images with adequate resolution and quality. These limitations naturally motivate the integration of alternative sensing modalities that are more time-efficient and remain robust under adverse conditions, such as weather or illumination ones, that degrade visual cues and reduce the quality of the captured images. In this work, we will focus on RF-based sensing as a complementary modality to vision and demonstrate the benefits of integrating it into the GS pipeline for efficient 3D rendering. Specifically, we aim to: (i) leverage a single radar transmission, providing only sparse RF-based depth measurements $\{ y _ { t } \} _ { t = 1 } ^ { T }$ corresponding to locations $\{ \mathbf { x } _ { t } \} _ { t = 1 } ^ { T } .$ , to efficiently predict depth yunobserved at any unobserved location xunobserved, addressing the RF-based depth-map reconstruction problem1; and (ii) use the resulting RF-driven depth map to construct a 3D PC as an alternative to the vision-based SfM or pre-trained models, and evaluate how this RF-informed initialization translates to GS-based rendering performance.

The next section introduces the proposed RF-based depthmap reconstruction approach, which adapts conventional GPs through a localization scheme. In this framework, different spatial regions are modeled by distinct local GPs to improve computational efficiency and ensure that only the most relevant measurements influence each region, thereby yielding high-quality depth predictions with well-controlled uncertainty, as will be delineated next.

## III. RF-DRIVEN DEPTH-MAP RECONSTRUCTION FOR GS

Given a single radar transmission that provides sparse RFbased depth measurements $\mathbf { y } _ { T } = [ y _ { 1 } , \dots , y _ { T } ] ^ { \top }$ at corresponding spatial locations $\mathbf { X } _ { T } = [ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { T } ] ^ { \top }$ , the objective is to learn a function $f ( \cdot ) : { \bf x } _ { t }  y _ { t } , \forall t$ such that the depth value yunobserved can be accurately predicted at any unobserved location $\mathbf { X } _ { \mathrm { u n o b s e r v e d } } .$ . In this work, we will capitalize on GPs as an efficient Bayesian modeling framework capable of learning an unknown function while simultaneously providing principled uncertainty estimates [15], [9].

## A. Conventional GP-based learning

GP-based learning begins by assuming that the unknown function f is a random function endowed with a Gaussian prior over its evaluations; specifically, ${ \bf f } _ { T } : =$ $\begin{array} { r l r } { [ f ( { \bf x } _ { 1 } ) , \ldots , \bar { f } ( { \bf x } _ { T } ) ] ^ { \top } } & { { } \sim } & { \mathcal { N } ( \mathbf { 0 } _ { T } , \mathbf { K } _ { T } ) } \end{array}$ where $\mathbf { K } _ { t }$ is the covariance matrix whose $( m , m ^ { \prime } )$ entry is $\begin{array} { r l } { [ \mathbf { K } _ { t } ] _ { m , m ^ { \prime } } } & { { } = } \end{array}$ cov $\begin{array} { r } { \left( f ( \mathbf { x } _ { m } ) , f ( \mathbf { x } _ { m ^ { \prime } } ) \right) : = \kappa ( \mathbf { x } _ { m } , \mathbf { x } _ { m ^ { \prime } } ) } \end{array}$ , and Îº denotes the kernel function that assesses the pairwise similarity between distinct inputs ${ \bf x } _ { m }$ and $\mathbf { x } _ { m ^ { \prime } }$ [15]. The next assumption links the observed measurements $\mathbf { y } _ { T }$ to the latent function values ${ \bf f } _ { T }$ through a factored batch conditional likelihood $\begin{array} { r } { p ( \mathbf { y } _ { T } | \mathbf { f } _ { T } ; \mathbf { X } _ { T } ) \ = \ \prod _ { t = 1 } ^ { T } p ( y _ { t } | f ( \mathbf { x } _ { t } ) ) } \end{array}$ where $p ( y _ { t } | f ( \mathbf { x } _ { t } ) ) =$ $\mathcal { N } ( f ( \mathbf { x } _ { t } ) , \sigma _ { n } ^ { 2 } )$ since yt can be expressed as $y _ { t } = f ( \mathbf { x } _ { t } ) + n _ { t }$ with $n _ { t } \sim \mathcal N ( 0 , \sigma _ { n } ^ { 2 } )$ being Gaussian noise uncorrelated across t.

With the GP prior and batch conditional likelihood at hand, the function posterior pdf of $f ( \mathbf { x } )$ at any unobserved location x can be computed through Bayesâ rule as [15]

$$
p ( f ( \mathbf { x } ) | \mathbf { X } _ { T } , \mathbf { y } _ { T } ) = \mathcal { N } ( \mu _ { T } ( \mathbf { x } ) , \sigma _ { T } ^ { 2 } ( \mathbf { x } ) )\tag{4}
$$

with mean and variance given in closed form as

$$
\mu _ { T } ( \mathbf { x } ) = \mathbf { k } _ { T } ^ { \top } ( \mathbf { x } ) ( \mathbf { K } _ { T } + \sigma _ { n } ^ { 2 } \mathbf { I } _ { T } ) ^ { - 1 } \mathbf { y } _ { t }\tag{5a}
$$

$$
\sigma _ { T } ^ { 2 } ( \mathbf { x } ) = \kappa ( \mathbf { x } , \mathbf { x } ) - \mathbf { k } _ { T } ^ { \top } ( \mathbf { x } ) ( \mathbf { K } _ { T } + \sigma _ { n } ^ { 2 } \mathbf { I } _ { T } ) ^ { - 1 } \mathbf { k } _ { T } ( \mathbf { x } )\tag{5b}
$$

where $\mathbf k _ { T } ( \mathbf x ) \ : = \ [ \kappa ( \mathbf x _ { 1 } , \mathbf x ) , \hdots , \kappa ( \mathbf x _ { T } , \mathbf x ) ] ^ { \top }$ . Note that the posterior mean in (5a) provides a point prediction of the depth value corresponding to the unobserved location x, and the posterior variance in (5b) quantifies the associated uncertainty.

Although effective in diverse practical settings, conventional GP-based learning incurs $\mathcal { O } ( T ^ { 3 } )$ complexity (c.f. (5)), making its adoption impractical as T becomes large. Moreover, in the RF-based depth prediction setting considered here, depth measurements obtained from faraway locations have negligible influence on the depth at a specific point, as only spatially proximate observations are informative. Motivated by these insights, we next introduce a more computationally efficient, localization-based GP approach that yields wellcalibrated uncertainty and more accurate depth predictions.

## B. Localized GPs for efficient depth prediction

Rather than relying on a global GP model defined over the entire depth domain ${ \mathcal { R } } ,$ we propose a principled localization strategy in which the space is partitioned into non-overlapping regions, $\mathcal { R } = \{ r _ { 1 } , \ldots , r _ { R } \}$ . For each region $r \in \{ r _ { 1 } , \ldots , r _ { R } \}$ we instantiate a separate GP that conditions only on the observations $\{ y _ { i } ^ { ( r ) } \} _ { i } ^ { T ^ { ( r ) } }$ associated with that region. Specifically, for any query location x within region r, the GP-posterior pdf $p \dot { ( } f \dot { ( } \mathbf { x } ) \dot { } \dot { } \mathbf { X } _ { T } ^ { ( r ) } , \mathbf { y } _ { T } ^ { ( r ) } ) = \mathcal { N } ( \mu _ { T } ^ { ( r ) } ( \mathbf { \bar { x } } ) , \sigma _ { T } ^ { ( r ) 2 } ( \mathbf { x } ) )$ is computed using (5) considering only the $T ^ { ( r ) }$ relevant region-specific data $\{ \mathbf { X } _ { T ^ { ( r ) } } , \mathbf { y } _ { T ^ { ( r ) } } \}$

The intuition behind this approach is threefold: (i) depth measurements originating from distant regions are largely irrelevant and have negligible influence on predictions within the local region; (ii) each region processes only $T ^ { ( r ) } \ll T$ observations, reducing the GP computational complexity to $\mathcal { O } ( T ^ { ( r ) 3 } )$ ; and (iii) by restricting GP-modeling and inference to the most pertinent measurements, the localization strategy yields better-controlled posterior variance and more accurate depth predictions within each region. It is worth noting that, although the localization strategy introduces R separate GP models, each model can be evaluated independently, enabling full parallelization and thereby preserving computational efficiency.

With the PC obtained from the proposed localized GPbased depth-map reconstruction approach, the GS Gaussian functions are initialized accordingly, and their parameters are subsequently optimized using the available training images.

Remark. Although the multimodal framework in this work focuses primarily on vision and RF-based sensing, it can naturally incorporate LiDAR measurements for PC prediction whenever available. Exploring alternative strategies for jointly leveraging all three sensing modalities, beyond PC construction, belongs to our future research agenda.

<!-- image-->  
Fig. 1: Initial depth map from a single radar transmission consisting of sparse depth measurements/observations.

## IV. NUMERICAL TESTS

## A. Implementation details

We evaluate the 3D rendering performance of the proposed approach on the View-of-Delft dataset [12], which contains urban driving scenes captured from a vehicle equipped with both camera and radar sensors. The part of the urban scene used in this study is captured by $M = 3 5$ total images with $N _ { \mathrm { t r a i n } } = 1 2$ allocated for training and the remaining $N _ { \mathrm { t r a i n } } =$ 23 used for testing. In addition to RGB images captured from camera sensors, each radar transmission produces a sparse depth map computed from returned echoes at specific azimuth and elevation angles, as determined by the radar signal processing pipeline. The dataset provides depth measurements for sparse points spanning azimuth angles in $[ - 9 0 ^ { \circ } , 9 0 ^ { \circ } ]$ and elevation angles in $[ - 2 0 ^ { \circ } , 2 0 ^ { \circ } ]$ . This setting is particularly challenging for conventional GS-based rendering due to the very limited number of available images for GS initialization and training, combined with the confined scene coverage they provide.

<!-- image-->  
(a) Conventional GP prediction

<!-- image-->  
(b) Proposed localized GP prediction

<!-- image-->  
(c) Ground truth

Fig. 2: Comparison of (a) the conventional âglobalâ GP depth predictor and (b) the proposed localized GP predictor, shown alongside (c) the ground-truth depth map obtained from five radar transmissions.  
<!-- image-->  
Fig. 3: Depth variance at different angles using the conventional GP-based depth predictor.

<!-- image-->  
Fig. 4: Depth variance at different angles using the proposed localized GP-based depth predictor.

In our experimental setup for the proposed multimodal framework, the observed RF-based measurements consist of observed depth values obtained from a single radar transmission captured in the same time slot as the first training image. For the proposed localized GP-based approach for efficient depth-map (and corresponding PC) reconstruction from RF measurements, each regional GP model employs an RBF kernel $\kappa ^ { ( r ) }$ whose lengthscale hyperparameter is optimized by maximizing the marginal log-likelihood using only the depth observations within that region. With the PC obtained either from our proposed localized GP-based method or from traditional vision-only baselines, used for GS initialization, we follow the standard GS pipeline of [6] for rendering. The GS model is trained using $N _ { \mathrm { t r a i n } } = 1 2$ images, optimizing the combined L1 and D-SSIM loss function as in [6], over 30000 training iterations.

The localized GP-based approach for depth prediction was executed on an Intel Core i7-5930K CPU, while the GS training process was conducted on a Tesla V100-SXM2- 16GB GPU hosted on Amazon servers. For GS training and rendering, we use the publicly available implementation at https://github.com/graphdeco-inria/gaussian-splatting. For the vision-only GS baseline, the PC used for GS initialization was constructed using COLMAP.

## B. Numerical results

To demonstrate the merits of the proposed multimodal GP framework, we adopt a twofold evaluation strategy: (i) we first assess the effectiveness of the localized GP approach in predicting unobserved depth values from sparse radar-based depth measurements/observations; and (ii) we evaluate how the RF-derived PC produced by the localized GP method enhances GS rendering performance, while also reducing the processing cost for GS initialization compared to its visiononly GS-based counterpart.

1) Efficient RF-based depth prediction: Starting with a sparse set of depth values obtained from a single radar

Multimodal GS (ours)

<!-- image-->  
Fig. 5: Visual comparison of the proposed multimodal GS with the conventional vision-only 3DGS on two indicative test (novel) viewpoints. It is evident that the rendered images produced by the proposed multimodal GS approach exhibit substantially improved quality compared to the conventional unimodal 3DGS baseline.

transmission, depicted in Fig. 1, we use the proposed localized GP-based framework to predict the depth values at unobserved angles to obtain a more informative PC to initialize the Gaussian functions of GS. Using as ground truth the depth values obtained from five radar transmissionsâavailable in the View-of-Delft dataset but treated as unknown during prediction and used solely for evaluationâwe compare in Fig. 2 the performance of the proposed localized GP approach against the conventional âglobalâ GP predictor [15]. The results demonstrate that the depth map produced by the localized GP approach is closer to the ground truth, highlighting the advantages of the proposed localization strategy in terms of prediction accuracy. Quantitatively, the conventional GP method yields an overall mean absolute error of 13.07 m, whereas the proposed localized strategy substantially reduces this error to 10.57 m.

In addition to the predicted mean depth values, we also illustrate the variance of the conventional GP predictor and the proposed localized GP predictor in Figs. 3 and 4, respectively. The results indicate that our proposed predictor provides a more detailed and spatially coherent representation of uncertainty, as the predicted depth variance adapts to local measurement characteristics. Lastly, we compare the localized approach with the conventional GP counterpart in terms of running time. Table I reports the runtime of the competing alternatives, where the proposed approach exhibits a substantially lower computational cost. This improvement is expected, as our method processes only the observations within each region rather than operating on the full set simultaneously, thereby reducing the overall computational complexity.

2) Gaussian splatting rendering performance: Next, we demonstrate how the proposed multimodal GS framework, leveraging the RF-assisted PC generated by our localized GP method from a single radar transmission, can assist GS performance relative to conventional vision-only GS approaches.

TABLE I: Running time comparison for depth map reconstruction
<table><tr><td>Method</td><td>Running time â</td></tr><tr><td>Conventional GP-based prediction</td><td>9.39 s</td></tr><tr><td>Localized GP-based prediction (ours)</td><td>0.81 s</td></tr></table>

For our multimodal GS approach, note that a single radar transmission provides depth measurements only at a limited set of detected angles; it offers no information about depth at undetected angles or which of those angles contain objects. Therefore, unlike the previous subsectionâwhere detected angles from five radar transmissions were used to validate the effectiveness of the proposed localized GP depth predictorâin practical scenarios such information is not available a priori. To mimic such a realistic practical setting, we instead generate random points within the azimuth and elevation ranges, predict their depth using our localized GP approach, and subsequently use the resulting PC for GS initialization. It is worth noting that, when constructing the PC, we retain only the depth estimates with lower posterior variance (those corresponding to higher confidence) ensuring a more reliable and accurate PC.

In Table II, we compare the rendering performanceâusing the widely adopted LPIPS, SSIM, and PSNR metricsâof the proposed multimodal GS approach, which leverages the RF-assisted PC for GS initialization and uses the available training images for GS training, against the conventional unimodal GS baseline, which relies solely on the training images for both initialization and training. It can be clearly seen that the multimodal GS approach achieves a markedly improved rendering performance compared to the vision-only GS baseline. Qualitatively, Fig. 5 illustrates renderings from both approaches at two representative unseen (novel) test viewpoints, showing that the multimodal method produces outputs that more closely match the corresponding groundtruth images. This underscores the advantages of properly integrating RF-based and vision-based sensing modalities, compared to conventional unimodal vision-only GS approaches.

TABLE II: LPIPS, SSIM and PSNR values for all competing methods in a certain scene of the View-of-Delft datset.
<table><tr><td>Method</td><td>LPIPS â</td><td>SSIM â</td><td>PSNR â</td></tr><tr><td>3DGS-Vision only sensing</td><td>0.5114</td><td>0.4161</td><td>13.339</td></tr><tr><td>Multimodal GS (ours)</td><td>0.4727</td><td>0.4628</td><td>15.032</td></tr></table>

Runtime comparison. In contrast to conventional GS relying solely on training images and COLMAP to generate a PC, requiring 4.43 mins in our setting, the proposed radarbased depth predictor produces a complete PC from sparse depth measurements in approximately 1 sec, demonstrating a substantial improvement in computational efficiency for GS initialization.

## V. CONCLUSIONS

In this paper, we introduced a multimodal 3D scene rendering framework that integrates RF-based and visual sensing modalities to address key limitations of unimodal, vision-only GS pipelines. Leveraging sparse radar depth measurements, we developed a localized GP framework for efficient depthmap reconstruction that produces highly informative PCs with improved depth-prediction accuracy, better-calibrated uncertainty, and substantially reduced computational complexity compared to conventional âglobalâ GP predictors. The resulting RF-driven PC is then used for GS initialization as an alternative to traditional vision-based pipelines. Numerical tests on a 3D scene from the View-of-Delft dataset demonstrated that (i) even a single radar transmission, when processed through the proposed localized GP approach, provides meaningful structural cues for 3D rendering; and (ii) RF-informed GS initialization achieves superior rendering fidelity compared to its vision-only counterpart, as evidenced by improvements in LPIPS, SSIM, and PSNR, while exhibiting reduced processing costs. These results highlight the strong potential of multimodal sensing, particularly the integration of RF and vision, for efficient, and high-quality 3D scene rendering.

## REFERENCES

[1] A. Bacharis, K. D. Polyzos, G. B. Giannakis, and N. Papanikolopoulos, âBosfm: A view planning framework for optimal 3d reconstruction of agricultural scenes,â arXiv preprint arXiv:2509.24126, 2025.

[2] A. Bacharis, K. D. Polyzos, H. J. Nelson, G. B. Giannakis, and N. Papanikolopoulos, âEfficient 3d reconstruction in noisy agricultural environments: A bayesian optimization perspective for view planning,â IEEE Robotics and Automation Letters, 2025.

[3] Y. Bao, T. Ding, J. Huo, Y. Liu, Y. Li, W. Li, Y. Gao, and J. Luo, â3d gaussian splatting: Survey, technologies, challenges, and opportunities,â IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[4] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-nerf: A multiscale representation for antialiasing neural radiance fields,â in Proceedings of the IEEE/CVF International conference on Computer Vision, 2021, pp. 5855â5864.

[5] C. Cui, Y. Ma, J. Lu, and Z. Wang, âRadar enlightens the dark: Enhancing low-visibility perception for automated vehicles with cameraradar fusion,â in 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2023, pp. 2726â2733.

[6] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Transactions on Graphics, vol. 42, no. 4, pp. 139â1, 2023.

[7] V. Leroy, Y. Cabon, and J. Revaud, âGrounding image matching in 3d with mast3r,â in Proceedings of the European Conference on Computer Vision. Springer, 2024, pp. 71â91.

[8] H. Liu, B. Liu, Q. Hu, P. Du, J. Li, Y. Bao, and F. Wang, âA review on 3d gaussian splatting for sparse view reconstruction,â Artificial Intelligence Review, vol. 58, no. 7, p. 215, 2025.

[9] Q. Lu, K. D. Polyzos, B. Li, and G. B. Giannakis, âSurrogate modeling for bayesian optimization beyond a single gaussian process,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 11 283â11 296, 2023.

[10] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[11] D.-H. Paek, S.-H. Kong, and K. T. Wijaya, âK-radar: 4d radar object detection for autonomous driving in various weather conditions,â Advances in Neural Information Processing Systems, vol. 35, pp. 3819â 3829, 2022.

[12] A. Palffy, E. Pool, S. Baratam, J. F. P. Kooij, and D. M. Gavrila, âMulticlass road user detection with 3+1d radar in the view-of-delft dataset,â IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4961â4968, 2022.

[13] K. D. Polyzos, A. Bacharis, S. Madhuvarasu, N. Papanikolopoulos, and T. Javidi, âActiveinitsplat: How active image selection helps gaussian splatting,â arXiv preprint arXiv:2503.06859, 2025.

[14] K. D. Polyzos, A. Sadeghi, W. Ye, S. Sleder, K. Houssou, J. Calder, Z.-L. Zhang, and G. B. Giannakis, âBayesian active learning for sample efficient 5g radio map reconstruction,â IEEE Transactions on Wireless Communications, 2024.

[15] C. E. Rasmussen and C. K. Williams, Gaussian processes for machine learning. MIT press Cambridge, MA, 2006.

[16] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,âÂ¨ in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[17] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, âDust3r: Geometric 3d vision made easy,â in Proceedings of the IEEE/CVF International conference on Computer Vision, 2024, pp. 20 697â20 709.