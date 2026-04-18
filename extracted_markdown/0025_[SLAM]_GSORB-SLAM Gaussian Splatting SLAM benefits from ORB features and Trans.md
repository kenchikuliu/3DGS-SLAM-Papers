# GSORB-SLAM: Gaussian Splatting SLAM benefits from ORB features and Transmittance information

Wancai Zheng1, Xinyi Yu1,â, Jintao Rong1, Linlin Ou1, Yan Wei1, Libo Zhou1

Abstractâ The emergence of 3D Gaussian Splatting (3DGS) has recently ignited a renewed wave of research in dense visual SLAM. However, existing approaches encounter challenges, including sensitivity to artifacts and noise, suboptimal selection of training viewpoints, and the absence of global optimization. In this paper, we propose GSORB-SLAM, a dense SLAM framework that integrates 3DGS with ORB features through a tightly coupled optimization pipeline. To mitigate the effects of noise and artifacts, we propose a novel geometric representation and optimization method for tracking, which significantly enhances localization accuracy and robustness. For high-fidelity mapping, we develop an adaptive Gaussian expansion and regularization method that facilitates compact yet expressive scene modeling while suppressing redundant primitives. Furthermore, we design a hybrid graph-based viewpoint selection mechanism that effectively reduces overfitting and accelerates convergence. Extensive evaluations across various datasets demonstrate that our system achieves state-of-the-art performance in both tracking precisionâimproving RMSE by 16.2% compared to ORB-SLAM2 baselinesâand reconstruction qualityâimproving PSNR by 3.93 dB compared to 3DGS-SLAM baselines. The project: https://aczheng-cai.github. io/gsorb-slam.github.io/

## I. INTRODUCTION

Over the past two decades, Simultaneous Localization and Mapping (SLAM) has remained a prominent research topic, evolving from traditional SLAM, which emphasizes improving localization accuracy, to neural radiance field (NeRF) SLAM, which offers rich scene representations. As a critical component in fields such as autonomous driving, virtual reality (VR), and embodied artificial intelligence [15], SLAM has gained increasing significance in map representation, which is essential for downstream tasks. Visual SLAM has introduced various map representations, including point/surfel clouds [8]â[10], mesh representations [11], [12], and voxels [13], [14].

Recently, NeRF-based novel view synthesis [16] has garnered widespread attention among researchers due to its high-fidelity output, leading to the development of numerous advanced dense neural SLAM methods that have made significant progress. However, the computational expense of volumetric rendering in NeRF limits its ability to produce full-resolution images, resulting in outputs that fall short of the desired photorealism.

Notably, the 3D Gaussian splatting (3DGS) technique [4] has enabled high-quality, full-pixel novel view synthesis and real-time scene rendering within standard GPU-accelerated frameworks. Consequently, several SLAM methods based on 3DGS [19]â[21] have emerged, significantly enhancing rendering quality and achieving rendering speeds up to 100 times faster than NeRF-based SLAM. Nonetheless, pressing challenges remain, such as the Bundle Adjustment (BA) problem and sensitivity to artifacts, which can degrade tracking accuracy. Additionally, the absence of multi-view constraints and strong anisotropy often leads to Gaussian overfitting to the current viewpoint.

To address these challenges, we present GSORB-SLAM, a tightly coupled 3DGS and ORB feature SLAM system. We develop an adaptive extended Gaussian strategy for dense mapping that dynamically initializes Gaussians in underreconstructed regions by integrating accumulated transmittance analysis with geometric cues, facilitating the construction of higher-quality maps. Furthermore, to mitigate overfitting resulting from insufficient multi-view constraints, we propose a hybrid graph-based viewpoint selection mechanism that combines overlap graphs with co-visibility graphs, effectively enhancing rendering quality and convergence speed. In the tracking module, we employ a two-stage approach comprising frame-to-frame and frame-to-model modes. The former provides an initial coarse estimate for the latter, which is subsequently refined by integrating joint photometricreprojection error minimization with surface depth, a novel geometric representation for tracking introduced in this paper, thus improving tracking robustness. Ultimately, the presence of feature points enables us to perform lightweight backend BA, reducing accumulated errors and alleviating the computational burden on the device. Our main contributions are summarized as follows:

â¢ We propose an adaptive extended differentiable Gaussian strategy along with a rendering frame selection mechanism that utilizes a hybrid graph to achieve highfidelity and compact scene representations.

â¢ We propose a method that utilizes accumulated transmittance surface depth in conjunction with the joint optimization of reprojection and photometric errors, thereby further enhancing tracking performance.

â¢ We conduct experiments on various datasets, and the results indicate that our method surpasses the baseline in both tracking and mapping.

<!-- image-->  
Fig. 1. In the 3D Gaussian representation, RGB-D sequences serve as input. Gaussians are generated in the scene based on the re-rendering of color, depth, and transmittance. A viewpoint selection strategy is employed to select the rendering frames for map training. Furthermore, we tightly couple feature points with surface depth and conduct lightweight global optimization using sparse point clouds, ultimately achieving accurate and robust tracking.

## II. RELATED WORK

## A. NeRF SLAM

Neural Radiance Fields (NeRF) [16] achieves realistic image rendering through pixel-ray sampling of images. By employing a multi-layer perceptron and volumetric differentiable rendering, NeRF facilitates novel view synthesis of unobserved regions. iMAP [29] is the first method to integrate NeRF into SLAM, achieving implicit neural representation. Since then, several advanced NeRF-based SLAM methods [33]â[35] have been developed. NICE-SLAM [30] introduces a hierarchical multi-feature grid to enhance high-quality scene reconstruction, updating only the visible grid features at each step, in contrast to iMAP. Orbeeze-SLAM [31] utilizes ORB features for triangulation, enabling monocular tracking without a depth estimator, while also leveraging NeRF for implicit scene representation. Point-SLAM [32] distributes point clouds onto object surfaces, demonstrating improved reconstruction quality through neural point cloud representation. However, the ray sampling process in NeRF for individual pixels incurs significant computational costs, resulting in rendering speeds of less than 15 frames per second (FPS).

## B. 3DGS SLAM

Recently, 3D Gaussian Splatting (3DGS) has achieved significantly faster rendering speeds, reaching up to 300 FPS, by employing tile-based rasterization for high-resolution image rendering. Many researchers have sought to integrate this advanced technology into SLAM [17], [18], [22], [23]. SplatTAM [5] optimizes camera poses by minimizing photometric error through differentiable Gaussians guided by silhouettes, while adding 3DGS to unobserved areas based on the geometric median depth error. Nevertheless, this approach can lead to an excessive number of Gaussian primitives, significantly increasing memory requirements and computational burden. Gaussian-SLAM [6] explores the limitations of 3DGS in SLAM, utilizing submaps to seed and optimize Gaussians, thereby achieving camera pose tracking and map rendering. GS-SLAM [19] derives an analytical formula for backward optimization with re-rendering of RGB-D loss and employs depth filtering to eliminate unstable Gaussians during tracking. However, establishing the appropriate depth filtering threshold poses challenges, as it must accommodate noise and convergence levels. Photo-SLAM [20] uses feature points to determine camera poses and proposes a Gaussian-pyramid-based training approach for scene construction. TANBRIDGE [21] develops a method that connects ORB-SLAM3 with 3DGS, facilitating the integration of these techniques. Although both Photo-SLAM [20] and TANBRIDGE [21] employ feature points, they do not fully leverage the potential performance benefits of these features. Our analysis of 3DGS indicates that feature points can significantly enhance 3DGS-SLAM in various aspects.

## C. Traditional SLAM

Traditional visual SLAM methods have evolved into two primary categories: direct methods and feature methods. Direct methods [24], [25] estimate camera poses by minimizing photometric error and comparing pixel differences between adjacent frames, relying on a strong assumption of photometric consistency. In contrast, feature methods [26], [27] utilize image features such as corners and lines, minimizing pixel errors between corresponding features across frames using reprojection error. The ORB-SLAM series [1], [10], [28] achieves real-time accurate localization based on ORB features and performs global optimization of sparse landmarks and poses through graph optimization. While these methods excel in real-time performance and localization accuracy compared to advanced scene representation techniques (e.g., NeRF, 3DGS), their limited environmental representation capabilities constrain their applicability to more sophisticated tasks.

## III. METHOD

The overview of our propose GSORB-SLAM system is illustrated in Fig. 1. Given a set of RGB-D sequences as input, we first introduce a 3D Gaussian representation, denoted as G, and describe the acquisition of surface depth (see III-A). Next, we present an efficient method for utilizing Gaussians in incremental mapping and outline the criteria for selecting a rendering frame (RF) for training (refer to III-B). Finally, we detail how to achieve accurate and robust pose tracking by jointly employing ORB features and lightweight global optimization based on sparse landmarks (see III-C).

## A. 3D Gaussian Representation

Gaussian map representation. Our scene of SLAM is represented by a collection of anisotropic 3D Gaussian with opacity attributes:

$$
\mathbf { G } = \left\{ G _ { i } : ( X _ { i } ^ { w } , \Sigma _ { i } , o _ { i } , c _ { i } ) | i = 1 , . . . , N \right\} .\tag{1}
$$

Each Gaussian, as noted in [4], is characterized by a set of parameters: the position $X _ { i } ^ { w } \in \mathbb { R } ^ { 3 }$ in the world coordinate system, the opacity $o _ { i } \in [ 0 , 1 ]$ , the trichromatic vector $c _ { i } \in$ $\bar { \mathbb { R } ^ { 3 } }$ which represents the color of each Gaussian, and the 3D covariance matrix $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ :

$$
\Sigma _ { i } = R S S ^ { T } R ^ { T } .\tag{2}
$$

This covariance matrix is composed of a rotation matrix $R \in$ ????(3) and a scale diagonal matrix $S \in \mathbb { R } ^ { 3 \times 3 }$

Differentiable rendering. The 3DGS technique renders colors by blending Gaussian distributions along rays, progressing from the near field to the far field, and subsequently projecting them onto the pixel screen. The color rendering can be mathematically expressed as follows:

$$
\tilde { C } = \sum _ { i = 1 } ^ { n } c _ { i } \alpha _ { i } T _ { i } , T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{3}
$$

where the density $\alpha _ { i }$ represents the opacity contribution of each Gaussian to the pixel, which is determined by both the Gaussian function and its opacity. The term $T _ { i }$ denotes the accumulated transmittance, which decreases as the ray traverses through multiple Gaussians. In order to capture geometric information, the depth ?? in the camera coordinate system is utilized instead of the color ?? for rendering:

$$
\tilde { D } = \sum _ { i = 1 } ^ { n } z _ { i } \alpha _ { i } T _ { i } .\tag{4}
$$

Surface depth. Currently, the geometric feature tracking in 3DGS-SLAM [5], [17], [19] is achieved through rerendering depth, which has demonstrated promising results. However, the depth obtained through re-rendering is influenced by each Gaussian that the ray traverses, leading to potential instability. When a ray intersects a Gaussian with specific opacity, such as noise or artifacts (illustrated by index 1 in Fig. 2), this contribution is also incorporated into the final rendering result. Inspired by the work of [3], [41], [42], we obtain the Gaussian that closely approximates the geometric surface by accumulated transmittance, which represents the surface depth:

<!-- image-->  
Fig. 2. Surface Depth Analysis: The coordinate system illustrates the extent to which transmittance diminishes after a ray traverses the Gaussian, alongside a visualization of the Gaussian corresponding position. The parameter ?? represents a predefined threshold; the Gaussian at which transmittance $T ^ { ' }$ decreases to ?? signifies the depth value nearest to the surface.

$$
\tilde { D } _ { s } = \mathrm { m a x } \{ z _ { i } | T _ { i } > \theta \} .\tag{5}
$$

Since the surface depth is determined based on the median of the accumulated transmittance, if the transmittance remains below full transmittance 2?? (?? set to 0.5) even in a converged state, the surface depth will remain uncertain. Therefore, it is essential to estimate the accumulated transmittance of the pixel to ensure that the derived surface depth is stable.

$$
\tilde { T } = \sum _ { i = 1 } ^ { n } 2 \theta \alpha _ { i } T _ { i } .\tag{6}
$$

## B. Mapping

Initialization. In SLAM, large-scale Gaussian primitives can occupy extensive spatial regions, potentially impeding the initialization of new Gaussians in areas that are underreconstructed. To mitigate this issue, it is recommended to initialize the Gaussian scale to approximately the size of a single pixel [5], [6]. For the ??-th Gaussian initialization, the position $X _ { i } ^ { w }$ is determined by the 3D points $X _ { i } ^ { c }$ in the camera coordinate system and the extrinsic matrix $W _ { w c } ,$ which denotes the transformation from camera to world coordinates. The color $c _ { i }$ is extracted from the RGB image, the opacity is initialized to 1, and the rotation matrix $R _ { i }$ is initialized to the unit quaternion $q \in S E ( 3 )$ . The scale is initialized as the three-dimensional vector $s _ { i } \in \mathbb { R } ^ { 3 }$ as follows:

$$
s _ { i } = \frac { \left( K W _ { w c } X _ { i } ^ { c } \right) _ { z } } { d _ { f } } , d _ { f } = \left| \frac { f _ { x } + f _ { y } } { 2 } \right| ,\tag{7}
$$

where ?? is the camera intrinsic matrix, and $( \cdot ) _ { z }$ denotes the third dimension, corresponding to depth. Here, ?? represents the camera focal length.

Adaptive densification. To enhance the convergence speed of differentiable Gaussians and achieve high-fidelity results, it is essential to spawn new Gaussians that model the geometry and appearance of newly observed areas. We propose an appearance mask $M _ { c } .$ , generated from the grayscale values of the re-rendered image, and a reconstructed geometric mask $M _ { d } = ( \left| D - \tilde { D } \right| < 0 . 0 5 ) ( \tilde { D } > 0 )$

The geometric accuracy of rendering ??Ë is contingent upon the convergence of differentiable Gaussians. Consequently, we establish an adaptive depth error threshold $\sigma _ { d }$ to mitigate redundant additions:

$$
\sigma _ { d } = \mu \{ E _ { d } \} + 4 0 \eta \{ E _ { d } \}\tag{8}
$$

$$
E _ { d } = \left| D ( x , y ) - \tilde { D } ( x , y ) \right| _ { ( x , y ) \in M _ { d } }\tag{9}
$$

Here, $\mu \{ \cdot \}$ denotes the mean operation, and $\eta \{ \cdot \}$ signifies the median operation. New Gaussians will be spawned in regions that satisfy at least one of the following criteria:

$$
A d d ( G _ { i } ) , \mathrm { i f } \ ( a ) \ \mathrm { o r } \ ( b ) .\tag{10}
$$

Criterion (??) is based on accumulated transmittance: $\tilde { \mathrm { T } } ( \mathrm { x } , \mathrm { y } ) < 0 . 8$ . This criterion facilitates improvements in areas where gradient descent is slow due to insufficient supervision by re-initializing them. Additionally, a new Gaussian must be generated if the accumulated transmittance has not reached 80% of the maximum transmittance to ensure tracking accuracy.

Criterion (??) is based on geometry: $( M _ { c } ~ < ~ 5 0 ) ( E _ { d } ~ < ~$ $\sigma _ { d } ) ( \tilde { T } ( x , y ) < 0 . 9 9 )$ . The condition $( M _ { c } < 5 0 ) ( E _ { d } < \sigma _ { d } )$ is employed to identify under-reconstructed areas by analyzing errors in geometry and appearance, while $( \tilde { T } ( x , y ) < 0 . 9 9 )$ prevents the continuous addition of Gaussians due to significant geometric errors at the edges. As illustrated in Fig. 1, the accumulated transmittance demonstrates greater robustness against edge errors.

Keyframe generation. Feature SLAM [1] primarily focuses on the quantity of feature points while neglecting their spatial distribution, as illustrated in Fig. 3. The strategy may result in a lack of diverse training perspectives. To address this limitation, we propose an overlap graph method for generating keyframes that takes into account the degree of overlap, defined as follows:

$$
O G ( i , j ) = \frac { \pi ( W _ { i j } , \mathcal { K } _ { j } ) } { \mathcal { K } _ { i } } ,\tag{11}
$$

where ??(Â·) is projection function, $W _ { i j }$ is the transformation matrix from frame ??-th to frame $j \mathrm { - t h }$ , and $\mathcal { K } _ { i }$ indicates a set of 3D points in the ?? frame.

Rendering frame selection mechanism. The rendering frames (RF) are selected from keyframes, and an effective selection mechanism can significantly enhance rendering quality. The TANBRIDGE [21] achieves remarkable reconstruction results by selecting a keyframe based on the covisibility graph [1]. However, we have observed that relying exclusively on the set of rendering frames derived from the co-visibility graph may impose limitations on the training perspectives, potentially covering only a small portion of the primary object, as illustrated in Fig. 3. This limitation may degrade the quality of reconstruction. To address this issue, we propose a novel rendering frame selection mechanism based on a hybrid graph.

<!-- image-->  
Fig. 3. The green points represent feature points. At this moment, a keyframe should be generated, but in this case, since the tracking requirements are met, a keyframe is not generated and inserted into the map. As a result, observations of this region will be lost during subsequent training.

Algorithm 1 Rendering frame selection mechanism   
Input: the keyframe set ??, the co-visibility graph of the current   
frame ????, the overlap graph ????, the co-visibility threshold $\beta _ { 1 }$   
the overlap threshold $\bar { \beta } _ { 2 } ,$ the current frame $f _ { 1 } ,$ , the maximum   
capacity $n _ { s } ,$ the random number of frames $n _ { r }$   
Output: rendering frame set R   
1: Add adjacency $n _ { a }$ frames and optimized $n _ { b }$ frames to ${ \mathcal { R } } .$   
2: Initialize a window ??.   
3: $f _ { c } = W$ .front();   
4: for each $f _ { i } \in K$ do   
5: if $f _ { i } \in C G \& O G ( f _ { c } , f _ { i } ) < \beta _ { 1 }$ then   
6: R.push( ????);??.push( $f _ { i } ) ; f _ { c } = W$ .front();   
7: else if $f _ { i } \notin C G \ \bar { \& } \ O G ( f _ { 1 } , f _ { i } ) > \beta _ { 2 }$ then   
8: R.push( ???? );?? .push( ???? );   
9: end if   
10: if $\mathcal { R } . \mathrm { s i z e } ( ) > n _ { s }$ then   
11: break;   
12: end if   
13: end for   
14: Randomly select $n _ { r }$ frames into $\mathcal { R } ;$   
15: return R

Initially, we select the adjacency $n _ { a }$ frames and the optimized $n _ { b }$ frames to augment the RF set R. The adjacency frames enhance the probability of training that includes the current viewpoint, as this viewpoint necessitates additional optimization for tracking. Furthermore, the optimized frames are added due to the presence of the backend pose optimization thread.

Secondly, to prevent the selection of nearly identical viewpoints into R, particularly in scenarios of weak tracking, keyframes are continuously generated to improve tracking performance. Therefore, we initialize a window ?? with the current frame $f _ { 1 } .$ Next, we employ a straightforward yet effective approach: If the candidate frame is derived from the co-visibility graph and meets condition $O G ( f _ { c } , f _ { i } ) < \beta _ { 1 }$ , it will be added to the set R. This condition not only ensures the acquisition of additional viewpoints from common regions but also minimizes the inclusion of redundant viewpoints. Conversely, if the candidate frame is not belong to the co-visibility graph but satisfies condition $O G ( f _ { 1 } , f _ { i } ) > \beta _ { 2 }$ , it will still be incorporated, as certain viewpoints may lack a co-visibility relationship based on feature points, yet still possess overlapping information with the current viewpoint. This procedure ensures diverse training viewpoints, thereby preventing overfitting. The pseudocode is provided in Alg. 1.

Finally, $n _ { r }$ frames are randomly selected and added to R to mitigate catastrophic forgetting of the global map due to

continuous learning.

Map optimization. In frame-to-model tracking mode, map optimization is crucial. In continuous SLAM, we propose an isotropic regularization loss to overcome anisotropic influence (strip-shaped Gaussian primitives):

$$
\mathcal { L } _ { i s o } = \frac { 1 } { N } \sum _ { i = 0 } ^ { N } ( m a x \{ s _ { i } \} - m i n \{ s _ { i } \} | s _ { i } > \gamma ) ,\tag{12}
$$

and a scalar regularization loss:

$$
\mathcal { L } _ { s } = \sum _ { i = 0 } ^ { N } ( s _ { i } - \gamma | s _ { i } > \gamma ) , \gamma = 0 . 0 3 * \operatorname* { m a x } \{ ( \mathbf { G } ) _ { z } \} .\tag{13}
$$

To enhance the Gaussian fit to the surface, we introduce a surface depth loss:

$$
\mathcal { L } _ { s u r } = \left| \tilde { D } _ { s } - D \right| _ { 1 } .\tag{14}
$$

Additionally, since the surface depth requires geometric depth for maintenance, we incorporate geometric supervision:

$$
\mathcal { L } _ { d } = \left| \tilde { D } - D \right| _ { 1 } .\tag{15}
$$

For color supervision, we combine L1 and SSIM [7] losses:

$$
\mathcal { L } _ { r g b } = \lambda \big | \tilde { C } - C \big | _ { 1 } + ( 1 - \lambda ) ( 1 - S S I M ( \tilde { C } , C ) ) .\tag{16}
$$

Final map optimization loss:

$$
\begin{array} { r } { \mathcal { L } _ { m a p p i n g } = \omega _ { 1 } ^ { m } \mathcal { L } _ { r g b } + \omega _ { 2 } ^ { m } \mathcal { L } _ { d } + \omega _ { 3 } ^ { m } \mathcal { L } _ { s u r } + \omega _ { 4 } ^ { m } ( \mathcal { L } _ { i s o } + 2 \mathcal { L } _ { s } ) , } \end{array}\tag{17}
$$

where $\omega ^ { m }$ is a set of weights for mapping.

## C. Tracking

Frame-to-model tracking. We jointly optimize the photometric error derived from 3DGS re-rendering and ground truth, as well as the reprojection error based on feature points, to achieve accurate pose estimation. The reprojection loss is defined as follows:

$$
\mathcal { L } _ { r p j } = \sum _ { j \in \mathcal { M } } \sum _ { i \in \mathcal { P } } \varphi ( \| p _ { i } - \pi ( W _ { \mathrm { c j } } , P _ { i } ^ { j } ) \| _ { 2 } ) ,\tag{18}
$$

where M denotes the set of local keyframes, P represents the matched features, $p _ { i }$ is the pixel observation in the current image, $W _ { \mathrm { c j } }$ indicates the pose of the ?? -th camera relative to the current camera and $P _ { i } ^ { j }$ is the ??-th 3D point observed by the ??-th camera, and ?? represents the information matrix.

Building on the analysis presented in III-A, we propose using surface depth instead of re-rendering depth to enhance tracking performance. Furthermore, we only utilize Gaussian surfaces with approximately complete transmittance as geometric features. Additionally, we optimize the combined loss using the Adam optimizer:

$$
\mathcal { L } _ { t r a c k i n g } = ( \tilde { T } > 0 . 9 9 ) ( \omega _ { 1 } ^ { t } \mathcal { L } _ { r g b } + \omega _ { 2 } ^ { t } \mathcal { L } _ { s u r } ) + \omega _ { 3 } ^ { t } \mathcal { L } _ { r p j } .\tag{19}
$$

During the optimization process, to prevent incorrect feature point matches from impacting the reduction of total loss, we will remove outliers in advance, allowing the re-rendering loss to be prioritized as the primary component.

Bundle adjustment. In the backend thread, we use graph optimization to jointly optimize map points and camera poses based on sparse feature points, similar to [1]. The backend operates as an independent thread, unaffected by the training. Therefore, once the camera poses are optimized, they can promptly provide more accurate camera poses for reconstruction, enhancing the reconstruction quality.

## IV. EXPERIMENT

## A. Experiment Setup

Dataset. To thoroughly evaluate the performance of the method presented in this paper, we select three sequences from the real-world TUM dataset [40] and eight sequences from the synthetic Replica dataset [39], following the methodologies outlined in [5], [30], [32].

Baesline. We choose state-of-the-art NeRF SLAM methods, specifically NICE-SLAM [30] and Point-SLAM [32]. In addition, we include 3DGS-based SLAM methods such as SplatTAM [5], GS-SLAM [19], MonoGS [18], and Sun [37]. We also consider loosely coupled feature points with 3DGS methods, specifically Photo-SLAM [20] and TAMBRIDGE [21]. Furthermore, we incorporate the traditional SLAM method ORB-SLAM2 [1].

Metrics. To measure RGB rendering performance, we utilize PSNR (dB), SSIM, and LPIPS. For camera pose estimation tracking, we employ the average absolute trajectory error (ATE RMSE [cm]) [38]. The best results will be highlighted in red, while the second-best results will be highlighted in blue. The results represent the average of five trials.

Implementation details. We fully implement our SLAM algorithm in C++ and CUDA. Additionally, the SLAM algorithm runs on a desktop equipped with Intel i7-12700KF and an NVIDIA RTX 4060ti 16G GPU. We set parameters $\omega ^ { m } \ = \ \{ 1 . 0 , 0 . 7 , 0 . 1 , 5 \} , \ \lambda \ = \ 0 . 8 , \ \{ \beta _ { 1 } , \beta _ { 2 } \} \ = \ \{ 0 . 0 7 , 0 . 3 \}$ , $\{ n _ { a } , n _ { b } , n _ { s } , n _ { r } \} ~ = ~ \{ 5 , 5 , 1 3 , 7 \}$ . For the TUM datasets, we set weight $\omega ^ { t } ~ = ~ \{ 1 . 0 , 0 . 7 , 0 . 1 \}$ , and we set weight $\omega ^ { t } \mathbf { \Sigma } = \mathbf { \Sigma }$ {0.7, 1.0, 0.1} for the Replica datasets.

## B. Localization and Rendering Quality Evaluation

Localization. Table I presents the tracking results on the TUM RGB-D dataset. Our method outperforms other baseline methods, including the state-of-the-art ORB-SLAM2.

TABLE I  
TRACKING RESULTS ON RGB-D TUM DATASET (ATE RMSEâ [CM]).  
âââ INDICATES UNAVAILABLE DATA BECAUSE THE RELATED WORK IS NOT OPEN.
<table><tr><td>Method</td><td>Fr1/desk1</td><td>Fr2/xyz</td><td>Fr3/office</td><td>Avg.</td></tr><tr><td>ORB-SLAM2</td><td>1.60</td><td>0.40</td><td>1.00</td><td>1.00</td></tr><tr><td>Point-SLAM</td><td>4.34</td><td>1.31</td><td>3.48</td><td>3.04</td></tr><tr><td>NICE-SLAM</td><td>4.26</td><td>31.73</td><td>3.87</td><td>13.28</td></tr><tr><td>MonoGS(RGB-D)</td><td>1.52</td><td>1.58</td><td>1.65</td><td>1.58</td></tr><tr><td>GS-SLAM</td><td>3.30</td><td>1.30</td><td>6.60</td><td>3.73</td></tr><tr><td>SplatTAM</td><td>3.35</td><td>1.24</td><td>5.16</td><td>3.25</td></tr><tr><td>Sun</td><td>3.38</td><td>â</td><td>5.12</td><td>â</td></tr><tr><td>Photo-SLAM</td><td>2.60</td><td>0.35</td><td>1.00</td><td>1.31</td></tr><tr><td>TANBRIDGE</td><td>1.75</td><td>0.32</td><td>1.42</td><td>1.16</td></tr><tr><td>Ours</td><td>1.48</td><td>0.39</td><td>0.88</td><td>0.91</td></tr></table>

<!-- image-->  
Fig. 4. The render visualization results on the Replica dataset.  
TABLE II

TRACKING RESULTS ON THE REPLICA DATASET (ATE RMSEâ [CM]).
<table><tr><td>Method</td><td>R0</td><td>R1</td><td>R2</td><td>Of0</td><td>Of1</td><td>Of2</td><td>Of3</td><td>Of4</td></tr><tr><td>ORB-SLAM2</td><td>0.45</td><td>0.29</td><td>Lost</td><td>0.47</td><td>0.28</td><td>0.75</td><td>0.69</td><td>0.59</td></tr><tr><td>Point-SLAM NICE-SLAM</td><td>0.61 0.97</td><td>0.41</td><td>0.37</td><td>0.38</td><td>0.48</td><td>0.54</td><td>0.69</td><td>0.72</td></tr><tr><td>MonoGS(RGB-D)</td><td>0.47</td><td>1.31 0.43</td><td>1.07 0.31</td><td>0.88 0.70</td><td>1.00 0.57</td><td>1.06 0.31</td><td>1.10 0.31</td><td>1.13 3.20</td></tr><tr><td>GS-SLAM</td><td>0.48</td><td>0.53</td><td>0.33</td><td>0.52</td><td>0.41</td><td>0.59</td><td>0.46</td><td>0.70</td></tr><tr><td>SplatTAM</td><td>0.31</td><td>0.40</td><td>0.29</td><td>0.47</td><td>0.27</td><td>0.29</td><td>0.32</td><td>0.55</td></tr><tr><td>Ours</td><td>0.35</td><td>0.22</td><td>0.33</td><td>0.33</td><td>0.19</td><td>0.54</td><td>0.63</td><td>0.45</td></tr></table>

TANBRIDGE and Photo-SLAM are loosely coupled methods based on ORB-SLAM3 [10], while our tightly coupled method improves localization accuracy by an average of 26% compared to these methods. This improvement is attributed not only to the joint optimization of ORB feature and 3DGS but also to our novel surface geometry-based tracking method. Table II reports the results on the synthetic Replica dataset. The feature-based ORB-SLAM2 fails to track sequence R2 due to the presence of a weakly textured wall. In contrast, our approach not only achieves superior tracking across all sequences compared to ORB-SLAM2, resulting in a 16.2% improvement in RMSE, but also surpasses other 3DGS-based SLAM methods.

Rendering quality. We quantitatively evaluate rendering quality on the Replica and TUM RGB-D datasets, with results shown in Table IV and Table V, respectively. The results in Table IV indicate that our method improves PSNR by an average of 3.93 dB compared to other approaches. In Table V, our rendering quality improves by an average of 2.21 dB compared to loosely coupled methods, while also achieving a 1.16 dB improvement over 3DGS-based methods. This enhancement is attributed to the adaptive expansion and rendering frame selection mechanism we proposed. Visualization results in Fig. 4 demonstrate the

TABLE III  
TRACKING RESULTS AT DIFFERENT DEPTHS ON THE TUM DATASETS.  
(ATE RMSEâ [CM])
<table><tr><td>Method</td><td>Fr1/desk1</td><td>Fr2/xyz</td><td>Fr3/office</td><td>Tracking/Iter.(ms)</td></tr><tr><td>Re-rendering depth</td><td>2.46</td><td>0.41</td><td>0.94</td><td>16</td></tr><tr><td>Surface depth (Ours)</td><td>1.48</td><td>0.39</td><td>0.88</td><td>11</td></tr></table>

TABLE IV

RENDERING PERFORMANCE COMPARISON OF RGB-D SLAM METHODS
<table><tr><td colspan="7">ON KEPLICA.</td></tr><tr><td>Sequence</td><td>Metric R0 PSNRâ 32.40 34.80 35.50 38.26 39.16 33.99 33.48 33.49</td><td>R1 R2</td><td>Of0 Of1</td><td></td><td>Of2 Of3 Of4</td><td></td></tr><tr><td>Point-SLAM</td><td>SSIMâ 0.97 LPIPSâ 0.11 0.12 0.11 0.10 0.12 0.16 0.13 0.14</td><td>0.98 0.98</td><td>0.98 0.99 0.96 0.96 0.98</td><td></td><td></td><td></td></tr><tr><td>NICE-SLAM</td><td>PSNRâ 22.12 22.47 24.52 29.07 30.34 19.66 22.23 24.49 SSIMâ 0.69 0.76 0.81 0.87 0.89 0.80 0.80 0.86 LPIPSâ 0.33 0.27 0.21 0.23 0.18 0.24 0.21 0.20</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SplatTAM</td><td>PSNRâ 32.86 33.89 35.25 38.26 39.17 31.97 29.70 31.81 SSIMâ 0.98 0.97 LPIPSâ 0.07 0.10</td><td>0.98</td><td></td><td>0.98 0.98 0.97 0.95 0.95</td><td></td><td></td></tr><tr><td>GS-SLAM</td><td>PSNRâ 31.56 32.86 32.59 38.70 41.17 32.36 32.03 32.92 SSIMâ 0.968 0.973 0.971 0.986 0.993 0.978 0.970 0.968</td><td>0.08</td><td></td><td></td><td></td><td>0.09 0.09 0.10 0.12 0.15</td></tr><tr><td>Ours</td><td>LPIPS â 0.0940.075 0.0930.050 0.033 0.094 0.110 0.112 PSNRâ 35.46 37.96 38.43 41.89 42.05 36.27 35.86 37.04 SSIMâ 0.986 0.990 0.992 0.993 0.993 0.988 0.989 0.989</td><td></td><td>LPIPS â0.037 0.039 0.038 0.034 0.030 0.050 0.041 0.053</td><td></td><td></td><td></td></tr></table>

ability of our method to generate higher-quality realistic images.

## C. Surface Depth Analysis

We select the TUM dataset due to its inclusion of noise from real-world environments, effectively highlighting the advantages of our method. As shown in Table III, our surface depth method significantly outperforms the tracking method based on re-rendering depth. This is due to the ability of our method to reduce the impact of artifacts and noise on tracking, as illustrated in Fig. 2. Furthermore, our method demonstrates superior speed compared to the re-rendering approach. This efficiency is achieved through our technique, which selects Gaussian primitives based on accumulated transmittance to approximate surface depth, thereby eliminating the need for re-rendering depth and enhancing the computation of geometric error gradients.

## D. Ablation Study

To highlight the contributions of our components within GSORB-SLAM, we conduct a series of ablation experiments, with results averaged from the office sequence in Table VI.

RENDERING PERFORMANCE COMPARISON OF RGB-D SLAM METHODS ON TUM DATASETS. âââ INDICATES UNAVAILABLE DATA BECAUSE THE

RELATED WORK IS NOT OPEN.
<table><tr><td>Sequence</td><td>Metric</td><td>Fr1/desk1</td><td>Fr2/xyz</td><td>Fr3/office</td></tr><tr><td rowspan="3">Point-SLAM</td><td>PSNRâ</td><td>13.87</td><td>17.56</td><td>18.43</td></tr><tr><td>SSIMâ</td><td>0.63</td><td>0.71</td><td>0.75</td></tr><tr><td>LPIPSâ</td><td>0.54</td><td>0.59</td><td>0.45</td></tr><tr><td rowspan="3">NICE-SLAM</td><td>PSNRâ</td><td>12.00</td><td>18.20</td><td>16.34</td></tr><tr><td>SSIMâ</td><td>0.42</td><td>0.60</td><td>0.55</td></tr><tr><td>LPIPSâ</td><td>0.51</td><td>0.31</td><td>0.39</td></tr><tr><td rowspan="3">TANBRIDGE</td><td>PSNRâ</td><td>21.22</td><td>23.44</td><td>20.15</td></tr><tr><td>SSIMâ</td><td>0.88</td><td>0.90</td><td>0.82</td></tr><tr><td>LPIPSâ</td><td>0.19</td><td>0.10</td><td>0.25</td></tr><tr><td rowspan="3">Photo-SLAM</td><td>PSNRâ</td><td>20.870</td><td>22.094</td><td>22.744</td></tr><tr><td>SSIMâ</td><td>0.743</td><td>0.765</td><td>0.780</td></tr><tr><td>LPIPSâ</td><td>0.239</td><td>0.169</td><td>0.154</td></tr><tr><td rowspan="3">Sun</td><td>PSNRâ</td><td>22.60</td><td></td><td>22.30</td></tr><tr><td>SSIMâ</td><td>0.91</td><td>â</td><td>0.89</td></tr><tr><td>LPIPSâ</td><td>0.15</td><td>â</td><td>0.16</td></tr><tr><td rowspan="3">SplatTAM</td><td>PSNRâ</td><td>22.00</td><td>24.50</td><td>21.90</td></tr><tr><td>SSIMâ</td><td>0.86</td><td>0.95</td><td>0.88</td></tr><tr><td>LPIPSâ</td><td>0.19</td><td>0.10</td><td>0.20</td></tr><tr><td rowspan="3">Ours</td><td>PSNRâ</td><td>23.02</td><td>24.78</td><td>24.08</td></tr><tr><td>SSIMâ</td><td>0.887</td><td>0.935</td><td>0.914</td></tr><tr><td>LPIPSâ</td><td>0.176</td><td>0.114</td><td>0.171</td></tr></table>

TABLE VI

THE ABLATION ANALYSIS ON REPLICA OFFICE SEQUENCES. THE  
RESULT IS THE AVERAGE VALUE OF ALL OFFICE SEQUENCES.
<table><tr><td colspan="3">Variable</td><td colspan="2">Average</td></tr><tr><td>RF Mech.</td><td>KF Gen.</td><td>Reg.</td><td>ATE â</td><td>PSNR â</td></tr><tr><td></td><td></td><td></td><td>0.422</td><td>39.01</td></tr><tr><td></td><td></td><td></td><td>0.425</td><td>37.32</td></tr><tr><td>Ã&gt;&gt;</td><td></td><td></td><td>0.551</td><td>36.95</td></tr><tr><td></td><td>&gt;*&gt;</td><td>&gt;S</td><td>0.430</td><td>38.51</td></tr></table>

To make comparisons more meaningful, we refer to the covisibility selection strategy. Additionally, to demonstrate the limitations of generating keyframes solely based on sparse feature points in constructing the 3D Gaussian representation, we compare the original keyframes generated by feature points in [1] with those produced by our method.

Through comprehensive analysis, we found that keyframe generation not only improves localization accuracy but also enhances rendering quality, achieving a 2 dB increase in PSNR. In contrast, the primary contribution of the other components lies in improving rendering quality. This improvement results from the combined effect of careful training viewpoint selection and regularization, which prevents the Gaussians from elongating into strip-like shapes.

## V. CONCLUSION

In this paper, we introduced GSORB-SLAM, a tightly coupled system that integrates 3DGS with ORB features. Experimental results demonstrated that our joint optimization method, which leverages geometric surfaces and feature points, effectively reduces the system sensitivity to noise. Additionally, we developed a viewpoint selection strategy based on a hybrid graph and an adaptive Gaussian expansion method to enhance the rendering quality of dense SLAM. Our experiments showcased the impressive performance of the proposed approach. However, the method currently necessitates a considerable amount of time for training and is unable to perform real-time localization, highlighting areas for future research.

## REFERENCES

[1] Mur-Artal R, Tardos J D. Orb-slam2: An open-source slam system for Â´ monocular, stereo, and rgb-d cameras. IEEE transactions on robotics, 2017, 33(5): 1255-1262.

[2] Kummerle R, Grisetti G, Strasdat H, et al. g 2 o: A general framework Â¨ for graph optimization 2011 IEEE international conference on robotics and automation. IEEE, 2011: 3607-3613.

[3] Huang B, Yu Z, Chen A, et al. 2d gaussian splatting for geometrically accurate radiance fields ACM SIGGRAPH 2024 Conference Papers. 2024: 1-11.

[4] Kerbl B, Kopanas G, Leimkuhler T, et al. 3D Gaussian Splatting Â¨ for Real-Time Radiance Field Rendering. ACM Trans. Graph., 2023, 42(4): 139:1-139:14.

[5] Keetha N, Karhade J, Jatavallabhula K M, et al. SplaTAM: Splat Track & Map 3D Gaussians for Dense RGB-D SLAM Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 21357-21366.

[6] Yugay V, Li Y, Gevers T, et al. Gaussian-slam: Photo-realistic dense slam with gaussian splatting. arXiv preprint arXiv:2312.10070, 2023.

[7] Wang Z, Bovik A C, Sheikh H R, et al. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 2004, 13(4): 600-612.

[8] Stuckler J, Behnke S. Multi-resolution surfel maps for efficient dense Â¨ 3D modeling and tracking. Journal of Visual Communication and Image Representation, 2014, 25(1): 137-147.

[9] Whelan T, Leutenegger S, Salas-Moreno R F, et al. ElasticFusion: Dense SLAM without a pose graph Robotics: science and systems. 2015, 11: 3.

[10] Campos C, Elvira R, RodrÂ´Ä±guez J J G, et al. Orb-slam3: An accurate open-source library for visual, visualâinertial, and multimap slam. IEEE Transactions on Robotics, 2021, 37(6): 1874-1890.

[11] Schops T, Sattler T, Pollefeys M. Surfelmeshing: Online surfel- Â¨ based mesh reconstruction. IEEE transactions on pattern analysis and machine intelligence, 2019, 42(10): 2494-2507.

[12] Ruetz F, Hernandez E, Pfeiffer M, et al. Ovpc mesh: 3d free-space Â´ representation for local ground vehicle navigation 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019: 8648- 8654.

[13] Maier R, Schaller R, Cremers D. Efficient online surface correction for real-time large-scale 3D reconstruction. arXiv preprint arXiv:1709.03763, 2017.

[14] Newcombe R A, Izadi S, Hilliges O, et al. Kinectfusion: Real-time dense surface mapping and tracking 2011 10th IEEE international symposium on mixed and augmented reality. Ieee, 2011: 127-136.

[15] Duan J, Yu S, Tan H L, et al. A survey of embodied ai: From simulators to research tasks. IEEE Transactions on Emerging Topics in Computational Intelligence, 2022, 6(2): 230-244.

[16] Mildenhall B, Srinivasan P P, Tancik M, et al. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021, 65(1): 99-106.

[17] Li M, Liu S, Zhou H. Sgs-slam: Semantic gaussian splatting for neural dense slam. arXiv preprint arXiv:2402.03246, 2024.

[18] Matsuki H, Murai R, Kelly P H J, et al. Gaussian splatting slam Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 18039-18048.

[19] Yan C, Qu D, Xu D, et al. Gs-slam: Dense visual slam with 3d gaussian splatting Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 19595-19604.

[20] Huang H, Li L, Cheng H, et al. Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular Stereo and RGB-D Cameras Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 21584-21593.

[21] Jiang P, Liu H, Li X, et al. TAMBRIDGE: Bridging Frame-Centered Tracking and 3D Gaussian Splatting for Enhanced SLAM. arXiv preprint arXiv:2405.19614, 2024.

[22] Hu J, Chen X, Feng B, et al. CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field. arXiv preprint arXiv:2403.16095, 2024.

[23] Li M, Huang J, Sun L, et al. NGM-SLAM: Gaussian Splatting SLAM with Radiance Field Submap. arXiv preprint arXiv:2405.05702, 2024.

[24] Wang R, Schworer M, Cremers D. Stereo DSO: Large-scale direct sparse visual odometry with stereo cameras Proceedings of the IEEE international conference on computer vision. 2017: 3903-3911.

[25] Engel J, Schops T, Cremers D. LSD-SLAM: Large-scale direct monoc- Â¨ ular SLAM European conference on computer vision. Cham: Springer International Publishing, 2014: 834-849.

[26] Davison A J, Reid I D, Molton N D, et al. MonoSLAM: Realtime single camera SLAM. IEEE transactions on pattern analysis and machine intelligence, 2007, 29(6): 1052-1067.

[27] Gomez-Ojeda R, Moreno F A, Zuniga-Noel D, et al. PL-SLAM: Â¨ A stereo SLAM system through the combination of points and line segments. IEEE Transactions on Robotics, 2019, 35(3): 734-746.

[28] Mur-Artal R, Montiel J M M, Tardos J D. ORB-SLAM: a versatile and accurate monocular SLAM system. IEEE transactions on robotics, 2015, 31(5): 1147-1163.

[29] Sucar E, Liu S, Ortiz J, et al. imap: Implicit mapping and positioning in real-time Proceedings of the IEEE/CVF international conference on computer vision. 2021: 6229-6238.

[30] Zhu Z, Peng S, Larsson V, et al. Nice-slam: Neural implicit scalable encoding for slam Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 12786-12796.

[31] Chung C M, Tseng Y C, Hsu Y C, et al. Orbeez-slam: A real-time monocular visual slam with orb features and nerf-realized mapping 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023: 9400-9406.

[32] Sandstrom E, Li Y, Van Gool L, et al. Point-slam: Dense neural Â¨ point cloud-based slam Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 18433-18444.

[33] Johari M M, Carta C, Fleuret F. Eslam: Efficient dense slam system based on hybrid representation of signed distance fields Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 17408-17419.

[34] Wang H, Wang J, Agapito L. Co-slam: Joint coordinate and sparse parametric encodings for neural real-time slam Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 13293-13302.

[35] Sandstrom E, Ta K, Van Gool L, et al. Uncle-slam: Uncertainty learn- Â¨ ing for dense neural slam Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 4537-4548.

[36] Luiten J, Kopanas G, Leibe B, et al. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713, 2023.

[37] Sun S, Mielle M, Lilienthal A J, et al. High-Fidelity SLAM Using Gaussian Splatting with Rendering-Guided Densification and Regularized Optimization. arXiv preprint arXiv:2403.12535, 2024.

[38] Sturm J, Engelhard N, Endres F, et al. A benchmark for the evaluation of RGB-D SLAM systems 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012: 573-580.

[39] Straub J, Whelan T, Ma L, et al. The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019.

[40] Sturm J, Engelhard N, Endres F, et al. A benchmark for the evaluation of RGB-D SLAM systems 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012: 573-580.

[41] Peng, Zhexi, et al. âRtg-slam: Real-time 3d reconstruction at scale using gaussian splatting.â ACM SIGGRAPH 2024 Conference Papers. 2024.

[42] Cheng, Kai, et al. âGaussianpro: 3d gaussian splatting with progressive propagation.â Forty-first International Conference on Machine Learning. 2024.