# PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations

Yu Wei1 Jiahui Zhang1 Xiaoqin Zhang2 Ling Shao3 Shijian Lu1

1Nanyang Technological University 2Zhejiang University of Technology   
3UCAS-Terminus AI Lab, University of Chinese Academy of Sciences

Rendered Images  
<!-- image-->  
Estimated Poses

(a) CF-3DGS  
<!-- image-->  
I I I I I I I I I I I I I I I I I

Rendered Images  
<!-- image-->  
Estimated Poses

<!-- image-->  
(b) PCR-GS (Ours)  
Figure 1. The proposed PCR-GS can model scenes with complex camera trajectories without using any camera pose priors. It achieves more accurate camera pose estimation and more realistic novel view synthesis as compared with the state-of-the-art method CF-3DGS [6]. The graphs in (a) and (b) show the Rendered Images and the Estimated Poses by CF-3DGS and our proposed PCR-GS, respectively.

## Abstract

COLMAP-free 3D Gaussian Splatting (3D-GS) has recently attracted increasing attention due to its remarkable performance in reconstructing high-quality 3D scenes from unposed images or videos. However, it often struggles to handle scenes with complex camera trajectories as featured by drastic rotation and translation across adjacent camera views, leading to degraded estimation of camera poses and further local minima in joint optimization of camera poses and 3D-GS. We propose PCR-GS, an innovative COLMAP-free 3DGS technique that achieves superior 3D scene modeling and camera pose estimation via camera pose co-regularization. PCR-GS achieves regularization from two perspectives. The first is feature reprojection regularization which extracts view-robust DINO features from adjacent camera views and aligns their semantic information for camera pose regularization. The second is wavelet-

based frequency regularization which exploits discrepancy in high-frequency details to further optimize the rotation matrix in camera poses. Extensive experiments over multiple real-world scenes show that the proposed PCR-GS achieves superior pose-free 3D-GS scene modeling under dramatic changes of camera trajectories.

## 1. Introduction

Photo-realistic 3D scene reconstruction and rendering has attracted increasing attention due to a wide range of applications in embodied artificial intelligence, virtual reality, etc. With the advent of 3D Gaussian Splatting (3D-GS) [13], 3D reconstruction has made great progress by leveraging learnable 3D Gaussians to explicitly model scenes. Given a sequence of RGB images and corresponding camera poses, 3D-GS and its variants [37, 46] have demonstrated superior performance in novel view synthesis. However, the impressive performance relies heavily on the availability of accurate camera poses, which are typically obtained with COLMAP [29], a Structure-from-Motion (SfM) technique that is computationally intensive and often fails for images with sparse textures or repetitive patterns.

Several studies explore pose-free 3D-GS for novel view system without using COLMAP. One representative work is COLMAP-Free 3D Gaussian Splatting (CF-3DGS) [6] which optimizes camera poses and 3D Gaussians sequentially as cameras move. Specifically, CF-3DGS estimates the relative camera pose between adjacent camera views and employs it to regularize camera pose estimation based on the alignment between the adjacent views. However, the accuracy of relative pose estimation is highly susceptible to complex camera trajectories, as dramatic camera movements can easily lead to limited overlaps and further compromised alignment across adjacent camera views. With inaccurate relative camera poses, joint optimization of camera poses and 3D Gaussians tends to converge to local optima as illustrated in Fig. 1(a).

We propose PCR-GS, a COLMAP-free 3D-GS that coregularizes camera poses from the perspective of feature reprojection and wavelet-based frequency regularization. Leveraging the robust DINO features against viewpoint changes [1], feature reprojection exploits DINO [3] to extract semantic features from each camera view and performs feature reprojection across adjacent camera views. It regularizes the estimation of relative camera poses via semantic feature alignment that minimizes the feature discrepancy between the re-projected and the target camera views. We also design an initialization strategy that initializes relative camera poses by establishing sparse feature correspondences across adjacent camera views. Such initialization facilitates the subsequent feature reprojection and mitigates the risk of local minima in camera pose estimation.

The wavelet-based frequency regularization focuses on optimizing the rotation matrix of camera poses. Specifically, rotational errors cause spatial shifts of geometries and textures, leading to loss or distortion that is well represented in high-frequency details. RGB-space regularization primarily focuses on pixel intensity changes, which is insensitive to structural shifts caused by small angular rotations. In contrast, frequency-space regularization captures rotational errors and structural shifts well with the decomposed highfrequency information. We achieve the frequency regularization with wavelet transform by decomposing images into multiple frequency bands, each having different levels of image details. As illustrated in Fig. 1(b), PCR-GS achieves superior camera pose estimation and novel view synthesis without COLMAP priors.

The contribution of this work can be summarized in three aspects. First, we propose PCR-GS, an innovative

COLMAP-free 3D Gaussian Splatting that introduces pose co-regularization and achieves superior camera pose estimation and 3D scene reconstruction under drastic camera movements across adjacent views. Second, we leverage the robustness of DINO features against viewpoint changes and propose feature reprojection regularization which regularizes relative camera poses by reprojecting extracted DINO features between adjacent views. Third, we design a wavelet-based frequency regularization to ensure the accurate estimation of the rotation matrix of camera poses. It can extract multi-level high-frequency components to leverage various levels of image detail to amplify the error of the rotation matrix, enabling the model to focus on the accurate prediction of the rotation matrix of camera poses.

## 2. Related Work

## 2.1. 3D Scene Representation and Rendering

Traditional 3D representations, such as meshes [27, 28], multi-plane images [32, 45], and point clouds [38, 42], are widely adopted to depict 3D geometry explicitly.

Recently, Neural Radiance Fields (NeRFs) [22] have demonstrated an exceptional capability for high-quality 3D representation. They implicitly employ MLPs to reconstruct 3D scenes from multi-view posed images and achieve photorealistic novel view synthesis by differential volume rendering. Due to their nature of multi-view consistency, many NeRF variants have been developed to handle various new challenges, such as dynamic scenes [19, 20, 24], fewshot modeling [12, 15, 33], model generalization [9, 16, 39] and large-scale scenes [31, 44].

However, NeRF often suffers from time-consuming training and rendering as volume rendering requires dense sampling with ray marching. Several methods [7, 10, 23] have been proposed to accelerate the training and rendering of NeRF. For instance, Muller et al. [23] introduce multi-resolution hash encoding that maps spatial coordinates to feature vectors, greatly shortening the training and rendering times. Nonetheless, these acceleration methods often sacrifice rendering quality for fast training. Kerbl et al. recently propose 3D Gaussian Splatting (3D-GS) [13], which achieves real-time, high-quality rendering with explicit 3D Gaussians and efficient splatting. Several 3D-GS variants have been proposed to address various new challenges, such as dynamic scenes [8, 11, 46], sparseview settings [37, 41, 48], and SLAM [5, 17, 47]. However, the impressive performance of 3D-GS relies heavily on pre-computed camera parameters which are often obtained with COLMAP. This has motivated the development of COLMAP-free algorithms that aim to remove this dependency and enhance flexibility in 3D scene reconstruction.

<!-- image-->  
Figure 2. The framework of the proposed PCR-GS. PCR-GS performs feature reprojection regularization and wavelet-based frequency regularization concurrently to optimize relative camera poses. For the feature reprojection regularization, we adopt DINO to extract semantic feature maps Fi and $F _ { i + 1 }$ of adjacent frames Ii and $I _ { i + 1 }$ , and optimize the relative camera pose by minimizing the discrepancies between the reprojected feature $F _ { i + 1 } ^ { \prime }$ and $F _ { i + 1 }$ . For the wavelet-based frequency regularization, we employ the relative pose to transform the camera pose from $P _ { i } ^ { * }$ to $P _ { i + 1 } ^ { * }$ to render an image under $P _ { i + 1 } ^ { * }$ , and then apply wavelet transform to the rendered image and the groundtruth image to optimize the relative camera pose via frequency-space regularization.

## 2.2. Pose-Free Neural Fields

Recently, several studies [2, 6, 18, 21, 36, 40] investigate how to train NeRF and 3D-GS without camera pose priors. For example, NeRFmm [36] introduces learnable camera parameters and jointly optimizes them with NeRF. BARF [18] introduces a progressive joint optimization approach for both camera poses and NeRF by employing a coarse-to-fine positional encoding strategy. Zhang et al. [40] propose VMRF that leverages unbalanced optimal transport to align features between rendered images and ground truth, thereby optimizing relative camera poses and NeRF scene representations. Nope-NeRF [2] constrains relative poses by using undistorted depth prior. CF-3DGS [6], the pioneer study on pose-free 3DGS, estimates relative camera poses of adjacent frames and achieves progressive pose refinement and 3DGS-based scene modeling. However, most of these methods suffer from significant performance degradation while handling scenes with drastic camera rotations and translations across adjacent views. The proposed PCR-GS introduces novel pose co-regularization, achieving superior rendering quality and pose estimation accuracy under complex camera trajectories.

## 3. Method

This section presents the proposed PCR-GS that achieves pose-free and COLMAP-Free 3D-GS under complex camera trajectories. We first provide a brief review of the original 3D-GS and the adopted baseline CF-3DGS [6] in Sec. 3.1. Sec. 3.2 then describes the proposed feature reprojection regularization that exploits the robust DINO features to estimate relative camera poses against drastic camera view changes, as well as an initialization strategy that extracts feature correspondences across adjacent views. Sec. 3.3 presents wavelet-based frequency regularization that further optimizes the rotation matrix of relative camera poses. Fig. 2 shows the overview of PCR-GS.

## 3.1. Preliminaries

3D-GS. 3D-GS [13] explicitly represents scenes with 3D Gaussians, offering advantages in differentiability and scalability. Initialized from COLMAP-generated sparse point cloud, each Gaussian is defined by a center Âµ, covariance matrix Î£, opacity Î±, and spherical harmonics (SH) coefficients c. The covariance matrix is decomposed into a scaling and rotation matrix for differentiable optimization.

Rendering projects 3D Gaussians onto a 2D image plane based on the camera pose. Pixel colors are computed via Î±-blending of N overlapping Gaussians:

$$
C = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{1}
$$

where $c _ { i }$ and $\alpha _ { i }$ are obtained by multiplying $\Sigma _ { i }$ of the i-th 2D Gaussian with SH coefficients and opacity.

COLMAP-Free 3DGS (CF-3DGS). CF-3DGS [6] introduces a set of 3D Gaussians to predict camera poses between nearby frames. Specifically, CF-3DGS initializes a set of 3D Gaussians $G _ { i }$ based on a single frame $I _ { i }$ . It extracts a point cloud from the mono-depth map generated by DPT [26] and sets the camera pose of frame $I _ { i }$ as an identity matrix. $G _ { i }$ is learned by minimizing the photometric loss between the rendered image and $I _ { i }$

$$
G _ { i } ^ { * } = \arg \operatorname* { m i n } _ { c _ { i } , \Sigma _ { i } , \alpha _ { i } } \mathcal { L } _ { \mathrm { r g b } } ( \mathcal { R } ( G _ { i } ) , I _ { i } ) ,\tag{2}
$$

where R represents the 3D-GS rendering process for $G _ { i }$ and $c _ { i } , \Sigma _ { i }$ , and $\alpha _ { i }$ are Gaussian parameters.

To estimate relative camera poses of adjacent frames, CF-3DGS transforms the pre-trained 3D Gaussians $G _ { i }$ into frame (i + 1) using a transformation matrix $T _ { i }$ , defined as $T _ { i } = P _ { i } ^ { - 1 } P _ { i + 1 }$ , where $P _ { i }$ is the camera pose of frame i. $T _ { i }$ is optimized by minimizing the photometric loss between the rendered image from transformed 3D Gaussians in frame (i + 1) and the corresponding ground truth $I _ { i + 1 }$ :

$$
T _ { i } ^ { * } = \arg \operatorname* { m i n } _ { T _ { i } } \mathcal { L } _ { \mathrm { r g b } } ( \mathcal { R } ( T _ { i } \circ G _ { i } ) , I _ { i + 1 } ) ,\tag{3}
$$

where â¦ denotes the transformation from $G _ { i }$ to $G _ { i + 1 }$ The photometric loss combines an L1 loss with D-SSIM:

$$
\mathcal { L } _ { \mathrm { r g b } } = ( 1 - \lambda ) | | I _ { i } - \hat { I } _ { i } | | + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } ,\tag{4}
$$

where $\lambda = 0 . 2$ and $\hat { I } _ { i }$ is the rendered image of frame i.

While dealing with scenes with complex camera trajectories, the relative camera pose may experience dramatic changes due to large camera movement. Under this circumstance, regularizing camera poses over RGB images becomes susceptible due to limited overlaps across adjacent frames. In comparison, DINO features are much more stable under drastically view changess [1] as illustrated in Fig. 3 We therefore align the robust DINO-based semantic features across adjacent views to optimize the camera pose with drastic rotations and translations. Specifically, with DINO features extracted from every frame, we design a feature reprojection regularization technique that optimizes relative camera pose as illustrated in Fig. 2.

## 3.2. Feature Reprojection Regularization

The pre-trained $G _ { i } ^ { * }$ can be rendered into RGB images and depth maps under the camera pose $P _ { i }$ , from which we can obtain the depth value of each pixel on $I _ { i } { \mathrm { : } }$

$$
D = \sum _ { i = 1 } ^ { N } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{5}
$$

where $D$ is the depth value of each pixel, and $d _ { i }$ denotes the distance from the $i _ { t h }$ overlapping Gaussian to the camera.

Family

<!-- image-->

<!-- image-->

<!-- image-->  
Frame i

<!-- image-->  
Figure 3. Visualization of robust DINO features against drastic camera motion. Point correspondences as searched by DINO feature maps remain stable under drastic camera pose changes.

We regularize the relative camera pose by aligning DINO features across two adjacent views. Specifically, we first reproject 2D pixels of $I _ { i }$ into 3D camera coordinate $P _ { i }$ and then transform them into the adjacent camera coordinate $P _ { i + 1 }$ with the transformation matrix $T _ { i }$ . The transformed 3D points are then projected back to 2D image $I _ { i + 1 }$ . By sampling features from two adjacent feature maps based on their original and reprojected positions, we minimize the discrepancy between features from $I _ { i }$ and the reprojected features from $I _ { i + 1 }$ . During the projection and reprojection, we use the camera intrinsic matrix K to transform points across 3D camera coordinate and 2D image coordinate.

$$
T _ { i } ^ { * } = \arg \operatorname* { m i n } _ { T _ { i } } \mathcal { L } _ { \mathrm { f e a t } } ( F _ { i } \langle P _ { i } \rangle , F _ { i + 1 } \langle \mathbf { K } P _ { i } T _ { i } \rangle ) ,\tag{6}
$$

where $F _ { i }$ is the DINO feature map extracted from $I _ { i }$ and $\langle \cdot \rangle$ represents a sampling operation on feature map. The feature loss is computed by L2 loss between feature map and reprojected feature map.

$$
\mathcal { L } _ { \mathrm { f e a t } } = | | F _ { i } \langle P _ { i } \rangle - F _ { i + 1 } \langle \mathbf { K } P _ { i } T _ { i } \rangle | | _ { 2 } .\tag{7}
$$

In addition, poor initialization of camera poses can trap the optimization in local optima, especially when the camera poses experience dramatic changes. We establish sparse correspondences between adjacent frames by searching the DINO features as represented by $C P _ { i }$

$$
C P _ { i } = \left\{ ( p , q ) \ | \ p \in I _ { i } , q \in I _ { i + 1 } \right\} ,\tag{8}
$$

where $p$ and $q$ denote the positions of correspondent key points on the respective image. We then utilize a saliency map derived from DINO feature map to construct a foreground mask, filtering out background pixels and selecting candidate key points on the feature map. Next, we compute the similarity between these key feature points across frames with the Best Buddies algorithm [4] and determine correspondences by the best-matched point pairs. Since such derived correspondences often contain noises, we randomly select $N _ { s } = 2 0$ sparse correspondences to optimize the relative camera pose and apply the optimization to initialize the camera pose as follows:

<table><tr><td>Methods</td><td colspan="2">CF-3DGS</td><td colspan="3">Nope-NeRF</td><td colspan="3">Barf</td><td colspan="3">NeRFmm</td><td colspan="3">PCR-GS (Ours)</td></tr><tr><td>Scenes</td><td>PSNRâ  SSIMâ  LPIPS</td><td></td><td></td><td>PSNRâ  SSIMâ  LPIPS</td><td></td><td>PSNRâ</td><td></td><td>SSIMâ  LPIPS</td><td></td><td>PSNRâ SSIMâ  LPIPS</td><td></td><td></td><td>PSNRâ  SSIM LPIPS</td><td></td></tr><tr><td>Church</td><td>27.71</td><td>0.88 0.13</td><td>23.32</td><td>0.65</td><td>0.47</td><td></td><td>13.74</td><td>0.37</td><td>0.70</td><td>16.94</td><td>0.42</td><td>0.61</td><td>27.90</td><td>0.89 0.12</td></tr><tr><td>Barn</td><td>23.49</td><td>0.71 0.22</td><td>23.34</td><td>0.60</td><td>0.51</td><td></td><td>15.69 0.52</td><td>0.60</td><td></td><td>17.51 0.53</td><td></td><td>0.58</td><td>28.57 0.84</td><td>0.13</td></tr><tr><td>Museum</td><td>18.43</td><td>0.54 0.37</td><td>22.42</td><td>0.55</td><td>0.51</td><td>14.75</td><td>0.39</td><td>0.65</td><td>11.43</td><td>0.22</td><td></td><td>0.74</td><td>21.56 0.66</td><td>0.25</td></tr><tr><td>Family</td><td>18.03</td><td>0.57 0.37</td><td>22.72</td><td>0.59</td><td>0.54</td><td>13.94</td><td>0.43</td><td>0.70</td><td>14.11</td><td>0.37</td><td></td><td>0.67</td><td>25.45 0.82</td><td>0.16</td></tr><tr><td>Horse</td><td>18.34</td><td>0.64 0.32</td><td>21.74</td><td>0.66</td><td>0.44</td><td>13.43</td><td>0.54</td><td>0.63</td><td>13.31</td><td>0.46</td><td></td><td>0.62</td><td>24.20 0.79</td><td>0.17</td></tr><tr><td>Ballroom</td><td>17.05</td><td>0.48 0.39</td><td>18.84</td><td>0.46</td><td>0.58</td><td>12.55</td><td>0.31</td><td>0.76</td><td>12.43</td><td>0.24</td><td></td><td>0.67</td><td>19.70 0.60</td><td>0.30</td></tr><tr><td>Francis</td><td>16.23</td><td>0.50 0.49</td><td>21.72</td><td>0.59</td><td>0.59</td><td>15.38</td><td>0.52</td><td>0.68</td><td>13.19</td><td>0.38</td><td></td><td>0.69</td><td>21.91 0.65</td><td>0.34</td></tr><tr><td>Ignatius</td><td>19.07</td><td>0.49 0.35</td><td>21.49</td><td>0.47</td><td>0.58</td><td></td><td>12.93 0.31</td><td>0.79</td><td>13.95</td><td>0.33</td><td></td><td>0.66</td><td>20.15 0.60</td><td>0.33</td></tr><tr><td>Mean</td><td>19.79</td><td>0.60 0.33</td><td>21.95</td><td>0.57</td><td></td><td>0.52</td><td>14.05</td><td>0.42</td><td>0.69</td><td>14.10</td><td>0.36</td><td>0.66</td><td>23.68 0.73</td><td>0.23</td></tr></table>

Table 1. Quantitative comparisons of novel view synthesis on Tanks&Temples [14]. Each baseline method is trained with its public code under the original settings and evaluated with the same evaluation protocol. The best score of all the results is in bold. Note the Tanks&Temples here has more drastic camera motions as detailed in the Dataset part and appendix.

$$
T _ { i } ^ { * } = \arg \operatorname* { m i n } _ { T _ { i } } \sum _ { s = 1 } ^ { N _ { s } } | | \mathbf { K } p _ { s } T _ { i } - \mathbf { K } q _ { s } | | .\tag{9}
$$

Finally, we use the optimized relative camera pose instead of the Identity matrix for initialization which improves the camera pose estimation clearly.

## 3.3. Wavelet-based Frequency Regularization

The camera pose is defined by a rotation matrix $R \in { \mathfrak { s o } } ( 3 )$ and a translation vector $t ~ \in ~ \mathbb { R } ^ { 3 }$ Our empirical studies show that the rotation matrix is much more challenging to optimize, where a small error in $R \in { \mathfrak { s o } } ( 3 )$ often leads to shifts of geometric structures such as edges and textures. Regularization in the RGB space primarily focuses on pixel intensity which is not sensitive to the structural changes due to camera rotations. We therefore propose a frequency regularization approach to further optimize the rotation matrix of camera poses. Specifically, we adopt wavelet transformation that decomposes an RGB image into multiple frequency components along horizontal and vertical directions. Unlike other frequency transformations, wavelet transform provides rich high-frequency details that retain their original spatial location within the image and also highlight subtle rotation errors for optimization.

Specifically, the wavelet transform decomposes an image $I ( u , v )$ into four components: LL, LH, HL, and $H H ,$ representing low-frequency component, horizontal highfrequency component, vertical high-frequency component and diagonal high-frequency component, respectively.

$$
\begin{array} { c } { { L L ( u , v ) = \displaystyle \sum _ { m } \sum _ { n } I ( m , n ) \cdot h ( u - m ) \cdot h ( v - n ) , } } \\ { { { } } } \\ { { L H ( u , v ) = \displaystyle \sum _ { m } \sum _ { n } I ( m , n ) \cdot h ( u - m ) \cdot g ( v - n ) , } } \\ { { { } } } \\ { { { \displaystyle H L ( u , v ) = \sum _ { m } \sum _ { n } I ( m , n ) \cdot g ( u - m ) \cdot h ( v - n ) , } } } \\ { { { } } } \\ { { { \displaystyle H H ( u , v ) = \sum _ { m } \sum _ { n } I ( m , n ) \cdot g ( u - m ) \cdot g ( v - n ) , } } } \end{array}\tag{10}
$$

where $I ( m , n )$ is the pixel value at position $( m , n )$ in the input image I, and $h ( \cdot )$ and $g ( \cdot )$ respectively represent lowpass filter, and high-pass filter.

The discrepancies of each component (denoted as d) between rendered image ËI and ground truth I can be obtained with the Euclidean metric as follows:

$$
d = \sum _ { x \in \{ L L , L H , H L , H H \} } w _ { x } \left\| W _ { x } ( I _ { t } ) - W _ { x } \left( \hat { I } _ { t } \right) \right\| ,\tag{11}
$$

where $W _ { x }$ represents the wavelet transformation that extract component x from an image and the $w _ { x }$ denotes the weight for each wavelet component.

Direct optimization of high-frequency components often introduces noise and complicates the network training. We address this issue by introducing an annealing strategy that gradually minimizes the discrepancy from low frequency to high frequency and accordingly achieves progressive pose regularization. Specifically, the annealing strategy regularizes low-frequency information first and then progressively increases the weight of high-frequency information in the loss computation. Such progressive frequency regularization $\mathcal { L } _ { f r e q }$ can be formulated as follows:

$$
\mathcal { L } _ { f r e q } = \left\{ \begin{array} { l l } { d _ { L L } } & { 0 < n \le n _ { 0 } , } \\ { ( 1 - w _ { h } ) d _ { L L } + w _ { h } d _ { H } } & { n _ { 0 } < n \le n _ { 1 } , } \\ { d _ { H } } & { n > n _ { 1 } , } \end{array} \right.\tag{12}
$$

where $d _ { H }$ denotes the discrepancies of all high-frequency components LH, HL, HH as mentioned in Eq. 11. wh denotes the weight for the high-frequency components, computed as follows:

$$
w _ { h } ( n ) = \frac { n - n _ { 0 } } { n _ { 1 } - n _ { 0 } } ,\tag{13}
$$

where $n _ { 0 } ~ = ~ 1 0 0$ and $n _ { 1 } ~ = ~ 2 0 0$ represent the iterations for introducing high-frequency components and deactivating low-frequency components, respectively.

## 3.4. Overall Training Pipeline

With all defined loss terms, the overall training objective can be formulated as follows:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { 0 } \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { f e a t } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { f r e q } } , } \end{array}\tag{14}
$$

where $\lambda _ { 0 } , \lambda _ { 1 }$ and $\lambda _ { 2 }$ are the weights for each loss term, setting to 0.6, 0.2 and 0.2, respectively.

## 4. Experiment

## 4.1. Datasets and Evaluation Metrics

Datasets. We conduct experiments over 11 real-world scenes with complex camera trajectories from the widely adopted benchmarks Tanks&Temples [14] (8 scenes), Free-Dataset [34] (3 scenes).

Tanks&Temples. The Tanks&Temples dataset used in CF-3DGS samples video frames at 20 fps on average from eight videos of different scenes. However, the resulting camera trajectories are overly smooth and lack camera motion complexity which is typically encountered in realworld scenarios. In contrast, our method targets pose-free 3DGS under challenging conditions with drastic camera motions and complex camera trajectories where pose estimation becomes significantly more difficult. CF-3DGS struggles in such settings due to its reliance on smooth camera trajectories, limiting its effectiveness in realistic environments. Thus, to evaluate our proposed method, we reconstruct this dataset by sampling video frames at 4 fps on average from the eight videos which produces much more complex camera trajectories and drastic camera motions. We retain the original training and testing data split strategy. Please refer to the appendix for details on video frame selection.

Free-Dataset. The Free-Dataset comprises large-scale scenes with significant camera motion and complex camera trajectories but no central objects. We use 50 frames per scene for training and testing [25], with ground-truth poses from COLMAP. We select one frame as test data every 8 frames in each scene.

Evaluation Metrics. For quantitative experiments, we adopt the standard evaluation metrics including PSNR, SSIM [35], and LPIPS [43] for evaluation of novel view synthesis. For pose estimation, we report relative pose rotation error $( R P E _ { r } )$ , relative pose translation error $( R P E _ { t } )$ and Absolute Trajectory Error(ATE) [30].

## 4.2. Implementation Details

Implementation. For every video frame, we extract the DINO features from the 9th layer of the DINO model [1] where the features can retain good details of the image structure. Following CF-3DGS [6], we employ the monodepth map extracted by the dense prediction transformer (DPT) [26] to initialize the point cloud for each frame. Besides, the training process follows the incremental optimization strategy and configuration setting as in CF-3DGS. Before the pose co-regularization, we initialize the relative camera pose by establishing correspondences between adjacent views. The optimization iteration for the initialization stage is set as 200 and the learning rate gradually decays from $1 0 ^ { - 4 } ~ \mathrm { t o } ~ 1 0 ^ { - 5 }$ . We utilize a quaternion rotation $\mathbf { q } \in { \mathfrak { s o } } ( 3 )$ , and translation vector $\mathbf { t } \in \mathbb { R } ^ { 3 }$ to represent camera poses, enabling smooth optimization. All experiments are conducted on a single NVIDIA A100 40G GPU. More details are provided in the supplementary material.

## 4.3. Comparisons with COLMAP-free methods

We benchmark our method with several state-of-the-art COLMAP-free models including NeRFmm [36], BARF [18], Nope-NeRF [2], and CF-3DGS [6]. The benchmarking focuses on two aspects: novel view synthesis and pose estimation. All the models are trained and tested over the same data and hardware for fairness.

Novel View Synthesis. Since the camera pose of each test frame is unknown, we freeze the parameters of the 3DGS model trained on the training set and predict the camera pose of test frames as in CF-3DGS. We then render images on test sets and compare the quality of rendered images across different models. As Tables 1 and 3 show, our method achieves superior novel view synthesis consistently over all three metrics and 11 scenes from the two benchmarks. We also conducted qualitative experiments and obtain well-aligned experimental results. As illustrated in Fig. 4 and Fig. 5, the images synthesized by PCR-GS are noticeably clearer and produce fewer artifacts as compared with those generated by the state-of-the-art models. Note for Tanks&Temples, CF-3DGS obtains much lower performance than what was presented in the original paper. The lower performance is largely attributed to the more sparsely sampled training data as detailed in the Dataset part, where we sample video frames at 4 fps (instead of 20 fps as in CF-3DGS) to have data with more dynamic camera trajectory.

<table><tr><td>Methods</td><td>CF-3DGS</td><td></td><td>Nope-NeRF</td><td></td><td>Barf</td><td></td><td>NeRFmm</td><td></td><td></td><td>PCR-GS (Ours)</td></tr><tr><td>Scenes</td><td>RPEt+ RPEr ATE~</td><td></td><td>RPEt+ RPEr ATE</td><td></td><td>RPEt RPEr+ ATE</td><td></td><td>RPEt+ RPEr ATE</td><td></td><td>RPEt+ RPEr+ ATEâ</td><td></td></tr><tr><td>Church</td><td>0.052 0.079</td><td>0.008 1.426</td><td>0.246 0.303</td><td>3.626</td><td>1.079</td><td>0.579</td><td>3.599</td><td>0.647 0.586</td><td>0.049</td><td>0.077 0.007</td></tr><tr><td>Barn</td><td>0.307 0.144</td><td>0.016 2.557</td><td>0.723</td><td>0.259 2.968</td><td>0.312</td><td>0.097</td><td>4.785</td><td>0.586 0.141</td><td>0.114</td><td>0.103 0.010</td></tr><tr><td>Museum</td><td>0.116 0.606</td><td>0.021 3.134</td><td>0.897 0.444</td><td>9.240</td><td>2.192</td><td>0.459</td><td>9.732 2.108</td><td>0.530</td><td>0.074</td><td>0.397 0.014</td></tr><tr><td>Family</td><td>0.366 0.618</td><td>0.018 4.086</td><td>0.274 0.404</td><td>8.982</td><td>2.288</td><td>0.229</td><td>8.496 2.112</td><td>0.329</td><td>0.093</td><td>0.210 0.005</td></tr><tr><td>Horse</td><td>0.080 0.186</td><td>0.005 1.982</td><td>0.474 0.181</td><td>6.540</td><td>1.570</td><td>0.564</td><td>6.201 1.162</td><td>0.535</td><td>0.061</td><td>0.126 0.005</td></tr><tr><td>Ballroom</td><td>0.517 1.392</td><td>0.021 4.530</td><td>0.985 0.560</td><td>13.595</td><td>4.375</td><td>0.472</td><td>15.436 4.796</td><td>0.515</td><td>0.250</td><td>0.854 0.010</td></tr><tr><td>Francis</td><td>0.216 0.971</td><td>0.012 6.810</td><td>2.013 0.531</td><td>11.802</td><td>3.390</td><td>0.507</td><td>11.696 2.200</td><td>0.538</td><td>0.197</td><td>0.891 0.010</td></tr><tr><td>Ignatius</td><td>0.032 0.164</td><td>0.006 3.630</td><td>0.397 0.544</td><td>4.375</td><td>1.761</td><td>0.579</td><td>6.150</td><td>1.992 0.395</td><td>0.030</td><td>0.145 0.006</td></tr><tr><td>Mean</td><td>0.211 0.520</td><td>0.013 3.519</td><td>0.751 0.403</td><td>7.641</td><td>2.121</td><td>0.436</td><td>8.261</td><td>1.950 0.446</td><td>0.109</td><td>0.350 0.008</td></tr></table>

Table 2. Quantitative comparisons of pose estimation on Tanks&Temples [14]. Each baseline method is trained with its public code under the original settings and evaluated with the same evaluation protocol. The best score of all the results is in bold. Note the Tanks&Temples here has more drastic camera motions as detailed in the Dataset part and appendix.

<!-- image-->  
Nope-NeRF  
CF-3DGS  
PCR-GS (Ours)  
Ground Truth  
Figure 4. Qualitative comparisons of PCR-GS with CF-3DGS [6] and Nope-NeRF [2] in novel view synthesis (on Tanks&Temples). PCR-GS achieves superior image rendering as compared with the two state-of-the-art methods.

Pose Estimation. We apply the Procrustes analysis as in [2] to post-process the estimated camera poses, transforming the estimated camera poses and the ground truth into a common coordinate space for comparisons. Table. 2 shows experimental results on Tanks&Temples. It can be observed that our method outperforms the CF-3DGS consistently in all scenes. We also evaluate the pose estimation on another benchmark Free-Dataset. As Table. 4 shows, our method achieves better performance than CF-3DGS as well. The quantitative comparisons demonstrate the effectiveness of our proposed method in camera pose estimation.

## 4.4. Ablation Study

We conduct ablation experiments to examine how our proposed feature reprojection regularization and wavelet-based frequency regularization contribute to scene reconstruction.

<!-- image-->  
CF-3DGS

<!-- image-->  
PCR-GS (Ours)

<!-- image-->  
Ground Truth

Figure 5. Qualitative comparison on novel view synthesis over the Free-dataset [34]. The proposed PCR-GS generates better details with less artifacts consistently.
<table><tr><td rowspan="2"></td><td rowspan="2">Scenes</td><td>CF-3DGS</td><td>PCR-GS (Ours)</td></tr><tr><td>PSNRâ SSIMâ LPIPSâ</td><td>PSNRâ SSIMâLPIPSâ</td></tr><tr><td rowspan="4">Fere-r</td><td>Pillar</td><td>14.47 0.41 0.61</td><td>17.15 0.49 0.52</td></tr><tr><td>Stair</td><td>16.82 0.48 0.48</td><td>20.81 0.60 0.35</td></tr><tr><td>Hydrant</td><td>14.02 0.22 0.56</td><td>15.37 0.37 0.52</td></tr><tr><td>Mean</td><td>15.00 0.37 0.55</td><td>17.78 0.49 0.46</td></tr></table>

Table 3. Quantitative comparisons on novel view synthesis over the Free-Dataset [34]. The best score is in bold.

<table><tr><td rowspan="2"></td><td rowspan="2">Scenes</td><td>CF-3DGS</td><td>PCR-GS (Ours)</td></tr><tr><td>RPEt RPEr â ATEâ</td><td>RPEt RPErâ ATEâ</td></tr><tr><td rowspan="4">Fre-r</td><td>Pillar</td><td>0.779 4.482 0.014</td><td>0.314 1.481 0.008</td></tr><tr><td>Stair 1.222</td><td>2.205 0.024</td><td>0.654 0.540 0.013</td></tr><tr><td>Hydrant</td><td>3.609 7.331 0.088</td><td>1.739 3.223 0.051</td></tr><tr><td>Mean 1.870</td><td>4.673 0.042</td><td>0.902 1.748 0.024</td></tr></table>

Table 4. Quantitative comparisons on pose estimation over the Free-Dataset [34]. The best score is in bold.

Feature Reprojection Regularization We first examine how our proposed feature reprojection regularization affects PSNR, SSIM, and LPIPS in Gaussian Splatting. We adopt CF-3DGS as the baseline Base, which simply uses RGB regularization to constrain camera pose estimation. On top of the Base, we train a new model Base+FRR that incorporates our proposed feature reprojection regularization. As Table.5 shows, the Base+FRR outperforms the Base clearly in PSNR, SSIM and LPIPS, demonstrating the effectiveness of the proposed feature reprojection regularization.

<table><tr><td rowspan=2 colspan=1>Models</td><td rowspan=1 colspan=1>Evaluation Metrics</td></tr><tr><td rowspan=1 colspan=1>PSNR â SSIM â LPIPS+</td></tr><tr><td rowspan=5 colspan=1>Base (CF-3DGS)Base+FRRBase+WRF(w/o high-freq)Base+WRF(w/ high-freq)Base+FRR+WRF(w/high-freq)</td><td rowspan=1 colspan=1>18.34  0.64   0.32</td></tr><tr><td rowspan=1 colspan=1>23.16  0.72   0.17</td></tr><tr><td rowspan=1 colspan=1>18.40  0.65   0.32</td></tr><tr><td rowspan=1 colspan=1>19.31  0.66   0.28</td></tr><tr><td rowspan=1 colspan=1>24.20  0.79   0.17</td></tr></table>

Table 5. Ablation studies of PCR-GS on Tanks& Temples. We report the performance on novel view synthesis with metrics PSNR, SSIM and LPIPS. With CF-3DGS as the baseline model Base, Base+FRR and Base+WFR introduce feature reprojection regularization (FRR) and wavelet-based frequency regularization (WFR), respectively. High-freq denotes the highfrequency component in wavelet-based frequency regularization. Base+FRR+WFR (i.e. PCR-GS) further introduces WFR on top of Base+FRR to optimize the rotation matrix in camera poses. The best results are highlighted in bold.

Wavelet-based Frequency Regularization We train two independent models that incorporate WFR on top of the Base. One model utilizes the high-frequency component, while the other does not. On top of the Base+FRR, we train a new model Base+FRR+WFR (i.e. complete PCR-GS) that further incorporates the proposed wavelet-based frequency regularization to evaluate how it contributes to the 3D scene reconstruction and novel view synthesis. Table 5 shows experimental results. We can observe that the proposed wavelet-based frequency regularization further improves the novel view synthesis effectively, with the highfrequency component playing an important role.

## 5. Conclusion

This paper presents PCR-GS, an innovative COLMAPfree 3DGS method that leverages pose co-regularization to achieve superior rendering quality and pose estimation accuracy while handling scenes with complex camera trajectories. Specifically, we design the feature reprojection regularization that extracts view-robust DINO features and aligns the semantic information between adjacent frames to regularize the relative camera poses. In addition, we design wavelet-based frequency regularization to regularize the rotation matrix of camera poses by minimizing discrepancies in high-frequency details. Extensive experiments demonstrate the effectiveness of our method on large-scale scenes with complex camera trajectories.

## 6. Acknowledgements

This project is funded by the Ministry of Education Singapore, under the Tier-1 project with the project number RT18/22.

## References

[1] Shir Amir, Yossi Gandelsman, Shai Bagon, and Tali Dekel. Deep vit features as dense visual descriptors. arXiv preprint arXiv:2112.05814, 2(3):4, 2021. 2, 4, 6

[2] Wenjing Bian, Zirui Wang, Kejie Li, Jia-Wang Bian, and Victor Adrian Prisacariu. Nope-nerf: Optimising neural radiance field with no pose prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4160â4169, 2023. 3, 6, 7

[3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J Â´ egou, Â´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650â9660, 2021. 2

[4] Tali Dekel, Shaul Oron, Michael Rubinstein, Shai Avidan, and William T Freeman. Best-buddies similarity for robust template matching. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2021â 2029, 2015. 5

[5] Tianchen Deng, Yaohui Chen, Leyan Zhang, Jianfei Yang, Shenghai Yuan, Jiuming Liu, Danwei Wang, Hesheng Wang, and Weidong Chen. Compact 3d gaussian splatting for dense visual slam. arXiv preprint arXiv:2403.11247, 2024. 2

[6] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20796â20805, 2024. 1, 2, 3, 4, 6, 7

[7] Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14346â 14355, 2021. 2

[8] Zhiyang Guo, Wengang Zhou, Li Li, Min Wang, and Houqiang Li. Motion-aware 3d gaussian splatting for efficient dynamic scene reconstruction. arXiv preprint arXiv:2403.11447, 2024. 2

[9] Shoukang Hu, Fangzhou Hong, Liang Pan, Haiyi Mei, Lei Yang, and Ziwei Liu. Sherf: Generalizable human nerf from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9352â9364, 2023. 2

[10] Tao Hu, Shu Liu, Yilun Chen, Tiancheng Shen, and Jiaya Jia. Efficientnerf efficient neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12902â12911, 2022. 2

[11] Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An, Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, and Shanghang Zhang. S3 gaussian: Self-supervised street gaussians for autonomous driving. arXiv preprint arXiv:2405.20323, 2024. 2

[12] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5885â5894, 2021. 2

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time

radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 3

[14] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36 (4):1â13, 2017. 5, 6, 7

[15] Min-Seop Kwak, Jiuhn Song, and Seungryong Kim. Geconerf: Few-shot neural radiance fields via geometric consistency. arXiv preprint arXiv:2301.10941, 2023. 2

[16] Dongwoo Lee and Kyoung Mu Lee. Dense depth-guided generalizable nerf. IEEE Signal Processing Letters, 30:75â 79, 2023. 2

[17] Mingrui Li, Shuhong Liu, Heng Zhou, Guohao Zhu, Na Cheng, Tianchen Deng, and Hongyu Wang. Sgs-slam: Semantic gaussian splatting for neural dense slam. In European Conference on Computer Vision, pages 163â179. Springer, 2025. 2

[18] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5741â5751, 2021. 3, 6

[19] Jia-Wei Liu, Yan-Pei Cao, Jay Zhangjie Wu, Weijia Mao, Yuchao Gu, Rui Zhao, Jussi Keppo, Ying Shan, and Mike Zheng Shou. Dynvideo-e: Harnessing dynamic nerf for large-scale motion-and view-change human-centric video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7664â 7674, 2024. 2

[20] Achleshwar Luthra, Shiva Souhith Gantha, Xiyun Song, Heather Yu, Zongfang Lin, and Liang Peng. Deblur-nsff: Neural scene flow fields for blurry dynamic scenes. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3658â3667, 2024. 2

[21] Quan Meng, Anpei Chen, Haimin Luo, Minye Wu, Hao Su, Lan Xu, Xuming He, and Jingyi Yu. Gnerf: Gan-based neural radiance field without posed camera. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6351â6361, 2021. 3

[22] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 2

[23] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022. 2

[24] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10318â10327, 2021. 2

[25] Yunlong Ran, Yanxu Li, Qi Ye, Yuchi Huo, Zechun Bai, Jiahao Sun, and Jiming Chen. Ct-nerf: Incremental optimizing neural radiance field and poses with complex trajectory. arXiv preprint arXiv:2404.13896, 2024. 6

[26] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi- Â´ sion transformers for dense prediction. In Proceedings of

the IEEE/CVF international conference on computer vision, pages 12179â12188, 2021. 4, 6

[27] Gernot Riegler and Vladlen Koltun. Free view synthesis. In Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part XIX 16, pages 623â640. Springer, 2020. 2

[28] Gernot Riegler and Vladlen Koltun. Stable view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12216â12225, 2021. 2

[29] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104â4113, 2016. 2

[30] Jurgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Â¨ Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 573â580. IEEE, 2012. 6

[31] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8248â8258, 2022. 2

[32] Richard Tucker and Noah Snavely. Single-view view synthesis with multiplane images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 551â560, 2020. 2

[33] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9065â9076, 2023. 2

[34] Peng Wang, Yuan Liu, Zhaoxi Chen, Lingjie Liu, Ziwei Liu, Taku Komura, Christian Theobalt, and Wenping Wang. F2- nerf: Fast neural radiance field training with free camera trajectories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4150â 4159, 2023. 6, 8

[35] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004. 6

[36] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. Nerfâ: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064, 2021. 3, 6

[37] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay, Pradyumna Chari, and Achuta Kadambi. Sparsegs: Realtime 360 {\deg} sparse view synthesis using gaussian splatting. arXiv preprint arXiv:2312.00206, 2023. 1, 2

[38] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Pointnerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5438â5448, 2022. 2

[39] Jianglong Ye, Naiyan Wang, and Xiaolong Wang. Featurenerf: Learning generalizable nerfs by distilling foundation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8962â8973, 2023. 2

[40] Jiahui Zhang, Fangneng Zhan, Rongliang Wu, Yingchen Yu, Wenqing Zhang, Bai Song, Xiaoqin Zhang, and Shijian Lu. Vmrf: View matching neural radiance fields. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6579â6587, 2022. 3

[41] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In European Conference on Computer Vision, pages 335â352. Springer, 2025. 2

[42] Qiang Zhang, Seung-Hwan Baek, Szymon Rusinkiewicz, and Felix Heide. Differentiable point-based radiance fields for efficient view synthesis. In SIGGRAPH Asia 2022 Conference Papers, pages 1â12, 2022. 2

[43] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 6

[44] MI Zhenxing and Dan Xu. Switch-nerf: Learning scene decomposition with mixture of experts for large-scale neural radiance fields. In The Eleventh International Conference on Learning Representations, 2022. 2

[45] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 2

[46] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21634â21643, 2024. 1, 2

[47] Siting Zhu, Renjie Qin, Guangming Wang, Jiuming Liu, and Hesheng Wang. Semgauss-slam: Dense semantic gaussian splatting slam. arXiv preprint arXiv:2403.07494, 2024. 2

[48] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In European Conference on Computer Vision, pages 145â163. Springer, 2025. 2