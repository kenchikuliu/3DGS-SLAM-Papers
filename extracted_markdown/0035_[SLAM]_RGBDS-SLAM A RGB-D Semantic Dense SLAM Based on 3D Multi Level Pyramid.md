# RGBDS-SLAM: A RGB-D Semantic Dense SLAM Based on 3D Multi Level Pyramid Gaussian Splatting

Zhenzhong Cao1, Chenyang Zhao1, Qianyi Zhang1, Jinzheng Guang1, Yinuo Song1, Jingtai Liu1â

AbstractâHigh-fidelity reconstruction is crucial for dense SLAM. Recent popular methods utilize 3D gaussian splatting (3D GS) techniques for RGB, depth, and semantic reconstruction of scenes. However, these methods ignore issues of detail and consistency in different parts of the scene. To address this, we propose RGBDS-SLAM, a RGB-D semantic dense SLAM system based on 3D multi-level pyramid gaussian splatting, which enables high-fidelity dense reconstruction of scene RGB, depth, and semantics. In this system, we introduce a 3D multi-level pyramid gaussian splatting method that restores scene details by extracting multi-level image pyramids for gaussian splatting training, ensuring consistency in RGB, depth, and semantic reconstructions. Additionally, we design a tightly-coupled multifeatures reconstruction optimization mechanism, allowing the reconstruction accuracy of RGB, depth, and semantic features to mutually enhance each other during the rendering optimization process. Extensive quantitative, qualitative, and ablation experiments on the Replica and ScanNet public datasets demonstrate that our proposed method outperforms current state-of-the-art methods, which achieves great improvement by 11.13% in PSNR and 68.57% in LPIPS. The open-source code will be available at: https://github.com/zhenzhongcao/RGBDS-SLAM.

## I. INTRODUCTION

Visual SLAM is a fundamental problem in the field of robotics, aimed at solving the problem of simultaneously locating a robot and constructing a map of its surrounding environment. Dense mapping is an important component of visual SLAM; on the one hand, it enables the robot to perceive its surroundings more comprehensively, and on the other hand, it provides a foundational map for downstream tasks such as grasping, manipulation, and interaction. However, traditional dense visual SLAM [1]â[6] relies solely on point clouds to reconstruct scenes, and due to the limited number of points and their discontinuous distribution, it faces significant bottlenecks and cannot achieve high-fidelity reconstructions of the environment.

With the advent of NeRF (Neural Radiance Fields) [7], scene representation based on implicit neural radiance fields has gradually become popular. Through training, the reconstruction accuracy has significantly improved, and many approaches have incorporated NeRF into SLAM [8]â[15], achieving highprecision RGB, depth, and semantic Reconstructions. However, NeRF itself suffers from issues such as long training times and slow rendering speeds, meaning that NeRF-based SLAM solutions cannot run in real time, which contradicts the original goal of SLAM.

3D GS [16] technology, with its efficient optimization framework and real-time rendering capability, improves upon the shortcomings of NeRF. As a result, many 3D GS-based SLAM [17]â[24] solutions have emerged. However, these methods typically train using only raw image features, which are insufficient to fully capture the fine-grained details of certain scene parts, leading to poor reconstruction consistency. Moreover, when performing multi-feature reconstruction, these approaches do not effectively fuse and optimize the features through reasonable constraints, preventing them from mutually enhancing each other.

To address the key issues of insufficient detail restoration, poor reconstruction consistency, ineffective fusion of multifeature information, and real-time challenges in reconstruction, we propose the RGBDS-SLAM algorithm in this paper. First, we introduce a 3D multi-level pyramid gaussian splatting method, which constructs a multi-level image pyramid to extract rich detail information at different resolution levels and perform gaussian splatting training. This method significantly improves the sceneâs detail restoration capability, and through stepwise optimization across levels, it ensures effective global consistency during reconstruction, providing a solid foundation for precise restoration of complex scenes. Second, we design a tightly coupled multi-features reconstruction optimization mechanism, which reasonably couples RGB, depth, and semantic features through various constraints. In the rendering optimization process, these three features collaborate and promote each other. Semantic information enhances depth understanding, depth information supports semantic refinement, and at the same time, the realism and consistency of RGB rendering are optimized, thereby comprehensively improving the accuracy and reliability of reconstruction. Finally, we develop a complete RGB-D Semantic Dense SLAM system, achieving high-quality dense reconstruction of scene RGB color, depth information, and semantic color. This system is based on the current classic ORB-SLAM3 algorithm [6], capable of processing complex scenes in real time and meeting the dual requirements of speed and accuracy for online applications.

The main contributions of this work are as follows:

â¢ We introduce a 3D Multi-Level Pyramid Gaussian Splatting (MLP-GS) method, which extracts multi-level image pyramids for gaussian splatting training, restoring scene details and ensuring consistency during reconstruction.

â¢ We design a Tightly Coupled Multi-Features Reconstruction Optimization(TCMF-RO) mechanism, which promotes mutual improvement of RGB, depth, and semantic map reconstruction accuracy during the optimization rendering process.

<!-- image-->  
Fig. 1. Overview of the proposed RGBDS-SLAM. Our method is an enhancement of ORB-SLAM3 [6], taking RGB, depth, and semantic frames as input and outputting a map database with the point map, gaussian origin map, and gaussian semantic map. It consists of four threads: Tracking, LocalMapping, GaussianMapping, and LoopClosing.

â¢ We develop a complete RGB-D Semantic Dense SLAM system capable of high-quality dense reconstruction of scene RGB, depth, and semantic information, and the system can operate in real time. We will also open source our code once the paper is accepted.

## II. RELATED WORK

## A. NeRF-based SLAM

The development in neural implicit representations, particularly those based on NeRF, have significantly enhanced the performance of SLAM systems. Among them, NICE-SLAM [8] is the first solution to combine NeRF and SLAM, which incorporates multi-level local information by introducing a hierarchical scene representation, enabling efficient map construction and robust tracking. However, NICE-SLAM suffers from computational efficiency issues. Therefore, [9], [10], [11] and [12] have introduced voxel-based neural representations, coordinate and sparse parameters, hybrid representation of signed distance fields (SDF) and neural point cloud respectively to optimize and improve computational efficiency. The above solutions do not consider semantic mapping, so based on these solutions, NIDS-SLAM [13] introduce a novel approach for dense 3D semantic segmentation, based on 2D semantic color information of keyframes, which are able to accurately learn the dense 3D semantics of the scene online while simultaneously learning geometry. However, this work does not integrate semantic with other features of the environment, such as geometry and appearance. Therefore, DNS-SLAM [14] integrates multi-view geometry constraints with image-based feature extraction to improve appearance details and to output color, density, and semantic class information. SNI-SLAM [15] introduce cross-attention based feature fusion to incorporate semantic, appearance, and geometry features, thus improving the accuracy of mapping, tracking, and semantic segmentaion. Although these NeRF-based SLAM schemes achieve highquality reconstruction effects, they suffer from poor scalability, low efficiency and poor real-time performance due to NeRF.

## B. 3D GS-based SLAM

The emergence of 3D GS have led to significant advancements in both general and semantic SLAM systems. [17]â[20] pioneered the introduction of 3D GS technology into SLAM systems, which are all committed to continuously expanding and optimizing gaussian map parameters in the incremental process of SLAM to achieve high-fidelity incremental reconstruction of scenes. However, their camera tracking modules all rely on gradient optimization of image loss, so the realtime performance of the systems is relatively poor. Photo-SLAM [21] introduces ORB-SLAM3 as the basic framework to improve this problem. None of the above solutions performs semantic mapping of the scene. Therefore, based on these solutions, SGS-SLAM [24] proposes to employ multichannel optimization during the mapping process, integrating appearance, geometric, and semantic constraints with keyframe optimization to enhance reconstruction quality. NEDS-SLAM [22] propose a spatially consistent feature fusion model to reduce the effect of erroneous estimates from pre-trained segmentation head on semantic reconstruction, achieving robust 3D semantic gaussian mapping. Although these 3D GS-based SLAM schemes achieve high-efficiency and high-precision dense reconstruction, they do not restore enough scene details, have poor consistency, and have low coupling of multi-feature information.

## III. RGBDS-SLAM ALGORITHMN

## A. Overall System Framework

The Fig.1 illustrates the overall framework of the proposed RGBDS-SLAM, which is based on ORB-SLAM3 [6]. The system takes RGB, depth, and semantic frames as input data and outputs a map database containing the point map, gaussian origin map, and gaussian semantic map. It primarily consists of four threads: Tracking Thread, LocalMapping Thread, GaussianMapping Thread, and LoopClosing Thread. The specific data flow between these threads is as follows:

Tracking Thread: Receives RGB-D frame data and estimates the camera pose for the current frame.

LocalMapping Thread: Receives the initial pose provided by the Tracking Thread, determines whether a new keyframe can be created, and if so, creates new keyframes and map points, optimizes the local map, and updates the point cloud map.

GaussianMapping Thread: Receives the new keyframe and map point data created by the LocalMapping Thread, converts it into 3D gaussian primitives (including position, color, semantics, depth, opacity, etc.), then performs the 3D multi-level pyramid gaussian splatting operation. Finally, the gaussian origin map and gaussian semantic map are updated through the tightly coupled multi-features reconstruction optimization mechanism.

Loop Closing Thread: Accepts new keyframe data from the map, performs loop closure, and if a loop is detected, executes global optimization and updates the entire map.

## B. 3D Gaussian Primitives Representation

We define that each 3D gaussian primitive includes position, shape, RGB color, depth value, and semantic color information. Referring to the operation in [25] that simplifies the gaussian parameters by reducing the shape component (transforming the covariance matrix from anisotropic to isotropic), we can define the expression for the influence of a 3D gaussian primitive on other spatial locations as follows:

$$
g ^ { 3 D } ( \pmb { x } ) = o \exp \left( - \frac { \left\| \pmb { x } - \pmb { \mu } \right\| ^ { 2 } } { 2 r ^ { 2 } } \right)\tag{1}
$$

where $\pmb { \mu }$ is the position of the 3D gaussian primitive, r is the shape, x is the spatial location, and o is the opacity.

As for data preparation of gaussian splatting, we convert the parameters in (1) into 2D using the cameraâs intrinsic parameters $\boldsymbol { K } ~ \in ~ \boldsymbol { R } ^ { 3 \times 3 }$ (symmetric matrix), focal length $f ,$ and extrinsic parameters $T _ { c w } \in R ^ { 3 \times 4 }$ (the transformation from world coordinates to camera coordinates):

$$
\pmb { \mu } ^ { 2 D } = K \frac { T _ { c w } \pmb { \mu } } { d } , r ^ { 2 D } = \frac { f r } { d } , d = ( T _ { c , w } \pmb { \mu } ) _ { z }\tag{2}
$$

By using the above equation, we project the 3D gaussian primitive onto the image plane to obtain a 2D gaussian primitive. We can then define the expression for the influence of the 2D gaussian primitive on other image pixels as follows:

$$
g ^ { 2 D } ( \pmb { p } ) = o \exp ( - \frac { \left\| \pmb { p } - \pmb { \mu } ^ { 2 D } \right\| ^ { 2 } } { 2 \left( r ^ { 2 D } \right) ^ { 2 } } )\tag{3}
$$

Using the above equation, we can proceed with the subsequent gaussian splatting operations. Additionally, for each 3D gaussian primitive, we convert its RGB color and semantic color information into multi-dimensional feature vectors r and s using the SH (Spherical Harmonics) method to represent them.

## C. 3D Multi-Level Pyramid Gaussian Splatting

Unlike the standard 3D gaussian splatting process, we refer to the progressive training process proposed in [26]â[30] and introduce a 3D multi-level pyramid gaussian splatting. In this process, the resolution of various feature images (RGB, depth, and semantic images) is gradually increased during training. This not only reduces training time and difficulty, but also allows for the gradual reconstruction of multi-scale information for different features at different resolutions.

<!-- image-->  
Fig. 2. Multi level image pyramid construction. During the training process, it is carried out from top to bottom, with the resolution of the image gradually increasing. First, low resolution is used for quick initialization, and then the details are gradually improved.

Therefore, we construct an n-layer image pyramid for RGB, depth, and semantic images.

The i-th layer of the RGB pyramid image can be represented as:

$$
I _ { r } ^ { g t } ( i ) = P y r a m i d I m a g e E x t r c a t i o n ( I _ { R G B } ^ { g t } , i )\tag{4}
$$

The i-th layer of the depth pyramid image can be represented as:

$$
I _ { d } ^ { g t } ( i ) = P y r a m i d I m a g e E x t r c a t i o n ( I _ { d e p t h } ^ { g t } , i )\tag{5}
$$

The i-th layer of the semantic pyramid image can be represented as:

$$
I _ { s } ^ { g t } ( i ) = P y r a m i d I m a g e E x t r c a t i o n ( I _ { s e m a n t i c } ^ { g t } , i )\tag{6}
$$

During the training process, to ensure comprehensive training for each viewpoint and each layer of the image pyramid, in each iteration, we randomly select a set of multi-feature images $\{ I _ { r } ^ { g t } ( i ) , I _ { d } ^ { g t } ( i ) , I _ { s } ^ { g t } ( i ) \}$ . We extract all relevant information for that viewpoint (such as pose, image size, etc.), and based on this information, we perform rendering operations for RGB, depth, and semantic images, referring to the rendering formula proposed in [16].

We perform RGB rendering operation using:

$$
R ( \pmb { p } ) = \sum _ { i \in N } r _ { i } g _ { i } ^ { 2 D } ( \pmb { p } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - g _ { j } ^ { 2 D } ( \pmb { p } ) \right)\tag{7}
$$

We perform depth rendering operation using:

$$
D ( \pmb { p } ) = \sum _ { i \in N } d _ { i } g _ { i } ^ { 2 D } ( \pmb { p } ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - g _ { j } ^ { 2 D } ( \pmb { p } ) )\tag{8}
$$

We perform semantic rendering operation using:

$$
S ( \pmb { p } ) = \sum _ { i \in N } s _ { i } g _ { i } ^ { 2 D } ( \pmb { p } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - g _ { j } ^ { 2 D } ( \pmb { p } ) \right)\tag{9}
$$

where, the set N represents the sorted 2D gaussian primitives required to render the RGB, depth and semantic of the p pixel, and the cumulative multiplication operation represents the cumulative effect of the previous 2D gaussian primitives on the current one.

Through our proposed MLP-GS progressive training process, we can gradually restore the scene details to the maximum extent.

D. Tightly Coupled Multi-Feature Reconstruction Optimization

In the previous section, we performed MLP-GS operations on the 3D gaussian primitives in the map, resulting in a set of rendered images $\{ I _ { r } ^ { r d } ( i ) , I _ { d } ^ { r d } ( i ) , I _ { s } ^ { r d } ( i ) \}$ . This is the forward rendering process of gaussian splatting. We now need to compute the loss between the rendered images and the ground truth images and perform backpropagation to optimize the 3D gaussian primitives in the map.

Referring to the calculation of L1 loss and SSIM loss for rendered images and the groundtruth images in [24], we perform a similar loss calculation on the rendered images $\{ I _ { r } ^ { g t } ( i ) , I _ { d } ^ { g t } ( i ) , I _ { s } ^ { g t } ( i ) \}$ of the i-th pyramid perspective obtained in the previous section.

For RGB images, we consider L1 and SSIM loss:

$$
\begin{array} { r } { L _ { r } ( i ) = ( 1 - \lambda _ { r } ) \left| I _ { r } ^ { r d } ( i ) - I _ { r } ^ { g t } ( i ) \right| + \lambda _ { r } S S I M ( I _ { r } ^ { r d } ( i ) , I _ { r } ^ { g t } ( i ) ) } \end{array}\tag{10}
$$

For depth images, we only consider L1 loss:

$$
L _ { d } ( i ) = \left| I _ { d } ^ { r d } ( i ) - I _ { d } ^ { g t } ( i ) \right|\tag{11}
$$

For semantic images, we similarly consider L1 and SSIM loss:

$$
\begin{array} { r } { L _ { s } ( i ) = ( 1 - \lambda _ { s } ) \left| I _ { s } ^ { r d } ( i ) - I _ { s } ^ { g t } ( i ) \right| + \lambda _ { s } S S I M ( I _ { s } ^ { r d } ( i ) , I _ { s } ^ { g t } ( i ) ) } \end{array}\tag{12}
$$

Finally, we tightly couple multiple features into a reconstruction optimization framework to perform joint optimization:

$$
L _ { r e c o n s t r u c t i o n } ( i ) = L _ { r } ( i ) + L _ { d } ( i ) + L _ { s } ( i )\tag{13}
$$

Through the proposed TCMF-RO, which couples multiple features within a single framework, the RGB, depth, and semantic features in the 3D gaussian primitives can promote and enhance each other during optimization.

## IV. EXPERIMENT AND EVALUATION

## A. Experimental Setup

Datasets: We comprehensively evaluated the proposed method on both synthetic and real-world datasets, including 8 sequences from the Replica dataset [25], 6 sequences from the ScanNet dataset [31].

Metrics: Following the evaluation section of NEDS-SLAM [22], we use RSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity) [32], and LPIPS (Learned Perceptual Image Patch Similarity) [33] for evaluating RGB reconstruction quality. For depth reconstruction quality, we use the L1. For semantic reconstruction quality, we use mIoU (mean Intersection over Union). For camera localization accuracy, we use ATE Mean and ATE RMSE.

Baselines: We selected several NeRF-based SLAM systems, including NICE-SLAM [8], Vox-Fusion [9], Co-SLAM [10], ESLAM [11], NIDS-SLAM [13], DNS-SLAM [14], and SNI-SLAM [15], for comparison with our method. Additionally, we chose 3D GS-based SLAM systems, such as Splatam [17], Photo-SLAM [21], NEDS-SLAM [22], and SGS-SLAM [24], to compare with our approach. All comparative data in this paper are derived from the original texts of the aforementioned baselines.

Platform: The hardware platform used for the experiments is a laptop equipped with an NVIDIA RTX 3060 GPU and an AMD Ryzen 7 5800H CPU. The software platform is Ubuntu 18.04, with the code written in C++. For convenience of reimplement, we have created a docker container for the code and dependencies.

Parameters: We set the number of image pyramid levels to 3. We set $\lambda _ { r } = 0 . 2$ and $\lambda _ { s } = 0 . 2$

## B. Quantitative Experiments

Table.I shows the quantitative comparison of RGB reconstruction quality between our method and the baselines on 8 sequences of the Replica dataset. As can be seen, our proposed method performs well in RGB reconstruction quality, especially in PSNR and LPIPS metrics, achieving the best results and surpassing the current state-of-the-art methods. Compared to the second-best results, our method improves by 11.13% in PSNR and 68.57% in LPIPS. This improvement is due to the introduction of 3D multi-level pyramid gaussian splatting in our method, which better restores the scene details compared to SGS-SLAM [24] and Photo-SLAM [21]. Our method also achieves competitive second-best performance in SSIM.

Table.II shows the average quantitative comparison of Depth, ATE, and FPS metrics between our method and the baselines on 8 sequences of the Replica dataset. Our method demonstrates competitive performance in both depth and FPS metrics. The performance of ATE is close to Photo-SLAM [21], as we directly use the tracking module of ORB-SLAM3 [6] without further optimization. Our method also achieves better performance of Tracking FPS and Mapping FPS compared with SGS-SLAM [24](implement with Python code), which enables our system to run in real-time.

Table.III shows the quantitative comparison of semantic image reconstruction quality between our method and the baselines on 4 sequences of the Replica dataset. Compared to the currently best-performing SGS-SLAM [24], our method achieves a higher average mIoU of 94.32.

TABLE I  
QUANTITATIVE COMPARISON OF RGB RECONSTRUCTION QUALITY BETWEEN OUR METHOD AND BASELINES ON 8 SEQUENCES OF REPLICA DATASET.
<table><tr><td colspan="2">Method</td><td>Metric PSNRâ</td><td>office0</td><td>office1</td><td>office2</td><td>office3</td><td>office4</td><td>room0</td><td>room1</td><td>room2</td><td>avg</td></tr><tr><td rowspan="7"></td><td rowspan="2">NICE-SLAM [8]</td><td> SIM</td><td>29.07 0.874</td><td>30.34 0.886</td><td>19.66 0.797</td><td>22.23 0.801</td><td>24.94 0.856</td><td>22.12 0.689</td><td>22.47 0.757</td><td>24.52 0.814</td><td>24.42 0.809</td></tr><tr><td>LPIPS</td><td>0.229</td><td>0.181</td><td>0.235</td><td>0.209</td><td>0.198</td><td>0.330</td><td>0.271</td><td>0.208</td><td>0.233</td></tr><tr><td rowspan="2">Vox-Fusion [9]</td><td rowspan="2">PSNRâ SIM</td><td>27.79</td><td>29.83 0.876</td><td>20.33</td><td>23.47</td><td>25.21</td><td>22.39</td><td>22.36</td><td>23.92</td><td>24.41</td></tr><tr><td>0.857</td><td></td><td>0.794</td><td>0.803</td><td>0.847</td><td>0.683</td><td>00.751</td><td>0..798</td><td>0.801</td></tr><tr><td rowspan="2"></td><td></td><td>0.241</td><td>0.184</td><td>0.243</td><td>0.213</td><td>0.199</td><td>0.303</td><td>0.269</td><td>0.234</td><td>0.236</td></tr><tr><td>LPIPSâ PSNRâ</td><td>34.14</td><td>34.87</td><td>28.43</td><td>28.76</td><td>30.91</td><td>27.27</td><td>28.45</td><td>29.06</td><td>30.24</td></tr><tr><td rowspan="2">Co-SLAM [10]</td><td rowspan="2">SSIM LPIPS</td><td>0.961 0.209</td><td>0.969</td><td>0.938</td><td>0.941</td><td>0.955</td><td>0.910</td><td>0.909</td><td>0.932</td><td>0.939</td></tr><tr><td></td><td>0.196</td><td>0.258</td><td>0.229</td><td>0.236</td><td>0.324</td><td>0.294</td><td>0.266</td><td>0.252</td></tr><tr><td rowspan="6"></td><td rowspan="3">ESLAM [10]</td><td>PSNRâ SSIM</td><td>33.71 0.960</td><td>30.20 0.923</td><td>28.09 0.943</td><td>28.77</td><td>29.71</td><td>25.32</td><td>27.77</td><td>29.08</td><td>29.08</td></tr><tr><td></td><td></td><td></td><td></td><td>0.948</td><td>0.945</td><td>0.875</td><td>0.902</td><td>0.932</td><td>0.929</td></tr><tr><td>LPIPS PSNRâ</td><td>0.184</td><td>0.228</td><td>0.241</td><td>0.196</td><td>0.204</td><td>0.313</td><td>0.298</td><td>0.248</td><td>0.239</td></tr><tr><td rowspan="2">SplaTAM [17]</td><td>38.26</td><td></td><td>39.17</td><td>31.97</td><td>29.70</td><td>31.81</td><td>32.86</td><td>33.89</td><td>35.25</td><td>34.11</td></tr><tr><td>SSIM</td><td>0.98</td><td>0.98</td><td>0.97</td><td>0.95</td><td>0.95</td><td>0.98</td><td>0.97</td><td>0.98</td><td>0.970</td></tr><tr><td rowspan="2">Photo-SLAM [21]</td><td>LPIPSâ</td><td>0.09</td><td>0.09</td><td>0.10</td><td>0.12</td><td>0.15</td><td>0.07</td><td>0.10</td><td>0.08</td><td></td><td>0.100</td></tr><tr><td>PSNRâ SSIM</td><td>38.48</td><td>39.09</td><td>33.03</td><td>33.79</td><td>36.02</td><td></td><td>30.72</td><td>33.51</td><td>35.03</td><td>34.96</td></tr><tr><td rowspan="6">3D GS-based SLAM</td><td rowspan="2"></td><td>LPIPS</td><td>0.964</td><td>0.961</td><td>0.938</td><td>0.938</td><td>0.952</td><td>0.899</td><td>0.934</td><td>0.951</td><td>0.942</td></tr><tr><td></td><td>0.050 1</td><td>0.047</td><td>0.077</td><td>0.066</td><td>0.054</td><td>0.075</td><td>0.057</td><td>0.043</td><td>0.059</td></tr><tr><td rowspan="2">PSNRâ NEDS-SLAM [22]</td><td></td><td></td><td>/</td><td>T</td><td>/</td><td>1</td><td>1</td><td>1</td><td>/</td><td>34.76</td></tr><tr><td>SSIMâ LPIPSâ</td><td>I I</td><td>I I</td><td>/</td><td>I</td><td>/</td><td>/</td><td>I</td><td>I</td><td>0.962</td></tr><tr><td rowspan="2"></td><td></td><td></td><td></td><td>/</td><td>/</td><td>/</td><td>1</td><td>I</td><td>I</td><td>0.88</td></tr><tr><td>PSNRâ SIM</td><td>38.54</td><td>39.20</td><td>32.90</td><td>32.05</td><td>32.75</td><td>32.50</td><td>34.25</td><td>35.10</td><td>34.66</td></tr><tr><td rowspan="2">SGS-SLAM [24]</td><td>LPIPS</td><td>0.984 0.086</td><td>0.982 0.087</td><td>0.965 0.101</td><td></td><td>0.966</td><td>0.949</td><td>0.976</td><td>0.978</td><td>0.982</td><td>0.973</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.115</td><td>0.148</td><td>0.070</td><td>0.094</td><td>0.070 39.58</td><td>0.096</td></tr><tr><td rowspan="2">RGBDS-SLAM(Ours)</td><td>PSNRâ SSIM</td><td>42.46 0.981</td><td>42.57 0.976</td><td>35.80 0.959</td><td></td><td>36.53 0.958</td><td>39.47 0.969</td><td>35.77 0.955</td><td>38.59 0.968</td><td>0.973</td><td>38.85 00.967</td></tr><tr><td>LPIPSâ</td><td>.023</td><td>.029</td><td></td><td>0.052</td><td>.046</td><td>0.034</td><td>0.037</td><td>.029</td><td>0.027</td><td>.035</td></tr></table>

/ indicates that the paper does not provide relevant data, bold data indicates optimal data, and underlined data indicates suboptimal data.

TABLE II  
QUANTITATIVE COMPARISON OF AVERAGE RESULTS ON DEPTH, ATE, AND FPS METRICS BETWEEN OUR METHOD AND BASELINES ON 8 SEQUENCES OF REPLICA DATASET.
<table><tr><td colspan="2">Method</td><td>Depth(cm)â</td><td>ATE Mean (cm)â</td><td>ATE RMSE (cm)â</td><td>Tracking FPSâ</td><td>Mapping FPSâ</td></tr><tr><td rowspan="5">NeRF-based SLAM</td><td>NICE-SLAM [8]</td><td>1.903</td><td>1.795</td><td>2.503</td><td>13.70</td><td>0.20</td></tr><tr><td>Vox-Fusion [9]</td><td>2.913</td><td>1.027</td><td>1.473</td><td>2.11</td><td>2.17</td></tr><tr><td>Co-SLAM [10]</td><td>1.513</td><td>0.935</td><td>1.059</td><td>17.24</td><td>10.20</td></tr><tr><td>ESLAM [11]</td><td>0.945</td><td>0.545</td><td>0.678</td><td>18.11</td><td>3.62</td></tr><tr><td>SNI-SLAM [15]</td><td>0.766</td><td>0.397</td><td>0.456</td><td>16.03</td><td>2.48</td></tr><tr><td rowspan="5">3D GS-based SLAM</td><td>SplaTAM [17]</td><td>0.490</td><td>/</td><td>0.360</td><td>5.26</td><td>3.03</td></tr><tr><td>Photo-SLAM [21]</td><td>/</td><td>I</td><td>0.604</td><td>42.49</td><td>I</td></tr><tr><td>NEDS-SLAM [22]</td><td>0.470</td><td>/</td><td>0.354</td><td>/</td><td>/</td></tr><tr><td>SGS-SLAM [24]</td><td>0.356</td><td>0.327</td><td>0.412</td><td>5.27</td><td>3.52</td></tr><tr><td>RGBDS-SLAM(Ours)</td><td>0.342</td><td>0.499</td><td>0.589</td><td>29.55</td><td>32.22</td></tr></table>

## C. Qualitative Experiments

Fig.3 shows the qualitative results of randomly rendered RGB images on 8 sequences of the Replica dataset. It can be seen that our method accurately restores fine details in the scene, such as small numbers, textures, and boundaries.

Additionally, Fig.4 shows the qualitative comparison results between rendered depth images and groundtruth depth images for our method on the office0 sequence of Replica dataset. It is worth mentioning that even though the input depth image has missing areas, our method is still able to render the depth information in these regions, maintaining good consistency with the surrounding depth information.

Furthermore, Fig.5 shows the qualitative comparison results of semantic image rendering on 4 sequences of the

TABLE III  
QUANTITATIVE COMPARISON OF SEMANTIC IMAGE RECONSTRUCTION QUALITY BETWEEN OUR METHOD AND BASELINES ON 4 SEQUENCES OF REPLICA DATASET.
<table><tr><td>Method</td><td>AVG.mIoU(%)â</td><td>room0</td><td>room1</td><td>room2</td><td>office0</td></tr><tr><td>NIDS-SLAM [13]</td><td>82.37</td><td>82.45</td><td>84.08</td><td>76.99</td><td>85.94</td></tr><tr><td>DNS-SLAM [14]</td><td>84.77</td><td>88.32</td><td>84.90</td><td>81.20</td><td>84.66</td></tr><tr><td>SNI-SLAM [15]</td><td>87.41</td><td>88.42</td><td>87.43</td><td>86.16</td><td>87.63</td></tr><tr><td>NEDS-SLAM [22]</td><td>90.78</td><td>90.73</td><td>91.20</td><td>I</td><td>90.42</td></tr><tr><td>SGS-SLAM [24]</td><td>92.72</td><td>92.95</td><td>92.91</td><td>92.10</td><td>92.90</td></tr><tr><td>RGBDS-SLAM(Ours)</td><td>94.32</td><td>92.67</td><td>95.77</td><td>94.91</td><td>93.91</td></tr></table>

Replica dataset. Our method significantly restores the semantic segmentation results of the scene, especially at the boundaries. The comparison before and after optimization further demonstrates the effectiveness of our proposed semantic image rendering and optimization method.

<!-- image-->  
Fig. 3. Qualitative performance of our proposed method on RGB image rendering details from 8 sequences of the Replica dataset is shown. The first and third rows display the randomly rendered RGB images from the 8 sequences, while the second and fourth rows show the corresponding zoomed-in details. The regions of interest in the zoomed-in images are indicated with orange boxes and arrow lines to highlight the magnified details.

<!-- image-->  
Fig. 4. Qualitative comparison of rendered depth images and groundtruth depth images of our method on office0 sequence of Replica dataset. The first row is the randomly rendered depth images, and the second row is the corresponding groundtruth depth images. The red boxes indicate the differences. The red boxes on the groundtruth depth indicate the areas with missing depth.

## D. Ablation Study

Effectiveness of MLP-GS Module: Fig.6 shows the ablation study of the multi-level pyramid gaussian splatting module in our proposed method on ScanNet dataset. It can be seen that the rendered images using the MLP-GS process clearly preserve more scene details, including object contours, boundaries between objects, and the fine-grained details of small objects.

Effectiveness of TCMF-RO Module: Table.IV shows the ablation study of the tightly-coupled multi-feature reconstruction optimization module in our method, which focuses on the impact of depth and semantic features on various metrics. As can be seen, when both depth and semantic features are included in the optimization, the best performance is achieved. This demonstrates the effectiveness of our proposed tightly-coupled multi-feature reconstruction optimization mechanism, where RGB, depth, and semantic features mutually promote each other, leading to an overall improvement in the reconstruction quality.

TABLE IV  
ABLATION STUDY OF THE TIGHTLY-COUPLED MULTI-FEATURERECONSTRUCTION OPTIMIZATION MECHANISM IN OUR PROPOSEDMETHOD.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Depthâ</td><td>mIoUâ</td></tr><tr><td>w/o depth &amp; semantic</td><td>36.62</td><td>0.950</td><td>0.050</td><td>7</td><td></td></tr><tr><td>w/o depth</td><td>38.36</td><td>0.966</td><td>0.035</td><td>I</td><td>94.20</td></tr><tr><td>w/o semantic</td><td>38.44</td><td>0.965</td><td>0.040</td><td>0.345</td><td>I</td></tr><tr><td>w/ depth &amp; semantic</td><td>38.85</td><td>0.967</td><td>0.035</td><td>0.342</td><td>94.32</td></tr></table>

w/o means without, w/ means with.

<!-- image-->

Fig. 5. Qualitative comparison of semantic image rendering of our method on four sequences of Replica dataset. The first row is the RGB image rendered from a random perspective, and the second and third rows are the corresponding rendered semantic images, where the second row is the image before optimization and the third row is the image after optimization. The yellow box indicates the difference comparison with clear semantic segmentation boundaries in the corresponding area.  
<!-- image-->  
Fig. 6. Ablation study of the multi-level pyramid gaussian splatting in our proposed method on ScanNet dataset. The first row shows the multi-frame RGB image rendering results using the standard GS process instead of our proposed MLP-GS. The second row shows the corresponding multi-frame RGB image rendering results using MLP-GS. The areas with significant differences in the images are highlighted with green boxes.

Correction of Semantic Information: In the above experiments, we only used the groundtruth semantic images for training from the Replica dataset. The current best-performing SGS-SLAM [24] also relies solely on groundtruth semantic images for evaluation. However, since groundtruth semantic images are difficult to obtain and cannot be scaled to real-world scenarios, we used the SAM2 network [34] to obtain semantic segmentation results and replaced the original groundtruth semantic images for our experiments. Fig.7 shows a comparison between the SAM2 segmentation results and the rendered results after semantic reconstruction results of our method. We observed that, compared to semantic groundtruth, the SAM2 segmentation results lack consistency and continuity, with many instances of missed and incorrect segmentation. However, our method does not directly optimize based on the SAM2 segmentation results; instead, it uses multi-frame observations to correct the semantic information, which addresses issues like unclear object boundaries and object omissions in the segmentation. It demonstrates that our proposed method is scalable and can be easily extended to real-world applications.

## V. CONCLUSION

In this paper, we propose RGBDS-SLAM, which is a complete RGB-D semantic dense SLAM system, focusing on gaussian mapping. We first introduce a 3D multi-level pyramid gaussian splatting method to reconstruct the details and consistency of the scene. We futhermore design a tightly coupled multi-feature reconstruction optimization mechanism that promotes the optimization of RGB, depth, and semantic features, enhancing each other. Experiments also demonstrate the effectiveness and scalability of our proposed method. However, we have not considered the issue of dynamic scenes. Robustly reconstructing the RGB, depth, and semantic information in dynamic scenes will be the focus of our future work.

<!-- image-->

<!-- image-->

Object Boundary Comparison for Semantic Segmentation  
<!-- image-->

<!-- image-->  
Object Existence Comparison for Semantic Segmentation

Fig. 7. Comparison between the SAM2 segmentation results and the rendered results after our method performs semantic reconstruction. The first row displays a comparison of object boundaries in the semantic segmentation, while the second row shows a comparison of object existence in the semantic segmentation.

## REFERENCES

[1] T. Whelan, R. F. Salas-Moreno, B. Glocker, A. J. Davison, and S. Leutenegger, âElasticfusion: Real-time dense slam and light source estimation,â The International Journal of Robotics Research, vol. 35, no. 14, pp. 1697â1716, 2016.

[2] J. Engel, V. Koltun, and D. Cremers, âDirect sparse odometry,â IEEE transactions on pattern analysis and machine intelligence, vol. 40, no. 3, pp. 611â625, 2017.

[3] R. Mur-Artal and J. D. Tardos, âOrb-slam2: An open-source slam Â´ system for monocular, stereo, and rgb-d cameras,â IEEE transactions on robotics, vol. 33, no. 5, pp. 1255â1262, 2017.

[4] R. Scona, M. Jaimez, Y. R. Petillot, M. Fallon, and D. Cremers, âStaticfusion: Background reconstruction for dense rgb-d slam in dynamic environments,â in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 3849â3856.

[5] T. Zhang, H. Zhang, Y. Li, Y. Nakamura, and L. Zhang, âFlowfusion: Dynamic dense rgb-d slam based on optical flow,â in 2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020, pp. 7322â7328.

[6] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, Â´ âOrb-slam3: An accurate open-source library for visual, visualâinertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[8] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 12 786â12 796.

[9] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, âVoxfusion: Dense tracking and mapping with voxel-based neural implicit representation,â in 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2022, pp. 499â507.

[10] H. Wang, J. Wang, and L. Agapito, âCo-slam: Joint coordinate and sparse parametric encodings for neural real-time slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 293â13 302.

[11] M. M. Johari, C. Carta, and F. Fleuret, âEslam: Efficient dense slam system based on hybrid representation of signed distance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 408â17 419.

[12] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald, âPoint-slam:Â¨ Dense neural point cloud-based slam,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 433â18 444.

[13] Y. Haghighi, S. Kumar, J.-P. Thiran, and L. Van Gool, âNeural implicit dense semantic slam,â arXiv preprint arXiv:2304.14560, 2023.

[14] K. Li, M. Niemeyer, N. Navab, and F. Tombari, âDns slam: Dense neural semantic-informed slam,â arXiv preprint arXiv:2312.00204, 2023.

[15] S. Zhu, G. Wang, H. Blum, J. Liu, L. Song, M. Pollefeys, and H. Wang, âSni-slam: Semantic neural implicit slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 167â21 177.

[16] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussianÂ¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[17] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[18] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, âGsslam: Dense visual slam with 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 595â19 604.

[19] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[20] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, âGaussian-slam: Photo-realistic dense slam with gaussian splatting,â arXiv preprint arXiv:2312.10070, 2023.

[21] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Realtime simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 584â21 593.

[22] Y. Ji, Y. Liu, G. Xie, B. Ma, Z. Xie, and H. Liu, âNeds-slam: A neural explicit dense semantic slam framework using 3d gaussian splatting,â IEEE Robotics and Automation Letters, 2024.

[23] S. Zhu, R. Qin, G. Wang, J. Liu, and H. Wang, âSemgauss-slam: Dense semantic gaussian splatting slam,â arXiv preprint arXiv:2403.07494, 2024.

[24] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, âSgsslam: Semantic gaussian splatting for neural dense slam,â in European Conference on Computer Vision. Springer, 2025, pp. 163â179.

[25] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[26] L. Liu, J. Gu, K. Zaw Lin, T.-S. Chua, and C. Theobalt, âNeural sparse voxel fields,â Advances in Neural Information Processing Systems, vol. 33, pp. 15 651â15 663, 2020.

[27] C. Sun, M. Sun, and H.-T. Chen, âDirect voxel grid optimization: Superfast convergence for radiance fields reconstruction,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5459â5469.

[28] T. Takikawa, J. Litalien, K. Yin, K. Kreis, C. Loop, D. Nowrouzezahrai, A. Jacobson, M. McGuire, and S. Fidler, âNeural geometric level of detail: Real-time rendering with implicit 3d shapes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 11 358â11 367.

[29] Z. Li, T. Muller, A. Evans, R. H. Taylor, M. Unberath, M.-Y. Liu, and Â¨ C.-H. Lin, âNeuralangelo: High-fidelity neural surface reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8456â8465.

[30] Y. Xiangli, L. Xu, X. Pan, N. Zhao, A. Rao, C. Theobalt, B. Dai, and D. Lin, âBungeenerf: Progressive neural radiance field for extreme multiscale scene rendering,â in European conference on computer vision. Springer, 2022, pp. 106â122.

[31] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. NieÃner, âScannet: Richly-annotated 3d reconstructions of indoor scenes,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828â5839.

[32] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE transactions on image processing, vol. 13, no. 4, pp. 600â612, 2004.

[33] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.

[34] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, Â¨ C. Rolland, L. Gustafson et al., âSam 2: Segment anything in images and videos,â arXiv preprint arXiv:2408.00714, 2024.