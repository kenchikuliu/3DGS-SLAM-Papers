# GSFF-SLAM: 3D SEMANTIC GAUSSIAN SPLATTING SLAM VIA FEATURE FIELD

Zuxing Lu, Xin Yuan, Shaowen Yang, Jingyu Liu, Changyin Sunâ

Southeast University

Nanjing

{luzuxing, xinyuan, 220232017, 230239497, cysun}@seu.edu.cn

Jiawei Wang,   
Tongji University   
Shanghai   
wangjw@tongji.edu.cn

## ABSTRACT

Semantic-aware 3D scene reconstruction is essential for autonomous robots to perform complex interactions. Semantic SLAM, an online approach, integrates pose tracking, geometric reconstruction, and semantic mapping into a unified framework, shows significant potential. However, existing systems, which rely on 2D ground truth priors for supervision, are often limited by the sparsity and noise of these signals in real-world environments. To address this challenge, we propose GSFF-SLAM, a novel dense semantic SLAM system based on 3D Gaussian Splatting that leverages feature fields to achieve joint rendering of appearance, geometry, and N-dimensional semantic features. By independently optimizing feature gradients, our method supports semantic reconstruction using various forms of 2D priors, particularly sparse and noisy signals. Experimental results demonstrate that our approach outperforms previous methods in both tracking accuracy and photorealistic rendering quality. When utilizing 2D ground truth priors, GSFF-SLAM achieves state-of-the-art semantic segmentation performance with 95.03% mIoU, while achieving up to 2.9Ã speedup with only marginal performance degradation.

## 1 Introduction

Visual Simultaneous Localization and Mapping (Visual SLAM) is a widely used technique in robotics and computer vision [1, 2, 3, 4, 5, 6], enabling intelligent agents to reconstruct 3D scenes of unknown environments using monocular or multiple cameras while simultaneously tracking their positions over time. Neural Radiance Fields (NeRF) [7] and 3D Gaussian Splatting (3DGS) [8] are two recently emerging 3D reconstruction methods. Notably, their high-quality 3D geometric representations and novel view synthesis capabilities significantly enhance the utilization efficiency of high-dimensional image inputs. These advancements have facilitated the transition from traditional sparse Visual SLAM to learnable dense Visual SLAM systems [9, 10, 11, 12, 13].

Semantic SLAM aims to maintain robust tracking accuracy over long sequences without relying on precise Structurefrom-Motion (SFM) [14] poses, and it performs reconstruction from sparse multi-view frames while compressing dense semantic information into 3D scene representations. For instance, SemanticFusion [15] associates each point in the point cloud with a semantic label, thereby enriching the scene representations. Inspired by offline reconstruction works [16, 17], NeRF-based semantic SLAM methods [18, 19, 20] represent scenes using the learnable implicit neural networks and achieve semantic map rendering by integrating additional semantic Multi-Layer Perceptrons (MLPs). However, these methods are more prone to catastrophic forgetting than offline approaches, especially in long sequences, due to the absence of memory replay mechanisms. In contrast, the SGS-SLAM [21] avoids this issue by utilizing explicit 3D Gaussians as the scene representation, and it achieves high-speed semantic map rendering by converting semantic labels into RGB images. These Semantic SLAM methods [18, 19, 20, 21] integrate semantic losses into tracking and mapping, improving localization accuracy with multi-view consistent priors such as ground truth annotations. However, this design relies heavily on high-quality priors, restricting its use in scenarios lacking such annotations. Moreover, the effectiveness of semantic SLAM methods utilizing noisy and sparse priors remains unverified.

<!-- image-->  
Figure 1: Our GSFF-SLAM leverages different forms of signals to enhance various downstream online tasks. Our method projects 2D priors into the 3D feature field, enabling high-precision close-set segmentation, text-guided segmentation, and dense feature map rendering.

To address these limitations, we propose a novel approach that represents the semantic information through Ndimensional feature fields. We decouple the semantic optimization process from scene reconstruction by independently optimizing semantic embedding gradients. Specifically, we first reconstruct the input images to obtain high-quality geometric representations before performing semantic optimization. This decoupling of the mapping process into two steps enables the framework to support diverse forms of supervision signals. As demonstrated in the text-guided segmentation results shown in Figure 1, our method achieves high-quality semantic reconstruction even with noisy and sparse textual priors. Furthermore, the feature field densifies sparse features extracted from foundation models, enriching each 3D Gaussian in the scene with comprehensive semantic information. Overall, our contributions are summarized as follows:

â¢ We propose GSFF-SLAM, a novel Semantic SLAM framework based on 3D Gaussian Splatting, which leverages N-dimensional feature fields to achieve high-quality semantic reconstruction and dense feature map rendering.

â¢ Our framework decouples semantic optimization from scene reconstruction by independently optimizing semantic embedding gradients, ensuring robust performance even with noisy and sparse 2D priors.

â¢ On the Replica dataset [22], our method achieves state-of-the-art semantic segmentation performance of 95.03% mIoU with 2D ground truth priors, while delivering a runtime improvement of up to 2.9Ã with only a slight trade-off in performance.

## 2 Related Work

3D Gaussian Splatting. 3DGS [8] is an explicit 3D scene representation that introduces learned 3D Gaussians, Î±-blending, and an efficient parallel Gaussian rasterizer. Benefiting from its high rendering speed and explicit representation capabilities, 3DGS has facilitated significant advancements in various applications, including Visual SLAM [12, 13, 23], language-guided scene editing [24] and offline semantic reconstruction [25, 26]. MonoGS [13] proposes a monocular SLAM system based on 3DGS, which selects keyframes based on inter-frame co-visibility instead of fixed frame intervals or predefined distance-angle thresholds. LEGaussians [25] introduces semantic feature embedding into 3DGS, enabling offline semantic reconstruction. GaussianEditor [24] utilizes the SAM [27] model to segment and annotate semantics in 2D images, followed by explicit semantic information retrieval and editing.

<!-- image-->  
Figure 2: Overview of GSFF-SLAM. Our method takes an RGB-D stream as input, leveraging 3D Gaussian Splatting with semantic feature embedding f to generate RGB images, depth images, and dense feature maps. Semantic signals, derived from foundation models or ground truth, supervise the learning process, while the feature embedding f is optimized independently.

Dense Visual SLAM. SLAM typically divided into two main tasks: mapping and tracking [28]. Unlike sparse SLAM methods that primarily focus on pose estimation [29, 4, 5], dense visual SLAM methods aim to reconstruct detailed 3D maps [30, 10, 9]. Map representations in dense SLAM can be broadly categorized into two types: frame-centric and map-centric. Frame-centric methods anchor 3D geometry to specific keyframes, estimating frame depth and inter-frame motion [31, 6]. On the other hand, Map-centric methods convert 2D images into 3D geometry aligned with a unified world coordinate system. Common 3D geometry primitives include point clouds [32, 11, 33, 34], voxel grids [35, 36, 10, 9], and 3D Gaussians [12, 13, 23]. Point clouds can flexibly adjust the sparsity of spatial points, but due to the lack of correlation between primitives, the design of optimization is more challenging. Voxel grids enable fast 3D retrieval but incur high memory and computational costs. The NeRF-based SLAM methods [30, 9] significantly reduce spatial occupancy by converting explicit voxel grids into neural implicit representations. As a novel primitive, 3D Gaussians exhibit differentiable and continuous properties, enabling 3DGS-based SLAM methods [12, 13, 23] to achieve efficient training and inference.

Semantic Reconstruction. Semantic reconstruction focuses on enriching occupancy maps by integrating semantic information into map structures [18, 37, 17, 38, 39, 16, 26, 20]. 2D semantic reconstruction methods typically fuse 2D priors with the map through projection, generating top-down multi-layer map structures [37, 39]. These methods emphasize interaction with complex natural language, generating one or more trajectories, with a focus on languageguided planning capabilities. In contrast, 3D methods prioritize precise localization, accurate contours, and edge details [18, 20]. NeRF-based methods output semantic features using additional MLPs, often sharing parameters with spatial occupancy networks. For instance, Semantic-NeRF [16] embeds noisy and sparse semantic signals into its framework, enabling robust reconstruction of semantic information even under challenging conditions. NeRF-DFF [17] proposes distilling feature fields to achieve zero-shot semantic segmentation. However, these methods suffer from catastrophic forgetting and local detail loss, necessitating memory replay and hash position encoding [40]. 3DGSbased methods, such as Feature-3DGS [26], leverage explicit representations for stable feature binding, distilling teacher models (LSeg [41] and SAM [27]) to achieve up to 2.7Ã faster rendering. Nevertheless, applying offline methods [16, 17, 26] to online tasks remains challenging due to inaccurate pose estimation and insufficient semantic supervision signals. Our method leverage sparse multi-view information to achieve accurate pose estimation and online semantic reconstruction.

## 3 Method

Given an RGB-D video stream in unknown static environments, our goal is to reconstruct the 3D scene structure and semantic information while tracking the camera pose. Figure 2 outlines the general framework of GSFF-SLAM.

## 3.1 Scene Representation and Rendering

We utilize isotropic Gaussian points to represent the scene, where the number of 3D Gaussians $\mathcal { N }$ dynamically adapts during the optimization process. To implement the semantic reconstruction, we introduce a new trainable parameter, the semantic feature embedding $f \in \mathbb { R } ^ { \hat { N } }$ , for each Gaussian point. The optimizable properties of each Gaussian point are given by $\mathcal { G } _ { i } = \{ x _ { i } , q _ { i } , s _ { i } , \bar { \alpha _ { i } } , c _ { i } , f _ { i } \}$ , where $x \in \mathbb { R } ^ { 3 }$ denotes the mean $\mu$ (geometric center) and the 3D covariance matrix Î£ is expressed:

$$
\Sigma = R S S ^ { T } R ^ { T } ,\tag{1}
$$

where R is the rotation matrix, derived from a quaternion $q \in \mathbb { R } ^ { 4 }$ , and S is the scaling matrix, constructed from the scaling factor $s \in \mathbb { R } ^ { 3 }$ . Equation 1 ensures that the 3D covariance matrix is positive semi-definite and physically meaningful. The opacity value $\alpha \in$ R and the third-order spherical harmonics (SH) coefficients $c \in \mathbb { R } ^ { 3 }$ govern the color of the rendered image.

We use the rendering pipeline based on the differentiable Gaussian splatting framework proposed in [8] as the foundation, extending it to render depth images and dense feature maps. These projected 2D Gaussian points are sorted by depth in a front-to-back order and rendered in the camera view using Î±-blending:

$$
\mathbf { c } = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,
$$

$$
\mathbf { d } = \sum _ { i \in \mathcal { N } } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

$$
\mathbf { f } = \left\{ \sum _ { i \in \mathcal { N } } f _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , \begin{array} { l l } { \mathrm { i f ~ r e n d e r i n g ~ f e a t u r e s } , } \\ { 0 , } \end{array} \right.\tag{3}
$$

Specifically, during the gradient backward process, our pipeline computes the gradient of the feature f solely through the supervised semantic maps, without further propagating to the mean $\mu$ and variance Î£. To enhance rendering efficiency, we introduce a parameter that indicates whether to render the feature maps and backpropagate the gradient of the feature embedding f. This design enables the rendering pipeline to achieve high-speed rendering, even during camera pose tracking or scene reconstruction without semantic features.

## 3.2 Tracking and Mapping Optimization

We adopt the tracking strategy of most Visual SLAM methods, which estimates the relative pose changes between consecutive frames and accumulates them to determine the current frameâs pose. Additionally, we construct the map by selecting keyframes rather than using all frames, which improves efficiency.

Camera Pose Optimization. To avoid the overhead of automatic differentiation, we follow MonoGS [13] and compute the gradient of the camera viewpoint transformation matrix directly using Lie algebra, integrating it into the rendering pipeline as shown in Equation 4:

$$
\frac { d \mu _ { c } } { d T _ { C W } } = [ I , - \mu _ { c } ^ { \times } ] , \frac { d W } { d T _ { C W } } = [ \pmb { \theta } , - W ^ { \times } ] ,\tag{4}
$$

where $\times$ denotes the skew symmetric matrix. To ensure stable tracking accuracy, we check the convergence of $\Delta \mu _ { c }$ and âW to determine if the current frame has completed tracking.

Keyframe Select. By evaluating the co-visibility, which is the rendering overlap between the current frame and the previous keyframe, we determine whether to add a new keyframe. During rendering, the 3D Gaussian points are sorted based on their distance to the camera, and the transmittance $\begin{array} { r } { T _ { i } = \prod _ { i = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } \end{array}$ is calculated, defined as the product of the opacity values of the previous Gaussian points in front of the current point along the ray. Gaussian points with a transmittance greater than 0.5 are marked as visible. In detail, we record the visibility $\mathbf { V _ { i d } }$ of the rendered Gaussian points during tracking.

$$
\mathbf { V _ { i d } } = \left[ \begin{array} { l } { v _ { 1 } } \\ { v _ { 2 } } \\ { \vdots } \\ { v _ { N } } \end{array} \right] , v _ { i } \in \{ 0 , 1 \} , i = 1 , 2 , \dotsc , N ,\tag{5}
$$

where id denotes the frame index. We then calculate the intersection-over-union (IoU), and if the IoU is less than the threshold $\tau _ { \mathrm { t h r e s h } }$ , the current frame is set as a keyframe. As shown in Figure 2, we record the visibility of all keyframes and use the co-visibility to determine whether to add a new keyframe.

Tracking Loss. During the tracking process, we first freeze the Gaussian point parameters $\mathcal { G }$ and then generate the current frameâs RGB image and depth image through the rendering pipeline. We also record the 2D visibility map as the mask $m _ { v }$ . Next, we adjust the camera viewpoint $T _ { C W }$ parameters Î´, re-render the images, and iterate until convergence. For the RGB image, we compute the image edge gradient to extract edges and retain regions with significant gradients as $m _ { e } .$ . The tracking loss function is defined as:

$$
\mathcal { L } _ { \mathrm { t r a c k i n g } } = \lambda _ { t } m _ { v } m _ { e } \mathcal { L } _ { c } ( \delta ) + ( 1 - \lambda _ { t } ) m _ { v } \mathcal { L } _ { d } ( \delta ) ,\tag{6}
$$

where $\lambda _ { t }$ is the hyperparameter of tracking process, and $\mathcal { L } _ { c }$ and $\mathcal { L } _ { d }$ represent the color loss and depth loss with L1 loss, respectively.

Mapping Loss. For the selected keyframes, we use Open3D [42] to initialize the point cloud positions in unseen regions. By downsampling the point cloud density $\rho _ { \mathrm { p c } }$ , we control the sparsity of the point cloud to regulate the number of 3D Gaussian points. Then we freeze the camera viewpoint $T _ { C W }$ and optimize the Gaussian point parameters G. By minimizing the difference between the rendered images and the ground truth, we can progressively refine the geometric and color attributes of the Gaussian points. Additionally, to address the issues of Gaussian points being overly elongated in unobserved or growing regions and the uneven sparsity distribution, we introduce a regularization term:

$$
\mathcal { L } _ { \mathrm { r e g } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | s _ { i } - \bar { s } | _ { 1 } ,\tag{7}
$$

where sÂ¯ is the average scaling factor of all Gaussian points. The mapping loss function is defined as:

$$
\mathcal { L } _ { \mathrm { m a p p i n g } } = \lambda _ { m } \mathcal { L } _ { c } ( \mathcal { G } ) + ( 1 - \lambda _ { m } ) \mathcal { L } _ { d } ( \mathcal { G } ) + \lambda _ { r } \mathcal { L } _ { \mathrm { r e g } } ,\tag{8}
$$

where $\lambda _ { m }$ is the hyperparameter of mapping process, and $\lambda _ { r }$ regulates the strength of the regularization term.

## 3.3 Feature Field Optimization

Using the pipeline described in ??, we render the dense feature map $\hat { F } \in \mathbb R ^ { H \times W \times N }$ . Our goal is to bind the feature embedding $f$ of each Gaussian point to its 3D position through the Splatting process, whereas NeRF-based methods achieve this by sharing part of the network parameters.

Ground Truth Supervision. For the input supervision signal, the ground truth label $L \in \mathbb { N } ^ { H \times W }$ , we use the cross-entropy loss function to guide the learning of the dense feature map. The semantic loss is defined as:

$$
\mathcal { L } _ { s } = - \frac { 1 } { H W } \sum _ { i = 1 } ^ { H } \sum _ { j = 1 } ^ { W } \sum _ { c = 1 } ^ { N } L _ { i , j } ( c ) \log ( \hat { F } _ { i , j } ( c ) ) .\tag{9}
$$

During evaluation, we use the argmax function to convert $\hat { F }$ into the predicted label $\hat { L } \mathrm { : }$ :

$$
\hat { L } = \arg \operatorname* { m a x } ( \hat { F } ) ,\tag{10}
$$

Noisy Textual Label Supervision. To optimize the feature embedding $f \in \mathbb { R } ^ { N }$ , we minimize the difference between the rendered feature map $\hat { F } \in \mathbb { R } ^ { H \times W \times \hat { N } }$ and the predicted feature map $I ^ { f } \in \mathbb { R } ^ { H \times W \times M }$ . Inspired by the concept of distilled feature fields [17], we employ foundation models: Grounding-DINO [43] for open-vocabulary detection and SAM [44] for dense segmentation. Additionally, we use CLIP [45] to encode text queries into feature vectors, with M typically set to 512. Specifically, Grounding-DINO takes an RGB image I as input and generates a triplet output: bounding boxes $b \in \mathbb { R } ^ { K \times 4 }$ , open-vocabulary text labels $L = \{ l _ { 1 } , l _ { 2 } , . . . , \bar { l _ { k } } \}$ , and confidence scores $s \in \mathbb { R } ^ { K }$ . SAM then processes the bounding boxes b to produce dense binary segmentation masks $M \in \{ 0 , 1 \} ^ { H \times W }$ . The text encoder transforms the text labels into feature vectors, yielding $I ^ { f } \in \dot { \mathbb { R } ^ { H \times W \times M } }$ . The semantic supervision loss is defined as:

$$
\mathcal { L } _ { s } = | | I ^ { f } - o ( \hat { F } ) | | _ { 1 } ,\tag{11}
$$

where $o ( \cdot )$ denotes the convolutional upsampling operation. The probability of each pixel x in the rendered feature map belonging to a label l is computed as:

$$
p ( l | x ) = \frac { \exp { ( f ( x ) q ( l ) ^ { T } ) } } { \sum _ { l ^ { \prime } \in L } \exp { ( f ( x ) q ( l ^ { \prime } ) ^ { T } ) } } ,\tag{12}
$$

where $q ( l )$ is the query vector for label l generated by the text encoder.

The optimization of the feature field is performed after multiple iterations of optimizing the geometric and color attributes. This decoupling is motivated by two key reasons: first, simultaneous optimization significantly increases the computational burden on the optimizer; second, empirical results show that a well-constructed map effectively reduces the iteration count for feature field optimization, enhancing overall efficiency.

## 4 Experiments

## 4.1 Datasets, Metrics, and Baselines

Datasets. We evaluate our approach on three widely-used datasets. Replica [22] is a high-quality synthetic dataset of 3D indoor environments with detailed geometry and texture. Following previous works, we select 8 scenes and use provided camera poses for experiments. ScanNet [46] is a real-world dataset with RGB-D images and semantic annotations. We choose 5 representative scenes to evaluate the detail performance. TUM-RGBD [47] is another real-world dataset with RGB-D images and highly accurate camera poses. We use 5 scenes to evaluate the pose estimation performance.

Metrics. Our method focuses on two main tasks: SLAM and semantic reconstruction. For SLAM evaluation, We follow metrics from SNI-SLAM [20]. Specifically, we use ATE RMSE(cm) (Absolute Trajectory Error Root Mean Square Error) for tracking accuracy evaluation. For evaluating rendering quality, we employ PSNR(dB) (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and LPIPS (Learned Perceptual Image Patch Similarity). For semantic reconstruction evaluation, we utilize total pixel accuracy Acc(%) and mean class-wise intersection over union mIoU(%).

Baselines. We compare the semantic construction accuracy with state-of-the-art NeRF-based and 3DGS-based methods, including SNI-SLAM [20], DNS-SLAM [19], and SGS-SLAM [21]. We also compare the SLAM performance with the other baselines [9, 10, 11, 12, 13].

Implementation Details. We set the feature dimension N to 128 to represent semantic information. All experiments are conducted on a single NVIDIA RTX 4090 GPU. Please refer to the supplementary material for further details of our implementation.

## 4.2 Evaluation of SLAM Metrics

Tracking Accuracy. On the Replica dataset, we evaluate the tracking performance of our method compared to the state-of-the-art methods. As shown in Table 1, our method achieves the best tracking accuracy with an average ATE RMSE of 0.311cm. In the high-quality synthetic environments with accurate depth information and minimal motion blur, our method achieves submillimeter tracking accuracy that is comparable to other state-of-the-art 3DGS-based methods, while consistently outperforming NeRF-based methods. Compared to the NeRF-based SLAM, our method reduces the tracking error by more than 25% from 0.456cm to 0.311cm. Compared to the recent SplaTAM and MonoGS, which achieve average ATE RMSE of 0.384cm and 0.321cm respectively, our method reduces the tracking error by 19.0% and 3.1%. On real-world datasets, Table 2 and Table 3 present the quantitative comparison of tracking accuracy. Our method achieves performance with average ATE RMSE of 10.90cm on ScanNet and 3.95cm on TUM-RGBD. In the ScanNet dataset, we observe certain limitations of our approach. Specifically, in scene0000 and scene0181, the tracking performance exhibits significant degradation. For scene0000, which represents the longest sequence in our evaluation, we observe gradual drift in trajectory estimation after approximately 3000 frames, primarily due to the absence of loop closure detection. Meanwhile, scene0181 presents additional challenges due to significant motion blur, which adversely affects the mapping quality. These observations highlight potential areas for future improvement, particularly in handling extended sequences and scenes with degraded image quality. On the TUM-RGBD dataset, we conduct comprehensive experiments to further verify the robustness of our approach. Our method achieves competitive performance with an average ATE RMSE of 3.95cm, ranking among all learning-based methods. It is noteworthy that our algorithm performs poorly in the scene with motion blur, fr1/desk2, where the ATE RMSE of our method is 7.30 cm. In comparison, the NeRF-based SNI-SLAM [20] achieve ATE RMSE values of 4.35 cm, respectively.

Rendering Quality. We report the rendering performance in Table4 on the three datasets, our method achieves the highest PSNR and LPIPS scores in Replica dataset, indicating that our method can generate high-quality rendered images. As shown in Figure 3, we show the qualitative comparison of rendering quality between the baseline and our method on Replica dataset and our method achieves better texture details in the scene reconstruction. On realworld datasets, our method behaves robust in rendering quality, which is consistent with the quantitative results. While performance on real-world datasets lags behind synthetic data, this underscores the challenge of reconstructing high-quality scenes from noisy and motion-blurred sequences.

<table><tr><td>Method</td><td>00</td><td>01</td><td>02</td><td>03</td><td>04</td><td>RO</td><td>R1</td><td>R2</td><td>Avg.</td></tr><tr><td colspan="8">Neural Implicit Fields</td><td></td><td></td><td></td></tr><tr><td>NICE-SLAM [9]</td><td>0.88</td><td>1.00</td><td>1.06</td><td>1.10</td><td>1.13</td><td>0.97</td><td>1.31</td><td>1.07</td><td>1.061</td></tr><tr><td>Vox-Fusion [10]</td><td>8.48</td><td>2.04</td><td>2.58</td><td>1.11</td><td>2.94</td><td>1.37</td><td>4.70</td><td>1.47</td><td>3.086</td></tr><tr><td>DNS-SLAM [19]</td><td>0.84</td><td>0.84</td><td>0.69</td><td>1.11</td><td>0.66</td><td>0.72</td><td>1.08</td><td>0.76</td><td>0.838</td></tr><tr><td>SNI-SLAM [20]</td><td>0.41</td><td>0.38</td><td>0.48</td><td>0.71</td><td>0.52</td><td>0.40</td><td>0.38</td><td>0.35</td><td>0.456</td></tr><tr><td>Point-SLAM [11]</td><td>0.38</td><td>0.48</td><td>0.54</td><td>0.69</td><td>0.72</td><td>0.61</td><td>0.41</td><td>0.37</td><td>0.525</td></tr><tr><td colspan="10">3D Gaussian Splatting</td></tr><tr><td>SplaTAM [12]</td><td>0.47</td><td>0.27</td><td>0.29</td><td>0.32</td><td>0.72</td><td>0.31</td><td>0.40</td><td>0.29</td><td>0.384</td></tr><tr><td>MonoGS [13]</td><td>0.36</td><td>0.19</td><td>0.25</td><td>0.12</td><td>0.81</td><td>0.33</td><td>0.22</td><td>0.29</td><td>0.321</td></tr><tr><td>SGS-SLAM [21]</td><td>0.44</td><td>0.41</td><td>0.52</td><td>0.46</td><td>0.43</td><td>0.32</td><td>0.39</td><td>0.37</td><td>0.418</td></tr><tr><td>GSFF-SLAM(Ours)</td><td>0.35</td><td>0.24</td><td>0.25</td><td>0.15</td><td>0.54</td><td>0.34</td><td>0.29</td><td>0.33</td><td>0.311</td></tr></table>

Table 1: Comparison of tracking performance on Replica dataset [22]. We report ATE RMSE[cm]â for different sequences.

<table><tr><td>Method</td><td>0000</td><td>0059</td><td>0169</td><td>0181</td><td>0207</td><td>Avg.</td></tr><tr><td>NICE-SLAM [9]</td><td>12.0</td><td>14.0</td><td>10.9</td><td>13.4</td><td>6.2</td><td>11.30</td></tr><tr><td>Vox-Fusion [10]</td><td>16.6</td><td>24.2</td><td>27.3</td><td>23.3</td><td>9.4</td><td>20.14</td></tr><tr><td>Point-SLAM [11]</td><td>10.2</td><td>7.8</td><td>22.2</td><td>14.8</td><td>9.5</td><td>12.90</td></tr><tr><td>DNS-SLAM [19]</td><td>12.1</td><td>5.5</td><td>35.6</td><td>10.0</td><td>6.4</td><td>13.92</td></tr><tr><td>SNI-SLAM [20]</td><td>6.9</td><td>7.4</td><td></td><td></td><td>4.7</td><td></td></tr><tr><td>SplaTAM [12]</td><td>12.8</td><td>10.1</td><td>12.1</td><td>11.1</td><td>7.5</td><td>10.72</td></tr><tr><td>MonoGS [13]</td><td>9.8</td><td>6.4</td><td>10.7</td><td>23.8</td><td>8.1</td><td>11.76</td></tr><tr><td>SGS-SLAM [21]</td><td>12.6</td><td>14.0</td><td>17.6</td><td>13.0</td><td>8.0</td><td>13.04</td></tr><tr><td>LoopSplat* [48]</td><td>6.2</td><td>7.1</td><td>10.6</td><td>8.5</td><td>6.6</td><td>6.48</td></tr><tr><td>GSFF-SLAM(Ours)</td><td>9.6</td><td>7.8</td><td>9.6</td><td>21.9</td><td>5.6</td><td>10.90</td></tr></table>

Table 2: Comparison of tracking performance on ScanNet dataset [46](ATE RMSEâ[cm]). LoopSplat\* is a 3DGS-based method using loop closure.

## 4.3 Evaluation of Semantic Reconstruction

Ground Truth Supervision. As shown in Table5, our method outperforms the all baseline methods in segmentation metrics of Replica dataset by using the ground truth semantic labels. Specifically, our approach demonstrates superior semantic segmentation accuracy with an average pixel accuracy of 99.41% and mIoU of 95.03% across all sequences, significantly outperforming previous methods. Compared to the recent SNI-SLAM, our method improves the accuracy by 0.68% and notably boosts the mIoU by 10.41%. We also show the qualitative comparison of rendering quality between the baseline and our method in Figure 4. It can be observed that SNI-SLAM suffers from blurry boundaries, poor recognition of small objects, and loss of high-frequency details. SGS-SLAM exhibits some noise in the reconstructed local regions. In contrast, our method produces clearer semantic boundaries while preserving small object information, showing overall superior stability. These results further demonstrate that, due to the noise and uncertainty inherent in the reconstruction process, incorporating semantic supervision signals and semantic loss to improve tracking accuracy is not particularly crucial. The adoption of this separate gradient design makes our algorithm framework more flexible, and the semantic supervision learning process more stable. As shown in Table 6, we compare the performance of SNI-SLAM in semantic segmentation, inference rendering speed, and runtime. Our optimized semantic rendering pipeline can achieve a semantic rendering speed of up to 19.2 fps. When using $\tau _ { \mathrm { t h r e s h } } = 0 . 8$ and $\rho _ { \mathrm { p c } } = 1 / 6 4$ , our method with speedup still outperforms SNI-SLAM in semantic segmentation performance, but achieves a 2.9Ã improvement in runtime. In contrast, SGS-SLAM does not directly render features but instead reconstructs them through a label-to-RGB conversion strategy, achieving the highest rendering speed.

<table><tr><td>Method</td><td>fr1/ desk1</td><td>fr1/ desk2</td><td>fr1/ room</td><td>fr2/</td><td>fr3/ office</td><td>Avg.</td></tr><tr><td>NICE-SLAM [9]</td><td>4.26</td><td>4.99</td><td>34.49</td><td>xyz 6.19</td><td>3.87</td><td>10.76</td></tr><tr><td>Vox-Fusion [10]</td><td>3.52</td><td>6.00</td><td>19.53</td><td>1.49</td><td>26.01</td><td>11.31</td></tr><tr><td>Point-SLAM [11]</td><td>4.34</td><td>4.54</td><td>30.92</td><td>1.31</td><td>3.48</td><td>8.92</td></tr><tr><td>SNI-SLAM [20]</td><td>2.56</td><td>4.35</td><td>11.46</td><td>1.12</td><td>2.27</td><td>4.35</td></tr><tr><td>SplaTAM [12]</td><td>3.35</td><td>6.54</td><td>11.13</td><td>1.24</td><td>5.16</td><td>5.48</td></tr><tr><td>MonoGS [13]</td><td>1.59</td><td>7.03</td><td>8.55</td><td>1.44</td><td>1.49</td><td>4.02</td></tr><tr><td>LoopSplat* [48]</td><td>2.08</td><td>3.54</td><td>6.24</td><td>1.58</td><td>3.22</td><td>3.33</td></tr><tr><td>ORB-SLAM2* [5]</td><td>1.6</td><td>2.2</td><td>4.7</td><td>0.4</td><td>1.0</td><td>1.98</td></tr><tr><td>GSFF-SLAM(Ours)</td><td>1.61</td><td>7.30</td><td>7.29</td><td>1.61</td><td>1.96</td><td>3.95</td></tr></table>

Table 3: Comparison of tracking performance on TUM-RGBD dataset [47](ATE RMSEâ[cm]). ORB-SLAM2â is a feature-based SLAM method using loop closure.  
NICE-SLAM  
SNI-SLAM

SplaTAM  
Ours  
Ground Truth  
<!-- image-->  
Figure 3: Qualitative comparison on rendering quality of baseline and our method. We select 4 scenes of Replica dataset and highlighted the differences with red color boxes.

Noisy Textual Label Supervision. Figure 5 shows the visualization results using noisy textual priors. Compared to the Feature-3DGS [26] method, which utilizes LSeg [41] as the base model, our online rendering pipeline demonstrates that sparse priors with lower noise and accurate edges significantly enhance semantic reconstruction in detailed regions, as opposed to using noisy dense semantic priors. Furthermore, it is noteworthy that some unlabelled objects, represented in black, further validating the feasibility of incorporating the foundation model into our semantic reconstruction pipeline for downstream tasks. However, sparse detection may result in certain objects being contaminated by surrounding areas due to low detection rates, with the ceiling light and vent shown in the Figure 5 being a typical example.

<table><tr><td>Dataset</td><td colspan="3">Replica</td><td colspan="3">Scannet</td><td colspan="3">TUM-RGBD</td></tr><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>NICE-SLAM[9]</td><td>24.42</td><td>0.892</td><td>0.233</td><td>17.54</td><td>0.621</td><td>0.548</td><td>14.86</td><td>0.614</td><td>0.441</td></tr><tr><td>Vox-Fusion[10]</td><td>24.41</td><td>0.801</td><td>0.236</td><td>18.17</td><td>0.673</td><td>0.504</td><td>16.46</td><td>0.677</td><td>0.471</td></tr><tr><td>DNS-SLAM[19]</td><td>20.91</td><td>0.758</td><td>0.208</td><td>10.17</td><td>0.551</td><td>0.725</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Point-SLAM[11]</td><td>35.05</td><td>0.975</td><td>0.124</td><td>10.34</td><td>0.557</td><td>0.748</td><td>16.62</td><td>0.696</td><td>0.526</td></tr><tr><td>SplaTAM[12]</td><td>33.55</td><td>0.971</td><td>0.090</td><td>19.04</td><td>0.699</td><td>0.367</td><td>22.25</td><td>0.878</td><td>0.181</td></tr><tr><td>MonoGS[13]</td><td>35.77</td><td>0.951</td><td>0.067</td><td>17.11</td><td>0.734</td><td>0.581</td><td>18.60</td><td>0.713</td><td>0.334</td></tr><tr><td>SGS-SLAM[21]</td><td>33.53</td><td>0.966</td><td>0.090</td><td>18.07</td><td>0.695</td><td>0.417</td><td>-</td><td></td><td>-</td></tr><tr><td>Ours</td><td>38.67</td><td>0.974</td><td>0.035</td><td>18.90</td><td>0.697</td><td>0.391</td><td>20.57</td><td>0.736</td><td>0.311</td></tr></table>

Table 4: Quantitative comparison of rendering performance on the Replica [22], ScanNet [46], and TUM-RGBD [47] datasets. The best results are highlighted as first , second , and third .

<!-- image-->  
Figure 4: Qualitative comparison of semantic reconstruction performance using ground truth labels on the Replica dataset [22].

## 4.4 Ablation Study

We conducted an ablation study to examine the effects of two key hyperparameters: point cloud downsampling rate $\rho _ { \mathrm { p c } }$ and co-visibility overlap threshold $\tau _ { \mathrm { t h r e s h } }$ on system performance, as shown in Table 7. Our analysis reveals that $\rho _ { \mathrm { p c } }$ plays a crucial role in determining tracking accuracy. This suggests that denser point clouds provide more detailed information, which is beneficial for tracking accuracy. On the other hand, $\tau _ { \mathrm { t h r e s h } }$ significantly affects semantic reconstruction performance. A higher threshold improves the mIoU by allowing for better alignment of semantic features, which leads to more accurate semantic reconstruction. This is evident in the marked improvement in mIoU when increasing $\tau _ { \mathrm { t h r e s h } }$ from 0.8 to 0.95. both hyperparameters have minimal impact on rendering quality, but they do affect runtime. From the results, we observe that the hyperparameter settings in the first row $( \rho _ { \mathrm { p c } } = 1 / 6 4 , \tau _ { \mathrm { t h r e s h } } = 0 . 8 )$ strike a balance between memory usage, runtime, and accuracy. These settings minimize memory requirements while maintaining competitive tracking performance and rendering quality, making them the optimal choice for resource-constrained deployment scenarios.

<!-- image-->  
Figure 5: Qualitative comparison of semantic reconstruction performance using noisy textual labels on the Replica dataset [22]. We merge semantically similar objects with high 3D spatial overlap, such as windows and blinds, rugs and floors.

<table><tr><td>Method</td><td>Metric</td><td>00</td><td>01</td><td>02</td><td>03</td><td>04</td><td>R0</td><td>R1</td><td>R2</td><td>Avg.</td></tr><tr><td>NIDS- [18]</td><td>Accâ</td><td>98.89</td><td>-</td><td></td><td>-</td><td>-</td><td>97.76</td><td>98.50</td><td>98.76</td><td>98.47</td></tr><tr><td rowspan="2">SLAM DNS-</td><td>mIoUâ</td><td>85.94</td><td></td><td></td><td></td><td></td><td>82.45</td><td>84.08</td><td>76.99</td><td>82.37</td></tr><tr><td>Accâ</td><td>98.27</td><td>97.79</td><td>98.36</td><td>97.20</td><td>90.59</td><td>97.85</td><td>97.62</td><td>98.27</td><td>96.81</td></tr><tr><td>[19] SLAM</td><td>mIoUâ</td><td>84.10</td><td>82.75</td><td>73.33</td><td>74.13</td><td>64.74</td><td>85.81</td><td>85.80</td><td>77.86</td><td>78.56</td></tr><tr><td rowspan="2">SNI- [20] SLAM</td><td>Accâ</td><td>98.73</td><td>98.98</td><td>98.93</td><td>98.58</td><td>99.00</td><td>98.32</td><td>98.63</td><td>98.73</td><td>98.73</td></tr><tr><td>mIoUâ</td><td>87.66</td><td>85.13</td><td>84.00</td><td>79.39</td><td>78.11</td><td>87.22</td><td>88.11</td><td>84.62</td><td>84.62</td></tr><tr><td rowspan="2">SGS- [21] SLAM</td><td>Accâ</td><td>99.52</td><td>99.59</td><td>99.18</td><td>98.29</td><td>98.77</td><td>99.12</td><td>99.36</td><td>99.36</td><td>99.14</td></tr><tr><td>mIoUâ</td><td>92.29</td><td>95.25</td><td>91.62</td><td>87.25</td><td>90.34</td><td>92.47</td><td>92.43</td><td>92.83</td><td>91.81</td></tr><tr><td rowspan="2">Ours</td><td>Accâ</td><td>99.53</td><td>99.46</td><td>99.55</td><td>99.50</td><td>99.11</td><td>99.16</td><td>99.55</td><td>99.56</td><td>99.41</td></tr><tr><td>mIoUâ</td><td>95.12</td><td>93.44</td><td>94.96</td><td>94.74</td><td>93.97</td><td>95.20</td><td>96.90</td><td>95.98</td><td>95.03</td></tr></table>

Table 5: Quantitative comparison of GSFF-SLAM with existing semantic SLAM methods for semantic segmentation metrics Acc(%) and mIou(%) on the Replica dataset [22].

## 5 Conclusion

We propose GSFF-SLAM, a dense semantic SLAM system that employs 3D Gaussian Splatting as the map representation and incorporates semantic features for efficient semantic map rendering. We leverage the overlap between consecutive frames to select keyframes and independently optimize the gradients of features. Extensive experiments demonstrate the robustness of our method in tracking and mapping, as well as the accuracy of semantic edges. We verify the feasibility of semantic reconstruction using noisy sparse signals, opening up a pathway for online semantic reconstruction in previously unseen scenes. Future work will focus on extending GSFF-SLAM to dynamic environments and further improving pipeline efficiency.

<table><tr><td>Method</td><td>mIoU(%)â</td><td>Inference(fps)â</td><td>Time(min)â</td></tr><tr><td>SNI-SLAM [20]</td><td>84.62</td><td>0.87</td><td>132</td></tr><tr><td>SGS-SLAM [21]</td><td>91.81</td><td>392</td><td>234</td></tr><tr><td>Ours (w/ speedup)</td><td>90.54</td><td>19.2</td><td>45</td></tr><tr><td>Ours</td><td>95.03</td><td>15.8</td><td>114</td></tr></table>

Table 6: Performance of semantic SLAM on Replica dataset [22] compared to SNI-SLAM [20] and SGS-SLAM [21].

<table><tr><td> $\rho _ { \mathrm { p c } }$ </td><td>Tthresh</td><td>RMSE(cm)â</td><td>PSNR(dB)â</td><td>mIoU(%)â</td><td>Mem(MB)â</td><td>Time(min)â</td></tr><tr><td>1/64</td><td>0.8</td><td>0.582</td><td>36.41</td><td>90.54</td><td>5281</td><td>45</td></tr><tr><td>1/64</td><td>0.95</td><td>0.495</td><td>37.78</td><td>93.68</td><td>9088</td><td>63</td></tr><tr><td>1/16</td><td>0.8</td><td>0.396</td><td>37.64</td><td>92.74</td><td>8360</td><td>64</td></tr><tr><td>1/16</td><td>0.95</td><td>0.390</td><td>38.67</td><td>95.03</td><td>12062</td><td>114</td></tr></table>

Table 7: Ablation study on Replica dataset using ground truth. We studied the effects of co-visibility overlap threshold Ïthresh and point cloud downsampling rate $\rho _ { \mathrm { p c } }$ on the system.

## References

[1] Andrew J Davison, Ian D Reid, Nicholas D Molton, and Olivier Stasse. Monoslam: Real-time single camera slam. IEEE transactions on pattern analysis and machine intelligence, 29(6):1052â1067, 2007.

[2] Jorge Fuentes-Pacheco, JosÃ© Ruiz-Ascencio, and Juan Manuel RendÃ³n-Mancha. Visual simultaneous localization and mapping: a survey. Artificial intelligence review, 43:55â81, 2015.

[3] Georg Klein and David Murray. Parallel tracking and mapping for small ar workspaces. In 2007 6th IEEE and ACM international symposium on mixed and augmented reality, pages 225â234. IEEE, 2007.

[4] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: a versatile and accurate monocular slam system. IEEE transactions on robotics, 31(5):1147â1163, 2015.

[5] Raul Mur-Artal and Juan D TardÃ³s. Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras. IEEE transactions on robotics, 33(5):1255â1262, 2017.

[6] Richard A Newcombe, Steven J Lovegrove, and Andrew J Davison. Dtam: Dense tracking and mapping in real-time. In 2011 international conference on computer vision, pages 2320â2327. IEEE, 2011.

[7] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

[8] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

[9] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12786â12796, 2022.

[10] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pages 499â507. IEEE, 2022.

[11] Erik SandstrÃ¶m, Yue Li, Luc Van Gool, and Martin R Oswald. Point-slam: Dense neural point cloud-based slam. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18433â18444, 2023.

[12] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21357â21366, 2024.

[13] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18039â18048, 2024.

[14] Johannes Lutz SchÃ¶nberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[15] John McCormac, Ankur Handa, Andrew Davison, and Stefan Leutenegger. Semanticfusion: Dense 3d semantic mapping with convolutional neural networks. In 2017 IEEE International Conference on Robotics and automation (ICRA), pages 4628â4635. IEEE, 2017.

[16] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J Davison. In-place scene labelling and understanding with implicit scene representation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15838â15847, 2021.

[17] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via feature field distillation. Advances in Neural Information Processing Systems, 35:23311â23330, 2022.

[18] Yasaman Haghighi, Suryansh Kumar, Jean-Philippe Thiran, and Luc Van Gool. Neural implicit dense semantic slam. arXiv preprint arXiv:2304.14560, 2023.

[19] Kunyi Li, Michael Niemeyer, Nassir Navab, and Federico Tombari. Dns-slam: Dense neural semantic-informed slam. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7839â7846. IEEE, 2024.

[20] Siting Zhu, Guangming Wang, Hermann Blum, Jiuming Liu, Liang Song, Marc Pollefeys, and Hesheng Wang. Sni-slam: Semantic neural implicit slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21167â21177, 2024.

[21] Mingrui Li, Shuhong Liu, Heng Zhou, Guohao Zhu, Na Cheng, Tianchen Deng, and Hongyu Wang. Sgs-slam: Semantic gaussian splatting for neural dense slam. In European Conference on Computer Vision, pages 163â179. Springer, 2024.

[22] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019.

[23] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Oswald. Gaussian-slam: Photo-realistic dense slam with gaussian splatting. arXiv preprint arXiv:2312.10070, 2023.

[24] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21476â21485, 2024.

[25] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for openvocabulary scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5333â5343, 2024.

[26] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676â21685, 2024.

[27] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr DollÃ¡r, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023.

[28] Georg Klein and David Murray. Parallel tracking and mapping on a camera phone. In 2009 8th IEEE International Symposium on Mixed and Augmented Reality, pages 83â86. IEEE, 2009.

[29] Carlos Campos, Richard Elvira, Juan J GÃ³mez RodrÃ­guez, JosÃ© MM Montiel, and Juan D TardÃ³s. Orb-slam3: An accurate open-source library for visual, visualâinertial, and multimap slam. IEEE Transactions on Robotics, 37(6):1874â1890, 2021.

[30] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davison. imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6229â6238, 2021.

[31] Jan Czarnowski, Tristan Laidlow, Ronald Clark, and Andrew J Davison. Deepfactors: Real-time probabilistic dense monocular slam. IEEE Robotics and Automation Letters, 5(2):721â728, 2020.

[32] Yan-Pei Cao, Leif Kobbelt, and Shi-Min Hu. Real-time high-accuracy three-dimensional reconstruction with consumer rgb-d cameras. ACM Transactions on Graphics (TOG), 37(5):1â16, 2018.

[33] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558â16569, 2021.

[34] Qian-Yi Zhou, Stephen Miller, and Vladlen Koltun. Elastic fragments for dense scene reconstruction. In Proceedings of the IEEE International Conference on Computer Vision, pages 473â480, 2013.

[35] Angela Dai, Matthias NieÃner, Michael ZollhÃ¶fer, Shahram Izadi, and Christian Theobalt. Bundlefusion: Realtime globally consistent 3d reconstruction using on-the-fly surface reintegration. ACM Transactions on Graphics (ToG), 36(4):1, 2017.

[36] Manasi Muglikar, Zichao Zhang, and Davide Scaramuzza. Voxel map for visual slam. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 4181â4187. IEEE, 2020.

[37] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 10608â10615. IEEE, 2023.

[38] William Shen, Ge Yang, Alan Yu, Jansen Wong, Leslie Pack Kaelbling, and Phillip Isola. Distilled feature fields enable few-shot language-guided manipulation. arXiv preprint arXiv:2308.07931, 2023.

[39] Meng Wei, Tai Wang, Yilun Chen, Hanqing Wang, Jiangmiao Pang, and Xihui Liu. Ovexp: Open vocabulary exploration for object-oriented navigation. arXiv preprint arXiv:2407.09016, 2024.

[40] Thomas MÃ¼ller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022.

[41] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and RenÃ© Ranftl. Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546, 2022.

[42] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3d: A modern library for 3d data processing. arXiv preprint arXiv:1801.09847, 2018.

[43] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.

[44] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman RÃ¤dle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr DollÃ¡r, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

[45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748â8763. PMLR, 2021.

[46] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828â5839, 2017.

[47] JÃ¼rgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 573â580. IEEE, 2012.

[48] Liyuan Zhu, Yue Li, Erik SandstrÃ¶m, Shengyu Huang, Konrad Schindler, and Iro Armeni. Loopsplat: Loop closure by registering 3d gaussian splats. In International Conference on 3D Vision (3DV), 2025.