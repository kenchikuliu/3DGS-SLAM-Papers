# GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localizationâ 

Gennady Sidorov 1,2, Malik Mohrat 1,2, Denis Gridusov1, Ruslan Rakhimov 3, and Sergey Kolyubin1

Abstractâ Although various visual localization approaches exist, such as scene coordinate regression and camera pose regression, these methods often struggle with optimization complexity or limited accuracy. To address these challenges, we explore the use of novel view synthesis techniques, particularly 3D Gaussian Splatting (3DGS), which enables the compact encoding of both 3D geometry and scene appearance. We propose a two-stage procedure that integrates dense and robust keypoint descriptors from the lightweight XFeat feature extractor into 3DGS, enhancing performance in both indoor and outdoor environments. The coarse pose estimates are directly obtained via 2D-3D correspondences between the 3DGS representation and query image descriptors. In the second stage, the initial pose estimate is refined by minimizing the rendering-based photometric warp loss. Benchmarking on widely used indoor and outdoor datasets demonstrates improvements over recent neural rendering-based localization methods, such as NeRFMatch and PNeRFLoc. Project page: https://gsplatloc.github.io

## I. INTRODUCTION

Visual localization is a key task in computer vision that estimates the pose of a moving camera relative to a prebuilt environment map. It is a crucial function for mobile and humanoid robots navigating indoor and outdoor environments, enabling autonomous navigation and interaction within the environment. Moreover, this capability is required for robots to perceive their position in the 3D environment, which constitutes one of the core components of SLAM systems. Among various localization approaches, visionbased methods have received significant attention due to the widespread availability and low cost of cameras. Equipped with vision sensors, these methods can be implemented with minimal modification, making them highly adaptable for real-world robotic applications [1], [2].

Early methods for visual re-localization primarily relied on image retrieval techniques, where a query image was compared to a database of images with known poses to infer an approximate location. Although computationally efficient, these methods face scalability challenges and exhibit reduced accuracy in dynamic environments.

Structured feature-matching approaches enhance localization by leveraging 3D point clouds generated through Structure-from-Motion (SfM). These methods establish explicit 2D-3D correspondences between features extracted from query images and the 3D map, typically using Perspective-n-Point (PnP) solvers with RANSAC. Although these approaches offer high localization accuracy, they impose significant memory and computational requirements when dealing with large-scale environments [3].

<!-- image-->  
Fig. 1: GSplatLoc constructs a 3D Gaussian Splatting (3DGS) model with distilled descriptor features. For localization, the initial coarse pose is estimated through structural matching with these features and refined during test-time optimization using rendering-based photometric warp loss to enhance accuracy.

Pose regression methods employ deep neural networks to estimate camera poses directly from input images, reducing dependence on large-scale 3D maps. These methods integrate the environmental structure into the networkâs architecture, enabling end-to-end training. However, they often underperform compared to structure-based approaches due to limited generalization capabilities [4]. Absolute Pose Regression (APR) techniques provide a trade-off between computational efficiency and accuracy, while Scene Coordinate Regression (SCR) further improves performance by learning compact scene representations. However, SCR methods demand extensive optimization to achieve state-of-the-art accuracy, which constrains their applicability in real-time scenarios.

Recent advancements in neural 3D scene representations, such as Neural Radiance Fields (NeRF) [5] and 3D Gaussian Splatting (3DGS) [6], have enabled high-fidelity view synthesis. These methods enhance visual localization and contribute to Neural Render Pose (NRP) estimation by generating novel perspectives, enabling data augmentation during training and improving robust feature matching. NeRF-based methods, while effective, often suffer from slow inference speeds, long training time, and artifacts in rendered images. Enhancements such as incorporating feature fields into NeRF representations have improved robustness against such artifacts and enhanced pose estimation accuracy [7], [8], [9], [10]. On the other hand, 3DGS-based pipelines, such as GS-CPR [11] and HGSLoc [12] leverage this 3D representation as a test-time pose refinement framework, yet they depend on external pose estimators for initialization.

To address these limitations, we propose GSplatLoc, a novel visual localization framework that integrates structurebased coarse pose estimation with photometric renderingbased optimization in a unified, end-to-end pipeline. Our approach leverages the point-based physically-consistent 3D Gaussian representation to distill scene-agnostic feature descriptors, enabling efficient initial pose estimation and subsequent refinement. (see Figure 1).

Our main contributions are:

â¢ A novel visual localization framework that combines structure-based keypoint matching with rendering-based pose refinement in a unified 3D Gaussian Splatting (3DGS) pipeline.

â¢ An efficient test-time pose refinement strategy that leverages fast differentiable rendering and photometric warping loss minimization to improve localization accuracy, particularly in outdoor and dynamic scenes.

â¢ A demonstration of state-of-the-art performance on indoor and outdoor benchmarks, outperforming other NRP approaches based on NeRF, while maintaining lower computational overhead and requiring a single RGB modality.

## II. RELATED WORK

Absolute Pose Regression (APR) methods directly regress camera poses from query images using deep neural networks [13], [14], [15], [16], [17], [18], [19], [20]. PoseNet [13] introduced this approach by employing a pre-trained GoogLeNet for feature extraction. Subsequent studies enhanced APR by incorporating additional modules: MapNet [15] jointly estimates both absolute and relative poses, while AtLoc [16] leverages self-attention to extract salient features. Recently, MaRePo [21] introduced a twostage approach, first regressing scene-specific geometry and then refining the pose with a transformer. Although APR methods are computationally efficient, their accuracy remains inferior to that of structure-based approaches.

Scene Coordinate Regression (SCR) methods predict dense 2D-3D correspondences by regressing the 3D scene coordinates directly from images [22], [23], [24], [25], [26]. Early SCR approaches relied on random forests [22], whereas more recent methods leverage deep learning. DSAC [23] introduced differentiable RANSAC for end-toend training, later extended by DSAC++ [24] and ESAC [27], which utilize a gating network to divide the problem into simpler sub-tasks. ACE [28] and GLACE [29] improve efficiency by avoiding end-to-end supervision and leveraging shuffled pixel-based training. Although SCR methods achieve higher accuracy, they necessitate high-quality 3D models and extensive training data.

Neural Render Pose (NRP) estimation utilizes neural rendering techniques such as NeRF [5] and 3D Gaussian Splatting (3DGS) [6] to improve localization by synthesizing novel views and refining camera poses. These methods refine pose estimates by minimizing the photometric or featurebased discrepancies between observed and rendered images. iNeRF [30] employs inverse rendering to iteratively refine poses, while LENS [31] generates synthetic training data using NeRF-W [32]. DFNet [19] enhances pose estimation through direct feature matching between query and rendered images, and PNeRFLoc [33] aligns 2D-3D correspondences via NeRF-based feature warping. Recent approaches [8], [7], [34] integrate learned feature fields to enhance localization; however, they incur high computational costs due to extended training and rendering times. In contrast, our method exploits the efficiency of 3DGS for rapid rendering and distills feature fields to refine camera pose estimates using only RGB data.

Keypoint detection and descriptor learning have evolved through deep learning, enhancing feature matching for localization. CNN-based methods, including Super-Point [35], D2-Net [36], and R2D2 [37], improve keypoint extraction and description. Transformer-based models like LoFTR [38] and LightGlue [39] further refine feature matching by capturing long-range dependencies. Recent methods such as XFeat [40] offer lightweight feature extraction for improved efficiency. Feature matching-based localization methods [41], [10], [19] employ deep descriptors to enhance pose refinement robustness. Our method integrates keypointbased descriptors with 3DGS feature fields to improve pose estimation accuracy.

## III. PRELIMINARIES

Our method integrates keypoint descriptor models with 3D Gaussian Splatting, embedding keypoint features into the 3D representation to enhance re-localization accuracy. We provide a concise overview of this process.

Building on the Feature-3DGS framework [9], we employ a modified 3DGS to distill high-dimensional features into a feature field while simultaneously constructing a radiance field. This approach utilizes N-dimensional parallel Gaussian rasterization to accelerate computation, facilitating seamless integration with various 2D foundation models.

We initialize the 3D Gaussians using a point cloud obtained through Structure-from-Motion (SfM) reconstruction. Their projection into 2D space involves transforming covariance matrices and incorporating rotation, scaling, opacity, spherical harmonics, color, and other visual features.

Pixel colors and feature values are computed via Î±- blending under the supervision of a teacher model that guides the feature distillation process. The joint optimization method rasterizes both RGB images and feature maps simultaneously, ensuring high fidelity and per-pixel accuracy. The optimizable attributes of the i-th 3D Gaussian Î¨i are:

<!-- image-->  
Fig. 2: Overview of the GSplatLoc Base pipeline. First, we model the scene using a feature-based 3D Gaussian Splatting (3DGS) approach, leveraging the XFeat [40] network for feature extraction and distillation. In the test stage, the initial coarse pose is estimated by matching 2D keypoints from the query image to 3D features in the 3DGS model, which is then refined using a Perspective-n-Point (PnP) solver within a RANSAC loop. We then refine the coarse pose by aligning the image rendered from 3DGS with the input query image using an RGB warping loss. This process enhances pose accuracy via test-time optimization.

$$
\Psi _ { i } = \left\{ y _ { i } , q _ { i } , s _ { i } , \alpha _ { i } , c _ { i } , f _ { i } \right\} ,\tag{1}
$$

where $y _ { i } \in \mathbb { R } ^ { 3 }$ represents the 3D position, $q _ { i } \in \mathbb { R } ^ { 4 }$ denotes the rotation quaternion, $s _ { i } \in \mathbb { R }$ is the scaling factor, $\alpha _ { i } \in \mathbb { R }$ is the opacity value, $c _ { i } \in \mathbb { R } ^ { 3 }$ represents the diffuse color from Spherical Harmonics (SH), and $f _ { i } \in \mathbb { R } ^ { V }$ is the feature embedding from the supervised model $F _ { t } ,$ where V denotes the dimension of the feature vector. Each Gaussian $\Psi _ { i }$ is positioned at $y _ { i }$ , and is associated with a feature vector $f _ { i }$ that encodes local spatial and visual content.

The following equations define the computation of pixel color C and pixel feature $F _ { r }$ during rendering:

$$
C = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } T _ { i } , \quad F _ { r } = \sum _ { i \in \mathcal { N } } f _ { i } \alpha _ { i } T _ { i } .\tag{2}
$$

where $\mathcal { N }$ denotes the set of overlapping 3D Gaussians for a specified pixel, and $T _ { i }$ represents the transmittance, defined as the cumulative opacity of preceding Gaussians overlapping the given pixel.

To train the 3DGS model on a specific scene with grounded feature maps, we define the loss function ${ \mathcal { L } } _ { \mathrm { G S } }$ as follows:

$$
\mathcal { L } _ { \mathrm { G S } } = \mathcal { L } _ { \mathrm { c o l o r } } + \mathcal { L } _ { \mathrm { f e a t u r e s } }\tag{3}
$$

The photometric loss $\scriptstyle { \mathcal { L } } _ { \mathrm { c o l o r } }$ measures the difference between the ground truth image I and the rendered image $\hat { I } { : }$ $\mathcal { L } _ { \mathrm { c o l o r } } = ( \bar { 1 } - \lambda ) \mathcal { L } _ { 1 } ( I , \hat { I } ) \bar { + } \lambda \mathcal { L } _ { \mathrm { S S I M } } ( I , \hat { I } )$ . The feature loss ${ \mathcal { L } } _ { \mathrm { f e a t u r e s } }$ enforces consistency between the supervised feature map $F _ { t } ( I )$ and the rendered feature map $F _ { r } \colon \ \mathcal { L } _ { \mathrm { f e a t u r e s } } \ =$ $\lVert \boldsymbol { F } _ { t } ( \boldsymbol { I } ) - \boldsymbol { F } _ { r } \rVert _ { 1 }$

## IV. METHODOLOGY

Our method consists of a two-stage pipeline, as depicted in Figure 2. The first stage involves modeling the scene, and the second stage focuses on estimating an initial coarse pose, followed by its refinement to improve accuracy.

Initially, we model the scene using a feature-based 3D Gaussian Splatting (3DGS) approach [9], guided by a keypoint descriptor network. We use the deep feature extractor XFeat [40] due to its robustness in extracting reliable and distinctive features across various environments, for both indoor and outdoor, even in the presence of dynamic elements.

For each training image $\boldsymbol { I _ { t } } \in \mathbb { R } ^ { W \times H \times 3 }$ , the XFeat network computes a feature map $F _ { t } ( I ) \in \mathbb { R } ^ { ( W / 8 ) \times ( H / 8 ) \times 6 4 }$ , which is bilinearly upsampled to the original resolution. We train the 3D Gaussian Splatting model by minimizing the loss function as described in Equation 3.

Once the scene is learned by 3DGS, we estimate the pose for the query image through a two-phase process: first, estimating a coarse pose, and then refining it.

Obtaining the Initial Coarse Pose. This stage aims to establish correspondences between 2D keypoints in the query image and the 3D points in the 3DGS model of the scene. We use the Perspective-n-Point (PnP) solver within a RANSAC loop to provide an initial pose estimate.

For a given query image q accompanied by extracted keypoints $P _ { q }$ and features $f _ { q }$ from a descriptor model, we perform 2D-3D correspondence matching with the 3D Gaussian point cloud $P \in \bar { \mathbb { R } } ^ { N \times 3 }$ and the associated distilled XFeat features $f _ { p } \in \mathbb { R } ^ { N \times 6 4 }$ , where N denotes the number of points. We employ cosine similarity to match the query image 2D features with the 3D scene features distilled in the 3D Gaussians. The 2D-3D correspondences $V ( i )$ , which represent the matching for the i-th pixel, are determined by maximizing the cosine similarity measure:

$$
V ( i ) = \arg \operatorname* { m a x } _ { i \in P } { \frac { \mathbf { f } _ { q } ^ { i } \cdot \mathbf { f } _ { p } ^ { l } } { \| \mathbf { f } _ { q } ^ { i } \| \| \mathbf { f } _ { p } ^ { l } \| } }\tag{4}
$$

Some methods speed up the search by learning a reliability score for each point. Instead, we use sparse and reliable keypoints from the query image via XFeat and match them with all points in the 3D point cloud, enabling efficient pose estimation through semi-dense matching with the Gaussian cloud of distilled features.

<!-- image-->  
Fig. 3: Test-time camera pose refinement aligns the rendered images to the query image at different optimization iterations. The first row shows the rendered images blended with the query image based on the estimated pose at each step, while the second row visualizes the absolute errors between the two, demonstrating how the warping loss reduces this error over time, thereby improving pose accuracy.

Test-time Camera Pose Refinement. We enhance the coarse pose estimate in two phases: optional feature-based pose refinement and warping-based pose refinement. These methods progressively improve the pose accuracy by minimizing the error between the query and rendered images.

Feature-based Pose Refinement. The refinement process begins by rendering a feature map based on the coarse pose estimate. In this step, we match keypoints between the query image and the rendered feature map to iteratively refine the pose. Descriptors from the query image are matched with those in the rendered feature map, enabling the identification of corresponding keypoints. These matched keypoints are backprojected into world space using the rendered depth map and the initially estimated pose. This process allows for re-estimation of the pose via the Perspective-n-Point (PnP) algorithm.

This iterative refinement improves pose alignment, with each iteration reducing the error between the query image and the rendered scene. It is particularly effective in complex environments, where even small initial errors can lead to significant misalignments. By refining the pose in this manner, we enhance both the accuracy and robustness of pose estimation, providing more precise results under challenging conditions. However, this step introduces additional computational overhead. Thus, we treat it as optional, depending on the desired trade-off between computation and accuracy. An analysis of this trade-off is provided in the experimental section.

Warping-based Pose Refinement. In contrast to the feature-based approach, we refine the pose photometrically by aligning the rendered image with the query image through a warping loss. This method is computationally more efficient, requiring only one render pass, compared to iterative rendering in feature-based refinement.

Previous works [30], [42] employed gradient descent to minimize photometric residuals, which quantify the difference between the rendered and query images. This process requires neural rendering at each step, making it computationally expensive. PNeRFLoc [33] improved this process by introducing a warping loss function. We adopt this warping loss method, which significantly reduces computational cost by rendering the image only once, compared to the repeated rendering required in previous methods. Additionally, the 3DGS framework enhances speed with its faster rendering process compared to NeRF-based methods.

For a query image q and a coarse pose (R, t), we first render the image $q _ { r }$ and the depth map $d _ { r }$ using the initial pose. We optimize the pose estimate $( \mathbf { R } ^ { \prime } , \mathbf { t } ^ { \prime } )$ by minimizing a warping loss, which is defined as the sum of pixel-wise RGB differences between the reference and query images:

$$
\begin{array} { r } { \mathcal { L } _ { r g b - w a r p } = \sum _ { p _ { i } } \lVert Y ( q , W ( p _ { i } , \mathbf { R } , \mathbf { t } , \mathbf { R } ^ { \prime } , \mathbf { t } ^ { \prime } ) ) - Y ( q _ { r } , p _ { i } ) \rVert _ { 2 } , ( 5 ) } \end{array}
$$

where

$$
\begin{array} { r } { W ( p _ { i } , \mathbf { R } , \mathbf { t } , \mathbf { R } ^ { \prime } , \mathbf { t } ^ { \prime } ) = \prod ( \mathbf { R } ( \mathbf { R } ^ { \prime } ^ { - 1 } \prod ^ { - 1 } ( p _ { i } , z _ { r } ( p _ { i } ) ) - \mathbf { R } ^ { \prime - 1 } \mathbf { t } ^ { \prime } ) + \mathbf { t } ) . } \end{array}
$$

Here, $Y ( q _ { r } , p _ { i } ) \in \mathbb { R } ^ { 3 }$ is the RGB color at pixel $p _ { i } \in \mathbb { R } ^ { 2 }$ on the rendered image $q _ { r }$ . The warp function $W ( \cdot )$ finds the corresponding pixel on the query image q by warping $p _ { i }$ from the reference image $q _ { r }$ . More precisely, the function W back-projects $p _ { i }$ into the 3D space of $q _ { r } \mathbf { \dot { s } }$ coordinate system by utilizing the rendered depth $z _ { r } ,$ transforms it into the camera coordinate system of q using the optimized pose $( \mathbf { R } ^ { \prime } , \mathbf { t } ^ { \prime } )$ , and then projects it onto the image q.

TABLE I: Comparison of methods on the 7Scenes dataset: median translation and rotation errors (cm/Â°) across various approaches. APR denotes absolute pose regression, SCR represents scene coordinate regression, and NRP stands for neural render pose estimation. The best results are highlighted as first and second .
<table><tr><td></td><td>Methods</td><td>Chess</td><td>Fire</td><td>Heads</td><td>Office</td><td>Pumpkin</td><td>Redkitchen</td><td>Stairs</td><td>Avg. â [cm/]</td></tr><tr><td rowspan="5">APR</td><td>PoseNet [13]</td><td>10/4.02</td><td>27/10.0</td><td>18/13.0</td><td>17/5.97</td><td>19/4.67</td><td>22/5.91</td><td>35/10.5</td><td>21/7.74</td></tr><tr><td>MS-Transformer [43]</td><td>11/6.38</td><td>23/11.5</td><td>13/13.0</td><td>18/8.14</td><td>17/8.42</td><td>16/8.92</td><td>29/10.3</td><td>18/9.51</td></tr><tr><td>DFNet [19]</td><td>3/1.12</td><td>6/2.30</td><td>4/2.29</td><td>6/1.54</td><td>7/1.92</td><td>7/1.74</td><td>12/2.63</td><td>6/1.93</td></tr><tr><td>Marepo [21]</td><td>1.9/0.83</td><td>2.3/0.92</td><td>2.1/1.24</td><td>2.9/0.93</td><td>2.5/0.88</td><td>2.9/0.98</td><td>5.9/1.48</td><td>2.9/1.04</td></tr><tr><td>SCR ACE [44]</td><td>0.5/0.18</td><td>0.8 /0.33</td><td>0.5/0.33</td><td>1.0 / 0.29</td><td>1 / 0.22</td><td>0.8/0.2</td><td>2.9 / 0.81</td><td>1.1/ 0.34</td></tr><tr><td rowspan="7">NRP</td><td>FQN-MN [45]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CrossFire [46]</td><td>4.1/1.31 1/0.4</td><td>10.5/2.97 5/1.9</td><td>9.2/2.45 3/2.3</td><td>3.6/2.36 5/1.6</td><td>4.6/1.76 3/0.8</td><td>16.1/4.42 2/0.8</td><td>139.5/34.67 12/1.9</td><td>28/7.3 4.4/1.38</td></tr><tr><td>PNeRFLoc [33]</td><td>2/0.8</td><td>2/0.88</td><td>1/0.83</td><td>3/1.05</td><td>6/1.51</td><td>5/1.54</td><td>32/5.73</td><td>7.28/1.76</td></tr><tr><td>NeRFMatch [34]</td><td>0.9/0.3</td><td>1.1/0.4</td><td>1.5/1.0</td><td>3.0/0.8</td><td>2.2/0.6</td><td>1.0/0.3</td><td>10.1/1.7</td><td>2.8/0.7</td></tr><tr><td>GSplatLoc (Coarse)</td><td>3.17/0.49</td><td>3.34/0.7</td><td>1.96/0.76</td><td>3.8/0.62</td><td>5.12/0.7</td><td>4.54/0.64</td><td>10.97/2.63</td><td>4.7/0.94</td></tr><tr><td>GSplatLoc (Base)</td><td>0.43/0.16</td><td>1.03/0.32</td><td>1.06/0.62</td><td>1.85/0.4</td><td>1.80/0.35</td><td>2.71/0.55</td><td>8.83/2.34</td><td>2.53/0.68</td></tr><tr><td>GSplatLoc (Fine)</td><td>0.39 /0.13</td><td>0.91 /0.29</td><td>0.94 0.50</td><td>1.41/ 0.32</td><td>1.41 / 0.26</td><td>1.3210.29</td><td>3.44 /0.82</td><td>1.40 / 0.37</td></tr></table>

## V. EXPERIMENTS

Experimental Setup. We evaluate our method using the 7Scenes dataset [47] for indoor validation and the Cambridge Landmarks dataset [13] for outdoor validation.

In the modeling phase, COLMAP [48] is used to generate point clouds and initialize poses for each scene. We then train the 3D Gaussian Splatting (3DGS) model from [9] on each scene for 15,000 iterations. XFeat [40] is used to extract dense features for 3DGS.

In the testing phase, we start by obtaining an initial coarse pose. We sample 1,000 of the most reliable descriptors and match them to the 3D feature cloud. The number of RANSAC iterations is set to 20,000.

For the refinement step, we render a visual reference for the coarse pose once and use the Adam optimizer with a learning rate of 0.001. We optimize both translation and rotation in quaternion form.

GSplatLoc Variants. We evaluate three variants of the GSplatLoc model:

â¢ GSplatLoc (Coarse): Uses only the initial coarse pose estimate.

â¢ GSplatLoc (Base): Incorporates photometric warpingbased pose refinement for enhanced accuracy.

â¢ GSplatLoc (Fine): Combines both feature-based and warping-based refinements for the highest precision.

Each variant progressively improves localization accuracy, with trade-offs in computational cost and refinement quality.

Figure 3 demonstrates that the warping-based optimization progressively enhances localization accuracy from the initial coarse pose, requiring approximately 250 iterations for indoor scenes and 350 for outdoor scenes, with the visual reference rendered only once. For the feature-based refinement stage, five iterations were selected as the optimal tradeoff between accuracy and efficiency.

Figure 4 demonstrates the optimization of the camera pose estimated by the Base version of GsplatLoc using a rendering-based photometric warp loss, which iteratively minimizes errors in translation and rotation. Accuracy is evaluated based on the percentage of frames where the pose error is less than 1 cm and 1Â°. The plot highlights the improvement in accuracy achieved by adding the warp loss. Performance Results. We compare our results with several state-of-the-art methods and report the median translation and rotation errors in (cm/degree) for each scene and the average across all scenes in Tables I and II.

TABLE II: Comparison of methods on the Cambridge Landmarks dataset: median translation and rotation errors (cm/Â°) for various approaches. APR denotes absolute pose regression, SCR represents scene coordinate regression, and NRP stands for neural render pose estimation. The best results are highlighted as first and second
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>|Methods</td><td rowspan=1 colspan=5>Kings  Hospital  Shop  Church|Avg. â [cm/]</td></tr><tr><td rowspan=1 colspan=1>APR</td><td rowspan=1 colspan=4>|PoseNet [13][MS-Transformer [43]]LENS [31]DFNet [19]</td><td rowspan=1 colspan=3>93/2.73 224/7.88 147/6.62237/5.9485/1.45 175/2.43 88/3.20 166/4.1233/0.5  44/0.9  27/1.6  53/1.673/2.37 200/2.98 67/2.21 137/4.02</td><td rowspan=1 colspan=2>175/5.79129/2.8039/1.15119/2.90</td></tr><tr><td rowspan=1 colspan=1>SCR</td><td rowspan=1 colspan=4>|Ace [44]</td><td rowspan=1 colspan=3>29/0.38 31/0.61 50.3  19/0.6</td><td rowspan=1 colspan=2>21/0.47</td></tr><tr><td rowspan=5 colspan=1>NRP</td><td rowspan=5 colspan=4>[FQN-MN [45]CrossFire [46]PNeRFLoc [33]GSplatLoc (Coarse)GSplatLoc (Base)GSplatLoc (Fine)</td><td rowspan=1 colspan=3>28/0.4  54/0.8  13/0.6   58/247/0.7  43/0.7  20/1.2  39/1.4</td><td rowspan=2 colspan=2>38/137/124.5/0.37</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>24 / 0.29</td><td rowspan=1 colspan=2>280.376/0.2740/0.55</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Coar:</td><td rowspan=1 colspan=1>rse)</td><td rowspan=1 colspan=1>41/0.50</td><td rowspan=1 colspan=1>32/0.87</td><td rowspan=1 colspan=1>11/0.40 31/0.72</td><td rowspan=1 colspan=1>29/0.62</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>27/0.46</td><td rowspan=1 colspan=1>20/0.71</td><td rowspan=1 colspan=1>5/0.36 16/0.61</td><td rowspan=1 colspan=1>17/0.53</td></tr><tr><td rowspan=1 colspan=3>31/0.49 160.684/0.3414 /0.42</td></tr></table>

TABLE III: Comparison of methods on Our Custom Dataset. The table reports accuracy as the percentage of frames with position and orientation errors below the thresholds (10cm/5â¦), (5cm/5â¦), (2cm/2â¦), and (1cm/1â¦). It also includes the median translation and rotation errors (cm/Â°). The best results are highlighted as first and second .
<table><tr><td>Method</td><td></td><td>|10cm/5Â° 5cm/5Â° 2cm/2Â° 1cm/1 â(%)</td><td>â(%)</td><td>â(%)</td><td>â(%)</td><td>P Median Error â(cm/)</td></tr><tr><td>SCR </td><td>|ACE [44]</td><td>98.6</td><td>93.9</td><td>76.9</td><td>53.1</td><td>1.0/0.20</td></tr><tr><td rowspan="3">NRP</td><td>GSplatLoc (Coarse)</td><td>95.2</td><td>83.0</td><td>37.4</td><td>10.2</td><td>2.5/0.40</td></tr><tr><td>GSplatLoc (Base)</td><td>97.3</td><td>90.5</td><td>73.5</td><td>57.8</td><td>0.8/ 0.12</td></tr><tr><td>GSplatLoc (Fine)</td><td>99.3</td><td>99.3</td><td>90.5</td><td>78.2</td><td>0.40.06</td></tr></table>

TABLE IV: Effect of feature descriptors. The table reports the accuracy metric as the percentage of frames below (1cm/1â¦)/( 5cm/5â¦) position and orientation discrepancies thresholds, descriptor dimension, and size per scene in MB. Tested on the 7Scenes dataset. The best results are highlighted in first and second .
<table><tr><td>Backbone</td><td>Dim.â|</td><td>Size(MB)â</td><td>Chess(%)</td><td>Fire (%)</td><td>Heads(%)</td><td>Office(%)</td><td>Pumpkin(%)</td><td>Redkitchen(%)</td><td>Stairs (%)</td><td>Avg.(%) â</td></tr><tr><td>DeDoDe-G</td><td>256</td><td>~800</td><td>37.1/94.2</td><td>17.2/69.8</td><td>33.1/86.8</td><td>1.8/50.9</td><td>1.6/38.5</td><td>4.7/48.7</td><td>0.6/20.5</td><td>13.7/58.5</td></tr><tr><td>SuperPoint</td><td>256</td><td>~800</td><td>70.9/99.7</td><td>38.2 93.0</td><td>25.9/76.9</td><td>19.7/88.0</td><td>13.2 81.2</td><td>27.3/82.8</td><td>5.1/56.2</td><td>28.6 82.56</td></tr><tr><td>XFeat</td><td>64</td><td>~300</td><td>81.2/99.8</td><td>46.8 88.3</td><td>51.6/87.6</td><td>16.2/83.1</td><td>18.5 78.2</td><td>26.9/77.4</td><td>1.2/31.8</td><td>34.6 78.0</td></tr></table>

<!-- image-->  
Fig. 4: Camera pose optimization via the Base variant is performed using a rendering-based photometric warp loss, progressively enhancing accuracy. The plot shows the percentage of frames that fall below the 1cm/1â¦ threshold, highlighting the improvement in accuracy over iterations.

For the indoor 7Scenes dataset, our method, GSplatLoc, surpasses the previous state-of-the-art neural render pose estimation method, NeRFMatch, in six out of seven scenes, setting a new state-of-the-art among NRP methods. However, GSplatLoc is still outperformed by ACE, a scene coordinate regression approach.

On the outdoor dynamic Cambridge Landmarks dataset, GSplatLoc achieves the best overall performance, surpassing ACEâthe second-best methodâby an average of more than 5 cm in translation error. This emphasizes that the proposed method excels in handling dynamic scenarios by incorporating a robust outlier rejection step,

Furthermore, we position our solution as a downstream approach for robot localization. To validate this capability, we built a custom dataset captured in a dynamic indoor environment. Specifically, a quadruped robot equipped with a Livox Mid-360 LiDAR and a ZED X camera navigated through an office-like area filled with moving people. The environment also includes numerous transparent objects such as mirrors, which present significant challenges for many localization techniques. Some qualitative results of this custom dataset can be seen in figure 5. Instead of relying on COLMAP, we leveraged LiDAR priors for scene reconstruction. Each scene was optimized for 15,000 iterations. During inference, we used 250 iterations for warping-based refinement and 5 iterations for feature-based pose refinement. In this challenging scenario, our method outperforms the state-of-the-art indoor relocalization approach (ACE). The corresponding results are presented in Table III.

TABLE V: Runtime analysis of the execution time per a query frame for each method across different stages.
<table><tr><td>Method</td><td>Feature Est. (s)</td><td>Rendering (s)</td><td>Refinement (s)</td><td>Overall Query (s)</td></tr><tr><td>PNeRFLoc</td><td>N/A</td><td>N/A</td><td>N/A</td><td>5.560</td></tr><tr><td>NeRFMatch</td><td>0.157</td><td>0.141</td><td>0.846</td><td>1.144</td></tr><tr><td>GSplatLoc (Base)</td><td>0.018</td><td>0.140</td><td>0.651</td><td>0.809</td></tr><tr><td>GSplatLoc (Fine)</td><td>0.018</td><td>0.140</td><td>1.911</td><td>2.069</td></tr></table>

<!-- image-->  
Fig. 5: Qualitative results on Our Custom Dataset. The diagonal line separates the test query images from the renders synthesized using poses estimated by GSplatLoc Fine.

Effect of Feature Descriptors. We evaluated XFeat [40], DeDoDe-G [49], and SuperPoint [35] in the GSplatLoc Base setting to assess the impact of different feature descriptors. Results in Table IV show that XFeat, the lightest, outperforms DeDoDe-G and matches SuperPoint at the 5cm/5â¦ threshold, while surpassing both at the 1cm/1â¦ threshold. XFeat provides a higher percentage of frames within the tighter accuracy threshold but has more "outlier frames", slightly reducing its accuracy compared to SuperPoint. Additionally, XFeat requires less storage, making it a more efficient choice.

Runtime Analysis. Table V shows the execution time of our method on the 7Scenes dataset. The shortest processing time is required for coarse pose search and rendering, while the longest time is taken by the iterative process of refining with warp loss over 250 iterations. NeRFMatchâs refinement time is computed by averaging across six iterations, with each optimization step taking 141 ms [34].

Our pipeline outperforms NRP methods by balancing execution speed and estimation accuracy. NeRFMatch and PNeRFLoc require several seconds per query, whereas our method delivers a coarse pose estimate in just 0.2 seconds. On an RTX 4090 GPU, our method achieves superior accuracy compared to competitors in under 1 second.

## VI. DISCUSSION

Our approach outperforms other neural rendering pose estimation (NRP) methods, achieving the best overall accuracy in indoor settings. For outdoor settings, which involve dynamic objects and lighting variations, we demonstrate superior position estimation accuracy. Although PNeRFLoc [33] achieves higher accuracy on certain sequences, its performance significantly degrades on others. This variability shows that our method offers more robust performance and greater reliability when transitioning between indoor and outdoor environments, making it well-suited for diverse use cases, such as delivery robots and AR applications.

Iterative optimization is crucial for accurate localization, but poor reconstruction can degrade warp loss and photometric optimization. In the Stairs scene from the 7Scenes dataset, the Fine version of GSplatLoc, which includes featuremetric pose optimization, outperforms the Base version. This underscores the importance of feature-based optimization for reliable localization, though it comes with a tradeoff between accuracy and computational efficiency.

Additionally, the use of 3DGS for spatial representations significantly accelerates the training process compared to NeRF-based NRP methods, playing a key role in scaling for large outdoor scenes. Another strong competitor, ACE [44], being better indoors, it struggles with outdoor scenarios, since its shallow MLP-based scene encoding design constrains its modeling capabilities.

At the same time, NRP methods, to which GSplatLoc belongs, are a better alternative, since it is a more versatile framework than SCR, as it allows solving multiple tasks in parallel using the same scene representation. For example, 3DGS can be used to encode semantic or language-aligned instances or even model dynamic scenesâtasks that are crucial for robotics. This flexibility enables a broader range of interactions with the environment and enhances localization.

We explain the overall good performance of GSplatLoc by two main reasons. First, feature distillation combined with 3DGS enables accurate structure-based coarse pose estimation. Second, representations that provide realistic images facilitate effective photometric optimization.

## VII. CONCLUSIONS AND FUTURE WORK

We present GSplatLoc, a framework using 3D Gaussian Splatting (3DGS) for visual localization in indoor and outdoor environments, including dynamic scenes. The method employs a two-stage process: a coarse pose estimate from 2D-3D correspondences, followed by warping loss refinement. By combining structure-based matching with rendering optimization from scene-agnostic 3DGS descriptors, we improve accuracy and efficiency. Our approach outperforms APR and NRP methods indoors and surpasses SCR-based ACE outdoors, handling dynamic objects and lighting while being the fastest among NRP methods. We also demonstrate the effectiveness of the lightweight XFeat feature extractor. Future work will focus on removing floaters from the 3DGS model, extending it to large-scale outdoor scenarios, and applying NRP for semantic SLAM and navigation.

[1] Z. Dong, G. Zhang, J. Jia, and H. Bao, âKeyframe-based real-time camera tracking,â in 2009 IEEE 12th International Conference on Computer Vision, 2009, pp. 1538â1545.

[2] L. Heng, B. Choi, Z. Cui, M. Geppert, S. Hu, B. Kuan, P. Liu, R. Nguyen, Y. C. Yeo, A. Geiger, G. H. Lee, M. Pollefeys, and T. Sattler, âProject autovision: Localization and 3d scene perception for an autonomous vehicle with a multi-camera system,â in 2019 International Conference on Robotics and Automation (ICRA). IEEE, May 2019, p. 4695â4702.

[3] H. Lim, S. N. Sinha, M. F. Cohen, and M. Uyttendaele, âReal-time image-based 6-dof localization in large-scale environments,â in 2012 IEEE Conference on Computer Vision and Pattern Recognition, 2012, pp. 1043â1050.

[4] P.-E. Sarlin, A. Unagar, M. Larsson, H. Germain, C. Toft, V. Larsson, M. Pollefeys, V. Lepetit, L. Hammarstrand, F. Kahl, and T. Sattler, âBack to the feature: Learning robust camera localization from pixels to pose,â in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2021, p. 3246â3256.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, p. 99â106, Dec. 2021.

[6] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, p. 1â14, July 2023.

[7] V. Tschernezki, I. Laina, D. Larlus, and A. Vedaldi, âNeural feature fusion fields: 3d distillation of self-supervised 2d image representations,â in 2022 International Conference on 3D Vision (3DV). IEEE, Sept. 2022.

[8] S. Kobayashi, E. Matsumoto, and V. Sitzmann, âDecomposing nerf for editing via feature field distillation,â in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 23 311â23 330.

[9] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi, âFeature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2024, p. 21676â21685.

[10] S. Chen, Y. Bhalgat, X. Li, J.-W. Bian, K. Li, Z. Wang, and V. A. Prisacariu, âNeural refinement for absolute pose regression with feature synthesis,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2024, p. 20987â20996.

[11] C. Liu, S. Chen, Y. S. Bhalgat, S. HU, M. Cheng, Z. Wang, V. A. Prisacariu, and T. Braud, âGS-CPR: Efficient camera pose refinement via 3d gaussian splatting,â in The Thirteenth International Conference on Learning Representations, 2025.

[12] Z. Niu, Z. Tan, J. Zhang, X. Yang, and D. Hu, âHgsloc: 3dgs-based heuristic camera pose refinement,â arXiv preprint arXiv:2409.10925, 2024.

[13] A. Kendall, M. Grimes, and R. Cipolla, âPosenet: A convolutional network for real-time 6-dof camera relocalization,â in 2015 IEEE International Conference on Computer Vision (ICCV). IEEE, Dec. 2015, p. 2938â2946.

[14] E. Brachmann, F. Michel, A. Krull, M. Y. Yang, S. Gumhold, and C. Rother, âUncertainty-driven 6d pose estimation of objects and scenes from a single rgb image,â in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2016.

[15] S. Brahmbhatt, J. Gu, K. Kim, J. Hays, and J. Kautz, âGeometryaware learning of maps for camera localization,â in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, June 2018, p. 2616â2625.

[16] B. Wang, C. Chen, C. Xiaoxuan Lu, P. Zhao, N. Trigoni, and A. Markham, âAtloc: Attention guided camera localization,â Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 06, p. 10393â10401, Apr. 2020.

[17] H. Hu, Z. Qiao, M. Cheng, Z. Liu, and H. Wang, âDasgil: Domain adaptation for semantic and geometric-aware image-based localization,â IEEE Transactions on Image Processing, vol. 30, p. 1342â1353, 2021.

[18] E. Arnold, J. Wynn, S. Vicente, G. Garcia-Hernando, A. Monszpart, V. Prisacariu, D. Turmukhambetov, and E. Brachmann, Map-Free Visual Relocalization: Metric Pose Relative to a Single Image. Springer Nature Switzerland, 2022, p. 690â708.

[19] S. Chen, X. Li, Z. Wang, and V. Prisacariu, âDfnet: Enhance absolute pose regression with direct feature matching,â in Proceedings of the European Conference on Computer Vision (ECCV), 2022.

[20] Y. Shavit and Y. Keller, Camera Pose Auto-encoders for Improving Pose Regression. Springer Nature Switzerland, 2022, p. 140â157.

[21] S. Chen, T. Cavallari, V. A. Prisacariu, and E. Brachmann, âMaprelative pose regression for visual re-localization,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2024, p. 20665â20674.

[22] J. Shotton, B. Glocker, C. Zach, S. Izadi, A. Criminisi, and A. Fitzgibbon, âScene coordinate regression forests for camera relocalization in rgb-d images,â in 2013 IEEE Conference on Computer Vision and Pattern Recognition, 2013, pp. 2930â2937.

[23] E. Brachmann, A. Krull, S. Nowozin, J. Shotton, F. Michel, S. Gumhold, and C. Rother, âDsac â differentiable ransac for camera localization,â in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, July 2017.

[24] E. Brachmann and C. Rother, âLearning less is more - 6d camera localization via 3d surface regression,â in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, June 2018.

[25] X. Li, J. Ylioinas, J. Verbeek, and J. Kannala, Scene Coordinate Regression with Angle-Based Reprojection Loss for Camera Relocalization. Springer International Publishing, 2019, p. 229â245.

[26] E. Brachmann and C. Rother, âVisual camera re-localization from rgb and rgb-d images using dsac,â IEEE Transactions on Pattern Analysis and Machine Intelligence, p. 1â1, 2021.

[27] â, âExpert sample consensus applied to camera re-localization,â in 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2019, p. 7524â7533.

[28] E. Brachmann, T. Cavallari, and V. A. Prisacariu, âAccelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses,â in CVPR, 2023.

[29] F. Wang, X. Jiang, S. Galliani, C. Vogel, and M. Pollefeys, âGlace: Global local accelerated coordinate encoding,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2024, p. 5819â5828.

[30] L. Yen-Chen, P. Florence, J. T. Barron, A. Rodriguez, P. Isola, and T.-Y. Lin, âinerf: Inverting neural radiance fields for pose estimation,â in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, Sept. 2021.

[31] A. Moreau, N. Piasco, D. Tsishkou, B. Stanciulescu, and A. de La Fortelle, âLens: Localization enhanced by nerf synthesis,â in Conference on Robot Learning. PMLR, 2022, pp. 1347â1356.

[32] R. Martin-Brualla, N. Radwan, M. S. M. Sajjadi, J. T. Barron, A. Dosovitskiy, and D. Duckworth, âNerf in the wild: Neural radiance fields for unconstrained photo collections,â in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2021, p. 7206â7215.

[33] B. Zhao, L. Yang, M. Mao, H. Bao, and Z. Cui, âPnerfloc: Visual localization with point-based neural radiance fields,â Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, p. 7450â7459, Mar. 2024.

[34] Q. Zhou, M. Maximov, O. Litany, and L. Leal-TaixÃ©, The NeRFect Match: Exploring NeRF Features for Visual Localization. Springer Nature Switzerland, Nov. 2024, p. 108â127.

[35] D. DeTone, T. Malisiewicz, and A. Rabinovich, âSuperpoint: Self-supervised interest point detection and description,â in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, June 2018.

[36] M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler, âD2-net: A trainable cnn for joint description and detection of local features,â in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2019, p. 8084â8093.

[37] J. Revaud, P. Weinzaepfel, C. R. de Souza, and M. Humenberger, âR2D2: repeatable and reliable detector and descriptor,â in NeurIPS, 2019.

[38] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, âLoftr: Detector-free local feature matching with transformers,â in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2021, p. 8918â8927.

[39] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, âLightglue: Local feature matching at light speed,â in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2023.

[40] G. Potje, F. Cadar, A. Araujo, R. Martins, and E. R. Nascimento, âXfeat: Accelerated features for lightweight image matching,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2024, p. 2682â2691.

[41] P. Lindenberger, P.-E. Sarlin, V. Larsson, and M. Pollefeys, âPixelperfect structure-from-motion with featuremetric refinement,â in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2021, p. 5967â5977.

[42] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 786â12 796.

[43] Y. Shavit, R. Ferens, and Y. Keller, âLearning multi-scene absolute pose regression with transformers,â in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2021, p. 2713â2722.

[44] E. Brachmann, T. Cavallari, and V. A. Prisacariu, âAccelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses,â in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2023.

[45] H. Germain, D. DeTone, G. Pascoe, T. Schmidt, D. Novotny, R. Newcombe, C. Sweeney, R. Szeliski, and V. Balntas, âFeature query networks: Neural surface description for camera pose refinement,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2022, pp. 5071â5081.

[46] A. Moreau, N. Piasco, M. Bennehar, D. Tsishkou, B. Stanciulescu, and A. de La Fortelle, âCrossfire: Camera relocalization on self-supervised features from an implicit representation,â in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, Oct. 2023, p. 252â262.

[47] B. Glocker, S. Izadi, J. Shotton, and A. Criminisi, âReal-time rgbd camera relocalization,â in 2013 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), 2013, pp. 173â179.

[48] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, June 2016.

[49] J. Edstedt, G. BÃ¶kman, M. WadenbÃ¤ck, and M. Felsberg, âDedode: Detect, donât describe â describe, donât detect for local feature matching,â 2024 International Conference on 3D Vision (3DV), pp. 148â157, 2023.