# GLC-SLAM: Gaussian Splatting SLAM with Efficient Loop Closure

Ziheng Xu1, Qingfeng Li1, Chen Chen2, Xuefeng Liu1 and Jianwei Niu1â

Abstractâ 3D Gaussian Splatting (3DGS) has gained significant attention for its application in dense Simultaneous Localization and Mapping (SLAM), enabling real-time rendering and high-fidelity mapping. However, existing 3DGS-based SLAM methods often suffer from accumulated tracking errors and map drift, particularly in large-scale environments. To address these issues, we introduce GLC-SLAM, a Gaussian Splatting SLAM system that integrates global optimization of camera poses and scene models. Our approach employs frame-to-model tracking and triggers hierarchical loop closure using a globalto-local strategy to minimize drift accumulation. By dividing the scene into 3D Gaussian submaps, we facilitate efficient map updates following loop corrections in large scenes. Additionally, our uncertainty-minimized keyframe selection strategy prioritizes keyframes observing more valuable 3D Gaussians to enhance submap optimization. Experimental results on various datasets demonstrate that GLC-SLAM achieves superior or competitive tracking and mapping performance compared to state-of-the-art dense RGB-D SLAM systems.

## I. INTRODUCTION

Visual SLAM plays a crucial role in various applications such as virtual/augmented reality (VR/AR), robot navigation, and autonomous driving. Over the past decade, visual SLAM methods with various scene representation have been developed, ranging from traditional approaches using point clouds [1], [2], surfels [3], [4] and voxels [5], [6] to neural implicit methods [7]â[12] leveraging neural radiance fields (NeRF) [13]. Traditional SLAM methods provide accurate tracking and real-time mapping but struggle to generate high-quality, texture-rich maps or synthesize novel views. In contrast, NeRF-based SLAM methods offer coherent mapping and accurate surface reconstruction but are limited by the high computational cost of volume rendering, hindering real-time performance.

Recently, 3DGS [14] has emerged as a promising alternative to NeRF, offering comparable high-quality rendering with significantly faster rendering and training speeds. Consequently, SLAM methods [15]â[20] based on Gaussian Splatting representation demonstrate advancements in terms of photo-realistic rendering, high-fidelity reconstruction and real-time performance. It is worth noting that 3D Gaussian maps can be explicitly edited and deformed, making them particularly suitable for map correction.

However, existing 3DGS-based SLAM methods face the challenge of error accumulation and map distortion due to the absence of loop closure for global adjustment of camera poses and the constructed map. While Photo-SLAM [18] incorporates loop closure based on ORB-SLAM [1], its dependence on a feature-based tracker constrains the effectiveness of loop closure, as the tracker is unable to exploit the map refinements. NeRF-based SLAM methods [21]â[23] integrate online loop closure to achieve accurate and robust tracking, yet require storing historical frames and costly retraining the entire implicit map to update loop correction. The lack of a robust, efficient loop closure in 3DGS-based SLAM remains a key limitation to achieving global consistency in large-scale environments.

<!-- image-->  
Fig. 1. Reconstruction results on ScanNet [24] 0054. Our method effectively mitigates the severe map drift inherent in Gaussian-SLAM [16], while also providing superior scene geometry and detail compared to GO-SLAM [21].

To address these challenges, we propose GLC-SLAM, a Gaussian Splatting SLAM system with efficient Loop Closure, designed to mitigate accumulated tracking errors and reduce map drift in large indoor scenes. Our approach incrementally builds 3D Gaussian submaps, with each submap anchored to a corresponding global keyframe. To maintain global consistency, we employ a hierarchical loop closing strategy that enhances global loop closure by drift-free submaps refined via local optimization. Upon loop detection, nodes and edges are added to the pose graph, followed by pose graph optimization. The optimization results are then updated to relevant submaps through direct map adjustment. Furthermore, we explicitly model Gaussian uncertainty and introduce an uncertainty-minimized keyframe selection method for robust active submap optimization. As shown in Fig. 1, GLC-SLAM successfully address map drift and yields improved scene geometry and detail, achieving high-fidelity and global consistent mapping. We conduct experiments on various datasets that demonstrate our method achieves robust tracking and accurate mapping performance compared to existing dense RGB-D SLAM methods.

Our main contributions are summarized as follows:

â¢ A Gaussian Splatting SLAM system that achieves robust frame-to-model tracking and global consistent mapping of 3D Gaussian submaps in large-scale environments.

â¢ The efficient loop closure module, including globalto-local loop detection, pose graph optimization and direct map updates, reducing accumulated errors and map drift.

â¢ The uncertainty-minimized keyframe selection strategy, which selects informative keyframes observing more stable 3D Gaussians during submap optimization to enhance mapping accuracy and robustness.

## II. RELATED WORK

## A. Visual SLAM

Early methods like ORB-SLAM [2] utilize feature-based approaches to estimate camera trajectories and construct 3D maps. While traditional SLAM systems, which typically employ explicit representations like voxels and point clouds, excel in tracking accuracy and efficiency, they are limited in providing high-fidelity maps and often lack generalization capabilities.

In recent years, NeRF [13] have gained significant attention in SLAM algorithms, with notable examples like iMAP [7], NICE-SLAM [8], and ESLAM [9] leveraging neural implicit representations for accurate and dense 3D surface reconstruction. However, these neural implicit methods are constrained by the high computational demands of volume rendering and face challenges in performing robust tracking in large-scale environments. To improve tracking robustness, some approaches incorporate loop closure and online global bundle adjustment (BA) to mitigate accumulated error. For example, MIPS-Fusion [25] employs a multi-implicitsubmap representation, achieving global optimization by refining and integrating these submaps, while GO-SLAM [21] combines loop closure with online full BA across all keyframes to ensure global consistency in large-scale environments. However, these methods either require storing the entire history of input frames or involve time-consuming retraining for map updates after loop closure.

## B. 3DGS-based SLAM

SLAM methods based on 3D Gaussian representation have recently garnered broad interest due to their ability to combine the strength of explicit and implicit expressions. Compared to NeRF-based methods, 3DGS-based methods capture high-fidelity 3D scenes through a differentiable rasterization process, avoiding the per-pixel ray casting required by neural fields, thus achieving real-time rendering. Gaussian-SLAM [16] organize scenes as 3D Gaussian submaps, allowing for efficient optimization and preventing catastrophic forgetting. SplaTAM [15] employs simplified 3D Gaussian representation, enabling real-time efficient optimization and high-quality rendering. However, these methods lack online loop correction, leading to the accumulation of errors and map drift. Photo-SLAM [18], building on ORB-SLAM [1], integrates loop closure to reduce cumulative errors and enhance tracking robustness, yet its design decouples tracking from mapping, which diminishes the effectiveness of loop closure and increases communication overhead. Our method constructs 3D Gaussian submaps incrementally and employs frame-to-model tracking, achieving a coupled SLAM system while reducing unnecessary storage consumption. By applying hierarchical loop closure and rapidly updates the scene through map deformation, we ensure robust tracking and efficient mapping in large-scale environments.

## III. PRELIMINARY

## A. Scene Representation

We represent the scene using 3D Gaussian submaps, where each submap $P ^ { s }$ consists of a collection of N 3D Gaussian distributions:

$$
P ^ { s } = \{ G _ { i } ^ { s } ( \mu , \Sigma , o , C ) \mid i = 1 , \ldots , N \} ,\tag{1}
$$

each 3D Gaussian is parameterized by mean $\mu ,$ , covariance Î£, opacity $^ { O , }$ and RGB color C. The covariance matrix Î£ is decomposed into a rotation vector r and a scale vector s. By using differentiable splatting to render color and depth maps, 3D Gaussians are optimized through an iterative process that involves calculating errors with input RGB-D images and updating the Gaussian parameters accordingly.

The color image $\hat { C }$ and depth map DË can be rendered by alpha-blending proposed in [14]:

$$
\begin{array} { c } { { \hat { C } = \displaystyle \sum _ { i = 1 } ^ { n } c _ { i } \alpha _ { i } T _ { i } , \hat { D } = \displaystyle \sum _ { i = 1 } ^ { n } d _ { i } \alpha _ { i } T _ { i } , \nonumber } } \\ { { T _ { i } = \displaystyle \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , } } \end{array}\tag{2}
$$

where $c _ { i }$ and $d _ { i }$ are the color and depth value of a 3D Gaussian. $\alpha _ { i }$ is computed by the pixel coordinate u, mean $\mu$ and covariance matrix $\Sigma _ { 2 D }$ of the splatted 2D Gaussian in pixel space:

$$
\alpha _ { i } = o _ { i } \exp ( - \frac { 1 } { 2 } ( u - \mu ) ^ { T } \Sigma _ { 2 D } ^ { - 1 } ( u - \mu ) ) .\tag{3}
$$

## B. Uncertainty Modeling

Uncertainty modeling introduces non-uniform weights to select more valuable pixels and 3D Gaussians during optimization, rather than treating them with equal importance. Following [26], we explicitly model the uncertainty of the rendered depth images and 3D Gaussians.

We render the depth uncertainty map $U$ as:

$$
U = \sum _ { i = 1 } ^ { N } \alpha _ { i } T _ { i } ( d _ { i } - D ) ^ { 2 } ,\tag{4}
$$

where D represents the ground truth depth values.

We define the dominated pixels of a 3D Gaussian same as [26] and calculate the uncertainty $\nu _ { i }$ of the i-th 3D Gaussian by the difference between its depth and depth observations from all its dominated pixels $P = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { n } \}$ within a keyframe window.

$$
\nu _ { i } = \frac 1 n \sum _ { p _ { k } \in P } \alpha _ { i } ^ { k } T _ { i } ^ { k } \left( D ^ { k } - d _ { i } ^ { k } \right) ^ { 2 } .\tag{5}
$$

<!-- image-->  
Fig. 2. System Overview. Our system consists of three processes: tracking, mapping and loop closing. The tracking process estimates and refines camera poses {R, t} by minimizing the tracking loss. The scene is managed as Gaussian submaps while the local mapping process select keyframes with an uncertainty-minimized strategy to optimize the active submap. If a loop is detected, the loop closing process triggers loop closure online, followed by efficient map adjustment to correct accumulated error and mitigate map drift.

$\alpha _ { i } ^ { k }$ and $T _ { i } ^ { k }$ represent the opacity and transmittance of the i-th 3D Gaussian on a pixel $p _ { k } . \ D ^ { k }$ and $d _ { i } ^ { k }$ represent the ground truth depth on a pixel $p _ { k }$ and the depth value of the i-th 3D Gaussian respectively.

## IV. SYSTEM

The overview of our proposed GLC-SLAM system is shown in Fig. 2. In this section, we introduce our system from the following aspects: tracking (IV-A), local mapping (IV-B) and loop closing (IV-C).

## A. Tracking

We adopt a coupled system design by performing frameto-model tracking based on the mapped scene. We first initialize the current camera pose $T _ { i }$ with a constant speed assumption:

$$
T _ { i } = T _ { i - 1 } + ( T _ { i - 1 } - T _ { i - 2 } ) ,\tag{6}
$$

where camera pose $T _ { i } = \{ R _ { i } , t _ { i } \}$ can be decompose into a rotation matrix and a translation vector. $T _ { i }$ is then optimized by minimize the tracking loss $L _ { \mathrm { t r a c k i n g } }$ with respect to relative camera pose $T _ { i - 1 , i }$ between frames i â 1 and i. We apply an alpha mask $M _ { \mathrm { a l p h a } }$ and an inlier mask $M _ { \mathrm { i n l i e r } }$ in the tracking loss to address gross errors caused by poorly reconstructed or previously unobserved areas as follows:

$$
L _ { \mathrm { t r a c k i n g } } = \sum M _ { \mathrm { i n } } { \cdot } M _ { \mathrm { a l p h a } } { \cdot } ( \lambda _ { c } | \hat { C } { - } C | _ { 1 } { + } ( 1 - \lambda _ { c } ) | \hat { D } { - } D | _ { 1 } ) ,\tag{7}
$$

where $\lambda _ { c }$ is a weight that balances the color and depth losses, and C and D are the input color and depth map.

## B. Local Mapping

We grow submaps of 3D Gaussians in a progressive manner and anchor each submap to a global keyframe. All Gaussians in the active submap are jointly optimized every time new Gaussians are added to the submap for a fixed number of iterations minimizing the loss Eq. (13) and only the selected keyframes are included in the optimization.

1) Map Building: We grow submaps incrementally with newly incoming keyframes and initialize new submaps when the camera motion exceeds a threshold, with the first keyframe serving as a global keyframe. At any time, only the active submap is processed. This approach bounds the compute cost and ensures that optimization remains fast while exploring larger scenes.

Each new keyframe adds 3D Gaussians to the active submap, capturing newly observed regions of the scene. Specifically, a dense point cloud is computed from the RGB-D input following the pose estimation for the current keyframe. We apply a densification mask to fill holes of unobserved regions and avoid local minima in rendered images. Points are sampled uniformly from the regions with the accumulated alpha values lower than a threshold $\alpha _ { \mathrm { t h r e } }$ or large rendered color and depth error occurs. New 3D Gaussians are added to the submap using sampled points that have no neighbors within a search radius in the current submap. The new Gaussians are anisotropic and their scales are defined based on the nearest neighbor distance within the active submap.

2) Uncertainty-minimized Keyframe Selection: For a new input frame, we insert the frame into the keyframe set if the frame overlap ratio $r _ { o }$ between the current frame and the last inserted frame is lower than a threshold, where $r _ { o }$ is defined as:

$$
r _ { o } = { \frac { G _ { i } \cap G _ { i - 1 } } { G _ { i } \cup G _ { i - 1 } } } .\tag{8}
$$

Here, $G _ { i }$ and $G _ { i - 1 }$ are the 3D Gaussian sets observed by the current frame and last keyframe respectively.

Inspired by [27], we adopt an uncertainty-aware keyframe selection strategy in each map training iteration. This strategy, with the aid of Gaussian uncertainty, aims to select keyframes that observe more valuable 3D Gaussians which are likely to have a positive effect on the optimization. An informative score is defined for each keyframe as:

$$
s _ { \mathrm { i n f o r } } = { \frac { 1 } { | G | } } \sum _ { g \in G } \nu _ { g } ,\tag{9}
$$

where |G| is the number of observed 3D Gaussians by the keyframe.

We begin by selecting k keyframes that cover the Gaussians with the highest sum of scores. After labeling the covered Gaussians as observed we use the same selection strategy but only consider the remaining unobserved Gaussians when calculating $s _ { \mathrm { i n f o r } }$ in the next time step. If all Gaussians have been labeled as observed, the process is repeated by resetting the Gaussians to be labeled as unobserved.

3) Loss Function: We employ various loss functions to optimize Gaussian parameters. For depth supervision, we use the loss:

$$
{ \cal L } _ { \mathrm { d e p t h } } = \frac { 1 } { U } \| D - \hat { D } \| _ { 1 } ,\tag{10}
$$

with $D$ and $\hat { D }$ being the ground-truth and reconstructed depth maps, respectively. The depth loss $L _ { \mathrm { d e p t h } }$ is weighted by the uncertainty map $U$ to ensure that the pixels with high uncertainty are weighted less. For the color supervision we use a weighted combination of L1 and SSIM [28] losses:

$$
L _ { \mathrm { c o l o r } } = ( 1 - \lambda ) \cdot | \hat { C } - C | _ { 1 } + \lambda ( 1 - \mathrm { S S I M } ( \hat { C } , C ) ) ,\tag{11}
$$

where C is the original image, $\hat { C }$ is the rendered image, and $\lambda = 0 . 2$ . We also add an isotropic regularization term $L _ { \mathrm { r e g } } \mathrm { : }$

$$
L _ { \mathrm { r e g } } = \frac { 1 } { | P | } \sum _ { p \in P } | s _ { p } - \bar { s } _ { p } | _ { 1 }\tag{12}
$$

where P is a submap, $s _ { p }$ is the scale of a 3D Gaussian, $\bar { s } _ { p }$ is the mean submap scale, and $| P |$ is the number of 3D Gaussians in the submap. The final loss function for mapping is finally formulated as:

$$
L _ { \mathrm { m a p p i n g } } = \lambda _ { \mathrm { c o l o r } } \cdot L _ { \mathrm { c o l o r } } + \lambda _ { \mathrm { d e p t h } } \cdot L _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { r e g } } \cdot L _ { \mathrm { r e g } }\tag{13}
$$

where $\lambda _ { \mathrm { c o l o r } } , \lambda _ { \mathrm { d e p t h } } , \lambda _ { \mathrm { r e g } }$ are weights for the corresponding losses.

## C. Loop Closing

We employ hierarchical loop closure to achieve global consistency within and between submaps. Global loop closure corrects large inter-submap cumulative errors while local loop closure aid global correction with refined global keyframe poses and accurate intra-submap geometry.

1) Loop Detection: For place recognition, we use the pretrained NetVLAD [29] model to extract a feature descriptor for each keyframe. The extracted features are stored in global and local keyframe databases. Cosine similarity between descriptors serves as the criterion for loop detection.

Global loop detection is triggered when a new submap is created. We select the best match from the global keyframe database if the visual similarity score is higher than a threshold $s _ { \mathrm { g l o b a l } }$ , which is dynamically computed as the minimum score between the global keyframe and the keyframes within active submap. Local loop detection operates during the local mapping process, accepting the most similar keyframe with the similarity score exceeds a predefined threshold $s _ { \mathrm { l o c a l } }$ . To avoid false loops, especially in indoor scenes with repetitive objects like chairs or tables, we further apply a geometry check. We evaluate the frame overlap ratio between two loop candidate keyframes, and accept them if $r _ { o }$ exceeds a threshold.

2) Pose Graph Optimization: We construct a pose graph model where the nodes represent keyframe poses, and the edges correspond to sequential relative poses. Loop edge constraints are computed from the relative poses between loop nodes and subsequently added to the pose graph.

We perform pose graph optimization across the entire pose graph to align the estimated trajectory more closely to the ground truth. Pose graph optimization effectively mitigates cumulative error and improves tracking accuracy. We use the Levenberg-Marquarelt algorithm to solve this nonlinear pose graph optimization problem described by Eq. (14), where v is the set of nodes, $E _ { s }$ is the set of sequential edges, $E _ { l }$ is the set of loop edges and $\Lambda _ { i }$ represents the uncertainty of corresponding edges.

$$
v ^ { * } = \arg \operatorname* { m i n } _ { v } \frac { 1 } { 2 } \sum _ { e _ { i } \in E _ { s } , E _ { l } } e _ { i } ^ { T } \Lambda _ { i } ^ { - 1 } e _ { i } ,\tag{14}
$$

3) Map Adjustment: To maintain map consistency after pose graph optimization, we rearrange the 3D Gaussian submaps using a keyframe-centric adjustment strategy. Each 3D Gaussian $g _ { i }$ is associated to a keyframe, and submap adjustment is achieved by updating Gaussian means based on the optimized pose of the associated keyframe. Association is determined by which keyframe added the 3D Gaussian to the scene. The mean $\mu _ { i }$ is projected into $T ^ { \prime }$ to find the pixel correspondence. Specifically, assume that a keyframe with camera pose $T = \{ R , t \}$ is updated to $T ^ { \prime } = \{ R ^ { \prime } , t ^ { \prime } \}$ , we update the mean and rotation of all 3D Gaussians $g _ { i }$ associated with the keyframe. We update $\mu _ { i }$ and $r _ { i }$ accordingly as:

$$
\pmb { \mu } _ { i } ^ { \prime } = T ^ { \prime } T ^ { - 1 } \pmb { \mu } _ { i } , r _ { i } ^ { \prime } = R ^ { \prime } R ^ { - 1 } r _ { i } .\tag{15}
$$

After map adjustment, we perform a set of refinement steps on the updated submap. We disable pruning and densification of the 3D Gaussians and simply perform a set of optimization iterations using the same loss function Eq. (13).

## V. EXPERIMENT

## A. Experimental Setup

We describe our experimental setup and evaluate our method against state-of-the-art dense RGB-D SLAM methods on Replica [29] as well as the real world TUM-RGBD [32] and the ScanNet [24] datasets.

TABLE I  
TRACKING PERFORMANCE ON REPLICA [29]. THE BEST RESULTS ARE HIGHLIGHTED AS FIRST , SECOND , AND THIRD â INDICATES METHODS LEVERAGING EXTERNAL TRACKER.
<table><tr><td>Method</td><td>rm0</td><td>rml</td><td>rm2</td><td>offo</td><td>offl</td><td>off2</td><td>off3</td><td>off4</td><td>Avg.</td></tr><tr><td colspan="8">NeRF-based</td></tr><tr><td>NICE-SLAM [8]</td><td>0.97</td><td>1.31</td><td>1.07</td><td>0.88</td><td>1.00</td><td>1.06</td><td>1.10</td><td>1.13</td><td>1.06</td></tr><tr><td>Vox-Fusion [30]</td><td>1.37</td><td>4.70</td><td>1.47</td><td>8.48</td><td>2.04</td><td>2.58</td><td>1.11</td><td>2.94</td><td>3.09</td></tr><tr><td>ESLAM [9]</td><td>0.71</td><td>0.70</td><td>0.52</td><td>0.57</td><td>0.55</td><td>0.58</td><td>0.72</td><td>0.63</td><td>0.63</td></tr><tr><td>Point-SLAM [31]</td><td>0.61</td><td>0.41</td><td>0.37</td><td>0.38</td><td>0.48</td><td>0.54</td><td>0.69</td><td>0.72</td><td>0.52</td></tr><tr><td>MIPS-Fusion [25]</td><td>1.10</td><td>1.20</td><td>1.10</td><td>0.70</td><td>0.80</td><td>1.30</td><td>2.20</td><td>1.10</td><td>1.19</td></tr><tr><td>*GO-SLAM [21]</td><td>0.34</td><td>0.29</td><td>0.29</td><td>0.32</td><td>0.30</td><td>0.39</td><td>0.39</td><td>0.46</td><td>0.35</td></tr><tr><td colspan="10">3DGS-based</td></tr><tr><td>SplaTAM [15]</td><td>0.31 0.40 0.29</td><td></td><td></td><td>0.47</td><td>0.27</td><td>0.29</td><td>0.32</td><td>0.72</td><td>0.38</td></tr><tr><td>Gaussian-SLAM [16]</td><td>0.29 0.29 0.22</td><td></td><td></td><td>0.37</td><td>0.23</td><td>0.41</td><td>0.30</td><td>0.35</td><td>0.31</td></tr><tr><td>*Photo-SLAM [18]</td><td>0.54 0.39</td><td></td><td>0.31</td><td>0.52</td><td>0.44</td><td>1.28</td><td>0.78</td><td>0.58</td><td>0.60</td></tr><tr><td>GLC-SLAM (Ours)</td><td>0.20 0.19 0.13 0.31</td><td></td><td></td><td></td><td>0.13</td><td>0.32</td><td>0.21</td><td>0.33</td><td>0.23</td></tr></table>

TABLE II

TRACKING PERFORMANCE ON TUM-RGBD [32]. LC INDICATES LOOP CLOSURE.
<table><tr><td>Method</td><td>LC</td><td>frl/desk</td><td>fr2/xyz</td><td>fr3/off.</td><td> $\operatorname { A v g } .$ </td></tr><tr><td colspan="6">NeRF-based</td></tr><tr><td>NICE-SLAM [8]</td><td>x</td><td>4.26</td><td>6.19</td><td>3.87</td><td>4.77</td></tr><tr><td>Vox-Fusion [30]</td><td>X</td><td>3.52</td><td>1.49</td><td>26.01</td><td>10.34</td></tr><tr><td>ESLAM [9]</td><td>X</td><td>2.47</td><td>1.11</td><td>2.42</td><td>2.00</td></tr><tr><td>Point-SLAM [31]</td><td>X</td><td>4.34</td><td>1.31</td><td>3.48</td><td>3.04</td></tr><tr><td>MIPS-Fusion [25]</td><td></td><td>3.00</td><td>1.40</td><td>4.60</td><td>3.00</td></tr><tr><td colspan="6">3DGS-based</td></tr><tr><td>SplaTAM [15]</td><td></td><td>3.35</td><td>1.24</td><td>5.16</td><td>3.25</td></tr><tr><td>Gaussian-SLAM [16]</td><td></td><td>2.73</td><td>1.39</td><td>5.31</td><td>3.14</td></tr><tr><td>*Photo-SLAM [18]</td><td></td><td>2.60</td><td>0.35</td><td>1.00</td><td>1.32</td></tr><tr><td>GLC-SLAM (Ours)</td><td></td><td>1.85</td><td>1.30</td><td>3.53</td><td>2.23</td></tr></table>

1) Datasets: The Replica dataset [29] consists of highquality 3D reconstructions of diverse indoor scenes. We leverage the publicly available dataset by Sucar et al. [7], which contains trajectories from an RGB-D sensor. Additionally, we showcase our framework on real-world data using the TUM-RGBD dataset [32] and the ScanNet dataset [24]. The TUM-RGBD poses were captured utilizing an external motion capture system, while ScanNet uses poses from BundleFusion [33].

2) Metrics: We evaluate camera tracking accuracy using ATE RMSE [32]. Rendering quality is evaluated by comparing full-resolution rendered images to input training views using peak signal-to-noise ratio (PSNR), SSIM [28], and LPIPS [34] metrics. Reconstruction performance is measured on meshes produced by marching cubes [35] using the F1- score, which is the harmonic mean of the Precision (P) and Recall (R). We also report the depth L1 metric, which compares mesh depth at random poses to its ground truth.

3) Baseline Methods: We primarily compare our method to existing state-of-the-art dense RGB-D SLAM methods such as ESLAM [9], GO-SLAM [21], SplaTAM [15] and Gaussian-SLAM [16]. We use the reported numbers from the respective papers where available and for others, we reproduce the results by running the official code.

4) Implementation details: We run GLC-SLAM on a desktop PC with an Intel Core i9-12900KF CPU and an NVIDIA RTX 3090 GPU. In all our experiments, we set alpha threshold $\alpha _ { \mathrm { t h r e } } ~ = ~ 0 . 6$ and the local loop detection threshold $s _ { \mathrm { l o c a l } } = 0 . 8 .$ . For submap optimization, we select k = 5 keyframes with Î»color, $\lambda _ { \mathrm { d e p t h } }$ and $\lambda _ { \mathrm { r e g } }$ to 1.

TABLE III  
TRACKING PERFORMANCE ON SCANNET [24].
<table><tr><td rowspan=1 colspan=8>Scene ID         000000590106016901810207Avg.NeRF-based</td></tr><tr><td rowspan=1 colspan=1>NICE-SLAM [8]</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>7.9</td><td rowspan=1 colspan=1>10.9</td><td rowspan=1 colspan=1>13.4</td><td rowspan=1 colspan=1>6.2</td><td rowspan=1 colspan=1>10.7</td></tr><tr><td rowspan=1 colspan=1>Vox-Fusion [30]</td><td rowspan=1 colspan=1>16.6</td><td rowspan=1 colspan=1>24.2</td><td rowspan=1 colspan=1>8.4</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>23.3</td><td rowspan=1 colspan=1>9.4</td><td rowspan=1 colspan=1>18.2</td></tr><tr><td rowspan=1 colspan=1>ESLAM [9]</td><td rowspan=1 colspan=1>7.3</td><td rowspan=1 colspan=1>8.5</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>6.5</td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=1>7.4</td></tr><tr><td rowspan=1 colspan=1>Point-SLAM [31]</td><td rowspan=1 colspan=1>10.2</td><td rowspan=1 colspan=1>7.8</td><td rowspan=1 colspan=1>8.7</td><td rowspan=1 colspan=1>22.2</td><td rowspan=1 colspan=1>14.8</td><td rowspan=1 colspan=1>9.5</td><td rowspan=1 colspan=1>12.2</td></tr><tr><td rowspan=1 colspan=1>MIPS-Fusion [25]</td><td rowspan=1 colspan=1>7.9</td><td rowspan=1 colspan=1>10.7</td><td rowspan=1 colspan=1>9.7</td><td rowspan=1 colspan=1>9.7</td><td rowspan=1 colspan=1>14.2</td><td rowspan=1 colspan=1>7.8</td><td rowspan=1 colspan=1>10.0</td></tr><tr><td rowspan=1 colspan=1>*GO-SLAM [21]</td><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>7.0</td><td rowspan=1 colspan=1>7.7</td><td rowspan=1 colspan=1>6.8</td><td rowspan=1 colspan=1>6.9</td><td rowspan=1 colspan=1>_6.9</td></tr><tr><td rowspan=1 colspan=1>3DGS-based</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>MonoGS [17]</td><td rowspan=1 colspan=1>9.8</td><td rowspan=1 colspan=1>32.1</td><td rowspan=1 colspan=1>8.9</td><td rowspan=1 colspan=1>10.7</td><td rowspan=1 colspan=1>21.8</td><td rowspan=1 colspan=1>7.9</td><td rowspan=1 colspan=1>15.2</td></tr><tr><td rowspan=2 colspan=2>SplaTAM [15]       12.8Gaussian-SLAM [16]24.8</td><td rowspan=1 colspan=1>12.8</td><td rowspan=1 colspan=1>10.1</td><td rowspan=1 colspan=1>17.7</td><td rowspan=1 colspan=1>12.1</td><td rowspan=1 colspan=1>11.1</td><td rowspan=1 colspan=1>7.5</td></tr><tr><td rowspan=1 colspan=1>12.8</td><td rowspan=1 colspan=1>13.5</td><td rowspan=1 colspan=1>16.3</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>14.3</td><td rowspan=1 colspan=1>17.1</td></tr><tr><td rowspan=1 colspan=2>GLC-SLAM (Ours) 12.9</td><td rowspan=1 colspan=1>7.9</td><td rowspan=1 colspan=1>6.3</td><td rowspan=1 colspan=1>10.5</td><td rowspan=1 colspan=1>11.0</td><td rowspan=1 colspan=1>6.3</td><td rowspan=1 colspan=1>9.2</td></tr></table>

TABLE IV

RENDERING PERFORMANCE ON REPLICA [29].
<table><tr><td>Metric</td><td>ESLAM [9]</td><td>Point- SLAM [31]</td><td>SplaTAM [15]</td><td>Photo- SLAM [29]</td><td>Ours</td></tr><tr><td>PSNR â</td><td>27.80</td><td>35.17</td><td>34.11</td><td>34.96</td><td>41.07</td></tr><tr><td>SSIM â</td><td>0.921</td><td>0.975</td><td>0.970</td><td>0.942</td><td>0.995</td></tr><tr><td>LPIPS â</td><td>0.245</td><td>0.124</td><td>0.100</td><td>0.059</td><td>0.021</td></tr></table>

## B. Tracking Evaluation

We report the camera tracking performance in Tables I to III. On the Replica dataset, our approach outperforms all competing techniques, achieving a 26% improvement in average accuracy over the second-best method. On the TUM-RGBD dataset, our method surpasses all 3DGS-based approaches except Photo-SLAM [18], which incorporates ORB-SLAM [1] tracker. On the ScanNet dataset, our method achieves the highest pose accuracy among all 3DGS-based baselines, showing the effectiveness of our proposed loop closure strategy in reducing accumulated tracking errors in real-world environments.

## C. Mapping Evaluation

Tab. IV compares rendering performance on the Replica dataset and shows that our approach achieves superior results on all three evaluated metrics compared to competing methods. We evaluate the reconstruction performance on both ScanNet and Replica. As shown in Fig. 3, our method accurately recovers geometric details and mitigates map drift as highlighted in red boxes, especially in edge areas. In Tab. V, we present a quantitative comparison where GLC-SLAM shows competitive performance against 3GDS-based methods but falls behind NeRF-based methods due to its limited hole-filling capability.

## D. Runtime and Memory Analysis

In Tab. VI we compare runtime and memory usage on the Replica office0 scene. We report both per-iteration and per-frame runtime profiled on a RTX 3090 GPU. Our method achieves the fastest per-iteration and comparable per-frame running speed while maintaining the lowest GPU memory consumption.

Point-SLAM [31]  
GO-SLAM [21]  
Gaussian-SLAM [16]  
GLC-SLAM (Ours)  
Ground Truth  
<!-- image-->  
Fig. 3. Mesh Evaluation on ScanNet [24]. The red boxes show map drift or poor details.

TABLE V  
RECONSTRUCTION PERFORMANCE ON REPLICA [29].
<table><tr><td>Method</td><td>Metric</td><td>rm0</td><td>rml</td><td>rm2</td><td>offo</td><td>offl</td><td>off2</td><td>off3</td><td>off4 Avg.</td><td></td></tr><tr><td>NeRF-based</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NICE-</td><td>Depth L1 [cm] â 1.81</td><td></td><td>1.44</td><td>2.04</td><td>1.39</td><td>1.76</td><td>8.33</td><td>4.99</td><td>2.01</td><td>2.97</td></tr><tr><td>SLAM [8]</td><td>F1_ [%] â</td><td>45.0</td><td>44.8</td><td>43.6</td><td>50.0</td><td>51.9</td><td>39.2</td><td>39.9</td><td>36.5</td><td>43.9</td></tr><tr><td>&quot;Voxx--</td><td>Depth L1 [cm] â</td><td>1.09</td><td>â1.90</td><td>2.21</td><td>2.32</td><td>3.40</td><td>4.19</td><td>22.96</td><td>1.61</td><td>2.46</td></tr><tr><td>Fusion [30]</td><td>F1 [%]â</td><td>17.3</td><td>33.4</td><td>24.0</td><td>43.0</td><td>31.8</td><td>21.8</td><td>17.3</td><td>22.0</td><td>26.3</td></tr><tr><td>ESLAM [9]</td><td>Depth L1 [cm] â</td><td>0.97</td><td>-1.07</td><td>1.28</td><td>0.86</td><td>1.26</td><td>1.71</td><td>1.43</td><td>1.06</td><td>1.18</td></tr><tr><td></td><td>F1_ [%]â</td><td>81.0</td><td>82.2</td><td>83.9</td><td>78.4</td><td>75.5</td><td>77.1</td><td>75.5</td><td>79.1</td><td>79.1</td></tr><tr><td>Point-</td><td>Depth L1 [cm] â</td><td>0.53</td><td></td><td>0.220.46</td><td>0.30</td><td>0.57</td><td>0.49</td><td>0.51</td><td>0.46</td><td>0.44</td></tr><tr><td>SLAM [31]</td><td>F1 [%]â</td><td>86.9</td><td>92.3</td><td>90.8</td><td>93.8</td><td>91.6</td><td>89.0</td><td>88.2</td><td>85.6</td><td>89.8</td></tr><tr><td>GO-</td><td>Depth L[cm]â</td><td>4.56</td><td>1.97</td><td>3.43</td><td>2.47</td><td>3.03</td><td>10.3</td><td>7.31</td><td>4.34</td><td>4.68</td></tr><tr><td>SLAM [21]</td><td>F1 [%] â</td><td>69.9</td><td></td><td>34.4 59.7</td><td>_46.5</td><td>40.8</td><td>51.0</td><td>64.6</td><td>50.7</td><td>52.2</td></tr><tr><td>3DGS-based</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SplaTAM [15]</td><td>Depth L1 [cm] â</td><td>0.43</td><td>0.38</td><td>0.54</td><td>0.44</td><td>0.66</td><td>1.05</td><td>1.60</td><td>0.68</td><td>0.72</td></tr><tr><td></td><td>F1 [%] â</td><td>89.3</td><td>88.2</td><td>88.0</td><td>91.7</td><td>90.0</td><td>85.1</td><td>77.1</td><td>80.1</td><td>86.1</td></tr><tr><td>Gaussian-</td><td>Depth L1 [cm] â</td><td>0.61</td><td>0.25</td><td>0.54</td><td>0.50</td><td>0.52</td><td>0.98</td><td>1.63</td><td>0.42</td><td>0.68</td></tr><tr><td>SLAM [16]</td><td>F1 [%] â</td><td>88.8</td><td>91.4</td><td>90.5</td><td>91.7</td><td>90.1</td><td>87.3</td><td>84.2</td><td>87.4</td><td>88.9</td></tr><tr><td>Ours</td><td>Depth L1 [cm] </td><td>0.57</td><td>Â¯0.24</td><td>0.50</td><td>0.44</td><td>0.48</td><td>1.06</td><td>1.85</td><td>0.45</td><td>0.0</td></tr><tr><td></td><td>F1 [%] â</td><td>89.3</td><td>91.3</td><td>90.5</td><td>92.3</td><td>90.0</td><td>87.7</td><td>84.4</td><td>87.3</td><td>89.1</td></tr></table>

In Tab. VII we ablate the effectiveness of loop closure and keyframe selection for the tracking and mapping performance on Replica room0 scene. The results indicate that the absence of loop closure significantly degrades tracking accuracy and reduces robustness. We also test our method using random keyframe selection . In contrast, our uncertaintyminimized strategy enhances the optimization process by incorporating more valuable keyframes, which is crucial for achieving accurate mapping.

## VI. CONCLUSIONS

TABLE VI

We present GLC-SLAM, a dense RGB-D SLAM system which utilizes submaps of 3D Gaussians for local mapping

## E. Ablation Study

RUNTIME MEMORY PERFORMANCE ON REPLICA [29] O F F I C E0.  
TABLE VII
<table><tr><td>Method</td><td>Mapping /Iter(ms)</td><td>Mapping /Frame(s)</td><td>Tracking /Iter(ms)</td><td>Tracking /Frame(s)</td><td>Peak GPU Use(GiB)</td></tr><tr><td>NICE-SLAM [8]</td><td>70</td><td>4.43</td><td>20</td><td>1.76</td><td>10.4</td></tr><tr><td>ESLAM [9]</td><td>36</td><td>0.62</td><td>17</td><td>0.14</td><td>17.5</td></tr><tr><td>Point-SLAM [31]</td><td>41</td><td>2.56</td><td>20</td><td>0.85</td><td>7.3</td></tr><tr><td>SplaTAM [15]</td><td>80</td><td>4.81</td><td>66</td><td>2.65</td><td>10.5</td></tr><tr><td>GLC-SLAM (Ours)</td><td>18</td><td>0.80</td><td>16</td><td>1.07</td><td>7.0</td></tr></table>

ABLATION STUDY ON REPLICA [29] R O O M0. LC AND KF INDICATE LOOP CLOSURE AND KEYFRAME.
<table><tr><td>LC</td><td>KF Selection</td><td>ATE [cm]</td><td>Depth L1 [cm]</td><td>F1 [%]</td></tr><tr><td></td><td></td><td>0.29</td><td>0.61</td><td>88.8</td></tr><tr><td></td><td></td><td>0.26</td><td>0.61</td><td>89.0</td></tr><tr><td>ÃÃ&gt;</td><td>xx</td><td>0.27</td><td>0.60</td><td>89.1</td></tr><tr><td></td><td></td><td>0.20</td><td>0.57</td><td>89.3</td></tr></table>

and tracking and a pose graph for global pose and map optimization. The proposed loop closure module efficiently reduces accumulated errors and map drift thanks to the hierarchical loop detection and rapid map updates. To further improve the robustness of submap optimization, we design a uncertainty-minimized keyframe selection strategy to select keyframes observing more informative 3D Gaussians. Our experiments show that GLC-SLAM leverages the benefit of the 3D Gaussian representation and equips it with loop closure to demonstrate superior tracking and rendering performance as well as competitive mapping accuracy on various datasets.

## ACKNOWLEDGMENT

This work was supported by the Key R&D Program of Zhejiang Province (No. 2023C01181)

[1] C. Campos, R. Elvira, J. J. G. RodrÃ­guez, J. M. Montiel, and J. D. TardÃ³s, âOrb-slam3: An accurate open-source library for visual, visualâinertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[2] R. Mur-Artal and J. D. TardÃ³s, âOrb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras,â IEEE transactions on robotics, vol. 33, no. 5, pp. 1255â1262, 2017.

[3] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison, âElasticfusion: Dense slam without a pose graph.â in Robotics: science and systems, vol. 11. Rome, Italy, 2015, p. 3.

[4] K. Wang, F. Gao, and S. Shen, âReal-time scalable dense surfel mapping,â in 2019 International conference on robotics and automation (ICRA). IEEE, 2019, pp. 6919â6925.

[5] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon, âKinectfusion: Real-time dense surface mapping and tracking,â in 2011 10th IEEE international symposium on mixed and augmented reality. Ieee, 2011, pp. 127â136.

[6] X. Yang, Y. Ming, Z. Cui, and A. Calway, âFd-slam: 3-d reconstruction using features and dense matching,â in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 8040â8046.

[7] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, âimap: Implicit mapping and positioning in real-time,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 6229â6238.

[8] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 12 786â12 796.

[9] M. M. Johari, C. Carta, and F. Fleuret, âEslam: Efficient dense slam system based on hybrid representation of signed distance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 17 408â17 419.

[10] Z. Xu, J. Niu, Q. Li, T. Ren, and C. Chen, âNid-slam: Neural implicit representation-based rgb-d slam in dynamic environments,â arXiv preprint arXiv:2401.01189, 2024.

[11] H. Wang, J. Wang, and L. Agapito, âCo-slam: Joint coordinate and sparse parametric encodings for neural real-time slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 293â13 302.

[12] Z. Xin, Y. Yue, L. Zhang, and C. Wu, âHero-slam: Hybrid enhanced robust optimization of neural slam,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 8610â8616.

[13] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[14] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[15] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[16] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, âGaussian-slam: Photo-realistic dense slam with gaussian splatting,â arXiv preprint arXiv:2312.10070, 2023.

[17] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[18] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Real-time simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 584â 21 593.

[19] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, âGsslam: Dense visual slam with 3d gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 595â19 604.

[20] Z. Peng, T. Shao, Y. Liu, J. Zhou, Y. Yang, J. Wang, and K. Zhou, âRtg-slam: Real-time 3d reconstruction at scale using gaussian splatting,â in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1â11.

[21] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, âGo-slam: Global optimization for consistent 3d instant reconstruction,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3727â3737.

[22] C.-M. Chung, Y.-C. Tseng, Y.-C. Hsu, X.-Q. Shi, Y.-H. Hua, J.-F. Yeh, W.-C. Chen, Y.-T. Chen, and W. H. Hsu, âOrbeez-slam: A real-time monocular visual slam with orb features and nerf-realized mapping,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 9400â9406.

[23] Y. Mao, X. Yu, Z. Zhang, K. Wang, Y. Wang, R. Xiong, and Y. Liao, âNgel-slam: Neural implicit representation-based global consistent low-latency slam system,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 6952â6958.

[24] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. NieÃner, âScannet: Richly-annotated 3d reconstructions of indoor scenes,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828â5839.

[25] Y. Tang, J. Zhang, Z. Yu, H. Wang, and K. Xu, âMips-fusion: Multi-implicit-submaps for scalable and robust online neural rgb-d reconstruction,â ACM Transactions on Graphics (TOG), vol. 42, no. 6, pp. 1â16, 2023.

[26] J. Hu, X. Chen, B. Feng, G. Li, L. Yang, H. Bao, G. Zhang, and Z. Cui, âCg-slam: Efficient dense rgb-d slam in a consistent uncertainty-aware 3d gaussian field,â arXiv preprint arXiv:2403.16095, 2024.

[27] C. Jiang, H. Zhang, P. Liu, Z. Yu, H. Cheng, B. Zhou, and S. Shen, âH _{2}-mapping: Real-time dense mapping using hierarchical hybrid representation,â IEEE Robotics and Automation Letters, 2023.

[28] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE transactions on image processing, vol. 13, no. 4, pp. 600â612, 2004.

[29] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[30] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, âVoxfusion: Dense tracking and mapping with voxel-based neural implicit representation,â in 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2022, pp. 499â507.

[31] E. SandstrÃ¶m, Y. Li, L. Van Gool, and M. R. Oswald, âPointslam: Dense neural point cloud-based slam,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 433â18 444.

[32] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of rgb-d slam systems,â in 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012, pp. 573â580.

[33] A. Dai, M. NieÃner, M. ZollhÃ¶fer, S. Izadi, and C. Theobalt, âBundlefusion: Real-time globally consistent 3d reconstruction using onthe-fly surface reintegration,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, p. 1, 2017.

[34] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.

[35] W. E. Lorensen and H. E. Cline, âMarching cubes: A high resolution 3d surface construction algorithm,â in Seminal graphics: pioneering efforts that shaped the field, 1998, pp. 347â353.