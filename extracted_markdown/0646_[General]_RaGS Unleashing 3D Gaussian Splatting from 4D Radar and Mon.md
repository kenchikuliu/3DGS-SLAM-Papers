# RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cues for 3D Object Detection

Xiaokai Bai1, Chenxu Zhou2, Lianqing Zheng3, Si-Yuan Cao1, Jianan Liu4, Xiaohan Zhang1, Yiming Li1, Zhengzhuang Zhang5, Hui-liang Shen1

1College of Information Science and Electronic Engineering, Zhejiang University 2College of Computer Science and Technology, Zhejiang University 3School of Automotive Studies, Tongji University 4Momoni AI, Gothenburg, Sweden 5College of Energy Engineering, Zhejiang University

shawnnnkb@gmail.com

## Abstract

4D millimeter-wave radar is a promising sensing modality for autonomous driving, yet effective 3D object detection from 4D radar and monocular images remains challenging. Existing fusion approaches either rely on instance proposals lacking global context or dense BEV grids constrained by rigid structures, lacking a flexible and adaptive representation for diverse scenes. To address this, we propose RaGS, the first framework that leverages 3D Gaussian Splatting (GS) to fuse 4D radar and monocular cues for 3D object detection. 3D GS models the scene as a continuous field of Gaussians, enabling dynamic resource allocation to foreground objects while maintaining flexibility and efficiency. Moreover, the velocity dimension of 4D radar provides motion cues that help anchor and refine the spatial distribution of Gaussians. Specifically, RaGS adopts a cascaded pipeline to construct and progressively refine the Gaussian field. It begins with Frustum-based Localization Initiation (FLI), which unprojects foreground pixels to initialize coarse Gaussian centers. Then, Iterative Multimodal Aggregation (IMA) explicitly exploits image semantics and implicitly integrates 4D radar velocity geometry to refine the Gaussians within regions of interest. Finally, Multi-level Gaussian Fusion (MGF) renders the Gaussian field into hierarchical BEV features for 3D object detection. By dynamically focusing on sparse and informative regions, RaGS achieves object-centric precision and comprehensive scene perception. Extensive experiments on View-of-Delft, TJ4DRadSet, and OmniHD-Scenes demonstrate its robustness and SOTA performance. Code will be released.

## 1. Introduction

<!-- image-->  
Figure 1. 4D radar and camera fusion pipelines. (a) Instancebased fusion relies on 2D detection, limiting scene understanding. (b) BEV-based fusion uses predefined grids, causing inefficiencies in background modeling and fixed anchor sampling. (c) Our Gaussian-based fusion offers adaptive sparse objects attention while preserving scene perception.

Autonomous driving requires accurate 3D perception for safe decision-making [20]. Recently, 4D millimeter-wave radar has emerged as a highly promising sensor, known for its robustness in challenging environments, long-range detection capabilities, and ability to capture both velocity and elevation data. Cameras, on the other hand, provide highresolution semantic information. The complementary nature of these two modalities is crucial for enhancing perception in autonomous driving, particularly for 3D object detecting. However, effective as well as efficient fusion across modalities remains a challenge, as the task is inherently sparse while still requiring a comprehensive understanding of the entire scene.

Recent research has focused on either instance-based or BEV-based fusion for 4D radar and camera 3D object detection. Instance-based approaches [11, 19, 21] rely on 2D detectors to generate proposals, which are refined by aligning radar features. While these approaches directly maintain the 3D detection flow by focusing on sparse objects, they lack global scene understanding and are constrained by cascaded network designs, as shown in Fig. 1(a). On the other hand, BEV-based approaches [1, 12, 15, 29, 41] project multimodal features into a predefined, fixed top-down space, enabling global reasoning. However, these approaches suffer from rigid voxelization, fixed anchor positions for sampling image semantics, and inefficiencies due to excessive background aggregation, as illustrated in Fig. 1(b). While they offer promising performance in building scene perception, they are misaligned with the sparse nature of 3D object detection tasks, in which focusing primarily on foreground objects is significantly more efficient.

In parallel, 3D Gaussian Splatting (GS) [10] has emerged as a compact, continuous scene representation. Originally developed for neural rendering, GS models scenes as collections of anisotropic Gaussians optimized in 3D space, offering physical interpretability, sparsity, and flexibility. Recent extensions have applied GS to dynamic scenes [35] and LiDAR-camera fusion [7]. However, these works have primarily focused on rendering or occupancy prediction [9], without exploring its potential for multi-modal fusion or 3D object detection, despite its natural alignment with the flexible needs of such tasks. Unlike 2D instance-based or BEVbased approaches, GS can dynamically allocate resources and adjust its attention, making it particularly efficient for identify interested objects within entire scene.

To address the limitations of existing fusion approaches and better adapt to the sparse nature of 3D object detection, we propose RaGS, a novel framework that leverages 3D Gaussian Splatting to fuse 4D radar and monocular cues, as presented in Fig. 1(c). Unlike traditional dense voxel grids, RaGS represents the scene as a continuous field of 3D Gaussians, allowing for flexible aggregation without the constraints of fixed grid resolutions and fixed anchor sampling. By dynamically allocating resources iteratively, RaGS focuses on sparse foreground objects while preserving global scene perception through Gaussiansâs continuous probability distribution. Specifically, we follow a cascaded pipeline composed of three key modules, as presented in Fig. 2. First, the Frustum-based Localization Initiation (FLI) identifies foreground regions and unprojecting pixels to initialize coarse, grounded 3D Gaussian localization. Next, the Iterative Multimodal Aggregation (IMA) aggregates radar and image features within frustum regions, refining Gaussian positions towards objects and enhancing both semantic and geometric expressiveness. Finally, the Multi-level Gaussian Fusion (MGF) integrates features rendered from Gaussians at different levels to produce a rich representation for object prediction. We evaluate RaGS on three 4D radar and camera benchmarks: View-of-Delft (VoD) [23], TJ4DRadSet [40], and OmniHD-Scenes [42], achieving state-of-the-art performance across the datasets, demonstrating the effectiveness of continuous Gaussian representation for multi-modal 3D object detection. Our contributions are summarized as

â¢ We propose RaGS, the first framework that leverages 3D Gaussian Splatting to fuse 4D radar and monocular camera inputs for 3D object detection.

â¢ We design three modules to construct and refine the Gaussian from multi-modal inputs, enabling dynamic focus on objects while maintaining scene perception.

â¢ Extensive experiments on View-of-Delft, TJ4DRadSet and OmniHD-Scenes benchmarks demonstrate the effectiveness of our RaGS.

## 2. Related Work

## 2.1. 4D Radar based 3D Object Detection

In recent years, research on 4D millimeter-wave radar has advanced rapidly, leading to the release of several benchmark datasets and an increasing focus on multi-modal fusion with cameras [22, 23, 40, 42]. Recent studies on 3D object detection using 4D radarâcamera fusion typically follow two paradigms: instance-based and BEV-based paradigms. Instance-based approaches generate 2D proposals from images, refined with radar features. Examples include CenterFusion [21] that aligns radar points with 2D boxes for better center predictions, CRAFT [11] that integrates radar range and velocity cues, and RADIANT [19] that recovers missed detections under occlusion. While effective for foreground localization, these methods depend heavily on proposal quality and lack global perception.

BEV-based approaches project multi-modal features into a top-down space for global spatial reasoning. RCFusion [41] utilizes orthographic projection for semantics but neglects depth ambiguity, whereas other approaches like LXL [29] and CRN [12] introduce various improvements for image view transformation. L4DR [8] and Interfusion [27], on the other hand, further explored the role of LiDAR and 4D radar fusion for 3D object detection. Despite their strengths, BEV approaches rely on fixed-resolution voxel grids, leading to inefficiencies, excessive background aggregation, and reduced flexibility in 3D object detection. In contrast, we propose a dynamic 3D Gaussian-based representation, modeling multi-modal features as a continuous Gaussian field. This allows for more flexible and finegrained fusion, making it particularly suitable for sparse 3D object detection.

<!-- image-->  
Figure 2. Pipeline of RaGS. RaGS consists of a Feature Extractor & Head, Frustum-based Localization Initiation (FLI), Iterative Multimodal Aggregation (IMA), and Multi-level Gaussian Fusion (MGF). The positions of the Gaussians are initialized using the FLI module, along with learnable attributes such as rotation, scale, opacity, and implicit feature embeddings. These Gaussians are then passed into the IMA module, where they are projected onto the image plane to gather semantic information. Next, they are processed as voxels using sparse convolution with height-extended radar geometry, which implicitly utilizes radar velocity to guide residuals movement. Residuals relative to regions of interest are computed iteratively, updating the positions towards sparse objects. Finally, the multi-level Gaussians are rendered into Birdâs Eye View (BEV) features and fused through MGF, followed by cross-modal fusion for 3D object detection.

## 2.2. 3D Gaussian Splatting in Autonomous Driving

3D Gaussian Splatting (GS) has emerged as a key technique for real-time 3D scene reconstruction. Initially introduced as a faster alternative to neural radiance fields for rendering, GS has extended to tasks like non-rigid motion [35], large-scale environments [43], and multi-modal fusion with LiDAR-camera [7, 38] or radar-only inputs [13].

Despite these advancements, current GS applications are predominantly in dense prediction tasks, such as rendering or occupancy prediction [9]. However, its potential for 3D object detection remains largely unexplored, even though the sparse nature of detection task inherently aligns with Gaussian flexibility. While NeRF-based approaches [25, 31] use volumetric representations for 3D detection, GS offers advantages like faster rendering and better integration with detection pipelines.

In this work, we leverage GS for sparse object modeling and scene perception. Unlike traditional BEV grid models that rely on fixed anchors for feature sampling, our method utilizes GS to enable dynamic feature aggregation by adaptively refining Gaussian positions towards targets, which perfectly aligns with the 3DGS foreground fitting insight pioneered by indoor models (3DGS-DET [2], Gaussian-Det [33]), although RaGS is designed for outdoor environments.

## 3. Method

Fig. 2 illustrates the overall architecture of our RaGS framework, which consists of four main components: Feature Extractor & Head, Frustum-based Localization Initiation (FLI), Iterative Multimodal Aggregation (IMA), and Multilevel Gaussian Fusion (MGF). RaGS encodes image and radar depth into features, and radar point clouds into sparse geometric features. It then initializes 3D Gaussians via FLI, refines them with radar geometry and image semantics via IMA, and finally renders these Gaussians into multi-scale BEV features via MGF for 3D object detection.

## 3.1. Feature Extractor

The feature extractor processes image and 4D radar. For the monocular image, we employ a ResNet-50 backbone with an FPN to extract and fuse multi-scale features C â $\mathbb { R } ^ { H \times W \times C }$ , which are further fused with sparse radar depth $\textbf { S } ~ \in ~ \mathbb { R } ^ { H \times W }$ to estimate depth probability $\begin{array} { r l } { \mathbf { D } ^ { \mathrm { p r o b . } } } & { { } \in } \end{array}$ $\mathbb { R } ^ { H \times W \times D }$ and enhanced feature map $\mathbf { F } ^ { \mathrm { 2 D } } \in \mathbb { R } ^ { H \times W \times C } .$ Here, C and D denotes the number of channel and depth bins, and (H, W ) denotes the resolution. The process can be expressed as $( { \bf F } ^ { \mathrm { 2 D } } , { \bf D } ^ { \mathrm { p r o b . } } ) = \mathsf { C o n v } \big ( \mathsf { C o n c a t } ( { \bf C } , { \bf S } ) \big )$ , where Conv denotes convolution that output (C + D) channel feature map.

Then, the metric depth D is computed by summing over predefined depth bins $\begin{array} { r } { \mathbf { \dot { D } } = \sum _ { d = 1 } ^ { D } \mathbf { \dot { P } } _ { d } \cdot d , } \end{array}$ where $P _ { d }$ represents the probability of each depth bin d. A lightweight segmentation network is further used to predict foreground regions from image features $\mathbf { F } ^ { 2 \mathrm { D } }$ , producing segmentation logits $\textbf { L } \in \ \mathbb { R } ^ { H \times W }$ For the 4D radar point cloud, to fully utilize its RCS and velocity characteristics, we pillarize it and apply RCS-Vel-Aware sparse convolution to extract channel-specific information followed [41]. These channel-splited features are then stacked into sparse pillars Fpillar $\in \overset { \mathbf { \hat { \mathbb { R } } } ^ { G \times C } } { \mathbb { R } ^ { G \times C } }$ , where G is the number of pillars. This separate extraction helps avoid confusion and guides the subsequent feature aggregation of the Gaussian.

## 3.2. Gaussian Aggregation

Frustum-based Localization Initiation (FLI). Gaussian initialization is crucial for guiding downstream feature aggregation. Previous approaches use learnable embeddings to initialize localization based on dataset-specific distributions. In this work, we initialize the position of each Gaussian using monocular cues, which provide a coarse but structured foreground prior focused on sparse objects. Thus, the Gaussians are modeled as a combination of two components: explicit physical properties (position P, rotation R, scale S, opacity O) and implicit feature embeddings. The position is inferred from the FLI.

Specifically, we begin by selecting the top-K foreground pixels from the segmentation logits L and metric depth D. Each selected foreground pixel $( u , v )$ with its corresponding depth $d = \mathbf { D } ( u , v )$ is then unprojected into 3D space:

$$
\mathbf { P } _ { \mathrm { u n p r o j } } = d \cdot \mathbf { K } ^ { - 1 } ( u , v , 1 ) ^ { T } ,\tag{1}
$$

where K is the camera intrinsic matrix. Furthermore, we also include 4D radar points $\mathbf { P } _ { \mathrm { r a d a r } }$ and randomly sample points in the frustum space to enrich spatial coverage and eliminate potential instability of foreground identification. For the random sampling process, we begin by defining the candidate points $\mathbf { P } _ { \mathrm { c a n d } }$ as the centers of predefined voxels in the radar coordinate system. These points are projected onto the image space using the cameraâs extrinsic parameters: $( u d , v d , d ) = \mathbf { K } \left( \mathbf { R P } _ { \mathrm { c a n d } } + \mathbf { T } \right)$ , where R and T are the rotation matrix and translation vector from radar to camera coordinates. We select points that lie within the image boundaries using the condition $\mathbf { I } = \{ i \ | \ 0 \leq u _ { i } < W , 0 \leq$ $v _ { i } < H \}$ , and apply Furthest Sampling:

$$
\mathbf { P } _ { \mathrm { s a m p l e } } = \mathrm { F u r t h e s t S a m p l i n g } \left( \mathbf { P } _ { \mathrm { c a n d } } [ \mathbf { I } ] \right) ,\tag{2}
$$

The final position P is a concatenation of the unprojected foreground points, 4D radar points, and sampled points:

$$
\mathbf { P } = \mathrm { C o n c a t } ( \mathbf { P } _ { \mathrm { u n p r o j } } , \mathbf { P } _ { \mathrm { s a m p l e } } , \mathbf { P } _ { \mathrm { r a d a r } } ) \in \mathbb { R } ^ { N \times 3 } ,\tag{3}
$$

where N is the number of Gaussian anchors. This ensures that all initialized Gaussians are within the field of view (FoV) and offer coarse-but-structured initial localization, unlike BEV-based approaches that perceive the entire scene. The fully learnable explicit properties include a rotation vector $\dot { \mathbf { R } } \in \mathbb { R } ^ { N \times 4 }$ , scale $\mathbf { S } ^ { \mathbf { \bar { \alpha } } } \in \mathbf { \mathbb { R } } ^ { N \times 3 }$ , and opacity $\mathbf { O } \in \mathbb { R } ^ { N \times 1 }$ , which are further concatenated with the position $\mathbf { P } \in \mathbb { R } ^ { N \times 3 }$ to get final explicit physical attributes FE:

<!-- image-->

Figure 3. Procedure of Iterative Multimodal Aggregation (IMA). IMA involves the iterative aggregation of multi-modal features, followed by the updating of Gaussian locations within the frustum.  
<!-- image-->  
Figure 4. Dynamic Object Attention of RaGS. We visualize activated Gaussians (approximately 30% of total) in the scene. RaGS focuses on sparse foreground objects while maintaining scene understanding.

$$
\mathbf { F } ^ { \mathrm { E } } = \mathtt { C o n c a t } ( \mathbf { P } , \mathbf { R } , \mathbf { S } , \mathbf { O } ) \in \mathbb { R } ^ { N \times 1 1 } .\tag{4}
$$

Additionally, we assign each Gaussian with a learnable embedding $\mathbf { F } ^ { \mathrm { I } } \in \mathbb { R } ^ { N \times C }$ , which serves as a query for aggregating image semantics and radar geometry. As a result, each Gaussian is represented as $\mathbf { G } = \{ \mathbf { F } ^ { \mathrm { E } } , \mathbf { F } ^ { \mathrm { I } } \}$

Iterative Multimodal Aggregation (IMA). To enrich the Gaussians with multi-modal information and progressively guide them toward the foreground regions, we iteratively aggregate features from both image and radar domains. In particular, the spatial cues from radar serve as anchors for residual position updates, while the velocity features provide implicit guidance for gradual convergence. This iterative process outputs the final set of M refined Gaussians, as illustrated in Fig. 3.

For semantic aggregation, instead of directly projecting Gaussians into 2D image space and applying 2D deformable attention for semantic aggregation, we first construct a depth-aware 3D image feature space out product image feature $\mathbf { F } ^ { 2 \mathrm { D } }$ with depth probability $\mathbf { D } ^ { \mathrm { p r o b } }$ .. We then perform 3D deformable cross attention (3D-DCA) within it, enabling the Gaussians to interact with semantically and geometrically aligned image features in a spatially consistent manner. Specifically, given 3D Gaussian query $\mathbf { F } ^ { \mathrm { I } }$ at location P, we aggregate image semantics by

$$
\mathbf { F } ^ { \mathrm { I } } = \sum _ { n = 1 } ^ { T } \mathbf { A } _ { n } \mathbf { W } \cdot \phi ( \mathbf { F } ^ { \mathrm { 2 D } } \otimes \mathbf { D } ^ { \mathrm { p r o b . } } , \mathcal { P } ( \mathbf { P } ) + \Delta \mathbf { q } ) ,\tag{5}
$$

where n indexes $T$ sampled offsets around P, P(P) denotes projection function, $\Delta \mathbf { q } \in \mathbb { R } ^ { 3 }$ denotes learnable offsets, and $\phi ( \cdot )$ performs trilinear interpolation to extract features. The attention weights $\mathbf { A } _ { n } \in [ 0 , 1 ]$ and projection matrix W are learned to guide the aggregation. The utilization of 3D deformable cross-attention (3D-DCA) [1] facilitates coherent interaction between Gaussians in 3D space and image features in the perspective view.

Consequently, we further enhance the implicit feature representation using sparse convolution, which integrates radar velocity as motion cues to refine the spatial distribution of Gaussians. Specifically, each optimized Gaussian point is treated as a point located at its center, and the resulting point cloud is voxelized into a sparse grid $\mathbf { V } ^ { \mathrm { g s } } \in \mathbb { R } ^ { N \times C }$ Then, we incorporate radar pillars $\mathbf { F } ^ { \mathrm { p i l l a r } }$ consisting of rich geometry to enable synergistic multi-modal learning. Since radar pillars lack vertical resolution, we replicate each pillar along the height dimension to get Vradar $\in \mathbf { \mathbb { R } } ^ { ( G \times Z ) \times C }$ , here Z is the voxel number along the height dimension, which can be expressed as $\mathbf { V } ^ { \mathrm { r a d a r } } = \mathrm { R e p e a t } ( \mathbf { F } ^ { \mathrm { p i l l a r } } )$ . These radar voxels are concatenated with $\mathbf { V } ^ { \mathrm { g s } }$ , and the RCS-velocity aware pillars provide clear physical meaning, thereby better assisting the Gaussians. By concatenating both modalities in a unified voxel space and applying sparse convolution, we treat each Gaussian as voxels and enable deep interaction between radar geometry and image semantics. This process is formulated as

$$
\mathbf F ^ { \mathrm { I } } \gets \mathbf V ^ { \mathrm { g s } } = \mathtt { S p c o n v } ( \mathtt { C o n c a t } ( \mathbf V ^ { \mathrm { g s } } , \mathbf V ^ { \mathrm { r a d a r } } ) ) [ : N ] ,\tag{6}
$$

where Spconv denotes sparse convolution. Finally, to maintain consistency with the original Gaussian set, we only output former N voxels using [: N ] and reassign to $\mathbf { F } ^ { \mathrm { I } } .$ , since the order remains unchanged. After updating the implicit feature, we begin explicit feature refinement as below.

Specifically, to achieve the finer foreground modeling required for 3D object detection, we refine each Gaussian localization within the frustum, which allows a more precise match with the monocular settings. To achieve this, we first reproject each Gaussian into frustum space and then using an MLP conditioned on its implicit feature $\mathbf { F } ^ { \mathrm { I } }$ and transformed position P(P) to predict a residual spatial offset as

$$
\Delta \mathbf { p } = ( \Delta h , \Delta w , \Delta d ) = \mathbb { M L P } \big ( \mathrm { C o n c a t } ( \mathbf { F } ^ { \mathrm { I } } , \mathcal { P } ( \mathbf { P } ) ) \big ) ,\tag{7}
$$

where $\Delta h , \Delta w , \Delta d$ corresponds to vertical, horizontal, and depth adjustments in perspective view and $\mathcal { P }$ denotes the projection from 3D space to the image frustum. The predicted residual is then added to frustum-space location of each Gaussian before being transformed back to 3D space, which ensures consistency with preceding modules. This

<table><tr><td>Approach</td><td>Res.</td><td>Input</td><td>mAP(%)</td><td>ODS(%)</td></tr><tr><td>PointPillars</td><td>-</td><td>R</td><td>23.82</td><td>37.20</td></tr><tr><td>RadarPillarNet</td><td>-</td><td>R</td><td>24.88</td><td>37.81</td></tr><tr><td>Lift, Splat, Shoot</td><td>544x960</td><td>C</td><td>22.44</td><td>26.01</td></tr><tr><td>BEVFormer</td><td>544x960</td><td>C</td><td>26.49</td><td>28.10</td></tr><tr><td>PanoOcc</td><td>544x960</td><td>C</td><td>29.17</td><td>28.55</td></tr><tr><td>BEVFusion</td><td>544x960</td><td>C+R</td><td>33.95</td><td>43.00</td></tr><tr><td>RCFusion</td><td>544x960</td><td>C+R</td><td>34.88</td><td>41.53</td></tr><tr><td>RaGS (Ours)</td><td>544x960</td><td>C+R</td><td>35.88</td><td>43.45</td></tr></table>

Table 1. Comparison on the OmniHD-Scenes [42] test set.
<table><tr><td rowspan=1 colspan=4>Approach</td><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1> $\mathrm { A P } _ { 3 \mathrm { D } }$ (%)</td><td rowspan=1 colspan=1>APBEV (%)</td></tr><tr><td rowspan=1 colspan=4>ImVoxelNet (WACV 2022) [24]</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>14.96</td><td rowspan=1 colspan=1>17.12</td></tr><tr><td rowspan=1 colspan=4>RadarPillarNet (TIM 2023) [41]SMURF (TIV 2023) [17]</td><td rowspan=1 colspan=1>RR</td><td rowspan=1 colspan=1>30.3732.99</td><td rowspan=1 colspan=1>39.2440.98</td></tr><tr><td rowspan=11 colspan=4>FUTR3D (CVPR 2023) [3]BEVFusion (ICRA 2023) [18]LXL (TIV 2024) [29]RCFusion (TIM 2023) [41]LXLv2 (RAL 2025) [30]UniBEVFusion (ICRA 2025) [39]MSSF (TITS 2025) [16]HGSFusion (AAAI 2025) [6]SGDet3D (RAL 2025) [1]</td><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>32.42</td><td rowspan=1 colspan=1>37.51</td></tr><tr><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>32.71</td><td rowspan=1 colspan=1>41.12</td></tr><tr><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>36.32</td><td rowspan=1 colspan=1>41.20</td></tr><tr><td rowspan=1 colspan=3></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>33.85</td><td rowspan=1 colspan=1>39.76</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>37.32</td><td rowspan=1 colspan=1>42.35</td></tr><tr><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>37.76</td><td rowspan=1 colspan=1>42.92</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>37.97</td><td rowspan=1 colspan=1>43.11</td></tr><tr><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>37.21</td><td rowspan=1 colspan=1>43.23</td></tr><tr><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>41.82</td><td rowspan=1 colspan=1>47.16</td></tr><tr><td rowspan=1 colspan=4>RaGS (Ours)</td><td rowspan=1 colspan=1>R+C</td><td rowspan=1 colspan=1>41.95</td><td rowspan=1 colspan=1>51.04</td></tr></table>

Table 2. Comparison on the test set of TJ4DRadSet [40].
<table><tr><td rowspan=2 colspan=1>Models</td><td rowspan=1 colspan=1>View-of-Delft[23]</td><td rowspan=1 colspan=1>TJ4DRadSet[40]</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { m A P _ { E A A } }$   $\mathrm { m A P _ { D C } }$ </td><td rowspan=1 colspan=1> $\mathrm { m A P } _ { 3 \mathrm { D } }$   $\mathrm { m A P _ { B E V } }$ </td></tr><tr><td rowspan=2 colspan=1>FUTR3D (CVPR2023) [3]RaCFormer (CVPR2025) [5]</td><td rowspan=1 colspan=1>49.03  69.32</td><td rowspan=1 colspan=1>32.42  37.51</td></tr><tr><td rowspan=1 colspan=1>54.44  78.57</td><td rowspan=1 colspan=1>-       -</td></tr><tr><td rowspan=1 colspan=1>RaGS (Ours)</td><td rowspan=1 colspan=1>61.86  81.63</td><td rowspan=1 colspan=1>41.95  51.04</td></tr></table>

Table 3. Comparison with DETR-based models.

process is formulated as

$$
\mathbf { P } \gets \mathcal { P } ^ { - 1 } ( \mathcal { P } ( \mathbf { P } ) + \Delta \mathbf { p } ) .\tag{8}
$$

This iterative process, depicted in Fig. 4, progressively refines Gaussian localization for sparse objects, in contrast to the fixed grid sampling of previous BEV-based models, which suffer from inflexibility and inefficiency due to the inevitable excessive background aggregation. All other Gaussian attributes are also updated simultaneously.

Multi-level Gaussian Fusion (MGF). After obtaining multiple Gaussian representations that capture different levels of semantic and geometric abstraction, we further integrate them to form a unified, multi-level feature representation for subsequent rendering and detection. Through IMA, we obtain Gaussians set consisting of M refined Gaussians $\mathcal { G } = \{ \mathbf { G } _ { i } \} _ { i = 1 } ^ { M }$ Each Gi has N Gaussian, inheriting its explicit attributes $\left( \mathbf { P } _ { i } , \mathbf { R } _ { i } , \mathbf { S } _ { i } , \mathbf { O } _ { i } \right)$ and implicit feature $\mathbf { F } _ { i } ^ { \mathrm { I } }$ from the previous stage, which can be formally represented as $\mathbf { G } _ { i } = ( \mu _ { i } , \Sigma _ { i } , \alpha _ { i } , \mathbf { f } _ { i } )$ , where the Gaussian center $\pmb { \mu } _ { i }$ corresponds to $\mathbf { P } _ { i } ,$ the covariance $\Sigma _ { i }$ is derived from the rotation $\mathbf { R } _ { i }$ and scale $\mathbf { S } _ { i } , \alpha _ { i }$ denotes opacity (from $\mathbf { O } _ { i } )$ , and $\mathbf { f } _ { i }$ is the learned feature embedding.

<!-- image-->

Figure 5. Visualization results on the VoD validation set (first row) and TJ4DRadSet test set (second row) . Each figure corresponds to a frame. Orange and yellow boxes represent ground-truths in the perspective and birdâseye views, respectively. Green and blue boxes indicate predicted results. Zoom in for better view.
<table><tr><td rowspan="2">Approach</td><td rowspan="2">Input</td><td colspan="4">Entire Annotated Area (%)</td><td colspan="4">Driving Corridor (%)</td><td rowspan="2">FPS</td></tr><tr><td>Car</td><td>Ped</td><td>Cyclist</td><td>mAP</td><td>Car</td><td>Ped</td><td>Cyclist</td><td>mAP</td></tr><tr><td>ImVoxelNet (WACV 2022) [24]</td><td>C</td><td>19.35</td><td>5.62</td><td>17.53</td><td>14.17</td><td>49.52</td><td>9.68</td><td>28.97</td><td>29.39</td><td>11.1</td></tr><tr><td>PillarNeXt (CVPR 2023) [14]</td><td>R</td><td>30.81</td><td>33.11</td><td>62.78</td><td>42.23</td><td>66.72</td><td>39.03</td><td>85.08</td><td>63.61</td><td>-</td></tr><tr><td>RadarPillarNet (TIM 2023) [41]</td><td>R</td><td>39.30</td><td>35.10</td><td>63.63</td><td>46.01</td><td>71.65</td><td>42.80</td><td>83.14</td><td>65.86</td><td>98.8</td></tr><tr><td>CenterPoint (CVPR 2021) [37]</td><td>R</td><td>35.84</td><td>41.03</td><td>67.11</td><td>47.99</td><td>70.65</td><td>50.14</td><td>85.67</td><td>68.82</td><td>38.3</td></tr><tr><td>VoxelNeXt (CVPR 2023) [4]</td><td>R</td><td>36.98</td><td>42.37</td><td>68.15</td><td>49.17</td><td>70.95</td><td>51.85</td><td>87.33</td><td>70.04</td><td>31.6</td></tr><tr><td>SMURF (TIV 2023) [17]</td><td>R</td><td>42.31</td><td>39.09</td><td>71.50</td><td>50.97</td><td>71.74</td><td>50.54</td><td>86.87</td><td>69.72</td><td>-</td></tr><tr><td>SCKD (AAAI 2025) [32]</td><td>R</td><td>41.89</td><td>43.51</td><td>70.83</td><td>52.08</td><td>77.54</td><td>51.06</td><td>86.89</td><td>71.80</td><td>39.3</td></tr><tr><td>FUTR3D (CVPR 2023) [3]</td><td>R+C</td><td>46.01</td><td>35.11</td><td>65.98</td><td>49.03</td><td>76.98</td><td>43.10</td><td>86.19</td><td>69.32</td><td>7.3</td></tr><tr><td>BEVFusion (ICRA 2023) [18]</td><td>R+C</td><td>37.85</td><td>40.96</td><td>68.95</td><td>49.25</td><td>70.21</td><td>45.86</td><td>89.48</td><td>68.52</td><td>7.1</td></tr><tr><td>RCFusion (TIM 2023) [41]</td><td>R+C</td><td>41.70</td><td>38.95</td><td>68.31</td><td>49.65</td><td>71.87</td><td>47.50</td><td>88.33</td><td>69.23</td><td>9.0</td></tr><tr><td>RCBEVDet (CVPR 2024) [15]</td><td>R+C</td><td>40.63</td><td>38.86</td><td>70.48</td><td>49.99</td><td>72.48</td><td>49.89</td><td>87.01</td><td>69.80</td><td>-</td></tr><tr><td>ZFusion (CVPR 2025) [34]</td><td>R+C</td><td>43.89</td><td>39.48</td><td>70.46</td><td>51.28</td><td>79.51</td><td>52.95</td><td>86.37</td><td>72.94</td><td>-</td></tr><tr><td>IS-Fusion (CVPR 2024) [36]</td><td>R+C</td><td>48.57</td><td>46.17</td><td>68.48</td><td>54.40</td><td>80.42</td><td>55.50</td><td>88.33</td><td>74.75</td><td>-</td></tr><tr><td>LXL (TIV 2024) [29]</td><td>R+C</td><td>42.33</td><td>49.48</td><td>77.12</td><td>56.31</td><td>72.18</td><td>58.30</td><td>88.31</td><td>72.93</td><td>6.1</td></tr><tr><td>SGDet3D (RAL 2025) [1]</td><td>R+C</td><td>53.16</td><td>49.98</td><td>76.11</td><td>59.75</td><td>81.13</td><td>60.91</td><td>90.22</td><td>77.42</td><td>9.2</td></tr><tr><td>RaGS (Ours)</td><td>R+C</td><td>58.15</td><td>50.81</td><td>76.62</td><td>61.86</td><td>88.15</td><td>61.67</td><td>95.07</td><td>81.63</td><td>10.5</td></tr></table>

Table 4. Comparison with state-of-the-art approaches on the val set of View-of-Delft [23].

For each of N Gaussian within $\mathbf { G } _ { i } ,$ it contributes to BEV cells through differentiable Gaussian Splatting. Specifically, given a BEV pixel at position $\mathbf { q } = ( x , y )$ , its feature value is computed as the weighted accumulation of all projected Gaussians, formulated as

$$
\sum _ { n = 1 } ^ { N } \alpha _ { n } \exp \bigl ( - \frac { 1 } { 2 } ( \mathbf { q } - \pmb { \mu } _ { n , x y } ) ^ { \top } \pmb { \Sigma } _ { n , x y } ^ { - 1 } ( \mathbf { q } - \pmb { \mu } _ { n , x y } ) \bigr ) \mathbf { f } _ { n } ,\tag{9}
$$

where $\scriptstyle \mu _ { n , x y }$ and $\Sigma _ { n , x y }$ are the mean and covariance of the Gaussian projected onto the BEV plane. This operation defines a smooth, differentiable mapping from the 3D Gaussian field to the 2D BEV domain, implemented efficiently via a CUDA-based rasterizer [9]. The implicit features of the last $L \left( L { \leq } M \right)$ Gaussians are rasterized into multi-level BEV-aligned feature maps $\{ \mathbf { F } ^ { ( l ) } \in \mathbb { R } ^ { X \times Y \times C } \}$ as

$$
{ \bf F } ^ { ( l ) } = \mathtt { R a s t e r i z e } \big ( \mathbf { G } _ { M - L + l } \big ) , \quad l = 1 , \ldots , L ,\tag{10}
$$

where (X, Y ) denotes the BEV resolution, Rasterize de-

notes the rasterizer. These rendered maps form a hierarchical Gaussian representation across spatial scales. We then fuse the multi-level features via convolution as

$$
\mathbf { F } ^ { \mathrm { g s } } = \mathsf { C o n v } \Big ( \mathsf { C o n c a t } ( \mathbf { F } ^ { ( 1 ) } , \ldots , \mathbf { F } ^ { ( L ) } ) \Big ) .\tag{11}
$$

The fused Gaussian feature Fgs is further enhanced by radar-derived sparse pillars Fpillar through a cross-modal fuser, formulated as $\bar { { \bf F } } ^ { \mathrm { B E V } } = \mathrm { C M F } ( { \bf F } ^ { \mathrm { g s } } , \bar { { \bf F } } ^ { \mathrm { p i l l a r } } )$ , where CMF denotes a lightweight module composed of concatenation and convolution layers. As a result, our MGF generates a hierarchical representation that compensates for the sparse and low-resolution nature of the Gaussian from IMA.

Additionally, we render final-layer Gaussians into depth supervised by LiDAR, and predict a BEV segmentation map from fused BEV features with occupancy guidance. Crucially, while MGF rasterizes the Gaussian field into BEV space, the Gaussians themselves are dynamically refined toward object-centric regions, which allows adaptive aggregation that prioritizes sparse foreground objects and mitigates the background redundancy of grid-based BEV methods.

## 3.3. Detection Head and Loss Function

On one hand, following [1], we apply the depth loss ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ and the perspective segmentation loss $\mathcal { L } _ { \mathrm { s e g } }$ to supervise the raw feature extraction for pretraining, expressed as

$$
\mathcal { L } _ { \mathrm { p r e t r a i n } } = \mathcal { L } _ { \mathrm { d e p t h } } + \mathcal { L } _ { \mathrm { s e g } } .\tag{12}
$$

On the other hand, we adopt the 3D object detection loss $\mathcal { L } _ { \mathrm { d e t } }$ following [41], along with auxiliary rendering losses: the rendered depth loss in the perspective view ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ render, and the semantic segmentation loss in the BEV $\mathcal { L } _ { \mathrm { s e g . } }$ render. The total joint training loss is defined as

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { d e t } } + \lambda \big ( \mathcal { L } _ { \mathrm { d e p t h . r e n d e r } } + \mathcal { L } _ { \mathrm { s e g . r e n d e r } } \big ) ,\tag{13}
$$

where Î» is the hyperparameter balancing detection and auxiliary tasks. In this work, we simply set $\lambda ~ = ~ 0 . 1$ The perspective and BEV segmentation ground-truths can be simply generated through detectron2 [28] and 3D bounding boxes. See supplementary material for more details.

## 3.4. Implementation Details

Datesets and Evaluation Metrics. We evaluate our model on three 4D radar benchmarks: VoD [23], TJ4DRadSet [40], and OmniHD-Scenes [42]. The VoD dataset, collected in Delft, includes 5,139 training frames, 1,296 validation frames, and 2,247 test frames. The TJ4DRadSet, captured in Suzhou, contains 7,757 frames with 5,717 for training and 2,040 for testing, posing challenges due to complex scenarios such as nighttime and glare. The OmniHD-Scenes dataset features multimodal data from six cameras, six 4D radars, and a 128-beam LiDAR, with 11,921 keyframes annotated, consisting of 8,321 for training, 3,600 for testing.

We follow the official evaluation protocols for each dataset. For VoD, we report 3D Average Precision (AP) across the full area and drivable corridor. For TJ4DRadSet, we evaluate 3D AP and BEV AP within 70 meters of the radar origin. For OmniHD-Scenes, we use mean Average Precision (mAP) and the OmniHD-Scenes Detection Score (ODS) for detection within a Â±60 m longitudinal and Â±40 m lateral range around the ego vehicle.

Network Settings and Training details. For all datasets, the final voxel is set to a cube of size 0.32 m, and the image sizes are 800Ã1280 for VoD, 640Ã800 for TJ4DRadSet, and 544Ã960 for OmniHD-Scenes. Anchor size and point cloud range are kept as in [42]. The models are trained on 4 NVIDIA GeForce RTX 4090 GPUs with a batch size of 4 per GPU. AdamW is used as the optimizer, with 12 epochs for pretraining and 24 epochs for joint training.

<table><tr><td>Baseline</td><td>w/FLI</td><td>w/IMA</td><td>w/ MGF</td><td> $\underline { { \mathrm { m A P } _ { \mathrm { E A A } } } }$ </td><td> $\mathrm { \ m A P _ { D C } }$ </td></tr><tr><td>â</td><td></td><td></td><td></td><td>55.33</td><td>72.32</td></tr><tr><td>â</td><td>â</td><td></td><td></td><td>57.40</td><td>75.80</td></tr><tr><td>â</td><td>â</td><td>â</td><td></td><td>59.12</td><td>76.68</td></tr><tr><td>â</td><td>â</td><td>â</td><td>â</td><td>59.45</td><td>76.98</td></tr></table>

Table 5. Overall ablation of RaGS.
<table><tr><td rowspan="2">Baseline</td><td colspan="3">FLI</td><td rowspan="2"> $\mathrm { \ m A P _ { E A A } }$ </td><td rowspan="2"> $\mathrm { \ m A P _ { D C } }$ </td></tr><tr><td>Frustum</td><td>Radar</td><td>Depth</td></tr><tr><td>â</td><td></td><td></td><td></td><td>55.33</td><td>72.32</td></tr><tr><td>â</td><td>â</td><td></td><td></td><td>56.26</td><td>73.10</td></tr><tr><td>â</td><td>â</td><td></td><td></td><td>56.72</td><td>75.78</td></tr><tr><td>â</td><td>â</td><td>&gt;&gt;</td><td>â</td><td>57.40</td><td>75.80</td></tr></table>

Table 6. Ablation study of FLI on the VoD dataset.
<table><tr><td rowspan="2">FLI</td><td colspan="3">IMA</td><td rowspan="2"> $\mathrm { \ m A P _ { E A A } }$ </td><td rowspan="2"> $\mathrm { \ m A P _ { D C } }$ </td></tr><tr><td>3D-DCA</td><td>Pillars</td><td>Frustum</td></tr><tr><td>â</td><td></td><td></td><td></td><td>57.40</td><td>75.80</td></tr><tr><td>â</td><td>â</td><td></td><td></td><td>58.79</td><td>76.16</td></tr><tr><td>â</td><td>â</td><td>&gt;&gt;</td><td></td><td>58.93</td><td>76.35</td></tr><tr><td>â</td><td>â</td><td></td><td>â</td><td>59.45</td><td>76.98</td></tr></table>

Table 7. Ablation study of IMA on the VoD dataset.
<table><tr><td rowspan="2">N</td><td colspan="3">Gaussian Anchor</td><td rowspan="2"> $\mathrm { \ m A P _ { E A A } }$ </td><td rowspan="2"> $\mathrm { \ m A P _ { D C } }$ </td><td rowspan="2">FLOPS</td></tr><tr><td> $\overline { { \mathrm { C a r } } }$ </td><td>Ped</td><td> $\overline { { \mathbf { C y c } } }$  </td></tr><tr><td>3200</td><td>52.24</td><td>41.79</td><td>70.29</td><td>54.77</td><td>76.13</td><td>599.39</td></tr><tr><td>6400</td><td>52.23</td><td>41.92</td><td>76.35</td><td>56.83</td><td>76.22</td><td>611.95</td></tr><tr><td>12800</td><td>52.94</td><td>48.31</td><td>77.10</td><td>59.45</td><td>76.98</td><td>639.82</td></tr><tr><td>19200</td><td>52.63</td><td>48.84</td><td>76.94</td><td>59.47</td><td>76.70</td><td>666.06</td></tr></table>

Table 8. Ablation of anchor number N.

## 4. Experiments

## 4.1. 3D Object Detection Results

Results on OmniHD-Scenes Dataset. We evaluated the performance of RaGS in a surround-view scenario, as shown in Table 1. RaGS consistently outperforms traditional fixed-grid models RCFusion and BEVFusion on the mAP and ODS. This underscores the advantage of utilizing Gaussian representations for the fusion of 4D radar and surround-view data in the context of 3D object detection. For multi-view setting, our ongoing work is exploring more elegant initialization prior like VGGT [26].

Results on TJ4DRadSet and View-of-Delft Dataset. Compared to the VoD dataset, the TJ4DRadSet dataset [40] presents greater challenges, including complex scenarios such as nighttime, under-bridge conditions, and the addition of a truck category with highly variable object sizes. Despite these difficulties, RaGS achieves state-of-the-art performance, with the highest m $\mathsf { A P } _ { \mathrm { B E V } }$ of 51.04% and the best m $\mathsf { A P } _ { 3 \mathrm { D } }$ of 41.95%, as shown in Table 2. On the other side, Table 4 presents the 3D object detection results on the VoD validation set [23], where our RaGS consistently outperforms state-of-the-art 4D radar and camera fusion models. For cars, we achieve exceptional accuracy, leveraging the flexible occupancy capability of the Gaussian model and an effective fusion design. For pedestrians and cyclists, we also deliver near best performance. When compared to the strong BEV-based baseline LXL [29], our method shows significant improvements, with gains of 5.55% in $\mathrm { \Delta \ n A P _ { E A A } }$ and 8.70% in $\mathrm { \ m A P _ { D C } }$ . Furthermore, it achieves a frame rate of 10.5 FPS, demonstrating its suitability for real-world applications. Visualization results of VoD and TJ4DRadSet are shown in the first and second row of Fig. 5, respectively. Qualitative comparison are presented in Fig. 6. Compared to DETR-based methods such as FUTR3D (49.03%) and RaCFormer (54.44%), as shown in Table 3, our RaGS (61.86%) demonstrates a significant performance advantage. Unlike implicit end-to-end query-based frameworks, the 3D Gaussian Splatting in RaGS offers explicit physical interpretability, serving as a multi-modal aggregator that fuses semantic and geometric features, leveraging radar for 3D perception and monocular cues for enriched observation.

## 4.2. Ablation Study

Overall Ablation. Table 5 reports the overall ablation on RaGS, conducted with half training epochs for efficiency. Starting from the baseline [9] with 4D radar input, each module consistently contributes to performance improvement. FLI provides reliable Gaussian initialization via depth-guided localization, IMA progressively refines multimodal aggregation, and MGF complements them with multi-level feature fusion. Together, these modules form a coherent framework that balances accuracy and efficiency while ensuring stable performance across different conditions.

Ablation on FLI. As shown in Table 6, FLI consists of three components: frustum-based initialization, integration of 4D radar, and depth back-projection. The initialization helps avoid resource wastage from out-of-view objects, while the 4D radar and depth back-projection provide prior knowledge, reducing the potential impact of unstable segmentation and depth estimation. Additionally, the coarse but structured Gaussian from FLI will be refined through IMA for eliminating instability.

Ablation on IMA. IMA aggregates features from both image and radar modalities, and subsequently updates the gaussian positions within the frustum. Table 7 presents the contributions of these components. The integration of radar sparse pillars shows a relatively modest improvement, as feature aggregation has already been performed during the FLI phase and at the cross modal fuser.

Ablation on Hyper-parameters. We conduct an ablation study on the the anchor number N, as shown in Table 8, where both mAP and GFLOPS are reported. To balance performance and computational cost, we select 2-level MGF fusion, 3 layers for IMA, and 12,800 Gaussian anchors. For fusion level L in MGF and the aggregation number M in IMA, see supplementary material for further hyperparameter ablation details.

<table><tr><td rowspan="2">Disturbance</td><td colspan="2"> $\mathrm { m A P _ { E A A } }$ </td><td colspan="2"> $\mathrm { \ m A P _ { D C } }$ </td></tr><tr><td>LXL</td><td>RaGS</td><td>LXL</td><td>RaGS</td></tr><tr><td> $\overline { { \pm 0 ^ { \circ } , \pm 0 . 0 \mathrm { m } } }$ </td><td>56.42</td><td>61.86</td><td>73.03</td><td>81.63</td></tr><tr><td> $\pm 2  { ^ \circ } , \pm 0 . 2  { \mathrm { m } }$ </td><td>56.26</td><td>61.75</td><td>73.16</td><td>81.42</td></tr><tr><td> $\pm 5 ^ { \circ } , \pm 0 . 5 \mathrm { m }$ </td><td>50.25</td><td>56.66</td><td>72.07</td><td>76.39</td></tr></table>

Table 9. Performance comparison under calibration disturbance.
<table><tr><td rowspan="2">Condition</td><td colspan="2">mAPEAA</td><td colspan="2">mAPDC</td></tr><tr><td>LXL</td><td>RaGS</td><td>LXL</td><td>RaGS</td></tr><tr><td>Rain</td><td>51.03</td><td>56.79</td><td>72.62</td><td>75.96</td></tr><tr><td>Haze</td><td>51.16</td><td>56.72</td><td>69.49</td><td>72.29</td></tr><tr><td>Lowlight</td><td>52.13</td><td>56.89</td><td>70.26</td><td>76.02</td></tr></table>

Table 10. RaGS performance under simulated adverse weather.  
<!-- image-->

Figure 6. Qualitative comparison with state-of-the-art.
<table><tr><td>Models</td><td>mAPVoD</td><td>mAPTJ4D</td><td>Param.</td><td>FPS</td><td>GFLOPs</td></tr><tr><td>RCFusion [41]</td><td>49.65</td><td>33.85</td><td>55.3M</td><td>9.0</td><td>680.8</td></tr><tr><td>LXL [29]</td><td>56.31</td><td>36.32</td><td>64.0M</td><td>6.1</td><td>1169.2</td></tr><tr><td>HGSFusion [6]</td><td>58.96</td><td>37.21</td><td>64.6M</td><td>3.2</td><td>1899.6</td></tr><tr><td>SGDet3D [1]</td><td>59.75</td><td>41.82</td><td>73.6M</td><td>9.2</td><td>1309.3</td></tr><tr><td>RaGS (ours)</td><td>61.86</td><td>41.95</td><td>56.5M</td><td>10.5</td><td>781.0</td></tr></table>

Table 11. Comparison with state-of-the-art models.
<table><tr><td>Metrics</td><td>Extra.</td><td>FLI</td><td>IMA</td><td>MGF</td><td>All</td></tr><tr><td>Param. (M)</td><td>38.4</td><td>1.9</td><td>5.9</td><td>10.3</td><td>56.5</td></tr><tr><td>FPS</td><td>48.3</td><td>376.8</td><td>15.9</td><td>109.1</td><td>10.5</td></tr><tr><td>GFLOPS</td><td>508.5</td><td>4.2</td><td>162.8</td><td>98.3</td><td>781.0</td></tr></table>

Table 12. Analysis of efficiency across RaGS components.

## 4.3. Robustness Analysis

As shown in Table 9 and Table 10, RaGS maintains stable performance under calibration disturbance and adverse weather. Even with Â±5â¦/Â±0.5m perturbations or simulated challenging weather condition (rain, haze, low light), both $\mathrm { \ m A P _ { E A A } }$ and $\mathrm { \ m A P _ { D C } }$ remain consistently higher than LXL [29]. This demonstrates that the proposed frustum-guided localization and Gaussian fusion ensure robust and reliable detection across diverse real-world conditions.

## 4.4. Computational Cost

We compared RaGS with existing 4D radar and camera fusion models on mAP, FPS, and GFLOPs. As shown in

Table 11, RaGS outperforms with fewer parameters and lower GFLOPS. Unlike BEV-grid methods, RaGS uses iterative refinement to focus on sparse objects while maintaining scene understanding. SGDet3D uses 160Ã160Ã8 fixed sampling points, about 16 times more than our 12,800. Table 12 presents efficiency analysis of each component within the RaGS Framework.

## 5. Conclusions

In this work, we propose RaGS, the first framework to leverage 3D Gaussian Splatting for fusing 4D radar and monocular images in 3D object detection. RaGS models the fused multi-modal features as continuous 3D Gaussians, enabling dynamically object attention while preserving comprehensive scene perception. The architecture, consisting of FLI, IMA, and MGF, progressively constructs and refines the Gaussian field through frustum-based localization, crossmodal aggregation, and multi-level fusion. Experiments on public benchmarks show state-of-the-art performance.

Limitations. Our ongoing work focuses on extending it to sequential modeling by explicitly exploiting radar velocity within 4D GS for enhanced temporal object awareness.

## References

[1] Xiaokai Bai, Zhu Yu, Lianqing Zheng, Xiaohan Zhang, Zili Zhou, Xue Zhang, Fang Wang, Jie Bai, and Hui-Liang Shen. SGDet3D: Semantics and Geometry Fusion for 3D Object Detection Using 4D Radar and Camera. IEEE Robotics and Automation Letters, pages 1â8, 2024. 2, 5, 6, 7, 8

[2] Yang Cao, Yuanliang Jv, and Dan Xu. 3DGS-DET: Empower 3D Gaussian Splatting with Boundary Guidance and Box-Focused Sampling for 3D Object Detection. arXiv preprint arXiv:2410.01647, 2024. 3

[3] Xuanyao Chen, Tianyuan Zhang, Yue Wang, Yilun Wang, and Hang Zhao. FUTR3D: A Unified Sensor Fusion Framework for 3D Detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 172â181, 2023. 5, 6

[4] Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, and Jiaya Jia. VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21674â21683, 2023. 6

[5] Xiaomeng Chu, Jiajun Deng, Guoliang You, Yifan Duan, Houqiang Li, and Yanyong Zhang. Racformer: Towards high-quality 3d object detection via query-based radarcamera fusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Conference, pages 17081â 17091, 2025. 5

[6] Zijian Gu, Jianwei Ma, Yan Huang, Honghao Wei, Zhanye Chen, Hui Zhang, and Wei Hong. HGSFusion: Radar-Camera Fusion with Hybrid Generation and Synchronization for 3D Object Detection. In AAAI Conference on Artificial Intelligence, pages 3185â3193, 2025. 5, 8

[7] Georg Hess, Carl Lindstrom, Maryam Fatemi, Christoffer Â¨ Petersson, and Lennart Svensson. SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussia Splatting for Autonomous Driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11982â11992, 2025. 2, 3

[8] Xun Huang, Ziyu Xu, Hai Wu, Jinlong Wang, Qiming Xia, Yan Xia, Jonathan Li, Kyle Gao, Chenglu Wen, and Cheng Wang. L4DR: LiDAR-4DRadar Fusion for Weather-Robust 3D Object Detection. In AAAI Conference on Artificial Intelligence, pages 3806â3814, 2025. 2

[9] Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction. In European Conference on Computer Vision, pages 376â393. Springer, 2024. 2, 3, 6, 8

[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4):139â1, 2023. 2

[11] Youngseok Kim, Sanmin Kim, Jun Won Choi, and Dongsuk Kum. CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer. In AAAI Conference on Artificial Intelligence, pages 1160â1168, 2023. 2

[12] Youngseok Kim, Juyeb Shin, Sanmin Kim, In-Jae Lee, Jun Won Choi, and Dongsuk Kum. CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception. In IEEE/CVF International Conference on Computer Vision, pages 17615â 17626, 2023. 2

[13] Pou-Chun Kung, Skanda Harisha, Ram Vasudevan, Aline Eid, and Katherine A Skinner. RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes. arXiv preprint arXiv:2506.01379, 2025. 3

[14] Jinyu Li, Chenxu Luo, and Xiaodong Yang. PillarNeXt: Rethinking Network Designs for 3D Object Detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17567â17576, 2023. 6

[15] Zhiwei Lin, Zhe Liu, Zhongyu Xia, Xinhao Wang, Yongtao Wang, Shengxiang Qi, Yang Dong, Nan Dong, Le Zhang, and Ce Zhu. RCBEVDet: Radar-camera Fusion in Birdâs Eye View for 3D Object Detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14928â 14937, 2024. 2, 6

[16] Hongsi Liu, Jun Liu, Guangfeng Jiang, and Xin Jin. MSSF: A 4D Radar and Camera Fusion Framework with Multi-Stage Sampling for 3D Object Detection in Autonomous Driving, 2024. 5

[17] Jianan Liu, Qiuchi Zhao, Weiyi Xiong, Tao Huang, Qing-Long Han, and Bing Zhu. SMURF: Spatial Multi-Representation Fusion for 3D Object Detection with 4D Imaging Radar. IEEE Transactions on Intelligent Vehicles, 9 (1):799â812, 2024. 5, 6

[18] Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela L Rus, and Song Han. BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Birdâs-Eye View Representation. In IEEE International Conference on Robotics and Automation, pages 2774â2781, 2023. 5, 6

[19] Yunfei Long, Abhinav Kumar, Daniel Morris, Xiaoming Liu, Marcos Castro, and Punarjay Chakravarty. RADIANT: Radar-Image Association Network for 3D Object Detection. In AAAI Conference on Artificial Intelligence, pages 1808â 1816, 2023. 2

[20] Jiageng Mao, Shaoshuai Shi, Xiaogang Wang, and Hongsheng Li. 3D Object Detection for Autonomous Driving: A Comprehensive Survey. International Journal of Computer Vision, 131(8):1909â1963, 2023. 1

[21] Ramin Nabati and Hairong Qi. CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1527â1536, 2021. 2

[22] Dong-Hee Paek, Seung-Hyun Kong, and Kevin Tirta Wijaya. K-Radar: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions. Advances in Neural Information Processing Systems, 35:3819â3829, 2022. 2

[23] Andras Palffy, Ewoud Pool, Srimannarayana Baratam, Julian FP Kooij, and Dariu M Gavrila. Multi-Class Road User Detection with 3+1D Radar in the View-of-Delft Dataset. IEEE Robotics and Automation Letters, 7(2):4961â4968, 2022. 2, 5, 6, 7

[24] Rukhovich. ImVoxelNet: Image to Voxels Projection for Monocular and Multi-View General-Purpose 3D Object Detection. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2397â2406, 2022. 5, 6

[25] Fengrui Tian, Shaoyi Du, and Yueqi Duan. MonoNeRF: Learning a Generalizable Dynamic Radiance Field from Monocular Videos. In IEEE/CVF International Conference on Computer Vision, pages 17903â17913, 2023. 3

[26] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Conference, pages 5294â5306, 2025. 7

[27] Li Wang, Xinyu Zhang, Baowei Xv, Jinzhao Zhang, Rong Fu, Xiaoyu Wang, Lei Zhu, Haibing Ren, Pingping Lu, Jun Li, et al. InterFusion: Interaction-based 4D Radar and Li-DAR Fusion for 3D Object Detection. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 12247â12253. IEEE, 2022. 2

[28] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github. com/facebookresearch/detectron2, 2019. 7

[29] Weiyi Xiong, Jianan Liu, Tao Huang, Qing-Long Han, Yuxuan Xia, and Bing Zhu. LXL: LiDAR Excluded Lean 3D Object Detection with 4D Imaging Radar and Camera Fusion. IEEE Transactions on Intelligent Vehicles, 9(1):79â92, 2024. 2, 5, 6, 8

[30] Weiyi Xiong, Zean Zou, Qiuchi Zhao, Fengchun He, and Bing Zhu. LXLv2: Enhanced LiDAR Excluded Lean 3D Object Detection with Fusion of 4D Radar and Camera. IEEE Robotics and Automation Letters (RAL), 2025. 5

[31] Chenfeng Xu, Bichen Wu, Ji Hou, Sam Tsai, Ruilong Li, Jialiang Wang, Wei Zhan, Zijian He, Peter Vajda, Kurt Keutzer, et al. NeRF-Det: Learning Geometry-Aware Volumetric Representation for Multi-View 3D Object Detection.

In IEEE/CVF International Conference on Computer Vision, pages 23320â23330, 2023. 3

[32] Ruoyu Xu, Zhiyu Xiang, Chenwei Zhang, Hanzhi Zhong, Xijun Zhao, Ruina Dang, Peng Xu, Tianyu Pu, and Eryun Liu. SCKD: Semi-Supervised Cross-Modality Knowledge Distillation for 4D Radar Object Detection. In AAAI Conference on Artificial Intelligence, pages 8933â8941, 2025. 6

[33] Hongru Yan, Yu Zheng, and Yueqi Duan. Gaussian-Det: Learning Closed-Surface Gaussians for 3D Object Detection. In International Conference on Learning Representations, 2025. 3

[34] Sheng Yang, Tong Zhan, Shichen Qiao, Jicheng Gong, Qing Yang, Jian Wang, and Yanfeng Lu. ZFusion: An Effective Fuser of Camera and 4D Radar for 3D Object Perception in Autonomous Driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Conference, pages 3768â3777, 2025. 6

[35] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20331â20341, 2024. 2, 3

[36] Junbo Yin, Jianbing Shen, Runnan Chen, Wei Li, Ruigang Yang, Pascal Frossard, and Wenguan Wang. IS-Fusion: Instance-Scene Collaborative Fusion for Multimodal 3D Object Detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14905â14915, 2024. 6

[37] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Centerbased 3D Object Detection and Tracking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11784â11793, 2021. 6

[38] Cheng Zhao, Su Sun, Ruoyu Wang, Yuliang Guo, Jun-Jun Wan, Zhou Huang, Xinyu Huang, Yingjie Victor Chen, and Liu Ren. TCLC-GS: Tightly Coupled LiDAR-Camera Gaussian Splatting fo Autonomous Driving. In European Conference on Computer Vision, pages 91â106. Springer, 2024. 3

[39] Haocheng Zhao, Runwei Guan, Taoyu Wu, Ka Lok Man, Limin Yu, and Yutao Yue. UniBEVFusion: Unified Radar-Vision BEVFusion for 3D Object Detection. arXiv preprint arXiv:2409.14751, 2024. 5

[40] Lianqing Zheng, Zhixiong Ma, Xichan Zhu, Bin Tan, Sen Li, Kai Long, Weiqi Sun, Sihan Chen, Lu Zhang, Mengyue Wan, et al. TJ4DRadSet: A 4D Radar Dataset for Autonomous Driving. In IEEE International Conference on Intelligent Transportation Systems, pages 493â498, 2022. 2, 5, 7

[41] Lianqing Zheng, Sen Li, Bin Tan, Long Yang, Sihan Chen, Libo Huang, Jie Bai, Xichan Zhu, and Zhixiong Ma. RCFusion: Fusing 4-D Radar and Camera with Birdâs-Eye View Features for 3-D Object Detection. IEEE Transactions on Instrumentation and Measurement, 72:1â14, 2023. 2, 4, 5, 6, 7, 8

[42] Lianqing Zheng, Long Yang, Qunshu Lin, Wenjin Ai, Minghao Liu, Shouyi Lu, Jianan Liu, Hongze Ren, Jingyue Mo, Xiaokai Bai, et al. OmniHD-Scenes: A Next-Generation Multimodal Dataset for Autonomous Driving. arXiv preprint arXiv:2412.10734, 2024. 2, 5, 7

[43] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. DrivingGaussian:

Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21634â 21643, 2024. 3