# GEVO: Memory-Efficient Monocular Visual Odometry Using Gaussians

Dasong Gao\* , Peter Zhi Xuan Li\* , Vivienne Sze , Sertac Karaman

AbstractâConstructing a high-fidelity representation of the 3D scene using a monocular camera can enable a wide range of applications on low-energy devices, such as micro-robots, smartphones, and AR/VR headsets. On these devices, memory is often limited in capacity and its access often dominates the consumption of compute energy. Although Gaussian Splatting (GS) allows for high-fidelity reconstruction of 3D scenes, current GS-based SLAM is not memory efficient as a large number of past images is stored to retrain Gaussians for reducing catastrophic forgetting. These images often require two-ordersof-magnitude higher memory than the map itself and thus dominate the total memory usage. In this work, we present GEVO, a GS-based monocular SLAM framework that achieves comparable fidelity as prior methods by rendering (instead of storing) them from the existing map. Novel Gaussian initialization and optimization techniques are proposed to remove artifacts from the map and delay the degradation of the rendered images over time. Across a variety of environments, GEVO achieves comparable map fidelity while reducing the memory overhead to around 58 MBs, which is up to 94Ã lower than prior works.

Index TermsâSLAM, Mapping, Incremental Learning

## I. INTRODUCTION

E Nergy-constrained devices, such as AR/VR headsets andmicro-robots, enable a wide range of applications that micro-robots, enable a wide range of applications that involve long-term and safe interactions with 3D environments through their high fidelity 3D representation. For instance, constructing 3D representation in real-time allows i) AR/VR headsets to warn users when they are too close to the realworld obstacles obscured by virtual ones, and ii) microrobots to perform collision checking and path planning during autonomous exploration. Due to limited battery capacity, these devices rely on low-power passive sensors to perceive the environment. Thus, it is crucial for online simultaneous localization and mapping (SLAM) to produce a high fidelity 3D map with a monocular RGB camera.

On energy-constrained devices, memory is often limited in capacity and its access could dominate the total compute energy. For instance, the energy for accessing data in an 8 KB cache is 10Ã more than performing a 32-bit floating point multiplication [1]. Furthermore, the energy for accessing data in off-chip DRAM (GBs of capacity) is orders of magnitude higher than that for on-chip cache (KBs to MBs of capacity) [1]. Thus, algorithms should be memory-efficient with low memory usage so that variables can be effectively cached on chip to reduce energy.

<!-- image-->

<!-- image-->  
(a) Reconstruction before catastrophic forgetting

(b) MonoGS [2] no past images 8 images stored (7 MB)  
<!-- image-->

<!-- image-->  
(c) GEVO (This work)  
8 images stored (7 MB)  
(d) MonoGS [2]  
114 images stored (100 MB)  
Fig. 1. During online GS-based SLAM, the map (consisting of 3D Gaussians) is built by rendering and optimizing at each viewpoint using a sliding window buffer of images. a) The region visible during the current sliding window achieves high fidelity after initial optimization. b) However, without storing and retraining the map on a large number of past images, the fidelity of the same region degrades over time due to forgetting (artifacts in rectangles). c) While alleviating forgetting, our GEVO avoids storing past images to reduce the memory overhead. d) To achieve similar map fidelity, MonoGS [3] stores all past keyframes and incurs a memory overhead at least 50Ã higher than the size of the map.

Constructing a high-fidelity map requires RGB images to guide its optimization process. To achieve real-time operation, many SLAM frameworks [4], [5], [6], [7] optimize the pose and map using a small sliding window of images. However, for dense SLAM, the map tends to catastrophically forget and degrade over time after the sliding window has passed (see Figure 1a vs. 1b). To alleviate forgetting, prior frameworks additionally store a large number of past images outside the current sliding window to repeatedly retrain the map. These frameworks use either neural networks [8], [9], [10] or learnable 3D Gaussians [3], [11] to map both occupancy and color of the environment. Unfortunately, the overhead memory used to store these images dominates the total memory and is orders of magnitude higher than both the current sliding

window and the map itself.

Our contribution is a memory-efficient SLAM framework using Gaussians, called GEVO, that significantly reduces memory overhead by rendering past images from the existing map instead of storing them in memory. However, the fidelity of these images is lower than the original and can slowly degrade overtime due to the artifacts in the map caused by forgetting. Thus, using these images to guide Gaussian optimization via splatting (i.e., GS [12]) alone is insufficient for constructing a high fidelity map. To complement GS, GEVO contains the following procedures to further reduce incorrect occlusion and overfitting due to catastrophic forgetting:

1) Occupancy-Preserving Initialization: To reduce incorrect occlusions, Gaussians that lie within the obstaclefree regions (i.e., orange Gaussian in the blue region of Figure 2a) are pruned. Thus, in addition to representing obstacles, Gaussians representing free regions are initialized to identify incorrect occlusions.

2) Consistency-Aware Optimization: To reduce overfitting of the map to the current window, we only optimize a small subset of Gaussians that are both inconsistent and sufficiently visible to the camera (see orange Gaussians in Figure 2b). To ensure rendered images maintain high fidelity, we locally optimize noisy Gaussians created from the current sliding window before merging them to the map for global optimization.

Across a variety of environments, GEVO achieves comparable map fidelity (see Figure 1c) and reduces the memory overhead to around 58MBs, which is up to 94Ã lower than prior works. Thus, GEVO makes a significant stride towards the deployment of GS-based SLAM on low-energy devices.

## II. RELATED WORK

Monocular SLAM frameworks can be classified based on the type of scene representation (e.g., points, planes, neural networks, Gaussians). Constructing these representations online exhibits different trade-offs in efficiency and fidelity.

Traditional Monocular SLAM: Traditional SLAM frameworks can be classified as indirect or direct, and excel at real-time localization by tracking and optimizing over a set of points representing the 3D scene. For indirect frameworks [4], [5], [6], [7], the set of points are selected using feature extractors [13], [14] that seek to uniquely identify certain characteristics of the environment such as corners. Although these frameworks are often memory-efficient and real-time, the amount of unique features is very sparse which produces a map with very low coverage of the environment. In contrast, direct frameworks [15], [16], [17], [18] seek to track a denser set of points with high photometric gradients at the expense of larger memory and computational overhead. Although the coverage of the environment increases, reconstruction is not photo-realistic and contains significant noise in regions with less texture [7].

To reduce sparsity, other traditional frameworks explore more descriptive geometric primitives, such as planes [19], quadrics [20] and meshes [21]. However, these frameworks coarsely track the locations of objects that conform to their respective primitives and thus struggle with modeling remaining objects in the scene that do not conform.

Neural Monocular SLAM: To provide a photo-realistic reconstruction of the environment, neural-based frameworks, such as GO-SLAM [22], NICER-SLAM [10], and iMODE [9] represent the environment using a Neural Radiance Field (NeRF). Due to the volumetric rendering required for training NeRFs, the training process is computationally intensive. Thus, most of these frameworks propose techniques that accelerate training, some of which include i) using a hybrid scene representation with the voxel grids (in NICER-SLAM) or hash table (in GO-SLAM), and ii) training on a carefully selected subset of input images (in most prior works including iMODE).

Even though throughput was enhanced by these techniques, almost all Neural SLAM frameworks suffer from catastrophic forgetting, which is reduced by periodic re-training on images acquired throughout the entire experiment. Thus, these images need to be stored as overhead in memory, which quickly grows with the duration of the experiment to dominate the total memory usage.

Gaussian Monocular SLAM: To improve throughput and achieve photo-realistic rendering, recent frameworks, such as MonoGS [2], Photo-SLAM [23], and SplatSLAM [24], use Gaussian Splatting (GS) to train learnable Gaussians for 3D representation. There frameworks propose different localization techniques to complement GS. For instance, both MonoGS and SplatSLAM localize the camera against the global map via minimizing a photometric cost function, while Photo-SLAM utilizes ORB-SLAM [4].

Similar to Neural SLAM, current Gaussian SLAM frameworks also suffer from catastrophic forgetting and thus require the storage of a large number of images to periodically retrain all Gaussians. In this work, we propose memory-efficient techniques that reduce catastrophic forgetting in GS-based frameworks by rendering most images from the map instead of storing them in memory. From Figure 1, our framework, named GEVO, can also achieve comparable rendering accuracy while requiring significantly less memory overhead compared with prior frameworks.

## III. PROPOSED METHODS

In this section, we present GEVO, a memory-efficient GSbased monocular SLAM framework, that reduces catastrophic forgetting by relying on images rendered from the map to guide the GS optimization process. Recall that online SLAMs operate on a sliding window of images for localization and mapping. Catastrophic forgetting can occur when Gaussians created from the current window occlude the ones from the past windows. As illustrated in Figure 2a, the inconsistency is caused by retrospective occlusion (RO) where Gaussians from the current view (orange) lie within the obstacle-free region (blue) of the prior view.

Catastrophic forgetting can also occur when a previously observed region from a past sliding window overfit to images from the current sliding window. As illustrated in Figure 2b, overfitting is caused by incomplete ray obscuration (IRO)

<!-- image-->  
(a) RO: The current view inconsistently inserts or moves a new Gaussian into the obstacle-free region of previous views to occlude the existing Gaussians.  
(b) IRO: Sensor rays of the current view pass through new Gaussians to cause overfitting of an existing Gaussian created from previous view.  
Fig. 2. Two scenarios that cause catastrophic forgetting in Gaussian Splatting: b) retrospective occlusion (RO) and a) incomplete ray obscuration (IRO). RO causes the new Gaussians to occlude ones in the past view (red rectangles in Figure 1b). IRO causes the existing Gaussians to overfit to the current view (green rectangles in Figure 1b).

where Gaussians that are associated with the current view (red) do not completely obscure the sensor rays (blue). Thus, Gaussians created from prior views (orange) are still partially visible such that their parameters update to match the appearances in the current view. In prior works, both RO and IRO are reduced by storing images from all sliding windows to retrain the Gaussians. These images typically dominate the total memory usage which grows over time.

Using rendered images from the existing map is not sufficient for reducing catastrophic forgetting during optimization. In particular, the fidelity of these images slowly degrades over time due to artifacts in the map caused by forgetting from RO and IRO. To reduce RO, our framework consists of an accurate Gaussian initialization procedure (Section III-A) that compactly encodes obstacle-free regions. These regions are used to identify instances where new Gaussians occlude existing ones, which are pruned at the end of optimization. To reduce IRO, we propose a two-stage optimization procedure (Section III-B) to update a small subset of Gaussians that are both inconsistent and sufficiently visible to the current sliding window so that the remaining Gaussians do not overfit to the current window.

## A. Occupancy-Preserving Initialization

In this section, we present an efficient procedure that initializes Gaussian parameters representing obstacles and obstacle-free regions in the current sliding window. The free regions are used to prune Gaussians that causes RO.

To achieve efficiency and good generalization across a variety of environments, our procedure is adapted from an efficient implementation of multi-view stereo (MVS) [26] and does not rely on time-consuming COLMAP [27], [28] or less accurate random sampling [2]. Figure 3 summarizes our procedure. Given a sequence of RGB keyframes in a sliding window, we construct a cost volume that captures the photometric consistency for each pixel in the most recent image at different depth hypotheses. From the cost volume, belief propagation [29] is performed to extract a depth image associated with the most recent image. Finally, Gaussian parameters for both obstacles (red) and free regions (blue) are computed by the memory-efficient $\mathrm { S P G F ^ { * } }$ algorithm [25].

Our procedure can be integrated with many localization and keyframe selection strategies, such as map-centric direct methods in MonoGS [2] and feature-based methods in ORB-SLAM [4]. Details about our procedure are described below.

1) Cost Volume Generation: Given a sequence of $N =$ 8 or 10 keyframes $\left( I _ { 0 } , \ldots , I _ { N - 1 } \right)$ from the sliding window buffer (with $I _ { N - 1 }$ being the most recent), the value of the photometric cost $V ( \mathbf { u } , d )$ for pixel u in $I _ { N - 1 }$ when depth is hypothesized to be d is defined as

$$
V ( \mathbf { u } , d ) = \frac { 1 } { N - 1 } \sum _ { i = 0 } ^ { N - 2 } | I _ { N - 1 } ( \mathbf { u } ) - I _ { i } \left( \pi ( \mathbf { u } , d , \mathbf { T } _ { N - 1 } ^ { i } ) \right) | ,\tag{1}
$$

where $I ( \cdot )$ is the intensity of a specific pixel in image I, $\mathbf { T } _ { N - 1 } ^ { i } \in \mathbb { S E } ( 3 )$ is the transformation matrix from image $I _ { N - 1 }$ to $I _ { i } , \pi ( \cdot )$ warps the coordinate u from image $I _ { N - 1 }$ to $I _ { i }$ given the depth hypotheses d.

In our experiments, we choose 64 depth hypothesis $( i . e . ,$ $\{ d _ { 0 } , \ldots , d _ { 6 3 } \}$ that are equally spaced from 0.25 â 25 m. To reduce memory overhead, we downsample the image by 4Ã in each dimension before creating the cost volume in order to exploit the spatial redundancy in the image. Assuming that each image has height $H = 4 8 0$ and width $W = 6 4 0$ , the resulting cost volume V only requires 4.7 MBs. Since we use full-resolution RGB images for subsequent optimizations, potential loss in spatial details will be recovered.

2) Gaussian Generation: To determine the most likely depth hypothesis for each pixel under the assumption that each obstacle has smooth surfaces, belief propagation (BP) from [29] is used to extract a depth image for the most recent keyframe in the sliding window buffer. Under the assumption that neighboring pixels with the same color in the keyframe are likely to describe the same surface, we use an efficient algorithm proposed in [30] to upsample the depth image from BP to the full resolution of the keyframe.

Given the most recent keyframe and its depth image, we enhanced a memory-efficient algorithm, called SPGF\* [25], to generate a set of Gaussians representing obstacles (red) and free (blue) regions (see Figure 3). Unlike prior approaches [31], [32], [33], [34] that process the depth image in multiple passes, SPGF\* exploits the connectivity encoded in the depth image to efficiently generate the Gaussians in a single pass with comparable accuracy. Since SPGF\* was mainly designed for accurate depth reconstruction, each Gaussian that represents an obstacle could enclose a surface containing multiple colors. To enhance the fidelity of color representation, we modified SPGF\* to ensure that each Gaussian can only represent a surface with a similar color.

## B. Consistency-Aware Optimization

After Gaussians are initialized in Section III-A, they are fused into the global map as illustrated in Figure 4. Two challenges arise when integrating the new Gaussians into the existing map using the current sliding window: i) Since these

<!-- image-->

Fig. 3. Occupancy-Preserving Initialization. Given a set of recently acquired keyframes and poses in a sliding window buffer, the depth image for the most recent keyframe is computed using belief propagation on a photometric cost volume at a quarter of the imageâs resolution. Then, the depth and RGB image are used to initialize a set of Gaussians (Gt) for representing obstacles (red) and free region (blue) using the SPGF\* algorithm [25]. Gaussians representing free regions are fused across multiple keyframes to identify instances of retrospective occlusion (RO) during consistency-aware optimization.  
<!-- image-->  
Fig. 4. Consistency-Aware Optimization. Given newly initialized Gaussians (Gt), we perform a GS-based optimization in two stages: i) Local stage performs GS to optimize a local map $\bar { \mathcal { M } } _ { t }$ that represents all geometries visible from the sliding window, and ii) Global stage selectively optimizes a small active set At of Gaussians (green) consisting of the local map $\bar { \mathcal { M } } _ { t }$ and existing Gaussians $\scriptstyle { \mathcal { E } } _ { t }$ with high rendering error. This active set selection tends to exclude Gaussians obscured from camera views used for training and thus reduces IRO. Since Gaussians from the local map are sufficiently accurate, images from randomly selected past views are rendered from the global map to guide the global optimization stage. Finally, Gaussians that causes RO are pruned with the help of the obstacle-free regions created during initialization (blue). Note that Gaussians representing free regions are omitted in the global stage except for the pruning step for ease of visualization.

Gaussians are potentially noisy, inserting them into the map likely causes $\operatorname { R O } ;$ ii) During optimization, both IRO and RO tend to occur due to the lack of constraints from past views. In prior works [2], [23], [22], both are resolved by training Gaussians on keyframes sampled from past sliding windows. In our work, we rely on past keyframes rendered from the existing map to reduce memory overhead.

However, the fidelity of these rendered images degrades over time which leads to significant degradation of the map itself. To maintain fidelity, we employ a two-stage optimization that first enhances the initialized Gaussians in the local stage before optimizing them with other existing Gaussians in the global stage. Since existing Gaussians are not perturbed during the local stage, they are used to render past keyframes with high fidelity. To reduce IRO, we optimize a small subset of Gaussians that are both inconsistent and visible from the current sliding window. In addition, the free regions encoded by Gaussians from past keyframes allow us to further reduce RO via occupancy-based pruning.

1) Local Stage: The initialized Gaussians $\mathcal { G } _ { t }$ are appended with Gaussians $\bar { \mathcal { M } } _ { t - 1 }$ that are also visible from the current sliding window to form an updated local map $\widetilde { \mathcal { M } } _ { t }$ . To reduce the noise of the update map caused by recently initialized Gaussians, the map is optimized using images from only the current sliding window via the objective [2],

$$
\operatorname * { a r g m i n } _ { \mathbf { T } _ { k } , \mathcal { \widetilde { M } } _ { t } } E _ { p h o } + E _ { i s o } ,\tag{2}
$$

where $E _ { p h o }$ is the photometric loss between rendered and ground truth images, $E _ { i s o }$ is the isotropic loss, which prevents the formation of elongated or thin Gaussians, and $\mathbf { T } _ { k } \in \mathbb { S E } ( 3 )$ is its estimated pose.

2) Global Stage: To resolve remaining inconsistencies of the local map $\mathcal { \breve { M } } _ { t }$ with prior measurements, we merge Gaussians $\widetilde { \mathcal { M } } _ { t }$ into the global map $\mathcal { M } _ { t - 1 }$ . During the merging process, we reduce catastrophic forgetting by identifying and updating a small subset of the Gaussians that are temporally inconsistent across past viewpoints by using keyframes rendered from these viewpoints. Since the global map $\mathcal { M } _ { t - 1 }$ is not perturbed during the local stage, these rendered keyframes from $\mathcal { M } _ { t - 1 }$ maintain high quality and are sufficient for resolving the remaining inconsistencies.

The global stage consists of the following three sequential steps: insertion, selective optimization, and pruning.

Insertion and activation: We insert the local map $\widetilde { \mathcal { M } } _ { t }$ and the previous global map $\mathcal { M } _ { t - 1 }$ to create the pre-optimzied global map: $\mathcal { M } _ { t } ^ { \prime }  \tilde { \mathcal { M } } _ { t } \cup \mathcal { M } _ { t - 1 }$ . To prevent retrospective occlusion (RO) caused by local map $\bar { \mathcal { M } } _ { t }$ , we lower its opacity to 0.2 prior to the insertion.

Selective optimization: We employ an optimization procedure similar to Equation (2) but with two modifications: 1)

<!-- image-->  
Fig. 5. GEVO achieves comparable rendering accuracy with other monocular methods on Replica (top) and TUM-RGBD (bottom). In particular, GEVO achieves high fidelity by reducing RO, especially for Gaussians representing distant and/or large objects (in green rectangles). Since rendered images degrade in quality slowly over time, optimizating Gaussians using these images in GEVO leads to minor loss of details in some close-up, feature-rich regions (in red rectangles).

To prevent Gaussians in the pre-optimized map $\mathcal { M } _ { t } ^ { \prime }$ from overfitting to the images from the current sliding window buffer $\mathcal { W } _ { t } .$ , we select and only optimize an active subset $\mathcal { A } _ { t } \subseteq \mathcal { M } _ { t } ^ { \prime }$ , and 2) we additionally introduce a photometric consistency loss $E _ { p c }$ to further ensure consistency with the prior global map $\mathcal { M } _ { t - 1 }$ . The overall objective is therefore:

$$
\operatorname * { a r g m i n } _ { \mathbf { T } _ { k } , A _ { t } } E _ { p h o } + E _ { i s o } + E _ { p c } .\tag{3}
$$

Specifically, we choose $\mathcal { A } _ { t } ~ = ~ \widetilde { \mathcal { M } } _ { t } \cup \mathcal { E } _ { t } .$ , which contains the newly inserted Gaussians and a subset $\mathcal { E } _ { t }$ that incurs high rendering error in the current window:

$$
\mathcal { E } _ { t } = \left\{ g \in \mathcal { M } _ { t - 1 } : \operatorname* { m a x } _ { k \in \mathcal { W } _ { t } } E _ { k } ( g ) > \epsilon \right\} ,\tag{4}
$$

where the per-Gaussian rendering error $E _ { k } ( g )$ as in [35]:

$$
E _ { k } ( g ) = \sum _ { \mathbf { u } } w ( g , \mathbf { u } ) \vert R ( \mathcal { M } _ { t - 1 } , \mathbf { T } _ { k } ) ( \mathbf { u } ) - C _ { k } ( \mathbf { u } ) \vert ,\tag{5}
$$

with u being the pixel coordinate, and $w ( g , \mathbf { u } )$ is the alphablending coefficient of g at pixel u. Since the Gaussians that are more visible to the camera contribute more to $E _ { k } ( g )$ , the active set tends to exclude existing Gaussians that are not wellobserved by the current window and reduce IRO.

To further ensure that the fidelity of the global map does not degrade over time, the objective in Equation (3) includes a photometric consistency loss $E _ { p c }$ evaluated on keyframes at four past camera views outside the sliding window:

$$
E _ { p c } = \sum _ { l = 1 } ^ { 4 } \big \| R ( \mathcal { M } _ { t } ^ { \prime } , \mathbf { T } _ { k _ { l } } ) - \big \widecheck { \big { X } } ( R ( \mathcal { M } _ { t - 1 } , \mathbf { T } _ { k _ { l } } ) ) \big \| _ { 1 } ,\tag{6}
$$

where $R ( \cdot )$ is the rendering function, frame index kl is uniformly sampled from past timesteps 1 . . . tâW , and $\mathcal { \vec { X } } ( \cdot )$ is the stop gradient operator [35], which avoids back-propagating gradient through $R ( \mathcal { M } _ { t - 1 } , \mathbf { T } _ { k _ { l } } )$

Pruning: After selective optimization, remaining Gaussians that have a) an opacity less than 0.7 or b) an occupancy probability less than 0.9 are likely to cause RO and thus pruned. The occupancy probability is computed using Gaussian Mixture Regression [25] on the initialized Gaussians representing free regions from Section III-A.

## IV. EXPERIMENTS

We evaluate GEVO against state-of-the-art (SOTA) monocular dense SLAM frameworks in terms of accuracy and efficiency. To demonstrate the trade-offs among different system configurations, we choose the following frameworks: GO-SLAM1 [22] (learning-based tracking + neural-based mapping), MonoGS2 [2] (direct tracking + Gaussian-based mapping), and Photo-SLAM3 [23] (feature-based tracking + Gaussian-based mapping). Compared with prior methods in multiple environments, GEVO reduces the overhead memory by 8-145Ã with up to 10% computation overhead while maintaining comparable accuracy (Figure 5).

This section is organized as follows. After describing implementation details and dataset selection in Section IV-A, we compare the memory usage (Section IV-B), computational efficiency (Section IV-C), rendering and localization accuracy (Section IV-D) against SOTA methods. An ablation study for the design of GEVO is presented in Section IV-E.

## A. Experiment Setup

GEVO is implemented in C++ with CUDA acceleration and can be found at https://github.com/mit-lean/gevo. We benchmarked GEVO and prior methods with an Intel Xeon Gold 6130 and NVIDIA TITAN RTX GPU. We use a sliding window buffer that stores either 8 (TUM) or 10 (Replica) keyframes. For prior methods, we use the default settings from the open-source code release for supported datasets or otherwise perform fine-tuning from default settings.

Our method is compatible with various tracking methods. For fairness, we present results of two variants of GEVO: Ours (Direct) employs the photometric tracking from MonoGS whereas Ours (ORB-SLAM) uses ORB-SLAM [4] for tracking as in Photo-SLAM. For Photo-SLAM and Ours (ORB-SLAM), we disabled loop closure and downsampled the ORB vocabulary to 1/100 of the original to reduce memory usage without sacrificing accuracy. We disabled spherical harmonics for all GS-based methods.

TABLE I  
MEMORY USAGE AND RENDERING ACCURACY OF GEVO COMPARED WITH PRIOR WORKS ON SCENES FROM THE REPLICA [36] AND TUM RGB-D [37] DATASETS. THE BEST RESULTS ARE HIGHLIGHTED AS first AND SECOND .
<table><tr><td>Metrics</td><td>Methods</td><td> office0 office1</td><td></td><td>office2 office3 office4</td><td></td><td></td><td>room0</td><td>room1</td><td>room2</td><td>Average</td><td>fr1_desk fr2_xyz fr3_office Aveage</td><td></td><td></td><td></td></tr><tr><td rowspan="4">Overhead Memory (MB) â</td><td>GO-SLAM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2673.6 2794.6 3066.2 3001.3 2575.7 3606.0 3651.6 3796.0</td><td>3145.6</td><td>3747.0 106.7</td><td>2870.8</td><td>4087.5</td><td>3568.5</td></tr><tr><td>MonoGS</td><td></td><td>618.6</td><td>664.8</td><td>678.4</td><td>383.2</td><td>697.0</td><td>168.2</td><td>683.7</td><td>53.6</td><td></td><td>-96.3</td><td>213.3</td><td>138.8</td></tr><tr><td>Ours (Direct)</td><td>72.8</td><td>71.1</td><td>70.5</td><td>_75.5</td><td>67.6</td><td>82.3</td><td>71.2</td><td>71.6</td><td>72.8</td><td>25.1</td><td>_24.2</td><td>24.3</td><td>24.6</td></tr><tr><td>PhooSLA</td><td></td><td>606.2 1504.7</td><td>631.1</td><td>559.2</td><td>633.3</td><td>675.1</td><td>791.6</td><td>650.5</td><td>756.5</td><td>135.6</td><td>336.9</td><td>629.3</td><td>367.3</td></tr><tr><td rowspan="5">Map Memory (MB) â</td><td>Ours (ORB-SLAM)</td><td>85.6 48.1</td><td>86.1 48.1</td><td>100.8 48.1</td><td>93.4 48.1</td><td>100.3 48.1</td><td>92.3 48.1</td><td>95.6</td><td>98.5</td><td>94.1</td><td>40.4 48.1</td><td>30.7</td><td>43.4</td><td>38.1</td></tr><tr><td>GO-SLAM</td><td></td><td></td><td></td><td>6.2</td><td></td><td></td><td>48.1</td><td>48.1</td><td>48.1</td><td></td><td>48.1</td><td>48.1</td><td>48.1</td></tr><tr><td>MonoGS</td><td></td><td>2.8</td><td>3.1</td><td>4.3</td><td></td><td>2.4 9.1</td><td>2.9</td><td>4.4</td><td>4.4</td><td>1.3</td><td>2.4</td><td>2.1</td><td>-1.9</td></tr><tr><td>Ours_ (Direct)</td><td></td><td>4.6 3.4</td><td>_6.6</td><td>5.9</td><td></td><td>_4.5 9.1</td><td>5.8</td><td>5.7</td><td>5.7</td><td>0.8</td><td>0.4</td><td>1.0</td><td>0.7</td></tr><tr><td>Photo-SLAM Ours (ORB-SLAM)</td><td>4.8</td><td>10.2</td><td>6.4</td><td>4.5</td><td></td><td>5.0 9.4</td><td>-9.0</td><td>5.4</td><td>6.8</td><td>2.1</td><td>3.8</td><td>5.1</td><td>3.6</td></tr><tr><td rowspan="5">PSNR1 (dB) â</td><td>GO-SLAM</td><td>2.9 28.4</td><td>2.1 _28.2</td><td>5.9 19.6</td><td>4.9 _22.4</td><td>4.6 23.9</td><td>7.3 _20.8_</td><td>5.3 24.8</td><td>4.4 _13.2_</td><td>4.7 _22.7</td><td>1.0 16.0</td><td>0.4</td><td>0.7 15.4</td><td>0.7 15.7</td></tr><tr><td>MonGS</td><td>25.3</td><td>29.1</td><td>22.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>17.7.8</td><td>15.7 114.1</td><td></td><td></td></tr><tr><td>Ours (Direct)</td><td>24.6</td><td></td><td></td><td>24.3</td><td>15.2</td><td>21.9</td><td>-8.2</td><td>22.4</td><td>21.1</td><td></td><td></td><td>18.1</td><td>16.7</td></tr><tr><td>Photo-SLAM</td><td></td><td>28.7</td><td>26.5</td><td>25.5</td><td>15.9</td><td>24.3</td><td>20.9</td><td>23.9</td><td>23.8</td><td>17.6</td><td>19.8</td><td>15.9</td><td>17.8</td></tr><tr><td>Ours (ORB-SLAM)</td><td>35.8 33.4</td><td>37.3 34.6</td><td>30.2 27.7</td><td>30.9 28.1</td><td>33.2</td><td>29.0</td><td>31.0</td><td>32.3 28.3</td><td>32.4</td><td>20.5</td><td>23.0</td><td>18.0</td><td>20.5</td></tr><tr><td rowspan="5">SSIM2 â</td><td>GO-SLAM</td><td>0.76</td><td>0.76</td><td>0.63</td><td>_0.68</td><td>27.5 0.76</td><td>27.0 0.54</td><td>28.4 0.72</td><td>0.44</td><td>29.4 0.66</td><td>18.5 0.46</td><td>19.8</td><td>17.1 0.44</td><td>18.5 0.45</td></tr><tr><td>MonoGS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>_0.45</td><td></td><td></td></tr><tr><td></td><td>00.67</td><td>0.86</td><td>0.81</td><td>0.82</td><td>0.40</td><td>0.71</td><td>0.16</td><td>0.80</td><td>0.65</td><td>0.66</td><td>0.54</td><td>0.67</td><td>0.62</td></tr><tr><td>Ours (Direct)</td><td>0.81</td><td>0.84</td><td>0.85</td><td>0.85</td><td>0.81</td><td>0.78</td><td>0.75</td><td>0.85</td><td>0.82</td><td>0.63</td><td>0.68</td><td>0.61</td><td>0.64</td></tr><tr><td>Photo-SLAM Ours (ORB-SLAM)</td><td>0.95</td><td>0.95</td><td>0.92</td><td>0.91</td><td>0.93</td><td>0.85</td><td>0.90</td><td>0.93</td><td>0.92</td><td>0.73</td><td>0.78</td><td>0.65</td><td>0.72</td></tr><tr><td rowspan="5">LPIPS3 â</td><td>GO-SLAM</td><td>0.91 0.46</td><td>0.90 0.42</td><td>0.88 0.48</td><td>0.88 0.49</td><td>0.91 0.49</td><td>0.82 0.58</td><td>0.86 0.50</td><td>0.89</td><td>0.88</td><td>0.65 0.56</td><td>0.68</td><td>0.64 0.63</td><td>0.66</td></tr><tr><td>MonoGS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.69</td><td>0.51</td><td></td><td>0.54</td><td></td><td>0.57</td></tr><tr><td></td><td>0.40</td><td>0.28</td><td>0.36</td><td>0.27</td><td>0.59</td><td>0.37</td><td>0.78</td><td>0.38</td><td>0.43</td><td>0.39</td><td>0.49</td><td>0.42</td><td>0.43</td></tr><tr><td>Ours (Direct)</td><td>0.35</td><td>0.27</td><td>0.27</td><td>0.23</td><td>0.41</td><td>0.26</td><td>0.39</td><td>0.27</td><td>0.31</td><td>0.45</td><td>0.43</td><td>0.56</td><td>0.48</td></tr><tr><td>Photo5SLA</td><td>0.07</td><td>0.06</td><td>0.10</td><td>0.10</td><td>0.08</td><td>0.12</td><td>0.08</td><td>0.08</td><td>0.09</td><td>0.25</td><td>0.14</td><td>0.29</td><td>0.22</td></tr><tr><td rowspan="5">ATE (cm) â</td><td>Ours (ORB-SLAM) GO-SLAM</td><td>0.20</td><td>0.21 0.7</td><td>0.24 _0.4</td><td>0.20 0.5_</td><td>0.21</td><td>0.22 0.5</td><td>0.22 _0.3</td><td>0.20 0.3</td><td>0.21 0.5</td><td>0.43</td><td>0.44</td><td>0.50 1.8</td><td>0.46</td></tr><tr><td>MonoGS</td><td></td><td>_0.3</td><td></td><td></td><td></td><td>_0.8</td><td></td><td></td><td></td><td></td><td>2.3</td><td>0.3</td><td></td><td>1.5</td></tr><tr><td></td><td>40.7</td><td>30.5</td><td>56.3</td><td></td><td>19.2</td><td>58.3</td><td>19.9</td><td>32.9 34.7</td><td></td><td>36.6</td><td>2.9</td><td>20.8</td><td>10.4</td><td>11.4</td></tr><tr><td>Ours (Direct)</td><td>14.0</td><td>12.9</td><td>7.8</td><td></td><td>4.2 187.4 0.5</td><td>12.3 0.7</td><td>0.4</td><td>41.2 0.7</td><td>7.3 0.2</td><td>35.9 0.6</td><td>8.0 1.6</td><td>2.6 0.4</td><td>39.1 1.6</td><td>16.6 1.2</td></tr><tr><td>hoSLAM Ours (ORB-SLAM)</td><td></td><td>0.6 0.4 0.5</td><td>0.3</td><td>1.0 1.6 1.0</td></table>

1-3 Peak Signal-to-Noise Ratio, Structural Similarity Index Measure [38], and Learned Perceptual Image Patch Similarity [39].

We benchmarked all frameworks on Replica [37], a highly detailed synthetic dataset that provides noiseless RGB images for estimating the upper bound performance of the methods, and TUM RGB-D [36], a real-world dataset for testing the methods under noisy images from a Kinect camera. Similar to prior works, we select eight sequences from Replica: office_0-4 and room_0-2 and three from TUM RGB-D: fr1_desk, fr2_xyz and fr3_office.

## B. Memory Usage

In this section, we compare the memory usage of GEVO against prior frameworks. The total memory is comprised of: 1) the map, which is the size of the resulting NeRF or Gaussians, and 2) the overhead, which is the extra memory for storing input and temporal variables during the execution in order to produce the output (the map). Note that the overhead memory is highly dependent on the algorithm and its implementation. To reduce the impact of implementation, we only track variables that are essential to the algorithm4 and avoid ones cached solely for acceleration.

The overhead memory of GEVO and other methods is summarized in Table I. Recall that all methods store keyframes in full resolution. Overall, GEVO requires the lowest average overhead memory of 83.5 MB on Replica and 31.4 MB on TUM, which are dominated by the keyframes stored in the current window. In MonoGS and Photo-SLAM, many keyframes from both current and past windows are stored in memory to periodically train Gaussians to reduce catastrophic forgetting. For these frameworks, the keyframes dominate (96% to 99%) the overhead memory. By only storing images in the current sliding window, GEVO reduces the overhead memory by 114 MB on average on TUM compared with MonoGS and Photo-SLAM. Since the image resolution is 2.66Ã higher on Replica, GEVO achieves an even higher overhead memory reduction (480 MB on average).

Among the benchmarked methods, GO-SLAM has the highest overhead memory due to the storage of large temporary variables in addition to all keyframe images. Specifically, with a tracker derived from DROID-SLAM, GO-SLAM computes a 4D correlation volume [40] between each pair of pixels of multiple feature maps to obtain the optical flow. With a size of at least 1 GB, the 4D correlation volume contributes to 50% to 60% of the overhead memory. Thus, GEVO reduces the overhead memory by up to 53Ã on Replica and 168Ã on TUM compared with GO-SLAM.

From Table I, the map memory of all prior frameworks only consumes a tiny fraction (0.5% to 2.4%) of the total memory. Thus, the total memory in each framework is dominated by the overhead memory, which our framework addresses. Recall that maps in GS-based frameworks (GEVO, MonoGS, and Photo-SLAM) consist of Gaussians whose number scales with the size of the environment. In contrast, GO-SLAM is a neuralbased method whose network size cannot adapt to the size of the environment. Thus, the map size of GO-SLAM for both datasets is constant at 48 MB, which is consistently 4.7Ã to 124Ã higher than GS-based frameworks for both datasets.

## C. Computational Efficiency

GEVO reduces memory overhead at the cost of a negligible increase in computation. To measure the compute overhead, we compare GEVO (Direct) and MonoGS in single-threaded CPU mode5 such that latency is proportional to the compute and is not hidden by multi-threading.

The average latency per image for GEVO is 340 ms and 621 ms per image on TUM and Replica, respectively, which is only 8% to 10% higher than MonoGS. In both works, rasterization dominates up to 70% of the computation time. Since GEVO only optimizes Gaussians in the active set (less than 15% of all Gaussians), a specialized rasterizer can be implemented in the future to greatly reduce overall latency.

## D. Rendering and Localization Accuracy

We compare the accuracy of GEVO against prior methods by computing the PSNR, SSIM [38], and LPIPS [39]6 metrics of every five non-keyframes in addition to the RMSE of average translation error (ATE) of keyframes. From Table I, GEVO (ORB-SLAM) achieves the second-best rendering and comparable localization accuracy on almost all sequences compared with the best-performing Photo-SLAM. Our PSNR degrades by 3 dB on Replica and 2 dB on TUM compared with Photo-SLAM due to the usage of rendered images, which are lower fidelity than original ones from camera.

Since our framework is compatible with many localization methods, we configure our framework using the same direct localization method as MonoGS for its comparison. Due to a more geometrically verified Gaussian initialization procedure based on traditional multi-view stereo, our framework is less likely to incur a large ATE compared with MonoGS, which randomly initializes Gaussians using priors from the global map. Due to our consistency-aware optimization, our framework often yields a better rendering accuracy in more than half of the sequences compared with MonoGS without training on up to 300 keyframes stored in memory.

Figure 5 shows sample rendering from the benchmarked sequences. The rendering of GEVO shows minimal visual difference compared with Photo-SLAM and suffers from fewer artifacts than MonoGS. Despite robust localization, GO-SLAM shows the most noticeable artifacts, especially on TUM, which contains motion blur and lighting changes.

TABLE II  
IMPACTS OF VARIOUS TECHNIQUES IN GEVO ON THE PSNR FOR THE TUM RGB-D DATASET.
<table><tr><td colspan="2">Stored References Rendered References</td><td rowspan="2">â</td><td colspan="3">â â</td><td>â</td></tr><tr><td colspan="2">Two-Stage Optimization Selective Optimization Free Space Pruning</td><td>â V</td><td>â â</td><td>â â â</td><td></td></tr><tr><td rowspan="2">PSNR* (dB)</td><td>Initial</td><td>25.9 25.1 23.8</td><td>23.7</td><td>22.7</td><td>â 22.6</td><td>25.0</td></tr><tr><td>Final</td><td>15.5 17.0 16.5</td><td>18.4</td><td>18.8</td><td>19.0</td><td>19.4</td></tr></table>

\* To avoid the impacts of localization, ground truth poses are used.

## E. Ablation Studies

In this section, we show the impact of our proposed techniques on alleviating catastrophic forgetting. To measure forgetting, we evaluate the rendering PSNR of each keyframe upon its evacuation from the sliding window and after the full sequence is processed, which we label as the initial and final PSNR, respectively. Table II shows the initial and final PSNR averaged across all keyframes of three sequences in the TUM dataset. To avoid the impact of localization on map fidelity, we used the ground truth trajectory. Since the reduced resolution of the cost volume during Gaussian initialization has negligible effect on PSNR (i.e., less than 0.2 dB), we perform ablation study on remaining techniques.

Without retraining with any images outside the current sliding window (leftmost column of Table II), the rendering quality of keyframes degrades significantly over time due to catastrophic forgetting (see Figure 1b). Simply replacing stored images with rendered ones (column 2) alone noticeably increases the final PSNR. However, since rendered images degrade in fidelity over time, using them alone is not sufficient without additional techniques (columns 3-6). All techniques combined recover the final average PSNR to 19 dB (see Figure 1c), which is only 0.4 dB lower than storing and training with original images (rightmost column). Although each additional technique gradually lowers the initial PSNR due to stronger regularization, they tend to impede forgetting during optimization. Thus, all our techniques combined recover most of the loss in fidelity due to not storing and retraining using the original past images.

## V. CONCLUSION

In this letter, we presented GEVO, a memory-efficient GSbased monocular SLAM that avoids catastrophic forgetting due to incomplete sensor obscuration (IRO) and retrospective occlusion (RO) without storing past images. By using rendered images to guide the optimization and introducing occupancypreserving initialization and consistency-aware optimization to retain their fidelity, map consistency is maintained. Experiments on the TUM and Replica datasets show that while maintaining comparable rendering accuracy with SOTA methods, GEVO reduces the memory overhead to 58 MBs, which is up to 94Ã lower than prior methods. Thus, GEVO has made a significant step towards deploying GS-based SLAM on energy-constrained devices.

## REFERENCES

[1] M. Horowitz, â1.1 computingâs energy problem (and what we can do about it),â in 2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers (ISSCC), 2014, pp. 10â14.

[2] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[3] H. Matsuki, R. Murai, P. H. J. Kelly, and A. J. Davison, âGaussian Splatting SLAM,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[4] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, âOrb-slam3: An accurate open-source library for visual, visualâ Â´ inertial, and multimap slam,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[5] C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza, âOn-manifold preintegration for real-time visualâinertial odometry,â IEEE Transactions on Robotics, vol. 33, no. 1, pp. 1â21, 2016.

[6] A. J. Davison, I. D. Reid, N. D. Molton, and O. Stasse, âMonoslam: Real-time single camera slam,â IEEE transactions on pattern analysis and machine intelligence, vol. 29, no. 6, pp. 1052â1067, 2007.

[7] R. Mur-Artal and J. Tardos, âProbabilistic semi-dense mapping from highly accurate feature-based monocular slam,â in Proceedings of Robotics: Science and Systems, Rome, Italy, July 2015.

[8] A. Rosinol, J. J. Leonard, and L. Carlone, âNerf-slam: Real-time dense monocular slam with neural radiance fields,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 3437â3444.

[9] H. Matsuki, E. Sucar, T. Laidow, K. Wada, R. Scona, and A. J. Davison, âimode: Real-time incremental monocular dense mapping using neural field,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 4171â4177.

[10] Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and M. Pollefeys, âNicer-slam: Neural implicit scene encoding for rgb slam,â in 2024 International Conference on 3D Vision (3DV). IEEE, 2024, pp. 42â52.

[11] H. Huang, L. Li, C. Hui, and S.-K. Yeung, âPhoto-slam: Real-time simultaneous localization and photorealistic mapping for monocular, stereo, and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[12] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics (ToG), vol. 42, no. 4, pp. 1â14, 2023.

[13] G. Lowe, âSift-the scale invariant feature transform,â Int. J, vol. 2, no. 91-110, p. 2, 2004.

[14] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, âOrb: An efficient alternative to sift or surf,â in 2011 International conference on computer vision. Ieee, 2011, pp. 2564â2571.

[15] J. Engel, T. Schops, and D. Cremers, âLsd-slam: Large-scale direct Â¨ monocular slam,â in European conference on computer vision. Springer, 2014, pp. 834â849.

[16] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, âDtam: Dense tracking and mapping in real-time,â in 2011 International Conference on Computer Vision, 2011, pp. 2320â2327.

[17] G. Klein and D. Murray, âParallel tracking and mapping for small ar workspaces,â in 2007 6th IEEE and ACM international symposium on mixed and augmented reality. IEEE, 2007, pp. 225â234.

[18] J. Engel, V. Koltun, and D. Cremers, âDirect sparse odometry,â IEEE transactions on pattern analysis and machine intelligence, vol. 40, no. 3, pp. 611â625, 2017.

[19] S. Yang and S. Scherer, âMonocular object and plane slam in structured environments,â IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3145â3152, 2019.

[20] L. Nicholson, M. Milford, and N. Sunderhauf, âQuadricslam: Dual Â¨ quadrics from object detections as landmarks in object-oriented slam,â IEEE Robotics and Automation Letters, vol. 4, no. 1, pp. 1â8, 2018.

[21] A. Rosinol, M. Abate, Y. Chang, and L. Carlone, âKimera: an opensource library for real-time metric-semantic localization and mapping,â in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 1689â1696.

[22] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, âGo-slam: Global optimization for consistent 3d instant reconstruction,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3727â3737.

[23] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, âPhoto-slam: Realtime simultaneous localization and photorealistic mapping for monocular stereo and rgb-d cameras,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 584â21 593.

[24] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat, track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[25] P. Z. X. Li, S. Karaman, and V. Sze, âGmmap: Memory-efficient continuous occupancy map using gaussian mixture model,â IEEE Transactions on Robotics, vol. 40, pp. 1339â1355, 2024.

[26] K. Wang, W. Ding, and S. Shen, âQuadtree-accelerated real-time monocular dense mapping,â in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 1â9.

[27] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,âÂ¨ in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[28] J. L. Schonberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, âPixel- Â¨ wise view selection for unstructured multi-view stereo,â in European Conference on Computer Vision (ECCV), 2016.

[29] P. F. Felzenszwalb and D. P. Huttenlocher, âEfficient belief propagation for early vision,â International journal of computer vision, vol. 70, pp. 41â54, 2006.

[30] D. Min, S. Choi, J. Lu, B. Ham, K. Sohn, and M. N. Do, âFast global image smoothing based on weighted least squares,â IEEE Transactions on Image Processing, vol. 23, no. 12, pp. 5638â5653, 2014.

[31] B. Eckart, K. Kim, A. Troccoli, A. Kelly, and J. Kautz, âAccelerated generative models for 3d point cloud data,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 5497â 5505.

[32] C. OâMeadhra, W. Tabib, and N. Michael, âVariable resolution occupancy mapping using gaussian mixture models,â IEEE Robotics and Automation Letters, vol. 4, no. 2, pp. 2015â2022, 2018.

[33] A. Dhawale and N. Michael, âEfficient parametric multi-fidelity surface mapping,â in Robotics: Science and Systems (RSS), vol. 2, no. 3, 2020, p. 5.

[34] K. Goel, N. Michael, and W. Tabib, âProbabilistic point cloud modeling via self-organizing gaussian mixture models,â IEEE Robotics and Automation Letters, vol. 8, no. 5, pp. 2526â2533, 2023.

[35] S. R. Bulo, L. Porzi, and P. Kontschieder, âRevising densification in \` gaussian splatting,â arXiv preprint arXiv:2404.06109, 2024.

[36] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[37] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, âA benchmark for the evaluation of rgb-d slam systems,â in Proc. of the International Conference on Intelligent Robot Systems (IROS), Oct. 2012.

[38] J. Nilsson and T. Akenine-Moller, âUnderstanding ssim,â Â¨ arXiv preprint arXiv:2006.13846, 2020.

[39] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in CVPR, 2018.

[40] Z. Teed and J. Deng, âRaft: Recurrent all-pairs field transforms for optical flow,â in Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part II 16. Springer, 2020, pp. 402â419.