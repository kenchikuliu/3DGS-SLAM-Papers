# MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM

Vladimir Yugay

Theo Gevers

Martin R. Oswald

University of Amsterdam, Netherlands

{vladimir.yugay, th.gevers, m.r.oswald}@uva.nl

vladimiryugay.github.io/magic_slam

<!-- image-->  
Figure 1. MAGiC-SLAM is a multi-agent SLAM method capable of novel view synthesis. Given single-camera RGBD input streams from multiple simultaneously operating agents MAGiC-SLAM estimates their trajectories and reconstructs a 3D Gaussian map that can be rendered from previously unseen viewpoints. We showcase the high-fidelity 3D Gaussian map of a real-world environment alongside multiple agent trajectories (depicted in green, yellow, and blue) within it. Our method effectively utilizes information from multiple agents to achieve centimeter-level tracking accuracy. Our mapping and map merging strategies allow for realistic rendering of color and depth, significantly improving the state of the art. Unlike previous methods, MAGiC-SLAM is flexible in the number of agents it can handle.

## Abstract

Simultaneous localization and mapping (SLAM) systems with novel view synthesis capabilities are widely used in computer vision, with applications in augmented reality, robotics, and autonomous driving. However, existing approaches are limited to single-agent operation. Recent work has addressed this problem using a distributed neural scene representation. Unfortunately, existing methods are slow, cannot accurately render real-world data, are restricted to two agents, and have limited tracking accuracy. In contrast, we propose a rigidly deformable 3D Gaussian-based scene representation that dramatically speeds up the system. However, improving tracking accuracy and reconstructing a globally consistent map from multiple agents remains challenging due to trajectory drift and discrepancies across agentsâ observations. Therefore, we propose new tracking and map-merging mechanisms and integrate loop closure in the Gaussian-based SLAM pipeline. We evaluate MAGiC-SLAM on synthetic and real-world datasets and find it more accurate and faster than the state of the art.

## 1. Introduction

Visual Simultaneous Localization and Mapping (SLAM) enables a machine to build a 3D map of its surroundings while determining its location relying solely on its camera input. Imagine an autonomous car navigating a busy city or a robot exploring a cluttered room. For these systems to move safely and make decisions, they need a reliable understanding of their surroundings. Over the years, visual SLAM has advanced [9, 14, 37], learning to handle complex scenes with increasing accuracy. This capability forms the backbone for various systems from self-driving cars to virtual reality glasses, where spatial understanding is crucial for navigation, planning, and interactive experiences.

Recently, SLAM systems have been enhanced with novel view synthesis (NVS) capabilities [15, 23, 36], allowing them to generate realistic, immersive views of scenes. This allows more detailed scene exploration and supports the creation of high-quality virtual environments. However, these NVS-capable SLAM methods are slow, and tracking accuracy remains limited [15]. These limitations become even more pronounced in a multi-agent setup, with larger scenes and numerous observations.

Effectively processing information from multiple agents operating simultaneously opens up new possibilities for SLAM systems. First, this naturally accelerates 3D reconstruction by enabling horizontal parallelization; each agent can map different parts of the environment concurrently, leading to faster, more efficient coverage of large areas [5]. Second, agents can collaboratively refine their location estimates by sharing their observations [18]. Finally, the global map constructed from the perspectives of multiple agents is more geometrically consistent and accurate [33].

The only solution for multi-agent NVS-capable SLAM hitherto has combined a distributed neural scene representation with a traditional loop closure mechanism [11]. However, this approach faces several issues. Firstly, it fails to reconstruct a coherent map capable of accurate NVS. Neural scene representations inherently lack support for rigid body transformations [20] making it impossible to update and merge the global map efficiently. This in turn leads to poor rendering quality of the novel views. Secondly, localization accuracy remains limited, as neural-based camera pose optimization struggles with the variability and imperfections inherent to real-world data. Finally, the system is prohibitively slow, demands extensive computational resources, and supports only two agents.

We present MAGiC-SLAM, a multi-agent NVS-capable SLAM pipeline designed to overcome these limitations. Firstly, MAGiC-SLAM utilizes 3D Gaussians as the scene representation supporting rigid body transformations. Every agent processes the input RGBD sequence in chunks (sub-maps) which can be effectively corrected and merged into a coherent global map. Secondly, we implement a loop closure mechanism to improve trajectory accuracy by leveraging information from all agents. It is further enhanced with a novel loop detection module, based on a foundational vision model, which enables better generalization to unseen environments. Finally, our flexible tracking and mapping modules allow our system to achieve superior speed and easily scale to varying numbers of agents.

Overall, our contributions can be summarized as follows:

â¢ A multi-agent NVS-capable SLAM system for consistent 3D reconstruction supporting an arbitrary number of simultaneously operating agents

â¢ A loop closure mechanism for Gaussian maps leveraging a foundational vision model for loop detection

â¢ Efficient map optimization and fusion strategies that reduce required disk storage and processing time

â¢ A robust Gaussian-based tracking module

## 2. Related Work

Neural SLAM. Neural radiance fields (NeRF) [24] have achieved remarkable results in novel view synthesis [2, 25, 42]. [36] started a whole new line of research [20, 21,

32, 38, 41, 45, 47] by proposing a SLAM system using neural fields to represent the map and optimize the camera poses. Various neural field approaches have provided insights for neural SLAM developments. For instance, NICE-SLAM [47] uses a voxel grid to store neural features, while Vox-Fusion [41] improves the grid with adaptive sizing. Point-SLAM [32] attaches feature embeddings to point clouds on object surfaces, offering more flexibility and the ability to encode concentrated volume density. Co-SLAM [38] employs a hybrid representation combining coordinate encoding and hash grids to achieve smoother reconstruction and faster convergence. A group of methods [6, 30] use neural fields solely for mapping while relying on traditional feature point-based visual odometry for tracking. Loopy-SLAM [20] explores the usage of posegraph optimization to handle trajectory drift. While these methods have shown success in NVS, they are computationally intensive, slow, and struggle to render real-world environments accurately [15, 43]. Additionally, neural maps inherently lack support for rigid body transformations [20], which greatly limits the efficiency of map correction. In contrast, our scene representation is fast to render and optimize, significantly accelerating our SLAM pipeline. With native support for rigid body transformations, our approach enables faster map updates and seamless map merging.

Gaussian SLAM. Recently, 3D Gaussian Splatting(3DGS) [16] revolutionized novel view synthesis, achieving photorealistic real-time rendering at over 100 FPS without relying on neural networks. Compared to NeRFs, 3DGS is more efficient in terms of memory, and faster to optimize. These factors inspired the line of SLAM systems [12, 13, 15, 23, 40, 43] using 3D Gaussians instead of neural fields for tracking and mapping. [12, 23, 40] estimated camera pose by computing camera gradients from the 3DGS field analytically. Others [15, 43] design a warp-based camera pose optimization. Finally, [13, 29] use existing sparse SLAM systems to speed up tracking. While addressing many limitations of neural SLAM, the proposed methods overlook global consistency, leading to trajectory error accumulation [34]. In contrast, MAGiC-SLAM integrates a loop closure mechanism within the 3DGS SLAM pipeline to correct accumulated trajectory errors. It effectively generalizes to unknown environments leveraging a foundational vision model for loop detection. Concurrently, Zhu et al. [46] proposed a way to do loop closure in a 3DGS-based SLAM pipeline. However, the method cannot handle multiple agents, uses a standard loop closure detection mechanism, and is less efficient in tracking and mapping.

Multi-agent Visual SLAM. Despite significant progress in single-agent systems, multi-agent SLAM remains less developed due to the complexity of designing a multi-agent pipeline. Multi-agent SLAM can be categorized into two types: centralized and distributed. In a distributed system [5, 19] agents communicate sporadically. While being more robust in environments with stringent networking constraints, their performance is limited by the computing power of a separate agent. Centralized systems [18, 33] use a centralized server to manage agentsâ maps and globally optimize their trajectories. Despite the success of these methods, they do not provide a map with novel view synthesis capabilities. Hu et al. [11] proposed a centralized neural-based multi-agent NVS-capable SLAM to address this. However, it inherited all the problems associated with neural representations such as low speed, compute requirements, and ability to render real-world data [15]. In addition, CP-SLAM [11], can only support two agents operating simultaneously. In contrast, MAGiC-SLAM employs an efficient scene representation that enables seamless global map reconstruction with NVS capabilities suited for real-world environments. Additionally, it delivers improved tracking accuracy through robust tracking and loop closure modules. Finally, our system is flexible in handling multiple agents, constrained only by the capacity of the centralized server.

## 3. Method

We introduce MAGiC-SLAM, which architecture is shown in Figure 2. Every agent processes an RGB-D stream performing local mapping and tracking. 3D Gaussians are used to represent the agentsâ sub-maps and to improve tracking accuracy. The foundation vision model-based loop detection module extracts features from RGB images and sends them to the centralized server with the local sub-map data. The server detects the loops based on the image encodings, performs pose graph optimization, and sends the optimized poses back to the agents. At the end of the run, the server fuses the agentsâ sub-maps into a global Gaussian map. This section introduces per-agent mapping and tracking mechanisms and describes global loop closure and map construction processes.

## 3.1. Mapping

Every agent processes a single sub-map of a limited size, represented as a collection of 3D Gaussians. Sub-maps are initialized from the first frame. $\theta _ { \mathrm { s a m p l e } }$ points sampled from the lifted to 3D RGBD frame serve as the means for new 3D Gaussians. New Gaussians are added to regions of the active sub-map with low Gaussian density, based on rendered opacity, and are optimized using the loss:

$$
L _ { \mathrm { m a p p i n g } } = \lambda _ { \mathrm { c o l o r } } \cdot L _ { \mathrm { c o l o r } } + \lambda _ { \mathrm { d e p t h } } \cdot L _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { r e g } } \cdot L _ { \mathrm { r e g } }\tag{1}
$$

where are $\lambda _ { * } ~ \in ~ R$ are hyperparameters. The color loss $L _ { \mathrm { c o l o r } }$ is is defined as:

$$
\boldsymbol { L _ { \mathrm { c o l o r } } } = ( 1 - \lambda ) \cdot | \hat { I } - I | _ { 1 } + \lambda \big ( 1 - \mathrm { S S I M } ( \boldsymbol { \hat { I } } , I ) \big )\tag{2}
$$

where ${ \hat { I } } , I$ are the rendered and input images respectively, and $\lambda \ \in \ R$ is the weighting factor. Depth loss $L _ { \mathrm { d e p t h } }$ is formulated as:

$$
{ \cal L } _ { \mathrm { d e p t h } } = | \hat { D } - D | _ { 1 } \ ,\tag{3}
$$

where $\hat { D } , D$ are the rendered and input depth maps. The regularization loss $L _ { \mathrm { r e g } }$ is represented as:

$$
L _ { \mathrm { r e g } } = | K | ^ { - 1 } \sum _ { k \in K } | s _ { k } - \overline { { s } } _ { k } | _ { 1 } ,\tag{4}
$$

where $s _ { k } ~ \in ~ \mathbb { R } ^ { 3 }$ is the scale of a 3D Gaussian, $\overline { { s } } _ { k }$ is the mean sub-map scale, and $| K |$ is the number of Gaussians in the sub-map. We do not optimize the spherical harmonics of the Gaussians to reduce their memory footprint and improve tracking accuracy [23].

A new sub-map is created, and the previous one is sent to the server after every $\theta _ { \mathrm { s u b m a p } }$ frame. Creating new sub-maps in already mapped areas introduces some computational redundancy, but it limits overall computational cost and maintains tracking and mapping speed as the scene expands. Moreover, unlike in [43], only Gaussians with zero rendered opacity in the current camera frustum are dispatched to the server. This approach significantly reduces the disk space required to store the sub-maps and speeds up the map merging process. The supplementary material provides a quantitative evaluation of the strategy.

## 3.2. Tracking

Typically GS SLAM systems optimize camera pose either explicitly or implicitly. In the explicit case, the camera pose gradient is derived [22] or approximated [12, 40] analytically. In the implicit case [15, 43], the relative pose between two frames is optimized by warping the GS point clouds and minimizing re-rendering depth and color errors. In both cases, the camera pose is estimated based on the existing map, following a frame-to-model paradigm [14].

Considering the advantages and disadvantages of frameto-frame and frame-to-model tracking paradigms [9, 14, 37] and the convenience of using sub-maps for loop closure, we propose a hybrid implicit tracking approach that combines the strengths of both. Specifically, we initialize the relative pose using a deterministic frame-to-frame dense registration and then refine it through frame-to-model optimization. This approach differs from previous NVS SLAM methods, initializing the relative pose based on a constant speed assumption.

Moreover, we found implicit tracking to be more accurate than the explicit approach given robust pose initialization. At the same time, explicit tracking methods do not benefit from pose initialization since camera poses are optimized across the whole co-visibility window at every mapping step. We refer to supplementary material for experiments highlighting this phenomenon.

<!-- image-->  
Figure 2. MAGiC-SLAM Architecture. Agent Side: Each agent processes a separate RGBD stream, maintaining a local sub-map and estimating its trajectory. When an agent starts a new sub-map, it sends the previous sub-map and image features to the centralized server. Server Side: The server stores the image features and sub-maps from all agents and performs loop closure detection, loop constraint estimation, and pose graph optimization. It then updates the stored sub-maps and returns the optimized poses to the agents. Once the algorithm completes (denoted by green arrows), the server merges the accumulated sub-maps into a single unified map and refines it.

Pose Initialization. At time t, given the input colored point cloud $P _ { t } ,$ , the goal is to estimate its pose $T _ { t - 1 , t } \in S E ( 3 )$ relative to the previous input point cloud $P _ { t - 1 }$ . The registration process is performed iteratively at several scales, starting from the coarsest, $l = 0$ , down to the finest, $l = L$ At every scale, both point clouds are voxelized, and the set of correspondences $K = \{ ( p , q ) \}$ } between $P _ { t } , P _ { t - 1 }$ is computed using ICP [3]. For every scale, a loss function:

$$
\begin{array} { r } { E ( \mathbf { T _ { t - 1 , t } } ) = ( 1 - \sigma ) \displaystyle \sum _ { ( \mathbf { p } , \mathbf { q } ) \in \mathcal { K } } \left( r _ { C } ^ { ( \mathbf { p } , \mathbf { q } ) } ( \mathbf { T _ { t - 1 , t } } ) \right) ^ { 2 } } \\ { + \sigma \displaystyle \sum _ { ( \mathbf { p } , \mathbf { q } ) \in \mathcal { K } } \left( r _ { G } ^ { ( \mathbf { p } , \mathbf { q } ) } ( \mathbf { T _ { t - 1 , t } } ) \right) ^ { 2 } , } \end{array}\tag{5}
$$

is optimized where $\sigma \in [ 0 , 1 ]$ is a scalar weight. $r _ { G } ^ { \left( p , q \right) }$ is a geometric residual defined as:

$$
r _ { G } ^ { ( p , q ) } = \Big ( \big ( T _ { t - 1 , t } ( q ) - p \big ) n _ { p } \Big ) ^ { 2 } .\tag{6}
$$

where $n _ { p }$ is a normal vector of $p .$ The color residual $r _ { C } ^ { \left( p , q \right) }$ is computed as:

$$
\begin{array} { r } { r _ { C } ^ { ( p , q ) } = C _ { p } \Big ( f _ { p } \big ( T _ { t , t - 1 } ( q ) \big ) - p \Big ) - C ( q ) , } \end{array}\tag{7}
$$

where $f _ { p }$ is a function projecting a point on the tangent plane of $p , C ( q )$ is the intensity of point $q ,$ and $C _ { p }$ is a continuous intensity function defined over point cloud $P _ { t }$ . The loss is optimized until convergence with the Gauss-Newton method. For more details about the registration mechanism please refer to [28].

Pose Refinement. The pose obtained in the initialization step is further refined using re-rendering losses [43] of the scene. To refine the initial pose estimate, we freeze all Gaussian parameters and minimize the loss:

$$
\underset { T _ { t - 1 , t } } { \arg \operatorname* { m i n } } L _ { \mathrm { t r a c k i n g } } \Big ( \hat { I } ( T _ { t - 1 , t } ) , \hat { D } ( T _ { t - 1 , t } ) , I _ { t } , D _ { t } , \alpha _ { t } \Big )\tag{8}
$$

where $\hat { I } ( T _ { t - 1 , t } )$ and $\hat { D } ( T _ { t - 1 , t } )$ are the rendered color and depth from the sub-map warped with the relative transformation $T _ { t - 1 , t } , I _ { t }$ and $D _ { t }$ are the input color and depth map at frame t.

We use soft alpha and color rendering error masking to avoid contaminating the tracking loss with pixels from previously unobserved or poorly reconstructed areas [15, 43]. Soft alpha mask $M _ { \mathrm { a l p h a } }$ is a polynomial of the alpha map rendered from the 3D Gaussians. Error boolean mask $M _ { \mathrm { i n l i e r } }$ discards all the pixels where the color and depth errors are larger than a frame-relative error threshold:

$$
L _ { \mathrm { t r a c k i n g } } = \sum M _ { \mathrm { i n l i e r } } \cdot M _ { \mathrm { a l p h a } } \cdot \left( \lambda _ { c } | \hat { I } - I | _ { 1 } + ( 1 - \lambda _ { c } ) | \hat { D } - D | _ { 1 } \right) .\tag{9}
$$

The weighting ensures the optimization is guided by wellreconstructed regions where the accumulated alpha values are close to 1 and rendering quality is high.

## 3.3. Loop Closure

Loop closure is the process that detects when a system revisits a previously mapped area and adjusts the map and camera poses to reduce accumulated drift, ensuring global consistency. It includes four key steps: loop detection, loop constraint estimation, pose graph optimization, and integration of optimized poses. Loop detection identifies when a previously mapped location is revisited. Loop constraint estimation calculates the relative pose between frames in the loop. Pose graph optimization then refines all camera poses, minimizing discrepancies between odometry and loop constraints to maintain global consistency. Finally, the optimized poses are integrated into the reconstructed map.

Loop Detection. Throughout the run, each agent extracts features from the first frame in each sub-map and sends them to a centralized server. The features are stored in a GPU database [8] optimized for similarity search. The server queries the database for potential loop candidates for every new sub-map frame. A pair of frames is considered a loop if the distance between them is less than a threshold $\theta _ { \mathrm { f e a t u r e } }$ in the image feature space. Loops belonging to the same agent are additionally filtered based on time threshold $\theta _ { \mathrm { t i m e } }$ to avoid too many uninformative loops.

Loop detection heavily influences the accuracy of loop edge constraint estimation since large frame overlap is crucial for registration [20]. Common approaches for loop closure detection are based on ORB [10] or NetVLAD [1] image descriptors. Hu et al. [11] argue that NetVLAD global image descriptors are better suited for loop closure detection. However, NetVLAD is trained on a relatively small dataset which leads to a lack of generalization to unknown environments. To overcome this limitation we propose to use a foundational vision model as a feature extractor. We use a small variation of DinoV2 [26] because of the large amount of data it was trained on, its compactness, and the quality of the features it produces for the downstream tasks. Loop Constraints. We use a coarse-to-fine registration approach to estimate loop edge constraints. For the coarse alignment, we apply the global registration method of Rusu et al. [31], which extracts Fast Point Feature Histograms (FPFH) from downsampled versions of the source $( P _ { s } )$ and target $( P _ { t } )$ point clouds. Correspondence search is then performed in the FPFH feature space rather than in Euclidean space. Optimization is embedded in a RANSAC framework to reject outlier correspondences, producing a rigid transformation of the source point cloud $S _ { s }$ to align with the target $S _ { t }$ . Finally, ICP [3] is applied to the full-resolution point clouds to refine the coarse alignment estimate.

We found that directly registering Gaussian means from different agents is unreliable, as Gaussians representing overlapping regions can have widely varying distributions across agents. To address this, we anchor an input point cloud at the start of each sub-map and use it for registration. Pose Graph Optimization. Each node in a pose graph, $T _ { i } ~ \in ~ S E ( 3 )$ , represents a distinct sub-map. Neighboring sub-maps are connected by odometry edges, derived from the tracker, representing the relative transformations between them. Loop edge constraints $T _ { s t } ~ \in ~ S E ( 3 )$ are added between non-adjacent sub-maps and computed using a registration method different from that of the tracker. The error term between two nodes i and $j$ is defined as:

$$
e _ { i j } = \log \big ( T _ { i j } ^ { - 1 } ( T _ { i } ^ { - 1 } T _ { j } ) \big ) ,\tag{10}
$$

where $T _ { i j }$ is the relative transformation between the nodes i and $j , T _ { i } , T _ { j }$ are the node poses, and log is the logarithmic map from ${ \bar { S } } E ( 3 )$ to $s e ( 3 )$ . To get the optimized camera poses we minimize the error term:

$$
F ( T ) = \sum _ { \langle i , j \rangle \in \mathcal { C } } \left( e ( T _ { i } , T _ { j } ) ^ { \top } \Omega _ { i j } e ( T _ { i } , T _ { j } ) \right)\tag{11}
$$

where $\Omega _ { i j } \in R ^ { 6 \times 6 }$ is a positive semi-definite information matrix reflecting the uncertainty of the constraint estimate - the higher the confidence of an edge, the larger the weight is applied to the residual. The error terms are linearized using a first-order Taylor expansion, and the loss is minimized with the Gauss-Newton method. For further details, please see [17].

Pose Update Integration. The pose graph optimization module provides pose corrections $\{ T _ { c } ^ { i } \stackrel { - } { \in } S E ( 3 ) \} _ { i = 1 } ^ { N _ { s } }$ 1 for every sub-map of every agent. All camera poses $\{ T _ { j } ^ { i } \} _ { j = 1 } ^ { N _ { p } }$ belonging to a sub-map i are corrected as:

$$
T _ { j } ^ { i } \gets T _ { i } ^ { c } T _ { j } ^ { i } .\tag{12}
$$

All Gaussians $\{ G _ { j } ^ { i } ( \mu _ { j } ^ { i } , \Sigma _ { j } ^ { i } , o _ { j } ^ { i } , c _ { j } ^ { i } ) \} _ { j = 1 } ^ { N _ { g } }$ 1 belonging to submap i are updated as well:

$$
\mu _ { j } ^ { i }  T _ { i } ^ { c } \mu _ { j } ^ { i } , \quad \Sigma _ { j } ^ { i }  T _ { i , R } ^ { c } \Sigma _ { j } ^ { i } ,\tag{13}
$$

where $T _ { i , R } ^ { c }$ is a rotation component of $T _ { i } ^ { c } \in S E ( 3 )$ . We do not correct Gaussian colors since we do not optimize spherical harmonics (Subsection 3.1).

## 3.4. Global Map Construction

Once the agents complete processing their data, the server merges the sub-maps from all agents into a unified global map. The map is merged in two stages: coarse and fine. During the coarse stage, the server loads the cached submaps and appends them into a single global map. Caching Gaussians that are not visible from the first keyframe of the next sub-map allows for appending the Gaussians without the need for costly intersection check [43]. However, this results in visual artifacts at the edges of the renderings. This happens because some Gaussians with zero opacity for a given view still influence the rendering through the projected 2D densities at the edges. Additionally, the coarse merging of sub-maps might introduce geometric artifacts at their intersections. The fine merging stage addresses these issues by optimizing the Gaussian parameters using color and depth rendering losses for a small number of iterations and pruning Gaussians with zero opacity. The visual effects of the refining step are shown in Fig. 3.

## 4. Experiments

We describe our experimental setup and compare MAGiC-SLAM with state-of-the-art baselines. We assess tracking and rendering performance on synthetic and real-world multi-agent datasets and provide ablation studies on key components of our pipeline. Please refer to the supplementary material for the implementation details and hyperparameters for tracking, mapping, and loop closure modules. Baselines. To evaluate tracking, following [11] we compare our method to the state-of-the-art multi-agent SLAM systems like SWARM-SLAM [18], CCM-SLAM [33], and CP-SLAM [11]. To make the comparison more comprehensive, we include several single-agent systems like Gaussian-SLAM [43], MonoGS [23], and Orb-SLAM3 [4]. We include Orb-SLAM3 since it is one of the most popular and reliable single-agent SLAM methods, Gaussian-SLAM since our approach uses sub-maps, and MonoGS as the most accurate single-agent NVS SLAM system [37]. We evaluate rendering against CP-SLAM [11] since it is the only existing multi-agent NVS-capable SLAM system.

<!-- image-->  
(a) Visual artifacts

<!-- image-->  
(b) Refined view

<!-- image-->  
(c) Geometric artifacts

<!-- image-->  
(d) Refined view  
Figure 3. Map Merging. Our coarse-to-fine strategy effectively removes (a) visual artifacts caused by the GS mechanism and (c) geometric artifacts resulting from Gaussian sub-map intersections.

Datasets. We test our method on the MultiagentReplica [11] dataset which contains four two-agent RGB-D sequences in a synthetic environment. Each sequence consists of 2500 frames, except for the Office-0 scene, which has 1500 frames. MultiagentReplica is a synthetic dataset and does not have a test set for novel view synthesis evaluation. Therefore, we extend our evaluation to real-world scenes using the ego-centric Aria [27] dataset. The Aria dataset provides ground-truth depth and camera information from recordings of two rooms within a real-world environment. We selected this dataset for its high-quality groundtruth data and the growing prevalence of egocentric wearable devices. However, the original dataset includes many dynamic objects, and since our method is not designed for dynamic environments, this restricted the amount of usable data. We selected three sequences from every room to simulate multi-agent operations, choosing sequences with sufficient consecutive frames without dynamic objects. We then extracted 500 consecutive frames from each sequence for training. We sampled 100 unseen frames from each room to test our methodâs NVS capabilities. We refer to this dataset as AriaMultiagent in all tables and figures.

Evaluation Metrics. To assess tracking accuracy, we use ATE RMSE [35], and for rendering we compute PSNR, SSIM [39] and LPIPS [44]. Rendering metrics on ReplicaMultiagent are evaluated by rendering full-resolution images along the estimated trajectory with mapping intervals similar to [32]. The same metrics over training and holdout test frames are used for the AriaMultiagent dataset. We evaluate the depth error using the L1 norm in centimeters.

## 4.1. Tracking Performance

In Table 1 we compare our method against state-of-theart multi-agent SLAM systems on the ReplicaMultiagent dataset processing two agents simultaneously. In Table 2 we evaluate MAGiC-SLAM on the AriaMultiagent dataset processing three agents simultaneously. Since CP-SLAM does not support the operation of more than two agents, we report its results on the first two out of three agents. To make our evaluation more comprehensive, we compare our method against single-agent SLAM systems in Table 3. We run single-agent systems on each agent separately and report their average performance. MAGiC-SLAM achieves superior tracking accuracy thanks to our novel two-stage tracking mechanism. The loop closure mechanism effectively utilizes multi-agent information to enhance accuracy even further.

<table><tr><td>Method</td><td>Agent</td><td>Off-0</td><td>Apt-0</td><td>Apt-1</td><td>Apt-2</td></tr><tr><td>CCM-SLAM [33]</td><td>Agent 1</td><td>9.84</td><td>X</td><td>2.12</td><td>0.51</td></tr><tr><td>Swarm-SLAM [18]</td><td></td><td>1.07</td><td>1.61</td><td>4.62</td><td>2.69</td></tr><tr><td>CP-SLAM [11]</td><td></td><td>0.50</td><td>0.62</td><td>1.11</td><td>1.41</td></tr><tr><td>MAGiC-SLAM (w.o. Loop Closure)</td><td></td><td>0.44</td><td>0.30</td><td>0.48</td><td>0.91</td></tr><tr><td>MAGiC-SLAM</td><td></td><td>0.31</td><td>0.13</td><td>0.21</td><td>0.42</td></tr><tr><td>CCM-SLAM [33]</td><td>Agent 2</td><td>0.76</td><td>X</td><td>9.31</td><td>0.48</td></tr><tr><td>Swarm-SLAM [18]</td><td></td><td>1.76</td><td>1.98</td><td>6.50</td><td>8.53</td></tr><tr><td>CP-SLAM [11]</td><td></td><td>0.79</td><td>1.28</td><td>1.72</td><td>2.41</td></tr><tr><td>MAGiC-SLAM (w.o. Loop Closure)</td><td></td><td>0.41</td><td>0.46</td><td>0.61</td><td>0.41</td></tr><tr><td>MAGiC-SLAM</td><td></td><td>0.24</td><td>0.21</td><td>0.30</td><td>0.22</td></tr><tr><td>CCM-SLAM [33]</td><td>Average</td><td>5.30</td><td>X</td><td>5.71</td><td>0.49</td></tr><tr><td>Swarm-SLAM [18]</td><td></td><td>1.42</td><td>1.80</td><td>5.56</td><td>5.61</td></tr><tr><td>CP-SLAM [11]</td><td></td><td>0.65</td><td>0.95</td><td>1.42</td><td>1.91</td></tr><tr><td>MAGiC-SLAM (w.o. Loop Closure)</td><td></td><td>0.42</td><td>0.38</td><td>0.54</td><td>0.66</td></tr><tr><td>MAGiC-SLAM</td><td></td><td>0.27</td><td>0.16</td><td>0.26</td><td>0.32</td></tr></table>

Table 1. Tracking performance on ReplicaMultiagent [11] dataset (ATE RMSE [cm]â). Comparison between MAGiC-SLAM (w.o. Loop Closure) and MAGiC-SLAM reveals the importance of loop closure for trajectory consistency. â indicates invalid results due to the failure of CCM-SLAM.

## 4.2. Rendering Performance

We evaluate the rendering performance on the merged scene obtained from all the agents on ReplicaMultiagent in Table 4 and AriaMultiagent datasets in Table 5. Since CP-SLAM does not support the operation of more than two agents, we report its results on the first two out of three agents in the AriaMultiagent dataset. Thanks to 3D Gaussian splatting, agentsâ sub-maps can accurately render realworld data. Moreover, they are efficiently corrected using optimized poses from the loop closure module. Combined with an effective map-merging strategy, this approach allows our method to significantly outperform previous work in rendering both real-world training data and novel views. We also provide qualitative results in Fig. 4.

<table><tr><td>Method Agent</td><td>Room0</td><td>Room1</td></tr><tr><td>CCM-SLAM [33] Agent 1</td><td>X</td><td>X</td></tr><tr><td>Swarm-SLAM [18]</td><td>6.11</td><td>4.29</td></tr><tr><td>CP-SLAM [11]</td><td>0.68</td><td>5.06</td></tr><tr><td>MAGiC-SLAM (w.o. Loop Closure)</td><td>0.92</td><td>1.15</td></tr><tr><td>MAGiC-SLAM</td><td>0.67</td><td>0.96</td></tr><tr><td>CCM-SLAM [33] Agent 2</td><td>X</td><td>X</td></tr><tr><td>Swarm-SLAM [18]</td><td>8.43</td><td>4.95</td></tr><tr><td>CP-SLAM [11]</td><td>5.39</td><td>0.68</td></tr><tr><td>MAGiC-SLAM (w.o. Loop Closure)</td><td>1.72</td><td>1.33</td></tr><tr><td>MAGiC-SLAM</td><td>1.13</td><td>0.53</td></tr><tr><td>CCM-SLAM [33] Agent 3</td><td>X</td><td>X</td></tr><tr><td>Swarm-SLAM [18] CP-SLAM [11]</td><td>4.82</td><td>5.12</td></tr><tr><td>MAGiC-SLAM (w.o. Loop Closure)</td><td></td><td></td></tr><tr><td>MAGiC-SLAM</td><td>3.76 1.67</td><td>1.25 0.46</td></tr><tr><td>CCM-SLAM [33] Average</td><td></td><td></td></tr><tr><td>Swarm-SLAM [18]</td><td>X</td><td>X</td></tr><tr><td>CP-SLAM [11]</td><td>6.45</td><td>4.78</td></tr><tr><td>MAGiC-SLAM (w.o. Loop Closure)</td><td>3.03</td><td>2.87</td></tr><tr><td>MAGiC-SLAM</td><td>2.13</td><td>1.24</td></tr><tr><td></td><td>1.15</td><td>0.65</td></tr></table>

Table 2. Tracking performance on AriaMultiagent dataset (ATE RMSE [cm]â). Our method shows strong performance on real-world data. â indicates invalid results due to the failure of CCM-SLAM. â indicates that CP-SLAM does not support the operation of more than two agents.

<table><tr><td rowspan="2">Methods</td><td colspan="3">AriaMultiagent</td><td colspan="5">ReplicaMultiagent</td></tr><tr><td>Room0 Room1 Avg.</td><td></td><td></td><td></td><td>Offo Apt0 Apt1</td><td></td><td>Apt2</td><td>Avg.</td></tr><tr><td>ORB-SLAM3 [4]</td><td>3.18</td><td>2.85</td><td>3.01</td><td>0.60</td><td>1.07</td><td>4.94</td><td>1.36</td><td>1.99</td></tr><tr><td>Gaussian-SLAM [43]</td><td>X</td><td>X</td><td>X</td><td>0.33</td><td>0.41</td><td>30.13</td><td>121.96</td><td>38.21</td></tr><tr><td>MonoGS [23]</td><td>1.90</td><td>2.71</td><td>2.30</td><td>0.38</td><td>0.21</td><td>3.33</td><td>0.54</td><td>1.15</td></tr><tr><td>MAGiC-SLAM w.0. LC</td><td>2.13</td><td>1.24</td><td>1.69</td><td>0.42</td><td>0.38</td><td>0.54</td><td>0.66</td><td>0.50</td></tr><tr><td>MAGiC-SLAM</td><td>1.15</td><td>0.65</td><td>0.90</td><td></td><td>0.27 0.16</td><td>0.26</td><td>0.32</td><td>0.25</td></tr></table>

Table 3. Tracking performance compared to single-agent methods(ATE RMSE [cm]â). We compare our method with traditional and state-of-the-art Gaussian-based single-agent SLAM systems. Our method outperforms all the baselines even without loop closure (LC).

## 4.3. Ablation Studies

Effect of Pose Initialization. In Table 6 we numerically evaluate our two-stage tracking mechanism on the Aria-

<table><tr><td>Methods</td><td>Metrics</td><td>Off-0</td><td>Apt-0</td><td>Apt-1</td><td>Apt-2</td><td>Avg.</td></tr><tr><td rowspan="4">CP-SLAM [11]</td><td>PSNR [dB] â</td><td>28.56</td><td>26.12</td><td>12.16</td><td>23.98</td><td>22.71</td></tr><tr><td>SSIM â</td><td>0.87</td><td>0.79</td><td>0.31</td><td>0.81</td><td>0.69</td></tr><tr><td>LPIPS â</td><td>0.29</td><td>0.41</td><td>0.97</td><td>0.39</td><td>0.51</td></tr><tr><td>Depth L1 [cm] â</td><td>2.74</td><td>19.93</td><td>66.77</td><td>2.47</td><td>22.98</td></tr><tr><td rowspan="4">MAGiC-SLAM</td><td>PSNR [dB] â</td><td>39.32</td><td>36.96</td><td>30.01</td><td>30.73</td><td>34.26</td></tr><tr><td>SSIM â</td><td>0.99</td><td>0.98</td><td>0.95</td><td>0.96</td><td>0.97</td></tr><tr><td>LPIPS â</td><td>0.05</td><td>0.09</td><td>0.18</td><td>0.17</td><td>0.12</td></tr><tr><td>Depth L1 [cm] â</td><td>0.41</td><td>0.64</td><td>3.16</td><td>0.99</td><td>1.30</td></tr></table>

Table 4. Training view synthesis performance on Replica-Multiagent dataset. The global map built by merging the maps from two agents is evaluated by synthesizing training views. Our method significantly outperforms previous state of the art.

<table><tr><td>Methods</td><td>Metrics</td><td colspan="3">Novel Views</td><td colspan="3">Training Views</td></tr><tr><td></td><td></td><td>S1</td><td>S2</td><td>Avg.</td><td>S1</td><td>S2</td><td>Avg.</td></tr><tr><td>CP-SLAM [11] PSNR [dB] â</td><td></td><td>8.96</td><td>9.17</td><td>9.06</td><td>10.01</td><td>10.45</td><td>10.23</td></tr><tr><td></td><td>SSIMâ</td><td>0.32</td><td>0.24</td><td>0.28</td><td>0.24</td><td>0.30</td><td>0.27</td></tr><tr><td></td><td>LPIPS â</td><td>0.91</td><td>0.95</td><td>0.93</td><td>0.93</td><td>0.98</td><td>0.95</td></tr><tr><td></td><td>Depth L1 [cm] â</td><td>17.14</td><td>13.23</td><td>15.18</td><td>18.42</td><td>12.01</td><td>15.12</td></tr><tr><td>MAGiC-SLAM PSNR [dB] â</td><td></td><td>23.45</td><td>21.78</td><td>22.61</td><td>24.11</td><td>26.17</td><td>25.14</td></tr><tr><td></td><td>SSIMâ</td><td>0.89</td><td>0.85</td><td>0.87</td><td>0.91</td><td>0.93</td><td>0.92</td></tr><tr><td></td><td>LPIPS â</td><td>0.22</td><td>0.21</td><td>0.22</td><td>0.20</td><td>0.14</td><td>0.17</td></tr><tr><td></td><td>Depth L1 [cm] â</td><td>1.33</td><td>4.96</td><td>3.15</td><td>1.87</td><td>1.30</td><td>1.59</td></tr></table>

Table 5. Novel and training view synthesis performance on AriaMultiagent dataset. The global map built by merging the maps from two agents is evaluated by synthesizing novel and training views. Previous state-of-the-art methods struggle to render realworld data. Note. CP-SLAM is evaluated only on two agents since it does not support more than two.

Multiagent and ReplicaMultiagent [11] datasets. This result highlights the importance of pose initialization for tracking accuracy. In addition, we show that pose initialization does not add significant computational overhead.
<table><tr><td>Methods</td><td>Dataset</td><td>Tracking Frame [s]</td><td>Avg. ATE RMSE [cm] â</td></tr><tr><td rowspan="2">MAGiC-SLAM (w.o. Pose Initialization)</td><td>AriaMultiagent</td><td>0.66</td><td>0.874</td></tr><tr><td>ReplicaMultiagent</td><td>0.68</td><td>0.825</td></tr><tr><td rowspan="2">MAGiC-SLAM</td><td>AriaMultiagent</td><td>0.67</td><td>0.670</td></tr><tr><td>ReplicaMultiagent</td><td>0.69</td><td>0.365</td></tr></table>

Table 6. Ablation on pose initialization on AriaMultiagent and ReplicaMultiagent datasets. Our method benefits from robust pose initialization without much computational overhead.

Loop Closure Detection. We evaluate the performance of our loop closure detection against the NetVLAD-based loop closure detection mechanism proposed in [11]. For this, we integrate a NetVLAD-based loop closure detection module into our pipeline keeping all the thresholds from [11]. Thanks to the huge amount of data DinoV2 [26] was trained, its features can accurately encode a large variety of data. This leads to more accurate loop closure detection as shown in Table 7.

Memory and Runtime Analysis. In Table 8, we compare runtime and memory usage against the most recent multiagent neural SLAM system [11]. Specifically, we evaluate the time required for tracking, mapping, map merging, and peak VRAM consumption. Using Gaussian sub-maps for agent mapping and tracking significantly accelerates the pipeline and limits the VRAM required per agent. Additionally, our sub-map caching and merging strategies reduce the merging time.

CP-SLAM[47]  
<!-- image-->

MAGiC-SLAM (ours)  
<!-- image-->

Ground-Truth  
<!-- image-->  
Figure 4. Rendering performance on ReplicaMultiagent [11]. Thanks to GS scene representation and effective merging strategy, MAGiC-SLAM encodes more high-frequency details and substantially increases the quality of the renderings.

<table><tr><td>Methods</td><td>Datasets</td><td>Avg.</td></tr><tr><td>MAGiC-SLAM (w. NetVLAD [1])</td><td>AriaMultiagent</td><td>1.363</td></tr><tr><td>MAGiC-SLAM (w. NetVLAD [1])</td><td>ReplicaMultiagent</td><td>0.351</td></tr><tr><td>MAGiC-SLAM</td><td>AriaMultiagent</td><td>0.900</td></tr><tr><td>MAGiC-SLAM</td><td>ReplicaMultiagent</td><td>0.252</td></tr></table>

Table 7. Ablation on loop detection mechanism on ReplicaMultiagent and AriaMultiagent datasets (ATE RMSE [cm]â). Our foundation vision model-based loop detection mechanism generalizes better than Net-VLAD descriptors to unseen data.

Limitations. As shown in Table 8, MAGiC-SLAM is significantly faster than the previous state-of-the-art in tracking, mapping, and global map reconstruction. However, the system operates slightly faster than 1 FPS. The implicit tracking mechanism is accurate but requires many iterations to converge. Future work could investigate solutions that enable faster convergence without compromising the accuracy of the current pipeline.

<table><tr><td>Methods</td><td>Mapping Frame [s]</td><td>Tracking Frame [s]</td><td>Map Merging [s]</td><td>Peak GPU [GiB]</td></tr><tr><td>CP-SLAM [11]</td><td>16.95</td><td>3.36</td><td>1448</td><td>7.70</td></tr><tr><td>MAGiC-SLAM</td><td>0.71</td><td>0.69</td><td>167</td><td>1.12</td></tr></table>

Table 8. Runtime and Memory Analysis on ReplicaMultiagent office0. Mapping, tracking, and peak GPU consumption are computed per agent. Map merging is the time required to construct the global map from the agentsâ sub-maps. All metrics are computed using an NVIDIA RTX A6000 GPU.

## 5. Conclusion

We present MAGiC-SLAM, a multi-agent SLAM with novel view synthesis capabilities. Thanks to our tracking and loop closure modules, MAGiC-SLAM demonstrates superior tracking accuracy on both synthetic and real-world datasets. Our efficient map-merging strategy allows highquality rendering in various scenarios. The Gaussian-based scene representation significantly reduces processing time, disk, and VRAM consumption compared to the previous state of the art. Finally, our method can handle a varied number of simultaneously operating agents.

Acknowledgements. This work was supported by Tom-Tom, the University of Amsterdam, and the allowance of Top Consortia for Knowledge and Innovation (TKIs) from the Netherlands Ministry of Economic Affairs and Climate Policy.

## References

[1] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pa- Â´ jdla, and Josef Sivic. Netvlad: Cnn architecture for weakly supervised place recognition, 2016. 5, 8, 2

[2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields, 2021. 2

[3] P.J. Besl and Neil D. McKay. A method for registration of 3-d shapes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2):239â256, 1992. 4, 5

[4] Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M. M. Montiel, and Juan D. Tardos. Orb-slam3: An accurate open-source library for visual, visualâinertial, and multimap slam. IEEE Transactions on Robotics, 37(6): 1874â1890, 2021. 6, 7

[5] Yun Chang, Yulun Tian, Jonathan P. How, and Luca Carlone. Kimera-multi: a system for distributed multi-robot metricsemantic simultaneous localization and mapping. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 11210â11218, 2021. 2, 3

[6] Chi-Ming Chung, Yang-Che Tseng, Ya-Ching Hsu, Xiang-Qian Shi, Yun-Hung Hua, Jia-Fong Yeh, Wen-Chin Chen, Yi-Ting Chen, and Winston H Hsu. Orbeez-slam: A realtime monocular visual slam with orb features and nerfrealized mapping. arXiv preprint arXiv:2209.13274, 2022. 2

[7] Brian Curless and Marc Levoy. Volumetric method for building complex models from range images. In SIGGRAPH Conference on Computer Graphics. ACM, 1996. 1

[8] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazare, Maria Â´ Lomeli, Lucas Hosseini, and Herve J Â´ egou. The faiss library, Â´ 2024. 4

[9] Jorge Fuentes-Pacheco, Jose Ruiz-Ascencio, and Â´ Juan Manuel Rendon-Mancha.Â´ Visual simultaneous localization and mapping: a survey. Artificial intelligence review, 43:55â81, 2015. 1, 3

[10] Dorian Galvez-L Â´ opez and J. D. Tard Â´ os. Bags of binary words Â´ for fast place recognition in image sequences. IEEE Transactions on Robotics, 28(5):1188â1197, 2012. 5

[11] Jiarui Hu, Mao Mao, Hujun Bao, Guofeng Zhang, and Zhaopeng Cui. Cp-slam: Collaborative neural point-based slam system, 2023. 2, 3, 5, 6, 7, 8, 1

[12] Jiarui Hu, Xianhao Chen, Boyin Feng, Guanglin Li, Liangjing Yang, Hujun Bao, Guofeng Zhang, and Zhaopeng Cui. Cg-slam: Efficient dense rgb-d slam in a consistent uncertainty-aware 3d gaussian field, 2024. 2, 3

[13] Huajian Huang, Longwei Li, Cheng Hui, and Sai-Kit Yeung. Photo-slam: Real-time simultaneous localization and photorealistic mapping for monocular, stereo, and rgb-d cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2

[14] Iman Abaspur Kazerouni, Luke Fitzgerald, Gerard Dooly, and Daniel Toal. A survey of state-of-the-art on visual slam. Expert Systems with Applications, 205:117734, 2022. 1, 3

[15] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. arXiv preprint, 2023. 1, 2, 3, 4

[16] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 2

[17] Rainer Kummerle, Giorgio Grisetti, Hauke Strasdat, KurtÂ¨ Konolige, and Wolfram Burgard. G2o: A general framework for graph optimization. In 2011 IEEE International Conference on Robotics and Automation, pages 3607â3613, 2011. 5, 1

[18] Pierre-Yves Lajoie and Giovanni Beltrame. Swarm-slam: Sparse decentralized collaborative simultaneous localization and mapping framework for multi-robot systems. IEEE Robotics and Automation Letters, 9(1):475â482, 2024. 2, 3, 6, 7

[19] Pierre-Yves Lajoie, Benjamin Ramtoula, Yun Chang, Luca Carlone, and Giovanni Beltrame. Door-slam: Distributed, online, and outlier resilient slam for robotic teams. IEEE Robotics and Automation Letters, 5(2):1656â1663, 2020. 3

[20] Lorenzo Liso, Erik Sandstrom, Vladimir Yugay, Luc Â¨ Van Gool, and Martin R Oswald. Loopy-slam: Dense neural slam with loop closures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20363â20373, 2024. 2, 5

[21] Mohammad Mahdi Johari, Camilla Carta, and FrancÂ¸ois Fleuret. Eslam: Efficient dense slam system based on hybrid representation of signed distance fields. arXiv e-prints, pages arXivâ2211, 2022. 2

[22] Hidenobu Matsuki, Keisuke Tateno, Michael Niemeyer, and Federic Tombari. Newton: Neural view-centric mapping for on-the-fly large-scale slam. arXiv preprint arXiv:2303.13654, 2023. 3

[23] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. Gaussian Splatting SLAM. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 1, 2, 3, 6, 7

[24] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In European Conference on Computer Vision (ECCV). CVF, 2020. 2

[25] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4):1â15, 2022. 2

[26] Maxime Oquab, Timothee Darcet, Th Â´ eo Moutakanni, Huy Â´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Je- Â´ gou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. 5, 7, 1, 2

[27] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Carl Yuheng Ren. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception, 2023. 6

[28] Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Colored point cloud registration revisited. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 143â 152, 2017. 4

[29] Zhexi Peng, Tianjia Shao, Liu Yong, Jingke Zhou, Yin Yang, Jingdong Wang, and Kun Zhou. Rtg-slam: Real-time 3d reconstruction at scale using gaussian splatting. In ACM SIGGRAPH Conference Proceedings, Denver, CO, United States, July 28 - August 1, 2024, 2024. 2

[30] Antoni Rosinol, John J. Leonard, and Luca Carlone. NeRF-SLAM: Real-Time Dense Monocular SLAM with Neural Radiance Fields. arXiv, 2022. 2

[31] Radu Bogdan Rusu, Nico Blodow, and Michael Beetz. Fast point feature histograms (fpfh) for 3d registration. In 2009 IEEE International Conference on Robotics and Automation, pages 3212â3217, 2009. 5

[32] Erik Sandstrom, Yue Li, Luc Van Gool, and Martin R Os- Â¨ wald. Point-slam: Dense neural point cloud-based slam. In International Conference on Computer Vision (ICCV). IEEE/CVF, 2023. 2, 6

[33] Patrik Schmuck and Margarita Chli. CCM-SLAM: Robust and efficient centralized collaborative monocular simultaneous localization and mapping for robotic teams. In Journal of Field Robotics (JFR), 2018. 2, 3, 6, 7

[34] Randall C. Smith and Peter Cheeseman. On the representation and estimation of spatial uncertainty. The International Journal of Robotics Research, 5(4):56â68, 1986. 2

[35] Jurgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Â¨ Burgard, and Daniel Cremers. A benchmark for the evaluation of RGB-D SLAM systems. In International Conference on Intelligent Robots and Systems (IROS). IEEE/RSJ, 2012. 6

[36] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J. Davison. iMAP: Implicit Mapping and Positioning in Real-Time. In International Conference on Computer Vision (ICCV). IEEE/CVF, 2021. 1, 2

[37] Fabio Tosi, Youmin Zhang, Ziren Gong, Erik Sandstrom, Â¨ Stefano Mattoccia, Martin R. Oswald, and Matteo Poggi. How nerfs and 3d gaussian splatting are reshaping slam: a survey, 2024. 1, 3, 6, 2

[38] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Coslam: Joint coordinate and sparse parametric encodings for neural real-time slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13293â13302, 2023. 2

[39] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004. 6

[40] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting, 2024. 2, 3

[41] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pages 499â507. IEEE, 2022. 2

[42] Alex Yu, Sara Fridovich-Keil, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks, 2021. 2

[43] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Oswald. Gaussian-slam: Photo-realistic dense slam with gaussian splatting, 2024. 2, 3, 4, 5, 6, 7, 1

[44] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 6

[45] Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo Poggi. Go-slam: Global optimization for consistent 3d instant reconstruction. arXiv preprint arXiv:2309.02436, 2023. 2

[46] Liyuan Zhu, Yue Li, Erik Sandstrom, Shengyu Huang, Kon- Â¨ rad Schindler, and Iro Armeni. Loopsplat: Loop closure by registering 3d gaussian splats, 2024. 2

[47] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12786â12796, 2022. 2, 8

# MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM

Supplementary Material

## Abstract

This supplementary material includes a video of MAGiC-SLAM running on multiple agents in a real-world indoor environment, showcasing the effectiveness of tracking, mapping, loop closure, and map merging modules. Furthermore, we provide implementation details, ablations on efficiency, pose initialization, and loop constraints estimation.

## A. Video

We provide the video magic slam.mp4 in the supplementary material. In the video, we overview the high-level architecture of our method. We also showcase MAGiC-SLAM online tracking and reconstruction capabilities on the Aria-Multiagent room0 sequence. The video highlights the effectiveness of our globally consistent reconstruction process. It visualizes how the agents explore the environment and their estimated trajectories. For clarity, the cached sub-maps are visualized as meshes. Additionally, the video demonstrates the loop edges connecting nodes in the agentsâ trajectories and shows how sub-maps are updated using the optimized poses from pose graph optimization. Finally, it presents the optimized trajectories and the resulting merged global map.

## B. Implementation Details

Hyperparameters. Tab. B.1 lists the hyperparameters used in our system, including $\lambda _ { c }$ in the tracking loss, learning rates $l _ { r }$ for rotation and $l _ { t }$ for translation, and the number of optimization iterations itert for tracking and $\mathrm { i t e r } _ { m }$ for mapping on the reported ReplicaMultiagent [11] and AriaMultiagent datasets. Additionally we set Î»color, $\lambda _ { \mathrm { d e p t h } }$ , and $\lambda _ { \mathrm { r e g } }$ to 1 in the mapping loss $\mathcal { L } _ { \mathrm { r e n d e r } }$ for all datasets.

<table><tr><td>Params</td><td>ReplicaMultiagent</td><td>AriaMultiagent</td></tr><tr><td> $\lambda _ { c }$ </td><td>0.95</td><td>0.6</td></tr><tr><td> $l _ { r }$ </td><td>0.0002</td><td>0.002</td></tr><tr><td> $l _ { t }$ </td><td>0.002</td><td>0.01</td></tr><tr><td> $\mathrm { i t e r } _ { t }$ </td><td>60</td><td>200</td></tr><tr><td> $\mathrm { i t e r } _ { m }$ </td><td>100</td><td>100</td></tr></table>

Table B.1. Per-dataset Hyperparameters for tracking and mapping modules.

Tracking. The inlier mask $M _ { \mathrm { i n l i e r } }$ in the tracking loss filters pixels with depth errors 50 times larger than the median depth error of the current re-rendered depth map. Pixels without valid depth input are also excluded as the inconsistent re-rendering in those areas can hinder the pose optimization. For the soft alpha mask, we adopt $M _ { \mathrm { { a l p h a } } } = \alpha ^ { 3 }$ for per-pixel loss weighting, where Î± refers to the Gaussian opacity value.

Mapping. A new submap is triggered every 50 frames for ReplicaMultiagent and 20 frames for AriaMultiagent. We do this to synchronize the communication between the agents and the server. When selecting candidates to add to the submap at a new keyframe, we uniformly sample $M _ { k }$ points from pixels that meet either the alpha value condition or the depth discrepancy condition. $M _ { k }$ is set to 60K for ReplicaMultiagent and 100K for AriaMultiagent. The threshold $\alpha _ { \mathrm { t h r e } }$ is set to 0.98 across all datasets. The depth discrepancy condition masks pixels where the depth error exceeds 40 times the median depth error of the current frame. Newly added Gaussians are initialized with opacity values 0.5 and their initial scales are set to the nearest neighbor distances within the submap. After the mapping optimization for the new keyframe, we prune Gaussians with opacity values lower than a threshold $O _ { \mathrm { t h r e } }$ . We set $o _ { \mathrm { t h r e } } ~ = ~ 0 . 1$ for all the datasets. Upon completing the mapping and tracking of all frames for the agentsâ input sequences, we merge the saved sub-maps into a global map. The mesh is extracted via TSDF fusion [7] using the rendered depth maps and estimated poses from the sub-maps. We perform color and depth refinement on the global map for 3K iterations using the same losses and hyperparameters as in the mapping stage.

Loop Closure. We use DinoV2 [26] based on the small ViT architecture for loop closure. We set feature threshold $\theta _ { \mathrm { f e a t u r e } }$ to 0.35 for both ReplicaMultiagent and AriaMultiagent datasets. In the pose graph, we set the information matrix $\Omega _ { i j }$ to identity for both odometry and loop closure edges. We perform pose graph optimization at the end of the run using the g2o [17] library.

## C. Efficiency Evaluation

We numerically evaluate the efficiency of our novel mapping mechanism, comparing it with Gaussian-SLAM [43], which was the first to incorporate sub-maps in a 3DGS SLAM pipeline. Unlike Gaussian-SLAM, our approach reduces the sub-map size on disk by avoiding caching all Gaussians. Additionally, we initialize sub-maps with significantly fewer seeded Gaussians. Gaussian-SLAM does not support multi-agent operations, so we evaluate its performance using single-agent sequences from the Replica-Multiagent [11] dataset.

<table><tr><td>Methods</td><td>Peak disk [MB]â Peak GPU [GiB]â</td><td></td></tr><tr><td>Gaussian-SLAM [43]</td><td>181</td><td>4.16</td></tr><tr><td>MAGiC-SLAM</td><td>54</td><td>1.12</td></tr></table>

Table C.1. Sub-map disk and GPU memory ablation on ReplicaMultiagent. Thanks to our novel mapping mechanism, MAGiC-SLAM consumes more than three times less VRAM and disk space to process and store a single sub-map. All metrics are profiled using an NVIDIA H100 GPU.

## D. Pose Initialization

When selecting a tracking mechanism for our method, we experimented with both implicit and explicit approaches. In the paper, we have shown that implicit tracking with initialization used in our method is more accurate than explicit tracking. Moreover, we found that our pose initialization mechanism is not always beneficial for explicit tracking. For this experiment, we integrated our pose initialization module into the explicit tracking pipeline. We chose an explicit tracking approach from MonoGS [23] since it was the most accurate 3DGS-SLAM pipeline by the time of the submission [37]. MonoGS does not support multiple agents, therefore we ran our experiments on a single agent from AriaMultiagent dataset.

<table><tr><td>Method</td><td>Room-0</td><td>Room-1</td></tr><tr><td>Explicit tracking w.o. PI</td><td>0.85</td><td>1.65</td></tr><tr><td>Explicit tracking w. PI</td><td>1.79</td><td>3.21</td></tr></table>

Table D.1. Pose initialization ablation on AriaMultiagent dataset (ATE RMSE [cm]â). We show that our pose initialization (PI) strategy is not always beneficial for explicit tracking. We integrate our pose initialization module to the existing pipeline [23] and evaluate its performance on single agent sequences from the AriaMultiagent dataset.

## E. Loop Detection Feature Extractors

We evaluated the inference speed of the ViT-small DinoV2 [26] feature extractor, used in our pipeline, compared to the NetVLAD [1] feature extractor. The same NetVLAD checkpoint was used in [11]. DinoV2 demonstrates better generalization due to its training data without compromising speed.

## F. Loop Constraints Estimation

We compute loop constraints by registering full-resolution point clouds from the first frames of the respective submaps. This method outperforms registering 3D Gaussian means, as the distributions of 3D Gaussian means in loop sub-maps often vary significantly between agents and are

<table><tr><td>Method</td><td>Inference Time [s]â</td></tr><tr><td>NetVlad [1]</td><td>0.045</td></tr><tr><td>DinoV2 ViT-small [26]</td><td>0.028</td></tr></table>

Table E.1. Inference Speed Comparison for Loop Detection Feature Extractors. We compare the inference time of ViT-small DinoV2 [26] and NetVLAD [1] feature extractors, highlighting DinoV2âs improved performance without compromising on speed. We use the same NetVLAD checkpoint as in [11]. The reported numbers are averaged over the ReplicaMultiagent office0. All evaluations are done on RTX3090.

more sparse. These variations lead to less accurate loop constraints, ultimately reducing the performance of pose graph optimization.
<table><tr><td>Method</td><td>ATE [cm]â</td></tr><tr><td>MAGiC-SLAM with Gaussian Means Registration</td><td>5.621</td></tr><tr><td>MAGiC-SLAM with Point Cloud Registration</td><td>0.985</td></tr></table>

Table F.1. Loop Constraints Ablation on AriaMultiagent room0. We compare the impact of registering the means of submap 3D Gaussians across multiple agents with registering input point clouds from the first frame of each sub-map. Thanks to the similarly higher resolution of the input point clouds, point cloud registration is more accurate.