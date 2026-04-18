# GRAND-SLAM: Local Optimization for Globally Consistent Large-Scale Multi-Agent Gaussian SLAM

Annika Thomas, Aneesa Sonawalla, Alex Rose, Jonathan P. How

Abstractâ3D Gaussian splatting has emerged as an expressive scene representation for RGB-D visual SLAM, but its application to large-scale, multi-agent outdoor environments remains unexplored. Multi-agent Gaussian SLAM is a promising approach to rapid exploration and reconstruction of environments, offering scalable environment representations, but existing approaches are limited to small-scale, indoor environments. To that end, we propose Gaussian Reconstruction via Multi-Agent Dense SLAM, or GRAND-SLAM, a collaborative Gaussian splatting SLAM method that integrates i) an implicit tracking module based on local optimization over submaps and ii) an approach to inter- and intra-robot loop closure integrated into a pose-graph optimization framework. Experiments show that GRAND-SLAM provides state-of-the-art tracking performance and 28% higher PSNR than existing methods on the Replica indoor dataset, as well as 91% lower multi-agent tracking error and improved rendering over existing multi-agent methods on the large-scale, outdoor Kimera-Multi dataset.

## I. INTRODUCTION

Visual Simultaneous Localization and Mapping (SLAM) is a foundational technology for a variety of applications including real-time spatial awareness for Virtual Reality and Augmented Reality (AR/VR), autonomous driving, and robot navigation. In these contexts, visual SLAM allows systems to localize themselves within an environment while creating maps that can support complex tasks like navigation, object interaction and scene understanding. SLAM has been an active area of development for the past two decades [1]â [3], and many approaches have been developed using sparse scene representations with point clouds [4]â[6], surfels [7], [8] and voxels [9]. While these methods achieve accurate tracking and real-time mapping, they struggle to render highquality textures and are limited in capabilities for novel view synthesis.

More recently, the development of neural implicit methods [10]â[13], particularly those leveraging neural radiance fields (NeRFs) [14], began to address shortcomings of sparse methods by providing photorealistic map quality. 3D Gaussian splatting (3DGS) [15] recently emerged as an efficient alternative, with comparable rendering quality to NeRFs but significantly faster rendering and training times, making it more suitable to robotics applications. Unlike NeRFs, 3D Gaussian maps can be explicitly edited or deformed, which is beneficial for tasks like map correction and adaptation in dynamic scenes.

3DGS has recently been integrated into visual SLAM using monocular [16] and RGB-D [17], [18] sensing modalities, but these approaches have not been scaled to outdoor, largescale environments. Existing visual 3DGS SLAM approaches accumulate error and map drift due to a lack of robust loop closures and global consistency mechanisms. This creates pose errors and map distortions over time, particularly when scaling the maps to large-scale or complex environments. While loop closure integration implicitly addresses the limitations of drift accumulation in large-scale environments, existing implementations [19]â[21] are limited to small-scale, indoor scenes. Extension to large-scale environments in the existing literature assumes the presence of LiDAR scans to seed the mapping process [22], [23], but sparse scans can degrade performance and expensive, bulky LiDAR scanners may not be available.

<!-- image-->  
Fig. 1: GRAND-SLAM introduces local optimization by submap for each agent, integrates inter-agent loop closure by submap and reconstructs large-scale environments via pose graph optimization.

Additionally, multi-agent SLAM is an extension of the visual SLAM problem area that holds promise for rapid exploration of novel environments while providing global consistency mechanisms. Multi-agent SLAM with traditional 3D reconstruction methods often scales poorly in large environments due to slow optimization. We address this by formulating the 3DGS SLAM problem in a way that generates compact environment representations which can be efficiently optimized in local frames, enabling deployment at scale. We combine locally optimized submaps with an explicit ICP-based registration process for inter- and intra-robot loop closures and integrate this in a distributed pose graph optimization framework, as shown in Fig. 1.

To support robust mapping in large-scale environments, we propose GRAND-SLAM, the first method to scalably integrate inter- and intra-robot loop closures in multi-agent RGB-D Gaussian Splatting SLAM. Firstly, GRAND-SLAM incrementally builds submaps of 3D Gaussians using an RGB-D input stream for seeding and optimization. Secondly, we implement inter- and intra-robot loop closure to improve agent trajectories with a coarse-to-fine optimization process. After determining potential map overlap regions using keyframebased descriptors, we register loop closures using a dense ICPbased approach. Finally, once loop closures are detected, the system adds nodes and edges to a pose graph and performs optimization to directly update the map based on the detected loop closures. Our approach demonstrates comprehensive improvements over existing RGB-D SLAM methods, delivering higher-fidelity maps, more accurate geometry, and more robust tracking in both indoor simulations and outdoor real-world settings. The contributions of our work are:

1) A tracking module with local submap-based optimization for RGB-D Gaussian Splatting SLAM, scaling more robustly to large-scale scene representations than other radiance-field-based SLAM methods;

2) Inter- and intra-robot loop closure detection using a coarse-to-fine optimization procedure, including an optimization procedure to jointly minimize photometric and geometric error followed by a fine ICP-based refinement;

3) Integration of loop closures using single- and multiagent pose graph optimization to reduce map drift over time and reduce tracking errors, as demonstrated through real-world results.

## II. RELATED WORKS

## A. Vision-Based Tracking and Reconstruction

Classical visual SLAM systems, such as PTAM [5] and MonoSLAM [4], extract sparse visual features from images and track them across frames to estimate camera motion. ORB-SLAM3 [24] is a widely used feature-based SLAM approach, associating keypoints with binary descriptors to enable fast and robust data association. However, the descriptor extraction and matching process can be computationally intensive. FastORB-SLAM [25] addresses this with a lightweight approach that omits descriptor computation for more efficient tracking.

In texture-sparse scenes, point features may be unreliable. To mitigate this, alternative geometric primitives such as lines [26] and planes [27], [28] have been used. Surfel-based systems like ElasticFusion [8] and BAD-SLAM [7] enable dense map fusion and loop closure, offering continuous surface reconstruction. Voxel-based and volumetric approaches (e.g., Voxel Hashing [29]) and implicit TSDF fusion methods [9], [30], [31] have also laid important groundwork for dense tracking and mapping. Direct methods such as DTAM [32] and real-time variational approaches [33] demonstrate the benefit of leveraging depth maps for robust camera tracking and geometry reconstruction.

## B. Dense Neural SLAM and Implicit Representations

Neural SLAM approaches extend dense mapping with photorealistic reconstructions by replacing explicit maps with implicit functions. NeRF [14] represents scenes as radiance fields parameterized by MLPs, enabling high-quality novel view synthesis. Systems like NICE-SLAM [11] and Vox-Fusion [34] adapt these ideas by learning per-voxel features, while Point-SLAM [35] attaches learned features directly to sparse geometry. Hybrid methods such as Co-SLAM [13] combine coordinate-based encodings with hash grids for efficiency.

While powerful in appearance modeling, neural SLAM pipelines often suffer from long optimization times, difficulty with real-time performance, and lack of support for rigidbody map transformations. Moreover, separating tracking from mapping can reduce drift correction capability. Loop closure techniques such as those used in Loopy-SLAM [19] and Loop-Splat [20] apply pose-graph optimization to correct trajectory errors, but performance bottlenecks remain due to the heavy reliance on learned representations.

## C. Gaussian Splatting SLAM

3D Gaussian Splatting (3DGS) [15] offers an efficient alternative to neural representations, enabling photorealistic novel view synthesis through explicit anisotropic Gaussians and differentiable rendering. Unlike NeRFs, these representations can be rendered and optimized in real time. They also natively support rigid transformations, making them particularly attractive for scene editing applications.

Recent works such as MonoGS [16] integrate 3DGS into incremental monocular SLAM. SplaTAM [18] further improves speed and scalability by leveraging silhouette masks for adaptive Gaussian selection. RGBD GS-ICP [36] replaces image-space alignment with 3D Gaussian matching, using Generalized ICP to align frames and maps while preserving rendering quality.

## D. Gaussian Splatting SLAM Methods with Loop Closure

To address long-term drift in 3DGS-based SLAM, recent methods have introduced loop closure mechanisms. GLC-SLAM [21] integrates a pretrained image-based retrieval network to associate keyframe descriptors, enabling loop detection via cosine similarity. Robust GSSLAM [37] maintains dual sets of Gaussians and computes similarity for loop detection, although practical reproducibility remains limited. MAGiC-SLAM [38] extends 3DGS SLAM to support loop closure by leveraging a pretrained vision-language model for loop candidate selection and incorporating a differentiable alignment process, enabling drift correction and robustness in novel environments.

## E. Multi-Agent SLAM

Collaborative SLAM has evolved to accommodate multiagent settings via centralized or distributed architectures. Centralized systems like CCM-SLAM [39] and CVI-SLAM [40] use a central server for map fusion and global optimization, with individual agents performing lightweight tracking. In contrast, distributed systems such as Swarm-SLAM [41] support peer-to-peer communication with robust inter-agent loop closure and increased resilience to network limitations.

<!-- image-->  
Fig. 2: GRAND-SLAM can operate independently as a single-agent system or as a multi-agent system. On the agent side, RGB-D images are fed into the tracking and mapping modules for each submap, where keyframe descriptors are saved in a database for intra-agent loop closure detection. Submaps with keyframe descriptors are shared with the server side where inter-agent loop closures are detected, and the global map is refined.

More recently, neural and hybrid representations have been applied to multi-agent systems. CP-SLAM [42] incorporates NeRF-like volumetric fusion for collaborative scene reconstruction, though it remains limited in speed and agent scalability. MAGiC-SLAM [38] builds on this direction by offering real-time multi-agent reconstruction with explicit Gaussian representations. Its centralized architecture supports multiple agents contributing partial maps and integrating them globally. Unlike NeRF-based methods, it achieves fast rendering, rigidbody map merging, and supports frequent loop closures to maintain global consistency across agents, but it is limited to small-scale indoor settings.

While CP-SLAM and MAGiC-SLAM introduce global consistency mechanisms for photorealistic SLAM in multi-agent operations, they are limited to deployment in small-scale, indoor settings. GRAND-SLAM is the first large-scale RGB-D Gaussian splatting SLAM approach that leverages inter- and intra-robot loop closure for drift reduction. It delivers higher fidelity maps and more robust tracking across small scale and large scale environments than prior work.

## III. METHOD

## A. Preliminary: Gaussian Splatting

We use 3D Gaussian splatting (3DGS) [15] as the underlying scene representation for mapping. In contrast to implicit neural field representations, 3DGS leverages an explicit, differentiable point-based formulation, enabling real-time rendering and optimization while supporting rigid-body transformations.

Firstly, 3DGS generates a dense point cloud from an RGB-D image using camera intrinsics and per-pixel depth. To initialize the map, a set of anisotropic 3D Gaussians is seeded from the point cloud. Each Gaussian is parameterized by a position $\mu \in \mathbb { R } ^ { 3 }$ , a 3D covariance matrix $\Sigma \in \mathbb { S } _ { + } ^ { 3 }$ , an opacity value $o \in \mathbb { R }$ , and an RGB color vector $c \in \mathbb { R } ^ { 3 }$ . These Gaussians are then projected onto the image plane via a differentiable rasterization pipeline, producing color and depth renderings $\hat { I } , \hat { D }$ that are compared to the ground truth images I, D for optimization.

The optimization of Gaussian parameters is performed via gradient descent using photometric and geometric supervision from observed keyframes. During optimization, the Gaussians are adjusted to minimize a combination of image reconstruction loss and sparsity-based regularization. This procedure progressively improves the fidelity and coverage of the map while preserving real-time performance.

## B. Mapping

We follow prior work [17], [19] and represent the scene as a collection of 3DGS submaps where each submap $\mathbf { P } _ { a , l } ^ { s }$ is defined for agent a as:

$$
{ \bf P } _ { a , l } ^ { s } = \left\{ G _ { i } ^ { s } ( \mu , \Sigma , o , c ) \vert i = 1 , \ldots , N \right\} ,\tag{1}
$$

where $G _ { i } ^ { s } ( \mu , \Sigma , o , c )$ represents a point with individual Gaussian mean $\mu \in \mathbb { R } ^ { 3 }$ , covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ , opacity value $o \in \mathbb { R }$ , and RGB color $c \in \mathbb { R } ^ { 3 }$

Unlike prior work, we define each submap with respect to a local frame l, where the first keyframe begins at the identity matrix, I. After pose graph optimization, submaps are transformed to the global frame g using the optimized transformation $T _ { a , l } ^ { g }$ for agent a and local submap l:

$$
\mathbf { P } _ { a , g } ^ { s } = \left\{ T _ { a , l } ^ { g } \cdot G _ { i , l } ^ { s } ( \mu , \Sigma , o , c ) \mid i = 1 , \ldots , N \right\} ,\tag{2}
$$

where $T _ { a , l } ^ { g } \in S E ( 3 )$ maps the submap from its local frame to the global coordinate frame. This transformation is initialized from odometry and refined through pose graph optimization using both sequential and loop closure constraints, detailed in Section III-E.

1) Sub-map Initialization: The first submap $P _ { l } ^ { s }$ starts with the first frame $\mathbf { I } _ { f } ^ { s }$ where the pose is defined in the local frame l at the origin and models a sequence of keyframes $f ~ \in ~ \{ 1 , \ldots , K \}$ . Rather than processing the entire global submap simultaneously, we follow [17], [19] and initialize new submaps $P ^ { 2 } , \dots , P ^ { M }$ after the current keyframe pose exceeds a translation threshold $d _ { \mathrm { m a x } }$ or rotation threshold $\theta _ { \mathrm { m a x } }$ with respect to the first frame of the submap. We redefine the pose at the first keyframe of each submap $\mathbf { I } _ { f } ^ { s }$ as the origin to ensure globally consistent optimization. At any time, only the current active submap which is bounded to a limited size is processed in order to reduce the computational cost of exploring larger scenes.

2) Sub-map Building: For subsequent keyframes, 3D Gaussians are added to newly observed or sparse parts of the active submap. Once the following keyframe pose $\mathbf { I } _ { k } ^ { s }$ has been tracked, defining a pose transformation from $\mathbf { \bar { \Phi } } T _ { k } ^ { k - 1 } ,$ we use a dense point cloud from the keyframe RGB-D measurement to seed new Gaussians. We sample points uniformly in sparse regions of the map as determined by the rendered opacity of the active map where rendered Î± values are below a threshold $\alpha _ { \mathrm { { m i n } } }$ . These Gaussians are then optimized using photometric and geometric supervision with gradient descent.

3) Sub-map Optimization: All Gaussian points in the active submap are optimized every time new Gaussians are added based on a fixed number of iterations. We optimize the submap over the rendering loss of depth and color of each keyframe f in the submap after filtering by a soft alpha mask $M _ { \alpha }$ , a polynomial of the alpha map that suppresses poorly observed or sparsely reconstructed regions and an error inlier mask $M _ { \mathrm { i n } } ,$ a binary mask that discards pixels where either the color or depth error exceeds a frame-relative threshold:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { m a p } } = \displaystyle \sum { M _ { \mathrm { i n } } M _ { \alpha } \cdot \left( \lambda _ { c } \| \hat { I } _ { j } ^ { s } - I _ { j } ^ { s } \| _ { \mathcal { L } 1 } \right. } } \\ & { \qquad \quad \left. + ( 1 - \lambda _ { c } ) \| \hat { D } _ { j } ^ { s } - D _ { j } ^ { s } \| _ { \mathcal { L } 1 } \right) } ,  \end{array}\tag{3}
$$

where $\lambda _ { c }$ weights the color and depth losses, and $\| \cdot \| _ { \mathcal { L } 1 }$ denotes the $\mathcal { L } _ { 1 }$ loss between two images.

## C. Tracking

We perform tracking in two stages: an initial frame-to-frame pose estimation followed by a frame-to-model refinement using rendering-based optimization. To initialize the current camera pose $T _ { i }$ , we use a coarse relative pose estimate obtained with a relative transformation $T _ { i - 1 , i }$ found by minimizing a hybrid color-depth odometry objective to provide a fast, coarse initialization for $T _ { i }$ :

$$
T _ { i } = T _ { i - 1 } \cdot T _ { i - 1 , i } ,\tag{4}
$$

where $T _ { i - 1 , i }$ is the relative transformation estimated by visual odometry. In the refinement stage, we separately optimize the rotation and translation components of $T _ { i }$ in a local frame. That is, we express updates relative to the local coordinate system centered at the current pose rather than in the global world frame. This is important because optimizing rotations and translations with respect to the world frame origin that the agentâs current pose may be far away from, can result in imbalanced gradients and degraded convergence. We show a qualitative example of degraded rendering performance when optimizing rotation with respect to the world origin in Fig. 3.

<!-- image-->  
Fig. 3: Without local optimization, optimizing rotation results in large movements with respect to the origin which may be far from the agentâs current position. This example demonstrates the renders of a scene in the Kimera-Multi Outdoor dataset [43] optimization (left) and after rotating with respect to the origin (right).

We formulate the tracking loss ${ \mathcal { L } } _ { \mathrm { t r a c k i n g } }$ as the photometric and geometric difference between the input frame and the model rendering under the current pose:

$$
\arg \operatorname* { m i n } _ { R , \mathbf { t } } \mathcal { L } _ { \mathrm { t r a c k } } \left( \hat { I } ( R , \mathbf { t } ) , \hat { D } ( R , \mathbf { t } ) , I , D , \alpha \right) ,\tag{5}
$$

where R and t denote the rotation and translation to be optimized, $\hat { I } ( R , { \bf t } )$ and $\hat { D } ( R , { \bf t } )$ are the rendered color and depth maps produced by transforming the reconstructed model with rotation R and translation t, and I and D are the input color and depth maps at frame i. The tracking loss follows the rendering loss from Eq. 3 optimized over rotation R and translation t. During optimization, all 3D Gaussian parameters of the scene are kept frozen. Only the pose parameters (rotation R and translation t) are updated to minimize the tracking loss.

This two-stage tracking strategy of coarse frame-to-frame hybrid color-depth odometry optimization followed by local frame-to-model optimization enables both fast initialization and accurate alignment even in large, complex environments.

## D. Loop Closure Detection

To ensure global consistency across agent submaps, we incorporate both intra-agent and inter-agent loop closure detection based on keyframe similarity and geometric alignment. Each keyframe stores a NetVLAD descriptor [44] computed at capture time associated with its respective submap. These descriptors are stored in a local submap database, and loop closure candidates are detected by querying the current keyframe against this database. For intra-agent loop closures, each agent queries its own submap database, while inter-agent loop closures query databases of other agents when available.

Given a query keyframe descriptor $\mathbf { v } _ { q } ,$ we retrieve the topk nearest neighbors $\{ \mathbf { v } _ { j } \} _ { j = 1 } ^ { k }$ from the relevant database using cosine similarity and accept a match if the similarity exceeds a threshold $\tau _ { \mathrm { s i m } }$ and the spatial distance between the associated keyframe poses exceeds a minimum baseline âmin.

For candidate matches, we estimate an initial transformation between submaps to initialize refinement. In the case of intraagent loop closures, we use the previously tracked camera poses to compute an initial transform:

$$
T _ { \mathrm { i n i t } } = T _ { j } ^ { - 1 } T _ { q } ,\tag{6}
$$

where $T _ { q }$ and $T _ { j }$ are the estimated poses of the query and match keyframes in the global frame.

For inter-agent loop closures or long-range intra-agent matches where drift may have accumulated, we compute $T _ { \mathrm { i n i t } }$ using a coarse geometric alignment method. Specifically, we estimate a relative transformation between the query and match frames using dense RGB-D registration with a hybrid depthcolor loss:

$$
T _ { \mathrm { i n i t } } = \arg \operatorname* { m i n } _ { T } \mathcal { L } _ { \mathrm { h y b r i d } } ( T ) ,\tag{7}
$$

where $\mathcal { L } _ { \mathrm { h y b r i d } }$ jointly minimizes photometric and geometric error between consecutive RGB-D frames using a multi-scale registration pyramid and an iterative solver.

Given an initial transformation $T _ { \mathrm { i n i t } } .$ we refine the loop closure alignment via point-to-plane ICP on dense colored point clouds of each submap:

$$
\mathcal { L } _ { \mathrm { I C P } } = \sum _ { i } \left( \left( \mathbf { n } _ { i } ^ { \top } ( \mathbf { p } _ { i } ^ { T } - \mathbf { q } _ { i } ) \right) ^ { 2 } \right) ,\tag{8}
$$

where $\mathbf { p } _ { i }$ is the source point, $\mathbf { q } _ { i }$ is the target point, and $\mathbf { n } _ { i }$ is the surface normal at the target point. Optimization proceeds until convergence using the following a distance threshold $d _ { \operatorname* { m a x } } ,$ maximum number of iterations $N _ { \mathrm { i t e r } }$ and relative fitness and RMSE thresholds. This produces the final transformation $T _ { \mathrm { l o o p } } ^ { * } ,$ as well as alignment metrics including fitness score $f$ and inlier RMSE $\rho .$

Candidate loop closures are filtered using alignment quality metrics. We accept a loop closure if $f > \tau _ { f }$ and $\rho < \tau _ { \rho }$ where $\tau _ { f }$ and $\tau _ { \rho }$ are thresholds on fitness and inlier RMSE, respectively. Accepted loop closures are incorporated into a pose graph as additional constraints between submap origins. Let $T _ { a , l } ^ { g }$ and $T _ { b , l } ^ { g }$ be the local-to-global transformations of submaps $P _ { a , l } ^ { s }$ and $P _ { b , l } ^ { s }$ , then the loop closure constraint is added as:

$$
T _ { a , l } ^ { g } T _ { a b } = T _ { b , l } ^ { g } ,\tag{9}
$$

where $T _ { a b } ~ = ~ T _ { \mathrm { l o o p } } ^ { * }$ is the estimated transformation from submap b to a. All constraints are subsequently optimized jointly via pose graph optimization.

## E. Pose Graph Optimization

To jointly refine the trajectories of all agents, we construct a global pose graph where nodes correspond to the submap keyframe poses $\{ T _ { a , f } ^ { g } \}$ of agent a in the global frame g, and edges represent relative pose constraints from both tracking and loop closure detections.

We assign the first keyframe of the first agent as the fixed origin of the global coordinate frame, denoted as $\begin{array} { r } { T _ { 0 . 0 } ^ { g } = \mathbf { I } _ { 4 \times 4 } . } \end{array}$ Each agentâs trajectory is initialized using the output of the local tracking process described in Section III-C. Intraagent loop closures provide relative pose constraints between keyframes within the same agent:

$$
\mathcal { C } _ { \mathrm { i n t r a } } = \left\{ ( i , j , \hat { T } _ { i , j } , \Sigma _ { i , j } ) \right\} ,\tag{10}
$$

where $\hat { T } _ { i , j }$ is the relative transformation estimated by registration and $\Sigma _ { i , j }$ is the corresponding information matrix.

Inter-agent loop closures transform one agentâs trajectory into the frame of another, yielding cross-agent constraints:

$$
\mathcal { C } _ { \mathrm { i n t e r } } = \left\{ ( i , j , \hat { T } _ { i , j } ^ { a , b } , \Sigma _ { i , j } ^ { a , b } ) \right\} ,\tag{11}
$$

with $a \neq b$ and $T _ { i , j } ^ { a , b }$ representing the relative transform from keyframe i of agent a to keyframe $j$ of agent b.

The full graph $\mathcal { G }$ is then defined over the union of agent poses and loop closure constraints:

$$
\mathcal { G } = ( \mathcal { V } , \mathcal { E } ) = \left( \left\{ T _ { i } \right\} , \mathcal { C } _ { \mathrm { t r a c k i n g } } \cup \mathcal { C } _ { \mathrm { i n t r a } } \cup \mathcal { C } _ { \mathrm { i n t e r } } \right) .\tag{12}
$$

We solve for globally consistent poses $\{ T _ { i } ^ { * } \}$ by minimizing the total error across all relative pose constraints using nonlinear least squares:

$$
\mathcal { L } _ { \mathrm { g r a p h } } = \sum _ { ( i , j ) \in \mathcal { E } } \left. \log \left( \hat { T } _ { i , j } ^ { - 1 } T _ { i } ^ { - 1 } T _ { j } \right) \right. _ { \Sigma _ { i , j } } ^ { 2 } ,\tag{13}
$$

where $\log ( \cdot )$ denotes the $\mathfrak { s e } ( 3 )$ logarithm map, and the norm is weighted by the information matrix $\Sigma _ { i , j }$ . We implement the multi-agent pose graph optimization in GTSAM [45] and use Levenberg-Marquardt optimization, though we note that any nonlinear optimization framework may be used here.

Once the globally consistent submap poses $T _ { a , l } ^ { g }$ are obtained from pose graph optimization, each keyframe pose $T _ { j } ^ { ( l ) }$ and all associated 3D Gaussian parameters within submap $\mathbf { P } _ { a , l } ^ { s }$ are transformed into the global frame. The updated global keyframe pose becomes:

$$
T _ { j } ^ { ( g ) } \gets T _ { a , l } ^ { g } \cdot T _ { j } ^ { ( l ) } ,\tag{14}
$$

where $T _ { j } ^ { ( l ) }$ is the keyframeâs original pose in the local submap frame l, and $T _ { j } ^ { ( g ) }$ is its pose in the global frame. Similarly, each Gaussian $G _ { i , l } ^ { s } ( \mu , \Sigma , o , c )$ in the submap is transformed as:

$$
\boldsymbol { \mu } ^ { ( g ) } = \boldsymbol { R } \boldsymbol { \mu } ^ { ( l ) } + t , \quad \boldsymbol { \Sigma } ^ { ( g ) } = \boldsymbol { R } \boldsymbol { \Sigma } ^ { ( l ) } \boldsymbol { R } ^ { \top } ,\tag{15}
$$

where $T _ { a , l } ^ { g } ~ = ~ [ R ~ | ~ t ]$ is the corresponding transformation applied to the Gaussian means and covariances. Color values remain fixed, as we do not recompute spherical harmonics during this update. This produces globally aligned submaps $\mathbf { P } _ { a , g } ^ { s }$ suitable for joint rendering and evaluation.

## IV. EVALUATION

## A. Experimental Setup

1) Datasets: To evaluate our method against existing multiagent dense SLAM approaches, we conduct experiments on two complementary datasets that reflect both idealized and real-world conditions. We first benchmark on the Multiagent Replica dataset [42], [46], which includes four synthetic indoor sequences featuring two collaborating RGB-D agents. Each sequence offers clean depth and color data without noise, blur, or reflective artifacts. This setting allows for controlled comparison of trajectory accuracy and reconstruction fidelity under ideal conditions.

To test scalability and robustness in real-world scenarios, we also evaluate on subsets of the Kimera-Multi Outdoor dataset [43], a large-scale outdoor RGB-D dataset involving six agents with varying trajectory overlap. We evaluate on a combined

TABLE I: Single-Agent tracking performance on Multiagent Replica (ATE RMSE â [cm]), where colors denote first second and third best performance.
<table><tr><td>Method</td><td>Off0</td><td>Apt0</td><td>Apt1</td><td>Apt2</td><td>Avg</td></tr><tr><td>ORB-SLAM3 [24]</td><td>0.60</td><td>1.07</td><td>4.94</td><td>1.36</td><td>1.99</td></tr><tr><td>Gaussian-SLAM [17]</td><td>0.33</td><td>0.41</td><td>30.13</td><td>121.96</td><td>38.21</td></tr><tr><td>MonoGS [16]</td><td>0.38</td><td>0.21</td><td>3.33</td><td>0.54</td><td>1.15</td></tr><tr><td>MAGiC-SLAM [38]</td><td>0.42</td><td>0.38</td><td>0.54</td><td>0.66</td><td>0.50</td></tr><tr><td>GRAND-SLAM (w/0 LC)</td><td>0.44</td><td>0.28</td><td>0.46</td><td>0.27</td><td>0.36</td></tr><tr><td>GRAND-SLAM</td><td>0.32</td><td>0.23</td><td>0.35</td><td>0.17</td><td>0.27</td></tr></table>

1.85 km of sequences individually ranging from 239.25 m to 409.74 m that present significant challenges such as depth sensor noise, reflections, and motion blur. In our evaluation, Agents 1, 2, and 3 refer to the ACL Jackal, Hathor and Thoth traverses for the Outside 1 team and the Sparkal1, Sparkal2 and ACL Jackal2 traverses for the Outside 2 team respectively. Together, these two datasets allow us to assess performance across both clean and noisy multi-agent environments.

2) Implementation Details: We evaluate GRAND-SLAM on a machine with two NVIDIA GeForce RTX 3090 GPUs, each with 24 GB VRAM. During evaluation of multi-agent settings, we run GRAND-SLAM on each agent individually then apply the server module with inter-agent loop closures and multi-agent pose graph optimization separately to evaluate both single-agent and multi-agent performance.

3) Baseline Methods: We evaluate our systemâs tracking and mapping performance against both classical and learningbased SLAM baselines. For single-agent tracking, GRAND-SLAM is compared against representative RGB-D and stereo visual SLAM methods, including ORB-SLAM3 [24] and Point-SLAM [35], as well as recent Gaussian splattingbased approaches: MonoGS [16], Gaussian-SLAM [17], and

TABLE II: Multi-Agent Tracking performance on Multiagent Replica (ATE RMSE â [cm]), where colors denote first second and third best performance.
<table><tr><td>Method</td><td>Agent</td><td>0-0</td><td>A-0</td><td>A-1</td><td>A-2</td><td>Avg</td></tr><tr><td>CCM-SLAM [39]</td><td>Agt 1</td><td>9.84</td><td>X</td><td>2.12</td><td>0.51</td><td>-</td></tr><tr><td>Swarm-SLAM [41]</td><td></td><td>1.07</td><td>1.61</td><td>4.62</td><td>2.69</td><td>2.50</td></tr><tr><td>CP-SLAM [42]</td><td></td><td>0.50</td><td>0.62</td><td>1.11</td><td>1.41</td><td>0.91</td></tr><tr><td>MAGiC-SLAM (w/o LC)</td><td></td><td>0.44</td><td>0.30</td><td>0.48</td><td>0.91</td><td>0.53</td></tr><tr><td>MAGiC-SLAM [38]</td><td></td><td>0.31</td><td>0.13</td><td>0.21</td><td>0.42</td><td>0.27</td></tr><tr><td>GRAND-SLAM (w/o LC)</td><td></td><td>0.27</td><td>0.27</td><td>0.47</td><td>0.33</td><td>0.34</td></tr><tr><td>GRAND-SLAM</td><td></td><td>0.28</td><td>0.27</td><td>0.28</td><td>0.18</td><td>0.25</td></tr><tr><td>CCM-SLAM [39]</td><td>Agt 2</td><td>0.76</td><td>X</td><td>9.31</td><td>0.48</td><td>-</td></tr><tr><td>Swarm-SLAM [41]</td><td></td><td>1.76</td><td>1.98</td><td>6.50</td><td>8.53</td><td>4.69</td></tr><tr><td>CP-SLAM [42]</td><td></td><td>0.79</td><td>1.28</td><td>1.72</td><td>2.41</td><td>1.55</td></tr><tr><td>MAGiC-SLAM (w/o LC)</td><td></td><td>0.41</td><td>0.46</td><td>0.61</td><td>0.41</td><td>0.47</td></tr><tr><td>MAGiC-SLAM [38]</td><td></td><td>0.24</td><td>0.21</td><td>0.30</td><td>0.22</td><td>0.24</td></tr><tr><td>GRAND-SLAM (w/o LC)</td><td></td><td>0.43</td><td>0.22</td><td>0.44</td><td>0.20</td><td>0.32</td></tr><tr><td>GRAND-SLAM</td><td></td><td>0.25</td><td>0.19</td><td>0.36</td><td>0.18</td><td>0.25</td></tr><tr><td>CCM-SLAM [39]</td><td>Avg</td><td>5.30</td><td>X</td><td>5.71</td><td>0.49</td><td>-</td></tr><tr><td>Swarm-SLAM [41]</td><td></td><td>1.42</td><td>1.80</td><td>5.56</td><td>5.61</td><td>3.60</td></tr><tr><td>CP-SLAM [42]</td><td></td><td>0.65</td><td>0.95</td><td>1.42</td><td>1.91</td><td>1.23</td></tr><tr><td>MAGiC-SLAM (w/o LC)</td><td></td><td>0.42</td><td>0.38</td><td>0.54</td><td>0.66</td><td>0.50</td></tr><tr><td>MAGiC-SLAM [38]</td><td></td><td>0.28</td><td>0.17</td><td>0.26</td><td>0.32</td><td>0.26</td></tr><tr><td>GRAND-SLAM (w/o LC)</td><td></td><td>0.44</td><td>0.28</td><td>0.46</td><td>0.27</td><td>0.36</td></tr><tr><td>GRAND-SLAM</td><td></td><td>0.27</td><td>0.23</td><td>0.32</td><td>0.18</td><td>0.25</td></tr></table>

TABLE III: Tracking Performance on Kimera-Multi Outdoor (ATE RMSE â [m]), where colors denote first and second best performance, and partial results from failed runs are denoted with an asterisk\* and averages including partial runs are shown in gray.
<table><tr><td>Method</td><td>Agent</td><td>Outside 1</td><td>Outside 2</td><td>Avg</td></tr><tr><td>ORB-SLAM3 [24]</td><td>Agt 1</td><td>14.11</td><td>2.72</td><td>8.42</td></tr><tr><td>Gaussian SLAM [12]</td><td></td><td>356.61</td><td>71.16</td><td>213.89</td></tr><tr><td>MAGiC-SLAM [38]</td><td></td><td>24.13</td><td>11.13</td><td>17.63</td></tr><tr><td>GRAND-SLAM (w/o LC)</td><td></td><td>6.43</td><td>10.19</td><td>8.31</td></tr><tr><td>GRAND-SLAM</td><td></td><td>3.95</td><td>10.93</td><td>7.44</td></tr><tr><td>ORB-SLAM3 [24]</td><td>Agt 2</td><td>7.07</td><td>13.12</td><td>10.10</td></tr><tr><td>Gaussian SLAM [12]</td><td></td><td>119.68</td><td>7.66*</td><td>63.67</td></tr><tr><td>MAGiC-SLAM [38]</td><td></td><td>98.33</td><td>10.50*</td><td>54.42</td></tr><tr><td>GRAND-SLAM (w/o LC)</td><td></td><td>9.74</td><td>4.63</td><td>7.19</td></tr><tr><td>GRAND-SLAM</td><td></td><td>8.93</td><td>4.54</td><td>6.74</td></tr><tr><td>ORB-SLAM3 [24]</td><td>Agt 3</td><td>7.48</td><td>18.99</td><td>13.24</td></tr><tr><td>Gaussian SLAM [12]</td><td></td><td>1150.42</td><td>195.95</td><td>673.19</td></tr><tr><td>MAGiC-SLAM [38]</td><td></td><td>172.81</td><td>47.86</td><td>110.34</td></tr><tr><td>GRAND-SLAM (w/o LC)</td><td></td><td>1.05</td><td>7.93</td><td>4.49</td></tr><tr><td>GRAND-SLAM</td><td></td><td>1.30</td><td>6.25</td><td>3.78</td></tr><tr><td>ORB-SLAM3 [24]</td><td>Avg</td><td>9.55</td><td></td><td></td></tr><tr><td>Gaussian SLAM [12]</td><td></td><td></td><td>11.61</td><td>10.58</td></tr><tr><td>MAGiC-SLAM [38]</td><td></td><td>542.24</td><td>91.59 23.16</td><td>316.92</td></tr><tr><td>GRAND-SLAM (w/o LC)</td><td></td><td>98.42 5.74</td><td>7.58</td><td>60.79 6.66</td></tr><tr><td>GRAND-SLAM</td><td></td><td>4.73</td><td>7.24</td><td>4.99</td></tr></table>

MAGiC-SLAM [38].

In the collaborative setting, we benchmark tracking performance against multi-agent SLAM systems such as SWARM-SLAM [41], CCM-SLAM [39], CP-SLAM [42] and MAGiC-SLAM [38].

To assess the quality of dense map reconstruction, we compare rendering fidelity against MAGiC-SLAM [38], Gaussian-SLAM [17] and CP-SLAM [42], which use dense scene representations for single- and multi-agent mapping.

Since GRAND-SLAM supports both single- and multiagent operation, with or without loop closures, we report results under each setting to allow for fair comparisons across the full range of baseline systems.

## B. Camera Tracking

Table I compares GRAND-SLAM to state of the art singleagent visual SLAM approaches. In these cases, we report the average of the two agentsâ performances for each dataset. In the case of MAGiC-SLAM, Table I includes the single agent results. GRAND-SLAM achieves superior tracking accuracy than other methods without loop closure and with intra-agent loop closure, we are consistent with or outperform baseline approaches in all cases.

We also compare against multi-agent approaches on the Multiagent Replica dataset in Table II. GRAND-SLAM is consistent with or marginally outperforms MAGiC-SLAM, while outperforming all other baseline methods. This demonstrates that GRAND-SLAM is capable of running at high performance in small-scale indoor settings while also generalizing to large-scale challenging real-world datasets.

On the Kimera-Multi dataset, we evaluate against visual SLAM approaches to demonstrate the current limitations of the state of the art large-scale outdoor settings using realworld datasets, shown in Table III. We note that both MAGiC-SLAM and Gaussian SLAM failed partway through the Agent 2 traverse of the Outside 2 Team, so partial results are included. MAGiC-SLAM also fails to consistently find correct loop closures in these settings, so single-agent results are reported. GRAND-SLAM outperforms all baseline methods on the Kimera-Multi dataset with and without loop closures.

Gaussian SLAM  
MAGiC SLAM  
GRAND SLAM (Ours)  
<!-- image-->  
Ground Truth  
Fig. 4: Renders from the Kimera-Multi dataset demonstrate GRAND-SLAMâs superior performance compared to baseline methods in large-scale scenes where it maintains photorealistic rendering results.

TABLE IV: Training view synthesis performance on Kimera-Multi dataset, where best performance is shown in bold, partial results from failed runs are denoted with an asterisk\* and averages including partial runs are shown in gray.
<table><tr><td>Methods</td><td>Metrics</td><td>Outside 1</td><td>Outside 2</td><td>Avg</td></tr><tr><td rowspan="4">Gaussian SLAM</td><td>PSNR â</td><td>24.59</td><td>24.31*</td><td>24.45</td></tr><tr><td>SSIM â</td><td>0.90</td><td>0.89*</td><td>0.90</td></tr><tr><td>LPIPS â</td><td>0.18</td><td>0.17*</td><td>0.18</td></tr><tr><td>Depth L1 â</td><td>1.19</td><td>1.45*</td><td>1.32</td></tr><tr><td rowspan="4">MAGiC SLAM</td><td>PSNR â</td><td>16.12</td><td>15.63*</td><td>15.88</td></tr><tr><td>SSIM â</td><td>0.49</td><td>0.50*</td><td>0.50</td></tr><tr><td>LPIPS â</td><td>0.54</td><td>0.53*</td><td>0.54</td></tr><tr><td>Depth L1 â</td><td>3.86</td><td>4.71*</td><td>4.29</td></tr><tr><td rowspan="4">GRAND SLAM</td><td>PSNR â</td><td>28.48</td><td>26.62</td><td>27.44</td></tr><tr><td>SSIM â</td><td>0.97</td><td>0.96</td><td>0.97</td></tr><tr><td>LPIPS â</td><td>0.10</td><td>0.12</td><td>0.11</td></tr><tr><td>Depth L1 â</td><td>1.17</td><td>1.61</td><td>1.39</td></tr></table>

## C. Rendering

We evaluate the rendering performance on the Multiagent Replica dataset of submaps compared to MAGiC-SLAM and CP-SLAM, two multiagent neural SLAM approaches, in Table V. GRAND-SLAM significantly outperforms baseline approaches on training view synthesis in the small-scale indoor setting. We also evaluate the rendering performance on the Kimera-Multi outdoor dataset of Gaussian SLAM and MAGiC

TABLE V: Training view synthesis performance on Multiagent Replica dataset, where best performance is shown in bold.
<table><tr><td>Methods</td><td>Metrics</td><td>0-0</td><td>A-0</td><td>A-1</td><td>A-2</td><td>Avg</td></tr><tr><td></td><td>PSNR â</td><td>28.56</td><td>26.12</td><td>12.16</td><td>23.98</td><td>22.71</td></tr><tr><td>CP</td><td>SSIM â</td><td>0.87</td><td>0.79</td><td>0.31</td><td>0.81</td><td>0.69</td></tr><tr><td>SLAM</td><td>LPIPS â</td><td>0.29</td><td>0.41</td><td>0.97</td><td>0.39</td><td>0.51</td></tr><tr><td></td><td>Depth L1 â</td><td>2.74</td><td>19.93</td><td>66.77</td><td>2.47</td><td>22.98</td></tr><tr><td></td><td>PSNR â</td><td>39.32</td><td>36.96</td><td>30.01</td><td>30.73</td><td>34.26</td></tr><tr><td>MAGiC</td><td>SSIM â</td><td>0.99</td><td>0.98</td><td>0.95</td><td>0.96</td><td>0.97</td></tr><tr><td>SLAM</td><td>LPIPS â</td><td>0.05</td><td>0.09</td><td>0.18</td><td>0.17</td><td>0.12</td></tr><tr><td></td><td>Depth L1 â</td><td>0.41</td><td>0.64</td><td>3.16</td><td>0.99</td><td>1.30</td></tr><tr><td></td><td>PSNR â</td><td>43.12</td><td>44.15</td><td>38.65</td><td>39.46</td><td>41.35</td></tr><tr><td>GRAND</td><td>SSIM â</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td></tr><tr><td>SLAM</td><td>LPIPS â</td><td>0.03</td><td>0.03</td><td>0.05</td><td>0.05</td><td>0.04</td></tr><tr><td></td><td>Depth L1 â</td><td>0.25</td><td>0.31</td><td>0.77</td><td>0.29</td><td>0.41</td></tr></table>

SLAM in Table IV, where GRAND-SLAM outperforms all baseline methods as well. We note that MAGiC-SLAM and Gaussian-SLAM both failed partway through one traverse, so partial results prior to the failure are reported. GRAND-SLAM has superior rendering performance than baseline methods in both small-scale indoor and large-scale outdoor settings.

## V. CONCLUSION

We present GRAND-SLAM, a multi-agent dense SLAM approach using 3DGS as the underlying scene representation, designed for large-scale outdoor environments. In future work, we plan to integrate compression techniques to better manage communication and memory constraints for larger scenes. GRAND-SLAM outperforms state of the art neural and 3DGS SLAM methods in rendering and tracking performance, showing promise as a method using local optimization to scale effectively for large-scale photorealistic SLAM.

[1] C. Cadena, L. Carlone, H. Carrillo, Y. Latif, D. Scaramuzza, J. Neira, I. Reid, and J. J. Leonard, âPast, present, and future of simultaneous localization and mapping: Toward the robust-perception age,â IEEE Transactions on robotics, vol. 32, no. 6, pp. 1309â1332, 2016.

[2] J. Fuentes-Pacheco, J. Ruiz-Ascencio, and J. M. Rendon-Mancha, Â´ âVisual simultaneous localization and mapping: a survey,â Artificial intelligence review, vol. 43, pp. 55â81, 2015.

[3] I. A. Kazerouni, L. Fitzgerald, G. Dooly, and D. Toal, âA survey of state-of-the-art on visual slam,â Expert Systems with Applications, vol. 205, p. 117734, 2022.

[4] A. J. Davison, I. D. Reid, N. D. Molton, and O. Stasse, âMonoslam: Real-time single camera slam,â IEEE transactions on pattern analysis and machine intelligence, vol. 29, no. 6, pp. 1052â1067, 2007.

[5] G. Klein and D. Murray, âParallel tracking and mapping for small ar workspaces,â in 2007 6th IEEE and ACM international symposium on mixed and augmented reality. IEEE, 2007, pp. 225â234.

[6] R. Mur-Artal and J. D. Tardos, âOrb-SLAM2: An open-source SLAMÂ´ system for monocular, stereo, and RGB-D cameras,â IEEE transactions on robotics, vol. 33, no. 5, pp. 1255â1262, 2017.

[7] T. Schops, T. Sattler, and M. Pollefeys, âBad SLAM: Bundle adjusted direct RGB-D SLAM,â in Proceedings of the IEEE/CVF CVPR, 2019, pp. 134â144.

[8] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison, âElasticfusion: Dense SLAM without a pose graph.â in Robotics: science and systems, vol. 11. Rome, Italy, 2015, p. 3.

[9] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohi, J. Shotton, S. Hodges, and A. Fitzgibbon, âKinectfusion: Real-time dense surface mapping and tracking,â in 2011 10th IEEE international symposium on mixed and augmented reality. Ieee, 2011, pp. 127â136.

[10] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, âImap: Implicit mapping and positioning in real-time,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 6229â6238.

[11] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-SLAM: Neural implicit scalable encoding for SLAM,â in Proceedings of the IEEE/CVF CVPR, 2022, pp. 12 786â 12 796.

[12] M. M. Johari, C. Carta, and F. Fleuret, âESLAM: Efficient dense SLAM system based on hybrid representation of signed distance fields,â in Proceedings of the IEEE/CVF CVPR, 2023, pp. 17 408â17 419.

[13] H. Wang, J. Wang, and L. Agapito, âCo-SLAM: Joint coordinate and sparse parametric encodings for neural real-time SLAM,â in Proceedings of the IEEE/CVF CVPR, 2023, pp. 13 293â13 302.

[14] B. Mildenhall and P. P. e. a. Srinivasan, âNERF: representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[15] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d Gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[16] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting SLAM,â in Proceedings of the IEEE/CVF CVPR, 2024, pp. 18 039â 18 048.

[17] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, âGaussian-SLAM: Photo-realistic dense SLAM with Gaussian splatting,â arXiv preprint arXiv:2312.10070, 2023.

[18] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplaTAM: splat track & map 3D Gaussians for dense RGB-D slam,â in IEEE CVPR, 2024, pp. 21 357â21 366.

[19] L. Liso, E. Sandstrom, V. Yugay, L. Van Gool, and M. R. Oswald, Â¨ âLoopy-SLAM: Dense neural SLAM with loop closures,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 363â20 373.

[20] L. Zhu, Y. Li, E. Sandstrom, S. Huang, K. Schindler, and I. Armeni, Â¨ âLoopsplat: Loop closure by registering 3D Gaussian splats,â arXiv preprint arXiv:2408.10154, 2024.

[21] Z. Xu, Q. Li, C. Chen, X. Liu, and J. Niu, âGlc-SLAM: Gaussian splatting SLAM with efficient loop closure,â arXiv preprint arXiv:2409.10982, 2024.

[22] K. Wu, K. Zhang, Z. Zhang, M. Tie, S. Yuan, J. Zhao, Z. Gan, and W. Ding, âHGS-mapping: Online dense mapping using hybrid Gaussian representation in urban scenes,â IEEE RA-L, 2024.

[23] S. Hong, J. He, X. Zheng, C. Zheng, and S. Shen, âLIV-GaussMap: LiDAR-inertial-visual fusion for real-time 3D radiance field map rendering,â IEEE Robotics and Automation Letters, 2024.

[24] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, âOrb-SLAM3: An accurate open-source library for visual, Â´ visualâinertial, and multimap SLAM,â IEEE transactions on robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[25] Q. Fu, H. Yu, X. Wang, Z. Yang, Y. He, H. Zhang, and A. Mian, âFast orb-slam without keypoint descriptors,â IEEE transactions on image processing, vol. 31, pp. 1433â1446, 2021.

[26] R. Gomez-Ojeda, F.-A. Moreno, D. Zuniga-Noel, D. Scaramuzza, and Â¨ J. Gonzalez-Jimenez, âPL-SLAM: a stereo SLAM system through the combination of points and line segments,â IEEE Transactions on Robotics, vol. 35, no. 3, pp. 734â746, 2019.

[27] Y. Li, R. Yunus, N. Brasch, N. Navab, and F. Tombari, âRGB-D SLAM with structural regularities,â in 2021 IEEE international conference on Robotics and automation (ICRA). IEEE, 2021, pp. 11 581â11 587.

[28] B. Gong and Z. e. a. Zhu, âPlanefusion: Real-time indoor scene reconstruction with planar prior,â IEEE Transactions on Visualization and Computer Graphics, vol. 28, no. 12, pp. 4671â4684, 2021.

[29] M. NieÃner, M. Zollhofer, S. Izadi, and M. Stamminger, âReal-time Â¨ 3D reconstruction at scale using voxel hashing,â ACM Transactions on Graphics (ToG), vol. 32, no. 6, pp. 1â11, 2013.

[30] B. Curless and M. Levoy, âA volumetric method for building complex models from range images,â in Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, 1996, pp. 303â 312.

[31] A. Dai, M. NieÃner, M. Zollhofer, S. Izadi, and C. Theobalt, âBundle- Â¨ fusion: Real-time globally consistent 3D reconstruction using on-the-fly surface reintegration,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, p. 1, 2017.

[32] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, âDtam: Dense tracking and mapping in real-time,â in 2011 ICCV. IEEE, 2011, pp. 2320â2327.

[33] J. Stuhmer, S. Gumhold, and D. Cremers, âReal-time dense geometry Â¨ from a handheld camera,â in Joint Pattern Recognition Symposium. Springer, 2010, pp. 11â20.

[34] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, âVoxfusion: Dense tracking and mapping with voxel-based neural implicit representation,â in 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2022, pp. 499â507.

[35] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald, âPoint-SLAM: Â¨ Dense neural point cloud-based SLAM,â in IEEE ICCV, 2023, pp. 18 433â18 444.

[36] S. Ha, J. Yeon, and H. Yu, âRGBD GS-ICP SLAM,â arXiv preprint arXiv:2403.12550, 2024.

[37] Z. Zhu, Y. Fang, X. Li, C. Yan, F. Xu, C. Yuen, and Y. Li, âRobust Gaussian splatting SLAM by leveraging loop closure,â arXiv preprint arXiv:2409.20111, 2024.

[38] V. Yugay, T. Gevers, and M. R. Oswald, âMagic-slam: Multi-agent gaussian globally consistent slam,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 6741â6750.

[39] P. Schmuck and M. Chli, âCCM-SLAM: Robust and efficient centralized collaborative monocular simultaneous localization and mapping for robotic teams,â Journal of Field Robotics, vol. 36, no. 4, pp. 763â781, 2019.

[40] M. Karrer, P. Schmuck, and M. Chli, âCVI-SLAMâcollaborative visualinertial SLAM,â IEEE Robotics and Automation Letters, vol. 3, no. 4, pp. 2762â2769, 2018.

[41] P.-Y. Lajoie and G. Beltrame, âSwarm-SLAM: Sparse decentralized collaborative simultaneous localization and mapping framework for multi-robot systems,â IEEE Robotics and Automation Letters, vol. 9, no. 1, pp. 475â482, 2023.

[42] J. Hu, M. Mao, H. Bao, G. Zhang, and Z. Cui, âCP-SLAM: Collaborative neural point-based SLAM system,â Advances in Neural Information Processing Systems, vol. 36, pp. 39 429â39 442, 2023.

[43] Y. Tian, Y. Chang, L. Quang, A. Schang, C. Nieto-Granda, J. P. How, and L. Carlone, âResilient and distributed multi-robot visual SLAM: Datasets, experiments, and lessons learned,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 11 027â11 034.

[44] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and J. Sivic, âNetvlad: Cnn architecture for weakly supervised place recognition,â in IEEE CVPR, 2016, pp. 5297â5307.

[45] F. Dellaert, âFactor graphs and GTSAM: a hands-on introduction,â Georgia Institute of Technology, Tech. Rep, vol. 2, no. 4, 2012.

[46] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.