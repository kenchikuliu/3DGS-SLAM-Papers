# IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation

Wenxuan Guo1â Xiuwei Xu1â Hang Yin1 Ziwei Wang2

Jianjiang Feng1â  Jie Zhou1 Jiwen Lu1

1Tsinghua University 2Nanyang Technological University

{gwx22, xxw21, yinh23}@mails.tsinghua.edu.cn ziwei.wang@ntu.edu.sg {jfeng, jzhou, lujiwen}@tsinghua.edu.cn

## Abstract

Visual navigation with an image as goal is a fundamental and challenging problem. Conventional methods either rely on end-to-end RL learning or modular-based policy with topological graph or BEV map as memory, which cannot fully model the geometric relationship between the explored 3D environment and the goal image. In order to efficiently and accurately localize the goal image in 3D space, we build our navigation system upon the renderable 3D gaussian (3DGS) representation. However, due to the computational intensity of 3DGS optimization and the large search space of 6-DoF camera pose, directly leveraging 3DGS for image localization during agent exploration process is prohibitively inefficient. To this end, we propose IGL-Nav, an Incremental 3D Gaussian Localization framework for efficient and 3D-aware image-goal navigation. Specifically, we incrementally update the scene representation as new images arrive with feed-forward monocular prediction. Then we coarsely localize the goal by leveraging the geometric information for discrete space matching, which can be equivalent to efficient 3D convolution. When the agent is close to the goal, we finally solve the fine target pose with optimization via differentiable rendering. The proposed IGL-Nav outperforms existing state-of-the-art methods by a large margin across diverse experimental configurations. It can also handle the more challenging free-view imagegoal setting and be deployed on real-world robotic platform using a cellphone to capture goal image at arbitrary pose. Project page: https://gwxuan.github.io/IGL-Nav/.

## 1. Introduction

Image-goal navigation, which requires an agent initialized in unknown environment to navigate to the location and orientation specified by an image [39], is a fundamental problem in a wide range of robotic tasks. This task requires the agent to precisely understand spatial information, as well as to reason how to explore the scene with past observations, which is hard to learn with end-to-end RL [32] due to low sample efficiency and catastrophic forgetting.

<!-- image-->  
Figure 1. IGL-Nav effectively guides the agent to reach free-view image goal via incremental 3D gaussian localization.

Recent advances in visual navigation have witnessed significant progress in modular-based approaches [2, 3, 7, 14, 23, 34, 35], which establish an explicit memory to cache observed environmental information and derive navigation policies based on the memory representations. While these approaches demonstrate enhanced capabilities for longhorizon reasoning and temporal dependency modeling in object-goal navigation tasks, their extension to image-goal navigation remains challenging. Unlike object-goal scenarios that primarily rely on high-level semantic understanding, image-goal navigation necessitates the preservation and processing of low-level visual features, including fine-grained texture patterns and color distributions. Consequently, conventional representation paradigms of memory such as topological graphs prove insufficient for effectively encoding the requisite environmental information in image-goal settings. To address these limitations, RNR-Map [14] introduces a renderable neural radiance map representation. Drawing inspiration from NeRF [20], this representation enables photorealistic image rendering from arbitrary camera viewpoints. The renderable nature ensures the preservation of crucial low-level visual features, which has demonstrated superior performance in image-goal navigation. However, since NeRF is an implicit field with high computational cost, RNR-Map has to maintain the renderable representation in a 2D BEV map for efficient and explicit memory management. This 2D projection inherently loses critical 3D structural information, forcing RNR-Map to impose strict constraints on goal image acquisition, specifically requiring horizontal camera angles to ensure alignment with its BEV map. This significantly reduces its applicability in real-world scenarios. Therefore, an efficient 3D-aware memory representation is still desirable for image-goal navigation.

In this paper, we propose to leverage 3D Gaussian Splatting (3DGS) [10] as the scene representation for imagegoal navigation. The 3DGS representation demonstrates exceptional suitability for the task: (1) as an explicit representation, 3DGS can be easily initialized with the observed RGB-D image and be incrementally accumulated in 3D space; (2) it supports efficient differentiable rendering, which can be used to localize the camera pose of goal image with iterative optimization. Despite these compelling properties, adapting 3DGS representations for image-goal navigation presents significant challenges. While 3DGS achieves rendering speeds orders of magnitude faster than NeRF, their optimization process remains computationally prohibitive for real-time online inference required in navigation tasks. Furthermore, goal image localization within scene-level 3DGS maps becomes intractable due to the exponential search space complexity inherent in 6-DoF camera pose estimation. To this end, we propose IGL-Nav, an Incremental 3D Gaussian Localization framework that (1) progressively constructs 3DGS through feed-forward prediction, eliminating offline optimization; and (2) enables efficient hierarchical goal search by harnessing both geometric and photometric attributes of 3DGS through our novel coarse-to-fine localization strategy. Extensive experiments on various datasets in Habitat simulator show our IGL-Nav significantly outperforms previous state-of-the-art imagegoal navigation methods. Moreover, benefit from our explicit 3D representation, IGL-Nav is also able to handle the more practical free-view image-goal setting, where there is no assumption on both camera intrinsics and extrinsics of the goal image. We further deploy our method on real-world robot, where a casually taken photo from a cellphone can be used as goal to guide the agent navigating to specified location in complicated and large-scale environments.

## 2. Related Work

3D Gaussian Splatting. 3DGS [10] has emerged as a powerful technique for 3D scene representation. It represents a scene as a dense set of points with gaussian embedding and leverages efficient rasterization techniques for high-fidelity, real-time rendering. Recently, feed-forward 3DGS models [4â6] have been proposed, primarily to address the issue of sparse-view scene reconstruction. Unlike traditional methods that iteratively optimize 3DGS parameters, feed-forward 3DGS predicts the gaussian distribution through a network, significantly improving modeling efficiency. In embodied AI, 3DGS has been applied to manipulation tasks, with dynamic 3DGS frameworks and gaussian world models used to model and predict robotic actions [19]. And systems like Gaussian-Grasper [38] leverage RGB-D inputs for language-guided grasping. 3DGS also helps bridge the gap between simulated and real-world environments for generalizing learned behaviors. Techniques such as Robo-GS [18] and SplatSim [22] improve Sim-to-Real transfer by leveraging efficient representation of 3DGS. The incremental 3DGS scene representation used in our IGL-Nav also follows the paradigm of feed-forward 3DGS, making it suitable for scene modeling based on online inputs in navigation tasks.

Image-goal Navigation. Image-goal navigation involves the agent navigating to the location where the goal image is captured [39], requiring precise alignment in both position and orientation. To address this challenge, researchers have employed various strategies. Some focus on optimizing reinforcement learning (RL) policies [1, 32, 33] that directly map observations to actions. Others concentrate on constructing detailed maps [3, 11, 14, 24], or on developing carefully crafted matching algorithms [29]. However, image-goal navigation requires that the query image be captured by the agentâs camera, which limits the cameraâs intrinsic parameters, height, and the fact that it can only rotate around the Z-axis. Considering these constraints in practical applications, we propose free-view image-goal navigation, where the target image can be captured by any camera at free 3D position and orientation. In addition, instance image-goal navigation [12] is a similar task, where the target image focuses on specific categories of objects within the scene. In instance image-goal task, GaussNav [16] also uses a 3DGS-based scene representation. However, Gauss-Nav requires first completing the exploration of the entire building to optimize the 3DGS representation, and then render images at multi poses for comparison with the target image. This approach limits efficiency in practical applications. In contrast, our IGL-Nav simultaneously performs exploration, incremental modeling, and target localization, and it incorporates a coarse-to-fine localization strategy, making full use of the 3DGS representation.

## 3. Approach

In this section, we first describe our task definition. Next, we explain several core modules of IGL-Nav, including in-

cremental scene representation with 3DGS, and coarse-tofine target localization. Finally, we detail the overall navigation pipeline.

## 3.1. Problem Statement

We study the problem of free-view image-goal navigation, which is a more challenging and practical setting. In this task, a mobile agent is instructed with navigating to a specified location depicted by an image $\scriptstyle { \cal I } _ { g } ,$ , taken by camera A with pose $\mathbf { T } _ { g }$ . The agent is equipped with camera B. It receives posed RGB-D video stream $\{ I _ { t } , D _ { t } , \mathbf { T } _ { t } \} _ { t = 1 } ^ { T }$ and is required to execute an action $a \in { \mathcal { A } }$ at each time it receiving a new RGB-D observation. ${ \cal I } _ { t } , { \cal D } _ { t } , { \bf T } _ { t }$ refer to RGB image, depth image and camera pose at time instant t. A is the set of actions, which consists of move forward, turn left, turn right and stop. The task is considered successfully completed if the agent terminates within a horizontal neighborhood of the target pose, satisfying $| | \mathcal { P } ( \mathbf { T } _ { f i n a l } ) - \mathcal { P } ( \mathbf { T } _ { g } ) | | _ { 2 } < \epsilon$ within a maximum of T navigation steps. Here $\mathcal { P }$ refers to 3D-to-BEV projection.

Comparison with Relevant Tasks. In the free-view image-goal navigation, there is no assumption on the correlation between camera A and B. For example, in real application scenarios, A can be a cellphone, and B is a RGB-D camera with totally different camera intrinsics. Previous image-goal setting [7, 14] can be regarded as a special case of our task where $A \equiv B$ and $\mathbf { T } _ { g }$ is restricted to lie within camera Bâs achievable pose space. Instance-imagegoal navigation [13] also aims to decouple camera A and B. However, this setting requires that there must be an instance located at image center, and only 6 categories of instances are supported. These limitations fundamentally constrain the systemâs operational flexibility and real-world deployment potential. In this paper, we conduct experiments on both conventional and free-view image-goal settings for a comprehensive evaluation of different approaches.

## 3.2. Incremental Scene Representation

We adopt 3DGS as our scene representation due to its explicit nature and efficient rendering capability. However, the original 3DGS are obtained through offline optimization on image set and thus hard to be applied in real-time tasks. Recent feed-forward methods [4â6] abandon optimization and directly predict pixel-aligned 3DGS parameters, but they still rely on multi-view images to reconstruct geometric information of the scene. In visual navigation, the agent needs to incrementally build scene representation along with its exploration, so the 3DGS should be generated in real-time and update as new images arrive. To accommodate streaming video input while effectively leveraging camera pose and depth priors, we present the first feedforward 3DGS reconstruction model for monocular RGB-D sequences, which supports real-time 3DGS reconstruction and incremental accumulation.

Gaussian Parameters Prediction. At time step t, the agent receives new RGB-D observations $\pmb { I } _ { t } \in \mathbb { R } ^ { \hat { H } \times \hat { W } \times 3 }$ and $\ b { D } _ { t } ~ \in ~ \mathbb { R } ^ { H \times W \times 1 }$ Our incremental reconstruction model is essentially a mapping $f _ { \theta }$ from observations to 3DGS parameters, including position $\pmb { \mu } _ { k }$ , opacity $\alpha _ { k }$ , covariance $\Sigma _ { k }$ and spherical harmonics $\mathbf { } _ { c k } \mathbf { : }$

$$
f _ { \pmb \theta } : ( \pmb { I } _ { t } , \pmb { D } _ { t } ) \mapsto \{ ( \pmb { \mu } _ { k } , \alpha _ { k } , \pmb { \Sigma } _ { k } , \pmb { c } _ { k } ) \} _ { k = 1 } ^ { H \times W }\tag{1}
$$

The 3DGS parameters are predicted in a pixel-aligned manner, thus an observation input of size $H \times W$ corresponds to an output of $H \times W$ gaussians.

The feed-forward model $f _ { \theta }$ is shown in Figure 2. We first concatenate the normalized RGB and depth images, and then extract dense monocular scene embedding $\mathbf { } E _ { t } ^ { \prime }$ with a UNet-based encoder E. Then 3DGS parameters are regressed through a gaussian head ${ \mathcal { H } } ,$ composed of a few CNN and linear layers. This process can be expressed as:

$$
\Delta C _ { 2 D } , \Delta D , \alpha , \Sigma , c = \mathcal { H } ( E ^ { \prime } ) , \quad E ^ { \prime } = \mathcal { E } ( I , D )\tag{2}
$$

where $\Delta C _ { 2 D }$ and $\Delta \pmb { D }$ are residuals of image coordinates and depth. We omit subscript t for simplicity. Using the camera intrinsic matrix M, pose $\mathbf { T } _ { t }$ and inverse projection $\mathrm { P r o j } ^ { - 1 }$ , we can compute the 3DGS positions as:

$$
\mu = \mathrm { P r o j } ^ { - 1 } ( C _ { \mathrm { 2 D } } + \Delta C _ { \mathrm { 2 D } } , D + \Delta D \mid \mathbf { M } , \mathbf { T } _ { \mathrm { t } } )\tag{3}
$$

We also lift $E ^ { \prime }$ from 2D to the corresponding 3D positions. Finally, the 3DGS scene representation G and the corresponding 3D embedding E can be updated as: $G _ { t } =$ $G _ { t - 1 } \cup ( \pmb { \mu } _ { t } , \alpha _ { t } , \pmb { \Sigma } _ { t } , \pmb { c } _ { t } )$ and $E _ { t } = E _ { t - 1 } \cup E _ { t } ^ { \prime }$ . When the number of 3DGS in the scene is large, we prune $G _ { t }$ and $\scriptstyle { E _ { t } }$ based on opacity and 3DGS density to reduce memory footprint. Additionally, we can use $\mathcal { E }$ to extract the 3D embedding $E _ { g }$ of the target image $\scriptstyle { \cal I } _ { g } .$ . If depth and camera intrinsics are unavailable for $\scriptstyle { \cal I } _ { g } .$ , we simply use a monocular depth estimator [21] to predict them.

Training and Loss. Our feed-forward model can be trained using passive offline RGB-D video streams. We randomly sample training episodes from navigation training set. In each episode, K frames are randomly selected to predict 3DGS parameters, and images from other viewpoints are rendered for loss computation. The training loss is a linear combination of L-2 and LPIPS [37] losses.

## 3.3. Coarse-to-fine Localization

Since the target image is captured by an arbitrary camera at any pose (6-DoF), the search space of the target is extremely large. To perform efficient and accurate visual navigation, we design a coarse-to-fine target localization strategy. Coarse localization leverages the incremental scene embedding $\scriptstyle { E _ { t } }$ to predict the approximate target location in real-time during exploration. Once the agent is close to the target, fine localization is employed to accurately determine the accurate target position and guide the agent to reach it.

<!-- image-->  
Figure 2. Illustration of IGL-Nav. (a) We maintain an incremental 3DGS scene representation with feed-forward prediction. (b) The coarse target localization is modeled as a 5-dimension matching problem, which is efficiently implemented by leveraging the target embedding as 3D convolutional kernel. (c) Fine target localization via differentiable 3DGS rendering and matching-constrained optimization.

<!-- image-->

<!-- image-->  
(b) Sphere-based Pose Space.  
Figure 3. Modeling of the camera pose space. (a) Line LR is almost always parallel to the ground. (b) Line $A O ^ { \prime }$ is parallel to Plane $X O Y$ . Plane $A O ^ { \prime } B$ is perpendicular to Plane XOY .

## 3.3.1. Coarse Target Localization

Although camera A can capture the target image at an arbitrary pose, we observe that the top frame of the camera is almost always parallel to the ground when taking a photo, as shown in Figure 3. Therefore, we can represent the actual camera rotation with $( \theta , \phi )$ , which denotes a rotation around the X-axis by Î¸ degrees, followed by a rotation towards the Z-axis by Ï degrees. Based on this observation, we define a sphere-based space $\mathcal { S } : \{ ( x , y , z , \theta , \phi ) \}$ to represent camera pose. Here $( x , y , z )$ represents the position of camera A and $( \theta , \phi )$ refers to Aâs rotation. We can thus represent the target pose $\mathbf { T } _ { g }$ as $( x _ { g } , y _ { g } , z _ { g } , \theta _ { g } , \phi _ { g } )$ . The 3D embedding of the target $E _ { g }$ is initialized at the origin of the sphere-based space S and should be aligned with the scene embedding $\scriptstyle { E _ { t } }$ under translation $( x _ { g } , y _ { g } , z _ { g } )$ and rotation $( \theta _ { g } , \phi _ { g } )$ , which are unknown.

To efficiently search the target camera pose in the fivedimensional space, we discretize S to reduce the search space. For $( x , y , z )$ , the 3D space is voxelized into grids $\{ ( x _ { i } , y _ { i } , z _ { i } ) \mid x _ { i } = \textstyle \left\lfloor { \frac { x } { v } } \right\rfloor , y _ { i } = \lfloor { \frac { y } { v } } \rfloor , z _ { i } = \lfloor { \frac { z } { v } } \rfloor \}$ , where v is voxel size. For $( \theta , \phi )$ , we discretize the spherical surface into N vertices of a hierarchical mesh via Î³-level subdivision of a regular icosahedron, as shown in Figure 2. In this way, we can rotate $E _ { g }$ according to the discretized sphere to obtain N 3D embeddings $\{ E _ { g } ^ { 1 } , . . . , E _ { g } ^ { N } \}$ . By translating these embeddings to the discretized voxel grids and computing the extent of alignment between the translated embedding and $\scriptstyle { \mathbf { } } E _ { t }$ , the coarse target pose can be determined by:

$$
\operatorname* { m a x i m i z e } _ { i , k } \ A ( E _ { t } , { \mathcal { T } } ( E _ { g } ^ { k } , ( x _ { i } , y _ { i } , z _ { i } ) ) )\tag{4}
$$

where A computes the extent of alignment between two sets of 3D features. T stands for translation operation. i and k are used to query the corresponding $( x , y , z , \theta , \phi )$

However, the above operation is still hard to achieve realtime inference. We need to traverse all voxel grids and compare the translated 3D embedding with $\scriptstyle { E _ { t } }$ . Assume there are V grids at all, then $V \times N$ times comparisons should be performed. Moreover, during each comparison, we should compute the geometric similarity between two 3D pointclouds as well as their feature similarity, which is especially time-consuming and hard to be accelerated on GPUs. To solve this problem, we propose to further discretize the 3D embeddings $\scriptstyle { E _ { t } }$ and $E _ { g }$ . For $\scriptstyle { E _ { t } }$ , we can simply quantize the pointclouds into voxels, where voxel features are obtained by taking average of the pointcloud features inside each voxel. For $\{ \boldsymbol { E } _ { g } ^ { 1 ^ { - } } , . . . , \boldsymbol { E } _ { g } ^ { N } \}$ , we uniformly quantize them into $L \times L \times L$ voxels. Note that although $\{ E _ { g } ^ { 1 } , . . . , E _ { g } ^ { N } \}$ are different pointclouds, they will share the same shape after voxelization, which forms a 3D convolutional kernel $K \in \mathbb { R } ^ { L \times L \times L \times C _ { i n } \times C _ { o u t } }$ . Here $C _ { i n }$ refers to the output channel of $\mathcal { E } , C _ { o u t }$ equals to the number of kernels N . Therefore, Eq (4) can be rewritten as:

<!-- image-->  
Figure 4. Navigation pipeline of IGL-Nav.

$$
\underset { x , y , z , k } { \mathrm { a r g m a x } } \mathcal { C } ( f _ { 1 } ( \mathcal { V } ( E _ { t } ) ) , f _ { 2 } ( K ) ) [ x ] [ y ] [ z ] [ k ]\tag{5}
$$

where C means 3D convolution operation, V quantizes scene embedding $\scriptstyle { E _ { t } }$ into $X \times Y \times Z$ voxels. We use two MLP $f _ { 1 } \ / \ f _ { 2 }$ with input channel $C _ { i n }$ and output channel $C ^ { \prime }$ to project scene embedding and convolutional kernel to a learnable feature space before convolution, which further aligns the embedding space of $\scriptstyle { E _ { t } }$ and $E _ { g } .$ . The activation map after 3D convolution is of shape $\bar { X ^ { \mathrm { ~ } } } \times \bar { Y } \times Z \times N$ from which we query index of the maximum value and thus obtain a coarse localization of the target pose. To further improve computational efficiency, we use pillar-based voxelization [15, 36].

Training and Loss. Similar to the scene representation training, we train the coarse localization module using offline passive video streams. In each training segment, we randomly select a position and capture target images with arbitrary intrinsic parameters and orientations. We use focal loss [17] to supervise the activation map after 3D convolution. Additionally, we apply cross-entropy loss to supervise the outputs nearby target pose in the activation map.

## 3.3.2. Fine Target Localization

Our fine localization method aims to accurately determine the targetâs 6-DoF pose once the agent is close to the target region. It leverages the differentiable rendering ability of 3DGS to reach target pose via iterative optimization.

Rendering-based Stopper. First, we use a renderingbased stopper to determine if the agent is close to the target. Since the intrinsics of camera A and B may differ significantly, directly comparing the current observation with the target image with feature matching is difficult. Thanks to the real-time rendering capability of 3DGS $G _ { t } ,$ we can render an image at camera Bâs current viewpoint with the same intrinsic parameters as camera A. We use a local feature matching method, LoFTR [27], to predict matching pairs $( \pmb { x } _ { g } , \pmb { x } _ { t } )$ between the target image and the rendered image. Here $( \pmb { x } _ { g } , \pmb { x } _ { t } )$ is the coordinate set of matched pixels. If the number of matching pairs exceeds a threshold $\tau ,$ it is considered that $I _ { g }$ appears in agentâs field of view.

Matching-constrained Optimization. Via differentiable rendering, we can optimize the current camera pose with photometric loss between the rendered image and $I _ { g } ,$ and between rendered depth and $D _ { g } \left( D _ { g } / \mathbf { M } _ { g } \right.$ are the depth / intrinsics of $I _ { g } ,$ which are estimated by [21] if not available), as done in [9, 28]. Although this is an intuitive way to solve $\mathbf { T } _ { g } ,$ , we empirically find it leads to unsatisfactory performance in our case where the quality of $G _ { t }$ may degrade due to incremental accumulation without optimization. Fortunately, we observe the pixels that are successfully matched are of high quality. In order to overcome the imperfect details in the rendering results, we propose to only focus on the matching pairs in 3D space for accurate camera pose optimization. The problem can be formulated as:

$$
\hat { \mathbf { T } } = \underset { \mathbf { T } \in S E ( 3 ) } { \mathrm { a r g m i n } } ~ \mathcal { L } ( \mathbf { T } \mid I _ { g } , D _ { g } , \mathbf { M } _ { g } , G _ { t } )\tag{6}
$$

We iteratively optimize the pose T to minimize the geometric discrepancy between rendering results and target image. At each iteration of optimization, we leverage current T for rendering and obtain the matched points in Euclidean space:

$$
( \boldsymbol { x } _ { g } , \boldsymbol { x } ) , ( d _ { g } , d ) = \mathcal { M } ( I _ { g } , D _ { g } , \mathcal { R } ( G _ { t } \mid \mathbf { M } _ { g } , \mathbf { T } ) )\tag{7}
$$

$$
( X _ { g } , X ) = \mathrm { P r o j } ^ { - 1 } ( ( x _ { g } , x ) , ( d _ { g } , d ) \mid \mathbf { M } _ { g } )\tag{8}
$$

where R is differentiable rendering of color and depth. The matching and querying operation M first adopts LoFTR to get matching pair $( { \pmb x } _ { g } , { \pmb x } )$ between $I _ { g }$ and the rendered RGB image, and then queries the corresponding depth value $( d _ { g } , d )$ from $D _ { g }$ and the rendered depth respectively. Then we formulate the optimization loss as:

$$
\mathcal { L } = \frac { 1 } { Q } \sum _ { i = 0 } ^ { Q - 1 } ( | X _ { g } ^ { i } - X ^ { i } | _ { 2 } )\tag{9}
$$

where $Q$ is the number of matching pairs. Note that LoFTR predicts $( { \pmb x } _ { g } , { \pmb x } )$ in a differentiable way, so gradient can be backpropagated through both the rendered color and depth images. In this way, we effectively align T and $\mathbf { T } _ { g }$ by focusing on the most confident rendering results.

## 3.4. Navigation

We divide the navigation process into two stages: exploration based on coarse localization and target reaching based on fine localization. Figure 4 illustrates the workflow of IGL-Nav. We will describe each stage in this section.

Exploration for Target Discovery. When the agent is initialized in a new environment, its observations of the scene are insufficient. Therefore, we combine coarse target localization with frontier-based exploration to explore the scene and discover potential targets. Based on the posed RGB-D inputs, we maintain an online occupancy map to indicate explored, unexplored and occupied area in BEV, where the frontiers of explored area can be computed. At each time step, we select the nearest frontier to the agent and generate binary scores $S _ { f }$ on the BEV map, where points on the selected frontier are set to 1, others are set to 0. We then project the activation map obtained in our coarse target localization module to BEV to get $S _ { a }$ . We first filter the activation map $S _ { a }$ by a threshold $\sigma _ { a } ,$ , setting scores below the threshold to zero. The agent then prioritizes exploring the location with the highest value in $S _ { a }$ . If all values in $S _ { a }$ are zero, the agent selects the nearest location where the frontier score map $S _ { f }$ equals 1. We adopt Fast Marching Method [26] (FMM) for path planning and action generation given the to-be-explored location.

Table 1. Image-goal Navigation Results. SR: Success Rate, SPL: Success weighted by Path Length. The best result in each column is bold, and the second best is underlined.
<table><tr><td rowspan="3">Method</td><td colspan="8">Straight</td><td colspan="8">Curved</td></tr><tr><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="2">Hard</td><td colspan="2">Overall</td><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="2">Hard</td><td colspan="2">Overall</td></tr><tr><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td></tr><tr><td>DDPPO [30]</td><td>43.2</td><td>38.5</td><td>36.4</td><td>34.8</td><td>7.4</td><td>7.2</td><td>29.0</td><td>26.8</td><td>22.2</td><td>16.5</td><td>20.7</td><td>18.5</td><td>4.2</td><td>3.7</td><td>15.7</td><td>12.9</td></tr><tr><td>NNS [7]</td><td>64.1</td><td>55.4</td><td>47.9</td><td>39.5</td><td>25.2</td><td>18.1</td><td>45.7</td><td>37.7</td><td>27.3</td><td>10.6</td><td>23.1</td><td>10.4</td><td>10.5</td><td>5.6</td><td>20.3</td><td>8.8</td></tr><tr><td>ZL [1]</td><td>-</td><td></td><td>-</td><td>-</td><td>-</td><td></td><td>-</td><td>-</td><td>41.0</td><td>28.2</td><td>27.3</td><td>13.9</td><td>18.6</td><td>9.3</td><td>25.9</td><td>17.6</td></tr><tr><td>OVRLL [3]</td><td>53.6</td><td>34.7</td><td>48.6</td><td>33.3</td><td>32.5</td><td>21.9</td><td>44.9</td><td>30.0</td><td>53.6</td><td>31.8</td><td>47.6</td><td>30.2</td><td>35.6</td><td>22.0</td><td>45.6</td><td>28.0</td></tr><tr><td>NRNS + SLING [29]</td><td>85.3</td><td>74.4</td><td>66.8</td><td>49.3</td><td>41.1</td><td>28.8</td><td>64.4</td><td>50.8</td><td>58.6</td><td>16.1</td><td>47.6</td><td>16.8</td><td>24.9</td><td>10.1</td><td>43.7</td><td>14.3</td></tr><tr><td>OVRL + SLING [29]</td><td>71.2</td><td>54.1</td><td>60.3</td><td>44.4</td><td>43.0</td><td>29.1</td><td>58.2</td><td>42.5</td><td>68.4</td><td>47.0</td><td>57.7</td><td>39.8</td><td>40.2</td><td>25.5</td><td>55.4</td><td>37.4</td></tr><tr><td>RNR-Map [14]</td><td>76.4</td><td>55.3</td><td>73.6</td><td>46.1</td><td>54.6</td><td>30.2</td><td>68.2</td><td>43.9</td><td>75.3</td><td>52.5</td><td>70.9</td><td>42.3</td><td>51.0</td><td>27.4</td><td>65.7</td><td>40.8</td></tr><tr><td>FeudalNav [8]</td><td>82.6</td><td>75.0</td><td>71.0</td><td>57.4</td><td>49.0</td><td>34.2</td><td>67.5</td><td>5.5</td><td>72.5</td><td>51.3</td><td>64.4</td><td>40.7</td><td>43.7</td><td>25.3</td><td>60.2</td><td>39.1</td></tr><tr><td>IGL-Nav (Ours)</td><td>87.9</td><td>82.5</td><td>80.8</td><td>69.0</td><td>61.7</td><td>40.9</td><td>76.8</td><td>64.1</td><td>82.8</td><td>77.7</td><td>80.7</td><td>70.0</td><td>57.0</td><td>39.6</td><td>73.5</td><td>62.4</td></tr></table>

Reaching Target. During exploration, the agent gradually approaches the target. We use the rendering-based stopper to determine if the target appears in agentâs field of view. Once the target is detected, we switch to fine localization to compute the precise target pose. The XY coordinates of the computed pose is set to be destination, for which we apply FMM again for navigation.

## 4. Experiment

In this section, we first describe our experimental setting. Then we compare IGL-Nav with state-of-the-art image-goal navigation methods. Finally we conduct in-depth modulebased analysis on our framework and further provide realworld deployment results.

## 4.1. Experimental Setup

We conduct experiments on image-goal navigation and the more challenging free-view image-goal navigation tasks.

Datasets and Benchmarks. For image-goal navigation, we follow the public Gibson [31] image-goal navigation dataset within the Habitat simulator [25] introduced by NRNS [7]. The Gibson dataset includes 72 houses for training and 14 for validation. The NRNS dataset contains two path types (straight and curved), each with three difficulty levels (easy, medium, hard). For free-view image-goal navigation as introduced in Sec. 3.1, we collect a large amount of data with Gibson for validation. Given the significant impact of the cameraâs field of view (FOV) on scene matching, we categorize our dataset into two FOV-based groups $( 5 0 ^ { \circ } \sim 7 5 ^ { \circ }$ and $7 5 ^ { \circ } \sim 1 0 0 ^ { \circ } )$ , which can be intuitively understood as portrait and landscape orientations. Each category further includes three difficulty levels based on distance. Additionally, compared to the NRNS dataset, our free-view image-goal navigation dataset features target images captured from arbitrary angles and heights. Each of the six subsets contains 500 randomly sampled episodes.

Compared Methods. We compare IGL-Nav with existing state-of-the-art image-goal navigation methods [1, 7, 8, 14, 29, 30, 33]. For image-goal setting, we report results from the respective papers. For the proposed free-view image-goal setting, we evaluate open-sourced methods on this benchmark and compare with them. Since some methods [7, 29, 30, 33] only release test code, we perform zeroshot transfer to apply them to the new setting without retraining. We also report the zero-shot performance of IGL-Nav for fair comparison. For methods [7, 30] that provide training scripts, we train them on the free-view image-goal navigation data for comparison.

## 4.2. Comparison with State-of-the-art

We compare with state-of-the-art image-goal navigation methods on the two benchmarks described above. Table 1 demonstrates the results on image-goal navigation task. IGL-Nav establishes new state-of-the-art performance and outperforms previous methods by a large margin on all metrics, which validates the effectiveness of 3D gaussian representation and the proposed coarse-to-fine target localization strategy for image-goal navigation.

The results on free-view image-goal navigation task is shown in Table 2. As this task is much more challenging than conventional image-goal setting, we observe a significant performance drop on each metric. When directly transferred from image-goal to free-view image-goal setting, IGL-Nav still maintains a huge performance lead compared with other state-of-the-art methods. The performance of IGL-Nav can be further boosted with training data on the free-view image-goal task. Note that the zero-shot transferring performance of IGL-Nav is even better than other methods under supervised setting, which demonstrates the great generalization ability of our approach.

Table 2. Free-view Image-goal Navigation Results. SR: Success Rate, SPL: Success weighted by Path Length.
<table><tr><td rowspan="3">Method</td><td colspan="8">Narrow FOV  $( 5 0 ^ { \circ } \sim 7 5 ^ { \circ } )$ </td><td colspan="8">Wide FOV (75Â° â¼ 100Â°)</td></tr><tr><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="2">Hard</td><td colspan="2">Overall</td><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="2">Hard</td><td colspan="2">Overall</td></tr><tr><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td>SR</td><td>SPL</td><td></td><td>SPL</td><td>SR</td><td>SPL</td><td></td><td>SPL</td></tr><tr><td colspan="10">Zero-shot Transfer (Training on Image-goal Navigation Data)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDPPO [30]</td><td>15.8</td><td>10.5</td><td>9.6</td><td>7.2</td><td>5.4</td><td>3.1</td><td>10.3</td><td>6.9</td><td>20.2</td><td>16.5</td><td>16.6</td><td>12.5</td><td>9.8</td><td>5.7</td><td>15.5</td><td>11.6</td></tr><tr><td> RNS [17</td><td>19.8</td><td>10.6</td><td>15.8</td><td>9.0</td><td>7.8</td><td>4.0</td><td>14.5</td><td>7.9</td><td>28.4</td><td>16.6</td><td>21.2</td><td>14.5</td><td>10.6</td><td>5.9</td><td>20.1</td><td>12.3</td></tr><tr><td>O VRL [33]</td><td>23.8</td><td>16.6</td><td>19.2</td><td>10.5</td><td>8.2</td><td>6.9</td><td>17.1</td><td>11.3</td><td>27.6</td><td>19.2</td><td>22.8</td><td>12.6</td><td>14.8</td><td>8.6</td><td>21.7</td><td>13.5</td></tr><tr><td>NRNS + SLING [29]</td><td>32.8</td><td>15.3</td><td>23.6</td><td>13.2</td><td>9.8</td><td>5.6</td><td>22.1</td><td>11.4</td><td>38.6</td><td>19.1</td><td>32.6</td><td>18.5</td><td>17.2</td><td>8.3</td><td>29.5</td><td>15.3</td></tr><tr><td>OVRL + SLING [29]</td><td>28.2</td><td>20.1</td><td>23.2</td><td>18.7</td><td>11.8</td><td>7.1</td><td>21.1</td><td>15.3</td><td>36.4</td><td>25.9</td><td>31.6</td><td>18.5</td><td>15.2</td><td>7.6</td><td>27.7</td><td>17.3</td></tr><tr><td>IGL-Nav (Ours)</td><td>53.2</td><td>45.1</td><td>47.8</td><td>40.5</td><td>28.2</td><td>22.0</td><td>43.1</td><td>35.9</td><td>56.2</td><td>48.3</td><td>55.2</td><td>46.1</td><td>30.8</td><td>23.9</td><td>47.4</td><td>39.4</td></tr><tr><td colspan="10">Supervised (Training on Free-view Image-goal Navigation Data)</td><td colspan="7"></td></tr><tr><td>BC + GRU</td><td>13.2</td><td>6.8</td><td>10.4</td><td>8.8</td><td>6.6</td><td>4.9</td><td>10.1</td><td>6.8</td><td>22.0</td><td>14.4</td><td>16.0</td><td>11.4</td><td>9.2</td><td>6.9</td><td>15.7</td><td>10.9</td></tr><tr><td>BC + Metric Map</td><td>22.8</td><td>15.9</td><td>20.6</td><td>15.6</td><td>7.4</td><td>5.2</td><td>16.9</td><td>12.2</td><td>25.4</td><td>19.5</td><td>22.8</td><td>18.5</td><td>4.8</td><td>3.5</td><td>17.7</td><td>13.8</td></tr><tr><td>DDPO [30]</td><td>19.4</td><td>11.3</td><td>16.4</td><td>10.4</td><td>9.6</td><td>6.0</td><td>15.1</td><td>9.2</td><td>26.8</td><td>17.8</td><td>19.0</td><td>12.4</td><td>15.6</td><td>9.8</td><td>20.5</td><td>13.3</td></tr><tr><td> RNS [7]</td><td>30.8</td><td>24.4</td><td>27.8</td><td>24.5</td><td>11.2</td><td>8.9</td><td>23.3</td><td>19.3</td><td>39.6</td><td>35.0</td><td>35.8</td><td>30.1</td><td>13.8</td><td>8.3</td><td>29.7</td><td>24.5</td></tr><tr><td>NRNS + SLING [29]</td><td>40.2 70.4</td><td>31.8</td><td>37.2</td><td>23.9</td><td>19.8</td><td>9.8</td><td>32.4</td><td>21.8</td><td>49.8</td><td>35.0</td><td>40.8</td><td>30.4</td><td>21.6</td><td>12.7</td><td>37.4</td><td>26.0</td></tr><tr><td>IGL-Nav (Ours)</td><td></td><td>64.2</td><td>60.6</td><td>51.4</td><td>40.0</td><td>28.9</td><td>57.0</td><td>48.2</td><td>77.2</td><td>73.1</td><td>69.8</td><td>60.1</td><td>42.8</td><td>31.9</td><td>63.3</td><td>55.0</td></tr></table>

Table 3. Performance of IGL-Nav when depth and camera intrinsics are unavailable.
<table><tr><td>Method</td><td colspan="2">Narrow FOV  $( 5 0 ^ { \circ } \sim 7 5 ^ { \circ } )$  SR SPL</td><td>Wide FOV (75 SR</td><td>â¼ 100Â°) SPL</td></tr><tr><td>Predicted Depth</td><td>53.8</td><td>44.7</td><td>61.0</td><td>51.7</td></tr><tr><td>Measured Depth</td><td>57.0</td><td>48.2</td><td>63.3</td><td>55.0</td></tr></table>

<!-- image-->  
Figure 5. Rendering results of our incremental 3DGS.

## 4.3. Analysis of IGL-Nav

We further conduct in-depth module-by-module analysis on our IGL-Nav framework with sufficient visualization results and ablation studies, which is divided into three parts according to our module design. All ablation studies are conducted on the free-view image-goal setting.

Incremental 3DGS Prediction. Following the setting of RNR-Map [14], we assume depth information and camera intrinsics are known in our experiments. When these information is unavailable, we can simply adopt a depth estimator [21] to predict them. As shown in Table 3, with predicted depth and camera intrinsics, the performance of IGL-Nav is still robust. We further visualize rendering results of our 3DGS representation in Figure 5. Although maintained in an incremental and feed-forward manner, our 3DGS still demonstrates photorealistic novel view synthesis capability.

Table 4. Effects of different subdivision levels in coarse target localization to the final performance.
<table><tr><td>Level (Î³)</td><td>Narrow FOV</td><td> $( 5 0 ^ { \circ } \sim 7 5 ^ { \circ } )$  SPL</td><td>Wide FOV (75Â° SR</td><td>â¼ 100Â°) SPL</td></tr><tr><td></td><td>SR</td><td></td><td></td><td>16.8</td></tr><tr><td>1 2</td><td>19.7 41.3</td><td>12.0 34.4</td><td>24.9 48.9</td><td>42.1</td></tr><tr><td>3</td><td>57.0</td><td>48.2</td><td>63.3</td><td>55.0</td></tr></table>

Table 5. Effects of different stoppers in fine target localization to the final performance.
<table><tr><td>Stopper</td><td colspan="2">Narrow FOV (50 â¼ 75Â°) SR SPL</td><td>Wide FOV (75 â¼ 100Â°) SR</td><td>SPL</td></tr><tr><td>IGL-Nav w/out Stopper</td><td>45.7</td><td>32.9</td><td>46.2</td><td>37.6</td></tr><tr><td>IGL-Nav w/ SLING [29]</td><td>49.0</td><td>40.7</td><td>52.4</td><td>45.0</td></tr><tr><td>IGL-Nav</td><td>57.0</td><td>48.2</td><td>63.3</td><td>55.0</td></tr></table>

Coarse Target Localization. In our coarse localization module, the sphere space is discretized with a regular icosahedron and its Î³-level fractal. A larger value of $\gamma$ leads to finer discretization of the spherical surface and results in a greater number of convolution kernels N. We study the the effects of different levels to the final performance in Table 4. It is shown that using a 3-level subdivision achieves best performance, because a finer discretization will reduce quantization error and improve the accuracy of coarse localization. However, a larger Î³ results in high computational cost, making training inefficient.

Fine Target Localization. We compare different stoppers in Table 5. The first row refers to only using coarse target localization, and the second row refers to using the widely adopted SLING [29] as the stopper and fine localization module. It is shown that our 3DGS-based stopper and matching-constrained optimization is more suitable for the overall navigation system of IGL-Nav.

<!-- image-->  
Figure 7. Visualization of navigation process in the real world. The agent is successfully guided to a free-view goal image captured by a cellphone in complex indoor environments. IGL-Nav exhibits strong generalization ability and sim-to-real transfer performance.

We also visualize the navigation process in Figure 6. The agent is guided with frontier location, activation map obtained with 3D convolution and iterative pose optimization during the exploration. It is shown that our IGL-Nav can effectively localize the target image even with partial observation, and accurately guide the agent to final location with fine-grained rendering-based optimization.

## 4.4. Real-world Deployment

We further deploy IGL-Nav on real-world robotic platform to test its generalization ability. The model is directly taken from the free-view image-goal setting (supervised) without any finetuning on real-world data. As shown in Figure 1 and 7, we use a cellphone to capture the target image in a viewpoint that is unreachable by the robotic agentâs camera. Benefit from the flexible rendering capability of 3DGS representation, the agent effectively reaches this free-view goal with the coarse-to-fine localization method.

## 5. Conclusion

In this paper , we have proposed IGL-Nav for efficient and 3D-aware image-goal navigation. We incrementally maintain a 3DGS scene representation in feed-forward manner, which is then utilized for coarse-to-fine target localization. We analyze the pose space of the goal image and discretize both the pose space and scene embedding to apply efficient 3D convolution-based coarse matching. When the agent is close to the goal, we switch to fine localization by optimizing the camera pose via differentiable rendering on the confident matching pairs. The proposed IGL-Nav significantly outperforms existing state-of-the-art methods on image-goal and free-view image-goal settings. Real-world experiments further demonstrate our generalization ability. A limitation of IGL-Nav is that it requires depth and camera intrinsics of goal image. However, as we show in experiments, using existing monocular depth estimation [21] to predict them can satisfactorily solve this problem.

## References

[1] Ziad Al-Halah, Santhosh Kumar Ramakrishnan, and Kristen Grauman. Zero experience required: Plug & play modular transfer learning for semantic visual navigation. In CVPR, pages 17031â17041, 2022. 2, 6

[2] Devendra Singh Chaplot, Dhiraj Prakashchand Gandhi, Abhinav Gupta, and Russ R Salakhutdinov. Object goal navigation using goal-oriented semantic exploration. In NeurIPS, pages 4247â4258, 2020. 1

[3] Devendra Singh Chaplot, Ruslan Salakhutdinov, Abhinav Gupta, and Saurabh Gupta. Neural topological slam for visual navigation. In CVPR, pages 12875â12884, 2020. 1, 2

[4] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR, pages 19457â19467, 2024. 2, 3

[5] Anpei Chen, Haofei Xu, Stefano Esposito, Siyu Tang, and Andreas Geiger. Lara: Efficient large-baseline radiance fields. In ECCV, pages 338â355. Springer, 2024.

[6] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In ECCV, pages 370â386. Springer, 2024. 2, 3

[7] Meera Hahn, Devendra Singh Chaplot, Shubham Tulsiani, Mustafa Mukadam, James M Rehg, and Abhinav Gupta. No rl, no simulation: Learning to navigate without navigating. NeurIPS, 34:26661â26673, 2021. 1, 3, 6, 7

[8] Faith Johnson, Bryan Bo Cao, Ashwin Ashok, Shubham Jain, and Kristin Dana. Feudal networks for visual navigation. arXiv preprint arXiv:2402.12498, 2024. 6

[9] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In CVPR, pages 21357â21366, 2024. 5

[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. TOG, 42(4):139â1, 2023. 2

[11] Nuri Kim, Obin Kwon, Hwiyeon Yoo, Yunho Choi, Jeongho Park, and Songhwai Oh. Topological semantic graph memory for image-goal navigation. In CoRL, pages 393â402. PMLR, 2023. 2

[12] Jacob Krantz, Stefan Lee, Jitendra Malik, Dhruv Batra, and Devendra Singh Chaplot. Instance-specific image goal navigation: Training embodied agents to find object instances. arXiv preprint arXiv:2211.15876, 2022. 2

[13] Jacob Krantz, Theophile Gervet, Karmesh Yadav, Austin Wang, Chris Paxton, Roozbeh Mottaghi, Dhruv Batra, Jitendra Malik, Stefan Lee, and Devendra Singh Chaplot. Navigating to objects specified by images. In ICCV, pages 10916â10925, 2023. 3

[14] Obin Kwon, Jeongho Park, and Songhwai Oh. Renderable neural radiance map for visual navigation. In CVPR, pages 9099â9108, 2023. 1, 2, 3, 6, 7

[15] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders

for object detection from point clouds. In CVPR, pages 12697â12705, 2019. 5

[16] Xiaohan Lei, Min Wang, Wengang Zhou, and Houqiang Li. Gaussnav: Gaussian splatting for visual navigation. T-PAMI, 2025. 2

[17] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In Â´ ICCV, pages 2980â2988, 2017. 5

[18] Haozhe Lou, Yurong Liu, Yike Pan, Yiran Geng, Jianteng Chen, Wenlong Ma, Chenglong Li, Lin Wang, Hengzhen Feng, Lu Shi, et al. Robo-gs: A physics consistent spatialtemporal model for robotic arm with hybrid representation. arXiv preprint arXiv:2408.14873, 2024. 2

[19] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu, Jiwen Lu, and Yansong Tang. Manigaussian: Dynamic gaussian splatting for multi-task robotic manipulation. In ECCV, pages 349â366. Springer, 2024. 2

[20] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, pages 405â421. Springer, 2020. 1

[21] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In CVPR, pages 10106â10116, 2024. 3, 5, 7, 8

[22] Mohammad Nomaan Qureshi, Sparsh Garg, Francisco Yandun, David Held, George Kantor, and Abhisesh Silwal. Splatsim: Zero-shot sim2real transfer of rgb manipulation policies using gaussian splatting. arXiv preprint arXiv:2409.10161, 2024. 2

[23] Santhosh Kumar Ramakrishnan, Devendra Singh Chaplot, Ziad Al-Halah, Jitendra Malik, and Kristen Grauman. Poni: Potential functions for objectgoal navigation with interaction-free learning. In CVPR, pages 18890â18900, 2022. 1

[24] Nikolay Savinov, Alexey Dosovitskiy, and Vladlen Koltun. Semi-parametric topological memory for navigation. arXiv preprint arXiv:1803.00653, 2018. 2

[25] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A platform for embodied ai research. In ICCV, pages 9339â9347, 2019. 6

[26] James A Sethian. Fast marching methods. SIAM review, 41 (2):199â235, 1999. 6

[27] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In CVPR, pages 8922â8931, 2021. 5

[28] Yuan Sun, Xuan Wang, Yunfan Zhang, Jie Zhang, Caigui Jiang, Yu Guo, and Fei Wang. icomma: Inverting 3d gaussians splatting for camera pose estimation via comparing and matching. arXiv preprint arXiv:2312.09031, 2023. 5

[29] Justin Wasserman, Karmesh Yadav, Girish Chowdhary, Abhinav Gupta, and Unnat Jain. Last-mile embodied visual navigation. In CoRL, pages 666â678. PMLR, 2023. 2, 6, 7, 8

[30] Erik Wijmans, Abhishek Kadian, Ari Morcos, Stefan Lee, Irfan Essa, Devi Parikh, Manolis Savva, and Dhruv Batra. Dd-ppo: Learning near-perfect pointgoal navigators from 2.5 billion frames. arXiv preprint arXiv:1911.00357, 2019. 6, 7

[31] Fei Xia, Amir R Zamir, Zhiyang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson env: Real-world perception for embodied agents. In CVPR, pages 9068â9079, 2018. 6

[32] Karmesh Yadav, Arjun Majumdar, Ram Ramrakhya, Naoki Yokoyama, Alexei Baevski, Zsolt Kira, Oleksandr Maksymets, and Dhruv Batra. Ovrl-v2: A simple stateof-art baseline for imagenav and objectnav. arXiv preprint arXiv:2303.07798, 2023. 1, 2

[33] Karmesh Yadav, Ram Ramrakhya, Arjun Majumdar, Vincent-Pierre Berges, Sachit Kuhar, Dhruv Batra, Alexei Baevski, and Oleksandr Maksymets. Offline visual representation learning for embodied navigation. In ICLRW, 2023. 2, 6, 7

[34] Hang Yin, Xiuwei Xu, Zhenyu Wu, Jie Zhou, and Jiwen Lu. Sg-nav: Online 3d scene graph prompting for llm-based zero-shot object navigation. In NeurIPS, 2024. 1

[35] Hang Yin, Xiuwei Xu, Linqing Zhao, Ziwei Wang, Jie Zhou, and Jiwen Lu. Unigoal: Towards universal zero-shot goaloriented navigation. In CVPR, pages 19057â19066, 2025. 1

[36] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Centerbased 3d object detection and tracking. In CVPR, pages 11784â11793, 2021. 5

[37] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586â595, 2018. 3

[38] Yuhang Zheng, Xiangyu Chen, Yupeng Zheng, Songen Gu, Runyi Yang, Bu Jin, Pengfei Li, Chengliang Zhong, Zengmao Wang, Lina Liu, et al. Gaussiangrasper: 3d language gaussian splatting for open-vocabulary robotic grasping. RA-L, 2024. 2

[39] Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi. Target-driven visual navigation in indoor scenes using deep reinforcement learning. In ICRA, pages 3357â3364. IEEE, 2017. 1, 2