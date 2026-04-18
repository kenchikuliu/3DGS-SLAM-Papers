# ActiveSplat: High-Fidelity Scene Reconstruction through Active Gaussian Splatting

Yuetao Li1,2â, Zijia Kuang2â, Ting Li2, Qun Hao1, Zike Yan2â , Guyue Zhou2,3, Shaohui Zhang1â 

AbstractâWe propose ActiveSplat, an autonomous highfidelity reconstruction system leveraging Gaussian splatting. Taking advantage of efficient and realistic rendering, the system establishes a unified framework for online mapping, viewpoint selection, and path planning. The key to ActiveSplat is a hybrid map representation that integrates both dense information about the environment and a sparse abstraction of the workspace. Therefore, the system leverages sparse topology for efficient viewpoint sampling and path planning, while exploiting viewdependent dense prediction for viewpoint selection, facilitating efficient decision-making with promising accuracy and completeness. A hierarchical planning strategy based on the topological map is adopted to mitigate repetitive trajectories and improve local granularity given limited time budgets, ensuring highfidelity reconstruction with photorealistic view synthesis. Extensive experiments and ablation studies validate the efficacy of the proposed method in terms of reconstruction accuracy, data coverage, and exploration efficiency. The released code will be available on our project page1.

Index TermsâAutonomous Agents, Mapping, RGB-D perception

## I. INTRODUCTION

F INE-GRAINED reconstruction of three-dimensional en-vironments has long been a central research focus in vironments has long been a central research focus in robotics, computer vision, and computer graphics. Within the robotics community, there is a growing demand for highfidelity digitization of the physical world, not only to facilitate immersive applications like teleoperation [1], but also to narrow the sim-to-real gap, advancing generalizable robot autonomy through photorealistic simulation [2].

Recent progress in differentiable rendering has significantly improved the quality of the reconstructed environments. Neural radiance fields (NeRF) [3] and its variant [4] leverage neural networks as compact scene representations, using volume rendering to synthesize high-quality novel views. However, the computational inefficiencies caused by volume integration along rays pose challenges in terms of memory and processing. To address these limitations, Gaussian splatting [5], [6] has been introduced, enabling efficient rasterization and achieving promising rendering quality through Î±-blending. Despite these advances, scene-specific and data-dependent optimization makes these methods highly sensitive to noise and artifacts, which often emerge due to insufficient view coverage, especially without ground-truth supervision during data collection.

<!-- image-->  
Fig. 1. The agent explores the environment autonomously to build a 3D map on the fly. The integration of a Gaussian map and a Voronoi graph ensures efficient and complete exploration, resulting in high-fidelity reconstruction and a high completion ratio (Comp. %).

In this work, we aim to address these issues through active mapping, where a mobile agent reconstructs the environment on the fly, assesses the instant quality of the map, and plans its path to cover the entire environment. We find Gaussian splatting to be particularly suitable for high-fidelity active mapping, owing to its capability for view-dependent dense predictions. This characteristic enables the system to efficiently and accurately extract the workspace2 within the horizontal plane, while also quantifying data coverage in a unified manner by splatting Gaussians of interest onto the actively sampled views. The proposed system, dubbed ActiveSplat, incrementally updates a renderable Gaussian map through gradient-based optimization, progressively refining and completing the scene representation with high fidelity.

To balance reconstruction accuracy and exploration efficiency, our system adopts a hybrid map representation inspired by [7], but replaces the volumetric neural fields with explicit 3D Gaussians, enabling significantly faster convergence and real-time rendering essential for online mapping. A set of 3D Gaussians is maintained as a dense map to provide view-dependent dense predictions, while a Voronoi graph is extracted as a topological map to represent the abstraction of the workspace. Sparse yet representative view positions are derived from this graph, guiding the agent to extend the boundaries of the workspace. Meanwhile, the viewing orientation at each position is determined by view-dependent completeness measures of the Gaussian map. Based on this, a viewpoint decoupling method is proposed to reduce the infinite number of possible viewpoints in free space to a manageable set of positions and rotation angles, ensuring efficient and safe traversal. Additionally, a hierarchical planning strategy based on the topological map is employed to reduce redundant trajectories during global exploration and improve the overall efficiency of the autonomy process. The key contributions of the letter can then be categorized as follows:

â¢ A novel system that actively splats Gaussians of interest to build a unified, autonomous, and high-fidelity reconstruction system.

â¢ A hybrid map representation combining dense predictions with Gaussians and sparse abstraction of Voronoi graph for comprehensive viewpoint selection and safe path planning.

â¢ A hierarchical planning strategy based on the Voronoi graph, which prioritizes local areas to minimize redundant exploration, decoupling viewpoint selection to balance exploration efficiency and reconstruction accuracy.

## II. RELATED WORK

## A. Autonomous Exploration

In the robotic community, autonomous exploration aims to best acquire observations to cover the entire space traversed by the robot. Existing strategies seek to balance exploration completeness and efficiency, and can be broadly categorized as frontier-based methods and sampling-based methods. Frontierbased methods [8], [9] focus on expanding the exploration area by navigating to the boundary between explored and unexplored regions until full coverage is achieved. However, these methods rely on the discrete grid representation to discern the decision boundary, thus lacking adaptive granularity given diverse geometry complexity. In contrast, samplingbased methods [7], [10], [11] sample candidate viewpoints and prioritize those that maximize uncertainty reduction or expected information gain, thus improving scene coverage by reducing environmental uncertainty. Efforts are made to design proper sampling strategies for efficiency and precise scoring techniques given the samples. TARE [12] introduces a hierarchical strategy for LiDAR-based exploration, where a local subspace is traversed at a fine-grained level while global target goals at a coarse level are maintained, thus achieving a balance between exploration efficiency and mapping completeness. Similarly, [13] adopts fine-grained next-best-view planning for local exploration, while leveraging frontier-based strategies for global coverage. Topology maps have become widely used as sparse scene abstractions in autonomous exploration [7], [14]. Recently, most relevant to ours is [7], which also leverages a hybrid representation containing a dense neural map and a topology map for exploration. However, the neural map suffers from slow convergence and inefficient volume rendering. We, on the other hand, leverage the efficient optimization and rendering of Gaussian primitives to achieve reconstruction with high fidelity. Furthermore, we adopt a decoupled viewpoint selection to separately address translation and rotation during exploration, along with a topology-based hierarchical planning strategy, ultimately achieving a balance between exploration efficiency and completeness.

## B. High-Fidelity Scene Reconstruction

Recent progress in differentiable rendering attracts significant attention in the research community. Parameterized by implicit NeRF representations [3], [4] or explicit 2D/3D Gaussian representations [5], [6], photorealistic images of novel views can be rendered with promising efficiency. Gradientbased optimization has also been applied in an online setting to incrementally update the neural map [15], [16] or Gaussian parameters [17]â[19] through differentiable rendering. Recently, continual learning of the neural map has turned into an active fashion through uncertainty-guided autonomous exploration [7], [11], [20]â[22]. However, the neural representation has to sacrifice model capacity to achieve real-time performance, requiring inevitable tradeoffs between accuracy and convergence efficiency with different network architectures. In contrast, we adopt a set of Gaussian primitives as the scene representation, which allows consistent optimization of Gaussian parameters in an online update or offline post-processing setting. Relevant works have also developed Gaussian splatting-based mapping systems using unmanned ground vehicle (UGV) [23]â[26] and unmanned aerial vehicle (UAV) [27], [28] platforms. Notably, both the safe navigation system [23] and HGS-Planner [25] build upon FisherRF [29], which utilizes Fisher information for quantifying the uncertainty of Gaussians. ActiveGS [28] tackles similar challenges by modeling Gaussian confidence and integrating splatting with voxel maps to improve scene reconstruction. The major difference lies in our proposed hybrid map representation, which enforces safe and hierarchical path planning based on a topological map.

## III. METHODOLOGY

Our ActiveSplat system is a Gaussian splatting-based active mapping framework that aims to maximize scene completeness and reconstruction accuracy through autonomous exploration using posed RGB-D input. The hybrid map and hierarchical planning are introduced to improve exploration efficiency within a limited number of steps, balancing the tradeoff between exploration path length and reconstruction completeness. The overview of the system is illustrated in Fig. 2, which shows that Gaussians of interest are splatted onto the image plane, serving as a consistent technique for online map updating (Sec. III-A), viewpoint selection (Sec. III-B), and path planning (Sec. III-C).

<!-- image-->  
Fig. 2. Overview of ActiveSplat: The proposed active mapping system achieves high-fidelity reconstruction through a perception-action closed loop, utilizing a hybrid map that combines dense Gaussians with topological abstractions. Splatting Gaussians of interest on the fly provides a consistent approach for online map updating, viewpoint selection, and path planning. Note: Subregions are distinguished by node color, with node scores indicated by color intensity.

## A. Hybrid Map Updating

Central to the proposed ActiveSplat system is a hybrid map representation containing both Gaussian primitives, which allow dense prediction, and a topological structure, which provides sparse abstraction of the workspace. A Gaussian primitive is an explicit representation parameterized by color c, center position Âµ, anisotropic covariance Î£, and opacity o, where the influence of each Gaussian can be expressed as:

$$
f _ { i } ( \mathbf { u } _ { k } ) = o \cdot \exp \left( - \frac { 1 } { 2 } ( { \mathbf { x } } ( \mathbf { u } _ { k } ) - { \pmb \mu } _ { i } ) ^ { \top } { \Sigma } ^ { - 1 } ( { \mathbf { x } } ( \mathbf { u } _ { k } ) - { \pmb \mu } _ { i } ) \right) .\tag{1}
$$

View synthesis can then be implemented through splatting given the Gaussian map and a camera pose, where the color of each pixel u is linearly affected by the projected 3D Gaussians as:

$$
\hat { C } _ { k } = \sum _ { i = 1 } ^ { n _ { k } } \mathbf { c } _ { i } f _ { i } ( \mathbf { u } _ { k } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \mathbf { u } _ { k } ) \right) .\tag{2}
$$

Similarly, the differentiable rendering can also be applied for depth and visibility (accumulated opacity) estimation:

$$
{ \hat { D } } _ { k } = \sum _ { i = 1 } ^ { n _ { k } } d _ { i } f _ { i } ( \mathbf { u } _ { k } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \mathbf { u } _ { k } ) \right) ,\tag{3}
$$

$$
{ \hat { O } } _ { k } = \sum _ { i = 1 } ^ { n _ { k } } f _ { i } ( \mathbf { u } _ { k } ) \prod _ { j = 1 } ^ { i - 1 } \left( 1 - f _ { j } ( \mathbf { u } _ { k } ) \right) ,\tag{4}
$$

where $d _ { i }$ is the depth of the Gaussian center in the camera coordinate.

The optimization of the Gaussian map is performed given photometric and geometric losses defined in [17]:

$$
{ \cal L } _ { p h o } = \lambda _ { 1 } \left| C _ { k } - \hat { C } _ { k } \right| + \lambda _ { 2 } \left( 1 - { \cal { S } } { \cal { S } } { \cal { I } } { \bf { M } } ( C _ { k } , \hat { C } _ { k } ) \right) ,\tag{5}
$$

$$
L _ { g e o } = | D _ { k } - \hat { D } _ { k } | ,\tag{6}
$$

$$
L = w _ { c } L _ { p h o } + w _ { d } L _ { g e o } ,\tag{7}
$$

where $\lambda _ { 1 } = 0 . 8 , \lambda _ { 2 } = 0 . 2 , w _ { c } = 0 . 5 , w _ { d } = 1 . 0$ and $C _ { k } , D _ { k }$ are the captured RGB-D images of the k-th frame.

During the online mapping process, new Gaussians are dynamically initialized to cover newly observed areas, and redundant Gaussians with near-zero opacity or large covariances are removed as in [5]. Following [17], areas with low accumulated opacity or geometric deviations are identified as newly-observed areas:

$$
M _ { k } = \left( \hat { O } _ { k } < \tau _ { \mathrm { o l } } \right) \vee \left( ( D _ { k } < \hat { D } _ { k } ) \wedge ( | D _ { k } - \hat { D } _ { k } | > \epsilon _ { \mathrm { M D E } } ) \right) ,\tag{8}
$$

where $\tau _ { \mathrm { o 1 } } = 0 . 9 8$ , and ÏµMDE represents 50 times median depth error.

The dense prediction of the Gaussian map allows convenient extraction of the workspace and the obstacles. As illustrated in Box 3 of Fig. 2, the top-down view can be efficiently rendered using a large focal length as the orthographic projection of the dense map. The region with sufficient accumulated opacity is taken as occupied, where areas above the ground and within the agentâs height represent the obstacles. Navigable workspace can then be extracted as occupied areas at the ground level, excluding the obstacles. A Voronoi graph ${ \mathcal { G } } =$ {V, E} is generated through Voronoi tessellation [30], with edges E being equidistant from the obstacles and nodes N as intersections where the edges terminate.

The Gaussian map and the Voronoi graph are complementary: The Gaussian map provides dense and complete information about previously visited regions of the scene, while the Voronoi graph offers a sparse structure of the workspace, which can also be seen as a strong deformation retract of the global free space [31] in topology. We show in the following that the integration of the two representations leads to an adaptive granularity of the environment and guarantees an effective trade-off between efficiency and accuracy during the autonomous reconstruction process.

## B. Active Viewpoint Selection

The objective of active mapping is to traverse the workspace and best capture the information of previously unseen areas. This is usually achieved by iteratively selecting target views, where the sampling strategy of accessible viewpoint candidates and the selection criteria are crucial to efficiency and overall coverage. We sample viewpoints on the Voronoi graph to maintain a compact and accessible set that covers the entire scene. Besides, the Voronoi graph generates a path that stays as far away from the obstacles as possible, thus guaranteeing a safe traversal [7].

1) Decoupled Position and Rotation Candidates: To best exploit the compact structure of the Voronoi graph along with the dense geometry and appearance of the Gaussian map, we propose decoupling the position and rotation candidates, rather than uniformly [25] or randomly [28] sampling viewpoints across the workspace. This approach reduces the search dimensionality while ensuring thorough observations. The dynamically updated map leads to a progressively refined graph that completely describes the partially-observed workspace. We iteratively select Voronoi nodes as viewpoint position candidates, where the node with the highest score (calculated as in Eq. 10) favors unvisited areas that push the boundary of the workspace for traversal. Regarding the view rotation, we adopt the yaw and pitch rotations at the selected view positions to get the observation toward a specific region. The target node position and the target rotation angle are determined in a viewdependent manner as follows.

2) Coverage Evaluation: The actions regarding translation and rotation during the autonomous exploration process undergo different granularity. We aim to efficiently traverse the entire set of Voronoi nodes to maintain complete coverage, while conducting careful inspection in an area with intricate intersections of paths [7]. In practice, we splat the constructed Gaussians to quickly obtain surrounding information of a specified pose, render panoramic images of the visibility at all unvisited nodes, as illustrated in Fig. 2. To generate a sparse and representative set of rotation candidates, we apply the DBSCAN algorithm to cluster low-visibility regions (highlighted in red) within the panoramic view, where the pixel coordinate of the cluster center indicates yaw and pitch angles, as elaborated in Sec. III-D2.

Note that the view-dependent accumulated opacity does not precisely reflect the reconstruction accuracy of the entire space. Firstly, the proportion of low-visibility areas in the image domain does not reflect the actual unseen space in three dimensions, as a node close to the unseen area results in a large amount of invisible pixels. Secondly, the accumulated opacity leads to an over-confident evaluation of completeness, as the splatting from backside geometry can also result in high accumulated opacity. To address these challenges, our method back-projects contour pixels, utilizing convex hull [32] to get the approximated volume of the incomplete geometry. We further maintain a set of high-loss samples through joint analysis of observation depth $D _ { k }$ and rendered depth $\hat { D } _ { k } ^ { \mathrm { ~ ~ } }$

<!-- image-->  
Fig. 3. Once the agent gets sufficient observations within a local region $R _ { l }$ (the green nodes), it selects the next sub-area $R _ { n }$ within the highest score (the orange node) globally for further exploration.

$$
M _ { h } = ( \hat { O } _ { k } > \tau _ { \mathrm { o 2 } } ) \wedge ( D _ { k } < \hat { D } _ { k } ) \wedge ( | D _ { k } - \hat { D } _ { k } | > \epsilon _ { 1 } ) ,\tag{9}
$$

where $\tau _ { \mathrm { o 2 } } ~ = ~ 0 . 8 ,$ and $\epsilon _ { 1 } ~ = ~ 0 . 3$ . The high-loss areas at each frame, before densification, are clustered with temporal propagation to keep track the newly observed regions.

3) Determination of Target Views: The viewpoint selection is then conducted in two stages. The agent first selects the node with the most invisible areas, taking both panorama visibility measures and convex hull volumes into consideration. This strategy forces the agent to reach a closed space by fast marching the nodes with high scores, as detailed in Sec. III-D3, therefore expanding the workspace efficiently. Once the agent arrives at its goal position, the panorama image and the maintained high-loss samples guide the agent to rotate, as illustrated in Fig. 2 and Eq. 9, where invisible and high-loss areas get observations. The proposed method keeps the agent along the Voronoi graph that compresses potential viewpoints into a finite sparse set, reducing the computational complexity while guaranteeing completeness and safety.

## C. Hierarchical Planning with Voronoi Graph

In active mapping of multi-room environments, we observed that the prioritization of areas with high scores overlooks local coverage, resulting in paths that are unnecessarily visited multiple times, as shown in Fig. 5. The nodes of a Voronoi graph represent distinct reachable regions, and the edges efficiently assess traversal costs between them. Therefore, we propose a hierarchical planning strategy based on Voronoi graphs, which includes subregion partitioning and local-global goal selection, as illustrated in Fig. 3.

1) Subregion Partition: Building upon the topological structure of the Voronoi graph, we aim to dynamically partition it during the exploration into n subregions $R _ { n } .$ , including a local subregion $R _ { l }$ where the agent is currently located, ensuring fine-grained local granularity with global guidance. In practice, we adopt the agglomerative hierarchical clustering method (UPGMA) [33] for partitioning, with pairwise distances based on both Euclidean distance and travel path length. The hierarchy allows the flexibility to choose partitions at different levels and adapts to spatial data, without specifying the number of clusters beforehand.

2) Local-Global Goal Selection: The proposed method favors local exploration before global exploration. Once local areas are thoroughly explored, the next-best-subregion with the highest score is selected. Local planning is conducted by quantifying the incomplete score, as specified in Eq. 10, within the local subregion. The node with the highest score above a threshold is selected iteratively. Once the maximum score of the nodes within the local subregion falls below the threshold or all nodes are visited, the agent performs global planning by selecting the node outside the local subregion with the highest score. The global score not only takes the coverage into account, but also considers the distance cost along with the visited probabilities during past exploration. The active mapping process is then conducted to iterate between rigorous local mapping and coarse global exploration, to balance the reconstruction accuracy and efficiency.

## D. Implementation Details

Given the selected target positions and rotations, the agent actively explores the unknown environment and captures new information. In the following, we discuss the details regarding bootstrapping, panorama rendering, path planning, and postprocessing.

1) Boostrapping: Due to the limited field of view of the camera, we force the agent to look around at the very beginning. The agent takes discrete actions to execute 360 degrees of yaw rotation to obtain a complete ambient view.

2) Panorama Rendering: As the Gaussian splatting technique allows efficient rendering of pinhole images, we use three virtual cameras with 150 degrees of FOV vertically and 120 degrees of FOV horizontally to get the panoramic images, holistically quantifying the node-wise scores. The size of each panoramic image is set to be 360 Ã 150 to allow convenient selection of the rotation angle.

3) Path Planning: Once the target goal position is determined, the shortest path can then be found through Dijkstraâs algorithm. The score of the i-th node, denoted as $S _ { i } ,$ is implemented as a weighted sum of the following factors:

$$
S _ { i } = w _ { o } \cdot s _ { o } ( i ) + w _ { c } \cdot s _ { c } ( i ) + w _ { u } \cdot s _ { u } ( i ) + w _ { h } \cdot s _ { h } ( i ) ,\tag{10}
$$

where $w _ { o } ~ = ~ 2 0 , ~ w _ { c } ~ = ~ 1 0 , ~ w _ { u } ~ = ~ w _ { h } ~ = ~ 1 0 . ~ s _ { o }$ and $s _ { c }$ are the portion of areas regarding the 2D invisible subregion and the 3D convex hull. $s _ { u }$ and $s _ { h }$ are the boolean values of unvisited and in-horizon states. Nodes with the same score are ranked according to their distance from the agent, where the nearer node is favored. We set a fixed distance threshold of 2 meters to control the granularity of subregion partitioning. We also enforce rotation once the agent arrives at multi-connected nodes in the graph as they are often intersection points between regions that require careful decision-making. Experiments indicate the efficacy of this strategy, achieving better efficiency for thorough exploration compared to previous state-of-the-art approaches [7].

TABLE I  
COMPARISON AGAINST RELEVANT METHODS REGARDING THECOMPLETENESS OF THE OBSERVED DATA
<table><tr><td rowspan="2"></td><td colspan="2">Gibson</td><td colspan="2">Matterport3D</td></tr><tr><td> $\overline { { \% \uparrow } }$ </td><td>cmâ</td><td>%â</td><td>cmâ</td></tr><tr><td>FBE [8]</td><td>68.30</td><td>14.42</td><td>74.30</td><td>9.29</td></tr><tr><td>UPEN [20]</td><td>63.30</td><td>21.09</td><td>75.56</td><td>9.72</td></tr><tr><td>ANM [11]</td><td>80.45</td><td>7.44</td><td>79.36</td><td>7.40</td></tr><tr><td>NARUTO [21]</td><td>79.16</td><td>3.52</td><td>84.90</td><td>5.94</td></tr><tr><td>ANM-S [7]</td><td>92.10</td><td>2.83</td><td>89.74</td><td>4.14</td></tr><tr><td>Ours</td><td>92.24</td><td>2.43</td><td>92.48</td><td>2.84</td></tr></table>

4) Post-Processing: Unlike NeRF-based SLAM algorithms that sacrifice model capacity for fast convergence to meet real-time demand, Gaussian splatting-based approaches maintain a consistent parameter space that allows post-processing. Therefore, we further apply adaptive density controls, as well as depth and normal regularization [5], [6] to refine the online-constructed map given stored keyframe data. The reconstruction results are shown in Fig. 6.

## IV. EXPERIMENTS

The experiments are conducted on a desktop PC with an Intel Core i9-12900K CPU and an NVIDIA RTX 3090 GPU. The reported results are averaged across 5 trials.

## A. Experimental Setup

To ensure fairness across exploration strategies, we perform qualitative and quantitative evaluations on the visually realistic Gibson [34] and Matterport3D [35] datasets using the Habitat simulator [36], following the protocol of [11]. The singlefloor scenes in the test/validation set are divided into small (less than 5 rooms) and medium (5â10 rooms) scenes. Unless otherwise specified, the agent collects posed RGB-D data at a resolution of 256 Ã 256 and performs discrete actions: MOVE_FORWARD by 6.5 cm, TURN_LEFT and TURN_RIGHT by 10â¦, TURN_UP and TURN_DOWN by 15â¦, and STOP. The agent height is set to 1.25 m, with vertical and horizontal FOV of 90â¦. Additionally, the agent takes a $4 5 ^ { \circ }$ downward pitch rotation to ensure a closed ground surface before departure.

## B. Evaluation Metrics

Following [21] and [11], we evaluate exploration coverage using completion ratio (%) and completion (cm) across different map representations. To evaluate rendering quality, we use PSNR (dB), SSIM, and LPIPS for RGB rendering [5], and Depth L1 (cm) distance for depth rendering performance. Additionally, we further evaluate the path length traveled by the agent during exploration.

## C. Comparison to Other Methods

We first evaluate the exploration coverage across 13 different scenes following the setup of [11]. As shown in Tab. I, the proposed system outperforms all relevant methods within the limited steps (1000 for small scenes and 2000 for mediumscale scenes). NARUTO [21] achieves 92% coverage in simple scenes with a single room (e.g., MP3D-gZ6f7), thanks to its unrestricted movement capabilities, but is limited in complex scenes with multiple rooms (e.g., Cantwell and Eastville) due to its greedy strategy. Even though [7] adopts a similar strategy of topology-guided exploration, the proposed method outperforms the baseline due to hierarchical planning that balances local reconstruction granularity and global scene coverage. We also performed a qualitative evaluation of novel view synthesis. As shown in Fig. 4, the proposed method achieves better rendering quality of novel views when compared to the NeRF-based system [7], exhibiting sharp edges and ample textures, as detailed in the supplementary Sec. VI-A.

Gibson-Eudora  
Gibson-Ribera  
MP3D-pLe4w  
MP3D-gZ6f7  
<!-- image-->  
Fig. 4. The novel view synthesis results of ours compared to the NeRF-based active mapping [7] on Gibson and MP3D datasets. The bottom row of each picture shows the average PSNR (dB) and Depth L1 error (cm) of the scene from 50 randomly sampled test views.

TABLE II  
ABLATION OF EXPLORATION STRATEGY
<table><tr><td rowspan="2"></td><td colspan="2">Gibson</td><td colspan="2">Matterport3D</td></tr><tr><td>%â</td><td>cmâ</td><td>%â</td><td>cmâ</td></tr><tr><td>Random</td><td>84.20</td><td>6.13</td><td>83.91</td><td>5.50</td></tr><tr><td>Position</td><td>90.41</td><td>2.74</td><td>89.54</td><td>3.67</td></tr><tr><td>Viewpoint</td><td>91.76</td><td>2.30</td><td>92.38</td><td>2.85</td></tr><tr><td>Ours</td><td>92.24</td><td>2.43</td><td>92.48</td><td>2.84</td></tr></table>

## D. Ablation Study and System Performance

To validate the rationale behind our solution, we conduct ablation studies and performance analyses to justify the effectiveness of different modules.

1) Exploration Strategy: We first analyze the impact of different exploration strategies for thorough exploration, including a randomly sampled baseline (Random), exploration with node selection while ignoring target rotations (Position), decoupled selection of view positions and rotations (Viewpoint), and our proposed approach further incorporating multi-connected regions and hierarchical planning (Ours). As shown in Tab. II, different exploration strategies lead to diverse behaviors for efficiency-accuracy tradeoffs. The Random baseline indicates that the Voronoi graph guarantees complete exploration. Nevertheless, the traversal of all nodes without proper order is inefficient and overlooks certain areas. This issue remains for the greedy Position strategy as it only strives to push the boundaries of the workspace toward thorough traversal. The completeness can be improved with the rotation involved. We provide more detailed scene-wise results in the supplementary Sec. VI-B. Finally, the careful treatment of multi-connected nodes and the hierarchical planning (HP) strategy bring further advantages due to different inspection granularity locally and globally. Tab. III shows the results of an ablation in which we increase the number of steps to 4000 (in all scenes) and compare the results with and without hierarchical planning at different stages. Our method benefits from fine-grained observations within local subregions, enabling higher completeness in the later stages. Additionally, we visualize the exploration trajectory when exploring a mediumsized environment. As illustrated in Fig. 5, even though the greedy strategy leads to a rapid increase of completion in the beginning, coarsely exploring the neighboring areas results in repetitive trajectories.

TABLE III  
ABLATION OF HIERARCHICAL PLANNING
<table><tr><td rowspan="3"></td><td colspan="4">Gibson</td><td colspan="4">Matterport3D</td></tr><tr><td colspan="2">w/o HP</td><td colspan="2">Ours</td><td colspan="2">w/o HP</td><td colspan="2">Ours</td></tr><tr><td>%â</td><td>cmâ</td><td>%â</td><td>cmâ</td><td>%â</td><td>cmâ</td><td>%â</td><td>cmâ</td></tr><tr><td>25%</td><td>87.48</td><td>4.96</td><td>88.73</td><td>4.49</td><td>92.61</td><td>2.73</td><td>91.50</td><td>3.04</td></tr><tr><td>50%</td><td>93.00</td><td>2.15</td><td>94.26</td><td>1.75</td><td>95.14</td><td>2.18</td><td>95.07</td><td>2.21</td></tr><tr><td>75%</td><td>95.75</td><td>1.20</td><td>96.20</td><td>1.04</td><td>95.28</td><td>2.14</td><td>95.31</td><td>2.14</td></tr><tr><td>100%</td><td>96.38</td><td>1.01</td><td>96.77</td><td>0.90</td><td>95.33</td><td>2.11</td><td>95.42</td><td>2.11</td></tr></table>

Completion ratio (%) and completion error (cm) under different allowed path lengths (expressed as percentages of the full trajectory).

TABLE IV  
ABLATION OF COVERAGE EVALUATION
<table><tr><td rowspan="2"></td><td colspan="2">Gibson</td><td colspan="2">Matterport3D</td></tr><tr><td>%â</td><td>cmâ</td><td>%â</td><td>cmâ</td></tr><tr><td>Visibility only</td><td>90.19</td><td>3.09</td><td>91.40</td><td>3.15</td></tr><tr><td>Convex hull only</td><td>91.09</td><td>2.86</td><td>91.50</td><td>2.98</td></tr><tr><td>Ours</td><td>92.24</td><td>2.43</td><td>92.48</td><td>2.84</td></tr></table>

TABLE V

ABLATION OF POST-PROCESSING ON GIBSON-DENMARK
<table><tr><td></td><td>Depth loss</td><td>Split</td><td>Depth L1â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Online</td><td>True</td><td>Train Test</td><td>1.91 9.01</td><td>25.28 21.72</td><td>0.83 0.76</td><td>0.22 0.29</td></tr><tr><td rowspan="2">Refined with 3DGS</td><td>False</td><td>Train Test</td><td>4.49 11.2</td><td>39.10</td><td>0.98</td><td>0.03</td></tr><tr><td>True</td><td>Train</td><td>0.77</td><td>26.27 38.90</td><td>0.86 0.99</td><td>0.19 0.03</td></tr><tr><td rowspan="3">Refined with 2DGS</td><td>False</td><td>Test Train</td><td>7.81 3.94</td><td>26.87 38.84</td><td>0.88 0.99</td><td>0.17 0.04</td></tr><tr><td>True</td><td>Test Train</td><td>10.10 0.80</td><td>27.13 38.90</td><td>0.88 0.99</td><td>0.18 0.04</td></tr><tr><td></td><td>Test</td><td>7.56</td><td>27.58</td><td>0.88</td><td>0.17</td></tr></table>

2) Coverage Evaluation: As mentioned in Sec. III-B2, the quantification of visibility in the panoramic view guides the agent to push the boundary of the workspace. To verify the effectiveness of the integration of both the invisible mask area and convex hull volume, we compare different visibility quantification methods: guiding the agent to the node with the largest rendered area of invisible regions (Visibility only) and favoring the node with the largest 3D convex hull of the invisible boundary (Convex hull only). As shown in Tab. IV, using the view-dependent 2D results or the 3D volume quantification alone may not best evaluate the candidate nodes. The integration of both strategies (Ours) as weighted averaging of the normalized scores takes the relative extent of invisible areas near the Voronoi nodes into consideration, achieving promising completeness during the active mapping.

<!-- image-->  
Fig. 5. Ablation of hierarchical planning (Gibson Quantico): The online reconstruction progress with increased completion ratio (%) and path length (m) at different stages. The hierarchical planning strategy results in better completeness and reduced path length during the exploration.

<!-- image-->  
Fig. 6. Reconstruction results: The autonomous reconstruction lead to photorealistic rendering and accurate geometry at a resolution of 512 Ã 512. The left and right sides of each image show rendered RGB and depth.

3) Post-Processing: The recent Gaussian splatting technique allows convenient post-processing with the stored keyframe buffer. We here compare our results before and after the post-processing using 3DGS [5] and 2DGS [6] on Gibson Denmark. 50 frames of observations are selected uniformly as the train split, and 50 images with randomly sampled camera poses within the free space are taken as the test split. As shown in Tab. V, refinement can drastically enhance the reconstruction quality in terms of both geometry and appearance with the incorporation of RGB and depth during the optimization. The online autonomous exploration process allows active data capture for complete and high-fidelity reconstruction. It can be noted that the two-dimensional flattened Gaussian parameter representation of 2DGS [6], along with the geometric regularization terms, shows better results in the test split, while 3DGS [5] indicates better overfitting in the train split. Besides, the use of depth loss, as defined in Eq. 6, not only leads to better geometry (lower Depth L1) in both train and test splits, but also enhances the generalization of the map (better quality in the test split). Refinement without depth loss may result in more realistic view synthesis results on training views (severe overfitting), but the geometry may deteriorate due to ambiguities in textureless areas.

TABLE VI  
AVERAGE PROCESSING TIME PER STEP
<table><tr><td>Mapper</td><td>Get workspace</td><td>Get Voronoi</td><td>Get subregion</td></tr><tr><td>45.14ms</td><td>43.87ms</td><td>0.69ms</td><td>1.56ms</td></tr><tr><td>Rotation selection</td><td>Position selection</td><td>Path planner</td><td>Visualizer (optional)</td></tr><tr><td>2.76ms</td><td>6.46ms</td><td>0.20ms</td><td>67.43ms</td></tr></table>

<!-- image-->  
Fig. 7. Real-world experimental result in an office scene.

4) System Performance: The average processing time per step for each module is presented in Tab. VI. The proposed system runs at 8 fps in a headless mode, demonstrating realtime capable performance. Furthermore, the sparse decisionmaking ensures efficient viewpoint selection, while path planning constitutes only a small portion of the overall processing load. The major costs lie in the mapping and workspace extraction modules, which can be further accelerated with compressed Gaussian parameters.

## E. Deployment in Real World

To evaluate the practical usage of the proposed system in the real world, we deploy it on an omnidirectional mobile robot (Agile-X Ranger Mini) equipped with an Azure Kinect RGB-D sensor, where the camera pose is estimated by a linebased SLAM system [37] in a parallel thread. Compared to simulation settings, we adjusted the radius hyperparameter of visited nodes for more detailed active mapping. The online reconstruction results are shown in Fig. 7, depicting the complete exploration that progressively and autonomously reconstructs the indoor environment. The detailed reconstruction process with additional results is provided in the supplementary video.

## V. CONCLUSION

In this paper, we introduce an active mapping system for high-fidelity reconstruction of indoor scenes. Benefiting from the accurate dense prediction of a Gaussian splatting-based differentiable renderer and the workspace abstraction through Voronoi graph extraction, we employ the hybrid map along with a novel topology-based hierarchical planning strategy to achieve promising tradeoffs between exploration efficiency and completeness. Detailed experimental results indicate that the proposed system achieves effective tradeoffs between exploration coverage and reconstruction quality.

While ActiveSplat performs well, future work could further enhance its robustness, scalability, and real-world applicability. Integrating LiDAR into the Gaussian splatting-based SLAM system would enable robust tracking and more accurate geometric reconstruction in large-scale scenes. Moreover, highfidelity exploration in unknown indoor environments opens new avenues for research in robotic autonomy, extending to tasks like lifelong navigation and mobile manipulation.

## REFERENCES

[1] V. Patil and M. Hutter, âRadiance fields for robotic teleoperation,â in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2024, pp. 13 861â13 868.

[2] M. Torne, A. Simeonov, Z. Li, A. Chan, T. Chen, A. Gupta, and P. Agrawal, âReconciling reality through simulation: A real-to-sim-toreal approach for robust manipulation,â in Robotics: Science and Systems (RSS), 2024.

[3] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Commun. ACM, vol. 65, no. 1, pp. 99â106, 2021.

[4] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and p. Hedman, âZip-nerf: Anti-aliased grid-based neural radiance fields,â in Intl. Conf. on Computer Vision (ICCV), 2023, pp. 19 697â19 705.

[5] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Trans. Graphics, vol. 42, no. 4, pp. 1â14, 2023.

[6] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2d gaussian splatting for geometrically accurate radiance fields,â in SIGGRAPH. Association for Computing Machinery, 2024.

[7] Z. Kuang, Z. Yan, H. Zhao, G. Zhou, and H. Zha, âActive neural mapping at scale,â in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2024.

[8] B. Yamauchi, âA frontier-based approach for autonomous exploration,â in IEEE Intl. Sym. on Computational Intelligence in Robotics and Automation (CIRA), 1997, pp. 146â151.

[9] H. Umari and S. Mukhopadhyay, âAutonomous robotic exploration based on multiple rapidly-exploring randomized trees,â in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2017, pp. 1396â1402.

[10] L. Schmid, M. Pantic, R. Khanna, L. Ott, R. Siegwart, and J. Nieto, âAn efficient sampling-based method for online informative path planning in unknown environments,â IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 1500â1507, 2020.

[11] Z. Yan, H. Yang, and H. Zha, âActive neural mapping,â in Intl. Conf. on Computer Vision (ICCV), 2023.

[12] C. Cao, H. Zhu, H. Choset, and J. Zhang, âTare: A hierarchical framework for efficiently exploring complex 3d environments.â in Robotics: Science and Systems (RSS), vol. 5, 2021, p. 2.

[13] M. Selin, M. Tiger, D. Duberg, F. Heintz, and P. Jensfelt, âEfficient autonomous exploration planning of large-scale 3-d environments,â IEEE Robotics and Automation Letters, vol. 4, no. 2, pp. 1699â1706, 2019.

[14] Q. Dong, H. Xi, S. Zhang, Q. Bi, T. Li, Z. Wang, and X. Zhang, âFast and communication-efficient multi-uav exploration via voronoi partition on dynamic topological graph,â in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2024, pp. 14 063â14 070.

[15] Z. Zhu, S. Peng, V. Larsson, Z. Cui, M. R. Oswald, A. Geiger, and M. Pollefeys, âNicer-slam: Neural implicit scene encoding for rgb slam,â in Intl. Conf. on 3D Vision (3DV), March 2024.

[16] C. Jiang, H. Zhang, P. Liu, Z. Yu, H. Cheng, B. Zhou, and S. Shen, âH2-mapping: Real-time dense mapping using hierarchical hybrid representation,â IEEE Robotics and Automation Letters, 2023.

[17] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 357â21 366.

[18] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18 039â18 048.

[19] J. Wei and S. Leutenegger, âGsfusion: Online rgb-d mapping where gaussian splatting meets tsdf fusion,â IEEE Robotics and Automation Letters, 2024.

[20] G. Georgakis, B. Bucher, A. Arapin, K. Schmeckpeper, N. Matni, and K. Daniilidis, âUncertainty-driven planner for exploration and navigation,â in IEEE Intl. Conf. on Robotics and Automation (ICRA), 2022.

[21] Z. Feng, H. Zhan, Z. Chen, Q. Yan, X. Xu, C. Cai, B. Li, Q. Zhu, and Y. Xu, âNaruto: Neural active reconstruction from uncertain target observations,â in IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 572â21 583.

[22] X. Pan, Z. Lai, S. Song, and G. Huang, âActivenerf: Learning where to see with uncertainty estimation,â in European Conf. on Computer Vision (ECCV). Springer, 2022, pp. 230â246.

[23] G. Liu, W. Jiang, B. Lei, V. Pandey, K. Daniilidis, and N. Motee, âBeyond uncertainty: Risk-aware active view acquisition for safe robot navigation and 3d scene understanding with fisherrf,â arXiv preprint arXiv:2403.11396, 2024.

[24] R. Jin, Y. Gao, Y. Wang, Y. Wu, H. Lu, C. Xu, and F. Gao, âGsplanner: A gaussian-splatting-based planning framework for active highfidelity reconstruction,â in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2024, pp. 11 202â11 209.

[25] Z. Xu, R. Jin, K. Wu, Y. Zhao, Z. Zhang, J. Zhao, F. Gao, Z. Gan, and W. Ding, âHgs-planner: Hierarchical planning framework for active scene reconstruction using 3d gaussian splatting,â arXiv preprint arXiv:2409.17624, 2024.

[26] Y. Tao, D. Ong, V. Murali, I. Spasojevic, P. Chaudhari, and V. Kumar, âRt-guide: Real-time gaussian splatting for information-driven exploration,â arXiv preprint arXiv:2409.18122, 2024.

[27] T. Chen, O. Shorinwa, J. Bruno, A. Swann, J. Yu, W. Zeng, K. Nagami, P. Dames, and M. Schwager, âSplat-nav: Safe real-time robot navigation in gaussian splatting maps,â IEEE Trans. Robotics, 2025.

[28] L. Jin, X. Zhong, Y. Pan, J. Behley, C. Stachniss, and M. Popovic,Â´ âActivegs: Active scene reconstruction using gaussian splatting,â IEEE Robotics and Automation Letters, 2025.

[29] W. Jiang, B. Lei, and K. Daniilidis, âFisherrf: Active view selection and mapping with radiance fields using fisher information,â in European Conf. on Computer Vision (ECCV). Springer, 2024, pp. 422â440.

[30] Q. Du, V. Faber, and M. Gunzburger, âCentroidal voronoi tessellations: Applications and algorithms,â SIAM review, vol. 41, no. 4, pp. 637â676, 1999.

[31] J. Canny and B. Donald, âSimplified voronoi diagrams,â in Proceedings of the third annual symposium on Computational geometry, 1987, pp. 153â161.

[32] C. B. Barber, D. P. Dobkin, and H. Huhdanpaa, âThe quickhull algorithm for convex hulls,â ACM Trans. Math. Softw., vol. 22, no. 4, pp. 469â483, 1996.

[33] O. Arslan, D. P. Guralnik, and D. E. Koditschek, âCoordinated robot navigation via hierarchical clustering,â IEEE Trans. Robotics, vol. 32, no. 2, pp. 352â371, 2016.

[34] F. Xia, A. R. Zamir, Z. He, A. Sax, J. Malik, and S. Savarese, âGibson Env: real-world perception for embodied agents,â in IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2018.

[35] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Niessner, M. Savva, S. Song, A. Zeng, and Y. Zhang, âMatterport3d: Learning from rgb-d data in indoor environments,â in Intl. Conf. on 3D Vision (3DV), 2017.

[36] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik et al., âHabitat: A platform for embodied ai research,â in Intl. Conf. on Computer Vision (ICCV), 2019, pp. 9339â9347.

[37] Q. Wang, Z. Yan, J. Wang, F. Xue, W. Ma, and H. Zha, âLine flow based simultaneous localization and mapping,â IEEE Trans. Robotics, vol. 37, no. 5, pp. 1416â1432, 2021.

## Supplementary Material

## VI. FURTHER EXPERIMENTAL ANALYSIS

## A. Comparison of Rendering Quality with ANM-S

We add the quantitative results regarding the rendering quality when compared against ANM-S [7]. The evaluation follows the setup of Tab. V at a rendered resolution of 512 Ã 512 for both systems, where the robot trajectories vary given different mapping strategies. Therefore, the training views are not the same, while the test views are taken from the same set through random sampling. Post-processing requires 30,000 iterations for both methods. The quantitative results are presented as shown in Tab. VII. We present the averaged quantitative results in Fig. 4 as a supplement. As mentioned in Sec. II-B, Gaussian splatting-based representations allow more consistent optimization compared to NeRF-based representations, which sacrifice network capacity for efficiency, thereby achieving better rendering quality.

## B. Scene-wise Ablation Study on Robustness

Since the main paper only provides the mean values across all scenes, as shown in Tab. I and Tab. II, the overall variance is less informative due to the large performance gap among scenes. To better demonstrate the robustness of our method, we provide detailed results based on 5 ablation trials for each of the 13 scenes. The mean and standard deviation of the completion ratio for each scene are listed in Tab.VIII and Tab. IX, respectively.

## C. Supplementary Analysis on Coverage Evaluation

Here, in supplementary to Tab. IV, we provide one empirical result to validate the design. As shown in Fig. 8, the nodes in red and blue have higher invisibility scores compared to the green node, as they lie closer to the invisible areas. On the contrary, the green node has a higher 3D volume score of the corresponding convex hull compared to the red and blue nodes, as it lies near a larger area of open space to be explored. This indicates the biased scoring through view-based rendering criteria. Nonetheless, for the red and blue nodes that share similar non-visited areas, we favor the node that is closer to the areas, which can be reflected as a higher 2D invisibility score. Despite this empirical design, the selected target node usually has significantly higher scores compared to other nodes. Either case may not significantly affect the final results.

COMPARISON OF RENDERING QUALITY BETWEEN OUR METHOD AND ANM-S ON THE GIBSON AND MATTERPORT3D DATASETS, INCLUDING DEPTH L1 [cm], PSNR [DB], SSIM, AND LPIPS METRICS.  
TABLE VII
<table><tr><td rowspan="2">Scene</td><td rowspan="2">Split</td><td colspan="2">Depth L1â</td><td colspan="2">PSNRâ</td><td colspan="2">SSIMâ</td><td colspan="2">LPIPSâ</td></tr><tr><td>ANM-S</td><td>Ours</td><td>ANM-S</td><td>Ours</td><td>ANM-S</td><td>Ours</td><td>ANM-S</td><td>Ours</td></tr><tr><td rowspan="2">Gibson-Eudora</td><td>Train</td><td>2.38</td><td>1.07</td><td>28.46</td><td>37.92</td><td>0.96</td><td>0.98</td><td>0.32</td><td>0.05</td></tr><tr><td>Test</td><td>5.34</td><td>1.63</td><td>24.55</td><td>27.82</td><td>0.89</td><td>0.90</td><td>0.44</td><td>0.16</td></tr><tr><td rowspan="2">Gibson-Ribera</td><td>Train</td><td>2.11</td><td>0.90</td><td>31.53</td><td>40.19</td><td>0.96</td><td>0.98</td><td>0.31</td><td>0.07</td></tr><tr><td>Test</td><td>13.90</td><td>1.85</td><td>26.51</td><td>30.86</td><td>0.88</td><td>0.91</td><td>0.42</td><td>0.17</td></tr><tr><td rowspan="2">MP3D-pLe4w</td><td>Train</td><td>6.27</td><td>1.54</td><td>27.60</td><td>33.81</td><td>0.91</td><td>0.94</td><td>0.32</td><td>0.13</td></tr><tr><td>Test</td><td>40.39</td><td>10.67</td><td>22.03</td><td>25.17</td><td>0.76</td><td>0.77</td><td>0.50</td><td>0.28</td></tr><tr><td rowspan="2">MP3D-gZ6f7</td><td>Train</td><td>5.31</td><td>0.93</td><td>24.62</td><td>33.82</td><td>0.91</td><td>0.96</td><td>0.31</td><td>0.09</td></tr><tr><td>Test</td><td>11.35</td><td>2.91</td><td>21.32</td><td>23.97</td><td>0.80</td><td>0.78</td><td>0.47</td><td>0.26</td></tr></table>

TABLE VIII

EXPERIMENTAL RESULTS ON THE GIBSON DATASET, INCLUDING COMPLETION RATIO (%), COMPLETION (cm). BOLD AND UNDERLINED INDICATE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.
<table><tr><td></td><td>Metric</td><td>Denmark</td><td>Elmira</td><td>Eudora</td><td>Greigs.</td><td>Pablo</td><td>Ribera</td><td>Cantwell</td><td>Eastvi.</td><td>Swormv.</td><td>Avg.</td></tr><tr><td rowspan="3">Random</td><td>%â</td><td>92.72Â±0.43</td><td>90.77Â±1.90</td><td>89.52Â±3.34</td><td>89.42Â±2.26</td><td>71.67Â±1.71</td><td>85.82Â±1.36</td><td>77.31Â±5.13</td><td>75.74Â±4.50</td><td>78.16Â±4.70</td><td>83.46</td></tr><tr><td>cmâ</td><td>2.01Â±0.19</td><td>2.64Â±0.45</td><td>2.67Â±0.81</td><td>3.39Â±0.75</td><td>13.99Â±3.02</td><td>4.48Â±0.67</td><td>9.31Â±2.84</td><td>9.34Â±2.95</td><td>7.52Â±2.89</td><td>6.15</td></tr><tr><td>%â</td><td>93.91Â±0.67</td><td>92.25Â±0.98</td><td>92.80Â±0.25</td><td>98.96Â±0.02</td><td>80.35Â±1.07</td><td>86.85Â±0.03</td><td>90.92Â±0.64</td><td>83.48Â±5.90</td><td>90.62Â±1.36</td><td>90.02</td></tr><tr><td rowspan="3">Position</td><td>cmâ</td><td>1.62Â±0.16</td><td>2.31Â±0.16</td><td>1.74Â±0.04</td><td>0.60Â±0.00</td><td>5.55Â±0.78</td><td>5.32Â±0.03</td><td>2.24Â±0.11</td><td>5.93Â±2.68</td><td>1.97Â±0.31</td><td>3.03</td></tr><tr><td>%â</td><td>95.19Â±0.04</td><td>94.07Â±2.21</td><td>94.17Â±0.05</td><td>98.33Â±0.02</td><td>79.89Â±0.08</td><td>94.62Â±0.30</td><td>86.33Â±0.11</td><td>83.33Â±3.25</td><td>90.53Â±0.95</td><td>90.72</td></tr><tr><td>cmâ</td><td>1.35Â±0.01</td><td>1.85Â±0.67</td><td>1.39Â±0.05</td><td>0.74Â±0.00</td><td>6.05Â±0.01</td><td>1.26Â±0.06</td><td>4.80Â±0.03</td><td>5.93Â±1.61</td><td>2.07Â±0.27</td><td>2.83</td></tr><tr><td>Viewpoint</td><td>%â</td><td>94.63Â±0.28</td><td>97.70Â±0.16</td><td>93.83Â±0.03</td><td>94.55Â±0.15</td><td>85.83Â±1.91</td><td>96.42Â±0.10</td><td>86.99Â±0.56</td><td>87.12Â±4.09</td><td>92.04Â±0.84</td><td>92.12</td></tr><tr><td>Ours</td><td>cmâ</td><td>1.57Â±0.11</td><td>0.86Â±0.03</td><td>1.50Â±0.01</td><td>1.60Â±0.02</td><td>3.47Â±0.45</td><td>0.96Â±0.02</td><td>4.03Â±0.98</td><td>4.25Â±1.91</td><td>1.90Â±0.23</td><td>2.24</td></tr></table>

TABLE IX

EXPERIMENTAL RESULTS ON THE MP3D DATASET, INCLUDING COMPLETION RATIO (%), COMPLETION (cm). BOLD AND UNDERLINED INDICATE THE BEST AND SECOND-BEST RESULTS, RESPECTIVELY.
<table><tr><td></td><td>Metric</td><td>gZ6f7yhEvPG</td><td>pLe4wQe7qrG</td><td>GdvgFV5R1Z5</td><td>YmJkqBEsHnH</td><td>Avg.</td></tr><tr><td rowspan="3">Random</td><td>%â</td><td>90.29Â±0.63</td><td>85.78Â±4.95</td><td>85.76Â±3.77</td><td>79.97Â±3.66</td><td>85.45</td></tr><tr><td>cmâ</td><td>3.27Â±0.46</td><td>5.09Â±1.66</td><td>5.15Â±1.05</td><td>6.20Â±1.69</td><td>4.93</td></tr><tr><td>%â</td><td>90.99Â±0.58</td><td>93.96Â±0.75</td><td>90.56Â±0.10</td><td>86.68Â±0.50</td><td>90.55</td></tr><tr><td>Position</td><td>cmâ</td><td>3.44Â±0.28</td><td>2.23Â±0.17</td><td>3.98Â±0.03</td><td>3.70Â±0.18</td><td>3.34</td></tr><tr><td rowspan="3">Viewpoint</td><td>%â</td><td>96.35Â±0.71</td><td>96.95Â±0.24</td><td>91.85Â±0.08</td><td>82.00Â±1.37</td><td>91.79</td></tr><tr><td>cmâ</td><td>1.49Â±0.06</td><td>1.47Â±0.04</td><td>3.81Â±0.02</td><td>4.88Â±0.39</td><td>2.92</td></tr><tr><td>%â</td><td>95.91Â±0.30</td><td>97.09Â±0.41</td><td>91.78Â±0.06</td><td>83.91Â±0.48</td><td>92.17</td></tr><tr><td>Ours</td><td>cmâ</td><td>1.61Â±0.04</td><td>1.44Â±0.07</td><td>3.81Â±0.02</td><td>4.39Â±0.16</td><td>2.82</td></tr></table>

<!-- image-->

<!-- image-->  
2D invisibility: 13136.68 3D volume: 0.92

<!-- image-->  
2D inisibility: 11228.20 3D volume: 0.99

<!-- image-->  
2D inisibility: 5016.95 3D volume: 1.68

Fig. 8. The rendered panoptic images and the opacity maps for the node scoring.  
<!-- image-->  
Fig. 9. Real-world experimental results in an office and a meeting room.