# ActiveGS: Active Scene Reconstruction Using Gaussian Splatting

Liren Jin1, Xingguang Zhong1, Yue Pan1, Jens Behley1, Cyrill Stachniss1,3 and Marija PopovicÂ´2

Mission Time

<!-- image-->

<!-- image-->  
Fig 1: Our approach actively reconstructs an unknown scene. We illustrate the reconstruction progress over mission time, displaying planned camera viewpoints (green pyramids) and paths (red lines). We present examples of RGB and confidence maps (redder colour indicates lower confidence) rendered at the same viewpoint at different mission times (magenta and cyan arrows) and at two distinct viewpoints at the same mission time (cyan and blue arrows). By integrating confidence modelling into the Gaussian splatting pipeline, our approach enables targeted view planning to build a high-fidelity Gaussian splatting map. The complete camera path and final reconstruction results, including RGB rendering and surface mesh, are visualised on the right.

AbstractâRobotics applications often rely on scene reconstructions to enable downstream tasks. In this work, we tackle the challenge of actively building an accurate map of an unknown scene using an RGB-D camera on a mobile platform. We propose a hybrid map representation that combines a Gaussian splatting map with a coarse voxel map, leveraging the strengths of both representations: the high-fidelity scene reconstruction capabilities of Gaussian splatting and the spatial modelling strengths of the voxel map. At the core of our framework is an effective confidence modelling technique for the Gaussian splatting map to identify under-reconstructed areas, while utilising spatial information from the voxel map to target unexplored areas and assist in collision-free path planning. By actively collecting scene information in under-reconstructed and unexplored areas for map updates, our approach achieves superior Gaussian splatting reconstruction results compared to state-of-the-art approaches. Additionally, we demonstrate the real-world applicability of our framework using an unmanned aerial vehicle.

## Index TermsâMapping; RGB-D Perception

## I. INTRODUCTION

A CTIVE exploration and reconstruction of unknownscenes are relevant capabilities for developing fully scenes are relevant capabilities for developing fully autonomous robots [3]. For scenarios, including search and rescue, agricultural robotics, and industrial inspection, online active reconstruction using mobile robots demands both mission efficiency and reconstruction quality. Achieving this requires two key components: high-fidelity map representations for modelling fine-grained geometric and textural details of the scenes and adaptive view planning strategies for effective sensor data acquisition.

In this work, we tackle the problem of actively reconstructing unknown scenes using posed RGB-D camera data. Given a limited budget, e.g. mission time, our goal is to obtain an accurate 3D representation of the scene by actively positioning the robot with its camera online during a mission. Existing active scene reconstruction frameworks mainly rely on conventional map representations such as voxel grids, meshes, or point clouds [2, 17, 30, 31, 32, 39, 40]. However, these methods often do not deliver high-fidelity reconstruction results due to their sparse representations. Recent advances in implicit neural representations, e.g. neural radiance fields (NeRFs) [21], have attracted significant research interest for their accurate dense scene reconstruction capabilities and low memory footprints. In the context of active reconstruction, several emerging works [5, 13, 14, 25, 27, 37] incorporate uncertainty estimation in NeRFs and exploit it to guide view planning. While these approaches demonstrate promising results, the rather costly volumetric rendering procedure during online incremental mapping poses limitations for NeRF-based active scene reconstruction.

Dense reconstruction using Gaussian splatting (GS) [16] offers a promising alternative to NeRFs, addressing rendering inefficiencies while preserving representation capabilities. GS explicitly models scene properties through Gaussian primitives and utilises efficient differentiable rasterisation to achieve dense reconstruction. Its fast map updates and explicit structure make it well-suited for online incremental mapping. Building on these strengths, we adopt GS for active scene reconstruction in this work.

Incorporating GS into an active scene reconstruction pipeline presents significant challenges. First, active reconstruction often requires evaluating the reconstruction quality to guide view planning. However, this is difficult without ground truth information at novel viewpoints. Second, the Gaussian primitives represent only occupied space, making it hard to distinguish between unknown and free space, which are important for exploration and path planning.

The key contribution of this work is addressing these challenges through our novel GS-based active scene reconstruction framework, denoted as ActiveGS. To tackle the first challenge, we propose a simple yet effective confidence modelling technique for Gaussian primitives based on viewpoint distribution, enabling view planning for inspecting under-reconstructed surfaces. For the second challenge, we combine the GS map with a conventional coarse voxel map, exploiting the strong representation capabilities of GS for scene reconstruction with the spatial modelling strengths of voxel maps for exploration and path planning.

We make the following three claims: (i) our ActiveGS framework achieves superior reconstruction performance compared to state-of-the-art NeRF-based approach and GS-based baselines; (ii) our explicit confidence modelling for Gaussian primitives enables informative viewpoint evaluation and targeted inspection around under-reconstructed surfaces, further improving mission efficiency and reconstruction quality; (iii) we validate our approach in different synthetic indoor scenes and in a real-world scenario using an unmanned aerial vehicle (UAV). To support reproducibility and future research, we open-source our implementation code at: https://github.com/dmar-bonn/active-gs.

## II. RELATED WORK

Our work uses Gaussian splatting as the map representation for active scene reconstruction. In this section, we overview state-of-the-art high-fidelity map representations, focusing on GS and active scene reconstruction methods.

## A. Gaussian Splatting as Map Representation

Conventional map representations such as voxel grids [9], meshes [1, 34], and point clouds [39] often only capture coarse scene structures, struggling to provide fine-grained geometric and textural information crucial for many robotics applications [3]. Recent advances in implicit neural representations, such as NeRFs [21, 22], show promising results in highfidelity scene reconstruction by modelling scene attributes continuously. Although they achieve impressive reconstruction results, NeRFs require dense sampling along rays for view synthesis, a computationally intensive process that limits their online applicability.

3D GS [16] offers an efficient alternative for high-fidelity scene reconstruction by combing explicit map structures with volumetric rendering. Unlike NeRFs, 3D GS stores scene information using explicit 3D Gaussian primitives, eliminating the need for inefficient dense sampling during volumetric rendering. This explicit nature also makes it well-suited for online incremental mapping, which requires frequent data fusion and scene attributes modification. Follow-up works further enhance geometric quality by regularising 3D GS training [7] or directly adopting 2D GS for improved surface alignment [4, 10]. 2D GS collapses the 3D primitive volume into 2D oriented planar Gaussian primitives, enabling more accurate depth estimation and allowing the integration of normal information into the optimisation process. Motivated by its strong performance, we utilise 2D GS, specifically Gaussian surfel [4], as our GS map representation for active scene reconstruction.

## B. Active Scene Reconstruction

Active scene reconstruction using autonomous mobile robots is an area of active research [3]. Given an unknown scene, the goal is to explore and map the scene by actively planning the robotâs next viewpoints for effective data acquisition. Existing active scene reconstructions utilise map representations such as voxel maps [2, 11, 17, 30, 40], meshes [31, 32], or point clouds [6, 39]. These approaches primarily focus on fully covering the unknown space, rather than achieving high-fidelity reconstruction. However, highfidelity scene reconstruction is crucial for downstream robotic tasks that rely on accurate map information.

To address this, recent research explores implicit neural representations for active reconstruction applications. In an object-centric setup, several methods incorporate uncertainty estimation into NeRFs [13, 14, 25, 27, 37] and use this information to select next best viewpoints. For scene-level reconstruction, Yan et al. [38] and Kuang et al. [18] investigate the loss landscape of implicit neural representations during training to identify under-reconstructed areas. NARUTO [5] learns an uncertainty grid map alongside a hybrid neural scene representation, guiding data acquisition in uncertain regions. These implicit neural representations often face challenges such as inefficient map updates and catastrophic forgetting during incremental mapping.

Several concurrent works propose using GS for active scene reconstruction. GS-Planner [15] and HGS-planner [35] incorporate unknown voxels into the GS rendering pipeline and detect unseen regions for exploration. Li et al. [19] use a Voronoi graph to extract a traversable topological map from the GS representation for path planning. The approach is designed for a 2D planning space, reducing its effectiveness in cluttered environments. FisherRF [12] evaluates the information content of a novel view by measuring the Fisher information value in the GS parameters. This procedure requires computationally expensive gradient calculations at each previously visited and candidate viewpoint, making view planning inefficient for online missions. We build upon the idea of using GS for active scene reconstruction, while introducing a key innovation: we explicitly model the confidence of each Gaussian primitive, enabling viewpoint sampling around low-confidence Gaussian primitives for targeted inspection and fast feed-forward confidence rendering for efficient viewpoint evaluation.

<!-- image-->  
Fig 2: An overview of our framework. Our hybrid map representation consists of a 2D GS map for high-fidelity scene reconstruction with a coarse voxel map for exploration and path planning. Our view planner leverages unexplored regions in the voxel map for exploration and low-confidence Gaussian primitives for targeted inspection, collecting informative measurements at planned viewpoints for map updates. We iterate between the map update and view planning steps until the pre-allocated mission time is reached.

## III. OUR APPROACH

We introduce ActiveGS, a novel framework for active scene reconstruction using GS for autonomous robotic tasks. An overview of our framework is shown in Fig. 2. Our goal is to reconstruct an unknown scene using a mobile robot, e.g. a UAV, equipped with an on-board RGB-D camera. Given posed RGB-D measurements as input, we update a coarse voxel map to model the spatial occupancy and incrementally train a GS map for high-fidelity scene reconstruction. To actively guide view planning to reconstruct the scene in a targeted manner, we propose using our confidence modelling technique in the GS map and information about unexplored regions in the voxel map as the basis for planning. The framework alternates between mapping and planning steps until a predefined mission time is reached.

## A. Hybrid Map Representation

Given the bounding box of the scene to be reconstructed, we uniformly divide the enclosed space into voxels, forming our voxel map V, where each voxel $v _ { i } ~ \in ~ \mathcal { V }$ represents the volume occupancy probability in the range [0 , 1].

Our GS map is based on Gaussian surfel [4], a state-ofthe-art 2D GS representation. The GS map G comprises a collection of Gaussian primitives. Each primitive $\mathbf { \Psi } _ { \mathbf { { \mathcal { g } } } _ { i } } \in { \mathcal { G } }$ is defined by its parameters $g _ { i } = ( x _ { i } , q _ { i } , s _ { i } , c _ { i } , o _ { i } , k _ { i } )$ , where $\pmb { x } _ { i } \in \mathbb { R } ^ { 3 }$ denotes the position of the primitive centre; $q _ { i } \in \mathbb { R } ^ { 4 }$ is its rotation in the form of a quaternion; $\pmb { s } _ { i } = [ s _ { i } ^ { x } , s _ { i } ^ { y } ] \in \mathbb { R } _ { + } ^ { 2 }$ are the scaling factors along the two axes of the primitive; $\boldsymbol { c } _ { i } \in [ 0 , 1 ] ^ { 3 }$ represents the RGB colour; $o _ { i } ~ \in ~ [ 0 , 1 ]$ is the opacity; and $k _ { i } \in \mathbb { R } _ { + }$ is the confidence score introduced later in Sec. III-C. The distribution of the Gaussian primitive $\mathbf { \vec { \mathbf { g } } } _ { i }$ in world coordinate is represented as:

$$
\mathcal { N } ( \pmb { x } ; \pmb { x } _ { i } , \pmb { \Sigma } _ { i } ) = \exp \left( - \frac { 1 } { 2 } ( \pmb { x } - \pmb { x } _ { i } ) ^ { \top } \pmb { \Sigma } _ { i } ^ { - 1 } ( \pmb { x } - \pmb { x } _ { i } ) \right) ,\tag{1}
$$

where $\pmb { \Sigma } _ { i } = \mathbf { R } ( \pmb q _ { i } )$ diag $\left( ( s _ { i } ^ { x } ) ^ { 2 } , ( s _ { i } ^ { y } ) ^ { 2 } , 0 \right) { \bf R } ( { \pmb q } _ { i } ) ^ { \top }$ is the covariance matrix, with $\mathbf { R } ( q _ { i } ) \in S O ( 3 )$ as the rotation matrix derived from the corresponding quaternion $\pmb q _ { i }$ and diag( Â· ) indicating a diagonal matrix with the specified diagonal elements. The normal of the Gaussian primitive can be directly obtained from the last column of the rotation matrix as ${ \pmb n } _ { i } = { \bf R } ( { \pmb q } _ { i } ) _ { : , 3 } .$

Given the GS map, we can render the colour map I, depth map D, normal map N, opacity map O, and confidence map K at a viewpoint using the differentiable rasterisation pipeline [4]. Without loss of generality, the rendering function for a pixel u on the view is formulated as:

$$
\mathbf { O } ( \pmb { u } ) = \sum _ { i = 1 } ^ { n } w _ { i } , \mathbf { M } ( \pmb { u } ) = \sum _ { i = 1 } ^ { n } w _ { i } m _ { i } ,\tag{2}
$$

$$
w _ { i } = T _ { i } \alpha _ { i } , T _ { i } = \prod _ { j < i } ( 1 - \alpha _ { j } ) , \ L \alpha _ { i } = \mathcal { N } ( \boldsymbol { u } ; \boldsymbol { u } _ { i } , \Sigma _ { i } ^ { \prime } ) o _ { i } ,\tag{3}
$$

where $\mathbf { M } \in \{ \mathbf { I } , \mathbf { D } , \mathbf { N } , \mathbf { K } \}$ and $m _ { i } \in \{ c _ { i } , d _ { i } , { \pmb n } _ { i } , k _ { i } \}$ is the corresponding modality feature, with $d _ { i }$ being the distance from the viewpoint centre to the intersection point of the camera ray and the Gaussian primitive ${ \bf \nabla } _ { { \bf { \mathit { g } } } _ { i } ; }$ wi indicates the rendering contribution of $\mathbf { \pmb { g } } _ { i }$ to pixel u; and $\Sigma _ { i } ^ { \prime }$ and $\mathbf { \delta } \mathbf { u } _ { i }$ are the primitiveâs covariance matrix and centre projected onto the image space [41]; For more technical details about the rendering process, please refer to Gaussian surfel [4].

## B. Incremental Mapping

We collect measurements captured at planned viewpoints and incrementally update our map representation. Given the current RGB image Iâ and depth image Dâ measurements, we generate a per-pixel point cloud using known camera parameters. We then update our voxel map V probabilistically based on the new point cloud, following OctopMap [9].

For GS map update, we first add Gaussian primitives to G where needed. To this end, we render the colour map I, depth map D, and opacity map O at the current camera viewpoint. We calculate a binary mask to identify the pixels that should be considered for densifying the GS map:

$$
\begin{array} { c } { { { \bf B } ( { \pmb u } ) = ( { \bf O } ( { \pmb u } ) < 0 . 5 ) \vee ( \arg ( | { \bf I } ( { \pmb u } ) - { \bf I } ^ { \star } ( { \pmb u } ) | ) > 0 . 5 ) } } \\ { { \vee ( { \bf D } ( { \pmb u } ) - { \bf D } ^ { \star } ( { \pmb u } ) > \lambda { \bf D } ^ { \star } ( { \pmb u } ) ) , } } \end{array}\tag{4}
$$

where $\mathrm { a v g } ( \cdot )$ is the channel-wise averaging operation to calculate per-pixel colour error and Î» is a constant accounting for depth sensing noise, set to 0.05 in our pipeline. This mask indicates areas where opacity is low, colour rendering is inaccurate, or new geometry appears in front of the current depth estimate, signalling the need for new Gaussian primitives. We spawn new Gaussian primitives by unprojecting pixels on these areas into 3D space, with initial parameters defined by the corresponding point cloud position, pixel colour, and normal estimated by applying central differencing on the bilateralfiltered depth image [23], which helps mitigate noise contained in the depth sensing. We also set scale values to 1 cm, opacity value to 0.5, and confidence value to 0.

At each mapping step, we train our GS map $\mathcal { G }$ using all collected RGB-D measurements for 10 iterations. Specifically, for each iteration, we select the 3 most recent frames and 5 random frames from the measurement history. The loss for a frame {ËI , DË } in the training batch is formulated as the weighted sum of individual loss terms:

$$
\mathcal { L } = w _ { c } \mathcal { L } _ { c } + w _ { d } \mathcal { L } _ { d } + w _ { n } \mathcal { L } _ { n } ,\tag{5}
$$

where the photometric loss $\mathcal { L } _ { c } = L _ { 1 } ( \mathbf { I } , \hat { \mathbf { I } } )$ and the depth loss ${ \mathcal { L } } _ { d } = L _ { 1 } ( \mathbf { D } , { \hat { \mathbf { D } } } )$ are both calculated using the $L _ { 1 }$ distance. The normal loss $\mathcal { L } _ { n } = D _ { c o s } ( \mathbf { N } , \tilde { \mathbf { N } } ) + T V ( \mathbf { N } )$ consists of the cosine distance $D _ { c o s }$ between the rendered normal map and the normal map N derived from the rendered depth map [4], along with the total variation $T V$ loss [28] to enforce smooth normal rendering between neighbouring pixels. Note that the training process involves only a subset of the Gaussian primitive parameters $( { \pmb x } _ { i } , { \pmb q } _ { i } , { \pmb s } _ { i } , { \pmb c } _ { i } , o _ { i } )$ , while the modelling of non-trainable $k _ { i }$ is introduced in Sec. III-C.

After every 5 mapping steps, we perform a visibility check on all Gaussian primitives and delete those invisible to all history views to compact the GS map. We consider a Gaussian primitive visible from a viewpoint if at least one pixel rendered in that view receives its rendering contribution greater than a threshold, as defined in Eq. (3). Unlike previous works utilising density control during offline training [4, 10, 16], our approach adds necessary primitives and removes invisible ones during online missions, achieving computationally efficient scene reconstruction.

## C. Confidence Modelling for Gaussian Primitives

A Gaussian primitive can be effectively optimised if observed from different viewpoints. Based on this insight, we derive the confidence of a Gaussian primitive from the spatial distribution of viewpoints in the measurement history. Specifically, we connect the Gaussian centre $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ to the viewpoint centre $\scriptstyle { \pmb { x _ { p _ { j } } } }$ , denoted as ${ \pmb d } _ { i j } = { \pmb x } _ { { \pmb p } _ { j } } - { \pmb x } _ { i } = d _ { i j } { \pmb v } _ { i j }$ , where $d _ { i j }$ is the distance and $\mathbf { \boldsymbol { v } } _ { i j }$ is the normalised view direction, with $j \in \mathcal { S } ( g _ { i } )$ and S being the index set of viewpoints from which the Gaussian primitive $\mathbf { \pmb { g } } _ { i }$ has been observed. The confidence $k _ { i }$ is finally formulated as:

$$
k _ { i } = \gamma _ { i } \exp ( \beta _ { i } ) ,\tag{6}
$$

$$
\gamma _ { i } = \sum _ { j \in S ( \pmb { g } _ { i } ) } \left( 1 - \frac { d _ { i j } } { d _ { \mathrm { f a r } } } \right) \pmb { n } _ { i } \cdot \pmb { v } _ { i j } ,\tag{7}
$$

$$
\beta _ { i } = 1 - \left\| \pmb { \mu } _ { i } \right\| , \pmb { \mu } _ { i } = \frac { 1 } { \vert \pmb { S } ( \pmb { g } _ { i } ) \vert } \sum _ { j \in \pmb { S } ( \pmb { g } _ { i } ) } \pmb { v } _ { i j } ,\tag{8}
$$

where $\gamma _ { i }$ accounts for distance-weighted cosine similarity between the Gaussian primitiveâs normal $\mathbf { n } _ { i }$ and view direction $\mathbf { } v _ { i j }$ , with $d _ { \mathrm { f a r } }$ as the maximum depth sensing range. Note that we increase the impact of viewpoints that are closer to the Gaussian primitiveâs centre or provide view directions similar to the primitiveâs normal. $\beta _ { i }$ measures the dispersion of directions from which $\mathbf { \pmb { g } } _ { i }$ is observed, with $\beta _ { i }$ closer to 0 indicating similar view directions. Our confidence formulation assigns higher confidence to Gaussian primitives densely observed from viewpoints with varying angles, whereas lower confidence to those with sparse and similar observations.

## D. Viewpoint Utility Formulation

Active scene reconstruction requires both exploration, to cover unknown areas, and exploitation, to closely inspect under-reconstructed surfaces. In this work, we combine utility derived from the voxel map for exploration and the GS map for exploitation, enabling these behaviours effectively.

A candidate viewpoint $\pmb { p } _ { i } ^ { c } \in \mathbb { R } ^ { 5 }$ is defined by its 3D position, yaw, and pitch angles in our framework. To simplify path planning, we constrain the positions to a discrete lattice placed at the centres of all free voxels. We follow existing active scene exploration frameworks [2, 11, 24, 30, 40] and define the exploration utility based on the number of unexplored voxels visible from a candidate viewpoint. Without relying on timeconsuming ray-casting operations, our framework leverages efficient rendering of our GS map to identify visible voxels. We achieve this by checking whether the projected depth of the in-view voxel centres in the camera coordinate is smaller than the corresponding depth value in the rendered depth from the GS map.

Combining unexplored region information in the voxel map and confidence rendering from the GS map, we define the utility of a candidate viewpoint $\mathbf { \Delta } _ { \pmb { p } _ { i } ^ { c } }$ as:

$$
U _ { \mathrm { v i e w } } ( \pmb { p } _ { i } ^ { c } ) = \phi U _ { \mathscr { V } } ( \pmb { p } _ { i } ^ { c } ) + U _ { \mathscr { G } } ( \pmb { p } _ { i } ^ { c } ) ,\tag{9}
$$

where $\begin{array} { r } { U _ { \mathcal { V } } ( \pmb { p } _ { i } ^ { c } ) ~ = ~ \frac { N _ { u } ( \pmb { p } _ { i } ^ { c } ) } { | \mathcal { V } | } } \end{array}$ is the exploration utility, defined by the ratio of the number of visible unexplored voxels $N _ { u } ( \pmb { p } _ { i } ^ { c } )$ to the total number of voxels in the voxel map; $U _ { \mathcal { G } } ( \pmb { p } _ { i } ^ { c } ) = - \mathrm { m e a n } ( \mathbf { K } _ { i } )$ is the exploitation utility, calculated as the negative mean of the confidence map Ki rendered at $\mathbf { \Delta } _ { \pmb { p } _ { i } ^ { c } }$ following Eq. (2); and Ï is the exploration weight.

## E. Viewpoint Sampling and Evaluation

Our viewpoint sampling strategy involves two types of candidate viewpoints. First, we randomly sample $N _ { \mathrm { r } }$ andom candidate viewpoints within a specified range around the current viewpoint. However, relying solely on random local sampling can lead to local minima. To address this, we introduce additional candidate viewpoints based on regions of interest (ROI) defined in the voxel map. We begin by identifying frontier voxels [36] and add them to our ROI set R. By explicitly modelling the confidence of each Gaussian primitive, we can identify and also include voxels containing low-confidence Gaussian primitives in R. Inspired by previous work [17], we define normals for each voxel in R to indicate the most informative viewing direction. For voxels with lowconfidence Gaussian primitives, this is simply the average normal of these Gaussian primitives. The normal of frontier voxels is determined by finding their neighbouring free voxels and calculating the average directional vector from the frontier voxel to these neighbours. To generate ROI-based candidate viewpoints, we create a fixed number of candidate viewpoints within the cone defined by the minimum and maximum sampling distances to each ROIâs centre, and the maximum angular difference relative to its normal. Starting from the closest ROI voxel, we continue outward until we have collected up to $N _ { \mathrm { R O I } }$ viewpoints in free space. We illustrate the sampling process in Fig. 3.

<!-- image-->  
Fig 3: We show a 2D case of our ROI-based candidate viewpoint generation. We identify ROI voxels by retrieving voxels containing low-confidence Gaussian primitives and frontier voxels. Normals for low-confidence voxels are generated by averaging the normals of low-confidence primitives, while frontier voxel normals are calculated using average vectors to neighbouring free voxels. Given the voxel centres and directional normals, we generate candidate viewpoints within the sampling region, as illustrated on the right.

We evaluate the utility of all candidate viewpoints following Eq. (9). We use the Aâ algorithm [8] to find the shortest traversable path from the current viewpoint position to all candidate viewpoint positions. Taking travel distance into account, we select the next best viewpoint $p ^ { \star }$ by:

$$
p ^ { \star } = \underset { p _ { i } ^ { c } } { \arg \operatorname* { m a x } } \left( \frac { U _ { \mathrm { v i e w } } ( \pmb { p } _ { i } ^ { c } ) } { \sum _ { i = 1 } ^ { N _ { \mathrm { t e u l } } } U _ { \mathrm { v i e w } } ( \pmb { p } _ { i } ^ { c } ) } - \delta \frac { U _ { \mathrm { p a t h } } ( \pmb { p } _ { i } ^ { c } ) } { \sum _ { i = 1 } ^ { N _ { \mathrm { t e u l } } } U _ { \mathrm { p a t h } } ( \pmb { p } _ { i } ^ { c } ) } \right) .\tag{10}
$$

where $N _ { \mathrm { t o t a l } } = N _ { \mathrm { r a n d o m } } + N _ { \mathrm { R O I } } ; U _ { \mathrm { p a t h } }$ is the travel distance to the candidate viewpoint positions; and Î´ is a weighting factor for the travel cost.

## IV. EXPERIMENTAL EVALUATION

Our experimental results support our three claims: (i) we show that our ActiveGS framework outperforms state-of-theart NeRF-based and GS-based active scene reconstruction methods; (ii) we show that our confidence modelling of Gaussian primitives enables informative viewpoint evaluation and targeted candidate viewpoint generation, improving reconstruction performance; (iii) we validate our framework in different simulation scenes and in a real-world scenario to show its applicability.

## A. Implementation Details

Mapping. We use a voxel size of $2 0 \mathrm { c m } \times 2 0 \mathrm { c m } \times 2 0$ cm for the voxel map. We set the loss weights in Eq. (5) as: $w _ { c } = 1 . 0 , w _ { d } = 0 . 8 .$ and $w _ { n } = 0 . 1$ . For visibility checks, a minimum rendering contribution threshold of 0.3 is applied.

Planning. We set the exploration weight $\begin{array} { l l l } { \phi } & { = } & { 1 0 0 0 } \end{array}$ in Eq. (9) to encourage exploratory behaviour during the initial phase of an online mission. We set $\delta \ : = \ : 0 . 5$ for weighting travel costs in Eq. (10). We consider $N _ { \mathrm { t o t a l } } = 1 0 0$ candidate viewpoints, including up to $N _ { \mathrm { R O I } } = 3 0$ ROI-based samples and $N _ { \mathrm { r a n d o m } } = N _ { \mathrm { t o t a l } } - N _ { \mathrm { R O I } }$ random samples generated within 0.5 m of the current viewpoint position.

We test our implementation on a desktop PC with an Intel Core i9-10940X CPU and an NVIDIA RTX A5000 GPU. In this setup, one mapping and planning steps take approximately 1 s and 0.5 s, respectively. The whole framework consumes 4 â 5 GB GPU RAM during an online mission, with approximately 10% allocated to the voxel map update.

## B. Simulation Experiments

Setup. We conduct our simulation experiments using the Habitat simulator [29] and the Replica dataset [33]. The experiments utilise an RGB-D camera with a field of view of [60â¦, 60â¦] and a resolution of [512, 512] pixels. The camera has a depth sensing range of [0.1, 5.0] m and Gaussian noise in the depth measurements with linearly increased standard deviation $\sigma = 0 . 0 1 d ,$ where d represents the depth value.

Evaluation Metrics. We report the reconstruction performance over total mission time, defined as the summation of mapping time, planning time, and action time, assuming a constant robot velocity of 1 m/s. The reconstruction performance is evaluated on both rendering and mesh quality. For the rendering evaluation, we generate ground truth RGB images captured from 1000 uniformly distributed test viewpoints in the sceneâs free space. We report PSNR [21] of RGB images rendered from our GS map at test viewpoints as the rendering quality metric. For the mesh evaluation, we run TSDF fusion [23] on depth images rendered at training viewpoints and extract the scene mesh using Marching Cubes [20]. We use the completeness ratio [5] as the mesh quality metric with a distance threshold set to 2 cm.

We consider the following methods:

â¢ Ours: Our full ActiveGS framework utilising both exploration and exploitation utility measures. We consider ROI-based sampling to achieve targeted candidate viewpoint generation as described in Sec. III-E.

â¢ Ours (w/o ROI): A variant of our ActiveGS that leverages only local random sampling, with $N _ { \mathrm { R O I } } = 0$

â¢ Oursâ : A variant of our ActiveGS with an alternative confidence formulation, assigning higher confidence to Gaussian primitives with more visible viewpoints, without considering their spatial distribution.

â¢ FBE [36]: Frontier-based exploration framework that solely focuses on covering unexplored regions, without accounting for the quality of the GS map. We use the collected RGB-D data to update the GS map, similar to our framework.

<!-- image-->  
Fig 4: We report the reconstruction performance evaluated in rendering and mesh quality over online mission time. Our ActiveGS outperforms baselines in all test scenes. Our view planner considers unexplored regions for exploration, while exploiting low-confidence Gaussian primitives for further inspection. Compared to GS-based approaches, our approach proposes explicit confidence modelling of Gaussian primitives, enabling targeted candidate viewpoint generation and fast viewpoint evaluation. Our approaches demonstrate a large performance gain to the state-of-the-art NeRF-based approach, NARUTO, motivating the use of GS in active scene reconstruction.

â¢ FisherRF [12]: GS-based active scene reconstruction using frontier voxels for ROI-based candidate viewpoint generation and Fisher information for viewpoint evaluation. We replace its 3D GS map with our 2D GS.

â¢ NARUTO [5]: A state-of-the-art NeRF-based active scene reconstruction pipeline.

We run 5 trials for all methods across 8 test scenes. We set the maximum mission time to 300 s and evaluate reconstruction performance every 60 s. We report the mean and standard deviation for PSNR and completeness ratio.

We present the results of simulation experiments in Fig. 4. Our approach achieves the best performance in both rendering and mesh quality across all test scenes, supporting our first claim that it outperforms state-of-the-art NeRF and GSbased methods. The NeRF-based active scene reconstruction framework, NARUTO, exhibits a significant performance gap compared to our approach, particularly in RGB rendering. This disparity arises because NeRF-based methods often compromise model capacity for faster map updates, limiting their representation quality in scene-level reconstruction. FisherRF evaluates viewpoint utility by calculating the Fisher information in the parameters of the Gaussian primitives within its field of view. This requires computationally expensive gradient calculation for all candidate and training viewpoints, leading to prolonged planning times and incomplete reconstruction under limited mission time. Additionally, since Fisher information is conditioned on the candidate viewpoint, the viewpoint must be selected before its utility can be evaluated, preventing direct viewpoint sampling informed by Fisher information. In contrast, our approach models the confidence of each Gaussian primitive, enabling fast feed-forward confidence rendering for viewpoint evaluation and identification of low-confidence surfaces for targeted candidate viewpoint generation, significantly enhancing reconstruction quality and efficiency. FBE focuses solely on exploration and ignores surface reconstruction quality, limiting its performance. While our approach balances exploration and exploitation by accounting for both unexplored regions and low-confidence Gaussian primitives. The ablation study comparing Ours and Ours (w/o ROI) demonstrates the benefits of ROI-based sampling for targeted inspection, reflected by higher means and smaller standard deviations in both evaluation metrics. Our confidence formulation also outperforms the variant in Oursâ  by considering viewpoint distribution. These results confirm that our confidence modelling is effective in achieving efficient and high-fidelity active scene reconstruction, validating our second claim. We visually compare the reconstruction results in Fig. 5.

<!-- image-->  
Fig 5: Visual comparison of reconstruction results using different approaches. We show RGB rendering and surface meshes for two scenes, with red circles highlighting areas of low-quality reconstruction from baseline approaches. Our ActiveGS considers both unexplored regions in voxel map and confidence value of GS map to enable targeted view planning, achieving complete and high-fidelity scene reconstruction.

## C. Real-World Experiments

We demonstrate the applicability our framework in a realworld experiment using a UAV equipped with an Intel RealSense 455 RGB-D camera to reconstruct a scene of size $6 \mathrm { { m } \times 6 \mathrm { { m } \times 3 \mathrm { { m } } } }$ . Unlike simulation experiments, we do not account for the pitch angle of viewpoints in this experiment due to control limitations. The UAV pose is tracked by an OptiTrack motion capture system. Given the limited on-board resources, we run ActiveGS on our desktop PC, where it receives RGB-D and pose data from the UAV for map updates and sends planned collision-free waypoints to guide the UAV. All communication is handled via ROS [26].

Our real-world experiments indicate that our approach is effective for actively reconstructing unknown scenes by considering both unexplored regions in the voxel map and under-reconstructed surfaces in the GS map. We show the experimental setup in Fig. 6 and the online active scene reconstruction in the supplementary video.

<!-- image-->  
Fig 6: Our real-world experiments using a UAV equipped with an RGB-D camera. We show the experimental setup (left) and the RGB rendering from our GS map (right).

## V. CONCLUSIONS AND FUTURE WORK

In this paper, we propose ActiveGS, a GS-based active scene reconstruction framework. Our approach employs a hybrid map representation, combining the high-fidelity scene reconstruction capabilities of Gaussian splatting with the spatial modelling strengths of the voxel map. We propose an effective method for confidence modelling of Gaussian primitives, enabling targeted viewpoint generation and informative viewpoint evaluation. Our view planning strategy leverages the confidence information of Gaussian primitives to inspect under-reconstructed areas, while also considering unexplored regions in the voxel map for exploration. Experimental results demonstrate that ActiveGS outperforms baseline approaches in both rendering and mesh quality.

A limitation of our current framework is the assumption of perfect localisation. Future work will incorporate localisation uncertainty into confidence modelling of Gaussian primitives. We also plan to better integrate the voxel map with GS map for more efficient mapping and leverage optimisation-based approaches to enhance view planning quality.

## VI. ACKNOWLEDGMENT

We thank Hang Yu, Jakub Plonka, and Moji Shi for their assistance in conducting the real-world experiments.

## REFERENCES

[1] J. Behley and C. Stachniss, âEfficient surfel-based slam using 3d laser range data in urban environments,â in Proc. of Robotics: Science and Systems, 2018.

[2] A. Bircher, M. Kamel, K. Alexis, H. Oleynikova, and R. Siegwart, âReceding horizon ânext-best-viewâ planner for 3d exploration,â in Proc. of the IEEE Intl. Conf. on Robotics & Automation, 2016.

[3] S. Chen, Y. Li, and N. M. Kwok, âActive vision in robotic systems: A survey of recent developments,â Intl. Journal of Robotics Research, vol. 30, no. 11, pp. 1343â1377, 2011.

[4] P. Dai, J. Xu, W. Xie, X. Liu, H. Wang, and W. Xu, âHigh-quality surface reconstruction using gaussian surfels,â in Proc. of the Intl. Conf. on Computer Graphics and Interactive Techniques, 2024.

[5] Z. Feng, H. Zhan, Z. Chen, Q. Yan, X. Xu, C. Cai, B. Li, Q. Zhu, and Y. Xu, âNaruto: Neural active reconstruction from uncertain target observations,â in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2024.

[6] Y. Gao, Y. Wang, X. Zhong, T. Yang, M. Wang, Z. Xu, Y. Wang, Y. Lin, C. Xu, and F. Gao, âMeeting-merging-mission: A multi-robot coordinate framework for large-scale communication-limited exploration,â in Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems, 2022.

[7] A. Guedon and V. Lepetit, âSugar: Surface-aligned gaussian splatting Â´ for efficient 3d mesh reconstruction and high-quality mesh rendering,â in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2024.

[8] P. E. Hart, N. J. Nilsson, and B. Raphael, âA formal basis for the heuristic determination of minimum cost paths,â IEEE Trans. on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100â107, 1968.

[9] A. Hornung, K. M. Wurm, M. Bennewitz, C. Stachniss, and W. Burgard, âOctomap: An efficient probabilistic 3d mapping framework based on octrees,â Autonomous Robots, vol. 34, pp. 189â206, 2013.

[10] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2d gaussian splatting for geometrically accurate radiance fields,â in Proc. of the Intl. Conf. on Computer Graphics and Interactive Techniques, 2024.

[11] S. Isler, R. Sabzevari, J. Delmerico, and D. Scaramuzza, âAn information gain formulation for active volumetric 3d reconstruction,â in Proc. of the IEEE Intl. Conf. on Robotics & Automation, 2016.

[12] W. Jiang, B. Lei, and K. Daniilidis, âFisherrf: Active view selection and mapping with radiance fields using fisher information,â in Proc. of the Europ. Conf. on Computer Vision, 2024.

[13] L. Jin, X. Chen, J. Ruckin, and M. Popovi Â¨ c, âNeu-nbv: Next best view Â´ planning using uncertainty estimation in image-based neural rendering,â in Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems, 2023.

[14] L. Jin, H. Kuang, Y. Pan, C. Stachniss, and M. Popovic, âStair: Semantic-Â´ targeted active implicit reconstruction,â in Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems, 2024.

[15] R. Jin, Y. Gao, H. Lu, and F. Gao, âGs-planner: A gaussian-splattingbased planning framework for active high-fidelity reconstruction,â in Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems, 2024.

[16] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussianÂ¨ splatting for real-time radiance field rendering,â ACM Trans. on Graphics, vol. 42, no. 4, pp. 139â1, 2023.

[17] Y. Kompis, L. Bartolomei, R. Mascaro, L. Teixeira, and M. Chli, âInformed sampling exploration path planner for 3d reconstruction of large scenes,â IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 7893â7900, 2021.

[18] Z. Kuang, Z. Yan, H. Zhao, G. Zhou, and H. Zha, âActive neural mapping at scale,â in Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems, 2024.

[19] Y. Li, Z. Kuang, T. Li, G. Zhou, S. Zhang, and Z. Yan, âActivesplat: High-fidelity scene reconstruction through active gaussian splatting,â arXiv preprint arXiv:2410.21955, 2024.

[20] W. E. Lorensen and H. E. Cline, âMarching cubes: A high resolution 3d surface construction algorithm,â in Seminal Graphics: Pioneering Efforts that Shaped the Field, 1998, pp. 347â353.

[21] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â in Proc. of the Europ. Conf. on Computer Vision, 2020.

[22] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graph- Â¨ ics primitives with a multiresolution hash encoding,â ACM Trans. on Graphics, vol. 41, no. 4, pp. 1â15, 2022.

[23] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, S. Hodges, and A. Fitzgibbon, âKinectFusion: Real-Time Dense Surface Mapping and Tracking,â in Proc. of the Intl. Symp. on Mixed and Augmented Reality, 2011.

[24] E. Palazzolo and C. Stachniss, âEffective Exploration for MAVs Based on the Expected Information Gain,â Drones, vol. 2, no. 1, pp. 59â66, 2018.

[25] X. Pan, Z. Lai, S. Song, and G. Huang, âActivenerf: Learning where to see with uncertainty estimation,â in Proc. of the Europ. Conf. on Computer Vision, 2022.

[26] M. Quigley, K. Conley, B. Gerkey, J. Faust, T. Foote, J. Leibs, R. Wheeler, and A. Y. Ng, âRos: an open-source robot operating system,â in ICRA Workshop on Open Source Software, 2009.

[27] Y. Ran, J. Zeng, S. He, J. Chen, L. Li, Y. Chen, G. Lee, and Q. Ye, âNeurar: Neural uncertainty for autonomous 3d reconstruction with implicit neural representations,â IEEE Robotics and Automation Letters, vol. 8, no. 2, pp. 1125â1132, 2023.

[28] L. Rudin and S. Osher, âTotal variation based image restoration with free local constraints,â in Proc. of the IEEE Intl. Conf. on Image Processing, 1994.

[29] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik et al., âHabitat: A platform for embodied ai research,â in Proc. of the IEEE/CVF Intl. Conf. on Computer Vision, 2019.

[30] L. Schmid, M. Pantic, R. Khanna, L. Ott, R. Siegwart, and J. Nieto, âAn efficient sampling-based method for online informative path planning in unknown environments,â IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 1500â1507, 2020.

[31] S. Song and S. Jo, âOnline inspection path planning for autonomous 3d modeling using a micro-aerial vehicle,â in Proc. of the IEEE Intl. Conf. on Robotics & Automation, 2017.

[32] ââ, âSurface-based exploration for autonomous 3d modeling,â in Proc. of the IEEE Intl. Conf. on Robotics & Automation, 2018.

[33] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, A. Clarkson, M. Yan, B. Budge, Y. Yan, X. Pan, J. Yon, Y. Zou, K. Leon, N. Carter, J. Briales, T. Gillingham, E. Mueggler, L. Pesqueira, M. Savva, D. Batra, H. M. Strasdat, R. D. Nardi, M. Goesele, S. Lovegrove, and R. Newcombe, âThe Replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[34] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison, âElasticfusion: Dense slam without a pose graph.â in Proc. of Robotics: Science and Systems, 2015.

[35] Z. Xu, R. Jin, K. Wu, Y. Zhao, Z. Zhang, J. Zhao, F. Gao, Z. Gan, and W. Ding, âHgs-planner: Hierarchical planning framework for active scene reconstruction using 3d gaussian splatting,â arXiv preprint arXiv:2409.17624, 2024.

[36] B. Yamauchi, âA frontier-based approach for autonomous exploration,â in Proc. of the IEEE Intl. Symp. on Computational Intelligence in Robotics and Automation, 1997.

[37] D. Yan, J. Liu, F. Quan, H. Chen, and M. Fu, âActive implicit object reconstruction using uncertainty-guided next-best-view optimization,â IEEE Robotics and Automation Letters, vol. 8, no. 10, pp. 6395â6402, 2023.

[38] Z. Yan, H. Yang, and H. Zha, âActive neural mapping,â in Proc. of the IEEE/CVF Intl. Conf. on Computer Vision, 2023.

[39] R. Zeng, W. Zhao, and Y.-J. Liu, âPc-nbv: A point cloud based deep network for efficient next best view planning,â in Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems, 2020.

[40] B. Zhou, Y. Zhang, X. Chen, and S. Shen, âFuel: Fast uav exploration using incremental frontier structure and hierarchical planning,â IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 779â786, 2021.

[41] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, âEwa volume splatting,â in Proc. of Visualization, 2001.

## CERTIFICATE OF REPRODUCIBILITY

The authors of this publication declare that:

1) The software related to this publication is distributed in the hope that it will be useful, support open research, and simplify the reproducability of the results but it comes without any warranty and without even the implied warranty of merchantability or fitness for a particular purpose.

2) Liren Jin primarily developed the implementation related to this paper. This was done on Ubuntu20.04.

3) Yue Pan verified that the code can be executed on a machine that follows the software specification given in the Git repository available at:

## https://github.com/dmar-bonn/active-gs

4) Yue Pan verified that the experimental results presented in this publication can be reproduced using the implementation used at submission, which is labeled with a tag in the Git repository and can be retrieved using the command:

## git checkout ral2025