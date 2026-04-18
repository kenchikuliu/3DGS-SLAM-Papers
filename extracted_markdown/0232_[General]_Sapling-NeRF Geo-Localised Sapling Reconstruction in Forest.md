Received XX Month, XXXX; revised XX Month, XXXX; accepted XX Month, XXXX; Date of publication XX Month, XXXX; date of current version XX Month, XXXX.

Digital Object Identifier 10.1109/XXXX.2022.1234567

# Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring

Miguel Angel Mu Â´ noz-Ba Ë nË onÂ´ 1.2, Nived Chebrolu1, Sruthi M. Krishna Moorthy3, Yifu Tao1, Fernando Torres2, Roberto Salguero-Gomez Â´ 3 and Maurice Fallon1

1Oxford Robotics Institute, Department of Engineering Science, University of Oxford, Oxford, UK 2Group of Automation, Robotics and Computer Vision, University of Alicante, Alicante, Spain 3Department of Biology, University of Oxford, Oxford, UK

Corresponding author: Miguel Angel Mu Â´ noz-Ba Ë nËon (email: miguelangel.munoz@ua.es). Â´

This work is supported in part by the EU Horizon 2020 Project 101070405 (DigiForest) and a Royal Society University Research Fellowship. Miguel Angel Mu Â´ noz-Ba Ë nËon is supported by the Valencian Community Government and the Â´ European Union through the fellowship CIAPOS/2023/101

ABSTRACT Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5 m and 2 m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.

INDEX TERMS 3D Reconstruction, Neural Radiance Fields (NeRF), Simultaneous Localization and Mapping (SLAM), Environmental Monitoring, Forestry.

## I. INTRODUCTION

Saplings play a crucial ecological role in forest dynamics, serving as key indicators of forest growth potential. They form the recruitment pool from which mature, canopydominating trees eventually emerge. These trees define the future composition, structure, and diversity of a forest. The architectural traits of saplings include stem height, branching patterns, leaf distribution, and leaf-to-wood ratios. They are key indicators of the immediate survival and growth of a sapling: they dictate a saplingâs efficiency in capturing light, transporting water, maintaining mechanical stability, and optimising growth strategies in diverse light environments [1].

In addition, these traits also influence broader ecological processes such as competition for resources, habitat creation, and biodiversity maintenance [2]. These architectural attributes vary significantly between species, and are shaped by adaptive strategies to specific ecological niches, whether in shaded understories or sunlit canopy gaps.

Despite the ecological importance of the sapling architecture measurements, precisely capturing these fine-scale structural traits in the field remains a significant challenge. Established methods using Terrestrial Laser Scanning (TLS) [3] are unable to capture fine structural details of saplings, such as thin branches and foliage, due to occlusion and insufficient spatial resolution [4]. Classic photogrammetry techniques produce sparse reconstruction results, especially within complex, natural habitats. Additionally, these approaches rely on imprecise GNSS fixes to localise the sapling in a common coordinate system - with a new coordinate frame established from scratch for each SfM session. As purely visual mapping systems they also suffer from scale ambiguity. This makes automatic long-term sapling monitoring difficult, especially when months or years pass between data recordings.

<!-- image-->  
FIGURE 1. We propose a joint NeRF and LiDAR SLAM system for sapling reconstruction in situ in the forest using data from a handheld sensing device. 1) The core LiDAR SLAM system supports hectare-scale multi-session map merging as well as GNSS alignment. 2) We extract a sub-trajectory from a mapping session which encircles an individual sapling of interest and run Structure-from-Motion (SfM, COLMAP) while using the SLAM-derived trajectory to determine consistent metric scale and global localisation. The images and poses are then used to train NeRF models for each sapling. 3) Finally, we demonstrate that a NeRF-derived point cloud s sufficiently accurate to detect the tree skeleton and to measure leaf-wood separation and the leaf distribution. In Section V.C we demonstrate longitudinal monitoring of saplings - from summer to winter.

Emerging technologies such as Neural Radiance Fields (NeRF) [5] and 3D Gaussian Splatting (3DGS) [6] are capable of producing dense reconstructions using relatively few views. Prior works have demonstrated reconstruction of fine detail such as bike spokes and cables as well as being able to infer the location of ambient lighting. Thus, NeRF and 3DGS are promising approaches sapling reconstruction. Some works have shown promis in controlled indoor conditions [7], [8], however it is still challenging to use these technologies in situ in natural environments due to light conditions, wind, and the difficulty of recording data around a sapling without actually disturbing the plant and blurring the representation. Furthermore, these vision-based approaches cannot determine the underlying metric scale of the scene, making it impossible to determine quantitative ecological metrics and to track them over time.

To overcome the limitations mentioned above, we propose to loosely couple NeRF reconstruction with LiDARbased SLAM (Simultaneous Localisation and Mapping). We combining the work with GNSS information (Figure 1), we can then represent the saplings with 3 levels of positioning and reconstruction: imprecise Earth frame location; precise proximal location (and coarse 3D reconstruction) within a

LiDAR map of the forest; and detailed NeRF-derived dense reconstruction and novel-view synthesis.

This strategy has two key advantages: (i) Large-scale and detailed representation: It can build object-centric NeRF-derived reconstruction of specific plants/saplings 1 . This is in contrast with other approaches that build scenecentric reconstructions by dividing the environment spatially and creating a submap for each cell [9]. Our approach can build efficient yet precise NeRF models which capture the fine details of the saplings with less attention given to the background. We can embed these NeRF-derived sapling reconstructions within the LiDAR-based reconstruction of a larger hectare-sized plot, which accurately captures the geometry of the background. (ii) Long-term representation: Our SLAM system can combine multiple sessions recorded over extended time periods (months or years), allowing us to record individual saplings as they grow and evolve. This allows to obtain metrics such as tree structure or leaf distribution to be monitored over multiple seasons or potentially years.

In summary, our contributions are the following:

Object-centric NeRF-derived reconstruction of saplings in challenging natural environments, which allows structural details of the saplings to be captured, which is typically challenging for TLS and MLS-based reconstruction.

â¢ A direct linkage between the NeRF output and a multisession geo-referenced SLAM system, which defines the geometric scale of the NeRFs and their centimetreaccurate localisation which allows monitoring of individual saplings over multiple seasons.

â¢ Demonstrations showing that stem height, branching patterns, leaf distribution, and leaf-to-wood ratios can be measured for small saplings (0.5 m-2 m in height) in situ using this approach, achieving improved accuracy compared to those obtained using TLS.

The rest of this paper is organised as follows: In Section II, we review related works. In Section III, we provide an overview of the entire system, with each module described in Section IV. In Section V, we evaluate each module, as well as demonstrating the measurement of different sapling architectural traits.

## II. RELATED WORK

In recent years, there has been increasing interest in using robotics and computer vision to monitor plant and tree health in forests [10]. In Section II.A, we review recent research on radiance field reconstruction and its application for trees and plant modelling. In Section II.B we review methods for radiance field reconstruction of large-scale environments as well as their use in long-term monitoring.

## A. Radiance Fields for trees and plants

Techniques for digital forest modelling have predominantly relied on LiDAR technology and focused on reconstructing 3D models of mature trees to measure structural attributes such as height, crown volume, branch topology, and biomass [11]. However, capturing saplings accurately remains challenging, because expensive survey-grade LiDAR sensors are typically too sparse to capture fine details such as leaves and small branches (smaller than about 1 cm).

Neural radiance fields (NeRF), introduced in the seminal work by Mildenhall et al. [12], demonstrated that using accurately posed images one can reconstruct a dense textured scene representation. This representation can be used to synthesise photorealistic novel-view images capturing fine details of small objects. Subsequent improvements have extended the approach to better handle unbounded scenes [13], aliasing [14], and to have faster training and inference speed using explicit representations such as voxels [15] and hash grids [16]. An alternative, 3D Gaussian Splatting, has gained popularity since it combines rasterisation and uses an explicit 3D Gaussian representations. 3DGS achieves highly-efficient test-time rendering â much faster than prior works [15], [16]. Follow-up works include surface regularisation [17] and anti-aliasing features [18] to improve the geometry and rendering quality.

The precise, detailed and visually accurate reconstructions created by radiance field approaches make them appealing for quantitative ecological analysis and effective forest monitoring [4]. In [19], the authors used NeRF-derived reconstructions to measure the Diametre-at-Breast-Height (DBH) of individual trees. However, this approach was not been demonstrated in large-scale scenes or for longterm monitoring. Although DBH can be performed with Mobile Laser Scanners (MLS) [20], [21], Koryckiâs work demonstrates that a simple low-cost camera can measure this parameter. Regardless, we feel that the potential of NeRF to reconstructe fine tree details has not get been sufficiently explored.

Other works [22], [23] have compared NeRF-derived reconstruction to LiDAR-based approaches for mature trees, but not the fine structures of saplings or plants. In [7], the authors exploit the capabilities of 3D Gaussian Splatting (3DGS) to monitor the growth of small trees. Different from the aforementioned works which target controlled indoor facilities [7] and parks [22], [23], our work instead targets forest environments which have poor lighting conditions, rough terrain and complicated data recording logistics. Our approach can process measurements taken in situ in a forest, allowing the monitoring of saplings within large hectarescale plots over extended periods.

In this work, we use a NeRF-based approach because we prioritise memory efficiency (to support larger representations) over faster rendering speed. 3D Gaussian Splatting would be an appropriate alternative but requires a larger memory footprint.

## B. Large-scale and long-term Radiance Fields

Many works that extend NeRFs to large-scale environments adopt a submapping approach, where each submap is presented as an individual radiance field model. Block-NeRF [9] proposed to reconstruct a large-scale urban environment by dividing the scene at road intersections and later merging the submaps to achieve continuous image rendering. Mega-NeRF [24] adopts a grid-based partitioning strategy for largescale environments. SiLVR [25] partitions sensor trajectories based on visibility and filters reconstruction artifacts based on uncertainty estimates when merging submaps. Methods that extend the scalability of 3D Gaussian Splatting have used hierarchical structures [26] such as Octrees [27], which enable efficient and high-fidelity rendering at different levels of detail. Different from these works, which focus on reconstructing open environments, our work focuses on regions of interest (the saplings) and localises them within a globally consistent point cloud map built by a LiDAR SLAM system.

Finally some works have sought to capture the dynamics of environments [28], [29]. These approaches focus on dynamic objects moving in front of the camera and focus only on short time frames. For longer-term monitoring, recent research has focused on object-level change detectionl [30], [31]. In our work, we implement a different strategy. We have the objects (the sapling) geo-localised, and we are interested in monitoring how their structural traits change over time such as height and leaf distribution.

## III. SYSTEM OVERVIEW

The main idea of this work is to augment lidar maps with NeRF-derived reconstructions of individual saplings to support their long-term monitoring within large hectarescale plots. To approach this problem, we used the prototype mapping device shown in Figure 2, which contains LiDAR, IMU, GNSS sensors as well as three cameras. The device can be mounted on a robot, but in this work it was hand-carried. We first build a plot-level forest reconstruction by walking through the forest in a âlawnmowerâ pattern, which is suitable for generating plot-level 3D LiDAR maps. We then augment the representation by recording dense camera views cover the entirity of a sapling from all surrounding directions. To achieve this, when we arrive at a sapling that we want to represent with a NeRF, we scan the tree with an inwardsfacing âdomeâ pattern circumscribing the tree (Figure 7). The data acquisition for this can be challenging in a forest when there is understory and branches close to the saplings. A lack of care during data acquisition can affect the quality of the final reconstruction.

<!-- image-->  
FIGURE 2. The proposed system maps a forest environment at three levels of representation. Levels 3 is coarse GNSS-based localisation. Level 2 is a centimetre-accurate representation from a own multi-session LiDAR SLAM system while Level 1 is generated using images captured around a sapling, using SfM to estimate image poses and localising and scaling those poses according to the SLAM system, and finally training a scale-consistent, geo-localised NeRF model for each sapling.

To integrate the dense NeRF-derived reconstruction of the sapling into a consistent geo-localised, multi-session metric map, we loosely couple the NeRF pipeline with a multi-session SLAM pipeline. In Figure 2, we show an overview of the proposed system. Conceptually, we divide our representation into three levels of abstraction. Level 3 is at Earth level, using 2 Hz GNSS. Level 2 is a point cloud map of a complete forest plot where previous work has used the cloud to extract a forest inventory of individual trees, their diameters, species and heights. Mobile Laser Scanners (MLS) are however not accurate enough to capture fine detail about the structure of small objects such as saplings. This is addressed by Level 1 that represents the dense NeRF-derived reconstructions of the saplings which capture detailed point clouds and allow for novel-view synthesis.

<!-- image-->

<!-- image-->

<!-- image-->  
FIGURE 3. By co-registering multiple mapping sessions, we can create a unified map made up of trajectories across different time periods. In this way, we can monitor saplings from summer to winter.

## IV. METHODOLOGY

In this section, we provide a detailed description of each part of the system. In Section IV.A, we describe the multi-session SLAM system used for plot-level LiDAR mapping and trajectory estimation (Levels 3 and 2). Section IV.B presents the process of obtaining a NeRF-derived representation that is metrically scaled and geo-localised by combining with the SLAM system output (Level 1). Given the dense point clouds obtained from the NeRF, we describe in Section IV.C how existing methods for branch skeletonisation can be used to obtain metrics useful for monitoring the growth and health of saplings.

## A. Multi-session and geo-referenced SLAM

The core of the proposed approach is VILENS-SLAM, an online pose graph based LiDAR SLAM system which combines LiDAR-Inertial Odometry [32] with a place recognition module [33] to identify loop closure constraints.

The maps and trajectories from individual SLAM sessions can be merged into a single multi-session pose graph map offline using the approach described in [34]. It is worth noting that by merging the maps at the pose graph level the point cloud maps are deformed and kept locally consistent. In Figure 3, we depict the strategy of using the multimission SLAM system to monitor individual saplings. The $i ^ { \mathrm { { t h } } }$ SLAM session is comprised of a map $\mathcal { M } ^ { \hat { M } _ { i } }$ , a set of LiDAR scans $\mathbf { P } _ { i } ^ { L }$ and a set of poses:

$$
\mathbf { x } _ { i } ^ { M _ { i } } = ( x _ { i _ { 1 } } , \ldots x _ { i _ { k } } ) , \quad i = 1 , \ldots , N , \quad x \in S E ( 3 ) .\tag{1}
$$

The super-index $M _ { i }$ indicates that each session is represented with respect to its own map coordinate frame.

The pose graph map from each session is fed to the multisession SLAM module, where each individual point cloud $P \in \mathbf { P } _ { i } ^ { L }$ and its associated pose $x \in \mathbf { x } _ { i } ^ { M _ { i } }$ is used by the place recognition module to identify new loop closures. When a successful loop closure constraint is found, it is added to the combined pose graph using the first session as a reference frame. This process generates a single combined point cloud map $\mathcal { M } ^ { M _ { 1 } }$ and a combined set of trajectories:

$$
\begin{array} { r } { { \bf X } ^ { M _ { 1 } } = ( { \bf x } _ { 1 } , \ldots , { \bf x } _ { N } ) . } \end{array}\tag{2}
$$

Separately from this process, we also estimate a single transformation between the map frame $M _ { 1 }$ and the earth frame E. This is done in a loosely coupled manner. To estimate the transformation ${ \bf T } _ { M } ^ { E } = ( { \bf \dot { R } } , { \bf t } )$ , we first associate a GNSS latitude and longitude measurement with each pose. We transform those coordinates into a local Northing and Easting frame in metres. To implement this, we associate the first $u ^ { \mathrm { t h } }$ poses with the first $u ^ { \mathrm { t h } }$ GNSS measurements in g using the timestamps that are closest to each other. We then search for the transformation that minimises the error between the positions in the trajectory and the GNSS observations:

$$
\mathbf { T } _ { M } ^ { E * } = \mathop { \arg \operatorname* { m i n } } _ { \mathbf { T } } \left\| \left( \mathbf { R } \mathbf { x } _ { 1 : u } ^ { x , y } + \mathbf { t } \right) - \mathbf { g } _ { 1 : u } \right\| _ { 2 } ^ { 2 } .\tag{3}
$$

This allows us to determine the location of each local pose in the global coordinate frame as well as the linked point cloud and semantic details. This information is crucial for long-term inventory and monitoring (of mature trees). We have found our GNSS alignment to be accurate to about one metre, with GNSS reception only affected in the densest forest canopy.

Sections of each trajectory will correspond to where individual saplings have been circumscribed for image capture. We index each sapling within a plot as $j = 1 , \dots , M$ . Then, for each $\mathbf { x } _ { i } ^ { M _ { 1 } }$ trajectory, we can define a set of subtrajectories as:

$$
\mathbf { Y } _ { i } ^ { M _ { 1 } } = ( \mathbf { y } _ { i _ { 1 } } , \ldots , \mathbf { y } _ { i _ { M } } ) .\tag{4}
$$

In this work, the sapling subtrajectory was extracted manually, but for future work we plan to log the start and end of the sapling during scanning. The methodology described in this section provides the structure to cover saplings in large-scale environments for long-term monitoring. In the next section, we explain how to produce NeRFderived reconstructions using the images associated with the subtrajectories $\mathbf { Y } _ { i } ^ { M _ { 1 } }$

## B. NeRF with loose coupling to LiDAR SLAM

To obtain a detailed representation of each sapling, we use Neural Radiance Fields (NeRF) [5]. NeRF pipelines can render novel views of a scene and also generate dense colorised point clouds. Each sapling is small relative to the size of hectare-scale forest plot. Thus we take the approach of training a single NeRF model for each sapling by scanning it from all directions. The NeRF training pipeline requires a set of images of the scene with precise pose estimates. As we mentioned in the previous section, our SLAM system produces a trajectory around each $j ^ { \mathbf { t h } }$ sapling from each $i ^ { \mathbf { t h } }$ trajectory and we also have good synchronization between our LiDAR data and our camera image. However, we have found that the poses obtained by the SLAM system are not sufficiently precise to achieve the highest possible quality NeRF-derived reconstruction. To refine the sensor poses, we use Structure from Motion (SfM), which is a widely used approach to estimate the input poses for NeRF. Specifically, we use COLMAP [35], [36] to produce a trajectory around each sapling:

<!-- image-->  
Transform and scale SfM

<!-- image-->  
FIGURE 4. Rescaling and aligning multiple SfM reconstructions: From each SLAM-derived session of a plot (upper), we extract a subtrajectory where a sapling is scanned in detail, and then estimate a COLMAP SfM trajectory which has a scale ambiguity. This is corrected using the SLAM-derived trajectory (lower) to produce the (co-aligned) trajectories shown in red in the example.

$$
\mathbf { Z } = ( \mathbf { z } _ { 1 1 } ^ { F _ { 1 1 } } , \ldots , \mathbf { z } _ { N M } ^ { F _ { N M } } ) .\tag{5}
$$

The superscript F means that each new sapling scan trajectory is represented in itâs own local SfM frame. However, because COLMAP is a monocular vision-based approach, this trajectory is only accurate up to a scale ambiguity. To represent each refined trajectory in the map frame $M _ { 1 }$ and to resolve the correct scaling factor, we use the Umeyama method [37].

By following this process, the pipeline can obtain a set of precise camera poses, represented in a single consistent coordinate frame $M _ { 1 }$ , as depicted in red in Figure 4. Then for each $i ^ { \mathbf { t h } }$ trajectory, we have:

$$
\begin{array} { r } { { \bf Z } _ { i } ^ { M _ { 1 } } = ( { \bf z } _ { i _ { 1 } } , \ldots , { \bf z } _ { i _ { M } } ) . } \end{array}\tag{6}
$$

Given $\mathbf { Z } ^ { M _ { 1 } }$ trajectories and their associated camera images, we then train a NeRF model for each sapling. In this way, we can generate dense, scaled and geo-referenced point clouds $\mathbf { S } _ { i j }$ in the LiDAR map frame and position them with the larger hectare-scale plots.

## C. Skeletonisation and leaf-wood separation

To illustrate how the output of our pipeline can be useful for individualised ecological monitoring, we processed the output NeRF point clouds with some publicly available tools to extract the branching structure. This allows us to quantitively measure attributes of very small saplings to enable ecologists to monitor sapling health.

We first extracted the skeleton of a sapling from its point cloud $\mathbf { S } _ { i j }$ using PC-Skeletor [38]. PC-Skeletor was originally developed to extract branch structure from tree point clouds by extracting curve-like structures from unorganised 3D data. The algorithm builds a k-nearest neighbour graph from the raw sapling point clouds and applies an iterative contraction process that progressively reduces redundant geometry while preserving global connectivity. During this process, spurious branches are pruned based on geometric saliency, resulting in a compact and topologically consistent structure.

This approach is particularly well-suited for data captured in natural forests, where point clouds are often noisy, incomplete, or occluded. Using this approach we can obtain a skeletonised point cloud $\mathbf { K } _ { i j }$ that captures the saplingâs main branches, along with an open graph representation $\mathcal { G } _ { i j } = ( \mathbf { V } , \mathbf { E } )$ encoding its topology. In Figures 13 and 14, we show results of this process. It is worth noting that the point clouds obtained from NeRF are very dense. To achieve best performance with PC-Skeletor, we subsample the clouds to avoid overskeletonisation.

To separate leaf points from woody components, we initially evaluated several pre-trained models for leaf segmentation [39], [40]. However, these models are intended for large adult trees. The methods tended to misclassify the points of a sapling point cloud $\mathbf { S } _ { i j }$ and did not generalise to small-scale structures.

To overcome this limitation, we implemented a geometrydriven segmentation pipeline tailored to the fine-scale architecture of saplings. First, we compute the skeleton $\mathbf { K } _ { i j }$ of the sapling from the high-density point cloud $\mathbf { S } _ { i j }$ , without applying any downsampling. This will oversegment the skeleton, as depicted in Figure 5. The resulting skeleton captures the main trunk and branch architecture, but due to the high surface complexity of the leaves, it also introduces a large number of terminal bifurcations at the canopy surface (Figure 5). These bifurcations manifest as bottom-level vertices in the skeleton graph $\mathcal { G } _ { i j }$

We then identify the corresponding leaf regions by segmenting all points in $\mathbf { S } _ { i j }$ that lie within a spatial neighbourhood around these terminal vertices. This results in two disjoint point clouds: $\mathbf { L } _ { i j }$ for the leaf points, and $\mathbf { W } _ { i j }$ for the wood components (trunk and branches). The result of this segmentation is shown in Figures 13 and 14.

This approach allows our method to extract the structural attributes that are critical for understanding ecological processes. Specifically, we derive stem height from the original point cloud $\mathbf { S } _ { i j }$ , branching architecture from the skeleton $\mathbf { K } _ { i j } .$ , spatial leaf distribution from the segmented leaf points $\mathbf { L } _ { i j }$ , and leaf-to-wood ratio by comparing $\mathbf { L } _ { i j }$ and $\mathbf { W } _ { i j }$

<!-- image-->  
FIGURE 5. Leaf segmentation: (a) A detailed view of the NeRF-derived point cloud of sapling 01 showing individual leaves. (b) Leaf segmentation with overskeletonization for segmentation purposes, which can be avoided with appropriate parameter turning.

## V. EVALUATION

The experimental evaluation was conducted using our selfdeveloped Frontier device, which integrates a Hesai Pandar QT64 LiDAR with an IMU, and three RGB cameras from E-CON (with model number e-CAM20 CUOAGX). Data collection took place in Wytham Woods (UK) during July, August and December 2025 (so covering Summer and Winter) and recorded a total of 23 saplings. Five individual saplings were captured in the three recordings to demonstrate reacquisition of individual sampling for temporal and interseasonal monitoring. In addition, we evaluate our approach on another dataset collected in June in the Evo forest (Finland), which includes 3 conifer saplings so as to demonstrate that the approach can be used in other environment. To compare the results with the established methodology used for ecological monitoring, we scanned the saplings using a Leica RTC360 TLS scanner.

In the following, we present the results of our experimental evaluation. In Section V.A, we show the evaluation of the multi-session SLAM and geo-localisation module. Section V.B reports on the novel-view synthesis performance, highlighting the quality of the rendered views. Section V.C focuses on the reconstruction of 3D point clouds and their applicability for long-term monitoring, assessing both geometric quality and temporal consistency. Finally, Section V.D examines the extraction of sapling architectural attributes relevant to ecological monitoring, including stem height, branching patterns, and leaf-to-wood ratios.

<!-- image-->  
FIGURE 6. Multi-session geo-localisation results: Geo-referenced aerial image of the area in Wytham Woods containing saplings 01â09. The saplings 10-13 are in another plot, 100 meters further west. The trajectories of each session are shown in different colours, combining data from July and August 2025. Each sapling is georeferenced and marked with a red star. Example point clouds generated using NeRF for several geo-localised saplings are also displayed.

## A. Evaluating multi-session SLAM and geo-localisation

To assess the multi-session SLAM module, multiple mapping sessions were fused using the method described in Section IV.A. They were also geo-referenced using GNSS measurements. Consequently, each NeRF-derived sapling reconstruction can then be accurately positioned in a common global reference frame. Multi-session geo-localisation results are presented in Figure 6, showing the locations of saplings 01â09 overlaid on a geo-referenced aerial image of the forest area at Wytham Woods (saplings 10-13 are 100 meters west). The trajectories of each session, collected in July, August and December 2025, are depicted with distinct colors, while each sapling is marked with a red star. Example NeRFderived point clouds for several geo-localised saplings are also shown. These results demonstrate how the proposed framework can support long-term monitoring and larger scale deployment in future.

## B. Evaluating novel-view synthesis accuracy

In Figure 7, we show an example of a trajectory recorded around a sapling using the handheld device. These camera poses are used to train the NeRF model of the saplings, as shown in Figure 8. In Table 1, we summarise the quantitative evaluation metrics (PSNR, SSIM, and LPIPS) obtained for each sapling. The saplings are clustered into three size ranges. We can see that our method performs best for smaller saplings <1 m. This is because it is easier to record a full set of views around the saplings for a smaller sapling (as in Figure 7). Additionally, since the sequences were recorded in an object-centric manner and metrics where obtained from images including background, part of the variability in performance may be attributed to differences in the background composition rather than the saplings themselves. This effect can be observed in Figure 8, where Sapling 03, which achieved the highest PSNR, appears qualitatively comparable to Sapling 08, whose reconstruction appears to be sharper despite lower PSNR.

<!-- image-->  
FIGURE 7. Camera trajectory for image acquisition: (a) Example 3D reconstruction (of sapling 08). (b) Camera poses around the sapling, where blue points indicate the camera focal point and red/grey rectangles represent the camera frustrums.

<!-- image-->  
FIGURE 8. Novel-view synthesis results for four saplings from Wytham Woods. Each sapling is shown from two perspectives: the top row shows views from above while the lower rows shows views from the side. Background pixels have been removed using a bounding box when rendering each tree to better highlight the reconstructed structure.

These results confirm the robustness of our approach for novel-view synthesis, showing that despite background complexity, the reconstructed saplings maintain a consistent level of visual fidelity suitable for downstream ecological analysis.

TABLE 1. Evaluation of Novel View Synthesis Quality
<table><tr><td>Size Range (m)</td><td>Sapling ID</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td rowspan="5">0.5 - 0.75</td><td>02</td><td>19.72</td><td>0.571</td><td>0.369</td></tr><tr><td>03</td><td>21.12</td><td>0.737</td><td>0.296</td></tr><tr><td>04</td><td>19.57</td><td>0.638</td><td>0.367</td></tr><tr><td>08</td><td>15.36</td><td>0.360</td><td>0.603</td></tr><tr><td>12</td><td>19.87</td><td>0.518</td><td>0.395</td></tr><tr><td rowspan="5">0.75 - 1.0</td><td>01</td><td>19.18</td><td>0.643</td><td>0.384</td></tr><tr><td>07</td><td>15.49</td><td>0.429</td><td>0.639</td></tr><tr><td>09</td><td>19.21</td><td>0.589</td><td>0.324</td></tr><tr><td>10</td><td>18.44</td><td>0.541</td><td>0.378</td></tr><tr><td>11</td><td>20.31</td><td>0.592</td><td>0.312</td></tr><tr><td rowspan="3">&gt;1.0</td><td>05</td><td>17.88</td><td>0.456</td><td>0.466</td></tr><tr><td>06</td><td>17.23</td><td>0.412</td><td>0.487</td></tr><tr><td>15</td><td>16.24</td><td>0.427</td><td>0.526</td></tr></table>

<!-- image-->  
FIGURE 9. Comparison between reconstructions of a conifer sapling from the Evo forest (Finland): a) MLS point cloud, B) NeRF-derived point cloud, and c) NeRF-derived novel-view rendering. The MLS result is very noisy and is unable to represent the details of the sapling structure, while the NeRF provides a much clearer and more coherent reconstruction.

## C. Point cloud accuracy and usefulness for temporal monitoring

From the reconstructed NeRFs, we generate 3D point clouds that can then be used for structural and temporal analysis. Figure 9 presents a representative example of Sapling 14 from the Evo forest (Finland), comparing the MLS reconstruction, the NeRF-derived point cloud, and a novel-view rendering. In this case, our MLS pipeline is clearly too noisy and is unable to represent the fine structure of the sapling2, whereas NeRF produces a much more detailed reconstruction, capturing both stem and branch geometry as well as realistic appearance in the rendered views.

Beyond individual examples, we systematically evaluate the quality of NeRF-derived point clouds against the Leica

<!-- image-->  
FIGURE 10. Comparison of NeRF-derived (top row) and TLS (bottom row) point clouds for four saplings (01, 02, 03, and 05) from Wytham Woods. NeRF-derived reconstructions capture fine-scale geometry and foliage much more accurately than TLS, with the superiority being especially evident for smaller saplings (e.g., Sapling 03 and 02), where TLS fails to recover thin branches and leaves.

TLS scans (which is much more accurate than the LiDAR sensor in the MLS device). Figure 10 presents views for four saplings (01, 02, 03, and 05), demonstrating that NeRFderived reconstructions are consistently superior in detail and fidelity, particularly for smaller saplings, while the TLS results are sparse and noisy.

Sapling 03, which measures only 56 cm in height, is much more poorly reconstructed by the TLS, while the NeRF preserves both stem and branching structures. For taller saplings such as Sapling 05 (>1 m), TLS can capture the structure better and some branches can be distinguished, but NeRF still resolves fine structure more clearly.

To assess the potential of using NeRF-derived point clouds for long-term monitoring, we analysed data recorded over three recordings - from summer until winter. Figure 11 (a) shows the Sapling 01, reconstructed in July (red) and August (yellow), and overlaid for comparison. Only small variations in leaf distribution are observed between the two captures. Moving to December in Figure 11 (b), we see the representation of Sapling 01 where the leaves have mostly fallen.

In Figure 12 we see Sapling 02 suffering from more significant branch damage. Moving from July to August a single branch (red arrow) is missing however by December, all the leaves have fallen off. This demonstrates the ability of our approach to track sapling growth but abrupt structural damage relevant to ecological monitoring.

These experiments clearly demonstrate that MLS is unable to measure elements smaller than about 3cm. NeRF consistently delivers denser, cleaner, and more faithful reconstructions of small saplings, enabling the recovery of architectural details that are otherwise lost in TLS point clouds. Even for taller saplings, where TLS performs better, NeRF still provides a noticeable improvement in geometric accuracy and completeness. This demonstrates the potential of using NeRF-derived reconstructions as a scalable and reliable alternative to traditional TLS methods for ecological monitoring of small trees.

## D. Sapling attributes for ecological monitoring

In addition to visual and geometric evaluation, we assess the suitability of using NeRF-derived reconstructions to measure ecologically-relevant sapling attributes. Table 2 reports a comparison between three key metrics from NeRFderived and TLS representations: stem height, bifurcations in the sapling structure (skeleton) and the leaf-to-wood ratio (LWR), defined at the point level as:

$$
\mathrm { L W R } = \frac { N _ { l } } { N _ { w } } ,\tag{7}
$$

where $N _ { l }$ and $N _ { w }$ correspond to the number of points classified as leaf and wood, respectively. Ground truth for sapling height is unavailable, as it cannot be measured precisely with a physical ruler: the irregularity of the forest floor and the non-vertical orientation of sapling stems would lead to inconsistencies depending on the chosen reference point. Nevertheless, the estimations obtained from NeRF and TLS differ by only 1 â2 cm, which indicates consistency across the methods. A larger deviation is observed for Sapling 06 (â 3 m tall), with a difference of 4 cm. This was due to incomplete capture of the upper canopy during handheld data collection.

<!-- image-->  
a) Sapling 01 - July (red) / August (yellow)

<!-- image-->  
FIGURE 11. Upper: (a) Multi-session monitoring of Sapling 01 using NeRF point clouds for July (red) and August (yellow) - showing little change. Lower: (b) Here Sapling 1 exhibits more differences with part of the leaves fallen.

<!-- image-->  
FIGURE 12. Sapling 02 undergoing structure damage and leave off: the upper stem and highest branch was damaged between July and August (red arrows). By December the leaves have fallen off.

For the leaf-to-wood ratio, results differ significantly. The TLS-estimated value is consistently much lower, as most points are incorrectly classified as wood due to its poorly defined skeleton and limited branching detail. By contrast, NeRF-derived LWR estimates are more realistic3.

TABLE 2. Evaluation of sapling architectural traits
<table><tr><td></td><td>Height (m)</td><td>Leaf Wood Ratio</td><td>Bifurcations</td><td></td></tr><tr><td>Sapling ID</td><td>NeRF TLS</td><td>NeRF TLS</td><td>NeRF</td><td>TLS</td></tr><tr><td>01</td><td>0.89 0.91</td><td>12.54</td><td>0.03 98</td><td>38</td></tr><tr><td>02</td><td>0.58 0.59</td><td>23.62 0.09</td><td>27</td><td>11</td></tr><tr><td>03</td><td>0.56 0.55</td><td>19.24 0.11</td><td>35</td><td>18</td></tr><tr><td>04</td><td>0.70 0.72</td><td>17.66 0.10</td><td>47</td><td>22</td></tr><tr><td>05</td><td>1.36 1.33</td><td>14.21 0.06</td><td>112</td><td>39</td></tr><tr><td>06</td><td>2.89 2.93</td><td>21.24 0.04</td><td>124</td><td>69</td></tr><tr><td>07</td><td>0.77 0.76</td><td>18.55</td><td>0.12 84</td><td>35</td></tr><tr><td>08</td><td>0.73 0.74</td><td>32.59</td><td>0.27 87</td><td>43</td></tr><tr><td>09</td><td>0.77 0.79</td><td>21.22</td><td>0.15 78</td><td>37</td></tr><tr><td>10</td><td>0.85 0.86</td><td>19.87</td><td>0.22 89</td><td>38</td></tr><tr><td>11</td><td>0.93 0.91</td><td>22.54</td><td>0.13 101</td><td>46</td></tr><tr><td>12</td><td>0.61 0.62</td><td>24.18</td><td>0.18</td><td>91 33</td></tr></table>

Additionally, for each sapling, we counted the number of bifurcations in the branch topology that describe the saplingâs skeleton structure. In the case of the skeletons obtained from TLS, some fine branches are not reconstructed due to limited structural detail. As illustrated in Figure 13, a thin branch near the base of the sapling is fused with the trunk in the TLS skeleton, whereas it is correctly distinguished in the NeRF-derived skeleton. This effect is reflected in the increase number of detected bifurcations showin in Table 2.

To compare the TLS and NeRF-derived outputs, Figures 13 and 14 show examples for Saplings 01 and 02. The skeleton, leaf points, and wood points are visualised alongside the vertical leaf distribution profile. The distributions are computed using Kernel Density Estimation (KDE) [41], [42], which provides a smooth characterisation of leaf density along the stem. For both saplings, we see that the majority of the TLS points are classified as wood, resulting in overly sparse and uniform leaf distributions. For the NeRF achieves a segmentation which is denser and captures both foliage clusters and their vertical variability.

In summary, NeRF-derived reconstructions not only outperform TLS in terms of geometric fidelity but also enable the consistent estimation of ecologically meaningful attributes such as sapling height and leaf-to-wood ratio. While TLS often fails to capture fine-scale foliage and branch structures, leading to unrealistic attribute values, NeRF produces more accurate and temporally stable measurements. This capability highlights the potential of NeRF as a powerful tool for long-term ecological monitoring.

<!-- image-->  
FIGURE 13. Attributes of Sapling 01 - using either a NeRF (top row) or a TLS pipeline (bottom row). The subfigures show the skeleton, leaf points, wood points (left to right) as well as a vertical leaf distribution plot. TLS points are misclassified as wood. Using NeRF-derived representation produces a clearer separation of leaf and wood points and a more realistic vertical distribution of foliage.

## VI. CONCLUSIONS

Saplings are fundamental indicators of forest regeneration, yet their fine-scale architectural traits are difficult to capture with conventional 3D sensing technologies. TLS, MLS, and photogrammetry are limited in their ability to reconstruct thin branches and dense foliage, and often lack the geometric consistency required for long-term monitoring. In this work, we introduced a system that fuses NeRF, LiDAR SLAM, and GNSS, to provide repeatable, geo-localised reconstructions that enable quantitative evaluation of the structure and growth of small saplings over time while localizing them within hectare-scale plots.

Our results demonstrate that the proposed approach consistently outperforms TLS and MLS for a variety of measurement tasks. In novel-view synthesis, NeRF delivers visually faithful reconstructions despite background variability. For 3D point clouds, NeRF captures greater geometric detail, especially for small saplings where TLS reconstructions often lose structural details, while also enabling long-term comparisons that reveal subtle growth patterns as well as abrupt structural changes such as branch loss. In terms of ecological attributes, we show that NeRF-derived reconstructions provide consistent height measurements (within 1 â2 cm of TLS-estimated result) and more accurate leaf-towood ratios, thanks to the improved classification of foliage and skeletal structures. These advantages make it possible to capture ecologically relevant metrics such as vertical leaf distribution profiles with a level of detail unattainable by TLS or MLS.

Overall, the fusion of NeRF with LiDAR SLAM and GNSS represents a scalable and reliable solution for monitoring young trees in situ. By enabling accurate reconstruction of saplings between 0.5 and 2 m tall, our method provides ecologists with richer structural data on stem height, branching patterns, and leaf-to-wood allocation. This capability opens new opportunities for long-term, repeatable ecological monitoring, offering quantitative insights into regeneration dynamics that are crucial for understanding forest composition, competition, and resilience.

Future work could explicitly couple the extracted sapling traits with demographic and survival models to quantify traitmediated recruitment dynamics. Integrating species identity and functional strategies would further enable scaling the system to regeneration cohorts across disturbance gradients, positioning this framework as a foundation for trait-informed forest regeneration modelling.

## REFERENCES

[1] L. Poorter and M. J. Werger, âLight environment, sapling architecture, and leaf display in six rain forest tree species,â American Journal of Botany, vol. 86, no. 10, pp. 1464â1473, 1999.

[2] T. Kohyama and M. Hotta, âSignificance of allometry in tropical saplings,â Functional ecology, pp. 515â521, 1990.

<!-- image-->

<!-- image-->  
FIGURE 14. Measured attributes of Sapling 02 - using either a NeRF pipeline (top row) or a TLS pipeline (bottom row). As in Figure 13, NeRF-derived yields a well-defined skeleton and denser leaf distribution profiles using our described procedure, whereas from TLS data, the skeletonisation pipeline has failed to extract small branches, thus classifying leaves as wood, which leads to less informative and oversimplified distributions.

[3] X. Liang, V. Kankare, J. Hyyppa, Y. Wang, A. Kukko, H. Haggr Â¨ en, Â´ X. Yu, H. Kaartinen, A. Jaakkola, F. Guan et al., âTerrestrial laser scanning in forest inventories,â ISPRS Journal of Photogrammetry and Remote Sensing, vol. 115, pp. 63â77, 2016.

[4] H. Cerbone, S. M. K. Moorthy, R. Salguero-Gomez, and G. Taylor, âDemocratizing 3d ecology: Mobile neural radiance field for scalable ecosystem mapping in change detection,â 2025.

[5] G. Wang, L. Pan, S. Peng, S. Liu, C. Xu, Y. Miao, W. Zhan, M. Tomizuka, M. Pollefeys, and H. Wang, âNerf in robotics: A survey,â arXiv preprint arXiv:2405.01333, 2024.

[6] S. Zhu, G. Wang, X. Kong, D. Kong, and H. Wang, â3d gaussian splatting in robotics: A survey,â arXiv preprint arXiv:2410.12262, 2024.

[7] S. Adebola, S. Xie, C. M. Kim, J. Kerr, B. M. van Marrewijk, M. van Vlaardingen, T. van Daalen, E. van Loo, J. L. S. Rincon, E. Solowjow et al., âGrowSplat: Constructing temporal digital twins of plants with gaussian splats,â arXiv preprint arXiv:2505.10923, 2025.

[8] S. Wu, C. Hu, B. Tian, Y. Huang, S. Yang, S. Li, and S. Xu, âA 3D reconstruction platform for complex plants using OB-NeRF,â Frontiers in Plant Science, vol. 16, p. 1449626, 2025.

[9] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan, J. T. Barron, and H. Kretzschmar, âBlock-nerf: Scalable large scene neural view synthesis,â in IEEE Int. Conf. Computer Vision and Pattern Recognition, 2022, pp. 8248â8258.

[10] M. Mattamala, N. Chebrolu, J. Frey, L. FreiÃmuth, H. Oh, B. Casseau, M. Hutter, and M. Fallon, âBuilding forest inventories with autonomous legged robotsâsystem, lessons, and challenges ahead,â J. of Field Robotics, vol. 2, pp. 418â436, 2025.

[11] A. A. Borsah, M. Nazeer, and M. S. Wong, âLidar-based forest biomass remote sensing: A review of metrics, methods, and assessment criteria for the selection of allometric equations,â Forests, vol. 14, no. 10, p. 2095, 2023.

[12] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields

for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[13] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-NeRF 360: Unbounded anti-aliased neural radiance fields,â in IEEE Int. Conf. Computer Vision and Pattern Recognition, 2022, pp. 5470â5479.

[14] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-NeRF: A multiscale representation for antialiasing neural radiance fields,â in IEEE Int. Conf. Computer Vision and Pattern Recognition, 2021, pp. 5855â5864.

[15] A. Yu, R. Li, M. Tancik, H. Li, R. Ng, and A. Kanazawa, âPlenoctrees for real-time rendering of neural radiance fields,â in Intl. Conf. on Computer Vision (ICCV), 2021, pp. 5752â5761.

[16] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[17] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2D gaussian splatting for geometrically accurate radiance fields,â in SIGGRAPH, 2024.

[18] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, âMip-Splatting: Alias-free 3D gaussian splatting,â in IEEE Int. Conf. Computer Vision and Pattern Recognition, 2024, pp. 19 447â19 456.

[19] A. Korycki, C. Yeaton, G. S. Gilbert, C. Josephson, and S. McGuire, âNeRF-accelerated ecological monitoring in mixed-evergreen redwood forest,â Forests, vol. 16, no. 1, p. 173, 2025.

[20] J.-F. Tremblay, M. Beland, R. Gagnon, F. Pomerleau, and P. Gigu Â´ ere, \` âAutomatic three-dimensional mapping for tree diameter measurements in inventory operations,â J. of Field Robotics, vol. 37, no. 8, pp. 1328â1346, 2020.

[21] J. Tang, Y. Chen, A. Kukko, H. Kaartinen, A. Jaakkola, E. Khoramshahi, T. Hakala, J. Hyyppa, M. Holopainen, and H. Hyypp Â¨ a,Â¨ âSLAM-aided stem mapping for forest inventory with small-footprint mobile LiDAR,â Forests, vol. 6, no. 12, pp. 4588â4606, 2015.

[22] H. Huang, G. Tian, and C. Chen, âEvaluating the point cloud of individual trees generated from images based on neural radiance fields (NeRF) method,â Remote Sensing, vol. 16, no. 6, p. 967, 2024.

[23] A. Masiero, E. I. Parisi, A. Guarnieri, and F. Pirotti, âComparing nerf and lidar-based plant reconstruction,â in IEEE Intl. Workshop on Metrology for Agriculture and Forestry, 2024, pp. 167â172.

[24] H. Turki, D. Ramanan, and M. Satyanarayanan, âMega-NeRF: Scalable construction of large-scale NeRF for virtual fly-throughs,â in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 12 922â12 931.

[25] Y. Tao and M. Fallon, âSilvr: Scalable lidar-visual radiance field reconstruction with uncertainty quantification,â IEEE Trans. Robotics, 2025.

[26] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and G. Drettakis, âA hierarchical 3D gaussian representation for real-time rendering of very large datasets,â ACM Transactions on Graphics, vol. 43, no. 4, July 2024.

[27] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, âOctree-GS: Towards consistent real-time rendering with LOD-structured 3D Gaussians,â IEEE Trans. Pattern Anal. Machine Intell., 2024.

[28] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, âDnerf: Neural radiance fields for dynamic scenes,â in IEEE Int. Conf. Computer Vision and Pattern Recognition, 2021, pp. 10 318â10 327.

[29] Z. Guo, W. Zhou, L. Li, M. Wang, and H. Li, âMotion-aware 3d gaussian splatting for efficient dynamic scene reconstruction,â IEEE Trans. on Circuits and Systems for Video Technology, 2024.

[30] Z. Lu, J. Ye, and J. Leonard, â3dgs-cd: 3d gaussian splatting-based change detection for physical object rearrangement,â IEEE Robotics and Automation Letters, 2025.

[31] R. Huang, P. Li, H. Tao, and B. Jiang, âSemanticDifference: Change detection with multi-scale vision-language representation difference,â in Intl. Conf. on Intelligent Computing. Springer, 2025, pp. 146â157.

[32] D. Wisth, M. Camurri, and M. Fallon, âVilens: Visual, inertial, lidar, and leg odometry for all-terrain legged robots,â IEEE Trans. Robotics, vol. 39, no. 1, pp. 309â326, 2022.

[33] M. Ramezani, G. Tinchev, E. Iuganov, and M. Fallon, âOnline LiDAR-SLAM for legged robots with robust registration and deep-learned loop closure,â in IEEE Intl. Conf. on Robotics and Automation (ICRA). IEEE, 2020, pp. 4158â4164.

[34] H. Oh, N. Chebrolu, M. Mattamala, L. FreiÃmuth, and M. Fallon, âEvaluation and deployment of lidar-based place recognition in dense forests,â in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 12 824â12 831.

[35] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,âÂ¨ in IEEE Int. Conf. Computer Vision and Pattern Recognition, 2016.

[36] J. L. Schonberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, âPixel-Â¨ wise view selection for unstructured multi-view stereo,â in Eur. Conf. on Computer Vision (ECCV), 2016.

[37] S. Umeyama, âLeast-squares estimation of transformation parameters between two point patterns,â IEEE Trans. Pattern Anal. Machine Intell., vol. 13, no. 04, pp. 376â380, 1991.

[38] L. Meyer, A. Gilson, O. Scholz, and M. Stamminger, âCherrypicker: Semantic skeletonization and topological reconstruction of cherry trees,â in IEEE Int. Conf. Computer Vision and Pattern Recognition, 2023.

[39] H. Li, G. Wu, S. Tao, H. Yin, K. Qi, S. Zhang, W. Guo, S. Ninomiya, and Y. Mu, âAutomatic branchâleaf segmentation and leaf phenotypic parameter estimation of pear trees based on three-dimensional point clouds,â Sensors, vol. 23, no. 9, p. 4572, 2023.

[40] T. Jiang, Q. Zhang, S. Liu, C. Liang, L. Dai, Z. Zhang, J. Sun, and Y. Wang, âLWSNet: A point-based segmentation network for leafwood separation of individual trees,â Forests, vol. 14, no. 7, p. 1303, 2023.

[41] M. Luotamo, M. Yli-Heikkila, and A. Klami, âDensity estimates Â¨ as representations of agricultural fields for remote sensing-based monitoring of tillage and vegetation cover,â Applied Sciences, vol. 12, no. 2, p. 679, 2022.

[42] J. Shi, G. Qiu, X. Liu, X. Zhang, M. Zhao, C. Wu, and D. Dong, âSpatiotemporal remote sensing monitoring and driving mechanism analysis of vegetation quality in crested ibis habitats: A case study of deqing county,â All Earth, 2025.