# Active Semantic Mapping of Horticultural Environments Using Gaussian Splatting

Jose Cuaran1, Naveen Kumar Uppalapati3, and Girish Chowdhary1,2

AbstractâSemantic reconstruction of agricultural scenes plays a vital role in tasks such as phenotyping and yield estimation. However, traditional approaches that rely on manual scanning or fixed camera setups remain a major bottleneck in this process. In this work, we propose an active 3D reconstruction framework for horticultural environments using a mobile manipulator. The proposed system integrates the classical Octomap representation with 3D Gaussian Splatting to enable accurate and efficient target-aware mapping. While a low-resolution Octomap provides probabilistic occupancy information for informative viewpoint selection and collisionfree planning, 3D Gaussian Splatting leverages geometric, photometric, and semantic information to optimize a set of 3D Gaussians for high-fidelity scene reconstruction. We further introduce simple yet effective strategies to enhance robustness against segmentation noise and reduce memory consumption. Simulation experiments demonstrate that our method outperforms purely occupancy-based approaches in both runtime efficiency and reconstruction accuracy, enabling precise fruit counting and volume estimation. Compared to a 0.01 m-resolution Octomap, our approach achieves an improvement of 6.6% in fruit-level F1 score under noise-free conditions, and up to 28.6% under segmentation noise. Additionally, it achieves a 50% reduction in runtime, highlighting its potential for scalable, real-time semantic reconstruction in agricultural robotics.

Index TermsâAgricultural Robotics, Active Mapping, Gaussian Splatting, Horticulture, Mobile Manipulator

## I. INTRODUCTION

Modern agriculture increasingly relies on automation and data-driven technologies to address challenges such as labor shortages, sustainability demands, and the need for improved productivity [1]. Within this context, agricultural phenotyping plays a fundamental role in plant breeding, driving advances in yield, resilience, and resource efficiency. At the core of phenotyping tasks lies the collection of image data, which is subsequently used for the 3D reconstruction of plants. These 3D plant models enable the accurate estimation of phenotypic traits such as plant height, fruit size, and biomass content. However, generating high-quality 3D plant models remains a labor-intensive and time-consuming process due to challenges such as occlusions, varying lighting conditions, and the need for precise camera positioning [2]. Traditional methods often rely on scanning plants with handheld cameras or using fixed-camera setups in controlled environments [3], [4]. Robotic platforms have emerged as an alternative for automating data collection in agricultural fields, but most still depend on fixed or predefined camera configurations [5], [6].

To overcome these limitations, several studies have explored active mapping methods for agricultural environments [6]â[10], in which the camera pose is continuously adjusted to capture specific targets, such as fruits or leaves. Despite their effectiveness in viewpoint planning, most of these approaches rely on Octomap-based representations. While Octomap provides a probabilistic occupancy model that is useful for information-driven exploration and collision-free planning, it often results in low-quality reconstructions and inaccurate volume estimations due to its dependence on voxel resolution [10]. Increasing map resolution can improve accuracy but also leads to significantly higher computational costs due to the intensive ray-casting operations required.

Recently, neural representations such as Neural Radiance Fields (NeRF) [11] and 3D Gaussian Splatting (3DGS) [12] have emerged as powerful techniques for high-quality novel-view synthesis and 3D reconstruction by leveraging geometric and photometric cues. Unlike NeRF, 3DGS offers an explicit representation that enables real-time rendering and efficient optimization. However, 3DGS does not inherently model unknown or unobserved space, which is essential for information-driven active mapping and collision-free planning.

To bridge this gap, we propose an approach for active semantic mapping that leverages the complementary strengths of Octomap and 3DGS. In our hybrid framework, a low-resolution octomap provides probabilistic occupancy information for safe manipulation and viewpoint selection, while 3DGS refines the reconstruction by exploiting geometric, photometric, and semantic information for highfidelity modeling. In addition to integrating these two representations, we address several challenges, including the large memory footprint of 3DGS, the segmentation noise from imperfect models, and the computation of exploration frontiers. We introduce simple yet effective strategies that improve robustness to noise, efficiency, and scalability. Our results demonstrate significant improvements in both reconstruction accuracy and runtime, advancing a step toward highthroughput field phenotyping.

In summary, the main contributions of this paper are:

â¢ A novel framework for active semantic mapping that combines a low-resolution octomap and 3DGS for efficient and accurate reconstruction.

<!-- image-->

<!-- image-->

<!-- image-->  
Semantic OctoMap under noiseless conditions.

<!-- image-->  
Our 3DGS-based approach with segmentation noise

<!-- image-->  
Semantic OctoMap with segmentation noise.

Fig. 1. Top: Simulation environment with the Terrasentia robot. Bottom: Sample reconstruction of a single crop row demonstrating the robustness of our approach under segmentation noise, compared to a high-resolution octomap.

â¢ A strategy for handling uncertain semantic predictions.

â¢ A graph-based planning approach that exploits prior farm layout information for efficient exploration.

## II. RELATED WORKS

3D reconstruction of plants has emerged as a valuable tool for plant phenotyping, enabling precise quantification of traits such as plant height, leaf area, stem thickness, canopy volume, and fruit size or count [13]. These traits are critical for evaluating plant health, monitoring stress conditions, and supporting breeding programs.

Approaches for 3D reconstruction in agriculture can broadly be divided into passive and active methods. Passive methods typically rely on fixed camera setups in controlled environments, handheld imaging sensors, or mobile platforms with limited viewpoint control [3], [5], [14]. Multiview stereo (MVS) has been widely adopted in this context, enabling the reconstruction of plant structures from overlapping 2D images captured under varying viewpoints [15]. More recently, learning-based techniques such as neural radiance fields (NeRFs) [16] and 3D Gaussian Splatting [12] have shown advantages in producing smooth and photorealistic reconstructions, with improved handling of complex geometries such as thin leaves and occluded plant regions [4], [17]â[21]. Despite these advances, passive methods still face several challenges. They are often labor-intensive, requiring dense image capture and accurate image registration. Moreover, these reconstructions are typically performed offline and can be slow, limiting their applicability for large-scale phenotyping or real-time agricultural monitoring.

In contrast, active methods aim to autonomously guide data collection to improve reconstruction efficiency and completeness. A common approach in agricultural environments is next-best-view (NBV) planning [7], [10], [22], [23], where a robot or imaging system actively selects the next camera viewpoint to maximize information gain. Most of of these approaches rely on occupancy grid representations, such as Octomap [24], to model known and unknown regions of the environment. Octomap provides a probabilistic 3D voxel-based map that facilitates exploration and collision-free navigation. However, its main limitation lies in the raycasting process, which becomes computationally expensive as the map resolution increases. Building high-resolution octomaps for complex plant structures, where fine details such as leaves and stems matter, is therefore often impractical in real-world agricultural deployments.

To address these challenges, recent active mapping approaches have explored the use of Gaussian splatting in domains outside agriculture, such as indoor environments [25], [26]. Gaussian splatting provides a continuous and differentiable scene representation that encodes not only geometric information but also photometric attributes, enabling higher-quality reconstructions and realistic rendering. However, a fundamental limitation is that Gaussian splatting does not naturally represent unknown space, which is critical for exploration and view planning. Some works have attempted to overcome this by using high-uncertainty Gaussians as frontiers to approximate unexplored regions. While this strategy introduces a way to reason about unknown space, it can also lead to ambiguities, since high uncertainty in Gaussians may stem from noise in sensor data rather than genuine unexplored areas.

A few recent works have demonstrated that combining occupancy-based maps with Gaussian splatting can leverage the strengths of both representations [27]â[29]. In particular, a low-resolution octomap can be used for exploration, frontier detection, and collision-free navigation, while the Gaussian splatting representation captures fine-grained scene details with high fidelity. Such hybrid methods have been applied primarily in indoor environments, where conditions are relatively structured and sensors can provide accurate depth and semantic information. However, none of these approaches have explored target-aware mapping in horticultural environments, where occlusions, thin plant structures, and noisy sensor measurements present unique challenges. In this paper, we extend the use of this hybrid representation to targetaware mapping in agricultural environments, considering the noisy characteristics of segmentation models.

## III. METHODS

We aim to build a semantic 3D reconstruction of plants from RGBD observations, focusing on a specific target (e.g., fruits). Fig. 2 presents an overview of our system. The system receives RGBD images from the manipulatorâs tip camera, and outputs a sequence of viewpoints that progressively improve the reconstruction of the target semantics. At each observation, a semantic extractor predicts pixel-level class labels. These semantic images and depth data are used to update two complementary representations: a low-resolution semantic octomap to maintain occupancy information, and a dense semantic 3DGS model for detailed reconstruction. The gaussian representation enables identifying target regions by clustering semantic gaussians. A viewpoint sampling, filtering and evaluation module takes these semantic clusters, sample multiple viewpoints and evaluate them based on the occupancy information. Finally, a graph-based planner selects and executes a subset of viewpoints that balance information gain and actuation cost. The following subsections describe each module in detail.

## A. Semantic extractor

The semantic extractor takes the RGB image as input and outputs a per-pixel semantic label and segmentation confidence. We consider the semantic set $\begin{array} { r l } { s } & { { } = } \end{array}$ {fruits, leaves, background}. For simulated experiments, we use color-based segmentation; in real-world tests, the module can be replaced by any learned segmentation network. The resulting semantic masks are used to update both the lowresolution octomap and the semantic GS model, ensuring consistency between the coarse and dense representations.

## B. Low-resolution Octomap

Octomap [24] has been widely used for active mapping and exploration. In our system, we utilize it primarily for viewpoint planning rather than detailed mapping. This allows us to maintain a low-resolution octomap, which is computationally efficient and sufficient for our planning needs. We use a semantic extension [30] that takes RGBD observations and semantic labels as input and updates the occupancy and class probability of the environment. Specifically, Semantic Octomap creates a probabilistic semantic octree where each voxel x encodes a categorical distribution $p _ { c } ( x )$ over different classes as well as a binary occupancy probability value $p o ( x )$ . This semantic octomap enables the computation of a semantics-aware information gain for viewpoint evaluation, as in [7], and collision checking during planning

## C. Semantic Gaussian Splatting

We represent the environment as a collection of 3D gaussians, each parameterized by position $\mu \in \mathbb { R } ^ { 3 }$ , radius $r \in \mathbb { R } ,$ , color $\mathbf { c } \in \{ R , G , B \}$ , opacity $o \in [ 0 , 1 ]$ , and semantic label $\mathbf { s } \in { \mathcal { S } }$ . These parameters are optimized by minimizing the photometric, depth, and semantic discrepancies between rendered and input RGBD images. Our approach builds upon SplaTAM [31] and SGS-SLAM [32], two works focused on 3DGS-based SLAM. Similar to these works, we use isotropic 3D Gaussians defined as follows:

$$
\begin{array} { r } { f ( \mathbf { x } ) = o \exp \left( - \frac { \left\| \mathbf { x } - \pmb { \mu } \right\| ^ { 2 } } { 2 r ^ { 2 } } \right) } \end{array}\tag{1}
$$

where $\mathbf { x } \in \mathbb { R } ^ { 3 }$ is a 3D point in space.

<!-- image-->  
Fig. 2. System overview. Our framework integrates two semantic representations: a low-resolution octomap for collision-free planning and viewpoint evaluation, and a 3DGS representation for high-fidelity reconstruction. Candidate exploitation viewpoints are sampled around semantic clusters, while exploration viewpoints are sampled along crop rows. A graph-based planner determines the optimal sequence of viewpoints to execute, considering information gain and actuation cost.

Color images $\mathbf { C } ( \mathbf { p } )$ are rendered by alpha-compositing the 2D projections of all gaussians onto the image plane, as follows:

$$
\begin{array} { r } { C ( \mathbf { p } ) = \sum _ { i = 1 } ^ { N } \mathbf { c _ { i } } f _ { 2 D _ { i } } ( \mathbf { p } ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - f _ { 2 D _ { j } } ( \mathbf { p } ) ) } \end{array}\tag{2}
$$

where $\mathbf { c _ { i } }$ is the color value of the ith gaussian, $f _ { 2 D _ { j } } ( \mathbf { p } )$ is the value of $f$ projected onto the 2D image plane at the pixel ${ \bf p } = ( u , v )$ , leveraging the camera intrinsic matrix and the camera pose. Semantic, depth and silhouette (visibility) images are rendered analogously by replacing $\mathbf { c _ { i } }$ in equation 2 by $\mathbf { s _ { i } } ,$ depth value $\mathbf { z _ { i } }$ and 1.0 respectively.

We assume in this work that accurate camera pose estimation is available, which, to a certain extent, is ensured by the manipulatorâs forward kinematics. Therefore, unlike [31] and [32], we do not use 3DGS for localization, as it requires continuous camera tracking and thus introduces additional computational overhead. In addition, the following implementation choices are crucial to make our approach efficient and robust to sensor noise.

Initialization. For each new RGBD observation, new gaussians are initialized using depth data. Unlike previous works that add a gaussian per pixel, we apply semantic-aware densification: target semantics receive finer sampling, while non-target semantics are downsampled by 90%, substantially reducing memory use and improving scalability.

Mapping. Similar to [31], [32], we jointly optimize depth, color, and semantic consistency using the mapping loss:

$$
\begin{array} { r } { L _ { m } = \sum _ { \mathbf { p } } \lambda _ { 1 } L _ { d } ( \mathbf { p } ) + \lambda _ { 2 } L _ { c } ( \mathbf { p } ) + \lambda _ { 3 } L _ { s } ( \mathbf { p } ) } \end{array}\tag{3}
$$

With:

$$
\begin{array} { c } { { L _ { d } ( \mathbf { p } ) = \left| D ( \mathbf { p } ) - D _ { g t } ( \mathbf { p } ) \right| } } \\ { { \ } } \\ { { L _ { c } ( \mathbf { p } ) = \alpha { \left| C ( \mathbf { p } ) - C _ { g t } ( \mathbf { p } ) \right| } + ( 1 - \alpha ) ( 1 - S S I M ( C ( \mathbf { p } ) , C _ { g t } ( \mathbf { p } ) ) ) } } \\ { { \ } } \\ { { L _ { s } ( \mathbf { p } ) = \left| c o n f ^ { 2 } ( \mathbf { p } ) \left( S ( \mathbf { p } ) - S _ { g t } ( \mathbf { p } ) \right) \right| } } \end{array}\tag{}
$$

where $\lambda _ { 1 } , \lambda _ { 1 } , \lambda _ { 1 }$ , and Î± are weighting hyperparameters summarized in table I, and $C _ { g t } , \ D _ { g t }$ and $S _ { g t }$ are the input color, depth and semantic images respectively. Note that unlike previous works, we modify the semantic loss to account for the segmentation confidence $c o n f ^ { 2 } ( \mathbf { p } )$ of the semantic extractor, thereby reducing the effect of segmentation noise.

## D. Viewpoint Sampling, Filtering and Evaluation

To identify target regions, we apply DBSCAN clustering to gaussians with the target semantic label. We then generate viewpoint candidates around the identified semantic clusters (exploitation viewpoints) and around crop rows (exploration viewpoints). Exploitation viewpoints are sampled uniformly on a sphere around each cluster, while exploration viewpoints are sampled on planes parallel to the crop rows, leveraging prior knowledge of row spacing and plant height. In this way, we avoid computing frontier voxels, potentially reducing planning time as shown in [7].

Since many viewpoints may be unreachable due to the manipulatorâs limited workspace, we move some of them along their z-axis until they lie within the reachable space. To this end, we discretize the manipulatorâs joint space and compute forward kinematics to obtain feasible 6-DoF end-effector poses. The intersection between a candidate viewpoint and the reachable workspace is determined using the Nearest Neighbor algorithm. In this way, we ensure that all candidate viewpoints are executable.

Finally, we leverage the occupancy map to evaluate the information gain of each viewpoint, using the Unknown Voxel Count (UVC) metric for exploration viewpoints [33] and the Occlusion- and Semantic-Aware Multi-Class Entropy with Proximity Count (OSAMCEP) metric [7] for exploitation viewpoints. Each subset of viewpoints is normalized by its maximum information value, and the top K candidate viewpoints are selected to construct a graph.

## E. Graph-based planning

Inspired by [34], once we have a set of viewpoints with their corresponding information gains, we insert each viewpoint and its associated joint configuration as a node in a graph. We connect each node to $N _ { n e a r }$ neighboring nodes based on proximity in the manipulatorâs joint space. In this way, we encourage the manipulator to perform smooth and efficient transitions between viewpoints.

We use a best-first path search algorithm to find the next sequence of viewpoints to be executed. The algorithm employs a priority queue (initialized with the current camera pose) to store and retrieve nodes based on their utility values. At each step, the node with the highest utility is extracted from the priority queue and expanded. When unexpanded neighbors are discovered, their utility is updated as the sum of their predecessorâs utility and their own, and they are inserted into the priority queue. This process continues until no nodes remain in the queue. Finally, the optimal path is obtained by backtracking from the highest-utility node to the starting position. We execute the top $K _ { e x e c }$ viewpoints along this best path before replanning.

## F. Control

We use the ROS Moveit package to execute manipulation viewpoints following collision-free trajectories.

## IV. EXPERIMENTS

TABLE I  
PARAMETERS USED DURING EVALUATION
<table><tr><td>Category</td><td>Parameter</td><td>Value</td><td>Description</td></tr><tr><td rowspan="2">Low resolution Octomap</td><td>Î´S [m]</td><td>0.05</td><td>Map resolution</td></tr><tr><td>max_range [m]</td><td>1.0</td><td>Max depth range for mapping</td></tr><tr><td rowspan="2">Segmentation</td><td> $P _ { g t }$ </td><td>0.7</td><td>Probability of correct classification during semantic segmentation</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="3">Viewpoint Saampling</td><td> $r \ [ \mathrm { m } ]$  NÏ</td><td>0.4 10</td><td>Radius of sphere for viewpoint sampling Number of azimuth samples</td></tr><tr><td> $\overset { \cdot } { N _ { \theta } }$ </td><td>5</td><td>Number of elevation samples</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="3">Graph-based planning</td><td> $K$ </td><td>20</td><td>Number of viewpoints for graph Number of neighbors per node</td></tr><tr><td> $N _ { n e a r }$   $K _ { e x e c }$ </td><td>4 4</td><td>Viewpoints executed before replanning</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="2">DBSCAN</td><td> $\epsilon \ [ \mathrm { m } ]$  min â samples [m]</td><td>0.02 10</td><td>Radius of a neighborhood Minimum points per cluster</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="4">Gaussian Splatting</td><td> $T _ { s } ~ [ \mathrm { m } ]$ </td><td>0.9 1.0</td><td>Silhouette threshold Weight for depth loss</td></tr><tr><td> $\lambda _ { 1 }$   $\bar { \lambda _ { 2 } }$ </td><td>0.5</td><td>Weight for rgb loss</td></tr><tr><td> $\lambda _ { 3 }$ </td><td>0.1</td><td>Weight for semantic loss</td></tr><tr><td> $_ \alpha$ </td><td>0.8</td><td>Second weight for rgb loss</td></tr></table>

Simulation environment. We validate our approach entirely in simulation to ensure the availability of high-quality ground truth data. To this end, we construct a realistic horticultural environment in the Gazebo simulator. The simulated scene comprises six crop rows, including two variants of bell pepper plants and one variant of tomato plants. These plant models exhibit variations in appearance, height, cluster size, and occlusion characteristics, as illustrated in Fig. 1.

To increase environmental diversity, we generate five replicas of this setup by randomizing the orientation of the plant models. For each crop row consisting of five plants each, we define a fixed sequence of four waypoints that the mobile manipulator follows during all trials, thereby minimizing the influence of navigation variability. The robot performs mapping operations along each row, starting from the left side, turning at the end, and proceeding along the right side. At each waypoint, the robot selects and executes up to ten successful viewpoints for data acquisition. We consider three semantics: fruits, leaves and background, with fruits being the target of interest for mapping. We simulate segmentation noise by assigning each pixel the correct label with probability $P ,$ and an incorrect label with probability 1 â P . All simulations are conducted on an NVIDIA Jetson AGX Orin platform equipped with 32 GB of unified memory.

Robot setup. Our mobile manipulator consists of 6DoF Xarm Lite6 Manipulator mounted on the Terrasentia robot. A 640x480 RGBD camera is attached to the manipulatorâs end effector, with the camera axis aligned to the rotation axis of the last joint.

Baselines. We compare our approach against a purely Octomap-based active mapping method, which has been the predominant representation for mapping in horticultural environments [7], [22], [34]. Within our modular framework, this baseline is implemented by removing the 3D Gaussian Splatting module while retaining the semantic octomap, configured with a higher spatial resolution for mapping. The resulting octomap is employed to identify fruit clusters and to evaluate candidate viewpoints. In our analysis, we consider two voxel resolutions: 0.01 m and 0.015 m. The rest of the modules are kept the same for a fair comparison.

Metrics. We aim to evaluate the accuracy and efficiency of our approach and the baselines on 3D reconstruction of fruits. We use standard metrics commonly employed to assess 3D reconstruction accuracy, namely the Chamfer Distance (CD), Precision $( P )$ , Recall (R), and F1 Score $( F _ { 1 } )$ . In addition, we compute two task-oriented metrics relevant to horticultural applications: fruit volume accuracy and fruit count accuracy, both of which are particularly useful for phenotyping and yield estimation. Finally, we report the average runtime per crop row as an indicator of computational efficiency.

The Chamfer Distance quantifies the geometric discrepancy between the reconstructed fruit point cloud P and the ground-truth fruit point cloud Q as

$$
\mathrm { C D } ( \mathcal { P } , \mathcal { Q } ) = \frac { 1 } { | \mathcal { P } | } \sum _ { p \in \mathcal { P } } \operatorname* { m i n } _ { q \in \mathcal { Q } } | p - q | _ { 2 } + \frac { 1 } { | \mathcal { Q } | } \sum _ { q \in \mathcal { Q } } \operatorname* { m i n } _ { p \in \mathcal { P } } | q - p | _ { 2 } .
$$

<!-- image-->  
Fig. 3. Sample reconstructions of two variants of bell pepper rows. Our 3DGS-based approach produces denser and more complete fruit reconstructions compare to high-resolution octomaps.

To evaluate point-level correspondence accuracy, we define Precision and Recall as

$$
P = \frac { | p \in \mathcal { P } : d ( p , \mathcal { Q } ) < \tau | } { | \mathcal { P } | } , \quad R = \frac { | q \in \mathcal { Q } : d ( q , \mathcal { P } ) < \tau | } { | \mathcal { Q } | } ,\tag{5}
$$

where $d ( x , y )$ denotes the minimum Euclidean distance from point x to the set $\mathcal { V } ,$ , and Ï is a fixed distance threshold. We set Ï at 0.015 m, the coarsest octomap resolution considered in this study. The harmonic mean of Precision and Recall yields the F1 Score:

$$
F _ { 1 } = 2 \times \frac { P \times R } { P + R } .\tag{6}
$$

In practice, Recall represents fruit coverage, and it is a good indicator of our methodâs ability to find viewpoints that reveal fruit areas despite the self-occlusion characteristics of plants. For fruit-related metrics, we first identify fruit clusters by applying the DBSCAN algorithm to the Gaussian centers whose semantic labels correspond to fruits. The fruit volume for each cluster is estimated as the volume of the convex hull formed by the 3D points within the cluster. Fruit count accuracy is then determined by comparing the number of detected clusters with the ground-truth fruit count.

Finally, the runtime per row measures the total computation time required for the robot to complete mapping along a single crop row, reflecting the overall efficiency of the proposed system.

## V. RESULTS

## A. Fruit Reconstruction Accuracy

Table II shows accuracy and completeness metrics for our approach and Semantic Octomap with two different resolutions. Our approach exhibits high precision and recall values (greater than 0.89), even in the presence of segmentation noise. The semantic octomap with a resolution of 0.01 m achieves performance close to ours under ideal conditions, but its accuracy drops significantly when segmentation noise is introduced. The higher precision and recall values of our method can be attributed to the dense reconstruction provided by 3DGS (clearly noticeable in Fig. 3), which does not suffer from the quantization effects inherent in octomap representations. Moreover, the robustness to segmentation noise can be attributed to our robust mapping loss, which accounts for color, depth, semantics, and detection confidence. This observation is supported by Table IV, where we compare the same metrics with and without the confidence term in the segmentation loss (Equation 4). Further discussion is presented in Section V-D.

## B. Fruit Volume and Count Estimation

Table III shows the fruit volume and count accuracy averaged over six crop rows. Our approach achieves highly accurate volume and count estimations, with low variance across different crop types. In contrast, the semantic octomap reconstruction provides poor accuracy in volume estimation, which aligns with previous findings [10]. This indicates that even a resolution of 0.01 m is insufficient to accurately represent small objects such as fruits.

Furthermore, when segmentation noise is considered, the fruit volume and count estimation accuracy remains above 82% when using our approach, but is severely affected when using Octomap. This can be explained by the fact that our method maintains an accurate reconstruction of the environment, enabling reliable sampling of new viewpoints. In contrast, although Semantic Octomap performs Bayesian updates with each new observation, its current implementation does not support confidence weighting for individual detections and instead assumes a uniform confidence model.

## C. Runtime

We decompose the total runtime into four main components: Octomap mapping, 3DGS mapping, viewpoint planning (including viewpoint sampling, filtering, and evaluation), and viewpoint execution (including collision-free trajectory planning and execution time). Figure 4 shows these results averaged over six crop rows. The runtime of our approach is slightly higher than that of a 0.015 m resolution octomap. However, our Gaussian Splatting module runs fully in parallel with the other tasks. In contrast, when relying solely on Octomap, viewpoint evaluation requires waiting for the latest octomap update before initiating a new planning iteration. As high-resolution octomaps require longer update and ray-casting times for viewpoint evaluation, the mapping and planning stages are significantly slower for the highestresolution octomap.

## D. Ablation Studies

In this section, we address the following questions: (i) How much do exploitation and exploration viewpoints contribute to reconstruction accuracy? (ii) How much does our proposed segmentation loss improve reconstruction under segmentation noise? (iii) How much memory do we save by downsampling irrelevant semantics? Table IV presents reconstruction metrics for: (i) our full pipeline under segmentation noise, (ii) our pipeline without the confidence term in Equation 4, and (iii) our pipeline using exploration viewpoints only, without viewpoint sampling around fruit clusters.

When segmentation confidence is not considered, there is a precision drop of about 5% and a recall drop of 15%. This can be explained by noisy clusters leading to suboptimal viewpoint sampling. Intuitively, an inaccurate reconstruction results in low-quality candidate viewpoints, which in turn yield an incomplete reconstruction. Furthermore, when only exploration viewpoints are executed, we still achieve high precision values, but fruit coverage (recall) decreases by approximately 9% compared to the full pipeline. This indicates that our active, semantics-driven approach improves fruit coverage by 9% under noisy conditions.

Finally, GPU memory consumption experiments without downsampling revealed that our pipeline reaches a peak of 12.4 GB when mapping a single row containing five plants. Each file storing the 3DGS parameters has an average size of 140 MB. In contrast, using a downsampling factor of 0.9 (as in previous experiments) reduces the peak GPU memory usage to 4.6 GB per row and the parameter file size to approximately 20 MB. This indicates that our downsampling strategy for irrelevant semantics reduces GPU memory consumption by about 2.7Ã and produces a 3DGS representation that is roughly 7Ã more compact.

<!-- image-->  
Fig. 4. Runtime broken down into Octomap mapping, Gaussian Splatting mapping, viewpoint planning, and viewpoint execution. Overall, our method requires approximately half the runtime of a high-resolution octomap baseline.

## VI. LIMITATIONS

Our approach has demonstrated significant improvements in achieving accurate and complete reconstructions of fruits in horticultural environments. However, it has so far been evaluated only in simulation. Real-world experiments are necessary to fully validate its performance under practical conditions. The authors anticipate several challenges when deploying the system in real-world environments, including limited localization accuracy, depth measurement noise, and dynamic elements such as moving leaves or fruits caused by wind or manipulator interactions. These factors can degrade both the quality of the reconstruction and the reliability of viewpoint planning, and they remain open research problems in agricultural robotics.

Moreover, all experiments in this work have focused on fruits as the primary semantic target. Extending the method to other semantic classes, such as stems or peduncles, which are typically thinner and smaller than fruits, may require additional parameter tuning and adaptations to the viewpoint sampling strategy. Nonetheless, Gaussian Splatting has shown great promise for modeling small and detailed objects, suggesting that our framework could be effectively extended to other semantic categories.

## VII. CONCLUSION

We presented an approach for active semantic mapping in horticultural environments that combines a low-resolution octomap with a 3D Gaussian Splatting representation. Our method significantly improves reconstruction accuracy, enabling precise estimation of phenotyping traits such as fruit volume and fruit count. A carefully designed loss function enhances robustness to segmentation noise. Moreover, our approach is not resolution-dependent and runs considerably faster than high-resolution octomaps. Overall, we found that 3DGS-based mapping shows great promise for reconstructing horticultural environments and can be seamlessly integrated into robotic frameworks, providing a powerful tool for highthroughput phenotyping.

## REFERENCES

[1] M. Padhiary, A. Kumar, and L. N. Sethi, âEmerging technologies for smart and sustainable precision agriculture,â Discover Robotics, vol. 1, no. 1, p. 6, 2025.

[2] N. Harandi, B. Vandenberghe, J. Vankerschaver, S. Depuydt, and A. Van Messem, âHow to make sense of 3d representations for plant phenotyping: a compendium of processing and analysis techniques,â Plant Methods, vol. 19, no. 1, p. 60, 2023.

[3] W. Dong, P. Roy, and V. Isler, âSemantic mapping for orchard environments by merging two-sides reconstructions of tree rows,â Journal of Field Robotics, vol. 37, no. 1, pp. 97â121, 2020.

[4] Z. Wang, S. Xiao, Z. Miao, R. Liu, H. Chen, Q. Wang, K. Shao, R. Wang, and Y. Ma, âP3dfusion: A cross-scene and high-fidelity 3d plant reconstruction framework empowered by vision foundation models and 3d gaussian splatting,â European Journal of Agronomy, vol. 171, p. 127811, 2025.

[5] J. Dong, J. G. Burnham, B. Boots, G. Rains, and F. Dellaert, â4d crop monitoring: Spatio-temporal reconstruction for agriculture,â in 2017 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2017, pp. 3878â3885.

[6] Y. Pan, F. Magistri, T. Labe, E. Marks, C. Smitt, C. McCool, J. Behley, Â¨ and C. Stachniss, âPanoptic mapping with fruit completion and pose estimation for horticultural robots,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 4226â4233.

[7] J. Cuaran, K. S. Ahluwalia, K. Koe, N. K. Uppalapati, and G. Chowdhary, âActive semantic mapping with mobile manipulator in horticultural environments,â in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 12 716â12 722.

[8] H. Freeman and G. Kantor, âAutonomous apple fruitlet sizing with next best view planning,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 15 847â15 853.

[9] C. Lehnert, D. Tsai, A. Eriksson, and C. McCool, â3d move to see: Multi-perspective visual servoing towards the next best view within unstructured and occluded environments,â in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019, pp. 3890â3897.

[10] T. Zaenker, C. Smitt, C. McCool, and M. Bennewitz, âViewpoint planning for fruit size and position estimation,â in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 3271â3277.

[11] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[12] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[13] C. Maraveas, âImage analysis artificial intelligence technologies for plant phenotyping: current state of the art,â AgriEngineering, vol. 6, no. 3, pp. 3375â3407, 2024.

[14] P. Gao, J. Jiang, J. Song, F. Xie, Y. Bai, Y. Fu, Z. Wang, X. Zheng, S. Xie, and B. Li, âCanopy volume measurement of fruit trees using robotic platform loaded lidar data,â IEEE Access, vol. 9, pp. 156 246â 156 259, 2021.

[15] S. Paulus, âMeasuring crops in 3d: using geometry for plant phenotyping,â Plant methods, vol. 15, no. 1, p. 103, 2019.

TABLE II  
RECONSTRUCTION METRICS
<table><tr><td>Method</td><td>Chamfer Distanceâ [m]</td><td>Precisionâ</td><td>Recallâ</td><td>F1 Scoreâ</td></tr><tr><td>Ours - No noise</td><td>0.010</td><td>0.987</td><td>0.944</td><td>0.965</td></tr><tr><td>Semantic Oct. Res: 0.015m - No noise</td><td>0.021</td><td>0.946</td><td>0.764</td><td>0.840</td></tr><tr><td>Semantic Oct. Res: 0.01m - No noise</td><td>0.016</td><td>0.954</td><td>0.853</td><td>0.899</td></tr><tr><td>Ours with seg. noise</td><td>0.014</td><td>0.978</td><td>0.891</td><td>0.931</td></tr><tr><td>Semantic Oct. Res: 0.015m with seg. noise</td><td>0.108</td><td>0.476</td><td>0.508</td><td>0.481</td></tr><tr><td>Semantic Oct. Res: 0.01m with seg. noise</td><td>0.082</td><td>0.583</td><td>0.753</td><td>0.645</td></tr></table>

TABLE III

PHENOTYPING METRICS ACROSS SIX CROP ROWS.
<table><tr><td>Method</td><td>Fruit volume Accuracy [%]</td><td>Fruit count Accuracy[%]</td></tr><tr><td>Ours - No noise</td><td> $\mathbf { 1 0 0 . 5 6 \pm 1 3 . 8 7 }$ </td><td> $\mathbf { 9 4 . 5 5 \pm 1 0 . 7 1 }$ </td></tr><tr><td>Semantic Oct. Res: 0.015m - No noise</td><td> $8 5 . 2 3 \pm 3 2 . 6 6$ </td><td> $9 1 . 2 0 \pm 1 2 . 6 0$ </td></tr><tr><td>Semantic Oct. Res: 0.01m - No noise</td><td> $1 4 4 . 7 3 \pm 1 9 . 4 5$ </td><td> $8 9 . 6 8 \pm 1 6 . 2 3$ </td></tr><tr><td>Ours with seg. noise</td><td> ${ \bf 8 2 . 9 0 \pm 7 . 9 3 }$ </td><td> ${ \bf 9 0 . 1 5 \pm 7 . 7 6 }$ </td></tr><tr><td>Semantic Oct. Res: 0.015m with seg. noise</td><td> $4 9 . 6 4 \pm 2 7 . 9 9$ </td><td> $7 7 . 7 6 \pm 2 0 . 5 3$ </td></tr><tr><td>Semantic Oct. Res: 0.01m with seg. noise</td><td> $1 4 2 . 9 3 \pm 4 5 . 0 9$ </td><td> $1 8 4 . 3 1 \pm 1 0 3 . 1 1$ </td></tr></table>

TABLE IV

ABLATION RESULTS. RECONSTRUCTION METRICS
<table><tr><td>Method</td><td>Chamfer Distanceâ [m]</td><td>Precisionâ</td><td>Recallâ</td><td>F1 Scoreâ</td></tr><tr><td>Ours with seg. noise</td><td>0.014</td><td>0.978</td><td>0.891</td><td>0.931</td></tr><tr><td>Ours with seg. noise - no conf.</td><td>0.030</td><td>0.926</td><td>0.740</td><td>0.818</td></tr><tr><td>Ours with seg. noise - Explor. only</td><td>0.024</td><td>0.956</td><td>0.799</td><td>0.867</td></tr></table>

[16] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â 2020.

[17] D. Zhang, J. Gajardo, T. Medic, I. Katircioglu, M. Boss, N. Kirchgessner, A. Walter, and L. Roth, âWheat3dgs: In-field 3d reconstruction, instance segmentation and phenotyping of wheat heads with gaussian splatting,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5360â5370.

[18] L. A. Stuart, D. M. Wells, J. A. Atkinson, S. Castle-Green, J. Walker, and M. P. Pound, âHigh-fidelity wheat plant reconstruction using 3d gaussian splatting and neural radiance fields,â GigaScience, vol. 14, p. giaf022, 2025.

[19] J. Chen, Y. Jiao, F. Jin, X. Qin, Y. Ning, M. Yang, and Y. Zhan, âPlant sam gaussian reconstruction (psgr): A high-precision and accelerated strategy for plant 3d reconstruction,â Electronics, vol. 14, no. 11, p. 2291, 2025.

[20] A. McAfee, T. Pluck, R. Dahyot, and G. Lacey, âEvaluation of 3d gaussian splatting in plant reconstruction,â IMVIP 2025, p. 26, 2025.

[21] T. Ojo, T. La, A. Morton, and I. Stavness, âSplanting: 3d plant capture with gaussian splatting,â in SIGGRAPH Asia 2024 Technical Communications, 2024, pp. 1â4.

[22] A. K. Burusa, E. J. van Henten, and G. Kootstra, âAttention-driven active vision for efficient reconstruction of plants and targeted plant parts,â arXiv preprint arXiv:2206.10274, 2022.

[23] R. Menon, T. Zaenker, N. Dengler, and M. Bennewitz, âNbv-sc: Next best view planning based on shape completion for fruit mapping and reconstruction,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 4197â4203.

[24] A. Hornung, K. M. Wurm, M. Bennewitz, C. Stachniss, and W. Burgard, âOctomap: An efficient probabilistic 3d mapping framework based on octrees,â Autonomous robots, vol. 34, no. 3, pp. 189â206, 2013.

[25] Y. Tao, D. Ong, V. Murali, I. Spasojevic, P. Chaudhari, and V. Kumar, âRt-guide: Real-time gaussian splatting for information-driven exploration,â arXiv preprint arXiv:2409.18122, 2024.

[26] Y. Li, Z. Kuang, T. Li, Q. Hao, Z. Yan, G. Zhou, and S. Zhang, âActivesplat: High-fidelity scene reconstruction through active gaussian splatting,â IEEE Robotics and Automation Letters, 2025.

[27] L. Jin, X. Zhong, Y. Pan, J. Behley, C. Stachniss, and M. Popovic,Â´ âActivegs: Active scene reconstruction using gaussian splatting,â IEEE Robotics and Automation Letters, 2025.

[28] R. Jin, Y. Gao, Y. Wang, Y. Wu, H. Lu, C. Xu, and F. Gao, âGs-planner:

A gaussian-splatting-based planning framework for active high-fidelity reconstruction,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 11 202â11 209.

[29] Z. Xu, R. Jin, K. Wu, Y. Zhao, Z. Zhang, J. Zhao, F. Gao, Z. Gan, and W. Ding, âHgs-planner: Hierarchical planning framework for active scene reconstruction using 3d gaussian splatting,â in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 14 161â14 167.

[30] A. Asgharivaskasi and N. Atanasov, âSemantic octree mapping and shannon mutual information computation for robot exploration,â IEEE Transactions on Robotics, 2023.

[31] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat, track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[32] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, âSgs-slam: Semantic gaussian splatting for neural dense slam,â in European Conference on Computer Vision. Springer, 2024, pp. 163â 179.

[33] J. Delmerico, S. Isler, R. Sabzevari, and D. Scaramuzza, âA comparison of volumetric information gain metrics for active 3d object reconstruction,â Autonomous Robots, vol. 42, no. 2, pp. 197â208, 2018.

[34] T. Zaenker, J. Ruckin, R. Menon, M. Popovi Â¨ c, and M. Bennewitz, Â´ âGraph-based view motion planning for fruit detection,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 4219â4225.