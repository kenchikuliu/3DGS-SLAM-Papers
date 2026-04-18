# DrivingGaussian++: Towards Realistic Reconstruction and Editable Simulation for Surrounding Dynamic Driving Scenes

Yajiao XiongÂ§, Xiaoyu ZhouÂ§, Yongtao Wang\*, Deqing Sunâ¡, Ming-Hsuan Yangâ¡

AbstractâWe present DrivingGaussian++, an efficient and effective framework for realistic reconstructing and controllable editing of surrounding dynamic autonomous driving scenes. DrivingGaussian++ models the static background using incremental 3D Gaussians and reconstructs moving objects with a composite dynamic Gaussian graph, ensuring accurate positions and occlusions. By integrating a LiDAR prior, it achieves detailed and consistent scene reconstruction, outperforming existing methods in dynamic scene reconstruction and photorealistic surround-view synthesis. DrivingGaussian++ supports training-free controllable editing for dynamic driving scenes, including texture modification, weather simulation, and object manipulation, leveraging multi-view images and depth priors. By integrating large language models (LLMs) and controllable editing, our method can automatically generate dynamic object motion trajectories and enhance their realism during the optimization process. DrivingGaussian++ demonstrates consistent and realistic editing results and generates dynamic multi-view driving scenarios, while significantly enhancing scene diversity. More results and code can be found at the project site: https://drivinggaussian-plus.github.io/DrivingGaussian plus.github.io

Index Termsâ3D Simulation, Autonomous Driving, Gaussian Splatting, Controllable Editing

## 1 INTRODUCTION

W ITH the rapid advances of autonomous driving tech-nologies, the availability of driving data has intro- duced new opportunities to enhance autonomous driving simulation. These datasets encapsulate rich semantic and spatial information, significantly improving the performance of downstream perception tasks. Among these, 3D scene editing is crucial in enhancing the data of autonomous driving. It facilitates the generation of diverse driving scenarios, which is particularly valuable in synthesizing complex and rare cases critical for testing and validation. Using 3D scene editing, it becomes possible to simulate diverse real-world driving conditions, thereby enhancing the robustness and safety of autonomous driving systems.

The 3D scene editing task encompasses a variety of components, including style transfer, motion modification, weather simulation, and the addition or removal of objects. However, due to the inherent diversity and complexity of these tasks, existing 3D scene editing methods [1]â[6] are often highly specialized, lacking a framework capable of comprehensively addressing multiple editing tasks. While the widely adopted strategy of leveraging 2D editing techniques achieves superior results, it requires repeated 2D editing during training to maintain multi-view consistency. This significantly increases computational costs, making it impractical for large-scale autonomous driving scenarios.

<!-- image-->  
Fig. 1: DrivingGaussian++ achieves favorable trade-offs between speed, accuracy, and task diversity. Our method enables fast, training-free execution of various 3D controllable editing tasks while delivering comparable or better performance to state-of-the-art task-specific models.

Representing and modeling large-scale dynamic scenes is the foundation for 3D scene editing and contributes to a series of autonomous driving tasks. Unfortunately, reconstructing complex 3D scenes from sparse vehicle-mounted sensor data is challenging, particularly when the ego vehicle is moving at high speeds. Furthermore, it becomes even more challenging in multi-camera settings due to their outward views, minimal overlaps, and variations in light from different directions. Complex geometry, diverse optical degradation, and spatiotemporal inconsistency also pose significant challenges in modeling such a 360-degree, largescale driving scene.

Neural radiance fields [7] (NeRFs) have recently emerged as a promising method for modeling scenery at the object or room level. Some recent studies [8]â[11] have extended NeRF to large-scale, unbounded static scenes and multiple dynamic objects [12], [13]. However, NeRFbased methods are computationally intensive and require densely overlapping views and consistent lighting. These issues affect their ability to construct driving scenes with outward multi-camera setups at high speeds. Furthermore, network capacity limitations make it challenging for them to model long-term, dynamic scenes with multiple objects, leading to visual artifacts and blurring. The 3D Gaussian Splatting (3DGS) [14] methods represent scenes with explicit 3D Gaussian representation and achieve state-of-the-art performance in novel view synthesis. However, the original 3DGS still encounters significant challenges in modeling large-scale dynamic driving scenes due to fixed Gaussians and constrained representation capacity.

In this paper, our method is based on a framework that represents the surrounding and dynamic driving scenes. The key idea is to hierarchically model the complex driving scene using sequential data from multiple sensors. We adopt Composite Gaussian Splatting to decompose the scene into static background and dynamic objects, reconstructing each part separately. Based on these, global rendering via Gaussian Splatting captures occlusion in the real world, encompassing static backgrounds and dynamic objects. Furthermore, we incorporate a LiDAR prior into the Gaussian representation, which enables the recovery of more precise geometry and maintains better multi-view consistency.

We also introduce a multitask editing framework for autonomous driving scenes. Within this framework, we meticulously implement three key tasks: texture modification, weather simulation, and object manipulation. To mitigate computational and temporal costs, we propose the post-editing strategy, which effectively decouples the reconstruction and editing processes, as illustrated in Figure 2. We compare our method with IN2N [1], IGS2GS [15], and ClimateNeRF [6], which demonstrate superior processing speed and competitive performance. This approach enables editing without additional training, significantly reducing the overall training overhead. DrivingGaussian++ can be seamlessly applied to any pre-reconstructed 3D scene. Specifically, we employ Composite Gaussian Splatting to construct explicit scene representations using Gaussian distributions. By identifying and editing these Gaussians at the 3D level, we circumvent the issue of multi-view inconsistency that often arises when 2D models are directly applied to 3D spaces. Furthermore, we enhance the editing results by integrating advanced image-processing tools, demonstrating the potential of cross-dimensional technological integration in achieving high-quality scene modifications.

Differences with preliminary results published in CVPR 2024. We have extended our work in several aspects: (i) We represent large-scale, dynamic driving scenes based on Composite Gaussian Splatting, which introduces two novel modules, including Incremental Static 3D Gaussians and Composite Dynamic Gaussian Graphs. The former reconstructs the static background incrementally, while the latter models multiple dynamic objects with a Gaussian graph. (ii) We construct a scene editing framework to edit the reconstructed scene in a training-free manner, covering multiple tasks including texture modification, weather simulation, and object manipulation. It contributes to generating novel and realistic simulation data. (iii) We facilitate dynamic editing of driving scenes, which predicts the motion trajectory of particles inserted into the scene. (iv) We construct a foreground asset bank with 3D generation and reconstruction and validate the quality of the data.

<!-- image-->  
Fig. 2: Comparison between the traditional and our editing strategy. Our method differs from previous work in three key aspects: 1) supporting diverse multi-task 3D simulation; 2) enabling training-free controllable editing; 3) demonstrating strong performance in complex dynamic driving environments.

## 2 RELATED WORK

## 2.1 3D Reconstruction

## 2.1.1 Neural Radiance Fields

Rapid progress in neural rendering for novel view synthesis has received significant attention. Neural Radiance Fields (NeRFs), which utilizes multi-layer perceptrons (MLPs) and differentiable volume rendering, can reconstruct 3D scenes and synthesize novel views from a set of 2D images and corresponding camera pose information.

NeRF for Bounded Scene. Typical NeRF models are limited to bounded scenes, requiring a consistent distance between the center object and the camera. NeRF models also struggle with scenes captured with slight overlaps and outward capture methods. Numerous advances have expanded the capabilities of NeRFs, leading to improvements in training speed [16]â[18], pose optimization [19]â[21], scene editing [6], [22], and dynamic scene representation [23], [24]. However, applying NeRF to large-scale unbounded scenes, such as autonomous driving scenarios, remains challenging. NeRF for Unbounded Scenes. For large-scale unbounded scenes, numerous methods [8]â[11], [25] have introduced refined versions of NeRF to model multi-scale urban-level static scenes. Inspired by the mipmapping approach to preventing aliasing, MIP-NeRF models [26], [27] have been developed to account for unbounded scenes. To enable high-fidelity rendering, Xu et al. [28] combine the compact multi-resolution ground feature planes with NeRF for large urban scenes. On the other hand, Guo et al. [29] propose a disentanglement approach that can model unbounded street views but ignores dynamic objects on the road. However, these methods model scenes under the assumption that the scene remains static and face challenges in effectively capturing dynamic elements.

Although most existing NeRF-based methods rely on accurate camera poses, several approaches have been developed [30], [31] to synthesize the view of dynamic monocular videos. However, these methods are limited to forward monocular views and are unable to effectively handle inputs from surrounding multi-camera setups. For dynamic urban scenes, Ost et al. [12], [13] extend NeRF to dynamic scenes with multiple objects using a scene graph, and Wu et al. [32], [33] propose instance-aware realistic simulators for monocular dynamic scenes. On the other hand, Xie et al. [34] improve the parameterization and camera poses of the surrounding views while using LiDAR as additional depth supervision. In addition, several methods [35], [36] decompose the scene into static background and dynamic objects and construct the scene with the help of LiDAR and 2D optical flow.

The quality of views synthesized by NeRF-based methods deteriorates in scenarios with multiple dynamic objects and variations and lighting variations, owing to their dependency on ray sampling. In addition, the utilization of LiDAR is confined to providing auxiliary depth supervision, and its potential benefits in reconstruction, such as delivering geometric priors, have not been explored.

To address these limitations, we utilize Composite Gaussian Splatting to model the unbounded dynamic scenes, where the static background is incrementally reconstructed as the ego vehicle moves, and multiple dynamic objects are modeled and integrated into the entire scene through Gaussian graphs. LiDAR is employed as the initialization for Gaussians, providing a more accurate geometric shape prior and a comprehensive scene description rather than solely serving as depth supervision for images.

## 2.1.2 3D Gaussian Splatting

The recent 3D Gaussian Splatting method [14] represents a static scene with numerous 3D Gaussians and achieves state-of-the-art results in novel view synthesis and training speeds. Compared with existing explicit scene representations (e.g., mesh, voxels), the 3DGS can model complex shapes with fewer parameters. Unlike implicit neural rendering, the 3DGS allows fast rendering and differentiable computation with splat-based rasterization.

Dynamic 3D Gaussian Splatting. While the original 3DGS is designed to represent static scenes, several methods have been developed for dynamic objects/scenes. Given a set of dynamic monocular images, Yang et al. [37] introduce a deformation network to model the motion of Gaussians. In addition, Wu et al. [38] connect adjacent Gaussians via a HexPlane, enabling real-time rendering. However, these two approaches are explicitly designed for monocular singlecamera scenes focused on a central object. Luiten et al. [39] parameterize the entire scene using a set of dynamic Gaussians that evolve. However, it requires a camera array with dense multi-view images as inputs.

In real-world autonomous driving scenes, the highspeed movement of data collection platforms leads to extensive and complex background variations, often captured by sparse views (e.g., 2-4 views). Moreover, fast-moving dynamic objects with intense spatial changes and occlusion further complicate the situation. Collectively, these factors pose significant challenges for existing methods.

## 2.2 3D Scene Cotrollable Editing

Representing and modeling 3D scenes is fundamental for novel view synthesis. Neural Radiation Field (NeRF) and 3D Gaussian Splatting are two prominent methods for 3D scene reconstruction. NeRF implicitly encodes scene geometry and appearance within a multilayer perceptron (MLP), while 3D Gaussian Splatting explicitly represents the scene using 3D Gaussian ellipsoids. Despite the demonstrated reconstruction capabilities, editing these representations remains a significant challenge. Current approaches can be broadly classified into two categories: editing guided by diffusion models and editing based on 3D particle systems.

## 2.2.1 Editing Based on Diffusion Guidance

Diffusion models have gained much attention for their ability to enable text-driven image editing. Recent methods [1], [3]â[5], [40], [41] have extended this capability to 3D scenes using pre-trained diffusion models. These methods add noise to images rendered from 3D models. The noisy images, augmented with additional control conditions, are processed by a 2D diffusion model, which predicts the noise that represents the discrepancy between the target output and the input. This predicted noise is then used to compute the Score Distillation Sampling (SDS) loss, guiding the optimization of the 3D model. Despite their effectiveness, these approaches struggle with maintaining multi-view consistency and managing complex scene dynamics, especially in large-scale environments. Instruct-NeRF2NeRF (IN2N) [1] uses text commands to edit 3D models by transforming the 3D editing task into a 2D image editing problem.

However, due to the inability to ensure consistent editing across surrounding views, this approach suffers from instability, slow processing speeds, and noticeable artifacts, particularly in 360-degree scenes. ViCA-NeRF [5] adopts a similar strategy to IN2N, selecting a subset of reference images from the scene dataset and editing them, while blending the remaining images based on the projection results. Although this blending approach mitigates some issues, it does not resolve consistency problems and often results in blurry edits.

Recently, DreamEditor [40] converts NeRF representations into mesh surfaces and directly optimizes the mesh using SDS loss and DreamBooth. HiFA [41] improves multiview consistency by dynamically adjusting the diffusion timestep and reducing the weight of the noise signal. Gaussian-based approaches [3], [4] extend these NeRF editing techniques to 3D Gaussian Splatting, leveraging depth models to estimate image depth as a geometric prior for the conditional diffusion model. Although these methods achieve more consistent 3D editing, they are still limited to texture modification due to the fixed nature of depth estimation. Additionally, most of them rely on static 2D and 3D masks to constrain editing regions, which are ineffective for dynamic 3D model training. These approaches have been primarily validated on object-centric datasets and remain unexplored in complex driving scenes.

<!-- image-->  
Fig. 3: Overview Pipeline of our method. DrivingGaussian++ facilitates the reconstruction and controllable editing of surrounding dynamic scenes in autonomous driving by leveraging a compositional, controllable 3D representation with a unified global optimization strategy.

In contrast, DrivingGaussian++ employs a training-free paradigm to effectively address the challenges of existing methods in dynamic driving scene editing, achieving exceptional editing consistency and visual quality.

## 2.2.2 Editing Based on 3D Particle Systems.

In addition to text-driven editing guided by diffusion models, numerous domain-specific editing methods operate without relying on additional target images [2], [6]. ClimateNeRF [6] simulates particle-based entities for various weather conditions, such as snow, fog, and floods, and integrates them into the original neural field to achieve realistic weather simulation. Similarly, GaussianEditor and Infusion [2] leverage 3D Gaussian Splatting for controlled editing of 3D scenes. GaussianEditor identifies editing regions by training Gaussians with semantic attributes, enabling precise insertion and deletion at the 3D level. Infusion employs a depth completion model to establish depth information, which serves as a control signal for Gaussian completion. These methods demonstrate higher editing efficiency and superior multi-view consistency compared to diffusion model-guided approaches. Inspired by the recent advances, DrivingGaussian++ adopts 3D particle-level editing and further extends it to multiple tasks, including texture, object, and weather editing. Through a training-free paradigm, our approach achieves explicit, controllable, and efficient editing for large-scale autonomous driving scenes.

## 3 METHOD

Our goal is to achieve training-free editing within 3D autonomous driving scenes. To address multiple editing tasks, we propose a controllable and efficient framework. First, we accurately reconstruct dynamic driving scenes employing Composite Gaussian Splatting. Next, we identify specific Gaussians within the scene for modification or generate new Gaussians to simulate specific physical entities. These targeted Gaussians are then integrated into the original scene, where we predict the future trajectories of the objects. Finally, we refine the results using image processing techniques to enhance realism. Using this framework, we develop detailed editing methodologies for three key tasks:

texture modification, weather simulation, and object manipulation. Our method is described in Fig. 3

## 3.1 Composite Gaussian Splatting

3DGS performs well in static scenes, but has significant limitations in mixed scenes involving large-scale static backgrounds and multiple dynamic objects. As illustrated in Figure 4, our objective is to represent surrounding largescale driving scenes with Composite Gaussian Splatting for unbounded static backgrounds and dynamic objects.

## 3.1.1 LiDAR Prior with surrounding views

The primitive 3DGS attempts to initialize Gaussians via structure-from-motion (SfM). However, unbounded urban scenes for autonomous driving contain many multi-scale backgrounds and foregrounds. Nevertheless, they are only glimpsed through exceedingly sparse views, resulting in erroneous and incomplete recovery of geometric structures.

To provide better initialization for Gaussians, we introduce the LiDAR prior to 3D Gaussian to obtain better geometries and maintain multi-camera consistency in surrounding view registration. At each timestep $t \in T$ , given a set of multi-camera images $\{ I _ { t } ^ { i } | i = 1 \ldots N \}$ collected from the moving platform and multi-frame LiDAR sweeps $L _ { t }$ We minimize multi-camera registration errors using LiDARimage multi-modal data and obtain accurate point positions and geometric priors.

We first merge multiple frames of LiDAR sweeps to obtain the complete point cloud of the scene, denoted as $L .$ We follow Colmap [42] and extract image features $X = x _ { n } ^ { q }$ from each image individually. Next, we project the LiDAR points onto the surrounding images. For each LiDAR point $\hat { l } ,$ we transform its coordinates to the camera coordinate system and match it with the 2D-pixel of the camera image plane through projection:

$$
\mathbf { x _ { p } ^ { q } } = \mathbf { K } [ \mathbf { R _ { t } ^ { i } } \cdot l _ { s } + \mathbf { T _ { t } ^ { i } } ] ,\tag{1}
$$

where $\mathbf { \Delta x _ { p } ^ { q } }$ is the 2D pixel of the image, $\mathbf { I _ { t } ^ { i } } , \ \mathbf { R _ { t } ^ { i } }$ and $\mathbf { T _ { t } ^ { i } }$ are orthogonal rotation matrices and translation vectors, respectively. In addition, $\mathbf { K } \in \mathbb { R } ^ { 3 \times 3 }$ represents the known camera intrinsic parameters. Notably, points from LiDAR might be projected onto multiple pixels across multiple images. Therefore, we select the point with the shortest Euclidean distance to the image plane and retain it as the projected point, assigning color.

<!-- image-->  
Fig. 4: Pipeline of our reconstruction method. Left: DrivingGaussian++ takes sequential data from multi-sensor, including multi-camera images and LiDAR, as input. Middle: To represent large-scale dynamic driving scenes, we propose Composite Gaussian Splatting, which consists of two components. The first part incrementally reconstructs the extensive static background, while the second constructs multiple dynamic objects with a Gaussian graph and dynamically integrates them into the scene. Right: DrivingGaussian++ demonstrates good performance across multiple tasks and downstream applications.

Similar to existing 3D reconstruction methods [43], [44], we extend the dense bundle adjustment (DBA) to a multicamera setup and obtain the updated LiDAR points. Experimental results show that initializing with LiDAR prior to aligning with surrounding multi-camera aids in providing the Gaussian model with more precise geometry priors.

## 3.1.2 Incremental Static 3D Gaussians

The static backgrounds of driving scenes pose challenges for scene modeling and editing due to their large scale, long duration, and variations caused by ego vehicle movement with multi-camera transformation. As a vehicle moves, the static background frequently undergoes temporal shifts and dynamic changes. Due to the perspective principle, prematurely incorporating distant street scenes from time steps far away from the current can lead to scale confusion, resulting in unpleasant artifacts and blurring. To address this issue, we improve 3DGS by introducing Incremental Static 3D Gaussians, leveraging the perspective changes introduced by the vehicleâs movement and the temporal relationships between adjacent frames, as shown in Figure 5.

We uniformly divide the static scene into N bins based on the depth range provided by the LiDAR prior (Section 3.1.1). These bins are arranged in chronological order, denoted as $\{ \mathsf { b } _ { i } \} ^ { N }$ , where each bin contains multi-camera images from one or more time steps. Neighboring bins have a small overlap region, which is used to align the static backgrounds of two bins. The latter bin is then incrementally fused into the Gaussian field of the previous bins. For the scene within the first bin, we initialize the Gaussian model using the LiDAR prior (similarly applicable to SfM points):

$$
p _ { b _ { 0 } } ( l | \mu , \Sigma ) = e ^ { - \frac { 1 } { 2 } ( 1 - \mu ) ^ { \top } \Sigma ^ { - 1 } ( 1 - \mu ) } ,\tag{2}
$$

where $1 \in \mathbb { R } ^ { 3 }$ is the position of the LiDAR prior; $\mu$ is the mean of the LiDAR points; $\textbf { \textsf { E } } \in \ \mathbb { R } ^ { 3 \times 3 }$ is an anisotropic covariance matrix; and $\top$ is the transpose operator. We utilize the surrounding views within this bin segment as supervision to update the parameters of the Gaussian model, including position $P ( x , y , z )$ , covariance matrix $\Sigma ,$ coefficients of spherical harmonics for view-dependent color $C ( r , g , b )$ , along with an opacity Î±.

For the subsequent bins, we use the Gaussians from the previous bin as the position priors and align the adjacent bins based on their overlapping regions. The 3D center for each bin can be defined as:

$$
\begin{array} { r } { \hat { P } _ { b + 1 } ( G _ { s } ) = P _ { b } ( G _ { s } ) \bigcup ( x _ { b + 1 } , y _ { b + 1 } , z _ { b + 1 } ) , } \end{array}\tag{3}
$$

where $\hat { P }$ is the collection of 3D center for Gaussians $G _ { s }$ of all currently visible regions, $( x _ { b + 1 } , y _ { b + 1 } , z _ { b + 1 } )$ is the Gaussians coordinate within the $b + 1$ region. We incorporate scenes from the subsequent bins into the previously constructed Gaussians with multiple surrounding frames as supervision. The incremental static Gaussian model $G _ { s }$ is defined by:

$$
\hat { C } ( G _ { s } ) = \sum _ { b = 1 } ^ { N } \Gamma _ { b } \alpha _ { b } C _ { b } , \quad \Gamma _ { b } = \prod _ { i = 1 } ^ { b - 1 } ( 1 - \alpha _ { b } ) \ : ,\tag{4}
$$

where $C$ denotes the color corresponding to each Gaussian in a certain view, Î± is the opacity, and Î is the accumulated transmittance of the scene according to Î± in all bins. During this process, the overlapping regions between surrounding multi-camera images are used to form the Gaussian modelsâ implicit alignment jointly.

Note that during the incremental construction of static Gaussian models, there may be differences in sampling the same scene between the front and rear cameras. As such, we use a weighted averaging to reconstruct the sceneâs colors as accurately as possible during the 3D Gaussian projection:

$$
\tilde { C } = \varsigma ( G _ { s } ) \sum \omega ( \hat { C } ( G _ { s } ) | \mathbf { R } , \mathbf { T } ) ,\tag{5}
$$

where $\tilde { C }$ is the optimized pixel color, Ï denotes the differential splatting, Ï is the weight for different views, [R, T] is the view-matirx for aligning multi-camera views.

<!-- image-->  
Fig. 5: Composite Gaussian Splatting with Incremental Static 3D Gaussians and Dynamic Gaussian Graph. We adopt Composite Gaussian Splatting to decompose the whole scene into static background and dynamic foreground objects, reconstructing each part separately and integrating them for global rendering.

## 3.1.3 Composite Dynamic Gaussian Graph

The autonomous driving environment is highly complex, involving multiple dynamic objects and temporal changes. As shown in Figure 5, objects are often observed from limited views (e.g., 2-4 views) due to the egocentric movements of the vehicle and dynamic objects. In addition, fast moving objects also lead to significant appearance, making it challenging to represent them using fixed Gaussians.

To address the challenges, we introduce the Composite Dynamic Gaussian Graph, enabling the construction of multiple dynamic objects in long-term, large-scale driving scenes. We first decompose dynamic foreground objects from static backgrounds to build the dynamic Gaussian graph using bounding boxes provided by the datasets. Dynamic objects are identified by their object ID and the corresponding timestamps of appearance. Additionally, the Grounded SAM Models [45] are employed for precise pixelwise extraction of dynamic objects based on the range of bounding boxes.

We construct a dynamic Gaussian graph using

$$
H = < O , G _ { d } , M , P , A , T > ,\tag{6}
$$

where each node stores an instance object $o \in O , g _ { i } \in G _ { d }$ denotes the corresponding dynamic Gaussians, and $m _ { o } \in$ M is the transform matrix for each object. Here, $p _ { o } ( x _ { t } , y _ { t } , z _ { t } ) \in P$ is the center coordinate of the bounding box, and $\begin{array} { r l } { a _ { o } } & { { } = } \end{array}$ $( \theta _ { t } , \phi _ { t } ) \in A$ is the orientation of the bounding box at time step $t \in T$ . We compute one Gaussian separately for each dynamic object. Using the transformation matrix $m _ { o } ,$ we transform the coordinate system of the target object o to the world coordinate where the static background resides:

$$
\mathbf { m } _ { \mathbf { o } } ^ { - 1 } = \mathbf { R } _ { o } ^ { - 1 } \mathbf { S } _ { o } ^ { - 1 } ,\tag{7}
$$

where $\mathbf { R } _ { o } ^ { - 1 }$ and $\mathbf { S } _ { o } ^ { - 1 }$ are the rotation and translation matrices corresponding to each object.

After optimizing all nodes in the dynamic Gaussian graph, we combine dynamic objects and static backgrounds using a Composite Gaussian Graph. Each nodeâs Gaussian distribution is concatenated into the static Gaussian field based on the bounding box position and orientation in chronological order. In cases of occlusion between multiple dynamic objects, we adjust the opacity based on the distance from the camera center: closer objects have higher opacity, following the principles of light propagation:

$$
\alpha _ { o , t } = \sum \frac { ( p _ { t } - b _ { o } ) ^ { 2 } \cdot \cot a _ { o } } { \| ( b _ { o } | \mathbf { R _ { o } } , \mathbf { S _ { o } } ) - \rho \| ^ { 2 } } \alpha _ { p _ { 0 } } ,\tag{8}
$$

where $\alpha _ { o , t }$ is the adjusted opacity of Gaussians for object o at time step $t , p _ { t } = \dot { ( } x _ { t } , y _ { t } , \dot { z _ { t } } )$ is the center of Gaussians for the object. $[ \mathbf { R _ { o } } , \mathbf { S _ { o } } ]$ denotes the object-to-world transform matrix, $\rho$ denotes the center of camera view, and $\alpha _ { p _ { 0 } }$ is the opacity of Gaussians.

The composite Gaussian field, including both static background and multiple dynamic objects, is formulated by:

$$
G _ { c o m p } = \sum H < O , G _ { d } , M , P , A , T > + G _ { s } ,\tag{9}
$$

where $G _ { s }$ is obtained in Section 3.1.2 through Incremental Static 3D Gaussians and H denots the optimized dynamic Gaussian graph.

3D driving scene editing is based on the composite Gaussians of static background and dynamic objects, reconstructed by Composite Gaussian Splatting, and performs multiple editing tasks on them without extra training.

## 3.2 Global Rendering via Gaussian Splatting

We adopt the differentiable 3D Gaussian splatting renderer Ï from [14] and project the global composite 3D Gaussian into 2D with the covariance matrix Î£:

$$
\widetilde { { \boldsymbol \Sigma } } = { \mathbf J } { \mathbf E } { \boldsymbol \Sigma } { \mathbf E } ^ { \top } { \mathbf J } ^ { \top } ,\tag{10}
$$

where J is the Jacobian matrix of the perspective projection, and E denotes the world-to-camera matrix.

The composite Gaussian field projects the global 3D Gaussian onto multiple 2D planes and is supervised using surrounding views at each time step. In the global rendering process, Gaussians from the next time step are initially invisible to the current and subsequently incorporated with the supervision of the corresponding global images.

The loss function of our method consists of three parts. Similar to [14], [46], we introduce the Tile Structural Similarity (TSSIM) to Gaussian Splatting, which measures the similarity between the rendered tile and the corresponding ground truth:

$$
L _ { T S S I M } ( \delta ) = 1 - \frac { 1 } { Z } \sum _ { z = 1 } ^ { Z } S S I M ( \Psi ( \hat { C } ) , \Psi ( C ) ) ,\tag{11}
$$

where we split the screen into M tiles, Î´ is the training parameters of the Gaussians, $\Psi ( \hat { C } )$ denotes the rendered tile from Composite Gaussian Splatting, and $\Psi ( C )$ denotes the paired ground-truth tile.

We also introduce a robust loss to reduce outliers in 3D Gaussians, which can be defined as:

$$
L _ { R o b u s t } ( \delta ) = \kappa ( \| \hat { \cal I } - { \cal I } \| _ { 2 } ) ,\tag{12}
$$

where $\kappa \in ( 0 , 1 ]$ is the shape parameter that controls the robustness of the loss, I and ËI denote the ground truth and the synthesis image, respectively.

The LiDAR loss is further employed by supervising the expected Gaussiansâ position from the LiDAR, obtaining better geometric structure and edge shapes:

$$
L _ { L i D A R } ( \delta ) = \frac { 1 } { s } \sum \| P ( G _ { c o m p } ) - L _ { s } \| ^ { 2 } ,\tag{13}
$$

where $P ( G _ { c o m p } )$ is the position of 3D Gaussians, and $L _ { s }$ is the LiDAR point prior.

We optimize the Composite Gaussians by minimizing the sum of three losses in Eq 11-13. The proposed editing method leverages globally rendered images to identify editing targets and utilizes depth information derived from 3DGS as geometry priors, enabling effective and realistic multitask editing.

## 3.3 Controllable Editing for Dynamic Driving Scenes

We tackle three key editing tasks for autonomous driving simulations: texture modification, weather simulation, and object manipulation. To support these diverse editing tasks, we have developed a framework that sequentially operates on the Gaussians of the reconstructed scene using 3D geometric priors, Large Language Models (LLMs) for dynamic predictions, and advanced editing techniques to ensure overall coherence and realism.

Texture Modification: This task involves applying patterns to the surfaces of 3D objects. In autonomous driving, texture modification extends beyond aesthetics to allow the addition of critical road features, such as cracks, manhole covers, and signage, which are crucial to building more robust testing environments. We show failure cases of object detection model [47] in Figure 7, highlighting the importance of editing simulation. Before editing, the perception model accurately identifies objects within the scene. However, after editing with DrivingGaussian++, challenging cases in the 3D scene become undetectable to the model, providing a more effective testing environment to assess the reliability and robustness of various components within autonomous driving systems.

Weather Simulation: This task focuses on integrating dynamic meteorological phenomena, such as precipitation, snowfall, and fog, into autonomous driving scenarios. Weather simulation is critical for replicating driving conditions in severe weather, demonstrating its importance in augmenting training datasets.

Object Manipulation: This task is divided into object deletion and insertion within the reconstructed scene. Object insertion is further categorized into static and dynamic types, with dynamic insertion adaptively predicting the objectâs motion trajectory. These manipulations are essential for building robust autonomous driving simulation systems.

To enable multitask editing, we propose a framework that performs sequential operations on the Gaussians of the reconstructed scene without extra training. The process begins by identifying target Gaussians to edit using 3D geometric priors, followed by their integration into the scene. We employ large-language models (LLMs) to predict dynamic object trajectories and apply image-processing techniques to refine the results, ensuring coherence and realism. The editing pipeline is illustrated in Figure 6.

## 3.3.1 Initialization

In the proposed editing framework, we refer to those introduced to or removed from the original scene as target Gaussians, while those reconstructed from the initial scene are termed original Gaussians. The approach for determining target Gaussians depends on the specific editing task. For object removal, the target Gaussians, corresponding to the subset of original Gaussians marked for removal, are identified by refining the 3D bounding box provided by the dataset. Since the LiDAR prior is integrated during the reconstruction process, we can accurately locate their positions without additional alignment of the coordinate system. For other editing tasks, new Gaussians are generated as target Gaussians, designed with specific shapes and distributions to meet the requirements of each task.

Texture Modification. We enhance the surface texture of an object by introducing new flat Gaussians onto the surface of the designated editing region. The process begins by selecting a viewpoint and using a diffusion model or similar tools to edit the original image, generating a target image and a corresponding mask to guide the 3D editing. Specifically, we randomly select a viewpoint that provides clear visibility of the target region and render the image to be edited along with its associated depth map. Next, we define the 2D mask of the target region and apply the diffusion model or image processing software to modify the image in a 2D space, producing the target image.

Using the target image and the mask, we generate target Gaussians and assign appropriate attributes through inverse projection. As shown in Fig. 8, DrivingGaussian++ projects the edited content onto the corresponding position based on the rendered depth map and pixel-wise correspondence.

However, discrepancies may arise between the surface reconstructed by 3D Gaussian Splatting and the actual objectâs surface. These discrepancies can lead to inconsistencies between the rendered depth and the real depth of the object, potentially causing the surface of the target Gaussians to appear uneven and unrealistic, thereby compromising the editing quality.

<!-- image-->

Fig. 6: Pipeline of our editing framework. Left: We separately determine the target Gaussians for diverse tasks. For texture modification, first edit images and then conduct inverse projection with depth information. For weather simulation, design the attributes and distribution of weather particles in detail. For object manipulation, use models from the foreground bank as inserted objects, or delete objects based on annotations. Top right: We composite the Gaussians and predict the trajectory of new objects with LLM. Bottom right: To produce realistic editing, we perform shadow addition and inpainting. Our method achieves training-free, multi-task editing specifically for dynamic driving scenarios.  
<!-- image-->  
Fig. 7: Failed Object Detection Cases Simulation. We use GroundingDINO [47] as the object detection model. It struggles to completely identify vehicles in foggy weather and fails to accurately recognize textures, such as cracks.

To address this issue, we perform equalization on the depth map. Specifically, we normalize the depth of the editing area to ensure a relatively uniform depth distribution along the horizontal axis, while preserving the depth distribution along the vertical axis:

$$
D _ { o p t } ( M _ { e d i t } , x , y ) = A v e r a g e ( D _ { o r i } ( M _ { e d i t } , y ) ) ,\tag{14}
$$

where $D _ { o r i } , D _ { o p t }$ separately represent the rendered depth before and after the depth equalization, $M _ { e d i t }$ denotes the binary mask of the editing region and x, y are the image coordinates. This approach yields a flat surface for the target Gaussians, significantly enhancing the visual quality and realism of the texture modification.

Weather Simulation. We simulate weather particles by incorporating Gaussians with specific physical properties into the current scene and achieve dynamic effects by adjusting the positions of these Gaussians at each time step. The first step in weather simulation is to design particles that align with the desired physical properties. We calculate the number of original Gaussians along with the range of their positions and introduce new Gaussians with specific shapes and colors in a particular distribution within the scene. Specifically, we use narrow, semi-transparent, white Gaussians to represent raindrops, irregular white ellipsoidal Gaussians to represent snowflakes, and Gaussians following a random distribution in the scene to represent fog. As an example, for snow simulation, we define the target Gaussians $\bar { G } _ { s n o w }$ by:

<!-- image-->  
Fig. 8: Implementation of Snow Coverage Effect. We compute image normals, generate a snow mask, and fuse it with the original image to produce the target image. Depth priors are used to back-project the result into 3D space.

$$
G _ { s n o w } = \{ G _ { k } \} ,\tag{15}
$$

where the $k _ { t h }$ Gaussian $G _ { k }$ satisfies $p _ { k } = \zeta * \left( p _ { \operatorname* { m a x } } - p _ { \operatorname* { m i n } } \right) +$ $p _ { \operatorname* { m i n } } , c _ { k } = ( 1 , 1 , 1 ) + \epsilon , s _ { k , y } = \operatorname* { m i n } ( \operatorname* { m i n } ( s _ { k , x } , s _ { k , x } ) , 0 ) + \varepsilon ,$ and $p _ { k } , c _ { k } , s _ { k , y }$ separately denotes its 3D coordinate, color, and scale attribute.

Second, to realize dynamic weather effects that include the falling of raindrops, the drifting of snowflakes, and the spread of fog, we add a specific trajectory to weather Gaussians according to the current time step. We describe the trajectory of snowflakes using an example:

$$
p _ { k , t + 1 } = p _ { k , t } + t r a j j \_ f u n c ( t ) ,\tag{16}
$$

where $p _ { k , t }$ denotes the position of $k _ { t h }$ Gaussian in $G _ { s n o w }$ at timestep t, and $t r a j j \_ f u n c$ is a function that calculates the relative movement between consecutive positions in the time sequence.

<!-- image-->  
Fig. 9: Qualitative comparison with EmerNeRF [36] and 3DGS [14] on dynamic reconstruction for 4D driving scenes of nuScenes. DrivingGaussian++ enables the high-quality reconstruction of dynamic objects at high speed while maintaining temporal consistency.

We also implement the 3D snow coverage effect, as shown in Figure 8. Specifically, we first calculate the normal map of training images based on Depth-Anything [48] and the Sobel filter [49] as:

$$
s _ { i , x } , s _ { i , y } = S o b e l _ { x } ( D _ { i } ) , S o b e l _ { y } ( D _ { i } ) ,\tag{17}
$$

$$
N _ { i } = ( \frac { s _ { i , x } } { \sqrt { s _ { i , x } ^ { 2 } + s _ { i , y } ^ { 2 } } } , \frac { s _ { i , y } } { \sqrt { s _ { i , x } ^ { 2 } + s _ { i , y } ^ { 2 } } } , \frac { 1 } { \sqrt { s _ { i , x } ^ { 2 } + s _ { i , y } ^ { 2 } } } ) ,\tag{18}
$$

where $D _ { i }$ and $N _ { i }$ denote the depth and normal map of the image i, while $s _ { i , x }$ and $s _ { i , y }$ are gradient magnitudes in the horizontal and vertical directions. Based on the normal map, a snow mask is added in the region with a large vertical y component. Using a processed image with snow coverage and rendered depth, an inverse projection is taken to calculate the snow particle Gaussiansâ 3D positions from this viewpoint. Finally, we combine the positions under different viewpoints to realize a consistent snow-covering effect between frames. To avoid inconsistencies arising from repeated calculations in overlapping areas between frames, we construct a KD-Tree, and prune nodes that are closer to each other:

$$
P _ { s n o w } = \bigcup _ { i = 1 } ^ { n } [ P _ { i } - K N N ( \bigcup _ { j = 1 } ^ { i - 1 } P _ { j } , \ K D T ( P _ { i } ) , \ 1 _ { t h } ) ] ,\tag{19}
$$

where $P _ { s n o w }$ denotes positions of the target snow particle Gaussians, ${ \bar { P } } _ { i }$ denotes positions calculated from viewpoints in the $i _ { t h }$ frame, $K D \dot { T }$ refers to the constructed KD-Tree, and KNN stands for K Nearest Neighbor (KNN) function, which takes three parameters as input: the search range, the KDTree of the search target, and the number of top k neighbors. We insert the final target snow particle Gaussians into the scene and realize the snow-covering effect.

Object Manipulation. Due to the distinct nature of the operations, object insertion and deletion differ in their implementation. For object removal, the target Gaussians correspond to the object to be deleted. First, we extract the objectâs 3D bounding box matrix from the dataset annotations and crop the Gaussians within the bounding box. To address holes caused by insufficient reconstruction in the occluded areas, we further use a diffusion model to locally paint the rendered image (Section 3.3.3). For object insertion, we construct a 3D foreground bank containing objects reconstructed using 3D Gaussian Splatting, which can be directly utilized for insertion. The objects in the bank are acquired through 3DGS reconstruction of Blender models collected online and sparse reconstruction of vehicles from autonomous driving datasets. Additionally, the lighting of the foreground object can be adjusted using MCLight [50] to better match the current scene.

## 3.3.2 Gaussians Composition with Trajectory Prediction

After identifying the target Gaussians, we integrate them with the original scene. This process aligns both components within the same coordinate system to establish physically accurate occlusion relationships. Notably, the covariance matrices of the two Gaussian groups may interfere with each other during rasterization rendering, potentially leading to blurry results. Thus, we perform an additional forward process for the added Gaussians and store the covariance matrices of the transformed Gaussians. Finally, the combined scene is rendered for visualization.

For the object insertion task, to ensure that dynamic objects have reasonable and diverse motion trajectories, we utilize the powerful scene comprehension capability of the Large Language Models [51] to predict the future trajectories of the inserted objects as:

$$
\begin{array} { c } { { P _ { t + 1 } = P _ { t } + T r a j j \_ p r e d ( t ) , } } \\ { { T r a j j \_ p r e d = L L M ( P _ { 0 } , d i r _ { s k y } , d e s ) , } } \end{array}\tag{20}
$$

where $P _ { t }$ denotes the position of inserted object at timestep t while $\dot { P } _ { 0 }$ is the initial position, T rajj pred(t) denotes the relative positions at time step t generated by LLM, $d i r _ { s k y }$ is the sky direction and des denotes the description of the expected trajectory. Specifically, we take the initial vehicleâs position, the sky direction, and the trajectory description as prompts, and generate a series of possible future trajectory sequences through GPT-4o [51].

<!-- image-->  
Fig. 10: Trajectory Prediction with LLM. With the sky direction, trajectory prompt, and initial position, we utilize LLM to predict the future trajectory of the added car.

## 3.3.3 Global Refinement with Differentiable Rendering

Leveraging the recent advances in diffusion models and 2D image processing, our approach integrates these techniques to enhance the results of object manipulation tasks. For object removal, we use diffusion models to locally inpaint the damaged regions of the rendered image. First, we delete the target Gaussians of the specified region based on 3D annotations. However, due to occlusions and limitations in the data acquisition viewpoint, surrounding areas of the deleted Gaussians often contain artifacts or holes with poor reconstruction quality. To address this issue, we use the K Nearest Neighbor lgorithm to identify a set of Gaussians requiring repair around the target region. We then perform a binarization rendering on these Gaussians to generate the corresponding inpainting masks:

$$
M _ { i n p a i n t } = \left\{ G _ { i } \in G _ { l } \vert d ( p _ { i } , G _ { d e l } ) < d _ { t h r } \right\} ,\tag{21}
$$

where $M _ { i n p a i n t }$ is a binary mask with the Gaussians to be inpainted set to $1 , G _ { l }$ represents the remaining Gaussians after removal, while $G _ { d e l }$ denotes the removed Gaussians, $p _ { i }$ denotes the position of $G _ { i } , d _ { t h r }$ is the distance threshold that determines which Gaussians should be inpainted. The nearest distance between $G _ { i }$ and the Gaussians in $G _ { d e l }$ is given by $d ( p _ { i } , G _ { d e l } )$ , which is computed as $d ( p _ { i } , G _ { d e l } ) =$ $K N N ( p _ { i } , K D T ( P _ { d e l } )$ . Subsequently, the images to be repaired, along with corresponding masks, are fed into the diffusion model as inputs. DrivingGaussian++ performs partial inpainting to restore the integrity and visual authenticity of the scene, achieving more realistic and seamless object removal.

For the object insertion task, when extracting data from autonomous driving datasets, we perform sparse reconstruction to generate the target Gaussians. The reconstructed vehicles lack shadow information, which leads to a levitation effect in the rendered images. To enhance the realism of object insertion without additional training, we adopt a shadow synthesis approach inspired by ARShadowGAN [52]. Specifically, we synthesize shadows for the inserted objects to eliminate the levitation effect, thereby improving the visual consistency and realism of the scene.

<!-- image-->  
Fig. 11: Refinement on Object Insertion. We sparsely reconstruct the car from nuScenes with one-shot generation and then generate the shadow for the car.

## 4 EXPERIMENTS

## 4.1 Datasets

The nuScenes [54] dataset comprises 1000 driving scenes collected using multiple sensors (6 cameras, 1 LiDAR, etc.). The images are annotated with accurate 3D bounding boxes from 23 object classes. Our experiments utilize the keyframes of six challenging scenes with surrounding views, collected from 6 cameras and corresponding LiDAR sweeps (optional), as input. The KITTI-360 [55] dataset contains multiple sensors, corresponding to over 320k images and point clouds. Although the dataset provides stereo camera images, we use only a single camera to demonstrate that our method also performs well in monocular scenes.

## 4.2 Implementation Details

Our implementation is based on the 3DGS framework, with fine-tuned optimization parameters to fit the large-scale unbounded scenes. Instead of using SfM points or randomly initialized points as input, we employ the LiDAR prior mentioned in Section 3.1.1 for initializations. Considering the computational cost, we use a voxel grid filter for LiDAR points, reducing the scale without losing geometric features. We employ random initialization for dynamic objects with initial points set to 3000, since objects are relatively small in large-scale scenes. We increase the total training iterations to 50,000, set the threshold for densifying grad to 0.001, and reset the opacity interval to 900. The learning rate of Incremental Static 3D Gaussians remains the same as in the official setting, while the learning rate of the Composite Dynamic Gaussian Graph exponentially decays from 1.6e-3 to 1.6e-6. We assess our models using various metrics, including PSNR, SSIM, and LPIPS, and report the average results of all camera frames in the scenes. All experiments are carried out on an 8 RTX8000 with 384 GB memory.

## 4.3 Reconstruction Results and Comparisons

## 4.3.1 Comparisons of surrounding views on nuScenes.

We evaluate the proposed model against state-of-the-art approaches, including NeRF-based methods [16], [26], [27], [34]â[36], [53] and 3DGS-based schemes [14]. As shown in Table 1, our method outperforms Instant-NGP [16], which employs a hash-based NeRF for novel view synthesis. While Mip-NeRF [26] and Mip-NeRF360 [27] are designed specifically for unbounded outdoor scenes, our method performs favorably in all metrics.

TABLE 1: Overall perforamnce of DrivingGaussian++ with existing state-of-the-art approaches on the nuScenes dataset. Ours-S denotes the DrivingGaussian++ with SfM initialization, and Ours-L denotes training with LiDAR prior.
<table><tr><td>Methods</td><td>Input</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>Instant-NGP [16]</td><td>Images</td><td>16.78</td><td>0.519</td><td>0.570</td></tr><tr><td>NeRF+Time</td><td>Images</td><td>17.54</td><td>0.565</td><td>0.532</td></tr><tr><td>Mip-NeRF [26]</td><td>Images</td><td>18.08</td><td>0.572</td><td>0.551</td></tr><tr><td>Mip-NeRF360 [27]</td><td>Images</td><td>22.61</td><td>0.688</td><td>0.395</td></tr><tr><td>Urban-NeRF [53]</td><td>Images + LiDAR</td><td>20.75</td><td>0.627</td><td>0.480</td></tr><tr><td>S-NeRF [34]</td><td>Images + LiDAR</td><td>25.43</td><td>0.730</td><td>0.302</td></tr><tr><td>SUDS [35]</td><td>Images + LiDAR</td><td>21.26</td><td>0.603</td><td>0.466</td></tr><tr><td>EmerNeRF [36]</td><td>Images + LiDAR</td><td>26.75</td><td>0.760</td><td>0.311</td></tr><tr><td>3DGS [14]</td><td>Images + SfM Points</td><td>26.08</td><td>0.717</td><td>0.298</td></tr><tr><td>Ours-S</td><td>Images + SfM Points</td><td>28.36</td><td>0.851</td><td>0.256</td></tr><tr><td>Ours-L</td><td>Images + LiDAR</td><td>28.74</td><td>0.865</td><td>0.237</td></tr></table>

TABLE 2: Editing Efficiency. We demonstrate the efficient multi-task editing capabilities of our method in comparison with existing state-of-the-art 3D editing approaches. The evaluated multi-task editing scenarios include object manipulation, weather simulation, texture modification, and dynamic editing. âExec. Timeâ indicates the total execution time for completing each editing task under a fair comparison setting.

<table><tr><td>Method</td><td></td><td></td><td>Object Manipulation Weather Simulation Texture Modification Dynamic Editing Exec. Time</td><td></td><td></td><td>CLIP â</td></tr><tr><td>InstructNeRF2NeRF</td><td>X</td><td></td><td></td><td>X</td><td>â274 minutes</td><td>0.1570</td></tr><tr><td>InstructGS2GS</td><td>X</td><td></td><td></td><td>X</td><td>â60 minutes</td><td>0.0918</td></tr><tr><td>ClimateNeRF</td><td>X</td><td></td><td>X</td><td>X</td><td>â107 minutes</td><td>0.1105</td></tr><tr><td>Ours</td><td>â</td><td>â</td><td>â</td><td>J</td><td>â8 minutes</td><td>0.2327</td></tr></table>

<!-- image-->  
Fig. 12: Editing Results of nuScenes dataset. We demonstrate the results of DrivingGaussian++ across different tasks. DrivingGaussian++ enables realistic and consistent 3D editing of texture, weather, and objects in driving scenes.

Urban-NeRF [53] uses depth cues from LiDAR in a NeRF model to reconstruct urban scenes. In contrast, we leverage LiDAR as a geometric prior in the proposed Gaussian models to achieve more effective large-scale scene reconstruction. Our method performs favorably against S-NeRF [34] and SUDS [35], where both decompose the scene into static background and dynamic objects and construct the scene with the help of LiDAR. Compared to EmerNeRF [36], which applies a spatial-temporal representation for dynamic driving scenes using flow fields, our method achieves stateof-the-art results in all metrics, eliminating the need to estimate scene flow. For Gaussian-based approaches, our method enhances the performance of our baseline method, 3DGS [14], on large-scale scenes in all metrics.

We show qualitative evaluation results on challenging nuScenes driving scenes. For multicamera surround view synthesis, as shown in Figure 9, our method enables the generation of photorealistic rendering images and ensures view consistency across multicameras. Meanwhile, EmerNeRF [36] and 3DGS [14] do not perform well in challenging regions, exhibiting undesirable visual artifacts such as ghosting, dynamic object disappearance, loss of plant tex-

<!-- image-->

Fig. 13: Dynamic Simulation of Snow. We first add dynamic particles at each time step. Secondly, we estimate the surface normal to obtain the particle deposition position and specify the particle motion trajectory. Our method can generate snowy scenes that follow physical principles.  
<!-- image-->  
Fig. 14: Qualitative Comparison of Weather Simulation. We compare our method with IN2N, IGS2GS, and ClimateNeRF for weather simulation on 4D driving scenes from the nuScenes dataset. DrivingGaussian++ delivers realistic and coherent weather editing while maintaining high efficiency.

TABLE 3: Overall perforamcne on the KITTI-360. Comparisions of DrivingGaussian++ with existing state-of-the-art approaches on the KITTI-360 dataset.
<table><tr><td>Methods</td><td>PSNR â</td><td>SSIM â</td></tr><tr><td>NeRF [7]</td><td>21.94</td><td>0.781</td></tr><tr><td>Point-NeRF [56]</td><td>21.54</td><td>0.793</td></tr><tr><td>NSG [12]</td><td>22.89</td><td>0.836</td></tr><tr><td>Mip-NeRF360 [27]</td><td>23.27</td><td>0.836</td></tr><tr><td>SUDS [35]</td><td>23.30</td><td>0.844</td></tr><tr><td>DNMP [57]</td><td>23.41</td><td>0.846</td></tr><tr><td>Ours-S</td><td>25.18</td><td>0.862</td></tr><tr><td>Ours-L</td><td>25.62</td><td>0.868</td></tr></table>

ture details, lane markings, and blurring in distant scenes.

We next demonstrate the reconstruction results for dynamic temporal scenes. Our method accurately models dynamic objects within large-scale scenes, mitigating issues such as loss, ghosting, or blurring of these dynamic elements. The proposed model consistently constructs dynamic objects over time, despite their relatively high speed of movement. As depicted in Figure 9, other approaches [14], [36] are inadequate for rapidly moving dynamic objects.

## 4.3.2 Comparisons of mono-view on KITTI-360

To further validate the effectiveness of our method in the setting of a monocular driving scene, we conduct experiments with the KITTI-360 data set with comparisons to existing state-of-the-art approaches, including NeRF [7], Mip-NeRF360 [27], Point-NeRF [56], NSG [12], SUDS [35], and DNMP [57]. As shown in Table 3, our method performs favorably in monocular driving scenes against other models.

## 4.4 Editing Results and Comparisons

We first demonstrate our editing results on the nuScenes dataset for multiple tasks. Compared with state-of-the-art 2D and 3D editing methods, our approach achieves superior visual realism and better quantitative consistency.

To support flexible editing on driving scenes, we additionally created a 3D Gaussian foreground bank containing specialized driving scene objects. This foreground bank is critically important for autonomous driving simulation and model validation.

## 4.4.1 Qualitative Results and Comparisons

We perform training-free editing of the reconstructed nuScenes data by DrivingGaussian++ in three domains: texture, weather, and object manipulation. The comprehensive results are shown in Fig 12, showcasing the ability of DrivingGaussian++ to perform various editing operations in dynamic driving scenarios.

For weather editing, we achieve realistic effects through particle-based simulation, as described in Section 3.3.1. For snow simulation particularly, we add snow particle Gaussians at each timestep and estimate surface normals to determine deposition locations. This produces realistic snow accumulation, as shown in Figure 13. For object manipulation, we first obtain actors using 3D reconstruction or 4D generation methods such as DreamGaussian4D [58]. This allows for the insertion of dynamic objects, including nonrigid bodies such as humans and animals. By adapting the deformation module to foreground contexts, we achieve flexible and diverse dynamic object integration. Additionally, we employ LLM-based trajectory prediction to obtain the trajectory of inserted objects. The results of dynamic objects insertion are shown in Figure 15.

<!-- image-->  
Fig. 15: Dynamic Object Insertion and Scene Integration. We generate deformable foreground objects and seamlessly insert them into the scene. (a) A rigid car is added. (b) A generated 4D excavator is integrated. (c) Multiple objects are inserted with correct occlusion. The results demonstrate the effectiveness of our method in achieving natural object insertion with realistic spatial interactions.

Fig 14 provides a performance comparison with existing 3D editing approaches. While InstructNeRF2Nerf [1] and InstructGS2GS [15] employ diffusion models for iterative 3D scene editing across multiple tasks, they exhibit limitations in maintaining photorealism and view consistency. ClimateNeRF [6] specializes in particle-level weather editing through surface normal computations, but its application lacks generalizability to other editing tasks and remains constrained to static environments. Our method addresses these limitations while achieving high-quality results across all editing tasks.

## 4.4.2 Quantitative Results and Comparisons

To evaluate the consistency and realism of our editing approach, we compare DrivingGaussian++ with state-of-theart 3D and 2D editing techniques.

For 3D scene editing, we compare with ClimateNeRF [6], IN2N [1], and IGS2GS [15] in terms of task diversity, processing time, and CLIP-direction similarity [63]. As shown in Table 2, DrivingGaussian++ performs favorably against all other methods in terms of diversity, efficiency, and textaligned consistency. In particular, the editing time of DrivingGaussian++ is typically within 3â¼10 minutes for scenes from the NuScenes dataset, significantly lower than that of other 3D editing models that require long training time.

To evaluate the performance of DrivingGaussian++ on single-view editing, we also compare it with 2D editing methods [59]â[64], [66] across different tasks, as shown in Table 4. For texture modification and object insertion, we compare with inpainting methods [59]â[61]. While Any-Door [60] and Paint-by-Example [59] utilize 2D images for conditional editing, they produce inconsistent perspective relationships and poor consistency with the condition image. SD-Inpainting [61] takes text prompts and 2D masks as input, but suffers from limited performance and controllability. For weather simulation, we evaluate textguided editing methods [62]â[64]. Although FreePromptEditing [64], InstructPix2Pix [63], and InstructDiffusion [62] exhibit good text understanding, their results often lack physical plausibilityâfor instance, snow is rendered merely as a stylistic change rather than as accumulated precipitation. InstructDiffusion [62] editing results in these weather scenes are less realistic. For object removal, we evaluate inpainting and text-guided methods [61], [62], [66]. SD-Inpainting [61] and InstructDiffusion [62] leave residual artifacts, while LaMa [66] introduces visible inconsistencies in scene restoration.

We evaluate editing consistency using CLIP direction similarity metric for texture, weather editing and object insertion. For object removal, we evaluate the quality using

<!-- image-->  
Fig. 16: Multiple Objects Insertion with Foreground Bank. We insert distinct objects from the foreground bank into each scene across various viewpoints. The results exhibit strong cross-view consistency in both geometry and appearance, highlighting the robustness of our method for 3D object insertion in complex driving scenes.

TABLE 4: Quantitative Comparison of DrivingGaussian++ with State-of-the-Art Image-Editing Approaches. We compare our method with existing state-of-the-art techniques on the nuScenes dataset across various editing tasks. The results show competitive performance, with improved quantitative metrics and enhanced editing quality.
<table><tr><td>Methods</td><td>Type</td><td> $\mathbf { C L I P } _ { d i r } \uparrow$ </td></tr><tr><td>Paint-by-Example [59]</td><td>2D</td><td>0.0282</td></tr><tr><td>AnyDoor [60</td><td>2D</td><td>0.0027</td></tr><tr><td>Ours</td><td>3D</td><td>0.0866</td></tr></table>

(a) Object Insertion Task.

<table><tr><td>Methods</td><td>Type</td><td>CLIPdir â</td></tr><tr><td>InstructDiffusion [62]</td><td>2D</td><td>0.0040</td></tr><tr><td>InstructPix2Pix [63]</td><td>2D</td><td>0.2241</td></tr><tr><td>FreePromptEditing [64]</td><td>2D</td><td>0.1709</td></tr><tr><td>UltraEdit [65]</td><td>2D</td><td>0.2292</td></tr><tr><td>Ours</td><td>3D</td><td>0.2462</td></tr></table>

(c) Weather Edit Task.  
LPIPS and FID (as shown in SPIn-NeRF [67]). DrivingGaussian++ achieves superior performance on all tasks.

## 4.4.3 3D Gaussian Foreground Bank for Driving Scenes

We construct a comprehensive 3D Gaussian foreground bank containing various traffic elements: vehicles, bicycles, motorcycles, pedestrians, animals, and static objects such as signs and traffic cones. Figure 16 shows our foreground bank and insertion results.

Online Model Reconstruction. We collect 3D models (pedestrians, vehicles, etc.) from online sources and Chatsim [50], then reconstruct them using 3DGS [14]. For each model, we render 360Â°views in Blender and perform 3DGS reconstruction with COLMAP [42]. We adjust lighting using environment maps extracted from nuScenes.

<table><tr><td>Methods</td><td>Type</td><td>LPIPS â</td></tr><tr><td>SD-inpainting [61]</td><td>2D</td><td>0.3435</td></tr><tr><td>InstructDiffusion [62]</td><td>2D</td><td>0.3271</td></tr><tr><td>Ours</td><td>3D</td><td>0.3286</td></tr></table>

Sparse Reconstruction of vehicles from nuScenes. We efficiently sparse reconstruct vehicles from nuScenes using SplatterImage [68]. Each vehicle requires about 2â¼4 reference images for Gaussian reconstruction.

Image-based Object Generation. To expand our dataset, we generate 3D objects with image inputs. We first extract clean object images with SAM [69]. Subsequently, we create static and dynamic 3D models using dreamgaussian [70] and dreamgaussian4d [58] for few-shot 3D generation, enabling the creation of static and dynamic objects with high fidelity and efficiency.

(b) Object Removal Task.
<table><tr><td>Methods</td><td>Type</td><td>CLIPdir â</td></tr><tr><td>Paint-by-Example [59]</td><td>2D</td><td>0.0940</td></tr><tr><td>AnyDoor [60]</td><td>2D</td><td>0.1358</td></tr><tr><td>SD-inpainting [61]</td><td>2D</td><td>0.0493</td></tr><tr><td>UltraEdit [65]</td><td>2D</td><td>0.0427</td></tr><tr><td>Ours</td><td>3D</td><td>0.2019</td></tr></table>

(d) Texture Edit Task.

## 4.5 Ablation Study

## 4.5.1 Initialization prior for Gaussians

Comparative experiments are conducted to analyze the effect of different priors and initialization methods on the Gaussian model. The original 3DGS provides two initialization modes: randomly generated points and SfM points computed by COLMAP [42]. We additionally offer two other approaches: point clouds from a pre-trained NeRF model and points generated with LiDAR prior.

Meanwhile, to analyze the effect of point cloud quantity, we down-sample the LiDAR to 600K and apply adaptive filtering (1M) to control the number of generated Li-DAR points. We also set different maximum thresholds for randomly generated points (600K and 1M). Here, SfM-600KÂ± 20K represents the number of points computed by COLMAP, NeRF-1MÂ±20K denotes the total points generated by the pre-trained NeRF model, and LiDAR-2MÂ±20k refers to the original quantity of LiDAR points.

<!-- image-->  
Fig. 17: Rendering with or w/o the Incremental Static 3D Gaussians (IS3G) and Composite Dynamic Gaussian Graph (CDGG). IS3G ensures good geometry and topological integrity for static backgrounds in large-scale driving scenes. CDGG enables the reconstruction of dynamic objects at arbitrary speeds in driving scenes (e.g., vehicles, bicycles, and pedestrians).

<!-- image-->  
Fig. 18: Visualization comparison using different initialization methods on KITTI-360. Compared to initialization with SfM points [14], using LiDAR prior allows Gaussians to restore more accurate geometric structures in the scene.

As shown in Table 5, randomly generated points lead to the worst results as they lack any geometric prior. Initializing with SfM points also cannot adequately recover the sceneâs precise geometries due to the sparse points and intolerable structural errors. Leveraging point clouds generated from a pre-trained NeRF model provides a relatively accurate geometric prior, but there are still noticeable outliers. For the model initialized with the LiDAR prior, although downsampling results in loss of geometric information in some local regions, it still retains relatively accurate structural priors, thus surpassing SfM (Figure 18). We note that the experimental results do not change linearly with increasing LiDAR point quantities. This can be attributed

TABLE 5: Effect of different initialization methods on the Gaussian model. LiDAR-600K â  denotes downsampling the original LiDAR data to a corresponding point cloud magnitude. LiDAR-1M â¡ denotes denoising and removing outliers in LiDAR points, which is used in our method.
<table><tr><td>Methods</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td rowspan="5">Random-600K Random-1M SfM-600K NeRF-1M</td><td>22.18</td><td>0.653</td><td>0.424</td></tr><tr><td>22.23</td><td>0.653</td><td>0.421</td></tr><tr><td>28.36</td><td>0.851</td><td>0.256</td></tr><tr><td>28.51</td><td>0.858</td><td>0.251</td></tr><tr><td>28.49</td><td>0.854</td><td>0.245</td></tr><tr><td>LiDAR-600K â  LiDAR-1Mâ¡</td><td>28.74</td><td>0.865</td><td>0.237</td></tr><tr><td>LiDAR-2M</td><td>28.78</td><td>0.867</td><td>0.237</td></tr></table>

to that overly dense points store redundant features that interfere with the optimization of the Gaussian model.

## 4.5.2 Effectiveness of Model Component

We analyze the contribution of each module of the proposed model. As shown in Table 6 and Figure 17, the Composite Dynamic Gaussian Graph module plays a crucial role in reconstructing dynamic driving scenes, while the Incremental Static 3D Gaussians module enables high-quality large-scale background reconstruction. These two novel modules significantly enhance the modeling quality of complex driving scenes. Regarding the proposed loss functions, the ablation results indicate that both $\bar { L } _ { T S S I M }$ and $L _ { R o b u s t }$ significantly improve the rendering quality, improve texture details and remove artifacts. In addition, $L _ { L i D A R }$ from LiDAR prior helps Gaussians achieve better geometric priors. The experimental results also demonstrate that DrivingGaussian++ performs well even without the prior LiDAR, demonstrating strong robustness for various initialization methods.

## 5 CONCLUSION

We introduce DrivingGaussian++, a framework for reconstructing and editing large-scale dynamic autonomous driving scenes. Our approach progressively models the static background using incremental static 3D Gaussians and captures multiple moving objects through a composite dynamic Gaussian graph. By leveraging LiDAR priors, we achieve accurate geometric structures and robust multi-view consistency, significantly enhancing the quality of scene reconstruction. DrivingGaussian++ facilitates training-free editing for tasks such as texture modification, weather simulation, and object manipulation, enabling the generation of realistic and diverse driving scenes. Experimental results on datasets such as nuScenes and KITTI-360 demonstrate that our framework achieves state-of-the-art performance in both reconstruction and editing tasks, enabling high-quality surrounding view synthesis and dynamic scene editing.

TABLE 6: Effect of each module in our proposed method. IS3G is short for the Incremental Static 3D Gaussians module, and CDGG is short for the Composite Dynamic Gaussian Graph module.
<table><tr><td>Model</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>w/o IS3G</td><td>27.72</td><td>0.771</td><td>0.295</td></tr><tr><td>w/o CDGG</td><td>26.97</td><td>0.752</td><td>0.306</td></tr><tr><td> $\mathbf { w } / \mathbf { o } \ L _ { T S S I M }$ </td><td>27.88</td><td>0.783</td><td>0.280</td></tr><tr><td> $\mathbf { w } / \mathbf { o \ } L _ { R o b u s t }$ </td><td>28.05</td><td>0.814</td><td>0.271</td></tr><tr><td> $\mathbf { w } / \mathbf { o } L _ { L i D A R }$ </td><td>28.45</td><td>0.854</td><td>0.248</td></tr><tr><td> Ours-S</td><td>28.36</td><td>0.851</td><td>0.256</td></tr><tr><td>Ours-L</td><td>28.74</td><td>0.865</td><td>0.237</td></tr></table>

## REFERENCES

[1] A. Haque, M. Tancik, A. A. Efros, A. Holynski, and A. Kanazawa, âInstruct-nerf2nerf: Editing 3d scenes with instructions,â in ICCV, 2023, pp. 19 740â19 750. 1, 2, 3, 13

[2] A. Khandelwal, âInfusion: Inject and attention fusion for multi concept zero-shot text-based video editing,â in ICCV, 2023, pp. 3017â3026. 1, 4

[3] J. Wu, J.-W. Bian, X. Li, G. Wang, I. Reid, P. Torr, and V. A. Prisacariu, âGaussctrl: multi-view consistent text-driven 3d gaussian splatting editing,â arXiv preprint arXiv:2403.08733, 2024. 1, 3

[4] Y. Chen, Z. Chen, C. Zhang, F. Wang, X. Yang, Y. Wang, Z. Cai, L. Yang, H. Liu, and G. Lin, âGaussianeditor: Swift and controllable 3d editing with gaussian splatting,â in CVPR, 2024, pp. 21 476â21 485. 1, 3

[5] J. Dong and Y.-X. Wang, âVica-nerf: View-consistency-aware 3d editing of neural radiance fields,â NIPS, vol. 36, 2024. 1, 3

[6] Y. Li, Z.-H. Lin, D. Forsyth, J.-B. Huang, and S. Wang, âClimatenerf: Extreme weather synthesis in neural radiance field,â in ICCV, 2023, pp. 3227â3238. 1, 2, 4, 13

[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, pp. 99â106, 2021. 2, 12

[8] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan, J. T. Barron, and H. Kretzschmar, âBlock-nerf: Scalable large scene neural view synthesis,â in CVPR, 2022, pp. 8248â8258. 2

[9] H. Turki, D. Ramanan, and M. Satyanarayanan, âMega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs,â in CVPR, 2022, pp. 12 922â12 931. 2

[10] Z. Wang, T. Shen, J. Gao, S. Huang, J. Munkberg, J. Hasselgren, Z. Gojcic, W. Chen, and S. Fidler, âNeural fields meet explicit geometric representations for inverse rendering of urban scenes,â in CVPR, 2023, pp. 8370â8380. 2

[11] M. Zhenxing and D. Xu, âSwitch-nerf: Learning scene decomposition with mixture of experts for large-scale neural radiance fields,â in The Eleventh International Conference on Learning Representations, 2022. 2

[12] J. Ost, F. Mannan, N. Thuerey, J. Knodt, and F. Heide, âNeural scene graphs for dynamic scenes,â in CVPR, 2021, pp. 2856â2865. 2, 3, 12

[13] Y. Song, C. Kong, S. Lee, N. Kwak, and J. Lee, âTowards efficient neural scene graphs by learning consistency fields,â arXiv preprint arXiv:2210.04127, 2022. 2, 3

[14] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D Gaus- Â¨ sian splatting for real-time radiance field rendering,â TOG, vol. 42, no. 4, pp. 1â14, 2023. 2, 3, 6, 7, 9, 10, 11, 12, 14, 15

[15] C. Vachha and A. Haque, âInstruct-gs2gs: Editing 3d gaussian splats with instructions (2024),â URL https://instruct-gs2gs.github.io, 2024. 2, 13

[16] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural Â¨ graphics primitives with a multiresolution hash encoding,â TOG, vol. 41, no. 4, pp. 1â15, 2022. 2, 10, 11

[17] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, âPlenoxels: Radiance fields without neural networks,â in CVPR, 2022, pp. 5501â5510. 2

[18] S. J. Garbin, M. Kowalski, M. Johnson, J. Shotton, and J. Valentin, âFastnerf: High-fidelity neural rendering at 200fps,â in ICCV, 2021, pp. 14 346â14 355. 2

[19] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, âBarf: Bundleadjusting neural radiance fields,â in ICCV, 2021, pp. 5741â5751. 2

[20] Z. Wang, S. Wu, W. Xie, M. Chen, and V. A. Prisacariu, âNerfâ: Neural radiance fields without known camera parameters,â arXiv preprint arXiv:2102.07064, 2021. 2

[21] W. Bian, Z. Wang, K. Li, J.-W. Bian, and V. A. Prisacariu, âNopenerf: Optimising neural radiance field with no pose prior,â in CVPR, 2023, pp. 4160â4169. 2

[22] V. Rudnev, M. Elgharib, W. Smith, L. Liu, V. Golyanik, and C. Theobalt, âNerf for outdoor scene relighting,â in ECCV. Springer, 2022, pp. 615â631. 2

[23] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, âDnerf: Neural radiance fields for dynamic scenes,â in CVPR, 2021, pp. 10 318â10 327. 2

[24] X. Huang, Q. Zhang, Y. Feng, H. Li, X. Wang, and Q. Wang, âHdrnerf: High dynamic range neural radiance fields,â in CVPR, 2022, pp. 18 398â18 408. 2

[25] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Dosovitskiy, and D. Duckworth, âNerf in the wild: Neural radiance fields for unconstrained photo collections,â in CVPR, 2021, pp. 7210â7219. 2

[26] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-nerf: A multiscale representation for anti-aliasing neural radiance fields,â in ICCV, 2021, pp. 5855â5864. 2, 10, 11

[27] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in CVPR, 2022, pp. 5470â5479. 2, 10, 11, 12

[28] L. Xu, Y. Xiangli, S. Peng, X. Pan, N. Zhao, C. Theobalt, B. Dai, and D. Lin, âGrid-guided neural radiance fields for large urban scenes,â in CVPR, 2023, pp. 8296â8306. 2

[29] J. Guo, N. Deng, X. Li, Y. Bai, B. Shi, C. Wang, C. Ding, D. Wang, and Y. Li, âStreetsurf: Extending multi-view implicit surface reconstruction to street views,â arXiv preprint arXiv:2306.04988, 2023. 2

[30] Y.-L. Liu, C. Gao, A. Meuleman, H.-Y. Tseng, A. Saraf, C. Kim, Y.- Y. Chuang, J. Kopf, and J.-B. Huang, âRobust dynamic radiance fields,â in CVPR, 2023, pp. 13â23. 3

[31] A. Meuleman, Y.-L. Liu, C. Gao, J.-B. Huang, C. Kim, M. H. Kim, and J. Kopf, âProgressively optimized local radiance fields for robust view synthesis,â in CVPR, 2023, pp. 16 539â16 548. 3

[32] Z. Wu, T. Liu, L. Luo, Z. Zhong, J. Chen, H. Xiao, C. Hou, H. Lou, Y. Chen, R. Yang et al., âMars: An instance-aware, modular and realistic simulator for autonomous driving,â arXiv preprint arXiv:2307.15058, 2023. 3

[33] Z. Yang, Y. Chen, J. Wang, S. Manivasagam, W.-C. Ma, A. J. Yang, and R. Urtasun, âUnisim: A neural closed-loop sensor simulator,â in CVPR, 2023, pp. 1389â1399. 3

[34] Z. Xie, J. Zhang, W. Li, F. Zhang, and L. Zhang, âS-nerf: Neural radiance fields for street views,â arXiv preprint arXiv:2303.00749, 2023. 3, 10, 11

[35] H. Turki, J. Y. Zhang, F. Ferroni, and D. Ramanan, âSuds: Scalable urban dynamic scenes,â in CVPR, 2023, pp. 12 375â12 385. 3, 10, 11, 12

[36] J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che, D. Xu, S. Fidler, M. Pavone et al., âEmernerf: Emergent spatialtemporal scene decomposition via self-supervision,â arXiv preprint arXiv:2311.02077, 2023. 3, 9, 10, 11, 12

[37] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, âDeformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,â arXiv preprint arXiv:2309.13101, 2023. 3

[38] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â arXiv preprint arXiv:2310.08528, 2023. 3

[39] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, âDynamic 3d gaussians: Tracking by persistent dynamic view synthesis,â arXiv preprint arXiv:2308.09713, 2023. 3

[40] J. Zhuang, C. Wang, L. Lin, L. Liu, and G. Li, âDreameditor: Textdriven 3d scene editing with neural fields,â in SIGGRAPH Asia 2023 Conference Papers, 2023, pp. 1â10. 3

[41] J. Zhu, P. Zhuang, and S. Koyejo, âHifa: High-fidelity text-to-3d generation with advanced diffusion guidance,â arXiv preprint arXiv:2305.18766, 2023. 3

[42] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â in CVPR, 2016, pp. 4104â4113. 4, 14

[43] A. Schmied, T. Fischer, M. Danelljan, M. Pollefeys, and F. Yu, âR3d3: Dense 3d reconstruction of dynamic scenes from multiple cameras,â in ICCV, 2023, pp. 3216â3226. 5

[44] Q. Herau, N. Piasco, M. Bennehar, L. Roldao, D. Tsishkou, Ë C. Migniot, P. Vasseur, and C. Demonceaux, âMoisst: Multi-modal optimization of implicit scene for spatiotemporal calibration,â arXiv preprint arXiv:2303.03056, 2023. 5

[45] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, Z. Zeng, H. Zhang, F. Li, J. Yang, H. Li, Q. Jiang, and L. Zhang, âGrounded sam: Assembling open-world models for diverse visual tasks,â arXiv preprint arXiv:2401.14159, 2024. 6

[46] Z. Xie, X. Yang, Y. Yang, Q. Sun, Y. Jiang, H. Wang, Y. Cai, and M. Sun, âS3im: Stochastic structural similarity and its unreasonable effectiveness for neural fields,â in ICCV, 2023, pp. 18 024â 18 034. 7

[47] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu et al., âGrounding dino: Marrying dino with grounded pre-training for open-set object detection,â arXiv preprint arXiv:2303.05499, 2023. 7, 8

[48] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, âDepth anything: Unleashing the power of large-scale unlabeled data,â in CVPR, 2024, pp. 10 371â10 381. 9

[49] N. Kanopoulos, N. Vasanthavada, and R. L. Baker, âDesign of an image edge detection filter using the sobel operator,â IEEE Journal of solid-state circuits, vol. 23, no. 2, pp. 358â367, 1988. 9

[50] Y. Wei, Z. Wang, Y. Lu, C. Xu, C. Liu, H. Zhao, S. Chen, and Y. Wang, âEditable scene simulation for autonomous driving via collaborative llm-agents,â in CVPR, 2024, pp. 15 077â15 087. 9, 14

[51] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., âGpt-4 technical report,â arXiv preprint arXiv:2303.08774, 2023. 9, 10

[52] D. Liu, C. Long, H. Zhang, H. Yu, X. Dong, and C. Xiao, âArshadowgan: Shadow generative adversarial network for augmented reality in single light scenes,â in CVPR, 2020, pp. 8139â8148. 10

[53] K. Rematas, A. Liu, P. P. Srinivasan, J. T. Barron, A. Tagliasacchi, T. Funkhouser, and V. Ferrari, âUrban radiance fields,â in CVPR, 2022, pp. 12 932â12 942. 10, 11

[54] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, ânuscenes: A multimodal dataset for autonomous driving,â in CVPR, 2020. 10

[55] Y. Liao, J. Xie, and A. Geiger, âKitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 3, pp. 3292â3310, 2022. 10

[56] Q. Xu, Z. Xu, J. Philip, S. Bi, Z. Shu, K. Sunkavalli, and U. Neumann, âPoint-nerf: Point-based neural radiance fields,â in CVPR, 2022, pp. 5438â5448. 12

[57] F. Lu, Y. Xu, G. Chen, H. Li, K.-Y. Lin, and C. Jiang, âUrban radiance field representation with deformable neural mesh primitives,â in ICCV, 2023, pp. 465â476. 12

[58] J. Ren, L. Pan, J. Tang, C. Zhang, A. Cao, G. Zeng, and Z. Liu, âDreamgaussian4d: Generative 4d gaussian splatting,â arXiv preprint arXiv:2312.17142, 2023. 13, 14

[59] B. Yang, S. Gu, B. Zhang, T. Zhang, X. Chen, X. Sun, D. Chen, and F. Wen, âPaint by example: Exemplar-based image editing with diffusion models,â in CVPR, 2023, pp. 18 381â18 391. 13, 14

[60] X. Chen, L. Huang, Y. Liu, Y. Shen, D. Zhao, and H. Zhao, âAnydoor: Zero-shot object-level image customization,â in CVPR, 2024, pp. 6593â6602. 13, 14

[61] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, âHigh-resolution image synthesis with latent diffusion models,â in CVPR, 2022, pp. 10 684â10 695. 13, 14

[62] Z. Geng, B. Yang, T. Hang, C. Li, S. Gu, T. Zhang, J. Bao, Z. Zhang, H. Li, H. Hu et al., âInstructdiffusion: A generalist modeling interface for vision tasks,â in CVPR, 2024, pp. 12 709â12 720. 13, 14

[63] T. Brooks, A. Holynski, and A. A. Efros, âInstructpix2pix: Learning to follow image editing instructions,â in CVPR, 2023, pp. 18 392â 18 402. 13, 14

[64] B. Liu, C. Wang, T. Cao, K. Jia, and J. Huang, âTowards understanding cross and self-attention in stable diffusion for text-guided image editing,â in CVPR, 2024, pp. 7817â7826. 13, 14

[65] H. Zhao, X. Ma, L. Chen, S. Si, R. Wu, K. An, P. Yu, M. Zhang, Q. Li, and B. Chang, âUltraedit: Instruction-based fine-grained image editing at scale,â arXiv preprint arXiv:2407.05282, 2024. 14

[66] R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha, A. Silvestrov, N. Kong, H. Goka, K. Park, and V. Lempitsky, âResolution-robust large mask inpainting with fourier convolutions,â in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2022, pp. 2149â2159. 13

[67] A. Mirzaei, T. Aumentado-Armstrong, K. G. Derpanis, J. Kelly, M. A. Brubaker, I. Gilitschenski, and A. Levinshtein, âSpin-nerf: Multiview segmentation and perceptual inpainting with neural radiance fields,â in CVPR, 2023, pp. 20 669â20 679. 14

[68] S. Szymanowicz, C. Rupprecht, and A. Vedaldi, âSplatter image: Ultra-fast single-view 3d reconstruction,â in CVPR, 2024, pp. 10 208â10 217. 14

[69] X. Zou, J. Yang, H. Zhang, F. Li, L. Li, J. Gao, and Y. J. Lee, âSegment everything everywhere all at once,â arXiv preprint arXiv:2304.06718, 2023. 14

[70] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng, âDreamgaussian: Generative gaussian splatting for efficient 3d content creation,â arXiv preprint arXiv:2309.16653, 2023. 14