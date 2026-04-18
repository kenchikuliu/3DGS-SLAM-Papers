# A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting

Ziyang Sun1, Lingfan Bao1 Tianhu Peng1 Jingcheng Sun1 Chengxu Zhou1,\*

1 Department of Computer Science, University College London, London, United Kingdom

\* Correspondence author; E-mail: chengxu.zhou@ucl.ac.uk.

## Highlights:

â¢ Proposes a novel, end-to-end pipeline leveraging 3DGS to create photorealistic, interactive digital twins from sparse RGB views in minutes. The system integrates perception, planning, real-robot execution ,and tested on real robot.

â¢ Introduces a robust method for semantic understanding by lifting 2D masks from foundation models like SAM into the 3D scene using multi-view consensus. It also contributes an efficient conversion of the 3DGS model into planning-ready collision geometry for physics-based simulation.

â¢ Demonstrates outstanding real-world performance and closes the sim-to-real gap. The framework is validated through physical experiments on a robotic manipulator, confirming that motion plans generated in the digital twin are effective for real-world execution.

## Abstract:

Developing high-fidelity, interactive digital twins is crucial for enabling closed-loop motion planning and reliable real-world robot execution, which are essential to advancing sim-to-real transfer. However, existing approaches often suffer from slow reconstruction, limited visual fidelity, and difficulties in converting photorealistic models into planning-ready collision geometry. We present a practical framework that constructs high-quality digital twins within minutes from sparse RGB inputs. Our system employs 3D Gaussian Splatting (3DGS) for fast, photorealistic reconstruction as a unified scene representation. We enhance 3DGS with visibility-aware semantic fusion for accurate 3D labelling and introduce an efficient, filter-based geometry conversion method to produce collision-ready models seamlessly integrated with a UnityâROS2âMoveIt physics engine. In experiments with a Franka Emika Panda robot performing pick-and-place tasks, we demonstrate that this enhanced geometric accuracy effectively supports robust manipulation in real-world trials. These results demonstrate that 3DGS-based digital twins, enriched with semantic and geometric consistency, offer a fast, reliable, and scalable path from perception to manipulation in unstructured environments.

Keywords: 3D Gaussian Splatting, Digital Twin, Robotic Manipulation, Real2Sim2Real

## 1. Introduction

The field of robotics is rapidly moving towards full autonomy in complex and unstructured environments. While, effective autonomous manipulation in unstructured environments fundamentally relies on the robotâs ability to rapidly construct a high-fidelity, actionable understanding of its surroundings, which is a core requirement for achieving advanced tasks such as fine-grained manipulation. This actionable understanding relies heavily on the construction of an accurate virtual replica, commonly known as a digital twin [1]. Digital twins are essential tools that enable safe, repeatable validation, closed-loop motion planning, and reliable sim-to-real transfer, which is important for advancing the deployment of robotic systems in the real world.

However, existing reconstruction pipelines [2] introduce significant bottlenecks that impede their seamless integration into real-time robotic workflows. The pursuit of visual fidelity often conflicts with the need for computational efficiency and physical utility. Specifically, Neural Radiance Fields (NeRFs) [3, 4] offer high photorealism but are computationally expensive, often requiring minutes to hours of time for optimization, which severely limits rapid deployment. Conversely, traditional methods based on point clouds or mesh reconstruction [5] are faster but often suffer from insufficient fidelity, noise sensitivity, and the difficulty of projecting reliable, consistent semantic labels from sparse multi-view inputs.

In this context, 3DGS [6], a novel explicit radiance field reconstruction method, has shown outstanding performance in both reconstruction quality and speed, emerged as a potential breakthrough representation. 3DGS successfully balances the trade-off between speed and fidelity, achieving photorealistic rendering quality within minutes, which is highly promising for rapid robotic scene capture. However, the core representation used by 3DGS, anisotropic Gaussian splats (or "balls"), defines each scene primitive not merely by a point but by a position, a covariance matrix, a colour, and an opacity. Although this representation excels at combining colour and alpha (opacity) components to create a visually convincing and continuous surface view, the underlying explicit geometry remains inherently ambiguous and problematic for physical interaction. Specifically, the resulting cloud of stretched Gaussian primitives is not a clean, watertight surface, but is riddled with reconstruction artefacts. These issues include floaters (isolated clusters derived from optimisation residuals), ghost artefacts (semi-transparent, low-opacity points near reflective or occluded boundaries), and overall surface fuzziness. These geometric imperfections, while visually hidden in the rendered image, render the raw 3DGS output unsuitable for precise robotic tasks such as collision checking and motion planning.

The fragmentation between these approaches highlights a critical, unsolved challenge: generating a digital twin that is simultaneously photorealistic, rapidly reconstructed, and equipped with planning-ready collision geometry and consistent semantic structures. Addressing this triple constraint, speed, fidelity, and actionability, is the central challenge for next-generation robotic perception systems.

To bridge the gap between visual fidelity, reconstruction efficiency, and physical utility, and thereby address the digital twins construction problem, we present an end-to-end framework that constructs interactive, semantically structured digital twins directly from sparse RGB inputs. This framework utilises 3DGS [6] as a unified representation to deliver a rapid real-to-sim-to-real pipeline. This leverages 3DGS for rapid, photorealistic reconstruction (addressing speed and fidelity) that provides the highfidelity visual experience. Furthermore, we introduce two key components to guarantee a geometrically interactive environment: a visibility-aware semantic fusion module that aggregates multi-view cues [7, 8] to ensure consistent 3D labelling (addressing semantic structure), which is then applied to the 3D Gaussian primitives. Following this, a geometric refinement process converts the semantically-labelled, but still noisy, Gaussian primitives into precise collision meshes (addressing planning-ready geometry). Finally, we deploy the generated assets into a specially designed Unity environment where and then robot can simulate with the digital twins, evaluating the manipulation process. Therefore, our framework proposes a unified closed-loop Real-to-simulated-to-real pipeline [9] illustrated in Fig. 1 for complex robotic manipulation. It seamlessly integrates photorealistic 3DGS reconstruction with our refined collision geometry, semantic understanding, and a Unity based Robot simulation/manipulation interface. This complete, validated sim-to-real workflow is a key advancement, ensuring that motion plans generated against the high-fidelity digital twin are reliable for real-world execution on the real robot.

The main contributions of this work are summarized as follows:

1. Unified Real-to-Sim-to-Real Framework: This work proposes a unified real-to-sim-to-real framework that synergizes 3DGS with robust point cloud processing. This approach generates actionable digital twins within minutes in a desk environment, effectively bridging the gap between neural rendering and robotic manipulation.

2. Visibility-Aware Semantic Fusion: This work introduces a view-dependent semantic aggregation with occlusion-aware confidence weighting strategy. This method distils 2D segmentation cues from vision foundation models [7, 10] into consistent 3D attributes, resolving projection ambiguities to ensure accurate labelling under occlusion and providing a semantic, manipulatable environment for the robot.

3. Planning-Ready Geometry Conversion: This work implements a multi-scale geometric filtering process with statistical outlier removal and adaptive mesh decimation. By employing alpha-based threshold and robust mesh generation algorithms, this step converts raw Gaussian splat point clouds into precise, planning-ready collision meshes.

4. Experimental Demonstration: This work demonstrates the systemâs practical efficacy through a complete perception-planning-execution loop. The framework verifies reliable performance on both standard benchmarks and complex real-world pick-and-place tasks using a Franka Emika robot [11].

The remainder of this paper is organised as follows. Section II reviews related work on 3D reconstruction, semantic understanding, and digital twins for manipulation. Section III presents the proposed methodology, including high-fidelity reconstruction, semantic projection, geometry generation, and system integration. Section IV describes the experimental setup and evaluation metrics, followed by Section V, which discusses the findings, limitations, and future research directions before concluding the paper.

<!-- image-->  
Figure 1. The overall pipeline of this framework uses multi-view video input and 3DGS to reconstruct the scene geometry. Grounded-SAM provides semantic masks, which are fused with the 3D projection to form a semantically-aware digital twin. This twin enables collision-aware motion planning for real robot manipulation.

## 2. Related Work

## 2.1. 3D Scene Reconstruction for Robotics

While dense mapping pipelines like TSDF [12] and Voxblox [13] have long served as the backbone for robotic navigation, their dependence on voxel discretization fundamentally limits their utility for manipulation. The resulting over-smoothed geometry fails to capture the high-frequency surface details necessary for fine motor control. Similarly, while implicit representations such as NeRFs [3] and Instant-NGP [14] offer visual photorealism, they remain constrained by prohibitive inference latencies and dense view requirements, creating a bottleneck for real-time robotic exploration.

Recently, 3DGS [6] has emerged to bridge this gap by explicitly representing scenes as anisotropic Gaussian primitives, combining differentiable optimization with fast rasterization, offering photorealistic rendering at real-time speeds. However, despite this visual fidelity, raw 3DGS representations are inherently ill-suited for physical interaction. The presence of reconstruction artifacts, such as floaters, ghost points, and surface fuzziness, causes the resulting point clouds functionally useless for direct collision checking and robot manipulation.

In contrast, our framework is specifically designed to close the gap between visual realism and physical validity. By systematically structuring raw 3DGS outputs, we transform noisy visual primitives into manipulation-ready geometry without sacrificing the rendering speed required for online operation.

## 2.2. Semantic Scene Understanding

Robotic manipulation requires a precise understanding of object identity beyond mere geometry. Although foundation models like SAM [7] and Grounded SAM [8] have revolutionized 2D segmentation, lifting these predictions into 3D space remains a critical challenge. Existing feature distillation methods, like, SegmentAnyGaussian [15], attempt to solve this by appending high-dimensional vectors to primitives, but this approach drastically increases memory footprints and training overhead, limiting deployment agility. Conversely, direct projection methods often suffer from "bleeding" labels and inconsistencies caused by depth discontinuities.

What sets our approach apart is that it diverges from these computationally heavy or unstable methods by introducing a Visibility-Aware Semantic Fusion module. Instead of relying on extensive retraining or naive projection, our pipeline is grounded in a rigorous geometric consensus mechanism. This method integrates a depth and visibility check, a confidence weighted voting scheme, ensuring that semantic labels are not just projected, but geometrically verified across views. This foundation empowers our system to achieve high-fidelity 3D labelling that is both consistent and computationally lightweight.

## 2.3. Digital Twins for Interactive Manipulation

The ultimate goal of robotic perception is to enable interaction. Ideally, a digital twin must unify three capabilities: photorealistic rendering, semantic understanding, and physical collision handling. Traditional simulators (e.g., Gazebo, MuJoCo) achieve physics but lack visual fidelity, while neural renderers achieve fidelity but lack physical structure. Bridging this "Sim-to-Real" gap requires a hybrid representation.

Recent efforts have begun to integrate 3DGS into planning frameworks. Splat-Nav [16] utilizes Gaussian representations for navigation, employing ellipsoid abstractions for collision checking. However, its focus is global path planning rather than the object-level granularity required for grasping. Similarly, RoboGSim [9] focuses on the simulation aspect, providing a platform for offline testing rather than an online perception-to-action pipeline. Other works like Splat-MOVER [17] and Grasp Splats [18] explore open-vocabulary manipulation and 3D feature splatting for grasping, respectively.

Despite these advancements, existing frameworks face fundamental gaps regarding manipulation. First, they lack robust mechanisms to convert raw, noisy 3DGS point clouds, often plagued by floaters and ghost artifacts, into planning-ready collision geometry. Second, they rarely provide validated sim-to-real pipelines that verify motion plans against this geometry prior to execution. Addressing these limitations requires an end-to-end framework that integrates sparse-view reconstruction [19], geometrically-verified semantic consensus, and physics-based post-processing [20, 21] into a unified, reliable robotic workflow.

Our proposed framework addresses this fragmentation by establishing a unified closed-loop Realto-Sim-to-Real pipeline Fig. 1. It seamlessly integrates photorealistic 3DGS reconstruction with our refined collision geometry (using Alpha Shapes Meshing), semantic understanding, and a Unity [22, 23] based Robot simulation/manipulation interface. This complete, validated sim-to-real workflow is a key advancement, ensuring that motion plans generated against the high-fidelity digital twin are reliable for real-world execution on the real robot.

## 3. Methodology

Our proposed framework establishes a comprehensive digital twin generation pipeline tailored for closedloop robotic manipulation. Follow the framework pipeline illustrated in Fig. 1, the system operates via two parallel processing streams: (1) a geometric reconstruction stream that uses an optimized 3DGS approach to generate a high-fidelity 3D scene, and (2) a semantic segmentation stream that identifies and isolates manipulable objects. The subsequent stages focus on rigorously transforming this semantically-annotated 3D model into clean, planning-ready collision geometry and integrating it into the simulation environment for validation.

## 3.1. High-Fidelity Scene Reconstruction

We employ a 3DGS-based approach for scene reconstruction due to its superior balance of rendering quality and rapid optimization speed. This choice is important for meeting the fast reconstruction speed and high quality digital twins, contrasting sharply with NeRFs-based methods which latest algorithms commonly require tens of minutes or more for comparable fidelity. To address the challenges posed by sparse and uncalibrated input images, which often lead to failures in traditional Structure-from-Motion (SfM) pipelines, we use the InstantSplat methodology [19]. This streamlined approach eliminates the need for a separate SfM step by utilizing a pre-trained geometric prior, such as MASt3R [24], to directly estimate an initial point cloud and camera poses.

The core of the reconstruction remains an end-to-end, self-supervised optimization process. The scene is represented by a set of 3D gaussians, each defined by a position $\mathbf { \mu } _ { \mu } ,$ a covariance matrix Î£, a colour, and an opacity. The unnormalized density is given by [6]:

$$
G ( { \pmb x } ) = \exp \left( - \frac { 1 } { 2 } ( { \pmb x } - { \pmb \mu } ) ^ { \top } { \pmb \Sigma } ^ { - 1 } ( { \pmb x } - { \pmb \mu } ) \right)\tag{1}
$$

This set of gaussians is jointly optimized with the camera poses to minimize the photometric rendering error between the rendered images and the input views. This technique bypasses the traditional, timeconsuming adaptive density control steps of vanilla 3DGS, enabling extremely fast convergence and yielding a high-fidelity 3D representation suitable for photorealistic rendering and depth extraction.

## 3.2. Spatially-Consistent Semantic Lifting

Achieving a reliable, actionable semantic understanding is the prerequisite for robot interaction. However, a fundamental conflict arises when lifting 2D perception to 3D: standard single-view segmentation models (e.g., SAM) suffer from the "bleeding effect," where background pixels near the object boundary are erroneously included in the foreground mask. To resolve this, we propose a Spatially-Consistent Semantic Lifting framework. Unlike naive projection methods, our approach treats 2D masks as noisy spatial hypotheses and enforces 3D geometric consistency to filter out segmentation outliers.

Spatial Isolation via Depth Clustering The core premise of our method is that semantic coherence implies spatial coherence. While a 2D segmentation mask $M _ { j }$ may loosely cover both the target object and the adjacent background, the underlying 3D geometry exhibits a distinct depth discontinuity.

To exploit this, we perform Depth-Guided Isolation for each view. We project the 3D Gaussians contained within the 2D mask and apply density-based clustering (DBSCAN) on their depth values. We assume the largest cluster corresponds to the true object geometry, while smaller, spatially detached clusters represent background artifacts included by the 2D model.

Confidence-Weighted Consensus To aggregate these observations into a unified 3D semantic field, we employ a weighted voting mechanism governed exclusively by spatial validity. We define the fusion weight $W _ { i , j }$ for Gaussian i in view j as:

$$
W _ { i , j } = w _ { \mathrm { c l u s t e r } } ( i , j )\tag{2}
$$

Here, $w _ { \mathrm { c l u s t e r } }$ acts as a soft spatial gate. Points belonging to the primary depth cluster are assigned high confidence $( w _ { \mathrm { c l u s t e r } } \approx 1 )$ , while spatial outliers are suppressed $( w _ { \mathrm { c l u s t e r } } \approx 0 )$ . This formulation is robust against the geometric ambiguities of the 2D masks, ensuring that only points physically co-located with the object contribute to the semantic label.

The final semantic label for a 3D point $\mathbf { p } _ { i }$ is determined by accumulating these spatially-weighted votes across all visible views $V _ { i } ^ { \prime } \colon$

$$
\begin{array} { r } { \mathbf p _ { i } \in \left\{ \begin{array} { l l } { \mathcal { P } _ { \mathrm { o b j } } } & { \mathrm { i f } \sum _ { j \in V _ { i } ^ { \prime } } W _ { i , j } \cdot \mathbb { I } \big ( \big ( u _ { i } , \nu _ { i } \big ) \in M _ { j } \big ) \geq \tau _ { \mathrm { c o n s e n s u s } } } \\ { \mathcal { P } _ { \mathrm { b g } } } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}\tag{3}
$$

where $\tau _ { \mathrm { { c o n s e n s u s } } }$ is an adaptive threshold. This consensus mechanism effectively "carves" the correct semantic shape out of the noisy 2D predictions.

Iterative Semantic Refinement To further sharpen the boundaries, we implement an iterative feedback loop (max_iter = 3). As the 3D semantic model improves, it generates cleaner depth maps for the next iterationâs visibility checks. We incorporate a Boundary Refinement step using K-Nearest Neighbors (KNN) to smooth local label inconsistencies, ensuring continuous surface semantics.

## 3.3. Physics-Ready Geometry Reconstruction

Following semantic fusion, the scene is partitioned into the target object $\mathcal { P } _ { \mathrm { o b j } }$ and the environmental background $\mathcal { P } _ { \mathrm { b g } }$ . However, semantic identity does not guarantee geometric utility. The raw 3DGS representation is plagued by "visual fog", including low-density floaters, semi-transparent ghosts, and stretched needle-like artifacts, which creates false obstacles for the physics engine.

To resolve this problem, we implement a three-stage reconstruction pipeline applied identically to both the object and background point clouds. This process systematically converts the noisy visual representation into a clean, collision-free physical environment.

## 3.3.1. Stage 1: Intrinsic Attribute Filtering

The initial phase acts as a global statistical cleaner, removing primitives that contribute to visual haze but lack physical substance. We apply two rigorous filters to the entire scene:

â¢ Opacity Threshold: We discard primitives with low opacity $( \mathbf { e . g . } , \alpha < 0 . 1 )$ . This effectively eliminates the semi-transparent "mist" often found hovering above surfaces.

â¢ Geometric Regularization: We analyse the covariance scales to identify and remove overly stretched, "needle-like" primitives. These artifacts, common in sparse-view areas, are pruned to prevent the physics engine from registering false collisions with non-existent spikes.

## 3.3.2. Stage 2: Semantic-Guided Connectivity Pruning

Even after statistical filtering, isolated clusters of noise (floaters) may persist. We leverage the semantic prior established in the previous section to perform topological cleaning. For any given semantic partition (whether object or background), we assume the physical entity corresponds to the dominant geometric structure.

We employ DBSCAN clustering to segment the point cloud into spatially disjoint groups. By retaining only the largest connected cluster and aggressively pruning all smaller detached components, we effectively wipe out floating artifacts. This ensures that $\mathcal { P } _ { \mathrm { o b j } }$ resolves to a single coherent object and $\mathcal { P } _ { \mathrm { b g } }$ resolves to a clean static environment (e.g., the table surface), free from phantom obstacles.

## 3.3.3. Stage 3: Watertight Meshing via Alpha Shapes

Finally, to bridge the gap to robotic manipulation, we convert the cleaned point geometry into a mesh. We select the alpha shapes algorithm, which functions as a "shrink-wrap" operation. Unlike implicit smoothing methods, alpha shapes tightly conform to the point distribution, preserving sharp geometric features, such as box corners and handle edges, that are critical for stable grasping contact.

## 3.4. Interactive Digital Twin and Planning

The semantically-segmented and geometrically refined 3D models are imported into the Unity engine. The background point cloud $\mathcal { P } _ { \mathrm { b g } }$ forms the static environment. Each manipulable object in $\mathcal { P } _ { \mathrm { o b j } }$ is assigned a MeshCollider for accurate collision detection and a Rigidbody for realistic physicsbased interactions. The Unity environment acts as the digital twin.

We establish a seamless, high-bandwidth communication bridge between Unity and the standard robotic software stack (ROS 2) using the ROS2ForUnity plug-in. This enables bidirectional state synchronization: the robotâs state is sent to Unity, and the digital twinâs environment geometry and object poses (derived from the segmented point clouds) are dynamically sent to the MoveIt 2 [11] planning scene. The system uses this complete and rapidly generated information to perform collision-aware motion planning. Generated trajectories are first validated in the physics-enabled simulation before being sent to the physical robot for execution, forming a reliable perception-to-planning-to-validation sim-to-real workflow.

It is worth noting that the scope of this work is currently limited to static scenes. This design choice prioritizes reconstruction efficiency and geometric stability, key requirements for the proposed rapid scan-and-plan workflow, over the computational complexity associated with dynamic modeling. Furthermore, the static assumption remains valid for the targeted tabletop rearrangement tasks, where the environment is assumed to remain stable during the planning phase.

## 4. Experiments and Results

To comprehensively validate the effectiveness and robustness of our proposed high-fidelity 3DGS digital twin framework for robotic manipulation tasks, we designed and conducted a series of rigorous quantitative

experiments. This section details the experimental setup, task definition, baselines for comparison, and quantitative analysis of key evaluation metrics.

## 4.1. Experimental Setup

Our experimental platform centres on a Franka Emika 7-DOF robot arm equipped with an Intel RealSense D435i RGB-D camera mounted on its end-effector. The camera provides high-resolution colour images that serve as input for our 3DGS framework. All computations were performed on a workstation equipped with an NVIDIA GeForce RTX 4090 GPU, ensuring rapid 3DGS training and rendering to meet the demanding requirements for reconstruction efficiency.

To systematically evaluate reconstruction robustness across varying geometric and optical challenges, we selected eight representative objects spanning three difficulty levels. The L1-Basic category includes a Blue Box and Yellow Cube, featuring convex geometry with Lambertian surfaces that serve as baseline objects. The L2-Complex category comprises a Toy Hammer and Scissors, presenting non-convex shapes with thin structures that challenge geometric reconstruction. The L3-Textured category contains a Diet Coke Bottle, Glue Stick, and Pen, exhibiting high-frequency surface details that test the frameworkâs ability to capture fine visual features. These objects enable targeted analysis of reconstruction quality, semantic segmentation accuracy, and geometric fidelity across distinct challenge categories.

As illustrated in Fig. 2, we constructed a challenging, unstructured tabletop scene to test the systemâs zero-shot generalization capabilities. The scene includes objects varying in geometry, texture, and function: a toy hammer with complex geometric shape, a simple-surfaced blue box, and a small yellow cube. Additionally, a cardboard box serves dual rolesâacting as a static obstacle initially and subsequently becoming a target placement area. This dynamic role assignment tests the systemâs adaptability to environmental changes.

<!-- image-->  
(a) Unity digital twin

<!-- image-->  
(b) Rviz visualization

<!-- image-->  
(c) Real robot setup  
Figure 2. Integration and validation of the digital twin framework across simulation and reality. The Unity view Fig.2a shows the high-fidelity, photorealistic digital twin built with 3DGS and integrated with the physics engine. This model generates and validates collision-aware motion plans visualized in the Rviz interface Fig.2b, which uses simplified geometry for MoveIt planning. The validated plan is then executed by the real Franka Emika robot Fig.2c, completing the sim-to-real workflow.

## 4.1.1. Task Definition

The core evaluation task is defined as a long-horizon, zero-shot rearrangement comprising three sequential manipulation steps. First, in the object-obstacle interaction phase, the robot grasps the blue box and places it atop the cardboard box, testing planning and manipulation capabilities in the presence of obstacles. Second, the object-object interaction phase requires grasping the yellow cube and placing it on the blue box, demanding accurate perception and interaction with previously moved objects. Third, the irregular object manipulation phase involves grasping the geometrically complex toy hammer and placing it within a designated target frame, evaluating robustness in reconstructing and manipulating irregular shapes. Successful execution requires proactive planning in a dynamic environment rather than purely reactive perception-based control. Since all objects and scene layouts were unseen during system development, this constitutes a zero-shot manipulation problem.

## 4.1.2. Comparison Methods

To quantitatively evaluate our approach, we compared against two representative baselines focusing on 3D reconstruction fidelity and efficiency. Our 3DGS-based method performs a single scene scan using sparse multi-view RGB images (10â20 views). This view count was experimental determined to represent the optimal trade-off for the "scan-and-plan" workflow: it provides sufficient parallax for robust reconstruction while keeping the robotic data acquisition time within a practical minimum. This allows us to rapidly build a high-fidelity digital twin in minutes, which is then imported into the Unity physics engine for complete planning and pre-validation.

Baseline 1 employs traditional point cloud reconstruction using the Intel RealSense D435i to perform multi-view depth fusion, establishing a benchmark for reconstruction efficiency and geometric accuracy. Baseline 2 utilizes a state-of-the-art NeRFs framework (Instant-NGP) for 3D scene reconstruction from identical sparse multi-view RGB images, enabling a direct comparison of efficiency and photorealistic quality between 3DGS and NeRFs approaches in this sparse-data regime.

## 4.1.3. Evaluation Metrics

Our evaluation encompasses five categories of metrics. For reconstruction fidelity and efficiency, we measure reconstruction time from image capture to model completion, along with Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) computed on held-out novel views. Semantic segmentation accuracy is assessed through 3D projection consistency score (percentage of points with consistent foreground classification across views), ghost index (artifact rate in background regions), and mean Intersection over Union (mIoU) against manually annotated ground truth. Geometric fidelity is evaluated using Chamfer Distance, Precision, and F1-Score against manually cleaned ground truth models. Manipulation task performance is measured through end-to-end success rate, collision count, and placement error (Euclidean distance between achieved and target object positions).

For the consistency and ghost metrics specifically, given a 3D point p visible in $N _ { p }$ views (visibility determined by depth consistency within tolerance $\tau _ { \mathrm { d e p t h } } = 5$ mm), with $N _ { p } ^ { \mathrm { f g } }$ views classifying it as foreground, we define the consistency score as Consistency $( p ) = N _ { p } ^ { \mathrm { f g } } / N _ { p }$ . The dataset-level consistency is the percentage of points achieving Consistency $( p ) \geq 0 . 8$ . The ghost index measures artifact introduction as the percentage increase in foreground points when relaxing the voting threshold, normalized by total point count.

## 4.1.4. Evaluation Metrics

Our evaluation encompasses five categories of metrics. For reconstruction fidelity and efficiency, we measure reconstruction time from image capture to model completion, along with Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) computed on held-out novel views.

To assess semantic segmentation accuracy, we evaluate mean Intersection over Union (mIoU) against manually annotated ground truth, alongside two custom consistency metrics. First, we define the 3D Projection Consistency score. Let $\mathcal { V } _ { p }$ be the set of views where a 3D point p projects strictly within the image sensor boundaries. We denote the number of valid observations as $N _ { p } = | \mathcal { V } _ { p } |$ . Within these valid views, let $N _ { p } ^ { \mathrm { f g } }$ be the count of views where the projected pixel falls within the foreground mask region. The consistency score is defined as Consistency $\mathbf { \Psi } ( \mathbf { p } ) = N _ { p } ^ { \mathrm { f g } } / N _ { p }$ . We report the dataset-level consistency as the percentage of valid points (where $N _ { p } > 0 )$ that achieve a high confidence of Consistency $\mathbf { \delta } ( \mathbf { p } ) \geq 0 . 8$ Second, the Ghost Index measures artifact introduction in background regions, defined as the percentage increase in foreground points when relaxing the voting threshold, normalized by the total point count.

Geometric fidelity is evaluated using Chamfer Distance, Precision, and F1-Score against manually cleaned ground truth models. Finally, manipulation task performance is measured through end-to-end success rate, collision count, and placement error (Euclidean distance between achieved and target object positions).

Table 1. Reconstruction performance comparison. Results averaged across test scenes with varying complexity (10â20 input views, 3000 iterations). Times measured on NVIDIA RTX 4090. PSNR/SSIM computed on 5 held-out novel views.
<table><tr><td>Method</td><td>Time (s) â</td><td>PSNR (dB) â</td><td>SSIM â</td></tr><tr><td>Point Cloud</td><td>~20</td><td>N/A</td><td>N/A</td></tr><tr><td>NeRFs (Instant-NGP)</td><td> $1 1 2 3 \pm 2 0 0$ </td><td> $2 8 . 3 3 \pm 1 . 0$ </td><td> $0 . 9 0 3 7 \pm 0 . 0 2 3$ </td></tr><tr><td>Ours (3DGS)</td><td> ${ \bf 2 2 9 \pm 1 2 0 }$ </td><td> ${ \bf 3 7 . 0 3 \pm 5 . 0 }$ </td><td> $\mathbf { 0 . 9 8 2 1 } \pm \mathbf { 0 . 0 1 1 }$ </td></tr></table>

## 4.2. Reconstruction Performance

The quantitative results presented in Table 1 demonstrate the superior performance of our proposed 3DGSbased method in both efficiency and rendering quality. Our method achieves an average reconstruction time of 229 seconds across test scenes with varying input sparsity (10â20 views, 3000 iterations), representing a 5Ã speed-up over the NeRFs baseline (1123 seconds) and effectively reducing the reconstruction bottleneck from hours to minutes suitable for rapid deployment. Reconstruction time scales with scene complexity, ranging from 109 seconds for simple textured objects to 349 seconds for scenes with complex geometry or view-dependent effects.

Regarding rendering quality, our method achieves an average PSNR of 37.03 dB and SSIM of 0.9821, representing an 8 dB improvement over NeRFs (28.33 dB / 0.9037). Performance varies predictably with object category: well-textured objects achieve exceptional quality (PSNR 41.34 dB, SSIM 0.9934) with sparse inputs, preserving fine details such as legible text and sharp edges, while geometrically simpler objects with uniform surfaces yield intermediate performance around 36.77 dB. Although traditional point cloud approaches offer faster processing (approximately 20 seconds), they lack the photorealistic textures essential for high-fidelity digital twins and suffer from significant drift on consumer-grade depth sensors, rendering them unsuitable for our application.

## 4.3. Semantic Segmentation Accuracy

## 4.3.1. Multi-View Consistency Analysis

A core challenge in lifting 2D masks to 3D is balancing completeness with noise suppression. Table 2 presents the trade-off between consistency score and ghost index under different voting thresholds, where N denotes the number of visible views. A loose threshold (N/2.0) achieves low artifact rate (ghost index 22.48%) but suffers from incomplete segmentation (82.41% consistency), often missing object boundaries. Conversely, a strict threshold (N/1.0) guarantees 100% consistency but introduces excessive noise (ghost index 67.23%), generating floating obstacles that interfere with motion planning. We selected N/1.5 as the operating point, achieving 93.72% consistency while maintaining the ghost index below 50%, ensuring robust 3D object definition without compromising the free space required for collision-free planning.

Table 2. Multi-view consistency versus artifact rate at different voting thresholds. Results measured on 8-object benchmark with 15 views per scene.
<table><tr><td>Voting Threshold</td><td>Consistency (%) â Ghost Index (%) â</td></tr><tr><td>N /2.0 82.41</td><td>22.48</td></tr><tr><td>N/1.8</td><td>87.69 37.42</td></tr><tr><td>N/1.5</td><td>93.72 46.27</td></tr><tr><td>N/1.2</td><td>100.0 53.19</td></tr><tr><td>N/1.0</td><td>100.0 67.23</td></tr></table>

## 4.3.2. Overall Semantic Quality

To assess end-to-end semantic understanding, we computed mean Intersection over Union (mIoU) of projected masks against manually annotated ground truth. Our framework achieves a 2D segmentation mIoU of 0.87 averaged across all views and objects, and 3D projection consistency reaches 0.93. These results confirm that our multi-view fusion approach effectively bridges 2D perception and 3D geometric reconstruction, providing a reliable semantic layer for robotic manipulation.

## 4.4. Ablation Study on Point Cloud Cleaning

To validate the effectiveness of each component within our cleaning pipeline, we conducted a rigorous ablation study on four representative objects from the L1-Basic and L2-Complex categories (Blue Box, Yellow Cube, Toy Hammer, Scissors). The raw point cloud generated by 3DGS typically contains floaters and ghosting artifacts that compromise geometric accuracy.

We compared four configurations: the original 3DGS output serving as baseline; de-noise only, applying attribute-based filtering (opacity and colour thresholds); cluster only, applying DBSCAN spatial

<!-- image-->  
Figure 3. Qualitative efficacy of the point cloud cleaning pipeline. Top: Raw 3DGS point clouds exhibiting floaters and surface fuzziness, which impede precise collision checking. Bottom: Refined geometries after applying our multi-stage filtering (heuristic filtering and DBSCAN). The process effectively removes artifacts and sharpens boundaries, yielding planning-ready digital twins for manipulation tasks.

clustering (eps = 0.02, min_samples = 10); and our full method combining both components sequentially.   
Results are presented in Table 3.

The results reveal the contribution of each component. Applying clustering alone provides noticeable improvement over baseline, reducing Chamfer distance from 0.0052 to 0.0043 and increasing F1-Score from 0.9369 to 0.9429, indicating effectiveness in removing spatially isolated noise. Interestingly, applying de-noising in isolation yields no improvement (Chamfer distance slightly increased to 0.0055), suggesting that attribute-based filtering alone cannot handle complex artifacts where floaters remain spatially connected to the main body. However, the full method achieves dramatic performance gains, reducing Chamfer distance to 0.0020 and elevating F1-Score to 0.9989. This demonstrates a synergistic effect: clustering first removes primary outliers, enabling the de-noising module to effectively filter subtle attribute-based artifacts near the main object. This two-stage process is essential for producing high-fidelity point clouds required for reliable manipulation.

Table 3. Ablation study on geometric fidelity of the cleaning pipeline. Results averaged across four test objects. Ground truth obtained via manual cleaning in CloudCompare. Chamfer Distance computed at 1mm resolution.
<table><tr><td>Method</td><td>Chamfer Dist. â</td><td>Precision â</td><td>F1-Score â</td></tr><tr><td>Original</td><td>0.0052</td><td>0.8846</td><td>0.9369</td></tr><tr><td>De-noise Only</td><td>0.0055</td><td>0.8838</td><td>0.9369</td></tr><tr><td>Cluster Only</td><td>0.0043</td><td>0.8958</td><td>0.9429</td></tr><tr><td>De-noise + Cluster</td><td>0.0020</td><td>0.9977</td><td>0.9989</td></tr></table>

## 4.5. Real-World Robotic Validation

This framework finally validate on a Franka Emika arm to show the ability to enable successful real-world manipulation. We conducted 10 independent trials of the long-horizon rearrangement task, with the complete execution sequence visualized in Fig. 4.

Success Criteria To ensure rigorous evaluation, we define a trial as successful only if it meets three conditions: (1) the robot successfully detects and grasps the correct target object; (2) the object is transported and placed stably within the designated goal region; and (3) the entire trajectory is collisionfree with respect to both static obstacles and the environment.

<!-- image-->

(a) Real robot execution  
<!-- image-->  
(b) Digital twin simulation  
Figure 4. Execution sequence of the multi-step rearrangement task in (a) the real world and (b) the digital twin. The robot grasps the blue box and places it on the cardboard box, then grasps the yellow cube and stacks it on the blue box, and finally grasps the toy hammer and places it in the target area. This demonstrates the frameworkâs capability for complex, zero-shot manipulation with proactive planning validated in simulation.

Results Analysis Under these strict criteria, our framework achieved a 100% success rate in simulation validation and a 90% success rate in real-world execution (9/10 trials). The solitary failure occurred during the grasping attempt of the 2.5 cm yellow cube. Due to the objectâs diminutive scale, a minor gripper alignment error resulted in a missed grasp. This failure case highlights the high-precision challenges inherent in manipulating small-scale objects that approach the resolution limits of the 3DGS-based geometric reconstruction and gripper finger geometry.

In terms of placement accuracy, qualitative assessment confirmed that all manipulated objects were correctly deposited strictly within the designated regions. While exact metric error was not instrumented, this consistent alignment demonstrates that the system effectively satisfied the spatial tolerances required for the rearrangement task. It is important that, zero collisions were observed during any trial, validating the high fidelity of our collision geometry generation. These results demonstrate that 3DGS-based digital twins, combine with semantic and geometric consistency, provide a reliable foundation for complex manipulation in unstructured environments.

## 5. Conclusion and Future Direction

This paper presented an end-to-end framework for rapidly creating high-fidelity, interactive digital twins for robotic manipulation from sparse RGB views. The approach combines 3DGS [6] for fast photorealistic reconstruction, Grounded-SAM [8] for zero-shot semantic segmentation, and a novel filtering pipeline to generate clean, planning-ready collision geometries. The system achieves reconstruction speeds of under 4 minutes (avg. 229 s) with high visual fidelity (36.35 dB PSNR), representing a 5-fold speed-up over NeRFs-based approaches while maintaining comparable quality.

Real-world experiments on a long-horizon rearrangement task demonstrate the practical utility of the framework. Motion plans generated and validated within the digital twin achieved a 90% success rate when executed on a Franka Emika robot, with zero collisions in successful trials and an average placement error of 0.83 cm. These results provide strong evidence that the framework effectively addresses the sim-to-real gap, enabling reliable robot operation in unstructured environments.

The multi-stage de-noising and meshing pipeline proved essential for converting the unstructured 3DGS output into planner-compatible geometries. The ablation study confirms that both heuristic filtering and cluster-based de-noising contribute synergistically, achieving a near-perfect F1-score of 0.9989 against manually cleaned ground truth models. This gaussian-to-mesh conversion represents a critical bridge between modern neural rendering and traditional motion planning frameworks.

Several directions warrant future investigation. First, the current framework assumes static scenes and performs a one-time reconstruction. Integrating dynamic 3DGS variants [25] or implementing continuous update mechanisms would enable the digital twin to remain consistent with evolving environments. Second, the system currently models only geometry and appearance. Incorporating methods for online physical property estimation [26] would enable more sophisticated manipulation strategies involving contact-rich interactions. Third, while this work focuses on motion planning given predefined grasps, integrating robust grasp planning modules [18] that operate directly on gaussian representations would further automate the pipeline.

A particularly promising direction involves leveraging the digital twin as an enabler for learned policies. The framework could serve dual purposes: as a safety validation platform where policies are tested through thousands of simulated iterations before deployment, and as a data generation engine that autonomously creates diverse training datasets without physical resource consumption. This capability could significantly accelerate the development of vision-language-action models and reinforcement learning approaches for manipulation.

The work demonstrates that the synergy between 3DGS efficiency, foundation model capabilities, and robust geometric processing provides a practical paradigm for robotic manipulation in unstructured environments. By unifying perception, reconstruction, and planning into a closed-loop system, the framework represents a step toward autonomous robots that can rapidly adapt to novel surroundings with both speed and reliability.

## 6. Supplementary data

The authors confirm that the supplementary data are available within this article.

## Acknowledgments

This work was supported by the Advanced Research and Invention Agency [grant number SMRB-SE01- P06].

## Authorâs contribution

Ziyang Sun: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing â original draft.

Lingfan Bao: Methodology, Validation, Writing â review & editing.

Tianhu Peng: Writing â review & editing.

Jingcheng Sun: Writing â review & editing.

Chengxu Zhou: Conceptualization, Resources, Supervision, Project administration, Funding acquisition, Writing â review & editing.

## Conflicts of Interests

The authors declare no competing interests.

## References

[1] J. Li and S. X. Yang, âDigital twins to embodied artificial intelligence: review and perspective,â Intelligence & Robotics, vol. 5, no. 1, 2025.

[2] L. Zhou, G. Wu, Y. Zuo, X. Chen, and H. Hu, âA comprehensive review of vision-based 3d reconstruction methods,â Sensors, vol. 24, no. 7, 2024.

[3] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â 2020.

[4] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-nerf: A multiscale representation for anti-aliasing neural radiance fields,â 2021.

[5] C. Lv, W. Lin, and B. Zhao, âVoxel structure-based mesh reconstruction from a 3d point cloud,â IEEE Transactions on Multimedia, vol. 24, p. 1815â1829, 2022.

[6] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering,â 2023.

[7] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. DollÃ¡r, and R. Girshick, âSegment anything,â 2023.

[8] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, Z. Zeng, H. Zhang, F. Li, J. Yang, H. Li, Q. Jiang, and L. Zhang, âGrounded sam: Assembling open-world models for diverse visual tasks,â 2024.

[9] X. Li, J. Li, Z. Zhang, R. Zhang, F. Jia, T. Wang, H. Fan, K.-K. Tseng, and R. Wang, âRobogsim: A real2sim2real robotic gaussian splatting simulator,â 2025.

[10] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. RÃ¤dle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. DollÃ¡r, and C. Feichtenhofer, âSam 2: Segment anything in images and videos,â 2024.

[11] D. Coleman, I. Sucan, S. Chitta, and N. Correll, âReducing the barrier to entry of complex robotic software: a moveit! case study,â 2014.

[12] B. Curless and M. Levoy, âA volumetric method for building complex models from range images,â in Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pp. 303â312, ACM, 1996.

[13] H. Oleynikova, Z. Taylor, M. Fehr, R. Siegwart, and J. Nieto, âVoxblox: Incremental 3d euclidean signed distance fields for on-board mav planning,â in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 1366â1373, IEEE, 2017.

[14] T. MÃ¼ller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics primitives with a multiresolution hash encoding,â ACM Transactions on Graphics, vol. 41, p. 1â15, July 2022.

[15] J. Cen, J. Fang, C. Yang, L. Xie, X. Zhang, W. Shen, and Q. Tian, âSegment any 3d gaussians,â 2025.

[16] T. Chen, O. Shorinwa, J. Bruno, A. Swann, J. Yu, W. Zeng, K. Nagami, P. Dames, and M. Schwager, âSplat-nav: Safe real-time robot navigation in gaussian splatting maps,â 2025.

[17] O. Shorinwa, J. Tucker, A. Smith, A. Swann, T. Chen, R. Firoozi, M. K. III, and M. Schwager, âSplat-mover: Multi-stage, open-vocabulary robotic manipulation via editable gaussian splatting,â 2024.

[18] M. Ji, R.-Z. Qiu, X. Zou, and X. Wang, âGraspsplats: Efficient manipulation with 3d feature splatting,â 2024.

[19] Z. Fan, K. Wen, W. Cong, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos, Z. Wang, and Y. Wang, âInstantsplat: Sparse-view gaussian splatting in seconds,â 2025.

[20] M. Kazhdan, M. Bolitho, and H. Hoppe, âPoisson surface reconstruction,â in Proceedings of the fourth Eurographics symposium on Geometry processing, pp. 61â70, Eurographics Association, 2006.

[21] A. GuÃ©don and V. Lepetit, âSugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering,â 2023.

[22] A. Pranckevicius, âUnitygaussiansplatting.â https://github.com/aras-p/UnityGaussianSplatting, 2024.

[23] Robotec.AI, âRos2 for unity.â https://github.com/RobotecAI/ros2-for-unity, 2024. Accessed: 2025-04-28.

[24] V. Leroy, Y. Cabon, and J. Revaud, âGrounding image matching in 3d with mast3r,â 2024.

[25] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, âDrivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,â 2023.

[26] A. Cherian, R. Corcodel, S. Jain, and D. Romeres, âLlmphy: Complex physical reasoning using large language models and world models,â 2024.