# ObserverâActor: Active Vision Imitation Learning with Sparse-View Gaussian Splatting

Yilong Wang, Cheng Qian, Ruomeng Fan, and Edward Johns

<!-- image-->  
Fig. 1: Active vision for imitation learning in a mug-handle pickup task across five scenarios. When a static camera struggles (top row), alternative placements (bottom row, and coloured frustums in top row) provide better observations. In our method, at test time an observer robot (robot on right in above examples) computes and moves to such an optimal view from its wrist-cam, after which an actor robot (robot on left in above examples) performs the task conditioned on this view.

Abstractâ We propose Observer-Actor (ObAct), a novel framework for active vision imitation learning in which the observer moves to optimal visual observations for the actor. We study ObAct on a dual-arm robotic system equipped with wristmounted cameras. At test time, ObAct dynamically assigns observer and actor roles: the observer arm constructs a 3D Gaussian Splatting (3DGS) representation from three images, virtually explores this to find an optimal camera pose, then moves to this pose; the actor arm then executes a policy using the observerâs observations. This formulation enhances the clarity and visibility of both the object and the gripper in the policyâs observations. As a result, we enable the training of ambidextrous policies on observations that remain closer to the occlusion-free training distribution, leading to more robust policies. We study this formulation with two existing imitation learning methods â trajectory transfer and behaviour cloning â and experiments show that ObAct significantly outperforms static-camera setups: trajectory transfer improves by 145% without occlusion and 233% with occlusion, while behavior cloning improves by 75% and 143%, respectively. Videos are available at https://obact.github.io.

## I. INTRODUCTION

Current imitation learning methods for robotic manipulation primarily rely on static cameras [1, 2], egocentric wristmounted cameras [3, 4], or a combination of both [5, 6]. Static cameras are typically placed in a task-agnostic manner, and once a policy is trained, it often must be deployed in the same setup. Wrist-mounted cameras offer greater flexibility but suffer from limited global awareness and a restricted field of view. Combining both viewpoints can partially mitigate these issues but often introduces redundant observations that may distract the modelâespecially in low-data regimes [7,

8]. These limitations are further amplified by occlusions, highlighting the need for active vision, where the camera can move dynamically to provide the best viewpoint for the task.

Realizing active vision requires both movable camera hardware and sophisticated active vision strategies. Recent studies [7, 8, 9] achieve this by developing a dedicated activevision arm for perception, with its policy learned through teleoperation and behavior cloning. However, this approach has two significant limitations. First, the active vision arm is restricted to a fixed observer role, limiting its reachable viewpoints and preventing its use as regular manipulator. Second, the active vision strategy demands extensive human demonstrations, which, although collectable via a VR-based teleoperation system, imposes additional burden on operators.

In this paper, we present ObAct (short for ObserverâActor), a novel framework for active vision in imitation learning, where an observer robot computes optimal visual observations that guide the actor robotâs actions. As illustrated in Figure 1 with a mug-handle pickup task, ObAct positions the observer arm at test-time optimal viewpoints, providing visual observations for the actor arm during manipulation. Unlike prior approaches, ObAct dynamically assigns observer and actor roles based on the scene configuration, without requiring a separately trained active vision strategy for each arm. To enable this, we construct a test-time 3D Gaussian Splatting (3DGS) representation [10] from sparse-view images and optimize the camera pose to obtain observations that are similar to the demonstration observations, while minimizing occlusions. We further extend existing imitation learning methodsâtrajectory transfer [2] and behavior cloning [5]âto leverage this view-conditioned framework. Experiments show that ObAct substantially improves success rates over staticcamera baselines: trajectory transfer increases by 145% without occlusion and 233% with occlusion, while behavior cloning increases by 75% and 143%, respectively.

In summary, our contributions are threefold:

1) ObAct Framework: We introduce a decoupled observerâactor framework for active vision imitation learning, which allows the system to be robust against visual edge cases that static cameras cannot.

2) Active Vision via Sparse-view 3DGS: We develop an RGB-based active vision system that uses a test-time 3DGS model, constructed from sparse-view images, to optimize viewpoints for demonstration consistency and occlusion reduction. To our knowledge, this is the first use of sparse-view 3DGS in active vision.

3) Active Vision for Imitation Learning: We extend trajectory transfer and behavior cloning to the active vision setting, demonstrating substantial performance gains and, for behavior cloning, improved data efficiency.

## II. RELATED WORK

Active Vision for Robotic Manipulation. Active vision (AV) [11] refers to actively moving the camera to maximize perceptual quality for downstream tasks, and has been widely applied in robotics for object recognition [12], pose estimation [12, 13], and 3D reconstruction [14, 15]. It has also been extensively studied in robotic grasping, where unobserved views are sampled and evaluated using depth images based on information gain [16], grasp affordances [17], graspness inconsistency [18], and velocity fields [19], yielding improved performance in occluded and cluttered environments. In contrast, our method renders and evaluates unobserved viewpoints directly in the RGB domain, the prevalent modality in modern imitation learning systems [1, 5], and extends beyond grasping to support a broader range of manipulation tasks. Our setup is most closely related to [20], where an agent first observes the environment and then interacts with it by jointly training a next-best-view camera policy and a next-best-pose gripper policy using few-shot reinforcement learning. However, their approach assumes a freely moving camera and is validated only in simulation, whereas we demonstrate a real-world dual-arm system that achieves active vision without requiring a separate policy. In concurrent work, [21] also avoid training by leveraging 3DGS to build a scene model from dense scans and applying Bayesian optimization to determine an initial robot pose for task execution on a mobile platform. However, our focus is on sparse-view inputs in a dual-arm setting. [22] capture demonstrations with a 360Â° camera and jointly train an eyeball gaze controller and arm policies using a BCâRL loop, but their 2-DoF gaze mechanism is less capable of handling occlusions compared to our 6-DoF observer arm.

View-conditioned Imitation Learning. For manipulation systems to generalize, visuomotor policies must remain robust to discrepancies in observational viewpointsâparticularly unseen camera posesâwhich is especially critical in active vision settings where camera positions are dynamic rather than static. Recent works address this by training endto-end behavior cloning policies that jointly control both the active camera and the manipulation arms using human demonstrations [8, 9, 7]. While effective, these approaches increase data requirements and model complexity. In contrast, our approach trains policies over a compact distribution of viewpoints and explicitly optimizes the test-time viewpoint relative to demonstrations. Another class of methods builds explicit 3D scene representations, such as point clouds or voxels [23, 24, 25], or renders canonical 2D views from these structures [26]. These approaches improve robustness to viewpoint variation but require precise extrinsic calibration and incur high computational costs. Data augmentation offers a complementary direction: [27] generate multi-view datasets from single-view demonstrations using a pre-trained zeroshot novel-view synthesis model [28], while [29, 30] edit reconstructed 3D scenes built from dense scans to create synthetic training views. Beyond end-to-end training, some works transfer a single demonstration trajectory at test time by estimating relative object pose changes between demonstration and execution views, either via pose estimation [2] or visual servoing [6]. However, such methods remain highly sensitive to occlusions. In this paper, we extend behavior cloning and trajectory transfer within our observerâactor framework, demonstrating consistent performance gains in both non-occluded and occluded scenarios.

## III. METHOD

Figure 2 illustrates the proposed system framework. To collect a demonstration, the operator selects a demonstration optimal viewpoint (Section III-A) for the task object, positions the observer accordingly, and records a trajectory using the actor arm, creating a dataset for downstream imitation learning. At test time, given a novel object configuration with potential occlusions, each arm captures three predefined scene views, yielding six views in total. The system dynamically assigns observer and actor roles based on these views and constructs a 3DGS representation from the observerâs three views (Section III-B). Through view optimization (Section III-C), it identifies the test-time optimal viewpoint within this representation. The observer arm then moves to this viewpoint, allowing the actor arm to execute the manipulation task conditioned on it (Section III-D).

## A. Definition of Optimal View

Demonstration Optimal View. We define a demonstration optimal view as a camera viewpoint relative to the target object that maximizes the visibility of task-relevant features while minimizing occlusion. The actorâs interaction with the object must remain clearly observable from this view, without obstruction from the actor arm during manipulation. In practice, demonstration optimal views avoid selfocclusion, robot-induced occlusion while preserving fine object details, as illustrated in Figure 1. These views are determined by the operator using their own subjective estimation of where an optimal view is. We denote the demonstration optimal view as $v _ { \mathrm { d e m o } } ^ { \ast } \in \mathrm { S E } ( 3 )$ , where v is the camera pose expressed in the object frame. A dataset then contains a set of such demonstration optimal views: ${ \mathcal { V } } _ { \mathrm { d e m o } } ^ { * } = \{ { v } _ { \mathrm { d e m o } , 1 } ^ { * } , { v } _ { \mathrm { d e m o } , 2 } ^ { * } , \ldots , { v } _ { \mathrm { d e m o } , N } ^ { * } \}$

<!-- image-->  
1. Select an optimal view

<!-- image-->  
2. Collect demonstration

<!-- image-->

<!-- image-->  
1. Sparse-view exploration

<!-- image-->  
2. 3DGS and optimization

<!-- image-->  
3. Move to best view and act  
Fig. 2: Framework Overview. (1) Train: The operator selects a demonstration optimal view, moves the observer arm to this view, and records a demonstration. This process is repeated as required by the imitation learning method. (2) Test: The robots explore six views of the scene to construct a 3DGS representation. View optimization within this representation identifies the test-time optimal view. The observer arm then moves to this view, after which the actor arm executes the task.

Test-time Optimal View. At test time, the system selects a viewpoint from the kinematically feasible set V that is closest to the demonstration optimal view while minimizing occlusion. Formally, this is expressed as:

$$
v _ { \mathrm { t e s t } } ^ { * } = \arg \operatorname* { m i n } _ { v \in \mathcal { V } } \left( d _ { \mathrm { p o s e } } ( v , v _ { \mathrm { d e m o } } ^ { * } ) + \lambda \mathcal { O } ( \mathrm { o b j } , v ) \right)\tag{1}
$$

Here, Î» balances the trade-off between proximity to the demonstration optimal view and occlusion, $\mathcal { O } ( \cdot )$ quantifies occlusion, $d _ { \mathrm { p o s e } } ( \cdot , \cdot )$ measures the SE(3) distance between camera poses, and obj denotes the target object. Since v is expressed in the object frame, the first term encourages test-time viewpoints that are visually consistent with the training views, ensuring in-distribution observations for robust inference. Real-world examples of such optimal viewpoints are shown in Figure 3.

## B. Sparse-view Gaussian Splatting

Directly computing $v _ { \mathrm { t e s t } } ^ { \ast }$ using Equation 1 requires depth information and an occlusion function. We show that an image-level surrogate loss provides an effective approximation. This approach entails simulating images from unseen viewpoints, which in turn requires test-time 3D reconstruction. Rather than performing time-consuming full scans, we capture sparse exploratory views of the scene, assign observer and actor roles, and construct a 3DGS representation.

Exploratory Views. We begin by capturing six scene viewpoints, evenly spaced at $6 0 ^ { \circ }$ intervals to cover the full $3 6 0 ^ { \circ }$ workspace. These exploratory poses are predefined and fixed. In each iteration, both arms simultaneously move to two of the poses, producing six images along with their corresponding camera poses expressed in the robot frame. Role Assignment. Roles are assigned to each arm based on how closely their captured views match $v _ { \mathrm { d e m o } } ^ { * }$ . Instead of performing explicit relative pose estimation, we use the robust dense feature matcher RoMa [31], using the number of confident matches on the segmented object as a proxy. The arm with more aggregated matches is designated as the observer, indicating it is closer to the demonstration view, while the other arm becomes the actor.

Three-View GS. We reconstruct the scene using the three images captured by the observer, via InstantSplat [32], a sparse-view method that leverages geometric priors. It employs Mast3R [33], a large-scale 3D geometric model, to estimate camera poses from RGB images and perform joint optimization. These optimized poses are more accurate than those from our low-cost robot arms, resulting in higherquality reconstructions. We use only three images to train the 3DGS model, as the additional views provide little extra information for rendering $v _ { \mathrm { t e s t } } ^ { \ast }$ , and training is considerably faster. We demonstrate that three views offer the best trade-off between performance and computation time (Section IV-C). Frame Alignment. The 3DGS reconstruction is generated in an arbitrary coordinate system and must be aligned with the robotsâ frame. To achieve this, we use the Umeyama algorithm. By combining camera poses from the robotâs encoders and hand-eye calibration with the estimated poses from InstantSplat, we solve for a similarity transform via singular value decomposition (SVD) to align the two sets of poses. This process also recovers the scale of the reconstruction.

## C. View Optimization

To compute the test-time optimal view $v _ { \mathrm { t e s t } } ^ { \ast }$ , we first generate candidate viewpoints by global sampling within the 3DGS representation. We then select the best candidate and refine it using local gradient-based optimization. Finally, the resulting view is aligned to the real world is then aligned to the real world for the observer arm to reach.

Candidate View Sampling. To initialize the optimization, we sample a hemisphere of candidate viewpoints around the task objectâs center, following [16]. The object center is estimated from its 3D bounding box, obtained by lifting 2D masks from GroundedSAM [34] into 3D using aligned depth maps. After filtering out kinematically infeasible poses, we obtain a candidate set $\mathcal { V } _ { \mathrm { c a n d i d a t e } } \subseteq \mathcal { V } _ { }$ . Owing to the efficiency of 3DGS, all candidate views can be rendered at â¼250 FPS.

Optimal View Initialization. After rendering a set of virtual RGB images, we select the candidate view with the highest score as the initialization. The scoring function, identical to that used for role assignment, is based on the number of confidence-weighted feature matches between images from $\mathcal { V } _ { \mathrm { c a n d i d a t e } }$ and $v _ { \mathrm { d e m o ^ { \prime } } } ^ { * }$ . This criterion implicitly accounts for occlusion, as occluded regions naturally produce fewer and less reliable matches. In practice, we found that this matchingbased strategy outperforms an alternative based on DINOV2 [35] cosine distance for measuring view similarity. Even when combined with object segmentation, DINOV2 struggles to discriminate between viewpointsâan observation consistent with [36]. To reduce computational cost, we employ Tiny RoMa [31], a lightweight variant of RoMa with faster runtime.

Differentiable Rendering. After obtaining an initialization, we refine the viewpoint v using differentiable rendering while explicitly accounting for gripper-induced occlusions. This is particularly important in our setting, where the observer armâs gripper consistently remains in the field of view and may block the object, as shown in Figure 3. To handle this, we use SAM2 [37] for segmentation of the rendered image, guided by the demonstration imageâs mask as a prompt. This is effective because the initialized and demonstration viewpoints are already closely aligned. To mitigate errors in estimating the object center, we first re-center the objectâs mask in the image plane. We then refine v by optimizing a loss that aligns the image rendered from the 3DGS model (denoted G) with the segmented demonstration image $I _ { \mathrm { d e m o } } ^ { * }$ , while penalizing occlusions. Let $I ( v ; { \mathcal { G } } )$ denotes the rendered image. The loss $\mathcal { L }$ as a particular view v is defined as:

$$
\begin{array} { r l } & { \mathcal { L } ( v ) = - \lambda _ { 1 } \mathcal { L } _ { \mathrm { s i m } } \big ( \phi ( I ( v ; \mathcal { G } ) \odot \mathcal { M } _ { \mathrm { o b j } } ( v ) \big ) , \phi ( I _ { \mathrm { d e m o } } ^ { * } ) \big ) } \\ & { \quad \quad \quad + \lambda _ { 2 } \mathcal { L } _ { \mathrm { I O U } } ( \mathcal { M } _ { \mathrm { o b j } } ( v ) , \mathcal { M } _ { \mathrm { g r i p } } ) , } \end{array}\tag{2}
$$

where $\phi ( \cdot )$ denotes the DINOV2 feature extractor, $\mathcal { M } _ { \mathrm { o b j } } ( v )$ is the soft object mask at viewpoint v obtained from SAM2, and $\mathcal { M } _ { \mathrm { g r i p } }$ is the fixed soft gripper mask. The first term enforces feature alignment, while the second penalizes overlap between the object and gripper masks. Unlike photometric losses employed in prior work [38], our loss explicitly accounts for occlusions. We note although DINOV2 is unreliable for global initialization, it is effective for local refinement.

Moving to the Optimal View. Once the optimal camera pose $v _ { \mathrm { t e s t } } ^ { \ast }$ is computed, we transform it back into the realworld coordinate frame and use a motion planner to move the observer arm to the target pose. Figure 3 illustrates the demonstration view, the optimal view obtained from the 3DGS representation, and the final aligned view executed in the real world. Minor discrepancies in viewpoint arise from robot kinematic errors, handâeye calibration, and frame alignment, but these do not noticeably affect downstream imitation learning performance.

## D. View-Conditioned Imitation Learning

In our framework, we extend two categories of imitation methods for task execution once the camera has been moved to the optimal view: trajectory transfer and behavior cloning. AV Trajectory Transfer. Trajectory transfer (TT) methods estimate the relative pose change of the object between the demonstration and the test instance, and then transfer the demonstration trajectory in a one-shot manner within the $\operatorname { S E } ( 3 )$ manifold [2]. Consequently, only a single demonstration optimal view is required. In our setting, demonstrations are recorded in the actor armâs coordinate frame, while observations are captured from the observer armâs camera. To resolve this mismatch, we extend trajectory transfer by applying the following sequence of frame transformations:

$$
{ ^ { \mathrm { O } } } { \bf T } _ { \mathrm { E } _ { \mathrm { O } } } ( \mathrm { t e s t } ) = { ^ { \mathrm { O } } } { \bf T } _ { \mathrm { E } _ { \mathrm { O } } } { ^ { \mathrm { E } _ { \mathrm { O } } } } { \bf T } _ { \mathrm { C } } \Delta ^ { \mathrm { C } } { \bf T } _ { \mathrm { o b j } } { ^ { \mathrm { C } } } { \bf T } _ { \mathrm { E } _ { \mathrm { O } } }
$$

$$
\Delta ^ { \mathrm { O } } \mathbf { T } _ { \mathrm { o b j } } = ^ { \mathrm { O } } \mathbf { T } _ { \mathrm { E } _ { \mathrm { O } } } ( \mathrm { t e s t } ) ^ { \mathrm { E } _ { \mathrm { O } } } \mathbf { T } _ { \mathrm { O } } ( \mathrm { d e m o } )
$$

$$
\Delta ^ { \mathrm { A } } \mathbf { T } _ { \mathrm { o b j } } = { } ^ { \mathrm { O } } \mathbf { T } _ { \mathrm { A } } \Delta ^ { \mathrm { O } } \mathbf { T } _ { \mathrm { o b j } } ^ { \mathrm { ~ A } } \mathbf { T } _ { \mathrm { O } }\tag{3}
$$

(4)

$$
\mathbf { \Sigma } ^ { \mathrm { A } } \mathbf { T } _ { \mathrm { E A } } \left( \mathrm { t e s t } \right) = \Delta ^ { \mathrm { A } } \mathbf { T } _ { \mathrm { o b j } } \mathbf { \Sigma } ^ { \mathrm { A } } \mathbf { T } _ { \mathrm { E A } } \left( \mathrm { d e m o } \right)\tag{5}
$$

(6)

Here, $\mathrm { { } ^ { O } T _ { E o } ( t e s t ) }$ is the observer end-effector pose at $v _ { \mathrm { t e s t } } ^ { \ast } ,$ and ${ } ^ { \mathrm { O } } \mathbf { T } _ { \mathrm { E } _ { \mathrm { O } } }$ is its pose at $v _ { \mathrm { t r a i n } } ^ { * } ,$ respectively, both expressed in the observer base frame O. The handâeye calibration from the observer end-effector to the camera frame is $\mathbf { E } _ { \mathrm { { O } } }  \mathbf { T } _ { \mathrm { { C } } }$ with inverse ${ { \mathrm { C } } _ { \mathbf { T } _ { \mathrm { E _ { \mathrm { O } } } } } }$ . The objectâs relative pose change from demonstration to test is denoted by $\Delta ^ { \mathrm { C } } \mathbf { T } _ { \mathrm { o b j } }$ in the camera frame, $\Delta ^ { \mathrm { o } } \mathbf { T } _ { \mathrm { o b j } }$ in the observer base frame, and $\Delta ^ { \mathrm { A } } \mathbf { T } _ { \mathrm { o b j } }$ in the actor base frame A. The fixed transform between the observer and actor base frames is ${ } ^ { \mathrm { O } } \mathbf { T } _ { \mathrm { A } }$ , with inverse ${ } ^ { \mathrm { A } } \mathbf { T } _ { \mathrm { O } }$ Finally, ${ \bf { \ddot { A } T _ { E _ { A } } } }$ (demo) and ${ \bf { \ddot { A } T _ { E _ { A } } } }$ (test) denote the actorâs end-effector poses for task execution during demonstration and test, both expressed in the actor base frame A.

We estimate $\Delta ^ { \mathrm { C } } \mathbf { T } _ { \mathrm { o b j } }$ using RoMa by lifting matched feature points to 3D via aligned depth maps. The relative SE(3) transformation is then computed using Procrustes alignment within a RANSAC loop. The transferred trajectory is executed in open-loop, and its accuracy further depends on the handâeye calibration $\mathbf { E } _ { \mathrm { { O } } }  \mathbf { T } _ { \mathrm { { C } } }$ and the fixed transform between the observer and actor base frames ${ } ^ { \mathrm { O } } \mathbf { T } _ { \mathrm { A } }$

AV Behavior Cloning. Behavior cloning (BC) trains a closedloop policy Ï to map observations to actions. Unlike TT, which exploits pose estimation models, BC requires several demonstration optimal views to generalize across diverse camera perspectives covering $v _ { \mathrm { t e s t } } ^ { \ast }$ . Since optimizing over all viewpoints in $\mathcal { V } _ { \mathrm { t r a i n } } ^ { * }$ is computationally expensive, we use Mast3R to estimate camera poses in the object frame and select the viewpoint closest to the center of all estimated poses as $v _ { \mathrm { d e m o } } ^ { * } .$ , which we found to work well empirically. Moreover, unlike prior work [9, 7], which represents the end-effector pose in a static world frame and provides the camera pose explicitly as policy input, we express the actor armâs end-effector pose directly in the camera frame:

$$
\mathrm { { ^ C T _ { E _ { A } } } = { ^ C T _ { E _ { O } } } { ^ { E _ { O } } T _ { O } } { ^ { O } T _ { A } } { ^ { A } T _ { E _ { A } } } . }\tag{7}
$$

This representation simplifies the state space and, as we show in Section IV-B, improves both data efficiency and policy performance. At each time step t, the policy receives an RGB observation $I _ { t }$ from the test-time optimal view $v _ { \mathrm { t e s t } } ^ { \ast }$ , along with the proprioceptive state $S _ { t }$ . The proprioceptive state includes the actor end-effector pose expressed in the camera frame and the gripper state. The policy outputs an action sequence $a _ { t : t + n _ { p } - 1 } = \pi { \bigl ( } I _ { t } , S _ { t } { \bigr ) }$ of length $n _ { p }$ , which is then transformed back to the actorâs base frame for execution.

Ambidextrous Inference. By representing actions in the camera frame for both of our imitation learning methodsâAV TT and AV BCâwe enable ambidextrous inference: when the roles of observer and actor differ from those in the demonstration, the transferred trajectory and trained policy can still be executed without any additional data or training.

## IV. EVALUATION

We instantiated our method on a real-world dual-arm ALOHA setup [5], mounted on a table and equipped with two calibrated RealSense D405 cameras. To evaluate our approach, we selected five diverse manipulation tasks involving selfoccluding objects. In Pick Mug, the robot grasps a mug by its handle. In Hammer Nail, it uses a pre-grasped hammer to drive a nail. Open Drawer requires opening the second drawer of a multi-layer unit. In Retrieve Pack, the robot retrieves a package from a deep box. Finally, Insert Coin involves placing a coin into a storage container. All tasks require observing specific object parts or features for successful manipulation, with the key regions highlighted in red in Figure 3.

## A. Comparison with Static-Camera

Baselines and Implementation. We evaluate our method for both TT and BC approaches, comparing against static-camera setups. For the TT baseline, we select the best image from the initial 3-view exploration and perform pose estimation relative to the demonstration, as described in Section III-D. In contrast, our method leverages the test-time optimal view for pose estimation. For the BC baseline, we use a static camera chosen from one of the poses in the 3-view exploration.

To ensure fairness, we collect separate datasets for BC with and without active vision, rather than reusing the same dataset as was done in [7]. Reusing would introduce visual inconsistencies, as the static camera would capture the observer arm moving across different poses, providing irrelevant information that could confuse the learned policy. In total, we collected 70 demonstrations for each of the five tasks, with and without active vision, resulting in $7 0 \times 5 \times 2 = 7 0 0$ demonstrations. All demonstrations were recorded using an iPhone-based teleoperation system built with ARKit.

We implement BC with ACT [5], using DINOv2 as the vision backbone and absolute Cartesian action representation. For BC with AV, we remove any frames in the dataset where the actor armâs gripper is invisible in the camera view, as these introduce ambiguities. This also ensures that the trained policy outputs a pose where the actorâs gripper is likely to be visible in the camera view at the first timestep.

Experiments. For all tasks except Retrieve Pack, which inherently involves occlusion due to its setup, we evaluate both unoccluded and occluded scenarios. We perform 10 rollouts on novel scene configurations with variations in object poses and the presence of distractors. For each variation, we first run the baseline and then our method under identical conditions.

For the static-camera BC baseline, both training and evaluation are restricted to object poses where task-relevant parts remain visible, as self-occlusion leads to prohibitively low success rates. Consequently, pose variations for the static-camera BC baseline are limited to within $4 5 ^ { \circ }$ of the demonstration poses during testing. In contrast, our method naturally handles self-occlusion arising from larger rotations. Results. Table I summarizes the experiment results. Using the test-time optimal view for inference, both TT and BC consistently outperform the static-camera setup across all tasks. Specifically, TT achieves performance improvements of 146% in the unoccluded setting and 233% in the occluded setting, while BC improves by 75% and 143%, respectively.

For TT, this improvement arises because the test-time optimal image has greater feature overlap with the demonstration optimal image and experiences less occlusion, leading to more accurate pose estimation. For BC, the test-time observations are more in-distribution relative to the training data and encounter fewer occlusions, resulting in more reliable policy execution. The performance drop under occlusion may stem from selected viewpoints being slightly out-of-distribution in order to compensate for occluded regions.

Data Efficiency of BC with AV. Figure 4 shows the success rates versus the number of demonstrations, using 30, 50, and 70 demos for the Pick Mug, Retrieve Pack, and Open Drawer tasks. We observe that, given the same number of demonstrations, BC with AV consistently outperforms the static-camera setup, highlighting the advantages of inference from test-time optimal views. Notably, for the Retrieve Pack task, BC with a static camera fails completely. This is because the task suffers from severe occlusion, particularly from the actor armâs gripper when reaching inside the box. Some training frames contain the package fully occluded by the gripper, which confuses the policy and prevents it from learning reliable grasp decisions.

Failure Modes. Our method can fail if the AV pipeline selects a suboptimal view. In practice, the pipeline typically generates views that are considered reasonable by humans, validating the effectiveness of sparse-view 3DGS for active vision. However, we occasionally observe occlusion of critical regions, such as drawer handles during rollouts, because the current formulation does not explicitly enforce full visibility of key object parts. Downstream imitation learning methods can also fail despite receiving a good view. TT with AV may fail due to inaccurate feature matching or calibration errorsâeither from handâeye calibration or from the relative calibration between the observer and actor arms. For BC with AV, failures mostly occur when the actor armâs initial end-effector state in the camera frame deviates from those seen in the demonstrations, causing compounding errors. This is more likely when the test-time optimal views differ significantly from the demonstration view. Conversely, tasks typically succeed when the actorâs gripper is sufficiently close to the target object, as fine-grained actorâobject interactions are accurately captured in the optimal demonstration-view dataset. Finally, because the learned policy relies on a single RGB observation, it remains sensitive to depth ambiguities.

TABLE I: Success Rates of Different Methods on the Five Tasks (without and with occlusions).
<table><tr><td rowspan="2">Method</td><td colspan="2">Pick Mug</td><td colspan="2">Hammer Nail</td><td colspan="2">Open Drawer</td><td colspan="2">Retrieve Pack</td><td colspan="2">Insert Coin</td><td colspan="2">Sum</td></tr><tr><td>no occ</td><td>occ</td><td>no occ</td><td>occ</td><td>no occ</td><td>occ</td><td>no occ</td><td>occ</td><td>no occ</td><td>occ</td><td>no occ</td><td>occ</td></tr><tr><td>TT (3-views)</td><td>5/10</td><td>3/10</td><td>4/10</td><td>4/10</td><td>2/10</td><td>1/10</td><td>N/A</td><td>1/10</td><td>0/10</td><td>0/10</td><td>11/40</td><td>9/50</td></tr><tr><td>TT (ours with AV)</td><td>8/10</td><td>8/10</td><td>10/10</td><td>7/10</td><td>6/10</td><td>4/10</td><td>N/A</td><td>9/10</td><td>3/10</td><td>2/10</td><td>27/40</td><td>30/50</td></tr><tr><td>BC (static camera) BC (ours with AV)</td><td>4/10 6/10</td><td>4/10 5/10</td><td>3/10 4/10</td><td>2/10 5/10</td><td>1/10 3/10</td><td>1/10 3/10</td><td>N/A N/A</td><td>0/10 3/10</td><td>0/10 1/10</td><td>0/10 1/10</td><td>8/40 14/40</td><td>7/50 17/50</td></tr></table>

<!-- image-->

Fig. 3: Images of Optimal Views. Top row: demonstration optimal views. Middle row: test-time optimal views in 3DGS with gripper mask overlay. Bottom row: real world test-time optimal views. Red boxes indicate the task-relevant object parts. Test-time optimal views are derived by reconstructing the demonstrationâs optimal viewpoints subject to minimal occlusion.  
<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 4: Data Efficiency of Behavior Cloning with Active Vision. With the same number of demonstrations, our method outperforms the static camera setup across three tasks, evaluated using 30, 50, and 70 demonstrations.

Experiment videos are available on our project webpage at https://obact.github.io, including demonstrations of dynamic role assignment, timed task executions, object generalization, and object tracking capabilities of our system.

## B. Representing Actions in the Camera Frame

We evaluate the effect of representing actions in the camera frame for BC with AV. We compare against prior approaches [7, 8], where actions are represented in the fixed actor robot frame and the camera pose is provided as an input. Results in Table II show that representing actions directly in the camera frame improves both generalization and task success. We hypothesize that providing the camera pose as input is less effective because object poses span a wide distribution while camera views remain relatively similar, making it difficult for the policy to infer the robotâs pose in the static frame. On the other hand, representing actions directly in the camera frame provides a consistent reference.

<!-- image-->  
Fig. 5: Effect of Number of Exploration Views.

TABLE II: Impact of action representation on success rates.
<table><tr><td>Method</td><td>Mug</td><td>Hammer</td></tr><tr><td>Camera pose as input, action in robot frame</td><td>1/10</td><td>0/10</td></tr><tr><td>Action in camera frame (ours)</td><td>6/10</td><td>4/10</td></tr></table>

However, representing actions in the camera frame also introduces a potential issue. In practice, we found that the training viewpoints must be sufficiently varied; otherwise, the model can shortcut learning by relying primarily on the camera pose. When trajectories relative to the camera are too similar, the policy may ignore the image inputs entirely.

## C. Impact of Number of Exploration Views

We evaluate the impact of exploring three views per arm at test time to acquire scene information. Specifically, we compare 2â5 exploration viewpoints per arm to study how the extent of visual coverage affects downstream manipulation success. We use BC with AV and evaluate on the Pick Mug task under occlusion, with results shown in Figure 5. Our findings indicate that three views strike a good balance between performance and execution time, providing sufficient scene coverage for reliable role assignment and optimal viewpoint selection. Since frame alignment requires at least three views, using only two necessitates relying on camera poses inferred from the robotâs encoders and handâeye calibration, bypassing full alignment. This leads to higher variance in performance because the resulting 3DGS reconstructions are less accurate, widening the sim-to-real gap. Consequently, test-time optimal views may also be less reliable. This limitation could be mitigated with more precise robots and better calibration.

TABLE III: Time breakdown for each component of our AV pipeline, measured with a RTX 4080Ti GPU.
<table><tr><td>Component</td><td>Time (s)</td></tr><tr><td>Six-view Exploration</td><td>18</td></tr><tr><td>InstantSplat Geometric Initialization</td><td>20</td></tr><tr><td>InstantSplat 3DGS Training</td><td>23</td></tr><tr><td>Optimal View Initialization</td><td>3</td></tr><tr><td>Differential Rendering</td><td>12</td></tr><tr><td>Total</td><td>76</td></tr></table>

Table III reports the time breakdown of our AV pipeline. The majority of time is spent on InstantSplat, while the exploration step could be further reduced by increasing the armsâ speed. We believe that as sparse-view 3DGS methods continue to improve, our system will become more efficient.

## V. CONCLUSIONS

We have present ObAct, a novel observerâactor framework for imitation learning in active vision, in which an observer arm computes and moves to optimal visual observations to guide the actor armâs actions. Our method captures fine-grained gripperâobject interactions, supporting robust manipulation across diverse scenarios using both trajectory transfer and behavior cloning. Experimental results demonstrate significant gains in success rates compared to a static camera setup across both unoccluded and occluded scenarios.

Despite the effectiveness of our method, several limitations remain. The active-vision pipeline is relatively slow, the approach is tailored to short-horizon tasks, and it lacks reactivity to environmental changes. Moreover, the current setup cannot handle tasks that require two arms acting simultaneously. We identify several promising directions for future work. One is to explore dynamic viewpoints that continuously track gripperâobject interactions during execution, enabling richer visual feedback and closed-loop occlusion avoidance. Another is to extend our approach to long-horizon tasks and to deformable-object manipulation, both of which introduce additional challenges for active vision and imitation learning. Finally, for dual-arm manipulation, we envision expanding the system to a three-arm configuration, in which one arm dynamically acts as the observer while the remaining two serve as the manipulators. We believe these extensions will enable more resilient active-vision robotic systems capable of handling complex manipulation tasks in diverse, unstructured real-world environments.

## REFERENCES

[1] Cheng Chi et al. âDiffusion policy: Visuomotor policy learning via action diffusionâ. In: The International Journal of Robotics Research (2023), p. 02783649241273668. 1, 2

[2] Kamil Dreczkowski et al. âLearning a thousand tasks in a dayâ. In: Science Robotics 10.108 (2025), eadv7594. 1, 2, 4

[3] Cheng Chi et al. âUniversal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robotsâ. In: Proceedings of Robotics: Science and Systems (RSS). 2024. 1

[4] Norman Di Palo and Edward Johns. âDinobot: Robot manipulation via retrieval and alignment with vision foundation modelsâ. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE. 2024, pp. 2798â2805. 1

[5] Tony Zhao et al. âLearning Fine-Grained Bimanual Manipulation with Low-Cost Hardwareâ. In: Robotics: Science and Systems XIX (2023). 1, 2, 5

[6] Yilong Wang and Edward Johns. âOne-Shot Dual-Arm Imitation Learningâ. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). 2025, pp. 5660â5668. 1, 2

[7] Ian Chuang et al. âActive vision might be all you need: Exploring active vision in bimanual robotic manipulationâ. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE. 2025. 1, 2, 4â6

[8] Haoyu Xiong et al. âVision in Action: Learning Active Perception from Human Demonstrationsâ. In: arXiv preprint arXiv:2506.15666 (2025). 1, 2, 6

[9] Xuxin Cheng et al. âOpen-TeleVision: Teleoperation with Immersive Active Visual Feedbackâ. In: Proceedings of The 8th Conference on Robot Learning. Ed. by Pulkit Agrawal, Oliver Kroemer, and Wolfram Burgard. Vol. 270. Proceedings of Machine Learning Research. PMLR, June 2025, pp. 2729â2749. 1, 2, 4

[10] Bernhard Kerbl et al. â3D Gaussian Splatting for Real-Time Radiance Field Renderingâ. In: ACM Transactions on Graphics 42.4 (2023), pp. 1â14. 1

[11] John Aloimonos, Isaac Weiss, and Amit Bandyopadhyay. âActive visionâ. In: International journal of computer vision 1 (1988), pp. 333â356. 2

[12] Kanzhi Wu, Ravindra Ranasinghe, and Gamini Dissanayake. âActive recognition and pose estimation of household objects in clutterâ. In: 2015 IEEE International Conference on Robotics and Automation (ICRA). IEEE. 2015, pp. 4230â4237. 2

[13] Boshi An et al. âRgbmanip: Monocular image-based robotic manipulation through active object pose estimationâ. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE. 2024, pp. 7748â7755. 2

[14] Stefan Isler et al. âAn information gain formulation for active volumetric 3D reconstructionâ. In: 2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE. 2016, pp. 3477â3484. 2

[15] Soomin Lee et al. âUncertainty guided policy for active robotic 3d reconstruction using neural radiance fieldsâ. In: IEEE Robotics and Automation Letters 7.4 (2022), pp. 12070â12077. 2

[16] Michel Breyer et al. âClosed-loop next-best-view planning for target-driven graspingâ. In: 2022 IEEE/RSJ

International Conference on Intelligent Robots and Systems (IROS). IEEE. 2022, pp. 1411â1416. 2, 4

[17] Xuechao Zhang et al. âAffordance-Driven Next-Best-View Planning for Robotic Graspingâ. In: Proceedings of The 7th Conference on Robot Learning. Ed. by Jie Tan, Marc Toussaint, and Kourosh Darvish. Vol. 229. Proceedings of Machine Learning Research. PMLR, June 2023, pp. 2849â2862. 2

[18] Haoxiang Ma et al. âActive Perception for Grasp Detection via Neural Graspness Fieldâ. In: Advances in Neural Information Processing Systems. Ed. by A. Globerson et al. Vol. 37. Curran Associates, Inc., 2024, pp. 38122â38141. 2

[19] Yitian Shi et al. âVISO-Grasp: Vision-Language Informed Spatial Object-centric 6-DoF Active View Planning and Grasping in Clutter and Invisibilityâ. In: arXiv preprint arXiv:2503.12609 (2025). 2

[20] Guokang Wang et al. âObserve Then Act: Asynchronous Active Vision-Action Model for Robotic Manipulationâ. In: IEEE Robotics and Automation Letters (2025). 2

[21] Jingyun Yang et al. âMobi-Ï: Mobilizing Your Robot Learning Policyâ. In: arXiv preprint arXiv:2505.23692 (2025). 2

[22] Justin Kerr et al. âEye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loopâ. In: 9th Annual Conference on Robot Learning. 2025. 2

[23] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. âPerceiver-actor: A multi-task transformer for robotic manipulationâ. In: Conference on Robot Learning. PMLR. 2023, pp. 785â799. 2

[24] Yanjie Ze et al. â3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representationsâ. In: Proceedings of Robotics: Science and Systems (RSS). 2024. 2

[25] Theophile Gervet et al. âAct3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulationâ. In: 7th Annual Conference on Robot Learning. 2

[26] Ankit Goyal et al. âRvt: Robotic view transformer for 3d object manipulationâ. In: Conference on Robot Learning. PMLR. 2023, pp. 694â710. 2

[27] Stephen Tian et al. âView-Invariant Policy Learning via Zero-Shot Novel View Synthesisâ. In: arXiv (2024). 2

[28] Kyle Sargent et al. âZeronvs: Zero-shot 360-degree view synthesis from a single imageâ. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024, pp. 9420â9429. 2

[29] Sizhe Yang et al. âNovel demonstration generation with gaussian splatting enables robust one-shot manipulationâ. In: arXiv preprint arXiv:2504.13175 (2025). 2

[30] Yifei Ren and Edward Johns. âLearning in ImaginationLand: Omnidirectional Policies through 3D Generative Models (OP-Gen)â. In: arXiv preprint arXiv:2509.06191 (2025). 2

[31] Johan Edstedt et al. âRoma: Robust dense feature matchingâ. In: Proceedings of the IEEE/CVF Con-

ference on Computer Vision and Pattern Recognition. 2024, pp. 19790â19800. 3, 4

[32] Zhiwen Fan et al. âInstantSplat: Sparse-view Gaussian Splatting in Secondsâ. In: arXiv preprint arXiv:2403.20309 (2024). 3

[33] Vincent Leroy, Yohann Cabon, and JerÂ´ ome Revaud. Ë âGrounding image matching in 3d with mast3râ. In: European Conference on Computer Vision. Springer. 2024, pp. 71â91. 3

[34] Tianhe Ren et al. âGrounded sam: Assembling openworld models for diverse visual tasksâ. In: arXiv preprint arXiv:2401.14159 (2024). 4

[35] Maxime Oquab et al. âDINOv2: Learning Robust Visual Features without Supervisionâ. In: Transactions on Machine Learning Research Journal (2024). 4

[36] Kai Chen et al. âVision Foundation Model Enables Generalizable Object Pose Estimationâ. In: Advances in Neural Information Processing Systems. Ed. by A. Globerson et al. Vol. 37. Curran Associates, Inc., 2024, pp. 19975â20002. 4

[37] Nikhila Ravi et al. âSAM 2: Segment Anything in Images and Videosâ. In: The Thirteenth International Conference on Learning Representations. 4

[38] Lin Yen-Chen et al. âinerf: Inverting neural radiance fields for pose estimationâ. In: 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE. 2021, pp. 1323â1330. 4