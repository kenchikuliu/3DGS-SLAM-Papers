# SurgCalib: Gaussian Splatting-Based Hand-Eye Calibration for Robot-Assisted Minimally Invasive Surgery

Zijian Wu1,â, Shuojue Yang2,â, Yu Chung Lee1, Eitan Prisman3, Yueming Jin2 and Septimiu E. Salcudean1

Abstractâ We present a Gaussian Splatting-based framework for hand-eye calibration of the da Vinci surgical robot. In a vision-guided robotic system, accurate estimation of the rigid transformation between the robot base and the camera frame is essential for reliable closed-loop control. For cable-driven surgical robots, this task faces unique challenges. The encoders of surgical instruments often produce inaccurate proprioceptive measurements due to cable stretch and backlash. Conventional hand-eye calibration approaches typically rely on known fiducial patterns and solve the AX = XB formulation. While effective, introducing additional markers into the operating room (OR) environment can violate sterility protocols and disrupt surgical workflows. In this study, we propose SurgCalib, an automatic, markerless framework that has the potential to be used in the OR. SurgCalib first initializes the pose of the surgical instrument using raw kinematic measurements and subsequently refines this pose through a two-phase optimization procedure under the RCM constraint within a Gaussian Splattingâbased differentiable rendering pipeline. We evaluate the proposed method on the public dVRK benchmark, SurgPose. The results demonstrate average 2D tool-tip reprojection errors of 12.24 px (2.06 mm) and 11.33 px (1.9 mm), and 3D tool-tip Euclidean distance errors of 5.98 mm and 4.75 mm, for the left and right instruments, respectively.

## I. INTRODUCTION

In general robotics, hand-eye calibration is a fundamental procedure that establishes the 3D transformation between the robot base and the camera coordinate systems. This transformation enables measurements acquired in the camera frame to be mapped into the robot coordinate system for control and interaction. In Robot-Assisted Minimally Invasive Surgery (RAMIS), e.g., da Vinci surgical system, articulated surgical instruments teleoperated as patient-side manipulators (PSM) and the endoscope, which provides a stereoscopic view of the surgical scene, inherently form a hand-eye system.

Accurate hand-eye calibration can compensate for systematic biases in raw API-reported kinematic measurements and ensure robust surgical instrument end-effector pose tracking and tool tip localization, which serve as the foundation for downstream applications. For example, in a surgical Augmented Reality (AR) guidance system, accurate instrument tip positioning facilitates intraoperative registration between real-time ultrasound and the da Vinci robot [1]. This registration is an inevitable intermediate step for ultimately aligning multimodal preoperative medical imaging data (CT, MRI, and 3D Ultrasound) to the endoscopic video [2]. For autonomous surgical sub-task manipulation, such as suture needle grasping [3], hand-eye calibration enables the detected needle pose in the camera frame to be transformed into the robot frame, thereby facilitating subsequent motion planning and control. More fundamentally, hand-eye calibration aligns the robotâs two essential sensing modalities, perception (vision) and proprioception, into a unified coordinate framework.

Despite its importance, hand-eye calibration on the da Vinci robotic system remains challenging due to inherent hardware limitations. The accuracy of the Cartesian endeffector pose cannot be arrived at solely from joint angle information, because of cable backlash and the structural flexibility of the tendon-driven instruments. This issue is further exacerbated by the set-up joint (SUJ) controllers; since the SUJs lack active actuators and rely on passive encoders, if these are not read to update the kinematics in real time, the robot position is uncertain. In addition, there is a long and heavily loaded kinematic chain from the setup arms to the camera of the system, leading to unreliable instrument tip positioning [4]. Prior efforts have attempted to mitigate these inaccuracies using data-driven or marker-based approaches. For instance, Hwang et al. [5] propose a method using the RGB-D fiducial markers and a recurrent neural network. Yet, the reliance on a specialized depth camera and the time-consuming model training limit its practicality. Similarly, a hierarchical scheme introduced by Lu et al. [6] requires manual labeling of instrument features and specific controlled motions for local refinement. Such dependencies not only hinder full system automation but also significantly increase preoperative setup time, impairing their practicability in clinical applications.

Recent advances in robot pose estimation demonstrate significant potential to replace marker-based pose estimation and tracking approaches [7], [8]. Deep learning-based feature detection methods enable reliable 2D keypoint localization of surgical instruments directly from visual input [9], [10], [11], providing essential geometric cues for markerless pose estimation [12], [13]. Furthermore, leveraging differentiable rendering techniques, prior work [14], [15] performs single-frame robot pose estimation in a renderand-compare paradigm, where the robot pose is iteratively refined by minimizing discrepancies between rendered silhouettes and observed segmentation masks. More recently, with the emergence of neural rendering, particularly Gaussian Splatting (GS) [16], a high-fidelity and differentiable robot representation [14], one may incorporate photometric consistency between rendered textures and real images, leading to improved pose estimation accuracy. A distinctive characteristic of RAMIS robot arms is their Remote Center of Motion (RCM): surgical instruments are inserted into the patientâs body through small incisions, and the motions of the instrument shaft are constrained to pivot around these fixed entry points. However, existing single-frame pose estimation methods typically neglect this physical constraint. Consequently, the estimated shaft axes may fail to satisfy the RCM geometry, resulting in kinematically inconsistent solutions and degraded pose accuracy.

In this work, we propose SurgCalib, a fully automatic and markerless hand-eye calibration framework for the da Vinci surgical system. SurgCalib first estimates the initial pose of the surgical instrument using a set of 2D-3D correspondences. The 2D keypoints are detected by a deep learningbased keypoint detector, and the 3D keypoints are derived from forward kinematics given raw joint angles. Then we integrate 3D GS-based instrument representation within a differentiable rendering pipeline to refine the surgical instrument pose in a render-and-compare manner, without the need for manual annotation or fiducial markers. By adopting a two-phase optimization strategy that explicitly incorporates the RCM constraint, our framework effectively estimates instrument poses across a sequence while preserving geometric consistency. Finally, the hand-eye transformation is computed by solving a least-squares optimization problem given the refined pose estimates.

This approach achieves promising accuracy while requiring minimal input, namely, a monocular video of arbitrary instrument motion and the corresponding kinematic data from the control software. The main contributions of this work are summarized as follows:

â¢ We develop an automatic hand-eye calibration pipeline that requires only random instrument motion captured by a monocular endoscopic camera and raw kinematic measurements, eliminating the need for manual feature labeling or carefully designed calibration trajectories.

â¢ We present the first application of 3D Gaussian Splatting to the surgical robot hand-eye calibration problem, leveraging its high-fidelity and differentiable rendering for robust pose optimization.

â¢ We propose a RCM-aware, two-phase pose optimization strategy that progressively refines the RCM position and instrument poses across a sequence, enforcing geometric consistency and compensating for kinematic uncertainties.

â¢ Finally, we quantitatively evaluate the proposed framework on the public dVRK benchmark, SurgPose. We report both 2D reprojection errors and 3D tool-tip localization errors after hand-eye calibration.

Section II presents related work, our problem is formulated in Section III, solution methods in Section IV, quantitative results in Section V and a discussion in Section VI. Qualitative results and visualizations are provided in the accompanying supplementary video.

## II. RELATED WORK

Hand-eye calibration has been a long-standing problem in robotics. Existing approaches can be categorized into two classes: 1) classic approaches; 2) learning-based approaches. In this study, we place particular emphasis on hand-eye calibration methods customized for RAMIS applications.

Classic Hand-eye Calibration: Since the 1980s, studies on hand-eye calibration have centered on solving the wellknown formulation $A X \ = \ X B ,$ , first proposed by Shiu and Ahmad [17]. In this formulation, A and B denote the motion of the robot end-effector and the camera, while X represents the hand-eye transformation to be estimated. Early analytical methods, e.g., [18], decouple rotation and translation for closed-form estimation. Various mathematical representations, e.g., Lie groups [19] and quaternions [20], have been proposed to improve numerical compactness and stability. All above-mentioned methods treat hand-eye calibration as a sensor-agnostic rigid-body motion problem, without incorporating modality-specific geometric objectives (e.g., reprojection error in vision systems), potentially limiting achievable accuracy.

Learning-Based Hand-Eye Calibration: With the advancements of deep learning, many components in the hand-eye calibration pipeline have been revolutionized by learning based techniques. Lee et al. [8] present a method that uses a deep neural network (DNN) to detect keypoints of the robot and then uses Perspective-n-point (PnP) to estimate the camera-to-robot transformation. LabbeÂ´ et al. [21] propose a camera-to-robot pose estimation model in a render-andcompare manner requiring only a single view. Lu et al. [13] further improve this line of work by introducing a backpropagatable PnP solver. Tang et al. [22] propose an easy-to-setup approach with two basic prerequisites, the robotâs kinematic chain and a predefined reference point on the robot, based on the point tracking foundation model. These methods are primarily designed for general-purpose robotic manipulators, which typically feature rigid structures, large workspaces, and accurate proprioceptive sensing. Consequently, they cannot be directly transferred to surgical robotic systems.

Hand-Eye Calibration for Surgical Robots: Early studies on hand-eye calibration for surgical robots primarily adopted classical numerical or analytical methods. Pachtrachai et al. [23] and Zhang et al. [24] both employ a dual-quaternion representation of the hand-eye rigid transformation. The former introduces a customized calibration pattern that can be grasped by the surgical instrument, whereas the latter proposes a calibration object-free method. Hwang et al. [5], and Pachtrachai et al. [25] adopt DNNs to predict the hand-eye matrix. Compared to methods that recover a fixed matrix, these learning-based approaches can model dynamic kinematic errors and compensate for time-varying system inaccuracies. Zhong et al. [3] propose a method that leverages interactive manipulation of the instrument without requiring visual feature detection. In vision systems where focus changes often, e.g., the da Vinci Si surgical system, the camera matrix varies with depth. To adapt this changing case, Kalia et al. [2] jointly estimate the hand-eye and camera calibration by minimizing the keypoint reprojection loss. Recently, Cui et al. [26] propose an online handeye calibration framework based on a training-free keypoint association algorithm using analytical Jacobian matrices.

<!-- image-->  
Fig. 1. (a) The diagram of task definition. The RCM point lies within the yellow dashed circle. The shaft centerlines are shown as green lines. (b) The frame definition of the EndoWrist surgical instrument.

## III. TASK FORMULATION

As depicted in Fig. 1(a), let $\mathcal { F } _ { c }$ and $\mathcal { F } _ { e e }$ denote the coordinate systems of the left endoscopic camera and the surgical instrumentâs end-effector, respectively. We define $\mathbf { p } _ { r c m } \in \mathbb { R } ^ { 3 }$ as the position of the Remote Center of Motion (RCM) with respect to (w.r.t.) the camera frame $\mathcal { F } _ { c }$

The objective of this hand-eye calibration is to determine the camera to robot base transformation, $^ { c } \mathbf { T } _ { r b } \in \ S E { ( 3 ) }$ which transforms the end-effector pose ${ { r } _ { { { \mathbf { T } } _ { e e } } } }$ w.r.t. the robot base frame $\mathcal { F } _ { r b }$ to the pose ${ } ^ { c } { \bf T } _ { e e }$ w.r.t. camera frame $\mathcal { F } _ { c }$ . This relationship is formulated as:

$$
{ } ^ { c } \mathbf { T } _ { e e } = { } ^ { c } \mathbf { T } _ { r b } \cdot { } ^ { r b } \mathbf { T } _ { e e } ,\tag{1}
$$

where all transformation matrices belong to the $S E ( 3 )$ . The raw pose ${ { r } _ { { \bf { ^ { b } } } } } { \bf { { T } } } _ { e e }$ can be obtained via open-source control software, e.g., da Vinci Research Kit (dVRK) [27], or the built-in research API (dVAPI) of the da Vinci robotic system.

The specific definition of the robot base frame varies across robot versions and control software.

Following [14], the forward kinematics of the Large Needle Driver (LND) is defined in Fig. 1(b). The coordinate frames for the shaft, wrist, and the left and right grippers are denoted by $\mathcal { F } _ { s } , \mathcal { F } _ { w } , \mathcal { F } _ { l }$ , and $\mathcal { F } _ { r }$ , respectively. Notably, $\mathcal { F } _ { l } , ~ \mathcal { F } _ { r }$ , and the end-effector frame $\mathcal { F } _ { e e }$ share a common origin and a coincident z-axis. Their rotational relationship is defined such that the x-axis of $\mathcal { F } _ { e e }$ bisects the gripper jaw angle. This is expressed as:

$$
\beta = \frac { \theta _ { l } + \theta _ { r } } { 2 } ,\tag{2}
$$

where $\theta _ { l } \geq \beta \geq \theta _ { \ i }$ r to ensure the definition is consistent with the mechanical constraints of the instrument.

The articulation and global pose of the surgical instrument w.r.t. $\mathcal { F } _ { c }$ is represented by the set $\mathbf { q } = \{ \theta _ { l } , \theta _ { r } , \alpha , ^ { c } \mathbf { T } _ { s } \}$ . Based on the forward knematics, the corrected pose ${ } ^ { c } { \bf T } _ { e e }$ can be computed as:

$$
{ } ^ { c } \mathbf { T } _ { e e } = { } ^ { c } \mathbf { T } _ { s } \cdot { } ^ { s } \mathbf { T } _ { w } \cdot { } ^ { w } \mathbf { T } _ { l } \cdot { } ^ { l } \mathbf { T } _ { e e } ,\tag{3}
$$

where each transformation $\mathbf { T } \in S E ( 3 )$

Let the centerline of the instrumentâs shaft be denoted by the line $\mathcal { L } _ { s }$ . The origin of $\mathcal { F } _ { s }$ , denoted by ${ \bf o } _ { s } \in \mathbb { R } ^ { 3 }$ , is a point lying on $\mathcal { L } _ { s }$ . The unit vector aligned with the shaftâs longitudinal axis is assigned to the x-axis of $\mathcal { F } _ { s } .$ , denoted by $\mathbf { x } _ { s }$ . Consequently, any point $ { \mathbf { p } } \in \mathbb { R } ^ { 3 }$ on the shaft axis can be parameterized as:

$$
\mathbf { p } ( \gamma ) = \mathbf { o } _ { s } + \boldsymbol { \gamma } \cdot \mathbf { x } _ { s } ,\tag{4}
$$

where $\gamma \in \mathbb { R }$ is a scalar parameter representing the signed distance from the origin $\mathbf { o } _ { s }$ along the axis.

## IV. METHODS

## A. GS Representation of Surgical Instruments

With the emergence of neural rendering, GS-based robot representations [28] combine the merits of both controllability and high-fidelity textures. With a fully differentiable pipeline, these representations allow for the optimization of robot states, e.g., joint angles and end-effector poses, directly through image-based losses. In this work, we adopt Instrument-Splatting [14] to represent the surgical instrument as an articulated collection of 3D Gaussians. Unlike traditional mesh-based rendering, this representation achieves photorealistic rendering while maintaining the computational efficiency required for iterative optimization.

Each Gaussian is defined by its position $\mu \ \in \ \mathbb { R } ^ { 3 }$ , a covariance matrix $\Sigma _ { j }$ (scaling vector $\mathbf { s } \in \mathbb { R } ^ { 3 }$ and a quaternion $\textbf { r } \in \ \mathbb { R } ^ { 4 } )$ , opacity $\textbf { \em { \alpha } } \in \mathbb { \Lambda }$ , and spherical harmonic coefficients sh $\in \mathbb { R } ^ { 2 7 }$ for view-dependent appearance. The surgical instrument is represented as 3D Gaussians that are partitioned into semantic sets $\mathcal { G } _ { k }$ , each bound to a rigid part $k \in \{ s , w , l , r \}$ , corresponding to the shaft, wrist, and left/right grippers, respectively. For a Gaussian point j of part k, its position $\pmb { \mu } _ { j } ^ { \prime }$ and rotation $\mathbf { r } _ { j } ^ { \prime }$ w.r.t. the camera frame are computed via the kinematic chain:

$$
\pmb { \mu } _ { j } ^ { \prime } = { } ^ { c } \mathbf { T } _ { \boldsymbol { k } } \cdot \pmb { \mu } _ { j } ; ~ \pmb { r } _ { j } ^ { \prime } = { } ^ { c } \mathbf { R } _ { \boldsymbol { k } } \cdot \pmb { r } _ { j } ,\tag{5}
$$

<!-- image-->  
Fig. 2. The schematic of our proposed pose initialization and refinement method.

<!-- image-->  
Fig. 3. Example images rendered by Instrument-Splatting.

in which

$$
{ } ^ { c } \mathbf { T } _ { k } = \left\{ { \begin{array} { l l } { ^ { c } \mathbf { T } _ { s } , } & { k = s } \\ { ^ { c } \mathbf { T } _ { s } \cdot { } ^ { s } \mathbf { T } _ { w } , } & { k = w } \\ { ^ { c } \mathbf { T } _ { s } \cdot { } ^ { s } \mathbf { T } _ { w } \cdot { } ^ { w } \mathbf { T } _ { l / r } , } & { k = l / r } \end{array} } \right.\tag{6}
$$

where $\mathbf { \Pi } ^ { s } \mathbf { T } _ { w }$ and $\boldsymbol { w } _ { \mathbf { T } _ { k } }$ are local transformations derived from the joint angles, ${ } ^ { c } \mathbf { R } _ { k }$ denotes the rotational component of rigid transformation ${ } ^ { c } \mathbf { T } _ { k }$

As shown in Fig. 3, Instrument-Splatting enables the rendering of a Large Needle Driver (LND) with high visual fidelity on arbitrary poses. Using differentiable rasterization, the gradients from a visual-wise loss can be back-propagated through the pipeline. This makes the representation uniquely suited for our two-phase optimization strategy, as it provides a dense, texture-aware signal that is more robust than sparse keypoint matching alone.

## B. Visual Features Extraction

With advances in computer vision, segmentation [29], [30] and keypoint detection models [31] have achieved significant performance improvements. In the proposed framework, we integrate two visual perception modules, Instance Segmentation and Keypoint Detection, to provide necessary visual cues for pose initialization and subsequent pose optimization. For instance segmentation, we utilize Segment Anything Model 2 [32] (SAM 2) to segment the surgical instrument. SAM 2 can robustly generate accurate instance masks of surgical instruments in the dry-lab environment without any taskspecific fine-tuning.

For keypoint detection, we adopt a deep learning-based approach, MFC-tracker [10]. This model is based on supervised deep learning architectures, e.g., DeepLabV3 [33], adding an additional refinement model for robust tracking. We choose the SurgPose [9] videos 5-7, 30-33 as the training set. All training images are resized to 640Ã512 and perform the default data augmentation strategy. The training is conducted for 200 epochs on a single NVIDIA RTX 3090Ti GPU.

## C. Pose Initialization

Given a monocular image and its corresponding raw joint angles, this pose initialization module estimates a coarse initial pose of the surgical instrument. Optimization-based pose estimation methods, such as the render-and-compare approach, require a computationally feasible initial estimate to ensure convergence and avoid local minima. We utilize noisy joint angle readings from the dVRK or research API to configure the instrumentâs kinematic model. From this configuration, the 3D positions of the instrumentâs keypoints are derived via forward kinematics.

While these 3D keypoints are subject to noisy joint angles, they provide sufficient spatial constraints for a PnP solver. We leverage the keypoint detector introduced in the previous section to extract the corresponding 2D keypoints from the image. Using these 2D-3D correspondences, we adopt the EPnP [34] algorithm to solve the initial pose qinit.

## D. RCM Estimation

The remote center of motion (RCM) is a key design in robot-assisted minimally invasive surgery (MIS) systems, as surgical instruments must pivot around a fixed incision point on the patientâs body to minimize trauma. In practice, due to mechanical tolerances and joint misalignments, the RCM is often not a singular point but a localized region (frequently modeled as a sphere [6]) during motion. Given a sequence of N observations, we denote the estimated shaft centerlines as $\{ \mathcal { L } _ { s , n } \} _ { n = 1 } ^ { N }$ . The RCM $ { \mathbf { p } } _ { r c m }$ is formulated as the point that minimizes the sum of squared perpendicular distances to these lines:

$$
\mathbf { p } _ { r c m } = \underset { \mathbf { p } } { \arg \operatorname* { m i n } } \sum _ { n = 1 } ^ { N } \| d i s t ( \mathbf { p } , \mathcal { L } _ { s , n } ) \| ^ { 2 }\tag{7}
$$

where the distance function $d i s t ( \cdot )$ is defined using the orthogonal projection onto the line:

$$
d i s t ( \mathbf { p } , \mathcal { L } _ { s , n } ) = ( \mathbf { I } - \mathbf { x } _ { s , n } \mathbf { x } _ { s , n } ^ { \top } ) ( \mathbf { p } - \mathbf { o } _ { s , n } ) .\tag{8}
$$

This formulation is a linear least-squares problem, which can be solved efficiently in closed form via the method of leastsquares intersection. According to equations (5) and (6), we can compute prcm,init given $\{ \mathcal { L } _ { s , i n i t , n } \} _ { n = 1 } ^ { N }$

## E. Two-Phase Pose Optimization

To obtain accurate poses $\{ \mathbf { q } _ { o p t , n } \} _ { n = 1 } ^ { N }$ , we perform a twophase optimization initialized from $\{ \mathbf { q } _ { i n i t , n } \} _ { n = 1 } ^ { N }$ . Since the initial RCM position $\mathbf { p } _ { r c m , i n i t }$ is estimated from noisy shaft centerlines, it does not reliably represent the true RCM point. To progressively enforce geometric consistency while avoiding early over-constraint, we design a two-phase optimization strategy, which is illustrated in Algorithm 1.

Phase 1 (Global RCM Refinement): We jointly optimize the pose parameters while dynamically updating the RCM position per epoch. This phase runs for a fixed number of M epochs. At the end of each epoch, the RCM position $ { \mathbf { p } } _ { r c m }$ is recomputed based on the updated pose estimates to progressively refine the RCM geometric constraint. The loss is defined as

$$
L _ { p h a s e 1 } = \lambda _ { s } L _ { s i l h } + \lambda _ { p } L _ { p x } + \lambda _ { k } L _ { k p t } ,\tag{9}
$$

where $L _ { s i l h }$ and $L _ { p x }$ denote the $L _ { 1 }$ losses between the rendering and the segmented instrument; $L _ { k p t }$ is the Chamfer loss of keypoints. Note that we exclude the RCM loss in this phase. As the RCM position is still being refined, prematurely enforcing the RCM constraint could steer the optimization toward an incorrect kinematic configuration and hinder convergence.

Since shaft centerline extraction may be noisy, directly fitting the RCM using all lines can lead to biased estimates. To address this, we introduce an iterative outlier rejection least-squares procedure. Specifically, after estimating the RCM, shaft centerlines with large orthogonal residuals are removed, and the RCM is recomputed using only inlier lines. This robust estimation step mitigates the influence of erroneous shaft axes observations and improves global geometric consistency.

Phase 2 (Per-Frame Pose Refinement with RCM Constraint): After Phase 1, the RCM position is frozen and treated as a fixed geometric constraint. We then perform a single-frame pose refinement for each frame independently. For image $\mathcal { T } _ { n } .$ , the render-and-compare optimization runs iteratively, and is terminated early if the total loss does not decrease for K consecutive iterations. The loss function in Phase 2 is defined as:

Algorithm 1 Two-Phase Pose Optimization   
1: procedure OPTIMIZEPOSE({qinit,n}Nn=1, {In}Nn=1)   
2: $\{ \mathbf { q } _ { n } \} _ { n = 1 } ^ { N }  \{ \mathbf { q } _ { i n i t , n } \} _ { n = 1 } ^ { N }$   
3: Phase 1: Global Refinement   
4: 5: for epoch = 1 to M doUpdate {qn}Nn=1   
6: Estimate prcm   
7: Perform outlier rejection and re-estimate prcm   
8: end for   
9: Freeze prcm   
10: Phase 2: Per-Frame Refinement   
11: for $n = 1$ to N do   
12: while loss decreases within K iterations do   
13: Update $\mathbf { q } _ { n }$   
14: end while   
15: qopt,n â qn   
16: end for   
17: return $\{ \mathbf { q } _ { o p t , n } \} , \mathbf { p } _ { r c m }$   
18: end procedure

$$
L _ { p h a s e 2 } = \lambda _ { s } L _ { s i l h } + \lambda _ { p } L _ { p x } + \lambda _ { k } L _ { k p t } + \lambda _ { r } L _ { r c m } .\tag{10}
$$

As formulated in equation (8), RCM Loss $L _ { r c m }$ is the mean squared orthogonal distance, which can be denoted as:

$$
L _ { r c m } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } d i s t ( \mathbf { p } _ { r c m } , \mathcal { L } _ { s , n } )\tag{11}
$$

After this two-phase pose optimization, we have a set of refined poses $\{ { \bf q } _ { o p t } \} _ { n = 1 } ^ { N }$ of the surgical instrument endeffector w.r.t. the camera frame $\mathcal { F } _ { c }$

## F. Compute Compensation Transformation

Given $\{ ( ^ { c } \mathbf { T } _ { e e , n } , ^ { r b } \mathbf { T } _ { e e , n } ) \} _ { n = 1 } ^ { N } .$ , N pairs of end-effector poses in $\mathcal { F } _ { c }$ and ${ \mathcal { F } } _ { r b } ,$ we aim to determine the optimal rigid transformation ${ } ^ { c } { \bf T } _ { r b }$ that minimizes the discrepancy between the end-effector poses in these two frames. To solve this in a least-squares sense, we extract their translational components (two sets of points) $\left\{ \mathbf { p } _ { c , n } \right\} _ { n = 1 } ^ { N }$ and $\{ \mathbf { p } _ { r b , n } \} _ { n = 1 } ^ { N }$ The transformation ${ } ^ { c } { \bf T } _ { r b }$ can be solved using the Kabsch-Umeyama algorithm [35], which finds the optimal rotation $\mathbf { R } ^ { \star }$ and translation tâ by minimizing the root-mean-square deviation (RMSD) between the two point clouds:

$$
\mathbf { R } ^ { \star } , \mathbf { t } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { R } , \mathbf { t } } \sum _ { n = 1 } ^ { N } w _ { n } | \mathbf { p } _ { c , n } - ( \mathbf { R } \cdot \mathbf { p } _ { r b , n } + \mathbf { t } ) | ^ { 2 } ,\tag{12}
$$

The optimal transformation matrix is ${ } ^ { c } { \bf T } _ { r b } = [ { \bf R } ^ { \star } | { \bf t } ^ { \star } ]$

## V. EVALUATION & RESULTS

To evaluate the accuracy and effectiveness of our methods, we perform experiments on a public dataset. We adopt SurgPose [9], which is collected on the first-generation da Vinci robotic system, as the public benchmark to evaluate the performance of SurgCalib on the dVRK platform. Surg-Pose provides binocular videos, stereo camera calibration, annotation of 2D keypoints, and the associated kinematic data (7D joint angles and 6D end-effector poses) read from dVRK. In this study, we resize all frames to size 640Ã512. Specifically, we use the video 0 to 4 (1000 frames per video) of SurgPose for experiments. We take video 0 as the training data to estimate the hand-eye transformation ${ ^ c { \hat { \mathbf { T } } } _ { r b } }$ . We directly use this ${ ^ c { \hat { \mathbf { T } } } _ { r b } }$ to correct the kinematic data of Videos 1-4 and compute the metrics without any training. According to the previous problem formulation, we are solving the least-squares problem:

<!-- image-->

<!-- image-->

<!-- image-->

Fig. 4. The trajectories of end-effector origins. From left to right: raw data from the dVRK, estimated positions by the proposed pose refinement approach, and the aligned trajectories using hand-eye transformation.  
<!-- image-->  
Fig. 5. The visualization of the compensated end-effector pose. The green and purple dots are the ground truth and reprojected tool tips, respectively. The upper and bottom row refers to frames with fewer and larger errors.

$$
\begin{array} { r } { ^ { c } \hat { \mathbf { T } } _ { e e } = { ^ { c } \hat { \mathbf { T } } _ { r b } } \cdot { ^ { r b } \mathbf { T } _ { e e } ^ { \# } } . } \end{array}\tag{13}
$$

SurgPose provides the $\{ { } ^ { r b } \mathbf { T } _ { e e } ^ { \# } \} _ { n = 1 } ^ { 1 0 0 0 }$ reported by the dVRK. For the data sequence in Video 0, the proposed method estimated a set of poses $\{ { ^ { c } \hat { \mathbf { T } } _ { e e , n } } \} _ { n = 1 } ^ { N }$ , where N denotes the number of valid frames after excluding outlier poses. We then use the Kabsch-Umeyama algorithm to recover the ${ ^ c { \hat { \mathbf { T } } } _ { r b } }$

Fig. 4 is a visual comparison of compensated dVRK trajectories and estimated trajectories using the proposed pose refine strategy. Fig. 5 is the visualization of the end-effector pose after hand-eye calibration. We visualize the shaft centerlines before and after adding RCM loss in Fig 6 to qualitatively demonstrate that the RCM constraint can effectively ensure that the poses satisfy this physical constraint.

To quantitatively validate this ${ ^ { c } \hat { \mathbf { T } } _ { r b } } .$ , we project the positions of tool tips to the image given the camera matrix. We separately do the hand-eye calibration for the left and right instruments. We employ the average and median Euclidean distance (unit: pixel/millimeter) between the reprojected tool tips and their ground truth as metrics for evaluation. The results are shown in Table I. Note that the original unit of this error is in pixels. We convert the pixel error to metric error (mm) by multiplying a scaling factor $\begin{array} { r } { s = \frac { Z } { f _ { x } } } \end{array}$ , where $Z$ and $f _ { x }$ are the tool tip depth from triangulation and x-axis focal length, respectively. As shown in Fig 7, we plot the 2D error versus frames to better analyze the error distribution.

<!-- image-->  
Fig. 6. Visual comparison of the shaft centerline convergence around RCM. (a) Shaft axes after phase 1 optimization; (b) Shaft axes after phase 2 optimization with the RCM constrain. Note that the RCM (blue point) is re-estimated after optimization.

TABLE I  
TOOL TIPS 2D REPROJECTION ERROR (PX/MM), L - LEFT, R - RIGHT.
<table><tr><td>Video 0</td><td>Video 1</td><td>Video 2</td><td>Video 3</td><td>Video 4</td></tr><tr><td>Avg. Err. (L) 9.74/1.52</td><td>15.47/2.33</td><td>15.04/2.38</td><td>8.64/1.71</td><td>12.31/2.44</td></tr><tr><td>Mdn. Err. (L) 9.45/1.44</td><td>17.08/2.44</td><td>12.94/2.05</td><td>8.57/1.71</td><td>12.00/2.42</td></tr><tr><td> $\overline { { A \nu } } g . \ \overline { { E } } r r \overline { { { \bf \Omega } } } ( \overline { { { \bf R } } } )$  7.71/1.49</td><td>10.33/3/1.50</td><td>7.52/1.41</td><td></td><td>17.43/2.9013.65/2.22</td></tr><tr><td>Mdn. Err. (R) 6.83/1.29</td><td>9.84/1.48</td><td>7.08/1.34</td><td></td><td>16.59/2.63 11.98/2.05</td></tr></table>

Using the stereo calibration parameters in SurgPose, corresponding keypoints ${ \bf x } _ { r }$ and $\mathbf { x } _ { l }$ are triangulated to recover their 3D positions x in the left camera frame:

$$
\mathbf { x } = \arg \operatorname* { m i n } _ { \mathbf { x } } \left( \| \mathbf { x } _ { l } - \pi ( \mathbf { P } _ { l } \mathbf { x } ) \| ^ { 2 } + \| \mathbf { x } _ { r } - \pi ( \mathbf { P } _ { r } \mathbf { x } ) \| ^ { 2 } \right)\tag{14}
$$

where $\pi ( \cdot )$ denotes the perspective projection that converts homogeneous coordinates to image coordinates. $\mathbf { P } _ { l } =$ ${ \bf K } _ { L } [ { \bf I } \mid \mathbf { 0 } ]$ and ${ \bf P } _ { r } = { \bf K } _ { R } [ { \bf R } \mid { \bf t } ]$ are the stereo projection matrices derived from calibration.

We assume this x is the ground truth for the tool tip position. Given the joint angles and the corrected 6D pose, we can derive xË, the 3D positions of the tool tips in the camera frame, using forward kinematics. As shown in Table II, we adopt the tool tip 3D Euclidean distance error $\lVert \mathbf x - \hat { \mathbf x } \rVert$ to evaluate the performance.

TABLE II  
TOOL TIPS 3D EUCLIDEAN ERROR (MM), L - LEFT, R - RIGHT.
<table><tr><td></td><td></td><td></td><td></td><td></td><td>Video 0 Video 1 Video 2 Video 3 Video 4</td></tr><tr><td>Avg. Err. (L)</td><td>4.84</td><td>5.46</td><td>5.66</td><td>7.22</td><td>6.76</td></tr><tr><td>Mdn. Err. (L)</td><td>4.75</td><td>5.45</td><td>5.65</td><td>7.11</td><td>6.75</td></tr><tr><td>Avg. Err. (R)</td><td>6.53</td><td>4.51</td><td>4.57</td><td>4.19</td><td>4.56</td></tr><tr><td>Mdn. Err. (R)</td><td>6.53</td><td>4.55</td><td>4.57</td><td>4.30</td><td>4.71</td></tr></table>

## VI. LIMITATION & DISCUSSION

In da Vinci Classic Surgical System, the reported dVRK end-effector pose is w.r.t. the ECM tip coordinate system. According to equation (1), the formulation can be denoted as ${ } ^ { c } \mathbf { T } _ { e e } = { } ^ { c } \mathbf { T } _ { e c m } \cdot { } ^ { e c m } \mathbf { T } _ { e e } ,$ and we aim to recover the camera to ECM transformation ${ } ^ { c } \mathbf { T } _ { e c m }$ . In the dVRK convention, the ECM frame origin is located at the midpoint of the stereo camera baseline. A naive assumption is therefore that the camera translation is approximately half of the baseline (â¼2.5mm) [36]. However, in practice this rough approximation leads to centimeter-level translation errors of the end-effector, and the reprojected end-effector may even fall outside the image frame.

<!-- image-->  
Fig. 7. The tool tip reprojection error distribution across the frame numbers.

While Gaussian Splatting (GS) has demonstrated strong potential for general robotic self-representation, its application to articulated surgical instruments remains in an early stage of development. Based on our empirical observations, several limitations of the current instrument GS representation persist and highlight important directions for future improvement: 1) Limited novel-view rendering fidelity. Although the geometry-aware pretraining of Instrument-Splatting enables robust semantic mask rendering, the photorealistic quality of novel-view synthesis remains limited. 2) Single instrument category coverage. The current framework is only validated on LND, because training an Instrument-Splatting model for a specific instrument requires its corresponding CAD model. Thus, developing CAD-free GS representations across multiple instrument categories is a promising direction for future work. 3) Absence of explicit lighting modeling. The present formulation does not explicitly model illumination, which is a major source of domain discrepancy between rendered images and real observations. Incorporating lighting-aware rendering or appearance adaptation may further reduce this gap and improve robustness.

In addition to the limitations of the GS representation, several sources of uncertainty are not explicitly modeled in the current framework. The optical system of the modern endoscopic camera is highly complex. Kalia et al. [37] suggest that the depth of the effective focal plane may vary with the surgical operative distance, i.e., the instrument depth. Furthermore, the classic Brown-Conrady model is insufficient to represent the intricate distortion of the endoscopic lens stack. Inspired by advances in neural lens modeling [38], [39], integrating a learnable camera model into the GSbased differentiable rendering pipeline to jointly optimize the camera intrinsics and distortion is an attractive direction for future work. As shown in Fig. 7, the reprojection error exhibits temporal variation, suggesting a potential correlation with the kinematic parameters. In the proposed framework, we do not model this correlation. Incorporating a lightweight learning module to capture such correlations might further enhance the hand-eye calibration accuracy.

## VII. CONCLUSION

We present SurgCalib, an automatic and markerless handeye calibration framework tailored for the da Vinci surgical system. To the best of our knowledge, this work is the first exploration of leveraging Gaussian Splatting (GS) representations for hand-eye calibration in surgical robotics. Experimental evaluation on the public dVRK SurgPose demonstrates its effectiveness in achieving accurate 2D reprojection and 3D tool-tip localization.

## REFERENCES

[1] O. Mohareri, J. Ischia, P. C. Black, C. Schneider, J. Lobo, L. Goldenberg, and S. E. Salcudean, âIntraoperative registered transrectal ultrasound guidance for robot-assisted laparoscopic radical prostatectomy,â The Journal of urology, vol. 193, no. 1, pp. 302â312, 2015.

[2] M. Kalia, A. Avinash, N. Navab, and S. Salcudean, âPreclinical evaluation of a markerless, real-time, augmented reality guidance system for robot-assisted radical prostatectomy,â International Journal of Computer Assisted Radiology and Surgery, vol. 16, no. 7, pp. 1181â 1188, 2021.

[3] F. Zhong, Z. Wang, W. Chen, K. He, Y. Wang, and Y.-H. Liu, âHand-eye calibration of surgical instrument for robotic surgery using interactive manipulation,â IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 1540â1547, 2020.

[4] Z. Cui, J. Cartucho, S. Giannarou, and F. R. y Baena, âCaveats on the first-generation da vinci research kit: Latent technical constraints and essential calibrations [survey],â IEEE Robotics & Automation Magazine, vol. 32, no. 2, pp. 113â128, 2023.

[5] M. Hwang, B. Thananjeyan, S. Paradis, D. Seita, J. Ichnowski, D. Fer, T. Low, and K. Goldberg, âEfficiently calibrating cable-driven surgical robots with rgbd fiducial sensing and recurrent neural networks,â IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 5937â5944, 2020.

[6] B. Lu, B. Li, Q. Dou, and Y. Liu, âA unified monocular camera-based and pattern-free hand-to-eye calibration algorithm for surgical robots with rcm constraints,â IEEE/ASME Transactions on Mechatronics, vol. 27, no. 6, pp. 5124â5135, 2022.

[7] N. Greene, A. Long, Y. Long, Z. Han, Q. Dou, and P. Kazanzides, âMarkerless tracking of robotic surgical instruments with head mounted display for augmented reality applications,â Healthcare Technology Letters, vol. 12, no. 1, p. e70044, 2025.

[8] T. E. Lee, J. Tremblay, T. To, J. Cheng, T. Mosier, O. Kroemer, D. Fox, and S. Birchfield, âCamera-to-robot pose estimation from a single image,â in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 9426â9432.

[9] Z. Wu, A. Schmidt, R. Moore, H. Zhou, A. Banks, P. Kazanzides, and S. E. Salcudean, âSurgpose: a dataset for articulated robotic surgical tool pose estimation and tracking,â in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 10 507â10 514.

[10] B. Ghanekar, L. R. Johnson, J. L. Laughlin, M. K. OâMalley, and A. Veeraraghavan, âVideo-based surgical tool-tip and keypoint tracking using multi-frame context-driven deep learning models,â in 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI). IEEE, 2025, pp. 1â5.

[11] Z. Wu, S. Yang, Y. Jin, and S. E. Salcudean, âTooltipnet: A segmentation-driven deep learning baseline for surgical instrument tip detection,â arXiv preprint arXiv:2504.09700, 2025.

[12] Z. Liang, K. Miyata, X. Liang, F. Richter, and M. C. Yip, âEfficient surgical robotic instrument pose reconstruction in real world conditions using unified feature detection,â arXiv preprint arXiv:2510.03532, 2025.

[13] J. Lu, F. Richter, and M. C. Yip, âMarkerless camera-to-robot pose estimation via self-supervised sim-to-real transfer,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21 296â21 306.

[14] S. Yang, Z. Wu, M. Hong, Q. Li, D. Shen, S. E. Salcudean, and Y. Jin, âInstrument-splatting: Controllable photorealistic reconstruction of surgical instruments using gaussian splatting,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2025, pp. 301â311.

[15] Z. Liang, Z.-Y. Chiu, F. Richter, and M. C. Yip, âDifferentiable rendering-based pose estimation for surgical robotic instruments,â in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 20 898â20 905.

[16] B. Kerbl, G. Kopanas, T. Leimkuhler, G. Drettakis Â¨ et al., â3d gaussian splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[17] Y. Shiu and S. Ahmad, âCalibration of wrist-mounted robotic sensors by solving homogeneous transform equations of the form ax=xb,â IEEE Transactions on Robotics and Automation, vol. 5, no. 1, pp. 16â29, 1989.

[18] R. Y. Tsai, R. K. Lenz et al., âA new technique for fully autonomous and efficient 3 d robotics hand/eye calibration,â IEEE Transactions on robotics and automation, vol. 5, no. 3, pp. 345â358, 1989.

[19] F. C. Park and B. J. Martin, âRobot sensor calibration: solving ax= xb on the euclidean group,â IEEE Transactions on Robotics and Automation, vol. 10, no. 5, pp. 717â721, 1994.

[20] K. Daniilidis, âHand-eye calibration using dual quaternions,â The International Journal of Robotics Research, vol. 18, no. 3, pp. 286â 298, 1999.

[21] Y. Labbe, J. Carpentier, M. Aubry, and J. Sivic, âSingle-view robot Â´ pose and joint angle estimation via render & compare,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 1654â1663.

[22] T. Tang, M. Liu, W. Xu, and C. Lu, âKalib: Easy hand-eye calibration with reference point tracking,â in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 17 948â17 955.

[23] K. Pachtrachai, M. Allan, V. Pawar, S. Hailes, and D. Stoyanov, âHand-eye calibration for robotic assisted minimally invasive surgery without a calibration object,â in 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2016, pp. 2485â2491.

[24] Z. Zhang, L. Zhang, and G.-Z. Yang, âA computationally efficient method for handâeye calibration,â International journal of computer assisted radiology and surgery, vol. 12, no. 10, pp. 1775â1787, 2017.

[25] K. Pachtrachai, F. Vasconcelos, P. Edwards, and D. Stoyanov, âLearning to calibrate-estimating the hand-eye transformation without calibration objects,â IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 7309â7316, 2021.

[26] Z. Cui et al., âOn-the-fly hand-eye calibration for the da vinci surgical robot,â arXiv preprint arXiv:2601.14871, 2026.

[27] P. Kazanzides, Z. Chen, A. Deguet, G. S. Fischer, R. H. Taylor, and S. P. DiMaio, âAn Open-Source Research Kit for the da VinciÂ® Surgical System,â in IEEE Intl. Conf. on Robotics and Automation (ICRA), 2014, pp. 6434â6439.

[28] R. Liu, A. Canberk, S. Song, and C. Vondrick, âDifferentiable robot rendering,â arXiv preprint arXiv:2410.13851, 2024.

[29] Z. Wu, A. Schmidt, P. Kazanzides, and S. E. Salcudean, âAugmenting efficient real-time surgical instrument segmentation in video with point tracking and Segment Anything,â Healthcare Technology Letters, vol. 12, no. 1, p. e12111, 2025.

[30] W. Yue, J. Zhang, K. Hu, Y. Xia, J. Luo, and Z. Wang, âSurgicalsam: Efficient class promptable surgical instrument segmentation,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 6890â6898.

[31] Z. Han, C. Budd, G. Zhang, H. Tian, C. Bergeles, and T. Vercauteren, âRobust-mips: A combined skeletal pose and instance segmentation dataset for laparoscopic surgical instruments,â arXiv preprint arXiv:2508.21096, 2025.

[32] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson Â¨ et al., âSam 2: Segment anything in images and videos,â arXiv preprint arXiv:2408.00714, 2024.

[33] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam, âRethinking atrous convolution for semantic image segmentation,â arXiv preprint arXiv:1706.05587, 2017.

[34] V. Lepetit, F. Moreno-Noguer, and P. Fua, âEp n p: An accurate o (n) solution to the p n p problem,â International journal of computer vision, vol. 81, no. 2, pp. 155â166, 2009.

[35] S. Umeyama, âLeast-squares estimation of transformation parameters between two point patterns,â IEEE Transactions on pattern analysis and machine intelligence, vol. 13, no. 4, pp. 376â380, 2002.

[36] A. Avinash, A. E. Abdelaal, P. Mathur, and S. E. Salcudean, âA âpickupâ stereoscopic camera with visual-motor aligned control for the da vinci surgical system: a preliminary study,â International journal

of computer assisted radiology and surgery, vol. 14, no. 7, pp. 1197â 1206, 2019.

[37] M. Kalia, P. Mathur, K. Tsang, P. Black, N. Navab, and S. Salcudean, âEvaluation of a marker-less, intra-operative, augmented reality guidance system for robot-assisted laparoscopic radical prostatectomy,â International Journal of Computer Assisted Radiology and Surgery, vol. 15, no. 7, pp. 1225â1233, 2020.

[38] W. Xian, A. BoziË c, N. Snavely, and C. Lassner, âNeural lens model- Ë ing,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8435â8445.

[39] Y. Deng, W. Xian, G. Yang, L. Guibas, G. Wetzstein, S. Marschner, and P. Debevec, âSelf-calibrating gaussian splatting for large field-ofview reconstruction,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 25 124â25 133.