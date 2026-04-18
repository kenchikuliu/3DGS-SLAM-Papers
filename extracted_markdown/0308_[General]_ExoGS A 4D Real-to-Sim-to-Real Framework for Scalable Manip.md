# ExoGS: A 4D Real-to-Sim-to-Real Framework for Scalable Manipulation Data Collection

Yiming Wang, Ruogu Zhang, Minyang Li, Hao Shi, Junbo Wang, Deyi Li, Jieji Ren, Wenhai Liu, Weiming Wang, Hao-Shu Fang Shanghai Jiao Tong University

<!-- image-->  
Fig. 1. Overview of ExoGS. (a) Real-world demonstrations captured by AirExo-3 are reconstructed into editable 3D Gaussian scenes. (b) These assets enable scalable data augmentation across viewpoints and lighting. (c) A semantic Mask Adapter facilitates robust sim-to-real policy transfer for manipulation tasks.

AbstractâReal-to-Sim-to-Real technique is gaining increasing interest for robotic manipulation, as it can generate scalable data in simulation while having narrower sim-to-real gap. However, previous methods mainly focused on environment-level visual real-to-sim transfer, ignoring the transfer of interactions, which could be challenging and inefficient to obtain purely in simulation especially for contact-rich tasks. We propose ExoGS, a robotfree 4D Real-to-Sim-to-Real framework that captures both static environments and dynamic interactions in the real world and transfers them seamlessly to a simulated environment. It provides a new solution for scalable manipulation data collection and policy learning. ExoGS employs a self-designed robot-isomorphic passive exoskeleton AirExo-3 to capture kinematically consistent trajectories with millimeter-level accuracy and synchronized RGB observations during direct human demonstrations. The robot, objects, and environment are reconstructed as editable 3D Gaussian Splatting assets, enabling geometry-consistent replay and large-scale data augmentation. Additionally, a lightweight Mask Adapter injects instance-level semantics into the policy to enhance robustness under visual domain shifts. Real-world experiments demonstrate that ExoGS significantly improves data efficiency and policy generalization compared to teleoperationbased baselines. Code and hardware files have been released on https://github.com/zaixiabalala/ExoGS.

Index TermsâRobotic manipulation, imitation learning, demonstration data collection, real-to-sim-to-real, 3D Gaussian Splatting, data augmentation, sim-to-real transfer.

## I. INTRODUCTION

MITATION learning has enabled robots to acquire complex manipulation skills from demonstrations. However, its efficacy is heavily based on the scale and quality of the training data. Simulation-based data synthesis offers scalability, but suffers from the well-known sim-to-real gap due to discrepancies in geometric representation, visual appearance, and physical interaction, particularly in visual fidelity for RGBbased policies.

Recent Real-to-Sim-to-Real (R2S2R) pipelines with neural scene representations, such as NeRF and 3D Gaussian Splatting (3DGS) [1], [2], have shown promise in bridging the visual gap through photorealistic rendering. However, most existing R2S2R pipelines are limited to static scene reconstruction and rely on reinforcement learning to obtain manipulation data. This challenge lies in acquiring physically valid, high-fidelity interaction data without the burden of deploying expensive robotic hardware.

To address this, this paper introduces ExoGS, a low-cost, robot-free R2S2R framework that allows users to capture 4D sequences comprising 3D environments and spatiotemporal robot interactions with objects. Our framework consists of a custom passive exoskeleton with a 3DGS-based data generation pipeline. The core insight is to leverage a robot-isomorphic exoskeleton, AirExo-3, to sensorize and capture human manipulation. The manipulation data including exoskeleton joint configurations and object trajectories are captured by our platform (see Sec. III-B1) and then digitized into a motion sequence of editable 3DGS assets of the robot and the objects. This hardware-software synergy allows us to decouple the robot, objects, and environment, enabling massive, geometry-consistent data augmentation in simulation. Furthermore, to mitigate the residual domain shift, we introduce a Mask Adapter, a lightweight module that injects instance-level semantic constraints into the policy, guiding attention toward interaction-relevant features.

The main contributions of this work are threefold:

â¢ A 4D, robot-free R2S2R framework that reconstructs real-world assets and manipulation sequences into editable 3DGS assets and their dynamics, enabling scalable, embodiment-consistent data generation.

â¢ AirExo-3, an open-source, low-cost, accurate, and durable manipulation data collection device.

â¢ A Mask Adapter that injects semantic mask information into the policy to steer attention toward interactionrelevant regions, thereby enhancing robustness.

## II. RELATED WORKS

## A. Sim-to-Real Transfer and 3D Gaussian Splatting

Due to persistent discrepancies in appearance, geometry, and contact dynamics, policies trained in simulation often fail to generalize to the real world, motivating Real-to-Sim-to-Real (R2S2R) pipelines. R2S2R pipelines reconstruct photorealistic digital twins from real scenes and perform data generation and policy learning in reconstructed environments, leveraging neural radiance fields or 3D Gaussian Splatting for high-fidelity rendering and novel view synthesis [3], [4]. These approaches generally fall into interaction-oriented frameworks that support closed-loop training and rendering-centered pipelines that scale data generation via photorealistic replay [5]â[9], yet limitations in physical interactability and contact modeling persist.

Compared with NeRF [1], 3DGS [2] offers an explicit, point-based representation with fast optimization and realtime rendering, making it suitable for robotics. With recent extensions incorporating dynamics and physical priors [10], [11], 3DGS has emerged as an editable, geometry-aware world model for manipulation and navigation [6], [12]â[15], enabling geometry-consistent edits beyond 2D augmentation [16]. Nevertheless, effectively leveraging 3DGS for robot learning requires tight integration of demonstration acquisition, pose modeling, and interaction-aware replay.

Therefore, a Real-to-Sim-to-Real framework that tightly couples real-world demonstration acquisition, accurate pose modeling, and geometry-consistent 3DGS replay is still needed to support scalable, contact-rich manipulation learning with high visual fidelity, a gap that ExoGS addresses.

## B. Demonstration Data Collection

Demonstration data is fundamental to scalable and generalizable imitation learning in robotic manipulation [17]â[21]. Although teleoperation enables embodiment-consistent robot demonstrations, its scalability is fundamentally constrained by hardware cost, deployment overhead, and non-intuitive humanrobot interfaces.

To reduce these limitations, an alternative direction focuses on in-the-wild demonstration collection using human-operated devices such as handheld grippers, VR/AR systems, motioncapture gloves, and exoskeletons [22]â[27]. These systems reduce deployment cost and enable more natural demonstrations, but often suffer from limited kinematic fidelity, trajectory accuracy, or environment diversity.

Therefore, existing approaches lack a low-cost, robot-free data collection framework that preserves robot-aligned kinematics and supporting scalable, geometry-consistent replay, motivating the Real-to-Sim-to-Real design of ExoGS.

## III. METHOD

ExoGS first employs AirExo-3, a low-cost exoskeleton (detailed in Sec. III-A), to capture accurate robot manipulation demonstrations without needing a real robot (detailed in Sec. III-B (1)). Based on these demonstrations, 3D Gaussian Splatting is utilized to replay and augment the demonstrated interactions (detailed in Sec. III-B (2)-(4)). Building upon this real-to-sim dataset, we further propose a novel plug-in module, Mask Adapter, to facilitate more effective sim-to-real transfer of visuomotor policies (detailed in Sec. III-C).

## A. Hardware Design: AirExo-3

AirExo-3 extends prior low-cost exoskeleton-based demonstration systems [22], [23] by improving kinematic accuracy and ease of deployment. The design of AirExo-3 targets the following objectives:

â¢ Ease of deployment: AirExo-3 supports accurate zeroposition calibration and straightforward installation without specialized tools, enabled by a simplified mechanical design.

â¢ High accuracy: Unlike handheld devices such as UMI [24], AirExo-3 achieves robust and accurate motion tracking with millimeter-level accuracy via forward kinematics using high-precision rotary encoders, remaining insensitive to visual conditions while enforcing robotconsistent kinematic constraints.

â¢ Ease of use: AirExo-3 enables intuitive, contact-rich manipulation through direct gripper interaction, while its lightweight design supports prolonged use with minimal operator fatigue.

<!-- image-->  
Fig. 2. Mechanical structure of the proposed data acquisition device. (a) Structural design of an individual joint module. (b) Overall structure of AirExo-3, consisting of seven articulated joints and a parallel gripper. (c) The target robotic platform used in this work.

The structure of AirExo-3 is shown in Fig. 2. It is a serial link-joint chain geometrically matched to the target robot and equipped with a replaceable handheld gripper, sharing identical kinematic parameters, joint limits, and gripper opening range to ensure workspace consistency. All components are 3D-printed with glass-fiber-reinforced nylon for a lightweight and robust design, with an optional spring-based suspension at Joint 4 to reduce operator effort.

The core component of AirExo-3 is the joint module, which consists of a joint shaft, bearings, and a joint body housing a 12-bit miniature rotary encoder, as shown in Fig. 2 (a). Eight encoders are connected via a shared bus, enabling synchronized joint state acquisition at up to approximately 300 Hz. The joint shaft defines the rotational limits, while crossroller bearings connect the shaft and body, providing high stiffness and rotational accuracy. An established evaluation protocol for exoskeleton-based manipulation systems [23] is adopted to assess kinematic accuracy. The results demonstrate that AirExo-3 achieves an average end-effector positioning error of less than 1 mm.

Limit slots are machined into the joint shaft, and a highstrength pin fixed to the joint body engages with the slot to define the jointâs angular limits. When the pin reaches the slot boundary, further rotation is mechanically prevented. The pin hole on the joint body also serves as a reference for zeroposition calibration. Calibration is performed by mechanically locking the joint at its zero position using an extended pin and recording the corresponding encoder reading.

Each joint incorporates a groove with a 3D-printed retaining ring to provide tunable passive damping, enabling stable and comfortable operation.

## B. 4D Data Generation and Augmentation

<!-- image-->  
Fig. 3. Overview of the pipeline for reconstructing manipulation demonstrations collected with AirExo-3 using 3D Gaussian representations. Camerabased object pose tracking and joint angle encoding from AirExo-3 are combined to generate robot motions, which, together with the reconstructed digital assets, allow the full manipulation process to be faithfully replayed in simulation.

1) Manipulation Demonstration Collection with AirExo-3: Let the joint-angle vector of the robotic arm be denoted as $\pmb { q } = [ q _ { 1 } , q _ { 2 } , \ldots , q _ { n } ] ^ { \top }$ , and let the gripper opening be $g \in [ 0 , 1 ]$

Each demonstration trajectory can then be represented as a discrete-time sequence:

$$
{ \boldsymbol { \tau } } = \{ ( \boldsymbol { q } _ { t } , \boldsymbol { g } _ { t } ) \} _ { t = 1 } ^ { H } ,\tag{1}
$$

where H denotes the number of time steps.

Since AirExo-3 is geometrically and kinematically identical to the target robot, the collected joint states $\pmb { q } _ { t }$ can be directly used for forward kinematics computation, enabling high-fidelity reproduction of manipulation trajectories in both real and simulated environments.

Multiple calibrated Intel RealSense D415 cameras synchronously capture multi-view RGB-D observations $\begin{array} { r l } { { \mathcal { Z } } } & { { } = } \end{array}$ $\{ I _ { t } ^ { ( k ) } \} _ { t = 1 , k = 1 } ^ { H , \check { K } }$ , which are unified in a common world coordinate frame to provide geometric constraints for subsequent 3D reconstruction and pose estimation.

2) Digital Asset Generation via Multi-View Reconstruction: We adopt 3D Gaussian Splatting (3DGS) to digitize real-world scenes into editable simulation assets. Following the standard formulation [2], we represent the scene (including the robot, objects, and environment) as a collection of 3D Gaussians, each parameterized by position, covariance, opacity, and spherical harmonics. We implement a âcapture-reconstruct-assetizeâ pipeline: multi-view images are first captured and processed via COLMAP [28] to recover camera poses. These poses initialize the optimization of Gaussian parameters, which is driven by minimizing a weighted L1 and SSIM photometric loss between rendered and captured views. This process yields high-fidelity, decoupled 3D assets for the robotic arm and manipulated objects, enabling independent manipulation and geometry-consistent replay in the simulation environment.

3) Object Pose Estimation and Trajectory Processing: Multi-view RGB-D sequences are processed with FoundationPose [29] for object pose tracking. The pose of object o at time t in camera k is denoted as $\mathbf { T } _ { o , t } ^ { ( k ) } \in \ S E ( 3 )$ . To improve robustness and consistency, we integrate these multiview estimates by adopting the rotation from the primary camera as the global orientation and averaging the translation vectors across all views. This fusion results in a unified pose sequence $\{ \mathbf { T } _ { o , t } \} _ { t = 1 } ^ { H }$ in the robot base coordinate frame.

To enhance data diversity across task scenarios, we introduce a lightweight pose-processing module, PoseProcess, which performs normalization and recomposition of object pose sequences. A fix operation constrains the object pose to the robot end-effector frame, enabling rigid attachment during manipulation. By substituting object models, the same pose sequence can be directly transferred to different objects, allowing the generation of diverse task instances without additional data collection.

Robot kinematics are computed using the URDF model of the target robot with the recorded joint-angle sequence $\left\{ \pmb q _ { t } \right\}$ The pose of each link â is obtained via forward kinematics:

$$
\begin{array} { r } { \mathbf { T } _ { \ell , t } = \mathrm { F K } _ { \ell } ( \mathbf { q } _ { t } ) , \quad \ell = 1 , \dots , L , t = 1 , \dots , H , } \end{array}\tag{2}
$$

where L denotes the number of robot links. The resulting robot link poses $\mathbf { T } _ { \ell , t }$ are combined with object poses $\mathbf { T } _ { o , t }$ in the Gaussian rendering pipeline, enabling multi-view rendering of robot-object scenes for policy training.

<!-- image-->  
Fig. 4. Overview of the proposed Mask Adapter. The module is trained in two stages. Stage 1 performs semantic segmentation pretraining using pixel-level supervision generated by the 3D Gaussian Splatting pipeline, yielding stable patch-level semantic labels for the background, robotic arm, and manipulated objects. Stage 2 incorporates these semantic cues into a ViT-based imitation learning policy via enhanced positional encodings and mask-guided attention, encouraging interaction-relevant token communication to improve robustness and cross-scene generalization under visual domain shifts.

4) Data Augmentation under Gaussian Rendering: To improve policy robustness under real-world variations, we employ four data augmentation strategies leveraged by the editable 3DGS representation. First, camera viewpoint augmentation renders demonstrations from perturbed extrinsics to simulate camera placement changes. Second, color and illumination augmentation applies random scaling to Gaussian color attributes and global/local brightness, addressing appearance and lighting gaps. Third, background augmentation composites diverse real-world images as background textures behind the geometry-consistent Gaussian foregrounds, encouraging background-invariant policy learning. Finally, object pose augmentation perturbs object poses and scales, or substitutes objects with affordance-compatible alternatives, enabling trajectory reuse and improving robustness to physical variations.

## C. Policy Module: Mask Adapter

1) Introduction: Although in-the-wild demonstrations can be converted into pseudo-robot data, domain gaps in viewpoint, background, and lighting still hinder the generalization of 2D visuomotor policies. Most imitation learning methods lack explicit interaction-centric inductive biases, making them susceptible to spurious background correlations under distribution shifts.

Leveraging the explicit instance-level representation of 3D Gaussian Splatting (3DGS), our pipeline generates pixel-wise semantic masks and supports geometry-consistent data augmentation via multi-view rendering, appearance perturbation, and background replacement, effectively narrowing the sim-toreal gap. Building on an enhanced ACT [26] backbone with a DINOv3 ViT encoder [30] and LoRA fine-tuning [31], we introduce Mask Adapter, a lightweight module that injects patch-level semantic cues into ViT-based policies to guide attention toward interaction-relevant regions and improve robustness and cross-scene generalization. The full pipeline is shown in Fig. 4

Given an observation image sequence $I _ { 1 : T }$ , a ViT encoder produces patch tokens and base positional encodings:

$$
\begin{array} { r } { \mathbf { x } = E _ { \mathrm { v i t } } ( I _ { 1 : T } ) \in \mathbb { R } ^ { B \times ( T N ) \times D } , \qquad \mathbf { p } \in \mathbb { R } ^ { B \times ( T N ) \times D } , } \end{array}
$$

where N is the number of patches per frame and D is the hidden dimension. Mask Adapter augments this pipeline with a mask head and label-driven attention constraints, trained in two stages.

2) Mask Head: When trained only with action supervision, 2D policies receive sparse and highly task-specific gradients, making it difficult to learn stable semantic structures (e.g., background vs. arm vs. objects). Under domain shifts, this often amplifies attention drift. Therefore, in Stage 1 we fine-tune the visual encoder and a segmentation head with pixel-level supervision to obtain transferable semantics and to provide patch-level labels for Stage 2. We adopt a lightweight multiscale segmentation head $H _ { \mathrm { m a s k } }$ , following ASPP [32] style. After reshaping tokens into a feature map $\mathbf { F } \in \mathbb { R } ^ { B \times D \times h \times w }$ , the head predicts pixel logits:

$$
\mathbf { S } = H _ { \mathrm { m a s k } } ( \mathbf { F } ) \in \mathbb { R } ^ { B \times C \times H \times W } .
$$

To align with the token sequence, we aggregate pixel predictions into patch-level labels $\ell \in \{ 0 , \ldots , C - 1 \} ^ { T N }$ . Let $\Omega _ { n }$ be the set of pixels belonging to patch $n ;$ we compute:

$$
\ell _ { n } = \arg \operatorname* { m a x } _ { c \in \{ 0 , \ldots , C - 1 \} } \ \frac { 1 } { | \Omega _ { n } | } \sum _ { u \in \Omega _ { n } } \mathrm { s o f t m a x } ( \mathbf { S } _ { u } ) _ { c } .
$$

3) Mask-guided Token Modeling: Standard Transformers in imitation learning policies typically treat all patches uniformly and provide no structural priors on which tokens should interact. As a result, under occlusions or background changes, the model may aggregate irrelevant context and fail to generalize. In Stage 2, we use patch labels â to enhance positional encodings and impose label-driven interaction constraints, encouraging semantic-consistent and interaction-relevant token communication. We add a learnable label embedding to the base positional encoding:

<!-- image-->  
Fig. 5. Real-world experimental setup and task illustration. A Flexiv Rizon 4s robotic arm is used together with two eye-on-base Intel RealSense D415 cameras. During standard experiments and teleoperation data collection, only the left camera is used, while the other camera is reserved exclusively for evaluating camera viewpoint variations. Three manipulation tasks are designed in this work, whose detailed descriptions are provided in Sec. IV.

$$
\begin{array} { r } { \tilde { \bf p } = { \bf p } + { \bf E } _ { \mathrm { l a b e l } } ( \ell ) . } \end{array}
$$

We define a label relation set R and construct an additive attention mask A:

$$
\mathbf { A } _ { i j } = \left\{ { \begin{array} { l l } { 0 , } & { ( \ell _ { i } , \ell _ { j } ) \in \mathcal { R } } \\ { - \infty , } & { ( \ell _ { i } , \ell _ { j } ) \notin \mathcal { R } } \end{array} } , \right.
$$

which is added to attention logits.

4) Training Objective: We train segmentation with (optionally class-weighted) pixel-level cross-entropy:

$$
\mathcal { L } _ { \mathrm { s e g } } = - \frac { 1 } { | \Omega | } \sum _ { u \in \Omega } w _ { y _ { u } } \log \operatorname { s o f t m a x } ( \mathbf { S } _ { u } ) _ { y _ { u } } .
$$

Stage 2 optimizes the original policy action loss $\mathcal { L } _ { \mathrm { a c t } }$ jointly with segmentation to stabilize semantic alignment:

$$
\mathcal { L } _ { \mathrm { s t a g e 2 } } = \mathcal { L } _ { \mathrm { a c t } } + \lambda \mathcal { L } _ { \mathrm { s e g } } .
$$

When ground-truth masks are unavailable, one can optimize $\mathcal { L } _ { \mathrm { a c t } }$ only while still using Stage-1 predictions to provide â for positional enhancement and attention constraints. Since Mask Adapter only requires ViT tokens, positional encodings, and an attention interface, it can be integrated into most ViT-based 2D imitation learning policies with minimal architectural changes.

## IV. EXPERIMENTS

We conduct multiple real-world experiments to validate the efficiency and effectiveness of our data generation pipeline. The detailed experimental scenarios and task settings are illustrated in Fig. 5.

We design three manipulation tasks: Pick and Place, Pick Place Close, and Unscrew Bottle Cap. In Pick and Place, the robot picks a target object and places it into a container. In Pick Place Close, the robot additionally closes the container lid after placement. In Unscrew Bottle Cap, the robot unscrews and removes the cap from a jar fixed to the tabletop, involving contact-rich interactions.

Experiments are conducted on a Flexiv Rizon 4s robotic arm with a Xense Aurora Lite gripper under position control, using an overhead RealSense D415 camera for RGB observations. For each task, policies trained on data generated by our pipeline are compared against those trained on teleoperation data.

For the Pick and Place task, objects are randomly placed within a 40 cm Ã 50 cm tabletop workspace. Multiple target objects (e.g., cube, fruits, and plush toy) and containers of different materials are considered. Only the cube-box case is physically collected, while all other object-container combinations are synthesized by replacing 3D Gaussian assets in simulation. The Pick Place Close task follows the same workspace configuration but requires closing the container lid after placement. In the Unscrew Bottle Cap task, the jar is fixed on the table, and the robot unscrews and removes the cap.

Through these experiments, we evaluate ExoGS pipeline efficiency, policy performance, and robustness gains enabled by the proposed augmentation strategies. Across all tasks, we collect 60 raw demonstrations per task using AirExo-3 within our pipeline, and an additional 60 demonstrations via teleoperation for comparison. Data collection time and success rate are reported as evaluation metrics. For policy evaluation, each task is executed over 25 trials, with success rates as the primary performance measure.

## A. Evaluation of Demonstration Data Collection Efficiency

To evaluate data collection efficiency, we recruited 10 volunteers without robotics background, each receiving approximately 10 minutes of training. During data collection, failed attempts were discarded, and only successful demonstrations were used to compute metrics. Each participant recorded six valid demonstrations per task. We report the success ratio and the average time per successful demonstration as evaluation metrics.

<!-- image-->  
Fig. 6. Task completion time comparison between AirExo-3 and teleoperation. Bars show the average over all volunteers, and colored dots denote individual averages, computed using successful trials only.

Fig. 6 compares average data collection time across manipulation tasks using two acquisition methods. Overall, AirExo-3 achieves faster data collection than teleoperation, with comparable performance on simple tasks and increasingly larger advantages as task complexity grows. This benefit arises from more natural manipulation feedback and lower operational burden, and is further reflected by reduced inter-subject variance, indicating improved consistency across users.

TABLE I  
TASK SUCCESS RATIO COMPARISON: AIREXO-3 VS. TELEOPERATION
<table><tr><td>Task</td><td>AirExo-3</td><td>Teleoperation</td></tr><tr><td>Pick and place</td><td>100%</td><td>92.3%</td></tr><tr><td>Pick place close</td><td>100%</td><td>83%</td></tr><tr><td>Unscrew bottle cap</td><td>87%</td><td>17%</td></tr></table>

Table I reports task success ratios for the two data collection methods. Consistent with the timing results, AirExo-3 achieves higher success ratios than teleoperation across all tasks, with the performance gap increasing as task difficulty grows. The advantage is most pronounced in the bottle-cap unscrewing task, where teleoperation requires many failed attempts and often causes unintended collisions and object damage. In contrast, AirExo-3 yields more stable executions and higher success ratios, demonstrating its effectiveness for collecting reliable demonstrations in contact-rich manipulation tasks.

## B. Policy Performance without Data Augmentation

We evaluate policies trained on data generated by our pipeline against those trained on teleoperation-collected data to assess whether the generated data can match the effectiveness of real-world demonstrations. For the Pick and Place task, we further expand the dataset by replacing the manipulated object with different digital assets, yielding a dataset ten times larger than the original. In the Pick and Place (New Object) setting, the objects are exclusively drawn from this augmented set and have never appeared during physical data collection, with demonstrations generated solely through trajectory transfer via digital asset replacement. All experiments in this setting are conducted using the modified ACT policy described in Sec. III-C1.

TABLE II  
SUCCESS RATE COMPARISON BETWEEN EXOGS AND TELEOPERATION
<table><tr><td>Task</td><td>ExoGS</td><td>Teleop</td></tr><tr><td>Pick and place</td><td>50%</td><td>72%</td></tr><tr><td>Pick place close</td><td>48%</td><td>64%</td></tr><tr><td>Unscrew bottle cap</td><td>24%</td><td>8%</td></tr><tr><td>Pick and place (New Object)</td><td>76%</td><td>0%</td></tr></table>

For most manipulation tasks, policies trained on the original generated demonstrations underperform those trained on teleoperation data, primarily due to the remaining visual gap between rendered and real-world observations. In the Unscrew Bottle Cap task, this trend is reversed, likely because teleoperation is particularly difficult, leading to noisy and inconsistent trajectories that hinder effective policy learning.

Moreover, fully synthetic derived demonstrations significantly benefit policy training. By enabling low-cost dataset expansion, these demonstrations allow policies to achieve performance comparable to those trained on real-world data.

## C. Policy Performance with Data Augmentation

Despite lower visual fidelity, the generated data supports flexible augmentation across viewpoints, appearance, backgrounds, and object pose, enabling systematic evaluation of their effects on policy generalization.

<!-- image-->  
Fig. 7. Illustration of the generalization evaluation scenarios. Camera viewpoints, object colors, background appearances, and lighting conditions are varied to assess the generalization capability of the learned policies across different environments.

Using these four augmentation strategies, we obtain a dataset that is twenty times the size of the original one for policy training. We then introduce variations in both the environment and the manipulated objects to assess model performance under generalization settings. The generalization evaluation scenarios are illustrated in Fig. 7.

As shown in Fig. 8 (a), data augmentation substantially improves policy generalization. Policies trained with augmented data consistently outperform those trained on nonaugmented synthetic data and even real-world data, especially under variations in object color, background, and lighting. In these challenging settings, policies trained solely on real-world data often fail completely, highlighting the effectiveness of the proposed augmentation strategies.

In contrast, the improvement is limited for the Unscrew Bottle Cap task, where failures are dominated by strong kinematic constraints from the threaded coupling, such as slippage and jamming, rather than visual perception. These issues mainly arise from suboptimal demonstration quality due to operator proficiency, inherently limiting the benefits of data augmentation.

## D. Ablation Study of Data Augmentation Methods

To evaluate the individual contributions of different data augmentation strategies to policy generalization, we construct three augmented datasets for ablation studies. Since color jitter inherently affects background appearance, it is not considered separately from background replacement. The datasets are defined as follows:

<!-- image-->  
(a)

<!-- image-->  
(b)

Fig. 8. Effect of data augmentation on policy generalization. (a) Success rates under various visual perturbations for policies trained on teleoperation and augmented data. (b) Ablation study using three augmented datasets (A, B, and C) to evaluate the impact of different augmentation strategies on generalization performance.  
<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 9. Example segmentation results of the proposed Mask Adapter. The model is trained using only containers and green blocks, yet achieves robust segmentation when tested on novel objects and backgrounds, demonstrating strong generalization capability.

â¢ Dataset A (Viewpoint): Each demonstration is rendered from ten novel camera viewpoints using 3DGS, resulting in a dataset ten times the size of the original one.

â¢ Dataset B (Appearance): Background textures and lighting conditions are randomized, and large-range color jitter is applied to all Gaussian models, producing a dataset ten times the size of the original one.

â¢ Dataset C (Object Pose): Additional demonstrations are generated by applying random perturbations to the poses of the manipulated objects, yielding a dataset ten times the size of the original one.

Since the performance bottleneck of the Unscrew Bottle Cap task mainly arises from physical constraints rather than visual representation, we conduct the ablation study on the Pick Place Close task.

As shown in Fig. 8 (b), the generalization improvements from data augmentation largely align with intuition. Viewpoint variation and color jitter most effectively expand the training domain and thus contribute the largest performance gains, with color jitter yielding the strongest overall improvement due to persistent color and illumination discrepancies between synthetic and real data.

In contrast, pose augmentation provides only marginal benefits, as object poses are already sufficiently diverse during data collection and pose perturbations do not address the primary visual domain gap, resulting in limited performance gains.

## E. Effect of Mask Adapter on Policy Performance

Mask Adapter is a lightweight module designed to enhance policy generalization by guiding attention toward interactioncentric features and facilitating the modeling of interactionrelevant relationships, thereby mitigating the sim-to-real gap. While sharing a similar goal with data augmentation, conventional augmentation relies on substantially enlarged synthetic datasets, incurring notable storage and computational overhead. In contrast, Mask Adapter operates directly on the original data and remains fully compatible with augmented data, offering a more efficient alternative.

We train ACT policies equipped with Mask Adapter using non-augmented demonstrations generated by our pipeline and evaluate their generalization performance under varying environmental and object conditions.

<!-- image-->  
Fig. 10. Impact of the Mask Adapter on policy generalization. The adapter effectively narrows the sim-to-real gap, enabling policies to outperform teleoperation baselines in standard and color-varied scenarios, while performance remains sensitive to severe background and lighting perturbations.

Fig. 9 illustrates segmentation results produced by the Mask Adapter during evaluation. The adapter enables fully automatic and accurate segmentation of the target objects, the background, and the robot, and is able to generalize to objects that do not appear in the training dataset.

As shown in Fig. 10, the Mask Adapter is highly effective at mitigating the sim-to-real gap introduced by training on synthetic data. Benefiting from the higher-quality motion trajectories collected using AirExo-3, the resulting policies even outperform those trained on teleoperation data. In addition, the policies exhibit a certain degree of robustness to object color variations. However, under more severe conditions, such as drastic background changes and colored lighting that adversely affect the segmentation model, performance still degrades noticeably. Considering that the Mask Adapter is designed as a lightweight module to counter the sim-to-real gap, its performance is fully satisfactory.

## V. CONCLUSION

In this work, we presented ExoGS, a scalable, robot-free Real-to-Sim-to-Real framework for manipulation data collection. By integrating a low-cost (approx. \$400) passive exoskeleton with a 3D Gaussian Splatting pipeline, we enable the efficient acquisition of kinematically consistent demonstrations and their digitization into editable simulation assets. This approach allows for massive, geometry-aware data augmentation that significantly enhances policy generalization. Furthermore, our proposed Mask Adapter effectively bridges the sim-toreal gap by injecting instance-level semantic constraints into the policy. Extensive real-world experiments demonstrate that ExoGS reduces data collection costs while yielding policies that outperform those trained via traditional teleoperation.

However, the current framework relies on the rigid-body assumption of 3DGS, which restricts the modeling of deformable objects with variable geometry. Future work will plan to further improve the automation level of the data generation pipeline.

## REFERENCES

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, 2021.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussianÂ¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., 2023.

[3] A. Byravan, J. Humplik, L. Hasenclever, A. Brussee, F. Nori, T. Haarnoja, B. Moran, S. Bohez, F. Sadeghi, B. Vujatovic et al., âNerf2real: Sim2real transfer of vision-guided bipedal motion skills using neural radiance fields,â in ICRA, 2023.

[4] M. N. Qureshi, S. Garg, F. Yandun, D. Held, G. Kantor, and A. Silwal, âSplatsim: Zero-shot sim2real transfer of rgb manipulation policies using gaussian splatting,â in CoRL 2024 Workshop on Mastering Robot Manipulation in a World of Abundant Data, 2024.

[5] M. T. Villasevil, A. Simeonov, Z. Li, A. Chan, T. Chen, A. Gupta, and P. Agrawal, âReconciling reality through simulation: A real-to-simto-real approach for robust manipulation,â in Robotics: Science and Systems, 2024.

[6] H. Lou, Y. Liu, Y. Pan, Y. Geng, J. Chen, W. Ma, C. Li, L. Wang, H. Feng, L. Shi et al., âRobo-gs: A physics consistent spatial-temporal model for robotic arm with hybrid representation,â CoRR, 2024.

[7] X. Li, J. Li, Z. Zhang, R. Zhang, F. Jia, T. Wang, H. Fan, K.-K. Tseng, and R. Wang, âRobogsim: A real2sim2real robotic gaussian splatting simulator,â arXiv preprint arXiv:2411.11839, 2024.

[8] J. Yu, L. Fu, H. Huang, K. El-Refai, R. A. Ambrus, R. Cheng, M. Z. Irshad, and K. Goldberg, âReal2render2real: Scaling robot data without dynamics simulation or robot hardware,â arXiv preprint arXiv:2505.09601, 2025.

[9] Z. Yuan, T. Wei, S. Cheng, G. Zhang, Y. Chen, and H. Xu, âLearning to manipulate anywhere: A visual generalizable framework for reinforcement learning,â in Conference on Robot Learning. PMLR, 2025, pp. 1815â1833.

[10] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024.

[11] T. Xie, Z. Zong, Y. Qiu, X. Li, Y. Feng, Y. Yang, and C. Jiang, âPhysgaussian: Physics-integrated 3d gaussians for generative dynamics,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[12] O. Shorinwa, J. Tucker, A. Smith, A. Swann, T. Chen, R. Firoozi, M. D. Kennedy, and M. Schwager, âSplat-mover: Multi-stage, openvocabulary robotic manipulation via editable gaussian splatting,â in 8th Annual Conference on Robot Learning, 2024.

[13] M. Ji, R.-Z. Qiu, X. Zou, and X. Wang, âGraspsplats: Efficient manipulation with 3d feature splatting,â in 8th Annual Conference on Robot Learning, 2024.

[14] B. P. Duisterhof, Z. Mandi, Y. Yao, J.-W. Liu, J. Seidenschwarz, M. Z. Shou, R. Deva, S. Song, S. Birchfield, B. Wen, and J. Ichnowski, âDeformGS: Scene flow in highly deformable scenes for deformable object manipulation,â WAFR, 2024.

[15] J. Abou-Chakra, K. Rana, F. Dayoub, and N. Sunderhauf, âPhysically Â¨ embodied gaussian splatting: A realtime correctable world model for robotics,â CoRR, 2024.

[16] S. Yang, W. Yu, J. Zeng, J. Lv, K. Ren, C. Lu, D. Lin, and J. Pang, âNovel demonstration generation with gaussian splatting enables robust one-shot manipulation,â arXiv preprint arXiv:2504.13175, 2025.

[17] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid et al., âRt-2: Vision-language-action models transfer web knowledge to robotic control,â in Conference on Robot Learning, 2023.

[18] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, âBc-z: Zero-shot task generalization with robotic imitation learning,â in Conference on Robot Learning, 2022.

[19] T. Zhang, Z. McCarthy, O. Jow, D. Lee, X. Chen, K. Goldberg, and P. Abbeel, âDeep imitation learning for complex manipulation tasks from virtual reality teleoperation,â in ICRA, 2018.

[20] S. Song, A. Zeng, J. Lee, and T. Funkhouser, âGrasping in the wild: Learning 6dof closed-loop grasping from low-cost demonstrations,â IEEE Robotics and Automation Letters, 2020.

[21] K. Bousmalis, G. Vezzani, D. Rao, C. Devin, A. X. Lee, M. Bauza, T. Davchev, Y. Zhou, A. Gupta, A. Raju et al., âRobocat: A selfimproving foundation agent for robotic manipulation,â arXiv preprint arXiv:2306.11706, vol. 1, no. 8, 2023.

[22] H. Fang, H.-S. Fang, Y. Wang, J. Ren, J. Chen, R. Zhang, W. Wang, and C. Lu, âAirexo: Low-cost exoskeletons for learning whole-arm manipulation in the wild,â in ICRA, 2024.

[23] H. Fang, C. Wang, Y. Wang, J. Chen, S. Xia, J. Lv, Z. He, X. Yi, Y. Guo, X. Zhan et al., âAirexo-2: Scaling up generalizable robotic imitation learning with low-cost exoskeletons,â in 7th Robot Learning Workshop: Towards Robots with Human-Level Abilities, 2025.

[24] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, âUniversal manipulation interface: In-the-wild robot teaching without in-the-wild robots,â in Robotics: Science and Systems, 2024.

[25] C. Wang, H. Shi, W. Wang, R. Zhang, L. Fei-Fei, and K. Liu, âDexcap: Scalable and portable mocap data collection system for dexterous manipulation,â in RSS 2024 Workshop: Data Generation for Robotics, 2024.

[26] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, âLearning fine-grained bimanual manipulation with low-cost hardware,â in ICML Workshop on New Frontiers in Learning, Control, and Dynamical Systems, 2023.

[27] Q. Ben, F. Jia, J. Zeng, J. Dong, D. Lin, and J. Pang, âHomie: Humanoid loco-manipulation with isomorphic exoskeleton cockpit,â in RSS 2025 Workshop on Whole-body Control and Bimanual Manipulation: Applications in Humanoids and Beyond, 2025.

[28] J. L. Schonberger and J.-M. Frahm, âStructure-from-Motion Revisited,âÂ¨ in CVPR, 2016.

[29] B. Wen, W. Yang, J. Kautz, and S. Birchfield, âFoundationpose: Unified 6d pose estimation and tracking of novel objects,â in CVPR, 2024.

[30] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, Â´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., âDinov3,â arXiv preprint arXiv:2508.10104, 2025.

[31] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., âLora: Low-rank adaptation of large language models.â ICLR, 2022.

[32] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, âEncoderdecoder with atrous separable convolution for semantic image segmentation,â in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 801â818.