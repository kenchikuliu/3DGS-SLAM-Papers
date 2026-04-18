# RoboArmGS: High-Quality Robotic Arm Splatting via Bezier Curve Refinement Â´

Hao Wang \* 1 Xiaobao Wei \* 1 Ying Li 1 Qingpo Wuwu 1 Dongli Wu 1 Jiajun Cao 1 Ming Lu 1 Wenzhao Zheng 2 Shanghang Zhang 1

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 1. The illustration of RoboArmGS and RoboArm4D. We refine URDF-rigged motion with learnable Bezier curves that correct Â´ per-joint residuals, bridging the gap between idealized kinematics and noisy real-world dynamics. This enables accurate motion modeling and coherent 3D Gaussian binding across arm parts. We evaluate on RoboArm4D, our carefully collected dataset of widely used robotic arms, achieving state-of-the-art rendering quality.

## Abstract

Constructing photorealistic and controllable robotic arm digital assets from real observations is fundamental to robotic applications. Current approaches naively bind static 3D Gaussians according to URDF links, forcing them to follow an URDF-rigged motion passively. However, the idealized URDF-rigged motion cannot accurately model the actual motion captured in real-world observations, leading to severe rendering artifacts in 3D Gaussians. To address these challenges, we propose RoboArmGS, a novel hybrid representation that refines the URDF-rigged motion with learnable Bezier curves, enabling more accu-Â´ rate real-world motion modeling. To be more

specific, we present a learnable Bezier Curve Â´ motion refiner that corrects per-joint residuals to address mismatches between real-world motion and URDF-rigged motion. RoboArmGS enables the learning of more accurate real-world motion while achieving a coherent binding of 3D Gaussians across arm parts. To support future research, we contribute a carefully collected dataset named RoboArm4D, which comprises several widely used robotic arms for evaluating the quality of building high-quality digital assets. We evaluate our approach on RoboArm4D, and RoboArmGS achieves state-of-the-art performance in realworld motion modeling and rendering quality. The code and dataset will be released.

## 1. Introduction

Robotic arms play a crucial role as the primary executors of tasks in contemporary automation and intelligent systems, making them a significant area of focus in the fields of robotics (Liang et al., 2024; Ma et al., 2024; Sun et al., 2025; Jiang et al., 2024). Reconstructing high-fidelity, interactable digital assets of robotic arms from real-world observations is a foundational step for creating robust simulation platforms, which is critical for applications in robot control, policy learning, and system monitoring (Wang et al., 2025a; Pfaff et al., 2025; Abou-Chakra et al., 2025). By generating these digital assets from video inputs, we can build simulation environments that faithfully mirror their real-world counterparts. This capability is essential for bridging the sim-to-real gap, yet is fundamentally hindered by the discrepancy between idealized kinematic models and physical execution. While standard calibration techniques effectively rectify static geometric errors, they often fall short in capturing the complex dynamic deviations inherent in real-world motion. Consequently, the challenge of automatically generating dynamic, high-fidelity assets from video remains a key bottleneck in robotics (Xie et al., 2025; Jiang et al., 2025).

Current approaches for building dynamic digital assets of robotic arms typically involve complex, multi-stage pipelines, which hinder their scalability and robustness (Lou et al., 2025; Han et al., 2025; Li et al., 2024a). These pipelines usually begin with a strictly controlled data capture phase, requiring multi-view images or videos of the scene. Subsequently, the robotic arm requires precise segmentation to isolate it from the background and distinguish its individual articulated parts, a process that necessitates extensive labor-intensive manual annotation (Yang et al., 2025; Lou et al., 2025). Such requirements not only limit the methodâs applicability outside laboratory settings but also introduce significant fragility. Crucially, this paradigm often assumes a perfect correspondence between the kinematic model and visual observations, thereby failing to account for dynamic discrepancies caused by real-world factors like control latency or joint friction.

In contrast, learning directly from a single, casually captured monocular video offers a more practical and scalable paradigm (Li et al., 2025a; Tao et al., 2025). While this approach drastically lowers the data collection barrier, it significantly intensifies the challenge of aligning the rigid kinematic prior with the actual, non-ideal motion observed in the footage. Addressing this requires accurately modeling these dynamic deviations to reconcile them with the standard Universal Robot Description Format (URDF), a step vital for enabling high-fidelity, motion-accurate simulation.

To address these challenges, we propose RoboArmGS (Fig. 1), a novel hybrid representation designed to harmonize the rigid kinematic prior with actual visual observations. Our approach dynamically refines the idealized URDF-driven motion using a learnable Bezier curve model, Â´ enabling the creation of digital assets that are temporally coherent and precisely aligned with real-world footage. Central to our approach are two key insights: (1) Instead of treating 3D Gaussians as an unstructured collection, we introduce a structured binding mechanism that anchors them to a geometric prior, specifically the robotâs mesh, to fundamentally enforce topological consistency during motion. (2) The complex, time-varying discrepancies between the idealized URDF model and the physical robotâs dynamics can be explicitly captured and corrected using a flexible, smooth parameterization.

Guided by these insights, our architecture integrates two core modules. First, the Structured Gaussian Binding (SGB) anchors each Gaussian to a face on the robotâs mesh within its local coordinate frame. This ensures that the model inherits the robotâs rigid geometric structure while retaining the flexibility to capture intricate appearances via learnable offsets. Second, the Bezier-based Motion Refiner (BMR) Â´ addresses dynamic discrepancies by acting as a lightweight motion residual field. It leverages URDF kinematics as a strong motion prior and learns a smooth, continuous-time residual transformation to reconcile the rigid model with actual observations, thereby compensating for systematic errors in real-world motion. By synergizing these components, RoboArmGS produces digital assets that are both structurally consistent and kinematically accurate, significantly reducing rendering artifacts caused by motion mismatch.

To validate our approach and support future research, we present RoboArm4D, a specialized dataset featuring representative robotic arms. Designed as a rigorous benchmark for high-fidelity asset creation, it specifically targets the evaluation of how well the generated digital assets conform to real-world visual dynamics.

Our main contributions are as follows:

â¢ We propose RoboArmGS, which synergizes Structured Gaussian Binding (SGB) and Bezier-based Motion Re-Â´ finer (BMR) to enforce geometric consistency while correcting dynamic discrepancies, thereby reconciling rigid kinematic priors with actual observations.

â¢ We contribute RoboArm4D, a benchmark tailored for robotic arm digitization. Featuring representative mainstream robotic arms with distinct morphologies, it evaluates rendering fidelity and motion accuracy against physical reality.

â¢ Extensive experiments on RoboArm4D demonstrate that RoboArmGS outperforms state-of-the-art methods, setting a new standard for photorealistic and physically aligned simulation.

## 2. Related Work

Dynamic Scene Modeling with Gaussian Splatting. 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) has emerged as a dominant representation for high-fidelity scene modeling (Wei et al., 2025c;a; Wang et al., 2025d). To extend this to dynamic environments, Deformable 3DGS (Yang et al., 2024) introduces deformation fields to capture monocular temporal variations, while others (Zhang et al., 2024; Wei et al., 2025b) leverage graph neural networks to learn object dynamics under physical interactions. In the realm of articulated avatars, mesh-guided approaches have shown significant promise. For instance, GaussianAvatars (Qian et al., 2024; Chen et al., 2025) bind 3D Gaussians to the FLAME mesh for precise facial control, and Animatable Gaussians (Li et al., 2024b) apply similar binding strategies to SMPL-based bodies. These methods demonstrate the efficacy of anchoring Gaussians to a geometric prior, a principle we adapt for robotic structures. In urban scene reconstruction, methods like S3Gaussian (Huang et al., 2024b), StreetGaussians (Yan et al., 2024), and HUGS (Zhou et al., 2024) decompose dynamic entities (e.g., vehicles) from static backgrounds to achieve high-quality rendering and editing. More recently, BezierGS ( Â´ Ma et al., 2025) utilizes learnable Bezier curves to parameterize the global motion Â´ trajectories of vehicles, enforcing geometric consistency via inter-group losses. While BezierGS employs curves to Â´ model the absolute motion path, our approach fundamentally differs by using Bezier curves to model the residual Â´ deviation from a kinematic prior. Specifically, we leverage the curves to refine an idealized forward kinematics chain, correcting the dynamic mismatch between the theoretical model and real-world observations.

Robotic Synthesis using 3D Reconstruction. Highfidelity 3D reconstruction is pivotal for robotic policy learning and data synthesis (Lu et al., 2024; Yu et al., 2025b; Chai et al., 2025; Wang et al., 2025b). Recent approaches have integrated 3DGS to bridge the gap between simulation and reality. Several methods focus on integrating 3DGS into existing simulators: SplatSim (Qureshi et al., 2025) replaces mesh rendering with 3DGS for photorealistic output, while Real2Render2Real (Yu et al., 2025a) leverages scanned objects and human demonstrations to scale up training data generation in IsaacLab. However, for asset creation, most existing pipelines rely on complex multi-view setups. For instance, RoboGSim (Li et al., 2024a) and REÂ³SIM (Han et al., 2025) reconstruct scene assets from multi-view videos to build digital twins. Similarly, RoboSplat (Yang et al., 2025) uses a unified Gaussian representation from multiview inputs to model the entire workspace. While effective, these multi-stage pipelines often suffer from scalability issues. On the other hand, monocular approaches like ManipDreamer3D (Li et al., 2025c;b) synthesize videos from single-view occupancy but focus less on precise kinematic control. Most relevant to our work is Robo-GS (Lou et al., 2025), which binds Gaussians to meshes to reconstruct robotic arms from monocular video. However, Robo-GS primarily targets static asset generation and relies on precise panoramic annotations. In contrast, our RoboArmGS not only utilizes monocular input without such heavy supervision but also explicitly models the dynamic discrepancies between the kinematic prior and real-world motion, ensuring both visual and kinematic fidelity during execution.

## 3. Methodology

In this section, we present RoboArmGS as shown in Fig. 2, a framework designed to bridge the sim-to-real motion discrepancy in high-fidelity robotic asset creation. Instead of relying solely on rigid kinematic assumptions, we introduce a hybrid representation that harmonizes explicit geometric priors with learnable dynamic refinements. Specifically, Structured Gaussian Binding (SGB) enforces topological consistency by anchoring Gaussians to the robotâs mesh, while the Bezier-based Motion Refiner (BMR) explicitly Â´ captures continuous residuals between idealized Foward Kinematic (FK) models and real-world observations. Finally, we detail the optimization and regularization strategies that ensure robust training from monocular video.

## 3.1. Preliminaries

3D Gaussian Splatting. 3D Gaussian Splatting (Kerbl et al., 2023) is a real-time radiance field method that represents a scene as a collection of unstructured 3D Gaussians. Each Gaussian is parameterized by a position $\mu ,$ rotation q, anisotropic scaling s, opacity Î±, and Spherical Harmonics for view-dependent color. During optimization, a valid positive semi-definite covariance matrix Î£ is constructed from the learnable scaling and rotation components. To render an image, these 3D Gaussians are projected onto the 2D image plane, sorted by depth, and then composited to compute the final pixel color $C$ via alpha blending:

$$
C = \sum _ { i \in N } c _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{1}
$$

where $c _ { i }$ is the color of the i-th Gaussian and $\alpha _ { i } ^ { \prime }$ is its projected opacity. This entire process is made differentiable and efficient by a dedicated tile-based rasterizer.

Bezier Curves. Â´ A Bezier curve ( Â´ Mortenson, 1999) is a smooth, differentiable parametric curve defined by a set of $n + 1$ control points $\{ P _ { i } \} _ { i = 0 } ^ { n }$ . A point on the curve is computed as a weighted sum of these control points using Bernstein basis polynomials as weights:

$$
B ( t ) = \sum _ { i = 0 } ^ { n } B _ { i , n } ( t ) P _ { i } , \quad t \in [ 0 , 1 ]\tag{2}
$$

<!-- image-->  
Figure 2. Overview of RoboArmGS. (a) SGB anchors Gaussians to URDF links with binding-aware densification for structural consistency. (b) BMR refines kinematic discrepancies via learnable Bezier curves. Jointly optimized from monocular input, our method Â´ enables photorealistic rendering for novel robot poses and viewpoints.

where $\begin{array} { r } { B _ { i , n } ( t ) = { \binom { n } { i } } t ^ { i } ( 1 - t ) ^ { n - i } } \end{array}$ . As t varies from 0 to 1, the curve provides a compact and continuous parameterization that smoothly interpolates within the convex hull of its control points. This makes it an effective tool for modeling temporally coherent trajectories and motion residuals.

## 3.2. Structured Gaussian Binding

Unlike existing methods that naively bind static 3D Gaussians to URDF links, our Structured Gaussian Binding (SGB) is designed to preserve the piecewise rigid structure of robotic arms. SGB explicitly binds each Gaussian to a specific face on the robotâs mesh, driven by physics-aware local parameterization and binding-aware adaptive densification. This mesh-level binding ensures all Gaussians on a single link undergo a collective rigid transformation, guaranteeing structural coherence and high-fidelity representation of the articulated motion.

Physics-driven Local Parameterization. While inspired by recent advances in rigged avatars (Qian et al., 2024; Xu et al., 2024), we introduce a critical adaptation for rigid-body robotic systems. Unlike Linear Blend Skinning (LBS) (Li et al., 2017) models that often introduce non-rigid artifacts, we drive the motion using the robotâs forward kinematics to strictly maintain piecewise rigidity. Specifically, we bind each 3D Gaussian $j$ to a specific face i on the kinematic mesh. This allows us to decompose the Gaussian properties into dynamic, pose-dependent states and learnable, static attributes. The dynamic properties consist of the local origin $T _ { i } ( t )$ and orientation $R _ { i } ( t )$ , which are determined by the link poses obtained from the MuJoCo engine (Todorov et al., 2012) at time t. The static properties are defined as learnable local offsets that include position $\mu _ { j }$ , rotation $r _ { j } ,$ scale $s _ { j }$ color, and opacity. The final world-space state of a Gaussian for rendering is computed as:

$$
\mu _ { j } ^ { \prime } ( t ) = R _ { i } ( t ) \mu _ { j } + T _ { i } ( t )\tag{3}
$$

$$
r _ { j } ^ { \prime } ( t ) = R _ { i } ( t ) r _ { j }\tag{4}
$$

$$
s _ { j } ^ { \prime } ( t ) = s _ { j }\tag{5}
$$

where $\mu _ { j } ^ { \prime } ( t ) , r _ { j } ^ { \prime } ( t )$ , and $s _ { j } ^ { \prime } ( t )$ denote the world-space parameters. This ensures that all Gaussians anchored to the same link inherit a collective rigid transformation, while the learnable offsets $\mu _ { j } , r _ { j }$ provide the degrees of freedom necessary to reconstruct complex surface geometries that exceed the resolution of the proxy URDF mesh.

Binding-Aware Adaptive Densification. To capture highfrequency surface details while respecting the robotâs rigid structure, we adapt the density control mechanism of 3DGS (Kerbl et al., 2023) through two binding-aware modifications. First, we implement binding inheritance, ensuring that newly densified primitives automatically inherit the face index of their progenitor. This anchoring mechanism preserves piecewise rigid motion by preventing Gaussians from drifting away from their designated kinematic links during splitting or cloning. Second, we introduce a structural preservation constraint that prevents the pruning of the final Gaussian associated with any mesh face. This safeguard is critical for maintaining the geometric integrity of the robotic structure, effectively eliminating visual âholesâ that might otherwise emerge during large-scale articulated movements. Together, these adaptations ensure that the optimization process yields a high-fidelity representation that strictly adheres to the robotâs physical configuration and topological constraints.

## 3.3. Bezier-based Motion RefinerÂ´

To reconcile idealized kinematic models with physical reality, we introduce the Bezier-based Motion Refiner (BMR). Â´ While standard FK provides a strong motion prior, they often fail to account for real-world deviations such as control latency or joint friction. Drawing inspiration from the smooth temporal parameterization in BezierGS ( Â´ Ma et al., 2025), our BMR models these discrepancies as a continuous residual field. We employ a hierarchical decomposition that couples a learnable, time-varying global correction with static, per-joint offsets. This design leverages the inherent smoothness of Bezier curves to ensure global temporal co-Â´ herence while providing the local flexibility necessary to compensate for systematic kinematic errors.

Learnable Bezier Correction. Â´ The time-varying global correction, $\mathbf { T } _ { \mathrm { B e z i e r } } ( t )$ , is modeled by a learnable Bezier Â´ curve that operates in a 9-dimensional parameter space. Unlike BezierGS ( Â´ Ma et al., 2025), which uses Bezier curves Â´ to represent the full motion trajectory of objects, we employ the curve here to model the residual SE(3) deviation from the rigid FK prior. We parameterize the residual transformations as 9D vectors $\pmb { \delta } = [ \Delta \mathbf { x } , \Delta \mathbf { r } ]$ , consisting of a 3D translation $\Delta \mathbf { x }$ and a continuous 6D rotation representation $\Delta \mathbf { r }$ (Zhou et al., 2019). The Bezier curve, defined by Â´ $K + 1$ learnable control points $\{ \mathbf { p } _ { k } \} _ { k = 0 } ^ { K } \subset \mathbb { R } ^ { 9 }$ , outputs a 9D residual vector for any time t:

$$
\delta _ { \mathrm { B \acute { e } z i e r } } ( t ) = \omega \sum _ { k = 0 } ^ { K } B _ { k } ^ { K } ( t ) \cdot { \bf p } _ { k }\tag{6}
$$

where $B _ { k } ^ { K } ( t )$ are the Bernstein basis polynomials. We use a high-order curve $( K = 1 9 )$ to capture complex motion patterns and scale the output by an influence factor $\omega$ to ensure it acts as a refinement. The resulting 9D vector is then converted to a valid SE(3) matrix $\mathbf { T } _ { \mathrm { B e z i e r } } ( t )$ via Gram-Schmidt orthogonalization for the rotational part.

Per-Joint Static Correction. To address the articulated structure of the robot, we introduce a static, per-joint offset $\mathbf { T } _ { \mathrm { e m b e d } } ^ { ( k ) } ,$ , represented by a learnable embedding $\mathbf { e } ^ { ( k ) } \in \mathbb { R } ^ { 9 }$ for each moving joint k. These time-invariant embeddings are optimized to correct for intrinsic structural discrepancies, such as link length errors or joint calibration drift. By integrating these local offsets into the kinematic chain alongside the global Bezier correction, our model specifi-Â´ cally accounts for the hierarchical dependencies of robotic motionâa feature that distinguishes our approach from general, unstructured trajectory modeling.

Final Pose Composition. The final refined pose for each joint is obtained by integrating the nominal FK pose from MuJoCo with our two hierarchical corrections. The composition order is designed to handle errors at different levels of the kinematic chain: the global Bezier correction first com- Â´ pensates for temporal drifts or base misalignments, while the static embedding further refines the local joint-to-link transformation. For a specific joint $k ,$ the final world-space transformation is formulated as

$$
{ \bf T } _ { \mathrm { f i n a l } } ^ { ( k ) } ( t ) = { \bf T } _ { \mathrm { F K } } ^ { ( g ) } ( t ) \circ { \bf T } _ { \mathrm { B \acute { e } z i e r } } ( t ) \circ { \bf T } _ { \mathrm { e m b e d } } ^ { ( k ) }\tag{7}
$$

where $\mathbf { T } _ { \mathrm { F K } } ^ { \left( g \right) } ( t )$ denotes the global pose derived from the URDF model. This unified formulation ensures that the digital asset remains kinematically sound while precisely aligning with the visual observations in the video.

## 3.4. Optimization and Regularization

The overall optimization of our model is supervised by a combination of a primary rendering loss and several crucial regularization terms designed to ensure geometric stability and temporal coherence.

Rendering Loss. The primary supervision signal is a rendering loss comparing the rendered images with ground truth frames. Following standard practice (Kerbl et al., 2023), we use a combination of an L1 photometric loss and a D-SSIM term:

$$
\mathcal { L } _ { \mathrm { r g b } } = ( 1 - \lambda ) \mathcal { L } _ { 1 } + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } \quad ( \lambda = 0 . 2 )\tag{8}
$$

Geometric Regularization. To enforce the topological constraints of SGB module and ensure Gaussians remain well-aligned with their parent mesh faces, we adopt two regularization terms inspired by (Qian et al., 2024). We penalize the local position offset $\mu _ { j }$ and scale $s _ { j }$ if they exceed predefined thresholds, preventing primitives from detaching from the rigid structure or exhibiting visual jitter:

$$
\mathcal { L } _ { \mathrm { p o s } } = \| \operatorname* { m a x } ( | \mu _ { j } | - \epsilon _ { \mathrm { p o s } } , 0 ) \| _ { 2 } ^ { 2 }
$$

$$
\mathcal { L } _ { \mathrm { s c a l e } } = \| \operatorname* { m a x } ( | s _ { j } | - \epsilon _ { \mathrm { s c a l e } } , 0 ) \| _ { 2 } ^ { 2 }\tag{9}
$$

(10)

Bezier Velocity Regularization. Â´ To enforce that the corrective motions learned by our BMR module are temporally smooth, we regularize the squared L2 norm of the learnable Bezier curveâs instantaneous velocity, Â´ v(t). We approximate this velocity using a numerically stable central difference scheme, and define the loss as:

$$
\mathcal { L } _ { \mathrm { v e l } } = \Vert \mathbf { v } ( t ) \Vert _ { 2 } ^ { 2 }\tag{11}
$$

This loss penalizes abrupt changes in the learned 9D residual parameters, encouraging a smooth motion trajectory.

Total Objective. Our final training objective is a weighted sum of the aforementioned losses:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { p o s } } { \mathcal { L } } _ { \mathrm { p o s } } + \lambda _ { \mathrm { s c a l e } } { \mathcal { L } } _ { \mathrm { s c a l e } } + \lambda _ { \mathrm { v e l } } { \mathcal { L } } _ { \mathrm { v e l } }\tag{12}
$$

All learnable parameters, including Gaussian attributes, Bezier control points, and per-joint embeddings, are jointly Â´ optimized by minimizing this objective.

## 4. Experiments

## 4.1. Setup

Settings. We evaluate our method, RoboArmGS, across two challenging settings to assess its reconstruction and motion modeling capabilities. The first setting is Novel-View Synthesis, where the model is trained on multi-view images of a stationary arm and is evaluated on its ability to render photorealistic images from unseen camera viewpoints. The second, more demanding setting is Novel-Pose Synthesis. In this scenario, the model is trained on a video sequence of the arm in motion and is then required to render it from a fixed camera viewpoint, but for a sequence of held-out, unseen joint configurations. Due to space limitations, please refer to the appendix for more experiments.

Datasets. Existing public robotics datasets often lack the precise alignment between visual observations and kinematic states required for high-fidelity asset creation. To facilitate research in this area, we present RoboArm4D, a benchmark designed for the generation and evaluation of dynamic digital assets. This dataset features several widely-used industrial arms, including the Franka Research 3, UR-5e, and ABB IRB 120. For each platform, we provide synchronized data packages comprising monocular video sequences, calibrated camera parameters, time-stamped joint trajectories, and the corresponding URDF files. This collection is intended to support the development of methods that reconcile rigid kinematic priors with actual physical motion. Further details regarding the data collection and package structure are available in the appendix.

Metrics. To quantitatively evaluate rendering and motion fidelity, we adopt three standard image-based quality metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). These metrics jointly assess both pixel-level accuracy and perceptual quality of rendered views, providing a comprehensive evaluation of real-world motion consistency and rendering fidelity across static and dynamic robotic scenarios.

Implementation Details. Our model, RoboArmGS, is implemented in PyTorch and trained for 600,000 iterations on a single NVIDIA H100 GPU using the Adam optimizer. For 3D Gaussian attributes, we follow the standard learning rate schedule from 3DGS (Kerbl et al., 2023), with position learning rates exponentially decayed to 1% of initial values. The BMR parameters are optimized with a learning rate of 0.0015, weight decay of 0.0001, and influence factor $\omega = 0 . 1$ . Gaussians are initialized by uniform sampling on the kinematic mesh surface, with binding-aware adaptive densification activated every 100 iterations from iteration 500 to 60,000; unstable Gaussian opacities are reset every 3,000 iterations to prune floaters. FK is computed via MuJoCo (Todorov et al., 2012) engine, and regularization hyperparameters are set as $\lambda _ { \mathrm { p o s } } = 0 . 0 1 , \epsilon _ { \mathrm { p o s } } = 1 . 0$ $\lambda _ { \mathrm { s c a l e } } = 1 . 0 , \epsilon _ { \mathrm { s c a l e } } = 0 . 6 ,$ and $\lambda _ { \mathrm { v e l } } = 0 . 0 0 1$

## 4.2. Main Results

Novel-View Synthesis. We first evaluate static 3D reconstruction fidelity through the novel-view synthesis task. As reported in Tab. 1, RoboArmGS achieves state-of-the-art results across all metrics and significantly outperforms other methods. Although Robo-GS (Lou et al., 2025) is a robotspecific method, it still relies on the vanilla 3D Gaussian model for rendering, which performs unconstrained spatial optimization. In contrast, our SGB module explicitly anchors Gaussians to the mesh surface to incorporate the robotâs kinematic structure as a strong geometric prior. This structured approach prevents the formation of floating artifacts and maintains strict structural coherence, providing a more robust foundation for dynamic modeling. Qualitative comparisons in Fig. 3 further demonstrate that our method reconstructs fine-grained surface details with higher visual fidelity than these unconstrained baselines.

Novel-Pose Synthesis. We evaluate dynamic modeling capability through the novel-pose synthesis task, which involves rendering the robot in unseen joint configurations. As shown in Tab. 1, RoboArmGS substantially outperforms all baselines across all metrics. Generic 4D reconstruction methods (Wu et al., 2024; Yang et al., 2024) model temporal evolution using time indices rather than kinematic parameters. Consequently, they lack the ability to render from specified joint angles, a capability that is essential for digital assets applications. Even on held-out test frames, these generic deformation models fail to capture complex articulated motions and produce significant artifacts. To evaluate kinematic-driven approaches, we implement a 3DGS + FK baseline. This baseline binds 3D Gaussians to the mesh and drives their motion via nominal forward kinematics without any motion refinement. While Robo-GS (Lou et al., 2025) shares a similar philosophy, its reliance on manual part segmentation and intensive parameter tuning limits its scalability. Therefore, we utilize 3DGS + FK as the representative kinematic baseline. Although this approach maintains the robotâs rigid structure, it produces significant rendering errors, particularly in color and texture fidelity. These discrepancies arise because the nominal URDF poses deviate from the physical reality in the video, causing the rendered Gaussians to misalign with the actual image content. Our Bezier-based Motion Refiner addresses this by Â´ integrating the refinement of robotic motion directly into the 3D Gaussian optimization loop. By jointly optimizing the Bezier parameters alongside Gaussian attributes, our model Â´ reconciles nominal kinematics with actual visual dynamics through temporally smooth corrections. As demonstrated in Fig. 3, RoboArmGS produces sharp, structurally coherent images that align precisely with the ground truth, validating its accuracy as a controllable digital asset.

<table><tr><td colspan="4">Novel-View Synthesis</td><td></td><td colspan="4">Novel-Pose Synthesis</td></tr><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td></td><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Robo-GS (Lou et al., 2025)</td><td>26.522</td><td>0.918</td><td>0.099</td><td></td><td>4DGS (Wu et al., 2024)</td><td>17.812</td><td>0.867</td><td>0.128</td></tr><tr><td>2DGS (Huang et al., 2024a)</td><td>26.798</td><td>0.920</td><td>0.101</td><td></td><td>Deformable 3DGS (Yang et al., 2024)</td><td>18.844</td><td>0.880</td><td>0.119</td></tr><tr><td>SuGaR (Guedon &amp; Lepetit, 2024)</td><td>26.969</td><td>0.929</td><td>0.085</td><td></td><td>3DGS + FK</td><td>21.038</td><td>0.901</td><td>0.095</td></tr><tr><td>Ours</td><td>28.669</td><td>0.938</td><td>0.065</td><td></td><td>Ours</td><td>31.704</td><td>0.967</td><td>0.039</td></tr></table>

Table 1. Quantitative comparison. Our model demonstrates superior performance in both static Novel-View Synthesis (left) and dynamic Novel-Pose Synthesis (right). Green indicates the best and yellow indicates the second-best performance.

<!-- image-->  
Figure 3. Qualitative comparison on novel-view and novel-pose synthesis. RoboArmGS achieves photorealistic rendering with precise geometric alignment across both tasks. While baseline methods suffer from severe blurring, structural distortion, or rendering artifacts under unseen configurations, our approach maintains high fidelity and kinematic accuracy, ensuring sharp and coherent results.

## 4.3. Ablation Study

Module Effectiveness Analysis. We validate the contribution of each proposed component through comprehensive ablation studies, as presented in Tab. 2. For the novel-view synthesis task, the removal of the SGB module leads to a severe degradation in reconstruction quality. Further ablating its sub-components, such as adaptive densification and binding mechanisms, confirms their necessity for achieving high-fidelity static representation. Regarding the novel-pose synthesis task, the BMR module plays a critical role in reconciling kinematic discrepancies. As illustrated in Fig. 4, the impact of BMR is highly motion-dependent. In regions with minimal movement, such as the robot base (green box), the differences between the full model and the ablated variant are negligible. Conversely, in regions experiencing large-scale articulation, such as the gripper (red box), the absence of BMR results in significant visual artifacts and misalignments. These results demonstrate that the learnable Bezier correction for global dynamics and the per-joint Â´ static offsets for local errors are complementary. This study empirically validates that all components of SGB and BMR are indispensable for the photorealistic and kinematically accurate performance of RoboArmGS.

Bezier Influence Factor Analysis.Â´ We examine the sensitivity of the influence factor Ï, which modulates the learnable Bezier correction. The results for the novel-pose syn-Â´ thesis task are summarized in Tab. 3. Setting Ï = 0.0 effectively disables the global motion refinement, which yields the lowest performance and confirms that nominal kinematics alone are insufficient for precise physical alignment. Even a minimal correction factor of Ï = 0.01 leads to a substantial improvement across all metrics. The performance peaks at $\omega = 0 . 1$ , providing the optimal degree of refinement for aligning the digital asset with visual observations. However, further increasing Ï to 1.0 results in a slight performance decline, suggesting that an excessively large correction magnitude may introduce training instability or overfitting to specific motion trajectories. Consequently, we select $\omega = 0 . 1$ as the standard setting to ensure a robust balance between motion accuracy and stability.

<table><tr><td colspan="4">Novel-View Synthesis Ablation</td><td rowspan="2"></td><td colspan="4">Novel-Pose Synthesis Ablation</td></tr><tr><td>Model Variant</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Model Variant</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td rowspan="4">w/o SGB - w/o Adaptive Densification - w/o Binding</td><td>20.452</td><td>0.809</td><td>0.438</td><td rowspan="4"></td><td>w/o BMR</td><td>29.565</td><td>0.957</td><td>0.052</td></tr><tr><td>20.414</td><td>0.803</td><td>0.451</td><td>- w/o BÃ©zier Correction</td><td>29.384</td><td>0.956</td><td>0.052</td></tr><tr><td>21.621</td><td>0.831</td><td>0.205</td><td>- w/o Per-Joint Offset</td><td>29.418</td><td>0.956</td><td>0.053</td></tr><tr><td>28.669</td><td>0.938</td><td>0.065</td><td>Ours (Full Model)</td><td>31.704</td><td>0.967</td><td>0.039</td></tr></table>

Table 2. Quantitative comparison of module effectiveness analysis. We validate the contributions of our key components on both tasks. The SGB module and its sub-components are crucial for high-quality static reconstruction. The BMR and its sub-components are essential for correcting kinematic errors in novel-pose synthesis. Green indicates the best and yellow indicates the second-best performance.

<table><tr><td>Influence Factor (Ï)</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Ï = 0.0</td><td>29.384</td><td>0.956</td><td>0.052</td></tr><tr><td>Ï = 0.01</td><td>30.541</td><td>0.962</td><td>0.045</td></tr><tr><td>Ï = 0.1 (Ours)</td><td>31.704</td><td>0.967</td><td>0.039</td></tr><tr><td>Ï = 1.0</td><td>31.285</td><td>0.965</td><td>0.042</td></tr></table>

Table 3. Ablation on Influence Factor Ï. We analyze the sensitivity of the learnable Bezier correction on the Novel-Pose Synthesis Â´ task. A value of Ï = 0.1 provides the best balance between effective motion correction and stability. Green indicates the best and yellow indicates the second-best performance.

<!-- image-->  
Figure 4. Ablation study visualization. Low-motion regions (green) show consistent results, whereas highly articulated components (red) exhibit artifacts without BMR, underscoring its necessity for complex motion alignment.

## 4.4. Digital Assets Visualization

We visualize the interactive testing pipeline of our digital assets in Fig. 5. At test time, users can control the robotic arm in real-time by specifying arbitrary joint angles. The underlying URDF-rigged mesh deforms according to the input kinematics, which are then dynamically refined by our learned Bezier curves to match real-world motion. This re-Â´ fined mesh deformation drives the coherent transformation of all bound 3D Gaussians. These Gaussians are subsequently rendered into photorealistic images from any novel viewpoint, faithfully reflecting the specified pose. By seamlessly integrating a controllable kinematic structure with a high-fidelity neural representation, our digital assets achieve both precise kinematic control and superior rendering quality, enabling flexible and realistic simulation for various downstream applications.

<!-- image-->  
Figure 5. Digital Asset Testing Pipeline. Our trained digital assets enable controllable rendering at test time. Given target joint angles, the URDF-rigged mesh deforms accordingly (left to middle), driving coherent Gaussian transformation via our learned Bezier refinement. The deformed Gaussians render photorealistic Â´ images (right) with precise kinematic control.

## 5. Conclusion

In this paper, we present RoboArmGS, a hybrid representation that synergizes Structured Gaussian Binding (SGB) with a Bezier-based Motion Refiner (BMR) to achieve both Â´ geometric consistency and precise motion modeling. By anchoring Gaussians to the robotâs kinematic structure and explicitly modeling physical motion residuals, our method effectively reconciles the discrepancy between nominal URDF models and real-world observations. To support future research, we contribute RoboArm4D, a specialized benchmark designed for the reconstruction and evaluation of dynamic robotic digital assets. Extensive experiments demonstrate that RoboArmGS achieves state-of-the-art performance in both novel-view and novel-pose synthesis, providing a robust and scalable solution for building photorealistic, kinematically accurate digital assets.

## References

Abou-Chakra, J., Sun, L., Rana, K., May, B., Schmeckpeper, K., Vittoria Minniti, M., and Herlant, L. Real-is-sim: Bridging the sim-to-real gap with a dynamic digital twin for real-world robot policy evaluation. arXiv e-prints, pp. arXivâ2504, 2025.

Chai, Y., Deng, L., Shao, R., Zhang, J., Xing, L., Zhang, H., and Liu, Y. Gaf: Gaussian action field as a dvnamic world model for robotic mlanipulation. arXiv preprint arXiv:2506.14135, 2025.

Chen, P., Wei, X., Wuwu, Q., Wang, X., Xiao, X., and Lu, M. Mixedgaussianavatar: Realistically and geometrically accurate head avatar via mixed 2d-3d gaussians. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 945â954, 2025.

Guedon, A. and Lepetit, V. Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and highquality mesh rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5354â5363, 2024.

Han, X., Liu, M., Chen, Y., Yu, J., Lyu, X., Tian, Y., Wang, B., Zhang, W., and Pang, J. Re3sim: Generating high-fidelity simulation data via 3d-photorealistic real-to-sim for robotic manipulation. arXiv preprint arXiv:2502.08645, 2025.

Huang, B., Yu, Z., Chen, A., Geiger, A., and Gao, S. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pp. 1â11, 2024a.

Huang, N., Wei, X., Zheng, W., An, P., Lu, M., Zhan, W., Tomizuka, M., Keutzer, K., and Zhang, S. S3gaussian: Self-supervised street gaussians for autonomous driving. arXiv preprint arXiv:2405.20323, 2024b.

Jiang, G., Sun, Y., Huang, T., Li, H., Liang, Y., and Xu, H. Robots pre-train robots: Manipulation-centric robotic representation from large-scale robot datasets. arXiv preprint arXiv:2410.22325, 2024.

Jiang, H., Hsu, H.-Y., Zhang, K., Yu, H.-N., Wang, S., and Li, Y. Phystwin: Physics-informed reconstruction and simulation of deformable objects from videos. arXiv preprint arXiv:2503.17973, 2025.

Kerbl, B., Kopanas, G., Leimkuhler, T., and Drettakis, G. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

Li, Q., Deng, Y., Liang, Y., Luo, L., Zhou, L., Yao, C., Zeng, L., Feng, Z., Liang, H., Xu, S., et al. Scalable

vision-language-action model pretraining for robotic manipulation with real-life human activity videos. arXiv preprint arXiv:2510.21571, 2025a.

Li, T., Bolkart, T., Black, M. J., Li, H., and Romero, J. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 36(6):194â1, 2017.

Li, X., Li, J., Zhang, Z., Zhang, R., Jia, F., Wang, T., Fan, H., Tseng, K.-K., and Wang, R. Robogsim: A real2sim2real robotic gaussian splatting simulator. arXiv preprint arXiv:2411.11839, 2024a.

Li, Y., Wei, X., Chi, X., Li, Y., Zhao, Z., Wang, H., Ma, N., Lu, M., and Zhang, S. Manipdreamer: Boosting robotic manipulation world model with action tree and visual guidance. arXiv preprint arXiv:2504.16464, 2025b.

Li, Y., Wei, X., Chi, X., Li, Y., Zhao, Z., Wang, H., Ma, N., Lu, M., and Zhang, S. Manipdreamer3d: Synthesizing plausible robotic manipulation video with occupancyaware 3d trajectory. arXiv preprint arXiv:2509.05314, 2025c.

Li, Z., Zheng, Z., Wang, L., and Liu, Y. Animatable gaussians: Learning pose-dependent gaussian maps for highfidelity human avatar modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19711â19722, 2024b.

Liang, Y., Ellis, K., and Henriques, J. Rapid motor adaptation for robotic manipulator arms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16404â16413, 2024.

Lou, H., Liu, Y., Pan, Y., Geng, Y., Chen, J., Ma, W., Li, C., Wang, L., Feng, H., Shi, L., et al. Robo-gs: A physics consistent spatial-temporal model for robotic arm with hybrid representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 15379â15386. IEEE, 2025.

Lu, G., Zhang, S., Wang, Z., Liu, C., Lu, J., and Tang, Y. Manigaussian: Dynamic gaussian splatting for multitask robotic manipulation. In European Conference on Computer Vision, pp. 349â366. Springer, 2024.

Ma, X., Patidar, S., Haughton, I., and James, S. Hierarchical diffusion policy for kinematics-aware multi-task robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18081â18090, 2024.

Ma, Z., Jiang, J., Chen, Y., and Zhang, L. B\âeziergs: Dynamic urban scene reconstruction with b\âezier curve gaussian splatting. arXiv preprint arXiv:2506.22099, 2025.

Mortenson, M. E. Mathematics for computer graphics applications. Industrial Press Inc., 1999.

Pfaff, N., Fu, E., Binagia, J., Isola, P., and Tedrake, R. Scalable real2sim: Physics-aware asset generation via robotic pick-and-place setups. arXiv preprint arXiv:2503.00370, 2025.

Qian, S., Kirschstein, T., Schoneveld, L., Davoli, D., Giebenhain, S., and Niessner, M. Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20299â20309, 2024.

Qureshi, M. N., Garg, S., Yandun, F., Held, D., Kantor, G., and Silwal, A. Splatsim: Zero-shot sim2real transfer of rgb manipulation policies using gaussian splatting. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 6502â6509. IEEE, 2025.

Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Radle, R., Rolland, C., Gustafson, L., et al. Â¨ Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

Sun, B., Wang, N., Ma, X., Zou, A., Lu, Y., Fan, C., Wang, Z., Lu, K., and Wang, Z. Robava: A large-scale dataset and baseline towards video based robotic arm action understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13985â13994, 2025.

Tao, T., Zhang, L., Wen, Y., Zhang, K., Bian, J.-W., Zhou, X., Yan, T., Zhan, K., Jia, P., Wu, H., et al. Robopearls: Editable video simulation for robot manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10118â10129, 2025.

Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pp. 5026â5033. IEEE, 2012.

Wang, B., Meng, X., Wang, X., Zhu, Z., Ye, A., Wang, Y., Yang, Z., Ni, C., Huang, G., and Wang, X. Embodiedreamer: Advancing real2sim2real transfer for policy training via embodied world modeling. arXiv preprint arXiv:2507.05198, 2025a.

Wang, H., Wei, X., Zhang, X., Li, J., Bai, C., Li, Y., Lu, M., Zheng, W., and Zhang, S. Embodiedocc++: Boosting embodied 3d occupancy prediction with plane regularization and uncertainty sampler. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 925â934, 2025b.

Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., and Novotny, D. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294â5306, 2025c.

Wang, Y., Wei, X., Lu, M., and Kang, G. Plgs: Robust panoptic lifting with 3d gaussian splatting. IEEE Transactions on Image Processing, 2025d.

Wei, X., Chen, P., Li, G., Lu, M., Chen, H., and Tian, F. Gazegaussian: High-fidelity gaze redirection with 3d gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13293â 13303, 2025a.

Wei, X., Chen, P., Lu, M., Chen, H., and Tian, F. Graphavatar: Compact head avatars with gnn-generated 3d gaussians. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 8295â8303, 2025b.

Wei, X., Wuwu, Q., Zhao, Z., Wu, Z., Huang, N., Lu, M., Ma, N., and Zhang, S. Emd: Explicit motion modeling for high-quality street gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 28462â28472, 2025c.

Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., and Wang, X. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20310â20320, 2024.

Xie, Z., Liu, Z., Peng, Z., Wu, W., and Zhou, B. Vid2sim: Realistic and interactive simulation from video for urban navigation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1581â1591, 2025.

Xu, Y., Chen, B., Li, Z., Zhang, H., Wang, L., Zheng, Z., and Liu, Y. Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1931â1941, 2024.

Yan, Y., Lin, H., Zhou, C., Wang, W., Sun, H., Zhan, K., Lang, X., Zhou, X., and Peng, S. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In European Conference on Computer Vision, pp. 156â173. Springer, 2024.

Yang, S., Yu, W., Zeng, J., Lv, J., Ren, K., Lu, C., Lin, D., and Pang, J. Novel demonstration generation with gaussian splatting enables robust one-shot manipulation. arXiv preprint arXiv:2504.13175, 2025.

Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., and Jin, X. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20331â20341, 2024.

Yu, J., Fu, L., Huang, H., El-Refai, K., Ambrus, R. A., Cheng, R., Irshad, M. Z., and Goldberg, K. Real2render2real: Scaling robot data without dynamics simulation or robot hardware. arXiv preprint arXiv:2505.09601, 2025a.

Yu, T., Lu, G., Yang, Z., Deng, H., Chen, S. S., Lu, J., Ding, W., Hu, G., Tang, Y., and Wang, Z. Manigaussian++: General robotic bimanual manipulation with hierarchical gaussian world model. arXiv preprint arXiv:2506.19842, 2025b.

Zhang, M., Zhang, K., and Li, Y. Dynamic 3d gaussian tracking for graph-based neural dynamics modeling. arXiv preprint arXiv:2410.18912, 2024.

Zhou, H., Shao, J., Xu, L., Bai, D., Qiu, W., Liu, B., Wang, Y., Geiger, A., and Liao, Y. Hugs: Holistic urban 3d scene understanding via gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21336â21345, 2024.

Zhou, Y., Barnes, C., Lu, J., Yang, J., and Li, H. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5745â5753, 2019.

## A. Overview

This appendix provides supplementary details regarding the proposed RoboArm4D dataset and presents additional experimental evaluations for RoboArmGS. We first elaborate on the dataset construction in the following section, covering the hardware specifications, data capture protocols, and the processing pipeline used to generate the monocular sequences. Subsequently, we report extended quantitative and qualitative results on the Universal Robots UR5e and ABB IRB 120 sequences. These additional experiments on Novel-View Synthesis and Novel-Pose Synthesis further validate the generalizability and high-fidelity rendering capabilities of our method across diverse robotic morphologies.

## B. The RoboArm4D Dataset Details

To facilitate research in high-fidelity robotic arm reconstruction and motion modeling, we introduce the RoboArm4D dataset. This dataset features monocular video sequences of several common robotic arms performing diverse motions. This section provides a detailed overview of our data capture hardware, protocol, and processing pipeline.

## B.1. Hardware Setup

Our data collection setup was designed for simplicity and accessibility, requiring minimal specialized equipment. We captured all sequences using a single, handheld Intel RealSense L515 camera, recording RGB video at a resolution of 640x480 and a rate of 30 frames per second to simulate casual capture conditions. The dataset features three widely-used robotic arms: the Franka Research 3 (7-DoF), the Universal Robots UR5e (6-DoF), and the ABB IRB 120 (6-DoF). Each arm was mounted on a workbench within a standard laboratory environment, characterized by diffuse overhead lighting and a relatively static, yet typical, background containing various lab equipment.

## B.2. Data Capture Protocol

## B.2.1. CAMERA POSE CALIBRATION

Accurately determining the pose of a static camera is crucial. To achieve this, we first captured a short calibration video (approx. 30 seconds) where the handheld camera was moved extensively around the static robotic arm and its environment. This multi-view sequence was processed using VGGT (Wang et al., 2025c) to generate a sparse 3D reconstruction of the scene and to precisely compute the camera poses for each frame of the calibration video. From this set of calibrated poses, we selected a single, fixed viewpoint for the subsequent motion capture. This process effectively pre-calibrates the static cameraâs extrinsic parameters within the sceneâs coordinate frame.

## B.2.2. MOTION TRAJECTORY AND RECORDING

With the camera now fixed in its pre-calibrated position, the robotic arm was programmed to execute a pre-defined, smooth trajectory. These trajectories were designed to cover a wide range of joint configurations, including both simple single-joint movements and complex multi-joint coordinated motions. While the arm was in motion, we recorded a continuous video sequence from the static viewpoint. Each motion sequence lasts approximately 30-60 seconds, resulting in 900-1800 frames. Simultaneously, joint angles were recorded directly from the robotâs controller API to be synchronized with the video frames during post-processing.

## B.2.3. DATA PROCESSING PIPELINE

The raw video and joint angle data were processed into a format suitable for training our model using the following steps:

â¢ Video to Frames: The captured videos were decomposed into individual PNG frames.

â¢ Camera Pose Estimation: We used VGGT (Wang et al., 2025c) to estimate the camera intrinsics and the per-frame extrinsic poses (camera-to-world transformation). Poses were optimized over the entire sequence to ensure global consistency.

â¢ Foreground Segmentation: To separate the robotic arm from the background, we employed the Segment Anything Model 2 (SAM2) (Ravi et al., 2024). We provided a few initial keyframe masks, and SAM2 (Ravi et al., 2024) automatically propagated the segmentation to the entire sequence, followed by minor manual refinement where

necessary. This resulted in a pixel-perfect foreground mask for each frame.

â¢ Data Synchronization: The high-frequency joint angle data were synchronized with the video frames. We used linear interpolation to obtain the precise joint angle configuration corresponding to the capture time of each frame.

## B.3. Dataset Splitting Protocol

We adopt a uniform sampling strategy to partition the dataset into training, validation, and test sets with an 8:1:1 ratio. Specifically, we select every 10th frame for the test set and every 10th frame for the validation set, ensuring distinct frames for each subset (non-overlapping). The remaining frames constitute the training set. This interval-based splitting ensures that the evaluation covers the full range of motion and viewpoints present in the recorded sequences.

## C. Additional Experiments Results

In this section, we provide a detailed per-scene quantitative analysis of RoboArmGS on the RoboArm4D dataset, specifically focusing on the Universal Robots UR5e and ABB IRB 120 sequences. As shown in Tab. 4, our method achieves consistently superior performance across different robotic morphologies, significantly outperforming existing state-of-the-art (SOTA) methods and validating the generalizability of our proposed framework.

## C.1. Novel-View Synthesis

We first evaluate the static reconstruction quality on the Novel-View Synthesis task. As reported in Tab. 4 (left), RoboArmGS achieves leading rendering fidelity on both robotic arms. Specifically, for the UR5e, our method attains a PSNR of 35.220 dB, surpassing the best baseline (SuGaR) by a substantial margin of 4.768 dB. Similarly, for the ABB IRB 120, we achieve a PSNR of 34.815 dB, outperforming Robo-GS by 4.628 dB. These metrics indicate that our Structured Gaussian Binding (SGB) strategy effectively leverages the underlying kinematic mesh to provide strong geometric constraints for the 3D Gaussians. Unlike unconstrained methods like 2DGS or SuGaR, SGB allows for precise geometry reconstruction even

Novel-View Synthesis

<!-- image-->

Figure 6. Qualitative results for Novel-View Synthesis. RoboArmGS synthesizes photorealistic images from unseen viewpoints for both the Universal Robots UR5e (left) and ABB IRB 120 (right). Our method faithfully reconstructs high-frequency details and preserves the visual fidelity of the robotâs appearance, while maintaining sharp geometric boundaries against the background.

<table><tr><td colspan="4">Novel-View Synthesis</td><td colspan="4">Novel-Pose Synthesis</td></tr><tr><td>Method</td><td></td><td>PSNRâ SSIMâ LPIPSâ</td><td></td><td>Method</td><td></td><td></td><td>PSNRâ SSIMâ LPIPSâ</td></tr><tr><td colspan="8">Universal Robots UR5e</td></tr><tr><td>2DGS (Huang et al., 2024a)</td><td>29.972</td><td>0.939</td><td>0.074</td><td>3DGS + FK</td><td>26.086</td><td>0.945</td><td>0.056</td></tr><tr><td>Robo-GS (Lou et al., 2025)</td><td>29.811</td><td>0.938</td><td>0.074</td><td>4DGS (Wu et al., 2024)</td><td>28.239</td><td>0.967</td><td>0.039</td></tr><tr><td>SuGaR (Guedon &amp; Lepetit, 2024)</td><td>30.452</td><td>0.951</td><td>0.058</td><td>Deformable 3DGS (Yang et al., 2024)</td><td>30.959</td><td>0.982</td><td>0.022</td></tr><tr><td>Ours</td><td>35.220</td><td>0.969</td><td>0.025</td><td>Ours</td><td>39.559</td><td>0.989</td><td>0.011</td></tr><tr><td colspan="8">ABB IRB 120</td></tr><tr><td>2DGS (Huang et al., 2024a)</td><td>30.179</td><td>0.954</td><td>0.065</td><td>3DGS + FK</td><td>24.801</td><td>0.933</td><td>0.055</td></tr><tr><td>Robo-GS (Lou et al., 2025)</td><td>30.187</td><td>0.954</td><td>0.061</td><td>4DGS (Wu et al., 2024)</td><td>30.190</td><td>0.966</td><td>0.047</td></tr><tr><td>SuGaR (Guedon &amp; Lepetit, 2024)</td><td>28.368</td><td>0.950</td><td>0.072</td><td>Deformable 3DGS (Yang et al., 2024)</td><td>34.706</td><td>0.984</td><td>0.025</td></tr><tr><td>Ours</td><td>34.815</td><td>0.968</td><td>0.038</td><td>Ours</td><td>36.379</td><td>0.982</td><td>0.018</td></tr></table>

Table 4. Quantitative comparison on UR5e and ABB We evaluate our method against baselines on both Novel-View Synthesis (left) and Novel-Pose Synthesis (right). The top section shows results for the UR5e arm, and the bottom section for the ABB arm.

Novel-Pose Synthesis

<!-- image-->  
Figure 7. Qualitative results for Novel-Pose Synthesis. We visualize the rendered results of the robotic arms under unseen joint configurations (held-out test poses). Even in challenging poses that deviate significantly from the canonical training configurations, RoboArmGS maintains structural integrity and visual fidelity. The results demonstrate precise kinematic alignment and realistic shading changes consistent with the robotâs motion.

from sparse monocular inputs. Qualitative results (see Fig. 6) further corroborate these findings: our method renders sharp boundaries and preserves fine-grained texture details, effectively capturing the specific material characteristics of each robot without the âcloudyâ floaters or structural artifacts often seen in previous Gaussian-based reconstructions.

## C.2. Novel-Pose Synthesis

The Novel-Pose Synthesis task evaluates the modelâs ability to generate photorealistic images under unseen joint configurations, which is critical for high-fidelity digital assets simulation. Remarkably, as shown in Tab. 4 (right), RoboArmGS achieves even more better results on this dynamic task. For the UR5e arm, our method achieves a PSNR of 39.559 dB, which is a massive 8.6 dB improvement over the second-best method (Deformable 3DGS, 30.959 dB). On the ABB arm, we also maintain a significant lead with 36.379 dB PSNR. The LPIPS scores for both scenes remain exceptionally low, reflecting near-perfect perceptual realism. This superior performance is primarily attributed to our Bezier-based Motion Â´ Refiner (BMR). While methods like 4DGS and Deformable 3DGS struggle with large-range articulated movements and cumulative joint errors, BMR successfully compensates for the residual offsets between the idealized URDF model and real-world mechanical motion. By decoupling these motion artifacts from appearance learning, BMR ensures that Gaussians remain accurately attached to their respective links during complex movements. As visualized in Fig. 7, our rendered arms align perfectly with the target poses while maintaining consistent lighting and occlusion handling, proving that RoboArmGS is a highly robust solution for dynamic robotic simulation.