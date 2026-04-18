# Zero-Shot UAV Navigation in Forests via Relightable 3D Gaussian Splatting

Zinan Lv, Yeqian Qian\*, Member, IEEE, Chen Sang, Hao Liu, Danping Zou, Member, IEEE, and Ming Yang\*, Member, IEEE,

<!-- image-->  
Fig. 1: The Real-Sim-Real pipeline. By training in a photorealistic 3DGS simulator with active Relightable 3D Gaussian Splatting, our method enables zero-shot, high-speed monocular navigation in complex real-world environments.

AbstractâUAV navigation in unstructured outdoor environments using passive monocular vision is hindered by the substantial visual domain gap between simulation and reality. While 3D Gaussian Splatting enables photorealistic scene reconstruction from real-world data, existing methods inherently couple static lighting with geometry, severely limiting policy generalization to dynamic real-world illumination. In this paper, we propose a novel end-to-end reinforcement learning framework designed for effective zero-shot transfer to unstructured outdoors. Within a high-fidelity simulation grounded in real-world data, our policy is trained to map raw monocular RGB observations directly to continuous control commands. To overcome photometric limitations, we introduce Relightable 3D Gaussian Splatting, which decomposes scene components to enable explicit, physically

grounded editing of environmental lighting within the neural representation. By augmenting training with diverse synthesized lighting conditions ranging from strong directional sunlight to diffuse overcast skies, we compel the policy to learn robust, illumination-invariant visual features. Extensive real-world experiments demonstrate that a lightweight quadrotor achieves robust, collision-free navigation in complex forest environments at speeds up to 10 m/s, exhibiting significant resilience to drastic lighting variations without fine-tuning.

Index TermsâAutonomous Navigation, UAV, 3D Gaussian Splatting, End-to-end Reinforcement learning.

## I. INTRODUCTION

UTONOMOUS navigation of Unmanned Aerial Vehicles (UAVs) in unstructured outdoor environments is crucial, enabling diverse tasks ranging from disaster response [1], [2] to infrastructure inspection [3], [4]. To ensure flight safety in these complex settings, contemporary navigation systems predominantly rely on active geometric perception, employing sensors such as LiDAR [5] and depth cameras [6], [7]. While providing reliable depth information, these active solutions impose severe system-level penalties: they increase payload weight, which degrades flight agility, induce computational latency that bottlenecks high-speed control, and suffer from sensing failures and infrared interference in outdoor sunlight. These constraints motivate the critical inquiry: Is it possible to replicate bird-like, high-speed agility in dense, unstructured forests using solely a passive monocular RGB camera?

Monocular vision-based navigation has emerged as a compelling alternative, offering a pathway toward agile and lowcost autonomy. Learning-based frameworks have demonstrated remarkable success in structured environments, such as drone racing circuits [8], [9] and indoor corridors [10], where simulation modeling is straightforward. However, extending these capabilities to unstructured, cluttered outdoor environments remains a complex challenge due to the difficulty of modeling irregular geometries with traditional simulators [11], [12]. Recently, 3D Gaussian Splatting (3DGS) [13] has gained significant attention in navigation research owing to its photorealistic reconstruction. Approaches like Splat-Nav [14] and [15] leverage 3DGS effectively but have predominantly focused on structured indoor environments with stable lighting. Consequently, the distinct challenges of unstructured outdoor settings, particularly characterized by uncontrollable illumination, remain less explored. Addressing this challenge necessitates the capability to simulate dynamic illumination. However, standard relighting approaches often struggle to accurately disentangle geometry from illumination in complex wild scenes, erroneously encoding shadows into the surface albedo. This physical ambiguity limits the synthesis of diverse training data, thereby increasing the risk of overfitting to static capture conditions rather than generalizing to real-world dynamic illumination.

In this paper, we present a novel end-to-end deep reinforcement learning (RL) framework for autonomous UAV navigation, specifically engineered to achieve zero-shot transfer from simulation to the physical domain. By directly mapping raw monocular RGB imagery to continuous control commands, this end-to-end architecture effectively mitigates compounding errors inherent in traditional modular pipelines. Complementarily, the utilization of RL facilitates the acquisition of robust navigation policies for complex behaviors. To enable highfidelity training, our proposed Real-Sim-Real pipeline, as illustrated in Fig. 1, leverages unstructured, in-the-wild video sequences to reconstruct photorealistic and geometrically consistent digital twins. Grounding the simulation in empirical real-world data significantly narrows the visual sim-to-real domain gap, enabling agile autonomous navigation in complex environments at speeds exceeding 10 m/s. Remarkably, this agile autonomous flight is achieved using a monocular RGB camera solely without requiring any real-world fine-tuning.

Furthermore, to ensure robustness against the unpredictability of outdoor environments, we introduce Relightable 3D Gaussian Splatting. Addressing the intrinsic limitation of standard 3DGS, where lighting conditions are statically coupled with scene geometry, our method decomposes the scene components to allow for explicit editing of environmental lighting. This capability enables us to augment the training process with a diverse spectrum of synthesized lighting conditions, ranging from strong directional sunlight to diffuse overcast scenarios. By exposing the agent to these variations during simulation, we force the policy to learn illumination-invariant visual features, preventing overfitting to the specific conditions of the data collection time and ensuring reliable performance across different times of day and weather conditions.

The main contributions of this work are summarized as follows:

â¢ We develop a novel end-to-end deep reinforcement learning framework for vision-based UAV navigation, which directly maps raw visual observations to continuous control commands, enabling agile autonomous flight without relying on handcrafted features or modular components.

â¢ To enable effective zero-shot transfer to unstructured outdoor environments, we leverage Relightable 3D Gaussian Splatting to construct high-fidelity simulations with diverse lighting synthesis, ensuring policy robustness to varying real-world illumination conditions.

â¢ We empirically validate our system through extensive flight experiments, demonstrating that our policy achieves robust, collision-free autonomous navigation at speeds up to 10 m/s in complex forest environments under varying real-world illumination conditions.

## II. RELATED WORKS

## A. Vision-Based Autonomous Navigation for UAVs

Traditional frameworks [16]â[18] typically adopt a modular pipeline, integrating explicit mapping (e.g., voxel grids, ESDFs), localization, and trajectory planning. Representative systems like FASTER [19] ensure safety by maintaining backup trajectories in known free space. While theoretically robust in static settings, these methods require precise state estimation and low-latency depth perception. Maintaining highresolution dense maps on onboard hardware is computationally expensive. Furthermore, in texture-less, dynamic, or highly cluttered environments where state estimation drifts or depth sensors fail, modular pipelines often become brittle due to error propagation, where a failure in perception leads to catastrophic failure in planning.

To bypass these bottlenecks, learning-based approaches [20]â[24] have garnered significant attention. Imitation Learning (IL) [25] has been successfully applied to drone racing [8] and trail following [26], yet it suffers from covariate shift. Deep Reinforcement Learning (DRL) offers a robust alternative by allowing agents to learn from failures. Recent works have achieved high-speed flight using active sensors [5], [6], [27], [28]. However, active sensors impose strict size, weight, and power constraints, thereby severely limiting their practical deployment on agile, sub-250g micro-UAVs.

Recently, 3D Gaussian Splatting has gained significant attention in navigation research owing to its rapid rendering capabilities. Splat-Nav [14] attempts to integrate 3DGS directly into the online planning stack; however, the computational demands of real-time map densification remain prohibitive for resource-constrained micro-UAVs. Similarly, GRaD-Nav++ [29] exploits the differentiable nature of 3DGS for trajectory optimization, yet the heavy reliance on iterative rendering and gradient computation poses severe challenges for efficient onboard deployment. Alternatively, [15] leverages 3DGS for offline Sim-to-Real training to mitigate these computational constraints. Yet, these methods have primarily been validated in structured indoor environments, leaving the distinct challenges of complex, unstructured outdoor settings, such as dense forests characterized by irregular occlusion and uncontrollable illumination, largely unaddressed.

Monocular vision offers a compelling, lightweight alternative but faces greater perception challenges due to scale ambiguity. Our work targets this specific gap by leveraging the real-time rendering capability of 3DGS to construct a massive, diverse simulation environment. Unlike prior works that either demand heavy onboard mapping or suffer from slow data generation, we utilize 3DGS solely for efficient Sim-to-Real training, allowing the UAV to learn robust depth cues from single RGB images without explicit depth supervision or heavy online computation.

## B. High-Fidelity Simulation Environments

The development of autonomous UAV algorithms faces significant hurdles regarding safety, cost, and scalability when relying exclusively on real-world experimentation. Simulation provides a critical alternative, evolving through distinct technical paradigms [30] to address the growing demand for fidelity.

Initial simulation platforms prioritized accurate flight dynamics and control systems [12], [31], [32], often integrating seamlessly with standard autopilot stacks like PX4 [33] or ArduPilot [34]. While indispensable for low-level control verification and swarm dynamics, these platforms typically rely on rudimentary geometric primitives and synthetic textures. To address the resulting visual realism gap, researchers increasingly adopted modern game engines (e.g., Unreal Engine, Unity) to render complex scenes [35], [36]. Prominent frameworks such as AirSim [11] and AirSim360 [37] provide rich, visually coherent synthetic data tailored for robotic learning. However, these environments depend on artist-crafted assets or procedural generation. Consequently, they inevitably exhibit stylistic discrepancies, where textures appear artificially clean or repetitive compared to the stochastic complexity of the real world. This discrepancy results in a persistent domain gap, where policies trained on synthetic assets fail to generalize to the intricate, disordered geometry found in real-world unstructured environments.

Recently, data-driven 3D reconstruction has emerged as a promising solution for creating realistic digital twins. Neural Radiance Fields (NeRF) [38] demonstrated implicit scene modeling capabilities, inspiring works like [39] to use NeRFs for trajectory optimization. However, the implicit nature of NeRF necessitates computationally expensive volumetric querying for every pixel, rendering it unsuitable for the highfrequency rendering required in large-scale Reinforcement

Learning (RL) pipelines. In contrast, 3D Gaussian Splatting (3DGS) [13] has offered a transformative representation, combining the visual fidelity of NeRFs with the rasterization speed of game engines. Recent robotics applications, such as RAD [40] for autonomous driving and Splat-Nav [14] for dense mapping, have validated its potential. Despite these advances, standard 3DGS-based simulators often inherently entangle static lighting conditions within the scene representation. This lack of environmental controllability restricts the diversity of training data, causing policies to overfit to specific lighting conditions. In this work, we address this limitation by integrating a 3DGS-based simulator with a novel Relightable 3D Gaussian Splatting specifically designed for scalable, randomized RL training.

## C. Photometric Domain Adaptation and Relighting

Traditional Domain Adaptation (DA) methods rely on 2D image-space augmentations. Domain Randomization (DR) [41] applies random textures and colors to simulation meshes to force the network to learn geometric invariants. However, this often results in non-physical visual inputs that may hinder the learning of realistic features. Other approaches utilize GANs [42] or Style Transfer [43] to map synthetic images to the real domain. While effective for static images, these generative methods often suffer from temporal inconsistency (flickering) and can generate spurious artifacts, which are detrimental to high-speed control policies that rely on consistent optical flow cues.

In contrast, the advent of 3DGS has enabled more physically grounded appearance editing. Recent approaches like Relightable 3D Gaussians [44] and GI-GS [45] decompose the scene into geometry, material properties (albedo, roughness), and lighting, allowing for realistic relighting and shadow synthesis. Others, such as GaussCtrl [46], leverage diffusion models for semantic editing. However, these rigorous inverserendering methods typically incur high computational overheads or require complex deferred shading pipelines. For instance, accurately solving the inverse rendering equation for every frame is computationally prohibitive when running hundreds of parallel environments. Our approach introduces an efficient, geometry-aware illumination editing mechanism that balances physical plausibility with the rendering speed necessary for large-scale policy optimization, enabling effective zero-shot transfer.

## III. RELIGHTABLE 3D GAUSSIAN SPLATTING

To achieve robust zero-shot transfer, we require a simulation environment capable of synthesizing diverse lighting conditions. However, standard 3DGS inherently couples static environmental illumination with scene geometry, limiting its adaptability. In this section, we first review the standard 3DGS formulation to identify its limitations, and then present our Relightable 3D Gaussian Splatting framework, which explicitly decomposes the scene into geometry, material albedo, and environmental lighting.

## A. Preliminaries on Standard 3DGS

3D Gaussian Splatting represents a scene as a collection of anisotropic 3D Gaussians. Each Gaussian $G _ { i }$ is defined by a center position ${ \pmb { \mu } } _ { i } \in \mathbb { R } ^ { 3 }$ , a covariance matrix $\pmb { \Sigma } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ an opacity scalar $\alpha _ { i } ~ \in ~ [ 0 , 1 ]$ , and view-dependent color coefficients $\mathbf { f } _ { i }$ represented via Spherical Harmonics (SH). The geometry of the i-th Gaussian is described by:

$$
G _ { i } ( \mathbf { x } ) = \exp \left( - \frac { 1 } { 2 } ( \mathbf { x } - \pmb { \mu } _ { i } ) ^ { \top } \pmb { \Sigma } _ { i } ^ { - 1 } ( \mathbf { x } - \pmb { \mu } _ { i } ) \right) .\tag{1}
$$

To ensure the covariance matrix $\Sigma _ { i }$ remains positive semidefinite during optimization, it is decomposed into a rotation matrix $\mathbf { R } _ { i }$ and a scaling matrix $\mathbf { S } _ { i } ,$ such that $\begin{array} { l l } { \pmb { \Sigma } _ { i } } & { = } \end{array}$ ${ \bf R } _ { i } { \bf S } _ { i } { \bf S } _ { i } ^ { \top } { \bf R } _ { i } ^ { \top }$

During rendering, the 3D Gaussians are projected onto the 2D image plane. The visible color $C ( \mathbf { p } )$ of a pixel $\mathbf { x }$ is computed using point-based Î±-blending, sorting the $N$ overlapping Gaussians by depth:

$$
C ( \mathbf { x } ) = \sum _ { i \in N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $c _ { i }$ is the view-dependent color computed directly from the SH coefficients fi based on the viewing direction.

In the standard formulation, the term $c _ { i }$ represents the total observed radiance, which mathematically entangles the intrinsic surface albedo with the specific lighting conditions present at the time of capture. This entanglement prevents the independent modification of illumination, making standard 3DGS unsuitable for synthesizing the novel lighting scenarios required for domain adaptation.

## B. Relightable Formulation

To overcome the static nature of standard 3DGS, where lighting is intrinsically entangled with surface properties, we propose a physically grounded decomposition of the radiance term $c _ { i } .$ . We re-formulate the rendering process to explicitly model the interaction between learnable diffuse albedo, environmental lighting, and geometric occlusion. The final color $\mathbf { c } _ { i }$ of the $i ^ { t h }$ Gaussian is synthesized via a shading equation that modulates the global environmental light by local visibility and surface material properties:

$$
\mathbf { c } _ { i } = \pmb { \rho } _ { i } \odot \left( \sum _ { m = 1 } ^ { ( l + 1 ) ^ { 2 } } \mathbf { L } _ { e n v } ^ { m } \cdot \mathbf { O } _ { i } ^ { m } \cdot \mathbf { d } _ { i } ^ { m } \right) ,\tag{3}
$$

where ${ \rho } _ { i } \in \mathbb { R } ^ { 3 }$ represents the learnable diffuse albedo, describing the intrinsic material color independent of illumination. $\mathbf { L } _ { e n \tau } ^ { m }$ denotes the $m ^ { t h }$ SH coefficient of the global environmental lighting. In contrast to standard 3DGS, which optimizes lighting parameters per-Gaussian, we model $\mathbf { L } _ { e n v }$ as a global variable shared across the scene to ensure physical consistency. $\mathbf { 0 } _ { i } ^ { m }$ encodes the occlusion coefficient at the Gaussianâs center, characterizing the directional light blockage derived from the scene geometry, while ${ \bf d } _ { i } ^ { m }$ is a learnable transfer coefficient that models local geometric interactions, such as surface normal orientation and inter-reflections.

This formulation effectively decouples geometry from illumination. By treating $\mathbf { L } _ { e n v }$ as an independent global variable, we can alter the sceneâs appearance solely by modifying these coefficients while keeping the geometry $\mathbf { O } _ { i }$ and material $\rho _ { i }$ fixed, which is the cornerstone of our zero-shot domain transfer capability.

## C. Occlusion-Aware Visibility Modeling

Accurate shadow synthesis is critical for realistic outdoor simulation but is often neglected in standard splatting frameworks. To address this, we introduce a pre-computed Occlusion Field to model global light transport via a voxelized probe network.

Visibility Field Construction. The scene space is discretized into a uniform voxel grid. At each grid node $\mathbf q _ { k }$ , we render six orthographic depth maps $D _ { k , f }$ along the cardinal axes $( f \in \{ \pm X , \pm Y , \pm Z \} )$ to capture the surrounding geometry. To mitigate geometric noise inherent in 3DGS reconstructions, we apply a robust depth threshold $d _ { t h r e s h }$ . The binary visibility $V _ { k , f } ( \mathbf { x } )$ for a direction mapped to pixel x is determined by:

$$
V _ { k , f } ( { \bf x } ) = \left\{ \begin{array} { l l } { 0 , } & { D _ { k , f } ( { \bf x } ) < d _ { t h r e s h } \quad \mathrm { ( O c c l u d e d ) } } \\ { 1 , } & { D _ { k , f } ( { \bf x } ) \geq d _ { t h r e s h } \quad \mathrm { ( V i s i b l e ) } } \end{array} \right. .\tag{4}
$$

SH Projection and Interpolation. To enable efficient differentiable rendering, we project this discrete visibility map onto a Spherical Harmonics basis. The occlusion SH coefficients $\mathbf { B } _ { k }$ for probe k are computed by integrating the visibility over the sphere:

$$
\mathbf { B } _ { k } ^ { l m } = \sum _ { f } \sum _ { \mathbf { x } } V _ { k , f } ( \mathbf { x } ) \cdot Y _ { l m } ( \boldsymbol { \omega } _ { \mathbf { x } } ) \cdot \Delta \boldsymbol { \Omega } ,\tag{5}
$$

where $Y _ { l m }$ are the real SH basis functions, $\omega _ { \mathbf { x } }$ denotes the normalized direction vector corresponding to pixel $\mathbf { x } ,$ and $\Delta \Omega$ is the solid angle. Finally, to assign occlusion values to the continuous space of Gaussian centers, we perform trilinear interpolation. For an arbitrary Gaussian at position $\mu _ { i } ,$ its occlusion coefficient $\mathbf { O } _ { i }$ is derived from the eight nearest voxel probes $\mathcal { N } ( \pmb { \mu } _ { i } )$ :

$$
\mathbf { O } _ { i } = \sum _ { k \in \mathcal { N } ( \pmb { \mu } _ { i } ) } w _ { k } \cdot \beta _ { k } \cdot \mathbf { B } _ { k } ,\tag{6}
$$

where $w _ { k }$ is the spatial interpolation weight and $\beta _ { k }$ is a backface culling mask ensuring the probe direction aligns with the Gaussianâs normal.

## D. High-Fidelity HDR Lighting Prior

Jointly optimizing geometry and lighting often leads to ambiguity, where dark textures are incorrectly modeled as shadows or shadows are baked into albedo. To enforce stable disentanglement, we employ a Deep Panorama Lighting (DPL) pipeline to recover high-dynamic-range (HDR) illumination from the Low Dynamic Range (LDR) training images.

Standard LDR images often clip high-intensity sunlight, leading to inaccurate lighting estimation. We first lift the LDR input to an HDR representation using a pre-trained auto-encoder. The global environmental lighting $I _ { e n v }$ is then represented by projecting the HDR map onto SH coefficients $\mathbf { L } _ { l m } \colon$

<!-- image-->  
Fig. 2: The pipeline of our proposed framework for monocular RGB vision-based autonomous UAV navigation comprises three key stages: 1) Photorealistic Environment Construction: Real-world unstructured scenes are captured and reconstructed using 3D Gaussian Splatting to build a high-fidelity simulator. 2) Sim-to-Real Adaptation: Domain adaptation techniques, including action noise injection, latency simulation, camera pose perturbation, and Relightable 3D Gaussian Splatting, are employed to bridge the visual and dynamics gaps. 3) End-to-end Vision-Based Policy Learning: A reinforcement learning policy processes monocular RGB images and drone state information through CNN and MLP encoders with GRU-based temporal modeling, generating control commands via actor-critic network heads.

$$
{ \bf L } _ { l m } = \frac { 4 \pi } { N _ { s } } \sum _ { \omega } I _ { H D R } ( \omega ) Y _ { l m } ( \omega ) ,\tag{7}
$$

where $N _ { s }$ is the number of sampling points on the panoramic sphere.

Crucially, rather than learning these coefficients online, we pre-calculate $\mathbf { L } _ { l m }$ for all training frames and freeze them as a strong prior during the 3DGS optimization. This constraint forces the network to attribute residual photometric errors to surface albedo and occlusion updates, ensuring geometrically consistent relighting and enabling the synthesis of realistic shadows under novel illumination conditions.

## IV. END-TO-END NAVIGATION FRAMEWORK

As illustrated in Fig. 2, we propose a closed-loop framework for zero-shot UAV navigation that seamlessly integrates realworld data acquisition with physical deployment. Our pipeline commences with the reconstruction of unstructured outdoor environments using the proposed Relightable 3DGS to establish a photorealistic and physically consistent simulation substrate. Within this high-fidelity digital twin, we train an end-to-end reinforcement learning policy that maps monocular RGB inputs and proprioceptive states directly to continuous control commands. To bridge the sim-to-real gap, we leverage the disentangled nature of our scene representation to execute aggressive photometric domain adaptation. This strategy exposes the agent to diverse synthesized illumination conditions during training, ensuring that the learned policy remains robust to dynamic lighting and enabling effective zero-shot transfer to challenging real-world forests.

## A. Photorealistic Simulation Environment

To mitigate the visual domain gap, we construct a digital twin of unstructured outdoor environments utilizing the Relightable 3DGS model. Unlike traditional game engines relying on handcrafted assets, our environment faithfully captures the intricate textures and irregular geometries characteristic of real-world forests.

Visual Rendering. The simulator synthesizes high-fidelity RGB observations $I _ { t } ^ { s i m }$ by querying the Relightable 3DGS model. At each time step t, given the droneâs pose $\mathbf { T } _ { t }$ , the system utilizes the relightable rendering equation to generate views. This mechanism allows for the real-time synthesis of photorealistic imagery that reflects the current, randomized lighting configuration.

Scene Interaction and Collision Detection. To enable physical interaction, a 3D point cloud P is extracted from the reconstructed scene by sampling the centers of the Gaussian ellipsoids. A KDTree spatial index is constructed on $\mathcal { P }$ to facilitate efficient proximity queries. The droneâs collision volume is approximated as a cylinder with radius $r _ { c o l }$ and height tolerance $h _ { t o l }$ . At each simulation step, we query all scene points within a radius $r _ { q u e r y } = r _ { c o l } + \delta _ { s a f e }$ . A collision event is triggered if any point $\mathbf { p } _ { p o i n t } \in \mathcal { P }$ satisfies:

<!-- image-->  
Fig. 3: Visual illustration of the adaptive speed schedule defined in Eq. 11. The target forward speed $v _ { t a r }$ is maximized at $v _ { b a s e }$ during straight flight and is smoothly attenuated towards $v _ { \mathrm { m i n } }$ as the yaw rate |u| approaches the limit $u _ { \mathrm { m a x } } .$ preventing sideslip during sharp turns.

$$
\| { \bf p } _ { d } ^ { x y } - { \bf p } _ { p o i n t } ^ { x y } \| _ { 2 } \leq r _ { c o l } \quad \mathrm { a n d } \quad | p _ { d } ^ { z } - p _ { p o i n t } ^ { z } | \leq h _ { t o l } ,\tag{8}
$$

where $\mathbf { p } _ { d }$ denotes the droneâs position and $\mathbf { p } _ { p o i n t }$ represents the position of a scene point.

Quadrotor Dynamics Model. We employ a simplified yet effective decoupled dynamics model to balance physical fidelity with computational efficiency. The droneâs kinematic state is defined as $\mathbf { x } = [ \mathbf { p } ^ { \top } , \mathbf { v } ^ { \top } , \boldsymbol \psi , \dot { \psi } ] ^ { \top }$ . To ensure numerical stability, the dynamics are integrated using $\mathcal { N }$ substeps within each control interval $\Delta t .$ . Within each substep, the yaw dynamics follow a first-order response:

$$
\ddot { \psi } = K _ { \psi } ( u - \dot { \psi } ) ,\tag{9}
$$

where $K _ { \psi }$ is the yaw control gain and u is the target yaw rate. The total force acting on the drone is computed as:

$$
\mathbf { F } _ { \mathrm { t o t a l } } = \mathbf { F } _ { \mathrm { t h r u s t } } + \mathbf { F } _ { \mathrm { d r a g } } + \mathbf { F } _ { \mathrm { g r a v i t y } } + \mathbf { F } _ { \mathrm { a l t i t u d e } } ,\tag{10}
$$

where $\mathbf { F } _ { \mathrm { t h r u s t } }$ is the thrust force aligned with the current heading direction $\mathbf { d } ~ = ~ [ \cos \psi , \sin \psi , \bar { 0 } ] ^ { \top }$ . The aerodynamic drag is modeled as $\mathbf { F } _ { \mathrm { d r a g } } = - c _ { d } \| \mathbf { v } \| \mathbf { v }$ with drag coefficient $c _ { d } ,$ and $\mathbf { F } _ { \mathrm { a l t i t u d e } }$ is the output of a PID controller maintaining a constant flight altitude. The resultant acceleration is used to update the velocity v and position p via semi-implicit Euler integration.

Adaptive Speed Schedule. To ensure dynamic stability during high-speed maneuvers, we incorporate a forward dynamics schedule that couples the target forward velocity $v _ { t a r }$ with the yaw rate action u. The velocity is modulated as:

$$
v _ { t a r } = v _ { \operatorname* { m i n } } + \left( v _ { b a s e } - v _ { \operatorname* { m i n } } \right) \cdot \left( 1 - \frac { | \boldsymbol { u } | } { u _ { \operatorname* { m a x } } } \right) ^ { 0 . 5 } ,\tag{11}
$$

where $v _ { b a s e }$ and $v _ { \mathrm { m i n } }$ represent the nominal and minimum forward speeds, respectively. As visualized in Fig. 3, this profile enforces a smooth attenuation of forward velocity as the yaw rate increases, thereby preventing sideslip during aggressive turns.

## B. End-to-End Vision-Based RL Policy Network

Network Architecture. We develop a neural network-based policy designed to map multimodal observations to continuous control commands. At each time step t, the policy processes two distinct input streams: a high-dimensional visual observation $I _ { t } \in \mathbb { R } ^ { H \times \bar { W } \times 3 }$ and the droneâs kinematic state vector $\mathbf { s } _ { t } .$ The state is defined in relative coordinates as $\mathbf { s } _ { t } ~ = ~ [ \mathbf { p } _ { r e l } ^ { \top } , \mathbf { v } ^ { \top } , \boldsymbol { \psi } , \dot { \boldsymbol { \psi } } ] ^ { \top }$ . The network outputs a continuous scalar action $a _ { t } \in [ - u _ { \operatorname* { m a x } } , u _ { \operatorname* { m a x } } ]$ , corresponding to the target yaw rate.

The architecture integrates three specialized modules. A Convolutional Neural Network (CNN) serves as the visual backbone, extracting spatial geometric features from $I _ { t } .$ . Concurrently, a Multi-Layer Perceptron (MLP) encodes the proprioceptive state $\mathbf { s } _ { t }$ into a compact latent embedding. To address the partial observability inherent in monocular navigation, we incorporate a Gated Recurrent Unit (GRU) [47] to capture sequential dynamics. The GRU maintains a hidden state by processing the concatenated visual and state features, effectively aggregating temporal context. The final control output is generated via a fusion module followed by fully connected layers that parameterize a Gaussian distribution $\pi ( \boldsymbol { a } _ { t } | \cdot )$ for action sampling.

Reward Function Design. To guide the policy toward safe and efficient navigation, we design a composite reward function $r _ { t }$ consisting of dense shaping signals. The formulation and weights are detailed in Table I.

TABLE I: Components and weights of the reward function
<table><tr><td>Reward Component</td><td>Mathematical Formulation</td><td>Weight</td></tr><tr><td>Progress Reward</td><td> $r _ { p r o g r e s s } = d _ { t - 1 } - d _ { t }$ </td><td>1.0</td></tr><tr><td>Alignment Reward</td><td> $r _ { a l i g n } = \cos ( \psi _ { t a r g e t } - \psi _ { t } )$ </td><td>0.1</td></tr><tr><td>Obstacle Penalty</td><td> $\begin{array} { r } { r _ { o b s t a c l e } = \operatorname* { m i n } \left( 0 , \frac { d _ { o b s } - R _ { s a f e } } { R _ { s a f e } } \right) } \end{array}$ </td><td>0.2</td></tr><tr><td>Success Reward</td><td> $r _ { s u c c e s s } = \mathbb { I } ( d _ { t } < R _ { g o a l } )$ </td><td>100.0</td></tr><tr><td>Collision Penalty</td><td> $r _ { c o l l i s i o n } = \mathbb { I } ( \mathrm { c o l l i s i o n } )$ </td><td>-50.0</td></tr></table>

The progress reward $r _ { p r o g r e s s }$ incentivizes the agent to approach the goal by maximizing the reduction in Euclidean distance $d _ { t }$ . The alignment reward $r _ { a l i g n }$ promotes smooth path following. A sparse success reward $r _ { s u c c e s s }$ is granted upon reaching the target vicinity. To ensure safety, an obstacle penalty $r _ { o b s t a c l e }$ imposes a negative cost linearly proportional to proximity when the distance to the nearest obstacle falls below a safety margin. Finally, a discrete collision penalty rcollision is applied upon impact. The total reward is the weighted sum:

$$
r _ { t } = \sum _ { i } \lambda _ { i } \cdot r _ { t } ^ { i } .\tag{12}
$$

Training Curriculum. The objective is to optimize the policy parameters Î¸ to maximize the expected cumulative reward

<!-- image-->

<!-- image-->  
(a) Natural light

<!-- image-->  
(b) Overcast

<!-- image-->  
(c) Cool-toned dusk

<!-- image-->  
(d) Warm-toned morning sunlight

Fig. 4: Examples of photorealistic Relightable 3D Gaussian Splatting. The columns display the original natural light (a) and synthesized variations: overcast (b), cool-toned dusk (c), and warm-toned morning sunlight (d) across different outdoor scenes.

using Proximal Policy Optimization (PPO) [48]. The training curriculum is structured into two distinct phases. Initially, the policy is trained on 3DGS scenes with static, original illumination to establish fundamental geometric understanding. Subsequently, the second phase activates the Relightable 3D Gaussian Splatting to introduce randomized lighting variations. This two-stage curriculum facilitates photometric domain adaptation, enabling the policy to learn robust features invariant to lighting changes.

## C. Training with Photometric Domain Adaptation

To enable zero-shot transfer to the physical world, we leverage the disentangled structure of our Relightable 3DGS to implement an aggressive Photometric Domain Adaptation strategy.

Instead of training on static lighting, we randomize the global illumination coefficients $\mathbf { L } _ { e n v }$ at the start of every training episode. We apply three types of perturbations to the source HDR lighting coefficients:

â¢ Rotation: The SH coefficients are rotated around the vertical axis to simulate varying times of day and solar azimuths.

â¢ Intensity Scaling: The coefficients are scaled to mimic diverse exposure levels and cloud cover densities.

â¢ Chromatic Tinting: Color shifts are applied to the coefficients to simulate different weather conditions $( \mathrm { e . g . }$ warmer tones for dusk, cooler tones for overcast days).

By exposing the RL agent to this continuous spectrum of synthesized illumination while strictly preserving the underlying geometry $\mathbf { O } _ { i }$ and material albedo $\rho _ { i } ,$ , we compel the policy to learn robust geometric features that are invariant to photometric variations. This ensures that the navigation policy remains reliable when deployed in real-world forests where lighting conditions are unpredictable and dynamic.

## V. EXPERIMENT

## A. Experiment Set

High-Fidelity 3DGS Simulation Environment Construction. To establish a photorealistic training and testing ground, we captured a diverse set of real-world forest scenes. The data acquisition was performed using a handheld device equipped with a stereo fisheye camera rig and a high-precision LiDAR integrated with a Real-Time Kinematic (RTK) positioning system. This sensor suite enabled the collection of synchronized, geo-referenced multimodal data. The RTK system provided centimeter-accurate global poses, which served as the initial camera parameters for the 3DGS optimization pipeline, while the LiDAR delivered a dense initial point cloud for scene geometry initialization. In total, ten distinct forest scenes were captured, each spanning an area of approximately $6 0 \times 6 0$ meters. These scenes were subsequently reconstructed into high-fidelity 3D models using the 3DGS framework, forming the core of our photorealistic simulator.

Simulation Implementation Details. The navigation policy is trained using Proximal Policy Optimization. Training is conducted in two phases across parallel simulation environments. The simulation operates at a control frequency of 10 $\mathrm { H z , }$ corresponding to a timestep of $\Delta t = 0 . 1 \mathrm { ~ s ~ }$ . The droneâs forward speed follows an adaptive schedule up to a maximum of $v _ { \operatorname* { m a x } } = 1 0$ m/s, with a maximum yaw rate command of $u _ { \mathrm { m a x } } = 1 . 0$ rad/s. For each training episode, a valid start-goal pair is randomly sampled from the scene; a pair is deemed valid if a collision-free navigation path exists between the points and the Euclidean distance exceeds 30 meters, ensuring episodes of sufficient challenge. The droneâs collision volume is modeled as a cylinder with a radius of $r _ { c o l } = 0 . 3$ m and a height tolerance of $h _ { t o l } = 0 . 2 \mathrm { m }$ . The safety distance threshold for the obstacle-aware reward component is set to $R _ { s a f e } = 2 . 0$ m. An episode terminates successfully when the drone arrives within 2.0 m of the goal position. All experiments were executed on a workstation equipped with an NVIDIA RTX 3090 GPU (24GB memory).

TABLE II: Parameter Randomization for Domain Adaptation.
<table><tr><td>Parameter</td><td>Distribution / Value</td></tr><tr><td>Action Noise</td><td> $\overline { { \mathcal { N } ( 0 , 1 . 0 ^ { 2 } ) } }$ </td></tr><tr><td>Latency Delay</td><td>U(0, 80) ms</td></tr><tr><td>Control Interval</td><td>U(10, 100) ms</td></tr><tr><td>XY Position Noise</td><td> $\mathcal { N } ( 0 , \dot { 0 } . 0 5 ^ { \dot { 2 } } )$  m</td></tr><tr><td>Z Position Noise</td><td> $\mathcal { N } ( 0 , 0 . 0 3 ^ { 2 } )$  m</td></tr><tr><td>Velocity Noise</td><td> $\mathcal { N } ( \dot { 0 } , 0 . 0 8 ^ { 2 } ) \ \mathrm { m / s }$ </td></tr><tr><td>Camera Position Offset</td><td> $\mathcal { U } ( - 0 . 1 , 0 . 1 ) \mathrm { ~ m ~ }$ </td></tr><tr><td>Camera Orientation Offset</td><td> $\dot { \mathcal { U } } ( - 5 ^ { \circ } , 5 ^ { \circ } )$ </td></tr><tr><td>Relightable 3DGS</td><td>Rotation, Intensity, Chromatic Tinging</td></tr></table>

To bridge the reality gap and ensure robust zero-shot transfer, we employ a comprehensive domain randomization strategy that perturbs dynamics, perception, and photometric parameters during training, as detailed in Table II. To mimic the imperfections of physical hardware, we introduce Gaussian noise into state estimates (position, velocity, and height) and simulate actuator dynamics through action noise, variable control intervals, and stochastic latency. Perception robustness is further enforced by applying random perturbations to the cameraâs extrinsic pose to account for calibration errors. Crucially, addressing the visual domain gap, we leverage our Relightable 3D Gaussian Splatting framework to go beyond standard 2D augmentations. By explicitly manipulating the global spherical harmonic coefficients within the scene representation, we randomize the illumination direction, intensity, and spectral properties, effectively synthesizing diverse and physically consistent outdoor lighting conditions that prepare the agent for unstructured real-world environments.

Real-world Implementation Details. For real-world validation, we employed a custom-built lightweight quadrotor platform. The hardware configuration was designed to prioritize low weight and onboard computational capability for autonomous vision-based flight. The core flight controller runs the PX4 open-source autopilot stack. For perception and high-level decision-making, the platform is equipped with an NVIDIA Jetson Orin NX onboard computer. The primary exteroceptive sensor is a monocular RGB camera operating at 60 frames per second. An inertial measurement unit (IMU) integrated with a GNSS receiver provides state estimation; this fused positioning and attitude data are refined in real-time via an extended Kalman filter (EKF) [49] to supply robust odometry. No LiDAR, depth cameras, or other active ranging sensors were used, ensuring the system relies purely on passive monocular vision.

## B. Simulation Results

Photorealistic Domain Adaptation via Relightable 3D Gaussian Splatting. Outdoor environments are characterized by high-frequency photometric variations, presenting a formidable challenge for vision-based navigation. Standard 3DGS reconstructions inherently entangle static environmental illumination with scene geometry, restricting the simulation to the specific lighting conditions captured during data collection. This limitation creates a significant sim-to-real domain gap, as policies trained on static illumination often fail to generalize to the dynamic lighting scenarios encountered in physical deployment.

<!-- image-->  
Fig. 5: Simulation training performance evolution across two stages. The top panel illustrates the mean reward, while the bottom panel displays the navigation success rate over simulation steps. The vertical gray dashed line marks the curriculum transition from Stage 1 (Baseline training, blue curves) to Stage 2 (training with Domain Adaptation, red curves) at approximately 1.6M steps.

To systematically bridge this gap, we integrate our Relightable 3D Gaussian Splatting module into the training pipeline. This mechanism allows for the synthesis of diverse, photorealistic illumination conditions while strictly preserving the underlying geometric consistency. As demonstrated in Fig. 4, starting from a baseline reconstruction under natural light (a), our approach generates physically plausible variations ranging from diffuse overcast skies (b) to cool-toned dusk (c) and warm-toned morning sunlight (d). By augmenting the training curriculum with this continuous spectrum of synthesized lighting, we compel the navigation policy to learn robust visual representations that are invariant to photometric shifts, thereby significantly enhancing its zero-shot transfer capability.

Two-Stage Training and Domain Adaptation Performance. Figure 5 presents the learning progress of our navigation policy, quantified by the mean reward and the navigation success rate throughout the simulation training process. The training follows a two-stage curriculum, delineated by a vertical dashed line at approximately 1.6M simulation steps.

Stage 1: Baseline Policy Training. In the initial stage (blue curves), the policy is trained in the standard 3DGS simulator without domain adaptation. The results show rapid initial learning, with the mean reward converging to a plateau around 120 and the navigation success rate stabilizing near 90%. This baseline phase establishes a competent policy for the original, static environment.

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->

Fig. 6: Real-world flight trajectories across multiple unstructured forest environments. Each subplot shows a successful navigation trial from a distinct location, with the droneâs path overlaid in color. The trajectories demonstrate the policyâs ability to generalize to various cluttered scenes and execute collision-free navigation.  
<!-- image-->  
(a) Sunshine

<!-- image-->  
(b) Overcast

<!-- image-->  
(c) Twilight  
Fig. 7: Flight trajectory comparison under different natural illumination conditions. Each column represents trials conducted under (a) Sunshine, (b) Overcast, and (c) Twilight lighting in the same forest area. The consistent, goal-directed paths across all three conditions demonstrate the illumination robustness achieved through our domain adaptation method.

TABLE III: Quantitative comparison with RL-based UAV navigation frameworks. We evaluate methods based on sensor modality, input observation, deployment environment, and illumination adaptation capability. The symbol \* denotes results reported in the simulation.
<table><tr><td>Method</td><td>Sensor</td><td>Observation</td><td>Environment</td><td>Max Speed</td><td>Success Rate</td><td>Random Light</td></tr><tr><td>FPCRL [5]</td><td>LiDAR</td><td>Point Cloud</td><td>Indoor &amp; Outdoor</td><td>3.0 m/s</td><td>80%</td><td>-</td></tr><tr><td>MAVRL [6]</td><td>RGB-D Camera</td><td>Depth Image</td><td>Indoor</td><td>5.5 m/s*</td><td>60%*</td><td></td></tr><tr><td>NavRL [7]</td><td>RGB-D Camera</td><td>Depth Image</td><td>Indoor</td><td>2.0 m/s*</td><td>80%*</td><td></td></tr><tr><td>Reinforcement [24]</td><td>RGB-D Camera</td><td>Depth Image</td><td>Indoor</td><td>1.2 m/s</td><td>70%</td><td></td></tr><tr><td>Differentiable [20]</td><td>RGB-D Camera</td><td>Depth Image</td><td>Indoor &amp; Outdoor</td><td>7-23m/s</td><td>90%</td><td></td></tr><tr><td>Learning [28]</td><td>RGB-D Camera</td><td>3D Occupancy</td><td>Indoor &amp; Outdoor</td><td>3.5 m/s</td><td>-</td><td>-</td></tr><tr><td>D3QN [22]</td><td>Monocular</td><td>Depth Image</td><td>Indoor</td><td>0.4 m/s</td><td>-</td><td>X</td></tr><tr><td>Seeing [27]</td><td>Monocular</td><td>Optical Flow</td><td>Indoor &amp; Outdoor</td><td>6.0 m/s</td><td>60%</td><td>X</td></tr><tr><td>CAD2RL [50]</td><td>Monocular</td><td>RGB</td><td>Indoor</td><td>0.2 m/s*</td><td>40%*</td><td>X</td></tr><tr><td>Flying [15]</td><td>Monocular</td><td>RGB</td><td>Indoor</td><td>2.0 m/s</td><td>80%</td><td>X</td></tr><tr><td>Splat-Nav [14]</td><td>Monocular</td><td>RGB</td><td>Indoor</td><td>1.5 m/s</td><td>99%</td><td>Ã</td></tr><tr><td>Ours</td><td>Monocular</td><td>RGB</td><td>Outdoor</td><td>10.0 m/s</td><td>80%</td><td>â</td></tr></table>

Stage 2: Domain Adaptation Integration. In the second stage (red curves), we activate our full suite of Domain Adaptation (DA) techniques, including the proposed Relightable 3D Gaussian Splatting. Introducing these environmental variations initially causes a predictable, transient performance dip as the policy adapts to the increased complexity. Crucially, the policy recovers and then surpasses its previous performance, with the success rate exceeding the 90% target threshold. The final mean reward also reaches a higher plateau compared to Stage 1. This progression validates our two-stage curriculum:

a stable policy is first established in a consistent environment, and then its robustness and generalization are systematically enhanced through diversified training facilitated by domain adaptation.

## C. Real-world Flight

The core objective of our framework is to achieve robust, high-speed autonomous navigation in complex, unstructured outdoor environments using only monocular vision. To this end, we deployed our policy, trained exclusively within the 3DGS simulator with Relightable 3D Gaussian Splatting, onto our custom quadrotor platform for extensive real-world validation.

<!-- image-->  
Fig. 8: Qualitative visualization of learned spatial attention maps overlaid on input RGB frames. Warmer colors (red/orange) indicate regions with higher activation weights within the feature extractor. The examples demonstrate that the policy consistently focuses on salient obstacles critical for collision avoidance.

As illustrated in Fig. 6, we conducted flight tests across multiple distinct forest environments. Each subfigure depicts a complete navigation trial, overlaying the droneâs estimated trajectory onto the scene. The trajectories demonstrate the policyâs ability to successfully plan and execute collisionfree paths through dense clutter, navigating around trees and through narrow gaps to reach designated target points.

A key claim of our work is the policyâs robustness to significant illumination changes. Fig. 7 validates this claim by comparing flight trajectories under three challenging lighting conditions: Sunshine, Overcast, and Twilight. The trajectories remain consistently smooth and goal-directed across all conditions. In the bright sunshine condition, the policy successfully manages high-contrast shadows that could be mistaken for obstacles. Under the uniform lighting of overcast skies, it maintains precise navigation. Most notably, in the low-light, high-dynamic-range twilight setting, the policy continues to operate reliably.

To benchmark our system against RL-based navigation frameworks, we present a quantitative comparison in Table III. While prior learning-based methods are predominantly restricted to indoor environments, rely on active depth sensors (e.g., RGB-D, LiDAR), or lack explicit mechanisms for photometric adaptation, our method achieves a breakthrough in unstructured outdoor settings using a passive monocular camera. Distinctively, our policy supports robust autonomous flight at speeds up to 10.0 m/s under varying illumination, significantly outperforming existing baselines and validating the efficacy of our relightable Sim-to-Real pipeline.

## D. Ablation Study

Visual Feature Learning. To elucidate the perceptual strategies acquired by our end-to-end monocular vision policy, we perform an introspective analysis of its convolutional feature extractor. By computing and visualizing spatial attention maps from key intermediate network layers, we identify which regions within the input RGB frame are prioritized during the policyâs decision-making process. These attention heatmaps highlight the image areas that yield the strongest activations, effectively revealing the focus of the model.

TABLE IV: Ablation study on Relightable 3D Gaussian Splatting for real-world domain adaptation. The table reports realworld navigation success rates (successful trials / total trials) in unstructured outdoor scenes under three distinct illumination conditions at a target speed of 10 m/s.
<table><tr><td>Relightable 3DGS</td><td>Sunshine</td><td>Overcast</td><td>Twilight</td></tr><tr><td rowspan="2">â</td><td>6/10</td><td>9/10</td><td>3/10</td></tr><tr><td>8/10</td><td>10/10</td><td>8/10</td></tr></table>

As shown in Fig. 8, overlaying the attention maps onto the original images provides a qualitative understanding of the policyâs perceptual priorities. The visualizations consistently demonstrate that the network learns to concentrate on structural features directly relevant to safe navigation: the boundaries of nearby obstacles such as trees and branches, as well as openings that indicate traversable corridors. Importantly, this implicit obstacle awareness emerges purely through reinforcement learning, without any direct geometric or segmentation supervision. This result confirms that our pipeline successfully distills essential navigational cues from raw visual input, validating that the learned visual representations are both meaningful and functionally aligned with the task of highspeed collision avoidance in clutter.

Relightable 3D Gaussian Splatting Domain Adaptation. The ultimate test of our domain adaptation strategy is its ability to improve real-world flight robustness under natural illumination changes. We conducted field trials in three distinct outdoor lighting conditions to evaluate our navigation policy, comparing performance with and without Relightable 3D Gaussian Splatting.

As shown in Table IV, we tested the policy navigating unstructured forest terrain at 10 m/s under bright sunshine (with strong directional shadows), overcast skies (with soft diffuse illumination), and dusk (with low ambient illumination). The baseline policy, trained using only the static illumination from the original 3DGS reconstruction, showed clear sensitivity to these variations. Its performance was notably unstable, with a high failure rate in high-contrast sunshine and near-complete failure in the perceptually challenging twilight condition.

In contrast, the policy enhanced with our Relightable 3D Gaussian Splatting domain adaptation demonstrated consistent and superior robustness. Exposure to a synthetically diversified spectrum of lighting conditions during training enabled the policy to learn an illumination-invariant representation of the environment. This is evidenced by sustained high success rates across all tested conditions, with the most significant improvement observed in the challenging dusk scenario. These real-world results confirm that our Relightable 3D Gaussian Splatting method is a critical component for enabling reliable, high-speed vision-based navigation in the face of natural photometric variance.

## VI. CONCLUSION

In this paper, we present a novel end-to-end framework to address the challenge of high-speed monocular UAV navigation in unstructured outdoor environments. To bridge the critical visual sim-to-real gap, we leverage a photorealistic simulation environment built upon our novel Relightable 3D Gaussian Splatting representation, which allows for controllable and physically consistent illumination synthesis. By training within this high-fidelity environment, the reinforcement learning policy is able to acquire robust visual representations that generalize across lighting changes. Consequently, we enable effective zero-shot transfer to the physical world. Experimental results demonstrate successful, collision-free flight at speeds up to 10 m/s in cluttered forest scenarios under diverse lighting conditions, such as strong sunlight and twilight. Ultimately, our work demonstrates the feasibility of achieving agile and perception-aware autonomy on lightweight aerial platforms in the real world.

## REFERENCES

[1] Amina Khan, Sumeet Gupta, and Sachin Kumar Gupta. Emerging uav technology for disaster detection, mitigation, response, and preparedness. Journal of Field Robotics, 39(6):905â955, 2022.

[2] Arman Nedjati, Bela Vizvari, and Gokhan Izbirak. Post-earthquake response by small uav helicopters. Natural Hazards, 80(3):1669â1688, 2016.

[3] Yi Lu, Dominique Macias, Zachary S Dean, Nicole R Kreger, and Pak Kin Wong. A uav-mounted whole cell biosensor system for environmental monitoring applications. IEEE transactions on nanobioscience, 14(8):811â817, 2015.

[4] Carlos A Trasvina-Moreno, Rub Ë en Blasco, Â´ Alvaro Marco, Roberto Â´ Casas, and Armando Trasvina-Castro. Unmanned aerial vehicle basedË wireless sensor network for marine-coastal environment monitoring. Sensors, 17(3):460, 2017.

[5] Guangtong Xu, Tianyue Wu, Zihan Wang, Qianhao Wang, and Fei Gao. Flying on point clouds with reinforcement learning. arXiv preprint arXiv:2503.00496, 2025.

[6] Hang Yu, Christophede Wagter, and Guido CH E de Croon. Mavrl: Learn to fly in cluttered environments with varying speed. IEEE Robotics and Automation Letters, 10(2):1441â1448, 2024.

[7] Zhefan Xu, Xinming Han, Haoyu Shen, Hanyu Jin, and Kenji Shimada. Navrl: Learning safe flight in dynamic environments. IEEE Robotics and Automation Letters, 2025.

[8] Elia Kaufmann, Leonard Bauersfeld, Antonio Loquercio, Matthias Muller, Vladlen Koltun, and Davide Scaramuzza. Champion-level drone Â¨ racing using deep reinforcement learning. Nature, 620(7976):982â987, 2023.

[9] Yunlong Song, Angel Romero, Matthias Muller, Vladlen Koltun, and Â¨ Davide Scaramuzza. Reaching the limit in autonomous racing: Optimal control versus reinforcement learning. Science Robotics, 8(82):eadg1462, 2023.

[10] Abhik Singla, Sindhu Padakandla, and Shalabh Bhatnagar. Memorybased deep reinforcement learning for obstacle avoidance in uav with limited environment knowledge. IEEE transactions on intelligent transportation systems, 22(1):107â118, 2019.

[11] Shital Shah, Debadeepta Dey, Chris Lovett, and Ashish Kapoor. Airsim: High-fidelity visual and physical simulation for autonomous vehicles. In Field and service robotics: Results of the 11th international conference, pages 621â635. Springer, 2017.

[12] Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza. Flightmare: A flexible quadrotor simulator. In Conference on Robot Learning, pages 1147â1157. PMLR, 2021.

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Â¨ Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):139â1, 2023.

[14] Timothy Chen, Ola Shorinwa, Joseph Bruno, Aiden Swann, Javier Yu, Weijia Zeng, Keiko Nagami, Philip Dames, and Mac Schwager. Splatnav: Safe real-time robot navigation in gaussian splatting maps. IEEE Transactions on Robotics, 2025.

[15] Xijie Huang, Jinhan Li, Tianyue Wu, Xin Zhou, Zhichao Han, and Fei Gao. Flying in clutter on monocular rgb by learning in 3d radiance fields with domain adaptation. arXiv preprint arXiv:2512.17349, 2025.

[16] Yunfan Ren, Fangcheng Zhu, Guozheng Lu, Yixi Cai, Longji Yin, Fanze Kong, Jiarong Lin, Nan Chen, and Fu Zhang. Safety-assured high-speed navigation for mavs. Science Robotics, 10(98):eado6187, 2025.

[17] Xin Zhou, Zhepei Wang, Hongkai Ye, Chao Xu, and Fei Gao. Egoplanner: An esdf-free gradient-based local planner for quadrotors. IEEE Robotics and Automation Letters, 6(2):478â485, 2020.

[18] Yunfan Ren, Fangcheng Zhu, Wenyi Liu, Zhepei Wang, Yi Lin, Fei Gao, and Fu Zhang. Bubble planner: Planning high-speed smooth quadrotor trajectories using receding corridors. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 6332â6339. IEEE, 2022.

[19] Jesus Tordesillas, Brett T Lopez, Michael Everett, and Jonathan P How. Faster: Fast and safe trajectory planner for navigation in unknown environments. IEEE Transactions on Robotics, 38(2):922â938, 2021.

[20] Yuang Zhang, Yu Hu, Yunlong Song, Danping Zou, and Weiyao Lin. Learning vision-based agile flight via differentiable physics. Nature Machine Intelligence, pages 1â13, 2025.

[21] Jiaxu Xing, Leonard Bauersfeld, Yunlong Song, Chunwei Xing, and Davide Scaramuzza. Contrastive learning for enhancing robust scene transfer in vision-based agile flight. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 5330â5337. IEEE, 2024.

[22] Minwoo Kim, Jongyun Kim, Minjae Jung, and Hyondong Oh. Towards monocular vision-based autonomous flight through deep reinforcement learning. Expert Systems with Applications, 198:116742, 2022.

[23] Yunlong Song, Kexin Shi, Robert Penicka, and Davide Scaramuzza. Learning perception-aware agile flight in cluttered environments. arXiv preprint arXiv:2210.01841, 2022.

[24] Mihir Kulkarni and Kostas Alexis. Reinforcement learning for collisionfree flight exploiting deep collision encoding. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 15781â 15788. IEEE, 2024.

[25] Ahmed Hussein, Mohamed Medhat Gaber, Eyad Elyan, and Chrisina Jayne. Imitation learning: A survey of learning methods. ACM Computing Surveys (CSUR), 50(2):1â35, 2017.

[26] Alessandro Giusti, JerÂ´ ome Guzzi, Dan C CiresÂ¸an, Fang-Lin He, Juan P Ë RodrÂ´Ä±guez, Flavio Fontana, Matthias Faessler, Christian Forster, Jurgen Â¨ Schmidhuber, Gianni Di Caro, et al. A machine learning approach to visual perception of forest trails for mobile robots. IEEE Robotics and Automation Letters, 1(2):661â667, 2015.

[27] Yu Hu, Yuang Zhang, Yunlong Song, Yang Deng, Feng Yu, Linzuo Zhang, Weiyao Lin, Danping Zou, and Wenxian Yu. Seeing through pixel motion: learning obstacle avoidance from optical flow with one camera. IEEE Robotics and Automation Letters, 2025.

[28] Guangyu Zhao, Tianyue Wu, Yeke Chen, and Fei Gao. Learning speed adaptation for flight in clutter. IEEE Robotics and Automation Letters, 9(8):7222â7229, 2024.

[29] Qianzhong Chen, Naixiang Gao, Suning Huang, JunEn Low, Timothy Chen, Jiankai Sun, and Mac Schwager. Grad-nav++: Vision-language model enabled visual drone navigation with gaussian radiance fields and differentiable dynamics. arXiv preprint arXiv:2506.14009, 2025.

[30] Cora A Dimmig, Giuseppe Silano, Kimberly McGuire, Chiara Gabellieri, Wolfgang Honig, Joseph Moore, and Marin Kobilarov. Survey of Â¨ simulators for aerial robots: An overview and in-depth systematic comparisons [survey]. IEEE Robotics & Automation Magazine, 32(2):153â 166, 2024.

[31] Fadri Furrer, Michael Burri, Markus Achtelik, and Roland Siegwart. Rotorsâa modular gazebo mav simulator framework. In Robot Operating System (ROS) The Complete Reference (Volume 1), pages 595â625. Springer, 2016.

[32] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026â5033. IEEE, 2012.

[33] PX4 Development Team. Gazebo simulation, 2024. PX4 Autopilot Documentation.

[34] ArduPilot Development Team. Using sitl with gazebo, 2024. ArduPilot Documentation.

[35] NVIDIA Corporation. Nvidia isaac sim, 2024. NVIDIA Developer Documentation.

[36] Mihir Kulkarni, Theodor JL Forgaard, and Kostas Alexis. Aerial gymâ isaac gym simulator for aerial robots. arXiv preprint arXiv:2305.16510, 2023.

[37] Xian Ge, Yuling Pan, Yuhang Zhang, Xiang Li, Weijun Zhang, Dizhe Zhang, Zhaoliang Wan, Xin Lin, Xiangkai Zhang, Juntao Liang, et al. Airsim360: A panoramic simulation platform within drone view. arXiv preprint arXiv:2512.02009, 2025.

[38] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

[39] Michal Adamkiewicz, Timothy Chen, Adam Caccavale, Rachel Gardner, Preston Culbertson, Jeannette Bohg, and Mac Schwager. Vision-only robot navigation in a neural radiance world. IEEE Robotics and Automation Letters, 7(2):4606â4613, 2022.

[40] Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, et al. Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. arXiv preprint arXiv:2502.13144, 2025.

[41] James Tobin. On limiting the domain of inequality. The Journal of Law and Economics, 13(2):263â277, 1970.

[42] Konstantinos Bousmalis, Nathan Silberman, David Dohan, Dumitru Erhan, and Dilip Krishnan. Unsupervised pixel-level domain adaptation with generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3722â 3731, 2017.

[43] Philip TG Jackson, Amir Atapour Abarghouei, Stephen Bonner, Toby P Breckon, and Boguslaw Obara. Style augmentation: data augmentation via style randomization. In CVPR workshops, volume 6, pages 10â11, 2019.

[44] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. Relightable 3d gaussians: Realistic point cloud relighting with brdf decomposition and ray tracing. In European Conference on Computer Vision, pages 73â89. Springer, 2024.

[45] Hongze Chen, Zehong Lin, and Jun Zhang. Gi-gs: Global illumination decomposition on gaussian splatting for inverse rendering. arXiv preprint arXiv:2410.02619, 2024.

[46] Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian Reid, Philip Torr, and Victor Adrian Prisacariu. Gaussctrl: Multi-view consistent text-driven 3d gaussian splatting editing. In European Conference on Computer Vision, pages 55â71. Springer, 2024.

[47] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.

[48] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[49] Maria Isabel Ribeiro. Kalman and extended kalman filters: Concept, derivation and properties. Institute for Systems and Robotics, 43(46):3736â3741, 2004.

[50] Fereshteh Sadeghi and Sergey Levine. Cad2rl: Real single-image flight without a single real image. arXiv preprint arXiv:1611.04201, 2016.