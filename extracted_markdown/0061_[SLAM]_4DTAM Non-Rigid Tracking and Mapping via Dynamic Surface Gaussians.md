# 4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians

Hidenobu Matsuki

Gwangbin Bae

Andrew J. Davison

Dyson Robotics Laboratory, Imperial College London {h.matsuki20, g.bae, a.davison}@imperial.ac.uk

Website: https://muskie82.github.io/4dtam/ Video: https://youtu.be/MRGhggLmTF0?si=51bqfAe9pYQNWgf-/

<!-- image-->  
Figure 1. 4DTAM jointly estimates camera-egomotion, appearance, geometry and scene dynamics without any template.

## Abstract

We propose the first 4D tracking and mapping method that jointly performs camera localization and non-rigid surface reconstruction via differentiable rendering. Our approach captures 4D scenes from an online stream of color images with depth measurements or predictions by jointly optimizing scene geometry, appearance, dynamics, and camera ego-motion. Although natural environments exhibit complex non-rigid motions, 4D-SLAM remains relatively underexplored due to its inherent challenges; even with 2.5D signals, the problem is ill-posed because of the high dimensionality of the optimization space. To overcome these challenges, we first introduce a SLAM method based on Gaussian surface primitives that leverages depth signals more effectively than 3D Gaussians, thereby achieving accurate surface reconstruction. To further model non-rigid deformations, we employ a warp-field represented by a multi-layer perceptron (MLP) and introduce a novel camera pose estimation technique along with surface regularization terms that facilitate spatio-temporal reconstruction. In addition to these algorithmic challenges, a significant hurdle in 4D SLAM research is the lack of reliable ground truth and evaluation protocols, primarily due to the difficulty of

4D capture using commodity sensors. To address this, we present a novel open synthetic dataset of everyday objects with diverse motions, leveraging large-scale object models and animation modeling. In summary, we open up the modern 4D-SLAM research by introducing a novel method and evaluation protocols grounded in modern vision and rendering techniques.

## 1. Introduction

The world we live in has many moving elements. Rivers flow, trees sway, cookies crumble, and humans walk. Although Simultaneous Localization and Mapping (SLAM) methods which assume that most of the world is static are highly useful, embodied agents which aim to navigate and interact with their environments in the most general way should be able to operate in dynamic scenes. There are several ways to segment and ignore moving scene elements, and a SLAM system can be assembled by integrating these individual modules so that it can reconstruct the static parts of a scene and estimate camera ego-motion. However, in this work, we aim for a more comprehensive spatiotemporal (4D) reconstruction of scenes exhibiting significant dynamic motion. Our primary focus is on a unified framework that leverages intrinsic capabilities of the underlying scene representation without heavily relying on prior assumptions about moving elements. 4D-SLAM with general scene motion is difficult primarily because of the complex and high-dimensional nature of modeling non-rigid motions (and potential topological changes) while simultaneously optimizing the pose of a moving camera. There is much more redundancy than in rigid SLAM, and some prior assumptions are needed to combat this. Another challenge lies in the lack of datasets to train and/or evaluate techniques. Recent advances in computer vision and graphics make it a good time to revisit this problem. New 3D representations (e.g. neural fields and Gaussian splats) allow differentiable rendering of complex 3D scenes and optimization via 2D observations, and to model deformation fields smoothly without more specific assumptions. Also, the availability of high-quality 3D meshes on Internet and rendering software (e.g. Blender) gives the ability to render non-rigidly moving objects with ground truth.

We present 4DTAM, a novel approach for 4D Tracking And Mapping in dynamic scenes. We use Gaussian surface primitives to represent the scene and introduce a neural warp-field represented by a multi-layer perception (MLP) to model continuous temporal changes. We then utilize differentiable rendering to jointly optimize the scene geometry, appearance, dynamics, and camera ego-motion from an online stream of a single RGB-D camera. This enables accurate 3D reconstruction and real-time rendering, even in the presence of complex non-rigid deformations. To facilitate future research, we also introduce a new synthetic dataset of dynamic objects. Our focus in this dataset is realistic, complex motion of scenes that are not well represented by existing deformable object models. Animated 3D meshes are rendered and the ground truth depth, surface normals, and foreground masks are extracted together with the camera poses/intrinsics. This dataset provides challenging scenarios for 4D reconstruction methods. We also release the full rendering script to allow the generation of custom 4D datasets. Our experimental results demonstrate that 4DTAM achieves good performance in both camera tracking and scene reconstruction in the presence of dynamic objects. It can handle the complex motion of articulated objects (e.g., drawers) and non-rigid objects (e.g., curtains, flags, and animals), showcasing its potential for applications in robotics, augmented reality, and other fields requiring real-time dynamic scene understanding. We primarily use RGB-D sensor input, but also demonstrate an extension to monocular RGB streams by incorporating a monocular depth prediction network in the supplementary material.

In summary, the contributions of this paper are:

â¢ 4DTAM, the first 4D tracking and mapping method that uses differentiable rendering and Gaussian surface primitives for dynamic environments.

â¢ The first 2DGS [17]-based SLAM method with analytic camera pose gradients, normal initialization, and regularization to fully exploit depth signals.

â¢ An MLP-based warp-field for modeling non-rigid scene, complemented by a novel camera localization technique and rigidity regularization of surface Gaussians.

â¢ A novel 4D-SLAM dataset with complex object motions, ground-truth camera trajectories, and dynamic object meshes, along with an evaluation protocol.

â¢ Extensive evaluations demonstrating that the method achieves state-of-the-art performance.

## 2. Related Work

## 2.1. Visual SLAM

Visual SLAM has been an extensively researched field, with Dense SLAM specifically focusing on capturing detailed scene geometry [35] and semantics [30]. A central aspect of these methods lies in the choice of scene representation and the corresponding optimization framework. Dense SLAM methods based on traditional scene representations, such as volumetric Truncated Signed Distance Functions (TSDF) [22, 34, 60] or Surfels [44, 61], project 2D observations into 3D space and employ specific data fusion algorithms. While effective, these methods often fail to keep consistency between the model and sensor observations across multiple viewpoints, posing challenges for long-term operation.

However, recent advancements in graphics hardware have facilitated the adoption of differentiable rendering frameworks, which have revolutionized inverse rendering and scene reconstruction [23, 31, 33, 37]. Differentiable rendering ensures multi-view consistency through streamlined backpropagation, enhancing scene reconstruction accuracy. Notably, 3D Gaussian Splatting (3DGS) [25] has gained attention due to its flexible resource allocation and rapid forward rendering capabilities. Initially developed for photorealistic view synthesis, recent research has extended its application to surface reconstruction [15, 66]. Enhanced methods, such as 2D Gaussian Splatting (2DGS) [17], achieve superior geometry reconstruction by reducing the Gaussian dimension and explicitly defining surface normals. These differentiable rendering representations have been applied to visual SLAM, from coordinatebased MLPs [52] to explicit voxel grids [21, 55, 63, 67], points [43], and 3D Gaussians [24, 29, 62].

## 2.2. SLAM for 4D Scene Reconstruction

3D reconstruction of dynamic scenes has been extensively studied, with notable achievements using optimization methods, even for unknown non-rigid objects observed by a single moving RGB camera [14, 53]. However, these approaches typically require batch optimization and are limited to smaller scenes. In contrast, dynamic SLAM targets incremental, reconstruction and tracking of large, continuously moving scenes ideally in real-time. Most methods to date have relied on RGB-D data from moving depth cameras.

While many methods detect and exclude dynamic objects to focus on static scene reconstruction [45], full spatiotemporal reconstruction (which we refer to as 4D-SLAM) requires more advanced solutions. For instance, tracking and reconstructing rigid moving objects separately [42] or employing parametric shape models for known semantic classes like humans or animals [26] are effective strategies. Specialized domains, such as endoscopic imaging, have utilized scene-specific priors or deformation models to handle non-rigid dynamics [28, 41].

An incremental 4D-SLAM for general dynamic scenes has remained more challenging, but has been addressed based on various regularizing assumptions and representations. DynamicFusion [36] pioneered a line of work [13, 19, 47, 48] which captures temporal evolution in the scene geometry by jointly optimizing a canonical volumetric representation (e.g., TSDF volume [36]) and a deformation field. As the solution space is extremely high-dimensional, additional constraints are often introduced to regularize the motion field [47, 48] or to align visual features [4, 19]. Recent advances in 3D representations, such as neural fields and Gaussian primitives, have opened new possibilities for dynamic scene reconstruction. Canonical radiance and motion fields can be jointly optimized via differentiable rendering, as demonstrated with NeRF [38, 40, 54] and SDF [7, 56]. For 3D Gaussians, which can explicitly represent points, motion can be estimated either through perprimitive trajectories [27] or learnable motion bases [57]. However, warp-field-based motion representation offers inherent smoothness regularization, leveraging the properties of neural fields [10, 18, 64, 65]. Most existing methods, however, rely on known camera poses or multi-camera setups to capture dense spatiotemporal observations. While DyNoMo [46] supports camera pose optimization, its 3D Gaussian representation is not suited for geometrically accurate reconstruction. In contrast, our 4DTAM framework enables 4D reconstruction using a single RGB-D camera, jointly optimizing camera poses, appearance, geometry, and dynamics, making it practical for most embodied agents.

## 2.3. Datasets for 4D Reconstruction

4D reconstruction has been studied extensively for the case of the human body. Datasets like Human3.6M [20], Deep-Cap [16], and ZJU-MoCap [39] capture diverse human motions under a multi-camera setup. The cameras are fixed, synchronized, and calibrated to reduce the difficulty in establishing dense multi-view correspondences. Only a small number of datasets provide single-stream RGB-

D sequences captured from a moving camera [5, 12, 47]. Recovering the camera poses is not trivial for such realworld captures, and additional post-processing (e.g. robust depth map alignment [56]) is required. Another challenge lies in ground truth acquisition. Besides the depth measurements, other ground truths (e.g., scene flow, object mask) often require manual labeling. On the contrary, synthetic datasets [6, 59] provide perfect ground truths. Recent advances in open-source datasets [9] and rendering software [8] also close the synthetic-to-real domain gap significantly. To this end, we introduce a new high-quality synthetic dataset tailored for 4D reconstruction and camera pose estimation.

## 3. Method

## 3.1. 2D Gaussian Splatting

Our geometric scene representation is based on 2D Gaussian Splatting (2DGS) [17]. Unlike 3D Gaussian Splatting (3DGS), which uses blob-like splats, 2DGS functions as a stretchable surfel with explicitly defined surface normal directions. This property makes 2DGS particularly wellsuited for non-rigid scene reconstruction with a single camera, where effectively handling 2.5D input signals is critical.

Each 2D Gaussian G is represented by its 3D mean position $\mathbf { P } _ { \mu } ,$ rotation $\mathbf { R } \in S O ( 3 )$ , color $\mathbf { c } ,$ opacity $^ { O , }$ and a scaling vector $\mathbf { S } \in \mathbb { R } ^ { 2 }$ . The rotation matrix R is decomposed as $\mathbf { R } = [ \mathbf { t } _ { u } , \mathbf { t } _ { v } , \mathbf { t } _ { w } ]$ , where $\mathbf { t } _ { u }$ and $\mathbf { t } _ { v }$ represent two principal tangential vectors, and $\mathbf { t } _ { w }$ is the normal vector, defined as $\mathbf { t } _ { w } = \mathbf { t } _ { u } \times \mathbf { t } _ { v }$ . For simplicity, spherical harmonics are omitted in this work.

The 2D Gaussian function is parameterized on the local tangent plane in world space as:

$$
P ( u , v ) = \mathbf { P } _ { \mu } + s _ { u } \mathbf { t } _ { u } u + s _ { v } \mathbf { t } _ { v } v = \mathbf { H } ( u , v , 1 , 1 ) ^ { \mathrm { T } }\tag{1}
$$

$$
\mathrm { w h e r e } \mathbf { H } = \left[ { \begin{array} { c c c c } { s _ { u } \mathbf { t } _ { u } } & { s _ { v } \mathbf { t } _ { v } } & { \mathbf { 0 } } & { \mathbf { p } _ { k } } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} } \right] = \left[ { \begin{array} { c c } { \mathbf { R } \mathbf { S } } & { \mathbf { p } _ { k } } \\ { \mathbf { 0 } } & { 1 } \end{array} } \right]\tag{2}
$$

For a point $\textbf { u } = \mathbf { \Omega } ( u , v )$ in the tangential plane of 2D Gaussian (uv space), its projection onto the image plane is given by

$$
\mathbf { x } = ( x z , y z , z , 1 ) ^ { \mathrm { T } } = \mathbf { W P } ( u , v ) = \mathbf { W H } ( u , v , 1 , 1 ) ^ { \mathrm { T } }\tag{3}
$$

where $\mathbf { W } \in \mathbb { R } ^ { 4 \times 4 }$ is the transformation matrix from world space to screen space.

To avoid numerically unstable matrix inversion of $\mathbf { M } =$ $( \mathbf { W } \mathbf { H } ) ^ { - 1 }$ , 2DGS applies ray-splat intersection by finding the intersection of non-parallel planes (x-plane and yplane). The ray $\mathbf { x } = ( x , y )$ is determined by the intersection of the x-plane $\mathbf { h } _ { x }$ and the y-plane $\mathbf { h } _ { y } ,$ , represented as $\mathbf { h } _ { x } = ( - 1 , 0 , 0 , x ) ^ { \mathrm { T } }$ and $\mathbf { h } _ { y } = ( 0 , - 1 , 0 , \bar { y } ) ^ { \mathrm { T } }$ , respectively. In the uv coordinates of the 2D Gaussian, this is expressed

<!-- image-->  
Figure 2. Method overview of 4DTAM.

as:

$$
\mathbf { h } _ { u } = ( \mathbf { W } \mathbf { H } ) ^ { \mathrm { T } } \mathbf { h } _ { x } \quad \mathrm { a n d } \quad \mathbf { h } _ { v } = ( \mathbf { W } \mathbf { H } ) ^ { \mathrm { T } } \mathbf { h } _ { y }\tag{4}
$$

The intersection point meets the following condition,

$$
\mathbf { h } _ { u } \cdot ( u , v , 1 , 1 ) ^ { \mathrm { T } } = \mathbf { h } _ { v } \cdot ( u , v , 1 , 1 ) ^ { \mathrm { T } } = 0\tag{5}
$$

This leads to an solution for the intersection point u(x):

$$
u ( \mathbf { x } ) = \frac { \mathbf { h } _ { u } ^ { 2 } \mathbf { h } _ { v } ^ { 4 } - \mathbf { h } _ { u } ^ { 4 } \mathbf { h } _ { v } ^ { 2 } } { \mathbf { h } _ { u } ^ { 1 } \mathbf { h } _ { v } ^ { 2 } - \mathbf { h } _ { u } ^ { 2 } \mathbf { h } _ { v } ^ { 1 } } \qquad v ( \mathbf { x } ) = \frac { \mathbf { h } _ { u } ^ { 4 } \mathbf { h } _ { v } ^ { 1 } - \mathbf { h } _ { u } ^ { 1 } \mathbf { h } _ { v } ^ { 4 } } { \mathbf { h } _ { u } ^ { 1 } \mathbf { h } _ { v } ^ { 2 } - \mathbf { h } _ { u } ^ { 2 } \mathbf { h } _ { v } ^ { 1 } }\tag{6}
$$

where $\mathbf { h } _ { u } ^ { i } , \mathbf { h } _ { v } ^ { i }$ are the i-th parameter of the 4D homogeneous plane parameters.

The 2D Gaussian at $( u , v )$ is evaluated as:

$$
\mathcal { G } ( \mathbf { u } ) = \exp \left( - \frac { u ^ { 2 } + v ^ { 2 } } { 2 } \right)\tag{7}
$$

The 2D Gaussians are sorted along the camera ray by their center depth and organized into image tiles. Per-pixel color is rendered via volumetric alpha blending:

$$
c ( \mathbf { x } ) = \sum _ { i = 1 } \mathbf { c } _ { i } \alpha _ { i } \mathcal { G } _ { i } ( \mathbf { u } ( x ) ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } \mathcal { G } _ { j } ( \mathbf { u } ( x ) ) )\tag{8}
$$

where depth and normal can be rendered similarly.

## 3.2. Analytic Camera Pose Jacobian

One major advantage of Gaussian Splatting is its analytical formulation of gradient flow for model parameters, enabling real-time full-resolution rendering. However, it assumes posed images as input and does not provide gradients for camera poses. To accelerate optimization, we derive the analytic Jacobian of the camera pose for 2D Gaussian Splatting and implement it using a CUDA kernel. This formulation has potential applications for a wide range of tasks involving pose estimation in surface-based Gaussian Splatting.

We use Lie algebra to derive the minimal Jacobians for the camera pose matrix from the world coordinate system to the cameraâs local coordinate system, defining ${ \mathbf { } } T _ { C W } \in$ $S E ( 3 )$ and $\tau \in \mathfrak { s e } ( 3 )$ . Since 2DGS backpropagates gradients to $\mathbf { M } ^ { T } = \mathbf { W } \mathbf { H }$ during the optimization of the 3D mean, we require the partial derivative $\frac { \partial \mathbf { M } ^ { T } } { \partial \tau }$ . Let $\textbf { K } \in$ $\mathbb { R } ^ { 4 \times 4 }$ represent the camera projection matrix. Then, equation 3 is rewritten as:

$$
\mathbf { x } = \mathbf { M } ^ { T } ( u , v , 1 , 1 ) ^ { \mathrm { T } } = \mathbf { K } T _ { C W } \mathbf { H } ( u , v , 1 , 1 ) ^ { \mathrm { T } }\tag{9}
$$

Using the chain rule, the partial derivatives are computed as:

$$
\frac { \partial { \bf M } ^ { T } } { \partial \tau } = \frac { \partial { \bf M } ^ { T } } { \partial { \bf W } } \frac { \partial { \bf W } } { \partial T _ { C W } } \frac { \partial T _ { C W } } { \partial \tau } ,\tag{10}
$$

$$
\frac { \partial { \pmb T } _ { C W } } { \partial \pmb \tau } = \left[ \begin{array} { c c } { \mathbf 0 } & { - { \pmb R } _ { C W } \times \mathbf { \Sigma } } \\ { \mathbf 0 } & { - { \pmb R } _ { C W } \times \mathbf \Sigma } \\ { \mathbf 0 } & { - { \pmb R } _ { C W } \times \mathbf \Sigma , 3 } \\ { { \pmb I } } & { - \mathbf t _ { C W } \times \mathbf \Sigma } \end{array} \right]\tag{11}
$$

where $\pmb { R } _ { C W } \in S O ( 3 )$ and $\mathbf { t } _ { C W } \in \mathbb { R } ^ { 3 }$ denote the rotation and translation parts of ${ \pmb T } _ { C W }$ , respectively. The notation Ã represents the skew-symmetric matrix of a 3D vector, and $R _ { C W : , i }$ denotes the ith column of $R _ { C W }$

2DGS also renders a normal map, which can be supervised using the loss computed from the rendered normals. Let $\mathbf { n } _ { c }$ denote the camera-space normal. The normal of a 2D Gaussian in the cameraâs local coordinate system is defined as:

$$
\mathbf { n } _ { c } = \pmb { T } _ { C W } \mathbf { t } _ { w }\tag{12}
$$

where $\mathbf { t } _ { w }$ is the surface normal in the world coordinate system.

Borrowing the notation of the left Jacobian for Lie groups from [49], the partial derivative is given by:

$$
\frac { \partial \mathbf { n } _ { c } } { \partial \pmb { \tau } } = \frac { \mathcal { D } \mathbf { n } _ { c } } { \mathcal { D } \pmb { T } _ { C W } } = \left[ \pmb { I } \quad - \mathbf { n } _ { c } ^ { \times } \right]\tag{13}
$$

Further details of the derivation are provided in the supplementary material.

## 3.3. Warp Field

To model time-varying deformations, we use a warp-field represented by a coordinate-based network [64, 65]. In our hand-held single-camera setup, the limited view coverage of dynamic objects necessitates structural priors in the motion representation. For this, we employ a compact MLP as the warp-field to estimate transitions from the canonical Gaussians following [64].

Given time t and center position x of 2D Gaussians in canonical space as inputs, the deformation MLP fÎ¸ produces offsets, which subsequently transform the canonical 2D Gaussians to the deformed space:

$$
( \delta \pmb { x } , \delta \pmb { r } , \delta \pmb { s } ) = \mathbf { f } _ { \pmb { \theta } } ( \gamma _ { 1 } ( \pmb { x } ) , \gamma _ { 2 } ( t ) )\tag{14}
$$

where $\delta \pmb { x } \in \mathbb { R } ^ { 3 } , \delta \pmb { r } \in S O ( 3 ) , \delta s \in \mathbb { R } ^ { 2 }$ denotes the offsets of 2D Gaussianâs mean position, rotation and scale respectively, Î³ denotes the frequency-based positional encoding [31]. For deformable SLAM applications, we leverage a CUDA-optimized MLP implementation [33] to enable fast, interactive reconstruction.

## 3.4. Tracking and Mapping Framework

Our SLAM method follows the standard tracking and mapping architecture, where the tracking module is in charge of fast online camera pose estimation while the mapping performs a relatively more involved joint opimtization of the camera poses, geometry and motion of selected keyframes. Further details of the hyperparameters are available in the supplementary material.

## 3.4.1. Tracking

The tracking module estimates the coarse camera pose for the latest incoming frame. This is achieved by minimizing the photometric and depth rendering errors between the sensor observation and the rendering from the deformable Gaussian model. Unlike static 3DGS SLAM methods, we estimate the camera pose relative to the warped Gaussians at the latest keyframes timestamp $t _ { k f }$ , assuming the deformed scene structure at $t _ { k f }$ is closest to the current state. We define photometric rendering loss as:

$$
L _ { p } = \left\| I ( \mathcal { G } _ { c a n o } , T _ { C W } , t _ { k f } ) - \bar { I } \right\| _ { 1 }\tag{15}
$$

<!-- image-->  
Figure 3. 2D Gaussianâs Surface Normal Rendering based on Different Initialization. Left: Random initialization. Right: Our initialization aligned with sensor measurement.

Here $I ( \mathcal { G } , \pmb { T } _ { C W } )$ denotes a rendered color image from the cannonical Gaussians $\mathcal { G } _ { c a n o } .$ , timestamp of the latest keyframe $t _ { k f }$ and camera pose ${ \bf { \mathit { T } } } _ { \mathit { C W } }$ , and Â¯I is an observed image. Similarly, we also minimize geometric depth error:

$$
L _ { g } = \left\| D ( \mathcal { G } _ { c a n o } , T _ { C W } , t _ { k f } ) - \bar { D } \right\| _ { 1 }\tag{16}
$$

Following MonoGS [29], we further optimize affine brightness parameters. Keyframes are selected every N-th frame and sent to the mapping process for further refinement.

## 3.4.2. Mapping

The mapping module performs joint optimization of the camera pose, canonical Gaussians, and the warp field within a sliding window.

Gaussian Management When a new keyframe is registered, we add new Gaussians to the canonical Gaussians $\mathcal { G } _ { c a n o } ,$ based on the back-projected point cloud from the RGB-D observations. Unlike 3DGS, 2DGS explicitly encodes surface normal information in its rotation vector, making it beneficial to initialize using surface normals estimated from sensor depth measurements. To achieve this, we compute the surface normals of the current depth observation by taking the finite difference of neighboring backprojected depth points and assign them as the normal vectors of the 2D Gaussianss $\mathbf { t } _ { w }$ . This is formulated as:

$$
\mathbf { t } _ { w } = \frac { \nabla _ { x } \mathbf { p } _ { d } \times \nabla _ { y } \mathbf { p } _ { d } } { | \nabla _ { x } \mathbf { p } _ { d } \times \nabla _ { y } \mathbf { p } _ { d } | }\tag{17}
$$

where $\mathbf { p } _ { d }$ denotes points back-projected by the current sensor depth observation. We store the computed normal information as a 2D image $\mathbf { N } _ { s e n s o r }$ for normal supervision. Pruning and densification parameters follow MonoGS, which effectively prunes the wrongly inserted Gaussians in the canonical space due to the object movement.

4D Map optimization We perform joint optimization of the camera ego-motion, appearance, geometry and scene dynamics. In a single-camera setup, the lack of spatiotemporally dense observations makes fully capturing dynamic scenes challenging, as complete spatial (xyz) coverage over time (t) is only feasible with multi-camera systems. To address this, we introduce regularization terms for both shape and motion.

In addition to photometric and depth losses, we apply a normal regularization based on sensor measurements to better align 2D Gaussians. Unlike the original 2DGS methods, which compute normals by finite differences of rendered depth during every optimization stepâleading to high computational costsâwe instead propose to use normals precomputed from depth input as supervision. This reduces computational overhead, as normals are calculated only when a new keyframe is inserted:

$$
{ \cal L } _ { n } = \sum _ { i \in h \times w } ( 1 - { \bf n } _ { i } ^ { \mathrm { T } } { \bf N } _ { s e n s o r , i } )\tag{18}
$$

To constrain motion in unobserved regions, we apply an as-rigid-as possible regularization loss $L _ { A R A P }$ from [27] to the Gaussian means. Additionally, we introduce a novel surface normal rigidity loss, constraining the 2D Gaussiansâ surface normals to stay similar between timesteps $t _ { 1 }$ and $t _ { 2 } ,$ preserving local surface rigidity:

$$
\begin{array} { r } { L _ { A R A P . n } = w _ { i , j } \left. ( \mathbf { t } _ { w } ) _ { i , t _ { 1 } } ^ { T } ( \mathbf { t } _ { w } ) _ { j , t _ { 1 } } - ( \mathbf { t } _ { w } ) _ { i , t _ { 2 } } ^ { T } ( \mathbf { t } _ { w } ) _ { j , t _ { 2 } } \right. _ { 1 } } \\ { ( 1 9 ) } \end{array}
$$

where $w _ { i , j }$ is a distance-based weighting factor like $L _ { A R A P }$ . We apply ARAP regularizers between the oldest and latest keyframe in the current window.

Together with the isotropic loss $\boldsymbol { L } _ { i s o }$ proposed in [29], we minimize the following total cost function:

$$
\begin{array} { r l } & { L _ { t o t a l } = \lambda _ { p } L _ { p } + \lambda _ { g } L _ { g } + \lambda _ { n } L _ { n } } \\ & { ~ + \lambda _ { i s o } L _ { i s o } + L _ { A R A P } + L _ { A R A P , n } } \end{array}\tag{20}
$$

The optimization is based on the sliding window heuristics in [11], with two additional keyframes randomly $\mathrm { s e - }$ lected from the history.

Global Optimization Sliding window-based optimization prioritizes the latest frame, causing past keyframe information to degrade over time. After tracking, if required we can perform global optimization to finalize the map, which takes less than 1 minute on an RTX 4090. During this step, the poses and number of Gaussians are fixed, and one keyframe is randomly selected per iteration. The process uses the normal consistency loss of 2DGS, ensuring global consistency despite being relatively slow.

## 3.5. Dataset Generation

We introduce Sim4D, a new synthetic dataset for 4D reconstruction. Recently, a large number of photo-realistic, animated 3D meshes have become available [2, 9]. Combined with open-source graphics software [3, 8], such meshes provide a scalable way of generating datasets for non-rigid 4D reconstruction. The data generation pipeline is illustrated in Fig. 4.

Meshes and background. We collected 50 high-quality, animated 3D meshes from Objaverse [9] and Sketchfab [2], all of which are under CC-BY license. The collected meshes exhibit a wide variety of motions, including nonrigid deformation and topological changes. We then place the object inside a cube and randomize the background texture. Texture maps are collected from Poly Haven [1] and are all under CC0 license.

Rendering. We render 240 to 540 frames for each object. The camera trajectories are defined along arcs of 20 degrees, and test viewpoints are defined outside of these arcs to evaluate the performance of novel-view synthesis and to quantify the accuracy of the reconstructed geometry. At each timestamp, the RGB image, ground truth depth, surface normals, and foreground mask are rendered and the camera intrinsics/extrinsics saved. Please refer to the supplementary material for additional details.

## 4. Evaluation

## 4.1. Experimental Setup

We extensively evaluate our non-rigid SLAM method on both synthetic and real-world datasets. Previous non-rigid RGB-D SLAM work has primarily focused on qualitative demonstrations using limited datasets, showcasing the early-stage potential of the field. To advance research, we introduce a quantitative evaluation protocol with the new Sim4D dataset. Our evaluation covers camera pose accuracy, as well as the appearance and geometric quality of the reconstructed models. Additionally, we demonstrate realworld performance using a self-captured dataset.

While designed primarily for dynamic scenes, our method is the first to leverage surface Gaussian splatting for both static SLAM and non-rigid RGB-D reconstruction. To further validate our approach, we perform a detailed quantitative component-wise ablation analysis.

Metrics and Datasets For our main Non-Rigid SLAM evaluation, we evaluate our method on 8 sequences from the Sim4D dataset. We first report ATE RMSE for trajectory evaluation. To assess SLAM map quality, we report depth rendering error (L1 error) for geometry and PSNR, SSIM, and LPIPS for appearance evaluation. For Sim4D, metrics are calculated from test views (extrapolated positions across different timestamps). The estimated and ground truth trajectories are aligned on the first frame, and test view positions are queried in the ground truth trajectoryâs coordinate system. Details about the test viewpoints are in the supplementary material. Since SurfelWarp [13] requires explicit foreground segmentation, we collect its results only on pixels with valid reconstruction. For Static

<!-- image-->  
Animated Meshes

<!-- image-->  
Background Textures

<!-- image-->  
Dataset for 4D Reconstruction

<!-- image-->  
Figure 4. Sim4D dataset. We create a new dataset for 4D reconstruction by rendering animated 3D meshes.

SLAM ablation, we report ATE RMSE, rendering performance, and TSDF-fused mesh metrics, following the protocol in [43]. We evaluate our method on the Replica [50] dataset and the TUM RGB-D dataset [51]. To isolate the impact of scene representation from system differences, we replaced MonoGSâs representation with 2DGS while keeping all other system configurations identical. For Offline Non-Rigid Reconstruction ablation, we report the average geometry and appearance rendering metrics on subsets of the DeepDeform [5], KillingFusion [47], and iPhone datasets [12], which are used in [56]. Numerical quantities for each sequence is available in supplementary material. Since [56] primarily focuses on object shape completion, metrics are calculated only within the given segmentation mask. The camera pose is provided by the dataset, and pose optimization is disabled to focus solely on reconstruction performance. We perform 30000 iteration for training, which takes approximately 30 mins.

Baseline Methods For quantitative non-rigid SLAM evaluation, we compare our method with SurfelWarp [13], the only non-rigid RGB-D SLAM method with publicly available code. For component-wise ablation analysis, we compare against MonoGS [29] for static SLAM evaluation and Morpheus [56] for offline reconstruction.

Implementation Details Our SLAM system runs on a desktop equipped with an Intel Core i9-12900K (3.50GHz) processor and a single NVIDIA GeForce RTX 4090 GPU. The camera pose jacobian for 2DGS, described in Section 3.2, is implemented using a CUDA rasterizer, similar to other gradients in Gaussian Splatting. For real-world data capture, we used the Realsense D455.

## 4.2. Quantitative Evaluation

Table 1 compares our method with SurfelWarp [13]. Our method outperforms SurfelWarp across all metrics. To analyze this further, Fig. 5 provides qualitative visualizations and trajectory plots for the modular vehicle sequence.

<!-- image-->  
Figure 5. Qualitative comparison to SurfelWarp. Left: Rendered image, Middle: Rendered normal map, Right: Estimated camera trajectory

Since SurfelWarp relies on a foreground mask, its reconstruction lacks scene completeness. In contrast, our method reconstructs the entire scene within a joint optimization framework, providing more comprehensive coverage. Additionally, compared to SurfelWarpâs back-projection and Surfel fusion scheme, our differentiable rendering-based optimization enforces multi-view consistency over time, resulting in superior camera tracking and consistent 3D reconstruction. Our method achieves camera pose estimation at approximately 1.5 fps and completes the final global optimization in 1 minute.

## 4.3. Qualitative Evaluation

Fig. 6 presents qualitative reconstruction results on realworld dynamic scenes. Our method successfully reconstructs dynamic scenes with non-rigid deformations, whereas MonoGS fails to handle such complexities.

## 4.4. Ablation Study

Static SLAM Table 2 provides the camera ATE and 3D reconstruction evaluation results. Our 2DGS-based implementation shows competitive performance and achieves the best result in 6 out of 8 sequences for camera ATE, and consistently better result on rendering and 3D reconstruction metrics. The reconstruction is visualized in Fig 7 which shows the comparison of the mesh generated by TSDF Fusion between MonoGS and MonoGS-2D. Table 3 provides the camera ATE and rendering metrics evaluation on TUM dataset. Our method shows on par camera ATE but shows the increased geometric reconstruction quality.

<table><tr><td rowspan="2">Method</td><td>Category</td><td>Metric</td><td>curtain</td><td>flag</td><td>mercedes</td><td>modular_vehicle</td><td>rhino</td><td>shoe_rack</td><td>water_effect</td><td>wave_toy</td></tr><tr><td>Trajectory</td><td>ATE RMSE[cm]â</td><td>6.10</td><td>31.9</td><td>5.21</td><td>4.21</td><td>2.81</td><td>2.16</td><td>2.60</td><td>1.45</td></tr><tr><td rowspan="5">SurfelWarp [13]</td><td>Geometry</td><td>L1 Depth[cm]â</td><td>49.1</td><td>50.8</td><td>5.2</td><td>10.3</td><td>44.9</td><td>4.25</td><td>46.7</td><td>47.5</td></tr><tr><td></td><td>PSNR [dB] â</td><td>15.78</td><td>11.04</td><td>25.7</td><td>19.45</td><td>16.76</td><td>26.3</td><td>17.3</td><td>16.4</td></tr><tr><td>Appearance</td><td>SSIM â</td><td>0.468</td><td>0.343</td><td>0.779</td><td>0.362</td><td>0.188</td><td>0.795</td><td>0.325</td><td>0.364</td></tr><tr><td></td><td>LPIPS â</td><td>0.56</td><td>0.659</td><td>0.483</td><td>0.638</td><td>0.665</td><td>0.397</td><td>0.587</td><td>0.555</td></tr><tr><td>Trajectory</td><td>ATE RMSE[cm]â</td><td>0.25</td><td>1.00</td><td>0.18</td><td>0.31</td><td>0.25</td><td>0.18</td><td>0.29</td><td>0.32</td></tr><tr><td rowspan="5">Ours</td><td>Geometry</td><td>L1 Depth[cm]â</td><td>0.96</td><td>3.58</td><td>0.62</td><td>1.44</td><td>1.85</td><td>0.99</td><td>3.20</td><td>3.43</td></tr><tr><td></td><td>PSNR [dB] â</td><td>28.01</td><td>21.01</td><td>32.13</td><td>30.59</td><td>24.13</td><td>31.7</td><td>27.12</td><td>27.10</td></tr><tr><td>Appearance</td><td>SSIMâ</td><td>0.787</td><td>0.601</td><td>0.894</td><td>0.801</td><td>0.742</td><td>0.901</td><td>0.795</td><td>0.794</td></tr><tr><td></td><td>LPIPS </td><td>0.096</td><td>0.150</td><td>0.138</td><td>0.210</td><td>0.260</td><td>0.12</td><td>0.908</td><td>0.097</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1. Non-rigid SLAM Evaluation on Sim4D Dataset.

<!-- image-->  
Figure 6. Qualitative Results on Real-World Datset. Our method effectively handles dynamic objects compared to MonoGS.

<!-- image-->  
Figure 7. 3D Reconstruction Result on Replica Office4. Left: MonoGS. Right: Ours (MonoGS-2D). Our surface Gaussianbased approach yields more accurate geometric reconstructions.

<table><tr><td></td><td>Metric</td><td>r0</td><td>r1</td><td>r2</td><td>o0</td><td>o1</td><td>o2</td><td>03</td><td>04</td><td>avg</td></tr><tr><td rowspan="5">MonoGS</td><td>ATE RMSE[cm]â</td><td>0.44</td><td>0.32</td><td>0.31</td><td>0.44</td><td>0.52</td><td>0.23</td><td>0.17</td><td>2.25</td><td>0.59</td></tr><tr><td>Depth L1[cm]â</td><td>3.00</td><td>3.47</td><td>4.66</td><td>3.10</td><td>6.08</td><td>6.15</td><td>4.77</td><td>4.94</td><td>4.52</td></tr><tr><td>Precision[%]â</td><td>39.0</td><td>28.8</td><td>28.9</td><td>39.7</td><td>15.8</td><td>28.0</td><td>32.5</td><td>25.5</td><td>29.7</td></tr><tr><td>Recall[%]â</td><td>44.2</td><td>34.5</td><td>32.8</td><td>47.6</td><td>24.3</td><td>30.0</td><td>35.4</td><td>28.5</td><td>34.6</td></tr><tr><td>F(%]</td><td>41.5</td><td>31.4</td><td>30.7</td><td>43.3</td><td>19.1</td><td>29.0</td><td>33.9</td><td>26.9</td><td>31.9</td></tr><tr><td rowspan="5">MonoGS-2D</td><td>ATE RMSE[cm]â</td><td>0.42</td><td>0.43</td><td>0.35</td><td>0.19</td><td>0.19</td><td>0.22</td><td>0.27</td><td>0.80</td><td>0.36</td></tr><tr><td>Depth L1[cm]â</td><td>0.45</td><td>0.28</td><td>0.57</td><td>0.37</td><td>0.59</td><td>0.85</td><td>0.62</td><td>0.63</td><td>0.54</td></tr><tr><td>Precision[%]â</td><td>97.0</td><td>97.0</td><td>97.0</td><td>97.1</td><td>97.9</td><td>95.8</td><td>94.8</td><td>83.9</td><td>95.0</td></tr><tr><td>eall[(%]â </td><td>85.5</td><td>86.0</td><td>84.8</td><td>89.4</td><td>85.1</td><td>81.8</td><td>81.5</td><td>72.4</td><td>83.3</td></tr><tr><td>F1[%]</td><td>90.9</td><td>91.3</td><td>90.5</td><td>93.1</td><td>91.1</td><td>88.2</td><td>87.6</td><td>77.7</td><td>88.8</td></tr></table>

Table 2. Static SLAM Ablation on Replica.

<table><tr><td>Method</td><td>Metric</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>avg.</td></tr><tr><td>MonoGS</td><td>ATE RMSE[cm]â Depth L1[cm]â</td><td>1.50 6.2</td><td>1.44 13.0</td><td>1.49 13.0</td><td>1.47 10.7</td></tr><tr><td>MonoGS-2D</td><td>ATE RMSE[cm]â Depth L1[cm]â</td><td>1.58 3.00</td><td>1.20 2.30</td><td>1.83 4.30</td><td>1.57 3.2</td></tr></table>

Table 3. Static SLAM Ablation on TUM

<table><tr><td>Method</td><td>Metric</td><td>KillingFusion</td><td>DeepDeform</td><td>iPhone</td></tr><tr><td rowspan="4">Morpheus [56]</td><td>Depth L1[cm]â</td><td>3.2</td><td>1.9</td><td>2.4</td></tr><tr><td>PSNR [dB] â</td><td>27.02</td><td>26.81</td><td>25.28</td></tr><tr><td>SSIMâ</td><td>0.77</td><td>0.81</td><td>0.46</td></tr><tr><td>LPIPS â</td><td>0.40</td><td>0.38</td><td>0.63</td></tr><tr><td rowspan="4">Ours</td><td>Depth L1[cm]â</td><td>4.9</td><td>1.1</td><td>0.57</td></tr><tr><td>PSNR [dB] â</td><td>31.13</td><td>24.15</td><td>27.54</td></tr><tr><td>SSIMâ</td><td>0.93</td><td>0.90</td><td>0.79</td></tr><tr><td>LPIPS â</td><td>0.13</td><td>0.27</td><td>0.26</td></tr></table>

Table 4. Offline Non-Rigid Reconstruction Ablation: Rendering Error Metrics on Real-world Dataset.

<!-- image-->  
Figure 8. Non-rigid Reconstruction Results. Our method flexibly models non-rigid deformations without requiring any shape templates or foreground/background separation.

Offline Non-rigid RGB-D Surface Reconstruction Table 4 reports offline reconstruction results, where camera poses are given. Our 2DGS+MLP deformation model shows competitive rendering performance compared to NeRF based methods. Note that Gaussian Splatting has the additional advantage of its rendering speed. We further provide qualitative visualizations in Fig. 8.

## 5. Conclusion

We presented the first tracking and mapping method for non-rigid surface reconstruction using Surface Gaussian Splatting. Our approach integrates a 2DGS + MLP warpfield SLAM framework with camera pose estimation and regularization, leveraging RGB-D input. To support further research, we also introduced a novel dataset for dynamic scene reconstruction with reliable ground truth. Experimental results demonstrate that our method outperforms traditional non-rigid SLAM approaches.

Limitations: Our method has primarily been tested on small-scale scenes; extending it to complex real-world scenarios may require 2D priors like point tracking or optical flow. The current implementation runs at 1.5 fps, limiting real-time use. Developing interactive dynamic scene scanning remains important future work.

## 6. Acknowledgement

Research presented in this paper has been supported by Dyson Technology Ltd. We are very grateful to members of the Dyson Robotics Lab for their advice and insightful discussions.

# 4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians

Supplementary Material

We encourage readers to watch the supplementary video for additional details and qualitative results.

## 7. Implementation Details

## 7.1. System Details and Hyper parameters

Non-Rigid SLAM: We set the learning weights as follows: $\lambda _ { p } = 0 . 9 , \lambda _ { g } = 0 . 1 , \lambda _ { i s o } = 1 0 . 0$ and $\lambda _ { n } = 0 . 0 0 2$ For the ARAP regularization [27], we use a nearest neighbor count of 20, a radius of 0.05, and an exponential decay weight of 500. Keyframes are selected with $N = 1$ . For the MLP, we use an 8-layer architecture with 256 neurons per layer. Frequency encoding is set to 1 for time and 4 for position. MLP is implemented with CUDA-optimized CutlassMLP in tiny-cuda-nn [32] for the fast optimization.

Static SLAM Ablation: We followed the same hyperparameters as MonoGS [29], but we use normal loss $L _ { n }$ with the weight $\lambda _ { n } = 0 . 0 1$ for the entire mapping process and $\lambda _ { g } = 0 . 5$ for the final refinement. For the Replica 3D reconstruction evaluation, we have used the script introduced in [43].

Offline Non-rigid RGB-D Reconstruction Ablation: Camera poses are provided by the dataset and remain fixed during training. For the MLP, we adopt the same architecture described in [64], consisting of an 8-layer network with 256 dimensions per layer, where a concatenated feature vector is input to the fourth layer. The positional encoding frequencies are set to 6 for time and 10 for position. Following the approach in [7, 56], we evaluate the geometric and appearance metrics against the input views and report the average values.

## 8. Camera Pose Jacobian

We provide the detail of the derivation of camera pose jacobian of 2D Gaussian Splatting in 3.2.

We use the notation from [49]. Let $T \in S E { \mathrm { ( 3 ) } }$ and $\tau = ( \rho , \pmb { \theta } ) \in \mathfrak { s e } ( 3 )$ , the left-side partial derivative on the manifold is defined as:

$$
\frac { \mathcal { D } f ( \pmb { T } ) } { \mathcal { D } \pmb { T } } \triangleq \operatorname* { l i m } _ { \tau  0 } \frac { \mathrm { L o g } ( f ( \mathrm { E x p } ( \tau ) \circ \pmb { T } ) \circ f ( \pmb { T } ) ^ { - 1 } ) } { \pmb { \tau } }\tag{21}
$$

Eq 11:

$$
\begin{array} { r l } & { \pmb { T } = \mathrm { E x p } ( \pmb { \tau } ) = \exp ( \pmb { \tau } ^ { \wedge } ) } \\ & { \quad = \exp \left( \displaystyle \sum _ { j = 1 } ^ { 6 } \mathbf { E } _ { j } \tau _ { j } \right) , \quad j = 1 , \ldots , 6 , \quad \pmb { \tau } \in \mathbb { R } ^ { 6 } . } \end{array}\tag{22}
$$

where the matrices $\mathbf { E } _ { j } ~ \in ~ \mathbb { R } ^ { 4 \times 4 }$ are the $S E ( 3 )$ group generators and form a basis for se(3):

$$
\mathbf { E } _ { 1 } = { \left[ \begin{array} { l l l l } { 0 } & { 0 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right] } \quad \mathbf { E } _ { 2 } = { \left[ \begin{array} { l l l l } { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right] }
$$

$$
\mathbf { E } _ { 3 } = { \left[ \begin{array} { l l l l } { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right] } \quad \mathbf { E } _ { 4 } = { \left[ \begin{array} { l l l l } { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { - 1 } & { 0 } \\ { 0 } & { 1 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right] }\tag{23}
$$

$$
\mathbf { E } _ { 5 } = \left[ { \begin{array} { c c c c } { 0 } & { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { - 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} } \right] \quad \mathbf { E } _ { 6 } = \left[ { \begin{array} { c c c c } { 0 } & { - 1 } & { 0 } & { 0 } \\ { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} } \right] .
$$

We get the partial derivative as follows:

$$
\frac { \partial } { \partial \tau _ { j } } \exp ( \pmb { \tau } ^ { \wedge } ) \bigg \vert _ { \pmb { \tau } = 0 } = \mathbf { E } _ { j } , \quad j = 1 , \dots , 6 .\tag{24}
$$

Therefore, the full derivative is given as:

$$
 \frac { \partial { \pmb T } } { \partial \tau } | _ { \tau = 0 } = \pmb T \frac { \partial ( \sum _ { j = 1 } ^ { 6 } { \bf E } _ { j } \tau _ { j } ) } { \partial \tau } | _ { \tau = 0 }\tag{25}
$$

Since the meaningful elements of the camera T is 12 number variables, we stack the elements for $1 2 \times 6$ matrix and we obtain

$$
\left. \frac { \partial { \pmb T } } { \partial { \pmb \tau } } \right| _ { { \pmb \tau } = 0 } = \left[ \begin{array} { c c } { \mathbf { 0 } } & { - \mathbf { R } _ { : , 1 } ^ { \times } } \\ { \mathbf { 0 } } & { - \mathbf { R } _ { : , 2 } ^ { \times } } \\ { \mathbf { 0 } } & { - \mathbf { R } _ { : , 3 } ^ { \times } } \\ { { \pmb I } } & { - \mathbf { t } ^ { \times } } \end{array} \right] .\tag{26}
$$

where $\mathbf { R } \in S O ( 3 )$ and $\mathbf { t } \in \mathbb { R } ^ { 3 }$ denote the rotation and translation parts of T .

Eq 13:

$$
 { \frac { \partial \mathbf { n } _ { c } } { \partial \tau } } | _ { \tau = 0 } = { \frac { \mathcal { D } \mathbf { n } _ { c } } { \mathcal { D } T _ { C W } } } = \operatorname* { l i m } _ { \tau  0 } { \frac { \mathrm { E x p } ( \tau ) \mathbf { n } _ { c } - \mathbf { n } _ { c } } { \tau } }\tag{27}
$$

$$
= \operatorname* { l i m } _ { \tau \to 0 } { \frac { ( I + \tau ^ { \wedge } ) \cdot \mathbf { n } _ { c } - \mathbf { n } _ { c } } { \tau } }\tag{28}
$$

$$
= \operatorname* { l i m } _ { \tau \to 0 } \frac { \pmb { \tau } ^ { \wedge } \cdot \mathbf { n } _ { c } } { \pmb { \tau } }\tag{29}
$$

$$
= \operatorname* { l i m } _ { \tau  0 } \frac { \pmb { \theta } ^ { \times } \mathbf { n } _ { c } + \pmb { \rho } } { \tau }\tag{30}
$$

$$
= \operatorname* { l i m } _ { \tau  0 } \frac { - \mathbf { n } _ { c } ^ { \times } \pmb { \theta } + \pmb { \rho } } { \tau }\tag{31}
$$

$$
\begin{array} { r l } { \mathbf { \Pi } } & { { } = \left[ { \pmb { I } } ^ { \phantom { \dagger } } \right. \left. - { \bf n } _ { c } ^ { \times } \right] } \end{array}\tag{32}
$$

## 9. Sim4D Training/Test Views

We define the training and test views on a sphere, with its center representing the target object. In spherical coordinates $( r , \theta , \phi )$ , we set $r = 2 . 0$ The training view is sampled from two arcs on the sphereâs surface, defined by $\theta \in$ $[ - 1 0 ^ { \circ } , 1 0 ^ { \circ } ]$ and $\phi \in [ - 1 0 ^ { \circ } , 1 0 ^ { \circ } ]$ . The test views are sampled from a circle on the sphereâs surface that pass through four key points: $( \theta , \phi ) = ( 5 ^ { \circ } , 0 ^ { \circ } ) , ( 0 ^ { \circ } , 5 ^ { \circ } ) , ( - 5 ^ { \circ } , 0 ^ { \circ } )$ , and $( 0 ^ { \circ } , - 5 ^ { \circ } )$ . These points are chosen to ensure uniform sampling around the target object while maintaining a clear separation between the training and test views.

<!-- image-->

<!-- image-->  
Figure 9. Training and Test Views on the Sim4D Dataset: Blue indicates training views, and Red indicates test views. Views are sampled (top right) from an arc on an object-centered sphere (top left) for dynamic scene reconstruction (bottom).

## 10. Further Ablation Analysis

## 10.1. Normal Rigidity Loss

Table 5 presents the quantitative results demonstrating the effect of the normal rigidity loss defined in Equation 19. The normal rigidity loss improves the overall geometric metrics, such as camera ATE and L1 Depth, for the benchmark sequences by preserving the local geometric consistency of 2D Gaussians.

<table><tr><td></td><td>ATE RMSE</td><td>L1 Depth</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>Ours full</td><td>0.28</td><td>1.71</td><td>28.47</td><td>0.820</td><td>0.12</td></tr><tr><td>w/o  $L _ { A R A P - n }$ </td><td>0.52</td><td>2.00</td><td>29.04</td><td>0.853</td><td>0.13</td></tr></table>

Table 5. Ablation Study on $L _ { A R A P - n } .$ We report the average number of Sim4D dataset.

## 10.2. Monocular Depth Prior

While our method was primarily tested with RGB-D camera input, we conducted an ablation study using depth input from the state-of-the-art monocular prediction network [58], as shown in Table 9. The results demonstrate performance competitive with SurfelWarp, highlighting the potential for purely monocular non-rigid SLAM.

## 10.3. Static SLAM Ablation Analysis

Replica: Table 8 shows the photometric rendering performance analysis on the Replica dataset. The results demonstrate that the 2DGS-based SLAM approach offers an advantage in achieving accurate appearance reconstruction.

TUM: Table 6 presents the full ablation analysis on the TUM dataset. The 2DGS-based approach maintains competitive ATE and appearance metrics while achieving significantly better geometric rendering accuracy, as reflected in the Depth L1 error.
<table><tr><td>Method</td><td>Metric</td><td>fr1</td><td>fr2</td><td>fr3</td></tr><tr><td rowspan="3">MonoGS</td><td>ATE RMSE [cm] â Depth L1 [cm] â</td><td>1.50 6.2</td><td>1.44 13.0</td><td>1.49 13.0</td></tr><tr><td>PSNR [dB] â SSIM â LPIPS â</td><td>23.5 0.775 0.26 1</td><td>24.65 0.785 0.201</td><td>25.09 0.842</td></tr><tr><td>ATE RMSE [cm] â Depth L1 [cm] â</td><td>1.58 3.0</td><td>1.2 2.3</td><td>0.200 1.83 4.3</td></tr><tr><td rowspan="2"></td><td>PSNR [dB] â SSIMâ</td><td>23.63 0.782</td><td>24.47 0.79</td><td>24.05 0.826</td></tr><tr><td>LPIPS â</td><td>0.251</td><td>0.228</td><td>0.223</td></tr></table>

Table 6. Static SLAM Ablation on TUM Dataset. Comparison of ATE RMSE, Depth L1, and Rendering Performance Metrics.

Memory Analysis Table 7 presents the average memory usage on the TUM dataset sequences. Due to the geometrically accurate alignment, 2D Gaussians require fewer primitives to represent the scene, resulting in reduced memory consumption.

<table><tr><td colspan="2">Memory Usage [MB]</td></tr><tr><td>MonoGS-2D</td><td>MonoGS</td></tr><tr><td>2.73MB</td><td>3.97MB</td></tr></table>

Table 7. Memory Analysis on TUM RGB-D dataset.

## 10.4. Offline Non-Rigid RGB-D Reconstruction Ablation

Table 10 provides the full evaluation details of the offline non-rigid RGB-D reconstruction ablation analysis.

<table><tr><td></td><td>Metric</td><td>room0</td><td>room1</td><td>room2</td><td>office0</td><td>office1</td><td>office2</td><td>office3</td><td>avg</td></tr><tr><td rowspan="3">MonoGS</td><td>PSNR [dB] â</td><td>34.83</td><td>36.43</td><td>37.49</td><td>39.95</td><td>42.09</td><td>36.24</td><td>36.70</td><td>37.50</td></tr><tr><td>SSIMâ</td><td>0.954</td><td>0.959</td><td>0.9665</td><td>0.971</td><td>0.977</td><td>0.964</td><td>0.963</td><td>0.96</td></tr><tr><td>LPIPS â</td><td>0.068</td><td>0.076</td><td>0.075</td><td>0.072</td><td>0.055</td><td>0.078</td><td>0.065</td><td>0.07</td></tr><tr><td rowspan="3">MonoGS-2D</td><td>PSNR [dB] â</td><td>36.21</td><td>37.81</td><td>38.7</td><td>43.45</td><td>43.8</td><td>37.48</td><td>37.43</td><td>39.14</td></tr><tr><td>SSIM â</td><td>0.966</td><td>0.969</td><td>0.9737</td><td>0.985</td><td>0.984</td><td>0.972</td><td>0.971</td><td>0.975</td></tr><tr><td>LPIPS â</td><td>0.04</td><td>0.042</td><td>0.044</td><td>0.025</td><td>0.029</td><td>0.04</td><td>0.039</td><td>0.038</td></tr></table>

Table 8. Static SLAM Ablation: Rendering Performance Metrics [43] on Replica Dataset

<table><tr><td>Method</td><td>Category</td><td>Metric</td><td>curtain</td><td>flag</td><td>mercedes</td><td>modular_vehicle</td><td>rhino</td><td>shoe_rack</td><td>water_effect</td><td>wave_toy</td></tr><tr><td rowspan="5">Ours (Monocular)</td><td>Trajectory</td><td>ATE RMSE[cm]â</td><td>6.23</td><td>16.29</td><td>4.90</td><td>1.86</td><td>3.17</td><td>8.02</td><td>5.52</td><td>7.21</td></tr><tr><td>Geometry</td><td>L1 Depth[cm]â</td><td>74.2</td><td>155</td><td>59.2</td><td>38.0</td><td>37.7</td><td>89.8</td><td>72.4</td><td>80.8</td></tr><tr><td>Appearance</td><td>PSNR [dB] â</td><td>17.73</td><td>16.22</td><td>20.72</td><td>26.28</td><td>21.48</td><td>17.49</td><td>18.86</td><td>17.98</td></tr><tr><td></td><td>SSIM LPIPS â</td><td>0.461</td><td>0.455</td><td>0.636</td><td>0.578</td><td>0253</td><td>0.448</td><td>0.390</td><td>0.441</td></tr><tr><td></td><td></td><td>0.297</td><td>0.517</td><td>0.282</td><td>0.380</td><td>0.339</td><td>0.391</td><td>0.258</td><td>0.281</td></tr></table>

Table 9. Non-rigid SLAM Evaluation on Sim4D Dataset with Monocular Depth Prior.

<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">KillingFusion</td><td colspan="3">DeepDeform</td><td colspan="3">iPhone</td></tr><tr><td>frog</td><td>duck</td><td>snoopy</td><td>seq002</td><td>seq004</td><td>seq028</td><td>teddy</td><td>mochi</td><td>haru</td></tr><tr><td rowspan="4">Morpheus [56]</td><td>Depth L1 [cm]</td><td>4.37</td><td>3.01</td><td>2.30</td><td>2.08</td><td>1.24</td><td>2.26</td><td>5.40</td><td>0.31</td><td>1.63</td></tr><tr><td>PSNR [dB] â</td><td>27.2</td><td>28.17</td><td>25.73</td><td>27.21</td><td>26.94</td><td>26.30</td><td>23.40</td><td>28.12</td><td>24.34</td></tr><tr><td>SSIM â</td><td>0.802</td><td>0.716</td><td>0.779</td><td>0.809</td><td>0.823</td><td>0.795</td><td>0.237</td><td>0.623</td><td>0.510</td></tr><tr><td>LPIPS â</td><td>0.31</td><td>0.419</td><td>0.483</td><td>0.301</td><td>0.428</td><td>0.397</td><td>0.776</td><td>0.55</td><td>0.564</td></tr><tr><td rowspan="4">Ours</td><td>Depth L1 [cm]</td><td>0.65</td><td>1.91</td><td>12.1</td><td>0.78</td><td>1.07</td><td>1.30</td><td>0.32</td><td>0.22</td><td>0.12</td></tr><tr><td>PSNR [dB] â</td><td>33.72</td><td>32.75</td><td>26.95</td><td>24.36</td><td>24.13</td><td>24.02</td><td>23.89</td><td>36.15</td><td>22.60</td></tr><tr><td>SSIMâ</td><td>0.941</td><td>0.949</td><td>0.899</td><td>0.897</td><td>0.897</td><td>0.902</td><td>0.739</td><td>0.926</td><td>0.690</td></tr><tr><td>LPIPS â</td><td>0.063</td><td>0.073</td><td>0.257</td><td>0.245</td><td>0.313</td><td>0.241</td><td>0.259</td><td>0.131</td><td>0.391</td></tr></table>

Table 10. Offline RGB-D Reconstruction Results

## References

[1] Poly haven. https://polyhaven.com/textures/ fabric. Accessed: 2024-11-01. 6

[2] Sketchfab. https://sketchfab.com/. Accessed: 2024-11-01. 6

[3] Oliver Boyne. Blendersynth. https://ollieboyne. github.io/BlenderSynth, 2023. 6

[4] Aljaz Bo Ë ziË c, Pablo Palafox, Michael Zollh Ë ofer, Justus Thies, Â¨ Angela Dai, and Matthias NieÃner. Neural deformation graphs for globally-consistent non-rigid reconstruction. arXiv preprint arXiv:2012.01451, 2020. 3

[5] Aljaz Bo Ë ziË c, Michael Zollh Ë ofer, Christian Theobalt, and Â¨ Matthias NieÃner. Deepdeform: Learning non-rigid rgb-d reconstruction with semi-supervised data. 2020. 3, 7

[6] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black. A naturalistic open source movie for optical flow evaluation. In Proceedings of the European Conference on Computer Vision (ECCV), 2012. 3

[7] Hongrui Cai, Wanquan Feng, Xuetao Feng, Yan Wang, and Juyong Zhang. Neural surface reconstruction of dynamic scenes with monocular rgb-d camera. In Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS), 2022. 3, 1

[8] Blender Online Community. Blender - a 3d modelling and rendering package, 2018. 3, 6

[9] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142â13153, 2023. 3, 6

[10] Bardienus P Duisterhof, Zhao Mandi, Yunchao Yao, Jia-Wei Liu, Jenny Seidenschwarz, Mike Zheng Shou, Ramanan Deva, Shuran Song, Stan Birchfield, Bowen Wen, and Jeffrey Ichnowski. DeformGS: Scene flow in highly deformable scenes for deformable object manipulation. WAFR, 2024. 3

[11] J. Engel, V. Koltun, and D. Cremers. Direct sparse odometry. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2017. 6

[12] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. In NeurIPS, 2022. 3, 7

[13] Wei Gao and Russ Tedrake. Surfelwarp: Efficient nonvolumetric single view dynamic reconstruction. In Proceedings of Robotics: Science and Systems (RSS), 2018. 3, 6, 7, 8

[14] R. Garg, A. Roussos, and L. Agapito. Dense variational reconstruction of non-rigid surfaces from monocular video. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013. 2

[15] Antoine Guedon and Vincent Lepetit. Sugar: Surface- Â´ aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. 2024. 2

[16] Marc Habermann, Weipeng Xu, Michael Zollhofer, Gerard Pons-Moll, and Christian Theobalt. Deepcap: Monocular human performance capture using weak supervision. In Pro-

ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5052â5063, 2020. 3

[17] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In Proceedings of SIGGRAPH, 2024. 2, 3

[18] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3

[19] Matthias Innmann, Michael Zollhofer, Matthias NieÃner, Â¨ Christian Theobalt, and Marc Stamminger. VolumeDeform: Real-time Volumetric Non-rigid Reconstruction. In Proceedings of the European Conference on Computer Vision (ECCV), 2016. 3

[20] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence, 36(7):1325â1339, 2013. 3

[21] M. M. Johari, C. Carta, and F. Fleuret. ESLAM: Efficient dense slam system based on hybrid representation of signed distance fields. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2

[22] Olaf Kahler, Victor Adrian Prisacariu, and David W. Murray. Â¨ Real-time large-scale dense 3d reconstruction with loop closure. In Proceedings of the European Conference on Computer Vision (ECCV), 2016. 2

[23] Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada. Neural 3D mesh renderer. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3907â3916, 2018. 2

[24] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track and map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2

[25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 2023. 2

[26] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1â248:16, 2015. 3

[27] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. 3DV, 2024. 3, 6, 1

[28] Ruibin Ma, Rui Wang, Yubo Zhang, Stephen Pizer, Sarah K McGill, Julian Rosenman, and Jan-Michael Frahm. Rnnslam: Reconstructing the 3d colon to visualize missing regions during a colonoscopy. Medical image analysis, 72: 102100, 2021. 3

[29] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. Gaussian Splatting SLAM. 2024. 2, 5, 6, 7, 1

[30] J. McCormac, A. Handa, A. J. Davison, and S. Leutenegger. SemanticFusion: Dense 3D semantic mapping with convolutional neural networks. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2017. 2

[31] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Proceedings of the European Conference on Computer Vision (ECCV), 2020. 2, 5

[32] Thomas Muller. tiny-cuda-nn, 2021. Â¨ 1

[33] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (TOG), 2022. 2, 5

[34] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, S. Hodges, and A. Fitzgibbon. KinectFusion: Real-Time Dense Surface Mapping and Tracking. In Proceedings of the International Symposium on Mixed and Augmented Reality (ISMAR), 2011. 2

[35] R. A. Newcombe, S. Lovegrove, and A. J. Davison. DTAM: Dense Tracking and Mapping in Real-Time. In Proceedings of the International Conference on Computer Vision (ICCV), 2011. 2

[36] Richard A Newcombe, Dieter Fox, and Steven M Seitz. Dynamicfusion: Reconstruction and tracking of non-rigid scenes in real-time. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 3

[37] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2

[38] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Steven M. Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. ICCV, 2021. 3

[39] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9054â9063, 2021. 3

[40] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural Radiance Fields for Dynamic Scenes. 3

[41] Juan J. Gomez Rodriguez, J. M. M Montiel, and Juan D. Tardos. Nr-slam: Non-rigid monocular slam. IEEE Transactions on Robotics (T-RO), 2023. 3

[42] Martin Runz and Lourdes Agapito. Co-fusion: Real- Â¨ time segmentation, tracking and fusion of multiple objects. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2017. 3

[43] Erik Sandstrom, Yue Li, Luc Van Gool, and Martin R. Os- Â¨ wald. Point-slam: Dense neural point cloud-based slam. In Proceedings of the International Conference on Computer Vision (ICCV), 2023. 2, 7, 1, 3

[44] Thomas Schops, Torsten Sattler, and Marc Pollefeys. Bad Â¨ slam: Bundle adjusted direct rgb-d slam. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2

[45] Raluca Scona, Mariano Jaimez, Yvan R Petillot, Maurice Fallon, and Daniel Cremers. StaticFusion: Background reconstruction for dense rgb-d slam in dynamic environments. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2018. 3

[46] Jenny Seidenschwarz, Qunjie Zhou, Bardienus Duisterhof, Deva Ramanan, and Laura Leal-Taixe. Dynomo: Online Â´ point tracking by dynamic online monocular gaussian reconstruction, 2024. 3

[47] Miroslava Slavcheva, Maximilian Baust, Daniel Cremers, and Slobodan Ilic. Killingfusion: Non-rigid 3d reconstruction without correspondences. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 3, 7

[48] Miroslava Slavcheva, Maximilian Baust, and Slobodan Ilic. Sobolevfusion: 3d reconstruction of scenes undergoing free non-rigid motion. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 3

[49] J. Sola, J. Deray, and D. Atchuthan. A micro Lie theory for \` state estimation in robotics. arXiv:1812.01537, 2018. 5, 1

[50] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan, Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva, Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael Goesele, Steven Lovegrove, and Richard Newcombe. The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019. 7

[51] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. A Benchmark for the Evaluation of RGB-D SLAM Systems. In Proceedings of the IEEE/RSJ Conference on Intelligent Robots and Systems (IROS), 2012. 7

[52] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison. iMAP: Implicit mapping and positioning in real-time. In Proceedings of the International Conference on Computer Vision (ICCV), 2021. 2

[53] L. Torresani, A. Hertzmann, and C. Chris Bregler. Nonrigid structure-from-motion: Estimating shape and motion with hierarchical priors. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 30(5), 2008. 2

[54] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhofer, Christoph Lassner, and Christian Theobalt. Non- Â¨ rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. 2021. 3

[55] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Coslam: Joint coordinate and sparse parametric encodings for neural real-time slam. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2

[56] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Morpheus: Neural dynamic 360deg surface reconstruction from

monocular rgb-d video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20965â20976, 2024. 3, 7, 8, 1

[57] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. 2024. 3

[58] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision, 2024. 2

[59] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909â4916. IEEE, 2020. 3

[60] T. Whelan, M. Kaess, H. Johannsson, M. F. Fallon, J. J. Leonard, and J. B. McDonald. Real-time large scale dense RGB-D SLAM with volumetric fusion. International Journal of Robotics Research (IJRR), 34(4-5):598â626, 2015. 2

[61] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison. ElasticFusion: Dense SLAM without a pose graph. In Proceedings of Robotics: Science and Systems (RSS), 2015. 2

[62] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting. In CVPR, 2024. 2

[63] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In Proceedings of the International Symposium on Mixed and Augmented Reality (ISMAR), 2022. 2

[64] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. 2024. 3, 5, 1

[65] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. Proceedings of the International Conference on Learning Representations (ICLR), 2024. 3, 5

[66] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics (TOG), 2024. 2

[67] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2