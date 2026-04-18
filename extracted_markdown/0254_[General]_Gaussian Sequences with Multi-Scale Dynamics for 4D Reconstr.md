# Gaussian Sequences with Multi-Scale Dynamics for 4D Reconstruction from Monocular Casual Videos

Can Li1,#, Jie Gu2, Jingmin Chen2, Fangzhou Qiu2, and Lei Sun1

Abstractâ Understanding dynamic scenes from casual videos is critical for scalable robot learning, yet four-dimensional (4D) reconstruction under strictly monocular settings remains highly ill-posed. To address this challenge, our key insight is that realworld dynamics exhibits a multi-scale regularity from object to particle level. To this end, we design the multi-scale dynamics mechanism that factorizes complex motion fields. Within this formulation, we propose Gaussian sequences with multi-scale dynamics, a novel representation for dynamic 3D Gaussians derived through compositions of multi-level motion. This layered structure substantially alleviates ambiguity of reconstruction and promotes physically plausible dynamics. We further incorporate multi-modal priors from vision foundation models to establish complementary supervision, constraining the solution space and improving the reconstruction fidelity. Our approach enables accurate and globally consistent 4D reconstruction from monocular casual videos. Experiments of dynamic novel-view synthesis (NVS) on benchmark and real-world manipulation datasets demonstrate considerable improvements over existing methods.

## I. INTRODUCTION

Embodied AI aims to endow robotic agents with the ability to perceive, reason, and act in the physical world. Achieving this goal critically depends on large-scale training data that capture both real environments and robot actions [1]â[4]. Unfortunately, real environments in such training data are typically captured as videos, providing no insight into the underlying three-dimensional space or the scene dynamics behind the observations. This gap motivates reconstructing four-dimensional (4D) representations directly from monocular videos. By bridging this gap, 4D reconstruction allows robots to effectively exploit visual data to infer spatiotemporal geometry and predictive dynamics or world models [5], ultimately contributing to scalable robot learning. Therefore, it is not only technically compelling but also strategically essential in the era of embodied intelligence.

Reliable 4D reconstruction of dynamic scenes remains highly challenging, especially under in-the-wild monocular conditions. Although static spatial reconstruction has been greatly advanced by 3D Gaussian Splatting (3DGS) [6], monocular 4D reconstruction is fundamentally ill-posed: the solution space is vast and supervision is extremely limited, often resulting in fragile and unstable reconstructions. The difficulty arises from the nature of monocular video data, e.g., sparse observations, limited parallax, depth ambiguity, and inherent scene dynamics. Moreover, dynamic scenes typically involve nonlinear deformations, frequent occlusions, and complex object interactions, further compounding the challenges.

<!-- image-->  
Monocular dynamic videos

<!-- image-->  
MS-Dynamics

<!-- image-->  
MS-Dynamics

(a) Monocular 4D Gaussian reconstruction of HOI.  
<!-- image-->

<!-- image-->

<!-- image-->  
(b) Application to hand-held data collection.  
Fig. 1: (a) From a casually captured monocular video with complex handâobject interactions (left), our MS-Dynamics models multi-scale dynamics to drive 4D Gaussians, producing a temporally-coherent Gaussian sequence that synthesizes novel views with fine hand details (middle). Without MS-Dynamics, the 4D reconstruction is blurry and lacks structural fidelity (right). (b) An application to hand-held data collection (left): starting from a video demonstration of cup deformation (middle), our method generates novel-view demonstrations (right).

A natural strategy for 4D reconstruction is to extend 3DGS by implicitly learning the temporal evolution of Gaussians. Warping Gaussians over time using implicit deformation fields [7], [8] shows promise, but these approaches often oversmooth dynamic details and are primarily evaluated in controlled settings. Furthermore, their âmonocularâ setups typically rely on camera orbits around dynamic objects that approximate multi-view capture, leaving their effectiveness in real-world monocular videos unclear.

Explicitly modeling the temporal evolution of Gaussians offers a promising alternative [9], [10]. MoSca [9] uses a large number of SE(3)-deformation nodes to model Gaussian motion, achieving high expressiveness but introducing excessive degrees of freedom that are prone to overfitting. Shape-of-motion [10] instead employs a low-dimensional motion field that exploits motion smoothness, improving stability but lacking the flexibility to model fine-grained deformations. These limitations highlight the need for a representation that is expressive enough to capture fine-grained dynamics yet sufficiently regularized to avoid overfitting under monocular supervision.

A central insight of this work is that the ill-posedness of monocular 4D reconstruction can be substantially alleviated by leveraging the inherent multi-scale regularities of realworld dynamics. Specifically, motion in natural scenes is not arbitrary. It is structured, unfolding from global object-level trajectories down to fine local deformations. These multiscale patterns serve as implicit physical constraints that all dynamic scenes must satisfy.

Building on this insight, we design a multi-scale dynamics (MS-Dynamics) mechanism that explicitly models this layered motion structure. By factorizing dynamics across object-level motion, primitive-level transformations, and fine-grained local deformations, MS-Dynamics reduces ambiguity and guides the optimization toward physically plausible solutions. This structured factorization injects strong inductive bias, enabling the representation to remain highly expressive while simultaneously constraining the solution spaceâstriking precisely the balance required for robust monocular 4D reconstruction.

We further observe that priors from vision foundation models, whose capabilities have grown significantly in recent years [11]â[14], provide valuable complementary supervision beyond raw RGB values, alleviating the inherent sparsity of supervision in monocular videos. Although the generated priors may not be perfectly accurate and mild noise is unavoidable in practice, their combination effectively constrains the solution space and improves the fidelity of dynamic reconstruction.

Consequently, we propose Gaussian Sequences with MS-Dynamics, where dynamic Gaussians are derived with multiscale motion and supervised by multi-modal prior signals. This formulation enables reliable reconstruction of Gaussian sequences from monocular videos and supports high-quality dynamic novel-view synthesis (NVS), as presented in Figure 1.

In summary, our key contributions are as follows:

â¢ We design the MS-Dynamics mechanism that represents complex dynamics through multi-scale motion fields. This factorization provides strong inductive bias and effectively reduces motion ambiguity.

â¢ We propose Gaussian sequences with MS-Dynamics for robust 4D reconstruction from monocular casual videos, guided by complementary multi-modal priors. This formulation strongly alleviates the high ill-posedness of 4D monocular reconstruction, improving reconstruction stability and consistency in strictly-monocular settings.

â¢ We provide custom datasets with rigid, articulated, and deformable objects, and extensively evaluate our approach on both benchmark and custom data, showing considerable gains in dynamic NVS.

## II. RELATED WORKS

## A. 3D Reconstruction

Recent years have witnessed rapid progress in NVS for static scenes. Neural Radiance Fields (NeRF) represent scenes using coordinate-based MLPs [15], achieving impressive photo-realistic results but suffering from slow rendering. 3DGS [6] offers a more efficient alternative by representing scenes as a set of anisotropic 3D Gaussians, achieving realtime rendering without sacrificing quality. Building on this, many recent works have enhanced 3DGS and extended its applications to robotics, spanning SLAM [16], manipulation [17], [18], and teleoperation [19].

However, existing methods focus mainly on static scenes. Our work addresses this gap by a dynamic Gaussian-based representation tailored for monocular 4D reconstruction.

## B. Dynamic Reconstruction

The goal of dynamic reconstruction is to recover the geometry, appearance, and motion that evolve over time. Although impressive results have been achieved using multiview synchronized cameras [20], [21], such settings are often impractical in real-world applications. As a result, monocular reconstruction has attracted increasing attention for its simplicity and applicability [9], [10], [22], [23].

NeRF-based methods learn deformation fields jointly with radiance fields [24], [25]. These deformation fields parameterized by MLPs benefit from continuity and compactness but struggle to capture fine-grained or highly non-rigid motion due to their over-smoothness.

Several Gaussian splatting-based methods adopt a similar latent deformation-based strategy [7], [8]. However, despite the rendering efficiency of 3DGS, the combination with implicit deformation fields still inherits limitations in motion fidelity and computational cost. To address these issues, some works replace implicit fields with explicit motion modeling [9], [10], [26]. For example, Shape-of-motion [10] utilizes shared low-dimension motion bases, and SplineGS [26] leverages motion-adaptive Hermite splines and joint optimization to enable NVS from monocular videos.

These advances highlight the importance of dynamic representations, yet notable challenges remain: low-rank models struggle to capture fine-grained motion, whereas high-dimensional formulations easily overfit in monocular settings. These limitations motivate us to pursue a balanced representation with multi-scale dynamics.

## III. METHODS

As shown in Figure 2, we propose Gaussian Sequences with MS-Dynamics for 4D reconstruction from monocular casual videos, yielding a dynamic Gaussian representation that enables photo-realistic NVS at arbitrary times. First of all, section III-A reviews the fundamentals of 3DGS. Section III-B provides the problem formulation, casting 4D reconstruction as the optimization of time-varying Gaussians. In section III-C, we thoroughly explain our framework of Gaussian sequences with MS-Dynamics. Finally, section III-D describes the well-designed loss functions for supervision.

<!-- image-->  
Fig. 2: Overview of Gaussian Sequences with MS-Dynamics for 4D monocular reconstruction. The pipeline first preprocesses monocular videos to obtain depths, masks, point tracks, and camera parameters. Our MS-Dynamics performs multi-scale factorization from object $( L _ { 1 } ) .$ , through sparse-primitive $( L _ { 2 } )$ , to fine-grained level $( L _ { 3 } )$ , capturing both global motion and local detailed deformation. Cross-frame Gaussian dynamics from canonical to target frame is modeled by shared weighted MS-Dynamics, constructing globally consistent Gaussian sequences. Both Gaussian sequences and MS-Dynamics are supervised by the aggregation of multi-modal signals (such as RGBs, depths, and tracks), which provides complementary cues for globally consistent optimization. The resulting Gaussian sequences enable high-quality dynamic NVS.

## A. Preliminaries

3DGS [6] models a static scene through a collection of anisotropic 3D Gaussian primitives, allowing efficient photorealistic rendering. Each Gaussian is defined as:

$$
{ \mathcal G } ( \mu , \Sigma , \alpha , { \mathbf c } )\tag{1}
$$

with a mean $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , a covariance matrix $ { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ , an opacity $\alpha \in \mathbb { R } ^ { + }$ , and a view-dependent color $\mathbf { c } \in \mathbb { R } ^ { 3 ( l + 1 ) ^ { 2 } }$ parameterized via spherical harmonics of degree l. Moreover, the covariance matrix Î£ can be decomposed into a rotation and a scale component as:

$$
\pmb { \Sigma } = \mathbf { R } _ { \Sigma } \mathbf { S } \mathbf { S } ^ { \top } \mathbf { R } _ { \Sigma } ^ { \top } ,\tag{2}
$$

where $\mathbf { R } _ { \Sigma } ~ \in ~ S O ( 3 )$ is a rotation matrix and $\textbf { S } \in \ \mathbb { R } _ { + } ^ { 3 }$ determines the scales.

To render the scene from a camera with parameters Î¸, each 3D Gaussian is projected to 2D as:

$$
\pmb { \mu } ^ { \prime } = \Pi ( \pmb { \mu } ; \pmb { \theta } ) \in \mathbb { R } ^ { 2 } , ~ \pmb { \Sigma } ^ { \prime } = \Pi ( \pmb { \Sigma } ; \pmb { \theta } ) \in \mathbb { R } ^ { 2 \times 2 } ,\tag{3}
$$

where $\Pi ( \cdot )$ denotes the projection function. The 2D Gaussians are then sorted by depth and composited using alpha blending:

$$
C ( \mathbf { p } ) = \sum _ { i = 1 } ^ { N } \mathbf { c } _ { i } \sigma _ { i } ( \mathbf { p } ) \left( \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma _ { j } ( \mathbf { p } ) ) \right) ,\tag{4}
$$

$$
\sigma _ { i } ( \mathbf { p } ) = \alpha _ { i } \exp \left( - \frac { 1 } { 2 } ( \mathbf { p } - \pmb { \mu } _ { i } ^ { \prime } ) ^ { \top } ( \Sigma _ { i } ^ { \prime } ) ^ { - 1 } ( \mathbf { p } - \pmb { \mu } _ { i } ^ { \prime } ) \right) ,\tag{5}
$$

where $\mathbf { p _ { \lambda } } \in \mathbb { R } ^ { 2 }$ is the 2D pixel coordinate, and N is the number of Gaussians intersecting the ray through p.

## B. Problem Formulation

To effectively handle dynamic conditions, we need to represent the evolving state and appearance of dynamic objects over time. Thus, we formulate 4D reconstruction as estimating a time-varying and temporally consistent Gaussian field $\mathcal { G } _ { t }$ :

$$
\mathcal { G } _ { t } : = \left\{ \mathcal { G } _ { t } ^ { i } \right\} _ { i = 1 } ^ { N } = \left\{ \mathcal { T } _ { t } ^ { i } \left( \mathcal { G } _ { 0 } ^ { i } \right) \right\} _ { i = 1 } ^ { N } = \left\{ \mathbf { T } _ { t } ^ { i } \odot \mathcal { G } _ { 0 } ^ { i } \right\} _ { i = 1 } ^ { N } ,\tag{6}
$$

where t denotes the time index and $\mathcal { G } _ { t } ^ { i }$ is a dynamic Gaussian derived by applying a transformation operator $\mathcal { T } _ { t } ^ { i }$ to the canonical Gaussian $\mathcal { G } _ { 0 } ^ { i }$ . If we parameterize $\mathcal { G } _ { 0 } ^ { i } \ \stackrel { \cdot } { = }$ $\left( \mu _ { 0 } ^ { i } , \Sigma _ { 0 } ^ { i } , \mathbf { c } ^ { i } , \alpha ^ { i } \right)$ and $\mathbf { T } _ { t } ^ { i } = \left[ \mathbf { \tilde { R } } _ { t } ^ { i } , \mathbf { t } _ { t } ^ { i } \right]$ , the operator $\mathcal { T } _ { t } ^ { i }$ acts on $\mathcal { G } _ { 0 } ^ { i }$ as:

$$
\pmb { \mu } _ { t } ^ { i } = \mathbf { R } _ { t } ^ { i } \pmb { \mu } _ { 0 } ^ { i } + \mathbf { t } _ { t } ^ { i } , \quad \pmb { \Sigma } _ { t } ^ { i } = \mathbf { R } _ { t } ^ { i } \pmb { \Sigma } _ { 0 } ^ { i } \left( \mathbf { R } _ { t } ^ { i } \right) ^ { \top } .\tag{7}
$$

Here, we assume that only the position and orientation of each Gaussian varies over time, while scale, color, and opacity remain static.

Therefore, the 4D Gaussian reconstruction problem is formulated as the following joint optimization:

$$
\operatorname* { m i n } _ { \{ \mathcal { G } _ { 0 } ^ { i } \} _ { N _ { G } } , \{ \mathbf { T } _ { t } ^ { i } \} _ { N _ { T } } } \sum _ { t = 1 } ^ { T } \mathcal { L } _ { \mathrm { r e n d e r } } \left( \mathcal { R } \Big ( \left\{ \mathbf { T } _ { t } ^ { i } \odot \mathcal { G } _ { 0 } ^ { i } \right\} _ { i = 1 } ^ { N } \Big ) , O _ { \mathrm { m o n o } } \right) ,\tag{8}
$$

where $\{ \mathcal { G } _ { 0 } ^ { i } \} _ { i = 1 } ^ { N _ { G } }$ and $\{ \mathbf { T } _ { t } ^ { i } \} _ { i = 1 } ^ { N _ { T } } , \forall t \in { 1 , \dots , T }$ are the variables to be optimized, R is a render operator, $O _ { \mathrm { m o n o } }$ denotes the monocular observation, and $\mathcal { L } _ { \mathrm { r e n d e r } }$ is the rendering loss.

Although Eq. (8) implies estimating an independent motion for each Gaussian, doing so would lead to an intractably high-dimensional optimization. Instead, we learn a shared low-rank, multi-scale motion field for dynamic Gaussians. Below, we describe how this is constructed.

## C. Gaussian Sequences with MS-Dynamics

Dynamic scenes captured from a monocular video often exhibit object interactions, non-rigid deformations, and spatially heterogeneous motion. Modeling such complex motion directly at per Gaussian is highly under-constrained: each Gaussian moves in 3D space, but its motion is only weakly observed from monocular frames. This leads to severe ambiguity, temporal drift, and overfitting when learning fully independent transformations for every Gaussian.

To address these challenges, our key insight is that realworld dynamic scenes contain strong multi-scale and lowrank regularities. Object-level motion tends to be globally coherent; within deformable objects, motion can be decomposed into a small number of principal deformation patterns; and at the finest scale, individual particles vary smoothly around these shared bases. This motivates a multi-scale dynamics modeling approach that shares motion structures across all Gaussians while allowing each Gaussian to make fine-scale adjustments through its own learnable weights.

1) Overview of MS-Dynamics: Figure 2 illustrates the proposed method of Gaussian sequences with MS-Dynamics. To model complex high-dimension dynamics, we propose shared weighted MS-Dynamics to explicitly factorize the per-Gaussian transformation $\mathbf { T } _ { t } ^ { i }$ in Eq. (6), capturing nonrigid motion with low-rank but multi-scale motion field. The general transformation $\mathbf { T } _ { t } ^ { i }$ is first formulated hierarchically in three layers:

$$
\mathbf { T } _ { t } ^ { i } = \prod _ { l = 1 } ^ { 3 } \mathbf { T } _ { t } ^ { i , ( L _ { l } ) } ,\tag{9}
$$

representing the composition of relative transformations in a coarse-to-fine manner, with layers corresponding to objectlevel $( L _ { 1 } )$ , sparse-primitive-level $\left( L _ { 2 } \right)$ , and fine-grained-level $( L _ { 3 } )$ motion.

In each layer $L _ { l } , \mathbf { T } _ { t } ^ { i , ( L _ { l } ) }$ itself is a weighted combination of the shared principal motion patterns. To simplify notation in $S E ( 3 )$ , we approximate the transformations of $\dot { \mathbf { T } } _ { t } ^ { i , ( L _ { l } ) }$ as:

$$
\mathbf { T } _ { t } ^ { i , ( L _ { l } ) } \approx \sum _ { k = 1 } ^ { K _ { l } } w _ { t } ^ { i , ( L _ { l } , k ) } \mathbf { P } _ { t } ^ { ( L _ { l } , k ) } , \quad l = 1 , 2 , 3 ,\tag{10}
$$

where $K _ { l }$ is the number of principal motion patterns in the layer $L _ { l } , \mathbf { P } _ { t } ^ { ( L _ { l } , k ) } \in S E ( 3 )$ is the k-th shared motion pattern, and $w _ { t } ^ { i , \top ( L _ { l } , \dot { k } ) }$ are the learnable weights for Gaussian i.

This design allows for sharing global dynamics across objects, low-rank deformation patterns inside each object, and flexible local corrections at grain level.

2) Object Level $( L _ { 1 } ) \colon$ General objects, even deformable ones, often exhibit motion that is coherent to some extent: parts of the object tend to move in a correlated manner, even when the overall shape deforms. We derive object-level motion patterns by clustering 3D point tracks into object groups and estimating their dominant rigid (or approximately coherent) transformations. Then, dynamic Gaussians share several object-level motion patterns. The reason for sharing is that different objects, especially during object interactions, often exhibit similar motion modes.

Given preprocessed 3D point tracks $\{ \pmb { x } _ { t } ^ { j } \} _ { j = 1 } ^ { N }$ and object segmentation masks, we first cluster the tracks into objectlevel groups. For each object-level cluster $\mathcal { C } ^ { ( L _ { 1 } , k ) } , k ~ =$ $1 , . . . , K _ { \mathrm { o b j e c t } }$ , we compute a dominant rigid transformation $\mathbf { P } _ { t } ^ { ( L _ { 1 } , k ) }$ of each cluster via weighted Procrustes alignment:

$$
\mathbf { P } _ { t } ^ { ( L _ { 1 } , k ) } = \arg \operatorname* { m i n } _ { \mathbf { R } , \mathbf { t } } \sum _ { \mathbf { \boldsymbol { x } } ^ { j } \in \mathcal { C } ^ { ( L _ { 1 } , k ) } } w _ { j } \left\| \mathbf { R } \mathbf { \boldsymbol { x } } _ { 0 } ^ { j } + \mathbf { t } - \mathbf { \boldsymbol { x } } _ { t } ^ { j } \right\| ^ { 2 } ,\tag{11}
$$

where $\pmb { x } _ { 0 } ^ { j }$ and $ { \boldsymbol { { x } } } _ { t } ^ { j }$ are the corresponding point sets of canonical and target frames, respectively, and $w _ { j }$ encodes the confidence of each point track.

With $K _ { \mathrm { o b j e c t } }$ shared motion patterns $\mathbf { P } _ { t } ^ { ( L _ { 1 } , k ) }$ , canonicalframe Gaussians can be transformed to target-frame ones using Eq. (10) at the object level, where each Gaussian is assigned soft learning weights $w _ { t } ^ { i , \left( L _ { 1 } , k \right) }$ based on its spatial proximity to the corresponding object cluster.

The object-level motion field is too coarse to be directly applied to represent Gaussian dynamics. However, capturing this object level provides a stable coarse estimate that anchors the subsequent layers of the multi-scale motion hierarchy.

3) Sparse-primitive Level $( L _ { 2 } ) \colon$ Non-rigid motion exists within deformable objects. Directly learning per-Gaussian motion is unstable; instead, deformation usually lies in a low-rank subspace. Inspired by the low-rank property, at this level, we capture object-internal deformation using a set of sparse motion primitives, each representing a principal mode in non-rigid deformation. With shared weighted sparse primitives, Gaussian deformation can be derived.

Within each object region, we first group the tracked 3D trajectories according to their movement directions. A weighted Procrustes alignment is then applied to each trajectory cluster, similar to Eq. (11), yielding $K _ { \mathrm { p r i m i t i v e } }$ motion bases per object. By assigning soft weights to these shared motion bases as in Eq. (10), the dynamic Gaussians can be driven by a blend of them, enabling expressive non-rigid motion while maintaining a compact motion representation.

Consequently, this level produces low-rank, sparse, and shared deformation modes. Note that $L _ { 2 }$ deformation is defined under the coordinate reference of object levels.

4) Fine-grained Level $( L _ { 3 } ) .$ : Although object-level and sparse-primitive-level motion capture the coarse and midlevel dynamics, they are insufficient to represent the subtle surface variations required for high-quality dynamic Gaussian reconstruction.

At the fine-grained level, we move toward denser and more local motion patterns, refining local deformations and correcting misalignments. Specifically, around each sparse primitive, we compute the residual motion between the primitive and the nearby point trajectories. These residuals are then clustered, and for each cluster we estimate a residual fine-grained motion modes using Eq. (11). $K _ { \mathrm { g r a i n } }$ groups are borned at the $L _ { 3 }$ level for every sparse primitive $( L _ { 2 } )$

Therefore, these fine-grained residual modes further enhance the expressiveness of the motion field, enabling the Gaussians to faithfully capture fine-scale and highly local deformations that cannot be explained by preceding motion levels.

5) MS-Dynamics Structure: After obtaining the motion fields from the three levels described above, we combine them according to Eq. (9) to construct the MS-Dynamics Hierarchy, as illustrated in Figure 2. Note that motion across levels is defined relatively, with each level expressed in the reference frame of the previous one.

Although theoretically one could further explore a dense particleâlevel dynamics representation, such an approach would significantly increase the dimensionality of optimization and lead to unstable solutions. In contrast, our MS-Dynamics hierarchy achieves a desirable trade-off between low-rank structure and multi-scale expressiveness.

Each Gaussian shares the MS-Dynamics hierarchy to follow globally coherent object-level motion while simultaneously capturing local fine-grained deformations. This exploits the underlying low-rank and multi-scale structure of dynamic scenes, enabling motion primitives to be efficiently learned and reused across all Gaussians, thereby improving both computational efficiency and temporal consistency.

## D. Loss Functions

To alleviate the inherently ill-posed nature of monocular video reconstruction, i.e., the optimization in Eq. (8), we incorporate a set of comprehensive multi-modal supervisions from monocular observations $O _ { \mathrm { m o n o } }$ . These multi-modal signals are acquired from off-the-shelf visual foundation models, contributing to constrain the solution space and improve the reconstruction fidelity.

Specifically, we employ a segmentation foundation model [11] to extract dynamic object masks, a monocular depth estimator [12], [13] to obtain depth maps, and a trackany-points model [14] to estimate long-term dense point trajectories.

The above results are then integrated to form a comprehensive multi-modal supervision signal. Accordingly, the loss function of the optimization in Eq. (8) comprises a RGB-image loss $\mathcal { L } ^ { \mathrm { r g b } }$ , a dynamic mask loss $\mathcal { L } ^ { \mathrm { m a s k } }$ , a depth alignment loss ${ \mathcal { L } } ^ { \mathrm { d e p t h } }$ , and a tracking loss ${ \mathcal { L } } ^ { \mathrm { t r a c k } }$ . In addition, we include a local rigidity loss $\mathcal { L } ^ { \mathrm { l o c a l - r i g i d } }$ to regularize the motion of dynamic Gaussians, enforcing the motion consistency among neighbors of Gaussians.

Consequently, we supervise the optimization of Eq. (8) with total loss $\mathcal { L } ^ { \mathrm { t o t a l } }$ as:

$$
\begin{array} { r l } & { \mathcal { L } ^ { \mathrm { t o t a l } } = \lambda _ { 1 } \mathcal { L } ^ { \mathrm { r g b } } + \lambda _ { 2 } \mathcal { L } ^ { \mathrm { m a s k } } + \lambda _ { 3 } \mathcal { L } ^ { \mathrm { d e p t h } } } \\ & { ~ + \lambda _ { 4 } \mathcal { L } ^ { \mathrm { t r a c k } } + \lambda _ { 5 } \mathcal { L } ^ { \mathrm { l o c a l - r i g i d } } . } \\ & { ~ \mathrm { I V . ~ E X P E R I M E N T S } } \end{array}\tag{12}
$$

## A. Datasets, Metrics, and Implementation

1) Datasets: We first evaluate methods on the well-known benchmark for monocular video reconstruction and NVS,

DyCheck (iPhone) [27] . The datasets contain diverse casual strictly-monocular dynamic videos captured by a handheld camera, without circling the dynamic objects to approximate multi-view coverage. In addition, seven scenes include additional cameras that provide videos suitable for NVS evaluation. From these scenes, we select five representative dynamic scenes (Apple, Block, Teddy, Spin, and Paperwindmill), where Apple, Block, and Teddy feature handobject interaction (HOI), Spin has a spinning human, and Paper-windmill is self-rotating.

We further evaluate our method on custom datasets, as shown in Figure 3. Videos of dynamic scenes are also recorded from a strictly monocular viewpoint, with an additional fixed camera for evaluation. The videos of two cameras are temporally synchronized by chessboard detection, achieving frame-level alignment with greater stability than audiobased synchronization in [27]. Relative poses of cameras are calibrated using chessboards. Our datasets cover a diverse set of object categories, containing rigid objects (Keyboard), articulated objects (Laptop), and deformable objects (Papercup and Mouse-pad). These objects are manipulated by hands or a gripper.

<!-- image-->  
Fig. 3: Experimental setup for our custom datasets.

2) Metrics: We adopt covisibility-masked image metrics [27], e.g., mPSNR, mSSIM, and mLPIPS, for NVS evaluation since the test view includes regions that are not observed by the monocular training view.

3) Implementation: We preprocess monocular videos with off-the-shelf vision foundation models to acquire masks [11], depths [12], [13], tracks [14], and camera poses [28]. The canonical frame is selected as the one with the least occlusion, determined by cross-frame point-tracking visibility. Adaptive dense control of both dynamic and static Gaussians follows the implementation of 3DGS [6].

For the MS-Dynamics configuration, we set $K _ { \mathrm { o b j e c t } } =$ number of instances at the object level $( L _ { 1 } ) , K _ { \mathrm { p r i m i t i v e } } = 5$ at the sparse-primitive level $( L _ { 2 } )$ , and $K _ { \mathrm { g r a i n } } = 1 0$ at the finegrained level $( L _ { 3 } )$ . This configuration is set according to the trade-off between motion expressiveness and optimization performance (see the ablation study Sec. IV-C).

We adopt Adam [29] for optimization and gsplat [30] for CUDA-accelerated differentiable rasterization. Our method is implemented with PyTorch and trained with 500 epochs on a NVIDIA A800 GPU. Training a sequence containing 150 frames takes approximately 60 minutes. The rendering speed at test time reaches 40fps, allowing for real-time rendering.

TABLE I: Evaluation of dynamic NVS on iPhone datasets [27]. Our method outperforms the sevral representative baselines, including NeRF-based dynamic reconstruction methods HyperNeRF [25] and T-NeRF [27], as well as dynamic Gaussianbased methods Deform-3DGS [7], Dynamic marbles [23], and Shape-of-motion [10]. (HOI: hand-object interaction)
<table><tr><td rowspan="2">Methods</td><td rowspan="2">mPSNR â</td><td colspan="2">Overall</td><td colspan="3">Apple (HOI)</td><td colspan="3">Block (HOI)</td></tr><tr><td>mSSIM â</td><td>mLPIPS â</td><td>mPSNR â</td><td>mSSIM â</td><td>mLPIPS â</td><td>mPSNR â</td><td>mSSIM â</td><td>mLPIPS â</td></tr><tr><td>HyperNeRF [25]</td><td>14.28</td><td>0.38</td><td>0.51</td><td>16.12</td><td>0.43</td><td>0.53</td><td>14.05</td><td>0.47</td><td>0.59</td></tr><tr><td>T-NeRF [27]</td><td>15.09</td><td>0.41</td><td>0.50</td><td>15.98</td><td>0.38</td><td>0.60</td><td>14.38</td><td>0.52</td><td>0.55</td></tr><tr><td>Deform-3DGS [7]</td><td>10.77</td><td>0.27</td><td>0.69</td><td>10.82</td><td>0.27</td><td>0.73</td><td>10.04</td><td>0.29</td><td>0.72</td></tr><tr><td>Dynamic marbles [23]</td><td>15.95</td><td></td><td>0.45</td><td>16.39</td><td>-</td><td>0.55</td><td>16.10</td><td>-</td><td>0.37</td></tr><tr><td>Shape-of-motion [10]</td><td>16.68</td><td>0.64</td><td>0.41</td><td>16.41</td><td>0.74</td><td>0.54</td><td>16.34</td><td>0.65</td><td>0.45</td></tr><tr><td>Ours</td><td>17.07</td><td>0.66</td><td>0.38</td><td>17.15</td><td>0.76</td><td>0.51</td><td>16.49</td><td>0.67</td><td>0.42</td></tr></table>

<table><tr><td rowspan="2">Methods</td><td colspan="3">Paper-Windmill (rotating)</td><td colspan="3">Spin (spinning human)</td><td colspan="3">Teddy (HOI)</td></tr><tr><td>mPSNR â</td><td>mSSIM â</td><td>mLPIPS</td><td>mPSNR â</td><td>mSSIM â</td><td>mLPIPS â</td><td>mPSNR â</td><td>mSSIM â</td><td>mLPIPS â</td></tr><tr><td>HyperNeRF [25]</td><td>14.59</td><td>0.26</td><td>0.37</td><td>14.61</td><td>0.42</td><td>0.44</td><td>12.04</td><td>0.31</td><td>0.62</td></tr><tr><td>T-NeRF [27]</td><td>15.63</td><td>0.35</td><td>0.27</td><td>15.98</td><td>0.45</td><td>0.50</td><td>13.48</td><td>0.34</td><td>0.57</td></tr><tr><td>Deform-3DGS [7]</td><td>10.32</td><td>0.24</td><td>0.61</td><td>12.26</td><td>0.31</td><td>0.71</td><td>10.42</td><td>0.25</td><td>0.68</td></tr><tr><td>Dynamic marbles [23]</td><td>16.17</td><td>-</td><td>0.45</td><td>17.45</td><td></td><td>0.43</td><td>13.65</td><td></td><td>0.46</td></tr><tr><td>Shape-of-motion [10]</td><td>19.50</td><td>0.56</td><td>0.22</td><td>17.42</td><td>0.70</td><td>0.30</td><td>13.73</td><td>0.55</td><td>0.53</td></tr><tr><td>Ours</td><td>19.83</td><td>0.57</td><td>0.20</td><td>17.85</td><td>0.72</td><td>0.29</td><td>14.05</td><td>0.57</td><td>0.50</td></tr></table>

<!-- image-->  
Train view  
Test view (GT)  
Dynamic marbles  
Deform-3DGS  
Shape-of-motion  
Ours  
Fig. 4: Qualitative results of NVS on iPhone datasets [27]. Ours synthesizes finer details than baselines. (GT: Ground truth)

## B. Results

We compare our method with several representative baselines. These include the NeRF-based dynamic reconstruction methods HyperNeRF [25] and T-NeRF [27], as well as dynamic Gaussian-based methods Deform-3DGS [7], Dynamic marbles [23], and Shape-of-motion [10]. Among them, T-NeRF and HyperNeRF extend neural radiance fields to dynamic settings. Deform-3DGS introduces MLP-based deformation fields into 3D Gaussian Splatting, while Dynamic marbles build isotropic Gaussian reconstructions from casual videos. Shape-of-motion [10] explicitly models the motion field for Gaussians, which is a state-of-the-art method for dynamic monocular Gaussian reconstruction. These methods serve as strong baselines to evaluate the ability of our framework to handle dynamic monocular reconstruction.

Table I presents quantitative results in iPhone datasets. Overall, our method achieves the best performance across all evaluation metrics with mPSNR of 17.07, mSSIM of 0.66, and mLPIPS of 0.38, consistently outperforming baselines. This is because T-NeRF and HyperNeRF rely heavily on an effective multi-view setting and struggle to handle strict monocular input. Deform-3DGS implicitly encodes Gaussian dynamics through a deformation field, which tends to oversmooth dynamic details and leads to poor reconstruction under monocular conditions. Dynamic marbles adopts isotropic Gaussian spheres for reconstruction, which reduces the degrees of freedom of Gaussian parameters in optimization but compromises the quality of NVS. Our method outperforms Shape-of-motion thanks to the proposed MS-Dynamics mechanism, enabling finer and more faithfully captured dynamic details.

TABLE II: Evaluation of dynamic NVS on custom datasets covering a wide range of object types with rigid, articulated, and deformable objects. Our method surpasses Shape-of-motion (state-of-the-art) [10] across all scenes.
<table><tr><td>Methods</td><td colspan="3">Keyboard (rigid) mPSNRâmSSIMâ mLPIPSâ|</td><td colspan="3">Laptop (articulated) |mPSNRâ mSSIMâ mLPIPSâ|</td><td colspan="3">Paper-cup (deformable) mPSNRâ mSSIMâ mLPIPSâ</td><td colspan="3">Mouse-pad (rigid-deformable) mPSNRâmSSIMâ mLPIPSâ</td></tr><tr><td>Shape-of-motion [10]]</td><td>11.86</td><td>0.23</td><td>0.11</td><td></td><td>13.87 0.59</td><td>0.12</td><td>12.76</td><td>0.25</td><td>0.03</td><td>8.41</td><td>0.47</td><td>0.13</td></tr><tr><td>Ours</td><td>14.23</td><td>0.57</td><td>0.06</td><td></td><td>15.77</td><td>0.75 0.07</td><td>19.11</td><td>0.73</td><td>0.02</td><td>10.81</td><td>0.63</td><td>0.11</td></tr></table>

<!-- image-->

<!-- image-->

<!-- image-->  
(a) Keyboard (rigid).

<!-- image-->

<!-- image-->  
(b) Laptop (articulated).

<!-- image-->

<!-- image-->  
(c) Paper cup (deformable).  
Fig. 5: Qualitative results of dynamic NVS of our method on custom datasets containing hand- or gripper-object interactions. Our MS-Dynamics effectively represents these interaction dynamics and, even when trained under strictly monocular views, produces detailed novel views under considerable viewpoint changes.

Figure 4 presents qualitative results of dynamic NVS in iPhone datasets. Our method produces noticeably higherquality renderings than the baselines, particularly in fine hand details. We primarily present rendering results from Gaussian-based methods whose rendering efficiency and fidelity are known to surpass NeRF-based methods [6].

Tab. II illustrates quantitative results in our custom datasets. We specifically compare our method with the stateof-the-art, Shape-of-motion. As shown, our approach outperforms existing methods across different object types with rigid, articulated, and deformable objects. This improvement is attributed to our effective decomposition of multi-scale interaction dynamics that guides the high-quality dynamic Gaussian reconstruction.

Figures 5 and 6 show the qualitative results of our methods. Under strictly monocular settings, MS-Dynamics effectively handles dynamic scenes with hand- or gripperobject interactions and successfully reconstructs high-fidelity novel views with significant viewpoint changes.

## C. Ablation

As shown in Table III, the ablation study evaluates how different representations of dynamics and loss configurations affect reconstruction quality on the iPhone datasets [27]. The results demonstrate that progressively enriching the motion representationâfrom object-level dynamics $( L _ { 1 }$ only, mPSNR = 11.34) to incorporating sparse motion primitives at the primitive level $( L _ { 1 }$ and $L _ { 2 }$ , mPSNR = 16.70)âyields substantial improvements across all metrics, highlighting the importance of fine-grained motion modeling.

Interestingly, further increasing the density of $L _ { 2 }$ primitives (e.g., raising $K _ { \mathrm { p r i m i t i v e } }$ from 5 to 10, denoted as $L _ { 2 } \mathrm { + } )$

<!-- image-->

<!-- image-->  
(Shape-of-motion)  
NVS (Ours)  
Fig. 6: Qualitative results of dynamic NVS on the challenged custom scene Mouse-pad (rigid-deformable). Our method (yellow) achieves clearer novel-view-synthesis than the baseline Shape-of-motion (red). Please watch the supplementary video for more details.

leads to a slight performance drop (mPSNR = 16.02), suggesting a trade-off between motion expressiveness and NVS performance when over-parameterizing intermediate motion layers. This indicates that deepening the motion hierarchy by introducing fine-grained residual modes at $L _ { 3 }$ is more effective than simply expanding the breadth of intermediate primitives, as these residual components further enhance the expressiveness of the motion field.

We also experimented with an additional dense particlelevel layer $( L _ { 1 } { - } L _ { 4 } )$ , but observed no significant improvement in reconstruction quality (mPSNR = 16.81 vs. 17.07) while incurring substantially longer optimization time. This suggests that the motion representation capacity of the $\boldsymbol { L } _ { 1 } { - } \boldsymbol { L } _ { 3 }$ hierarchy is already sufficient to capture the dominant deformation modes, and further refinement leads to diminishing returns due to motion-scale saturation.

<!-- image-->  
(a) Ablation of MS-Dynamics.

<!-- image-->

<!-- image-->  
NVS with full loss

(b) Ablation of loss function.  
<!-- image-->  
Fig. 7: Qualitative results of ablation studies in iPhone datasets [27]. Compared to our method, the variant with motion representation of $L _ { \mathrm { 1 } } { - } L _ { \mathrm { 2 } }$ (a) or one with only RGB-loss $\mathcal { L } ^ { \mathrm { r g b } }$ (b) yields blurrier NVS results.

Moreover, replacing the full loss with RGB-only supervision $( \mathcal { L } ^ { \mathrm { r g b } }$ only) achieves competitive results (mPSNR $= 1 6 . 6 0 )$ , yet still lags behind the complete formulation. Only when combining the full MS-Dynamics hierarchy $( L _ { 1 } -$ $L _ { 3 } )$ with the foundation-prior regularized loss (Eq. 12) do we obtain the best overall performance $( \mathrm { m P S N R } = 1 7 . 0 7 $ $\mathrm { m S S I M } = 0 . 6 6 $ , m $\mathrm { \underline { { P I P S } } } = \mathrm { 0 . 3 8 } )$ . This confirms that both components, including structured multi-scale dynamics and foundation-prior loss regularization, are essential for highfidelity dynamic novel view synthesis.

TABLE III: Ablation of MS-Dynamics and loss functions on Dycheck-iPhone [27]. Without MS-Dynamics or foundationprior loss, the performance of dynamic NVS is degraded. (Variants of MS-Dynamics, $L _ { 1 }$ only: object-level motion patterns; $L _ { 1 }$ and $L _ { 2 } { \mathrm { : } }$ object- and sparse-primitive-level motion representation; $L _ { 1 }$ and $L _ { 2 } + : L _ { 1 }$ with denser $L _ { 2 }$ motion primitives $( K _ { \mathrm { p r i m i t i v e } } ~ = ~ 1 0 ~ \mathrm { { \ v s } }$ . baseline $K _ { \mathrm { p r i m i t i v e } } ~ = ~ 5 )$ Variants of loss function, $\mathcal { L } ^ { \mathrm { r g b } }$ only: supervision with only RGB loss.)
<table><tr><td>Variant methods</td><td colspan="3">iPhone datasets [27] mPSNR â</td></tr><tr><td></td><td>11.34</td><td>mSSSIM</td><td>mLPIPS â 0.67</td></tr><tr><td> $L _ { 1 }$  only  $L _ { 1 }$  and  $L _ { 2 }$ </td><td>16.70</td><td>0.30 0.63</td><td>0.41</td></tr><tr><td> $L _ { 1 }$  and  $L _ { 2 } +$ </td><td>16.02</td><td>0.61</td><td>0.43</td></tr><tr><td> $\mathcal { L } ^ { \mathrm { r g b } }$ </td><td>16.60</td><td>0.63</td><td>0.39</td></tr><tr><td>only MS-Dynamics  $( L _ { 1 } { - } L _ { 3 } )$ </td><td>17.07</td><td>0.66</td><td>0.38</td></tr></table>

Figure 7 shows some qualitative results of ablation. It can be observed that our method with MS-Dynamics (Figure 7a) and full loss (Figure 7b) produces clearer NVS results, whereas the variant with motion representation of $L _ { 1 } { - } L _ { 2 }$ or one with only RGB-loss yields blurrier reconstructions. This further demonstrates the importance of the MS-Dynamics mechanism and the foundation-prior loss.

## V. CONCLUSION

In this work, we propose Gaussian sequences with MS-Dynamics as a unified framework for 4D reconstruction from monocular videos. By factorizing complex motion into structured levels and integrating complementary multi-modal cues of vision foundation, our method effectively reduces reconstruction ambiguity and stabilizes optimization. Experiments on benchmark and real-world custom data demonstrate clear gains in fidelity of dynamic NVS. Despite these achievements, our method may still struggle with severe occlusions or large-scale topology changes under strictly monocular conditions; extending the framework toward more robust representation of dynamics and applying it to generation of robot learning data are promising directions for future work.

## REFERENCES

[1] A. OâNeill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al., âOpen x-embodiment: Robotic learning datasets and rt-x models: Open xembodiment collaboration 0,â in 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 6892â6903, IEEE, 2024.

[2] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al., âDroid: A large-scale in-the-wild robot manipulation dataset,â arXiv preprint arXiv:2403.12945, 2024.

[3] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, âDiffusion policy: Visuomotor policy learning via action diffusion,â The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684â1704, 2025.

[4] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, âLearning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,â in Proceedings of Robotics: Science and Systems, (Daegu, Republic of Korea), July 2023.

[5] B. Ai, S. Tian, H. Shi, Y. Wang, T. Pfaff, C. Tan, H. I. Christensen, H. Su, J. Wu, and Y. Li, âA review of learning-based dynamics models for robotic manipulation,â Science Robotics, vol. 10, no. 106, p. eadt1497, 2025.

[6] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussianÂ¨ splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[7] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, âDeformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20331â20341, 2024.

[8] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20310â20320, 2024.

[9] J. Lei, Y. Weng, A. W. Harley, L. Guibas, and K. Daniilidis, âMosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds,â in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6165â6177, 2025.

[10] Q. Wang, V. Ye, H. Gao, W. Zeng, J. Austin, Z. Li, and A. Kanazawa, âShape of motion: 4d reconstruction from a single video,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9660â9672, 2025.

[11] J. Yang, M. Gao, Z. Li, S. Gao, F. Wang, and F. Zheng, âTrack anything: Segment anything meets videos,â arXiv preprint arXiv:2304.11968, 2023.

[12] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, âDepth anything: Unleashing the power of large-scale unlabeled data,â 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10371â10381, 2024.

[13] L. Piccinelli, Y.-H. Yang, C. Sakaridis, M. Segu, S. Li, L. Van Gool, and F. Yu, âUniDepth: Universal monocular metric depth estimation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[14] C. Doersch, Y. Yang, M. VecerÂ´Ä±k, D. Gokay, A. Gupta, Y. Aytar, J. Carreira, and A. Zisserman, âTapir: Tracking any point with per-frame initialization and temporal refinement,â 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10027â10038, 2023.

[15] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â 106, 2021.

[16] S. Hong, J. He, X. Zheng, and C. Zheng, âLiv-gaussmap: Lidarinertial-visual fusion for real-time 3d radiance field map rendering,â IEEE Robotics and Automation Letters, vol. 9, no. 11, pp. 9765â9772, 2024.

[17] Z. Lu, J. Ye, and J. Leonard, â3dgs-cd: 3d gaussian splatting-based change detection for physical object rearrangement,â IEEE Robotics and Automation Letters, vol. 10, pp. 2662â2669, 2024.

[18] S. Yang, W. Yu, J. Zeng, J. Lv, K. Ren, C. Lu, D. Lin, and J. Pang, âNovel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation,â in Proceedings of Robotics: Science and Systems, (LosAngeles, CA, USA), June 2025.

[19] Y. Lee, H. Kim, H. Ji, J. Heo, Y. Lee, J. Kang, J. Lee, and D. Lee, âHuman-in-the-loop gaussian splatting for robotic teleoperation,â IEEE Robotics and Automation Letters, vol. 11, no. 1, pp. 105â 112, 2026.

[20] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, âK-planes: Explicit radiance fields in space, time, and appearance,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12479â12488, 2023.

[21] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, âDynamic 3d gaussians: Tracking by persistent dynamic view synthesis,â in 2024 International Conference on 3D Vision (3DV), pp. 800â809, IEEE, 2024.

[22] L. Qingming, Y. Liu, J. Wang, X. Lyu, P. Wang, W. Wang, and J. Hou, âModgs: Dynamic gaussian splatting from casually-captured monocular videos with depth priors,â in The Thirteenth International Conference on Learning Representations, 2025.

[23] C. Stearns, A. Harley, M. Uy, F. Dubost, F. Tombari, G. Wetzstein, and L. Guibas, âDynamic gaussian marbles for novel view synthesis of casual monocular videos,â in SIGGRAPH Asia 2024 Conference Papers, pp. 1â11, 2024.

[24] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz, and R. Martin-Brualla, âNerfies: Deformable neural radiance fields,â in Proceedings of the IEEE/CVF international conference on computer vision, pp. 5865â5874, 2021.

[25] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, and S. M. Seitz, âHypernerf: A higherdimensional representation for topologically varying neural radiance fields,â arXiv preprint arXiv:2106.13228, 2021.

[26] J. Park, M.-Q. V. Bui, J. L. G. Bello, J. Moon, J. Oh, and M. Kim, âSplinegs: Robust motion-adaptive spline for real-time dynamic 3d gaussians from monocular video,â in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26866â26875, 2025.

[27] H. Gao, R. Li, S. Tulsiani, B. Russell, and A. Kanazawa, âMonocular dynamic view synthesis: A reality check,â Advances in Neural Information Processing Systems, vol. 35, pp. 33768â33780, 2022.

[28] Z. Li, R. Tucker, F. Cole, Q. Wang, L. Jin, V. Ye, A. Kanazawa, A. Holynski, and N. Snavely, âMegasam: Accurate, fast and robust structure and motion from casual dynamic videos,â in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10486â 10496, 2025.

[29] D. Kinga, J. B. Adam, et al., âA method for stochastic optimization,â in International conference on learning representations (ICLR), vol. 5, California;, 2015.

[30] V. Ye, R. Li, J. Kerr, M. Turkulainen, B. Yi, Z. Pan, O. Seiskari, J. Ye, J. Hu, M. Tancik, et al., âgsplat: An open-source library for gaussian splatting,â Journal of Machine Learning Research, vol. 26, no. 34, pp. 1â17, 2025.