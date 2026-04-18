# Scale-GS: Efficient Scalable Gaussian Splatting via Redundancy-filtering Training on Streaming Content

Jiayu Yang, Weijian Su, Songqian Zhang, Yuqi Han, Jinli Suo, Qiang Zhang, Senior Member, IEEE

Abstractâ3D Gaussian Splatting (3DGS) enables high-fidelity real-time rendering, a key requirement for immersive applications. However, the extension of 3DGS to dynamic scenes remains limitations on the substantial data volume of dense Gaussians and the prolonged training time required for each frame. This paper presents Scale-GS, a scalable Gaussian Splatting framework designed for efficient training in streaming tasks. Specifically, Gaussian spheres are hierarchically organized by scale within an anchor-based structure. Coarser-level Gaussians represent the low-resolution structure of the scene, while finer-level Gaussians, responsible for detailed high-fidelity rendering, are selectively activated by the coarser-level Gaussians. To further reduce computational overhead, we introduce a hybrid deformation and spawning strategy that models motion of inter-frame through Gaussian deformation and triggers Gaussian spawning to characterize wide-range motion. Additionally, a bidirectional adaptive masking mechanism enhances training efficiency by removing static regions and prioritizing informative viewpoints. Extensive experiments demonstrate that Scale-GS achieves superior visual quality while significantly reducing training time compared to state-of-the-art methods.

Keywords: Streaming Gaussian Splatting, multi-scale representation, dynamic scene rendering, novel view synthesis.

## I. INTRODUCTION

The rapid advancement of 3D Gaussian Splatting (3DGS) [1] has significantly reshaped the domain of realtime 3D rendering. In particular, the introduction of Gaussian training methods designed for dynamic scenes [2]â[9] has greatly enhanced the feasibility of 3D streaming applications, including virtual reality (VR), augmented reality (AR), and immersive telepresence systems. By explicitly representing scenes with differentiable Gaussians, these methods facilitate real-time renderingâcapabilities that are critical for interactive applications demanding low-latency visual feedback. However, as computational demands scale sharply with scene and temporal complexity, the training time for dynamic scenesâranging from tens of minutes to hoursâconflicts with the low-latency requirements of real-time streaming.

The primary reason for the slow training time in Gaussian Splatting stems from redundant computations involving Gaussian spheres. The standard 3D Gaussian splatting methods process each frame independently, thereby incurring repetitive calculations on predominantly static Gaussians due to the limited extent of dynamic regions. Although recently some research partitions the scene into static and dynamic components to focus computational resources on dynamic Gaussians, the overall volume remains substantial. Consequently, significant overlapping computations occur across spatially and temporally adjacent regions, leading to inefficient resource usage and bottlenecks that hinder real-time performance.

We observe that the size and number of Gaussian spheres to represent the 3D scene vary significantly depending on scene complexity. For example, textureless planar regions can be effectively modeled by a small number of large Gaussian spheres, whereas highly textured areas necessitate dozens or even hundreds of smaller Gaussians. The larger Gaussian spheres often contribute more significantly to scene representation and therefore warrant higher training priority. Inspired from scalable video coding, this work proposes Scale-GS, a scalable Gaussian splatting framework aimed at mitigating redundancy in 3D spatial representation, thereby enabling accelerated training. Specifically, Gaussian spheres are organized by scale, with each level independently trained under the viewpoints at corresponding resolutions. For each frame, large scale Gaussians are first optimized using low-resolution views to approximate the scene. Upon convergence, a triggering criterion evaluates whether small scale Gaussians should be activated for refinement, thereby reducing the redundant Gaussian spheres optimization. As shown in Fig. 1, the proposed method achieves high rendering quality with short training time compared to the SOTA algorithm. The framework conducts a coarse-to-fine principle where the training at each level of scale is triggered by the preceding level of scale, thereby filtering redundant training of irrelevant Gaussians.

To enable efficient training for streaming content, we introduce a hybrid strategy that combines deformation and spawning to infer dynamic changes in the current frame based on the preceding frame. Generally, the Gaussian deformation [6], [9] models the motion of Gaussian sphere, but insufficient for capturing newly appearing objects or wide-range motion. In contrast, Gaussian spawning [10], [11] introduces new Gaussians to fine-tune dynamic regions but requires considerably longer training time. To balance the trade-off, the hybrid strategy first applies deformation to model inter-frame motion and then determines, based on the training outcome, whether spawning should be triggered for finer-grained refinement. Specifically, under the Scale-GS framework, if the deformation of a determined scale fails to adequately represent the dynamic, either the emergence of new content or the motion of smaller-scale Gaussians happens. Thus, the deformation result triggers new Gaussians spawning at the current scale and deformation at the next scale. This sequential activation ensures Gaussians are progressively introduced at locations with actual dynamics along with increasingly finer scales, enabling efficient and high-fidelity temporal GS representation.

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 1. The proposed Scale-GS under dynamic scene achieves best rendering quality with the shortest training time. The left figures show results of our Scale-GS on N3DV Coffee martini and MeetRoom Trimming datasets, where âResâ indicates video resolution. The right figure is tested on the N3DV dataset, where the radius of the circle corresponds to the average storage per frame and the method in the top left corner demonstrates the best performance.

To enhance training efficiency, we propose a bidirectional adaptive masking mechanism that simultaneously suppresses static regions and selects informative training viewpoints. The forward masking component detects dynamic and static anchors via inter-frame change analysis, where pixel-wise differences between consecutive frames are back-projected to estimate motion patterns. For backward camera viewpoint selection, we define a relevance score between projected dynamic anchors and camera fields of view, further weighted by directional factors that prioritize orthogonal or novel viewpoints. The top-ranked views, as determined by the relevance score, are selected to form the active viewpoint set. This bidirectional masking mechanism reduces computational redundancy caused by uninformative viewpoints and facilitates accurate reconstruction of dynamic scenes.

Comprehensive experiments conducted on three challenging real-world datasetsâNV3D, MeetRoom, and Google Immersiveâdemonstrate the superior performance of the proposed framework across multiple evaluation metrics. Qualitative comparisons show that our method reconstructs significantly sharper fine-grained details, particularly in complex scenarios involving human interactions, dynamic phenomena such as flames, and intricate textures. Furthermore, experimental results demonstrate that Scale-GS not only improves visual quality but also outperforms current state-of-the-art methods in both training and rendering time. These findings validate the effectiveness of Scale-GS, which prioritizes more important Gaussian spheres and improves the average training efficiency.

The main contributions of this work are as follows:

1) We propose Scale-GS, a scalable GS framework performing redundancy-filtering training on streaming content to improve the efficiency. The Scale-GS achieves the most efficient training compared to the existing Gaussian training methods on streaming content.

2) The Scale-GS integrates a hybrid deformation-spawning Gaussian training strategy that prioritizes large-scale Gaussians and selectively activating finer ones to reduce redundancy in 3D scene representation while preserving high-fidelity dynamic representations.

3) Extensive evaluations show that Scale-GS achieves superior efficiencyâquality trade-offs for streaming novel view synthesis, improving visual quality, reducing training time, and supporting real-time rendering.

In the following, we first introduce the research related to the Scale-GS, including the novel view synthesis for static scene and videography at Sec. II. Later we thoroughly present the detail of the method at Sec. III. The qualitative and quantitative experimental results are exhibits at Sec. IV. Finally, we draw the conclusion and propose the future work at Sec. V.

## II. RELATED WORK

In this section, we separately review research on implicit and explicit novel view synthesis methods for both static and dynamic scenes. These studies primarily focus on improving visual quality and enhancing training efficiency.

## A. Novel View Synthesis for Static Scenes

Early novel view synthesis methods predominantly rely on geometric interpolation, with approaches such as the Lumigraph [12] and Light Field rendering [13], [14], laying the groundwork through advanced interpolation techniques applied to densely sampled input images.

Neural Radiance Fields (NeRF) [15] introduces a breakthrough in photorealistic view synthesis by modeling scene radiance through implicit neural representations using multilayer perceptrons. This innovation has spurred extensive research aiming at overcoming NeRFâs inherent limitations across various dimensions. Key efforts include accelerating training procedures [16]â[19], achieving real-time rendering performance [20]â[22], improving synthesis fidelity in complex scenes [21], [23], [24], and enhancing robustness under sparse input conditions [25]â[27]. However, the computational overhead inherent in NeRFâs volume rendering paradigmâwhich requires numerous neural network computation per frameâpresents significant challenges in balancing training efficiency, rendering speed, and visual fidelity.

To address these limitations, Kerbl et al. [1] proposes 3DGS, which leverages explicit 3D Gaussian primitives combined with differentiable rasterization-based rendering to enable real-time, high-quality view synthesis. The 3DGS inspires a broad range of research efforts exploring various aspects of Gaussian-based scene representations. Some studies focus on enhancing rendering fidelity [28]â[30], while others aim to improve geometric precision and accuracy [31], [32].

Furthermore, considerable efforts are devoted to developing compression techniques to mitigate storage overhead [28], [33]â[36]. Recent studies investigate the joint optimization of camera parameters alongside Gaussian field estimation [37], as well as the extension of Gaussian splatting to broader 3D content generation tasks [38]â[40]. Despite the 3DGS in static scenes achieves high-quality rendering, the development of an on-demand training framework for 3DGS remains an open challenge.

Generalization is introduced as an effective strategy to enhance inference speed. Recent developments in NeRF methodologies [27], [41], [42] and 3DGS [43]â[45] focus on generalizable reconstruction networks trained on large-scale datasets. Specifically, PixelSplat [46] uses Transformers to encode features and decode Gaussian primitive parameters. DepthSplat [47] leverages monocular depth to recover 3D details from sparse inputs. Other frameworks [48]â[50] combine Transformer or Multi-View Stereo (MVS) [51] methods to build geometric cost volumes, enabling real-time generalization. However, due to the insufficient diversity of available 3D datasets, the generalization performance of these methods remains to be further improved.

## B. Novel View Synthesis for Dynamic Scenes

Novel view synthesis for dynamic scenes naturally extends static models, with early approaches building on NeRF [11], [52]â[57] and 3DGS [1], leveraging their efficient rendering capabilities for dynamic scene reconstruction. While Gaussianbased methods [2]â[9] learn temporal attributes to model dynamic scenes as unified representations and improve reconstruction quality, their requirement to load all data simultaneously results in high memory usage, limiting their feasibility for long-sequence streaming.

To address these challenges, streaming-based methods, such as ReRF [58], NeRFPlayer [59], and StreamRF [60] reformulate dynamic scene reconstruction as an online problem. Moreover, 3DGStream [10] utilizes Gaussian-based representations combined with Neural Transformation Caches to model inter-frame motion, though it still requires over 10 seconds per frame. HiCoM [61] introduces an online reconstruction pipeline for multi-view video streams employing perturbationbased smoothing for robust initialization and hierarchical motion coherence mechanisms. Instant Gaussian Stream (IGS) [62] proposes enables single-pass motion computation guided by keyframes, reducing error accumulation and achieving reconstruction times around 4 seconds per frame. Alternative frame-tracking methods [63], [64] track Gaussian evolution across frames, supporting streaming protocols but incurring substantial per-frame data overhead.

In contrast to existing methods that require full sequence processing or incur significant per-frame optimization overhead, we propose Scale-GS, a scalable Gaussian splatting framework for efficient streaming rendering. By constructing multi-scale Gaussian representations combined with selective training, Scale-GS enables on-the-fly novel view synthesis while preserving high rendering quality.

## III. METHOD

In this section, we first introduce the preliminaries in Sec. III-A. Later we present the key pipeline of Scale-GS. The framework of Scale-GS is presented as in Fig. 2. After decompositing the dynamic part, Scale-GS follows an anchor-based multi-scale Gaussian representation (Sec. III-B) as its core framework. We apply hybrid deformation-spawning Gaussian optimization (Sec. III-C) to model inter-frame motion. When deformation is insufficient to capture dynamics, we activate the next scale and selectively spawn new Gaussians. Once all scales have converged, redundant Gaussians are pruned to optimize the representation (Sec. III-D). In addition, Scale-GS employs bidirectional bidirectional adaptive masking (Sec. III-E) to identify dynamic anchors and select informative viewpoints.

## A. Preliminaries

3DGS uses a dense set of Gaussian spheres to represent the whole space, and renders viewpoints via differentiable splatting combined with tile-based rasterization of these Gaussian components. For each Gaussian sphere i, the expectation position $\mu _ { i }$ and variance $\Sigma _ { i }$ determine the formation of the Gaussian $G _ { i } ( x )$ . The point x on the Gaussian sphere $G _ { i } ( x )$ is noted as

$$
G _ { i } ( x ) = \exp \left( - \frac { 1 } { 2 } ( x - \mu _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( x - \mu _ { i } ) \right) ,\tag{1}
$$

where $\Sigma _ { i }$ is composed of a rotation matrix $R _ { i } ~ \in ~ \mathbb { R } ^ { 3 \times 3 }$ and a diagonal scale matrix $S _ { i } ~ \in ~ \mathbb { R } ^ { 3 \times 3 }$ , denoted as $\Sigma _ { i } =$ $R _ { i } S _ { i } S _ { i } ^ { T } R _ { i } ^ { \overline { { T } } }$ . The color and opacity of Gaussian sphere i are denoted as $c _ { i }$ and $\alpha _ { i }$

We use $x ^ { \prime }$ to define the 2D projection pixel position, and the pixel value $C \in \mathbb { R } ^ { 3 }$ is rendered via Î±-composite blending. Specifically, we assume the light ray projected onto a pixel $x ^ { \prime }$ intersects with N Gaussian surfaces along its path. The color $C ( \boldsymbol { x } ^ { \prime } )$ is defined as

$$
C ( x ^ { \prime } ) = \sum _ { i \in N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where N Gaussians are sorted from near to far and $\alpha _ { i }$ signifies the opacity of Gaussian $G _ { i } ( x )$ . With the differentiable rasterizer, all attributes of the 3D Gaussians become learnable and can be directly optimized in an end-to-end manner through training view reconstruction.

To enable structural rendering in the Gaussian splatting model, we introduce an anchor-based mechanism [28], [33]. The 3D space is uniformly partitioned into multiple voxels, with each voxel assigned a dedicated anchor responsible for managing all Gaussian primitives within its region.

The anchor-based framework is initialized by voxelizing the sparse point cloud generated from Structure-from-Motion (SfM) pipelines. Let v denote the index of a specific anchor, and V represent the complete set of anchors across the entire space. Each $v \in V$ corresponds to a local context feature ${ \hat { f } } _ { v } ,$ a 3D scaling factor $c _ { v } ,$ and k learnable offsets $O _ { v } \in \mathbb { R } ^ { \dot { k } \times 3 }$ to identify the attribute of the anchor v and k corresponding Gaussians (indexed from 0 to k â 1). Specifically, given the location $x _ { v }$ of anchor v, the positions of Gaussians winthin the anchor v are derived as

<!-- image-->  
Fig. 2. The framework of Scale-GS. (a) The multi-scale decomposition across different resolution levels, where finer scales capture increasingly detailed scene dynamics. (b) Anchor-based multi-scale Gaussian representation. After completing training at each level, all scales are combined, followed by redundant Gaussians removing. (c-d) The hybrid deformation and spawning Gaussian optimization. (c) Gaussian deformation module that models temporal changes through anchor-guided MLPs, (b) Octree Gaussian Spawning that adaptively adds new Gaussians based on Octree subdivision.

$$
\left\{ \mu _ { 0 } , \dots , \mu _ { k - 1 } \right\} = x _ { v } + \left\{ O _ { 0 } , \dots , O _ { k - 1 } \right\} \cdot c _ { v } .\tag{3}
$$

Since the Gaussians associated with the same anchor share similar attributes, the other factors of Gaussians could be predicted by lightweight MLPs $F _ { \alpha } , F _ { c } , F _ { q } , F _ { s }$ taking the anchor attributes as input. Specifically, we denote the relative viewing distance from anchor v to the camera as $\delta _ { v _ { . } }$ and the viewing direction from anchor v to the camera as $\vec { d _ { v } }$ . Taking opacity prediction as an example, the opacity values of all k Gaussians within anchor v are derived as

$$
\{ \alpha _ { 0 } , \dots , \alpha _ { k - 1 } \} = F _ { \alpha } ( \hat { f } _ { v } , \delta _ { v } , \vec { d } _ { v } ) .\tag{4}
$$

The color, rotation, and scale predictions follow similar formulations using their respective MLPs $F _ { c } , F _ { q } ,$ , and $F _ { s }$

By partitioning the whole space into voxels and assigning an anchor to each voxel, the anchor-based approach facilitates localized organization and efficient indexing of Gaussian distributions, significantly reducing the overhead of traversing and computing irrelevant Gaussians. Moreover, the neural Gaussians, which infer all Gaussian factors from the anchor attribute improves the efficiency of training.

## B. Anchor-based Multi-Scale Gaussian Representation

Variations in texture detail of the 3D scene lead to Gaussian representations of differing granularity. To improve computational efficiency, we introduce a multi-scale Gaussian optimization framework, drawing inspiration from scalable video encoding [65], [66]. In this framework, coarse-scale Gaussians are first optimized using low-resolution inputs, followed by the refinement of fine-scale Gaussians guided by high-resolution images. We separate static and dynamic regions and observe the dynamic parts. When inter-frame variations arise, optimization starts at the coarse scale and progressively refines finer scales, ensuring global structures and local details, as shown in Fig. 2(a).

We define a multi-scale structure with L levels of scale and l corresponds to a specific level. The M training viewpoint at time t with original resolution as $I _ { 0 , t } , . . . , I _ { M - 1 , t } .$ The viewpoint resolution at the level l is denoted as $I _ { 0 , t } ^ { l } , . . . , I _ { M - 1 , t } ^ { l } .$ At each level l, the corresponding set of Gaussians $\mathcal { G } ^ { ( l ) }$ is supervised by viewpoints $I _ { 0 , t } ^ { l } , . . . , I _ { M - 1 , t } ^ { l } .$ Considering that variations in Gaussian representations between temporally adjacent frames of the same scene are limited, the distribution learned from the initial frame can reasonably approximate that of all frames. Thus, we initialize the 3DGS on the first frame to estimate the scale of each Gaussian. We define the maximum scale $s _ { \mathrm { m a x } } ^ { ( 0 ) }$ , minimum $s _ { \mathrm { m i n } } ^ { ( 0 ) }$ , and mean $s _ { \mathrm { m e a n } } ^ { ( 0 ) }$ at each level l.

Given that Gaussian spheres with larger scales encode less fine-grained detail, we employ a binary partitioning strategy to divide the scale space. Specifically, once the scale range for a given level is established, the subsequent level is recursively defined within the finer half of the current levelâs range. Specifically, we assume the size of level l is defined as $[ s _ { \mathrm { m i n } } ^ { ( l ) ^ { \bullet } } , s _ { \mathrm { m a x } } ^ { ( l ) } ]$ . If the level $l + 1$ is required, the scale of level l is revised to $[ s _ { \mathrm { m e a n } } ^ { ( l ) } , s _ { \mathrm { m a x } } ^ { ( l ) } ]$ and the the scale of level l + 1 is revised to $\big [ s _ { \mathrm { m i n } } ^ { ( l ) } , s _ { \mathrm { m e a n } } ^ { ( l ) } \big ] , \mathrm { i . e . , ~ } \big [ s _ { \mathrm { m i n } } ^ { ( l + 1 ) } , s _ { \mathrm { m a x } } ^ { ( l + 1 ) } \big ] \gets$ $\bigl [ s _ { \mathrm { m i n } } ^ { ( l ) } , s _ { \mathrm { m e a n } } ^ { ( l ) } \bigr ] , \quad \bigl [ s _ { \mathrm { m i n } } ^ { ( l ) } , s _ { \mathrm { m a x } } ^ { ( l ) } \bigr ] \ \gets \ \bigl [ s _ { \mathrm { m e a n } } ^ { ( l ) } , s _ { \mathrm { m a x } } ^ { ( l ) } \bigr ]$ . This ensures that higher-indexed levels (higher resolution) consistently receive smaller Gaussian scale ranges.

We introduce a clamp function to ensure that Gaussians at each level are constrained within their designated scale ranges, which is represented as clamp(, ), to enforce both upper and lower bounds on the scale of the Gaussians, i.e.,

$$
s _ { i } = \mathrm { c l a m p } ( s _ { \mathrm { m i n } } ^ { ( l ) } , s _ { \mathrm { m a x } } ^ { ( l ) } , s _ { i } ) , \mathrm { i f } \ s _ { i } \in l .\tag{5}
$$

<!-- image-->  
Fig. 3. The detail of hybrid deformation-spawning strategy across multiscale levels. At level l, Gaussians undergo temporal deformation via MLPs. l When the average gradient exceeds the threshold , the next level l + 1 is leactivated for finer-grained optimization. Meanwhile, within each subspace at nelevel l, when the mean gradient exceeds the threshold, new Gaussians are eezspawned within that subspace. Conversely, when the average gradient is less Fthan the threshold, the hierarchical progression stops and multi-scale fusion is performed to integrate representations across all active levels.

According to Eq. (5), scale parameters at each level are constrained within their designated ranges, ensuring that coarser and finer Gaussians are optimized independently.

## C. Hybrid Deformation-spawning Gaussian Optimization

As the scene changes over time, existing methods rely on either deformation, which lacks expressiveness for complex dynamics, or spawning, which is computationally inefficient due to the need for many new Gaussians. To overcome the limitations, we propose a hybrid approach that combines deformation and spawning policy. After performing deformation at scale level l, unresolved dynamicsâeither from higherresolution variations or newly emerged componentsâare addressed by guided spawning at the current scale and activating deformation at the l + 1, as shown in Fig. 2(b).

We use the anchor v as an example to present the following description. For scale level l, the deformation process is processed through two lightweight Multi-Layer Perceptrons (MLPs) to model the dynamics of each Gaussian from t â 1 to t. We adopt a decoupled design to separately model geometry and appearance. Geometric deformation leverages spatial context via hash encoding to capture 3D structure, while photometric changesâreflecting intrinsic material propertiesâare processed directly without spatial encoding. The framework of Gaussian deformation is presented as Fig. 2(c). At each level l, an geometric deformation MLP $\mathrm { ( M L P _ { g } ) }$ takes the previous frameâs center position $\mu _ { i , t - 1 } ^ { ( l ) }$ as input, which is encoded via multi-scale hash encoding $h ( \mu _ { i , t - 1 } ^ { ( l ) } )$ to capture both local and global spatial context. The $\mathrm { M L P _ { g } }$ processes the hash-encoded position to predict geometric changes

$$
\Delta \mu _ { i } ^ { ( l ) } , \Delta q _ { i } ^ { ( l ) } = \mathrm { M L P _ { g } } ( h ( \mu _ { i , t - 1 } ^ { ( l ) } ) ) .\tag{6}
$$

The output is a 7-dimensional vector, where the first 3 dimensions represent the position offset $\Delta \mu _ { i } ^ { ( l ) }$ and the last 4 dimensions denote the quaternion increment $\Delta q _ { i } ^ { ( l ) }$

The appearance deformation MLP (MLPa) receives the original color $c _ { i , t - 1 } ^ { ( l ) }$ and opacity $\alpha _ { i , t - 1 } ^ { ( l ) }$ as inputs, and predicts the updated values $c _ { i , t } ^ { ( l ) }$ and $\alpha _ { i , t } ^ { ( l ) }$ , i.e.,

$$
\begin{array} { r } { c _ { i , t } ^ { ( l ) } , \alpha _ { i , t } ^ { ( l ) } = \mathrm { M L P _ { a } } ( c _ { i , t - 1 } ^ { ( l ) } , \alpha _ { i , t - 1 } ^ { ( l ) } ) . } \end{array}\tag{7}
$$

Overall, we model temporal change of all Gaussian attributes at level l via residual updates from the previous frame

$$
\begin{array} { r l } & { \theta _ { i } ^ { ( l , t ) } = \theta _ { i } ^ { ( l , t - 1 ) } + \Delta \theta _ { i } ^ { ( l ) } , \quad \theta \in \{ \mu , \Sigma , \alpha , \mathbf { c } \} , } \\ & { \quad q _ { i , t } ^ { ( l ) } = \mathrm { n o r m } ( q _ { i , t - 1 } ^ { ( l ) } ) \cdot \mathrm { n o r m } ( \Delta q _ { i } ^ { ( l ) } ) , } \end{array}\tag{8}
$$

where norm(Â·) indicates quaternion normalization to ensure unit quaternion constraints for valid rotations.

The rendering of scale level l follows the volume rendering approach where each level processes its own Gaussians independently

$$
C ^ { ( l ) } ( \boldsymbol { x } ^ { \prime } ) = \sum _ { i \in N } c _ { i , t } ^ { ( l ) } \alpha _ { i , t } ^ { ( l ) } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j , t } ^ { ( l ) } ) .\tag{9}
$$

The reconstruction loss at scale level l is computed with respect to the corresponding resolution images, including an $\ell _ { 1 }$ loss and a structural similarity loss to enforce perceptual fidelity.

$$
\mathcal { L } ^ { ( l ) } = \mathcal { L } _ { 1 } ^ { ( l ) } ( I _ { n , t } ^ { l } , \hat { C } _ { n , t } ^ { l } ) + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } } ^ { ( l ) } ( I _ { n , t } ^ { l } , \hat { C } _ { n , t } ^ { l } ) ,\tag{10}
$$

where $\hat { C } _ { n , t }$ indicates the volume rendering result of viewpoint n at time t with Gaussian scale level l.

For each anchor v at level l, we compute the average gradients of constituent Gaussians over d training iterations, denoted as $\nabla g _ { v } ^ { ( l ) }$ , denoted as

$$
\nabla g _ { v } ^ { ( l ) } = \frac { 1 } { \left| G _ { v } ^ { l } \right| } \sum _ { i \in G _ { v } ^ { l } } \left\| \frac { \partial \mathcal { L } } { \partial \theta _ { i } ^ { ( l ) } } \right\| ,\tag{11}
$$

where $G _ { v } ^ { l }$ represents the set of Gaussians belonging to anchor v at scale level l.

We define a level-specific gradient thresholds [28], denoted as

$$
\tau _ { \mathrm { a d d } } ^ { ( l ) } = \frac { V o l } { 4 ^ { l - 1 } } ,\tag{12}
$$

where V ol indicates the volume size of each anchor and the threshold decreases exponentially with scale level l, guiding progressively finer control at higher resolution levels. If $\begin{array} { r } { \nabla g _ { v } ^ { ( l ) } > \tau _ { \mathrm { a d d } } ^ { ( l ) } } \end{array}$ after deformation at $l ,$ suggesting that the resolution of l is not sufficient to capture the underlying dynamic changes, the Scale-GS framework triggers octree Gaussian Spawning at l and deformation at l + 1. The detail of hybrid deformation-spawning strategy across multi-scale is represented as Fig. 3.

Octree Gaussian Spawning of Scale l. As shown in Fig. 2(d), for the Spawning of level l, Gaussians are spawned at predefined anchor locations, followed by an optimization phase to adapt the newly added Gaussians to the dynamic scene. After training, these Gaussians are associated with their corresponding anchors and carried forward for overfitting in the subsequent frame.

We construct an octree-based Gaussian representation to model dynamics with minimal Gaussian usage. The octree structure enables adaptive allocation of Gaussian guided by gradient information. We refer to each region partitioned by the octree within an anchor space as a subspace. At each level l, we compute the mean gradient $\nabla g _ { \mathrm { v } } ^ { ( l ) }$ of all Gaussians within each subspace. If $\nabla g _ { \mathrm { v } } ^ { ( \zeta ) } > \tau _ { \mathrm { a d d } } ^ { ( l ) } ,$ a fixed number of

Gaussians are randomly assigned within the subspace, which is then recursively subdivided. This process continues until all subspaces fall below the threshold or the spatial resolution reaches one-thousandth of the original domain.

Gaussian Deformation of Scale l + 1. To enable finergrained deformation modeling, the optimization process activates Gaussians at the next resolution level l+1. These Gaussians are inherited from frame tâ1 and share the same anchor structure as level l, but operate at a higher spatial resolution through hierarchical refinement.

## D. Redundant Gaussian Removing

As the hybrid deformation-spawning strategy introduces additional Gaussians in each frame, a redundant Gaussian removing step is applied after all scales converging to prevent uncontrolled growth in the number of Gaussian over time. For each Gaussian i managed by anchor v, we define a 1-D optimizable mask $M _ { v , i }$ . The $M _ { v , i }$ is passed through a sigmoid function $\sigma ( \cdot )$ to ensure differentiability. When $M _ { v , i }$ is large, $\sigma ( M _ { v , i } )  1 ;$ conversely, when $M _ { v , i }$ is small, $\sigma ( M _ { v , i } )  0$ effectively removing the corresponding Gaussian.

For rendering pixels covered by anchor v, the projection color after masking is defined as:

$$
C ( x ^ { \prime } ) = \sum _ { i \in N } \sigma ( M _ { v , i } ) c _ { i , t } \alpha _ { i , t } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma ( M _ { v , i } ) \alpha _ { j , t } ) .\tag{13}
$$

When the sigmoid value $\sigma ( M _ { v , i } )$ approaches zero, the corresponding Gaussian i under anchor v is excluded from the volume rendering process, regardless of the scale level from which it originated. Thus, the loss function is then employed to optimize the Gaussians, filtering out redundant ones. The total loss combines the reconstruction error from the multiscale fused rendering with a sparsity regularization term, i.e.,

$$
\mathcal { L } = \sum _ { l = 1 } ^ { L } \mathcal { L } ^ { ( l ) } + \lambda _ { \mathrm { r } } \sum _ { v \in V } \sum _ { i = 1 } ^ { N } \sigma ( M _ { v , i } ) ,\tag{14}
$$

where $\boldsymbol { \mathcal { L } ^ { ( l ) } }$ represents the rendering loss on each scale, and $\lambda _ { \mathrm { r } }$ controls the redundancy removing regularization. The redundancy removing term reduces redundant Gaussians by penalizing non-zero mask values, thereby promoting a compact representation across all anchors and scale levels.

## E. Bidirectional Adaptive Masking

We propose a bidirectional adaptive masking mechanism that selects dynamic spatial anchors and informative camera views for each training frame, serving as a preprocessing step to reduce redundant computation in static regions and mitigate supervision from views with limited marginal utility.

The forward masking process selects dynamic anchors by distinguishing motion across adjacent multi-view frames. Specifically, temporal variations between two consecutive frames enable a coarse estimation of spatial motion for individual image pixels. The motion is then back-projected into 3D space to identify spatial position with significant displacement. If an anchorâs coverage consistently exhibits prominent motion patterns over multiple frames, it is designated as a dynamic anchor, which subsequently guides the optimization of dynamic objects.

Algorithm 1 The optimization of Scale-GS   
Require: Prior Gaussians $\overline { { { \mathcal { G } } _ { t - 1 } } } .$ , frame $I _ { t } ,$ scale levels $\overline { { L , } }$   
thresholds $\tau _ { \mathrm { a d d } } ^ { ( l ) }$   
Ensure: Updated $\mathcal { G } _ { t } ;$   
1: Initialize: $\mathcal { G } _ { t }  \mathcal { G } _ { t - 1 } ;$   
2: Apply bidirectional adaptive masking (III-E) to identify   
dynamic anchors $V ^ { \mathrm { d y n } }$ using Eq.(16);   
3: for each level $l = 1$ to L do   
4: Apply scale constraints using Eq.(5);   
5: for each dynamic anchor $v \in V ^ { \mathrm { d y n } }$ do   
6: Apply deformation using Eq.(6-7): geometric and   
appearance MLPs;   
7: Compute reconstruction loss using Eq.(10);   
8: Compute gradient $\nabla g _ { v } ^ { ( l ) }$ ;   
9: if $\nabla \dot { g } _ { v } ^ { ( l ) } > \tau _ { \mathrm { a d d } } ^ { ( l ) }$ (Eq.(12)) then   
10: Spawn Gaussians at scale $l ;$   
11: Activate deformation at level $l + 1 ;$   
12: else   
13: break;   
14: end if   
15: end for   
16: end for   
17: for each anchor $v \in V$ do   
18: Apply redundant Gaussian removing using Eq.(13) for   
masked rendering;   
19: end for   
20: Optimize total loss using Eq.(14);   
21: return Updated $\mathcal { G } _ { t } .$

The backward masking is designed to select a subset of camera views aligned with these dynamic regions. We introduce a relevance score for each camera view $c _ { k }$ . Let $\mathrm { I o U } ( c _ { k } , v )$ denote the normalized intersection-over-union between the image-space projection of anchor v and the field-of-view of camera $c _ { k }$ . Given a threshold $\tau _ { \mathrm { v i e w } }$ , we define the view relevance score as

$$
S ( c _ { k } ) = \sum _ { v \in V } \mathbf { 1 } \left[ \mathrm { I o U } ( c _ { k } , v ) > \tau _ { \mathrm { v i e w } } \right] \cdot \omega ( c _ { k } , v ) ,\tag{15}
$$

where $\mathbf { 1 } ( )$ is the indicator function and $\omega ( c _ { k } , v )$ is a direction weight, defined as

$$
\omega ( c _ { k } , v ) = | \mathbf n _ { v } ^ { \top } \mathbf d _ { c _ { k } } | ,\tag{16}
$$

where $\mathbf { n } _ { v }$ is the average normal vector of Gaussians in anchor $v , d _ { v }$ denotes the normal direction of the viewpoint $d _ { c _ { k } }$

Given the ranking of $S ( c _ { k } )$ , the top-ranked views are selected for training, as they are considered the most relevant to the dynamics. This bidirectional masking mechanism reduces computational redundancy caused by uninformative viewpoints of dynamic scenes.

## F. Algorithmic Summarization

We summarize Scale-GS dynamic scene optimization approach in Algorithm 1, which performs multi-scale guided dynamic updates through hybrid deformation-spawning optimization (III-C) combined with bidirectional adaptive masking (III-E).

The algorithm operates on the pre-initialized multiresolution Gaussian hierarchy, applying the hybrid deformation-spawning optimization strategy across multiple resolution levels while using bidirectional adaptive masking to focus computation on dynamic regions and informative views. This approach ensures efficient and targeted optimization for dynamic scene reconstruction while maintaining scale-aware structure throughout the process.

Implementation Details. All experiments are conducted in a virtual environment using Python 3.9 and PyTorch 2.1.0 as the primary deep learning framework. Additional libraries include plyfile 0.8.1 for data handling, torchaudio 0.12.1 for audio processing, and torchvision 0.13.1 for computer vision tasks. Training and inference were performed on a workstation equipped with an NVIDIA RTX 4090 GPU with CUDA 12.6 support, complemented by cudatoolkit 11.8 for GPU acceleration. The proposed method is implemented in PyTorch and trained under the above configuration, ensuring seamless integration of hardware capabilities and software functionalities for reliable experimental outcomes.

## IV. EXPERIMENT RESULTS

In this section, we thoroughly analyze the experiment results of Scale-GS. Specifically, we first introduce the dataset and the experimental setup. Later, we present the qualitative results and quantitative results. Finally, we conduct the ablation study to illustrate the effectiveness of each key module.

## A. Datasets

We evaluate Scale-GS method on three real-world dynamic scene datasets: the MeetRoom dataset [60], the NV3D (Neural 3D Video) dataset [67], and the Google Immersive Light Field Video dataset [68]. All of the datasets exhibit complex motion and occlusion patterns, thus suitable for evaluating the freeviewpoint rendering.

NV3D Dataset [67]. The Neural 3D Video dataset contains six dynamic scenes captured using a synchronized 21-camera array arranged in a semi-circular configuration. Each camera records the frames at a resolution of 2704 Ã 2028 with 300 frames, including various human motion interactions under indoor lighting conditions.

MeetRoom Dataset [60]. The MeetRoom dataset provides 3 indoor dynamic scenes recorded with 13 synchronized cameras, each capturing at $1 2 8 0 \times 7 2 0$ resolution with 300 frames. The subjects are performing structured activities such as sitting, walking, or conversing in an office-like setting.

Google Immersive Dataset [68]. This dataset contains 15 complex dynamic scenes captured using a high-fidelity immersive light field video rig consisting of 46 time-synchronized cameras on a 92 cm diameter hemisphere, each capturing at 2560 Ã 1920 resolution. The system supports a large viewing baseline (up to 80 cm) and a wide field of view (>220Â°), posing very challenging novel view synthesis scenes.

## B. Experimental Setup

Hyper-parameter settings. In the qualitative experiment and quantitative experiment, we adopt a three-level hierarchical structure $( L ~ = ~ 3 )$ , corresponding to an image resolution pyramid obtained by downsampling the original input to $\{ 1 / 4 , 1 / 2 , 1 \}$ for levels $l \ = \ 1 , 2 , 3 ,$ respectively. We set Î»SSIM = 0.2, $\begin{array} { r l r } { \lambda _ { \mathrm { r } } } & { { } = } & { 0 . 0 0 1 } \end{array}$ and the level-specific gradient thresholds $\tau _ { \mathrm { a d d } } ^ { ( l ) } ~ = ~ 0 . 0 1 / 4 ^ { l - 1 }$ , which yields $\{ \tau _ { \mathrm { a d d } } ^ { ( l ) } \} ~ =$ $\{ 0 . 0 1 , 0 . 0 0 2 5 , 0 . 0 0 0 6 2 5 \}$ for levels l = 1, 2, 3 respectively.

Metrics. We assess the rendering fidelity of Scale-GS method using three widely adopted perceptual and photometric metrics: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and LPIPS (Learned Perceptual Image Patch Similarity). These metrics respectively reflect pixellevel accuracy, structural coherence, and perceptual similarity. Furthermore, we compare the training time and rendering time to specify the efficiency of the proposed Scale-GS.

Baselines. We evaluate Scale-GS against five state-of-theart dynamic scene rendering methods that represent different approaches to temporal modeling and real-time performance.

Deformable 3D Gaussians [6] extends the static 3D Gaussian splatting framework to handle dynamic scenes through a canonical space representation. The method learns a set of 3D Gaussians in canonical space and captures temporal variations using a deformation field implemented as an MLP that predicts position, rotation, and scaling offsets. To address potential jitter from inaccurate camera poses, the authors introduce an annealing smooth training mechanism, enabling both highfidelity rendering quality and real-time performance.

4DGS [8] takes a different approach by incorporating a Gaussian deformation field network that operates on canonical 3D Gaussians. The method employs a spatial-temporal structure encoder coupled with a multi-head deformation decoder to predict Gaussian transformations across time. By modeling both Gaussian motion and shape changes through decomposed neural voxel encoding, 4DGS achieves efficient real-time rendering while maintaining temporal consistency.

3DGStream [10] focuses on streaming applications, enabling on-the-fly training for photo-realistic free-viewpoint videos. The method introduces a Neural Transformation Cache (NTC) to model 3D Gaussian transformations and employs an adaptive Gaussian spawn strategy for handling newly appearing objects. The framework operates through a two-stage pipeline: first training the NTC for existing Gaussians, then spawning and optimizing additional frame-specific Gaussians to accommodate emerging scene content.

HiCoM [61] presents a comprehensive framework specifically designed for streamable dynamic scene reconstruction. It combines three key components: a perturbation smoothing strategy for robust initial 3D Gaussian representation, a hierarchical coherent motion mechanism that captures multigranular motions through parameter sharing within regional hierarchies, and a continual refinement process that evolves scene content while maintaining representation compactness.

IGS [62] offers a generalized streaming framework centered around an Anchor-driven Gaussian Motion Network (AGM-Net). This network projects multi-view 2D motion features into 3D space using strategically placed anchor points to drive Gaussian motion. The method further incorporates a keyframe-guided streaming strategy that refines key frames and effectively mitigates error accumulation during long sequences.

<!-- image-->  
Fig. 4. Qualitative comparison results on the NV3D datasets(scene flame-steak). The frame index is 2, 151, and 183 from up to down. For each frame index, we present the result of 4DGS, 3DStream, HiCom, IGS, Scale-GS , and ground truth.

## C. Qualitative Results

We choose 2 scenes from NV3D and the Google Immersive dataset to conduct a comprehensive qualitative evaluation, including indoor scene and outdoor scene.

1) Comparison in Indoor Scene: Fig. 4 presents the visual result on the NV3D dataset on 3 progressive frame index. In this scenario, it is critical to analyze the motion of the human and the dog and accurately reconstruct the dynamics of the flame. According to Fig. 4, Scale-GS reconstructs significantly sharper and accurate fine-grained details compared to the baselines.

Specifically, in the frame 2, the 4DGS and 3DGStream fail to capture the trailing motion of hands, exhibiting noticeable blur. Meanwhile, four baselines render blurred occluded backgrounds, showing a large difference from the ground truth. The proposed Scale-GS not only reconstructs the clear texture of the blender, but renders the contour of the occluded object.

<!-- image-->  
Fig. 5. Qualitative comparison results on the Google Immersive datasets(scene Dog). The frame index is 1, 18 and 33 from up to down. For each frame index, we present the result of 4DGS, 3DStream, HiCom, IGS, Scale-GS , and ground truth.

In the frame 151, 4DGS and IGS fail to properly render the dogâs eyes. The 3DGStream and HiCom display obvious abnormal Gaussian points in the dogâs head area. Compared to the baselines, the Scale-GS renders a richly furred dog head without introducing floaters. The frame 183 indicates that the Scale-GS accurately captures the natural shape variations of the flame and intricate details. Although the baselines achieve flame rendering from a global perspective, they do not match the ground truth in the zoomed-in regions.

2) Comparison in Outdoor Scene: Fig. 5 presents outdoor comparative results on the Google Immersive dataset. Compared to indoor scenes, outdoor scenes cover a larger area, involve more rendering details, and are more challenging. The red bounding boxes highlight regions showing the dogâs head. In addition to reconstructing the changes in the dogâs facial features and head, it is necessary to render the complex fur.

According to Fig. 5, the proposed Scale-GS reconstruct clear and consistent face of the dog, while the baselines fail to accurately represent the dogâs facial contours. Specifically, 4DGS, 3DGStream, and HiCom exhibit strong blurring around the dogâs nose area in the frame 18. Moreover, the 3DGStream and HiCom methods display abnormal Gaussian representations at the dogâs eyebrows. In contrast, the subtle details around the eyes and nose of the Scale-GS are preserved with high fidelity, especially maintaining consistent quality across different viewing angles.

TABLE I  
The quantitative results on different datasets. NV, MR, and GI are separately denoted NV3D dataset, Meeting room dataset, and Google Immersive dataset. The best and second-best results are red and purple , respectively.
<table><tr><td></td><td colspan="3"> $\mathrm { S S I M _ { \uparrow } }$ </td><td colspan="3"> $\mathrm { P S N R } _ { \uparrow }$ </td><td colspan="3"> $\mathrm { L P I P S } _ { \downarrow }$ </td><td colspan="3">Training  $\mathrm { t i m e ( s ) _ { \downarrow } }$ </td><td colspan="3"> $\mathrm { F P S } _ { \uparrow }$ </td></tr><tr><td>Method</td><td>NV</td><td>MR</td><td>GI</td><td>NV</td><td>MR</td><td>GI</td><td>NV</td><td>MR</td><td>GI</td><td>NV</td><td>MR</td><td>GI</td><td>NV</td><td>MR</td><td>GI</td></tr><tr><td>Deform</td><td>0.956</td><td>0.857</td><td>0.843</td><td>32.10</td><td>27.81</td><td>26.46</td><td>0.127</td><td>0.206</td><td>0.216</td><td>38</td><td>29</td><td>81</td><td>40</td><td>45</td><td>35</td></tr><tr><td>4DGS</td><td>0.959</td><td>0.870</td><td>0.853</td><td>32.23</td><td>28.69</td><td>27.88</td><td>0.109</td><td>0.184</td><td>0.214</td><td>8.2</td><td>7.5</td><td>73.5</td><td>30</td><td>38</td><td>26</td></tr><tr><td>3DG-S</td><td>0.958</td><td>0.906</td><td>0.868</td><td>32.93</td><td>29.09</td><td>28.60</td><td>0.101</td><td>0.145</td><td>0.209</td><td>9.6</td><td>7.7</td><td>76.1</td><td>210</td><td>252</td><td>190</td></tr><tr><td>HiCoM</td><td>0.967</td><td>0.909</td><td>0.852</td><td>33.28</td><td>29.15</td><td>28.68</td><td>0.111</td><td>0.152</td><td>0.208</td><td>7.1</td><td>3.9</td><td>60.8</td><td>256</td><td>284</td><td>212</td></tr><tr><td>IGS</td><td>0.965</td><td>0.909</td><td>0.909</td><td>33.62</td><td>30.13</td><td>29.72</td><td>0.109</td><td>0.143</td><td>0.168</td><td>3.6</td><td>3.2</td><td>43.2</td><td>204</td><td>251</td><td>186</td></tr><tr><td>Scale-GS</td><td>0.966</td><td>0.908</td><td>0.912</td><td>34.47</td><td>31.58</td><td>31.18</td><td>0.106</td><td>0.142</td><td>0.169</td><td>3.2</td><td>3.0</td><td>37.3</td><td>274</td><td>276</td><td>199</td></tr></table>

Overall, Fig. 4 and Fig. 5 demonstrate that Scale-GS achieves superior visual quality and faithfulness compared to the baselines, particularly in challenging scenarios involving complex human interactions, dynamic elements like flame, and intricate textures such as animal fur.

## D. Quantitative Results

Tab. I presents the quantitative comparison to baselines on the 3 different datasets. We evaluate the novel view quality and rendering efficiency. To ensure a fair comparison, all competing methods are evaluated using the same set of Gaussians initialized from the 0-th frame, and the same variant of Gaussian splatting rasterization is applied consistently across all approaches.

According to Table I, Scale-GS outperforms the baselines and achieves the best reconstruction quality with the fastest training speed. In the rendering quality aspect, Scale-GS achieves a PSNR improvement from 33.62dB (second-best IGS) to 34.47dB on NV3D dataset, from 30.13dB to 31.58dB on Meeting Room dataset, and from 29.72dB to 31.18dB on Google Immersive dataset, demonstrating consistent quality enhancement across all evaluation scenarios. For SSIM metrics, Scale-GS achieves competitive performance with 0.912 on Google Immersive dataset (best), 0.966 on NV3D dataset (0.001 below the best), and 0.908 on Meeting Room dataset (0.001 below the best), indicating excellent structural preservation. The LPIPS scores show perceptual quality improvements, with Scale-GS achieving the best result of 0.142 on Meeting Room dataset and maintaining competitive performance on other datasets.

The sub-optimal performance of competing methods can be attributed to several limitations: (1) Global deformation inefficiency: Methods like Deform and 4DGS apply deformation to all Gaussians uniformly, leading to unnecessary computations on static regions and reduced optimization focus on truly dynamic areas. (2) Single-scale representation constraints: Traditional approaches like 3DG-S and HiCoM operate at fixed resolutions, failing to leverage hierarchical optimization strategies. In contrast, our multi-scale approach enables rapid convergence by first focusing on coarse-scale

Gaussians that capture major scene changes, then progressively refining finer-scale Gaussians with increasing precision. This hierarchical refinement allows the optimization to quickly identify and target only the Gaussians that require updates at each scale level, resulting in accelerated training convergence and progressively improved rendering quality as finer details are incorporated. (3) Lack of adaptive supervision: Existing methods fail to adaptively select the most informative views and spatial regions during training, resulting in suboptimal resource allocation and slower convergence. In contrast, our multi-scale framework with hybrid deformation-spawning policy and bidirectional adaptive masking addresses these limitations systematically, enabling both superior reconstruction quality and computational efficiency.

## E. Ablation Study

1) The Evaluation of Scale Number: We conduct a group of ablation study to investigate the influence of the scale number of Scale-GS. We choose the Face Print scene from the Google Immersive Dataset and set the scale to 2, 3, 4, and 5, respectively. The result of average SSIM, PSNR, LPIPS, and training time is demonstrated in Table II.

The experimental results indicate that the configuration with 3 scales achieves the best performance across all metrics, yielding an SSIM of 0.916, a PSNR of 31.233 dB, and an LPIPS score of 0.154. It is noted that with the increase of the scale number, the rendering quality may decline under the same training iterations. This is because the redundant scale requires training computation, thus the convergence of each scale becomes insufficient. Therefore, 3 level of scales strike the tradeoff between visual quality and training efficiency.

To further validate the visual performance gap across different scale settings, in Fig. 6, we present a qualitative comparison of the zoom-in region in the same scene. It can be observed that the 3-scale configuration produces the most faithful reconstruction, particularly in facial regions such as the eyes, eyelashes, and hair strands, closely resembling the ground truth. The hand region near the drawing board is reconstructed the clearest details with the scale number as 3. In comparison, other scale configurations exhibit varying degrees of Gaussian transparency or blurring in the hand area, which negatively affects the rendering quality.

2) The Effectiveness of Hybrid Deformation-spawning: To evaluate the effectiveness of the hybrid deformation-spawning strategy, we conduct an ablation study on the MeetRoom dataset by comparing three variants: (1) the full hybrid strategy, (2) w/o deformation, and (3) w/o spawn. The results, summarized in Table III, show that the hybrid approach achieves the best performance across all evaluated metrics, in which SSIM is 0.913, PSNR is 31.602 dB, and LPIPS is 0.138, and the least training time per frame (2.9 seconds). In contrast, removing either component leads to a noticeable degradation in both reconstruction quality and efficiency.

TABLE II  
Ablation study of scale number on Google Immersive Dataset, where time indicates training time.
<table><tr><td>Scale</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>Time(s)â</td></tr><tr><td>2</td><td>0.893</td><td>30.602</td><td>0.168</td><td>36</td></tr><tr><td>3</td><td>0.916</td><td>31.233</td><td>0.154</td><td>33</td></tr><tr><td>4</td><td>0.881</td><td>29.917</td><td>0.170</td><td>35</td></tr><tr><td>5</td><td>0.857</td><td>28.242</td><td>0.191</td><td>39</td></tr></table>

\* Our setting is highlighted with a2ã3ã4ã5å°ºåº¦ shaded background .

<!-- image-->  
Fig. 6. The visual quality comparison with different numbers of scales. From top-left to bottom-right: results with 2, 3, 4, and 5 scales, respectively.

Furthermore, we demonstrate the visual quality(top) and the statistical PSNR and training time(bottom) comparison among the full hybrid strategy, without deformation, and without spawning in Fig. 7. According to the visual result, ours w/o deformation w/o spawnthe full hybrid strategy exhibits clear edge and accurate color representation. Moreover, the details of the plant leaves and the fabric textures are also well preserved. These visual results further validate that the hybrid mechanism achieves highquality rendering and efficient training of dynamic scenes.

As shown in Fig. 7, our hybrid approach consistently achieves the shortest training time per frame (averaging around 3.2 seconds), while both ablated variants require significantly longer training periods. Specifically, the âw/o spawnâ variant, though taking more time than our method, has relatively more stable training duration compared to âw/o deformationâ; the âw/o deformationâ variant shows considerable training time fluctuations throughout the sequence and exhibits the highest training costs. Fig. 7(b) demonstrates the PSNR evolution across frames, where our method maintains relatively stable and high PSNR values (around 34.5 dB). The âw/o spawnâ variant can keep a decent level of PSNR for a while but shows more pronounced fluctuations than our method as frames progress, while the âw/o deformationâ variant experiences a more gradual yet evident decline in PSNR over time and overall lower reconstruction quality. The results demonstrate that the Scale-GS method not only trains faster but also maintains more stable rendering quality throughout the sequence.

TABLE III  
Ablation study of the hybrid deformation-spawning policy, where time indicates training time.
<table><tr><td></td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>Time(s)â</td></tr><tr><td>deform + spawn</td><td>0.913</td><td>31.602</td><td>0.138</td><td>2.9</td></tr><tr><td>w/o deformation</td><td>0.854</td><td>26.719</td><td>0.245</td><td>4.6</td></tr><tr><td>w/o spawning</td><td>0.887</td><td>29.304</td><td>0.127</td><td>3.8</td></tr></table>

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 7. The visual quality(from left to right: full hybrid, w/o deformation, and w/o spawning) and statistical results of the ablation study on hybrid deformation-spawning in the Meeting Room dataset.

Even the dynamic nature of the scene leads to unstable PSNR variations, the hybrid deformation and spawning strategy performs the best in both visual quality and training efficiency in most cases. This suggests that the joint optimization of deformation and spawning enables more efficient and stable learning of dynamic scene representations.

3) Viewpoint Selection: To validate the effectiveness of Bidirectional Adaptive Masking (BAM) mechanism, we conduct ablation studies on five challenging scenes from the Google Immersive Dataset: Car, Goats, Dogs, Face Paint, and Welder. These scenes feature diverse dynamic content ranging from fast-moving objects to complex deformations, providing a comprehensive testbed for evaluating view selection strategy.

The quantitative results are presented in Table IV. Scale-GS BAM-based view selection consistently outperforms the baseline across all tested scenes, achieving improvements of 0.7-1.7 dB in PSNR and 0.007-0.035 in SSIM. The most significant improvement is observed in the Welder scene, where BAM selection achieves 30.281 dB PSNR compared to 28.552 dB without selection, representing a substantial 1.729 dB gain. This scene benefits particularly due to its complex welding sparks and rapid lighting changes, where targeted view selection effectively focuses learning on the most informative perspectives.

The improvement across diverse scene types demonstrates that BAM successfully identifies and prioritizes views that provide meaningful supervision for dynamic regions. By filtering out redundant or less informative viewpoints, the BAM not only improves reconstruction quality but also enhances training efficiency. The bidirectional maskingâselecting both spatial anchors and camera viewsâproves essential for handling the complexity of multi-view dynamic scene reconstruction.

TABLE IV  
The ablation study of viewpoint selection results, where time indicates training time.
<table><tr><td rowspan="2">Scene</td><td colspan="2">BAM selection</td><td colspan="2">w/o BAM selection</td></tr><tr><td> $\mathrm { P S N R } _ { \uparrow }$ </td><td>SSIMâ</td><td> $\mathrm { P S N R } _ { \uparrow }$ </td><td> $\mathrm { S S I M _ { \uparrow } }$ </td></tr><tr><td>Car</td><td> $3 1 . 3 2 0 _ { + 1 . 0 1 2 }$ </td><td> $0 . 9 1 7 _ { + 0 . 0 1 1 }$ </td><td>30.308</td><td>0.906</td></tr><tr><td>Goats</td><td> $3 1 . 7 7 8 _ { + 1 . 1 5 0 }$ </td><td> $0 . 9 2 2 _ { + 0 . 0 0 7 }$ </td><td>30.628</td><td>0.915</td></tr><tr><td>Dogs</td><td> $3 1 . 0 2 2 _ { + 0 . 7 3 7 }$ </td><td> $0 . 9 1 4 _ { + 0 . 0 2 1 }$ </td><td>30.285</td><td>0.893</td></tr><tr><td>Paint</td><td> $3 1 . 7 5 1 _ { + 0 . 9 0 6 }$ </td><td> $0 . 9 1 6 _ { + 0 . 0 0 6 }$ </td><td>30.845</td><td>0.910</td></tr><tr><td>Welder</td><td> $3 0 . 2 8 1 _ { + 1 . 7 2 9 }$ </td><td> $0 . 9 0 1 _ { + 0 . 0 3 5 }$ </td><td>28.552</td><td>0.866</td></tr></table>

These results validate that bidirectional view selection mechanism effectively ensure that Gaussians receive supervision from the most relevant camera perspectives, leading to more accurate and stable dynamic scene modeling.

## V. CONCLUSION AND FUTURE WORK.

Conclusion. In this work, we present Scale-GS, a scalable Gaussian Splatting framework for efficient and redundancyaware dynamic scene training. The proposed Scale-GS proposes the anchor-based multi-scale Gaussian representation integrating a hybrid deformationâspawning optimization, redundant Gaussian removing, and a bidirectional adaptive masking module. Our method effectively reduces the computational overhead of existing approaches by filtering static or irrelevant Gaussian spheres. The hybrid deformationâspawning strategy preserves the structured inference from deformation while enhancing the modelâs capacity to represent large-scale dynamic motions. Extensive experiments demonstrate that Scale-GS achieves superior visual fidelity and significantly accelerates training. The proposed framework opens up promising opportunities for remote immersive experiences such as VR, AR, and immersive video conferencing.

Future work. To further reduce the redundant computation and storage, we aim to incorporate semantic understanding into the multi-scale representation to improve the representation efficiency of Gaussian spheres. The semantic information guides the allocation and prioritization of Gaussian spheres by adapting their scale, density, and training frequency according to the semantic importance and structural complexity of different areas. Furthermore, integrating Scale-GS with neural compression could further enhance scalability and enable deployment in bandwidth and resource-constrained environments.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D Gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[2] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi, âScgs: Sparse-controlled gaussian splatting for editable dynamic scenes,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 4220â 4230.

[3] X. Tong, T. Shao, Y. Weng, Y. Yang, and K. Zhou, âAs-rigid-as-possible deformation of gaussian radiance fields,â IEEE Trans. Vis. Comput. Graph., 2025.

[4] Z. Fan, S.-S. Huang, Y. Zhang, D. Shang, J. Zhang, Y. Guo, and H. Huang, âRGAvatar: Relightable 4D Gaussian avatar from monocular videos,â IEEE Trans. Vis. Comput. Graph, 2025.

[5] R. Fan, J. Wu, X. Shi, L. Zhao, Q. Ma, and L. Wang, âFov-GS: Foveated 3D Gaussian splatting for dynamic scenes,â IEEE Trans. Vis. Comput. Graph, 2025.

[6] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, âDeformable 3D Gaussians for high-fidelity monocular dynamic scene reconstruction,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 20 331â20 341.

[7] Z. Li, Z. Chen, Z. Li, and Y. Xu, âSpacetime Gaussian feature splatting for real-time dynamic view synthesis,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 8508â8520.

[8] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 20 310â20 320.

[9] J. Yan, R. Peng, L. Tang, and R. Wang, â4D Gaussian splatting with scale-aware residual field and adaptive optimization for real-time rendering of temporally complex dynamic scenes,â in Proc. 32nd ACM Int. Conf. Multimedia, 2024, pp. 7871â7880.

[10] J. Sun, H. Jiao, G. Li, Z. Zhang, L. Zhao, and W. Xing, â3DGStream: On-the-fly training of 3D Gaussians for efficient streaming of photorealistic free-viewpoint videos,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 20 675â20 685.

[11] R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, and Y. Liu, âTensor4d: Efficient neural 4D decomposition for high-fidelity dynamic reconstruction and rendering,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 16 632â16 642.

[12] S. J. Gortler, R. Grzeszczuk, R. Szeliski, and M. F. Cohen, âThe lumigraph,â in Proc. 23rd Annu. Conf. Comput. Graph. Interactive Techn., 2023, pp. 453â464.

[13] X. Meng, R. Du, J. F. JaJa, and A. Varshney, â3D-kernel foveated rendering for light fields,â IEEE Trans. Vis. Comput. Graph, vol. 27, no. 8, pp. 3350â3360, 2020.

[14] M. Levoy and P. Hanrahan, âLight field rendering,â in Proc. 23rd Annu. Conf. Comput. Graph. Interactive Techn., 2023, pp. 441â452.

[15] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNeRF: Representing scenes as neural radiance fields for view synthesis,â Commun. ACM, vol. 65, no. 1, pp. 99â106, 2021.

[16] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, âTensorf: Tensorial radiance fields,â in Proc. Eur. Conf. Comput. Vis. Springer, 2022, pp. 333â350.

[17] X.-S. Hu, X.-Y. Lin, Y.-J. Liu, M.-H. Xiang, Y.-Q. Guo, Y. Xing, and Q.- H. Wang, âCulling-based real-time rendering with accurate ray sampling for high-resolution light field 3D display,â IEEE Trans. Vis. Comput. Graph, 2024.

[18] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM Trans. Graph., vol. 41, no. 4, pp. 1â15, 2022.

[19] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, âPlenoxels: Radiance fields without neural networks,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 5501â 5510.

[20] Z. Chen, T. Funkhouser, P. Hedman, and A. Tagliasacchi, âMobileNeRF: Exploiting the polygon rasterization pipeline for efficient neural field rendering on mobile architectures,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 16 569â16 578.

[21] K. Ye, H. Wu, X. Tong, and K. Zhou, âA real-time method for inserting virtual objects into neural radiance fields,â IEEE Trans. Vis. Comput. Graph, 2024.

[22] A. Yu, R. Li, M. Tancik, H. Li, R. Ng, and A. Kanazawa, âPlenoctrees for real-time rendering of neural radiance fields,â in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021, pp. 5752â5761.

[23] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-NeRF: A multiscale representation for antialiasing neural radiance fields,â in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021, pp. 5855â5864.

[24] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-NeRF 360: Unbounded anti-aliased neural radiance fields,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 5470â 5479.

[25] J. Lu, T. Shao, H. Wang, Y.-L. Yang, Y. Yang, and K. Zhou, âRelightable detailed human reconstruction from sparse flashlight images,â IEEE Trans. Vis. Comput. Graph, 2024.

[26] F. Wimbauer, N. Yang, C. Rupprecht, and D. Cremers, âBehind the scenes: Density fields for single view reconstruction,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 9076â9086.

[27] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, âpixelNeRF: Neural radiance fields from one or few images,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 4578â4587.

[28] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffold-GS: Structured 3D Gaussians for view-adaptive rendering,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 20 654â 20 664.

[29] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, âOctree-GS: Towards consistent real-time rendering with lod-structured 3D Gaussians,â arXiv:2403.17898, 2024.

[30] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao, and G. Zhang, âPGSR: Planar-based Gaussian splatting for efficient and high-fidelity surface reconstruction,â IEEE Trans. Vis. Comput. Graph, 2024.

[31] Z. Yu, T. Sattler, and A. Geiger, âGaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes,â ACM Trans. Graph., vol. 43, no. 6, pp. 1â13, 2024.

[32] J. Lin, J. Gu, L. Fan, B. Wu, Y. Lou, R. Chen, L. Liu, and J. Ye, âHybridGS: Decoupling transients and statics with 2D and 3D Gaussian splatting,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 788â797.

[33] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, âHac: Hash-grid assisted context for 3D Gaussian splatting compression,â in Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 422â438.

[34] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, âCompact 3D Gaussian representation for radiance field,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 21 719â21 728.

[35] S. Niedermayr, J. Stumpfegger, and R. Westermann, âCompressed 3D Gaussian splatting for accelerated novel view synthesis,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 10 349â 10 358.

[36] D. Li, S.-S. Huang, and H. Huang, âMPGS: Multi-plane Gaussian splatting for compact scenes rendering,â IEEE Trans. Vis. Comput. Graph, 2025.

[37] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, âColmapfree 3D Gaussian splatting,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 20 796â20 805.

[38] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu, âLGM: Large multi-view Gaussian model for high-resolution 3D content creation,â in Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 1â18.

[39] K. Tang, S. Yao, and C. Wang, âiVR-GS: Inverse volume rendering for explorable visualization via editable 3D Gaussian splatting,â IEEE Trans. Vis. Comput. Graph, 2025.

[40] Z.-X. Zou, Z. Yu, Y.-C. Guo, Y. Li, D. Liang, Y.-P. Cao, and S.-H. Zhang, âTriplane meets Gaussian splatting: Fast and generalizable single-view 3D reconstruction with transformers,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 10 324â10 335.

[41] A. Chen, Z. Xu, F. Zhao, X. Zhang, F. Xiang, J. Yu, and H. Su, âMVSNeRF: Fast generalizable radiance field reconstruction from multiview stereo,â in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021, pp. 14 124â14 133.

[42] M. M. Johari, Y. Lepoittevin, and F. Fleuret, âGeoNeRF: Generalizing NeRF with geometry priors,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 18 365â18 375.

[43] S. Szymanowicz, C. Rupprecht, and A. Vedaldi, âSplatter image: Ultrafast single-view 3D reconstruction,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 10 208â10 217.

[44] C. Zhang, Y. Zou, Z. Li, M. Yi, and H. Wang, âTransplat: Generalizable 3D Gaussian splatting from sparse multi-view images with transformers,â in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 9, 2025, pp. 9869â 9877.

[45] S. Zheng, B. Zhou, R. Shao, B. Liu, S. Zhang, L. Nie, and Y. Liu, âGPS-Gaussian: Generalizable pixel-wise 3D Gaussian splatting for real-time human novel view synthesis,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 19 680â19 690.

[46] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, âpixelSplat: 3D Gaussian splats from image pairs for scalable generalizable 3D reconstruction,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 19 457â19 467.

[47] H. Xu, S. Peng, F. Wang, H. Blum, D. Barath, A. Geiger, and M. Pollefeys, âDepthSplat: Connecting gaussian splatting and depth,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 16 453â16 463.

[48] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.- J. Cham, and J. Cai, âMVSplat: Efficient 3D Gaussian splatting from sparse multi-view images,â in Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 370â386.

[49] T. Liu, G. Wang, S. Hu, L. Shen, X. Ye, Y. Zang, Z. Cao, W. Li, and Z. Liu, âMVSGaussian: Fast generalizable gaussian splatting reconstruction from multi-view stereo,â in Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 37â53.

[50] K. Zhang, S. Bi, H. Tan, Y. Xiangli, N. Zhao, K. Sunkavalli, and Z. Xu, âGS-LRM: Large reconstruction model for 3D Gaussian splatting,â in Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 1â19.

[51] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, âMVSNet: Depth inference for unstructured multi-view stereo,â in Proc. Eur. Conf. Comput. Vis., 2018, pp. 767â783.

[52] A. Cao and J. Johnson, âHexPlane: A fast representation for dynamic scenes,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 130â141.

[53] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, âK-planes: Explicit radiance fields in space, time, and appearance,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 12 479â12 488.

[54] X. Guo, J. Sun, Y. Dai, G. Chen, X. Ye, X. Tan, E. Ding, Y. Zhang, and J. Wang, âForward flow for novel view synthesis of dynamic scenes,â in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2023, pp. 16 022â16 033.

[55] H. Lin, S. Peng, Z. Xu, T. Xie, X. He, H. Bao, and X. Zhou, âHighfidelity and real-time novel view synthesis for dynamic scenes,â in SIGGRAPH Asia 2023 Conference Papers, 2023, pp. 1â9.

[56] J.-W. Liu, Y.-P. Cao, W. Mao, W. Zhang, D. J. Zhang, J. Keppo, Y. Shan, X. Qie, and M. Z. Shou, âDeVRF: Fast deformable voxel radiance fields for dynamic scenes,â Adv. Neural Inf. Process. Syst., vol. 35, pp. 36 762â 36 775, 2022.

[57] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, âD-NeRF: Neural radiance fields for dynamic scenes,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 10 318â10 327.

[58] L. Wang, Q. Hu, Q. He, Z. Wang, J. Yu, T. Tuytelaars, L. Xu, and M. Wu, âNeural residual radiance fields for streamably free-viewpoint videos,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 76â87.

[59] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y. Xu, and A. Geiger, âNeRFPlayer: A streamable dynamic scene representation with decomposed neural radiance fields,â IEEE Trans. Vis. Comput. Graph, vol. 29, no. 5, pp. 2732â2742, 2023.

[60] L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan, âStreaming radiance fields for 3D video synthesis,â Adv. Neural Inf. Process. Syst., vol. 35, pp. 13 485â13 498, 2022.

[61] Q. Gao, J. Meng, C. Wen, J. Chen, and J. Zhang, âHiCoM: Hierarchical coherent motion for dynamic streamable scenes with 3D Gaussian splatting,â Adv. Neural Inf. Process. Syst., vol. 37, pp. 80 609â80 633, 2024.

[62] J. Yan, R. Peng, Z. Wang, L. Tang, J. Yang, J. Liang, J. Wu, and R. Wang, âInstant Gaussian stream: Fast and generalizable streaming of dynamic scene reconstruction via gaussian splatting,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 16 520â16 531.

[63] Z. Guo, W. Zhou, L. Li, M. Wang, and H. Li, âMotion-aware 3D Gaussian splatting for efficient dynamic scene reconstruction,â IEEE Trans. Circuits Syst. Video Technol., 2024.

[64] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, âDynamic 3D Gaussians: Tracking by persistent dynamic view synthesis,â in Proc. IEEE Int. Conf. 3D Vis. IEEE, 2024, pp. 800â809.

[65] T. Mizuho, T. Narumi, and H. Kuzuoka, âReduction of forgetting by contextual variation during encoding using 360-degree video-based immersive virtual environments,â IEEE Trans. Vis. Comput. Graph, 2024.

[66] C. Groth, S. Fricke, S. Castillo, and M. Magnor, âWavelet-based fast decoding of 360 videos,â IEEE Trans. Vis. Comput. Graph, vol. 29, no. 5, pp. 2508â2516, 2023.

[67] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim, T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe et al., âNeural 3D video synthesis from multi-view video,â in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 5521â5531.

[68] M. Broxton, J. Flynn, R. Overbeck, D. Erickson, P. Hedman, M. Duvall, J. Dourgarian, J. Busch, M. Whalen, and P. Debevec, âImmersive light field video with a layered mesh representation,â ACM Trans. Graph., vol. 39, no. 4, pp. 86â1, 2020.