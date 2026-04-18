# Touch-Augmented Gaussian Splatting for Enhanced 3D Scene Reconstruction

Yuchen Gaoa, Xiao Xub, Eckehard Steinbachb, Daniel E. Lucania, Qi Zhanga

aDepartment of Electrical and Computer Engineering, Aarhus University, Aarhus, Denmark

bTUM School of Computation, Information and Technology, Technical University of Munich, Munich, Germany Email: {yuchen, daniel.lucani, qz}@ece.au.dk, {xiao.xu, eckehard.steinbach}@tum.de

AbstractâThis paper presents a multimodal framework that integrates touch signals (contact points and surface normals) into 3D Gaussian Splatting (3DGS). Our approach enhances scene reconstruction, particularly under challenging conditions like low lighting, limited camera viewpoints, and occlusions. Different from the visual-only method, the proposed approach incorporates spatially selective touch measurements to refine both the geometry and appearance of the 3D Gaussian representation. To guide the touch exploration, we introduce a two-stage sampling scheme that initially probes sparse regions and then concentrates on high-uncertainty boundaries identified from the reconstructed mesh. A geometric loss is proposed to ensure surface smoothness, resulting in improved geometry. Experimental results across diverse scenarios show consistent improvements in geometric accuracy. In the most challenging case with severe occlusion, the Chamfer Distance is reduced by over 15Ã, demonstrating the effectiveness of integrating touch cues into 3D Gaussian Splatting. Furthermore, our approach maintains a fully online pipeline, underscoring its feasibility in visually degraded environments.

Index TermsâTouch-Aided Reconstruction, Multimodal Data Integration, 3D Reconstruction

## I. INTRODUCTION

Across diverse fields, from robotic manipulation [1] and industrial inspection [2] to the emerging Tactile Internet (TI) [3], [4], there is a growing need to transcend the physical boundaries by building a 3D model of the environment. It allows users to perceive and interact with hazardous or inaccessible scenes. The core of this need lies in fast and reliable 3D reconstruction that strikes a balance among geometric accuracy, robustness, and time. Unlike classical point-to-point teleoperation [5], [6], these applications prioritize holistic environmental understanding, both visually and geometrically. While vision-based methods form the backbone of such tasks, visual-only methods are unreliable when image data is incomplete. To be specific, holistic environmental understanding must address three key challenges: (1) highly efficient incremental updates; (2) robustness to lighting/occlusion degradations; (3) adaptive refinement of uncertain regions.

To meet these challenges, the fidelity and efficiency of a 3D reconstruction method play critical roles. However, traditional point cloud-based approaches [11] lack surface topology, especially when the input is sparse, and the visual quality can be compromised. While mesh-based approaches accurately capture the geometry and physics of the model, they are not flexible in updates, and remeshing can be costly [19]. Meanwhile, advanced neural implicit representations such as

NeRF [12] and InstantNGP [13] enable photorealistic rendering but may struggle to adapt quickly to dynamic changes.

In contrast, 3D Gaussian Splatting (3DGS) [10] employs anisotropic Gaussian primitives to represent scenes explicitly while supporting differentiable optimization. We choose 3DGS because (i) each Gaussian can be inserted or pruned in O(1), matching per-touch online update loop in Section III, (ii) the analytical covariance gives a closed-form directional radius in Eq. (11) that our geometric loss differentiates cheaply to guide incremental updates. However, like all vision-centric methods, 3DGS depends on photometric consistency. Sparse yet reliable touch points resolve these ambiguities. While visual information is easy to obtain with todayâs wealth of camera solutions, it may miss fine details. In contrast, touch information is more accurate but less efficient to acquire.

While recent visionâtouch studies show that multimodal cues help basic manipulation and offline shape completion, they seldom model environmental degradation due to poor lighting, occlusion, or limited viewpoints. Early teleoperationoriented approaches by Xu et al. [7], [8] ensured haptic stability through a local proxy model, but paid less attention to fine-grained scene geometry. Antonsen et al. [9] fused contact data into a global signed-distance field, incurring expensive full-volume updates. More recent systems exploit high-resolution tactile skins or task-specific pipelines: Touch-Fusion [14] fuses GelSight measurements with RGB-D offline. Yi et al. [15] and Ottenhaus et al. [16] drive an active probe with Gaussian-process or implicit-surface uncertainty, while Rustler et al. [17] focus on shape completion for grasp success. All rely on special hardware or offline pipelines, so they are not online-ready.

Crucially, the above work is tuned for task completion in teleoperation (i.e., maximizing grasp or insertion success), whereas our goal is to deliver a visually, metrically accurate environment model that users can inspect even when vision degrades. Most touch-aided systems (e.g. TouchFusion [14], Yi et al. [15]) optimize these task-level rewards and run offline. None of those systems provides guarantees on reconstruction fidelity under poor lighting, occlusion, or restricted viewpoints, a gap that our touch-augmented 3DGS framework tries to bridge. Thus, direct metric-wise comparison would be misleading.

The main contributions are summarized as follows:

â¢ Introduction of an online training pipeline that integrates

<!-- image-->  
Fig. 1. Overview of the touch-augmented 3D Gaussian Splatting framework. (1) Depth cameras capture images to generate an initial point cloud (2). This point cloud is then represented using Gaussian ellipsoids. RGB cameras acquire training images $I _ { v _ { i } } ( 3 ) ,$ , compared against the rendered images $\hat { I } _ { v _ { i } }$ (4). A touch sampling strategy (5) refines the Gaussian model in areas of sparse coverage or high uncertainty, spawning new Gaussians at xi where contact points and normals are acquired (6). The overall loss function consists of both an image-based loss (comparing $I _ { v _ { i } }$ and $\hat { I } _ { v _ { i } } )$ and a geometric loss (enforcing local surface consistency from the contact points and normals), which improve the 3D reconstruction.

3DGS with selective touch patches to incrementally refine geometry, enhancing the reliability of visual-based reconstruction under hazardous environments.

â¢ Design of an iterative touch sampling strategy that identifies under-reconstructed regions and injects localized touch measurements, enforcing consistency among neighboring Gaussian primitives.

The proposed method achieves a maximum of a 15Ã reduction in the Chamfer Distance (CD), improves the F-score by as much as 43%, and decreases the Jensen-Shannon Divergence (JSD) by up to 88% across various scenarios. It demonstrates the feasibility of rendering a comprehensible scene in an environment with poor visibility.

## II. BACKGROUND

We will explain the fundamentals of 3DGS and touch sampling in this section.

## A. 3D Gaussian Splatting

1) Mathematical Representation: 3DGS is a differentiable rendering technique that represents three-dimensional scenes using Gaussian primitives. Each Gaussian is defined by its center $\mu \in \mathbb { R } ^ { 3 }$ and its covariance matrix $\textbf { \textsf { E } } \in \ \bar { \mathbb { R } ^ { 3 \times 3 } }$ , which encodes the spatial extent and anisotropic shape of the Gaussian. The Gaussian function is expressed as [8]:

$$
\mathcal { G } ( \mathbf { x } ) = \exp \left( - \frac { 1 } { 2 } ( \mathbf { x } - \boldsymbol { \mu } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ( \mathbf { x } - \boldsymbol { \mu } ) \right) ,\tag{1}
$$

where x is the coordinate of the points on Gaussianâs surface. 2) Covariance Matrix Parameterization: To ensure that the covariance matrix Î£ remains positive semi-definite during optimization, it is parameterized as $[ 8 ] \colon \Sigma = \mathbf { R S S } ^ { \top } \mathbf { R } ^ { \top }$ , where: $\mathbf { S } \in \mathbb { R } ^ { 3 \times 3 }$ is a diagonal scaling matrix that determines the anisotropic spread of the Gaussian. R is a Special Orthogonal (SO(3)) rotation matrix representing the orientation.

3) Projection and Rendering: To render 3D Gaussians in 2D, the covariance matrix Î£ is transformed into the camera coordinate system using a viewing transformation W. The resulting covariance matrix $\Sigma _ { \mathbf { p } }$ is:

$$
\begin{array} { r } { \pmb { \Sigma } _ { p } = \mathbf { J } \mathbf { W } \pmb { \Sigma } \mathbf { W } ^ { \top } \mathbf { J } ^ { \top } , } \end{array}\tag{2}
$$

where J is the Jacobian of the projective transformation.

The rendering corresponds to step 4 in Fig. 1, it employs Î±-blending, where the color c of a pixel is computed as [8]:

$$
\mathbf { c } = \sum _ { i = 1 } ^ { N } T _ { i } \alpha _ { i } \mathbf { c _ { i } } ,\tag{3}
$$

where $\alpha _ { i } = 1 - \exp ( - \sigma _ { i } \delta _ { i } )$ is per-sample opacity (derived from density $\sigma _ { i } )$ , and the transmittance is $\begin{array} { r } { T _ { i } = \prod _ { i = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } \end{array}$ Color c is taken along the ray with intervals Î´. ci is the color contribution of the i-th Gaussian.

## B. Touch sampling strategy

Haptic feedback has been widely explored in digital reconstruction and modeling to enhance geometry understanding. The key is to decide where to touch in terms of both locating the object and deciding where the next sample patch should be [14]â[17]. Previous works proposed mainly two ways of sampling decisions: uncertainty-driven solutions and random selection [14]. Despite the fact that random selection is easier, it is less efficient and less explainable than the uncertaintydriven solutions. For example, Z. Yi et al. [15] model the surface using the Gaussian process to decide the uncertain region to sample. S. Ottenhaus et al. [16] utilize a constantly updated implicit surface, and sampling points are dynamically adjusted to prioritize the exploration of uncovered or errorprone regions. L. Rustler et al. [17] extend the concept of implicit surface guiding sampling to incorporate both touch and visual for more efficient information extraction and a higher success rate of task completion.

## III. TOUCH-AUGMENTED 3D GAUSSIAN SPLATTING

In this section, we detail the workflow of touch-augmented 3DGS framework. Starting from camera and sensor setup, we will describe online training procedure, including touch patch integration and boundary-based sampling. The description is shown in Fig. 1. The framework starts from an initialized point cloud. By expanding each point into an ellipsoid, each point contains more information (e.g., opacity, rotation). The Gaussian primitives are rendered through rasterization.

TABLE I NOTATION TABLE
<table><tr><td>Notation Meaning</td><td></td><td>Notation Meaning</td><td></td><td>Notation Meaning</td><td></td><td>Notation Meaning</td><td></td></tr><tr><td> $\mu$ </td><td>Mean of a Gaussian</td><td>X</td><td>points on Gaussian</td><td>â</td><td>Covariance matrix</td><td> $\mathcal G (  { \mathbf { x } } )$ </td><td>Gaussian function</td></tr><tr><td>R</td><td>Rotation matrix</td><td>S</td><td>Scaling matrix</td><td> $\Sigma _ { p }$ </td><td>Projected covariance</td><td>J</td><td>Jacobian</td></tr><tr><td>W</td><td>Viewing transformation</td><td> $\alpha _ { i }$ </td><td>Opacity of Gaussian i</td><td> $\mathbf { c } _ { i }$ </td><td>Color contribution</td><td> $T _ { i }$ </td><td>Transmittance</td></tr><tr><td> $\sigma _ { i }$ </td><td>Density of Gaussian i</td><td> $\delta _ { i }$ </td><td>Distance interval</td><td>c</td><td>Final rendered color</td><td> $\mathbf { R } _ { c } ^ { w } , \mathbf { t } _ { c } ^ { w }$ </td><td>Rotation &amp; Translation</td></tr><tr><td> $\mathbf { T } _ { c } ^ { w }$ </td><td>Camera-to-world</td><td> $\mathbf { p }$ </td><td>Point in world axes</td><td> $\mathbf { p } _ { c }$ </td><td>Point in camera axes</td><td> $( u , v )$ </td><td>Pixel coordinates</td></tr><tr><td> $z$ </td><td>Depth</td><td> $f _ { x } , f _ { y }$ </td><td>Focal lengths</td><td> $c _ { x } , c _ { y }$ </td><td>Principal point</td><td> $\mathcal { L } _ { \mathrm { i m a g e } }$ </td><td>Visual loss</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { t o u c h } }$ </td><td>Touch loss</td><td> $\lambda$ </td><td>Loss weight</td><td> $N$ </td><td>Number of pixels</td><td> $\mathbf { v } _ { i j }$ </td><td>Directional Vector</td></tr><tr><td> $a , b , c$ </td><td>Ellipsoid principal axes</td><td> $d _ { i j }$ </td><td>Euclidean distance</td><td> $r _ { i } , r _ { j }$ </td><td>Directional radius</td><td> $\mathbf { M } _ { i }$ </td><td>Ellipsoid matrix</td></tr><tr><td> $\mathcal { P } _ { i }$ </td><td>Touch patch</td><td>Qi</td><td>k-nearest point set</td><td> $v _ { i }$ </td><td>i-th view angle</td><td> $\hat { I } _ { i }$ </td><td>Rendered image</td></tr><tr><td> $I _ { i }$ </td><td>Ground truth image</td><td>R</td><td>Rendering function</td><td> $\| \cdot \|$ </td><td>Euclidean distance</td><td></td><td></td></tr></table>

## A. Data acquisition

Required raw data includes depth images, RGB images and touch patches. The visual data is acquired during steps 1-3 in Fig. 1, and touch samples are retrieved during step 6.

a) Visual data acquisition: Point clouds are generated from depth images by projecting the depth measurements using camera intrinsics in step 1 in Fig. 1. Each point $x _ { c } , y _ { c } , z _ { c }$ is derived from the pixel column $( u , v , z ) ^ { T }$ . Depth images are transformed into 3D point clouds using intrinsic parameters $( f _ { x } , f _ { y } , c _ { x } , c _ { y } )$ by considering that:

$$
x _ { c } = { \frac { ( u - c _ { x } ) \cdot z } { f _ { x } } } , \quad y _ { c }  &  = { \frac { ( v - c _ { y } ) \cdot z } { f _ { y } } } , \quad z _ { c } = z\tag{4}
$$

where $f _ { x }$ and $f _ { y }$ are the focal lengths, and $c _ { x }$ and $c _ { y }$ represent the camera center coordinates. The point in camera space pc is then transformed to the world coordinate system using the known rigid body transformation matrix, $\mathbf { T } _ { c } ^ { w } = \left[ \begin{array} { c c } { \mathbf { R } _ { c } ^ { w } } & { \mathbf { \bar { t } } _ { c } ^ { w } } \\ { \mathbf { 0 } } & { 1 } \end{array} \right]$ and the pixel in world space is given by

$$
{ \left[ \begin{array} { l } { \mathbf { p } } \\ { 1 } \end{array} \right] } = \mathbf { T } _ { c } ^ { w } \left[ { \mathbf { p } } _ { c } \right] ,\tag{5}
$$

where p denotes the point in the world space. Similarly, the image data is transformed in step 2 in Fig. 1 through standard coordinate translation procedures as the point cloud.

b) Touch acquisition: During touch sample acquisition, we sample the Unified Robot Description Format (URDF) mesh directly: at each iteration, a virtual probe emulating a parallel-jaw (two-finger) gripper selects the two most sparse Gaussian centers. Let k be the number of nearest neighbors (here, k = 400). We then build a k-dimensional (k-d) tree to retrieve the k nearest mesh points to each center, forming a touch patch $\mathcal { P } _ { i } = \{ ( \mathbf { p } _ { i j } , \mathbf { n } _ { i j } ) ~ | ~ \mathbf { p } _ { i j } \in \mathcal { Q } _ { i } \}$ . This simulates dual end-effector contacts on the model, providing local position and normal information for geometric optimization.

This structure facilitates rapid querying of neighboring points within specified spatial constraints. For each sampling center k-nearest neighbors from the ground truth point cloud are retrieved, forming a touch patch $\mathcal { P } _ { i } = \{ ( \mathbf { p } _ { i j } , \mathbf { n } _ { i j } ) \mid \mathbf { p } _ { i j } \in$ $\mathcal { Q } _ { i } \}$ , where $\mathcal { Q } _ { i } \ : = \ : \left\{ \mathbf { p } _ { i 1 } , \mathbf { p } _ { i 2 } , \ldots , \mathbf { p } _ { i k } \right\}$ represents the set of k-nearest ground truth points to each sampling center.

## B. Training

Data flow consists of two modalities, visual and touch, which are interleaved and fed into the training of the Gaussian model.

1) Visual: The training pipeline initializes the scene and Gaussian model of the object (Fig. 1 step 2) based on the point cloud from the depth cameras (Fig. 1 step 1). Images are taken to further train the Gaussian model (Fig. 1 step 3). The optimization of the parameters of the Gaussians is based on the forward and backpropagation. Gaussians G are rendered in parallel on the GPU through rasterization at the corresponding view angle. The Gaussians are optimized by calculating the difference between the rendered image $\hat { I } _ { i }$ and ground truth $I _ { i } .$ The discrepancy between the rendered image $\hat { I } _ { i }$ and the ground truth image $I _ { i }$ is quantified using a composite visual loss function [8]:

$$
\mathcal { L } _ { \mathrm { i m a g e } } = ( 1 - \lambda ) \frac { 1 } { N } \sum _ { i } \Vert \hat { I } _ { i } - I _ { i } \Vert _ { 1 } + \lambda \left( 1 - \mathrm { S S I M } ( \hat { I } _ { i } , I _ { i } ) \right)\tag{6}
$$

The $\Vert \hat { I } _ { i } - I _ { i } \Vert$ measure is the absolute difference between the rendered and ground truth images. $\mathrm { S S I M } ( \hat { I } _ { i } , I _ { i } )$ denotes the Structural Similarity Index Measure (SSIM), assessing the perceived quality and structural similarity between the two images. Î» is a weighting factor that balances the contributions. Despite these advantages, 3DGS still suffers from low light and occlusion, motivating the touch cue we introduce next.

2) Touch: We have two different strategies (Fig. 1 step 5), Sparsity stage and Boundary stage:

â¢ Sparsity stage: Identify the two Gaussians with the largest nearest-neighbor gaps. For each, sample their k nearest mesh points and create new Gaussians there whose centers will no longer be adjusted.

â¢ Boundary stage: We intermittently build a temporary Poisson-surface proxy solely to reveal boundary holes, leveraging the meshâs geometric accuracy while retaining the Gaussiansâ efficiency and flexibility for subsequent refinement.

a) Sparsity-based sampling: Since robotic manipulation tasks often involve bimanual interactions or multiple contact points (e.g., two-fingered grasping). We therefore take two contact regions as an example in our experiments. Although our probe is virtual, limiting each iteration to two patches faithfully emulates common parallel-jaw or two-fingered grasps and keeps the added computation bounded. Our sparsity-based sampling method identifies the two most sparse regions in the current Gaussian model by computing nearest neighbor distances. For each Gaussian $G _ { i } ,$ , calculate the distance to its nearest neighbor $d _ { i }$ . Gaussians are sorted in descending order of sparsity, and the top k are selected as sampling centers.

b) Uncertainty-based sampling: After sparsity-based sampling, we build a temporary Poisson proxy solely to expose boundary holes. It is discarded right after identifying uncertain regions. The mesh edges pinpoint high-uncertainty holes or unreliable parts, steering the next touch patch to geometry that vision alone undersamples. Because this fit is invoked only intermittently, it adds negligible overhead while providing a more reliable uncertainty cue than point-wise density heuristics.

Mesh edges are identified by analyzing the connectivity and normal consistency of the mesh vertices. An edge represents the regions with discontinuity. These boundary points are organized using a k-d tree to accelerate inquiry. A greedy search will be applied to select the coverage centers that maximize the coverage of uncovered boundary points. By iteratively choosing the center, the algorithm ensures coverage of uncertainty regions. From the newly sampled touch points, new Gaussians are spawned based on the touch information and are locked. Conversely, the visually spawned Gaussians nearby are recognized as inaccurate Gaussians. They are pruned to maintain model fidelity.

At each optimization step, we deliberately sample two new surface patches, one per fingertip of a virtual parallel-jaw gripper to match the budget of a realistic two-finger grasp. Across successive iterations these patch pairs are reselected in the most uncertain regions, so the set of contacted areas gradually expands over the whole object.

c) Geometry loss: The geometric loss is defined using two basic principles: minimizing the distance between two neighboring Gaussians and minimizing their overlap. The touch loss ${ \mathcal { L } } _ { \mathrm { t o u c h } }$ quantifies the discrepancy between the actual distances between the centroids of two Gaussians and the expected distances based on their directional radius. Defining the vector $\mathbf { v } _ { i j } = \mu _ { i } - \mu _ { j }$ that points from Gaussians $\mathcal { G } _ { i }$ to $\mathcal { G } _ { j }$ for each pair $( \mathcal { G } _ { i } , \mathcal { G } _ { j } )$ , the directional radius $r _ { i }$ is the distance from the centroid to the surface of $\mathcal { G } _ { i }$ along $\mathbf { v } _ { i j }$

Let Euclidean distance $d _ { i j } = \| \mathbf { v } _ { i j } \|$ . The touch loss for each pair $( \mathcal { G } _ { i } , \mathcal { G } _ { j } ) , \delta _ { i j }$ , is defined:

$$
\delta _ { i j } = d _ { i j } - ( r _ { i } + r _ { j } ) .\tag{7}
$$

Consider an axis-aligned ellipsoid centered at the origin with semi-axes lengths $a , b , c .$ . Its implicit equation can be expressed using the inverse covariance matrix:

$$
\begin{array} { r } { \mathbf { x } ^ { T } \mathbf { \Sigma } ^ { - 1 } \mathbf { x } = 1 , } \end{array}\tag{8}
$$

where $\Sigma ^ { - 1 } ~ = ~ \mathrm { d i a g } ( a ^ { - 2 } , b ^ { - 2 } , c ^ { - 2 } )$ is a diagonal matrix, encodes the ellipsoidâs shape.

For a rotation matrix R â SO(3) (satisfying $\mathbf { R } ^ { - 1 } = \mathbf { R } ^ { T } )$ the transformed coordinates $\mathbf { x } ^ { \prime } = \mathbf { R } \mathbf { x }$ yield:

$$
( \mathbf { R } ^ { T } \mathbf { x } ^ { \prime } ) ^ { T } \Sigma ^ { - 1 } ( \mathbf { R } ^ { T } \mathbf { x } ^ { \prime } ) = 1 \Rightarrow \mathbf { x } ^ { \prime T } ( \mathbf { R } \Sigma ^ { - 1 } \mathbf { R } ^ { T } ) \mathbf { x } ^ { \prime } = 1 .\tag{9}
$$

For a Gaussian $\mathcal { G } _ { i }$ centered at $\mu _ { i }$ with rotation $\mathbf { R } _ { i } ,$ , its worldspace representation becomes:

$$
\begin{array} { r } { ( { \bf x } - { \mu _ { i } } ) ^ { T } \underbrace { { { \bf R } _ { i } } { \Sigma _ { i } ^ { - 1 } } { { \bf R } _ { i } ^ { T } } } _ { { \bf M } _ { i } } ( { \bf x } - { \mu _ { i } } ) = 1 , } \end{array}\tag{10}
$$

where $\mathbf { M } _ { i }$ is precomputed to accelerate.

The directional radius can be calculated directly using the transformed covariance:

$$
r _ { i } = \frac { 1 } { \sqrt { \mathbf { v } _ { i j } ^ { T } \mathbf { M } _ { i } \mathbf { v } _ { i j } } } = \frac { 1 } { \sqrt { \mathbf { v } _ { i j } ^ { T } ( \mathbf { R } _ { i } \boldsymbol { \Sigma } _ { i } ^ { - 1 } \mathbf { R } _ { i } ^ { T } ) \mathbf { v } _ { i j } } } .\tag{11}
$$

## IV. PERFORMANCE EVALUATION

## A. Experimental setup

The experimental setup was designed to evaluate the effectiveness of touch-aided Gaussian splatting for improving 3D reconstruction in environments with poor visibility. A set of virtual cameras (both depth and RGB) was positioned around the target object to ensure spatial coverage. In particular, four depth cameras were placed at different viewpoints around the object. Nine RGB cameras were placed in overhead and corner positions, providing supplementary visual information. To simulate harsh visual scenarios, the objects were placed in a controlled environment that allowed for variable lighting conditions and introducing occlusions. The experiments include three objects (i.e., Fire Hydrant, Cube, and Can) under the three harsh visual conditions: (1) under deteriorated lighting (i.e., using optimal lighting for point cloud initialization with nine light sources evenly distributed above the object and camera, and employing intentionally poor lighting for 3D reconstruction training with only one light source positioned above the object and no side illumination); (2) with the coverage points missing because some camera views are unavailable; (3) occlusion with three blocks. We deliberately pick three mutually-distinct shapes to span the typical curvature spectrum. Because our algorithm is shape-agnostic, running it on additional models yielded the same relative improvements. We therefore report these representative results.

We conducted experiments using Gazebo [18]. Object models were imported, each described in COLLADA format. To obtain high-fidelity ground-truth data, we sampled the COL-LADA files and exported dense point clouds that represent the objectâs geometry. This ground-truth sampling serves as a reference to measure the accuracy of the reconstructed models. To facilitate data acquisition and system integration, each virtual camera in Gazebo was connected to a Robot Operating System (ROS) framework. Specifically, each camera publishes images (or depth images) that are stored as ground truth for training the 3DGS model.

<!-- image-->

Fig. 2. Reconstruction quality over training iterations in three separate challenging scenarios. Rows: (1) Fire-Hydrant under deteriorated lighting; (2) Cube with missing camera viewpoints; (3) Can under severe occlusion. Columns: Chamfer Distance (CD, mm), F-score (%), and JensenâShannon Divergence (JSD). (Solid blue is the visual-only baseline; dashed green is the proposed touch-augmented method; Red arrows is the relative improvement.)  
<!-- image-->  
Fig. 3. Qualitative reconstructions at iteration 1400. For each object (top to bottom: Fire-Hydrant, Cube, Can) we show, from left to right: (i) groundtruth mesh, (ii) touch-augmented result, and (iii) visual-only baseline. Small gaps visible in the second column arise because our method actively prunes Gaussians whose geometry is contradicted by the latest contacts. Those regions are re-filled once they are revisited by later touch samples.

## B. Experimental Results and Evaluation

We evaluated the reconstruction quality using three complementary metrics. Chamfer Distance (CD) quantifies the geometric discrepancy between the reconstructed point cloud and the ground truth. Lower CD values indicate a more accurate reconstruction. F-score provides a balance between precision and recall for 3D reconstructions. Higher F-scores suggest a more complete and accurate surface recovery. Jensen-Shannon Divergence (JSD) measures the dissimilarity between the reconstructed and ground-truth shape distributions. In this paper, JSD is computed with a base-2 logarithm, ranging from [0, 1]. Lower JSD values indicate that the reconstructed geometry is better aligned with the real object.

Table II compares the reconstruction performance results of the visual-only and touch-augmented methods for three objects (Fire Hydrant, Cube and Can) under three test conditions measured at iteration 750 of training. The results show that integrating contact points and normals improve geometric accuracy across all test scenarios. The extra computation stems mainly from (i) twice KD-tree range queries at O(k log N ) (k = 400, N â¤ 5 Ã 105) and (ii) the optimization of the newly spawned Gaussians (â¤ 1 %). Empirically, this method introduces less than 20% overhead compared to the visualonly runtime on a modern GPU, so no noticeable slowdown was observed during training.

Fig. 2 shows the reconstruction performance improvement with more iterations. This figure shows the experimental results of (1) Fire Hydrant under deteriorated lighting (i.e., with good light condition for initializing point cloud but poor light condition during training for 3D reconstruction); (2) Cube with the coverage points missing due to missing some of the camera views; (3) Can with heavy occlusion.

Quantitative results in Fig. 2 highlight these advancements through three key metrics: Chamfer Distance (CD), F-score, and Jensen-Shannon Divergence (JSD). Taking an example of iteration at 1400, for the Fire Hydrant case under deteriorated lighting, the touch-augmented method improves the CD by 89% (from 7 mm to 0.7 mm), the F-score by 21% (from 76% to 97%), and JSD by 88% (from 0.7 to below 0.1). Similarly, in the experiment of the Cube with missing views at iteration 1400, the CD improves by 73% (from 13 mm to 3.5 mm), accompanied by a 16% increase in F-score (from 81% to 97%), and JSD by 58% (from 0.5 to 0.21). The improvement of reconstruction performance is most significant in the scenario under severe occlusion. From Fig. 2, it shows that at iteration 1400, the visual-only method reaches a CD around 30 mm and an F-score below 60%, integrating touch modality drastically reduces the CD from 30 mm to 2 mm (15Ã) and boosts the F-score above 95%, and JSD by 83% (from 0.9 to lower than 0.2).

TABLE II  
PERFORMANCE METRICS AT ITERATION 750 (CD IN MM, F-SCORE IN %, JSD)
<table><tr><td rowspan="2">Case</td><td colspan="2">Deteriorated Light</td><td colspan="3">Missing View Angle</td><td colspan="2">Occlusion</td></tr><tr><td>CD</td><td>F-score JSD</td><td>CD</td><td>F-score</td><td>JSD</td><td>CD F-score</td><td>JSD</td></tr><tr><td>Fire Hydrant (Visual Only)</td><td>7.6 mm</td><td>77.02% 0.7151</td><td>6.7 mm</td><td>83.59%</td><td>0.7135</td><td>5.7 mm 89.23%</td><td>0.5557</td></tr><tr><td>Fire Hydrant (With Touch)</td><td>1.68 mm</td><td>96.92% 0.1461</td><td>1.82 mm</td><td>96.10%</td><td>0.1520 1.68 mm</td><td>97.33%</td><td>0.1452</td></tr><tr><td>Cube (Visual Only)</td><td>10.2 mm</td><td>92.93% 0.5955</td><td>13.0 mm</td><td>81.98%</td><td>0.5094</td><td>18.0 mm 73.21%</td><td>0.6021</td></tr><tr><td>Cube (With Touch)</td><td>4.71 mm</td><td>95.82% 0.2210</td><td>5.02 mm</td><td>95.96%</td><td>0.2595</td><td>6.24 mm 88.97%</td><td>0.2575</td></tr><tr><td>Can (Visual Only)</td><td>12.1 mm</td><td>25.65% 0.9173</td><td>13.8 mm</td><td>20.64%</td><td>0.9630</td><td>31.21 mm 53.57%</td><td>0.9393</td></tr><tr><td>Can (With Touch)</td><td>4.35 mm</td><td>78.36% 0.4496</td><td>4.73 mm</td><td>72.58%</td><td>0.4826</td><td>5.43 mm 91.46%</td><td>0.3220</td></tr></table>

These improvements can also be reflected by qualitative analysis. As illustrated in Fig. 3, the touch-augmented method shows enhanced geometric details in visually ambiguous regions. For instance, the bolts on the fire hydrant and ridges of the can are lost in vision-only reconstructions; In contrast, they are accurately captured despite adverse conditions. Touch information directly compensates for missing view information or occluded areas, enabling the framework to resolve ambiguities where visual data is sparse or unreliable. Both quantitative and qualitative analysis illustrate the feasibility of the touch-augmented approach in refining 3D reconstructions in scenarios where traditional vision-based methods fail.

## V. CONCLUSION

We introduced an online touch-augmented 3DGS that preserves geometry when vision fails. By integrating touch modality into the 3DGS training process, our framework effectively completes visual data, ensuring more robust geometric modeling even under challenging conditions such as poor lighting, limited camera viewpoints, and severe occlusions. The proposed method employs a two-stage touch sampling strategy, ensuring targeted and efficient refinement of the 3D model. Our study highlights the potential of multimodal data fusion in advancing 3D geometric modeling, particularly in environments where traditional vision-based techniques are insufficient. This touch-augmented 3DGS framework offers a robust solution for 3D perception in robotics, teleoperation, and other settings demanding high-fidelity geometry under adverse visual conditions.

## ACKNOWLEDGMENTS

This research was supported by the TOAST project, funded by the European Unionâs Horizon Europe research and innovation program under the Marie SkÅodowska-Curie Actions Doctoral Network (Grant Agreement No. 101073465), the Danish Council for Independent Research project eTouch (Grant No. 1127-00339B), and NordForsk Nordic University Cooperation on Edge Intelligence (Grant No. 168043).

## REFERENCES

[1] A. Billard and D. Kragic, âTrends and challenges in robot manipulation,â Science, vol. 364, no. 6446, p. eaat8414, 2019.

[2] A. D. H. Thomas, M. G. Rodd, J. D. Holt and C.J. Neill, âReal-time industrial visual inspection: A review,â Real-Time Imaging, vol. 1, no. 2, pp. 139â158, 1995.

[3] O. Holland et al., âThe IEEE 1918.1 âTactile Internetâ Standards Working Group and its Standards,â Proc. IEEE Inst. Electr. Electron. Eng., vol. 107, no. 2, pp. 256â279, 2019.

[4] K. Antonakoglou, X. Xu, E. Steinbach, T. Mahmoodi, and M. Dohler, âToward haptic communications over the 5G tactile internet,â IEEE Commun. Surv. Tutor., vol. 20, no. 4, pp. 3034â3059, 2018.

[5] P. Mitra and G. Niemeyer, âModel-mediated telemanipulation,â Int. J. Rob. Res., vol. 27, no. 2, pp. 253â262, 2008.

[6] X. Xu, B. Cizmeci, C. Schuwerk, and E. Steinbach, âModel-mediated teleoperation: Toward stable and transparent teleoperation systems,â IEEE Access, vol. 4, pp. 425â449, 2016.

[7] X. Xu, J. Kammerl, R. Chaudhari, and E. Steinbach, âHybrid signalbased and geometry-based prediction for haptic data reduction,â in Proc. 2011 IEEE Int. Workshop on Haptic Audio Visual Environ. and Games, 2011, pp. 68â73.

[8] X. Xu, B. Cizmeci, A. Al-Nuaimi, and E. Steinbach, âPoint cloud-based model-mediated teleoperation with dynamic and perception-based model updating,â IEEE Trans. Instrum. Meas., vol. 63, no. 11, pp. 2558â2569, 2014.

[9] M. Antonsen, S. Liu, X. Xu, E. Steinbach, F. Chinello, and Q. Zhang, âDigital twin-empowered model-mediated teleoperation using multimodality data with signed distance fields,â in Proc. IEEE Haptics Symp. (HAPTICS), 2024, pp. 353â359.

[10] B. Kerbl, G. Keramidas, T. Leimkuhler, and G. Drettakis, â3D Gaus- Â¨ sian splatting for real-time radiance field rendering,â arXiv preprint arXiv:2303.13495, 2023.

[11] M. Dou et al., âMotion2fusion: Real-time volumetric performance capture,â ACM Trans. Graph., vol. 36, no. 6, pp. 1â16, 2017.

[12] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNeRF: Representing scenes as neural radiance fields for view synthesis,â Commun. ACM, vol. 65, no. 1, pp. 99â106, 2022.

[13] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM Trans. Graph., vol. 41, no. 4, pp. 1â15, 2022.

[14] E. J. Smith et al., â3D shape reconstruction from vision and touch,â Neural Inf. Process. Syst., vol. 33, pp. 14193â14206, 2020.

[15] Z. Yi, R. Calandra, F. Veiga, H. van Hoof, T. Hermans and Y. Zhang, âActive tactile object exploration with Gaussian processes,â in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2016, pp. 4925â4930.

[16] S. Ottenhaus, M. Miller, D. Schiebener, N. Vahrenkamp, and T. Asfour, âLocal implicit surface estimation for haptic exploration,â in Proc. IEEE-RAS Int. Conf. Humanoid Robots (Humanoids), 2016, pp. 850â856.

[17] L. Rustler, J. Matas, and M. Hoffmann, âEfficient visuo-haptic object shape completion for robot manipulation,â in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2023, pp. 3121â3128.

[18] N. Koenig and A. Howard, âDesign and use paradigms for Gazebo, an open-source multi-robot simulator,â in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2004, pp. 2149â2154.

[19] Y. Zhang, J. Zhang, and L. Li, âFastMESH: Fast surface reconstruction by hexagonal mesh-based neural rendering,â arXiv preprint arXiv:2305.14295, 2023.