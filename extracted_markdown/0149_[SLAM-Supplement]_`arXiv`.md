# Globally Consistent RGB-D SLAM with 2D Gaussian Splatting

Xingguang Zhong1, Yue Pan1, Liren Jin1, Marija PopovicÂ´2, Jens Behley1, and Cyrill Stachniss1,3

<!-- image-->  
Fig. 1: Reconstruction results of 2DGS-SLAM on synthetic dataset Replica [72] and real-world dataset ScanNet [10]. We present the reconstructed 2D Gaussian splatting maps, along with RGB and normal renderings from zoomed-in local views. These results demonstrate that our method achieves both high-fidelity image rendering and precise geometric reconstruction.

AbstractâRecently, 3D Gaussian splatting-based RGB-D SLAM displays remarkable performance of high-fidelity 3D reconstruction. However, the lack of depth rendering consistency and efficient loop closure limits the quality of its geometric reconstructions and its ability to perform globally consistent mapping online. In this paper, we present 2DGS-SLAM, an RGB-D SLAM system using 2D Gaussian splatting as the map representation. By leveraging the depth-consistent rendering property of the 2D variant, we propose an accurate camera pose optimization method and achieve geometrically accurate 3D reconstruction. In addition, we implement efficient loop detection and camera relocalization by leveraging MASt3R, a 3D foundation model, and achieve efficient map updates by maintaining a local active map. Experiments show that our 2DGS-SLAM approach achieves superior tracking accuracy, higher surface reconstruction quality, and more consistent global map reconstruction compared to existing renderingbased SLAM methods, while maintaining high-fidelity image rendering and improved computational efficiency.

Index TermsâSLAM, mapping, localization, RGB-D perception

## I. INTRODUCTION

IMULTANEOUS localization and mapping (SLAM) is a fundamental problem in computer vision and robotics.   
The ability to reconstruct unknown environments is a basis   
for various robotic tasks, including navigation [30], [48],   
[88] and exploration [4], [36]. Recently, radiance field-based   
map representations like neural radiance field (NeRF) [52]   
and Gaussian splatting (GS) [40], have opened up new   
possibilities for dense RGB-D SLAM by enabling high  
fidelity reconstruction with photorealistic rendering. Among   
them, Gaussian splatting has gained popularity due to its fast

rendering speed and flexible scalability, establishing itself as the more favorable map representation for radiance fieldbased RGB-D SLAM.

Most existing GS-based methods [38], [50], [96], [102] directly adopt classical 3D Gaussian splatting (3DGS) for mapping and frame-to-map camera tracking. However, the depth images rendered from 3DGS at different viewpoints often exhibit inconsistency, negatively impacting pose optimization with depth information and geometric reconstruction accuracy. Furthermore, since pose drift in long-term camera tracking is inevitable, SLAM systems need to incorporate loop closures as well as map correction and update mechanisms for global consistency [71]. Some radiance field-based RGB-D SLAM methods [46], [102] address this issue by using multiple submaps and applying global transformations to the submaps after loop closure. However, they often rely on computationally expensive point cloud registration for relocalization and typically require complex post-processing to merge all submaps, making them impractical for online robotic applications.

In this paper, we investigate the problem of realizing a RGB-D SLAM system that builds geometrically accurate and globally consistent radiance field reconstructions online. Instead of using 3DGS, we adopt 2D Gaussian splatting (2DGS) [29] as our map representation. 2DGS replaces 3D ellipsoids with 2D disks and explicitly computes ray-disk intersections, ensuring consistent depth rendering while maintaining high-fidelity radiance field reconstruction required for novel view synthesis. Leveraging these properties, we develop an accurate rendering-based method for frame-to-map camera pose estimation. In addition, 2DGS represents the environment with discrete Gaussian splats distributed in 3D space, offering a point cloud-like structure allowing for elastic properties when closing loops. By associating each Gaussian splat with nearby keyframes, we can update the poses of keyframes and their corresponding splats after pose graph optimization. Building on this strategy, we further address two key challenges to achieve globally consistent map reconstruction in an online manner. First, inspired by classical surfel-based dense SLAM methods [5], [90], we maintain a continuously updated local active map and design a mechanism to transition Gaussian primitives between active and inactive states. In this way, we prevent tracking and relocalization failures due to the accumulation of new and old map structures, while removing the need of complicated submap management. Second, after detecting a loop closure, we need to accurately estimate the relative pose between existing frame and the current frame to add proper constraints to the pose graph. Unlike prior works that rely on computationally expensive 3D point cloud registration, we leverage MASt3R [44], a recently introduced 3D foundation model with remarkable generalization capability, to estimate an initial relative pose. This initial estimate is then refined through further frame-to-map tracking within the active map, achieving accurate relocalization.

The main contribution of this paper is a 2DGS-based RGB-D SLAM system, termed 2DGS-SLAM. Our 2DGS-SLAM addresses two key limitations of existing 3DGSbased SLAM systems. First, to overcome the limited tracking and reconstruction accuracy caused by inconsistent depth rendering in existing 3DGS-based SLAM approaches, we derive a camera pose estimation method specifically adapted to the 2DGS rendering process and implement it efficiently in CUDA. Leveraging the inherent depth rendering consistency of 2DGS, we construct an accurate and robust tracking algorithm and achieve precise surface reconstruction at the same time. Second, to address the lack of robust loop closure in current systems, we introduce an efficient Gaussian splat management strategy and integrate MASt3R to realize reliable loop closure. This allows for the online reconstruction of globally consistent radiance fields. As shown in Fig. 1, our 2DGS-SLAM achieves outstanding reconstruction results in synthetic dataset and real-world scenes.

In summary, we make three key claims: (i) Our proposed 2DGS-SLAM achieves superior tracking accuracy compared to state-of-the-art rendering-based approaches; (ii) Our approach surpasses or is on-par with 3DGS-based methods in surface reconstruction quality and demonstrates more consistent mapping results in real-world scenes compared to other loop-closure-enabled methods. At the same time, 2DGS-SLAM maintains high-fidelity image rendering performance that is either superior to or on-par with baseline approaches. (iii) Compared to other radiance field-based methods that support loop closure, our approach is more efficient on runtime and has a more compact map representation. These claims are backed up by our experimental evaluation. The open-source implementation of our 2DGS-SLAM is available at: https://github.com/PRBonn/2DGS-SLAM.

## II. RELATED WORK

## A. Map-centric RGB-D SLAM

Compared to sparse feature-based visual SLAM systems [7], [20], [54], that target pose and feature location estimation, dense visual SLAM systems generate 3D maps beneficial for robotic tasks like interaction and navigation. RGB-D SLAM predominates indoor dense SLAM systems, as the depth camera enables direct acquisition of metricallyscaled dense geometry. Dense visual SLAM systems can be further classified into frame-centric and map-centric approaches based on their tracking strategies. Frame-centric methods estimate poses through either sparse feature matching [19], [43] or by minimizing photometric and geometric errors between consecutive frames [13], [41], [42], [78]. In these methods, the map is merely a by-product constructed by accumulating frame-wise point clouds. In contrast, similar to LiDAR-based SLAM systems [6], [25], [61], [83], mapcentric methods incrementally build a 3D model of the environment and perform frame-to-map tracking for robust pose estimation.

In the past decade, numerous works employ truncated signed distance function (TSDF) [9], [11], [56], [58], [82], [89], Octomap [19], [27], or surfels [39], [69], [73], [90] as map representations and use weighted moving average for efficient incremental mapping. Despite their effective mapping and localization capabilities, these methods suffer from limited scalability and map fidelity, constrained by their discrete map representations.

Recent advancements in radiance fields and implicit neural representations [3], [62], [100] have enabled high-fidelity scene modeling, offering new opportunities for map-centric SLAM. With the radiance field as the map, camera tracking can be performed by minimizating of photometric and geometric discrepancies between the current frame and the rendered image from the radiance field. iMap [75] pioneered the use of neural radiance fields (NeRF) [52] as a map representation, demonstrating the advantages of neural implicit representations in handling the sparse observations or occlusions through inpainting. However, despite being memoryefficient, the use of a single multi layer perceptron (MLP) to represent the whole scene limits its ability to capture fine-grained details in complex, large-scale environments. To improve scalability and rendering performance, subsequent works propose hybrid map representations that combine locally-defined optimizable features with a globally-shared shallow MLP. These features can be structured in various forms such as hierarchical voxel grids [103], octrees [93], spatial hashing [85], tri-plane grids [15], [37], or unordered points [46], [68], [97]. Nevertheless, the rendering process remains computationally intensive due to ray-wise sampling and volumetric integration.

3D Gaussian splatting [40] introduces a novel radiance field based on rasterization of optimizable Gaussian primitives, offering superior training and rendering efficiency while maintaining or exceeding the rendering quality of NeRF. These properties have facilitated various robotic applications, such as active sensing [36], scene-level mapping [35], and simulation [101], thereby encouraging the adoption of 3DGS as the map representation for SLAM.

3DGS-based visual SLAM systems can be split into coupled and decoupled ones, based on whether the online-built 3DGS map is utilized for rendering-based tracking. Decoupled systems [26], [31], [63], [67], [91] employ external trackers [7], [54], [70], [78] for camera pose estimation. However, these systems require maintaining a separate map for the external tracker, which is distinct from the 3DGS map, resulting in architectural redundancy in the system design. In contrast, coupled systems [22], [38], [50], [76], [92], [96] utilize 3DGS as the sole map representation for both tracking and mapping through rendering-based gradient descent optimization. These systems typically employ a keyframe-based strategy, where mapping is performed using keyframes, while tracking is applied to all frames.

Although achieving comparable tracking performance and superior map photorealism to previous map-centric SLAM systems, the aforementioned coupled 3DGS SLAM systems face two main challenges. First, geometric ambiguity in 3D Gaussian splatting limits the accuracy of geometry-based tracking and surface reconstruction. Second, these systems function primarily as visual odometries, lacking the capability to handle loop closures necessary to create a globally consistent map.

To address the first challenge, one solution is to flatten the 3D Gaussian ellipsoids into optimizable 2D surfels, as demonstrated in 2DGS [12], [29]. 2DGS provides enhanced geometric representation with multi-view consistent depth and normal rendering, motivating its use over 3DGS as the map representation to improve geometry-based tracking accuracy and surface reconstruction quality. While several concurrent works [32], [60], [91] adopt 2DGS as their map representation, none have implemented on-manifold camera pose optimization using the 2DGS rasterizer, as MonoGS does for 3DGS. Our work addresses this gap by explicitly deriving Jacobians for 2DGS-based camera tracking and implementing them in an efficient CUDA-based rasterizer. In the next section, we discuss related works addressing the second challenge of globally consistent mapping.

## B. Visual Loop Closure and Globally Consistent Mapping

For visual SLAM, closuring loop is crucial for correcting accumulated odometry drift and ensuring a globally consistent map. Loop closure correction typically involves a place recognition step to identify loop closure candidates, followed by a relocalization step to estimate the relative pose between the current frame and the loop candidate. This relative pose is subsequently used in graph optimization to correct drift errors of trajectory and deform the map.

Compared to distance-based loop candidate search [42], [77], appearance-based place recognition is more versatile, as it can operate without prior knowledge of the camera position and remains effective even when odometry drift is significant. Early approaches primarily rely on aggregating handcrafted local features using bag-of-words [21], [24], random ferns [23], hamming distance embedding binary search tree [16], or VLAD [2] to build databases for efficient searching and matching [7], [43], [54], [90], or match image sequences [53], [84]. Recently, there has been a shift towards learning-based approaches using NetVLAD [1] and DINOv2 [33], [34], [57]

The relocalization step aims to estimate the relative pose between the current frame and the detected historical frame. This transformation serves as a loop constraint edge for pose graph optimization in graph-based SLAM systems. In cases where odometry drift is small, relocalization becomes a local pose tracking problem, i.e., tracking the current frame against the historical map. However, for larger loops, where the initial pose often lies outside the convergence basin of pose tracking, a coarse global localization step becomes necessary. This is typically achieved using the PnP or Umeyama [80] algorithm together with RANSAC, which relies on keypointbased feature matching [65], [66].

Recent data-driven 3D foundation models, particularly DUSt3R [86] and MASt3R [44], have demonstrated promising performance in various 3D vision tasks. Given a pair of RGB images, MASt3R generates a metrically-scaled 3D point map for both images in the first cameraâs coordinate frame, along with confidence maps. From this point map, additional properties including relative camera poses, depth images, and pixel correspondences can be derived. Furthermore, features extracted by the MASt3R encoder can be aggregated using the ASMK framework [18] for efficient image retrieval. Several concurrent SLAM systems leverage MASt3R for different purposes: camera pose and Gaussian splats initialization for 3DGS SLAM [94], loop closure detection and camera pose tracking [55], and twoview loop constraint construction [45]. In our approach, we employ MASt3R exclusively for loop closure correction. Unlike previous approaches that use separate features for loop closure detection and relocalization [46], [95], we utilize MASt3R for both tasks.

Although loop closure correction is a common practice in traditional SLAM systems, it has been adopted by only a few radiance field-based SLAM systems, as it is challenging to maintain a globally consistent radiance field throughout the SLAM process. Among coupled systems supporting loop closure, most existing approaches utilize a collection of submaps, treating each submap as a rigid body for pose adjustment. Within each submap, the radiance field can be represented using MLP-based [77], neural octree-based [49], or neural point-based [28], [46] implicit fields, as well as the 3DGS radiance field [95], [102]. While the submap-based strategy is efficient for pose graph optimization and map management [11], [59], [64], it presents challenges such as drift within the submap and additional effort required for merging submaps and refining the merged map, particularly for the radiance field [46], [95], [102]. Redundant memory usage occurs in overlapping submap areas, and discrepancies among the submaps are often unfavorable features of the submap-based strategy.

For map representations that are inherently elastic, such as surfels, neural points, and Gaussian splats, one can take a point-based deformation strategy [60], [61], [90] which associates each map primitive with a frame and adjusts frames instead of submaps during pose graph optimization.

As the first coupled 3DGS SLAM system with loop closure, LoopSplat [102] employs NetVLAD [1] for loop closure detection and estimates loop constraints through rendering-based keyframe-to-submap tracking. While it achieves superior performance in pose accuracy and map global consistency in larger indoor scenes, the use of 3DGS submaps necessitates a computationally intensive map refinement step after submap merging. Moreover, without a coarse global localization step, LoopSplat may struggle with relocalization when closing a large loop, where the loop candidate is distant from the current frame. In contrast, our approach leverages MASt3R for both loop closure detection and coarse relocalization, while adopting a submap-free strategy that associates Gaussian surfels with keyframes. This design enables direct map correction during camera pose adjustments, avoiding redundant memory usage and submap merging overhead while achieving superior geometric accuracy in a globally consistent 2DGS map.

<!-- image-->  
Fig. 2: Overview of 2D Gaussian splatting. The pose and shape of a splat g in the world space are defined by its center Âµ and two scaled tangent vectors $s _ { u } t _ { u } , s _ { v } t _ { v }$ . Given a camera with pose ${ \mathbf { } } T _ { c w } ,$ the splat g can be projected onto the image space. Points within the local space of the splat is mapped to their corresponding pixel on the imageâs x-y plane via a homography T .

## III. OUR APPROACH

Our proposed RGB-D SLAM system aims to reconstruct a globally consistent radiance field online while maintaining the precise geometric structure of the 3D environment. In the following sections, we first introduce the primary map representation used in our system, 2D Gaussian splatting [29], and derive how to backpropagate gradients to the camera pose with 2DGS-based differentiable rendering. Next, we describe the structure of our system and provide a detailed explanation of each module in the individual subsections.

## A. 2D Gaussian Splatting

Unlike 3DGS, 2DGS compresses one dimension of the 3D ellipsoid to zero, using 2D Gaussian disks as primitives to represent the 3D environment. By explicitly calculating the intersection of the rays from the camera with the diskâs plane, 2DGS can realize multi-view consistency in depth rendering, thereby achieving a more accurate geometric representation.

As illustrated in Fig. 2, a 2D Gaussian splat g is defined within a local tangent plane in a 3D global coordinate system. This plane is determined by the splatâs central point $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ and two principal tangential vectors $\mathbf { \Delta } _ { t _ { u } }$ and $\mathbf { \Delta } _ { t _ { v } }$ , with two scale factors $s _ { u }$ and $s _ { v }$ controlling the variances along the tangential vectors, respectively. By representing the rotation matrix of the 2D Gaussian splat as $\pmb { R } = [ \pmb { t } _ { u } , \pmb { t } _ { v } , \pmb { t } _ { n } ] \in \mathbb { R } ^ { 3 \times 3 }$ where $\mathbf { } t _ { n } = \mathbf { } t _ { u } \times \mathbf { } t _ { \iota }$ is the normal vector, and arranging scale factors as a $3 \times 3$ diagonal matrix $\boldsymbol { S } = \mathrm { d i a g } ( s _ { u } , s _ { v } , 0 )$ , the 2D local frame can be parameterized as follows:

$$
P ( u , v ) = \pmb { \mu } + s _ { u } \pmb { t } _ { u } u + s _ { v } \pmb { t } _ { v } v = \pmb { H } \left( u , v , 0 , 1 \right) ^ { \top } ,\tag{1}
$$

$$
\mathrm { w h e r e } \ H = \left[ \begin{array} { c c c c } { { s _ { u } } } & { { s _ { v } } } & { { 0 } } & { { \mu } } \\ { { 0 } } & { { 0 } } & { { 0 } } & { { 1 } } \end{array} \right] = \left[ \begin{array} { c c } { { R S } } & { { \mu } } \\ { { 0 } } & { { 1 } } \end{array} \right] .\tag{2}
$$

Here, H is the homogeneous transformation matrix from 2D local uv space to the global coordinate system.

The mapping from the uv space to the rendering imageâs screen space can be formulated as a 2D-to-2D homography transformation [51]. Let $W \in \mathbb { R } ^ { 4 \times 4 }$ be the transformation matrix from camera space to image space and $\pmb { T } _ { c w } \in S E ( 3 )$ be the pose of the view camera, combining Eq. (1) yields:

$$
\pmb { x } = ( x z , y z , z , 1 ) ^ { \top } = W \pmb { T } _ { c w } \pmb { P } ( u , v )\tag{3}
$$

$$
= W T _ { c w } H \left( u , v , 0 , 1 \right) ^ { \top } ,\tag{4}
$$

where $\mathbf { \delta } \mathbf { T } _ { c w }$ transforms the splat in the world space to the camera space and then W transforms it to the image space, and the x represents the ray corresponding to pixel (x, y) intersecting the 2D Gaussian splat at depth of z. For convenience, we define:

$$
{ \pmb { H } } _ { c } = { \pmb { T } } _ { c w } { \pmb { H } }\tag{5}
$$

$$
\mathbf { \Sigma } = \mathbf { T } _ { c w } \left[ \begin{array} { c c } { \mathbf { \boldsymbol { R S } } } & { \boldsymbol { \mu } } \\ { 0 } & { 1 } \end{array} \right] = \left[ \begin{array} { c c c c } { s _ { u } \mathbf { \boldsymbol { t } } _ { u c } } & { s _ { v } \mathbf { \boldsymbol { t } } _ { v c } } & { 0 } & { \pmb { \mu } _ { c } } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} \right] ,\tag{6}
$$

where $\mathbf { \delta t } _ { u c } , \mathbf { \delta t } _ { v c }$ and $\pmb { \mu } _ { c }$ are Gaussian splatâs tangential vectors and central point in camera space. Furthermore, we define the whole homography $\tau$ as:

$$
\mathcal { T } = W T _ { c w } H = W H _ { c } .\tag{7}
$$

To render the value of pixel $\pmb { p } = ( x , y ) ^ { \top }$ from the splats, 2DGS solves the inverse problem of Eq. (4), computing the intersection of ray x with the 2D Gaussian splat in uv space, while avoiding the need to compute the inverse of T . For further details, we refer the reader to the original paper [29].

Apart from these geometric parameters mentioned above, each splat also contains color feature c and opacity Î± to represent its visual appearance. After computing the raysplat intersections of all splats within field of view, 2DGS sorts them by depth and uses volumetric alpha blending to integrate weighted appearance values $V _ { p }$ of pixel $^ { p , }$ as follows:

$$
V _ { p } = \sum _ { i = 0 } ^ { N } { v _ { i } \alpha _ { i } \mathcal { G } ( { \boldsymbol { \mathbf { u } } _ { i } ^ { p } } ) \prod _ { j = 0 } ^ { i - 1 } \left( 1 - \alpha _ { j } \mathcal { G } ( { \boldsymbol { \mathbf { u } } _ { j } ^ { p } } ) \right) } ,\tag{8}
$$

where $\begin{array} { r } { \mathcal { G } ( u ) ~ = ~ \mathcal { G } ( u , v ) ~ = ~ \exp { \left( - \frac { u ^ { 2 } + v ^ { 2 } } { 2 } \right) } ~ } \end{array}$ represent the Gaussian weight of the intersection u in the uv space, $\mathbf { \Delta } u _ { i } ^ { p }$ means the i-th intersection along the ray of pixel $^ { p , }$ and N denotes the number of Gaussian splats that intersect with the ray. It should be noted that the appearance value v can be a view-dependent color generated from color feature $^ { c , }$ depth d and normal vector $\mathbf { \mathbf { \mathit { t } } } _ { n c } = \mathbf { \mathbf { \mathit { t } } } _ { u c } \times \mathbf { \mathbf { \mathit { t } } } _ { v c }$ , meaning that the color image, depth image and the normal image are rendered in the same way. In addition, we can also render an opacity image O if we set $v = 1$

To summarize, each 2D Gaussian splat g contains parameters $( \mu , R , t _ { u } , t _ { v } , s _ { u } , s _ { v } , c , \alpha )$ to describe its geometric and visual information. These parameters can be progressively optimized through differentiable rasterization using a rendering loss to achieve high-fidelity reconstruction. 2DGS implements both the forward rendering and backward gradient propagation in CUDA, enabling efficient and scalable operation.

## B. Camera Pose Optimization

Our proposed SLAM system does not rely on external visual odometry. Instead, we directly use rendering-based frame-to-map tracking to estimate the pose of each frame. The core problem of rendering-based tracking is computing the gradient of the rendering loss with respect to the camera pose. However, similar to 3DGS, the original 2DGS assumes that the camera poses of input frames are fixed and the loss generated from the forward rendering cannot propagate to them. Some 3DGS-based SLAM systems [38], [96], [102] apply the pose matrix directly to all Gaussian splats, and derive the gradient with respect to each element of the matrix by automatic differentiation. They then leverage differentiable transformation between quaternion and rotation matrix to obtain the quaternionâs gradient. However, these methods cannot guarantee that the gradient remains in SE(3) during the optimization process, resulting in a method that is neither efficient nor accurate.

To address this limitation, MonoGS [50] derives analytical Jacobians of camera pose in $S E ( 3 )$ for 3DGS and achieves efficient tracking. However, due to the difference of rendering mechanism, this derivation cannot be transfered to 2DGS directly. In our work, we bridge the gap and derive the camera Jacobians explicitly based on Lie algebra for 2DGS. To save memory overhead, the 2DGS map in our system does not use a spherical harmonic function to generate view dependent colors, so spherical harmonic function is not considered in the derivation below.

Since both the ray-splat intersection and alpha blendingbased rendering are differentiable, given the loss L generate between rendered image and input image, the per-element gradients of L with respect to the homography $\tau$ , denoted as $\overline { { \frac { \partial L } { \partial T } } }$ , can be obtained from 2DGSâs original implementation. Based on Eq. (6) and Eq. (7), we can derive the gradients of $\pmb { H } _ { c }$ from $\frac { \partial \bar { L } } { \partial \mathcal { T } }$ by applying the chain rule:

$$
\frac { \partial L } { \partial H _ { c } } = \left\lceil \frac { \partial L } { s _ { u } \partial t _ { u c } } \frac { \partial L } { s _ { v } \partial t _ { v c } } 0 \frac { \partial L } { \partial \pmb { \mu } _ { c } } \right\rceil = \pmb { W } ^ { \top } \frac { \partial L } { \partial \pmb { \mathcal { T } } } .\tag{9}
$$

From this gradient matrix, we can directly extract the gradients with respect to $\mathbf { \nabla } \mathbf { \mathcal { t } } _ { u c } , \mathbf { \nabla } \mathbf { \mathcal { t } } _ { v c } .$ , and $\pmb { \mu } _ { c }$ , which are given by $\begin{array} { r } { \frac { \partial L } { \partial t _ { u c } } , \frac { \partial L } { \partial t _ { v c } } } \end{array}$ , and $\frac { \partial L } { \partial \pmb { \mu } _ { c } }$ . Furthermore, according to Eq. (8), 2DGS can render normal images from splatsâ normal vector $\mathbf { \delta } _ { t _ { n c } }$ in the camera space. Then, the gradient of loss L with respect to $\mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta } \mathbf { \cdot } \mathbf { \Delta }$ , i.e., $\frac { \partial L } { \partial t _ { n c } }$ , can be computed through the backpropagation of alpha blending. Combining these results, we obtain the full gradient of $\scriptstyle { R _ { c } }$

$$
\frac { \partial L } { \partial \pmb { R _ { c } } } = \left[ \frac { \partial L } { \partial t _ { u c } } , \frac { \partial L } { \partial t _ { v c } } , \frac { \partial L } { \partial t _ { n c } } \right] .\tag{10}
$$

The camera pose $\mathbf { \delta } \mathbf { T } _ { c w }$ affects the rendered image by transforming each 2D Gaussian splat from world space to camera space. This transformation impacts both the center $\pmb { \mu }$ and the orientation R of the splat, producing their transformed counterparts $\pmb { \mu } _ { c }$ and $\scriptstyle { R _ { c } }$ in the camera coordinate system. Accordingly, the gradient of $\mathbf { \delta } \mathbf { T } _ { c w }$ is composed of two distinct components:

$$
\frac { \partial L } { \partial T _ { c w } } = \frac { \partial L } { \partial \pmb { \mu } _ { c } } \frac { \mathcal { D } \pmb { \mu } _ { c } } { \mathcal { D } \pmb { T } _ { c w } } \oplus \frac { \partial L } { \partial \pmb { R } _ { c } } \frac { \mathcal { D } \pmb { R } _ { c } } { \mathcal { D } \pmb { T } _ { c w } } ,\tag{11}
$$

where â ensures that both terms are projected into the same tangent space of $S E ( 3 )$ before summation, guaranteeing dimensional consistency. Adopting the same notation as in MonoGS [50], we define the partial derivative on the manifold as:

$$
\frac { \mathscr { D } f ( \pmb { T } ) } { \mathscr { D } \pmb { T } } = \operatorname* { l i m } _ { \tau \to 0 } \frac { \mathrm { L o g } ( f ( \mathrm { E x p } ( \tau ) \circ \pmb { T } ) \circ f ( \pmb { T } ) ^ { - 1 } ) } { \tau } ,\tag{12}
$$

where $\pmb { T } \in S E ( 3 )$ and $\tau \in s e ( 3 )$ , â¦ is a group composition operation. Then the two derivatives in Eq. (11) can be derived as following:

$$
\frac { \mathcal { D } \pmb { \mu } _ { c } } { \mathcal { D } \pmb { T } _ { c w } } = \left[ I \mathrm { \quad } - \pmb { \mu } _ { c } ^ { \times } \right] , \frac { \mathcal { D } \pmb { R } _ { c } } { \mathcal { D } \pmb { T } _ { c w } } = \left[ 0 \mathrm { \quad } - \pmb { R } _ { c , : , 2 } ^ { \times } \right] ,\tag{13}
$$

where Ã denotes the skew symmetric matrix of a 3D vector, and $; , i$ refers to the i-th column of the matrix. To ensure computational efficiency, we implement the above process in CUDA as well.

## C. System Overview

Leveraging the depth-consistent rendering capability of 2DGS, we develop a RGB-D SLAM system to enable accurate camera pose estimation alongside geometrically precise radiance field reconstruction. To further achieve online reconstruction of a globally consistent map, we extend the parameterization of 2D Gaussian splats. More specifically, our map representation can be expressed as:

$$
\begin{array} { r } { \mathcal { M } = \left\{ \pmb { g } _ { i } , \delta _ { i } , t _ { i } ^ { c } , d _ { i } ^ { c } , t _ { i } ^ { l } \mid i = 1 , . . . , N \right\} , } \end{array}\tag{14}
$$

where $\pmb { g } _ { i } = ( \pmb { \mu } , \pmb { R } , t _ { u } , t _ { v } , s _ { u } , s _ { v } , \pmb { c } , \alpha )$ is the original Gaussian splatâs learnable parameters as discussed in Sec. III-A. The item $t _ { i } ^ { c }$ denotes the sequential ID of the frame that observed $\mathbf { \pmb { g } } _ { i }$ at the closest distance, which is used to associate $\mathbf { \pmb { g } } _ { i }$ with its corresponding keyframe $t _ { i } ^ { c }$ for global map correction, and the closest distance is stored as $d _ { i } ^ { c }$ Meanwhile, $t _ { i } ^ { l }$ denotes the ID of the last frame that observed $\mathbf { \pmb { g } } _ { i }$ . The Boolean variable $\delta _ { i } = \{ 0 , 1 \}$ represents $\mathbf { \Xi } _ { \pmb { { g } } _ { i } \mathbf { \Xi } ^ { \ast } } \mathbf { { s } }$ active state. Based on this state, we can split all the Gaussian splats in the map $\mathcal { M }$ into two subsets $\mathcal { M } _ { A } = \{ \delta _ { i } = 1 | g _ { i } \in \mathcal { M } \}$ and $\mathcal { M } _ { I } = \{ \delta _ { i } = 0 \ | \ g _ { i } \in \mathcal { M } \}$ , representing active and inactive Gaussian splats respectively.

<!-- image-->  
Fig. 3: System overview of 2DGS-SLAM. Our system consists of two parallel processes: a front-end and a back-end. Taking RGB-D frames as input, the front-end performs frame-to-map camera tracking using the currently active map ${ \mathcal { M } } _ { A } ,$ and searches for potential loop closures. The selected keyframe is sent to the back-end, which uses it to expand and optimize the map (mapping). If a loop closure is detected in the front-end, we send the computed loop constraint to the back-end, where pose graph optimization and map correction will be performed. After mapping or pose graph optimization, the back-end updates the active state of the map and synchronizes it with the front-end. When no message is received, the back-end keeps refining the map based on previously stored keyframes.

As shown in Fig. 3, our system comprises two main process: the front-end and the back-end. The front-end is responsible for estimating the current camera pose and detecting potential loop closures. The back-end focuses on expanding and optimizing the map using frames with estimated poses, updating the active map, as well as globally deforming the map after a loop closure. We summarize the main components of our system as follows:

1) Tracking and keyframe selection (Sec. III-D): In the front-end, upon acquiring a new RGB-D frame, we estimate the camera pose using the currently active map $\mathcal { M } _ { A }$ through a frame-to-map tracking approach. Then, we select keyframes based on covisibility and send them to the back-end for mapping.

2) Mapping (Sec. III-E): We expand the map by projecting 2D Gaussian splats into the world space based on keyframes. The active map $\mathcal { M } _ { A }$ is then optimized for a few iterations using both current and historical keyframes. Afterwards, we send both $\mathcal { M } _ { A }$ and $\mathcal { M } _ { I }$ to the front-end for synchronization. Additionally, $\mathcal { M } _ { A }$ and $\mathcal { M } _ { I }$ are continuously refined using a selection of historical keyframes in the back-end, respectively.

3) Map state update (Sec. III-F): To prevent outdated map regions from negatively impacting tracking, we mark Gaussian splat $\mathbf { \pmb { g } } _ { i }$ that have not been observed for a certain period as inactive, i.e., $\delta _ { i } \ : = \ : 0$ . After a loop closure, we reactivate the observed inactive Gaussian splats to avoid redundancy.

4) Loop detection and relocalization (Sec. III-G): In the front-end, we identify loop closures by comparing each incoming frame with previous keyframes. If a candidate keyframe is found, we estimate the relative pose between this keyframe and current frames based on MASt3R [44]. The pose is then sent to the back-end and used to add a loop constraint to the pose graph.

5) Map correction (Sec. III-H): After receiving loop constraint from the front-end, we perform pose graph optimization, and transform every Gaussian according to the updated pose of its associating keyframe $t _ { i } ^ { c } .$ Finally, we reactivate observed inactive Gaussian splats and send the deformed map to the front-end.

The following sections will provide more comprehensive explanations of each component.

## D. Tracking and Keyframe Selection

Based on the derivation in III-B, we can optimize the current cameraâs pose $\textbf { \textit { T } } _ { c w }$ through gradient descent once the rendering loss is known. Given the input RGB image I and depth image D, we define the color rendering loss $\pmb { L } _ { I } \in \mathbb { R } ^ { H \times W }$ and depth rendering loss $\pmb { L } _ { D } \in \mathbb { R } ^ { H \times W }$ as:

$$
\begin{array} { r } { { \cal L } _ { I } = \| { \cal I } _ { r } - { \cal I } \| _ { 1 } , } \end{array}\tag{15}
$$

$$
L _ { D } = \lVert { \boldsymbol { D } } _ { r } - { \boldsymbol { D } } \rVert _ { 1 } ,\tag{16}
$$

where $\scriptstyle { I _ { r } }$ and $D _ { r }$ are RGB image and depth image rendered from active submap $\mathcal { M } _ { A }$ at pose $\mathbf { \delta } \mathbf { T } _ { c w }$ , and $\lVert \cdot \rVert _ { 1 }$ 1 means element-wise L1 distance.

According to Eq. (8), 2DGS can render normal vector images $\bar { \mathbf { N } _ { r } } \mathbf { \bar { \Lambda } } \in \mathbb { R } ^ { H \times \mathbf { \bar { W } } \times 3 }$ from 2D Gaussian splats. Utilizing this property, we can filter out the influence of back-facing splats on tracking by applying a normal mask $M _ { n }$ , which can be calculated by:

$$
M _ { n } ( x , y ) = [ [ \boldsymbol { N } _ { r } ( x , y ) ^ { \top } \boldsymbol { r } ( x , y ) > 0 ] ] ,\tag{17}
$$

where $\boldsymbol { r } ( x , y ) \in \mathbb { R } ^ { 3 }$ represents the normalized ray vector emitted from the cameraâs optic center and passing through pixel $( x , y )$ on the image plane, and Â· is the indicator J Kfunction returning 1 if the statement is true, otherwise 0. Note that all vectors in Eq. (17) are defined in the camera space. Besides, we apply another mask $M _ { o }$ obtained from the rendered opacity image $\pmb { o } \in \mathbb { R } ^ { H \times W }$ to ignore the loss generated from under-reconstructed area. The opacity mask is defined as:

$$
M _ { o } ( x , y ) = \mathbb { [ } O ( x , y ) > 0 . 9 5 \mathbb { ] } .\tag{18}
$$

Then, the total tracking loss $L _ { t } \in \mathbb { R }$ can be written as:

$$
L _ { t } = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } { M _ { n } ( p ) \cdot M _ { o } ( p ) \cdot ( L _ { I } ( p ) + \lambda _ { d } L _ { D } ( p ) ) } ,\tag{19}
$$

where $\Omega \ = \ \{ ( u , v ) \mid u \in 1 , . . . , W , v \in 1 , . . . , H \}$ represent all the pixels and $\lambda _ { d }$ is the weight used to balance these two losses, and Â· represents the per-element product. We directly initialize the optimization using last frameâs pose and employ the AdamW [47] algorithm to iteratively optimize the pose until convergence or reaching the maximum number of iterations $n _ { \mathrm { i t e r } }$ . To ensure that the gradients are stable, during the optimization process of camera tracking, the parameters of these Gaussian splats remain fixed. It should be noted that, during tracking, we only render images from the active Gaussian splats, i.e., $\mathbf { \omega } _ { \mathbf { { \mathbf { g } } } } \in \mathcal { M } _ { A }$

As with most SLAM frameworks, rather than using all frames for mapping, we selectively choose keyframes to improve efficiency. Similar to MonoGS [50], we primarily determine keyframes based on the covisibility.

First, from Eq. (8), the alpha-blending coefficient of Gaussian splat $\mathbf {  { g } } _ { k }$ in the rendering of pixel p is:

$$
w _ { k } ( \pmb { p } ) = \alpha _ { k } \mathcal { G } ( \pmb { u } _ { k } ^ { p } ) \prod _ { j = 0 } ^ { k - 1 } ( 1 - \alpha _ { j } \mathcal { G } ( \pmb { u } _ { j } ^ { p } ) ) .\tag{20}
$$

We define the contribution of $\mathbf {  { g } } _ { k }$ to a rendered frame from a given view $V$ as the sum of its rendering coefficients across all pixels, expressed by:

$$
\mathcal { C } _ { k } ^ { V } = \sum _ { p \in \Omega } w _ { k } ( \pmb { p } ) ,\tag{21}
$$

where $p \in \Omega$ denotes all pixels. Intuitively, $\mathcal { C } _ { k } ^ { V }$ represents how many pixels the Gaussian splat $\mathbf {  { g } } _ { k }$ contributes to the rendering. Therefore, we directly define that $\mathbf {  { g } } _ { k }$ is visible in the given view $V$ if its $\mathcal { C } _ { k } ^ { V }$ is larger than 0.5. Furthermore, Given two camera views A, B and current active map ${ \mathcal { M } } _ { A } ,$ we define the covisibility score between these two views as follows:

$$
S _ { c o v } ( A , B ) = \frac { | G _ { A } \cap G _ { B } | } { | G _ { A } \cup G _ { B } | } ,\tag{22}
$$

where $G _ { A } = \{ g _ { k } \in \mathcal { M } _ { A } \mid \mathcal { C } _ { k } ^ { A } > 0 . 5 \}$ and similarly for $G _ { B }$ with $\mathcal { C } _ { k } ^ { B }$ . They are the sets of all visible Gaussian splats at view A and $B ,$ respectively. If the covisibility score between the current view and the last keyframe falls below a threshold $c _ { k }$ , or if their distance between the translation vectors of $\mathbf { \delta } \mathbf { T } _ { c w }$ surpasses a threshold $d _ { k } ,$ , the current view is selected as a keyframe.

## E. Mapping

After completing pose estimation in the front-end, the new keyframe $K _ { n }$ observed by the robot is sent to the backend process for map expansion and optimization. To reduce memory consumption, we first convert the RGB-D data into a colored point cloud P, and then apply random downsampling to obtain $\mathcal { P } _ { s }$ before projecting it into the world space. Each 3D point is then initialized as a Gaussian splat, where its initial scales $s _ { u }$ and $s _ { v }$ are determined by the distance d to its nearest neighbor in ${ \mathcal { P } } _ { s } ,$ , and the opacity Î± is set to an initial value of 0.99. The initial normal vector $\scriptstyle { { \ t } _ { n } }$ is obtained from the normal image $N _ { D }$ , which is computed from the pixel gradient of the depth image D. Specifically, we derive $N _ { D }$ from the cross product of neighboring pixel differences in $_ { D }$

$$
N _ { D } ( x , y ) = \frac { \nabla _ { x } D ( x , y ) \times \nabla _ { y } D ( x , y ) } { | \nabla _ { x } D ( x , y ) \times \nabla _ { y } D ( x , y ) | } .\tag{23}
$$

Here, we assign $\scriptstyle { { \ t } _ { n } }$ to each splat based on its corresponding position. We than randomly initialize two principal tangential vectors $\mathbf { \Delta } _ { t _ { u } }$ and $\mathbf { \Delta } _ { t _ { v } } .$ , which are perpendicular to $\scriptstyle { { \ t } _ { n } }$ . In addition, the closest frame ID $t _ { i } ^ { c }$ and last observed frame ID $t _ { i } ^ { l }$ of the new splat are both initialized as the ID of the current keyframe.

To prevent the generation of excessively redundant Gaussian splats, we maintain a voxel hash table with resolution $r _ { h }$ for the active submap $\mathcal { M } _ { A }$ , which represents the spatial occupancy state. New Gaussian splats are only created in spatially unoccupied voxels where no existing Gaussians are present. This voxel hash table is updated whenever the map expands by adding new splats and when modifications occur in the active state, such as transitioning Gaussians between active and inactive states. Due to its relatively low resolution and restriction to active regions, the memory overhead remains minimal.

In the back-end process, we continuously optimize all the Gaussian splats $\{ g \in \mathcal { M } \}$ to ensure they not only produce high-fidelity image renderings but also align well with the actual surface, accurately capturing the geometric structure of the environment. To achieve this, we train the map with multiple loss functions. Firstly, the color image rendering loss $\dot { \pmb { L } } _ { c } \in \mathbb { R } ^ { H \times W }$ is expressed as:

$$
\boldsymbol { L _ { c } } = \lambda _ { c } \left. \boldsymbol { I _ { r } } - \boldsymbol { I } \right. _ { 1 } + ( 1 - \lambda _ { c } ) L _ { S S I M } ( \boldsymbol { I _ { r } } , \boldsymbol { I } ) ,\tag{24}
$$

where $\lambda _ { c } \in [ 0 , 1 ]$ and the $L _ { S S I M }$ represents the structural similarity index measure (SSIM) [87]. We also apply an L1 loss $\mathbf { \delta L } _ { D }$ like Eq. (16) to directly supervise the depth rendering optimization using the input depth image. Following 2DGS, to ensure that the Gaussian splats conform to the surface locally, we add a normal consistency loss $\pmb { L } _ { n } \in \mathbb { R } ^ { H \times W }$ between the rendered depth image $D _ { r }$ and the rendered normal image $N _ { r } ,$ formulated as:

$$
\pmb { L } _ { n } = \mathbf { 1 } _ { H \times W } - N \pmb { D } _ { r } \cdot N _ { r } ,\tag{25}
$$

where $\smash { N _ { D _ { r } } \ \in \ \mathbb { R } ^ { H \times W \times 3 } }$ denotes the normal image estimated from the rendered depth image $D _ { r }$ by applying Eq. (23), and Â· indicates a per-pixel 3D vector dot product.

Finally, we optimize the map M using the combination of above loss functions, which can be written as:

$$
L _ { m } = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } ( { \cal L } _ { c } ( p ) + w _ { d } L _ { D } ( p ) + w _ { n } L _ { n } ( p ) ) ,\tag{26}
$$

where $w _ { d } , w _ { n }$ are weights to balance the contributions of the corresponding loss terms. Utilizing the loss function $L _ { m } ,$ we continually optimize the active map $\mathcal { M } _ { A }$ and inactive map $\mathcal { M } _ { I }$ separately in the back-end. For $\mathcal { M } _ { A }$ , we maintain an active frames set $\scriptstyle { \mathcal { S } } _ { a }$ , which is defined as:

$$
S _ { a } = \{ t _ { i } ^ { c } \mid g _ { i } \in \mathcal { M } _ { A } \} ,\tag{27}
$$

where $t _ { i } ^ { c } ,$ , as described in Eq. (14), is the ID of the closest observing frame of $\mathbf { \mathscr { g } } _ { i } .$ . In each iteration, we randomly sample $N _ { a }$ frames from $ { \boldsymbol { S } } _ { a }$ and $N _ { i }$ frames from the other frames to perform optimization for Gaussian splats in $\mathcal { M } _ { A }$ and $\mathcal { M } _ { I }$ , respectively.

## F. Map State Update

Due to the accumulation of tracking errors, directly integrating newly observed data into the global map can cause a misalignment between new and existing structures, negatively impacting re-localization after loop detection. To mitigate this issue, as mentioned in Sec. $\mathrm { I I I - C } .$ , we maintain two separate maps: an active map $\mathcal { M } _ { A }$ , which stores recently observed Gaussian splats, and an inactive map $\mathcal { M } _ { I }$ , which preserves historical splats.

Given the latest posed keyframe $K _ { n }$ sent from the frontend, where n is its sequential ID, we first assign active status $( \delta \ = \ 1 )$ to all new Gaussian splats generated from $K _ { n }$ and add them to active map $\mathcal { M } _ { A }$ . Using the visibility criterion defined in Eq. (21), we identify which Gaussian splats in $\mathcal { M } _ { A }$ are visible from the view of $K _ { n }$ and then update their last observed frame ID to $t _ { i } ^ { l } = n$ . Meanwhile, we compute the distance from these Gaussian splat to the viewpoint of $K _ { n }$ . If the distance is smaller than the historical minimum distance $d _ { i } ^ { c }$ , we update their closest frame ID as $t _ { i } ^ { c } = n$ . For all $\mathbf { \Delta } _ { \mathbf { \mathcal { G } } _ { i } } \in \mathcal { M } _ { A }$ , we mark $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { i } }$ as inactive $( \delta = 0 )$ if $( n - t _ { i } ^ { l } )$ exceeds a predefined time threshold, indicating that $\mathbf { \pmb { g } } _ { i }$ has not been observed for a long time.

As illustrated in Fig. 4, to avoid accumulating redundant Gaussian splats in the same region, we reactivate inactive splats in $\mathcal { M } _ { I }$ when the robot revisits previously observed areas. More specifically, if a loop closure is detected between the current frame $\pmb { F } _ { c }$ and a historical keyframe $K _ { h } ,$ , we first perform pose graph optimization followed by map correction. For the subsequent $T$ keyframes, where $T$ is a predefined hyperparameters, if an inactive Gaussian splat $\mathbf { \Psi } _ { \mathbf { \psi } _ { \mathbf { \hat { \mu } } } } \in \mathcal { M } _ { I }$ is observed, and its closed observed frame ID $t _ { i } ^ { c } > h ,$ indicating that its position has been corrected by pose graph optimization, we reassign its state as active and update its last observed frame ID $t _ { i } ^ { l }$ to the current frame ID c. In addition, during the mapping process, we continuously sample historical keyframes, such as $K _ { r }$ , and evaluate the contributions of the Gaussian splats associated with $K _ { r } ,$ i.e., $\pmb { \mathscr { g } } \in \{ t _ { i } ^ { c } = r \ | \ \pmb { \mathscr { g } } _ { i } \in \mathscr { M } \}$ based on Eq. (21). If the contribution of a Gaussian splat falls below 0.5, we consider it occluded by surrounding splats and remove it from the map to maintain map compactness.

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->

<!-- image-->  
(d)  
Fig. 4: Illustration of the state update process. Images (a)-(d) are shown in temporal order. The bright regions represent the active map, while the dim regions indicate the inactive map. The red cones represent the current camera views. (a) Active and inactive maps during running. (b) The camera observes part of the inactive map, but no loop closure has been detected. Due to pose drift, the active and inactive maps misalign (see red arrows). (c) The system detects a loop closure, aligns inactive and active maps, and reactivates the observed inactive Gaussian splats. (d) As the camera moves, more inactive Gaussian splats are progressively reactivated.

## G. Loop Detection and Relocalization

In the back-end, we maintain a pose graph G, where each keyframe serves as a vertex, and the relative pose between adjacent keyframes forms an edge in the graph. When a loop closure is detected, we compute the relative pose between the current frame and the candidate frame searched from all the keyframes to introduce a loop closure constraint for pose graph optimization.

We primarily utilize MASt3R [44] for loop detection and re-localization. Given a pair of input RGB images $\langle { \cal I } _ { i } , { \cal I } _ { j } \rangle$ , MASt3R extracts their image features $\mathcal { F } _ { i } , \mathcal { F } _ { j }$ through a vision transformer-based model [17] and directly outputs pixelwise point clouds, $P _ { i }$ and $P _ { j }$ , along with their respective confidence maps, $C _ { i }$ and $C _ { j }$ . Notably, both $P _ { i }$ and $P _ { j }$ are represented in the camera coordinate frame of view i. By leveraging these dense point clouds, we can estimate the camera parameters of the two frames and subsequently solve for their relative pose using the PnP algorithm [81], thereby obtaining the depth maps $D _ { i } ^ { p }$ and $D _ { j } ^ { p }$ in their respective camera space.

Inspired by recent works [18], [55], we use features $\mathcal { F }$ from the vision transformer encoder as local descriptors, and employ the aggregated selective match kernel (ASMK) [79] for image retrieval. ASMK quantizes and binarizes these features using a precomputed k-means codebook, producing high-dimensional sparse binary representations. The similarity between images can be efficiently computed via a kernel function over shared codebook elements. We integrate this process into our online system. For each keyframe $K _ { n } .$ , we utilize MASt3Râs feature encoder to extract image feature $\mathcal { F } _ { n }$ and store it, along with the corresponding image sequence ID $n ,$ in a feature database managed using ASMK.

<!-- image-->  
Fig. 5: We input the current frame and the loop candidate into MAsT3R to estimate their relative pose and dense point clouds. After confidence and overlap checks, we optimize the scale of computed relative pose by aligning the point cloud to the local map. The scaled pose is then transfered to the world space and refined by tracking it on the active map $\mathcal { M } _ { A }$

Fig. 5 illustrates the main pipeline of our loop closure detection. After tracking the current image $\displaystyle I _ { c } ,$ we extract its features and compute similarities with all keyframes. We then identify the keyframe with the highest similarity to $\mathbf { \nabla } _ { I _ { c } . }$ If its similarity score exceeds a predefined threshold $s _ { r } ,$ we designate it as a loop closure candidate. Let $\pmb { I } _ { l }$ and $D _ { l }$ denote the RGB and depth images of the candidate keyframe $\pmb { K } _ { l }$ , respectively. To further validate the loop closure, we feed the image pair $\langle \pmb { I } _ { c } , \pmb { I } _ { l } \rangle$ to MASt3R, obtaining the predicted point clouds $P _ { c }$ and $P _ { l }$ , the corresponding depth maps $D _ { c } ^ { p }$ and $D _ { l } ^ { p }$ , their confidence maps $C _ { c }$ and $C _ { l }$ and the estimated relative pose $\mathbf { \nabla } T _ { l c }$ . If the mean of confidence map $C _ { c }$ is below a predefined threshold $c _ { s } ,$ we consider the prediction unreliable and discard this loop closure candidate. Otherwise, we estimate the overlap between the two frames as follows. Since the predicted point clouds are both expressed in the coordinate frame of $\displaystyle I _ { c } ,$ we directly reproject $P _ { l }$ onto the image plane of $\pmb { I _ { c } }$ and compare the reprojected depth $D _ { c } ^ { l }$ with the predicted depth $D _ { c } ^ { p }$ . Inspired by [8], the overlap ratio $O _ { l c }$ is computed as:

$$
O _ { l c } = \frac { \sum _ { \boldsymbol { u } \in \mathcal { V } } \mathbf { 1 } \left( \left| D _ { c } ^ { l } ( \boldsymbol { u } ) - D _ { c } ^ { p } ( \boldsymbol { u } ) \right| < \tau _ { d } \right) } { | \mathcal { V } | } ,\tag{28}
$$

where V denotes the pixels falling within the image boundaries after projection, 1 is an indicator function that returns 1 if the condition inside is true and 0 otherwise. $\tau _ { d }$ is a depth consistency threshold, which is 0.05 in our setting. If $O _ { l c }$ is smaller than a predefined threshold $\delta _ { o } ,$ we determine that the candidate frame lacks sufficient overlap for reliable relocalization and reject the loop closure attempt.

If the candidate passes the filtering, we further compute the accurate relative pose to provide a loop closure constraint for pose graph optimization. Although MASt3R is trained on a large amount of metric-scale data, its predicted depth maps remain up to scale. Therefore, we first estimate the scale factor $s ^ { * }$ using real observed depth image $D _ { c } .$ , given by:

$$
s ^ { * } = \arg \operatorname* { m i n } _ { s } \| C _ { c } \cdot ( D _ { c } - s D _ { c } ^ { p } ) \| _ { 2 } \mathrm { ~ , ~ }\tag{29}
$$

where Â· represents the per-element product for matrices. This is a weighted least squares problem that can be solved in closed form. Then, we multiply this scale factor with the translation component of $\mathbf { { T } } _ { l c }$ to obtain the scaled relative pose $\pmb { T } _ { l c } ^ { r }$ . Consequently, we derive the candidate keyframeâs pose in the world space as $\pmb { T } _ { l } = \pmb { T } _ { l c } ^ { r } \pmb { T } _ { c } ,$ , where $\mathbf { \delta } _ { \mathbf { \mathcal { T } } _ { c } }$ is the pose of the current camera. To obtain a more accurate estimation, we use $\mathbf { \delta } _ { \mathbf { \mathcal { T } } _ { l } }$ as the initial estimate and perform a scan-tomodel tracking in current active map $\mathcal { M } _ { A } .$ . After convergence or reaching the maximum number of iterations $n _ { i t e r }$ we re-render a depth image $D _ { r }$ from the active map at the optimized pose $\boldsymbol { T } _ { l } ^ { t }$ and compute its L1 error $e _ { t }$ against the input depth image D using the same formulation as Eq. (19):

$$
\boldsymbol { e } _ { t } = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } \boldsymbol { M } _ { n } ( p ) \cdot \boldsymbol { M } _ { o } ( p ) \cdot \left. \boldsymbol { D } _ { r } ( p ) - \boldsymbol { D } ( p ) \right. _ { 1 } ,\tag{30}
$$

where $M _ { n }$ and $M _ { o }$ are normal and opacity masks, respectively. Only frame with an error $e _ { t }$ below a threshold $\varepsilon _ { t }$ retained for further refinement. Then, the successfully optimized pose $\boldsymbol { \mathbf { \mathit { T } } } _ { l } ^ { t }$ is used to caculate the accurate relative pose as $\mathbf { \dot { T } } _ { l c } ^ { t } = \mathbf { \dot { T } } _ { l } ^ { t } \mathbf { T } _ { c } ^ { - 1 }$ . With $\boldsymbol { T } _ { l c } ^ { t }$ as the loop constraint, we perform pose graph optimization and update the poses of all keyframes.

To increase the number of valid loop closures and further improve the mapping accuracy, we extend our loop detection beyond image feature querying by also revisiting the inactive map. Specifically, after performing tracking based on the active map $\mathcal { M } _ { A }$ for each incoming frame, we additionally render images using the inactive Gaussian splats $\{ \pmb { g } \in \mathcal { M } _ { I } \}$ If the area of valid region in the rendered opacity image $O _ { i }$ exceeds a threshold $a _ { v } .$ , indicating that the robot has observed part of the historical map. In this case, we count the occurrence numbers of all the closest observing frame ID $t _ { i } ^ { c }$ among all observed Gaussian splats and accordingly select the keyframe with the highest count as the loop closure candidate. We then input this candidate and the current frame into MASt3R, applying the same selection and tracking pipeline as described earlier.

## H. Map Correction

After pose graph optimization, each keyframe $K _ { i }$ with pose $\mathbf { \nabla } T _ { i }$ is updated with an optimized pose increment:

$$
\Delta { { T } _ { i } } = { { T } _ { i } ^ { o } } { { T } _ { i } ^ { - 1 } } ,\tag{31}
$$

where $\mathbf { \nabla } _ { \mathbf { \boldsymbol { T } } _ { i } }$ is the original pose, and $\mathbf { \delta } _ { \mathbf { \mathcal { T } } _ { i } ^ { o } }$ is the optimized pose. For each Gaussian splat $\pmb { g } _ { k } \in \mathcal { M }$ , let $f _ { c } ^ { k }$ be the frame ID of its closest observed keyframe. We apply the corresponding pose increment $\Delta \boldsymbol { T } _ { f _ { c } ^ { k } }$ to update the Gaussianâs position $\pmb { \mu } _ { k }$ and orientation $\scriptstyle { R _ { k } } :$

$$
\pmb { \mu } _ { k } ^ { \prime } = \Delta \pmb { T } _ { f _ { c } ^ { k } } \pmb { \mu } _ { k } , \pmb { R } _ { k } ^ { \prime } = \Delta \pmb { R } _ { f _ { c } ^ { k } } \pmb { R } _ { k } ,\tag{32}
$$

where $\pmb { \mu } _ { k }$ and $\scriptstyle { \mathbf { } } _ { R _ { k } }$ represent the original position and rotation of the Gaussian $\mathbf {  { g } } _ { k }$ , and $ { \boldsymbol { { x } } } _ { k } ^ { \prime }$ and $\scriptstyle { R _ { k } ^ { \prime } }$ are the updated values.

<!-- image-->  
Fig. 6: Result of loop closure and map correction. The left figures illustrate the associated keyframe ID of each Gaussian splat and the camera trajectory. Both IDs and the trajectory are colored by time. The right figures compare the reconstruction results before and after loop closure. It can be observed that the map suffers from severe drift and misalignment before the loop correction (highlighted by red arrows). After that, the map structure becomes cleaner and more consistent.

As illustrated in Fig. 6, our method ensures that the entire Gaussian map is deformed consistently with the optimized keyframe poses, preserving spatial coherence.

## IV. EXPERIMENTAL EVALUATION

The main focus of this work is a rendering-based RGB-D SLAM system for building geometrically accurate and globally consistent radiance fields using 2D Gaussian splatting.

We present our experiments to show the capabilities of our method and analyze its performance. The results of our experiments also support our key claims, which are (i) Our proposed 2DGS-SLAM system demonstrates higher tracking accuracy than state-of-the-art rendering-based methods and traditional dense SLAM methods based on TSDF or surfel representations. (ii) Our method outperforms 3DGS-based approaches in terms of surface reconstruction quality. The incorporation of the efficient loop closure mechanism ensures more globally consistent reconstruction results. At the same time, our method also achieves comparable or superior image rendering quality, making the reconstruction result wellsuited for downstream tasks. (iii) Our 2DGS-SLAM is much more efficient in terms of runtime than other rendering-based SLAM systems with loop closures and generates a more compact map representation.

## A. Experimental Setup

1) Datasets: We conduct our experiments on three public datasets that are widely adopted for performance evaluation in rendering-based SLAM methods as well as self-recorded data from a mobile robot. These datasets are the synthetic dataset Replica [72], and two real datasets, TUM-RGBD [74] and ScanNet [10]. The Replica dataset provides ground-truth camera poses along with an accurate mesh of the target environment. The TUM-RGBD dataset includes accurate camera poses captured using a motion capture system, while the reference poses in the ScanNet dataset are provided by BundleFusion [11]. It is worth noting that the depth images in the Replica dataset are rendered directly from the mesh and therefore free from noise. In contrast, both TUM-RGBD and ScanNet datasets are captured using consumergrade structured-light-based RGB-D sensors, which introduce noticeable motion blur and depth measurement noise, presenting additional challenges for rendering-based SLAM algorithms. In addition to the three public datasets mentioned above, to evaluate the performance of our method on a real robotic platform, we recorded data using a wheeled robot in an indoor environment and conducted quantitative experiments on pose estimation.

TABLE I: Hyperparameters of our approach
<table><tr><td>symbol</td><td>value</td><td>description</td></tr><tr><td colspan="3">Tracking and Keyframe Selection, Sec. III-D</td></tr><tr><td> $c _ { k }$   $d _ { k }$   $n _ { \mathrm { i t e r } }$ </td><td>0.9 15 (cm) 120</td><td>covisibility threshold for keyframe selection distance threshold for keyframe selection maximum number of iterations</td></tr><tr><td colspan="3">Mapping, Sec. III-E</td></tr><tr><td> $\lambda _ { c }$   $w _ { d }$   $w _ { n }$ </td><td>0.125 0.5 0.02</td><td>weight of the SSIM loss weight of depth loss weight of normal loss</td></tr><tr><td colspan="3"> $N _ { a }$  3 active map training frames per iteration  $N _ { i }$  2 inactive map training frames per iteration</td></tr><tr><td> $s _ { r }$ </td><td>0.025</td><td>Loop Detection and Relocalization, Sec. III-G similarity score threshold for image retrieval</td></tr><tr><td> $c _ { s }$ </td><td>3.0 0.5</td><td>mean confidence score threshold valid region threshold for revisiting loop</td></tr></table>

2) Implementation details: We summarize the hyperparameters of our SLAM system, previously mentioned throughout the paper, in Tab. I. These settings are kept consistent across all experiments. In addition, optimizationrelated parameters for 2D Gaussian splats, such as learning rates for different components, are also fixed for all datasets. Due to variations in depth sensor accuracy, however, we adjust the tracking depth loss weight $\lambda _ { d }$ and the tracking success threshold $\varepsilon _ { t }$ individually for each dataset. The pose graph optimization is carried out using GTSAM [14], employing the Levenberg-Marquardt method with a maximum iteration limit of 50. We implement our system mainly using PyTorch, and all reported experiments are conducted on an NVIDIA A6000 GPU.

After completing the pose estimation of all frames, we directly merge the active map $\mathcal { M } _ { A }$ and inactive map $\mathcal { M } _ { I }$ to form a complete scene representation, which is then used to evaluate both reconstruction and rendering quality. Following prior works [46], [50], [68], [102], we incorporate a map refinement stage to further enhance reconstruction results. Specifically, we perform an additional optimization of the map using all keyframes for 26,000 iterations.

TABLE II: Absolute Trajectory Error (ATE) on the Replica dataset, reported in centimeters. LC denotes that loop closure is enabled. We highlight the best results in bold and the second best results are underscored.
<table><tr><td>Method</td><td>Map Representation</td><td>LC</td><td>Rm 0</td><td>Rm 1</td><td>Rm 2</td><td>Off0</td><td>Off1</td><td>Off2</td><td>Off3</td><td>Off4</td><td>Avg.</td></tr><tr><td>NICE-SLAM [103]</td><td>feature grids</td><td>X</td><td>0.97</td><td>1.31</td><td>1.07</td><td>0.88</td><td>1.00</td><td>1.06</td><td>1.10</td><td>1.13</td><td>1.06</td></tr><tr><td>GO-SLAM [99]</td><td>feature grids</td><td></td><td>0.34</td><td>0.29</td><td>0.29</td><td>0.32</td><td>0.30</td><td>0.39</td><td>0.39</td><td>0.46</td><td>0.35</td></tr><tr><td>E-SLAM [37]</td><td>feature planes</td><td></td><td>0.71</td><td>0.70</td><td>0.52</td><td>0.57</td><td>0.55</td><td>0.58</td><td>0.72</td><td>0.63</td><td>0.63</td></tr><tr><td>Point-SLAM [68]</td><td>feature points</td><td></td><td>0.61</td><td>0.41</td><td>0.37</td><td>0.38</td><td>0.48</td><td>0.54</td><td>0.69</td><td>0.72</td><td>0.52</td></tr><tr><td>Loopy-SLAM [46]</td><td>feature points</td><td></td><td>0.24</td><td>0.24</td><td>0.28</td><td>0.26</td><td>0.40</td><td>0.29</td><td>0.22</td><td>0.35</td><td>0.29</td></tr><tr><td>PIN-SLAM [61]</td><td>feature points</td><td></td><td>0.27</td><td>0.31</td><td>0.13</td><td>0.22</td><td>0.30</td><td>0.28</td><td>0.16</td><td>0.28</td><td>0.24</td></tr><tr><td>RTG-SLAM [63]</td><td>3DGS</td><td></td><td>0.20</td><td>0.18</td><td>0.13</td><td>0.22</td><td>0.12</td><td>0.22</td><td>0.20</td><td>0.19</td><td>0.18</td></tr><tr><td>MonoGS [50]</td><td>3DGS</td><td></td><td>0.33</td><td>0.22</td><td>0.29</td><td>0.36</td><td>0.19</td><td>0.25</td><td>0.12</td><td>0.81</td><td>0.32</td></tr><tr><td>SplaTAM [38]</td><td>3DGS</td><td></td><td>0.31</td><td>0.40</td><td>0.29</td><td>0.47</td><td>0.27</td><td>0.29</td><td>0.32</td><td>0.72</td><td>0.38</td></tr><tr><td>Gaussian-SLAM [96]</td><td>3DGS</td><td></td><td>0.29</td><td>0.29</td><td>0.22</td><td>0.37</td><td>0.23</td><td>0.41</td><td>0.30</td><td>0.35</td><td>0.31</td></tr><tr><td>LoopSplat [102]</td><td>3DGS</td><td>&gt;xx*xx</td><td>0.28</td><td>0.22</td><td>0.17</td><td>0.22</td><td>0.16</td><td>0.49</td><td>0.20</td><td>0.30</td><td>0.26</td></tr><tr><td>2DGS-SLAM (ours)</td><td>2DGS</td><td>â</td><td>0.06</td><td>0.08</td><td>0.10</td><td>0.04</td><td>0.07</td><td>0.07</td><td>0.06</td><td>0.09</td><td>0.07</td></tr></table>

TABLE III: Absolute Trajectory Error (ATE) on the TUM dataset, reported in centimeters. LC denotes that loop closure is enabled. We highlight the best results in bold and the second best results are underscored. We separately compare the performance of renderingbased methods and classical approaches.
<table><tr><td>Method</td><td>LC</td><td>desk</td><td>desk2</td><td>room</td><td>xyz</td><td>office</td><td>Avg.</td></tr><tr><td colspan="8">Rendering-based approach</td></tr><tr><td>NICE-SLAM [103]</td><td>X</td><td>4.26</td><td>4.99</td><td>34.49</td><td>6.19</td><td>3.87</td><td>10.76</td></tr><tr><td>E-SLAM [37]</td><td></td><td>2.47</td><td>3.69</td><td>29.73</td><td>1.11</td><td>2.42</td><td>7.89</td></tr><tr><td>Point-SLAM [68]</td><td>ÃÃ&gt;</td><td>4.34</td><td>4.54</td><td>30.92</td><td>1.31</td><td>3.48</td><td>8.92</td></tr><tr><td>Loopy-SLAM [46]</td><td></td><td>3.79</td><td>3.38</td><td>7.03</td><td>1.62</td><td>3.41</td><td>3.85</td></tr><tr><td>MonoGS [50]</td><td></td><td>1.59</td><td>7.03</td><td>8.55</td><td>1.44</td><td>1.49</td><td>4.02</td></tr><tr><td>SplaTAM [38]</td><td></td><td>3.35</td><td>6.54</td><td>11.13</td><td>1.24</td><td>5.16</td><td>5.48</td></tr><tr><td>Gaussian-SLAM [96]</td><td></td><td>2.73</td><td>6.03</td><td>14.92</td><td>1.39</td><td>5.31</td><td>6.08</td></tr><tr><td>LoopSplat [102]</td><td>Ã*Ã&gt;&gt;</td><td>2.08</td><td>3.54</td><td>6.24</td><td>1.58</td><td>3.22</td><td>3.33</td></tr><tr><td>2DGS-SLAM (ours)</td><td></td><td>1.84</td><td>2.76</td><td>5..98</td><td>1.16</td><td>1.97</td><td>2.74</td></tr><tr><td colspan="8">Classical SLAM approach</td></tr><tr><td>Kintinuous [89]</td><td></td><td>3.7</td><td>7.1</td><td>7.5</td><td>2.9</td><td>3.0</td><td>4.84</td></tr><tr><td>ElasticFusion [90]</td><td>â</td><td>2.0</td><td>4.8</td><td>6.8</td><td>1.1</td><td>1.7</td><td>3.28</td></tr><tr><td>ORB-SLAM2 [54]</td><td>â</td><td>1.6</td><td>2.2</td><td>4.7</td><td>0.4</td><td>1.0</td><td>2.0</td></tr><tr><td>RTAB-Map [43]</td><td>â</td><td>2.9</td><td>4.4</td><td>6.6</td><td>0.5</td><td>2.1</td><td>3.3</td></tr></table>

TABLE IV: Absolute trajectory error (ATE) on the ScanNet dataset (cm). LC denotes that loop closure is enabled. We highlight the best results in bold and the second best results are underscored.
<table><tr><td>Method</td><td>LC</td><td>00</td><td>59</td><td>106</td><td>169</td><td>181</td><td>207</td><td>54</td><td>233</td><td>Avg.</td></tr><tr><td>NICE-SLAM</td><td>X</td><td>12.0</td><td>14.0</td><td>7.9</td><td>10.9</td><td>13.4</td><td>6.2</td><td>20.9</td><td>9.0</td><td>13.0</td></tr><tr><td>GO-SLAM</td><td>&gt;*x*x*&gt;</td><td>5.4</td><td>7.5</td><td>7.0</td><td>7.7</td><td>6.8</td><td>6.9</td><td>8.8</td><td>4.8</td><td>6.9</td></tr><tr><td>E-SLAM</td><td></td><td>7.3</td><td>8.5</td><td>7.5</td><td>6.5</td><td>9.0</td><td>5.7</td><td>36.3</td><td>4.3</td><td>10.6</td></tr><tr><td>Point-SLAM</td><td></td><td>10.2</td><td>7.8</td><td>8.7</td><td>22.0</td><td>14.8</td><td>9.5</td><td>28.0</td><td>6.1</td><td>14.3</td></tr><tr><td>Loopy-SLAM</td><td></td><td>4.2</td><td>7.5</td><td>8.3</td><td>7.5</td><td>10.6</td><td>7.9</td><td>7.5</td><td>5.2</td><td>7.7</td></tr><tr><td>MMonoGs</td><td></td><td>9.8</td><td>32.1</td><td>8.9</td><td>10.7</td><td>21.8</td><td>7.9</td><td>17.5</td><td>12.4</td><td>15.2</td></tr><tr><td>platTAM</td><td></td><td>12.8</td><td>10.1</td><td>17.7</td><td>12.1</td><td>11.1</td><td>7.5</td><td>56.8</td><td>4.8</td><td>16.6</td></tr><tr><td>Gaussian-SLAM</td><td></td><td>21.2</td><td>12.8</td><td>13.5</td><td>16.3</td><td>21.0</td><td>14.3</td><td>37.1</td><td>11.1</td><td>18.4</td></tr><tr><td>LoopSplat</td><td></td><td>6.2</td><td>7.1</td><td>7.4</td><td>10.6</td><td>8.5</td><td>6.6</td><td>16.0</td><td>4.7</td><td>8.4</td></tr><tr><td>2DGS-SLAM</td><td>â</td><td>6.6</td><td>6.9</td><td>7.1</td><td>6.5</td><td>8.2</td><td>6.0</td><td>11.0</td><td>4.7</td><td>7.1</td></tr></table>

## B. Tracking Performance

The first experiment evaluate how well our approach estimates the camera poses and compare it to existing baselinws. The results of theis experiment support our first claim that our 2DGS-SLAM system demonstrates higher tracking accuracy. We evaluate tracking performance on all three datasets using the ATE RMSE [74] as the metric. Among them, the Replica dataset is widely adopted for benchmarking rendering-based SLAM systems. On this synthetic dataset, we compare our 2DGS-SLAM method with several state-of-the-art approaches based on NeRF [52], 3D Gaussian splatting, and neural signed distance fields. As shown in Tab. II, our proposed 2DGS-SLAM outperforms all baselines, achieving sub-millimeter tracking accuracy. The trajectory error of our method is only half that of the secondbest method, RTG-SLAM [63], which estimates camera poses by integrating multi-level ICP with ORB-SLAM2 [54], demonstrating the advantage and potential of rendering-based methods compared to traditional approaches. This strong performance is largely attributed to the high-quality, noise-free depth images provided by Replica, which enable our 2DGS representation to fully exploit the advantages of consistent depth rendering. These results also validate the effectiveness of our rendering-based camera pose optimization approach.

For the tracking results on the TUM-RGBD dataset, in addition to rendering-based methods, we also compare against classical RGB-D SLAM approaches such as Kintinuous [89], ElasticFusion [90], ORB-SLAM2 [54], and RTAB-Map [43]. As reported in Tab. III, among rendering-based methods, our 2DGS-SLAM outperforms all baselines in terms of average accuracy. In smaller-scale sequences such as desk, xyz, and office, our approach performs on par with state-of-the-art methods. For larger scenes like room and sequences with more motion blur such as desk2, benefiting from the strength of our efficient loop closure mechanism, our method achieves the best performance. Compared to classical methods, 2DGS-SLAM demonstrates superior performance over dense fusion approaches such as Kintinuous and ElasticFusion, but still falls slightly short of ORB-SLAM2. Furthermore, the ScanNet dataset poses additional challenges, as all eight sequences are captured in roomscale or multi-room-scale indoor environments, where robust loop closure becomes critical. As shown in Tab. IV, SLAM methods without explicit loop closure mechanisms, such as Point-SLAM [68], MonoGS [50], and SplaTAM [38], suffer from significantly higher pose estimation errors. Our method ranks second in average trajectory accuracy across all eight sequences, demonstrating the strength of our loop closure strategy. It is worth noting that the best-performing method, GO-SLAM [99], relies heavily on optical flowbased DROID-SLAM [78] for tracking and loop closure. Additionally, the ground-truth trajectories in ScanNet are generated by BundleFusion [11] rather than a high-precision motion capture system, and thus the results on this dataset should be considered as indicative rather than definitive.

## C. 3D Reconstruction Performance

The second set of experiments evaluate the quality of the resulting model. The results support our second claim that our method outperforms 3DGS-based approaches in terms

<!-- image-->  
GO-SLAM [99]

<!-- image-->  
Loopy-SLAM [46]

<!-- image-->  
LoopSplat [102]

<!-- image-->  
2DGS-SLAM (Ours)

Fig. 7: Qualitative comparison of reconstruction results on the ScanNet dataset. The first row presents the reconstructed meshes on sequence 0169. We highlight the map duplication caused by LoopSplatâs pose drift by a red dashed box.The second and third rows show the results on sequence 0233, including both the overall meshes and zoomed-in local views. It can be observed that our 2DGS-SLAM achieves the most globally consistent and smooth reconstruction among all methods.

of surface reconstruction quality and global consistency. We render depth images at keyframe poses using the global Gaussian splat map, followed by TSDF fusion [9] to obtain the final reconstructed mesh. We conduct quantitative evaluations on the Replica dataset, which provides groundtruth meshes for all sequences. Two commonly used metrics, Depth L1 error and F1 score are employed for the evaluation. Depth L1 error measures the difference between the reconstructed and ground-truth meshes by rendering depth images from 1,000 randomly sampled camera poses and computing the per-pixel L1 distance. The F1 score (F 1) evaluates the geometric accuracy of the mesh by jointly considering precision (P ) and recall (R), and is calculated as their harmonic mean: $\begin{array} { r } { F 1 = 2 \frac { P R } { P + R } } \end{array}$ . Here, precision (P ) denotes the percentage of points on the predicted mesh that lie within 1 cm of any point on the ground-truth mesh, while recall (R) measures the percentage of ground-truth points that are similarly close to the predicted mesh. Our evaluation setup is consistent with previous works [46], [68], [102], [103]. We select both Gaussian splatting-based and NeRFstyle volume rendering-based methods as baselines for our quantitative experiments on the Replica dataset.

As shown in Tab. V, in terms of Depth L1 error, our method ranks second, behind NeRF-based method Loopy-SLAM [46], outperforms other Gaussian Splatting-based methods. For the F1-score, our approach comes third, following Loopy-SLAM and LoopSplat [102]. It is worth noting that, during depth rendering, Loopy-SLAM requires ground-truth depth to guide its sampling process. While this contributes to its high accuracy on synthetic data, it limits the methodâs applicability in real-world scenarios where depth measurements are noisy. On the other hand, LoopSplat does not maintain a global map representation. Instead, it continuously generates local submaps during operation. For mesh reconstruction, LoopSplat renders depth images from these submaps, typically dozens per scene, and performs TSDF fusion. Since each submap only covers a limited range of viewpoints, the rendered depths tend to closely resemble the original input depth images. While this approach performs well on synthetic data, it often leads to excessive artifacts and map inconsistencies in real-world environments due to the lack of global information fusion. Moreover, maintaining a large number of overlapping submaps significantly increases memory consumption and complicates downstream robotic tasks such as planning. Fig. 7 shows our qualitative results on the ScanNet dataset. We can observe that, in realworld environments with challenging lighting conditions and noisy depth measurements, Loopy-SLAM, which performs best on synthetic datasets, produces noticeably coarse mesh reconstructions. Similarly, LoopSplat suffers from issues such as more artifacts and map inconsistencies. In contrast, our approach demonstrates superior global consistency and produces smoother surface reconstructions in real-world datasets.

TABLE V: Reconstruction comparison on the Replica dataset. We highlight the best results in bold and the second best results are underscored. \* indicates methods that use ground-truth depth for sampling.
<table><tr><td>Method</td><td>Map Representation</td><td>Metric</td><td>Rm 0</td><td>Rm 1</td><td>Rm 2</td><td>Off0</td><td>Off1</td><td>Off2</td><td>Off3</td><td>Off4</td><td>Avg.</td></tr><tr><td>NICE-SLAM [103]</td><td>feature grids</td><td>Depth L1[cm]â FF [(%]</td><td>1.81 45.0</td><td>1.44 44.8</td><td>2.04 43.6</td><td>1.39 50.0</td><td>1.76 51.9</td><td>8.33 39.2</td><td>4.99 39.9</td><td>2.01 36.5</td><td>2.97 43.9</td></tr><tr><td>E-SLAM [37]</td><td>feature planes</td><td>Depth L1[cm]â PF1 [%}]</td><td>0.97 81.0</td><td>1.07 82.2</td><td>1.28 83.9</td><td>0.86 78.4</td><td>1.26 75.5</td><td>1.71 77.1</td><td>1.43 75.5</td><td>1.06 79.1</td><td>1.18 79.1</td></tr><tr><td>Loopy-SLAM* [46]</td><td>feature points</td><td>Depth L1[cm]â F1 [%]â</td><td>0.30 91.6</td><td>0.20 92.4</td><td>0.42 90.6</td><td>0.23 93.9</td><td>0.46 91.6</td><td>0.60 88.5</td><td>0.37 89.0</td><td>0.24 88.7</td><td>0.35 90.8</td></tr><tr><td>SplaTAM [38]</td><td>3DGS</td><td>Depth L1[cm]â F1 [%]â</td><td>0.43 89.3</td><td>0.38 88.2</td><td>0.54 88.0</td><td>0.44 91.7</td><td>0.66 90.0</td><td>1.05 85.1</td><td>1.60 77.1</td><td>0.68 80.1</td><td>0.72 86.1</td></tr><tr><td>Gaussian- SLAM [96]</td><td>3DGS</td><td>Depth L1[cm] â F1 [%] â</td><td>0.61 88.8</td><td>0.25 91.4</td><td>0.54 90.5</td><td>0.50 911.7</td><td>0.52 90.1</td><td>0.98 87.3</td><td>1.63 84.2</td><td>0.42 87.4</td><td>0.68 88.9</td></tr><tr><td>LoopSplat [102]</td><td>3DGS</td><td>Depth L1[cm]â PF [(%]</td><td>0.39 90.6</td><td>0.23 91.9</td><td>0.52 91.1</td><td>0.32 93.3</td><td>0.51 90.4</td><td>0.63 88.9</td><td>1.09 88.7</td><td>0.40 88.3</td><td>0.51 90.4</td></tr><tr><td>2DGS-SLAM (ours)</td><td>2DGS</td><td>Depth L1[cm] â F1 [%] â</td><td>0.34 90.8</td><td>0.21 91.6</td><td>0.43 90.6</td><td>0.27 93.1</td><td>0.41 90.1</td><td>1.08 87.0</td><td>0.67 87.6</td><td>0.28 87.5</td><td>0.46 889.7</td></tr></table>

TABLE VI: Rendering performance comparison on the Replica dataset. We report three metrics: PSNR [dB], SSIM, and LPIPS. The best results are highlighted in bold, and the second best results are underscored.
<table><tr><td>Method</td><td>Metric</td><td>Rm 0</td><td>Rm 1</td><td>Rm 2</td><td>Off0</td><td>Off1</td><td>Off2</td><td>Off3</td><td>Off4</td><td>Avg.</td></tr><tr><td rowspan="3">Point-SLAM [68]</td><td>PSNR â</td><td>32.40</td><td>34.08</td><td>35.50</td><td>38.26</td><td>39.16</td><td>33.99</td><td>33.48</td><td>33.49</td><td>35.17</td></tr><tr><td>SSIM â</td><td>0.974</td><td>0.977</td><td>0.982</td><td>0.983</td><td>0.986</td><td>0.960</td><td>0.960</td><td>0.979</td><td>0.975</td></tr><tr><td>LPIPS â</td><td>0.113</td><td>0.116</td><td>0.110</td><td>0.118</td><td>0.156</td><td>0.132</td><td>0.142</td><td>0.124</td><td>0.126</td></tr><tr><td rowspan="3">SplaTAM [38]</td><td>PSNR â</td><td>32.86</td><td>33.89</td><td>35.25</td><td>38.26</td><td>39.17</td><td>31.97</td><td>29.70</td><td>31.81</td><td>34.11</td></tr><tr><td>SSIM â</td><td>0.980</td><td>0.970</td><td>0.980</td><td>0.980</td><td>0.980</td><td>0.970</td><td>0.950</td><td>0.970</td><td>0.970</td></tr><tr><td>LPIPS </td><td>0.070</td><td>0.100</td><td>0.080</td><td>0.090</td><td>0.090</td><td>0.090</td><td>0.120</td><td>0.150</td><td>0.100</td></tr><tr><td rowspan="3">MonoGS [50]</td><td>PSNR â</td><td>34.83</td><td>36.43</td><td>37.49</td><td>39.50</td><td>42.09</td><td>36.24</td><td>36.70</td><td>36.07</td><td>37.50</td></tr><tr><td>SSIM â</td><td>0.954</td><td>00.959</td><td>0.965</td><td>0.971</td><td>0.977</td><td>0.964</td><td>0.963</td><td>0.957</td><td>0.960</td></tr><tr><td>LPIPS â</td><td>0.068</td><td>0.076</td><td>0.075</td><td>0.072</td><td>0.055</td><td>0.078</td><td>0.065</td><td>0.099</td><td>0.070</td></tr><tr><td rowspan="3">LoopSplat [102]</td><td>PSNR â</td><td>33.07</td><td>35.32</td><td>36.16</td><td>40.82</td><td>40.21</td><td>34.67</td><td>35.67</td><td>37.10</td><td>36.63</td></tr><tr><td>SSIM</td><td>0.973</td><td>0.978</td><td>0.985</td><td>0.992</td><td>0.990</td><td>0.985</td><td>0.990</td><td>0.989</td><td>0.985</td></tr><tr><td>LPIPS â</td><td>0.116</td><td>0.122</td><td>0.111</td><td>0.085</td><td>0.123</td><td>0.140</td><td>0.096</td><td>0.106</td><td>0.112</td></tr><tr><td rowspan="3">2DGS-SLAM (ours)</td><td>PSNR â</td><td>35.63</td><td>37.09</td><td>38.47</td><td>43.14</td><td>42.39</td><td>36.33</td><td>36.16</td><td>38.8</td><td>38.50</td></tr><tr><td>SSIM â</td><td>0.965</td><td>0.968</td><td>0.973</td><td>0.985</td><td>0.980</td><td>0.968</td><td>0.966</td><td>0.971</td><td>0.972</td></tr><tr><td>LPIPS â</td><td>0.044</td><td>0.048</td><td>0.05</td><td>0.029</td><td>0.046</td><td>0.049</td><td>0.046</td><td>0.049</td><td>0.045</td></tr></table>

## D. Rendering Quality

The next set of experiments is designed to evaluate the rendering quality of our method. The results support the second part of our second claim, i.e., our approach enables highfidelity rendering suitable for online robotic applications. We evaluate the rendering quality of our method by computing the differences between the rendered images at all training views and their corresponding input images. The evaluation metrics include peak signal-to-noise ratio (PSNR), structural similarity (SSIM) [87], and learned perceptual image patch similarity (LPIPS) [98]. For baselines, we select the state-ofthe-art NeRF-based methods, Point-SLAM [68], as well as Gaussian splatting based method, including SplaTAM [38], MonoGS [50], and LoopSplat [102]. We conduct quantitative evaluations on the Replica and ScanNet datasets. As shown in Tab. VI, 2DGS-SLAM achieves the best PSNR and LPIPS scores on the Replica dataset, with its SSIM score also being on par with other Gaussian splatting-based methods. On the real-world ScanNet dataset, our method ranks second in average metric scores, with PSNR and SSIM worse than LoopSplat. However, it is important to note that LoopSplat employs complex post-processing to merge its submaps. Specifically, after completing pose estimation for all frames, LoopSplat first performs TSDF fusion using depth images rendered from different submaps to obtain the global mesh, then initializes a new set of Gaussian splats from the vertices of resulting mesh and optimizes them using all RGB-D keyframes for 30,000 iterations to generate a global radiance field. To isolate the impact of post-processing, we report extra rendering results of our method, MonoGS and LoopSplat without any map refinement on Tab. VIII. Since LoopSplat stores sub-maps instead of a unified global map, we directly merged its sub-maps using their respective poses to construct a global Gaussian splat map. The results show that our method outperforms the baselines in terms of PSNR, SSIM, and LPIPS, and maintains competitive performance compared to the results obtained with map refinement. Due to the severe pose drift, which can be seen in Tab. IV, MonoGS struggles to reconstruct a reliable radiance field for larger scenes online. Meanwhile, LoopSplat does not maintain a globally consistent map, leading to significant artifacts in the accumulated Gaussian splat submaps and making it unsuitable for high-quality rendering required by online robotic applications.

TABLE VII: Rendering performance comparison on the ScanNet dataset. We report three metrics: PSNR [dB], SSIM, and LPIPS. The best results are highlighted in bold, and the second best results are underscored.
<table><tr><td>Method</td><td> Metric</td><td>|0000</td><td>0059</td><td>0106</td><td>0169</td><td>0181</td><td>0207</td><td>Avg.</td></tr><tr><td rowspan="3">NICE-SLAM [103]</td><td>PSNR â</td><td>18.71</td><td>16.55</td><td>17.29</td><td>18.75</td><td>15.56</td><td>18.38</td><td>17.54</td></tr><tr><td>SSIM</td><td>0.641</td><td>0.605</td><td>0.646</td><td>0.629</td><td>0.562</td><td>0.646</td><td>0.621</td></tr><tr><td>LPPIPS </td><td>0.561</td><td>0.534</td><td>0.510</td><td>0.534</td><td>0.602</td><td>0.552</td><td>0.548</td></tr><tr><td rowspan="3">Point-SLAM [68]</td><td>PSNR â</td><td>19.06</td><td>16.38</td><td>18.46</td><td>18.69</td><td>16.75</td><td>19.66</td><td>18.17</td></tr><tr><td>SSIM â</td><td>0.662</td><td>0.615</td><td>0.753</td><td>0.650</td><td>0.666</td><td>0.696</td><td>0.673</td></tr><tr><td>LPIPSâ</td><td>0.515</td><td>0.528</td><td>0.439</td><td>0.513</td><td>0.532</td><td>0.500</td><td>0.504</td></tr><tr><td rowspan="3">SplaTAM [38]</td><td>PSNR â</td><td>19.33</td><td>19.27</td><td>17.73</td><td>21.97</td><td>16.76</td><td>19.8</td><td>19.14</td></tr><tr><td>SSIM â</td><td>0.660</td><td>0.792</td><td>0.690</td><td>0.776</td><td>0.683</td><td>0.696</td><td>0.716</td></tr><tr><td>LPIPSâ</td><td>0.438</td><td>0.289</td><td>0.376</td><td>0.281</td><td>0.420</td><td>0.341</td><td>0.358</td></tr><tr><td rowspan="3">MonoGS [50]</td><td>PSNR â</td><td>21.13</td><td>19.70</td><td>21.35</td><td>22.44</td><td>22.02</td><td>20.95</td><td>21.26</td></tr><tr><td>SSIM â</td><td>0.723</td><td>0.722</td><td>0.808</td><td>0.781</td><td>0.814</td><td>0.725</td><td>0.762</td></tr><tr><td>LPIPS â</td><td>0.448</td><td>0.436</td><td>0.339</td><td>0.362</td><td>0.432</td><td>0.459</td><td>0.412</td></tr><tr><td rowspan="3">LoopSplat [102]</td><td>PSNR â</td><td>24.99</td><td>23.23</td><td>23.35</td><td>26.80</td><td>24.82</td><td>26.33</td><td>|24.92</td></tr><tr><td>SSIM â</td><td>0.840</td><td>0.831</td><td>0.846</td><td>0.877</td><td>0.824</td><td>0.854</td><td>0.845</td></tr><tr><td>LPIPS â</td><td>0.450</td><td>0.400</td><td>0.409</td><td>0.346</td><td>0.514</td><td>0.430</td><td>0.425</td></tr><tr><td rowspan="3">2DGS-SLAM</td><td>PSNR â</td><td>23.36</td><td>19.00</td><td>20.53</td><td>24.67</td><td>21.27</td><td>23.71</td><td>22.09</td></tr><tr><td>SSIM â</td><td>0.767</td><td>0.729</td><td>0.795</td><td>0.796</td><td>0.821</td><td>0.779</td><td>0.781</td></tr><tr><td>LPIPS â</td><td>0.440</td><td>0.444</td><td>0.357</td><td>0.362</td><td>0.485</td><td>0.425</td><td>0.418</td></tr></table>

TABLE VIII: Rendering performance comparison on the ScanNet dataset. All the reported results are evaluated from the raw Gaussians splatting map without any refinement. We report three metrics: PSNR [dB], SSIM, and LPIPS. The best results are highlighted in bold, and the second best results are underscored.
<table><tr><td>Method</td><td>Metric</td><td>0000</td><td>0059</td><td>0106</td><td>0169</td><td>0181</td><td>0207</td><td>Avg.</td></tr><tr><td rowspan="3">MonoGS [50]</td><td>PSNR â</td><td>15.40</td><td>15.98</td><td>18.34</td><td>18.75</td><td>15.43</td><td>16.34</td><td>16.70</td></tr><tr><td>SSIM â</td><td>0.597</td><td>00.591</td><td>0.701</td><td>0.683</td><td>0.642</td><td>00.651</td><td>00.644</td></tr><tr><td>LPIPS â</td><td>0.646</td><td>0.591</td><td>0.500</td><td>0.525</td><td>0.577</td><td>0.577</td><td>0.569</td></tr><tr><td rowspan="3">LoopSplat [102]</td><td>PSNR â</td><td>12.35</td><td>12.95</td><td>10.26</td><td>10.86</td><td>11.47</td><td>13.17</td><td>11.84</td></tr><tr><td>SSIM â</td><td>0.413</td><td>0.411</td><td>0.318</td><td>0.495</td><td>0.541</td><td>0.504</td><td>0.447</td></tr><tr><td>LPIIS </td><td>0.840</td><td>0.724</td><td>0.798</td><td>0.791</td><td>0.698</td><td>0.704</td><td>0.759</td></tr><tr><td rowspan="3">2DGS-SLAM</td><td>PSNR â</td><td>21.95</td><td>16.16</td><td>17.71</td><td>22.72</td><td>19.74</td><td>22.00</td><td>|20.05</td></tr><tr><td>SSSIM</td><td>0.740</td><td>0.639</td><td>0.710</td><td>0.763</td><td>0.793</td><td>0.744</td><td>0.731</td></tr><tr><td>LPIPS â</td><td>0.453</td><td>0.501</td><td>0.456</td><td>0.392</td><td>0.464</td><td>0.435</td><td>0.450</td></tr></table>

This observation is also supported by qualitative results on TUM dataset, as illustrated in the Fig. 8. Here, we compare the Gaussian splat maps obtained directly from each method after pose estimation, without any post-processing applied to the maps. As shown, LoopSplatâs rendering results suffer from severe artifacts. Moreover, the normal renderings reveal that due to inconsistencies in 3DGS-based depth rendering, LoopSplat and MonoGS fail to produce smooth surface reconstructions. In comparison, our method not only achieves high-fidelity RGB renderings but also accurately reconstructs scene geometry. While SplaTAM achieves comparable reconstruction quality, it requires a much larger number of Gaussian splats than our approach. We provide a detailed comparison of memory and time consumption in the next section.

## E. Runtime and Memory Evaluation

The following experiment and results support the claim that our approach is more efficient in terms of runtime and produces a more compact map compared to the baselines. To compare the performance of different methods, we evaluate frames per second (FPS), calculated as the total number of frames in the sequence divided by the total time, as well as the memory usage of the map without post-processing and peak GPU memory consumption on the ScanNet sequence scene0000, which contains a total of 5,578 frames. We selected main baselines from the previous experiments, including rendering-based methods such as Point-SLAM [68], Loopy-SLAM [46], MonoGS [50], SplaTAM [38], and LoopSplat [102], for comparison. As shown in Tab. IX, our method is only slower than MonoGS in terms of FPS. This is expected, as our approach involves additional tasks such as image feature extraction, loop closure detection, relocalization, and map updates, which are not exist in MonoGS as it does not incorporate loop closure. In comparison with other methods that do support loop closure, such as Loopy-SLAM [46] and LoopSplat [102], our approach demonstrates significantly higher time efficiency, achieving a 6-7Ã speedup.

TABLE IX: Statistics of runtime and memory. We report three metrics: FPS, map size (MB), and peak GPU memory (MB). LC denotes that loop closure is enabled. The best results are highlighted in bold, and the second best results are underscored.
<table><tr><td>Method</td><td>| LC |</td><td></td><td></td><td>| FPS (Hz) â | Map size (MB) â | GPU Memory (MB) â</td></tr><tr><td>Point-SLAM [68]</td><td></td><td>0.05</td><td>99.4</td><td>8236</td></tr><tr><td>Loopy-SLAM [46]</td><td>*&gt;**&gt;&gt;</td><td>0.13</td><td>195.3</td><td>12475</td></tr><tr><td>MonoGS [50]</td><td></td><td>1.92</td><td>13.2</td><td>9062</td></tr><tr><td>SplaTAM [38]</td><td></td><td>0.18</td><td>213.1</td><td>12939</td></tr><tr><td>LoopSplat [102]</td><td></td><td>0.17</td><td>4608</td><td>9616</td></tr><tr><td>2DGS-SLAM (ours)</td><td></td><td>0.92</td><td>9.7</td><td>10822</td></tr></table>

Additionally, and thanks to our efficient map management mechanism, our final map has the smallest memory footprint, suggesting that the number of redundant Gaussian splats in our system is much lower than in other Gaussian splattingbased methods. In contrast, due to the lack of removing redundant Gaussian splats, SplaTAMâs map memory usage is more than 20 times higher than ours. The continuously accumulating redundant splats also lead to a decrease in its pose estimation efficiency over time. Furthermore, since LoopSplat stores overlapping submaps rather than maintaining a global map, its memory usage for map storage is very high due to the accumulation of redundant splats. In terms of peak GPU memory consumption, our method is slightly higher than LoopSplat. However, this is because LoopSplat offloads all submaps to disk in order to minimize runtime memory usage. Unfortunately, the frequent disk I/O and CPU-GPU data transfers significantly slow down its speed compared to ours. In summary, when compared to other rendering-based SLAM methods with loop closure support, 2DGS-SLAM outperforms the baselines in both memory usage and runtime efficiency.

## F. Experiments on Self-recorded Robot Data

To evaluate the effectiveness of our method in real-world robotic applications beyond publicly available datasets, we also collected data using a wheeled mobile robot equipped with Intel RealSense D455 RGB-D cameras in indoor environments. The experimental scenes include (1) corridor, a 20-meter-long straight corridor used to evaluate the robustness of our pose estimation in low-texture, repetitive environments; (2,3) kitchen and office, two rooms measuring approximately 7 m Ã 6 m, where the robot performs challenging maneuvers such as rapid pure rotations during recording. As shown in Fig. 9a, we use AprilTags mounted on the ceiling to compute near ground-truth poses with approximately 1 cm global accuracy for evaluation. It is also worth noting that, compared to the structured-light-based RGB-D cameras used in datasets such as ScanNet [11] and TUM-RGBD [74], the stereo-vision-based RealSense D455 typically produces noisier depth images.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
LoopSplat [102]

<!-- image-->  
MonoGS [50]

<!-- image-->  
SplaTAM [38]

<!-- image-->  
2DGS-SLAM (Ours)  
Fig. 8: Qualitative comparison of Rendering results on the TUM dataset. For a fair comparison, we selected non-training views and used the raw Gaussian splatting maps from each method without any map refinement. The first and second rows show comparisons on sequence fr2_xyz, including both RGB and normal renderings. As 3DGS does not support direct normal rendering, we compute normal images from the rendered depth images using Eq. (23) for visualization. The third row shows results on sequence fr3_office. Our method achieves the most photorealistic RGB renderings and the smoothest normal image.

As illustrated in Fig. 9b, our method achieves high-quality scene reconstruction, demonstrating not only a high-fidelity radiance field but also smooth surface normal rendering. We further conducted quantitative pose estimation experiments, comparing our method with the main baselines evaluated in the aforementioned public datasets. As shown in Tab. X, our method yields substantially lower average trajectory error than all baseline methods, highlighting its robustness to depth noise and rapid camera motion. Moreover, our method successfully performs loop closures on both the kitchen and office sequences, significantly reducing pose drift compared to competing methods. Consistent with the observations in experiment IV-B, methods that lack loop closure support, such as Point-SLAM [68], MonoGS [50], and SplaTAM [38], suffer from severe pose drift, making them unsuitable for room-scale reconstruction and realworld mobile robot applications. compared with renderingbased methods with loop closure capability, including Loopy-

TABLE X: Absolute trajectory error (ATE) on the self-recorded dataset (cm). LC denotes that loop closure is enabled. The best results are highlighted in bold, and the second best results are underscored.
<table><tr><td>Method</td><td>| LC</td><td>corridor</td><td>kitchen</td><td>office</td><td>Avg.</td></tr><tr><td>Point-SLAM [68]</td><td></td><td>30.8</td><td>15.9</td><td>23.9</td><td>23.5</td></tr><tr><td>Loopy-SLAM [46]</td><td>Ã&gt;</td><td>Failed</td><td>63.9</td><td>7.9</td><td>-</td></tr><tr><td>MonoGS [50]</td><td>Ã</td><td>9.5</td><td>17.6</td><td>13.9</td><td>13.6</td></tr><tr><td>SplaTAM [38]</td><td></td><td>29.8</td><td>130.5</td><td>16.7</td><td>59.0</td></tr><tr><td>LoopSplat [102]</td><td>Ã&gt;</td><td>2.1</td><td>10.1</td><td>22.6</td><td>11.6</td></tr><tr><td>2DGS-SLAM (ours)</td><td>â</td><td>3.4</td><td>6.8</td><td>4.7</td><td>5.0</td></tr></table>

SLAM [46] and LoopSplat [102], our approach demonstrates superior robustness in both motion estimation and loop closure, highlighting the practical value of our method in robotic applications.

## V. CONCLUSION

In this paper, we proposed 2DGS-SLAM, a novel RGB-D SLAM framework that enables globally consistent radiance field reconstruction based on 2D Gaussian splatting. Taking advantage of the consistent depth rendering of 2D Gaussian splatting, we propose an accurate camera tracking framework. We further introduced an efficient map management strategy and integrated a strong 3D foundation model MASt3R to enable robust loop closure detection and relocalization. We implemented and evaluated our approach on different datasets and provided comparisons to other existing techniques and supported all claims made in this paper. The results demonstrate that our method achieves superior pose estimation accuracy compared to other renderingbased approaches, while delivering comparable or even better surface reconstruction quality. Moreover, our 2DGS-SLAM consistently outperforms 3D Gaussian splatting-based systems in terms of surface smoothness and global consistency. At the same time, our method maintains competitive image rendering quality with significantly improved efficiency compared with other rendering based method with loop closure support.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Kitchen

<!-- image-->  
Corridor  
Fig. 9: (a) The wheeled robot platform used in our experiments and AprilTags mounted on the ceiling. We use the fisheye camera installed on the robot to detect AprilTags for pose evaluation. (b) Reconstructed Gaussian splat maps and camera trajectories of our method on three experimental scenes: office, kitchen, and corridor. For the corridor, we demonstrate zoomed-in views of both RGB and normal renderings. No map refinement was applied after tracking.

## VI. ACKNOWLEDGEMENTS

We thank Haofei Kuang and Niklas Trekel for providing the real-world robot-collected datasets along with groundtruth poses, and Liyuan Zhu for sharing the baseline results.

## REFERENCES

[1] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and J. Sivic. NetVLAD: CNN Architecture for Weakly Supervised Place Recognition. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2016.

[2] R. Arandjelovic and A. Zisserman. All About VLAD. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2013.

[3] D. Azinovic, R. Martin-Brualla, D.B. Goldman, M. NieÃner, and Â´ J. Thies. Neural RGB-D Surface Reconstruction. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2022.

[4] L. Bartolomei, L. Teixeira, and M. Chli. Fast Multi-UAV Decentralized Exploration of Forests. IEEE Robotics and Automation Letters (RA-L), 8(9):5576â5583, 2023.

[5] J. Behley and C. Stachniss. Efficient Surfel-Based SLAM using 3D Laser Range Data in Urban Environments. In Proc. of Robotics: Science and Systems (RSS), 2018.

[6] J.L. Blanco-Claraco. A flexible framework for accurate lidar odometry, map manipulation, and localization. Intl. Journal of Robotics Research (IJRR), 0(0):02783649251316881, 2025.

[7] C. Campos, R. Elvira, J.J.G. RodrÂ´Ä±guez, J.M. Montiel, and J.D. Tardos. Orb-slam3: An accurate open-source library for visual, Â´ visualâinertial, and multimap slam. IEEE Trans. on Robotics (TRO), 37(6):1874â1890, 2021.

[8] X. Chen, T. Labe, A. Milioto, T. R Â¨ ohling, O. Vysotska, A. Haag, Â¨ J. Behley, and C. Stachniss. OverlapNet: Loop Closing for LiDARbased SLAM. In Proc. of Robotics: Science and Systems (RSS), 2020.

[9] B. Curless and M. Levoy. A Volumetric Method for Building Complex Models from Range Images. In Proc. of the Intl. Conf. on Computer Graphics and Interactive Techniques (SIGGRAPH), 1996.

[10] A. Dai, A. Chang, M. Savva, M. Halber, T. Funkhouser, and M. NieÃner. ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2017.

[11] A. Dai, M. NieÃner, M. Zollhofer, S. Izadi, and C. Theobalt. Â¨ BundleFusion: Real-time Globally Consistent 3D Reconstruction using Online Surface Re-integration. ACM Trans. on Graphics (TOG), 36(3):1â18, 2017.

[12] P. Dai, J. Xu, W. Xie, X. Liu, H. Wang, and W. Xu. Highquality Surface Reconstruction using Gaussian Surfels. In Proc. of the Intl. Conf. on Computer Graphics and Interactive Techniques (SIGGRAPH), 2024.

[13] B. Della Corte, I. Bogoslavskyi, C. Stachniss, and G. Grisetti. A General Framework for Flexible Multi-Cue Photometric Point Cloud Registration. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2018.

[14] F. Dellaert. Factor graphs and GTSAM: A hands-on introduction. Georgia Institute of Technology, Tech. Rep, 2:4, 2012.

[15] T. Deng, G. Shen, T. Qin, J. Wang, W. Zhao, J. Wang, D. Wang, and W. Chen. PLGSLAM: Progressive Neural Scene Represenation with Local to Global Bundle Adjustment. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[16] L. Di Giammarino, L. Brizi, T. Guadagnino, C. Stachniss, and G. Grisetti. Md-slam: Multi-cue direct slam. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2022.

[17] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In Proc. of the Intl. Conf. on Learning Representations (ICLR), 2021.

[18] B. Duisterhof, Z. Lojze, W. Philippe, L. Vincent, C. Yohann, and R. Jerome. MASt3R-SfM: A Fully-Integrated Solution for Unconstrained Structure-from-Motion. In Proc. of the Intl. Conf. on 3D Vision (3DV), 2025.

[19] F. Endres, J. Hess, J. Sturm, D. Cremers, and W. Burgard. 3D Mapping with an RGB-D Camera. IEEE Trans. on Robotics (TRO), 30(1):177â187, 2014.

[20] C. Forster, M. Pizzoli, and D. Scaramuzza. SVO: Fast semi-direct monocular visual odometry. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2014.

[21] D. Galvez-Lopez and J.D. Tard Â´ os. Bags of Binary Words for Fast Â´ Place Recognition in Image Sequences. IEEE Trans. on Robotics (TRO), 28(5):1188â1197, 2012.

[22] E. Giacomini, L. Di Giammarino, L.D. Rebott, G. Grisetti, and M.R. Oswald. Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping. arXiv preprint, arXiv:2503.17491, 2025.

[23] B. Glocker, J. Shotton, A. Criminisi, and S. Izadi. Real-time RGB-D camera relocalization via randomized ferns for keyframe encoding. IEEE Trans. on Visualization and Computer Graphics, 21(5):571â 583, 2014.

[24] A. Glover, W. Maddern, M. Warren, S. Reid, M. Milford, and G. Wyeth. Openfabmap: An open source toolbox for appearance-

based loop closure detection. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2012.

[25] T. Guadagnino, B. Mersch, S. Gupta, I. Vizzo, G. Grisetti, and C. Stachniss. KISS-SLAM: A Simple, Robust, and Accurate 3D LiDAR SLAM System With Enhanced Generalization Capabilities. arXiv preprint, arXiv:2503.12660, 2025.

[26] S. Ha, J. Yeon, and H. Yu. RGBD GS-ICP SLAM. In Proc. of the Europ. Conf. on Computer Vision (ECCV), 2024.

[27] A. Hornung, K. Wurm, M. Bennewitz, C. Stachniss, and W. Burgard. OctoMap: An Efficient Probabilistic 3D Mapping Framework Based on Octrees. Autonomous Robots, 34(3):189â206, 2013.

[28] J. Hu, M. Mao, H. Bao, G. Zhang, and Z. Cui. CP-SLAM: Collaborative Neural Point-based SLAM System. In Proc. of the Conf. on Neural Information Processing Systems (NeurIPS), 2023.

[29] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In Proc. of the Intl. Conf. on Computer Graphics and Interactive Techniques (SIGGRAPH), 2024.

[30] C. Huang, O. Mees, A. Zeng, and W. Burgard. Audio visual language maps for robot navigation. In Proc. of the Intl. Symp. on Experimental Robotics (ISER), 2023.

[31] H. Huang, L. Li, C. Hui, and S.K. Yeung. Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[32] Y. Huang, B. Cui, L. Bai, Z. Chen, J. Wu, Z. Li, H. Liu, and H. Ren. Advancing Dense Endoscopic Reconstruction with Gaussian Splatting-driven Surface Normal-aware Tracking and Mapping. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2025.

[33] S. Izquierdo and J. Civera. Close, But Not There: Boosting Geographic Distance Sensitivity in Visual Place Recognition. In Proc. of the Europ. Conf. on Computer Vision (ECCV), 2024.

[34] S. Izquierdo and J. Civera. Optimal Transport Aggregation for Visual Place Recognition. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[35] W. Jiaxin and S. Leutenegger. GSFusion: Online RGB-D Mapping Where Gaussian Splatting Meets TSDF Fusion. IEEE Robotics and Automation Letters (RA-L), 9(12):11865â11872, 2024.

[36] L. Jin, X. Zhong, Y. Pan, J. Behley, C. Stachniss, and M. Popovic. ActiveGS: Active Scene Reconstruction using Gaussian Splatting. IEEE Robotics and Automation Letters (RA-L), 10(5):4866â4873, 2025.

[37] M.M. Johari, C. Carta, and F. Fleuret. ESLAM: Efficient Dense SLAM System Based on Hybrid Representation of Signed Distance Fields. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2023.

[38] N. Keetha, J. Karhade, K.M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten. SplaTAM: Splat Track & Map 3D Gaussians for Dense RGB-D SLAM. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[39] M. Keller, D. Lefloch, M. Lambers, and S. Izadi. Real-time 3D Reconstruction in Dynamic Scenes using Point-based Fusion. In Proc. of the Intl. Conf. on 3D Vision (3DV), 2013.

[40] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis. 3D Gaussian Â¨ Splatting for Real-Time Radiance Field Rendering. ACM Trans. on Graphics (TOG), 42(4):1â14, 2023.

[41] C. Kerl, J. Sturm, and D. Cremers. Robust Odometry Estimation for RGB-D Cameras. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2013.

[42] C. Kerl, J. Sturm, and D. Cremers. Dense visual slam for rgb-d cameras. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2013.

[43] M. Labbe and F. Michaud. RTAB-Map: An open-source lidar and visual simultaneous localization and mapping library for large-scale and long-term online operation. Journal of Field Robotics (JFR), 36(1):416â446, 2019.

[44] V. Leroy, Y. Cabon, and J. Revaud. Grounding Image Matching in 3D with MASt3R. In Proc. of the Europ. Conf. on Computer Vision (ECCV), 2024.

[45] T.Y. Lim, B. Sun, M. Pollefeys, and H. Blum. Loop Closure from Two Views: Revisiting PGO for Scalable Trajectory Estimation through Monocular Priors. arXiv preprint, arXiv:2503.16275, 2025.

[46] L. Liso, E. Sandstrom, V. Yugay, L. Van Gool, and M.R. Oswald. Â¨ Loopy-slam: Dense neural slam with loop closures. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[47] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In Proc. of the Intl. Conf. on Learning Representations (ICLR), 2019.

[48] D. Maier, A. Hornung, and M. Bennewitz. Real-time navigation in 3D environments based on depth camera data. In Proc. of the IEEE Intl. Conf. on Humanoid Robots, 2012.

[49] Y. Mao, X. Yu, K. Wang, Y. Wang, R. Xiong, and Y. Liao. NGEL-SLAM: Neural Implicit Representation-based Global Consistent Low-Latency SLAM System. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2024.

[50] H. Matsuki, R. Murai, P.H. Kelly, and A.J. Davison. Gaussian splatting SLAM. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[51] Z. Matthias, R. Jussi, B. Mario, D. Carsten, and P. Mark. Perspective accurate splatting. In Proc. of Graphics Interface (GI), 2004.

[52] B. Mildenhall, P. Srinivasan, M. Tancik, J. Barron, R. Ramamoorthi, and R. Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In Proc. of the Europ. Conf. on Computer Vision (ECCV), 2020.

[53] M. Milford and G. Wyeth. SeqSLAM: Visual route-based navigation for sunny summer days and stormy winter nights. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2012.

[54] R. Mur-Artal and J. Tardos. ORB-SLAM2: An Open-Source SLAM Â´ System for Monocular, Stereo, and RGB-D Cameras. IEEE Trans. on Robotics (TRO), 33(5):1255â1262, 2017.

[55] R. Murai, E. Dexheimer, and A.J. Davison. MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.

[56] R.A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A.J. Davison, P. Kohli, J. Shotton, S. Hodges, and A. Fitzgibbon. KinectFusion: Real-Time Dense Surface Mapping and Tracking. In Proc. of the Intl. Symp. on Mixed and Augmented Reality (ISMAR), 2011.

[57] M. Oquab, T. Darcet, T. Moutakanni, H.V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.Y. Huang, H. Xu, V. Sharma, S.W. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski. DINOv2: Learning Robust Visual Features without Supervision. Trans. on Machine Learning Research (TMLR), pages 1â31, 2024.

[58] E. Palazzolo, J. Behley, P. Lottes, P. Giguere, and C. Stachniss. ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2019.

[59] Y. Pan, P. Xiao, Y. He, Z. Shao, and Z. Li. MULLS: Versatile LiDAR SLAM Via Multi-Metric Linear Least Square. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2021.

[60] Y. Pan, X. Zhong, L. Jin, L. Wiesmann, M. Popovic, J. Behley, and Â´ C. Stachniss. PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map. In Proc. of Robotics: Science and Systems (RSS), 2025.

[61] Y. Pan, X. Zhong, L. Wiesmann, T. Posewsky, J. Behley, and C. Stachniss. PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency. IEEE Trans. on Robotics (TRO), 40:4045â4064, 2024.

[62] J.J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove. DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2019.

[63] Z. Peng, T. Shao, L. Yong, J. Zhou, Y. Yang, J. Wang, and K. Zhou. RTG-SLAM: Real-time 3D Reconstruction at Scale using Gaussian Splatting. In Proc. of the Intl. Conf. on Computer Graphics and Interactive Techniques (SIGGRAPH), 2024.

[64] V. Reijgwart, A. Millane, H. Oleynikova, R. Siegwart, C. Cadena, and J. Nieto. Voxgraph: Globally consistent, volumetric mapping using signed distance function submaps. IEEE Robotics and Automation Letters (RA-L), 5(1):227â234, 2019.

[65] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski. Orb: an efficient alternative to sift or surf. In Proc. of the IEEE Intl. Conf. on Computer Vision (ICCV), 2011.

[66] R. Rusu, N. Blodow, and M. Beetz. Fast point feature histograms (fpfh) for 3d registration. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2009.

[67] E. Sandstrom, K. Tateno, M. Oechsle, M. Niemeyer, L. Van Gool, Â¨ M.R. Oswald, and F. Tombari. Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians. arXiv preprint, arXiv:2405.16544, 2024.

[68] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald. Point-SLAM: Â¨ Dense Neural Point Cloud-based SLAM. In Proc. of the IEEE/CVF Intl. Conf. on Computer Vision (ICCV), 2023.

[69] T. Schops, T. Sattler, and M. Pollefeys. BAD SLAM: Bundle Adjusted Direct RGB-D SLAM. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2019.

[70] A. Segal, D. Haehnel, and S. Thrun. Generalized-ICP. In Proc. of Robotics: Science and Systems (RSS), 2009.

[71] C. Stachniss. Springer Handbuch der Geodasie Â¨ , chapter Simultaneous Localization and Mapping. Springer Verlag, 2016. In German, invited.

[72] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J.J. Engel, R. Mur-Artal, C. Ren, S. Verma, et al. The Replica dataset: A digital replica of indoor spaces. arXiv preprint, arXiv:1906.05797, 2019.

[73] J. Stuckler and S. Behnke. Multi-Resolution Surfel Maps for Efficient Â¨ Dense 3D Modeling and Tracking. Journal of Visual Communication and Image Representation (JVCIR), 25(1):137â147, 2014.

[74] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. A Benchmark for the Evaluation of RGB-D SLAM Systems. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2012.

[75] E. Sucar, S. Liu, J. Ortiz, and A.J. Davison. imap: Implicit mapping and positioning in real-time. In Proc. of the IEEE/CVF Intl. Conf. on Computer Vision (ICCV), 2021.

[76] S. Sun, M. Mielle, A.J. Lilienthal, and M. Magnusson. High-Fidelity SLAM Using Gaussian Splatting with Rendering-Guided Densification and Regularized Optimization. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2024.

[77] Y. Tang, J. Zhang, Z. Yu, H. Wang, and K. Xu. MIPS-Fusion: Multi-Implicit-Submaps for Scalable and Robust Online Neural RGB-D Reconstruction. ACM Trans. on Graphics (TOG), 42(6):1â14, 2023.

[78] Z. Teed and J. Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. In Proc. of the Conf. on Neural Information Processing Systems (NeurIPS), 2021.

[79] G. Tolias, Y. Avrithis, and H. Jegou. To aggregate or not to aggregate: Â´ Selective match kernels for image search. In Proc. of the IEEE/CVF Intl. Conf. on Computer Vision (ICCV), 2013.

[80] S. Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Trans. on Pattern Analysis and Machine Intelligence (TPAMI), 13(4):376â380, 1991.

[81] L. Vincent, M.N. Francesc, and F. Pascal. EPnP: An Accurate O(n) Solution to the PnP Problem. Intl. Journal of Computer Vision (IJCV), 81:255â166, 2009.

[82] I. Vizzo, T. Guadagnino, J. Behley, and C. Stachniss. VDBFusion: Flexible and Efficient TSDF Integration of Range Sensor Data. Sensors, 22(3):1296, 2022.

[83] I. Vizzo, T. Guadagnino, B. Mersch, L. Wiesmann, J. Behley, and C. Stachniss. KISS-ICP: In Defense of Point-to-Point ICP â Simple, Accurate, and Robust Registration If Done the Right Way. IEEE Robotics and Automation Letters (RA-L), 8(2):1029â1036, 2023.

[84] O. Vysotska and C. Stachniss. Lazy Data Association For Image Sequences Matching Under Substantial Appearance Changes. IEEE Robotics and Automation Letters (RA-L), 1(1):213â220, 2016.

[85] H. Wang, J. Wang, and L. Agapito. Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2023.

[86] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud. DUst3R: Geometric 3D Vision Made Easy. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[87] Z. Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Trans. on Image Processing, 13(4):600â612, 2004.

[88] A. Werby, C. Huang, M. Buchner, A. Valada, and W. Burgard. Â¨ Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. In Proc. of Robotics: Science and Systems (RSS), 2024.

[89] T. Whelan, M. Kaess, M. Fallon, H. Johannsson, J. Leonard, and J. McDonald. Kintinuous: Spatially Extended KinectFusion. In Proc. of the RSS Workshop on RGB-D: Advanced Reasoning with Depth Cameras, 2012.

[90] T. Whelan, S. Leutenegger, R.S. Moreno, B. Glocker, and A. Davison. ElasticFusion: Dense SLAM Without A Pose Graph. In Proc. of Robotics: Science and Systems (RSS), 2015.

[91] K. Wu, Z. Zhang, M. Tie, Z. Ai, Z. Gan, and W. Ding. VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes. arXiv preprint, arXiv:2501.08286, 2025.

[92] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li. GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[93] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang. Vox-Fusion: Dense Tracking and Mapping with Voxel-based Neural Implicit Representation. In Proc. of the Intl. Symp. on Mixed and Augmented Reality (ISMAR), 2022.

[94] S. Yu, C. Cheng, Y. Zhou, X. Yang, and H. Wang. OpenGS-SLAM: RGB-Only Gaussian Splatting SLAM for Unbounded Outdoor Scenes. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2025.

[95] V. Yugay, T. Gevers, and M.R. Oswald. MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2025.

[96] V. Yugay, Y. Li, T. Gevers, and M.R. Oswald. Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting. arXiv preprint, arXiv:2312.10070, 2023.

[97] G. Zhang, E. Sandstrom, Y. Zhang, M. Patel, L. Van Gool, and M.R. Â¨ Oswald. GlORIE-SLAM: Globally Optimized Rgb-only Implicit Encoding Point Cloud SLAM. arXiv preprint, arXiv:2403.19549, 2024.

[98] R. Zhang, P. Isola, A.A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2018.

[99] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi. GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction. In Proc. of the IEEE/CVF Intl. Conf. on Computer Vision (ICCV), 2023.

[100] X. Zhong, Y. Pan, J. Behley, and C. Stachniss. SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit Neural Representations. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2023.

[101] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.H. Yang. DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2024.

[102] L. Zhu, Y. Li, E. Sandstrom, S. Huang, K. Schindler, and I. Armeni. Â¨ LoopSplat: Loop Closure by Registering 3D Gaussian Splats. In Proc. of the Intl. Conf. on 3D Vision (3DV), 2025.

[103] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M.R. Oswald, and M. Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2022.