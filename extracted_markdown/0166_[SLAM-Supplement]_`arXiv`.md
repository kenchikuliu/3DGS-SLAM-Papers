# Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping

Emanuele Giacomini1 Luca Di Giammarino1 Lorenzo De Rebotti1

Giorgio Grisetti1 Martin R. Oswald2

1Sapienza University of Rome 2University of Amsterdam

## Abstract

LiDARs provide accurate geometric measurements, making them valuable for ego-motion estimation and reconstruction tasks. Although its success, managing an accurate and lightweight representation of the environment still poses challenges. Both classic and NeRF-based solutions have to trade off accuracy over memory and processing times. In this work, we build on recent advancements in Gaussian Splatting methods to develop a novel LiDAR odometry and mapping pipeline that exclusively relies on Gaussian primitives for its scene representation. Leveraging spherical projection, we drive the refinement of the primitives uniquely from LiDAR measurements. Experiments show that our approach matches the current registration performance, while achieving SOTA results for mapping tasks with minimal GPU requirements. This efficiency makes it a strong candidate for further exploration and potential adoption in real-time robotics estimation tasks. Open source will be released at https://github.com/rvp-group/ Splat-LOAM.

## 1. Introduction

LiDAR sensors provide accurate spatial measurements of the environment, making them valuable for ego-motion estimation and reconstruction tasks. Since the measurements already capture the 3d structure, many LiDAR Simultaneous Localization And Mapping (SLAM) pipelines do not explicitly refine the underlying point representation [2, 3, 7, 9, 12, 37]. Moreover, a global map can be obtained by directly stacking the measurements together. However, this typically leads to extremely large point clouds that cannot be explicitly used for online applications. Several approaches attempted to use surfels [1, 31] and meshes [32]. Although these approaches manage to simultaneously estimate the sensorâs ego-motion while optimizing the map representation, they result in a trade-off between accuracy, memory usage, and runtime.

The recent advent of NeRF [26] sparked fresh interest in Novel View Synthesis (NVS) tasks. Specifically, given a set of input views and triangulated points (i.e. obtainable via COLMAP [35, 36]), NeRF learns a continuous volumetric scene function. Color and density information is propagated throughout the radiance field by ray-casting pixels from the view cameras and, through a Multi-Layer Perceptron (MLP), the method learns a representation that ensures multi-view consistency. Concurrently, NeRFs inspired the computer vision community to tackle the problem of dense visual SLAM through implicit methods. The pioneering work in this context was iMAP [40], which stores the global appearance and geometry of the scene through a single MLP. Despite its success in driving a new research direction, the method suffered from issues related to the limited capacity of the model, leading to low reconstruction quality and catastrophic forgetting during the exploration of larger areas. These issues were later handled by shifting the paradigm and by moving some of the appearance and geometric information over a hierarchical feature grid [54] or neural point clouds [22, 33]. Similarly, these techniques were employed on LiDAR measurements to provide more accurate and lighter explicit dense representations [8, 16, 30, 39, 52]. However, these methods still require tailored sampling techniques to estimate the underlying Signed Distance Function (SDF) accurately. This bottleneck still poses problems for online execution.

<!-- image-->

<!-- image-->  
GPU Memory (MB)â  
Runtime per Frame (s) â  
Figure 1. Performance overview of Splat-LOAM. F1 score to Active Memory [Mb] and Runtime [s]. The plots provide a quantitative comparison between state-of-the-art mapping pipelines, while PIN-SLAM and ours also perform online odometry.

A recent, explicit alternative to NeRFs is 3D Gaussian Splatting (3DGS) [20]. This approach leverages 3D Gaussian-shaped primitives and a differentiable, tile-based rasterizer to generate an appearance-accurate representation. Furthermore, having no need to model empty areas and no neural components, 3DGS earned a remarkable result in accuracy and training speeds. Additionally, several approaches further enhanced the reconstruction capabilities of this representation [6, 13, 15]. Being fast and accurate, this representation is now sparking interest in dense visual SLAM. Recently, 3D Gaussians were employed in several works, yielding superior results over implicit solutions [24, 47, 53].

One issue with Gaussian Splatting relates to the primitive initialization. In areas where few or no points are provided by SFM, adaptive densification tends to fail, often yielding under-reconstructed regions. LiDAR sensor is quite handy at solving this problem as it provides explicit spatial measurements that can be used to initialize the local representation [14, 46].

However, to our knowledge, no attempt has been made to evaluate the performance of this representation for pure Li-DAR data. This technique could prove interesting for visual NVS initialization and LiDAR mapping as it could produce a lightweight, dense, and consistent representation. These insights led us to the development of Splat-LOAM, the first LiDAR Odometry and Mapping pipeline that only leverages Gaussian primitives as its surface representation. Our system demonstrates results on par with other SOTA pipelines at a fraction of the computational demands, proving as an additional research direction for real-time perception in autonomous systems.

## 2. Related Work

Classic LiDAR Odometry and Mapping. Feature-based methods that leverage specific points or groups of points to perform incremental registration. For instance, [38, 49] track feature points on sharp edges and planar surface patches, enabling high-frequency odometry estimation. On the other hand, Direct methods leverage the whole cloud to perform registration. Specifically, these methods can be categorized based on the subjects of the alignment. Scanto-Scan methods matches subsequent clouds, either explicitly [2, 3, 37] or through neural methods [21, 44], while Scan-to-Model methods match clouds with either a local or a global map. Typically, the map is represented using points [7, 12], surfels [1, 31], meshes [32]. Another explored solution project the measurements onto a spherical projection plane to leverage visual techniques for egomotion estimation [9, 10, 28, 51].

Implicit Methods. Concerning mapping only methods, Zhong et al. [52] proposed the first method for LiDAR implicit mapping that, given a set of point clouds and the corresponding sensor poses, used hierarchical feature grids to estimate the SDF of the scene. Their results demonstrated once again the advantages of such representation for surface reconstruction in terms of accuracy and memory footprint. A similar approach from Song et al. [39] improves the mapping accuracy by introducing SDF normal guided sampling and a hierarchical, voxel-guided sampling strategy for local optimization. Building on these advancements, several LiDAR odometry and mapping techniques were proposed. Deng et al. [8] presented the first implicit LiDAR LOAM system using an octree-based feature representation to encode the sceneâs SDF, used both for tracking and mapping. Similarly, Isaacson et al. [16] proposes a hierarchical feature grid to store the SDF information while using point-to-plane ICP to register new clouds. Pan et al. [30] leverages a neural point cloud representation to ensure a globally consistent estimate. The new clouds are registered using a correspondence-free point-to-implicit model approach. These methods prove that implicit representation can offer SOTA results in accuracy at the cost of potentially high execution times or memory requirements. Although targeting a different problem setting, related are also a series of visual neural SLAM methods with RGB or RGBD input [19, 22, 23, 33, 34, 47, 48, 54], see [42] for a survey.

Gaussian Splatting. Few works tackle the use of Li-DAR within the context of Gaussian Splatting. Wu et al. [46] propose a multi-modal fusion system for SLAM. Specifically, by knowing the LiDAR to camera rigid pose, the initial pose estimate is obtained through point cloud registration and further refined via photometric error minimization. In this framework, the LiDAR points are leveraged to initialize the new 3D Gaussians. In a related approach, Hong et al. [14] proposes a LiDAR-Inertial-Visual SLAM system. The initial estimate is computed through a LiDAR-Inertial odometry. Points are partitioned using size-adaptive voxels to initialize 3D Gaussians using per-voxel covariances. Both the primitives and the trajectory are further refined via photometric error minimization. While these methods introduce LiDAR measurement, 3D Gaussians are still inherently processed by cameras. Recently, Chen et al.[5] applies 3D Gaussians for the task of LiDAR NVS for re-simulation. The authors propose the use of Periodic Vibrating 3D Gaussian primitives to account for dynamic objects present in the scene. The primitives are initialized using a lightweight MLP, and the rasterization is carried out in a spherical frame by computing a per-primitive plane orthogonal to the ray that connects the primitiveâs mean to the LiDAR frame, thus removing any distortion in the projection process. Focusing on a different formulation, Jiang et al. [17] propose a method of LiDAR NVS that leverages Periodic Vibrating 2D Gaussian primitives. The primitives are initialized by randomly sampling points and, further optimized using the losses described in [15], along with a Chamfer loss is introduced to constrain the 3D structures of the synthesized point clouds, and an additional ray-drop term to account for phenomena like non-returning laser pulses. This term is further refined through a U-Net that considers other factors, such as the distance of the surface from the sensor. Compared to this work, we provide a thorough methodology to render 2D Gaussians on spherical frames while accounting for coordinates singularity and a cloud registration technique to allow for simultaneous odometry and mapping. To our knowledge, Splat-LOAM is the first pure LiDAR Odometry and Mapping pipeline that leverages Gaussian primitives both for mapping and tracking. In sum, our contributions are

<!-- image-->  
Figure 2. Splat-LOAM Overview. Given a LiDAR point cloud, we leverage the spherical projection to generate an image-like representation. Moreover, using an ad-hoc differentiable rasterizer, we guide the optimization for structural parameters of 2D Gaussians. The underlying representation is concurrently used to incrementally register new measurements.

â¢ A differentiable, tile-based rasterizer for 2D Gaussians for spherical frames.

â¢ A mapping pipeline that allows the merging of sequential LiDAR measurements into a 2D Gaussian representation.

â¢ A tracking schema that leverages both 3D and 2D representations to register new measurements and estimate the sensor ego-motion.

## 3. Method

This section introduces our novel LiDAR odometry and mapping method based on 2D Gaussian primitives. We detail a mapping strategy for initializing, refining, and integrating these primitives alongside a registration method that leverages geometric and photometric cues from the continuous local model for ego-motion estimation. Additional details can be found in the supplementary material.

## 3.1. Spherical Projection Model

While original Gaussian Splatting leverages pinholecamera projection to render or refine 3D primitives, Li-

DAR input provides $3 6 0 ^ { \circ }$ panoramic input. To this end, we employ spherical projection to encode LiDAR measurements into an image-like representation that we can use to guide the Gaussian primitives optimization. A projection is a mapping $\phi : \mathbb { R } ^ { 3 } \to \Gamma \subset \mathbb { R } ^ { 2 }$ from a world point ${ \bf p } = ( x , y , z ) ^ { T }$ to image coordinates $\mathbf { u } = ( u , v ) ^ { T }$ . Knowing the range $d = \| \mathbf { p } \|$ of an image point u, we can calculate the inverse mapping $\phi ^ { - 1 } : \Gamma \times \mathbb { R } \to \mathbb { R } ^ { 3 }$ , more explicitly $\mathbf { p } = \phi ^ { - 1 } ( \mathbf { u } , d )$ We refer to this operation as backprojection. To ease the clarity of this work, it is worth mentioning that, compared to the classical pinhole camera, the optical reference frame is rearranged; the x-axis points forward, y-axis points to the left, and z-axis points upwards. Let K be a camera intrinsics matrix that can be computed directly from the point cloud (see Supplementary Material), with function Ï mapping a 3D point to azimuth and elevation. Thus, the spherical projection of a point is given by

$$
\phi ( \mathbf { p } ) = \mathbf { K } \psi ( \mathbf { p } ) ,\tag{1}
$$

$$
\psi ( \mathbf { v } ) = \left[ \underset { 1 } { \mathrm { a t a n 2 } } \left( v _ { y } , v _ { x } \right) \right] .\tag{2}
$$

We used spherical projection to obtain a range map DË of the LiDAR point cloud $\{ \bar { \mathbf { q } } _ { p } \} _ { p = 1 } ^ { Q }$ of size Q.

## 3.2. 2D Gaussian Splatting

Our method employs 2D Gaussians as the only scene representation, unifying the required design for accurate, efficient odometry estimation, mapping, and rendering. Due to the inherent thin structure and the explicit encoding of surface normals, 2D Gaussians have demonstrated excellent performance in surface reconstruction [6, 15], making them our preferred choice for primitives.

We define a 2D Gaussian by its opacity $o \in [ 0 , 1 ]$ , centroid $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , two tangential vectors $\mathbf { t } _ { \alpha } \in \mathbb { R } ^ { 3 }$ and $\mathbf { t } _ { \beta } \in \mathbb { R } ^ { 3 }$ defining its plane, and scaling vector $\mathbf { s } = ( s _ { \alpha } , s _ { \beta } ) \in \mathbb { R } ^ { 2 }$ that controls the variance of the Gaussian kernel along the plane [15]. Let $\mathbf { t } _ { n } = \mathbf { t } _ { \alpha } \times \mathbf { t } _ { \beta }$ be the normal of the plane, then the rotation matrix $\mathbf { R } = ( \mathbf { t } _ { \alpha } , \mathbf { t } _ { \beta } , \mathbf { t } _ { n } ) \in \mathbb { S O } ( 3 )$ . By arranging the scale factors into a diagonal matrix $\mathbf { S } \in \mathbb { R } ^ { 3 \times 3 }$ , whose last entry is zero, we obtain the following homogeneous transform that maps points $( \alpha , \beta )$ from the splatspace to the world-space w:

$$
\mathbf { H } = \left( \begin{array} { c c c c } { s _ { \alpha } \mathbf { t } _ { \alpha } } & { s _ { \beta } \mathbf { t } _ { \beta } } & { \mathbf { 0 } } & { \pmb { \mu } } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} \right) = \left( \begin{array} { c c } { \mathbf { R } \mathbf { S } } & { \pmb { \mu } } \\ { 0 } & { 1 } \end{array} \right)\tag{3}
$$

Additionally, the Gaussian kernel can be evaluated in the splat-space using:

$$
\mathcal { G } ( \alpha , \beta ) = \exp \left( - \frac { \alpha ^ { 2 } + \beta ^ { 2 } } { 2 } \right) .\tag{4}
$$

## 3.2.1. Rasterization

We render the 2D Gaussians via Î±-blending as in [20]. This technique requires, for each pixel, the list of primitives to blend, sorted by range. Instead of computing the per-pixel sorting of primitives, which is too expensive, we subdivide the image into a grid of 16 Ã 16 tiles and, concurrently, generate a per-tile list of primitives sorted from closer to further. Using the method described in Sec. 3.2.3, we compute the bounding box for each primitive and generate a per-tile list of primitives to be rasterized. Then, for each tile, we sort the T Gaussians based on their range, and finally, for each pixel u, we integrate them using Î±-blending from front to back to obtain a range d, normal n and opacity o values, as follows:

$$
d = \sum _ { i = 1 } ^ { T } o _ { i } \mathcal { G } _ { i } d _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - o _ { j } \mathcal { G } _ { j } )\tag{5}
$$

$$
\mathbf { n } = \sum _ { i = 1 } ^ { T } o _ { i } \mathcal { G } _ { i } \mathbf { t } _ { n _ { i } } \prod _ { j = 1 } ^ { i - 1 } ( 1 - o _ { j } \mathcal { G } _ { j } )\tag{6}
$$

$$
o = \sum _ { i = 1 } ^ { T } o _ { i } \mathcal { G } _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - o _ { j } \mathcal { G } _ { j } )\tag{7}
$$

We do not rely on the local affine approximation proposed in [20] due to numerical instabilities for slanted views and distortions for larger primitives. Instead, we rely on an explicit ray-splat intersection proposed in [45]. This is efficiently found by locating the intersection of three non-parallel planes.

## 3.2.2. Ray-splat Intersection

Let $\mathbf { v } = \phi ^ { - 1 } \left( \mathbf { K } ^ { - 1 } \mathbf { u } \right)$ be the normalized direction of pixel u. We parametrize v as the intersection of two orthogonal

<!-- image-->  
Figure 3. Bounding box computation for near-singularity splats. (a) shows the 3D configuration of a splat that approximately lies behind the camera. (b) shows the corresponding spherical image with the projected bounding box vertices. The distortion is removed by shifting the vertices along the horizontal direction to align the projected center to the image center. (c) being far from the coordinate singularity, we compute the maximum extent of the splat. (d) we revert the shift and propagate to the corresponding tiles via an offset from the central vertex, matching with the tiles highlighted on (a).

planes:

$$
\mathbf { h } _ { x } = \frac { \mathbf { v } \times \mathbf { u } _ { z } } { \| \mathbf { v } \times \mathbf { u } _ { z } \| } , \qquad \mathbf { h } _ { y } = \mathbf { h } _ { x } \times \mathbf { v } ,\tag{8}
$$

where $\mathbf { u } _ { z }$ is the unit z-direction vector. Then, we represent the planes in the splatâs space as:

$$
\mathbf { h } _ { \alpha } = \left( \mathbf { T } _ { w } ^ { c } \mathbf { H } \right) ^ { T } \mathbf { h } _ { x } , \mathbf { h } _ { \beta } = \left( \mathbf { T } _ { w } ^ { c } \mathbf { H } \right) ^ { T } \mathbf { h } _ { y } ,\tag{9}
$$

where $\mathbf { T } _ { w } ^ { k } \in \mathbb { S E } ( 3 )$ describes the world in the k-th sensor reference frame. We express the intersection of the two planes bundle as:

$$
\mathbf { h } _ { \alpha } \left( \alpha , \beta , 1 , 1 \right) ^ { T } = \mathbf { h } _ { \beta } \left( \alpha , \beta , 1 , 1 \right) ^ { T } = 0 \ \mathrm { ~ . ~ }\tag{10}
$$

The solution Î± and $\beta$ is then computed by solving the homogeneous linear system:

$$
\alpha ( { \mathbf { u } } _ { s } ) = \frac { \mathbf { h } _ { \alpha } ^ { 2 } \mathbf { h } _ { \beta } ^ { 4 } - \mathbf { h } _ { \alpha } ^ { 4 } \mathbf { h } _ { \beta } ^ { 2 } } { \mathbf { h } _ { \alpha } ^ { 1 } \mathbf { h } _ { \beta } ^ { 2 } - \mathbf { h } _ { \alpha } ^ { 2 } \mathbf { h } _ { \beta } ^ { 1 } } \mathrm { ~ , ~ } \beta ( { \mathbf { u } } _ { s } ) = \frac { \mathbf { h } _ { \alpha } ^ { 4 } \mathbf { h } _ { \beta } ^ { 1 } - \mathbf { h } _ { \alpha } ^ { 1 } \mathbf { h } _ { \beta } ^ { 4 } } { \mathbf { h } _ { \alpha } ^ { 1 } \mathbf { h } _ { \beta } ^ { 2 } - \mathbf { h } _ { \alpha } ^ { 2 } \mathbf { h } _ { \beta } ^ { 1 } }\tag{11}
$$

## 3.2.3. Bounding Box Computation

To leverage the efficient tile-based rasterizer, we need to know the tiles that should rasterize each primitive. Typically, this is achieved by estimating a bounding box around the projected central point of the splat. On spherical images, however, we need to account for the coordinate singularity at the horizontal boundaries {0, W }. Figure 3 describes the approach we designed to compute the splat image-space bounding box for spherical projection models. First, we compute the 3Ï splat-space bounding box and use Eq. (1) to obtain its image-space vertices. Moreover, we shift the vertices to match the central vertex to the image center to ensure that no vertex is projected near the coordinate singularity at the horizontal image boundary. Eventually, we compute the horizontal extent of the splat, revert the previous rotation, and propagate the splatâs ID to the nearby tiles while accounting for the coordinate singularity. This allows us to obtain consistent bounding boxes even when the splat lies approximately behind the sensorâs frame.

## 3.3. Odometry And Mapping

Following the modern literature of RGB-D SLAM [22, 34, 47, 53], we rely on keyframing to optimize local maps. We choose this approach for the following advantages: first, continuous integration over the same model can have adverse effects on artifacts and, more importantly, on runtime. Instead of decreasing the local density, we reset to a new local model if certain conditions are met, while also restricting the number of frames joining the optimization stage to allow effective online processing Based on this, we define each local map as an individual Gaussian model Ps:

$$
\mathbf { P } ^ { s } = \{ \mathcal { G } ( \pmb { \mu } , \pmb { \Sigma } , o ) | i = 1 , \ldots , N \} .\tag{12}
$$

## 3.3.1. Local model initialization

We initialize a new local model using the input LiDAR point cloud whenever necessary, such as at system startup or when visibility conditions require it. As a first step, we generate a valid pixel mask via the indicator function 1[Â·] as

$$
\mathbf { M } _ { v } = \mathbb { 1 } [ \hat { \mathbf { D } } > 0 ] ,\tag{13}
$$

and compute the range gradients $\nabla _ { \mathbf { D } }$ to construct a weight map over the range image. We use a weighted sampling of $n _ { d }$ points to prioritize complex regions. Splat positions are computed by back-projecting the range image, while their orientations are initialized by directing surface normals toward the sensor center to enhance initial visibility.

## 3.3.2. Local Model Refinement

We perform a limited number of refinement iterations $n _ { o }$ on the most recent keyframes. Unlike [47, 53], we employ a geometric distribution-based sampling scheme that guarantees at least 40% probability of selecting the most recent keyframe and progressively decreases the likelihood of choosing the older ones. To filter out artifacts caused by ray-drop phenomena in the LiDAR measurements, we only consider valid pixels to refine the local model parameters x. We start by applying a densification strategy similar to the one described in Sec. 3.3.1, with two extra terms. The first one, $\mathbf { M } _ { n } = \mathbb { 1 } [ \mathbf { O } _ { i } \leq \lambda _ { d , o } ]$ , target newly discovered areas while the second, $\mathbf { M } _ { e } = \mathbb { 1 } [ | \mathbf { D } _ { i } - \hat { \mathbf { D } } _ { i } | \geq \lambda _ { d , e } ]$ , target under-reconstructed regions.

To optimize the geometric consistency of the local model, we employ a loss term that minimizes the $L _ { 1 }$ error:

$$
\mathcal { L } _ { d } = \sum _ { \mathbf { u } \in \mathbf { M } _ { v } } \rho _ { d } \| \mathbf { D } ( \mathbf { u } , \mathbf { x } ) - \hat { \mathbf { D } } ( \mathbf { u } ) \| ,\tag{14}
$$

where $\rho _ { d }$ is a weight function dependent on the measurementâs range. In addition, we employ a self-regularization term to align the splatâs normals to the surface normals N estimated by the gradients of the range map D [15]:

$$
\mathcal { L } _ { n } = \sum _ { \mathbf { u } \in \mathbf { M } _ { v } } 1 - \mathbf { n } ^ { T } \mathbf { N } \big ( \mathbf { D } ( \mathbf { u } , \mathbf { x } ) \big )\tag{15}
$$

Furthermore, to promote the expansion of splats over uniform surfaces, we introduce an additional term that operates on the opacity channel of the rasterized images. Specifically, we drive the splats to cover the areas of the image containing valid measurements by correlating the opacity image O with the valid mask $\mathbf { M } _ { v }$

$$
\mathcal { L } _ { o } = \sum _ { \mathbf { u } \in \mathbf { M } _ { v } } - \log \big ( \mathbf { O } ( \mathbf { u } , \mathbf { x } ) \big ) .\tag{16}
$$

While $\mathcal { L } _ { o }$ allows the splats to grow, it can also cause some splats to become extremely large in unobserved areas. We employ an additional regularization term on the scaling parameter of all primitives N to compensate for this effect. We propose a novel regularization that allows the splat to extend up to a certain value $\tau _ { s }$ before penalizing its growth:

$$
\mathcal { L } _ { s } = \left\{ \begin{array} { c l } { \tau _ { s } - \operatorname* { m a x } \left( s _ { \alpha } , s _ { \beta } \right) } & { \mathrm { i f } \operatorname* { m a x } \left( s _ { \alpha } , s _ { \beta } \right) > \tau _ { s } } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{17}
$$

We found this term to be more effective rather than directly minimizing the deviation from the average [24, 47, 53] as it allows for anisotropic splats that are particularly useful for mapping details and edges, and provides more control on the density and structure of the model.

The final mapping objective function is defined as:

$$
\mathcal { L } _ { \mathrm { m a p } } = \mathcal { L } _ { d } + \lambda _ { o } \mathcal { L } _ { o } + \lambda _ { n } \mathcal { L } _ { n } + \lambda _ { s } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { s _ { i } } ,\tag{18}
$$

with $\lambda _ { o } , \lambda _ { n } , \lambda _ { s } \in \mathbb { R }$ being loss weights. We do not perform an opacity reset step, which could lead to catastrophic forgetting.

## 3.3.3. Frame-To-Model Registration

Each time a new keyframe is selected, we sample the local model by back-projecting the rasterized range image D onto a point cloud $\{ \mathbf { p } _ { q } \} _ { q = 1 } ^ { W \times \breve { H } }$ at its estimated pose. We design an ad-hoc tracking schema to benefit from both geometric and photometric consistencies provided by the Li-DAR and the rendered local model. Hence, the total odometry loss is composed of the sum of both residuals:

$$
\mathcal { L } _ { \mathrm { o d o m } } = \mathcal { L } _ { \mathrm { g e o } } + \mathcal { L } _ { \mathrm { p h o t o } } .\tag{19}
$$

Geometric Registration. To associate geometric entities, we employ a PCA-based kd-tree [12]. The kd-tree is built on the back-projected rendered range map $\{ \mathbf { p } _ { q } \} _ { q = 1 } ^ { W \times H }$ and segmented into tree leaves, corresponding to planar patches. Each leaf corresponds to $l = \langle \mathbf { p } _ { l } , \mathbf { n } _ { l } \rangle$ , that ${ \mathrm { i s } } ,$ the mean point $\mathbf { p } _ { l } \in \mathbb { R } ^ { 3 }$ and the surface normal ${ \mathbf n } _ { l } \in \mathbb { R } ^ { 3 }$ . The geometric loss $\mathcal { L } _ { \mathrm { g e o } }$ represents the sum of the point-to-plane distance between the mean leaf $\mathbf { p } _ { l }$ and the point of the current measurement point cloud k with $\{ \mathbf { q } _ { p } \} _ { p = 1 } ^ { Q }$ , along the normal ${ \bf n } _ { l }$ expressed in the local reference frame $\mathbf { T } _ { w } ^ { k } .$ Specifically, we have:

$$
\mathcal { L } _ { \mathrm { g e o } } = \sum _ { p , q \in \{ a \} } \rho _ { \mathrm { H u b e r } } \left( ( \mathbf { T } _ { w } ^ { k } \mathbf { n } _ { l _ { q } } ) ^ { T } ( \mathbf { T } _ { w } ^ { k } \mathbf { q } _ { p } - \mathbf { p } _ { q _ { i } } ) \right) ,\tag{20}
$$

where w is the global frame, k is the local frame, $\{ a \}$ is the set of leaf associations with the point from the measurement point cloud, and $\rho _ { \mathrm { H u b e r } }$ is the Huber robust loss function.

Photometric Registration. Leveraging the rendered range map D, we employ photometric registration for subpixel refinement, minimizing the photometric distance between the rendered and the spherical projected query point cloud DË . The photometric loss is formulated as:

$$
\mathcal { L } _ { \mathrm { p h o t o } } = \sum _ { \mathbf { u } } \Big \Vert \rho _ { \mathrm { H u b e r } } \Big ( \mathbf { D } ( \mathbf { u } ) - \hat { \mathbf { D } } \underbrace { \big ( \phi \big ( \mathbf { T } _ { w } ^ { k } \phi ^ { - 1 } ( \mathbf { u } , \hat { d } ) \big ) \big ) } _ { \mathbf { u } ^ { \prime } } \Big ) \Big \Vert ^ { 2 } .\tag{21}
$$

The evaluation point $\mathbf { u } ^ { \prime }$ in the query image $\hat { \bf D }$ is computed by first back-projecting the pixel u, applying the transform $\mathbf { T } _ { w } ^ { k } .$ , and then projecting it back. Pose updates Î´ are parameterized as local updates in the Lie algebra $\mathfrak { s e } ( 3 )$ . Therefore, the transformation $\mathbf { T } _ { k } ^ { w }$ of the local reference frame, expressed in the global reference frame, is updated as:

$$
\begin{array} { r } { \mathbf { T } _ { k } ^ { w } \gets \mathbf { T } _ { k } ^ { w } \exp ( \delta ) . } \end{array}\tag{22}
$$

This update is carried out using a second-order Gauss-Newton method. Local updates ensure that rotation updates are well-defined [11].

## 4. Experiments

In this section, we report the results of our method on different publicly available datasets. We evaluate our pipeline on both tracking and mapping. We recall that, to our knowledge, this is the first Gaussian Splatting LiDAR Odometry and Mapping pipeline, and no direct competitors are available. We compared our approach with other well-known geometric and neural implicit methods. The experiments were executed on a PC with an Intel Core i9-14900K @ 3.20Ghz, 64GB of RAM, and an NVIDIA RTX 4090 GPU with 24 GB of VRAM.

For the odometry experiments, we evaluate over several baselines. The first one is a basic point-to-plane ICP odometry, as a minimal example. Then, we include two geometric pipelines: SLAMesh simultaneously estimates a mesh representation of the scene via Gaussian Processes and perform registration onto it [32]. Moreover, MAD-ICP leverages a forest of PCA-based KD-Trees to perform accurate registration. Furthermore, we include PIN-SLAM as SOTA baseline for implicit LiDAR SLAM [30]. It leverages neural points as primitive and interleaves an incremental learning of the modelâs SDF and a correspondencefree, point-to-implicit registration schema. We also highlight that we could not run the official implementation of NeRF-LOAM [8], and thus, we do not include it in the evaluation.

We evaluate the mapping capabilities of our method against popular mapping SOTA pipelines. We include OpenVDB [27] and VoxBlox [29] as âclassicâ baselines. OpenVDB provides a robust volumetric data structure to handle 3D Points, while VoxBlox combines adaptive weights and grouped ray-casting for an efficient and accurate Truncated Signed Distance Function (TSDF) integration. Additionally, we include two neural-implicit mapping pipelines. $N ^ { 3 }$ -Mapping is a neural-implicit non-projective SDF mapping pipeline [39]. It leverages normal guidance to produce more accurate SDFs, leading to SOTA results for offline LiDAR mapping. PIN-SLAM [30] is included also for the mapping experiments. In fact, using marching cubes, it can produce a mesh from the underlying implicit SDF. Below, we report the datasets and the evaluation metrics employed. We highlight that we could not run the official implementation of SLAMesh [32] with ground-truth poses. Hence, we do not include the pipeline for quantitative comparisons.

## 4.1. Datasets

We used the following four publicly available datasets:

â¢ Newer College Dataset (NC) [50]: Collected with a handheld Ouster OS0-128 LiDAR in structured and vegetated areas. Ground truth was generated using the Leica BLK360 scanner, achieving centimeter-level accuracy over poses and points in the map.

<!-- image-->  
Figure 4. RPE evaluation. Number of successful sequences across RPE thresholds. It includes the sequences of Newer College [49], VBR [4], Oxford Spires [41] and Mai City [43].

<table><tr><td rowspan="3">Dataset</td><td colspan="4">Newer College[50]</td><td colspan="10">Oxford Spires[41]</td></tr><tr><td colspan="4">quad-easy</td><td colspan="4">keble-college02</td><td colspan="4">bodleian-library-02</td><td colspan="4">observatory-01</td></tr><tr><td>Accâ</td><td>Comâ</td><td>C-11â</td><td>F-scoreâ</td><td>Accâ</td><td>Comâ</td><td>C-11â</td><td>F-scoreâ</td><td>Accâ</td><td>Comâ</td><td>C-11â</td><td>F-scoreâ</td><td>Accâ</td><td>Comâ</td><td>C-11â</td><td>F-scoreâ</td></tr><tr><td>Approach OpenVDB[27]</td><td>11.45</td><td>4.38</td><td>7.92</td><td>88.85</td><td>7.46</td><td>6.92</td><td>7.19</td><td>91.74</td><td>10.34</td><td>4.68</td><td>7.51</td><td>89.68</td><td>9.58</td><td>9.60</td><td>9.59</td><td>86.16</td></tr><tr><td>VoxBlox[29]</td><td>20.36</td><td>12.64</td><td>16.5</td><td>64.63</td><td>15.81</td><td>14.25</td><td>15.03</td><td>71.63</td><td>18.92</td><td>11.56</td><td>15.24</td><td>58.77</td><td>15.09</td><td>15.15</td><td>15.12</td><td>70.45</td></tr><tr><td>N 3-Mapping[39]</td><td>6.32</td><td>9.75</td><td>8.04</td><td>94.54</td><td>6.21</td><td>7.82</td><td>7.01</td><td>93.47</td><td>10.16</td><td>5.62</td><td>7.89</td><td>90.36</td><td>8.27</td><td>10.44</td><td>9.35</td><td>87.94</td></tr><tr><td>PIN-SLAM[30]</td><td>15.28</td><td>10.5</td><td>12.89</td><td>88.05</td><td>13.73</td><td>9.94</td><td>11.83</td><td>79.65</td><td>14.34</td><td>7.14</td><td>10.74</td><td>82.71</td><td>16.91</td><td>12.07</td><td>14.49</td><td>72.31</td></tr><tr><td>Ours</td><td>6.64</td><td>4.09</td><td>5.37</td><td>96.74</td><td>6.18</td><td>8.69</td><td>7.43</td><td>94.41</td><td>10.87</td><td>4.33</td><td>7.6</td><td>90.09</td><td>9.35</td><td>11.76 10.56</td><td></td><td>83.04</td></tr></table>

Table 1. Reconstruction quality evaluation. The pipelines were run with ground-truth poses. Voxel size is set to 20 cm and F-score is computed with a 20 cm error threshold. Splat-LOAM yields competitive mapping performance on both the Newer College[50] and Oxford Spires[41] datasets and outperforms most competitive approaches.

â¢ A Vision Benchmark in Rome (VBR) [4]: Recorded in Rome using OS1-64 (car-mounted) and OS0-128 (handheld) LiDARs, capturing large-scale urban scenarios with narrow streets and dynamic objects. Ground truth trajectories were obtained by fusing LiDAR, IMU, and RTK GNSS data, ensuring centimeter-level accuracy over the poses.

â¢ Oxford Spires [41]: Recorded with a hand-held Hesai QT64 LiDAR featuring a 360â¦ horizontal FoV, 104â¦ vertical FoV, 64 vertical channels, and 60 meters of range detection. Similar to [49], each sequence includes a prior map obtained via a survey-grade 3D imaging laser scanner, used for ground-truth trajectory estimation and mapping evaluation. Specifically, we choose the Keble College, Bodleian Library, and Radcliffe Observatory sequences to include both indoor and outdoor scenarios with different levels of detail.

â¢ Mai City [43]: A synthetic dataset captured using a carlike simulated LiDAR with 120 meters of range detection. The measurements are generated via ray-casting on an underlying mesh, providing error-free, motion-undistorted data. We select the 01 and 02 sequences, which capture similar scenarios with different vertical resolutions.

## 4.2. Evaluation

We use Relative Pose Error (RPE) computed with progressively increasing delta steps to evaluate the odometry accuracy. Specifically, we adapt the deltas to the trajectory length to provide a more meaningful result [4]. Differently, to evaluate mapping quality, we use several metrics [25]: Accuracy (Acc) measures the mean distance of points on the estimated mesh with their nearest neighbors on the reference cloud. Completeness (Comp) measures the opposite distance, and Chamfer-l1 (C-l1) describes the mean of the two. Additionally, we use the F-score computed with 20 cm error threshold.

## 4.3. Ablation Study

Our approach employs Gaussian primitives for LiDAR odometry estimation and mapping, yielding results comparable to state-of-the-art methods while significantly enhancing computational efficiency. In this section, we evaluate some key aspects of our pipeline and evaluate their contributions.

Memory and Runtime Analysis. In Fig. 5, we report how the increment of active primitives affects the active GPU memory requirements and the per-iteration mapping frequency. It shows an experiment ran over the large-scale campus sequence [4] where we set a maximum of 100 keyframes per local map and sample, at most, 50% of points for the incoming point cloud. Itâs possible to notice that the mapping FPS remains stable between 200k and 300k primitives. This is most likely related to the saturation of the rasterizer.

Odometry. Fig. 7 shows the results for different tracking methods over our scene representation. Using both geometric and photometric components, we achieve a better result

<!-- image-->  
Figure 5. Memory and Runtime Analysis. The plot relates the used GPU Memory and the mapping iteration frequency with the number of active primitives. The measurements were obtained over the longest sequence we reported: campus [4].

<!-- image-->  
Figure 6. Comparison of Mesh Reconstruction. The figure shows reconstruction results for quad-easy sequence from the newer college dataset. Our method recovers a geometry with much higher data fidelity. PIN-SLAM lacks many details and exhibits a large level of noise. N 3-Mapping performs more similar to ours, but oversmoothes fine geometric details.

<!-- image-->  
Figure 7. Ablation on registration methods. The plot reports the RPE (%) of several tested registration methods on the quadeasy sequence. Enabling both geometric and photometric factors in sequence, provides a more robust estimate. the quad-easy sequence [50]

than using point-to-point or point-to-plane. The last three bars show an ablation of geometric and photometric loss components. The results are best for the joint use of both terms which support our design choice.

Mesh generation. To generate a mesh from the underlying representation, we sample $n _ { m }$ points from each keyframe rendered range and normal maps. Moreover, similar to [13], we accumulate the points into a single point cloud and generate the mesh using Poisson Reconstruction [18]. Tab. 2 shows the results of two additional methods for meshing that we tested: the Poisson Reconstruction of the Gaussian primitivesâ centers and normals and the TSDF integration using the rendered depth and normal maps from the keyframes.

<table><tr><td>Extraction Method</td><td>Accâ</td><td>Comâ</td><td>C-11â</td><td>F-scoreâ</td></tr><tr><td>Marching Cubes[27]</td><td>16.76</td><td>5.53</td><td>11.14</td><td>76.76</td></tr><tr><td>Poisson (centers)</td><td>10.15</td><td>6.70</td><td>8.43</td><td>92.33</td></tr><tr><td>Ours</td><td>6.64</td><td>4.09</td><td>5.37</td><td>96.74</td></tr></table>

Table 2. Ablation on Meshing Methods. We report mapping results with varying meshing methods on the quad-easy sequence [50]. Our method yields consistently better results.

## 5. Conclusion

We present the first LiDAR Odometry and Mapping pipeline that leverages 2D Gaussian primitives as the sole scene representation. Through an ad-hoc tile-based Gaussian rasterizer for spherical images, we leverage LiDAR measurements to optimize the local model. Furthermore, we demonstrate the effectiveness of combining a geometric and photometric tracker to register new LiDAR point clouds over the Gaussian local model. The experiments show that our pipeline obtains tracking and mapping scores that are on par with the current SOTA at a fraction of the computational demands.

Future Work. We plan on improving Splat-LOAM by simultaneously estimating the sensor pose and velocity to compensate for motion skewing of the LiDAR measurements. Moreover, we plan on including Loop Closure (LC) to improve the pose estimates, along with the mapping accuracy.

## References

[1] Jens Behley and Cyrill Stachniss. Efficient Surfel-Based SLAM using 3D Laser Range Data in Urban Environments. In Proc. of Robotics: Science and Systems (RSS). Robotics: Science and Systems Foundation, 2018. 1, 2

[2] P.J. Besl and Neil D. McKay. A method for registration of 3-D shapes. IEEE TPAMI, 14(2):239â256, 1992. 1, 2

[3] Dorit Borrmann, Jan Elseberg, Kai Lingemann, Andreas Nuchter, and Joachim Hertzberg. Globally consistent 3D Â¨ mapping with scan matching. Journal on Robotics and Autonomous Systems (RAS), 56(2):130â142, 2008. 1, 2

[4] Leonardo Brizi, Emanuele Giacomini, Luca Di Giammarino, Simone Ferrari, Omar Salem, Lorenzo De Rebotti, and Giorgio Grisetti. VBR: A Vision Benchmark in Rome. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), pages 15868â15874, 2024. 6, 7

[5] Qifeng Chen, Sheng Yang, Sicong Du, Tao Tang, Peng Chen, and Yuchi Huo. LiDAR-GS:Real-time LiDAR Re-Simulation using Gaussian Splatting, 2024. 2

[6] Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu, Huamin Wang, and Weiwei Xu. High-quality Surface Reconstruction using Gaussian Surfels. In ACM SIGGRAPH 2024 Conference Papers, pages 1â11, New York, NY, USA, 2024. Association for Computing Machinery. 2, 3

[7] Pierre Dellenbach, Jean-Emmanuel Deschaud, Bastien Jacquet, and FrancÂ¸ois Goulette. CT-ICP: Real-time Elastic LiDAR Odometry with Loop Closure. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), pages 5580â 5586, 2022. 1, 2

[8] Junyuan Deng, Qi Wu, Xieyuanli Chen, Songpengcheng Xia, Zhen Sun, Guoqing Liu, Wenxian Yu, and Ling Pei. NeRF-LOAM: Neural Implicit Representation for Large-Scale Incremental LiDAR Odometry and Mapping. In ICCV, pages 8184â8193, 2023. 1, 2, 6

[9] Luca Di Giammarino, Leonardo Brizi, Tiziano Guadagnino, Cyrill Stachniss, and Giorgio Grisetti. MD-SLAM: Multicue Direct SLAM. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 11047â11054, 2022. 1, 2

[10] Luca Di Giammarino, Emanuele Giacomini, Leonardo Brizi, Omar Salem, and Giorgio Grisetti. Photometric LiDAR and RGB-D Bundle Adjustment. IEEE Robotics and Automation Letters (RA-L), 8(7):4362â4369, 2023. 2

[11] Chris Engels, Henrik Stewenius, and David Nist Â´ er. Bundle Â´ Adjustment Rules. Photogrammetric computer vision, 2(32), 2006. 6

[12] Simone Ferrari, Luca Di Giammarino, Leonardo Brizi, and Giorgio Grisetti. MAD-ICP: It is All About Matching Data â Robust and Informed LiDAR Odometry. IEEE Robotics and Automation Letters (RA-L), 9(11):9175â9182, 2024. 1, 2, 6

[13] Antoine Guedon and Vincent Lepetit. SuGaR: Surface- Â´ Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering. In CVPR, pages 5354â5363, 2024. 2, 8

[14] Sheng Hong, Junjie He, Xinhu Zheng, and Chunran Zheng. LIV-GaussMap: LiDAR-Inertial-Visual Fusion for Real-

Time 3D Radiance Field Map Rendering. IEEE Robotics and Automation Letters (RA-L), 9(11):9765â9772, 2024. 2

[15] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In ACM SIGGRAPH 2024 Conference Papers, pages 1â11, New York, NY, USA, 2024. Association for Computing Machinery. 2, 3, 4, 5

[16] Seth Isaacson, Pou-Chun Kung, Mani Ramanagopal, Ram Vasudevan, and Katherine A. Skinner. LONER: LiDAR Only Neural Representations for Real-Time SLAM. IEEE Robotics and Automation Letters (RA-L), 8(12):8042â8049, 2023. 1, 2

[17] Junzhe Jiang, Chun Gu, Yurui Chen, and Li Zhang. GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting. In ICLR, 2025. 2

[18] Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe. Poisson surface reconstruction. In Proceedings of the Fourth Eurographics Symposium on Geometry Processing, pages 61â70, Goslar, DEU, 2006. Eurographics Association. 8

[19] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track and map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2

[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM TOG, 42(4):139:1â139:14, 2023. 2, 4

[21] Qing Li, Shaoyang Chen, Cheng Wang, Xin Li, Chenglu Wen, Ming Cheng, and Jonathan Li. LO-Net: Deep Real-Time Lidar Odometry. In CVPR, pages 8465â8474, 2019. 2

[22] Lorenzo Liso, Erik Sandstrom, Vladimir Yugay, Luc Â¨ Van Gool, and Martin R. Oswald. Loopy-SLAM: Dense Neural SLAM with Loop Closures. In CVPR, pages 20363â 20373, 2024. 1, 2, 5

[23] Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18039â18048, 2024. 2

[24] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. Gaussian Splatting SLAM. In CVPR, pages 18039â18048, 2024. 2, 5

[25] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In CVPR, 2019. 7

[26] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV, pages 405â421, Cham, 2020. Springer International Publishing. 1

[27] Ken Museth. VDB: High-resolution sparse volumes with dynamic topology. ACM TOG, 32(3):27:1â27:22, 2013. 6, 7, 8

[28] Austin Nicolai, Ryan Skeele, Christopher Eriksen, and Geoffrey A. Hollinger. Deep Learning for Laser Based Odometry Estimation. In RSS workshop Limits and Potentials of Deep Learning in Robotics, page 1, 2016. 2

[29] Helen Oleynikova, Zachary Taylor, Marius Fehr, Roland Siegwart, and Juan Nieto. Voxblox: Incremental 3D Euclidean Signed Distance Fields for on-board MAV planning. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 1366â1373, 2017. 6, 7

[30] Yue Pan, Xingguang Zhong, Louis Wiesmann, Thorbjorn Â¨ Posewsky, Jens Behley, and Cyrill Stachniss. PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency. IEEE Trans. on Robotics, 40:4045â4064, 2024. 1, 2, 6, 7

[31] Jan Quenzel and Sven Behnke. Real-time Multi-Adaptive-Resolution-Surfel 6D LiDAR Odometry using Continuoustime Trajectory Optimization. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 5499â5506, 2021. 1, 2

[32] Jianyuan Ruan, Bo Li, Yibo Wang, and Yuxiang Sun. SLAMesh: Real-time LiDAR Simultaneous Localization and Meshing. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), pages 3546â3552, 2023. 1, 2, 6, 3

[33] Erik Sandstrom, Yue Li, Luc Van Gool, and Martin R. Â¨ Oswald. Point-SLAM: Dense Neural Point Cloud-based SLAM. In ICCV, pages 18387â18398, 2023. 1, 2

[34] Erik Sandstrom, Keisuke Tateno, Michael Oechsle, MichaelÂ¨ Niemeyer, Luc Van Gool, Martin R. Oswald, and Federico Tombari. Splat-SLAM: Globally Optimized RGB-only SLAM with 3D Gaussians, 2024. 2, 5

[35] Johannes L. Schonberger and Jan-Michael Frahm. Structure- Â¨ from-Motion Revisited. In CVPR, pages 4104â4113, 2016. 1

[36] Johannes L. Schonberger, Enliang Zheng, Jan-Michael Â¨ Frahm, and Marc Pollefeys. Pixelwise View Selection for Unstructured Multi-View Stereo. In ECCV, pages 501â518, Cham, 2016. Springer International Publishing. 1

[37] A. Segal, D. Haehnel, and S. Thrun. Generalized-ICP. In Proc. of Robotics: Science and Systems (RSS). Robotics: Science and Systems Foundation, 2009. 1, 2

[38] Tixiao Shan and Brendan Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 4758â4765, 2018. 2

[39] Shuangfu Song, Junqiao Zhao, Kai Huang, Jiaye Lin, Chen Ye, and Tiantian Feng. N3-Mapping: Normal Guided Neural Non-Projective Signed Distance Fields for Large-Scale 3D Mapping. IEEE Robotics and Automation Letters (RA-L), 9 (6):5935â5942, 2024. 1, 2, 6, 7

[40] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J. Davison. iMAP: Implicit Mapping and Positioning in Real-Time. In ICCV, pages 6209â6218, 2021. 1

[41] Yifu Tao, Miguel Angel Mu Â´ noz-Ba Ë nËon, Lintong Zhang, Ji- Â´ ahao Wang, Lanke Frank Tarimo Fu, and Maurice Fallon. The Oxford Spires Dataset: Benchmarking Large-Scale

LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods, 2024. 6, 7, 3, 4

[42] Fabio Tosi, Youmin Zhang, Ziren Gong, Erik Sandstrom, Â¨ Stefano Mattoccia, Martin R. Oswald, and Matteo Poggi. How nerfs and 3d gaussian splatting are reshaping slam: a survey, 2024. 2

[43] Ignacio Vizzo, Xieyuanli Chen, Nived Chebrolu, Jens Behley, and Cyrill Stachniss. Poisson Surface Reconstruction for LiDAR Odometry and Mapping. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), pages 5624â5630, 2021. 6, 7

[44] Guangming Wang, Xinrui Wu, Zhe Liu, and Hesheng Wang. PWCLO-Net: Deep LiDAR Odometry in 3D Point Clouds Using Hierarchical Embedding Mask Optimization. In CVPR, pages 15905â15914, 2021. 2

[45] Tim Weyrich, Simon Heinzle, Timo Aila, Daniel B. Fasnacht, Stephan Oetiker, Mario Botsch, Cyril Flaig, Simon Mall, Kaspar Rohrer, Norbert Felber, Hubert Kaeslin, and Markus Gross. A hardware architecture for surface splatting. ACM TOG, 26(3):90âes, 2007. 4

[46] Chenyang Wu, Yifan Duan, Xinran Zhang, Yu Sheng, Jianmin Ji, and Yanyong Zhang. MM-Gaussian: 3D Gaussianbased Multi-modal Fusion for Localization and Reconstruction in Unbounded Scenes. In Proc. of the IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 12287â12293, 2024. 2

[47] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Oswald. Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting, 2024. 2, 5

[48] Ganlin Zhang, Erik Sandstrom, Youmin Zhang, Manthan Pa- Â¨ tel, Luc Van Gool, and Martin R Oswald. Glorie-slam: Globally optimized rgb-only implicit encoding point cloud slam. arXiv preprint arXiv:2403.19549, 2024. 2

[49] Ji Zhang and Sanjiv Singh. LOAM: Lidar Odometry and Mapping in Real-time. In Proc. of Robotics: Science and Systems (RSS). Robotics: Science and Systems Foundation, 2014. 2, 6, 7

[50] Lintong Zhang, Marco Camurri, David Wisth, and Maurice Fallon. Multi-Camera LiDAR Inertial Extension to the Newer College Dataset, 2021. 6, 7, 8

[51] Xin Zheng and Jianke Zhu. Efficient LiDAR Odometry for Autonomous Driving. IEEE Robotics and Automation Letters (RA-L), 6(4):8458â8465, 2021. 2

[52] Xingguang Zhong, Yue Pan, Jens Behley, and Cyrill Stachniss. SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit Neural Representations. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), pages 8371â8377, 2023. 1, 2

[53] Liyuan Zhu, Yue Li, Erik Sandstrom, Shengyu Huang, Kon- Â¨ rad Schindler, and Iro Armeni. LoopSplat: Loop Closure by Registering 3D Gaussian Splats. In Proc. of the International Conference on 3D Vision (3DV), 2025. 2, 5

[54] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. NICE-SLAM: Neural Implicit Scalable Encoding for SLAM. In CVPR, pages 12776â12786, 2022. 1, 2

# Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping

Supplementary Material

In this supplementary material, we provide additional details on various components and design choices that were not fully elaborated in the main paper. These include the computation of the camera matrix, the rationale behind the bounding box computation, the incorporation of color features, and the analytical derivation of our spherical rasterizer.

## A. Camera Matrix

Using hard-coded field of views from the sensorâs datasheet may lead to empty areas inside the image (i.e., when parts of the FoV contain no observable environment or, due to rounding errors). To solve these issues, we recompute the field of views, along with a camera matrix, for each input LiDAR point cloud.

Let $\{ \mathbf { p } _ { q } \} _ { q = 1 } ^ { N }$ be a set of points expressed at the sensor origin. Let $\{ \mathbf { v } _ { q } = ( \gamma , \theta , 1 ) ^ { T } = \psi ( \mathbf { p } _ { q } ) \} _ { q = 1 } ^ { N }$ be the same set of points expressed in spherical coordinates. From this representation, we can directly estimate the camera matrix K as follows. First, we compute the maximum horizontal and vertical angular values within the set:

$$
\gamma _ { m } = \operatorname* { m i n } _ { \gamma } \left\{ \mathbf { v } _ { q } \right\} \qquad \quad \gamma _ { M } = \operatorname* { m a x } _ { \gamma } \left\{ \mathbf { v } _ { q } \right\} ,\tag{23}
$$

$$
\theta _ { m } = \operatorname* { m i n } _ { \theta } \left\{ { \bf v } _ { q } \right\} \qquad \quad \theta _ { M } = \operatorname* { m a x } _ { \theta } \left\{ { \bf v } _ { q } \right\} .\tag{24}
$$

Moreover, we compute the horizontal $\mathrm { F o V } _ { h } = \gamma _ { M } - \gamma _ { m }$ and vertical F $\mathrm { , } \mathrm { V } _ { h } = \theta _ { M } - \theta _ { m }$ field of views and, provided an image size $( H , W )$ , we estimate the camera matrix as:

$$
\mathbf { K } \left( \left\{ \mathbf { p } _ { q } \right\} \right) = \left( \begin{array} { c c c } { - \frac { W - 1 } { \mathrm { F o V } _ { h } } } & { 0 } & { \frac { W } { 2 } \left( 1 + \frac { \gamma _ { M } + \gamma _ { m } } { \mathrm { F o V } _ { h } } \right) } \\ { 0 } & { - \frac { H - 1 } { \mathrm { F o V } _ { v } } } & { \frac { H } { 2 } \left( 1 + \frac { \theta _ { M } + \theta _ { m } } { \mathrm { F o V } _ { v } } \right) } \\ { 0 } & { 0 } & { 1 } \end{array} \right)\tag{25}
$$

## B. Bounding Box

In this section, we report a supplementary study concerning the computation of the tightly aligned bounding box on spherical images. Efficiently computing the tightly aligned bounding box for a splat on the view space requires solving a 4-th-order polynomial due to the complexity of the underlying manifold. While fixing the azimuth angle Î³ results in a planar surface in $\mathbb { R } ^ { 3 }$ , fixing the altitude angle Î¸ leads to a cone subspace in $\mathbb { R } ^ { 3 }$ . To find the tightly aligned bounding box, we should search the spherical coordinates $( \gamma , \theta )$ that exactly intersect tangentially the splat space at a distance 3Ï from the origin. Projecting the Î±-plane onto the splat frame results in a line, and the intersection condition can be solved via a linear equation. Projecting the Î³-cone onto the splatâs frame results in a parametric 2D conic equation. Enforcing two tangent solutions leads to a polynomial of 4th-degree. Given the small image sizes of LiDAR images and the relatively high cost of solving higher-order polynomials, we opt for an easier but less optimal solution. We relax the tight constraint and obtain an image-space bounding box by projecting the individual bounding box vertices. This typically results in a bounding box that includes more pixels but is faster in computation.

## C. On color features

Modern LiDARs provide a multitude of information on the beam returns. Specifically, they provide details concerning the mean IR light level (ambient) and the returned intensity (intensity). Through this information, it is also possible to compute the reflectivity of the surface using the inverse square law for Lambertian objects in far fields. Throughout this study, we opted to omit the color information to focus on the geometric reconstruction capabilities of our approach. Moreover, we think incorporating intensity and reflectivity channels can pose a challenge due to the inherent nature of LiDARs. Both properties cannot be explicitly related to a portion of the space but rather from a combination of the surface and sensor position with respect to the former.

## D. Rasterizer Details

In this section, we describe the process of rasterization over spherical images. First, we provide a detailed analysis of the rasterization process for a Gaussian primitive. Furthermore, we provide the analytical derivatives for the components of the process. Recall that a Gaussian primitive G is defined by its centroid $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , its covariance matrix decomposed as a rotation matrix $\mathbf { R } \in \mathbb { S O } ( 3 )$ and a scaling matrix S, and its opacity $o \in \mathbb { R }$ . To obtain the homogeneous transform that maps points (Î±, Î²) from the splat-space to the sensor-space c, we decouple the axes of the rotation matrix $\mathbf { R } = ( \mathbf { t } _ { \alpha } , \mathbf { t } _ { \beta } , \mathbf { t } _ { n } )$ and the per-axis scaling parameters ${ \bf s } = ( s _ { \alpha } , s _ { \beta } )$ , and assume $\mathbf { T } _ { w } ^ { c } \in \mathbb { S E } ( 3 )$ be the transform the world in camera frame. By concatenating $\mathbf { T } _ { w } ^ { c }$ with Eq. (3), we obtain the following transform:

$$
\mathbf { T } _ { 4 \times 3 } = \mathbf { T } _ { w } ^ { c } \mathbf { H } = \left( \begin{array} { c c c } { \mathbf { b } _ { \alpha } } & { \mathbf { b } _ { \beta } } & { \mathbf { b } _ { c } } \\ { 0 } & { 0 } & { 1 } \end{array} \right) ,\tag{26}
$$

where $\mathbf { b } _ { c } = \mathbf { R } _ { w } ^ { c } \pmb { \mu } + \mathbf { t } _ { w } ^ { c }$ . We omit the third column of $\mathbf { T } ,$ which is zeroed by construction.

## D.1. Forward Process

In this section, we describe the rasterization process for a pixel $\mathbf { u } = ( u , v )$ . We assume that primitives are already presorted. As described in Sec. 3.2.2, we compute the orthogonal planes in splat-space by pre-multiplying each plane by T:

$$
\mathbf h _ { \alpha } = \mathbf T ^ { T } \mathbf h _ { x } \qquad \mathbf h _ { \beta } = \mathbf T ^ { T } \mathbf h _ { y } ,\tag{27}
$$

and compute the intersection point pË:

$$
\hat { \mathbf { p } } = \mathbf { h } _ { \alpha } \times \mathbf { h } _ { \beta }\tag{28}
$$

$$
\hat { \mathbf { s } } = \left( \hat { s } _ { \alpha } , \hat { s } _ { \beta } \right) = \left( \frac { \hat { \mathbf { p } } _ { x } } { \hat { \mathbf { p } } _ { z } } , \frac { \hat { \mathbf { p } } _ { y } } { \hat { \mathbf { p } } _ { z } } \right) ^ { T } .\tag{29}
$$

We use Ës to estimate two quantities. First, we measure the Gaussian kernel at the intersection point G (Ës) to compute the Gaussian density

$$
\alpha = o \mathcal { G } \left( \hat { \mathbf { s } } \right) ,\tag{30}
$$

and second, we compute the range as

$$
\pmb { \nu } = \hat { s } _ { \alpha } \mathbf { b } _ { \alpha } + s _ { \beta } \mathbf { b } _ { \beta } + \mathbf { b } _ { c }\tag{31}
$$

$$
d = \| \nu \| .\tag{32}
$$

We follow Eq. (5), Eq. (6), and Eq. (7) to Î±-blend the sorted Gaussians and compute the pixel contributions.

## D.2. Gradient Computation

From the rasterizer perspective, we assume to already have the per-pixel channel derivatives, namely the depth $\overline { { { \frac { \partial \mathcal { L } } { \partial \mathcal { d } } } } } ~ \in \mathbf { \partial }$ R and normal $\frac { \partial \mathcal { L } } { \partial \mathbf { n } } ~ \in ~ \mathbb { R } ^ { 3 }$ To improve the readability, each partial derivative also includes its dimension using the $\frac { \partial A } { \partial B } \left| _ { \mathrm { d i m } ( A ) \times \mathrm { d i m } ( B ) } \right.$ notation. Finally, we show the computation for the k-th Gaussian over the m Gaussians contributing to the pixel.

First, we derive the gradients w.r.t the density:

$$
\frac { \partial d } { \partial d _ { k } } \bigg | _ { 1 \times 1 } = d _ { k } A _ { k } - \frac { B _ { d , k } } { 1 - \alpha _ { k } }\tag{33}
$$

$$
\left. \frac { \partial \mathbf { n } } { \partial \mathbf { n } _ { k } } \right| _ { 1 \times 3 } = \mathbf { n } _ { k } A _ { k } - \frac { B _ { \mathbf { n } , k } } { 1 - \alpha _ { k } }\tag{34}
$$

$$
\frac { \partial \mathcal { L } } { \partial \alpha _ { k } } \Bigg | _ { 1 \times 1 } = \frac { \partial \mathcal { L } _ { d } } { \partial d } \frac { \partial d } { \partial \alpha _ { k } } + \frac { \partial \mathcal { L } _ { n } } { \partial \mathbf { n } } \frac { \partial \mathbf { n } } { \partial \mathbf { n } _ { k } } ,\tag{35}
$$

where $\begin{array} { r } { A _ { k } = \prod _ { i = 1 } ^ { k - 1 } ( 1 - \alpha _ { i } ) , B _ { d , k } = \sum _ { i > k } d _ { i } \alpha _ { i } A _ { i } , } \end{array}$ and $\begin{array} { r } { B _ { \mathbf { n } , k } = \sum _ { i > k } \mathbf { n } _ { i } \alpha _ { i } A _ { i } } \end{array}$ . We leverage the sorting of the primitives to efficiently compute these values during the backpropagation of the gradients.

Furthermore, we propagate the gradients to the homogeneous transform matrix T from the intersection of planes

pËk:

(36)

$$
\begin{array} { r l } & { \frac { \partial \mathcal { L } } { \partial \hat { \mathbf { s } } _ { k } } \bigg \vert _ { 1 \times 2 } = \frac { \partial \mathcal { L } } { \partial \alpha } \frac { \partial \alpha } { \partial \hat { \mathbf { s } } _ { k } } + \frac { \partial \mathcal { L } } { \partial d _ { k } } \frac { \partial d _ { k } } { \partial \hat { \mathbf { s } } _ { k } } } \\ & { \qquad = - \frac { \partial \mathcal { L } } { \partial \alpha } \alpha \kappa \hat { \mathbf { s } } _ { k } ^ { r } + \frac { \partial \mathcal { L } } { \partial d _ { k } } \frac { \alpha \kappa A _ { k } } { d _ { k } } \left( \nu ^ { T } \mathbf { b } _ { \alpha } \right) ^ { T } } \\ & { \frac { \partial \mathcal { L } } { \partial \hat { \mathbf { p } } _ { k } } \bigg \vert _ { 1 \times 3 } = \frac { \partial \mathcal { L } } { \partial \hat { \mathbf { s } } _ { k } } \frac { \partial \hat { \mathbf { s } } _ { k } } { \partial \hat { \mathbf { p } } _ { k } } } \\ & { \qquad = \frac { 1 } { \hat { \mathbf { p } } _ { z } } \left( \frac { \partial \mathcal { L } / \partial \hat { \mathbf { s } } _ { \alpha } } { \partial \hat { \mathbf { s } } _ { k } \partial \hat { \mathbf { z } } / \partial \mathbf { s } _ { \alpha } } \right) ^ { T } . } \end{array}\tag{37}
$$

Thus, we can derive the gradients over the matrix T. We keep the three accumulators $\textstyle { \frac { \partial { \mathcal { L } } } { \partial \mathbf { b } _ { \alpha } } } , \ { \frac { \partial { \mathcal { L } } } { \partial \mathbf { b } _ { \beta } } }$ , and $\frac { \partial \mathcal { L } } { \partial \mathbf { b } _ { c } }$ decoupled to correctly integrate the contributions from each pixel:

$$
{ \pmb \rho } _ { \alpha } = \left( \frac { \partial \mathcal { L } } { \partial \mathbf { p } _ { k } } \times \mathbf { h } _ { \alpha } \right) \qquad { \pmb \rho } _ { \beta } = \left( \frac { \partial \mathcal { L } } { \partial \mathbf { p } _ { k } } \times \mathbf { h } _ { \beta } \right)\tag{38}
$$

$$
\left. \frac { \partial \mathcal { L } } { \partial  { \mathbf { b } } _ { \alpha } } \right| _ { 1 \times 3 } = \frac { \partial \mathcal { L } } { \partial \hat {  { \mathbf { p } } } _ { k } } \frac { \partial \hat {  { \mathbf { p } } } _ { k } } { \partial  { \mathbf { b } } _ { \alpha } } + \frac { \partial \mathcal { L } } { \partial d _ { k } } \frac { \partial d _ { k } } { \partial  { \mathbf { b } } _ { \alpha } }\tag{39}
$$

$$
\begin{array} { l } { \displaystyle \frac { \partial \mathcal { L } } { \partial { \bf b } _ { \beta } } \bigg \vert _ { { \bf 1 } \times { \boldsymbol 3 } } = \frac { \partial \mathcal { L } } { \partial \hat { \bf p } _ { k } } \frac { \partial \hat { \bf p } _ { k } } { \partial { \bf b } _ { \beta } } + \frac { \partial \mathcal { L } } { \partial d _ { k } } \frac { \partial d _ { k } } { \partial { \bf b } _ { \beta } } } \\ { = - \rho _ { \beta , 2 } { \bf h } _ { x } + \rho _ { \alpha , 2 } { \bf h } _ { y } + \frac { \mathcal { L } } { \partial d _ { k } } \frac { \hat { s } _ { \beta } } { d _ { k } } \nu ^ { T } } \\ { \displaystyle \frac { \partial \mathcal { L } } { \partial { \bf b } _ { c } } \bigg \vert _ { { \bf 1 } \times { \boldsymbol 3 } } = \frac { \partial \mathcal { L } } { \partial \hat { \bf p } _ { k } } \frac { \partial \hat { \bf p } _ { k } } { \partial { \bf b } _ { c } } + \frac { \partial \mathcal { L } } { \partial d _ { k } } \frac { \partial d _ { k } } { \partial { \bf b } _ { c } } } \\ { = - \rho _ { \beta , 3 } { \bf h } _ { x } + \rho _ { \alpha , 3 } { \bf h } _ { y } + \frac { \mathcal { L } } { \partial d _ { k } } \frac { 1 } { d _ { k } } \nu ^ { T } , } \end{array}\tag{40}
$$

(41)

where $\rho _ { k , i }$ is the i-th element of $\rho _ { k }$

Finally, we compute the gradients w.r.t. the Gaussian parameters.

$$
{ \cfrac { \partial { \mathcal { L } } } { \partial \mathbf { R } _ { k } } } ~ { \bigg | } _ { 3 \times 3 } = \left( s _ { \alpha } { \frac { \partial { \mathcal { L } } } { \partial \mathbf { b } _ { \alpha } } } ^ { T } \quad s _ { \beta } { \frac { \partial { \mathcal { L } } } { \partial \mathbf { b } _ { \beta } } } ^ { T } \quad { \frac { \partial { \mathcal { L } } } { \partial \mathbf { n } _ { k } } } ^ { T } \right) \mathbf { R } _ { w } ^ { c }\tag{42}
$$

$$
\begin{array} { r l } { \left. { \frac { \partial { \mathcal { L } } } { \partial \mathbf { S } _ { k } } } \right| _ { 1 \times 2 } = \left( { \frac { \partial { \mathcal { L } } } { \partial \mathbf { b } _ { \alpha } } } \mathbf { R } _ { w } ^ { c } \mathbf { R } _ { [ 1 ] } \right. } & { { } \left. { \frac { \partial { \mathcal { L } } } { \partial \mathbf { b } _ { \alpha } } } \mathbf { R } _ { w } ^ { c } \mathbf { R } _ { [ 1 ] } \right) } \end{array}\tag{43}
$$

$$
\left. \frac { \partial \mathcal { L } } { \partial \pmb { \mu } _ { k } } \right| _ { 1 \times 3 } = \frac { \partial \mathcal { L } } { \partial \mathbf { b } _ { c } } \mathbf { R } _ { w } ^ { c } ,\tag{44}
$$

$$
\left. \frac { \partial \mathcal { L } } { \partial o _ { k } } \right| _ { 1 \times 1 } = \frac { \partial \mathcal { L } } { \partial \alpha _ { k } } \exp \left( - \frac { 1 } { 2 } \mathbf { s } _ { k } ^ { T } \mathbf { s } _ { k } \right)\tag{45}
$$

where $\mathbf { R } _ { [ i ] }$ is the i-th column of R.

<!-- image-->

Figure D.1. Qualitative Mapping Results. The images show the mapping results for different pipelines in the quad-easy sequence. We also include the SLAMesh pipeline, which was evaluated on a self-estimated trajectory.  
<!-- image-->  
Figure F.1. Effects of motion distortion during registration. The images show the projective range error between our model and the incoming measurement (a) before a rotation, (b) during a rotation, and (c) after the rotation. The images are resized over the horizontal axis for visibility purposes. The error is expressed in meters.

## E. Additional Qualitative Comparison

In this section, we show additional results over the meshing reconstruction. Figure D.1 includes the results we obtained using the work of Ruan et al. [32]. We remark that we did not include these results in the manuscript as we could not run the official implementation over the Ground Truth trajectory. Additionally, Figure F.2 shows the reconstruction results over the Oxford Spires dataset [41].

## F. Motion Distortion

Throughout the experiments, we noticed that Splat-LOAM is very sensitive to the motion distortion effect caused by the continual acquisition of LiDARs. Figure F.1 shows how the projective error over the estimated model changes while the sensor rotates during the acquisition, hindering both the registration and mapping phases. We plan to compensate for the motion distortion effect by simultaneously estimating the sensor pose and velocity.

<!-- image-->  
Figure F.2. Qualitative Mapping Results. The images show the mapping results for different pipelines in the Oxford Spires dataset [41].