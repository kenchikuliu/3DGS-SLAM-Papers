# FGGS-LiDAR: Ultra-Fast, GPU-Accelerated Simulation from General 3DGS Models to LiDAR

Junzhe Wu1â, Yufei Jia2â, Yiyi Yan3, Zhixing Chen1, Tiao Tan1, Zifan Wang4, Guangyu Wang1, BoKui Chen1â , Guyue Zhou4â 

<!-- image-->  
Fig. 1: Overview of FGGS-LiDAR framework. (a) Visualization of a generic 3DGS dataset. (b) Schematic illustration of our 3DGS2Mesh conversion pipeline, which transforms Gaussian primitives into mesh-based scene representations. (We use the vase as an example for the illustration.) (c) Conceptual diagram of our LiDAR ray-casting procedure, simulating range measurements through efficient GPU-based traversal. (d) Visualization of rendered LiDAR scans.

Abstractâ While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic rendering, its vast ecosystem of assets remains incompatible with high-performance LiDAR simulation, a critical tool for robotics and autonomous driving. We present FGGS-LiDAR, a framework that bridges this gap with a truly plug-and-play approach. Our method converts any pretrained 3DGS model into a high-fidelity, watertight mesh without requiring LiDAR-specific supervision or architectural alterations. This conversion is achieved through a general pipeline of

volumetric discretization and Truncated Signed Distance Field (TSDF) extraction. We pair this with a highly optimized, GPUaccelerated ray-casting module that simulates LiDAR returns at over 500 FPS. We validate our approach on indoor and outdoor scenes, demonstrating exceptional geometric fidelity; By enabling the direct reuse of 3DGS assets for geometrically accurate depth sensing, our framework extends their utility beyond visualization and unlocks new capabilities for scalable, multimodal simulation. Our open-source implementation is available at https://github.com/TATP-233/FGGS-LiDAR.

## I. INTRODUCTION

LiDAR is a cornerstone modality for 3D perception, underpinning autonomous driving, localization, odometry, mapping, and indoor navigation [1], [2], [3], [4], [5]. To mitigate the prohibitive expense and logistical challenges of curating large-scale real-world datasets, simulation offers a controllable and reproducible source of data for training and benchmarking perception algorithms. Consequently, the fidelity, efficiency, and scalability of LiDAR simulators are critical factors that directly determine their utility in both research and real-world deployment.

The landscape of LiDAR simulation has historically been defined by a trade-off between asset creation and rendering performance. Traditional simulation pipelines, built on explicit mesh representations, can produce geometrically accurate and controllable data but are fundamentally bottlenecked by the need for high-quality, often manually created, 3D assets [6], [7], [8], [9]. This dependency on specialized assets limits their scalability and adaptability to diverse, real-world environments. In response, recent advances in neural fields, particularly Neural Radiance Fields (NeRF), have enabled the reconstruction of scenes directly from sensor data [10]. However, while NeRF-based LiDAR simulators achieve impressive fidelity, their reliance on implicit representations and exhaustive volumetric ray marching renders them computationally intensive, with slow training times and low inference throughput that preclude real-time applications [11], [12], [13], [14], [15].

The emergence of 3D Gaussian Splatting (3DGS) promised to resolve this impasse, offering a transformative representation that combines photorealistic quality with realtime rendering speeds [16], [17], [18], [19]. Yet, the standard 3DGS formulation, optimized for visual fidelity, is ill-suited for geometric sensing tasks. Its rendering process often produces blurred surfaces and incoherent depth estimates, failing to model the precise first-return mechanics of a LiDAR sensor [20], [21], [22], [23], [24]. Recognizing this, a new class of specialized methods has sought to adapt the Gaussian framework specifically for LiDAR simulation [25], [26]. These approaches achieve high physical realism by integrating new, learnable attributes for intensity and raydrop directly into the Gaussian primitives. However, this specialization comes at a cost: they require direct supervision from ground-truth LiDAR data and modify the core 3DGS architecture. This dependency on LiDAR-specific training data fundamentally limits their generality. As a result, the vast and rapidly expanding ecosystem of pre-existing, photometrically-trained 3DGS assets remains incompatible with LiDAR simulation, creating a significant âgenerality gapâ in the field.

To bridge this gap, we introduce FGGS-LiDAR, a framework that establishes a new paradigm for LiDAR simulation. Instead of specializing the scene representation, we propose a general-purpose pipeline that converts arbitrary, off-the-shelf 3DGS assets into a simulation-ready format. Our approach is founded on two core contributions. First, we develop a novel 3DGS-to-Mesh conversion process that reliably produces watertight and topologically-consistent meshes from any pretrained 3DGS model, crucially without requiring dataset-specific supervision, architectural changes, or external priors like COLMAP [27]. Second, we contribute a highly optimized, GPU-accelerated ray-casting module that simulates LiDAR returns from these generated meshes at speeds exceeding 500 FPS, even on complex scenes with millions of triangles. By decoupling the 3DGS scene representation from the LiDAR simulation, our framework provides a truly âplug-and-playâ solution.

Our contributions can be summarized as follows: Our contributions can be summarized as follows:

â¢ We introduce a general, prior-free â3DGS2Meshâ pipeline that converts any pretrained 3DGS model into a high-quality, watertight mesh suitable for geometric simulation.

â¢ We release a high-performance, open-source LiDAR ray-casting module that achieves ultra-fast simulation speeds (>500 FPS) through hardware-conscious optimizations.

â¢ We demonstrate state-of-the-art geometric fidelity, achieving millimeter-level accuracy in reconstructing scene geometry from 3DGS assets, validated through direct comparison with ground-truth scans.

## II. RELATED WORKS

## A. 3DGS for novel view synthesis

3DGS[16] introduced a differentiable splatting algorithm to render Gaussian primitives, establishing a practical and effective framework for novel view synthesis. Subsequent studies[24], [28], [21] extended this paradigm with improvements in anti-aliasing[24] and view-consistent rendering [21]. For example, PGSR [20] flattens 3D Gaussians into planar forms combined with unbiased depth rendering, while incorporating both single and multiple view regularizations to enforce global geometric consistency, thereby improving surface fidelity without compromising efficiency. Other methods such as 2D Gaussian Splatting [22] project volumetric Gaussians into oriented planar disks, offering a more compact and geometrically precise formulation.

## B. Mesh reconstruction from 3DGS

Mesh reconstruction from 3DGS has been investigated as a means to extend Gaussian representations toward geometryoriented applications. Existing methods can be broadly divided into two categories. One line of work, exemplified by frameworks such as GS2Mesh [29] and MILo [30], reconstructs meshes from Gaussian primitives through volumetric sampling, signed distance field (SDF) extraction, and meshing algorithms. Although these pipelines can generate watertight surfaces, they typically rely on auxiliary reconstructions such as COLMAP [31] [32] or multiview stereo to provide camera poses and depth priors. This dependency couples them tightly to the image acquisition and 3DGS training process, preventing direct application to arbitrary pretrained 3DGS models.

## C. Gaussian-based LiDAR simulation

Recent studies have also explored extending Gaussian representations for LiDAR simulation. LiDAR-RT [25] proposed a Gaussian-based ray tracing framework, introducing learnable LiDAR-specific attributes (e.g., reflection intensity and ray-drop probability) into Gaussian primitives. In parallel, GS-LiDAR [26] adopts a panoramic Gaussian projection approach, representing the scene with oscillatory 2D Gaussian primitives and modeling explicit rayâGaussian interactions for LiDAR novel view synthesis. It further incorporates spherical harmonic coefficients to encode LiDAR-specific properties, yielding higher efficiency and fidelity than NeRFbased approaches.

However, these efforts face several limitations. They remain anchored to autonomous-driving datasets and require supervision from raw LiDAR scans, narrowing transferability to other domains.They also alter the standard 3DGS formulation with task-specific extensions, undermining drop-in compatibility with large pretrained 3DGS models. In addition, LiDAR-RTâs ray-traced pipeline is computationally heavy, whereas GS-LiDARâs rasterization lacks an explicit time-offlight model. Collectively, these factors limit scalability and plug-and-play use , motivating a general, efficient, and fully 3DGS compatible solution.

## III. PRELIMINARIES

## A. 3D Gaussian Splatting

3D Gaussian Splatting represents a 3D scene as a collection of anisotropic Gaussian primitives [16]. Each primitive $G _ { i }$ is defined by a set of learnable attributes: a mean position $\mu _ { i } \in \mathbb { R } ^ { 3 }$ , a 3D covariance matrix $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ , an opacity $\sigma _ { i } ~ \in ~ ( 0 , 1 )$ , and spherical harmonics (SH) coefficients to model view-dependent appearance. The influence of the i-th Gaussian on a spatial location $x \in \mathbb { R } ^ { 3 }$ is described by an unnormalized Gaussian function:

$$
G _ { i } ( x ) = \exp \left( - \frac { 1 } { 2 } ( x - \mu _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( x - \mu _ { i } ) \right) ,\tag{1}
$$

where the covariance matrix $\Sigma _ { i }$ is factorized as $\Sigma _ { i } ~ =$ $R _ { i } S _ { i } S _ { i } ^ { \top } R _ { i } ^ { \top }$ , using a scaling matrix $S _ { i }$ and a rotation matrix $R _ { i }$ . Our work takes such a collection of primitives, derived from a pretrained 3DGS model, as the geometric foundation for our simulation pipeline.

## B. Truncated Signed Distance Functions

A Truncated Signed Distance Function (TSDF) is a volumetric data structure used to represent an implicit surface, making it an effective bridge between discrete point or voxel data and a continuous mesh representation [33]. Given a surface ââ¦, the signed distance for any point x in space is defined as:

$$
d ( x ) = \operatorname { s g n } ( x ) \operatorname* { m i n } _ { y \in \partial \Omega } \| x - y \| ,\tag{2}
$$

where the sign is negative if x is inside the surface and positive if it is outside. In practice, computing this distance field across an entire volume is computationally expensive and often unnecessary. The TSDF improves efficiency by restricting the computation to a narrow band of width $\pm \tau$ around the surface:

$$
d _ { \tau } ( x ) = \operatorname* { m a x } ( - \tau , \operatorname* { m i n } ( \tau , d ( x ) ) ) .\tag{3}
$$

This truncation stabilizes the representation by ignoring voxels far from the object boundary. The zero-level set of the resulting TSDF volume, $\{ x | d _ { \tau } ( x ) = 0 \}$ , corresponds to the objectâs surface. This implicit surface can then be efficiently converted into an explicit, watertight polygonal mesh using an isosurface extraction algorithm such as Marching Cubes [34].

## IV. METHODS

Our objective is to develop a pipeline that converts any pretrained 3DGS model G into a high-quality, watertight mesh M suitable for ultra-fast LiDAR simulation. Our approach is a three-stage process: (1) We first discretize the continuous Gaussian representation into a structured, binary occupancy volume, a process accelerated by a Bounding Volume Hierarchy (Sec. IV-A). (2) From this volumetric data, we reconstruct a smooth and topologically consistent implicit surface using a narrow-band TSDF (Sec. IV-B). (3) Finally, we extract the explicit mesh M and perform LiDAR simulation using a highly optimized, GPU-accelerated raycasting engine (Sec. IV-C).

## A. Gaussian to BVH

Overview. To enable geometric reasoning from Gaussian primitives, we discretize continuous representations into voxel grids. It partitions the 3D space $\Omega \bar { \subset } \mathbb { R } ^ { 3 }$ into a regular lattice of cubic cells with spacing $h ,$ where the center of a voxel is $v _ { i j k }$ . It is then assigned a binary occupancy volume $V ( i , j , k )$ , yielding a structured volumetric approximation of the Gaussian scene. For each voxel center $v _ { i j k }$ , we estimate density $D ( v _ { i j k } )$ with an opacity weighting function $f ( \sigma _ { i } )$ and then evaluate its occupancy with the surface mask $\operatorname { S u r f } ( i , j , k )$ .

BVH Construction Given a pretrained 3DGS $\begin{array} { r l } { \mathcal { G } } & { { } = } \end{array}$ $\{ ( \mu _ { i } , q _ { i } , s _ { i } , \alpha _ { i } ) \} _ { i = 1 } ^ { N }$ we build an BVH over per-Gaussian axis-aligned bounding box (AABB). Each Gaussian $G _ { i }$ (center $\mu _ { i }$ , orientation $R _ { i } \equiv R ( q _ { i } )$ , scale $s _ { i } )$ is conservatively enclosed by an AABB $b _ { i } = ( b _ { i } ^ { \operatorname* { m i n } } , b _ { i } ^ { \operatorname* { m a x } } )$ with half-extents

$$
r _ { i } = \kappa | R _ { i } | \ : s _ { i } , \qquad b _ { i } ^ { \operatorname* { m i n } } = \mu _ { i } - r _ { i } , \quad b _ { i } ^ { \operatorname* { m a x } } = \mu _ { i } + r _ { i } ,\tag{4}
$$

where $| R _ { i } |$ is taken elementwise (projecting the oriented axes to world axes) and $\kappa \geq 1$ is a padding factor.

We then sort primitives by Morton codes computed from their centers $\mu _ { i }$ within a scene box $[ o , o + L ]$ using b bits per axis:

$$
\begin{array} { r } { m _ { i } = \mathrm { i n t e r l e a v e } \Big ( \big \lfloor 2 ^ { b } \frac { \mu _ { i , x } - o _ { x } } { L _ { x } } \big \rfloor , ~ \big \lfloor 2 ^ { b } \frac { \mu _ { i , y } - o _ { y } } { L _ { y } } \big \rfloor , ~ \big \lfloor 2 ^ { b } \frac { \mu _ { i , z } - o _ { z } } { L _ { z } } \big \rfloor \Big ) . } \end{array}\tag{5}
$$

Morton code generation and radix sort run massively in parallel. Given sorted codes $\{ m _ { i } \}$ , internal node ranges are determined by the longest common prefix (LCP) of adjacent codes,

$$
\operatorname { L C P } ( i , j ) = \operatorname* { m a x } \{ \ell : \ \mathrm { t h e ~ f i r s t } \ \ell \ \mathrm { b i t s ~ o f } \ m _ { i } \ \mathrm { a n d } \ m _ { j } \ \mathrm { m a t c h } \} ,\tag{6}
$$

these values are computed with bitwise operations, and prefix scans form node ranges without recursion. Parent bounds are reduced in parallel as the union of child bounds:

$$
B ( n ) = B ( n _ { \mathrm { l e f t } } ) \cup B ( n _ { \mathrm { r i g h t } } ) .\tag{7}
$$

<!-- image-->  
Fig. 2: LiDAR visualization across scenes. A 4 Ã 3 panel: each row corresponds to one scene. From left to right, the columns show (1) 3DGS visualization, (2) the voxelized mesh, and (3) the LiDAR point cloud rendered by FGGS-LiDAR.

Node AABBs are reduced bottom-up in parallel, and nodes are laid out in breadth-first or Morton order to maximize coalesced memory access; traversal is stackless, avoiding global stacks. Consequently, BVH construction is near-linear in the number of Gaussians and contributes only a small fraction of the overall voxelization cost, even with millions of primitives.

Grid query and occupancy evaluation. The next step is to convert AABBs in BVH to a binary occupancy volume V for mesh reconstruction. We partition the volumetric grid into tiles of size $B ^ { 3 }$ , and for a tile with AABB $\boldsymbol { B } _ { \mathrm { t i l e } }$ , we cull candidates by intersecting per-Gaussian AABBs $b _ { i }$ :

$$
{ \mathcal { C } } _ { \mathrm { t i l e } } = \{ G _ { i } \ | \ b _ { i } \cap B _ { \mathrm { t i l e } } \neq \emptyset \} .\tag{8}
$$

Each voxel $v \in \mathcal { V } _ { \mathrm { t i l e } }$ accumulates density only over $\mathcal { C } _ { \mathrm { t i l e } } \mathrm { : }$

$$
D ( v ) = \sum _ { G _ { i } \in \mathcal C _ { \mathrm { i i } } } \exp \Bigl ( - \frac 1 2 \bigl ( v - \mu _ { i } \bigr ) ^ { \top } \Sigma _ { i } ^ { - 1 } \bigl ( v - \mu _ { i } \bigr ) \Bigr ) f ( \sigma _ { i } ) .\tag{9}
$$

Finally, we create the binary occupancy volume from D(v). A voxel is occupied if

$$
V ( i , j , k ) = \left\{ \begin{array} { l l } { 1 , } & { D ( v _ { i j k } ) > \theta , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{10}
$$

where Î¸ is a user-defined density threshold that controls the trade-off between completeness and sparsity of the voxelized geometry.

We check whether a voxel is interior by looking at its

<!-- image-->  
Fig. 3: From Gaussian primitives to surface voxels. We first organize 3D Gaussian primitives into an BVH structure for efficient spatial indexing. The scene is then voxelized within the global AABB, followed by occupancy thresholding and filtering to obtain a clean volumetric representation. Finally, surface voxels are extracted to yield contour points that approximate the underlying scene geometry.

6-neighborhood:

$$
\begin{array} { r l } & { \mathrm { I n t } ( i , j , k ) = V _ { i , j , k } \wedge V _ { i - 1 , j , k } \wedge V _ { i + 1 , j , k } } \\ & { \qquad \wedge V _ { i , j - 1 , k } \wedge V _ { i , j + 1 , k } \wedge V _ { i , j , k - 1 } \wedge V _ { i , j , k + 1 } , } \end{array}\tag{11}
$$

With it, we create a surface mask to apply over the occupancy volume $V _ { i , j , k } \mathrm { : }$

$$
\operatorname { S u r f } ( i , j , k ) = V _ { i , j , k } \wedge \neg \operatorname { I n t } ( i , j , k ) .\tag{12}
$$

B. Mesh Reconstruction

Overview. Given a binary occupancy volume $V ~ : ~ \Omega \ \to ~ \{ 0 , 1 \}$ with voxel spacing $\mathrm { ~ { ~ \bf ~ s ~ } ~ } = \mathrm { ~ \bf ~ ( ~ } s _ { x } , s _ { y } , s _ { z } \mathrm { ) }$ and origin o, we reconstruct a watertight surface by (i) denoising and re-thresholding the binary field, (ii) building a narrow-band TSDF with reliable sign via outside flood-fill, (iii) extracting an isosurface with Marching Cubes, and (iv) performing structure-first simplification followed by non-shrinking smoothing.

Binary denoising and re-thresholding. To suppress saltand-pepper artifacts while remaining resolution-agnostic, we convolve V with a Gaussian kernel of metric scale Ï (expressed in meters and mapped to voxel units via s):

$$
V ^ { \prime } ( x ) = ( G _ { \sigma } * V ) ( x ) .\tag{13}
$$

A denoised occupancy $\tilde { V }$ is obtained by either a fixed threshold Ï or a quantile threshold $q \colon$

$$
\tilde { V } _ { \mathrm { f i x e d } } ( x ) = \left\{ \begin{array} { l l } { 1 , } & { V ^ { \prime } ( x ) \ge \tau , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{14a}
$$

$$
\tilde { V } _ { \mathrm { q u a n t } } ( x ) = \left\{ \begin{array} { l l } { 1 , } & { V ^ { \prime } ( x ) \geq \mathrm { Q u a n t i l e } _ { q } ( V ^ { \prime } ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{14b}
$$

Narrow-band TSDF. Given a binary occupancy grid $V ~ : ~ \Omega ~  ~ \{ 0 , 1 \}$ , we construct a signed distance field $\phi : \Omega \to [ - r , r ]$ in three logical stages. We first assign signs by identifying the outside region: a flood-fill is performed over free voxels, seeded from a padded frame Î surrounding the domain, so that the 6-connected component O connected to Î is labeled as outside. Voxels in O are assigned $s ( x ) =$ +1, while all others (occupied cells or enclosed voids) are assigned $s ( x ) = - 1$ , ensuring stable sign labeling even in the presence of cavities and tunnels.

Next, unsigned distances are computed by layered propagation from the boundary set

$$
S _ { 0 } = \{ x \mid \exists y \in N _ { 6 } ( x ) , V ( x ) \neq V ( y ) \} ,\tag{15}
$$

where each expansion shell $S _ { m }$ grows over the 6- neighborhood and newly visited voxels record their firstarrival shell index $\kappa ( x )$ . The unsigned distance is then approximated as

$$
\delta ( x ) = \kappa ( x ) v _ { \mathrm { m i n } } , \qquad v _ { \mathrm { m i n } } = \mathrm { m i n } ( s _ { x } , s _ { y } , s _ { z } ) ,\tag{16}
$$

which provides a conservative lower bound of the Euclidean distance and prevents diagonal leakage on anisotropic grids.

Finally, the signed distance field is obtained by combining the seeded sign and unsigned distance with truncation,

$$
\phi ( x ) = \mathrm { c l i p } \big ( s ( x ) \delta ( x ) , - r , r \big ) ,\tag{17}
$$

which restricts values to the radius-r band and avoids the memory and time overhead of a global Euclidean distance transform. All steps are executed fully on the GPU, where flood-fill, layered expansions, and truncation are carried out in parallel. By leveraging massive parallelism, the method achieves high efficiency and scales effectively to large volumes, avoiding the overhead of full-grid distance transforms.

Isosurface extraction. Next we apply isosurface extraction to get a smooth surface from the grid. The target surface is the level set

$$
S = \{ x \in \Omega \mid \phi ( x ) = \mathrm { i s o } \} ,\tag{18}
$$

with iso = 0 by default (optional millimeter-scale bias Â±iso). We discretize S via Marching Cubes with step-size parameter mc step and map vertices to world coordinates using o and s. Per-vertex normals are estimated from âÏ for consistent outward orientation, yielding a watertight raw mesh $\mathcal { M } _ { \mathrm { r a w } }$

Mesh optimization. To make the reconstruction scalable for large scenes, we first apply structure-prioritized simplification to obtain $\mathcal { M } _ { \mathrm { s i m p } }$ , targeting a prescribed face count or ratio while protecting boundaries and removing tiny components. We then perform non-shrinking smoothing (e.g., Taubin) on $\mathcal { M } _ { \mathrm { s i m p } }$ with parameters $( \lambda , \mu )$ and a small number of iterations, reducing staircase/normal noise while preserving sharp features and thin walls:

$$
\mathcal { M } _ { \mathrm { f i n a l } } = \mathrm { S m o o t h } _ { \lambda , \mu , T } \big ( \mathrm { S i m p l i f y } ( \mathcal { M } _ { \mathrm { r a w } } ) \big ) .
$$

Unless stated otherwise, all scale parameters (Ï, r, iso) are specified in meters through s, decoupling control from voxel resolution.

<!-- image-->  
Fig. 4: From surface voxels to refined mesh reconstruction. The pipeline starts from surface voxels, which are denoised and smoothed through filtering and thresholding. We then construct a TSDF to extract an isosurface using the Marching Cubes algorithm for generating an initial mesh. Finally, mesh simplification and smoothing are applied to obtain the refined mesh representation.

## C. Ray-casting for LiDAR Simulation

Ray to mesh measurement. In a LiDAR scan, the j-th beam is modeled as a ray

$$
r _ { j } ( t ) = x _ { s } + t d _ { j } , \quad t \geq 0 ,\tag{19}
$$

where $T _ { s } = \left\lceil \mathbf { 0 } _ { s } \quad t _ { s } \right\rceil \in S E ( 3 )$ is the sensor pose in world coordinates, $x _ { s } : = t _ { s }$ is the beam origin, and $d _ { j } \in \mathbb { S } ^ { 2 }$ is a unit direction determined by the scanning pattern. The environment is represented as a triangle mesh

$$
\begin{array} { r } { \mathcal { M } = \{ \triangle _ { k } = ( v _ { k , 0 } , v _ { k , 1 } , v _ { k , 2 } ) \} _ { k = 1 } ^ { T } . } \end{array}
$$

For each ray, the LiDAR return corresponds to the nearest intersection

$$
\begin{array} { r } { t _ { j } ^ { \star } = \operatorname* { m i n } \Big \{ \tau ( r _ { j } , \triangle _ { k } ) \ \Big | \ 1 \le k \le T , \ \tau \in [ t _ { \operatorname* { m i n } } , t _ { \operatorname* { m a x } } ] \Big \} , } \end{array}\tag{20}
$$

where $\tau ( r _ { j } , \triangle _ { k } )$ is the intersection parameter with triangle $\triangle _ { k }$ . The measured range equals $\rho _ { j } = t _ { j } ^ { \star }$ since $\| d _ { j } \| = 1$

GPU-accelerated ray-casting. We implement ray-casting entirely on the GPU by assigning one thread to each LiDAR beam, thereby transforming the inherently independent nature of beam propagation into massive parallelism. In our design, every thread traverses the preconstructed BVH and typically visits only O(log T ) nodes before reaching a small set of candidate triangles. Traversal is strictly guided by the best-so-far depth $t _ { j } ^ { \star } \colon$ nodes whose entry distance exceeds $t _ { j } ^ { \star }$ are discarded together with their subtrees, and candidate triangles lying beyond this threshold are likewise excluded. This early-termination mechanism ensures that computation remains focused only on geometrically relevant regions.

Formally, the per-ray work can be written as

$$
\mathrm { c o s t } ( r _ { j } ) = C _ { \mathrm { t r a v } } \ N _ { \mathrm { n o d e s } } ( r _ { j } ) + C _ { \mathrm { t r i } } \ K _ { j } ,\tag{21}
$$

where $C _ { \mathrm { t r a v } }$ and $C _ { \mathrm { t r i } }$ are the costs of a rayâAABB and rayâ triangle test, $N _ { \mathrm { n o d e s } } ( r _ { j } )$ is the number of BVH nodes visited, and $K _ { j }$ is the number of triangles tested at leaves. Summing over all rays gives

$$
\mathcal { C } _ { \mathrm { G P U } } = \sum _ { j = 1 } ^ { N _ { r } } \mathrm { c o s t } ( r _ { j } ) \approx \mathcal { O } ( N _ { r } \cdot ( \log T + \overline { { K } } ) ) ,\tag{22}
$$

with $\begin{array} { r } { \overline { { K } } \ = \ \frac { 1 } { N _ { r } } \sum _ { j } K _ { j } \ \ll \ T } \end{array}$ in practice. This stands in contrast to the naive baseline

$$
\mathcal { C } _ { \mathrm { n a i v e } } = \mathcal { O } ( N _ { r } \cdot T ) ,
$$

and explains why the GPU design achieves near-logarithmic scaling and millisecond-level simulation time even on million-triangle meshes.

Beyond hierarchical pruning, we incorporate several architectural optimizations to maximize GPU efficiency. Mesh vertices, triangle indices, and BVH bounds are organized in a structure-of-arrays layout, enabling coalesced memory accesses across warps. Frequently reused node bounds are cached in shared memory, which substantially reduces global memory traffic. Warp-synchronous traversal further enforces execution coherence, mitigating branch divergence among neighboring rays. Since beam queries are fully independent, the nearest-hit depth $t _ { j } ^ { \star }$ and the corresponding intersection point $\boldsymbol { x } _ { j } ^ { \star }$ are written directly to global output buffers in a lock-free manner. Collectively, these design choices integrate algorithmic pruning with hardware-conscious optimization, ensuring that traversal, intersection, and memory access are all jointly accelerated, and rendering LiDAR-scale simulation feasible at millisecond latency even for million-triangle meshes.

## V. EXPERIMENTS

## A. Experiment Setup

Datasets. We evaluate on two benchmark scenes: one indoor and one outdoor. For each scene, we acquire a ground-truth (GT) watertight mesh via real LiDAR scanning followed by SLAM-based reconstruction, and we prepare a corresponding 3DGS asset of the same scene. All assets are registered to a common world frame and share identical LiDAR extrinsics. From a fixed sensor pose and scanning pattern, we render one LiDAR frame per scene for each of three sensor configurations (HDL64, OS128, VLP32), matching the beam layouts used in the tables.

Competitors. We compare two sources of geometry. Our method converts off-the-shelf 3DGS assets into a watertight mesh with our pipeline and renders a first-hit LiDAR frame under the fixed pose and scan pattern. The GT baseline renders a LiDAR frame from the mesh reconstructed from real LiDAR scans using the same pose and scan pattern. All point clouds are evaluated in the same coordinate frame.

Metrics. We report symmetric Chamfer Distance (CD; lower is better) and F-score, Precision, and Recall at standard distance thresholds (indoor and outdoor thresholds differ). Results are averaged over the two scenes and over the three LiDAR configurations. Distances are measured in meters.

Implementation details. All experiments are conducted on a workstation with an Intel W3545 CPU at 3.2 GHz and an NVIDIA RTX 4090 GPU. All timings and throughputs reported in this section are measured on this machine.

TABLE I Quantitative comparison on indoor LiDAR simulation. Results are averaged over three LiDAR types (HDL64, OS128, VLP32) by comparing simulated and ground-truth point clouds.
<table><tr><td>LiDAR</td><td>CD â</td><td>F-score â</td><td>Precision â</td><td>Recall â</td></tr><tr><td>HDL64</td><td>0.0034</td><td>0.9950</td><td>0.9985</td><td>0.9916</td></tr><tr><td>OS128</td><td>0.0034</td><td>0.9950</td><td>0.9956</td><td>0.9944</td></tr><tr><td>VLP32</td><td>0.0053</td><td>0.9918</td><td>0.9964</td><td>0.9872</td></tr><tr><td>Avg.</td><td>0.0041</td><td>0.9939</td><td>0.9968</td><td>0.9911</td></tr></table>

TABLE II Quantitative comparison on outdoor LiDAR simulation. Results are averaged over three LiDAR types (HDL64, OS128, VLP32) by comparing simulated and ground-truth point clouds.
<table><tr><td>LiDAR</td><td>CD â</td><td>F-score â</td><td>Precision â</td><td>Recall â</td></tr><tr><td>HDL64</td><td>0.0157</td><td>0.9816</td><td>0.9826</td><td>0.9806</td></tr><tr><td>OS128</td><td>0.0104</td><td>0.9867</td><td>0.9883</td><td>0.9853</td></tr><tr><td>VLP32</td><td>0.0250</td><td>0.9789</td><td>0.9844</td><td>0.9736</td></tr><tr><td>Avg.</td><td>0.0170</td><td>0.9824</td><td>0.9851</td><td>0.9798</td></tr></table>

## B. Comparisons with Ground Truth

Indoor. Table I reports results for the indoor scene. Our simulation closely matches the GT-scan mesh, with CD in the 3â5 mm range (avg. 4.07 mm), mean F-score 0.994, Precision 0.997, and Recall 0.991. Among beam layouts, OS128 attains the lowest CD (0.003432 m) and the highest Recall (0.994356), while HDL64 yields the top F-score (0.995043) and Precision (0.9985). VLP32 is consistently weakerâexpected from its sparser vertical samplingâyet still remains within a few millimeters of GT.

<!-- image-->  
Fig. 5: LiDAR frame rate vs. mesh complexity.

Outdoor. Table II summarizes the outdoor scene. Errors increase but remain small: average CD is 17.0 mm with Fscore 0.982, Precision 0.985, and Recall 0.980. OS128 dominates across metrics (CD 0.010429 m; F-score 0.986726), reflecting the benefit of denser beams for long-range structure and complex occlusions. Compared with indoor, the gap is mainly due to longer ranges and clutter; nonetheless, the averages indicate our pipeline produces ranges that approach those rendered from meshes reconstructed by real LiDAR scans across distinct LiDAR geometries.

## C. LiDAR Simulation Performance

We evaluate the simulation performance of four different LiDAR sensor configurations across six distinct 3DGS scenes; the horizontal axis in the plots corresponds to the number of faces (triangles) in the converted mesh model.

Frame rate. As shown in Fig. 5, owing to the BVH acceleration structure, across all evaluated 3DGS scenes (up to 6M Gaussian primitives) we observe no systematic degradation of simulation frame rate with increasing primitive count. This indicates that, within this scale, performance is effectively decoupled from the raw number of primitives. Instead, traversal efficiency is primarily governed by spatial distribution characteristicsâsuch as clustering patterns, local density heterogeneity, and occlusion layeringârather than absolute cardinality. This demonstrates that our work achieves far beyond real-time performance, ranking as the fastest among peer methods, and highlights the efficiency advantage of the proposed pipeline.

Throughput. Fig. 6 reports point throughput as a function of mesh complexity. Despite lower frame rates, denser sensors (OS128, VLP32) in our work achieve significantly higher throughput, exceeding $1 0 ^ { 8 }$ points/s on lightweight meshes and sustaining $> 7 . 5 \times 1 0 ^ { 7 }$ points/s even at multimillion triangle scales. This indicates that our method fully exploits BVH-accelerated parallelism and maintains high utilization across diverse LiDAR types.

## D. Comparison with 3DGS Depth

Comparison of Depth and LiDAR Imaging Principles. Depth in 3D Gaussian Splatting is computed as an opacityweighted expectation along the camera ray, yielding viewdependent averages rather than true geometric intersections.

<!-- image-->  
Fig. 6: LiDAR throughput vs. mesh complexity.

This causes two characteristic artifacts: edges and thin structures are blurred or widened due to splat blending, and underconstrained regionsâsuch as back-facing or rarely observed surfaces, low-texture areas, and glossy materialsâmanifest as dropouts or holes in rendered depth maps (see Fig. 7). LiDAR, in contrast, reports the first-return geometric distance, a physically grounded metric that directly corresponds to scene geometry and avoids averaging artifacts.

Computational Efficiency. Depth rendering in 3D Gaussian Splatting requires per-pixel accumulation over many anisotropic Gaussians, with cost and memory scaling with both image resolution and splat footprint, making highresolution rendering computationally expensive. LiDAR rendering instead performs BVH-accelerated rayâtriangle intersections on a mesh. Its cost grows mainly with beam and triangle count, and the independence of rays allows highly efficient parallelization.

<!-- image-->  
Fig. 7: Edge blurring and depth dropouts in 3DGS depth maps. The figure illustrates blurred object boundaries and missing depth values in 3DGS-rendered maps, indicating limitations in geometric accuracy and stability.

## VI. CONCLUSION

We present FGGS-LiDAR, an ultra-fast GPU-accelerated LiDAR simulation framework for general 3DGS assets. Our fully GPU-resident framework operates directly on off-the-shelf 3DGS models without LiDAR supervision or COLMAP-style priors by passing 3DGS assets through BVH-based volumetric discretization and narrowband TSDF, followed by isosurface extraction to create a watertight surface. We then perform BVH-accelerated per-ray first-hit ranging in the LiDAR spherical image and achieve over 500 FPS for 200k+ rays in a 6M+ triangle scene. In both indoor and outdoor scenarios, LiDAR simulations based on meshes modeled by our method exhibit strong agreement with those derived from real-scanned meshes, as reflected by average Chamfer Distances of 4.07 mm indoors and 17.0 mm outdoors, alongside mean F-scores of 0.994 and 0.982, respectively. In addition, our method can serve as a plugin LiDAR module for 3DGS-based simulators, integrating into existing pipelines to support large-scale sensor data generation.

Remaining limitations include residual internal cavities at high voxel resolutions, depending on the quality of the 3DGS assets, and high GPU memory usage; future work will improve memory scalability, robustness to low-quality assets, and add more realistic LiDAR sensor physics with closed-loop evaluation for dynamic scenes.

## REFERENCES

[1] Y. Zhang, P. Shi, and J. Li, âLidar-based place recognition for autonomous driving: A survey,â ACM Computing Surveys, vol. 57, no. 4, pp. 1â36, 2024.

[2] H. Yin, X. Xu, S. Lu, X. Chen, R. Xiong, S. Shen, C. Stachniss, and Y. Wang, âA survey on global lidar localization: Challenges, advances and open problems,â International Journal of Computer Vision, vol. 132, no. 8, pp. 3139â3171, 2024.

[3] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, âFast-lio2: Fast direct lidar-inertial odometry,â IEEE Transactions on Robotics, vol. 38, no. 4, pp. 2053â2073, 2022.

[4] A. Charroud, K. El Moutaouakil, V. Palade, A. Yahyaouy, U. Onyekpe, and E. U. Eyo, âLocalization and mapping for self-driving vehicles: a survey,â Machines, vol. 12, no. 2, p. 118, 2024.

[5] M. Savva, A. X. Chang, A. Dosovitskiy, T. Funkhouser, and V. Koltun, âMinos: Multimodal indoor simulator for navigation in complex environments,â arXiv preprint arXiv:1712.03931, 2017.

[6] S. Manivasagam, S. Wang, K. Wong, W. Zeng, M. Sazanovich, S. Tan, B. Yang, W.-C. Ma, and R. Urtasun, âLidarsim: Realistic lidar simulation by leveraging the real world,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11167â11176, 2020.

[7] S. Shah, D. Dey, C. Lovett, and A. Kapoor, âAirsim: High-fidelity visual and physical simulation for autonomous vehicles,â in Field and service robotics: Results of the 11th international conference, pp. 621â 635, Springer, 2017.

[8] C. Li, Y. Ren, and B. Liu, âPcgen: Point cloud generator for lidar simulation,â arXiv preprint arXiv:2210.08738, 2022.

[9] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, âCarla: An open urban driving simulator,â in Conference on robot learning, pp. 1â16, PMLR, 2017.

[10] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â 106, 2021.

[11] Z. Zheng, F. Lu, W. Xue, G. Chen, and C. Jiang, âLidar4d: Dynamic neural fields for novel space-time view lidar synthesis,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5145â5154, 2024.

[12] Z. Yang, Y. Chen, J. Wang, S. Manivasagam, W.-C. Ma, A. J. Yang, and R. Urtasun, âUnisim: A neural closed-loop sensor simulator,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1389â1399, 2023.

[13] W. Xue, Z. Zheng, F. Lu, H. Wei, G. Chen, et al., âGeonlf: Geometry guided pose-free neural lidar fields,â Advances in Neural Information Processing Systems, vol. 37, pp. 73672â73692, 2024.

[14] J. Zhang, F. Zhang, S. Kuang, and L. Zhang, âNerf-lidar: Generating realistic lidar point clouds with neural radiance fields,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 7178â 7186, 2024.

[15] S. Huang, Z. Gojcic, Z. Wang, F. Williams, Y. Kasten, S. Fidler, K. Schindler, and O. Litany, âNeural lidar fields for novel view synthesis,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18236â18246, 2023.

[16] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, pp. 139:1â139:14, 2023.

[17] Y. Liu, C. Luo, L. Fan, N. Wang, J. Peng, and Z. Zhang, âCitygaussian: Real-time high-quality large-scale scene rendering with gaussians,â in European Conference on Computer Vision, pp. 265â282, Springer, 2024.

[18] G. Feng, S. Chen, R. Fu, Z. Liao, Y. Wang, T. Liu, B. Hu, L. Xu, Z. Pei, H. Li, et al., âFlashgs: Efficient 3d gaussian splatting for largescale and high-resolution rendering,â in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26652â26662, 2025.

[19] M. T. I. SpatialVerse Research Team, âInteriorgs: A 3d gaussian splatting dataset of semantically labeled indoor scenes.â https://huggingface.co/datasets/spatialverse/ InteriorGS, 2025.

[20] D. Chen, H. Li, W. Ye, Y. Wang, W. Xie, S. Zhai, N. Wang, H. Liu, H. Bao, and G. Zhang, âPgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction,â IEEE Transactions on Visualization and Computer Graphics, 2024.

[21] L. Radl, M. Steiner, M. Parger, A. Weinrauch, B. Kerbl, and M. Steinberger, âStopthepop: Sorted gaussian splatting for view-consistent realtime rendering,â ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1â17, 2024.

[22] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2d gaussian splatting for geometrically accurate radiance fields,â in SIGGRAPH 2024 Conference Papers, Association for Computing Machinery, 2024.

[23] Z. Qian, S. Wang, M. Mihajlovic, A. Geiger, and S. Tang, â3dgsavatar: Animatable avatars via deformable 3d gaussian splatting,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5020â5030, 2024.

[24] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, âMip-splatting: Alias-free 3d gaussian splatting,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19447â 19456, 2024.

[25] C. Zhou, L. Fu, S. Peng, Y. Yan, Z. Zhang, Y. Chen, J. Xia, and X. Zhou, âLiDAR-RT: Gaussian-based ray tracing for dynamic lidar re-simulation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[26] J. Jiang, C. Gu, Y. Chen, and L. Zhang, âGs-lidar: Generating realistic lidar point clouds with panoramic gaussian splatting,â in International Conference on Learning Representations (ICLR), 2025.

[27] J. L. Schonberger, T. Price, T. Sattler, J.-M. Frahm, and M. Pollefeys, Â¨ âA vote-and-verify strategy for fast spatial verification in image retrieval,â in Asian Conference on Computer Vision (ACCV), 2016.

[28] Y. Xu, K. Ye, T. Shao, and Y. Weng, âAnimatable 3d gaussians for modeling dynamic humans,â Frontiers of Computer Science, vol. 19, no. 9, p. 199704, 2025.

[29] Y. Wolf, A. Bracha, and R. Kimmel, âGs2mesh: Surface reconstruction from gaussian splatting via novel stereo views,â in European Conference on Computer Vision, pp. 207â224, Springer, 2024.

[30] A. GuAË Sdon, D. Gomez, N. Maruani, B. Gong, G. Drettakis, and Ë M. Ovsjanikov, âMilo: Mesh-in-the-loop gaussian splatting for detailed and efficient surface reconstruction,â arXiv preprint arXiv:2506.24096, 2025.

[31] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â Â¨ in Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[32] J. L. Schonberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, âPixel- Â¨ wise view selection for unstructured multi-view stereo,â in European Conference on Computer Vision (ECCV), 2016.

[33] B. Curless and M. Levoy, âA volumetric method for building complex models from range images,â in Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pp. 303â 312, 1996.

[34] W. E. Lorensen and H. E. Cline, âMarching cubes: A high resolution 3d surface construction algorithm,â in Seminal graphics: pioneering efforts that shaped the field, pp. 347â353, 1998.