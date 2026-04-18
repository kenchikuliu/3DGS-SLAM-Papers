# GaussianPlant: Structure-aligned Gaussian Splatting for 3D Reconstruction of Plants

Yang Yang1 Risa Shinoda1 Hiroaki Santo1 Fumio Okura1

1The University of Osaka

AbstractâWe present a method for jointly recovering the appearance and internal structure of botanical plants from multiview images based on 3D Gaussian Splatting (3DGS). While 3DGS exhibits robust reconstruction of scene appearance for novel-view synthesis, it lacks structural representations underlying those appearances (e.g., branching patterns of plants), which limits its applicability to tasks such as plant phenotyping. To achieve both high-fidelity appearance and structural reconstruction, we introduce GaussianPlant, a hierarchical 3DGS representation, which disentangles structure and appearance. Specifically, we employ structure primitives (StPs) to explicitly represent branch and leaf geometry, and appearance primitives (ApPs) to the plantsâ appearance using 3D Gaussians. StPs represent a simplified structure of the plant, i.e., modeling branches as cylinders and leaves as disks. To accurately distinguish the branches and leaves, StPâs attributes (i.e., branches or leaves) are optimized in a self-organized manner. ApPs are bound to each StP to represent the appearance of branches or leaves as in conventional 3DGS. StPs and ApPs are jointly optimized using a re-rendering loss on the input multi-view images, as well as the gradient flow from ApP to StP using the binding correspondence information. We conduct experiments to qualitatively evaluate the reconstruction accuracy of both appearance and structure, as well as real-world experiments to qualitatively validate the practical performance. Experiments show that the GaussianPlant achieves both high-fidelity appearance reconstruction via ApPs and accurate structural reconstruction via StPs, enabling the extraction of branch structure and leaf instances.

## I. INTRODUCTION

R ECOVERING the internal structure of objects fromimages, i.e., estimating underlying structures that support their shape and appearance, is a long-standing yet underexplored challenge in computer graphics (CG) and vision (CV). Among various object categories, plants pose particularly challenging cases due to their complex branching architectures and dense foliage, where substantial portions of the internal structure are occluded. Accurately reconstructing plant branch skeletons and leaf instances is crucial for applications such as high-throughput phenotyping, growth modeling, botanical analysis, and creating CG assets. Nevertheless, the recovery of these internal structural components of plants has not yet been addressed.

Traditional methods for reconstructing plant structures typically rely on explicit geometric models or hand-made rules. Skeletonization from 3D point clouds (e.g., [1], [2]) typically requires high-quality 3D scans and extensive manual pruning to remove wrongly generated branches. Procedural modeling, such as L-systems [3], requires carefully tuned species-specific parameters, which hinders scalability across plants with vastly different morphologies. These limitations highlight the need for methods that can directly capture plant structures from RGB observations, without relying on dense 3D supervision or species-specific priors.

<!-- image-->  
Fig. 1. GaussianPlant reconstructs both appearance and 3D structure of plants using a hierarchical 3D Gaussian Splatting (3DGS) representation, while maintaining the high capability of novel-view appearance synthesis of 3DGS. Our method enables plant-aware tasks such as branch structure extraction and leaf-wise 3D reconstruction.

To reconstruct both the structure and appearance of plants without species-specific training or manual design, we introduce GaussianPlant, a hierarchical 3D Gaussian Splatting (3DGS) [4], [5], [6] representation specifically designed for plants. Leveraging the recent evolution of the family of 3DGS, the core idea of the GaussianPlant is to disentangle and jointly optimize plantsâ structure and appearance from given multi-view images in a self-organized manner. Specifically, GaussianPlant separates the representation into two complementary tiers. The structure primitive (StP) is invisible low-frequency Gaussian primitives, which are converted to cylinders or disks, representing a part of a branch or a leaf, respectively, which form the plantâs structure. On the other hand, the appearance primitive (ApP) is a visible Gaussian primitive densely sampled and bound to the surfaces of a StP, capturing fine appearance and geometry details.

A key aspect of our method is the gradient flow between appearance and structure primitives: while ApPs are optimized using photometric re-rendering losses as in standard 3DGS, StPs are refined indirectly through the motion of their bound ApPs, enabling invisible structural primitives to benefit from appearance cues. This joint optimization enables ApPs to progressively shape the geometry of StPs and recover internal structural components, such as branches and leaves, from multi-view RGB observations.

Specifically, we begin by classifying ordinary 3DGS primitives into branch and leaf StPs, and attach high-frequency

ApPs to their surfaces. We then jointly optimize the geometry and semantic labels (i.e., branch vs. leaf) of StPs along with the geometry and appearance parameters of ApPs. The process is guided by complementary cues: an appearance-geometric cue derived from the binding relationships between StPs and ApPs, and a semantic cue distilled from pretrained visionlanguage models. We also optimize the branchesâ tree graph structure, considering smoothness and connectivity, which aims to recover partially occluded branches. From the final output by GaussianPlant, we can easily cluster leaf-labeled StPs to obtain leaf instances as well as extract a structural branch graph from branch StPs.

Experiments using both indoor and outdoor real-world plants show that the GaussianPlant simultaneously achieves robust recovery of branch structure, leaf instances, and finegrained appearance rendering, as shown in Fig. 1. This supports applications such as structural, instance-level analysis and editing, directly benefiting plant-related and CG-oriented applications.

Contributions. The chief contribution of this paper is as follows:

â¢ We introduce a hierarchical 3DGS-based representation that disentangles coarse structures (i.e., StPs) and finegrained appearances (i.e., ApPs) specialized for 3D plant reconstruction. To jointly optimize them, we combine photometric supervision with geometry-aware and semantic-aware binding from an ApP to a StP, allowing for a gradient flow to update the StPâs geometry and attributes (i.e., branch or leaf).

â¢ We introduce a set of complementary regularizers, driven by color, semantics, and structure, that guide our model toward self-organized leaf/branch segmentation and a faithful, gap-free branching structure, without requiring manual labeling.

â¢ We capture a new dataset containing real-world plants to evaluate our methods, as well as establish a benchmark for future 3D plant structure recovery and related research.

Our implementation and the benchmark dataset will be made publicly available.

## II. RELATED WORK

## A. 3D Reconstruction of Plants

A wide variety of methods have been developed to recover the shape and structure of plants, such as leaf, branch, and root shapes from 3D observations or images [7], [8], [9], [10]. Classical photogrammetry pipelines use structure from motion (SfM) [11] and multi-view stereo (MVS) [12] to reconstruct plant-level meshes or point clouds. However, repetitive textures and severe occlusions often lead to holes or noisy geometry. Volumetric silhouettes overcome some occlusion but are limited by voxel resolution unless one resorts to costly octree optimizations [13].

Recent surveys [14], [15] on plant 3D reconstruction summarize a spectrum from classical multi-view geometry, highlighting the need for structural priors to recover thin branches under occlusion. From the recovered 3D shapes, conventional skeletonization and graph-based approaches extract branching structure as a graph via shortest-path or minimum-spanningtree (MST) algorithms (e.g., [16], [17], [18], [19]), which assume high-quality 3D point clouds as input (e.g., captured via LiDAR) and often break down under dense foliage. Recently, Masks-to-Skeleton [20] proposes a method to estimate a 3D tree skeleton directly from multi-view segmentation masks and a mask-guided graph optimization, which still relies on occlusion-free 2D branch masks. Closer to our setting, Isokane et al. [7] infer partly hidden branch graphs from multi-view images via a generative model that reasons about occluded branches; however, they rely on a pretrained image-to-image (i.e., leafy-to-branch) translation network that predicts per view, which is not capable of generalization across different plant species and makes cross-view consistency fragile.

In contrast, our method leverages the self-organized optimization of a hierarchical 3DGS representation, jointly yielding a photorealistic appearance and an underlying structure, without relying on species-specific training, occlusion-free branch masks, and post-hoc skeletonization.

## B. 3D Gaussian Splatting (3DGS)

3DGS [4] is an emerging technique for representing and rendering 3D scenes using anisotropic Gaussian primitives. 3DGS models a scene as a collection of 3D Gaussians, each parameterized by its position, scale, orientation, opacity, and spherical harmonics for view-dependent appearance. This representation enables efficient and high-quality rendering, demonstrating significant advantages in real-time novel-view synthesis compared to NeRF-based methods (e.g., [21]). 3DGS has been widely applied in various tasks, including SLAM [22], text-to-3D generation [23], human avatar modeling [24], and dynamic scene reconstruction [25].

## C. 3DGS for Geometry Reconstruction

Recent efforts have explored the use of 3DGS for explicit surface extraction. SuGaR [26] regularizes Gaussian locations and orientations, ensuring that they remain on and are well-distributed along surfaces. By refining an initial Poisson-reconstructed mesh, SuGaR efficiently optimizes a high-quality surface mesh in under an hour. 2D Gaussian splatting [6] takes essentially similar approach, locating 2D Gaussian primitives in a surface-aligned manner. However, while these surface-aligned approaches may successfully reconstruct the macro-scale surface of plants, they are fundamentally limited in resolving their fine-grained internal structure. Meanwhile, the primary goal of these methods is the surface shape reconstruction, which cannot directly be used for the reconstruction of the internal structure of objects (e.g., human and plant skeletons).

## D. 3DGS for Plants

3DGS has been used for plant-specific applications. Splanting [27] introduces a fast capture pipeline for plant phenotyping based on 3DGS. Wheat3DGS [28] combines 3DGS with Segment Anything [29] to build high-fidelity point-based reconstructions of field wheat plots and to segment hundreds of individual heads for bulk trait measurement. GrowSplat [30] constructs temporal digital twins of plants by combining 3DGS with a temporal registration. Although not focusing on reconstruction, PlantDreamer [31] generates realistic plants using diffusion-guided Gaussian splatting. While these methods advance plant-specific 3DGS pipelines, their primary focus is appearance reconstruction or generation. In contrast, our approach aims to go beyond appearance fidelity by explicitly recovering the plantâs internal structure.

<!-- image-->  
Fig. 2. Overview of GaussianPlant. Starting from a 3DGS point cloud, we cluster points into StPs and approximate them with cylinders and disks, respectively representing branches and leaves. ApPs are densely located on the surface of StPs, which roles for appearance rendering similar to ordinary 3DGS. Our method jointly optimizes structure (i.e., StPs) and appearance (i.e., ApPs) using binding information between StPs and ApPs. 1) Appearancegeometry flow: Photometric loss computed on ApPs is propagated to StPs to update both ApPsâ appearance and StPsâ position. 2) Semantic flow: We compare the DINO features of the input images with $\mathrm { \Delta A p \bar { P s } ^ { \prime } }$ attributes, which are also propagated to StPs to optimize the branch-leaf labels in a self-organized manner. 3) Structure flow: The structure graph extracted from StPs is evaluated to ensure the StPs are aligned to a tree graph, which optimizes the tree graph, enabling the extraction of branching structure without special post-hoc processes.

## III. GAUSSIANPLANT OVERVIEW

GaussianPlant reconstructs both plant structure and appearance from multi-view RGB images using a hierarchical 3D Gaussian representation. Here, we first briefly recap 3DGS, which is the basis of our method, followed by our GaussianPlantâs representation (Sec. IV) and optimization (Sec. V) methods.

Preliminary: 3D Gaussian splatting (3DGS). 3DGS [4] uses a large set of 3D Gaussian primitives to represent a 3D scene. Each Gaussianâs geometry is defined by the position of the Gaussian kernelâs center $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , rotation matrix $R \in \mathrm { S O ( 3 ) }$ , which is internally represented by a quaternion, and scaling factors $\pmb { S } = \mathrm { d i a g } ( s _ { 1 } , s _ { 2 } , s _ { 3 } )$ , where $s _ { i }$ denotes the axis-aligned scale. The covariance matrix Î£ is defined as $\Sigma = R S S ^ { \top } R ^ { \top }$ and thus, the 3D Gaussian distribution is defined as

$$
G ( \pmb { x } ) = \exp \left( - \frac { 1 } { 2 } ( \pmb { x } - \pmb { \mu } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ( \pmb { x } - \pmb { \mu } ) \right) ,\tag{1}
$$

where $\textbf { \em x } \in \ \mathbb { R } ^ { 3 }$ is an arbitrary 3D point. To represent the appearance of Gaussian primitives, each 3D Gaussian has the color c and opacity $\alpha ,$ where the color attribute c contains the base color and spherical harmonics (SH) coefficients. The color C of a pixel is given by volumetric rendering along a ray:

$$
C = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } T _ { i } \mathrm { \mathrm { ~ } w i t h } T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $\mathcal { N }$ is the list of 3D Gaussians whose 2D projections overlap the pixel.

## IV. GAUSSIANPLANT REPRESENTATION: HIERARCHICAL PRIMITIVES

We use a hierarchical representation consisting of appearance primitives (ApPs) for appearance representation, and structure primitives (StPs) for representing the underlying structure, whose combination is the key component of our GaussianPlant.

StPs are invisible primitives that have a dual geometry representation: 1) 3D Gaussian for easy initialization and optimization, and 2) explicit surface to faithfully represent the structure of the plants, $i . e . .$ , a cylinder if it is considered as a part of a branch and an elliptic disk if it is on a leaf. Highfrequency $\mathsf { A p P s }$ , represented as ordinary 3D Gaussians, are bound on the invisible StPsâ surfaces and jointly optimized.

## A. StP: Dual-Geometry Representation

StPs represent the coarse plantâs underlying structure using invisible primitives, using both 3D Gaussian and explicit surfaces.

<!-- image-->

<!-- image-->  
Fig. 3. Dual-geometry representation of StPs. StPsâ 3D Gaussians are converted to cylinders and disks, representing branches and leaves, respectively. Gaussian StPsâ color (leaf: green, branch: brown) is just for visualization.

1) Gaussian representation: As the same manner to the original 3DGS, a StP is represented using the center position $\mu _ { \mathrm { s t } } \ \in \ \mathbb { R } ^ { 3 }$ , rotation matrix $R _ { \mathrm { s t } } ~ \in ~ \mathrm { S O ( 3 ) }$ , and scale matrix $S _ { \mathrm { s t } } = \mathrm { d i a g } ( s _ { \mathrm { s t } } \in \mathbb { R } ^ { 3 } )$ . The StPs also equips the learnable branch-leaf probability $p _ { \mathrm { s t } } \in \mathsf { \Gamma } ( 0 , 1 )$ as their attribute, where $p _ { \mathrm { s t } } = 1$ indicates a part of a branch, and $p _ { \mathrm { s t } } = 0$ indicates a part of a leaf, respectively.

2) Surface representation: Given the StPâs parameters $\{ \mu _ { \mathrm { s t } } , R _ { \mathrm { s t } } , S _ { \mathrm { s t } } , p _ { \mathrm { s t } } \}$ , we convert it to explicit surface models, namely, a cylinder for branches (if $p _ { \mathrm { s t } } \geq 0 . 5 )$ and a disk for leaves (if $p _ { \mathrm { s t } } < 0 . 5 )$ , as shown in Fig. 3.

Branches: Cylinder representation. When a StP is estimated as a part of a branch, $i . e . , p _ { \mathrm { s t } } \ge 0 . 5$ , the primitive is converted to a cylinder represented by the following parameters:

$$
\begin{array} { r } { \{ \mu _ { \mathrm { c y } } = \mu _ { \mathrm { s t } } , \ u _ { \mathrm { c y } } = v _ { \mathrm { s t } } ^ { 1 } , \ r _ { \mathrm { c y } } = s _ { \mathrm { s t } } ^ { 2 } , \ l _ { \mathrm { c y } } = 3 s _ { \mathrm { s t } } ^ { 1 } \} , } \end{array}\tag{3}
$$

where $\mu _ { \mathrm { c y } }$ is the center point, $u _ { \mathrm { c y } } \in \mathbb { R } ^ { 3 }$ denotes the axis of the cylinder, $r _ { \mathrm { c y } } \in \mathbb { R } _ { + }$ and $l _ { \mathrm { c y } } \in \mathbb { R } _ { + }$ denote the radius and length, respectively.

Leaves: Disk representation. For a StP is leaf-like, i.e., $p _ { \mathrm { s t } } <$ 0.5, we define the elliptic disk using a set of following surface parameters:

$$
\{ \mu _ { \mathrm { d i } } = \mu _ { \mathrm { s t } } , \ n _ { \mathrm { d i } } = v _ { \mathrm { s t } } ^ { 3 } , \ a _ { \mathrm { d i } } = 2 s _ { k } ^ { 1 } , \ b _ { \mathrm { d i } } = s _ { \mathrm { s t } } ^ { 2 } \} ,\tag{4}
$$

where $\pmb { \mu } _ { \mathrm { d i } }$ denotes the center of the disk, ${  { n _ { \mathrm { d i } } } } \in \mathbb { R } ^ { 3 }$ denotes the surface normal, $a \in \mathbb { R } _ { + }$ and $b \in \mathbb { R } _ { + }$ denote the major radius and minor radius of the disk, respectively.

## B. ApP: Appearance and Feature Gaussian Representation

An ApP inherits the attributes of the original 3DGS, where its geometry is represented using the center position $\mu _ { \mathrm { a p } } \in$ $\mathbb { R } ^ { 3 }$ , rotation matrix $R _ { \mathrm { a p } } \ \in \ \mathrm { S O } ( 3 )$ , and scale matrix $S _ { \mathrm { a p } } = { }$ diag $ { \left( s _ { \mathrm { a p } } \in \mathbb { R } ^ { 3 } \right) }$ . For appearance representation, an $\mathrm { A p P }$ has its color coefficients $c _ { \mathrm { a p } }$ and opacity $O _ { \mathrm { a p } }$ in the same manner as the original 3DGS. An $\mathsf { A p P }$ is bound to a corresponding StP, where we do not explicitly denote the correspondence function for simplicity. To the optimization of branch-leaf label of corresponding StP, we also introduce a learnable semantic feature $\pmb { f } _ { \mathsf { a p } } \in \mathbb { R } ^ { D }$ for each $\mathrm { A p P } ,$ , which aggregates the image features from the observation.

## C. Initialization of Primitives

We initialize StPs from an ordinary 3DGS point cloud generated from multi-view images. We first group preoptimized 3DGS points into k groups using k-means clustering, and then perform principal component analysis (PCA) for each cluster to determine the Gaussian axis. Let the center position of a cluster as $\mu _ { \mathrm { s t } } \in \mathbb { R } ^ { 3 }$ , the eigenvalues as $\left\{ \lambda _ { \mathrm { s t } } ^ { 1 } , \lambda _ { \mathrm { s t } } ^ { \bar { 2 } } , \lambda _ { \mathrm { s t } } ^ { 3 } \mid \left( \lambda _ { \mathrm { s t } } ^ { 1 } \geq \lambda _ { \mathrm { s t } } ^ { 2 } \geq \lambda _ { \mathrm { s t } } ^ { 3 } \right) \right\}$ , and corresponding eigenvectors as $\left\{ v _ { \mathrm { s t } } ^ { 1 } , \dot { v _ { \mathrm { s t } } } , v _ { \mathrm { s t } } ^ { 3 } \right\}$ . The rotation $R _ { \mathrm { s t } } ~ \in ~ \mathbb { R } ^ { 3 \times 3 }$ and scale matrices $S _ { \mathrm { s t } } ~ \in ~ \mathrm { D i a g } ( 3 , \mathbb { R } )$ of the corresponding StP are defined as

$$
\pmb { R } _ { \mathrm { s t } } = \left[ \pmb { v } _ { \mathrm { s t } } ^ { 1 } \quad \pmb { v } _ { \mathrm { s t } } ^ { 2 } \quad \pmb { v } _ { \mathrm { s t } } ^ { 3 } \right] ,\tag{5}
$$

$$
S _ { \mathrm { s t } } = \mathrm { d i a g } \left( s _ { \mathrm { s t } } ^ { 1 } , s _ { \mathrm { s t } } ^ { 2 } , s _ { \mathrm { s t } } ^ { 3 } \right) = \alpha _ { \mathrm { s t } } \mathrm { d i a g } \left( \sqrt { \lambda _ { \mathrm { s t } } ^ { 1 } } , \sqrt { \lambda _ { \mathrm { s t } } ^ { 2 } } , \sqrt { \lambda _ { \mathrm { s t } } ^ { 3 } } \right)\tag{6}
$$

We initialize the branch-leaf probability $p _ { \mathrm { s t } }$ from simple geometric cues: for each initial cluster, we measure the principal components of its points and measure the anisotropy between the dominant and secondary scales, and the third scale, which suggests the thickness. Clusters that are highly anisotropic and thin are treated as branch-like, whereas more planar clusters are treated as leaf-like. We then initialize $p _ { \mathrm { s t } }$ to 0.6 for branch-like primitives and 0.4 for leaf-like ones, and let the subsequent optimization refine these probabilities.

ApPs are densely sampled on StPâs explicit surfaces. The position and attributes on ApPs are further optimized jointly with StPs. Letting the sampled point as $\pmb { x } \in \mathbb { R } ^ { 3 }$ and the surface normal of the StP at the point x as $\mathbf { \boldsymbol { n } } \in \mathbb { R } ^ { 3 }$ , the center point of the ApP is set to $\mu _ { \mathrm { a p } } = x$ . The rotation $ { R _ { \mathrm { a p } } } \in \mathbb { R } ^ { 3 \times 3 }$ is computed to align the z-axis with the corresponding surface normal n. Inspired by 2D Gaussian splatting [6], we initialize $\boldsymbol { S } _ { \mathrm { a p } } = \mathrm { d i a g } ( s _ { x } , s _ { y } , 0 )$ representing a flat surface.

## V. GAUSSIANPLANT OPTIMIZATION

Our model is trained end-to-end by jointly optimizing the StPs and ApPs. As summarized in Fig. 2, we organize the objective into three main flows and several regularizers: an appearance-geometry flow $\mathcal { L } _ { \mathrm { a g } }$ that couples the photometric supervision with the geometry binding between ApPs and StPs, a semantic flow $\mathcal { L } _ { \mathrm { s e m } }$ that aligns StPsâ semantics with $\mathrm { \bf A p P s ^ { \prime } }$ , and a structure flow $\mathcal { L } _ { \mathrm { s t } }$ that regularizes the explicit branch graph. We also introduce several regularizations $\mathcal { L } _ { \mathrm { r e g } } .$ Formally, the overall objective is written as

$$
\mathcal { L } = \lambda _ { \mathrm { a g } } \mathcal { L } _ { \mathrm { a g } } + \lambda _ { \mathrm { s e m } } \mathcal { L } _ { \mathrm { s e m } } + \lambda _ { \mathrm { s t } } \mathcal { L } _ { \mathrm { s t } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } ,\tag{7}
$$

where $\lambda _ { \mathrm { a g } } , \lambda _ { \mathrm { s e m } } , \lambda _ { \mathrm { s t } } , \lambda _ { \mathrm { r e g } }$ balance their contributions.

## A. Appearance-Geometry Flow $\mathcal { L } _ { \mathrm { a g } }$

Appearance-geometry flow contain photometric loss $\mathcal { L } _ { \mathrm { p } }$ and binding loss $\mathcal { L } _ { \mathrm { b i n d } }$ . For photometric loss, we use the same setting as in 3DGS that combines an L1 term ${ \mathcal { L } } _ { \mathrm { L 1 } }$ and an

<!-- image-->  
Fig. 4. Illustration of the binding loss ${ \mathcal { L } } _ { \mathrm { b i n d } } .$ . Left: the distance definition from $\mathrm { A p P }$ to a cylinder $( i . e .$ , branch) $d _ { \mathrm { c y } } .$ . Right: the distance definition from ApP to a disk (i.e., leaf) $d _ { \mathrm { d i } } ,$ which is defined as dplane or $\sqrt { d _ { \mathrm { p l a n e } } ^ { 2 } + d _ { \mathrm { e d g e } } ^ { 2 } }$ (i.e., the diagonal length in the figure). The binding loss $\dot { \mathcal { L } _ { \mathrm { b i n d } } }$ is defined as a weighted sum of these two distances, $d _ { \mathrm { c y } }$ and $d _ { \mathrm { d i } }$

SSIM term $\mathcal { L } _ { \mathrm { s s i m } }$ to minimize the color difference between the rendered image $\hat { I } _ { \mathrm { a g } }$ using ApPs and its corresponding input image I:

$$
\mathcal { L } _ { \mathrm { p } } = \lambda _ { \mathrm { L 1 } } \mathcal { L } _ { \mathrm { L 1 } } ( \hat { I } _ { \mathrm { a g } } , I ) + \lambda _ { \mathrm { s s i m } } \mathcal { L } _ { \mathrm { s s i m } } ( \hat { I } _ { \mathrm { a g } } , I ) .\tag{8}
$$

Then, binding loss $\mathcal { L } _ { \mathrm { b i n d } }$ serves as a flexible structural constraint, allowing ApPs to loosely bind to a StP rather than enforcing a rigid attachment. This loose binding preserves structural consistency while still permitting local adaptation, i.e., ApPs can adjust to finer appearance details while staying aligned with the overall structure. When significant structural changes occur $( e . g .$ , branch splitting), the binding loss naturally increases, signaling the need for the StP to refine and densify. Specifically, we measure the Euclidean distance between the $\mathrm { A p P ^ { \circ } s }$ center and the closest point p on its corresponding StP surface. For each $\mathrm { A p P }$ centered at $\mu _ { \mathrm { a g } } \in \mathbb { R } ^ { 3 }$ , we define two distance functions1, as illustrated in Fig. 4.

Letting the corresponding cylinderâs position, main axis, and radius as $\mu _ { \mathrm { c y } } , \mathrm { ~ } u _ { \mathrm { c y } }$ , and $r _ { \mathrm { c y } } .$ , respectively, the ApP-to-cylinder distance $d _ { \mathrm { c y } }$ is given by

$$
d _ { \mathrm { c y } } = \operatorname* { m a x } ( 0 , \lVert \pmb { \mu } _ { \mathrm { a p } } - \pmb { \mu } _ { \mathrm { a p } } ^ { \mathrm { p } } \rVert _ { 2 } - r _ { \mathrm { c y } } ) ,\tag{9}
$$

where $\pmb { \mu } _ { \mathrm { a p } } ^ { \mathrm { p } } = \pmb { \mu } _ { \mathrm { c y } } + \left[ ( \pmb { \mu } _ { \mathrm { a p } } - \pmb { \mu } _ { \mathrm { c y } } ) { \cdot } \pmb { u } _ { \mathrm { c y } } \right]$ ucy is the projection of $\mu _ { \mathrm { a p } }$ on the cylinder main axis ucy.

Similarly, the ApP-to-disk distance $d _ { \mathrm { d i } }$ is

$$
d _ { \mathrm { d i } } = \left\{ \begin{array} { l l } { d _ { \mathrm { p l a n e } } , } & { \rho \leq 1 , } \\ { \sqrt { d _ { \mathrm { p l a n e } } ^ { 2 } + d _ { \mathrm { e d g e } } ^ { 2 } } , } & { \rho > 1 , } \end{array} \right.\tag{10}
$$

where $d _ { \mathrm { p l a n e } } , \ d _ { \mathrm { e d g e } }$ refer to the perpendicular distance to the diskâs plane and the planar distance from the projected point to the ellipse boundary, defined as

$$
d _ { \mathrm { p l a n e } } = | ( \pmb { \mu } _ { \mathrm { a p } } - \pmb { \mu } _ { \mathrm { d i } } ) \pmb { n } _ { \mathrm { d i } } | ,\tag{11}
$$

$$
\begin{array} { r } { d _ { \mathrm { e d g e } } = \big \| [ x , y ] ^ { \top } ~ - \left[ \frac { a _ { k } x } { \rho } , \frac { b _ { \mathrm { d i } } y } { \rho } \right] ^ { \top } \big \| _ { 2 } , } \end{array}\tag{12}
$$

$$
\begin{array} { r } { \rho = \sqrt { \left( \frac { x } { a _ { \mathrm { d i } } } \right) ^ { 2 } + \left( \frac { y } { b _ { \mathrm { d i } } } \right) ^ { 2 } } , } \end{array}\tag{13}
$$

where $a _ { \mathrm { d i } } , b _ { \mathrm { d i } }$ are the axes of the ellipse, $\rho$ measures the normalized radial distance within the ellipse, and $[ x , y ] ^ { \top }$ is the

in-plane coordinates of the $\pmb { \mu } _ { \mathrm { a g } }$ that are obtained by projecting onto an orthonormal basis $e _ { 1 } , e _ { 2 }$ of the diskâs plane:

$$
{ \big [ } { x \atop y } { \big ] } = { \left[ { \begin{array} { l l } { ( \mu _ { \mathrm { a p } } - \pmb { \mu } _ { \mathrm { d i } } ) \pmb { e } _ { 1 } } \\ { ( \mu _ { \mathrm { a p } } - \pmb { \mu } _ { \mathrm { d i } } ) \pmb { e } _ { 2 } } \end{array} } \right] } .\tag{14}
$$

A non-trivial issue here is leaf-branch ambiguity because the initial clustering cannot accurately determine whether a StP represents a branch or a leaf. As mentioned above, every StP is assigned with a class probability $p _ { \mathrm { s t } } .$ , and we fold this uncertainty into the loss as

$$
\mathcal { L } _ { \mathrm { b i n d } } = \frac { 1 } { \vert A \vert } \sum _ { A } p _ { \mathrm { s t } } d _ { \mathrm { c y } } + \frac { 1 } { \vert A \vert } \sum _ { A } ( 1 - p _ { \mathrm { s t } } ) d _ { \mathrm { d i } } ,\tag{15}
$$

where A is the entire set of $\mathrm { \bf A p P s } .$ , and $| { \cal A } |$ is the total number of ApPs. The probability $p _ { \mathrm { s t } }$ acts as a soft mask if an $\mathrm { A p P }$ is likely to belong to a branch StP $( i . e . , p _ { \mathrm { s t } } \approx 1 )$ , only the cylinder term matters, and vice versa for a leaf StP. Formally, $\mathcal { L } _ { \mathrm { a g } }$ can be written as

$$
\mathcal { L } _ { \mathrm { a g } } = \lambda _ { \mathrm { p } } \mathcal { L } _ { \mathrm { p } } + \lambda _ { \mathrm { b i n d } } \mathcal { L } _ { \mathrm { b i n d } } ,\tag{16}
$$

where $\lambda _ { \mathfrak { p } }$ and $\lambda _ { \mathrm { b i n d } }$ balance the contributions of the photometric term and the geometry binding term, respectively.

## B. Semantic Flow $\mathcal { L } _ { \mathrm { s e m } }$

While the geometry-based probability $p _ { \mathrm { s t } }$ captures shape regularities, it may be ambiguous in regions where slender stems appear planar or overlapping leaves exhibit branch-like geometry. To enhance the discriminability of StPs, we thus incorporate semantic priors distilled from DINOv3 [32] during the optimization.

Specifically, we assign each ApP a learnable semantic feature vector $f _ { \mathrm { a p } } ~ \in ~ \bar { \mathbb { R } ^ { D } }$ . When projecting Gaussians into the image plane, the rendered semantic feature of a pixel is computed analogously to the volumetric color compositing in Eq. (2):

$$
F _ { s } = \sum _ { i \in \mathcal { N } } f _ { i } \alpha _ { i } T _ { i } .\tag{17}
$$

To supervise the rendered semantic map, we first construct a ground-truth feature map $\hat { F } _ { s }$ by extracting the DINOv3 feature map from the input images, upsampling them with a pretrained upsampling model [33]. Considering the computational and memory cost during rendering, we further down-project them via PCA, which also stabilizes optimization. In practice, we find that a moderate PCA dimension $D = 1 2 8$ preserves the essential discriminative power of the original features while significantly reducing redundancy. Subsequently, we minimize the pixel-wise difference between the rendered semantic map and the ground-truth:

$$
\mathcal { L } _ { \mathrm { f } } = \Vert F _ { s } - \hat { F } _ { s } \Vert _ { 1 } .\tag{18}
$$

We then aggregate the semantic feature for each StP by pooling the features of its bound ApPs:

$$
\mathbf { \star } f _ { \mathrm { s t } } = \frac { 1 } { \vert \mathcal { N } ( \mathrm { s t } ) \vert } \sum _ { j \in \mathcal { N } ( \mathrm { s t } ) } \mathbf { \nabla } f _ { \mathrm { a p } } ^ { j } .\tag{19}
$$

Then, let $\pmb { f } _ { t } \in \mathbb { R } ^ { D }$ be DINOv3 text features projected into the same PCA space, we obtain a semantic likelihood for âleafâ and âbranchâ via cosine similarity as

$$
\pi _ { c } = \frac { \exp ( \cos ( f _ { \mathrm { s t } } , f _ { t } ^ { c } ) / \tau ) } { \sum _ { c ^ { \prime } \in \{ \mathrm { b r a n c h } , \mathrm { l e a f } \} } \exp ( \cos ( f _ { \mathrm { s t } } , f _ { t } ^ { c ^ { \prime } } ) / \tau ) } .\tag{20}
$$

This likelihood reflects how likely a StP semantically resembles a leaf or a branch in the visionâlanguage embedding space. Similar to the binding loss, which leverages the same learnable probability $p _ { \mathrm { s t } }$ to weight geometric distances between cylinder and disk primitives, we also use $p _ { \mathrm { s t } }$ to bridge the semantic guidance by using a cross-entropy objective

$$
\mathcal { L } _ { \mathrm { s e m } } = - \Bigl [ ( 1 - p _ { \mathrm { s t } } ) \log \pi _ { \mathrm { l e a f } } + p _ { \mathrm { s t } } \log \pi _ { \mathrm { b r a n c h } } \Bigr ] .\tag{21}
$$

By optimizing both $\mathcal { L } _ { \mathrm { b i n d } }$ and $\mathcal { L } _ { \mathrm { s e m } }$ with respect to the same $p _ { \mathrm { s t } }$ , the model jointly enforces geometry- and semanticsdriven supervision on the StPs that ensures geometry-semantic consistency.

## C. Structure Flow $\mathcal { L } _ { \mathrm { s t } }$

The structure flow encourages the recovered branch primitives to form a coherent, tree-like structure with locally smooth geometry. Here we first extract an explicit graph as the plantstructure representation, where endpoints of StPs are graph nodes and candidate connections are graph edges. For each branch StP (modeled as a cylinder-like primitive), we take its center $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ , principal axis ${ \pmb u } _ { \mathrm { c y } } \in \mathbb { R } ^ { 3 } ~ ( \| { \pmb u } _ { \mathrm { c y } } \| _ { 2 } = 1 )$ , and longitudinal scale $s _ { \mathrm { s t } } ^ { 1 } > 0$ to define its two endpoints

$$
p ^ { \mathrm { t o p / b o t t o m } } = \mu \pm s _ { \mathrm { s t } } ^ { 1 } u _ { \mathrm { c y } } .\tag{22}
$$

Each primitive thus yields one inner edge connecting its two endpoints; we denote the set of all such inner edges by $E _ { \mathrm { i n } }$

To capture global tree-like topology, we form a k-nearest neighbors (KNN) graph over all endpoints with Euclidean weights and apply Kruskalâs algorithm to extract a minimum spanning tree (MST). However, a pure Euclidean MST may introduce undesirable âshortcutâ edges across gaps or sharp turns. Prior works often initialize tree skeletons from MSTstyle graphs and then prune or reweight edges to improve biological plausibility, e.g., AdTree [19] builds an initial MST and refines it via optimization, while earlier work by Livny et al. [34] constructs an MST skeleton and applies global constraints to enforce a directed acyclic branch-structure graph. Following these ideas, we therefore reweight each candidate cross-StP edge with two data-driven penalties before the final MST: Axis-consistency penalty and Void-crossing penalty. For a cross-StP edge with endpoints $p _ { i }$ and $p _ { j }$ and their unit direction $\begin{array} { r } { \mathbf { { } t } \ : = \ : \frac { \left( p _ { i } - p _ { j } \right) } { \left\| p _ { i } - p _ { j } \right\| } } \end{array}$ , given the main axis $\mu _ { i }$ and $\mu _ { j }$ , the axis-consistency terms $c _ { t i } = | \pmb { t } ^ { T } \pmb { \mu } _ { i } |$ and $c _ { t j } = | \pmb { t } ^ { T } \pmb { \mu } _ { j } |$ measure the collinearity between two StP axes and define the penalty as

$$
\mathrm { p e n } _ { \mathrm { a x i s } } = 1 + \gamma _ { \mathrm { t a n } } \big ( 1 - \frac { 1 } { 2 } \big ( c _ { t i } + c _ { t j } \big ) \big ) ,\tag{23}
$$

making oblique or misaligned connections become more expensive. To suppressâgap-crossingâ shortcuts, along each cross-StP edge we sample points and count neighbors within radius

<!-- image-->  
Fig. 5. Structure graph built from endpoints: inner-StP edges (blue) and cross-StP MST edges (red). These edges drive the regularizers in Eq. (28).

Ï from $\mathrm { \bf A p P s . }$ . Edges traversing sparsely supported space are penalized as

$$
\mathrm { p e n } _ { \mathrm { o c c } } = 1 + \gamma _ { \mathrm { o c c } } \frac { 1 } { K } \sum _ { k } 1 [ n _ { k } < \theta ] ,\tag{24}
$$

where Î¸ refers to the threshold of neighbor number. The final structure-aware edge cost is

$$
\omega _ { i j } = d _ { i j } { \tt p e n } _ { \mathrm { a x i s } } { \tt p e n } _ { \mathrm { o c c } } .\tag{25}
$$

Running Kruskalâs algorithm on these reweighted edges yields an MST that better follows branch directionality and avoids unsupported shortcuts. Edges in the MST that connect endpoints from different StPs are collected as cross-StP edges $E _ { \mathrm { c r o s s } }$ together with $E _ { \mathrm { i n } }$ they define our structure graph $G =$ $( V , E _ { \mathrm { i n } } \cup E _ { \mathrm { c r o s s } } )$ with node set V the endpoints (see Fig. 5).

Given the structural graph G, we encourage geometric continuity by pulling together endpoints connected by cross-StP edges:

$$
\mathcal { L } _ { \mathrm { g r a p h } } = \frac { 1 } { | E _ { \mathrm { c r o s s } } | } \sum _ { ( i , j ) \in E _ { \mathrm { c r o s s } } } \left. \pmb { p } _ { i } - \pmb { p } _ { j } \right. _ { 2 } .\tag{26}
$$

To promote local smoothness, we further apply a Laplacian penalty on the endpoint coordinates:

$$
\mathcal { L } _ { \mathrm { l a p } } = \frac { 1 } { | V | } \sum _ { i \in V } \Big | \Big | p _ { i } - \frac { 1 } { \deg ( i ) } \sum _ { j \in \mathcal { N } ( i ) } \pmb { p } _ { j } \Big | \Big | _ { 2 } ^ { 2 } ,\tag{27}
$$

where $\mathcal { N } ( i )$ and $\deg ( i )$ denote the neighbor set and degree of node i in G. The total structural loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s t } } = \mathcal { L } _ { \mathrm { g r a p h } } + \mathcal { L } _ { \mathrm { l a p } } . } \end{array}\tag{28}
$$

## D. Regularizers $\mathcal { L } _ { \mathrm { r e g } }$

We additionally introduce several regularizers that stabilize optimization and refine the primitives: a depth loss $\mathcal { L } _ { \mathrm { d } }$ to better constrain geometry, an overlap loss $\mathcal { L } _ { \mathrm { o p } }$ to discourage different StPs from collapsing onto each other, and a self-organized classification loss $\mathcal { L } _ { \mathrm { c l s } }$ to encourage confident branch/leaf labeling.

1) Overlap loss $\mathcal { L } _ { o p } .$ The overlap loss is introduced to enforce clear boundaries between neighboring StP, promoting a more compact and well-separated structure representation. The overlap is defined as a Gaussian-weighted Mahalanobis distance between the i-th StP and its nearest neighbor as

$$
\mathcal { L } _ { \mathrm { o p } } = \sum _ { i } \sum _ { j \in \mathcal { N } ( i ) } \exp \left( - \frac { 1 } { 2 } ( \pmb { \mu } _ { i } - \pmb { \mu } _ { j } ) ^ { \top } \pmb { \Sigma } _ { j } ^ { - 1 } ( \pmb { \mu } _ { i } - \pmb { \mu } _ { j } ) \right) ,\tag{29}
$$

where $\mathcal { N } ( \cdot )$ refers to the k-nearest neighbors, in which we set k as 3 in all experiments.

2) Depth loss $\mathcal { L } _ { d } \mathrm { : }$ Since monocular depth is used to guide the training in a lot of 3DGS works [4], [35], [36], we also incorporate this term:

$$
\mathcal { L } _ { d } = \left. \hat { D } _ { \mathrm { a g } } - D \right. _ { 2 } ,\tag{30}
$$

where $\hat { D } _ { \mathrm { a g } }$ is the rendered depth and D is the monocular depth estimated by [37].

3) Classification loss $\mathcal { L } _ { c l s } .$ To encourage each StP to confidently choose between a branch/leaf label while leveraging the color cues, we combine two terms into a classification loss between branches and leaves:

$$
\mathcal { L } _ { \mathrm { c l s } } = \mathcal { L } _ { \mathrm { c o l } } + \mathcal { L } _ { \mathrm { c o n f } } ,\tag{31}
$$

$$
\mathcal { L } _ { \mathrm { c o l } } = \frac { 1 } { \left| \mathcal { A } \right| } \sum _ { \mathcal { A } } \left[ p _ { \mathrm { s t } } \left| \left| c _ { \mathrm { a g } } - \bar { c } _ { \mathrm { l e a f } } \right| \right| _ { 2 } ^ { 2 } + \left( 1 - p _ { \mathrm { s t } } \right) \left| \left| c _ { \mathrm { a g } } - \bar { c } _ { \mathrm { b r a n c h } } \right| \right| _ { 2 } ^ { 2 } \right]\tag{32}
$$

$$
\mathcal { L } _ { \mathrm { c o n f } } = \frac { 1 } { \left| S \right| } \sum _ { s } p _ { \mathrm { s t } } { \left( 1 - p _ { \mathrm { s t } } \right) } ,\tag{33}
$$

where S is the entire set of StPs, and |S| is the total number of the StPs. We use a sigmoid function to limit the range of $p _ { \mathrm { s t } }$ in (0, 1), letting the probability of being a part of a branch. Here, $c _ { \mathrm { a g } }$ denotes the $\mathrm { A p P ^ { \circ } s }$ color feature, and $\bar { c } _ { \mathrm { l e a f } }$ and $\bar { c } _ { \mathrm { b r a n c h } }$ are the mean color features of the current leaf and branch sets, respectively. The first term $\mathcal { L } _ { \mathrm { c o l } }$ pulls each primitiveâs label toward the class whose average color it matches, while the second term ${ \mathcal { L } } _ { \mathrm { c o n f } }$ penalizes probabilities near 0.5 (controlled by weight Î²) to speed up the binarization.

## E. Adaptive Density Control of Primitives

The densification strategy plays a crucial role in optimizing 3D Gaussians for capturing fine details while maintaining efficiency. However, the standard densification strategy in 3DGS, which relies on appearance-based gradient thresholds, is not suitable for structure extraction. We therefore introduce a new density control strategy for StPs, encompassing both densification and merging.

1) StP densification: The densification of StPs uses the gradient from both geometry binding loss $\mathcal { L } _ { \mathrm { b i n d } }$ and semantic binding loss $\mathcal { L } _ { \mathrm { s e m } }$ , where StPs with large binding losses are densified. This ensures that StPs remain adaptive to evolving structures without overfitting to fine-grained textures, preventing StPs from being influenced by high-frequency details. To maintain a compact and interpretable structure representation, we impose a hard upper bound on the total number of StPs. Once this bound is reached, we perform only pruning without further densification. During optimization, StPs with small spatial scales or low opacity are removed, as they contribute negligibly to the structural geometry.

TABLE I  
QUANTITATIVE COMPARISON ON NOVEL-VIEW SYNTHESIS BETWEEN THE PROPOSED AND THE BASELINE METHOD AVERAGED OVER ALL THE 10 PLANTS. THE BEST RESULTS ARE HIGHLIGHTED BOLD. OUR METHOD  
(GAUSSIANPLANT) ACHIEVES BETTER NOVEL-VIEW SYNTHESIS ACCURACIES IN MOST CASES COMPARED TO THE ORIGINAL 3DGS.
<table><tr><td>Method</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>3DGS [4]</td><td>28.732</td><td>0.932</td><td>17.16</td></tr><tr><td>Ours (w/o  ${ \mathcal { L } } _ { \mathrm { b i n d } } )$ </td><td>28.835</td><td>0.935</td><td>17.09</td></tr><tr><td>Ours</td><td>29.486</td><td>0.947</td><td>16.31</td></tr></table>

2) Branch StP merging & filtering: We observe that branch StPs often fragment into many small, thin cylinders when fitting thick branches. We thus merge nearby StPs, whose endpoints lie within a small spatial radius and whose axes align closely, computed as the cosine similarity smaller than a threshold. For each merged group, we collect all endpoints and compute their centroid to serve as the new cylinder center. We then perform PCA on these endpoints and set the first principal component as the new cylinder axis. By projecting each endpoint onto this axis, we obtain the minimum and maximum projection values, whose difference defines the cylinder height. Finally, we set the radius to the average of the original primitive radii, yielding a single, coherent branch cylinder represented by a StP. Despite geometry and semantic binding providing a reasonable classification result, there still could be some misclassified StP. We add a simple radiusgrowth sanity check on the MST to remove noise (i.e., leaf StP). Along each parent-child edge, branch radii are expected to be non-increasing. Therefore, child StP whose radius is abnormally larger than its parent will be pruned.

3) Leaf StP clustering for leaf instance extraction: To extract leaf instances, we use a merging-based clustering approach. Note that this process is only performed once after the optimization. Given the leaf StPs, we group those whose centers lie within a small spatial radius, as well as their normals and disk main axis, align within a predefined angular threshold.

## VI. GAUSSIANPLANT DATASET & APPLICATIONS

The task addressed by GaussianPlant, i.e., 3D structure extraction of leafy plants from multi-view images, has been underexplored. While the application field of our Gaussian-Plant spreads beyond CV and CG, to plant science and agriculture, there are no existing datasets to assess the success of our task. We summarize here our new dataset, as well as the potential applications by GaussianPlant.

## A. GaussianPlant Dataset

We created a new real-world benchmark dataset for 3D plant modeling and structural reconstruction, named GaussianPlant dataset. The dataset contains 10 artificial plants all captured under indoor lighting conditions, as shown in Fig. 6. We use a variety of plant species, including those with large leaves (e.g., UMBELLATA) and small leaves (e.g., BENJAMINA), where the characteristics of shape, structure, and occlusion differ.

<!-- image-->  
Fig. 6. GaussianPlant dataset. Ten real-world artificial plants are carefully captured in indoor environments and reconstructed with MVS. For each plant, we manually annotate the ground-truth labels of 3D branch-leaf and leaf instance segmentation (visualized on the right side of the plantsâ appearances).

For each scene, we acquire more than 120 multi-view images and reconstruct the plants using multi-view stereo2 to create a dense and reliable point cloud. During the capture, we carefully plan viewpoints to minimize occlusions, ensuring that the reconstructed point clouds are as complete as possible. The reconstructed point clouds are then manually cleaned to remove background and floating noise, which we use as the ground-truth shapes. On top of these point clouds, we annotate two types of ground-truth labels for benchmarking: (i) a binary leaf-branch segmentation and (ii) leaf instance labels. Specifically, we import each dense point cloud into Blender3 and use its vertex selection tools to interactively assign labels to points. For plants with extremely dense foliage, such as ASH and BENJAMINA, reliable per-leaf delineation is infeasible, so we only provide binary leaf-branch labels and exclude them from the leaf instance segmentation evaluation.

## B. Applications

Since our approach achieves an explicit and sparse structure representation, given the correspondence between StP and ApP, it enables a range of practical applications beyond ordinary 3D shape reconstruction and photorealistic novelview synthesis.

1) 3D branch structure extraction: By filtering StPs by their branch label, we obtain a compact cylinder-based skeleton and further construct a 3D graph, as introduced in Sec. V, which directly represents the connectivity and topology of the plantâs branching system. Moreover, using the binding correspondence to branch StPs, we can also recover a dense branch point cloud as a subset of ApPs, facilitating accurate trait measurement and structural analysis, as shown in Fig. 1.

This application can be useful for plant structural analysis in plant science and breeding, where the number of joints (i.e., tillers) and the branching pattern are important traits to evaluate the plant genotypes.

2) Instance-wise leaf 3D reconstruction: Leaf StPs naturally separate foliage from woody parts. We perform a simple clustering for leaf instance extraction, allowing instance-level leaf segmentation without 2D mask supervision, as shown in Fig. 1. The 3D instance segmentation of leaves enables the direct evaluation of specific leaf traits, such as leaf area, which is directly applicable for plant phenotyping and the automation of agriculture.

3) Plant CG asset creation and editing: The disentangled structure and appearance representation also support interactive editing operations, such as leaf addition/removal or branch-wise deformation, enabling the creation of plant CG assets capable of dynamic simulation and specific design.

## VII. EXPERIMENTS

Our primary objective is to recover the plantâs structure while preserving its high-fidelity appearance. To this end, we evaluate GaussianPlant on four tasks: (1) Novel-view synthesis, to verify that the introduction of structural primitives does not harm and can even improve image-level rendering quality; (2) 3D branch segmentation, to test whether our geometrysemantics-structure coupling yields cleaner leaf/branch separation than purely appearance-based, open-vocabulary segmentation on 3DGS; (3) Branch structure reconstruction, to evaluate the accuracy of the recovered branch geometry beyond dense branch regions, since accurate dense branch segmentation results cannot directly demonstrate whether the recovered branches are structurally correct (i.e., a method can place points near true branches yet still produce wrong shortcuts or broken connections); and (4) 3D leaf instance segmentation, to examine whether the learned leaf StPs provide a more instance-friendly representation than directly clustering dense leaf ApPs.

TABLE II  
QUANTITATIVE COMPARISONS OF 3D BRANCH SEGMENTATION AND ABLATION STUDY. THE BEST RESULTS ARE HIGHLIGHTED BOLD.
<table><tr><td></td><td colspan="10">Chamfer distance [mm] â</td></tr><tr><td>Methods</td><td>UMBELLATa-1</td><td>AsH</td><td>UmbeLlata-2</td><td>UMBELLaTA-3</td><td>UMBELLata-4</td><td>Benjamina</td><td>MONsteRa-1</td><td>MOnstera-2</td><td>UMBELLaTa-5</td><td>ARtifICiaL</td><td>Mean</td></tr><tr><td>Feature-3DGS [38]</td><td>334.2</td><td>14.4</td><td>31.6</td><td>952.1</td><td>163.5</td><td>13.8</td><td>346.2</td><td>2351.0</td><td>178.6</td><td>555.2</td><td>494.06</td></tr><tr><td>LangSplat [40]</td><td>171.7</td><td>14.5</td><td>7.2</td><td>132.8</td><td>18.9</td><td>2.0</td><td>124.6</td><td>734.2</td><td>82.0</td><td>44.5</td><td>133.24</td></tr><tr><td>Ours w/o  ${ \mathcal { L } } _ { \mathrm { b i n d } }$ </td><td>73.8</td><td>8.3</td><td>22.6</td><td>75.2</td><td>67.5</td><td>33.2</td><td>105.4</td><td>116.5</td><td>76.8</td><td>67.7</td><td>64.7</td></tr><tr><td>urs w/o  ${ \mathcal { L } } _ { \mathrm { s e m } }$ </td><td>256.6</td><td>15.2</td><td>27.1</td><td>106.2</td><td>21.6</td><td>2.1</td><td>24.8</td><td>39.4</td><td>64.3</td><td>120.1</td><td>67.7</td></tr><tr><td>rs w/lo  $\mathcal { L } _ { \mathrm { o p } }$ </td><td>3.5</td><td>2.5</td><td>11.2</td><td>96.0</td><td>16.5</td><td>8.5</td><td>76.4</td><td>23.8</td><td>56.4</td><td>45.7</td><td>34.1</td></tr><tr><td>Ours w/o  ${ \mathcal { L } } _ { \mathrm { c l s } }$ </td><td>6.2</td><td>5.7</td><td>93.3</td><td>79.3</td><td>27.8</td><td>148.6</td><td>58.7</td><td>83.9</td><td>245.2</td><td>30.7</td><td>78.0</td></tr><tr><td>Ours w/o  $\mathcal { L } _ { \mathrm { g } }$ </td><td>5.9</td><td>3.8</td><td>23.2</td><td>56.2</td><td>42.3</td><td>74.5</td><td>42.2</td><td>42.3</td><td>323.3</td><td>56.5</td><td>67.2</td></tr><tr><td>Ours (full model)</td><td>2.4</td><td>3.4</td><td>1.8</td><td>73.7</td><td>13.1</td><td>6.2</td><td>53.2</td><td>10.7</td><td>34.8</td><td>30.1</td><td>22.9</td></tr></table>

## A. Setup

1) Baselines: For the novel-view synthesis task, we compare GaussianPlant with the original 3DGS [4].

For the 3D branch segmentation task, we assess the segmented 3D branch point clouds against two 3DGS-based openvocabulary pipelines applied to the original Gaussian field. Feature-3DGS [38], which distills high-dimensional features from 2D foundation models (e.g., SAM [29], CLIP [39], and LSeg [40]) into per-Gaussian embeddings; and LangSplat [41], which attaches a scale-gated affinity feature to each Gaussian and trains it with SAM-distilled, scale-aware contrastive learning. For GaussianPlant, we use the learned branch/leaf probabilities on StPs to select branch StPs and their attached dense ApPs. For Feature-3DGS and LangSplat, we follow their open-vocabulary setting by computing the visualâtext similarity between each Gaussian feature and the text prompts âleafâ and âbranchâ, assigning the label with higher similarity, and obtaining the dense branch points. We use LSeg and CLIP as the 2D encoders for Feature-3DGS and LangSplat, respectively.

For the branch structure reconstruction task, to the best of our knowledge, there is no existing 3DGS-based method that outputs an explicit branch structure graph with radii comparable to our StP graph. We therefore evaluate only within our framework, using ablations that remove structurerelated components to isolate their contribution to branch topology.

For the leaf instance segmentation task, we compare our StPs-based instance clustering with a classical clustering baseline, namely, DBSCAN [42] applied directly on leaf ApP point clouds.

2) Datasets: For the quantitative evaluation, we use our GaussianPlant dataset. For each plant instance, we randomly split the multi-view images of each scene into training and testing sets with a fixed ratio of 9 : 1. The same train/test split is shared by all methods during evaluation.

To further test practicality and generalization, we also capture several outdoor real plants in the wild. Due to extremely dense foliage and occlusions, fine instance-level annotations are infeasible. Thus, for these scenes, we focus on the qualitative evaluation.

3) Metrics: For the novel-view synthesis task, we use the standard rendering quality metrics using PSNR, SSIM, and LPIPS for comparison.

For the 3D branch segmentation task, we measure how accurately the segmented branch points match the ground-truth branch points. Given the ground-truth branch point cloud and the branch points segmented from each method, we compute the bi-directional Chamfer Distance (CD) between the two sets.

<!-- image-->  
Fig. 7. Feature map visualization. We visualize the rendered feature map on three plants under varying visibility conditions. We observe that our method gives the most accurate branch discrimination results.

For the branch structure reconstruction task, we evaluate the structural accuracy of the branch graph recovered by StPs. In practice, each graph edge is interpreted as a cylindrical segment whose radius is taken from the corresponding StP scale (inner-StP edges use their own radius, cross-StP edges use the mean radius of the two primitives). To assess the geometric accuracy of branch graphs, we densely sample points on the center of the branch StPsâ cylinders, and compute the bi-directional CD with the ground-truth branch points.

For the leaf instance segmentation task, we evaluate how well individual leaves are separated. We obtain dense points for each instance using the clustering label, and report the mean CD over all instances.

4) Implementation details: All our experiments are trained on a NVIDIA Quadro RTX 8000 GPU. Although our framework defines several objectives, we never optimize them all at once. Instead, we activate losses progressively so that each stage stabilizes the variables it is best suited for. We first pretrain Gaussians with DINOv3 distilled semantic feature using $\mathcal { L } _ { f } , \mathcal { L } _ { d }$ and ${ \mathcal { L } } _ { p }$ with $\lambda _ { p } = 1 , \lambda _ { f } = 1 , \lambda _ { d } = 0 . 1 , \lambda _ { \mathrm { s s i m } } =$

<!-- image-->  
Fig. 8. Qualitative comparison on 3D branch segmentation. From left to right: reference photo, the ground-truth branch points (GT), Feature-3DGS [38], LangSplat [41], and Ours. Rows show different plants. Baselines driven purely by appearance features tend to either miss thin branchlets or include leaf clutter; our method recovers cleaner, more continuous branch scaffolds with fewer leaf artifacts.

0.2, $\lambda _ { \mathrm { L 1 } } = 0 . 8$ for 30000 iterations.

After that, we first remove the background Gaussians and initialize the StPs. Let $N _ { p }$ denote the number of pretrained points. The initial number of clusters is set to one per 100 points, i.e., $\begin{array} { r c l } { K } & { = } & { N _ { p } / 1 0 0 } \end{array}$ . A warm-up stage (first 500 iterations) is applied to initial StPs using ${ \mathcal { L } } _ { p }$ and $\mathcal { L } _ { d }$ without densification to learn coarse geometry. Then, for each StP we spawn 50 ApPs, which are then optimized jointly using $\mathcal { L } _ { f } ,$ $\mathcal { L } _ { d } , \mathcal { L } _ { p }$ and $\mathcal { L } _ { \mathrm { b i n d } }$ with $\lambda _ { p } = 1 , \lambda _ { f } = 1 , \lambda _ { d } = 0 . 1 , \lambda _ { \mathrm { b i n d } } = \dot { 0 . 1 }$ and follow the standard 3DGS densification strategy.

Once ApPs begin optimizing, two binding losses $\mathcal { L } _ { \mathrm { s e m } }$ and $\mathcal { L } _ { \mathrm { b i n d } }$ become the primary objectives for StPs, along with overlap regularizer $\mathcal { L } _ { \mathrm { o p } }$ and classification loss $\mathcal { L } _ { \mathrm { c l s } }$ where $\lambda _ { \mathrm { b i n d } } ~ = ~ 1 , \lambda _ { \mathrm { s e m } } ~ = ~ 0 . 2 , \lambda _ { \mathrm { o p } } ~ = ~ 0 . 0 5 , \lambda _ { \mathrm { c l s } } ~ = ~ 0 . 1$ . Similar to 3DGS, we stop densification for both ApPs/StPs after 15000 iterations. Following the filtering strategy introduced in Sec. V-E2, we remove likely leaf StPs and simultaneously enable the $\mathcal { L } _ { g }$ to refine the structure with $\lambda _ { g } = 1$

View 1

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

View 2

<!-- image-->  
Umbellata-2

<!-- image-->  
Umbellata-3

<!-- image-->  
Benjamina

<!-- image-->  
Monstera-2

<!-- image-->  
Umbellata-5

Fig. 9. Visual examples of branch structure reconstruction. The front view (View 1) and the top view (View 2) show two viewpoints across plants. We render the inferred branch graph with edge volumes parameterized by the corresponding StP scales and display it in brown over the original image. Our method accurately captures the underlying branching structure, even across heavily occluded regions.  
TABLE III  
QUANTITATIVE COMPARISONS OF BRANCH STRUCTURE RECONSTRUCTION ACCURACY. THE BEST RESULTS ARE HIGHLIGHTED BOLD.
<table><tr><td rowspan="2">Methods</td><td colspan="10">Structural error [mm] â</td></tr><tr><td>UmBELLata-1</td><td>AsH</td><td>UmbeLLata-2</td><td>UMBELLata-3</td><td>UMBELLaTA-4</td><td>BeNJamina</td><td>MONsTERA-1</td><td>MONSTERA-2</td><td>UMBELLaTA-5</td><td>ARTIFICIaL Mean 64.7</td></tr><tr><td>W/o Lsem</td><td>14.4</td><td>174.2</td><td>2.5</td><td>15.6</td><td>64.1</td><td>3.0</td><td>10.7</td><td>19.5</td><td>44.7</td><td>41.34</td></tr><tr><td>W/o Lbind</td><td>74.2</td><td>54.8</td><td>42.3</td><td>124.9</td><td>164.5</td><td>7.3</td><td>32.8</td><td>178.2</td><td>64.2</td><td>79.54</td></tr><tr><td>W/o Lop</td><td>17.7</td><td>13.5</td><td>3.6</td><td>7.3</td><td>55.8</td><td>4.4</td><td>15.9 24.3</td><td>17.7 43.2</td><td>52.2 68.4 73.7</td><td>25.76</td></tr><tr><td>W/o Lcs</td><td>32.9</td><td>46.6</td><td>2.7</td><td>3.2</td><td>86.7</td><td>3.6</td><td>41.8</td><td></td><td>53.3 40.1</td><td>35.70</td></tr><tr><td>Wo Lg</td><td>57.8</td><td>26.8</td><td>3.1</td><td>6.9</td><td>105.4</td><td>4.7</td><td>23.3 34.3</td><td>96.7</td><td>58.7</td><td>42.52</td></tr><tr><td>W/o reweighted MST</td><td>88.2</td><td>21.0</td><td>4.2</td><td>96.1</td><td>71.4</td><td>16.3 2.5</td><td></td><td>64.9</td><td>77.2</td><td>54.58</td></tr><tr><td>Ours (full model)</td><td>12.1</td><td>14.7</td><td>2.1</td><td>5.5</td><td>54.3</td><td></td><td>16.5</td><td>37.9</td><td>46.3</td><td>20.43</td></tr></table>

<!-- image-->

<!-- image-->

<!-- image-->  
Scene 1

<!-- image-->

<!-- image-->  
Scene 2

<!-- image-->  
Scene 3

<!-- image-->

<!-- image-->  
Scene 4

<!-- image-->

<!-- image-->  
Scene 5  
Fig. 10. In-the-wild applications. Several outdoor scenes with complex geometry and strong occlusions. Top row: 3DGS renderings. Bottom row: Reconstructed branch structures. Despite clutter and heavy foliage, our method infers plausible, tree-like structures in the wild.

TABLE IV  
QUANTITATIVE COMPARISONS OF LEAF INSTANCE SEGMENTATION ACCURACY. WE REPORT THE MEAN BI-DIRECTIONAL CHAMFER DISTANCE [MM] OVER ALL LEAF INSTANCES IN EACH PLANT (ASH AND BENJAMINA ARE EXCLUDED BECAUSE THEIR LEAVES ARE TOO DENSELY PACKED TO OBTAIN RELIABLE MANUAL ANNOTATIONS). THE BEST RESULTS ARE HIGHLIGHTED BOLD.
<table><tr><td rowspan="2">Methods</td><td colspan="9">Chamfer distance [mm] â(mean over leaf instances)</td></tr><tr><td>UmbeLLata-1</td><td>UmbeLLata-2</td><td>UmbeLlata-3</td><td>UmbeLlata-4</td><td>MONsTERA-1</td><td>MONStERA-2</td><td>UMbeLLata-5</td><td>ARtificial</td><td>Mean</td></tr><tr><td>Clustering on leaf ApPs points [42]</td><td>10.1</td><td>1.52</td><td>11.1</td><td>1.71</td><td>2.64</td><td>4.05</td><td>1.31</td><td>6.23</td><td>4.83</td></tr><tr><td>Clustering on leaf StPs (ours)</td><td>3.21</td><td>1.05</td><td>4.23</td><td>1.19</td><td>1.76</td><td>2.61</td><td>1.42</td><td>3.14</td><td>2.33</td></tr></table>

## B. Results

1) Novel-view synthesis: To evaluate the effectiveness of our proposed method, we conduct novel-view synthesis experiments on multiple real-world plant datasets and compare the rendering quality against the baseline 3DGS. As shown in Table I, our method outperforms the baseline 3DGS across all samples, demonstrating improved rendering quality. Additionally, we conduct an ablation study to analyze the impact of our proposed soft binding between ApPs and StPs. Specifically, we evaluate our method without the binding loss $\mathcal { L } _ { \mathrm { b i n d } }$ . The results show that removing the binding loss leads to a degradation in rendering quality, highlighting their importance in refining structure-aligned Gaussian splitting.

2) 3D branch segmentation: Quantitative results on 3D branch extraction are summarized in Table II, where GaussianPlant achieves the lowest CD on average over all plants. Fig. 7 further visualizes the rendered feature maps and shows that our semantic field achieves clearer separation between leaves and branches. These observations are consistent with the visual examples shown in Fig. 8. Baselines driven purely by appearance features tend to either miss thin branchlets or be confused with leaf clutter, whereas our geometry-semanticsstructure coupling recovers cleaner, more continuous branch structures.

3) Branch structure reconstruction: Estimated branch graphs are visualized in Fig. 9, and the quantitative results for our ablations are summarized in Table III. Consistent with the evaluation on the 3D branch segmentation, our full model achieves the best accuracy. Importantly, removing structurerelated components (i.e., $\mathcal { L } _ { \mathrm { g } }$ or the reweighted MST) noticeably degrades graph accuracy, indicating that our structureaware optimization indeed improves the recovered branch topology. Fig. 10 shows visual examples of our rendered results as well as the branch structure reconstruction for outdoor in-the-wild scenes. Our method successfully recovers plausible branching in practical scenes.

4) 3D leaf instance segmentation: Table IV reports the quantitative comparisons over all instances, and Figs. 1 and 11 shows the visual examples. Our StPs-based clustering accurately segments leaf instances compared to classical clustering on the dense leaf points (recovered as ApPs in our method).

The accurate leaf instance segmentation highlights applications, such as the progressive leaf removal in Fig. 12.

## VIII. CONCLUSIONS

In this work, we presented GaussianPlant, a hierarchical 3D Gaussian Splatting framework that jointly reconstructs plant structure and appearance from multi-view RGB imagery.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
(a) Reference

<!-- image-->  
(b) Ours

<!-- image-->  
(c) DBSCAN  
Fig. 11. Visualization of leaf instance segmentation results using 3DGSâs point clouds. From left to right: reference image, the clustering on leaf StPs (ours), and naive DBSCAN clustering on leaf ApPsâ dense points. Each predicted instance is displayed in a different color.

Unlike conventional 3DGS methods that focus primarily on photorealistic rendering, GaussianPlant explicitly disentangles coarse structural primitives (StPs) from high-frequency appearance primitives (ApPs), enabling the recovery of branch topology, leaf instances, and fine-grained geometry while preserving rendering fidelity. Through geometryâappearance coupling, semantic guidance, and structure-aware regularization, our method progressively organizes 3D Gaussians into biologically meaningful representations that align with true plant architecture.

<!-- image-->  
Fig. 12. Progressive leaf removal. Instance-aware 3D leaf reconstruction can be used for practical applications. We here show an example of 3D model manipulation by removing the leaf instancesâ ApPs according to the instance labels obtained from leaf StPs clustering. Samples from left to right show increasing removal strength, and leaving a clean branch in the last column.

Experiments on both indoor and in-the-wild datasets demonstrate that GaussianPlant achieves robust structural reconstruction, outperforming baselines in 3D branch extraction, leaf/branch segmentation, and instance-wise leaf recovery, while maintaining high-quality novel-view synthesis. The resulting structured representation further enables downstream applications such as branch graph analysis, instance-level phenotyping, and CG asset manipulation.

Looking forward, GaussianPlant lays the foundation for more tightly integrated semantic-structural modeling within 3DGS. Future directions include introducing richer geometric primitives beyond cylinders and disks, improving robustness under extreme occlusion, and incorporating stronger priors or temporal cues to handle dynamic or highly complex botanical structures.

Limitations. Our accuracy is bounded by (i) the fidelity of 3DGS-derived point clouds and (ii) the discriminative power of 2D foundation features projected into 3D. Dense foliage and severe occlusions remain challenging and can lead to missing or spurious branches. In addition, the current primitives (cylinders and disks) are an approximation that may be suboptimal for certain species (e.g., conifer). We plan to incorporate richer primitives (e.g., sphere and Bezier curves) and stronger priors Â´ to improve robustness under heavy occlusion and complex geometry.

## ACKNOWLEDGMENTS

This work was supported by JSPS KAKENHI Grant Numbers JP23H05491 and JP25K03140, and JST FOREST Grant Number JPMJFR206F.

## REFERENCES

[1] L. Fu, J. Liu, J. Zhou, M. Zhang, and Y. Lin, âTree skeletonization for raw point cloud exploiting cylindrical shape prior,â IEEE Access, vol. 8, pp. 27 327â27 341, 2020.

[2] A. Jiang, J. Liu, J. Zhou, and M. Zhang, âSkeleton extraction from point clouds of trees with complex branches via graph contraction,â The Visual Computer, vol. 37, pp. 2235â2251, 2021.

[3] A. Lindenmayer, âMathematical models for cellular interactions in development ii. simple and branching filaments with two-sided inputs,â Journal of Theoretical Biology, vol. 18, pp. 300â315, 1968.

[4] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D gaussian Â¨ splatting for real-time radiance field rendering.â ACM Transactions on Graphics (TOG), vol. 42, no. 4, pp. 139â1, 2023.

[5] P. Dai, J. Xu, W. Xie, X. Liu, H. Wang, and W. Xu, âHigh-quality surface reconstruction using Gaussian surfels,â in Proceedings of ACM SIGGRAPH, 2024, pp. 1â11.

[6] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2D gaussian splatting for geometrically accurate radiance fields,â in Proceedings of ACM SIGGRAPH, 2024, pp. 1â11.

[7] T. Isokane, F. Okura, A. Ide, Y. Matsushita, and Y. Yagi, âProbabilistic plant modeling via multi-view image-to-image translation,â in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 2906â2915.

[8] Y. Yang, D. Mao, H. Santo, Y. Matsushita, and F. Okura, âNeuraLeaf: Neural parametric leaf models with shape and deformation disentanglement,â in Proceedings of IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 28 167â28 176.

[9] X. Liu, H. Santo, Y. Toda, and F. Okura, âTreeFormer: Single-view plant skeleton estimation via tree-constrained graph generation,â in Proceedings of IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025, pp. 8165â8175.

[10] Y. Lu, Y. Wang, Z. Chen, A. Khan, C. Salvaggio, and G. Lu, â3D plant root system reconstruction based on fusion of deep structure-frommotion and IMU,â Multimedia Tools and Applications, vol. 80, no. 11, p. 17315â17331, 2021.

[11] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104â4113.

[12] Y. Furukawa, C. Hernandez Â´ et al., âMulti-view stereo: A tutorial,â Foundations and Trends in Computer Graphics and Vision, vol. 9, no. 1-2, pp. 1â148, 2015.

[13] M. Klodt and D. Cremers, âHigh-resolution plant shape measurements from multi-view stereo reconstruction,â in Proceedings of European Conference on Computer Vision (ECCV), 2014.

[14] F. Okura, â3D modeling and reconstruction of plants and trees: A crosscutting review across computer graphics, vision, and plant phenotyping,â Breeding Science, vol. 72, no. 1, pp. 31â47, 2022.

[15] J. Li, X. Qi, S. H. Nabaei, M. Liu, D. Chen, X. Zhang, X. Yin, and Z. Li, âA survey on 3D reconstruction techniques in plant phenotyping: from classical methods to neural radiance fields (NeRF), 3D gaussian splatting (3DGS), and beyond,â arXiv preprint arXiv:2505.00737, 2025.

[16] A. Verroust and F. Lazarus, âExtracting skeletal curves from 3D scattered data,â in Proceedings of International Conference on Shape Modeling and Applications, 1999, pp. 194â201.

[17] A. Bucksch, R. Lindenbergh, and M. Menenti, âSkelTre: Robust skeleton extraction from imperfect point clouds,â The Visual Computer, vol. 26, pp. 1283â1300, 2010.

[18] Y. Livny, F. Yan, M. Olson, B. Chen, H. Zhang, and J. El-Sana, âAutomatic reconstruction of tree skeletal structures from point clouds,â ACM Transactions on Graphics (TOG), vol. 29, no. 6, pp. 1â8, 2010.

[19] S. Du, R. Lindenbergh, H. Ledoux, J. Stoter, and L. Nan, âAdTree: Accurate, detailed, and automatic modelling of laser-scanned trees,â Remote Sensing, vol. 11, no. 18, 2019.

[20] X. Liu, K. Xu, R. Shinoda, H. Santo, and F. Okura, âMasks-to-skeleton: Multi-view mask-based tree skeleton extraction with 3D gaussian splatting,â Sensors, vol. 25, no. 14, 2025.

[21] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNeRF: Representing scenes as neural radiance fields for view synthesis,â in Proceedings of European Conference on Computer Vision (ECCV), 2020, pp. 405â421.

[22] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting SLAM,â in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18 039â18 048.

[23] Z. Chen, F. Wang, Y. Wang, and H. Liu, âGSGEN: Text-to-3D using gaussian splatting,â in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 401â21 412.

[24] L. Hu, H. Zhang, Y. Zhang, B. Zhou, B. Liu, S. Zhang, and L. Nie, âGaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians,â in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 634â644.

[25] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4D gaussian splatting for real-time dynamic scene rendering,â in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20 310â20 320.

[26] A. Guedon and V. Lepetit, âSuGaR: Surface-aligned gaussian splatting Â´ for efficient 3D mesh reconstruction and high-quality mesh rendering,â in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 5354â5363.

[27] T. Ojo, T. La, A. Morton, and I. Stavness, âSplanting: 3D plant capture with gaussian splatting,â in Proceedings of SIGGRAPH Asia Technical Communications, 2024.

[28] D. Zhang, J. Gajardo, T. Medic, I. Katircioglu, M. Boss, N. Kirchgessner, A. Walter, and L. Roth, âWheat3DGS: In-field 3D reconstruction, instance segmentation and phenotyping of wheat heads with gaussian splatting,â in Proceedings of Computer Vision and Pattern Recognition (CVPR) Workshop, 2025.

[29] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick, âSegment anything,â in Proceedings of IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4015â4026.

[30] S. Adebola, S. Xie, C. M. Kim, J. Kerr, B. M. van Marrewijk, M. van Vlaardingen, T. van Daalen, E. van Loo, J. L. S. Rincon, E. Solowjow et al., âGrowSplat: Constructing temporal digital twins of plants with gaussian splats,â in Proceedings of IEEE International Conference on Automation Science and Engineering (CASE), 2025, pp. 1766â1773.

[31] Z. K. Hartley, L. A. Stuart, A. P. French, and M. P. Pound, âPlant-Dreamer: Achieving realistic 3D plant models with diffusion-guided gaussian splatting,â in Proceedings of IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2025.

[32] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, Â´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., âDINOv3,â arXiv preprint arXiv:2508.10104, 2025.

[33] P. Couairon, L. Chambon, L. Serrano, J.-E. Haugeard, M. Cord, and N. Thome, âJAFAR: Jack up any feature at any resolution,â in Proceedings of Annual Conference on Neural Information Processing Systems (NeurIPS), 2025.

[34] Y. Livny, F. Yan, M. Olson, B. Chen, H. Zhang, and J. El-Sana, âAutomatic reconstruction of tree skeletal structures from point clouds,â ACM Transactions on Graphics (TOG), vol. 29, no. 6, 2010.

[35] Q. Wu, J. Zheng, and J. Cai, âSurface reconstruction from 3D gaussian splatting via local structural hints,â in Proceedings of European Conference on Computer Vision (ECCV), 2024, pp. 441â458.

[36] Y. Li, C. Lyu, Y. Di, G. Zhai, G. H. Lee, and F. Tombari, âGeoGaussian: Geometry-aware gaussian splatting for scene rendering,â in Proceedings of European Conference on Computer Vision (ECCV), 2024, pp. 441â 457.

[37] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, âDepth anything v2,â in Proceedings of Annual Conference on Neural Information Processing Systems (NeurIPS), vol. 37, 2024, pp. 21 875â 21 911.

[38] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi, âFeature 3DGS: Supercharging 3D Gaussian splatting to enable distilled feature fields,â in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21 676â21 685.

[39] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., âLearning transferable visual models from natural language supervision,â in Proceedings of International Conference on Machine Learning (ICML), 2021, pp. 8748â 8763.

[40] B. Li, K. Q. Weinberger, S. Belongie, V. Koltun, and R. Ranftl, âLanguage-driven semantic segmentation,â in Proceedings of International Conference on Learning Representations (ICLR), 2022.

[41] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, âLangsplat: 3D language gaussian splatting,â in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20 051â 20 060.

[42] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, âA density-based algorithm for discovering clusters in large spatial databases with noise,â in Proceedings of ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), vol. 96, no. 34, 1996, pp. 226â231.