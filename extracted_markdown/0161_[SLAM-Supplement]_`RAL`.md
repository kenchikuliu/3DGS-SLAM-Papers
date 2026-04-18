# Rad-GS: Radar-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments

Renxiang Xiaoâ , Wei Liuâ , Yuanfan Zhang, Yushuai Chen, Jinming Chen, Zilu Wang, Liang Huâ

AbstractâWe present Rad-GS, a 4D radar-camera SLAM system designed for kilometer-scale outdoor environments, utilizing 3D Gaussian as a differentiable spatial representation. Rad-GS combines the advantages of raw radar point cloud with Doppler information and geometrically enhanced point cloud to guide dynamic object masking in synchronized images, thereby alleviating rendering artifacts and improving localization accuracy. Additionally, unsynchronized image frames are leveraged to globally refine the 3D Gaussian representation, enhancing texture consistency and novel view synthesis fidelity. Furthermore, the global octree structure coupled with a targeted Gaussian primitive management strategy further suppresses noise and significantly reduces memory consumption in largescale environments. Extensive experiments and ablation studies demonstrate that Rad-GS achieves performance comparable to traditional 3D Gaussian methods based on camera or LiDAR inputs, highlighting the feasibility of robust outdoor mapping using 4D mmWave radar. Real-world reconstruction at kilometer scale validates the potential of Rad-GS for large-scale scene reconstruction.

Index TermsâRadar, Mapping, SLAM, Range Sensing, Multisensor Fusion, 3D Gaussian Splatting

## I. INTRODUCTION

4D millimeter wave (mmWave) radar has emerged as an important sensing modality complementary to cameras in autonomous driving and mobile robots, offering reliable allweather perception performance. Despite its widespread adoption, high-fidelity mapping in large-scale outdoor environments using radar-vision fusion remains underexplored. The existing 4D Radar-camera fusion Simultaneous Localization and Mapping (SLAM) framework [1], [2] relies on sparse features or volumetric grids and is limited by the noise and inherent sparsity of radar signals. The issues, including random multipath effects and low point density, impair the geometric fidelity of radar point clouds and hinder accurate alignment with visual data. Although existing attempts to denoise and densify millimeter-wave radar point clouds have achieved initial results [3]â[5], the enhanced and densified radar point cloud discards Doppler information and suffers from inconsistency between consecutive data due to random noises in the data generation process. These limitations pose significant challenges for achieving high-fidelity mapping with 4D radar-camera fusion.

The advent of 3DGS [6], [7] has opened up a promising path for radar-vision mapping. By using Gaussian primitives to represent scenes, 3DGS can provide photo-realistic scenario generation for simulation, and enables the creation of domainaccurate synthetic data for training perception networks, which is well-suited for building detailed maps. However, dynamic occlusions and viewpoint-related artifacts remain unresolved with visual-only multi-view registration. Notably, Doppler velocity measurements from 4D radar can serve as a cue for identifying moving objects. Yet, segmenting and accurately reprojecting dynamic scatterers remains non-trivial due to the limited spatial resolution of raw 4D radar. To address these challenges, we propose a hybrid strategy that associates the sparse raw radar measurements containing Doppler information with the enhanced point cloud to identify dynamic regions. Then the radar-based dynamic perception is fused with visual information to construct a dynamic-free, 3D Gaussian map for large-scale outdoor scenes.

To summarize, we present Rad-GS, the first SLAM framework that fuses 4D radar and images through 3D Gaussian representations. To bridge the gap between sparse raw radar and dense visual data, we incorporate a radar enhancement module [5] that fuses sparse, noisy radar measurements with image-guided denoised representations, supporting singleframe dynamic object removal. To enable efficient, scalable mapping, we design a global Gaussian octree strategy with adaptive anisotropic covariance assignment based on local roughness, allowing compact yet expressive scene encoding across large environments. The main contributions of the paper are threefold:

1) We present the first unified 4D radarâcamera SLAM framework using 3D Gaussian representation, enabling real-time, dynamic-free scene reconstruction of outdoor environments;

2) We propose a single-frame dynamic object removal method that leverages raw sparse 4D radar data with Doppler information combined with densely enhanced geometric point clouds to guide dynamic object masking in pixels;

3) We propose a global octree maintenance strategy, combined with Gaussian primitives merging and splitting, for memory-efficient incremental mapping with highprecision positioning over expanded areas.

## II. RELATED WORKS

## A. Colmap-Free 3DGS

To address the scale ambiguity in 3DGS methods that rely on monocular structure from motion, recent research efforts have sought to introduce scale-aware alternatives. CF-3DGS [8] utilizes a monocular deep network to provide initial geometric scales and priors as a replacement for COLMAP. LIV-GaussMap [9] combines an existing global map generated by LiDAR and Inertial data. SfM-Free 3DGS [10] uses video frame interpolation to smooth camera motion and improve pose estimation. InstantSplat [11] uses off-the-shelf models and pre-computed initial camera poses to generate dense pixellevel multi-view stereo point clouds.

<!-- image-->  
Fig. 1: Overview of Rad-GS: The system comprises a dynamic object removal module, followed by a Gaussian map construction that relies on tracking and map refinement. An octree-based management strategy employs adaptive merging and pruning for Gaussian primitives, yielding a coherent pipeline that transforms raw 4D radar and image data of a kilometerscale dynamic environment into a memory-efficient, dynamic-free static 3D Gaussian map.

Another approach is to use SLAM methods from sequential inputs during the optimization process. MonoGS [12] implements 3DGS-based indoor SLAM. In large outdoor scenes, LiV-GS [13] uses Gaussian ellipsoids and LiDAR point clouds for normal alignment for faster and more exact localization. GS-LIVM [14], VINGS-Mono [15] and GS-LIVO [16] integrate 3D Gaussian representations into tightly coupled multisensors for real-time mapping. Although existing methods have achieved excellent performance, they overlook the effect of dynamic objects on localization and rendering. Our method extends this line of work by integrating image and 4D radar Doppler information to suppress dynamic object artifacts. By combining radar-derived motion cues with vision-based reconstruction, we retain the core advantages of SLAM, including continuous pose refinement, drift suppression, and scalability, while delivering dynamic-free, high-fidelity maps.

## B. Dynamic Object Removal in 3DGS

Dynamic object suppression is critical for ensuring the geometric and visual fidelity of 3DGS-based reconstructions. One popular approach is to use Language-driven techniques. Langsplat [17] first introduces 3DGS for modeling language fields in 3d space, and they employ Segment Model [18] and CLIP [19] to produce hierarchical semantic masks. T-3DGS [20] proposes an explicit dynamic object filtering mechanism based on geometric occlusion detection and semantic priors.

An alternative direction relies on residual-based segmentation. Robust 3DGS [21] directly removes dynamic and unstable pixel regions through geometric occlusion confidenceguided training loss. SLS [22] utilizes sparse pixel sampling and reprojection consistency loss to automatically remove unstable Gaussian primitives. DGD [23] detects dynamic objects by tracking changes in color or position over time via rendering residuals, while Hybrid-GS [24] identifies potential dynamic pixels by using optical flow and depth consistency detection. DGGS [25] analyzes the changing characteristics of spatial regions in the temporal dimension to distinguish dynamic targets and construct dynamic masks.

However, existing methods relying on image residual calculation or text semantic alignment are unsuitable for largescale outdoor scenes with varying illumination. To address these issues, our method exploits the Doppler capability of mmWave radar to detect dynamic objects directly in 3D space using only a single frame. This enables robust and precise mask generation, even under occlusion or sparse visual cues, and enhances the stability of outdoor reconstructions at scale.

## III. METHODOLOGY

Our Rad-GS is a SLAM system that fuses 4D radar and a monocular camera, leveraging 3D Gaussian representations to reconstruct kilometer-scale outdoor scenes. The core element of scene representation is modelled as a Gaussian:

$$
G ( x ) = e ^ { - \frac { 1 } { 2 } ( x - \mu ) ^ { T } \Sigma ^ { - 1 } ( x - \mu ) } .\tag{1}
$$

where Âµ and Î£ respectively denote the center and covariance matrix of the Gaussians. Then in blending process, these Gaussians will be multiplied by the opacity factor Î±.

## A. System Overview

As illustrated in Fig. 1, the proposed pipeline consists of three modules: 4D radar input augmentation and Dopplerguided dynamic removal, front-end pose tracking and backend refinement, and octree-based large-scale point cloud management.

First, the sparse raw 4D radar point cloud is denoised and densified by CMDF [5], which obtains a visually enhanced radar point cloud that keeps target continuity but omits

<!-- image-->  
(a) Dynamic Labelling (Red points) in Raw Radar Data

<!-- image-->

(b) Enhanced Radar Segment  
<!-- image-->  
(c) Image Mask Generation  
Fig. 2: Doppler-guided dynamic object removal process. (a) Utilize self-motion estimation to detect dynamic points and initialize the octree. (b) Propagate dynamic points and octree nodes to the enhanced radar point cloud. (c) Project octree cells onto the image plane for dynamic object segmentation.

Doppler velocity information. The enhanced radar point cloud is projected onto the image pixel to generate a depth image. Then, the dynamic mask is directly generated in the Dopplerguided dynamic object removal module without requiring any prior pose estimation.

In the front-end, the pose is estimated by aligning the enhanced radar point cloud to the map with keyframes selected based on shared visibility. The depth fields of these keyframes are merged into the existing octree, after which new Gaussian basis cells are inserted as needed. Finally, the images unsynchronized with 4D radar frames are combined with interpolated pose constraints to refine the rendering quality in the back-end, facilitating higher-fidelity localization and photorealistic Gaussian rendering. Throughout all time, a global octree-based Gaussian map is maintained to ensure lightweight and consistent large-scale mapping.

## B. Data Augmentation and Doppler-guided Dynamic Removal

To segment dynamic objects directly in the image domain, we adopt a three-step pipeline: 1) extract dynamic objects indices in raw radar with Doppler information, 2) label dynamic points in the enhanced radar point cloud, and 3)

project dynamic objects voxels onto the image plane for pixel segmentation.

Dynamic index generation. Based on the Doppler-based ego-motion model in [26], we extend it to classify each radar detection as dynamic or static while estimating the platform velocity. The relation between vehicle velocity and the Doppler measurements is modelled as below:

$$
\boldsymbol { v } _ { \mathrm { d o p } , i } = \mathbf { r } _ { i } ^ { \mathsf { T } } \mathbf { v } _ { m } = \left[ \sin \theta _ { y , i } \cos \theta _ { p , i } \right] ^ { \mathsf { T } } \mathbf { v } _ { m }\tag{2}
$$

where $\mathbf { v } _ { m } \ \triangleq \ \left[ v _ { m x } v _ { m y } v _ { m z } \right] ^ { \mathsf { T } } , \ \theta _ { y , i }$ and $\theta _ { p , i }$ denote azimuth and elevation, respectively.

The unbiased least-squares solution to (2) is given below:

$$
\hat { \mathbf { v } } _ { m } = ( \mathbf { X } ^ { \mathsf { T } } \mathbf { X } ) ^ { - 1 } \mathbf { X } ^ { \mathsf { T } } \mathbf { y } ,\tag{3}
$$

where $\hat { \mathbf { v } } _ { m }$ is the estimated platform velocity.

For each detection, the estimated Doppler is $\hat { v } _ { \mathrm { d o p } , i } = \mathbf { r } _ { i } ^ { \mathsf { T } } \hat { \mathbf { v } } _ { m }$ As shown in Fig. 2 (a)-(b), a detection is marked dynamic index if

$$
\left| v _ { \mathrm { d o p } , i } - \hat { v } _ { \mathrm { d o p } , i } \right| > \delta _ { v } ,\tag{4}
$$

where $\delta _ { v }$ is a threshold consistent with the Doppler noise variance. All dynamic detections are excluded from the subsequent static-map construction, and the final estimated velocity is used for back-end refinement.

Octree-Guided Pixel Mask Generation. We propose an octreeâguided mask generation that fuses Doppler information and enhanced radar geometry to delimit dynamic objects in the image plane as shown in Fig. 2 (b)-(c). Let $\mathcal { P } = \{ p _ { i } \} , \hat { \mathcal { P } } =$ $\{ \hat { p } _ { i } \}$ denote the raw sparse radar point cloud and the enhanced radar point cloud for a single frame, respectively. We build an adaptive octree $\mathcal { O } ^ { \lambda }$ of depth Î» over , where each leaf node $n \in \mathcal { O } ^ { \lambda }$ is assigned a binary dynamic label:

$$
L ( n ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } n { \mathrm { ~ c o n t a i n s ~ d y n a m i c ~ i n d e x ~ p o i n t } } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{5}
$$

To incorporate the enhanced geometry representation, let $\{ n _ { k } \} _ { k = 1 } ^ { K }$ be the K nearest leaf nodes of $\hat { p } _ { i }$ and define

$$
L ( { \widehat { p } } ) = { \left\{ \begin{array} { l l } { 1 , } & { \displaystyle \sum _ { k = 1 } ^ { K } L ( n _ { k } ) > { \frac { K } { 2 } } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{6}
$$

Dynamic labels are propagated from the raw sparse octant to the enhanced cloud by assigning each label in the enhanced radar. Then we reconstruct an adaptive octree with the same maximum depth Î». The deepest nodes have the same label as the corresponding points, and if the leaf node is empty, we inherit the label of its parent node to maintain continuity.

Next, we propagate dynamic labels upward to enforce continuity. We first build an adjacency graph on the set of dynamic leaves $\{ n : L ( n ) = 1 \}$ , where two leaves are adjacent if their octree cells share a face. Let $\{ C _ { j } \}$ be the connected components (clusters) of this graph. To guarantee that only the minimal connected subtree covering each dynamic cluster is marked, for each cluster $C _ { j }$ , we compute its lowest common ancestor. Then mark every node on the path from each leaf $d \in C _ { j }$ up to $m _ { j }$ as dynamic.

Projecting the leaf with dynamic label boundaries through the camera model produces a 2D bounding box. The box guides EfficientSAM [27] to extract precise object contours within the box only rather than the entire image, which accelerates the segmentation process and increases dynamic object removal accuracy. Subsequently, the point cloud with static label and the image with the dynamic mask are fed into the front-end module as inputs.

## C. Radar-Visual System

Front-End Tracking: Inspired by [13], the front-end module adopts covariance-guided radar-Gaussian matching for shape-adaptive feature matrices between the Gaussian primitives and the input point cloud. The covariance matrix $\Sigma =$ diag $( \sigma _ { 1 } ^ { 2 } , \sigma _ { 2 } ^ { 2 } , \sigma _ { 3 } ^ { 2 } )$ is extracted from each candidate Gaussian primitive, where the eigenvalues satisfy $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \sigma _ { 3 } > 0 .$ For the enhanced radar point cloud, local geometric anisotropy is estimated via SVD. According to the shape ratio and the constant threshold $\tau \in ( 0 , 1 )$ , we divide it into three types of shape-adaptive feature matrix $\pmb { v } \in \mathbb { R } ^ { 3 \times 3 }$

$$
\begin{array} { r } { v = \left\{ \begin{array} { l l } { \mathbf { e } _ { 3 } \mathbf { e } _ { 3 } ^ { \top } , } & { \frac { \sigma _ { 3 } } { \sigma _ { 2 } } \leq \tau , } \\ { \mathbf { e } _ { 1 } \mathbf { e } _ { 1 } ^ { \top } , } & { \frac { 1 } { \tau } \leq \frac { \sigma _ { 1 } } { \sigma _ { 2 } } , } \\ { \beta _ { 1 } \mathbf { e } _ { 3 } \mathbf { e } _ { 3 } ^ { \top } + \beta _ { 2 } \mathbf { e } _ { 1 } \mathbf { e } _ { 1 } ^ { \top } + ( 1 - \beta _ { 1 } - \beta _ { 2 } ) \Sigma ^ { - 1 } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{7}
$$

where $\mathbf { e } _ { i }$ is the unit eigenvector corresponding to the $\sigma _ { i } , \beta _ { 1 }$ and $\beta _ { 2 }$ are hyperparameters.

We align the input point cloud with its closest Gaussian primitives using covariance-guided matching, and iteratively optimize the rigid-body pose $T _ { W } ^ { C _ { t - 1 } ( k ) }$ at the k-th iteration:

$$
E = \sum _ { \pmb { x } _ { p } \in P _ { R } } w ( \pmb { x } _ { p } ) \big ( \pmb { v } _ { \pmb { x } _ { p } } \cdot ( \pmb { T } _ { W } ^ { C _ { t - 1 } ( k ) } \pmb { x } _ { p } - \pmb { \mu } _ { g } ) \big ) ^ { 2 } + R ( \pmb { v } _ { \pmb { x } _ { p } } , \pmb { v } _ { g } ) ,\tag{8}
$$

where $\mu _ { g }$ is the nearest Gaussian centroid to the point $x _ { p }$ in the enhanced radar set $P _ { R } . \ v _ { x _ { p } }$ and $v _ { g }$ are computed via (7) for the enhanced radar point and the corresponding Gaussian, respectively. The function $w ( \cdot )$ represents densitybased weighting, while the regularization term $R ( \cdot )$ enforces directional consistency, the same as [13].

Back-End Refinement: The synchronized radar and image frames are used for localization, while the rest of the unsynchronized images are used for refining rendering. We interpolate the unsynchronized images to obtain the camera poses using cubic Hermite interpolation. Let $P _ { 0 }$ and $P _ { 1 }$ denote the poses of two adjacent keyframes, and $V _ { 0 } , V _ { 1 }$ be the egovelocity estimated from (3), respectively. The cubic Hermite polynomials obey the following bound constraints:

$$
H ( 0 ) = P _ { 0 } , H ( 1 ) = P _ { 1 } , \dot { H } ( 0 ) = V _ { 0 } , \dot { H } ( 1 ) = V _ { 1 } .\tag{9}
$$

We interpolate translation with cubic Hermite using world-frame linear-velocity boundary conditions $\begin{array} { r l } { \dot { \bf p } ( t _ { i } ) } & { { } = } \end{array}$ $\mathbf { v } _ { i } ^ { w } , \ \dot { \mathbf { p } } ( t _ { i + 1 } ) = \mathbf { v } _ { i + 1 } ^ { w }$ , and we interpolate rotation on SO(3) via a constant angular velocity ${ \boldsymbol { \omega } } = { \bar { \Delta } { t } ^ { - 1 } } \log ( { R _ { i } ^ { \top } } { R _ { i + 1 } } ) ^ { \vee }$ . The resulting pose curve $H ( t ) = [ R ( t ) , \mathbf { p } ( t ) ; 0 , 1 ]$ satisfies $\dot { H } ( t ) =$ $H ( t ) \xi ^ { \wedge } ( t )$ with body-frame twist $\xi ( t ) = [ \pmb { \omega } ; R ( t ) ^ { \top } \dot { \mathbf { p } } ( t ) ] \ \in$ $s e ( 3 )$

The pose of any frame between $P _ { 0 }$ and $P _ { 1 }$ is then obtained from samples in the interpolated curve. The depth and color rendering process of the Gaussian image $G ^ { s }$ is expressed as $\begin{array} { r } { \begin{array} { l } { D _ { \mathrm { r e n d e r } } } \\ { \qquad = } \end{array} \sum _ { g _ { i } \in G ^ { s } } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \begin{array} { l } { { \bf \bar { \alpha } } _ { \alpha _ { j } } } \end{array} ) } \end{array}$ and $\begin{array} { r l } { C _ { \mathrm { r e n d e r } } } & { { } = } \end{array}$ $\textstyle \sum _ { g _ { i } \in G ^ { s } } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 ^ { * } } ( 1 - \alpha _ { j } )$ , where $d _ { i }$ and $c _ { i }$ represent the depth distance and color along the camera ray to the Gaussian image $g _ { i }$ , respectively.

Isotropic Loss  
Ours  
Normal Loss  
<!-- image-->  
Fig. 3: Effect of Roughness Restriction: Top: Render images. Bottom: Magnified details. The isotropic constraint shows blurred edges of buildings, the normal loss-guided Gaussian primitive rendering sacrifices the rendering of non-planar surfaces, while our loss function achieves the best balance between structural fidelity and texture preservation.

Loss function: To accommodate the diverse surface roughness of outdoor objects (as shown in Fig. 3), ranging from vertically cracked tree trunks to uniformly smooth buildings and road signs, we construct the composite loss:

$$
\begin{array} { r } { \mathcal { L } = ( 1 - \lambda _ { 1 } ) E _ { p h o } + \lambda _ { 1 } E _ { g e o } + \lambda _ { 2 } E _ { r o u } , } \end{array}\tag{10}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are hyperparameters.

The photometric error $E _ { p h o }$ and the geometric error $E _ { g e o }$ are widely used. The former represents the difference between the real visual image and the rendered image, and the latter measures the difference between the enhanced radar input and the rendered depth image. In addition, we introduce a third roughness constraint $E _ { r o u }$ as:

$$
E _ { r o u } = \left\{ \begin{array} { l l } { \| \delta _ { \sigma } \| ^ { 2 } } & { \tau \leq \frac { \sigma _ { m e a n } } { \sigma _ { 3 } } \leq \frac { 1 } { \tau } } \\ { \| \delta _ { \sigma } \| ^ { 2 } + \gamma \| \sigma _ { m i n } \| ^ { 2 } } & { e l s e } \end{array} \right.\tag{11}
$$

where $\sigma _ { m e a n }$ represents the average of the two closest scales of the Gaussian, and $\delta _ { \sigma }$ represents the numerical difference between the two scales. Ï is the same as (7) and $\gamma$ is hyperparameters. This constraint ensures that the Gaussian ellipsoid encoding the local object shape is appropriately adapted to the roughness level of each surface.

## D. Adaptive Gaussian Octree Management

Unlike [16] and [28], we introduce an incremental global management strategy to control the growth of Gaussian ellipsoid primitives during optimization, thereby eliminating the redundant splitting and pruning process. The results of the incremental global optimization are shown in Fig. 4.

Starting from the initial frame, all points are set as the initial Gaussian map, and a multi-level octree is constructed based on the primitive size. At level $l ,$ the voxel size $\delta _ { l }$ and mean gradient threshold $\epsilon _ { l }$ are predefined, and updated by

<!-- image-->

<!-- image-->  
Fig. 4: Illustration of the visual improvement through incremental global optimization. The initialized global 3D Gaussian map (left) and the refined representation with enhanced geometric fidelity and texture realism (right).

$$
\delta _ { l + 1 } = \frac { \delta _ { l } } { 2 } , \quad \epsilon _ { l + 1 } = 2 \epsilon _ { l } ,\tag{12}
$$

where the value of parameter $l , \delta _ { l }$ and $\epsilon _ { l }$ is the same as [29]. In our system, with 15,000 points, constructing a 6-level octree requires approximately 0.5 ms, which is consistent with the theoretical complexity O(N log N), where N denotes the total number of points in the point cloud.

Gaussians Merge: To reduce the impact of noisy depth measurement noise from enhanced radar point cloud and to prevent redundant Gaussian creation, we merge point sets whose projections overlap the Gaussian centers in the global octree. The Gaussian center in each affected node is adjusted accordingly.

The depth uncertainty is modeled by:

$$
d = d _ { \mathrm { t r u e } } + \varepsilon , \quad \varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) ,\tag{13}
$$

where d denotes the radar range observation, $d _ { \mathrm { t r u e } }$ the true distance, and Îµ zero-mean Gaussian noise with variance $\sigma ^ { 2 }$ Each Gaussian primitive $G _ { i }$ in the map is characterized by a mean $\pmb { \mu } _ { i }$ and covariance $\Sigma _ { i }$ , while every new point p inherits its own covariance $\Sigma _ { p }$ derived from front-end alignment (8). Assuming an isotropic measurement covariance $\sigma _ { \varepsilon } ^ { 2 } \mathbf { I } _ { 3 }$ , the primitive is updated to $G _ { i } ^ { \prime } = \{ \mu _ { i } ^ { \prime } , \Sigma _ { i } ^ { \prime } \}$ by

$$
\pmb { \Sigma } _ { i } ^ { \prime } = \big ( \pmb { \Sigma } _ { i } ^ { - 1 } + \pmb { \sigma } _ { \varepsilon } ^ { - 2 } \pmb { \mathrm { I } } _ { 3 } + \pmb { \Sigma } _ { p } ^ { - 1 } \big ) ^ { - 1 } ,\tag{14}
$$

$$
\pmb { \mu } _ { i } ^ { \prime } = \pmb { \Sigma } _ { i } ^ { \prime } \big ( \pmb { \Sigma } _ { i } ^ { - 1 } \pmb { \mu } _ { i } + \pmb { \sigma } _ { \varepsilon } ^ { - 2 } d \mathbf { e } _ { r } + \pmb { \Sigma } _ { p } ^ { - 1 } \mathbf { p } \big ) ,\tag{15}
$$

where $\mathbf { e } _ { r }$ is the unit vector along the radar beam.

Gaussians Split: We introduce the octree conditional Gaussian constraint, which ensures that spatial splitting follows the structure of the octree without increasing node count. Specifically, for each point a acquired through back-end refinement, the nearest Gaussian b is its nearest neighbor in level Î»  1; then:

$$
p ( a \mid b ) \sim { \mathcal { N } } { \big ( } \mu _ { a } ( b ) , \Sigma _ { b } { \big ) } .\tag{16}
$$

<!-- image-->  
Fig. 5: Comparison of dynamic object removal.

TABLE I: Comparison of Dynamic Removal
<table><tr><td rowspan="2">Method</td><td colspan="3">Nyl1</td><td colspan="3">Nyl2</td><td colspan="3">Loop2</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>SLS [22]</td><td>21.080</td><td>0.750</td><td>0.486</td><td>17.430</td><td>0.499</td><td>0.696</td><td>18.830</td><td>0.615</td><td>0.473</td></tr><tr><td>T-3DGS [20]</td><td>20.800</td><td>0.710</td><td>0.411</td><td>19.010</td><td>0.685</td><td>0.398</td><td>20.050</td><td>0.729</td><td>0.385</td></tr><tr><td>3DGS [6]</td><td>21.230</td><td>0.726</td><td>0.575</td><td>21.490</td><td>0.695</td><td>0.401</td><td>17.794</td><td>0.594</td><td>0.620</td></tr><tr><td>3DGS w removal</td><td>22.340</td><td>0.746</td><td>0.508</td><td>21.860</td><td>0.718</td><td>0.366</td><td>18.102</td><td>0.613</td><td>0.594</td></tr><tr><td>Ours w/o removal</td><td>21.420</td><td>0.762</td><td>0.473</td><td>21.460</td><td>0.601</td><td>0.449</td><td>19.510</td><td>0.718</td><td>0.319</td></tr><tr><td>Ours</td><td>23.650</td><td>0.798</td><td>0.389</td><td>21.990</td><td>0.613</td><td>0.437</td><td>20.690</td><td>0.780</td><td>0.319</td></tr></table>

where $\mu _ { x } ( b )$ is the center of new Gaussian ellipsoid split from Gaussian b through normal distribution sampling.

## IV. EXPERIMENT & ANALYSIS

## A. Implementation Details

We evaluate real-time 3D Gaussian map construction from four perspectives: dynamic object removal, rendering quality, localization accuracy, and computational efficiency. Additionally, we conduct an ablation study on the loss function design. All experiments were run on a single RTX 4090 GPU.

The dynamic object removal module is compared with two baseline methods SLS [22] and T-3DGS [21]. The localization accuracy is compared with classic 4D radar SLAM method 4DRadarSLAM [30], learning-based visual-aided radar SLAM [2], LiDAR-based SLAM [31], visual ORB-SLAM3 [32], and 3DGS-based SLAM framework (MonoGS [12], LiV-GS [13]) as a baseline. For rendering and computation efficiency evaluation, we compare Rad-GS with LiV-GS [13], MonoGS [12], and classic 3DGS with ground truth (GT) pose and with odometry, respectively.

To evaluate trajectory error, we used the open-source tool rpg trajectory evaluation [33] to compute both Absolute Trajectory Error (ATE) and Relative Error (RE), measuring the ATE root-mean-square error (RMSE) drift (m) and average rotational RMSE drift (Â°/100 m). Rendering quality is evaluated using the metrics of peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), and learned perceptual patch similarity (LPIPS).

## B. Datasets

NTU4DRadLM [34] provide measurements from a Livox Horizon LiDAR, a 640 480 monocular camera, an Eagle Oculii G7 4D millimetre-wave radar and the ground-truth of robot pose. Four sequences are evaluated in our experiments:

TABLE II: Quantitative Analysis for Rendering [PSNRâ SSIMâ LPIPSâ]
<table><tr><td rowspan="2">Method</td><td colspan="3">Nyl1</td><td colspan="3">Nyl2</td><td colspan="3">Garden</td><td colspan="3"> $\mathrm { L o o p } 2$ </td><td colspan="3">Campus Road</td><td colspan="3">Campus Loop</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>3DGS</td><td>21.23</td><td>0.726</td><td>0.575</td><td>21.49</td><td>0.695</td><td>0.401</td><td>20.04</td><td>0.661</td><td>0.539</td><td>17.80</td><td>0.594</td><td>0.620</td><td>19.79</td><td>0.550</td><td>0.641</td><td>19.26</td><td>0.524</td><td>0.701</td></tr><tr><td>MonoGS(Odom)</td><td>21.23</td><td>0.774</td><td>0.466</td><td>18.55</td><td>00.525</td><td>0.590</td><td>118.90</td><td>00.573</td><td>0.611</td><td>16.47</td><td>0.546</td><td>0 .556</td><td>16.69</td><td>0.389</td><td>00.707</td><td>117.36</td><td>0.478</td><td>0.633</td></tr><tr><td>MonoGS(GT)</td><td>22.54</td><td>0..801</td><td>0402</td><td>19.91</td><td>0559</td><td>0..545</td><td>19.52</td><td>00.609</td><td>00.563</td><td>17.91</td><td>0.588</td><td>00.512</td><td>117.88</td><td>0.429</td><td>00.675</td><td>18.43</td><td>00.517</td><td>00.585</td></tr><tr><td>LiV-GSs(Odom)</td><td>21.97</td><td>0.704</td><td>0.444</td><td>18.79</td><td>0.529</td><td>0586</td><td>19.29</td><td>0588</td><td>0558</td><td>116.71</td><td>0.535</td><td>0.524</td><td>18.58</td><td>0.472</td><td>0.657</td><td>18.16</td><td>0.477</td><td>0.634</td></tr><tr><td>LiV-GS(GT)</td><td>22.50</td><td>0.785</td><td>0.4133</td><td>119.81</td><td>0.549</td><td>0.542</td><td>20.02</td><td>0.633</td><td>0.510</td><td>18.01</td><td>0.579</td><td>00.501</td><td>19.57</td><td>0.503</td><td>0.630</td><td>19.53</td><td>0.511</td><td>0.598</td></tr><tr><td>Ou us(0dom)</td><td></td><td>0.798</td><td>0.3889</td><td>$2.90</td><td>0.6133</td><td>0.437</td><td>$22.95 ,</td><td>0.666</td><td>0.463</td><td>18.83</td><td>0.625</td><td>0.473</td><td>20.87</td><td>0523</td><td>0.597</td><td>20.20</td><td>0.584</td><td>0.493</td></tr><tr><td>Ours(GT)</td><td>23.92</td><td>00.812</td><td>0.37</td><td></td><td>0.633</td><td>0.400</td><td></td><td>00.715</td><td>00.372</td><td>19.65</td><td>0.647</td><td>0.446</td><td>21.40</td><td>0.558</td><td>00.562</td><td>21.35</td><td>0.617</td><td>0.445</td></tr></table>

<!-- image-->  
Fig. 6: The commercial vehicle and sensor suite visualization.

Cp, Garden, and Nyl (recorded at approximately $3 . 6 \mathrm { k m h ^ { - 1 } }$ Nyl is split equally into Nyl1 and Nyl2) and Loop2 (captured at about 25 km hâ1, of which the first 400 m are used).

Self-collected dataset was collected with a commercial vehicle travelling at roughly 20 km $\mathrm { h } ^ { - 1 }$ . As depicted in Fig. 6, the sensor suite comprises a 128-channel LiDAR, a 1920 Ã 1080 monocular camera, and the same radar. GT poses come from an RTK+IMU-aided BeiDou GNSS system. Two trajectories are provided: a 1.2 km closed campus loop and a 0.5 km open loop (called campus road). Each route contains diverse static structures, such as buildings and trees, and numerous dynamic objects, including pedestrians and vehicles.

## C. Dynamic Object Removal

This subsection validates the effectiveness of the dynamic component removal module in Rad-GS. Furthermore, we import the results with dynamic masks into the original 3DGS baseline, showing the necessity of this module.

Evaluation of dynamic object removal: The qualitative and quantitative comparisons are shown in Fig. 5 and Tab. I, respectively. Compared with T-3DGS and SLS, our method removes moving objects, e.g., several cars and tree trunks clearly, while the other two methods yield artifacts in the marked area of Fig. 5. Moreover, our approach better restores background details such as road markings and building facade edges. Even in the Loop2 sequence where limited and sparse viewpoints hinder full background recovery, our approach still outperforms T-3DGS and SLS.

We further integrate our dynamic object removal module into 3DGS and compare it with the Vallina 3DGS. As shown in the bottom four rows of Tab. I, the introduced dynamic object removal improves both our systems and 3DGS. For 3DGS, the introduced dynamic object removal boosts mean PSNR by +0.60 dB, mean SSIM by +0.021, and reduces mean LPIPS by â0.043.

TABLE III: Quantitative Analysis for Localization Accuracy
<table><tr><td rowspan="2">Method</td><td colspan="2"> $\mathrm { C p }$ </td><td colspan="2"> $\mathrm { N y l } 2$ </td><td colspan="2">Campus Loop</td></tr><tr><td> $t _ { a b s } \downarrow$ </td><td> $r _ { r e l } \downarrow$ </td><td> $t _ { a b s } \downarrow$ </td><td> $r _ { r e l } \downarrow$ </td><td> $t _ { a b s } \downarrow$ </td><td> $r _ { r e l } \downarrow$ </td></tr><tr><td>HDL-graph-slam [31]</td><td>2.691</td><td>2.157</td><td>1.342</td><td>5.998</td><td>33.515</td><td>7.206</td></tr><tr><td>4DRadarSLAM [30]</td><td>2.853</td><td>1.205</td><td>1.413</td><td>4.571</td><td>11.542</td><td>2.163</td></tr><tr><td>ORB-SLAM3 [32]</td><td>9.827</td><td>16.527</td><td>1.114</td><td>69.190</td><td>29.283</td><td>6.179</td></tr><tr><td>Visual-Aided-SLAM [2]</td><td>2.769</td><td>1.851</td><td>1.871</td><td>11.646</td><td>17.762</td><td>4.037</td></tr><tr><td>Mono-GS [12]</td><td>22.602</td><td>173.781</td><td>1.236</td><td>69.797</td><td>14.835</td><td>5.627</td></tr><tr><td>LiV-GS [13]</td><td>6.111</td><td>8.192</td><td>1.125</td><td>71.275</td><td>20.139</td><td>5.542</td></tr><tr><td>Ours</td><td>9.607</td><td>7.163</td><td>0.567</td><td>2.973</td><td>6.454</td><td>1.130</td></tr></table>

## D. Rendering Evaluation

Unlike 3DGS, which relies on Colmap as SfM input for offline reconstruction, our Rad-GS, MonoGS, and LiV-GS jointly estimate pose and reconstruct the scene. For fair comparison, all methods are fed with dynamic-cleaned enhanced radar point clouds and corresponding images.

As shown in Tab. II and Fig. 7, our Rad-GS consistently outperforms other baselines, particularly in large-scale outdoor environments characterized by complex vegetation and structural occlusions. The adaptive merging strategy within the octree efficiently handles radar noise and avoids splat artifacts. Furthermore, the proposed roughness-aware loss enhances rendering fidelity, as further validated by the ablation study in Subsection IV-G.

Notably, the performance gap between Rad-GS using GT pose and odometry is minimal, implying accurate localization and reduced error accumulation during incremental mapping.

## E. Localization Evaluation

This subsection compares Rad-GS against existing radarbased localization approaches. As shown in Fig. 8 and Tab. III, Rad-GS attains the highest localization accuracy across all sequences. Unlike 4DRadarSLAM, which removes dynamic points based on ego-motion and sometimes over-prunes, Rad-GS preserves stable global static structure in the Gaussian octree map, thereby sustaining robust localization in dynamic environments.

Loop Detection Validation: In the Cp sequence, loop-closure-based methods (HDL-graph-SLAM and 4DRadarSLAM) perform well due to LiDARâs geometric consistency, even when radar is corrupted by through-glass reflections. In this case, Rad-GS exhibits drift caused by mismatches between image and radar point clouds due to the transparent surfaces. However, on Campus Loop, where the static structure is clearer and more consistent, Rad-GS achieves drift-free performance.

## F. Memory Consumption

To assess the impact of Gaussian management, we tracked GPU memory usage over time on the NTU Cp dataset. Our system can run in real time at the perception frequency of 4D radar, with asynchronous image frame optimization and pose refinement taking approximately 0.037ms, and Gaussian map ellipsoid optimization taking approximately 0.082ms. As depicted in Fig. 9, all methods consume a similar GPU at the early stage, but the memory usage rises sharply as the number of scenes increases for other comparative methods, while Rad-GS stabilizes after 1200 frames and caps around 11 GB. At 2400 frame, Rad-GS reduces memory usage by approximately 45 %, 39 %, 35 %, and 27 % compared to LiV-GS (LiDAR), MonoGS (LiDAR), LiV-GS (radar), and MonoGS (radar), respectively. The efficient memory management results from the integration of global octree maintenance and the adaptive merging of Gaussian primitives in our Rad-GS, mitigating redundant map expansion during long-term operation.

Nyl1  
Nyl2  
Garden  
Loop2  
Campus Road  
Campus Loop  
<!-- image-->  
Fig. 7: Comparison of Rendering Results.

<!-- image-->  
(a) Cp

<!-- image-->  
(b) Campus Loop

<!-- image-->  
(c) Nyl2

Fig. 8: Localization Comparison of Different Methods. TABLE IV: Comparison of different Loss functions
<table><tr><td rowspan="2">Method</td><td colspan="3">Nyl2</td><td colspan="3">Cp</td><td colspan="3">Garden</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Isotropic [12]</td><td>20.10</td><td>0.564</td><td>0.538</td><td>20.93</td><td>0.642</td><td>0.458</td><td>20.53</td><td>0.591</td><td>0.546</td></tr><tr><td>Normal [13]</td><td>20.18</td><td>0.551</td><td>0.541</td><td>22.27</td><td>0.775</td><td>0.336</td><td>19.28</td><td>0.536</td><td>0.550</td></tr><tr><td>Rough</td><td>23.06</td><td>0.633</td><td>0.400</td><td>22.20</td><td>0.745</td><td>0.334</td><td>22.05</td><td>0.715</td><td>0.372</td></tr></table>

<!-- image-->  
Fig. 9: GPU Memory Consumption of Different Methods

## G. Ablation Study of Loss Function

We compare three candidate loss functions of Gaussian shapes adapted for the surface roughness of outdoor objects. All variants are run with identical GT poses to isolate the impact of loss design. As shown in Tab. IV, the roughnessaware loss gains 1.24 dB in PSNR, 0.053 in SSIM relative to the second-best method on average. Fig. 3 shows that the isotropic loss over-smooths high-frequency content: colour bleeding appears at building edges and the tree trunk undergoes severe geometric drift (red and blue boxes). The normalguided loss sharpens planar regions but creates wavy artefacts on rough surfaces and blurs thin objects such as the warningsign pole (yellow box). By adapting each Gaussian ellipsoid to local surface roughness, our proposed loss preserves crisp sign boundaries, corrects trunk geometry, and suppresses background streaks across the frame, yielding sharper and more faithful renderings.

## V. CONCLUSION

Our proposed Rad-GS framework unifies monocular imagery and 4D radar Doppler signals within a 3D Gaussian representation to achieve kilometer-scale outdoor localization and mapping. Doppler measurement originated from raw data together with enhanced dense radar point clouds guides 2D dynamic-object contours masks, suppressing moving-object interference before pose optimisation. Continuous map expansion is supported by a global octree maintenance strategy that incrementally merges and splits Gaussian primitives, delivering lightweight mapping without loss of localisation accuracy. Diverse experiments demonstrate that our approach achieves clear dynamic removal, enhanced map reconstruction fidelity, and reduced GPU memory consumption.

## REFERENCES

[1] J. Zhang, R. Xiao, H. Li, Y. Liu, X. Suo, C. Hong, Z. Lin, and D. Wang, â4DRT-SLAM: Robust SLAM in Smoke Environments using 4D Radar and Thermal Camera based on Dense Deep Learnt Features,â in 2023 IEEE International Conference on CIS-RAM. IEEE, 2023, pp. 19â24.

[2] Y. Zhang, R. Xiao, Z. Hong, L. Hu, and J. Liu, âAdaptive Visual-Aided 4D Radar Odometry Through Transformer-Based Feature Fusion,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 12 529â12 535.

[3] R. Zhang, D. Xue, Y. Wang, R. Geng, and F. Gao, âTowards dense and accurate radar perception via efficient cross-modal diffusion model,â IEEE Robotics and Automation Letters, 2024.

[4] K. Luan, C. Shi, N. Wang, Y. Cheng, H. Lu, and X. Chen, âDiffusionbased point cloud super-resolution for mmwave radar data,â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 11 171â11 177.

[5] Y. Zhang, R. Xiao, Y. Han, C. Ding, L. Hu, and J. Liu, âCMDF: Cross-Modal Diffusion and Fusion for 4D Radar Point Cloud Enhancement,â in Submitted to 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025.

[6] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D Gaussian Â¨ Splatting for Real-Time Radiance Field Rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[7] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and G. Drettakis, âA hierarchical 3d gaussian representation for real-time rendering of very large datasets,â ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1â15, 2024.

[8] Y. Fu, X. Wang, S. Liu, A. Kulkarni, J. Kautz, and A. A. Efros, âCOLMAP-Free 3D Gaussian Splatting,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20 796â 20 805.

[9] S. Hong, J. He, X. Zheng, C. Zheng, and S. Shen, âLIV-GaussMap: LiDAR-Inertial-Visual Fusion for Real-Time 3D Radiance Field map rendering,â IEEE Robotics and Automation Letters, 2024.

[10] B. Ji and A. Yao, âSfM-Free 3D Gaussian Splatting via Hierarchical Training,â arXiv preprint arXiv:2412.01553, 2024.

[11] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos et al., âInstantSplat: Unbounded Sparse-View Pose-Free Gaussian Splatting in 40 Seconds,â arXiv preprint arXiv:2403.20309, vol. 2, no. 3, p. 4, 2024.

[12] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian Splatting SLAM,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[13] R. Xiao, W. Liu, Y. Chen, and L. Hu, âLiV-GS: LiDAR-Vision integration for 3D Gaussian Splatting SLAM in Outdoor Environments,â IEEE Robotics and Automation Letters, vol. 10, no. 1, pp. 421â428, 2025.

[14] Y. Xie, Z. Huang, J. Wu, and J. Ma, âGS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting,â ArXiv, vol. abs/2410.17084, 2024.

[15] K. Wu, Z. Zhang, M. Tie, Z. Ai, Z. Gan, and W. Ding, âVINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes,â arXiv preprint arXiv:2501.08286, 2025.

[16] S. Hong, C. Zheng, Y. Shen, C. Li, F. Zhang, T. Qin, and S. Shen, âGS-LIVO: Real-Time LiDAR, Inertial, and Visual Multi-Sensor Fused Odometry with Gaussian Mapping,â arXiv preprint arXiv:2501.08672, 2025.

[17] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, âLangSplat: 3D Language Gaussian Splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 051â20 060.

[18] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., âSegment anything,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015â4026.

[19] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, âHierarchical text-conditional image generation with clip latents,â arXiv preprint arXiv:2204.06125, vol. 1, no. 2, p. 3, 2022.

[20] V. Pryadilshchikov, A. Markin, A. Komarichev, R. Rakhimov, P. Wonka, and E. Burnaev, âT-3DGS: Removing Transient Objects for 3D Scene Reconstruction,â arXiv preprint arXiv:2412.00155, 2024.

[21] P. Ungermann, A. Ettenhofer, M. NieÃner, and B. Roessle, âRobust 3d gaussian splatting for novel view synthesis in presence of distractors,â arXiv preprint arXiv:2408.11697, 2024.

[22] S. Sabour, L. Goli, G. Kopanas, M. Matthews, D. Lagun, L. Guibas, A. Jacobson, D. Fleet, and A. Tagliasacchi, âSpotlesssplats: Ignoring distractors in 3D Gaussian Splatting,â ACM Transactions on Graphics, 2024.

[23] I. Labe, N. Issachar, I. Lang, and S. Benaim, âDGD: Dynamic 3D Gaussians Distillation,â in European Conference on Computer Vision. Springer, 2024, pp. 361â378.

[24] J. Lin, J. Gu, L. Fan, B. Wu, Y. Lou, R. Chen, L. Liu, and J. Ye, âHybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting,â arXiv preprint arXiv:2412.03844, 2024.

[25] Y. Bao, J. Liao, J. Huo, and Y. Gao, âDistractor-Free Generalizable 3D Gaussian Splatting,â arXiv preprint arXiv:2411.17605, 2024.

[26] C. Doer and G. F. Trommer, âAn ekf based approach to radar inertial odometry,â in 2020 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI). IEEE, 2020, pp. 152â159.

[27] Y. Xiong, B. Varadarajan, L. Wu, X. Xiang, and F. Xiao, âEfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16 111â16 121.

[28] M. Wang, J. Wang, C. Xia, C. Wang, and Y. Qi, âOg-mapping: Octreebased structured 3d gaussians for online dense mapping,â arXiv preprint arXiv:2408.17223, 2024.

[29] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffoldgs: Structured 3d gaussians for view-adaptive rendering,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20 654â20 664.

[30] J. Zhang, H. Zhuge, Z. Wu, G. Peng, M. Wen, Y. Liu, and D. Wang, â4dradarslam: A 4d imaging radar slam system for large-scale environments based on pose graph optimization,â in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 8333â8340.

[31] K. Koide, J. Miura, and E. Menegatti, âA Portable Three-Dimensional LiDAR-based system for long-term and wide-area people behavior measurement,â International Journal of Advanced Robotic Systems, vol. 16, no. 2, p. 1729881419841532, 2019.

[32] C. Campos, R. Elvira, J. J. G. RodrÂ´Ä±guez, J. M. Montiel, and J. D. Tardos, âORB-SLAM3: An accurate open-source library for visual, Â´ visualâinertial, and multimap SLAM,â IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874â1890, 2021.

[33] Z. Zhang and D. Scaramuzza, âA Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry,â in IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS), 2018.

[34] J. Zhang, H. Zhuge, Y. Liu, G. Peng, Z. Wu, H. Zhang, Q. Lyu, H. Li, C. Zhao, D. Kircali et al., âNTU4DRaDLM: 4d radar-centric multi-modal dataset for localization and mapping,â in 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2023, pp. 4291â4296.