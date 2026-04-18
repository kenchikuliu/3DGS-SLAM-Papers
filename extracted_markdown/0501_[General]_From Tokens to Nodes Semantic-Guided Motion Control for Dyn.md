# FROM TOKENS TO NODES: SEMANTIC-GUIDED MO-TION CONTROL FOR DYNAMIC 3D GAUSSIAN SPLAT-TING

Jianing Chen1,2â, Zehao Li1,2â, Yujun Cai3, Hao Jiang1,2, Shuqin Gao1, Honglong Zhao1, Tianlu Mao1,2, Yucheng Zhang1,2

1Institute of Computing Technology, Chinese Academy of Sciences, ICT

2University of Chinese Academy of Sciences, UCAS

3The University of Queensland

{chenjianing23s, jianghao}@ict.ac.cn

## ABSTRACT

Dynamic 3D reconstruction from monocular videos remains difficult due to the ambiguity inferring 3D motion from limited views and computational demands of modeling temporally varying scenes. While recent sparse control methods alleviate computation by reducing millions of Gaussians to thousands of control points, they suffer from a critical limitation: they allocate points purely by geometry, leading to static redundancy and dynamic insufficiency. We propose a motion-adaptive framework that aligns control density with motion complexity. Leveraging semantic and motion priors from vision foundation models, we establish patch-token-node correspondences and apply motion-adaptive compression to concentrate control points in dynamic regions while suppressing redundancy in static backgrounds. Our approach achieves flexible representational density adaptation through iterative voxelization and motion tendency scoring, directly addressing the fundamental mismatch between control point allocation and motion complexity. To capture temporal evolution, we introduce spline-based trajectory parameterization initialized by 2D tracklets, replacing MLP-based deformation fields to achieve smoother motion representation and more stable optimization. Extensive experiments demonstrate significant improvements in reconstruction quality and efficiency over existing state-of-the-art methods.

## 1 INTRODUCTION

Dynamic 3D reconstruction from monocular videos is critical for virtual reality, autonomous systems, and content creation. The task requires capturing complex object motions and deformations from limited viewpoints while maintaining real-time rendering performance. This remains challenging due to the fundamental ambiguity of inferring 3D motion from 2D observations and the computational demands of modeling temporally varying scenes.

Recent advances in 3D Gaussian Splatting Kerbl et al. (2023) have enabled efficient static scene reconstruction through explicit primitive representations and fast rasterization. Extensions to dynamic scenes follow two approaches: dense methods that parameterize each Gaussianâs temporal evolution, achieving high quality but poor scalability, and sparse control methods that use a small set of control points to govern scene deformation. Sparse approaches like SC-GS Huang et al. (2023) and 4D-Scaffold Cho et al. (2025) offer significant computational savings by reducing the optimization space from hundreds of thousands of Gaussians to thousands of control points.

However, existing sparse methods suffer from a fundamental limitation: they allocate control points based purely on geometric considerations. Methods typically use Farthest Point Sampling Huang et al. (2023); Diwen Wan (2024); Chen et al. (2025) or voxel centers Cho et al. (2025); Kong et al. (2025) to ensure uniform spatial coverage, but this geometric uniformity does not align with motion complexity. Real scenes exhibit highly non-uniform motion where static backgrounds dominate spatial extent while dynamic objects occupy smaller regions but require detailed motion modeling. This mismatch leads to static redundancy yet dynamic insufficiency, where control points are wasted on static regions while dynamic areas remain under-represented.

We address this through motion-adaptive control point allocation guided by vision foundation models. Our approach is built on the insight that semantic understanding can predict motion patterns: certain object categories exhibit predictable motion behaviors that can be learned from large-scale video datasets. We leverage pre-trained vision foundation models to extract semantic tokens from image patches and establish patch-token-node correspondence, enabling direct transfer of 2D semantic priors to 3D control point placement.

Our method operates in three stages. First, we generate candidate nodes by back-projecting image patches into 3D space using estimated depth and camera poses, with each node retaining its semantic token as a descriptor. Second, we apply motion-adaptive compression that iteratively merges nodes based on semantic similarity and motion tendency scores derived from vision foundation models. This compression concentrates control points in dynamic regions while reducing redundancy in static areas, directly addressing the static-dynamic resource allocation mismatch. Third, we parameterize node trajectories using cubic splines rather than MLPs, initialized from 2D tracklets to provide stable motion guidance during optimization. This spline formulation offers several advantages. It ensures temporal smoothness, reduces optimization complexity by decoupling trajectory learning from other parameters, and provides a compact representation that scales better than dense deformation fields.

In summary, our main contributions are:

â¢ We propose a motion-adaptive node initialization method using semantic and motion priors from vision foundation models to align control density with motion complexity.

â¢ We introduce a spline-based parameterization of node trajectories, which provides a compact, smooth, and differentiable motion basis for the entire dynamic scene.

â¢ We present a complete optimization framework demonstrating superior reconstruction quality and efficiency over existing methods.

## 2 RELATED WORK

Dynamic NeRF. Neural Radiance Fields (NeRF) Mildenhall et al. (2020) pioneered static view synthesis via implicit volumetric MLPs. Subsequent works Guo et al. (2023); Gafni et al. (2021); Park et al. (2021a;b); Pumarola et al. (2021); Fang et al. (2022); Wang et al. (2023) extended NeRF to dynamic scenes with temporal structures such as deformation fields and canonical mappings, but remain inefficient due to dense ray sampling and costly volume rendering. To improve efficiency, recent methods introduce grid-based representations Liu et al. (2022) and multi-view supervision Lin et al. (2022; 2023), while explicit representations such as multi-plane Chen et al. (2022); Fridovich-Keil et al. (2023); Shao et al. (2023) and grid-plane hybrids Song et al. (2023) further accelerate training. Nonetheless, their rendering speed is still insufficient for real-time applications.

Dynamic Gaussian Splatting. 3D Gaussian Splatting (3DGS) Kerbl et al. (2023) enables realtime rendering with explicit point-based representations and shows potential for broader 3D tasks Li et al. (2024); Qu et al. (2024); Cai et al. (2019; 2020); Yuan et al. (2025c;a;b; 2024a;b; 2025d). Recent works have extended 3DGS to dynamic scenes by learning time-varying Gaussian transformations. Several approaches Yang et al. (2024); Li et al. (2025) adopt per-Gaussian deformation fields, but such designs often incur redundant computation and slow training. Later methods adopt compact structural representations, such as plane encodings or hash-based schemes Wu et al. (2024); Xu et al. (2024), to improve deformation efficiency. Alternatively, sparse control points have been introduced Huang et al. (2023); Diwen Wan (2024); Kong et al. (2025); Lei et al. (2025); Chen et al. (2025); Liang et al. (2025) as a lightweight mechanism to govern Gaussian motion via interpolation, supporting both high-quality rendering and motion editing. Existing approaches differ in how control points are initialized: SC-GS, SP-GS, and HAIF-GS Huang et al. (2023); Diwen Wan (2024); Chen et al. (2025) adopt FPS sampling to ensure uniform spatial coverage, while 4D-Scaffold and EDGS Cho et al. (2025); Kong et al. (2025) use voxelization, which proves suboptimal in real-world scenes dominated by static backgrounds. More recent methods, such as MoSca and HiMoR Lei et al. (2025); Liang et al. (2025), leverage 2D tracklets from vision foundation models, but they remain sensitive to tracking errors and struggle with large topological variations. Despite these advances, sparse control methods still fail to adapt control density to motion complexity, often resulting in static redundancy and dynamic insufficiency. To address this, we propose a motion-adaptive 3DGS framework that reallocates control points according to motion cues and further stabilizes trajectory learning through spline parameterization.

<!-- image-->  
Figure 1. The overview of our method. (A) Given a monocular video, we extract semantic and motion priors from pre-trained vision foundation models. (B) These priors guide motion-adaptive node initialization, yielding compact distributions aligned with dynamic regions. (C) The initialized nodes are assigned splineparameterized trajectories to provide a motion basis. (D) Node motions are propagated to Gaussians through deformation, transforming the canonical representation. (E) The deformed model is rendered and optimized for consistent reconstruction.

## 3 PRELIMINARY: 3D GAUSSIAN SPLATTING

3D Gaussian Splatting (3DGS) Kerbl et al. (2023) models a static scene as anisotropic 3D Gaussians, each parameterized by center $\mu \in \mathbb { R } ^ { 3 }$ , covariance $ { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ , opacity $\alpha \in ( 0 , 1 )$ , and spherical harmonics (SH) coefficients $\mathbf { c } \in \mathbb { R } ^ { 3 ( l + 1 ) ^ { 2 } }$ for view-dependent color, denoted as $G ( \mu , { \pmb \Sigma } , \alpha , { \bf c } )$

Each Gaussian is projected to the image plane through the camera projection, forming a 2D Gaussian that contributes to pixel colors. The 2D Gaussians are sorted by depth and rendered via an Î±- blending scheme. The color at pixel $p$ is obtained by compositing the contributions of N ordered Gaussians overlapping the pixel:

$$
C ( \boldsymbol { p } ) = \sum _ { i \in { \cal N } } { \bf c } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{1}
$$

where $\mathbf { c } _ { i }$ is the color of the i-th Gaussian and $\alpha _ { i }$ is its image-space density determined by the projected covariance. The parameters are optimized with a photometric reconstruction loss, and adaptive density control dynamically prunes or spawns Gaussians to improve efficiency and fidelity.

Extending 3DGS to dynamic scenes is commonly formalized by endowing the representation with explicit temporal parameterization instead of a purely canonical configuration. Following prior work Liang et al. (2025); Wang et al. (2024), we introduce a temporal transformation that maps each Gaussian from the canonical space to its state at frame $t ,$ written as ${ \bf T } _ { t } ~ = ~ [ { \bf R } _ { t }$ $\mathbf { t } _ { t } \big ] \mathbf { \bar { \mu } } \in \mathrm { S E } ( 3 )$ . Applying $\mathbf { T } _ { t }$ to a canonical Gaussian $G ( \mu _ { 0 } , { \pmb \Sigma } _ { 0 } , \alpha , { \pmb \mathrm c } )$ yields its time-varying form $\bar { G _ { t } } = { G } ( \bar { \bf T } _ { t } \mu _ { 0 } , \bar { \bf R } _ { t } \bar { \pmb { \Sigma } } _ { 0 } , \bar { \alpha } , { \bf c } )$ , which provides a compact parameterization of dynamic scenes.

## 4 METHOD

## 4.1 OVERVIEW

Given a monocular image sequence $\left\{ I _ { t } \right\}$ , our goal is to reconstruct a dynamic 3DGS representation that enables temporally consistent, photorealistic novel-view renderings. The central challenge lies in the spatially non-uniform motion complexity and the need for smooth, stable trajectories under sparse supervision. To address this, we adopt a sparse node-based deformation representation that controls canonical Gaussians (Sec. 4.2) through motion-adaptive allocation. we first initialize nodes from image patches and leverage semantic and motion cues from vision foundation models to compress redundant nodes in static regions while preserving those in dynamic regions (Sec. 4.3). We then parameterize node trajectories with a spline to provide a compact, smooth, and differentiable motion basis, initialized from 2D tracklets for stable early-stage optimization (Sec. 4.4). Finally, we propagate node transforms to Gaussians through dual quaternion blending and jointly optimize geometry, appearance, and motion with multi-view photometric and motion-consistency constraints (Sec. 4.5). Figure 1 summarizes our pipeline, which integrates motion-adaptive compression with iterative voxelization to flexibly adapt representational density according to motion complexity.

## 4.2 NODE-BASED DEFORMATION REPRESENTATION

Modeling deformations in dynamic Gaussian scenes requires balancing expressiveness with tractability. Direct per-primitive formulations are prohibitively high-dimensional, while real-world motion often exhibits low-rank structure dominated by rigid and smooth patterns. This motivates a compact node-based representation, where each node carries an SE(3) trajectory and an RBF kernel defining its spatial influence. Gaussians inherit motion from their K nearest nodes through weighted aggregation, forming an efficient basis for our subsequent initialization and trajectory modeling.

Node Representation. We introduce a sparse set of nodes $\mathcal { N } = \{ \mathcal { N } _ { i } \} _ { i = 1 } ^ { N _ { n } }$ to capture the dominant smooth motion patterns of the scene, where the number of nodes $N _ { n }$ is significantly smaller than the number of Gaussian primitives $N _ { g } .$ Each node is formally defined as

$$
\mathcal { N } _ { i } = \{ \mathbf { T } _ { i } ( t ) , \rho _ { i } \} ,\tag{2}
$$

where $\mathbf { T } _ { i } ( t ) \in \mathrm { S E } ( 3 )$ denotes the trajectory of ${ \mathcal { N } } _ { i }$ across time, and $\rho _ { i } \in \mathbb { R } ^ { + }$ specifies the radius of its radial basis function (RBF), which determines the spatial extent of its influence. Thus, $\mathbf { T } _ { i } ( t )$ governs rigid motion over time, while $\rho _ { i }$ determines the spatial scope of influence. This node formulation further supports motion-adaptive initialization, allowing dynamic regions to be modeled with higher fidelity (Sec. 4.3). To ensure smooth and compact temporal modeling, each trajectory is parameterized by splines (Sec. 4.4).

Gaussian-to-Node Binding and Deformation. We derive the rigid transformation of each Gaussian primitive $\mathcal { G } _ { j }$ at any query time t by leveraging the trajectories of its neighboring nodes. Given the node set $\mathcal { N } = \{ \mathcal { N } _ { i } \} _ { i = 1 } ^ { N _ { n } }$ , each Gaussian $\mathcal { G } _ { j }$ is associated with a neighborhood of K nodes, denoted $\gamma ( G _ { j } ) \subset \mathcal N$ . The binding weight of node ${ \mathcal { N } } _ { i }$ to Gaussian $\mathcal { G } _ { j }$ is defined as

$$
w _ { i j } = \frac { \exp \left( - \frac { \| \mathbf { x } _ { j } - \mathbf { c } _ { i } \| ^ { 2 } } { 2 \rho _ { i } ^ { 2 } } \right) } { \sum _ { k \in \mathcal { V } ( G _ { j } ) } \exp \left( - \frac { \| \mathbf { x } _ { j } - \mathbf { c } _ { k } \| ^ { 2 } } { 2 \rho _ { k } ^ { 2 } } \right) } ,\tag{3}
$$

where $\mathbf { x } _ { j }$ is the canonical center of Gaussian $\mathcal { G } _ { j } , \mathbf { c } _ { i }$ is the canonical center of node $\mathcal { N } _ { i }$ . These normalized weights act as interpolation coefficients in the blending stage.

To propagate node motion to Gaussians, we construct a dense deformation field that interpolates per-Gaussian rigid motions from sparse node trajectories. Following prior work Lei et al. (2025), we instantiate this field with Dual Quaternion Blending (DQB) Kavan et al. (2007), which provides better interpolation quality. Concretely, for a node $\bar { \mathcal { N } } _ { i }$ , its SE(3) transform at time t is written as $\mathbf { T } _ { i } ( t ) = [ \dot { \mathbf { R } _ { i } } ( t ) \vert \mathbf { t } _ { i } ( \dot { t } ) ]$ . Its dual quaternion representation $\mathbf { Q } _ { i } ( t ) \dot { \in } \mathbb { D } \mathbb { Q }$ is constructed as

$$
\begin{array} { r } { \mathbf { Q } _ { i } ( t ) = q _ { r , i } ( t ) + \epsilon q _ { d , i } ( t ) , \quad q _ { d , i } ( t ) = \frac { 1 } { 2 } p _ { i } ( t ) q _ { r , i } ( t ) , } \end{array}\tag{4}
$$

where $q _ { r , i } ( t )$ is the unit quaternion corresponding to ${ \bf R } _ { i } ( t ) , p _ { i } ( t )$ is the pure quaternion of the translation vector $\mathbf { t } _ { i } ( t )$ , and Ïµ is the dual unit with $\epsilon ^ { 2 } = 0$

The blended transformation for Gaussian $\mathcal { G } _ { j }$ is obtained by normalizing the weighted sum of neighboring nodesâ dual quaternions and mapping the result back to $\operatorname { S E } ( 3 )$ :

$$
\hat { \mathbf { Q } } _ { j } ( t ) = \frac { \sum _ { i \in \mathcal { V } ( G _ { j } ) } w _ { i j } \mathbf { Q } _ { i } ( t ) } { \left\| \sum _ { i \in \mathcal { V } ( G _ { j } ) } w _ { i j } \mathbf { Q } _ { i } ( t ) \right\| } , \quad \mathbf { T } _ { j } ( t ) = \mathrm { D Q 2 S E 3 } \Big ( \hat { \mathbf { Q } } _ { j } ( t ) \Big ) .\tag{5}
$$

Here normalization guarantees that $\hat { \mathbf { Q } } _ { j } ( t )$ remains a unit dual quaternion, while DQ2SE3(Â·) denotes the standard conversion from a unit dual quaternion to a rigid transform. This formulation enables Gaussian motion to be obtained through weighted blending of neighboring node trajectories, ensuring physical consistency and temporal smoothness.

## 4.3 MOTION-ADAPTIVE NODE INITIALIZATION

Building upon the node representation in Sec. 4.2, we now address how to initialize nodes in a way that adapts to motion complexity. Uniform sampling tends to oversample static backgrounds while failing to capture sufficient detail in dynamic regions, resulting in biased motion modeling. To overcome this imbalance, we introduce a semantic-guided, motion-adaptive initialization that allocates more nodes to dynamic areas while reducing redundancy elsewhere. Given calibrated keyframes with depth and semantics, this procedure generates a compact node set in canonical space that serves as the starting point for subsequent deformation modeling.

Patch-to-Node Generation. To better integrate semantic cues with geometry, we generate candidate nodes directly from image patches rather than uniformly sampling point clouds or voxelizing 3D space. Specifically, we select a set of keyframes $\{ I _ { t } \} _ { t = 1 } ^ { T }$ and divide each image into fixed-size patches $\{ p \}$ . A frozen vision foundation model provides a token embedding $z _ { t , p }$ for each patch p at frame t, along with estimated depth maps. Each patch center $\mathbf { u } _ { t , p }$ is back-projected into 3D space to obtain its coordinate $\mathbf { x } _ { t , p } .$ The resulting collection $\left\{ ( \mathbf { x } _ { t , p } , z _ { t , p } ) \right\}$ forms the initial candidate node set, where each node is anchored at the patch center and retains the semantic token as its descriptor. This preserves a patchâtokenânode correspondence that can be exploited during subsequent compression.

Dynamic Motion-Adaptive Node Compression. The candidate node set is still excessively large for direct modeling, necessitating a principled compression strategy. A naive voxelization with fixed resolution is insufficiently adaptive across regions and often mixes features of distinct objects. We therefore propose an iterative motion-adaptive compression that iteratively merges nodes while preserving fidelity in dynamic areas. Starting from a small initial voxel size $v _ { \mathrm { i n i t } } .$ , the voxel resolution is progressively enlarged during compression. In each iteration, bipartite soft matching Huang et al. (2025) is applied within every voxel. For each node in A, we connect it to the most similar node in B, and the top r% pairs with the highest similarity are merged by retaining one representative node. After completing all voxels in the current iteration, the voxel size is enlarged by a fixed step $\Delta v ,$ , and the process is repeated until the node count falls below a target threshold.

To ensure that merging respects both appearance and geometry, we define a joint similarity between nodes ${ \mathcal { N } } _ { i } \in A$ and $\bar { \mathcal { N } _ { j } } \in \bar { B }$ as

$$
\sin ( \mathcal { N } _ { i } , \mathcal { N } _ { j } ) = \cos ( z _ { i } , z _ { j } ) - \eta \cdot \tilde { M } _ { \mathrm { f g } } ( \mathcal { N } _ { i } , \mathcal { N } _ { j } ) ,\tag{6}
$$

where $\cos ( z _ { i } , z _ { j } )$ measures the token-based appearance similarity, and $\tilde { M } _ { \mathrm { f g } } ( N _ { i } , N _ { j } ) \in [ 0 , 1 ]$ denotes a foreground prior predicted by a frozen VFM. Tokens from VFMs encode both semantic context and local appearance. Static regions yield consistent tokens across views, whereas motion causes variations that lower their similarity. Thus, cosine similarity serves as an effective cue to distinguish dynamic from static areas. The mask prior provides coarse localization of dynamic areas, discouraging premature merging in regions with high dynamic likelihood.

However, simply applying a uniform compression ratio across all voxels fails to leverage this motion-aware similarity information effectively. Such uniform treatment leads to an unfavorable trade-off: a high ratio prematurely merges dynamic nodes during early fine-voxel stages, while a low ratio fails to sufficiently reduce redundancy in static regions. To address this limitation, we propose an adaptive compression strategy that adjusts the compression ratio according to the motion tendency of each voxel cluster. Concretely, we define a dynamic tendency score $p _ { \mathrm { d y n } } ( C )$ for a cluster C by combining the mean foreground prior with the pairwise similarity within the cluster:

$$
p _ { \mathtt { d y n } } ( C ) = \sigma \left( \alpha \cdot \frac { 1 } { | \mathscr { U } _ { C } | } \sum _ { \mathcal { N } _ { k } \in \mathscr { U } _ { C } } m ( \mathcal { N } _ { k } ) - \beta \cdot \frac { 1 } { | \mathscr { M } _ { C } | } \sum _ { ( \mathcal { N } _ { i } , \mathcal { N } _ { j } ) \in \mathscr { M } _ { C } } \mathrm { s i m } ( \mathcal { N } _ { i } , \mathcal { N } _ { j } ) \right) ,\tag{7}
$$

where $\mathcal { U } _ { C }$ denotes the set of nodes in cluster C, and $\mathcal { M } _ { C }$ the set of their matched pairs. This score is then used to modulate the compression ratio of each cluster:

$$
r \% ( C ) = r _ { \operatorname* { m i n } } + ( 1 - p _ { \mathrm { d y n } } ( C ) ) \cdot ( r _ { \operatorname* { m a x } } - r _ { \operatorname* { m i n } } ) ,\tag{8}
$$

so that static voxels with low $p _ { \mathrm { d y n } }$ are merged aggressively with a high $r \%$ , while dynamic voxels with high $p _ { \mathrm { d y n } }$ are preserved with a low $r \%$

In this way, compression reduces redundancy in static regions while maintaining sufficient node density in dynamic areas, striking a balance between efficiency and temporal modeling fidelity.

## 4.4 SPLINE-PARAMETERIZED NODE TRAJECTORIES

Given the motion-adaptive node set in the canonical space, the next challenge is to represent their temporal evolution. Directly optimizing node positions at every frame is unstable and computationally expensive, as it lacks temporal regularization and entangles motion learning with Gaussian attribute updates. To achieve sparse yet stable control, we parameterize each node trajectory with a small set of keyframes connected by cubic splines. This spline-based formulation enforces smooth and differentiable trajectories, alleviates early-stage optimization difficulty, and provides reliable motion guidance for the associated Gaussian primitives.

Spline-Based Formulation. To obtain the motion of each Node at arbitrary time steps, we represent its trajectory with a cubic Hermite spline Park et al. (2025); Ahlberg et al. (2016); Goodfellow et al. (2016). Concretely, we select a set of keyframes $\{ t _ { k } \} _ { k = 1 } ^ { K }$ along the timeline and assign learnable positions $\{ P _ { k } \} _ { k = 1 } ^ { K }$ 1 to the Node at these frames. The trajectory Î¾(t) between two neighboring keyframes $( t _ { k } , t _ { k + 1 } )$ is then interpolated as

$$
\xi ( t ) = h _ { 0 0 } ( \tau ) P _ { k } + h _ { 1 0 } ( \tau ) \left( t _ { k + 1 } - t _ { k } \right) \dot { P } _ { k } + h _ { 0 1 } ( \tau ) P _ { k + 1 } + h _ { 1 1 } ( \tau ) \left( t _ { k + 1 } - t _ { k } \right) \dot { P } _ { k + 1 , \xi ( t _ { k + 1 } ) } ,\tag{9}
$$

where $\begin{array} { r } { \tau = \frac { t - t _ { k } } { t _ { k + 1 } - t _ { k } } } \end{array}$ , and the Hermite basis functions are

$$
\begin{array} { r l } & { h _ { 0 0 } ( \tau ) = 2 \tau ^ { 3 } - 3 \tau ^ { 2 } + 1 , \quad h _ { 1 0 } ( \tau ) = \tau ^ { 3 } - 2 \tau ^ { 2 } + \tau , } \\ & { h _ { 0 1 } ( \tau ) = - 2 \tau ^ { 3 } + 3 \tau ^ { 2 } , \quad h _ { 1 1 } ( \tau ) = \tau ^ { 3 } - \tau ^ { 2 } . } \end{array}\tag{10}
$$

This spline-based construction ensures temporal continuity by keeping both positions and first-order derivatives consistent across time. More importantly, it provides a compact and differentiable representation that avoids the instability and heavy joint optimization associated with MLP-based deformation fields, thereby offering stable guidance for the Gaussian primitives bound to these nodes.

Trajectory Initialization. To provide stable guidance at the early stage, we initialize the splineparameterized node trajectories from geometry-consistency, instead of using random parameters. Concretely, we extract long-term 2D tracklets Doersch et al. (2023) from a sequence of frames, and unproject them into world coordinates using estimated depth Piccinelli et al. (2024) and camera poses. Formally, given a pixel coordinate $u _ { t }$ on the 2D track at time t with depth $D _ { t } ( u _ { t } )$ , its worldspace position is computed as

$$
x _ { t } = \mathbf { R } _ { t } ^ { \top } \pi _ { \mathbf { K } } ^ { - 1 } \big ( u _ { t } , D _ { t } ( u _ { t } ) \big ) - \mathbf { R } _ { t } ^ { \top } \mathbf { T } _ { t } ,\tag{11}
$$

where $\pi _ { \mathbf { K } } ^ { - 1 } ( \cdot )$ denotes the back-projection from image to camera space with intrinsic K, and $( \mathbf { R } _ { t } , \mathbf { T } _ { t } )$ are the estimated extrinsics. We then initialize the translational spline by fitting a Hermite trajectory Î¾(t), over keyframes $\{ t _ { k } \} _ { k = 1 } ^ { K }$ , to the 3D tracklets $\{ x _ { t } \}$ via least-squares optimization:

$$
\operatorname* { m i n } _ { \{ P _ { k } \} _ { k = 1 } ^ { K } } \ \sum _ { t = 0 } ^ { N _ { f } - 1 } \left\| x _ { t } - \xi ( t ) \right\| _ { 2 } ^ { 2 } ,\tag{12}
$$

where $\{ P _ { k } \} \subset { \mathbb { R } } ^ { 3 }$ denote the learnable node positions at the keyframes, and $\xi ( t )$ between $\left( { t _ { k } , t _ { k + 1 } } \right)$ follows the cubic Hermite basis described previously. For the rotational component, we initialize $\mathbf { R } ^ { \mathrm { n o d e } } ( t ) = \mathbf { I } _ { 3 }$ for all t, and defer its refinement to the joint optimization stage.

Table 1. Quantitative comparison on Hyper-NeRF(vrig) dataset per-scene. We highlight the second best and the third best results in each scene.
<table><tr><td rowspan="2">Method</td><td rowspan="2">PSNRâ</td><td rowspan="2">Broom SSIMâ</td><td rowspan="2"></td><td colspan="3">3D-Printer</td><td rowspan="2">PSNRâ</td><td rowspan="2">Chicken SSIMâ</td><td rowspan="2">LPIPSâ</td><td rowspan="2">PSNRâ</td><td rowspan="2"></td><td rowspan="2">Banana SSIMâ LPIPSâ</td><td rowspan="2">PSNRâ</td><td rowspan="2">Mean SSIMâ</td><td rowspan="2">LPIPSâ</td></tr><tr><td>LPIPSâ PSNRâ</td><td>SSIMâ</td></tr><tr><td>HyperNeRF Park et al. (2021b)</td><td>19.51</td><td>0.210</td><td>-</td><td>20.04</td><td>0.635</td><td>LPIPSâ -</td><td>27.46</td><td>0.828</td><td>-</td><td>22.15</td><td>0.719</td><td>-</td><td>22.29</td><td>0.598</td><td>-</td></tr><tr><td>TiNeuVox Fang et al. (2022)</td><td>21.28</td><td>0.307</td><td>-</td><td>22.80</td><td>0.725</td><td>-</td><td>28.22</td><td>0.785</td><td>-</td><td>24.50</td><td>0.646</td><td></td><td>24.20</td><td>0.616</td><td>-</td></tr><tr><td>D-3DGS Yang et al. (2024)</td><td>19.99</td><td>0.269</td><td>0.700</td><td>20.71</td><td>0.656</td><td>0.277</td><td>22.77</td><td>0.640</td><td>0.363</td><td>25.95</td><td>0.853</td><td>0.155</td><td>22.36</td><td>0.605</td><td>0.374</td></tr><tr><td>4DGS Wu et al. (2024)</td><td>22.01</td><td>0.366</td><td>0.557</td><td>21.98</td><td>0.705</td><td>0.327</td><td>28.49</td><td>0.806</td><td>0.297</td><td>27.73</td><td>0.847</td><td>0.204</td><td>25.05</td><td>0.681</td><td>0.346</td></tr><tr><td>ED3DGS Bae et al. (2024)</td><td>21.84</td><td>0.371</td><td>0.531</td><td>22.34</td><td>0.715</td><td>0.294</td><td>28.75</td><td>0.836</td><td>0.185</td><td>28.80</td><td>0.867</td><td>0.178</td><td>25.43</td><td>0.697</td><td>0.297</td></tr><tr><td>MoDec-GS Kwak et al. (2025)</td><td>21.04</td><td>0.303</td><td>0.666</td><td>22.00</td><td>0.706</td><td>0.265</td><td>28.77</td><td>0.834</td><td>0.197</td><td>28.25</td><td>0.873</td><td>0.173</td><td>25.02</td><td>0.679</td><td>0.325</td></tr><tr><td>Grid4D Xu et al. (2024)</td><td>21.78</td><td>0.414</td><td>0.423</td><td>22.36</td><td>0.723</td><td>0.245</td><td>29.27</td><td>0.848</td><td>0.199</td><td>28.44</td><td>0.875</td><td>0.176</td><td>25.46</td><td>0.715</td><td>0.261</td></tr><tr><td>SC-GS Huang et al. (2023)</td><td>18.66</td><td>0.269</td><td>0.505</td><td>18.79</td><td>0.613</td><td>0.269</td><td>21.85</td><td>0.616</td><td>0.257</td><td>25.49</td><td>0.806</td><td>0.215</td><td>21.20</td><td>0.576</td><td>0.312</td></tr><tr><td>SC-GS+MANI</td><td>19.93</td><td>0.284</td><td>0.491</td><td>20.61</td><td>0.653</td><td>0.255</td><td>23.20</td><td>0.684</td><td>0.230</td><td>26.88</td><td>0.823</td><td>0.207</td><td>22.66</td><td>0.611</td><td>0.296</td></tr><tr><td>Ours</td><td>22.37</td><td>0.421</td><td>0.405</td><td>22.53</td><td>0.729</td><td>0.232</td><td>29.66</td><td>0.863</td><td>0.161</td><td>28.55</td><td>0.879</td><td>0.168</td><td>25.78</td><td>0.723</td><td>0.242</td></tr></table>

Table 2. Quantitative comparison on N3DV dataset per-scene. We highlight the best , second best and the third best results in each scene.
<table><tr><td rowspan="2">Method</td><td colspan="2">Coffee Martini</td><td colspan="2">Cook Spinach</td><td colspan="2">Cut Beef</td><td colspan="2">Flame Salmon</td><td colspan="2">Flame Steak</td><td colspan="2">Sear Steak</td><td colspan="2">Mean</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td><td>PSNRâ</td><td>SSIMâ</td></tr><tr><td>HexPlane Cao &amp; Johnson (2023)</td><td>13.26</td><td>0.405</td><td>16.95</td><td>0.729</td><td>16.76</td><td>0.538</td><td>11.16</td><td>0.342</td><td>16.97</td><td>0.753</td><td>16.89</td><td>0.589</td><td>15.33</td><td>0.559</td></tr><tr><td>D-3DGS Yang et al. (2024)</td><td>19.23</td><td>0.701</td><td>17.20</td><td>0.720</td><td>22.20</td><td>0.780</td><td>18.48</td><td>0.704</td><td>16.62</td><td>0.752</td><td>23.56</td><td>0.810</td><td>19.55</td><td>0.745</td></tr><tr><td>4DGS Wu et al. (2024)</td><td>20.95</td><td>0.761</td><td>22.64</td><td>0.779</td><td>23.18</td><td>0.793</td><td>20.64</td><td>0.758</td><td>21.83</td><td>0.787</td><td>23.38</td><td>0.829</td><td>22.10</td><td>0.785</td></tr><tr><td>SC-GS Huang et al. (2023)</td><td>19.02</td><td>0.712</td><td>16.70</td><td>0.737</td><td>20.69</td><td>0.741</td><td>17.65</td><td>0.683</td><td>17.31</td><td>0.753</td><td>21.23</td><td>0.787</td><td>18.77</td><td>0.736</td></tr><tr><td>MoDGS Qingming et al. (2025)</td><td>21.37</td><td>0.796</td><td>22.40</td><td>0.782</td><td>23.89</td><td>0.822</td><td>21.33</td><td>0.804</td><td>23.23</td><td>0.808</td><td>23.53</td><td>0.812</td><td>22.63</td><td>0.804</td></tr><tr><td>Grid4D Xu et al. (2024)</td><td>21.32</td><td>0.791</td><td>22.58</td><td>0.788</td><td>23.51</td><td>0.827</td><td>21.04</td><td>0.800</td><td>23.45</td><td>0.815</td><td>23.14</td><td>0.806</td><td>22.51</td><td>0.805</td></tr><tr><td>Ours</td><td>22.53</td><td>0.824</td><td>22.97</td><td>0.795</td><td>24.36</td><td>0.836</td><td>21.97</td><td>0.823</td><td>23.89</td><td>0.821</td><td>24.13</td><td>0.827</td><td>23.31</td><td>0.821</td></tr></table>

This geometry-driven initialization strategy grounds the spline trajectories in observed motion patterns, producing stable translational paths while preserving rotational flexibility, which facilitates more robust convergence during optimization.

## 4.5 OPTIMIZATION

To stabilize optimization under the monocular setting, we design a composite loss that integrates photometric, geometric, and motion-related constraints:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { r g b } } \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } } + \lambda _ { \mathrm { d e p t h } } \mathcal { L } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { t r a c k } } \mathcal { L } _ { \mathrm { t r a c k } } + \lambda _ { \mathrm { a r a p } } \mathcal { L } _ { \mathrm { a r a p } } .\tag{13}
$$

The photometric loss $\mathcal { L } _ { \mathrm { r g b } }$ follows the standard practice in 3DGS Kerbl et al. (2023), encouraging rendered views to be consistent with the input images. The mask loss $\mathcal { L } _ { \mathrm { m a s k } }$ employs foreground masks predicted by an off-the-shelf segmentation model Yang et al. (2023) as supervision signals. The depth loss ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ leverages relative depth maps estimated from a monocular depth prediction model Hu et al. (2025), aligned with sparse geometric priors to improve structural accuracy. For motion supervision, the tracking loss ${ \mathcal { L } } _ { \mathrm { t r a c k } }$ enforces temporal consistency by constraining the projected motion of rendered points against trajectories obtained from a pre-trained 2D tracking model Doersch et al. (2023). Finally, the ARAP loss $\mathcal { L } _ { \mathrm { a r a p } }$ Huang et al. (2024); Lei et al. (2025) regularizes control point motion by penalizing non-rigid distortions in local neighborhoods, thereby ensuring locally rigid deformations and preventing unrealistic stretching. Detailed formulations of the above loss terms are provided in Appendix.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Datasets and Metrics. We evaluate our method on two real-world datasets: Hyper-NeRF Park et al. (2021b) and Neural 3D Video (N3DV) Li et al. (2022). Hyper-NeRF dataset was captured using a handheld rig equipped with two Pixel 3 cameras. We utilize data from one camera and conduct evaluations on the held-out views captured by the other. N3DV dataset consists of 18â20 synchronized cameras per scene, recording 10â30 second sequences. To conduct monocular experiments, we follow the experimental protocol of MoDGS Qingming et al. (2025), using cam0 for training and reporting evaluations on cam5 and cam6. For quantitative evaluation, we employ three standard metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM) Wang et al. (2004), and Learned Perceptual Image Patch Similarity (LPIPS) Zhang et al. (2018).

Baselines and Implementation. We compare our method with state-of-the-art methods in dynamic scene reconstruction, including NeRF-based methods (TiNeuVox Fang et al. (2022), Hyper-

<!-- image-->  
GT

Ours  
<!-- image-->  
Gird4D

<!-- image-->

<!-- image-->

SC-GS+MANI  
<!-- image-->  
GT

<!-- image-->  
Ours

<!-- image-->

<!-- image-->  
Grid4D

<!-- image-->  
SC-GS

SC-GS+MANI

Figure 2. Qualitative comparison on the Hyper-NeRF(vrig) dataset Park et al. (2021b). Compared with other SOTA methods,our method reconstructs finer details of the moving objects.

<!-- image-->  
GT

<!-- image-->  
Ours

<!-- image-->  
Grid4D

<!-- image-->  
4DGS

<!-- image-->  
D-3DGS  
Figure 3. Qualitative comparison on the N3DV dataset Li et al. (2022).

NeRF Park et al. (2021b), HexPlanes Cao & Johnson (2023)) and 3DGS-based methods (D-3DGS Yang et al. (2024), 4DGS Wu et al. (2024), ED3DGS Bae et al. (2024),MoDec-GS Kwak et al. (2025), Grid4D Xu et al. (2024), SC-GS Huang et al. (2023), MoDGS Qingming et al. (2025)). All implementations are based on PyTorch framework and trained on a single V100 GPU with 32 GB of VRAM. For more implementation details, please refer to Appendix.

## 5.2 COMPARISONS

Results on Hyper-NeRF. As shown in Table 1, our method outperforms state-of-the-art baselines across all scenes and evaluation metrics. The qualitative results in Figure 2 further illustrate that our approach captures scene dynamics with higher fidelity, producing more complete and detailed reconstructions of moving objects. In addition, we augment SC-GS Huang et al. (2023) with our Motion-Adaptive Node Initialization (MANI), denoted as SC-GS+MANI. The last three rows of Table 1 show that SC-GS+MANI achieves clear improvements over the original SC-GS, and this advantage is also visible in Figure 2: for instance, in the Broom and Chicken scenes, SC-GS+MANI reconstructs dynamic regions more thoroughly with richer details, benefiting from the motion-aware initialization of control nodes. More results are available in Appendix.

Results on N3DV. Table 2 reports the per-scene results on the N3DV dataset. Under the monocular setting, our method achieves state-of-the-art performance with a mean PSNR of 23.31 dB. Figure 3 provides qualitative comparisons, where the highlighted red boxes show sharper and more coherent motion with fewer artifacts. For example, in fast hand motions, our method produces clearer contours and structures, while others yield blurry reconstructions. These improvements arise from placing more control points in motion-dominant areas and modeling their trajectories with spline parameterization, offering a robust alternative to implicit MLP deformation fields.

(a) Key components
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>baseline</td><td>22.35</td><td>0.613</td><td>0.335</td></tr><tr><td>+MANI</td><td>23.89</td><td>0.635</td><td>0.315</td></tr><tr><td>+MS</td><td>24.51</td><td>0.658</td><td>0.278</td></tr><tr><td>+MS (w/o Init)</td><td>24.13</td><td>0.639</td><td>0.284</td></tr><tr><td>Ours</td><td>25.78</td><td>0.722</td><td>0.242</td></tr></table>

(b) Node Init.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>FPS</td><td>24.49</td><td>0.678</td><td>0.280</td></tr><tr><td>Voxel</td><td>24.06</td><td>0.652</td><td>0.271</td></tr><tr><td>Tracklet</td><td>24.83</td><td>0.681</td><td>0.253</td></tr><tr><td>MANI (ours)</td><td>25.78</td><td>0.722</td><td>0.242</td></tr></table>

(c) Node Traj.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>MLP</td><td>23.95</td><td>0.633</td><td>0.317</td></tr><tr><td>Grid</td><td>24.28</td><td>0.649</td><td>0.271</td></tr><tr><td>Tracklet</td><td>24.59</td><td>0.671</td><td>0.263</td></tr><tr><td>MS (ours)</td><td>25.78</td><td>0.722</td><td>0.242</td></tr></table>

Table 3. Ablation studies on the Hyper-NeRF Park et al. (2021b) dataset.

<!-- image-->  
(a)ataset CD Ours (before comp.()Ours w/ $P _ { d y n } ( C ) )$ (d) Ours (MANI)  
(e) Ours (FPS)  
(f) Ours (Voxel)  
Figure 4. Visualization of different Node init. meth. on Chicken scene of Hyper-NeRF data Park et al. (2021b).

## 5.3 ABLATION STUDY

We conduct ablation studies on our method using the Hyper-NeRF Park et al. (2021b) dataset, and summarize the results in Table 3, Figure 4 and Figure 5. Our baseline follows a design similar to SC-GS Huang et al. (2023), with more details provided in Appendix.

Motion-Adaptive Node Initialization (MANI). As shown in Table 3a, introducing MANI on top of the baseline yields clear performance gains. Table 3b further compares MANI with alternative initialization strategies (FPS Huang et al. (2023), voxel-based Kong et al. (2025), tracklet-based Liang et al. (2025)), confirming the superiority of our motion-adaptive design. Figure 4 visualizes the initialization. (a) shows the raw point cloud provided by the dataset, where COLMAP Schonberger & Frahm (2016) fails to recover dynamic regions due to view inconsistency, causing static sampling to poorly cover moving areas.(b) shows our patch-to-node strategy yields better distribution, with red region indicating dynamic area in Chicken scene. (c,d) shows adding the dynamic tendency score $P _ { d y n } ( C )$ (Eq. 7) further merges static redundancy and preserves dynamic details. (e,f) shows replacing our strategy with FPS or voxel-based initialization results in inferior performance.

Spline-Parameterized Node Trajectories (MS). As shown in Table 3a, adding MS to the baseline (row 3) yields a significant performance gain, and initializing node splines with 2D tracklets from VFM models (row 4) further boosts the results. To validate its effectiveness, we replace MS with alternative deformation methods, including an MLP Yang et al. (2024), a grid-based method Wu et al. (2024), and a tracklet-based method Liang et al. (2025). Table 3c reports the quantitative results. MLP

<!-- image-->  
Figure 5. Qualitative results of ablation.

and grid-based approaches suffer from entangled optimization with large parameter spaces, leading to suboptimal performance under sparse control nodes. Tracklet-based deformation benefits from motion priors and achieves better reconstruction, but its reliance on predicted trajectories and clustering introduces noise, resulting in less stable optimization. In addition, qualitative results on the N3DV dataset (Figure 5) show that our method produces clearer and more complete reconstructions of dynamic regions.

## 6 CONCLUSION

In this work, we introduced a motion-adaptive framework for dynamic 3D Gaussian Splatting that addresses the imbalance between static redundancy and dynamic insufficiency in existing sparse control methods. By leveraging vision foundation model priors for node initialization, applying motion-aware compression to adapt representational density, and employing a spline-based trajectory formulation for stable optimization, our approach achieves substantial improvements in reconstruction quality. Extensive experiments validate its superiority over prior state-of-the-art methods, highlighting the effectiveness of aligning node allocation with motion complexity. Looking ahead, we believe this framework opens the door to incorporating stronger motion priors and handling more complex topological variations in dynamic scenes.

## REFERENCES

J Harold Ahlberg, Edwin Norman Nilson, and Joseph Leonard Walsh. The Theory of Splines and Their Applications: Mathematics in Science and Engineering: A Series of Monographs and Textbooks, Vol. 38, volume 38. Elsevier, 2016.

Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun Bang, and Youngjung Uh. Pergaussian embedding-based deformation for deformable 3d gaussian splatting. In European Conference on Computer Vision, pp. 321â335. Springer, 2024.

Yujun Cai, Liuhao Ge, Jun Liu, Jianfei Cai, Tat-Jen Cham, Junsong Yuan, and Nadia Magnenat Thalmann. Exploiting spatial-temporal relationships for 3d pose estimation via graph convolutional networks. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2272â2281, 2019.

Yujun Cai, Lin Huang, Yiwei Wang, Tat-Jen Cham, Jianfei Cai, Junsong Yuan, Jun Liu, Xu Yang, Yiheng Zhu, Xiaohui Shen, et al. Learning progressive joint propagation for human motion prediction. In Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part VII 16, pp. 226â242. Springer, 2020.

Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 130â141, 2023.

Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European conference on computer vision, pp. 333â350. Springer, 2022.

Jianing Chen, Zehao Li, Yujun Cai, Hao Jiang, Chengxuan Qian, Juyuan Kang, Shuqin Gao, Honglong Zhao, Tianlu Mao, and Yucheng Zhang. Haif-gs: Hierarchical and induced flow-guided gaussian splatting for dynamic scene. In arXiv preprint arXiv:2506.09518, 2025.

Woong Oh Cho, In Cho, Seoha Kim, Jeongmin Bae, Youngjung Uh, and Seon Joo Kim. 4d scaffold gaussian splatting with dynamic-aware anchor growing for efficient and high-fidelity dynamic scene reconstruction, 2025. URL https://arxiv.org/abs/2411.17044.

Gang Zeng Diwen Wan, Ruijie Lu. Superpoint gaussian splatting for real-time high-fidelity dynamic scene reconstruction. In Forty-first International Conference on Machine Learning, 2024.

Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10061â10072, 2023.

Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias NieÃner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia 2022 Conference Papers, pp. 1â9, 2022.

Sara Fridovich-Keil, Giacomo Meanti, Frederik RahbÃ¦k Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12479â12488, 2023.

Guy Gafni, Justus Thies, Michael Zollhofer, and Matthias NieÃner. Dynamic neural radiance fields for monocular 4d facial avatar reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8649â8658, 2021.

Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT Press, 2016.

Xiang Guo, Jiadai Sun, Yuchao Dai, Guanying Chen, Xiaoqing Ye, Xiao Tan, Errui Ding, Yumeng Zhang, and Jingdong Wang. Forward flow for novel view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16022â16033, 2023.

Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In CVPR, 2025.

Hsiang-Wei Huang, Fu-Chen Chen, Wenhao Chai, Che-Chun Su, Lu Xia, Sanghun Jung, Cheng-Yen Yang, Jenq-Neng Hwang, Min Sun, and Cheng-Hao Kuo. Zero-shot 3d question answering via voxel-based dynamic token compression. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19424â19434, 2025.

Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. arXiv preprint arXiv:2312.14937, 2023.

Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Scgs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4220â4230, 2024.

Ladislav Kavan, Steven Collins, JiËrÂ´Ä± ZË ara, and Carol OâSullivan. Skinning with dual quaternions. In Â´ Proceedings of the 2007 symposium on Interactive 3D graphics and games, pp. 39â46, 2007.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian splat- Â¨ ting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

Hanyang Kong, Xingyi Yang, and Xinchao Wang. Efficient gaussian splatting for monocular dynamic scene rendering via sparse time-variant attribute modeling. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 4374â4382, 2025.

Sangwoon Kwak, Joonsoo Kim, Jun Young Jeong, Won-Sik Cheong, Jihyong Oh, and Munchurl Kim. Modec-gs: Global-to-local motion decomposition and temporal interval adjustment for compact dynamic 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 11338â11348, 2025.

Jiahui Lei, Yijia Weng, Adam W Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6165â6177, 2025.

Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5521â5531, 2022.

Zehao Li, Wenwei Han, Yujun Cai, Hao Jiang, Baolong Bi, Shuqin Gao, Honglong Zhao, and Zhaoqi Wang. Gradiseg: Gradient-guided gaussian segmentation with enhanced 3d boundary precision. arXiv preprint arXiv:2412.00392, 2024.

Zehao Li, Hao Jiang, Yujun Cai, Jianing Chen, Baolong Bi, Shuqin Gao, Honglong Zhao, Yiwei Wang, Tianlu Mao, and Zhaoqi Wang. Stdr: Spatio-temporal decoupling for real-time dynamic scene rendering. arXiv preprint arXiv:2505.22400, 2025.

Yiming Liang, Tianhan Xu, and Yuta Kikuchi. Himor: Monocular deformable gaussian reconstruction with hierarchical motion representation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 886â895, 2025.

Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Efficient neural radiance fields for interactive free-viewpoint video. In SIGGRAPH Asia 2022 Conference Papers, pp. 1â9, 2022.

Haotong Lin, Sida Peng, Zhen Xu, Tao Xie, Xingyi He, Hujun Bao, and Xiaowei Zhou. High-fidelity and real-time novel view synthesis for dynamic scenes. In SIGGRAPH Asia 2023 Conference Papers, pp. 1â9, 2023.

Jia-Wei Liu, Yan-Pei Cao, Weijia Mao, Wenqiao Zhang, David Junhao Zhang, Jussi Keppo, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Devrf: Fast deformable voxel radiance fields for dynamic scenes. Advances in Neural Information Processing Systems, 35:36762â36775, 2022.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. pp. 405â421, 2020.

Jongmin Park, Minh-Quan Viet Bui, Juan Luis Gonzalez Bello, Jaeho Moon, Jihyong Oh, and Munchurl Kim. Splinegs: Robust motion-adaptive spline for real-time dynamic 3d gaussians from monocular video. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26866â26875, 2025.

Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5865â5874, 2021a.

Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021b.

Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10106â10116, 2024.

Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10318â10327, 2021.

LIU Qingming, Yuan Liu, Jiepeng Wang, Xianqiang Lyu, Peng Wang, Wenping Wang, and Junhui Hou. Modgs: Dynamic gaussian splatting from casually-captured monocular videos with depth priors. In The Thirteenth International Conference on Learning Representations, 2025.

Haoxuan Qu, Zhuoling Li, Hossein Rahmani, Yujun Cai, and Jun Liu. Disc-gs: Discontinuity-aware gaussian splatting. Advances in Neural Information Processing Systems, 37:112284â112309, 2024.

Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4104â4113, 2016.

Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu, Hongwen Zhang, and Yebin Liu. Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16632â16642, 2023.

Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE Transactions on Visualization and Computer Graphics, 29(5):2732â2742, 2023.

Chaoyang Wang, Lachlan Ewen MacDonald, Laszlo A Jeni, and Simon Lucey. Flow supervision for deformable nerf. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21128â21137, 2023.

Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764, 2024.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â 612, 2004.

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20310â20320, 2024.

Jiawei Xu, Zexin Fan, Jian Yang, and Jin Xie. Grid4d: 4d decomposed hash encoding for highfidelity dynamic gaussian splatting. arXiv preprint arXiv:2410.20815, 2024.

Jinyu Yang, Mingqi Gao, Zhe Li, Shang Gao, Fangjing Wang, and Feng Zheng. Track anything: Segment anything meets videos. arXiv preprint arXiv:2304.11968, 2023.

Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20331â20341, 2024.

Zhenlong Yuan, Jiakai Cao, Zhaoxin Li, Hao Jiang, and Zhaoqi Wang. SD-MVS: Segmentation-Driven Deformation Multi-View Stereo with Spherical Refinement and EM Optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 6871â6880, 2024a.

Zhenlong Yuan, Jiakai Cao, Zhaoqi Wang, and Zhaoxin Li. Tsar-mvs: Textureless-aware segmentation and correlative refinement guided multi-view stereo. Pattern Recognition, 154:110565, 2024b.

Zhenlong Yuan, Cong Liu, Fei Shen, Zhaoxin Li, Jinguo Luo, Tianlu Mao, and Zhaoqi Wang. MSP-MVS: Multi-granularity segmentation prior guided multi-view stereo. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 9753â9762, 2025a.

Zhenlong Yuan, Jinguo Luo, Fei Shen, Zhaoxin Li, Cong Liu, Tianlu Mao, and Zhaoqi Wang. DVP-MVS: Synergize depth-edge and visibility prior for multi-view stereo. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 9743â9752, 2025b.

Zhenlong Yuan, Zhidong Yang, Yujun Cai, Kuangxin Wu, Mufan Liu, Dapeng Zhang, Hao Jiang, Zhaoxin Li, and Zhaoqi Wang. SED-MVS: Segmentation-Driven and Edge-Aligned Deformation Multi-View Stereo with Depth Restoration and Occlusion Constraint. IEEE Transactions on Circuits and Systems for Video Technology, 2025c.

Zhenlong Yuan, Dapeng Zhang, Zehao Li, Chengxuan Qian, Jianing Chen, Yinda Chen, Kehua Chen, Tianlu Mao, Zhaoxin Li, Hao Jiang, and Zhaoqi Wang. Dvp-mvs++: Synergize depthnormal-edge and harmonized visibility prior for multi-view stereo, 2025d. URL https:// arxiv.org/abs/2506.13215.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586â595, 2018.