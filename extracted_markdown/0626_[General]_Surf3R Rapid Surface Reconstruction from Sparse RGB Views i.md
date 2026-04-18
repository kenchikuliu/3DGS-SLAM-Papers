# Surf3R: Rapid Surface Reconstruction from Sparse RGB Views in Seconds

Haodong Zhu1,4â Changbai Li1â Yangyang Ren1 Zichao Feng1 Xuhui Liu1

Hanlin Chen2â  Xiantong Zhen3 Baochang Zhang1

1Beihang University, China

2National University of Singapore, Singapore

3United Imaging, China

4Zhongguancun Academy, China

## Abstract

Current multi-view 3D reconstruction methods rely on accurate camera calibration and pose estimation, requiring complex and time-intensive pre-processing that hinders their practical deployment. To address this challenge, we introduce Surf3R, an end-to-end feedforward approach that reconstructs 3D surfaces from sparse views without estimating camera poses and completes an entire scene in under 10 seconds. Our method employs a multi-branch and multi-view decoding architecture in which multiple reference views jointly guide the reconstruction process. Through the proposed branch-wise processing, cross-view attention, and inter-branch fusion, the model effectively captures complementary geometric cues without requiring camera calibration. Moreover, we introduce a D-Normal regularizer based on an explicit 3D Gaussian representation for surface reconstruction. It couples surface normals with other geometric parameters to jointly optimize the 3D geometry, significantly improving 3D consistency and surface detail accuracy. Experimental results demonstrate that Surf3R achieves state-of-the-art performance on multiple surface reconstruction metrics on ScanNet++ and Replica datasets, exhibiting excellent generalization and efficiency.

## 1 Introduction

3D surface reconstruction is a long-standing problem that aims to create 3D surfaces of an object or scene captured from multiple viewpoints (Broadhurst, Drummond, and Cipolla 2001; Kutulakos and Seitz 1999; Seitz and Dyer 1999). This technique has wide applications in robotics, graphics, virtual reality, and other fields. Traditional 3D surface reconstruction methods typically include two main approaches: Structure-from-Motion (SfM) combined with Multi-View Stereo (MVS) (Chen, Li, and Lee 2023; Huang et al. 2024; Chen et al. 2024a) or volumetric methods (Zak Murez and Rabinovich 2020; Sun et al. 2021). The SfM+MVS pipeline involves first estimating camera poses and generating sparse 3D point clouds using SfM (Crandall et al. 2013; Charatan et al. 2024; Schonberger and Frahm 2016). This is followed Â¨ by computing per-view depth maps and fusing them into the final 3D surface using MVS techniques (Huang et al. 2024; Guedon and Lepetit 2024; Chen et al. 2024a). On the other Â´ hand, volumetric methods, such as Atlas (Zak Murez and Rabinovich 2020) and NeuralRecon (Sun et al. 2021), predict 3D volumes like Truncated Signed Distance Function (TSDF) from multiple views, often avoiding the explicit depth map computation. Although these two kinds of methods achieve high-quality surface reconstruction, they often rely on prior knowledge or require nontrivial pre-processing steps, such as SfM to estimate camera intrinsics and extrinsics. These pre-processing steps often require heavy GPU computation and are time-consuming (typically taking 1â2 hours per scene on a modern GPU), making real-time inference challenging and reducing their practical usability.

<!-- image-->  
Figure 1: Comparison Between Traditional Methods and Our Approach. Traditional methods rely on SfM for sparse point clouds and calibrated poses, followed by different 3D Reconstruction Methods (RMs). In contrast, our method directly reconstructs the scene from uncalibrated images in under 10 s, eliminating the need for calibration or iterative refinement.

To address the aforementioned limitations, inspired by DUSt3R (Wang et al. 2024), we propose Surf3R, the first feed-forward network that performs pose-free surface reconstruction from sparse RGB inputs in a single pass. Specifically, we first encode all input views using a shared encoder to extract multi-scale visual features. To effectively model cross-view information interactions, we introduce Feature-Refine (FR) blocks that jointly learn not only the pairwise relationships between a selected reference view and all other source views, but also the interactions among source views themselves. When reconstructing a large scene from sparse multi-view images, the geometric correspondence between a selected reference view and certain source views could be insufficient. This is because substantial changes in camera poses make it difficult to directly infer the relation between the reference view and those source views. To mitigate this issue, we further introduce a cross-reference fusion mechanism, implemented via a multi-branch design where multiple reference views are independently selected. Each branch processes the input views through its own FR blocks and integrates information using dedicated Cross-Reference Fusion (CRF) blocks, enabling effective propagation of long-range and complementary information across views. Based on the fused multi-view features, we first generate a sparse 3D point cloud for reconstruction. While directly converting this point cloud into a mesh using NKSR (Huang et al. 2023) is feasible, our experiments (see Sec. 4.2) show that this naive approach yields poor reconstruction quality. The underlying limitation is that point-cloud supervision is applied in a view-separated manner and thus lacks global 3D consistency. We therefore adopt a Gaussian representation: each Gaussian resides in a unified 3D space and is projected into every view during rendering, so the per-view loss implicitly regularizes the entire scene and yields smoother, more accurate surfaces. The final per-pixel Gaussian primitives are derived from specifically designed Gaussian heads.

To further facilitate accurate surface reconstruction from the predicted Gaussian parameters, we introduce a Depth-Normal Regularization strategy (Chen et al. 2024b) designed to enhance the geometric fidelity of the reconstructed surfaces. Specifically, we first apply a flattening operation to the Gaussian primitives to better align them with the local surface geometry. Subsequently, we introduce a D-Normal formulation, in which surface normals are not directly blended from 3D Gaussians, but instead derived from the gradient of the rendered depth map. As a result, it allows the Gaussian parameters to be directly supervised by surface normals, jointly optimizing the geometry and markedly enhancing 3D consistency and surface detail. Extensive experiments on the ScanNet++ datasets demonstrate that Surf3R achieves stateof-the-art surface reconstruction performance. It significantly outperforms both optimization-based and feed-forward baselines in terms of accuracy and completeness. Moreover, when evaluated on the unseen Replica dataset in a zero-shot setting, our model maintains competitive accuracy, demonstrating robust generalization to novel scenes. And it also delivers strong performance on novel view synthesis tasks. Notably, Surf3R reconstructs an entire scene in under 10 seconds, making it highly efficient for real-time or interactive applications. We summarize our contributions as follows:

â¢ We present Surf3R, the first feed-forward network for pose-free surface reconstruction from sparse multi-view RGB inputs. It achieves real-time surface reconstruction in under 10 seconds, offering both high efficiency and scalability.

â¢ We employ a multi-branch architecture where multiple reference views are jointly leveraged to capture longrange cross-view interactions. Furthermore, we introduce a Depth-Normal Regularization strategy to enhance geometric fidelity.

â¢ Extensive experiments on ScanNet++ and Replica datasets demonstrate that Surf3R achieves state-of-the-art surface reconstruction, generalizes zero-shot to new scenes, and remains competitive for novel-view synthesis.

## 2 Related Works

Multi-View Surface Reconstruction. Multi-view surface reconstruction recovers dense geometry from images captured at multiple viewpoints. Classical pipelines fuse depth maps obtained by multi-view stereo (Furukawa, Hernandez Â´ et al. 2015; Seitz et al. 2006; Schonberger et al. 2016; Zhang Â¨ et al. 2020; Yao et al. 2018; Bleyer, Rhemann, and Rother 2011) or optimize voxel occupancy fields (Bonet and Viola 1999; Kutulakos and Seitz 2000; Broadhurst, Drummond, and Cipolla 2001; Seitz and Dyer 1999), but these approaches are limited by memory and cross-view noise (Tulsiani et al. 2017; Ummenhofer et al. 2017). Implicit neural representations such as signed distance fields (Park et al. 2019; Liu et al. 2020; Ma et al. 2023; Sitzmann et al. 2020) and neural volume rendering (Yariv et al. 2021; Wang et al. 2021a) alleviate some constraints, yet methods like NeuS (Wang et al. 2021a), MonoSDF (Yu et al. 2022), and Geo-NeuS (Fu et al. 2022) remain optimization intensive and do not scale well. 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) introduces an explicit alternative that rasterizes anisotropic Gaussians in real time (Yifan et al. 2019), though the vanilla version lacks geometric supervision. Extensions such as VastGaussian (Lin et al. 2024), SuGaR (Guedon and Lepetit 2024) and HRGS Â´ (Li et al. 2025) incorporate view-consistent depth and normal constraints, substantially improving accuracy and convergence. However, these methods often require pre-processing step, limiting their practical deployment. To bridge this gap, we present a feedforward network that dispenses with costly pre-processing while reconstructs high idelity 3D susrface from sparse views in under 10 seconds.

Novel View Synthesis. Novel-view synthesis has progressed from geometry-aware volumetric grids (e.g., Soft3D (Penner and Zhang 2017), voxel colouring (Seitz and Dyer 1999)) to neural radiance fields such as NeRF (Mildenhall et al. 2021) and Mip-NeRF (Barron et al. 2021), which deliver high fidelity but are hampered by dense ray sampling. Hashencoded feature grids and sparse voxel accelerations (Instant-NGP (Muller et al. 2022), Plenoxels (Fridovich-Keil et al. Â¨ 2022), KiloNeRF (Reiser et al. 2021)) mitigate runtime yet remain limited in sparse-view or large-scale scenarios (Garbin et al. 2021; Chen et al. 2022). Recent work on 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) achieves real-time rendering by rasterising anisotropic Gaussians, achieving faster and higher-quality rendering without neural components. In this work, we utilize the advantage of Gaussian Splatting to perform surface reconstruction and incorporate normal priors to guide the reconstruction.

Learning-based 3D Reconstruction. Learning-based 3D reconstruction approaches have recently seen a lot of progress.

<!-- image-->  
Figure 2: Overall Framework of Surf3R. The reference view branches are shown in orange, while the branches of other source views are shown in blue. Each model path uses a different reference view. For clarity, only one of the stacked FRBlock and CRFBlock is displayed.

Notably, DUSt3R (Wang et al. 2024) generates two point maps for an input image pair within a shared coordinate system, implicitly incorporating both intrinsic and extrinsic camera parameters. Nevertheless, the method is intrinsically pair-wise and lacks native support for multi-view input. Multi-view operation is possible only by appending a separate global-alignment stage that registers the individual reconstructions. This alignment must be performed offline, incurs substantial computational overhead, and cannot adapt on-the-fly when additional views become available. Several recent studies (Leroy, Cabon, and Revaud 2024; Wang and Agapito 2024; Zhang et al. 2025; Wang et al. 2025) replace DUSt3Râs test-time optimization with feed-forward neural networks to accelerate inference, yet their primary focus lies in view synthesis and related tasks rather than high-fidelity surface reconstruction. In this work, we close this gap by introducing a feed-forward, geometry-aware framework that delivers state-of-the-art surface reconstruction while retaining real-time efficiency.

## 3 Methodology

Our proposed Surf3R framework enables accurate surface reconstruction from a sparse set of RGB images, without the need for known camera intrinsics or poses. First, we introduce a feedforward network that extracts multi-view features and predicts per-pixel 3D Gaussian parameters (Sec.3.1). These Gaussians are then flattened into 2D planes to better represent the surface, which helps achieve more accurate depth estimation for reconstruction (Sec.3.2). Finally, the model is optimized using geometry-aware loss functions (Sec. 3.3).

## 3.1 Feedforward Geometry Reconstruction

As shown in Fig. 2, Surf3R employs a feedforward network architecture to extract and fuse visual information across multiple input views without camera intrinsics and poses. Unlike traditional methods, which depend on estimating these parameters, Surf3R employs a single feedforward pass to directly reconstruct 3D surfaces through view-specific processing. Multi-view Feature Encoding and Fusion. Given N input images $\{ I _ { i } \} _ { i = 1 } ^ { N }$ , a shared-weight Vision Transformer (ViT)

is first employed to encode each image $I _ { i }$ into visual tokens $F _ { 0 } ^ { i } = \mathrm { V i } \hat { \mathrm { T } } ( \bar { I _ { i } } )$ , with a spatial resolution reduced by a factor of 16. To mitigate the limitation that a single reference view may not provide uniformly accurate geometric cues across the entire scene, we introduce a multi-branch architecture in which multiple reference views collaboratively participate in the reconstruction process. Specifically, we select M reference views $\{ r _ { m } \} _ { m = 1 } ^ { M ^ { - } }$ to construct M decoding branches $\lbrace b _ { m } \rbrace _ { m = 1 } ^ { M }$ , each centered around a different reference view. Within each branch $b _ { m } ,$ a dedicated multiview decoder, consisting of D stacked Feature-Refine (FR) blocks, is employed to refine features via cross-view attention. These blocks are denoted as $\mathrm { F R B l o c k } _ { d } ^ { \mathrm { r e f } }$ for the reference view $r _ { m }$ and $\mathrm { F R B l o c k } _ { d } ^ { \mathrm { s r c } }$ for the remaining $N - 1$ source views, where $d \in \{ 1 , \dots , \tilde { D } \}$ . At each decoding layer d, the token representations are updated in a view-specific manner. For a given view $I _ { v } ,$ the decoder block takes as input the primary tokens $F _ { d - 1 } ^ { v , m }$ from view $I _ { v }$ and the secondary tokens $\mathcal { F } _ { d - 1 } ^ { - v , m } = \{ F _ { d - 1 } ^ { i , m } | i \neq v \}$ from all other views. The update is performed as:

$$
f _ { d } ^ { v , m } = \left\{ \begin{array} { l l } { \mathrm { F R B l o c k } _ { d } ^ { \mathrm { r e f } } ( F _ { d - 1 } ^ { v , m } , \mathcal { F } _ { d - 1 } ^ { - v , m } ) , } & { \mathrm { i f } v = r _ { m } } \\ { \mathrm { F R B l o c k } _ { d } ^ { \mathrm { s r e } } ( F _ { d - 1 } ^ { v , m } , \mathcal { F } _ { d - 1 } ^ { - v , m } ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

To further enhance the expressiveness of feature representations, we introduce a Cross-Reference Fusion (CRF) block after each decoder block to fuse and update per-view tokens computed under different reference views. Specifically, the updated feature is computed as:

$$
\mathcal { F } _ { d } ^ { v , m } = \mathrm { C R F B l o c k } _ { d } ( f _ { d } ^ { v , m } , f _ { d } ^ { v , - m } ) ,\tag{2}
$$

where $f _ { d } ^ { v , - m } = \{ f _ { d } ^ { v , 1 } , \ldots , f _ { d } ^ { v , m - 1 } , f _ { d } ^ { v , m + 1 } , \ldots , f _ { d } ^ { v , M } \}$ denotes the set of representations for view v at layer d across all other reference branches.

Gaussian Parameterizing. Based on the fused multi-view features, we derive a sparse 3D point cloud. Direct meshing with NKSR (Huang et al. 2023) is possible, but Sec. 4.2 shows that it yields poor surface fidelity because global 3D consistency is absent. Accordingly, we adopt a unified 3D Gaussian representation whose projections into all views let per-view losses regularize the entire scene, producing smoother and more accurate surfaces. To predict the final Gaussian parameters from $F _ { D } ^ { v , m }$ , we introduce two types of heads: ${ \mathrm { H e a d } } ^ { \mathrm { r e f } }$ for the reference view and Headsrc for the remaining source views. Each head comprises two sets of regression branches. The first set includes a pointmap head and a confidence head, which respectively predict a 3D pointmap $P ^ { v , m } \in \mathbb { R } ^ { H \times W \times \mathbf { \dot { 3 } } }$ and a confidence map $C ^ { v , m } \stackrel { \bullet } { \in } \mathbb { R } ^ { H \times \dot { W } }$ for each view. The second set consists of Gaussian-specific heads that regress the per-pixel Gaussian parameters, including scaling factors $S ^ { v , m } \in \mathbb { R } ^ { H \times W \times 3 }$ , rotation quaternions $q ^ { v , m } \in \breve { \mathbb { R } } ^ { H \times W \times 4 }$ , and opacity values $\alpha ^ { v , m } \ \in \ \mathbb { R } ^ { H \times W }$ which are essential for novel view synthesis. Notably, the predicted pointmap serves as the center of the Gaussian, the input pixel color $I _ { v }$ is used for its color and fix the spherical harmonics degree to be 0. During inference, A model with M branches is used but the final per-view Gaussian predictions are computed using the heads in the first branch.

## 3.2 Planar Geometry Formulation

To facilitate surface reconstruction from the predicted Gaussian parameters, we introduce a Depth-Normal Regularization strategy aimed at enhancing the accuracy of depth representation. This strategy leverages two fundamental planar geometric properties: normal and depth from our predicted 3D Gaussian primitives.

Flattening 3D Gaussians. To enhance the capacity of Gaussians in modeling surface geometry, we first apply a flattening operation to the Gaussian primitives. Inspired by (Chen, Li, and Lee 2023), we specifically introduce a scale regularization loss $\mathcal { L } _ { s } ,$ , which minimizes the smallest of the three scaling factors $\mathbf { S } = \left( s _ { 1 } , s _ { 2 } , s _ { 3 } \right) ^ { \top } \in \mathbb { R } ^ { 3 }$ for each Gaussian:

$$
\mathcal { L } _ { \mathrm { s } } = \left. \operatorname* { m i n } \left( s _ { 1 } , s _ { 2 } , s _ { 3 } \right) \right. _ { 1 } .\tag{3}
$$

By minimizing the loss, the Gaussian is driven towards a flat shape, effectively approximating a local surface plane.

Normal Map Rendering. Once a Gaussian is flattened onto a local plane, the surface normal n is computed from the predicted rotation quaternion $q$ and scaling factors $S$ predicted by our feedforward network. We first convert $q$ into a rotation matrix $\boldsymbol { R } \in \mathbb { R } ^ { 3 \times 3 }$ . The normal is then defined as the direction corresponding to the smallest scaling factor: ${ \bf n } = R [ k , : ] \in \mathbb { R } ^ { 3 } , { \bf \dot { k } } =$ argmin $\left( \left[ s _ { 1 } , s _ { 2 } , s _ { 3 } \right] \right)$ ). The normal n is subsequently transformed into the camera coordinate system. Finally, a rendered normal map $\hat { \bf N }$ is generated by a weighted summation of individual Gaussian normals ${ \bf n } _ { i }$ and their opacities $\alpha _ { i }$ along each ray:

$$
\hat { \mathbf { N } } = \sum _ { i \in K } \mathbf { n } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) / \sum _ { i \in K } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{4}
$$

Depth Map Rendering. To achieve more precise and geometrically consistent depth values than simply using the Gaussianâs center position (Tang et al. 2024), we compute the depth as the intersection point of a viewing ray originating from the camera center with the plane represented by the flattened Gaussian. Formally, the intersection depth $\mathbf { d } ( \mathbf { n } , \mathbf { p } , \mathbf { r } )$ is calculated by:

$$
\mathbf { d } ( \mathbf { n } , \mathbf { p } , \mathbf { r } ) = \mathbf { r } _ { z } * ( \mathbf { n } \cdot \mathbf { p } ) / ( \mathbf { n } \cdot \mathbf { r } ) ,\tag{5}
$$

where $\mathbf { r } _ { z }$ is the z-value of the ray direction. This formulation reveals that the intersection depth of a Gaussian is jointly determined by its position and surface normal, thereby enabling a more geometrically grounded and accurate depth estimation. Leveraging this property, a view-consistent depth map $\hat { D }$ is rendered through a weighted summation of these intersection depths $d _ { i } ,$ weighted by their opacities $\alpha _ { i } { : }$

$$
\hat { D } = \frac { \sum _ { i \in K } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } { \sum _ { i \in K } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) } ,\tag{6}
$$

where $K$ denotes the set of Gaussians along a ray, sorted by depth. The properties form the foundation for regularization.

## 3.3 Geometry-aware Loss Functions

To train our model effectively, we employ a set of loss functions tailored to guide the learning of geometry-aware representations. We begin with a confidence-aware pointmap

<table><tr><td rowspan="2">Method</td><td colspan="3">Per-Scene</td><td rowspan="2">Feedforward</td><td colspan="3"></td></tr><tr><td>NeuS 2DGS</td><td>SuGaR</td><td>PGSR</td><td>DUSt3R Surf3R-P</td><td>Surf3R-G</td><td>Surf3R-GD (Ours)</td></tr><tr><td>Precision â</td><td>29.42</td><td>23.01 38.30</td><td>35.33</td><td>4.62</td><td>63.75</td><td>78.50</td><td>80.24</td></tr><tr><td>Recall â</td><td>22.14 16.04</td><td>34.92</td><td>21.70</td><td>4.84</td><td>62.36</td><td>75.34</td><td>77.55</td></tr><tr><td>F1-score â</td><td>25.13 18.30</td><td>36.12</td><td>24.92</td><td>4.06</td><td>62.89</td><td>76.72</td><td>78.71</td></tr><tr><td>Time</td><td colspan="3">&gt; 30 min</td><td>&gt; 1 min</td><td colspan="3"> $< l O s$ </td></tr></table>

Table 1: Quantitative comparison on ScanNet++ dataset. Bold indicates best result. Our method achieves state-of-the-art performance across all metrics. Surf3R-P: point-map heads, trained with $\mathcal { L } _ { c } \mathrm { : }$ ; Surf3R-G: + Gaussian heads, adds $\mathcal { L } _ { r } ;$ Surf3R-GD (Full model): + D-Normal regularization, adds $\mathcal { L } _ { s } , \bar { \mathcal { L } } _ { n }$ and $\mathcal { L } _ { d n }$

regression loss, denoted as ${ \mathcal { L } } _ { \mathrm { c o n f } }$ , which supervises the predicted 3D pointmaps $P ^ { k , m }$ using their associated confidence maps $Q ^ { k , \dot { m } }$ . The loss is defined as:

$$
\mathcal { L } _ { c } = \sum _ { k , m } \sum _ { p \in P ^ { k , m } } Q _ { p } ^ { k , m } \left\| P _ { p } ^ { k , m } - \mathbf { P } _ { g t , p } \right\| _ { 1 } - \beta \log Q _ { p } ^ { k , m }\tag{7}
$$

where $P _ { p } ^ { k , m }$ is the predicted 3D point, $\mathbf { P } _ { g t , p }$ is the ground truth, $Q _ { p } ^ { \dot { k } , m }$ is the confidence score, and Î² is a regularization parameter. We also employ a standard RGB rendering loss ${ \mathcal { L } } _ { \mathrm { r } }$ (Charatan et al. 2024), which supervises the rendered image against the ground-truth RGB image to preserve photometric fidelity.

To improve surface regularity, we introduce a flattening loss $\mathcal { L } _ { \mathrm { s } } ,$ , which encourages the predicted Gaussian primitives to lie on locally planar surfaces. In addition, we employ a rendered normal map loss ${ \mathcal { L } } _ { n }$ to align the rendered normal map $\hat { \bf N }$ with a reference normal map $\mathbf { N } _ { \mathrm { g t } }$ , which is computed from the ground-truth depth via finite difference-based gradients. The loss combines an $\ell _ { 1 }$ term and a cosine similarity term, and is defined as:

$$
\mathcal { L } _ { n } = | \hat { \mathbf { N } } - \mathbf { N } _ { \mathrm { g t } } | _ { 1 } + \left( 1 - \hat { \mathbf { N } } \cdot \mathbf { N } _ { \mathrm { g t } } \right) ,\tag{8}
$$

where the first term enforces per-pixel accuracy, and the second term promotes angular alignment. This supervision encourages the rendered normals to more faithfully reflect the underlying scene geometry.

While explicit normal regularization can effectively refine the orientation of 3D Gaussians, it has less impact on their positions. To address this limitation and ensure robust 3D surface reconstruction, we introduce a Depth-Normal (D-Normal) regularization strategy (Chen et al. 2024b), which enables joint optimization of both the orientation and positional accuracy of the Gaussians. The D-Normal $\overline { { \mathbf { N } } } _ { d }$ is derived from the rendered depth $\hat { D }$ by computing the crossproduct of horizontal and vertical finite differences from neighboring points:

$$
\overline { { \mathbf { N } } } _ { d } = \frac { \nabla _ { v } \mathbf { d } \times \nabla _ { h } \mathbf { d } } { | \nabla _ { v } \mathbf { d } \times \nabla _ { h } \mathbf { d } | } ,\tag{9}
$$

where d represents the 3D coordinates of a pixel obtained via back-projection from the depth map. Finally, the D-Normal regularization loss $\mathcal { L } _ { d n }$ is defined as:

$$
\mathcal { L } _ { d n } = \left( \| \bar { \mathbf { N } } _ { d } - \mathbf { N } _ { g t } \| _ { 1 } + \left( 1 - \bar { \mathbf { N } } _ { d } \cdot \mathbf { N } _ { g t } \right) \right) ,\tag{10}
$$

Overall Loss. The final total loss $\mathcal { L } _ { t o t a l }$ combines the geometric regularization losses for Gaussian primitives and the pointmap regression loss:

$$
\mathcal { L } _ { t o t a l } = \lambda _ { c } \mathcal { L } _ { c } + \lambda _ { r } \mathcal { L } _ { r } + \lambda _ { s } \mathcal { L } _ { s } + \lambda _ { n } \mathcal { L } _ { n } + \lambda _ { d n } \mathcal { L } _ { d n }\tag{11}
$$

The weighting factors $\lambda _ { c } , \lambda _ { r } , \lambda _ { s } , \lambda _ { n } ,$ , and $\lambda _ { d m }$ balance the contributions of each loss term, ensuring a holistic optimization of the reconstructed geometry and pointmaps.

## 4 Experiments

We begin by presenting the experimental setup in Sec. 4.1. We assess the effectiveness and generalization capability of our approach for surface reconstruction in Sec. 4.2. We further demonstrates the novel view synthesis capability of our method in Sec. 4.3. Additionally, we validate the effectiveness of the proposed techniques in Sec. 4.4.

## 4.1 Implementation Details

We train our model on ScanNet++ dataset (Yeshwanth et al. 2023). View sequences $\{ I _ { v } \} _ { v = 1 } ^ { N }$ are generated with an overlapâbased sampler. Starting from a random keyframe, a candidate view is appended whenever the overlap between its point cloud and the accumulated scene cloud falls within 30%â70%. For training, each scene provides 100 trajectories of 10 views, yielding diverse yet geometrically consistent inputs. For validation, we construct 1 000 trajectories with 30 views on the ScanNet++ validation split and retain the 50 widest-baseline views per scene for surface fusion to maximise spatial coverage. Additionally, we conduct zero-shot generalization experiments on Replica dataset (Straub et al. 2019) to assess the cross-dataset adaptability of our model.

We train using 32 NVIDIA H800 GPUs, processing input views at a resolution of 224 Ã 224. For each training trajectory, the first N = 8 views are used as input, from which $M = 4$ reference views are randomly selected. The model is trained for 50 epochs, resulting in a total training time of approximately 40 hours. Additional training details are provided in the Appendix.

## 4.2 Surface Reconstruction

We evaluate three progressively enhanced variants of our framework. Surf3R-P employs only the pointmap heads and is trained with the reconstruction loss $\mathcal { L } _ { c }$ . Surf3R-G extends this baseline by introducing Gaussian heads together with the associated rendering loss $\mathcal { L } _ { r }$ . Surf3R-GD, the full approach, further incorporates the D-Normal regularization strategy and is optimized with the additional terms $\mathcal { L } _ { s } , \mathcal { L } _ { n }$ and $\mathcal { L } _ { d n }$ As shown in Tab. 1, the results for per-scene methods are computed on the eight ScanNet++ validation scenes, while feed-forward models are evaluated across all 50 scenes, with the table reporting the dataset-wide averages. Surf3R-GD achieves state-of-the-art surface reconstruction performance, with an F1-score of 78.71. Compared to traditional per-scene reconstruction approaches such as NeuS (Wang et al. 2021b), 2DGS (Huang et al. 2024), SuGaR (Guedon and Lepetit Â´ 2024), and PGSR (Chen et al. 2024a), our method achieves significantly higher surface reconstruction quality. In particular, compared to the concurrent method SuGaR (Guedon Â´ and Lepetit 2024), our approach yields a substantial improvement (78.71 vs. 36.12 in F1-score). Moreover, our model exhibits exceptional efficiency, offering a reconstruction speed that is approximately 180Ã faster than per-scene methods. We further compare our method with feedforward-based approach DUSt3R (Wang et al. 2024). DUSt3R and Surf3R-P reconstruct surfaces by first back-projecting point clouds to depth maps, which are then fused via TSDF. In contrast, both Surf3R-G and Surf3R-GD leverage Gaussian rendering to directly estimate high-quality depth maps for mesh reconstruction. As shown in Tab. 1, Surf3R-P achieves a significant improvement over DUSt3R, which requires explicit global alignment, with an F1-score increasing from 4.06 to 62.89.

<!-- image-->  
Figure 3: Qualitative comparison of surface reconstruction results on ScanNet++ dataset.

<!-- image-->  
Figure 4: Qualitative comparison of zero-shot surface reconstruction results on Replica dataset.

This underscores the advantage of aggregating geometric cues across all input views rather than relying on pairwise stereo matches processed one at a time. Moreover, enriching Surf3R-P with a 3D Gaussian representation (Surf3R-G) enhances global geometric consistency and raises the F1- score from 62.89 to 76.72. And the additional introduction of D-Normal regularization (Surf3R-GD) pushes it further to 78.71, yielding the best overall performance. As shown in Fig 3, our approach yields more accurate and complete reconstructions, particularly excelling at recovering planar surfaces and capturing fine-grained geometric details.

Moreover, our method demonstrates strong generalization capabilities. As shown in Tab. 2, under zero-shot inference on the Replica dataset, Surf3R-GD also achieves state-ofthe-art performance with an F1-score of 41.92. Compared to traditional methods such as NeuralRecon (Sun et al. 2021), DUSt3R (Wang et al. 2024), and Surf3R-P+NKSR (Huang et al. 2023), it consistently outperforms all baselines. As shown in Fig. 4, our method produces more complete and faithful surfaces, highlighting the superior generalization capability of our approach under unseen scenes.

## 4.3 Muliti-view NVS on ScanNet++

As shown in Tab. 3, Surf3R-GD consistently achieves the best novel view synthesis performance across all multi-view configurations on the ScanNet++ dataset. With only 4 input views, it outperforms all baselines, achieving a PSNR of 15.06, SSIM of 0.66, and LPIPS of 0.26, demonstrating robust geometry-aware synthesis even under sparse view conditions. As the number of input views increases to 12 and 24, our method maintains leading performance, particularly in perceptual quality metrics. Notably, the LPIPS score drops to 0.23 at 24 views, outperforming DUSt3R (0.68), indicating more stable view synthesis and sharper geometric details. These results underscore the strong generalization capability of our geometry-guided surface reconstruction framework, which not only delivers accurate 3D geometry but also enables high-quality NVS across varying input densities.

<table><tr><td>Method</td><td>NeuralRecon</td><td>DUSt3R</td><td>Surf3R-P</td><td>Surf3R-P + NKSR</td><td>Surf3R-G</td><td>Surf3R-GD (Ours)</td></tr><tr><td>Precision â</td><td>14.61</td><td>20.16</td><td>22.31</td><td>24.14</td><td>24.86</td><td>36.66</td></tr><tr><td>Recall â</td><td>12.33</td><td>14.80</td><td>21.52</td><td>31.37</td><td>32.06</td><td>49.04</td></tr><tr><td>F1-score â</td><td>13.41</td><td>16.90</td><td>21.88</td><td>27.16</td><td>27.96</td><td>41.92</td></tr></table>

Table 2: Quantitative comparison on Replica dataset. Bold indicates best result. Our method achieves superior performance across all metrics. Surf3R-P: point-map heads, trained with $\mathcal { L } _ { c } ; S u r f 3 R  â G ; +$ Gaussian heads, adds $\mathcal { L } _ { \boldsymbol { r } } ; S u r f 3 R \ â G D$ (Full model): + D-Normal regularization, adds $\mathcal { L } _ { s } , \bar { \mathcal { L } } _ { n }$ and $\mathcal { L } _ { d n }$
<table><tr><td rowspan="2"></td><td colspan="3">4 Views</td><td colspan="3">12 Views</td><td colspan="3">24 Views</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â SSIM â</td><td>LPIPS â</td></tr><tr><td>DUSt3R</td><td>11.66</td><td>0.47</td><td>0.63</td><td>10.72</td><td>0.46</td><td>0.67</td><td>10.81 0.40</td><td>0.68</td></tr><tr><td>Surf3R-G</td><td>13.21</td><td>0.64</td><td>0.31</td><td>16.77</td><td>0.55</td><td>0.30</td><td>17.69 0.55</td><td>0.26</td></tr><tr><td>Surf3R-GD</td><td>15.06</td><td>0.66</td><td>0.26</td><td>17.72</td><td>0.61</td><td>0.24</td><td>18.08 0.58</td><td>0.23</td></tr></table>

Table 3: NVS results on ScanNet++ dataset. Bold indicates best result. Our method achieves NVS rendering quality comparable with other Gaussian-based methods. Surf3R-G: + Gaussian heads, adds $\mathcal { L } _ { r } \mathrm { : }$ Surf3R-GD (Full model): + D-Normal regularization, adds $\mathcal { L } _ { s } , \mathcal { L } _ { n }$ and $\mathcal { L } _ { d n }$

<table><tr><td>Views</td><td>10 Views</td><td>30 Views</td><td>50 Views</td><td>70 Views</td><td>100 Views</td></tr><tr><td>Precision â</td><td>9.84</td><td>18.41</td><td>36.66</td><td>35.42</td><td>34.87</td></tr><tr><td>Recall â</td><td>17.57</td><td>30.38</td><td>49.04</td><td>47.24</td><td>45.29</td></tr><tr><td>F1-score â</td><td>12.29</td><td>22.88</td><td>41.92</td><td>40.47</td><td>39.37</td></tr></table>

Table 4: View Ablation Study on Replica dataset. Bold indicates best result. Performance comparison under different numbers of input views.

## 4.4 Ablation Study

View ablations. To investigate the impact of input view count on reconstruction quality, we conduct an ablation study by varying the number of input views during inference. As shown in Table 4, our method achieves the best performance when using 50 input views, with an F1-score of 41.92. Interestingly, increasing the number of views beyond this point does not lead to further improvements and may even slightly degrade performance. We attribute this to accumulated pose estimation errors from the input point clouds, which become more pronounced as the number of views increases, ultimately affecting the mesh reconstruction quality.

Branch and Loss ablations. As shown in Tab. 5, when restricting the network to a single branch (one reference view) degrades the F1-score from 36.66 to 23.24 (Row A). This indicates that, with sparse views, a single reference cannot establish reliable geometric correspondences across wide baselines, resulting in poor reconstructions. We also verify the effectiveness of different regularization terms on reconstruction quality. As shown in Tab. 5, both components contribute significantly to the reconstruction quality. Excluding the scale term (Row B) and the normal term (Row C) results in a notable decline in all metrics. Notabaly, removing the D-Normal term (Row D) leads to a substantial drop in performance, with the F1-score decreasing from 41.92 to 30.96. This suggests that the D-Normal regularization plays a critical role in encouraging the predicted normals to align with the underlying surface geometry. Our full model (Row E) achieves the best performance across all metrics, demonstrating the benefits of both components in producing surface reconstructions.

<table><tr><td>Ablation Item</td><td>Precision â</td><td>Recall â</td><td>F-score â</td></tr><tr><td>A. w/o Multi-branch</td><td>23.24</td><td>30.90</td><td>26.53</td></tr><tr><td>B. w/o Scale</td><td>32.9</td><td>43.20</td><td>37.35</td></tr><tr><td>C. w/o Normal</td><td>34.5</td><td>45.80</td><td>39.35</td></tr><tr><td>D. w/o D-Normal</td><td>25.38</td><td>39.69</td><td>30.96</td></tr><tr><td>E. Full</td><td>36.66</td><td>49.04</td><td>41.92</td></tr></table>

Table 5: Branch and Loss Ablation Study on Replica dataset. Bold indicates best result. Performance with different regularization terms.

## 5 Conclusion

In this paper, we propose Surf3R, a novel feed-forward framework for pose-free 3D surface reconstruction from sparse multi-view RGB inputs. Unlike traditional MVS methods that rely heavily on accurate camera calibration and iterative alignment, Surf3R eliminates the need for camera intrinsics or extrinsics by leveraging a cross-view attention mechanism and a multi-branch cross-reference fusion strategy. This enables effective feature propagation across arbitrarily selected views and mitigates the degradation caused by large viewpoint gaps. Additionally, We introduce a novel Depth-Normal Regularizer grounded in 3D Gaussian representations, which integrates normal estimation into the geometric parameter learning process, yielding more consistent and detailed surfaces. Extensive experiments on benchmark datasets such as ScanNet++ and Replica demonstrate that Surf3R achieves state-of-the-art performance in surface reconstruction while maintaining strong generalization ability in unseen scenarios.

## References

Barron, J. T.; Mildenhall, B.; Tancik, M.; Hedman, P.; Martin-Brualla, R.; and Srinivasan, P. P. 2021. Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields. In ICCV.

Bleyer, M.; Rhemann, C.; and Rother, C. 2011. PatchMatch Stereo-Stereo Matching with Slanted Support Windows. In BMVC.

Bonet, J. S. D.; and Viola, P. 1999. Poxels: Probabilistic Voxelized Volume Reconstruction.

Broadhurst, A.; Drummond, T.; and Cipolla, R. 2001. A Probabilistic Framework for Space Carving. In ICCV.

Charatan, D.; Li, S. L.; Tagliasacchi, A.; and Sitzmann, V. 2024. PixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction. In CVPR.

Chen, A.; Xu, Z.; Geiger, A.; Yu, J.; and Su, H. 2022. Tensorf: Tensorial Radiance Fields. In ECCV.

Chen, D.; Li, H.; Ye, W.; Wang, Y.; Xie, W.; Zhai, S.; Wang, N.; Liu, H.; Bao, H.; and Zhang, G. 2024a. PGSR: Planarbased Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction. CoRR, abs/2406.06521.

Chen, H.; Li, C.; and Lee, G. H. 2023. NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance. CoRR, abs/2312.00846.

Chen, H.; Wei, F.; Li, C.; Huang, T.; Wang, Y.; and Lee, G. H. 2024b. VCR-GauS: View Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction. In NeuralIPS.

Crandall, D. J.; Owens, A.; Snavely, N.; and Huttenlocher, D. P. 2013. SfM with MRFs: Discrete-Continuous Optimization for Large-Scale Structure from Motion. PAMI, 35(12).

Fridovich-Keil, S.; Yu, A.; Tancik, M.; Chen, Q.; Recht, B.; and Kanazawa, A. 2022. Plenoxels: Radiance Fields without Neural Networks.

Fu, Q.; Xu, Q.; Ong, Y. S.; and Tao, W. 2022. Geo-neus: Geometry-consistent neural implicit surfaces learning for multi-view reconstruction. In NeurIPS.

Furukawa, Y.; Hernandez, C.; et al. 2015. Multi-view stereo: Â´ A tutorial. Foundations and TrendsÂ® in Computer Graphics and Vision.

Garbin, S. J.; Kowalski, M.; Johnson, M.; Shotton, J.; and Valentin, J. 2021. FastNeRF: High-Fidelity Neural Rendering at 200 FPS. In ICCV.

Guedon, A.; and Lepetit, V. 2024. SuGaR: Surface-AlignedÂ´ Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering. In CVPR.

Guedon, A.; and Lepetit, V. 2024. SuGaR: Surface-Aligned Â´ Gaussian Splatting for Efficient 3D Mesh Reconstruction. In CVPR.

Huang, B.; Yu, Z.; Chen, A.; Geiger, A.; and Gao, S. 2024. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In SIGGRAPH.

Huang, J.; Gojcic, Z.; Atzmon, M.; Litany, O.; Fidler, S.; and Williams, F. 2023. Neural Kernel Surface Reconstruction. In CVPR.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G.Â¨ 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics.

Kutulakos, K. N.; and Seitz, S. M. 1999. A Theory of Shape by Space Carving. In Proceedings of the International Conference on Computer Vision, Kerkyra, Corfu, Greece, September 20-25, 1999, 307â314. IEEE Computer Society.

Kutulakos, K. N.; and Seitz, S. M. 2000. A Theory of Shape by Space Carving. International Journal of Computer Vision, 38: 199â218.

Leroy, V.; Cabon, Y.; and Revaud, J. 2024. Grounding image matching in 3d with mast3r. arXiv preprint arXiv:2406.09756.

Li, C.; Zhu, H.; Chen, H.; Zhang, J.; Chen, T.; Yang, S.; Shao, S.; Dong, W.; and Zhang, B. 2025. HRGS: Hierarchical Gaussian Splatting for Memory-Efficient High-Resolution 3D Reconstruction. CoRR, abs/2506.14229.

Lin, J.; Li, Z.; Tang, X.; Liu, J.; Liu, S.; Liu, J.; Lu, Y.; Wu, X.; Xu, S.; Yan, Y.; and Yang, W. 2024. VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction. In CVPR.

Liu, S.; Zhang, Y.; Peng, S.; Shi, B.; Pollefeys, M.; and Cui, Z. 2020. Dist: Rendering deep implicit signed distance function with differentiable sphere tracing. In CVPR.

Ma, B.; Zhou, J.; Liu, Y.-S.; and Han, Z. 2023. Towards better gradient consistency for neural signed distance functions via level set alignment. In CVPR.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM.

Muller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In- Â¨ stant Neural Graphics Primitives with a Multiresolution Hash Encoding. ACM Transactions on Graphics.

Park, J. J.; Florence, P.; Straub, J.; Newcombe, R.; and Lovegrove, S. 2019. Deepsdf: Learning continuous signed distance functions for shape representation. In CVPR.

Penner, E.; and Zhang, L. 2017. Soft 3D Reconstruction for View Synthesis. ACM Transactions on Graphics.

Reiser, C.; Peng, S.; Liao, Y.; and Geiger, A. 2021. Kilonerf: Speeding Up Neural Radiance Fields with Thousands of Tiny MLPs. In ICCV.

Schonberger, J. L.; and Frahm, J. 2016. Structure-from- Â¨ Motion Revisited. In CVPR.

Schonberger, J. L.; Zheng, E.; Frahm, J.-M.; and Pollefeys, M. Â¨ 2016. Pixelwise view selection for unstructured multi-view stereo. In ECCV.

Seitz, S. M.; Curless, B.; Diebel, J.; Scharstein, D.; and Szeliski, R. 2006. A comparison and evaluation of multiview stereo reconstruction algorithms. In CVPR.

Seitz, S. M.; and Dyer, C. R. 1999. Photorealistic Scene Reconstruction by Voxel Coloring. Int. J. Comput. Vis., 35(2): 151â173.

Sitzmann, V.; Chan, E.; Tucker, R.; Snavely, N.; and Wetzstein, G. 2020. Metasdf: Meta-learning signed distance functions. In NeurIPS.

Straub, J.; Whelan, T.; Ma, L.; Chen, Y.; Wijmans, E.; Green, S.; Engel, J. J.; Mur-Artal, R.; Ren, C. Y.; Verma, S.; Clarkson, A.; Yan, M.; Budge, B.; Yan, Y.; Pan, X.; Yon, J.; Zou, Y.; Leon, K.; Carter, N.; Briales, J.; Gillingham, T.; Mueggler, E.; Pesqueira, L.; Savva, M.; Batra, D.; Strasdat, H. M.; De Nardi, R.; Goesele, M.; Lovegrove, S.; and Newcombe, R. A. 2019. The Replica Dataset: A Digital Replica of Indoor Spaces. CoRR.

Sun, J.; Xie, Y.; Chen, L.; Zhou, X.; and Bao, H. 2021. Neural-Recon: Real-Time Coherent 3D Reconstruction From Monocular Video. In CVPR.

Tang, J.; Ren, J.; Zhou, H.; Liu, Z.; and Zeng, G. 2024. DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation. In ICLR.

Tulsiani, S.; Zhou, T.; Efros, A. A.; and Malik, J. 2017. Multiview Supervision for Single-view Reconstruction via Differentiable Ray Consistency. In CVPR.

Ummenhofer, B.; Zhou, H.; Uhrig, J.; Mayer, N.; Ilg, E.; Dosovitskiy, A.; and Brox, T. 2017. Demon: Depth and Motion Network for Learning Monocular Stereo. In CVPR.

Wang, H.; and Agapito, L. 2024. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061.

Wang, J.; Chen, M.; Karaev, N.; Vedaldi, A.; Rupprecht, C.; and Novotny, D. 2025. VGGT: Visual Geometry Grounded Â´ Transformer. In CVPR.

Wang, P.; Liu, L.; Liu, Y.; Theobalt, C.; Komura, T.; and Wang, W. 2021a. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-View Reconstruction. arXiv preprint arXiv:2106.10689.

Wang, P.; Liu, L.; Liu, Y.; Theobalt, C.; Komura, T.; and Wang, W. 2021b. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. In NeurIPS.

Wang, S.; Leroy, V.; Cabon, Y.; Chidlovskii, B.; and Revaud, J. 2024. Dust3r: Geometric 3d vision made easy. In CVPR.

Yao, Y.; Luo, Z.; Li, S.; Fang, T.; and Quan, L. 2018. MVSnet: Depth Inference for Unstructured Multi-View Stereo. In ECCV.

Yariv, L.; Gu, J.; Kasten, Y.; and Lipman, Y. 2021. Volume Rendering of Neural Implicit Surfaces. NeurIPS.

Yeshwanth, C.; Liu, Y.-C.; NieÃner, M.; and Dai, A. 2023. Scannet++: A high-fidelity dataset of 3d indoor scenes. In ICCV.

Yifan, W.; Serena, F.; Wu, S.; Oztireli, C.; and Sorkine- Â¨ Hornung, O. 2019. Differentiable Surface Splatting for Point-Based Geometry Processing. ACM Transactions on Graphics, 38(6): 1â14.

Yu, Z.; Peng, S.; Niemeyer, M.; Sattler, T.; and Geiger, A. 2022. MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction. In NeurIPS.

Zak Murez, J. B. A. S. V. B., Tarrence van As; and Rabinovich, A. 2020. Atlas: Endto-End 3D Scene Reconstruction from Posed Images. In ECCV.

Zhang, J.; Yao, Y.; Li, S.; Luo, Z.; and Fang, T. 2020. Visibility-Aware Multi-View Stereo Network. In BMVC.

Zhang, S.; Wang, J.; Xu, Y.; Xue, N.; Rupprecht, C.; Zhou, X.; Shen, Y.; and Wetzstein, G. 2025. FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views. In CVPR.