# FTSplat: Feed-forward Triangle Splatting Network

Jinlin Xiong, Can Li, Jiawei Shen, Zhigang Qi, Lei Sunâ, Dongyang Zhao

<!-- image-->  
Fig. 1: Overview of the proposed FTSplat. Given multi-view input images, our feed-forward FTSplat directly and efficiently predicts a triangular surface representation of the scene. The reconstructed mesh supports photo-realistic novel view rendering and can be readily imported into simulation software such as Blender for downstream applications. Compared to existing optimization-based triangular surface methods that typically require several minutes for reconstruction, our approach enables scene modeling within sub-second.

Abstractâ High-fidelity three-dimensional (3D) reconstruction is essential for robotics and simulation. While Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) achieve impressive rendering quality, their reliance on timeconsuming per-scene optimization limits real-time deployment. Emerging feed-forward Gaussian splatting methods improve efficiency but often lack explicit, manifold geometry required for direct simulation. To address these limitations, we propose a feed-forward framework for triangle primitive generation that directly predicts continuous triangle surfaces from calibrated multi-view images. Our method produces simulation-ready models in a single forward pass, obviating the need for perscene optimization or post-processing. We introduce a pixelaligned triangle generation module and incorporate relative 3D point cloud supervision to enhance geometric learning stability and consistency. Experiments demonstrate that our method achieves efficient reconstruction while maintaining seamless compatibility with standard graphics and robotic simulators.

## I. INTRODUCTION

High-fidelity 3D reconstruction is a fundamental capability for robotic perception, simulation, and digital twin applications. As robotic systems increasingly require accurate environmental understanding and physically consistent scene representations, the efficient and accurate recovery of 3D scene models from images has become a fundamental problem in robotics and computer vision. Recently, Neural radiance fields (NeRF) [1] methods have demonstrated remarkable progress by representing scenes with implicit continuous functions, significantly improving novel view synthesis and reconstruction quality. Building upon this line of work, 3D Gaussian Splatting (3DGS) [2] further introduces explicit Gaussian primitives to enable high-quality and real-time rendering, substantially advancing the practical deployment of 3D reconstruction systems.

However, most of NeRF-based and 3DGS-based methods rely on per-scene optimization or iterative procedures, which significantly limits their deployment efficiency in large-scale scene reconstruction and online robotic applications. To address this limitation, recent studies have explored feedforward Gaussian Splatting approaches [3]â[5] that directly predict Gaussian primitives from multi-view images in a single forward pass, achieving substantially improved inference efficiency while maintaining competitive reconstruction quality.

Despite the advantages of 3DGS in rendering performance, Gaussian primitives lack explicit geometric structure, making them difficult to directly integrate with mainstream physicsbased simulators and robotic simulation platforms. To bridge this gap, recent mesh-based representation methods [6]â[8] replace Gaussian primitives with triangular facets for scene representation, offering more standardized and simulationcompatible geometric structures while preserving efficient rendering. Such explicit mesh representations can be directly imported into widely used simulation and graphics software, such as Blender, without requiring additional surface reconstruction or post-processing. However, their reliance on iterative optimization introduces significant computational cost, limiting scalability and reducing practicality in timesensitive robotic applications.

To bridge the gap between feed-forward efficiency and explicit geometric representation, we propose a feed-forward triangular primitive generation framework that directly predicts continuous triangular surface primitives from multiview images. By combining the efficiency of feed-forward reconstruction with the structural advantages of mesh-based representations, our method generates simulation-compatible surface structures in a single forward pass. This enables efficient and high-fidelity 3D reconstruction while producing continuous triangular surfaces that can be directly imported into existing graphics and robotics simulation environments, such as Blender, without additional mesh reconstruction or post-processing, thereby facilitating seamless integration with simulation and digital twin pipelines.

Specifically, we present a feed-forward framework that directly generates a triangular surface representation of a 3D scene from multiple images. Given a set of calibrated multiview images, our method extracts image features using a pretrained vision transformer and a multi-view transformer encoder. Based on the feature matching strategy [4], a sparse feature point cloud is constructed and subsequently decoded into a set of 3D vertices augmented with opacity and color represented by spherical harmonics. The resulting pointbased representation is further processed by a pixel-aligned triangular surface generation module, which predicts the face connectivity and constructs an explicit triangular surface representation. The generated triangular primitives are rendered via a differentiable triangle rasterizer [8], and the rendered images are supervised using photometric losses against the ground-truth views. In addition, since triangular surface reconstruction critically depends on accurate 3D geometry, we introduce an auxiliary 3D point cloud supervision during early training stages with a higher weight to encourage rapid geometric convergence. As training progresses, the weight of the 3D supervision is gradually reduced, allowing the optimization to focus on high-quality appearance and texture reconstruction.

In summary, the main contributions of this paper include:

â¢ We propose the first feed-forward framework that directly predicts continuous triangular surface representations of 3D scenes from multi-view images, which are natively compatible with existing graphics and robotics simulation platforms. This enables efficient generation of continuous surface models without per-scene optimization or additional post-processing.

â¢ We design a pixel-aligned triangular surface generation module that converts feature point clouds predicted by the feed-forward network into explicit triangular surface primitives suitable for efficient rasterization.

â¢ We introduce a relative 3D point cloud supervision and adopt a geometry-to-appearance training strategy, which emphasizes geometric consistency in the early training stage and gradually shifts the optimization focus toward high-quality texture reconstruction, enabling stable and accurate convergence of the proposed framework.

## II. RELATED WORK

## A. Optimization-based 3D Reconstruction Methods

Optimization-based 3D reconstruction methods have long been the dominant paradigm for high-fidelity scene reconstruction. These approaches typically rely on explicit or implicit per-scene parameter optimization, aiming to minimize multi-view consistency or reprojection errors to recover detailed geometric and appearance representations. Neural Radiance Field (NeRF) methods represent scenes as continuous implicit functions and optimize them via volumetric rendering, achieving remarkable progress in novel view synthesis and high-quality reconstruction. Subsequent works have focused on improving the training efficiency and representational capacity of NeRF, including approaches such as Instant-NGP [9], which significantly accelerates optimization through multi-resolution hash encoding, as well as various variants that explore alternative sampling strategies, network architectures, and regularization schemes [10]â[12]. While these methods strike different trade-offs between reconstruction quality and computational efficiency, they generally rely on per-scene optimization, which limits their scalability to large-scale environments and real-time applications.

To address the aforementioned limitations, 3D Gaussian Splatting introduces an explicit scene representation based on Gaussian primitives. By directly optimizing the spatial parameters and appearance attributes of Gaussians, 3DGS enables high-quality reconstruction with efficient real-time rendering. Building upon this framework, subsequent studies have proposed a variety of extensions and improvements in terms of geometric modeling, appearance representation, and training stability, such as regularizing Gaussian distributions, introducing hierarchical representations, and extending the formulation to dynamic or editable scenes [13]â[15]. While these methods further enhance reconstruction quality and broaden the applicability of Gaussian-based representations, their core pipelines still rely on scene-specific optimization procedures. More recently, 2D Gaussian Splatting [16] reformulates the representation by anchoring Gaussian primitives in image-aligned or surface-aligned two-dimensional domains, which improves geometric consistency and reduces ambiguities caused by unconstrained 3D Gaussians. Nevertheless, 2DGS remains a scene-specific optimization-based approach, requiring per-scene training to obtain high-quality reconstructions.

Recent studies have further explored Mesh Splatting methods that adopt triangular facets as explicit scene representations. By replacing Gaussian primitives with triangular elements or explicit surface primitives, these approaches retain the rendering efficiency of splatting-based pipelines while introducing more standardized geometric representations. Compared to Gaussian-based formulations, triangular meshes offer superior compatibility with mainstream graphics and physics simulation engines, facilitating the direct use of highquality reconstructed models in downstream tasks such as simulation, collision detection, and physical dynamics analysis. However, existing Mesh Splatting methods still rely on per-scene optimization strategies, resulting in considerable computational cost and limited deployment efficiency, which remain insufficient for online robotic applications.

<!-- image-->  
Fig. 2: Overview of the proposed feed-forward triangular surface reconstruction network. Multi-view images are processed by a Multi-View Depth Estimation module to obtain fused features enriched with depth information. The fused features are used to predict depth maps and back-project an initial 3D point cloud, while a 2D U-Net with a triangle head decodes additional vertex attributes (opacity and spherical harmonics color). A surface generation module infers face connectivity to produce the final triangular surface. Differentiable rasterization enables photometric supervision, and external 3D point cloud supervision provides explicit geometric constraints during training.

## B. Feed-forward 3D Reconstruction Methods

In contrast to optimization-based approaches, feed-forward 3D reconstruction methods aim to directly predict scene representations from multi-view images in a single forward pass, thereby significantly improving inference efficiency and system scalability.

PixelSplat [3] is among the earliest works to explore feedforward Gaussian Splatting, where a neural network directly predicts Gaussian primitives from multi-view images, enabling fast 3D reconstruction without per-scene optimization. Building upon PixelSplat, Mvsplat [4] introduces multiview feature matching and explicit geometric constraints, significantly improving geometric consistency and reconstruction quality of the feed-forward predicted Gaussians. Furthermore, Depthsplat [5] incorporates monocular depth features into the feed-forward Gaussian Splatting framework, leveraging depth priors to enhance geometric recoverability and achieve more stable reconstruction results in complex scenes.

Despite the significant improvements in inference efficiency achieved by the aforementioned approaches, they continue to rely on Gaussian primitives as scene representations, which lack explicit geometric surface structures and thus limit their direct applicability to simulation and robotic tasks. In contrast, our method further explores a feed-forward triangular surface generation strategy that preserves the efficiency of feed-forward inference while introducing explicit triangular surface representations, enabling high-fidelity 3D reconstruction that is more compatible with physics-based simulation platforms.

## III. ALGORITHM

This section provides a detailed description of the proposed approach. In Section III-A, we review the necessary background, including the definition of triangular surface representations and the rendering process of triangular primitives. In Section III-B, we present the technical details of our method, covering the overall network architecture, loss function design, and training strategy.

## A. Preliminaries

To ensure better compatibility with existing graphics and simulation platforms, the triangular surface representation generated by our feed-forward network follows the formulation of MeshSplatting [6], namely a continuous triangular surface model. Specifically, the model consists of a set of vertices and their associated attributes, including vertex positions v, opacity values o, RGB colors c represented using spherical harmonics (SH) coefficients, and a smoothing parameter Ï. The corresponding dimensions are $\mathbf { v } \in \mathbb { R } ^ { N \times 3 }$ ${ \boldsymbol { o } } \in \mathbb { R } ^ { N }$ , and $\mathbf { c } \in \mathbb { R } ^ { \bar { N } \times 3 \times d _ { h } }$ . In our implementation, the smoothing parameter Ï is set to zero by default. In addition to vertex attributes, the triangular surface model includes a face connectivity tensor $\mathbf { f } \in \mathbb { R } ^ { T \times 3 }$ , where each row stores the indices of the three vertices forming a triangular face. This connectivity explicitly defines a surface composed of T triangular primitives.

Based on the triangular surface representation described above, we employ a differentiable triangle rasterization process to render the reconstructed geometry. Specifically, the triangular vertices v are first projected onto the 2D image plane using a standard pinhole camera model, yielding their corresponding positions in pixel space. Since the smoothing parameter Ï is fixed to zero in our formulation, the reconstructed triangular surfaces exhibit explicit and welldefined boundaries, without introducing additional boundary smoothing or soft blending effects. The rendering strategy of the triangular primitives follows the same formulation as that adopted in 3D Gaussian Splatting, which can be expressed as follows:

$$
C ( \mathbf { p } ) = \sum _ { i = 1 } ^ { n } { c _ { i } o _ { i } \prod _ { j = 1 } ^ { i - 1 } { ( 1 - o _ { j } ) } }\tag{1}
$$

## B. Feed-forward Triangular Surface Reconstruction

Figure 2 illustrates the overall architecture of our feedforward network. Given n multi-view input images, we first extract image features and predict depth maps using a multi-view depth estimation module. The features are then processed by a 2D U-Net followed by a triangle head to decode the vertex representations. Finally, a Surface Generation module is applied to infer face connectivity, producing the final triangular surface model.

Feature Extraction. We first extract image features using a lightweight ResNet [17] with shared weights across views. The resulting features are then processed by a six-layer multi-view Swin Transformer [18]â[20] to exchange information among different views, producing multi-view features $\{ F _ { m v } ^ { i } \} _ { i = 1 } ^ { n } , \breve { F _ { m v } ^ { i } } \ \in \ \mathbb { R } ^ { \frac { H } { s } \times \frac { W } { s } \times \stackrel {  } { C _ { 1 } } }$ . In addition, we employ a pretrained Depth Anything ${ \tt V } 2 \ [ 2 1 ]$ model to extract monocular depth-aware features $\{ F _ { m o n o } ^ { i } \} _ { i = 1 } ^ { n } , F _ { m v } ^ { i } \in \mathbb { R } ^ { \frac { H } { s } \times \frac { W } { s } \times C _ { 2 } }$ The two types of features are subsequently fused to form the final image feature representation.

Multi-View Depth Estimation. We estimate depth maps for the input images using a cost volume based approach. Specifically, we uniformly sample D depth hypotheses within a predefined near and far depth range. For the feature map $F _ { j }$ of view $j ,$ we leverage the known camera intrinsics and extrinsics to project the features onto the image plane of view i at each depth hypothesis, resulting in D warped features $F _ { j  i }$ . The warped features $F _ { j  i }$ are then compared with the reference view features $F _ { i }$ using a dotproduct operation to compute feature correlations, which are aggregated to construct the cost volume $\boldsymbol { C _ { i } } \in \mathbb { R } ^ { \frac { H } { s } \times \frac { W } { s } \times D }$ Finally, the cost volume is used as weights to compute a weighted sum over the D depth hypotheses, yielding the final depth estimate for each view.

Triangular Surface Generation. The constructed cost volume, multi-view features, monocular features, and predicted depth maps are jointly fed into a 2D U-Net [22] to produce per-pixel feature representations. Meanwhile, using the estimated depth maps and known camera intrinsics and extrinsics, image pixels are back-projected into 3D space to obtain an initial point cloud. We then employ a lightweight multi-layer perceptron (MLP), referred to as the triangle head, to decode per-point attributes, including opacity o and color represented by spherical harmonics (SH) coefficients. This process yields the vertices of the triangular surface representation along with their associated attributes.

To handle depth discontinuities, we initially explored depth-gradient-based edge pruning and 3D KNN connectivity in Euclidean space. However, empirical results show that these strategies tend to introduce uncontrollable holes or break surface continuity. In contrast, direct pixel-level connectivity achieves the highest computational efficiency and provides more stable global topology in practice.

To construct the surface connectivity, we further adopt a simple yet effective pixel-level face generation strategy. Specifically, the number of generated points is $\begin{array} { r } { V = n \times \frac { H } { s } \times \frac { \bar { W } } { s } } \end{array}$ , where n denotes the number of input views, H and W are the input image resolution, and s is the downsampling factor. Each 3D point is mapped back to its corresponding image pixel coordinate $( u , v )$ . For each pixel, we generate two adjacent triangular faces by connecting the vertices corresponding to neighboring pixels $( u + 1 , v ) , ( u , v + 1 )$ and (u â 1, v), (u, v â 1), respectively. This strategy ensures full coverage of the visible surface while minimizing the number of generated triangular faces, resulting in a compact and efficient triangular surface representation.

Loss Function Design. The overall loss function of our method consists of two components: an image-renderingrelated photometric loss and a geometry-related loss defined on the reconstructed 3D point cloud. The formula is as follows:

$$
\mathcal { L } = \mathcal { L } _ { p h o t o } + \lambda _ { p o i n t s } \mathcal { L } _ { p o i n t s }\tag{2}
$$

The photometric loss $\mathcal { L } _ { p h o t o }$ is defined as:

$$
\mathcal { L } _ { p h o t o } = \mathcal { L } _ { L 1 } + \lambda _ { l p i p s } \mathcal { L } _ { L P I P S } + \lambda _ { d s } \mathcal { L } _ { d s }\tag{3}
$$

where $\mathcal { L } _ { L 1 }$ denotes the L1 difference between the rendered image and the corresponding ground truth image, and $\mathcal { L } _ { L P I P S }$ represents the perceptual similarity loss between the rendered image and the ground-truth image measured by LPIPS [29]. In addition, we introduce a depth smoothness loss $\mathcal { L } _ { d s }$ [24], formulated as:

$$
\mathcal { L } _ { d s } = | \partial _ { x } D _ { r e n d e r } | e ^ { - | \partial _ { x } I _ { g t } | } + | \partial _ { y } D _ { r e n d e r } | e ^ { - | \partial _ { y } I _ { g t } | }\tag{4}
$$

This term serves as a regularization on the estimated depth maps, encouraging the depth gradients to be consistent with the image gradients, thereby promoting smooth depth variations while preserving sharp depth discontinuities at image edges.

Given that triangular surface representations require stronger 3D geometric consistency than Gaussian-based methods, we introduce an additional geometry-aware loss defined on the reconstructed 3D point cloud. Specifically, we use multi-view 3D point clouds predicted by multiple foundation depth models, including Depth Anything V3 [25] and VGGT [26], as supervisory signals to guide training. The geometric loss is defined as:

$$
\mathcal { L } _ { \mathrm { p o i n t s } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. \boldsymbol { N } ( \mathbf { V } _ { b , i } ) - \boldsymbol { N } ( \mathbf { P } _ { b , i } ) \right. _ { 2 } ^ { 2 } ,\tag{5}
$$

where V denotes the triangular vertices predicted by the proposed feed-forward network, and P represents the 3D

<!-- image-->  
Fig. 3: Qualitative comparison of reconstruction quality between our method and optimization-based triangle rasterization methods.

point cloud predicted by Depth Anything V3. $\mathcal { N } ( \cdot )$ denotes a robust normalization operator that removes global translation and scale ambiguity, which is defined as follows:

$$
\mathcal { N } ( \mathbf { X } _ { b , i } ) = \frac { \mathbf { X } _ { b , i } - \mathrm { m e d i a n } ( \mathbf { X } _ { b } ) } { Q _ { \alpha } \big ( \| \mathbf { X } _ { b , i } - \mathrm { m e d i a n } ( \mathbf { X } _ { b } ) \| _ { 2 } \big ) }\tag{6}
$$

where $Q _ { \alpha } ( \cdot )$ denotes the Î±-quantile operator, which suppresses the influence of outliers by using the quantile of the distance distribution as a robust scale estimator.

Specifically, since the supervisory point clouds provide only relative geometric structure without absolute scale information, we impose the geometric constraint in a relative coordinate space on the vertices predicted by our feed-forward network. This encourages the reconstructed triangular surface representations to be more compact and effectively alleviates the floating primitives near object boundaries that are commonly observed in feed-forward Gaussian Splatting methods, thereby improving geometric consistency in 3D space.

Furthermore, to accelerate training convergence, we assign a larger weight $\lambda _ { p o i n t s }$ to the geometry loss during the early training stage, allowing the network to focus on learning stable 3D geometric structures. As training progresses, the weight $\lambda _ { p o i n t s }$ is gradually reduced, enabling the imagerendering-related photometric loss to dominate the optimization and guiding the network to emphasize high-quality texture and appearance reconstruction on triangular surfaces.

## IV. EXPERIMENT

We conduct experiments on the RealEstate10K [27] dataset at a resolution of $2 5 6 \times 2 5 6$ . Our FTSplat model is trained on the RealEstate10K dataset for 400k iterations with a batch size of 2. Using identical reconstruction and evaluation viewpoints, we first compare our method with existing optimization-based triangle rasterization approaches, including Triangle Splatting [8] and MeshSplatting [6]. In addition, we include 3D Gaussian Splatting (3DGS) [2] and 2D Gaussian Splatting (2DGS) [16] as representative Gaussian-based baselines. For all methods, each scene is reconstructed from two input views and evaluated on three target views. Reconstruction quality is quantitatively evaluated using PSNR, SSIM [28], and LPIPS [29]. Furthermore, we compare FTSplat with several representative feed-forward Gaussian-based methods [4], [5] to evaluate reconstruction quality. Finally, we perform ablation studies to validate the effectiveness of the proposed geometry-related loss.

<!-- image-->  
Fig. 4: Qualitative comparison of 3D spatial consistency between our method and feed-forward Gaussian Splatting methods.

A. Reconstruction Quality Comparison with Optimizationbased Methods

In this section, we compare our feed-forward triangle surfaces generation approach with existing optimizationbased triangle rasterization methods. In addition, conventional 3D Gaussian Splatting and 2D Gaussian Splatting are included as reference baselines. Since optimization-based methods require scene-specific iterative refinement and incur substantial computational cost, we conduct experiments on a subset of several dozen scenes from the RealEstate10K test set. For fair comparison, all methods use identical input and evaluation viewpoints. The camera poses of the input views are provided, with two views used for reconstruction and three additional views used for evaluation in each scene.

TABLE I: Quantitative comparison of reconstruction quality with optimization-based methods. (iter.: number of iterations; con.: connectivity, i.e., whether the generated representation is geometrically connected.)
<table><tr><td>algorithm</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>iter. â con.</td></tr><tr><td>3DGS</td><td>18.35</td><td>0.601</td><td>0.350 30k</td><td>Ã</td></tr><tr><td>2DGS</td><td>16.17</td><td>0.526</td><td>0.419 30k</td><td>Ã</td></tr><tr><td>trianglesplatting</td><td>18.53</td><td>0.683</td><td>0.279 30k</td><td>Ã</td></tr><tr><td>meshsplatting</td><td>19.78</td><td>0.685</td><td>0.340 30k</td><td>â</td></tr><tr><td>FTSplat(Ours)</td><td>20.39</td><td>0.707</td><td>0.257 1</td><td>â</td></tr></table>

As shown in Table I, our method consistently achieves higher PSNR, SSIM, and LPIPS scores than optimizationbased triangle rasterization methods under sparse-view settings, while requiring only a single forward pass. In contrast to optimization-based methods that typically require minutes of per-scene iterative refinement, our approach reconstructs each scene in only 0.17s. Moreover, similar to MeshSplatting, the generated surfaces produced by our method preserve structural connectivity, which further enables the resulting triangular surface models to be directly compatible with existing graphics and robotics simulation platforms without additional post-processing.

As illustrated in Fig. 3, we further present a qualitative comparison between our method and existing optimizationbased triangle rasterization approaches under sparse-view settings. Although optimization-based methods can achieve high reconstruction quality when a large number of training views are available, their performance degrades when only a limited number of views with large viewpoint variations are provided. In such cases, iterative optimization tends to converge to local minima during triangular surface reconstruction, resulting in noticeable noise and structural artifacts when rendering from novel viewpoints. In particular, as the disparity between training views increases, the generalization ability of optimization-based methods to unseen viewpoints becomes increasingly limited. In contrast, our feed-forward framework is able to produce geometrically consistent triangular surface models across viewpoints through a single forward pass, yielding more stable and coherent reconstructions under sparse-view inputs.

B. Reconstruction Quality Comparison with Feed-forward Gaussian Splatting Methods

In this subsection, we conduct quantitative evaluations to compare our method with existing feed-forward Gaussian Splatting (GS) approaches in terms of novel view synthesis quality. In addition, we provide qualitative comparisons to assess the geometric consistency of the reconstructed 3D models in the actual 3D space between our method and feedforward GS methods.

TABLE II: Quantitative comparison of reconstruction quality with feed-forward Gaussian Splatting methods.
<table><tr><td>algorithm</td><td>PSNRâ</td><td>SSIMâ LPIPSâ</td></tr><tr><td>Mvsplat</td><td>27.03</td><td>0.891 0.106</td></tr><tr><td>Depthsplat</td><td>27.61</td><td>0.903 0.099</td></tr><tr><td>FTSplat(Ours)</td><td>20.39</td><td>0.707 0.257</td></tr></table>

As shown in Table II, in terms of quantitative novel view rendering performance, the reconstructed triangular surface produced by our method is slightly inferior to the results rendered by feed-forward GS methods. This observation is consistent with the comparison between mesh-based methods and the original 3DGS reported in [6]. However, regarding 3D spatial consistency, as illustrated in Fig. 4, the triangle surfaces reconstructed by our method effectively eliminates the fog-like floating artifacts commonly observed in 3DGS reconstructions. As a result, our method produces a cleaner and more geometrically consistent 3D representation, which is more suitable for direct use in robotic perception tasks.

## C. Ablation Study

Quantitative Comparison. We conduct an ablation study to evaluate the effectiveness of the proposed relative 3D point cloud supervision in our feed-forward triangle surfaces generation framework. In particular, we investigate the impact of different point cloud generation methods used for supervision, including VGGT and Depth Anything 3. The quantitative results are summarized in Table III.

TABLE III: Quantitative ablation study on relative 3D point cloud supervision.(w/o: without point cloud supervision; w/: point cloud signals provided by VGGT or Depth Anything 3.)
<table><tr><td>algorithm</td><td>PSNRâ SSIMâ LPIPSâ</td></tr><tr><td>w/o</td><td>13.06 0.401 0.509</td></tr><tr><td>w/ vggt 20.07</td><td>0.692 0.275</td></tr><tr><td>w/ da3</td><td>20.39 0.707 0.257</td></tr></table>

As shown in the table, employing Depth Anything 3 to generate supervisory point clouds leads to the best overall performance. Using VGGT as the external point cloud generator yields slightly inferior results, while removing the relative 3D point cloud supervision results in a significant performance degradation. These findings demonstrate that the proposed supervision strategy plays a critical role in improving reconstruction quality, and that the choice of point cloud generation method further influences the final performance.

<!-- image-->  
(a) w/ Point Supervision

<!-- image-->  
(b) w/o Point Supervision  
Fig. 5: Qualitative comparison of rendering quality under different ablation settings

Qualitative Comparison. As illustrated in Fig. 5, removing the relative 3D point cloud supervision leads to a noticeable degradation in the rendering quality of the reconstructed triangle surfaces produced by the feed-forward model. In particular, the synthesized novel views exhibit significant blurring and visible artifacts when the supervision is absent. This behavior suggests that, without explicit geometric regularization, the network struggles to learn structurally consistent surface representations during training.

<!-- image-->

<!-- image-->  
(a) w/ Point Supervision  
(b) w/o Point Supervision  
Fig. 6: Qualitative comparison of 3D structure reconstruction under different ablation settings

More pronounced differences emerge when analyzing the reconstructed 3D geometry itself. As shown in Fig. 6, in the absence of relative 3D point cloud supervision, the model lacks explicit geometric constraints to guide structural recovery. Consequently, although the optimization may appear to converge in the image rendering space, it tends to reach a pseudo-converged solution corresponding to a local minimum. This results in a degenerate 3D reconstruction, where the recovered triangle surface collapses into a nearly planar, image-stitching-like structure rather than forming a geometrically coherent surface.

Both quantitative and qualitative results consistently validate the effectiveness of the proposed relative 3D point cloud supervision. Quantitatively, it improves all evaluation metrics, while qualitatively enhancing rendering fidelity and geometric consistency of the reconstructed surfaces. These findings confirm its critical role in boosting the overall performance of the feed-forward triangle surface generation framework.

## V. CONCLUSIONS

In this work, we presented FTSplat, a feed-forward framework that directly generates continuous triangular surface representations from multi-view images. By combining efficient single-pass inference with mesh-based geometric structure, the proposed method enables fast and stable 3D reconstruction without per-scene optimization while producing simulation-ready surfaces. The pixel-aligned triangle generation module and relative 3D point-cloud supervision improve geometric consistency, leading to reliable reconstruction under sparse-view settings. Experimental results demonstrate strong reconstruction quality, improved crossview consistency, and direct compatibility with graphics and robotic simulation platforms.

Despite these advantages, limitations remain in handling occluded regions, where incomplete geometric cues may degrade surface estimation. Future work will explore more robust surface generation strategies and stronger geometric priors to further enhance reconstruction performance in challenging scenarios.

## REFERENCES

[1] Mildenhall, Ben, et al. âNerf: Representing scenes as neural radiance fields for view synthesis.â Communications of the ACM 65.1 (2021): 99-106.

[2] Kerbl, Bernhard, et al. â3D Gaussian splatting for real-time radiance field rendering.â ACM Trans. Graph. 42.4 (2023): 139-1.

[3] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR, 2024. 1, 2, 5, 7

[4] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In ECCV, 2024. 1, 2, 3, 4, 5, 7, 8, 12

[5] Xu, Haofei, et al. âDepthsplat: Connecting gaussian splatting and depth.â Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[6] Held, Jan, et al. âMeshSplatting: Differentiable Rendering with Opaque Meshes.â arXiv preprint arXiv:2512.06818 (2025).

[7] Mai, Alexander, et al. âRadiance Meshes for Volumetric Reconstruction.â arXiv preprint arXiv:2512.04076 (2025).

[8] Held, Jan, et al. âTriangle Splatting for Real-Time Radiance Field Rendering.â arXiv preprint arXiv:2505.19175 (2025).

[9] Muller, Thomas, et al. âInstant neural graphics primitives with a Â¨ multiresolution hash encoding.â ACM transactions on graphics (TOG) 41.4 (2022): 1-15.

[10] Fridovich-Keil, Sara, et al. âPlenoxels: Radiance fields without neural networks.â Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[11] Barron, Jonathan T., et al. âMip-nerf 360: Unbounded anti-aliased neural radiance fields.â Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[12] Deng, Kangle, et al. âDepth-supervised nerf: Fewer views and faster training for free.â Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[13] Li, Changbai, et al. âHRGS: Hierarchical Gaussian Splatting for Memory-Efficient High-Resolution 3D Reconstruction.â arXiv preprint arXiv:2506.14229 (2025).

[14] Lan, Lei, et al. â3dgs2: Near second-order converging 3d gaussian splatting.â Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 2025.

[15] Sridhara, Shashank N., et al. âRegion-Adaptive Learned Hierarchical Encoding for 3D Gaussian Splatting Data.â arXiv preprint arXiv:2510.22812 (2025).

[16] Huang, Binbin, et al. â2d gaussian splatting for geometrically accurate radiance fields.â ACM SIGGRAPH 2024 conference papers. 2024.

[17] He, Kaiming, et al. âDeep residual learning for image recognition.â Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[18] Liu, Ze, et al. âSwin transformer: Hierarchical vision transformer using shifted windows.â Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[19] Xu, Haofei, et al. âGmflow: Learning optical flow via global matching.â Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[20] Xu, Haofei, et al. âUnifying flow, stereo and depth estimation.â IEEE Transactions on Pattern Analysis and Machine Intelligence 45.11 (2023): 13941-13958.

[21] Yang, Lihe, et al. âDepth anything v2.â Advances in Neural Information Processing Systems 37 (2024): 21875-21911.

[22] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. âU-net: Convolutional networks for biomedical image segmentation.â International Conference on Medical image computing and computer-assisted intervention. Cham: Springer international publishing, 2015.

[23] Zhang, Richard, et al. âThe unreasonable effectiveness of deep features as a perceptual metric.â Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[24] Godard, Clement, et al. âDigging into self-supervised monocular depth Â´ estimation.â Proceedings of the IEEE/CVF international conference on computer vision. 2019.

[25] Lin, Haotong, et al. âDepth anything 3: Recovering the visual space from any views.â arXiv preprint arXiv:2511.10647 (2025).

[26] Wang, Jianyuan, et al. âVggt: Visual geometry grounded transformer.â Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[27] Zhou, Tinghui, et al. âStereo magnification: Learning view synthesis using multiplane images.â arXiv preprint arXiv:1805.09817 (2018).

[28] Wang, Zhou, et al. âImage quality assessment: from error visibility to structural similarity.â IEEE transactions on image processing 13.4 (2004): 600-612.

[29] Zhang, Richard, et al. âThe unreasonable effectiveness of deep features as a perceptual metric.â Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.