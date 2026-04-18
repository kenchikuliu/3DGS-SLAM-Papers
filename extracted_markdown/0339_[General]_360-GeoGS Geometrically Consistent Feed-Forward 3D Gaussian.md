# 360-GeoGS: Geometrically Consistent Feed-Forward 3D Gaussian Splatting Reconstruction for 360 Images

Jiaqi Yaoâ, Zhongmiao Yanâ, Jingyi Xuâ, Songpengcheng Xiaâ, Yan Xiangâ, Ling Peiââ 

â Shanghai Key Laboratory of Navigation and Location Based Services, Shanghai Jiao Tong University

â  State Key Laboratory of Submarine Geoscience, School of Automation and Intelligent Sensing,

Shanghai Jiao Tong University

Abstractâ3D scene reconstruction is fundamental for spatial intelligence applications such as AR, robotics, and digital twins. Traditional multi-view stereo struggles with sparse viewpoints or low-texture regions, while neural rendering approaches, though capable of producing high-quality results, require per-scene optimization and lack real-time efficiency. Explicit 3D Gaussian Splatting (3DGS) enables efficient rendering, but most feed-forward variants focus on visual quality rather than geometric consistency, limiting accurate surface reconstruction and overall reliability in spatial perception tasks. This paper presents a novel feed-forward 3DGS framework for 360 images, capable of generating geometrically consistent Gaussian primitives while maintaining high rendering quality. A Depth-Normal geometric regularization is introduced to couple rendered depth gradients with normal information, supervising Gaussian rotation, scale, and position to improve point cloud and surface accuracy. Experimental results show that the proposed method maintains high rendering quality while significantly improving geometric consistency, providing an effective solution for 3D reconstruction in spatial perception tasks.

Index Termsâ3D Reconstruction, 3D Gaussian Splatting, 360 Image.

## I. Introduction

3D scene reconstruction aims to recover scene geometry and appearance from multi-view observations and is essential for applications such as autonomous driving, AR/VR, robotic perception, and digital twins. In indoor navigation, accurate and efficient 3D modeling is crucial for spatial perception and localization. Prior research has explored robust sensing and efficient inference in complex environments [1]â[3], highlighting the importance of balancing accuracy, robustness, and computational efficiency.

Multi-View Stereo (MVS) achieves high-precision reconstruction via multi-view matching and depth estimation, but performance degrades under low-texture conditions. Neural Radiance Fields (NeRF) [4] improve view synthesis but require dense inputs and per-scene optimization. For efficient inference, explicit 3D Gaussian Splatting (3DGS) [5] represents scenes with Gaussian ellipsoids, enabling fast rendering and gradient-based optimization. Following this approach, feed-forward variants [6], [7] improve inference efficiency and generalization through end-to-end prediction, while still often struggling to maintain geometric consistency. In contrast, optimizationbased methods, such as VCR-GauS [8] and NeuSG [9], incorporate geometric priors and normal constraints to refine scene structure, achieving higher accuracy at the cost of efficiency and generalization.

With the increasing adoption of panoramic cameras, 360 images have become an effective source for sparse-view reconstruction [10], capturing the entire scene in a single shot. Feed-forward panoramic methods focus on rendering quality, but projection distortions and unstable depth estimation often cause structural drift, limiting faithful geometric recovery.

To address these challenges, we propose 360-GeoGS, a feed-forward 3DGS framework for 360 image inputs that incorporates Depth-Normal (D-Normal) regularization. The framework predicts multi-view depth from 360 images and fuses various features, which are then processed by a U-Net to regress pixel-aligned Gaussian primitives. D-Normal regularization is applied in the rendering space to jointly supervise Gaussian position, scale, and orientation. Experiments on multiple panoramic benchmarks demonstrate that our method substantially improves geometric consistency and surface continuity while maintaining high rendering quality, outperforming existing feed-forward panoramic 3DGS methods. Our main contributions are as follows:

1) We propose a feed-forward 3DGS network for 360 image inputs, which employs a SphereCNN backbone to extract spherical features and build a spherical cost volume for depth estimation. Based on the estimated depth, the network performs feed-forward inference to rapidly predict 3D Gaussian parameters, achieving efficient and accurate 3D scene reconstruction.

2) We introduce D-Normal regularization, which jointly optimizes surface normals and Gaussian positions to ensure that neighboring Gaussian primitives form coherent local surfaces with consistent orientation and spatial alignment, thereby enhancing the geometric consistency and accuracy of the predicted 3DGS points.

<!-- image-->  
Fig. 1: Our pipeline extracts matching features from 360 images using a SphereCNN to construct a spherical cost volume and regress initial depth. Multi-scale features are also extracted by an image encoder and modulated via a FiLM module. The spherical cost volume, modulated multi-scale features, initial RGB images, and depth estimates are then fused to form a unified multi-modal representation, which is decoded by a U-Net and further refined by an adapter to produce per-pixel 3D Gaussian parameters. The network is trained with four losses: $\mathcal { L } _ { \mathrm { r g b } }$ 2 ${ \mathcal { L } } _ { \mathrm { s } } .$ ${ \mathcal { L } } _ { \mathrm { d n } }$ , and ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ (the definitions of ${ \mathcal { L } } _ { \mathrm { s } }$ and ${ \mathcal { L } } _ { \mathrm { d n } }$ are provided in Section III-C).

3) Extensive experiments across multiple benchmarks demonstrate that our approach delivers superior geometric performance while preserving high rendering quality.

## II. Related Work

## A. Sparse View Scene Reconstruction and Synthesis

Recent advances in 3D reconstruction and novel view synthesis have been largely driven by NeRF [4] and 3DGS [5]. Although they were initially designed for dense-view settings, increasing attention has been paid to achieving high-quality reconstruction and synthesis under sparse-view conditions. Existing methods can be divided into per-scene optimization methods [8], [9], [11], [12] and cross-scene feed-forward inference methods. The former enhance geometric and appearance stability by designing effective regularization constraints, but the computational cost is high due to the optimization process. In contrast, the latter learn strong priors from large-scale datasets, enabling fast reconstruction through a single forward pass, thus significantly improving inference efficiency.

## B. Feed-Forward 3DGS

3DGS leverages rasterization-based splatting to efficiently synthesize novel views, representing a scene with learnable Gaussian primitives. To further accelerate reconstruction and handle sparse-view settings, feed-forward

3DGS variants have been proposed. PixelSplat [6] introduced a feed-forward framework for scene-level Gaussian prediction. MVSplat [7] enhanced geometric accuracy through cost volumes, and DepthSplat [13] enhances multi-view consistency with depth estimation. Despite these advances, feed-forward methods often lack geometric consistency, particularly at indoor scene boundaries with discontinuous depth. Moreover, most existing methods are designed for perspective images, and their performance degrades significantly on panoramic inputs due to the wide field of view and projection distortions.

## C. Panoramic View Scene Reconstruction and Synthesis

Reconstruction and novel view synthesis from 360 images encounter challenges caused by geometric distortions in equirectangular projection and unstable depth estimation at high resolutions. Most methods assume dense panoramic inputs [14], while sparse views make depth and geometry estimation harder. 360Recon [15] predicts panoramic depth using an improved MVS approach, achieving accurate mesh geometry but limited rendering quality. PanoGRF [16] aggregates geometric and appearance features for high-quality synthesis; however, its large fusion network restricts inference and rendering speed.

Feed-forward 3DGS methods have been extended to 360 images, improving efficiency in panoramic view scene reconstruction and synthesis. Splatter-360 [17], based on MVSplat, adds depth constraints to enhance geometry but exhibits inconsistencies near scene boundaries.

PanSplat [18] enables high-resolution, real-time synthesis; nevertheless, its geometric constraints are insufficient to fully preserve 3D structure.

## III. Method

Our goal is to directly predict 3DGS parameters from 360 image inputs, enabling geometrically consistent scene reconstruction in a feed-forward manner. Section III-A introduces the overall architecture, Section III-B details the feed-forward prediction pipeline, and Section III-C presents the proposed D-Normal constraint that enforces geometric consistency.

## A. Framework Overview

Our network employs a feed-forward design mapping 360 images to 3D Gaussian primitives. As illustrated in Fig. 1, reconstruction starts with extracting multi-view matching features to build a spherical cost volume, which is then used to estimate an initial dense depth map as a geometric prior. Simultaneously, multi-scale features are extracted and modulated through Feature-wise Linear Modulation (FiLM) for cross-scale interaction. The fused multi-modal features, together with RGB inputs and the depth prior, are processed by a U-Net decoder and an adapter to regress per-pixel Gaussian parameters, including positions, covariance, opacity, and color. Predicted Gaussians are jointly supervised by geometric and photometric losses, with geometric supervision emphasizing surface consistency via D-Normal, ensuring accurate local geometry.

## B. Pipeline of Feed-forward 3DGS Prediction

1) Feature Encoding: We adopt the 360Recon framework as our baseline for feature extraction on spherical inputs. A SphereCNN backbone is used to obtain matching features from 360 images, which are used to construct a spherical cost volume and estimate an initial dense depth map as a geometric prior. Meanwhile, a set of multi-scale feature maps $\{ F _ { i } \} _ { i = 0 } ^ { 4 }$ is extracted, where low-level features retain detailed geometry and high-level features capture global context. To facilitate interaction across scales, we employ FiLM to adaptively modulate multi-scale features. Specifically, high-level features are first aggregated to form a global conditioning representation:

$$
C _ { \mathrm { c o n d } } = \Phi \left( \{ F _ { i } \} _ { i = 2 } ^ { 4 } \right) ,\tag{1}
$$

where $\Phi ( \cdot )$ denotes the compression and aggregation operation. Then, the low-level features are modulated as:

$$
\hat { F } = \gamma ( C _ { \mathrm { c o n d } } ) \cdot F + \beta ( C _ { \mathrm { c o n d } } ) ,\tag{2}
$$

where $\gamma ( \cdot )$ and $\beta ( \cdot )$ generate per-channel scaling and shifting parameters. The fused features are combined with the matching features, dense depth predictions, and RGB inputs to form a multi-modal representation for subsequent 3DGS regression.

2) 3DGS Parameter Prediction: The fused features are decoded by a U-Net decoder to produce initial Gaussian primitive parameters, which are then refined through an adapter module for rendering compatibility. The adapter normalizes rotations, adjusts scales with depth, and transforms spherical harmonic coefficients to yield per-pixel Gaussian primitives at full resolution (512Ã1024) aligned with the input panoramas. The predicted parameters include:

Gaussian centers Âµ. The network predicts per-pixel offsets in image space, which are combined with depth to project points into 3D camera coordinates and further transformed to world coordinates using the camera-toworld matrix.

Opacity Î±. Opacity is derived from the matching confidence, computed as a normalized probability distribution from the cost volume.

Covariance Î£. The covariance is defined by a scale factor s and rotation matrix $R ( \theta )$ :

$$
\Sigma = R ( \theta ) ^ { T } \mathrm { d i a g } ( s ) R ( \theta ) ,\tag{3}
$$

where s is mapped through a Sigmoid function to preserve proportionality to depth and image resolution, and $R ( \theta )$ is parameterized via a normalized quaternion.

Spherical harmonics c. The spherical harmonic coefficients c are regressed from the fused features to encode view-dependent color representations.

## C. Geometric Constraint

To enhance the geometric accuracy of feed-forward 3DGS predictions and better align Gaussian points with object surfaces, we introduce a geometric constraint.

1) Normal and Intersection Depth: The spatial positions of feed-forward 3DGS points primarily depend on the estimated depth and are theoretically expected to lie on object surfaces. However, since Gaussians are represented as ellipsoids, their centers often deviate from the true surface, leading to geometric inconsistencies. To address this, we follow NeuSG and compress each ellipsoid along its smallest scale direction into a height-flattened form, allowing the Gaussian to better adhere to the underlying surface.

Specifically, the scale factor ${ \bf s } = ( s _ { 1 } , s _ { 2 } , s _ { 3 } ) ^ { T }$ defines the ellipsoidâs extent along each principal axis. The normal vector n is then defined along the direction of the minimal scale component. Minimizing this component effectively flattens the ellipsoid, and a scale regularization loss $\mathcal { L } _ { \mathrm { s } }$ is applied to constrain it towards zero:

$$
\mathcal { L } _ { \mathrm { s } } = \| \operatorname* { m i n } ( s _ { 1 } , s _ { 2 } , s _ { 3 } ) \| _ { 1 } .\tag{4}
$$

In depth computation, conventional methods typically obtain the depth from the center position $\mathrm { p } = ( p _ { x } , p _ { y } , p _ { z } )$ of each Gaussian in the camera coordinate system. However, this ignores the normal vector n and thus limits the effectiveness of geometric constraints. We therefore adopt a more appropriate approach, computing the intersection depth between the camera ray r and the flattened Gaussian surface, defined as:

<!-- image-->  
GT

<!-- image-->  
MVSplat

<!-- image-->  
PanSplat

<!-- image-->  
Splatter-360

<!-- image-->  
Ours

Fig. 2: Predicted 3D Gaussian spatial distributions of the same scene reconstructed by different methods.  
<!-- image-->  
Fig. 3: Novel view rendering comparison of our method, Splatter-360, PanSplat, and MVSplat on the HM3D dataset.

<!-- image-->  
Fig. 4: Novel view depth comparison among MVSplat, Splatter-360, and our method on the HM3D dataset.

$$
\mathbf { d ( n , p ) } = r _ { z } ( \mathbf { n \cdot p } ) / ( \mathbf { n \cdot r } ) ,\tag{5}
$$

Here, $r _ { z }$ denotes the z-component of the ray r. The intersection depth depends on both the position p and

the normal vector n of the Gaussian, allowing them to be jointly constrained during optimization to improve depth estimation accuracy.

2) D-Normal Regularization: Following this approach, we adopt the D-Normal regularization. Specifically, a depth map is generated using the 3DGS renderer, following

TABLE I: Quantitative comparison of depth estimation metrics on the HM3D and Replica datasets. â  indicates the model that was trained by us on the panoramic dataset. Best in each column is bolded.
<table><tr><td rowspan="2">Method</td><td colspan="4">HM3D</td><td colspan="4">Replica</td></tr><tr><td>Abs Diffâ</td><td>Abs Relâ</td><td>RMSEâ</td><td> $\delta < 1 . 2 5 \uparrow$ </td><td>Abs Diffâ</td><td>Abs Relâ</td><td>RMSEâ</td><td> $\delta < 1 . 2 5 \uparrow$ </td></tr><tr><td>MVSplatâ </td><td>0.140</td><td>0.094</td><td>0.258</td><td>91.150</td><td>0.186</td><td>0.111</td><td>0.282</td><td>88.216</td></tr><tr><td>Splatter-360</td><td>0.0988</td><td>0068</td><td>0.193</td><td>5.417</td><td>00.103</td><td>0.068</td><td>0.185</td><td>95.412</td></tr><tr><td> Ours</td><td>0.053</td><td>0.069</td><td>0.141</td><td>96.423</td><td>0.055</td><td>0.068</td><td>0.138</td><td>966.528</td></tr></table>

TABLE II: Quantitative comparison of novel view synthesis metrics on the HM3D and Replica datasets. Best in each column is bolded.
<table><tr><td rowspan="3">Method</td><td colspan="3">HM3D</td><td colspan="3">Replica</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>MVSplatâ </td><td>29.537</td><td>0.892</td><td>0.138</td><td>28.682</td><td>0.915</td><td>0.117</td></tr><tr><td>PanSplat</td><td>29.733</td><td>0.925</td><td>0.126</td><td>31.821</td><td>0.960</td><td>0.067</td></tr><tr><td>Splatter-360</td><td>31.669</td><td>0.925</td><td>0.100</td><td>31.584</td><td>0.952</td><td>0.064</td></tr><tr><td>Ours</td><td>31.043</td><td>0.920</td><td>0.098</td><td>31.137</td><td>0.945</td><td>0.066</td></tr></table>

a procedure analogous to RGB rendering.

$$
\hat { D } = \sum _ { i \in M } d _ { i } \alpha _ { i } T _ { i } / ( \sum _ { i \in M } \alpha _ { i } T _ { i } ) \qquad T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } )\tag{6}
$$

where $d _ { i }$ is the intersection depth from (5) and M is the number of Gaussians the ray passes through. Subsequently, the rendered normal $\bar { \mathrm { N } } _ { d } ( \mathrm { n } , \mathrm { p } )$ is obtained by computing finite differences of the depth map along the horizontal and vertical directions and taking their cross product. This normal depends on both the Gaussian normal n and the position p:

$$
\bar { \mathbf { N } } _ { d } ( \mathbf { n } , \mathbf { p } ) = \frac { \nabla _ { v } \mathbf { d } ( \mathbf { n } , \mathbf { p } ) \times \nabla _ { h } \mathbf { d } ( \mathbf { n } , \mathbf { p } ) } { | \nabla _ { v } \mathbf { d } ( \mathbf { n } , \mathbf { p } ) \times \nabla _ { h } \mathbf { d } ( \mathbf { n } , \mathbf { p } ) | } .\tag{7}
$$

The D-Normal regularization enforces consistency between the rendered normal $\bar { \mathbf { N } } _ { d }$ and the target normal N, enabling joint optimization of Gaussian positions and orientations, as illustrated in Fig. 5. The regularization loss is formulated as:

$$
\mathcal { L } _ { \mathrm { d n } } = \| \bar { \mathbf { N } } _ { d } - \mathbf { N } \| _ { 1 } + ( 1 - \bar { \mathbf { N } } _ { d } \cdot \mathbf { N } ) .\tag{8}
$$

Our overall loss function is defined as:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { 1 } { \mathcal { L } } _ { \mathrm { s } } + \lambda _ { 2 } { \mathcal { L } } _ { \mathrm { d e p t h } } + \lambda _ { 3 } { \mathcal { L } } _ { \mathrm { d n } } .\tag{9}
$$

## IV. Experiments

## A. Implementation Details

Our method is implemented in PyTorch, with intersection distance computations accelerated using custom CUDA kernels. All experiments are conducted on a single NVIDIA A100 GPU with 80 GB VRAM. The RGB loss for training is a linear combination of MSE and LPIPS, weighted 1 and 0.05, respectively, and the hyperparameters $\lambda _ { 1 } , \ \lambda _ { 2 } .$ , and $\lambda _ { 3 }$ are empirically set to 1, 0.1, and 0.01.

<!-- image-->  
Fig. 5: Illustration of the D-Normal regularization. $\bar { \mathbf { N } } _ { d }$ is supervised by the ground-truth normal through ${ \mathcal { L } } _ { \mathrm { d n } }$ (defined in subsection III-C2 ), guiding the flattened Gaussians to better fit the true surface.

## B. Datasets and Metrics

We evaluate our model on two panoramic datasets, HM3D [19] and Replica [20], which contain diverse indoor scenes. For quantitative comparison, we compare our method with several state-of-the-art 360 approaches, including Splatter-360 and PanSplat. Additionally, MVSplat is retrained on the 360 datasets and included as another baseline. We evaluate the performance of all methods on novel view synthesis using standard metrics, including PSNR, SSIM, and LPIPS, and on depth prediction using Abs Diff, Abs Rel, RMSE, and $\delta \ < \ 1 . 2 5$ . Moreover, geometric reconstruction is assessed on HM3D using Accuracy, Completeness, and Chamfer Distance.

## C. Qualitative Results

Qualitative comparisons are presented in Fig. 2, Fig. 3, and Fig. 4. As Fig. 3 illustrates, state-of-the-art methods achieve visually similar results for novel view synthesis, with our method and Splatter-360 producing slightly better appearance in the right-side sample. In Fig. 4, MVSplat exhibits notable depth errors, such as the left edge of the door in Sample 2, while Splatter-360 improves overall reconstruction but still shows limited surface depth consistency in Samples 2 and 4. In contrast, our method generates depth predictions with stronger geometric consistency and higher accuracy. The predicted 3DGS point clouds in Fig. 2 further highlight this improvement, demonstrating that our approach produces 3DGS points with clearly enhanced surface.

TABLE III: Quantitative results of the ablation study. $^ { 6 6 } \mathrm { W } / \mathrm { O } ^ { 7 }$ indicates âwithoutâ. Best in each column is bolded.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>| Abs Diffâ</td><td>Abs Relâ</td><td>RMSEâ</td><td> $\delta < 1 . 2 5 \uparrow$ </td><td>Acc(m)â</td><td>Comp(m)â</td><td>Chamfer(m)â</td></tr><tr><td>w/o D-N</td><td>29.078</td><td>0.887</td><td>0.161</td><td>0.086</td><td>0.068</td><td>0.177</td><td>94.878</td><td>0.054</td><td>0.715</td><td>0.769</td></tr><tr><td>w/o D-N+Scales</td><td>28.189</td><td>0.868</td><td>0.158</td><td>0.11</td><td>0..102</td><td>0.205</td><td>93.715</td><td>0059</td><td>0.731</td><td>0.790</td></tr><tr><td>w/o D-N+Scales+Fusion</td><td>27.713</td><td>0.841</td><td>0.176</td><td>0.120</td><td>0.108</td><td>0.228</td><td>93.430</td><td>00.061</td><td>0.742</td><td>0.802</td></tr><tr><td>Full</td><td>31.043</td><td>0.920</td><td>00.098</td><td>053</td><td>0.069</td><td>0.141</td><td>96.423</td><td>.049</td><td>0.691</td><td>0.740</td></tr></table>

TABLE IV: Quantitative comparison of 3D reconstruction metrics on the HM3D dataset. Best in each column is bolded.
<table><tr><td>Method</td><td>Acc(m)â</td><td>Comp(m)â</td><td>Chamfer(m)â</td></tr><tr><td>MVSplatâ </td><td>0.076</td><td>0.862</td><td>0.938</td></tr><tr><td>Splatter-360</td><td>0.062</td><td>0.719</td><td>0.780</td></tr><tr><td>Ours</td><td>0.049</td><td>0.691</td><td>0.740</td></tr></table>

## D. Quantitative Results

Tables I, II, and IV summarize the quantitative performance of our method compared with MVSplat, Splatter-360, and PanSplat on the HM3D and Replica datasets. As shown in Tables I and IV, our approach outperforms Splatter-360 in geometric reconstruction and depth estimation metrics, indicating that the predicted 3DGS points exhibit stronger geometric consistency and better alignment with object surfaces. Table II reports novel view synthesis metrics, where our method achieves rendering quality comparable to the current state-of-the-art, with minor differences across multiple metrics. Overall, these results demonstrate that our feed-forward 3DGS framework achieves high-fidelity geometric reconstruction while maintaining competitive rendering performance.

## E. Ablation Results

We conduct an ablation study to evaluate the contributions of D-Normal (D-N), scale flattening (Scales), and multi-scale feature fusion (Fusion). As shown in Table III, the full model consistently outperforms the ablated variants across rendering, depth, and geometric reconstruction metrics. Removing D-N or scale flattening degrades depth accuracy and geometric consistency, while omitting multi-scale fusion reduces rendering quality. These results indicate that each component contributes complementarily, with the integrated model producing the most accurate and geometrically consistent 3DGS predictions.

## V. Conclusion

In this paper, we propose a feed-forward 3DGS framework for 360 image inputs, integrating multi-view matching features, multi-scale feature encoding with FiLM, depth priors from a spherical cost volume, and D-Normal regularization. The encoded features are decoded by a U-Net and refined via an adapter to produce per-pixel Gaussian primitive parameters. Our method enables accurate scene reconstruction and high-fidelity novel view synthesis under sparse-view conditions. Experiments demonstrate competitive rendering quality, precise depth estimation, and enhanced geometric consistency compared to stateof-the-art methods, while ablation studies confirm the effectiveness of each component.

Limitations and future work. Our approach currently targets indoor scenes and relies on accurate camera poses. Future work will explore pose-free reconstruction from 360 images and investigate whether occluded regions can be recovered using generative models, further enhancing the completeness and realism of panoramic scene reconstruction.

## References

[1] Y. Chen, R. Chen, L. Pei, T. KrÃ¶ger, H. Kuusniemi, J. Liu, and W. Chen, âKnowledge-based error detection and correction method of a multi-sensor multi-network positioning platform for pedestrian indoor navigation,â in IEEE/ION Position, Location and Navigation Symposium, 2010, pp. 873â879.

[2] F. Wen, L. Adhikari, L. Pei, R. F. Marcia, P. Liu, and R. C. Qiu, âNonconvex regularization-based sparse recovery and demixing with application to color image inpainting,â IEEE Access, vol. 5, pp. 11 513â11 527, 2017.

[3] Y. Li, K. Yan, Z. He, Y. Li, Z. Gao, L. Pei, R. Chen, and N. El-Sheimy, âCost-effective localization using rss from single wireless access point,â IEEE Transactions on Instrumentation and Measurement, vol. 69, no. 5, pp. 1860â1870, 2020.

[4] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[5] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering,â ACM Trans. Graph., vol. 42, no. 4, July 2023.

[6] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, âpixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 19 457â19 467.

[7] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J. Cham, and J. Cai, âMvsplat: Efficient 3d gaussian splatting from sparse multi-view images,â in European Conference on Computer Vision. Springer, 2024, pp. 370â386.

[8] H. Chen, F. Wei, C. Li, T. Huang, Y. Wang, and G. H. Lee, âVcr-gaus: view consistent depth-normal regularizer for gaussian surface reconstruction,â in Proceedings of the 38th International Conference on Neural Information Processing Systems, 2024.

[9] H. Chen, C. Li, and G. H. Lee, âNeusg: Neural implicit surface reconstruction with 3d gaussian splatting guidance,â CoRR, vol. abs/2312.00846, 2023.

[10] Q. Wu, X. Xu, X. Chen, L. Pei, C. Long, J. Deng, G. Liu, S. Yang, S. Wen, and W. Yu, â360-vio: A robust visualâinertial odometry using a 360 camera,â IEEE Transactions on Industrial Electronics, vol. 71, no. 9, pp. 11 136â11 145, 2023.

[11] J. Deng, Q. Wu, X. Chen, S. Xia, Z. Sun, G. Liu, W. Yu, and L. Pei, âNerf-loam: Neural implicit representation for large-scale incremental lidar odometry and mapping,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 8218â8227.

[12] T. Ye, Q. Wu, J. Deng, G. Liu, L. Liu, S. Xia, L. Pang, W. Yu, and L. Pei, âThermal-nerf: Neural radiance fields from an infrared camera,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 1046â1053.

[13] H. Xu, S. Peng, F. Wang, H. Blum, D. Barath, A. Geiger, and M. Pollefeys, âDepthsplat: Connecting gaussian splatting and depth,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 16 453â16 463.

[14] J. Bai, L. Huang, J. Guo, W. Gong, Y. Li, and Y. Guo, â360-gs: Layout-guided panoramic gaussian splatting for indoor roaming,â CoRR, vol. abs/2402.00763, 2024.

[15] Z. Yan, Q. Wu, S. Xia, J. Deng, X. Mu, R. Jin, and L. Pei, â360recon: An accurate reconstruction method based on depth fusion from 360 images,â CoRR, vol. abs/2411.19102, 2024.

[16] Z. Chen, Y.-P. Cao, Y.-C. Guo, C. Wang, Y. Shan, and S.- H. Zhang, âPanogrf: Generalizable spherical radiance fields for wide-baseline panoramas,â Advances in Neural Information Processing Systems, vol. 36, pp. 6961â6985, 2023.

[17] Z. Chen, C. Wu, Z. Shen, C. Zhao, W. Ye, H. Feng, E. Ding, and S.-H. Zhang, âSplatter-360: Generalizable 360 gaussian splatting for wide-baseline panoramic images,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 21 590â21 599.

[18] C. Zhang, H. Xu, Q. Wu, C. C. Gambardella, D. Phung, and J. Cai, âPansplat: 4k panorama synthesis with feed-forward gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[19] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. Turner, E. Undersander, W. Galuba, A. Westbury, A. X. Chang et al., âHabitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai,â CoRR, vol. abs/2109.08238, 2021.

[20] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., âThe replica dataset: A digital replica of indoor spaces,â CoRR, vol. abs/1906.05797, 2019.