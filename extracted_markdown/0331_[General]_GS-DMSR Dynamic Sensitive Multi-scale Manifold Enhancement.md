# GS-DMSR: Dynamic Sensitive Multi-scale Manifold Enhancement for Accelerated High-Quality 3D Gaussian Splatting

Nengbo Lu1, Minghua Pan (B)2,3, Shaohua Sun1, and Yizhou Liang1

1 School of Artificial Intelligence, Guilin University of Electronic Technology, Guilin, Guangxi, 541004, China

2 School of Computer Science and Information Security, Guilin University of Electronic Technology, Guilin 541004, China

panmh@guet.edu.cn

3 Guangxi Key Laboratory of Cryptography and Information Security, Guilin University of Electronic Technology, Guilin 541004, China

Abstract. In the field of 3D dynamic scene reconstruction, how to balance model convergence rate and rendering quality has long been a critical challenge that urgently needs to be addressed, particularly in high-precision modeling of scenes with complex dynamic motions. To tackle this issue, this study proposes the GS-DMSR method. By quantitatively analyzing the dynamic evolution process of Gaussian attributes, this mechanism achieves adaptive gradient focusing, enabling it to dynamically identify significant differences in the motion states of Gaussian models. It then applies differentiated optimization strategies to Gaussian models with varying degrees of significance, thereby significantly improving the model convergence rate. Additionally, this research integrates a multi-scale manifold enhancement module, which leverages the collaborative optimization of an implicit nonlinear decoder and an explicit deformation field to enhance the modeling efficiency for complex deformation scenes. Experimental results demonstrate that this method achieves a frame rate of up to 96 FPS on synthetic datasets, while effectively reducing both storage overhead and training time.Our code and data are available at https://anonymous.4open.science/r/GS-DMSR-2212.

Keywords: Dynamic Scene Rendering Â· Gaussian Representation Â· View Synthesis.

## 1 Introduction

Novel view synthesis, as a core research direction in 3D vision, plays a pivotal role in industries such as virtual reality, augmented reality, and film production. This technology aims to construct spatio-temporally continuous representations of scenes from sparse 2D image inputs, enabling dynamic rendering from arbitrary viewpoints and timestamps. Particularly in dynamic scene modeling, the precise reconstruction of complex motion patterns from spatio-temporally limited input data remains a critical challenge in current research.

Neural Radiance Fields (NeRF)[1] have achieved groundbreaking progress in novel view synthesis by representing scenes through implicit functions. This method employs volume rendering techniques to establish mappings between 2D images and 3D scenes. However, the original NeRF suffers from inefficiencies in training and rendering. Although subsequent improvements have reduced training times from days to minutes, its rendering process still incurs significant latency, falling short of real-time requirements.

3D Gaussian Splatting (3D-GS)[2], an innovative approach using explicit scene representation, marks a major breakthrough in 3D reconstruction. By modeling scenes with explicit 3D Gaussian distributions, 3D-GS elevates rendering speeds to real-time levels. Unlike the computationally intensive volume rendering in NeRF, 3D-GS introduces differentiable splatting techniques to directly project 3D Gaussians onto 2D imaging planes. This representation not only achieves real-time rendering but also provides an explicit scene structure, facilitating scene manipulation and editing, thereby expanding possibilities for scene reconstruction.

<!-- image-->  
Fig. 1. Motion Saliency-Driven Dynamic Gaussian Optimization.For each Gaussian, we quantify the dynamic variation properties of its attributes and classify them into Gaussians with different saliency levels. We then label these Gaussians and optimize the high-saliency ones.

To address the spatio-temporal representation limitations of traditional 3D-GS in dynamic scene reconstruction, Wu et al. proposed the 4D Gaussian Splatting (4D-GS)[3] framework, achieving breakthroughs in dynamic scene modeling through a hierarchical spatio-temporally coupled representation architecture. This method innovatively constructs a hybrid Gaussian deformation field network, whose core consists of a spatio-temporal structure encoder and a lightweight multi-head Gaussian deformation decoder. The former realizes low-rank compressed representation of motion fields through spatio-temporal basis function decomposition, while the latter enables parameterized modeling of cross-frame deformation fields via attention mechanisms. However, the framework still requires global optimization of millions of Gaussian parameters, where redundant updates of static or low-frequency dynamic parameters significantly degrade convergence efficiency. To resolve this, our study proposes a dynamics-aware parameter update mechanism that adaptively focuses gradient updates based on motion saliency coefficients, thereby accelerating convergence. Furthermore, we introduce a multi-scale manifold enhancement module that synergizes non-linear implicit decoders with explicit deformation fields, enhancing reconstruction capability while preserving spatio-temporal continuity. Our contributions are as follows:

<!-- image-->  
Fig. 2. The overall pipeline of our model. For a set of 3D Gaussians G, we extract the center coordinates and timestamp t of each Gaussian by querying multi-resolution voxel planes; using MS â DGO, we distinguish Gaussians with different saliency coefficients and optimize those with high saliency coefficients; next, we compute the voxel features; decoding these features via a miniaturized multi-head Gaussian deformation decoder yields the deformed 3D Gaussian Gâ² at timestamp t; finally, applying Gaussian splatting to the deformed Gaussians generates the final rendered image.

â¢ By introducing the Motion Saliency-Driven Dynamic Gaussian Optimization (MS-DGO) method, which quantifies dynamic changes in Gaussian properties to achieve adaptive gradient focusing, distinguishes between Gaussians with varying saliency levels in real time, and implements differentiated optimization, the modelâs convergence is significantly accelerated while storage redundancy is reduced.

â¢ Relying on the synergistic optimization of an implicit nonlinear decoder and an explicit deformation field, it enhances the modeling capability of complex dynamic deformation scenarios, effectively compensates for potential rendering quality degradation caused by MS-DGO, and maintains spatiotemporal continuity.

## 2 Related Work

## 2.1 Novel View Synthesis

Novel view synthesis, as a core technical challenge in 3D reconstruction, has witnessed the emergence of diverse scene representations and rendering strategies in recent years. Among classical explicit-structure methods, approaches like light field mapping [4] , parameterized meshes [5â7], discrete voxels [8â10], and multiplane projection [11, 12]rely on dense supervision for high-fidelity rendering, while NeRF-based methods [13, 14] construct continuous scene representations through implicit neural radiance fields, achieving breakthrough progress in view generation accuracy. Addressing dynamic scene modeling demands, researchers in [38,39,42] deconstructed static constraints; particularly, [15] proposed a temporally aware explicit dynamic voxel model, elevating training efficiency to under 30 minutes, a paradigm subsequently advanced in works like [16, 17]. Within dynamic modeling, deformation-driven methods [18, 19] utilize cross-frame optical flow fields for pixel-level spatial transformations, while emerging temporal decoupling neural volumetric techniques [20â22] significantly accelerate dynamic modeling by decoupling spatial sampling along the temporal dimension. For multi-view systems, studies such as [23, 24] have designed tailored optimization architectures. However, despite breakthroughs in training efficiency, monocular dynamic scene reconstruction still faces significant real-time inference bottlenecks. This study innovatively constructs a joint optimization framework that, through hierarchical representation design and computational path compression, simultaneously achieves accelerated training and guarantees real-time rendering quality under sparse input conditions.

## 2.2 Neural Rendering with Point Clouds

## 3 Preliminary

This section analyzes the geometric representation and rasterization process of 3D Gaussian Splatting in subsection 3.1.

## 3.1 3D Gaussian Splatting

3D Gaussian distribution is an explicit 3D scene representation that exists in the form of point clouds. Each 3D Gaussian distribution is characterized by a

covariance matrix Î£ and a center point x, where x is referred to as the Gaussian mean.

$$
G ( X ) = e ^ { - \frac { 1 } { 2 } x ^ { T } \Sigma ^ { - 1 } x } .\tag{1}
$$

To enable separate optimization of parameters, the covariance matrix Ï can be decomposed into a scaling matrix S and a rotation matrix R:

$$
\boldsymbol { \Sigma } = \mathbf { R } \mathbf { S } \mathbf { S } ^ { T } \mathbf { R } ^ { T } .\tag{2}
$$

When rendering a new viewpoint, the system employs differentiable rasterization techniques to project 3D Gaussian distributions onto the camera plane. According to the derivation in literature , the covariance matrix $\Sigma ^ { \prime }$ in the camera coordinate system can be computed using the following formula, which involves the view transformation matrix W and the Jacobian matrix J of the affine approximation of the projection transformation.

$$
\Sigma ^ { \prime } = J W \Sigma W ^ { T } J ^ { T } .\tag{3}
$$

Each 3D Gaussian contains the following optimizable parameters: a spatial coordinate $X ~ \in ~ R ^ { 3 }$ ,color defined by k-dimensional spherical harmonic coefficients $R \in \ R ^ { k }$ ,where k is the order of spherical harmonics.opacity $\alpha \in R$ â², quaternion rotation parameters $r \in R ^ { 4 }$ and 3D scaling factors $s \in R ^ { 3 }$ . During the pixel shading process, the color and opacity values of each Gaussian point are computed based on the radiance field expression in Equation 1.For N ordered Gaussian points covering this pixel, their color blending formula is given by:

$$
C = \sum _ { i \in N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { i } ) .\tag{4}
$$

Here, $c _ { i } , \alpha _ { i }$ represents the density and color of this point computed by a 3D Gaussian G with covariance Î£ multiplied by an optimizable per-point opacity and SH color coefficients.

## 4 Method

## 4.1 Motion Saliency-Driven Dynamic Gaussian Optimization

To accelerate model convergence, this study proposes a dynamically sensitive gradient updating mechanism based on a parametric motion saliency coefficient. This mechanism achieves adaptive gradient focusing by quantifying the dynamic variations in Gaussian attributes. After completing the standard 3D Gaussian splatting framework and deformation field processing, the model performs realtime screening of scene Gaussians according to the motion saliency coefficient, which provides an accurate mathematical representation of Gaussian motion state changes. The screening process employs a preset saliency threshold to partition Gaussians into two categories: high-saliency Gaussians and low-saliency Gaussians.

<!-- image-->  
Fig. 3. For the synthetic dataset, visualization comparison experiments with other models were conducted [3, 21, 25]. The rendering results retain the default green background, and this study adopts the rendering parameter configurations of [3].

High-saliency Gaussians exhibit significant deviation from the target scene state and undergo iterative refinement through deformation fields. Low-saliency Gaussians reach convergence with photorealistic approximation and are exempted from subsequent deformation field updates. By dynamically allocating computational resources and gradient updates, this mechanism continuously focuses optimization on the most promising Gaussian units, substantially enhancing training efficiency while reducing computational redundancy.

## 4.2 Multi-scale Disentangled Manifold Deformation

Upon completion of 3D Gaussian feature encoding, our framework employs a multi-head Gaussian deformation decoder $\mathcal { D } = ( \phi _ { x } , \phi _ { r } , \phi _ { s } )$ where three independent multilayer perceptrons respectively predict positional deformation $\varDelta \mathbf { X } \ = \ \phi _ { x } ( \mathbf { f } _ { d } )$ , rotational deformation $\varDelta \mathbf { r } = \phi _ { r } ( \mathbf { f } _ { d } )$ , and scaling deformation $\varDelta \mathbf { s } = \phi _ { s } ( \mathbf { f } d )$ using input feature vector fd.To augment reconstruction capacity while strictly preserving spatiotemporal continuity, we introduce a multi-scale manifold enhancement module that achieves hierarchical feature enhancement through synergistic optimization of an implicit nonlinear decoder and explicit deformation field. This module implements three critical operations: dynamic feature dimension adaptation aligning implicit decoder layers with deformation field outputs, cross-scale feature pyramid construction integrating low-frequency geometric structures $\mathcal { L } \mathrm { c }$ with high-frequency details $\mathcal { H } .$ and nonlinear manifold mapping to boost complex deformation modeling capability.

$$
\begin{array} { r } { \varDelta \mathbf { X } , \varDelta \mathbf { r } , \varDelta \mathbf { s } = \phi ( \mathbf { f } _ { d } ) . } \end{array}\tag{5}
$$

$$
( \mathcal { X } ^ { \prime } , r ^ { \prime } , s ^ { \prime } ) = ( \mathcal { X } + \varDelta \mathcal { X } , r + \varDelta r , s + \varDelta s ) .\tag{6}
$$

Finally, we obtain the deformed 3D Gaussians $\boldsymbol { G } ^ { \prime } = \{ x ^ { \prime } , s ^ { \prime } , r ^ { \prime } , \sigma , C \}$

## 5 Experiment

This section will first present a systematic analysis of the characteristics of the employed datasets, followed by a comparative evaluation of our methodâs performance across multiple datasets. We then conduct rigorous ablation studies to demonstrate the efficacy of the proposed approach.

GT  
TiNeuVox  
V4D  
4D-GS  
Ours  
<!-- image-->  
Fig. 4. Visualization of the HyperNeRF [26] dataset compared with other methods[21, 25, 3] . âGTâ stands for ground truth images.Please zoom in for better observation.

## 5.1 Experimental Settings

Our implementation is built upon the $\mathrm { P y }$ Torch framework, with all experiments conducted on RTX 3090 GPU. We fine-tuned the hyperparameter settings described in 4D-GS and optimized the parameters accordingly.

Synthetic Dataset. This study employs the synthetic dataset constructed by D-NeRF [27] as the primary evaluation benchmark, specifically designed for monocular dynamic scene reconstruction tasks. Its defining characteristics include quasi-randomly distributed camera poses at each temporal node, with dynamic sequence lengths per scene rigorously constrained to the 50-200 frame range.

Real-world Datasets. This study employs the HyperNeRF [26] dataset as the benchmark for real-world performance evaluation. The dataset is captured using monocular or binocular cameras, with camera motion constrained to simple linear trajectories. In our experimental pipeline, 200 frames are randomly sampled for modeling analysis, and the initial point clouds are reconstructed via Structure-from-Motion (SfM) algorithm.

Dynamic Object dataset. We also used another artificially synthesized dynamic 3D dataset, namely the Dynamic Object dataset [28]. This dataset contains 6 different 3D objects, each exhibiting unique movement patterns, including rigid or deformable motions in 3D space. Examples include Bat and Fan, among others.

GT  
<!-- image-->  
Fig. 5. Experimental results show that in the visualization results, our method can recover more details compared to other methods [2, 26, 3].

Table 1. Quantitative results on the synthetic dataset. The best and the second best results are denoted by pink and yellow. The rendering resolution is set to 800Ã800. âTimeâ in the table stands for training times.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=2>PSNR (dB) SSIM</td><td rowspan=1 colspan=1>LPIPS</td><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=1>FPS</td></tr><tr><td rowspan=1 colspan=1>TiNeuVox-B [25]</td><td rowspan=1 colspan=1>32.67</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>28 mins</td><td rowspan=1 colspan=1>1.5</td></tr><tr><td rowspan=1 colspan=1>KPlanes [20]</td><td rowspan=1 colspan=1>31.61</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>52 mins</td><td rowspan=1 colspan=1>0.97</td></tr><tr><td rowspan=1 colspan=1>HexPlane-Slim [29]</td><td rowspan=1 colspan=1>31.04</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>11m 30s</td><td rowspan=1 colspan=1>2.5</td></tr><tr><td rowspan=1 colspan=1>3D-GS [2]</td><td rowspan=1 colspan=1>23.19</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>10 mins</td><td rowspan=1 colspan=1>170</td></tr><tr><td rowspan=1 colspan=1>FFDNeRF [16]</td><td rowspan=1 colspan=1>32.68</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>&lt;1</td></tr><tr><td rowspan=1 colspan=1>MSTH [30]</td><td rowspan=1 colspan=1>31.34</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>6 mins</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>V4D [21]</td><td rowspan=1 colspan=1>33.72</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>6.9 hours</td><td rowspan=1 colspan=1>2.08</td></tr><tr><td rowspan=1 colspan=1>4D-GS [3]</td><td rowspan=1 colspan=1>34.05</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>8 mins</td><td rowspan=1 colspan=1>82</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>34.56</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>8 mins</td><td rowspan=1 colspan=1>96</td></tr></table>

## 5.2 Results

This study conducts a systematic evaluation of the experimental results based on a multi-dimensional metric system, specifically covering core evaluation criteria such as peak signal-to-noise ratio (PSNR), perceptual quality metric LPIPS, and structural similarity index (SSIM). To address the requirements for visual quality assessment in the novel view synthesis task, comparative experiments are performed with representative state-of-the-art methods in the field, including the methods proposed in literatures [25, 20, 29, 2, 16, 30, 21, 3, 31, 26]. Among these, the experimental results of other comparative methods on the synthetic dataset are directly cited from the original literature of 4D-GS. The quantitative analysis results on the synthetic dataset are detailed in Table 1. Notably, although existing dynamic hybrid representation methods can achieve relatively high-quality reconstruction effects, limitations persist in the optimization of their dynamic motion modeling components, leaving room for improvement in the detail reconstruction performance of methods such as 4D-GS. In sharp contrast, our method not only achieves optimal rendering quality on the synthetic dataset but also exhibits superior convergence efficiency while maintaining an extremely low level of storage consumption.

The experimental results on the real-scene dataset are detailed in Table 2. Notably, our method exhibits significant advantages in comprehensive performance compared to partial NeRF methods and other grid-based neural radiance field methods [25, 29, 20, 30].. Further comparison with the 4D-GS method reveals that our method demonstrates a slight leading edge in its core evaluation metrics. From an overall performance perspective, our method is on par with existing mainstream methods in terms of rendering quality, while boasting more efficient convergence characteristics and exhibiting superior real-time performance in indoor scene free-view rendering tasks.

Table 2. Quantitative results on HyperNeRF [39] vrig dataset with the rendering resolution of 960Ã540.
<table><tr><td rowspan=1 colspan=3>Model</td><td rowspan=1 colspan=1>PSNRb (dB)</td><td rowspan=1 colspan=1>MS-SSIM</td><td rowspan=1 colspan=1>Times</td><td rowspan=1 colspan=1>FPS</td></tr><tr><td rowspan=3 colspan=3>Nerfies [31]HyperNeRF [26]TiNeuVox-B [25]</td><td rowspan=1 colspan=1>22.2</td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>~ hours</td><td rowspan=1 colspan=1>&lt;1</td></tr><tr><td rowspan=1 colspan=1>22.4</td><td rowspan=1 colspan=1>0.814</td><td rowspan=1 colspan=1>32 hours</td><td rowspan=1 colspan=1>&lt;1</td></tr><tr><td rowspan=1 colspan=1>24.3</td><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>30 mins</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=3 colspan=3>3D-GS [2]FFDNeRF [16]V4D [21]</td><td rowspan=1 colspan=1>19.7</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>40 mins</td><td rowspan=1 colspan=1>55</td></tr><tr><td rowspan=1 colspan=1>RF [1</td><td rowspan=1 colspan=1>24.2</td><td rowspan=1 colspan=1>0.842</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=1>24.8</td><td rowspan=1 colspan=1>0.832</td><td rowspan=1 colspan=1>5.5 hours</td><td rowspan=1 colspan=1>0.29</td></tr><tr><td rowspan=1 colspan=3>4D-GS [3]</td><td rowspan=1 colspan=1>25.2</td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>30 mins</td><td rowspan=1 colspan=1>34</td></tr><tr><td rowspan=1 colspan=3>Ours</td><td rowspan=1 colspan=1>25.5</td><td rowspan=1 colspan=1>0.853</td><td rowspan=1 colspan=1>26 mins</td><td rowspan=1 colspan=1>28</td></tr></table>

## 5.3 Ablation experiment

Motion Saliency-Driven Dynamic Gaussian Optimization. The MS-DGO mechanism achieves adaptive gradient focus by quantifying the dynamic changes of Gaussian attributes, which can be interpreted through Ïr. Dynamically partitions scene Gaussians into two categories in real time: high-saliency Gaussians and low-saliency Gaussians. For high-saliency Gaussians, iterative optimization is performed using deformation fields; low-saliency Gaussians, upon meeting the convergence criteria of approximating the real state, halt subsequent deformation field updates. Experimental results demonstrate that by dynamically allocating computational resources and optimizing focus, this mechanism effectively accelerates model convergence and reduces storage requirements. When this module is removed, and only deformation fields are employed to continuously optimize a large number of low-saliency Gaussians, experiments observe a significant increase in storage overhead and a notable slowdown in convergence speed.

Table 3. Ablation studies conducted using our proposed method on the Dynamic Object dataset.
<table><tr><td>Model</td><td>PSNR(dB)</td><td>SSIM</td><td>LPIPS</td><td>Time</td><td>FPS</td></tr><tr><td>Ours w/o Ïr</td><td>24.429</td><td>0.950</td><td>0.058</td><td>12.5 mins</td><td>72</td></tr><tr><td>Ours w/o Ïs</td><td>24.518</td><td>0.949</td><td>0.058</td><td>14 mins</td><td>63</td></tr><tr><td>Ours</td><td>24.534</td><td>0.952</td><td>0.059</td><td>13.5 mins</td><td>66</td></tr></table>

Multi-scale Disentangled Manifold Deformation. To further enhance reconstruction quality and strictly maintain spatiotemporal continuity, this study introduces a multi-scale manifold enhancement module that can be interpreted via $\phi _ { s }$ . This module strengthens the modeling capability of complex deformation scenarios by leveraging a cooperative optimization mechanism between an

<!-- image-->  
Fig. 6. Visualization of tracking with 3D Gaussians.From top to bottom are the quantitative visualization results of Ours w/o Ïr, Ours w/o Ïs, and Ours, respectively.

Fan

Telescope

Bat

Shark

<!-- image-->  
Fig. 7. Point cloud distribution visualization.

implicit nonlinear decoder and an explicit deformation field, combined with a nonlinear manifold mapping technique. Experimental results show that removing this module leads to a significant degradation in rendering quality. Although the MS-DGO mechanism introduces a certain degree of rendering quality loss, the incorporation of this module partially offsets such impacts, thereby balancing the overall performance.

## 6 Conclusion

This paper addresses the critical challenge of balancing model convergence speed and rendering quality in 3D dynamic scene reconstruction by proposing a method named GS-DMSR. Through quantifying the dynamic changes of Gaussian attributes to achieve adaptive gradient focusing, along with synergistic optimization between an implicit nonlinear decoder and an explicit deformation field, it significantly accelerates convergence, reduces storage redundancy, and strengthens the modeling capability of complex deformation scenarios. This approach effectively achieves a balance between efficient convergence and real-time rendering.

## References

1. B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

2. B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

3. G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20310â 20320, 2024.

4. P. Wang, Y. Liu, G. Lin, J. Gu, L. Liu, T. Komura, and W. Wang, âProgressivelyconnected light field network for efficient view synthesis,â 2022.

5. B. Yang, C. Bao, J. Zeng, H. Bao, Y. Zhang, Z. Cui, and G. Zhang, âNeumesh: Learning disentangled neural mesh-based implicit field for geometry and texture editing,â 2022.

6. J. Peng, X. Chen, and J. Liu, â3dmeshnet: A three-dimensional differential neural network for structured mesh generation,â 2024.

7. L. HÃ¶llein, J. Johnson, and M. NieÃner, âStylemesh: Style transfer for indoor 3d scene reconstructions,â 2022.

8. G. Zhang, L. Fan, C. He, Z. Lei, Z. Zhang, and L. Zhang, âVoxel mamba: Group-free state space models for point cloud based 3d object detection,â 2024.

9. X. Han, Y. Tang, Z. Wang, and X. Li, âMamba3d: Enhancing local features for 3d point cloud analysis via state space model,â 2024.

10. Z. Xing, T. Ye, Y. Yang, G. Liu, and L. Zhu, âSegmamba: Long-range sequential modeling mamba for 3d medical image segmentation,â 2024.

11. Y. Liu, S. Dong, S. Wang, Y. Yin, Y. Yang, Q. Fan, and B. Chen, âSlam3r: Realtime dense scene reconstruction from monocular rgb videos,â 2025.

12. J. Sun, Y. Xie, L. Chen, X. Zhou, and H. Bao, âNeuralrecon: Real-time coherent 3d reconstruction from monocular video,â 2021.

13. M. Kocabas, J.-H. R. Chang, J. Gabriel, O. Tuzel, and A. Ranjan, âHugs: Human gaussian splats,â 2023.

14. J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â 2022.

15. J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. NieÃner, and Q. Tian, âFast dynamic radiance fields with time-aware neural voxels,â in SIGGRAPH Asia 2022 Conference Papers, SA â22, p. 1â9, ACM, Nov. 2022.

16. X. Guo, J. Sun, Y. Dai, G. Chen, X. Ye, X. Tan, E. Ding, Y. Zhang, and J. Wang, âForward flow for novel view synthesis of dynamic scenes,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16022â16033, 2023.

17. Y.-L. Liu, C. Gao, A. Meuleman, H.-Y. Tseng, A. Saraf, C. Kim, Y.-Y. Chuang, J. Kopf, and J.-B. Huang, âRobust dynamic radiance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13â23, 2023.

18. C. Gao, A. Saraf, J. Kopf, and J.-B. Huang, âDynamic view synthesis from dynamic monocular video,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5712â5721, 2021.

19. K. Zhou, J.-X. Zhong, S. Shin, K. Lu, Y. Yang, A. Markham, and N. Trigoni, âDynpoint: Dynamic neural point for view synthesis,â Advances in Neural Information Processing Systems, vol. 36, pp. 69532â69545, 2023.

20. S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, âKplanes: Explicit radiance fields in space, time, and appearance,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12479â12488, 2023.

21. W. Gan, H. Xu, Y. Huang, S. Chen, and N. Yokoya, âV4d: Voxel for 4d novel view synthesis,â IEEE Transactions on Visualization and Computer Graphics, vol. 30, no. 2, pp. 1579â1591, 2023.

22. R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, and Y. Liu, âTensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16632â16642, 2023.

23. X. Gao, J. Yang, J. Kim, S. Peng, Z. Liu, and X. Tong, âMps-nerf: Generalizable 3d human rendering from multiview images,â IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.

24. F. Wang, S. Tan, X. Li, Z. Tian, Y. Song, and H. Liu, âMixed neural voxels for fast multi-view video synthesis,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19706â19716, 2023.

25. J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. NieÃner, and Q. Tian, âFast dynamic radiance fields with time-aware neural voxels,â in SIGGRAPH Asia 2022 Conference Papers, pp. 1â9, 2022.

26. K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, and S. M. Seitz, âHypernerf: A higher-dimensional representation for topologically varying neural radiance fields,â arXiv preprint arXiv:2106.13228, 2021.

27. A. Pumarola Peris, E. Corona Puyane, G. Pons-Moll, and F. Moreno-Noguer, âDnerf: Neural radiance fields for dynamic scenes,â in Proceding of 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10313â 10322, Institute of Electrical and Electronics Engineers (IEEE), 2021.

28. J. Li, Z. Song, and B. Yang, âNvfi: Neural velocity fields for 3d physics learning from dynamic videos,â Advances in Neural Information Processing Systems, vol. 36, pp. 34723â34751, 2023.

29. A. Cao and J. Johnson, âHexplane: A fast representation for dynamic scenes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 130â141, 2023.

30. F. Wang, Z. Chen, G. Wang, Y. Song, and H. Liu, âMasked space-time hash encoding for efficient dynamic scene reconstruction,â Advances in neural information processing systems, vol. 36, pp. 70497â70510, 2023.

31. K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz, and R. Martin-Brualla, âNerfies: Deformable neural radiance fields,â in Proceedings of the IEEE/CVF international conference on computer vision, pp. 5865â5874, 2021.