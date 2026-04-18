# MULTIMODAL-PRIOR-GUIDED IMPORTANCE SAMPLING FOR HIERARCHICAL GAUSSIAN SPLATTING IN SPARSE-VIEW NOVEL VIEW SYNTHESIS

Kaiqiang Xiong1,2, Zhanke Wang1, Ronggang Wang1,2,3 \*

1Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Shenzhen Graduate School, Peking University, China 2Peng Cheng Laboratory, China 3Migu Culture Technology Co., Ltd., China

<!-- image-->  
(a) CoR-GS

<!-- image-->  
(b) NexusGS

<!-- image-->  
(c) Ours

<!-- image-->  
(d) Ground Truth  
Fig. 1: Qualitative comparison of Sparse-view Novel View Synthesis quality with the SOTA methods CoR-GS [1] and NexusGS [2] on DTU [3] dataset(3 views). Our method renders more accurate detailed textures.

## ABSTRACT

We present multimodal-prior-guided importance sampling as the central mechanism for hierarchical 3D Gaussian Splatting (3DGS) in sparse-view novel view synthesis. Our sampler fuses complementary cues â photometric rendering residuals, semantic priors, and geometric priors â to produce a robust, local recoverability estimate that directly drives where to inject fine Gaussians. Built around this sampling core, our framework comprises (1) a coarse-to-fine Gaussian representation that encodes global shape with a stable coarse layer and selectively adds fine primitives where the multimodal metric indicates recoverable detail; and (2) a geometric-aware sampling and retention policy that concentrates refinement on geometrically critical and complex regions while protecting newly added primitives in underconstrained areas from premature pruning. By prioritizing regions supported by consistent multimodal evidence rather than raw residuals alone, our method alleviates overfitting textureinduced errors and suppresses noise from appearance inconsistencies. Experiments on diverse sparse-view benchmarks demonstrate state-of-the-art reconstructions, with up to +0.3 dB PSNR on DTU.

Index Termsâ Novel view synthesis, Sparse-view rendering, 3D Gaussian Splatting

## 1. INTRODUCTION

Novel view synthesis is a central problem in computer vision with broad applications in virtual/augmented reality, robotics, and content creation. While 3D Gaussian Splatting (3DGS) [4] provides high-fidelity, real-time rendering with dense multi-view inputs, its performance deteriorates under sparse-view conditions because (i) geometric supervision becomes spatially sparse and uneven, and (ii) the default densification-and-pruning strategy blindly scatters Gaussians, wasting capacity on well-observed surfaces while under-fitting thin structures, object boundaries and texture-rich regions that are essential for photo-realism. Consequently, a key question arises: how can we allocate the limited budget of Gaussians to locations where fine detail is actually recoverable?

Prior efforts to mitigate sparse-view failures include depth-based regularization [5â7], multi-field regularization [1] , dropout-style regularizers [8, 9], and approaches that leverage pretrained models to recover dense correspondences or initialize dense point clouds [2]. These strategies improve robustness, yet they either impose spatially-uniform constraints or rely on heuristics that do not tell where additional geometry can be reliably recovered.

In contrast, we propose a hierarchical, geometric-aware 3DGS pipeline driven by multimodal-prior-guided importance sampling. The sampler fuses complementary cuesâphotometric rendering residuals, semantic priors and geometric priors (e.g., depth and normals) â to produce a local recoverability score that directly determines where to propose fine Gaussians. Built around this sampling core, our framework comprises three key elements: (1) a multimodal importance metric that discriminates true geometric edges from high-frequency appearance or noise, avoiding the overfitting pitfalls of residual-only strategies; and (2) a coarse-to-fine Gaussian representation in which a stable coarse layer encodes global shape and fine Gaussians are injected selectively where multimodal evidence indicates recoverable detail; and (3) a geometric-aware sampling and retention policy that focuses refinement on geometrically critical and complex regions while protecting newly added primitives in underconstrained areas from premature pruning until sufficient geometric evidence accumulates. Together, these components concentrate modeling capacity where geometry is actually recoverable, preserve sharp boundaries and structural detail, and stabilize optimization under sparse supervision, as shown in Fig. 1.

<!-- image-->  
Fig. 2: Framework. Our hierarchical Gaussian Splatting framework with Multimodal-Prior-Guided importance sampling for sparse-view novel view synthesis. It consists of three main components: (a) a Hierarchical Gaussian representation with coarse and fine levels (in Sec. 2.2), (b) a Multi-Modal Importance Assessment module (in Sec. 2.3), and (c) a Geometric-aware Sampling and pruning strategy (in Sec. 2.4).

We summarize our contributions as follows:

â¢ A multimodal-prior-guided importance metric that fuses photometric, geometric, and semantic signals to localize where fine Gaussians should be allocated.

â¢ A hierarchical 3D Gaussian Splatting framework for sparseview novel view synthesis that stabilizes optimization via a coarse-to-fine representation driven by multimodal importance estimates.

â¢ A geometric-aware sampling and pruning strategy that concentrates resources on geometrically critical regions and prevents premature removal of newly added primitives in underconstrained areas.

## 2. METHOD

We center our design on multimodal-prior-guided importance sampling to address sparse-view novel view synthesis within a hierarchical Gaussian Splatting framework. Fig. 2 depicts the pipeline, whose three main components are: (1) a hierarchical Gaussian representation ( Sec. 2.2) preserves both coarse global shape and selectively added fine detail; (2) a multimodal importance assessment module (Sec. 2.3) that computes a local recoverability score by fusing photometric residuals with semantic cues and geometric priors (e.g., depth, normals); and (3) a geometric-aware sampling strategy ( Sec. 2.4)that uses the multimodal score to propose, place, and retain fine Gaussians where they are most likely to improve geometry. By making the multimodal sampler the driving mechanism, our method explicitly distinguishes true geometric edges and underconstrained regions from high-frequency appearance or noise, thereby guiding refinement to regions where added primitives yield reliable gains and avoiding wasted or harmful densification.

Given sparse input views $\{ I _ { i } \} _ { i = } ^ { N }$ 1 with corresponding camera poses $\{ P _ { i } \} _ { i = 1 } ^ { \hat { N } } ,$ , our goal is to optimize a 3D scene representation that enables high-quality novel view synthesis. Unlike dense-view scenarios where uniform densification suffices, sparse-view reconstruction must account for spatially varying geometric reliability and selectively allocate modeling capacity accordingly.

## 2.1. Preliminaries

3D Gaussians Splatting [4] is an explicit scene representation that supports high-fidelity real-time renderings. It represents the 3D scene as a set of 3D Gaussians $G s = \{ G _ { j } | j \in \bar { \{ 1 , . . . , M \} } \}$ and renders images via splatting: ${ \cal I } _ { i } ^ { r } \ = \ \Psi ( G s , P _ { i } , K _ { i } )$ , where $I _ { i } ^ { r }$ is the rendered image and Pi, Ki are the corresponding camera extrinsics and intrinsic. Each 3D Gaussian $G _ { j }$ is characterized as $\{ \mu , \Sigma , \alpha , F \}$ . Âµ is the mean of 3D Gaussian distribution. Î£ is the 3D covariance matrix . Besides, Î±, F are opacity and spherical harmonics coefficients for rendering. During rendering, Gaussians are projected to the 2D plane via viewing transformation W and the Jacobian of the affine approximation J of the projection transformation: $\Sigma ^ { \prime } = J W \Sigma W ^ { \hat { T } } \bar { J } ^ { T }$ . The final color $\bar { C }$ of each pixel is acquired in an alpha-blending way according to the depth order of the O overlapping Gaussians. A tile-based rasterizer is used for efficient rendering. During optimization, the Gaussian parameters are updated under the reconstruction loss and SSIM loss [10] between rendered and ground-truth images: $L = ( 1 - \lambda ) L _ { 1 } + \lambda L _ { \mathrm { S S I M } }$ , where Î» is the weighting parameter. More details can be found in the [4].

<!-- image-->  
(a) CoR-GS  
(b) NexusGS  
(c) Ours  
(d) Ground Truth  
Fig. 3: Qualitative results on LLFF [11] and DTU [3] Datasets.

## 2.2. Hierarchical Gaussian Representation

We introduce a two-level hierarchical structure to balance global shape stability and local detail adaptivity under sparse-view constraints:

Coarse Level Gaussians $( \mathcal { G } _ { c } ) \colon$ These Gaussians establish global geometric consistency and provide a stable foundation for the scene structure. They are initialized using the method proposed in [2] and remain relatively stable throughout training $\begin{array} { r l } { \cdot { \mathcal { G } } _ { c } } & { { } = } \end{array}$ $\{ ( \mu _ { i } ^ { c } , \Sigma _ { i } ^ { c } , \alpha _ { i } ^ { c } , c _ { i } ^ { c } ) \} _ { i = 1 } ^ { M _ { c } }$ where $\mu _ { i } ^ { c } , \Sigma _ { i } ^ { c } , \alpha _ { i } ^ { c }$ , and $c _ { i } ^ { c }$ represent the position, covariance, opacity, and color of the i-th coarse Gaussian, respectively.

Fine Level Gaussians $( { \mathcal { G } } _ { f } ) \colon$ These Gaussians capture detailed geometric features and are adaptively placed based on the propose multimodal importance sampling. They undergo dynamic densification and pruning during training:

$$
\mathcal { G } _ { f } = \{ ( \mu _ { j } ^ { f } , \Sigma _ { j } ^ { f } , \alpha _ { j } ^ { f } , c _ { j } ^ { f } ) \} _ { j = 1 } ^ { M _ { f } }\tag{1}
$$

The final rendering combines contributions from both levels: $I _ { \mathrm { r e n d e r } } = R ( \mathcal { G } _ { c } \cup \mathcal { G } _ { f } , P )$ where $R ( \cdot , P )$ denotes the Gaussian splatting rendering function for camera pose $P .$

## 2.3. Multi-Modal Importance Assessment

To address the limitations of single-criterion sampling, we design a multi-modal importance assessment that integrates three complementary signals:

Rendering Residual $( S _ { \mathbf { r e n d e r } } ) { \mathrm { : } }$ Measures reconstruction error at each pixel:

$$
S _ { \mathrm { r e n d e r } } ( x , y ) = \| I _ { \mathrm { g t } } ( x , y ) - I _ { \mathrm { r e n d e r } } ( x , y ) \| _ { 2 }\tag{2}
$$

Semantic Prior $( S _ { \mathrm { s e m a n t i c } } ) { \mathrm { : } }$ Leverages a lightweight semantic segmentation network to identify object boundaries and semantically important regions:

$$
S _ { \mathrm { s e m a n t i c } } ( x , y ) = \mathcal { F } _ { \mathrm { s e g } } ( I _ { \mathrm { r e f } } ) ( x , y ) \cdot ( \omega _ { \mathrm { b o u n d a r y } } ( x , y ) + \omega _ { \mathrm { f o r e g r o u n d } } ( x , y ) )\tag{3}
$$

where $\mathcal { F } _ { \mathrm { s e g } }$ is a ResNet18-based [16] segmentation network pretrained with 21 semantic classes, Ïboundary and Ïforeground enhances object boundary regions and foreground regions.

Geometric Complexity $( S _ { \mathrm { g e o m e t r y } } ) { : }$ Evaluates local geometric variation with depth gradients:

$$
S _ { \mathrm { g e o m e t r y } } ( x , y ) = \| \nabla D ( x , y ) \| _ { 2 } + \lambda _ { \mathrm { c u r v } } \kappa ( x , y )\tag{4}
$$

where $D ( x , y )$ is the the monocular depth estimated with the Dense Prediction Transformer (DPT) [17] and $\kappa ( x , y )$ represents surface curvature estimated with the second-order gradient of the depth.

The final importance score combines these signals: $S _ { \mathrm { i m p o r t a n c e } } ( x , y )$ $\mathbf { \Sigma } = \mathbf { w } ^ { T } \mathbf { s } ( x , y )$ , where $\mathbf { w } = [ w _ { 1 } , w _ { 2 } , w _ { 3 } ] ^ { T }$ are the weighting coefficients and $\begin{array} { r c l } { \mathbf { s } ( x , y ) } & { = } & { [ \dot { S _ { \mathrm { r e n d e r } } } , S _ { \mathrm { s e m a n i c } } , S _ { \mathrm { g e o m e t r y } } ] ^ { T } } \end{array}$ contains the individual importance scores.

## 2.4. Geometric-Aware Sampling

To ensure robust training, our sampling strategy targets regions with strong geometric constraints while avoiding poorly-constrained areas:

Reliability Assessment: To ensure robust training, we identify wellconstrained regions: $M _ { \mathrm { r e l i a b l e } } ( x , y ) ~ = ~ I _ { g } ( x , y )$ , where $I _ { g } ( x , y ) =$ $\mathbb { 1 } \left[ S _ { \mathrm { g e o m e t r y } } ( x , y ) > \tau _ { \mathrm { g e o m e t r y } } \right]$ is indicator functions for geometry constraints.

Adaptive Gaussian Placement: New Gaussians are placed probabilistically based on the importance score, but only in reliable regions: $\begin{array} { r } { P _ { \mathrm { s a m p l e } } ( x , y ) = \frac { \bar { S _ { \mathrm { i m p o r t a n c e } } } ( x , y ) \cdot M _ { \mathrm { r e l i a b l e } } ( x , y ) } { \sum _ { ( i , j ) } S _ { \mathrm { i m p o r t a n c e } } ( i , j ) \cdot M _ { \mathrm { r e l i a b l e } } ( i , j ) } } \end{array}$ . The probabilistic sampling approach prevents over-concentration in highscoring regions while maintaining exploration of moderately important areas. It avoids local optima in Gaussian placement and ensures better spatial coverage compared to deterministic top-k selection, leading to more robust scene representation. For a sampled pixel $( u , v )$ , we back-project it to 3D space to sample fine gaussian using the rendering depth d and camera intrinsics/extrinsics:

$$
{ \bf p } _ { w } = R ^ { - 1 } \left( d , K ^ { - 1 } \left[ u v 1 \right] ^ { T } - t \right) ,\tag{5}
$$

where $K , R , t$ are the camera intrinsic matrix, rotation matrix, and translation vector, respectively. The covariance Î£ is initialized as an isotropic Gaussian with a predefined scale.

Protection Mechanism: To prevent premature pruning under sparse supervision, newly added Gaussians are protected for $T _ { \mathrm { p r o t e c t } }$ iterations: $\alpha _ { \mathrm { p r o t e c t e d } } ~ = ~ \mathrm { m a x } ( \alpha _ { \mathrm { o r i g i n a l } } , \alpha _ { \mathrm { m i n i m u m } } )$ for $t \ < \ T _ { \mathrm { p r o t e c t } }$ This protection mechanism prevents premature removal of newly added Gaussians, which may initially appear suboptimal but possess significant representational potential. By maintaining a minimum opacity threshold during the protection period, we ensure adequate optimization time for new primitives to demonstrate their value, particularly crucial under sparse supervision where individual Gaussians may only contribute meaningfully to a subset of views initially.

## 2.5. Training Strategy

Our training procedure alternates between Gaussian optimization and Gaussian sampling:

Phase 1 - Coarse Initialization: Initialize coarse Gaussians from point clouds and optimize for $N _ { \mathrm { c o a r s e } }$ iterations to establish basic scene geometry. Phase 2 - Hierarchical Refinement: Iteratively add fine Gaussians based on multimodal importance sampling every $T _ { \mathrm { s a m p l e } }$ iterations. The sampling frequency decreases over training: $T _ { \mathrm { s a m p l e } } ( t ) = T _ { \mathrm { b a s e } } \cdot ( 1 + \gamma \cdot t )$ . Phase 3 - Stabilization: In the final training phase, we freeze the Gaussian placement and focus on optimizing parameters for stable convergence.

Table 1: Quantitative comparison on LLFF [11] , DTU [3], and MipNeRF-360 [12] datasets. We color each cell as best , second best .
<table><tr><td rowspan="2">Method</td><td colspan="3">LLFF (3 Views)</td><td colspan="3">DTU (3 Views)</td><td colspan="3">MipNeRF-360 (24 Views)</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>DietNeRF [13]</td><td>14.94</td><td>0.370</td><td>0.496</td><td>11.85</td><td>0.633</td><td>0.314</td><td>20.21</td><td>0.557</td><td>0.387</td></tr><tr><td>reRNeRF [14</td><td>19.63</td><td>0.612</td><td>0.308</td><td>19.92</td><td>0.787</td><td>0.182</td><td>22.78</td><td>0.689</td><td>0.323</td></tr><tr><td>SparseNeRrF [15]</td><td>19.86</td><td>0.624</td><td>0.328</td><td>19.55</td><td>0.769</td><td>0.201</td><td>22.85</td><td>0.693</td><td>0.315</td></tr><tr><td>3DGS [4]</td><td>18.54</td><td>0.588</td><td>0.272</td><td>17.65</td><td>0.816</td><td>0.146</td><td>21.71</td><td>0.672</td><td>0.248</td></tr><tr><td>DNGaussian [6]</td><td>19.12</td><td>0.591</td><td>0.294</td><td>18.91</td><td>0.790</td><td>00.176</td><td>18.06</td><td>0.423</td><td>0.584</td></tr><tr><td>FSGS [7]</td><td>20.43</td><td>0.682</td><td>0.248</td><td>17.14</td><td>0.818</td><td>0.162</td><td>23.40</td><td>0.33</td><td>0.238</td></tr><tr><td>Co-GS [1]</td><td>20.45</td><td>0.712</td><td>0.196</td><td>19.21</td><td>0.853</td><td>0.119</td><td>23.55</td><td>0.727</td><td>0.226</td></tr><tr><td>NexusGS [2]</td><td>21.07</td><td>0.738</td><td>0.177</td><td>20.21</td><td>0.869</td><td>0.102</td><td>23.86</td><td>0.753</td><td>0.206</td></tr><tr><td>Ours</td><td>21.17</td><td>0.746</td><td>0.175</td><td>20.51</td><td>0.872</td><td>0.104</td><td>23.88</td><td>0.754</td><td>0.208</td></tr></table>

This hierarchical framework with multimodal importance sampling enables robust sparse-view novel view synthesis by intelligently allocating computational resources where they are most beneficial while maintaining geometric consistency across challenging viewing conditions. During training, we employ the same loss as 3DGS[4].

Table 2: Ablation study on DTU [3] with 3 training views.
<table><tr><td>Hier</td><td> $S _ { r e n d }$ </td><td> $S _ { s e m }$ </td><td> $S _ { g e o }$ </td><td>RA</td><td>AGP</td><td>PM</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Ã</td><td></td><td></td><td></td><td></td><td></td><td>&gt;&gt;&gt;&gt;&gt;</td><td>20.35</td><td>0.870</td><td>0.103</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.32</td><td>0.870</td><td>0.104</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.41</td><td>0.871</td><td>0.101</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.43</td><td>0.871</td><td>0.104</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.30</td><td>0.869</td><td>0.103</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.36</td><td>0.872</td><td>0.104</td></tr><tr><td>&gt;&gt;&gt;&gt;&gt;&gt;</td><td>&gt;Ã&gt;&gt;&gt;&gt;&gt;</td><td>&gt;Ã&gt;&gt;&gt;</td><td>&gt;&gt;&gt;Ã&gt;&gt;&gt;&gt;</td><td>&gt;&gt;&gt;&gt;x&gt;&gt;&gt;</td><td>&gt;&gt;&gt;&gt;&gt;Ã&gt;&gt;</td><td>&gt;Ã</td><td>20.25</td><td>0.869</td><td>0.102</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>â</td><td>20.51</td><td>0.872</td><td>0.104</td></tr></table>

## 3. EXPERIMENTS

We conduct comprehensive experiments to evaluate our hierarchical 3D Gaussian Splatting framework on sparse-view novel view synthesis tasks. Our evaluation demonstrates superior performance compared to existing methods across multiple datasets and metrics.

## 3.1. Experimental Setup

Datasets & Implementation Details. We evaluate on three standard benchmarks for novel view synthesis: (1) LLFF [11] containing 8 forward-facing real scenes, (2) DTU [3] with 15 object-centric scenes under controlled lighting, and (3) Mip-NeRF360 [12] synthetic dataset with 9 scenes featuring complex materials and lighting. For sparse-view evaluation, we use 3 training views for DTU and LLFF, and24 views for Mip-NeRF360 following prior work [2, 7].Our framework is implemented in PyTorch with CUDA acceleration. The hierarchical training consists of three phases: coarse initialization (2K iterations), hierarchical refinement (25K iterations for LLFF, 5K iterations for DTU and Mip-NeRF360), and stabilization (3K iterations). Multimodal-prior-guided importance sampling weights are set as w1 = 0.4, w2 = 0.2, w3 = 0.4 based on validation performance.

Baselines & Evaluation Metrics. We compare against NeRF methods including: DietNeRF [13], FreeNeRF [14], SparseNeRF [15], and the 3D Gaussian Splatting methods including: 3DGS [4], FSGS [7], DNGaussian [6], CoR-GS [1] and NexusGS [2]. All methods are trained using identical sparse view configurations for fair comparison. We report standard image quality metrics: PSNR, SSIM [10], and LPIPS [18]. Higher PSNR and SSIM indicate better quality, while lower LPIPS suggests better perceptual similarity.

## 3.2. Quantitative Results

Tab. 1 presents quantitative comparisons on all three datasets. Our method consistently outperforms existing approaches across different sparse-view settings. On LLFF, our method achieves 21.17 dB PSNR with 3 views, outperforming the best baseline by 0.1 dB. The improvement is even more significant on DTU with 3 views, where we achieve 0.3 dB versus SOTA method NexusGS [2]. These results demonstrate the effectiveness of our hierarchical framework in handling severely under-constrained scenarios.

## 3.3. Ablation Studies

We conduct ablation studies to analyze the contribution of each component in our framework, as shown in Tab. 2, where Hier refers to the hierarchical setting, $S _ { r e n d } , S _ { s e m } , S _ { g e o }$ refers to the three importance metrics in Sec. 2.3. RA, AGP , P M refers to the Reliability Assessment, Adaptive Gaussian Placement and Protection Mechanism in Sec. 2.4. The results show that all components of the multimodal importance assessment contribute to the final outcome, and the reliability assessment ensures robust training. Without adaptive Gaussian placement, the Gaussians tend to over-concentrate on certain regions, leading to degraded performance. Likewise, without the protection mechanism, most newly added fine Gaussians would be pruned before they can take effect. The full model achieves the best performance, confirming the complementary benefits of each design choice.

## 3.4. Qualitative Results

Fig. 3 presents visual comparisons on challenging scenes. Our method produces sharper details and more consistent geometry compared to baselines, particularly in regions with limited view coverage. The improvements are most visible in: (1) fine geometric details like texture patterns, as shown in the first row, (2) reduced artifacts in under-constrained regions, as shown in the second row. These qualitative results align with our quantitative metrics.

## 4. CONCLUSION

We present a hierarchical framework with multimodal-prior-guided importance sampling to address sparse-view challenges in 3D Gaussian Splatting. Our method introduces hierarchical gaussians, multimodal importance assessment, and reliability-aware masking to improve geometric supervision and Gaussian placement. Experiments demonstrate significant improvements over existing methods. The framework improve rendering quality, enabling practical applications in mobile AR/VR and rapid prototyping. This work provides a foundation for sparse-view novel view synthesis, demonstrating the effectiveness of integrating multimodal-prior-guided importance sampling with hierarchical gaussians.

## References

[1] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai, âCoR-GS: Sparse-view 3D Gaussian splatting via co-regularization,â in Proceedings of the European Conference on Computer Vision. Springer, 2024, pp. 335â 352.

[2] Yulong Zheng, Zicheng Jiang, Shengfeng He, Yandu Sun, Junyu Dong, Huaidong Zhang, and Yong Du, âNexusGS: Sparse view synthesis with epipolar depth priors in 3D Gaussian splatting,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26800â26809.

[3] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik AanÃ¦s, âLarge scale multi-view stereopsis evaluation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2014, pp. 406â413.

[4] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis, â3D Gaussian Splatting for Real-Time Radiance Field Rendering,â ACM Transactions on Graphics (TOG), vol. 42, no. 4, pp. 1â14, 2023.

[5] Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee, âDepthregularized optimization for 3D Gaussian splatting in few-shot images,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 811â820.

[6] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu, âDngaussian: Optimizing sparse-view 3D Gaussian radiance fields with global-local depth normalization,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20775â20785.

[7] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang, âFSGS: Real-time few-shot view synthesis using Gaussian splatting,â in Proceedings of the European Conference on Computer Vision. Springer, 2024, pp. 145â163.

[8] Hyunwoo Park, Gun Ryu, and Wonjun Kim, âDropGaussian: Structural regularization for sparse-view Gaussian splatting,â in Proceedings of the Computer Vision and Pattern Recognition, 2025, pp. 21600â21609.

[9] Yexing Xu, Longguang Wang, Minglin Chen, Sheng Ao, Li Li, and Yulan Guo, âDropoutGS: Dropping out Gaussians for better sparse-view rendering,â in Proceedings of the Computer Vision and Pattern Recognition, 2025, pp. 701â710.

[10] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600â612, 2004.

[11] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar, âLocal light field fusion: Practical view synthesis with prescriptive sampling guidelines,â ACM Transactions on Graphics (ToG), vol. 38, no. 4, pp. 1â14, 2019.

[12] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman, âMip-NeRF 360: Unbounded anti-aliased neural radiance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5470â5479.

[13] Ajay Jain, Matthew Tancik, and Pieter Abbeel, âPutting NeRF on a diet: Semantically consistent few-shot view synthesis,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5885â5894.

[14] Jiawei Yang, Marco Pavone, and Yue Wang, âFreeNeRF: Improving few-shot neural rendering with free frequency regularization,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8254â8263.

[15] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu, âSparseNeRF: Distilling depth ranking for few-shot novel view synthesis,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9065â9076.

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, âDeep residual learning for image recognition,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2016, pp. 770â778.

[17] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun, âVi-Â´ sion transformers for dense prediction,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 12179â12188.

[18] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 586â595.