# Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction

Binxiao Huang, Zhengwu Liu, Ngai Wong

The University of Hong Kong

## Abstract

3D Gaussian Splatting (GS) has emerged as a powerful representation for high-quality scene reconstruction, offering compelling rendering quality. However, the training process of GS often suffers from slow convergence due to inefficient densification and suboptimal spatial distribution of Gaussian primitives. In this work, we present a comprehensive analysis of the split and clone operations during the densification phase, revealing their distinct roles in balancing detail preservation and computational efficiency. Building upon this analysis, we propose a global-to-local densification strategy, which facilitates more efficient growth of Gaussians across the scene space, promoting both global coverage and local refinement. To cooperate with the proposed densification strategy and promote sufficient diffusion of Gaussian primitives in space, we introduce an energy-guided coarseto-fine multi-resolution training framework, which gradually increases resolution based on energy density in 2D images. Additionally, we dynamically prune unnecessary Gaussian primitives to speed up the training. Extensive experiments on MipNeRF-360, Deep Blending, and Tanks & Temples datasets demonstrate that our approach significantly accelerates trainingâachieving over 2x speedup with fewer Gaussian primitives and superior reconstruction performance.

## Introduction

Reconstructing high-quality 3D representations from unordered image collections remains a fundamental challenge in computer vision and graphics. Neural radiance fields (NeRF) (Mildenhall et al. 2020) have revolutionized this domain through their implicit scene representation paradigm, combining deep learning with volumetric rendering to achieve unprecedented view synthesis quality (Oechsle, Peng, and Geiger 2021; Park et al. 2021; Wang et al. 2021; Yariv et al. 2021). Despite these successes, the computational demands of ray-based volumetric rendering present critical limitations. The requirement for dense spatial sampling along viewing rays significantly hinders both training convergence and rendering efficiency. Recent advancements in 3D reconstruction have highlighted 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) as a promising technique for high-fidelity scene modeling. By representing scenes as collections of anisotropic Gaussian primitives, plenty of works (Waczynska et al. 2024; Yan et al. 2024; Yang et al. 2024) based on the 3DGS achieve impressive visual quality with explicit geometric distribution and efficient rendering pipelines. However, there still exists an urgent requirement for computational efficiency improvements to deploy 3D Gaussian Splatting on resource-constrained devices or enable its practical application in real-time reconstruction and dynamic modeling scenarios where training time constitutes a critical bottleneck (Cong et al. 2025; Javed et al. 2024; Tan et al. 2024).

Based on the 3DGS pipeline,several recent approaches(Chen et al. 2025; Fang and Wang 2024a; Hanson et al. 2024; Mallick et al. 2024) have pursued optimization acceleration from the perspectives of geometric distribution, optimizer, and multi-resolution and so on. Taming 3DGS (Mallick et al. 2024) makes each tile uses a parallelization scheme over the 2D splats instead of pixels. Mini-splatting (Fang and Wang 2024a) utilizes depth to achieve efficient reinitialization of the Gaussian primitives from the perspective of spatial geometry. EDC (Deng et al. 2024) proposes a long-axis split operation and a pruning strategy to efficiently control the Gaussian densification. DashGaussian (Chen et al. 2025) introduce a resolution scheduler and a primitive scheduler to accelerate the training time.

In this work, we systematically analyze the bottlenecks in 3D Gaussian Splatting reconstruction, particularly focusing on inefficient spatial spread and redundant Gaussian primitives during optimization. We reveal that the split operation takes charge of the global spread while the clone operation governs the local refinement (cf. Table 1 and Fig. 3). Then We identify clone operations in the early stage as the primary cause of excessive Gaussian clustering during optimization, where redundant primitives aggregate while contributing minimally to reconstruction fidelity (cf. Fig. 2). To address these limitations, we propose a global-to-local strategy that decouples split and clone across densification phases.

We further design a energy-aware multi-resolution training strategy to facilitate this global-to-local optimization strategy. Specifically, we promote Gaussian primitivesâ global spread with split operation at the lower resolution and suppress clone operations. This prevents premature local clustering and ensures efficient scene coverage. Only when transitioning to full-resolution training do we reintroduce clone operations to refine high-frequency details. Additionally, we integrates an opacity pruning strategy with a adaptive threshold to remove the unnecessary Gaussian primitives. The pipeline is shown in Fig. 1. By jointly utilizing the proposed approaches, our method achieves an approximately 2Ã acceleration in training speed compared to baseline implementations, with a comparable or even better reconstruction quality. In summary, our contributions are as follows:

â¢ We first reveal the split takes charge of the global spread and the clone governs the local refinement and propose a global-to-fine densification to accelerate the optimization.

â¢ We introduce a energy-aware multi-resolution framework to promote the global-to-fine densification for further acceleration.

â¢ Comprehensive experiments conducted on three datasets demonstrate that our method achieves an approximately 2Ã speedup over the baseline, while maintaining or even enhancing the performance.

## Related Works

## 3D Gaussian Splatting

3D Gaussian Splatting (Kerbl et al. 2023) has emerged as a compelling approach for 3D scene reconstruction, enabling real-time rendering while preserving photorealistic quality. Unlike implicit neural fields (e.g., NeRF (Mildenhall et al. 2020)) that rely on computationally intensive ray marching for volume rendering, 3DGS formulates scenes as collections of anisotropic Gaussian primitives with full covariance matrices. This explicit representation allows efficient tilebased rasterization through differentiable projection and alpha blending, bypassing the limitations of neural rendering pipelines.

In recent years, there has been a surge in research efforts that have pushed forward the technological frontiers of 3DGS across multiple domains, with particularly transformative impacts on human avatar generation (Cha, Lee, and Joo 2024; Jiang et al. 2024; Lyu et al. 2024; Zielonka et al. 2025), Autonomous Driving (Chen et al. 2024; Hess et al. 2024; Lei et al. 2025; Zhou et al. 2024), and photorealistic scene renderings (Chao et al. 2024; Cheng et al. 2024; Xie et al. 2024; Xu, Mei, and Patel 2024).

## Acceleration for 3DGS Optimization

Although the rendering speed of 3D Gaussian splatting is much faster than that of NeRF, it still takes tens of minutes to complete the rendering of a scene on a high-performance GPU. Plenty of subsequent works accelerated the optimization process from the perspectives of optimization strategies, the number of Gaussian spheres, etc. Taming 3DGS (Mallick et al. 2024) reformulates the original per-pixel parallelization into per-splat parallel backpropagation, significantly accelerating the optimization process of 3D Gaussian Splatting and establishing a strong baseline for following research. Mini-Splatting (Fang and Wang 2024b) saves the training time and memory cost by maintaining the most important primitive for each pixel through depth reinitialization. Speedy-Splat (Hanson et al. 2024) calculates a precise tile allocation of Gaussians when projected to the 2D image planes and prunes a fixed high proportion of Gaussians in specific iterations. Meanwhile, reducing the resolution of renderings in the optimization stage is also a promising option for acceleration. EAGLES (Girish, Gupta, and Shrivastava 2023) adopts several schedules for gradually increasing the resolution empirically. From the perspective of the frequency domain, DashGaussian (Chen et al. 2025) designs a resolution scheduler and a primitive scheduler to efficiently reconstruct the scene from low frequency to high frequency.

However, these methods adopt the default densification strategy and do not explore the actual roles of split and clone. A similar work EDC (Deng et al. 2024) replaced the clone operation with a proposed long-axis split based on AbsGS (Ye et al. 2024) with limited improvement of training speed. In contrast, our methods analyze the behaviors of the split and clone operations and propose a global-to-local densification strategy to facilitates efficient growth of Gaussians across the scene. Then we design an energy-guided coarse-to-fine multi-resolution training framework to cooperate with the proposed densification strategy.

## Preliminary

3D Gaussian Splatting 3DGS (Kerbl et al. 2023) represents 3D scenes through anisotropic Gaussian primitives and demonstrates state-of-the-art performance in both visual quality and rendering efficiency. Each Gaussian primitive $\bar { \boldsymbol { g } } _ { i }$ is formally defined by five core contributions: spatial position $\mathbf { } \mathbf { } u _ { i } \in \dot { \mathbb { R } } ^ { 3 }$ , opacity $\alpha _ { i }$ , orthogonal rotation matrix $\pmb { R } _ { i } \doteq \mathbb { R } ^ { 3 \times 3 }$ , diagonal scaling matrix $\boldsymbol { S } _ { i } ^ { \top } \in \mathbb { R } ^ { 3 \times 3 }$ , and spherical harmonics (SH) coefficients for view-dependent color representation. The Gaussian distribution is mathematically expressed as:

$$
\mathcal { G } _ { i } ( \pmb { x } ) = \exp \left( - \frac { 1 } { 2 } ( \pmb { x } - \pmb { u } _ { i } ) ^ { T } \Sigma _ { i } ^ { - 1 } ( \pmb { x } - \pmb { u } _ { i } ) \right) ,\tag{1}
$$

where $\begin{array} { r l r } { \Sigma _ { i } } & { { } = } & { R _ { i } S _ { i } S _ { i } ^ { T } R _ { i } ^ { T } } \end{array}$ ensures positive semidefiniteness. For real-time rendering, Gaussian primitives are projected onto the 2D image plane with the Jacobian affine approximation. Given camera extrinsic parameters W and projection matrix Jacobian J, the 2D covariance in screen space becomes $\Sigma _ { i } ^ { \prime } = J W \Sigma _ { i } W ^ { T } J ^ { T }$ . The final pixel color $\bar { \mathcal { C } ( \pmb { p } ) }$ is computed via alpha compositing of depthsorted Gaussians:

$$
\pmb { C } ( \pmb { p } ) = \sum _ { i \in \mathcal { N } _ { p } } \pmb { c } _ { i } \sigma _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma _ { j } ) , \quad \sigma _ { i } = \alpha _ { i } \mathcal { G } _ { i } ^ { \prime } ( \pmb { p } ) ,\tag{2}
$$

where $c _ { i }$ denotes SH-evaluated color and $\mathcal { N } _ { p }$ indexes visible Gaussians at pixel p. The model is optimized using a hybrid loss combining $\mathcal { L } _ { 1 }$ and structural similarity (i.e., D-SSIM term):

$$
\mathcal { L } _ { \mathrm { t o t a l } } = ( 1 - \lambda ) \| \pmb { I } - \hat { \pmb { I } } \| _ { 1 } + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } ( \pmb { I } , \hat { \pmb { I } } ) ,\tag{3}
$$

with default weight $\lambda = 0 . 2 .$ , where I and $\hat { \boldsymbol { I } }$ denote ground truth and rendered images, respectively.

<!-- image-->  
Figure 1: Pipeline of the global-to-local and coarse-to-fine densification.

During the densification stage, the norm of the average position gradient of each Gaussian primitive is calculated every 100 iterations. If the gradient norm exceeds a predefined threshold, the corresponding Gaussian primitive will either be split or cloned. Specifically, if the maximum scale of the Gaussian exceeds a given scale threshold, it will be split into smaller components; otherwise, it is simply cloned with the same parameters.

## Methodology

In this section, we present how the proposed method reduces optimization complexity to accelerate 3D Gaussian splatting, while preserving rendering quality without compromise. In Sec. , we analyze the distinct behaviors of the split and clone operations and proposes a global-to-local densification strategy to improve optimization efficiency. In Sec. , we introduce a coarse-to-fine multi-resolution scheme based on the energy density in 2D images to better support the proposed densification strategy. In Sec., we adopt an adaptive opacity threshold to better balance the trade-off between training efficiency and rendering quality.

## Global-to-local densification

To achieve the 3D reconstruction using the Gaussian Splatting, split and clone operations are applied simultaneously during the densification stage to densify and spread the Gaussian primitives spatially. We dig into the differences between the these two operations and claims two statements neglected by preview researches: 1. The split operation takes charge of the diffusion of the Gaussian primitives in space (cf. Sec. ); 2. The number of Gaussian primitives produced by clone is much higher than that produced by split (cf. Sec. ).

Spatial diffusion In this section, we prove that the spatial diffusion is predominantly governed by the split operation, whereas the clone operation contributes to local feature refinement. We first revealed that the clone operation is the cause of the cluster phenomenon observed in Mini-Splatting (Fang and Wang 2024a), which can be alleviated by our methods.

We regard the points extracted by structure from motion (SFM) for initialization as the parent points. Each newly generated Gaussian maintains a one-to-one correspondence with its parent point. During the optimization, We equip each Gaussian primitive with three more attributes: original position, split count, and clone count. The original position stores the initial coordinates of its parent point and the split/clone count quantify cumulative split/clone operations executed with respect to its parent point during optimization. After optimization, we classify Gaussians into three categories: split-dominated (split Â¿ clone), clonedominated (clone Â¿ split), and equal (split = clone). Specifically, Gaussian $\mathcal { G } _ { A }$ undergoes split to produce $\mathcal { G } _ { B } .$ , and $\mathcal { G } _ { B }$ clones to produce $\mathcal { G } _ { C }$ . Consequently, the parent point of $\mathcal { G } _ { C }$ is $\mathcal { G } _ { A }$ and $\mathcal { G } _ { C }$ belongs to the equal category. We compute average Euclidean distances between final positions and initial positions of each gaussian for each category. As shown in Table 1, the average displacement distances across three datasets show that split-dominated Gaussians exhibit displacements around twenty times greater than clone-dominated countparts, demonstrating that spatial expansion is primarily driven by split operations. The camera extent is 1.1 times the radius of the smallest sphere covering all camera positions defined in the 3D Gaussian splatting. It is regarded as a measurement representing the size of the scene. The displacement of clone-dominated primitives is less than 2% of the scene size, so we argue that the clone operation is mainly responsible for local refinement, while the split operation takes charge of the global diffusion.

Table 1: Average displacement distances for splitdominated, clone-dominated and equal primitives before and after optimization.
<table><tr><td>Category</td><td>MipNeRF-360</td><td>Deep Blending</td><td>Tanks &amp; Temples</td></tr><tr><td>split-dominated</td><td>2.42</td><td>0.75</td><td>2.40</td></tr><tr><td>clone-dominated</td><td>0.09</td><td>0.15</td><td>0.10</td></tr><tr><td>equal</td><td>0.17</td><td>0.23</td><td>0.15</td></tr><tr><td>camera extent</td><td>5.16</td><td>7.79</td><td>6.65</td></tr></table>

We present qualitative comparisons of Gaussian distributions across three densification strategies: (1) the standard adaptive approach from 3D Gaussian splatting that dynamically selects splits/clones based on Gaussian scale, versus (2) split-only and (3) clone-only variants where all densification operations are enforced to use a single type. As shown in Fig 2, clone-only version intensifies the local cluster phenomenon of the gaussian primitives (cf. bicycle frame and decoration on the wall) and fails to spread the Gaussian primitives, leading to a blurry rendering output due to insufficient spatial distribution (cf. houses in the distance of bicycle scene). Although the split-only variant produces a more uniform distribution compared to other approaches, the lack of clone operation prevents it from efficiently adapting to scene details. As a result, it requires a larger number of Gaussian primitives to fit the scene, leading to a reconstruction quality that is still inferior to that of the baseline.

In summary, both quantitative results and qualitative visual analysis indicate that the split operation primarily takes charge of the global distribution of Gaussian primitives, while the clone operation is mainly responsible for the local refinement.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

Figure 2: Visualization of the distribution of Gaussian primitives (left) and the rendered images (right) after optimization .  
<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 3: The number of Gaussian primitives generated through split and clone operations and the ratio of clone to split during the densification stage.

Split-clone ratio We present quantitative analysis of Gaussian primitive evolution during the densification phase in 3D Gaussian Splatting. As demonstrated in Fig. 3, approximately 80% of new primitives originate from cloning operations rather than splitting. To fit the rendering error caused by the opacity reset performed every 3k steps, both splits and clones were increased simultaneously. Subsequently, as the optimization progresses, the number of splits begins to decline, while the number of clones continues to rise. Comprehensive results across multiple scenes from different datasets confirm this trend.

According to the discover in the previous section, we argue that the local refinement at early densification phase cannot reduce the rendering error through clone operations, resulting in a persistently high gradient. As a result, a large number of Gaussian primitives are repeatedly cloned, causing computational redundancy and failing to improve the rendering quality.

These findings suggest two key implications: First, the majority of primitive growth stems from clone operations that may not contribute meaningfully to representation capacity in early stage. Second, the observed trend indicates potential for algorithmic optimization through adaptive densification strategies that can achieve computational efficiency with comparable performance.

## Global-to-local framework

Qualitative and quantitative experiments have proved that the clone operation used for local refinement has led to the generation of a large number of redundant Gaussian primitives, which is unnecessary for the optimization of the scene reconstruction in the early stage. Leveraging the empirically observed inverse correlation between Gaussian density and computational efficiency, We implement a two-phase densification framework: global spread and local refinement. At the first phase, we only apply the split operation to achieve fast and effective spatial distribution, leveraging minimal Gaussian counts to minimize redundant computation on localized details. In the second phase, both the split and clone operations are employed. Given that a satisfactory spatial distribution has already been established, the clone operation enables efficient local refinement. The proposed phased training strategy can reduce training time by optimizing a substantially smaller number of primitives in the first phase.

To further enhance the effectiveness of the global-to-local densification process, we propose a coarse-to-fine multiresolution scheduler based on the energy density in 2D images, which is elaborated in the following section. This approach eliminates the need for manually defining the boundary between the two phases.

## Coarse-to-fine multi-resolution densification

The optimization of 3D Gaussian Splatting is conducted by projecting Gaussian primitives from 3D space to 2D pixel plane, where the rendering error is computed to update both their spatial distribution and attribute parameters. During the global spread phase of densification, we aim for a fast and efficient coverage of the target scene volume, without overemphasizing the reconstruction of fine image details at this early stage. Therefore, using full-resolution images for supervision can lead to unnecessary computational consumption and suboptimal behavior. Specifically, each pixel corresponds to a small 3D voxel, which encourages Gaussians primitives to converge prematurely to local optima by overfitting to individual pixels, thereby limiting their spatial expansion. Furthermore, when a large Gaussian projects to a large number of pixels, the accumulated gradient vectors may cancel each other out due to opposing vector directions in space, resulting in a small gradient error below the threshold and preventing further split operations (Ye et al. 2024).

Inspired by frequency analysis techniques in 2D image processing and DashGaussian (Chen et al. 2025), we propose a coarse-to-fine training strategy based on energy density to mitigate the aforementioned issues. Specifically, during the global spread phase, we train with downsampled images to enable efficient spatial diffusion of Gaussian primitives. Once a sufficient scene coverage is attained, we switch to full-resolution supervision for the local refinement phase, allowing accurate reconstruction of high-frequency image content.

We analyze the image energy distribution in frequency domain to design an adaptive resolution scheduling mechanism. Given an input image $\textbf { I } \in \mathbb { R } ^ { H \times W \times C }$ , we compute its energy spectrum through Fourier transform: $\mathcal { E } ( \mathbf { I } ) =$ $\sqrt { \Re ( \mathcal { F } ( \mathbf { I } ) ) ^ { 2 } + \Im ( \mathcal { F } ( \mathbf { I } ) ) ^ { 2 } }$ , where $\mathcal { F } ( \cdot ) , \Re$ , andâ denotes 2D FFT, real part and imaginary part in frequency domain, and $\mathcal { E } ( \cdot )$ calculates the energy density.

For multi-resolution analysis, we define a downscaling operator $\mathcal { D } _ { r } ( \cdot )$ that reduces spatial dimensions by factor r using bilinear interpolation with anti-aliasing:

$$
\mathbf { I } _ { r } = \mathcal { D } _ { r } ( \mathbf { I } ) = B i l i n e a r ( \mathbf { I } , { \mathrm { s c a l e } } = 1 / r )\tag{4}
$$

The energy density across resolutions is quantified as:

$$
\boldsymbol { \mathcal { E } } _ { r } = \| \boldsymbol { \mathcal { E } } ( \mathbf { I } _ { r } ) \| _ { 1 } \cdot r ^ { 2 }\tag{5}
$$

where the scaling term $r ^ { 2 }$ normalizes energy values across different resolutions. Our resolution scheduler dynamically allocates training iterations based on energy ratios as follows:

$$
\mathbf { T } _ { r } = \operatorname { R o u n d } ( ( \mathcal { E } _ { r } / \mathcal { E } _ { 1 } ) \cdot \mathbf { T } _ { \mathrm { d e n s i f y } } ) , r \in \mathcal { A }\tag{6}
$$

where $\mathcal { A } = \{ 1 , 2 , . . . , K \}$ denotes candidate scale factors, and $\mathbf { T } _ { r } , \mathbf { T } _ { \mathrm { d e n s i f y } }$ represents the allocated iteration at r-scaled resolution and total densification iteration. The training proceeds from coarsest $( r = K )$ to finest $( r = 1 )$ resolution following reversed order. For scale factor r, the training starts at ${ \bf T } _ { r + 1 }$ and end at $\mathbf { T } _ { r }$ . This energy-aware strategy ensures optimal balance between global scene coverage at early phases and detail reconstruction in later phases.

## Adaptive Opacity Pruning

To prove the optimization stuck with floaters close to input cameras and unjustified increase in the Gaussian density, 3DGS (Kerbl et al. 2023) reset the opacity of all Gaussians primitives with an opacity value greater than 0.01 to 0.01, and prune those with opacities below this threshold. However, numerous Gaussians exhibit minimal visibility contribution and add little to rendered outputs, a fixed small threshold is suboptimal for pruning unnecessary Gaussians during optimization.

To maintain an efficient and compact Gaussian distribution during optimization, we implement an adaptive opacity threshold with a upper limit to prune these visually insignificant and redundant primitives. Let ${ \pmb { \alpha } } \in \mathbb { R } ^ { N }$ denote the opacity vector of all N Gaussian primitives. We first sort Î± in ascending order to obtain $\pmb { \alpha } _ { \mathrm { s o r t e d } }$ , where the k-th element satisfies:

$$
\tau _ { k } = \alpha _ { \mathrm { s o r t e d } } [ k ] , \quad k = \lfloor N \cdot p \rfloor
$$

Here, $p \in \mathsf { \Gamma } ( 0 , 1 )$ controls the pruning ratio. The adaptive opacity threshold Ï is then determined with a upper limit $\tau _ { u }$ as:

$$
\tau = \operatorname* { m i n } \left( \tau _ { k } , \tau _ { u } \right)
$$

This pruning operation with the dual-constrained threshold effectively eliminates redundant Gaussians while preserving structurally important primitives.

## Experiments

Datasets and metrics. We perform experiments on three real-world datasets: MipNeRF-360 (Barron et al. 2022), Deep Blending (Hedman et al. 2018) and Tanks&Temples (Knapitsch et al. 2017). Following the default data pre-processing in the 3D Gaussian splatting (Kerbl et al. 2023), we initialize the Gaussian primitives with the point clouds extracted from the structure from motion (SFM). we selected one out of every eight images to evaluate the average peak signal-to-noise ratio (PSNR), structural similarity index (SSIM) (Wang et al. 2004) and learned perceptual image patch similarity (LPIPS) (Zhang et al. 2018). Additionally, we report the number of Gaussian primitives and average training time (in minutes) on each dataset to prove the efficiency of the proposed method.

Implementation details We build our method upon the open-source accelerated version of 3DGS code base. Following (Kerbl et al. 2023), we train our models for 30K iterations across all scenes. We extend the iteration of densification $\mathbf { T } _ { \mathrm { d e n s i f y } }$ to 25K and set the default max scale factor $K ,$ pruning ratio $p ,$ and pruning upper limit $\tau _ { u }$ to 8, 0.03 and 0.05, respectively. To encourage efficient spatial diffusion of Gaussian primitives, we keep the positional learning rate constant during training with downsampled resolutions and reduce it after restoring full resolution. All experiments are conducted on an NVIDIA GeForce RTX 3090 GPU with a AMD EPYC 7413 24-Core processor CPU to ensure a fair comparison.

## Quantitative results

As shown in Table 2, We report the comparison with the state-of-the-art (SOTA) 3DGS reconstruction methods, i.e., 3DGS (Kerbl et al. 2023), 3DGS-accel1, EDC (Deng et al. 2024), mini-splatting (Fang and Wang 2024a), and Dash-Gaussian (Chen et al. 2025) in Table 2 in terms of training time, the number of Gaussian primitives and standard visual quality metrics. It is worth noting that mini-splatting is built upon 3DGS, whereas EDC and DashGaussian are based on 3DGS-accel.

Table 2: Quantitative evaluation comparing the proposed method with existing 3DGS optimization works. We report SSIM, PSNR (dB), LPIPS, number of Gaussian Primitives and training time (mins). The proposed method achieves superior performance with much less time cost.
<table><tr><td rowspan="2">Method</td><td colspan="5">MipNeRF-360 (Barron et al. 2022)</td><td colspan="5">Deep Blending (Hedman et al. 2018)</td><td colspan="5">ans &amp; Temples (Knapitsch et al. 0 SSIM â PSNR â LPIPS â</td></tr><tr><td>SSIM â</td><td>PSNR</td><td>LPIPS â</td><td>NGsâ</td><td>Time â</td><td>SIM </td><td>PSNR â</td><td>LPIS </td><td>Ngs â</td><td>me </td><td></td><td></td><td></td><td>Ngsâ</td><td>Time â</td></tr><tr><td>3DGS (Kerbl et al. 2023)</td><td>0.8263</td><td>27.72</td><td>0.2016</td><td>2.578 M</td><td>25.01</td><td>0.9075</td><td>29.44</td><td>0.2381</td><td>2.475 M</td><td>23.32</td><td>0.8471</td><td>23.62</td><td>0.1772</td><td>1.576 M</td><td>15.80</td></tr><tr><td>Mini-splatting (Fang and Wang 2024a)</td><td>0.8325</td><td>27.566</td><td>0.2011</td><td>0.493 M</td><td>18.21</td><td>0.9085</td><td>30.01</td><td>0.2409</td><td>0.555 M</td><td>15.51</td><td>0.8467</td><td>23.45</td><td>0.1804</td><td>0.301 M</td><td>10.54</td></tr><tr><td>3DGS-accel (Mallick et al. 2024)</td><td>0.8213</td><td>27.57</td><td>0.2095</td><td>2.331M</td><td>11.18</td><td>0.9027</td><td>29.54</td><td>0.2537</td><td>2.394M</td><td>8.16</td><td>0.8460</td><td>23.58</td><td>01756</td><td>1.550M</td><td>7.73</td></tr><tr><td>EDC (Deng et al. 2024)</td><td>0.8342</td><td>27.86</td><td>01964</td><td>1253M</td><td>10.41</td><td>09093</td><td>29.92</td><td>0.2415</td><td>0.623M</td><td>7.57</td><td>0.8496</td><td>23.98</td><td>01771</td><td>0.570M</td><td>6.68</td></tr><tr><td>DashGaussian (Chen et al. 2025)</td><td>00.8261</td><td>27.90</td><td>0.2084</td><td>2.081M</td><td>6.34</td><td>0.9026</td><td>30.01</td><td>0.2511</td><td>1.955M</td><td>5.12</td><td>0.8468</td><td>23.95</td><td>0.1824</td><td>1.198M</td><td>5.57</td></tr><tr><td>Ours</td><td>0.8257</td><td>27.79</td><td>0.2136</td><td>1.469M</td><td>5.33</td><td>0.9094</td><td>30.05</td><td>0.2540</td><td>1.272M</td><td>4.54</td><td>0.8461</td><td>24.06</td><td>0.1891</td><td>0.867M</td><td>4.10</td></tr></table>

<!-- image-->  
Figure 4: Qualitative comparison between our method and prior 3DGS approaches, along with the corresponding ground truth images from test viewpoints.

Compared to the baseline 3DGS-accel, our approach demonstrates a significant 2Ã speedup with 40% fewer Gaussian primitives across all three datasets. Thanks to the proposed efficient global-to-local optimization and energyaware multi-resolution densification strategies , our method not only improves computational efficiency but also enhances reconstruction quality. Specifically, it achieves an average improvement of +0.004 in SSIM and +0.31 dB in PSNR , while maintaining strong perceptual fidelity with only a negligible 0.0049 increase in LPIPS. In comparison to existing SOTA methods, our approach achieves the fastest convergence speed while preserving competitive rendering quality.

MSv2 (Fang and Wang 2024b) is an extended version of mini-splatting (Fang and Wang 2024a), which adopts an aggressive densification strategy and limits the optimization of Gaussian primitives to 18K iterations. For a fair comparison, we also train our proposed method for 18K iteration, with 15K iterations allocated to densification. Results on the MipNeRF-360 dataset, as shown in Table 3, demonstrates that our method achieves a better performance with a less training time.

## Qualitative results

The qualitative performance is displayed as rendered images in Fig. 4. These results align well with the quantitative results provided in Table 2. Our method achieves comparable or even better rendering quality with a less training time. Besides, due to the efficient diffusion of the Gaussian primitives in space, our method enables accurate reconstruction of small objects (i.e., lamp bulbs) and produces clear renderings for distant views (i.e., remote house), shown in Fig 5. Although the limited projected 2D pixel coverage of small and distant objects prevents the improvement from being clearly reflected in the quantitative metrics, the visual results highlight practical benefits that go beyond numerical measurements. These findings underscore the effectiveness and real-world applications of the proposed method.

Table 3: Comparison to Msv2 within 18K optimization iterations.
<table><tr><td>Method</td><td>SSIM â</td><td>PSNR â</td><td>LPIPS â</td><td>NGsâ</td><td>Time â</td></tr><tr><td>MSv2 (Fang and Wang 2024b)</td><td>0.8206</td><td>27.35</td><td>0.2149</td><td>0.618 M</td><td>3.55</td></tr><tr><td>Ours-18K</td><td>0.8237</td><td>27.65</td><td>0.2137</td><td>1.085 M</td><td>3.47</td></tr></table>

## Ablation studies

We use 3DGS-accel (Mallick et al. 2024) as the backbone and individually add each densification method to explore their effect on the rendering quality and optimization speed. We conducted experiments on MipNeRF-360 dataset (Barron et al. 2022), since it contains both indoor and outdoor scenes.

<!-- image-->  
3DGS-accel

<!-- image-->  
Ours  
Figure 5: Qualitative results for small and distant object reconstruction.

Global-to-local densification We adopt the same experimental configurations as those used in 3DGS-accel and employ iteration $\mathbf { T } _ { 2 }$ (refer to eqn. 6) as the critical boundary threshold that delineates the transition between the global spread and the local refinement.

Coarse-to-fine densification Following the sec. , we calculate the number of iterations for different resolution of each scene. The impact is evaluated with and without globalto-fine strategy.

Table 4: Ablation studies of the proposed method on the MipNeRF-360 dataset. G2L and C2F denote global-to-local and coarse-to-fine densification.
<table><tr><td>Method</td><td>SSIMâ</td><td>PSNR â</td><td>LPIPS â</td><td>NGsâ</td><td>Time â</td></tr><tr><td>3DGS-accel</td><td>0.8213</td><td>27.57</td><td>0.2095</td><td>2.331 M</td><td>11.18</td></tr><tr><td>+ G2L</td><td>0.8066</td><td>27.47</td><td>0.2235</td><td>1.887 M</td><td>8.46</td></tr><tr><td>+ C2F</td><td>0.8246</td><td>27.84</td><td>0.2202</td><td>2.018 M</td><td>7.56</td></tr><tr><td>+ G2L + C2F</td><td>0.8176</td><td>27.75</td><td>0.2203</td><td>1.853 M</td><td>6.95</td></tr><tr><td>+Pruning</td><td>0.8211</td><td>27.56</td><td>0.2234</td><td>1.685 M</td><td>9.21</td></tr><tr><td>Full</td><td>0.8257</td><td>27.79</td><td>0.2136</td><td>1.469M</td><td>5.33</td></tr></table>

As shown in Tab. 4, the global-to-local strategy effectively reduces computational costs but incurs a slight degradation in image quality. In contrast, the coarse-to-fine densification approach demonstrates improved PSNR while maintaining low LPIPS scores, simultaneously reducing computational overhead. The combination of both global-to-local and coarse-to-fine components further optimizes efficiency with minimal compromise in visual quality. We also test the effect of adaptive opacity pruning by applying it to 3D-accel. It can reduces limited computational cost but introduces a minor trade-off in image quality. Ultimately, the full model achieves the lowest optimization time with superior performance than baseline. These findings underscore the efficacy of our method in enhancing both the visual fidelity and computational efficiency.

Hyperparameters We evaluate the impact of various hyperparameters on actual training efficiency and final reconstruction quality, including densification iteration Tdensify (25K) in Tab. 5, pruning ratio p (0.03) and pruning upper limit $\tau _ { u } ~ ( 0 . 0 5 )$ in Tab. 6. The numbers in parentheses represent the default values used in this paper.

A smaller $\mathbf { T } _ { \mathrm { d e n s i f y } }$ indicates that densification is completed earlier, leaving more iterations for full-precision optimization. However, this typically leads to increased computational cost with only marginal performance improvement. For opacity pruning, a lower pruning ratio $p$ and a lower pruning upper limit $\tau _ { u }$ preserve more Gaussian primitives with small opacity values, which in turn increases computational overhead. In contrast, aggressive pruning may lead to excessive removal of informative primitives, resulting in a noticeable decline in reconstruction quality. Overall, there exists a trade-off between training efficiency and rendering fidelity that must be carefully balanced.

Table 5: Ablation studies of the densification iteration $\mathbf { T } _ { \mathrm { d e n s i f y } }$
<table><tr><td> $\mathbf { T } _ { \mathrm { d e n s i f y } }$ </td><td>SSIM â</td><td>PSNR â</td><td>LPIPS â</td><td> $N _ { G S \downarrow }$ </td><td>Time â</td></tr><tr><td>15 K</td><td>0.8272</td><td>27.81</td><td>0.2115</td><td>1.476 M</td><td>6.14</td></tr><tr><td>20 K</td><td>0.8268</td><td>27.79</td><td>0.2141</td><td>1.426 M</td><td>5.73</td></tr><tr><td>25K</td><td>0.8257</td><td>27.79</td><td>0.2136</td><td>1.469 M</td><td>5.33</td></tr></table>

Table 6: Ablation studies of the pruning hyperparameters.
<table><tr><td>p</td><td> $\tau _ { u }$ </td><td>SSIMâ</td><td>PSNR â</td><td>LPIPS â</td><td> $N _ { G S \downarrow }$ </td><td>Time â</td></tr><tr><td></td><td>fixed Ï = 0.01</td><td>0.8272</td><td>27.86</td><td>0.2098</td><td>1.682 M</td><td>7.04</td></tr><tr><td>0.01</td><td>0.05</td><td>0.8295</td><td>27.88</td><td>0.2112</td><td>1.524 M</td><td>5.74</td></tr><tr><td>0.03</td><td>0.05</td><td>0.8257</td><td>27.79</td><td>0.2136</td><td>1.469 M</td><td>5.33</td></tr><tr><td>0.05</td><td>0.05</td><td>0.8235</td><td>27.68</td><td>0.2168</td><td>1.412 M</td><td>5.24</td></tr><tr><td>0.03</td><td>0.01</td><td>0.8291</td><td>28.04</td><td>0.2081</td><td>1.812 M</td><td>6.84</td></tr><tr><td>0.03</td><td>0.05</td><td>0.8257</td><td>27.79</td><td>0.2136</td><td>1.469 M</td><td>5.33</td></tr><tr><td>0.03</td><td>0.10</td><td>0.8185</td><td>27.59</td><td>0.2277</td><td>1.286 M</td><td>4.80</td></tr></table>

## Conclusion and limitations

In this paper, we present a simple but efficient approach to accelerate 3D Gaussian Splatting for efficient 3D scene reconstruction by decomposing the densification. Through systematic analysis, we reveal that split operations primarily govern global spatial spread of Gaussian primitives, while clone operations focus on local refinement. Leveraging this insight, we propose a global-to-local densification strategy that decouples split and clone operations across training phases, enabling efficient scene coverage followed by detail-preserving refinement. Subsequenctly, we introduce an energy-guided coarse-to-fine multi-resolution framework and a dynamic pruning mechanism to further enhance acceleration. Numerous experiments across three real-world datasets highlight the effectiveness of our strategy in balancing computational efficiency with high-fidelity rendering. This paper primarily aim at a training acceleration and does not tackle the inherent blur problem present in 3DGS, which stems from insufficient gradient accumulation of big Gaussians. We will consider how to design a reasonable gradient threshold to achieve better renderings.

## References

Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.; and Hedman, P. 2022. Mip-nerf 360: Unbounded antialiased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5470â5479.

Cha, H.; Lee, I.; and Joo, H. 2024. PERSE: Personalized 3D Generative Avatars from A Single Portrait. arXiv preprint arXiv:2412.21206.

Chao, B.; Tseng, H.-Y.; Porzi, L.; Gao, C.; Li, T.; Li, Q.; Saraf, A.; Huang, J.-B.; Kopf, J.; Wetzstein, G.; et al. 2024. Textured Gaussians for Enhanced 3D Scene Appearance Modeling. arXiv preprint arXiv:2411.18625.

Chen, Y.; Jiang, J.; Jiang, K.; Tang, X.; Li, Z.; Liu, X.; and Nie, Y. 2025. DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds. arXiv preprint arXiv:2503.18402.

Chen, Z.; Yang, J.; Huang, J.; de Lutio, R.; Esturo, J. M.; Ivanovic, B.; Litany, O.; Gojcic, Z.; Fidler, S.; Pavone, M.; et al. 2024. Omnire: Omni urban scene reconstruction. arXiv preprint arXiv:2408.16760.

Cheng, K.; Long, X.; Yang, K.; Yao, Y.; Yin, W.; Ma, Y.; Wang, W.; and Chen, X. 2024. Gaussianpro: 3d gaussian splatting with progressive propagation. In Forty-first International Conference on Machine Learning.

Cong, W.; Zhu, H.; Wang, K.; Lei, J.; Stearns, C.; Cai, Y.; Wang, D.; Ranjan, R.; Feiszli, M.; Guibas, L.; et al. 2025. VideoLifter: Lifting Videos to 3D with Fast Hierarchical Stereo Alignment. arXiv preprint arXiv:2501.01949.

Deng, X.; Diao, C.; Li, M.; Yu, R.; and Xu, D. 2024. Efficient Density Control for 3D Gaussian Splatting. arXiv preprint arXiv:2411.10133.

Fang, G.; and Wang, B. 2024a. Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians. In Leonardis, A.; Ricci, E.; Roth, S.; Russakovsky, O.; Sattler, T.; and Varol, G., eds., Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXVII, volume 15135 of Lecture Notes in Computer Science, 165â181. Springer.

Fang, G.; and Wang, B. 2024b. Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification. arXiv preprint arXiv:2411.12788.

Girish, S.; Gupta, K.; and Shrivastava, A. 2023. EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS. CoRR, abs/2312.04564.

Hanson, A.; Tu, A.; Lin, G.; Singla, V.; Zwicker, M.; and Goldstein, T. 2024. Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives. arXiv preprint arXiv:2412.00578.

Hedman, P.; Philip, J.; Price, T.; Frahm, J.; Drettakis, G.; and Brostow, G. J. 2018. Deep blending for free-viewpoint image-based rendering. ACM Trans. Graph., 37(6): 257.

Hess, G.; Lindstrom, C.; Fatemi, M.; Petersson, C.; and Â¨ Svensson, L. 2024. SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving. arXiv preprint arXiv:2411.16816.

Javed, S.; Khan, A. J.; Dumery, C.; Zhao, C.; and Salzmann, M. 2024. Temporally Compressed 3D Gaussian Splatting for Dynamic Scenes. arXiv preprint arXiv:2412.05700.

Jiang, Y.; Shen, Z.; Hong, Y.; Guo, C.; Wu, Y.; Zhang, Y.; Yu, J.; and Xu, L. 2024. Robust dual gaussian splatting for immersive human-centric volumetric videos. ACM Transactions on Graphics (TOG), 43(6): 1â15.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G.Â¨ 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph., 42(4): 139:1â139:14.

Knapitsch, A.; Park, J.; Zhou, Q.-Y.; and Koltun, V. 2017. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36(4): 1â 13.

Lei, X.; Wang, M.; Zhou, W.; and Li, H. 2025. Gaussnav: Gaussian splatting for visual navigation. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Lyu, W.; Zhou, Y.; Yang, M.-H.; and Shu, Z. 2024. FaceLift: Single Image to 3D Head with View Generation and GS-LRM. arXiv preprint arXiv:2412.17812.

Mallick, S. S.; Goel, R.; Kerbl, B.; Carrasco, F. V.; Steinberger, M.; and la Torre, F. D. 2024. Taming 3DGS: High-Quality Radiance Fields with Limited Resources. CoRR, abs/2406.15643.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In Vedaldi, A.; Bischof, H.; Brox, T.; and Frahm, J., eds., Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part I, volume 12346 of Lecture Notes in Computer Science, 405â421. Springer.

Oechsle, M.; Peng, S.; and Geiger, A. 2021. UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, 5569â5579. IEEE.

Park, K.; Sinha, U.; Barron, J. T.; Bouaziz, S.; Goldman, D. B.; Seitz, S. M.; and Martin-Brualla, R. 2021. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, 5865â5874.

Tan, B.; Yu, R.; Shen, Y.; and Xue, N. 2024. PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes. arXiv preprint arXiv:2412.03451.

Waczynska, J.; Borycki, P.; Tadeja, S. K.; Tabor, J.; and Spurek, P. 2024. GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting. CoRR, abs/2402.01459.

Wang, P.; Liu, L.; Liu, Y.; Theobalt, C.; Komura, T.; and Wang, W. 2021. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. In Ranzato, M.; Beygelzimer, A.; Dauphin, Y. N.; Liang, P.; and Vaughan, J. W., eds., Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, 27171â27183.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600â612.

Xie, T.; Chen, X.; Xu, Z.; Xie, Y.; Jin, Y.; Shen, Y.; Peng, S.; Bao, H.; and Zhou, X. 2024. Envgs: Modeling viewdependent appearance with environment gaussian. arXiv preprint arXiv:2412.15215.

Xu, J.; Mei, Y.; and Patel, V. 2024. Wild-gs: Real-time novel view synthesis from unconstrained photo collections. Advances in Neural Information Processing Systems, 37: 103334â103355.

Yan, Y.; Lin, H.; Zhou, C.; Wang, W.; Sun, H.; Zhan, K.; Lang, X.; Zhou, X.; and Peng, S. 2024. Street Gaussians for Modeling Dynamic Urban Scenes. CoRR, abs/2401.01339.

Yang, C.; Li, S.; Fang, J.; Liang, R.; Xie, L.; Zhang, X.; Shen, W.; and Tian, Q. 2024. GaussianObject: Just Taking Four Images to Get A High-Quality 3D Object with Gaussian Splatting. CoRR, abs/2402.10259.

Yariv, L.; Gu, J.; Kasten, Y.; and Lipman, Y. 2021. Volume Rendering of Neural Implicit Surfaces. In Ranzato, M.; Beygelzimer, A.; Dauphin, Y. N.; Liang, P.; and Vaughan, J. W., eds., Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, 4805â4815.

Ye, Z.; Li, W.; Liu, S.; Qiao, P.; and Dou, Y. 2024. Absgs: Recovering fine details in 3d gaussian splatting. In Proceedings of the 32nd ACM International Conference on Multimedia, 1053â1061.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586â595.

Zhou, H.; Shao, J.; Xu, L.; Bai, D.; Qiu, W.; Liu, B.; Wang, Y.; Geiger, A.; and Liao, Y. 2024. Hugs: Holistic urban 3d scene understanding via gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21336â21345.

Zielonka, W.; Garbin, S. J.; Lattas, A.; Kopanas, G.; Gotardo, P.; Beeler, T.; Thies, J.; and Bolkart, T. 2025. Synthetic Prior for Few-Shot Drivable Head Avatar Inversion. arXiv preprint arXiv:2501.06903.