# HGS: Hybrid Gaussian Splatting with Static-Dynamic Decomposition for Compact Dynamic View Synthesis

Kaizhe Zhang\* Xiâan Jiaotong University Caixia Yan Xiâan Jiaotong University Xiâ

Haipeng Du an Jiaotong University

yugui xie chinamobile.com

Weizhan Zhang Xiâan Jiaotong University

Yu-Hui Wen en Beijing Jiaotong U niversity

Xuanyu Wang Xiâan Jiaotong University Yong-Jin Liu Tsinghua University

<!-- image-->  
(a)High Quality Result

<!-- image-->  
(b)Real-Time 4K at 128 FPS

<!-- image-->  
(c)Comparison on Quality, Speed and Size  
Figure 1: Performance comparison with existing SOTA [21, 10, 39, 1, 41, 2, 43]. Our approach achieves higher-quality reconstruction in (a) temporally complex scenes, (b) while maintaining real-time rendering, (c) with an improvement in performance.

## ABSTRACT

Dynamic novel view synthesis (NVS) is essential for creating immersive experiences. Existing approaches have advanced dynamic NVS by introducing 3D Gaussian Splatting (3DGS) with implicit deformation fields or indiscriminately assigned time-varying parameters, surpassing NeRF-based methods. However, due to excessive model complexity and parameter redundancy, they incur large model sizes and slow rendering speeds, making them inefficient for real-time applications, particularly on resource-constrained devices. To obtain a more efficient model with fewer redundant parameters, in this paper, we propose Hybrid Gaussian Splatting (HGS), a compact and efficient framework explicitly designed to disentangle static and dynamic regions of a scene within a unified representation. The core innovation of HGS lies in our StaticâDynamic Decomposition (SDD) strategy, which leverages Radial Basis Function (RBF) modeling for Gaussian primitives. Specifically, for dynamic regions, we employ time-dependent RBFs to effectively capture temporal variations and handle abrupt scene changes, while for static regions, we reduce redundancy by sharing temporally invariant parameters. Additionally, we introduce a twostage training strategy tailored for explicit models to enhance temporal coherence at static-dynamic boundaries. Experimental results demonstrate that our method reduces model size by up to 98% and achieves real-time rendering at up to 125 FPS at 4K resolution on a single RTX 3090 GPU. It further sustains 160 FPS at 1352Ã1014 on an RTX 3050 and has been integrated into the VR system. Moreover, HGS achieves comparable rendering quality to state-of-the-art methods while providing significantly improved visual fidelity for high-frequency details and abrupt scene changes.

Index Terms: Real-time Rendering, Gaussian Splatting, Dynamic Scene Modeling, Virtual Reality.

## 1 INTRODUCTION

Dynamic novel view synthesis (NVS) aims to reconstruct photorealistic images of dynamic scenes from novel viewpoints. This task has been consistently pursued in computer vision and graphics due to its broad applications in augmented reality, virtual reality, telepresence, and immersive media streaming.

Recent advances in implicit neural representations, particularly Neural Radiance Fields (NeRF) [28], have significantly enhanced the fidelity and convenience of static scene modeling. However, NeRF-based methods typically suffer from prolonged training times and slow rendering speed. Although substantial research [10, 3, 25, 6, 7, 30, 11, 37] has aimed to mitigate these limitations, it remains challenging to achieve both high-quality and fast rendering simultaneously.

To address NeRFâs computational bottlenecks from dense volume sampling, explicit approaches like 3D Gaussian Splatting (3DGS) [16] have emerged. 3DGS explicitly models scenes using differentiable 3D Gaussian primitives that can be efficiently rasterized onto image planes. This rasterization strategy significantly improves rendering speed while maintaining high visual fidelity, addressing NeRFâs limitations in computational efficiency. Despite the success of 3DGS in static scene modeling, its extension to dynamic scenarios remains challenging. Prior works [19, 26, 29] demonstrate effectiveness in static settings but encounter interframe discontinuities in dynamic scenes. Recent methods adopt implicit representations [13, 18, 23, 41, 45, 43] to capture dynamics, but these incur high computational costs, slow rendering, and failure under abrupt motion changes. To improve efficiency, explicit techniquesâsuch as time-dependent modules [24, 21, 15] or 4D Gaussian extensions [8, 46]âhave been explored. However, they assign dynamic parameters to static regions, causing parameter redundancy, excessive complexity, and reduced efficiency.

In summary, existing dynamic 3DGS approaches face two key challenges: (1) excessive temporal parameters assigned to static content, resulting in significant computational and memory overhead; and (2) visual degradation due to temporal artifacts and the loss of high-frequency details in static regions. Moreover, commonly used temporal modeling is inadequate for capturing abrupt scene transitions, limiting the fidelity and robustness of dynamic scene reconstruction.

To address the aforementioned limitations, we introduce a novel Hybrid Gaussian Splatting (HGS) framework for dynamic scene representation with lower storage cost, higher rendering speed, and high quality, specifically designed for dynamic scene representation. To reduce storage cost, we adopt a Static-Dynamic Decomposition (SDD) strategy by deploying separate sets of static and dynamic Gaussian primitives, enabling explicit separation of static and dynamic regions. For static regions, we share the parameters to enable modeling without the burden of unnecessary dynamic parameters, reducing computational overhead. For dynamic regions, we restrict temporal Radial Basis Function (RBF) [21] to them, avoiding implicit motion modeling such as deformation fields. This scheme further reduces computational overhead and enables even faster rendering speed.

To suppress temporal artifacts, our proposed SDD strategy constrains temporal RBF [21] to dynamic regions only. This constraint effectively reduces cross-region temporal interference, preserving high-frequency details and visual fidelity within static content. Our explicit separation, which disentangles dynamic motion from static regions, not only enhances motion accuracy but also significantly improves the overall rendering quality. To further strengthen consistency at the boundary between static and dynamic regions, we introduce a two-stage training strategy specifically tailored for our explicit Gaussian representation. Within each optimization cycle, the training alternates between optimizing static and dynamic primitives. This iterative optimization strategy facilitates mutual adaptation between static and dynamic components. Combining it with modeling dynamic regions explicitly using temporal RBF, our approach effectively copes with abrupt scene changes and substantially reduces artifacts along static-dynamic boundaries.

Our HGS achieves over 300 FPS at 1080p resolution on a single RTX 3090, while requiring significantly fewer parameters, thanks to the explicit SDD strategy. The temporal RBF used for dynamic regions and our two-stage training strategy effectively reduces temporal artifacts. Therefore, with a reduced number of parameters, our HGS achieves superior or comparable performance to recent state-of-the-art methods in terms of both PSNR and SSIM [47]. In most scenarios, improved visual quality can be observed, attributed to the suppression of temporal artifacts and the preservation of highfrequency details in static regions. Our main contributions are summarized as follows:

â¢ We propose a novel HGS framework for compact and efficient dynamic scene reconstruction, explicitly separating static and dynamic components within a unified rendering and optimization pipeline.

â¢ We introduce the SDD strategy, which employs parameter sharing by assigning a compact set of temporally invariant parameters to all static primitives. Combined with explicit modeling, SDD significantly reduces model complexity, thereby achieving faster rendering while maintaining compatibility with dynamic modeling.

â¢ We design a two-stage training strategy specifically for our explicit Gaussian representation. This iterative approach enhances temporal consistency and effectively reduces artifacts at static-dynamic boundaries through mutual adaptation.

â¢ As shown in Fig. 1, our approach delivers rendering quality comparable to state-of-the-art methods while reducing model size by up to 98%, yielding a far more compact and efficient representation. It also better preserves high-frequency details in static regions and suppresses temporal artifacts under abrupt scene changes.

## 2 RELATED WORKS

## 2.1 Dynamic NeRF

Recent progress in dynamic NeRF [28] has primarily focused on extending static scene methods to handle temporal dynamics. Many studies leverage deformation-based methods [12, 22, 32, 33, 34, 38, 42], introducing neural deformation fields to warp dynamic scenes into a canonical static frame. For example, methods such as D-NeRF [34], Nerfies [32], and HyperNeRF [33] effectively capture complex scene dynamics through learned deformations but incur substantial computational costs due to dense volumetric sampling. To alleviate these computational issues, structured representation methods [5, 9, 10, 35, 39, 40, 36, 1] have emerged. K-Planes [10] and MixVoxels [39] factorize the spacetime domain into structured planes or adaptive voxel grids, significantly reducing computational complexity. NeRFPlayer [36] integrates explicit time decomposition to better handle dynamic variations, while HyperReel [1] proposes low-dimensional temporal embeddings to enhance efficiency. However, these structured methods may still struggle to represent rapid or highly intricate motion accurately.

## 2.2 Dynamic 3DGS

The substantial advancements in rendering quality and efficiency brought by 3DGS have motivated subsequent research [13, 14, 18, 23, 41, 45, 24, 21, 8, 46, 27, 2], aimed at extending the originally static 3DGS framework to dynamic scene reconstruction. Early works [27, 41] employ MLPs to regress per-frame deformations of Gaussian parameters, enabling flexibility but incurring significant computational overhead. ED3DGS [2] enhances this formulation by introducing per-Gaussian latent embeddings for deformation and decomposing motion into coarse and fine levels. RoDyGS [14] incorporates a motion basis and separates static and dynamic points during the training of deformation fields to improve robustness and efficiency. However, all of these methods rely on implicit motion modeling, which limits rendering speed and increases model complexity.

To overcome the limitations of implicit modeling, 4DGS [46] extends 3D Gaussians into 4D space by attaching temporal features to each primitive, while STGS [21] models the temporal evolution of positions and orientations via polynomial functions and the temporal RBF. These methods improve runtime efficiency and avoid the need for implicit deformation fields. However, they often treat all Gaussian primitives as dynamic, including those belonging to static regions, leading to parameter redundancy and potential temporal blurring.

Unlike prior works that rely on implicit deformation fields to model dynamic motion, our approach adopts a temporal RBF that avoids such computational overhead. Additionally, we introduce a SSD strategy, assigning time-dependent parameters only to dynamic regions while modeling static regions with a shared, timeindependent representation. This design reduces parameter redundancy, avoids temporal artifacts in static regions, and enables significantly faster rendering while maintaining visual quality comparable to state-of-the-art methods.

## 3 PRELIMINARY: 3D GAUSSIAN SPLATTING

3DGS [16] is an explicit representation method that models scenes with differentiable anisotropic 3D Gaussians. Given a set of cali brated images captured from multiple views, 3DGS optimizes Gaussian parameters of these Gaussians via differentiable rasterization to rep resent scenes effectively and render novel views efficiently.

Specifically, each Gaussian $G _ { i }$ in a scene is parameterized by several attributes: a spatial position $\mu _ { i } \in \mathbb { R } ^ { 3 }$ , a covariance matrix $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ that defines its shape and orientation, an opacity scalar $\sigma _ { i } \in \mathbb { R }$ controlling transparency, and spherical harmonics (SH) coefficients ${ \bf h } _ { i }$ encoding view-dependent appearance.

<!-- image-->  
Figure 2: Overview of the proposed HGS pipeline, which comprises two components: (top) Static-Dynamic Decomposition Strategy, including (a) Initialization, selecting every m th frame for initialization to reduce the number of SFM points; (b) Temporal RBF, adding time-dependent parameters for dynamic Gaussian primitives; and (c) Parameter Sharing, enforcing time-independent parameters for static Gaussian primitives. (bottom) Two-Stage Training: Stage I updates static Gaussian primitives, while Stage II jointly optimizes dynamic Gaussian primitives using the refined static primitives from Stage I.

The 3D Gaussian kernel at any spatial location $x \in \mathbb { R } ^ { 3 }$ can be formally expressed as:

$$
\alpha _ { i } ( x ) = \sigma _ { i } \exp \left( - \frac { 1 } { 2 } ( x - \mu _ { i } ) ^ { T } \Sigma _ { i } ^ { - 1 } ( x - \mu _ { i } ) \right) ,\tag{1}
$$

where the covariance matrix $\Sigma _ { i }$ is symmetric and positive semidefinite. To ensure these constraints and achieve intuitive geometric interpretation, $\Sigma _ { i }$ is decomposed using scaling and rotation components:

$$
\Sigma _ { i } = R _ { i } S _ { i } S _ { i } ^ { T } R _ { i } ^ { T } ,\tag{2}
$$

with $S _ { i }$ being a diagonal scaling matrix parameterized by a scaling vector $s _ { i } \in \mathbb { R } ^ { 3 }$ and $R _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ a rotation matrix derived from a quaternion representation $q _ { i } \in \mathbb { R } ^ { 4 }$ , enforcing unit-norm constraint for stability.

To render an image from a specific viewpoint, each 3D Gaussian is projected from the 3D world coordinate system onto the 2D image plane via an approximate perspective projection. Given camera extrinsics (viewing matrix) $\dot { W } \in \dot { \mathbb { R } } ^ { 4 \times 4 }$ and intrinsics (projection matrix) $K \in \dot { \mathbb { R } } ^ { 3 \times 4 }$ , the 3D position Âµi and covariance $\Sigma _ { i }$ are projected to obtain the corresponding 2D Gaussian distribution with mean $\mu _ { i } ^ { 2 D }$ and covariance matrix $\Sigma _ { i } ^ { 2 D }$ :

$$
\mu _ { i } ^ { 2 D } = \left( K \frac { W \mu _ { i } } { \left( W \mu _ { i } \right) _ { z } } \right) _ { 1 : 2 } ,\tag{3}
$$

$$
\Sigma _ { i } ^ { 2 D } = ( J W \Sigma _ { i } W ^ { T } J ^ { T } ) _ { 1 : 2 , 1 : 2 } ,\tag{4}
$$

where $( W \mu _ { i } ) _ { : }$ denotes the depth after viewing transformation, and J is the Jacobian matrix of the affine approximation of the perspective projection, reflecting the local linearization of the nonlinear projective transformation.

After obtaining the projected Gaussians, rendering is achieved through alpha compositing along the camera viewing direction. The Gaussians are first sorted in descending order based on their depth relative to the camera. The final pixel color I at the pixel coordinate is computed through alpha-blending of sorted Gaussians that influence the pixel:

$$
I = \sum _ { i \in N } c _ { i } \alpha _ { i } ^ { 2 D } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { 2 D } ) ,\tag{5}
$$

where N denotes the ordered set of Gaussians influencing the pixel, and the opacity $\alpha _ { i } ^ { 2 D }$ at the pixel position $x _ { 2 D }$ is evaluated as the 2D counterpart of the 3D Gaussian kernel:

$$
\alpha _ { i } ^ { 2 D } ( x _ { 2 D } ) = \sigma _ { i } \exp \left( - \frac { 1 } { 2 } ( x _ { 2 D } - \mu _ { i } ^ { 2 D } ) ^ { T } ( \Sigma _ { i } ^ { 2 D } ) ^ { - 1 } ( x _ { 2 D } - \mu _ { i } ^ { 2 D } ) \right) ,\tag{6}
$$

and the color $c _ { i }$ is computed from spherical harmonics(SH) coefficients $h _ { i }$ given the corresponding viewing direction $d _ { \nu }$

$$
c _ { i } = \mathrm { S H } ( h _ { i } , d _ { \nu } ) ,\tag{7}
$$

where SH(Â·) denotes the evaluation of spherical harmonics.

## 4 METHOD

In this section, we introduce our HGS framework for dynamic scene representation with lower storage cost, higher rendering speed, and high quality. Our method achieves this by incorporating an SDD strategy and a two-stage training strategy. This design not only enables a more compact representation but also supports efficient, high-quality rendering. An overview of our method is illustrated in Fig. 2. Sec. 4.1 describes the hybrid Gaussian primitives. Sec. 4.2 presents our SDD strategy for detailed modeling. Sec. 4.3 introduces the two-stage training strategy, followed by optimization details in Sec. 4.4.

## 4.1 Hybrid Gaussians

To efficiently and compactly represent dynamic 4D scenes, we explicitly separate static and dynamic primitives within a unified rendering and optimization framework. Inspired by recent work on STGS [21], our approach extends traditional 3D Gaussian primitives into the 4D spacetime domain by explicitly modeling their temporal evolution. Specifically, each Gaussian primitive $G _ { i }$ is parameterized by temporally varying spatial position $\mu _ { i } ( t )$ , opacity $\sigma _ { i } ( t )$ , rotation quaternion $r _ { i } ( t )$ , as well as temporally invariant anisotropic scale vector $s _ { i }$ and color coefficients $c _ { i } .$ . The explicit formulations for these parameters are given in Eq. (1), Eq. (2), and Eq. (7), respectively. Within this modeling framework, we propose a novel approach that separates static and dynamic primitives using a binary indicator, and applies distinct parameterization strategies for each category of primitives.

## 4.1.1 Dynamic Primitives

For dynamic primitives, each Gaussian independently maintains its own temporally varying parameters. The temporal evolution of each dynamic primitive is explicitly parameterized. Specifically, each Gaussian parameter is modeled through polynomial interpolation. The detailed formulations are described in Sec. 4.2.2.

## 4.1.2 Static Primitives

All static Gaussian primitives share a unified, temporally invariant parameter set. By leveraging this shared parameterization, our method significantly reduces redundancy, thereby providing a compact and efficient representation of static scene components. The detailed parameter-sharing strategy is described in Sec. 4.2.3.

## 4.2 Static-Dynamic Decomposition Strategy

To realize this idea more effectively, we introduce the SDD strategy, which provides a concrete implementation of separate modeling for dynamic and static regions within our hybrid Gaussian framework. We first perform initialization to obtain Gaussian primitives for both categories. For dynamic regions, we apply temporal RBF [21] to capture time-varying behavior. In contrast, static regions share a single set of temporally invariant parameters, allowing for efficient modeling without the overhead of redundant dynamic Gaussians.

## 4.2.1 Initialization

To enable compact and efficient scene representation, we adapt the interleaved-frame initialization scheme. Rather than relying on dense SfM point clouds from all frames, hybrid Gaussian primitives are initialized using reconstructions from temporally subsampled frames. This approach limits the number of initial Gaussian primitives, ensuring a favorable trade-off between computational efficiency and the fidelity needed to capture the scene.

## 4.2.2 Radial Basis Function

For dynamic primitives, we adapt the temporal RBF of STGS [21] to explicitly parameterize the temporal evolution. The formulations

are as follows:

$$
\mu _ { i } ( t ) = \sum _ { k = 0 } ^ { 3 } b _ { i , k } ( t - \mu _ { i } ^ { \tau } ) ^ { k } ,\tag{8}
$$

$$
r _ { i } ( t ) = \sum _ { k = 0 } ^ { 1 } c _ { i , k } ( t - \mu _ { i } ^ { \tau } ) ^ { k } ,\tag{9}
$$

$$
\begin{array} { r } { \sigma _ { i } ( t ) = \sigma _ { i } ^ { \tau } \mathrm { e x p } \left( - s _ { i } ^ { \tau } | t - \mu _ { i } ^ { \tau } | ^ { 2 } \right) , } \end{array}\tag{10}
$$

where $\mu _ { i } ( t )$ and $r _ { i } ( t )$ explicitly encode trajectories of position and rotation through polynomial interpolation. Here, $b _ { i , k }$ and $c _ { i , k }$ denote the polynomial coefficients that are optimized during training, while the temporal center $\mu _ { i } ^ { \tau }$ defines the central time around which each Gaussianâs trajectory is modeled. The opacity $\sigma _ { i } ( t )$ is formulated as a Gaussian-shaped temporal function, parameterized by a spatial opacity factor ${ \sigma } _ { i } ^ { \bar { \tau } }$ , a temporal scaling factor $s _ { i } ^ { \tau } ,$ , and the temporal center $\mu _ { i } ^ { \tau }$ . For simplicity and computational efficiency, both the scale $s _ { i }$ and color $c _ { i }$ are treated as time-invariant, inspired by [19].

Instead of relying on implicit motion modeling, such as deformation fields [2, 14], which comes with a heavy parameter burden, our explicit modeling method uses fewer parameters while ensuring comparable rendering results.

## 4.2.3 Parameter Sharing

To reduce parameter redundancy and improve model compactness, we implement a dedicated parameter-sharing strategy. Specifically, during initialization, static Gaussian primitives are generated directly from the SfM point clouds. For each static primitive $G _ { i }$ in the scene, we explicitly set the temporal polynomial coefficients $b _ { i , k }$ $( k = 1 , 2 , 3 )$ , the first-order rotation coefficient $c _ { i , 1 } ,$ , and the temporal scaling factor $s _ { i } ^ { \tau }$ to zero, and set the temporal center $\mu _ { i } ^ { \tau }$ to one-half. Thus, while static primitives formally share the same parametric structure as dynamic counterparts, they effectively degenerate into temporally invariant primitives.

Since these parameters are identical across all static primitives, it is unnecessary to store them individually during model serialization. Consequently, our parameter-sharing strategy substantially reduces parameter redundancy, significantly decreasing the model size while preserving representational accuracy and rendering efficiency.

By combining temporally fixed parameter sharing for static primitives with flexible temporal modeling for dynamic primitives, our SDD strategy achieves significant parameter reduction while retaining the flexibility to represent complex scene dynamics. The fixed parameter sharing for static primitives also eliminates crossregion temporal interference, preserving high-frequency details and visual fidelity within static content.

## 4.3 Two-stage Training

Although the SDD strategy provides a compact and efficient representation, the explicit separation of static and dynamic regions can introduce artifacts near their boundaries. To address this issue, we propose a novel two-stage training strategy, specifically designed to facilitate smoother transitions and enhance consistency across static-dynamic boundaries. Our approach encourages dynamic primitives to better fit and adapt to static regions, significantly alleviating boundary artifacts and ensuring overall scene coherence.

During training, we decompose the Hybrid Gaussians into static and dynamic ones. When Gaussian primitives are split or generated during optimization, they inherit the static/dynamic labels of their parent primitives, ensuring coherence throughout the training process.

Within each optimization cycle, our two-stage training proceeds as follows:

## 4.3.1 Stage 1: Static Optimization

We first optimize only the parameters of the static primitives. Specifically, we render the hybrid Gaussians derived from the previous optimization iteration and compute the rendering loss between the rendered images and the ground truth images. This loss is utilized to update the parameters of static Gaussian primitives exclusively, stabilizing the static representation and refining scene geometry and appearance.

## 4.3.2 Stage 2: Dynamic Optimization

Subsequently, we merge the newly updated static primitives from Stage 1 with the previously optimized dynamic primitives, forming updated hybrid Gaussians. Rendering these hybrid Gaussians again, we compute the loss relative to the ground-truth images and use this loss to update only the parameters of the dynamic Gaussian primitives. Crucially, in this stage, the dynamic primitives adapt to the optimized static representation, further improving consistency at static-dynamic boundaries.

Compared to training static and dynamic primitives separately in an isolated manner, our sequential two-stage training approach effectively mitigates the boundary discontinuities and temporal blurring artifacts at the interfaces between static and dynamic regions. Although this iterative optimization slightly increases the total training time, it significantly enhances the seamless integration between static and dynamic primitives, leading to a coherent, high-quality reconstruction of dynamic scenes.

## 4.4 Optimization

With the modeling framework and two-stage training strategy in place, we now present the optimization of HGS parameters, performed via differentiable rasterization and gradient-based optimization. Due to the distinct nature of static and dynamic primitives in our framework, their parameter sets differ accordingly. Specifically, the optimization parameters for each static primitive include $\sigma _ { i } ^ { \tau } , \ b _ { i , 0 } , \ c _ { i , 0 } , \ s _ { i } ,$ and $c _ { i } .$ For each dynamic primitive, we optimize parameters controlling temporal variation, namely $\sigma _ { i } ^ { \tau } , s _ { i } ^ { \tau } , \bar { \mu } _ { i } ^ { \tau }$ polynomial trajectory coefficients $\{ b _ { i , k } \} _ { k = 0 } ^ { 3 } ,$ rotation coefficients $\{ c _ { i , k } \} _ { k = 0 } ^ { 1 } ,$ as well as anisotropic scale vector $s _ { i }$ and color coefficients $c _ { i } .$

The optimization process interleaves gradient-based parameter updates with Gaussian density control strategies such as splitting and pruning, ensuring that our model remains both compact and expressive throughout training.

Following previous methods [16], we define the rendering loss as a combination of an $L _ { 1 }$ photometric term and a structural similarity index (D-SSIM) term, which robustly captures visual differences between rendered and ground-truth images. Given a rendered image $I _ { \mathrm { r e n d e r } }$ and its corresponding ground-truth image $I _ { \mathrm { { g t } } }$ , the rendering loss $\mathcal { L }$ is formulated as follows:

$$
\mathcal { L } ( I _ { \mathrm { r e n d e r } } , I _ { \mathrm { g t } } ) = \Vert I _ { \mathrm { r e n d e r } } - I _ { \mathrm { g t } } \Vert _ { 1 } + \lambda \cdot \mathcal { L } _ { \mathrm { D - S S I M } } ( I _ { \mathrm { r e n d e r } } , I _ { \mathrm { g t } } ) ,\tag{11}
$$

where Î» balances the contribution between the two terms, and LD-SSIM denotes the D-SSIM loss term.

During training, we alternately optimize static and dynamic primitives following the proposed two-stage optimization strategy (detailed in Sec. 4.3), enabling effective modeling of dynamic interactions between scene components and improved consistency at static-dynamic boundaries.

Table 1: Quantitative comparisons on the Neural 3D Video Dataset.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>Storage(MB)â</td><td>Training timeâ</td><td>FPSâ</td></tr><tr><td>K-Planes [10]</td><td>31.39</td><td>0.944</td><td>311</td><td>-</td><td>0.3</td></tr><tr><td>MixVoxels [39]</td><td>30.55</td><td>0.924</td><td>508.07</td><td>1h 22m</td><td>16.7</td></tr><tr><td>HyperReel [1]</td><td>31.65</td><td>0.934</td><td>169.47</td><td>1h 41m</td><td>2.00</td></tr><tr><td>4DGS [41]</td><td>32.02</td><td>0.946</td><td>29.36</td><td>31m</td><td>114</td></tr><tr><td>ED3DGS [2]</td><td>32.35</td><td>0.949</td><td>31.41</td><td>1h 35m</td><td>75</td></tr><tr><td>STGS [21]</td><td>32.32</td><td>0.953</td><td>18.19</td><td>29m</td><td>140</td></tr><tr><td>SaRO-GS [43]</td><td>33.06</td><td>0.955</td><td>236.44</td><td>1h 31m</td><td>40</td></tr><tr><td>Ours</td><td>32.36</td><td>0.952</td><td>6.87</td><td>18m</td><td>300+</td></tr></table>

## 5 EXPERIMENTS

## 5.1 Experimental Setup

## 5.1.1 Implementation Details

We employ the Track Anything Model (TAM) [44] to perform explicit separation between static and dynamic regions in the input videos. The resulting segmentation masks from TAM are saved and utilized to prepare our datasets for training. According to [31], we employ 3DGS without the harmonic components for color to further reduce the number of parameters and achieve high-quality rendering. For optimization, we use the Adam optimizer [17]. All experiments and training are conducted on a single NVIDIA RTX 4090 GPU (24GB), except real-time rendering, which is performed on an RTX 3090 GPU (24GB).

## 5.1.2 Datasets.

We evaluate our approach quantitatively and qualitatively on two public datasets: the Neural 3D Video Dataset [20] and the Google Immersive Dataset [4]. The Neural 3D Video Dataset, captured using 19â21 cameras at 2704Ã2028 resolution and 30 FPS, contains dynamic scenes with complex motions and variable object appearances, posing challenges for consistent reconstruction. The Google Immersive Dataset, recorded with a 44â46 camera fisheye rig, includes indoor and outdoor scenes with abrupt transitionsâsuch as the sudden appearance of a large fire sourceâfurther complicating multi-view reconstruction. Owing to the inherent challenges of the Google Immersive Dataset, only a few works have attempted comparative evaluations on it.

## 5.1.3 Comparison Methods.

We compare our method with recent dynamic NeRF-based and Gaussian-based approaches for dynamic scene reconstruction from multi-view videos. The NeRF-based baselines include K-Planes [10], MixVoxels [39], HyperReel [1], and Nerf-Player [36], while the Gaussian-based baselines include 4DGS [41], ED3DGS [2], STGS [21] and SaRO-GS [43]. For all comparisons, we use the official implementations and keep the original hyperparameters unchanged.

## 5.2 Results

## 5.2.1 Quantitative Comparison

As shown in Tab. 1, our method achieves a PSNR of 32.36 dB and an SSIM of 0.952, outperforming strong baselines [1, 10, 39, 36, 41, 2] while attaining visual quality comparable to STGS [21] and offering significantly improved efficiency. Although the rendering quality is marginally lower than that of SaRO-GS [43], our approach offers notable benefits in terms of model compactness, training efficiency, and rendering speed. In particular, our model requires only 6.87 MB of storage, marking a 63% reduction compared to STGS [21] and up to a 98% reduction relative to MixVoxels [39]. These results attest to the effectiveness of our design in achieving a compact yet efficient dynamic scene representation without sacrificing visual quality. Furthermore, our method consistently performs well across all test scenes, as detailed in the supplementary material.

<!-- image-->  
Figure 3: Qualitative results comparison on Neural 3D Video Dataset. The red and blue highlight areas where the proposed method achieves notable visual quality improvements.

Table 2: Quantitative comparisons on the Google Immersive Dataset.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>Storage(MB)â</td><td>FPSâ</td></tr><tr><td>NeRFPlayer [36]</td><td>27.23</td><td>0.887</td><td>5130</td><td>0.12</td></tr><tr><td>HyperReel [1]</td><td>29.09</td><td>0.899</td><td>190.68</td><td>4</td></tr><tr><td>STGS [21]</td><td>28.87</td><td>0.943</td><td>19.74</td><td>99</td></tr><tr><td>Ours</td><td>29.60</td><td>0.925</td><td>12.73</td><td>300+</td></tr></table>

In terms of computational efficiency, our extensive experiments show that HGS delivers exceptional real-time performance, achieving speed over 300 FPS at a resolution of 1352 Ã 1014 on a single NVIDIA RTX 3090 GPU, far surpassing STGS [21] (140 FPS) and ED3DGS [2] (75 FPS). Moreover, HGS significantly improves training efficiency by completing training in only 18 minutes, considerably faster than competing methods such as SaRO-GS [43], which requires 1 hour and 31 minutes. This efficient pipeline further highlights the practicality and scalability of our approach for real-world dynamic scene applications.

Tab. 2 summarizes quantitative results on the Google Immersive Dataset. Characterized by abrupt scene transitions, our method attains a competitive PSNR of 29.60 dB and an SSIM of 0.925, outperforming baselines like HyperReel [1] and STGS [21]. To mitigate artifacts arising from sudden transitions, our approach supplements Gaussians, resulting in a modest increase in model size. Detailed visual results are provided in the qualitative comparison for further insight.

## 5.2.2 Qualitative Comparison.

To qualitatively evaluate the visual fidelity of our proposed method, we conduct comparisons on the two aforementioned datasets. On the Neural 3D Video Dataset, we visualize reconstruction results from two representative scenes, emphasizing fidelity in highfrequency texture regions. As shown in Fig. 3, compared to the two baseline methods [1, 41], our method produces reconstructions significantly closer to the ground truth, particularly preserving intricate textures such as the detailed surface of meat and the complex layering of vegetable leaves. This confirms our methodâs effectiveness in addressing the commonly observed issue of missing highfrequency details in static regions.

<!-- image-->  
Figure 4: Qualitative results comparison on Google Immersive Dataset. The red and blue highlight areas where the proposed method achieves notable visual quality improvements.

Table 3: Quantitative results of the ablation study on our framework, reported as averages over the Neural 3D Video Dataset.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>Storage(MB)â</td></tr><tr><td>w/o Initialization</td><td>32.5</td><td>0.954</td><td>13.51</td></tr><tr><td>w/o Parameter Sharing</td><td>32.22</td><td>0.951</td><td>11.56</td></tr><tr><td>w/o Two-stage Training</td><td>25.76</td><td>0.910</td><td>5.88</td></tr><tr><td>Ours</td><td>32.36</td><td>0.952</td><td>6.87</td></tr></table>

On the Google Immersive Dataset, we focus on artifacts induced by abrupt scene changes. As shown in Fig. 4, when a sudden flame brightens the scene, STGS [21] fails to capture the brightness accurately, resulting in artifacts in static regions. Similar issues occur in the reconstruction of moving objects for both STGS [21] and HyperReel [1]. In contrast, our method mitigates these artifacts, handling sudden illumination shifts and abrupt content changes.

Overall, our qualitative analysis shows that our approach preserves high-frequency texture details and mitigates artifacts from abrupt scene transitions, resulting in superior visual quality. For more qualitative comparisons with state-of-the-art methods, please refer to the appendix.

## 5.3 Ablation Study

To investigate the effectiveness of individual components within our framework, we conduct comprehensive ablation studies summarized in Tab. 3. In the following, we describe the configuration and performance of each ablation baseline.

Initialization: Omitting our interleaved-frame initialization increases storage size further to 13.51 MB. While PSNR slightly improves, this minor gain is outweighed by a substantial increase in model complexity. This highlights that our initialization strategy provides an optimal balance between rendering quality and model compactness.

Parameter Sharing: Removing the parameter-sharing strategy results in an increase in storage from 6.87 MB to 11.56 MB, nearly doubling the parameter count without any significant improvement in visual metrics. This underscores the effectiveness of our parameter-sharing strategy in dramatically reducing redundancy.

Two-stage Training: Removing our two-stage training strategy (i.e., training static and dynamic Gaussians separately without mutual adaptation) significantly reduces rendering quality, with PSNR dropping from 32.36 dB to 25.76 dB and SSIM from 0.952 to 0.910. This decline underscores the critical role of mutual adaptation between static and dynamic regions, facilitated by our two-stage strategy, in enhancing temporal consistency and reducing boundary artifacts. Although this iterative optimization slightly increases model size, it is justified by the considerable visual improvements demonstrated in Fig. 7.

## 5.3.1 RBF Order

To justify our choice of a third-order RBF, we conducted experiments with different polynomial orders. The results are summarized in the Tab. 4. The results demonstrate that selecting an RBF order of 3 provides the best overall balance. Compared to order 2, it achieves higher PSNR without sacrificing FPS, while only slightly increasing storage. Compared to order 5, it delivers nearly the same PSNR but with significantly lower storage requirements and faster FPS. Therefore, order 3 offers the most practical trade-off between reconstruction quality, efficiency, and resource consumption, justifying its use as the default choice in our method.

<!-- image-->  
Figure 5: Qualitative results of the Training Strategies for Static and Dynamic Regions.

<!-- image-->  
Figure 6: Qualitative results of the Analysis of Dynamic Region Proportion. The 7.8% setting corresponds to the dynamic region obtained through standard static-dynamic decomposition, while the 95% setting is generated by dilating the dynamic mask to artificially expand the dynamic area.

<!-- image-->  
Figure 7: Qualitative results of the ablation study.

## 5.3.2 Variants of Training Strategies for Static and Dynamic Regions

For training static and dynamic regions, there are three intuitive designs: training them separately, training them together (i.e., updating the parameters of static and dynamic primitives simultaneously), and our proposed alternating two-stage strategy. We conducted experiments under all three settings to validate the effectiveness of our design. The results are shown in Tab. 5. First, training static and dynamic regions separately leads to severe artifacts and a notable drop in quality. The results also clearly demonstrate that training static and dynamic blobs simultaneously is inefficient: the train-together strategy nearly doubles the storage cost while providing no gain in PSNR and even slightly underperforming the two-stage approach. In contrast, the two-stage strategy achieves higher PSNR with less storage and faster convergence while maintaining real-time FPS. This indicates that gradient interference between static and dynamic blobs hampers convergence when trained together, leading to inefficiency and potential overfitting. Therefore, adopting a two-stage training scheme is a more effective and practical solution. The qualitative results are shown in Fig. 5.

Table 4: Results of different RBF orders evaluated on the Neural 3D Video Dataset.
<table><tr><td>Order</td><td>PSNRâ</td><td>Storage(MB)â</td><td>Training timeâ</td><td>FPSâ</td></tr><tr><td>2</td><td>32.16</td><td>6.74</td><td>18m</td><td>300+</td></tr><tr><td>3 (Ours)</td><td>32.36</td><td>6.87</td><td>18m</td><td>300+</td></tr><tr><td>5</td><td>32.44</td><td>7.61</td><td>18m</td><td>290</td></tr></table>

Table 5: We tested the performance of three training methods on the Neural 3D Video Dataset.
<table><tr><td>Method</td><td>PSNRâ</td><td>Storage(MB)â</td><td>Training timeâ</td><td>FPSâ</td></tr><tr><td>train-separate</td><td>25.76</td><td>5.88</td><td>18m</td><td>300+</td></tr><tr><td>train-together</td><td>32.33</td><td>12.01</td><td>18m</td><td>300+</td></tr><tr><td>two-stage (Ours)</td><td>32.36</td><td>6.87</td><td>18m</td><td>300+</td></tr></table>

Table 6: Evaluation of the impact of increasing dynamic region proportion on the âSear Steakâ scene through progressive expansion of the dynamic mask.
<table><tr><td>Proportion</td><td>PSNRâ</td><td>LPIPSâ</td><td>Storage(MB)â</td><td>Training timeâ</td><td>FPSâ</td></tr><tr><td>7.8% (Ours)</td><td>33.34</td><td>0.032</td><td>4.77</td><td>18m 31s</td><td>149</td></tr><tr><td>20%</td><td>33.32</td><td>0.032</td><td>5.35</td><td>18m 30s</td><td>136</td></tr><tr><td>40%</td><td>33.44</td><td>0.032</td><td>6.18</td><td>19m 06s</td><td>130</td></tr><tr><td>60%</td><td>33.95</td><td>0.034</td><td>6.88</td><td>18m 11s</td><td>131</td></tr><tr><td>80%</td><td>33.61</td><td>0.033</td><td>7.73</td><td>19m 07s</td><td>131</td></tr><tr><td>95%</td><td>33.85</td><td>0.032</td><td>8.30</td><td>19m 21s</td><td>132</td></tr></table>

<!-- image-->  
Figure 8: Real-time rendering performance on resource-constrained devices, validated on a laptop equipped with an RTX 3050 GPU.

## 5.4 Effect of Dynamic Region Proportion on Performance

We assess the impact of dynamic region proportions on the performance of our method. To further highlight performance differences under the refresh-rate limitation of conventional displays, we increase the rendering resolution, which results in distinguishable variations in frame rates. Starting from the baseline where our method identifies only 7.8% of the scene as dynamic, we gradually expand this proportion up to 95% to stress-test performance. As shown in Tab. 6, our method maintains stable PSNR and low LPIPS across varying dynamic proportions, reflecting reliable quantitative performance. However, when dynamic modeling is applied to regions that are inherently static, we observe that the slight numerical gains come at the cost of perceptual quality: artifacts such as dynamic blur and temporal flickering emerge, which noticeably reduce visual stability during immersive viewing. These artifacts not only increase storage demands but also distract users in real-time rendering, where consistent and artifact-free perception is far more critical than marginal improvements in objective scores. Representative examples are provided in the Fig. 6. Additional demonstrations can be found in the supplementary video materials.

## 5.5 Real-Time Rendering on Resource-Constrained Devices

Prior works typically evaluate efficiency on high-end GPUs, neglecting the broader applicability of their methods in resourceconstrained environments. In practice, however, the majority of users rely on mid- or low-tier devices, where achieving real-time rendering is far more challenging. To this end, we validate our framework on a laptop equipped with an RTX 3050 GPU, achieving real-time 1352 Ã 1014 rendering as shown in Fig. 8. This result highlights the practical advantage of our method, which remains efficient even under strict hardware constraints. Regrettably, existing baselines do not provide real-time rendering players, making a direct runtime comparison impossible. Nevertheless, under the same resolution, our rendering frame rate on the RTX 3050 already surpasses that of all existing methods running on an RTX 3090, clearly demonstrating the superior performance of our approach on resource-limited devices.

<!-- image-->  
Figure 9: Real-time rendering performance on VR devices.

## 5.6 Real-Time Rendering in VR Systems

Compared to conventional 2D rendering, VR rendering imposes far stricter computational demands, since it requires higher resolutions and stereo rendering for both eyes. These requirements place a heavy burden on the rendering pipeline. Existing dynamic-Gaussian methods, with their large model sizes and high complexity, struggle to meet such constraints and, to our knowledge, have not demonstrated VR applications.

By contrast, our improved representation substantially reduces model complexity and computational cost, making real-time rendering feasible under VR conditions. As shown in Fig. 9, we develop a dedicated VR playback system that streams dynamic 3D Gaussian scenes directly into head-mounted displays (HMDs). This enables users to interactively explore dynamic scenes in immersive VR, moving beyond the limitations of 2D monitor demonstrations.

A discrepancy between the Unity and OpenGL coordinate systems leads to mirroring artifacts in our rendered scenes. Furthermore, while necessary optimizations are applied to adapt the 3DGS format for Unity rendering, these adjustments inevitably introduce aliasing, resulting in slightly lower visual fidelity compared to conventional 2D rendering.

## 6 LIMITATIONS AND FUTURE DIRECTIONS

While HGS demonstrates substantial improvements, several open challenges remain as natural extensions of our work.

Segmentation dependency. Our method partially relies on the accuracy of staticâdynamic separation. The precision of existing approaches is sufficient for our needs; nevertheless, we conducted additional tests to verify the robustness of our framework. We explicitly evaluated three possible error cases: minor boundary inaccuracies (mitigated by our two-stage training, Tab. 3, Fig. 7), static regions mislabeled as dynamic (negligible impact, Tab. 6), and dynamic regions mislabeled as static (0.8 dB PSNR drop, alleviated by our strategy with optional manual correction). These results suggest that HGS is robust to typical segmentation errors, and future work may integrate adaptive segmentation refinement.

Temporal motion modeling. As a local, kernel-based approach, explicit temporal RBF modeling may face challenges in capturing highly complex non-rigid motions. To address this, we employ a two-stage training strategy that allows dynamic Gaussians to appear or disappear over time, which proves effective in practice. Future extensions could combine RBF with more expressive formulations to better capture fine-grained deformations.

Training strategy. Our two-stage training design suppresses boundary artifacts and improves convergence stability. Although this introduces modest computational overhead, the trade-off is beneficial for quality, and exploring more streamlined training schedules will further enhance scalability to large-scale or ultra high-resolution scenarios.

## 7 CONCLUSION

We propose Hybrid Gaussian Splatting (HGS), a novel framework for dynamic scene reconstruction. By decomposing scenes into static and dynamic regions within a unified representation, HGS reduces parameter redundancy and improves computational efficiency. Our static and dynamic decomposition (SDD) strategy shares compact, temporally invariant parameters for static primitives, while a two-stage training strategy improves consistency at the boundaries between static and dynamic regions. Experiments demonstrate that HGS delivers comparable or superior rendering quality to state-of-the-art methods, while reducing model size by up to 98% and achieving real-time 4K rendering at 125 FPS on an RTX 3090. It further sustains 160 FPS at 1352Ã1014 on an RTX 3050, and has been successfully integrated into the VR system, corroborating its practicality for immersive applications. Qualitative results further demonstrate that HGS preserves fine textures and mitigates artifacts caused by abrupt scene changes, setting a new benchmark for efficient, high-fidelity dynamic view synthesis.

## ACKNOWLEDGMENTS

The authors wish to thank A, B, and C. This work was supported in part by a grant from XYZ.

## REFERENCES

[1] B. Attal, J.-B. Huang, C. Richardt, M. Zollhoefer, J. Kopf, M. OâToole, and C. Kim. Hyperreel: High-fidelity 6-dof video with ray-conditioned sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16610â16620, 2023. 1, 2, 5, 6, 7

[2] J. Bae, S. Kim, Y. Yun, H. Lee, G. Bang, and Y. Uh. Per-gaussian embedding-based deformation for deformable 3d gaussian splatting. In European Conference on Computer Vision, pp. 321â335. Springer, 2024. 1, 2, 4, 5, 6

[3] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan. Mip-nerf: A multiscale representation for antialiasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5855â5864, 2021. 1

[4] M. Broxton, J. Flynn, R. Overbeck, D. Erickson, P. Hedman, M. Duvall, J. Dourgarian, J. Busch, M. Whalen, and P. Debevec. Immersive light field video with a layered mesh representation. ACM Transactions on Graphics (TOG), 39(4):86â1, 2020. 5

[5] A. Cao and J. Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 130â141, 2023. 2

[6] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su. Tensorf: Tensorial radiance fields. In European conference on computer vision, pp. 333â 350. Springer, 2022. 1

[7] Z. Chen, Z. Li, L. Song, L. Chen, J. Yu, J. Yuan, and Y. Xu. Neurbf: A neural fields representation with adaptive radial basis functions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4182â4194, 2023. 1

[8] Y. Duan, F. Wei, Q. Dai, Y. He, W. Chen, and B. Chen. 4d-rotor gaussian splatting: towards efficient novel view synthesis for dynamic scenes. In ACM SIGGRAPH 2024 Conference Papers, pp. 1â11, 2024. 1, 2

[9] J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. NieÃner, and Q. Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH Asia 2022 Conference Papers, pp. 1â9, 2022. 2

[10] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12479â12488, 2023. 1, 2, 5

[11] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5501â5510, 2022. 1

[12] X. Guo, J. Sun, Y. Dai, G. Chen, X. Ye, X. Tan, E. Ding, Y. Zhang, and J. Wang. Forward flow for novel view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16022â16033, 2023. 2

[13] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi. Scgs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4220â4230, 2024. 1, 2

[14] Y. Jeong, J. Lee, H. Choi, and M. Cho. Rodygs: Robust dynamic gaussian splatting for casual videos. arXiv preprint arXiv:2412.03077, 2024. 2, 4

[15] K. Katsumata, D. M. Vo, and H. Nakayama. An efficient 3d gaussian representation for monocular/multi-view dynamic scenes. (No Title), 2023. 1

[16] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis. 3d gaussian Â¨ splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 5

[17] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 5

[18] A. Kratimenos, J. Lei, and K. Daniilidis. Dynmf: Neural motion factorization for real-time dynamic view synthesis with 3d gaussian splatting. In European Conference on Computer Vision, pp. 252â269. Springer, 2024. 1, 2

[19] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21719â 21728, 2024. 1, 4

[20] T. Li, M. Slavcheva, M. Zollhoefer, S. Green, C. Lassner, C. Kim, T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5521â5531, 2022. 5

[21] Z. Li, Z. Chen, Z. Li, and Y. Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8508â8520, 2024. 1, 2, 4, 5, 6, 7

[22] Z. Li, S. Niklaus, N. Snavely, and O. Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6498â6508, 2021. 2

[23] Y. Liang, N. Khan, Z. Li, T. Nguyen-Phuoc, D. Lanman, J. Tompkin, and L. Xiao. Gaufre: Gaussian deformation fields for real-time dynamic novel view synthesis. arXiv preprint arXiv:2312.11458, 2023. 1, 2

[24] Y. Lin, Z. Dai, S. Zhu, and Y. Yao. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21136â 21145, 2024. 1, 2

[25] L. Liu, J. Gu, K. Zaw Lin, T.-S. Chua, and C. Theobalt. Neural sparse voxel fields. Advances in Neural Information Processing Systems, 33:15651â15663, 2020. 1

[26] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai. Scaffoldgs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20654â20664, 2024. 1

[27] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 2024 International Conference on 3D Vision (3DV), pp. 800â809. IEEE, 2024. 2

[28] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[29] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert. Compact 3d scene representation via self-organizing gaussian grids. In European Conference on Computer Vision, pp. 18â34. Springer, 2024. 1

[30] T. Muller, A. Evans, C. Schied, and A. Keller. Instant neural graphics Â¨ primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022. 1

[31] K. Navaneet, K. Pourahmadi Meibodi, S. Abbasi Koohpayegani, and H. Pirsiavash. Compgs: Smaller and faster gaussian splatting with vector quantization. In European Conference on Computer Vision, pp. 330â349. Springer, 2024. 5

[32] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz, and R. Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5865â5874, 2021. 2

[33] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, and S. M. Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021. 2

[34] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer. Dnerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10318â10327, 2021. 2

[35] R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, and Y. Liu. Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16632â16642, 2023. 2

[36] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y. Xu, and A. Geiger. Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE Transactions on Visualization and Computer Graphics, 29(5):2732â2742, 2023. 2, 5, 6

[37] C. Sun, M. Sun, and H.-T. Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5459â5469, 2022. 1

[38] E. Tretschk, A. Tewari, V. Golyanik, M. Zollhofer, C. Lassner, and Â¨ C. Theobalt. Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 12959â12970, 2021. 2

[39] F. Wang, S. Tan, X. Li, Z. Tian, Y. Song, and H. Liu. Mixed neural voxels for fast multi-view video synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19706â 19716, 2023. 1, 2, 5

[40] L. Wang, Q. Hu, Q. He, Z. Wang, J. Yu, T. Tuytelaars, L. Xu, and M. Wu. Neural residual radiance fields for streamably free-viewpoint videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 76â87, 2023. 2

[41] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20310â20320, 2024. 1, 2, 5, 6

[42] W. Xian, J.-B. Huang, J. Kopf, and C. Kim. Space-time neural irradiance fields for free-viewpoint video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9421â 9431, 2021. 2

[43] J. Yan, R. Peng, L. Tang, and R. Wang. 4d gaussian splatting with scale-aware residual field and adaptive optimization for real-time rendering of temporally complex dynamic scenes. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 7871â7880, 2024. 1, 5, 6

[44] J. Yang, M. Gao, Z. Li, S. Gao, F. Wang, and F. Zheng. Track anything: Segment anything meets videos. arXiv preprint arXiv:2304.11968, 2023. 5

[45] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20331â20341, 2024. 1, 2

[46] Z. Yang, H. Yang, Z. Pan, and L. Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642, 2023. 1, 2

[47] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern