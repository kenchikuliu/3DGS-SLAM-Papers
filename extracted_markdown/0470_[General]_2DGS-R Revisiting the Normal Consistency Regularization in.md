# 2DGS-R: Revisiting the Normal Consistency Regularization in 2D Gaussian Splatting

Haofan Ren1, Qingsong Yan2, Ming Lu3, Rongfeng Lu1, Zunjie Zhu1

1Hangzhou Dianzi University 2Wuhan University 3Intel Labs China

## Abstract

Recent advancements in 3D Gaussian Splatting (3DGS) have greatly influenced neural fields, as it enables high-fidelity rendering with impressive visual quality. However, 3DGS has difficulty accurately representing surfaces. In contrast, 2DGS transforms the 3D volume into a collection of 2D planar Gaussian disks. Despite advancements in geometric fidelity, rendering quality remains compromised, highlighting the challenge of achieving both high-quality rendering and precise geometric structures. This indicates that optimizing both geometric and rendering quality in a single training stage is currently unfeasible. To overcome this limitation, we present 2DGS-R, a new method that uses a hierarchical training approach to improve rendering quality while maintaining geometric accuracy. 2DGS-R first trains the original 2D Gaussians with the normal consistency regularization. Then 2DGS-R selects the 2D Gaussians with inadequate rendering quality and applies a novel in-place cloning operation to enhance the 2D Gaussians. Finally, we fine-tune the 2DGS-R model with opacity frozen. Experimental results show that compared to the original 2DGS, our method requires only 1% more storage and minimal additional training time. Despite this negligible overhead, it achieves high-quality rendering results while preserving fine geometric structures. These findings indicate that our approach effectively balances efficiency with performance, leading to improvements in both visual fidelity and geometric reconstruction accuracy.

## Introduction

Novel view synthesis (NVS) and 3D surface reconstruction have consistently presented challenges in computer vision and graphics. Recently, 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) has emerged as a promising approach for NVS, achieving an effective balance between rendering quality and real-time performance. As a result, it rapidly achieved wide application in diverse fields, including dynamic scene reconstruction (Duan et al. 2024; Bae et al. 2024; Lu et al. 2024b), autonomous driving (Zhou et al. 2024a,b), and SLAM (Yan et al. 2024; Keetha et al. 2024; Matsuki et al. 2024). However, the multiview inconsistency of 3D Gaussians lacks a clear definition of surfaces, resulting in difficulties in extracting high-quality meshes.

To obtain better geometry reconstruction, subsequent work like SuGaR (Guedon and Lepetit 2024) forces the 3D Â´ Gaussians to align with the surface using geometry regularization. Another type of method, like 2DGS (Huang et al. 2024) and GaussianSurfels (Dai et al. 2024), utilizes the flatten primitive to fit the surface. While these methods enhance the quality of geometric reconstruction, they inevitably result in a slight decrease in rendering quality compared to the original. This observation reflects the ongoing challenge of reconciling rendering quality with geometric accuracy in neural field-based representations.

<!-- image-->  
Figure 1: (a) Rendered color. (b) Rendered normal. (c) Normal from rendered depth. Our motivation for this work is to achieve high-quality rendering and precise surface reconstruction for 2DGS (Huang et al. 2024) at the same time. Normal consistency regularization (NC) means minimizing the difference between (b) and (c).

Among all reconstruction methods based on Gaussian, 2DGS exhibits excellent simplicity and elegance. Compared with the 3DGS, 2DGS utilizes 2D-to-2D projection with homogeneous coordinates to avoid the unprecise approximation (Zwicker et al. 2004). It improves the quality of geometric reconstruction with a small training time load. We find that geometric regularization, particularly normal consistency, is nearly essential for high-quality geometric reconstruction, but it negatively impacts rendering quality. The results on DTU demonstrate that incorporating normal consistency improves reconstruction accuracy by 46% compared to cases without it, albeit with a 0.8 dB reduction in PSNR. We propose a novel and effective training strategy that significantly improves rendering quality while preserving geometric reconstruction accuracy. Our key contributions are summarized as follows:

â¢ We conduct extensive experiments to assess the impact of incorporating normal consistency (NC) on the 2DGS attributes. Based on our findings, we propose a hierarchical training strategy that dynamically adjusts different 2DGS attributes throughout the training process.

â¢ We analyze the limitations of directly adjusting 2DGS properties and introduce in-place clone densification as an effective complementary strategy to further enhance rendering quality, especially in regions with fine details or abrupt color changes.

â¢ By combining the two methods outlined above, we effectively mitigate the trade-off between rendering quality and reconstruction accuracy, with minimal impact on storage and computational load.

## Related Work

## Novel View Synthesis

NeRF (Mildenhall et al. 2020) leverages a multi-layer perceptron (MLP) for scene representation, encoding both geometry and view-dependent appearance information. Through volume rendering (Max 1995), the MLP is optimized with a photometric loss function. Subsequent advancements have focused on optimizing NeRFâs training through feature-grid representations (Yu et al. 2021; Fridovich-Keil et al. 2022; Muller et al. 2022; Chen et al.Â¨ 2022) and enhancing rendering speed via baking (Hedman et al. 2021; Reiser et al. 2021; Tang et al. 2023). Besides, NeRF has also been adapted to address anti-aliasing (Barron et al. 2023; Hu et al. 2023; Barron et al. 2021) and unbounded scenes (Barron et al. 2022; Zhang et al. 2020).

More recently, 3D Gaussian splatting (Kerbl et al. 2023) models complex scenes using 3D Gaussians. It achieved outstanding results, with efficient optimization and the capability to render high-resolution images in real-time rendering. Subsequent works improved its storage efficiency (Chen et al. 2024b; Navaneet et al. 2023; Niedermayr, Stumpfegger, and Westermann 2024; Wang et al. 2024) or rendering efficiency (Jo, Kim, and Park 2024; Lee et al. 2024). Moreover, some work (Lu et al. 2024a; Cheng et al. 2024) attempts to improve its rendering quality further. Besides, it also has been extended to surface reconstruction (Yu, Sattler, and Geiger 2024; Guedon and Lepetit 2024). Â´

## 3D reconstruction

3D reconstruction from multi-view images is a fundamental problem in computer vision. Multi-view stereo methods (Schonberger et al. 2016; Yao et al. 2018; Yu and Â¨ Gao 2020) often involve intricate, multi-stage processing pipelines. These typically include feature matching, depth estimation, point cloud fusion, and surface reconstruction. Despite their widespread application in academia and industry, these methods are susceptible to artifacts arising from incorrect feature matching and noise introduced at various pipeline stages. In contrast, neural surface reconstruction has leveraged pure deep neural networks to predict surface models directly from multiple image conditions in an endto-end manner. However, these methods typically involve substantial computational overhead during network inference and require extensively labeled 3D training models, limiting their real-time and practical applicability. While 3DGS (Kerbl et al. 2023) benefits from an explicit scene representation that accelerates training and rendering speed, the absence of well-defined boundaries adversely affects geometric reconstruction quality.

To resolve this challenge, several methods have been introduced, including SuGaR (Guedon and Lepetit 2024), Â´ GOF (Yu, Sattler, and Geiger 2024), Gaussian Surfels (Dai et al. 2024), and 2DGS (Huang et al. 2024). 2D Gaussian Splatting (2DGS) (Huang et al. 2024) has recently gained attention as a novel technique that simplifies 3D scene representation by converting volumetric data into 2D oriented Gaussian disks. 2DGS offers enhanced geometric reconstruction performance over 3DGS while maintaining efficiency. However, 2DGS exhibits limitations in rendering quality, as reflected by both qualitative and quantitative evaluations. In response to this limitation, we propose a novel training approach for 2DGS that significantly improves rendering quality without compromising geometric accuracy.

## Method

## Preliminaries

Given central position $\mu ,$ a scaling vector $\boldsymbol { S } = ( s _ { u } , s _ { v } )$ that governs the covariance of a 2D Gaussian (Huang et al. 2024) and a $3 \times 3$ rotation matrix $\boldsymbol { R } = [ t _ { u } , t _ { v } , t _ { w } ]$ that defines the orientation of the 2D Gaussian, the transformation between UV space and world space can be expressed as follows:

$$
H = \left[ \begin{array} { c c c c } { s _ { u } t _ { u } } & { s _ { v } t _ { v } } & { \mathbf { 0 } } & { p } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} \right] = \left[ \begin{array} { c c } { R S } & { \mu } \\ { \mathbf { 0 } } & { 1 } \end{array} \right]\tag{1}
$$

If W is marked as the transformation matrix from world space to screen space, a homogeneous ray emitted from the camera and passing through pixel (x, y) can be expressed as follows:

$$
\pmb { x } = ( x z , y z , z , 1 ) ^ { \top } = W \pmb { H } ( u , v , 1 , 1 ) ^ { \top }\tag{2}
$$

where z represents intersection depth. In the rasterization, we input pixel coordinate (x, y) and inquiry intersection in Gaussianâs local coordinate. To achieve that, we need to compute the inverse transformation of the projection (2). The intersection depth z is governed by the constraints of the view-consistent 2D Gaussian. By solving this constraint equation, we can obtain the final result, as described in (Huang et al. 2024):

$$
u ( x ) = \frac { h _ { u } ^ { 2 } h _ { v } ^ { 4 } - h _ { u } ^ { 4 } h _ { v } ^ { 2 } } { h _ { u } ^ { 1 } h _ { v } ^ { 2 } - h _ { u } ^ { 2 } h _ { v } ^ { 1 } } \quad v ( x ) = \frac { h _ { u } ^ { 4 } h _ { v } ^ { 1 } - h _ { u } ^ { 1 } h _ { v } ^ { 4 } } { h _ { u } ^ { 1 } h _ { v } ^ { 2 } - h _ { u } ^ { 2 } h _ { v } ^ { 1 } }\tag{3}
$$

$$
\begin{array} { r } { \pmb { h } _ { u } = ( \pmb { W } \pmb { H } ) ^ { \top } ( - 1 , 0 , 0 , x ) ^ { \top } \quad \pmb { h } _ { u } = ( \pmb { W } \pmb { H } ) ^ { \top } ( 0 , - 1 , 0 , y ) ^ { \top } } \end{array}\tag{4}
$$

where (x, y) is the pixel coordinate and $\boldsymbol { h } _ { u } ^ { i } , \boldsymbol { h } _ { v } ^ { i }$ represent the i-th parameter of the vector.

The loss function in 2DGS is as follows:

$$
\mathcal { L } = \mathcal { L } _ { c } + \alpha \mathcal { L } _ { d } + \beta \mathcal { L } _ { n }\tag{5}
$$

The depth distortion $\mathcal { L } _ { d }$ and normal consistency ${ \mathcal { L } } _ { n }$ regularization are as follows:

<table><tr><td>Î±</td><td> $\overline { { \beta } }$ </td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>F-scoreâ</td></tr><tr><td>0</td><td>0</td><td>24.54</td><td>0.838</td><td>0.200</td><td>0.15</td></tr><tr><td>100/10</td><td>0</td><td>23.56</td><td>0.817</td><td>0.225</td><td>0.14</td></tr><tr><td>100/10</td><td>0.05</td><td>23.08</td><td>0.809</td><td>0.235</td><td>0.32</td></tr><tr><td>0</td><td>0.05</td><td>24.30</td><td>0.836</td><td>0.203</td><td>0.33</td></tr><tr><td>0</td><td>0.1</td><td>24.07</td><td>0.832</td><td>0.207</td><td>0.36</td></tr><tr><td>0</td><td>0.3</td><td>22.74</td><td>0.804</td><td>0.243</td><td>0.32</td></tr></table>

Table 1: The effect of different regularization terms. The results from the TnT dataset. 100/10 in Î± denotes the settings for 360-degree and large-scale scenes as described in 2DGS, respectively.

$$
\mathcal { L } _ { d } = \sum _ { i , j } \omega _ { i } \omega _ { j } | z _ { i } - z _ { j } |\tag{6}
$$

where $\omega _ { i }$ means the weight of the i-th intersection and $z _ { i }$ is the depth of the intersection points.

$$
{ { \mathcal { L } } _ { n } } = \sum _ { i } { { { \omega _ { i } } \left( 1 - { { \mathbf { n } } _ { i } ^ { \mathrm { T } } } N \right) } }\tag{7}
$$

where i indexes the intersected splats along the ray, Ï signifies the blending weight at the intersection point, $\mathbf { \nabla } n _ { i }$ indicates the normal of the splat facing the camera, and N is the normal estimate from the gradient of the depth map. $\mathbf { p } _ { s }$ means the intersection between surfels and rays omitted from the pixel.

$$
\mathbf { N } ( x , y ) = \frac { \nabla _ { x } \mathbf { p } _ { s } \times \nabla _ { y } \mathbf { p } _ { s } } { | \nabla _ { x } \mathbf { p } _ { s } \times \nabla _ { y } \mathbf { p } _ { s } | }\tag{8}
$$

Once the training is finished, the depth map can be derived using the following formula to obtain the reconstructed result. Afterward, 2DGS utilizes TSDF fusion for mesh extraction with depth.

$$
z _ { \mathrm { m e a n } } = \sum _ { i } \omega _ { i } z _ { i } / ( \sum _ { i } \omega _ { i } + \epsilon )\tag{9}
$$

where $\omega _ { i } = T _ { i } \alpha _ { i } \hat { \mathcal { G } } _ { i } ( \mathbf { u } ( \mathbf { x } ) )$ represent the weight contribution of i-th Gaussian and $\begin{array} { r } { T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } \hat { \mathcal { G } } _ { j } ( \mathbf { u } ( \mathbf { x } ) ) ) } \end{array}$ means its visibility.

Besides, 2DGS also provides a more robust depth value extraction method. It determines the median depth as the greatest âvisibleâ depth, using $T _ { i } = 0 . 5$ as the threshold between surface and free space. The last Gaussian is selected if a rayâs accumulated alpha remains below 0.5.

$$
z _ { \mathrm { m e d i a n } } = \operatorname* { m a x } \{ z _ { i } | T _ { i } > 0 . 5 \}\tag{10}
$$

## The Impact of Normal Consistency

As shown in Table 1, depth distortion $\mathcal { L } _ { d }$ exhibits negligible impact on both rendering quality and geometric reconstruction. Normal consistency(NC) ${ \mathcal { L } } _ { n }$ plays a more important role in 2DGS surface reconstruction. It facilitates the local alignment of all 2D splats with the underlying surface. In fact, it is a very strong geometric constraint that affects all properties of 2DGS except for spherical harmonics. Increasing the weight of $\mathcal { L } _ { d }$ improves geometric reconstruction quality but degrades rendering quality. Moreover, if $\mathcal { L } _ { d }$ becomes too dominant, both geometric reconstruction and rendering quality deteriorate.

<!-- image-->  
Figure 2: Each sceneâs average value of 2DGS on the Tanks and Temples Dataset (Knapitsch et al. 2017). The introduction of NC increases $K _ { a }$ and opacity. The larger the $K _ { a }$ and opacity, the better the reconstruction quality, while the rendering quality will deteriorate.

First, we analyze the impact of NC loss on the spatial coverage of 2DGS. Each Gaussian has the following attributes {Âµ, Î£, Î±, SHs}: central position $\mu ,$ covariance matrix Î£, opacity Î± and SH coefficient. We primarily focus on S and $\alpha ,$ as it is challenging to analyze the other attributes using a fair and quantitative method. In particular, we utilize ${ \bar { \boldsymbol { K } } } _ { a }$ to characterize $\boldsymbol { S } = \left( s _ { u } , s _ { v } \right)$ , which represents the spatial coverage.

$$
K _ { a } = s _ { u } \times s _ { v }\tag{11}
$$

As shown in Figure 2, the introduction of NC increases both $K _ { a }$ and opacity $\alpha .$ Moreover, the larger $K _ { a }$ and opacity Î±, the higher the reconstruction quality, but the lower the rendering quality. Figure 2 also shows that the properties required for optimal rendering quality differ from those needed for the best reconstruction results in 2DGS. This suggests that resolving the conflict between rendering quality and geometric reconstruction may be impossible. However, the traditional texture mapping (Zhou and Koltun 2014; Lee et al. 2020; Ha et al. 2021), demonstrates that enhancing rendering quality is possible without altering the underlying geometry. Naturally, this observation leads to a simple yet effective approach to addressing the conflict. We prioritize training the spatial distribution of the 2DGS first, followed by fine-tuning the spherical harmonic coefficients to optimize rendering quality. However, the rendering results are still inferior to those of 2DGS without NC.

<!-- image-->

Figure 3: Our training process is divided into three stages. To illustrate this, we use a pure blue triangle as the ground truth example. Each ellipse denotes a 2D Gaussian, and the color variations reflect the evolution of appearance as training progresses.  
<!-- image-->  
Figure 4: $\mu$ and Î£ represent the geometric properties of the Gaussians, while Î± and SHs are used to represent the appearance model of the Gaussians.

As shown in Figure 4, we analyze how different Gaussian attributes affect rendering quality and geometric reconstruction. We observe that as training progresses, $\{ \mu , \Sigma \}$ no longer fluctuate as drastically as they did in the early stages of training. This implies that, with $\{ \mu , \Sigma \}$ fixed, we can achieve a balance between rendering quality and geometric reconstruction by adjusting {Î±, SHs}. In this case, Î± plays a crucial role in determining both geometry and rendering.

## Stage 1 and Stage 2

During Stage 1 and Stage 2, we focus on fine-tuning the appearance while preserving the geometry. In Stage 1, we train 2DGS with normal consistency, resulting in a model with well-distributed Gaussians in space but suboptimal appearance. Upon completion of this stage, we obtain N Gaussians. We then compute the color error between each Gaussian and the corresponding ground-truth pixel across all training views, accumulating the errors to form a per-Gaussian error metric. The detailed computation is provided in Eq. 12, and a simple illustrative example for computing $E _ { i }$ is shown in Figure 5.

$$
E _ { i } = \sum _ { v = 1 } ^ { \mathcal { V } } \sum _ { i = 1 } ^ { N } \mathbb { L } ( i , v ) \cdot \mathbb { K } ( i , v ) \cdot \left| C _ { i } ^ { v } - C _ { g t } ^ { v } \right|\tag{12}
$$

$$
\mathbb { L } ( i , v ) = { \left\{ \begin{array} { l l } { 1 , } & { G _ { i } \in { \mathcal { F } } ( v ) } \\ { 0 , } & { G _ { i } \notin { \mathcal { F } } ( v ) } \end{array} \right. }\tag{13}
$$

<!-- image-->  
Depth ReconstructFigure 5: We take a single Gaussian and two perspectives as examples to demonstrate how to calculate $E _ { i }$

$$
\mathbb { K } ( i , v ) = \left\{ { \begin{array} { l l } { 1 , } & { C _ { g t } ^ { v } \in S ( i , v ) } \\ { 0 , } & { C _ { g t } ^ { v } \notin S ( i , v ) } \end{array} } \right.\tag{14}
$$

where V means the training-set views, $C _ { i } ^ { v }$ means the color of i-th Gaussian under training-set view v, $C _ { g t } ^ { v }$ represents the ground truth color corresponding to the pixel positions occupied by i-th Gaussian $G _ { i }$ under view $v . { \bar { \mathcal { F } } } ( v )$ means the frustum of view v. S(i, v) means the set of pixel position occupied by i-th Gaussian $G _ { i }$ under view v.

As illustrated in Figure 6, for each observed viewpoint, the color of a Gaussian is computed via spherical harmonics (SH) using the direction vector from the camera to the Gaussian center. The Gaussian is then splatted onto the image plane, where the final pixel color is influenced by both the Gaussianâs color and its opacity. We design $E _ { i }$ to highlight regions with significant color variation but relatively flat geometryâsuch as the mottled area in the lower-left corner of Figure 6. In such regions, we expect $E _ { i }$ to yield higher values. To this end, $E _ { i }$ is computed without considering the effect of Î± (opacity), and it is not normalized by the number of occupied pixels. This design enhances its correlation with $K _ { a } .$ For i-th Gaussian, we obtain corresponding $E _ { i }$ . For the i-th Gaussian, we compute the corresponding error score $E _ { i }$ All Gaussians are then sorted in descending order based on their $E _ { i }$ values. We select the top K% as high-error Gaussians (HEGs), and designate the rest as low-error Gaussians (LEGs). During Stage 2 (Figure 3), we freeze all attributes of the LEGs, while for HEGs, only the SH coefficients remain trainable. Subsequently, both groups are jointly trained without applying normal consistency.

<!-- image-->  
Figure 6: The top image illustrates the rendering process of a single Gaussian, while the bottom showcases the impact on the rendered color and normal after introducing NC.

## Stage 3

In Stage 3, the optimization objective shifts to jointly finetuning appearance and geometry to achieve higher rendering fidelity and structural accuracy. We consider a typical scenario where the pixel coverage of fine structures is smaller than the Ka of a single 2D Gaussian. In such cases, merely adjusting Î±, SHs is insufficient to improve rendering quality, as illustrated in the bottom of Figure 6. When color varies rapidly over small spatial regions, large Gaussians with high opacity fail to capture these variations accurately, resulting in rendering artifacts.

To address this limitation, we propose a novel targeted enhancement strategy that adaptively increases modeling capacity in challenging regions. Specifically, we introduce a small number of additional Gaussians in fine-grained or mottled areas that are not well represented by the existing set. However, two key challenges arise: (1) where to insert new Gaussians, and (2) which attributes should be fine-tuned to enhance rendering while preserving geometry.

To solve the first challenge, we establish a guiding principle: improve rendering quality while minimizing geometric distortion. Based on this principle, we choose to add as few Gaussians as possible. Our key insight is that high-error Gaussians (HEGs) inherently identify regions that are difficult to reconstruct accurately. Due to their large coverage and small quantity, HEGs are ideal insertion points. Therefore, we perform in-place cloning of HEGs to generate new Gaussians specifically at these critical locations.

To address the second challenge, we carefully select the attributes to optimize. Since SHs influence appearance without affecting geometry, we fine-tune SH coefficients for the newly added Gaussians. In addition, we optimize Î£ to allow these Gaussians to better conform to complex spatial color patterns, such as mottled regions. However, opacity (Î±) affects both appearance and depth, and adjusting it can distort geometry. Hence, we adopt a freeze-opacity (FO) strategy during this stage. This design reflects another core contribution of our method: a decoupled fine-tuning mechanism that separates appearance adaptation from geometric refinement.

In essence, our three-stage pipeline is designed to progressively disentangle and resolve appearance-geometry conflicts. From geometry-aware initialization to error-driven refinement and targeted enhancement, each stage plays a distinct role. This strategy not only improves rendering quality in visually complex regions, but also maintains global geometric consistency with minimal additional overhead.

## Experiments and Results

## Datasets and Metrics

We assess the performance of our method using both synthetic and real datasets, specifically the NeRF-Synthetic (Mildenhall et al. 2020), Tanks & Temples (TNT) (Knapitsch et al. 2017) and DTU (Jensen et al. 2014) dataset. We evaluate geometric reconstruction and rendering quality on these three datasets at the same time. For training and evaluation, we keep the resolution of the NeRF Synthetic dataset and we downsample the resolution of the DTU dataset to 1/2. For TnT dataset, we downsample the original image to half its resolution.

For rendering quality, we use PSNR, SSIM (Wang et al. 2004) and LPIPS (Zhang et al. 2018) for quantitative comparison on NeRF Synthetic, TnT and DTU datasets. Following the strategy of 2DGS (Huang et al. 2024), we evaluated PSNR on the training set for the DTU dataset. To evaluate the quality of geometry reconstruction, we adopt different metrics depending on the characteristics of each dataset. On the DTU dataset, we leverage the availability of high-quality ground-truth meshes and calculate the bidirectional Chamfer Distance (CD) between them and the reconstructed mesh extracted from the predicted TSDF. For the NeRF Synthetic and TnT datasets, where ground-truth surface geometry is not directly accessible, we instead compute the F1-score, which provides a meaningful measure of geometric accuracy under the given constraints.

## Implementation and Baseline

We compare with NeuS (Wang et al. 2021), 3DGS (Bae et al. 2024), Sugar (Guedon and Lepetit 2024), GSDF (Yu et al. Â´ 2024), GOF (Yu, Sattler, and Geiger 2024), PGSR (Chen et al. 2024a), SVRaster (Sun et al. 2025) and 2DGS (Huang et al. 2024). Training stage 1 keeps the same strategy as the original 2DGS. Specifically, we increase the SH degree every 1000 steps until the maximum degree is reached. The number of iterations in the training stage is set to 30k, 10k, and 20k, respectively. We set K = 1 for all datasets, resulting in only a 1% increase in storage compared to 2DGS. All experiments are conducted on an RTX 4090 GPU.

<table><tr><td rowspan="2">Dataset</td><td colspan="4">TnT</td><td colspan="4">NeRF Synthetic</td><td colspan="4">DTU</td></tr><tr><td>F-Scoreâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>F-Scoreâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>CDâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Method NeuS</td><td>0.38</td><td>24.58</td><td>â</td><td>â</td><td>â</td><td>â</td><td>â</td><td>â</td><td>0.84</td><td>31.97</td><td>0.840</td><td></td></tr><tr><td>3DGS</td><td>â</td><td></td><td>â</td><td>â</td><td>â</td><td></td><td></td><td></td><td>4.03</td><td>32.91</td><td>0.943</td><td>0.092</td></tr><tr><td>Sugar</td><td>â</td><td>â</td><td></td><td>â</td><td>â</td><td></td><td></td><td></td><td>1.24</td><td>32.76</td><td>0.942</td><td>0.094</td></tr><tr><td>GSDF</td><td>â</td><td>â</td><td>â</td><td>â</td><td>â</td><td></td><td></td><td></td><td>0.80</td><td>33.65</td><td>0.948</td><td>0.092</td></tr><tr><td>GOF*</td><td>0.46</td><td>23.06</td><td>0.845</td><td>0.177</td><td>0.908</td><td>33.28</td><td>0.969</td><td>0.031</td><td>0.75</td><td>35.11</td><td>0.949</td><td>0.130</td></tr><tr><td>PGSR*</td><td>0.40</td><td>23.79</td><td>0.851</td><td>0.160</td><td>0.907</td><td>31.84</td><td>0.964</td><td>0.036</td><td>0.56</td><td>33.78</td><td>0.945</td><td>0.147</td></tr><tr><td>SVRaster</td><td>0.40</td><td>23.04</td><td></td><td>0.144</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2DGS</td><td>0.33</td><td>24.30</td><td>0.836</td><td>0.203</td><td>0.905</td><td>31.89</td><td>0.965</td><td>0.038</td><td>0.77</td><td>34.78</td><td>0.939</td><td>0.165</td></tr><tr><td>2DGS(w/o NC)</td><td>0.15</td><td>24.54</td><td>0.838</td><td>0.200</td><td>0.874</td><td>33.06</td><td>0.968</td><td>0.033</td><td>1.40</td><td>35.54</td><td>0.941</td><td>0.163</td></tr><tr><td>2DGS-R</td><td>0.33</td><td>24.73</td><td>0.841</td><td>0.197</td><td>0.911</td><td>32.92</td><td>0.967</td><td>0.034</td><td>0.74</td><td>35.69</td><td>0.943</td><td>0.159</td></tr></table>

Table 2: Quantitative comparison on three datasets. The rendering quality on DTU is derived from the training set, while the rendering quality on TnT and NeRF Synthetic datasets is evaluated from novel view synthesis. \* indicates results reproduced using the official source code. The F-Score reported in PGSR original paper is 0.52, which primarily due to using a smaller voxel size and leads to OOM under the 256GB memory. We set the voxel size to match that of the 2DGS.

<!-- image-->  
Figure 7: Visual comparison of mesh normal results on Synthetic NeRF dataset. In the drums case, our method can reconstruct a more complete scene. Our method can recover better details in lego case.

<!-- image-->  
Figure 8: Visual comparison of novel view synthesis on TnT dataset. Our method can better reconstruct the details of the scene in the Barn. As shown in Meetingroom, our approach helps reduce artifacts in low-texture areas.  
Reference

w/o NC  
w NC  
Ours  
<!-- image-->  
Figure 9: Visual comparison of geometric reconstruction on DTU dataset.

## Results and Comparision

Rendering Quality & Geometry Reconstruction We compare rendering quality (with training-set views) on the DTU dataset and novel view synthesis (NVS) on the NeRF Synthetic and TnT datasets. NVS results in TnT are shown in Figure 8. Geometric reconstruction is evaluated on all three datasets. The results as shown in Table 2 and Figure 7, 9, 10. Our method consistently achieves competitive performance across both rendering and reconstruction tasks.

As shown in Figure 8, our method successfully recovers high-frequency details and generates more accurate color representations, particularly in low-texture areas. Additionally, Figure 7 demonstrates that our approach reconstructs cleaner and more complete meshes, featuring sharper and better-defined edges.

As illustrated in Figure 9, our method yields more accurate reconstructions, especially along challenging regions such as the edges of the can and the area surrounding the Buddhaâs left eye. Similarly, in Figure 10, our approach effectively avoids noticeable breakages along the wall boundaries and delivers a smoother reconstruction of the excavatorâs bucket.

<!-- image-->  
Figure 10: Visual comparison of geometric reconstruction on TnT dataset. Compared to directly introducing NC, our method produces smoother reconstruction results.

## Discussion and Conclusion

We first conducted comprehensive ablation studies, including the influence of K, the impact of each training stage, the effect of fine-tuning settings, and the use of alternative operations to replace in-place cloning. We report the quantitative evaluation results in Table 3, 4, 5, 6.

## Ablation

The impact of K We validate the impact of different K values on rendering quality and reconstruction quality on the TnT dataset. As shown in Table 3, despite the increase in K, there is no noticeable improvement in rendering quality and geometric reconstruction.
<table><tr><td></td><td>F-Scoreâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>K=1</td><td>0.33</td><td>24.73</td><td>0.841</td><td>0.197</td></tr><tr><td>K=3</td><td>0.33</td><td>24.70</td><td>0.841</td><td>0.197</td></tr><tr><td>K=5</td><td>0.33</td><td>24.71</td><td>0.841</td><td>0.196</td></tr><tr><td>K=10</td><td>0.33</td><td>24.72</td><td>0.841</td><td>0.195</td></tr><tr><td>K=30</td><td>0.33</td><td>24.75</td><td>0.842</td><td>0.193</td></tr><tr><td>K=50</td><td>0.34</td><td>24.78</td><td>0.843</td><td>0.191</td></tr></table>

Table 3: Quantitative ablation study on K.

The impact of finetune attributes We experiment with different combinations of Gaussian attribute fine-tuning in training Stage 3 to observe their impact. Comparison is shown in Table 4. It can be seen that the impact on the F-score is not significant, but freezing different attributes causes a noticeable difference in rendering quality.

The impact of each module We evaluate the effectiveness of each module from the TnT dataset. As shown in Table 5(I), applying NC in 2DGS as a basic baseline. (II) While employing in-place clone operation, the improvement in PSNR is quite significant, whereas the F-score drops drastically. (III) By incorporating the freezing opacity operation, both PSNR and F-score show a slight improvement. (IV) Resuming NC in Stage 3 leads to a slight decrease in PSNR, but a significant improvement in F-score.

Replace the in-place clone with AbsGS We replace the in-place clone densification strategy with an alternative strategy, AbsGS (Ye et al. 2024), and the results are shown in Table 6. Since the original AbsGS is based on 3DGS, directly using its hyperparameters on 2DGS named AbsGS-D is not appropriate. Therefore, we also fine-tuned the hyperparameters for 2DGS named AbsGS-A. It can be observed that after parameter tuning, the rendering quality improves, but the geometric quality deteriorates. This is typically due to the fact that gradient-based densification strategies are highly sensitive to the loss function and often require multiple iterations, which can cause fluctuations in the number of Gaussians. As a result, it is usually necessary to adjust the hyperparameter settings for different scenes.

<table><tr><td>Freeze/Finetune</td><td>F-Scoreâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>()/(Âµ,Î±,Î£,SHs)</td><td>0.33</td><td>24.61</td><td>0.840</td><td>0.198</td></tr><tr><td>(Î±,SHs)/(Âµ,Î±,â)</td><td>0.33</td><td>24.52</td><td>0.838</td><td>0.201</td></tr><tr><td>(SHs)/(Î¼,Î±,Î£)</td><td>0.33</td><td>24.50</td><td>0.837</td><td>0.201</td></tr><tr><td>(Î±)/(Î¼,Î±,Î£,SHs)</td><td>0.33</td><td>24.73</td><td>0.841</td><td>0.197</td></tr></table>

Table 4: Ablation study on freeze/finetune of TNT dataset. The strategy of freezing opacity yields the best results compared with other strategies.

<table><tr><td>Methods</td><td>Clone</td><td>FO</td><td>RNC</td><td>F-Scoreâ</td><td>PSNRâ</td></tr><tr><td>(I)</td><td>X</td><td>X</td><td>X</td><td>0.33</td><td>24.30</td></tr><tr><td>(II)</td><td></td><td></td><td></td><td>0.24</td><td>24.84</td></tr><tr><td>(III)</td><td>&gt;</td><td>*&gt;&gt;</td><td></td><td>0.26</td><td>24.89</td></tr><tr><td>(IV)</td><td></td><td></td><td>ÃÃ&gt;</td><td>0.33</td><td>24.73</td></tr></table>

Table 5: Ablation study on the 6 scenes of TNT dataset. âCloneâ means the in-place clone operation. âFOâ means the freeze opacity of 2D Gaussians in Stage 3. âRNCâ means resuming the NC in Stage 3.

<table><tr><td></td><td colspan="2">TNT</td><td colspan="2">NeRF Synthetic</td><td colspan="2">DTU</td></tr><tr><td>Method</td><td>F-Scoreâ</td><td>PSNRâ</td><td>F-Scoreâ</td><td>PSNRâ</td><td>CDâ</td><td>PSNRâ</td></tr><tr><td>2DGS(w/o NC)</td><td>0.15</td><td>24.54</td><td>0.874</td><td>33.06</td><td>1.40</td><td>35.54</td></tr><tr><td>2DGS(w NC)</td><td>0.33</td><td>24.30</td><td>0.905</td><td>31.89</td><td>0.77</td><td>34.78</td></tr><tr><td>2DGS-R(AbsGS-D)</td><td>0.19</td><td>23.19</td><td>0.899</td><td>32.53</td><td>0.95</td><td>33.53</td></tr><tr><td>2DGS-R(AbsGS-A)</td><td>0.08</td><td>24.26</td><td>0.904</td><td>33.15</td><td>0.90</td><td>34.57</td></tr><tr><td>2DGS-R(ours)</td><td>0.33</td><td>24.73</td><td>0.911</td><td>32.92</td><td>0.74</td><td>35.69</td></tr></table>

Table 6: Quantitative comparison of rendering quality and reconstruction quality. AbsGS-D means using the original hyperparameters, while AbsGS-A means that the original hyperparameters have been adjusted with more sensitive thresholds for size and positional gradient.

## Conclusion

In this work, we study the effect of introducing Normal Consistency (NC) into Gaussian attributes. While NC improves reconstruction accuracy, it can degrade rendering quality. To balance this trade-off, we propose a simple yet effective multi-stage training strategy that progressively refines different attributes, mitigating the conflict between reconstruction and rendering with only a slight increase in training time.

## References

Bae, J.; Kim, S.; Yun, Y.; Lee, H.; Bang, G.; and Uh, Y. 2024. Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting. In European Conference on Computer Vision (ECCV).

Barron, J. T.; Mildenhall, B.; Tancik, M.; Hedman, P.; Martin-Brualla, R.; and Srinivasan, P. P. 2021. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, 5855â5864.

Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.; and Hedman, P. 2022. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. arXiv:2111.12077.

Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.; and Hedman, P. 2023. Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields. ICCV.

Chen, A.; Xu, Z.; Geiger, A.; Yu, J.; and Su, H. 2022. Tensorf: Tensorial radiance fields. In European Conference on Computer Vision, 333â350. Springer.

Chen, D.; Li, H.; Ye, W.; Wang, Y.; Xie, W.; Zhai, S.; Wang, N.; Liu, H.; Bao, H.; and Zhang, G. 2024a. Pgsr: Planarbased gaussian splatting for efficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics.

Chen, Y.; Wu, Q.; Lin, W.; Harandi, M.; and Cai, J. 2024b. HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression. In European Conference on Computer Vision.

Cheng, K.; Long, X.; Yang, K.; Yao, Y.; Yin, W.; Ma, Y.; Wang, W.; and Chen, X. 2024. Gaussianpro: 3d gaussian splatting with progressive propagation. In Forty-first International Conference on Machine Learning.

Dai, P.; Xu, J.; Xie, W.; Liu, X.; Wang, H.; and Xu, W. 2024. High-quality surface reconstruction using gaussian surfels. In ACM SIGGRAPH 2024 Conference Papers, 1â11.

Duan, Y.; Wei, F.; Dai, Q.; He, Y.; Chen, W.; and Chen, B. 2024. 4d gaussian splatting: Towards efficient novel view synthesis for dynamic scenes. arXiv preprint arXiv:2402.03307.

Fridovich-Keil, S.; Yu, A.; Tancik, M.; Chen, Q.; Recht, B.; and Kanazawa, A. 2022. Plenoxels: Radiance Fields without Neural Networks. In CVPR.

Guedon, A.; and Lepetit, V. 2024. SuGaR: Surface-Aligned Â´ Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering. CVPR.

Ha, H.; Lee, J. H.; Meuleman, A.; and Kim, M. H. 2021. Normalfusion: Real-time acquisition of surface normals for high-resolution rgb-d scanning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15970â15979.

Hedman, P.; Srinivasan, P. P.; Mildenhall, B.; Barron, J. T.; and Debevec, P. 2021. Baking neural radiance fields for realtime view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, 5875â5884.

Hu, W.; Wang, Y.; Ma, L.; Yang, B.; Gao, L.; Liu, X.; and Ma, Y. 2023. Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural Radiance Fields. In ICCV.

Huang, B.; Yu, Z.; Chen, A.; Geiger, A.; and Gao, S. 2024. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery.

Jensen, R.; Dahl, A.; Vogiatzis, G.; Tola, E.; and AanÃ¦s, H. 2014. Large scale multi-view stereopsis evaluation. In 2014 IEEE Conference on Computer Vision and Pattern Recognition, 406â413. IEEE.

Jo, J.; Kim, H.; and Park, J. 2024. Identifying Unnecessary 3D Gaussians using Clustering for Fast Rendering of 3D Gaussian Splatting. arXiv preprint arXiv:2402.13827.

Keetha, N.; Karhade, J.; Jatavallabhula, K. M.; Yang, G.; Scherer, S.; Ramanan, D.; and Luiten, J. 2024. SplaTAM: Splat Track & Map 3D Gaussians for Dense RGB-D SLAM. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21357â21366.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G.Â¨ 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4).

Knapitsch, A.; Park, J.; Zhou, Q.-Y.; and Koltun, V. 2017. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. ACM Transactions on Graphics, 36(4).

Lee, J.; Lee, S.; Lee, J.; Park, J.; and Sim, J. 2024. GSCore: Efficient Radiance Field Rendering via Architectural Support for 3D Gaussian Splatting. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, 497â511.

Lee, J. H.; Ha, H.; Dong, Y.; Tong, X.; and Kim, M. H. 2020. Texturefusion: High-quality texture acquisition for real-time rgb-d scanning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1272â 1280.

Lu, T.; Yu, M.; Xu, L.; Xiangli, Y.; Wang, L.; Lin, D.; and Dai, B. 2024a. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20654â20664.

Lu, Z.; Guo, X.; Hui, L.; Chen, T.; Yang, M.; Tang, X.; Zhu, F.; and Dai, Y. 2024b. 3D Geometry-aware Deformable Gaussian Splatting for Dynamic View Synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Matsuki, H.; Murai, R.; Kelly, P. H.; and Davison, A. J. 2024. Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18039â18048.

Max, N. 1995. Optical models for direct volume rendering. IEEE Transactions on Visualization and Computer Graphics, 1(2): 99â108.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV.

Muller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In- Â¨ stant neural graphics primitives with a multiresolution hash

encoding. ACM Transactions on Graphics (ToG), 41(4): 1â 15.

Navaneet, K.; Meibodi, K. P.; Koohpayegani, S. A.; and Pirsiavash, H. 2023. Compact3D: Smaller and Faster Gaussian Splatting with Vector Quantization. arXiv preprint arXiv:2311.18159.

Niedermayr, S.; Stumpfegger, J.; and Westermann, R. 2024. Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10349â10358.

Reiser, C.; Peng, S.; Liao, Y.; and Geiger, A. 2021. Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In Proceedings of the IEEE/CVF international conference on computer vision, 14335â14345.

Schonberger, J. L.; Zheng, E.; Frahm, J.-M.; and Pollefeys, Â¨ M. 2016. Pixelwise view selection for unstructured multiview stereo. In Computer VisionâECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, 501â518. Springer.

Sun, C.; Choe, J.; Loop, C.; Ma, W.-C.; and Wang, Y.-C. F. 2025. Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, 16187â16196.

Tang, J.; Zhou, H.; Chen, X.; Hu, T.; Ding, E.; Wang, J.; and Zeng, G. 2023. Delicate textured mesh recovery from nerf via adaptive surface refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 17739â17749.

Wang, H.; Zhu, H.; He, T.; Feng, R.; Deng, J.; Bian, J.; and Chen, Z. 2024. End-to-End Rate-Distortion Optimized 3D Gaussian Representation. In European Conference on Computer Vision.

Wang, P.; Liu, L.; Liu, Y.; Theobalt, C.; Komura, T.; and Wang, W. 2021. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. NeurIPS.

Wang, Z.; Bovik, A.; Sheikh, H.; and Simoncelli, E. 2004. Image Quality Assessment: From Error Visibility to Structural Similarity. Image Processing, IEEE Transactions on, 13: 600 â 612.

Yan, C.; Qu, D.; Xu, D.; Zhao, B.; Wang, Z.; Wang, D.; and Li, X. 2024. Gs-slam: Dense visual slam with 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19595â19604.

Yao, Y.; Luo, Z.; Li, S.; Fang, T.; and Quan, L. 2018. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), 767â783.

Ye, Z.; Li, W.; Liu, S.; Qiao, P.; and Dou, Y. 2024. Absgs: Recovering fine details in 3d gaussian splatting. In Proceedings of the 32nd ACM International Conference on Multimedia, 1053â1061.

Yu, A.; Li, R.; Tancik, M.; Li, H.; Ng, R.; and Kanazawa, A. 2021. Plenoctrees for real-time rendering of neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 5752â5761.

Yu, M.; Lu, T.; Xu, L.; Jiang, L.; Xiangli, Y.; and Dai, B. 2024. Gsdf: 3dgs meets sdf for improved rendering and reconstruction. arXiv preprint arXiv:2403.16964.

Yu, Z.; and Gao, S. 2020. Fast-mvsnet: Sparse-to-dense multi-view stereo with learned propagation and gaussnewton refinement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 1949â 1958.

Yu, Z.; Sattler, T.; and Geiger, A. 2024. Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes. ACM Transactions on Graphics.

Zhang, K.; Riegler, G.; Snavely, N.; and Koltun, V. 2020. Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.

Zhou, H.; Shao, J.; Xu, L.; Bai, D.; Qiu, W.; Liu, B.; Wang, Y.; Geiger, A.; and Liao, Y. 2024a. HUGS: Holistic Urban 3D Scene Understanding via Gaussian Splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 21336â21345.

Zhou, Q.-Y.; and Koltun, V. 2014. Color map optimization for 3d reconstruction with consumer depth cameras. ACM Transactions on Graphics (ToG), 33(4): 1â10.

Zhou, X.; Lin, Z.; Shan, X.; Wang, Y.; Sun, D.; and Yang, M.-H. 2024b. Drivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21634â21643.

Zwicker, M.; Rasanen, J.; Botsch, M.; Dachsbacher, C.; and Pauly, M. 2004. Perspective accurate splatting. In Proceedings-Graphics Interface, 247â254.