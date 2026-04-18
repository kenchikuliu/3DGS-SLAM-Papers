# DentalSplat: Dental Occlusion Novel View Synthesis from Sparse Intra-Oral Photographs

Yiyi Miao1,3â , Taoyu Wu2,4â ,â , Tong Chen1,3 , Sihao Li2,3 , Ji Jiang6 , Youpeng Yang5 , Angelos Stefanidis1 , Limin Yu2 Â¶, and Jionglong Su1 Â¶

1 School of AI and Advanced Computing, Xiâan Jiaotong Liverpool University, Suzhou, China

{Yiyi.Miao21, Tong.Chen19}@student.xjtlu.edu.cn, {Jionglong.Su, Angelos Stefanidis}@xjtlu.edu.cn

2 School of Advanced Technology, Xiâan Jiaotong Liverpool University, Suzhou, China

{Taoyu.Wu21, Sihao.li19}@student.xjtlu.edu.cn, Limin.yu@xjtlu.edu.cn 3 School of Electrical Engineering, Electronics and Computer Science , University of Liverpool, Liverpool, United Kingdom

4 School of Physical Sciences, University of Liverpool, Liverpool, United Kingdom 5 College of Computer Science and Technology, Zhejiang University, Hangzhou, China ypyang@zju.edu.cn

Abstract. In orthodontic treatment, particularly within telemedicine contexts, observing patientsâ dental occlusion from multiple viewpoints facilitates timely clinical decision-making. Recent advances in 3D Gaussian Splatting (3DGS) have shown strong potential in 3D reconstruction and novel view synthesis. However, conventional 3DGS pipelines typically rely on densely captured multi-view inputs and precisely initialized camera poses, limiting their practicality. Orthodontic cases, in contrast, often comprise only three sparse images, specifically, the anterior view and bilateral buccal views, rendering the reconstruction task especially challenging. The extreme sparsity of input views severely degrades reconstruction quality, while the absence of camera pose information further complicates the process. To overcome these limitations, we propose DentalSplat, an effective framework for 3D reconstruction from sparse orthodontic imagery. Our method leverages a prior-guided dense stereo reconstruction model to initialize the point cloud, followed by a scale-adaptive pruning strategy to improve the training efficiency and reconstruction quality of 3DGS. In scenarios with extremely sparse viewpoints, we further incorporate optical flow as a geometric constraint, coupled with gradient regularization, to enhance rendering fidelity. We validate our approach on a large-scale dataset comprising 950 clinical cases and an additional video-based test set of 195 cases designed to simulate real-world remote orthodontic imaging conditions. Experimental results demonstrate that our method effectively handles sparse input

Y. Miao et al.

scenarios and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art techniques.

Keywords: Orthodontics Â· 3D Reconstruction Â· Telemedicine

## 1 Introduction

Accurate dental occlusion reconstruction [37] is crucial for orthodontic treatment, impacting planning, adjustments, and long-term function. Traditionally, Cone Beam Computed Tomography (CBCT) [18] and Intraoral Scanning (IOS) [8] have been key tools, providing high-resolution 3D models of teeth and their relationships. CBCT visualizes hard tissues like teeth and jawbones, aiding occlusal analysis, while IOS captures surface details of dental arches to create precise digital models [24]. These are often combined with occlusion registration, where the patient bites into a set position to align the arches [34]. However, these methods require specialized equipment and expertise [16], limiting their use to remote monitoring. Thus, an alternative approach is needed to achieve accurate occlusal reconstruction with minimal input, reducing the reliance on advanced imaging.

Artificial Intelligence (AI) technologies, such as DentalMonitoring [15] and Invisalign Virtual Care AI [29], have become integral tools in dental clinics for remote orthodontic treatment monitoring [27]. While these systems enable patients to capture dental images using smartphone-connected devices, they primarily rely on single-view images, which significantly limit comprehensive assessments of occlusal relationships and spatial positioning [12]. Novel View Synthesis (NVS) offers a promising solution to this limitation by generating new viewpoints from a set of input images, thereby enabling more accurate occlusion evaluation. Recently, 3D Gaussian Splatting (3DGS) [19] has demonstrated superior performance in terms of rendering efficiency and photorealism through its use of explicit 3D Gaussian representations and differentiable rasterization [21]. However, despite its excellent rendering quality, 3DGS [19] heavily depends on Structure-from-Motion (SfM) methods such as COLMAP for camera pose estimation and initialization. This dependency presents a significant challenge for real-world applications like remote oral diagnostics, where image acquisition is inherently sparse. Several approaches have been proposed to reduce 3DGSâs reliance on dense image inputs. MVSplat [6] introduces a Transformer-based framework that incorporates pre-trained geometric priors and epipolar constraints to infer depth information and guide reconstruction. Similarly, Nope-NeRF [3] and CF-3DGS [10] utilize depth-based constraints to minimize dependence on COLMAP for pose estimation. Nevertheless, these methods typically assume overlapping views and continuous video inputs, making them less suitable for truly sparse, pose-free scenarios in novel view synthesis and scene reconstruction tasks. DUSt3R [30] addresses these limitations by requiring only two sparse, unposed images to generate point and confidence maps for end-to-end 3D reconstruction. By leveraging pre-trained models, DUSt3R produces high-quality

3D models directly from RGB images while simultaneously providing camera poses. This approach effectively supports various tasks, including intrinsic recovery and pose estimation, making it particularly suitable for applications with limited input data.

Despite addressing the limitations of SfM dependency in sparse input scenarios, DUSt3R faces significant challenges in orthodontic applications [33]. In remote orthodontic practice, patients often use various mobile devices with varying camera qualities and inherent noise, resulting in sparse and uncalibrated images that compromise accurate visualization [28]. The inherent characteristics of orthodontic imaging, including specular reflections from tooth enamel, motion blur during intraoral image capture, and variable lighting conditions [17], further impact the performance of the 3DGS pipeline. When utilizing DUSt3R for 3DGS initialization, the dense point clouds generated by DUSt3R may reduce computational efficiency and hinder convergence. The excessive density of these point clouds can significantly degrade 3DGS optimization. Moreover, in standard orthodontic imaging protocols, there are substantial angular differences between frontal and bilateral buccal views. Without proper geometric constraints, this can lead to suboptimal reconstruction and rendering quality. These challenges necessitate specialized adaptations to effectively integrate DUSt3R with 3DGS for orthodontic applications.

To address the challenges of sparse input and unknown camera poses in orthodontic scenarios, we present DentalSplat, a novel 3D dental reconstruction framework based on 3DGS and DUSt3R, designed to address the challenges of sparse input and unknown camera poses in orthodontic scenarios. This is the first framework capable of achieving high-quality novel view synthesis and 3D reconstruction from sparse, pose-free dental images within a minute. Specifically, we introduce a Scale-Adaptive Pruning (SAP) strategy for Gaussian Splatting, which operates on the dense point clouds generated by DUSt3R. This strategy analyzes the spatial distribution characteristics of point clouds to determine adaptive thresholds for different spatial regions, effectively handling both dense and sparse areas. This approach significantly reduces the initial 3D point cloud size while maintaining quality, thereby decreasing both optimization time and computational overhead in 3DGS. To address the challenges of reflections and motion blur in dental reconstruction, we incorporate optical flow constraints by computing the residual between the optical flow generated from 3DGS projections of adjacent frames and that from the original 2D images. This optical flow loss is integrated with the traditional photometric loss to enhance multi-view geometric consistency. Furthermore, to mitigate the blurring artifacts that can occur in 3DGS due to inaccurate gradients affecting splitting and cloning operations during optimization, we compute gradient weights for each Gaussian, effectively reducing local over-reconstruction artifacts.

Our main contributions can be summarized in threefold:

â We enhance the SAP strategy to mitigate the computational burden imposed by DUSt3Râs dense point clouds during 3DGS optimization.

â We propose an enhanced differential Gaussian rasterization module with optical flow and gradient-weighted optimization, effectively improving the rendering quality of complex dental structures.

â We validate our framework on a self-collected dataset of 956 clinical dental cases, demonstrating superior reconstruction speed and novel view synthesis quality under sparse input conditions compared to baseline methods.

## 2 Related Work

## 2.1 3D Scene Reconstruction

For decades, 3D reconstruction from images has been dominated by classical pipelines combining SfM and Multi-View Stereo (MVS). SfM systems, such as the widely-used COLMAP [26], first recover a sparse 3D point cloud and camera poses by matching local features across multiple views and performing bundle adjustment. Subsequently, MVS algorithms densify this sparse representation by leveraging photometric consistency across views. A paradigm shift occurred with the introduction of Neural Radiance Fields (NeRF) [25], which represents a scene as a continuous 5D function learned by a Multi-Layer Perceptron (MLP). By mapping 3D coordinates and a 2D viewing direction to volume density and color, NeRF achieves state-of-the-art photorealism for novel view synthesis through differentiable volume rendering. However, the original NeRF is slow to train and render, and critically, it requires a dense set of input images with accurate camera poses, typically pre-computed using COLMAP. Subsequent research has focused on mitigating these limitations. Mip-NeRF [1] addressed aliasing artifacts by rendering anti-aliased conical frustums instead of rays, improving detail representation across different scales. More recently, 3DGS [19] has emerged as a leading method, combining the benefits of explicit representations with the differentiability of neural rendering. 3DGS models a scene as a collection of 3D Gaussians, each with optimizable properties such as position, covariance, color, and opacity.

## 2.2 Camera Pose-Free Reconstruction

A significant research thrust has focused on eliminating the reliance on precomputed camera poses from SfM. These methods aim to jointly optimize the scene representation and camera parameters. BARF [22] was a pioneering work that enabled the joint optimization of camera poses and a NeRF model. It introduced a coarse-to-fine registration strategy by gradually unmasking highfrequency components of the positional encoding, which proved crucial for avoiding poor local minima. Nope-NeRF [3] incorporates geometric priors from a monocular depth estimator to constrain the relative poses between frames, stabilizing the joint optimization process. The pose-free paradigm has also been extended to 3D Gaussian Splatting. CF-3DGS [10] adapts 3DGS for video streams without SfM pre-processing by sequentially estimating the relative pose of each new frame and progressively growing the set of Gaussians. This approach leverages the temporal continuity of video input and the explicit nature of the Gaussian representation to achieve robust tracking and reconstruction.

## 2.3 Sparse-View Reconstruction

Another critical challenge is reconstructing scenes from a sparse set of input views, where per-scene optimization is highly under-constrained and prone to overfitting. PixelNeRF [36] conditions a NeRF on image features extracted by a convolutional network, allowing it to synthesize novel views from a single image in a feed-forward pass. MVSNeRF [5] integrates principles from MVS by constructing a plane-swept cost volume from as few as three views, providing a powerful geometric prior that enables high-quality generalization. For sparseview 3DGS, MVSplat [7] leverages a plane-swept cost volume to infer geometric cues from multi-view stereo, which then guides the direct, feed-forward prediction of 3D Gaussian parameters. This geometry-aware approach demonstrates strong generalization and efficiency for sparse inputs.

## 2.4 End-to-End Reconstruction from Unposed Images

The DUSt3R model [30] addresses the challenges of sparse inputs and unknown poses by enabling end-to-end 3D reconstruction from unposed, uncalibrated image pairs. It predicts relative camera poses and dense depth maps, acting as a general-purpose geometric foundation model. DUSt3R eliminates the need for traditional Structure from Motion (SfM) pipelines by leveraging a large, diverse training dataset. However, DUSt3Râs limitation lies in its pairwise input processing, which introduces computational inefficiencies, especially with larger image sets. To address this, MUSt3R [4] extends DUSt3R to multi-view reconstruction, allowing all views to be processed in a single forward pass.

Despite the power of these general-purpose models, as we identify in our work, their direct application for initializing 3DGS in specialized domains like orthodontics presents unique challenges, such as the computational burden of dense point cloud outputs and susceptibility to domain-specific artifacts. These limitations motivate our proposed contributions in DentalSplat, which adapt and refine this powerful prior for high-fidelity dental reconstruction.

## 3 Methodology

## 3.1 Preliminary

3D Gaussian Splatting. 3DGS [19] represents a scene using an explicit collection of anisotropic Gaussian primitives defined in 3D space. Each primitive $G _ { i }$ is parametrized by a mean position ${ \pmb { \mu } } _ { i } \in \mathbb { R } ^ { 3 }$ , an opacity $o _ { i } \in [ 0 , 1 ]$ , and a covariance matrix $\pmb { \mathcal { Z } } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ . The spatial density of each Gaussian is given as:

$$
G _ { i } ( \mathbf { X } ) = o _ { i } \cdot \exp \left\{ - \frac { 1 } { 2 } ( \mathbf { X } - \pmb { \mu } _ { i } ) ^ { \top } \pmb { \Sigma } _ { i } ^ { - 1 } ( \mathbf { X } - \pmb { \mu } _ { i } ) \right\} ,\tag{1}
$$

<!-- image-->  
Fig. 1. Overview of DentalSplat. Given a set of sparse and unposed input images, we leverage a stereo-dense reconstruction model to regress the dense point cloud of these input images in the global coordinate system and obtain the corresponding relative camera pose. Subsequently, we apply the SAP strategy to eliminate outlier points, followed by downsampling to obtain a sparse point cloud suitable for 3DGS initialization. During optimization, we incorporate optical flow constraints to ensure geometric consistency and employ gradient constraints to enhance the densification of the 3DGS.

where $\mathbf { X } \in \mathbb { R } ^ { 3 }$ denotes an arbitrary 3D point. The covariance matrix $\pmb { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ can be decomposed into a scaling matrix and a rotation quaternion for efficient optimization.

The rendering process projects these 3D Gaussians onto the 2D image plane for a given camera pose. The final color ${ \hat { C } } ( \mathbf { p } )$ and depth $\hat { D } ( \mathbf { p } )$ for each pixel p are synthesized by alpha-blending the contributions of all Gaussians that overlap with the pixel, sorted from front to back along the camera ray. The blending process is formulated as:

$$
\hat { C } ( \mathbf { p } ) = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) , \quad \hat { D } ( \mathbf { p } ) = \sum _ { i \in \mathcal { N } } d _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where $\mathcal { N }$ is the set of sorted Gaussians. For each Gaussian $i , \ c _ { i }$ is its color, determined by its SH coefficients for the current viewing direction, and $d _ { i }$ is its depth, corresponding to the z-coordinate of its center $\pmb { \mu } _ { i }$ in camera space. The blending weight, $\alpha _ { i } .$ , is crucial and is calculated by multiplying the learned opacity $o _ { i }$ with the value of the projected 2D Gaussianâs probability density function evaluated at the pixel center p. This formulation allows for differentiable rendering, enabling end-to-end optimization of the Gaussian attributes through gradient-based methods.

Dust3R. Dust3R [30] proposes a unified and calibration-free 3D reconstruction framework that bypasses the traditional reliance on keypoint correspondences and explicit camera parameters. It introduces the concept of a pointmap, a dense mapping from image pixels to 3D coordinates. Given an $i \times j$ RGB image I and its corresponding depth map D, the pointmap $\boldsymbol { X } \in \mathbb { R } ^ { W \times H \times \dot { 3 } }$ is computed in the camera coordinate system using the intrinsic matrix K as follows:

$$
\boldsymbol { X } _ { i , j } = K ^ { - 1 } \left[ i D _ { i , j } , j D _ { i , j } , D _ { i , j } \right] ^ { T } .\tag{3}
$$

Given two views $I _ { t _ { 1 } }$ and $I _ { t _ { 2 } } ,$ their respective pointmaps $X _ { t _ { 1 } } , X _ { t _ { 2 } }$ can be aligned via a rigid transformation:

$$
X _ { t _ { 1 }  t _ { 2 } } = T _ { t _ { 2 } } T _ { t _ { 1 } } ^ { - 1 } X _ { t _ { 1 } } ,\tag{4}
$$

where $T _ { t _ { 1 } } , T _ { t _ { 2 } } \in S E ( 3 )$ are the world-to-camera transformations for each view.

The training objective of Dust3R is based on 3D regression with scale normalization and confidence-aware optimization. For each valid pixel i in frames $I _ { t 1 }$ and $I _ { t 2 }$ , the regression loss is computed as the Euclidean distance between the predicted and ground-truth point maps, scaled to resolve scale ambiguity:

$$
\mathcal { L } _ { \mathrm { r e g } } ( v , i ) = \left. \frac { 1 } { z } X _ { i } ^ { v } - \frac { 1 } { z ^ { \prime } } X _ { i } ^ { g t } \right. .\tag{5}
$$

To address scale ambiguity, Dust3R normalizes the predicted and groundtruth point maps using scaling factors z and $z ^ { \prime }$ , which represent the average distance of valid points from the origin:

$$
z = { \frac { 1 } { | D _ { 1 } | + | D _ { 2 } | } } \sum _ { i \in D } \| X _ { i } \| .\tag{6}
$$

Dust3R also introduces a confidence-aware loss to mitigate issues arising from poorly defined 3D points, such as those in the sky or translucent objects. The confidence score for each pixel $C _ { v } ^ { i }$ is defined as $1 + \exp ( \tilde { C } _ { v } ^ { i } )$ , ensuring positivity and enabling adaptive loss weighting:

$$
\mathcal { L } _ { \mathrm { c o n f } } = \sum _ { v \in \{ 1 , 2 \} } \sum _ { i \in \mathcal { D } ^ { v } } C _ { v } ^ { i } \left. \frac { 1 } { z } \hat { X } _ { i } ^ { v } - \frac { 1 } { z ^ { \prime } } X _ { i } ^ { \mathrm { g t } } \right. - \alpha \log C _ { v } ^ { i } .\tag{7}
$$

Equation (7) promotes robustness to geometric ambiguities and provides a means to infer per-pixel confidence, which can be utilized in downstream tasks such as global alignment and visual localization. Dust3Râs output pointmaps serve as a strong initialization for 3DGS and enable consistent 3D representations without requiring extrinsic or intrinsic camera calibration.

## 3.2 3DGS Initialization

Initialization. For sparse and unposed orthodontic input images, we employ DUSt3R to generate a point cloud that serves as the initialization for 3DGS training. Specifically, once the DUSt3R network is optimized, it produces precise point maps for the given frames. These point maps enable the recovery of camera parameters and globally aligned point clouds, effectively resolving the convergence issues encountered by COLMAP with sparse, uncalibrated images. The point clouds derived from DUSt3R provide a robust foundation for initializing 3DGS primitives.

Scale-Adaptive Pruning. Although DUSt3R generates dense point clouds, their excessive density can adversely impact 3DGS optimization efficiency and scene representation quality. Unlike the original 3DGS approach, which relies on sparse point clouds from COLMAP, we propose the SAP method, which necessitates additional pruning to make the dense DUSt3R point clouds compatible with the Adaptive Density Control (ADC) mechanism. Inspired by [9,35], we implement an efficient pruning strategy that filters the initial point cloud after downsampling. This approach selectively retains the most significant points based on their spatial influence, as characterized by their scaling parameters.

We denote the scaling parameters of all Gaussians as $\mathbf { S } \in \mathbb { R } ^ { N \times 3 }$ , where ÂµS represents the mean magnitude of all scaling components. The pruning masks are computed as:

$$
\begin{array} { r l } & { \mathcal { M } _ { 1 } = \big \{ \mathbf S _ { i } ~ \big | ~ \mathrm { m a x } ( \mathbf S _ { i } ) > \mu _ { S } \big \} } \\ & { \mathcal { M } _ { 2 } = \{ \{ \mathbf S _ { i } ~ \big | ~ \mathrm { m a x } ( \mathbf S _ { i } ) > Q _ { i } ( \mathbf S ) \} \quad \mathrm { i f ~ } N < 5 \times 1 0 ^ { 6 } ,  } \\ & {  \mathcal { M } _ { 2 } = \{ \{ \mathbf S _ { i } ~ \big | ~ \mathrm { m a x } ( \mathbf S _ { i } ) > 4 \mu _ { S } \}  \quad \mathrm { o t h e r w i s e } , } \end{array}\tag{8}
$$

where $Q _ { i }$ is $( \cdot )$ denotes the percentile quantile function. The final pruning mask is obtained through a logical conjunction: $\mathcal { M } _ { \mathrm { f i n a l } } = \mathcal { M } _ { 1 } \cap \mathcal { M } _ { 2 }$

The pruning method removes outliers while retaining critical points in complex geometries. Adjusting the threshold based on point cloud size prevents excessive pruning in large scenes and ensures robust outlier removal in smaller ones. The surviving Gaussians meet the condition ${ \mathcal { G } } _ { \mathrm { s u r v i v e d } } = \{ \mathbf { G } _ { i } \ | \ \mathbf { S } _ { i } \in { \mathcal { M } } _ { \mathrm { f i n a l } } \}$ , where Gi represents the i-th Gaussian primitive.

## 3.3 3DGS Optimization

Gradient Constraint. The Gradient Collision issue represents a significant challenge in 3DGS, manifesting as poor reconstruction quality and regional blur [23,21]. This issue causes conflicts between gradient directions from different pixels. Each 3DGS influences multiple pixels, and each pixel is affected by multiple 3DGS elements. When gradients conflict, the accumulated gradient for a 3DGS weakens, hindering densification operations.

To simplify the notion, consider a single 3DGS element $G _ { i }$ projected onto the 2D image plane as a 2D Gaussian $g _ { i }$ centered at $\mu _ { i }$ , affecting n pixels. The total loss function L quantifies the discrepancy between predicted and actual values, with gradients calculated as $\scriptstyle \sum _ { j = 1 } ^ { n } { \frac { \partial L _ { j } } { \partial \mu _ { i , x } } }$ and $\textstyle \sum _ { j = 1 } ^ { n } { \frac { \partial L _ { j } } { \partial \mu _ { i , y } } }$ , where n denotes the number of pixels affected by $g _ { i } .$ , and $L _ { j }$ represents the loss computed for the j-th pixel. Significant variation in pixel gradient directions leads to Gradient Collision, causing gradient accumulation to decrease. This misalignment prevents accurate splitting direction estimation, resulting in ineffective splits and increased blur in over-reconstructed regions.

To address this issue, we conduct an absolute operation to constrain the gradients following [35]. The absolute operation method aligns gradient directions along their axes, ensuring consistency. By mitigating Gradient Collision, it reduces blur, especially in over-reconstructed regions. The absolute operation is defined as:

$$
\hat { g } _ { i , x } = \sum _ { j = 1 } ^ { n } \left| \frac { \partial L _ { j } } { \partial \mu _ { i , x } } \right| , \quad \hat { g } _ { i , y } = \sum _ { j = 1 } ^ { n } \left| \frac { \partial L _ { j } } { \partial \mu _ { i , y } } \right| .\tag{9}
$$

Per-Gaussian Pixel Flow. To enhance rendering quality, we incorporate optical flow loss as a geometric constraint. Optical flow results from both object and camera motion, with camera movement being the primary contributor in our scenarios involving sparse input data and significant viewpoint differences [2,32]. In 3D Gaussians Splatting processing, each pixel $\mathbf { x } _ { i }$ corresponds to a set of 3DGS, where the pixel colour is obtained by alpha-blending the 2D Gaussians projected from multiple 3D Gaussians. Building upon the work presented in [11], at time t, we render the i-th 3D Gaussian using the camera pose $\mathcal { T } _ { t }$ onto the 2D image plane, resulting in pixel ${ \bf x } _ { i , t }$ . This pixel is mapped to the canonical space using the mean $\mu _ { i , t }$ and covariance matrix $\Sigma _ { i , t }$ of the corresponding i-th 2D Gaussian. At time $t + 1$ , the the pixel position $\mathbf { x } _ { i , t + 1 }$ is determined by projecting the 3DGS through the unknown-but-sought camera pose $\hat { \mathcal { T } } _ { t + 1 }$ , as expressed by:

$$
\mathbf { x } _ { i , t + 1 } = \pi \left( \mathcal { G } _ { t } , \mathcal { T } _ { t + 1 } \right) ,\tag{10}
$$

where $\pi ( \cdot )$ denotes the camera projection. From this, we can obtain the corresponding mean $\pmb { \mu } _ { i , t + 1 }$ and covariance matrix $\Sigma _ { i , t + 1 }$ for the i-th Gaussian. The Gaussian flow for the i-th Gaussian is given by the positional displacement, which represents the difference between the position of the pixel:

$$
\mathrm { f l o w } _ { i } ^ { G } ( { \bf x } _ { t } ) = { \bf x } _ { i , t + 1 } - { \bf x } _ { i , t } .\tag{11}
$$

Simultaneous Optimization by Flow constraint. Unlike [11], in our work, the Gaussians are isotropic, where both covariance matrices are symmetric and positive definite. We jointly optimize the estimated camera pose $\hat { \mathcal { T } } _ { t + 1 }$ and 3DGS primitive $\hat { \mathcal G }$ by the flow loss. Consequently, the Cholesky factorization [13] of the covariance matrices $\Sigma _ { i , t }$ and $\Sigma _ { i , t + 1 }$ simplifies to the identity matrix. This enables us to express the Gaussian flow for the i-th Gaussian equivalent to:

$$
\mathrm { f l o w } _ { i } ^ { G } ( { \mathbf x } _ { t } ) = { \pmb { \mu } } _ { i , t + 1 } - { \pmb { \mu } } _ { i , t } .\tag{12}
$$

For each pixel with K overlapping Gaussians, we compute the composite flow through alpha-weighted blending:

$$
\mathrm { \mathrm { f o w } } ^ { G } ( \mathcal T _ { t + 1 } , \mathcal G _ { t } ) = \sum _ { i = 1 } ^ { K } w _ { i } ( \pmb { \mu } _ { i , t + 1 } - \pmb { \mu } _ { i , t } ) ,\tag{13}
$$

where $w _ { i }$ denotes the normalized blending weight of the i-th Gaussian along the camera ray. For adjacent frames $\mathbf { I } _ { t }$ and $\mathbf { I } _ { t + 1 }$ , we obtain the optical flow

<!-- image-->  
Fig. 2. Overview of Flow Constraint. At time $t ,$ each 2D pixel $x _ { t }$ is formed by projecting $K$ overlapping 3D Gaussians under camera pose $\tau _ { t }$ $\mathrm { A t }$ time $t + 1$ , their motions induce Gaussian flows whose projections are aggregated to estimate the overall optical flow. To jointly optimize the 3D Gaussian primitives $\hat { \mathcal G }$ and the camera pose $\mathcal { T } _ { t + 1 }$ , we minimize the residual between the estimated optical flow and the ground truth optical flow computed using an off-the-shelf method.

flow $\mathrm { \cdot } ^ { \mathrm { G t } } ( \mathbf { x } )$ using an off-the-shelf method as ground truth. We then define the flow loss aggregated over all pixels as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f l o w } } = \| \mathrm { f l o w } ^ { G } ( \mathcal { T } _ { t + 1 } , \mathcal { G } _ { t } ) - \mathrm { f l o w } ^ { G t } ( \mathbf { x } ) \| _ { 2 } . } \end{array}\tag{14}
$$

However, in sparse scenes, optical flow prediction tends to introduce more noise. To address this, we employ a bidirectional optical flow model, which computes the forward optical flow between Frame t and Frame $t + 1$ , as well as the backward optical flow from Frame $t + 1$ to Frame t. By leveraging the bidirectional optical flow process, we can obtain the corresponding optical flow confidence mask, denoted as $M ( \mathbf { x } _ { t _ { 1 } } )$ . The confidence mask is then applied to both the forward optical flow and the ground truth flow to compute the adjusted flow loss. The per-pixel flow loss is then calculated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f l o w } } ( \mathbf { x } _ { t _ { 1 } } ) = \| M ( \mathbf { x } _ { t _ { 1 } } ) \odot \mathrm { f l o w } ^ { \mathrm { G } } ( \mathbf { x } _ { t _ { 1 } } ) - M ( \mathbf { x } _ { t _ { 1 } } ) \odot \mathrm { f l o w } ^ { G t } ( \mathbf { x } _ { t _ { 1 } } ) \| _ { 2 } , } \end{array}\tag{15}
$$

where $M ( \mathbf { x } _ { t _ { 1 } } )$ represents the confidence mask applied to the pixel $\mathbf { x } _ { t _ { 1 } }$ , and $\odot$ denotes element-wise multiplication.

In the whole training process, we simultaneously optimize the estimated camera pose $\hat { \mathcal { T } } _ { t + 1 }$ and 3DGS primitive $\hat { \mathcal G }$ by minimizing the following objective function:

$$
\hat { \mathcal { T } } _ { t + 1 } , \hat { \mathcal { G } } = \underset { \mathcal { T } _ { t + 1 } , \mathcal { G } } { \mathrm { a r g m i n } } \left( \lambda _ { 1 } \mathcal { L } _ { \mathrm { r g b } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { f l o w } } \right) ,\tag{16}
$$

where the RGB loss $\mathcal { L } _ { r g b }$ measures the $\mathcal { L } _ { 1 }$ residual between the rendered RGB image $\hat { I } _ { t + 1 }$ (using pose Tt) and the ground truth image $I _ { t + 1 }$ :

## 4 Experiments

## 4.1 Implementation Details

Dataset Description. To evaluate the accuracy and robustness of our framework, we conducted comprehensive experiments on a clinical intra-oral dataset collected in collaboration with professional dental hospitals. All images were captured by certified orthodontists using a Canon EOS 700D camera equipped with a 100mm macro lens and operated in forced flash mode to ensure consistent illumination and minimize lighting variability.

The dataset comprises two distinct subsets designed for different experiments. The first video dataset consists of 195 clinical cases, each recorded as intra-oral video by professional orthodontists to simulate remote orthodontic scenarios. Each video captures a continuous transition from the right buccal view, through the frontal occlusal view, to the left buccal view. From each video, we uniformly sampled 24 frames based on the frame rate and video duration. These frames were then evenly divided into a training view set and a test view set, each containing 12 images. During training, only the camera poses and corresponding 2D images from the training set were provided as input. For training views, we examined four different sparse view scenarios using 3, 6, 9, and 12 viewpoints, respectively, to analyze the frameworkâs performance under different input conditions. Once training was completed, the optimized 3D model was used to render novel 2D views at the camera poses in the test set, thereby assessing the quality of novel view synthesis. The second image dataset contains 950 clinical cases, each consisting of only three intra-oral photographs: one anterior occlusal view capturing the full dentition from the front, and bilateral buccal views from the left and right sides. These cases were selected from routine orthodontic records and serve to evaluate the frameworkâs capacity to reconstruct and synthesize novel views under extremely sparse input conditions.

Experimental Setup. We conduct all experiments and evaluations on a desktop computer equipped with an Intel Core i9-13900KF CPU and an NVIDIA GeForce RTX 4090 GPU. We apply the same set of hyperparameters to all cases in the dataset. For the 3D Gaussians, we follow the default training parameters from the original Gaussian Splatting implementation [19]. We use the Adam optimizer [20] to update the Gaussian parameters. To balance rendering efficiency and quality, we set the number of training iterations to 2000.

## 4.2 Evaluation results

Comparative Experiments on Video Test Dataset. To evaluate the novel view synthesis capabilities of our framework, we conducted comprehensive qualitative and quantitative comparisons on our video dataset against the original

3DGS framework and several state-of-the-art baselines, including 3DGS, CF-3DGS, and InstantSplat. As shown in Table 1, we report the average values of three metrics across 195 cases, Peak Signal-to-Noise Ratio (PSNR) [14], Structural Similarity Index Measure (SSIM) [31], and Learned Perceptual Image Patch Similarity (LPIPS) [38]. Our method achieves the best performance across all three metrics. Standard 3DGS fails to converge during optimization when initialized with sparse multi-view inputs, as indicated by the "-" entries in the table. It is only trainable under the 12-view setting, yet still exhibits substantially lower rendering quality compared to other methods. This highlights a fundamental limitation of conventional approaches when applied to dental imaging scenarios, where observations are often restricted and highly sparse.

Table 1. Quantitative evaluation on video test dataset.
<table><tr><td>Algorithm</td><td>3 Training views</td><td></td><td>6 Training views</td></tr><tr><td></td><td>PSNRâ SSIMâ LPIPSâ</td><td></td><td>PSNRâ SSIMâ LPIPSâ</td></tr><tr><td>3DGS [19]</td><td>- -</td><td>-</td><td>- - -</td></tr><tr><td>InstantSplat [9]</td><td>23.81 0.826</td><td>0.304 27.01</td><td>0.863 0.268</td></tr><tr><td>CF-3DGS [10]</td><td>15.32 0.748</td><td>0.443 18.01</td><td>0.795 0.277</td></tr><tr><td>Ours</td><td>23.96 0.822</td><td>0.301 28.41</td><td>0.872 0.247</td></tr><tr><td>Algorithm</td><td>9 Training</td><td>views</td><td>12 Training views</td></tr><tr><td></td><td>PSNRâ SSIMâ</td><td>LPIPSâ PSNRâ</td><td>SSIMâ LPIPSâ</td></tr><tr><td>3DGS [19]</td><td>- </td><td>- 11.51</td><td>0.53 0.574</td></tr><tr><td>InstantSplat [9]</td><td>28.656 0.890</td><td>0.241 29.315</td><td>0.898 0.235</td></tr><tr><td>CF-3DGS [10]</td><td>21.29 0.812</td><td>0.374 23.02</td><td>0.853 0.337</td></tr><tr><td>Ours</td><td>29.363 0.891</td><td>0.237 30.174</td><td>0.897 0.213</td></tr></table>

Figure 3 illustrates the qualitative evaluations of novel view synthesis. Under the 6-view input condition, CF-3DGS suffers from noticeable blurring and floating artifacts. InstantSplat exhibits geometric distortions in the lower teeth when compared with the ground truth. With 9-view inputs, CF-3DGS eliminates major artifacts in the dental region but still produces blurry and low-resolution images, with evident overfitting and hallucinated geometry in the right buccal area. InstantSplat also suffers from over-reconstruction in the right molars, leading to texture degradation and shape distortion. These artifacts may adversely affect clinical assessment, particularly in remote orthodontic follow-ups. In contrast, our reconstructions remain artifact-free and preserve geometric fidelity across both 6-view and 9-view inputs, demonstrating strong generalization and high-quality novel view synthesis performance.

Comparative Experiments on 3 Views images Dataset. To further assess the frameworkâs robustness under extremely sparse input conditions, we conducted additional experiments on an image dataset comprising 950 clinical cases. For each case, occlusal reconstruction was performed using only three intra-oral images: anterior, left buccal, and right buccal views. Since no test views are available in this dataset, Table 2 reports the reconstruction performance on the training views using the same three evaluation metrics (PSNR, SSIM, LPIPS) averaged over 956 cases. The standard 3DGS fails to converge under such sparse conditions, as denoted by "-".

<!-- image-->  
Fig. 3. Novel View Synthesis Comparisons with 6 views and 9 views input.We qualitatively compare the quality of novel view synthesis and show that our method has better quality with more accurate texture details.

Table 2. Quantitative results with the other methods using 3 views.
<table><tr><td>Methods</td><td colspan="4">PSNRâ SSIMâ LPIPSâ Times (Seconds)â</td></tr><tr><td>3DGS [19]</td><td>-</td><td>-</td><td></td><td>-</td></tr><tr><td>InstantSplat [9]</td><td>32.78</td><td>0.945</td><td>0.160</td><td>57</td></tr><tr><td>CF-3DGS [10]</td><td>18.37</td><td>0.803</td><td>0.32</td><td>372</td></tr><tr><td>Ours</td><td>34.50</td><td>0.954</td><td>0.135</td><td>69</td></tr></table>

For qualitative evaluation, Figure 4 presents the input training views used in the experiments, and Figure 5 visualizes synthesized views that were not seen during training. Although ground truth is unavailable for these novel viewpoints, relative comparisons indicate that our method successfully reconstructs dental structures with high fidelity, free from geometric holes or blurring. This confirms the effectiveness of our framework in producing high-quality reconstructions even with extremely limited inputs.

## 4.3 Ablation Study

We present the ablation study in Table 3 to validate the contribution of each component in our framework. We conduct an ablation study on the proposed

Left buccal view  
Anterior view  
Right buccal view  
<!-- image-->  
Fig. 4. Reconstruction Comparisons with 3 views. Visualization of Rendered Images and GT with 3 views Input.  
3 views

Case_1  
Case_2  
<!-- image-->  
Fig. 5. Novel View Synthesis Comparisons with 3 views. Due to the lack of ground truth for the 3-view input setting, our analysis focuses on relative performance improvements.

SAP strategy, optical flow constraint(Flow), gradient constraint(Gradient), and confidence mask within the flow loss(FlowMask), as summarized in Table 3.

When the gradient constraint is removed, the performance drops significantly. This is because the gradient loss plays a key role in guiding the densification strategy of 3DGS, and its absence hinders effective Gaussian expansion. Similarly, removing both the optical flow and the associated confidence mask also leads to a notable decline in performance. However, this configuration results in a substantial increase in training efficiency, as the geometric constraints from optical flow introduce additional computational overhead and increase the number of parameters to optimize. When the SAP strategy is removed, the performance decreases slightly. This is because the initialization quality mainly affects the early-stage convergence speed of 3DGS. As training progresses, the network continuously optimizes the Gaussians through the joint minimization of photometric, flow, and gradient losses, gradually compensating for the impact of suboptimal initialization.

Table 3. Ablation study of our method under sparse-view(6 views) setup.
<table><tr><td>Ablation Setting PSNRâ SSIMâ LPIPSâ Times</td><td></td><td></td><td></td><td>(Seconds)â</td></tr><tr><td>w.o. Gradient</td><td>27.58</td><td>0.857</td><td>0.288</td><td>62</td></tr><tr><td>W.o. FlowMask</td><td>27.94</td><td>0.843</td><td>0.261</td><td>64</td></tr><tr><td>W.o. Flow</td><td>27.35</td><td>0.852</td><td>0.285</td><td>57</td></tr><tr><td>W.O. SAP</td><td>28.31</td><td>0.867</td><td>0.254</td><td>70</td></tr><tr><td>Full model</td><td>28.41</td><td>0.872</td><td>0.247</td><td>72</td></tr></table>

## 5 Conclusion

In this paper, we introduce DentalSplat, the first reconstruction framework for dental occlusion based on Dust3R, capable of supporting dynamic, sparse, and unposed input images. Extensive experiments with our collected dataset demonstrate that the incorporated geometric and gradient optimization strategies are highly effective for orthodontic scenarios, with the quality of synthesized novel views significantly surpassing that of state-of-the-art models. For remote orthodontics, the system requires only a video or a few images to complete scene training and high-quality novel view synthesis within a minute.

## References

1. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5855â5864 (2021)

2. Beauchemin, S.S., Barron, J.L.: The computation of optical flow. ACM computing surveys (CSUR) 27(3), 433â466 (1995)

3. Bian, W., Wang, Z., Li, K., Bian, J.W., Prisacariu, V.A.: Nope-nerf: Optimising neural radiance field with no pose prior. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4160â4169 (2023)

4. Cabon, Y., Stoffl, L., Antsfeld, L., Csurka, G., Chidlovskii, B., Revaud, J., Leroy, V.: Must3r: Multi-view network for stereo 3d reconstruction. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1050â1060 (2025)

5. Chen, A., Xu, Z., Zhao, F., Zhang, X., Xiang, F., Yu, J., Su, H.: Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 14124â14133 (2021)

6. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J., Cai, J.: Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In: European Conference on Computer Vision. pp. 370â386. Springer (2024)

7. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J., Cai, J.: Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In: European Conference on Computer Vision. pp. 370â386. Springer (2024)

8. Elgarba, B.M., Meeus, J., Fontenele, R.C., Jacobs, R.: Ai-based registration of ios and cbct with high artifact expression. Journal of Dentistry 147, 105166 (2024)

9. Fan, Z., Cong, W., Wen, K., Wang, K., Zhang, J., Ding, X., Xu, D., Ivanovic, B., Pavone, M., Pavlakos, G., et al.: Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds. arXiv preprint arXiv:2403.20309 2(3), 4 (2024)

10. Fu, Y., Liu, S., Kulkarni, A., Kautz, J., Efros, A.A., Wang, X.: Colmap-free 3d gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20796â20805 (2024)

11. Gao, Q., Xu, Q., Cao, Z., Mildenhall, B., Ma, W., Chen, L., Tang, D., Neumann, U.: Gaussianflow: Splatting gaussian dynamics for 4d content creation. arXiv preprint arXiv:2403.12365 (2024)

12. Hansa, I., Katyal, V., Semaan, S.J., Coyne, R., Vaid, N.R.: Artificial intelligence driven remote monitoring of orthodontic patients: clinical applicability and rationale. In: Seminars in Orthodontics. vol. 27, pp. 138â156. Elsevier (2021)

13. Higham, N.J.: Cholesky factorization. Wiley interdisciplinary reviews: computational statistics 1(2), 251â254 (2009)

14. Hore, A., Ziou, D.: Image quality metrics: Psnr vs. ssim. In: 2010 20th international conference on pattern recognition. pp. 2366â2369. IEEE (2010)

15. Impellizzeri, A., Horodinsky, M., Barbato, E., Polimeni, A., Philippe, S., Galluccio, G., et al.: Dental monitoring application: it is a valid innovation in the orthodontics practice? La Clinica Terapeutica 171(3), 260â267 (2020)

16. Johnson, A., Jani, G., Pandey, A., Patel, N.: Digital tooth reconstruction: An innovative approach in forensic odontology. The Journal of Forensic Odontostomatology 37(3), 12 (2019)

17. Kalpana, D., Rao, S.J., Joseph, J.K., Kurapati, S.K.R.: Digital dental photography. Indian Journal of Dental Research 29(4), 507â512 (2018)

18. Kapila, S., Conley, R., Harrell Jr, W.: The current status of cone beam computed tomography imaging in orthodontics. Dentomaxillofacial Radiology 40(1), 24â34 (2011)

19. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139â1 (2023)

20. Kingma, D.P.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

21. Li, Z., Yao, S., Wu, T., Yue, Y., Zhao, W., Qin, R., Garcia-Fernandez, A.F., Levers, A., Zhu, X.: Ulsr-gs: Ultra large-scale surface reconstruction gaussian splatting with multi-view geometric consistency. arXiv preprint arXiv:2412.01402 (2024)

22. Lin, C.H., Ma, W.C., Torralba, A., Lucey, S.: Barf: Bundle-adjusting neural radiance fields. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5741â5751 (2021)

23. Mallick, S.S., Goel, R., Kerbl, B., Steinberger, M., Carrasco, F.V., De La Torre, F.: Taming 3dgs: High-quality radiance fields with limited resources. In: SIGGRAPH Asia 2024 Conference Papers. pp. 1â11 (2024)

24. Mangano, F., Gandolfi, A., Luongo, G., Logozzo, S.: Intraoral scanners in dentistry: a review of the current literature. BMC oral health 17, 1â11 (2017)

25. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99â106 (2021)

26. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4104â4113 (2016)

27. Sharp, C.: Orthodontics in the city: top tips to improve aligner tracking. BDJ Team 12(1), 32â36 (2025)

28. Snider, V., Homsi, K., Kusnoto, B., Atsawasuwan, P., Viana, G., Allareddy, V., Gajendrareddy, P., Elnagar, M.H.: Effectiveness of ai-driven remote monitoring technology in improving oral hygiene during orthodontic treatment. Orthodontics & Craniofacial Research 26, 102â110 (2023)

29. Thurzo, A., KurilovÃ¡, V., Varga, I.: Artificial intelligence in orthodontic smart application for treatment coaching and its impact on clinical performance of patients monitored with ai-telehealth system. In: Healthcare. vol. 9, p. 1695. MDPI (2021)

30. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20697â20709 (2024)

31. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600â612 (2004)

32. Wu, T., Miao, Y., Li, Z., Zhao, H., Dang, K., Su, J., Yu, L., Li, H.: Endoflow-slam: Real-time endoscopic slam with flow-constrained gaussian splatting. arXiv preprint arXiv:2506.21420 (2025)

33. Xie, J., Zhang, C., Wei, G., Wang, P., Wei, G., Liu, W., Gu, M., Luo, P., Wang, W.: Tooth motion monitoring in orthodontic treatment by mobile device-based multi-view stereo. IEEE Transactions on Visualization and Computer Graphics (2024)

34. Xu, C., Tsuji, S., Makihara, Y., Li, X., Yagi, Y.: Occluded gait recognition via silhouette registration guided by automated occlusion degree estimation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3199â 3209 (2023)

35. Ye, Z., Li, W., Liu, S., Qiao, P., Dou, Y.: Absgs: Recovering fine details in 3d gaussian splatting. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 1053â1061 (2024)

36. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from one or few images. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4578â4587 (2021)

37. Zhang, J., Xia, J.J., Li, J., Zhou, X.: Reconstruction-based digital dental occlusion of the partially edentulous dentition. IEEE journal of biomedical and health informatics 21(1), 201â210 (2015)

38. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586â595 (2018)