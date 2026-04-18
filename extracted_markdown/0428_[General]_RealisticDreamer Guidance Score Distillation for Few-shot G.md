# RealisticDreamer: Guidance Score Distillation for Few-shot Gaussian Splatting

Ruocheng Wu1 Haolan He1 Yufei Wang2 Zhihao Li2 Bihan Wen2\*

1University of Electronic Science and Technology of China

2Nanyang Technological University

wuruocheng333@outlook.com, HaolanHe7777@gmail.com

yufei001@e.ntu.edu.sg, zhihao.li@ntu.edu.sg, bihan.wen@ntu.edu.sg

## Abstract

3D Gaussian Splatting (3DGS) has recently gained great attention in the 3D scene representation for its high-quality real-time rendering capabilities. However, when the input comprises sparse training views, 3DGS is prone to overfitting, primarily due to the lack of intermediate-view supervision. Inspired by the recent success of Video Diffusion Models (VDM), we propose a framework called Guidance Score Distillation (GSD) to extract the rich multi-view consistency priors from pretrained VDMs. Building on the insights from Score Distillation Sampling (SDS), GSD supervises rendered images from multiple neighboring views, guiding the Gaussian splatting representation towards the generative direction of VDM. However, the generative direction often involves object motion and random camera trajectories, making it challenging for direct supervision in the optimization process. To address this problem, we introduce an unified guidance form to correct the noise prediction result of VDM. Specifically, we incorporate both a depth warp guidance based on real depth maps and a guidance based on semantic image features, ensuring that the score update direction from VDM aligns with the correct camera pose and accurate geometry. Experimental results show that our method outperforms existing approaches across multiple datasets.

## 1. Introduction

Novel view synthesis, a fundamental problem in 3D vision, has extensive applications spanning virtual reality generation, such as VR video [6, 33] and 3D gaming [39], as well as real-world interactive domains like autonomous driving [15, 32] and 3D printing [34]. Since the introduction of Neural Radiance Fields (NeRF) [24], this field has undergone transformative changes, significantly enhancing novel view rendering quality. However, NeRFâs implicit representation poses challenges in terms of speed, both in optimization and rendering. For instance, Vanilla NeRF typically requires around a week of training and approximately one second to render a single novel view. Recently, 3D Gaussian Splatting [19] has emerged as a promising alternative for real-time, high-quality 3D scene rendering. By leveraging an explicit representation with 3D Gaussian ellipsoids, it dramatically improves optimization speed to just a few hours while enabling real-time rendering at high frame rates.

<!-- image-->  
Figure 1. Comparison of Score Distillation Sampling (SDS) [27] and the proposed Guidance Score Distillation (GSD) frameworks for video diffusion models. Our method introduces DDIM inversion [29] while also correcting the prediction noise by guidance.

Despite their success, both NeRF and 3DGS struggle with few-shot 3D reconstruction, where limited input images lead to inaccurate reconstructions. For NeRF, techniques like depth regularization [26], frequency annealing [31], and viewpoint distortion [2] have been introduced to mitigate overfitting and improve the quality of the synthesized views when the training set is sparse. These methods aim to regularize the modelâs learning representation, encouraging smoother and more consistent scene reconstructions. Similarly, 3DGS-based methods attempt to address few-shot challenges by using Gaussian densification to add intermediate splats [19], depth supervision to preserve geometric consistency [35], or floaters removal to eliminate non-representative Gaussians [48]. However, these methods fall short in addressing the core limitation of few-shot view synthesis: insufficient views lead to inaccurate 3D structures and overly smooth textures. 3DGS, in particular, is susceptible to misrepresented Gaussians and floaters. To mitigate these issues, current solutions attempt by involving additional prior from diffusion models, e.g., by using pseudo-view synthesis [48] or distilling 2D diffusion priors [40]. While improved results are achieved by integrating the image prior from 2D diffusion models, they still suffers from the multi-view consistency priors, resulting in view inconsistency and artifacts.

Inspired by the success of Video Diffusion Models (VDM) [4, 10, 13] to generate high-quality, multi-view consistent videos, we propose a novel framework that leverage VDMâs ability to enhance in few-shot 3D reconstruction for both viewpoint consistency and reconstruction quality without fine-tuning the whole model. Specifically, we introduce Guidance Score Distillation (GSD) framework, which is based on the existing experience with score distillation techniques [7, 27, 37, 47]. Our approach leverages the existing camera views and training images as guidance to correct the predicted scores at different time steps of DDIM inversion [29]. This method solves the issue of misalignment in position and geometry estimation for blurry images of VDMs, which typically causes incorrect score matching directions. To be specific, we introduce two novel guidance techniques: Depth Warp Guidance and Semantic Feature Guidance, designed to enhance geometric accuracy and semantic consistency. Depth Warp Guidance is a technique inspired by recent work in depth warping [22, 45]. By leveraging established depth estimation methods, this guidance enables the warping of images from training views to novel viewpoints.In addition to depth guidance, we introduce Semantic Feature Guidance using DINO-based features [5].

Our experimental results demonstrate the effectiveness of the proposed framework, showcasing significant improvements in 3D reconstruction quality compared to existing state-of-the-art methods. Overall, our contributions can be summarized as follows.

â¢ We propose Guidance Score Distillation (GSD), the framework to leverage pretrained Video Diffusion Models (VDMs) as generative priors for supervising few-shot 3D

Gaussian Splatting (3DGS) w/o further fine-tuning. GSD introduces a unified guidance formulation for VDMs, steering 3DGS optimization along a corrected generative path to enhance few-shot performance.

â¢ To ensure the generative direction of VDM is applicable to 3D reconstruction, we propose two guidance mechanisms to adjust the noise prediction direction. Specifically, we design Depth Warping Guidance based on depth prior from the pretrained the monocular estimator and Semantic Feature Guidance based on DINO features, which ensure that VDMâs score update direction aligns with accurate camera poses and geometric structure.

â¢ Experimental results show that our method outperforms existing approaches across various training view counts and datasets. Unlike prior methods such as FSGS, which suffer performance degradation compared to vanilla 3DGS as the number of views increases $( e . g . , \mathsf { a } \sim 0 . 1 $ dB drop with 9 views), our framework consistently enhances rendering quality, achieving a â¼ 0.4 dB improvement.

## 2. Related Work

## 2.1. Few-shot Novel View Synthesis

The original NeRF [24] and 3DGS [19] frameworks require hundreds of input views to optimize the 3D representation effectively. However, this is often infeasible in many real-world scenarios. To address these limitations, several pioneering works [8, 9, 16, 26, 26, 43, 48] have been proposed to reduce the reliance on the number of training views. Specifically, most of these approaches add extra constraints to 3D representations. For instance, methods like DepthNeRF [9], RegNeRF [26], and DNGaussian [8] utilize pretrained monocular depth estimation networks to introduce depth regularization, adding physical constraints to the 3D reconstruction process. Beyond these physical constraints, DietNeRF [16] and pixelNeRF [43] further enhance training by incorporating regularization in the hidden feature space. More recently, FSGS [48] proposed an innovative strategy to densify Gaussian points during training, which enhances the geometric consistency of the resulting 3D representations. Besides, [40] attempts to introduce 2D diffusion priors to guide the reconstructing process.

Despite these advancements, NeRF and 3DGS still struggle with sparse-view inputs, particularly in generating intermediate views with consistent geometry and semantics, leading to suboptimal reconstructions. This highlights the need for more robust approaches to generate new views effectively.

## 2.2. Score Distillation

Score Distillation Sampling (SDS) [27, 36] extends pretrained text-to-image diffusion models for sparse-view 3D reconstruction. SDS optimizes an image generator by

<!-- image-->  
Figure 2. The overall framework of our method. Our method first interpolates between vj and $v _ { k }$ to obtain a set of camera trajectories. Using these trajectories, we render x0 and apply DDIM inversion to obtain the noise images xt and $\mathbf { x } _ { t - \tau }$ at two time steps t and $t - \tau .$ After predicting the noise for both images using the video diffusion model, we apply $\mathcal { L } _ { P C C }$ and $\mathcal { L } _ { D I N O }$ to correct the noise, which refer to Eq. 14 and Eq. 11. Finally, the difference in the corrected results (i.e., LGSD) is used to supervise the reconstruction process.

## 3. Methodology

## 3.1. Preliminary

matching noisy rendered images to distributions learned by a pre-trained diffusion model, improving view consistency in sparse inputs. While SDS has shown promise, it suffers from issues like oversaturation and a lack of fine details, especially when a high Classifier-Free Guidance (CFG) value is used [38]. To address this, techniques such as time annealing [20] have been proposed, gradually reducing the diffusion timesteps to improve the generation quality. Additionally, methods like VSD [38] and HiFA [47] have reworked the distillation process to better handle noisy inputs and improve image fidelity. Futhermore, [7, 21] propose to apply DDIM inverion to SDS process, enhancing the genenration details. Despite these advancements, SDS-based approaches still struggle with artifacts such as geometry misalignment and overly smooth textures, making it challenging to apply effectively in sparse-view 3D reconstruction tasks. Further refinement is needed for consistent geometry and high realism in low-data scenarios.

Gaussian Splatting. 3D Gaussian Splatting (3DGS) [19] represents points in a 3D scene explicitly through a set of 3D Gaussian points. Each Gaussian point has attributes, including a position vector $\mu \in \mathbb { R } ^ { 3 }$ and a covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ . Any point p in 3D space is influenced by these Gaussians, which can be represented by the following 3D Gaussian distribution:

$$
G ( p ) = { \frac { 1 } { ( 2 \pi ) ^ { 3 / 2 } | \Sigma | ^ { 1 / 2 } } } e ^ { - { \frac { 1 } { 2 } } ( p - \mu ) ^ { T } \Sigma ^ { - 1 } ( p - \mu ) } .\tag{1}
$$

To ensure that the covariance matrix Î£ is positive semidefinite and physically meaningful, it is typically decomposed into a learnable rotation matrix R and a scaling matrix S, such that $\Sigma = R S S ^ { T } R ^ { T }$ . Besides, each Gaussian also stores an opacity logit $o \in \mathbb { R }$ and an appearance feature represented by n spherical harmonic (SH) coefficients $\{ \mathbf { c } _ { i } \in \mathbb { R } ^ { 3 } | i = 1 , 2 , \ldots , n \}$ , where $n = D ^ { 2 }$ is the number of SH coefficients, with degree D. When rendering, the 2D pixels are obtained by accumulating and weighting the contributions of different Gaussians along the ray direction. Thus, in the rendering pipeline, the complete Gaussian splatting representation g transforms all the learnable parameters Î¸ to produce the image $x = g ( \theta )$

The completely differentiable nature of $g ( \theta )$ has motivated the use of Differentiable Image Parameterization (DIP) [25] to directly optimize images x rendered from 3D representations. For instance, prior work [27, 36] has explored constructing loss functions with 2D diffusion models as priors, with one of the most representative groups of methods being Score Distillation [7, 27, 47].

Score Distillation. Score distillation is a group of methods that leverages diffusion model priors [11, 12, 14, 30] and has achieved significant success across various domains [11, 12, 14, 30]. The earliest work in this area is Score Distillation Sampling (SDS) [27]. Intuitively, SDS adds noise Ïµ to the rendered image $x ,$ and then uses a pretrained diffusion model to denoise it. The difference between the predicted noise $\epsilon _ { \phi }$ and the real noise Ïµ serves as the update direction for the 3D representation, which can be formalized as follows:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { S D S } } ( \theta ) = \mathbb { E } _ { t , \epsilon , c } \left[ \omega ( t ) ( \epsilon _ { \phi } ( x _ { t } , t , y ) - \epsilon ) \frac { \partial g ( \theta , c ) } { \partial \theta } \right] ,\tag{2}
$$

where the violet item and dark green item indicate the end and start points of the optimization direction (the same applies hereinafter), respectively.

However, due to the randomness of the noise step $t ,$ the use of high CFG values, and the generation diversity in the denoising process, the gradients obtained through SDS are often noisy and unstable, which leads to generated results that typically suffer from oversaturation and lack fine details [18, 27, 38]. Fortunately, recent research [7, 21] proposes modifications to SDS that calculate the difference in predicted noise for images with different noise steps as the optimization direction, where the noise process follows DDIM inversion to produce a deterministic trajectory. This method can effectively mitigate instability issues in SDS. The approach can be summarized as follows:

$$
\begin{array} { r l } & { \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S D S - D D I M } } ( \boldsymbol { \theta } ) = } \\ & { \mathbb { E } _ { t , c } \left[ \omega ( t ) \left( \epsilon _ { \boldsymbol { \phi } } ( x _ { t } , t , y ) - \epsilon _ { \boldsymbol { \phi } } ( x _ { t - \tau } , t - \tau , y ) \right) \frac { \partial g ( \boldsymbol { \theta } , c ) } { \partial \boldsymbol { \theta } } \right] . } \end{array}\tag{3}
$$

The $x _ { t }$ is acquired from DDIM inversion [29]

$$
x _ { t } = \sqrt { \bar { \alpha } _ { t } } \big ( \hat { x } _ { 0 } ^ { t - 1 } + \frac { \sqrt { 1 - \bar { \alpha } _ { t } } } { \sqrt { \bar { \alpha } _ { t } } } \epsilon _ { \phi } ( x _ { t - 1 } , t - 1 , y ) \big ) ,\tag{4}
$$

where

$$
\hat { x } _ { 0 } ^ { t - 1 } = \frac { x _ { t - 1 } } { \sqrt { \bar { \alpha } _ { t - 1 } } } - \frac { \sqrt { 1 - \bar { \alpha } _ { t - 1 } } \epsilon _ { \phi } ( x _ { t - 1 } , t - 1 , y ) } { \sqrt { \bar { \alpha } _ { t - 1 } } } .\tag{5}
$$

By iteratively applying Eq. 4 to the rendered image $x _ { 0 }$ , we can obtain images $x _ { t }$ and $x _ { t - \tau }$ at time steps t and $t - \tau ,$ respectively. Since Eq. 4 and Eq. 5 does not involve the random noise addition, the diffusion trajectory of the images across the two time steps is deterministic and consistent [7, 21, 29]. The above framework can be flexibly adapted to any diffusion model.

## 3.2. Guidance Score Distillation

Inspired by the recently popularized Video Diffusion Model (VDM) [4, 10, 13], we aim to apply Eq. 3 to VDM to further enhance the multi-view consistency. Specifically, for a continuous camera trajectory $\textbf { p } = \ \{ p _ { 1 } , p _ { 2 } , \dots , p _ { n } \}$ that contains n poses, we aim to render a sequence of continuous video frames $\mathbf { x } _ { 0 } = g ( \theta , \mathbf { p } )$ as input for the VDM model. At the same time, we ensure that the first frameâs view $p _ { 1 }$ is set to any one of the training views v, and we use the corresponding ground truth image $\mathbf { y } ^ { v }$ as the condition for the video generation diffusion model. In this case, Eq. 3 can be rewritten as:

$$
\begin{array} { r l } & { \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S D S - D D I M } } ^ { V i d e o } ( \boldsymbol { \theta } ) = } \\ & { \mathbb { E } _ { t , c } \left[ \omega ( t ) \left( \epsilon _ { \boldsymbol { \phi } } ( \mathbf { x } _ { t } , t , \mathbf { y } ^ { v } ) - \epsilon _ { \boldsymbol { \phi } } ( \mathbf { x } _ { t - \tau } , t - \tau , \mathbf { y } ^ { v } ) \right) \frac { \partial g ( \boldsymbol { \theta } , \mathbf { c } ) } { \partial \boldsymbol { \theta } } \right] . } \end{array}\tag{6}
$$

However, experiments have shown that the above method fail in few-shot novel view synthesis task due to its extremely high requirements for geometric precision and color fidelity. Compared to 2D diffusion models, VDMs introduce new sources of bias, including object motion within the scene and random camera trajectories. In novel view synthesis tasks, the scene is typically static, and the optimization direction for rendered views needs to strictly align with the camera poses. However, it is unrealistic to expect a VDM to estimate poses from noisy rendered images obtained from training 3DGS without providing true camera poses explicitly.

To address these issues, instead of fine-tuning the whole pre-trained VDM, which usually costs high, we propose a training-free method for video diffusion models called Guidance Score Distillation (GSD). Specifically, we believe that by providing additional guidance c for multiple frames, we can correct the noise predicted by the VDM model $\epsilon _ { \phi }$ thereby reducing or eliminating noise bias, which can be achieved by replacing the original noise $\epsilon _ { \phi }$ with a correction function $\mathbf { F } _ { t } \big ( c , \epsilon _ { \phi } \big )$ that takes c and $\epsilon _ { \phi }$ as inputs. Thus,

Eq. 6 can be rewritten in the following new form:

$$
\begin{array} { r l } & { \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { G S D } } ( \boldsymbol { \theta } ) = } \\ & { \mathbb { E } _ { t , c } \left[ \omega ( t ) \left( \mathbf { F } _ { t } ( c , \epsilon _ { \boldsymbol { \phi } } ) - \mathbf { F } _ { t - \tau } ( c , \epsilon _ { \boldsymbol { \phi } } ) \right) \frac { \partial g ( \boldsymbol { \theta } , c ) } { \partial \boldsymbol { \theta } } \right] . } \end{array}\tag{7}
$$

Inspired by [14, 44, 46], the distance measuring function $\mathbf { F } _ { t } \big ( c , \epsilon _ { \phi } \big )$ can be formulated as

$$
\mathbf { F } _ { t } ( c , \epsilon _ { \phi } ) = \epsilon _ { \phi } ( \mathbf { x } _ { t } , t , \mathbf { y } ^ { v } ) - \gamma ( t ) \rho _ { t } \nabla _ { \mathbf { x } _ { t } } \mathcal { D } _ { \theta } ( c , \mathbf { x } _ { t } ) ,\tag{8}
$$

where $\rho _ { t }$ is a time-dependent coefficient that controls the influence of guidance at different time steps; $\mathcal { D } _ { \boldsymbol { \theta } } ( c , \mathbf { x } _ { t } )$ denotes a distance measuring function between the condition c and the estimated clean image xt; $\begin{array} { r } { { \bf \Pi } ; \gamma ( t ) = \frac { \sqrt { \bar { \alpha } _ { t } } } { \sqrt { 1 - \bar { \alpha } _ { t } } } } \end{array}$ is the conversion factor used to convert image space guidance to noise space guidance. For simplicity, the following equations will use $\lambda _ { t } = \gamma ( t ) \rho _ { t }$ as the coefficient of $\nabla _ { \mathbf x _ { t } } \mathcal { D } _ { \theta } ( c , \mathbf x _ { t } )$ . Intuitively, Eq. 6 uses the difference between two biased noise predictions at different time steps as the model update direction, while Eq. 7 refines $\epsilon _ { \phi }$ to guide the optimization process with a more accurate noise direction difference.

Furthermore, we aim to use multiple guidance signals to supervise multiple frames. For the set of guidance ${ \bf C } = { \bf \Phi }$ $\{ \mathbf { c } _ { 1 } , \mathbf { c } _ { 2 } , . . . , \mathbf { c } _ { m } \}$ , we simply assume independence between different guidance, allowing Eq. 8 to contain the sum of multiple distance measurement functions:

$$
\mathbf { F } _ { t } ( c , \epsilon _ { \phi } ) = \epsilon _ { \phi } ( \mathbf { x } _ { t } , t , \mathbf { y } ^ { v } ) - \lambda _ { t } \nabla _ { \mathbf { x } _ { t } } \sum _ { i = 1 } ^ { m } \eta _ { i } \mathcal { D } _ { \theta _ { i } } ( c _ { i } , \mathbf { x } _ { t } ) ,\tag{9}
$$

where $\eta _ { i }$ denotes the weight of each conditional term. In the following two sections, we will introduce the specific form of $\mathcal { D } _ { \theta _ { i } }$ we choose.

## 3.3. Semantic Guidance in Distinct Views

Based on the GSD framework, a naive idea is to add more training camera views in the camera trajectory, thereby guiding the noise direction using multi-frame ground truth images. However, even the best current VDMs can only generate very short videos and cannot adapt to complex camera trajectories. Therefore, we ultimately choose to randomly sample two training camera views $v _ { j }$ and $v _ { k }$ from the training view list $\mathcal { V } ~ = ~ \{ v _ { 1 } , v _ { 2 } , . . . , v _ { r } \}$ ; where r is the number of training views, $v _ { j }$ and $v _ { k }$ (with the corresponding ground truth images $\mathbf { y } ^ { j }$ and $\mathbf { y } ^ { k } )$ serve as the first and last frame in the camera trajectory, respectively. However, in our experiments, we find that this setting may limit the viewable area of the VDM, thus we set $v _ { k }$ as the s-th frame of the camera trajectory. Fig. 3 shows their difference. Accordingly, the real image $\mathbf { y } ^ { j }$ is used as the condition for the VDM, and $\mathbf { y } ^ { k }$ is used as the guidance for correcting the s-th frame $\mathbf { x } _ { t } ^ { s }$ . The distance measuring function of time t for all the training views can be easily expressed as:

<!-- image-->  
Figure 3. Comparison of two different trajectory generation methods, where (b) obtains a larger viewing area. We also provide the corresponding relation between noisy image $\mathbf { x } _ { t }$ and two ground truth images $\mathbf { y } ^ { j }$ and $\mathbf { y } ^ { k }$

$$
\begin{array} { r } { \mathcal { D } _ { \theta } ( \mathbf { y } , \mathbf { x } _ { t } ^ { s } ) = \mathbb { E } _ { ( v _ { j } , v _ { k } ) \in \mathcal { V } , j \neq k } \| \mathbf { y } ^ { k } , \mathbf { x } _ { t } ^ { s } \| _ { 1 } , } \end{array}\tag{10}
$$

where V is a set of view combinations, and Y denotes the training image list. However, our experiments show that directly using the pixel-level loss is not effective. This may be because $\mathcal { L } _ { 1 }$ equally focuses on the entire image, whereas what most affects the VDM optimization direction should be the semantic features or precise geometric information of the image. Building on the above insights, we use the distance between DINO features [5] as the distance function, thus the semantic feature guidance can be written:

$$
\mathcal { L } _ { G S D } = \mathcal { D } _ { \theta } ( \mathbf { y } , \mathbf { x } _ { t } ^ { s } ) = \mathbb { E } _ { ( v _ { j } , v _ { k } ) \in \mathcal { V } , j \neq k } \| M ( \mathbf { y } ^ { k } ) , M ( \mathbf { x } _ { t } ^ { s } ) \| _ { 1 } ,\tag{11}
$$

where $M ( \cdot )$ refers to the extraction of DINO feature. The semantic features produced by the DINO network effectively utilizes image information, especially helpful in handling long-distance views and complex scenes.

## 3.4. Warped Depth Guidance

Recent works [40, 41] have developed image reprojection techniques that generate pseudo-view images by remapping pixel values in space between different views. However, due to the overlap and occlusion between objects, the pseudo-view images generated by this method are often incomplete and lack full semantic information. To address this issue, we propose a method that only warps the depth map in space. For a randomly training view $v _ { j } .$ , we can obtain the relative depth map $D ( \mathbf { y } ^ { j } )$ via the corresponding ground truth image $\mathbf { y } ^ { j }$ as follows,

$$
D ( \mathbf { y } ^ { j } ) = F _ { m } ( \mathbf { y } ^ { j } ) ,\tag{12}
$$

where $F _ { m } ( \cdot )$ denotes a pretrained monocular depth estimator. For another random view $p ^ { i }$ in the camera trajectory, we can similarly estimate the depth map $D ( \mathbf { x } _ { t } ^ { i } )$ of the rendered image $\mathbf { x } _ { t } ^ { i }$ . We employ a strategy similar to image reprojection to warp the relative depth $D ( y ^ { j } )$ from the view $v _ { j }$ to the view $p ^ { i }$ . It is worth mentioning that we follow the existing method [40] to estimate the depth range to scale the depth map via the depth of Gaussians. Specifically, for the depth value $d _ { \mathbf { y } ^ { j } }$ at position $( x , y )$ in $D ( \mathbf { y } ^ { j } )$ ), the corresponding depth $d _ { \mathbf { x } _ { t } ^ { i } }$ can be obtained using the following formula:

Table 1. Quantitative results on LLFF [23] with 3, 6, 9 training views. The best, second-best, and third-best entries are marked in red, orange, and yellow, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">PSNRâ</td><td colspan="3">SSIMâ</td><td colspan="3">LPIPSâ</td></tr><tr><td>3-view</td><td>6-view</td><td>9-view</td><td>3-view</td><td>6-view</td><td>9-view</td><td>3-view</td><td>6-view</td><td>9-view</td></tr><tr><td>Mip-NeRF [1]</td><td>16.11</td><td>22.91</td><td>24.88</td><td>0.401</td><td>0.756</td><td>0.826</td><td>0.460</td><td>0.213</td><td>0.160</td></tr><tr><td>DietNeRF [16]</td><td>14.94</td><td>21.75</td><td>24.28</td><td>0.370</td><td>0.717</td><td>0.801</td><td>0.496</td><td>0.248</td><td>0.183</td></tr><tr><td>RegNeRF [26]</td><td>19.08</td><td>23.10</td><td>24.76</td><td>0.587</td><td>0.760</td><td>0.819</td><td>0.336</td><td>0.206</td><td>0.182</td></tr><tr><td>FreeNeRF [42]</td><td>19.63</td><td>23.73</td><td>25.13</td><td>0.612</td><td>0.779</td><td>0.827</td><td>0.308</td><td>0.195</td><td>0.160</td></tr><tr><td>SparseNeRF [35]</td><td>19.86</td><td>23.80</td><td>-</td><td>0.624</td><td>0.814</td><td>-</td><td>0.328</td><td>0.125</td><td>-</td></tr><tr><td>3DGS [19]</td><td>19.22</td><td>23.80</td><td>25.44</td><td>0.649</td><td>0.814</td><td>0.860</td><td>0.229</td><td>0.125</td><td>0.096</td></tr><tr><td>FSGS [48]</td><td>20.31</td><td>24.09</td><td>25.31</td><td>0.652</td><td>0.823</td><td>0.860</td><td>0.288</td><td>0.145</td><td>0.122</td></tr><tr><td>Ours</td><td>20.58</td><td>24.55</td><td>25.79</td><td>0.673</td><td>0.828</td><td>0.866</td><td>0.208</td><td>0.117</td><td>0.083</td></tr></table>

$$
d _ { \mathbf { x } _ { t } ^ { i } } = \operatorname { p r o j } ( R \cdot ( d _ { \mathbf { Y } ^ { j } } \cdot K ^ { - 1 } \cdot \left[ \begin{array} { l } { x } \\ { y } \\ { 1 } \end{array} \right] ) + T ) _ { z } ,\tag{13}
$$

where R and $T$ are the rotation matrix and the translation vector between two views; $K ^ { - 1 }$ refers to the inverse of the camera intrinsic matrix and proj(Â·)z is the function to project a 3D point back to the 2D image plane and extract the z-coordinate from the 3D point after transformation, which corresponds to the depth value. By iterating over all pixels in $\mathbf { y } ^ { j }$ , we can obtain the warped depth map $\widetilde D ( \mathbf { y } ^ { j } )$ at viewpoint $p ^ { i }$ , where the areas that are not mapped automatically forms as a mask(Â·). Thus, we employ a regular depth constraint loss, known as Pearson Correlation Coefficient (PCC) [40, 48] loss for the warped depth maps. Applying Eq. 9, the distance measuring function can be written as:

$$
\begin{array} { r l } & { \mathcal { L } _ { d e p t h } = \mathcal { D } _ { \boldsymbol { \theta } } ( \mathbf { y } , \mathbf { x } _ { t } ) = \mathbb { E } _ { ( v _ { j } , v _ { k } ) \in \mathcal { V } , j \ne k } } \\ & { \displaystyle \sum _ { i = 1 } ^ { n } \left( 1 - P C C \left( m a s k ( \widetilde { D } ( \mathbf { y } ^ { j } ) ) , m a s k ( D ( \mathbf { x } _ { t } ^ { i } ) ) \right) \right) . } \end{array}\tag{14}
$$

With warped depth guidance, the prediction noise can be efficiently corrected for each video frame, which provides basic geometric information constraints for the model update direction.

## 3.5. Loss Function

The final total loss formula can be written as

$$
\mathcal { L } = \mathcal { L } _ { R G B } + \lambda _ { d e p t h } \mathcal { L } _ { d e p t h } + \lambda _ { G S D } \mathcal { L } _ { G S D } ,\tag{15}
$$

where $\lambda _ { G S D }$ and $\mathcal { L } _ { d e p t h }$ are defined in Eq. 11 and Eq. 14 respectively. In addition, we only use the ${ \mathcal { L } } _ { \mathrm { G S D } }$ in the later stages of the 3DGS optimization process, i.e., after 3000 iterations, as depth range estimation quality typically improves after the initial optimization phase.

## 4. Experiment

## 4.1. Setup

Datasets We conducted experiments on three datasets: LLFF [23], Mip-NeRF360 [3] and DTU [17]. For LLFF datasets [23], we select every eighth image as the test set, and evenly sample sparse views from the remaining images for training. Besides, we use an 8x downsampling rate to train on 3, 6, and 9 input views, respectively. For the Mip-NeRF360 dataset, we train on 24 input views with 8x and 4x downsampling rates, respectively. For the DTU dataset, an 8-view configuration is trained with a 4x downsampling rate. To focus on the target object and reduce background noise during evaluation, we apply object masks to DTU, similar to prior works [26].

Baselines. We compare GSD with several few-shot NVS methods on these datasets, including DietNeRF [16], Reg-NeRF [26], FreeNeRF [42], and SparseNeRF [35]. Additionally, we include comparisons with the high-performing Mip-NeRF [1], primarily designed for dense-view training, and point-based 3DGS [19], following its original denseview training recipe. Following [26, 40, 48], we report the average PSNR, SSIM, LPIPS scores for all the methods.

Settings. The video diffusion model we adopt is Stable Video Diffusion [4]. We implemented GSD by the Py-Torch framework, with the initial point cloud computed from SfM only with the training views. During training, we also applied the conventional densification and pruning operations [19, 48]. We densify the Gaussians every 100 iterations and start densification after 500 iterations. The total optimization steps are set to 10,000 on LLFF and DTU datasets and 30,000 on MipNeRF360 datasets. We start to use the $\mathcal { L } _ { G S D }$ after 3,000 iterations. We utilize the pretrained DPT model [28] for depth estimation. All results are obtained using an NVIDIA A40 GPU.

<!-- image-->  
Ground Truth

<!-- image-->  
3DGS

<!-- image-->  
FSGS

<!-- image-->  
Ours

Figure 4. Qualitative comparison on LLFF datasets with 3DGS [19], FSGS [48] and the proposed method. We provide both rendered RGB images and depth maps in the same view for comparsion.  
<!-- image-->  
Ground Truth

<!-- image-->  
3DGS

<!-- image-->  
FSGS

<!-- image-->  
Ours  
Figure 5. Qualitative comparison on MipNeRF360 datasets with 3DGS [19], FSGS [48] and the proposed method. We provide both rendered RGB images and depth maps in the same view for comparsion.

## 4.2. Comparisons

LLFF datasets. We present the quantitative results on the LLFF dataset in Tab 1. Our method consistently achieves the best performance across PSNR, SSIM, and LPIPS metrics with 3, 6, and 9 training views. Depth-supervised FSGS [48] enhances the performance of 3DGS with 3 and 6 views, but cannot avoid geometric ambiguities with more views. Our method incorporates view-consistent video diffusion model priors to prevent the reconstruction of incorrect geometry, leading to improvements across all metrics with various training views. Fig. 4 provides quantitative visualizations of rendered images and depth maps at novel views and Gaussian points. From the rendered depth maps, we observe that the original 3DGS exhibits significant errors in reconstructing distant regions. With depth supervision, FSGS can correct the erroneous depth of 3DGS but still struggles with the reconstruction of image details, such as distant leaves and railings. Our method effectively recovered photometric details, resulting in higher-quality outcomes.

Table 2. Quantitative results on Mip-NeRF360 datasets [3] at 1/8 and 1/4 resolutions. The best, second-best, and third-best entries are marked in red, orange, and yellow, respectively.
<table><tr><td rowspan=1 colspan=5>1/8 Resolution       1/4 ResolutionMethodsPSNRâSSIMâLPIPSâPSNRâSSIMâLPIPSâ</td></tr><tr><td rowspan=1 colspan=3>Mip-NeRF360 [3]21.230.613 0.351 19.78</td><td rowspan=1 colspan=2>0.5300.431</td></tr><tr><td rowspan=2 colspan=3>RegNeRF [26]   22.190.6430.335 20.55FreeNeRF [42]   22.780.6890.323 21.04</td><td rowspan=1 colspan=2>0.5460.398</td></tr><tr><td rowspan=1 colspan=1>0.587</td><td rowspan=1 colspan=1>0.377</td></tr><tr><td rowspan=1 colspan=3>SparseNeRF [35]22.850.6930.315 21.13</td><td rowspan=1 colspan=1>0.600</td><td rowspan=1 colspan=1>0.389</td></tr><tr><td rowspan=1 colspan=3>DietNeRF [16]   20.21 0.5570.387 19.11</td><td rowspan=1 colspan=1>0.482</td><td rowspan=1 colspan=1>0.452</td></tr><tr><td rowspan=1 colspan=3>3D-GS [19]      20.890.6330.317 19.93</td><td rowspan=1 colspan=1>0.588</td><td rowspan=1 colspan=1>0.401</td></tr><tr><td rowspan=1 colspan=1>FSGS [48]</td><td rowspan=1 colspan=2>23.700.745 0.220 22.82</td><td rowspan=1 colspan=1>0.693</td><td rowspan=1 colspan=1>0.293</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>23.740.755</td><td rowspan=1 colspan=1>0.208 22.95</td><td rowspan=1 colspan=1>0.691</td><td rowspan=1 colspan=1>0.277</td></tr></table>

Table 3. Quantitative results on DTU [17] with 3 training views. The best, second-best, and third-best entries are marked in red, orange, and yellow, respectively.
<table><tr><td rowspan=1 colspan=4>Method            PSNRâ  SSIMâ LPIPSâ</td></tr><tr><td rowspan=1 colspan=2>DietNeRF [16]      11.85</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.314</td></tr><tr><td rowspan=1 colspan=1>RegNeRF [26]</td><td rowspan=1 colspan=1>18.89</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>0.190</td></tr><tr><td rowspan=1 colspan=1>Mip-NeRF [1]</td><td rowspan=1 colspan=1>9.10</td><td rowspan=1 colspan=1>0.578</td><td rowspan=1 colspan=1>0.348</td></tr><tr><td rowspan=1 colspan=1>FreeNeRF [42]</td><td rowspan=1 colspan=1>19.92</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>0.182</td></tr><tr><td rowspan=1 colspan=1>SparseNeRF [35]</td><td rowspan=1 colspan=1>19.55</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.201</td></tr><tr><td rowspan=1 colspan=1>3DGS [19]</td><td rowspan=1 colspan=1>17.65</td><td rowspan=1 colspan=1>0.846</td><td rowspan=1 colspan=1>0.146</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>19.33</td><td rowspan=1 colspan=1>0.842</td><td rowspan=1 colspan=1>0.117</td></tr></table>

Table 4. Ablation study of the second camera guidance on the fern scene of the LLFF dataset. The best, second-best, and third-best entries are marked in red, orange, and yellow, respectively.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>w/o 2nd camera guidance</td><td>21.81</td><td>0.68</td><td>0.24</td></tr><tr><td>pixel level guidance</td><td>21.81</td><td>0.69</td><td>0.21</td></tr><tr><td>Ours</td><td>21.99</td><td>0.74</td><td>0.20</td></tr></table>

MipNeRF360 datasets. The quantitative results on the Mip-NeRF360 dataset are provided in Tab 2. Our method consistently achieves the best or second-best performance in PSNR, SSIM, and LPIPS metrics at downsampling rates of 1/8 and 1/4. Notably, the performance improvement brought by our method decreases at higher resolutions, which may be due to the fact that the video diffusion model [4] we used is trained at a lower resolution, leading to lower-quality gradient directions at higher resolutions. Fig. 5 presents quantitative visualizations of rendered images and depth at novel views and Gaussian points. From the figure, we can observe that our method achieves better reconstruction quality at the edges of the scene, which partly benefits from our extended sampling strategy for the

camera trajectory.

DTU datasets. The quantitative results on the DTU dataset are provided in Tab. 3. Our method achieves competitive results. It is worth mentioning that, due to the use of generative model priors, our method consistently maintains the best performance in LPIPS. The relatively lower results in other metrics may be because the training dataset of the video diffusion model contains very few examples of single objects with solid colored backgrounds, which presents a challenge for its generation capabilities.

Table 5. Ablation study of the camera trajectory sampling method on the fern scene of the LLFF dataset.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>limited camera trajectory</td><td>21.94</td><td>0.70</td><td>0.24</td></tr><tr><td>Ours</td><td>21.99</td><td>0.74</td><td>0.20</td></tr></table>

Table 6. Ablation study on the fern scene of the LLFF dataset. The best, second-best, and third-best entries are marked in red, orange, and yellow, respectively.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>w/o 2nd camera guidance</td><td>21.81</td><td>0.68</td><td>0.24</td></tr><tr><td>w/o warp depth guidance</td><td>21.83</td><td>0.71</td><td>0.29</td></tr><tr><td>w/o all guidance</td><td>21.58</td><td>0.64</td><td>0.33</td></tr><tr><td>w/o SDS-DDIM</td><td>21.52</td><td>0.66</td><td>0.31</td></tr><tr><td>Ours</td><td>21.99</td><td>0.74</td><td>0.20</td></tr></table>

## 4.3. Ablation Study

The second camera guidance. We conducted an ablation study on the guidance method for the second camera. We compared semantic-level guidance, pixel-level guidance, and no guidance (see Tab. 4). The experiments indicate that our method achieves optimal performance. Pixel-level guidance may not effectively help correct the noise bias of the latent diffusion model.

The camera sampling method. We also performed an ablation study on the sampling method for camera trajectories (see Tab. 5), confirming that expansive sampling can broaden the visible range and improve model performance. Others components of the model. Furthermore, we conducted an ablation study on the components of the model (see Tab. 6), confirming that both types of guidance we designed play a role. Comparatively, the improvements brought by the guidance for the second camera are more significant. Additional ablation studies and hyperparameter discussions can be found in the supplementary materials.

## 5. Conclusion

In this paper, we propose the Guidance Score Distillation (GSD) framework to address overfitting in 3D

Gaussian Splatting (3DGS) when dealing with few-shot views. Our approach leverages pretrained video diffusion models (VDM) to extract multi-view consistency priors, applying score distillation sampling (SDS) to guide the Gaussian representation toward VDMâs generative direction. We incorporate warping guidance based on real depth maps and additional guidance based on semantic image features to ensure alignment in geometry and camera poses. Experimental results demonstrate that GSD effectively enhances the performance of 3DGS and achieves superior results across multiple datasets.

## References

[1] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 5855â5864, 2021. 6, 8

[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470â5479, 2022. 2

[3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. arXiv preprint arXiv:2111.12077, 2022. 6, 8

[4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 4, 6, 8

[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J Â´ egou, Â´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650â9660, 2021. 2, 5

[6] Hochul Cho, Jangyoon Kim, and Woontack Woo. Novel view synthesis with multiple 360 images for large-scale 6- dof virtual reality system. In 2019 IEEE Conference on Virtual Reality and 3D User Interfaces (VR), pages 880â881. IEEE, 2019. 1

[7] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023. 2, 3, 4

[8] Jaeyoung Chung, Jeongtaek Oh, and Kyoung Mu Lee. Depth-regularized optimization for 3d gaussian splatting in few-shot images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 811â820, 2024. 2

[9] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised nerf: Fewer views and faster training for free. In Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition, pages 12882â 12891, 2022. 2

[10] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. International Conference on Learning Representations, 2024. 2, 4

[11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS 2020, 2020. 4

[12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Diffusion models beat gans on image synthesis. In NeurIPS 2021, 2021. 4

[13] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2, 4

[14] Jonathan Ho, Tim Salimans, Niru Maheswaranathan, Mohammad Amin Sadeghi, and Kevin Murphy. Classifying diffusion models with classifier guidance. In NeurIPS 2022, 2022. 4, 5

[15] Shengyu Huang, Zan Gojcic, Zian Wang, Francis Williams, Yoni Kasten, Sanja Fidler, Konrad Schindler, and Or Litany. Neural lidar fields for novel view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18236â18246, 2023. 1

[16] Ajay Jain, Matthew Tancik, and Pieter Abbeel. DietNeRF: Few-shot novel view synthesis via diffusion of viewpoints. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7226â7235, 2021. 2, 6, 8

[17] Rasmus Jensen, Anders Lindbjerg Dahl, George Vogiatzis, Engin Tola, and Henrik Aanaes. Large scale multi-view stereopsis evaluation. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 406â 413, 2014. 6, 8

[18] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. Noise-free score distillation. arXiv preprint arXiv:2310.17590, 2023. 4

[19] Bernhard Kerbl, Christian Reiser, Zexiang Liao, Georgios Kopanas, Thomas Zhang, Thomas Leimkuhler, Matthias Â¨ Parger, Fabrice Rousselle, Matthias NieÃner, and Petr Kellnhofer. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 42(4):1â 12, 2023. 1, 2, 3, 6, 7, 8

[20] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300â309, 2023. 3

[21] Artem Lukoianov, Haitz Saez de Oc Â´ ariz Borde, Kristjan Â´ Greenewald, Vitor Campagnolo Guizilini, Timur Bagautdinov, Vincent Sitzmann, and Justin Solomon. Score distillation via reparametrized ddim. arXiv preprint arXiv:2405.15891, 2024. 3, 4

[22] Armin Masoumian, Hatem A Rashwan, Julian Cristiano, Â´ M Salman Asif, and Domenec Puig. Monocular depth es-

timation using deep learning: A review. Sensors, 22(14): 5353, 2022. 2

[23] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. In ACM Transactions on Graphics (TOG), pages 1â14. ACM, 2019. 6

[24] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[25] Alexander Mordvintsev, Nicola Pezzotti, Ludwig Schubert, and Chris Olah. Differentiable image parameterizations. Distill, 2018. https://distill.pub/2018/differentiableparameterizations. 4

[26] Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall, Mehdi S. M. Sajjadi, Andreas Geiger, and Noha Radwan. RegNeRF: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5480â5490, 2022. 2, 6, 8

[27] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 1, 2, 4

[28] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi- Â´ sion transformers for dense prediction. ArXiv preprint, 2021.

[29] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1, 2, 4

[30] Yang Song and Stefano Ermon. Score-based generative modeling through stochastic differential equations. In ICLR 2021, 2021. 4

[31] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems, 33:7537â7547, 2020. 2

[32] Adam Tonderski, Carl Lindstrom, Georg Hess, William Â¨ Ljungbergh, Lennart Svensson, and Christoffer Petersson. Neurad: Neural rendering for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14895â14904, 2024. 1

[33] Kuan-Wei Tseng, Jing-Yuan Huang, Yang-Shen Chen, Chu-Song Chen, and Yi-Ping Hung. Pseudo-3d scene modeling for virtual reality using stylized novel view synthesis. In ACM SIGGRAPH 2022 Posters, New York, NY, USA, 2022. Association for Computing Machinery. 1

[34] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439â457. Springer, 2025. 1

[35] Cong Wang, Yinda Zhang, Zhoutong Li, Qi Zhang, Jianmin Zhang, and Hujun Bao. SparseNeRF: Distilling depth ranking for few-shot novel view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 1234â1243, 2023. 2, 6, 8

[36] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12619â12629, 2023. 2, 4

[37] Peihao Wang, Zhiwen Fan, Dejia Xu, Dilin Wang, Sreyas Mohan, Forrest Iandola, Rakesh Ranjan, Yilei Li, Qiang Liu, Zhangyang Wang, et al. Steindreamer: Variance reduction for text-to-3d score distillation via stein identity. arXiv preprint arXiv:2401.00604, 2023. 2

[38] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024. 3, 4

[39] Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. Novel view synthesis with diffusion models. arXiv preprint arXiv:2210.04628, 2022. 1

[40] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay, Pradyumna Chari, and Achuta Kadambi. Sparsegs: Realtime 360Â° sparse view synthesis using gaussian splatting. arXiv preprint arXiv:2312.00206, 2023. 2, 5, 6

[41] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Humphrey Shi, and Zhangyang Wang. Sinnerf: Training neural radiance fields on complex scenes from a single image. In European Conference on Computer Vision, pages 736â753. Springer, 2022. 5

[42] Jingyang Yang, Yinda Zhang, Zhoutong Li, Qi Zhang, Jianmin Zhang, and Hujun Bao. FreeNeRF: Improving few-shot neural rendering with free frequency regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18624â18633, 2023. 6, 8

[43] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4578â4587, 2021. 2

[44] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy-guided conditional diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 1859â1869, 2023. 5

[45] Huangying Zhan, Ravi Garg, Chamara Saroj Weerasekera, Kejie Li, Harsh Agarwal, and Ian Reid. Unsupervised learning of monocular depth estimation and visual odometry with deep feature reconstruction. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 340â349, 2018. 2

[46] Bowen Zhang, Chunyuan Li, Yichen Zhu, Jianfeng Zhang, Jianfeng Liu, and Jianfeng Gao. Universal guidance for diffusion models. arXiv preprint arXiv:2303.08919, 2023. 5

[47] Junzhe Zhu, Peiye Zhuang, and Sanmi Koyejo. Hifa: Highfidelity text-to-3d generation with advanced diffusion guidance. arXiv preprint arXiv:2305.18766, 2023. 2, 3, 4

[48] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In European Conference on Computer Vision, pages 145â163. Springer, 2025. 2, 6, 7, 8