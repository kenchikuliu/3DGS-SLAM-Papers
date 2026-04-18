# Enhancing Novel View Synthesis from extremely sparse views with SfM-free 3D Gaussian Splatting Framework

Zongqi He \* Hanmin Li â  Kin-Chung Chan\* Yushen Zuo\* Hao Xie\* Zhe Xiao\* Jun Xiao\* Kin-Man Lam\*

## Abstract

3D Gaussian Splatting (3DGS) has demonstrated remarkable real-time performance in novel view synthesis, yet its effectiveness relies heavily on dense multi-view inputs with precisely known camera poses, which are rarely available in real-world scenarios. When input views become extremely sparse, the Structure-from-Motion (SfM) method that 3DGS depends on for initialization fails to accurately reconstruct the 3D geometric structures of scenes, resulting in degraded rendering quality. In this paper, we propose a novel SfM-free 3DGS-based method that jointly estimates camera poses and reconstructs 3D scenes from extremely sparse-view inputs. Specifically, instead of SfM, we propose a dense stereo module to progressively estimates camera pose information and reconstructs a global dense point cloud for initialization. To address the inherent problem of information scarcity in extremely sparse-view settings, we propose a coherent view interpolation module that interpolates camera poses based on training view pairs and generates viewpoint-consistent content as additional supervision signals for training. Furthermore, we introduce multi-scale Laplacian consistent regularization and adaptive spatialaware multi-scale geometry regularization to enhance the quality of geometrical structures and rendered content. Experiments show that our method significantly outperforms other state-of-the-art 3DGS-based approaches, achieving a remarkable 2.75dB improvement in PSNR under extremely sparse-view conditions (using only 2 training views). The images synthesized by our method exhibit minimal distortion while preserving rich high-frequency details, resulting in superior visual quality compared to existing techniques.

## 1. Introduction

Novel view synthesis (NVS) is a foundational task in 3D vision, which aims to generate realistic images of unseen viewpoints based on a set of images captured from different known views. Due to its numerous real-world applications, such as virtual reality (VR) [9, 11, 17, 24, 44, 46], augmented reality (AR) [37, 59], and 3D content creation [1, 26, 28, 29, 31, 45], NVS has attracted significant research interest, and many methods have been proposed in the past years.

Neural Radiance Fields (NeRF) [2, 30, 34, 40, 48, 56] and 3D Gaussian Splatting (3DGS) [6, 18, 21, 33, 49, 58] are two leading approaches in NVS that have demonstrated impressive capabilities in synthesizing photorealistic images from novel viewpoints. 3DGS-based methods, in particular, have demonstrated superior performance in realtime rendering and have gradually become dominant in this field. However, both approaches require dense multi-view input images with precise camera pose information during training, which is often difficult to satisfy in real-world applications. When the training views become extremely sparse, the performance of these approaches degrades substantially, with 3DGS-based methods experiencing particularly notable limitations.

Recently, various methods [4, 23, 54, 60] have attempted to address sparse-view problems. DNGaussian [23] applies a soft-hard depth regularization strategy and optimizes Gaussian centers while fine-tuning opacity. Concurrently, Cor-GS [54] deploys dual Gaussian radiance fields with co-pruning and co-regularization processes to enhance the estimation accuracy of 3D Gaussian primitives. Recently, FSGS [60] leverages dense depth maps as regularization during the optimization process, which effectively generates hidden geometric information and improve performance. Chan et al. [4] propose unprojecting estimated dense depth maps into a 3D space, which provides accurate initial geometry structure of the 3D scene and improves the quality of neural rendering. However, these methods encounter significant constraints when confronted with more challenging real-world scenarios.

The performance of 3DGS models is inherently sensitive to the quality of point cloud initialization obtained through structure-from-motion (SfM). In extremely sparse-view settings, such as when only two viewpoints are available in the

Tanks and Temples dataset [19], SfM lacks sufficient information to accurately infer 3D geometric structures, often leading to corrupted content in synthesized images. Despite incorporating various geometry priors, existing methods still struggle to produce satisfactory results in complex environments, as shown in Figure ??. More critically, these methods commonly assume known camera pose information, which is indispensable for SfM to accurately initialize point clouds. This assumption is too restrictive for real-world applications in unconstrained environments where precise camera calibration is rarely available. Therefore, novel view synthesis from extremely sparse-view inputs with unknown camera poses remains a challenging yet crucial problem for practical 3D scene reconstruction.

To this end, we propose a robust 3DGS-based method for novel view synthesis that effectively handles extremely sparse-view inputsâeven as few as two views in some situationsâwith unknown camera poses. Unlike previous methods that rely on SfM for initialization, we propose a dense stereo module (DSM) that progressively estimates necessary camera pose information and reconstructs a global dense point cloud as initialization. This makes our method essentially SfM-free and significantly more robust to extremely sparse-view scenarios. While previous SfM-free 3DGS-based methods [12, 16] initialize local 3D Gaussian primitives using inaccurate mono-depth estimation, our proposed DSM leverages a transformer backbone to estimate dense pointmaps from image pairs, which substantially improves geometric structure accuracy and results in better camera pose estimation. Furthermore, we propose a coherent view interpolation module that interpolates camera poses based on training view pairs and harnesses generative priors through a video diffusion model to produce consistent views for the interpolated viewpoints, providing additional supervised signals during training. To enhance robustness in extremely sparse-view scenarios, where reconstructed 3D models typically produce poor-quality geometrical structures and rendering features, we propose two complementary regularization techniques: multi-scale Laplacian consistent regularization (MLCR) and adaptive spatial-aware multi-scale geometry regularization (ASMG). MLCR adopts Laplacian pyramids to decompose input images into multiple frequency subbands and ensures each subband of the rendered images closely matches the corresponding high-quality interpolated view, thereby reducing distorted content and enhancing high-frequency details in synthesized outputs. Complementarily, ASMG leverages multi-scale depth priors with a spatial-aware mask that emphasizes high-accuracy foreground content while applying an adaptive weighting strategy to control ASMGâs impact throughout different optimization stages.

The main contributions of this paper can be summarized as follows:

â¢ We propose a novel robust and SfM-free 3DGS-based approach for novel view synthesis with extremely sparseview inputs, which introduces a dense stereo module to progressively estimate camera pose information and reconstruct a global dense point cloud as initialization.

â¢ To address the inherent information scarcity in extremely sparse-view settings, we develop a coherent view interpolation module that interpolates camera poses between training view pairs and generates corresponding multiview consistent content as additional supervision signals during training.

â¢ We introduce multi-scale Laplacian consistent regularization (MLCR) and adaptive spatial-aware multi-scale geometry regularization (ASMG) to enhance geometrical structure quality and rendering features in sparse-view scenarios.

â¢ Experimental results demonstrate that our method significantly outperforms existing approaches in both reconstruction quality and computational efficiency, providing robust and high-quality 3D reconstructions from minimal input views. Even in the most challenging scenarios with only two viewpoints, our method produces promising results.

## 2. Related works

## 2.1. SfM-based Novel View Synthesis Methods

Recent advances in novel view synthesis (NVS) have been primarily driven by approaches such as Neural Radiance Fields (NeRF) [2, 30, 34, 40, 48, 56] and 3D Gaussian Splatting (3DGS) [6, 18, 21, 33, 49, 58]. While these methods demonstrate impressive performance when provided with dense multi-view images, their performance significantly degrades under sparse-view conditions due to insufficient geometric information provided during optimization [3, 55].

In particular, most 3DGS-based methods critically depend on initializing Gaussian primitives using point clouds derived from Structure-from-Motion (SfM) pipelines such as COLMAP, so the performance of 3DGS is directly tied to the quality of this initial point cloud representation. While effective with dense multi-view inputs, SfM methods fail to extract sufficient geometric information from extremely sparse views, leading to sparse, incomplete, and noisy point clouds. This fundamental limitation significantly compromises the representational capacity of 3D Gaussian primitives and severely hinders the capture of fine-grained geometric details during optimization [5, 10]. Consequently, 3DGS models trained on sparse views often exhibit severe overfitting to the limited training viewpoints and poor generalization to unseen views, resulting in corrupted content in synthesized images. To address sparse-view limitations, researchers have proposed various priors incorporated into the training process. Several works have explored semantic regularization [15, 47], smoothness priors [35, 51], and geometric constraints [8, 23, 38, 50, 53] to enhance reconstruction quality. More recent methods, such as FSGS [60] and SIDGS [13], improve depth map regularization using monocular depth priors.

However, these approaches often rely on relatively dense inputs and assume the availability of ground-truth camera poses, limiting their applicability in real-world scenarios.

## 2.2. SfM-Free Rendering Methods

As extremely sparse-view inputs significantly degrade the performance of SfM and lead to inaccurate camera pose estimation, many researchers have attempted to explore SfMfree methods in recent years. For example, CF-3DGS [12] locally estimates the relative pose between camera pairs and iteratively concatenates these camera relationships to reconstruct the complete camera pose configuration. Building upon this approach, SF-3DGS [16] proposes a hierarchical pipeline for optimizing 3D Gaussian primitives. Specifically, it employs a similar methodology to CF-3DGS to estimate local relative camera poses and local 3DGS models, which are then progressively integrated into a global 3D representation. However, these methods demonstrate limited generalizability across diverse datasets.

More recently, Dust3R [42] introduced a pipeline for reconstructing dense point maps and camera poses. This method first leverages transformer architectures to estimate dense depth maps from pairs of coherent frames, and then reconstructs camera poses by minimizing the reprojection error of points observed from different viewpoints. Building on Dust3R, Mast3R [22] improves performance in terms of dense point map accuracy and camera pose estimation by jointly training a feature matching component and proposing a fast yet effective Nearest Neighborhood method. Despite these advances, these approaches face significant challenges when applied to 3DGS from extremely sparse-view inputs. Due to substantial changes in view content and failures in region matching between input image pairs, their performance remains limited and often generates artifacts.

## 2.3. Diffusion-based Generative Prior for NVS

The emergence of diffusion-based generative models have achieved unprecedent successes in many vision tasks, such as image generation [7], image editing [14], and image restoration [25]. Recently, many researchers have attempted to leverage a frozen diffusion model trained on 2D images and âdistillâ the generative piror knowledge into 3D reconstruction [25, 27, 36]. However, these methods commonly suffer from the over-smoothing issue, leading to generative images with fewer details. To handle this issue, ProfileDreamer [43] leverages an additional diffusion model to specifically denoise the current 3D shape estimation. However, fine-tuning the second diffusion model is cumbersome and significantly increase the model complexity, which is time-consuming. 3DGS-Enhancer [32] restores multi-view images rendered by the initial 3DGS using a pre-trained conditional video diffusion network. Although this method improves rendering quality to some extent, its workflow relies on first generating each view and then post-processing them together, so the enhanced images remain independent 2D views. Other studies [39, 41, 52] focus on viewconsistent content generation and quality enhancement. Although their results can applied to train NeRF-based models, they commonly assume that input views contain sufficient information of 3D geometric structures. When a large portion of the 3D content is inaccessible in all input views, the synthesized content of the unseen views will be unavoidably deteriorated, leading to noticeable artifacts.

In contrast to previous studies, we propose a robust and SfM-free 3DGS method for novel view synthesis with extremely sparse view inputs, without requiring camera pose information. To address this challenging problem, we propose a coherent view interpolation module that interpolates camera poses between training view pairs and harnesses generative priors through a video diffusion model. This module produces consistent views for the interpolated viewpoints, providing additional supervision signals during training that significantly improve reconstruction quality under extremely sparse-view conditions.

## 3. Methedology

In this paper, we propose a robust and SfM-free 3DGS method to handle extremely sparse-view inputs for novel view synthesis. The overall pipeline of our proposed method is shown in Figure 1. Our proposed dense stereo module (DSM) first progressively estimates camera poses from the given sparse inputs and produces a global dense point cloud for initialization. Then, our coherent view interpolation module leverages a video diffusion model to produce view-consistent content during training. To effectively update the model parameters, we augment the original 3DGS optimization objective with our proposed MLCR and ASMG.

## 3.1. Preliminary: 3D Gaussian Splatting

3D Gaussian Splatting is a promising real-time rendering approach for 3D reconstruction and view synthesis. In this approach, each scene is represented by a large number of 3D Gaussians, where each Gaussian is modeled as an ellipsoidal object in the 3D space. The opacity contribution of a Gaussian at a given position $\pmb { x } \in \mathbb { R } ^ { 3 }$ is defined by the following 3D Gaussian function:

$$
G ( x ) = \exp \left( - { \frac { 1 } { 2 } } ( { \pmb x } - { \pmb \mu } ) ^ { \mathsf { T } } \Sigma ^ { - 1 } ( { \pmb x } - { \pmb \mu } ) \right) ,\tag{1}
$$

<!-- image-->  
Figure 1. The overall pipeline of our method. We first ulitize a dense stereo module to estimate necessary camera poses and reconstruct a global dense point cloud for 3DGS initialization. Afterward, our coherent view interpolation (CVI) module interpolates camera poses between training view pairs and generates corresponding multi-view consistent content as additional supervision signals during training. For 3DGS from sparse-view inputs, we leverage multi-scale Laplacian consistent regularization (MLCR) and adaptive spatial-aware multiscale geometry regularization (ASMG) to enhance geometrical structure quality and rendering features in sparse-view scenarios.

where $\mu \in \mathbb { R } ^ { 3 }$ denotes the center (or origin) of the Gaussian and $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ is a positive semi-definite covariance matrix that represents its spatial extent. In practice, Î£ can be further factorized as:

$$
\Sigma = R S S ^ { \mathsf { T } } R ^ { \mathsf { T } } ,\tag{2}
$$

where R is a rotation matrix that aligns the Gaussian with the local geometry and S is a diagonal scaling matrix that controls its size.

For rendering, 3DGS leverages a collection of 3D Gaussians that are projected into 2D image space based on depth information during the splatting process. Prior to rendering, the 2D covariance matrix $\Sigma ^ { \prime }$ is computed by:

$$
\Sigma ^ { \prime } = J W \Sigma W ^ { \mathsf { T } } J ^ { \mathsf { T } } ,\tag{3}
$$

where J is the Jacobian of the affine approximation of the projective transformation and W is the view transformation matrix. This projection accounts for perspective effects, ensuring that each 3D Gaussian is accurately mapped to an ellipse on the image plane

The final pixel color is synthesized by conducting pointbased Î± blending. Specifically, for a given pixel p, the color $\mathbf { C } _ { p }$ is computed as follows:

$$
\mathbf { C } _ { p } = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{4}
$$

where $\alpha _ { i }$ and $c _ { i }$ are the blending weight and the color coefficient of the i-th 3D Gaussian, respectively; N indicates the total number of 3D Gaussians. The synthesized performance of 3D Gaussian Splatting inevitably deteriorates when the number of available views is reduced.

## 3.2. Coherent View Interpolation Module

Compared with conventional problems of sparse-view rendering, extremely sparse-view inputs (i.e., two input views) provide limited contextual information for inferring the geometric structure of a 3D scene. Inspired by the impressive performance of video diffusion models in generating high-quality and consistent videos, we propose a Coherent View Interpolation (CVI) module to handle this challenge. The proposed module first interpolates camera poses based on training view pairs and then leverage a video diffusion model to synthesize view-consistent content of the interpolated views. The synthesized views maintain multi-view consistency, which effectively compensate for insufficient information in extremely sparse-view settings and provide strong supervision signals during training.

Specifically, we leverage a dense stereo module to estimate camera parameters (including intrinsics and extrinsic) and generate dense 3D point clouds for 3D Gaussian Primitive initialization. Given two training views $\mathbf { C } ^ { \mathrm { t r a i n } } = \{ C _ { i } ^ { \mathrm { t r a i n } } , C _ { j } ^ { \mathrm { t r a i n } } \}$ and the corresponding captured images $\mathbf { I } ^ { \mathrm { t r a i n } } = \{ I _ { i } ^ { \mathrm { t r a i n } } , I _ { j } ^ { \mathrm { t r a i n } } \}$ , we apply B-spline interpolation to generate a smooth trajectory between the training views, denoted by $\mathbf { C } ^ { \mathrm { I n t e r } } = \{ \bar { C _ { 0 } ^ { \mathrm { I n t e r } } } , \cdot \cdot \cdot , C _ { L - 1 } ^ { \mathrm { I n t e r } } \}$ , where L denotes the number of the interpolated views. By projecting the dense 3D point cloud onto 2D image space, we obtain a sequence of rendered RGB pointmaps, denoted by $\mathbf { P } ^ { \mathrm { I n t e r } } =$ $\{ \bar { P _ { 0 } ^ { \mathrm { I n t e r } } } , \cdot \cdot \cdot , P _ { L - 1 } ^ { \mathrm { I n t e r } } \}$ }. Finally, we employ a pre-trained conditional video diffusion model $p ( \cdot )$ to generate synthesized content for the corresponding interpolated views ${ \bf I } ^ { \mathrm { I n t e r } } =$ $\{ I _ { 0 } ^ { \mathrm { I n t e r } } , \cdot \cdot \cdot , I _ { L - 1 } ^ { \mathrm { I n t e r } } \}$ based on these rendered images P and the training images $\mathbf { I } ^ { \mathrm { t r a i n } } , \mathrm { i . e . , \mathbf { I } ^ { \mathrm { I n t e r } } } \sim p ( \mathbf { I } ^ { \mathrm { I n t e r } } | \mathbf { I } ^ { \mathrm { t r a i n } } , \mathbf { P } ^ { \mathrm { I n t e r } } )$ The resulting views inherit the geometry of the dense point clouds while maintaining multi-view consistency.

## 3.3. Multi-scale Laplacian Consistent Regularization

The synthesized pseudo views generated by the CVI module provide additional contextual information. However, these pseudo views suffer from the over-smoothing issue, leading to fine-grained detail loss in high-frequency regions. To address this issue, we propose a multi-scale Laplacian consistent regularization for 3DGS optimization for the pseudo views, which separates low-frequency and high-frequency information of the rendered images, facilitating 3DGS capturing fine-grained spatial features. Specifically, we apply the Laplacian pyramid to the synthesized images and the corresponding rendered images, leading to their decomposed components as follows:

$$
L ^ { ( i ) } = I ^ { ( i ) } - U ( D ( I ^ { ( i ) } ) ) ,\tag{5}
$$

where $L ^ { ( i ) }$ denotes the Laplacian component at level $i , I ^ { ( i ) }$ is the Gaussian-blurred image at the same level, $D ( \cdot )$ denotes the downsampling operation, and $U ( \cdot )$ denotes the bilinear upsampling operation.

With this regularization, our method separately regularizes the low-frequency content and high-frequency details of the rendered images by penalizing different components of the Laplacian pyramid, which is defined as follows:

$$
\mathcal { R } _ { \mathrm { L a p } } = \sum _ { i = 0 } ^ { L } w _ { i } \left\| L _ { r } ^ { ( i ) } - L _ { s } ^ { ( i ) } \right\| _ { 1 } ,\tag{6}
$$

where $L _ { s } ^ { ( i ) }$ and $L _ { r } ^ { ( i ) }$ are the Laplacian components the synthesized images and the rendered images on level i, respectively, and $w _ { i }$ is the weight of the L1 loss on level i.

## 3.4. Adaptive Spatial-aware Multi-scale Geometry Regularization

To address the geometric corruption in the 3D scenes, we propose a adaptive spatial-aware multi-scale geometry regularization. Overall, the regularization use a adaptive strategy, which progressively increase the impact of accurate depth priors. Specifically, we measure the multi-scale difference, between the rendering depth $D _ { \mathrm { r e n d } } ^ { ( s ) }$ and the estimated depth $D _ { \mathrm { r e f } } ^ { ( s ) }$ based on the interpolated views from CVI , by using Pearson correlation. The regular multi-scale geometry terms can be formulated as follows:

$$
\mathcal { R } _ { \mathrm { d e p t h } } = \sum _ { s \in S } w _ { s } \cdot \mathcal { R } _ { \mathrm { c o r r } } ^ { ( s ) } ,\tag{7}
$$

where $w _ { s }$ is the weight scale and $\mathcal { R } _ { \mathrm { c o r r } } ^ { ( s ) }$ is defined as

$$
\mathcal { R } _ { \mathrm { c o r r } } ^ { ( s ) } = \left. \mathbf { C o r r } \left( D _ { \mathrm { r e n d } } ^ { ( s ) } , D _ { \mathrm { r e f } } ^ { ( s ) } \right) \right. _ { 1 } ,\tag{8}
$$

measures the Pearson correlation, $C o r r ( \cdot , \cdot )$ , between two depth maps. Additionally, we propose a spacial-aware term, which capture the accurate depth information in the foreground regions and mask out the erroneous depth in distant regions. We formulate the spacial-aware term as follows:

$$
\mathcal { R } _ { \mathrm { d e p t h } } ^ { \mathrm { m a s k e d } } = \sum _ { s \in S } w _ { s } \cdot \left. \mathrm { C o r r } \left( M \odot D _ { \mathrm { r e n d } } ^ { ( s ) } , M \odot D _ { \mathrm { r e f } } ^ { ( s ) } \right) \right. _ { 1 } ,\tag{9}
$$

where $\odot$ denotes element-wise multiplication and $M$ denote the spatial-aware mask where the normalized $D _ { \mathrm { r e f } } ^ { ( s ) } <$ $0 . 4 .$

The overall Depth-aware Geometry Regularization term is formulated as follows:

$$
\mathcal { R } _ { \mathrm { t o t a l } } = \mathcal { R } _ { \mathrm { d e p t h } } + \beta \cdot \eta ( t ) \cdot \mathcal { R } _ { \mathrm { d e p t h } } ^ { \mathrm { m a s k e d } } ,\tag{10}
$$

During training, we particularly address the weight of the spatial-aware term in a adaptive manner. Specifically, we set the balancing weight $\beta = 0$ for a globally preliminary geometry reconstruction, when the iteration $t < \alpha T$ where $T$ denotes the total training iterations. We set $\beta > 0$ afterward, to introduce the spatial-aware term for geometry refinement. Additionally, the monotonically decreasing function:

$$
\eta ( t ) = \operatorname* { m a x } \left( 0 . 5 , 1 . 0 - \frac { t - \alpha T } { 0 . 5 T } \right) ,\tag{11}
$$

progress decreases the impact of the spatial-aware terms, removing the constrained supervision and allowing the model to refine based on the high-quality geometry.

## 3.5. Loss function

The total loss function of our proposed method is defined as follows:

$$
L = \lambda _ { 1 } L _ { 1 } + ( 1 - \lambda _ { 1 } ) S S I M + \lambda _ { 2 } \mathcal { R } _ { \mathrm { L a p } } + \lambda _ { 3 } \mathcal { R } _ { \mathrm { t o t a l } } ,\tag{12}
$$

where $\lambda _ { 1 } , \lambda _ { 2 }$ and $\lambda _ { 3 }$ are the weighting scales.

## 4. Experimental

Baselines. We compare our method on methods initialization from COLMAP, including basic 3D-GS, FSGS and SIDGS. Additionally, we comparisons on pose-free methods including CF-3DGS [12], which locally optimize the gaussian primitives and camera pose. InstantSplat [10] initializes Gaussian primitives from MASt3Râs point cloud and optimizes 3D Gaussians and camera poses jointly. 3D-GS initialization from DUSt3R [42] and 3D-GS initialization from MASt3R [22].

## 4.1. Experimental Setup

Dataset: We evaluate our method on the Tanks and Temples dataset [20]. Specifically, we test our method on 8 scenes containing indoor and outdoor scenes. We equally sample the test set every 8th frames, and then sample 12 views for

Table 1. Quantitative comparison on the Tanks and Temples [20] dataset under sparse-view settings (2, 3, and 6 input views). The best, second-best, and third-best results are highlighted in red, orange, and yellow, respectively.
<table><tr><td>Tanks and Temples</td><td colspan="3">2 views</td><td colspan="3">3 views</td><td colspan="3">6 views</td></tr><tr><td></td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>COLMAP + 3DGS</td><td>13.89</td><td>0.424</td><td>0.476</td><td>15.18</td><td>0.539</td><td>0.393</td><td>17.70</td><td>0.682</td><td>0.285</td></tr><tr><td>COLMAP + FSGS</td><td>14.63</td><td>0.474</td><td>0.451</td><td>16.44</td><td>0.547</td><td>0.433</td><td>21.73</td><td>0.736</td><td>0.257</td></tr><tr><td>COLMAP + SIDGS</td><td>14.85</td><td>0.495</td><td>0.447</td><td>16.61</td><td>0.594</td><td>0.418</td><td>21.80</td><td>0.785</td><td>0.222</td></tr><tr><td>CF-3DGS</td><td>13.05</td><td>0.361</td><td>0.488</td><td>14.20</td><td>0.398</td><td>0.458</td><td>18.40</td><td>0.556</td><td>0.466</td></tr><tr><td>InstantSplat</td><td>16.64</td><td>0.505</td><td>0.343</td><td>19.13</td><td>0.587</td><td>0.267</td><td>22.78</td><td>0.712</td><td>0.169</td></tr><tr><td>DUSt3R + 3DGS</td><td>14.76</td><td>0.416</td><td>0.414</td><td>16.76</td><td>0.505</td><td>0.338</td><td>19.15</td><td>0.630</td><td>0.256</td></tr><tr><td>MASt3R + 3DGS</td><td>16.99</td><td>0.509</td><td>0.388</td><td>20.07</td><td>0.628</td><td>0.292</td><td>24.11</td><td>0.767</td><td>0.237</td></tr><tr><td>Ours</td><td>19.74</td><td>0.592</td><td>0.374</td><td>22.24</td><td>0.677</td><td>0.286</td><td>25.17</td><td>0.787</td><td>0.208</td></tr></table>

<!-- image-->  
Figure 2. Visual comparison with different methods on Tanks and Temples [20] Dataset on 2 views setting.

testing, and the remaining images are the training images.   
We select 2, 3, and 6 views for the training images.

Table 2. Runtime comparison between the proposed method and other methods.
<table><tr><td></td><td>Training iteration</td><td>Training time (s)</td><td>Training time per iter</td><td>Inference FPS</td><td>Inference time (ms)</td><td>Number of Gaussian</td></tr><tr><td>3DGS</td><td>30000</td><td>242</td><td>0.008</td><td>334</td><td>3.0</td><td>320479</td></tr><tr><td>FSGS</td><td>10000</td><td>540</td><td>0.054</td><td>501</td><td>2.0</td><td>221417</td></tr><tr><td>InstantSplat_align</td><td>1000</td><td>60</td><td>0.060</td><td>106</td><td>9.4</td><td>400323</td></tr><tr><td>InstantSplat</td><td>1000</td><td>60</td><td>0.060</td><td>349</td><td>2.9</td><td>400323</td></tr><tr><td>Ours without CVI</td><td>6000</td><td>140</td><td>0.023</td><td>574</td><td>1.7</td><td>99774</td></tr><tr><td>Ours</td><td>6000</td><td>254</td><td>0.042</td><td>538</td><td>1.9</td><td>116134</td></tr></table>

We first generate all the camera poses with dense stereo model for including training and testing. Then, we generate training initialization point cloud from sparse input views. For quantitative comparisons, we report PSNR, SSIM, and LPIPS [57] scores.

Implementation Details. The number of iterations is fixed as $6 \times 1 0 ^ { 3 }$ . The parameter Î»1 is set to 0.8, Î»2 is set to 1, Î»3 is set to 0.5. For the dense stereo module, the input is configured with a resolution of 512. All experiments were conducted on an NVIDIA RTX 4090 GPU.

## 4.2. Experiments on Tanks and Temples

We evaluate our method on the Tanks and Temples dataset. As shown in Table 1 our method outperforms all baselines across all metrics. As shown in Figure 2. 3DGS show strong artifacts, such as Family and Francis. Although COLMAP can provide accurate camera positions, This reconstruction quality is limited by the extremely sparse input. CF-3DGS and InstantSplat designed to optimize the camera pose, finding the best view points. However, their ability are limited when handling sparse information input, they are finding the local best results, as seen in Barn and Francis scenes, they provide the misaligned outputs. Our coherent view interpolation module and handle such issue, providing sufficient inputs. DUSt3R can not provide accurate camera pose, while MASt3R can not provide the correct geometry initialization on unseen view, as seen in the eave in the Barn scene. Our adaptive spatial-aware multiscale geometry regularization can provide sufficient details. We also evaluate our method on the Mip-NeRF dataset [3], please refer to supplementary 7 for more details.

## 4.3. Runtime Analysis

To evaluate the computational overhead of the proposed method, we evaluated the training and inference performance of our method on the Tanks and Temples dataset. As shown in the table 2, in the extreme case of using only two training images, our method takes approximately six minutes to complete training, with an average runtime of 1.9 milliseconds per image rendering, corresponding to 538 FPS. The number of 3D Gaussian functions used is comparable to that of the standard 3DGS method. In contrast, existing works such as FSGS report a training time of 9 minutes under similar conditions, with an inference speed of 500 FPS, Our results show that introducing CVI and regularization terms only incurs a controllable time overhead, with inference speed remaining largely real-time. Overall, the proposed method achieves good efficiency while maintaining high visual quality and geometric consistency. More detailed results can be found in Table 2.

## 5. Ablation Study

## 5.1. General Effectiveness of Each Component

To validate the contribution of each proposed component, we conduct ablation experiments on the Tanks and Temples dataset. We evaluate three key modules: (1) the coherent view interpolation (CVI) module, (2) the multi-scale Laplacian consistent regularization (MLCR), and (3) the adaptive spatial-aware multi-scale geometry regularization (ASMG). Quantitative results are reported in Table 3, and qualitative examples are illustrated in Figure 3.

The CVI module provides pseudo-supervision by interpolating novel views along smooth camera trajectories. With only two sparse input views, the baseline 3DGS produces incomplete geometry and blurred textures. Incorporating CVI significantly improves rendering quality, as the interpolated views supplement the missing visual evidence. However, since the video diffusion model occasionally hallucinates unseen content, the generated pseudo-views may introduce local distortions or over-smoothing.

To address these issues, we further integrate the MLCR module, which enforces frequency-domain consistency between rendered and synthesized images across multiple scales. While the numerical gain from MLCR alone is moderate, visual comparisons show clearer edges, sharper textures, and more stable fine details. This demonstrates that MLCR effectively counteracts the over-smoothing artifacts introduced by CVI, ensuring structural fidelity in highfrequency regions.

Table 3. Quantitative results of the ablation studies on the proposed components.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PSNRâ</td><td rowspan=1 colspan=1>SSIMâ</td><td rowspan=1 colspan=1>LPIPSâ</td></tr><tr><td rowspan=1 colspan=1>Baseline</td><td rowspan=1 colspan=1>16.99</td><td rowspan=1 colspan=1>0.509</td><td rowspan=1 colspan=1>0.388</td></tr><tr><td rowspan=1 colspan=1>CVI + L1</td><td rowspan=1 colspan=1>19.21</td><td rowspan=1 colspan=1>0.578</td><td rowspan=1 colspan=1>0.382</td></tr><tr><td rowspan=1 colspan=1>CVI + MLCR</td><td rowspan=1 colspan=1>19.52</td><td rowspan=1 colspan=1>0.589</td><td rowspan=1 colspan=1>0.379</td></tr><tr><td rowspan=1 colspan=1>CVI + MLCR + ASMG</td><td rowspan=1 colspan=1>19.74</td><td rowspan=1 colspan=1>0.592</td><td rowspan=1 colspan=1>0.374</td></tr></table>

<!-- image-->  
Figure 3. Qualitative results of the ablation studies on the incremental components discussed in Table 3. Particularly, our coherent view interpolation (CVI) module significantly improves the rendering performance from the baseline model and adaptive spatial-aware multi-scale geometry regularization (ASMG) effectively regularizes the geometry reconstruction, drastically reducing the distortion.

## 5.2. Effectiveness of Adaptive Spatial-aware Multiscale Geometry Regularization

We further investigate the role of ASMG by comparing it with existing depth-based regularization strategies, as robust geometric supervision is crucial for 3D consistency under sparse views. Specifically, we compare ASMG with: (1) $L _ { 1 }$ depth regularization (L1), (2) Pearson correlation between depth maps (PearsonCorr), and (3) multi-scale Pearson correlation (MS-PearsonCorr). The quantitative results are reported in Table 4, and visual comparisons are shown in Figure 4.

The results highlight clear differences across strategies. L1 fails to constrain geometry, leading to severe distortions and reconstruction collapse. PearsonCorr enforces global correlation, yielding better results but still suffers from noticeable artifacts. MS-PearsonCorr captures both global and local structure, further improving geometry quality, but inconsistencies remain, particularly in challenging occluded regions.

By contrast, our proposed ASMG achieves the best performance, with accurate recovery of both foreground and background geometry. The advantage stems from two factors: (i) multi-scale depth supervision, which captures geometry across different resolutions, and (ii) an adaptive spatial mask, which dynamically increases the weight of foreground regions during training. This adaptive scheduling prevents unstable optimization while effectively suppressing depth distortions and floating artifacts. Visual comparisons confirm that ASMG produces the most coherent geometry and sharpest object boundaries.

Table 4. Quantitative comparison of different depth regularization strategies.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PSNRâ</td><td rowspan=1 colspan=1>SSIMâ</td><td rowspan=1 colspan=1>LPIPSâ</td></tr><tr><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>14.06</td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=1>0.654</td></tr><tr><td rowspan=1 colspan=1>PeasonCorr</td><td rowspan=1 colspan=1>19.40</td><td rowspan=1 colspan=1>0.576</td><td rowspan=1 colspan=1>0.389</td></tr><tr><td rowspan=1 colspan=1>MS-PeasonCorr</td><td rowspan=1 colspan=1>19.58</td><td rowspan=1 colspan=1>0.591</td><td rowspan=1 colspan=1>0.376</td></tr><tr><td rowspan=1 colspan=1>ASMG</td><td rowspan=1 colspan=1>19.74</td><td rowspan=1 colspan=1>0.592</td><td rowspan=1 colspan=1>0.374</td></tr></table>

<!-- image-->  
Figure 4. Qualitative results of the ablation studies on the impact of different geometrical regularization. Compare with other methods, our adaptive spatial-aware multi-scale geometry regularization can perceive multi-scale depth information, effectively reconstruct accurate geometry.

## 6. Conclusion

In this paper, we propose a robust 3D Gaussian Splattingbased method for extremely sparse-view input with unknown camera poses. The proposed method includes a dense stereo module, for accurate camera information restoration and dense 3D point cloud estimation, and a coherent view interpolation (CVI) module. Specifically, CVI interpolates extra camera pose based on the training view camera pairs and leverages generative priors through a video diffusion model to produce consistent views for the interpolated viewpoints, providing additional supervised signals during training. Additionally, we propose two efficient regularization techniques tailored for our interpolated views, including multi-scale Laplacian consistent regularization (MLCR) and adaptive spatialaware multi-scale geometry regularization (ASMG), which complementarily enhance geometrical structure quality and rendering features in sparse-view scenarios. MLCR leverages the subband decomposition ability of Laplacian pyramids to render images that closely match the corresponding high-quality interpolated view, and ASMG integrates spatial-aware multi-scale depth, which focus on foreground accurate content, with an adaptive weighting strategy to control ASMGâs impact throughout different optimization stages. Experiments demonstrate that our method improves novel view rendering performance by 2.75 dB in PSNR compared to baseline approaches, significantly enhancing the fidelity of synthesized scenes even in extremely sparse-view scenarios.

## References

[1] Tewodros Amberbir Habtegebrial, Varun Jampani, Orazio Gallo, and Didier Stricker. Generative view synthesis: From single-view semantics to novel-view images. Advances in neural information processing systems, 33:4745â4755, 2020. 1

[2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In ICCV, pages 5855â5864, 2021. 1, 2

[3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, pages 5470â 5479, 2022. 2, 7

[4] Kin-Chung Chan, Jun Xiao, Hana Lebeta Goshu, and Kinman Lam. Point cloud densification for 3d gaussian splatting from sparse input views. In ACM MM. 1

[5] Shen Chen, Jiale Zhou, Zhongyu Jiang, Tianfang Zhang, Zongkai Wu, Jenq-Neng Hwang, and Lei Li. Scalinggaussian: Enhancing 3d content creation with generative gaussian splatting. arXiv preprint arXiv:2407.19035, 2024. 2

[6] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023. 1, 2

[7] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak Shah. Diffusion models in vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9):10850â10869, 2023. 3

[8] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised nerf: Fewer views and faster training for free. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12882â 12891, 2022. 3

[9] Nianchen Deng, Zhenyi He, Jiannan Ye, Budmonde Duinkharjav, Praneeth Chakravarthula, Xubo Yang, and Qi Sun. Fov-nerf: Foveated neural radiance fields for virtual reality. IEEE Transactions on Visualization and Computer Graphics, 28(11):3854â3864, 2022. 1

[10] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds. arXiv preprint arXiv:2403.20309, 2(3):4, 2024. 2, 5

[11] Linus Franke, Laura Fink, and Marc Stamminger. Vrsplatting: Foveated radiance field rendering via 3d gaussian splatting and neural points. arXiv preprint arXiv:2410.17932, 2024. 1

[12] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20796â20805, 2024. 2, 3, 5

[13] Zongqi He, Zhe Xiao, Kin-Chung Chan, Yushen Zuo, Jun Xiao, and Kin-Man Lam. See in detail: Enhancing sparseview 3d gaussian splatting with local depth and semantic regularization. arXiv preprint arXiv:2501.11508, 2025. 3

[14] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Liangliang Cao, and Shifeng Chen. Diffusion model-based image editing: A survey. arXiv preprint arXiv:2402.17525, 2024. 3

[15] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5885â5894, 2021. 3

[16] Bo Ji and Angela Yao. Sfm-free 3d gaussian splatting via hierarchical training. arXiv preprint arXiv:2412.01553, 2024. 2, 3

[17] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng Gao, Yin Yang, et al. Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In ACM SIG-GRAPH 2024 Conference Papers, pages 1â1, 2024. 1

[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 1, 2

[19] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017. 2

[20] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36 (4):1â13, 2017. 5, 6

[21] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In CVPR, pages 21719â21728, 2024. 1, 2

[22] Vincent Leroy, Yohann Cabon, and JerÂ´ ome Revaud. Ground-Ë ing image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71â91. Springer, 2024. 3, 5

[23] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In CVPR, pages 20775â20785, 2024. 1, 3

[24] Ke Li, Tim Rolff, Susanne Schmidt, Reinhard Bacher, Simone Frintrop, Wim Leemans, and Frank Steinicke. Immersive neural graphics primitives. arXiv preprint arXiv:2211.13494, 2022. 1

[25] Xin Li, Yulin Ren, Xin Jin, Cuiling Lan, Xingrui Wang, Wenjun Zeng, Xinchao Wang, and Zhibo Chen. Diffusion models for image restoration and enhancementâa comprehensive survey. arXiv preprint arXiv:2308.09388, 2023. 3

[26] Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. Infinitenature-zero: Learning perpetual view generation of natural scenes from single images. In European Conference on Computer Vision, pages 515â534. Springer, 2022. 1

[27] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 300â309, 2023. 3

[28] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14458â14467, 2021. 1

[29] Jian Liu, Xiaoshui Huang, Tianyu Huang, Lu Chen, Yuenan Hou, Shixiang Tang, Ziwei Liu, Wanli Ouyang, Wangmeng Zuo, Junjun Jiang, et al. A comprehensive survey on 3d content generation. arXiv preprint arXiv:2402.01166, 2024. 1

[30] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. Advances in Neural Information Processing Systems, 33:15651â15663, 2020. 1, 2

[31] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298â9309, 2023. 1

[32] Xi Liu, Chaoyi Zhou, and Siyu Huang. 3dgs-enhancer: Enhancing unbounded 3d gaussian splatting with viewconsistent 2d diffusion priors. Advances in Neural Information Processing Systems, 37:133305â133327, 2024. 3

[33] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In CVPR, pages 20654â20664, 2024. 1, 2

[34] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[35] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In CVPR, pages 5480â5490, 2022. 3

[36] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 3

[37] Shi Qiu, Binzhu Xie, Qixuan Liu, and Pheng-Ann Heng. Advancing extended reality with 3d gaussian splatting: Innovations and prospects. In 2025 IEEE International Conference on Artificial Intelligence and eXtended and Virtual Reality (AIxVR), pages 203â208. IEEE, 2025. 1

[38] Barbara Roessle, Jonathan T Barron, Ben Mildenhall, Pratul P Srinivasan, and Matthias NieÃner. Dense depth priors for neural radiance fields from sparse input views. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12892â12901, 2022. 3

[39] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 3

[40] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems, 34: 19313â19325, 2021. 1, 2

[41] Shitao Tang, Jiacheng Chen, Dilin Wang, Chengzhou Tang, Fuyang Zhang, Yuchen Fan, Vikas Chandra, Yasutaka Furukawa, and Rakesh Ranjan. Mvdiffusion++: A dense highresolution multi-view diffusion model for single or sparseview 3d object reconstruction. In European Conference on Computer Vision, pages 175â191. Springer, 2024. 3

[42] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697â 20709, 2024. 3, 5

[43] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36:8406â8441, 2023. 3

[44] Zijun Wang, Jian Wu, Runze Fan, Wei Ke, and Lili Wang. Vprf: Visual perceptual radiance fields for foveated image synthesis. IEEE Transactions on Visualization and Computer Graphics, 2024. 1

[45] Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. Novel view synthesis with diffusion models. arXiv preprint arXiv:2210.04628, 2022. 1

[46] Lei Xiao, Salah Nouri, Joel Hegland, Alberto Garcia Garcia, and Douglas Lanman. Neuralpassthrough: Learned real-time view synthesis for vr. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1â9, 2022. 1

[47] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Humphrey Shi, and Zhangyang Wang. Sinnerf: Training neural radiance fields on complex scenes from a single image. In ECCV, pages 736â753. Springer, 2022. 3

[48] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf: Pointbased neural radiance fields. In CVPR, pages 5438â5448, 2022. 1, 2

[49] Zhiwen Yan, Weng Fei Low, Yu Chen, and Gim Hee Lee. Multi-scale 3d gaussian splatting for anti-aliased rendering. In CVPR, pages 20923â20931, 2024. 1, 2

[50] Bangbang Yang, Chong Bao, Junyi Zeng, Hujun Bao, Yinda Zhang, Zhaopeng Cui, and Guofeng Zhang. Neumesh: Learning disentangled neural mesh-based implicit field for geometry and texture editing. In European Conference on Computer Vision, pages 597â614. Springer, 2022. 3

[51] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In CVPR, 2023. 3

[52] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 3

[53] Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. Advances in neural information processing systems, 35:25018â25032, 2022. 3

[54] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai. Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In ECCV, pages 335â352. Springer, 2024. 1

[55] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492, 2020. 2

[56] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492, 2020. 1, 2

[57] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586â595, 2018. 7

[58] Zheng Zhang, Wenbo Hu, Yixing Lao, Tong He, and Hengshuang Zhao. Pixel-gs: Density control with pixelaware gradient for 3d gaussian splatting. arXiv preprint arXiv:2403.15530, 2024. 1, 2

[59] Xuening Zhu, Renjiao Yi, Xin Wen, Chenyang Zhu, and Kai Xu. Relighting scenes with object insertions in neural radiance fields. IEEE Transactions on Circuits and Systems for Video Technology, 2025. 1

[60] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting, 2023. 1, 3

# Enhancing Novel View Synthesis from extremely sparse views with SfM-free 3D Gaussian Splatting Framework

Supplementary Material

## 7. Supplementary

## 7.1. Experiments on Mip-NeRF 360

We further evaluate our method on the Mip-NeRF 360 dataset. Similarly, our method outperforms all baseline methods on all metrics. As shown in Fig. 5. Similarly, COLMAP-based 3DGS shows strong artifacts in Fig. 5, such as âfoggyâ geometries and pin-distorted Gaussians in the background. DUSt3R-based incorrectly estimates the camera parameters. Also, these methods incorrectly estimate the local color. In contrast, our method corrects these errors. InstantSplat with MASt3R-based 3DGS does not learn the distant view or object edges well. In contrast, our method solves the geometric inconsistency and thus improves the reconstruction results.

Table 5. Quantitative comparison on Mip-NeRF 360 [3] dataset under sparse-view settings (4, 6, and 9 input views).
<table><tr><td>Mip-NeRF 360</td><td colspan="3">4 views</td><td colspan="3">6 views</td><td colspan="3">9 views</td></tr><tr><td></td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>COLMAP + 3DGS</td><td>11.54</td><td>0.185</td><td>0.628</td><td>12.89</td><td>0.248</td><td>0.574</td><td>14.56</td><td>0.322</td><td>0.506</td></tr><tr><td>COLMAP + FSGS</td><td>13.37</td><td>0.302</td><td>0.619</td><td>14.25</td><td>0.318</td><td>0.591</td><td>16.07</td><td>0.366</td><td>0.544</td></tr><tr><td>COLMAP + SIDGS</td><td>13.60</td><td>0.320</td><td>0.598</td><td>14.41</td><td>0.338</td><td>0.587</td><td>16.10</td><td>0.394</td><td>0.536</td></tr><tr><td>CF-3DGS</td><td>12.89</td><td>0.226</td><td>0.588</td><td>13.09</td><td>0.236</td><td>0.601</td><td>13.68</td><td>0.264</td><td>0.601</td></tr><tr><td>InstantSplat</td><td>13.88</td><td>0.263</td><td>0.543</td><td>15.28</td><td>0.290</td><td>0.498</td><td>16.95</td><td>0.368</td><td>0.422</td></tr><tr><td>DUSt3R + 3DGS</td><td>13.34</td><td>0.228</td><td>0.567</td><td>14.37</td><td>0.259</td><td>0.527</td><td>15.05</td><td>0.289</td><td>0.505</td></tr><tr><td>MASt3R + 3DGS</td><td>13.77</td><td>0.276</td><td>0.559</td><td>14.96</td><td>0.301</td><td>0.524</td><td>16.67</td><td>0.371</td><td>0.482</td></tr><tr><td>Ours</td><td>14.92</td><td>0.328</td><td>0.573</td><td>16.52</td><td>0.376</td><td>0.514</td><td>17.96</td><td>0.429</td><td>0.473</td></tr></table>

<!-- image-->  
Figure 5. Visual comparison with different methods on Mip-NeRF 360 [3] Dataset on 4 views setting.