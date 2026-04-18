# Lumos3D: A Single-Forward Framework for Low-Light 3D Scene Restoration

Hanzhou Liu, Peng Jiang, Jia Huang, and Mi Lu hanzhou1996@tamu.edu, Texas A&M University

AbstractâRestoring 3D scenes captured under low-light conditions remains a fundamental yet challenging problem. Most existing approaches depend on precomputed camera poses and scene-specific optimization, which greatly restricts their scalability to dynamic real-world environments. To overcome these limitations, we introduce Lumos3D, a generalizable pose-free framework for 3D low-light scene restoration. Trained once on a single dataset, Lumos3D performs inference in a purely feedforward manner, directly restoring illumination and structure from unposed, low-light multi-view images without any perscene training or optimization. Built upon a geometry-grounded backbone, Lumos3D reconstructs a normal-light 3D Gaussian representation that restores illumination while faithfully preserving structural details. During training, a cross-illumination distillation scheme is employed, where the teacher network is distilled on normal-light ground truth to transfer accurate geometric information, such as depth, to the student model. A dedicated Lumos loss is further introduced to promote photometric consistency within the reconstructed 3D space. Experiments on real-world datasets demonstrate that Lumos3D achieves highfidelity low-light 3D scene restoration with accurate geometry and strong generalization to unseen cases. Furthermore, the framework naturally extends to handle over-exposure correction, highlighting its versatility for diverse lighting restoration tasks.

Impact StatementâAchieving reliable 3D restoration from lowlight inputs is crucial for autonomous perception and immersive vision systems operating under challenging illumination. Conventional low-light or relighting methods either rely on precomputed camera poses or require costly scene-specific optimization, hindering their scalability and real-world deployment. This work introduces Lumos3D, a geometry-aware and optimization-free framework that unifies illumination restoration and 3D reconstruction within a single feed-forward pipeline. By coupling a geometry-grounded backbone with a distillationbased learning strategy and the proposed Lumos loss, Lumos3D recovers accurate structure and consistent illumination directly from unposed multi-view inputs. The proposed paradigm establishes a foundation for generalizable and physically consistent 3D restoration applicable to autonomous driving, robotics, and immersive visual systems in complex lighting environments.

Index Termsâlow-light enhancement, 3D scene reconstruction, Gaussian Splatting, Visual Geometry Grounded Transformer.

## I. INTRODUCTION

Over the past few years, Neural Radiance Field (NeRF) [1] and Gaussian Splatting (3DGS) [2]-based methods have made remarkable progress in 3D scene reconstruction and novelview synthesis. However, these methods typically assume clean multi-view observations without degradations such as low-light or overexposure, which make them struggle in realworld environments with challenging illumination [3].

To alleviate these issues, recent state-of-the-art (SOTA) methods [4], [5] incorporate lighting priors or per-view adjustments to compensate for illumination inconsistencies across viewpoints. However, constrained by their underlying NeRF and 3DGS frameworks, these approaches still depend heavily on precomputed camera poses and per-scene optimization, making them difficult to generalize to unknown environments. As a result, there remains a strong need for a generalizable pose-free model capable of performing low-light 3D scene restoration without any additional per-scene optimization.

To this end, inspired by the recent success of Visual Geometry Grounded Transformer (VGGT) [6], we propose Lumos3D, a single-forward framework that restores illumination and geometry from unposed multi-view low-light inputs. Lumos3D is trained once with paired normal-light and synthetically darkened multi-view images, and performs illumination restoration and geometry reconstruction in a single forward pass on real-world scenarios, without any precomputed camera poses, pre-processing, or post-optimization. By unifying feedforward 3D reconstruction and geometry-consistent low-light enhancement within one framework, Lumos3D establishes a new paradigm for generalizable 3D scene restoration under challenging illumination conditions.

Built upon the VGGT backbone, Lumos3D first estimates geometric cues such as depth and camera poses from lowlight inputs. While it follows the general distillation framework used in previous VGGT extensions [7], [8], our design departs from these works in a key aspect, the frozen teacher network provides supervision from the normal-light counterparts, whereas the trainable student model learns from the low-light context images. This cross-illumination distillation strategy offers cleaner and more reliable geometric guidance through training on paired normal-light and low-light images.

Then, Lumos3D reconstructs a 3D Gaussian scene from the estimated geometry, following the explicit representation pipeline of AnySplat [7]. A dual-head decoder predicts perpixel depth and confidence maps, which are back-projected into 3D space using the estimated camera poses to obtain Gaussian centers. Meanwhile, a Gaussian head regresses opacity, orientation, scale, and spherical harmonic color coefficients, producing an explicit and differentiable 3D Gaussian representation that can be rendered through Gaussian rasterization [2]. This explicit formulation enables efficient end-to-end optimization while preserving geometric fidelity.

Finally, to further improve the quality of photorealistic relighting, we introduce a set of Lumos losses, including a content loss, an image-level $\ell _ { 1 }$ loss, and a voxel-level statistical loss, to enhance photometric consistency and preserve geometric fidelity in the reconstructed 3D scene. Together with the proposed cross-illumination distillation and the baseline explicit representation pipeline, these objectives enable the model to achieve geometry-aware illumination restoration that remains robust across varying low-light real-world scenes.

In summary, our main contributions are as follows:

â¢ We propose Lumos3D, a single-forward framework for low-light 3D scene restoration, without any precomputed camera pose or per-scene optimization required.

â¢ We introduce a cross-illumination distillation scheme that transfers geometric knowledge from normal-light supervision to low-light learning, enabling more reliable depth reasoning under challenging illumination.

â¢ We propose Lumos losses that jointly enforce photometric and geometric consistency for robust and coherent lowlight 3D scene restoration.

Collectively, these contributions establish a new paradigm for low-light 3D scene restoration, enabling pose-free reconstruction without any per-scene training in a single forward pass.

## II. BACKGROUND

## A. Low-Light Image Enhancement

Low-light image enhancement (LLIE) aims to improve the visibility, contrast, and color fidelity of images captured under insufficient illumination. Traditional methods, such as histogram equalization, gamma correction, and Retinex-based algorithms [9], [10], [11], [12], [13], design mathematical formulas or filtering methods to adjust the gray value of the image [14]. These approaches often rely on hand-crafted priors and parameter tuning, resulting in limited generalization, particularly in real-world conditions. With the rise of deep learning, CNN-based methods [15], [13], [16], [17], [18], [19], [20] and Transformer-based models [21], [22], [23] have achieved remarkable success by learning data-driven illumination priors. Recently, diffusion-based networks [24], [25], [26], [27] have been explored for illumination degradation image restoration by restoring the dark images in a generative manner. While these models produce high-fidelity results, the diffusion processes introduce high computational cost and potential misalignment between restored and original structures. Overall, the aforementioned LLIE methods focus primarily on 2D photometric restoration, lacking geometric reasoning and cross-view consistency, issues that Lumos3D addresses through a geometry-grounded 3D illumination framework.

## B. 3D Low-light Enhancement

In recent years, Neural Radiance Fields (NeRFs) [1] have been adapted to novel view synthesis (NVS) under challenging low-light conditions [28], [29], [30], [4]. For example, Aleth-NeRF [4] introduces a concealing field that models illumination attenuation between objects and the viewer, enabling NeRF to reconstruct and render scenes under lowlight and over-exposed conditions in a unified framework. In parallel, another line of research explores low-light enhancement within Gaussian Splatting (3DGS) frameworks [2]. The recent Luminance-GS [5] extends 3DGS by adopting per-view color-matrix mapping and view-adaptive curve adjustment, achieving illumination-consistent novel view synthesis under low-light and over-exposed conditions without modifying the explicit 3D representation. However, these 3D reconstruction-based methods typically rely on known camera poses, per-scene optimization, and scene-specific fine-tuning, which greatly limit their generalization to unseen scenes. In contrast, our proposed Lumos3D is trained in a single stage on one dataset and can be directly applied to unknown multiview low-light scenes, achieving effective and generalizable 3D reconstruction under real-world illumination conditions.

## C. Single-Forward 3D Reconstruction

Early neural-network models [31], [32], [33], [34] have demonstrated that end-to-end learning-based 3D reconstruction can be achieved. Subsequently, large-scale models such as DUSt3R [35] and its extension MASt3R [36] replace multistage SfM pipelines with single-pass predictions of geometry and poses. While effective, these models require additional post-processing and often struggle with alignment in complex multi-view settings. Later, Spann3R [37], CUT3R [38], and MUSt3R [39] further reduce reliance on classical optimization by incorporating memory or parallelized attention, albeit at relatively high computational cost. More recently, VGGT [6] introduces a novel Transformer architecture for joint multiview inference of depth, pose, and point-maps. Building upon this, AnySplat [7] extends VGGT into an efficient feed-forward 3DGS framework, bringing real-time novel-view synthesis within reach for unconstrained capture settings. Our work, Lumos3D, is inspired by these advancements and further extends them toward restoration of low-light 3D scenes.

## III. METHODOLOGY

## A. Problem Formulation

We aim to restore the geometry and illumination of a 3D scene captured under low-light conditions, without relying on precomputed camera poses or per-scene optimization. Given a set of N low-light images $\mathit { \bar { Z } _ { L } } \ = \ \{ I _ { \mathrm { L } } ^ { \dagger 1 } , I _ { \mathrm { L } } ^ { ( 2 ) } , \dots , I _ { \mathrm { L } } ^ { ( N ) } \}$ observed from different viewpoints, our goal is to reconstruct a normal-light 3D representation $\mathcal { G } _ { \mathrm { R } }$ that is both photometrically enhanced and geometrically consistent across views.

Input and Output. Each low-light input image $I _ { \mathrm { L } } ^ { ( i ) } \in \mathbb { \Gamma }$ $\mathbb { R } ^ { \tilde { H } \times W \times 3 }$ corresponds to an unknown camera pose and an uncalibrated illumination condition. The network predicts, in a single forward pass, the underlying 3D scene structure represented as a Gaussian representation $\mathcal { G } _ { \mathrm { R } }$ . Formally, the process can be expressed as:

$$
\mathcal { G } _ { \mathrm { R } } = \Phi _ { \theta } ( \mathbb { Z } _ { \mathrm { L } } ) ,\tag{1}
$$

where $\Phi _ { \theta }$ denotes the proposed network parameterized by Î¸.

Pose-Free Inference. Unlike traditional NeRF- or 3DGSbased methods that require known camera poses, Lumos3D implicitly estimates relative view geometry through a selflearned geometric attention mechanism. At inference time, the model directly performs 3D reconstruction and restoration in a single forward pass, without any per-scene fine-tuning.

<!-- image-->  
Fig. 1. Architecture overview. Given multi-view low-light context inputs, Lumos3D instantly predicts 3D Gaussian representations with restored light conditions and renders corresponding RGB image and depth maps, without scene-specific training OR optimization. The two key components are the crossillumination distillation loss $\lambda _ { d i s t i l l }$ and the proposed $\lambda _ { l u m o s } ,$ as discussed in III-C and III-D respectively. For simplicity, we omit the baseline loss Î»rec [7].

## B. Overall Architecture

The proposed Lumos3D framework adopts a single-forward encoderâdecoder architecture for low-light 3D scene restoration, as illustrated in Fig. 1. Given a set of unposed multiview low-light images $\overline { { \mathcal { T } } } _ { \mathrm { L } } = \{ I _ { \mathrm { L } } ^ { ( 1 ) } , \dots , I _ { \mathrm { L } } ^ { ( S ) } \}$ , Lumos3D simultaneously reconstructs the scene geometry and restores its illumination without any precomputed camera poses.

Architecture Overview. Built upon the Visual Geometry Grounded Transformer (VGGT) backbone, Lumos3D first extracts geometry-aware features from low-light inputs and estimates per-view camera parameters, depth maps and point maps. These features are then transformed into a set of pixelwise Gaussian primitives of the scene. After that, Lumos3D applies differentiable voxelization and produces voxel-wise Gaussian primitives [7], which are used to render restored RGB images and depth maps from arbitrary viewpoints.

Training Objective. During training, paired normal-light and synthetically darkened image sets $\{ \mathcal { T } _ { \mathrm { N } } , \mathcal { T } _ { \mathrm { L } } \}$ are provided. A frozen teacher network, supervised under normal-light conditions, serves as geometric guidance for the student model operating on low-light context inputs. The entire system is optimized using a unified objective that combines reconstruction, distillation, and the proposed Lumos Loss:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c } } + \omega _ { \mathrm { d i s t i l l } } \mathcal { L } _ { \mathrm { d i s t i l l } } + \omega _ { \mathrm { l u m o s } } \mathcal { L } _ { \mathrm { l u m o s } } .\tag{2}
$$

Here, $\mathcal { L } _ { \mathrm { r e c } }$ and ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ enforces photometric reconstruction and transfers geometric knowledge from the teacher model, serving as the baseline loss function; and ${ \mathcal { L } } _ { \mathrm { l u m o s } }$ integrates multi-level photometric and geometric consistency terms.

Inference. Once trained, Lumos3D performs real-time 3D scene reconstruction and relighting in a single feed-forward pass, directly predicting illumination-corrected color and depth outputs from unposed low-light multi-view inputs.

## C. Cross-Illumination Distillation

To enhance geometric reasoning under low-light, we employ a cross-illumination distillation strategy that transfers geometric knowledge from normal-light supervision to lowlight learning. The key idea is to leverage a teacherâstudent framework, where the teacher network is trained or frozen under normal illumination and provides geometry-rich guidance to the student network that learns from low-light inputs.

Specifically, given a pair of normal-light and synthetically darkened multi-view images $\{ \mathcal { T } _ { \mathrm { N } } , \mathcal { T } _ { \mathrm { L } } \}$ , the teacher network $\tau$ processes $\mathcal { T } _ { \mathrm { N } }$ to generate illumination-independent geometric cues, such as depth maps $\mathcal { D } _ { \mathrm { T } }$ and 3D point maps $\mathcal { P } _ { \mathrm { T } }$ . The student network S takes $\mathcal { T } _ { \mathrm { L } }$ as input and predicts its own geometric representations $\mathcal { D } _ { \mathrm { { S } } }$ and $\mathcal { P } _ { \mathrm { S } }$ . A distillation loss is then formulated to encourage the student to mimic the teacherâs geometry predictions:

$$
\mathcal { L } _ { \mathrm { d i s t i l } } = \frac { 1 } { B S H W } \sum _ { b , s , x , y } \left. \mathcal { P } _ { \mathrm { S } } ^ { ( b , s , x , y ) } - \mathcal { P } _ { \mathrm { T } } ^ { ( b , s , x , y ) } \right. _ { 1 } .\tag{3}
$$

Here, each $\mathcal { P } ^ { ( b , s , x , y ) } \in \mathbb { R } ^ { 3 }$ represents a 3D point coordinate vector $( X , Y , Z )$ corresponding to pixel location $( x , y )$ in the

s-th view of the b-th sample.

This supervision allows the student to learn illuminationinvariant geometric priors without explicit pose annotations, effectively bridging the gap between photometric degradation and geometric understanding. During training, the teacher network is kept frozen to ensure stable and reliable geometric guidance, while only the student model parameters are updated. Through this distillation mechanism, Lumos3D achieves robust geometry reconstruction and improved depth consistency across varying illumination conditions.

## D. Lumos Loss

To jointly optimize photometric fidelity and 3D structural consistency under low-light conditions, we design the $L u \mathrm { . }$ mos loss ${ \mathcal { L } } _ { \mathrm { L u m o s } }$ , which provides geometry-aware supervision from three complementary perspectives: content-level semantic alignment, image-level photometric reconstruction, and voxel-level 3D consistency. It is defined as:

$$
{ \mathcal { L } } _ { \mathrm { L u m o s } } = \lambda _ { c } { \mathcal { L } } _ { \mathrm { c o n t e n t } } + \lambda _ { i } { \mathcal { L } } _ { \mathrm { i m a g e } } + \lambda _ { v } { \mathcal { L } } _ { \mathrm { v o x e l } } ,\tag{4}
$$

where $\lambda _ { c } , \lambda _ { i } .$ , and $\lambda _ { v }$ are the weighting coefficients for the content-, image-, and voxel-level losses, respectively, with default settings of 0.1, 1.0, and 0.01. This formulation balances photometric reconstruction and geometric regularization, encouraging illumination-invariant and structure-preserving 3D restoration under diverse lighting conditions.

1) Content-Level Feature Loss: The content loss encourages high-level semantic consistency between the restored image and the ground-truth normal-light reference. Specifically, we extract intermediate activations from a pretrained VGG [40] network and compute their $\ell _ { 1 }$ difference:

$$
\mathcal { L } _ { \mathrm { c o n t e n t } } = \sum _ { i \in \{ L - 1 , L \} } w _ { i } ^ { ( c ) } \frac { 1 } { N _ { i } } \sum _ { p } \left\| F _ { i , p } ^ { ( r ) } - F _ { i , p } ^ { ( c ) } \right\| _ { 1 } ,\tag{5}
$$

where $F _ { i , p } ^ { ( r ) }$ and $F _ { i , p } ^ { ( c ) }$ are the feature activations at spatial location p in the i-th VGG layer of the restored and content images, respectively. $w _ { i } ^ { ( c ) }$ are normalized layer weights satisfying $\begin{array} { r } { \sum _ { i } w _ { i } ^ { ( c ) } = 1 } \end{array}$ . This term preserves semantic integrity and illumination-insensitive content under low-light degradation.

2) Image-Level Restoration Loss: To guarantee pixel-wise accuracy and color fidelity, we adopt an $\ell _ { 1 }$ restoration loss between the restored image $I ^ { ( r ) }$ and the normal-light groundtruth image $I ^ { ( c ) }$ :

$$
\mathcal { L } _ { \mathrm { i m a g e } } = \frac { 1 } { B S H W C } \sum _ { b = 1 } ^ { B } \sum _ { s = 1 } ^ { S } \sum _ { x , y , c } \left| I _ { b , s , x , y , c } ^ { ( r ) } - I _ { b , s , x , y , c } ^ { ( c ) } \right| .\tag{6}
$$

This term penalizes direct intensity deviations, providing stable low-level supervision for illumination recovery.

3) Voxel-Level 3D Consistency Loss: To enforce geometric coherence across multi-view observations, we construct a voxel-level consistency loss that aggregates 2D feature responses into 3D volumetric cells. For each feature scale i, the image features $\mathbf { f } _ { p }$ are projected onto their 3D coordinates p and grouped into voxel grids $\mathcal { P } _ { v }$ . Each voxel feature is obtained by averaging features of all pixels that fall into the same voxel:

$$
\mathbf { f } _ { v } ^ { ( r ) } = \frac { 1 } { | \mathcal { P } _ { v } | } \sum _ { p \in \mathcal { P } _ { v } } \mathbf { f } _ { p } .\tag{7}
$$

Given the voxelized features from the restored and distilled representations of the frozen teacher model, we align their meanâvariance statistics across scales:

$$
\mathcal { L } _ { \mathrm { v o x e l } } = \sum _ { i = 1 } ^ { 5 } w _ { i } ^ { ( v ) } \left( \left. \mu _ { i } ^ { ( r ) } - \mu _ { i } ^ { ( d ) } \right. _ { 1 } + \left. \sigma _ { i } ^ { ( r ) } - \sigma _ { i } ^ { ( d ) } \right. _ { 1 } \right) ,\tag{8}
$$

where $\mu _ { i } ^ { ( r ) }$ and $\boldsymbol { \sigma } _ { i } ^ { ( r ) }$ denote the mean and standard deviation of voxelized features for the restored branch at scale i, and $\mu _ { i } ^ { ( d ) } , \sigma _ { i } ^ { ( d ) }$ correspond to those from the teacher (distilled) branch. The scale weights $w _ { i } ^ { ( v ) }$ are normalized such that $\begin{array} { r } { \sum _ { i } w _ { i } ^ { ( v ) } = 1 } \end{array}$ . This term enforces geometry-aware feature alignment, ensuring that volumetric feature statistics remain consistent under varying illumination conditions.

In short, the Lumos loss thus unifies 2D perceptual alignment and 3D volumetric consistency into a single differentiable framework. It guides the model to restore low-light scenes that are not only visually faithful to the ground truth but also geometrically coherent across multiple views, achieving illumination-consistent 3D reconstruction.

## IV. EXPERIMENT

Datasets. DL3DV [41] is a large-scale scene dataset that contains around 11K scenes across 65 types of point-ofinterest locations. In ablation studies, our model is trained only on the full DL3DV dataset. To simulate low-light inputs, we randomly scale the exposure by a factor between 0.05 and 0.1, and apply a gamma correction of 1.3â1.4 in the linear RGB domain. For evaluation, we use the bike, buu, chair and sofa scenes in the LOM dataset [4].

Implementation Details. We train our network with a dynamic batch size of 22, corresponding to the maximum number of views per GPU. The entire training process consists of 30K iterations. The initial learning rate is set to $2 \times 1 0 ^ { - 4 }$ and follows a cosine annealing schedule with a warm-up phase of 1K iterations. Training converges within approximately 60 hours using eight NVIDIA GH200 GPUs on two nodes.

Evaluation Metrics. To assess reconstruciton and restoration quality, we report PSNR, SSIM, and LPIPS [41] between the predicted rendered images and normal-light ground-truth.

## A. Ablation Study

We conduct ablation studies to assess the contribution of each component in our network design. All experiments are performed under identical training strategies and hardware configurations, with only the component under investigation being modified. Unlike the final model training, all ablation variants are trained on the first 6K scenes of the DL3DV dataset with a dynamic batch size of 18 for up to 20,000 steps using four NVIDIA GH200 GPUs, ensuring computational efficiency. Throughout this section, the terms normal-light and ground truth are used interchangeably, whereas low-light and context images denote the same input modality.

(a) Normal-light ground-truth  
(b) Low-light context images  
<!-- image-->  
(c) Distillation on low-light

<!-- image-->  
(d) Distillation on normal-light  
Fig. 2. Qualitative comparison of different distillation schemes. Each visualization corresponds to the same scene, with depth on the left and the corresponding RGB image on the right. In the depth maps, blue denotes distant regions and red denotes nearby ones. Distillation on low-light images suffers from illumination ambiguity, whereas distillation on normal-light images yields more accurate and geometrically cleaner depth and relighting results.

TABLE I  
ABLATION ON DISTILLATION TARGETS OF THE TEACHER MODEL AND LUMOS LOSS VARIANTS. QUANTITATIVE RESULTS ARE AVERAGED ACROSS THE LOM DATASET USING MODELS TRAINED ON DL3DV.
<table><tr><td colspan="2">Distillation</td><td colspan="3">Lumos losses</td><td colspan="3">Metrics (Average)</td></tr><tr><td>Low1</td><td>GT2</td><td>Content</td><td>Image</td><td>Voxel</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>â</td><td></td><td></td><td></td><td></td><td>17.93</td><td>0.758</td><td>0.405</td></tr><tr><td></td><td>â</td><td></td><td></td><td></td><td>17.47</td><td>0.758</td><td>0.402</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>17.47</td><td>0.758</td><td>0.402</td></tr><tr><td></td><td>v</td><td>â</td><td></td><td></td><td>17.59</td><td>0.765</td><td>0.403</td></tr><tr><td></td><td>â</td><td>â</td><td>â</td><td></td><td>19.41</td><td>0.782</td><td>0.402</td></tr><tr><td></td><td>â</td><td>â</td><td>â</td><td>â</td><td>19.76</td><td>0.784</td><td>0.396</td></tr></table>

Note. 1 Low = low-light context images; 2 GT = normal-light ground-truth.

1) Distillation: We first compare the effect of two different distillation strategies, one using teacher predictions on lowlight context images and the other using teacher predictions on normal-light ground-truth images. As shown in Table I, although the ground-truthâbased distillation achieves slightly lower PSNR, it consistently outperforms the low-lightâbased one in SSIM and LPIPS, indicating improved structural fidelity and perceptual quality. To further assess whether scene information, particularly depth mappings, can be effectively transferred from the teacher model to the student model, we provide qualitative comparisons shown in Figure 2.

Depth visualization for both the normal-light and low-light images are generated using Depth Anything V2 [42]. The model distilled on ground-truth images produces more accurate depth estimations and noticeably cleaner reconstructions, with fewer blurry regions and more geometrically consistent edges. For instance, Figure 2 shows that the supporting rod beneath the bicycle saddle exhibits no ghosting, and the wheel contours appear smoother. Therefore, we adopt the groundtruthâbased distillation as the default configuration.

2) Losses: Finally, we evaluate the effect of different Lumos loss variants on the modelâs reconstruction quality. As shown in Table I, using only the baseline reconstruction loss yields limited improvement, achieving 17.47 dB PSNR and 0.758 SSIM. Introducing the content loss provides a small but consistent gain, indicating better structural alignment with the ground truth. Adding the image-level loss brings a clear enhancement, raising PSNR from 17.59 to 19.41 and SSIM from 0.765 to 0.782, which demonstrates its crucial role in improving global contrast and perceptual quality through pixel-level supervision. When the voxel-level loss is further introduced, the model attains the best overall performance, reaching 19.76 dB PSNR, 0.784 SSIM, and 0.396 LPIPS. These results confirm that combining multi-level supervision enables the most faithful and perceptually pleasing reconstructions under challenging low-light conditions.

## B. Comparison with State-of-the-Art Methods

To the best of our knowledge, no existing approach provides a single-forward solution for low-light 3D scene restoration. We therefore compare our Lumos3D with representative perscene optimization-based methods built upon NeRF and 3D Gaussian Splatting (3DGS), including Aleth-NeRF [4] and Luminance-GS [5]. As shown in Table II, although these optimization-based baselines are fine-tuned individually for each scene, Lumos3D achieves highly competitive performance without any scene-specific adaptation. Furthermore, we demonstrate that Lumos3D can be readily extended beyond low-light enhancement to address other illumination degradations, such as over-exposure restoration.

1) Low-light Enhancement: Specifically, in low-light conditions, Lumos3D attains the highest PSNR and SSIM in most scenes (Buu, Sofa) while maintaining favorable LPIPS scores across all four test cases. Notably, on the Sofa scene, Lumos3D achieves a PSNR of 22.21 dB and an LPIPS of 0.346, significantly surpassing the optimization-based Aleth-NeRF and Luminance-GS. These results demonstrate that even though Lumos3D is trained solely on synthetic data, it generalizes effectively to real-world low-light scenarios, producing geometrically faithful reconstructions.

TABLE II  
QUANTITATIVE COMPARISON OF DIFFERENT MODELS ON THE LOM DATASET. BEST RESULTS ARE HIGHLIGHTED IN BOLD.
<table><tr><td rowspan="2">Models</td><td colspan="3">Bike</td><td colspan="3">Buu</td><td colspan="3">Chair</td><td colspan="3">Sofa</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td colspan="10">Low-light</td><td colspan="3"></td></tr><tr><td>Aleth_NeRF [4]</td><td>16.50</td><td>0.661</td><td>0.481</td><td>16.52</td><td>0.707</td><td>0.418</td><td>16.54</td><td>0.768</td><td>0.536</td><td>16.53</td><td>0.805</td><td>0.408</td></tr><tr><td>Luminance-GS [5]</td><td>16.39</td><td>0.627</td><td>0.520</td><td>15.40</td><td>0.725</td><td>0.436</td><td>18.58</td><td>0.690</td><td>0.634</td><td>18.98</td><td>0.756</td><td>0.472</td></tr><tr><td>Lumos3D (Ours)</td><td>14.07</td><td>0.605</td><td>0.432</td><td>19.16</td><td>0.755</td><td>0.420</td><td>17.82</td><td>0.781</td><td>0.565</td><td>22.21</td><td>0.848</td><td>0.346</td></tr><tr><td colspan="9">Over-exposure</td><td colspan="3"></td></tr><tr><td>Aleth_NeRF [4]</td><td>19.02</td><td>0.705</td><td>0.423</td><td>15.16</td><td>0.709</td><td>0.682</td><td>19.02</td><td>0.789</td><td>0.545</td><td>18.14</td><td>0.822</td><td>0.459</td></tr><tr><td>Luminance-GS [5]</td><td>19.72</td><td>0.646</td><td>0.65</td><td>15.66</td><td>0.729</td><td>0.511</td><td>20.6</td><td>0.670</td><td>0.392</td><td>19.59</td><td>0.751</td><td>0.410</td></tr><tr><td>Lumos3D (Ours)</td><td>20.92</td><td>0.733</td><td>0.289</td><td>15.00</td><td>0.711</td><td>0.493</td><td>21.99</td><td>0.790</td><td>0.386</td><td>22.37</td><td>0.847</td><td>0.339</td></tr></table>

<!-- image-->  
Fig. 3. Qualitative comparison of different 3D low-light and over-exposure restoration schemes on the chair and sofa scenes in the LOM dataset.

2) Extension to Over-exposure Restoration: We further investigate Lumos3D in over-exposure cases. Following the lowlight synthesis strategy, we generate over-exposed inputs by simulating high-illumination conditions. As shown in Table II, Lumos3D consistently achieves the best overall performance, yielding the highest PSNR and SSIM on most scenes while maintaining the lowest LPIPS across all settings. Despite being trained exclusively on synthetic over-exposure data, Lumos3D generalizes effectively to real-world high-exposure scenes without any fine-tuning.

## V. CONCLUSION

In this work, we presented Lumos3D, a single-forward framework for low-light 3D scene restoration. Unlike prior approaches that rely on precomputed camera poses or perscene optimization, Lumos3D directly reconstructs and restore 3D scenes from unposed multi-view low-light inputs. Through the proposed cross-illumination distillation scheme and geometry-aware Lumos loss, our method jointly enforces photometric fidelity and spatial consistency. Extensive experiments on both synthetic and real-world datasets demonstrate that Lumos3D achieves visually consistent and geometrically accurate results, even when trained solely on synthetic data. Beyond low-light scenarios, the framework also generalizes to other challenging illumination conditions, such as overexposure. We believe Lumos3D establishes a new foundation for scalable, optimization-free 3D scene restoration, paving the way toward unified and real-time relighting systems.

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[3] W. Kwon, J. Sung, M. Jeon, C. Eom, and J. Oh, âR3evision: A survey on robust rendering, restoration, and enhancement for 3d low-level vision,â arXiv preprint arXiv:2506.16262, 2025.

[4] Z. Cui, L. Gu, X. Sun, X. Ma, Y. Qiao, and T. Harada, âAlethnerf: Illumination adaptive nerf with concealing field assumption,â in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 2, 2024, pp. 1435â1444.

[5] Z. Cui, X. Chu, and T. Harada, âLuminance-gs: Adapting 3d gaussian splatting to challenging lighting conditions with view-adaptive curve adjustment,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 472â26 482.

[6] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, âVggt: Visual geometry grounded transformer,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294â 5306.

[7] L. Jiang, Y. Mao, L. Xu, T. Lu, K. Ren, Y. Jin, X. Xu, M. Yu, J. Pang, F. Zhao et al., âAnysplat: Feed-forward 3d gaussian splatting from unconstrained views,â arXiv preprint arXiv:2505.23716, 2025.

[8] H. Liu, J. Huang, M. Lu, S. Saripalli, and P. Jiang, âStylos: Multi-view 3d stylization with single-forward gaussian splatting,â arXiv preprint arXiv:2509.26455, 2025.

[9] M. Abdullah-Al-Wadud, M. H. Kabir, M. A. A. Dewan, and O. Chae, âA dynamic histogram equalization for image contrast enhancement,â IEEE transactions on consumer electronics, vol. 53, no. 2, pp. 593â600, 2007.

[10] S.-C. Huang, F.-C. Cheng, and Y.-S. Chiu, âEfficient contrast enhancement using adaptive gamma correction with weighting distribution,â IEEE transactions on image processing, vol. 22, no. 3, pp. 1032â1041, 2012.

[11] E. H. Land, âThe retinex theory of color vision,â Scientific american, vol. 237, no. 6, pp. 108â129, 1977.

[12] X. Fu, D. Zeng, Y. Huang, X.-P. Zhang, and X. Ding, âA weighted variational model for simultaneous reflectance and illumination estimation,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 2782â2790.

[13] M. Li, J. Liu, W. Yang, X. Sun, and Z. Guo, âStructure-revealing lowlight image enhancement via robust retinex model,â IEEE transactions on image processing, vol. 27, no. 6, pp. 2828â2841, 2018.

[14] J. Guo, J. Ma, A. F. Garc Â´ Â´Ä±a-Fernandez, Y. Zhang, and H. Liang, âA Â´ survey on image enhancement for low-light images,â Heliyon, vol. 9, no. 4, 2023.

[15] K. G. Lore, A. Akintayo, and S. Sarkar, âLlnet: A deep autoencoder approach to natural low-light image enhancement,â Pattern Recognition, vol. 61, pp. 650â662, 2017.

[16] R. Wang, Q. Zhang, C.-W. Fu, X. Shen, W.-S. Zheng, and J. Jia, âUnderexposed photo enhancement using deep illumination estimation,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 6849â6857.

[17] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, âLearning enriched features for real image restoration and enhancement,â in European conference on computer vision. Springer, 2020, pp. 492â511.

[18] Y. Jiang, X. Gong, D. Liu, Y. Cheng, C. Fang, X. Shen, J. Yang, P. Zhou, and Z. Wang, âEnlightengan: Deep light enhancement without paired supervision,â IEEE transactions on image processing, vol. 30, pp. 2340â 2349, 2021.

[19] S. Zhou, C. Li, and C. Change Loy, âLednet: Joint low-light enhancement and deblurring in the dark,â in European conference on computer vision. Springer, 2022, pp. 573â589.

[20] Y. Fu, Y. Hong, L. Chen, and S. You, âLe-gan: Unsupervised lowlight image enhancement network using attention module and identity invariant loss,â Knowledge-Based Systems, vol. 240, p. 108010, 2022.

[21] X. Xu, R. Wang, C.-W. Fu, and J. Jia, âSnr-aware low-light image enhancement,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 17 714â17 724.

[22] Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, and Y. Zhang, âRetinexformer: One-stage retinex-based transformer for low-light image enhancement,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 12 504â12 513.

[23] S. Zhang, N. Meng, and E. Y. Lam, âLrt: An efficient low-light restoration transformer for dark light field images,â IEEE Transactions on Image Processing, vol. 32, pp. 4314â4326, 2023.

[24] C. He, C. Fang, Y. Zhang, T. Ye, K. Li, L. Tang, Z. Guo, X. Li, and S. Farsiu, âReti-diff: Illumination degradation image restoration with retinex-based latent diffusion model,â arXiv preprint arXiv:2311.11638, 2023.

[25] H. Jiang, A. Luo, H. Fan, S. Han, and S. Liu, âLow-light image enhancement with wavelet-based diffusion models,â ACM Transactions on Graphics (TOG), vol. 42, no. 6, pp. 1â14, 2023.

[26] X. Yi, H. Xu, H. Zhang, L. Tang, and J. Ma, âDiff-retinex++: Retinexdriven reinforced diffusion model for low-light image enhancement,â IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[27] T. Wang, K. Zhang, Y. Zhang, W. Luo, B. Stenger, T. Lu, T.-K. Kim, and W. Liu, âLldiffusion: Learning degradation representations in diffusion models for low-light image enhancement,â Pattern Recognition, vol. 166, p. 111628, 2025.

[28] H. Wang, X. Xu, K. Xu, and R. W. Lau, âLighting up nerf via unsupervised decomposition and enhancement,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 12 632â12 641.

[29] Z. Qu, K. Xu, G. P. Hancke, and R. W. Lau, âLush-nerf: Lighting up and sharpening nerfs for low-light scenes,â arXiv preprint arXiv:2411.06757, 2024.

[30] Y. Wang, C. Wang, B. Gong, and T. Xue, âBilateral guided radiance field processing,â ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1â13, 2024.

[31] B. Ummenhofer, H. Zhou, J. Uhrig, N. Mayer, E. Ilg, A. Dosovitskiy, and T. Brox, âDemon: Depth and motion network for learning monocular stereo,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5038â5047.

[32] H. Zhou, B. Ummenhofer, and T. Brox, âDeeptam: Deep tracking and mapping,â in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 822â838.

[33] Z. Teed and J. Deng, âDeepv2d: Video to depth with differentiable structure from motion,â in International Conference on Learning Representations, 2020. [Online]. Available: https://openreview.net/ forum?id=HJeO7RNKPr

[34] D. Wang, X. Cui, X. Chen, Z. Zou, T. Shi, S. Salcudean, Z. J. Wang, and R. Ward, âMulti-view 3d reconstruction with transformers,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5722â5731.

[35] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, âDust3r: Geometric 3d vision made easy,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 697â20 709.

[36] V. Leroy, Y. Cabon, and J. Revaud, âGrounding image matching in 3d with mast3r,â in European Conference on Computer Vision. Springer, 2024, pp. 71â91.

[37] H. Wang and L. Agapito, â3d reconstruction with spatial memory,â arXiv preprint arXiv:2408.16061, 2024.

[38] Q. Wang, Y. Zhang, A. Holynski, A. A. Efros, and A. Kanazawa, âContinuous 3d perception model with persistent state,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 10 510â10 522.

[39] Y. Cabon, L. Stoffl, L. Antsfeld, G. Csurka, B. Chidlovskii, J. Revaud, and V. Leroy, âMust3r: Multi-view network for stereo 3d reconstruction,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1050â1060.

[40] K. Simonyan and A. Zisserman, âVery deep convolutional networks for large-scale image recognition,â arXiv preprint arXiv:1409.1556, 2014.

[41] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.

[42] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, âDepth anything v2,â arXiv:2406.09414, 2024.