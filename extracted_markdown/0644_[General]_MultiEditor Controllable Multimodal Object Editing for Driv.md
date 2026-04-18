# MultiEditor: Controllable Multimodal Object Editing for Driving Scenarios Using???????? (????????,???????? , ???????????? ) 3D Gaussian Splatting Priors

Shouyi Lu1 Zihan Lin2\* Chao Lu2 Huanran Wang2 Guirong Zhuo1â  Lianqing Zheng1encoder NReconstruction Branch (????????,???????? , ???????????? )(????????,???? , ???????? )

1School of Automotive Studies, Tongji UniversityP 2Mach Drive

## Abstract

Autonomous driving systems rely heavily on multimodal perception data to understand complex environments. However, the long-tailed distribution of real-world data hinders generalization, especially for rare but safety-critical vehicle categories. To address this challenge, we propose MultiEditor, a dual-branch latent diffusion framework designed to edit images and LiDAR point clouds in driving scenarios jointly. At the core of our approach is introducing 3D Gaussian Splatting (3DGS) as a structural and appearance prior for target objects. Leveraging this prior, we design a multilevel appearance control mechanismâcomprising pixel-level pasting, semantic-level guidance, and multi-branch refinementâto achieve high-fidelity reconstruction across modalities. We further propose a depth-guided deformable crossmodality condition module that adaptively enables mutual guidance between modalities using 3DGS-rendered depth, significantly enhancing cross-modality consistency. Extensive experiments demonstrate that MultiEditor achieves superior performance in visual and geometric fidelity, editing controllability, and cross-modality consistency. Furthermore, generating rare-category vehicle data with MultiEditor substantially enhances the detection accuracy of perception models on underrepresented classes.

## Introduction

Autonomous driving systems rely on diverse sensors to perceive their surroundings. Among them, LiDAR and cameras are the primary modalities, capturing point clouds and RGB images that offer complementary geometric and semantic information crucial for scene understanding. Despite significant advances in multimodal perception (Redmon et al. 2016; Lang et al. 2019; Zhao et al. 2024; Chae, Kim, and Yoon 2024), the performance of existing models remains heavily dependent on large-scale, balanced datasets. In practice, real-world driving data often exhibit a pronounced long-tailed distribution: common vehicle categories are vastly overrepresented, while rare yet safetycritical classesâsuch as road rollers and excavatorsâare severely underrepresented. This data imbalance hinders generalization and undermines the robustness of perception models in long-tail and edge-case scenarios.

<!-- image-->  
(a)

<!-- image-->  
(b)  
Figure 1: (a) Editing image and point cloud separately using independent single-modal models (Chen et al. 2024; Hu, Zhang, and Hu 2024). (b) Our proposed MultiEditor leverages 3DGS priors to jointly edit image and LiDAR data, enhancing geometric and appearance accuracy and ensuring cross-modality consistency.

To mitigate the challenge of limited data diversity, recent research (Buburuzan et al. 2025; Huang et al. 2025; Liang et al. 2025; Hu, Zhang, and Hu 2024; Yan et al. 2025; Wei, Li, and Liu 2025) has explored the use of Latent Diffusion Models (LDMs) (Rombach et al. 2022) to synthesize tailored driving scenarios. However, most existing approaches are restricted to single-modality editingâfocusing either on images or point cloudsâand lack a unified framework for joint multimodal editing. Naively combining separate single-modal pipelines often leads to substantial crossmodality inconsistencies in geometry and appearance (see Figure 1(a)). Furthermore, image editing methods typically rely on dataset-derived object priors, offering limited flexibility in manipulating object pose and viewpoint. Likewise, point cloud editing methods frequently depend on scenelevel masks and overlook object-level structural priors, reducing editing precision and controllability.

In this paper, we introduce MultiEditor, a novel framework for joint editing of LiDAR and image data, as shown in Figure 1(b). MultiEditor is the first framework to incorporate 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) as a unified prior for both the appearance and structure of target objects. By leveraging 3DGSâs capability to render RGB images and depth maps from arbitrary viewpoints, our framework enables precise and controllable manipulation of object position and orientation. Architecturally, MultiEditor employs a dual-branch latent diffusion framework, comprising two dedicated generative pathways for point clouds and images, and unifies them via a shared 3DGS-based object representation.

To enable high-fidelity object editing, MultiEditor integrates 3DGS-rendered information into each branch and introduces a multi-level appearance control mechanism comprising pixel-level preservation through image pasting, semantic-level guidance via embedding features, and a multi-branch joint optimization strategy. To ensure crossmodality consistency, we propose a depth-guided deformable condition module that leverages 3DGS-rendered depth priors to facilitate latent-space feature alignment and mutual guidance across modalities. Additionally, to address challenges such as depth noise, we introduce a deformable attention-based (Xia et al. 2022) multimodal interaction mechanism that adaptively establishes cross-modality correspondences without requiring precise geometric registration, thereby mitigating performance degradation caused by misalignment.

We validate MultiEditorâs effectiveness through comprehensive evaluations on public datasets. Experimental results show our method surpasses state-of-the-art single-modality approaches in visual and geometric fidelity, editing controllability, and cross-modality consistency. MultiEditor is the first framework capable of flexibly editing atypical vehicles, offering a novel solution for enhancing perception robustness in long-tail scenarios.

Our contributions are summarized as follows.

â¢ We propose MultiEditor, the first dual-branch multimodal latent diffusion framework that integrates 3DGS as a unified prior for object appearance and structure, enabling flexible, precise, and consistent joint editing of point clouds and images.

â¢ We introduce a depth-guided deformable cross-modality condition module that leverages 3DGS-rendered depth and spatial transformations to align latent-space features and enable mutual guidance across modalities.

â¢ Extensive experiments demonstrate that MultiEditor achieves superior performance in visual and geometric fidelity, editing controllability, cross-modality consistency, and effectiveness in downstream perception tasks.

## Related Work

## Generation of Driving Scenarios

The growing demand for diverse multimodal data in autonomous driving has accelerated progress in synthetic data techniques. Most existing approaches focus on singlemodality generation by leveraging generative models. In image synthesis, prior work (Gao et al. 2023; Wang et al. 2024;

Zhao et al. 2025; Wen et al. 2024; Li et al. 2025a,b) has achieved controllable generation by conditioning pre-trained diffusion models on structured inputs such as road layouts and bounding boxes. For 3D point clouds, recent advances (Hu, Zhang, and Hu 2024; Ran, Guizilini, and Wang 2024; Nakashima and Kurazume 2024; Wu et al. 2024) project point clouds into range images, enabling efficient synthesis via 2D diffusion architectures. Despite notable success in modeling marginal distributions of single-modality data, existing methods largely overlook the mutual dependencies between modalities necessary to capture the complexity of driving scenes. To fill this gap, X-Drive (Xie et al. 2024) proposes interaction mechanisms between heterogeneous diffusion models, facilitating joint 2Dâ3D generation via crossmodality interaction. Despite these advances in full-scene synthesis, existing approaches remain constrained by global generation paradigms and struggle to support fine-grained, object-level editing.

## Editing of Driving Scenarios

Unlike scene generation, driving scene editing emphasizes fine-grained manipulation of specific objects within scenarios. R3D2 (Ljungbergh et al. 2025) performs asset insertion by leveraging synthetic assets and 3DGS-reconstructed scenes, employing a single-step diffusion model for sceneconsistent inpainting. PbE (Yang et al. 2023) proposes an exemplar-driven image editing method using diffusion models, enabling fine-grained semantic manipulation through self-supervised training and boundary artifact suppression. AnyDoor (Chen et al. 2024) introduces a diffusion-based framework augmented with identity and detail feature extractors, supporting flexible object synthesis at user-defined locations. SubjectDrive (Huang et al. 2025) design a video generation architecture that integrates a subject prompt adapter, a visual adapter, and augmented temporal attention, substantially improving controllability over generated subjects. Liang et al. (Liang et al. 2025) combine 3D bounding box priors with novel view synthesis to achieve objectlevel editing in autonomous driving videos. RangeLDM (Hu, Zhang, and Hu 2024) proposes a diffusion-based approach for synthesizing range-view LiDAR point clouds, which also supports mask-conditioned point cloud editing. MObI (Buburuzan et al. 2025) develops a unified diffusion model for cameraâLiDAR modalities to enable joint cross-sensor editing. GenMM (Singh et al. 2024) incorporates depth estimation into video diffusion models to support cross-modality editing propagation. Despite these advancements, existing methods remain constrained by predefined vehicle categories in the training data, provide limited control over object poses, and lack explicit mechanisms for cross-modality interaction. To overcome these limitations, we introduce a multimodal editing framework built upon 3DGS, leveraging a 3DGS-based vehicle template library to enable physically plausible insertions of diverse vehicle types with complex poses. In addition, we design a depthguided cross-modality condition module that utilizes 3DGSrendered depth to facilitate end-to-end joint optimization of images and point clouds.

## The Proposed Method

## Multimodal Joint Editing via Diffusion Models

Multimodal joint editing aims to precisely manipulate target objects across images and point clouds by jointly modeling and integrating cross-modality information. For each modality, we employ a conditional diffusion process to insert the target object at the location specified by the scene mask, ensuring seamless integration with the surrounding context. Moreover, both spatial and semantic consistency must be maintained across modalities.

Formally, let c denote an image, r the range-view representation of the point cloud, and $m _ { c }$ and mr the corresponding scene masks. We begin by masking the regions of interest in both modalities, defined as:

$$
c _ { 0 } = m _ { c } \odot c , \qquad r _ { 0 } = m _ { r } \odot r ,\tag{1}
$$

where $\odot$ denotes element-wise multiplication. A shared noise schedule $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ is applied to both masked inputs, and independent forward diffusion processes are constructed for each modality:

$$
\begin{array} { r } { q ( r _ { t } \mid r _ { t - 1 } ) = \mathcal { N } ( r _ { t } ; \sqrt { 1 - \beta _ { t } } r _ { t - 1 } , \beta _ { t } \mathbf { I } ) , } \\ { q ( c _ { t } \mid c _ { t - 1 } ) = \mathcal { N } ( c _ { t } ; \sqrt { 1 - \beta _ { t } } c _ { t - 1 } , \beta _ { t } \mathbf { I } ) . } \end{array}\tag{2}
$$

In the reverse process, we introduce two denoising models, $\epsilon _ { \theta _ { r } }$ and $\epsilon _ { \theta _ { r } }$ , to estimate the noise $\epsilon _ { r }$ and $\epsilon _ { c }$ added to the range image and RGB image separately. Under the guidance of target objects $o _ { r }$ and $o _ { c } ,$ each model progressively integrates the objects into the background. To enhance crossmodality consistency, the noise prediction for each modality incorporates complementary cues from the other modality. Intuitively, intermediate features from the diffusion process can serve as conditional inputs to guide the denoising of the other modality. The denoising models can thus be formulated as:

$$
\begin{array} { r l } & { \hat { \epsilon } _ { r } = \epsilon _ { \theta _ { r } } \big ( r _ { t } , c _ { C R } \big ( c _ { t } , t \big ) , o _ { r } , t \big ) , } \\ & { \hat { \epsilon } _ { c } = \epsilon _ { \theta _ { c } } \big ( c _ { t } , c _ { R C } \big ( r _ { t } , t \big ) , o _ { c } , t \big ) . } \end{array}\tag{3}
$$

A well-defined spatial correspondence between images and point clouds is established using the sensor extrinsics $T ( \cdot )$ and the target objectâs depth D. However, previous methods (Buburuzan et al. 2025) lack the depth of inserted objects. To address this limitation, we incorporate the 3DGS representation of the target object to render depth maps and facilitate cross-modality interaction. Accordingly, the crossmodality conditional encoders $c _ { C R } ( \cdot )$ and $c _ { R C } ( \cdot )$ are defined as:

$$
\begin{array} { r } { c _ { C R } ( c _ { t } , t ) = T _ { C R } ( \epsilon _ { \theta _ { c } } ^ { \prime } ( c _ { t } , t ) , D ) , } \\ { c _ { R C } ( r _ { t } , t ) = T _ { R C } ( \epsilon _ { \theta _ { r } } ^ { \prime } ( r _ { t } , t ) , D ) , } \end{array}\tag{4}
$$

where $\epsilon _ { \theta _ { c } } ^ { \prime } ( \cdot )$ and $\epsilon _ { \theta _ { r } } ^ { \prime } ( \cdot )$ represent intermediate features extracted from the image and point cloud denoising models, respectively. In this case, we rewrite the denoising models in Eq. (3) as follows:

$$
\begin{array} { r } { \hat { \epsilon } _ { r } = \epsilon _ { \theta _ { r } } \left( r _ { t } , T _ { C R } ( \epsilon _ { \theta _ { c } } ^ { \prime } ( c _ { t } , t ) , D ) , o _ { r } , t \right) , } \\ { \hat { \epsilon } _ { c } = \epsilon _ { \theta _ { c } } \left( c _ { t } , T _ { R C } ( \epsilon _ { \theta _ { r } } ^ { \prime } ( r _ { t } , t ) , D ) , o _ { c } , t \right) . } \end{array}\tag{5}
$$

They can be trained with a joint multi-modality objective function $\mathcal { L } _ { D M - M }$

$$
\mathcal { L } _ { D M - M } = \mathcal { L } _ { D M - C } + \mathcal { L } _ { D M - R } ,\tag{6}
$$

where $\mathcal { L } _ { D M - C }$ and $\mathcal { L } _ { D M - R }$ represent the loss terms for the image and point cloud modalities, respectively.

## Dual-Branch Joint Editing Framework

As illustrated in Figure 2, our framework is formulated based on Eq. (5) and comprises two denoising models: $\epsilon _ { \theta _ { c } } ( \cdot )$ for image editing and $\epsilon _ { \theta _ { r } } ( \cdot )$ for range image editing.

Latent Diffusion Model for Image Editing In the image branch, we employ an LDM tailored for image editing tasks. In addition to cross-modality guidance from the range image branch, we integrate three condition mechanisms to enable high-fidelity synthesis of the target object: (i) pixellevel preservation through image pasting, (ii) semantic-level guidance via embedding features, and (iii) a multi-branch joint optimization strategy.

Pixel-level detail preservation. To preserve the finegrained appearance detail of the target object, we adopt a simple yet effective paste-based strategy. During training, a pretrained image segmentation model (Kirillov et al. 2023; Liu et al. 2024) is used to extract the target object from the scene and paste it into the Region of Interest (ROI) specified by the scene mask. During inference, the 3DGS-rendered image of the target object is pasted into the ROI. The pasted image $c ^ { p } .$ , after being compressed by a variational autoencoder (VAE) (Kingma, Welling et al. 2013), is concatenated with the downsampled scene mask $m _ { c }$ to generate the pixellevel conditional features $h _ { c } ^ { p }$

$$
h _ { c } ^ { p } = \mathrm { C o n c a t } ( \mathrm { V A E } ( c ^ { p } ) , m _ { c } ) .\tag{7}
$$

This feature is then fused into the diffusion process through channel-wise concatenation with the latent representation.

Semantic consistency maintenance. To enhance semantic preservation and contextual alignment, we use a frozen image encoder coupled with a trainable MLPc to extract global semantic features from the target object. Following pioneering work (Yang et al. 2023), we adopt CLIP (Radford et al. 2021) as the image encoder due to its strong capability in capturing high-level semantic representations. The resulting CLIP embedding is a conditional signal integrated into the image denoising model via a cross-attention module. The semantic condition $h _ { c } ^ { s }$ is formulated as:

$$
h _ { c } ^ { s } = \mathbf { M } \mathbf { L } \mathbf { P } _ { c } \big ( \mathbf { C } \mathbf { L } \mathbf { I } \mathbf { P } ( o _ { c } ) \big ) .\tag{8}
$$

Multi-branch optimization. To more effectively utilize the coarse editing results of the pasted image, inspired by DCI-VTON (Gou et al. 2023), we adopt a dual-branch optimization strategy, comprising a reconstruction branch and a refinement branch, to train the denoising network $\epsilon _ { \theta _ { c } } ( \cdot )$ . The reconstruction branch synthesizes the driving scene conditioned on the reference input, promoting semantic alignment and structural consistency. In parallel, the refinement branch refines the pasted image by enhancing fine-grained details, thereby enabling high-fidelity data synthesis. Specifically, the reconstruction branch synthesizes images by applying a forward diffusion followed by a reverse denoising process on the real image $c ^ { g }$ . In line with standard LDM designs, $c ^ { g }$ is first encoded into a latent representation $z _ { c , 0 } ^ { g }$ via a VAE. Gaussian noise is then added according to $\operatorname { E q . } ( 2 )$ , and the resulting noisy latent is concatenated with the pixel-level condition $h _ { c } ^ { p }$ as input to the denoising network. In parallel, the refinement branch targets the masked region of the pasted image $c ^ { p }$ . The latent representation $z _ { c , 0 } ^ { p } ,$ obtained from VAE encoding of $c ^ { p }$ , is likewise perturbed with Gaussian noise and concatenated with the pixel-level condition $h _ { c } ^ { p }$

<!-- image-->  
Range Image FeatureFigure 2: Overview of the proposed MultiEditor framework. A dual-branch diffusion model is employed to edit multimodal data. 3DGS QEach branch incorporates a multi-level appearance control mechanism for fidelity, while a cross-modality condition module Driving Sceneenhances consistency between modalities.

(a) (b)The reconstruction branch guides the denoising model by supervising its noise prediction, with the objective function $\bar { \mathcal { L } _ { r e c o n - C } }$ defined as:

$$
\mathcal { L } _ { r e c o n - C } = \left| \left| \epsilon _ { c } - \epsilon _ { \theta _ { c } } ( z _ { c , t } ^ { g } , h _ { c } ^ { p } , h _ { c } ^ { s } , h _ { c } ^ { r } , t ) \right| \right| _ { 2 } ^ { 2 } ,\tag{9}
$$

where $h _ { c } ^ { r }$ denotes the cross-modality condition. The refinement branch obtains the denoised latent representation $\hat { z } _ { c , 0 } ^ { g }$ by reversing the noising process using the predicted noise. This latent is then decoded into an image $\hat { c } = D _ { V A E } \big ( \hat { z } _ { c , 0 } ^ { g } \big )$ To supervise the refinement, we compute both the L2 loss and the perceptual loss (Johnson, Alahi, and Fei-Fei 2016) between the reconstructed image cË and the real image $c ^ { g }$ defined as:

$$
\mathcal { L } _ { r e f i n e - C } = \| c ^ { g } - \hat { c } \| _ { 2 } ^ { 2 } + \sum _ { m = 1 } ^ { 5 } \| \phi _ { m } ( \hat { c } ) - \phi _ { m } ( c ^ { g } ) \| _ { 1 } .\tag{10}
$$

where $\phi _ { m } ( \cdot )$ indicates the m-th feature map in a VGG-19 (?) network pre-trained on ImageNet (?). Accordingly, the overall objective function for the image branch is formulated as:

$$
\mathcal { L } _ { D M - C } = \mathcal { L } _ { r e c o n - C } + \lambda _ { r e f i n e - C } \mathcal { L } _ { r e f i n e - C } ,\tag{11}
$$

where $\lambda _ { r e f i n e - C }$ serves as a hyperparameter to balance the contributions of the two loss components.

Point Projection Aggregated Image Latent Diffusion Model for Range Image Editing Given Sampled Image Featurethe structural similarity between range images and RGB im-Camera-to-LiDAR Conditionages, we adapt the LDM from the image branch to accommodate the characteristics of range data. While retaining the Linear Linear QCross Attentionsemantic condition signal and overall optimization strategy, ange Image Projection K VOffsets Weights Image Featurewe customize the VAE and the pixel-level condition mod-0.5 0.3 0.1 0.1ule to better model the spatial and geometric information in point cloud data.

Sampled Range Image FeatureWe adopt the standard VAE training paradigm, jointly op-LiDAR-to-Camera Conditiontimizing the encoder and decoder by maximizing the ELBO (Rezende, Mohamed, and Wierstra 2014) to learn a compact latent representation of range images. To mitigate the blurriness typically introduced by reconstruction loss, we further incorporate an adversarial discriminator (Isola et al. 2017; Rombach et al. 2022), encouraging sharper outputs and improved structural fidelity.

Pixel-level spatial preservation in the range view. To preserve the spatial structure of the target object, we also adopt a paste-based strategy in the point cloud branch. Due to the difficulty of acquiring LiDAR point clouds for the target object under diverse poses and positions, we leverage the depth rendering capabilities of 3DGS to synthesize dense depth maps that serve as spatial priors, effectively capturing the geometric structure of the object. During training, we interpolate sparse LiDAR data to generate a dense depth map and extract the target objectâs depth using segmentation masks produced by the image branch. The extracted depth is then pasted into the ROI specified by the scene mask. To reduce boundary artifacts caused by interpolation noise, we apply median filtering to suppress unreliable depth estimates. During inference, we directly use depth maps rendered by the 3DGS model. Analogous to the image branch, we construct the pixel-level condition feature $h _ { r } ^ { p }$ by concatenating the VAE-compressed pasted range image $\bar { r ^ { p } }$ with the downsampled scene mask $m _ { r }$

<!-- image-->  
Figure 3: Cross-modality condition module. We perform bidirectional conditioning between LiDAR and camera modalities on latent representations, guided by depth priors and spatial transformations.

$$
h _ { r } ^ { p } = \mathrm { C o n c a t } ( \mathrm { V A E } ( r ^ { p } ) , m _ { r } ) .\tag{12}
$$

This feature is integrated into the diffusion model of the range image branch via channel-wise concatenation.

## Cross-Modality Condition Module

The key to multimodal editing is boosting the crossmodality consistency, which potentially relies on crossmodality conditions. While the depth prior of the target object and the camera projection inherently establish a spatial correspondence between image and point cloud modalities, this alignment is often inaccurate due to depth estimation errors. To address these challenges, we introduce a depth-guided deformable condition module, illustrated in Figure 3. This module initially establishes a coarse modality alignment by leveraging the depth prior and geometric transformations, followed by a deformable cross-attention mechanism that adaptively retrieves local features from the complementary modality, thereby generating cross-modality control signals.

Camera-to-LiDAR condition Given a location (Ï, Î¸) in the range image latent $z _ { r }$ with corresponding range value $r ,$ we first convert it into 3D Cartesian coordinates:

$$
\begin{array} { l } { x = r \cdot \cos ( \phi ) \cdot \sin ( \theta ) , } \\ { y = r \cdot \cos ( \phi ) \cdot \cos ( \theta ) , } \\ { z = r \cdot \sin ( \phi ) . } \end{array}\tag{13}
$$

Subsequently, we project the 3D point onto the image plane using the camera intrinsic matrix K and the LiDARto-camera extrinsic transformation $T _ { C R } = [ R _ { C R } \ | \ t _ { C R } ] \mathrm { { : } }$ :

$$
\left[ \begin{array} { l l l } { \boldsymbol { u } } & { \boldsymbol { v } } & { 1 } \end{array} \right] ^ { \top } = \frac { K } { d } \left( R _ { C R } \left[ \boldsymbol { x } \quad \boldsymbol { y } \quad \boldsymbol { z } \right] ^ { \top } + t _ { C R } \right) ,\tag{14}
$$

where $( u , v )$ denotes the pixel location in the image and d is a normalization factor. Based on this geometric projection,

we employ a deformable cross-attention module to adaptively sample local reference features around (u, v) from the image latent $z _ { c } .$ , forming a camera-to-LiDAR control signal:

$$
h _ { r } ^ { c } = \mathrm { C r o s s D A t t n } ( z _ { r } ( \phi , \theta ) , z _ { c } , ( u , v ) ) .\tag{15}
$$

The output at $( \phi , \theta )$ is then computed as:

$$
z _ { r } ^ { \mathrm { o u t } } ( \phi , \theta ) = z _ { r } ( \phi , \theta ) + \operatorname { t a n h } ( \alpha _ { r } ) \cdot h _ { r } ^ { c } ,\tag{16}
$$

where $\alpha _ { r }$ is a zero-initialization gate.

LiDAR-to-camera condition Based on the established correspondence between the range image and the RGB image, we inject range features into the image branch. For a pixel $( u , v )$ in the image latent $z _ { c } .$ , corresponding to a range image location $( \phi , \theta )$ , the cross-modality condition is formulated as:

$$
h _ { c } ^ { r } = \mathrm { C r o s s D A t t n } ( z _ { c } ( u , v ) , z _ { r } , ( \phi , \theta ) ) .\tag{17}
$$

We then update the image latent representation as:

$$
z _ { c } ^ { \mathrm { o u t } } ( u , v ) = z _ { c } ( u , v ) + \operatorname { t a n h } ( \alpha _ { c } ) \cdot h _ { c } ^ { r } ,\tag{18}
$$

where $\alpha _ { c }$ is a zero-initialization gate. By exploiting the 2Dâ3D correspondences, our condition module adaptively enhances local cross-modality consistency.

## Experiments

## Experimental Setups

Dataset and Data Construction We train MultiEditor with a reconstruction-based strategy. Specifically, we simulate editing scenarios by deliberately occluding the same target object in image and point cloud modalities. The model is trained to reconstruct the occluded regions conditioned on the target object accurately. As no publicly available dataset is designed explicitly for joint multimodal editing, we construct a dedicated dataset derived from the widely adopted KITTI benchmark (Geiger et al. 2013). Details of the dataset construction process are provided in the supplementary materials.

Evaluation Metrics For joint multimodal data editing, we evaluate both the realism of single-modality data within masked regions and the cross-modality consistency between them. For image modality, we use the Frechet Inception Dis- Â´ tance (FID) (Heusel et al. 2017) for visual realism, Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al. 2018) for perceptual similarity, and CLIP-based image similarity (CLIP-I) (Hessel et al. 2021) for semantic alignment. For the point cloud modality, we adopt Chamfer Distance (CD) (Ma, Yang, and Latecki 2010) and Frechet Point Cloud Â´ Distance (FPD) (Shu, Park, and Kwon 2019) to quantify spatial and perceptual accuracy. Additionally, we follow the X-Drive (Xie et al. 2024) protocol and employ the Depth Alignment Score (DAS) to evaluate the consistency between generated images and point clouds.

<table><tr><td rowspan="2">Modality</td><td rowspan="2">Method</td><td colspan="3">Image quality</td><td colspan="2">Point clouds quality</td><td>Cross-modality</td></tr><tr><td>FID â</td><td>LPIPS â</td><td>CLIP-I â</td><td>CD â</td><td>FPD â</td><td>DAS â</td></tr><tr><td rowspan="3">C</td><td>SD</td><td>67.92</td><td>0.3183</td><td>0.7805</td><td></td><td></td><td></td></tr><tr><td>PbE</td><td>54.28</td><td>0.3293</td><td>0.7875</td><td></td><td></td><td></td></tr><tr><td>AnyDoor</td><td>26.45</td><td>0.2680</td><td>0.7925</td><td></td><td></td><td></td></tr><tr><td>L</td><td>RangeLDM</td><td></td><td></td><td>7</td><td>33.23</td><td>293.45</td><td></td></tr><tr><td rowspan="2">C+L</td><td>AnyDoor+RangeLDM</td><td></td><td></td><td></td><td></td><td></td><td>11.34</td></tr><tr><td>MultiEditor(ours)</td><td>25.07</td><td>0.1477</td><td>0.8063</td><td>1.65</td><td>97.49</td><td>3.16</td></tr></table>

Table 1: Quantitative comparison with driving data editing algorithms. For each column, the best value is highlighted by bold.A

<!-- image-->

<!-- image-->  
AnyDoor+RangeLDM

<!-- image-->  
MultiEditor (Ours)

sFigure 4: Editing results on regular vehicles. MultiEditor achieves better appearance and geometric fidelity than the baseline.  
<!-- image-->  
Figure 5: Editing results on atypical vehicles.

Baselines For single-modality editing, we compare stateof-the-art editing algorithms for images, i.e. Stable Diffusion Inpainting (SD) (Rombach et al. 2022), Paint-by-Example (PbE) (Yang et al. 2023), AnyDoor (Chen et al. 2024), and for point clouds, i.e. RangeLDM (Hu, Zhang, and Hu 2024). Furthermore, following the strategy adopted in X-Drive (Xie et al. 2024), we combine AnyDoor (Chen et al. 2024) and RangeLDM (Hu, Zhang, and Hu 2024) respectively for images and point clouds as a multi-modality baseline.

Training Setup Our MultiEditor has a dual-branch architecture. During training, both branches are initialized with pretrained weights from the PbE model, while the newly introduced parameters are initialized randomly. The training process is divided into five stages, with detailed strategies in the supplementary materials.

## Quantitative Results

)Table 1 presents quantitative comparisons with several baseline methods. For image quality, MultiEditor achieves the best performance across all metrics, reducing FID to 25.07 and LPIPS to 0.1477, indicating clear gains in visual fidelity and perceptual similarity. It also obtains a CLIP-I score of 0.8063, demonstrating strong semantic alignment with the target objects. For point cloud quality, our approach significantly outperforms RangeLDM, lowering CD from 33.23 to 1.65 and FPD from 293.45 to 97.49, reflecting substantial improvements in geometric accuracy and structural consistency. Regarding cross-modality consistency, MultiEditor achieves a DAS of 3.16, notably better than the baseline combination of AnyDoor and RangeLDM. This result validates the effectiveness of our cross-modality condition module in improving cross-modality consistency for joint editing tasks.

## Qualitative Analysis

Editing of Regular Vehicles Figure 4 presents multimodal editing results on regular vehicles. For the image modality, while existing editing methods maintain semantic consistency, they often struggle to preserve finegrained geometric structures, resulting in inaccurate shape reconstruction, incorrect pose estimation, and unnatural background blending. MultiEditor generates accurate and context-consistent image content, exhibiting superior visual fidelity and editing controllability. For the point cloud modality, mainstream approaches such as RangeLDM lack object-level priors, leading to geometric artifacts and semantic drift. Benefiting from the depth and structural priors provided by 3DGS, MultiEditor generates point clouds with clear structure.

<!-- image-->  
Figure 6: Qualitative results of cross-modality consistency.

<table><tr><td>Methods</td><td>LPIPS â</td><td>FPD â</td><td>DAS â</td></tr><tr><td>w/o pixel-level condition</td><td>0.2343</td><td>138.45</td><td>3.32</td></tr><tr><td>w/o semantic condition</td><td>0.1492</td><td>141.61</td><td>3.21</td></tr><tr><td>w/o reconstruction branch</td><td>0.1588</td><td>644.48</td><td>3.84</td></tr><tr><td>w/o cross-modality</td><td>0.1644</td><td>98.80</td><td>3.20</td></tr><tr><td>feature addition condition</td><td>0.1532</td><td>220.51</td><td>3.23</td></tr><tr><td>full model</td><td>0.1477</td><td>97.49</td><td>3.16</td></tr></table>

Table 2: Ablation studies of MultiEditor modules.

Editing of Atypical Vehicles Figure 5 presents multimodal editing results on atypical vehicles under diverse poses and environments. We construct 3DGS models of atypical vehicles using the 3DRealCar dataset (Du et al. 2024). In the image modality, existing methods often produce distorted or semantically inconsistent editing results on irregular structures. In contrast, MultiEditor preserves complex geometry and generates visually coherent content that blends naturally with the background. For point clouds, existing methods struggle with structural fidelity due to the lack of geometric priors. With 3DGS guidance, MultiEditor synthesizes complete and semantically aligned point clouds.

Cross-Modality Consistency To qualitatively evaluate cross-modality consistency, we visualize the results of joint editing by projecting real and MultiEditor-generated point clouds onto three types of images: real images, images generated by AnyDoor, and those generated by MultiEditor. As illustrated in Figure 6, MultiEditor yields better geometric alignment between point clouds and images than AnyDoor. Furthermore, projections of generated point clouds by MultiEditor onto their corresponding generated images exhibit noticeably superior alignment than those using real point clouds, underscoring its ability to produce structurally and visually consistent multimodal outputs.

<table><tr><td colspan="5">2D Detection</td></tr><tr><td>Method</td><td>DataType</td><td>AP@0.5 â</td><td>AP@0.7â</td><td>mAP 0.5-0.95 â</td></tr><tr><td>Yolo v5</td><td>Real Real + Gen</td><td>27.0 31.5(+4.5%)</td><td>21.2 23.0(+1.8%)</td><td>16.9 18.2(+1.3%)</td></tr><tr><td></td><td></td><td>3D Detection</td><td></td><td></td></tr><tr><td></td><td colspan="4"></td></tr><tr><td>Method</td><td>DataType</td><td>AP@0.7 Easy â</td><td>AP@0.7 Mod. â</td><td>AP@0.7 Hard â</td></tr><tr><td>PointPillars</td><td>Real Real + Gen</td><td>43.02 48.36(+5.34%)</td><td>34.23 35.45(+1.22%)</td><td>28.54 29.17(+0.63%)</td></tr></table>

Table 3: Detection performance on van-class objects using 2D and 3D detection models trained with real and augmented (real + generated) data.

## Ablation Studies

Multi-Level Appearance Control Mechanism Analysis We evaluate the influence of the multi-level appearance control mechanism on generation performance, with quantitative results summarized in Table 2. First, removing the pixellevel conditionâwhich prevents explicit priors from 3DGS from being pasted into masked regionsâsubstantially degrades image appearance and point cloud geometry, underscoring the importance of low-level cues for accurate editing. Second, we ablate the semantic condition by excluding global semantic features extracted via CLIP. While the image modality remains relatively unaffectedâpossibly due to compensation from other fine-grained signalsâthe quality of point cloud generation noticeably degrades, highlighting the vital role of semantic guidance in textureless 3D synthesis. Lastly, removing the reconstruction branch causes a marked decline in visual and geometric fidelity, confirming its necessity for refining the generated outputs.

Cross-Modality Module Analysis We ablate the crossmodality condition module to turn off mutual guidance between modalities. As shown in Table 2, removing this module substantially degrades cross-modality consistency. While the shared 3DGS prior offers some semantic and structural constraints, it alone is inadequate for achieving precise alignment between modalities. Moreover, substituting deformable attention with naive feature addition degrades the modelâs capacity to accurately align local structures across modalities, reducing consistency in the generated results.

## Downstream Task Benefits

To evaluate the effectiveness of MultiEditor in downstream perception tasks, we augment the training samples of the van class in the KITTI dataset to improve detection performance for this underrepresented category. Using YOLOv5 (Redmon et al. 2016) and PointPillars (Lang et al. 2019) as 2D and 3D detectors, we follow the PointPillars split of the KITTI dataset for training and evaluation. As shown in Table 3, van-class vehicles are scarce in the training set (with only 1,297 instances), leading to suboptimal detection accuracy. To mitigate this, we select 10 van instances from the 3DRealCar dataset and insert them into training images using MultiEditor. This process yields 1,192 new samples containing van-class vehicles, which are then merged with the original training set. Results show that incorporating

MultiEditor-generated samples significantly improves vanclass detection accuracy for both 2D and 3D detectors. This demonstrates the practical utility of MultiEditor in alleviating long-tail data imbalance and boosting detection accuracy for rare object classes.

## Conclusions

We present MultiEditor, a novel multimodal object editing framework that leverages 3DGS priors to enable joint editing of images and LiDAR point clouds. By incorporating a dual-branch diffusion architecture, multi-level control mechanisms, and a cross-modality condition module, MultiEditor effectively models the joint distribution across modalities, enabling high-fidelity and consistent scene editing. Extensive experiments demonstrate that our method enables accurate, consistent, and controllable multimodal editing for autonomous driving scenarios.

## References

Buburuzan, A.; Sharma, A.; Redford, J.; Dokania, P. K.; and Mueller, R. 2025. Mobi: Multimodal object inpainting using diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1974â1984.

Chae, Y.; Kim, H.; and Yoon, K.-J. 2024. Towards robust 3d object detection with lidar and 4d radar fusion in various weather conditions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15162â15172.

Chen, X.; Huang, L.; Liu, Y.; Shen, Y.; Zhao, D.; and Zhao, H. 2024. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6593â6602.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

Du, X.; Wang, Y.; Sun, H.; Wu, Z.; Sheng, H.; Wang, S.; Ying, J.; Lu, M.; Zhu, T.; Zhan, K.; et al. 2024. 3drealcar: An in-the-wild rgb-d car dataset with 360-degree views. arXiv preprint arXiv:2406.04875.

Gao, R.; Chen, K.; Xie, E.; Hong, L.; Li, Z.; Yeung, D.- Y.; and Xu, Q. 2023. Magicdrive: Street view generation with diverse 3d geometry control. arXiv preprint arXiv:2310.02601.

Geiger, A.; Lenz, P.; Stiller, C.; and Urtasun, R. 2013. Vision meets robotics: The kitti dataset. The international journal of robotics research, 32(11): 1231â1237.

Gou, J.; Sun, S.; Zhang, J.; Si, J.; Qian, C.; and Zhang, L. 2023. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In Proceedings of the 31st ACM International Conference on Multimedia, 7599â 7607.

Hessel, J.; Holtzman, A.; Forbes, M.; Bras, R. L.; and Choi, Y. 2021. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718.

Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; and Hochreiter, S. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30.

Hu, Q.; Zhang, Z.; and Hu, W. 2024. Rangeldm: Fast realistic lidar point cloud generation. In European Conference on Computer Vision, 115â135. Springer.

Huang, B.; Wen, Y.; Zhao, Y.; Hu, Y.; Liu, Y.; Jia, F.; Mao, W.; Wang, T.; Zhang, C.; Chen, C. W.; et al. 2025. Subjectdrive: Scaling generative data in autonomous driving via subject control. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 3617â3625.

Isola, P.; Zhu, J.-Y.; Zhou, T.; and Efros, A. A. 2017. Imageto-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 1125â1134.

Johnson, J.; Alahi, A.; and Fei-Fei, L. 2016. Perceptual losses for real-time style transfer and super-resolution. In European conference on computer vision, 694â711. Springer.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G.Â¨ 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139â1.

Kingma, D. P.; Welling, M.; et al. 2013. Auto-encoding variational bayes.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.-Y.; et al. 2023. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, 4015â4026.

Lang, A. H.; Vora, S.; Caesar, H.; Zhou, L.; Yang, J.; and Beijbom, O. 2019. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12697â12705.

Li, B.; Guo, J.; Liu, H.; Zou, Y.; Ding, Y.; Chen, X.; Zhu, H.; Tan, F.; Zhang, C.; Wang, T.; et al. 2025a. Uniscene: Unified occupancy-centric driving scene generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 11971â11981.

Li, H.; Yang, Z.; Qian, Z.; Zhao, G.; Huang, Y.; Yu, J.; Zhou, H.; and Liu, L. 2025b. Dualdiff: Dual-branch diffusion model for autonomous driving with semantic fusion. arXiv preprint arXiv:2505.01857.

Liang, Y.; Yan, Z.; Chen, L.; Zhou, J.; Yan, L.; Zhong, S.; and Zou, X. 2025. DriveEditor: A Unified 3D Information-Guided Framework for Controllable Object Editing in Driving Scenes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 5164â5172.

Liu, S.; Zeng, Z.; Ren, T.; Li, F.; Zhang, H.; Yang, J.; Jiang, Q.; Li, C.; Yang, J.; Su, H.; et al. 2024. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, 38â55. Springer.

Ljungbergh, W.; Taveira, B.; Zheng, W.; Tonderski, A.; Peng, C.; Kahl, F.; Petersson, C.; Felsberg, M.; Keutzer, K.;

Tomizuka, M.; et al. 2025. R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation. arXiv preprint arXiv:2506.07826.

Ma, T.; Yang, X.; and Latecki, L. J. 2010. Boosting chamfer matching by learning chamfer distance normalization. In European Conference on Computer Vision, 450â463. Springer.

Nakashima, K.; and Kurazume, R. 2024. Lidar data synthesis with denoising diffusion probabilistic models. In 2024 IEEE International Conference on Robotics and Automation (ICRA), 14724â14731. IEEE.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748â8763. PmLR.

Ran, H.; Guizilini, V.; and Wang, Y. 2024. Towards realistic scene generation with lidar diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14738â14748.

Redmon, J.; Divvala, S.; Girshick, R.; and Farhadi, A. 2016. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, 779â788.

Rezende, D. J.; Mohamed, S.; and Wierstra, D. 2014. Stochastic backpropagation and approximate inference in deep generative models. In International conference on machine learning, 1278â1286. PMLR.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684â 10695.

Shu, D. W.; Park, S. W.; and Kwon, J. 2019. 3d point cloud generative adversarial network based on tree structured graph convolutions. In Proceedings of the IEEE/CVF international conference on computer vision, 3859â3868.

Singh, B.; Kulharia, V.; Yang, L.; Ravichandran, A.; Tyagi, A.; and Shrivastava, A. 2024. Genmm: Geometrically and temporally consistent multimodal data generation for video and lidar. arXiv preprint arXiv:2406.10722.

Wang, T.; Hu, X.; Heng, P.-A.; and Fu, C.-W. 2022. Instance shadow detection with a single-stage detector. IEEE transactions on pattern analysis and machine intelligence, 45(3): 3259â3273.

Wang, X.; Zhu, Z.; Huang, G.; Chen, X.; Zhu, J.; and Lu, J. 2024. Drivedreamer: Towards real-world-drive world models for autonomous driving. In European conference on computer vision, 55â72. Springer.

Wei, D.; Li, Z.; and Liu, P. 2025. Omni-scene: Omnigaussian representation for ego-centric sparse-view scene reconstruction. In Proceedings of the Computer Vision and Pattern Recognition Conference, 22317â22327.

Wen, Y.; Zhao, Y.; Liu, Y.; Jia, F.; Wang, Y.; Luo, C.; Zhang, C.; Wang, T.; Sun, X.; and Zhang, X. 2024. Panacea: Panoramic and controllable video generation for

autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6902â 6912.

Wu, Y.; Zhang, K.; Qian, J.; Xie, J.; and Yang, J. 2024. Text2lidar: Text-guided lidar point cloud generation via equirectangular transformer. In European Conference on Computer Vision, 291â310. Springer.

Xia, Z.; Pan, X.; Song, S.; Li, L. E.; and Huang, G. 2022. Vision transformer with deformable attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4794â4803.

Xie, Y.; Xu, C.; Peng, C.; Zhao, S.; Ho, N.; Pham, A. T.; Ding, M.; Tomizuka, M.; and Zhan, W. 2024. X-Drive: Cross-modality consistent multi-sensor data synthesis for driving scenarios. arXiv preprint arXiv:2411.01123.

Yan, Y.; Xu, Z.; Lin, H.; Jin, H.; Guo, H.; Wang, Y.; Zhan, K.; Lang, X.; Bao, H.; Zhou, X.; et al. 2025. Streetcrafter: Street view synthesis with controllable video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 822â832.

Yang, B.; Gu, S.; Zhang, B.; Zhang, T.; Chen, X.; Sun, X.; Chen, D.; and Wen, F. 2023. Paint by example: Exemplarbased image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 18381â18391.

Yang, L.; Kang, B.; Huang, Z.; Zhao, Z.; Xu, X.; Feng, J.; and Zhao, H. 2024. Depth anything v2. Advances in Neural Information Processing Systems, 37: 21875â21911.

Yu, Z.; Wang, H.; Yang, J.; Wang, H.; Cao, J.; Ji, Z.; and Sun, M. 2025. Sgd: Street view synthesis with gaussian splatting and diffusion prior. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 3812â3822. IEEE.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586â595.

Zhao, G.; Wang, X.; Zhu, Z.; Chen, X.; Huang, G.; Bao, X.; and Wang, X. 2025. Drivedreamer-2: Llm-enhanced world models for diverse driving video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 10412â10420.

Zhao, Y.; Lv, W.; Xu, S.; Wei, J.; Wang, G.; Dang, Q.; Liu, Y.; and Chen, J. 2024. DETRs Beat YOLOs on Real-time Object Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16965â16974.

## Overview

This supplementary material first provides detailed implementation specifics. We then present additional qualitative results to further demonstrate the effectiveness of the proposed method. Lastly, we discuss the limitations of our framework and outline potential directions for future work.

## Detailed Implementation Specifics

In this part, we explain details in the implementation of MultiEditor including the data construction process, training schedule, and evaluation metrics. Our code will be made publicly available upon the acceptance of this paper.

## Data Construction Process

As no existing dataset is tailored for joint multimodal object editing, we construct a dedicated dataset based on the widely used KITTI benchmark (Geiger et al. 2013). Specifically, we first apply the Grounding DINO model (Liu et al. 2024) for text-guided object detection, followed by the Segment Anything Model (SAM) (Kirillov et al. 2023) to accurately segment target objects and generate corresponding scene masks. Subsequently, we leverage the calibrated geometric transformation between the camera and LiDAR to project the image-space masks onto the LiDAR range image coordinate system. This allows for accurate localization and extraction of the corresponding point clouds and their associated masks.

Traditional mask-guided image editing methods (Liang et al. 2025; Buburuzan et al. 2025) typically restrict the generative scope to within the predefined mask boundaries, limiting their applicability in real-world scenarios where visual elements such as shadows often extend beyond these regions. To address this issue, we enhance the image branch to support unconstrained editing by post-processing the generated image data.

Specifically, as shown in Figure 7, we employ the objectshadow detection model (Wang et al. 2022) to extract shadow masks cast by the target objects and utilize a pretrained image inpainting model (Rombach et al. 2022) to perform high-quality shadow removal, resulting in clean, shadow-free images suitable for downstream unconstrained editing tasks.

After completing the above inpainting process, we manually screened the generated data to ensure high-quality training samples. Specifically, we removed instances with poorquality object or shadow masks and those exhibiting visible artifacts in the inpainted regions. This rigorous filtering yields a high-quality dataset of 4,021 imageâpoint cloud pairs for training and 1,256 for testing.

## Training Schedule

The overall training pipeline of MultiEditor comprises five stages:

â¢ Stage 1: We train a variational autoencoder (VAE) from scratch for the range image modality in the point cloud branch.

â¢ Stage 2: The range image VAE is frozen, and a latent diffusion model is trained on its latent representations.

<!-- image-->  
Figure 7: The pipeline of dataset construction. We use the object-shadow detection model (Wang et al. 2022) to predict pairs of object and shadow masks in the real image. Then we apply the inpainting model (Rombach et al. 2022) to get a deshadowed image.

â¢ Stage 3: The image VAE is frozen, and a latent diffusion model is trained on unprocessed (i.e., shadowcontaining) images. This enables object insertion within masked regions, albeit still constrained to the mask.

â¢ Stage 4: We fine-tune the diffusion model from Stage 3 using only the refinement strategy in the image branch, with training performed on deshadowed image data. This encourages the model to learn generation distributions beyond the mask boundaries, enabling unconstrained editing capability.

â¢ Stage 5: Finally, we perform end-to-end joint training of the entire multimodal framework based on the module trainable states defined in Figure 2. We aim to improve cross-modal consistency and overall editing performance further.

During all training stages, we use four NVIDIA L20 GPUs. In Stage 1, the VAE for LiDAR range images is trained with a batch size of 2 and a learning rate of 4.5e-5 for 40 epochs, with the discriminator activated after 1000 iterations. In Stage 2, the LiDAR LDM is trained with a batch size of 2 and a learning rate 4.0e-5 for 100 epochs. In Stage 3, the image LDM is trained with a batch size of 2 and a learning rate 1.0e-5 for 40 epochs. In Stage 4, we fine-tune the image LDM from Stage 3 using deshadowed images, with a batch size of 2 and a learning rate of 1.0e-5 for 160 epochs. Finally, Stage 5 performs end-to-end training of the whole multimodal framework with a batch size of 1 and a learning rate 2.0e-5 for 60 epochs. For ablation studies, we adopt the same complete training configuration as described above.

In terms of image size configuration, following the experimental setup of SGD (Yu et al. 2025), we apply a center crop of size 600 Ã 375 to the input images. Correspondingly, the range images of point clouds are cropped based on the field of view (FOV) of the cropped images and resized to 128 Ã 64. For downstream tasks, the cropped image and point cloud patches are seamlessly stitched back into the original data.

To improve robustness, we perform data augmentation on the target objects. For the image branch, we apply Random-BrightnessContrast, Rotate, HueSaturationValue, Blur, and

<!-- image-->  
Figure 8: MultiEditor demonstrates high controllability and flexible editing of complex-shaped vehicles. A roller vehicle is inserted into the scene at 45â¦ intervals, showcasing consistent and precise object editing performance.

ElasticTransform, each with a 20% probability. For the point cloud branch, only the Rotate augmentation is used, also with a 20% probability.

In addition, the value of $\lambda _ { r e f i n e - C }$ in Equation (11) of the main text is set to 0.01.

## Evaluation Metrics

In our quantitative evaluation, we apply the FID metric and LPIPS metric from DriveEditor (Liang et al. 2025), CLIP-I metric from AnyDoor (Chen et al. 2024), CD metric from PVD (Ma, Yang, and Latecki 2010), and FPD metric from R2DM (Nakashima and Kurazume 2024). For the DAS metric, we follow the protocol of X-DRIVE (Xie et al. 2024) by applying a pretrained DepthAnythingV2 model (Yang et al. 2024) with a ViT-B backbone (Dosovitskiy et al. 2020) to the synthesized images for depth estimation. The estimated depth is then rescaled to an absolute scale using ground-truth LiDAR point clouds. Finally, we project the generated point clouds into the image space to obtain sparse depth values and compute the mean absolute error (MAE) against the estimated depth map as our DAS score.

## More Visualization Results

## More Editing Results

As shown in Figure 8, we demonstrate the high-fidelity and flexible editing capabilities of MultiEditor on atypical vehicles. Specifically, we reconstruct a 3D Gaussian Splatting (3DGS) representation of a roller vehicle from the 3DReal-Car dataset (Du et al. 2024). RGB images and depth maps are rendered at 45â¦ intervals, and the target object is inserted into the scene using MultiEditor. Figure 8 shows generation results across different viewpoints, highlighting MultiEditorâs strong controllability and geometric consistency in editing complex driving scenes.

## Visualization of Generated Data for Downstream Tasks

To evaluate the effectiveness of our generated data in downstream perception tasks, we augment existing driving scenes by inserting van-class vehicles with diverse poses and distances into both the image and point cloud modalities, as shown in Figure 9. This augmentation addresses the long-tail distribution problem commonly found in detection datasets, where van-class objects are underrepresented. By enrichingInpainting ModelInpainting Model the training set with high-quality, multimodal synthetic samples, we observe improved performance in 2D and 3D object detectors. The inserted objects are geometrically and photometrically consistent with the surrounding environment, demonstrating the practical utility of MultiEditor in realworld data enhancement scenarios.

<!-- image-->  
Figure 9: Multimodal data generation for downstream tasks. We insert van-class vehicles at varying poses and distances into image and point cloud modalities. This enhances the diversity of training data and improves the performance of 2D and 3D detectors in recognizing van-class objects.

## The Impact of the Shadow Refinement Training Stage

As shown in Figure 10, we evaluate the effect of the shadow refinement stage on editing quality. The left column shows results without refinement, where inserted objects suffer from lighting inconsistency, unnatural shadows, and poor scene integration. In contrast, the right column presents results with shadow refinement, leading to more coherent illumination, improved shadow consistency, and more realistic blending with the environment.

## Limitation and Future Works

This work focuses on the joint editing of consistent multimodal data. Due to limited computational resources, we leave several natural extensions of the MultiEditor framework for future exploration. First, our current implementation supports editing only a single-frame image and the corresponding LiDAR point cloud within the same field of view (FOV). A straightforward extension would support multiview image inputs combined with a single-frame LiDAR scan. Second, integrating temporal attention mechanisms, such as those proposed in Panacea (Wen et al. 2024), into MultiEditor could enable long-sequence multimodal, controllable, and temporally consistent video editing.

<!-- image-->

(a) Without the shadow refinement stage.  
<!-- image-->  
(b) With the shadow refinement stage.  
Figure 10: Visualization results with and without the shadow refinement stage.