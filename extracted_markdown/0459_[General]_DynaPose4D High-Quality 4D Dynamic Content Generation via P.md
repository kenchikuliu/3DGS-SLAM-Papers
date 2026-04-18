# DYNAPOSE4D: HIGH-QUALITY 4D DYNAMIC CONTENT GENERATION VIA POSE ALIGNMENT LOSS

Jing Yang

Yufeng Yang

Sun Yat-sen University

## ABSTRACT

Recent advancements in 2D and 3D generative models have expanded the capabilities of computer vision. However, generating high-quality 4D dynamic content from a single static image remains a significant challenge. Traditional methods have limitations in modeling temporal dependencies and accurately capturing dynamic geometry changes, especially when considering variations in camera perspective. To address this issue, we propose DynaPose4D, an innovative solution that integrates 4D Gaussian Splatting (4DGS) techniques with Category-Agnostic Pose Estimation (CAPE) technology. This framework uses 3D Gaussian Splatting to construct a 3D model from single images, then predicts multiview pose keypoints based on one-shot support from a chosen view, leveraging supervisory signals to enhance motion consistency. Experimental results show that DynaPose4D achieves excellent coherence, consistency, and fluidity in dynamic motion generation. These findings not only validate the efficacy of the DynaPose4D framework but also indicate its potential applications in the domains of computer vision and animation production.

Index Termsâ 4D Gaussian Splatting, Pose Estimation, Dynamic Content Generation

## 1. INTRODUCTION

Implicit neural rendering techniques have been widely adopted for various tasks, such as pose and shape estimation, novel view synthesis (NVS), and static 3D or dynamic 4D generation . Among these, 4D Gaussian Splatting (4DGS) has pioneered the generation of dynamic scenes from single images or video sequences by enforcing strict temporal consistency across frames [1, 2] [3, 4]. Despite its progress, 4D Gaussian Splatting still faces limitations, particularly in handling temporal sequence dependencies and dynamic changes. Issues arise when dealing with changes in camera perspectives, which lead to insufficient visual consistency and hinder the generation of long-duration, complex 3D motions. Consequently, dynamic scenes generated using these methods may lack coherence, appearing unnatural or inconsistent, which impairs the transition from static to dynamic content.

<!-- image-->  
Fig. 1. Overview of DynaPose4D. From a single image, 3D Gaussians are deformed into 4D content. Temporal consistency is enforced by $\mathcal { L } _ { \mathrm { S C L } }$ , while $\mathcal { L } _ { \mathrm { R e f } } + \mathcal { L } _ { \mathrm { K M I } }$ supervise visual consistency and refine pose transitions.

To address these challenges, we propose the Dyna-Pose4D, which integrates 4D Gaussian Splatting (4DGS) techniques with Category-Agnostic Pose Estimation (CAPE) technology [5]. The framework, shown as Fig. 1, begins by using 3D Gaussian Splatting to construct a 3D model from a single static view. CAPE is then employed to extract pose keypoints, utilizing CNNs for image feature extraction and GCNs for graph-structured data, thereby improving the accuracy of pose pattern detection in images. This allows precise pose keypoints to be extracted from dynamic objects in video sequences, which serve as supervisory signals to ensure smooth transitions between static and dynamic content.

In this research, we make several key contributions:

1. We introduce DynaPose4D, a 4D content generation framework that supports multimodal inputs, including single image and video sequences.

2. Involving the use of advanced pose estimation techniques, DynaPose4D can accurately infer the position of keypoints and their temporal aspects for dynamic objects in any given image or video.

3. We propose a novel method that uses pose keypoints as conditions to guide 4D video generation, ensuring a high level of spatio-temporal consistency of the generated 4D content, while accurately preserving the keypoint trajectories.

Experimental results demonstrate that DynaPose4D achieves significant improvements in dynamic objects generation, both for single images and video sequences. We observe that the performance of the model under dynamic objects is significantly improved by the accurate detection of CAPE [?] [?].

## 2. PRIOR WORK

Generating high-quality 4D dynamic content from a single static image integrates challenges from multiple fields in computer vision, including 3D reconstruction, dynamic scene modeling, and pose estimation. Traditional 3D reconstruction methods, such as Neural Radiance Fields (NeRF) [6], have made significant strides in novel view synthesis by representing scenes as volumetric radiance fields. However, NeRF requires multiple views for training, limiting its effectiveness for single-image reconstruction. Zero-1-to-3 [7]addresses this issue by leveraging large-scale diffusion models to predict novel views from a single image but is restricted to static scenes, lacking the ability to model temporal dynamics.

In dynamic scene generation and 4D modeling, methods like Dynamic NeRFs and 4D Gaussian Splatting (4DGS) extend NeRF to incorporate time as an additional input, allowing for the modeling of temporal changes. While these methods enable dynamic scene generation, they still rely on multiview and multi-frame data for training. Even with advancements like DreamGaussian4D, which enforces temporal coherence, accurately capturing dynamic geometry and temporal dependencies remains challenging, particularly when dealing with varying camera perspectives, resulting in artifacts and inconsistencies in generated content.

Pose estimation plays a critical role in understanding and generating motion within dynamic scenes. Techniques like OpenPose [8] and SMPL [9] have laid the groundwork for 2D and 3D pose estimation, while category-agnostic frameworks like PoseAnything have improved keypoint localization across diverse object categories. Generative models, such as Text2Video-Zero [10], utilize pose information to guide video generation, yet they often depend on multi-view or pose data inputs during inference, leaving gaps in their ability to generate dynamic 4D content directly from single images.

## 3. METHOD

Here, we introduce our proposed dynamic 4D Gaussian Splatting generation process and a pose-supervision loss designed to enforce visual consistency during training. First, we review the process of generating 3D models from a single image, followed by an analysis of the transition from static 3D models to dynamic 4D content. Finally, we present a pose-supervision loss to ensure high-quality and coherent motion generation.

<!-- image-->  
Fig. 2. KML minimizes MSE between poses extracted from rendered and predicted frames under one-shot support.

## 3.1. Neural Rendering for 4D Content

Static 3D generation from single images. For the image-to-3D task, we employ the Zero-1-to-3 model, which uses a single RGB image to generate novel 3D viewpoints by leveraging geometric priors learned from large-scale diffusion models. The model predicts different perspectives of the object by conditioning on camera transformations, enabling highquality 3D reconstructions from a single view.

Image-to-video diffusion for dynamic prior. We employed the Stable Video Diffusion (SVD) model [11] to generate the transformation process from a single input image to a driving video. SVD is a diffusion-based generative model initially used to create high-quality static images. To extend this model for video generation tasks, we leveraged its temporal dependency by inputting a single image and introducing random noise Ïµ, generating a time-sequence video. This process can be described by the following equation:

$$
\left\{ I ^ { \mathrm { R e f } } \right\} _ { \tau = 1 } ^ { T } = f _ { \psi } \left( \epsilon ; I ^ { \mathrm { S u p } } \right) ,\tag{1}
$$

where $I ^ { \mathrm { S u p } }$ represents the input image, $\left\{ I ^ { \mathrm { R e f } } \right\} _ { \tau = 1 } ^ { T }$ is the generated video sequence, Ïµ is the random noise, $f _ { \psi }$ is the imageto-video diffusion model, and $T$ is the number of time steps in the video. Through this model, we can generate a driving video containing dynamic motion information from a single input image.

Dynamic 4DGS generation. Next, we use 4D Gaussian Splatting (4DGS) to extend the static 3D model into dynamic 4D content [12] [13]. 4DGS explicitly models dynamic changes in both spatial and temporal dimension, predicting spatial position, rotation, and scaling variations while ensuring temporal consistency of the generated content. The deformation process can be formulated as:

$$
S _ { \tau } ^ { \prime } = \phi ( S , \tau ) ,\tag{2}
$$

where $\phi$ is the deformation network, $S$ represents the static 3D Gaussians, Ï denotes the time step, and $S _ { \tau } ^ { \prime }$ is the deformed

3D Gaussians at time Ï . To further enhance the quality of the generated dynamic content, we take Score Distillation Sampling (SDS) method for efficient initialization, while optimize the deformation process by minimizing the mean squared error (MSE) between the rendered result of 4DGS and the reference frames from SVD:

$$
\mathcal { L } _ { \mathrm { R e f } } = \frac { 1 } { T } \sum _ { \tau = 1 } ^ { T } \left. \boldsymbol { f } \left( \boldsymbol { S } _ { \tau } ^ { \prime } , \boldsymbol { o } ^ { \mathrm { R e f } } \right) - \boldsymbol { I } _ { \tau } ^ { \mathrm { R e f } } \right. _ { 2 } ^ { 2 } ,\tag{3}
$$

where $I _ { \tau } ^ { \mathrm { R e f } } \in \left\{ I ^ { \mathrm { R e f } } \right\} _ { \tau = 1 } ^ { T }$ is the reference image at time $\tau ,$ $\mathcal { L } _ { \mathrm { R e f } }$ is the MSE loss for the reference viewpoint ${ \cal O } _ { \mathrm { R e f } }$ which aligns to $I _ { \tau } ^ { \mathrm { R e f } }$ , T is the number of time steps, and $f \left( S _ { \tau } ^ { \prime } , o ^ { \mathrm { R e f } } \right)$ is the rendering function of the deformed Gaussians.

## 3.2. Pose Alignment Loss

To further enhance the quality and coherence of the generated motion while ensuring alignment with the input keypoint poses, we introduce a pose supervision losscite [14, 15] [16, 17], illustrated as Fig. 2. This loss guides the dynamic transformation of the 3D Gaussians, ensuring the generated motion achieves high spatio-temporal consistency and matches the pose keypoints in the driving video frames. Specifically, given the input static image and itâs pose keypoints, we use PoseAnything [18, 19] to predict pose keypoints from each $I _ { \tau } ^ { \mathrm { R e f } }$ and $f ( S _ { \tau } ^ { \bar { \prime } } , o ^ { \mathrm { R e f } } )$ with support of one-shot pose keypoints in $I ^ { S u p }$ , denote as $\mathbf { \chi } _ { p _ { \tau } } ^ { \prime } \in \mathbb { R } ^ { \mathbf { \grave { N } } , 2 }$ and $\hat { p } _ { \tau } \in \mathbb { R } ^ { N , \hat { 2 } }$ , respectively. The Keypoint Match Loss (KML) can be formulated as:

$$
\mathcal { L } _ { \mathrm { K M L } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { \tau = 1 } ^ { T - 1 } \Vert p _ { \tau } ^ { i } - \hat { p } _ { \tau } ^ { i } \Vert ^ { 2 }\tag{4}
$$

Furthermore, to ensure the origins of the generated 3D Gaussians $\mathcal { P } _ { \tau } \in \mathbb { R } ^ { M , 3 }$ maintains smoothness and temporal consistency, we define a Spatio-temporal Consistency Loss (SCL) [20] [21]. This loss prevents abrupt changes in the origins of the Gaussians between consecutive time steps.

$$
\mathcal { L } _ { \mathrm { S C L } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { \tau = 1 } ^ { T - 1 } \Vert \mathcal { P } _ { \tau + 1 } ^ { i } - \mathcal { P } _ { \tau } ^ { i } \Vert ^ { 2 }\tag{5}
$$

Here, N indicate the total number of keypoints, while M represents the total number of 3D Gaussians. In summary, we define the Pose Alignment Loss (PAL) as the weighted sum of KML and SCL.

## 4. EXPERIMENTS

We conducted extensive experiments to evaluate the effectiveness of the proposed DynaPose4D framework in generating high-quality 4D dynamic content from single images. All experiments were performed on a single NVIDIA RTX 3090 GPU with 24 GB of memory. Implementation Details

We utilized the open-source repository DreamGaussian4D [?] as the base framework for 4D Gaussian Splatting to generate dynamic 3D models. For pose supervision, we employed PoseAnything, where $N = 1 4$ , to infer subsequent dynamic pose keypoints from a single static image. This provided supervisory signals to enhance motion consistency and temporal coherence in the generated 4D content. To optimize the deformation process and further enhance the dynamic content of the motion, we used the Mean Squared Error (MSE) loss. We ran 500 iterations with a batch size of 16 to ensure model stability during the later stages of training. We initialized $M = 5 1 2$ control Gaussians uniformly within a sphere at a fixed radius of 2. The azimuth angles were sampled uniformly in the range [â180â¦, 180â¦]. We used consistent scaling parameters throughout the training process. The parameter $T _ { \mathrm { m a x } }$ was linearly decayed from 0.98 to 0.02 over the iterations to facilitate smooth deformation.

## 4.1. Evaluation Metrics

To assess the quality of the generated 4D dynamic content, we conducted evaluations on two publicly available datasets: Consistent4D [22] and Animate124 [23]. We employed three widely used metrics to quantitatively evaluate the results: Peak Signal-to-Noise Ratio (PSNR): [24] Measures the pixel-wise differences between the generated images and ground truth, with higher values indicating better quality. Structural Similarity Index Measure (SSIM): [25] Assesses the structural similarity and perceptual quality between images, focusing on luminance, contrast, and structure. Learned Perceptual Image Patch Similarity (LPIPS): [26] Utilizes deep neural network features to evaluate perceptual differences, particularly effective for assessing the coherence and visual quality of dynamic content.

## 4.2. Comparative Analysis

To validate the effectiveness of the proposed DynaPose4D framework, we compared our method with two state-ofthe-art open-source methods: DreamGaussian4D [27] and SC4D [28]. All methods were trained on the Consistent4D and Animate124 datasets using their official codes and default settings. Table 1 summarizes the quantitative results of

Table 1. Quantitative comparison of different methods on Consistent4D and Animate124 datasets.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>DreamGaussian4D</td><td>18.980</td><td>0.797</td><td>0.206</td></tr><tr><td>SC4D</td><td>18.164</td><td>0.805</td><td>0.209</td></tr><tr><td>DynaPose4D</td><td>22.761</td><td>0.863</td><td>0.122</td></tr></table>

the comparison. DynaPose4D achieves superior performance across all metrics, demonstrating significant improvements in both fidelity and perceptual quality.

<!-- image-->  
Fig. 3. visual comparisons among different methods. DynaPose4D produces more coherent and realistic dynamic content, with better temporal consistency and motion smoothness.

## 4.3. Ablation Study

To assess the contribution of the pose supervision component in DynaPose4D, we conducted an ablation study by removing the pose supervision, denoted as DynaPose4D w/o pose supervision. We evaluated the modelâs performance under this configuration to quantify the impact of pose supervision on the quality of the generated content.

Table 2. Ablation study on the impact of pose supervision.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>+Support*</td><td>43.0244</td><td>0.9981</td><td>0.0039</td></tr><tr><td>-Support*</td><td>40.5138</td><td>0.9956</td><td>0.0048</td></tr></table>

Table 2 reports the quantitative results of the ablation study. The results indicate that removing the pose supervision mechanism leads to a decrease in performance, particularly in terms of temporal coherence and motion smoothness in long-duration dynamic content. This demonstrates that pose supervision provides critical guidance for the model to capture temporal variations and maintain spatial consistency.

<!-- image-->  
Fig. 4. Result of ablation studies. Pose keypoint supervision enhances 4D visual consistency and yields 3D geometry better aligned with the source image.

## 4.4. Discussion

Fig. 4 illustrates visual examples from the ablation study. The images show that without pose supervision, the generated content exhibits artifacts, temporal jitter, and spatial inconsistencies, whereas the proposed DynaPose4D maintains higher-quality and more coherent motion. The experimental results further validate the effectiveness of the framework: by integrating pose estimation with CAPE technology, our method directly tackles the challenge of generating spatiotemporally consistent 4D content from single images. In particular, pose supervision guides the model to infer temporal dynamics more accurately and to preserve fine-grained spatial structure even in challenging scenes with complex motions or self-occlusion. These improvements are not only visible qualitatively but are also supported by quantitative metrics, demonstrating that DynaPose4D achieves more stable and reliable results. Overall, the ablation study highlights that pose supervision is not merely an auxiliary component but a fundamental factor in ensuring robustness and generalization.

## 5. CONCLUSION

This paper presents DynaPose4D, a framework that generates high-quality 4D dynamic content from a single image by integrating 4D Gaussian Splatting with pose supervision for spatio-temporal consistency. Experiments show clear improvements over state-of-the-art methods in PSNR, SSIM, and LPIPS, confirming both visual fidelity and quantitative gains. Ablation studies highlight pose supervision as crucial, since its removal degrades temporal coherence and motion smoothness. Overall, DynaPose4D effectively captures dynamic changes while preserving spatial consistency, offering a robust solution for challenging scenarios and strong potential for applications such as animation, AR/VR content creation, and motion-driven 3D reconstruction, while also opening opportunities for future research on spatio-temporal generative modeling.

## 6. REFERENCES

[1] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang, âNeural scene flow fields for space-time view synthesis of dynamic scenes,â 2021.

[2] Jusheng Zhang, Yijia Fan, Wenjun Lin, Ruiqi Chen, Haoyi Jiang, Wenhao Chai, Jian Wang, and Keze Wang, âGam-agent: Game-theoretic and uncertainty-aware collaboration for complex visual reasoning,â 2025.

[3] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely, âDynibar: Neural dynamic image-based rendering,â 2023.

[4] Jusheng Zhang, Zimeng Huang, Yijia Fan, Ningyuan Liu, Mingyan Li, Zhuojie Yang, Jiawei Yao, Jian Wang, and Keze Wang, âKabb: Knowledge-aware bayesian bandits for dynamic expert coordination in multi-agent systems,â 2025.

[5] Kaixin Xiong, Shi Gong, Xiaoqing Ye, Xiao Tan, Ji Wan, Errui Ding, Jingdong Wang, and Xiang Bai, âCape: Camera view position embedding for multi-view 3d object detection,â 2023.

[6] Ben Mildenhall, Pratul P. Srinivasan, and Matthew Tancik, âNerf: Representing scenes as neural radiance fields for view synthesis,â 2020.

[7] Ruoshi Liu, Rundi Wu, and Basile Van Hoorick, âZero-1-to-3: Zero-shot one image to 3d object,â 2023.

[8] Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh, âOpenpose: Realtime multi-person 2d pose estimation using part affinity fields,â 2019.

[9] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black, âSMPL: A skinned multi-person linear model,â ACM Trans. Graphics (Proc. SIGGRAPH Asia), vol. 34, no. 6, pp. 248:1â248:16, Oct. 2015.

[10] Levon Khachatryan and Andranik Movsisyan, âText2video-zero: Text-to-image diffusion models are zero-shot video generators,â 2023.

[11] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, and Yam Levi, âStable video diffusion: Scaling latent video diffusion models to large datasets,â 2023.

[12] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer, âD-nerf: Neural radiance fields for dynamic scenes,â 2020.

[13] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman,

Ricardo Martin-Brualla, and Steven M. Seitz, âHypernerf: A higher-dimensional representation for topologically varying neural radiance fields,â 2021.

[14] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh, âRealtime multi-person 2d pose estimation using part affinity fields,â 2017.

[15] Jusheng Zhang, Kaitong Cai, Yijia Fan, Ningyuan Liu, and Keze Wang, âMat-agent: Adaptive multi-agent training optimization,â 2025.

[16] Hao-Shu Fang, Shuqin Xie, Yu-Wing Tai, and Cewu Lu, âRmpe: Regional multi-person pose estimation,â 2018.

[17] Jusheng Zhang, Kaitong Cai, Yijia Fan, Jian Wang, and Keze Wang, âCf-vlm:counterfactual vision-language fine-tuning,â 2025.

[18] Or Hirschorn and Shai Avidan, A Graph-Based Approach for Category-Agnostic Pose Estimation, p. 469â485, Springer Nature Switzerland, Nov. 2024.

[19] Jusheng Zhang, Kaitong Cai, Qinglin Zeng, Ningyuan Liu, Stephen Fan, Ziliang Chen, and Keze Wang, âFailure-driven workflow refinement,â 2025.

[20] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz, âMocogan: Decomposing motion and content for video generation,â 2017.

[21] Jusheng Zhang, Kaitong Cai, Jing Yang, and Keze Wang, âLearning dynamics of vlm finetuning,â 2025.

[22] Yanqin Jiang and Li Zhang, âConsistent4d: Consistent 360Â° dynamic object generation from monocular video,â in The Twelfth International Conference on Learning Representations, 2024.

[23] Yuyang Zhao and Zhiwen Yan, âAnimate124: Animating one image to 4d dynamic scene,â 2024.

[24] Q. Huynh-Thu and M. Ghanbari, âScope of validity of psnr in image/video quality assessment,â Electronics Letters, vol. 44, pp. 800â801, 2008.

[25] Zhou Wang, Alan Bovik, Hamid Sheikh, and Eero Simoncelli, âImage quality assessment: From error visibility to structural similarity,â Image Processing, IEEE Transactions on, vol. 13, pp. 600 â 612, 05 2004.

[26] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â 2018.

[27] Jiawei Ren, Liang Pan, and Jiaxiang Tang, âDreamgaussian4d: Generative 4d gaussian splatting,â 2024.

[28] Zijie Wu, Chaohui Yu, and Yanqin Jiang, âSc4d: Sparse-controlled video-to-4d generation and motion transfer,â 2024.