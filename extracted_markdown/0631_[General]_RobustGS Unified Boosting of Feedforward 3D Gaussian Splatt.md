# RobustGS: Unified Boosting of Feedforward 3D Gaussian Splatting under Low-Quality Conditions

Anran Wu1\*, Long Peng1\*â , Xin Di1\*, Xueyuan Dai2, Chen Wu1, Yang Wang1,2â¡, Xueyang Fu1, Yang Cao1, Zheng-Jun Zha1

1University of Science and Technology of China, 2Changâan University {wuar, longp2001, dx9826}@mail.ustc.edu.cn, ywang120@ustc.edu.cn https://github.com/wuanran678/RobustGS

## Abstract

Feedforward 3D Gaussian Splatting (3DGS) overcomes the limitations of optimization-based 3DGS by enabling fast and high-quality reconstruction without the need for per-scene optimization. However, existing feedforward approaches typically assume that input multi-view images are clean and high-quality. In real-world scenarios, images are often captured under challenging conditions such as noise, low light, or rain, resulting in inaccurate geometry and degraded 3D reconstruction. To address these challenges, we propose a general and efficient multi-view feature enhancement module, RobustGS, which substantially improves the robustness of feedforward 3DGS methods under various adverse imaging conditions, enabling high-quality 3D reconstruction. The RobustGS module can be seamlessly integrated into existing pretrained pipelines in a plug-and-play manner to enhance reconstruction robustness. Specifically, we introduce a novel component, Generalized Degradation Learner, designed to extract generic representations and distributions of multiple degradations from multi-view inputs, thereby enhancing degradation-awareness and improving the overall quality of 3D reconstruction. In addition, we propose a novel semantic-aware state-space model. It first leverages the extracted degradation representations to enhance corrupted inputs in the feature space. Then, it employs a semantic-aware strategy to aggregate semantically similar information across different views, enabling the extraction of fine-grained crossview correspondences and further improving the quality of 3D representations. Extensive experiments demonstrate that our approach, when integrated into existing methods in a plug-and-play manner, consistently achieves state-of-the-art reconstruction quality across various types of degradations.

## Introduction

Recent progress in 3D reconstruction has seen a shift from neural implicit methods like NeRF to explicit representations such as 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023). 3DGS significantly advances the field by offering higher reconstruction fidelity and more efficient rendering compared to NeRF (Mildenhall et al. 2021), thanks to its explicit and compact modeling of scene geometry and appearance. To improve scalability and speed, recent feedforward variants (Szymanowicz, Rupprecht, and Vedaldi 2024; Tang et al. 2024; Charatan et al. 2024; Chen et al. 2024c; Smart et al. 2024; Ye et al. 2024) leverage neural networks to generate 3D Gaussian representations from a set of input images in a single pass, bypassing the need for per-scene optimization. However, these methods are typically trained and evaluated on curated datasets with clean, well-lit, and artifact-free images. In practical scenarios, input images often exhibit various degradations, such as noise, adverse environments, or challenging lighting conditions. Such degradations severely impact feature extraction and matching, leading to incomplete geometry, loss of detail, or distorted reconstructions (Riegler and Koltun 2020; Liu et al. 2023; Catley-Chandar et al. 2024; Wu et al. 2024b; Kwon et al. 2025). Addressing these challenges is crucial for robust 3D reconstruction in unconstrained environments.

<!-- image-->  
Figure 1: (a) Illustration of the proposed plug-and-play RobustGS integrated into the existing feedforward 3DGS pipeline to enhance robustness. (b) Our proposed RobustGS significantly outperforms existing methods in both visual quality and efficiency evaluations.

A straightforward solution is to apply image restoration techniques (Chee and Wu 2018; Liang et al. 2021; Su, Xu, and Yin 2022) to pre-process degraded inputs or to enhance rendered novel views. However, most restoration networks are designed to optimize perceptual quality in the image space, rather than ensuring geometric consistency across multiple views. This mismatch in objectives often results in artifacts and inconsistencies in the reconstructed 3D structure, ultimately undermining the fidelity and integrity of the scene representation (Philip et al. 2019; Jain, Tancik, and Abbeel 2021; Zhou and Tulsiani 2023; Chen et al. 2024a). Furthermore, several recent studies have explored adapting 3DGS to handle degraded input (Li et al. 2024e; Chen and Liu 2024; Cui, Chu, and Harada 2025). For example, WeatherGS (Qian et al. 2024) focuses on scenes affected by rain and snow, while DehazeGS (Ma, Zhao, and Chen 2025) only addresses hazy environments. Although effective in certain cases, these methods rely on per-scene optimization and are tailored to limited degradation types, restricting their generalization and scalability in scenarios with diverse and unknown degradations.

To address these challenges, we propose to enhance feature representations to improve the robustness of feedforward 3DGS methods under diverse and unknown degradations. Specifically, we introduce a plug-and-play, generalpurpose feature enhancement module that can be seamlessly integrated into existing methods, as illustrated in Figure 1(a). The proposed approach comprises two key components: a Generalized Degradation Learner and a Multi-View State-Space Enhancement Module. In particular, the Generalized Degradation Learner module is designed to extract generic representations and distributions of multiple degradations from multi-view inputs, thereby enhancing degradation-awareness and improving the overall quality of 3D reconstruction. The extracted degradation representations are then used to guide the Multi-View State-Space Enhancement Module, which enhances feature-level interactions across views through a semantic-aware state-space mechanism. This design enables the model to selectively propagate and aggregate semantically relevant features between views, thus improving the consistency and granularity of 3D reconstructions. By preserving the original 3DGS pipeline and eliminating the need for retraining, our method provides a simple yet powerful solution for robust feedforward 3D reconstruction under challenging and adverse conditions, as illustrated in Figure 1(b). Our main contributions are as follows:

â¢ We propose a novel Generalized Degradation Learner module that generically extracts various degradation representations from input images, thereby enhancing the networkâs degradation-awareness and robustness.

â¢ We introduce a novel degradation-guided feature enhancement mechanism by injecting the degradation representations extracted by Generalized Degradation Learner into the existing state-space model, enabling efficient and targeted enhancement of low-quality regions in the feature space.

â¢ We design a semantic-aware multi-view interaction module, Multi-View State-Space Enhancement Module, which aggregates features across different views based on semantic consistency, enabling fine-grained alignment and cross-view interaction in semantically similar regions.

â¢ We implement a plug-and-play modular design, RobustGS, that can be directly integrated into existing feedforward 3DGS methods. Our approach consistently improves reconstruction quality under six typical degradation types without requiring retraining of the original reconstruction pipeline.

## Related Work

## feedforward 3DGS

The advent of 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) has marked a major step forward in efficient 3D reconstruction and real-time novel view synthesis (Lyu et al. 2024; Chen, Zhou, and Li 2024; Yang et al. 2024; Lu et al. 2024; Chen et al. 2025). Compared to NeRF-based approaches (Barron et al. 2021; Muller et al. 2022; Gao et al. Â¨ 2022; Chen et al. 2023; Huang et al. 2023), 3DGS offers significant advantages in rendering speed and memory efficiency, making it highly suitable for interactive applications (Kim and Lee 2024; Liu et al. 2025; Khalid, Ibrahim, and Liu 2025; Gao et al. 2025). Nevertheless, the original 3DGS and many of its extensions (Fei et al. 2024; Wu et al. 2024a; Dalal et al. 2024) still rely on a large number of highquality input images and require time-consuming optimization to achieve accurate geometry and appearance. To overcome these limitations, several feedforward pipelines have recently been proposed. For example, PixelSplat (Charatan et al. 2024) leverages epipolar geometry to refine depth, enabling high-quality reconstruction from sparse views. No-PosSplat (Ye et al. 2024) further eliminates the need for camera poses and depth maps by exploiting cross-view feature correspondences, increasing flexibility in unconstrained scenarios. However, these methods typically assume clean, artifact-free multi-view inputs. In practice, images captured in the wild frequently suffer from degradations such as noise, blur, low light, or adverse weather, making the predicted Gaussian primitives unreliable and leading to degraded geometry, inconsistent colors, and visible artifacts in novel views. Despite these challenges, the robustness of feedforward 3DGS pipelines under degradations remains largely unexplored, significantly restricting their use in unconstrained environments.

## Image Restoration

Low-level vision focuses on restoring degraded visual content and enhancing perceptual quality by removing noise, artifacts, or distortions (Li et al. 2024b, 2025b; Peng et al. 2021; Li et al. 2023a; Ren et al. 2024b; Yan et al. 2025; Zhu, Li, and Fu 2025). It covers a wide range of tasks such as denoising, deblurring, deraining, and super-resolution, typically operating on individual images (Peng et al. 2020; Wang et al. 2023b; Peng et al. 2024b,d; Wang et al. 2023a; Xiao and Xiong 2025; Li et al. 2024a, 2025c,a; Li and Fu 2025). Over the years, methods have evolved from convolutionbased networks to transformer-based models (Yi et al. 2021a,b; Peng et al. 2024a; Conde et al. 2024; Li et al. 2024f; He et al. 2024a), with increasing capacity to handle diverse and complex degradations (Peng et al. 2025a; Yi et al. 2025; He et al. 2024b, 2025c,b,d). As the field progresses (Peng et al. 2025c,b; Xiao and Wang 2025; Li et al. 2025e; He et al. 2024c; Qi et al. 2025; Feng et al. 2025; Ren et al. 2021; Xia et al. 2024; Li et al. 2023b; Ren et al. 2024a; Wang et al. 2025), recent methods have begun to consider more realistic degradation scenarios and adaptive strategies (He et al. 2023; Ren et al. 2025b; Li et al. 2025d; Zhao et al. 2024; Peng et al. 2024c; He et al. 2025e,a; Zheng et al. 2024; Jin et al. 2024a; Sun et al. 2024), such as blind restoration and degradation-aware modeling (Ren et al. 2025a, 2023; Xiao et al. 2024; Zheng et al. 2023; Di et al. 2025; Ren et al. 2022; Du et al. 2024). Despite these advances, most approaches remain centered on single-view inputs and aim to optimize visual fidelity through pixel-wise or perceptual losses (Li et al. 2024d; Zhao et al. 2023; Zheng et al. 2022; Gong et al. 2024; Zeng, Gong, and Jiang 2024). These pixel-level objectives, while effective for image enhancement (Li et al. 2024c; Zheng et al. 2025a,b; Pan et al. 2025; Wu et al. 2025; Jiang et al. 2024; Ignatov et al. 2025; Gong, Huang, and Chen 2022; Lin et al. 2025a), may overlook deeper structural or semantic consistency crucial for downstream tasks like 3D reconstruction.

## 3DGS in Degraded Scenes

Most 3DGS methods are developed on clean images, while real-world scenes often suffer from degradations like rain, fog, or low light. Some recent approaches address these challenges through per-scene optimization (Jin et al. 2024b; Qiao et al. 2025; Bui et al. 2025), such as SRGS (Feng et al. 2024) for super-resolution, Deblurring 3DGS (Lee et al. 2024) for deblurring, DehazeGS (Ma, Zhao, and Chen 2025) for dehazing, and WeatherGS (Qian et al. 2024) for adverse weather. However, these methods are limited to specific degradation types and require slow optimization for each scene. Although HQGS (Lin et al. 2025b) extends to multiple degradations, it still relies on a sparse point cloud estimated by COLMAP. To our knowledge, no prior work has explored feature-level enhancement for robust, feedforward 3DGS under diverse and unknown degradations. Our work is the first to address this gap.

## Proposed Method

To enhance the robustness of feedforward 3DGS under degraded scenes, we propose a plug-and-play, unified feature enhancement framework, RobustGS, that can be seamlessly integrated into existing reconstruction and rendering pipelines. Unlike methods tailored to specific degradation types, our approach offers a unified solution capable of simultaneously handles diverse degradations such as rain, snow, noise and low-light. Our method consists of two main stages. Firstly, we propose a Generalized Degradation Learner that extracts compact and generalizable degradation representations from input images. Secondly, we design a Multi-View State-Space Enhancement Module to leverage degradation representations for suppressing input artifacts while modeling semantic consistency and structural correspondence across views for feature enhancement. This design significantly enhances the recovery of degraded regions by leveraging both degradation cues and cross-view dependencies. As shown in Figure 3(Left), the overall framework is efficient and broadly applicable to feedforward 3D reconstruction methods, leading to improved performance under a wide range of degradation conditions.

<!-- image-->  
Figure 2: The carefully designed training pipeline for the Generalized Degradation Learner incorporates various supervision signals to fully extract the distribution and types of degradation signals, laying the foundation for our RobustGS.

## Unified Degradation Representation Learning

To enable robust all-in-one enhancement across diverse degradation types in the feature space, it is essential for the model to perceive the degradation characteristics present in the input. Therefore, we propose a Generalized Degradation Learner (GenDeg) that distills high-level degradation information into a compact latent, as shown in Figure 2. This implicit representation provides a global degradation context to guide subsequent feature enhancement.

GenDeg is implemented as a neural degradation-aware encoder, which processes the degraded image $I _ { \mathrm { d e g } }$ to generate a degradation embedding $z _ { \mathrm { d e g } } ~ \in ~ \mathbb { R } ^ { C \times 1 }$ $z _ { \mathrm { d e g } }$ captures degradation-specific cues to enable condition-aware enhancement in downstream stages. To train GenDeg effectively, we design an auxiliary reconstruction pipeline. Specifically, a Content Encoder processes the clean counterpart $I _ { \mathrm { c l e a n } }$ to produce content features, which are then fused with $z _ { \mathrm { d e g } }$ via a Content Decoder to reconstruct the original degraded image $\hat { I } _ { \mathrm { d e g } }$ . This training mechanism enables ${ z } _ { \mathrm { d e g } }$ to be learned in a self-supervised pattern, such that it carries sufficient degradation cues to reconstruct the observed corrupted input when combined with clean content.

Instead of relying on explicit degradation labels, the degradation embedding ${ z } _ { \mathrm { d e g } }$ is learned as an implicit representation, capturing underlying degradation characteristics directly from the pixel distribution. This latent representation encapsulates high-level degradation semantics (e.g., rain, noise, fog) in a latent space, which enables generalization across degradation types. To encourage the embedding to capture degradation-relevant features, we optimize the module with two complementary objectives:

<!-- image-->  
Figure 3: Pipeline and key components of the proposed RobustGS framework. Left: The overall pipeline, in which RobustGS is integrated into the standard feedforward 3DGS architecture. (a) The architecture of Multi-View State-Space Enhancement Module (MV-SSEM), consisting of multiple Feature Enhancement Blocks (FEBs). (b) Multi-view semantic guidance module, where semantic prompts are extracted across views and spatial tokens are reordered accordingly. (c) Feature enhancement via the State-Space Module (SSM), jointly guided by degradation embeddings and semantic prompts.

Reconstruction Loss. To ensure that the degradation embedding $z _ { \mathrm { d e g } }$ effectively captures the intrinsic degradation characteristics, we train the decoder to reconstruct the original degraded image. The supervision combines both pixelwise and perceptual losses:

$$
\mathcal { L } _ { \mathrm { r e c } } = \lambda \Vert \hat { I } _ { \mathrm { d e g } } - I _ { \mathrm { d e g } } \Vert _ { 1 } + \mathcal { L } _ { \mathrm { p e r c } } ( \hat { I } _ { \mathrm { d e g } } , I _ { \mathrm { d e g } } ) .\tag{1}
$$

where $\mathcal { L } _ { \mathrm { p e r c } }$ is a perceptual loss computed from a pre-trained VGG network and Î» is empirically set to 0.1. The $\ell _ { 1 }$ term enforces pixel-level fidelity, while the perceptual term emphasizes high-level semantic and structural similarity. By minimizing this reconstruction loss, we indirectly encourage GenDeg to extract embeddings that faithfully encapsulate degradation-specific cues, since only an informative $z _ { \mathrm { d e g } }$ can guide the decoder to synthesize a visually and semantically consistent degraded image.

Contrastive Loss. To enhance the discriminability of the embedding space, we adopt a contrastive objective:

$$
\mathcal { L } _ { \mathrm { c o n } } = - \log \frac { \exp ( \sin ( z _ { i } , z _ { j } ) / \tau ) } { \sum _ { k \neq i } \exp ( \sin ( z _ { i } , z _ { k } ) / \tau ) } .\tag{2}
$$

where sim(Â·, Â·) denotes cosine similarity, and $( z _ { i } , z _ { j } )$ is a positive pair with the same degradation type. This encourages embeddings of similar degradations to be closer while pushing dissimilar types apart.

Classification Loss. To further regularize the learning of $z _ { \mathrm { d e g } }$ , we attach a lightweight classifier after the embedding to predict the degradation type. Although not used during inference, this auxiliary operation encourages the embedding to capture more explicit degradation cues. The supervision is provided by a cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { c l s } } = \mathrm { C r o s s E n t r o p y } ( f _ { \mathrm { c l s } } ( z _ { \mathrm { d e g } } ) , y _ { \mathrm { d e g } } ) .\tag{3}
$$

The overall training objective is defined as ${ \mathcal { L } } = \lambda _ { 1 } { \mathcal { L } } _ { \mathrm { r e c } } +$ $\lambda _ { 2 } \mathcal { L } _ { \mathrm { c o n } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { c l s } } .$ , with the weights are empirically set to $\lambda _ { 1 } = 1 . 0 , \lambda _ { 2 } = 0 . 5$ , and $\lambda _ { 3 } = 0 . { \overset { - } { 3 } }$

Through the joint optimization of $\mathcal { L } _ { \mathrm { r e c } } , \mathcal { L } _ { \mathrm { c o n } } .$ , and ${ \mathcal { L } } _ { \mathrm { c l s } } .$ the embedding module learns to represent degradation in a way that is reconstructive, discriminative, and semantically aligned with degradation categories. This degradation-aware signal serves as a high-level conditioning cue for featurelevel enhancement in downstream modules.

## Multi-view Feature Enhancement

To address feature degradation in the feedforward 3DGS pipeline, we introduce Multi-View State-Space Enhancement Module. Instead of modifying the 3DGS backbone, we design a standalone, plug-and-play enhancement block, as shown in Figure 3(a). Our method aims to enhance the quality of intermediate features extracted from degraded multi-view images, thereby providing cleaner and more reliable features for downstream 3D reconstruction. Specifically, degradation embeddings extracted from GenDeg are used to explicitly guide the feature enhancement process. By integrating these degradation priors into a carefully designed multi-scale framework, the network becomes aware of the type and severity of degradation at different spatial resolutions, enabling more effective feature enhancement.

<table><tr><td rowspan="2"></td><td colspan="3">Brightness</td><td colspan="3">Fog</td><td colspan="3">Contrast</td><td colspan="3">Snow</td></tr><tr><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR SSIM</td><td></td><td>LPIPS</td><td>| PSNR SSIM</td><td></td><td>LPIPS PSNR</td><td>SSIM</td><td></td><td>LPIPS</td></tr><tr><td rowspan="13">Pixelsplat PromptIR PromptIR PromptIR AdaIR</td><td>15.03</td><td>0.7919</td><td>0.1753</td><td>14.88</td><td>0.7321</td><td>0.2044</td><td>18.40</td><td>0.7195</td><td>0.2453</td><td>19.15</td><td>0.5809</td><td>0.4297</td></tr><tr><td>14.31</td><td>0.7664</td><td>0.1797</td><td>21.50</td><td>0.8160</td><td>0.2043</td><td>18.30</td><td>0.7161</td><td>0.2440</td><td>19.87</td><td>0.6151</td><td>0.4089</td></tr><tr><td>16.99</td><td>0.7811</td><td>0.1897</td><td>21.33</td><td>0.8100</td><td>0.2031</td><td>21.80</td><td>0.8047</td><td>0.2197</td><td>21.46</td><td>0.6794</td><td>0.3844</td></tr><tr><td>23.91</td><td>0.8431</td><td>0.1645</td><td>20.35</td><td>0.8034</td><td>0.2145</td><td>16.29</td><td>0.6770</td><td>0.2557</td><td>17.13</td><td>0.6085</td><td>0.4404</td></tr><tr><td>14.78</td><td>0.7817</td><td>0.1775</td><td>20.91</td><td>0.8146</td><td>0.1994</td><td>18.37</td><td>0.7197</td><td>0.2457</td><td>19.72</td><td>0.6151</td><td>0.4049</td></tr><tr><td>AdaIR 23.59</td><td>0.8501</td><td>0.1527</td><td>15.40</td><td>0.7115</td><td>0.2709</td><td>16.88</td><td>0.7064</td><td>0.2315</td><td>16.29</td><td>0.5908</td><td>0.4458</td></tr><tr><td>AdaIR 23.45</td><td>0.8493</td><td>0.1631</td><td>15.54</td><td>0.6018</td><td>0.4518</td><td>15.37</td><td>0.6750</td><td>0.2560</td><td>15.54</td><td>0.6018</td><td>0.4518</td></tr><tr><td>RobustGS 24.87</td><td>0.8512</td><td>0.1617</td><td>21.52</td><td>0.8209</td><td>0.1974</td><td>22.05</td><td>0.8062</td><td>0.2147</td><td>20.78</td><td>0.6495</td><td>0.4003</td></tr><tr><td rowspan="3"></td><td>Rain</td><td></td><td></td><td></td><td>Impulse noise</td><td></td><td>Average performance</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR SSIM</td><td></td><td>LPIPS</td><td>| PSNR SSIM</td><td></td><td></td><td></td><td>Complexity</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>LPIPS</td><td>FLOPs</td><td>Params</td><td>time(ms)</td></tr><tr><td>Pixelsplat PromptIR</td><td>19.62 21.65</td><td>0.7325 0.7473</td><td>0.3036 0.2921</td><td>14.08 14.33</td><td>0.7578 0.7633</td><td>0.2626 0.2610</td><td>16.86 18.33</td><td>0.7191</td><td>0.2701</td><td></td><td></td><td>|</td></tr><tr><td>PromptIR</td><td>21.88</td><td>0.7826</td><td>0.2662</td><td>20.71</td><td></td><td>0.2424</td><td>20.69</td><td>0.7374</td><td>0.2650</td><td>43.28 G</td><td>35.59 M</td><td>91.68 ms</td></tr><tr><td>PromptIR</td><td>21.30</td><td></td><td></td><td>21.67</td><td>0.8054</td><td>0.2529</td><td>20.11</td><td>0.7772</td><td>0.2509</td><td>43.28 G</td><td>35.59 M</td><td>91.68 ms</td></tr><tr><td>AdaIR</td><td>21.02</td><td>0.7779</td><td>0.2687</td><td>14.37</td><td>0.8109</td><td></td><td>18.20</td><td>0.7535 0.7383</td><td>0.2661</td><td>43.28 G</td><td>35.59 M</td><td>91.68 ms</td></tr><tr><td>AdaIR</td><td>23.06</td><td>0.7383 0.8150</td><td>0.3020 0.2111</td><td>20.37</td><td>0.7604 0.8045</td><td>0.2618 0.2297</td><td>19.26</td><td>0.7464</td><td>0.2652</td><td>40.54 G</td><td>28.78 M</td><td>226.01 ms</td></tr><tr><td>AdaIR</td><td>19.30</td><td>0.7576</td><td>0.2962</td><td>20.66</td><td>0.8013</td><td>0.2570</td><td>18.31</td><td>0.7145</td><td>0.2570 0.3127</td><td>40.54 G 40.54 G</td><td>28.78 M</td><td>226.01 ms</td></tr><tr><td></td><td>22.65</td><td>0.7809</td><td>0.2652</td><td>22.03</td><td>0.8129</td><td>0.2267</td><td>22.32</td><td>0.7869</td><td>0.2443</td><td>20.51 G</td><td>28.78 M</td><td>226.01 ms</td></tr><tr><td>RobustGS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>10.83 M</td><td>49.10ms</td></tr></table>

Table 1: Comparison of performance and computational complexity with existing methods across various degradation scenarios. Bold indicates the best performance, while underline denotes the second-best.

Degradation in images is typically distributed across the entire image. Therefore, effective removal of degradation and feature fusion requires network capable of capturing global context while maintaining computational efficiency. To address this, we adopt a set of State Space Model (SSM) based blocks, which enable more effective representation and enhancement of degraded features through global context modeling. The general form of a continuous-time Linear State Space Model is defined as:

$$
\dot { h } ( t ) = A h ( t ) + B x ( t ) , y ( t ) = C h ( t ) + D x ( t ) ,\tag{4}
$$

where $x ( t )$ is the input signal, h(t) is the hidden state, and $y ( t )$ is the output at time t. The matrices $A , B , C ,$ and D parameterize the system dynamics and output mapping. This formulation captures the influence of both the internal system state and external input on the output in a continuous, recursive manner. To adapt the continuous model to discrete image representations, we discretize the system using matrix exponentials:

$$
{ \overline { { A } } } = \exp ( \Delta A ) , { \overline { { B } } } = ( \Delta A ) ^ { - 1 } ( \exp ( \Delta A ) - I ) \Delta B ,\tag{5}
$$

where $\Delta$ is a learnable step size and I is the identity matrix.The resulting discrete-time SSM is updated as:

$$
h _ { k } = \overline { { { A } } } h _ { k - 1 } + \overline { { { B } } } x _ { k } , y _ { k } = C h _ { k } + D x _ { k } .\tag{6}
$$

This formulation allows directional and position-aware feature modeling by integrating the current input $x _ { k }$ with the memory state $h _ { k - 1 }$ (Guo et al. 2024).

To achieve robust all-in-one feature enhancement under unknown degradations, we introduce a degradation-aware modulation mechanism that dynamically adjusts the selective scan process based on degradation characteristics. Since the degradation type and severity are typically unknown during inference, a unified and learnable guidance signal is required to modulate the enhancement process accordingly. Based on this, we utilize the degradation embedding $z _ { \mathrm { d e g } } ,$ a compact vector extracted from GenDeg, which encodes high-level semantics of the degradation. As shown in Figure 3(c), the degradation embedding primarily modulates the input projection matrix B, which governs how degraded input tokens are integrated into the hidden state. The embedding ${ \mathcal { Z } } _ { \mathrm { d e g } }$ also modulates the output projection C and the update step size â, enabling global degradation-aware adaptation in the selective scan process. Specifically, the degradation embedding is fed into three separate lightweight MLPs (Ï) to generate modulation coefficients, which are then applied via element-wise multiplication:

$$
W _ { \mathrm { m o d } } = W \odot \phi _ { W } ( z _ { \mathrm { d e g } } ) , \quad W \in \{ B , C , \Delta \} .\tag{7}
$$

This degradation-guided modulation enables the selective scan to adapt its dynamic behavior to the degradation characteristics of the input. By conditioning the enhancement parameters on $z _ { \mathrm { d e g } } ,$ the model can restore degraded regions and preserve semantic structures better across diverse scenarios. This design enables unified feature enhancement without requiring degradation-specific tuning, making the module generalizable for different degradation scenes.

<!-- image-->

<!-- image-->  
PixelSplat

<!-- image-->  
PromptIR

<!-- image-->

<!-- image-->

<!-- image-->  
GT

Figure 4: Qualitative comparison. The visual quality of our method outperforms restore before reconstruct methods. Visual examples in the first and second rows correspond to foggy and dark degradation scenes, respectively.
<table><tr><td></td><td>mvsplat</td><td>PromptIR</td><td>PromptIR</td><td>PromptIR</td><td>AdaIR</td><td>AdaIR</td><td>AdaIR</td><td>RobustGS</td></tr><tr><td>Avg PSNR</td><td>17.20</td><td>18.57</td><td>20.04</td><td>19.31</td><td>18.43</td><td>19.58</td><td>18.14</td><td>21.48</td></tr><tr><td>Avg SSIM</td><td>0.7307</td><td>0.7512</td><td>0.7726</td><td>0.7476</td><td>0.7499</td><td>0.7601</td><td>0.7425</td><td>0.7851</td></tr></table>

Table 2: Average Performance Comparison on the MVSplat backbone. More details are presented in Appendix.

To mitigate feature loss caused by degradation, we leverage the complementary information across multi-view images and propose a semantic-aware consistency enhancement mechanism, inspired by the semantic-aware design in MambaIRv2 (Guo et al. 2025). The key motivation is to encourage feature interactions across views by reorganizing tokens based on their latent semantics, as shown in Figure 3(b). Concretely, we first extract the semantic representations from the feature map using a learnable projection network. Given a feature map $\mathbf { \bar { \textit { F } } } \in \ \mathbb { R } ^ { N \times d }$ containing N tokens, we generate semantic embeddings ${ \bf E } =$ $\{ \mathbf { e } _ { 1 } , \mathbf { e } _ { 2 } , \dots , \mathbf { e } _ { N } \}$ , where ${ \bf E } = f _ { \mathrm { s e m } } ( F ) , { \bf e } _ { i }$ represents the semantic feature of token i and $f _ { \mathrm { s e m } } ( \cdot )$ denotes the global semantic encoder. We then introduce a learnable semantic pool $\mathcal { P } ~ = ~ \left\{ \mathbf { p } _ { 1 } , \mathbf { p } _ { 2 } , \ldots , \mathbf { p } _ { K } \right\}$ composed of K semantic prototypes. The similarity between each tokenâs semantic embedding and the prototypes is computed, followed by a Gumbel-Softmax to obtain a discrete semantic assignment:

$$
\mathbf { w } _ { i } = \mathrm { G u m b e l S o f t m a x } ( \sin ( \mathbf { e } _ { i } , \mathcal { P } ) ) .\tag{8}
$$

where ${ \bf w } _ { i } \in \mathbb { R } ^ { K }$ denotes the categorical distribution over semantic classes, and sim(Â·) is a similarity function. Based on this semantic distribution, we reorder all tokens so that those with similar semantics are spatially grouped. This reorganization promotes inter-view interaction by aligning semantically related content, and further alleviates long-range forgetting problems in SSM model.

Moreover, we modulate the scan parameters using semantic-aware signals, as shown in Figure 3(c). Each tokenâs semantic distribution $\mathbf { w } _ { i }$ is used to adaptively adjust the scanning matrix C as follows:

$$
C _ { i } ^ { \mathrm { m o d } } = C + \sum _ { k = 1 } ^ { K } w _ { i k } \cdot \mathbf { p } _ { k } .\tag{9}
$$

where $C _ { i } ^ { \mathrm { m o d } }$ is the modulated scanning weight for token i. This modulation allows the scan operation to become more sensitive to semantic context, enabling a more effective feature enhancement pipeline. To further facilitate consistent information propagation, we introduce a cross-stage feedback mechanism in the decoder: the modulation matrix generated at a lower-resolution stage is passed to the next stage as state information to guide the selective scan. This design helps retain global structural cues from earlier stages and improves the quality of feature reconstruction at higher resolutions.

## Experiment and Analysis

## Experiment Setting

Datasets and implementation details. We first train Generalized Degradation Learner on the DIV2K dataset (Wang et al. 2021), a high-quality image dataset that is widely used for degradation simulation. Following the degradation protocol of RobustSAM (Chen et al. 2024b), we synthesize diverse low-quality variants to enable the learning of robust and generalizable degradation priors. Subsequently, we train Multi-View State-Space Enhancement Module on the

RealEstate10K (RE10K) dataset (Zhou et al. 2018), using the same degradation synthesis protocol in GenDeg stage. RE10K provides multi-view video sequences along with camera poses estimated via structure-from-motion (SfM) using COLMAP (Schonberger and Frahm 2016). For consis- Â¨ tency with existing feedforward 3DGS pipelines and to ensure fair comparisons, we adopt the official training and testing splits used in prior work and train our model at a resolution of 256Ã256. Notably, we observe that using only approximately 10% of the RE10K training data is sufficient to achieve competitive results, highlighting the efficiency and generalization capability of our network. Further implementation details are provided in the appendix.

Evaluation metrics and compared methods. We evaluate the quality of novel view synthesis using PSNR, SSIM (Wang et al. 2004), and LPIPS (Zhang et al. 2018). To assess the effectiveness of our method, we compare it against two state-of-the-art image restoration baselines, PromptIR (Potlapalli et al. 2023) and AdaIR (Cui et al. 2024), under three settings: (1) using official pretrained weights where restoration precedes 3D reconstruction; (2) retraining the restoration models under our degradation settings followed by reconstruction (â); and (3) reversing the pipeline order such that reconstruction precedes restoration on rendered images (â©). Additionally, we compare the parameters, FLOPs, and runtime to evaluate their efficiency and deployment feasibility.

## Quantitative Results

Quantitative comparisons. As presented in Table 1, the integration of the RobustGS module into the feedforward 3DGS framework Pixelsplat brings consistent and significant performance enhancements across diverse degradation scenarios and evaluation metrics. In comparison to state-ofthe-art general-purpose image restoration methods, whether applied prior to reconstruction or post-rendering, our approach demonstrates clear superiority in terms of average PSNR, SSIM, and LPIPS. To further showcase the versatility of RobustGS, we integrate it into another prominent feedforward 3DGS backbone, MVSplat. As illustrated in Table 2, RobustGS consistently achieves competitive performance across all metrics, reaffirming its robustness and adaptability to varying architectures and conditions.

<table><tr><td>Deg emb</td><td>DegâSSM</td><td>SR</td><td>MV</td><td>PSNR</td><td>SSIM</td></tr><tr><td></td><td></td><td>&gt;&gt;</td><td></td><td>21.52</td><td>0.7803</td></tr><tr><td></td><td></td><td></td><td></td><td>21.71</td><td>0.7846</td></tr><tr><td>&gt;&gt;&gt;&gt;</td><td>â</td><td></td><td>&gt;&gt;&gt;</td><td>22.05</td><td>0.7850</td></tr><tr><td></td><td>â</td><td>v</td><td></td><td>21.96</td><td>0.7794</td></tr><tr><td></td><td>V</td><td></td><td>â</td><td>22.32</td><td>0.7869</td></tr></table>

Table 3: Ablation study.

Complexity comparisons. In addition to evaluating performance metrics, we compare the computational complexity of different methods, including the number of parameters, FLOPs, and runtime, with FLOPs and runtime measured using two input images of size 64 Ã 64. As shown in Table 1, the RobustGS module not only delivers substantial performance gains but also achieves a lower computational cost compared to existing image restoration approaches. Furthermore, as a plug-and-play solution that requires no retraining of the original 3DGS model, RobustGS offers an efficient and practical approach to enhancing 3D reconstruction across diverse degradation scenarios.

<!-- image-->  
Figure 5: Feature map visualization under rain scenes.

## Qualitative Results

Figure 4 presents qualitative comparisons under two representative degradation scenarios, dark and fog, between our RobustGS and a pipeline combining representative IR methods with PixelSplat reconstruction. In these pipelines, IR methods are applied to enhance input images before 3D reconstruction. While effective for single-view enhancement, these methods fail to ensure geometric consistency across multiple views, often leading to rendering artifacts in reconstructed images. For instance, in the dark scenario (second column), AdaIR introduces noticeable black blob artifacts on the wall due to inconsistent restoration across views, resulting in erroneous geometry. In contrast, RobustGS explicitly models cross-view consistency at the feature level, effectively mitigating such artifacts and producing stable, visually reliable reconstructions. To further validate its effectiveness, we visualize intermediate feature maps in Figure 5. The enhanced features generated by RobustGS successfully remove most degradation artifacts from the original features, yielding clearer structures and more complete semantic representations.

## Ablation Study

In this section, we conduct ablation studies to evaluate the contributions of each key component in RobustGS. We first analyze the impact of the Generalized Degradation Learner and compare two integration strategies: direct concatenation of the degradation embedding versus injection into the SSM. Additionally, we examine the effects of semantic-guided token reordering in Multi-View State-Space Enhancement Module and the role of multi-view interaction versus singleview processing. As summarized in Table 3, the full design consistently achieves the best performance, where SR represents semantic reorder and MV denotes multi-view. To validate GenDeg, we visualize its degradation embeddings using t-SNE, where embeddings for different degradation types form distinct clusters, separated from clean images, demonstrating their discriminative power and generalizability. Further ablation results are included in the Appendix.

<!-- image-->  
Figure 6: t-SNE visualization.

## Conclusion

This paper presents RobustGS, a general and efficient feature enhancement module designed to improve the robustness of feedforward 3D Gaussian Splatting (3DGS) methods under challenging degraded conditions. By introducing the Generalized Degradation Learner to extract compact and generalizable degradation representations and the Multi-View State-Space Enhancement Module to leverage degradation-aware cues and semantic consistency for feature enhancement, RobustGS addresses the limitations of existing 3DGS approaches in handling degraded multi-view inputs. Extensive experiments demonstrate that RobustGS, integrated seamlessly into existing pipelines without requiring retraining, consistently achieves state-of-the-art performance across multiple degradation scenarios, significantly improving reconstruction quality in terms of PSNR, SSIM, and LPIPS metrics while maintaining lower computational complexity.

## References

Barron, J. T.; Mildenhall, B.; Tancik, M.; Hedman, P.; Martin-Brualla, R.; and Srinivasan, P. P. 2021. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, 5855â5864.

Bui, M.-Q. V.; Park, J.; Bello, J. L. G.; Moon, J.; Oh, J.; and Kim, M. 2025. MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video. arXiv preprint arXiv:2504.15122.

Catley-Chandar, S.; Shaw, R.; Slabaugh, G.; and Perez- Â´ Pellitero, E. 2024. RoGUENeRF: a robust geometryconsistent universal enhancer for NeRF. In European Conference on Computer Vision, 54â71. Springer.

Charatan, D.; Li, S.; Tagliasacchi, A.; and Sitzmann, V. 2024. pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction. In CVPR.

Chee, E.; and Wu, Z. 2018. Airnet: Self-supervised affine registration for 3d medical images using neural networks. arXiv preprint arXiv:1810.02583.

Chen, D.; Li, H.; Ye, W.; Wang, Y.; Xie, W.; Zhai, S.; Wang, N.; Liu, H.; Bao, H.; and Zhang, G. 2024a. Pgsr: Planarbased gaussian splatting for efficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics.

Chen, S.; Yan, B.; Sang, X.; Chen, D.; Wang, P.; Guo, X.; Zhong, C.; and Wan, H. 2023. Bidirectional optical flow NeRF: High accuracy and high quality under fewer views. In Proceedings of the AAAI Conference on Artificial Intelligence, 1, 359â368.

Chen, S.; Zhou, J.; and Li, L. 2024. Optimizing 3d gaussian splatting for sparse viewpoint scene reconstruction. arXiv preprint arXiv:2409.03213.

Chen, W.; Li, Z.; Guo, J.; Zheng, C.; and Tian, S. 2025. Trends and Techniques in 3D Reconstruction and Rendering: A Survey with Emphasis on Gaussian Splatting. Sensors, 25(12): 3626.

Chen, W.; and Liu, L. 2024. Deblur-gs: 3d gaussian splatting from camera motion blurred images. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 7(1): 1â15.

Chen, W.-T.; Vong, Y.-J.; Kuo, S.-Y.; Ma, S.; and Wang, J. 2024b. RobustSAM: Segment Anything Robustly on Degraded Images. In CVPR.

Chen, Y.; Xu, H.; Zheng, C.; Zhuang, B.; Pollefeys, M.; Geiger, A.; Cham, T.-J.; and Cai, J. 2024c. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, 370â 386. Springer.

Conde, M. V.; Lei, Z.; Li, W.; Katsavounidis, I.; Timofte, R.; Yan, M.; Liu, X.; Wang, Q.; Ye, X.; Du, Z.; et al. 2024. Realtime 4k super-resolution of compressed AVIF images. AIS 2024 challenge survey. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5838â5856.

Cui, Y.; Zamir, S. W.; Khan, S.; Knoll, A.; Shah, M.; and Khan, F. S. 2024. Adair: Adaptive all-in-one image restoration via frequency mining and modulation. arXiv preprint arXiv:2403.14614.

Cui, Z.; Chu, X.; and Harada, T. 2025. Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment. In Proceedings of the Computer Vision and Pattern Recognition Conference, 26472â26482.

Dalal, A.; Hagen, D.; Robbersmyr, K. G.; and Knausgard, Ë K. M. 2024. Gaussian splatting: 3D reconstruction and novel view synthesis: A review. IEEE Access, 12: 96797â96820.

Di, X.; Peng, L.; Xia, P.; Li, W.; Pei, R.; Cao, Y.; Wang, Y.; and Zha, Z.-J. 2025. Qmambabsr: Burst image superresolution with query state space model. In Proceedings of the Computer Vision and Pattern Recognition Conference, 23080â23090.

Du, Z.; Peng, L.; Wang, Y.; Cao, Y.; and Zha, Z.-J. 2024. FC3DNET: A Fully Connected Encoder-Decoder for Efficient Demoireing. In Â´ 2024 IEEE International Conference on Image Processing (ICIP), 1642â1648. IEEE.

Fei, B.; Xu, J.; Zhang, R.; Zhou, Q.; Yang, W.; and He, Y. 2024. 3d gaussian splatting as new era: A survey. IEEE Transactions on Visualization and Computer Graphics.

Feng, X.; He, Y.; Wang, Y.; Yang, Y.; Li, W.; Chen, Y.; Kuang, Z.; Fan, J.; Jun, Y.; et al. 2024. Srgs: Super-resolution 3d gaussian splatting. arXiv preprint arXiv:2404.10318.

Feng, Z.; Peng, L.; Di, X.; Guo, Y.; Li, W.; Zhang, Y.; Pei, R.; Wang, Y.; Cao, Y.; and Zha, Z.-J. 2025. PMQ-VE: Progressive Multi-Frame Quantization for Video Enhancement. arXiv preprint arXiv:2505.12266.

Gao, K.; Gao, Y.; He, H.; Lu, D.; Xu, L.; and Li, J. 2022. Nerf: Neural radiance field in 3d vision, a comprehensive review. arXiv preprint arXiv:2210.00379.

Gao, X.; Yang, Z.; Gong, B.; Han, X.; Yang, S.; and Jin, X. 2025. Towards realistic example-based modeling via 3d gaussian stitching. In Proceedings of the Computer Vision and Pattern Recognition Conference, 26597â26607.

Gong, Y.; Huang, L.; and Chen, L. 2022. Person reidentification method based on color attack and joint defence. In CVPR, 2022, 4313â4322.

Gong, Y.; Zhang, C.; Hou, Y.; Chen, L.; and Jiang, M. 2024. Beyond dropout: Robust convolutional neural networks based on local feature masking. In IJCNN, 2024. IEEE.

Guo, H.; Guo, Y.; Zha, Y.; Zhang, Y.; Li, W.; Dai, T.; Xia, S.-T.; and Li, Y. 2025. Mambairv2: Attentive state space restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, 28124â28133.

Guo, H.; Li, J.; Dai, T.; Ouyang, Z.; Ren, X.; and Xia, S.- T. 2024. Mambair: A simple baseline for image restoration with state-space model. In European conference on computer vision, 222â241. Springer.

He, C.; Fang, C.; Zhang, Y.; Li, K.; Tang, L.; You, C.; Xiao, F.; Guo, Z.; and Li, X. 2025a. Reti-diff: Illumination degradation image restoration with retinex-based latent diffusion model. ICLR.

He, C.; Li, K.; Xu, G.; Yan, J.; Tang, L.; Zhang, Y.; Wang, Y.; and Li, X. 2023. Hqg-net: Unpaired medical image enhancement with high-quality guidance. TNNLS.

He, C.; Li, K.; Zhang, Y.; Yang, Z.; Tang, L.; Zhang, Y.; Kong, L.; and Farsiu, S. 2025b. Segment concealed object with incomplete supervision. TPAMI.

He, C.; Shen, Y.; Fang, C.; Xiao, F.; Tang, L.; Zhang, Y.; Zuo, W.; Guo, Z.; and Li, X. 2025c. Diffusion Models in Low-Level Vision: A Survey. TPAMI.

He, C.; Zhang, R.; Xiao, F.; Fang, C.; Tang, L.; Zhang, Y.; and Farsiu, S. 2025d. UnfoldIR: Rethinking Deep Unfolding Network in Illumination Degradation Image Restoration. arXiv preprint arXiv:2505.06683.

He, C.; Zhang, R.; Xiao, F.; Fang, C.; Tang, L.; Zhang, Y.; Kong, L.; Fan, D.-P.; Li, K.; and Farsiu, S. 2025e. RUN: Reversible Unfolding Network for Concealed Object Segmentation. ICML.

He, Y.; Jiang, A.; Jiang, L.; Peng, L.; Wang, Z.; and Wang, L. 2024a. Dual-path coupled image deraining network via spatial-frequency interaction. In 2024 IEEE International Conference on Image Processing (ICIP), 1452â1458. IEEE.

He, Y.; Peng, L.; Wang, L.; and Cheng, J. 2024b. Latent degradation representation constraint for single image deraining. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 3155â3159. IEEE.

He, Y.; Peng, L.; Yi, Q.; Wu, C.; and Wang, L. 2024c. Multi-scale representation learning for image restoration with state-space model. arXiv preprint arXiv:2408.10145.

Huang, X.; Li, W.; Hu, J.; Chen, H.; and Wang, Y. 2023. Refsr-nerf: Towards high fidelity and super resolution view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8244â8253.

Ignatov, A.; Perevozchikov, G.; Timofte, R.; Pan, W.; Wang, S.; Zhang, D.; Ran, Z.; Li, X.; Ju, S.; Zhang, D.; et al. 2025. Rgb photo enhancement on mobile gpus, mobile ai 2025 challenge: Report. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1922â1933.

Jain, A.; Tancik, M.; and Abbeel, P. 2021. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, 5885â5894.

Jiang, A.; Wei, Z.; Peng, L.; Liu, F.; Li, W.; and Wang, M. 2024. Dalpsr: Leverage degradation-aligned language prompt for real-world image super-resolution. arXiv preprint arXiv:2406.16477.

Jin, X.; Guo, C.; Li, X.; Yue, Z.; Li, C.; Zhou, S.; Feng, R.; Dai, Y.; Yang, P.; Loy, C. C.; et al. 2024a. MIPI 2024 Challenge on Few-shot RAW Image Denoising: Methods and Results. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1153â1161.

Jin, X.; Jiao, P.; Duan, Z.-P.; Yang, X.; Li, C.; Guo, C.-L.; and Ren, B. 2024b. Lighting every darkness with 3dgs: Fast training and real-time rendering for hdr view synthesis. Advances in Neural Information Processing Systems, 37: 80191â80219.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139â1.

Khalid, S.; Ibrahim, M.; and Liu, Y. 2025. GaussianVAE: Adaptive Learning Dynamics of 3D Gaussians for High-Fidelity Super-Resolution. arXiv preprint arXiv:2506.07897.

Kim, H.; and Lee, I.-K. 2024. Is 3dgs useful?: Comparing the effectiveness of recent reconstruction methods in vr. In 2024 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), 71â80. IEEE.

Kingma, D. P.; and Ba, J. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Kwon, W.; Sung, J.; Jeon, M.; Eom, C.; and Oh, J. 2025. R3eVision: A Survey on Robust Rendering, Restoration, and Enhancement for 3D Low-Level Vision. arXiv preprint arXiv:2506.16262.

Lee, B.; Lee, H.; Sun, X.; Ali, U.; and Park, E. 2024. Deblurring 3d gaussian splatting. In European Conference on Computer Vision, 127â143. Springer.

Li, H.; and Fu, Y. 2025. FCDFusion: A fast, low color deviation method for fusing visible and infrared image pairs. Computational Visual Media, 11(1): 195â211.

Li, H.; Wu, Z.; Shao, R.; Zhang, T.; and Fu, Y. 2025a. Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Conference, 21287â21296.

Li, W.; Guo, H.; Hou, Y.; Gao, G.; and Ma, Z. 2025b. Dualdomain modulation network for lightweight image superresolution. arXiv preprint arXiv:2503.10047.

Li, W.; Guo, H.; Hou, Y.; and Ma, Z. 2025c. FourierSR: A Fourier Token-based Plugin for Efficient Image Super-Resolution. arXiv preprint arXiv:2503.10043.

Li, W.; Guo, H.; Liu, X.; Liang, K.; Hu, J.; Ma, Z.; and Guo, J. 2024a. Efficient face super-resolution via wavelet-based feature enhancement network. In Proceedings of the 32nd ACM International Conference on Multimedia, 4515â4523.

Li, W.; Li, J.; Gao, G.; Deng, W.; Yang, J.; Qi, G.-J.; and Lin, C.-W. 2024b. Efficient image super-resolution with feature interaction weighted hybrid network. IEEE Transactions on Multimedia.

Li, W.; Li, J.; Gao, G.; Deng, W.; Zhou, J.; Yang, J.; and Qi, G.-J. 2023a. Cross-receptive focused inference network for lightweight image super-resolution. IEEE Transactions on Multimedia, 26: 864â877.

Li, Y.; Li, Z.; Liu, D.; and Li, L. 2025d. Frequency Domain Intra Pattern Copy for JPEG XS Screen Content Coding. IEEE Transactions on Circuits and Systems for Video Technology.

Li, Y.; Zhang, Y.; Timofte, R.; Van Gool, L.; Yu, L.; Li, Y.; Li, X.; Jiang, T.; Wu, Q.; Han, M.; et al. 2023b. NTIRE 2023 challenge on efficient super-resolution: Methods and results. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1922â1960.

Li, Z.; Li, J.; Li, Y.; Li, L.; Liu, D.; and Wu, F. 2024c. Inloop filtering via trained look-up tables. In 2024 IEEE International Conference on Visual Communications and Image Processing (VCIP), 1â5. IEEE.

Li, Z.; Li, Y.; Tang, C.; Li, L.; Liu, D.; and Wu, F. 2024d. Uniformly accelerated motion model for inter prediction. In 2024 IEEE International Conference on Visual Communications and Image Processing (VCIP), 1â5. IEEE.

Li, Z.; Liao, J.; Tang, C.; Zhang, H.; Li, Y.; Bian, Y.; Sheng, X.; Feng, X.; Li, Y.; Gao, C.; et al. 2025e. USTC-TD: A Test Dataset and Benchmark for Image and Video Coding in 2020s. IEEE Transactions on Multimedia, 1â16.

Li, Z.; Wang, Y.; Kot, A.; and Wen, B. 2024e. From chaos to clarity: 3DGS in the dark. Advances in Neural Information Processing Systems, 37: 94971â94992.

Li, Z.; Yuan, Z.; Li, L.; Liu, D.; Tang, X.; and Wu, F. 2024f. Object segmentation-assisted inter prediction for versatile video coding. IEEE Transactions on Broadcasting.

Liang, J.; Cao, J.; Sun, G.; Zhang, K.; Van Gool, L.; and Timofte, R. 2021. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, 1833â1844.

Lin, J.; Zhenzhong, W.; Dejun, X.; Shu, J.; Gong, Y.; and Jiang, M. 2025a. Phys4DGen: A Physics-Driven Framework for Controllable and Efficient 4D Content Generation from a Single Image. In ACM MM, 2025.

Lin, X.; Luo, S.; Shan, X.; Zhou, X.; Ren, C.; Qi, L.; Yang, M.-H.; and Vasconcelos, N. 2025b. HQGS: High-Quality Novel View Synthesis with Gaussian Splatting in Degraded Scenes. In The Thirteenth International Conference on Learning Representations.

Liu, Y.; Jia, B.; Lu, R.; Ni, J.; Zhu, S.-C.; and Huang, S. 2025. Artgs: Building interactable replicas of complex articulated objects via gaussian splatting. arXiv preprint arXiv:2502.19459.

Liu, Y.-L.; Gao, C.; Meuleman, A.; Tseng, H.-Y.; Saraf, A.; Kim, C.; Chuang, Y.-Y.; Kopf, J.; and Huang, J.-B. 2023. Robust dynamic radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13â23.

Lu, Z.; Guo, X.; Hui, L.; Chen, T.; Yang, M.; Tang, X.; Zhu, F.; and Dai, Y. 2024. 3d geometry-aware deformable gaussian splatting for dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8900â8910.

Lyu, X.; Sun, Y.-T.; Huang, Y.-H.; Wu, X.; Yang, Z.; Chen, Y.; Pang, J.; and Qi, X. 2024. 3dgsr: Implicit surface reconstruction with 3d gaussian splatting. ACM Transactions on Graphics (TOG), 43(6): 1â12.

Ma, C.; Zhao, J.; and Chen, J. 2025. DehazeGS: 3D Gaussian Splatting for Multi-Image Haze Removal. IEEE Signal Processing Letters.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Muller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In-Â¨ stant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1â 15.

Pan, J.; Liu, Y.; He, X.; Peng, L.; Li, J.; Sun, Y.; and Huang, X. 2025. Enhance then search: An augmentation-search strategy with foundation models for cross-domain few-shot object detection. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1548â1556.

Peng, L.; Cao, Y.; Pei, R.; Li, W.; Guo, J.; Fu, X.; Wang, Y.; and Zha, Z.-J. 2024a. Efficient real-world image superresolution via adaptive directional gradient convolution. arXiv preprint arXiv:2405.07023.

Peng, L.; Cao, Y.; Sun, Y.; and Wang, Y. 2024b. Lightweight adaptive feature de-drifting for compressed image classification. IEEE Transactions on Multimedia, 26: 6424â6436.

Peng, L.; Di, X.; Feng, Z.; Li, W.; Pei, R.; Wang, Y.; Fu, X.; Cao, Y.; and Zha, Z.-J. 2025a. Directing mamba to complex textures: An efficient texture-aware state space model for image restoration. arXiv preprint arXiv:2501.16583.

Peng, L.; Jiang, A.; Wei, H.; Liu, B.; and Wang, M. 2021. Ensemble single image deraining network via progressive structural boosting constraints. Signal Processing: Image Communication, 99: 116460.

Peng, L.; Jiang, A.; Yi, Q.; and Wang, M. 2020. Cumulative rain density sensing network for single image derain. IEEE Signal Processing Letters, 27: 406â410.

Peng, L.; Li, W.; Guo, J.; Di, X.; Sun, H.; Li, Y.; Pei, R.; Wang, Y.; Cao, Y.; and Zha, Z.-J. 2024c. Unveiling hidden details: A raw data-enhanced paradigm for real-world superresolution. arXiv preprint arXiv:2411.10798.

Peng, L.; Li, W.; Pei, R.; Ren, J.; Xu, J.; Wang, Y.; Cao, Y.; and Zha, Z.-J. 2024d. Towards realistic data generation for real-world super-resolution. arXiv preprint arXiv:2406.07255.

Peng, L.; Wang, Y.; Di, X.; Fu, X.; Cao, Y.; Zha, Z.-J.; et al. 2025b. Boosting image de-raining via central-surrounding synergistic convolution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 6470â6478.

Peng, L.; Wu, A.; Li, W.; Xia, P.; Dai, X.; Zhang, X.; Di, X.; Sun, H.; Pei, R.; Wang, Y.; et al. 2025c. Pixel to gaussian: Ultra-fast continuous super-resolution with 2d gaussian modeling. arXiv preprint arXiv:2503.06617.

Philip, J.; Gharbi, M.; Zhou, T.; Efros, A. A.; and Drettakis, G. 2019. Multi-view relighting using a geometry-aware network. ACM Trans. Graph., 38(4): 78â1.

Potlapalli, V.; Zamir, S. W.; Khan, S. H.; and Shahbaz Khan, F. 2023. Promptir: Prompting for all-in-one image restoration. Advances in Neural Information Processing Systems, 36: 71275â71293.

Qi, X.; Li, R.; Peng, L.; Ling, Q.; Yu, J.; Chen, Z.; Chang, P.; Han, M.; and Xiao, J. 2025. Data-free Knowledge Distillation with Diffusion Models. arXiv preprint arXiv:2504.00870.

Qian, C.; Guo, Y.; Li, W.; and Markkula, G. 2024. Weathergs: 3d scene reconstruction in adverse weather conditions via gaussian splatting. arXiv preprint arXiv:2412.18862.

Qiao, Y.; Shao, M.; Meng, L.; and Xu, K. 2025. RestorGS: Depth-aware Gaussian Splatting for Efficient 3D Scene Restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, 11177â11186.

Ren, B.; Li, Y.; Mehta, N.; Timofte, R.; Yu, H.; Wan, C.; Hong, Y.; Han, B.; Wu, Z.; Zou, Y.; et al. 2024a. The ninth NTIRE 2024 efficient super-resolution challenge report. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6595â6631.

Ren, J.; Chen, H.; Ye, T.; Wu, H.; and Zhu, L. 2025a. Triplane-smoothed video dehazing with clip-enhanced generalization. International Journal of Computer Vision, 133(1): 475â488.

Ren, J.; Hu, X.; Zhu, L.; Xu, X.; Xu, Y.; Wang, W.; Deng, Z.; and Heng, P.-A. 2021. Deep texture-aware features for camouflaged object detection. IEEE Transactions on Circuits and Systems for Video Technology, 33(3): 1157â1167.

Ren, J.; Li, W.; Chen, H.; Pei, R.; Shao, B.; Guo, Y.; Peng, L.; Song, F.; and Zhu, L. 2024b. Ultrapixel: Advancing ultra high-resolution image synthesis to new peaks. Advances in Neural Information Processing Systems, 37: 111131â 111171.

Ren, J.; Li, W.; Wang, Z.; Sun, H.; Liu, B.; Chen, H.; Xu, J.; Li, A.; Zhang, S.; Shao, B.; et al. 2025b. Turbo2K: Towards Ultra-Efficient and High-Quality 2K Video Synthesis. arXiv preprint arXiv:2504.14470.

Ren, J.; Xu, C.; Chen, H.; Qin, X.; and Zhu, L. 2023. Towards flexible, scalable, and adaptive multi-modal conditioned face synthesis. arXiv preprint arXiv:2312.16274.

Ren, J.; Zheng, Q.; Zhao, Y.; Xu, X.; and Li, C. 2022. Dlformer: Discrete latent transformer for video inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 3511â3520.

Riegler, G.; and Koltun, V. 2020. Free view synthesis. In European conference on computer vision, 623â640. Springer.

Schonberger, J. L.; and Frahm, J.-M. 2016. Structure-from- Â¨ Motion Revisited. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4104â4113.

Smart, B.; Zheng, C.; Laina, I.; and Prisacariu, V. A. 2024. Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs. arXiv.

Su, J.; Xu, B.; and Yin, H. 2022. A survey of deep learning approaches to image restoration. Neurocomputing, 487: 46â 65.

Sun, H.; Li, W.; Liu, J.; Zhou, K.; Chen, Y.; Guo, Y.; Li, Y.; Pei, R.; Peng, L.; and Yang, Y. 2024. Beyond Pixels: Text Enhances Generalization in Real-World Image Restoration. arXiv preprint arXiv:2412.00878.

Szymanowicz, S.; Rupprecht, C.; and Vedaldi, A. 2024. Splatter image: Ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10208â10217.

Tang, J.; Chen, Z.; Chen, X.; Wang, T.; Zeng, G.; and Liu, Z. 2024. Lgm: Large multi-view gaussian model for highresolution 3d content creation. In European Conference on Computer Vision, 1â18. Springer.

Wang, H.; Peng, L.; Sun, Y.; Wan, Z.; Wang, Y.; and Cao, Y. 2023a. Brightness perceiving for recursive low-light image enhancement. IEEE Transactions on Artificial Intelligence, 5(6): 3034â3045.

Wang, X.; Xie, L.; Dong, C.; and Shan, Y. 2021. Realesrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, 1905â1914.

Wang, Y.; Liang, Z.; Zhang, F.; Tian, L.; Wang, L.; Li, J.; Yang, J.; Timofte, R.; Guo, Y.; Jin, K.; et al. 2025. NTIRE 2025 challenge on light field image super-resolution: Methods and results. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1227â1246.

Wang, Y.; Peng, L.; Li, L.; Cao, Y.; and Zha, Z.-J. 2023b. Decoupling-and-aggregating for image exposure correction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 18115â18124.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600â612.

Wu, C.; Wang, L.; Peng, L.; Lu, D.; and Zheng, Z. 2025. Dropout the high-rate downsampling: A novel design paradigm for uhd image restoration. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2390â2399. IEEE.

Wu, T.; Yuan, Y.-J.; Zhang, L.-X.; Yang, J.; Cao, Y.-P.; Yan, L.-Q.; and Gao, L. 2024a. Recent advances in 3d gaussian splatting. Computational Visual Media, 10(4): 613â642.

Wu, Z.; Wan, Z.; Zhang, J.; Liao, J.; and Xu, D. 2024b. RaFE: Generative Radiance Fields Restoration. In European Conference on Computer Vision, 163â179. Springer.

Xia, P.; Peng, L.; Di, X.; Pei, R.; Wang, Y.; Cao, Y.; and Zha, Z.-J. 2024. S3mamba: Arbitrary-scale superresolution via scaleable state space model. arXiv preprint arXiv:2411.11906, 6.

Xiao, Z.; Kai, D.; Zhang, Y.; Zha, Z.-J.; Sun, X.; and Xiong, Z. 2024. Event-adapted video super-resolution. In European Conference on Computer Vision, 217â235. Springer.

Xiao, Z.; and Wang, X. 2025. Event-based Video Super-Resolution via State Space Models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 12564â12574.

Xiao, Z.; and Xiong, Z. 2025. Incorporating degradation estimation in light field spatial super-resolution. Computer Vision and Image Understanding, 252: 104295.

Yan, Q.; Jiang, A.; Chen, K.; Peng, L.; Yi, Q.; and Zhang, C. 2025. Textual prompt guided image restoration. Engineering Applications of Artificial Intelligence, 155: 110981.

Yang, Z.; Gao, X.; Zhou, W.; Jiao, S.; Zhang, Y.; and Jin, X. 2024. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 20331â20341.

Ye, B.; Liu, S.; Xu, H.; Li, X.; Pollefeys, M.; Yang, M.-H.; and Peng, S. 2024. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207.

Yi, Q.; Li, J.; Dai, Q.; Fang, F.; Zhang, G.; and Zeng, T. 2021a. Structure-preserving deraining with residue channel prior guidance. In Proceedings of the IEEE/CVF international conference on computer vision, 4238â4247.

Yi, Q.; Li, J.; Fang, F.; Jiang, A.; and Zhang, G. 2021b. Efficient and accurate multi-scale topological network for single image dehazing. IEEE Transactions on Multimedia, 24: 3114â3128.

Yi, Q.; Li, S.; Wu, R.; Sun, L.; Wu, Y.; and Zhang, L. 2025. Fine-structure Preserved Real-world Image Superresolution via Transfer VAE Training. arXiv preprint arXiv:2507.20291.

Zeng, Q.; Gong, Y.; and Jiang, M. 2024. Cross-Task Attack: A Self-Supervision Generative Framework Based on Attention Shift. In IJCNN, 2024. IEEE.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586â595.

Zhao, R.; Xiong, R.; Zhang, J.; Yu, Z.; Zhu, S.; Ma, L.; and Huang, T. 2023. Spike camera image reconstruction using deep spiking neural networks. IEEE Transactions on Circuits and Systems for Video Technology, 34(6): 5207â5212.

Zhao, R.; Xiong, R.; Zhao, J.; Zhang, J.; Fan, X.; Yu, Z.; and Huang, T. 2024. Boosting spike camera image reconstruction from a perspective of dealing with spike fluctuations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 24955â24965.

Zheng, Y.; Zhong, B.; Liang, Q.; Li, G.; Ji, R.; and Li, X. 2023. Toward unified token learning for vision-language tracking. IEEE Transactions on Circuits and Systems for Video Technology, 34(4): 2125â2135.

Zheng, Y.; Zhong, B.; Liang, Q.; Li, N.; and Song, S. 2025a. Decoupled Spatio-Temporal Consistency Learning for Self-Supervised Tracking. In Proceedings of the AAAI Conference on Artificial Intelligence, 10, 10635â10643.

Zheng, Y.; Zhong, B.; Liang, Q.; Mo, Z.; Zhang, S.; and Li, X. 2024. Odtrack: Online dense temporal token learning for visual tracking. In Proceedings of the AAAI conference on artificial intelligence, 7, 7588â7596.

Zheng, Y.; Zhong, B.; Liang, Q.; Tang, Z.; Ji, R.; and Li, X. 2022. Leveraging local and global cues for visual tracking via parallel interaction network. IEEE Transactions on Circuits and Systems for Video Technology, 33(4): 1671â1683.

Zheng, Y.; Zhong, B.; Liang, Q.; Zhang, S.; Li, G.; Li, X.; and Ji, R. 2025b. Towards Universal Modal Tracking with Online Dense Temporal Token Learning. arXiv preprint arXiv:2507.20177.

Zhou, T.; Tucker, R.; Flynn, J.; Fyffe, G.; and Snavely, N. 2018. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817.

Zhou, Z.; and Tulsiani, S. 2023. Sparsefusion: Distilling view-conditioned diffusion for 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12588â12597.

Zhu, T.; Li, H.; and Fu, Y. 2025. TriM-SOD: A Multi-modal, Multi-task, and Multi-scale Spacecraft Optical Dataset. Space: Science & Technology, 27.

## Implementation Details

The training process consists of two stages. In the first stage, we use the DIV2K dataset as the source of high-quality (HQ) images and generate their corresponding low-quality (LQ) versions using the same degradation method as in Robust-SAM (Chen et al. 2024b). The degradation types used during training include dark, fog, rain, snow, high contrast, and impulse noise. We train Generalized Degradation Learner (GenDeg) for 20,000 iterations. In the second stage, we randomly select two HQ frames from each video sequence and generate their corresponding LQ frames using the same degradation strategy as in the first stage. The two LQ frames are passed through the pretrained feature extraction network provided by PixelSplat or other feedforward 3DGS backbone to obtain feature maps. These feature maps, along with their corresponding degradation embeddings extracted by the pretrained GenDeg, are then fed into Multi-View State-Space Enhancement Module (MV-SSEM) for enhancement. During this stage, GenDeg functions solely as a fixed degradation extractor, with its parameters frozen and excluded from the training process. The enhanced feature maps are supervised using the L1 loss function. Adam (Kingma and Ba 2014) is used as the optimizer, with the initial learning rate set to 1e-4 and halved every 100 epochs. The model is trained for 1000 epochs with a total batch size of 32 on 8 NVIDIA 4090 GPUs.

## Algorithm Workflow

To clearly demonstrate the details of the proposed semanticguided mechanism, we design an algorithm workflow, as illustrated in Algorithm 1. It outlines the entire process from multi-view feature flattening to token-level semantic assignment, reordering, prompt generation, and final enhancement via selective scan modulation. This mechanism plays a central role in enabling semantic-aware feature aggregation and enhancement across views.

## Addtional Comparison

## More Results of Mvsplat

In the main paper, we have compared our proposed method based on PixelSplat under six different types of degradations. We also reported the average performance over multiple degradations on MVSplat. As shown in Table 4, we further present detailed performance comparisons on MVSplat under each individual degradation type.

## Additional Visual Comparison Results

In this section, we present additional visual comparison results on novel view synthesis to further demonstrate the superiority of our proposed method, as shown in Figure 7. It can be observed that our method achieves the best visual satisfaction in terms of detailed textures, while also preserving the highest level of detail fidelity, making it closest to the GT image.

## Additional Ablation Study

Due to space limitations in the main text, we provide additional ablation experiments to demonstrate the effectiveness and rationality of the proposed method. Below, we present detailed descriptions of additional ablation studies and implementation details.

Algorithm 1: Semantic-Guided Mechanism   
Require: x â RBÃV ÃCÃHÃW // multi-view features   
Ensure: xenhanced $\in \mathbb { R } ^ { B \times L \times C }$   
1: $L \gets V \cdot H \cdot W$   
2: x â reshape(x, [B, L, C]) // flatten tokens   
3: $\mathbf { q }  \mathbf { M L P } ( \mathbf { x } )$ // token-wise semantic query   
4: $\dot { \mathbf { r } }  \mathrm { L i n e a r } ( \mathbf { \check { q } } ) \in \mathbb { R } ^ { B \times L \times T }$ // logits for routing to T   
semantic prompts   
5: P â GumbelSoftmax(r, dim = T ) // hard one-hot   
token assignment   
6: c â arg max(P, dim = T ) // prompt class index per   
token   
7: Ï â argsort(c) // sort tokens by assigned prompt class   
8: e â P Â· WE // embedding lookup from prompt pool   
9: p â Linear(e) // semantic prompt per token   
10: $\mathbf { x } ^ { \prime }  \mathrm { g a t h e r } ( \mathbf { x } , \pi )$ // token reordering   
11: $\mathbf { p } ^ { \prime }  \mathrm { g a t h e r } ( \mathbf { p } , \pi )$   
$1 2 \colon \hat { \mathbf { x } }  \mathrm { S e l e c t i v e S c a n } ( \mathbf { x } ^ { \prime } , \mathbf { p } ^ { \prime } )$ // SSM modulated by   
semantic prompt   
13: Ïâ1 â inverse argsort(Ï) // recover original order   
14: xenhanced â gather(xË, Ïâ1)   
15: return xenhanced

Ablation Study on GenDeg. Existing methods such as DegAE adopt a degradation-aware encoder to extract degradation embeddings from input images for restoration purposes, but these approaches typically rely on reconstruction loss alone. In contrast, our proposed GenDeg introduces two additional objectives: a contrastive loss to encourage discrimination between different degradation types and a classification loss to further promote semantic alignment with known degradation categories. Table 5 presents an ablation study on the loss functions used in GenDeg, demonstrating that our design contribute significantly to performance.

Ablation Study on MV-SSEM. In the MV-SSEM, we use a semantic prompt pool of size K to cluster tokens based on semantic similarity. We conduct an ablation study by varying the number of semantic prompts (K = 32, 64, 128), as shown in Table 6. The results indicate that K = 64 yields the best performance. A smaller K leads to overly coarse semantic grouping, while a larger K causes excessive fragmentation, making cross-view aggregation unstable. These experientments validate our design choice for prompt pool size.

We also conduct an ablation study to evaluate the impact of the internal dimensionality of each semantic prompt, denoted as $d _ { \mathrm { i n n e r } } ,$ on the final performance. As shown in Table 7, increasing the dimensionality from 32 to 128 leads to consistent improvements. However, further increasing beyond 128 brings no significant gain. Therefore, we choose $\dot { d } _ { \mathrm { i n n e r } } = 1 2 8$ in our implementation to balance representation capacity and computational cost.

<table><tr><td rowspan="2"></td><td colspan="2">Brightness</td><td colspan="2">Fog</td><td colspan="2">Rain</td><td colspan="2">Snow</td><td colspan="2">Contrast</td><td colspan="2">Impulse noise</td></tr><tr><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td>mvsplat</td><td>15.58</td><td>0.8035</td><td>15.52</td><td>0.7471</td><td>19.97</td><td>0.7498</td><td>19.17</td><td>0.5852</td><td>18.51</td><td>0.7298</td><td>14.47</td><td>0.7688</td></tr><tr><td>PromptIR</td><td>14.87</td><td>0.7791</td><td>21.52</td><td>0.8432</td><td>21.99</td><td>0.7637</td><td>19.93</td><td>0.6216</td><td>18.39</td><td>0.7262</td><td>14.70</td><td>0.7732</td></tr><tr><td>PrommptI</td><td>16.45</td><td>0.7722</td><td>20.47</td><td>0.8048</td><td>21.02</td><td>0.7765</td><td>21.52</td><td>0.6807</td><td>20.95</td><td>0.7704</td><td>19.84</td><td>0.7908</td></tr><tr><td>PromptIR</td><td>17.34</td><td>0.7703</td><td>20.78</td><td>0.8129</td><td>20.09</td><td>0.7546</td><td>19.22</td><td>0.6234</td><td>18.17</td><td>0.7274</td><td>20.26</td><td>0.7970</td></tr><tr><td>AdalIR</td><td>15.37</td><td>0.7950</td><td>20.95</td><td>0.8329</td><td>21.37</td><td>0.7560</td><td>19.74</td><td>0.6171</td><td>18.43</td><td>0.7285</td><td>14.73</td><td>0.7702</td></tr><tr><td>AdaIR</td><td>21.74</td><td>0.8460</td><td>15.95</td><td>0.7476</td><td>22.23</td><td>0.7839</td><td>17.58</td><td>0.6387</td><td>19.43</td><td>0.7412</td><td>20.57</td><td>0.8029</td></tr><tr><td>AdaIR</td><td>22.05</td><td>0.8455</td><td>16.93</td><td>0.7250</td><td>19.91</td><td>0.7689</td><td>15.30</td><td>0.6005</td><td>15.36</td><td>0.6826</td><td>20.28</td><td>0.8063</td></tr><tr><td>RobustGS</td><td>22.56</td><td>0.8481</td><td>20.63</td><td>0.8179</td><td>22.50</td><td>0.7858</td><td>21.10</td><td>0.6664</td><td>21.09</td><td>0.7853</td><td>21.01</td><td>0.8071</td></tr></table>

Table 4: Comparison of performance on Mvsplat with existing methods across various degradation scenarios. Bold indicates the best performance, while underline denotes the second-best.

<!-- image-->

<!-- image-->  
PixelSplat

<!-- image-->  
PromptIR

<!-- image-->  
AdaIR

<!-- image-->  
Ours

<!-- image-->  
GT

Figure 7: Additional qualitative comparison. The visual quality of our method outperforms restore before reconstruct methods. Visual examples in the first and second rows correspond to foggy and dark degradation scenes, respectively.
<table><tr><td> $\mathcal { L } _ { \mathrm { r e c } }$ </td><td> ${ \mathcal { L } } _ { \mathrm { c o n } }$ </td><td> $\mathcal { L } _ { \mathrm { c l s } }$ </td><td>PSNR</td><td>SSIM</td></tr><tr><td>â</td><td></td><td></td><td>21.87</td><td>0.7751</td></tr><tr><td></td><td>L</td><td></td><td>22.09</td><td>0.7816</td></tr><tr><td>â</td><td></td><td>â</td><td>21.95</td><td>0.7810</td></tr><tr><td>â</td><td></td><td>V</td><td>22.32</td><td>0.7869</td></tr></table>

Table 5: Ablation study on training loss of GenDeg.

<table><tr><td>Prompt Number (K)</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>32</td><td>21.93</td><td>0.7804</td><td>0.2513</td></tr><tr><td>128</td><td>21.75</td><td>0.7717</td><td>0.2488</td></tr><tr><td>64</td><td>22.32</td><td>0.7869</td><td>0.2443</td></tr></table>

Table 6: Ablation study on K.

<table><tr><td>Prompt Dim  $( d _ { \mathrm { i n n e r } } )$ </td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>32</td><td>21.95</td><td>0.7798</td><td>0.2564</td></tr><tr><td>64</td><td>22.24</td><td>0.7847</td><td>0.2471</td></tr><tr><td>128</td><td>22.32</td><td>0.7869</td><td>0.2443</td></tr></table>

Table 7: Ablation on $d _ { \mathrm { i n n e r } }$