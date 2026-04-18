# PSGS: TEXT-DRIVEN PANORAMA SLIDING SCENE GENERATION VIA GAUSSIANSPLATTING

Xin Zhang1, Shen Chen2, Jiale Zhou1â, Lei Li3â

1East China University of Science and Technology, Shanghai, China 2Zhejiang University, Hangzhou, China 3Beijing Institute of Technology, Beijing, China

## ABSTRACT

Generating realistic 3D scenes from text is crucial for immersive applications like VR, AR, and gaming. While text-driven approaches promise efficiency, existing methods suffer from limited 3D-text data and inconsistent multi-view stitching, resulting in overly simplistic scenes. To address this, we propose PSGS, a two-stage framework for high-fidelity panoramic scene generation. First, a novel two-layer optimization architecture generates semantically coherent panoramas: a layout reasoning layer parses text into structured spatial relationships, while a self-optimization layer refines visual details via iterative MLLM feedback. Second, our panorama sliding mechanism initializes globally consistent 3D Gaussian Splatting point clouds by strategically sampling overlapping perspectives. By incorporating depth and semantic coherence losses during training, we greatly improve the quality and detail fidelity of rendered scenes. Our experiments demonstrate that PSGS outperforms existing methods in panorama generation and produces more appealing 3D scenes, offering a robust solution for scalable immersive content creation.

Index Termsâ PSGS, scene generation, panorama sliding, immersive experience.

## 1. INTRODUCTION

Virtual reality (VR), augmented reality (AR), and metaverse technologies have created unprecedented demand for realistic 3D content. Such environments are also fundamental for downstream 3D tasks [1]. Text-to-3D scene generation with LLMs offers a compelling solution for efficient content creation across multiple industries, from entertainment to architectural visualization [2][3]. However, the shortage of high-quality text-to-3D paired datasets remains a significant barrier to widespread adoption [4].

Recent 2D generation models [5] show promise for text-to-3D creation, demonstrating impressive results for object-centric generation through diffusion-based approaches [6, 7]. However, when applied to complex scene generation, these methods face substantial limitations. Unlike object-centric generation, scene construction requires both semantic and geometric coherenceâdemanding proper spatial relationships, consistent lighting, and logical environmental context. Existing approaches [8, 9, 10] struggle to maintain consistency across multiple viewpoints, resulting in fragmented scenes that lack coherent spatial arrangement and fail to meet the requirements of immersive experiences.

Text2Room [8] creates room-scale 3D scenes by NeRF [11], but it suffers from pixel-level inconsistencies between adjacent views; LucidDreamer [9] generates multi-view scenes using 3D Gaussian

Splatting [12] but produces blurred results in complex environments; DreamScene360 [10] improves global consistency but lacks precise point cloud initialization. These approaches often require extensive computational resources while still delivering suboptimal results, highlighting the demand for more efficient and effective solutions that balance realistic quality with computational resources.

To address these challenges, we introduce PSGS(as shown in Fig. 1) â a novel two-stage framework for text-driven 3D scene generation. The first stage features an innovative two-layer architecture that progressively optimizes scene generation through semantic reasoning and visual feedback. The process begins with a concise scene description, which undergoes semantic-level Chain-of-Thought (CoT) [13] reasoning to infer detailed scene layouts and spatial relationships. This semantically enriched description is then fed into text-to-panorama (t2p) model to generate an initial 360Â° panorama. Subsequently, our framework employs a MLLM (Qwen) to evaluate the generated panorama, extracting optimized prompt that enhance the original semantic description. This enhanced prompt initiates a new optimization cycle (round+1), where the improved description is processed through the t2p model again. Through this iterative optimization process, our framework achieves continuous enhancement of panorama quality, ensuring both semantic accuracy and visual fidelity in the generated scenes.

In the second stage, to ensure information continuity in 3D space, we transform the generated panorama into a series of overlapping perspective-projected images. By leveraging geometric relationships between adjacent views, we initialize coherent point clouds. To further enhance spatial continuity and reduce artifacts, we introduce dual-constraint losses into the 3D Gaussian Splatting reconstruction process. Specifically, we employ the DPT 2D model [14] for high-quality depth estimation and DINOv2 [15] for semantic feature extraction, enabling robust cross-view consistency. As a result, our rendering system produces photorealistic 3D environments with strong fidelity to the original text descriptions.

In summary, our contributions are:

â¢ We propose PSGS framework, which, with just a concise text description, can generate high-quality panorama as well as 3D scenes with global consistency and high fidelity.

â¢ We introduce an innovative two-layer optimization architecture that progressively optimizes panorama generation through semantic reasoning and visual feedback, ensuring both semantic accuracy and visual fidelity.

â¢ We develop a point cloud initialization strategy called âPanorama Slidingâ. This method expands the overlapping coverage range of each view through a âlarge but fewâ sampling approach, while reducing the number of view images required for reconstruction. By reducing the computational pressure of the model, it can effectively cover the whole scene. This method not only enhances global consistency but also improves the robustness and efficiency of the generation process.

<!-- image-->  
Fig. 1. Method Overview: Two-stage pipeline of the PSGS framework. The first stage propose a novel two-layer architecture that progressively optimizes scene generation through semantic reasoning and visual feedback. The second stage establishes globally consistent 3D reconstruction through panorama sliding, combined with semantic and geometric consistency constraints for high-fidelity rendering.

â¢ We design specialized semantic and depth consistency losses for text-to-3D scene generation, ensuring geometric coherence and semantic uniformity for photorealistic reconstructions.

## 2. METHOD

## 2.1. Two-layer Panorama Generation

## 2.1.1. Layout Reasoning Layer

This layer uses the semantic-level CoT framework of the T2I-R1 model [16], which iteratively updates the generation probability of tokens in semantic-level CoT through policy gradients. In order to evaluate the quality of the generated images and guide the model training, a reward set composed of multiple visual experts was introduced to evaluate the generated images from multiple perspectives. In this layer, we employ CoT reasoning with multiple temperature settings to generate diverse semantic interpretations. Inspired by the efficacy of CoT prompting in visual tasks like semantic segmentation [17], we then fuse these results to obtain more generalizable layout information. This method guides the generation process of indoor scenes by parsing the input text prompts into structured semantic descriptions and constructing a reasoning chain including spatial layout, object relationships and functional constraints.

## 2.1.2. Self-optimization Layer

Building upon DreamScene360 [10], we develop an iterative optimization pipeline using the Qwen MLLM. Our process begins with diffusion model [5] that generates an initial panorama from CoT reasoning text prompts. This panorama is analyzed by the Qwen model, which generates optimized prompts for subsequent iterations. Through multiple iterations, we progressively enhance both the semantic alignment and visual quality of the panorama. This iterative feedback mechanism aligns with recent methodologies in LLMdriven controlled image editing [18], which leverages LLMs to ensure precise visual refinement.

For each iteration, we employ diffusion process with pre-trained model Î¦ : $\tau \times \mathcal { P }  \mathcal { T } .$ , where $\dot { \mathcal { T } } = \mathbb { R } ^ { H \times W \times C }$ represents the image space and $p \in \mathcal P$ denotes a text prompt in the conditional space. The diffusion sample updates leverage the quadratic Least-Squares algorithm:

$$
\Psi ( J _ { t } | z ) = \sum _ { i = 1 } ^ { n } \frac { F _ { i } ( w _ { i } ) } { \sum _ { j = 1 } ^ { n } F _ { j } ( w _ { j } ) } \otimes F _ { i } ( \Phi ( I _ { t } ^ { i } | p _ { i } ) ) ,\tag{1}
$$

where $w _ { i }$ denotes the per-pixel weight, set to 1 in our implementation.

To create seamless 360Â° panoramas, we implement StitchDiffusion [19], which applys the diffusion process to stitched planes to ensure boundary consistency. We then extract the central region of size $H \times 2 H$ as the focal area for each optimization iteration.

This process yields optimal panoramas that we further enhance using Real-ESRGAN super-resolution techniques, resulting in significantly sharper and more detailed images, which we denote as $I _ { p }$

## 2.2. Panorama Sliding Scene Reconstruction

## 2.2.1. Panorama Sliding

To overcome feature truncation and perceptual blind spots in conventional approaches [9, 10], we propose a panorama sliding method that implements strategic perspective sampling. The view-dependent rotation matrix $R _ { i }$ systematically transforms viewing directions:

$$
\boldsymbol { I _ { i } } = \mathrm { g e t } _ { - \mathrm { p e r s p e c t i v e \_ i m a g e } } \left( \boldsymbol { I _ { p } } , \boldsymbol { R _ { i } } , \mathrm { F O V } = 9 0 ^ { \circ } \right)\tag{2}
$$

$$
\begin{array} { r } { R _ { i } = \left[ \begin{array} { c c c } { \cos \left( i \cdot \theta \right) } & { 0 } & { \sin \left( i \cdot \theta \right) } \\ { 0 } & { 1 } & { 0 } \\ { - \sin \left( i \cdot \theta \right) } & { 0 } & { \cos \left( i \cdot \theta \right) } \end{array} \right] , \quad \theta = \frac { \pi } { 1 5 } . } \end{array}
$$

The angular index $i \in [ 0 , 2 9 ]$ establishes a comprehensive $3 6 0 ^ { \circ }$ sampling at 12Â° intervals, with 50% overlap between adjacent perspectives ensuring cross-view consistency for geometrically coherent reconstruction.

## 2.2.2. 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) [12] encodes scene geometry through differentiable Gaussian primitives. Each primitive ${ \mathcal { G } } _ { n }$ is characterized by position $\mu _ { n } \in \mathbb { R } ^ { 3 }$ , chromatic attributes $c _ { n } \in \mathbb { R } ^ { 3 }$ , opacity $\alpha _ { n } \in \mathbb { R }$ and covariance $\Sigma _ { n } \in \mathbb { R } ^ { 3 \times 3 }$ :

$$
\begin{array} { r } { \mathcal G _ { n } \left( p , \alpha _ { n } , \Sigma _ { n } \right) = \alpha _ { n } e ^ { - \frac { 1 } { 2 } \left( p - \mu _ { n } \right) ^ { T } \Sigma _ { n } ^ { - 1 } \left( p - \mu _ { n } \right) } } \end{array}\tag{3}
$$

View-dependent rendering leverages spherical harmonic coefficients for color determination, with per-pixel color $C ( \boldsymbol p )$ computed through alpha compositing:

$$
C \left( { \boldsymbol { p } } \right) = \sum _ { i = 1 } ^ { m } c _ { i } \sigma _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \sigma _ { j } \right)\tag{4}
$$

where $\sigma _ { i } = 1 - e ^ { - \frac { \alpha _ { i } } { \sqrt { \operatorname* { d e t } ( \Sigma _ { i } ) } } }$ integrates opacity and covariance properties.

## 2.2.3. MASt3R-Based Point Cloud Initialization

To initialize accurate 3D scene structure, we employ MASt3R [20] framework for point cloud generation. This multi-view stereo approach uses confidence-weighted regression to handle uncertain predictions:

$$
\mathcal { L } _ { \mathrm { c o n f } } = \sum _ { v \in \{ 1 , \dots , N \} } \sum _ { p \in P ^ { v } } C _ { p } ^ { v , r } \ell _ { \mathrm { r e g r } } ( v , p ) - \beta \log C _ { p } ^ { v , r }\tag{5}
$$

where $\begin{array} { r } { \ell _ { \mathrm { r e g r } } ( v , p ) = \left. \frac { 1 } { z } X _ { p } ^ { v , r } - \frac { 1 } { \bar { z } } \bar { X } _ { p } ^ { v , r } \right. } \end{array}$ applies scale-invariant normalization to ensure consistent depth estimation across different views.

## 2.2.4. Gaussian Bundle Adjustment

To optimize the initial point cloud and achieve globally consistent reconstruction, we optimize both scene geometry (G) and camera poses (T ) through Gaussian Bundle Adjustment:

$$
G ^ { * } , T ^ { * } = \arg \operatorname* { m i n } _ { G , T } \sum _ { v \in N } \sum _ { i = 1 } ^ { H W } \Big \| \tilde { C } _ { v } ^ { i } ( G , T ) - C _ { v } ^ { i } ( G , T ) \Big \|\tag{6}
$$

where C denotes the rasterization operator and $\tilde { C }$ represents observed imagery. This joint optimization minimizes the rendering error across all viewpoints while maintaining geometric consistency.

## 2.3. Training Objectives

## 2.3.1. Semantic Similarity Distillation

To ensure semantic consistency across different views, we leverage the [CLS] tokens in the pre-trained DINOv2 [15] model to calculate the semantic similarity loss:

$$
\begin{array} { r } { \mathcal { L } _ { s e m } = 1 - \cos ( [ C L S ] ( I _ { i } ) , [ C L S ] ( I _ { i } ^ { \prime } ) ) } \end{array}\tag{7}
$$

where $I _ { i }$ represents the GT image and $\hat { I _ { i } }$ represents the rendered image from the i-th viewpoint. This loss guides the 3D Gaussian to fill geometric gaps in invisible regions. It also aligns with recent advances in multimodal 3D point cloud understanding [21, 22].

## 2.3.2. Geometric Consistency Constraint

To maintain geometric coherence, we introduce a geometric consistency constraint loss $\mathcal { L } _ { g e o }$ using DPT depth estimator:

$$
\mathcal { L } _ { g e o } ( \hat { I } _ { i } , D _ { i } ) = 1 - \frac { \mathrm { C o v } ( D _ { i } , \mathrm { D P T } ( \hat { I } _ { i } ) ) } { \sqrt { \mathrm { V a r } ( D _ { i } ) \mathrm { V a r } ( \mathrm { D P T } ( \hat { I } _ { i } ) ) } }\tag{8}
$$

where $\hat { I _ { i } }$ and $D _ { i }$ represent the rendered image and its depth map at the i-th viewpoint. This loss reduces depth discontinuities and ensures smooth geometric transitions.

## 2.3.3. Optimization

Our supervised optimization combines photometric, semantic, and geometric losses:

$$
\mathcal { L } = \mathcal { L } _ { R G B } + \lambda _ { 1 } \cdot \mathcal { L } _ { s e m } + \lambda _ { 2 } \cdot \mathcal { L } _ { g e o }\tag{9}
$$

where $\lambda _ { 1 } = 0 . 1$ and $\lambda _ { 2 } = 0 . 0 3$ balance the semantic and geometric components, ensuring both visual fidelity and structural accuracy.

## 3. EXPERIMENTS

## 3.1. Metrics and Implementation Details

To ensure comprehensive evaluation, we select CLIP-Distance [23] to verify semantic alignment with text prompts, Q-Align [24] to assess professional visual quality, and BRISQUE [25] to examine natural image statistics. For rendering quality, following established protocols, we adopt PSNR, SSIM, and LPIPS to evaluate the fidelity and perceptual quality of the rendering process. In our implementation, we boost image quality with adaptive contrast and color optimization, stabilize training with an optimized learning rate, and apply dynamically adaptive loss weights across the training process to enhance both stability and rendering quality. We also phase the adjustment of edge and consistency loss weights in depth loss to improve geometric structuring and rendering detail. All experiments are conducted on a $\mathrm { P y }$ Torch framework with an NVIDIA RTX 4090D GPU, using consistent loss functions and hyperparameters over 3,000 iterations.

## 3.2. Main Result

## 3.2.1. Panorama Generation

The comparison depicted in the Fig. 2 highlights the advantages of our method over the baseline approach. Our method generates a more realistic panoramic effect, characterized by richer textures and a natural spherical distortion that enhances the overall visual experience.

<!-- image-->  
Fig. 2. Sample panoramas generated by LucidDreamer, Dream-Scene360 and our method under identical prompts.

Table 1. Performance comparison of panorama generation methods. Our method shows the best scores, indicating superior quality.
<table><tr><td>Method</td><td>Clip-Distanceâ</td><td>Q-Alignâ</td><td>BRISQUEâ</td></tr><tr><td>LucidDreamer</td><td>0.7221</td><td>0.0317</td><td>33.0635</td></tr><tr><td>DreamScene360</td><td>0.7229</td><td>0.0269</td><td>31.1409</td></tr><tr><td>ours</td><td>0.7074</td><td>0.0318</td><td>27.6830</td></tr></table>

Quantitatively, as demonstrated in Table 1, our method leads in all three key indicators, underscoring its superior generation capabilities. Our methodâs superior performance in clip-distance metrics indicates that our layout reasoning layer and self-optimization layer effectively optimize the initial prompt. This results in images that are more semantically aligned with the intended outcomes. Furthermore, the significant improvement in BRISQUE scores suggests that our generated panoramas closely resemble natural scenes, boasting excellent structure and contrast. This comprehensive set of results underscores the visual and semantic excellence of our rendering method.

## 3.2.2. Rendering quality

To demonstrate the superiority of our rendering method, we present a qualitative comparison with the baseline method. The images rendered by our method are visibly clearer and exhibit higher fidelity, showcasing our methodâs enhanced rendering capabilities.

Table 2. Ablation study results showing the impact of different layer combinations on model performance. layer1 represents layout reasoning layer, and layer2 represents self-optimization layer
<table><tr><td>layer 1</td><td>layer2</td><td>clip-distanceâ</td><td>Q-Alignâ</td><td>BRISQUEâ</td></tr><tr><td>Ã</td><td>Ã</td><td>0.7313</td><td>0.0092</td><td>32.2608</td></tr><tr><td>Ã</td><td>â</td><td>0.7252</td><td>0.0084</td><td>30.3939</td></tr><tr><td>â</td><td>Ã</td><td>0.7318</td><td>0.0104</td><td>34.7502</td></tr><tr><td>â</td><td>â</td><td>0.7224</td><td>0.0105</td><td>29.2609</td></tr></table>

<!-- image-->  
Fig. 3. Qualitative comparison of rendering results between our method and baseline approaches. Our method produces images with higher fidelity, better geometric details across different viewpoints.

Table 3. Ablation study results showing the impact of different loss combinations on model performance.
<table><tr><td>Lgeo</td><td> $\mathcal { L } _ { s e m }$ </td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Ã</td><td>Ã</td><td>33.85</td><td>0.9630</td><td>0.0601</td></tr><tr><td>Ã</td><td>â</td><td>33.77</td><td>0.9624</td><td>0.0612</td></tr><tr><td>â</td><td>Ã</td><td>33.88</td><td>0.9631</td><td>0.0599</td></tr><tr><td>â</td><td>â</td><td>34.98</td><td>0.9707</td><td>0.0510</td></tr></table>

## 3.3. Ablation Study

To evaluate the impact of our design choices, we conducted ablation studies on both the two-layer architecture and the loss functions. As shown in Table 2, the combination of the layout reasoning and self-refinement layers yields the best overall results, with the former enhancing structural consistency and the latter refining perceptual quality. Likewise, Table 3 shows that jointly using Lgeo and Lsem leads to superior performance, as the two losses complement each other in ensuring geometric accuracy and semantic coherence.

## 4. CONCLUSION

In this work, we present PSGS, a novel two-stage framework for generating globally consistent 3D scenes from text descriptions. Our approach first introduces a two-layer panorama optimization architecture, combining layout reasoning for spatial semantics and iterative MLLM-based refinement for visual fidelity. The subsequent panorama sliding mechanism enables geometrically coherent point cloud initialization for 3D Gaussian Splatting, while our joint semantic-geometric consistency losses ensure structural accuracy and contextual alignment. Extensive validation demonstrates that PSGS significantly outperforms state-of-the-art methods in cross-view consistency and photorealism. This work provides an efficient, highfidelity solution for immersive content creation. Future directions include modeling complex compositional text inputs and integrating physics-aware illumination for dynamic scenes.

## 5. REFERENCES

[1] Minling Zhu, Yadong Gong, Chunwei Tian, and Zuyuan Zhu, âA systematic survey of transformer-based 3d object detection for autonomous driving: Methods, challenges and trends,â Drones, vol. 8, no. 8, pp. 412, 2024.

[2] Naofumi Akimoto, Yuhi Matsuo, and Yoshimitsu Aoki, âDiverse plausible 360-degree image outpainting for efficient 3dcg background creation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11441â11450.

[3] Chengkun Cai, Xu Zhao, Haoliang Liu, Zhongyu Jiang, Tianfang Zhang, Zongkai Wu, Jenq-Neng Hwang, and Lei Li, âThe Role of Deductive and Inductive Reasoning in Large Language Models,â in Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2025.

[4] Wenrui Li, Fucheng Cai, Yapeng Mi, Zhe Yang, Wangmeng Zuo, Xingtao Wang, and Xiaopeng Fan, âScenedreamer360: Text-driven 3d-consistent scene generation with panoramic gaussian splatting,â arXiv preprint arXiv:2408.13711, 2024.

[5] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer, âHigh-resolution image synthesis Â¨ with latent diffusion models,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684â10695.

[6] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall, âDreamfusion: Text-to-3d using 2d diffusion,â arXiv preprint arXiv:2209.14988, 2022.

[7] Xiaoyu Zhou, Xingjian Ran, Yajiao Xiong, Jinlin He, Zhiwei Lin, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang, âGala3d: Towards text-to-3d complex scene generation via layout-guided generative gaussian splatting,â arXiv preprint arXiv:2402.07207, 2024.

[8] Lukas Hollein, Ang Cao, Andrew Owens, Justin Johnson, Â¨ and Matthias NieÃner, âText2room: Extracting textured 3d meshes from 2d text-to-image models,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7909â7920.

[9] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee, âLuciddreamer: Domain-free generation of 3d gaussian splatting scenes,â arXiv preprint arXiv:2311.13384, 2023.

[10] Shijie Zhou, Zhiwen Fan, Dejia Xu, Haoran Chang, Pradyumna Chari, Tejas Bharadwaj, Suya You, Zhangyang Wang, and Achuta Kadambi, âDreamscene360: Unconstrained text-to-3d scene generation with panoramic gaussian splatting,â in European Conference on Computer Vision. Springer, 2024, pp. 324â342.

[11] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and Â¨ George Drettakis, â3d gaussian splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[13] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al., âChain-ofthought prompting elicits reasoning in large language models,â Advances in neural information processing systems, vol. 35, pp. 24824â24837, 2022.

[14] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun, âVi- Â´ sion transformers for dense prediction,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 12179â12188.

[15] Maxime Oquab, Timothee Darcet, Th Â´ eo Moutakanni, Huy Vo, Â´ Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al., âDinov2: Learning robust visual features without supervision,â arXiv preprint arXiv:2304.07193, 2023.

[16] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li, âT2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot,â arXiv preprint arXiv:2505.00703, 2025.

[17] Lei Li, âImage Semantic Segmentation via Chain-of-Thought Prompts,â in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024.

[18] Chengkun Cai, Haoliang Liu, Xu Zhao, and et al., âBayesian Optimization for Controlled Image Editing via LLMs,â in Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2025.

[19] Hai Wang, Xiaoyu Xiang, Yuchen Fan, and Jing-Hao Xue, âCustomizing 360-degree panoramas through text-to-image diffusion models,â in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 4933â4943.

[20] Vincent Leroy, Yohann Cabon, and JerÂ´ ome Revaud, âGrounding Ë image matching in 3d with mast3r,â in European Conference on Computer Vision. Springer, 2024, pp. 71â91.

[21] Zhaochong An, Guolei Sun, Yun Liu, and et al., âGeneralized few-shot 3d point cloud segmentation with vision-language model,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[22] Zhaochong An, Guolei Sun, Yun Liu, Runjia Li, Min Wu, Ming-Ming Cheng, Ender Konukoglu, and Serge Belongie, âMultimodality helps few-shot 3d point cloud semantic segmentation,â in ICLR, 2025.

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PmLR, 2021, pp. 8748â8763.

[24] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al., âQ-align: Teaching lmms for visual scoring via discrete text-defined levels,â arXiv preprint arXiv:2312.17090, 2023.

[25] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik, âNo-reference image quality assessment in the spatial domain,â IEEE Transactions on image processing, vol. 21, no. 12, pp. 4695â4708, 2012.