# Toward Fine-Grained Facial Control in 3D Talking Head Generation

Shaoyang Xie, Xiaofeng Cong, Baosheng Yu, Zhipeng Gui, Member, IEEE, Jie Gui, Senior Member, IEEE, Yuan Yan Tang, Life Fellow, IEEE, James Tin-Yau Kwok, Fellow, IEEE

AbstractâAudio-driven talking head generation is a core component of digital avatars, and 3D Gaussian Splatting has shown strong performance in real-time rendering of high-fidelity talking heads. However, achieving precise control over fine-grained facial movements remains a significant challenge, particularly due to lip-synchronization inaccuracies and facial jitter, both of which can contribute to the uncanny valley effect. To address these challenges, we propose Fine-Grained 3D Gaussian Splatting (FG-3DGS), a novel framework that enables temporally consistent and high-fidelity talking head generation. Our method introduces a frequency-aware disentanglement strategy to explicitly model facial regions based on their motion characteristics. Low-frequency regions, such as the cheeks, nose, and forehead, are jointly modeled using a standard MLP, while high-frequency regions, including the eyes and mouth, are captured separately using a dedicated network guided by facial area masks. The predicted motion dynamics, represented as Gaussian deltas, are applied to the static Gaussians to generate the final head frames, which are rendered via a rasterizer using frame-specific camera parameters. Additionally, a high-frequency-refined post-rendering alignment mechanism, learned from large-scale audioâvideo pairs by a pretrained model, is incorporated to enhance per-frame generation and achieve more accurate lip synchronization. Extensive experiments on widely used datasets for talking head generation demonstrate that our method outperforms recent state-of-the-art approaches in producing high-fidelity, lip-synced talking head videos.

Index TermsâTalking head generation, 3D Gaussian Splatting.

## I. INTRODUCTION

A UDIO-driven 3D talking head generation is emerging asa transformative technology at the forefront of digital human synthesis, with far-reaching applications in digital avatars [1], film and content production [2], and real-time interactive communication [3]. Its main research goal is to use algorithms to produce high-fidelity, lip-synchronized facial animations from speech signals [4], [5], which is crucial for creating believable, emotionally expressive digital humans. However, achieving this requires a delicate balance: accurately modeling speech-driven facial dynamics while preserving a consistent and identity-faithful appearance remains a major challenge for the field.

<!-- image-->  
Fig. 1. Illustration of lip-synchronization errors and facial jitter. The top row shows generated frames, while the bottom row shows the corresponding ground-truth frames. Arrows highlight noticeable discrepancies, demonstrating that existing methods suffer from lip-synchronization inaccuracies and unstable facial motions.

Generative Adversarial Networks (GANs) [6] have been extensively applied to talking head generation [7]â[9], achieving better lip-audio synchronization by leveraging large-scale audio-visual datasets [10], [11]. Despite these advances, GANbased methods often struggle to preserve speaker identity and produce high-resolution outputs. To overcome these limitations, recent studies have shown that incorporating 3D geometric priors is crucial [2], [12]â[17]. Inspired by the 3D Morphable Model (3DMM) [18], recent methods such as SadTalker [14] address these challenges by predicting 3DMM coefficients for pose and expression directly from audio. While this strategy enhances identity preservation and ensures geometric consistency, it often results in animations that appear stiff or lack emotional expressiveness, due to the limited representational power of the linear 3DMM basis.

The success of NeRF [19]â[21] in novel view synthesis has significantly advanced the generation of high-fidelity, 3D-consistent talking heads [22]â[26]. However, NeRF-based approaches are often limited by slow training and rendering speeds, making them less practical for real-time applications. To overcome these limitations, 3DGS [27]â[29] has recently emerged as a powerful alternative, offering both high-quality rendering and real-time performance. More importantly, 3DGS provides an explicit spatial representation, enabling finer control over motion and appearance, which is an essential capability for talking head generation [30]â[38]. For example, GaussianTalker [31] animates a canonical set of Gaussians using audio-driven deformations, achieving efficient and expressive results. Similarly, TalkingGaussian [32] employs a motion prediction network to estimate point-wise offsets for each Gaussian, facilitating dynamic and accurate head animation. Though the above methods exhibit strong overall consistency, they often struggle to accurately capture fine-grained facial motions, especially in high-frequency regions like the mouth and eyes, which change rapidly during speech. As illustrated in Fig. 1, these subtle inaccuracies can lead to an uncanny valley effect, diminishing the perceived realism of the generated avatars.

In this paper, we propose Fine-Grained 3D Gaussian Splatting (FG-3DGS) to address the limitations of existing audiodriven talking head generation methods, particularly their difficulty in modeling fine-grained facial motions. Facial regions exhibit heterogeneous motion patterns: high-frequency areas such as the eyes and mouth undergo rapid, complex movements (e.g., lip-synchronization changes and blinking), while low-frequency regions like the cheeks, forehead, and jaw primarily follow rigid head-pose movements and exhibit relatively minor changes. Moreover, simply employing a unified model to capture motion across all regions often fails to effectively capture these disparate dynamics, leading to biased training toward the dominant low-frequency areas. Consequently, these models tend to prioritize appearance over accurate motion representation, leading to unsmooth results, particularly at the boundaries between high- and lowfrequency zones. To overcome this issue, FG-3DGS introduces a frequency-aware disentanglement strategy that explicitly separates the modeling of high- and low-frequency regions. For high-frequency regions, we use specialized subnetworks that independently predict Gaussian deltas, enabling precise animation of the eyes and mouth. In contrast, a shared lightweight MLP is used to model the more rigid motions in low-frequency regions. This design enables region-specific learning of motion dynamics from audio signals and is conceptually aligned with the slow-fast network paradigm used in spatiotemporal modeling [39]. Furthermore, leveraging the differentiability of Gaussian splatting rasterization, we incorporate a high-frequencyrefined post-rendering alignment mechanism to guide the training process using consecutive frames, thereby enhancing temporal coherence and improving lip synchronization. By combining frequency-specific branches based on facial region masks, our method produces expressive, temporally consistent talking head animations. Our main contributions are summarized as follows:

â¢ We propose FG-3DGS, a novel framework for talkinghead generation that effectively captures heterogeneous motion across facial regions.

â¢ We introduce a frequency-aware disentanglement strategy that separately models low- and high-frequency regions, mitigating the dominance of appearance-based gradients in highly dynamic areas.

â¢ We develop a high-frequency-refined post-rendering alignment mechanism that utilizes a pre-trained lipsynchronization discriminator on the rendered outputs. This ensures precise synchronization between audio and high-frequency lip movements through fine-grained supervision.

## II. RELATED WORK

## A. Audio-Driven Talking Head Generation

Audio-driven talking head generation aims to produce videos of a target person that satisfy the dual goals of photorealism and precise audio-visual synchronization. Early methods primarily rely on 2D GANs to generate audio-synchronized mouth movements from a single image or video. For example, Wav2Lip [10] employs pre-trained lip-synchronization discriminators to align mouth movements with speech, achieving high synchronization quality. Although these approaches produce plausible results under constrained viewpoints, they struggle to generalize to dynamic head poses due to the inherent limitations of 2D representations. Subsequent efforts address these geometric limitations by integrating intermediate 3D priors. For instance, several 3D model-based methods [1], [14], [40], [41] utilize facial landmarks and 3DMM to disentangle pose and expression parameters, enabling more precise control of talking heads. While these pipelines improve pose controllability, their reliance on error-prone intermediate estimates, such as landmark detection inaccuracies, often compromises identity preservation, especially under extreme head rotations or challenging lighting conditions.

## B. NeRF-Based Talking Head Generation

NeRFs have substantially advanced audio-driven 3D talking head generation by enabling the representation of detailed 3D facial structures. Early NeRF-based methods, such as AD-NeRF [22], demonstrate the viability of audio-conditioned NeRF for high-quality facial animation via person-specific training. However, vanilla NeRF suffers from slow rendering speeds and high computational costs, limiting its practical application. Subsequent efforts focus on improving rendering efficiency while maintaining high quality. RAD-NeRF [23] first introduces grid-based NeRF to accelerate computation. ER-NeRF [24] reformulates the standard 3D hash grid into a tri-plane hash representation [42], which reduces hash collisions and achieves a better balance between quality and efficiency. To enhance multimodal fusion, GeneFace [43] and SyncTalk [44] leverage large-scale audio-visual datasets to pre-train dedicated audio encoders before NeRF rendering. Although NeRF-based methods achieve high reconstruction fidelity and multi-view consistency, addressing the entangled representations of rigid craniofacial geometry and achieving faster rendering speeds remain ongoing challenges.

## C. 3DGS-Based Talking Head Generation

Unlike NeRF, which represents 3D structures implicitly through neural networks, 3DGS [27] uses an explicit point cloud representation to construct the radiance field. This approach achieves much faster rendering and higher-quality results for multi-view static scene reconstruction. Although 3DGS was originally developed for static scenes, dynamic variants such as 4DGS [45] and Dynamic3DGS [46] have extended its use to reconstructing dynamic 3D scenes. GSTalker [47] is the first to apply deformable Gaussian splatting to audio-driven talking head generation, opening new possibilities for dynamic 3DGS applications. Building on this, GaussianTalker [31] employs a cross-modal attention mechanism to better fuse audio and spatial features, resulting in vivid and expressive facial animations. Additionally, TalkingGaussian [32] improves fine-grained speech motion by separating the modeling of intra-oral regions from other facial areas. However, current methods often ignore the role of eye movements, which are crucial for identity preservation and emotional expression. Likewise, existing 3DGS approaches often overlook lip-synchronization accuracy, leading to unnatural mouth motion.

<!-- image-->  
Fig. 2. The main proposed FG-3DGS framework for 3D talking head generation. Given a specific portrait speech video, FG-3DGS first decomposes the head into three regions: the face, eyes, and mouth. After performing static 3D Gaussian reconstruction, a conditional deformation attention mechanism predicts Gaussian offsets based on the encoded audio feature $f _ { a }$ and expression feature $f _ { e } .$ . The outputs from the static and dynamic components are then combined, and a 3D Gaussian rasterizer renders the dynamic Gaussians into images under varying camera parameters. To enhance lip synchronization, the high-frequency refined post-rendering alignment is applied in the final stage.

Generating realistic talking heads hinges not just on visual quality, but on the faithful reproduction of complex, non-uniform facial dynamics. Although previous NeRF-based methods [22]â[24], [48], [49] and 3D Gaussian-based methods [31], [47], [50] have made a great contribution to the highquality talking head synthesis, they treat the talking head as a single integral entity, learning a direct mapping from audio signals to complete facial representation. These approaches often yield overly-smoothed or averaged facial dynamics, which contradict the true biomechanics of the human face: different facial regions move in an unsmooth manner and exhibit distinct motion characteristics. This often results in a rigid coupling of facial regions, leading to the uncanny valley effect.

## III. METHOD

In this section, we introduce the proposed FG-3DGS framework for talking head generation. We first describe the construction of static Gaussians, following the previous method [27]. After that, we describe the frequency-aware disentanglement and modeling of different regions. Next, we introduce a high-frequency-refined post-rendering alignment mechanism to synchronize lip movement and audio under supervision. Fig. 2 illustrates the main proposed FG-3DGS framework for talking head generation.

## A. Static Gaussian Reconstruction

To disentangle the frequency-aware attributes within the 3D Gaussian space, we reconstruct a static head Gaussian model in the canonical space by 3D Gaussian Splatting (3DGS) [27]. A collection of 3D Gaussians ${ \mathcal { G } } ~ = ~ \{ G _ { 1 } , G _ { 2 } , \ldots , G _ { i } , \ldots \}$ is employed to represent 3D structures, where i denotes the index. Each Gaussian primitive is defined by a set of optimizable attributes, which include a 3D mean position $\pmb { \mu } _ { i } \in \mathbb { R } ^ { 3 }$ , a positive semi-definite covariance matrix $\ b { \Sigma } _ { i } ~ \in ~ \mathbb { R } ^ { 3 \times 3 }$ , an opacity value $\alpha _ { i } \in \mathbb { R } .$ , and its color represented by Spherical Harmonics (SH) coefficients $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 \times 1 6 }$ . To ensure positive semi-definiteness and to control the rotation and scaling of each 3D Gaussian primitive, its covariance matrix is decomposed into a scaling matrix S and a rotation matrix R by

$$
\Sigma = R S S ^ { \top } R ^ { \top } ,\tag{1}
$$

where R and S are defined by a scaling factor $s \in \ \mathbb { R } ^ { 3 }$ and a rotation quaternion $r ~ \in ~ \mathbb { R } ^ { 4 }$ , respectively. Therefore, the complete parameters of the i-th Gaussian primitive $G _ { i }$ is defined as

$$
G _ { i } = \{ \mu _ { i } , r _ { i } , s _ { i } , \mathbf { c } _ { i } , \alpha _ { i } \} ,\tag{2}
$$

Following the above procedure, a complete static 3D Gaussian representation $\mathcal { G } ~ = ~ \{ \mu , \mathbf { r } , \mathbf { s } , \mathbf { c } , \alpha \}$ can be constructed.

Then, we can render a coarse static head of a specific person from a few-minute portrait video. Sharing the similar problem settings with previous NeRF-based methods [22]â[24], we track the head pose and inversely calculate the camera pose $\{ \pi _ { i } \} _ { i = 1 } ^ { N }$ from a few-minute portrait video $V ~ = ~ \{ I _ { n } \bar  \} _ { n = 1 } ^ { N }$ of a specific person to reconstruct different scene, where $N$ represents the length of frames. The pixel-wise L1 loss and the D-SSIM loss [51] $\mathcal { L } _ { D S }$ are employed to measure the difference between the mask ground-truth image $x ^ { m s }$ and the static rendered image $x ^ { s t }$ . The loss function for the static Gaussian reconstruction is

$$
\mathcal { L } _ { s t } = \mathcal { L } _ { 1 } ( x ^ { m s } , x ^ { s t } ) + \lambda _ { 1 } \mathcal { L } _ { D S } ( x ^ { m s } , x ^ { s t } ) ,\tag{3}
$$

where $\lambda _ { 1 }$ denotes the weight factors. Next, the static Gaussians $G _ { r }$ are jointly optimized with the frequency-aware modeling.

## B. Frequency-Aware Disentanglement

Previous methods [31], [47], [50] have made an impressive contribution to the high-quality talking head synthesis. They treat the talking head as a single integral entity, but this often results in a rigid coupling of facial regions, leading to the uncanny valley effect. In contrast to the previous method, we leverage the fine-grained facial region disentanglement, modeling the talking head as a collection of regions with different frequency characteristics namely: (i) the mouth region $\mathcal { G } _ { m }$ and eye region $\mathcal { G } _ { e }$ with high-frequency movements that are tightly correlated with the input audio, and (ii) the face region $\mathcal { G } _ { f }$ with low-frequency exhibit relatively minor changes driven a combination of both audio input and expression signals suggested by the upper face movement.

To parse the talking head into separate parts, we used the settings from TalkingGaussian. A semantic mask generated by two face parsing models is used to divide the entire face into distinct regions. First, the BiSeNet [52] parser, pretrained on the CelebAMask-HQ dataset [53], is used to predict a coarse mask of the high-frequency regions: eyes and mouth. Given the possible domain gap introduced by the parser, it may fail to cover the entire mouth in some âclosedâ cases. The other face parse model, a ResNet-based FPN pretrained on Easyportrait [54], is introduced to generate the tooth mask. Then the two masks are overlaid to obtain the finer one. $\Psi _ { x _ { i } } ^ { r }$ denotes the mask for the j-th frame $x _ { j }$ in the r-th region. The static Gaussian for each region can be obtained by

$$
\mathcal { G } _ { r } = \Phi _ { G } \big ( \{ \Psi _ { x _ { 0 } } ^ { r } \odot x _ { 0 } , . . . , \Psi _ { x _ { j } } ^ { r } \odot x _ { j } \} , \{ \pi _ { 0 } , . . . , \pi _ { j } \} \big ) ,\tag{4}
$$

where $r \in \{ f , m , e \} , \Phi _ { G }$ represents the Gaussian initialization process. The â denotes the pixel-wise product.

## C. Frequency-Aware Modeling

After establishing the static Gaussian head $\mathcal { G } _ { r }$ , we aim to learn the connection between the input audio a and the motion offset $\Delta G _ { r }$ of the three regions $G _ { r }$ . Recognizing that different facial regions exhibit different frequency characteristics, we employ a jointly emotion- and audio-driven motion prediction network for the facial region $\mathcal { G } _ { f }$ and a gated cross-modal motion prediction network for the eye region $\mathcal { G } _ { e }$ and mouth region $\mathcal { G } _ { m }$ . The efficient triplane plane hash encoder [24], [32] $\mathcal { H } _ { 3 }$ is adopted to encode the three-dimensional 3D Gaussian position $\mu$ into a multi-resolution representation. The modeling process for low-frequency and high-frequency regions is as follows.

Low-Frequency Region. Typically, the movements associated with the low-frequency region represent coarse-grained structural dynamics that remain relatively invariant to transient speech signals or subtle, rapid head repositioning. Given this inherent stability, employing a high-capacity or overly complex motion prediction network may introduce the risk of overfitting, leading the model to capture idiosyncratic noise rather than meaningful motion patterns. To mitigate this, we propose a streamlined, lightweight architecture that achieves an optimal balance between expressive sufficiency and computational efficiency. Specifically, for the 3D Gaussians residing within this region, we predict their corresponding motion offsets through a jointly emotion- and audio-driven framework. This framework is underpinned by a multi-layer perceptron (MLP) network that integrates a regional attention mechanism with a strategic feature concatenation approach. To initiate the process, we leverage the robust representative power of pre-trained models. Specifically, we adopt an audio-speech extractor [55], [56] and a dedicated emotion extractor to derive raw frequency-domain information a and emotional latent information e, respectively. As illustrated in Fig. 2, these raw inputs a and e are further transformed by learnable encoding networks $\tau _ { a }$ and $\tau _ { e }$ to generate refined audio features $f _ { a }$ and expression-rich features $f _ { e }$ . This encoding process is formally defined as

$$
{ \bf f } _ { a } = \tau _ { a } ( a ) , { \bf f } _ { e } = \tau _ { e } ( e ) .\tag{5}
$$

To incorporate spatial context, we utilize the encoded 3D Gaussian positions $h _ { f } ~ = ~ \mathcal { H } _ { 3 } ( \mu _ { f } )$ derived from the hash encoding. These spatial features are processed by dedicated MLPs to generate region-specific modulation weights

$$
{ \bf s } _ { a } = \Gamma _ { s p } ^ { a } ( h _ { f } ) , { \bf s } _ { e } = \Gamma _ { s p } ^ { e } ( h _ { f } ) ,\tag{6}
$$

where $\Gamma _ { s p } ^ { a }$ and $\Gamma _ { s p } ^ { e }$ denote the spatial MLPs for audio and expression modalities, respectively. The resulting spatial weights, ${ \bf s } _ { a }$ and $\mathbf { s } _ { e } ,$ , act as an attention mechanism, determining the localized influence of audio and emotional signals at different coordinates within the residual facial region. By applying this spatial gating, the model can adaptively prioritize relevant features for motion synthesis. The final integrated feature representation $\mathbf { z } ,$ which combines the spatial backbone with the modulated audio and emotion streams, is obtained through an adaptive fusion process:

$$
\mathbf { z } = [ \mathcal { H } _ { 3 } ( \mu _ { f } ) \oplus ( \mathbf { s } _ { a } \odot \mathbf { f } _ { a } ) \oplus ( \mathbf { s } _ { e } \odot \mathbf { f } _ { e } ) ] ,\tag{7}
$$

where â represents the concatenation operator and $\odot$ denotes the element-wise Hadamard product. This comprehensive representation z is subsequently fed into a deformation MLP, $\Gamma _ { \mathrm { d e } } ^ { f }$ , which is optimized to regress the dynamic offsets for the Gaussian attributes. This final step is formulated as

$$
\Delta G _ { f } = \Gamma _ { \mathrm { d e } } ^ { f } ( \mathbf { z } ) ,\tag{8}
$$

where $\Delta G _ { f } = \{ \Delta \mu _ { f } , \Delta s _ { f } , \Delta r _ { f } \}$ encompasses the predicted motion offsets for the 3D Gaussian parameters, including position, scale, and rotation, thereby enabling realistic and synchronized facial dynamics.

High-Frequency Regions. In contrast to the relatively stable facial areas, the motions of the mouth and eye regions are characterized by high-frequency dynamics and intricate local deformations. In these regions, the mouth movements are primarily and directly driven by temporal audio signals, while eye movements are predominantly governed by expression latent signals. Due to the non-linear complexity of these motions, a simple linear or shallow network fails to accurately model such rapid transitions, often leading to visual artifacts such as motion jitter, stiff transitions, or muted expressions that lack emotional depth. To effectively bridge and fuse multiresolution spatial geometry with heterogeneous conditional information for these highly expressive regions, we introduce a gated cross-modal motion prediction network. The process begins by extracting localized geometric context. The spatial features of the mouth and eye regions are obtained by

$$
\begin{array} { r } { h _ { m } = \mathcal { H } _ { 3 } ( \mu _ { m } ) , \quad h _ { e } = \mathcal { H } _ { 3 } ( \mu _ { e } ) . } \end{array}\tag{9}
$$

To achieve precise synchronization, we merge the canonical 3D Gaussians with dynamic conditional features through a cross-attention mechanism: audio features $\mathbf { f } _ { a }$ drive the rapid mouth movements, while expression features $\mathbf { f } _ { e }$ control the subtle eye motions. This integration effectively captures how varying input conditions influence the underlying Gaussian motion patterns across different facial topologies. Furthermore, a lightweight gating MLP, $\Gamma _ { g a } ,$ is adopted, which takes the spatial context $( \mathbf { f } _ { a } \ \mathrm { o r } \ \mathbf { f } _ { e } )$ as input to predict a learnable scalar gating value. This formulation offers a profound level of fine-grained control: the cross-modal attention block spatially aligns temporal driving signals with static facial geometry, while the parallel spatial gate dynamically filters and scales the intensity of motion based on the specific geometric location where it occurs. The scalar gating value for adaptive modulation is obtained by

$$
\lambda _ { r } = \sigma ( \Gamma _ { g a } ^ { r } ( \mathbf { f } _ { r } ) ) , \quad \mathbf { f } _ { r } \in \mathbf { f } _ { a } , \mathbf { f } _ { e } ,\tag{10}
$$

where $\sigma$ is the sigmoid function, constraining $\lambda _ { r } ~ \in ~ ( 0 , 1 )$ with r denoting the region index. The cross-modal prediction network consists of a cross-modal attention block, $\phi _ { c m } ,$ and multiple feed-forward layers, $\varphi _ { f f }$ , interconnected via skip connections to ensure stable gradient flow. The computational pipeline for the prediction network is defined as:

$$
\begin{array} { r l } & { \mathbf { z } _ { r } = \lambda _ { r } \odot \phi _ { c m } ( \mathbf { f } _ { r } , \mathbf { c } _ { r } ) , } \\ & { \mathbf { z } _ { r } ^ { \prime } = \boldsymbol { \varphi } ^ { \prime } { } _ { f f } ( \mathbf { f } _ { r } + \mathbf { z } _ { r } ) , } \\ & { \mathbf { z } _ { r } ^ { \prime \prime } = \boldsymbol { \varphi } ^ { \prime \prime } { } _ { f f } ( \mathbf { z } _ { r } ^ { \prime } + \mathbf { z } _ { r } ) . } \end{array}\tag{11}
$$

In this workflow, the condition and spatial features are first fused in the cross-modal attention block. Then, the resulting fused feature $\mathbf { z } _ { r }$ is modulated element-wise by the spatial gate and processed through the feed-forward network (FFN) with dual residual connections to iteratively refine and extract the final motion information $\mathbf { z } _ { r } ^ { \prime \prime } .$ . This final fused feature $ { \mathbf { z } } _ { r } ^ { \prime \prime }$ is ultimately passed to the deformation MLP $\Gamma _ { d e } ^ { r }$ , which is formulated as

$$
\Delta G _ { r } = \Gamma _ { \mathrm { d e } } ^ { r } ( \mathbf { z } _ { r } ^ { \prime \prime } ) = \Delta \mu _ { r } ,\tag{12}
$$

where $\Delta G _ { r } \ \in \ \{ \Delta G _ { m } , \Delta G _ { e } \}$ represents the motion offset $\Delta \mu _ { r }$ of the 3D Gaussian parameters, specifically targeting the displacement of means to capture high-frequency facial articulations. The loss function of the overall Frequency-Aware Modeling stage is formulated as:

$$
\mathcal { L } _ { F M } = \mathcal { L } _ { 1 } ( x ^ { m s } , x ^ { d y } ) + \lambda _ { 1 } \mathcal { L } _ { D S } ( x ^ { m s } , x ^ { d y } ) ,\tag{13}
$$

where $x ^ { d y }$ represents the dynamic rendered face.

Fuse & Render. To synthesize the final talking head sequence, the discrete renderings from the previously defined low-frequency and high-frequency regions must be seamlessly integrated into a unified image space. We implement the alpha-blending principles widely used in image processing to aggregate the regional outputs. This fusion process is formulated as:

$$
\begin{array} { r l } & { \mathcal { C } _ { f u s e } = [ ( 1 - \alpha _ { m } ) \times ( \mathcal { C } _ { e } + \Delta \mathcal { C } _ { e } ) + \alpha _ { m } \times ( \mathcal { C } _ { m } + \Delta \mathcal { C } _ { m } ) ] } \\ & { \quad \quad \quad \times \left( 1 - \alpha _ { f } \right) + ( \mathcal { C } _ { f } + \Delta \mathcal { C } _ { f } ) \times \alpha _ { f } , } \end{array}\tag{14}
$$

where $\alpha _ { f }$ and $\{ \alpha _ { e } , \alpha _ { m } \}$ are the opacity from the lowfrequency and high-frequency regions, C represents the pixel color. During the fusion stage, apart from the reconstruction loss, we randomly cut patches from the images and manipulate the LPIPS loss $\mathcal { L } _ { L P }$ [24], [32], [57] to improve the details of the generated images. The overall loss function between the fusion and rendered image $x ^ { f u }$ and x is

$$
\mathcal { L } _ { f u } = \mathcal { L } _ { 1 } ( x , x ^ { f u } ) + \lambda _ { 1 } \mathcal { L } _ { D S } ( x , x ^ { f u } ) + \lambda _ { 2 } \mathcal { L } _ { L P } ( x , x ^ { f u } ) ,
$$

where $\lambda _ { 2 }$ is the weight factor.

(15)

## D. High-Frequency-Refined Post-Rendering Alignment

To enhance lip synchronization in 3D Gaussian reconstruction, we introduce a lip-synchronization discriminator [10] to compute the lip-synchronization loss $\mathcal { L } _ { l i p }$ . The discriminator is pre-trained on a large-scale dataset of audio-image pairs, enabling it to assess synchronization discrepancies between lip motion and audio at the feature level. By incorporating the raw audio signal as supervision, the discriminator facilitates crossmodal fusion between visual and audio representations. During this stage, we optimize only the mouth prediction network and color parameters to ensure stable fine-tuning, i.e., the overall loss function is

$$
L = \mathcal { L } _ { f u } + \lambda _ { 3 } \mathcal { L } _ { l i p } ( x ^ { f u } , a ) ,\tag{16}
$$

where a is the input audio and $\lambda _ { 3 }$ denotes the weight factor.

## IV. EXPERIMENTS

## A. Experimental Setups

Datasets. The experiments utilize high-definition speaking portrait videos sourced from publicly-released datasets that are well-established in prior works [23], [24], [43]. Each subjectâs dataset consists of several minutes of video of

<!-- image-->  
Ground Truth

ER-NeRF

<!-- image-->  
Fig. 3. Qualitative comparison of talking head synthesis across different methods. From top to bottom, rows show the ground truth and results produced by GeneFace, ER-NeRF, TalkingGaussian, and the proposed method. Close-up views highlight lip movements and fine facial details. The proposed method produces more accurate lip synchronization and more stable facial details, closely matching the ground truth. Zooming in is recommended for better visualization.

TABLE I  
THE QUANTITATIVE RESULTS OF THE TALKING HEAD RECONSTRUCTION. THE BEST PERFORMANCES ARE IN BOLDFACE, AND THE UNDERLINE REPRESENTS THE SECOND-BEST PERFORMANCE.
<table><tr><td>Method Ground Truth</td><td>PSNR â N/A</td><td>LPIPS â 0</td><td>FID â 0</td><td>LMD â 0</td><td>LSE-C â 8.275</td><td>Time â -</td><td>FPS â I</td></tr><tr><td>TalkLip [8]</td><td>32.52</td><td>0.0782</td><td>18.500</td><td>5.861</td><td>5.947</td><td>-</td><td>3.41</td></tr><tr><td>DINet [11]</td><td>31.65</td><td>0.0443</td><td>9.430</td><td>4.373</td><td>6.565</td><td></td><td>23.74</td></tr><tr><td>AD-NeRF [22]</td><td>26.73</td><td>0.1536</td><td>28.986</td><td>3.000</td><td>4.500</td><td>16.4h</td><td>0.14</td></tr><tr><td>RAD-NeRF [23]</td><td>31.78</td><td>0.0778</td><td>8.657</td><td>2.912</td><td>5.522</td><td>5.2h</td><td>53.87</td></tr><tr><td>GeneFace [43]</td><td>24.82</td><td>0.1178</td><td>21.708</td><td>4.286</td><td>5.195</td><td>12.3h</td><td>7.79</td></tr><tr><td>ER-NeRF [24]</td><td>32.52</td><td>0.0334</td><td>5.294</td><td>2.814</td><td>5.775</td><td>3.1h</td><td>55.41</td></tr><tr><td>TalkingGaussian [32]</td><td>32.40</td><td>0.0355</td><td>7.693</td><td>2.967</td><td>6.516</td><td>1h</td><td>90</td></tr><tr><td>PointTalk [58]</td><td>32.77</td><td>0.0337</td><td>7.331</td><td>2.818</td><td>7.165</td><td>1h</td><td>90</td></tr><tr><td>FG-3DGS (Ours)</td><td>33.06</td><td>0.0252</td><td>4.846</td><td>2.620</td><td>6.260</td><td>2h</td><td>90</td></tr></table>

people speaking different languages, with a corresponding audio track, averaging approximately 6,500 frames per clip at 25 FPS. To ensure a fair and reproducible comparison, we follow the preprocessing procedures of previous works: the raw videos are cropped to focus on the central portrait and then resized. The majority of the videos are set to a resolution of 512 Ã 512, except for the âObamaâ video, which is resized to 450 Ã 450. All the videos are divided into training and test sets with a 10:1 ratio.

Baseline Methods. To validate the effectiveness of the proposed talking head model, the comparative methods included 2D-based, NeRF-based, and 3D Gaussian-based approaches. (i) 2D based methods include TalkLip [8] and DINet [11], (ii) NeRF-based methods include AD-NeRF [22],

RAD-NeRF [23], ER-NeRF [24], and GeneFace [43], and (iii) 3D Gaussian-based methods TalkingGaussian [32] and PointTalk [58].

Training Details. For each subject, we train both the low-frequency and high-frequency regions for approximately 50,000 iterations. The training process is divided into two stages: 3,000 iterations for Static Gaussian Reconstruction and the remaining 47,000 for Frequency-Aware Modeling. Then, 10,000 more iterations are set for the joint Fusion & Render stage. The hyperparameter settings for the first stage are inherited from 3DGS [27]. While the learning rate of the jointly emotion- and audio-driven network in the lowfrequency region and the cross-modal attention block in the high-frequency region during the second and fusion stage is set to 1e â 4, while the learning rate of all other MLP networks is set to 1e â 5. All experiments are conducted on a single NVIDIA RTX 4090 GPU, and each person-specific model requires approximately two hours of training.

<!-- image-->  
Fig. 4. User study results. Mean scores from 20 participants on a 5- point scale, where higher values indicate better performance. The evaluation covers image quality, video realism, and lip synchronization. Methods AâF correspond to TalkLip, DINet, ER-NeRF, GeneFace, TalkingGaussian, and the proposed method (FG-3DGS), respectively.

## B. Quantitative Evaluation

For the quantitative evaluation, we conduct a comprehensive assessment of our proposed method across two distinct experimental settings: 1) the talking head reconstruction setting and 2) the cross-subject lip synchronization setting. In the talking head reconstruction setting, we aim to evaluate the modelâs fidelity in reconstructing the appearance and motion of a specific subject driven by their original voice. In the cross-subject lip-synchronization setting, we focus on assessing the modelâs generalization performance when presented with out-of-distribution (OOD) audio inputs. For this purpose, we employ two standard audio clips, designated as Audio A and Audio B, sourced from public benchmarks established in prior work [59], following the settings of ER-NeRF [24]. Notably, both audio clips originate from speakers other than the test target subjects, making generalization challenging.

Metrics. For the evaluation of the image quality, we utilize Peak Signal-to-Noise Ratio (PSNR) to measure overall reconstruction fidelity, the Learned Perceptual Image Patch Similarity (LPIPS) [57] to evaluate the realism of high-frequency details, and the Frechet Inception Distance (FID) [60] to judge the image quality from the feature aspect. To evaluate lip synchronization and motion accuracy, we adopt the Landmark Distance (LMD) [61], which computes the direct Euclidean distance between generated and groundtruth lip landmarks. Furthermore, we leverage the pre-trained SyncNet [62] [63] to compute its lip synchronization estimation confidence score (LSE-C) and error distance (LSE-D), following the evaluation protocol of Wav2Lip [10].

Results. Table I shows the quantitative results of the talking head reconstruction experiments. Among all the methods, our FG-3DGS ranks highest in most metrics. Due to the frequency-aware disentanglement and modeling strategy, our method achieves the highest PSNR and LPIPS, along with the lowest FID and LMD, demonstrating its ability to reconstruct image details while maintaining the highest render speed. Quantitative evaluation results demonstrate that the proposed FG-3DGS achieves the best overall performance on the talking head task studied in this paper.

TABLE II  
THE QUANTITATIVE RESULTS OF THE LIP SYNCHRONIZATION. WE UTILIZE TWO DIFFERENT AUDIO SAMPLES TO DRIVE THE SAME SUBJECT. THE BOLDFACE INDICATES THE BEST PERFORMANCE, AND THE UNDERLINE INDICATES THE SECOND-BEST.
<table><tr><td></td><td colspan="2">Audio A</td><td colspan="2">Audio B</td></tr><tr><td>Method Ground Truth</td><td>LSE-D â 6.899</td><td>LSE-C â 7.354</td><td>LSE-D â 7.322</td><td>LSE-C â 8.682</td></tr><tr><td>DINet [11]</td><td>8.503</td><td>5.696</td><td>8.204</td><td>5.113</td></tr><tr><td>AD-NeRF [22]</td><td>14.432</td><td>1.274</td><td>13.896</td><td>1.877</td></tr><tr><td>RAD-NeRF [23]</td><td>11.639</td><td>1.941</td><td>11.082</td><td>3.135</td></tr><tr><td>GeneFace [43]</td><td>9.545</td><td>4.293</td><td>9.668</td><td>3.734</td></tr><tr><td>ER-NeRF [24]</td><td>11.813</td><td>2.408</td><td>10.734</td><td>3.024</td></tr><tr><td>TalkingGaussian [32]</td><td>9.171</td><td>5.327</td><td>9.061</td><td>5.745</td></tr><tr><td>FG-3DGS (Ours)</td><td>8.987</td><td>6.197</td><td>9.039</td><td>6.317</td></tr></table>

<!-- image-->  
Fig. 5. Qualitative results of the ablation study on the proposed components. Each column shows the output obtained by removing one module (w/o FAD, w/o FAM, w/o HRPA) compared with the ground truth and the full model (Ours). Red bounding boxes highlight regions with visible artifacts and degradation, such as over-smoothing and inaccurate lip-synchronization.

Meanwhile, Table II reports the quantitative results of the cross-subject lip synchronization by different driven audios. Our proposed method achieves the best LSE-C performance among all competitors, indicating superior lip-synchronization confidence and semantic alignment. Notably, while DINet (a 2D-based method) yields a slightly lower LSE-D, our approach significantly outperforms all 3D-based and NeRF-based frameworks (e.g., GeneFace, TalkingGaussian) by a large margin in both metrics. The competitive LSE-D, coupled with the stateof-the-art LSE-C, demonstrates that our method effectively bridges the gap between 3D controllability and high-fidelity lip synchronization, maintaining high-quality lip alignment while overcoming the typical limitations of 3D facial animation in capturing precise lip dynamics.

## C. Qualitative Evaluation

To more intuitively evaluate the performance of the generated talking heads, we present keyframes from our talking head reconstruction experiments. We mainly focus on the portraitâs reconstruction details and lip-synchronization effect. As shown in Fig. 3, when other methods fail to capture the detailed appearance of the mouth and eye region, FG-3DGS generates a high-quality talking face. Moreover, our method maintains high fidelity and lip-synchronization accuracy in sequential frames, as shown in the left part of Fig. 3.

To complement our quantitative metrics, we conduct a user study. We recruited 20 participants and presented them with 36 talking-head videos generated by our method and five others. Participants were asked to rate each video on a 5-point scale according to three distinct criteria: (1) Image Quality (clarity, texture, and fidelity), (2) Video Realness (naturalness of motion and expressions), and (3) Lip-Synchronization (the consistency between lip and the audio). The scores are summarized in Fig. 4, indicating that FG-3DGS achieved the highest ratings across all three categories, outperforming all baseline methods.

## D. Ablation Study and Discussion

This section primarily analyzes the components and configurations of the experimental design, including the proposed frequency decomposition strategy, the audio extractor, the selection of motion offsets, and a discussion of hyperparameter sensitivity. The details are as follows.

Proposed Components. We conduct an ablation study across different settings to demonstrate that our fine-grained strategy makes a significant contribution to high-quality talking head generation. The results in Table III. show that without Frequency-Aware Disentanglement (FAD), the total reconstruction metrics decline slightly, indicating that the finegrained strategy helps control facial details. The degradation in LPIPS and PSNR metrics is attributed to the absence of Frequency-Aware Modeling (FAM), which reduces the degree of fusion between audio and spatial features, thereby lowering the overall quality of the generated image. The lack of the High-frequency Refined Post-rendering Alignment (HRPA) results in reduced synchronization between the lips and the audio. These results demonstrate the efficacy of our proposed components. Meanwhile, the visual results in Fig. 5 further validate the effectiveness of the proposed components.

Audio Extractor. To ensure a fair comparison with existing state-of-the-art talking head methods, we primarily use Deepspeech [55] as the audio feature extractor. To further demonstrate the generalizability and performance upper bound of our proposed model, we also evaluate its performance when integrated with more sophisticated extractors. As quantitatively shown in Table IV, substituting the baseline with HuBERT [56] or Wav2Vec 2.0 [64] leads to consistent performance improvements. This outcome indicates that our model can effectively leverage high-quality phonetic representations to facilitate finer lip-audio synchronization and more realistic speech-driven dynamics.

TABLE III  
ABLATION STUDY OF PROPOSED COMPONENTS. BEST PERFORMANCE IS SHOWN IN BOLD.
<table><tr><td>Setting</td><td>PSNRâ</td><td>LPIPSâ</td><td>FIDâ</td><td>LMDâ</td><td>LSE-Câ</td></tr><tr><td>w/o FAD</td><td>32.93</td><td>0.0275</td><td>5.396</td><td>2.719</td><td>6.111</td></tr><tr><td>w/o FAM</td><td>32.86</td><td>0.0289</td><td>7.150</td><td>2.854</td><td>5.965</td></tr><tr><td>w/o HRPA</td><td>32.89</td><td>0.0261</td><td>6.715</td><td>2.773</td><td>5.950</td></tr><tr><td>FG-3DGS</td><td>33.06</td><td>0.0252</td><td>4.846</td><td>2.620</td><td>6.260</td></tr></table>

TABLE IV

QUANTITATIVE METRICS FOR ADOPTING DIFFERENT AUDIO EXTRACTORS IN THE TALKING HEAD RECONSTRUCTION SETTING.
<table><tr><td>Extractor</td><td>PSNRâ</td><td>LPIPSâ</td><td>FIDâ</td><td>LMDâ</td><td>LSE-Câ</td></tr><tr><td>Deepspeech</td><td>33.06</td><td>0.0252</td><td>4.846</td><td>2.645</td><td>6.260</td></tr><tr><td>HuBERT</td><td>33.05</td><td>0.0251</td><td>4.903</td><td>2.643</td><td>6.401</td></tr><tr><td>Wav2Vec 2.0</td><td>33.03</td><td>0.0254</td><td>4.961</td><td>2.642</td><td>6.269</td></tr></table>

Motion Offset. We extensively conduct experiments to explore how various selections of motion parameter subsets influence the final synthesis quality. Fig. 6 presents a visualization of the comparative results. Our empirical findings suggest that when simultaneously predicting the set of Gaussian attributes $\{ \Delta \mu , \Delta r , \Delta s \}$ in the high-frequency zones, the model tends to suffer from optimization instability. This overparameterization often leads to perceptible over-smoothing of fine details and significantly reduces lip-synchronization accuracy. Consequently, we select only the position offset $\Delta \mu$ as the target for our motion prediction network in high-frequency regions to ensure robust motion synthesis.

TABLE V  
QUANTITATIVE METRICS UNDER VARYING HYPERPARAMETERS. THE DEFAULT SETTINGS ARE Î»1 = 0.20, Î»2 = 0.50, AND $\lambda _ { 3 } = 0 . 0 3 .$
<table><tr><td>Setting</td><td>PSNR â</td><td>LPIPS â</td><td>FID â</td><td>LMD â</td><td>LSE-C â</td></tr><tr><td> $\lambda _ { 1 } = 0 . 1 0$ </td><td>33.03</td><td>0.0263</td><td>5.052</td><td>2.685</td><td>6.099</td></tr><tr><td> $\lambda _ { 1 } = 0 . 1 5$ </td><td>33.08</td><td>0.0256</td><td>4.946</td><td>2.623</td><td>6.233</td></tr><tr><td> $\lambda _ { 1 } = 0 . 2 0$ </td><td>33.06</td><td>0.0252</td><td>4.846</td><td>2.620</td><td>6.260</td></tr><tr><td> $\lambda _ { 1 } = 0 . 3 0$ </td><td>32.99</td><td>0.0251</td><td>5.101</td><td>2.641</td><td>5.987</td></tr><tr><td> $\lambda _ { 2 } = 0 . 4 0$ </td><td>32.94</td><td>0.0266</td><td>5.213</td><td>2.653</td><td>6.189</td></tr><tr><td> $\lambda _ { 2 } = 0 . 4 5$ </td><td>32.96</td><td>0.0261</td><td>5.058</td><td>2.644</td><td>6.210</td></tr><tr><td> $\lambda _ { 2 } = 0 . 5 0$ </td><td>33.06</td><td>0.0252</td><td>4.846</td><td>2.620</td><td>6.260</td></tr><tr><td> $\lambda _ { 2 } = 0 . 6 0$ </td><td>33.10</td><td>0.0251</td><td>5.106</td><td>2.671</td><td>6.015</td></tr><tr><td> $\lambda _ { 3 } = 0 . 0 1$ </td><td>33.01</td><td>0.0255</td><td>4.913</td><td>2.645</td><td>6.211</td></tr><tr><td> $\lambda _ { 3 } = 0 . 0 2$ </td><td>32.99</td><td>0.0249</td><td>5.718</td><td>2.680</td><td>6.125</td></tr><tr><td> $\lambda _ { 3 } = 0 . 0 3$ </td><td>33.06</td><td>0.0252</td><td>4.846</td><td>2.620</td><td>6.260</td></tr><tr><td></td><td></td><td>0.0254</td><td>4.905</td><td></td><td></td></tr><tr><td> $\lambda _ { 3 } = 0 . 0 4$ </td><td>33.03</td><td></td><td></td><td>2.633</td><td>6.116</td></tr></table>

<!-- image-->  
Fig. 6. Qualitative comparison of visual results under different motionoffset selection strategies. Red bounding boxes highlight regions exhibiting over-smoothing artifacts and inaccurate lip-synchronization. Results are shown for the ground truth (GT), the proposed mean offset only $( \Delta \mu ) .$ , and the full configuration incorporating mean, scale, and residual offsets (âÂµ, âs, âr).

Hyperparameter Sensitivity. During the fusion stage, we use the $L _ { 1 }$ loss, D-SSIM loss $\mathcal { L } _ { D S }$ , LPIPS loss $\mathcal { L } _ { L P } ,$ , and a taskspecific lip-synchronization loss $\mathcal { L } _ { l i p }$ derived from a pretrained lip-discriminator. The relative contributions within the total loss function $\mathcal { L } _ { f u }$ are empirically set as follows: $\lambda _ { 1 } = 0 . 2$ for structural constraints, $\lambda _ { 2 } = 0 . 5$ for perceptual alignment, and $\lambda _ { 3 } ~ = ~ 0 . 0 3$ for synchronization. We conduct a series of comprehensive quantitative experiments to identify the most effective hyperparameter configurations across different training phases. As detailed in Table V. (1) The sensitivity analysis for $\lambda _ { 3 }$ in the lip-synchronization loss demonstrates that $\lambda _ { 3 } ~ = ~ 0 . 0 3$ achieves the best results of PSNR, LPIPS, FID, and LMD. Deviating from this value often degrades either visual quality or speech alignment accuracy, making it the final parameter. (2) The coefficients $\lambda _ { 1 }$ and $\lambda _ { 2 }$ serve as regulators for reconstruction fidelity. They measure the magnitude of structural and textural error between the generated talking head image and the ground-truth frame, ensuring the synthesized results remain anchored to the target identityâs appearance.

## V. CONCLUSION

This paper presents FG-3DGS, a novel framework for finegrained audio-driven 3D talking head generation. FG-3DGS addresses the challenge of precise facial controlâcritical for avoiding the uncanny valley effectâby identifying that prior methods often overlook the distinct motion dynamics of different facial regions. To overcome this, the framework disentangles low- and high-frequency facial motions, enhancing high-frequency regions via a high-frequency-refined postrendering alignment. With these specialized components, FG-3DGS consistently outperforms recent state-of-the-art methods across various settings, demonstrating the effectiveness of frequency-aware disentanglement for fine-grained, realistic 3D talking head generation.

## VI. SUPPLEMENTARY VIDEO

We provide a set of video results to demonstrate our methodâs performance across both talking-head reconstruction and cross-subject lip-synchronization settings. The corresponding files are organized into the âreconstructionâ and âcross_subjectâ folders, respectively.

## REFERENCES

[1] J. Thies, M. Elgharib, A. Tewari, C. Theobalt, and M. NieÃner, âNeural voice puppetry: Audio-driven facial reenactment,â in European Conference on Computer Vision. Springer, 2020, pp. 716â731.

[2] Z. Zhang, L. Li, Y. Ding, and C. Fan, âFlow-guided one-shot talking face generation with a high-resolution audio-visual dataset,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2021, pp. 3661â3670.

[3] H. Kim, P. Garrido, A. Tewari, W. Xu, J. Thies, M. Niessner, P. PÃ©rez, C. Richardt, M. ZollhÃ¶fer, and C. Theobalt, âDeep video portraits,â ACM Transactions on Graphics (TOG), vol. 37, no. 4, pp. 1â14, 2018.

[4] Z. Sheng, L. Nie, M. Liu, Y. Wei, and Z. Gao, âToward fine-grained talking face generation,â IEEE Transactions on Image Processing, vol. 32, pp. 5794â5807, 2023.

[5] X. Li, X. Sheng, M. Wang, F.-Z. Ou, B. Chen, S. Wang, and S. Kwong, âCofaco: Controllable generative talking face video coding,â IEEE Transactions on Image Processing, 2026.

[6] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, âGenerative adversarial networks,â Communications of the ACM, vol. 63, no. 11, pp. 139â144, 2020.

[7] J. Guan, Z. Zhang, H. Zhou, T. Hu, K. Wang, D. He, H. Feng, J. Liu, E. Ding, Z. Liu et al., âStylesync: High-fidelity generalized and personalized lip sync in style-based generator,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023, pp. 1505â1515.

[8] J. Wang, X. Qian, M. Zhang, R. T. Tan, and H. Li, âSeeing what you said: Talking face generation guided by a lip reading expert,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 653â14 662.

[9] W. Zhong, C. Fang, Y. Cai, P. Wei, G. Zhao, L. Lin, and G. Li, âIdentitypreserving talking face generation with landmark and appearance priors,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023, pp. 9729â9738.

[10] K. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, and C. Jawahar, âA lip sync expert is all you need for speech to lip generation in the wild,â in Proceedings of the 28th ACM International Conference on Multimedia, 2020, pp. 484â492.

[11] Z. Zhang, Z. Hu, W. Deng, C. Fan, T. Lv, and Y. Ding, âDinet: Deformation inpainting network for realistic face visually dubbing on high resolution video,â in Proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 3, 2023, pp. 3543â3551.

[12] X. Ji, H. Zhou, K. Wang, Q. Wu, W. Wu, F. Xu, and X. Cao, âEamm: One-shot emotional talking face via audio-based emotion-aware motion model,â in ACM SIGGRAPH 2022 Conference Proceedings, 2022, pp. 1â10.

[13] J. Xing, M. Xia, Y. Zhang, X. Cun, J. Wang, and T.-T. Wong, âCodetalker: Speech-driven 3d facial animation with discrete motion prior,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 780â12 790.

[14] W. Zhang, X. Cun, X. Wang, Y. Zhang, X. Shen, Y. Guo, Y. Shan, and F. Wang, âSadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation,â arXiv preprint arXiv:2211.12194, 2022.

[15] Y. Ren, G. Li, Y. Chen, T. H. Li, and S. Liu, âPirenderer: Controllable portrait image generation via semantic neural rendering,â in Proceedings of the IEEE International Conference on Computer Vision, 2021, pp. 13 759â13 768.

[16] Y. Fan, Z. Lin, J. Saito, W. Wang, and T. Komura, âFaceformer: Speechdriven 3d facial animation with transformers,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022, pp. 18 770â18 780.

[17] Z. Peng, W. Hu, J. Ma, X. Zhu, X. Zhang, H. Zhao, H. Tian, J. He, H. Liu, and Z. Fan, âSynctalk++: High-fidelity and efficient synchronized talking heads synthesis using gaussian splatting,â arXiv preprint arXiv:2506.14742, 2025.

[18] V. Blanz and T. Vetter, âA morphable model for the synthesis of 3d faces,â in Seminal Graphics Papers: Pushing the Boundaries, Volume 2, 2023, pp. 157â164.

[19] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[20] C. Huang, Y. Hou, W. Ye, D. Huang, X. Huang, B. Lin, and D. Cai, âNerf-det++: Incorporating semantic cues and perspective-aware depth supervision for indoor multi-view 3d detection,â IEEE Transactions on Image Processing, 2025.

[21] M. Chen, L. Wang, Y. Lei, Z. Dong, and Y. Guo, âLearning spherical radiance field for efficient 360 unbounded novel view synthesis,â IEEE Transactions on Image Processing, vol. 33, pp. 3722â3734, 2024.

[22] Y. Guo, K. Chen, S. Liang, Y.-J. Liu, H. Bao, and J. Zhang, âAdnerf: Audio driven neural radiance fields for talking head synthesis,â in Proceedings of the IEEE International Conference on Computer Vision, 2021, pp. 5784â5794.

[23] J. Tang, K. Wang, H. Zhou, X. Chen, D. He, T. Hu, J. Liu, G. Zeng, and J. Wang, âReal-time neural radiance talking portrait synthesis via audio-spatial decomposition,â arXiv preprint arXiv:2211.12368, 2022.

[24] J. Li, J. Zhang, X. Bai, J. Zhou, and L. Gu, âEfficient region-aware neural radiance fields for high-fidelity talking portrait synthesis,â in Proceedings of the IEEE International Conference on Computer Vision, 2023, pp. 7568â7578.

[25] S. Shen, W. Li, X. Huang, Z. Zhu, J. Zhou, and J. Lu, âSd-nerf: Towards lifelike talking head animation via spatially-adaptive dual-driven nerfs,â IEEE Transactions on Multimedia, vol. 26, pp. 3221â3234, 2023.

[26] M. Wang, S. Zhao, X. Dong, and J. Shen, âHigh-fidelity and highefficiency talking portrait synthesis with detail-aware neural radiance fields,â IEEE Transactions on Visualization and Computer Graphics, 2024.

[27] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[28] Z. Bao, G. Liao, K. Zhou, K. Liu, Q. Li, and G. Qiu, âLoopsparsegs: Loop based sparse-view friendly gaussian splatting,â IEEE Transactions on Image Processing, 2025.

[29] Y. Wang, X. Wei, M. Lu, and G. Kang, âPlgs: Robust panoptic lifting with 3d gaussian splatting,â IEEE Transactions on Image Processing, 2025.

[30] S. Qian, T. Kirschstein, L. Schoneveld, D. Davoli, S. Giebenhain, and M. NieÃner, âGaussianavatars: Photorealistic head avatars with rigged 3d gaussians,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 299â20 309.

[31] K. Cho, J. Lee, H. Yoon, Y. Hong, J. Ko, S. Ahn, and S. Kim, âGaussiantalker: Real-time high-fidelity talking head synthesis with audio-driven 3d gaussian splatting,â arXiv preprint arXiv:2404.16012, 2024.

[32] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, âTalkinggaussian: Structure-persistent 3d talking head synthesis via gaussian splatting,â in European Conference on Computer Vision. Springer, 2024, pp. 127â145.

[33] J. Li, J. Zhang, X. Bai, J. Zheng, J. Zhou, and L. Gu, âInstag: Learning personalized 3d talking head from few-second video,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025, pp. 10 690â10 700.

[34] S. Gong, H. Li, J. Tang, D. Hu, S. Huang, H. Chen, T. Chen, and Z. Liu, âMonocular and generalizable gaussian talking head animation,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025, pp. 5523â5534.

[35] S. Shen, W. Li, Y. Zhang, Y.-P. Tan, and J. Lu, âAudio-plane: Audio factorization plane gaussian splatting for real-time talking head synthesis,â arXiv preprint arXiv:2503.22605, 2025.

[36] S. Aneja, A. Sevastopolsky, T. Kirschstein, J. Thies, A. Dai, and M. NieÃner, âGaussianspeech: Audio-driven personalized 3d gaussian avatars,â in Proceedings of the IEEE International Conference on Computer Vision, 2025, pp. 13 065â13 075.

[37] Y. Bao, T. Ding, J. Huo, Y. Liu, Y. Li, W. Li, Y. Gao, and J. Luo, â3d gaussian splatting: Survey, technologies, challenges, and opportunities,â IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[38] C. Ma, S. Tan, Y. Pan, J. Yang, and X. Tong, âEsgaussianface: Emotional and stylized audio-driven facial animation via 3d gaussian splatting,â IEEE Transactions on Visualization and Computer Graphics, 2026.

[39] C. Feichtenhofer, H. Fan, J. Malik, and K. He, âSlowfast networks for video recognition,â in Proceedings of the IEEE International Conference on Computer Vision, 2019, pp. 6202â6211.

[40] K. Wang, Q. Wu, L. Song, Z. Yang, W. Wu, C. Qian, R. He, Y. Qiao, and C. C. Loy, âMead: A large-scale audio-visual dataset for emotional talking-face generation,â in European Conference on Computer Vision. Springer, 2020, pp. 700â717.

[41] Y. Lu, J. Chai, and X. Cao, âLive speech portraits: real-time photorealistic talking-head animation,â ACM Transactions on Graphics (ToG), vol. 40, no. 6, pp. 1â17, 2021.

[42] E. R. Chan, C. Z. Lin, M. A. Chan, K. Nagano, B. Pan, S. De Mello, O. Gallo, L. J. Guibas, J. Tremblay, S. Khamis et al., âEfficient geometry-aware 3d generative adversarial networks,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 123â16 133.

[43] Z. Ye, Z. Jiang, Y. Ren, J. Liu, J. He, and Z. Zhao, âGeneface: Generalized and high-fidelity audio-driven 3d talking face synthesis,â arXiv preprint arXiv:2301.13430, 2023.

[44] Z. Peng, W. Hu, Y. Shi, X. Zhu, X. Zhang, H. Zhao, J. He, H. Liu, and Z. Fan, âSynctalk: The devil is in the synchronization for talking head synthesis,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 666â676.

[45] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, â4d gaussian splatting for real-time dynamic scene rendering,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 310â20 320.

[46] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, âDynamic 3d gaussians: Tracking by persistent dynamic view synthesis,â in 2024 International Conference on 3D Vision (3DV). IEEE, 2024, pp. 800â 809.

[47] B. Chen, S. Hu, Q. Chen, C. Du, R. Yi, Y. Qian, and X. Chen, âGstalker: Real-time audio-driven talking face generation via deformable gaussian splatting,â arXiv preprint arXiv:2404.19040, 2024.

[48] X. Liu, Y. Xu, Q. Wu, H. Zhou, W. Wu, and B. Zhou, âSemantic-aware implicit neural audio-driven video portrait generation,â in European Conference on Computer Vision. Springer, 2022, pp. 106â125.

[49] S. Shen, W. Li, Z. Zhu, Y. Duan, J. Zhou, and J. Lu, âLearning dynamic facial radiance fields for few-shot talking head synthesis,â in European Conference on Computer Vision. Springer, 2022, pp. 666â682.

[50] J. Wang, J.-C. Xie, X. Li, F. Xu, C.-M. Pun, and H. Gao, âGaussianhead: High-fidelity head avatars with learnable gaussian derivation,â 2024.

[51] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, âImage quality assessment: from error visibility to structural similarity,â IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600â612, 2004.

[52] C. Yu, J. Wang, C. Peng, C. Gao, G. Yu, and N. Sang, âBisenet: Bilateral segmentation network for real-time semantic segmentation,â in European Conference on Computer Vision, 2018, pp. 325â341.

[53] C.-H. Lee, Z. Liu, L. Wu, and P. Luo, âMaskgan: Towards diverse and interactive facial image manipulation,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020, pp. 5549â5558.

[54] K. Kvanchiani, E. Petrova, K. Efremyan, A. Sautin, and A. Kapitanov, âEasyportraitâface parsing and portrait segmentation dataset,â arXiv preprint arXiv:2304.13509, 2023.

[55] A. Hannun, C. Case, J. Casper, B. Catanzaro, G. Diamos, E. Elsen, R. Prenger, S. Satheesh, S. Sengupta, A. Coates et al., âDeep speech: Scaling up end-to-end speech recognition,â arXiv preprint arXiv:1412.5567, 2014.

[56] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, âHubert: Self-supervised speech representation learning by masked prediction of hidden units,â IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451â3460, 2021.

[57] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 586â595.

[58] Y. Xie, T. Feng, X. Zhang, X. Luo, Z. Guo, W. Yu, H. Chang, F. Ma, and F. R. Yu, âPointtalk: Audio-driven dynamic lip point cloud for 3d gaussian-based talking head synthesis,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 8, 2025, pp. 8753â 8761.

[59] S. Suwajanakorn, S. M. Seitz, and I. Kemelmacher-Shlizerman, âSynthesizing obama: learning lip sync from audio,â ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1â13, 2017.

[60] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, âGans trained by a two time-scale update rule converge to a local

nash equilibrium,â Advances in Neural Information Processing Systems, vol. 30, 2017.

[61] L. Chen, Z. Li, R. K. Maddox, Z. Duan, and C. Xu, âLip movements generation at a glance,â in European Conference on Computer Vision, 2018, pp. 520â535.

[62] J. S. Chung and A. Zisserman, âLip reading in the wild,â in Asian Conference on Computer Vision. Springer, 2016, pp. 87â103.

[63] J. S. Chung, âOut of time: automated lip sync in the wild,â in Asian Conference on Computer Vision, 2016, pp. 251â263.

[64] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, âwav2vec 2.0: A framework for self-supervised learning of speech representations,â Advances in Neural Information Processing Systems, vol. 33, pp. 12 449â12 460, 2020.