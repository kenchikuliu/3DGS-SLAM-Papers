<!-- page 1 -->
LiftAvatar: Kinematic-Space Completion for
Expression-Controlled 3D Gaussian Avatar Animation
Hualiang Weia, Shunran Jiab, Jialun Liuc and Wenhui Lia,∗
aCollege of Computer Science and Technology, Jilin University, No. 2699 Qianjin Street, Changchun, 130012, Jilin, China
bImpressed Inc DBA SocialBook, 950 Tower Ln, Foster City, 94404, California, USA
cInstitute of Artificial Intelligence of China Telecom, Zhonghui Building, Dongcheng District, 100007, Beijing, China
A R T I C L E I N F O
Keywords:
3D Avatar Reconstruction
Kinematic Space Completion
Fine-Grained Expression Control
Diffusion Transformer
A B S T R A C T
We present LiftAvatar, a new paradigm that completes sparse monocular observations in
kinematic space (e.g., facial expressions and head pose) and uses the completed signals to
drive high-fidelity avatar animation. LiftAvatar is a fine-grained, expression-controllable large-
scale video diffusion Transformer that synthesizes high-quality, temporally coherent expression
sequences conditioned on single or multiple reference images. The key idea is to lift incomplete
input data into a richer kinematic representation, thereby strengthening both reconstruction
and animation in downstream 3D avatar pipelines. To this end, we introduce (i) a multi-
granularity expression control scheme that combines shading maps with expression coefficients
for precise and stable driving, and (ii) a multi-reference conditioning mechanism that aggregates
complementary cues from multiple frames, enabling strong 3D consistency and controllability.
As a plug-and-play enhancer, LiftAvatar directly addresses the limited expressiveness and
reconstruction artifacts of 3D Gaussian Splatting-based avatars caused by sparse kinematic
cues in everyday monocular videos. By expanding incomplete observations into diverse pose-
expression variations, LiftAvatar also enables effective prior distillation from large-scale video
generative models into 3D pipelines, leading to substantial gains. Extensive experiments show
that LiftAvatar consistently boosts animation quality and quantitative metrics of state-of-the-art
3D avatar methods, especially under extreme, unseen expressions.
1. Introduction
The rapid progress of virtual reality, augmented reality, and telepresence is fueling an increasing demand for
realistic, animatable, and real-time renderable 3D digital human head models Cheng and Tsai (2014); Healey, Wang,
Wigington, Sun and Peng (2021); Kachach, Perez, Villegas and Gonzalez-Sosa (2020); Li, Zhang, Liu, Yang, Fu,
Tian, Han and Fan (2021). Commonly referred to as 3D head avatars, these models seek to reconstruct high-fidelity
geometry and fine-grained appearance from monocular or multi-view observations, enabling controllable synthesis
across viewpoints, head poses, and facial expressions. Such capabilities are central to immersive communication Li,
Olszewski, Xiu, Saito, Huang and Li (2020); Ma, Simon, Saragih, Wang, Li, De La Torre and Sheikh (2021); Tran,
Zakharov, Ho, Hu, Karmanov, Agarwal, Goldwhite, Venegas, Tran and Li (2024a), entertainment Sklyarova, Zakharov,
Hilliges, Black and Thies (2023); Zhu, Rematas, Curless, Seitz and Kemelmacher-Shlizerman (2020), and digital
content creation Li, Ma, Yan, Zhu and Yang (2023b); Naruniec, Helminger, Schroers and Weber (2020).
Recent years have pushed 3D head avatar reconstruction along the quality-efficiency Pareto frontier. In particular,
3D Gaussian Splatting (3DGS) Kerbl, Kopanas, Leimkühler and Drettakis (2023) has emerged as a strong representa-
tion, offering high-quality rendering with real-time performance, and has been widely adopted in avatar reconstruction
pipelines Svitov, Morerio, Agapito and Bue (2024); Shao, Wang, Li, Wang, Lin, Zhang, Fan and Wang (2024);
Chen, Wang, Li, Xiao, Zhang, Yao and Liu (2024b); Xu, Chen, Li, Zhang, Wang, Zheng and Liu (2024). Despite
this success, 3DGS-based avatars remain data-hungry: the reconstruction quality and downstream animation fidelity
strongly depend on whether the training observations sufficiently cover the underlying state space of the subject. This
limitation is especially pronounced for monocular videos captured in everyday settings, where head pose changes and
∗Corresponding author
weihl23@mails.jlu.edu.cn (H. Wei); shunran@socialbook.io (S. Jia); liujialun95@gmail.com (J. Liu); liwh@jlu.edu.cn
(W. Li)
ORCID(s): 0009-0001-7417-2687 (H. Wei); 0009-0006-1041-3484 (S. Jia); 0009-0001-6161-3842 (J. Liu); 0000-0001-6490-9852
(W. Li)
H. Wei et al.: Preprint submitted to Elsevier
Page 1 of 19
arXiv:2603.02129v1  [cs.CV]  2 Mar 2026

<!-- page 2 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
Figure 1: We propose a novel kinematic lifted framework, LiftAvatar, to complement the facial expressions and head poses
of the monotonous input video. LiftAvatar can promote subsequent reconstruction and driving tasks, resulting in significant
improvements.
facial expressions are often limited in diversity. We refer to these motion-related cues, including expression dynamics
and head pose, as kinematic information. When kinematic coverage is sparse, 3DGS models tend to overfit to the
observed motion patterns, leading to artifacts in reconstruction and brittle animation that collapses under unseen or
extreme expressions.
Most existing efforts address related but different bottlenecks. One line of work improves low-level capture quality,
such as motion blur removal or robust camera pose estimation from handheld videos Ma, Li, Liao, Zhang, Wang,
Wang and Sander (2022); Zhao, Wang and Liu (2024); Liu, Wu, Hoorick, Tokmakov, Zakharov and Vondrick (2023);
Wu, Zhang, Turki, Ren, Gao, Shou, Fidler, Gojcic and Ling (2025). Another line focuses on sparse-view or few-shot
reconstruction, including single-image 3D head avatar creation Deng, Wang, Ren, Chen and Wang (2024a); Deng,
Wang and Wang (2024b); Li, De Mello, Liu, Nagano, Iqbal and Kautz (2023a); Tran, Zakharov, Ho, Tran, Hu and
Li (2024b); Bao, Zhang, Li, Zhang, Yang, Bao, Pollefeys, Zhang and Cui (2024). These advances demonstrate that
plausible geometry and appearance can be recovered from limited viewpoints. However, even when the video is stable
H. Wei et al.: Preprint submitted to Elsevier
Page 2 of 19

<!-- page 3 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
and contains many frames, a fundamental issue often remains: the observations may still be kinematically incomplete.
Typical monocular capture sequences contain only a small subset of expressions (e.g., neutral or mild smile) and limited
pose range, leaving large regions of the expression-pose space unobserved. This kinematic-space incompleteness is not
directly solved by deblurring, pose refinement, or few-view geometry priors, and it becomes a primary failure mode
for 3DGS-based avatar animation: the reconstructed avatar appears stiff and produces unrealistic deformations when
driven beyond the narrow motion manifold seen during training.
We introduce LiftAvatar, a plug-and-play kinematic-space completion framework that lifts an expression-sparse
monocular video into a richer set of pose and expression observations using a large-scale video diffusion transformer.
The key idea is to leverage generative priors to expand the training observations along the kinematic dimensions while
preserving the subject identity and appearance details, thereby providing the downstream 3D reconstruction pipeline
with sufficiently diverse and semantically meaningful motion coverage. In contrast to prior avatar video generation
models that often suffer from temporal flickering and coarse motion control Xu, Yu, Zhou, Zhou, Jin, Hong, Ji, Zhu,
Cai, Tang, Lin, Li and Lu (2025); Cui, Li, Zhan, Shang, Cheng, Ma, Mu, Zhou, Wang and Zhu (2024); Guo, Zhang,
Liu, Zhong, Zhang, Wan and Zhang (2024); Tian, Wang, Zhang and Bo (2024); Siarohin, Lathuilière, Tulyakov, Ricci
and Sebe (2019), LiftAvatar is designed specifically for precision and consistency required by 3D training.
LiftAvatar is enabled by two core designs. First, for fine-grained and anatomically meaningful control, we adopt the
Neural Parametric Head Model (NPHM) Giebenhain, Kirschstein, Georgopoulos, Rünz, Agapito and Nießner (2023)
rather than linear parametric models such as FLAME Li, Bolkart, Black, Li and Romero (2017) or 3DMM Blanz
and Vetter (1999). We propose a multi-granularity kinematic conditioning scheme that jointly injects NPHM shading
maps and expression coefficient vectors into the diffusion transformer, enabling precise control over head pose and
facial dynamics while retaining photorealistic micro-details (e.g., wrinkles and lip-corner shapes). Second, to maximize
identity and appearance grounding, LiftAvatar supports multi-reference conditioning with an arbitrary number of input
frames. We integrate reference cues via patch embedding expansion and CLIP Radford, Kim, Hallacy, Ramesh, Goh,
Agarwal, Sastry, Askell, Mishkin, Clark et al. (2021) cross-attention, allowing the model to aggregate complementary
information across frames (identity, texture, hair, illumination) and reducing over-reliance on priors that can cause
subject drift. To further strengthen temporal coherence, we adopt the flow-matching training objective from Wan Wang,
Ai, Wen, Mao, Xie, Chen, Yu, Zhao, Yang, Zeng, Wang, Zhang, Zhou, Wang, Chen, Zhu, Zhao, Yan, Huang, Meng,
Zhang, Li, Wu, Chu, Feng, Zhang, Sun, Fang, Wang, Gui, Weng, Shen, Lin, Wang, Wang, Zhou, Wang, Shen, Yu, Shi,
Huang, Xu, Kou, Lv, Li, Liu, Wang, Zhang, Huang, Li, Wu, Liu, Pan, Zheng, Hong, Shi, Feng, Jiang, Han, Wu and Liu
(2025a). Importantly, LiftAvatar acts as a "rocket booster" during training: once the 3D avatar is learned, LiftAvatar
can be removed, incurring no additional cost at inference time for the final 3D system.
In summary, our contributions are:
• We propose LiftAvatar, a kinematic-space completion framework built on a large-scale video diffusion trans-
former that enriches expression-sparse monocular videos with diverse, high-fidelity head poses and fine-grained
expressions, directly addressing the key bottleneck in monocular 3D avatar animation.
• We introduce a multi-granularity expression control scheme that leverages NPHM shading maps and expression
coefficients to achieve high-precision, detail-preserving facial motion control suitable for downstream 3D
training.
• We design a multi-reference conditioning mechanism that fuses an arbitrary number of reference frames via
patch embedding expansion and CLIP cross-attention, improving identity grounding and reducing artifacts.
• Extensive experiments show that LiftAvatar substantially improves both visual quality and quantitative metrics
of state-of-the-art 3D avatar pipelines, MonoGaussianAvatar, especially under extreme and unseen expressions,
while preserving the inference efficiency of the final 3D avatar model.
2. Related Work
2.1. Data Augmentation for 3D Reconstruction
Data augmentation is widely used to improve robustness and visual quality in 3D reconstruction, especially for
NeRF Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi and Ng (2022) and 3D Gaussian Splatting (3DGS) Kerbl
et al. (2023). Existing methods can be roughly divided into three directions. Low-level enhancement methods primarily
H. Wei et al.: Preprint submitted to Elsevier
Page 3 of 19

<!-- page 4 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
address degradations such as noise, motion blur, and unstable camera motion, aiming to improve the quality of observed
frames for more stable reconstruction Ma et al. (2022); Lee, Lee, Sun, Ali and Park (2024). While effective when
the input is corrupted, these methods are limited to restoring what is already observed, and thus cannot resolve
structural incompleteness, such as missing viewpoints or sparse expression coverage. A second line of work distills
pretrained generative priors into 3D representations. Score distillation sampling (SDS) leverages diffusion models to
hallucinate missing observations and enables reconstruction from sparse inputs Shi, Chen, Zhang, Liu, Xu, Wei, Chen,
Zeng and Su (2023); Poole, Jain, Barron and Mildenhall (2023); Liu et al. (2023). Despite strong results, SDS-based
optimization is often unstable and can lead to over-saturation, over-smoothing, and blurred details. A third direction
emphasizes efficiency via feed-forward 3D inference, bypassing per-scene optimization by directly predicting 3D
attributes from images. Representative methods such as DUSt3R Wang, Leroy, Cabon, Chidlovskii and Revaud (2024a)
and VGGT Wang, Chen, Karaev, Vedaldi, Rupprecht and Novotny (2025b) achieve fast inference, but in head avatar
settings they typically struggle with fine-grained dynamic motion modeling and are often constrained to fixed-form
inputs. This limits their ability to exploit variable-length real-world observations such as monocular videos or sparse
multi-view sets, leading to underutilization of available data.
In contrast to these paradigms, our goal is to address kinematic incompleteness, a common but under-explored
bottleneck in monocular head avatar reconstruction. Rather than restoring low-level degradations or hallucinating
missing viewpoints, we explicitly complete the pose and expression space by synthesizing semantically meaningful,
identity-consistent kinematic variations. This enriches the training distribution for downstream 3D reconstruction and
improves both reconstruction robustness and animation fidelity.
2.2. Head Avatar Reconstruction and Animation
The pursuit of high-fidelity, animatable 3D head avatars has progressed from parametric face models to neural
rendering and generative priors. Early systems relied on 3DMM Blanz and Vetter (1999) and FLAME Li et al.
(2017) with learning-based refinements for expression and pose control Paysan, Knothe, Amberg, Romdhani and
Vetter (2009); Cudeiro, Bolkart, Laidlaw, Ranjan and Black (2019); Yang, Zhu, Wang, Huang, Shen, Yang and Cao
(2020); Feng, Feng, Black and Bolkart (2021); Wang, Chen, Yu, Ma, Li and Liu (2022); Daněček, Black and Bolkart
(2022); Ma, Zhang, Sun, Yan, Han and Xie (2024), but their linear parameterization limits fine-scale detail. NeRF-
based avatars Mildenhall et al. (2022); Athar, Xu, Sunkavalli, Shechtman and Shu (2022); Guo, Chen, Liang, Liu,
Bao and Zhang (2021); Liu, Xu, Wu, Zhou, Wu and Zhou (2022) significantly improved photorealism, yet remain
optimization-heavy and slow for real-time use. More recently, 3DGS Kerbl et al. (2023) enabled high-quality real-time
rendering and has been widely adopted for controllable head avatars Chen et al. (2024b); Xu et al. (2024); Qian,
Kirschstein, Schoneveld, Davoli, Giebenhain and Nießner (2024); Wang, Xie, Li, Xu, Pun and Gao (2025c); however,
these pipelines are data-hungry and often fail to generalize when monocular training videos exhibit limited pose or
expression diversity. Generative priors have also been exploited to compensate for incomplete observations. GAN-
based approaches such as EG3D Chan, Lin, Chan, Nagano, Pan, Mello, Gallo, Guibas, Tremblay, Khamis, Karras
and Wetzstein (2022) and ray conditioning Chen, Holalkere, Yan, Zhang and Davis (2023) leverage 2D priors Karras,
Laine and Aila (2019) for novel-view synthesis, while diffusion-based methods further extend to view and expression
generation Xu et al. (2025); Chen, Mihajlovic, Wang, Prokudin and Tang (2024a); Kirschstein, Giebenhain and Nießner
(2024); Wang, Yang, Kittler and Zhu (2024b); Ostrek and Thies (2024). Related controllable generation frameworks
model motion priors for long-horizon facial dynamics Shen, Wang, Gao, Guo, Dang, Tang and Chua and unify
conditional generation for pose-guided humans Shen and Tang (2024) and customizable virtual dressing Shen, Jiang,
He, Ye, Wang, Du, Li and Tang (2025). Nonetheless, generating unseen expressions or extreme motions from limited
2D evidence remains challenging, often causing temporal inconsistency, texture distortion, or identity drift.
To improve efficiency, feed-forward avatar models regress animatable heads from one or a few images within
seconds He, Gu, Ye, Xu, Zhao, Dong, Yuan, Dong and Bo (2025); Kirschstein, Romero, Sevastopolsky, Nießner and
Saito (2025), but are commonly constrained by fixed-form inputs and cannot fully exploit variable-length real-world
videos. Overall, existing paradigms work best when the input already contains rich kinematic variation. Our method
targets this bottleneck by combining a high-fidelity facial parameterization, NPHM Giebenhain et al. (2023), with a
large-scale video diffusion transformer to synthesize identity-consistent pose and expression variations, completing
the kinematic space as a plug-and-play pre-processor for downstream 3D avatar pipelines.
H. Wei et al.: Preprint submitted to Elsevier
Page 4 of 19

<!-- page 5 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
 Video Input
ℇRP
ΨR
Video token 
xD
Inference
Train
LiftAvatar
Lifted Training
Kinematic Lifted
 
Neural Radiance Fields
3D Gaussian Splatting
Rendering
SDF>0
SDF<0
Signed Distance Filed
...
...
It
I0
IT
I0
...
It
IT
...
Reference 
token xR
Removing 
Reference token
Remove Reference token
Reference token 
Unseen expression 
Train
Freeze
Reference image  
Concat
ER
SR
IR
…
…
VAE
Dec
DV
Wan DiT Blocks
Q-Lora
K-Lora
V-Lora
Driven 
NPHM
 Enc
ℇDN
Ref 
NPHM
 Enc
ℇRN
VAE
Enc
ℇV
Reference 
Exp Embedding
Video
Patch Embedding
Reference
Patch Embedding
ℇRP
Driven
Exp Embedding
ℇDP
Video token 
xD
SD
ED
…
…
Denoising Steps
NPHM Transfer
Data
base
Other diversity video
ER
…
…
ℇDN
ℇRN
ℇV
ΨD
Reference 
token xR
...
...
SD
Reference 
Selection
ℇDP
ED
Noise 
...
Wan DiT Blocks
Removing 
Reference token
P0
P1
P2
P3
P4
DV
Denoising Steps
OR
OR
Figure 2: Method Pipeline. LiftAvatar is a high-precision, expression-controlled video diffusion transformer that enriches
sparse observations to boost downstream avatar performance. It conditions on three groups of inputs: reference information
(𝐼𝑅, 𝑆𝑅, 𝐸𝑅) from the input video; driving information (𝑆𝐷, 𝐸𝐷) for the target motion; and the ground-truth driving video
𝑉𝐷during training (replaced by noise at inference). The reference images are encoded by a pre-trained VAE, and the
Reference NPHM Encoder encodes their shading maps 𝑆𝑅. These features are concatenated, projected via Reference
Patch Embedding, and summed with the embedded expression coefficients 𝐸𝑅to form the reference token 𝑥𝑅. Likewise,
the Driven NPHM Encoder processes the driving shading maps 𝑆𝐷; its output is projected by Driven Patch Embedding
and combined with the embedded 𝐸𝐷to produce the driving token 𝑥𝐷. The tokens 𝑥𝑅and 𝑥𝐷are concatenated and
fed as a unified condition into the Wan2.1 Wang et al. (2025a) video diffusion transformer backbone. Optimized with a
flow-matching objective, the model synthesizes high-fidelity, temporally coherent videos that accurately follow the driving
signals, thereby completing the kinematic space of the original input.
3. Methodology
3.1. Overview
Given a daily video 𝑉
= {𝑖0, 𝑖1, ⋯, 𝑖𝑇}, where 𝑇represents the total number of frames (typically featuring
neutral facial expressions and minimal head movement), the 3D avatar reconstructed from such a video usually lacks
expressiveness. In particular, when driven by unseen expressions, it often suffers from severe artifacts such as geometric
collapse. However, this type of monotonous video is prevalent both on the Internet and in personal recordings. Our
goal is to address this challenge by employing kinematic space completion techniques. To this end, we introduce
LiftAvatar, a novel kinematic lifted training framework. It employs a powerful large-scale video diffusion transformer
to lift the facial expressions and head poses from limited input, thereby significantly expanding the capacity of the input
information. Using these lifted kinematic priors, LiftAvatar substantially enhances the reconstruction and animation
quality of downstream 3D avatars. Crucially, the framework is decoupled from any specific head avatar reconstruction
method, serving as a plug-and-play module compatible with a wide range of existing 3D avatar techniques.
3.2. Framework of LiftAvatar
To achieve an effective increase in training data, the model must have sufficient fidelity and generation quality,
as lower quality kinematic lifted data compared to existing data would adversely affect 3D avatar training. Most
prior methods adopt ReferenceNet architectures Tian et al. (2024), but experimental observations indicate that such
frameworks suffer from suboptimal temporal consistency. Therefore, LiftAvatar employs the state-of-the-art open
source video model (i.e., Wan2.1 Wang et al. (2025a)) as its foundational framework. While Wan2.1 Wang et al.
(2025a) delivers compelling temporal consistency, its control precision falls short. To address this, we develop a
model based on the Wan2.1 Wang et al. (2025a) framework, as depicted in Figure 2. Our model generates high-
precision and high-fidelity videos by employing a multi-granularity expression control scheme and a multi-reference
H. Wei et al.: Preprint submitted to Elsevier
Page 5 of 19

<!-- page 6 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
conditioning mechanism. LiftAvatar accepts six types of inputs: reference images 𝐼𝑅= {𝑖𝑅
0 , ..., 𝑖𝑅
𝑀}, the corresponding
NPHM Giebenhain et al. (2023) shading map of reference images 𝑆𝑅= {𝑠𝑅
0 , ..., 𝑠𝑅
𝑀}, and NPHM expression
parameters 𝐸𝑅= {𝑒𝑅
0 , ..., 𝑒𝑅
𝑀}. A sequence of driving NPHM shading maps 𝑆𝐷= {𝑠𝐷
0 , ..., 𝑠𝐷
𝑁}; and the corresponding
NPHM Giebenhain et al. (2023) expression coefficients 𝐸𝐷= {𝑒𝐷
0 , ..., 𝑒𝐷
𝑁}. During training, a target driving video
𝑉𝐷= {𝐼𝐷
0 , ..., 𝐼𝐷
𝑁} is additionally provided as input, which is replaced by noise 𝑛𝐷during the inference phase. 𝑁
denotes the number of reference images and 𝑀denotes the length of the driving sequence. LiftAvatar accurately drives
the reference image into a corresponding video sequence guided by the driving expression sequence. It should be noted
that during training, the driving signal 𝑆𝐷aligns with 𝑉𝐷, which is extracted from the input video 𝑉. However, during
inference, to enrich the expressions in 𝑉, 𝑆𝐷and 𝐸𝐷are transferred from other expression-rich videos.
Fine-grained Expression Control. Unlike mainstream avatar models that rely primarily on linear parametric models
(e.g., FLAME Li et al. (2017) or 3DMM Blanz and Vetter (1999)), this paper adopts NPHM Giebenhain et al. (2023)
for high-precision expression representation. To this end, we design a multi-granularity expression control scheme. We
first use shading maps as fine-grained guidance signals, which effectively capture detailed facial variations. However,
relying solely on shading maps proves inadequate for achieving the desired level of control precision. Therefore, we
incorporate NPHM Giebenhain et al. (2023) expression coefficients as a complementary control mechanism to provide
more precise and structured guidance.
To achieve efficient and low-computational injection of the two types of control information mentioned above,
we designed two lightweight modules: the NPHM Encoder and the NPHM Exp Embedding. The NPHM Encoder is
divided into the Reference NPHM Encoder 𝑅𝑁and the Driven NPHM Encoder 𝐷𝑁. The former encodes shading
maps 𝑆𝑅corresponding to reference images, while the latter encodes shading maps 𝑆𝐷corresponding to driven
sequences. Both are small encoders composed of 3D convolutions, which encode shading maps into latent codes aligned
with the latent DiT. Given the different nature of our inputs, it has two instantiations:
(1) Driven NPHM Encoder. The structure of the Driven NPHM Encoder consists of 7 Conv3D layers, which
transform the input shading map of dimensions 𝐵×3×𝐹×512×512 into a latent code of dimensions 𝐵×16× 𝐹
4 ×64×64,
where 𝐵denotes the batch size and 𝐹the number of input frames. Its core architecture employs multi-level
downsampling blocks, progressively compressing spatial dimensions and expanding the number of channels through
stacked Conv3D layers, ultimately outputting condition embedding features aligned with video DiT of Wan2.1 Wang
et al. (2025a). Since the driven NPHM Giebenhain et al. (2023) constitutes a continuous video sequence, temporal
information extraction is required. Thus, we designed the Driven NPHM Encoder as a compact network composed
of Conv3D layers. SiLU activation functions are employed to enhance non-linearity. The output layer generates latent
code using Conv3D, matching the input dimensions of video DiT.
(2) Reference NPHM Encoder. In contrast to the driven NPHM Giebenhain et al. (2023) sequence, the NPHM Gieben-
hain et al. (2023) shading maps of the reference images do not constitute a temporally relevant sequence. Consequently,
temporal modeling is unnecessary. Thus, the Reference NPHM Encoder is designed as a straightforward six-layer
Conv2D convolutional neural network, which transform the input shading map of dimensions 𝐵× 3 × 512 × 512 into a
latent code of dimensions 𝐵×16×64×64, where 𝐵denotes batch size. This encoder incorporates three downsampling
stages, each implemented by Conv2D layers with 𝑠𝑡𝑟𝑖𝑑𝑒= 2.
The NPHM Exp Embeddings are also divided into two parts: one (i.e., 𝑅𝑃) for reference images and one (i.e.,
𝐷𝑃) for driven sequences. Both consist of a single MLP layer that aligns NPHM Giebenhain et al. (2023) expression
coefficients with the features after patch embedding in the Wan2.1 Wang et al. (2025a) model. 𝑆𝑅is encoded by 𝑅𝑁
into latent code and concatenated with the latent code of 𝐼𝑅processed by Wan2.1’s Wang et al. (2025a) VAE Encoder
𝑉along the channel dimension to obtain the feature representation of the reference information. This feature is then
embedded into the latent space of Wan2.1 Wang et al. (2025a) through the Reference Patch Embedding layer Ψ𝑅.
Subsequently, it is summed with the 𝑅𝑃encoded NPHM Giebenhain et al. (2023) expression parameters to generate
the reference token 𝑥𝑅. 𝑥𝑅simultaneously encodes fine-grained control information and coarse-grained expression
information, improving the precision of the reference input. Driving information undergoes a similar process: it is
encoded and extracted via 𝐷𝑁, 𝐷𝑃and Ψ𝐷to obtain the driving token 𝑥𝐷. 𝑥𝑅and 𝑥𝐷are concatenated along the
dimension axis and fed into the Wan2.1 Wang et al. (2025a) to achieve controlled generation of the driven video.
High-fidelity Generation. Unlike previous approaches Guo et al. (2024); Xu et al. (2025) that predominantly rely
on a single reference image, which fails to convey complex expressions and poses simultaneously, thereby forcing
heavy dependence on prior knowledge or resulting in over-smoothed details, this paper proposes LiftAvatar which
supports the flexible injection of arbitrary reference images. To this end, we design a multi-reference conditioning
H. Wei et al.: Preprint submitted to Elsevier
Page 6 of 19

<!-- page 7 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
mechanism. By leveraging multiple reference inputs, our method provides a richer and more precise source of
identity and appearance details. This significantly reduces the reliance on learned priors for "borrowing" details and
effectively mitigates the tendency to generate over-smoothed outputs, thereby aligning with our objective of high-
fidelity generation.
(1) Reference Image Selection. Selecting informative reference frames from in-the-wild videos is critical for final
quality. Instead of manual or random selection, we propose an automatic strategy based on K-means clustering
to maximize the expression diversity and informational coverage of the reference set. Specifically, we extract the
NPHM Giebenhain et al. (2023) expression coefficients (a 100-dimensional vector) for each frame of the input video
𝑉and cluster them into 𝑘groups based on expression similarity. For each resulting cluster, the frame whose coefficient
vector is closest to the cluster centroid is selected as a reference image. This approach ensures that the selected
references collectively represent the broadest spectrum of expressions present in 𝑉, minimizing redundancy and
reducing the model’s need to "guess" missing details.
Specifically, the K-means algorithm operates as follows. We first load the expression coefficients for all frames and
randomly initialize 𝑘cluster centers 𝐶= {𝐞1, 𝐞2, … , 𝐞𝑘}. Then, for each sample 𝐞𝑖, we compute its Euclidean distance
to every cluster center and assign it to the nearest one:
label𝑖= arg min
𝑘‖𝐞𝑖−𝐞𝑘‖2.
(1)
After assignment, the cluster centers are updated by taking the mean of all samples within each cluster. Then, for each
cluster, the frame with the smallest distance to the cluster center is selected as a reference:
𝐞𝑘=
1
𝑁𝑘
∑
𝑖∈cluster𝑘
𝐞𝑖,
(2)
where 𝑁𝑘denotes the number of samples in the 𝑘-th cluster.
(2) Multi-Reference Injection. To effectively inject the selected reference images into the generation process, we
employ two complementary approaches: 1) Latent Code Injection: The reference image is encoded by Wan2.1’s Wang
et al. (2025a) VAE encoder, and the resulting latent code is injected into the input of the Wan2.1 Wang et al. (2025a)
model. However, this method suffers from catastrophic forgetting. 2) Extended CLIP Radford et al. (2021) Context:
We extend the original CLIP Radford et al. (2021) context to a multi-frame CLIP Radford et al. (2021) context, thereby
generalizing the single-image injection mechanism to support an arbitrary number of images. Through the combination
of these two injection strategies, where latent codes provide robust identity grounding and the extended CLIP context
preserves precise appearance details, our model achieves high-fidelity generation of the driven video.
Training Objective. LiftAvatar builds upon the Wan2.1 Wang et al. (2025a) video generation framework by fine-tuning
all LoRA modules in the attention layers of the pre-trained Wan Wang et al. (2025a) model. The following components
are trained from scratch: 𝑅𝑁, 𝑅𝑃, 𝐷𝑁, 𝐷𝑃, Ψ𝑅, and Ψ𝐷, while the Video Patch Embedding and Reference
Patch Embedding modules undergo fine-tuning.
During the training phase, LiftAvatar employs the Flow Matching objective consistent with Wan2.1’s Wang et al.
(2025a) training framework. The training objective is formulated as follows:
= 𝔼𝑥0,𝑥1,𝑐,𝑡
[‖𝑢(𝑥𝑡, 𝑐, 𝑡; 𝜃) −𝑣𝑡‖2] ,
(3)
where, 𝑡denotes the timestep sampled from a logit-normal distribution, 𝑥1 represents the latent representation of the
clean driving image encoded through Wan2.1-VAE, 𝑥0 ∼(0, 𝐼) denotes random Gaussian noise, 𝑥𝑡= 𝑡𝑥1+(1−𝑡)𝑥0
is the intermediate latent representation, 𝑣𝑡= 𝑥1 −𝑥0 is the ground-truth velocity vector, 𝑢(𝑥𝑡, 𝑐, 𝑡; 𝜃) denotes the
velocity predicted by the LiftAvatar model, 𝑐represents all conditional inputs, including: 𝐼𝑅, 𝑆𝑅, 𝑆𝐷, 𝐸𝑅and 𝐸𝐷.
4. Experiments
4.1. Dataset
We conducted experiments on the NeRSemble dataset Kirschstein, Qian, Giebenhain, Walter and Nießner (2023),
which contains more than 4,700 sequences involving 267 IDs, each ID corresponding to 24 sets of expression
sequences. Each expression sequence includes information from 16 different angles. In total, there are 31.7 million
H. Wei et al.: Preprint submitted to Elsevier
Page 7 of 19

<!-- page 8 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
Monocular Video 
Driven
Expression 
FOMM
Face 
Vid2vid
Diffusion 
Avatars
LivePortrait
…
…
…
…
…
Ours
Hunyuan
Portrait
Figure 3: Qualitative results for kinematic lifted. We compared our results with non-diffusion-based methods (FOMM Siaro-
hin et al. (2019) and Face Vid2vid Wang et al. (2021)) as well as diffusion-based models (DiffusionAvatars Kirschstein
et al. (2024) and LivePortrait Guo et al. (2024) and HunyuanPortrait Xu et al. (2025)). It is evident that our method
provides better performance in generating extreme expressions, particularly in terms of facial texture details, teeth accuracy,
and pose accuracy.
frames of high-resolution videos covering a wide range of facial dynamics, including head movements, natural
expressions, emotions, and language expressions. We first divided the different expression sequences for each ID in
the dataset into training and testing sets, selecting the frontal view of all IDs as the video input. Subsequently, we fitted
the NPHM Giebenhain et al. (2023) to the different expression sequences for these IDs. We paired different expression
videos and fitted grids under predetermined angles.
Data Crop. We initiate the pipeline by processing frame sequences from camera view cam_222200037 across all
subject IDs and expression variations in the original NeRSemble dataset Kirschstein et al. (2023). The cropping
workflow consists of three core functions: We first perform frame-by-frame cropping on the cam_222200037 camera
angle data corresponding to all IDs in different expression sequences from the original NeRSemble dataset Kirschstein
et al. (2023). The specific functions include:
1. Facial Detection and Landmark Localization
Using the InsightFace library to detect faces in the images and obtain keypoints.
2. Automated Cropping Region Calculation
Determining a unified cropping area based on the face keypoints from multiple images.
3. Image Cropping and Standardization
Cropping all images according to the calculated cropping area and adjusting them to a uniform size of 512×512
pixels.
Data Generation. The data generation process involves reconstructing 3D face models from monocular videos and
rendering them with the Phong shading model. We will illustrate the overall data processing flow for each ID, as
shown in Figure 8. The algorithm describes the NPHM Giebenhain et al. (2023) monocular 3D face reconstruction
and rendering pipeline, consisting of four stages:
1. Preprocessing (face detection, segmentation, and landmark fitting using RetinaFace, MODNet, and PIPNet)
2. Dynamic 3D Reconstruction (optimizing latent codes 𝐳geo for geometry, 𝐳app for appearance, 𝐳𝑡
exp for per-frame
expressions, and lighting 𝜁via volumetric rendering)
H. Wei et al.: Preprint submitted to Elsevier
Page 8 of 19

<!-- page 9 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
MonoGaussianAvatar
SplattingAvatar
Single video
+ Exp-Aug
driving
Single video
+ Exp-Aug
driving
Single video
+ Exp-Aug
driving
Single video
+ Exp-Aug
driving
Single video
Single video
Single video
+ Kinematic Lifted
+ Kinematic Lifted
+ Kinematic Lifted
Single video
+ Kinematic Lifted
Figure 4: Qualitative results for head avatar animation. We compare the two head avatar animation methods before and
after kinematic lifted. The comparison shows that our proposed strategy can effectively enhance subsequent reconstruction
and driving.
Reference Images
Kinematic Lifted Frames with LiftAvatar
Figure 5: Lifted results with LiftAvatar.
3. Mesh Extraction (marching cubes to extract a canonical mesh with vertices 𝑉and faces 𝐹)
4. Phong Rendering (deforming the mesh per frame and shading using Phong lighting model with coefficients 𝑘𝑎,
𝑘𝑑, 𝑘𝑠)
The pipeline combines neural implicit representations with traditional mesh rendering for photorealistic results.
H. Wei et al.: Preprint submitted to Elsevier
Page 9 of 19

<!-- page 10 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
MonoGaussianAvatar / Driving Single video：
MonoGaussianAvatar/ Driving Single video +  LiftAvatar：
LAM：
MonoGaussianAvatar/ Driving Single video +  LiftAvatar：
MonoGaussianAvatar / Driving Single video：
LAM：
Figure 6: Additional Results from MonoGaussianAvatar and LAM.
4.2. Training
Training Setting. We trained on 8×NVIDIA H100 GPUs, using the AdamW optimizer. LiftAvatar is based on Wan2.1-
Fun-14B Control Wang et al. (2025a). LiftAvatar initializes the weights of the video patch embedding and reference
patch embedding using the weights from Wan2.1’s Wang et al. (2025a) original patch embedding, training them with a
learning rate of 1e-4. For the key, query, and value embedding layers in Wan2.1’s Wang et al. (2025a) attention module,
we fine-tune using LoRA with a rank of 64, setting the learning rate to 1e-5. The batch size is configured as 2 per GPU,
with training conducted over 3 days for 60,000 steps. When sampling reference images, we avoid selecting frames from
the same video in the NeRSemble Kirschstein et al. (2023). Instead, we extract frames from different videos featuring
the same person to prevent the model from failing to generate expressions with significant differences.
Kinematic Lifted Baseline. Our comparative analysis pits LiftAvatar against a set of non-diffusion-based (FOMM Siaro-
hin et al. (2019), Face Vid2vid Wang et al. (2021)) and diffusion-based (LivePortrait Guo et al. (2024), Diffusion-
Avatars Kirschstein et al. (2024), HunyuanPotrait Xu et al. (2025)) methods. We perform inference tests on the
NeRSemble dataset Kirschstein et al. (2023). As shown in the side-by-side visual comparison in Figure 3, our method
achieves an inference speed of 5 fps (e.g., processing 150 frames in 30 seconds). The additional time required for this
inference is negligible compared to the typical several-hour training time of 3DGS avatars.
3D Head Avatar. To evaluate the effectiveness of our LiftAvatar, we performed comparative experiments against state-
of-the-art methods, namely SplattingAvatar Shao et al. (2024) and MonoGaussianAvatar Chen et al. (2024b). These
H. Wei et al.: Preprint submitted to Elsevier
Page 10 of 19

<!-- page 11 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
SplattingAvatar/ Driving Single video：
SplattingAvatar/ Driving Single video +  LiftAvatar
Avatar：
SplattingAvatar/ Driving Single video：
SplattingAvatar/ Driving Single video +  LiftAvatar：
Figure 7: Additional Comparative Results of SplattingAvatar.
Table 1
Quantitative comparison of expression kinematic lifted.
Method
Image Quality
Other Metrics
PSNR ↑
LPIPS ↓
SSIM ↑
FID ↓
AED ↓
CSIM ↑
FOMM Siarohin et al. (2019)
21.9167
0.2047
0.6823
60.31
0.7433
0.9721
Face Vid2vid Wang et al. (2021)
21.3089
0.1980
0.7150
54.61
0.7851
0.9718
DiffusionAvatars Kirschstein et al. (2024)
25.3098
0.1479
0.8119
40.84
0.6525
0.9814
LivePortrait Guo et al. (2024)
25.6561
0.1470
0.8237
40.55
0.6372
0.9855
HunyuanPortrait Xu et al. (2025)
27.2258
0.1137
0.8412
39.24
0.6144
0.9891
LiftAvatar (Ours)
28.6850
0.1008
0.8478
38.65
0.5958
0.9934
methods are specifically designed for 3D head avatar reconstruction and animation, making them suitable candidates
for our analysis. The experiments encompassed both basic reconstruction tasks and the rendering of novel expressions.
Through a comprehensive evaluation, we are able to determine the contributions of the proposed framework in
enhancing the quality of 3D head avatar representations. To ensure a fair comparison between the original input and its
kinematic lifted counterpart, each group was trained for the same number of iterations under identical settings. These
settings were carefully aligned with the optimal configurations provided in the official GitHub repositories. Specifically,
for SplattingAvatar, we used the Adam optimizer and trained for 50,000 iterations. For MonoGaussianAvatar, we also
used the Adam optimizer with a learning rate of 1 × 10−4 and trained for 60,000 iterations, with a batch size of 16.
Parameters not explicitly mentioned follow the default settings.
4.3. Results and Comparisons
We present the experimental results, providing a comprehensive analysis of both qualitative and quantitative
outcomes. Our evaluation framework encompasses a diverse set of metrics to ensure a thorough assessment of the
H. Wei et al.: Preprint submitted to Elsevier
Page 11 of 19

<!-- page 12 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
…
…
Dataset source images
Cropped images
Shaded images
…
…
…
…
…
…
…
…
Figure 8: We first cropped the original data from 2200 × 3208 pixels to 512 × 512 pixels. Next, we performed
NPHM Giebenhain et al. (2023) preprocessing on the cropped images, and finally, shaded them into Phong images.
 
 
 
 
PSNR(dB)
26.23
26.59
26.84
27.12
30
29
28.73
26.03
28
27
26
25
Random-1
Random-2
Random-3
Random-4
Random-5
K-means-5
Figure 9: The Number of Reference Images. Optimal performance is achieved with five reference frames selected by
K-means (K-means-5) during inference.
experimental results. The evaluation metrics we employed include Peak Signal-to-Noise Ratio (PSNR), Structural
Similarity Index (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), Frechet Inception Distance (FID),
and Frechet Video Distance (FVD). We measure identity preservation (CSIM) by comparing the cosine similarity
between the embeddings of the predicted and real images in a face recognition network. Additionally, we evaluate the
performance of generative models in expression (AED) reconstruction.
Quantitative Comparison. In Table 1, we compare LiftAvatar with other SOTA methods in the expression kinematic
lifted task. The results are averaged over avatars generated from 10 individuals. Our method shows the best overall
performance, compared to DiffusionAvatars Kirschstein et al. (2024), our method incorporates a motion module that
enhances temporal consistency, eliminating the impact of artifacts, resulting in texture details that (such as hair) are
closer to reality. Although LivePortrait Guo et al. (2024) performs well due to the inclusion of a super-resolution module
H. Wei et al.: Preprint submitted to Elsevier
Page 12 of 19

<!-- page 13 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
FOMM
F
       ace Vid2vid
DiffusionAvatars
LivePortrait
HunyuanPortrait
Ours
0
5
10
User Preference
Figure 10: User Study (50 users). In the evaluation conducted by 50 users on the kinematic lifted facial expressions, our
model achieved the highest scores and significantly outperformed all other compared models.
LPIPS
0.18
0.16
0.14
FOMM+exp
0.12
F
        ace Vid2vid+exp
DiffusionAvatars+exp
LivePortrait+exp
HunyuanPortrait+exp
0.10
L
        iftAvatar(ours)+exp
2
1
  0
 
 
  
 
 
  
50
1
                        00
1
                        50
Figure 11: The Number of Kinematic Lifted Expressions. The impact of the number of expressions on the effectiveness of
different kinematic lifted methods.
and training on a large expression dataset, it lacks control over extreme expressions and poses, often generating overly
exaggerated expressions that lead to inaccuracies. HunyuanPortrait Xu et al. (2025) also employs a large-scale video
diffusion transformer, it demonstrates significant advantages over other methods. Despite these advantages, its reliance
on sparse keypoints as control conditions, while offering greater flexibility, results in inferior control precision and
detail preservation compared to our method. The results indicate that images generated by LiftAvatar excel in clarity
(LPIPS) and perform remarkably well on metrics such as PSNR and SSIM compared to other baseline methods. The
synthesized facial features, such as teeth, eyebrows, mouth corners, and wrinkles, are more closely aligned with real
expressions (AED).
H. Wei et al.: Preprint submitted to Elsevier
Page 13 of 19

<!-- page 14 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
Table 2
Quantitative comparisons of different expression kinematic lifted methods.
Method
FOMM+exp
Face Vid2vid+exp
DiffusionAvatars+exp
PSNR ↑SSIM ↑LPIPS ↓PSNR↑SSIM ↑
LPIPS ↓PSNR ↑SSIM ↑
LPIPS ↓
SplattingAvatar Shao et al. (2024)
20.92 0.683
0.247
21.65 0.710
0.198
24.96 0.840
0.145
MonoGaussianAvatar Chen et al. (2024b)
22.50 0.790
0.232
22.80 0.751
0.192
25.10 0.885
0.137
Method
LivePortrait+exp
HunyuanPortrait+exp
LiftAvatar+exp
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑
LPIPS↓PSNR↑SSIM↑
LPIPS↓
SplattingAvatar Shao et al. (2024)
25.06 0.840
0.186
25.32 0.843
0.153
25.71 0.849
0.132
MonoGaussianAvatar Chen et al. (2024b)
25.50 0.891
0.181
25.79 0.898
0.140
26.98 0.903
0.121
Table 3
Quantitative comparisons of expression coefficient injection. +Kin-Lift means Kinematic Lifted. (w/o exp. coe.)
means without expression coefficient injection, (w exp. coe.) means with expression coefficient injection.
Method
PSNR ↑
SSIM ↑
LPIPS ↓
MonoGaussianAvatar
19.73
0.689
0.373
+Kin-Lift (w/o exp. coe.)
25.09
0.849
0.158
+Kin-Lift (w exp. coe.)
26.98
0.903
0.121
SplattingAvatar
18.32
0.657
0.410
+Kin-Lift (w/o exp. coe.)
24.69
0.785
0.264
+Kin-Lift (w exp. coe.)
25.71
0.849
0.132
Table 4
Quantitative comparison of different reference image counts.
Reference image
PSNR ↑
FVD ↓
FID ↓
K-means-one
28.59
341.55
43.87
K-means-two
28.63
290.94
42.17
K-means-three
28.65
263.67
41.69
K-means-five
28.68
181.90
38.94
In Table 2, we first apply six different expression kinematic lifted methods to the single video. We then use the
kinematic lifted motion information to train MonoGaussianAvatar Chen et al. (2024b) and SplattingAvatar Shao et al.
(2024), resulting in the corresponding trained models. Finally, we drive these models with unseen extreme expressions
and compare their performance with the models trained on the original monocular video. The results indicate that our
proposed kinematic lifted strategy is effective, and thanks to the expression injection in LiftAvatar, and the reference
images injection, our method demonstrates superior overall quantitative metrics.
Qualitative Comparison. We further provide a qualitative comparison in Figure 3. We apply different baseline
methods for expression kinematic lifted on monocular videos of different IDs. In contrast, our LiftAvatar demonstrates
superior performance in generating texture details in facial regions, including the mouth, eyebrows, and teeth. Even
in areas where the NPHM Giebenhain et al. (2023) mesh is poorly explained, our method can reasonably fill these
regions, exhibiting excellent 3D consistency, precise expression generation, and a realistic appearance. In Figure 4, we
compare two state-of-the-art head avatar methods, and it is visually evident that the driving effects achieved through
expression kinematic lifted outperform those from single video driving. This further demonstrates the effectiveness of
our LiftAvatar. We provide additional kinematic lifted results Figure 5.It is evident that our LiftAvatar can effectively
generate highly natural and high-quality extreme expression images, even when the input Reference image has a
limited range of expressions.To further facilitate the reproducibility of our LiftAvatar, we will open-source all the
involved code and data. We also provide additional visual comparisons in Figure 6, comparing the results of: (a) our
H. Wei et al.: Preprint submitted to Elsevier
Page 14 of 19

<!-- page 15 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
MonoGaussianAvatar driven by a single video, (b) driving with facial expressions enhanced by LiftAvatar, and (c)
the LAM He et al. (2025) method. Although LAM He et al. (2025) enables the direct generation of animatable and
renderable Gaussian head models from a single image, and its feed-forward architecture aligns with current trends in
generative models, it still exhibits limitations in fine-grained detail generation. Moreover, existing approaches in this
line have yet to establish an efficient and reliable pathway for fully leveraging the information contained in monocular
videos. Therefore, the method we propose remains of significant research value and practical potential. Figure 7 shows
the comparative effect of SplattingAvatar. It is evident that the kinematic lifted training framework we proposed is
effective.
User Study. We conducted a user study with 50 participants to subjectively evaluate the kinematic lifted effect of our
method (LiftAvatar) against five state-of-the-art approaches: FOMM Siarohin et al. (2019), Face Vid2vid Wang et al.
(2021), DiffusionAvatars Kirschstein et al. (2024), LivePortrait Guo et al. (2024), and HunyuanPortrait Xu et al. (2025).
Each participant rated the generated videos on a scale from 0 (worst) to 10 (best), and the results are summarized in
Figure 10. Our method achieved the highest mean score of 8.5 ± 0.8, substantially outperforming HunyuanPortrait Xu
et al. (2025) (8.0±1.0), LivePortrait Guo et al. (2024) (6.0±0.9), DiffusionAvatars Kirschstein et al. (2024) (5.0±1.1),
Face Vid2vid Wang et al. (2021) (4.0 ± 1.3), and FOMM Siarohin et al. (2019) (3.5 ± 1.5). Notably, our approach not
only attained the highest average rating but also exhibited the smallest variance, indicating consistent user preference.
A one-way analysis of variance (ANOVA) confirmed a significant effect of the method on ratings (𝐹(5, 294) = 96.3,
𝑝< 0.001), and post-hoc Tukey HSD tests revealed that our method significantly outperforms all other methods
(𝑝< 0.001 for each pairwise comparison). These results demonstrate that the lifted effects generated by our approach
are perceived as more natural and visually appealing than those of existing techniques.
4.4. Ablation Study
Expression Coefficient Injection. In Table 3, we study whether injecting NPHM expression coefficients is necessary
for downstream 3D head avatar reconstruction and animation. In particular, we compare training with the original
single video and with the expression kinematic lifted video under two settings: (w/o exp. coe.) removes the coefficient
injection branch, while (w exp. coe.) keeps it enabled. The results show a clear and consistent advantage of coefficient
injection. Intuitively, shading maps provide fine-grained, pixel-level deformation cues, but they are not always sufficient
to disambiguate expression semantics (e.g., subtle mouth-corner movement versus cheek deformation), especially
under extreme or unseen expressions. Injecting expression coefficients supplies a structured, low-dimensional and
semantically aligned control signal, which stabilizes the diffusion-driven motion generation and reduces drifting
or over-exaggeration. Consequently, both MonoGaussianAvatar and SplattingAvatar exhibit noticeably improved
reconstruction quality and animation fidelity after enabling coefficient injection. Without this enhancement, the
lifted training data may still contain imprecise expression trajectories, leading to suboptimal avatar deformation and
perceptual artifacts. More importantly, integrating expression injection consistently improves all evaluation metrics,
indicating that coefficient injection is not merely a minor add-on but a key component for high-quality and controllable
head avatar animation.
Reference Image Selection. In the inference process, selecting reference images is crucial because reference frames
anchor identity, appearance details (e.g., skin texture, teeth, hair), and illumination cues for the diffusion model. A
single reference image is often insufficient to cover complex facial states, and may force the model to over-rely on
priors, resulting in over-smoothed details or identity drift. To address this, we adopt an automatic and diversity-aware
selection strategy. Specifically, we apply K-means clustering on the 100-dimensional expression coefficient space and
select the frames closest to cluster centers as reference images. This strategy reduces redundancy among references and
maximizes expression coverage within the observed video, thereby improving the conditioning quality. As shown in
Table 4, increasing the number of reference images generally improves inference results, as more complementary cues
are provided. When using five reference images, we observe consistent improvements across all metrics compared to
using only a single reference image. This gain is also reflected in Figure 9, where randomly selected references lead to
fluctuating performance, while the K-means-selected set yields more stable and stronger results. We further evaluate
𝑘= {2, 4, 6, 7} to identify the best trade-off. When 𝑘< 5, the reference set lacks diversity, limiting the model’s
ability to faithfully reconstruct or synthesize challenging expressions. When 𝑘> 5, the computational cost increases
significantly due to the expanded conditioning context, yet the visual quality gain becomes marginal. Therefore, 𝑘= 5
provides the optimal balance between performance and efficiency in our setting.
Number of Kinematic Lifted Expressions. We further investigate how the number of kinematic lifted expressions
affects the final quality. Figure 11 shows that increasing the number of lifted expressions consistently reduces LPIPS,
H. Wei et al.: Preprint submitted to Elsevier
Page 15 of 19

<!-- page 16 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
indicating improved perceptual similarity and better reconstruction/animation realism. This trend aligns with our
motivation: more lifted expressions effectively enrich the kinematic coverage of the training data, reducing overfitting to
a narrow motion manifold and improving generalization to unseen or extreme expressions. Notably, the improvement
exhibits diminishing returns. The steepest quality gain occurs when the number of expressions increases from 2 to
50, suggesting that early expansion of expression coverage rapidly alleviates kinematic sparsity. After reaching 100
expressions, the curve flattens, indicating that the downstream avatar pipeline approaches a saturation regime where
additional expressions contribute less new information. This observation suggests a practical guideline: choosing a
moderate number of lifted expressions can already deliver most of the benefits, while further increasing the number
mainly increases training cost and storage without proportional gains in perceptual quality.
5. Conclusion
Existing methods for 3D avatar reconstruction primarily focus on improving the models themselves. However, in
this paper, we approach the task from a completely different perspective by lifting input data within the kinematic
space. Specifically, we enhance monocular videos, which often suffer from limited expression and pose variations,
thereby reducing the difficulty of downstream reconstruction tasks. We propose a novel kinematic lifted training method
called LiftAvatar, built upon a video generation architecture. LiftAvatar directly completes the expressions and poses
of monotonous video inputs, significantly enriching the information content of the input data and integrating large-
scale, high-quality data priors. Extensive experiments demonstrate that LiftAvatar substantially enhances the model’s
expressive capability in 3DGS Kerbl et al. (2023) when dealing with monotonous input videos. It improves the stability
of extreme expressions, reduces the occurrence of artifacts, and significantly lowers the model’s requirements for data
quality and diversity.
Declaration of competing interest
The authors declare that they have no known competing finacial interests or personal relationships that could have
appeared to influence the work reported in this paper.
Data availability
Data will be made available on request.
CRediT authorship contribution statement
Hualiang Wei: Writing – review & editing, Writing – original draft, Methodology, Software, Conceptualization,
Data curation, Formal analysis, Visualization, Validation. Shunran Jia: Formal analysis, Resources, Software. Jialun
Liu: Writing – review & editing, Investigation. Wenhui Li: Writing – review & editing, Supervision, Funding
acquisition.
References
Athar, S., Xu, Z., Sunkavalli, K., Shechtman, E., Shu, Z., 2022. Rignerf: Fully controllable neural 3d portraits, in: Proceedings of the IEEE/CVF
conference on Computer Vision and Pattern Recognition, pp. 20364–20373.
Bao, C., Zhang, Y., Li, Y., Zhang, X., Yang, B., Bao, H., Pollefeys, M., Zhang, G., Cui, Z., 2024. Geneavatar: Generic expression-aware volumetric
head avatar editing from a single image, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8952–
8963.
Blanz, V., Vetter, T., 1999. A morphable model for the synthesis of 3d faces, in: Waggenspack, W.N. (Ed.), Proceedings of the 26th Annual
Conference on Computer Graphics and Interactive Techniques, SIGGRAPH 1999, Los Angeles, CA, USA, August 8-13, 1999, ACM. pp. 187–
194. URL: https://dl.acm.org/citation.cfm?id=311556.
Chan, E.R., Lin, C.Z., Chan, M.A., Nagano, K., Pan, B., Mello, S.D., Gallo, O., Guibas, L.J., Tremblay, J., Khamis, S., Karras, T., Wetzstein, G., 2022.
Efficient geometry-aware 3d generative adversarial networks, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR
2022, New Orleans, LA, USA, June 18-24, 2022, IEEE. pp. 16102–16112. URL: https://doi.org/10.1109/CVPR52688.2022.01565,
doi:10.1109/CVPR52688.2022.01565.
Chen, E.M., Holalkere, S., Yan, R., Zhang, K., Davis, A., 2023. Ray conditioning: Trading photo-consistency for photo-realism in multi-view image
generation, in: IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, IEEE. pp. 23185–23194.
URL: https://doi.org/10.1109/ICCV51070.2023.02124, doi:10.1109/ICCV51070.2023.02124.
H. Wei et al.: Preprint submitted to Elsevier
Page 16 of 19

<!-- page 17 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
Chen, X., Mihajlovic, M., Wang, S., Prokudin, S., Tang, S., 2024a. Morphable diffusion: 3d-consistent diffusion for single-image avatar creation, in:
IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, IEEE. pp. 10359–10370.
URL: https://doi.org/10.1109/CVPR52733.2024.00986, doi:10.1109/CVPR52733.2024.00986.
Chen, Y., Wang, L., Li, Q., Xiao, H., Zhang, S., Yao, H., Liu, Y., 2024b. Monogaussianavatar: Monocular gaussian point-based head avatar, in:
Burbano, A., Zorin, D., Jarosz, W. (Eds.), ACM SIGGRAPH 2024 Conference Papers, SIGGRAPH 2024, Denver, CO, USA, 27 July 2024- 1
August 2024, ACM. p. 58. URL: https://doi.org/10.1145/3641519.3657499, doi:10.1145/3641519.3657499.
Cheng, K.H., Tsai, C.C., 2014. Children and parents’ reading of an augmented reality picture book: Analyses of behavioral patterns and cognitive
attainment. Computers & Education 72, 302–312.
Cudeiro, D., Bolkart, T., Laidlaw, C., Ranjan, A., Black, M.J., 2019. Capture, learning, and synthesis of 3d speaking styles, in: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pp. 10101–10111.
Cui, J., Li, H., Zhan, Y., Shang, H., Cheng, K., Ma, Y., Mu, S., Zhou, H., Wang, J., Zhu, S., 2024. Hallo3: Highly dynamic and realistic portrait
image animation with diffusion transformer networks. CoRR abs/2412.00733.
Daněček, R., Black, M.J., Bolkart, T., 2022. Emoca: Emotion driven monocular face capture and animation, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 20311–20322.
Deng, Y., Wang, D., Ren, X., Chen, X., Wang, B., 2024a. Portrait4d: Learning one-shot 4d head avatar synthesis using synthetic data, in: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7119–7130.
Deng, Y., Wang, D., Wang, B., 2024b. Portrait4d-v2: Pseudo multi-view data creates better 4d head synthesizer, in: European Conference on
Computer Vision, Springer. pp. 316–333.
Feng, Y., Feng, H., Black, M.J., Bolkart, T., 2021. Learning an animatable detailed 3d face model from in-the-wild images. ACM Transactions on
Graphics (ToG) 40, 1–13.
Giebenhain, S., Kirschstein, T., Georgopoulos, M., Rünz, M., Agapito, L., Nießner, M., 2023.
Learning neural parametric head models, in:
IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, IEEE. pp. 21003–
21012. URL: https://doi.org/10.1109/CVPR52729.2023.02012, doi:10.1109/CVPR52729.2023.02012.
Guo, J., Zhang, D., Liu, X., Zhong, Z., Zhang, Y., Wan, P., Zhang, D., 2024.
Liveportrait: Efficient portrait animation with stitching and
retargeting control. CoRR abs/2407.03168. URL: https://doi.org/10.48550/arXiv.2407.03168, doi:10.48550/ARXIV.2407.03168,
arXiv:2407.03168.
Guo, Y., Chen, K., Liang, S., Liu, Y.J., Bao, H., Zhang, J., 2021. Ad-nerf: Audio driven neural radiance fields for talking head synthesis, in:
Proceedings of the IEEE/CVF international conference on computer vision, pp. 5784–5794.
He, Y., Gu, X., Ye, X., Xu, C., Zhao, Z., Dong, Y., Yuan, W., Dong, Z., Bo, L., 2025. Lam: Large avatar model for one-shot animatable gaussian
head, in: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–13.
Healey, J., Wang, D., Wigington, C., Sun, T., Peng, H., 2021. A mixed-reality system to promote child engagement in remote intergenerational
storytelling, in: 2021 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct), IEEE. pp. 274–279.
Kachach, R., Perez, P., Villegas, A., Gonzalez-Sosa, E., 2020. Virtual tour: An immersive low cost telepresence system, in: 2020 IEEE conference
on virtual reality and 3D user interfaces abstracts and workshops (VRW), IEEE. pp. 504–506.
Karras, T., Laine, S., Aila, T., 2019. A style-based generator architecture for generative adversarial networks, in: IEEE Conference on Computer
Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, Computer Vision Foundation / IEEE. pp. 4401–
4410.
URL: http://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_
for_Generative_Adversarial_Networks_CVPR_2019_paper.html, doi:10.1109/CVPR.2019.00453.
Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
139:1–139:14.
Kirschstein, T., Giebenhain, S., Nießner, M., 2024. Diffusionavatars: Deferred diffusion for high-fidelity 3d head avatars, in: IEEE/CVF Conference
on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, IEEE. pp. 5481–5492.
URL: https:
//doi.org/10.1109/CVPR52733.2024.00524, doi:10.1109/CVPR52733.2024.00524.
Kirschstein, T., Qian, S., Giebenhain, S., Walter, T., Nießner, M., 2023. Nersemble: Multi-view radiance field reconstruction of human heads. ACM
Trans. Graph. 42. URL: https://doi.org/10.1145/3592455, doi:10.1145/3592455.
Kirschstein, T., Romero, J., Sevastopolsky, A., Nießner, M., Saito, S., 2025. Avat3r: Large animatable gaussian reconstruction model for high-fidelity
3d head avatars. arXiv preprint arXiv:2502.20220 .
Lee, B., Lee, H., Sun, X., Ali, U., Park, E., 2024. Deblurring 3d gaussian splatting, in: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler,
T., Varol, G. (Eds.), Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part
LVIII, Springer. pp. 127–143. URL: https://doi.org/10.1007/978-3-031-73636-0_8, doi:10.1007/978-3-031-73636-0\_8.
Li, N., Zhang, Z., Liu, C., Yang, Z., Fu, Y., Tian, F., Han, T., Fan, M., 2021. Vmirror: Enhancing the interaction with occluded or distant objects in
vr with virtual mirrors, in: Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pp. 1–11.
Li, R., Olszewski, K., Xiu, Y., Saito, S., Huang, Z., Li, H., 2020. Volumetric human teleportation, in: ACM SIGGRAPH 2020 Real-Time Live!, pp.
1–1.
Li, T., Bolkart, T., Black, M.J., Li, H., Romero, J., 2017. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph. 36,
194:1–194:17. URL: https://doi.org/10.1145/3130800.3130813, doi:10.1145/3130800.3130813.
Li, X., De Mello, S., Liu, S., Nagano, K., Iqbal, U., Kautz, J., 2023a. Generalizable one-shot 3d neural head avatar. Advances in Neural Information
Processing Systems 36, 47239–47250.
Li, Y., Ma, C., Yan, Y., Zhu, W., Yang, X., 2023b. 3d-aware face swapping, in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 12705–12714.
Liu, R., Wu, R., Hoorick, B.V., Tokmakov, P., Zakharov, S., Vondrick, C., 2023. Zero-1-to-3: Zero-shot one image to 3d object, in: IEEE/CVF
International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, IEEE. pp. 9264–9275. URL: https://doi.org/
10.1109/ICCV51070.2023.00853, doi:10.1109/ICCV51070.2023.00853.
H. Wei et al.: Preprint submitted to Elsevier
Page 17 of 19

<!-- page 18 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
Liu, X., Xu, Y., Wu, Q., Zhou, H., Wu, W., Zhou, B., 2022. Semantic-aware implicit neural audio-driven video portrait generation, in: European
conference on computer vision, Springer. pp. 106–125.
Ma, H., Zhang, T., Sun, S., Yan, X., Han, K., Xie, X., 2024. Cvthead: One-shot controllable head avatar with vertex-feature transformer, in:
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 6131–6141.
Ma, L., Li, X., Liao, J., Zhang, Q., Wang, X., Wang, J., Sander, P.V., 2022. Deblur-nerf: Neural radiance fields from blurry images, in: IEEE/CVF
Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, IEEE. pp. 12851–12860.
URL: https://doi.org/10.1109/CVPR52688.2022.01252, doi:10.1109/CVPR52688.2022.01252.
Ma, S., Simon, T., Saragih, J., Wang, D., Li, Y., De La Torre, F., Sheikh, Y., 2021. Pixel codec avatars, in: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 64–73.
Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R., 2022. Nerf: representing scenes as neural radiance fields for
view synthesis. Commun. ACM 65, 99–106. URL: https://doi.org/10.1145/3503250, doi:10.1145/3503250.
Naruniec, J., Helminger, L., Schroers, C., Weber, R.M., 2020. High-resolution neural face swapping for visual effects, in: Computer Graphics Forum,
Wiley Online Library. pp. 173–184.
Ostrek, M., Thies, J., 2024.
Stable video portraits.
CoRR abs/2409.18083.
URL: https://doi.org/10.48550/arXiv.2409.18083,
doi:10.48550/ARXIV.2409.18083, arXiv:2409.18083.
Paysan, P., Knothe, R., Amberg, B., Romdhani, S., Vetter, T., 2009. A 3d face model for pose and illumination invariant face recognition, in: Tubaro,
S., Dugelay, J. (Eds.), Sixth IEEE International Conference on Advanced Video and Signal Based Surveillance, AVSS 2009, 2-4 September 2009,
Genova, Italy, IEEE Computer Society. pp. 296–301. URL: https://doi.org/10.1109/AVSS.2009.58, doi:10.1109/AVSS.2009.58.
Poole, B., Jain, A., Barron, J.T., Mildenhall, B., 2023. Dreamfusion: Text-to-3d using 2d diffusion, in: The Eleventh International Conference
on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, OpenReview.net. URL: https://openreview.net/forum?id=
FjNys5c7VyY.
Qian, S., Kirschstein, T., Schoneveld, L., Davoli, D., Giebenhain, S., Nießner, M., 2024. Gaussianavatars: Photorealistic head avatars with rigged
3d gaussians, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20299–20309.
Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., 2021. Learning transferable
visual models from natural language supervision, in: International conference on machine learning, PmLR. pp. 8748–8763.
Shao, Z., Wang, Z., Li, Z., Wang, D., Lin, X., Zhang, Y., Fan, M., Wang, Z., 2024. Splattingavatar: Realistic real-time human avatars with mesh-
embedded gaussian splatting, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June
16-22, 2024, IEEE. pp. 1606–1616. URL: https://doi.org/10.1109/CVPR52733.2024.00159, doi:10.1109/CVPR52733.2024.00159.
Shen, F., Jiang, X., He, X., Ye, H., Wang, C., Du, X., Li, Z., Tang, J., 2025. Imagdressing-v1: Customizable virtual dressing, in: Proceedings of the
AAAI Conference on Artificial Intelligence, pp. 6795–6804.
Shen, F., Tang, J., 2024. Imagpose: A unified conditional framework for pose-guided person generation. Advances in neural information processing
systems 37, 6246–6266.
Shen, F., Wang, C., Gao, J., Guo, Q., Dang, J., Tang, J., Chua, T.S., . Long-term talkingface generation via motion-prior conditional diffusion model,
in: Forty-second International Conference on Machine Learning.
Shi, R., Chen, H., Zhang, Z., Liu, M., Xu, C., Wei, X., Chen, L., Zeng, C., Su, H., 2023. Zero123++: a single image to consistent multi-view diffusion
base model.
CoRR abs/2310.15110.
URL: https://doi.org/10.48550/arXiv.2310.15110, doi:10.48550/ARXIV.2310.15110,
arXiv:2310.15110.
Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., Sebe, N., 2019. First order motion model for image animation, in: Wallach, H.M., Larochelle,
H., Beygelzimer, A., d’Alché-Buc, F., Fox, E.B., Garnett, R. (Eds.), Advances in Neural Information Processing Systems 32: Annual Conference
on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 7135–7145.
URL:
https://proceedings.neurips.cc/paper/2019/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html.
Sklyarova, V., Zakharov, E., Hilliges, O., Black, M.J., Thies, J., 2023. Haar: Text-conditioned generative model of 3d strand-based human hairstyles.
arXiv preprint arXiv:2312.11666 .
Svitov, D., Morerio, P., Agapito, L., Bue, A.D., 2024.
HAHA: highly articulated gaussian human avatars with textured mesh prior, in: Cho,
M., Laptev, I., Tran, D., Yao, A., Zha, H. (Eds.), Computer Vision - ACCV 2024 - 17th Asian Conference on Computer Vision, Hanoi,
Vietnam, December 8-12, 2024, Proceedings, Part IX, Springer. pp. 105–122. URL: https://doi.org/10.1007/978-981-96-0969-7_7,
doi:10.1007/978-981-96-0969-7\_7.
Tian, L., Wang, Q., Zhang, B., Bo, L., 2024. EMO: emote portrait alive - generating expressive portrait videos with audio2video diffusion model
under weak conditions. CoRR abs/2402.17485. URL: https://doi.org/10.48550/arXiv.2402.17485, doi:10.48550/ARXIV.2402.
17485, arXiv:2402.17485.
Tran, P., Zakharov, E., Ho, L.N., Hu, L., Karmanov, A., Agarwal, A., Goldwhite, M., Venegas, A.B., Tran, A.T., Li, H., 2024a. Voodoo xp: Expressive
one-shot head reenactment for vr telepresence. arXiv preprint arXiv:2405.16204 .
Tran, P., Zakharov, E., Ho, L.N., Tran, A.T., Hu, L., Li, H., 2024b. Voodoo 3d: Volumetric portrait disentanglement for one-shot 3d head reenactment,
in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10336–10348.
Wang, A., Ai, B., Wen, B., Mao, C., Xie, C., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K.,
Zhao, K., Yan, K., Huang, L., Meng, X., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T.,
Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang,
Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z., Liu, Z., 2025a. Wan:
Open and advanced large-scale video generative models. CoRR abs/2503.20314. doi:10.48550/ARXIV.2503.20314, arXiv:2503.20314.
Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D., 2025b. Vggt: Visual geometry grounded transformer, in: Proceedings of
the Computer Vision and Pattern Recognition Conference, pp. 5294–5306.
Wang, J., Xie, J.C., Li, X., Xu, F., Pun, C.M., Gao, H., 2025c. Gaussianhead: High-fidelity head avatars with learnable gaussian derivation. IEEE
Transactions on Visualization and Computer Graphics .
H. Wei et al.: Preprint submitted to Elsevier
Page 18 of 19

<!-- page 19 -->
Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation
Wang, L., Chen, Z., Yu, T., Ma, C., Li, L., Liu, Y., 2022. Faceverse: a fine-grained and detail-controllable 3d face morphable model from a hybrid
dataset, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, IEEE.
pp. 20301–20310. URL: https://doi.org/10.1109/CVPR52688.2022.01969, doi:10.1109/CVPR52688.2022.01969.
Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J., 2024a. Dust3r: Geometric 3d vision made easy, in: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 20697–20709.
Wang, T., Mallya, A., Liu, M., 2021.
One-shot free-view neural talking-head synthesis for video conferencing, in: IEEE Conference on
Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, Computer Vision Foundation / IEEE. pp. 10039–
10049. URL: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_One-Shot_Free-View_Neural_Talking-Head_
Synthesis_for_Video_Conferencing_CVPR_2021_paper.html, doi:10.1109/CVPR46437.2021.00991.
Wang, W., Yang, H., Kittler, J., Zhu, X., 2024b.
Single image, any face: Generalisable 3d face generation.
CoRR abs/2409.16990.
URL:
https://doi.org/10.48550/arXiv.2409.16990, doi:10.48550/ARXIV.2409.16990, arXiv:2409.16990.
Wu, J.Z., Zhang, Y., Turki, H., Ren, X., Gao, J., Shou, M.Z., Fidler, S., Gojcic, Z., Ling, H., 2025. DIFIX3D+: improving 3d reconstructions with
single-step diffusion models, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Computer Vision Foundation
/ IEEE. pp. 26024–26035.
Xu, Y., Chen, B., Li, Z., Zhang, H., Wang, L., Zheng, Z., Liu, Y., 2024. Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians,
in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, IEEE. pp. 1931–1941.
URL: https://doi.org/10.1109/CVPR52733.2024.00189, doi:10.1109/CVPR52733.2024.00189.
Xu, Z., Yu, Z., Zhou, Z., Zhou, J., Jin, X., Hong, F., Ji, X., Zhu, J., Cai, C., Tang, S., Lin, Q., Li, X., Lu, Q., 2025.
Hunyuanportrait:
Implicit condition control for enhanced portrait animation, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition,
CVPR 2025, Nashville, TN, USA, June 11-15, 2025, Computer Vision Foundation / IEEE. pp. 15909–15919.
URL: https:
//openaccess.thecvf.com/content/CVPR2025/html/Xu_HunyuanPortrait_Implicit_Condition_Control_for_Enhanced_
Portrait_Animation_CVPR_2025_paper.html.
Yang, H., Zhu, H., Wang, Y., Huang, M., Shen, Q., Yang, R., Cao, X., 2020. Facescape: A large-scale high quality 3d face dataset and detailed riggable
3d face prediction, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19,
2020, Computer Vision Foundation / IEEE. pp. 598–607. URL: https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_
FaceScape_A_Large-Scale_High_Quality_3D_Face_Dataset_and_Detailed_CVPR_2020_paper.html, doi:10.1109/CVPR42600.
2020.00068.
Zhao, L., Wang, P., Liu, P., 2024. Bad-gaussians: Bundle adjusted deblur gaussian splatting, in: ECCV 2024, Springer. pp. 233–250.
Zhu, L., Rematas, K., Curless, B., Seitz, S.M., Kemelmacher-Shlizerman, I., 2020. Reconstructing nba players, in: European conference on computer
vision, Springer. pp. 177–194.
H. Wei et al.: Preprint submitted to Elsevier
Page 19 of 19
