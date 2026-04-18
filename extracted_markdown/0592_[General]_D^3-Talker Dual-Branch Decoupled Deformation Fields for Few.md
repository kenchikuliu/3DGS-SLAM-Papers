# D3-Talker: Dual-Branch Decoupled Deformation Fields for Few-Shot 3D Talking Head Synthesis

Yuhang Guoa, Kaijun Denga, Siyang Songd, Jindong Xiea, Wenhui Maa and Linlin Shena,b,c,\*

aSchool of Computer Science and Software Engineering, Shenzhen University

bComputer Vision Institute, School of Artificial Intelligence, Shenzhen University, Shenzhen, China

cGuangdong Provincial Key Laboratory of Intelligent Information Processing

dHBUG Lab, University of Exeter

## Abstract.

A key challenge in 3D talking head synthesis lies in the reliance on a long-duration talking head video to train a new model for each target identity from scratch. Recent methods have attempted to address this issue by extracting general features from audio through pre-training models. However, since audio contains information irrelevant to lip motion, existing approaches typically struggle to map the given audio to realistic lip behaviors in the target face when trained on only a few frames, causing poor lip synchronization and talking head image quality. This paper proposes D3-Talker, a novel approach that constructs a static 3D Gaussian attribute field and employs audio and Facial Motion signals to independently control two distinct Gaussian attribute deformation fields, effectively decoupling the predictions of general and personalized deformations. We design a novel similarity contrastive loss function during pre-training to achieve more thorough decoupling. Furthermore, we integrate a Coarse-to-Fine module to refine the rendered images, alleviating blurriness caused by head movements and enhancing overall image quality. Extensive experiments demonstrate that D3-Talker outperforms state-of-the-art methods in both high-fidelity rendering and accurate audio-lip synchronization with limited training data.

## 1 Introduction

Audio-driven Talking Head Synthesis aims to generate consistent and lifelike human talking facial behavior and head movement videos based on the given audio and the target personâs portrait. It has been widely applied in fields like film production, game development, online education, and live-streaming sales. Currently, many methods [10, 26, 25, 20, 34, 33, 36, 35, 3, 21, 38, 8, 31, 22] using 3D reconstruction techniques, such as Neural Radiance Fields (NeRF) [23] and 3D Gaussian Splatting (3DGS) [15] have been proposed for the talking head synthesis task. These methods use individual speaking videos as the training data to reconstruct the head, which simultaneously employ audio as the driving condition, and finally generate animated talking head portraits. Compared with 2D methods [24, 41, 13, 43], they additionally utilize 3D structural information and learn more high-frequency details, thus achieving better detailed textures and image quality. However, these 3D-based methods require training a model from scratch for each new identity, necessitating large amounts of talking head video frames to achieve good reconstruction quality and realistic lip movements. A key limitation preventing the real-world application of such approaches is the difficulty of providing long-duration talking videos for training.

Various approaches [25, 19, 34, 33, 36, 35, 22] attempt to address this problem. Some NeRF-based methods [25, 19] incorporate features from reference images (i.e., images randomly selected from the training set) for better face reconstruction with limited training data. Some one-shot talking head generation methods [34, 33, 36] learn a generic model to avoid dependence on additional video data. However, the former are constrained by the speed of inference and training, while the latter struggle to learn personalized features of the novel identities. Recently, several 3DGS-based methods [21, 3, 38, 8, 31, 22] have been proposed. These methods utilize 3DGS [15] for face reconstruction, learning the deformation of Gaussian attributes through audio. While these methods demonstrate superiority over NeRF-based solutions [10, 26, 25, 20] in terms of synthesis quality and speed under few-shot conditions, it is difficult for them to learn the correct Gaussian attribute deformations only based on audio features when a limited amount of training data is available, making them suffering from poor-quality rendered lip animations.

Based on the above observations, we identify three critical challenges faced by existing few-shot 3D talking head synthesis methods: (1) Imprecise audio-to-lip animation mapping [19]. The limited training data restricts both the accuracy and generalization ability of the mapping from audio to lip movements. It leads to blurry, lowquality, and poorly synchronized lip animations when driving audio is from outside of the training set or from other speakers; (2) The difficulty of decoupling general features from audio. Audio carries identity-specific attributes irrelevant to lip motion, such as pitch and timbre [38]. It is challenging to train a generic model that can decouple lip movement features from the audio of different speakers; and (3) Motion blur caused by head movements. In the 3D talking head synthesis task, the speakerâs head movement alters the relative camera pose, potentially causing artifacts during rendering. In this work, we propose D3-Talker to address these issues. Leveraging Audio-Motion Dual-Branch Control Signals, our method adopts Decoupled Deformation Fields to cooperatively represent fine-grained facial motions.

Inspired by the identity-free pre-training strategy of InsTaG [22], we adopt two decoupled fields to represent the general and personalized 3DGS attributes deformations of the target speaker, respectively.

<!-- image-->  
Figure 1: Overview of the proposed D3-Talker. Given a speech video and the corresponding audio, D3-Talker first builds Static Gaussian Fields. Then the audio branch and motion branch encode audio signal and Facial Motion signal respectively, and the deformations of Gaussian attributes can be co-predicted from the decoupled Individual Field and General Field. After the 3DGS rasterizer renders the 2D images, the Coarse-to-Fine Module refines these coarse images and produce high-quality results.

During the two-stage training, we first pre-train a shared Gaussian deformation field on a multi-identity dataset to learn the general deformation features. Then, we adapt it together with a scratch-trained individual deformation field to the target identity to capture additional personal deformation features. However, we found that previous methods [25, 19, 22] struggle to train a general base model from multi-identity dataset, as audio contains both speakerâs identity information and content information, making it difficult to directly learn the general lip movement feature from audios of different speakers. Therefore, we additionally introduce Facial Motion, a type of 3D face prior feature map extracted from audio by a pre-trained generative model [36] to learn general Gaussian deformations, and only employ audio to learn personalized deformations. Furthermore, we design a novel contrastive loss function during the pre-training stage to better decouple speaker-independent features. To confront the challenge posed by motion blur caused by head movements, we introduce the coarse-to-fine module. Specifically, we train a popular neural renderer following existing methods [36, 5, 4] to refine the coarse images from Gaussian splatting. In summary, this paper presents the following contributions to improve few-shot talking head synthesis:

â¢ We incorporate an audio-driven Facial Motion prior as a more general prior representation to learn identity-agnostic deformations of Gaussian attributes, proposing a new contrastive loss function to encourage learning generalizability from Facial Motion.

â¢ We propose a dual-branch signals of audio and Facial Motion to separately control two Gaussian deformation fields, decoupling the prediction of general and individual deformation.

â¢ Extensive experiments show that our proposed D3-Talker outperforms the state-of-the-art methods in terms of high-fidelity and accurate audio-lip synchronization with limited training data.

## 2 Related Work

2D/3D Talking Head Synthesis. Early methods [39, 24] of 2D talking head synthesis directly employ generative models such as GANs [9] and Auto-encoders [17] to generate talking faces. These methods struggle to achieve high visual fidelity and are incapable of capturing personalized speaking styles. Later, approaches [2, 27, 7, 41, 13, 43] leverage 2D landmarks or 3D Morphable Models (3DMM) expression coefficients as structural intermediate representations for better face modeling and control. They demonstrate strong generalizability and can be rapidly applied to unseen identities. Nevertheless, the intermediate representation may lead to the loss of high-frequency details. To address the oversight of head structure information, AD-NeRF [10] first introduces NeRF into the talking head synthesis task, as the 3D reconstruction representation of head. Following its successful application, some works [26, 20] have realized several improvements in rendering quality and efficiency. Recently, 3DGS introduces an explicit point-based representation for radiance fields and demonstrates higher rendering speed and quality compared to NeRF. Based on this idea, TalkingGaussian [21] and GaussianTalker [3] have implemented talking head reconstruction based on 3DGS. Despite the excellent performance of the above methods, they rely on large-scale datasets to achieve acceptable visual quality.

Few-shot Talking Head Synthesis. Since NeRF and 3DGS require retraining for reconstructing each new scene, their process is not only time-consuming but also highly data-dependent. To alleviate the problem, several few-shot learning methods [28, 37, 29, 32, 42] have been proposed. Some of them [28, 37, 29] have been applied to the talking head synthesis task. DFRF [25] introduces 2D pixel features from reference images (i.e., images randomly selected from the training set) for each 3D query point and reduces training overhead on new identities through a pre-trained base model. AE-Nerf [19] further incorporates audio as guidance for aggregating reference image pixel features and employs two decoupled NeRFs to collaboratively reconstruct the face, thereby improving image quality. MimicTalk [35] attempts to utilize a one-shot generator, fine-tuning with LoRA [12] on few minutes videos to capture personalized features. However, significant computational costs during both training and inference stages hinder their practical deployment. InsTaG [22] utilizes 3DGS for face reconstruction, integrates the training of a universal model, and then adapts to the target individual to learn the personal deformation of Gaussian attributes. It significantly improves the speed of both training and inference. However, it overlooks the difficulty in decoupling general features from audio, which restricts the generalizability to unseen speakers. In this paper, we propose to decouple identity-agnostic features from 3D face prior knowledge to learn a more generalizable deformation field, and use a neural renderer to refine the 3DGS-rendered results. Compared to previous approaches, our method achieves a better balance between rendering speed, image quality, and lip synchronization.

## 3 Methodology

The overview of our $\mathrm { D } ^ { 3 } .$ -Talker is shown in Figure 1. D3-Talker starts with static head reconstruction based on Gaussian Splatting [15] (Section 3.1). Then, the Audio-Motion Dual-Branch Control Signals (Section 3.2) are encoded and separately input into the Decoupled Dual Deformation Fields (Section 3.3), co-predicting the deformations of Gaussian attributes. Finally, the results rendered by Gaussian Splatting are refined by the Coarse-to-Fine Module (Section 3.4). Additionally, training details are described in Section 3.5.

## 3.1 Preliminaries

3DGS for Talking Head Synthesis. 3D Gaussian Splatting (3DGS) [15] represents the scene as a learnable set of Gaussian primitives. Specifically, the i-th Gaussian primitive Gi can be described by a set of parameters as:

$$
\theta _ { i } = \left\{ \mu _ { i } , s _ { i } , q _ { i } , \alpha _ { i } , f _ { i } \right\} ,\tag{1}
$$

where $\mu _ { i } \in \mathbb { R } ^ { 3 }$ is the center position; $s _ { i } \in \mathbb { R } ^ { 3 }$ is a scaling factor; $q _ { i } \in \mathbb { R } ^ { 4 }$ is a rotation quaternion; $\alpha _ { i } \in \mathbb { R } ^ { 1 }$ denotes the opacity value, and $f _ { i } \in \mathbb { R } ^ { d }$ denotes the d-dimensional color feature. By optimizing the parameter Î¸ for all Gaussians, a static Gaussian field of the head is obtained. Then, we learn the deformation parameters Î´ from audio to deform the Gaussian head.

Face-Mouth Region-wise Reconstruction. Following Talking-Gaussian [21] and InsTaG [22], we leverage a face-mouth regionwise reconstruction for the head to learn the static fields and deformation fields separately for the face and inside-mouth regions. For the face region, the deformation field predicts the point-wise deformation $\delta _ { \mathrm { f a c e } } = \{ \triangle \mu , \triangle s , \triangle q \}$ . Then, we use the deformed Gaussian primitive parameters $\theta _ { \mathrm { f a c e } } = \{ \mu + \triangle \mu , s _ { i } + \triangle s , q _ { i } + \triangle q , \alpha , f \}$ for rendering. For the inside-mouth region, only the center position deformation $\delta _ { \mathrm { m o u t h } } = \{ \triangle \mu \}$ is predicted. All deformation fields utilize a tri-plane hash encoder H [20] to encode the position information of Gaussian primitives, a region attention (RA) [20] module to enhance the spatial perception of audio features, and an MLP decoder to predict the deformation. The process of predicting deformations can be represented as follows:

$$
\delta = \mathrm { M L P } \left( \mathcal { H } \left( \mu \right) \oplus \mathrm { R A } \left( F _ { a } \right) \right) ,\tag{2}
$$

<!-- image-->  
Figure 2: The detailed process of the dual-branch control signals.

where $\mu$ denotes the center position of the query primitive, Fa denotes the processed audio features and â denotes concatenation.

## 3.2 Audio-Motion Dual-Branch Control Signals

We notice that most previous few-shot solutions [25, 19, 22] attempt to train a generic model that can extract speaker-invariant lip motion features from different speakersâ audio signals. However, audio signals also carry attributes that are irrelevant to human lip motions and are unique to the speaker, (e.g., pitch and timbre), which prevent these methods from extracting generic features solely from audio. Inspired by PointTalk [31], we present Audio-Motion Dual-Branch Control Signals, using Facial Motion together with the audio as the control signal for the deformations of Gaussian attributes, where Facial Motion serves as the general representation of lip motion and audio complement additional individual traits. The detailed process is illustrated in Figure 2.

Facial Motion Signal. We utilize the pre-trained Audio-to-Motion model from Real3D-Potrait [36] to extract motion feature from audio and use PNCC [44, 16] to represent it as IPNCC $\in \mathbb { R } ^ { 3 \times H \times W }$ which is a appearance-agnostic 3D face prior feature map that possesses fine-grained facial expression information based on a 3DMM face. Considering that all Gaussian deformations are based on the static Gaussian field, we design a difference encoder $E _ { \mathrm { d i f f } }$ to better learn subtle facial movements. Specifically, we take the PNCC corresponding to an expression coefficient of 0 as the canonical representation $I _ { \mathrm { P N C C } } ^ { \mathrm { c a n o } }$ . For the i-th frame, the difference is taken between the corresponding motion feature $I _ { \mathrm { P N C C } } ^ { i }$ and IcanoPNCC. A multi-layer convolution is then utilized to further extract and refine features, mapping them into a corresponding latent space aligned with the audio:

$$
F _ { \mathrm { M o t i o n } } ^ { i } = E _ { \mathrm { d i f f } } ( \triangle I _ { \mathrm { P N C C } } ^ { i } ) = C o n v ( I _ { \mathrm { P N C C } } ^ { i } - I _ { \mathrm { P N C C } } ^ { \mathrm { c a n o } } ) ,\tag{3}
$$

where $F _ { \mathrm { M o t i o n } } ^ { i } \in \mathbb { R } ^ { 1 \times 3 2 }$ . Subsequently, we utilize a temporal convolutional network (TCN) to smooth a total of 2j +1 adjacent frames (j

preceding frames and j succeeding frames) for each frame, reducing jitter and eliminating noise signals:

$$
F _ { M } ^ { i } = \mathrm { T C N } ( F _ { \mathrm { M o t i o n } } ^ { i - j } \oplus \cdots \oplus F _ { \mathrm { M o t i o n } } ^ { i + j } ) ,\tag{4}
$$

where $F _ { M } ^ { i } \in \mathbb { R } ^ { 1 \times 3 2 }$ and TCN denotes the Temporal Convolutional Network.

Audio Signal. Following previous audio-driven methods [10, 26, 25, 20, 21, 22], we employ the popular DeepSpeech model [11] to predict audio features $\dot { F } _ { \mathrm { A u d i o } } \in \dot { \mathbb { R } } ^ { \hat { T } \times 1 6 \times 2 9 }$ from the original audio track. To enhance the temporal correlation between adjacent frames, smoothing operation and compression are further carried out to obtain $F _ { A } \in \bar { \mathbb { R } ^ { T \times 3 2 } }$

Dual-Branch Enhancement. To capture the relationships between two types of signal features and different spatial regions, we propose the Dual-Branch Enhancement module. For the motion branch, we employ the tri-plane hash encoder H [20] to extract multi-resolution regional information $F _ { X \_ G }$ from the center position of static Gaussians $\mu .$ Then, we feed $F _ { X \_ G }$ to a two-layer MLP to generate the region attention vector describing the face spatial information. Finally, the region-aware feature $\hat { F } _ { M }$ is calculated through the Hadamard product of $F _ { M }$ and the region attention vector as:

$$
\hat { F } _ { M } = \mathrm { M L P } \left( \mathcal { H } \left( \boldsymbol { \mu } \right) \right) \odot F _ { M } ,\tag{5}
$$

where â denotes Hadamard product. The processing of the audio branch is done in the same way.

## 3.3 Decoupled Dual Deformation Fields

Based on the Dual-Branch Control Signals from Section 3.2, we introduce the Decoupled Dual Deformation Fields and General Field and Individual Field, to separately predict the general and personalized deformations of Gaussian attributes. Furthermore, a two-stage training strategy is employed to train the Decoupled Dual Deformation Fields, during which we also design a novel Similarity Contrastive Loss to encourage General Field to be generalizable.

General Field. General Field aims to predict identity-agnostic deformations of Gaussian attributes. To avoid the influence of information irrelevant to lip motion on the General Field, we use Facial Motion instead of audio as the input to the General Field. Given the center position $\mu ,$ the point-wise deformation parameter Î´G is predicted with the condition feature $\hat { F } _ { M } :$

$$
\delta _ { G } = \mathrm { M L P } \left( \mathcal { H } \left( \mu \right) \oplus \hat { F } _ { M } \right) .\tag{6}
$$

Individual Field. To learn more personalized and expressive facial movements, we employ an additional deformation field, termed Individual Field, for each identity. The Gaussian deformations for each identity are jointly learned by its corresponding Individual Field and pre-trained General Field. Since audio contains identity-related information, we leverage audio to learn personalized deformations $\delta _ { I }$ For the k-th identity, a process similar to the General Field is applied to predict deformations:

$$
\delta _ { I } ^ { k } = \mathrm { M L P } \left( \mathcal { H } ^ { k } \left( \mu ^ { k } \right) \oplus \hat { F } _ { A } ^ { k } \right) ,\tag{7}
$$

where k represents the identity order. $\delta _ { I } ^ { k } , \mathcal { H } ^ { k }$ and $\hat { F } _ { A } ^ { k }$ denote the personalized deformation, Hash encoder and region-aware audio feature for the k-th individual, respectively.

Two-stage Training Strategy. A two-stage training strategy that is similar to previous works [25, 19, 22] is adopted, allowing prior learning of common motion knowledge from long videos as compensation for the target identity. The first stage, the pre-training stage, trains the shared General Field on a multi-identity dataset to learn the general deformation features.

In the second stage, the adaptation stage, given a short video clip of the target identity, we utilize the pre-trained General Field, together with a scratch-trained Individual Field, to adapt personalized features of the target identity. We posit that, during adaptation to a target identity, personalized information is more readily derived from audio. Therefore, unlike in the adaptation of InsTaG [22], our Individual Field predicts the same deformation attributes as the General Field, and the two are directly added to deform the static field of the target identity:

$$
\tilde { \theta } = \theta + \delta , \quad \delta = \delta _ { G } + \delta _ { I } ,\tag{8}
$$

where Î¸ is the static Gaussian attributes. After that, a 2D image I can be rendered as:

$$
I = \mathcal { R } \left( \tilde { \theta } , [ R , t ] \right) ,\tag{9}
$$

where $\mathcal { R } ( \cdot )$ is the 3DGS rasterizer and $[ R , t ]$ represents the camera pose.

Similarity Contrastive Loss. To better decouple personalized and general deformations, InsTaG [22] introduced the Negative Contrast Loss $\mathcal { L } _ { C }$ to encourage the diversity of personalized deformations. However, $\mathcal { L } _ { C }$ only focuses on discriminating personalized deformations across Individual Fields while neglecting the critical role of the General Field in the pre-training stage. To address this limitation, we introduce a novel Similarity Contrastive Loss for each identity that explicitly enhances the similarity between features learned by the General Field and those derived from the corresponding audio of that identity.

Given a total of N identities during pre-training, there exists a General Field and N identity-dependent Individual Fields. For the k-th identity, our objective is to enhance the similarity between the universal Âµ-deformation $\triangle \mu _ { G }$ from the General Field and the personalized $\mu \cdot$ -deformation $\triangle \mu _ { I } ^ { k }$ from the k-th Individual Field. Meanwhile, the similarity between $\triangle \mu _ { G }$ and Âµ-deformation from other Individual Fields queried by the same audio feature $\hat { F } _ { A } ^ { k }$ should be minimized. Therefore, the loss function LSC (k) is constructed as:

$$
\mathcal { L } _ { S C } \left( k \right) = - \log \frac { \exp \left( s i m ( \triangle \mu _ { G } , \triangle \mu _ { I } ^ { k } ) / \tau \right) } { \sum _ { i = 1 } ^ { N } \exp \left( s i m ( \triangle \mu _ { G } , \triangle \mu _ { I } ^ { i } ) / \tau \right) } ,\tag{10}
$$

where sim $( \cdot , \cdot )$ denotes the cosine similarity function, Ï is the temperature factor.

## 3.4 Coarse-to-Fine Module

Previous methods [21, 22] simply combine the rendered results from the mouth and face regions to produce final output images. However, due to limited camera viewpoints in fewer training image frames, the 3DGS rasterizer tends to render artifacts, especially when the speakerâs head moves. To this end, we propose the Coarse-to-Fine module to refine the coarse images from Gaussian Splatting. Specifically, we first sum the results rendered from the two regions. Then, we feed them into a StyleUnet-based neural renderer, following existing methods [36, 5, 4]. We treat the renderer as an inpainting model rather than super-resolution, with both input and output resolutions being 512 Ã 512. During training, we first train a base model from scratch on the multi-identity datasets from the pre-training stage, and then fine-tune it on the target identity.

Table 1: Quantitative comparison under the self-driven setting with 10s training data.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="3">Visual Quality</td><td colspan="3">Lip Synchronization</td><td colspan="2">Efficiency</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>LMDâ</td><td>Sync-Câ</td><td>Sync-Dâ</td><td>Training Timeâ</td><td>FPSâ</td></tr><tr><td>Ground Truth RAD-NeRF [26]</td><td rowspan="2"></td><td>N/A</td><td>1</td><td>0</td><td>0</td><td>8.685</td><td>6.852</td><td>-</td><td>-</td></tr><tr><td></td><td>29.511</td><td>0.907</td><td>0.053</td><td>3.678</td><td>1.792</td><td>12.575</td><td>6 hours</td><td>24</td></tr><tr><td rowspan="2">DFRF [25] ECCV&#x27;22 ER-NeRF [20] ICCV&#x27;23</td><td rowspan="2">NeRF-based</td><td>30.413</td><td>0.909</td><td>0.060</td><td>3.460</td><td>3.658</td><td>10.927</td><td>9 hours</td><td>0.02</td></tr><tr><td>30.966</td><td>0.913</td><td>0.041</td><td>3.656</td><td>2.363</td><td>12.108</td><td>2 hours</td><td>30</td></tr><tr><td rowspan="2">GaussianTalker [3] ACM MM&#x27;24 TalkingGaussian [21] ECCV&#x27;24</td><td rowspan="2">3DGS-based</td><td>30.895</td><td>0.921</td><td>0.039</td><td>3.793</td><td>2.086</td><td>12.523</td><td>53 min</td><td>62</td></tr><tr><td>30.833</td><td>0.918</td><td>0.039</td><td>3.473</td><td>3.055</td><td>11.553</td><td>37 min</td><td>89</td></tr><tr><td rowspan="2">InsTaG [22] CVPR&#x27;25 D3-Talker(Ours)</td><td rowspan="2"></td><td>31.129</td><td>0.921</td><td>0.042</td><td>3.195</td><td>4.411</td><td>10.379</td><td>16 min</td><td>69</td></tr><tr><td>31.518</td><td>0.927</td><td>0.043</td><td>3.194</td><td>6.015</td><td>9.037</td><td>32 min</td><td>65</td></tr></table>

Table 2: Quantitative comparison under the self-driven setting with different training data amounts.
<table><tr><td rowspan="2">Method</td><td colspan="4">DFRF [25]</td><td colspan="4">InsTaG [22]</td><td colspan="4">D3-Talker</td></tr><tr><td> $5 \mathrm { s }$ </td><td>10s</td><td>15s</td><td>20s</td><td> $5 \mathrm { s }$ </td><td>10s</td><td>15s</td><td> $2 0 \mathrm { s }$ </td><td> $5 s$ </td><td>10s</td><td>15s</td><td>20s</td></tr><tr><td>PSNRâ</td><td>29.774</td><td>30.413</td><td>30.772</td><td>30.896</td><td>30.493</td><td>31.129</td><td>31.686</td><td>32.068</td><td>30.595</td><td>31.518</td><td>32.291</td><td>32.566</td></tr><tr><td>SSIMâ</td><td>0.902</td><td>0.909</td><td>0.913</td><td>0.915</td><td>0.916</td><td>0.921</td><td>0.928</td><td>0.932</td><td>0.916</td><td>0.927</td><td>0.934</td><td>0.936</td></tr><tr><td>L PIPS</td><td>0.063</td><td>0.060</td><td>0.059</td><td>0.059</td><td>0.047</td><td>0.042</td><td>0.040</td><td>0.038</td><td>0.050</td><td>0.043</td><td>0.043</td><td>0.042</td></tr><tr><td>LMDâ</td><td>3.691</td><td>3.460</td><td>3.341</td><td>3.290</td><td>3.395</td><td>3.195</td><td>3.182</td><td>3.140</td><td>3.283</td><td>3.194</td><td>3.156</td><td>3.060</td></tr><tr><td>Sync-Câ</td><td>2.775</td><td>3.658</td><td>4.040</td><td>4.329</td><td>3.992</td><td>4.411</td><td>4.888</td><td>4.984</td><td>5.761</td><td>6.015</td><td>6.538</td><td>6.453</td></tr><tr><td>Sync-D</td><td>11.757</td><td>10.927</td><td>10.588</td><td>10.360</td><td>10.661</td><td>10.379</td><td>10.023</td><td>9.900</td><td>9.224</td><td>9.037</td><td>8.659</td><td>8.659</td></tr></table>

## 3.5 Stage-wise Training Details

Pre-training. In the first stage, the pre-training stage, we collect videos of N distinct identities to form the training dataset and initialize the static Gaussian fields for each identity. Following the original 3DGS [15], we utilize a combination of L1 pixel-wise loss $\mathcal { L } _ { 1 }$ and a D-SSIM [30] term LD-SSIM. After that, we add the shared General Field and N identity-dependent Individual Fields to predict the deformation parameters and additionally use the Similarity Contrastive Loss $\mathcal { L } _ { \mathrm { S C } }$ to supervise this process. We also train our Coarse-to-Fine model from scratch using the coarse images rendered during pretraining concurrently. Our primary objective during training is to ensure that the reenacted fine image aligns with the target image. We employ L1 loss and perceptual loss [14, 40] for this purpose:

$$
\mathcal { L } _ { \mathrm { C 2 F } } = \Vert I _ { f } - I _ { t } \Vert + \lambda _ { p } \Vert \varphi ( I _ { f } ) - \varphi ( I _ { t } ) \Vert ,\tag{11}
$$

where $I _ { f }$ is the generated fine image, $I _ { t }$ is the target image, and Ï is the AlexNet [18] used in the perceptual loss. The overall loss function for pre-training is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e } } = \mathcal { L } _ { 1 } + \lambda _ { \mathrm { D - S S I M } } \mathcal { L } _ { \mathrm { D - S S I M } } + \lambda _ { \mathrm { S C } } \mathcal { L } _ { \mathrm { S C } } + \lambda _ { \mathrm { C 2 F } } \mathcal { L } _ { \mathrm { C 2 F } } . } \end{array}\tag{12}
$$

Adaptation. During the adaptation stage, we jointly train an initialized Individual Field with the pre-trained General Field and Coarseto-Fine model. Following InsTaG [22], we add the geometry loss $\mathcal { L } _ { \mathrm { G e o } }$ to regularize the depth and surface normals. The overall loss function can be constructed as:

$$
\mathcal { L } _ { \mathrm { a d a } } = \mathcal { L } _ { 1 } + \lambda _ { \mathrm { D - S S I M } } \mathcal { L } _ { \mathrm { D - S S I M } } + \lambda _ { C } \mathcal { L } _ { C } + \lambda _ { \mathrm { G e o } } \mathcal { L } _ { \mathrm { G e o } } .\tag{13}
$$

## 4 Experiments

## 4.1 Experiment Setting

Datasets. Following existing practices [25, 21, 22], we collect 6 speaking videos provided by AD-NeRF [10], DFRF [25] and HDTF [41], including English and Chinese languages, for testing. For pretraining, we use 5 long speaking videos of different identities from ER-NeRF [20] and Geneface [34] to train the General Field and the Coarse-to-Fine model. There is no overlap between the training set and the test set. Each raw video is resampled to 25 FPS and the resolution is resized as 512 Ã 512.

Comparison Baselines. We compare our method with three NeRFbased methods RAD-NeRF [26], DFRF [25] and ER-NeRF [20], three 3DGS-based methods GaussianTalker [3], TalkingGaussian [21] and InsTaG [22]. For each method, we use their official code, and a DeepSpeech model [11] is used to extract basic audio features. DFRF and InsTaG also have the base model like our method, and we use their official pre-trained models.

Evaluation Metrics. We employ 6 evaluation metrics commonly used in talking head synthesis to comprehensively evaluate the visual quality and lip synchronization. Specifically, we utilize Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Metric (SSIM) [30] to evaluate the image level quality of generated results, Learned Perceptual Image Patch Similarity (LPIPS) [40] to evaluate the feature level quality. For lip synchronization, we use Landmark Distance (LMD) [1] which quantifies the distance between lip landmarks, SyncNet [6] confidence (Sync-C) and distance (Sync-D) to assess the synchronization between lip movements and audio.

## 4.2 Quantitative Evaluation

Results under Self-driven Setting. The evaluation results of selfdriven setting with 10s training data are illustrated in Table 1. It can be seen that our method achieves the best performance in multiple aspects compared to other methods. Specifically, in terms of image quality, our method outperforms other methods in both PSNR and SSIM. Particularly, our method demonstrates significantly better performance in lip synchronization metrics compared to all other methods. This is mainly due to the assistance of the Facial Motion. Additionally, our method maintains excellent performance in both training time and inference FPS, both of which are close to InsTaG. While our method does not achieve the best efficiency, all baselines can not simultaneously ensure high synchronization and real-time inference, in which our method strikes a good balance.

To more comprehensively evaluate the performance of our method in practical applications, we compare quantitative metrics under varying training data amounts with two few-shot methods, DFRF and InsTaG. The results with training data lengths of 5s, 10s, 15s and 20s are shown in Table 2. It is evident that our method still surpasses other methods in most aspects, maintaining high image generation quality and audio visual alignment.

Results under Cross-driven Setting. For the cross-driven setting,

policy

sure

maybe

local

came

are

way

<!-- image-->  
Figure 3: Qualitative comparison of generated key frames under the self-driven setting with different methods. The results depict the lip shape conforming to specific phonemes in the spoken words âpolicyâ, âsureâ, âmaybeâ, âIâ, âlocalâ, âcameâ, âareâ and âwayâ.

Table 3: Quantitative comparison under the cross-driven setting with 10s training data.
<table><tr><td rowspan="2">Method</td><td colspan="2">Audio A (Obama)</td><td colspan="2">Audio B (May)</td></tr><tr><td>Sync-Câ 8.456</td><td>Sync-Dâ 6.841</td><td>Sync-Câ</td><td>Sync-Dâ 5.757</td></tr><tr><td>Ground Truth RAD-NeRF [26]</td><td>3.031</td><td>10.986</td><td>8.809 3.399</td><td>10.814</td></tr><tr><td>DFRF [25]</td><td>3.256</td><td>11.056</td><td>4.196</td><td>10.390</td></tr><tr><td>ER-NeRF [20]</td><td>2.637</td><td>11.892</td><td>2.909</td><td>11.923</td></tr><tr><td>GaussianTalker [3]</td><td>1.860</td><td>13.227</td><td>1.501</td><td>13.558</td></tr><tr><td>TalkingGaussian [21]</td><td>3.810</td><td>10.519</td><td>3.291</td><td>11.284</td></tr><tr><td>InsTaG [22]</td><td>3.510</td><td>11.275</td><td>4.484</td><td>10.876</td></tr><tr><td>D3-Talker</td><td>6.966</td><td>8.345</td><td>6.142</td><td>9.262</td></tr></table>

we extract one unseen audio clip from Obamaâs speaking video [10] and one from May [34] to drive each method and compare lip synchronization. Owing to the absence of ground truth, we only evaluate the accuracy of lip synchronization using SyncNet confidence score (Sync-C) and distance score (Sync-D). We consider the metrics measured from the original video as the ground truth. The results are presented in Table 3, which shows that our D3-Talker offer broader application value and robustness across various scenarios.

## 4.3 Qualitative Evaluation

To more intuitively evaluate the image quality and lip synchronization, we showcase key frames and close-up details of generated videos from self-driven and cross-driven settings. The key frames of three portraits under self-driven setting with 10 seconds training data are displayed in Figure 3. We label the specific word and phoneme corresponding to each frame above the image. It can be observed from the four frames of the first identity that our method behaves better in lip synchronization, accurately capturing various phonetic elements. Particularly, for the frame corresponding to the spoken word âmaybeâ, our method is the only one that achieves lip closure. As for the word âIâ, we also exhibit the most realistic mouth opening, closely matching the style of the ground truth. Furthermore, while other methods generate blurred or severely detail loss in the reconstruction of face and eye blinking for the second and third portrait (red mark), our method generates photorealistic images with delicate details in non-rigid regions like eyes and wrinkles, benefiting from the Coarse-to-Fine module. The visualization results under the

Ground Truth

DFRF

Real3D-Portrait

InsTaG

Ours  
<!-- image-->  
Figure 4: Qualitative comparison of generated key frames under the cross-driven setting with different methods.

Table 4: Ablation study on different settings.
<table><tr><td>Setting Ground Truth</td><td>PSNRâ N/A</td><td>SSIMâ 1</td><td>LMDâ 0</td><td>Sync-Câ 8.685</td></tr><tr><td>D3-Talker</td><td>32.566</td><td>0.936</td><td>3.060</td><td>6.453</td></tr><tr><td>w/o FM</td><td>32.176</td><td>0.932</td><td>3.251</td><td>5.012</td></tr><tr><td>w/o SCLoss</td><td>32.539</td><td>0.936</td><td>3.228</td><td>6.021</td></tr><tr><td>w/o C2F</td><td>32.049</td><td>0.931</td><td>3.126</td><td>6.333</td></tr><tr><td>Audio Only</td><td>32.176</td><td>0.932</td><td>3.251</td><td>5.012</td></tr><tr><td>Facial Motion Only</td><td>32.552</td><td>0.938</td><td>3.133</td><td>6.128</td></tr><tr><td>General Field Only</td><td>30.546</td><td>0.913</td><td>3.472</td><td>4.414</td></tr><tr><td>Individual Field Only</td><td>30.391</td><td>0.901</td><td>3.930</td><td>0.681</td></tr><tr><td>NCLoss (InsTaG) Only</td><td>32.539</td><td>0.936</td><td>3.228</td><td>6.021</td></tr><tr><td>SCLoss (Ours) Only</td><td>32.528</td><td>0.935</td><td>3.157</td><td>6.217</td></tr><tr><td>None</td><td>30.584</td><td>0.916</td><td>3.907</td><td>5.004</td></tr><tr><td>âÂµ</td><td>32.356</td><td>0.935</td><td>3.114</td><td>6.326</td></tr><tr><td> $\Delta \dot { \mu } , \Delta r , \Delta s , \Delta \alpha , \Delta S H$ </td><td>32.552</td><td>0.935</td><td>3.117</td><td>5.495</td></tr><tr><td> $\Delta \mu , \Delta r , \Delta s$ </td><td>32.566</td><td>0.936</td><td>3.060</td><td>6.453</td></tr></table>

cross-driven setting are shown in Figure 4. These results demonstrate that our method maintains impressive lip synchronization capability even with cross-identity and cross-gender audio inputs.

## 4.4 Ablation Study

In this section, we provide multi-faceted ablation studies to validate the effectiveness of our contributions and design choices.

Main Contributions of Our Work. Experiments in Table 4 (lines 3- 6) show that removing Facial Motion (FM) or Similarity Contrastive Loss (SCLoss) results in lower quality in lip synchronization, while the output images without the Coarse-to-Fine (C2F) moduleâs refinement produce relatively poor visual quality. We also visualize the results with 10s training data in Figure 5 for intuitive comparison, demonstrating the effectiveness of our contributions.

Dual-Branch Control Signals. To evaluate the necessity of our Dual-Branch Control Signals, we respectively use only audio or only Facial Motion as the control signals for all deformation fields. The results in Table 4 (lines 7-8) show that both control signals contain specific lip-motion features and combining these two branches achieves the best performance among all settings.

Ground Truth

D3-Talker  
<!-- image-->  
Figure 5: Qualitative results of the ablation study.

Decoupled Deformation Fields. To validate the effectiveness of the two deformation fields in our method, we use the deformations from General Field or Individual Field solely to deform the Gaussian head in Table 4 (lines 9-10). Both deformation fields are valid, and General Field can more effectively influence lip synchronization.

Contrastive Loss. We compare our Similarity Contrastive Loss (SCLoss) with the Negative Contrast loss (NCLoss) of InsTaG. The results in Table 4 (lines 11-13) indicate the superiority of our SCLoss over NCLoss.

Selection of Deformed Attributes. We investigate different selections of Gaussian attributes for deformation in the Individual Field in Table 4 (lines 14-16). Only controlling the center positions leads to loss of reconstruction accuracy. However, deformation of all Gaussian attributes results in a lower performance in lip synchronization.

## 5 Conclusion

In this paper, we present $\mathrm { D } ^ { 3 } .$ -Talker, a 3D-Gaussian based method for few-shot talking head synthesis that addresses key limitations in existing approaches. By incorporating an audio-driven Facial Motion prior, our method effectively learns identity-agnostic general representations. The dual-branch architecture, which uses audio and Facial Motion signals to independently control two Gaussian attribute deformation fields, successfully decouples general and individual deformation predictions. Our comprehensive experiments demonstrate that D3-Talker achieves superior image quality and lip synchronization performance with limited training data, outperforming state-ofthe-art methods across various evaluation metrics and scenarios.

## Acknowledgements

This work was funded by Guangdong Basic and Applied Basic Research Foundation under Grant 2023A1515010688, and Guangdong Provincial Key Laboratory under Grant 2023B1212060076.

## References

[1] L. Chen, Z. Li, R. K. Maddox, Z. Duan, and C. Xu. Lip movements generation at a glance. In Proceedings of the European Conference on Computer Vision (ECCV), pages 520â535, 2018.

[2] L. Chen, R. K. Maddox, Z. Duan, and C. Xu. Hierarchical cross-modal talking face generation with dynamic pixel-wise loss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7832â7841, 2019.

[3] K. Cho, J. Lee, H. Yoon, Y. Hong, J. Ko, S. Ahn, and S. Kim. Gaussiantalker: Real-time talking head synthesis with 3d gaussian splatting. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 10985â10994, 2024.

[4] X. Chu and T. Harada. Generalizable and animatable gaussian head avatar. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[5] X. Chu, Y. Li, A. Zeng, T. Yang, L. Lin, Y. Liu, and T. Harada. GPAvatar: Generalizable and precise head avatar from image(s). In The Twelfth International Conference on Learning Representations, 2024.

[6] J. S. Chung and A. Zisserman. Out of time: automated lip sync in the wild. In Computer Vision-ACCV 2016 Workshops: ACCV 2016 International Workshops, Taipei, Taiwan, November 20-24, 2016, Revised Selected Papers, Part II 13, pages 251â263, 2017.

[7] D. Das, S. Biswas, S. Sinha, and B. Bhowmick. Speech-driven facial animation using cascaded gans for learning of motion and texture. In European Conference on Computer Vision, pages 408â424, 2020.

[8] K. Deng, D. Zheng, J. Xie, J. Wang, W. Xie, L. Shen, and S. Song. Degstalk: Decomposed per-embedding gaussian fields for hairpreserving talking face synthesis. In ICASSP 2025, 2025.

[9] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139â144, 2020.

[10] Y. Guo, K. Chen, S. Liang, Y.-J. Liu, H. Bao, and J. Zhang. Ad-nerf: Audio driven neural radiance fields for talking head synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5784â5794, 2021.

[11] A. Hannun, C. Case, J. Casper, B. Catanzaro, G. Diamos, E. Elsen, R. Prenger, S. Satheesh, S. Sengupta, A. Coates, et al. Deep speech: Scaling up end-to-end speech recognition. arXiv preprint arXiv:1412.5567, 2014.

[12] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

[13] X. Ji, H. Zhou, K. Wang, W. Wu, C. C. Loy, X. Cao, and F. Xu. Audiodriven emotional video portraits. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14080â 14089, 2021.

[14] J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In Computer VisionâECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11- 14, 2016, Proceedings, Part II 14, pages 694â711, 2016.

[15] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):139â1, 2023.

[16] H. Kim, P. Garrido, A. Tewari, W. Xu, J. Thies, M. Niessner, P. PÃ©rez, C. Richardt, M. ZollhÃ¶fer, and C. Theobalt. Deep video portraits. ACM Transactions on Graphics (TOG), 37(4):1â14, 2018.

[17] D. P. Kingma, M. Welling, et al. Auto-encoding variational bayes, 2013.

[18] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25, 2012.

[19] D. Li, K. Zhao, W. Wang, B. Peng, Y. Zhang, J. Dong, and T. Tan. Aenerf: Audio enhanced neural radiance field for few shot talking head synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 3037â3045, 2024.

[20] J. Li, J. Zhang, X. Bai, J. Zhou, and L. Gu. Efficient region-aware neural radiance fields for high-fidelity talking portrait synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7568â7578, 2023.

[21] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu. Talkinggaussian: Structure-persistent 3d talking head synthesis via gaussian splatting. In European Conference on Computer Vision, pages 127â145, 2024.

[22] J. Li, J. Zhang, X. Bai, J. Zheng, J. Zhou, and L. Gu. Instag: Learning personalized 3d talking head from few-second video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[23] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021.

[24] K. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, and C. Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings of the 28th ACM international Conference on Multimedia,

pages 484â492, 2020.

[25] S. Shen, W. Li, Z. Zhu, Y. Duan, J. Zhou, and J. Lu. Learning dynamic facial radiance fields for few-shot talking head synthesis. In European Conference on Computer Vision, 2022.

[26] J. Tang, K. Wang, H. Zhou, X. Chen, D. He, T. Hu, J. Liu, G. Zeng, and J. Wang. Real-time neural radiance talking portrait synthesis via audio-spatial decomposition. arXiv preprint arXiv:2211.12368, 2022.

[27] J. Thies, M. Elgharib, A. Tewari, C. Theobalt, and M. NieÃner. Neural voice puppetry: Audio-driven facial reenactment. European Conference on Computer Vision, 2020.

[28] A. Trevithick and B. Yang. Grf: Learning a general radiance field for 3d representation and rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15182â15192, 2021.

[29] Q. Wang, Z. Wang, K. Genova, P. P. Srinivasan, H. Zhou, J. T. Barron, R. Martin-Brualla, N. Snavely, and T. Funkhouser. Ibrnet: Learning multi-view image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4690â 4699, 2021.

[30] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600â612, 2004.

[31] Y. Xie, T. Feng, X. Zhang, X. Luo, Z. Guo, W. Yu, H. Chang, F. Ma, and F. R. Yu. Pointtalk: Audio-driven dynamic lip point cloud for 3d gaussian-based talking head synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 8753â8761, 2025.

[32] Q. Xu, Z. Xu, J. Philip, S. Bi, Z. Shu, K. Sunkavalli, and U. Neumann. Point-nerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5438â5448, 2022.

[33] Z. Ye, J. He, Z. Jiang, R. Huang, J. Huang, J. Liu, Y. Ren, X. Yin, Z. Ma, and Z. Zhao. Geneface++: Generalized and stable real-time audiodriven 3d talking face generation. arXiv preprint arXiv:2305.00787, 2023.

[34] Z. Ye, Z. Jiang, Y. Ren, J. Liu, J. He, and Z. Zhao. Geneface: Generalized and high-fidelity audio-driven 3d talking face synthesis. The Eleventh International Conference on Learning Representations, 2023.

[35] Z. Ye, T. Zhong, Y. Ren, Z. Jiang, J. Huang, R. Huang, J. Liu, J. He, C. Zhang, Z. Wang, et al. Mimictalk: Mimicking a personalized and expressive 3d talking face in minutes. Advances in Neural Information Processing Systems, 37:1829â1853, 2024.

[36] Z. Ye, T. Zhong, Y. Ren, J. Yang, W. Li, J. Huang, Z. Jiang, J. He, R. Huang, J. Liu, et al. Real3d-portrait: One-shot realistic 3d talking portrait synthesis. The Twelfth International Conference on Learning Representations, 2024.

[37] A. Yu, V. Ye, M. Tancik, and A. Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4578â4587, 2021.

[38] H. Yu, Z. Qu, Q. Yu, J. Chen, Z. Jiang, Z. Chen, S. Zhang, J. Xu, F. Wu, C. Lv, and G. Yu. Gaussiantalker: Speaker-specific talking head synthesis via 3d gaussian splatting. In Proceedings of the 32nd ACM International Conference on Multimedia, page 3548â3557, 2024.

[39] E. Zakharov, A. Shysheya, E. Burkov, and V. Lempitsky. Few-shot adversarial learning of realistic neural talking head models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9459â9468, 2019.

[40] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586â595, 2018.

[41] Z. Zhang, L. Li, Y. Ding, and C. Fan. Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3661â3670, 2021.

[42] H. Zhao, H. Wang, C. Yang, and W. Shen. Chase: 3d-consistent human avatars with sparse inputs via gaussian splatting and contrastive learning. arXiv e-prints, pages arXivâ2408, 2024.

[43] W. Zhong, C. Fang, Y. Cai, P. Wei, G. Zhao, L. Lin, and G. Li. Identitypreserving talking face generation with landmark and appearance priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9729â9738, June 2023.

[44] X. Zhu, Z. Lei, X. Liu, H. Shi, and S. Z. Li. Face alignment across large poses: A 3d solution. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 146â155, 2016.