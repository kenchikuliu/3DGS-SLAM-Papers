# EmbedTalk: Triplane-Free Talking Head Synthesis using Embedding-Driven Gaussian Deformation

Arpita Saggar1 , Jonathan C. Darling2 , Duygu Sarikaya1 , and David C. Hogg1

1 School of Computer Science, University of Leeds

2 Leeds Institute of Medical Education, School of Medicine, University of Leeds

<!-- image-->  
Fig. 1: Nearly all prior work on 3DGS-based talking head synthesis uses tri-planes, where approximation errors can hamper audio-visual alignment. EmbedTalk replaces tri-planes with learnable per-Gaussian embeddings, resulting in more accurate mouth movements and reduced computational overhead.

Abstract. Real-time talking head synthesis increasingly relies on deformable 3D Gaussian Splatting (3DGS) due to its low latency. Tri-planes are the standard choice for encoding Gaussians prior to deformation, since they provide a continuous domain with explicit spatial relationships. However, tri-plane representations are limited by grid resolution and approximation errors introduced by projecting 3D volumetric fields onto 2D subspaces. Recent work has shown the superiority of learnt embeddings for driving temporal deformations in 4D scene reconstruction. We introduce EmbedTalk, which shows how such embeddings can be leveraged for modelling speech deformations in talking head synthesis. Through comprehensive experiments, we show that EmbedTalk outperforms existing 3DGS-based methods in rendering quality, lip synchronisation, and motion consistency, while remaining competitive with stateof-the-art generative models. Moreover, replacing the tri-plane encoding with learnt embeddings enables significantly more compact models that achieve over 60 FPS on a mobile GPU (RTX 2060 6 GB). Our code will be placed in the public domain on acceptance.

Keywords: Talking Head Synthesis Â· 3D Gaussian Splatting

## 1 Introduction

Synthesising audio-driven talking heads in real-time is an important task in computer vision, with applicability in film production, teleconferencing, and virtual assistants. The standard pipeline involves the deformation of a canonical identity representation based on an arbitrary speech sample. Other control signals, such as emotion, gaze direction, or distance to camera may also be used for deformation. The canonical identity is typically represented in either two or three dimensional space. In recent years, methods for synthesising talking heads have been dominated by image-based generative models [2,20,29,44,45] and radiance fields [12,15,24,25,34]. Whilst generative models produce high resolution results, they typically do not provide pose control, are slow to render, and can generate uncharacteristic movements which reduce realism and applicability. Within radiance fields, 3D Gaussian Splatting [19] has emerged as the preferred choice due to its fast rendering speed and low memory requirements, compared to neural radiance fields (NeRFs) [30].

Generating talking heads with 3D Gaussian Splatting involves deforming a canonical Gaussian representation using the speech signal. To encode spatial continuity among the otherwise discrete Gaussians, each Gaussian is projected onto 2D subspaces through a tri-plane encoder prior to deformation. [10, 12, 16, 23, 24, 33]. However, tri-plane representations suffer from mirroring artefacts caused by feature entanglement between subspaces [22]. Furthermore, using 2D planes to deform a 3D volumetric field introduces approximation errors which subsequently affect audio-visual alignment, as shown in Figure 1. Finally, many previous methods rely on imprecise facial tracking for inferring camera pose, which leads to wobbling around the boundary of the face [1].

Prior work has demonstrated the superiority of learnable embeddings for modelling Gaussian deformations in 4D (3D + time) scene reconstruction [3]. In this work, we extend the embedding-driven deformation paradigm to talking head synthesis. We introduce EmbedTalk, which leverages per-Gaussian embeddings to generate stable talking heads with high audio-visual alignment. To accurately reconstruct high-frequency displacements in the mouth region, we apply positional encodings to the Gaussian embeddings. Spatial coherence is enforced through a local smoothness constraint that encourages similar embeddings for neighbouring Gaussians. Furthermore, we address the head wobbling issue by initialising Gaussians with a stable, dense reconstruction obtained via COLMAP [37]. Experimental results establish EmbedTalkâs superiority in facial fidelity, lip-synchronisation, and motion consistency compared to previous 3DGS-based works. Furthermore, by generating mouth movements consistent with an identityâs style, our method achieves higher realism than state-of-theart generative models that often exaggerate motion. Our key contributions are:

â A method to leverage learnt Gaussian embeddings for modelling speech deformations in audio-driven talking head synthesis

â A comprehensive comparative evaluation with recent 3DGS-based and generative methods, covering quantitative assessment, visual comparisons and a user study

â Ablations and experimental variations validating the efficacy of our design choices

## 2 Related Work

## 2.1 Audio-Driven Talking Head Synthesis

Various approaches have been proposed to generate videos of talking heads conditioned on arbitrary speech samples. Generative models are a popular choice for accomplishing this task since they do not require identity-specific training. Earlier work made use of generative adversarial networks (GANs) to predict deformations in the two-dimensional image space [2,35]. More recent works replace GANs with diffusion [9, 27, 29, 44, 45] and flow-based models [20, 39]. Some approaches like EDTalk [39] and EchoMimic [9] allow explicit pose control through a separate driving video. However, most generative methods are unable to faithfully recreate the identityâs speaking style and often produce uncharacteristic motions, leading to reduced realism [23].

In contrast, approaches based on radiance fields train a separate model for each new identity, enabling generation of personalised talking heads. Many such methods make use of neural radiance fields [30], with deformations generally driven via a tri-plane representation [7]. Earlier methods like AD-NeRF [15] directly conditioned NeRF training on speech input, resulting in slow training and rendering. RAD-NeRF [40] introduced grid-based NeRFs for quicker inference, but its rendering quality was limited by hash collisions. ER-NeRF [25] alleviated this issue through the use of tri-plane encoders, which have since emerged as the de-facto choice for intermediate representations [34, 48]. However, most NeRF-based methods are still unsuitable for real-time rendering due to the ray marching step, which requires evaluating the MLP at every sampled point in 3D space. Recently, 3D Gaussian Splatting has emerged as as popular alternative to NeRFs due its fast training and low latency. Similar to NeRF-based approaches, methods that use 3DGS also employ tri-planes to encode Gaussians prior to deformation [10, 12, 16, 23, 24, 33]. Methods like TalkingGaussian [23], InsTaG [24], and DEGSTalk [12] further decompose Gaussian fields into separate face and mouth branches to learn movements with different frequencies. But despite their improved lip-synchronisation and real-time rendering speeds, these methods are still prone to approximation errors introduced by tri-planes, which propagate to the deformation stage. Our work improves on current baselines by modelling speech deformations through learnt per-Gaussian embeddings instead of tri-plane representations.

## 2.2 Deformable 3DGS

General methods for deforming 3D Gaussians are typically aimed at 4D scene reconstruction. Here, the deformation signal is modelled as a temporal embedding conditioned on frame order. Both Deformable 3D Gaussians [47] and Dynamic 3D Gaussians [28] directly deform the Gaussian attributes without using any intermediate representations. 4DGS [46] combines conditional 3D Gaussians with a marginal 1D Gaussian for time-dependent view synthesis. 4D-GS [43] decomposes Gaussian centres into a multi-resolution HexPlane [6] (an extension of the tri-plane with an additional time axis) before deformation. Free-TimeGS [42] endows each Gaussian with a motion function to predict deformations. E-D3DGS [3] defines learnable embeddings for each Gaussian to model time-based deformations. However, temporal embeddings do not encode any acoustic information, and are therefore unsuitable for synthesising motions that arise from speech. In this work, we show how the Gaussian embedding-driven deformation setup can be adapted to talking head synthesis.

## 3 Method

<!-- image-->  
Fig. 2: EmbedTalk begins with a talking portrait video. The video frames enable a dense reconstruction of the head that initialises the 3D Gaussians. Each Gaussian is also associated with a learnable embedding $z _ { g } .$ For each frame, the corresponding speech signal a and upper-face movements e are fed into the deformation MLP, along with a positional encoding of $z _ { g }$ to predict the Gaussian deformations $( \varDelta \mu , \varDelta \alpha )$ . The deformed Gaussians are passed to the rasteriser, along with the viewing direction (camera), to render the head onto the combined torso and scene background.

As illustrated in Figure 2, EmbedTalk deforms learnable Gaussian embeddings to synthesise an identity-specific talking head from a monocular video and corresponding speech audio. We begin with a brief overview of 3D Gaussian Splatting (Section 3.1). Next, we formalise our problem and describe the way in which we use the embedding-driven approach to synthesise a talking head from audio (Section 3.2). Finally, Section 3.3 provides details of the training process.

## 3.1 Preliminary: 3D Gaussian Splatting

3DGS [19] optimises a set of anisotropic 3D Gaussians using differentiable tile rasterisation to learn a static 3D scene representation. Each Gaussian is defined using a centre mean $\mu \in \mathbb { R } ^ { 3 }$ and covariance matrix $\varSigma \in \mathbb { R } ^ { 3 \times 3 }$ as shown below:

$$
G ( x ) = e ^ { \frac { - 1 } { 2 } ( x - \mu ) ^ { T } \Sigma ^ { - 1 } ( x - \mu ) }\tag{1}
$$

for a 3D point $x \in \mathbb { R } ^ { 3 }$ . The covariance matrix Î£ is decomposed into a scaling matrix $S$ and a rotation matrix R, given by the scaling factor $s \in \mathbb { R } ^ { 3 }$ and the rotation quaternion $q \in \mathbb { R } ^ { 4 }$ :

$$
\Sigma = R S S ^ { T } R ^ { T }\tag{2}
$$

Each Gaussian also has an opacity value $\alpha \in \mathbb { R }$ , and a colour feature, $f \in$ R3(d+1)(d+1), described using a set of spherical harmonics with degree d. To render images, the 3D Gaussians are projected to 2D by calculating the covariance matrix $\Sigma ^ { \prime }$ in the camera space using the viewing transformation W :

$$
\Sigma ^ { \prime } = J W \Sigma W ^ { T } J ^ { T }\tag{3}
$$

where J is the Jacobian of the affine approximation of the projective transformation. The colour of each pixel is computed by blending all $\mathcal { N }$ depth-ordered Gaussians that overlap the pixel:

$$
C = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } ^ { \prime } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { \prime } )\tag{4}
$$

where $c _ { i }$ is the colour of each point determined using the colour feature $f _ { i }$ with the viewing direction, and $\alpha ^ { \prime }$ is the product of the Gaussian opacity Î± with the camera space covariance matrix $\Sigma ^ { \prime }$ . Cameras calibrated with Structure-from-Motion (SfM) [37, 38] serve as input, and the Gaussians are initialised using the sparse point cloud created during SfM.

## 3.2 3DGS for Talking Head Synthesis

Generating talking heads with 3DGS requires a talking portrait video $V = \{ v _ { n } \}$ consisting of |V | frames, along with corresponding speech signals $A = \left\{ a _ { n } \right\}$ and optional facial control signals $E = \{ e _ { n } \}$ . The goal is to construct a canonical 3D representation of the head $\left( G _ { c a n o n i c a l } \right)$ , and learn a module to predict deformations to $G _ { c a n o n i c a l }$ for each frame based on $a _ { n }$ and $e _ { n } ,$ where $1 \leq n \leq | V |$ . This involves projecting one or more Gaussian primitives to a set of continuous, lower dimensional subspaces. This lower dimensional representation is then combined with the audio embedding, and other control signals (if applicable), and reprojected back into the Gaussian space to produce an attribute-wise deformation with respect to the canonical representation. The reprojection is achieved using a decoder multi-layer perceptron (MLP) as shown below:

$$
G _ { c a n o n i c a l } = \{ \mu , r , s , f , \alpha \}\tag{5}
$$

$$
\{ \varDelta \mu _ { n } , \varDelta r _ { n } , \varDelta s _ { n } , \varDelta f _ { n } , \varDelta \alpha _ { n } \} = M L P ( H ( G _ { c a n o n i c a l } ) \oplus a _ { n } \oplus e _ { n } )\tag{6}
$$

$$
G _ { d e f o r m } ^ { n } = \{ \mu + \Delta \mu _ { n } , r + \Delta r _ { n } , s + \Delta s _ { n } , f + \Delta f _ { n } , \alpha + \Delta \alpha _ { n } \}\tag{7}
$$

where H is typically a multi-resolution tri-plane encoder [10, 23â25, 34] and â denotes concatenation.

But despite their popularity in radiance field-based talking head generation, tri-plane representations are limited by grid resolution and approximation errors. We therefore do not use tri-planes to drive the Gaussian deformations for our method. Instead, we start with the embedding-driven deformation paradigm introduced in E-D3DGS [3], where each Gaussian has a learnable embedding $z _ { g } \in$ $\mathbb { R } ^ { 3 2 }$ , in addition to the canonical attributes. E-D3DGS uses temporal embeddings (approximated from frame numbers) to deform Gaussian embeddings for 4D scene reconstruction. To allow deformations to be driven by acoustic features such as phonemes, pitch, and amplitude of the audio signal, we replace the temporal embeddings in E-D3DGS with audio embeddings derived from the speech signal. Furthermore, 4D scene reconstruction is dominated by large and (relatively) low-frequency movements, in contrast to portrait lip-synchronisation, where small (relative to the face) and high-frequency mouth movements are the primary source of deformation. To capture these high-frequency details, we apply alternating sine and cosine functions of varying frequencies to the per-Gaussian embeddings on input to the MLP, typically referred to as positional encodings (Î³) [30]. However, our motivation for applying a positional encoding is not to provide a notion of sequence [41]. For EmbedTalk, this mapping into a higher dimensional space enables the embeddings to disentangle motion discontinuous from smooth deformations (for instance, Gaussians at the lips moving apart due to speech, while also shifting together laterally due to a head tilt). Additionally, we note the presence of facial movements that are uncorrelated with the speech signal (eye blink, brow raise). We provide this information to the deformation module through a positional encoding of the feature set E, comprising of six facial action units [13]. Our Gaussian embedding-based deformation is given by:

$$
G _ { c a n o n i c a l } = \{ \mu , r , s , f , \alpha , z \}\tag{8}
$$

$$
\{ \varDelta \mu _ { n } , \varDelta \alpha _ { n } \} = M L P ( \gamma ( z ) \oplus a _ { n } \oplus \gamma ( e _ { n } ) )\tag{9}
$$

$$
G _ { d e f o r m } ^ { n } = \{ \mu + \varDelta \mu _ { n } , r , s , f , \alpha + \varDelta \alpha _ { n } \}\tag{10}
$$

The deformation module for EmbedTalk is a shallow MLP with a separate prediction head for each deformed attribute (Âµ and Î±). Unlike contemporary methods that deform all Gaussian attributes [3, 10], or attributes related to Gaussian size and orientation [12,23,24], we choose to deform only position and opacity. This selection is grounded in the fact that facial animation primarily involves changes in motion (head movements, mouth opening and closing) and visibility (teeth/tongue appearing or disappearing). The facial structure and characteristics (size of nose, distance between eyes) remain unchanged. We validate our selection through experimental variations presented in Section 4.3.

## 3.3 Training

Initialisation: Prior work infers camera poses for 3DGS by fitting a 3D Morphable Model (3DMM) [32] to the input frames [10,12,23,24]. The Gaussians are then initialised with a random point cloud [12,23,24], or using the coordinates of mesh vertices obtained from the 3DMM fitting [10]. However, 3DMM fitting is imprecise and often leads to wobbling around the facial region [1]. Additionally, SfM points-based initialisation provides better reconstruction than random initialisation [19]. Consequently, we initialise the Gaussians for EmbedTalk using a dense reconstruction obtained from COLMAP, downsampled $\mathrm { t o } \le 1 0 0 \mathrm { K }$ points (to prevent out of memory errors). During training, the number of Gaussians is controlled through adaptive densification and pruning strategies [19].

Rendering: By default, 3DGS renders Gaussians on a single colour background image, which is not ideal for our approach. Following previous radiance field-based talking head synthesis [15, 25], we render the deformed head onto a combined image containing the torso and scene background using a modified rasteriser [10]. This prevents artefacts around the contours of the face.

Optimisation: We jointly optimise the canonical Gaussian attributes, the per-Gaussian embeddings, and the deformation module to minimise the rendering loss $\mathcal { L } _ { 1 }$ . We also utilise a perceptual loss, $\mathcal { L } _ { L P I P S } \left[ 4 9 \right]$ , to capture finer details for the full image as well as the localised mouth region (obtained through facial landmark detection [5]). Furthermore, we note that nearby Gaussians tend to share similar canonical attributes. Since our deformation is embedding-driven, it follows that nearby Gaussians should also have similar embeddings to promote motion consistency. Motivated by prior work [3,28], we apply a local smoothness constraint to encourage similar embeddings for nearby Gaussians, given by:

$$
\mathcal { L } _ { e m b \_ r e g } = \frac { 1 } { k | G | } \sum _ { i \in G } \sum _ { j \in \mathrm { K N N } _ { i ; k } } ( w _ { i , j } | | \mathbf { z } _ { \mathrm { g } i } - \mathbf { z } _ { \mathrm { g } j } | | _ { 2 } )\tag{11}
$$

where $k = 2 0$ is the number of neighbouring Gaussians, $w _ { i , j } = e ^ { ( - \lambda _ { w } | | \mu _ { j } - \mu _ { i } | | _ { 2 } ^ { 2 } ) }$ is the weighting factor, and $\lambda _ { w }$ is set to 2000 [28]. To reduce the computational overhead, the k nearest neighbours are computed only post densification. Finally, we minimise the mean of the Gaussian opacities to mitigate floaters. The total loss for our setup is given by:

$$
\begin{array} { r } { \mathcal { L } _ { t o t a l } = \mathcal { L } _ { 1 } \big ( v _ { n } , \hat { v } _ { n } \big ) + \lambda _ { f a c e } \mathcal { L } _ { L P I P S } \big ( v _ { n } , \hat { v } _ { n } \big ) + \lambda _ { m o u t h } \mathcal { L } _ { L P I P S } \big ( v _ { n } ^ { m } , \hat { v } _ { n } ^ { m } \big ) } \\ { + \mathcal { L } _ { e m b \_ r e g } + \lambda _ { o p a } \mathcal { L } _ { o p a c i t y } } \end{array}\tag{12}
$$

where $v _ { n } \in V$ represents the ground truth frame, $\hat { v } _ { n }$ is the predicted frame, $v _ { n } ^ { m }$ denotes the ground truth frame cropped to the mouth region, and $\hat { v } _ { n } ^ { m }$ is the prediction cropped to the mouth region.

## 4 Experiments

Dataset and Comparisons: We source five high-definition audio-visual clips from open-source video datasets previously used in similar works [23, 48, 50]. The clips have an average length of around 6300 frames and are sampled at 25 frames per second (FPS). They comprise three male identities (Macron, Paul, Obama) and two female identities (May, Stabenow). Each clip is cropped (to centre the portrait) and resized to size 512x512, except the Obama clip, which has size 450x450. We compare our approach with recent 3DGS-based methods: TalkingGaussian [23], GaussianTalker [10] and DEGSTalk [12]. To contextualise our work beyond 3DGS-based synthesis, we also compare EmbedTalk with state-of-the-art image-based generative methods: AniTalker [27], FLOAT [20], KDTalker [45] and Sonic [18].

Implementation Details: All 3DGS-based methods, including ours, require identity-specific training. To ensure fair comparison, we use the same train-test split (10:1 ratio) and audio encoder (a pretrained HuBERT model [17]) for each of these methods. In contrast, generative methods are not tailored to any specific identity, and we report their performance using the pretrained weights. Our method is implemented using PyTorch [31]. For each identity, we train EmbedTalk for 50,000 iterations using the Adam optimiser [21]. All other Gaussianbased methods are trained using their default experimental configurations. The action units for the expression set E are extracted using OpenFace [4]. For the loss function, we set $\lambda _ { f a c e } = 0 . 0 1 , \lambda _ { m o u t h } = 0 . 0 0 2$ , and $\lambda _ { o p a } = 0 . 0 0 0 1$ . All experiments are conducted on a single NVIDIA L40S 48 GB GPU. The training takes around 1 hour for each identity (Table 3).

Table 1: Results for the self-driven setting, with the top three results in red (first), orange (second) and yellow (third). EmbedTalk achieves the best rendering quality, (personalised) lip-sync and motion consistency, along with a high inference speed.
<table><tr><td>Method</td><td>Framework</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>LMDâ</td><td>Sync-Câ</td><td>FPS (L40S)â</td><td>FVMDâ</td></tr><tr><td>Ground Truth</td><td>-</td><td>â</td><td>1</td><td>0</td><td>0</td><td>7.822</td><td>-</td><td>0</td></tr><tr><td>GaussianTalker [10]</td><td>3DGS</td><td>31.563</td><td>0.923</td><td>0.041</td><td>2.756</td><td>6.379</td><td>128</td><td>358.023</td></tr><tr><td>TalkingGaussian [23]</td><td>3DGS</td><td>31.674</td><td>0.923</td><td>0.034</td><td>2.461</td><td>6.024</td><td>303</td><td>347.990</td></tr><tr><td>DEGSTalk [12]</td><td>3DGS</td><td>32.511</td><td>0.930</td><td>0.035</td><td>2.545</td><td>6.028</td><td>296</td><td>406.064</td></tr><tr><td>KDTalker [45]</td><td>Diffusion</td><td>15.542</td><td>0.678</td><td>0.387</td><td>7.049</td><td>6.641</td><td>28</td><td>2974.676</td></tr><tr><td>Sonic [18]</td><td>Diffusion</td><td>18.901</td><td>0.740</td><td>0.216</td><td>5.042</td><td>8.094</td><td>&lt; 2</td><td>2313.824</td></tr><tr><td>AniTalker [27]</td><td>Diffusion</td><td>17.115</td><td>0.723</td><td>0.299</td><td>5.433</td><td>6.019</td><td>30</td><td>2445.557</td></tr><tr><td>FLOAT [20]</td><td>Flow Matching</td><td>17.999</td><td>0.718</td><td>0.256</td><td>5.192</td><td>7.101</td><td>32</td><td>2515.666</td></tr><tr><td>EmbedTalk (Ours)</td><td>3DGS</td><td>35.186</td><td>0.961</td><td>0.021</td><td>2.444</td><td>6.520</td><td>294</td><td>147.384</td></tr></table>

## 4.1 Quantitive Evaluation

We employ PSNR (Peak signal-to-noise ratio), SSIM (structural similarity index measure) and LPIPS (Learned Perceptual Image Patch Similarity) to assess rendering quality. To measure audio-visual alignment, we use Landmark Distance (LMD) [8], which measures the distance between mouth landmarks for the real and predicted frames, and the confidence scores (Sync-C) from the SyncNet model [11]. Furthermore, we leverage FrÃ©chet Video Motion Distance (FVMD) [26] to measure the motion consistency of the generated talking portrait videos. FVMD computes the FrÃ©chet distance between motion features of generated and ground-truth videos, derived using key point velocities and accelerations. We also report inference speed by measuring FPS (frames per second) on an enterprise GPU that can accommodate all models.

Table 2: Results for the cross-driven setting, with the top three results in red (first), orange (second) and yellow (third). Generative methods dominate due to exaggerated mouth movements. EmbedTalk has the best overall score among Gaussian methods.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Framework</td><td colspan="3">Sync-Câ</td></tr><tr><td>Cross-Gender</td><td>Cross-Lingual</td><td>Overall</td></tr><tr><td>GaussianTalker [10]</td><td>3DGS</td><td>6.226</td><td>2.380</td><td>5.511</td></tr><tr><td>TalkingGaussian [23]</td><td>3DGS</td><td>5.258</td><td>3.559</td><td>4.831</td></tr><tr><td>DEGSTalk [12]</td><td>3DGS</td><td>2.335</td><td>1.469</td><td>2.292</td></tr><tr><td>KDTalker [45]</td><td>Diffusion</td><td>7.208</td><td>7.348</td><td>7.067</td></tr><tr><td>Sonic [18]</td><td>Diffusion</td><td>8.142</td><td>9.336</td><td>8.302</td></tr><tr><td>AniTalker [27]</td><td>Diffusion</td><td>6.719</td><td>7.482</td><td>6.600</td></tr><tr><td>FLOAT [20]</td><td>Flow Matching</td><td>6.444</td><td>6.882</td><td>6.643</td></tr><tr><td>EmbedTalk (Ours)</td><td>3DGS</td><td>6.023</td><td>4.280</td><td>6.009</td></tr></table>

Videos are rendered under two different settings. The first is the self-driven setting, where videos generated for a given identity are driven by unseen speech sourced from the same identity (the 10:1 train-test split). The second is the cross-driven setting, where videos are generated using: (1) speech from other identities, and (2) synthetic speech samples created using a text-to-speech (TTS) model. For this, we source real (human) speech from the HDTF (High-Definition Talking Face) dataset [50], and generate AI samples using the voices Adaline, Autumn, Hale, Iron Rose, Jamie and Michael from ElevenLabs [14]. The text for the TTS model is taken from the works of William Shakespeare.

Table 1 reports performance for all methods under the self-driven setting. EmbedTalk performs best on all rendering metrics and achieves the most consistent motion. We attribute this to the stable dense initialisation (which mitigates wobbling effects), and positional encodings that capture fine-grained information. Our method also performs best for identity-specific lip synchronisation (LMD) and has the highest Sync-C score among all 3DGS-based methods. Generative methods fare poorly on rendering metrics due to lack of pose information, and are limited by inference speed. However, they achieve high Sync-C scores due to the production of exaggerated mouth movements, with Sonicâs score being even higher than the ground truth. For the cross-driven setting, we only report Sync-C due to lack of ground truth data. We further partition the cross-driven results by gender (model trained on male/female identity, driven by female/male speech) and language (model trained on one language, driven by speech from another language). Whilst partitioned results are presented for all methods, it is worth noting that cross-gender and cross-lingual settings only apply to identityspecific training regimes (3DGS). Within the cross-driven setting, generative methods again dominate the Sync-C scores, but our method has the highest overall Sync-C amongst the Gaussian methods. Interestingly, cross-lingual and cross-gender scores tend to be higher than the overall scores in many instances.

For the Gaussian methods, we additionally report training times, model size and FPS with a mobile (laptop) GPU in Table 3. Since EmbedTalk does not require a tri-plane encoder, our models are around 2x to 6x smaller than methods which use tri-planes. The inference speed-up provided through this elimination is best realised on a mobile GPU (RTX 2060 6GB), with EmbedTalk being nearly twice as fast as other methods. However, our training takes longer since we begin with a dense point cloud and focus on pruning Gaussians during training, as opposed to growing Gaussians from randomly initialised point clouds.

Table 3: Computational costs for each of the 3DGS-based methods. By eliminating tri-planes, EmbedTalk yields compact models that achieve 60+ FPS on a mobile GPU.
<table><tr><td>Method</td><td>Deformation Field</td><td>Training Time â</td><td>FPS (RTX 2060) â</td><td>Model Size â</td></tr><tr><td>GaussianTalker [10]</td><td>Tri-plane</td><td>03:06:19</td><td>33</td><td>19.51 MB</td></tr><tr><td>TalkingGaussian [23]</td><td>Tri-plane</td><td>00:27:50</td><td>38</td><td>27.08 MB</td></tr><tr><td>DEGSTalk [12]</td><td>Tri-plane + Embedddings</td><td>00:37:26</td><td>37</td><td>58.69 MB</td></tr><tr><td>EmbedTalk (Ours)</td><td>Embedddings</td><td>1:01:49</td><td>61</td><td>10.20 MB</td></tr></table>

## 4.2 Qualitative Evaluation

To qualitatively assess the rendering quality and audio-visual alignment, we present rendered and ground truth frames from the self-driven setting. A video file showing comparisons among renderings is also included in the supplementary materials. Figures 1 and 3 provide comparisons with 3DGS-based works and Figure 4 shows comparisons with generative methods. Among Gaussian Splatting-based approaches, EmbedTalk generates the most faithful lip movements, even when the mouth opening is narrow, while other methods often default to a closed-mouth state. For generative models, the mouth movements are accurate but tend to be uncharacteristically large, especially for FLOAT and Sonic, leading to reduced realism. KDTalker and FLOAT also tend to produce frames where the speakerâs gaze is directed away from the camera, which may reduce applicability in interactive settings. We also illustrate the wobbling effect, mentioned earlier, in 3DGS-based works through accumulated differences between consecutive frames (over a 20 frame interval). The white lines in the upper region of the heads in Figure 5 indicate temporal flickering in the renderings for GaussianTalker, TalkingGaussian and DEGSTalk. In contrast, EmbedTalk produces stable, wobble-free renderings that align closely with the ground truth.

GaussianTalker  
<!-- image-->

TalkingGaussian  
<!-- image-->

DEGSTalk  
<!-- image-->

EmbedTalk  
<!-- image-->

Ground Truth  
<!-- image-->  
Fig. 3: Qualitative comparison with recent 3DGS-based works. EmbedTalk reconstructs narrow mouth openings more faithfully than other methods.  
AniTalker

FLOAT  
KDTalker  
Sonic  
<!-- image-->  
EmbedTalk  
Ground Truth

Fig. 4: Qualitative comparison with generative methods. Despite accurate lip-sync, generative models produce exaggerated movements that reduce realism.  
<!-- image-->

<!-- image-->

DEGSTalk  
<!-- image-->

EmbedTalk  
<!-- image-->

Ground Truth  
<!-- image-->  
Fig. 5: Differences between consecutive frames accumulated over a 20 frame interval. The white pixels in the upper head region indicate the presence of temporal flickering.

Beyond visual comparisons, we also conducted a user study to assess the generated videos on three aspects: Image Quality, Lip-Synchronisation, and Video Realness. Three videos for each of the five identities (one self-driven and two cross-driven) were paired with corresponding videos generated using the seven comparative baselines. We recruited 20 assessors through the Prolific platform [36]. The participants were shown the paired videos and asked to select which video was better in terms of the evaluation aspect, the choices being âleftâ, ârightâ or âboth are the sameâ. Participants were blinded to the methods used to generate the videos for any given pair. They were asked to focus on sharpness of video frames and preservation of fine details for image quality, and audio-visual alignment for lip synchronisation. For video realness, the objective was to select the video that seemed less like it had been AI-generated. The protocol for evaluation was reviewed and approved by the Faculty Research Ethics Committee (FREC) for Engineering and Physical Sciences at the University of Leeds (EPS FREC - 2025 3516-5033). Participants were compensated for their time using the national living wage as the rate, scaled pro rata for the time involved. Figure 6 shows the results of the user study for each aspect, ordered by the win rate (where a win indicates a preference for our method). EmbedTalk scores high on realism and image quality, whilst being marginally worse than generative methods for lip-sync, owing to their highly pronounced lip movements. We also note frequent ties in paired comparisons with other 3DGS methods, which is expected, given the similarity of approach.

## 4.3 Ablations and Experiment Variations

To validate the efficacy of our design choices, we conduct ablations and experimental variations under the self-driven setting. For ablations, we report performance in the absence of (a) positional encodings and (b) the local smoothness constraint $( \mathcal { L } _ { e m b \_ r e g } )$ . For experimental variations, we present metrics for different embedding sizes $( d i m ( z _ { g } ) = 1 6 , 6 4 )$ and by varying the Gaussian attributes deformed. Table 4 presents the results of these additional experiments. Across all ablations and variations, there are negligible differences and multiple ties in the rendering metrics (consequently, we chose not indicate the top three in the PSNR, SSIM and LPIPS columns). Our design choices are most beneficial in promoting audio-visual alignment and motion consistency, as evidenced through the LMD, Sync-C, and FVMD values. The local smoothness constraint and our choice to focus on deforming Gaussian position and visibility are especially useful for encouraging smooth motion and high lip-synchronisation. We find the deformation of opacity and the colour feature $( \varDelta \alpha , \varDelta f$ setting) to be the most promising alternative to our setup, albeit with additional predictive overhead.

<!-- image-->

(a) Video Realness  
<!-- image-->

(b) Lip Synchronisation  
<!-- image-->  
Fig. 6: Results of the user study showcasing how often EmbedTalk wins/loses against, or is tied with another method in a paired setup. Assessment criteria include (a) Video Realness, (b) Lip Synchronisation, and (c) Image Quality.

Table 4: Results for Ablations and Experimental Variations to validate the design choices for EmbedTalk. Due to negligible differences and multiple ties, we do not indicate the top three values in the PSNR, SSIM, and LPIPS columns.
<table><tr><td colspan="2">Experiment Configuration</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>LMDâ</td><td>Sync-Câ</td><td>FVMDâ</td></tr><tr><td rowspan="2">Ablations</td><td>w/o pos. enc.</td><td>35.216</td><td>0.962</td><td>0.020</td><td>2.584</td><td>5.572</td><td>156.995</td></tr><tr><td>w/o emb. reg.</td><td>35.107</td><td>0.958</td><td>0.022</td><td>2.513</td><td>5.903</td><td>229.727</td></tr><tr><td rowspan="2">zg dimension</td><td>16</td><td>35.037</td><td>0.959</td><td>0.021</td><td>2.634</td><td>5.085</td><td>231.203</td></tr><tr><td>64</td><td>34.664</td><td>0.961</td><td>0.220</td><td>2.728</td><td>5.759</td><td>158.467</td></tr><tr><td rowspan="4">Deformed Attributes (âÂµ,)</td><td>âÎ±, âf</td><td>34.960</td><td>0.961</td><td>0.022</td><td>2.533</td><td>6.384</td><td>146.852</td></tr><tr><td>âr, âs</td><td>34.507</td><td>0.960</td><td>0.022</td><td>3.097</td><td>2.977</td><td>260.672</td></tr><tr><td>âr, âs, âf</td><td>34.657</td><td>0.959</td><td>0.023</td><td>2.859</td><td>3.026</td><td>251.760</td></tr><tr><td>âr, âs, âÎ±, âf</td><td>34.940</td><td>0.960</td><td>0.022</td><td>2.704</td><td>5.453</td><td>146.638</td></tr><tr><td colspan="2">Ours  $( z _ { g } = 3 2 , 4 \mu , \varDelta \alpha )$ </td><td>35.186</td><td>0.961</td><td>0.021</td><td>2.444</td><td>6.520</td><td>147.384</td></tr></table>

## 5 Conclusion

We present EmbedTalk, a new method that shows how learnt embeddings can be leveraged to deform 3D Gaussians for talking head synthesis. Results across quantitive and qualitative setups demonstrate that our approach produces high quality talking heads with accurate and realistic lip-synchronisation. Furthermore, we show that replacing tri-planes significantly reduces memory usage while improving inference speed on mobile GPUs.

We note some limitations of our work. Although EmbedTalk improves on current baselines, its reconstructions are not perfect (bottom row of Figure 3). Future work may explore alternative intermediate representations beyond triplanes and embeddings. Our generated videos are also limited to neutral vocal tone and expressions due to the nature of our training data. While we anticipate that EmbedTalk would generalise to new videos with diverse emotions and expressions, this remains to be experimentally validated. Additionally, our method is restricted to facial animation and could be integrated with full-body motion modelling techniques to support fully interactive avatars.

We hope that our work will support research and development in creative industries. However, our method carries potential for misuse through generation of deepfakes for identity theft or misinformation. We advocate for the use of explicit labelling and watermarking techniques to help distinguish real videos from AI-generated videos. Furthermore, we will release our source code to aid the community in developing novel methods for detecting synthetic content.

## Acknowledgements

AS is supported by a PhD studentship that is funded by UK Research and Innovation (CDT Grant Reference: EP/S024336/1). This work was undertaken on the Aire HPC system at the University of Leeds, UK.

## References

1. Agarwal, M., Zhang, M., Sevilla-Lara, L., McDonagh, S.: Gaussianheadtalk: Wobble-free 3d talking heads with audio driven gaussian splatting. In: IEEE/CVF Winter Conference on Applications of Computer Vision 2026 (2026). https: //doi.org/10.48550/arXiv.2512.10939

2. Alghamdi, M.M., Wang, H., Bulpitt, A.J., Hogg, D.C.: Talking Head from Speech Audio using a Pre-trained Image Generator. In: Proceedings of the 30th ACM International Conference on Multimedia. Association for Computing Machinery (2022). https://doi.org/10.1145/3503161.3548101

3. Bae, J., Kim, S., Yun, Y., Lee, H., Bang, G., Uh, Y.: Per-gaussian embeddingbased deformation for deformable 3d gaussian splatting. In: European Conference on Computer Vision (2024)

4. BaltruÅ¡aitis, T., Robinson, P., Morency, L.P.: Openface: an open source facial behavior analysis toolkit. In: IEEE Winter Conference on Applications of Computer Vision (2016)

5. Bulat, A., Tzimiropoulos, G.: How far are we from solving the 2d & 3d face alignment problem? (and a dataset of 230,000 3d facial landmarks). In: International Conference on Computer Vision (2017)

6. Cao, A., Johnson, J.: Hexplane: A fast representation for dynamic scenes. CVPR (2023)

7. Chan, E.R., Lin, C.Z., Chan, M.A., Nagano, K., Pan, B., Mello, S.D., Gallo, O., Guibas, L., Tremblay, J., Khamis, S., Karras, T., Wetzstein, G.: Efficient geometryaware 3D generative adversarial networks. In: CVPR (2022)

8. Chen, L., Li, Z., Maddox, R.K., Duan, Z., Xu, C.: Lip movements generation at a glance. In: European Conference on Computer Vision (2018), https://api. semanticscholar.org/CorpusID:4435268

9. Chen, Z., Cao, J., Chen, Z., Li, Y., Ma, C.: Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions. Proceedings of the AAAI Conference on Artificial Intelligence (2025). https://doi.org/10.1609/aaai. v39i3.32241

10. Cho, K., Lee, J., Yoon, H., Hong, Y., Ko, J., Ahn, S., Kim, S.: Gaussiantalker: Real-time talking head synthesis with 3d gaussian splatting. In: Proceedings of the 32nd ACM International Conference on Multimedia (2024)

11. Chung, J.S., Zisserman, A.: Out of time: automated lip sync in the wild. In: Workshop on Multi-view Lip-reading, ACCV (2016)

12. Deng, K., Zheng, D., Xie, J., Wang, J., Xie, W., Shen, L., Song, S.: DEGSTalk: Decomposed Per-Embedding Gaussian Fields for Hair-Preserving Talking Face Synthesis. In: IEEE International Conference on Acoustics, Speech and Signal Processing (2025). https://doi.org/10.1109/ICASSP49660.2025.10890278

13. Ekman, P., Friesen, W.V.: Facial Action Coding System. Consulting Psychologists Press (1978)

14. ElevenLabs: Free Text To Speech Online with Lifelike AI Voices, https:// elevenlabs.io/text-to-speech

15. Guo, Y., Chen, K., Liang, S., Liu, Y., Bao, H., Zhang, J.: Ad-nerf: Audio driven neural radiance fields for talking head synthesis. In: IEEE/CVF International Conference on Computer Vision (2021)

16. Guo, Y., Deng, K., Song, S., Xie, J., Ma, W., Shen, L.: D3-talker: Dual-branch decoupled deformation fields for few-shot 3d talking head synthesis. In: Proceedings of the 28th European Conference on Artificial Intelligence (2025). https://doi. org/10.3233/FAIA251143

17. Hsu, W.N., Bolte, B., Tsai, Y.H.H., Lakhotia, K., Salakhutdinov, R., Mohamed, A.: Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing (2021)

18. Ji, X., Hu, X., Xu, Z., Zhu, J., Lin, C., He, Q., Zhang, J., Luo, D., Chen, Y., Lin, Q., et al.: Sonic: Shifting focus to global audio perception in portrait animation. In: Proceedings of the Computer Vision and Pattern Recognition Conference (2025)

19. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (2023), https: //repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

20. Ki, T., Min, D., Chae, G.: Float: Generative motion latent flow matching for audiodriven talking portrait. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2025)

21. Kingma, D., Ba, J.: Adam: A method for stochastic optimization. International Conference on Learning Representations (2014)

22. Li, H., Liu, K., Qiu, L., Zuo, Q., Zheng, K., Dong, Z., Han, X.: Hyplanehead: Rethinking tri-plane-like representations in full-head image synthesis. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https://openreview.net/forum?id=BH9niRIiy2

23. Li, J., Zhang, J., Bai, X., Zheng, J., Ning, X., Zhou, J., Gu, L.: Talkinggaussian: Structure-persistent 3d talking head synthesis via gaussian splatting. In: European Conference on Computer Vision (2024)

24. Li, J., Zhang, J., Bai, X., Zheng, J., Zhou, J., Gu, L.: Instag: Learning personalized 3d talking head from few-second video. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2025)

25. Li, J., Zhang, J., Bai, X., Zhou, J., Gu, L.: Efficient region-aware neural radiance fields for high-fidelity talking portrait synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023)

26. Liu, J., Qu, Y., Yan, Q., Zeng, X., Wang, L., Liao, R.: FrÃ©chet video motion distance: A metric for evaluating motion consistency in videos. In: First Workshop on Controllable Video Generation @ ICMLâ24 (2024), https://openreview.net/ forum?id=tTZ2eAhK9D

27. Liu, T., Chen, F., Fan, S., Du, C., Chen, Q., Chen, X., Yu, K.: Anitalker: Animate vivid and diverse talking faces through identity-decoupled facial motion encoding. In: Proceedings of the 32nd ACM International Conference on Multimedia. Association for Computing Machinery (2024). https://doi.org/10.1145/3664647. 3681198

28. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In: 2024 International Conference on 3D Vision (3DV) (2024). https://doi.org/10.1109/3DV62453.2024.00044

29. Meng, R., Zhang, X., Li, Y., Ma, C.: Echomimicv2: Towards striking, simplified, and semi-body human animation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5489â5498 (June 2025)

30. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV (2020)

31. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems (2019)

32. Paysan, P., Knothe, R., Amberg, B., Romdhani, S., Vetter, T.: A 3d face model for pose and illumination invariant face recognition. In: Proceedings of the 2009 Sixth IEEE International Conference on Advanced Video and Signal Based Surveillance (2009). https://doi.org/10.1109/AVSS.2009.58

33. Peng, Z., Hu, W., Ma, J., Zhu, X., Zhang, X., Zhao, H., Tian, H., He, J., Liu, H., Fan, Z.: Synctalk++: High-fidelity and efficient synchronized talking heads synthesis using gaussian splatting. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025). https://doi.org/10.1109/TPAMI.2025.3630057

34. Peng, Z., Hu, W., Shi, Y., Zhu, X., Zhang, X., Zhao, H., He, J., Liu, H., Fan, Z.: SyncTalk: The Devil is in the Synchronization for Talking Head Synthesis. In: CVPR (2024), https://openaccess.thecvf.com/content/CVPR2024/html/ Peng_SyncTalk_The_Devil_is_in_the_Synchronization_for_Talking_Head_ CVPR_2024_paper.html

35. Prajwal, K.R., Mukhopadhyay, R., Namboodiri, V.P., Jawahar, C.: A lip sync expert is all you need for speech to lip generation in the wild. In: Proceedings of the 28th ACM International Conference on Multimedia. Association for Computing Machinery (2020). https://doi.org/10.1145/3394171.3413532

36. Prolific: Prolific | Easily collect high-quality data from real people, https://www. prolific.com

37. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2016)

38. Snavely, N., Seitz, S.M., Szeliski, R.: Photo tourism: exploring photo collections in 3d. ACM Transactions on Graphics p. 835â846 (Jul 2006). https://doi.org/10. 1145/1141911.1141964

39. Tan, S., Ji, B., Bi, M., Pan, Y.: Edtalk: Efficient disentanglement for emotional talking head synthesis. In: European Conference on Computer Vision. pp. 398â416. Springer (2024)

40. Tang, J., Wang, K., Zhou, H., Chen, X., He, D., Hu, T., Liu, J., Liu, Z., Zeng, G., Wang, J.: Real-time neural radiance talking portrait synthesis via audio-spatial decomposition. Int. J. Comput. Vision p. 6362â6373 (Jun 2025). https://doi.org/ 10.1007/s11263-025-02481-9, https://doi.org/10.1007/s11263-025-02481-9

41. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.u., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems. vol. 30 (2017), https://proceedings.neurips.cc/paper_ files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

42. Wang, Y., Yang, P., Xu, Z., Sun, J., Zhang, Z., Chen, Y., Bao, H., Peng, S., Zhou, X.: Freetimegs: Free gaussian primitives at anytime anywhere for dynamic scene reconstruction. In: CVPR (2025), https://zju3dv.github.io/freetimegs

43. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20310â20320 (June 2024)

44. Xu, S., Chen, G., Guo, Y.X., Yang, J., Li, C., Zang, Z., Zhang, Y., Tong, X., Guo, B.: VASA-1: Lifelike audio-driven talking faces generated in real time. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024), https://openreview.net/forum?id=5zSCSE0k41

45. Yang, C., Yao, K., Yan, Y., Jiang, C., Zhao, W., Sun, J., Cheng, G., Zhang, Y., Dong, B., Huang, K.: Unlock pose diversity: Accurate and efficient implicit

keypoint-based spatiotemporal diffusion for audio-driven talking portrait. International Journal of Computer Vision (2026). https://doi.org/10.1007/s11263- 025-02695-x

46. Yang, Z., Yang, H., Pan, Z., Zhang, L.: Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. In: International Conference on Learning Representations (ICLR) (2024)

47. Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024). https: //doi.org/10.1109/CVPR52733.2024.01922

48. Ye, Z., Jiang, Z., Ren, Y., Liu, J., He, J., Zhao, Z.: Geneface: Generalized and high-fidelity audio-driven 3d talking face synthesis. In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/forum? id=YfwMIDhPccD

49. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 586â595 (2018). https://doi.org/10.1109/CVPR.2018.00068

50. Zhang, Z., Li, L., Ding, Y., Fan, C.: Flow-guided One-shot Talking Face Generation with a High-resolution Audio-visual Dataset. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021). https://doi.org/10. 1109/CVPR46437.2021.00366