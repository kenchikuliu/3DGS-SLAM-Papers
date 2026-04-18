# Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting

Tong Shi1[0000â0001â5913â4095], Melonie de Almeida1[0009â0007â4851â0949], Daniela Ivanova1[0000â0002â3710â7413], Nicolas Pugeault1[0000â0002â3455â6280], and Paul Henderson1[0000â0002â5198â7445]

School of Computing Science, University of Glasgow 2431206s@student.gla.ac.uk

Abstract. Talking Head Generation aims at synthesizing natural-looking talking videos from speech and a single portrait image. Previous 3D talking head generation methods have relied on domain-specific heuristics such as warping-based facial motion representation priors to animate talking motions, yet still produce inaccurate 3D avatar reconstructions, thus undermining the realism of generated animations. We introduce Splat-Portrait, a Gaussian-splatting-based method that addresses the challenges of 3D head reconstruction and lip motion synthesis. Our approach automatically learns to disentangle a single portrait image into a static 3D reconstruction represented as static Gaussian Splatting, and a predicted whole-image 2D background. It then generates natural lip motion conditioned on input audio, without any motion driven priors. Training is driven purely by 2D reconstruction and score-distillation losses, without 3D supervision nor landmarks. Experimental results demonstrate that Splat-Portrait exhibits superior performance on talking head generation and novel view synthesis, achieving better visual quality compared to previous works. Our project code and supplementary documents are public available at https://github.com/stonewalking/Splat-portrait.

Keywords: Gaussian SplattingÂ· Talking Head Generation

## 1 Introduction

Talking Head Generation (THG) aims to synthesize natural-looking talking videos from conditioning information such as driving speech [43, 9, 40] or driving videos [16, 40, 21]. The generation of talking heads has received increasing attention due to its importance in various applications, including digital humans [18], virtual video conferencing [39] and visual dubbing [28]. Recent 2D methods [43, 38, 20] have achieved significant improvements in video quality and achieve expressive animation results. However, such 2D methods struggle to generate views with large head pose variations, and are not guaranteed to output 3D-consistent renderings from different poses.

3D THG methods [40, 16, 21, 21] have attracted increasing attention in the past two years, since they simultaneously reconstruct accurate 3D geometry and generate expressive facial motions, allowing realistic and 3D-consistent portrait rendering from arbitrary user-controllable viewpoints. The majority of such methods focus on personal talking head generation [24, 41], where they overfit a single personâs head; they maintain realistic 3D geometry and preserve rich texture details. However, in this work, we consider the more challenging setting where we synthesise a 3D talking head given just a single 2D image, learning a model that generalizes across identities even without 3D supervision.

To represent 3D or 4D faces in THG, Neural Radiance Fields (NeRF) [22] or 3D Gaussian Splatting (3DGS) [13] are commonly used. NeRF-based methods often exhibit problems such as visual jitters, unsynchronized lip movements, and rendering artifacts; this is because the implicit definition of NeRF entangles static facial geometry with dynamic motion, complicating simultaneous control of lip motions and 3D geometry reconstruction.

Other works [32, 1, 4] have explored 3D Gaussian Splatting (3DGS) [13] for 3D avatar generation in the person-generic setting. Compared to NeRF, 3DGS not only improves inference speed and visual quality, but is also more controllable due to its explicit point-cloud-based representation; this makes it possible to animate facial movements more directly and intuitively. For example, [41] drives Gaussian point clouds for facial motion using parametric 3D facial models [15], but it is still challenging to generalize to a person-generic setting. In addition, significant efforts have been made to design and improve 4D animation conditioned on driving information [40, 1, 41]; the model reconstructs 3D geometry from a single portrait image and learns the corresponding facial motions. These motions are predicted from either an audio sequence or a video sequence driven by motion representation priors, e.g., PNCC, SECC and FLAME [45, 16, 15]. These methods relax the difficulty of model training by injecting domain priors, e.g. 3D distillation [2] and motion driven priors, which can lead to unnatural results with limited 3D texture details.

Overall, existing works either reconstruct accurate 3D geometry but require multi-view inputs; or they learn it from monocular videos but yield inaccurate geometry.

In this work, we introduce Splat-Portrait, a novel audio-driven THG method based on 3DGS (see Fig. 1). Our method operates in the single-view settingâit reconstructs the 3D shape of the head directly from one image, outputting pixelaligned Gaussian splats. To enable lip motion during speech, our model learns to directly animate these splats conditioned on an audio sequence.

Our approach is self-supervised from monocular videos only, and does not rely on 3D morphable models such as FLAME to represent facial shape and expression. We first train our model for static splat reconstruction on a large dataset without audio, then fine-tune on a smaller dataset of portrait videos to learn the correct splat dynamics. This strategy avoids 3D supervision, with the exception of easily-obtained approximate camera intrinsics and extrinsics. During the fine-tuning stage, to further improve the realism of extreme viewpoints that are rare in the training data, we adopt score distillation sampling (SDS) [26], to extract knowledge from a powerful 2D diffusion prior [11].

Existing works [17, 16] typically model only the head region, or model the head and torso regions as a whole, while disregarding the background. This results in a video of a âfloating headâ, rather than a realistic video of the talking head in context. To address this, our model also predict a static RGB background image, and alpha-blend the rasterized splats over this. Driven only by the unsupervised frame-prediction loss, our model automatically learns to reduce the opacity of splats in the background region, and to inpaint the background even behind the head, resulting in realistic disocclusions when the head rotates.

In summary, our main contributions are as follows:

â A novel model architecture that disentangles a single portrait image into an accurate 3D splat representation of the head over an inpainted 2D background.

Given audio sequences and corresponding time deltas, we show how to directly animate the 3D splats by predicting and adding dynamic offsets, without any complex motion representation such as a deformation model.

â A self-supervised training recipe that uses only monocular videos, without 3D supervision, and integrates knowledge from a strong 2D face prior, distilling its knowledge to improve reconstruction of extreme views.

Experimental results on the HDTF [44] and TH-1KH [36] datasets demonstrate that our approach yields higher video fidelity and quality compared with OTAvatar [21], HiDe-NeRF [16], Real3D-Portrait[40], and NeRFFaceSpeech [14] and GAGavatar[4].

## 2 Related Work

## 2.1 3D Head Reconstruction.

3D Gaussian Splatting (3DGS) [13] has emerged as a popular method for 3D head reconstruction due to its efficient rendering speed and superior reconstruction quality [41, 32, 35, 1]. Neural Radiance Fields (NeRF) [22] have been widely adopted for 3D talking head generation; NeRF represents scenes through volumetric radiance fields encoded by neural networks, enabling photorealistic renderings from novel viewpoints. NeRF-based methods have naturally extended into talking-head synthesis [24, 8]. Early NeRF-driven approaches for talkinghead reconstruction [21, 8] often require subject-specific training, limiting their scalability. Recent methods leverage 3DGS to address these limitations by significantly improving rendering speed and depth estimation [13, 41]. 3DGS represents scenes explicitly with discrete geometric primitives (3D Gaussians), enabling efficient optimization and real-time rendering. Notably, Rivero et al. [27] introduced a dynamic head reconstruction framework using 3DGS, and GaussianHead [35] further advanced these capabilities. By binding the Gaussians to an underlying geometric model, dynamic talking heads can be generated. However, these works for directly regressing 3D representations require prediction in a canonical space, which often fails to handle extreme head poses or significant appearance variations, such as non-photorealistic or animated scenarios. Current techniques still exhibit overfitting issues and rely heavily on domain priors during training, such as the parametric FLAME model [15]. Our method builds upon 3DGS to reconstruct dynamic talking heads directly from a single image.

## 2.2 Probabilistic 3D Reconstruction.

Single-view 3D head reconstruction [7] is an ambiguous problem due to the fact that training data usually have limited variation in poses, particularly in face monocular videos. Recently, diffusion models have been employed for conditional novel view synthesis [37] and also multi-view synthesis [32]. Since the results usually have ambiguous geometry, the output rendered results can exhibit noticeable artifacts, particularly a lack of texture details in unseen views. This can be mitigated by distilling prior knowledge from a 2D model [26]. Existing 3D reconstruction works found that distilling knowledge from 2D images could help to make the 3D representation much more controllable by reconstructing a geometry at every step of the denoising process [33]. Other works pre-train a robust reconstructor [19] and use a 3D prior [23] which can be used in an imageconditioned auto-decoding framework. However, their work is complex and computationally heavy to train. We also leverage a pretrained 2D generative prior when training for 3D reconstruction; this helps our method with extreme-view 3D head reconstruction, but avoids expensive iterative sampling.

## 2.3 Face Animation.

Initial efforts for talking head animation utilized 2D approaches, employing generative adversarial networks, image-to-image translation [10] or diffusion models [30], to generate facial animations. Most 2D talking head generation methods design a mapping relationship between face images and audio feature. These methods [43, 38] often underestimate detailed individual differences. Recently, 3D facial animation methods [16, 40] became popular, however they adopt PNCC SECC as driving features, leading to unnatural expressions and lip motion. Some warping-based methods [34, 40] employ 3DMMs, or face blend shapes, which support animation via disentangled representation of shape, expression and pose. However, these approaches can fall short of accurately reproducing a talking face due to limited amplitude, leading to shortcomings in identity preservation and pose controllability. Our approach is designed to directly edit the 3D representation to animate lip motion over time.

## 3 Methodology

The overall architecture of our method Splat-Portrait is illustrated in Fig. 1. Splat-Portrait consists of two main stages: (1) pre-training to reconstruct 3D static splats (Sec. 3.1); (2) fine-tuning with an audio-conditioned dynamic decoder (Sec. 3.2), while also using score distillation (Sec. 3.3) to refine appearance from extreme viewpoints.

<!-- image-->  
Fig. 1: Overview of Splat-Portrait. The identity image $I _ { i }$ is passed through a U-Net Static Generator(SG) to reconstruct static 3D Gaussian Splats, alphablended over a predicted 2D background. The dynamic decoder estimates splat offsets at timestep $T _ { n }$ using audio features $A _ { r }$ and time embedding $\varDelta T$ . The training procedure consists of two stages, stage(I): an initial pre-training phase, where the static components are trained on a large-scale dataset using a static reconstruction loss $\mathcal { L } _ { \mathrm { s t a t i c } } .$ , and stage(II): a fine-tuning phase on a smaller dataset incorporating an additional dynamic reconstruction loss $\mathcal { L } _ { \mathrm { d y n a m i c } }$ . And a score distillation loss $ { \mathcal { L } } _ { \mathrm { S D S } }$ on extreme viewpoints applied during both stages.

## 3.1 Static Splat Generation

3D Gaussian Splatting (3DGS) [13] uses anisotropic 3D Gaussians as geometric primitives to explicitly represent 3D scenes. For our 3D head reconstruction, we first pre-train a static generator (SG) that outputs pixel-aligned splats, as shown in Fig. 1. The design of SG is based on Splatter-Image [31]. However, unlike [31] we do not have access to wide-baseline multi-view images for training; instead we use more challenging monocular video data. We also predict an inpainted 2D RGB background as well the per-pixel 3D splat attributes, and alpha-blend the rasterized splats over this. During training, we randomly choose pairs of frames from a video, denoted source image $I _ { i }$ and future image $I _ { n }$ . Given $I _ { i } ,$ the network predicts a set of Gaussian Splatting parameters GS, at each pixel: opacity o, scale s, depth $d ,$ static offset $\varDelta _ { s }$ , rotation $r ,$ splat colour c (encoding per-pixel 3D Gaussian attributes), and the 2D background colour RGB. The view-space 3D position $p$ of the Gaussian at a pixel with ray direction r is then given by $p = \mathbf { r } d + \varDelta _ { s }$ â¢

During training, we feed the network with $I _ { i }$ at time step $T i$ . Additionally, we inject the approximate camera-to-world translation and focal length $\pi .$ . We do so by encoding each entry via a sinusoidal positional embedding of order 9, resulting in 60 dimensions in total. These are applied to the U-Net blocks via FiLM [25] conditioning. During our experiments, we found this helps with convergence of depth predictions.

Given the Gaussian attributes described above, we use the differentiable rasterizer R from [13] to render the splats at canonical space with static offset enabled to reconstruct images $I _ { i } ^ { * }$ and $I _ { n } ^ { * }$ at the camera poses of both $I _ { i }$ and $I _ { n }$ respectively. We compute a combined L2 and LPIPS reconstruction loss $\mathcal { L } _ { \mathrm { s t a t i c } } ^ { \mathrm { r e c } }$ between corresponding rendered and ground-truth images, i.e.

$$
\begin{array} { r } { \mathcal { L } \mathrm { s t a t i c } = | | I _ { i } - I _ { i } ^ { * } | | _ { 2 } + | | I _ { n } - I _ { n } ^ { * } | | _ { 2 } + \lambda _ { \mathrm { L P I P S } } \left[ \mathcal { L } _ { \mathrm { L P I P S } } ( I _ { i } , I _ { i } ^ { * } ) + \mathcal { L } _ { \mathrm { L P I P S } } ( I _ { n } , I _ { n } ^ { * } ) \right] } \end{array}\tag{1}
$$

Here the LPIPS term combines VGGface and VGG19 features, and the weight Î» is empirically set to 0.01. For each image, we render (and calculate the loss) twice, once with a random coloured background and once with our predicted 2D background alpha-blended behind the splats. We found that this helps to improve the colour and opacity for both background and foreground regions, without incorporating any mask supervision.

## 3.2 Audio-Conditioned Dynamic Splats

For predicting audio-conditioned dynamics representing lip movements, we designed a dynamic decoder with skip connections from the SG decoder. We only use this during the fine-tuning stage, after a good static reconstruction model has been learnt during pre-training. It predicts time-dependent offsets for every splat, conditioned on the audio signal and a time delta indicating what instant in the audio we want the splat offsets for. In our experiments we found that including this time delta improves convergence of the dynamic decoder.

For a given input frame $I _ { i } ,$ and future frame $I _ { n }$ plus its contemporaneous audio segment, we first extract audio features using Wav2Vec2-XLSR 53 [6].

Our model employs dedicated networks to fuse audio and temporal information effectively. Specifically, audio features are first encoded through an audio feature extraction module (AudioNet), which comprises several 1D convolutional layers followed by fully connected layers to yield compact audio embeddings. These embeddings are further refined through an attention-based network (AudioAttNet); this consists of a series of convolutional layers with decreasing channel sizes (from 16 to 1) interleaved with LeakyReLU activations. The output from these convolutional layers is then reshaped and passed through a linear layer followed by a softmax operation to calculate attention weights across the audio sequence. The weighted audio embeddings are summed to produce a refined audio representation capturing temporal dependencies across audio frames. For temporal embeddings, positional encoding or Fourier-based embeddings are utilized to encode timestep information; then audio and temporal embeddings are combined to form the conditioning feature. This combined embedding is injected into the dynamic decoder using FiLM conditioning, allowing the audio and time delta to control the generated motion.

Our dynamic decoder outputs a dynamic offset $\varDelta _ { d }$ for the splat at each pixel, conditioned on time T . Hence the splat position at time T is $p _ { T } = p + \varDelta _ { d }$ . To effectively train our model and maintain the static reconstruction ability learnt during pre-training, we adopt both $\mathcal { L } _ { \mathrm { s t a t i c } }$ loss and the SDS loss introduced in

Sec. 3.3. We render the source frame $I _ { i } ^ { * }$ with dynamic offsets fixed to zero (as in Sec. 3.1), but now render the future frame $I _ { n } ^ { * * }$ using the predicted offsets. Our dynamic reconstruction loss in the fine-tuning stage is:

$$
\mathrm { \mathcal { L } d y n a m i c } = | | I _ { i } - I _ { i } ^ { * } | | _ { 2 } + | | I _ { n } - I _ { n } ^ { * * } | | _ { 2 } + \lambda _ { \mathrm { L P I P S } } \left[ \mathcal { L } _ { \mathrm { L P I P S } } ( I _ { i } , I _ { i } ^ { * } ) + \mathcal { L } _ { \mathrm { L P I P S } } ( I _ { n } , I _ { n } ^ { * * } ) \right]\tag{2}
$$

## 3.3 Distillation from a 2D diffusion prior

In the fine-tuning stage, we also use score distillation [26] to extract knowledge from a 2D diffusion model [11] to improve the appearance of extreme poses. We first render our predicted reconstruction at a randomly sampled extreme pose, then crop and align the image following [12] to match the distribution learnt by the 2D diffusion model. Given this aligned image $x _ { \mathrm { c l e a n } }$ , we then add a random amount of noise then run the reverse diffusion process.

Specifically, we define a sequence of noise levels Ï as follows:

$$
\sigma _ { i } = \left[ \sigma _ { \operatorname* { m a x } } ^ { \frac { 1 } { \rho } } + \frac { i } { N - 1 } \left( \sigma _ { \operatorname* { m i n } } ^ { \frac { 1 } { \rho } } - \sigma _ { \operatorname* { m a x } } ^ { \frac { 1 } { \rho } } \right) \right] ^ { \rho } ,\tag{3}
$$

where $\sigma _ { \mathrm { m a x } }$ and $\sigma _ { \mathrm { m i n } }$ denote maximum and minimum noise levels, $\rho$ is a hyper parameter controlling the distribution of timesteps, and N is the total number of discretized steps. We choose the noise level from 60%â80% of the original range used in training the diffusion model, since we found this range effectively preserves the portraitâs overall appearance while significantly improving texture inpainting for extreme viewpoints.

The noised image at the initial timestep $t _ { 0 }$ is generated by adding Gaussian noise to the normalized input image, i.e. $x _ { \mathrm { n o i s e d } } = x _ { \mathrm { c l e a n } } + \sigma _ { 0 } \cdot \epsilon$ , where $\epsilon \sim$ $\mathcal { N } ( 0 , I )$ . For each subsequent timestep, we perform an Euler integration step to progressively denoise the image. Specifically, given the current timestep $t _ { \mathrm { c u r } }$ and next timestep $t _ { \mathrm { n e x t } }$ , the Euler step is computed as:

$$
x _ { \mathrm { n e x t } } = x _ { \mathrm { c u r } } + ( t _ { \mathrm { n e x t } } - t _ { \mathrm { c u r } } ) \cdot d _ { \mathrm { c u r } } , \quad d _ { \mathrm { c u r } } = \frac { x _ { \mathrm { c u r } } - \mathrm { n e t } ( x _ { \mathrm { c u r } } , t _ { \mathrm { c u r } } ) } { t _ { \mathrm { c u r } } } ,\tag{4}
$$

where net represents the pre-trained denoiser model.

This sampling procedure yields a denoised face image (the final $x _ { \mathrm { n e x t } } )$ that is similar to the original rendered one, but more realistic according to the diffusion prior. We define a loss $ { \mathcal { L } } _ { \mathrm { S D S } }$ as the L2 reconstruction loss between the rendered frame $x _ { \mathrm { c l e a n } }$ and the denoised frame, back-propagating only into the former. This guides our rendered frames to look more like similar realistic samples from the diffusion model. During training, we randomly sample extreme viewpoints following a bullet-effect trajectory, with pitch variations up to $\pm 1 2 . 5 ^ { \circ }$ and yaw variations up to $\pm 4 5 ^ { \circ }$ from the canonical view, and apply the SDS loss between $I _ { i }$ and $x _ { \mathrm { c l e a n } } .$ . Note that unlike [26] and common practice, we apply the SDS loss during model training, not during inference, meaning the latter remains very fast.

## 3.4 Overall Loss

Our total losses are defined as follows. For stage one (static pretraining):

$$
{ \mathcal { L } } _ { \mathrm { t o t a l \_ s t a t i c } } = { \mathcal { L } } _ { \mathrm { s t a t i c } } ( I _ { i } , I _ { n } ) + { \mathcal { L } } _ { \mathrm { S D S } } .\tag{5}
$$

For stage two (audio-conditioned fine-tuning):

$$
{ \mathcal { L } } _ { \mathrm { t o t a l \_ d y n a m i c } } = { \mathcal { L } } _ { \mathrm { d y n a m i c } } ( I _ { i } , I _ { n } ) + { \mathcal { L } } _ { \mathrm { S D S } } .\tag{6}
$$

In both cases, we use AdamW for optimization, with a learning rate of $2 . 5 \times 1 0 ^ { - 5 }$ and weight decay of $1 0 ^ { - 5 }$

## 4 Experiments

Datasets and Implementation Details. We evaluate our approach on two widely used datasets of monocular talking portrait videos â HDTF [44] and TalkingHead-1KH [36]. HDTF consists of over 400 samples of talking videos from over 350 subjects. For TalkingHead-1KH, we manually select 1100 identity videos following a similar distribution as HDTF, such that there is no occlusion over the torso and mouth, and with static background. Each identity video contains minimum 300 frames and maximum 10000 frames. We extract frames at 25Hz, and the audio sampling rate is 16kHz. We resize the image frames to $2 5 6 \times 2 5 6$ . Following the steps in [24], we follow [8] to use 3DMM optimization to extract approximate intrinsic and extrinsic camera parameters. We use the complete video clips (often with substantial camera motion) for training. For evaluation, we randomly sample 50 identity videos as test sets, and use the first 5s of each. We adopt the SongUNet [29] architecture for our static encoder and dynamic decoder.

Metrics. We measure the quality of synthetic images using structural similarity (SSIM), peak signal-to-noise ratio (PSNR), Learned Perceptual Image Patch Similarity LPIPS [42], and FrÃ©chet Inception Distance (FID); we use Cosine similarity (CSIM) for measuring identity preservation, and SyncNet [5] to measure lip synchronization scores (LipSync).

Baselines. We compare our approach to several existing 3D talking head generation works. OTAvatar [21] is a video-driven method that uses a pre-trained 3D GAN to obtain a 3D talking portrait video; HiDe-NeRF [16], a 3D talking face model that uses a motion prior and deformation field for face animation; Real3D-Portrait [40], a nerf-based method that uses images generated by EG3D to train a 3D model; NeRFFaceSpeech [14] one nerf-based audio driven method for synthesising talking head video, and the state-of-the-art GAGavatar [4]. Additionally in the audio-driven setting, we extend GAGavatar with ARtalker [3]. Note OTAvatar and HiDe-NeRF are video-driven methods, they are not directly driven by audio; for fair comparison, we use the same identity video as driving video for evaluation. We set the input image size as 256Ã256 to enable fair comparison, upsampling for methods that require this. We compare the baselines using their preferred masking and cropping settings.

OTAvatar  
NerfFaceSpeech  
Hide-Nerf  
GAGAvatar  
Real3D  
Ours  
<!-- image-->  
Fig. 2: Qualitative results. Top: We show source frames from five videos, future predicted frames from ours and baselines, and future depths from ours. Bottom: Additional examples of 3D reconstruction, for our method and Real3D-Portrait, displaying the input frame, and the reconstructed depth-map from each method.

## 4.1 Quantitative Evaluation

We compare with the baselines in same identity and cross-identity settings. During testing, the driving motion condition and head pose are obtained from a reference video. Under the same-identity setting, we use the first frame of the reference video as the source image; otherwise, the source image is of a different identity. For the cross-identity setting, for Real3D-Portrait, we only compare with its audio-driven setting. Quantitative results concerning the quality and fidelity for the same-identity setting are listed in Tab. 1. These show that our method outperforms other state-of-the-art approaches on almost all fidelity metrics. This is despite our method being trained without 3D supervision, using only a dataset of monocular videos. Splat-Portrait achieves the best overall video quality, as well as higher LipSync score, demonstrating that our 3D deformable model without any motion representation could sync well on lip motions. Moreover, our model achieves the highest performance on CSIM, meaning it has a strong ability to preserve subject identity in different views. We also compare the baselines with cross-identity evaluation, where the driving videos are obtained from a reference video, and we use another identity for target. Since there is no ground truth for this setting, we evaluate the results only on CSIM, FID and Lip sync. The results are given in Tab. 2. We see that our method performs best on FID and CSIM, which indicates our model still yields high video generation quality even in this more challenging setting.

<table><tr><td>Method</td><td>PSNR â SSIM â LPIPS â CSIM â FID â LipSync â</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OTAvatar</td><td>13.85</td><td>0.488</td><td>0.432</td><td>0.559</td><td>78.98</td><td>5.908</td></tr><tr><td>NeRFFaceSpeech</td><td>13.90</td><td>0.520</td><td>0.480</td><td>0.580</td><td>64.60</td><td>4.880</td></tr><tr><td>HiDe-NeRF</td><td>21.44</td><td>0.685</td><td>0.221</td><td>0.716</td><td>28.63</td><td>5.552</td></tr><tr><td>Real3D-Portrait</td><td>22.40</td><td>0.758</td><td>0.191</td><td>0.761</td><td>35.69</td><td>6.681</td></tr><tr><td>GAGAvatar + ARtalker</td><td>23.08</td><td>0.786</td><td>0.182</td><td>0.753</td><td>37.89</td><td>6.580</td></tr><tr><td>Ours</td><td>23.87</td><td>0.814</td><td>0.128</td><td>0.811</td><td>25.58</td><td>6.328</td></tr></table>

Table 1: Quantitative evaluation of our method and baselines in the same identity setting.

<table><tr><td>Method</td><td>CSIM â FID â LipSync â</td><td></td></tr><tr><td>NeRFFaceSpeech</td><td>0.450 50.80</td><td>4.423</td></tr><tr><td>OTAvatar</td><td>0.521</td><td>79.32 5.032</td></tr><tr><td>HiDe-NeRF</td><td>0.628</td><td>31.23 5.652</td></tr><tr><td>Real3D-Portrait</td><td>0.691</td><td>40.82 6.521</td></tr><tr><td>GAGAvatar + ARtalker</td><td>0.687</td><td>35.82 6.503</td></tr><tr><td>Ours</td><td>0.726 28.62</td><td>6.218</td></tr></table>

Table 2: Quantitative evaluation of our method and baselines in the cross-identity setting.

## 4.2 Qualitative evaluation

In this section we provide visual comparisons of all tested methods (see Figure 2). We find that our method preserves face texture details, such as hair and wrinkles well, yielding high-quality novel views. In particular, our method preserves details such as earrings which move during the video. Since we do not require the head to be pre-segmented, our method handles fine details at the silhouette edges well, and effectively blends the rendered portrait over the estimated background. Fig 2 also compares depth-maps rendered by our model with those from Real3D-Portrait, to better visualise the quality of the 3D shape. Compared with Real3D-Portrait, it is clear that our method preserves much more detailed geometry information.

<table><tr><td>Method</td><td>PSNR â SSIM â LPIPS â</td></tr><tr><td> $\mathrm { w / o }$  time delta</td><td>22.68 0.768 0.146</td></tr><tr><td> $\mathrm { w / o }$  pre-training</td><td>23.30 0.758 0.149</td></tr><tr><td> $\mathrm { w / o }$  SDS</td><td>23.58 0.788 0.147</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  static offset</td><td>23.30 0.791 0.145</td></tr><tr><td>only future 12 loss 23.41</td><td>0.772 0.138</td></tr><tr><td>Full (SP)</td><td>23.87 0.814 0.128</td></tr></table>

Table 3: Ablation study showing the benefit of different components of our model.

<!-- image-->  
Fig. 3: Ablation study, with extreme head yaw angles (top row at -35Â°, the middle row at $0 ^ { \circ }$ , and the bottom row at +35Â°).

## 4.3 Ablation Study

We test four ablations of our model: (1) $\mathrm { w / o }$ time delta, which does not inject the time embedding (see Sec. 3.2); (2) w/o pre-training, which does not pretrain the static generator (see Sec. 3.1); (3) $\mathrm { w / o }$ SDS, which omits the score distillation loss during the fine-tuning stage (see Sec. 3.3); (4) without static offsets during fine-tuning stage; (5) without the initial-frame reconstruction loss, only the future reconstruction loss. We show the results in Fig. 3. Without the pre-training process, we see the 3D geometry accuracy drops significantly, the reconstructed 3D head exhibits flattened geometry, with reduced 3D structure. Training with only one frame for supervision instead of two randomly selected frames, it is hard to reconstruct depths and static offsets (and thus the static shape of the face) well, as some structural information is instead represented in the dynamic offsets. As shown in Fig. 3, when enabling static splat offsets, the visualized 3D representation shows a smooth, realistically curved geometry. Lastly our SDS loss greatly enhances the realism of extreme poses.

## 5 Conclusion

We proposed Splat-Portrait for Talking Head Generation. Our method is trained on monocular videos without 3D supervision, yet can synthesize accurate 3D geometry and plausible lip movements directly from a single portrait image, yielding state-of-the-art results. By effectively disentangling static and dynamic attributes and using a score-distillation loss, Splat-Portrait significantly enhances realism, particularly from extreme viewpoints rarely encountered during training. Additionally, the simplicity and efficiency of our model structure allow it to animate 3D splats effectively without complex deformation models, making it lightweight and practical for real-world applications.

## References

1. Aneja, S., Sevastopolsky, A., Kirschstein, T., Thies, J., Dai, A., NieÃner, M.: Gaussianspeech: Audio-driven gaussian avatars. arXiv preprint arXiv:2411.18675 (2024)

2. Chan, E.R., Lin, C.Z., Chan, M.A., Nagano, K., Pan, B., De Mello, S., Gallo, O., Guibas, L.J., Tremblay, J., Khamis, S., et al.: Efficient geometry-aware 3d generative adversarial networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16123â16133 (2022)

3. Chu, X., Goswami, N., Cui, Z., Wang, H., Harada, T.: Artalk: Speech-driven 3d head animation via autoregressive model (2025), https://arxiv.org/abs/2502.20323

4. Chu, X., Harada, T.: Generalizable and animatable gaussian head avatar. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024), https://openreview.net/forum?id=gVM2AZ5xA6

5. Chung, J.S., Zisserman, A.: Out of time: automated lip sync in the wild. In: Workshop on Multi-view Lip-reading, ACCV (2016)

6. Conneau, A., Baevski, A., Collobert, R., Mohamed, A., Auli, M.: Unsupervised cross-lingual representation learning for speech recognition. arXiv preprint arXiv:2006.13979 (2020)

7. Dhamo, H., Nie, Y., Moreau, A., Song, J., Shaw, R., Zhou, Y., PÃ©rez-Pellitero, E.: Headgas: Real-time animatable head avatars via 3d gaussian splatting. In: European Conference on Computer Vision. pp. 459â476. Springer (2024)

8. Guo, Y., Chen, K., Liang, S., Liu, Y.J., Bao, H., Zhang, J.: Ad-nerf: Audio driven neural radiance fields for talking head synthesis. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5784â5794 (2021)

9. He, T., Guo, J., Yu, R., Wang, Y., Zhu, J., An, K., Li, L., Tan, X., Wang, C., Hu, H., et al.: Gaia: Zero-shot talking avatar generation. arXiv preprint arXiv:2311.15230 (2023)

10. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1125â1134 (2017)

11. Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of diffusionbased generative models. Advances in neural information processing systems 35, 26565â26577 (2022)

12. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4401â4410 (2019)

13. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139â1 (2023)

14. Kim, G., Seo, K., Cha, S., Noh, J.: Nerffacespeech: One-shot audio-driven 3d talking head synthesis via generative prior. arXiv preprint arXiv:2405.05749 (2024)

15. Li, T., Bolkart, T., Black, M.J., Li, H., Romero, J.: Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph. 36(6), 194â1 (2017)

16. Li, W., Zhang, L., Wang, D., Zhao, B., Wang, Z., Chen, M., Zhang, B., Wang, Z., Bo, L., Li, X.: One-shot high-fidelity talking-head synthesis with deformable neural radiance field. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17969â17978 (2023)

17. Li, X., De Mello, S., Liu, S., Nagano, K., Iqbal, U., Kautz, J.: Generalizable oneshot 3d neural head avatar. Advances in Neural Information Processing Systems 36 (2024)

18. Liu, C.: An analysis of the current and future state of 3d facial animation techniques and systems (2009)

19. Liu, M., Xu, C., Jin, H., Chen, L., Varma T, M., Xu, Z., Su, H.: One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems 36, 22226â22246 (2023)

20. Liu, T., Ma, Z., Chen, Q., Chen, F., Fan, S., Chen, X., Yu, K.: Vqtalker: Towards multilingual talking avatars through facial motion tokenization. arXiv preprint arXiv:2412.09892 (2024)

21. Ma, Z., Zhu, X., Qi, G.J., Lei, Z., Zhang, L.: Otavatar: One-shot talking face avatar with controllable tri-plane rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16901â16910 (2023)

22. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99â106 (2021)

23. MÃ¼ller, N., Siddiqui, Y., Porzi, L., Bulo, S.R., Kontschieder, P., NieÃner, M.: Diffrf: Rendering-guided 3d radiance field diffusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4328â4338 (2023)

24. Peng, Z., Hu, W., Shi, Y., Zhu, X., Zhang, X., Zhao, H., He, J., Liu, H., Fan, Z.: Synctalk: The devil is in the synchronization for talking head synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 666â676 (2024)

25. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: Film: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)

26. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022)

27. Rivero, A., Athar, S., Shu, Z., Samaras, D.: Rig3dgs: Creating controllable portraits from casual monocular videos. arXiv preprint arXiv:2402.03723 (2024)

28. Saunders, J., Namboodiri, V.: Dubbing for everyone: Data-efficient visual dubbing using neural rendering priors. arXiv preprint arXiv:2401.06126 (2024)

29. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)

30. StypuÅkowski, M., Vougioukas, K., He, S., ZiÄba, M., Petridis, S., Pantic, M.: Diffused heads: Diffusion models beat gans on talking-face generation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 5091â5100 (2024)

31. Szymanowicz, S., Rupprecht, C., Vedaldi, A.: Splatter image: Ultra-fast singleview 3d reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10208â10217 (2024)

32. Taubner, F., Zhang, R., Tuli, M., Lindell, D.B.: Cap4d: Creating animatable 4d portrait avatars with morphable multi-view diffusion models. arXiv preprint arXiv:2412.12093 (2024)

33. Tewari, A., Yin, T., Cazenavette, G., Rezchikov, S., Tenenbaum, J., Durand, F., Freeman, B., Sitzmann, V.: Diffusion with forward models: Solving stochastic inverse problems without direct supervision. Advances in Neural Information Processing Systems 36, 12349â12362 (2023)

34. Thies, J., Zollhofer, M., Stamminger, M., Theobalt, C., NieÃner, M.: Face2face: Real-time face capture and reenactment of rgb videos. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2387â2395 (2016)

35. Wang, J., Xie, J.C., Li, X., Xu, F., Pun, C.M., Gao, H.: Gaussianhead: Impressive head avatars with learnable gaussian diffusion. arXiv preprint arXiv:2312.01632 (2023)

36. Wang, T.C., Mallya, A., Liu, M.Y.: One-shot free-view neural talking-head synthesis for video conferencing. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10039â10049 (2021)

37. Watson, D., Chan, W., Martin-Brualla, R., Ho, J., Tagliasacchi, A., Norouzi, M.: Novel view synthesis with diffusion models. arXiv preprint arXiv:2210.04628 (2022)

38. Xu, M., Li, H., Su, Q., Shang, H., Zhang, L., Liu, C., Wang, J., Yao, Y., Zhu, S.: Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801 (2024)

39. Ye, T., Zhang, Y., Jiang, M., Chen, L., Liu, Y., Chen, S., Chen, E.: Perceiving and modeling density for image dehazing. In: European conference on computer vision. pp. 130â145. Springer (2022)

40. Ye, Z., Zhong, T., Ren, Y., Yang, J., Li, W., Huang, J., Jiang, Z., He, J., Huang, R., Liu, J., et al.: Real3d-portrait: One-shot realistic 3d talking portrait synthesis. arXiv preprint arXiv:2401.08503 (2024)

41. Yu, H., Qu, Z., Yu, Q., Chen, J., Jiang, Z., Chen, Z., Zhang, S., Xu, J., Wu, F., Lv, C., et al.: Gaussiantalker: Speaker-specific talking head synthesis via 3d gaussian splatting. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 3548â3557 (2024)

42. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586â595 (2018)

43. Zhang, W., Cun, X., Wang, X., Zhang, Y., Shen, X., Guo, Y., Shan, Y., Wang, F.: Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8652â8661 (2023)

44. Zhang, Z., Li, L., Ding, Y., Fan, C.: Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3661â3670 (2021)

45. Zhu, X., Lei, Z., Liu, X., Shi, H., Li, S.Z.: Face alignment across large poses: A 3d solution. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 146â155 (2016)