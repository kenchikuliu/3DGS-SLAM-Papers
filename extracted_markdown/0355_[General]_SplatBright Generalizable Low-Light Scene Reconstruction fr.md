# SplatBright: Generalizable Low-Light Scene Reconstruction from Sparse Views via Physically-Guided Gaussian Enhancement

Yue Wen1, Liang Song2, Hesheng Wang1\* 1Shanghai Jiao Tong University, 2China DXR Technology CO.,Ltd, \*Corresponding Author

<!-- image-->  
Figure 1. From sparse low-light views, our method reconstructs a dark 3D Gaussian field and enhances it into a consistent normal-light field, achieving consistent illumination and geometry in novel view synthesis with depth and normal maps without per-scene training.

## Abstract

Low-light 3D reconstruction from sparse views remains challenging due to exposure imbalance and degraded color fidelity. While existing methods struggle with view inconsistency and require per-scene training, we propose Splat-Bright, which is, to our knowledge, the first generalizable 3D Gaussian framework for joint low-light enhancement and reconstruction from sparse sRGB inputs. Our key idea is to integrate physically guided illumination modeling with geometryâappearance decoupling for consistent low-light reconstruction. Specifically, we adopt a dual-branch predictor that provides stable geometric initialization of 3D Gaussian parameters. On the appearance side, illumination consistency leverages frequency priors to enable controllable and cross-view coherent lighting, while an appearance refinement module further separates illumination, material, and view-dependent cues to recover fine texture. To tackle the lack of large-scale geometrically consistent paired data, we synthesize dark views via a physics-based camera model for training. Extensive experiments on public and self-collected datasets demonstrate that SplatBright achieves superior novel view synthesis, cross-view consistency, and better generalization to unseen low-light scenes compared with both 2D and 3D methods.

## 1. Introduction

Low-light scenes remain a key challenge for 3D reconstruction. Reduced illumination lowers contrast, distorts color, and amplifies noise, hindering recovery of geometry and appearance. Such degradation affects localization, mapping, and rendering in robotics, autonomous driving, and immersive applications, calling for a framework that directly restores accurate structure and illumination under low light.

Conventional low-light enhancement methods improve visibility in 2D images. Classical approaches such as histogram equalization and Retinex decomposition [15, 28, 40] enhance brightness with handcrafted priors but often amplify noise and color shifts. Deep models [7, 9, 21, 65] restore exposure and color using paired or adversarial training, producing visually pleasing but view-inconsistent results, since they operate per frame without enforcing geometric or photometric consistency across views.

Neural scene representations unify geometry and appearance modeling. NeRF [37] and its variants [10, 16, 36, 38, 48, 51] achieve photorealistic rendering under complex lighting and address low-light scenes. However, these methods require dense inputs and expensive optimization. Explicit representations such as 3D Gaussian Splatting (3DGS) [26] enable efficient differentiable rendering via Gaussian primitives. Recent extensions [22, 27, 59] improve fidelity and robustness but still rely on dense views, scene-specific tuning, or RAW data. This motivates a 3D

Gaussian framework unifying illumination and sparse-view reconstruction for consistent low-light recovery from sRGB inputs without per-scene training.

Building such a framework remains challenging. Firstly, real datasets rarely offer geometry-aligned multi-view images with varying exposures, hindering the learning of illumination attenuation and cross-view consistency. Secondly, sparse views weaken geometric cues and make appearance estimation illumination-sensitive, while uniform color adjustments in 3D space amplify view-dependent differences. Finally, noise and unstable brightness destabilize optimization and degrade reconstruction quality.

To overcome these challenges, we propose SplatBright, the first 3D Gaussian framework that jointly enhances illumination and reconstructs geometry from sparse-view images, generalizing to unseen low-light scenes without perscene optimization. Firstly, a camera-inspired darkening model synthesizes controllable darkânormal pairs for training, complemented by a self-collected real world multiexposure dataset for evaluation. Secondly, a dual-branch Gaussian predictor then decouples geometry and appearance for stable modeling, while two frequency-guided modules enable hierarchical appearance enhancement: the Illumination Consistency Module (ICM) performs controllable global illumination adjustment and exposure alignment via frequency-guided cross attention (FGCA) and style modulation; and the Appearance Refinement Module (ARM) employs windowed 3D cross-attention across illumination, material, and view cues to restore local texture and reflection details. Finally, a progressive training strategy sequentially optimizes geometry, global illumination, and local appearance, achieving physically consistent, detail-preserving reconstruction under low-light and sparse-view conditions. At inference time, SplatBright enables controllable relighting by adjusting the illumination style in unseen scenes. Our main contributions are summarized as follows:

â¢ We present SplatBright, which is, to our knowledge, the first generalizable 3D Gaussian framework for low-light reconstruction from sparse views, trained on physically inspired darkânormal pairs. Through geometryâappearance decoupling and progressive optimization, it enhances illumination and structure from sRGB inputs.

â¢ We design a two-stage appearance model. The illumination consistency module offers controllable lighting and cross-view consistency via wavelet-based style modulation, while the appearance refinement module refines each Gaussian appearance using illumination, material, and view cues.

â¢ Extensive experiments on synthetic and real datasets, including public and self-collected datasets, show that SplatBright outperforms previous 2D and 3D methods with superior reconstruction quality and generalization.

## 2. Related Work

## 2.1. 2D Low-light Image Enhancement

Low-light image enhancement has evolved from traditional algorithms to data-driven deep models. Classical approaches based on histogram equalization, Retinex decomposition, and illumination correction [13, 15, 18, 23, 28, 40] are efficient but often amplify noise and distort color. Deep networks [6, 7, 9, 30, 33, 61, 65] learn paired mappings for exposure and color recovery, achieving high fidelity yet requiring aligned data. To relax data dependence, unsupervised methods [14, 19, 21, 32, 35, 54, 68] leverage selfregularization, adversarial learning, or diffusion priors for label-free enhancement. Despite progress, most 2D models still enhance frames independently, lacking the geometric and photometric consistency for multi-view reconstruction.

## 2.2. 3D-aware Low-light Reconstruction

Neural scene representations enable unified modeling for multi-view rendering. NeRF [37] and its variants [16, 38, 41, 48, 51] achieve photorealistic rendering under diverse lighting and materials. NeRF-W [36] handles in-thewild illumination with per-image embeddings, while Aleth-NeRF [10] models light attenuation via a concealing field. However, volumetric NeRFs require costly optimization. To address this, 3DGS [26] use explicit Gaussian primitives for faster optimization and real-time rendering. Recent works [1, 22, 27, 44, 45, 50, 62] further enhance fidelity and robustness under complex lighting. DarkGS [64] integrates neural illumination fields with cameraâlight calibration, Luminance-GS [11] aligns per-view exposures via tone mapping, and Gaussian-DK [59] embeds camera parameters to decouple radiance from sensor response. Despite these progress, most methods still require dense views, per-scene optimization, or RAW inputs, hindering generalization to sparse-view reconstruction from sRGB data.

## 2.3. Feed-forward NeRF and 3DGS Models

Feed-forward paradigms remove per-scene optimization by directly predicting 3D representations from sparse views. For implicit fields, pixelNeRF [60] generalizes to few views with pixel-aligned features, while [4, 12, 42, 52] improve consistency via cost volumes but are costly. Explicit 3D Gaussians enable efficient feed-forward reconstruction. PixelSplat [3] employs epipolar transformers for probabilistic Gaussian sampling, and [5, 46, 47, 56, 66] extend to single-view or stereo settings. MVSplat [8] builds a differentiable multi-view cost volume to infer Gaussian attributes. While recent methods [2, 49] advance sparseview reconstruction, they ignore low-light degradation. Our method jointly models scene geometry and illumination within a physically grounded Gaussian framework to achieve low-light reconstruction from sparse views.

<!-- image-->

Figure 2. Overview of SplatBright. The pipeline includes data preprocessing, multi-view feature extraction, and geometryâappearance dual-head Gaussian initialization with normal-guided geometry optimization. With the FGCA module, ICM performs controllable lighting adjustment, while ARM models illumination, material, and view branches to enhance details for physically consistent relighting.  
<!-- image-->  
Figure 3. Comparison of per RGB channel intensity statistics.

## 3. Method

As illustrated in Fig. 1, our goal is to reconstruct a physically consistent 3D Gaussian field from sparse low-light inputs and generate enhanced renderings in novel views. Given two low-light input images $\mathcal { T } = \overline { { \{ I _ { c _ { 1 } } ^ { \mathrm { d a r k } } , I _ { c _ { 2 } } ^ { \mathrm { d a r k } } \} } }$ with corresponding camera matrices $\mathcal { P } = \{ P _ { c _ { 1 } } , \bar { P } _ { c _ { 2 } } ^ { \mathrm { ~ ~ } } \}$ , the framework predicts a unified set of 3D Gaussian primitives $\mathcal { G } =$ $\{ g _ { j } = ( \mu _ { j } , \Sigma _ { j } , \alpha _ { j } , \mathbf { s h } _ { j } ) \} _ { j = 1 } ^ { M }$ , where $\mu _ { j } , \Sigma _ { j } , \alpha _ { j }$ , and $s h _ { j }$ denote the position, covariance, opacity, and spherical harmonics, respectively. Here, $M \ = \ H \times W \times K$ represents the total number of pixel-aligned Gaussians determined by the input image resolution and the number of source views $K$ . The reconstructed low-light field $\mathcal { G } _ { \mathrm { d a r k } }$ is subsequently enhanced into a normal-light field $\mathcal { G } _ { \mathrm { b r i g h t } } .$ , enabling rendering of photometrically consistent novel views $\hat { \mathcal { Z } } ^ { \mathrm { b r i g h t } } = \{ \hat { I } _ { t _ { i } } \} _ { i = 1 } ^ { \check { N } }$ , under known target camera poses $\mathcal { P } _ { t } =$ $\{ P _ { t _ { i } } \} _ { i = 1 } ^ { N }$ , where N denotes the number of target views. The pipeline is illustrated in Fig. 2.

## 3.1. Data preparation and Gaussian initialization

Dark data generation. Collecting multi-view low-light datasets with consistent geometry is highly challenging in practice. To overcome this, we synthesize dark-normal pairs $\left( I _ { \mathrm { d a r k } } , I _ { \mathrm { n o r m a l } } \right)$ on the RealEstate10K (RE10K) [67] dataset, which provides geometry-aligned normal light images, by applying a physically inspired degradation process that mimics how cameras perceive dim scenes:

$$
I _ { \mathrm { d a r k } } = { \mathcal { T } } _ { \mathrm { d a r k } } ( I _ { \mathrm { n o r m a l } } ; d ) ,\tag{1}
$$

where $\mathcal { T } _ { \mathrm { d a r k } }$ applies exposure drop, ISP-tone compression, and chroma suppression, controlled by d. However, darkening tends to wash out bright sky regions into dull gray. To prevent unrealistic artifacts, we use a soft sky mask $M _ { \mathrm { s k y } } { \mathrm { . } }$

$$
I _ { \mathrm { d a r k } }  ( 1 - M _ { \mathrm { s k y } } ) I _ { \mathrm { d a r k } } + M _ { \mathrm { s k y } } I _ { \mathrm { n o r m a l } } .\tag{2}
$$

Our synthetic dark images match real low-light RGB statistics in Fig. 3, confirming the realism of the degradation model. By varying darkness levels $( d _ { \mathrm { l o w } } , d _ { \mathrm { h i g h } } )$ , we create controllable supervision pairs, allowing the model to learn robust illumination adaptation.

Multi-view depth estimation. We first recover geometry via multi-view disparity estimation, convert disparity to depth, and unproject into 3D space to initialize Gaussian centers $\mu _ { j } ^ { d }$ and coarse opacity $\alpha _ { j } ^ { d } .$ . A transformerâU-Net backbone extracts geometric features. To stabilize depth reasoning under low-light conditions, the low-level encoder filters are kept fixed while the refinement layers remain learnable, producing latent features $F = \phi _ { \mathrm { g s } } ( f _ { \mathrm { r e f i n e } } )$ , where $f _ { \mathrm { r e f i n e } }$ denote the U-Net feature map from the depth branch. Considering the distribution discrepancy between geometry and appearance, we design a decoupled dual-head Gaussian predictor for stable low-light modeling:

<!-- image-->

<!-- image-->

Figure 4. Normal supervision $\mathcal { L } _ { \mathrm { { n o r m a l } } }$ in the geometry stage improves texture sharpness and depth estimation.6558c5f10d45a929  
<!-- image-->  
Figure 5. Inference results using different brightness values sbright ranging from â1.0 to 1.5..

$$
\begin{array} { r } { \mathbf { g } ^ { \mathrm { g e o } } = \psi _ { \mathrm { g e o } } ( F ) , \quad \mathbf { g } ^ { \mathrm { a p p } } = \psi _ { \mathrm { a p p } } ( F ) , } \end{array}\tag{3}
$$

which are concatenated as $\mathbf { g } ^ { d } \ = \ \mathrm { c o n c a t } ( \mathbf { g } ^ { \mathrm { g e o } } , \mathbf { g } ^ { \mathrm { a p p } } ) \in$ $\mathbb { R } ^ { C \times H \times W }$ , where C denotes the total number of geometric and appearance parameters. The geometry branch refines gaussian centers, scales, and opacities to obtain anisotropic covariances $\Sigma _ { j } ^ { d }$ , while the appearance branch predicts spherical harmonics(SH) $\mathbf { s h } _ { j } ^ { d }$ to encode intrinsic color and view-dependent illumination. The resulting tensor $\mathbf { g } ^ { d }$ is finally decoded into a set of gaussian primitives:

$$
\mathcal { G } _ { \mathrm { d a r k } } = \{ ( \mu _ { j } ^ { d } , \alpha _ { j } ^ { d } , \Sigma _ { j } ^ { d } , \mathbf { s } \mathbf { h } _ { j } ^ { d } ) \} _ { j = 1 } ^ { H \times W \times K } .\tag{4}
$$

## 3.2. Illumination Consistency Module (ICM)

After initializing the dark Gaussian field, direct brightness scaling often amplifies noise and causes grayish bias. To ensure global consistency, we design an Illumination Consistency Module (ICM) that integrates low-frequency information with style-modulated illuminationâcolor disentanglement, enabling global gaussian parameters adjustment and scene-level controllable brightness prediction.

Frequency-guided cross attention (FGCA). To obtain a cross-view consistent global signal, we apply a 2D discrete wavelet transform (DWT) to the refined 2D feature $f _ { \mathrm { r e f i n e } } ~ \in ~ \mathbb { R } ^ { C \times H \times W }$ , decomposing it into four frequency subbands [LL, HL, LH, $H H ] \ = \ f _ { \mathrm { D W T } } ( f _ { \mathrm { r e f i n e } } )$ , where the component $L L \in \mathbb { R } ^ { C \times \frac { H } { 2 } \times \overline { { 2 } } }$ captures low-frequency global smooth information and is less sensitive to noise. We downsample $f _ { \mathrm { r e f i n e } }$ to obtain $\tilde { f } _ { \mathrm { r e f i n e } }$ and perform crossattention to extract a robust global descriptor:

$$
f _ { \mathrm { l o w } } = \mathrm { s o f t m a x } \left( \frac { \phi _ { q } ( \tilde { f } _ { \mathrm { r e f i n e } } ) \phi _ { k } ( L L ) ^ { \top } } { \sqrt { C ^ { \prime } } } \right) \phi _ { v } ( L L ) ,\tag{5}
$$

where $\phi _ { q } , \phi _ { k } , \phi _ { v }$ denote the learnable projections for the query, key, and value, and $C ^ { \prime }$ represents the channel dimension used to normalize the attention logits. The attended

feature is upsampled and added back to the original feature in a residual manner, forming feature $f _ { \mathrm { L I } }$ constrained by low-frequency priors.

Global appearance adjustment. A lightweight dark style predictor encodes the low-light and normal inputs I:

$$
\begin{array} { r } { \left[ s _ { \mathrm { b r i g h t } } , s _ { \mathrm { l a t e n t } } \right] = f _ { \mathrm { s t y l e } } ( I ) , } \end{array}\tag{6}
$$

where $s _ { \mathrm { b r i g h t } }$ represents an explicit brightness difference factor and $s _ { \mathrm { l a t e n t } }$ captures other degradations such as contrast and color shifts. These interpretable style cues enable explicit illumination enhancement across views.

Guided by the frequency-aware feature $f _ { \mathrm { L L } }$ and the style code $s ~ = ~ [ s _ { \mathrm { b r i g h t } } , s _ { \mathrm { l a t e n t } } ]$ , a dual-branch global enhancer modulates the SH coefficients and opacity:

$$
\begin{array} { r } { \Delta \mathbf { s h } = f _ { \mathrm { b } } ( s _ { \mathrm { b r i g h t } } , f _ { \mathrm { r e f n e } } ) + f _ { \mathrm { l } } ( s _ { \mathrm { l a t e n t } } , f _ { \mathrm { r e f n e } } ) , } \\ { \Delta \rho = g _ { \mathrm { b } } ( s _ { \mathrm { b r i g h t } } , f _ { \mathrm { r e f n e } } ) + g _ { \mathrm { l } } ( s _ { \mathrm { l a t e n t } } , f _ { \mathrm { r e f n e } } ) , } \end{array}\tag{7}
$$

where $f _ { \mathrm { b } } , f _ { \mathrm { l } } , g _ { \mathrm { b } } , g _ { \mathrm { l } }$ denote the brightness and latent modulation branches. The brightness term controls global exposure ( Fig. 5), while the latent term refines tone and contrast.

SH coefficients are updated in a residual form, and opacity follows an exponential decay of radiance with optical density, analogous to light attenuation in hazy scenes:

$$
\tilde { \rho } = \rho \exp ( - \gamma _ { \rho } \Delta \rho ) , \qquad \hat { \mathbf { s h } } = \mathbf { s h } + \Delta \mathbf { s h } ,\tag{8}
$$

where $\rho$ is the per-gaussian density before opacity mapping, and the decay factor $\gamma _ { \rho }$ controls the sensitivity of density adjustment. A dominance regularization

$$
\mathcal { L } _ { \mathrm { d o m } } = \lambda _ { \mathrm { d o m } } \frac { \Vert \Delta \mathbf { s } \mathbf { h } ^ { ( l ) } \Vert _ { 1 } } { \Vert \Delta \mathbf { s } \mathbf { h } ^ { ( b ) } \Vert _ { 1 } + \epsilon }\tag{9}
$$

ensures interpretable control, letting $s _ { \mathrm { b r i g h t } }$ dominate exposure, where $\lambda _ { \mathrm { d o m } }$ sets the weight and Ïµ is a small constant.

## 3.3. Appearance Refinement Module (ARM)

While the global enhancer corrects scene-level illumination, real scenes still exhibit local variations such as uneven lighting, material-dependent reflectance, and view-dependent highlights. We therefore introduce an Appearance Refinement Module (ARM) to perform per-gaussian corrections on top of the global result.

High-level feature construction. High-frequency parts $( H L , L H , H H )$ are processed by FGCA similar to Eq. (5) to obtain fHF. A multi-scale pyramid $\{ F _ { i } ^ { ( l ) } ( f _ { \mathrm { H F } } ) \}$ is then built for each view, where $F _ { i } ^ { ( l ) } ( \cdot )$ denotes the feature map of view i at level l. Each gaussian center $x _ { j }$ is projected to all views, and multi-view features are bilinearly sampled as

$$
f _ { \mathrm { s a m p l e } } = \Phi _ { \mathrm { s a m p l e } } ( x _ { j } , \{ F _ { i } ^ { ( l ) } ( f _ { \mathrm { H F } } ) \} _ { i , l } ) ,\tag{10}
$$

We build two descriptors for each gaussian: the local descriptor $f _ { \mathrm { l o c a l } }$ merges $f _ { \mathrm { s a m p l e } }$ with multi-view and geometric cues, while the global descriptor $f _ { \mathrm { g l o b a l } }$ includes SH, opacity, and style to represent illumination. These features capture texture and shading for robust local texture refinement.

<!-- image-->

<!-- image-->  
(a) RE10K

<!-- image-->  
(b) RE10K â ACID

<!-- image-->  
Figure 6. Visualization of novel view synthesis results. (a) Results on the RE10K dataset, tested after training. (b) Generalization results on ACID via direct cross-dataset inference (RE10K â ACID).

Table 1. Novel view synthesis results of RE10K dataset. Best results are red, second-best are blue. \* denotes MVSplat.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>PSNR / SSIM / LPIPS</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>PSNR / SSIM /LPIPS</td></tr><tr><td rowspan=1 colspan=1>SCI + *</td><td rowspan=1 colspan=1>18.23 / 0.809 / 0.189</td><td rowspan=1 colspan=1>* + SCI</td><td rowspan=1 colspan=1>18.13 / 0.804 / 0.228</td></tr><tr><td rowspan=1 colspan=1>Zero-DCE + *</td><td rowspan=1 colspan=1>15.50 / 0.761 / 0.259</td><td rowspan=1 colspan=1>* + Zero-DCE</td><td rowspan=1 colspan=1>12.70 / 0.665 / 0.357</td></tr><tr><td rowspan=1 colspan=1>DLL + *</td><td rowspan=1 colspan=1>17.52 / 0.780 / 0.214</td><td rowspan=1 colspan=1>* + DLL</td><td rowspan=1 colspan=1>17.29 / 0.742 / 0.293</td></tr><tr><td rowspan=1 colspan=1>Retinex + *</td><td rowspan=1 colspan=1>15.11 / 0.718 / 0.243</td><td rowspan=1 colspan=1>* + Retinex</td><td rowspan=1 colspan=1>14.97 / 0.714 / 0.254</td></tr><tr><td rowspan=1 colspan=1>LIME + *</td><td rowspan=1 colspan=1>15.10 / 0.781 /0.189</td><td rowspan=1 colspan=1>* +LIME</td><td rowspan=1 colspan=1>14.88 / 0.767 / 0.218</td></tr><tr><td rowspan=1 colspan=1>KinD + *</td><td rowspan=1 colspan=1>14.20 / 0.716 / 0.290</td><td rowspan=1 colspan=1>* + KinD</td><td rowspan=1 colspan=1>13.66 / 0.708 / 0.306</td></tr><tr><td rowspan=1 colspan=1>LLFlow + *</td><td rowspan=1 colspan=1>16.48 / 0.787 / 0.208</td><td rowspan=1 colspan=1>* + LLFlow</td><td rowspan=1 colspan=1>16.31 / 0.777 / 0.218</td></tr><tr><td rowspan=1 colspan=1>EnlightenGAN + *</td><td rowspan=1 colspan=1>15.57 / 0.783 / 0.209</td><td rowspan=1 colspan=1>* + EnlightenGAN</td><td rowspan=1 colspan=1>15.30 / 0.770 / 0.228</td></tr><tr><td rowspan=1 colspan=1>RUAS + *</td><td rowspan=1 colspan=1>11.49 / 0.632 / 0.473</td><td rowspan=1 colspan=1>* + RUAS</td><td rowspan=1 colspan=1>11.50 / 0.624 / 0.477</td></tr><tr><td rowspan=1 colspan=1>Gaussian-DK</td><td rowspan=1 colspan=1>14.54 / 0.632 / 0.484</td><td rowspan=1 colspan=1>Aleth-NeRF</td><td rowspan=1 colspan=1>20.74 / 0.721 / 0.404</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>21.43 / 0.815 / 0.168</td><td rowspan=1 colspan=1>MVSplat</td><td rowspan=1 colspan=1>6.72 / 0.264 / 0.348</td></tr></table>

Table 2. Generalization on the ACID dataset (RE10K â ACID). Retrain Gaussian-DK and Aleth-NeRF. \* denotes MVSplat.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>PSNR / SSIM / LPIPS</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>PSNR / SSIM /LPIPS</td></tr><tr><td rowspan=1 colspan=1>SCI + *</td><td rowspan=1 colspan=1>17.52 / 0.755 / 0.232</td><td rowspan=1 colspan=1>* + SCI</td><td rowspan=1 colspan=1>17.52 / 0.755 / 0.232</td></tr><tr><td rowspan=1 colspan=1>Zero-DCE + *</td><td rowspan=1 colspan=1>19.27 / 0.792 / 0.202</td><td rowspan=1 colspan=1>* + Zero-DCE</td><td rowspan=1 colspan=1>19.38 / 0.794 / 0.216</td></tr><tr><td rowspan=1 colspan=1>DLL + *</td><td rowspan=1 colspan=1>19.84 / 0.779 / 0.244</td><td rowspan=1 colspan=1>* + DLL</td><td rowspan=1 colspan=1>19.94 / 0.753 / 0.278</td></tr><tr><td rowspan=1 colspan=1>Retinex + *</td><td rowspan=1 colspan=1>16.77 / 0.732 / 0.262</td><td rowspan=1 colspan=1>* + Retinex</td><td rowspan=1 colspan=1>16.77 / 0.732 / 0.262</td></tr><tr><td rowspan=1 colspan=1>LIME + *</td><td rowspan=1 colspan=1>17.97 / 0.708 / 0.223</td><td rowspan=1 colspan=1>* + LIME</td><td rowspan=1 colspan=1>17.31 / 0.709 / 0.236</td></tr><tr><td rowspan=1 colspan=1>KinD + *</td><td rowspan=1 colspan=1>16.72 / 0.777 / 0.276</td><td rowspan=1 colspan=1>* + KinD</td><td rowspan=1 colspan=1>16.72 / 0.777 / 0.276</td></tr><tr><td rowspan=1 colspan=1>LLFlow + *</td><td rowspan=1 colspan=1>17.95 / 0.772 / 0.221</td><td rowspan=1 colspan=1>* + LLFlow</td><td rowspan=1 colspan=1>17.61 / 0.770 / 0.237</td></tr><tr><td rowspan=1 colspan=1>EnlightenGAN + *</td><td rowspan=1 colspan=1>16.97 / 0.788 / 0.265</td><td rowspan=1 colspan=1>* + EnlightenGAN</td><td rowspan=1 colspan=1>16.58 / 0.778 / 0.279</td></tr><tr><td rowspan=1 colspan=1>RUAS + *</td><td rowspan=1 colspan=1>11.70 / 0.644 / 0.416</td><td rowspan=1 colspan=1>* + RUAS</td><td rowspan=1 colspan=1>11.63 / 0.640 / 0.424</td></tr><tr><td rowspan=1 colspan=1>Gaussian-DK</td><td rowspan=1 colspan=1>11.55 / 0.482 / 0.564</td><td rowspan=1 colspan=1>Aleth-NeRF</td><td rowspan=1 colspan=1>15.89 / 0.521 / 0.452</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>22.69 / 0.814 / 0.175</td><td rowspan=1 colspan=1>MVSplat</td><td rowspan=1 colspan=1>7.45 / 0.250 / 0.397</td></tr></table>

Tri-branch appearance modeling. To capture spatially varying appearance effects, ARM models each gaussianâs appearance as three components: illumination, material, and view-dependent factors. They respectively capture exposure variation, surface reflectance and texture, and directional effects. Each branch predicts a scalar and an embedding from $f _ { \kappa } ( x )$ generated by high-level feature:

$$
A _ { \kappa } ( x ) , e _ { \kappa } ( x ) = \Phi _ { \kappa } ( f _ { \kappa } ( x ) ) , \quad \kappa \in \{ L , M , V \} ,\tag{11}
$$

where $A _ { \kappa } ( x )$ controls modulation strength and $e _ { \kappa } ( x )$ encodes local context. Illumination and material scalars are constrained by a sigmoid, while the view scalar adopts tanh for bidirectional adjustment. An MLP fuses the three as $A _ { X } ( x ) = \Psi ( [ A _ { L } , A _ { M } , A _ { V } ] )$ , which, together with $\{ e _ { \kappa } ( x ) \}$ , drives attention-based refinement. The branches are supervised by normal-light illumination, reflectance, and illumination-difference maps generated by [65].

We design a physically guided cross-attention where $f _ { \mathrm { l o c a l } }$ acts as the query, the factor-specific embeddings $e _ { \kappa } ( x )$ serve as keys, and $f _ { \mathrm { g l o b a l } }$ attributes provide values. This allows each gaussian to dynamically exchange information with global attributes and retrieve only the physically relevant cues. For efficiency, the 3D scene is divided into voxel windows, and attention is computed within each window:

$$
f _ { \kappa } = \mathrm { W C A } ( q = f _ { \mathrm { l o c a l } } , k = e _ { \kappa } , v = f _ { \mathrm { g l o b a l } } , \mathcal { W } _ { \mathrm { i d } } ) ,\tag{12}
$$

where Îº denotes illumination(low), material(mid), and view factors(high), and $\mathcal { W } _ { \mathrm { i d } }$ specifies the window grouping. WCA represents windowed 3D cross-attention. The features are decoded into frequency-weighted residuals:

$$
\begin{array} { r l } & { \Delta \mathbf { s h } _ { \mathrm { l o c } } , \Delta \alpha _ { \mathrm { l o c } } = [ \lambda _ { 0 } h _ { \mathrm { s h } } ^ { ( 0 ) } ( f _ { \mathrm { l o w } } ) , \lambda _ { 1 } h _ { \mathrm { s h } } ^ { ( 1 ) } ( f _ { \mathrm { m i d } } ) , } \\ & { \qquad \lambda _ { 2 } h _ { \mathrm { s h } } ^ { ( 2 ) } ( f _ { \mathrm { h i g h } } ) ] , \lambda _ { \alpha } h _ { \alpha } ( f _ { \mathrm { m i d } } ) , } \end{array}\tag{13}
$$

where $\lambda _ { i }$ are weights balancing low-, mid-, and high-order corrections. The globally attenuated density is then mapped

<!-- image-->  
Figure 7. Generalization results on VV and DICM dataset via cross-dataset inference (RE10K â VV(top); RE10K â DICM(bottom)).

<!-- image-->  
SCI GTFigure 8. Generalization results on the L3DS dataset via cross-dataset inference (RE10K â L3DS).

Table 3. Generalization on five real-world datasets. Retrain Gaussian-DK and Aleth-NeRF. \* denotes MVSplat.
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=2 colspan=5>NIQEâ / MUSIQâVV     NPE    DICM    LIME    MEF</td></tr><tr><td rowspan=1 colspan=1>NPE</td><td rowspan=1 colspan=1>DICM</td><td rowspan=1 colspan=1>LIME</td><td rowspan=1 colspan=1>MEF</td></tr><tr><td rowspan=2 colspan=1>SCI + *Zero-DCE + *</td><td rowspan=2 colspan=1>4.25 / 40.524.13 / 49.15</td><td rowspan=1 colspan=1>7.27 / 37.34</td><td rowspan=1 colspan=1>4.70 / 42.07</td><td rowspan=1 colspan=1>4.38 / 49.56</td><td rowspan=2 colspan=1>3.80 / 48.263.96 / 50.82</td></tr><tr><td rowspan=1 colspan=1>6.32 / 45.85</td><td rowspan=1 colspan=1>3.46 / 50.33</td><td rowspan=1 colspan=1>4.50 / 53.15</td></tr><tr><td rowspan=1 colspan=1>DLL + *</td><td rowspan=1 colspan=1>3.89 / 43.45</td><td rowspan=1 colspan=1>5.48 / 41.01</td><td rowspan=1 colspan=1>3.35 / 45.40</td><td rowspan=1 colspan=1>4.17 / 44.89</td><td rowspan=1 colspan=1>3.76 / 45.18</td></tr><tr><td rowspan=1 colspan=1>Retinex + *</td><td rowspan=1 colspan=1>4.33 / 56.10</td><td rowspan=1 colspan=1>5.63 /50.85</td><td rowspan=1 colspan=1>3.54/57.70</td><td rowspan=1 colspan=1>4.87 / 52.47</td><td rowspan=1 colspan=1>|4.19 /54.80</td></tr><tr><td rowspan=1 colspan=1>LIME + *</td><td rowspan=1 colspan=1>3.93 / 48.88</td><td rowspan=1 colspan=1>4.97 / 46.59</td><td rowspan=1 colspan=1>3.40 / 52.60</td><td rowspan=1 colspan=1>4.55 / 51.93</td><td rowspan=1 colspan=1>3.84 / 50.84</td></tr><tr><td rowspan=1 colspan=1>KinD + *</td><td rowspan=1 colspan=1>4.52 / 45.16</td><td rowspan=1 colspan=1>6.96 / 45.65</td><td rowspan=1 colspan=1>3.62 / 52.76</td><td rowspan=1 colspan=1>4.74 / 50.95</td><td rowspan=1 colspan=1>4.03 / 49.69</td></tr><tr><td rowspan=1 colspan=1>LLFlow + *</td><td rowspan=1 colspan=1>4.31 / 45.31</td><td rowspan=1 colspan=1>6.91 / 43.94</td><td rowspan=1 colspan=1>3.41 / 51.34</td><td rowspan=1 colspan=1>4.61 / 50.54</td><td rowspan=1 colspan=1>3.98 / 48.06</td></tr><tr><td rowspan=1 colspan=1>EnlightenGAN + *</td><td rowspan=1 colspan=1>4.56 / 32.62</td><td rowspan=1 colspan=1>6.93 / 29.89</td><td rowspan=1 colspan=1>4.46 / 32.88</td><td rowspan=1 colspan=1>4.46 / 43.15</td><td rowspan=1 colspan=1>4.43 / 36.23</td></tr><tr><td rowspan=1 colspan=1>RUAS + *</td><td rowspan=1 colspan=1>5.56 / 30.86</td><td rowspan=1 colspan=1>9.41 /31.72</td><td rowspan=1 colspan=1>4.28 / 40.24</td><td rowspan=1 colspan=1>4.78 / 41.21</td><td rowspan=1 colspan=1>4.34 / 42.42</td></tr><tr><td rowspan=1 colspan=1>Aleth-NeRF</td><td rowspan=1 colspan=1>3.67 / 21.10</td><td rowspan=1 colspan=1>4.30 / 47.22</td><td rowspan=1 colspan=1>3.85 / 40.31</td><td rowspan=1 colspan=1>4.55 / 45.99</td><td rowspan=1 colspan=1>4.79 / 38.38</td></tr><tr><td rowspan=1 colspan=1>Gaussian-DK</td><td rowspan=1 colspan=1>6.35 / 16.92</td><td rowspan=1 colspan=1>5.54 / 38.60</td><td rowspan=1 colspan=1>4.51 / 46.94</td><td rowspan=1 colspan=1>4.26 / 30.72</td><td rowspan=1 colspan=1>4.77 / 30.92</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>3.88 / 49.18</td><td rowspan=1 colspan=1>3.89 / 45.46</td><td rowspan=1 colspan=1>3.91 / 46.70</td><td rowspan=1 colspan=1>4.10 / 53.66</td><td rowspan=1 colspan=1>3.59 / 56.59</td></tr></table>

to opacity $\tilde { \alpha } = \mathcal { M } ( \tilde { \rho } )$ , and the final gaussian attributes are

$$
\begin{array} { r } { \hat { \mathbf { s h } } = \hat { \mathbf { s h } } + \Delta \mathbf { s h } _ { \mathrm { l o c } } , \qquad \hat { \boldsymbol { \alpha } } = \mathrm { c l i p } \big ( \tilde { \boldsymbol { \alpha } } + \Delta \alpha _ { \mathrm { l o c } } , 0 , 1 \big ) . } \end{array}\tag{14}
$$

By combining factorized keys with windowed attention, this module performs local refinement where illumination, material, view, and geometry are jointly optimized. The resulting bright Gaussian field is defined as

$$
\mathcal { G } _ { \mathrm { b r i g h t } } = \{ ( \hat { \mu } _ { j } ^ { b } , \hat { \alpha } _ { j } ^ { b } , \hat { \Sigma } _ { j } ^ { b } , \hat { \mathbf { s h } } _ { j } ^ { b } ) \} _ { j = 1 } ^ { H \times W \times K } .\tag{15}
$$

Given $\mathcal { G } _ { \mathrm { b r i g h t } }$ , the 3DGS decoder [26] renders novel-view outputs, including RGB ËI, depth DË , and normals $\hat { N }$

## 3.4. Training Strategy and Loss Functions

Depth errors distort geometry, and appearance correction can be unstable during optimization. We therefore adopt a three-stage training strategy, progressively enabling each loss as its supervision becomes reliable.

Stage I: geometry pre-training. We train the geometry branch on dark images with photometric, gradient, depth, and normal constraints:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { g e o } } = \lambda _ { \mathrm { m s e } } \mathcal { L } _ { \mathrm { m s e } } ^ { \mathrm { d a r k } } + \lambda _ { \mathrm { g r a d } } \mathcal { L } _ { \mathrm { g r a d } } } \\ & { \qquad + \lambda _ { \mathrm { d e p t h } } \mathcal { L } _ { \mathrm { d e p t h } } + \lambda _ { \mathrm { n o r m a l } } \mathcal { L } _ { \mathrm { n o r m a l } } . } \end{array}\tag{16}
$$

$\mathcal { L } _ { \mathrm { g r a d } }$ enforces edge consistency between grayscale gradients of predicted and target images, ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ ensures depth smoothness guided by image edges, and ${ \mathcal { L } } _ { \mathrm { n o r m a l } } ~ = ~ 1 -$ $\langle \hat { \bf n } _ { \mathrm { p r e d } } , \bf n _ { \mathrm { g t } } \rangle$ , where $\mathbf { n } _ { \mathrm { g t } }$ is estimated by StableNormal [58].

Stage II: global enhancement. In addition to photometric supervision on both dark and enhanced images, we jointly optimize three style embeddings {i â {high, low, diff}} with brightness correlation and domain regularization:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { g l o b a l } } = \lambda _ { \mathrm { s t y l e } } \displaystyle \sum _ { i } \| s _ { \mathrm { b r i g h t } } ^ { i } - d _ { i } \| _ { 2 } ^ { 2 } } \\ & { ~ + \lambda _ { \mathrm { l u m } } \mathcal { L } _ { \mathrm { c o r r } } ( \Delta \mathrm { l u m } , s _ { \mathrm { b r i g h t } } ) + \lambda _ { \mathrm { d o m } } \mathcal { L } _ { \mathrm { d o m } } . } \end{array}\tag{17}
$$

Stage III: local refinement. ARM estimates illumination, material, and view factors with predicted illumination decomposition from [65] as supervision :

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p h y s } } = \lambda _ { L } \| A _ { L } - \hat { I } _ { \mathrm { i l l u m } } \| _ { 1 } + \lambda _ { M } \| A _ { M } - \hat { I } _ { \mathrm { r e f l } } \| _ { 1 } } \\ & { ~ + \lambda _ { x } \| A _ { X } - \hat { I } _ { \mathrm { i l l u m - d i f f } } \| _ { 1 } . } \end{array}\tag{18}
$$

Across appearance stages, photometric, perceptual, and color-space consistency losses are applied for stable appearance optimization::

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e c o n } } = \lambda _ { \mathrm { m s e } } \big ( \mathcal { L } _ { \mathrm { m s e } } ^ { \mathrm { d a r k } } + \mathcal { L } _ { \mathrm { m s e } } ^ { \mathrm { e n h } } \big ) , } \\ & { \mathcal { L } _ { \mathrm { l p i p s } } = \lambda _ { \mathrm { l p i p s } } \big ( \mathcal { L } _ { \mathrm { l p i p s } } ^ { \mathrm { d a r k } } + \alpha _ { \mathrm { e n h } } \mathcal { L } _ { \mathrm { l p i p s } } ^ { \mathrm { e n h } } \big ) , } \\ & { \mathcal { L } _ { \mathrm { h s v } } = \lambda _ { \mathrm { h s v } } \big ( \mathcal { L } _ { h } ^ { \mathrm { c i r c - L 1 } } + \mathcal { L } _ { s } ^ { \mathrm { m a s k e d - L 1 } } + \mathcal { L } _ { v } ^ { \mathrm { L 1 } } \big ) . } \end{array}\tag{19}
$$

The overall objective combines all terms:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { g e o } } + \mathcal { L } _ { \mathrm { g l o b a l } } + \mathcal { L } _ { \mathrm { p h y s } } + \mathcal { L } _ { \mathrm { r e c o n } } + \mathcal { L } _ { \mathrm { l p i p s } } + \mathcal { L } _ { \mathrm { h s v } } .
$$

Table 4. Quantitative comparison on the L3DS dataset (âcorridorâ, âwindowâ, âbalconyâ, âshelfâ, âplantâ, âcampusâ). Metrics: PSNRâ, SSIMâ, LPIPSâ. Retrain Gaussian-DK and Aleth-NeRF. \* denotes MVSplat.
<table><tr><td rowspan="2">Method</td><td colspan="3">&quot;corridor&quot;</td><td colspan="3">&quot;window&quot;</td><td colspan="3">balcony</td><td colspan="3"> $\mathbf { \hat { s h e l f } } ^ { 4 }$ </td><td colspan="3"> $\mathbf { \widetilde { p l a n t } } ^ { * }$ </td><td colspan="3"> $\mathbf { \hat { c a m p u s } } ^ { s }$ </td><td colspan="3">avg</td></tr><tr><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>SCI + *</td><td>21.36</td><td>0.910</td><td>0.224</td><td>16.65</td><td>0.867</td><td>0.183</td><td>14.86</td><td>0.777</td><td>0.180</td><td>14.40</td><td>0.830</td><td>0.214</td><td>12.33</td><td>0.759</td><td>0.278</td><td>12.33</td><td>0.622</td><td>0.261</td><td>15.32</td><td>0.789</td><td>0.223</td></tr><tr><td>Zero-DCE + *</td><td>20.15</td><td>0.933</td><td>0.236</td><td>21.23</td><td>0.908</td><td>0.161</td><td>18.58</td><td>0.813</td><td>0.175</td><td>20.57</td><td>0.839</td><td>0.209</td><td>20.37</td><td>0.806</td><td>0.202</td><td>19.79</td><td>0.657</td><td>0.169</td><td>20.11</td><td>0.824</td><td>0.192</td></tr><tr><td>DLL + *</td><td>18.45</td><td>0.837</td><td>0.345</td><td>14.93</td><td>0.810</td><td>0.279</td><td>20.07</td><td>0.794</td><td>0.212</td><td>18.90</td><td>0.783</td><td>0.262</td><td>13.18</td><td>0.765</td><td>0.273</td><td>13.42</td><td>0.600</td><td>0.255</td><td>16.49</td><td>0.770</td><td>0.271</td></tr><tr><td>Retinex + *</td><td>15.48</td><td>0.862</td><td>0.280</td><td>16.30</td><td>0.821</td><td>0.223</td><td>17.38</td><td>0.756</td><td>0.217</td><td>14.97</td><td>0.758</td><td>0.282</td><td>14.83</td><td>0.769</td><td>0.224</td><td>13.79</td><td>0.597</td><td>0.237</td><td>15.46</td><td>0.760</td><td>0.244</td></tr><tr><td>KinD + *</td><td>18.82</td><td>0.906</td><td>0.191</td><td>18.70</td><td>0.849</td><td>0.156</td><td>15.10</td><td>0.785</td><td>0.177</td><td>19.42</td><td>0.845</td><td>0.247</td><td>20.56</td><td>0.768</td><td>0.224</td><td>17.45</td><td>0.650</td><td>0.236</td><td>18.34</td><td>0.801</td><td>0.205</td></tr><tr><td>EnlightenGAN + *</td><td>19.68</td><td>0.909</td><td>0.245</td><td>19.16</td><td>0.845</td><td>0.196</td><td>16.65</td><td>0.804</td><td>0.227</td><td>14.86</td><td>0.832</td><td>0.254</td><td>12.56</td><td>0.767</td><td>0.322</td><td>12.58</td><td>0.639</td><td>0.271</td><td>15.91</td><td>0.800</td><td>0.253</td></tr><tr><td>RUAS + *</td><td>12.41</td><td>0.785</td><td>0.359</td><td>9.27</td><td>0.583</td><td>0.440</td><td>9.47</td><td>0.512</td><td>0.503</td><td>8.77</td><td>0.595</td><td>0.541</td><td>9.03</td><td>0.655</td><td>0.442</td><td>8.87</td><td>0.527</td><td>0.435</td><td>9.64</td><td>0.610</td><td>0.453</td></tr><tr><td>Gaussian-DK</td><td>17.81</td><td>0.745</td><td>0.193</td><td>16.66</td><td>0.538</td><td>0.318</td><td>16.30</td><td>0.462</td><td>0.332</td><td>16.36</td><td>0.555</td><td>0.367</td><td>16.69</td><td>0.596</td><td>0.324</td><td>11.69</td><td>0.375</td><td>0.629</td><td>15.92</td><td>0.545</td><td>0.361</td></tr><tr><td>Aleth-NeRF</td><td>12.24</td><td>0.448</td><td>0.529</td><td>11.09</td><td>0.360</td><td>0.612</td><td>10.59</td><td>0.311</td><td>0.588</td><td>17.59</td><td>0.601</td><td>0.610</td><td>17.24</td><td>0.573</td><td>0.581</td><td>16.05</td><td>0.467</td><td>0.616</td><td>14.13</td><td>0.460</td><td>0.589</td></tr><tr><td>Ours</td><td>22.84</td><td>0.912</td><td>0.181</td><td>21.31</td><td>0.858</td><td>0.143</td><td>20.35</td><td>0.806</td><td>0.162</td><td>21.17</td><td>0.848</td><td>0.202</td><td>20.91</td><td>0.778</td><td>0.215</td><td>20.07</td><td>0.636</td><td>0.181</td><td>21.11</td><td>0.806</td><td>0.181</td></tr></table>

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 9. Object detection results on the RE10K dataset.

## 4. Experiment

## 4.1. Experiment Setup

Datasets. We primarily train and evaluate SplatBright on the large-scale indoor dataset RE10K [67], following the protocols of pixelSplat [3] and MVSplat [8]. RE10K contains 67,477 training and 7,289 testing scenes with perframe camera intrinsics and extrinsics for multi-view supervision. To assess generalization, we evaluate on the crossdomain dataset ACID [31], which provides 11,075 training and 1,972 testing aerial outdoor scenes with similar data configuration. We also perform zero-shot evaluation on five standard low-light datasets (NPE [53], MEF [34], LIME [15], DICM [29], VV1). Since the lack camera poses, we set all poses to identity matrices during inference. Additionally, we construct a real-world dataset L3DS with six scenes (five indoor, one outdoor). Each scene contains 25-35 images captured by a tripod-mounted camera, with

2 â 3 exposure levels per viewpoint. Images are downsampled from $3 0 2 4 \times 3 0 2 4 \mathrm { t o } 4 0 0 \times 4 0 0 \mathrm { o r } 8 0 0 \times 8 0 0 .$ . Poses are estimated using COLMAP [43] on normal images. For evaluation across all datasets, we use two context views and three novel target views located between them per scene.

Implementation Details. SplatBright is implemented in PyTorch with a CUDA-based differentiable rasterizer. All models are trained on 256 Ã 256 images for 120,000 iterations using Adam optimizer $( 1 . 5 { \times } 1 0 ^ { - \bar { 4 } }$ initial learning rate) with cosine decay on a single A100 GPU. We adopt a window size of $1 6 ^ { 3 }$ for windowed 3D cross-attention (WCA). Our three-stage training is scheduled to end at 5, 000, 25, 000, and 120, 000 iterations for geometry, global, and local optimization, respectively. Training on RE10K takes approximately 20 hours, and inference runs at 40 FPS. More details are provided in the supplementary material.

Metrics. We evaluate rendering quality using PSNR [17], SSIM [55], LPIPS [63], and the no-reference NIQE [39] and MUSIQ [25] for perceptual realism.

Baseline Methods. Since no prior work addresses our task under the same setting, we compare with the following approaches: (1) original MVSplat; (2) SOTA 2D enhancement (RetinexNet [7], EnlightenGAN [21], KinD [65], LLFlow [54], LIME [15], RUAS [33], Zero-DCE [14], DiffLL [54], and SCI [35]) with MVSplat pipelines: both 2D pre-process (method+MVSplat) and 2D post-process (MVSplat+method); (3) 3D Low-light reconstruction baselines: Aleth-NeRF [10], Gaussian-DK [59], which model illumination explicitly but require per-scene optimization, whereas our method enables direct feed-forward inference.

## 4.2. Quantitative and qualitative results

We construct paired data on the RE10K and ACID datasets following Sec. 3.1. We train and evaluate our model on RE10K, with results shown in Fig. 6a and Tab. 1. Splat-Bright achieves the best results, demonstrating clear advantages in low-light multi-view reconstruction. Best and second-best results are in red and blue. To assess crossdataset generalization, we directly apply the RE10K-trained model to ACID. As shown in Fig. 6b and Tab. 2, our method reliably estimates illumination and recovers realistic colors in unseen scenes, consistently achieving top performance. We further evaluate on VV, LIME, MEF, DICM, and NPE using NIQE and MUSIQ. As shown in Tab. 3, our method performs strongly on . Although some metrics on NPE and DICM are not strictly the best, Fig. 7 shows that our results provide the most natural brightness and smoothest textures. In contrast, 2D methods often produce blotchy textures or amplified noise, Aleth-NeRF shows limited brightening with blurry edges, and Gaussian-DK frequently fails under sparse views. Finally, we evaluate on our self-captured L3DS dataset. As shown in Tab. 4 and Fig. 8, SplatBright substantially outperforms all 2D and 3D baselines: 2D methods tend to over-sharpen, Aleth-NeRF yields blurry novel views, and Gaussian-DK suffers from color shifts. Note that both Gaussian-DK and Aleth-NeRF are retrained per scene.

<!-- image-->  
Figure 10. Visual comparison of segmentation performance among 3D-based methods on the L3DS dataset.

<!-- image-->  
Figure 11. Comparison of subjective evaluation results on the NPE, DICM, MEF, LIME, and VV datasets.

## 4.3. Application to Downstream Vision Tasks

We further evaluate the effectiveness of our method on downstream vision tasks. For object detection, we adopt YOLOv8 [24] on the RE10K dataset. As shown in Fig. 9, enhancement improves detection confidence in dark regions and reduces false positives. Notably, only our method and Zero-DCE correctly detect the âpotted plantâ missing from the ground truth, while Zero-DCE fails to detect all objects. For semantic segmentation, we evaluate SegFormer [57] on our L3DS dataset. Our enhanced results produce more accurate segmentation than other 3D baselines in Fig. 10, demonstrating that our approach not only improves visual quality but also benefits downstream perception tasks.

<!-- image-->  
Figure 12. Visual comparison of ablation study on RE10K dataset.

Table 5. Ablation study on different module and loss components.
<table><tr><td colspan="2">Module Ablation (PSNRâ / SSIMâ / LPIPSâ)</td></tr><tr><td>w/o physical dark data generation (use simplified model)</td><td>19.95 / 0.781 / 0.216</td></tr><tr><td>w/o global residual SH (use multiplicative SH)</td><td>19.01 / 0.767 / 0.231</td></tr><tr><td>w/o global density optimization</td><td>18.20 / 0.735 / 0.281</td></tr><tr><td>w/o geo-freeze in appearance stage</td><td>21.32 / 0.807 / 0.170</td></tr><tr><td>w/o ARM</td><td>20.93 / 0.813 / 0.170</td></tr><tr><td>w/o FGCA</td><td>21.40/ 0.808 / 0.170</td></tr><tr><td>Full model (ours)</td><td>21.43 / 0.815 / 0.168</td></tr><tr><td colspan="2">Loss Ablation (PSNRâ / SSIMâ / LPIPSâ)</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { h s v } }$ </td><td>19.96 / 0.781 / 0.191</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { s t y l e } }$ </td><td>21.28 /0.809/ 0.174</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { { n o r m a l } } }$ </td><td>21.43 / 0.804 / 0.169</td></tr><tr><td>Full model (ours)</td><td>21.43 / 0.815 / 0.168</td></tr></table>

## 4.4. Subjective Evaluation

Since traditional quality metrics may not fully align with human perception [20] and some datasets lack ground truth, we conduct a subjective evaluation to assess visual quality. Twenty volunteers rated enhancement results of 15 real low-light scenes from NPE, DICM, MEF, LIME, and VV in terms of naturalness, brightness, and color fidelity (1â5 scale). Our method achieves the highest overall scores in Fig. 11, demonstrating superior perceptual quality.

## 4.5. Ablation Studies

Table 5 summarizes the influence of each component. Replacing our physically driven dark-data modeling with a simplified gray-decay model yields a clear performance drop, demonstrating the importance of realistic illumination modeling. In ICM, switching from residual SH to a multiplicative formulation causes unstable brightness correction and overexposed regions in Fig. 12, while disabling global density optimization produces floating artifacts. In local appearance modeling, removing ARM degrades color fidelity and yields grayish outputs, whereas discarding FGCA disrupts multi-frequency aggregation and produces blurred textures (e.g., wall corners) with higher LPIPS. Regarding losses, removing Lhsv or Lstyle reduces color and contrast consistency, and Fig. 4 shows that omitting ${ \mathcal { L } } _ { \mathrm { n o r m a l } }$ weakens geometryâappearance decoupling, harming reconstruction. More details are provided in the supplementary material.

## 5. Conclusion

We introduced SplatBright, a generalizable low-light 3D Gaussian reconstruction framework. With decoupled geometryâappearance modeling and physically guided illumination refinement, it achieves robust low-light reconstruction and consistently outperforms 2D and 3D baselines on public datasets and our self-captured multi-view data.

## References

[1] Yuanhao Cai, Zihao Xiao, Yixun Liang, Minghan Qin, Yulun Zhang, Xiaokang Yang, Yaoyao Liu, and Alan Yuille. Hdr-gs: Efficient high dynamic range novel view synthesis at 1000x speed via gaussian splatting. In NeurIPS, 2024. 2

[2] Hanzhi Chang, Ruijie Zhu, Wenjie Chang, Mulin Yu, Yanzhe Liang, Jiahao Lu, Zhuoyuan Li, and Tianzhu Zhang. Meshsplat: Generalizable sparse-view surface reconstruction via gaussian splatting. arXiv preprint arXiv:2508.17811, 2025. 2

[3] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19457â19467, 2024. 2, 7

[4] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14124â14133, 2021. 2

[5] Anpei Chen, Haofei Xu, Stefano Esposito, Siyu Tang, and Andreas Geiger. Lara: Efficient large-baseline radiance fields. In European Conference on Computer Vision, pages 338â355. Springer, 2024. 2

[6] Liangyu Chen, Xin Lu, Jie Zhang, Xiaojie Chu, and Chengpeng Chen. Hinet: Half instance normalization network for image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 182â 192, 2021. 2

[7] Wei Chen, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. In British Machine Vision Conference. British Machine Vision Association, 2018. 1, 2, 7

[8] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, pages 370â386. Springer, 2024. 2, 7

[9] Yu-Sheng Chen, Yu-Ching Wang, Man-Hsin Kao, and Yung-Yu Chuang. Deep photo enhancer: Unpaired learning for image enhancement from photographs with gans. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6306â6314, 2018. 1, 2

[10] Ziteng Cui, Lin Gu, Xiao Sun, Xianzheng Ma, Yu Qiao, and Tatsuya Harada. Aleth-nerf: Illumination adaptive nerf with concealing field assumption. In Proceedings of the AAAI conference on artificial intelligence, pages 1435â1444, 2024. 1, 2, 7

[11] Ziteng Cui, Xuangeng Chu, and Tatsuya Harada. Luminance-gs: Adapting 3d gaussian splatting to challenging lighting conditions with view-adaptive curve adjustment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26472â26482, 2025. 2

[12] Yilun Du, Cameron Smith, Ayush Tewari, and Vincent Sitzmann. Learning to render novel views from wide-baseline

stereo pairs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2

[13] Xueyang Fu, Delu Zeng, Yue Huang, Xiao-Ping Zhang, and Xinghao Ding. A weighted variational model for simultaneous reflectance and illumination estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2782â2790, 2016. 2

[14] Chunle Guo Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. Zeroreference deep curve estimation for low-light image enhancement. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pages 1780â 1789, 2020. 2, 7

[15] Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light image enhancement via illumination map estimation. IEEE Transactions on image processing, 26(2):982â993, 2016. 1, 2, 7

[16] Xin Huang, Qi Zhang, Ying Feng, Hongdong Li, Xuan Wang, and Qing Wang. Hdr-nerf: High dynamic range neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18398â18408, 2022. 1, 2

[17] Quan Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics letters, 44(13):800â801, 2008. 7

[18] Haidi Ibrahim and Nicholas Sia Pik Kong. Brightness preserving dynamic histogram equalization for image contrast enhancement. IEEE Transactions on Consumer Electronics, 53(4):1752â1758, 2007. 2

[19] Hai Jiang, Ao Luo, Haoqiang Fan, Songchen Han, and Shuaicheng Liu. Low-light image enhancement with wavelet-based diffusion models. ACM Transactions on Graphics (TOG), 42(6):1â14, 2023. 2

[20] Qiuping Jiang, Zhentao Liu, Ke Gu, Feng Shao, Xinfeng Zhang, Hantao Liu, and Weisi Lin. Single image superresolution quality assessment: a real-world dataset, subjective studies, and an objective metric. IEEE Transactions on Image Processing, 31:2279â2294, 2022. 8

[21] Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, and Zhangyang Wang. Enlightengan: Deep light enhancement without paired supervision. IEEE transactions on image processing, 30:2340â2349, 2021. 1, 2, 7

[22] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5322â5332, 2024. 1, 2

[23] Daniel J Jobson, Zia-ur Rahman, and Glenn A Woodell. A multiscale retinex for bridging the gap between color images and the human observation of scenes. IEEE Transactions on Image processing, 6(7):965â976, 1997. 2

[24] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics YOLO, 2023. 8

[25] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer.

In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148â5157, 2021. 7

[26] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 6

[27] Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc Pollefeys, and Torsten Sattler. Wildgaussians: 3d gaussian splatting in the wild. arXiv preprint arXiv:2407.08447, 2024. 1, 2

[28] Edwin H Land and John J McCann. Lightness and retinex theory. Journal of the Optical society of America, 61(1):1â 11, 1971. 1, 2

[29] Chulwoo Lee, Chul Lee, and Chang-Su Kim. Contrast enhancement based on layered difference representation of 2d histograms. IEEE transactions on image processing, 22(12): 5372â5384, 2013. 7

[30] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1833â1844, 2021. 2

[31] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14458â14467, 2021.

[32] Risheng Liu, Xin Fan, Ming Zhu, Minjun Hou, and Zhongxuan Luo. Real-world underwater enhancement: Challenges, benchmarks, and solutions under natural light. IEEE transactions on circuits and systems for video technology, 30(12): 4861â4875, 2020. 2

[33] Risheng Liu, Long Ma, Jiaao Zhang, Xin Fan, and Zhongxuan Luo. Retinex-inspired unrolling with cooperative prior architecture search for low-light image enhancement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10561â10570, 2021. 2, 7

[34] Kede Ma, Kai Zeng, and Zhou Wang. Perceptual quality assessment for multi-exposure image fusion. IEEE Transactions on Image Processing, 24(11):3345â3356, 2015. 7

[35] Long Ma, Tengyu Ma, Risheng Liu, Xin Fan, and Zhongxuan Luo. Toward fast, flexible, and robust low-light image enhancement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5637â 5646, 2022. 2, 7

[36] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7210â7219, 2021. 1, 2

[37] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 1, 2

[38] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P Srinivasan, and Jonathan T Barron. Nerf in the dark: High dynamic range view synthesis from noisy raw images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16190â16199, 2022. 1, 2

[39] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a âcompletely blindâ image quality analyzer. IEEE Signal processing letters, 20(3):209â212, 2012. 7

[40] Stephen M Pizer, E Philip Amburn, John D Austin, Robert Cromartie, Ari Geselowitz, Trey Greer, Bart ter Haar Romeny, John B Zimmerman, and Karel Zuiderveld. Adaptive histogram equalization and its variations. Computer vision, graphics, and image processing, 39(3):355â 368, 1987. 1, 2

[41] Zefan Qu, Ke Xu, Gerhard Petrus Hancke, and Rynson WH Lau. Lush-nerf: Lighting up and sharpening nerfs for lowlight scenes. arXiv preprint arXiv:2411.06757, 2024. 2

[42] Mehdi SM Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario LuciË c, Daniel Duckworth, Alexey Dosovitskiy, et al. Scene Â´ representation transformer: Geometry-free novel view synthesis through set-latent scene representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6229â6238, 2022. 2

[43] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104â4113, 2016. 7

[44] Shreyas Singh, Aryan Garg, and Kaushik Mitra. Hdrsplat: Gaussian splatting for high dynamic range 3d scene reconstruction from raw images. BMVC, 2024. 2

[45] Hao Sun, Fenggen Yu, Huiyao Xu, Tao Zhang, and Changqing Zou. Ll-gaussian: Low-light scene reconstruction and enhancement via gaussian splatting for novel view synthesis. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 4261â4270, 2025. 2

[46] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10208â 10217, 2024. 2

[47] Stanislaw Szymanowicz, Eldar Insafutdinov, Chuanxia Zheng, Dylan Campbell, Joao F Henriques, Christian Rupprecht, and Andrea Vedaldi. Flash3d: Feed-forward generalisable 3d scene reconstruction from a single image. In 2025 International Conference on 3D Vision (3DV), pages 670â 681. IEEE, 2025. 2

[48] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 1, 2

[49] Yecong Wan, Mingwen Shao, Yuanshuo Cheng, and Wangmeng Zuo. S2gaussian: Sparse-view super-resolution 3d gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 711â721, 2025. 2

[50] Chao Wang, Krzysztof Wolski, Bernhard Kerbl, Ana Serrano, Mojtaba Bemana, Hans-Peter Seidel, Karol Myszkowski, and Thomas Leimkuhler. Cinematic gaussians: Â¨ Real-time hdr radiance fields with depth of field. In Computer Graphics Forum, page e15214. Wiley Online Library, 2024. 2

[51] Haoyuan Wang, Xiaogang Xu, Ke Xu, and Rynson WH Lau. Lighting up nerf via unsupervised decomposition and enhancement. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12632â12641, 2023. 1, 2

[52] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multi-view image-based rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690â4699, 2021. 2

[53] Shuhang Wang, Jin Zheng, Hai-Miao Hu, and Bo Li. Naturalness preserved enhancement algorithm for non-uniform illumination images. IEEE transactions on image processing, 22(9):3538â3548, 2013. 7

[54] Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-Pui Chau, and Alex Kot. Low-light image enhancement with normalizing flow. In Proceedings of the AAAI conference on artificial intelligence, pages 2604â2612, 2022. 2, 7

[55] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004. 7

[56] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, and Jan Eric Lenssen. latentsplat: Autoencoding variational gaussians for fast generalizable 3d reconstruction. In European conference on computer vision, pages 456â473. Springer, 2024. 2

[57] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34: 12077â12090, 2021. 8

[58] Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal. ACM Transactions on Graphics (TOG), 43(6):1â18, 2024. 6

[59] Sheng Ye, Zhen-Hui Dong, Yubin Hu, Yu-Hui Wen, and Yong-Jin Liu. Gaussian in the dark: Real-time view synthesis from inconsistent dark images using gaussian splatting. In Computer Graphics Forum, page e15213. Wiley Online Library, 2024. 1, 2, 7

[60] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4578â4587, 2021. 2

[61] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF conference on

computer vision and pattern recognition, pages 5728â5739, 2022. 2

[62] Dongbin Zhang, Chuming Wang, Weitao Wang, Peihao Li, Minghan Qin, and Haoqian Wang. Gaussian in the wild: 3d gaussian splatting for unconstrained image collections. In European Conference on Computer Vision, pages 341â359. Springer, 2024. 2

[63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 7

[64] Tianyi Zhang, Kaining Huang, Weiming Zhi, and Matthew Johnson-Roberson. Darkgs: Learning neural illumination and 3d gaussians relighting for robotic exploration in the dark. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 12864â12871. IEEE, 2024. 2

[65] Yonghua Zhang, Jiawan Zhang, and Xiaojie Guo. Kindling the darkness: A practical low-light image enhancer. In Proceedings of the 27th ACM international conference on multimedia, pages 1632â1640, 2019. 1, 2, 5, 6, 7

[66] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu. Gpsgaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19680â19690, 2024. 2

[67] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. ACM Trans. Graph., 37(4), 2018. 3, 7

[68] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223â 2232, 2017. 2