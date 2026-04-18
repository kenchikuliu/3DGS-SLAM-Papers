# DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction

Fuzhen Jiangâ \*23060111@hdu.edu.cn Hangzhou Dianzi University Hangzhou, Zhejiang, China

Zhuoran Li   
23062207@hdu.edu.cn   
Hangzhou Dianzi University   
Hangzhou, Zhejiang, China   
Yilin Zhang   
13959886169@163.com   
Hangzhou Dianzi University   
Hangzhou, Zhejiang, China

## Abstract

3D scene reconstruction and novel-view synthesis are fundamental for VR, robotics, and content creation. However, most NeRF and 3D Gaussian Splatting pipelines assume clean inputs and degrade under real noise and artifacts. We therefore propose DenoiseSplat, a feedforward 3D Gaussian splatting method for noisy multi-view images. We build a large-scale, scene-consistent noisyâclean benchmark on RE10K by injecting Gaussian, Poisson, speckle, and salt-and-pepper noise with controlled intensities. With a lightweight MVSplat-style feed-forward backbone, we train end-to-end using only clean 2D renderings as supervision and no 3D ground truth. On noisy RE10K, DenoiseSplat outperforms vanilla MVSplat and a strong two-stage baseline (IDF + MVSplat) in PSNR/SSIM and LPIPS across noise types and levels.

## CCS Concepts

â¢ Computing methodologies â 3D imaging.

## Keywords

3D scene reconstruction, 3D Gaussian splatting, noisy multi-view inputs, image denoising, neural rendering

## 1 Introduction

3D reconstruction and novel view synthesis are core capabilities for applications such as virtual reality, robotic navigation, and digital content creation. Neural Radiance Fields (NeRF) [9] and its followups have achieved remarkable progress in novel view synthesis and high-fidelity 3D scene reconstruction, albeit at the cost of expensive per-scene optimization. More recently, 3D Gaussian Splatting (3DGS) [6] represents scenes as a set of anisotropic 3D Gaussians and achieves real-time rendering with high visual quality. Building on this idea, feed-forward frameworks such as MVSplat [3] directly predict 3D Gaussian primitives from sparse multi-view images without per-scene optimization, enabling efficient reconstruction from casually captured videos.

However, most of these methods implicitly assume that the input images are clean: captured under well-controlled conditions or carefully preprocessed, with negligible noise or degradation. In contrast, real-world multi-view dataâfor example, clips from web videos or mobile devicesâoften suffer from sensor noise, compression artifacts, and low-light degradation. Noise not only corrupts geometry estimation but also damages textures and fine details, especially when fusing information across sparse views. Under this realistic setting, 3D reconstruction from noisy multi-view inputs remains largely under-explored in the current NeRF/3DGS literature.

A straightforward engineering solution is a two-stage pipeline. One first denoises each frame in the 2D image domain using modern learning-based denoisers, such as the recent generalizable IDF network [7] or classical CNN-based models like DnCNN [14] and FFDNet [15], and then feeds the denoised images into NeRF- or 3DGS-based pipelines for 3D reconstruction. While simple and modular, this design has notable drawbacks: (1) denoising can easily introduce over-smoothing and remove subtle details; (2) independently denoising each view weakens multi-view consistency, which is crucial for reliable 3D fusion; and (3) decomposing the pipeline into multiple modules increases inference latency and system complexity. At the other extreme, directly feeding noisy images into vanilla 3DGS or MVSplat leads to rapid degradation of reconstruction quality as noise intensity increases, as we will empirically demonstrate in Sec. 4.

In this work, we instead tackle noise inside the 3D representation. Rather than treating denoising as an external pre-processing step, we aim to let a feed-forward 3D Gaussian splatting network itself learn to recover clean 3D scenes from noisy multi-view observations. Concretely, we build on the multi-view encoding and 3D decoding architecture of MVSplat and propose DenoiseSplat, a feed-forward 3D Gaussian splatting method tailored to noisy inputs. Given noisy images and camera parameters as input and clean renderings as the only supervision, DenoiseSplat is trained end-toend to predict a clean 3D Gaussian representation and its rendered views, without any access to ground-truth 3D supervision.

Fig. 1 offers a teaser overview of our works. To support this setting, we construct a multi-noise, scene-consistent noisyâclean dataset based on RealEstate10K (RE10K). For each scene, we inject one of four noise typesâGaussian, Poisson, speckle, or salt-andpepperâinto the 2D RGB images and keep the noise type and level consistent across all views within that scene, yielding paired noisyâ clean multi-view samples. This simple yet controlled degradation pipeline mimics realistic acquisition conditions with scene-level noise characteristics and provides a reproducible benchmark for studying noisy 3D reconstruction.

To further improve robustness under heavy noise, we introduce a dual-branch Gaussian head for geometryâappearance decoupling, as illustrated in Fig. 2. Instead of using a single shared head to regress all Gaussian parameters, we decouple geometry-related and appearance-related predictions into two lightweight branches: a geometric branch that outputs centers, rotations, scales, and opacities, and an appearance branch that predicts spherical-harmonics coefficients and colors. This design allows geometry to be estimated from more stable, noise-robust cues, while appearance is refined separately to absorb residual noise and color fluctuations. At test time, given only noisy multi-view inputs, a single forward pass through DenoiseSplat produces a denoised 3DGS scene and highquality novel-view renderings, without any test-time optimization. Our main contributions are summarized as follows:

<!-- image-->  
pla a 3D Gaussian representation.

â¢ Problem formulation and feed-forward framework for noisy multi-view reconstruction. We systematically study 3D reconstruction from noisy multi-view inputs and, building on MVSplat, propose DenoiseSplat, a feed-forward 3D Gaussian splatting framework specifically designed for noise-corrupted images. Our method preserves the advantages of no per-scene optimization and efficient rendering, while explicitly enhancing robustness to noise through architectural and training choices.

â¢ Dual-branch Gaussian head for geometryâappearance decoupling. We redesign the Gaussian head into a dual-branch structure, where a geometric branch predicts position-, rotation-, scale-, and opacity-related parameters and an appearance branch predicts spherical-harmonics coefficients and colors. This geometry appearance decoupling mitigates the interference between noisy appearance and stable geometry, leading to more consistent 3D structure and sharper textures under strong noise.

â¢ Multi-noise, scene-consistent noisyâclean data construction. We develop a simple yet reproducible noise degradation pipeline on RE10K, introducing four noise types (Gaussian, Poisson, speckle, and salt-and-pepper) and adopting a scene-level noise configuration, where all views of a scene share the same noise type and level. This setting better reflects realistic acquisition conditions and provides a useful benchmark for studying noisy 3D reconstruction.

â¢ Comprehensive experiments and ablations on noise levels and design choices. On the constructed noisy RE10K benchmark, we compare DenoiseSplat against vanilla MVSplat and strong two-stage baselines that combine state-of-the-art 2D denoisers with 3D reconstruction under multiple noise types and intensities. We further conduct ablation studies on the standard deviation of Gaussian noise, the hyperparameters of other noise types, and the proposed dual-branch Gaussian head, together with qualitative analyses under different noise settings. Across a wide range of noise regimes, our method consistently outperforms baselines in PSNR, SSIM, and LPIPS, while preserving more texture details and geometric consistency.

## 2 Related Works

## 2.1 Multi-view 3D Reconstruction and Gaussian Splatting Representations

Classical multi-view reconstruction methods rely on geometric techniques such as structure-from-motion (SfM), multi-view stereo (MVS), and bundle adjustment to recover scene geometry via feature matching and triangulation. While effective in many settings, their performance degrades in the presence of sparse textures, strong noise, or severe occlusions. NeRF [9] represents scenes as implicit neural radiance fields and employs differentiable volumetric rendering to achieve high-quality novel view synthesis. Subsequent works have made substantial progress in accelerating training and inference and in improving rendering quality, yet the cost of volumetric rendering remains relatively high.

3D Gaussian Splatting (3DGS) [6] instead represents the scene explicitly as a set of Gaussian primitives, each parameterized by position, covariance, opacity, and color, and uses rasterization-based rendering for efficient image synthesis. This formulation strikes a favorable balance between reconstruction quality and computational efficiency. Methods built on 3DGS have been successfully applied to few-view reconstruction, super-resolution, language-driven editing, and 3D segmentation, among other tasks. For example, MMGS improves sparse-view synthesis through multi-model synergy, SRSplat studies feed-forward super-resolution from sparse low-resolution multi-view inputs, REALM enables open-world reasoning, segmentation, and editing directly on Gaussian scenes, and Sparse4DGS extends Gaussian splatting to sparse-frame dynamic scene reconstruction [5, 10â12]. Feed-forward frameworks that directly predict Gaussian primitives from multi-view images via volumetric feature aggregation further eliminate per-scene optimization, enabling a single forward pass mapping from multi-view images to a 3DGS representation.

However, this entire line of work is largely designed and evaluated under clean-input assumptions, where image quality is high and noise is negligible. There is still a lack of systematic analysis and targeted modeling for realistic acquisition conditions, in which noisy measurements, diverse degradations, and cross-view consistency issues are ubiquitous.

## 2.2 Image Denoising and Noise-Robust Vision Models

Image denoising is a long-standing problem in low-level vision. Classical methods such as non-local means and BM3D [4] explicitly exploit image statistics and non-local self-similarity priors, and remain competitive for simple additive Gaussian noise. With the rise of deep learning, supervised denoising networks such as DnCNN [14], FFDNet [15], and RIDNet [2] trained on synthetic noisyâclean pairs have demonstrated superior performance on standard benchmarks.

However, many CNN-based denoisers are trained for a fixed noise type or a narrow range of noise levels, and often struggle to generalize to unseen degradations or mixed noise. Recently, IDF (Iterative Dynamic Filtering) [7] proposes a compact yet powerful denoising network that generates pixel-wise dynamic kernels and applies them iteratively, achieving strong generalization to diverse noise types and levels despite being trained only on single-level Gaussian noise. In this work, we adopt IDF as a strong 2D denoising expert in our Denoise-Then-MVSplat baseline due to its efficiency and generalization ability, and compare it against our proposed 3D noise-aware reconstruction pipeline.

## 2.3 Robust 3D Reconstruction from Degraded Inputs

For other forms of degradation such as blur, compression, or weather effects, several works have explored explicitly modeling input degradation in 3D reconstruction and novel view synthesis. Examples include jointly learning deblurring, deblocking, or deraining modules within NeRF-like frameworks, as well as introducing uncertainty modeling and robust estimation for dynamic or challenging scenes.

These studies collectively indicate that tightly coupling degradation modeling with 3D representations can improve reconstruction quality and robustness.

Nevertheless, dedicated studies on 3D reconstruction from noisy multi-view inputs remain limited. On the one hand, most existing methods either assume preprocessed inputs or do not systematically account for different noise types and levels. On the other hand, some robust approaches rely on per-scene optimization or complex inference procedures, which limits their scalability and applicability to real-time or large-scale scenarios.

Several recent works aim to make NeRF- or 3DGS-based methods more robust to degraded inputs. Deblur-NeRF [8] explicitly models spatially-varying blur kernels to recover sharp radiance fields from defocus or motion-blurred images. More recently, RobustGS [13] introduces a degradation-aware feature enhancement module that can be plugged into feed-forward 3DGS pipelines to better handle low-quality inputs such as noise, low light, or rain. These methods demonstrate that explicitly modeling degradations in the reconstruction pipeline can substantially improve robustness. However, they typically focus on specific degradations $( \mathrm { e . g . }$ , blur) or treat âlow-qualityâ conditions in a more generic way, and few works systematically study multi-type, multi-level additive noise on large-scale multi-view datasets. In contrast, our work targets noisy multi-view reconstruction with diverse noise types and intensities, and provides a dedicated noisyâclean benchmark together with a feed-forward 3D Gaussian Splatting network trained end-to-end on noisy inputs.

## 3 Methodology

Our goal is to directly predict a clean 3D Gaussian scene representation and its renderings from noisy multi-view inputs. The overall pipeline is illustrated in Fig. 2. We first construct multinoise-type noisyâclean paired multi-view data based on RE10K. Then, we design a feed-forward Gaussian splatting network on top of the MVSplat framework, which takes noisy images as input and is trained end-to-end with clean renderings as supervision. At test time, a single forward pass suffices to obtain a clean 3DGS scene and high-quality novel views.

## 3.1 Problem Definition and Overall Pipeline

Given a collection of 3D scenes $S = \left\{ s \right\}$ , each scene ?? provides ?? clean images $\{ I _ { s , v } ^ { \mathrm { G T } } \} _ { v = 1 } ^ { V }$ with corresponding camera parameters $\{ { C _ { s , v } } \} _ { v = 1 } ^ { V }$ . Following the noise degradation process in Sec. 3.2, we synthetically generate noisy observations by applying a 2D RGBdomain corruption operator $\mathcal { D } ( \cdot )$ to each clean image while keeping the viewpoints unchanged:

$$
\begin{array} { r l r } { \left\{ \tilde { I } _ { s , v } ^ { \mathrm { n o i s e } } , \ I _ { s , v } ^ { \mathrm { G T } } , \ C _ { s , v } \right\} _ { v = 1 } ^ { V } , } & { { } } & { \tilde { I } _ { s , v } ^ { \mathrm { n o i s e } } = \mathcal { D } \left( I _ { s , v } ^ { \mathrm { G T } } \right) . } \end{array}\tag{1}
$$

Here $\tilde { I } _ { s , v } ^ { \mathrm { n o i s e } }$ denotes the noisy image obtained in the 2D RGB domain; the camera parameters and underlying scene geometry are shared with the corresponding clean view by construction.

Our goal is to learn parameters ?? of a feed-forward predictor that maps multi-view noisy observations (and cameras) to a 3D Gaussian scene representation and its renderings:

$$
\begin{array} { r } { f _ { \theta } : ( \{ \tilde { I } _ { s , v } ^ { \mathrm { n o i s e } } \} _ { v = 1 } ^ { V } , \{ C _ { s , v } \} _ { v = 1 } ^ { V } ) \longrightarrow ( \mathcal { G } _ { s } , \{ \hat { I } _ { s , v } \} _ { v = 1 } ^ { V } ) , } \end{array}\tag{2}
$$

<!-- image-->  
rOvervih prooseFmework r retuc onoyulviputsStart iu a reconstruction under noise by decoupling structural and color-related predictions.

where $\mathcal { G } _ { s }$ is the reconstructed 3D scene represented by Gaussian primitives, and $\hat { I } _ { s , v }$ is rendered from $\mathcal { G } _ { s }$ under $C _ { s , v }$ . During training, we use only 2D image-domain supervision by comparing $\hat { I } _ { s , v }$ with $I _ { s , v } ^ { \mathrm { G T } }$ . At test time, the clean images are not available; the model predicts $\mathcal { G } _ { s }$ and $\hat { I } _ { s , v }$ conditioned on noisy multi-view inputs and camera parameters in a single forward pass.

## 3.2 Noise Degradation and Dataset Construction

Modeling multiple noise types. We build our noisy multi-view benchmark on top of the RealEstate10K (RE10K) dataset [1], which contains camera trajectories for approximately 80,000 video clips collected from around 10,000 YouTube real-estate videos, totalling about 10 million frames. For each clip, the dataset provides a sequence of calibrated camera poses along a viewing trajectory, making it a standard testbed for novel view synthesis and 3D scene reconstruction. Starting from the original clean multi-view RE10K data, we synthesize four common noise types in the 2D RGB image domain: Gaussian noise, Poisson noise, speckle noise, and salt-andpepper noise. Let $I \in [ 0 , 1 ] ^ { H \times W \times 3 }$ denote an input normalized image and Ë?? the degraded one. The four noise types can be summarized as:

â¢ Gaussian noise: $\tilde { I } = \mathrm { c l i p } ( I + N ( 0 , \sigma ^ { 2 } ) ) \mathrm { ; }$

â¢ Poisson noise: Ë?? = clip  ?? Â· Poisson(?? /??) ;

â¢ Speckle noise: $\tilde { I } = \mathrm { c l i p } \big ( I + I \odot N ( 0 , \sigma ^ { 2 } ) \big )$

â¢ Salt-and-pepper noise: each pixel is set to 0 or 1 with probability ?? .

Here ?? or ?? is uniformly sampled within a predefined range, e.g., the standard deviation of Gaussian noise is sampled from [0.08, 0.12], and the total ratio of salt-and-pepper noise is sampled from [0.015, 0.03]. All specific values are managed through a unified configuration file in our implementation.

Scene-level noise configuration. In practice, images from the same scene are typically captured with the same device and exposure settings, so their noise types and strengths tend to be consistent across views. To reflect this, we adopt a scene-level noise sampling strategy. For each scene ??:

(1) uniformly sample a noise type $t _ { s }$ from the four candidates; (2) sample a noise parameter $\phi _ { s }$ within the intensity range corresponding to type ???? ;

(3) apply the same pair $( t _ { s } , \phi _ { s } )$ to all views {?? } of that scene. The resulting noisyâclean multi-view data thus maintains noise consistency at the scene level while exhibiting diversity across scenes. All noise configurations (types and strengths) are recorded in a separate metadata file to facilitate reproducibility and ablation studies.

## 3.3 Feed-forward Gaussian Splatting Network

Gaussian scene representation and rendering. We follow 3DGS [6] and represent a scene ?? as a set of Gaussian primitives

$$
\begin{array} { r } { \mathcal G _ { s } = \{ g _ { i } \} _ { i = 1 } ^ { N } , \quad g _ { i } = ( \mathbf x _ { i } , \Sigma _ { i } , \alpha _ { i } , \mathbf c _ { i } ) , } \end{array}\tag{3}
$$

where $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ is the center position, $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ is the covariance matrix, $\alpha _ { i } \in [ 0 , 1 ]$ is the opacity, and c?? denotes the appearance parameters (spherical-harmonics coefficients plus a base color). Given camera parameters $C _ { s , v }$ for view ??, a differentiable Gaussian splatting renderer R produces

$$
\begin{array} { r } { \hat { I } _ { s , v } = \mathcal { R } ( { G } _ { s } , C _ { s , v } ) \in \mathbb { R } ^ { H \times W \times 3 } . } \end{array}\tag{4}
$$

MVSplat-based feed-forward architecture. Our network is built on the feed-forward design of MVSplat [3]: a shared multi-view 2D encoder extracts per-view features, which are then lifted and aggregated in a geometry-aligned manner to form a unified 3D representation (Fig. 2). Concretely, the pipeline consists of three stages:

â¢ Multi-view feature encoding. For each noisy view $\tilde { I } _ { s , v } ^ { \mathrm { n o i s e } } ;$ a shared-weight 2D convolutional encoder produces multi-scale feature maps $\{ \mathbf { F } _ { s , v } ^ { ( l ) } \} _ { l = 1 } ^ { L }$ . Rather than explicitly outputting a denoised image, the encoder learns feature representations that retain task-relevant structures while being tolerant to noise, under the downstream reconstruction objective.

â¢ Geometry-aligned 3D aggregation. Conditioned on camera parameters ${ \cal C } _ { s , v } ,$ we follow MVSplat and warp multi-view features into a plane-sweep volume via differentiable homography projection over discrete depth planes. The resulting 3D feature volume $\mathbf { F } _ { s } ^ { \mathrm { 3 D } }$ encodes multi-view consistency signals and serves as the input to the Gaussian prediction head.

â¢ Dual-branch Gaussian parameter prediction. Different from the single shared head in MVSplat, we adopt a dual-branch Gaussian head tailored for noisy inputs. The aggregated features $\mathbf { F } _ { s } ^ { \mathrm { 3 D } }$ are fed into two lightweight decoupled branches: (1) a geometry branch that predicts centers $\mathbf { x } _ { i } ,$ , rotations, scales, and opacities $\alpha _ { i } ,$ and (2) an appearance branch that predicts spherical-harmonics coefficients and colors $\mathbf { c } _ { i }$ . This decoupling is a modeling choice that allows the geometry branch to prioritize relatively stable structural cues, while the appearance branch can account for residual noise and color fluctuations. Empirically, this design improves structural consistency and texture sharpness across varying noise types and intensities.

2D rendering supervision and loss functions. During training, we supervise reconstruction only in the 2D image domain by comparing $\hat { I } _ { s , v }$ with $I _ { s , v } ^ { \mathrm { G T } }$ . We use a combination of pixel-wise $\ell _ { 1 }$ loss and structural similarity (SSIM):

$$
\mathcal { L } = \sum _ { s } \sum _ { v } \bigl ( \lambda _ { 1 } \| \hat { I } _ { s , v } - I _ { s , v } ^ { \mathrm { G T } } \| _ { 1 } + \lambda _ { 2 } \left( 1 - \mathrm { S S I M } ( \hat { I } _ { s , v } , I _ { s , v } ^ { \mathrm { G T } } ) \right) \bigr ) ,\tag{5}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are weighting coefficients. The loss is averaged over pixels within each view and summed over training scenes and views.

After end-to-end training on the noisyâclean paired data (Sec. 3.2), the model learns to predict a Gaussian scene representation whose renderings match the clean targets under the above objective. At test time, given only noisy multi-view inputs, a single forward pass through DenoiseSplat outputs $\mathcal { G } _ { s }$ and corresponding renderings, without test-time optimization or an external 2D denoiser.

## 3.4 Cross-Branch Boundary-Guided Appearance Correction (CBC)

Our dual-branch design decouples geometry and appearance to improve robustness under noisy inputs, yet appearance estimation may still degrade when the geometry branch is uncertain, especially near geometric boundaries (e.g., depth/disparity discontinuities and occlusion edges). In such regions, even small geometric inaccuracies can coincide with perceptually salient artifacts after rendering. To mitigate this cross-branch error propagation in a controlled manner, we introduce a lightweight cross-branch correction mechanism, termed CBC, which leverages geometry-derived boundary strength and confidence as conditional signals to gate a residual refinement in the appearance branch.

Geometry-conditioned boundary gating. We construct a boundaryguided gating map from the full-resolution disparity (or depth) predicted by the geometry branch, together with a confidence proxy derived from the candidate distribution over disparity/depth hypotheses (e.g., obtained by applying softmax to the cost-volume logits along the plane dimension). Using disparity ?? for clarity, we

compute

$$
E = \left\| \nabla \delta \right\| , \quad C = \operatorname* { m a x } ( \mathrm { p d f } ) , \quad B = \operatorname { n o r m } ( E ) \cdot ( 1 - C ) ,\tag{6}
$$

where ?? denotes boundary strength (the gradient magnitude of ??), and ?? is the maximum probability of the disparity/depth candidate distribution pdf as a confidence proxy. The resulting $B \in \left[ 0 , 1 \right]$ emphasizes regions that are simultaneously boundary-like and lowconfidence: ?? increases when ?? is strong and ?? is small.

Boundary-guided residual correction in appearance. Let ?? denote the output of the appearance branch $( \mathrm { e . g . }$ , SH/color-related parameters or logits). CBC predicts a residual correction Î?? via a lightweight CNN ??(Â·) conditioned on $( A , B , C )$ :

$$
\Delta A = g ( [ A , B , C ] ) , \quad A ^ { \prime } = A + B \odot \Delta A ,\tag{7}
$$

where [Â·] denotes channel-wise concatenation and â is the elementwise product. This design applies refinement primarily where ?? is large, thereby suppressing boundary artifacts while avoiding unnecessary modification in non-boundary or high-confidence regions. Implementation detail (gradient isolation). During training, we apply stop-gradient to the geometry-derived signals by detaching ?? and ?? (i.e., detach). As a result, CBC updates only the appearancebranch parameters and the CBC module parameters, preventing appearance gradients from back-propagating into the geometry branch. This realizes an explicit cross-branch interaction mechanism: geometry provides conditional boundary/confidence cues, and appearance performs a gated residual refinement accordingly. We provide qualitative evidence of CBC in Fig. 3.

<!-- image-->  
Figure 3: Zoom-in comparison near geometric boundaries.Without CBC, residual noise tends to be amplified around boundaries.CBC uses geometry-derived boundary strength and condenc t  hheil ennt  heppeance bran, supps boundary noise artifacts.

## 4 Experiments

In this section, we evaluate the proposed method on the noisy multi-view dataset constructed from RE10K. We first describe the experimental setup and baselines, then present overall results and qualitative analyses, and finally conduct ablation studies to examine the effect of key design choices.

## 4.1 Experimental Settings

Datasets and protocol. We use RealEstate10K (RE10K). For each scene, we uniformly sample a fixed number of frames as multi-view inputs and synthesize noisyâclean pairs as in Sec. 3.2. Specifically, we apply a single randomly sampled noise type (Gaussian, Poisson, speckle, or salt-and-pepper) with a randomly sampled intensity (within the specified ranges) to all views of the scene in the 2D RGB domain, producing Ë?? noise??,?? with clean targets $I _ { s , v } ^ { \mathrm { G T } }$ . Splits are scenelevel with no overlap across train/val/test. Unless stated otherwise, we report results on both seen views (training views) and novel views extrapolated along the camera trajectory, to assess reconstruction and generalization jointly.

(a)  
(b)  
Ours  
GT  
<!-- image-->  
Figure 4: Qualitative comparison of reconstruction results from noisy RE10K inputs. For each scene, we show the noisy input view, MVSplat-Noisy(a), Denoise-Then-MVSplat (IDF + MVSplat)(b), Ours, and the clean ground-truth images.

Metrics. We report PSNR (â), SSIM (â), and LPIPS (â), averaged over the test set; we additionally break down performance on seen vs. novel views.

Baselines. We compare: (i) MVSplat-GT (upper bound), trained and evaluated on clean RE10K; (ii) MVSplat-Noisy, the cleantrained MVSplat evaluated directly on noisy inputs without finetuning; (iii) Denoise-Then-MVSplat, a two-stage pipeline applying IDF [7] (official pretrained weights, default inference) per view and then feeding the restored images into clean-trained MVSplat [3], without fine-tuning either module; and (iv) Ours (DenoiseSplat), an MVSplat-style feed-forward 3DGS model retrained on our noisyâ clean benchmark with noisy inputs and clean 2D supervision, without test-time optimization.

Efficiency. To assess potential overhead, we report runtime/memory under a unified setting (single GPU, 256Ã256, 2 context views). We discard the first 5 batches for warm-up and report the mean runtime; results are summarized in Tab. 1.

Table 1: Inference efficiency comparison under the RE10K evaluation setting (256Ã256, 2 context views). Encoder time is measured per scene, while decoder time is measured per rendered target view.
<table><tr><td>Method</td><td>Enc. (ms)</td><td>Dec. (ms)</td><td>Peak Mem. (GiB)</td></tr><tr><td>MVSplat-Noisy</td><td>58.12</td><td>1.92</td><td>1.03</td></tr><tr><td>Denoise-Then-MVSplat</td><td>86.90</td><td>1.92</td><td>1.28</td></tr><tr><td>Ours</td><td>60.74</td><td>1.92</td><td>1.07</td></tr></table>

Table 2: Overall reconstruction performance on the noisy RE10K test set. We report average PSNRâ, SSIMâ, and LPIPSâ over all test views.
<table><tr><td>Method</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>MVSplat-GT(upper bound)</td><td>26.38</td><td>0.869</td><td>0.128</td></tr><tr><td>MVSplat-Noisy</td><td>24.46</td><td>0.702</td><td>0.349</td></tr><tr><td>Denoise-Then-MVSplat (IDF)</td><td>24.77</td><td>0.788</td><td>0.272</td></tr><tr><td>Ours</td><td>25.05</td><td>0.814</td><td>0.260</td></tr></table>

## 4.2 Main Results

Overall performance. Tab. 2 summarizes the reconstruction accuracy and perceptual quality on the noisy RE10K test set. Among methods that operate on noisy inputs, DenoiseSplat achieves the strongest overall trade-off across PSNR, SSIM, and LPIPS, and reduces the gap to the clean-input upper bound MVSplat-GT. In contrast, directly feeding noisy images to MVSplat-Noisy leads to a pronounced drop, most notably in SSIM and LPIPS, which is consistent with texture blurring and local structural inconsistencies under noise. Introducing a strong 2D denoiser in Denoise-Then-MVSplat improves all three metrics compared to MVSplat-Noisy. Nevertheless, a noticeable margin to MVSplat-GT remains, especially in perceptual quality, suggesting that view-independent 2D restoration does not fully preserve the cues needed for multi-view reconstruction under challenging noise.

Seen vs. novel views. Novel-view synthesis is generally more sensitive to geometric uncertainty than seen-view reconstruction. Consistent with this, MVSplat-Noisy exhibits a larger performance drop on novel views, indicating difficulty in maintaining reliable multi-view consistency when the inputs are corrupted. Denoise-Then-MVSplat narrows this gap to some extent, but its novel-view quality can still be limited when the 2D denoiser introduces viewdependent artifacts or removes fine details that are important for matching. By comparison, DenoiseSplat maintains more stable performance across seen and novel views, with a smaller gap between them, which is indicative of improved cross-view coherence when noise is handled within the 3D reconstruction pipeline.

Table 3: Ablation on the standard deviation ?? of Gaussian noise for our DenoiseSplat. We report average performance on the noisy RE10K test set.
<table><tr><td rowspan="2">Methods</td><td colspan="2">MVSplat-Noisy</td><td>Denoise-Then-MVSplat</td><td colspan="2">DenoiseSplat(Ours)</td></tr><tr><td>PSNR SSIM</td><td>LPIPS PSNR</td><td>SSIM LPIPS</td><td>PSNR â SSIMâ</td><td>LPIPSâ</td></tr><tr><td> $\sigma = 0 . 0 5$ </td><td>25.19 0.777</td><td>0.298</td><td>25.28 0.801 0.256</td><td>25.44 0.814</td><td>0.243</td></tr><tr><td> $\sigma = 0 . 0 8$ </td><td>24.62 0.700</td><td>0.365</td><td>24.82 0.732 0.300</td><td>24.86 0.784</td><td>0.282</td></tr><tr><td> $\sigma = 0 . 1 2$ </td><td>23.54 0.612</td><td>0.432</td><td>24.26 0.665 0.396</td><td>24.33 0.742</td><td>0.328</td></tr><tr><td> $\sigma = 0 . 1 5$ </td><td>22.79 0.557</td><td>0.472</td><td>23.14 0.603 0.436</td><td>23.87 0.712</td><td>0.360</td></tr></table>

Qualitative comparisons. Fig. 4 visualizes representative examples from the noisy RE10K test set. On scenes with rich textures and complex boundaries, MVSplat-Noisy often leaves visible residual noise and exhibits color shifts or localized distortions. Denoise-Then-MVSplat typically yields cleaner appearances, but frequently softens edges and suppresses high-frequency details, especially at higher noise levels. We further highlight recurring failure patterns of this two-stage pipeline in Fig. 5, including over-smoothing of fine textures, halo/ringing near strong boundaries, and erosion of thin structures (e.g., wires or railings). These effects are consistent with view-independent 2D restoration, which may remove or alter high-frequency cues in a manner that is not fully consistent across views, thereby weakening multi-view correspondence and boundary fidelity. In comparison, DenoiseSplat removes most visible noise while better preserving boundary sharpness and texture details, producing more coherent renderings with fewer artifacts in novel view synthesis.

<!-- image-->  
Figure 5: Failure case of the two-stage IDF+MVSplat pipeline. IDF tends to over-smooth high-frequency details, which can blur edges and reduce reconstruction quality in novel view synthesis (highlighted).

## 4.3 Ablation Study

We ablate robustness along the noise dimension, varying noise type and intensity to examine trends in both metrics and visual artifacts. Effect of Gaussian noise level

We vary the Gaussian noise standard deviation ${ \sigma } \in \{ 0 . 0 5 , 0 . 0 8 ,$ 0.12, 0.15} and report results in Tab. 3. As ?? increases, PSNR/SSIM decrease and LPIPS increases for all methods. However, degradation rates differ: MVSplat-Noisy drops sharply for $\sigma \ge 0 . 0 8 ,$ , while Denoise-Then-MVSplat remains competitive at low-to-medium noise but degrades faster at higher noise, consistent with front-end 2D denoising losing fine details required by downstream multiview reconstruction. In contrast, DenoiseSplat shows a smoother decline and maintains favorable LPIPS even under strong noise, indicating better generalization across noise intensities.

Table 4: Ablation on the hyper-parameters of other noise types for our DenoiseSplat. Each entry reports performances on the noisy RE10K test set.
<table><tr><td>Noise type</td><td>Parameter setting</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td rowspan="2">Poisson</td><td> $\mathbf { s c a l e } = 0 . 0 3$ </td><td>24.10</td><td>0.740</td><td>0.331</td></tr><tr><td> $\mathrm { s c a l e } = 0 . 0 4$ </td><td>23.74</td><td>0.723</td><td>0.350</td></tr><tr><td rowspan="2">Speckle</td><td> $\sigma = 0 . 0 2$ </td><td>25.43</td><td>0.846</td><td>0.177</td></tr><tr><td> $\sigma = 0 . 0 5$ </td><td>25.36</td><td>0.832</td><td>0.211</td></tr><tr><td rowspan="2">Salt-Pepper</td><td> $\alpha = 0 . 0 1$ </td><td>24.98</td><td>0.816</td><td>0.241</td></tr><tr><td> $\alpha = 0 . 0 3$ </td><td>24.53</td><td>0.776</td><td>0.298</td></tr></table>

## Effect of noise level for other noise types

We also vary noise hyperparameters for other types: Poisson (??????????), speckle (multiplicative variance), and salt-and-pepper (?? â {0.01, 0.03}); see Tab. 4.

We further include qualitative comparisons in Fig. 6 under representative settings (e.g., Gaussian ?? = 0.05 vs. 0.12; Poisson ?????????? = 0.03 vs. 0.04; salt-and-pepper $\alpha = 0 . 0 1$ vs. 0.03; and speckle low vs. high variance). At low noise, all methods recover main structures but MVSplat-Noisy often leaves residual noise; as noise increases, MVSplat-Noisy exhibits stronger artifacts and instability, Denoise-Then-MVSplat becomes smoother with weakened textures/boundaries, and DenoiseSplat better preserves contours and texture cues with more stable novel-view renderings.

## 5 Conclusion

In this paper, we studied 3D scene reconstruction from noisy multiview inputs. Unlike classical NeRF and 3D Gaussian Splatting-based methods that assume clean input images, we focused on a setting that more closely reflects real-world acquisition conditions, where multiple noise types, varying noise intensities, and cross-view consistency all play important roles.

To address this problem, we proposed DenoiseSplat, a feed-forward 3D Gaussian splatting network tailored to noisy inputs and built upon the MVSplat framework. We constructed a multi-noise, sceneconsistent noisyâclean paired multi-view dataset on RealEstate10K by injecting Gaussian, Poisson, speckle, and salt-and-pepper noise into the 2D RGB domain, and trained DenoiseSplat end-to-end to map noisy multi-view images to clean renderings without accessing any ground-truth 3D supervision. Within this framework, a dualbranch Gaussian head decouples geometry- and appearance-related parameters, enabling the network to perform geometry reconstruction and noise suppression inside the 3D representation. At test time, a single forward pass from noisy multi-view inputs yields a clean 3DGS scene and high-quality reconstructed views, without any test-time optimization.

<!-- image-->  
Figure 6: Qualitative results of our DenoiseSplat under different noise types and intensities. Each row corresponds to one of baseline, and columns show reconstructions from inputs corrupted by Gaussian, Poisson, Speckle, and Salt--Pepper noise with varying strengths. Our model maintains stable reconstruction quality across diverse noise conditions.

On the constructed noisy RE10K benchmark, we compared DenoiseSplat against the original MVSplat and a strong two-stage baseline that combines a state-of-the-art 2D denoiser with MVSplat. Experimental results show that DenoiseSplat achieves consistently better or competitive PSNR, SSIM, and LPIPS across diverse noise types and intensities, while offering better texture preservation and geometric consistency in novel view synthesis. Ablation studies further confirm that jointly training under multiple noise types and varying noise intensities, together with the proposed dual-branch Gaussian head, is crucial for obtaining robust 3DGS representations under noisy inputs.

Despite these promising results, our work still has several limitations that merit further exploration. First, our current noise modeling is based mainly on synthetic noise and does not yet capture more complex degradations such as real camera noise, motion blur, and compression artifacts. Second, our experiments are primarily conducted on RE10K, and the cross-dataset and real-world generalization of the model remains to be systematically evaluated. Future work could incorporate real-noise captures or more accurate camera noise models, extend the framework to broader degradation types and dynamic scenes, and integrate our approach with higher-level semantic cues, aiming for feed-forward 3D representations that are not only noise-robust but also better aligned with downstream 3D understanding tasks.

## References

[1] 2018. RealEstate10K: A Large Dataset of Camera Trajectories from Internet Video. https://google.github.io/realestate10k/. Accessed 2025-12-07.

[2] Saeed Anwar and Nick Barnes. 2019. Real Image Denoising with Feature Attention. In Proc. ICCV. 3155â3164.

[3] Yuedong Chen, Haofei Xu, Chuanxia Zheng, et al. 2024. MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-view Images. In Proc. ECCV.

[4] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. 2007. Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering. IEEE Transactions on Image Processing 16, 8 (2007), 2080â2095.

[5] Xinyuan Hu, Changyue Shi, Chuxiao Yang, Minghao Chen, Jiajun Ding, Tao Wei, Chen Wei, Zhou Yu, and Min Tan. 2025. SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images. arXiv preprint arXiv:2511.12040 (2025).

[6] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÃ¼hler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics 42, 4 (2023). https://repo-sam.inria.fr/fungraph/3dgaussian-splatting/

[7] Dongjin Kim, Jaekyun Ko, Muhammad Kashif Ali, and Tae Hyun Kim. 2025. IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising. In Proc. ICCV. https://github.com/dongjinkim9/IDF

[8] Li Ma, Xiaoyu Li, Jing Liao, Qi Zhang, Xuan Wang, Jue Wang, and Pedro V. Sander. 2022. Deblur-NeRF: Neural Radiance Fields from Blurry Images. In Proc. CVPR.

[9] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In Proc. ECCV.

[10] Changyue Shi, Minghao Chen, Yiping Mao, Chuxiao Yang, Xinyuan Hu, Jiajun Ding, and Zhou Yu. 2025. REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting. arXiv preprint arXiv:2510.16410 (2025).

[11] Changyue Shi, Chuxiao Yang, Xinyuan Hu, Minghao Chen, Wenwen Pan, Yan Yang, Jiajun Ding, Zhou Yu, and Jun Yu. 2025. Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction. arXiv preprint arXiv:2511.07122 (2025).

[12] Changyue Shi, Chuxiao Yang, Xinyuan Hu, Yan Yang, Jiajun Ding, and Min Tan. 2025. MMGS: Multi-Model Synergistic Gaussian Splatting for Sparse View Synthesis. Image and Vision Computing 158 (2025), 105512.

[13] Anran Wu, Long Peng, Xin Di, Xueyuan Dai, Chen Wu, Yang Wang, Xueyang Fu, Yang Cao, and Zheng-Jun Zha. 2025. RobustGS: Unified Boosting of Feedforward 3D Gaussian Splatting under Low-Quality Conditions. arXiv preprint arXiv:2508.03077 (2025).

[14] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. 2017. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising.

[15] Kai Zhang, Wangmeng Zuo, and Lei Zhang. 2018. FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising. IEEE Transactions on Image Processing 27, 9 (2018), 4608â4622.

IEEE Transactions on Image Processing 26, 7 (2017), 3142â3155.