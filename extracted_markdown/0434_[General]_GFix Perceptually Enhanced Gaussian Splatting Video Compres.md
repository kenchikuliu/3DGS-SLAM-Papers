# GFix: Perceptually Enhanced Gaussian Splatting Video Compression

Siyue Tengâ1, Ge Gaoâ1, Duolikun Danier2, Yuxuan Jiang1, Fan Zhang1, Thomas Davis3, Zoe Liu3, David Bull1

1Visual Information Laboratory, University of Bristol, Bristol, BS1 5DD, United Kingdom

{siyue.teng, ge1.gao, yuxuan.jiang, fan.zhang, dave.bull}@bristol.ac.uk

2School of Informatics, University of Edinburgh duolikun.danier@ed.ac.uk

3Visionular Inc., Los Altos, CA 94022 USA

{thomas, zoeliu}@visionular.com

Abstractâ3D Gaussian Splatting (3DGS) enhances 3D scene reconstruction through explicit representation and fast rendering, demonstrating potential benefits for various low-level vision tasks, including video compression. However, existing 3DGSbased video codecs generally exhibit more noticeable visual artifacts and relatively low compression ratios. In this paper, we specifically target the perceptual enhancement of 3DGSbased video compression, based on the assumption that artifacts from 3DGS rendering and quantization resemble noisy latents sampled during diffusion training. Building on this premise, we propose a content-adaptive framework, GFix, comprising a streamlined, single-step diffusion model that serves as an offthe-shelf neural enhancer. Moreover, to increase compression efficiency, We propose a modulated LoRA scheme that freezes the low-rank decompositions and modulates the intermediate hidden states, thereby achieving efficient adaptation of the diffusion backbone with highly compressible updates. Experimental results show that GFix delivers strong perceptual quality enhancement, outperforming GSVC with up to 72.1% BD-rate savings in LPIPS and 21.4% in FID.

Index TermsâVideo compression, Gaussian Splatting, LoRA, Diffusion models

## I. INTRODUCTION

Video compression is a key component in todayâs digital communication systems [1]. While video coding standards such as H.264/AVC [2] and H.265/HEVC [3] still dominate the market, research trends have shifted toward learning-based solutions, either by substituting or augmenting specific subcomponents with neural networks [4, 5], or by designing holistic, end-to-end optimizable neural codecs [6, 7]. Among these end-to-end solutions, scene-agnostic approaches (typically autoencoder-based) [8, 9] rely on sophisticated architectures to generalize across diverse spatio-temporal content. Despite their impressive compression efficiency [10], the associated computational complexity, particularly for decoding, precludes their practical deployment [11]. Moreover, these methods primarily target pixel-based distortions, such as MSE, and tend to overlook perceptual aspects that correlate better with human perception.

In contrast, scene-adaptive approaches, such as those based on Implicit Neural Representations (INRs) [12â14] learn a mapping from coordinates to pixel values, with the quantized network weights serving as the compressed representation. Decoding these scene-specific representations only requires forward passes of compact networks on pixel coordinates, hence resulting in significant complexity reductions. Building on this, recent INR-based approaches [15, 16] have achieved state-of-the-art compression performance whilst also surpassing scene-agnostic counterparts in terms of decoding speed.

More recently, 3D Gaussian Splatting (3DGS) has been widely adopted for neural rendering and reconstruction [17, 18], offering explicit geometric representation with advantages in handling occlusions and complex lighting compared to NeRF methods [19], despite limitations in transparency and dynamic texture modeling. It has also been extended to 2D image and video compression, delivering promising compression performance [20, 21] and, more importantly, much faster decoding speeds than the INR-based alternatives. However, 3DGS is often more prone to reconstruction artifacts, such as over-smoothing, blob-like structures, and boundary distortions, which become more pronounced under lossy quantization in video compression. These perceptually disruptive artifacts significantly impact the viewing experience, which, coupled with the substantial memory overhead of storing millions of Gaussians, introduces a challenging trade-off between bitrate and reconstruction quality, ultimately placing 3DGS-based codecs at a disadvantage compared to INRs.

Recently, DIFix3D+ [22] has demonstrated that images degraded by 3DGS rendering artifacts resemble those at a specific noise level in the forward diffusion process in the latent space (we refer to this as the NoiseâArtifact Alignment). This suggests a promising direction for enhancing perceptual quality by directly leveraging pre-trained diffusion models to mitigate 3DGS rendering artifacts. An investigation into whether the same principle holds for the additional distortions introduced by quantizing and entropy-constraining the Gaussian primitives in video coding is thus motivated.

To this end, our paper first validates the alignment assumption in the video compression setting and, building on this insight, proposes a streamlined framework, GFix, which exploits diffusion priors to suppress artifacts with minimal bitrate overhead. Unlike conventional multi-step refinement, our method performs single-step denoising with an adaptive, degradation-aware stepsize, enabling more effective artifact removal whilst maintaining high running efficiency. To further reduce the compression ratio, we introduce modulated LoRA (mLoRA), a lightweight adaptation technique that reduces the number of trainable parameters and the corresponding bitrate compared to vanilla LoRA [23] without quality degradation. Experimental results demonstrate the effectiveness of GFix in enhancing perceptual quality, yielding average BD-rate savings of 72.08% in terms of LPIPS, relative to the state-of-the-art Gaussian Splatting-based codec, GSVC [21].

<!-- image-->

<!-- image-->  
Fig. 1: (Left) Illustration of learnable stepsize. (Right) Average MMD between Gaussian compression artifacts (different compression ratios) and partially noisy images.

## II. METHODOLOGY

## A. ArtifactâNoise Alignment in 3DGS Compression

Diffusion models [24, 25] learn data distributions by reversing a gradual noising process, where clean samples are gradually corrupted with Gaussian perturbations following a variance schedule; a neural denoiser (e.g., U-Net) is trained to estimate and remove the noise. DIFix3D+ [22] suggests that visual distortions introduced by 3DGS rendering correspond to a particular stage (which we denote as Ï ) of this forward diffusion trajectory in the latent space. We refer to this effect as NoiseâArtifact Alignment.

In the context of video compression, we verify our hypothesis by measuring the Maximum Mean Discrepancy (MMD) between the VAE-encoded latents of GS rendering at different compression ratios and with ground-truth reference latents perturbed with varying noise levels on the UVG dataset. Fig. 1 (Right) shows the variation of normalized MMD averaged over the entire dataset; this exhibits a convex profile, with distributional similarity decreasing as the noise level deviates from the optimum. However, no clear correlation between similarity and compression ratio can be observed, likely due to inherent variations in the spatio-temporal complexity of different sequences. The artifactânoise alignment is further validated by the improved PSNR and MS-SSIM scores achieved at the MMD-optimal noise level, compared with those obtained at non-optimal levels. This is shown in Fig. 2 (bottom), which reports the average performance at the best noise level for each sequence, along with results for neighboring noise levels (i.e., steps offset by $\Delta \tau )$ . As illustrated in Fig. 2 (top), nonoptimal noise levels either fail to recover fine textures or hallucinate excessive details, leading to noticeable deviations from the original content. These findings substantiate our principle and highlight why diffusion models are particularly effective for mitigating Gaussian Splatting artifacts under lossy compression constraints.

<!-- image-->  
Fig. 2: Single-step denoising results at varying noise levels (âÏ ), with visual comparisons (top) and quantitative metrics (bottom).

## B. GFix: The Enhancement Framework

Fig. 3 illustrates the proposed GFix framework. Given an input video sequence $\mathbf { X } \in \mathbf { \bar { \Gamma } } \mathbb { R } ^ { T \times H \times W \times C }$ , we first obtain the encoded 3D Gaussian primitives from GSVC [21] (through encoding), a 3DGS-based video codec that represents the video sequence with a set of explicit 3D Gaussian primitives [17]. During decoding, these primitives are first decoded and then projected onto the image plane and rasterized to produce the rendered video frames, XË .

Our diffusion model refines $\tilde { \mathbf { X } }$ conditioned on a learnable prompt embedding (with the text prompt initialized as âremove degradationsâ) to produce the enhanced output XË . During encoding, the diffusion model is fine-tuned for only a small number of steps, while the VAE encoder and most decoder layers remain frozen. To enable efficient adaptation, we update only the prompt embedding and a selected subset of U-Net and decoder layers using our proposed mLoRA adapters. These parameters are then entropy coded into a compact form, which is transmitted alongside the original GSVC bitstream.

Based on the discussion in subsection II-A, we further introduce a mechanism to adaptively modulate the denoising strength via a learnable stepsize that adapts to the noise level induced by Gaussian rendering artifacts. As illustrated in Fig. 1 (Left), the learnable stepsize enables the diffusion model to automatically align its denoising dynamics with the artifactinduced noise distribution , ensuring optimal restoration after a single denoising step. On the receiver side, these adapter parameters are decoded and used to signal the updates to the diffusion model, which then performs content-adaptive enhancement to XË .

## C. Modulated LoRA

It is noted that fine-tuning the diffusion model in its original, full-rank parameter space, even when limited to just the decoder, is prohibitively expensive. It requires excessive training memory, introduces significant computational overhead and, more critically, produces parameter updates that are extremely costly to compress. Vanilla LoRA [23] alleviates this by restricting adaptation to a low-rank subspace, but the trade-off between performance gain (i.e., reduction in distortion) and rank (which correlates to rate) remains unsatisfactory, leaving the low-rank matrices still too large to deliver meaningful improvements in overall compression efficiency.

<!-- image-->  
Fig. 3: (Left) GFix framwork overview. During decoding, the bitstream is decoded by arithmetic decoding, restoring the reconstructed content of GSVC and the quantized modulation map (based on rounding during inference) MË . (Right) mLoRA construction.

To address this issue, inspired by recent advancements in the INR literature using modulation for parameter-efficient instance-adaptive overfitting [26, 27], we propose a novel mLoRA (modulated LoRA) technique. For an arbitrary layer with index i to be fine-tuned, we first apply truncated Singular Value Decomposition (SVD) to the base weight $\mathbf { W } _ { 0 } ^ { i } \in \mathbb { R } ^ { m \times n }$ (reshaped to 2D) of layer i:

$$
\mathbf { W } _ { 0 } ^ { i } = \mathbf { U } ^ { i } \mathbf { D } ^ { i } \mathbf { V } ^ { i \top } .\tag{1}
$$

The low-rank matrices are initialized as ${ \bf A } _ { r } ^ { i } = { \bf U } _ { r } ^ { i } { \bf D } _ { r } ^ { i }$ and $\mathbf { B } _ { r } ^ { i } = \mathbf { V } _ { r } ^ { i } \tau$ , where r denotes the rank, $\mathbf { U } _ { r } ^ { i } = \dot { \mathbf { U } } _ { [ : , 1 : r ] } ^ { i } \dot { \in } \dot { \mathbb { R } } ^ { m \times r }$ , $\mathbf { D } _ { r } ^ { i } = \mathbf { D } _ { [ 1 : r , 1 : r ] } ^ { i } \in \mathbb { R } ^ { r \times r }$ , and $\mathbf { V } _ { r } ^ { i } = \mathbf { V } _ { [ : , 1 : r ] } ^ { i } \stackrel { \cdot } { \in } \mathbb { R } ^ { \bar { n } \times { r } }$ . Compared to vanilla LoRA [23], instead of overfitting, quantizing, and entropy coding $\mathbf { A } _ { r } ^ { i }$ and $\mathbf { B } _ { r } ^ { i }$ , we keep them frozen and update only a much smaller modulation map $\mathbf { M } _ { r } ^ { i } \in \mathbb { R } ^ { r \times r }$ The updated layer is then parameterized as,

$$
\mathbf { W } _ { 0 } ^ { i } \ \to \ \mathbf { W } _ { 0 } ^ { i } + \Delta \mathbf { W } _ { 0 } ^ { i } , \quad \mathrm { w h e r e } \quad \Delta \mathbf { W } _ { 0 } ^ { i } = \mathbf { A } _ { r } ^ { i } \mathbf { M } _ { r } ^ { i } \mathbf { B } _ { r } ^ { i } .\tag{2}
$$

## D. Quantization and Entropy Modeling

The mLoRA adapters described above are attached to a total of $N _ { \mathrm { F T } }$ layers, including both convolution layers and attention layers. Their modulation maps are aggregated and concatenated across channels.

$$
\begin{array} { r } { \mathbf { M } = \mathsf { c o n c a t } \big ( \mathbf { M } _ { r } ^ { 0 } , \mathbf { M } _ { r } ^ { 1 } , \ldots , \mathbf { M } _ { r } ^ { N _ { \mathrm { F T } } } \big ) \in \mathbb { R } ^ { r \times r \times ( N _ { \mathrm { F T } } + 1 ) } . } \end{array}\tag{3}
$$

During training, quantization of M is simulated with uniform noise $\mathcal { U } ( - 0 . 5 , 0 . 5 )$ along the rate estimation path and with a Straight-Through Estimator (STE) along the distortion path, following prior work [15]. For entropy coding, we adopt an empirical non-parametric entropy model [26, 28], which we empirically verified to achieve performance comparable to more sophisticated autoregressive parametric counterparts, likely due to the compactness and high sparsity of the overfitted modulation parameters.

## E. Loss function

The diffusion backbone is fine-tuned using the following rate-distortion objective:

$$
\begin{array} { r } { \mathcal { L } _ { R D } = R + \lambda D , } \end{array}\tag{4}
$$

where $R = \mathbb { E } _ { p ( \tilde { \mathbf { M } } ) } [ - \log _ { 2 } { q ( \tilde { \mathbf { M } } ) } ]$ denotes the rate term, with $p ( \tilde { \mathbf { M } } )$ representing the true distribution of quantized modulation maps and $q ( \tilde { \mathbf { M } } )$ as the predicted probability from our entropy model, where MË denotes the noisy approximation of quantized modulation maps during training. The distortion term $D = \lambda _ { 1 } \mathcal { L } _ { \mathrm { L P I P S } } + \lambda _ { 2 } \ell _ { 2 }$ combines perceptual and pixelwise losses, following DIFix3D+ [22], where $\lambda , \lambda _ { 1 } , \lambda _ { 2 }$ are weighting coefficients that balance compression rate and reconstruction fidelity.

## III. RESULTS AND DISCUSSION

## A. Implementation Details

Our method builds on SD-Turbo [29] as the backbone diffusion model. Input HD frames are divided into non-overlapping $5 1 2 \times 5 1 2$ patches to mitigate boundary artifacts. The model was trained for 2,000 steps with a batch size of 2, an initial learning rate of 0.05, and a cosine decay schedule. Optimal mLoRA ranks are selected via grid search, yielding 1024 for the prompt embedding, 256 for the U-Net, and 512 for the VAE decoder. For each sequence, five compression levels are obtained using Î» â {0.03, 0.025, 0.01, 0.005, 0.002}.

We conducted our experiments on the widely used UVG dataset [30], which contains seven 1080p video sequences. Following [31], we used the first 96 frames of each sequence (which corresponds to a total of 672 frames) for evaluation. The sequences were converted from the raw YUV 4:2:0 colorspace to RGB 4:4:4 based on the BT.601 standard.

We compare against one state-of-the-art Gaussian Splattingbased codec, GSVC [21], one INR-based codec, NeRV [13], and one conventional codec, H.264 [2], under the medium preset and main profile setting). Results for GSVC and NeRV are reproduced using their original open-source implementations.

<!-- image-->  
Fig. 4: Average rate-quality curves on the UVG dataset. We notice a difference to the reported values of GSVC, which can be attributed to the much shorter sequence length used for evaluation (first 96 frames vs. 600 frames in GSVC).

Given our focus on perceptual quality [32], we report VMAF, LPIPS [33] and FID [34], in addition to the standard PSNR metric. The BjÃ¸ntegaard Delta Rate (BD-rate) [35] of the proposed method against different anchors are adopted to quantify the compression efficiency.

## B. Quantitative and Qualitative Analysis

Quantitative results demonstrate the effectiveness of our proposed method across different metrics. As shown in Fig. 4, in terms of LPIPS, our approach achieves significant BD-rate savings of 72.1% and 48.8% over GSVC and H.264, respectively, and outperforms other codecs across multiple bitrate levels. We also outperform GSVC and NeRV in terms of FID by over 20%, which further validates the diffusion modelâs capability in addressing the Gaussian artifacts and improving perceptual quality. With respect to VMAF, our method shows 3.0% BD-rate improvement over GSVC. Correspondingly, GFix is associated with inferior PSNR performance compared to those distortion-oriented benchmarks, which is aligned with the distortion-perception trade-off theory [36]. To ensure a fair comparison, the GSVC baseline (GSVC (FT)) is also finetuned with the same number of steps and perceptual loss. The proposed GFix achieves consistent improvements over GSVC (FT) across all evaluation metrics, confirming its effectiveness. The consistent superiority across all perceptual-oriented metrics measured highlight our GFix modelâs strong perceptual optimization capability. Qualitatively, as shown in Fig. 5, our method achieves significantly better visual quality than GSVC and NeRV, with notably improved detail preservation and a clear reduction in Gaussian and compression artifacts.

## C. Ablation Study

To validate the effectiveness of each component, we conducted ablation studies on the first 32 frames of sequence Beauty, Jockey, ReadySetGo, and YachtRide. As shown in

TABLE I: Ablation results on the UVG dataset. For v1.x, we only report the reduction in bitstream size (without entropy coding) at a comparable reconstruction quality, as the BD-rate difference would be off scale due to the large size reduction. For v2.x, v2.1 serves as the anchor for BD-rate evaluation.
<table><tr><td>Method</td><td>File Size (M)</td><td>BD-rate (%)</td></tr><tr><td>(v1.1) LoRA + decoder</td><td>242.04</td><td></td></tr><tr><td>(v1.2) mLoRA + decoder</td><td>40.05</td><td></td></tr><tr><td>(v2.1) + entropy model</td><td>0.093</td><td>0.00</td></tr><tr><td>(v2.2) + prompt</td><td>0.101</td><td>-6.36 (6.36â)</td></tr><tr><td>(v2.3) + U-Net</td><td>0.171</td><td>-15.61 (9.25â)</td></tr><tr><td>(v2.4) + learnable stepsize</td><td>0.171</td><td>-23.78 (8.17â)</td></tr></table>

<!-- image-->  
Fig. 5: Visual comparisons of NeRV, GSVC, and the proposed GFix.

Table I, our proposed mLoRA achieves a 6-fold reduction in parameter count compared to vanilla LoRA (from 242.04 MB to 40.05 MB) without compression. After incorporating the entropy model to constrain the parameter space during training, the model can be effectively compressed into a compact bitstream of less than 0.01 MB, which only accounts for approximately 7% of GSVCâs original bitstream size. Building upon this entropy-constrained baseline, we observe consistent improvements in BD-rate with each additional component.

## IV. CONCLUSION

In this paper, we present an entropy-constrained single-step diffusion pipeline with a learnable stepsize for adaptive artifact removal, complemented by a novel modulated LoRA module that improves bitrate compared to the vanilla LoRA, yielding over 6 times the compression ratio while maintaining visual quality. Experimental results on the UVG dataset demonstrate that GFix achieves superior performance against GSVC across all perceptual metrics, with notable improvements over H.264 in LPIPS and NeRV in FID. Future work should explore optimizing GFix for longer sequences through efficient GOP-based coding structures with residual coding and super-resolution [37, 38], which could potentially lead to further improvements in compression efficiency.

[1] D. Bull and F. Zhang, Intelligent image and video compression: communicating pictures. Academic Press, 2021.

[2] T. Wiegand, G. J. Sullivan, G. Bjontegaard, and A. Luthra, âOverview of the h. 264/avc video coding standard,â IEEE Transactions on circuits and systems for video technology, vol. 13, no. 7, pp. 560â576, 2003.

[3] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, âOverview of the high efficiency video coding (HEVC) standard,â TCSVT, vol. 22, no. 12, pp. 1649â1668, 2012.

[4] T. Laude and J. Ostermann, âDeep learning-based intra prediction mode decision for HEVC,â in PCS. IEEE, 2016, pp. 1â5.

[5] F. Zhang, D. Ma, C. Feng, and D. R. Bull, âVideo compression with CNN-based postprocessing,â IEEE MultiMedia, vol. 28, no. 4, pp. 74â83, 2021.

[6] G. Lu, W. Ouyang, D. Xu, X. Zhang, C. Cai, and Z. Gao, âDVC: An end-to-end deep video compression framework,â in CVPR, 2019, pp. 11 006â11 015.

[7] A. Habibian, T. v. Rozendaal, J. M. Tomczak, and T. S. Cohen, âVideo compression with rate-distortion autoencoders,â in ICCV, 2019, pp. 7033â7042.

[8] J. Li, B. Li, and Y. Lu, âDeep contextual video compression,â NeurIPS, vol. 34, pp. 18 114â18 125, 2021.

[9] , âNeural video compression with feature modulation,â in CVPR, 2024, pp. 26 099â26 108.

[10] L. Qi, Z. Jia, J. Li, B. Li, H. Li, and Y. Lu, âLong-term temporal context gathering for neural video compression,â in ECCV. Springer, 2024, pp. 305â322.

[11] S. Teng, Y. Jiang, G. Gao, F. Zhang, T. Davis, Z. Liu, and D. Bull, âBenchmarking conventional and learned video codecs with a low-delay configuration,â in VCIP. IEEE, 2024, pp. 1â5.

[12] V. Sitzmann, J. Martel, A. Bergman, D. Lindell, and G. Wetzstein, âImplicit neural representations with periodic activation functions,â NeurIPS, vol. 33, pp. 7462â7473, 2020.

[13] H. Chen, B. He, H. Wang, Y. Ren, S. N. Lim, and A. Shrivastava, âNeRV: Neural representations for videos,â NeurIPS, vol. 34, pp. 21 557â21 568, 2021.

[14] H. M. Kwan, G. Gao, F. Zhang, A. Gower, and D. Bull, âHiNeRV: Video compression with hierarchical encoding-based neural representation,â NeurIPS, vol. 36, pp. 72 692â72 704, 2023.

[15] âNVRC: Neural video representation compression,â NeurIPS, vol. 37, pp. 132 440â132 462, 2024.

[16] G. Gao, S. Teng, T. Peng, F. Zhang, and D. Bull, âGIViC: Generative implicit video compression,â arXiv preprint arXiv:2503.19604, 2025.

[17] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D Â¨ Gaussian splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[18] Y.-T. Zhan, C.-Y. Ho, H. Yang, Y.-H. Chen, J.-C. Chiang, Y.-L. Liu, and W.-H. Peng, âCAT-3DGS: A context-adaptive triplane approach to rate-distortion-optimized 3dgs compression,â in ICLR, 2025.

[19] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[20] X. Zhang, X. Ge, T. Xu, D. He, Y. Wang, H. Qin, G. Lu, J. Geng, and J. Zhang, âGaussianImage: 1000 fps image representation and compression by 2d gaussian splatting,â in ECCV. Springer, 2024, pp. 327â345.

[21] X. Liu, B. Chen, Z. Liu, Y. Wang, and S.-T. Xia, âAn exploration with entropy constrained 3d gaussians for 2d video compression,â in ICLR, 2023.

[22] J. Z. Wu, Y. Zhang, H. Turki, X. Ren, J. Gao, M. Z. Shou,

S. Fidler, Z. Gojcic, and H. Ling, âDifix3D+: Improving 3d reconstructions with single-step diffusion models,â in CVPR, 2025, pp. 26 024â26 035.

[23] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., âLoRA: Low-rank adaptation of large language models.â in ICLR, 2022.

[24] J. Song, C. Meng, and S. Ermon, âDenoising Diffusion Implicit Models,â in ICLR, 2021.

[25] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, âScore-Based Generative Modeling through Stochastic Differential Equations,â in ICLR, 2021.

[26] X. Zhang, R. Yang, D. He, X. Ge, T. Xu, Y. Wang, H. Qin, and J. Zhang, âBoosting neural representations for videos with a conditional decoder,â in CVPR, 2024, pp. 2556â2566.

[27] G. Gao, H. M. Kwan, F. Zhang, and D. Bull, âPNVC: Towards practical INR-based video compression,â in AAAI, vol. 39, no. 3, 2025, pp. 3068â3076.

[28] H. Kim, M. Bauer, L. Theis, J. R. Schwarz, and E. Dupont, âC3: High-performance and low-complexity neural compression from a single image or video,â in CVPR, 2024, pp. 9347â9358.

[29] A. Sauer, D. Lorenz, A. Blattmann, and R. Rombach, âAdversarial diffusion distillation,â in ECCV. Springer, 2024, pp. 87â103.

[30] A. Mercat, M. Viitanen, and J. Vanne, âUVG dataset: 50/120fps 4k sequences for video codec analysis and development,â in ACM MMSys, 2020, pp. 297â302.

[31] J. Li, B. Li, and Y. Lu, âNeural video compression with diverse contexts,â in CVPR, 2023, pp. 22 616â22 626.

[32] R. Yang and S. Mandt, âLossy image compression with conditional diffusion models,â NeurIPS, vol. 36, pp. 64 971â64 995, 2023.

[33] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in CVPR, 2018, pp. 586â595.

[34] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, âGANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,â NeurIPS, vol. 30, 2017.

[35] G. Bjontegaard, âCalculation of average PSNR differences between RD-curves,â ITU SG16 Doc. VCEG-M33, 2001.

[36] Y. Blau and T. Michaeli, âThe perception-distortion tradeoff,â in CVPR, 2018, pp. 6228â6237.

[37] Y. Jiang, C. Feng, F. Zhang, and D. Bull, âMTKD: Multi-Teacher Knowledge Distillation for Image Super-Resolution,â in ECCV. Springer, 2024, pp. 364â382.

[38] Y. Jiang, C. Zeng, S. Teng, F. Zhang, X. Zhu, J. Sole, and D. Bull, âC2D-ISR: Optimizing Attention-based Image Superresolution from Continuous to Discrete Scales,â arXiv preprint arXiv:2503.13740, 2025.