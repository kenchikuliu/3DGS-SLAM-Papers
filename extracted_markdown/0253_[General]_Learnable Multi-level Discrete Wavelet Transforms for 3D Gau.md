# Learnable Multi-level Discrete Wavelet Transforms for 3D Gaussian Splatting Frequency Modulation

Hung Nguyen, An Le, Truong Nguyen

Video Processing Lab, UC San Diego

{hun004, d0le, tqn001}@ucsd.edu

Abstractâ3D Gaussian Splatting (3DGS) has emerged as a powerful approach for novel view synthesis. However, the number of Gaussian primitives often grows substantially during training as finer scene details are reconstructed, leading to increased memory and storage costs. Recent coarse-to-fine strategies regulate Gaussian growth by modulating the frequency content of the ground-truth images. In particular, AutoOpti3DGS employs the learnable Discrete Wavelet Transform (DWT) to enable data-adaptive frequency modulation. Nevertheless, its modulation depth is limited by the 1-level DWT, and jointly optimizing wavelet regularization with 3D reconstruction introduces gradient competition that promotes excessive Gaussian densification. In this paper, we propose a multi-level DWT-based frequency modulation framework for 3DGS. By recursively decomposing the low-frequency subband, we construct a deeper curriculum that provides progressively coarser supervision during early training, consistently reducing Gaussian counts. Furthermore, we show that the modulation can be performed using only a single scaling parameter, rather than learning the full 2-tap high-pass filter. Experimental results on standard benchmarks demonstrate that our method further reduces Gaussian counts while maintaining competitive rendering quality.

Index Termsâ3D Gaussian Splatting, Discrete Wavelet Transform, Frequency Modulation, Gaussian Optimization

## I. INTRODUCTION

3DGS [1] has emerged as a dominant method for 3D scene reconstruction from multi-view 2D images, enabling highquality novel view synthesis with efficient training. Owing to its strong rendering fidelity and computational advantages, it has been adopted across a wide array of domains [2]â[4].

By representing a scene as a collection of anisotropic Gaussian primitives, 3DGS significantly accelerates optimization and improves visual quality compared to its predecessor, NeRF [5]. Nevertheless, the number of Gaussians is not directly constrained during training and often increases substantially as the model captures finer scene details. Controlling this growth is therefore critical to reducing GPU memory consumption and storage requirements, particularly for deployment on resourcelimited or edge platforms [6], [7]. A more compact Gaussian representation also benefits downstream tasks that depend on per-Gaussian embeddings [8]â[11].

Recently, multiple efforts have been aimed at optimizing Gaussian counts. In this work, we focus on coarse-tofine strategies, as they are prior-free and explicitly regulate structural growth by modulating the frequency content of the ground-truth images. By delaying the introduction of fine details, such methods reduce premature over-densification in homogeneous regions, reconstructing high-frequency (HF) structures only when introduced, thereby reducing unnecessary Gaussian proliferation. Opti3DGS [12] proposes blurring the input images with progressively smaller filter sizes as training progresses. However, this approach relies on manually predefined blur types and schedules, which are not dataset-adaptive and may limit rendering quality. AutoOpti3DGS [13] instead performs frequency modulation using the Discrete Wavelet Transform (DWT). The high-pass analysis filter is initialized to zero, producing an initially low-frequency reconstruction from the Inverse Transform. The reconstruction is gradually restored through a wavelet regularization objective that encourages approximate Perfect Reconstruction [14]. This enables data-adaptive coarse-to-fine modulation and reduces Gaussian counts while maintaining rendering fidelity. Nevertheless, the method is restricted to a 1-level DWT, limiting the depth of the modulation curriculum since substantial HF content remains in the ground-truth images. Moreover, the wavelet regularization is optimized jointly with the 3DGS reconstruction loss in different domains, potentially introducing competing gradients that might trigger excessive Gaussian densification.

In this paper, we build upon AutoOpti3DGS with two key extensions. Firstly, we generalize the 1-level DWT to multilevel, enabling a deeper frequency modulation curriculum. By recursively decomposing the LL subband, the initial training images become progressively coarser, leading to stronger early-stage low-frequency emphasis. This consistently results in further reductions in Gaussian counts. Secondly, to mitigate conflicting gradients between the wavelet regularization and the reconstruction objective, we reduce the number of learnable wavelet parameters. Specifically, for the 2-tap case, we show that learning only a scaling parameter for the highpass filter is sufficient. This removes one degree of freedom from the optimization, alleviating gradient competition across domains. As a result, the model can focus more on stable reconstruction while preserving the desired frequency modulation effect. In summary, the contributions are as follows:

â¢ We extend AutoOpti3DGS from 1-level to multi-level DWT, constructing a deeper frequency modulation curriculum that produces coarser initial supervision and consistently reduces Gaussian counts.

â¢ We show that the modulation can be performed using only a single scaling parameter, rather than learning the full 2-tap high-pass filter.

â¢ The proposed method demonstrates to further reduce Gaussian counts, while maintaining rendering quality across benchmark datasets.

## II. RELATED WORKS

Discrete Wavelet Transform (DWT) for Differentiable Novel View Synthesis (NVS). Among recent differentiable NVS approaches, NeRF [5] and 3DGS [1] have emerged as two dominant paradigms. Within NeRF-based frameworks, Rho et al. [15] apply wavelet-based masking to learn a compact representation. WaveNeRF [16] and TriNeRFLet [17] incorporate the multi-level DWT to enhance high-frequency (HF) detail preservation. DWTNeRF [18] performs waveletdomain supervision to reduce HF overfitting in sparse-view settings [19]. Within 3DGS-based frameworks, MW-GS [20] and Wavelet-GS [21] applies the DWT to simultaneously model coarse and fine details. DWTGS [22] performs lowfrequency-centric wavelet-domain supervision to suppress HF hallucinations under sparse views. Conversely, under sufficient views, 3D-GSW [23] utilizes HF-centric wavelet loss to improve detail fidelity. WaveletGaussian [24] performs wavelet-domain diffusion to efficiently generate pseudo samples for 3DGS-based object reconstruction. AutoOpti3DGS [13] applies the learnable DWT for coarse-to-fine frequency modulation of 3DGS input images, achieving Gaussian count optimization while maintaining reasonable rendering quality. In this work, we extend it with the multi-level DWT for a prolonged modulation curriculum, as well as additional analyses on filter choices.

Learnable DWT. Multiple efforts have aimed at learning the optimal, data-adaptive DWT filters. MLWNet [25] employs the learnable DWT to learn task-adaptive representations for motion deblurring. Subsequent works replace CNN pooling layers with the learnable DWT to achieve lossless and adaptive feature extraction. uWu [26] optimizes the wavelets with a relaxed Perfect Reconstruction condition [14] objective, after initializing from orthogonal versions. Later extensions [27]â [32] adopt orthogonal lattice and biorthogonal wavelet structures, as well as lifting schemes, to reduce filter complexity or enhance design flexibility. Generally, these methods focus on representation learning. Differently, our work and the predecessor AutoOpti3DGS [13] employ the learnable DWT as a differentiable image modulator, encouraging 3DGS to learn an efficient 3D scene representation.

## III. PRELIMINARY BACKGROUND

## A. 3D Gaussian Splatting

From a set of multi-view 2D images of a scene, 3DGS [1] reconstructs the scene in 3D by modeling it as a collection of anisotropic 3D Gaussians. Once the scene representation is learned, novel views can be synthesized by rendering from arbitrary viewpoints. Each Gaussian is characterized by learnable parameters: center $\mu ,$ opacity Ï, covariance matrix Î£, and color c.

To optimize these parameters, the following differentiable objective is employed:

$$
\mathcal { L } _ { \mathrm { 3 D G S } } = ( 1 - \lambda ) \mathcal { L } _ { 1 } ( \mathbf { X } ^ { \mathrm { g t } } , \mathbf { X } ) + \lambda \mathcal { L } _ { \mathrm { D \_ S S I M } } ( \mathbf { X } ^ { \mathrm { g t } } , \mathbf { X } )\tag{1}
$$

where $\mathbf { X } ^ { \mathrm { g t } }$ and X denote the ground-truth and rendered images from the same camera viewpoint, respectively. $\mathcal { L } _ { 1 }$ corresponds to the pixel-wise mean absolute error, while $\mathcal { L } _ { \mathrm { D } }$ SSIM captures perceptual similarity. The parameter Î» controls the trade-off between the two loss terms.

## B. Discrete Wavelet Transforms (DWT)

<!-- image-->

<!-- image-->  
e)  
f  
Fig. 1. Illustration of the 2D DWT operations. (a) Original image. (b) 1- level DWT subbands. (c) 2-level DWT obtained by further decomposing the 1-level LL subband. (d, e) Enlarged 1- and 2-level LL subbands, respectively. The higher-level LL subbands are coarser. (f) Reconstructed image using PRsatisfying wavelets.

The 1D Forward DWT applies a pair of analysis filters, a low-pass filter â and a high-pass filter $h ,$ to a 1D signal $\mathbf { x } ,$ resulting in the approximation coefficient $\mathbf { x } _ { L }$ and detail coefficient $\mathbf { x } _ { H } .$

$$
\mathbf { x } _ { \mathrm { L } } = ( \mathbf { x } * \boldsymbol { \ell } ) \downarrow _ { 2 } , \quad \mathbf { x } _ { \mathrm { H } } = ( \mathbf { x } * h ) \downarrow _ { 2 }\tag{2}
$$

where â denotes the convolution operator and $\downarrow _ { 2 }$ denotes downsampling by a factor of 2.

Given the coefficients, the 1D Inverse DWT provides the reconstructed signal xË:

$$
\hat { \mathbf { x } } = \left( \uparrow _ { 2 } \mathbf { x } _ { \mathrm { L } } \right) * \tilde { \ell } + \left( \uparrow _ { 2 } \mathbf { x } _ { \mathrm { H } } \right) * \tilde { h } .\tag{3}
$$

where $\uparrow _ { 2 }$ denotes upsampling by 2, and $\tilde { \ell }$ and $\tilde { h }$ are the lowpass and high-pass synthesis filters, respectively.

<!-- image-->  
Fig. 2. Overview of our framework. The multi-level DWT is employed as a differentiable image modulator. We freeze the original Haar filters and introduce a scaling parameter Î± on the high-pass analysis filters. When Î± = 0, all HF subbands vanish, yielding a coarse IDWT reconstruction, to be used as ground-truths for early-stage 3DGS. A PR-enforcing loss regularizes Î±, progressively restoring high frequencies for automatic coarse-to-fine modulation.

When the analysis and synthesis filters satisfy the Perfect Reconstruction (PR) conditions [14], the original signal is exactly recovered, i.e., xË = x holds. Figure 2 shows the coefficients of the Haar, the simplest PR-satisfying wavelet. Further details on the PR property are provided in Section IV.

Given a 2D image X, the 2D extension of the 1D Forward DWT yields four analysis filters. These filters can be denoted as $\mathbf { K } _ { \mathrm { L L } } , \mathbf { K } _ { \mathrm { L H } } , \mathbf { K } _ { \mathrm { H L } }$ , and ${ \bf K } _ { \mathrm { H H } }$ , each of which is constructed as the outer product of the associated 1D filters:

$$
\begin{array} { r } { \mathbf { K } _ { \mathrm { L L } } = \ell \otimes \ell , \quad \mathbf { K } _ { \mathrm { L H } } = \ell \otimes h , } \\ { \mathbf { K } _ { \mathrm { H L } } = h \otimes \ell , \quad \mathbf { K } _ { \mathrm { H H } } = h \otimes h . } \end{array}\tag{4}
$$

Using these filters, the four âsubbandsâ (XLL, XLH, XHL and XHH) are obtained through convolution followed by downsampling, analogous to the 1D case. In deep learning frameworks, this operation can be implemented as a convolution with stride 2 [25]. The LL subband captures the coarse structural content of the image. The LH and HL subbands emphasize horizontal and vertical details, respectively. The HH subband primarily represents diagonal details.

The 2D Inverse DWT also follows analogously to the 1D case. The 2D synthesis filters are constructed from the 1D synthesis filters Ëâ and hË via outer products, similar to Equation (4). Starting from the subbands, the reconstructed image XË is also obtained through upsampling, followed by convolution with said filters.

The multi-level DWT is obtained by recursively applying the Forward DWT to the LL subband of the previous level, yielding a hierarchical multi-resolution representation. We visualize the DWT operations in Fig. 1.

## IV. METHODOLOGY

Figure 2 illustrates our multi-level DWT-based framework for frequency modulation. Firstly, the input image is decomposed by the Forward DWT using the 2-tap Haar wavelet. The resulting 1-level LL subband is decomposed again. Noticeably, we freeze all wavelet filters and introduce a learnable scaling parameter Î± that modulates the high-pass filters h. Since Î± is initialized to zero, all high-frequency (HF) subbands are zero-valued initially (blacked out in Figure 2). The Inverse DWT is applied recursively to reconstruct the original image, which is highly coarse due to the absence of HF components. The resulting image is fed to 3DGS training and optimized with $\mathcal { L } _ { \mathrm { 3 D G S } }$ , as introduced in Section III-A. This represents the coarse stage of the modulation curriculum.

Another objective is required to optimize the wavelet towards PR, thus retrieving the full-spectrum image. For a 2- channel filter bank, the PR requirement first imposes the âAlias Cancellationâ condition [14]:

$$
\tilde { L } ( z ) H ( - z ) + \tilde { H } ( z ) L ( - z ) = 0\tag{5}
$$

where $L ( z ) , H ( z )$ and $\tilde { L } ( z ) , \tilde { H } ( z )$ denote the z-transforms of the 1D analysis and synthesis filters, respectively. Additionally, the âNo Distortionâ condition is required:

$$
\tilde { L } ( z ) L ( z ) + \tilde { H } ( z ) H ( z ) = 2\tag{6}
$$

We convert these conditions into residual-based losses:

$$
\mathcal { L } _ { \mathrm { a l i a s } } = \Big \| \tilde { L } ( z ) \alpha H ( - z ) + \tilde { H } ( z ) L ( - z ) \Big \| _ { 2 } ^ { 2 }\tag{7}
$$

and

$$
\mathcal { L } _ { \mathrm { d i s t } } = \Big \| \tilde { L } ( z ) L ( z ) + \tilde { H } ( z ) \alpha H ( z ) - 2 \Big \| _ { 2 } ^ { 2 }\tag{8}
$$

where only the scaling parameter Î±, which modulates $H ( z )$ is learnable and initialized to zero. The overall wavelet regularization objective is defined as:

$$
{ \mathcal { L } } _ { \mathrm { P R } } = { \mathcal { L } } _ { \mathrm { a l i a s } } + { \mathcal { L } } _ { \mathrm { d i s t } }\tag{9}
$$

and the total training objective is as follows:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { 3 D G S } } + \lambda _ { \mathrm { P R } } \mathcal { L } _ { \mathrm { P R } }\tag{10}
$$

where $\lambda _ { \mathrm { P R } }$ is the balancing term between the two objectives.

## V. EXPERIMENTS

## A. Dataset & Implementation Details

Our framework is evaluated on the LLFF dataset [33] with 3 input views and on the Mip-NeRF 360 dataset [34] with 12 input views. These benchmarks consist of multiple multi-view images capturing various objects and scenes. The novel view synthesis capability is assessed using held-out test images.

Rendering quality is measured using PSNR, SSIM, and LPIPS [35]. To quantify Gaussian optimization, we record the peak number of Gaussians during densification stages and average this value across all scenes in each dataset. We also report the average training time per scene.

Comparisons are conducted against Vanilla 3DGS [1], Opti3DGS [12] and AutoOpti3DGS [13], with all methods trained for 10K iterations under identical hyperparameters for fair comparison. We re-adopted AutoOpti3DGS but implemented the DWT via stride-2 convolutions following [25], enabling PyTorchâs autograd to directly optimize the 1D filter parameters. The original implementation realizes the DWT via Toeplitz-structured matrix multiplication following [36], requires hand-writing custom backpropagation equations, and optimizes the entire 2D filter. Under this re-implementation, AutoOpti3DGS becomes a special case of our framework that leverages 1-level DWT only, and that optimizes the full highpass analysis filter h. For our extension, we only learn the scaling parameter Î±, and set its learning rate to 1e-4. The balancing weight $\lambda _ { \mathrm { P R } }$ is 0.05.

<!-- image-->  
Fig. 3. Ablation results on DWT levels and scaling parameter effects (3-view LLFF [33] dataset).

## B. Quantitative Results

We provide the quantitative results at Tables I and II. Generally, compared to the 3DGS baseline, our multi-level strategy proves to reduce Gaussian counts significantly, by â¼50K and â¼100K Gaussians for the LLFF and Mip-NeRF 360 dataset, respectively. Compared to the closest efficient baseline, 1- level DWT-based AutoOpti3DGS, our method further reduces â¼30K Gaussians due to the expanded frequency modulation curriculum. Our method also incurs some additional training time, â¼10-20 seconds, because the multi-level DWT requires more computation than the 1-level version. We plan to tackle this in future works using lazy regularization, where $\mathcal { L } _ { \mathrm { P R } }$ is evaluated only every few iterations.

<table><tr><td></td><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS ()</td><td>#G (â)</td><td>Time ()(s)</td></tr><tr><td>Opti3DGS [12]</td><td>19.59</td><td>0.660</td><td>0.228</td><td>247K</td><td>105</td></tr><tr><td>AutoOpti3DGS [13]</td><td>20.29</td><td>0.703</td><td>0.200</td><td>249K</td><td>131</td></tr><tr><td>Ours</td><td>20.34</td><td>0.687</td><td>0.222</td><td>218K</td><td>142</td></tr><tr><td>3DGS [1]</td><td>20.40</td><td>0.706</td><td>0.197</td><td>272K</td><td>109</td></tr></table>

TABLE I

QUANTITATIVE RESULTS, 3-VIEW LLFF [33] DATASET
<table><tr><td></td><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS (â)</td><td>#G (â)</td><td>Time () (s)</td></tr><tr><td>Opti3DGS [12]</td><td>19.19</td><td>0.552</td><td>0.360</td><td>636K</td><td>151</td></tr><tr><td>AutoOpti3DGS [13]</td><td>19.24</td><td>0.541</td><td>0.381</td><td>615K</td><td>182</td></tr><tr><td>Ours</td><td>19.29</td><td>0.560</td><td>0.355</td><td>589K</td><td>200</td></tr><tr><td>3DGS [1]</td><td>19.30</td><td>0.564</td><td>0.352</td><td>701K</td><td>155</td></tr></table>

TABLE II  
QUANTITATIVE RESULTS, 12-VIEW MIP-NERF 360 [34] DATASET

## C. Ablation Study

Figure 3 presents the ablation results. The solid lines represent learning both filter coefficients, similar to AutoOpti3DGS (âwholeâ mode in the legend). The dotted ones represent learning the scaling parameter only (âscaleâ mode). The latter consistently result in lower Gaussian counts, and potentially better PSNR, as seen in the 2-level. We hypothesize this is because learning only the scaling parameter causes less gradient conflicts, so 3DGS can focus on the reconstruction task. Furthermore, increasing DWT levels leads to progressively decreasing Gaussian counts, at the expense of PSNR drops. We hypothesize that deeper levels produce overly coarse initial reconstructions with block-like artifacts that hamper reconstruction, especially as the Haar is used. There is a compromise between DWT levels and rendering quality. The best-performing configuration is marked with the green âXâ symbols.

## VI. CONCLUSION

We propose a learnable multi-level DWT-based frequency modulation framework for 3DGS that consistently reduces Gaussian counts while preserving rendering quality. The modulation can be performed using only a single scaling parameter, rather than learning the full 2-tap high-pass filter. Experiments demonstrate notable Gaussian reductions over Vanilla 3DGS and 1-level DWT baselines with competitive rendering performance.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[2] R. B. Li, M. Shaghaghi, K. Suzuki, X. Liu, V. Moparthi, B. Du, W. Curtis, M. Renschler, K. M. B. Lee, N. Atanasov, and T. Nguyen, âDynagslam: Real-time gaussian-splatting slam for online rendering, tracking, motion predictions of moving objects in dynamic scenes,â 2025. [Online]. Available: https://arxiv.org/abs/2503.11979

[3] R. B. Li, K. Suzuki, B. Du, K. M. B. Lee, N. Atanasov, and T. Nguyen, âSplatsdf: Boosting neural implicit sdf via gaussian splatting fusion,â 2024. [Online]. Available: https://arxiv.org/abs/2411.15468

[4] R. Li, U. Mahbub, V. Bhaskaran, and T. Nguyen, âMonoselfrecon: Purely self-supervised explicit generalizable 3d reconstruction of indoor scenes from monocular rgb views,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2024, pp. 656â666.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â in ECCV, 2020.

[6] S. Niedermayr, J. Stumpfegger, and R. Westermann, âCompressed 3d gaussian splatting for accelerated novel view synthesis,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 10 349â10 358.

[7] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and G. Drettakis, âReducing the memory footprint of 3d gaussian splatting,â Proceedings of the ACM on Computer Graphics and Interactive Techniques, vol. 7, no. 1, p. 1â17, May 2024. [Online]. Available: http://dx.doi.org/10.1145/3651282

[8] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister, âLangsplat: 3d language gaussian splatting,â 2024. [Online]. Available: https://arxiv.org/abs/2312.16084

[9] J.-C. Shi, M. Wang, H.-B. Duan, and S.-H. Guan, âLanguage embedded 3d gaussians for open-vocabulary scene understanding,â arXiv preprint arXiv:2311.18482, 2023.

[10] J. Kulhanek, S. Peng, Z. Kukelova, M. Pollefeys, and T. Sattler, âWildGaussians: 3D gaussian splatting in the wild,â NeurIPS, 2024.

[11] J. Bae, S. Kim, Y. Yun, H. Lee, G. Bang, and Y. Uh, âPer-gaussian embedding-based deformation for deformable 3d gaussian splatting,â in European Conference on Computer Vision (ECCV), 2024.

[12] U. Farooq, J.-Y. Guillemaut, G. Thomas, A. Hilton, and M. Volino, âOptimized 3d gaussian splatting using coarse-to-fine image frequency modulation,â in Proceedings of the 22nd ACM SIGGRAPH European Conference on Visual Media Production, ser. CVMP â25. New York, NY, USA: Association for Computing Machinery, 2025. [Online]. Available: https://doi.org/10.1145/3756863.3769707

[13] H. Nguyen, A. Le, B. R. Li, and T. Nguyen, âFrom coarse to fine: Learnable discrete wavelet transforms for efficient 3d gaussian splatting,â in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, October 2025, pp. 3139â3148.

[14] G. Strang and T. Nguyen, Wavelets and filter banks. SIAM, 1996.

[15] D. Rho, B. Lee, S. Nam, J. C. Lee, J. H. Ko, and E. Park, âMasked wavelet representation for compact neural radiance fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 20 680â20 690.

[16] M. Xu, F. Zhan, J. Zhang, Y. Yu, X. Zhang, C. Theobalt, L. Shao, and S. Lu, âWavenerf: Wavelet-based generalizable neural radiance fields,â 2023. [Online]. Available: https://arxiv.org/abs/2308.04826

[17] R. Khatib and R. Giryes, âTrinerflet: A wavelet based triplane nerf representation,â 2024. [Online]. Available: https://arxiv.org/abs/2401.06191

[18] H. Nguyen, B. R. Li, and T. Nguyen, âDwtnerf: Boosting few-shot neural radiance fields via discrete wavelet transform,â 2025. [Online]. Available: https://arxiv.org/abs/2501.12637

[19] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM Trans. Graph., vol. 41, no. 4, pp. 102:1â102:15, Jul. 2022. [Online]. Available: https://doi.org/10.1145/3528223.3530127

[20] Y. Li, C. Lv, H. Yang, and D. Huang, âMicro-macro wavelet-based gaussian splatting for 3d reconstruction from unconstrained images,â 2025. [Online]. Available: https://arxiv.org/abs/2501.14231

[21] B. Zhao, Y. Zhou, S. Yu, Z. Wang, and H. Wang, âWaveletgs: 3d gaussian splatting with wavelet decomposition,â in Proceedings of the 33rd ACM International Conference on Multimedia, ser. MM â25. New York, NY, USA: Association for Computing Machinery, 2025, p. 8616â8625. [Online]. Available: https://doi.org/10.1145/3746027.3755589

[22] H. Nguyen, R. Li, A. Le, and T. Nguyen, âDwtgs: Rethinking frequency regularization for sparse-view 3d gaussian splatting,â in Proceedings of

the 2025 IEEE International Conference on Visual Communications and Image Processing (VCIP). IEEE, 2025.

[23] Y. Jang, H. Park, F. Yang, H. Ko, E. Choo, and S. Kim, â3d-gsw: 3d gaussian splatting watermark for protecting copyrights in radiance fields,â arXiv preprint arXiv:2409.13222, 2024.

[24] H. Nguyen, R. Li, A. Le, and T. Nguyen, âWaveletgaussian: Waveletdomain diffusion for sparse-view 3d gaussian object reconstruction,â in Proceedings of the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026.

[25] X. Gao, T. Qiu, X. Zhang, H. Bai, K. Liu, X. Huang, H. Wei, G. Zhang, and H. Liu, âEfficient multi-scale network with learnable discrete wavelet transform for blind motion deblurring,â 2024. [Online]. Available: https://arxiv.org/abs/2401.00027

[26] A. D. Le, S. Jin, Y. S. Bae, and T. Nguyen, âA novel learnable orthogonal wavelet unit neural network with perfection reconstruction constraint relaxation for image classification,â in 2023 IEEE International Conference on Visual Communications and Image Processing (VCIP), 2023, pp. 1â5.

[27] A. D. Le, S. Jin, Y.-S. Bae, and T. Q. Nguyen, âA lattice-structure-based trainable orthogonal wavelet unit for image classification,â IEEE Access, vol. 12, pp. 88 715â88 727, 2024.

[28] A. D. Le, S. Jin, S. Seo, Y.-S. Bae, and T. Q. Nguyen, âBiorthogonal lattice tunable wavelet units and their implementation in convolutional neural networks for computer vision problems,â IEEE Open Journal of Signal Processing, pp. 1â16, 2025.

[29] A. Le, H. Nguyen, S. Seo, Y.-S. Bae, and T. Nguyen, âBiorthogonal tunable wavelet unit with lifting scheme in convolutional neural network,â in Proceedings of the 33rd European Signal Processing Conference (EUSIPCO), 2025.

[30] A. D. Le, H. Nguyen, S. Seo, Y.-S. Bae, and T. Q. Nguyen, âStop-band energy constraint for orthogonal tunable wavelet units in convolutional neural networks for computer vision problems,â in Proceedings of the 2025 IEEE International Conference on Visual Communications and Image Processing (VCIP). IEEE, 2025.

[31] A. D. Le, H. Nguyen, M. Tran, J. Most, D.-U. G. Bartsch, W. R. Freeman, S. Borooah, T. Q. Nguyen, and C. An, âUniversal wavelet units in 3d retinal layer segmentation,â 2025. [Online]. Available: https://arxiv.org/abs/2507.16119

[32] A. Le, N. Mehta, W. Freeman, I. Nagel, M. Tran, A. Heinke, A. Agnihotri, L. Cheng, D.-U. Bartsch, H. Nguyen, T. Nguyen, and C. An, âTunable wavelet unit based convolutional neural network in optical coherence tomography analysis enhancement for classifying type of epiretinal membrane surgery,â in Proceedings of the 33rd European Signal Processing Conference (EUSIPCO), 2025.

[33] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ramamoorthi, R. Ng, and A. Kar, âLocal light field fusion: Practical view synthesis with prescriptive sampling guidelines,â ACM Transactions on Graphics (TOG), 2019.

[34] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â CVPR, 2022.

[35] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in CVPR, 2018.

[36] Q. Li, L. Shen, S. Guo, and Z. Lai, âWavecnet: Wavelet integrated cnns to suppress aliasing effect for noise-robust image classification,â IEEE Transactions on Image Processing, vol. 30, pp. 7074â7089, 2021.