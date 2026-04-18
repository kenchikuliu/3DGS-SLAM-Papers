# WAVELETGAUSSIAN: WAVELET-DOMAIN DIFFUSION FOR SPARSE-VIEW 3D GAUSSIAN OBJECT RECONSTRUCTION

Hung Nguyen, Runfa Li, An Le, Truong Nguyen

Video Processing Lab, UC San Diego

## ABSTRACT

3D Gaussian Splatting (3DGS) has become a powerful representation for image-based object reconstruction, yet its performance drops sharply in sparse-view settings. Prior works address this limitation by employing diffusion models to repair corrupted renders, subsequently using them as pseudo references for later optimization. While effective, such approaches incur heavy computation from the diffusion fine-tuning and repair steps. We present WaveletGaussian, a framework for more efficient sparse-view 3D Gaussian object reconstruction. Our key idea is to shift diffusion into the wavelet domain: diffusion is applied only to the lowresolution LL subband, while high-frequency subbands are refined with a lightweight network. We further propose an efficient online random masking strategy to curate training pairs for diffusion fine-tuning, replacing the commonly used, but inefficient, leave-one-out strategy. Experiments across two benchmark datasets, Mip-NeRF 360 and OmniObject3D, show WaveletGaussian achieves competitive rendering quality while substantially reducing training time.

Index Termsâ Sparse-view 3DGS, wavelet transform, 3D object reconstruction, diffusion model, neural rendering.

## 1. INTRODUCTION

3D Gaussian Splatting (3DGS) [1] has become a leading approach for reconstructing 3D scenes or objects from 2D images, producing photorealistic novel views with relatively short training times. Nevertheless, it generally depends on densely captured training views with accurate camera poses, which demand significant effort in data collection. In scenarios with sparse views, the reconstructed geometry is poorly constrained, often leading to artifacts or unstable structures that severely degrade rendering quality. This limitation reduces its practicality in real-world settings, where acquiring dense, well-posed data is often impractical [2].

Therefore, sparse-view 3DGS has emerged as an active research direction. While multiple kinds of priors have been leveraged for the task [3], denoising diffusion models (DDMs) [4] have emerged as a powerful option due to their outstanding generative capabilities. Within a sparseview 3DGS framework, they are often used to repair the renders from novel viewpoints, which are often highly corrupted due to the lack of explicit supervision. The repaired views are subsequently used as pseudo references for later optimization, thus emulating artifact-free dense-view training [5, 6, 7, 2, 8, 9, 10, 11]. Despite producing high-quality pseudo ground-truths, this approach incurs significant computation due to the required fine-tuning step, which is necessary to adapt a pre-trained diffusion model to the specific scene or object at hand. The repair step is also costly, thus severely hindering the methodâs scalability. To shorten the overall training time, recent works leverage LoRA [12] adapters, but a single scene can still take up to an hour to train [2].

In this paper, we introduce the WaveletGaussian framework for 3D Gaussian object reconstruction under sparse views, aiming to significantly reduce overall training time while maintaining competitive rendering quality. To achieve this objective, WaveletGaussian proposes repositioning the diffusion fine-tuning and repair steps from the RGB to wavelet domain. The rationale is that the latter is only halfresolution, while still preserving all information through the lossless wavelet transform. Specifically, the diffusion model is only trained on, and applied to, the low-frequency LL subband, while the high-frequency subbands are processed using a lightweight U-Net-like [13] architecture. Additionally, we propose a novel online random masking method to curate the object-specific dataset for diffusion fine-tuning, replacing the commonly used, but inefficient, leave-one-out strategy [10, 14, 2]. In summary, our contributions are as follows:

â¢ We propose WaveletGaussian, a 3DGS-based framework for sparse-view object reconstruction with significantly reduced training times due to i) wavelet-domain, diffusion-based novel view repairs, and ii) an efficient method to curate the object-specific dataset for diffusion fine-tuning.

â¢ Through experiments on benchmark datasets, our WaveletGaussian demonstrates to significantly reduce overall training time, while maintain competitive rendering quality.

<!-- image-->  
Fig. 1. We propose WaveletGaussian, a framework for sparse-view 3D Gaussian object reconstruction based on wavelet-domain diffusion model repair, which significantly reduces training time while bettering rendering quality.

## 2. RELATED WORKS

Discrete Wavelet Transform (DWT) for 3DGS. Recently, the DWT has attracted growing attention within deep computer vision frameworks, as it disentangles frequency learning while providing efficiency benefits [3]. Extensions to 3DGS are also being explored, e.g., for fine detail enhancement [15], coarse-to-fine efficient learning [16] and frequency regularization [3]. Our WaveletGaussian novelly introduces the DWT to a sparse-view framework with diffusion-based repairs to improve efficiency.

Diffusion-based repair for sparse-view 3DGS. Denoising diffusion models (DDMs) [4], known for their strong generative capabilities, are widely used to repair the highly corrupted novel views of sparse-view 3DGS. However, this approach incurs significant computation, as it requires finetuning the diffusion model on large-scale datasets [5, 6, 7, 9]. Scene-specific fine-tuning and LoRA adapters [2, 10, 11, 8] improve efficiency, but the total training time may still require up to an hour [2], thus severely limiting the methodâs scalability. Our WaveletGaussian proposes repositioning the diffusion-related processes to the lower-resolution wavelet domain for efficiency benefits.

## 3. METHODOLOGY

## 3.1. Discrete Wavelet Transforms

Given a 2D image X, the Forward DWT decomposes it into four distinct subbands (LL, LH, HL, HH) as follows:

$$
\begin{array} { r } { \mathbf { X } _ { \mathrm { L L } } = \mathbf { L } _ { 0 } \mathbf { X } \mathbf { L } _ { 1 } , \quad \mathbf { X } _ { \mathrm { L H } } = \mathbf { H } _ { 0 } \mathbf { X } \mathbf { L } _ { 1 } , } \\ { \mathbf { X } _ { \mathrm { H L } } = \mathbf { L } _ { 0 } \mathbf { X } \mathbf { H } _ { 1 } , \quad \mathbf { X } _ { \mathrm { H H } } = \mathbf { H } _ { 0 } \mathbf { X } \mathbf { H } _ { 1 } } \end{array}\tag{1}
$$

where $\mathbf { L } _ { ( \cdot ) }$ and $\mathbf { H } _ { ( \cdot ) }$ are the low-pass and high-pass filtering matrices applied to either the columns or rows of X, as indicated by the subscript {0, 1}. As an example, the low-pass, vertically filtering matrix $\mathbf { L } _ { 0 }$ , based on Haar wavelet [18], is:

$$
\mathbf { L } _ { 0 } = \left[ { \begin{array} { c c c c c c } { { \frac { 1 } { \sqrt { 2 } } } } & { { \frac { 1 } { \sqrt { 2 } } } } & { 0 } & { 0 } & { 0 } & { \cdots } \\ { 0 } & { 0 } & { { \frac { 1 } { \sqrt { 2 } } } } & { { \frac { 1 } { \sqrt { 2 } } } } & { 0 } & { \cdots } \\ { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \vdots } & { \ddots } \end{array} } \right]
$$

which is constructed by shifting the low-pass, averagingâ â filter $[ 1 / \sqrt { 2 } , 1 / \sqrt { 2 } ]$ along rows. The shifts imply downsampling (in this case, by 2). The high-pass matrix $\mathbf { H } _ { 0 }$ is constructed similarly, using the high-pass, differencing filterâ â $[ - 1 / \sqrt { 2 } , 1 / \sqrt { 2 } ]$ instead. In Equation (1), the LL subband results from low-pass filtering in both directions, retaining the coarse structure of the image. The LH and HL subbands result from applying a high-pass filter in one direction and a low-pass filter in the other, capturing horizontal and vertical information, respectively. The HH subband, high-pass filtered in both directions, emphasizes fine diagonal textures.

Given the four subbands, the Inverse DWT provides the reconstruction XË as follows:

$$
\begin{array} { r } { \hat { \mathbf { X } } = \tilde { \mathbf { L } } _ { 0 } ^ { \top } \mathbf { X } _ { \mathrm { L L } } \tilde { \mathbf { L } } _ { 1 } ^ { \top } + \tilde { \mathbf { H } } _ { 0 } ^ { \top } \mathbf { X } _ { \mathrm { L H } } \tilde { \mathbf { L } } _ { 1 } ^ { \top } + \tilde { \mathbf { L } } _ { 0 } ^ { \top } \mathbf { X } _ { \mathrm { H L } } \tilde { \mathbf { H } } _ { 1 } ^ { \top } + \tilde { \mathbf { H } } _ { 0 } ^ { \top } \mathbf { X } _ { \mathrm { H H } } \tilde { \mathbf { H } } _ { 1 } ^ { \top } } \end{array}\tag{2}
$$

where the matrices used in the Forward and Inverse DWT are termed âanalysisâ and âsynthes $\mathrm { i } \mathrm { s } ^ { \prime \prime }$ , respectively. The Haar synthesis matrices,â $\tilde { \mathbf { L } } _ { 0 }$ and â $\tilde { \mathbf { H } } _ { 0 }$ , are constructed using the syn-â â thesis filters $[ 1 / \sqrt { 2 } , 1 / \sqrt { 2 } ]$ (low-pass) and $[ 1 / \sqrt { 2 } , - 1 / \sqrt { 2 } ]$ (high-pass). The âPerfect Reconstructionâ condition, which occurs when $\mathbf { X } = { \hat { \mathbf { X } } }$ and implies no loss of information, is satisfied when specific relationships exist between the analysisâsynthesis filter pairs [18].

<!-- image-->  
Fig. 2. The proposed WaveletGaussian framework for sparse-view 3D Gaussian object reconstruction. Central to WaveletGaussian is repositioning of the diffusion model [17] from the RGB to lower-resolution wavelet domain for novel view repairs.

## 3.2. Overall Framework

Figure 2 shows an overview of our proposed WaveletGaussian. Firstly, in the Coarse Training (a) stage, a 3DGS model Gc is trained on all N sparse views for some limited iterations to capture the overall geometry. As the training of Gc is terminated early, the resulting renders, even from known viewpoints, are moderately corrupted. We pass both the reference and coarse renders into the Forward DWT for later uses.

The Dataset Creation (b) stage involves synthesizing corruptedâclean image pairs to fine-tune a pre-trained diffusion model [22] D and a lightweight U-Net-like [13] model $\mathcal { U } .$ The fine-tuning is necessary to adapt them to object-specific details, enabling later repairs of novel views. To simulate corrupted patterns for D, a 3DGS model $\mathcal { G } _ { d }$ is optimized with a masking strategy, to be detailed in Section 3.3. The masked renders are paired with the reference ones, both transformed into the wavelet domain, where we retain only the LL subbands to form the LL-domain diffusion dataset. On the other hand, the corrupted patterns for U are retrieved from the highfrequency (HF) subbands (LH, HL and HH) of the coarse renders of $\mathcal { G } _ { c } ,$ also paired with the corresponding references.

The Diffusion Fine-Tuning (c) stage operates in the lowresolution LL domain. Here, D is essentially trained to be an inpainting model operating in low frequencies (LF), while U repairs the HF. By training separate models for LF/HF repairs, we disentangle frequency learning, allowing each model to specialize in LF/HF. Since both models operate at half resolutions, this remains considerably cheaper than fine-tuning a single RGB-domain D, as will be shown in Section 4.3.

Finally, in the Fine Training (d) stage, the coarse model $\mathcal { G } _ { c }$ is refined into $\mathcal { G } _ { f }$ . During this process, D, which is now frozen, repairs the LL renders of $\mathcal { G } _ { c }$ from novel viewpoints, which are especially corrupted due to the sparse reference views. Similarly, the frozen U repairs the HF subbands. The repaired outputs of both are mapped back to the RGB domain through the Inverse DWT. Figure 3 illustrates the use of the

Inverse DWT to map the repairs back to original resolution.

Alongside actual references, the resulting IDWT reconstructions serve as pseudo references in the fine optimization step, thus emulating artifact-free dense-view supervision.

## 3.3. Random Masking for Efficient Dataset Creation

To simulate corrupted patterns for Dataset Creation, many state-of-the-art methods [2, 10, 14] adopt a leave-one-out (LOO) strategy. This involves training N separate 3DGS models $\mathcal { G } _ { d 1 } , . . . , \mathcal { G } _ { d N }$ , each constructed using all but one of the N sparse reference views. The excluded view serves as the reference, while the render from same viewpoint is the corrupted counterpart. While effective at simulating corrupted patterns, training N separate 3DGS models solely for this purpose is highly inefficient. Therefore, we introduce the online random masking (ORM) strategy. As shown in Figure 2, it only requires training a single $\mathcal { G } _ { d } .$ , which is optimized using the typical pixel-wise rendering loss $\mathcal { L } _ { \mathrm { 3 D G S } }$ [1]. However, the references at index $n \in [ 1 , N ] , \mathbf { X } _ { n } ^ { \mathrm { g t } } ,$ , are randomly masked with a binary mask M. It consists of $n _ { \mathbf { m } }$ 0-valued regions, each denoted as m, to only mask certain regions of $\mathbf { X } _ { n } ^ { \mathrm { g t } }$ . Each region m drifts according to sinusoidal displacements during training to generate diverse corruption patterns for D. M is applied differently to each $\mathbf { X } _ { n } ^ { \mathrm { g t } }$ in the dataset, and simulates lack of coverage while using all N views at a time, thus bypassing the LOO strategy.

## 4. EXPERIMENTS

## 4.1. Datasets & Implementation Details

Datasets & Metrics. The object reconstruction performance of WaveletGaussian is evaluated by measuring the quality of novel view renderings on held-out views of the Mip-NeRF 360 [19] and OmniObject3D [20] datasets, using the PSNR,

Table 1. Quantitative results, 4-view Mip-NeRF 360 [19] and OmniObject3D [20] datasets
<table><tr><td rowspan="2">Method</td><td colspan="4">Mip-NeRF 360 [19]</td><td colspan="4">OmniObject3D [20]</td></tr><tr><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS ()</td><td>Time (mins)</td><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS (â)</td><td>Time (mins)</td></tr><tr><td>3DGS [1]</td><td>20.31</td><td>0.899</td><td>0.108</td><td></td><td>17.29</td><td>0.930</td><td>0.086</td><td></td></tr><tr><td>FSGS [21]</td><td>21.07</td><td>0.910</td><td>0.095</td><td></td><td>24.71</td><td>0.955</td><td>0.063</td><td>â</td></tr><tr><td>GaussianObject [2]</td><td>24.81</td><td>0.935</td><td>0.050</td><td>51</td><td>30.89</td><td>0.976</td><td>0.030</td><td>55</td></tr><tr><td>WaveletGaussian (Ours)</td><td>25.31</td><td>0.939</td><td>0.047</td><td>33</td><td>31.22</td><td>0.983</td><td>0.028</td><td>35</td></tr></table>

Table 2. Ablation studies on the 4-view Mip-NeRF 360 [19] dataset with (â) or without (â) proposed components.
<table><tr><td>Offline RM</td><td>Online RM</td><td>wavelet-D</td><td>U repair</td><td>PSNR (â)</td><td>SSIM (â)</td><td>LPIPS (â)</td><td>Time (mins)</td></tr><tr><td>X</td><td>X</td><td>X</td><td>X</td><td>24.81</td><td>0.935</td><td>0.050</td><td>51</td></tr><tr><td>â</td><td>X</td><td>X</td><td>X</td><td>24.95</td><td>0.934</td><td>0.051</td><td>43</td></tr><tr><td>X</td><td></td><td>X</td><td>X</td><td>25.10</td><td>0.934</td><td>0.051</td><td>43</td></tr><tr><td>X</td><td></td><td>â</td><td>X</td><td>24.99</td><td>0.934</td><td>0.051</td><td>30</td></tr><tr><td>Ã</td><td></td><td></td><td></td><td>25.31</td><td>0.939</td><td>0.047</td><td>33</td></tr></table>

SSIM and LPIPS metrics. Additionally, the end-to-end training time is recorded.

Implementation Details. Our implementation is built upon GaussianObject [2]. Different to ours, it leverages the LOO strategy and RGB-domain D for novel view repairs. Firstly, to replace LOO, we adopt the ORM strategy described at Section 3.3. During training $\mathcal { G } _ { d } .$ , we use a mask M with $n _ { \mathbf { m } } ~ = ~ 1 0$ masking regions, the total area of which covers 50% of the object. Secondly, similar to GaussianObject, we leverage a pre-trained ControlNet [17] for D. All training parameters remain the same, except D is fine-tuned on LF corrupted-clean pairs. The HF-repairing U processes concatenated HF subbands and is terminated based on early stopping to prevent overfitting.

## 4.2. Quantitative Results

We present the quantitative results in Table 1. Generally, compared to the closest baseline, GaussianObject [2], our proposed method achieves a 0.3-0.5 dB increase in PSNR and cuts the overall training time roughly by 40%.

## 4.3. Ablation Studies

Table 2 presents ablation results. Firstly, we replace the LOO strategy, utilized by the baseline, with two variations of the random masking strategy. Different from the Online RM strategy presented in Section 3.3, the Offline RM strategy does not incorporate drifting masks. The former achieves better PSNR because the more diverse corruption patterns make D more robust. Both strategy outperform LOO in training time due to training a single $\mathcal { G } _ { d } .$ , and without performance reductions. Having incorporated ORM, we then use wavelet diffusion (âwavelet-Dâ) for novel view repairs. This further decreases training time, but the PSNR suffers because D only rectifies the coarse LL subbands. Incorporating U to rectify HF subbands leads to the best results, at the cost of some minor additional training time.

<!-- image-->

<!-- image-->  
a

b)  
<!-- image-->  
c)

<!-- image-->

<!-- image-->  
e

d)  
<!-- image-->  
f  
Fig. 3. Pseudo view generation via LL-domain diffusion. Given corrupted LL and LH subbands (upsampled for better visualization) at a) and c), our framework provides the corresponding repairs at b) and d). Through the Inverse DWT, we generate a pseudo-sample at e). This bypasses RGB-domain diffusion at f), while providing comparable results.

## 5. CONCLUSION

We introduce WaveletGaussian, a sparse-view 3D Gaussian object reconstruction framework that leverages a waveletdomain diffusion model for novel view repairs. The switch from RGB to lower-resolution wavelet domain significantly reduces overall training time, while enabling frequencyseparated repairs without performance degradation, as supported by experimental results.

Compliance with Ethical Standards. This is a numerical simulation study for which no ethical approval was required.

Acknowledgements. The first author was supported by the Vingroup Science and Technology Scholarship Program for Overseas Study for Masterâs and Doctoral Degrees.

## 6. REFERENCES

[1] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis,Â¨ â3d gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, July 2023.

[2] Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian, âGaussianobject: High-quality 3d object reconstruction from four views with gaussian splatting,â ACM Transactions on Graphics, 2024.

[3] Hung Nguyen, Runfa Li, An Le, and Truong Nguyen, âDwtgs: Rethinking frequency regularization for sparse-view 3d gaussian splatting,â 2025.

[4] Jonathan Ho, Ajay Jain, and Pieter Abbeel, âDenoising diffusion probabilistic models,â in Proceedings of the 34th International Conference on Neural Information Processing Systems, 2020.

[5] Xinhang Liu, Jiaben Chen, Shiu hong Kao, Yu-Wing Tai, and Chi-Keung Tang, âDeceptivenerf/3dgs: Diffusion-generated pseudo-observations for high-quality sparse-view reconstruction,â 2024.

[6] Xi Liu, Chaoyi Zhou, and Siyu Huang, â3dgs-enhancer: Enhancing unbounded 3d gaussian splatting with viewconsistent 2d diffusion priors,â in Advances in Neural Information Processing Systems (NeurIPS), 2024.

[7] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling, âDifix3d+: Improving 3d reconstructions with single-step diffusion models,â in CVPR, 2025.

[8] Chong Bao, Xiyu Zhang, Zehao Yu, Jiale Shi, Guofeng Zhang, Songyou Peng, and Zhaopeng Cui, âFree360: Layered gaussian splatting for unbounded 360-degree view synthesis from extremely sparse and unposed views,â in CVPR, 2025.

[9] Sibo Wu, Congrong Xu, Binbin Huang, Geiger Andreas, and Anpei Chen, âGenfusion: Closing the loop between reconstruction and generation via videos,â in Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[10] Avinash Paliwal, Xilong Zhou, Wei Ye, Jinhui Xiong, Rakesh Ranjan, and Nima Khademi Kalantari, âRi3d: Few-shot gaussian splatting with repair and inpainting diffusion priors,â 2025.

[11] Hanyang Kong, Xingyi Yang, and Xinchao Wang, âGenerative sparse-view gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, âLoRA: Low-rank adaptation of large language models,â in International Conference on Learning Representations, 2022.

[13] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, âU-net: Convolutional networks for biomedical image segmentation,â 2015.

[14] Yutian Chen, Shi Guo, Tianshuo Yang, Lihe Ding, Xiuyuan Yu, Jinwei Gu, and Tianfan Xue, â4dslomo: 4d reconstruction for high speed scene with asynchronous capture,â in Proceedings of the ACM SIGGRAPH Asia 2025 Conference.

[15] Youngdong Jang, Hyunje Park, Feng Yang, Heeju Ko, Euijin Choo, and Sangpil Kim, â3d-gsw: 3d gaussian splatting for robust watermarking,â 2025.

[16] Hung Nguyen, An Le, Runfa Li, and Truong Nguyen, âFrom coarse to fine: Learnable discrete wavelet transforms for efficient 3d gaussian splatting,â 2025.

[17] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala, âAdding conditional control to text-to-image diffusion models,â 2023.

[18] Gilbert Strang and Truong Nguyen, Wavelets and filter banks, SIAM, 1996.

[19] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â CVPR, 2022.

[20] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, and Ziwei Liu, âOmniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation,â 2023.

[21] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang, âFsgs: Real-time few-shot view synthesis using gaussian splatting,â 2023.

[22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer, âHigh-resolution im- Â¨ age synthesis with latent diffusion models,â 2021.