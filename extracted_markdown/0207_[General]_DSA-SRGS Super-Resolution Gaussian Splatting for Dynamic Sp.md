# DSA-SRGS: Super-Resolution Gaussian Splatting for Dynamic Sparse-View DSA Reconstruction

Shiyu Zhang1â, Zhicong Wu2â, Huangxuan Zhao1â , Zhentao Liu3, Lei Chen4, Yong Luo1, Lefei Zhang1, Zhiming Cui3, Ziwen Ke3â , and Bo Du1â 

2 Institute of Artificial Intelligence, Xiamen University, Xiamen, China 3 School of Biomedical Engineering & State Key Laboratory of Advanced Medical

Materials and Devices, ShanghaiTech University, Shanghai, China 4 Union Hospital, Tongji Medical College, Huazhong University of Science and Technology, Wuhan, China

Abstract. Digital subtraction angiography (DSA) is a key imaging technique for the auxiliary diagnosis and treatment of cerebrovascular diseases. Recent advancements in gaussian splatting and dynamic neural representations have enabled robust 3D vessel reconstruction from sparse dynamic inputs. However, these methods are fundamentally constrained by the resolution of input projections, where performing naive upsampling to enhance rendering resolution inevitably results in severe blurring and aliasing artifacts. Such lack of super-resolution capability prevents the reconstructed 4D models from recovering fine-grained vascular details and intricate branching structures, which restricts their application in precision diagnosis and treatment. To solve this problem, this paper proposes DSA-SRGS, the first super-resolution gaussian splatting framework for dynamic sparse-view DSA reconstruction. Specifically, we introduce a Multi-Fidelity Texture Learning Module that integrates high-quality priors from a fine-tuned DSA-specific super-resolution model, into the 4D reconstruction optimization. To mitigate potential hallucination artifacts from pseudo-labels, this module employs a Confidence-Aware Strategy to adaptively weight supervision signals between the original lowresolution projections and the generated high-resolution pseudo-labels. Furthermore, we develop Radiative Sub-Pixel Densification, an adaptive strategy that leverages gradient accumulation from high-resolution sub-pixel sampling to refine the 4D radiative gaussian kernels. Extensive experiments on two clinical DSA datasets demonstrate that DSA-SRGS significantly outperforms state-of-the-art methods in both quantitative metrics and qualitative visual fidelity.

Keywords: Sparse-View DSA Reconstruction Â· Gaussian Splatting Â· Super Resolution.

<!-- image-->  
Fig. 1: Overview of traditional post-upsampling paradigm vs. our DSA-SRGS.

## 1 Introduction

Digital subtraction angiography (DSA) [10] clearly presents vascular dynamics with high spatiotemporal resolution two-dimensional imaging, providing a key basis for the diagnosis and treatment of lesions [18] [26] [2]. The clear imaging of traditional DSA relies on dense projection acquisition, resulting in high radiation dose and easy introduction of motion artifacts. Therefore, achieving high-quality 4D DSA reconstruction under sparse-view conditions has become an important research direction in this field [36] [35] [30] [29].

Although some studies have successfully applied 3D gaussian splatting [11] to the field of medical imaging [31] and introduced dynamic modeling and cumulative pruning strategies for DSA time series [32] [19], the reconstruction quality is still limited by the resolution of the input projection. Under low-resolution conditions, the rendering results are prone to problems such as edge blurring, aliasing artifacts and noise sensitivity, which seriously restricts its clinical application in precision diagnosis and treatment [6] [22] [24] [12] [23].

It is worth noting that the mainstream computer vision method for solving the problem of limited image resolution is image super-resolution (SR) technology. SR research mainly focuses on network architecture innovation, covering CNN methods [13] sSRCNN [5], EDSR [16], and RCAN [33]. The performance has been continuously improved by Transformer models [25] such as SwinIR [15], DAT [4], Mambair [9]and HAT [3], as well as GAN methods [8] like Real-ESRGAN [27]. In response to the demand for lightweighting, CATANet [17] achieves efficient and long-range information interaction through content-aware token aggregation, and has achieved leading performance in lightweight superresolution tasks. While recent frontiers like SRGS [7] have begun integrating SR priors into Gaussian splatting, their efficacy remains limited in medical contexts. Due to the significant domain gap, these general-purpose models often introduce "hallucination artifacts" that compromise the physical fidelity and clinical reliability of vascular structures.

<!-- image-->  
Fig. 2: Architecture of the proposed DSA-SRGS.

To address resolution bottlenecks, this paper proposes DSA-SRGS, a superresolution gaussian splatting framework for dynamic sparse-view DSA reconstruction that unifies super-resolution learning and 4D dynamic reconstruction in end-to-end optimization, as shown in Fig. 1. Specifically, our method first uses a fine-tuned super-resolution model to perform super-resolution on the lowresolution input image to restore the detailed information at the original resolution. Subsequently, a multi-fidelity learning is introduced, which absorbs the high-frequency texture information provided by the super-resolution model while maintaining the structural authenticity consistent with the original observations. To suppress the illusory textures that may occur in the super-resolution model, we further design a confidence-aware fusion mechanism to adaptively weight the multi-source supervised signals. During the rendering stage, we propose a radiation sub-pixel densification strategy. By accumulating gradients during the sampling process of high-resolution sub-pixels, we guide the gaussian kernel to adaptively split in texture-rich regions, thereby enhancing the modelâs ability to restore fine structures.

In summary, the main contributions of this article include:

1. We introduce DSA-SRGS, the first super-resolution gaussian splatting framework for dynamic sparse-view DSA reconstruction.

2. We propose the Multi-fidelity Texture Learning Module and Radiative Subpixel Densification Strategy to enhance micro-vascular modeling.

3. Experiments on two clinical DSA datasets have shown that DSA-SRGS outperforms the current SOTA method.

## 2 Method

## 2.1 Problem Definition

Given a low-resolution projection input $\mathcal { T } ^ { L R } = \{ I _ { t , v } ^ { L R } \in \mathbb { R } ^ { \frac { H } { 4 } \times \frac { W } { 4 } } \ | \ t = 1 , \dots , T , v =$ $1 , \ldots , V \}$ of a dynamic DSA sequence, where $T$ and V denote the total time frames and sparse viewpoints, respectively. For each view v, we construct the projection matrix $P _ { v } = K _ { v } [ R _ { v } \ | \ t _ { v } ]$ based on its corresponding collected geometric parameters. DSA-SRGS aims to directly reconstruct a high-fidelity fourdimensional vascular model from these sparsely sampled low-resolution dynamic projections, and render high-resolution DSA images $I ^ { H R } \in \mathbb { R } ^ { H \times W }$ from arbitrary viewpoints and at arbitrary time points.

## 2.2 Radiative Sub-pixel Densification

We represent vascular segments as gaussian kernels $\mathcal { G } = \{ g _ { i } \} _ { i = 1 } ^ { N }$ , while introducing a time-dependent central attenuation function to capture the dynamic evolution of contrast agent concentration. To predict this attenuation value, we design a Dynamic Neural Attenuation Field (DNAF):

$$
\rho _ { i } ( t ) = \varPhi \left( H _ { 3 D } ( \pmb { \mu _ { i } } ) \oplus H _ { 4 D } ( \pmb { \mu _ { i } } , t ) \right) ,\tag{1}
$$

where $\mu _ { i }$ denotes the position of the gaussian kernel, $H _ { 3 D } ( \cdot )$ [20] and $H _ { 4 D } ( \cdot , \cdot )$ [21] extract static scene features and spatio-temporal dynamic features respectively, and $\varPhi ( \cdot )$ is an MLP that maps the features to the attenuation value $\rho _ { i } ( t )$

For rendering, We employ X-ray rasterization pipelines and cumulative attenuation pruning strategies:

$$
\bar { \rho } _ { i } = \frac { 1 } { N _ { \mathrm { i t e r } } } \sum _ { k = 1 } ^ { N _ { \mathrm { i t e r } } } \rho _ { i } ( t _ { k } ) ,\tag{2}
$$

where $N _ { \mathrm { i t e r } }$ is the number of iterations within a pruning interval, and $t _ { k }$ is the timestamp at the k-th iteration.

Although pruning can clear the background, it cannot solve the problem of detail loss caused by insufficient resolution. To this end, inspired by Pixel-GS et al. [34] [1], this paper proposes the radiative sub-pixel densification strategy. By accumulating gradients in high-resolution sub-pixel sampling, it guides the gaussian kernel to adaptively split in texture-rich regions, thereby enhancing the ability to restore small vascular branches.

This strategy dynamically identifies regions requiring refinement based on the projection gradients of gaussian kernels. Regions with larger gradients require higher gaussian kernel density. Therefore, the densification condition is formulated as:

$$
\mathcal { D } _ { \mathrm { R S D } } = \{ g _ { i } \ | \ \Vert \nabla _ { i } \Vert > \tau _ { \mathrm { g r a d } } \cdot \eta _ { \mathrm { i t e r } } \} ,\tag{3}
$$

where $g _ { i }$ is the gaussian kernel, $\nabla _ { i }$ is the accumulated gradient, $\tau _ { \mathrm { g r a d } }$ is the gradient threshold and $\eta _ { \mathrm { i t e r } }$ is an iteration-dependent decay coefficient.

For each seed kernel $g _ { i }$ that meets the condition, RSD generates K subkernels, and the newly generated kernels form a finer overlay around the parent kernel:

$$
\pmb { \mu } _ { i } ^ { ( k ) } = \pmb { \mu } _ { i } + \Delta \pmb { \mu } _ { i } ^ { ( k ) } , \quad \Delta \pmb { \mu } _ { i } ^ { ( k ) } \sim \mathcal { N } ( 0 , \pmb { \Sigma } _ { i } \cdot \alpha ) ,\tag{4}
$$

$$
\pmb { s } _ { i } ^ { ( k ) } = \beta \cdot \pmb { s } _ { i } , \quad \beta \in ( 0 , 1 ) ,\tag{5}
$$

where Î± controls the position offset magnitude and $\beta$ is the scale decay factor.

Furthermore, this strategy introduces a residual-guided mechanism that actively adds small-scale gaussian kernels in high-residual regions using the difference map between rendered and ground-truth images.

## 2.3 Multi-fidelity Texture Learning

Although sub-pixel constraints promote gaussian kernel densification, the input low-resolution image lacks high-frequency textures, and it is difficult to restore the fine structure of blood vessels solely by internal constraints.

To provide external priors for 4D reconstruction, we introduce a dedicated super-resolution (SR) model for DSA to generate high-resolution textures. To mitigate the risk of the model learning unreliable illusory artifacts, we further design a multi-fidelity learning strategy. Specifically, a confidence-aware teaching reference image is constructed, which preserves SR-enhanced textures in highconfidence regions while reverting to the original upsampled observations in lowconfidence areas:

$$
C = \sigma \left( \alpha \cdot  { S } ( I _ { \mathrm { S R } } , I _ { \mathrm { L R } } ^ { \uparrow } ) + \beta \cdot \mathcal { T } ( I _ { \mathrm { S R } } ) \right) ,\tag{6}
$$

$$
I _ { \mathrm { t e a c h } } = C \odot I _ { \mathrm { S R } } + ( 1 - C ) \odot I _ { \mathrm { L R } } ^ { \uparrow } ,\tag{7}
$$

where $I _ { \mathrm { S R } }$ and $I _ { \mathrm { L R } } ^ { \uparrow }$ denote the SR-generated and upsampled LR images respectively; $ { \boldsymbol { S } } ( \cdot , \cdot )$ represents SSIM for local consistency; $\tau ( \cdot )$ is a texture richness assessment function; $\alpha , \beta$ are learnable scaling parameters; $\sigma ( \cdot )$ denotes the Sigmoid function; and â denotes element-wise multiplication.

The total loss is formulated as a weighted combination of a high-fidelity loss ${ \mathcal L } _ { \mathrm { g t } }$ and a low-fidelity loss $\mathcal { L } _ { \mathrm { s r } }$ . Specifically, ${ \mathcal L } _ { \mathrm { g t } }$ is computed in the low-resolution space to ensure strict adherence to the original observations, while $\mathcal { L } _ { \mathrm { s r } }$ provides supervision in the high-resolution space via the teaching reference image:

$$
\left\{ \begin{array} { l l } { \mathcal { L } _ { \mathrm { g t } } = ( 1 - \lambda _ { \mathrm { s s i m } } ) \Vert I _ { \mathrm { r e n d } } ^ { \downarrow } - I _ { \mathrm { L R } } \Vert _ { 1 } + \lambda _ { \mathrm { s s i m } } ( 1 - \mathrm { S S I M } ( I _ { \mathrm { r e n d } } ^ { \downarrow } , I _ { \mathrm { L R } } ) ) } \\ { \mathcal { L } _ { \mathrm { s r } } = ( 1 - \lambda _ { \mathrm { s s i m } } ) \Vert I _ { \mathrm { r e n d } } - I _ { \mathrm { t e a c h } } \Vert _ { 1 } + \lambda _ { \mathrm { s s i m } } ( 1 - \mathrm { S S I M } ( I _ { \mathrm { r e n d } } , I _ { \mathrm { t e a c h } } ) ) } \end{array} \right. ,\tag{8}
$$

where $I _ { \mathrm { r e n d } }$ and $I _ { \mathrm { r e n d } } ^ { \downarrow }$ denote the rendered high-resolution projection and its downsampled low-resolution version, respectively; $I _ { \mathrm { L R } }$ is the original input projection; $I _ { \mathrm { t e a c h } }$ represents the confidence-aware teaching reference; $\lambda _ { \mathrm { s s i m } }$ is the balancing weight between the $\ell _ { 1 }$ distance and the structural similarity index.

Table 1: Comparison with other reconstruction methods. Bold indicates the best results. Underline indicates the second-best result.
<table><tr><td rowspan="2">View</td><td rowspan="2">Method</td><td colspan="2">DSA-28</td><td colspan="2">DSA-15</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>PSNR â</td><td>SSIM â</td></tr><tr><td rowspan="4">30</td><td>R2-Gaussian</td><td>28.163</td><td>0.7895</td><td>28.499</td><td>0.7871</td></tr><tr><td>TOGS</td><td>33.338</td><td>0.8362</td><td>33.273</td><td>0.8349</td></tr><tr><td>4DRGS</td><td>33.819</td><td>0.8541</td><td>33.687</td><td>0.8433</td></tr><tr><td>Ours</td><td>34.323</td><td>0.8563</td><td>34.198</td><td>0.8543</td></tr><tr><td rowspan="4">40</td><td> $\mathrm { R } ^ { 2 } \cdot$  Gaussian</td><td>28.394</td><td>0.7947</td><td>28.716</td><td>0.7934</td></tr><tr><td>TOGS</td><td>33.448</td><td>0.8384</td><td>33.417</td><td>0.8377</td></tr><tr><td>4DRGS</td><td>34.129</td><td>0.8568</td><td>34.017</td><td>0.8472</td></tr><tr><td>Ours</td><td>34.742</td><td>0.8600</td><td>34.645</td><td>0.8587</td></tr></table>

## 3 Experiments

## 3.1 Experiment Setup

Datasets We evaluated the model performance on two clinical DSA datasets: DSA-15 was derived from the 4DRGS work (15 cases), and DSA-28 was selfcollected data (28 cases), covering multi-center samples from the left, right, and back of the brain. In addition, an extra 135 clinical dynamic DSA sequences were collected. After digital subtraction processing, 17,822 vascular subtraction images were obtained for fine-tuning the super-resolution model.

Evaluation Criteria We use three complementary metrics to evaluate the generation quality of DSA images using three complementary metrics: pixel-level PSNR for reconstruction quality, patch-level SSIM [28] for structural consistency.

Implementation Details For radiative sub-pixel densification, we set a gradient accumulation window of 100 iterations, using the same splitting threshold as the density control. Pruning follows the accumulated attenuation strategy with threshold $\epsilon = 1 \times 1 0 ^ { - 6 }$ . The initial number of gaussian kernels is $M = 3 0 k$ with threshold $\delta = 0 . 0 1 6$ . The multi-fidelity learning weight $\beta = 0 . 4$ and SSIM loss weight $\lambda _ { \mathrm { s s i m } } = 0 . 2$ . Experiments are conducted on a single RTX 3090 GPU, with each case reconstructed in approximately 15 minutes.

## 3.2 Comparison With State-of-The-Art

To evaluate DSA-SRGS, we conducted quantitative comparisons against $\mathrm { R ^ { 2 } } \mathrm { \cdot }$ Gaussian, TOGS, and 4DRGS on two clinical datasets.

<!-- image-->  
Fig. 3: Qualitative comparison of different models..

Quantitative analysis Table 1 show that DSA-SRGS achieves optimal performance under all settings. Compared with the existing methods, this method achieves a PSNR of 34.32dB under 30 views and an LPIPS as low as 0.147, significantly enhancing the visual realism. From 30 to 40 views, the PSNR improvement of DSA-SRGS is superior to that of 4DRGS, demonstrating stronger sparse sampling robustness. On the two datasets, the performance of DSA-SRGS was consistent, and the SSIM was all above 0.85, verifying its effectiveness and generalization ability in clinical scenarios.

Qualitative analysis Fig. 3) shows the comparison of DSA projections of different methods from 30 perspectives. The R2-Gaussian structure is missing and artifacts are severe. The TOGS contour is rough and shows a distinct mosaic effect when magnified. The structure of 4DRGS is complete, but the details of the small branches are lost. In contrast, DSA-SRGS is significantly superior to the comparison methods, restoring the vascular margins and fine structures.

## 3.3 Ablation Analysis

We validate key components and super-resolution model effectiveness via ablation studies on clinical datasets as shown in table 2 and 3.

Effect of Core Components Table 2 validates the incremental contribution of each component. Starting from the 4DRGS baseline (33.819 dB PSNR on DSA-15), Blurry Supervision (+BS) yields a marginal gain (+0.211 dB). While Pseudo-Labels (+PL) boost PSNR to 34.144 dB, they induce a notable SSIM degradation (-0.0327), signaling structural hallucinations. Integrating BS mitigates this distortion (+PL+BS: SSIM 0.8537). The Multi-Fidelity module (+MF) further optimizes the trade-off between high-frequency details and observational constraints. Ultimately, the full model (+MF+C) achieves peak performance (34.323 dB/0.8563), demonstrating that confidence-aware fusion effectively suppresses artifacts while Radiative Sub-pixel Densification enhances micro-vascular modeling. Consistent trends on DSA-28 verify generalization.

Table 2: Ablation Study of Core Components for DSA-SRGS. SR: SR Rendering, BS: Blurry Supervision, PL: Pseudo-label, MF: Multi-fidelity Learning, C: Confidence.
<table><tr><td></td><td colspan="5">Components</td><td colspan="2">DSA-28</td><td colspan="2">DSA-15</td></tr><tr><td>Baseline</td><td>SR</td><td>BS</td><td>PL</td><td>MF</td><td>C</td><td>PSNR â</td><td>SSIM â</td><td>PSNR â</td><td>SSIM â</td></tr><tr><td>â</td><td></td><td></td><td></td><td></td><td></td><td>33.163</td><td>0.8405</td><td>33.819</td><td>0.8541</td></tr><tr><td>â</td><td></td><td>â</td><td></td><td></td><td></td><td>33.243</td><td>0.8417</td><td>34.030</td><td>0.8543</td></tr><tr><td>â</td><td>&gt;&gt;</td><td></td><td>&gt;&gt;</td><td></td><td></td><td>33.533</td><td>0.8312</td><td>34.144</td><td>0.8214</td></tr><tr><td>v</td><td>â</td><td>â</td><td></td><td></td><td></td><td>33.572</td><td>0.8493</td><td>34.223</td><td>0.8537</td></tr><tr><td>v</td><td>â</td><td></td><td>â</td><td></td><td>â</td><td>33.673</td><td>0.8504</td><td>34.287</td><td>0.8544</td></tr><tr><td>â</td><td>â</td><td></td><td>V</td><td></td><td>â V</td><td>34.198</td><td>0.8543</td><td>34.323</td><td>0.8563</td></tr></table>

Table 3: Quantitative comparison of super-resolution models.
<table><tr><td rowspan="2">SR Model</td><td colspan="2">DSA-28</td><td colspan="2">DSA-15</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>PSNR â</td><td>SSIM â</td></tr><tr><td>SwinIR [15]</td><td>33.547</td><td>0.8479</td><td>34.190</td><td>0.8536</td></tr><tr><td>Real-ESRGAN [27]</td><td>32.988</td><td>0.8430</td><td>33.601</td><td>0.8468</td></tr><tr><td>EDT [14]</td><td>33.536</td><td>0.8475</td><td>34.176</td><td>0.8526</td></tr><tr><td>HAT [3]</td><td>33.526</td><td>0.8485</td><td>34.164</td><td>0.8532</td></tr><tr><td>Mambair [9]</td><td>33.621</td><td>0.8499</td><td>34.253</td><td>0.8554</td></tr><tr><td>CATANet [17]</td><td>33.805</td><td>0.8518</td><td>34.257</td><td>0.8554</td></tr><tr><td>Ours</td><td>34.198</td><td>0.8543</td><td>34.323</td><td>0.8563</td></tr></table>

Effect of Super-resolution Model To establish the optimal pseudo-label generator, we benchmarked mainstream super-resolution architectures across both clinical datasets as shown in table 3. CATANet emerged as the superior base model, exhibiting robust cross-center generalization with minimal performance variance, attributed to its content-aware token aggregation mechanism that effectively models vascular dynamics. Crucially, our domain-specific fine-tuning yielded consistent gains across both clinical datasets, surpassing the pre-trained CATANet by 0.066/0.393 dB in PSNR and 0.0009/0.0025 in SSIM, respectively. This simultaneous improvement in fidelity and structural similarity underscores the necessity of adapting general SR priors to the DSA domain.

## 4 Conclusion

This paper proposes DSA-SRGS, a super-resolution gaussian splatting framework for DSA reconstruction of dynamic sparse views. This method integrates the prior of the super-resolution model into 4D reconstruction through a multifidelity texture learning module, and uses a confidence-aware strategy to balance multi-source supervision and suppress illusion artifacts. At the same time, radiative sub-pixel densification is introduced to guide the gaussian kernel to adaptively split in texture-rich areas, enhancing the modeling ability for small blood vessels. Experiments show that DSA-SRGS outperforms existing methods in both quantitative indicators and visual quality. Future work will explore self-supervised super-resolution reconstruction and further optimize the reconstruction speed to expand its clinical application.

## References

1. BulÃ², S.R., Porzi, L., Kontschieder, P.: Revising densification in gaussian splatting. arXiv preprint arXiv:2404.06109 (2024)

2. Chen, L., Zhang, M., Wang, S., Liu, J.: The role of digital subtraction angiography in the diagnosis and treatment of cerebral arteriovenous malformations. Neurosurgical Review 43(2), 567â576 (2020)

3. Chen, X., Wang, X., Zhou, J., Dong, C.: Activating more pixels in image superresolution transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22367â22377 (2023)

4. Chen, Z., Zhang, Y., Gu, S., Timofte, R., Van Gool, L.: Dual aggregation transformer for image super-resolution. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12312â12321 (2023)

5. Dong, C., Loy, C.C., He, K., Tang, X.: Learning a deep convolutional network for image super-resolution. In: European conference on computer vision. pp. 184â199. Springer (2014)

6. El Sayed, R., Sharifi, A., Park, C.C., Haussen, D.C., Allen, J.W., Oshinski, J.N.: Optimization of 4d flow mri spatial and temporal resolution for examining complex hemodynamics in the carotid artery bifurcation. Cardiovascular engineering and technology 14(3), 476â488 (2023)

7. Feng, X., He, Y., Wang, Y., Yang, Y., Li, W., Chen, Y., Kuang, Z., Fan, J., Jun, Y., et al.: Srgs: Super-resolution 3d gaussian splatting. arXiv preprint arXiv:2404.10318 (2024)

8. Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)

9. Guo, H., Guo, Y., Zha, Y., Zhang, Y., Li, W., Dai, T., Xia, S.T., Li, Y.: Mambairv2: Attentive state space restoration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2025)

10. Harrington, D.P., Boxt, L.M., Murray, P.D.: Digital subtraction angiography: overview of technical principles. American Journal of roentgenology 139(4), 781â 786 (1982)

11. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139â1 (2023)

12. Lang, S., GÃ¶litz, P., Struffert, T., Roesch, J., Roessler, K., Kowarschik, M., Strother, C., DÃ¶rfler, A.: 4d dsa for dynamic visualization of cerebral vasculature: A single-center experience in 26 cases. American Journal of Neuroradiology 38(6), 1169â1176 (2017)

13. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.: Gradient-based learning applied to document recognition. Proceedings of the IEEE 86(11), 2278â2324 (2002)

14. Li, W., Lu, X., Qian, S., Lu, J., Zhang, X., Jia, J.: On efficient transformer-based image pre-training for low-level vision. arXiv preprint arXiv:2112.10175 (2021)

15. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1833â1844 (2021)

16. Lim, B., Son, S., Kim, H., Nah, S., Mu Lee, K.: Enhanced deep residual networks for single image super-resolution. In: Proceedings of the IEEE conference on computer vision and pattern recognition workshops. pp. 136â144 (2017)

17. Liu, X., Liu, J., Tang, J., Wu, G.: Catanet: Efficient content-aware token aggregation for lightweight image super-resolution. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 17902â17912 (2025)

18. Liu, Y., Wang, Y., Zhang, X., Li, H., Chen, J.: Digital subtraction angiography in the diagnosis of cerebrovascular diseases: a systematic review and meta-analysis. Journal of Stroke and Cerebrovascular Diseases 27(10), 2735â2743 (2018)

19. Liu, Z., Zha, R., Zhao, H., Li, H., Cui, Z.: 4drgs: 4d radiative gaussian splatting for efficient 3d vessel reconstruction from sparse-view dynamic dsa images. In: International Conference on Information Processing in Medical Imaging. pp. 361â 374. Springer (2025)

20. MÃ¼ller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG) 41(4), 1â15 (2022)

21. Park, S., Son, M., Jang, S., Ahn, Y.C., Kim, J.Y., Kang, N.: Temporal interpolation is all you need for dynamic neural radiance fields. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4212â4221 (2023)

22. Ruedinger, K., Schafer, S., Speidel, M., Strother, C.: 4d-dsa: development and current neurovascular applications. American Journal of Neuroradiology 42(2), 214â220 (2021)

23. Sandoval-Garcia, C., Royalty, K., Aagaard-Kienitz, B., Schafer, S., Yang, P., Strother, C.: A comparison of 4d dsa with 2d and 3d dsa in the analysis of normal vascular structures in a canine model. American Journal of Neuroradiology 36(10), 1959â1963 (2015)

24. Sandoval-Garcia, C., Royalty, K., Yang, P., Niemann, D., Ahmed, A., Aagaard-Kienitz, B., BaÅkaya, M.K., Schafer, S., Strother, C.: 4d dsa a new technique for arteriovenous malformation evaluation: a feasibility study. Journal of neurointerventional surgery 8(3), 300â304 (2016)

25. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Å., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

26. Wang, H., Li, Y., Zhang, J., Chen, Y.: Diagnostic value of digital subtraction angiography in intracranial aneurysms: a meta-analysis. BMC Neurology 19(1), 1â8 (2019)

27. Wang, X., Xie, L., Dong, C., Shan, Y.: Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. pp. 1905â1914 (2021)

28. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600â612 (2004)

29. Xu, Z., Zhao, H., Cui, Z., Liu, W., Zheng, C., Wang, X.: Most-dsa: modeling motion and structural interactions for direct multi-frame interpolation in dsa images. arXiv preprint arXiv:2407.07078 (2024)

30. Xu, Z., Zhao, H., Liu, W., Wang, X.: Garamost: parallel multi-granularity motion and structural modeling for efficient multi-frame interpolation in dsa images. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 28530â 28538 (2025)

31. Zha, R., Lin, T.J., Cai, Y., Cao, J., Zhang, Y.: Rectifying radiative gaussian splatting for tomographic reconstruction. arXiv preprint arXiv:2405.20693 2(6), 7 (2024)

32. Zhang, S., Zhao, H., Zhou, Z., Wu, G., Zheng, C., Wang, X., Liu, W.: Togs: Gaussian splatting with temporal opacity offset for real-time 4d dsa rendering. IEEE Journal of Biomedical and Health Informatics (2025)

33. Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., Fu, Y.: Image super-resolution using very deep residual channel attention networks. In: Proceedings of the European conference on computer vision (ECCV). pp. 286â301 (2018)

34. Zhang, Z., Hu, W., Lao, Y., He, T., Zhao, H.: Pixel-gs: Density control with pixelaware gradient for 3d gaussian splatting. In: European Conference on Computer Vision. pp. 326â342. Springer (2024)

35. Zhao, H., Bai, Y., Chen, L., Ma, J., Lei, Y., Sun, T., Wu, L., Zhang, R., Xu, Z., Liang, X., et al.: Generative ai-based low-dose digital subtraction angiography for intra-operative radiation dose reduction: a randomized controlled trial. Nature Medicine pp. 1â9 (2026)

36. Zhao, H., Xu, Z., Chen, L., Wu, L., Cui, Z., Ma, J., Sun, T., Lei, Y., Wang, N., Hu, H., et al.: Large-scale pretrained frame generative model enables real-time low-dose dsa imaging: an ai system development and multi-center validation study. Med 6(1) (2025)