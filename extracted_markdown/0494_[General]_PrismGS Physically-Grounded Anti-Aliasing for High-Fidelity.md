# PrismGS: Physically-Grounded Anti-Aliasing for High-Fidelity Large-Scale 3D Gaussian Splatting

Houqiang Zhong1â, Zhenglong Wu2â, Sihua Fu2â, Zihan Zheng2, Xin Jin3, Xiaoyun Zhang2, Li Song1â , Qiang Hu2â 

1 School of Information Science and Electronic Engineering, Shanghai Jiao Tong University, Shanghai, China

2 Cooperative Mediant Innovation Center, Shanghai Jiao Tong University, Shanghai, China

3 College of Information Science and Technology, Eastern Institute of Technology, Ningbo, China {song li,qiang.hu}@sjtu.edu.cn

Abstractâ3D Gaussian Splatting (3DGS) has recently enabled real-time photorealistic rendering in compact scenes, but scaling to large urban environments introduces severe aliasing artifacts and optimization instability, especially under high-resolution (e.g., 4K) rendering. These artifacts, manifesting as flickering textures and jagged edges, arise from the mismatch between Gaussian primitives and the multi-scale nature of urban geometry. While existing âdivide-and-conquerâ pipelines address scalability, they fail to resolve this fidelity gap. In this paper, we propose PrismGS, a physically-grounded regularization framework that improves the intrinsic rendering behavior of 3D Gaussians. PrismGS integrates two synergistic regularizers. The first is pyramidal multi-scale supervision, which enforces consistency by supervising the rendering against a pre-filtered image pyramid. This compels the model to learn an inherently anti-aliased representation that remains coherent across different viewing scales, directly mitigating flickering textures. This is complemented by an explicit size regularization that imposes a physically-grounded lower bound on the dimensions of the 3D Gaussians. This prevents the formation of degenerate, viewdependent primitives, leading to more stable and plausible geometric surfaces and reducing jagged edges. Our method is plugand-play and compatible with existing pipelines. Extensive experiments on MatrixCity, Mill-19, and UrbanScene3D demonstrate that PrismGS achieves state-of-the-art performance, yielding significant PSNR gains around 1.5 dB against CityGaussian, while maintaining its superior quality and robustness under demanding 4K rendering.

Index TermsâLarge-scale Scene Reconstruction, Gaussian Splatting, Novel View Synthesis

## I. INTRODUCTION

Recent advances in 3D Gaussian Splatting (3DGS) [1] have redefined the frontier of radiance field [2]â[7], offering real-time photorealistic rendering for compact, object-centric scenes. However, extending this capability to large-scale, unbounded urban environments presents two fundamental challenges: scalability and aliasing. Urban scenes exhibit massive geometric complexity and long-range visibility, which exacerbate rendering artifacts such as flickering textures and jagged edges, especially under high-resolution (e.g., 4K) rendering.

These aliasing issues not only degrade perceptual quality but also limit the practical deployment of 3DGS in applications like digital twins, autonomous simulation, and XR-based city modeling.

The evolution toward large-scale reconstruction begins with NeRF-based methods [2], [8]â[11] like Mega-NeRF [12], which pioneer modular divide-and-conquer paradigms to address the spatial complexity of city blocks. However, their reliance on implicit volumetric representations results in slow training and rendering speeds. This limitation prompts a community-wide shift toward explicit representations, notably 3DGS [1] and its variant 2DGS [13]. Subsequent work focuses on adapting these explicit representations to cityscale scenes, primarily addressing the challenge of scalability. Octree-GS [14] and others [15]â[18] employ hierarchical data structures to manage the large number of primitives. City-Gaussian [19] and related approaches [20]â[22], adopt distributed block-wise optimization with Level-of-Detail strategies to improve efficiency. Momentum-GS [23] introduces a momentum-based self-distillation mechanism to improve consistency across independently trained blocks. Efficiencyoriented methods such as FlashGS [24]â[28] significantly reduce training and rendering time. However, aliasing artifacts, particularly under multi-scale viewing, remain pervasive and unresolved. In many high-resolution scenes, fine details shimmer, contours break, and surface coherence collapses, revealing a persistent gap in fidelity.

To bridge this fidelity gap, we introduce PrismGS, a regularization framework to mitigate aliasing in large-scale 3D Gaussian Splatting. Unlike previous efforts that focus on system-level scalability or architectural redesign, PrismGS directly enhances the intrinsic behavior of Gaussian primitives by enforcing consistency across scales and promoting geometric stability. Our key insight is that aliasing in urban-scale reconstructions arises from a mismatch between Gaussian parameters and the multi-scale nature of scene geometry and textures.

To address this issue, we jointly supervise rendering outputs across a range of resolutions during training, ensuring that each primitive contributes consistently at both coarse and fine scales. Simultaneously, we impose a physically-motivated constraint on the spatial extent of each Gaussian to prevent degenerate shapes and unstable optimization. These principles are integrated into a unified framework, regularizing the reconstruction process through both scale-consistent supervision and size-aware geometry control, effectively reducing artifacts such as flickering and jagged edges in high-resolution urban scenes. Extensive experiments demonstrate that PrismGS achieves state-of-the-art results in both quantitative metrics and perceptual quality, especially under 4K rendering conditions. Our contributions are summarized as follows:

<!-- image-->  
Fig. 1. Overview of the PrismGS framework. Our method first partitions the scene into blocks for parallel training. During optimization, we introduce two key regularizers: a Multi-Scale Supervision loss for anti-aliasing and a Size Regularization loss for geometric stability.

â¢ We propose PrismGS, a physically-grounded regularization framework for large-scale 3DGS that improves anti-aliasing and rendering fidelity without compromising scalability.

â¢ We introduce pyramidal multi-scale supervision for crossresolution consistency, and Gaussian size regularization to enhance geometric stability and suppress highfrequency artifacts.

â¢ Our method consistently outperforms existing approaches on challenging benchmarks, achieving +1.0â1.5 dB PSNR gains and superior perceptual quality (SSIM, LPIPS) under high-resolution rendering.

## II. METHODS

We build upon 3DGS [1], which represents a scene as anisotropic Gaussians $G _ { i }$ with position $\mu _ { i } ,$ scale ${ \bf s } _ { i } ,$ rotation $\mathbf { q } _ { i }$ (defining $\Sigma _ { i } )$ , opacity $\alpha _ { i } .$ , and SH color $\mathbf { c } _ { i } .$ Primitives are initialized from SfM [29] and rendered via a differentiable rasterizer with alpha blending. While efficient, direct application to city-scale scenes leads to cross-scale inconsistency and geometric degeneracy [19], [23]. PrismGS addresses these two failure modes with multi-scale supervision and size regularization.

## A. Tackling Aliasing with Multi-Scale Image Pyramids

A primary challenge in large-scale rendering is aliasing, which manifests as flickering details and jagged edges when the scene is viewed from varying distances [1], [19]. This occurs because rendering high-frequency geometry and textures at a coarse resolution without proper pre-filtering is analogous to undersampling a signal. An ideal coarse-level rendering, $\hat { I } _ { \mathrm { c o a r s e } } .$ , should approximate a low-pass filtered version of the fine-level rendering, $\hat { I } _ { \mathrm { f i n e } }$ [30]. To enforce this constraint directly during optimization, we introduce a Multi-Scale Supervision (MSS) loss.

The core idea of MSS is to compel the model to maintain photometric consistency across a resolution pyramid, inspired by the classic mipmapping technique [30]. During each training iteration, for a rendered image and its corresponding ground-truth image $I _ { g t }$ , we construct a pair of L-level image pyramids, $\mathcal { P } _ { \mathrm { r e n d e r } } \stackrel { \sim } { = } \{ { \hat { I } } ^ { ( 0 ) } , \dotsc , { \hat { I } } ^ { ( L - 1 ) } \}$ and $\mathcal { P } _ { g t } = \{ I _ { g t } ^ { ( 0 ) } , \ldots , I _ { g t } ^ { ( L - 1 ) } \}$ . To create a properly anti-aliased ground-truth pyramid, each level is generated by applying a low-pass filter (a Gaussian blur, denoted by the operator $\textstyle { \mathcal { F } } _ { \sigma } )$ before downsampling (denoted by the operator $\mathcal { D } _ { s }$ with a factor $s = 2 )$ . This process is defined as:

$$
I _ { g t } ^ { ( l + 1 ) } = \mathcal { D } _ { 2 } ( \mathcal { F } _ { \sigma } ( I _ { g t } ^ { ( l ) } ) )\tag{1}
$$

Here, the superscript (l) denotes the pyramid level, with $l = 0$ being the original resolution. The rendered pyramid $\{ \hat { I } ^ { ( l ) } \}$ is produced by rendering the scene at each corresponding resolution. The MSS loss, $\mathcal { L } _ { m s s } ,$ is then formulated as the weighted sum of L1 norms across the downsampled levels:

$$
\mathcal { L } _ { m s s } = \sum _ { l = 1 } ^ { L - 1 } | | \hat { I } ^ { ( l ) } - I _ { g t } ^ { ( l ) } | | _ { 1 } .\tag{2}
$$

By penalizing discrepancies at lower resolutions against a properly pre-filtered ground truth, this loss function acts as an implicit, end-to-end differentiable anti-aliasing filter. It forces the optimizer to learn a set of Gaussian parameters that are not only accurate for the high-resolution view but also remain stable and coherent when downsampled, effectively baking anti-aliasing properties into the 3D primitives themselves.

## B. Preventing Geometric Degeneracy with Size Regularization

Another key challenge in 3DGS is geometric instability, where the optimization process, driven solely by a photometric loss, may produce physically implausible, degenerate primitives. These often take the form of extremely thin âneedle-likeâ or flat âpancake-likeâ Gaussians that overfit to high-frequency details in the training images. Such primitives are not robust and cause rendering artifacts like holes and flickering when viewed from novel angles or under high magnification.

To address this, we introduce an explicit 3D Gaussian size regularization loss, $\mathcal { L } _ { s i z e }$ . The goal is to prevent the model from creating primitives smaller than a physical limit. For each camera in the training set with focal length f , the pixel sampling interval in 3D space at a depth d is $T \ = \ d / f$ According to the Nyquist theorem, to reconstruct a signal without aliasing, the smallest resolvable 3D structure is approximately 2T . We can therefore establish a global minimum sampling interval, $T _ { m i n }$ , across all training views to define a physical lower bound on Gaussian size. The regularization loss penalizes any Gaussian whose smallest scaling axis falls below a defined threshold, $\tau _ { s i z e }$

$$
\mathcal { L } _ { s i z e } = \sum _ { i } \operatorname* { m a x } ( 0 , \tau _ { s i z e } - \operatorname* { m i n } ( \mathbf { s } _ { i } ) )\tag{3}
$$

where $\tau _ { s i z e }$ is a hyperparameter defining the minimum allowable scaling axis, and min(si) is the smallest component of the scaling vector si for the i-th Gaussian. This loss effectively suppresses high-frequency artifacts and encourages the formation of smoother, more continuous surfaces that better represent the true scene geometry.

## C. Joint Optimization for Robust Reconstruction

Our final training objective integrates the standard photometric loss with our two novel regularization terms. The base reconstruction loss, $\mathcal { L } _ { b a s e } ,$ is a weighted combination of an L1 norm and a structural dissimilarity (D-SSIM) loss, calculated at the highest resolution (l = 0):

$$
\mathcal { L } _ { b a s e } = ( 1 - \lambda _ { d s s i m } ) \cdot | | \hat { I } ^ { ( 0 ) } - I _ { g t } ^ { ( 0 ) } | | _ { 1 } + \lambda _ { d s s i m } \cdot ( 1 - \mathrm { S S I M } ( \hat { I } ^ { ( 0 ) } , I _ { g t } ^ { ( 0 ) } ) )\tag{4}
$$

The total loss function, $\mathcal { L } _ { t o t a l }$ , is then a weighted sum of the base loss and our two regularization terms:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { b a s e } + \lambda _ { m s s } \cdot \mathcal { L } _ { m s s } + \lambda _ { s i z e } \cdot \mathcal { L } _ { s i z e }\tag{5}
$$

The hyperparameters $\lambda _ { d s s i m } , ~ \lambda _ { m s s } ,$ , and $\lambda _ { s i z e }$ balance the influence of the structural dissimilarity, the multi-scale supervision, and the size regularization, respectively. This unified objective function guides the optimization to produce a 3D representation that is robust against both aliasing and geometric degradation, making it highly suitable for high-fidelity, large-scale scene reconstruction.

## III. EXPERIMENTS

Our framework builds on Momentum-GS [23] and is trained for 60,000 iterations on 8 NVIDIA 3090 GPUs. We fix all loss weights across experiments: $\lambda _ { d s s i m } = 0 . 2 , \lambda _ { m s s } = 0 . 1$ , and $\lambda _ { s i z e } = 0 . 0 1$ . Quantitative and qualitative evaluations are conducted on three large-scale benchmarks: MatrixCity [31], Mill-19 [12], and UrbanScene3D [32]. Following prior work [19], all input images are downsampled by a factor of 4 for standard evaluation. To assess anti-aliasing performance under challenging conditions, we additionally render high-resolution (3840 Ã 2160) novel views on the Building and Rubble scenes from Mill-19. We compare our method against SOTA approaches, including the NeRF-based Mega-NeRF [12], and Gaussian-based methods: 3DGS [1], 2DGS [13], Octree-GS [14], CityGaussian [19], and Momentum-GS [23]. All these methods are evaluated using their default training strategies and hyperparameter configurations to ensure a fair comparison.

## A. Quantitative Comparisons

As shown in Tab. I, PrismGS consistently outperforms prior methods. On the Building scene, our LPIPS of 0.185 marks a significant improvement over the baseline Momentum-GSâs 0.199. This directly reflects the success of our anti-aliasing objective, as LPIPS is highly sensitive to the flickering and texture shimmering. This advantage is further magnified in our 4K high-resolution in Tab. II. Here, PrismGS maintains its performance lead, outperforming all competitors across all metrics on the Rubble dataset. This robust performance at a demanding resolution directly validates the effectiveness of our physically-grounded regularization.

## B. Qualitative Comparisons

The qualitative results in Fig. 2 further corroborate our quantitative findings. Visual comparison reveals that PrismGS generates renderings with substantially higher clarity and detail compared to other methods like CityGaussian and Momentum-GS. As highlighted in the magnified insets, our approach excels at reconstructing fine-grained textures and sharp geometric details on distant structures, whereas other methods often suffer from blurriness or aliasing artifacts. Furthermore, our method effectively preserves structural consistency and mitigates the visual popping artifacts common in large-scale rendering, confirming the benefits of our proposed pyramidal supervision and geometric regularization.

## C. Ablation Study

In Tab. III, we conduct an ablation study on the MatrixCity scene. Adding only $\mathcal { L } _ { m s s }$ primarily reduces LPIPS, indicating fewer aliasing artifacts. Adding only the size regularization $( \mathcal { L } _ { s i z e } )$ yields a more substantial performance leap, boosting PSNR to 28.124 and significantly lowering LPIPS to 0.182. This highlights the critical role of constraining Gaussian sizes in preventing geometric degeneracy. Finally, our full model, which integrates both components, achieves the best performance, pushing PSNR to 28.272 and LPIPS to 0.173. This analysis confirms that our two modules are complementary: $\mathcal { L } _ { m s s }$ primarily targets view-dependent aliasing, while $\mathcal { L } _ { s i z e }$ enforces view-independent geometric stability. The qualitative results of the ablation in Fig. 3 further support this conclusion: excluding any module will lead to a decrease in reconstruction quality and other impacts. Only a complete model can achieve the best performance.

TABLE I  
QUANTITATIVE COMPARISON ACROSS FOUR LARGE-SCALE SCENES. WE PRESENT METRICS FOR PSNRâ, SSIMâ, AND LPIPSâ ON TEST VIEWS.
<table><tr><td>Scene</td><td colspan="3">Building</td><td colspan="3">Rubble</td><td colspan="3">Residence</td><td colspan="3">Sci-Art</td></tr><tr><td>Metrics</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>Mega-NeRF</td><td>21.134</td><td>0.557</td><td>0.494</td><td>24.150</td><td>0.558</td><td>0.509</td><td>22.054</td><td>0.632</td><td>0.485</td><td>25.719</td><td>0.772</td><td>0.386</td></tr><tr><td> 2DGS</td><td>19.186</td><td>0.647</td><td>0.401</td><td>24.293</td><td>0.734</td><td>0.336</td><td>21.077</td><td>0.763</td><td>0.276</td><td>20.046</td><td>0.792</td><td>0.290</td></tr><tr><td>3DGS</td><td>20.437</td><td>0.716</td><td>0.304</td><td>25.304</td><td>0.787</td><td>0.262</td><td>21.697</td><td>0.789</td><td>0.228</td><td>21.644</td><td>0.840</td><td>0.226</td></tr><tr><td>Octree-GS</td><td>17.748</td><td>0.439</td><td>0.613</td><td>21.521</td><td>0.478</td><td>0.629</td><td>18.721</td><td>0.526</td><td>0.519</td><td>18.056</td><td>0.598</td><td>0.521</td></tr><tr><td>CityGaussian</td><td>21.483</td><td>0.757</td><td>0.268</td><td>24.929</td><td>0.772</td><td>0.268</td><td>21.720</td><td>0.799</td><td>0.221</td><td>21.044</td><td>0.826</td><td>0.241</td></tr><tr><td>Momentum-GS</td><td>23.193</td><td>0.810</td><td>0.199</td><td>25.771</td><td>0.807</td><td>0.227</td><td>22.040</td><td>00.798</td><td>0.213</td><td>22.888</td><td>0.839</td><td>0.222</td></tr><tr><td>Ours</td><td>23.516</td><td>0.826</td><td>0.185</td><td>26.124</td><td>0.838</td><td>0.195</td><td>22.339</td><td>0.819</td><td>0.207</td><td>23.317</td><td>0.851</td><td>0.209</td></tr></table>

Mega-NeRF  
Octree-GS  
CityGaussian  
Momentum-GS  
Ours  
GT  
<!-- image-->  
Fig. 2. Qualitative comparisons of different methods (Mega-NeRF, Octree-GS, CityGaussian, Momentum-GS, Ours) against Ground Truth across four largescale scenes. Orange insets highlight patches that reveal notable visual differences, demonstrating the superiority of our method in capturing fine details and maintaining structural consistency.  
TABLE III

TABLE II  
QUANTITATIVE COMPARISON UNDER 4K RESOLUTION.
<table><tr><td>Dataset</td><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td rowspan="4">Building Dataset</td><td>Mega-NeRF</td><td>20.078</td><td>0.537 0.589</td><td>0.619</td></tr><tr><td>3DGS Octree-GS</td><td>17.937</td><td></td><td>0.481</td></tr><tr><td>CityGaussian</td><td>17.344</td><td>0.513</td><td>0.603</td></tr><tr><td>Momentum-GS</td><td>20.523 21.057</td><td>0.683 0.682</td><td>0.398 0.426</td></tr><tr><td rowspan="6">Rubble Dataset</td><td>ours</td><td>21.291</td><td>0.693</td><td>0.401</td></tr><tr><td>Mega-NeRF 3DGS</td><td>22.873</td><td>0.516</td><td>0.656</td></tr><tr><td></td><td>23.457</td><td>0.643</td><td>0.483</td></tr><tr><td>Octree-GS</td><td>20.887</td><td>0.516</td><td>0.625</td></tr><tr><td>CityGaussian</td><td>23.436</td><td>0.661</td><td>0.452</td></tr><tr><td>Momentum-GS ours</td><td>23.717 24.001</td><td>0.671 0.687</td><td>0.422 0.401</td></tr></table>

<!-- image-->

<!-- image-->

<!-- image-->  
+mss

<!-- image-->  
+size  
Fig. 3. Qualitative results of ablation study. Excluding any module leads to lower reconstruction quality and other impacts.

ABLATION STUDY ON DIFFERENT STRATEGY OF MEASURING THE RECONSTRUCTION QUALITY UNDER MATRIXCITY SCENE
<table><tr><td>Models</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>baseline</td><td>27.768</td><td>0.867</td><td>0.212</td></tr><tr><td> $+ \mathcal { L } _ { m s s }$ </td><td>27.892</td><td>0.870</td><td>0.206</td></tr><tr><td> $+ \mathcal { L } _ { s i z e }$ </td><td>28.124</td><td>0.878</td><td>0.182</td></tr><tr><td>Full model</td><td>28.272</td><td>0.888</td><td>0.173</td></tr></table>

## IV. CONCLUSION

In this paper, we introduced PrismGS, a regularization framework designed to significantly enhance the fidelity of large-scale 3D Gaussian Splatting. By building upon scalable block-based pipelines, our work specifically targets the pervasive issues of aliasing and geometric instability. PrismGS integrates two synergistic components: a pyramidal multiscale supervision loss that enforces rendering consistency across different resolutions, and a physically-grounded size regularization that prevents the formation of degenerate, viewdependent primitives. Extensive experiments on challenging benchmarks demonstrate that our method achieves SOTA results, significantly improving both quantitative metrics and perceptual quality, especially for high-fidelity 4K rendering. A limitation of our method is its assumption of a static environment, as it does not explicitly filter dynamic objects. One promising direction is the integration of semantic scene understanding to differentiate and model static and dynamic elements separately.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering,â ACM Trans. Graph., vol. 42, no. 4, July 2023.

[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: representing scenes as neural radiance fields for view synthesis,â Commun. ACM, vol. 65, no. 1, p. 99â106, December 2021.

[3] Z. Zheng, H. Zhong, Q. Hu, X. Zhang, L. Song, Y. Zhang, and Y. Wang, âJointrf: End-to-end joint optimization for dynamic neural radiance field representation and compression,â in 2024 IEEE International Conference on Image Processing (ICIP), 2024, pp. 3292â3298.

[4] Q. Hu, Z. Zheng, H. Zhong, S. Fu, L. Song, X. Zhang, G. Zhai, and Y. Wang, â4dgc: Rate-aware 4d gaussian compression for efficient streamable free-viewpoint video,â in Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), June 2025, pp. 875â885.

[5] Q. Hu, Q. He, H. Zhong, G. Lu, X. Zhang, G. Zhai, and Y. Wang, âVarfvv: View-adaptive real-time interactive free-view video streaming with edge computing,â IEEE Journal on Selected Areas in Communications, vol. 43, no. 7, pp. 2620â2634, 2025.

[6] Q. Hu, H. Zhong, Z. Zheng, X. Zhang, Z. Cheng, L. Song, G. Zhai, and Y. Wang, âVrvvc: Variable-rate nerf-based volumetric video compression,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 4, 2025, pp. 3563â3571.

[7] Z. Zheng, H. Zhong, Q. Hu, X. Zhang, L. Song, Y. Zhang, and Y. Wang, âHpc: Hierarchical progressive coding framework for volumetric video,â in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 7937â7946.

[8] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. P. Mildenhall, P. Srinivasan, J. T. Barron, and H. Kretzschmar, âBlock-nerf: Scalable large scene neural view synthesis,â in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 8238â8248.

[9] Y. Zhang, G. Chen, and S. Cui, âEfficient large-scale scene representation with a hybrid of high-resolution grid and plane features,â Pattern Recognition, vol. 158, p. 111001, 2025.

[10] K. Rematas, A. Liu, P. Srinivasan, J. Barron, A. Tagliasacchi, T. Funkhouser, and V. Ferrari, âUrban radiance fields,â in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 12 922â12 932.

[11] Y. Xiangli, L. Xu, X. Pan, N. Zhao, A. Rao, C. Theobalt, B. Dai, and D. Lin, âBungeenerf: Progressive neural radiance field for extreme multiscale scene rendering,â in Computer Vision â ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23â27, 2022, Proceedings, Part XXXII. Berlin, Heidelberg: Springer-Verlag, 2022, p. 106â122.

[12] H. Turki, D. Ramanan, and M. Satyanarayanan, âMega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 12 922â12 931.

[13] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2d gaussian splatting for geometrically accurate radiance fields,â in ACM SIGGRAPH 2024 Conference Papers, ser. SIGGRAPH â24. New York, NY, USA: Association for Computing Machinery, 2024.

[14] K. Ren, L. Jiang, T. Lu, M. Yu, L. Xu, Z. Ni, and B. Dai, âOctree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians,â IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1â15, 2025.

[15] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffold-gs: Structured 3d gaussians for view-adaptive rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 654â20 664.

[16] J. Cui, J. Cao, F. Zhao, Z. He, Y. Chen, Y. Zhong, L. Xu, Y. Shi, Y. Zhang, and J. Yu, âLetsgo: Large-scale garage modeling and rendering via lidar-assisted gaussian primitives,â ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp. 1â18, 2024.

[17] Y. Wang, Z. Li, L. Guo, W. Yang, A. Kot, and B. Wen, âContextGS : Compact 3d gaussian splatting with anchor level context model,â in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[18] Z. Yan, W. F. Low, Y. Chen, and G. H. Lee, âMulti-scale 3d gaussian splatting for anti-aliased rendering,â in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20 923â 20 931.

[19] Y. Liu, C. Luo, L. Fan, N. Wang, J. Peng, and Z. Zhang, âCitygaussian: Real-time high-quality large-scale scene rendering with gaussians,â in

Computer Vision â ECCV 2024: 18th European Conference, Milan, Italy, September 29âOctober 4, 2024, Proceedings, Part XVI. Berlin, Heidelberg: Springer-Verlag, 2024, p. 265â282.

[20] J. Lin, Z. Li, X. Tang, J. Liu, S. Liu, J. Liu, Y. Lu, X. Wu, S. Xu, Y. Yan, and W. Yang, âVastgaussian: Vast 3d gaussians for large scene reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 5166â 5175.

[21] Y. Chen and G. H. Lee, âDogs: Distributed-oriented gaussian splatting for large-scale 3d reconstruction via gaussian consensus,â in The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024.

[22] Y. Liu, C. Luo, Z. Mao, J. Peng, and Z. Zhang, âCitygaussianv2: Efficient and geometrically accurate reconstruction for large-scale scenes,â in The Thirteenth International Conference on Learning Representations, 2025.

[23] J. Fan, W. Li, Y. Han, and Y. Tang, âMomentum-gs: Momentum gaussian self-distillation for high-quality large scene reconstruction,â in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025.

[24] G. Feng, S. Chen, R. Fu, Z. Liao, Y. Wang, T. Liu, B. Hu, L. Xu, Z. Pei, H. Li, X. Li, N. Sun, X. Zhang, and B. Dai, âFlashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering,â in Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), June 2025, pp. 26 652â26 662.

[25] K. Song, X. Zeng, C. Ren, and J. Zhang, âCity-on-web: Real-time neural rendering of large-scale scenes on the web,â in Computer Vision â ECCV 2024, A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds. Cham: Springer Nature Switzerland, 2025, pp. 385â402.

[26] A. Meuleman, I. Shah, A. Lanvin, B. Kerbl, and G. Drettakis, âOnthe-fly reconstruction for large-scale novel view synthesis from unposed images,â ACM Transactions on Graphics, vol. 44, no. 4, Aug. 2025, nef/OPAL.

[27] H. Zhao, H. Weng, D. Lu, A. Li, J. Li, A. Panda, and S. Xie, âOn scaling up 3d gaussian splatting training,â in Computer Vision â ECCV 2024 Workshops, A. Del Bue, C. Canton, J. Pont-Tuset, and T. Tommasi, Eds. Cham: Springer Nature Switzerland, 2025, pp. 14â36.

[28] W. Liu, T. Guan, B. Zhu, L. Xu, Z. Song, D. Li, Y. Wang, and W. Yang, âEfficientgs: Streamlining gaussian splatting for large-scale high-resolution scene representation,â IEEE MultiMedia, vol. 32, no. 1, pp. 61â71, 2025.

[29] J. L. Schonberger and J.-M. Frahm, âStructure-from-motion revisited,â Â¨ in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104â4113.

[30] J. P. Ewins, M. D. Waller, M. White, and P. F. Lister, âImplementing an anisotropic texture filter,â Computers & Graphics, vol. 24, no. 2, pp. 253â267, 2000.

[31] Y. Li, L. Jiang, L. Xu, Y. Xiangli, Z. Wang, D. Lin, and B. Dai, âMatrixcity: A large-scale city dataset for city-scale neural rendering and beyond,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3205â3215.

[32] L. Lin, Y. Liu, Y. Hu, X. Yan, K. Xie, and H. Huang, âCapturing, reconstructing, and simulating: the urbanscene3d dataset,â in ECCV, 2022.