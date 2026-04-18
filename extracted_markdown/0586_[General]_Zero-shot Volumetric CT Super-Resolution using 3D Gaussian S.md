# Zero-shot CT Super-Resolution using Diffusion-based 2D Projection Priors and Signed 3D Gaussians

Jeonghyun Nohâ, Hyun-Jic Ohâ, Won-Ki Jeongâ 

Korea University {wjdgus0967, hyunjic0127, wkjeong}@korea.ac.kr

Abstract. Computed tomography (CT) is important in clinical diagnosis, but acquiring high-resolution (HR) CT is constrained by radiation exposure risks. While deep learning-based super-resolution (SR) methods have shown promise for reconstructing HR CT from low-resolution (LR) inputs, supervised approaches require paired datasets that are often unavailable. Zero-shot methods address this limitation by operating on single LR inputs; however, they frequently fail to recover fine structural details due to limited LR information within individual volumes. To overcome these limitations, we propose a novel zero-shot 3D CT SR framework that integrates diffusion-based upsampled 2D projection priors into the 3D reconstruction process. Specifically, our framework consists of two stages: (1) LR CT projection SR, training a diffusion model on abundant X-ray data to upsample LR projections, thereby enhancing the scarce information inherent in the LR inputs. (2) 3D CT volume reconstruction, using 3D Gaussian splatting with our novel Negative Alpha Blending (NAB-GS), which models positive and negative Gaussian densities to learn signed residuals between diffusion-generated HR and upsampled LR projections. Our framework demonstrates superior quantitative and qualitative performance on two public datasets, and expert evaluations present the frameworkâs clinical potential at 4Ã.

Keywords: Zero-shot CT Super-resolution Â· Diffusion Models Â· 3D Gaussian Splatting.

## 1 Introduction

High-quality computed tomography (CT) is an essential modality for accurate clinical diagnosis and treatment planning. However, acquiring high-resolution (HR) CT scans necessitates elevated radiation doses, risking potential DNA damage and radiation-induced malignancies [18]. While reducing radiation dose mitigates these concerns, it inevitably limits the data available for image reconstruction, thereby degrading spatial resolution [19]. This trade-off often obscures anatomical structures, potentially compromising diagnostic precision.

To address this, super-resolution (SR) techniques have emerged as a vital tool for recovering detailed structures from low-resolution (LR) inputs [11]. While deep learning has achieved significant milestones using convolutional neural networks [7, 35], transformers [3, 16, 27], generative adversarial networks [15], and diffusion models [13, 17, 22, 23, 30], these are primarily tailored to 2D images and often fail to exploit volumetric consistency in medical imaging. In the medical domain, supervised 3D SR approaches, ranging from spatially aware interpolation networks [21] to implicit neural representations (INR) [8, 32], have been proposed. Despite their effectiveness, their applicability is hindered by the scarcity of paired HR-LR training volumes. Consequently, zero-shot learning has gained traction as a data-efficient alternative [24, 25, 28], with recent innovations like CuNeRF [4] adapting neural radiance fields for continuous upsampling. Yet, relying solely on LR information, current zero-shot methods often struggle to recover structural details, yielding over-smoothed reconstructions.

To overcome these limitations, we propose a novel zero-shot 3D CT SR framework that reformulates volumetric SR as a reconstruction task using diffusionbased upsampled 2D projection priors. The proposed framework consists of two key stages: (1) LR CT projection SR, and (2) 3D reconstruction. First, given that acquiring 2D X-ray data is more feasible than collecting paired 3D HR-LR volumes, we adopt a diffusion model trained on large-scale 2D X-ray datasets. We then employ this diffusion prior within a Denoising Diffusion Null-space Model (DDNM) [29] to generate 2D HR CT projections from LR counterparts. This design choice intends LR projections to enforce data consistency, simultaneously enhancing the projection information through the diffusion model prior. Second, we introduce a Negative Alpha Blending Gaussian Splatting (NAB-GS) to reconstruct the HR 3D CT volume. NAB-GS learns the residual field between the diffusion-generated HR projections and the projections of the upsampled LR volume. To fully exploit this residual field, which inherently contains both positive and negative values, NAB-GS relaxes the non-negativity constraint of standard 3DGS [14], enabling precise encoding of signed residuals and substantially improving the recovery of structural details. Consequently, our framework achieves superior results on two public datasets, with expert evaluations suggesting its clinical potential at 4Ã. Our main contributions are summarized as follows:

â We propose a novel zero-shot 3D CT SR framework that reformulates volumetric SR as a 3D reconstruction task driven by diffusion-based upsampled 2D projection priors. Leveraging a diffusion model trained on large-scale Xray datasets, we mitigate the need for paired HR-LR volumes and enhance the limited information of LR inputs.

We introduce NAB-GS, a Gaussian splatting that relaxes the non-negativity constraint of standard 3DGS to learn signed residual fields between diffusiongenerated HR projections and upsampled LR projections, enabling precise residual encoding and improved structural details.

â We validate our framework on the UHRCT [6] and MELAâ  public datasets, achieving superior quantitative and qualitative performance compared to state-of-the-art (SOTA) zero-shot methods, alongside expert evaluations supporting its CT volume quality and clinical potential.

<!-- image-->  
Fig. 1. Overview of our framework. (a) LR projection SR using diffusion model: A pre-trained diffusion model with 2D X-ray data is employed within the DDNM to generate HR 2D CT projection images from LR counterparts. (b) 3D CT reconstruction via NAB-GS: Using both positive and negative density Gaussians, we model a signed residual field between diffusion-generated HR projections and LR counterparts. For HR volume generation, the learned residual field is added onto the upsampled LR volume.

## 2 Zero-Shot 3D CT SR Framework

## 2.1 LR Projection SR using Diffusion Model

The first stage focuses on generating high-fidelity 2D CT projections to serve as reliable guidance for 3D reconstruction, as shown in Fig. 1(a). We train a diffusion model on large-scale 2D X-ray datasets specifically to model a generative projection prior. Subsequently, this prior is adapted to our CT projection SR task using DDNM [29], enhancing the limited information in LR CT projections.

SR using DDNM. Let y = Ax denote an LR CT projection, where x is the unknown HR projection and A is the downsampling operator. Our goal is to estimate an HR projection xË that both matches the measured LR projection and lies on the manifold modeled by the diffusion prior.

Following the null-space based diffusion formulation in [29], we combine the learned prior with a data-consistency constraint defined by y. At each reverse diffusion step, the estimate is decomposed into range and null-space components:

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { 0 \mid t } = \mathbf { A } ^ { \dagger } \mathbf { y } + ( \mathbf { I } - \mathbf { A } ^ { \dagger } \mathbf { A } ) \mathbf { x } _ { 0 \mid t } , } \end{array}\tag{1}
$$

where $\mathbf { A } ^ { \dagger }$ denotes the pseudo-inverse of A and $\mathbf { x } _ { 0 \mid t }$ is the denoised estimate predicted by the diffusion model at step t. The first term enforces data consistency in the range space of A, while the second term allows the diffusion prior to refine details in the null space, injecting high-frequency structures.

To account for measurement noise or slight inconsistencies in LR projections, we further adopt the DDNM+ [29]:

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { 0 \mid t } = \mathbf { x } _ { 0 \mid t } - \varSigma _ { t } \mathbf { A } ^ { \dagger } \big ( \mathbf { A } \mathbf { x } _ { 0 \mid t } - \mathbf { y } \big ) , } \end{array}\tag{2}
$$

where $\Sigma _ { t }$ controls the strength of the correction at each diffusion step, balancing data fidelity and perceptual quality. This null-space diffusion process allows us to generate HR CT projections that inherit realistic projection appearance from the trained diffusion prior while leveraging LR projection information to enforce data consistency. In this way, an X-ray trained diffusion prior coupled with DDNM offers a practical design to enhance the limited information in LR CT inputs under a zero-shot SR setting.

## 2.2 3D CT Reconstruction via NAB-GS

The second stage performs 3D CT reconstruction via the proposed NAB-GS, as shown in Fig. 1(b). We obtain an initial upsampled LR volume by cubic upsampling of the LR volume and initialize a radiative Gaussian field from this upsampled LR volume. To reconstruct the HR volume, we learn a residual field between the diffusion-generated HR projections and the upsampled LR baseline. The Gaussians are optimized via rasterizer and voxelizer (following R2-GS [33]) under reconstruction and total variation (TV) losses to refine structural details.

Negative alpha blending. Conventional 3DGS [14] enforces non-negativity of density $\rho$ by softplus activation $\displaystyle ( \rho = \ln ( 1 + e ^ { z } ) )$ ) and alpha blending, where z is the raw density. While physically plausible, this restricts the representation to strictly positive densities. However, our residual formulation targets the discrepancy between the diffusion-generated HR projection and the upsampled LR baseline. Since the upsampled LR can locally over- or underestimate true intensities, the residual inherently contains both positive and negative values.

To faithfully encode signed residuals, first, we replace the softplus with Parametric ReLU (PReLU), allowing for negative densities:

$$
\phi ( z ) = \left\{ z , \quad \mathrm { i f } \ z \ge 0 \right. \ , \quad \phi ^ { - 1 } ( \rho ) = \left\{ \begin{array} { l l } { \rho , \quad \mathrm { i f } \ \rho \ge 0 } \\ { \gamma z , \quad \mathrm { o t h e r w i s e } } \end{array} \right. , \quad \phi ^ { - 1 } ( \rho ) = \left\{ \begin{array} { l l } { \rho , \quad \mathrm { i f } \ \rho \ge 0 } \\ { \frac { \rho } { \gamma } , \quad \mathrm { o t h e r w i s e } } \end{array} \right. ,\tag{3}
$$

where $\phi ( \cdot )$ and $\phi ^ { - 1 } ( \cdot )$ denote the PReLU and its inverse, and $\gamma$ is a learnable negative-slope parameter that controls the gradient magnitude. This regulation prevents divergence, stabilizing the optimization process across the residual field.

While z can now take negative values, standard alpha blending still constrains the accumulation to be non-negative for physical plausibility. Thus, we modify

the rendering formulation to permit negative contributions within the blending process. Following $\mathrm { R ^ { 2 }  â G S }$ , we model the linear integral of X-ray projection without the standard transmittance term:

$$
C = \sum _ { i = 1 } ^ { N } \alpha _ { i } , \quad \mathrm { s u b j e c t ~ t o } \ ( \alpha _ { i } \geq \epsilon ) ,\tag{4}
$$

where $\alpha _ { i }$ and Ïµ denote the i-th Gaussianâs contribution and a heuristic threshold, respectively. To actively accommodate negative values, we specifically remove the strict non-negativity filter $( \alpha _ { i } \geq \epsilon )$ , conventionally imposed to ensure physical plausibility, from Eq. (4). Despite this change, since our formulation is a linear accumulation, the gradient remains constant $\begin{array} { r } { \big ( \frac { \partial C } { \partial \alpha _ { i } } = 1 \big ) } \end{array}$ regardless of whether $\alpha _ { i }$ is positive or negative, ensuring stable optimization. Accordingly, we adapt our pruning process, discarding Gaussians only $\mathrm { i f } \ | z | < 1 0 ^ { - 5 }$ . Finally, the output HR volume is reconstructed by adding the learned residual field to the upsampled LR volume, followed by clipping (max(0, Â·)) to ensure physical plausibility. This mechanism, NAB-GS, can selectively amplify or suppress local intensities of the upsampled LR volume, successfully recovering fine structure details.

Loss function. To optimize 3D CT reconstruction, the total loss function is:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { r e c o n } + \lambda _ { 1 } \mathcal { L } _ { t v } , \mathcal { L } _ { r e c o n } = \mathcal { L } _ { 1 } ( y , x ) + \mathcal { L } _ { r e s } ( \hat { y } , \hat { x } ) + \lambda _ { 2 } \mathcal { L } _ { s s i m } ( y , x ) ,\tag{5}
$$

where y and x denote the diffusion-generated HR projection and our predicted projection, while $\hat { y }$ and xË represent residual projection $( i . e . , \ y \textrm { -- }$ upsampled projection) and residual projection. The terms $\mathcal { L } _ { 1 } ( \cdot , \cdot ) , \mathcal { L } _ { r e s } ( \cdot , \cdot ) , \mathcal { L } _ { s s i m } ( \cdot , \cdot )$ , and $\mathcal { L } _ { t v } ( \cdot )$ correspond to the L1, L1-based residual, structural similarity index measure (SSIM), and TV loss function, respectively. The hyperparameters are empirically set to $\lambda _ { 1 } = 0 . 0 5$ and $\lambda _ { 2 } = 0 . 5$ 5. Following $\mathrm { R ^ { 2 }  â G S }$ , we compute $\mathcal { L } _ { t v } ( \cdot )$ efficiently using randomly cropped $3 2 ^ { 3 }$ subvolumes during training.

## 3 Experiments

## 3.1 Setup

Dataset. [2D] We train our unconditional diffusion model on two large-scale X-ray datasets resized to 512Ã512. ChestX-ray14â  (112,120 frontal images) and CheXpertâ  (80,845 frontal and lateral images). [3D] Evaluation is performed on UHRCT [6] (20/10 train/test split) and MELA (60/20/20 train/val/test split). Volumes are resized to $5 1 2 ^ { 3 }$ , clipped to [-512, 3071], and normalized to [0, 1]. LR volumes are generated via Gaussian smoothing followed by sinc interpolation. Note that our zero-shot learning uses only test sets.

Implementation details. [Diffusion] is conducted in PyTorch 1.11.0. We trained an unconditional DDPM [12] using four NVIDIA A6000 GPUs. For inference, we used DDIM [26] sampling with 50 steps, employing DDNM+ [29] for both $4 \times$ and $8 \times \ \mathrm { S R }$ with a noise level of 0.0015. We upscaled 100 X-ray projections, uniformly spaced between $0 ^ { \circ }$ to $1 8 0 ^ { \circ }$ , from each LR volume using LR projection by TIGRE [1]. [NAB-GS] is implemented in PyTorch and trained for 5k iterations on a single RTX 3090 GPU. We adopt the default training parameter of $\mathrm { R ^ { 2 }  â G S }$ [33], except at 8Ã, where the learning rates for scaling (from $5 \times 1 0 ^ { - 4 }$ to $5 \times 1 0 ^ { - 5 } )$ and rotation (from $1 \times 1 0 ^ { - 4 }$ to $1 \times 1 0 ^ { - 5 } )$ . Our framework takes around 15 minutes per volume (Diffusion 10 min, NAB-GS 5 min). Performance is evaluated using PSNR and SSIM [31].

Table 1. Quantitative comparison of 3D CT SR with previous methods.
<table><tr><td>Methods</td><td>UHRCT 4Ã PSNRâ SSIMâ</td><td>UHRCT 8Ã PSNRâ SSIMâ</td><td>PSNRâ</td><td>MELA 4Ã SSIMâ</td><td>MELA 8Ã PSNRâ SSIMâ</td></tr><tr><td>Trilinear Cubic</td><td>24.56 0.8877 24.61 0.8783</td><td>21.14 21.46</td><td>0.8125 0.8013</td><td>33.64 0.9472 33.55 0.9431</td><td>30.27 0.9063 30.49 0.9005</td></tr><tr><td>NeRF [20] CuNeRF [4]</td><td>20.86 0.6745 25.25 0.8459</td><td>18.92</td><td>0.5797</td><td>29.76 0.8088</td><td>28.73 0.7873</td></tr><tr><td>Ours</td><td>25.42 0.8957</td><td>21.04 21.96</td><td>0.7572 0.8172</td><td>33.76 0.9096</td><td>30.11 0.8535</td></tr><tr><td></td><td></td><td></td><td></td><td>34.17 0.9525</td><td>30.81 0.9115</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Supervised (ArSSR [32])</td><td>22.78 0.8936</td><td>21.10</td><td>0.8336</td><td>33.00 0.9658</td><td>30.70 0.9343</td></tr></table>

## 3.2 Results and Evaluation

Quantitative results. Table 1 presents the comparison against conventional upsampling (Trilinear, Cubic), zero-shot (NeRF [20], CuNeRF [4]), and supervised (ArSSR [32]) methods. ArSSR was trained on the train set of each dataset; NeRF, implemented as in CuNeRF, used standard ray sampling for SR, unlike CuNeRFâs cube sampling. Overall, our method achieves superior PSNR and SSIM across both the UHRCT and MELA datasets. Notably, when compared to CuNeRF, which relies solely on internal volume information, our approach yields substantially higher SSIM (+0.04-0.06). Compared to ArSSR, our zeroshot framework demonstrates competitive performance. Furthermore, our framework is computationally efficient, completing in 15 minutes, whereas CuNeRF takes about 1 hour and ArSSR takes about 25 minutes per volume.

Qualitative results. Figure 2 visualizes the slice of reconstructed HR volumes and their corresponding L2 error maps. Cubic (b) suffers from severe over-smoothing, and CuNeRF (d) introduces noticeable high-frequency artifacts, particularly at 8Ã. In contrast, our method (e) yields more faithful structural details. Notably, at 4Ã upsampling (rows 1 and 3), ours more precisely restores fine structures such as bone boundaries. These quantitative and qualitative results clearly demonstrate the effectiveness of the proposed method.

Expertsâ evaluation. A user study with two domain experts was conducted to assess the clinical potential of the proposed method, comparing MELA (4Ã, 8Ã) volumes produced by three methods (Cubic, CuNeRF, and Ours). The experts commented that (1) our method at 4Ã shows potential for real-world clinical use, while further improvements are required at 8Ã and (2) Ours yields sharper results than Cubic, but the current visual gain does not substantially contribute to overall utility. They further noted that (3) it clearly outperforms CuNeRF by more accurately preserving fine structural details and (4) enhancing inter-slice consistency is an important direction for further improving clinical utility.

<!-- image-->  
(a)  
(b)  
(c)  
(d)  
(e)  
Fig. 2. Visual comparisons of 3D CT reconstruction results. (a) Ground truth (GT), (b) Cubic interpolation, (c) ArSSR [32], (d) CuNeRF [4], and (e) Ours. The green box and zoom-in highlight regions where our method excels at reconstruction. The error map is computed by the L2 norm between the prediction and the ground truth.

Ablation on UHRCT. (1) 2D projection SR. We evaluate the impact of 2D SR methods on 3D SR performance in Tab. 2. We compare our method against non-diffusion baselines, including bilinear interpolation, RDN [35], and Omni [27], as well as diffusion-based optimization methods, including GDP [9] and BIRD [5]. RDN and Omni were trained on projections generated from the UHRCT training set via TIGRE. RDN and Omni substantially degrade SSIM due to the scarcity of training data. While optimization-based methods (BIRD and GDP) with the same diffusion prior degrade both PSNR and SSIM, our method achieves clear gains in 3D SR performance, justifying our design choice.

Table 2. Ablation on 2D projection SR methods on 3D CT SR performance.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=5>UHRCT 4ÃPSNRâ SSIMâ</td><td rowspan=1 colspan=1>UHRCT 8ÃPSNRâ SSIMâ</td></tr><tr><td rowspan=1 colspan=1>Bilinear</td><td rowspan=1 colspan=5>24.570.8244</td><td rowspan=1 colspan=1>21.340.8015</td></tr><tr><td rowspan=1 colspan=1>RDN [35]</td><td rowspan=1 colspan=5>23.780.6410</td><td></td></tr><tr><td rowspan=1 colspan=1>Omni [27]</td><td rowspan=1 colspan=2>22</td><td></td><td rowspan=1 colspan=2>0.5621</td><td rowspan=1 colspan=1>20.560.5992</td></tr><tr><td rowspan=2 colspan=1>BIRD [5]GDP [9]</td><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=2>20.18</td><td rowspan=1 colspan=1>0.8111</td><td rowspan=2 colspan=1>19.620.784521.000.8075</td></tr><tr><td rowspan=1 colspan=5>22.840.8485</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=5>25.420.8957</td><td rowspan=1 colspan=1>21.960.8172</td></tr></table>

Table 3. Ablation on 3D reconstruction methods and activation functions.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=2>UHRCT 4ÃPSNRâSSIMâ</td><td rowspan=1 colspan=1>UHRCT 8ÃPSNRâSSIMâ</td></tr><tr><td rowspan=1 colspan=1>FDK [10]</td><td rowspan=1 colspan=2>21.880.5375</td><td rowspan=1 colspan=1>19.390.4337</td></tr><tr><td rowspan=2 colspan=1>NAF [34]SAX-NeRF [2]R2-GS [33]</td><td rowspan=1 colspan=2>24.230.8559</td><td rowspan=1 colspan=1>21.660.7982</td></tr><tr><td rowspan=1 colspan=2>24.810.871924.89 0.8839</td><td rowspan=1 colspan=1>21.400.782621.250.8056</td></tr><tr><td rowspan=1 colspan=1>SoftplusReLUSine</td><td rowspan=1 colspan=2>24.710.879024.620.878425.190.8475</td><td rowspan=1 colspan=1>21.61 0.802221.570.802121.260.7208</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=2>25.420.8957</td><td rowspan=1 colspan=1>21.960.8172</td></tr></table>

<!-- image-->  
(a)

<!-- image-->  
(b)

<!-- image-->  
(c)

<!-- image-->  
(d)

<!-- image-->  
(e)  
Fig. 3. Visual comparisons of 3D CT reconstruction across activation functions. (a) Ground truth, (b) Softplus, (c) ReLU, (d) Sine, and (e) Ours (PReLU). Ours effectively enhances structural details while suppressing grainy noise.

(2) Effectiveness of NAB-GS. Table 3 validates NAB-GS. First, we compare to existing reconstruction methods (FDK [10], NAF [34], SAX-NeRF [2], and $\mathrm { R } ^ { 2 } \cdot$ GS [33]). Specifically, ours outperforms R2-GS in both PSNR (4Ã: +0.53, 8Ã: +0.71) and SSIM $( 4 \times : + 0 . 0 1 1 8 , 8 \times : + 0 . 0 1 1 6 )$ , demonstrating the effectiveness of the residual learning-based strategy. Second, we compare alternative activations (softplus, ReLU, and sine) with PReLU. Although the raw density z can take negative values, softplus and ReLU strictly enforce non-negativity, failing to capture boundaries in overestimated regions (Fig. 3(b) and (c)). Although sine allows non-negative outputs, it introduces grainy artifacts as shown in Fig. 3(d). In contrast, PReLU leverages a learnable negative slope to accommodate negative values and achieves the best performance as in Fig. 3(e).

## 4 Conclusion

In this paper, we have proposed a novel zero-shot 3D CT SR framework that reformulates volumetric SR as a 3D reconstruction task driven by diffusionupsampled 2D projection priors. Furthermore, we introduced NAB-GS, which represents the negative densities to learn signed residual fields that enhance structural details. Our framework outperforms previous zero-shot methods in both quantitative and qualitative results on two public datasets. Moreover, an expertsâ evaluation highlights the clinical potential of the proposed method, especially at 4Ã. For future work, we will enhance inter-slice continuity for practical clinical use and conduct evaluations on real-world clinical data.

## References

1. Biguri, A., Dosanjh, M., Hancock, S., Soleimani, M.: TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction. Biomed. Phys. Eng. Express (BPEE) 2(5), 055010 (2016)

2. Cai, Y., Wang, J., Yuille, A., Zhou, Z., Wang, A.: Structure-aware sparse-view x-ray 3d reconstruction. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 11174â11183 (2024)

3. Chen, X., Wang, X., Zhou, J., Qiao, Y., Dong, C.: Activating more pixels in image super-resolution transformer. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 22367â22377 (2023)

4. Chen, Z., Yang, L., Lai, J.H., Xie, X.: CuNeRF: Cube-based neural radiance field for zero-shot medical image arbitrary-scale super resolution. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 21185â21195 (2023)

5. Chihaoui, H., Lemkhenter, A., Favaro, P.: Blind image restoration via fast diffusion inversion. Adv. Neural Inf. Process. Syst. (NeurIPS) 37, 34513â34532 (2024)

6. Chu, Y., Zhou, L., Luo, G., Qiu, Z., Gao, X.: Topology-preserving computed tomography super-resolution based on dual-stream diffusion model. In: Proc. Intâl Conf. Med. Image Comput. Comput. Assist. Interv. (MICCAI). pp. 260â270 (2023)

7. Dong, C., Loy, C.C., He, K., Tang, X.: Image super-resolution using deep convolutional networks. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 38(2), 295â307 (2015)

8. Fang, W., Tang, Y., Guo, H., Yuan, M., Mok, T.C., Yan, K., Yao, J., Chen, X., Liu, Z., Lu, L., et al.: CycleINR: Cycle implicit neural representation for arbitraryscale volumetric super-resolution of medical data. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 11631â11641 (2024)

9. Fei, B., Lyu, Z., Pan, L., Zhang, J., Yang, W., Luo, T., Zhang, B., Dai, B.: Generative diffusion prior for unified image restoration and enhancement. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 9935â9946 (2023)

10. Feldkamp, L.A., Davis, L.C., Kress, J.W.: Practical cone-beam algorithm. Journal of the Optical Society of America A 1(6), 612â619 (1984)

11. Frazer, L.L., Louis, N., Zbijewski, W., Vaishnav, J., Clark, K., Nicolella, D.P.: Super-resolution of clinical CT: Revealing microarchitecture in whole bone clinical CT image data. Bone 185, 117115 (2024)

12. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Adv. Neural Inf. Process. Syst. (NeurIPS) 33, 6840â6851 (2020)

13. Kawar, B., Elad, M., Ermon, S., Song, J.: Denoising diffusion restoration models. Adv. Neural Inf. Process. Syst. (NeurIPS) 35, 23593â23606 (2022)

14. Kerbl, B., Kopanas, G., LeimkÃ¼hler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. on Graph. (ToG) 42(4), 139â1 (2023)

15. Ledig, C., Theis, L., HuszÃ¡r, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z., et al.: Photo-realistic single image superresolution using a generative adversarial network. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 4681â4690 (2017)

16. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: SwinIR: Image restoration using swin transformer. In: Proc. of Intâl Conf. on Comput. Vis. (ICCV). pp. 1833â1844 (2021)

17. Liu, G., Sun, H., Li, J., Yin, F., Yang, Y.: Accelerating diffusion models for inverse problems through shortcut sampling. arXiv preprint arXiv:2305.16965 (2023)

18. Martin, D.R., Semelka, R.C.: Health effects of ionising radiation from diagnostic CT. The Lancet 367(9524), 1712â1714 (2006)

19. McCollough, C.H., Yu, L., Kofler, J.M., Leng, S., Zhang, Y., Li, Z., Carter, R.E.: Degradation of ct low-contrast spatial resolution due to the use of iterative reconstruction and reduced dose levels. Radiology 276(2), 499â506 (2015)

20. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99â106 (2021)

21. Peng, C., Lin, W.A., Liao, H., Chellappa, R., Zhou, S.K.: Saint: Spatially aware interpolation network for medical slice synthesis. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 7750â7759 (2020)

22. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 10684â10695 (2022)

23. Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D.J., Norouzi, M.: Image super-resolution via iterative refinement. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) 45(4), 4713â4726 (2022)

24. Shocher, A., Cohen, N., Irani, M.: Zero-shot super-resolution using deep internal learning. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 3118â3126 (2018)

25. Soh, J.W., Cho, S., Cho, N.I.: Meta-transfer learning for zero-shot super-resolution. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 3516â3525 (2020)

26. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)

27. Wang, H., Chen, X., Ni, B., Liu, Y., Liu, J.: Omni aggregation networks for lightweight image super-resolution. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 22378â22387 (2023)

28. Wang, J., Wang, R., Tao, R., Zheng, G.: UASSR: Unsupervised arbitrary scale super-resolution reconstruction of single anisotropic 3d images via disentangled representation learning. In: Proc. Intâl Conf. Med. Image Comput. Comput. Assist. Interv. (MICCAI). pp. 453â462 (2022)

29. Wang, Y., Yu, J., Zhang, J.: Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490 (2022)

30. Wang, Y., Yang, W., Chen, X., Wang, Y., Guo, L., Chau, L.P., Liu, Z., Qiao, Y., Kot, A.C., Wen, B.: SinSR: Diffusion-based image super-resolution in a single step. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 25796â25805 (2024)

31. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process. (TIP) 13(4), 600â612 (2004)

32. Wu, Q., Li, Y., Sun, Y., Zhou, Y., Wei, H., Yu, J., Zhang, Y.: An arbitrary scale super-resolution approach for 3d MR images via implicit neural representation. IEEE J. Biomed. Health Inform. (JBHI) 27(2), 1004â1015 (2022)

33. Zha, R., Lin, T.J., Cai, Y., Cao, J., Zhang, Y., Li, H.: R2-Gaussian: Rectifying radiative gaussian splatting for tomographic reconstruction. Adv. Neural Inf. Process. Syst. (NeurIPS) (2024)

34. Zha, R., Zhang, Y., Li, H.: NAF: Neural attenuation fields for sparse-view CBCT reconstruction. In: Proc. Intâl Conf. Med. Image Comput. Comput. Assist. Interv. (MICCAI). pp. 442â452 (2022)

35. Zhang, Y., Tian, Y., Kong, Y., Zhong, B., Fu, Y.: Residual dense network for image super-resolution. In: Proc. Comput. Vis. Pattern Recognit. (CVPR). pp. 2472â2481 (2018)