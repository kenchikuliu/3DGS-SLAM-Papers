# UGOD: Uncertainty-Guided Differentiable Opacity and Soft Dropout for Enhanced Sparse-View 3DGS

Zhihao Guo1, Peng Wang1\*, Zidong Chen2, Xiangyu Kong3, Yan Lyu4, Guanyu Gao5, Liangxiu Han1

1Manchester Metropolitan University 2Imperial College London 3University of Exeter 4Southeast University 5Nanjing University of Science and Technology

## Abstract

3D Gaussian Splatting (3DGS) has become a competitive approach for novel view synthesis (NVS) due to its advanced rendering efficiency through 3D Gaussian projection and blending. However, Gaussians are treated equally weighted for rendering in most 3DGS methods, making them prone to overfitting, which is particularly the case in sparse-view scenarios. To address this, we investigate how adaptive weighting of Gaussians affects rendering quality, which is characterised by learned uncertainties proposed. This learned uncertainty serves two key purposes: first, it guides the differentiable update of Gaussian opacity while preserving the 3DGS pipeline integrity; second, the uncertainty undergoes soft differentiable dropout regularisation, which strategically transforms the original uncertainty into continuous drop probabilities that govern the final Gaussian projection and blending process for rendering. Extensive experimental results over widely adopted datasets demonstrate that our method outperforms rivals in sparse-view 3D synthesis, achieving higher quality reconstruction with fewer Gaussians in most datasets compared to existing sparse-view approaches, e.g., compared to DropGaussian, our method achieves 3.27% PSNR improvements on the MipNeRF 360 dataset.

## Introduction

From 2D images as input to generate continuous high fidelity 3D reconstructed environments, which we also call it Novel View Synthetic (NVS), can contribute greatly to techniques like Digital Twinning, AR/VR and robotic embodiment (Wang et al. 2024c; Fei et al. 2024; Xiong et al. 2024; Wang et al. 2024a). 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023a) has emerged as a competitive approach for achieving real-time yet high-fidelity 3D scene synthesis. This is mostly attributed to its creative idea of representing scenes using a set of 3D Gaussians (ellipsoids) with learnable attributes like positions and colours. Once trained, these Gaussians are fed to the rasterisation-based forward rendering pipeline to be blended to yield the reconstructed scenes. Compared to classical rivals like Neural Radiance Fields (NeRF) and its variants (Mildenhall et al. 2021; Guo and Wang 2024) that rely on inefficient ray sampling, 3DGS enables efficient parallel rendering, making it a compelling choice for modern NVS applications.

However, despite its advantages, 3DGS faces challenges in handling sparse-view scenarios, where the initialisation of Gaussians from Structure-from-Motion (SfM) point clouds is often incomplete or noisy. This leads to overfitting during optimisation, where the model fits training views excessively while failing to generalise to novel (testing) views. SfM-free methods such as DUSt3R (Wang et al. 2024b) and MASt3R (Duisterhof et al. 2024) regress 3D points directly from images using pretrained models for Gaussian initialisation, which also suffer from overfitting, etc. As shown in Figure 1, the baseline 3DGS\* quickly stagnates and even declines in test PSNR due to overfitting, especially when initialised with sparse point clouds.

Another under-explored question in existing 3DGS methods is: whether all Gaussians should be treated equally during rendering once training is complete (Han and Dumery 2025; Li and Cheung 2024). In practice, Gaussians exhibit strong view-dependence: a Gaussian may appear certain from one direction but highly uncertain from another. This issue is further exacerbated in sparse-view scenarios, where the initialisation of Gaussians is often incomplete or noisy. These observations motivate a systematic investigation into how adaptive, view-dependent treatment of Gaussians impacts rendering quality.

This paper aims to address these challenges by primarily investigating how reliable each Gaussian is for rendering, which is learned and characterised by their âuncertaintyâ using a neural network module that is integrated to the 3DGS pipeline. The uncertainty refers to a per-Gaussian, viewdependent parameter, learned by the neural network conditioned on spatial features (position, rotation, scale) and the current viewing direction: a higher uncertainty indicates that the Gaussian is less reliable and thus should be handled with propotional caution, while a lower uncertainty suggests greater confidence and indicates potential of further exploitation. Our approach ensures that uncertainty is learned in a fully differentiable manner, enabling two key functionalities: (1) adaptive modulation of opacity without disrupting the integrity of the 3DGS pipeline, making our method easily reusable by the research community; (2) application of soft, differentiable dropout regularisation, which transforms the predicted uncertainty into continuous drop probabilities that govern the final Gaussian projection and blending process. In summary, our contributions are as fol-

<!-- image-->  
(a) Sparse initialisation leads to Overfitting

<!-- image-->  
(b) Our Approach avoids Overfitting

<!-- image-->  
Figure 1: Peak Signal-to-Noise Ratio (PSNR) over iterations on the Mip-NeRF 360 Bicycle scene (24,970 initial Gaussians) and Tanks and Temples Truck scene (66,674 initial Gaussians). (a) Sparse initialisation leads to rapid overfitting and stagnant testing PSNR in baseline 3DGS\*, while denser initialisation enables continued improvement. (b) The proposed work effectively suppress overfitting and improve generalisation, even under sparse initialisation. The bottom qualitative comparisons show that our approach yields better visual quality and higher PSNR than rivals.

## lows:

â¢ Gaussians Uncertainty Learning: We propose to learn an uncertainty for each Gaussian based on its spatial properties (position, scale, rotation) and the current viewing direction. This uncertainty captures per-Gaussian ambiguity within a view-dependent context, aligning with the fact that perceptions are view-dependent.

â¢ Uncertainty-Guided Opacity Modulation: We utilise the learned uncertainty to modulate the opacity, which plays a critical role in rendering quality. In comparison, the opacity in literature has been treated as fixed when learned, which overlooks view-dependent unceratainty of Gaussians undermine the rendering quality.

â¢ Uncertainty-Guided Differentiable Soft Dropout: We introduce an uncertainty-guided soft dropout module guided by the learned uncertainty, which drops Gaussians with high uncertainty softly to further improve rendering quality. Together our work suppresses overfitting and improves rendering quality in sparse-view 3D reconstruction.

## Related works

## 3D Gaussian Splatting

3DGS has rapidly become a leading approach for NVS due to the tradeoff between rendering efficiency and quality. By representing scenes as collections of 3D Gaussians, rendering can be achieved by simply projects them onto the 2D image plane. This process, combined with depth sorting and Î±-blending, enables efficient and high-fidelity scene reconstruction in real time. Recent works have further improved 3DGS by addressing camera pose sensitivity (Yu et al. 2024) and refining point management for enhanced rendering quality (Yang et al. 2024; Zhang et al. 2024b; Bulo, Porzi, and \` Kontschieder 2024).

However, opacity modeling in 3DGS remains relatively underexplored, despite its critical role in accurate scene reconstruction and generalisation. Most existing methods treat opacity as a fixed or independently optimised parameter, overlooking its geometric and view-dependent nature. Only a handful of studies have investigated opacity optimization for 3DGS: Celarek (Celarek et al. 2025) provides a mathematical analysis of opacity-based versus extinction-based formulations in 3DGS and volumetric rendering. Talegaonkar (Talegaonkar et al. 2024) proposes volumetrically consistent 3D Gaussian rasterisation, which improves opacity computation by integrating 1D Gaussian densities along the ray. OMG (Yong et al. 2025) introduces material-aware opacity modeling, linking opacity to material properties such as albedo and roughness. Nevertheless, these approaches do not explicitly model the geometric or viewdependent factors that influence opacity, leaving a gap in fully leveraging opacity for robust and generalisable 3DGS rendering.

## Sparse 3DGS Reconstruction

While 3DGS has achieved remarkable rendering quality, its performance is highly dependent on dense input views for reliable Gaussian initialisation. Practically, dense view can be challenging, resulting in incomplete or noisy initialisation of Gaussians, which lead to model overfitting and undermines the performance of 3DGS in sparse-vew scenarios.

To address these challenges, various methods have been proposed for sparse-view 3DGS. DropGaussian (Park, Ryu, and Kim 2025) combats overfitting by selectively dropping low-contributing Gaussians, allowing the remaining ones to receive stronger gradients and contribute more effectively to optimization. CoR-GS (Zhang et al. 2024a) introduces point disagreement and rendering disagreement to quantify geometric and appearance inconsistencies between reconstructions. They find these metrics negatively correlate with reconstruction quality, making them useful for quality assessment. To reduce these disagreements, they introduce co-pruning for geometry refinement and pseudo-view coregularization for appearance consistence. Although these approaches improve rendering under sparse supervision, they also introduce new limitations: some rely on external priors such as pretrained depth or diffusion models, increasing system complexity and computational requirements; others employ heuristic or non-differentiable dropout strategies, which restrict end-to-end training and adaptability to view-dependent uncertainty.

This work addresses sparse-view 3DGS from a new perspective by introducing Uncertainty-Guided Differentiable Soft Dropout directly into the splatting pipeline. Our approach is fully end-to-end trainable and does not require semantic labels or external pretrained models. The core idea is that not all Gaussians contribute equally to rendering quality, especially when initialised from sparse or noisy inputs, and their influence should be adaptively modulated based on view-dependent uncertainty.

## Methodology

## Preliminaries

3D Gaussian Splatting 3DGS normally initialises Gaussians using point clouds like those from SfM, and Gaussians are defined by parameters like positions, rotations, scales, covariances, opacities, and colours. Representing the position of a Gaussian as x $\in \mathbb { R } ^ { 3 \times 1 }$ , we define a 3D Gaussian G as follows:

$$
G ( \mathbf { x } ) = e ^ { - \frac { 1 } { 2 } ( \mathbf { x } - \pmb { \mu } ) ^ { T } \pmb { \Sigma } ^ { - 1 } ( \mathbf { x } - \pmb { \mu } ) } ,\tag{1}
$$

where $\pmb { \mu } \in \mathbb { R } ^ { 3 \times 1 }$ is the mean position, $ { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ is the covariance matrix that is positive semi-definite, which can be factored as $\mathbf { \Sigma } \mathbf { \Sigma } = \mathbf { R } \mathbf { S } \mathbf { S } ^ { T } \mathbf { R } ^ { T }$ , where $\mathbf { R } \in \mathbb { R } ^ { 3 \times 3 }$ is an orthogonal rotation matrix, and $ { \mathbf { S } } \in \mathbb { R } ^ { 3 \times 3 }$ is a diagonal scale matrix. 3D Gaussians will be projected to 2D image space using the splatting-based rasterisation technique (Zwicker et al. 2001). Specifically, the transformation for projecting a 3D Gaussian onto the 2D image plane is approximated using a first-order Taylor expansion of the projection function at the Gaussianâs mean, expressed in the camera coordinate frame. The 2D covariance matrix Î£â², which describes the elliptical shape of each Gaussian in the image space, is then computed as:

$$
\begin{array} { r } { \pmb { \Sigma ^ { \prime } } = \mathbf { J } \mathbf { W } \pmb { \Sigma } \mathbf { W } ^ { T } \mathbf { J } ^ { T } , } \end{array}\tag{2}
$$

where J is the Jacobian of the affine approximation of the projective transformation, and W denotes the view transformation matrix. The colour of each pixel is calculated by blending sorted Gaussians based on the opacity Î±:

$$
c = \sum _ { i = 1 } ^ { n } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \right) ,\tag{3}
$$

where n is the number of points, $c _ { i }$ is the color of the $i - t h$ point, $\alpha _ { i }$ can be obtained by evaluating a projected 2D Gaussian with covariance Î£â² multiplied with a learned opacity for each point.

Opacity in 3DGS In vanilla 3DGS, opacity Î± is treated as an independent parameter, optimised alongside other Gaussian parameters like positions and colours. Once trained, all Gaussian opacities will be equally used for rendering. In addition, opacity plays a key role in regulating the density of 3D Gaussians during training, to make sure Gaussians with very low Î± values are pruned, whereas remaining Gaussians are adjusted to refine the scene reconstruction.

While this opacity-driven density control enhances both the efficiency and quality of the final rendered output, the opacity is not explicitly modelled in 3DGS. This overlooks the fact that not all Gaussians contribute equally to rendering quality, especially when initialised from sparse or noisy inputs. We will address this limitation by introducing uncertainty modelling into the 3DGS pipeline, which will be discussed in the next section.

## Gaussians Uncertainty Learning

Uncertainty Learning Module We introduce and integrate a neural network into the 3DGS pipeline, which takes as input spatial features (position, rotation, scale) and the current viewing direction, and outputs a per-Gaussian uncertainty. This uncertainty is defined as view-dependent to capture the reliability of each Gaussian for rendering from a specific viewpoint. Essentially, the Gaussian uncertainty learning process can be formulated as

$$
\mathcal { F } _ { \mathrm { M L P } } ( \mathbf { I } _ { i } ; \boldsymbol { \Theta } ) = \mathbf { u } ,\tag{4}
$$

where $\mathcal { F } _ { \mathrm { M L P } }$ is the neural network parameterised by Î, which is optimised during training by minimising the difference between the rendered images and the reference images along with other parameters, u is a vector with entry $u _ { i } \in ( 0 , 1 )$ represents the uncertainty of the iâth Gaussian $G _ { i } ,$ , and $\mathbf { I } _ { i }$ wraps up neural network inputs and takes the form

<!-- image-->  
Figure 2: Overview of our proposed framework: Uncertainty-Guided Differentiable Opacity Modulation and Soft Dropout: (a) Gaussians Uncertainty Learning: we predict per-Gaussian view-dependent uncertainty using a proposed neural network conditioned on view direction and spatial features. (b) Uncertainty-Guided Opacity Modulation: The predicted uncertainty $u _ { i }$ is used to modulate the learned opacity $\alpha _ { i }$ via $\tilde { \alpha } _ { i } = \alpha _ { i } \cdot ( 1 - u _ { i } )$ , reducing the contribution of ambiguous Gaussians during rendering. (c) Uncertainty-Guided Soft Dropout: To further suppress overfitting, we apply a differentiable soft dropout mechanism, using uncertainty to govern the soft dropout of each Gaussian. We also freeze the uncertainty learning neural network after overfitting is detected to ensure stability.

$$
\mathbf { I } _ { i } = \left[ \mathbf { P } _ { i } ^ { T } , \mathbf { V } _ { i } ^ { T } , \mathbf { R } _ { i } ^ { T } , \mathbf { S } _ { i } ^ { T } \right] ^ { T } \in \mathbb { R } ^ { 1 3 \times 1 } ,\tag{5}
$$

which concatantes position $\mathbf { P } _ { i } \in \mathbb { R } ^ { 3 \times 1 }$ , view direction as a unit vector $\mathbf { V } _ { i } \in \bar { \mathbb { R } } ^ { 3 \times 1 }$ from the camera centre to $\mathbf { P } _ { i } ,$ the scaling factor $\mathbf { \bar { S } } _ { i } \in \mathbb { R } ^ { 3 \times 1 }$ 1, and the rotation represented by a quaternion $\mathbf { R } _ { i } \in \mathbb { R } ^ { 4 \times 1 }$

It is worth noting that the low dimensionality of $\mathbf { I } _ { i }$ fundamentally limits the neural networkâs representational capacity, constraining its ability to learn complex uncertainty patterns. This limitation is exacerbated by the spectral bias of neural networks (Rahaman et al. 2019), which favor lowfrequency functions and struggle with high-frequency scene details.

Input Ecoding To overcome these constraints, we employ multilevel HashGrid encoding (Muller et al. 2022) to Â¨ transform $\mathbf { I } _ { i }$ into a high-dimensional feature space. This dimensional expansion enables discrimination of subtle spatial variations that are indistinguishable in low-dimensional space. The high-dimensional representation makes complex patterns more linearly separable, allowing the neural network to capture both coarse and fine-scale details simultaneously. While one can encode the whole $\mathbf { I } _ { i }$ using a single HashGrid, we find that encoding the position $\mathbf { P } _ { i }$ separately yields better performance. We attribute this to the fact that positions of Gaussians tend to change smoothly (low frequency), and encode them to be high frequency can help better capture uncertainty.

To be specific, the position $\mathbf { P } _ { i }$ of Gaussian $G _ { i }$ is encoded using a multilevel HashGrid encoder $H ( \cdot )$ with L levels $( L = 6$ in our implementation) and $F = 4$ features per level, yielding a 24-dimensional embedding. With the position encoding, Equation (5) can be rewritten as:

$$
\mathbf { I } _ { i } = \left[ H ( \mathbf { P } ) _ { i } ^ { T } , \mathbf { V } _ { i } ^ { T } , \mathbf { R } _ { i } ^ { T } , \mathbf { S } _ { i } ^ { T } \right] ^ { T } \in \mathbb { R } ^ { 3 4 \times 1 } ,\tag{6}
$$

with

$$
H ( \mathbf { P } _ { i } ) = \oplus _ { l = 1 } ^ { L } \operatorname { I n t e r p } \left( T _ { l } , \phi _ { l } ( \mathbf { P } _ { i } \cdot r _ { l } ) \right) \in \mathbb { R } ^ { 2 4 \times 1 } ,\tag{7}
$$

where â denotes concatenation, Tl is the learnable feature table at level $l , ~ \phi _ { l } ( \cdot )$ is a spatial hash function mapping scaled 3D coordinates to table indices, and Interp(Â·) performs trilinear interpolation over lattice vertices. The resolution rl at each level increases geometrically as $\begin{array} { r l } { r _ { l } } & { { } = } \end{array}$ $r _ { \mathrm { b a s e } } \cdot b ^ { l - \mathrm { \bar { 1 } } }$ , where $b > 1$ is the per-level scaling factor. This encoding captures both low and high frequency spatial details efficiently.

## Uncertainty-Guided Differentiable Opacity Modulation and Soft Dropout

Our movitation of modelling Gaussian uncertainty is to suppress overfitting and improve the rendering quality of 3DGS, especially under sparse-view conditions. To achieve this, we propose two key mechanisms: Opacity Modulation and

<!-- image-->  
Figure 3: Soft dropout probability Ï as a function of the learned uncertainty $u ,$ with temperature $\tau = 0 . 1$ and clamping range $[ \omega _ { \mathrm { m i n } } , \dot { \omega _ { \mathrm { m a x } } } ] = [ 0 . 2 , \dot { 0 . 8 } ]$ . When $u _ { i } \approx 0 . 5 ,$ Gaussians are most ambiguous and softly suppressed to reduce overfitting. For u outside [0.47, 0.53], clamping keeps Ï stable, preserving gradient flow and allowing informative Gaussians to continuously contribute.

Uncertainty-Guided Soft Dropout. These mechanisms leverage the learned uncertainty to adaptively control Gaussian contributions during rendering, thereby enhancing robustness and generalisation.

Opacity Modulation The learned uncertainty $u _ { i } \in ( 0 , 1 )$ of Gaussian $G _ { i }$ is used to modulate the opacity $\alpha _ { i }$ before rendering. Specifically, we compute the updated opacity of $G _ { i } { \mathrm { ~ a s } } { \mathrm { : } }$

$$
\tilde { \alpha } _ { i } = \alpha _ { i } \cdot ( 1 - u _ { i } ) .\tag{8}
$$

This formulation acts as a soft gating mechanism, to ensure Gaussians with high uncertainty (i.e., low confidence from the current view direction) wonât contribute as significantly to the rendered outputs as Gaussian with low uncertainty. By modulating the opacity according to the learned uncertainty, we suppress the influence of unreliable Gaussians.

Uncertainty-Guided Soft Dropout To further regularise learning and reduce overfitting, we propose a differentiable soft dropout that is also governed by uncertainty. Inspired by the concrete distribution (Gal, Hron, and Kendall 2017), we generate a continuous dropout probability $\omega _ { i } \in \left( 0 , 1 \right)$ per Gaussian $G _ { i }$ via:

$$
\omega _ { i } = 1 - \mathrm { s i g m o i d } \Bigl ( \frac { 1 } { \tau } \cdot \bigl ( \log \frac { u _ { i } } { 1 - u _ { i } } + \log \frac { q _ { i } } { 1 - q _ { i } } \bigr ) \Bigr ) ,\tag{9}
$$

where $q _ { i } \sim \mathcal { U } ( 0 , 1 )$ is a random variable sampled from a uniform distribution for each Gaussian, which introduces controlled randomness in the dropout process, and $\tau > 0$ is a temperature hyperparameter (we use $\tau = 0 . 1 )$ . This produces a smooth, stochastic dropout probability $\omega _ { i }$ that softly âdropsâ unreliable Gaussians while retaining gradient flow.

To ensure numerical stability and prevent collapse, we clamp the dropout probability $\omega _ { i }$ following

$$
\tilde { \omega } _ { i } = \mathrm { c l a m p } ( \omega _ { i } , \omega _ { \mathrm { m i n } } , \omega _ { \mathrm { m a x } } ) ,\tag{10}
$$

with $[ \omega _ { \mathrm { m i n } } , \omega _ { \mathrm { m a x } } ]$ the range to clamp $\omega _ { i }$ to. This clamping avoids extreme dropout values (too high or too low) that could lead to vanishing gradients or excessive suppression of potentially informative Gaussians. The final effective opacity $\bar { \alpha } _ { i }$ for Gaussian $G _ { i }$ is then computed as:

$$
\bar { \alpha } _ { i } = \tilde { \alpha } _ { i } \cdot \tilde { \omega } _ { i } .\tag{11}
$$

We set $\omega _ { \mathrm { m i n } } = 0 . 2$ and $\omega _ { \mathrm { m a x } } = 0 . 8$ in our experiments, as shown in Fig. 3. This clamping ensures that the dropout probability $\omega _ { i }$ remains within a stable range, corresponding to uncertainty values $u _ { i } \in [ 0 . 4 7 , 0 . 5 3 ]$ . The insights behind are: when the learned uncertainty of a Gaussian is around 0.5, it is most ambiguous, i.e., neither clearly reliable nor unreliable, and is thus softly suppressed (partially dropped) to reduce its impact and mitigate overfitting. For Gaussians with uncertainty beyond [0.47, 0.53], i.e., Ïi approaches the higher (1) or lower (0) bounds, we use the clamping machnisms shown in Equation (10) to prevent the probability from becoming too big or small, maintaining gradient flow and allowing potentially informative Gaussians to continue contributing to learning.

## 3DGS Training Loss

To train our uncertainty-guided 3DGS framework, we use the final effective opacity $\bar { \alpha } _ { i }$ (Eq. 11) for rendering novel views ${ \hat { I } } ,$ which are compared against ground-truth images I. Our objective is to jointly optimise for pixel-level accuracy and perceptual quality. Specifically, we employ a composite colour reconstruction loss that combines the mean absolute error (L1 loss) and a differentiable SSIM loss (D-SSIM) to encourage both sharpness and structural consistency:

$$
\mathcal { L } _ { \mathrm { c o l o u r } } = \mathcal { L } _ { 1 } ( \hat { I } , I ) + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } ( \hat { I } , I ) ,\tag{12}
$$

where $\mathcal { L } _ { 1 }$ is the pixel-wise L1 loss, LD-SSIM is the differentiable SSIM loss, and Î» is a balancing hyperparameter (set to 0.2 in our experiments).

To further mitigate overfitting, we monitor the PSNR improvement during training and freeze the uncertainty learning neural network once the improvement falls below a threshold Ïµ. This stabilises uncertainty estimation and prevents excessive adaptation to the training views.

## Experiments

## Dataset and Implementation

Dataset We test our methods on widely adopted datasets for 3DGS, including: 1) MiPNeRF 360 (Barron et al. 2022) that provides real world scenes. This dataset better reflects real world small-scale reconstruction tasks. We take 24 views (aligning with sparse-view setting) for training and the rest for testing. 2) MVimgNet (Yu et al. 2023), which is a largescale dataset of multi-view images. We also use 24 input views for training and others for testing.

Implementation All experiments are conducted using the PyTorch framework on NVIDIA RTX A100. Our modules are integrated with the 3DGS\* and is adapted for sparseview reconstrubtion. The training iteration for our method and the benchmark methods is set to 6,000 following Drop-Gaussian for fair comparison. We set the HashGrid encoding configuration to (6, 0, 0, 0) for position, view direction, scale, and rotation respectively. The soft dropout temperature Ï is set to 0.2 for Mip-NeRF 360 and 0.1 for MVImgNet. The uncertainty learning neural network is frozen when the PSNR improvement âPSNR falls below the threshold $\epsilon = 0 . 2 .$ . Notably, we do not apply the opacity reset mechanism as used in original 3DGS-based methods because our uncertainty-guided opacity modulation and soft dropout already prevent opacity oversaturation and overfitting. These mechanisms dynamically downweight unreliable Gaussians during training, making periodic opacity resets unnecessary for convergence or stability.

Algorithm 1: Uncertainty-Guided Differentiable Opacity   
Modulation and Soft Dropout   
Require: 3D Gaussians $\left\{ G _ { i } \right\}$ , each with position $\mathbf { P } _ { i } ,$ view   
direction $\mathbf { V } _ { i } ,$ scale $\mathbf { S } _ { i } ,$ , rotation $\mathbf { R } _ { i } ,$ , and opacity $\alpha _ { i }$   
1: for each Gaussian $G _ { i }$ do   
2: Encode $\mathbf { P } _ { i }$ using multilevel HashGrid $( \mathrm { E q . } 7 )$   
3: Form feature vector $\mathbf { I } _ { i } = [ H ( \mathbf { P } _ { i } ) ^ { T } , \mathbf { V } _ { i } ^ { \hat { T } } , \bar { \mathbf { R } } _ { i } ^ { T } , \mathbf { S } _ { i } ^ { T } ] ^ { T }$   
4: Predict uncertainty $u _ { i } = \mathcal { F } _ { \mathrm { M L P } } ( \mathbf { I } _ { i } ; \Theta )$   
5: if PSNR improvement $\Delta \mathrm { P S N R } { } < \epsilon$ then   
6: Freeze MLP parameters Î to stabilise uncertainty   
learning   
7: end if   
8: Compute modulated opacity $\tilde { \alpha } _ { i } = \alpha _ { i } \cdot ( 1 - u _ { i } )$   
9: Sample $q _ { i } \sim \mathcal { U } ( 0 , 1 )$   
10: Compute soft dropout probability $\omega _ { i }$ via Concrete   
distribution (Eq. 9)   
11: Clamp mask: $\tilde { \omega } _ { i } =$ clamp $\left( \omega _ { i } , \omega _ { \operatorname* { m i n } } , \omega _ { \operatorname* { m a x } } \right)$   
12: Final opacity: $\bar { \alpha } _ { i } = \tilde { \alpha } _ { i } \cdot \tilde { \omega } _ { i }$   
13: end for   
14: Render image $\hat { I }$ using $\left\{ \bar { \alpha } _ { i } \right\}$ and Gaussian parameters   
15: Compute colour loss $\mathcal { L } _ { \mathrm { c o l o u r } }$ (Eq. 12)

Metrics We compare our method against state-of-theart sparse-view NVS approaches based on commonly used metrics: PSNR, Structural Similarity Index Measure (SSIM) (Wang et al. 2004), and Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al. 2018) on the rendered images in the test views. Our method outperforms rivals on most scenes across all three metrics, demonstrating improvements in both pixel-level accuracy (PSNR) and perceptual quality (SSIM & LPIPS).

## Sparse-View Synthesis Results

Quantitative Results The evaluation results of our method and the benchmarking methods on Mip-NeRF 360 and MVImgNet datasets are shown in Table 1. We can see that our method achieves the best or second-best results in all scenes across all metrics. For example, as show in $\mathsf { A p - }$ pendix Table 4 in the âbonsaiâ scene, we improve PSNR from 21.79 (3DGS), 22.01 (CoR-GS) and 21.79 (DropGaussian) to 22.27, and improve SSIM from 0.80 and 0.79 to 0.81, while using the fewest 3D Gaussians. Similarly, in the stump scene, we achieve the best PSNR (15.31) and SSIM (0.36), while maintaining a compact representation using the fewest 3D Gaussians. On the MVImgNet dataset (Appendix Table 5), our method also achieves top performance in all scenes, where we consistently outperform both baselines in PSNR and LPIPS while maintaining efficient Gaussian numbers. These results confirm that our method achieves higher fidelity, better perceptual quality, and improved compactness across diverse and complex scenes.

Qualitative Results As shown in Figure 4, our method consistently achieves higher visual fidelity compared to 3DGS\*, DropGaussian and CoR-GS across a range of scenes from the Mip-NeRF 360 dataset. For instance, in the âbicycleâ scene, our method reconstructs sharper edges and cleaner silhouettes, while DropGaussian and CoR-GS tends to over-blur occluded or uncertain regions. As shown in Figure 4, in the âkitchenâ and âbonsaiâ scenes, our approach yields more complete structures and visually plausible textures without the opacity holes and noisy blending observed in other methods, and our method reconstructs fine-grained structures and reflective regions more faithfully than 3DGS\*, DropGaussian, and CoR-GS. For example, in the âcarâ scene, the specular highlights and contours around the wind shield and hood are better preserved, while DropGaussian exhibits blur and distortion. These improvements demonstrate the effectiveness of our uncertaintyguided opacity modulation and soft dropout design.

## Ablation Study

Ablation studies are conducted to investigate the impact of HashGrid encoding configuration, Gaussians uncertainty on our method. We also test our method under dense views to show it can cope with dense view scenarios.

HashGrid encoding configuration We conduct an ablation study on the MipNeRF 360 âkitchenâ scene under sparse view settings, as shown in Table 2. Each configuration is represented as (P, S, R, V), denoting the number of encoding dimensions allocated to position P, scale S, rotation R, and view direction V, respectively. We observe that encoding only the position (e.g., (6, 0, 0, 0)) achieves the best rendering quality (PSNR: 19.15, SSIM: 0.66, LPIPS: 0.39), while adding scale, rotation, or view direction consistently degrades performance. This can be explained as HashGrid encoding is specifically designed to capture static, spatially local patterns across multiple scales, making it particularly effective for representing 3D positions. In contrast, view direction is a dynamic, per-ray input that lacks spatial coherence, encoding it using spatial HashGrids introduces aliasing and inconsistent gradients, leading to unstable training. Moreover, in 3DGS framework, both rotation and scaling are already explicitly modelled and directly applied during the anisotropic splatting process. Encoding them again is not only redundant but also potentially harmful, as it entangles global transformation parameters with spatial encodings, disrupting learning dynamics.

Gaussians Uncertainty Analyse While an ideal mapping from uncertainty $u _ { i }$ to soft dropout probability $\omega _ { i }$ follows a sigmoid-shaped function (Fig. 3), we observe in practice that by introducing a temperature Ï and some ramdomness through $q _ { i } ~ \sim \mathcal { U } ( 0 , 1 )$ into Equation (9), we can achieve better overall performance as indicated by Figure 5. We attribute this to the fact that when the temperature Ï is small $( \mathrm { e . g . , } \tau = 0 . 1 $ , the smaller Ï is the steeper sigmoid function is), even minor fluctuations in ui can shift the resulting Ïi sharply toward either extreme after clamping. This is particularly helpful when the majority of Gaussians are ambiguous $( \mathrm { i . e . , } u _ { i } \approx 0 . 5 )$ , as it allows us to randomly classify them as certain or uncertain, which is beneficial for training.

<table><tr><td rowspan="2">Methods</td><td colspan="4">Mip-NeRF360</td><td colspan="4">MVImgNet</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td></tr><tr><td>3DGS* (Kerbl et al. 2023b)</td><td>18.42</td><td>0.56</td><td>0.44</td><td>1040939</td><td>25.77</td><td>0.85</td><td>0.18</td><td>1536370</td></tr><tr><td>DropGaussian (Park, Ryu, and Kim 2025)</td><td>18.34</td><td>0.55</td><td>0.47</td><td>669839</td><td>25.35</td><td>0.83</td><td>0.22</td><td>972767</td></tr><tr><td>CoR-GS (Zhang et al. 2024a)</td><td>18.72</td><td>0.55</td><td>0.45</td><td>865395</td><td>25.54</td><td>0.84</td><td>0.20</td><td>1269568</td></tr><tr><td>Ours</td><td>18.94</td><td>0.57</td><td>0.44</td><td>878194</td><td>26.02</td><td>0.85</td><td>0.17</td><td>1467984</td></tr></table>

Table 1: Comparison of baseline methods and our method on the Mip-NeRF360 and MVImgNet datasets. Bold denotes best, underline denotes second-best. Our method consistently achieves the best sparse rendering quality while maintaining a compact Gaussian representation. More results can be found in Appendix.

<!-- image-->  
Figure 4: Qualitative comparison of NVS on the Mip-NeRF 360 and MVImgNet datasets. We compare our method with 3DGS\*, DropGaussian and CoR-GS across multiple challenging scenes: bicycle, kitchen, and bonsai from MipNeRF 360, and car from MVImgNet. Our method produces more faithful geometry and preserves sharper structural details (e.g., the bicycle frame, excavator body, and flower arrangement) while mitigating artifacts and overfitting. Notably, DropGaussian struggles with overblurring in occluded regions (e.g., excavator centre), and 3DGS\* suffers from noisy or incomplete opacity modelling. Our uncertainty-guided opacity and soft dropout allows better generalisation across both simple scenes (e.g., sparse backgrounds) and complex scenes with clutter, occlusion, or thin structures.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 5: Illustration of uncertainty, soft dropout probability, and their relationships. From left to right, 1) the first figure shows the uncertainty histogram at the initial stage, where the majority of Gaussians are ambiguous (uncertainty around 0.5); 2) these Gaussians will be assigned to be either certain or uncertain by Equation (9), and the rationale is given that Gaussians are ambiguous, one can randomly classify them as certain or uncertain as shown in the second figure; 3) The third figure shows the uncertainty distribution when training is complete, where the uncertainty of most Gaussians converges to 0, meaning we have got low uncertainty Gaussians that we can rely on for rendering.

<table><tr><td>(P, S, R, V)</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>(5,0,0,0)</td><td>18.88</td><td>0.64</td><td>0.41</td></tr><tr><td>(5,0,0,1)</td><td>18.50</td><td>0.64</td><td>0.41</td></tr><tr><td>(6,0,0,0)</td><td>19.15</td><td>0.66</td><td>0.39</td></tr><tr><td>(6,1,1,0)</td><td>18.96</td><td>0.65</td><td>0.40</td></tr><tr><td>(7,0,0,0)</td><td>18.69</td><td>0.63</td><td>0.43</td></tr></table>

Table 2: HashGrid input encoding configurations for the MipNeRF 360 kitchen scene under 24 sparse input views. Each tuple represents the dimensional allocation for position, scaling, rotation, and view direction. Bold is Best

Dense Views Study We also integrate our method to dense views to evaluate the NVS tasks. As show in Table 3, we present the comparison on the âbicycleâ scene between our method and 3DGS\* at selected training iterations. Our method demonstrates a consistent improvement in rendering quality throughout the training process, as reflected by the steadily increasing PSNR, SSIM and decreasing LPIPS values. In contrast, 3DGS\* begins to show signs of overfitting after iteration 20,000, with slight degradation in PSNR and LPIPS.

## Conclusion

We investigate and provide a solution to an under-explored question in 3DGS: if we treat Gaussians differently (compared to treating them equally in conventionally 3DGS methods) once trained, how will that affect the rendering quality? The answer to this question is of particular interests to sparse-view 3DGS as a measure and exploitation of view-dependent uncertainty is promising to suppress overfitting and improve rendering quality. We answer the question by learning a view-dependent uncertainty for each Gaussian, conditioned on its spatial properties and viewing direction. This uncertainty is then exploited in two key ways: first, to adaptively modulate opacity in a differentiable manner, thereby reducing the impact of unreliable Gaussians; and second, to drive a soft, differentiable dropout mechanism that regularises training by encouraging the model to suppress Gaussians with ambiguous contributions. Experimental results demonstrate consistent improvements in sparseview rendering quality and robustness.

<table><tr><td rowspan="2">Iter</td><td colspan="3">Ours</td><td colspan="3">3DGS*</td></tr><tr><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>5,000</td><td>18.68</td><td>0.55</td><td>0.39</td><td>18.92</td><td>0.59</td><td>0.33</td></tr><tr><td>10,000</td><td>19.42</td><td>0.59</td><td>0.33</td><td>19.39</td><td>0.63</td><td>0.28</td></tr><tr><td>15,000</td><td>19.59</td><td>0.61</td><td>0.31</td><td>19.36</td><td>0.64</td><td>0.26</td></tr><tr><td>20,000</td><td>19.73</td><td>0.65</td><td>0.24</td><td>19.33</td><td>0.64</td><td>0.25</td></tr><tr><td>25,000</td><td>19.79</td><td>0.66</td><td>0.24</td><td>19.30</td><td>0.64</td><td>0.25</td></tr><tr><td>30,000</td><td>19.83</td><td>0.66</td><td>0.23</td><td>19.28</td><td>0.64</td><td>0.24</td></tr></table>

Table 3: Evaluation on the MipNeRF 360 bicycle scene at selected iterations for Ours and 3DGS\*. Bold indicates Best.

## References

Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.; and Hedman, P. 2022. Mip-nerf 360: Unbounded antialiased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5470â5479.

Bulo, S. R.; Porzi, L.; and Kontschieder, P. 2024. Re-\` vising densification in gaussian splatting. arXiv preprint arXiv:2404.06109.

Celarek, A.; Kopanas, G.; Drettakis, G.; Wimmer, M.; and Kerbl, B. 2025. Does 3D Gaussian Splatting Need Accurate Volumetric Rendering? arXiv preprint arXiv:2502.19318.

Duisterhof, B.; Zust, L.; Weinzaepfel, P.; Leroy, V.; Cabon, Y.; and Revaud, J. 2024. MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion. arXiv preprint arXiv:2409.19152.

Fei, B.; Xu, J.; Zhang, R.; Zhou, Q.; Yang, W.; and He, Y. 2024. 3d gaussian splatting as new era: A survey. IEEE Transactions on Visualization and Computer Graphics.

Gal, Y.; Hron, J.; and Kendall, A. 2017. Concrete dropout. Advances in neural information processing systems, 30.

Guo, Z.; and Wang, P. 2024. Depth Priors in Removal Neural Radiance Fields. In Annual Conference Towards Autonomous Robotic Systems, 367â382. Springer.

Han, C.; and Dumery, C. 2025. View-Dependent Uncertainty Estimation of 3D Gaussian Splatting. arXiv preprint arXiv:2504.07370.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023a. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139â1.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023b. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4).

Li, R.; and Cheung, Y.-m. 2024. Variational multi-scale representation for estimating uncertainty in 3d gaussian splatting. Advances in Neural Information Processing Systems, 37: 87934â87958.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Muller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In- Â¨ stant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1â 15.

Park, H.; Ryu, G.; and Kim, W. 2025. Dropgaussian: Structural regularization for sparse-view gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, 21600â21609.

Rahaman, N.; Baratin, A.; Arpit, D.; Draxler, F.; Lin, M.; Hamprecht, F.; Bengio, Y.; and Courville, A. 2019. On the spectral bias of neural networks. In International conference on machine learning, 5301â5310. PMLR.

Talegaonkar, C.; Belhe, Y.; Ramamoorthi, R.; and Antipa, N. 2024. Volumetrically Consistent 3D Gaussian Rasterization. arXiv preprint arXiv:2412.03378.

Wang, P.; Guo, Z.; Sait, A. L.; and Pham, M. H. 2024a. Robot Shape and Location Retention in Video Generation Using Diffusion Models. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 7375â7382. IEEE.

Wang, S.; Leroy, V.; Cabon, Y.; Chidlovskii, B.; and Revaud, J. 2024b. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20697â20709.

Wang, S.; Zhang, J.; Wang, P.; Law, J.; Calinescu, R.; and Mihaylova, L. 2024c. A deep learning-enhanced Digital Twin framework for improving safety and reliability in humanârobot collaborative manufacturing. Robotics and computer-integrated manufacturing, 85: 102608.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600â612.

Xiong, T.; Wu, J.; He, B.; Fermuller, C.; Aloimonos, Y.; Huang, H.; and Metzler, C. 2024. Event3dgs: Event-based 3d gaussian splatting for high-speed robot egomotion. In 8th Annual Conference on Robot Learning.

Yang, H.; Zhang, C.; Wang, W.; Volino, M.; Hilton, A.; Zhang, L.; and Zhu, X. 2024. Gaussian Splatting with Localized Points Management. CoRR.

Yong, S.; Manivannan, V. N. P.; Kerbl, B.; Wan, Z.; Stepputtis, S.; Sycara, K.; and Xie, Y. 2025. OMG: Opacity Matters in Material Modeling with Gaussian Splatting. arXiv preprint arXiv:2502.10988.

Yu, X.; Xu, M.; Zhang, Y.; Liu, H.; Ye, C.; Wu, Y.; Yan, Z.; Zhu, C.; Xiong, Z.; Liang, T.; et al. 2023. Mvimgnet: A large-scale dataset of multi-view images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9150â9161.

Yu, Z.; Chen, A.; Huang, B.; Sattler, T.; and Geiger, A. 2024. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19447â19456.

Zhang, J.; Li, J.; Yu, X.; Huang, L.; Gu, L.; Zheng, J.; and Bai, X. 2024a. Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In European Conference on Computer Vision, 335â352. Springer.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586â595.

Zhang, Z.; Hu, W.; Lao, Y.; He, T.; and Zhao, H. 2024b. Pixel-gs: Density control with pixel-aware gradient for 3d gaussian splatting. arXiv preprint arXiv:2403.15530.

Zwicker, M.; Pfister, H.; Van Baar, J.; and Gross, M. 2001. Surface splatting. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, 371â378.

## Appendix

In this section, we provide the full quantitative results for all evaluated scenes across the MipNeRF 360 and MVImgNet datasets. The results include PSNR, SSIM, LPIPS, and the total number of Gaussians used for reconstruction. Our method is compared against three recent baselines: 3DGS\*, DropGaussian, and CoR-GS. We highlight the best and second-best results in bold and underline, respectively. These tables supplement the main paper by offering a comprehensive view of the per-scene performance, demonstrating that our approach consistently achieves a strong balance between sparse view rendering quality and Gaussian compactness.
<table><tr><td rowspan="2">Method</td><td colspan="4">bicycle</td><td colspan="4">kitchen</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td></tr><tr><td>3DGS* (Kerbl et al. 2023b)</td><td>15.13</td><td>0.28</td><td>0.58</td><td>1322955</td><td>18.73</td><td>0.64</td><td>0.41</td><td>663026</td></tr><tr><td>DropGaussian (Park, Ryu, and Kim 2025)</td><td>14.90</td><td>0.31</td><td>0.59</td><td>74891</td><td>18.92</td><td>0.63</td><td>0.43</td><td>44612</td></tr><tr><td>CoR-GS (Zhang et al. 2024a)</td><td>15.72</td><td>0.30</td><td>0.59</td><td>1152403</td><td>19.03 19.17</td><td>0.65</td><td>0.41</td><td>591237</td></tr><tr><td>Ours</td><td>15.99</td><td>00.31</td><td>0.58</td><td>1013827</td><td></td><td>0.65</td><td>0.40</td><td>635714</td></tr><tr><td>Method</td><td colspan="4">bonsai</td><td colspan="4">garden</td></tr><tr><td></td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td></tr><tr><td>3DGS* (Kerbl et al. 2023b)</td><td>21.79</td><td>0.80</td><td>0.31</td><td>695974</td><td>19.68</td><td>0.53</td><td>0.39</td><td>1731954</td></tr><tr><td>DropGaussian (Park, Ryu, and Kim 2025)</td><td>21.79</td><td>0.79</td><td>0.33</td><td>585740</td><td>19.48</td><td>0.47</td><td>0.47</td><td>1147084</td></tr><tr><td>CoR-GS (Zhang et al. 2024a)</td><td>22.01</td><td>0.80</td><td>0.30</td><td>584920</td><td>19.61</td><td>0.47</td><td>0.41</td><td>1401520</td></tr><tr><td>Ours</td><td>22.27</td><td>0.81</td><td>0.30</td><td>584832</td><td>19.85</td><td>0..53</td><td>0.39</td><td>1615037</td></tr><tr><td>Method</td><td colspan="4">counter</td><td colspan="4">stump</td></tr><tr><td></td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td></tr><tr><td>3DGS* (Kerbl et al. 2023b)</td><td>19.96</td><td>0.74</td><td>0.36</td><td>617813</td><td>15.22</td><td>0.35</td><td>0.57</td><td>1213913</td></tr><tr><td>DropGaussian (Park, Ryu, and Kim 2025)</td><td>20.74</td><td>0.73</td><td>0.38</td><td>500889</td><td>14.24</td><td>0.35</td><td>0.60</td><td>916739</td></tr><tr><td>CoR-GS (Zhang et al. 2024a)</td><td>20.80</td><td>0.73</td><td>0.37</td><td>523274</td><td>15.16</td><td>0.34</td><td>0.59</td><td>39018</td></tr><tr><td>Ours</td><td>21.06</td><td>0.75</td><td>0.36</td><td>515422</td><td>15.31</td><td>0.36</td><td>0.58</td><td>904329</td></tr></table>

Table 4: Comparison of various methods on MipNeRF 360 dataset. Each scene contains 24 views. Best results are in bold, and second-best results are underlined.

<table><tr><td rowspan="2">Method</td><td colspan="4">bench</td><td colspan="4">bicycle</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td></tr><tr><td>3DGS* (Kerbl et al. 2023b)</td><td>25.79</td><td>0.85</td><td>0.15</td><td>2059825</td><td>24.59</td><td>0.85</td><td>0.15</td><td>3061480</td></tr><tr><td>DropGaussian (Park, Ryu, and Kim 2025)</td><td>25.04</td><td>0.82</td><td>0.20</td><td>1257547</td><td>23.70</td><td>0.82</td><td>0.20</td><td>1791593</td></tr><tr><td>CoR-GS (Zhang et al. 2024a)</td><td>25.42</td><td>0.84</td><td>0.18</td><td>1658686</td><td>24.14 24.67</td><td>0.84</td><td>0.18</td><td>2426536</td></tr><tr><td>Ours</td><td>25.98</td><td>0.85</td><td>0.15</td><td>1930662</td><td></td><td>0.86</td><td>0.14</td><td>2818824</td></tr><tr><td rowspan="2">Method</td><td colspan="4">car</td><td colspan="4">chair</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td></tr><tr><td>3DGS* (Kerbl et al. 2023b)</td><td>28.61</td><td>0.92</td><td>0.16</td><td>397803</td><td>27.49</td><td>0.87</td><td>0.22</td><td>1160568</td></tr><tr><td>DropGaussian (Park, Ryu, and Kim 2025)</td><td>28.14</td><td>0.91</td><td>0.18</td><td>325706</td><td>27.65</td><td>0.85</td><td>0.27</td><td>770096</td></tr><tr><td>CoR-GS (Zhang et al. 2024a)</td><td>28.38</td><td>0.91</td><td>0.17</td><td>431755</td><td>27.57</td><td>0.86</td><td>0.24</td><td>965332</td></tr><tr><td>Ours</td><td>28.83</td><td>0.93</td><td>0.16</td><td>426449</td><td>28.31</td><td>0.87</td><td>0.21</td><td>1202907</td></tr><tr><td>Method</td><td colspan="4">ladder</td><td colspan="4">suv</td></tr><tr><td></td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Gaussiansâ</td></tr><tr><td>3DGS* (Kerbl et al. 2023b)</td><td>20.38</td><td>0.68</td><td>0.26</td><td>2202016</td><td>27.78</td><td>0.90</td><td>0.14</td><td>336527</td></tr><tr><td>DropGaussian (Park, Ryu, and Kim 2025)</td><td>20.39</td><td>0.67</td><td>0.29</td><td>1423981</td><td>27.18</td><td>0.90</td><td>0.16</td><td>267679</td></tr><tr><td>CoR-GS (Zhang et al. 2024a)</td><td>20.35</td><td>0.67</td><td>0.28</td><td>1812998</td><td>27.38</td><td>0.89</td><td>0.15</td><td>322103</td></tr><tr><td>Ours</td><td>20.42</td><td>0.68</td><td>0.25</td><td>2109951</td><td>27.90</td><td>0.90</td><td>0.13</td><td>319112</td></tr></table>

Table 5: Comparison of various methods on MVimgNet dataset. Each scene contains 24 views. Best results are in bold, and second-best results are underlined.