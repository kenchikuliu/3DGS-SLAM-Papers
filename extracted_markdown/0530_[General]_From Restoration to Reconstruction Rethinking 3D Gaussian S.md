# From Restoration to Reconstruction: Rethinking 3D Gaussian Splatting for Underwater Scenes

Guoxi Huang1\*, Haoran Wang1\*, Zipeng Qi2, Wenjun Lu 3, David Bull1, Nantheera Anantrasirichai1,

1 Visual Information Laboratory, University of Bristol, Bristol, UK 2 Beijing University of Aeronautics and Astronautics, Beijing, China 3 The University of Sydney

## Abstract

Underwater image degradation poses significant challenges for 3D reconstruction, where simplified physical models often fail in complex scenes. We propose R-Splatting, a unified framework that bridges underwater image restoration (UIR) with 3D Gaussian Splatting (3DGS) to improve both rendering quality and geometric fidelity. Our method integrates multiple enhanced views produced by diverse UIR models into a single reconstruction pipeline. During inference, a lightweight illumination generator samples latent codes to support diverse yet coherent renderings, while a contrastive loss ensures disentangled and stable illumination representations. Furthermore, we propose Uncertainty-Aware Opacity Optimization (UAOO), which models opacity as a stochastic function to regularize training. This suppresses abrupt gradient responses triggered by illumination variation and mitigates overfitting to noisy or view-specific artifacts. Experiments on Seathru-NeRF and our new BlueCoral3D dataset demonstrate that R-Splatting outperforms strong baselines in both rendering quality and geometric accuracy.

## 1 Introduction

Ocean exploration is gaining momentum due to its applications in underwater archaeology, geology, and marine sciences. However, it remains hindered by technical limitations, high costs, and a shortage of skilled divers. As a result, 3D scene reconstruction has become vital, enabling efficient, scalable, and remote analysis of underwater environments.

Yet, achieving accurate 3D reconstruction in underwater environments remains highly challenging due to the unique optical properties of water. Unlike terrestrial scenes, underwater imagery suffers from depth-dependent color attenuation, scattering-induced haze, and non-uniform illumination, all of which impair geometric and photometric consistency across views.

Reconstruction methods such as NeRF (Mildenhall et al. 2021; Muller et al. 2022) and 3D Gaussian Splatting (Kerbl Â¨ et al. 2023; Yu et al. 2024) typically assume clear-air conditions, and thus lack the modeling capacity to handle the complex light transport in participating media. When directly applied to underwater images, these methods often fail to recover fine structures or maintain consistent appearance, resulting in significant reconstruction artifacts.

<!-- image-->  
(b) Intra-View Lighting Inconsistency  
Figure 1: Illustration of illumination inconsistencies in underwater scenes. (a) Cross-view: the same object appears with different shading across viewpoints due to viewpointdependent lighting and refraction.(b) Intra-view: different restoration models applied to the same view introduce inconsistent illumination effects.

Recent methods (Levy et al. 2023; Li et al. 2025; Wang et al. 2025; Jiang et al. 2025a) incorporate simplified physical models into NeRF and 3DGS frameworks to account for underwater light attenuation and backscattering, thereby improving rendering quality to some extent. However, these approaches are fundamentally limited by the oversimplified assumptions in their physical modelsâfor instance, the common neglect of forward scattering (Tan 2008) and light reflection leads to failure in complex underwater scenes.

Underwater image restoration (UIR) techniques have demonstrated strong capabilities in enhancing degraded underwater images. However, their integration into 3D reconstruction pipelines remains largely unexplored. Existing neural rendering methods typically operate directly on raw underwater inputs, often ignoring degradations such as color distortion, haze, or contrast loss. As a result, reconstructions may suffer from inaccurate geometry and poor appearance fidelity.

No single UIR model is universally optimalâdifferent methods target specific degradations, such as color correction, haze removal, or contrast enhancement. Moreover, due to the ill-posed nature of underwater restoration, deep learning-based methodsâparticularly generative modelsâoften produce diverse outputs from the same input, resulting in intra-view lighting inconsistencies (see Fig. 1, bottom). Such variations can negatively impact the performance of downstream 3D reconstruction.

Training a separate 3D model for each enhanced version is not only computationally expensive but also incurs significant storage overhead. This motivates a unified framework that can jointly handle diverse restorations while maintaining geometry consistency. By leveraging complementary strengthsâe.g., better visibility, sharper textures, or corrected color balanceâwe can enrich the reconstruction process with varied enhancement perspectives that reveal more reliable geometry and appearance cues.

Meanwhile, underwater scenes often exhibit cross-view illumination inconsistencies due to view-dependent light reflectance. As shown in Fig. 1 (Top), the same object may appear differently across views depending on its orientation to the water surface and lighting. This makes it difficult for 3D models to learn consistent geometry and appearance.

To address these challenges, we propose Restoration Splatting (R-Splatting), a unified 3D reconstruction framework that integrates 3D-level uncertainty modeling and a neural field to condition illumination within the 3D Gaussian Splatting (3DGS) paradigm.

To resolve cross-view lighting inconsistencies, we introduce Uncertainty-Aware Opacity Optimization (UAOO; Sec. 3.2), which uses a random field to model per-point uncertainty. This enables our method to suppress artifacts while preserving fine-grained geometry.

To address intra-view lighting inconsistencies caused by different UIR methods, R-Splatting also incorporates a neural field that learns a conditional representation of image illumination, allowing the 3D model to adapt to diverse enhancement styles. This design supports rendering under multiple plausible lighting conditions. To further disentangle lighting cues, we employ a contrastive learning objective (Sec. 3.1) that separates appearance embeddings across styles.

By modeling both visual diversity and geometric consistency, R-Splatting bridges UIR and neural 3D reconstruction, enabling scalable, coherent scene modeling in challenging underwater environments.

## Our contributions can be summarized as follows:

â¢ We propose R-Splatting, a unified framework that fuses multiple UIR results into a single 3DGS model, enhancing reconstruction quality through diverse visual cues.

â¢ We introduce UAOO to model per-point uncertainty via stochastic opacity, improving robustness to illumination inconsistencies.

â¢ We release BlueCoral3D, a dataset with dynamic lighting, and validate our methodâs effectiveness through extensive experiments.

â¢ Our approach is flexible, supporting future UIR advances without requiring model changes.

## 2 Related Works

Underwater Image Restoration. Underwater image restoration (UIR) methods aim to address common issues in underwater photography, such as blurring effects, floating particles, and color cast, while also improving the overall visual quality by enhancing contrast and brightness. Existing UIR approaches can be broadly categorized into physical model-based methods (He, Sun, and Tang 2010; Yang et al. 2011; Chiang and Chen 2012; Wen et al. 2013; Galdran et al. 2015; Peng, Cao, and Cosman 2018; Liang et al. 2022; Zhang et al. 2025) and deep learning-based methods (Tang, Kawasaki, and Iwaguchi 2023; Li et al. 2021; Peng, Zhu, and Bian 2023; Guan et al. 2024; Huang et al. 2025a,b,c; Lin et al. 2024; Peng and Bian 2025; Malyugina et al. 2025). Recent studies demonstrate that deep learning-based methods outperform traditional physical model-based techniques in both restoration quality and robustness. Hence, in this work, we adopt off-the-shelf deep learning UIR models to enhance underwater images before further processing.

Underwater 3D Reconstruction. ScatterNeRF (Ramazzina et al. 2023) addressed participating media by separating attenuation from geometry, which applies to underwater settings. SeaThru-NeRF (Levy et al. 2023) introduced per-ray parameters based on a revised underwater formation model, later extended by SP-SeaNeRF (Chen et al. 2024) with learnable illumination. WaterNeRF (Sethuraman, Ramanagopal, and Skinner 2023) incorporated light transport and color correction via optimal transport, while WaterHE-NeRF (Zhou et al. 2025) applied a Retinex-inspired field for color compensation. For dynamic scenes, UWNeRF (Tang et al. 2024) separates motion via masks, and AquaNeRF (Gough et al. 2025) improves transmittance modeling per ray.

More recently, 3DGS has been adapted for underwater scenarios. Z-Splat (Qu et al. 2024) fuses sonar and RGB to address sparse-view issues but lacks medium modeling. RecGS (Zhang et al. 2024b) applies filtering and recurrence to improve view quality without modeling scattering. Gaussian Splashing (Mualem et al. 2024) embeds scattering directly in CUDA for efficiency. Aquatic-GS (Liu et al. 2024) and SeaSplat (Yang, Leonard, and Girdhar 2024) extend 3DGS by incorporating haze-aware image formation models. Building on this direction, UW-GS (Wang et al. 2025) and WaterSplatting (Li et al. 2025) combine physical priors with MLPs to estimate underwater attenuation coefficients for rendering restoration. SWAGSplatting (Jiang et al. 2025b) incorporates high-level semantics into the reconstruction process by leveraging Grounded-SAM (Ren et al. 2024).

However, no prior work has attempted to unify underwater image restoration and 3D Gaussian Splatting, a gap we address in this paper.

## 2.1 Preliminaries: 3DGS

3DGS (Kerbl et al. 2023) represents a 3D scene as a set of Gaussian primitives $\mathcal { G } _ { i }$ . Each Gaussian $\mathcal { G } _ { i }$ is defined by a mean position $X _ { i } ,$ a covariance matrix $\Sigma _ { i }$ , an opacity $\alpha _ { i }$ , and a view-dependent color $c _ { i }$ encoded via spherical harmonics (SH). The covariance matrix $\Sigma _ { i }$ is decomposed into a rotation matrix $R _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ and a scaling matrix $S _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ to enable differentiable optimization:

<!-- image-->  
Figure 2: Overview of the proposed R-Splatting pipeline. (1) We first apply multiple UIR models to obtain M+1 groups of enhanced images. (2) A neural field encodes each image into a latent code z, regularized by a contrastive loss to remove viewdependent cues. An illumination generator predicts z at inference. The latent is bilinearly sampled into point-wise features and fused with Gaussian attributes to produce a view-shared color. (3) An uncertainty-aware renderer models opacity as a distribution to suppress artifacts and improve multi-view consistency.

<!-- image-->  
Figure 3: Learning view-invariant latent code with dynamic latent queues. For each restoration style, the newly generated latent code is added to the front of its corresponding subqueue, while the oldest code at the end is removed to maintain a fixed queue length. We maximize the similarity between latent codes of different views under the same restoration style, and minimize the similarity between those from different restoration styles.

$$
\Sigma _ { i } = R _ { i } S _ { i } S _ { i } ^ { T } R _ { i } ^ { T } .\tag{1}
$$

During rendering, each 3D Gaussian is projected onto the image plane (Zwicker et al. 2001) using a viewing transformation $W$ . This projection yields a corresponding 2D Gaussian, whose image-space covariance matrix $\Sigma _ { i } ^ { \prime }$ is given by:

$$
\Sigma _ { i } ^ { \prime } = J W \Sigma _ { i } W ^ { T } J ^ { T } ,\tag{2}
$$

where J is the Jacobian matrix of an affine approximation to the projective transformation.

To render an image from a given camera pose, we compute the color of each pixel via alpha compositing: For each pixel, the Gaussians are sorted in front-to-back order based on their distances to the image plane. The 2D Gaussianâs mean $X _ { i } ^ { \prime }$ is obtained by projecting $X _ { i }$ into the camera coordinate system using W . Each Gaussian contributes a viewdependent color $c _ { i }$ . The final pixel color C is computed as:

$$
\alpha _ { i } ^ { \prime } = \alpha _ { i } \exp \left( - \textstyle { \frac { 1 } { 2 } } ( x - X _ { i } ^ { \prime } ) ^ { T } ( \Sigma _ { i } ^ { \prime } ) ^ { - 1 } ( x - X _ { i } ^ { \prime } ) \right) ,\tag{3}
$$

$$
C = \sum _ { i } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{4}
$$

where $\alpha _ { i }$ signify the opacity of point $i ,$ which is a learnable parameter, and $\alpha _ { i }$ represents the blending weight of the i-th Gaussian. To model view-dependent effects such as specularities, each Gaussian stores a base set of spherical harmonics (SH) coefficients, denoted as shi, which are used to compute the view-dependent color $c _ { i }$ based on the viewing direction.

## 3 Method

Given N multi-view underwater images, we apply multiple UIR models to produce M enhanced versions, resulting in M +1 image sets per view (including the original). Our goal is to reconstruct all M + 1 sets, denoted as $\{ \mathbf { \bar { \{ f } }  \{ \mathbf { I } _ { n } ^ { m } \} _ { n = 1 } ^ { N } \} _ { m = 0 } ^ { \tilde { M } } ,$ into a unified GS model. However, variations in illumination and geometry across these setsâespecially those produced by generative UIR modelsâcan introduce multiview inconsistencies, leading to blurred textures and inaccurate lighting representations. To overcome this, we propose Restoration Splatting (R-Splatting). The overall pipeline is shown in Fig. 2.

## 3.1 View-shared Color Modeling

To address intra-view lighting inconsistencies introduced by different UIR models, we integrate a neural field into

3DGS to model view-shared color. Intuitively, by assigning a shared illumination to all Gaussian primitives, we can achieve illumination-consistent rendering across views.

We first employ a tiny autoencoder (Bohan 2023) to extract a latent code z from image I, and then further map it onto feature maps F :

$$
\begin{array} { r } { \mathbf { z } = \mathcal { E } ( I ) , } \\ { F = \mathcal { D } ( \mathbf { z } ) . } \end{array}\tag{5}
$$

Then, given the known camera pose, each 3D point is projected to the 2D image space for feature sampling, as illustrated below:

$$
\begin{array} { r } { \ell _ { i } = \mathrm { B i l i n e a r S a m p l e } \left( F , \frac { 1 } { z _ { i } } \left[ \begin{array} { l } { u _ { i } } \\ { v _ { i } } \end{array} \right] \right) , } \\ { \mathrm { w h e r e } \quad \left[ \begin{array} { l } { u _ { i } } \\ { v _ { i } } \\ { z _ { i } } \end{array} \right] = P \cdot \left[ \begin{array} { l } { X _ { i } } \\ { 1 } \end{array} \right] . } \end{array}\tag{6}
$$

Each 3D point $X _ { i }$ is first converted into homogeneous coordinates and projected onto the 2D image plane using the projection matrix $P = K W$ , where K denotes the camera intrinsics. This yields a 2D coordinate $\left( { \frac { u _ { i } } { z _ { i } } } , { \frac { v _ { i } } { z _ { i } } } \right)$ . We then sample the corresponding feature $\ell _ { i }$ from the feature map $F$ at the pixel location using bilinear interpolation. Such that, each Gaussian point has its own illumination feature vector.

Subsequently, the point-wise latent code $\ell _ { i }$ is fed into a lightweight MLP, along with a learnable embedding $\mathbf { e } _ { i }$ for each Gaussian i, the 3D position $X _ { i }$ to predict the view shared color encoded as a set of SH coefficients:

$$
\pmb { v } _ { i } = \mathrm { M L P } ( \ b { \ell } _ { i } , \mathbf { e } _ { i } , X _ { i } )\tag{7}
$$

The learnable embedding $\mathbf { e } _ { i }$ plays a role analogous to parametric encodings (Muller et al. 2022), improving the expres- Â¨ siveness and capacity of a lightweight MLP without significantly increasing computational cost. The final color are then computed as:

$$
c _ { i } ( \mathbf { r } ) = \mathrm { S H } ( \mathbf { r } ; s h _ { i } + \pmb { v } _ { i } ) ,\tag{8}
$$

where $\mathrm { S H } ( \cdot )$ denotes the standard SH basis evaluation with the given coefficients.

Note that during multi-view evaluation, the entire Neural Field is executed only once to produce the view-shared color representation. Subsequently, multiple views can be rendered by simply updating the view direction $( i . e . , R _ { i }$ and Si) in the rasterizer, introducing only marginal overhead to the overall rendering latency.

Concurrent works such as (Kulhanek et al. 2024; Wang et al. 2025) also use a neural field for color prediction, but its view-dependent design requires per-view inference, leading to higher rendering latency.

Learning View-invariant Latent. When rendering multiple views from different angles, our conditional GS model uses a single latent code. This requires that the latent code be free of view-dependent information; otherwise, inconsistencies in lighting and appearance may arise across different viewpoints. However, without explicit constraints, the encoder tends to inadvertently encode view-dependent information into the latent code.

To address this issue, we introduce a contrastive loss (He et al. 2020), as depicted in Fig. 3. The goal is to pull latent codes from images with similar underwater âstylesâ together while pushing apart those from different styles, as follows:

$$
\mathcal { L } ^ { \mathrm { c o n t r a } } = - \log \frac { \sum _ { n } \exp ( \mathbf { z } _ { j } ^ { m } \cdot \mathbf { z } _ { n } ^ { m } / \tau ) } { \sum _ { m } \sum _ { n } \exp ( \mathbf { z } _ { j } ^ { m } \cdot \mathbf { z } _ { n } ^ { m } / \tau ) + \exp ( \mathbf { z } _ { j } ^ { m } \cdot \tilde { \mathbf { z } } / \tau ) } ,\tag{9}
$$

where $\tau$ is a temperature hyperparameter set to 0.07. zË denotes a latent code sampled from a random clean scene, which helps the encoder generalize to illumination conditions beyond those present in the reconstructed scene.

To provide rich sets of positive and negative examples, we maintain dynamic queues of recent latent codes for different restoration styles.

Illumination Generator. To enable diverse appearance synthesis without relying on image-based latent extraction at inference time, we introduce a lightweight illumination generator that learns to sample plausible lighting embeddings directly from noise. This module comprises five convolutional layers with ReLU activations, mapping a random noise vector to an illumination latent in the same embedding space as the autoencoderâs output. Both the input and output have the shape of latent code z. The generator is supervised by maximizing the similarity between its output and the autoencoder-generated latent codes. This loss is integrated into the contrastive objective described in Eq. (9).

## 3.2 Uncertainty-Aware Opacity Optimization (UAOO)

In scenes with inconsistent illumination across views, artifacts often emerge as Gaussian primitives overfit to viewspecific appearance variations, causing excessive opacity via large gradients on Î±. Prior works (Kerbl et al. 2023; Kulhanek et al. 2024; Zhang et al. 2024a) mitigate this by periodically resetting opacities or applying 2D mask-based filtering. However, these approaches risk removing real structures or fail to capture cross-view geometric inconsistencies.

To address cross-view lighting inconsistencies, we explicitly model uncertainty at the 3D point level by introducing a stochastic opacity perturbation during training. This strategy regularizes per-Gaussian opacity learning and suppresses abrupt gradient responses induced by dynamic illumination.

Specifically, instead of computing opacity from a fixed deterministic parameter, we model it as a stochastic function of a Gaussian-distributed latent variable. The compositing weight is computed as a perturbed sigmoid:

$$
\alpha _ { i } ^ { \mathrm { t r a i n } } = { \cal S } ( \mu _ { i } + \sigma _ { i } \cdot \epsilon ) , \quad \mathrm { w i t h } \quad \epsilon \sim { \mathcal N } ( 0 , 1 ) ,\tag{10}
$$

where $s$ denotes the sigmoid function, and $\mu _ { i }$ and $\sigma _ { i }$ represent the mean and standard deviation of the learned opacity distribution. This means that we no longer learn $\alpha _ { i }$ directly; instead, we model it implicitly through two separate pathways, $( \mu _ { i } , \sigma _ { i } )$ , thereby:

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

Figure 4: Comparison on the SeaThru-NeRF dataset. Right halves show the restored outputs from models with restoration capabilities (columns 3â5). Our method shows higher visibility than the baseline methods in the renderings.
<table><tr><td rowspan="2"></td><td rowspan="2">FPS / GPU hrs.</td><td colspan="3">CuraÃ§ao</td><td colspan="3">Panama</td><td colspan="2">IUI3</td><td rowspan="2"></td><td colspan="3">Japanese Gardens</td></tr><tr><td>PSNR</td><td>SIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR SSIM</td><td>ILPIPS</td></tr><tr><td>SeaThru-NeRF</td><td>&lt; 1/11.8</td><td>30.00</td><td>0.870</td><td>0.215</td><td>27.82</td><td>0.834</td><td>0.226</td><td>25.92</td><td>0.787</td><td>0.294</td><td>21.73</td><td>0.768</td><td>0.246</td></tr><tr><td>3DGS</td><td>163 / 0.38</td><td>29.28</td><td>0.923</td><td>0.202</td><td>23.68</td><td>0.749</td><td>0.250</td><td>29.76</td><td>0.875</td><td>0.121</td><td>21.45</td><td>0.852</td><td>0.206</td></tr><tr><td>WildGaussian</td><td>73 / 1.30</td><td>29.52</td><td>0.880</td><td>0.313</td><td>24.94</td><td>0.756</td><td>0.395</td><td>28.34</td><td>0.870</td><td>0.177</td><td>22.08</td><td>0.839</td><td>0.312</td></tr><tr><td>GS-W</td><td>42 / 1.69</td><td>24.20</td><td>0.843</td><td>0.382</td><td>24.30</td><td>0.748</td><td>0.387</td><td>27.64</td><td>0.863</td><td>0.263</td><td>21.46</td><td>0.850</td><td>0.244</td></tr><tr><td>UW-GS</td><td>30 / 0.38</td><td>31.77</td><td>0.943</td><td>0.144</td><td>31.79</td><td>0.936</td><td>0.116</td><td>28.65</td><td>0.933</td><td>0.125</td><td>23.05</td><td>0.860</td><td>0.190</td></tr><tr><td>R-Splatting (Ours)</td><td>107 / 0.86</td><td>32.98</td><td>0.956</td><td>0.163</td><td>32.52</td><td>0.930</td><td>0.107</td><td>30.15</td><td>0.947</td><td>0.105</td><td>24.03</td><td>0.868</td><td>0.211</td></tr></table>

Table 1: Quantitative evaluation on the SeaThru-NeRF dataset in terms of rendering speed (FPS)â, training time (GPU hrs)â, PSNRâ, SSIMâ, and LPIPSâ. The best , second-best , and third-best results are highlighted. All FPS and runtime values are measured on an NVIDIA RTX 4090.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Scene 1PSNRSSIMLPIPS</td><td rowspan=1 colspan=2>Scene 2PSNRSSIMLPIPS</td></tr><tr><td rowspan=1 colspan=1>Seathru-NeRF</td><td rowspan=1 colspan=2>15.920.4370.604</td><td rowspan=1 colspan=2>20.660.6190.541</td></tr><tr><td rowspan=2 colspan=1>3DGSUW-GS</td><td rowspan=1 colspan=1>21.170.772</td><td rowspan=1 colspan=1>0.285</td><td rowspan=1 colspan=2>24.450.8480.271</td></tr><tr><td rowspan=1 colspan=1>21.330.752</td><td rowspan=1 colspan=1>0.290</td><td rowspan=1 colspan=1>24.96</td><td rowspan=1 colspan=1>0.851 0.265</td></tr><tr><td rowspan=1 colspan=1>WaterSplatting</td><td rowspan=1 colspan=1>20.800.721</td><td rowspan=1 colspan=1>0.348</td><td rowspan=1 colspan=1>24.45</td><td rowspan=1 colspan=1>0.832 0.299</td></tr><tr><td rowspan=1 colspan=1>WildGaussian</td><td rowspan=1 colspan=1>21.760.748</td><td rowspan=1 colspan=1>0.387</td><td rowspan=1 colspan=1>24.98</td><td rowspan=1 colspan=1>0.8490.304</td></tr><tr><td rowspan=1 colspan=1>R-Splatting (our)</td><td rowspan=1 colspan=1>23.04 0.780</td><td rowspan=1 colspan=1>0.301</td><td rowspan=1 colspan=2>26.31 0.8600.253</td></tr></table>

Table 2: Quantitative evaluation on the new BlueCoral3D dataset in terms of PSNRâ, SSIMâ, and LPIPSâ.

â¢ Decomposing the gradient flow response of $\alpha _ { i }$ into a mean pathway $( \mu _ { i } )$ and a perturbation pathway $( \sigma _ { i } )$ :

$$
\frac { \partial \mathcal { L } } { \partial \mu _ { i } } \propto \frac { \partial \mathcal { L } } { \partial \alpha _ { i } } \cdot S ^ { \prime } ( \mu _ { i } + \sigma _ { i } \cdot \epsilon ) , \quad \frac { \partial \mathcal { L } } { \partial \sigma _ { i } } \propto \frac { \partial \mathcal { L } } { \partial \alpha _ { i } } \cdot S ^ { \prime } ( \mu _ { i } + \sigma _ { i } \cdot \epsilon ) \cdot \epsilon .\tag{11}
$$

â¢ Enabling the model to not only represent the opacity value itself, but also to capture its sensitivity to illumination-induced uncertainty.

This stochastic formulation introduces variability into the gradient path by modulating the same residual $\frac { \partial \mathcal { L } } { \partial \alpha _ { i } }$ with different samples of the noise term Ïµ. As a result, the model is prevented from consistently amplifying updates along any single direction based on a single noisy sample. This effectively suppresses overreaction to large residuals arising from outlier views or lighting inconsistencies, encouraging more stable opacity learning across varying illumination. Conceptually, our idea resembles gradient noise regularization (Neelakantan et al. 2015).

During inference, we replace the stochastic opacity with its expected value via a closed-form approximation:

$$
\begin{array} { l } { { \alpha _ { i } ^ { \mathrm { t e s t } } = \mathbb { E } _ { \epsilon } \left[ \sigma ( \mu _ { i } + \sigma _ { i } \cdot \epsilon ) \right] , } } \\ { { \alpha _ { i } ^ { \mathrm { t e s t } } \approx \sigma \left( \displaystyle \frac { \mu _ { i } } { \sqrt { 1 + \frac { \pi ^ { 2 } } { 8 } \sigma _ { i } ^ { 2 } } } \right) . } } \end{array}\tag{12}
$$

This yields a smooth, deterministic opacity that accounts for uncertainty. Intuitively, points with higher uncertainty $( i . e . ,$ larger $\sigma _ { i } )$ receive softer opacity values, reducing the influence of ambiguous or low-confidence regions in the final rendering. Conversely, confident points (i.e., small $\sigma _ { i } )$ maintain sharper transitions, ensuring structure preservation.

To encourage the model to reflect uncertainty in ambiguous regions, we use a regularization loss: $\begin{array} { r } { \mathcal { L } ^ { \mathrm { u c n } } \dot { = } - \sum _ { i } \left| \dot { \boldsymbol { \sigma } } _ { i } \right| } \end{array}$

## 4 Experiments

Datasets. Due to the challenges of illumination inconsistency, prior works have typically focused on small-scale datasets with around 20 front-facing images per scene. In contrast, our method robustly handles complex lighting variations, enabling large-scale reconstruction. To validate this, we introduce BlueCoral3D, a new dataset comprising two underwater scenes with dynamic illumination. Each scene contains approximately 500 high-resolution (1080p) images captured through a full 360-degree sweep, naturally introducing diverse lighting conditions across views. To create the training and test splits, we assign every 100th image in the dataset to the test set and use the rest for training. Additionally, we test our method on the SeaThru-NeRF dataset (Levy et al. 2023), a less challenging benchmark with only â¼25 front-view images per scene.

<!-- image-->  
Input

<!-- image-->  
WildGaussian

<!-- image-->  
UW-GS

<!-- image-->  
WaterSplatting

<!-- image-->  
R-Splatting (Ours)  
Figure 5: Comparison on the BlueCoral3D dataset. Restored renderings (right halves) show that UW-GS and WaterSplatting fail to recover accurate colors, due to unmodeled light reflections in shallow water.

Implementation details. We implement our method based on 3DGS (Kerbl et al. 2023) for its simplicity and speed. We build a unified interface that integrates four UIR models (Huang et al. 2025a; Fu et al. 2022; Han et al. 2022; Huang et al. 2023) to produce multiple sets of restored images. The autoencoder is implemented with the code by Bohan (2023), but we change the latent dimension to $2 4 \dot { \times } 3 2 \times 3 2$ . We directly borrow the masked decoder from (Van den Oord et al. 2016) to implement our illumination generator. Following 3DGS, the reconstruction loss ${ \mathcal { L } } ^ { \mathrm { r e c } }$ consists of a combination of DSSIM and $L _ { 1 }$ losses. Additionally, we jointly train the autoencoder and illumination generator in an end-to-end fashion. The final objective is given by $\mathcal { L } = \mathcal { L } ^ { \mathrm { r e c } } + \mathcal { L } ^ { \mathrm { c o n t r a } } + \lambda _ { \mathrm { u c n } } \mathcal { L } ^ { \mathrm { u c n } }$ , where $\lambda _ { \mathrm { u c n } }$ is set to 0.0005 in our experiments. We disable the periodic opacity resetting (POR) originally used in 3DGS, as our uncertaintyaware opacity optimization renders it unnecessary. We train our model for 15,000 and 30,000 iterations on the SeaThru-NeRF and BlueCoral3D datasets, respectively.

## 4.1 Evaluation

Baselines. We compare our method against SeaThru-NeRF (Levy et al. 2023), 3DGS (Kerbl et al. 2023), WildGaussian (Kulhanek et al. 2024), UW-GS (Wang et al. 2025), GS-W (Zhang et al. 2024a), and WaterSplatting (Li et al. 2025). As WildGaussian and GS-W are originally designed to handle appearance variations in uncontrolled images, they are potentially applicable to underwater illumination inconsistencies, and are therefore included as baselines in our evaluation. To ensure reproducibility and fairness, all baseline results are obtained using official implementations.

Quantitative comparison. The results on the Seathru-NeRF and BlueCoral3D datasets are provided in Tab. 1 and Tab. 2, respectively. Models that do not account for dynamic illumination consistently yield lower scores. While WildGaussian and GS-W mitigate artifacts using 2D uncertainty masks, these strategies can be overly aggressiveâoften suppressing valid structures and harming reconstruction quality. In contrast, our R-Splatting models opacity uncertainty directly in 3D space via UAOO, resulting in significantly improved performance across all metrics.

Notably, our R-Splatting achieves a significantly larger performance gain on the BlueCoral3D dataset compared to the SeaThru-NeRF dataset. This is because BlueCoral3D features more pronounced illumination changes and complex camera motion, making it a more challenging settingâone for which R-Splatting is specifically designed.

Render Speed. In Tab. 1, we also compare rendering speed and training time. Since all views in our approach share a single $\mathbf { v } _ { i }$ per point, the neural field only needs to be executed once for multi-view rendering, allowing R-Splatting to match 3DGS in rendering speed. In contrast, other methods with view-dependent neural fields require repeated evaluations, leading to slower rendering.

Qualitative comparison In Fig. 4, we present a visual comparison between our method and existing baselines. 3DGS and GS-W fail to reconstruct certain structures and textures, primarily due to aggressive opacity resets and reliance on 2D mask-based filtering. Other methods also exhibit reduced visibility and missing content. In contrast, our R-Splatting consistently produces clearer results, successfully recovering even distant objects. Notably, in the second row, our method reconstructs the distant color calibration chart with higher fidelity than all baselines.

Degraded  
Restored V1  
Restored V2  
Restored V3  
Restored V4  
<!-- image-->  
Figure 6: View-consistent renderings under diverse restoration styles. R-Splatting generates multiple restored outputs (V1âV4) for the same scene by sampling different latent codes from the illumination generator. While each column exhibits a unique restoration style, the renderings remain consistent across frames.

In Fig. 5, we show that in shallow water scenes, refracted sunlight through surface waves produces complex illumination patterns (caustics). Methods like UW-GS and Water-Splatting, which rely on simplified physical models, fail to recover these effects. In contrast, R-Splatting benefits from the strong restoration capacity of UIR models and achieves significantly better rendering quality.

View-consistent renderings. As shown in Fig. 6, our R-Splatting generates diverse restoration styles by sampling latent codes from the illumination generator. Owing to the view-invariant latent learning strategy (Sec. 3.1), each style maintains both geometric and illumination consistency across views.

## 4.2 Ablation Study

We evaluate the contributions of individual components in R-Splatting by constructing a series of model variants, each selectively enabling or disabling Neural Field (NF), Uncertainty-Aware Opacity Optimization (UAOO), and Periodic Opacity Resetting (POR). Results are summarized in Tab. 3.

Starting from the baseline 3DGS (M1), we first incorporate NF (M2), which improves PSNR from 22.81 to 23.05, suggesting the benefit of view-shared color modeling. In M3, we further introduce UAOO on top of M2, but observe a slight performance drop. To isolate this effect, M4 removes

<table><tr><td rowspan=1 colspan=1>Variant</td><td rowspan=1 colspan=1>NFUAOOPOR</td><td rowspan=1 colspan=2>MetricsPSNRSSIMLPIPS</td></tr><tr><td rowspan=2 colspan=1>M1 ( 3DGS)M2</td><td rowspan=3 colspan=1>*&gt;&gt;**&gt;   ÃÃ&gt;    &gt;J</td><td rowspan=1 colspan=1>22.810.810</td><td rowspan=1 colspan=1>0.278</td></tr><tr><td rowspan=2 colspan=1>M2M3M4</td><td rowspan=1 colspan=1>23.050.813</td><td rowspan=1 colspan=1>0.265</td></tr><tr><td rowspan=1 colspan=1>22.930.80522.710.798</td><td rowspan=1 colspan=1>0.2910.306</td></tr><tr><td rowspan=1 colspan=1>M5</td><td rowspan=1 colspan=1>â      X</td><td rowspan=1 colspan=1>24.320.817</td><td rowspan=1 colspan=1>0.282</td></tr><tr><td rowspan=1 colspan=1>M6 (R-Splatting)</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>24.670.820</td><td rowspan=1 colspan=1>0.277</td></tr></table>

Table 3: The performance of different model variants on BlueCoral3D. We report the average scores of PSNR, SSIM and LPIPs of the dataset.

NF but retains UAOO and POR, resulting in a further performance degradation. When POR is disabled (M5), the performance improves significantly (PSNR 24.32), indicating that POR may conflict with the stochastic nature of UAOO. Finally, M6 combines NF and the UAOOâno-POR setting, achieving the best overall performance.

## 5 Conclusion

We present R-Splatting, a unified 3D reconstruction framework tailored for underwater scenes with diverse restoration styles and dynamic illumination. Our method introduces two key components: (1) Uncertainty-Aware Opacity Optimization (UAOO) to robustly handle view-inconsistent lighting through stochastic opacity modeling, and (2) a conditional neural field to enable view-shared color reconstruction from multiple UIR-enhanced inputs. Compared to prior methods relying on simplified physical models, R-Splatting achieves superior geometric consistency and visual quality. Results on Seathru-NeRF and our newly collected BlueCoral3D dataset validate its effectiveness.

## Acknowledgements

This work was supported by the UKRI MyWorld Strength in Places Program (SIPF00006/1) and the EPSRC ECR International Collaboration Grants (EP/Y002490/1).

## References

Bohan, O. B. 2023. TAESD: Tiny Autoencoder for Stable Diffusion. https://github.com/madebyollin/taesd. Accessed: 2025-07-14.

Chen, L.; Xiong, Y.; Zhang, Y.; Yu, R.; Fang, L.; and Liu, D. 2024. SP-SeaNeRF: Underwater Neural Radiance Fields with strong scattering perception. Computers & Graphics, 123: 104025.

Chiang, J. Y.; and Chen, Y.-C. 2012. Underwater Image Enhancement by Wavelength Compensation and Dehazing. IEEE Transactions on Image Processing, 21(4): 1756â1769.

Fu, Z.; Wang, W.; Huang, Y.; Ding, X.; and Ma, K.-K. 2022. Uncertainty inspired underwater image enhancement. In European conference on computer vision, 465â482. Springer.

Galdran, A.; Pardo, D.; Picon, A.; and Alvarez-Gila, A. Â´ 2015. Automatic Red-Channel Underwater Image Restoration. Journal of Visual Communication and Image Representation, 26: 132â145.

Gough, L.; Azzarelli, A.; Zhang, F.; and Anantrasirichai, N. 2025. AquaNeRF: Neural Radiance Fields in Underwater Media with Distractor Removal. In IEEE International Symposium on Circuits and Systems.

Guan, M.; Xu, H.; Jiang, G.; Yu, M.; Chen, Y.; Luo, T.; and Song, Y. 2024. WaterMamba: Visual State Space Model for Underwater Image Enhancement. arXiv:2405.08419.

Han, J.; Shoeiby, M.; Malthus, T.; Botha, E.; Anstee, J.; Anwar, S.; Wei, R.; Armin, M. A.; Li, H.; and Petersson, L. 2022. Underwater image restoration via contrastive learning and a real-world dataset. Remote Sensing, 14(17): 4297.

He, K.; Fan, H.; Wu, Y.; Xie, S.; and Girshick, R. 2020. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 9729â9738.

He, K.; Sun, J.; and Tang, X. 2010. Single image haze removal using dark channel prior. IEEE transactions on pattern analysis and machine intelligence, 33(12): 2341â2353.

Huang, G.; Anantrasirichai, N.; Ye, F.; Qi, Z.; Lin, R.; Yang, Q.; and Bull, D. 2025a. Bayesian Neural Networks for Oneto-Many Mapping in Image Enhancement. arXiv preprint arXiv:2501.14265.

Huang, G.; Lin, R.; Li, Y.; Bull, D.; and Anantrasirichai, N. 2025b. BVI-Mamba: video enhancement using a visual state-space model for low-light and underwater environments. In Machine Learning from Challenging Data 2025, volume 13460, 74â81. SPIE.

Huang, G.; Wang, H.; Seymour, B.; Kovacs, E.; Ellerbrock, J.; Blackham, D.; and Anantrasirichai, N. 2025c. Visual enhancement and 3D representation for underwater scenes: a review. arXiv e-prints, arXivâ2505.

Huang, S.; Wang, K.; Liu, H.; Chen, J.; and Li, Y. 2023. Contrastive semi-supervised learning for underwater image restoration via reliable bank. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 18145â18155.

Jiang, Z.; Wang, H.; Huang, G.; Seymour, B.; and Anantrasirichai, N. 2025a. RUSplatting: Robust 3D Gaussian Splatting for Sparse-View Underwater Scene Reconstruction. arXiv preprint arXiv:2505.15737.

Jiang, Z.; Wang, H.; Huang, G.; Seymour, B.; and Anantrasirichai, N. 2025b. SWAGSplatting: Semanticguided Water-scene Augmented Gaussian Splatting. arXiv preprint arXiv:2509.00800.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3D gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139â1.

Kulhanek, J.; Peng, S.; Kukelova, Z.; Pollefeys, M.; and Sattler, T. 2024. Wildgaussians: 3d gaussian splatting in the wild. In Advances in Neural Information Processing Systems.

Levy, D.; Peleg, A.; Pearl, N.; Rosenbaum, D.; Akkaynak, D.; Korman, S.; and Treibitz, T. 2023. Seathru-nerf: Neural radiance fields in scattering media. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 56â65.

Li, C.; Anwar, S.; Hou, J.; Cong, R.; Guo, C.; and Ren, W. 2021. Underwater Image Enhancement via Medium Transmission-Guided Multi-Color Space Embedding. IEEE Transactions on Image Processing, 30: 4985â5000.

Li, H.; Song, W.; Xu, T.; Elsig, A.; and Kulhanek, J. 2025. WaterSplatting: Fast Underwater 3D Scene Reconstruction using Gaussian Splatting. 3DV.

Liang, Z.; Ding, X.; Wang, Y.; Yan, X.; and Fu, X. 2022. GUDCP: Generalization of Underwater Dark Channel Prior for Underwater Image Restoration. IEEE Transactions on Circuits and Systems for Video Technology, 32(7): 4879â 4884.

Lin, W.-T.; Lin, Y.-X.; Chen, J.-W.; and Hua, K.-L. 2024. PixMamba: Leveraging state space models in a dual-level architecture for underwater image enhancement. In Proceedings of the Asian Conference on Computer Vision, 3622â 3637.

Liu, S.; Lu, J.; Gu, Z.; Li, J.; and Deng, Y. 2024. Aquatic-GS: A Hybrid 3D Representation for Underwater Scenes. arXiv preprint arXiv:2411.00239.

Malyugina, A.; Huang, G.; Ruiz, E.; Leslie, B.; and Anantrasirichai, N. 2025. Marine Snow Removal Using Internally Generated Pseudo Ground Truth. arXiv preprint arXiv:2504.19289.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99â106.

Mualem, N.; Amoyal, R.; Freifeld, O.; and Akkaynak, D. 2024. Gaussian Splashing: Direct Volumetric Rendering Underwater. arXiv preprint arXiv:2411.19588.

Muller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. In-Â¨ stant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1â 15.

Neelakantan, A.; Vilnis, L.; Le, Q. V.; Sutskever, I.; Kaiser, L.; Kurach, K.; and Martens, J. 2015. Adding gradient noise improves learning for very deep networks. arXiv preprint arXiv:1511.06807.

Peng, L.; and Bian, L. 2025. Adaptive Dual-domain Learning for Underwater Image Enhancement. In Proceedings of the AAAI Conference on Artificial Intelligence, 6, 6461â 6469.

Peng, L.; Zhu, C.; and Bian, L. 2023. U-shape transformer for underwater image enhancement. IEEE Transactions on Image Processing.

Peng, Y.-T.; Cao, K.; and Cosman, P. C. 2018. Generalization of the dark channel prior for single image restoration. IEEE Transactions on Image Processing, 27(6): 2856â2868.

Qu, Z.; Vengurlekar, O.; Qadri, M.; Zhang, K.; Kaess, M.; Metzler, C.; Jayasuriya, S.; and Pediredla, A. 2024. Z-splat: Z-axis gaussian splatting for camera-sonar fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Ramazzina, A.; Bijelic, M.; Walz, S.; Sanvito, A.; Scheuble, D.; and Heide, F. 2023. Scatternerf: Seeing through fog with physically-based inverse neural rendering. In International Conference on Computer Vision (ICCV), 17957â17968.

Ren, T.; Liu, S.; Zeng, A.; Lin, J.; Li, K.; Cao, H.; Chen, J.; Huang, X.; Chen, Y.; Yan, F.; et al. 2024. Grounded SAM: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159.

Sethuraman, A. V.; Ramanagopal, M. S.; and Skinner, K. A. 2023. WaterNeRF: Neural Radiance Fields for Underwater Scenes. In OCEANS 2023 - MTS/IEEE U.S. Gulf Coast, 1â7.

Tan, R. T. 2008. Visibility in Bad Weather from a Single Image. In 2008 IEEE Conference on Computer Vision and Pattern Recognition, 1â8.

Tang, Y.; Kawasaki, H.; and Iwaguchi, T. 2023. Underwater image enhancement by transformer-based diffusion model with non-uniform sampling for skip strategy. In Proceedings of the 31st ACM International Conference on Multimedia, 5419â5427.

Tang, Y.; Zhu, C.; Wan, R.; Xu, C.; and Shi, B. 2024. Neural Underwater Scene Representation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11780â11789.

Van den Oord, A.; Kalchbrenner, N.; Espeholt, L.; Vinyals, O.; Graves, A.; et al. 2016. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29.

Wang, H.; Anantrasirichai, N.; Zhang, F.; and Bull, D. 2025. UW-GS: Distractor-aware 3d gaussian splatting for enhanced underwater scene reconstruction. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 3280â3289. IEEE.

Wen, H.; Tian, Y.; Huang, T.; and Gao, W. 2013. Single Underwater Image Enhancement with a New Optical Model. In 2013 IEEE International Symposium on Circuits and Systems (ISCAS), 753â756.

Yang, D.; Leonard, J. J.; and Girdhar, Y. 2024. Seasplat: Representing underwater scenes with 3d gaussian splatting and a physically grounded image formation model. arXiv preprint arXiv:2409.17345.

Yang, H.-Y.; Chen, P.-Y.; Huang, C.-C.; Zhuang, Y.-Z.; and Shiau, Y.-H. 2011. Low Complexity Underwater Image Enhancement Based on Dark Channel Prior. In 2011 Second International Conference on Innovations in Bio-inspired Computing and Applications, 17â20.

Yu, Z.; Chen, A.; Huang, B.; Sattler, T.; and Geiger, A. 2024. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 19447â19456.

Zhang, D.; Wang, C.; Wang, W.; Li, P.; Qin, M.; and Wang, H. 2024a. Gaussian in the wild: 3d gaussian splatting for unconstrained image collections. In European Conference on Computer Vision, 341â359. Springer.

Zhang, T.; Zhi, W.; Meyers, B.; Durrant, N.; Huang, K.; Mangelson, J.; Barbalata, C.; and Johnson-Roberson, M. 2024b. Recgs: Removing water caustic with recurrent gaussian splatting. IEEE Robotics and Automation Letters.

Zhang, W.; Liu, Q.; Lu, H.; Wang, J.; and Liang, J. 2025. Underwater image enhancement via wavelet decomposition fusion of advantage contrast. IEEE Transactions on Circuits and Systems for Video Technology.

Zhou, J.; Liang, T.; Zhang, D.; Liu, S.; Wang, J.; and Wu, E. Q. 2025. WaterHE-NeRF: Water-ray matching neural radiance fields for underwater scene reconstruction. Information Fusion, 115: 102770.

Zwicker, M.; Pfister, H.; Van Baar, J.; and Gross, M. 2001. Surface splatting. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, 371â378.