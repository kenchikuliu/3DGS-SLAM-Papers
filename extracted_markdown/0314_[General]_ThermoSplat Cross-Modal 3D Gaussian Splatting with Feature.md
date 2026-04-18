# ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling

Zhaoqi Su, Shihai Chen, Xinyan Lin, Liqin Huang, Zhipeng Su, and Xiaoqiang Lu

AbstractâMulti-modal scene reconstruction integrating RGB and thermal infrared data is essential for robust environmental perception across diverse lighting and weather conditions. However, extending 3D Gaussian Splatting (3DGS) to multi-spectral scenarios remains challenging. Current approaches often struggle to fully leverage the complementary information of multi-modal data, typically relying on mechanisms that either tend to neglect cross-modal correlations or leverage shared representations that fail to adaptively handle the complex structural correlations and physical discrepancies between spectrums. To address these limitations, we propose ThermoSplat, a novel framework that enables deep spectral-aware reconstruction through active feature modulation and adaptive geometry decoupling. First, we introduce a Spectrum-Aware Adaptive Modulation that dynamically conditions shared latent features on thermal structural priors, effectively guiding visible texture synthesis with reliable crossmodal geometric cues. Second, to accommodate modality-specific geometric inconsistencies, we propose a Modality-Adaptive Geometric Decoupling scheme that learns independent opacity offsets and executes an independent rasterization pass for the thermal branch. Additionally, a hybrid rendering pipeline is employed to integrate explicit Spherical Harmonics with implicit neural decoding, ensuring both semantic consistency and high-frequency detail preservation. Extensive experiments on the RGBT-Scenes dataset demonstrate that ThermoSplat achieves state-of-the-art rendering quality across both visible and thermal spectrums.

Index Termsâ3D Gaussian Splatting, RGBT scene reconstruction, multi-modal fusion, neural rendering, feature modulation.

## I. INTRODUCTION

3 D scene reconstruction has been widely used in the field of autonomous systems, remote sensing, surveillance, etc. Traditional RGB-based reconstruction methods, while achieving high fidelity in most conditions, often suffer from performance degradation in challenging environments, e.g., lowlight conditions, dense smoke, or darkness. To address these limitations, multi-modal reconstruction, especially those integrating RGB and thermal, has emerged as a critical research direction. Unlike RGB sensors, which depend on reflected light, thermal sensors capture long-wave infrared radiation emitted by objects, allowing for the acquisition of stable structural information and heat signatures that are inherently invariant to illumination changes. This provides a reliable reference for scene geometry under extreme conditions.

The evolution of neural implicit representations has inspired research in multi-modal 3D reconstruction. Previous studies [1], [2] extended the Neural Radiance Fields (NeRF) [3] framework to the infrared spectrum, demonstrating the potential to synthesize thermal views from multi-view observations. However, NeRF-based methods often suffer from high computational cost and slow inference speeds, limiting their abilities in real-time applications. Recently, the emergence of 3D Gaussian Splatting (3DGS) [4] has enabled significantly faster training and rendering, with improved rendering quality. Building on this, several multi-spectral 3DGS frameworks have been proposed. Current state-of-the-art methods can be categorized into two paradigms: either explicitly decomposing 3DGS into modality-specific components to handle property disparities [5], or integrating multi-spectral information into a unified latent space for MLP-based decoding [6], [7]. However, achieving an optimal balance between shared geometry and modality-specific appearance remains non-trivial. The former paradigm often introduces increased model complexity and may face challenges in maintaining cross-modal spatial consistency, while the latter tends to have a limited capacity to precisely model the inherent physical discrepancies and structural variations present across different spectrums.

To bridge these gaps, we present ThermoSplat, a novel cross-modal 3DGS framework that enables deep spectralaware reconstruction through active feature modulation and adaptive geometry decoupling. Unlike existing methods that either rely on explicit decomposition or unified latent representation, ThermoSplat introduces a Spectrum-Aware Adaptive Modulation. While drawing inspiration from the mathematical flexibility of linear modulation [8], this module dynamically conditions the shared latent representation on thermal structural priors, enabling the model to actively leverage structural infrared features to guide visible texture synthesis. Furthermore, to accommodate the inherent physical discrepancies across different spectral bands, we propose a modalityadaptive geometric decoupling scheme, which allows for independent geometric adjustments in the infrared spectrum, effectively resolving artifacts in regions where transparency or reflectivity varies. Finally, to overcome the detail-loss inherent in pure feature-based decoding, we employ a hybrid rendering pipeline, which integrates explicit Spherical Harmonics (SH) with implicit decoding, achieving high-frequency RGB details while maintaining consistent semantic information across modalities.

Experimental results demonstrate that ThermoSplat achieves state-of-the-art rendering quality across both visible and thermal spectrums. The main contributions of this work are summarized as follows:

â¢ Spectrum-Aware Adaptive Modulation: We design a Spectrum-Aware Adaptive Modulation framework to establish deep feature dependencies. By utilizing structural priors to modulate shared latent features, our method enhances texture recovery and cross-modal alignment.

â¢ Modality-Adaptive Geometric Decoupling: We introduce a learnable thermal opacity offset and execute an independent rasterization pass that decouples geometric representations between visible and infrared spectrums. This mechanism effectively resolves depth and occlusion misalignments caused by modality-inconsistent physical properties.

â¢ Hybrid Explicit-Implicit Rendering: We propose a hybrid rendering pipeline that integrates explicit Spherical Harmonics (SH) with feature-modulated neural decoding. This architecture preserves high-frequency RGB details while maintaining consistent low-frequency semantic information across different modalities.

## II. RELATED WORK

## A. 3D Neural Scene Representation

Recent methods in 3D neural scene representation have shifted from implicit to explicit methods. The implicit NeRFbased methods [3], [9]â[11] represent the scene as a continuous function in 3D space formulated by a shallow network like MLP, achieving view-dependent and photo-realistic scene rendering results. However, these methods suffer from timeconsuming training and rendering, limiting their practical use in real-time applications. To address this, 3D Gaussian Splatting (3DGS)-based methods [4], [12]â[14] propose an explicit scene representation paradigm, which leverages 3D Gaussian primitives for explicitly representing the geometric and texture information of the scene, enabling high-fidelity and real-time rendering through a differentiable tile-based rasterization pipeline. Some studies [15]â[17] leverage the idea of both feature-based decoding in NeRF and 3DGS representations to augment the representation capability by distilling high-dimensional latent features into each Gaussian primitive. By integrating these latent features with lightweight decoders, these methods can bypass the limitations of traditional Spherical Harmonics (SH), enabling more complex attribute modeling and cross-modal information interaction.

## B. Neural Thermal and RGBT Scene Reconstruction

Compared to visible light, thermal infrared signals possess distinct physical properties, such as being insensitive to lighting conditions and capable of reflecting the heat distribution of objects. Early attempts mostly extend NeRF-based representations for representing different modalities in a compact manner [1], [2], [18]â[20]. However, due to the volume rendering process in NeRF [3] and its reliance on dense sampling, these implicit methods often face challenges in precisely modeling the high-frequency details. In recent years, the emergence of 3DGS has shifted the focus toward explicit Gaussian-based RGBT (RGB + Thermal) scene modeling. ThermalGaussian [5] pioneered the extension of 3DGS to the RGBT scene, which optimizes the thermal Gaussian by finetuning the pretrained RGB Gaussians and incorporates thermal priors for better scene modeling. Also, it releases the RGBT-Scenes dataset to facilitate benchmarking for multi-modal reconstruction tasks. MS-Splatting [6] formulates the multispectral 3D scene using a unified latent space for decoding both RGB and other spectral channels, which is also applied to agricultural NDVI tasks. MS-Splattingv2 [21] uses the optimized joint strategy with RGB initialization to improve rendering quality. MMOne [7] introduces a unified framework that represents multiple modalities, such as RGB, thermal, and language, within a single scene, which designs a multimodal decomposition mechanism for better learning properties of different modalities. Ma et al. [22] decomposes appearance into reflectance and thermal radiance, leveraging the thermal modality as a stable geometric prior to rectify distorted surfaces in low-light RGB inputs. Beyond general multimodal representation, several studies focus on reconstructing thermal infrared signals to tackle ill-posed problems or extreme environmental constraints. Some studies [23], [24] inject physicsbased temperature or thermodynamics constraints into thermal 3DGS modeling. Veta-GS [25] introduces a view-dependent deformation field to capture the subtle thermal variations caused by emissivity and transmission effects, effectively reducing artifacts in infrared novel-view synthesis. Others extend the RGBT modeling into more-spectral or hyperspectral scenarios [26], [27]. Despite these advances, existing RGBT frameworks either treat different modalities as independent signals with limited feature interaction, or rely on a shared representation that tends to overlook modality-specific physical discrepancies. These limitations motivate us to explore a more flexible modulation and decoupled modeling for RGBT scene reconstruction.

## III. METHOD

## A. Overview

The overall architecture of ThermoSplat is designed to achieve high-fidelity multi-modal scene reconstruction by addressing spectral-varying properties. As illustrated in Fig. 1, we represent the scene using multi-modal feature-enhanced 3DGS [4]. The pipeline first performs active feature interaction via a Spectrum-Aware Adaptive Modulation on the rasterized latent representations, which utilizes thermal structural priors to guide visible texture synthesis. To account for geometric inconsistencies across spectrums, we introduce a modality-adaptive geometric decoupling scheme, which uses the learnable offset $\Delta _ { t } \alpha$ and executes an independent rasterization pass to accommodate modality-specific geometries. Finally, a hybrid rendering strategy is employed to combine explicit Spherical Harmonics (SH) with implicit feature-decoded outputs for preserving high-frequency details and view-dependent effects in the visible spectrum.

The remainder of this section provides a detailed formalization of our framework. We first briefly introduce 3D Gaussian Splatting and feature-based rasterization in Section III-B. In Section III-C, we describe the cross-modal feature modulation mechanism, which enables active spectral interaction. Section III-D presents the Multi-spectral Hybrid Rendering pipeline, where we first detail the modality-adaptive geometric decoupling for modal-specific geometries, followed by the hybrid rendering strategy for RGB synthesis to preserve highfrequency details.

<!-- image-->  
Fig. 1. Overview of the proposed ThermoSplat framework. Given multi-spectral inputs, our method optimizes 3D Gaussian primitives with decoupled properties. (a) Spectrum-Aware Adaptive Modulation dynamically conditions shared latent features on thermal structural priors to guide visible texture synthesis. (b) Modality-Adaptive Geometric Decoupling resolves geometric inconsistencies between visible and infrared spectrums. (c) The Hybrid Rendering pipeline integrates explicit Spherical Harmonics (SH) with implicit neural decoding, ensuring high-frequency detail preservation and cross-modal semantic consistency.

## B. Preliminaries

3D Gaussian Splatting (3DGS) [4] represents a 3D scene as a collection of N Gaussian primitives. Each Gaussian is characterized by its center position $\mu \in \mathbb { R } ^ { 3 }$ , an anisotropic covariance $\Sigma \ \stackrel { \cdot } { = } \ R S S ^ { T } R ^ { \hat { T } }$ , and an opacity value Î±. The influence of a Gaussian at a 3D point $\mathbf { x } ^ { \prime }$ is defined as:

$$
G ( { \pmb x } ^ { \prime } ; { \pmb \mu } _ { i } , { \Sigma } _ { i } ) = e ^ { - \frac { 1 } { 2 } ( { \pmb x } ^ { \prime } - { \pmb \mu } _ { i } ) ^ { T } { \Sigma } _ { i } ^ { - 1 } ( { \pmb x } ^ { \prime } - { \pmb \mu } _ { i } ) } ,\tag{1}
$$

where $\mathbf { x } ^ { \prime }$ denotes the 3D point in the camera coordinate system. Unlike traditional 3DGS that directly optimizes Spherical Harmonic (SH) coefficients for color, we follow a featurebased splatting paradigm [6] where each Gaussian carries a multi-dimensional latent feature $f \in \mathbb { R } ^ { d }$ . This feature serves as a unified latent representation that can be subsequently decoded into modality-specific signals (e.g., RGB or thermal) via neural networks.

The rendering process follows the point-based Î±-blending model. For a specific pixel, the attributes of projected 2D Gaussians are sorted by depth and blended to compute the aggregated pixel value:

$$
\mathcal { A } = \sum _ { i \in \mathcal { N } } \mathbf { a } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

where ${ \bf a } _ { i }$ denotes the generic attribute of the i-th Gaussian (such as latent feature $f _ { i }$ or explicit color attributes) and $\alpha _ { i }$ is the opacity of the Gaussian at that pixel.

In this work, we leverage this differentiable rasterization to bridge different modalities. By decoupling the feature decoding from the geometric projection, we can perform complex

cross-modal modulations in the latent space before generating the final visible and thermal images.

## C. Cross-Modal Feature Modulation

To address the spectral gap between visible and infrared modalities, we propose a cross-modal modulation mechanism. Instead of treating visible and thermal signals as independent entities, our framework leverages structural priors inherent in the thermal spectrum to guide the synthesis of visible textures. As shown in Fig. 1, the proposed Spectrum-Aware Adaptive Modulation integrates feature extraction and conditioning in a unified neural architecture.

Shared Latent Encoding. The process begins with the rendered feature map $\mathcal { A } _ { f } \in \bar { \mathbb { R } } ^ { H \times W \times \bar { d } }$ by rasterizing the per-Gaussian feature $f _ { i }$ through the 3DGS rendering pipeline [4]. To extract high-level semantic information, we first pass $\ b { A _ { f } }$ through a shared encoder $\Phi _ { s h a r e d }$ consisting of multiple pixelwise linear layers with SiLU activations [28]:

$$
h = \Phi _ { s h a r e d } ( \mathcal { A } _ { f } ) ,\tag{3}
$$

where h represents the intermediate feature representation that serves as the common basis for both modalities.

Spectrum-Aware Adaptive Modulation. Distinct from directly applying MLP layers for different spectrums, we introduce a Thermal Prior Head $\Phi _ { t h }$ and a Spectrum-Aware Adaptive Modulation [8] layer. Unlike generic conditioning methods, our modulation parameters $( \gamma , \beta )$ are dynamically derived from a dedicated Thermal Prior Head $\Phi _ { t h }$ . Crucially, as $\begin{array} { c c l } { h _ { t h } } & { = } & { \Phi _ { t h } ( h ) } \end{array}$ is directly supervised by the thermal rendering task through the subsequent decoding stage (Eq. 6), the modulation process is implicitly physics-aware, ensuring that the distilled structural priors from the infrared domain actively guide the visible feature synthesis. The feature $h _ { t h }$ is subsequently mapped to a set of modulation parameters $( \gamma , \beta )$ through a linear transformation:

<!-- image-->  
Fig. 2. Thermal rendering results with and without geometric decoupling. The thermal rendering results without geometric decoupling may inherit sharp textures and high-frequency noise from the visible spectrum.

$$
[ \gamma , \beta ] = \Phi _ { m o d } ( h _ { t h } ) .\tag{4}
$$

By treating the infrared information as a conditioning signal, we apply Spectrum-Aware Adaptive Modulation to the shared representation h:

$$
h _ { m o d } = \gamma \odot h + \beta ,\tag{5}
$$

where â denotes element-wise multiplication. This operation dynamically scales and shifts the latent features based on the thermal structural prior, effectively âmaskingâ or âenhancingâ regions where visible textures are likely to align with thermal boundaries.

Modality-Specific Decoding. Finally, the modulated feature $h _ { m o d }$ and the thermal feature $h _ { t h }$ are decoded into their respective spectral domains:

$$
\begin{array} { r l } & { \mathcal { C } _ { i m p l i c i t } ^ { r g b } = \mathrm { S i g m o i d } ( \Phi _ { r g b } ( h _ { m o d } ) ) , } \\ & { \mathcal { C } ^ { t h e r m a l } = \mathrm { S i g m o i d } ( \Phi _ { t h _ { - } o u t } ( h _ { t h } ) ) , } \end{array}\tag{6}
$$

where $\mathcal { C } _ { i m p l i c i t } ^ { r g b }$ provides the base color component for the subsequent hybrid rendering stage. Notably, the thermal-specific feature $h _ { t h }$ serves a dual purpose: it acts as the source for the proposed modulation parameter generation while simultaneously being decoded into the infrared signal Cthermal. This hierarchical modulation ensures that the synthesis of visible images is physically constrained by the cross-modal structural consistency.

## D. Multi-spectral Hybrid Rendering

Based on the modulated cross-modal features, we develop a dual-branch rendering pipeline to synthesize images in both visible and thermal spectrums. This pipeline addresses the geometric inconsistencies and texture fidelity requirements unique to each modality.

Modality-Adaptive Geometric Decoupling Typical multimodal Gaussian representations assume a shared geometry across all spectrums. However, physical properties such as transparency and reflectivity vary significantly between visible and infrared bands. To accommodate these discrepancies, we introduce a modality-adaptive geometric decoupling scheme.

For the thermal rendering branch, we define a modalityspecific opacity $\alpha _ { t , i }$ for each Gaussian i by adding a learnable offset $\Delta _ { t } \alpha _ { i }$ to the base opacity $\alpha _ { i } { : }$

$$
\alpha _ { t , i } = \mathrm { S i g m o i d } ( \mathrm { L o g i t } ( \alpha _ { i } ) + \Delta _ { t } \alpha _ { i } ) ,\tag{7}
$$

where $\Delta _ { t } \alpha _ { i }$ captures the fine-grained geometric deviations. Consequently, the thermal representation is generated via an independent rasterization pass:

$$
\begin{array} { r } { \mathcal { A } _ { f ( t ) } = \mathrm { R a s t e r i z e } ( \mu , \Sigma , \alpha _ { t } , f ) , } \end{array}\tag{8}
$$

where $\begin{array} { c c l } { \mathcal { A } _ { f ( t ) } } & { \in } & { \mathbb { R } ^ { H \times W \times d } } \end{array}$ represents the thermal-specific feature map generated using the decoupled opacity $\alpha _ { t } .$ The final thermal image Cthermal is then decoded from $\boldsymbol { \mathcal { A } } _ { f ( t ) }$ via the thermal head discussed in Section III-C. This independent pass ensures that occlusions and structural boundaries in the thermal image remain physically consistent with infrared sensors.

As shown in Fig. 2, without the geometric decoupling mechanism, the thermal branch tends to inherit redundant high-frequency textures from the visible spectrum that do not exist in the infrared domain. Our proposed decoupling module effectively filters out these cross-modal artifacts, ensuring that the thermal rendering preserves its natural smoothness while accurately representing its own structural boundaries.

Hybrid RGB Synthesis. While the thermal branch focuses on geometric consistency, the RGB branch requires high-frequency view-dependent details. We propose a hybrid strategy that bridges explicit Gaussian Splatting with implicit neural decoding.

<!-- image-->  
Fig. 3. Feature level reconstruction loss on the rasterized feature maps. Left: rendered feature map, right: reconstructed RGB-thermal scene. Note that $\boldsymbol { \mathcal { A } } _ { f }$ and $\boldsymbol { A } _ { f ( t ) }$ are only different in the opacity used in rasterization.

Specifically, the final RGB color $\mathbf { C } ^ { r g b }$ is formulated as the summation of two components:

$$
\mathbf { C } ^ { r g b } = \mathcal { R } _ { s h } ( \mu , \Sigma , \alpha , \mathbf { c } _ { s h } ) \oplus \mathcal { C } _ { i m p l i c i t } ^ { r g b } ,\tag{9}
$$

where $\mathcal { R } _ { s h }$ denotes the explicit color rendered via standard Spherical Harmonic (SH) coefficients $\mathbf { c } _ { s h } .$ , capturing viewdependent specular effects. The second term, $\mathcal { C } _ { i m p l i c i t } ^ { r g b } ,$ is the implicit component decoded from the modulated latent features $h _ { m o d } .$ providing multi-modal consistent textures. By combining these two components, our hybrid rendering scheme effectively preserves the high-frequency viewdependent properties of explicit rasterization, while simultaneously enriching the visible textures with the structural intelligence of neural-modulated latent features.

## E. Loss Functions

The training objective of ThermoSplat is to optimize the multi-modal Gaussian representation and the neural modulation networks through a composite loss function L. This objective ensures that the synthesized visible and thermal images adhere to the ground truth in terms of both pixel intensity and structural topology.

Spectral Reconstruction Loss For both the visible and infrared modalities, we employ a combination of $\ell _ { 1 }$ loss and Structural Similarity (SSIM) to supervise the final rendered images against the corresponding ground-truth images ${ \mathbf I } _ { m }$ :

$$
\mathcal { L } _ { r e c } ^ { m } = ( 1 - \lambda _ { s } ) \| \mathbf { I } _ { m } - \mathbf { C } ^ { m } \| _ { 1 } + \lambda _ { s } ( 1 - \mathrm { S S I M } ( \mathbf { I } _ { m } , \mathbf { C } ^ { m } ) ) ,\tag{10}
$$

where $m \in \{ r g b$ , thermal} denotes the spectral modality, and ${ \bf { C } } ^ { m }$ are the corresponding output images in our pipeline.

To provide structural guidance during the intermediate stages, we enforce consistency on the rasterized feature maps $\ b { A _ { f } }$ and $\boldsymbol { \mathcal { A } } _ { f ( t ) }$ by slicing specific channels corresponding to physical properties. As shown in Fig. 3, for the visible branch, we constrain the first three channels of $\ b { A _ { f } }$ to match the RGB appearance. In parallel, for the thermal branch, we supervise the subsequent latent channel (index 3) of $\boldsymbol { \mathcal { A } } _ { f ( t ) }$ using the transformed thermal map derived from the Ironbow colormap protocol. As this transformed map effectively serves as a proxy for physical temperature and thermal intensity, this constraint encourages the model to learn a compact and structural representation. By applying these latent constraints, we ensure the latent space captures the fundamental visual and thermal distribution before it is decoded. The feature-level reconstruction loss is thus formulated as:

$$
\mathcal { L } _ { r e c } ^ { f e a t } = \mathcal { L } ( A _ { f } [ : 3 ] , \mathbf { I } _ { r g b } ) + \eta \cdot \mathcal { L } ( A _ { f ( t ) } [ 3 ] , \mathbf { I } _ { t h } ^ { t r a n s } ) ,\tag{11}
$$

where $\mathbf { I } _ { t h } ^ { t r a n s }$ represents the temperature-correlated intensity map, $\mathcal { L }$ denotes the composite $\ell _ { 1 }$ and SSIM loss function as defined in Eq. 10.

Thermal Spatial Regularization Due to the high-contrast and often sparse nature of infrared signals, we introduce a spatial smoothness constraint on the predicted thermal image:

$$
\mathcal { L } _ { s m o o t h } = \sum _ { p \in \Omega } | \nabla { \bf C } ^ { t h e r m a l } ( p ) | ,\tag{12}
$$

where $\nabla$ denotes the spatial gradient operator at pixel $p .$ This term enforces the smooth structural characteristics of the thermal output.

Total Objective The final training objective is a weighted summation of the aforementioned reconstruction and regularization terms:

$$
\mathcal { L } = \mathcal { L } _ { r e c } + \lambda _ { r f } \mathcal { L } _ { r e c } ^ { f e a t } + \lambda _ { s m } \mathcal { L } _ { s m o o t h } ,\tag{13}
$$

where the image-level reconstruction loss is defined as $\mathcal { L } _ { r e c } =$ $\mathcal { L } _ { r e c } ^ { r g b } + \mathcal { L } _ { r e c } ^ { t h e r m a l } , \lambda _ { r f }$ and $\lambda _ { s m }$ are hyper-parameters balancing feature terms and smooth terms. By optimizing this joint objective, our framework ensures that the synthesized modalities satisfy both pixel-level accuracy and the inherent structural characteristics of thermal radiation.

## IV. EXPERIMENTS

## A. Implementation details.

We evaluate our model on the RGBT-Scenes dataset, which comprises over 1,000 calibrated RGB-thermal pairs across ten indoor and outdoor scenes under diverse environmental and lighting conditions. To demonstrate the effectiveness of our approach, we compare our model with state-of-the-art methods, including MMOne [7], MS-Splattingv2 [21] and ThermalGaussian [5]. We also compare our method with the 3DGS [4] baseline trained on both modalities separately. We use Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) [29] to evaluate the rendering quality of both visible and thermal modalities.

Our framework is implemented on PyTorch and trained with an NVIDIA 3090 GPU. We train our pipeline for 30K iterations, which is the same as the setting used in 3DGS [4], MMOne [7] and ThermalGaussian [5]. For MS-Splattingv2 [21], we follow the training strategy proposed in the paper and train them with 120K iterations. In our experiments, we set per-Gaussian feature dimension $d = 8 ,$ and $\lambda _ { s } = 0 . 2 , \eta = 0 . 5 , \lambda _ { r f } = 1 , \lambda _ { s m } = 0 . 3$ for loss weights.

<!-- image-->  
Fig. 4. Qualitative comparison of novel view synthesis results on the RGBT-Scenes dataset. We compare ThermoSplat against state-of-the-art multi-spectral reconstruction methods ThermalGaussian [5], MS-Splattingv2 [21], MMOne [7], and the 3DGS baseline [4]. Our method generates more accurate rendering results and structural details.

## B. Results and Comparisons

We evaluate the novel view synthesis performance on the test set to validate the rendering quality of our method against other state-of-the-art methods. As illustrated in Fig. 4, our method produces results with finer texture details and fewer visual artifacts compared to existing approaches and the 3DGS baseline. Specifically, our model performs better in recovering complex structures that are often blurred or misaligned in the baseline reconstructions, especially on the RGB branch, which is attributed to the proposed cross-modality modulation that effectively leverages structural priors of the scene.

As shown in Tab. I, our method achieves superior performance compared to state-of-the-art baselines across most scenes. Specifically, our model attains the highest average scores in all three metrics for both RGB and thermal modalities. Notably, on the average PSNR, our method outperforms the second-best competitor (MMOne [7]) by 0.34 dB in the RGB spectrum and 0.19 dB in the thermal spectrum. The consistent improvement in SSIM and LPIPS further demonstrates our modelâs capability to reconstruct fine-grained structural details and maintain perceptual fidelity.

While MMOne [7] and MS-Splattingv2 [21] show competitive results in specific scenes (e.g., Dim and Trk), our method demonstrates more robust generalization across diverse environments. These results validate that our modality modulation and geometric decoupling strategy successfully resolves the discrepancies between modalities without compromising the reconstruction quality of the individual branches.

## C. Ablation

To verify the contribution of each design element, we conduct ablation experiments as summarized in Tab. II. First, the comparison between our full model and the âMLP-basedâ variant demonstrates the advantage of our feature-guided modulation over a standard decoding structure, as the former better leverages spatial-aware features for high-quality appearance synthesis. Second, removing the geometric decoupling mechanism (âw.o. geo. decoup.â) leads to a consistent performance decline in both modalities, confirming that isolating physical geometry from modality-specific radiation is essential for robust RGBT scene modeling. Finally, the exclusion of latent constraints (âw.o. fea(rgb/th)â) results in degradation of perceptual details, which indicates that the feature-level reconstruction loss Lf eatrec is crucial for encouraging the model to capture fundamental structural and intensity distributions. Collectively, these results validate that the synergy of the proposed modulation mechanism, geometric decoupling, and latent supervision ensures optimal RGBT reconstruction.

## V. CONCLUSION

In this paper, we present ThermoSplat, a novel crossmodal 3D Gaussian Splatting framework designed for highfidelity RGBT scene reconstruction. To effectively bridge the gap between visible and thermal modalities, we introduce a Spectrum-Aware Adaptive Modulation mechanism that leverages thermal structural priors to guide visible texture synthesis. Furthermore, to address the inherent geometric inconsistencies caused by disparate physical sensing properties, we propose a Modality-Adaptive Geometric Decoupling scheme, which enables the model to accurately represent independent spectral characteristics without compromising spatial alignment. Extensive experiments on the RGBT-Scenes dataset demonstrate that our approach achieves state-of-the-art performance in both rendering quality and structural accuracy. By integrating explicit geometric representations with implicit neural feature modulation, ThermoSplat provides a robust and efficient solution for multi-spectral scene understanding in visually degraded environments.

TABLE I  
QUANTITATIVE EVALUATION OF RGB AND THERMAL (T) RENDERING RESULTS.
<table><tr><td>M</td><td>Metric</td><td>Method</td><td>Dim</td><td>DS</td><td>Ebk</td><td>RB</td><td>Trk</td><td>RK</td><td>Bldg</td><td>â¡</td><td>Pt</td><td>LS</td><td>Avg.</td></tr><tr><td>RGB</td><td>PSNR â</td><td>3DGS</td><td>23.27</td><td>21.18</td><td>26.17</td><td>28.23</td><td>22.45</td><td>20.74</td><td>21.80</td><td>24.40</td><td>25.65</td><td>20.18</td><td>23.41</td></tr><tr><td></td><td></td><td>ThermalGaussian</td><td>24.38</td><td>21.76</td><td>26.85</td><td>28.12</td><td>24.17</td><td>23.14</td><td>24.19</td><td>24.55</td><td>25.48</td><td>21.71</td><td>24.44</td></tr><tr><td></td><td></td><td>MS-Splattingv2</td><td>24.06</td><td>21.18</td><td>26.87</td><td>28.12</td><td>24.54</td><td>23.42</td><td>23.90</td><td>23.77</td><td>26.20</td><td>22.05</td><td>24.41</td></tr><tr><td></td><td></td><td>MMOne</td><td>24.65</td><td>22.05</td><td>27.43</td><td>29.03</td><td>23.96</td><td>24.12</td><td>24.16</td><td>25.65</td><td>26.01</td><td>2.81</td><td>24.89</td></tr><tr><td></td><td></td><td>Ours</td><td>24.59</td><td>22.12</td><td>27.21</td><td>28.96</td><td>24.31</td><td>24.20</td><td>24.14</td><td>25.98</td><td>26.48</td><td>24.31</td><td>25.23</td></tr><tr><td></td><td>SSIM â</td><td>3DGS</td><td>0.842</td><td>0.771</td><td>0.902</td><td>0.917</td><td>0.810</td><td>0.765</td><td>0.827</td><td>0.875</td><td>0.867</td><td>0.688</td><td>0.826</td></tr><tr><td></td><td></td><td>ThermalGaussian</td><td>0.858</td><td>0.797</td><td>0.905</td><td>0.920</td><td>0.840</td><td>0.822</td><td>0.849</td><td>0.884</td><td>0.855</td><td>0..739</td><td>0.847</td></tr><tr><td></td><td></td><td>MS-Splattingv2</td><td>0.859</td><td>0.788</td><td>0.914</td><td>0.922</td><td>0.859</td><td>0.827</td><td>0.855</td><td>0.877</td><td>0.878</td><td>0.739</td><td>0.852</td></tr><tr><td></td><td></td><td>MMOne</td><td>0.862</td><td>0.810</td><td>0.918</td><td>0.916</td><td>0.845</td><td>0.842</td><td>0.847</td><td>0.897</td><td>0.876</td><td>0.727</td><td>0.854</td></tr><tr><td></td><td></td><td>Ours</td><td>0.872</td><td>00.818</td><td>0.934</td><td>0.941</td><td>0.859</td><td>0.841</td><td>0.858</td><td>0.911</td><td>0.886</td><td>0.788</td><td>0.871</td></tr><tr><td></td><td>LPIPS â</td><td>3DGS</td><td>0.199</td><td>0.271</td><td>0.169</td><td>0.197</td><td>0.244</td><td>0.220</td><td>0.183</td><td>0.193</td><td>0.177</td><td>0.289</td><td>0.214</td></tr><tr><td></td><td></td><td>ThermalGaussian</td><td>0.194</td><td>0.253</td><td>0.169</td><td>00.199</td><td>0.211</td><td>0.184</td><td>0.170</td><td>0.186</td><td>0.195</td><td>0.268</td><td>0.203</td></tr><tr><td></td><td></td><td>MS-Splattingv2</td><td>0.150</td><td>0.224</td><td>0.145</td><td>0.197</td><td>0.170</td><td>0.141</td><td>0.145</td><td>0.161</td><td>0.132</td><td>0.211</td><td>0.168</td></tr><tr><td></td><td></td><td>MMOne</td><td>0.203</td><td>0.254</td><td>0.160</td><td>0.235</td><td>0.226</td><td>0.178</td><td>0.184</td><td>0.183</td><td>0.178</td><td>0.291</td><td>0.209</td></tr><tr><td>T</td><td></td><td>Ours</td><td>0.155</td><td>0.204</td><td>0.121</td><td>0.164</td><td>0.166</td><td>0.130</td><td>0.131</td><td>0.138</td><td>0.136</td><td>0.180</td><td>0.153</td></tr><tr><td></td><td>PSNR â</td><td>3DGS</td><td>25.99</td><td>18.71</td><td>20.61</td><td>26.55</td><td>25.30</td><td>26.45</td><td>26.83</td><td>29.69</td><td>24.09</td><td>18.48</td><td>24.27</td></tr><tr><td></td><td></td><td>ThermalGaussian</td><td>26.46</td><td>22.28</td><td>23.31</td><td>27.17</td><td>25.88</td><td>26.33</td><td>26.72</td><td>29.86</td><td>26.16</td><td>22.27</td><td>25.64</td></tr><tr><td></td><td></td><td>MS-Splattingv2</td><td>26.06</td><td>21.43</td><td>23.32</td><td>25.44</td><td>26.08</td><td>27.24</td><td>26.89</td><td>29.98</td><td>27.01</td><td>22.64</td><td>25.61</td></tr><tr><td></td><td></td><td>MOne</td><td>26.90</td><td>21.81</td><td>23.79</td><td>27.39</td><td>25.44</td><td>27.65</td><td>27.06</td><td>30.27</td><td>26.05</td><td>22.52</td><td>25.89</td></tr><tr><td></td><td></td><td>Ours</td><td>25.99</td><td>21.54</td><td>22.95</td><td>26.83</td><td>26.25</td><td>28.48</td><td>27.45</td><td>29.78</td><td>27.00</td><td>24.50</td><td>26.08</td></tr><tr><td></td><td>SSIM â</td><td>3DGS</td><td>0.889</td><td>0.787</td><td>0.812</td><td>0.914</td><td>0.863</td><td>0.922</td><td>0.896</td><td>0.892</td><td>0.867</td><td>0.768</td><td>0.861</td></tr><tr><td></td><td></td><td>ThermalGaussian</td><td>0.886</td><td>0.835</td><td>0.862</td><td>0.919</td><td>0.874</td><td>0.922</td><td>0.888</td><td>0.896</td><td>0.883</td><td>0.850</td><td>0.882</td></tr><tr><td></td><td></td><td>MS-Splattingv2</td><td>0.876</td><td>0.803</td><td>0.855</td><td>0.900</td><td>0.871</td><td>0.927</td><td>0.888</td><td>0.890</td><td>0.903</td><td>0.853</td><td>0.877</td></tr><tr><td></td><td></td><td>MMOne</td><td>0.894</td><td>0.840</td><td>0.874</td><td>0.926</td><td>0.870</td><td>0.933</td><td>0.902</td><td>0.906</td><td>0.895</td><td>0.861</td><td>0.890</td></tr><tr><td></td><td></td><td>Ours</td><td>0.890</td><td>0.839</td><td>0.865</td><td>0.928</td><td>0.889</td><td>0.941</td><td>0.909</td><td>0.910</td><td>0.912</td><td>0.889</td><td>0.897</td></tr><tr><td></td><td>LPIPS â</td><td>3DGS</td><td>0.127</td><td>0.259</td><td>0.307</td><td>0.209</td><td>0.142</td><td>0.126</td><td>0.185</td><td>0.091</td><td>0.227</td><td>0.378</td><td>0.205</td></tr><tr><td></td><td></td><td>ThermalGaussian</td><td>0.129</td><td>0.210</td><td>0.203</td><td>0.198</td><td>0.136</td><td>0.124</td><td>0.177</td><td>0.091</td><td>0.181</td><td>0.248</td><td>0.170</td></tr><tr><td></td><td></td><td>MS-Splattingv2</td><td>0.092</td><td>0.164</td><td>0.148</td><td>0.133</td><td>0.107</td><td>0.073</td><td>0.103</td><td>0.064</td><td>0.075</td><td>0.177</td><td>0.114</td></tr><tr><td></td><td></td><td>MMOne</td><td>0.125</td><td>0.194</td><td>0.201</td><td>0.213</td><td>0.142</td><td>0.17</td><td>0.198</td><td>0.083</td><td>0.205</td><td>0.272</td><td>0.176</td></tr><tr><td></td><td></td><td>Ours</td><td>0.100</td><td>0.149</td><td>0.149</td><td>0.096</td><td>0.094</td><td>0.059</td><td>0.085</td><td>0.057</td><td>0.075</td><td>0.139</td><td>0.101</td></tr></table>

TABLE II

ABLATION STUDY. WE CONDUCTED ABLATION EXPERIMENTS ON DIFFERENT MODULES OF OUR PIPELINE.
<table><tr><td rowspan="2">Method Variant</td><td colspan="3">RGB Modality</td><td colspan="3">Thermal Modality</td></tr><tr><td>PSNR â</td><td>SSIMâ</td><td>LPIPS â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPS â</td></tr><tr><td>MLP-based</td><td>25.14</td><td>0.869</td><td>0.154</td><td>25.88</td><td>0.895</td><td>0.104</td></tr><tr><td>w.o. geo. decoup.</td><td>25.07</td><td>0.868</td><td>0.159</td><td>25.82</td><td>0.892</td><td>0.106</td></tr><tr><td>w.o. hybrid rgb</td><td>25.07</td><td>0.867</td><td>0.157</td><td>25.86</td><td>0.893</td><td>0.105</td></tr><tr><td>w.o. fea(th)</td><td>24.98</td><td>0.868</td><td>0.155</td><td>25.77</td><td>0.894</td><td>0.107</td></tr><tr><td>w.o. fea(rgb)</td><td>24.88</td><td>0.858</td><td>0.190</td><td>25.93</td><td>0.897</td><td>0.104</td></tr><tr><td>Ours</td><td>25.23</td><td>0.871</td><td>0.153</td><td>26.08</td><td>0.897</td><td>0.101</td></tr></table>

limitations. Despite the promising results, ThermoSplat has certain limitations that offer directions for future research. First, the current geometric decoupling scheme primarily focuses on the thermal branch; however, in scenarios with extreme glass reflections or high-transparency surfaces, more complex multi-modal interactions might be required to fully resolve depth ambiguities. Second, the use of latent feature modulation introduces additional memory overhead during the neural decoding phase compared to vanilla 3DGS. Future work will explore more lightweight modulation architectures and investigate the potential of extending this framework to other spectral domains, such as near-infrared or hyperspectral data, to further enhance its versatility and robustness in all-weather environmental perception.

## ACKNOWLEDGMENTS

This paper is supported in part by the National Natural Science Foundation of China (Grant No. 62402274) and the Start-up Funding of Fuzhou University (Grant No. XRC-25164) to Zhaoqi Su; in part by the Education and Scientific Research Project for Middle-aged and Young Teachers of Fujian Province, China (Grant No. JZ250004) to Zhipeng

Su; in part by the Special Fund for Promoting High-Quality Development of Marine and Fishery Industries in Fujian Province (Grant No. FJHYF-L-2025-07-005) to Xiaoqiang Lu. The authors would like to acknowledge the use of Gemini to improve the language and readability of the manuscript during the writing process.

## REFERENCES

[1] Y. Y. Lin, X.-Y. Pan, S. Fridovich-Keil, and G. Wetzstein, âThermalnerf: Thermal radiance fields,â in 2024 IEEE International Conference on Computational Photography (ICCP). IEEE, 2024, pp. 1â12.

[2] M. Hassan, F. Forest, O. Fink, and M. Mielle, âThermonerf: A multimodal neural radiance field for joint rgb-thermal novel view synthesis of building facades,â Adv. Eng. Inform., vol. 65, no. PD, May 2025. [Online]. Available: https://doi.org/10.1016/j.aei.2025.103345

[3] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[4] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[5] R. Lu, H. Chen, Z. Zhu, Y. Qin, M. Lu, L. zhang, C. Yan, and a. xue, âThermalgaussian: Thermal 3d gaussian splatting,â in International Conference on Representation Learning, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, Eds., vol. 2025, 2025, pp. 1105â1117. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2025/ file/03bdba50e3741ac5e3eaa0e55423587e-Paper-Conference.pdf

[6] L. Meyer, J. Grun, M. Weiherer, B. Egger, M. Stamminger, and Â¨ L. Franke, âMulti-spectral gaussian splatting with neural color representation,â arXiv preprint arXiv:2506.03407, 2025.

[7] Z. Gu and B. Wang, âMmone: Representing multiple modalities in one scene,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 1088â1098.

[8] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, âFilm: Visual reasoning with a general conditioning layer,â in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[9] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, âMip-nerf: A multiscale representation for antialiasing neural radiance fields,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 5855â5864.

[10] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM transactions on graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[11] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, T. Wang, A. Kristoffersen, J. Austin, K. Salahi, A. Ahuja et al., âNerfstudio: A modular framework for neural radiance field development,â in ACM SIGGRAPH 2023 conference proceedings, 2023, pp. 1â12.

[12] B. Fei, J. Xu, R. Zhang, Q. Zhou, W. Yang, and Y. He, â3d gaussian splatting as new era: A survey,â IEEE Transactions on Visualization and Computer Graphics, 2024.

[13] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, âMip-splatting: Alias-free 3d gaussian splatting,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 19 447â19 456.

[14] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, âScaffold-gs: Structured 3d gaussians for view-adaptive rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 654â20 664.

[15] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, and A. Kadambi, âFeature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 676â21 685.

[16] Z. Dai, T. Liu, and Y. Zhang, âEfficient decoupled feature 3d gaussian splatting via hierarchical compression,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 11 156â11 166.

[17] R.-Z. Qiu, G. Yang, W. Zeng, and X. Wang, âLanguage-driven physicsbased scene synthesis and editing via feature splatting,â in European Conference on Computer Vision. Springer, 2024, pp. 368â383.

[18] J. Xu, M. Liao, R. P. Kathirvel, and V. M. Patel, âLeveraging thermal modality to enhance reconstruction in low-light conditions,â in European Conference on Computer Vision. Springer, 2024, pp. 321â339.

[19] T. Ye, Q. Wu, J. Deng, G. Liu, L. Liu, S. Xia, L. Pang, W. Yu, and L. Pei, âThermal-nerf: Neural radiance fields from an infrared camera,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 1046â1053.

[20] M. Ozer, M. Weiherer, M. Hundhausen, and B. Egger, âExploring Â¨ multi-modal neural scene representations with applications on thermal imaging,â in European Conference on Computer Vision. Springer, 2024, pp. 82â98.

[21] J. Grun, L. Meyer, M. Weiherer, B. Egger, M. Stamminger, and Â¨ L. Franke, âTowards Integrating Multi-Spectral Imaging with Gaussian Splatting,â in Vision, Modeling, and Visualization, B. Egger and T. Gunther, Eds. The Eurographics Association, 2025. Â¨

[22] Q. Ma, C. Zou, D. Wang, J. Wang, L. Xiang, and Z. He, âBeyond darkness: Thermal-supervised 3d gaussian splatting for low-light novel view synthesis,â arXiv preprint arXiv:2511.13011, 2025.

[23] Q. Chen, S. Shu, and X. Bai, âThermal3d-gs: Physics-induced 3d gaussians for thermal infrared novel-view synthesis,â in European Conference on Computer Vision. Springer, 2024, pp. 253â269.

[24] K. Yang, Y. Liu, Z. Cui, Y. Liu, M. Zhang, S. Yan, and Q. Wang, âNtrgaussian: Nighttime dynamic thermal reconstruction with 4d gaussian splatting based on thermodynamics,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 691â700.

[25] M. Nam, W. Park, M. Kim, H. Hur, and S. Lee, âVeta-gs: Viewdependent deformable 3d gaussian splatting for thermal infrared novelview synthesis,â in 2025 IEEE International Conference on Image Processing (ICIP). IEEE, 2025, pp. 965â970.

[26] S. N. Sinha, H. Graf, and M. Weinmann, âSpectralgaussians: Semantic, spectral 3d gaussian splatting for multi-spectral scene representation, visualization and analysis,â ISPRS Journal of Photogrammetry and Remote Sensing, vol. 227, pp. 789â803, 2025.

[27] C. Thirgood, O. Mendez, E. Ling, J. Storey, and S. Hadfield, âHypergs: Hyperspectral 3d gaussian splatting,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5970â5979.

[28] S. Elfwing, E. Uchibe, and K. Doya, âSigmoid-weighted linear units for neural network function approximation in reinforcement learning,â Neural networks, vol. 107, pp. 3â11, 2018.

[29] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586â595.