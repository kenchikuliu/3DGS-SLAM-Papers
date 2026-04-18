# LL-GaussianImage: Efficient Image Representation for Zero-shot Low-Light Enhancement with 2D Gaussian Splatting

Yuhan Chen, Wenxuan Yu, Guofa Li, Senior Member, IEEE, Yijun Xu, Ying Fang, Yicui Shi, Long Cao, Wenbo Chu, Keqiang Li

Abstractâ2D Gaussian Splatting (2DGS) is an emerging explicit scene representation method with significant potential for image compression due to high fidelity and high compression ratios. However, existing low-light enhancement algorithms operate predominantly within the pixel domain. Processing 2DGScompressed images necessitates a cumbersome decompressionenhancement-recompression pipeline, which compromises efficiency and introduces secondary degradation. To address these limitations, we propose LL-GaussianImage, the first zero-shot unsupervised framework designed for low-light enhancement directly within the 2DGS compressed representation domain. Three primary advantages are offered by this framework. First, a semantic-guided Mixture-of-Experts enhancement framework is designed. Dynamic adaptive transformations are applied to the sparse attribute space of 2DGS using rendered images as guidance to enable compression-as-enhancement without full decompression to a pixel grid. Second, a multi-objective collaborative loss function system is established to strictly constrain smoothness and fidelity during enhancement, suppressing artifacts while improving visual quality. Third, a twostage optimization process is utilized to achieve reconstruction-asenhancement. The accuracy of the base representation is ensured through single-scale reconstruction and network robustness is enhanced. High-quality enhancement of low-light images is achieved while high compression ratios are maintained. The feasibility and superiority of the paradigm for direct processing within the compressed representation domain are validated through experimental results.

Index TermsâImage enhancement, Gaussian Splatting, Image compression.

## I. INTRODUCTION

MAGE signals are inherently continuous and complex, yet computer vision has long relied on discrete pixel grids as the foundational representation. While this classical format has facilitated the growth of digital image processing, its discrete nature imposes inherent limitations achieving higher fidelity and extreme compression ratios [1-4]. Following revolutionary breakthroughs in 3D scene reconstruction via Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), explicit Gaussian representations have emerged as a promising alternative for 2D images [5-6]. As an explicit scene representation, 2D Gaussian Splatting (2DGS) models an image as a collection of anisotropic Gaussians defined by attributes such as position, color, opacity, and covariance. This approach inherits the real-time rendering and high-fidelity capabilities of 3DGS while demonstrating superior potential in image compression compared to conventional pixel-based or Implicit Neural Representation (INR) methods [7-9]. The synergy between compression and representation establishes a robust framework for next-generation image processing technologies.

In practical scenarios such as autonomous driving and security, images frequently suffer from low-light conditions due to insufficient illumination or underexposure. This degradation compromises visual quality and impedes downstream high-level vision tasks, including object detection, semantic segmentation, and cross-view tracking [10-12, 20-21]. To address these challenges, various deep learning-based enhancement methods have been developed. These approaches utilize powerful non-linear fitting capabilities to achieve brightness restoration, color correction, and noise suppression at the pixel level [13-28, 34-38]. However, existing methods operate exclusively in the pixel domainâa dependency that creates significant efficiency and quality bottlenecks when processing 2DGS compressed images. Current pipelines must execute a sequence of decompression, pixel-domain enhancement, and subsequent re-compression. This workflow introduces heavy computational redundancy and cumulative errors, undermining the real-time rendering advantages of 2DGS. Furthermore, noise and artifacts from the enhancement stage are often magnified during re-compression, causing secondary damage to image information [29-31]. Consequently, achieving illumination enhancement directly within the compressed representation domain while maintaining high compression ratios is a critical and challenging problem.

<!-- image-->  
Fig. 1. Comparison of low-light image enhancement schemes across (a) pixel domain and (b) 2DGS compressed representation domain. LL-GaussianImage is the first work to perform enhancement directly in the 2DGS compressed domain.

To address these limitations and bridge the gap between explicit compressed representations and low-level vision tasks, this work draws inspiration from recent advancements in 2DGS image compression and Gaussian semantic editing [1-4, 29-33]. It is recognized that illumination attributes reside within both the pixel values and the spatial distribution and attribute parameters of 2D Gaussians. Consequently, LL-GaussianImage is proposed as the first zero-shot unsupervised framework for low-light enhancement directly within the 2DGS compressed domain. This framework introduces a novel compression-asenhancement paradigm that explicitly decouples geometric fitting from illumination adjustment. The direct enhancement process within the 2DGS compressed representation domain is illustrated in Fig. 1. Specifically, a 2DGS compression strategy first achieves robust reconstruction of image structures to ensure geometric accuracy. Then, geometric attributes such as position and covariance are frozen and a unified Gaussian enhancement network is utilized for the optimization of illumination attributes. Multi-scale spatial features are extracted from rendered views using a lightweight CNN backbone to guide a set of learnable Multi-layer Perceptron (MLP) color operators. By dynamically predicting spatial mixing weights, the model fuses operator outputs in a residual form. This process applies spatially adaptive modulation to Gaussian color attributes, restoring visibility while preserving original data distribution characteristics.

A physics-oriented multi-objective loss function system is designed to constrain the zero-shot training process and prevent color distortion. The objective function incorporates a robust hue preservation term in the HSV space to maintain chromatic fidelity, a histogram contrast term to correct the global illumination distribution, and a spatial consistency constraint to suppress artifacts. These constraints enable the model to learn optimal enhancement parameters without paired ground-truth data. The primary contributions are summarized as follows:

LL-GaussianImage is proposed as the first zero-shot unsupervised framework for low-light enhancement in the 2DGS compressed domain, pioneering the compression-as-enhancement paradigm. High-fidelity low-light image restoration is achieved by decoupling geometric reconstruction from illumination enhancement. The structural integrity of the compressed representation is strictly maintained throughout the process.

A semantic-guided Mixture-of-Experts enhancement framework is designed. Multiple residual MLP color operators are orchestrated through a lightweight network. This framework achieves fine-grained and spatially varying adjustments to Gaussian color attributes while balancing local detail enhancement with global illumination restoration.

A robust zero-shot loss function is constructed through the integration of novel hue preservation constraints and histogram contrast regularization. These constraints guide the optimization process, ensure natural illumination and vivid colors while suppressing secondary artifacts inherent in compressed domain processing.

## II. RELATED WORKS

LL-GaussianImage is situated at the intersection of explicit neural representations and low-level vision tasks. It focuses on the emerging paradigm of direct image enhancement within the compressed domain. Consequently, related works are organized into three categories: recent advancements in low-light image enhancement, an overview of the development of Gaussian Splatting, the overall development of image in domain compression.

## A. Low-light image enhancement

Low-light image enhancement (LLIE) aims to improve the quality of images captured under insufficient illumination by restoring brightness and contrast while suppressing noise. Early traditional techniques primarily relied on histogram equalization and Retinex theory. Specifically, LIME [14] estimates the illumination map by applying structural priors through optimization. However, these handcrafted prior-based methods often depend on complex parameter tuning. Such approaches frequently struggle to balance noise suppression and detail restoration in highly complex scenes.

The field has been increasingly dominated by data-driven methods due to superior performance enabled by advancements in deep learning. Unsupervised learning paradigms have been adopted because paired low-light and normal-light images are difficult to obtain in practical scenarios. EnlightenGAN [19] utilizes generative adversarial networks to enhance images without paired supervision. A series of zero-reference deep curve estimation methods were developed by Guo et al. [15-16] to reduce computational complexity and enhance robustness. These approaches adjust the dynamic range of images by iteratively estimating pixel-level high-order curves to eliminate the need for reference images. Similarly, Pan et al. [23] optimized the curve estimation process via Chebyshev polynomials to achieve superior performance. Regarding the challenge of limited training data, Fu et al. [25] explored learning enhancers from paired low-light instances. Zhang et al. [26] introduced a noise autoregressive paradigm to achieve denoising and illumination enhancement without task-specific data, effectively expanding the boundaries of learning-based methods.

<!-- image-->  
Fig. 2. Overview of the proposed two-stage training framework for LL-GaussianImage. In Stage 1, a 2DGS compression-reconstruction strategy is employed for robust structural reconstruction to ensure the geometric accuracy of the explicit representation. In Stage 2, geometric attributes (position and covariance) are frozen, while a lightweight CNN backbone extracts multi-scale spatial features from rendered views to guide a set of learnable MLP color operators. Dynamically predicted spatial mixing weights facilitate the fusion of operator outputs in a residual form. This process modulates Gaussian color attributes in a spatially adaptive manner to restore visibility while preserving the original data distribution.

Despite significant progress, pure data-driven methods are often regarded as black boxes lacking physical interpretability. Several recent approaches address this limitation by integrating physical models into deep architectures. Liu et al. [18] designed a Retinex-inspired unfolding architecture that incorporates a cooperative prior search strategy. Zheng et al. [22] proposed an adaptive unfolding Total Variation network to map the iterative steps of traditional optimization into learnable modules. Furthermore, Wu et al. [27] introduced URetinex-Net, which unfolds the Retinex decomposition into an implicit regularization model to effectively combine data-driven and model-driven methodologies.

Beyond the enhancement of interpretability, lightweight architectures and real-time performance have emerged as significant focal points within the field alongside interpretability due to the proliferation of mobile applications. Ma et al. [17] developed a fast, flexible, and robust illumination enhancement framework. To accommodate resourceconstrained devices, Liu et al. [24] introduced EFINet, which improves efficiency through an iterative strategy of enhancement and fusion. Recent architectural designs emphasize efficiency, as demonstrated by the multi-scale residual (FMR-Net) and re-parameterized residual (FRR-Net) networks proposed by Chen et al. [20-21]. Additionally, Bai et al. [13] and Chen et al. [52] explored ultra-lightweight architectures to achieve real-time processing at minimal computational costs.

Generative models and implicit neural representations offer innovative perspectives for low-light illumination enhancement.

Specifically, Jiang et al. [34] and Lin et al. [35] explored Retinex-based latent diffusion models and training-free guided diffusion strategies to generate high-quality normal-light images within unsupervised settings. Huang et al. [36] subsequently proposed a zero-shot latent diffusion enhancement method to further improve visual quality. To overcome the limitations of convolutional neural networks regarding resolution and continuity, Yang et al. [37] integrated implicit neural representations into collaborative enhancement tasks. Similarly, Chobola et al. [38] achieved efficient and detail-rich image reconstruction through context-based fast neural implicit representations. The ZERO-IG framework [28] illustrates the potential of zero-shot illumination-guided joint denoising and adaptive enhancement. These advancements suggest that the integration of robust generative priors and innovative feature representations provides a critical pathway for addressing complex low-light degradation.

## B. Gaussian Splatting

3D Gaussian Splatting is established as a fundamental explicit scene representation paradigm in computer graphics and vision [5]. This approach addresses the training and rendering efficiency limitations of traditional implicit Neural Radiance Fields by parameterizing scenes as collections of discrete and learnable 3D Gaussian primitives [6]. Each primitive possesses anisotropic geometric attributes including position, rotation, and scale alongside appearance attributes such as opacity and spherical harmonic coefficients. A customized tile-based differentiable rasterization pipeline enables the synthesis of photo-realistic novel views. This framework achieves real-time or ultra-real-time rendering speeds and provides a robust mathematical foundation for complex scene modeling.

<!-- image-->  
Fig. 3. Visualization of semantic mixing weights for ?? = 16. The unsupervised model spontaneously generates attention maps with semantic awareness., where individual weight channels focus on distinct semantic regions to guide spatially adaptive color operations.

<!-- image-->  
Fig. 4. Visual decomposition of the Mixture-of-Experts enhancement module. The global application of each color operator $\Phi _ { k }$ is visualized, where the final identity operator serves as a stable anchor. The system learns a dictionary of diverse exposure and tonal styles, which are then fused locally to generate the final enhanced result.

3DGS has been rapidly extended to large-scale dynamic scene modeling and autonomous driving simulation through efficient explicit representation capabilities. Street Gaussians and Periodic Vibration Gaussian capture non-rigid object motions in complex urban environments by introducing temporal dimensions and vibration models [39, 44]. For specific large-span spatiotemporal motion reconstruction, recent methods incorporate physical events to enhance accuracy [42-43]. In the context of autonomous driving perception, DrivingGaussian implements composite Gaussian splatting for surround-view dynamic scenes. DriveDreamer4D and ReconDreamer integrate world models to achieve high-quality 4D driving scene representation and reconstruction [45, 49-50]. To overcome storage and rendering challenges in large-scale scenarios, CityGaussian and Momentum-GS employ tile-based training or momentum distillation strategies to maintain high fidelity while achieving city-level real-time rendering [47-48].

Current advancements emphasize reconstruction efficiency and generative capabilities alongside breakthroughs in scene scale and dynamics. Speedy-Splat [40] increases computational efficiency during training and rendering by utilizing sparse pixels and primitives. Similarly, MVSGaussian [41] and

GaussianPro [46] improve the generalizability and robustness of Gaussian representations via multi-view stereo priors and progressive propagation strategies. Gaussian Splatting also extends into generative tasks. For instance, GaussianDreamer [51] bridges 2D diffusion models with 3D Gaussian representations to facilitate rapid text-to-3D asset generation. These developments establish 3DGS as a flexible and versatile visual primitive. The explicit nature of this representation provides a robust foundation for its adaptation to broader vision tasks.

## C. Gaussian Splatting Based Image Representation

The success of 2DGS in 3D scenes has inspired the adaptation of this paradigm for 2D image representation. The GaussianImage framework was introduced by Zhang et al. as a pioneering effort to adapt 2DGS for compact image representation [3]. This framework discretizes image signals using eight parameters covering position, covariance, and color. An efficient rendering algorithm based on cumulative summation replaces the traditional alpha blending mechanism to improve computational efficiency. This design significantly reduces memory consumption and achieves rendering frame rates far exceeding those of implicit representations. Furthermore, integrating vector quantization techniques facilitates the construction of efficient image codecs. This strategy achieves competitive rate-distortion performance while significantly reducing storage requirements. Based on this framework, Zeng et al. [2] develop a generalizable and adaptive representation mechanism that dynamically allocates Gaussian primitives to improve fitting flexibility. Jiang et al. [4] apply sparse Gaussian representations to dataset distillation and demonstrate the effectiveness of this paradigm for feature extraction. To address large-scale image representation, Zhu et al. [1] introduce multi-level Gaussian Splatting strategies to mitigate detail loss at high resolutions.

As foundational representation capabilities mature, 2DGSbased methods have expanded into complex downstream vision tasks. In super-resolution (SR), GaussianSR and Pixel-to-Gaussian [30-31] overcome traditional grid pixel constraints. These methods achieve arbitrary-scale image reconstruction and detail enhancement by assigning learnable Gaussian kernels to each pixel or establishing continuous pixel-to-Gaussian mappings. Omri et al. [29] further expand these semantic boundaries by utilizing 2DGS as a compressed representation to facilitate efficient vision-language model alignment. Such advancements underscore the potential of this representation for multimodal tasks.

## III. PROPOSED METHOD

The LL-GaussianImage framework is presented in this section. As illustrated in Fig. 2, the architecture is partitioned into two components consisting of 2DGS-based image primitive reconstruction and semantic-guided zero-shot unsupervised Gaussian Splatting enhancement. LL-GaussianImage is established as the initial framework to achieve low-light image enhancement within the 2DGS compressed representation domain. This development establishes a foundation for future low-level vision tasks involving 2DGS-based compressed images. Subsequent sections provide a comprehensive analysis of the methodology and architecture.

## A. 2D Gaussian Splatting-based Image Primitive Reconstruction

The input low-light image $I _ { l o w } \in \mathfrak { R } ^ { H \times W \times 3 }$ is transformed into a set of discrete and differentiable geometric primitive representations in the initial stage of LL-GaussianImage. Distinguishable from conventional pixel grids or implicit neural representations, 2DGS is utilized as the explicit parameterized carrier for the image. This discretized format compactly preserves high-frequency texture details and establishes a geometry-aware data structure for subsequent color-space decoupling and illumination enhancement.

Explicit Parameterization of 2D Gaussian Primitives. Specifically, an image is modeled as a set $\Im =$ $\{ G _ { 1 } , G _ { 2 } , \dots , G _ { N } \}$ where ?? denotes the total number of Gaussian primitives. Each 2DGS primitive $G _ { i }$ is defined as a probability density distribution over the image plane $\Omega \subset \mathbb { R } ^ { 2 }$ Unlike 3DGS, these primitives are established directly within the 2D image space to eliminate geometric ambiguities inherent in projection processes. The i-th primitive is uniquely determined by a set of learnable parameters:

$$
\Theta _ { i } = \mu _ { i } , \Sigma _ { i } , c _ { i } , o _ { i } ,\tag{1}
$$

where $\mu _ { i } = ( u _ { i } , v _ { i } ) ^ { T } \in \mathfrak { R } ^ { 2 }$ represents the central coordinates of the primitive on the image plane; $c _ { i } \in \Re ^ { 3 }$ denotes the RGB color attributes; $\Sigma _ { i } \in \Re ^ { 2 \times 2 }$ is the 2D covariance matrix that dictates the anisotropic shape and spatial extent of the primitive; and $o _ { i } \in [ 0 , 1 ]$ signifies the opacity, which is utilized to govern the occlusion relationship with the background. Mathematically, the response value of the i-th Gaussian primitive at pixel coordinate $p _ { i } \in \Re ^ { 2 }$ is defined by the probability density function as follows:

$$
G _ { i } ( p ) = \exp { ( - \frac { 1 } { 2 } ( p - \mu _ { i } ) ^ { T } \Sigma _ { i } ^ { - 1 } ( p - \mu _ { i } ) ) } .\tag{2}
$$

Ensuring the physical validity and positive semi-definiteness of the covariance matrix $\Sigma _ { i }$ necessitates a specific parameterization during optimization. Rather than performing direct gradient descent on the elements of $\Sigma _ { i }$ , the matrix is decomposed into the product of a scaling matrix $S _ { i }$ and a rotation matrix $R _ { i }$

$$
\begin{array} { r l } & { \Sigma _ { i } = R _ { i } S _ { i } S _ { i } ^ { T } R _ { i } ^ { T } = \left[ \begin{array} { l l } { c o s \theta } & { - s i n \theta } \\ { s i n \theta } & { c o s \theta } \end{array} \right] \left[ \begin{array} { l l } { s _ { x } } & { \ 0 } \\ { 0 } & { s _ { y } } \end{array} \right] } \\ & { \quad \left[ \begin{array} { l l } { s _ { x } } & { \ 0 } \\ { 0 } & { s _ { y } } \end{array} \right] ^ { T } \left[ \begin{array} { l l } { c o s \theta } & { - s i n \theta } \\ { s i n \theta } & { c o s \theta } \end{array} \right] ^ { T } . } \end{array}\tag{3}
$$

This parameterization facilitates adaptive stretching and rotation of Gaussian primitives to accurately model edges and texture flows. Consequently, these primitives manifest as elongated ellipses in high-frequency regions and isotropic disks in low-frequency areas.

Accelerated computation via Conic matrix representation. An implicit conic parameterization method circumvents the frequent calculation of the inverse matrix $\Sigma _ { i } ^ { - 1 }$ for individual pixels to improve computational efficiency. The inverse covariance matrix is defined as the Conic matrix $Q _ { i }$ based on the properties of the Gaussian exponential function:

$$
Q _ { i } = \Sigma _ { i } ^ { - 1 } = { \bigl [ } { \begin{array} { l l } { A } & { B } \\ { B } & { C } \end{array} } { \bigr ] } .\tag{4}
$$

The offset between the pixel coordinate vector $p =$ $( x , y ) ^ { T }$ and the Gaussian center $\boldsymbol { \mu _ { i } } = ( u _ { i } , v _ { i } ) ^ { T }$ is denoted by $\Delta p = p - \mu _ { i }$ . The exponential component of the Gaussian response function is simplified into a scalar form to eliminate the requirement for matrix operations:

$$
\begin{array} { c } { { P ( p , \mu _ { i } , Q _ { i } ) = \displaystyle - \frac { 1 } { 2 } \varDelta p ^ { T } Q _ { i } \varDelta p = \displaystyle - \frac { 1 } { 2 } [ A ( x - u _ { i } ) ^ { 2 } ] } } \\ { { + 2 B ( x - u _ { i } ) ( y - v _ { i } ) + C ( y - v _ { i } ) ^ { 2 } . } } \end{array}\tag{5}
$$

This formulation significantly accelerates tile-based parallel computing and enables real-time rendering of tens of thousands of Gaussian primitives on GPUs.

Differentiable Rasterization and Image Synthesis. A differentiable Î±-blending process is executed to map the discrete set of Gaussian primitives back into the continuous image domain. The physical accumulation effect of light passing through multiple semi-transparent layers is simulated by this process. For a specific pixel location ?? , the set of overlapping primitives ??(??) is sorted based on depth. The final reconstructed color $C ( p )$ at pixel ?? is determined as the weighted sum of contributions from all relevant Gaussian primitives:

$$
C ( p ) = \sum _ { i \in \eta ( p ) } c _ { i } \cdot \alpha _ { i } ( p ) \cdot T _ { i } ( p ) ,\tag{6}
$$

where $c _ { i }$ denotes the instantaneous RGB value of the i-th Gaussian primitive; and $\alpha _ { i } ( p )$ represents the instantaneous opacity at pixel, which is jointly determined by the baseline opacity and a spatial decay term:

$$
\alpha _ { i } ( p ) = o _ { i } \cdot \exp { ( P ( p , \mu _ { i } , Q _ { i } ) ) } .\tag{7}
$$

Transmittance as light reaches the i-th primitive is represented by $T _ { i } ( p )$ . This parameter signifies the residual light intensity not occluded by the preceding ?? â 1 primitives:

$$
\begin{array} { r } { T _ { i } ( p ) = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ( p ) ) . } \end{array}\tag{8}
$$

The rasterization pipeline is fully differentiable. Consequently, pixel-level errors on the image plane are backpropagated to the parameters $\mu _ { i } \ , \ \Sigma _ { i }$ and $c _ { i }$ of each Gaussian primitive via the chain rule.

Optimization Objective and Compact Representation. Parameter set $\Theta = \cup _ { i } \Theta _ { i }$ is optimized following the establishment of the 2DGS representation within the compressed image domain. The rendered image ??Ì is required to approximate the original image $I _ { l o w }$ in both global color and structural characteristics. A composite loss function is formulated as follows:

$$
\begin{array} { r l } & { \ell _ { r e c } = ( 1 - \lambda _ { s s i m } ) \parallel \hat { I } - I _ { l o w } \parallel _ { 1 } } \\ & { + \lambda _ { s s i m } \left( 1 - M S - S S I M \left( \hat { I } , I _ { l o w } \right) \right) . } \end{array}\tag{9}
$$

Gaussian primitives manifest distinct self-organizing properties during iterative optimization. These primitives automatically migrate to high-frequency image regions including object boundaries. Adjusting the covariance matrix $\Sigma _ { i }$ facilitates the formation of elongated ellipses to fit edges accurately. A limited number of large-scale Gaussians provide sufficient coverage for smooth regions. This process ultimately yields a highly sparse parameter set $\Theta ^ { * }$

<!-- image-->  
Fig. 5. Visual comparison of LL-GaussianImage with SOTA methods on the LOL, LSRW-HUAWEI, and LSRW-NIKON datasets.

To quantify the compactness of this explicit representation, the parameter complexity of traditional pixel grids is compared against 2D Gaussian primitives. A raw pixel representation with a resolution of $H \times W \times 3$ requires degrees of freedom totaling $D o F _ { p i x e l } = H \times W \times 3$ By contrast, the Gaussian representation relies on the number of primitives ??. Each primitive is characterized by nine scalar parameters: two for position, three for covariance, three for color, and one for opacity. The total degrees of freedom are thus $D o F _ { g a u s s i a n } =$ $N \times 9 .$ . The resulting compression ratio between these representations is defined as:

$$
C r = \frac { D o F _ { p i x e l } } { D o F _ { g a u s s i a n } } = \frac { 3 H W } { 9 N } = \frac { H W } { 3 N } .\tag{10}
$$

Approximately $7 . 2 \times 1 0 ^ { 5 }$ numerical values are required by traditional representations for an image with a resolution of $6 0 0 \times 4 0 0$ in a typical experimental setup. By contrast, the proposed approach leverages spatial redundancy to reconstruct image structures using a minimal number of primitives. Under extremely compact settings where $N \approx 1 5 0 0$ , the image preserves primary structural and color characteristics. This configuration yields a compression ratio of $C r \approx 5 3 . 3$ and achieves effective encoding of continuous image signals using less than 2% of the original parameter volume. Such significant compression capability stems from the inherent properties of Gaussian primitives. A single primitive fits large-scale and lowfrequency smooth regions to eliminate the point-by-point storage of redundant information required by pixel grids. This highly decoupled and compact latent space provides the physical foundation for subsequent zero-shot illumination enhancement.

Consequently, 2DGS-based representation serves as an efficient compression format and decomposes images into a series of visual primitives with semantic potential. An ideal decoupled operation space is established for the second stage of unsupervised enhancement.

Optimization Strategy and First-Stage Training Workflow. The 2D image reconstruction task is performed under a cold start configuration. This approach is distinguished from 3D Gaussian Splatting which typically relies on sparse point clouds from Structure-from-Motion as geometric priors. An end-to-end adaptive training pipeline is established to facilitate convergence from an unordered random distribution to a highly structured image representation. The complete optimization process is formalized as a search problem within the parameter space â9??.

<!-- image-->  
Fig. 6. Visual comparison of detailed features on the LOL dataset. Magnified views of the regions marked by red boxes are provided for each method.

<!-- image-->  
Fig. 7. Visual comparison of detailed features on the LSRW dataset. Magnified views of the regions marked by red boxes are provided for each method.

Prior assumptions regarding image content are excluded. Center coordinates of the Gaussian primitives are initialized as a uniform random distribution $\mu _ { i } \sim \mu ( \Omega )$ on the image plane. Covariance matric Î£ is initialized as isotropic circle while color ?? is initialized to zero value and opacity ?? receives a small initial value to facilitate gradient flow. This initial state compels the model to rely exclusively on illumination gradients to determine the optimal geometric layout. A forward rasterization process $f ( \Theta ^ { ( t ) } ) \to { \hat { I } } ^ { ( t ) }$ is executed in each iteration ?? to generate the current rendered view. The reconstruction loss â is subsequently calculated and an automatic differentiation engine computes gradients for each primitive parameter:

$$
\begin{array} { r } { \nabla _ { \Theta } \ell = \Big [ \frac { \partial \ell } { \partial \mu _ { i } } , \frac { \partial \ell } { \partial \Sigma _ { i } } , \frac { \partial \ell } { \partial c _ { i } } , \frac { \partial \ell } { \partial o _ { i } } \Big ] . } \end{array}\tag{11}
$$

Attribute parameters exhibit varying scales and sensitivities. Coordinate variations typically exert a more substantial influence on the loss function than minor color perturbations. To address these discrepancies, the Adam optimizer facilitates a joint update of all parameters. The total iteration count ?? is set to an empirical value of 30,000.

## B. Semantically Guided Zero-Shot Unsupervised Enhancement

The discrete image primitive representation $\Im =$ $\{ ( \mu _ { i } , \Sigma _ { i } , c _ { i } ) \} _ { i = 1 } ^ { N }$ is acquired. Color attributes $c _ { i }$ are optimized in the second stage to achieve image enhancement while geometric structures $\mu _ { i }$ and $\Sigma _ { i }$ are frozen. Three core submodules are integrated into the enhancement workflow as illustrated in Fig. 2. These components consist of a semanticaware parameter estimation network, a set of differentiable color transformation operators and an iterative optimization strategy based on non-reference loss functions.

Semantic-Aware Feature Extraction and Weight Estimation. Traditional image enhancement methods typically apply a uniform transformation curve to the entire image and disregard variations in illumination response across diverse semantic regions including sky, shadows, and vegetation. Local adaptive enhancement is realized through a lightweight encoder that extracts semantic features and predicts mixing weights for each Gaussian primitive. With the low-light image ??Ì â $\Re ^ { H \times W \times 3 }$ as input, the first seven layers of a pre-trained MobileNetV2 function as the feature extractor ??(â) to capture low-level textures and mid-level semantic information. The feature extraction process is defined as follows:

$$
F = E ( I _ { l o w } ) , \ F \in \Re ^ { H ^ { \prime } \times W ^ { \prime } \times D } ,\tag{12}
$$

where ?? = 32 denotes the feature dimension; $H ^ { \prime }$ and $W ^ { \prime }$ signifies the resolution after downsampling. To map continuous image features onto the discrete Gaussian point cloud, a spatially aligned mixing weight head is designed. Specifically, the features ?? are mapped by the head network $H ( \cdot )$ into a weight space to generate a low-resolution weight map $M _ { l o w } \in \Re ^ { H ^ { \prime } \times W ^ { \prime } \times K }$ . The parameter ??denotes the number of predefined color transformation operators. To obtain a specific mixing weight vector $w _ { i } \in \Re ^ { K }$ for each Gaussian primitive $G _ { i } ,$ a bilinear interpolation operation $S ( \cdot )$ is employed

TABLE I  
PERFORMANCE COMPARISON RESULTS ON THE LOL DATASET, WHERE RED AND BLUE HIGHLIGHTS INDICATE SIGNIFY THE BEST AND SECOND-BESTPERFORMANCE FOR EACH INDIVIDUAL METRIC RESPECTIVELY.
<table><tr><td>Method</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>NIQEâ</td><td>LOEâ</td><td>DEâ</td><td>EMEâ</td></tr><tr><td>ZeroDCE++ [15]</td><td>0.82</td><td>21.68</td><td>0.17</td><td>3.83</td><td>13.31</td><td>1.75</td><td>10.78</td></tr><tr><td>ZeroDCE [16]</td><td>0.85</td><td>26.47</td><td>0.12</td><td>3.75</td><td>42.55</td><td>1.59</td><td>10.93</td></tr><tr><td>SCI [17]</td><td>0.81</td><td>20.84</td><td>0.18</td><td>4.42</td><td>17.91</td><td>1.96</td><td>13.49</td></tr><tr><td>RUAS [18]</td><td>0.81</td><td>20.44</td><td>0.14</td><td>3.58</td><td>11.06</td><td>1.50</td><td>14.34</td></tr><tr><td>EnlightenGAN [19]</td><td>0.81</td><td>18.20</td><td>0.30</td><td>3.67</td><td>82.64</td><td>1.51</td><td>6.17</td></tr><tr><td>UTV-NET [22]</td><td>0.79</td><td>15.22</td><td>0.29</td><td>3.44</td><td>21.71</td><td>0.97</td><td>6.99</td></tr><tr><td>ChebyLighter [23]</td><td>0.72</td><td>12.02</td><td>0.26</td><td>3.51</td><td>17.75</td><td>1.68</td><td>5.62</td></tr><tr><td>PairLIE [25]</td><td>0.75</td><td>15.74</td><td>0.34</td><td>6.49</td><td>72.52</td><td>1.72</td><td>7.28</td></tr><tr><td>NoiSER [26]</td><td>0.65</td><td>13.25</td><td>0.76</td><td>6.91</td><td>137.2</td><td>2.51</td><td>8.37</td></tr><tr><td>ZeroIG [28]</td><td>0.55</td><td>11.35</td><td>0.45</td><td>5.10</td><td>30.98</td><td>2.62</td><td>12.84</td></tr><tr><td>LL-GaussianImage</td><td>0.86</td><td>21.71</td><td>0.17</td><td>4.66</td><td>12.71</td><td>1.52</td><td>13.52</td></tr></table>

TABLE II

PERFORMANCE COMPARISON ON THE LSRW (HUAWEI) AND LSRW (NIKON) DATASETS. RED AND BLUE INDICATE THE BEST AND SECOND-BEST RESULTS FOR EACH METRIC RESPECTIVELY.
<table><tr><td>Method</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>NIQEâ</td><td>LOEâ</td><td>DEâ</td><td>EMEâ</td></tr><tr><td>ZeroDCE++ [15]</td><td>0.74</td><td>11.67</td><td>0.15</td><td>3.27</td><td>48.84</td><td>1.82</td><td>11.52</td></tr><tr><td>ZeroDCE [16]</td><td>0.70</td><td>11.01</td><td>0.17</td><td>3.23</td><td>30.29</td><td>1.57</td><td>10.77</td></tr><tr><td>SCI [17]</td><td>0.67</td><td>10.96</td><td>0.14</td><td>3.48</td><td>12.89</td><td>1.73</td><td>13.58</td></tr><tr><td>RUAS [18]</td><td>0.48</td><td>8.32</td><td>0.21</td><td>3.83</td><td>10.29</td><td>1.45</td><td>15.51</td></tr><tr><td>EnlightenGAN [19]</td><td>0.82</td><td>15.56</td><td>0.13</td><td>3.21</td><td>55.98</td><td>2.02</td><td>5.94</td></tr><tr><td>UTV-NET [22]</td><td>0.64</td><td>9.66</td><td>0.17</td><td>3.65</td><td>15.83</td><td>1.27</td><td>5.93</td></tr><tr><td>ChebyLighter [23]</td><td>0.87</td><td>18.43</td><td>0.09</td><td>3.39</td><td>16.16</td><td>1.82</td><td>5.09</td></tr><tr><td>PairLIE [25]</td><td>0.82</td><td>14.09</td><td>0.14</td><td>4.03</td><td>68.58</td><td>1.57</td><td>5.87</td></tr><tr><td>NoiSER [26]</td><td>0.77</td><td>17.6</td><td>0.36</td><td>4.71</td><td>31.87</td><td>2.11</td><td>4.38</td></tr><tr><td>ZeroIG [28]</td><td>0.76</td><td>19.12</td><td>0.15</td><td>3.5</td><td>20.48</td><td>2.41</td><td>11.38</td></tr><tr><td>LL-GaussianImage</td><td>0.88</td><td>18.65</td><td>0.12</td><td>3.59</td><td>18.72</td><td>1.84</td><td>12.08</td></tr></table>

to sample the weight map:

$$
w _ { i } = S o f t m a x ( S ( M _ { l o w } , \mu _ { i } ) ) ,\tag{13}
$$

where the ???????? ??????(â) operation is applied across the channel dimension to ensure that $\begin{array} { r } { \sum _ { k = 1 } ^ { K } w _ { i , k } = 1 } \end{array}$ . As illustrated in Fig. 3, the convexity of color attributes is guaranteed through this process. Concurrently, a differentiable connection between the image-space semantic context and the discrete Gaussian primitives is established via our training strategy, whereby the enhancement strategy for each Gaussian point is enabled to perceive its surrounding semantic environment.

Residual MLP-Based Mixture-of-Experts Color Transformation. A robust and flexible color mapping space is constructed through the learning of global color transformation operators instead of the direct regression of enhanced color values. These operators constitute a Mixture-of-Experts system where each expert is parameterized by a lightweight MLP. The k-th color operator is defined as $\Phi _ { k } \colon \mathfrak { R } ^ { 3 }  \mathfrak { R } ^ { 3 }$ . Each $\Phi _ { k }$ consists of three fully connected layers and ReLU activation functions. The adoption of a residual learning mechanism ensures stability during the initial training phase and preserves the reversibility of the enhancement process. The output $c _ { k } ^ { \prime }$ of the k-th operator for an input color $c _ { i n }$ is defined as follows:

$$
\Phi _ { k } ( c _ { i n } ) = C l a m p _ { [ 0 , 1 ] } \big ( c _ { i n } + \eta _ { k } ( c _ { i n } ) \big ) ,\tag{14}
$$

where $\eta _ { k }$ denotes the MLP network, whose final layer weights are initialized to zero to ensure that the initial transformation approximates an identity mapping; $c _ { i } ^ { e }$ represents the enhanced color for the i-th Gaussian primitive, which is generated via a weighted combination of the outputs from ?? â 1 learnable operators and one identity operator according to the semantic weights ???? :

$$
c _ { i } ^ { e } = w _ { i , i d } \cdot c _ { i } + \sum _ { k = 1 } ^ { K - 1 } w _ { i , k } \cdot \Phi _ { k } ( c _ { i } ) .\tag{15}
$$

Fig. 4 illustrates the network capability to preserve original colors in specific regions through high-weight identity operators while applying complex nonlinear enhancement in other areas. Fine-grained pixel-level control is realized through this design.

Zero-shot Unsupervised Loss Function Design. A comprehensive unsupervised loss function $\ell _ { t o t a l }$ is formulated for the enhancement stage as paired reference images are unavailable. The rasterizer projects optimized Gaussian primitives back into the image space to facilitate loss calculation within the image domain. The total loss function is defined as follows:

$$
\begin{array} { r l } & { \ell _ { t o t a l } = \lambda _ { 1 } \ell _ { e x p } + \lambda _ { 2 } \ell _ { h u e } + \lambda _ { 3 } \ell _ { s p a } + \lambda _ { 4 } \ell _ { c o l } } \\ & { \qquad + \lambda _ { 5 } \ell _ { c o n } + \ell _ { r e g } , } \end{array}\tag{16}
$$

where $\ell _ { e x p }$ denotes the exposure loss function. Correcting the illumination level involves aligning the average image brightness with a target value $E _ { h }$ . This target value is empirically set to 0.7. $Y ( I _ { l o w } )$ represents the brightness component of the image:

$$
\ell _ { e x p } = \parallel E [ Y ( I _ { e } ) ] - E _ { h } \parallel _ { 2 } ^ { 2 } .\tag{17}
$$

<!-- image-->  
Fig. 8. Visual comparison of enhancement quality across different iteration counts. The red box indicates the ground truth (GT), while the others show results at 5K, 10K, 20K, 50K and 100K iterations. As optimization progresses, image details refine and brightness stabilizes. Performance saturates after 50K iterations, representing an optimal trade-off between visual quality and inference efficiency.

A hue preservation constraint $\ell _ { h u e }$ based on the HSV color space is proposed to prevent color deviation during the enhancement process. Differentiable transformation from RGB to HSV is explicitly calculated as an alternative to traditional cosine similarity. A saturation mask $S _ { o r i g }$ is incorporated to address the instability of hue computation within low-saturation regions:

$$
\zeta _ { h u e } = \frac { \sum _ { p } l ( S _ { o r i g , p } > \tau ) \cdot m i n ( | H _ { e , p } - H _ { o r i g , p } | , l - | H _ { e , p } - H _ { o r i g , p } | ) } { \sum _ { p } l ( S _ { o r i g , p } > \tau ) + \varepsilon } ,\tag{18}
$$

where ??(â) denotes the indicator function; ?? = 0.1 represents the saturation threshold; and the ??????(â , 1 â â) term is utilized to address the periodicity of the hue circle.

Severe color attenuation frequently occurs under low illumination conditions. Color richness loss $\ell _ { c o l }$ builds on opponent color space theory to restore vivid visual effects. Maximizing the statistical dispersion of the two opponent channels enhances perceptual vividness. The formulation is defined as follows:

$$
\ell _ { c o l } = - \sqrt { \wp _ { r g } ^ { 2 } + \wp _ { y b } ^ { 2 } + \varepsilon } ,\tag{19}
$$

where variables $\wp _ { r g }$ and $\wp _ { y b }$ represent the spatial standard deviations of the ???? and ???? channels; ?? provides numerical stability. This constraint enables the model to effectively stretch the color distribution during illumination enhancement, thereby preventing a hazy visual appearance.

A histogram contrast loss $\ell _ { c o n }$ is proposed for the improvement of the dynamic range. This loss term ensures that the standard deviation of the enhanced image $\sigma _ { e }$ remains at least ?? times the original standard deviation $\sigma _ { o r i g }$ . The scaling factor ?? is set to 1.05. The formulation is defined as follows:

$$
\ell _ { c o n } = E \big [ R E L U ( \gamma \cdot \sigma _ { o r i g } - \sigma _ { e } ) \big ] .\tag{20}
$$

The contrast loss penalizes only insufficient contrast to allow further enhancement where necessary. Spatial consistency loss $\ell _ { s p a }$ prevents the introduction of noise or artifacts by enforcing the gradient distribution of the enhanced image to remain consistent with the original image. ?????????? operators are utilized for the extraction of gradient maps ???? and the calculation of $\ell _ { 1 }$ loss. The formulation is defined as follows:

$$
\ell _ { s p a } = \| \nabla _ { x } I _ { e } - \nabla _ { x } I _ { l o w } \| _ { 1 } + \left\| \nabla _ { y } I _ { e } - V _ { y } I _ { l o w } \right\| _ { 1 } .\tag{21}
$$

<!-- image-->  
Fig. 9. Impact of the number of operators ?? on image enhancement performance. The results correspond to ?? â {3, 8, 16, 32} from left to right. Image details and color fidelity improve significantly as ?? increases and stabilize beyond ?? = 16.

Edge and texture details are effectively preserved. The regularization term $\ell _ { r e g }$ incorporates total variation loss $\ell _ { T V }$ and entropy loss $\ell _ { e n t }$ to constrain the smoothness and certainty of the mixing weight map ?? . This formulation prevents truncation penalties resulting from pixel value overflow. The formulation is expressed as follows:

$$
\begin{array} { c } { \displaystyle \ell _ { r e g } = \lambda _ { 6 } \| \nabla { M } \| _ { 1 } + + \lambda _ { 7 } \sum _ { p } { \sum _ { k } } } \\ { \displaystyle - w _ { p , k } l o g ( w _ { p , k } + \varepsilon ) . } \end{array}\tag{22}
$$

The training process follows a self-supervised paradigm involving iterative optimization on a single input image without external datasets. Semantic-adaptive image enhancement is realized while the geometric structure of the original scene is faithfully maintained.

## â£. EXPERIMENTS

## A. Experimental Setup

Datasets. The LOL and LSRW datasets [53-54] facilitate the experimental evaluation of LL-GaussianImage. The LOL dataset serves as the first paired collection for supervised lowlight illumination enhancement and integrates synthetic and real-world imagery. The LSRW dataset represents the first large-scale real-world paired repository. This collection consists of two distinct subsets captured using HUAWEI P40 Pro and NIKON D7500 equipment.

Implementation Details. The overall network architecture of LL-GaussianImage is implemented within the PyTorch framework and executed on a single NVIDIA RTX 3090 GPU for both training and inference. Parameter settings and training strategies for the two-stage process are detailed to ensure experimental reproducibility [55]. The methodology employs a two-stage optimization paradigm to perform zero-shot instancelevel optimization for each input image.

The first stage involves the initialization of $N = 7 0 , 0 0 0 \ 2 \mathrm { D }$ Gaussian points. Optimization is conducted via the Adam optimizer with an initial learning rate of 0.01. A StepLR scheduler manages the learning rate by applying a decay factor of 0.9 every 7,000 iterations. This reconstruction process spans 30,000 iterations in total. The loss function â follows the configuration defined in LIG [1].

<!-- image-->  
(A)

<!-- image-->

<!-- image-->  
(E)

(B)  
<!-- image-->

<!-- image-->  
(F)

(C)  
<!-- image-->  
(G)

<!-- image-->  
(0)

<!-- image-->

Fig. 10. Visual comparison of different loss functions on enhancement results. $( \mathrm { A } ) { - } ( \mathrm { G } )$ show the enhancement results when removing â??????, ââ????,â??????, â??????, $\ell _ { c o n } , \ell _ { T V }$ and $\ell _ { e n t }$ respectively. (H) displays the result produced by LL-GaussianImage.  
<!-- image-->  
Fig. 11. Visualization of different weights for â????. (A)â(E) illustrate the results for $\lambda _ { 6 }$ values of 0, 100, 200, 300 and 500 respectively. The bottom row displays magnified views corresponding to each value to highlight local structural details.

The second stage freezes the Gaussian geometric parameters $\mu _ { i }$ and $\Sigma _ { i }$ from the initial stage and adjusts appearance color only by optimizing the unified enhancement framework to adjust appearance color. The first seven layers of a pre-trained MobileNetV2 are utilized as the feature extraction backbone. Feature dimensions are mapped to ?? = 32 . Illumination enhancement involves ?? = 16 MLP-based residual color operators. This stage executes 50,000 iterations using the Adam optimizer. An initial learning rate of 0.002 decays to 1% of the starting value through a cosine annealing strategy. Loss function weight coefficients are defined as $\lambda _ { 1 } =$ 50 , $\lambda _ { 2 } = 2 5$ $\lambda _ { 3 } = 3 2 . 5$ , $\lambda _ { 4 } = 2 . 9 1$ , $\lambda _ { 5 } = 0 . 4 4$ , $\lambda _ { 6 } = 3 0 0$ , $\lambda _ { 7 } = 0 . 0 1$ . Specific adjustments depend on the illumination levels of the low-light images. 2D Gaussian rendering is performed via a tile-based rasterizer with a tile size of $1 6 \times 1 6$ Reflection padding is applied to the image boundaries during feature extraction and gradient computation to eliminate edge artifacts.

Evaluation Metrics. Three widely recognized full-reference image metrics and four non-reference image metrics are selected within the field of image restoration and enhancement. Consistency between enhanced images and reference images for datasets with paired ground truth is measured through three full-reference metrics. These consist of Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM) and Learned Perceptual Image Patch Similarity (LPIPS). PSNR quantifies pixel-level fidelity. SSIM evaluates image similarity based on luminance, contrast, and structure. LPIPS measures the distance between enhanced and reference images within the feature space of a VGG network.

Four non-reference image metrics are employed to evaluate image naturalness, contrast and information content, particularly for performance in unpaired data or real-world scenarios. These metrics consist of Natural Image Quality Evaluator (NIQE), Lightness Order Error (LOE), Discrete Entropy (DE) and Enhancement Measure Evaluation (EME). The degree of image naturalness is quantified by NIQE through the measurement of the distance between enhanced images and a statistical model of natural images. LOE evaluates naturalness preservation during the illumination enhancement process. The richness of information contained within the images is measured by DE. EME is defined as a non-reference metric based on Weberâs law to quantify local contrast variations.

## B. Performance Comparison

Ten SOTA unsupervised methods within the LLIE domain are selected for comparison with LL-GaussianImage to ensure fairness [15-19, 22-23, 25-26, 28]. Evaluation results for the LOL and LSRW datasets are reported in TABLE I and TABLE II [53-54].

Visual comparison results for images selected from the LOL, LSRW (HUAWEI) and LSRW (NIKON) datasets are illustrated in Fig. 5. LL-GaussianImage achieves significant illumination enhancement while avoiding the underexposure observed in RUAS and UTV-Net or the overexposure found in ZeroIG. Regarding clarity, SCI, LL-GaussianImage, and ChebyLighter demonstrate superior visual quality. Conversely, EnlightenGAN produces haze-like artifacts. In terms of color reproduction, LL-GaussianImage exhibits slight oversaturation but remains free from the severe color bias present in NoiSER and EnlightenGAN. These visual results establish LL-GaussianImage as a robust framework for illumination enhancement within the compressed image domain.

The advantages of LL-GaussianImage regarding detail restoration are further validated through complex scenes from the LOL and LSRW datasets illustrated in Fig. 6 and Fig. 7. Leaf texture preservation by LL-GaussianImage is slightly inferior to pixel-domain low-light enhancement methods in Fig. 6. However, LL-GaussianImage maintains high similarity to ground-truth references in color saturation and contrast. Outstanding detail restoration is also exhibited by LL-GaussianImage in Fig. 7. Such performance is noteworthy because the data volume of compressed images is dozens of times smaller than that of pixel-domain images.

The evaluation framework incorporates full-reference metrics including PSNR, SSIM, and LPIPS, along with noreference metrics such as NIQE, LOE, DE, and EME. TABLE I and TABLE II summarize the quantitative performance across four datasets. LL-GaussianImage consistently ranks among the top two performers in terms of PSNR and SSIM. Superior performance is also demonstrated by the proposed method across the rankings of other evaluation metrics.

LL-GaussianImage balances reconstruction quality and compression performance despite certain limitations in detail preservation for specific scenes. This framework achieves a maximum compression ratio of 50x and represents the inaugural solution for low-light illumination enhancement within the 2DGS compressed domain. Consequently, this approach maintains a competitive advantage over various SOTA methods through its unique integration of efficiency and efficacy.

TABLE III  
PERFORMANCE EVALUATION OF LL-GAUSSIANIMAGE UNDER DIFFERENT ITERATION COUNTS. RED AND BLUE INDICATE THE BEST AND SECOND-BEST RESULTS FOR EACH METRIC RESPECTIVELY.
<table><tr><td>Enhance Iterations</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>NIQEâ</td><td>LOEâ</td><td>DEâ</td><td>EMEâ</td></tr><tr><td>5K</td><td>0.74</td><td>17.58</td><td>0.24</td><td>3.61</td><td>45.03</td><td>0.11</td><td>8.07</td></tr><tr><td>10K</td><td>0.73</td><td>17.55</td><td>0.24</td><td>3.53</td><td>56.85</td><td>0.12</td><td>7.27</td></tr><tr><td>20K</td><td>0.73</td><td>18.48</td><td>0.23</td><td>3.48</td><td>78.66</td><td>0.1</td><td>7.24</td></tr><tr><td>50K</td><td>0.76</td><td>18.56</td><td>0.21</td><td>3.44</td><td>68.41</td><td>0.25</td><td>7.26</td></tr><tr><td>100K</td><td>0.75</td><td>18.52</td><td>0.24</td><td>3.46</td><td>66.62</td><td>0.16</td><td>7.25</td></tr></table>

TABLE IV

PERFORMANCE EVALUATION OF LL-GAUSSIANIMAGE UNDER DIFFERENT ITERATION COUNTS. RED AND BLUE INDICATE THE BEST AND SECOND-BEST RESULTS FOR EACH METRIC RESPECTIVELY.
<table><tr><td>K</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>NIQEâ</td><td>LOEâ</td><td>DEâ</td><td>EMEâ</td></tr><tr><td>3</td><td>0.71</td><td>16.4</td><td>0.25</td><td>3.48</td><td>74.96</td><td>0.14</td><td>7.14</td></tr><tr><td>8</td><td>0.73</td><td>18.42</td><td>0.29</td><td>3.49</td><td>82.23</td><td>0.16</td><td>7.12</td></tr><tr><td>16</td><td>0.76</td><td>18.56</td><td>0.21</td><td>3.44</td><td>68.41</td><td>0.25</td><td>7.26</td></tr><tr><td>32</td><td>0.74</td><td>17.43</td><td>0.27</td><td>3.45</td><td>64.33</td><td>0.14</td><td>7.03</td></tr></table>

TABLE V

EVALUATION RESULTS FOR DIFFERENT WEIGHT VALUES OF $\ell _ { T V }$ . RED AND BLUE INDICATE THE BEST AND SECOND-BEST RESULTS FOR EACH METRIC RESPECTIVELY.
<table><tr><td> $\lambda _ { 6 }$ </td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>NIQEâ</td></tr><tr><td>0</td><td>0.49</td><td>11.76</td><td>0.42</td><td>4.00</td></tr><tr><td>100</td><td>0.74</td><td>18.46</td><td>0.23</td><td>3.48</td></tr><tr><td>200</td><td>0.70</td><td>17.68</td><td>0.25</td><td>3.45</td></tr><tr><td>300</td><td>0.75</td><td>18.60</td><td>0.21</td><td>3.58</td></tr><tr><td>500</td><td>0.76</td><td>18.63</td><td>0.20</td><td>3.46</td></tr></table>

## C. Ablation Study

A series of extensive ablation experiments was conducted on the LOL dataset to validate the effectiveness of individual components and the rationality of hyperparameter settings within the LL-GaussianImage framework. The baseline setup incorporates 16 color operators and 30,000 training iterations while utilizing the full set of proposed loss functions.

Impact of Enhancement Iterations. Optimal illumination enhancement parameters are fitted through iterative optimization on individual images due to the zero-shot learning strategy. Convergence and visual quality are directly determined by the iteration count. Model performance was evaluated across settings of 5,000, 10,000, 20,000, 50,000 and 100,000 iterations. Fig. 8 illustrates that brightness recovery and detail clarity improve as the iteration count increases, peaking at 50K before exhibiting a slight decline. Consequently, 50K iterations provide optimal brightness and color fidelity. Furthermore, quantitative results in TABLE III demonstrate that the 50K configuration yields the superior performance across most metrics, with the exceptions of LOE and EME. Based on these observations, the second stage adopts 50,000 iterations as the standard setting.

Effectiveness of Operator Quantity. The final enhanced image is synthesized through the blending of outputs from a

Mixture-of-Experts system using spatially adaptive weights. Fitting capacity for complex illumination mappings is determined by the operator quantity ??. Model performance is evaluated across ?? â {3, 8, 16, 32} to identify the optimal configuration. Image details and colors are improved as ?? increases according to Fig. 9. These attributes stabilize after ?? = 16 while occasionally exhibiting signs of oversaturation. As shown in TABLE IV, the configuration with ?? = 16 achieves the best results across all metrics except for LOE. Consequently, ?? = 16 is selected as the optimal value for the proposed framework.

Importance of Loss Functions. Multiple components for exposure, color, structure and regularization are incorporated into the loss function. Systematic removal of each term demonstrates its specific contribution to the enhancement process. Visualized results are summarized in Fig. 10. Each loss component is critical for generating high-quality images. Omitting the exposure loss $\ell _ { e x p }$ prevents effective illumination recovery. The removal of the hue preservation loss $\ell _ { h u e }$ , spatial consistency loss $\ell _ { s p a }$ and contrast loss $\ell _ { c o n }$ leads to significant color shifts and poor contrast stability. Furthermore, the absence of the total variation loss $\ell _ { T V }$ hinders smooth color transitions between Gaussian primitives. Therefore, all loss functions are consequently utilized in this framework.

Analysis of Specific Regularization Terms. Sensitivity to the weight hyperparameter of $\ell _ { T V }$ is specifically examined in addition to the general losses mentioned previously. $\ell _ { T V }$ constrains spatial smoothness of the mixture weight maps extracted by MobileNet. Color transitions of the ellipsoids become more uniform as the value of $\lambda _ { 6 }$ increases and reach stability at 300 as depicted in Fig. 11.

Magnified views indicate that smaller values of $\lambda _ { 6 }$ lead to disorganized primitive distributions and local illumination artifacts. Consequently, $\lambda _ { 6 } = 5 0 0$ serves as the final weight for $\ell _ { T V }$ . Optimal performance is achieved with $\lambda _ { 6 } = 5 0 0$ across all metrics except NIQE as shown in TABLE V.

## D. Limitation and Future works

Significant potential and generalizability are exhibited by LL-GaussianImage. However, optimization for rapid inference alongside image quality preservation is necessitated by the advancement of feed-forward Gaussian Splatting. Furthermore, the optimization of Gaussian ellipsoids for detailed texture reconstruction remains an unresolved challenge. LL-GaussianImage is expected to facilitate advancements in lowlevel vision applications within the 2DGS compressed image domain.

## V. CONCLUSION

LL-GaussianImage is formulated as the first zero-shot unsupervised framework for low-light illumination enhancement performed directly within the 2DGS compressed domain. The introduction of a semantic-guided Mixture-of-Experts module and a decoupling strategy for geometry and appearance transforms the enhancement task into manifold mapping within the attribute space. This framework achieves high-fidelity compression-as-enhancement while maintaining a frozen geometric structure. Such a paradigm avoids the computational redundancy and secondary distortion inherent in the traditional decompression-enhancement-recompression workflow. To address speed bottlenecks in iterative optimization and preserve fine-grained features, future work will focus on developing feed-forward Gaussian Splatting architectures, thereby establishing novel pathways for low-level vision tasks within the 2DGS compressed domain.

## REFERENCES

[1] L. Zhu, G. Lin, J. Chen, et al., "Large images are Gaussians: High-quality large image representation with levels of 2D Gaussian splatting," in Proc. AAAI Conf. Artif. Intell. (AAAI), 2025, pp. 10977-10985.

[2] Z. Zeng, Y. Wang, C. Yang, T. Guan, and L. Ju, "Instant GaussianImage: A generalizable and self-adaptive image representation via 2D Gaussian splatting," 2025, arXiv:2506.23479.

[3] X. Zhang, X. Ge, T. Xu, et al., "Gaussianimage: 1000 fps image representation and compression by 2D Gaussian splatting," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 327-345.

[4] C. Jiang, Z. Li, H. Zhao, Q. Shan, S. Wu, and J. Su, "Beyond pixels: Efficient dataset distillation via sparse Gaussian representation," 2025, arXiv:2509.26219.

[5] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, and G. Drettakis, "3D Gaussian splatting for real-time radiance field rendering," ACM Trans. Graph., vol. 42, no. 4, pp. 1-14, 2023.

[6] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, "NeRF: Representing scenes as neural radiance fields for view synthesis," Commun. ACM, vol. 65, no. 1, pp. 99-106, 2021.

[7] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, "2D Gaussian splatting for geometrically accurate radiance fields," in ACM SIGGRAPH 2024 Conf. Papers, 2024, pp. 1-11.

[8] V. Sitzmann, J. Martel, A. Bergman, D. Lindell, and G. Wetzstein,

"Implicit neural representations with periodic activation functions," Adv. Neural Inf. Process. Syst., vol. 33, pp. 7462-7473, 2020.

[9] K. O. Stanley, "Compositional pattern producing networks: A novel abstraction of development," Genet. Program. Evolvable Mach., vol. 8, no. 2, pp. 131-162, 2007.

[10] C. Li, C. Guo, L. Han, et al., "Low-light image and video enhancement using deep learning: A survey," IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 12, pp. 9396-9416, 2022.

[11] Q. Zhao, G. Li, B. He, et al., "Deep learning for low-light vision: A comprehensive survey," IEEE Trans. Neural Netw. Learn. Syst., 2025.

[12] M. T. Islam, I. Alam, S. S. Woo, S. Anwar, I. K. H. Lee, and K. Muhammad, "Loli-street: Benchmarking low-light image enhancement and beyond," in Proc. Asian Conf. Comput. Vis. (ACCV), 2024, pp. 1250- 1267.

[13] G. Bai, H. Yan, W. Liu, et al., "Towards lightest low-light image enhancement architecture for mobile devices," Expert Syst. Appl., vol. 296, p. 129125, 2026.

[14] X. Guo, Y. Li, and H. Ling, "LIME: Low-light image enhancement via illumination map estimation," IEEE Trans. Image Process., vol. 26, no. 2, pp. 982-993, 2017.

[15] C. Li, C. Guo, and C. C. Loy, "Learning to enhance low-light image via zero-reference deep curve estimation," IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 8, pp. 4225-4238, 2022.

[16] C. Guo, C. Li, J. Guo, et al., "Zero-reference deep curve estimation for low-light image enhancement," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 1780-1789.

[17] L. Ma, T. Ma, R. Liu, et al., "Toward fast, flexible, and robust low-light image enhancement," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 5637-5646.

[18] R. Liu, L. Ma, J. Zhang, et al., "Retinex-inspired unrolling with cooperative prior architecture search for low-light image enhancement," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 10561-10570.

[19] Y. Jiang, et al., "EnlightenGAN: Deep light enhancement without paired supervision," IEEE Trans. Image Process., vol. 30, pp. 2340-2349, 2021.

[20] Y. Chen, G. Zhu, X. Wang, et al., "FMR-Net: A fast multi-scale residual network for low-light image enhancement," Multimedia Syst., vol. 30, p. 73, 2024.

[21] Y. Chen, G. Zhu, X. Wang, et al., "FRR-NET: A fast reparameterized residual network for low-light image enhancement," Signal, Image Video Process., vol. 18, pp. 4925â4934, 2024.

[22] C. Zheng, D. Shi, and W. Shi, "Adaptive unfolding total variation network for low-light image enhancement," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 4439-4448.

[23] J. Pan, D. Zhai, Y. Bai, et al., "ChebyLighter: Optimal curve estimation for low-light image enhancement," in Proc. 30th ACM Int. Conf. Multimedia (ACM MM), 2022, pp. 1358â1366.

[24] C. Liu, F. Wu, and X. Wang, "EFINet: Restoration for low-light images via enhancement-fusion iterative network," IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 12, pp. 8486-8499, 2022.

[25] Z. Fu, Y. Yang, X. Tu, et al., "Learning a simple low-light image enhancer from paired low-light instances," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 22252-22261.

[26] Z. Zhang, et al., "Noise self-regression: A new learning paradigm to enhance low-light images without task-related data," IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 2, pp. 1073â1088, 2025.

[27] W. Wu, J. Weng, P. Zhang, et al., "URetinex-Net: Retinex-based deep unfolding network for low-light image enhancement," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 5901-5910.

[28] Y. Shi, D. Liu, L. Zhang, et al., "ZERO-IG: Zero-shot illumination-guided joint denoising and adaptive enhancement for low-light images," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 3015- 3024.

[29] Y. Omri, C. Ding, T. Weissman, and T. Tambe, "Vision-language alignment from compressed image representations using 2D Gaussian splatting," 2025, arXiv:2509.22615.

[30] J. Hu, B. Xia, B. Chen, W. Yang, and L. Zhang, "Gaussiansr: High fidelity 2D Gaussian splatting for arbitrary-scale image super-resolution," in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 39, no. 4, 2025, pp. 3554-3562.

[31] L. Peng, A. Wu, W. Li, et al., "Pixel to Gaussian: Ultra-fast continuous super-resolution with 2D Gaussian modeling," 2025, arXiv:2503.06617.

[32] Sun. H, Yu. F, Xu. H, Zhang. T, and Zou. C, âLL-Gaussian: Low-light scene reconstruction and enhancement via Gaussian splatting for novel view synthesis,â Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 4261-4270.

[33] H. Wang, J. Huang, L. Yang, T. Deng, G. Zhang, and M. Li, "LLGS: Unsupervised Gaussian splatting for image enhancement and reconstruction in pure dark environment," 2025, arXiv:2503.18640.

[34] H. Jiang, A. Luo, X. Liu, S. Han, and S. Liu, "Lightendiffusion: Unsupervised low-light image enhancement with latent-retinex diffusion models," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 161-179.

[35] Y. Lin, T. Ye, S. Chen, et al., "Aglldiff: Guiding diffusion models towards unsupervised training-free real-world low-light image enhancement," in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 39, no. 5, 2025, pp. 5307- 5315.

[36] Y. Huang, X. Liao, J. Liang, Q. Yan, B. Shi, and Y. Xu, "Zero-shot lowlight image enhancement via latent diffusion models," in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 39, no. 4, 2025, pp. 3815-3823.

[37] S. Yang, M. Ding, Y. Wu, Z. Li, and J. Zhang, "Implicit neural representation for cooperative low-light image enhancement," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 12918-12927.

[38] T. Chobola, Y. Liu, H. Zhang, J. A. Schnabel, and T. Peng, "Fast contextbased low-light image enhancement via neural implicit representations," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 413-430.

[39] Y. Yan, H. Lin, C. Zhou, et al., "Street Gaussians: Modeling dynamic urban scenes with Gaussian splatting," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 156-173.

[40] A. Hanson, A. Tu, G. Lin, V. Singla, M. Zwicker, and T. Goldstein, "Speedy-splat: Fast 3D Gaussian splatting with sparse pixels and sparse primitives," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 21537-21546.

[41] T. Liu, G. Wang, S. Hu, et al., "Mvsgaussian: Fast generalizable Gaussian splatting reconstruction from multi-view stereo," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 37-53.

[42] Y. Xu, J. Zhang, Y. Chen, D. Wang, L. Yu, and C. He, "PMGS: Reconstruction of projectile motion across large spatiotemporal spans via 3D Gaussian splatting," 2025, arXiv:2508.02660.

[43] Y. Xu, J. Zhang, H. Liu, et al., "PEGS: Physics-event enhanced large spatiotemporal motion reconstruction via 3D Gaussian splatting," 2025, arXiv:2511.17116.

[44] Y. Chen, C. Gu, J. Jiang, X. Zhu, and L. Zhang, "Periodic vibration Gaussian: Dynamic urban scene reconstruction and real-time rendering," 2023, arXiv:2311.18561.

[45] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M. H. Yang, "Drivinggaussian: Composite Gaussian splatting for surrounding dynamic autonomous driving scenes," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 21634-21643.

[46] K. Cheng, X. Long, K. Yang, et al., "Gaussianpro: 3D Gaussian splatting with progressive propagation," in Proc. 41st Int. Conf. Mach. Learn. (ICML), 2024.

[47] Y. Liu, C. Luo, L. Fan, N. Wang, J. Peng, and Z. Zhang, "Citygaussian: Real-time high-quality large-scale scene rendering with Gaussians," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 265-282.

[48] J. Fan, W. Li, Y. Han, T. Dai, and Y. Tang, "Momentum-GS: Momentum Gaussian self-distillation for high-quality large scene reconstruction," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 25250- 25260.

[49] G. Zhao, C. Ni, X. Wang, et al., "Drivedreamer4D: World models are effective data machines for 4D driving scene representation," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 12015-12026.

[50] C. Ni, G. Zhao, X. Wang, et al., "Recondreamer: Crafting world models for driving scene reconstruction via online restoration," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 1559- 1569.

[51] T. Yi, J. Fang, J. Wang, et al., "Gaussiandreamer: Fast generation from text to 3D Gaussians by bridging 2D and 3D diffusion models," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 6796- 6807.

[52] Y. Chen, Y. Shi, G. Li, et al., "A lightweight real-time low-light enhancement network for embedded automotive vision systems," 2025, arXiv:2512.02965.

[53] C. Wei, W. Wang, W. Yang, and J. Liu, "Deep Retinex decomposition for low-light enhancement," 2018, arXiv:1808.04560.

[54] J. Hai, Z. Xuan, R. Yang, et al., "R2rnet: Low-light image enhancement via real-low to real-normal network," J. Vis. Commun. Image Represent., vol. 90, p. 103712, 2023.

[55] A. Paszke, S. Gross, F. Massa, et al., "PyTorch: An imperative style, highperformance deep learning library," Adv. Neural Inf. Process. Syst., vol. 32, pp. 8024-8035, 2019.

<!-- image-->

Yuhan Chen received his master's degree in 2024 from the College of Mechanical Engineering at Chongqing University of Technology. He is currently pursuing the Ph.D. degree in College of Mechanical and Vehicle Engineering at Chongqing University, China. His research interests include deep learning, Low-level Vision tting.

and Gaussian Splatting.

<!-- image-->

Wenxuan Yu received the B.E. degree majoring in Mechanical Design, Manufacturing, and Automation at Chongqing University in 2025. He is currently pursuing the M.E. degree in Mechanical Engineering at Chongqing University, Chongqing, China. His research interests include computer vision,

Gaussian Splatting and deep learning.

<!-- image-->

Guofa Li received the Ph.D. degree in Mechanical Engineering from Tsinghua University, China, in 2016. He is currently a Professor with Chongqing University, China. His research interests include environment perception, driver behavior analysis, and smart decision-making based on artificial intelligence technologies in autonomous vehicles and intelligent

transportation systems. He serves as the Associate Editor for IEEE Transactions on Intelligent Transportation Systems, IEEE Transactions on Affective Computing, and IEEE Sensors Journal.

<!-- image-->

Yijun Xu received the B.E. degree from Southwest University, Chongqing, China, in 2024. He is currently pursuing the M.S. degree in Electronic Information at the School of Electronic Information, Wuhan University, Wuhan, China. His research interests include machine vision and signal processing.

<!-- image-->

Ying Fang received the B.E. degree majoring in Vehicle Engineering at Chongqing University of Technology. He is currently pursuing the M.E. degree in Mechanical Engineering at Chongqing University, Chongqing, China. His research interests include computer vision, Gaussian Splatting and deep learning.

<!-- image-->

Yicui Shi received the B.E. degree majoring in Automotive Engineering at Chongqing University in 2025. He is currently pursuing the M.E. degree in Automotive Engineering at Chongqing University, Chongqing, China. His research interests include computer vision, Gaussian Splatting and deep learning.

<!-- image-->

Long Cao received his master's degree from the School of Mechanical Engineering, Guangxi University, in 2025. He is currently pursuing the Ph.D. degree at the College of Mechanical and Vehicle Engineering, Chongqing University. His research interests include computer graphics, inverse rendering, and deep learning.

<!-- image-->

Wenbo Chu received his B.S. degree majored in Automotive Engineering from Tsinghua University, China, in 2008, and his M.S. degree majored in Automotive Engineering from RWTH-Aachen, German and Ph.D. degree majored in Mechanical Engineering from Tsinghua University, China, in 2014.

He is currently a research fellow at

Western China Science City Innovation Center of Intelligent and Connected Vehicles (Chongqing) Co, Ltd., and National Innovation Center of Intelligent and Connected Vehicles.

<!-- image-->

Keqiang Li received the B.E. degree from Tsinghua University, Beijing, China, in 1985, and the M.E. and Ph.D. degrees from Chongqing University, Chongqing, China, in 1988 and 1995, respectively.

He is currently a Professor with the School of Vehicle and Mobility, Tsinghua University. He is the Chief Scientist of Intelligent and Connected Vehicle

Innovation Center of China, and the Director of State Key Laboratory of Automotive Safety and Energy of China. His current research interests include intelligent connected vehicles, cloud-based control for vehicles, and vehicle dynamics systems.