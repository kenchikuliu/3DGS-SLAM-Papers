<!-- page 1 -->
A 3DGS-Diffusion Self-Supervised Framework for
Normal Estimation from a Single Image
Yanxing Liang
School of Artificial Intelligence and Computer Science
Jiangnan University
Wuxi, China
ares_liang@stu.jiangnan.edu.cn
Yinghui Wang *
School of Artificial Intelligence and Computer Science
Jiangnan University
Wuxi, China
wangyh@jiangnan.edu.cn
Jinlong Yang
School of Artificial Intelligence and Computer Science
Jiangnan University
Wuxi, China
yjlgedeng@163.com
Wei Li
School of Artificial Intelligence and Computer Science
Jiangnan University
Wuxi, China
cs_weili@jiangnan.edu.cn
Abstract
The lack of spatial dimensional information remains a core challenge in normal
estimation from a single image. Although recent diffusion-based methods have
demonstrated significant potential in 2D-to-3D implicit mapping, they rely on
data-driven statistical priors and miss the explicit modeling of light-surface inter-
action, leading to multi-view normal direction conflicts. Moreover, the discrete
sampling mechanism of diffusion models causes gradient discontinuity in differ-
entiable rendering reconstruction modules, preventing 3D geometric errors from
being backpropagated to the normal generation network, thereby forcing existing
methods to depend on dense normal annotations. This paper proposes SINGAD,
a novel Self-supervised framework from a single Image for Normal estimation
via 3D GAussian splatting guided Diffusion. By integrating physics-driven light-
interaction modeling and a differentiable rendering-based reprojection strategy, our
framework directly converts 3D geometric errors into normal optimization signals,
solving the challenges of multi-view geometric inconsistency and data dependency.
Specifically, the framework constructs a light-interaction-driven 3D Gaussian Splat-
ting (3DGS) reparameterization model to generate multi-scale geometric features
consistent with light transport principles, ensuring multi-view normal consistency.
Additionally, a cross-domain feature fusion module is designed within a conditional
diffusion model, embedding geometric priors to constrain normal generation while
Preprint. Under review.
arXiv:2508.05950v1  [cs.CV]  8 Aug 2025

<!-- page 2 -->
maintaining accurate geometric error propagation. Furthermore, a differentiable
3D reprojection loss strategy is introduced for self-supervised optimization that
minimizes geometric error between the reconstructed and input image, eliminat-
ing dependence on annotated normal datasets. Quantitative evaluations on the
Google Scanned Objects dataset demonstrate that our method outperforms state-
of-the-art approaches across multiple metrics. This work provides a pioneering
self-supervised paradigm for normal estimation from a single image, advancing
the process from data-driven learning to physics-aware modeling.
1
Introduction
3D surface normal estimation from a single image, as a fundamental task in 3D scene understanding
and reconstruction [1, 2], aims to infer geometric structures of object surfaces. The persistent
challenge of this task directly correlates with inherent spatial information loss, a critical limitation
overshadowing existing computer vision advancements.
Early studies [3–11] mainly use Convolutional Neural Networks (CNNs) or Vision Transformers to
extract multi-scale features with geometric consistency constraints. However, these non-generative
approaches could only estimate viewpoint-specific normals. The growing prominence of diffusion
models [12–15] with implicit prior modeling and progressive denoising mechanisms has successfully
achieved multi-view normal generation from a single image. Nevertheless, as diffusion models
operate as data-driven 2D image generators lacking explicit geometric representation, they often
estimate normals with shape ambiguity in concave/convex regions across views.
Recent state-of-the-art methods [2, 16–18] attempt to integrate diffusion models with 3D reconstruc-
tion techniques through multi-view joint probability distributions or Score Distillation Sampling
(SDS) strategies. Others [19–24] jointly learn depth-normal correlations by computing gradient
differences from depth maps. While these methods use Neural Radiance Fields (NeRF) [25] or 3D
Gaussian Splatting (3DGS) [26] for final normal rendering, the discrete sampling mechanism in
diffusion models creates discontinuous gradients. This problem prevents effective backpropagation
of 3D geometric errors through differentiable rendering modules, isolating 3D reconstruction to a
post-processing role rather than participating in normal optimization.
Zamir et al. [1] established that surface normals serve as critical intermediate representations for 3D
reconstruction, where estimation accuracy directly impacts geometric fidelity. This motivates us to
reexamine the existing approaches: Since the ultimate objective of normal estimation is to enable
precise 3D reconstruction, why not exploit the inherent multi-view geometric consistency of 3D
models? We could establish closed-loop feedback through "normal generation to reprojection error
optimization" with 3D reprojection loss backpropagation, rather than relying on multi-view fusion or
vision-language fusion. Notably, the rapid growth of "image-3D model" datasets [2, 16] now exceeds
"image-normal" datasets in scale, providing practical feasibility for our approach.
Building on these insights, this paper proposes SINGAD (see Figure 1), a novel Self-supervised
framework from a single Image for Normal estimation via 3D GAussian splatting guided Diffusion.
Our method integrates the differentiable rendering properties of 3DGS with the prior constraints
of conditional diffusion models, establishing an end-to-end self-supervised learning framework via
three core components: 1) A light-interaction-driven 3DGS parameter estimation model via an
MLP-based network by the light-interaction modeling with the Gabor kernel, ensuring compliance
with the optical reflection principle and diffusion model discretization. Its output features, multi-scale
geometric descriptors and preliminary normals, are generated through a Feature Pyramid Network
(FPN) augmented with Principal Component Analysis (PCA), functioning as geometric priors (see
Section 3.1). 2) A cross-domain feature-guided conditional diffusion model incorporates a novel
feature fusion layer that injects these priors into the denoising process, aligning geometric and RGB
domains to refine local normals while preserving multi-view consistency (see Section 3.2). 3)
A normal reprojection optimization strategy reconstructs a 3D model from predicted normals
via rasterization, computes reconstruction errors through a joint reprojection loss, and performs
self-supervised backpropagation to optimize the 3DGS parameter estimation and diffusion modules
(see Section 3.3).
Our contributions are summarized below:
2

<!-- page 3 -->
Figure 1: SINGAD network pipeline. The input image is processed by an MLP to estimate geometric
attributes (Geo Attr.) and texture attributes (Tex Attr.) for 3DGS. Multi-scale geometric features
are extracted via a Feature Pyramid Network (FPN) and fed into the 3DGS module to generate
preliminary geometric descriptors. A conditional diffusion model, guided by a cross-domain feature
fusion layer (incorporating attention-based weighting and gated fusion), iteratively denoises the
input by combining geometric and texture features. The differentiable rendering pipeline and 3D
reprojection loss jointly optimize network parameters. The predicted normal map is generated
through this closed-loop optimization. Arrow directions indicate data flow and dependencies between
modules.
• A light-interaction-driven 3DGS reparameterization method that explicitly links diffuse
irradiance with normal orientation quantized ellipsoid parameters through the Gabor kernel,
maintaining physical light transport principles, improving multi-scale geometric feature
accuracy.
• A cross-domain feature-guided diffusion model that injects multi-scale geometric features
to constrain geometry-aware normal generation while preserving gradient propagation.
• A annotation-free self-supervised strategy using differentiable rasterization and 3D re-
projection losses to jointly optimize 3DGS and diffusion networks, achieving supervised
performance without normal annotations.
2
Related Works
CNNs and Transformers. Early CNN-based approaches [3–8] pioneered multi-scale feature fusion
for joint normal estimation and scene understanding. Wang et al. [3] developed a multi-task
framework integrating normal prediction with room layout estimation through hierarchical feature
aggregation. Wei et al. [8] introduced virtual normals by randomly sampling 3D point triplets to
construct virtual planes, enhancing robustness against noise. However, these discriminative models
suffered from limited generalization due to their heavy reliance on normal annotations.
The advent of Vision Transformers [9–11] brought improved long-range dependency modeling in
complex scenes. PlaneTR [10] combined line segment features with Transformer attention for
3

<!-- page 4 -->
simultaneous normal and 3D plane recovery, while Bae et al. [11] incorporated uncertainty-aware
learning objectives. Nevertheless, constrained by conventional dense prediction approaches, these
methods still require learning highly precise normal annotations to determine per-pixel viewing
directions for providing essential geometric cues. This fundamental limitation restricts them to known
camera viewpoints, precluding multi-view normal estimation under unknown perspectives.
Diffusion Models. Diffusion models have revolutionized normal estimation through their prob-
abilistic generation framework. These approaches typically formulate normal estimation as an
inverse problem, generating multi-view normals through iterative denoising. Compared to traditional
CNN/Transformer models, diffusion models demonstrate superior capabilities in synthesizing novel
images while also enabling robust constraint enforcement during the inference phase through their it-
erative denoising mechanism. For instance, GeoNet [12, 13] established a depth-normal co-prediction
framework with 3D coordinate inversion constraints. ASN [14] developed latent-space bidirectional
mapping using geometric Jacobian matrices, and Wonder3D [15] achieved cross-domain generation
through orthogonal projection coordinate systems. OmniData [1] provided billion-scale normal
annotations via semi-automatic pipelines. However, these methods sacrifice geometric consistency
for novel view generalization, requiring increased training data and optimization iterations.
The integration of diffusion models with 3D reconstruction has become a research frontier. By
combining diffusion processes with explicit 3D geometric modeling, recent works attempt direct
generation of normal-equipped 3D meshes/point clouds from single images. Magic3D [18] and
RealFusion [21] combined neural radiance fields with probabilistic sampling, whereas SyncDreamer
[17] and Zero123 [19, 20] leveraged cross-view attention mechanisms. One-2-3-45 [22] and Metric3D
[23, 24] enhance surface continuity through geometry-aware fusion algorithms. Despite promising
results, two critical problems persist: 1) The normal-from-depth process suffers from gradient
discontinuity caused by diffusion models’ discrete sampling, leading to geometric detail loss. 2)
Depth maps’ inherent limitation in providing only 3D coordinates prevents unique tangent plane
determination of the pixel, causing multi-view normal inconsistency.
Normal estimation has evolved from CNN/Transformer architectures to diffusion models integrated
with 3D reconstruction, yet unresolved challenges remain in multi-view geometric consistency
and annotation dependency. Our method focuses on normal estimation through 3DGS explicit
representation, generating 3D models from predicted normals and enforcing geometric consistency
via 3D reprojection-based self-supervision. By comparing reprojected images with originals for
backward error propagation, we achieve multi-view normal estimation without requiring multi-view
images or normal annotations, establishing a fully self-supervised framework.
3
Method
3.1
3DGS for Geometry Feature Extraction
3DGS using discrete 3D Gaussian functions for modeling scenes [26]. For each pixel x in image
I(x), the corresponding Gaussian function is defined by its spatial mean µ (ellipsoid center) and
covariance matrix Σ (ellipsoid shape and scale):
I(x) =
X
Gs(x) =
X
e−1
2 (x−µ)⊤Σ−1(x−µ)
(1)
where Σ is decomposed into scaling matrix S and rotation matrix R through Σ = RSS⊤R⊤,
ensuring positive semi-definiteness during optimization.
Compared with NeRF’s implicit neural representation, 3DGS explicitly models both geometric
structure and color texture attributes, facilitating discrete quantization as geometric feature vectors for
conditional diffusion models. However, their linear independence limits 3DGS to forward rendering,
lacking backward gradient propagation for normal optimization. To address this problem, we propose
a reparameterization of 3DGS through light-interaction modeling, establishing explicit connections
between geometric and non-geometric attributes to enhance normal generation in the conditional
diffusion model.
Light-Interaction Model for 3DGS Reparameterization. Surfaces appear macroscopically flat but
exhibit complex height variations at microscopic scales. Surface normals represent these geometric
structures. To bridge the gap between micro-geometry structure features and physical constraints
in normal estimation, we decouple surface light fields into local reflection and global interference
4

<!-- page 5 -->
effects based on the Huygens-Fresnel principle [27]. This establishes a mathematical relationship
between height gradient ∇h(x) and outgoing radiance Lo(ωo), forming our light-interaction model:
I(x) = Lo(ωo) =
Z
Ω+ Li(ωi)R(ωi, ωo, ∇h(x))(ωi · n)dωi −LA(ωo)
(2)
where Ω+ denotes the normal-constrained hemisphere ensuring valid surface orientations, R repre-
sents the Bidirectional Reflectance Distribution Function (BRDF) [28], and LA(ωo) denotes invalid
normal directions.
Adopting Lambertian reflection [27], we simplify Equation 2 to a diffuse-dominated form:
LD = kD · n ·
Z
Ω+ Li(ωi)ωidωi
(3)
where kD denotes the learnable spatially-varying diffuse coefficient. This decouples normals from
the integral operator, establishing linear radiance-normal correlation.
To enable explicit surface gradient mapping, we reparameterize the integral term by using the Gabor
kernel:
Z
Ω+ Li(ωi)ωidωi = Ga(x; ξ, α) = G2D(x; ξ) · e−i·2π(α·x)
(4)
where the Gabor kernel combines a 2D Gaussian G2D(x; ξ) =
1
2πξ2 e−∥x∥2/(2ξ2) with a complex
exponential. Here, ξ follows scalar diffraction theory [27], and α = 2∇h(x)/λ encodes local height
gradients. This results in a differentiable radiance-normal relationship:
Lo(ωo) = kD · n · Ga
(5)
Our reparameterization explicitly quantifies optical processes in 3DGS, enabling surface gradient
propagation through Gabor kernel derivatives and simplifying mapping learning through MLPs [29]
instead of complex ViT architectures [30]. (Detailed in Appendix A.)
Ellipsoid Parameter Estimation. We design an MLP network to map a single image to 3D ellipsoid
parameters (µ, Σ, σ, kD). The network first extracts multi-scale features using a pretrained ResNet-
50 backbone [31], followed by global average pooling. These features pass through a 5-layer MLP
with ReLU activations and batch normalization, outputting ellipsoid parameters Gs(x) ∈RK×d.
Physical constraints are enforced through an energy-loss regularization term:
Lenergy = ∥kD · n · Ga∥min
(6)
This ensures compliance with the light-interaction model during optimization.
Multi-scale Feature Extraction and Preliminary Normal Estimation. We construct a Feature
Pyramid Network (FPN) [32] to encode ellipsoid parameters Gs into multi-scale geometric features
Fgeo = {F1, F2, F3} at 1/4, 1/8, and 1/16 resolutions. These features guide the conditional diffusion
model through cross-domain fusion.
Preliminary normals are generated via Principal Component Analysis (PCA) [33] on Fgeo:
Σk = VkKV⊤
k
(7)
n3DGS(p) =
K
X
k=1
wk(p) · vk,d
(8)
where vk,d denotes the eigenvector corresponding to the smallest eigenvalue, and weights wk(p) are
determined by Mahalanobis distance [34]. Differentiable rasterization produces preliminary normals
n3DGS ∈RH×W ×3, providing geometric priors for subsequent refinement.
3.2
Conditional Diffusion Model for Normal Generation
Diffusion Model [35], as the generative model that progressively adds and removes noise, has
demonstrated remarkable success in image generation tasks. The forward process gradually corrupts
target images with Gaussian noise through a Stochastic Differential Equation (SDE):
dxt = D(xt, t)dt + S(t)dW
(9)
5

<!-- page 6 -->
where xt denotes the diffusion state at timestep t, D(xt, t) represents the drift coefficient, S(t) is
the diffusion coefficient, and dW indicates standard Brownian motion. The drift term guides data
evolution while the diffusion term introduces stochasticity for generation diversity.
The reverse process iteratively denoises corrupted images through the reverse-time SDE:
dxt = D(xt, t)dt + S(t)dW t
(10)
where D(x, t) denotes the reversed denoising function derived from D(x, t), and W t represents
reversed Brownian motion.
Traditional diffusion models suffer from single-modal input limitations. Simply adding normal map
channels would interfere with pretrained weights and cause catastrophic forgetting. To address this
problem, we design a conditional SDE-based diffusion model by using the U-Net architecture, where
multi-scale geometric features from 3DGS-generated preliminary normals participate in multi-view
consistency constraints through cross-domain feature fusion layers.
Cross-Domain Feature Fusion. Images contain both geometric attributes (e.g., surface normals) and
non-geometric attributes (e.g., textures). We decompose the joint image distribution qimg in diffusion
models into geometric (qgeo) and non-geometric (qtex) components:
qimg(x) = qgeo(n) · qtex(x|n)
(11)
Instead of assuming complete linear independence between geometric and non-geometric attributes,
we model geometric properties as conditional probabilities for non-geometric attributes. This
formulation provides theoretical support for subsequent cross-domain feature fusion and conditional
injection.
To ensure consistency between the generated normal and the input RGB image, we design cross-
domain feature Ffuse fusion layers between the denoising and noising stages of the conditional
diffusion model through attention mechanisms:
Mc(Fimg) = σ (MLP(AvgPool(Ftex)) + MLP(MaxPool(Fgeo)))
(12)
Ms(Fimg) = σ (f7×7([AvgPool(Ftex), MaxPool(Fgeo)]))
(13)
Ffuse = Mc ⊙Ms ⊙Ftex ⊙Fgeo
(14)
where Mc generates channel weights through global pooling, Ms produces spatial weights via max-
pooling, and ⊙denotes element-wise multiplication. Fimg is the temporary feature in the middle. The
fused feature Ffuse concatenates geometric features Fgeo and texture features Ftex along the channel
dimension.
Conditional Injection and Gated Fusion We extend the original reverse SDE’s score function
Sθ(x, t) to a conditional version Sθ(x, t, I, Ffuse) that incorporates image content and geometric
priors. The modified reverse SDE becomes:
dxt = [D(xt, t) −S2
θ(xt, t, Ffuse)]dt + S(t)dW
(15)
The conditional vector Ffuse interacts with diffusion state xt and timestep t through residual connec-
tions and cross-domain attention mechanisms, generating condition-dependent noise estimates.
Final surface normals are obtained by gated fusion of diffusion-generated normals (ndiffusion) and
3DGS-derived normals (n3DGS). The gating function G : RH×W ×6 →[0, 1]H×W produces spatially
adaptive weights:
nfuse(x) = G · n3DGS(x, y) + (1 −G) · ndiffusion(x, y)
(16)
where the spatial-adaptive gate G learns to emphasize 3DGS normals in flat regions (G →1) and
diffusion outputs in detailed areas (G →0), achieving coherent normal map generation.
3.3
3D Reprojection-Based Loss for Optimization
To enable end-to-end self-supervised training of the normal generation network, we desion a dif-
ferentiable 3D reprojection error optimization strategy. This strategy introduces a differentiable
projection pipeline that parameterizes predicted normals into 3D geometric representations, computes
pixel-level alignment errors through reprojection, and directly propagates 3D spatial gradients back
6

<!-- page 7 -->
to the normal prediction network, forming a closed-loop optimization path from 2D images to 3D
geometry.
Differentiable Rendering. We utilize the tile-based rasterizer from 3DGS [26] to combine nor-
mal maps with 3D ellipsoid parameters for 3D reprojection. First, camera viewpoints and intrin-
sic/extrinsic parameters are extracted from 3DGS parameters. Each 3D Gaussian ellipsoid is then
projected onto 2D image space for rendering, where the projected 2D Gaussian’s center position and
color are directly obtained from its 3D parameters. These 2D Gaussians are sorted based on their
3D opacity and covariance matrices to properly handle occlusion relationships during rendering. To
improve sorting efficiency, we partition the image plane into multiple tiles and sort 2D Gaussians
within each tile. The reprojection loss is optimized through stochastic gradient descent to minimize
reconstruction errors. Finally, the projected 2D Gaussians are composited into rendered images for
comparison with input images.
3D Reprojection Loss Calculation. Our joint reprojection loss function evaluate both geometric
structure and texture reconstruction:
The scale Loss Lscale measures geometric scale differences between reprojected and input images. For
3D Gaussians with scale parameters s = (sx, sy, sz) ∈R3, we minimize the smallest component:
LScale = ∥min(sx, sy, sz)∥
(17)
This forces Gaussian ellipsoids to flatten into thin surfaces, aligning their centers with object surfaces.
The Contour Loss Lcontour evaluates silhouette differences:
LContour = 1
N
N
X
i=1
|Ipred(i) −Igt(i)|
(18)
where Ipred(i) and Igt(i) denote pixel values at position i in predicted and ground truth images,
respectively.
The Structural Loss LSSIM measures color and luminance similarity:
LSSIM =
(2µIpredµIgt + C1)(2σIpredIgt + C2)
(µ2
Ipred + µ2
Igt + C1)(σ2
Ipred + σ2
Igt + C2)
(19)
where µ and σ represent means and variances, σIpredIgt denotes covariance, with C1 and C2 as
stabilization constants.
The whole 3D reprojection loss combines these components:
LReprojection = λScaleLScale + λContourLContour + λSSIMLSSIM
(20)
4
Experiments
4.1
Implementation Details
Hardware and Software Environments. We evaluated our method and the baseline methods by
using their original released implementations with default hyperparameters on the PC with a single
RTX 4090 GPU (24GB) running Ubuntu 20.04, Python 3.9, PyTorch 2.1.0, and CUDA 11.8.
Training Details. The training process of our method consists of three progressive optimization
stages: 1) In the initial phase (1,000 training steps), we freeze the conditional diffusion model and
train only the MLP using image-3D model pairs with a learning rate of 1e-4 and batch size 16 to
establish the image-to-3D ellipsoid parameter mapping. 2) During the second phase (5,000 steps), we
freeze the MLP and train the conditional diffusion model parameters using image inputs, reducing the
learning rate to 1e-5 and batch size to 8. 3) The final fine-tuning phase incorporates a 3D reprojection
joint loss function with a refined learning rate of 1e-5 and micro-batch size of 4. We implement a
10% geometric condition dropout strategy to enhance multi-view generation robustness, along with
an improved cosine annealing algorithm in the noise scheduling module to balance low-frequency
structural stability and high-frequency detail generation. To obtain normals for comparisons, we
retrained most of the baseline methods.
7

<!-- page 8 -->
Datasets. Following the baselines [15, 17–23], we train our model on the Objaverse dataset [2] con-
taining approximately 80K objects, and evaluate on Google Scanned Objects (GSO) [16] containing
1,030 scanned daily objects.
Evaluation Metrics. We evaluate normal estimation using three metrics: Mean Angular Error
(MAE), Median Angular Error (MedAE), and accuracy below threshold θ ∈[11.25◦, 22.5◦, 30◦].
MAE reflects overall reconstruction accuracy (lower values preferred), MedAE indicates typical error
levels (robust to outliers), while threshold-based metrics measure precision at different tolerance
levels: 11.25◦for detail-sensitive scenarios, 22.5◦for general reconstruction tasks, and 30◦for
overall usability. Unlike comparative studies focusing on 3D mesh reconstruction, we exclude PSNR,
SSIM, and LPIPS metrics as our method specifically targets normal estimation.
4.2
Comparative with Baselines
Figure 2: Comparative results with baselines
Table 1: Quantitative metrics results with baselines
Method
Angular Error
Accuracy (°)
MAE↓
MedAE↓
11.25↑
22.5↑
30↑
(a) Magic3D
19.2
15.3
45.4
65.2
71.0
(b) RealFusion
16.2
13.0
62.9
77.8
82.3
(c) SyncDreamer
18.2
17.4
49.2
69.1
73.5
(d) Zero123
17.9
16.1
57.4
74.5
77.2
(e) One-2-3-45
18.6
17.0
52.5
71.1
72.9
(f) Metric3D
14.3
11.7
60.7
78.2
82.9
(g) Wonder3D
13.9
11.2
61.0
77.9
83.1
(h) Ours
13.2
10.7
63.2
79.7
84.6
We evaluate our method with baselines including Magic3D [18], RealFusion [21], SyncDreamer [17],
Zero123 [19], One-2-3-45 [22], Metric3D [23] and Wonder3D [15], as demonstrated in Figure 2 and
Table 1. Methods with implicit representations, such as Magic3D and RealFusion, generate smooth
surfaces with low noise but exhibit limited sensitivity to high-frequency geometric details, resulting
in oversmoothed reconstructions. SyncDreamer and Zero123 excel in multi-view consistency due to
their synchronized generation frameworks. However, insufficient geometric regularization introduces
localized distortions, particularly in regions with complex topological structures. One-2-3-45 and
Metric3D achieve superior surface continuity through normal consistency optimization, yet their
8

<!-- page 9 -->
texture reconstruction capabilities remain constrained, often failing to preserve sharpness in intricate
patterns. Notably, while Wonder3D adopts a conditional diffusion framework similar to ours, the ab-
sence of explicit 3DGS-based geometric constraints leads to incomplete reconstructions and reduced
geometric fidelity. In contrast, our method synergistically integrates explicit geometric representa-
tions with diffusion priors, establishing state-of-the-art performance across all metrics—geometric
accuracy, texture detail preservation, and view consistency. Both quantitative evaluations and visual
comparisons substantiate the superiority of our framework. Additionally, we provide some supple-
mentary results for failure cases (e.g., mental kettle, crystal stone), which are presented in Appendix
B.
4.3
Ablation Study
Figure 3: Comparative results about our proposed components
Table 2: Quantitative metrics results about our proposed components
Component Type
Angular Error
Accuracy (°)
MAE↓
MedAE↓
11.25↑
22.5↑
30↑
(a) 3DGS Only
23.7
17.5
35.2
58.1
59.4
(b) Diffusion Only
20.1
14.8
38.5
62.1
70.5
(c) 3DGS+Diffusion+LScale
15.9
12.8
59.5
72.9
75.3
(d) 3DGS+Diffusion+LContour
14.3
11.9
61.8
76.4
80.1
(e) 3DGS+Diffusion+LSSIM
15.1
12.5
58.7
74.2
77.6
(f) 3DGS+Diffusion+LReprojection
13.2
10.7
63.2
79.7
84.6
We conduct five ablation groups: 1) MLP-only 3DGS network reveals limitations in capturing local de-
tails despite strong function approximation, resulting in blurred edges. 2) Conditional diffusion model
alone generates distorted geometries without geometric constraints. 3) 3DGS+Diffusion+LScale
loss demonstrates the regularization effect of multi-scale feature alignment. Its absence causes
noise amplification. 4) 3DGS+Diffusion+LContour loss makes the generated normal conform to the
three-dimensional contour of the object. 5) 3DGS+Diffusion+LSSIM loss helps the normal smoother.
Figure 3 and Table 2 confirm that optimal normal estimation requires all components.
5
Conclusion and Limitions
Conclusion. We propose an innovative framework, SINGAD, that integrates 3D Gaussian Splatting
(3DGS) with conditional diffusion models to address geometric consistency and data dependency
challenges in normal estimation. This end-to-end self-supervised framework achieves three key
advancements: 1) A light-interaction model reparameterizes 3DGS to explicitly associate diffuse irra-
diance with normal orientation, generating multi-scale geometric features aligned with physical light
transport principles. 2) A cross-domain feature fusion strategy guides the conditional diffusion model
through geometric feature injection, constraining iterative normal refinement while ensuring accurate
3D error propagation. 3) A normal reprojection optimization strategy based on the differentiable
rasterization renderer, jointly training network parameters via a 3D reprojection loss without requiring
ground-truth normal annotations. This reconstruction-driven approach represents a paradigm shift
from passive geometry perception to physics-aware decoupled generation.
Limitions. Current limitations include difficulties in reconstructing thin, light-transmissive objects
(e.g., glass), specular reflection objects (e.g., metal), and severely occluded structures in complex
9

<!-- page 10 -->
scenes (see Appendix B). Future work focuses on extending our method to video-based normal esti-
mation and 3D reconstruction applications. Ongoing research also investigates hybrid representations
to enhance reconstruction fidelity for transparent materials and occluded geometries.
Broader Impacts. We proposes a technique aimed at reconstructing the geometric structures of
three-dimensional objects. It presents significant potential for advancing 3D reconstruction and
scene understanding in applications such as augmented/virtual reality, robotics navigation, and
digital content creation. By enabling self-supervised normal estimation, it lowers the barrier to
high-quality 3D modeling for resource-constrained domains. However, it should be noted that our
method is restricted to surface reconstruction based on input image data, and cannot extrapolate to
generate three-dimensional models of unseen object categories (e.g., producing shoes models from
cup images). Furthermore, while the proposed approach demonstrates efficacy in indoor and outdoor
scene reconstruction, it explicitly excludes applications involving sensitive subjects such as facial
reconstruction. While our method relies on per-object optimization to generate normals, this approach
incurs substantial computational overhead that may raise environmental sustainability concerns.
The energy-intensive nature of iterative neural rendering processes could contribute to increased
carbon footprints, particularly when scaling to large-scale scene reconstruction tasks. To address
this limitation, developing energy-efficient network architectures through model compression and
parameter quantization will be prioritized in one of our future works, aiming to reduce computational
demands while preserving reconstruction fidelity.
Acknowledgments and Disclosure of Funding
6
Acknowledgments
This work was supported by in part of the “National Key Research and Development Program
(No.2023YFC3805901)”, the “National Natural Science Foundation of China (No.62172190)”, in
part of the “Taihu Talent-Innovative Leading Talent Plan Team” of Wuxi City(Certificate Date:
20241220(8)).
10

<!-- page 11 -->
References
[1] Ahmad Eftekhar, Abhinav Sax, Jitendra Malik, and Arsha Zamir. Omnidata: A scalable pipeline
for making multi-task mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 10786–10796, Nashville, TN,
USA, 2021.
[2] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt,
Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe
of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, Vancouver, Canada, 2023. CVPR.
[3] Lubor Ladicky, Bernhard Zeisl, and Marc Pollefeys. Discriminatively trained dense surface
normal estimation. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer
Vision – ECCV 2014, volume 8689 of Lecture Notes in Computer Science, pages 468–484,
Zurich, Switzerland, 2014. Springer.
[4] Xiaolong Wang, David Fouhey, and Abhinav Gupta. Designing deep networks for surface
normal estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 539–547, Boston, MA, USA, 2015.
[5] David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with
a common multi-scale convolutional architecture. In Proceedings of the IEEE International
Conference on Computer Vision, pages 2650–2658, Santiago, Chile, 2015.
[6] Boyang Li, Chunhua Shen, Yong Dai, Anton van den Hengel, and Ming-Ming He. Depth
and surface normal estimation from monocular images using regression on deep features and
hierarchical crfs. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 1119–1127, Boston, MA, USA, 2015.
[7] Aayush Bansal, Bryan C. Russell, and Abhinav Gupta. Marr revisited: 2d–3d alignment via
surface normal prediction. In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pages 5965–5974, Las Vegas, NV, USA, 2016.
[8] Wen Yin, Yanyu Liu, Chunhua Shen, and Yonggang Yan. Enforcing geometric constraints of
virtual normal for depth prediction. In Proceedings of the IEEE International Conference on
Computer Vision, pages 5684–5693, Seoul, Republic of Korea, 2019.
[9] Justin Liang, Namdar Homayounfar, Wei-Chiu Ma, Yuwen Xiong, Rui Hu, and Raquel Urtasun.
Polytransform: Deep polygon transformer for instance segmentation. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9128–9137, Seattle,
WA, USA, 2020.
[10] Zhi Shi, Yanyu Liu, and Xi Zhang. Planetr: Structure-guided transformers for 3d plane
recovery. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
1234–1245, Montreal, Canada, 2021.
[11] Geonmo Bae, Ignas Budvytis, and Roberto Cipolla. Estimating and exploiting the aleatoric
uncertainty in surface normal estimation. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 13137–13146, Montreal, Canada, 2021.
[12] Xiaoqing Qi, Ruofan Liao, Zhe Liu, Raquel Urtasun, and Jiaya Jia. Geonet: Geometric neural
network for joint depth and surface normal estimation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 283–291, Salt Lake City, UT,
USA, 2018.
[13] Xiaoqing Qi, Zhe Liu, Ruofan Liao, Philip H. S. Torr, Raquel Urtasun, and Jiaya Jia. Geonet++:
Iterative geometric neural network with edge-aware refinement for joint depth and surface
normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(2):969–
984, 2020.
[14] Xiaoxiao Long, Yiru Zheng, Yuchen Zheng, Bowen Tian, Chenglei Lin, Li Liu, Hu Zhao,
Guang Zhou, and Wei Wang. Adaptive surface normal constraint for geometric estimation from
monocular images. arXiv preprint arXiv:2402.05869, 2024.
11

<!-- page 12 -->
[15] Xianzhi Long, Yizhi Guo, Cheng Lin, and et al. Wonder3d: Single image to 3d using cross-
domain diffusion. arXiv preprint arXiv:2310.15008, 2023.
[16] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Rey-
mann, Thomas B. McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality
dataset of 3d scanned household items. In Proceedings of the IEEE International Conference
on Robotics and Automation, Philadelphia, PA, USA, 2022.
[17] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping
Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv
preprint arXiv:2309.03453, 2023.
[18] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten
Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d
content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 1998–2008, Vancouver, Canada, 2023.
[19] Rui Liu, Rui Wu, Bram Van Hoorick, and et al. Zero-1-to-3: Zero-shot one image to 3d
object. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
9298–9309, Paris, France, 2023.
[20] Rui Shi, Hao Chen, Zhi Zhang, and et al. Zero123++: A single image to consistent multi-view
diffusion base model. arXiv preprint arXiv:2310.15110, 2023.
[21] Da Guo, Zhaoxin Li, Xiaowei Gao, and et al. Realfusion: A reliable deep learning-based
spatiotemporal fusion framework for generating seamless fine-resolution imagery. Remote
Sensing of Environment, 321:114689, 2025.
[22] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, and Hao Su. One-2-3-45:
Any single image to 3d mesh in 45 seconds without per-shape optimization. arXiv preprint
arXiv:2306.16928, 2023.
[23] Wen Yin, Cheng Zhang, Hao Chen, and et al. Metric3d: Towards zero-shot metric 3d prediction
from a single image. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pages 1–12, Paris, France, 2023.
[24] Wen Yin, Cheng Zhang, Hao Chen, and et al. Metric3d v2: A versatile monocular geometric
foundation model for zero-shot metric depth and surface normal estimation. 2024.
[25] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoor-
thi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis.
Communications of the ACM, 65(1):99–106, 2021.
[26] Bernhard Kerbl and et al. 3d gaussian splatting for real-time radiance field rendering. ACM
Transactions on Graphics, 42(4):Article 1, 2023.
[27] Max Born and Emil Wolf. Principles of Optics. Cambridge University Press, Cambridge, UK,
7th edition, 1999.
[28] Matt Pharr, Wenzel Jakob, and Greg Humphreys. Physically Based Rendering: From Theory to
Implementation. The MIT Press, Cambridge, MA, USA, 4th edition, 2023.
[29] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by
back-propagating errors. Nature, 323(6088):533–536, 1986.
[30] Alexey Dosovitskiy and et al. An image is worth 16×16 words: Transformers for image
recognition at scale. In International Conference on Learning Representations, 2021.
[31] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for im-
age recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 770–778, Las Vegas, NV, USA, 2016.
[32] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie.
Feature pyramid networks for object detection. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pages 2117–2125, Honolulu, HI, USA, 2017.
12

<!-- page 13 -->
[33] Ian T. Jolliffe. Principal Component Analysis. Springer, New York, NY, USA, 2nd edition,
2002.
[34] P. C. Mahalanobis. On the generalized distance in statistics. Proceedings of the National
Institute of Sciences of India, 2(1):49–55, 1936.
[35] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In
Advances in Neural Information Processing Systems, pages 6840–6851, 2020.
[36] Shlomi Steinberg, Pradeep Sen, and Ling-Qi Yan. Towards practical physical-optics rendering.
ACM Transactions on Graphics (TOG), 41(4):1–24, 2022. Proceedings of SIGGRAPH 2022.
[37] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T. Barron, and Pratul P.
Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pages 5481–5490, 2022.
[38] Meng Wei, Qianyi Wu, Jianmin Zheng, Hamid Rezatofighi, and Jianfei Cai. Normal-gs:
3d gaussian splatting with normal-involved rendering. In Advances in Neural Information
Processing Systems (NeurIPS), 2024. arXiv preprint arXiv:2410.20593.
[39] Ling-Qi Yan, Miloš Hašan, Steve Marschner, and Ravi Ramamoorthi. Position-normal distribu-
tions for efficient rendering of specular microstructure. ACM Transactions on Graphics (TOG),
35(4):1–9, 2016. Proceedings of SIGGRAPH 2016.
[40] Ling-Qi Yan, Miloš Hašan, Bruce Walter, Steve Marschner, and Ravi Ramamoorthi. Rendering
specular microgeometry with wave optics. ACM Transactions on Graphics (TOG), 37(4):1–10,
2018. Proceedings of SIGGRAPH 2018.
[41] Yang Zhou, Songyin Wu, and Ling-Qi Yan. Unified gaussian primitives for scene representation
and rendering. arXiv preprint arXiv:2406.09733, 2024.
13

<!-- page 14 -->
Technical Appendices and Supplementary Material
A
Mathematical Derivation
The Light-Interaction Model Construction. While object surfaces appear relatively flat at macro-
scopic scales, they exhibit complex height variations at microscopic scales as illustrated in Figure
4.
Figure 4: Schematic diagram of light interaction with surface micro-geometry structures
The micro-geometry structures of surfaces influence light propagation through surface height gradients
∇h(x). According to the Huygens-Fresnel principle [27], the light interaction process can be
decoupled into the superposition of local reflections and global interference effects. Building upon
the aforementioned theoretical framework, we propose a height gradient-based light-interaction
model for subsequent 3DGS reparameterization. As shown in Equation 21, the resulting image I(x)
represents the coupled light-surface interaction:
I(x) =
Z
Ω
L(ωi) · R(ωi, ωo, ∇h(x)) · (ωi · n)dωi
(21)
where ωi and ωo denote incoming and outgoing light directions, respectively. The incident angle
θi between ωi and surface normal n determines the attenuation of incident irradiance L(ωi). The
Bidirectional Reflectance Distribution Function (BRDF) [28] R is a function used to describe how
the surface of an object reflects and scatters incident light ωi, generates reflected light ωo, and BRDF
defines the reflection characteristics of light incident from different directions in different outgoing
directions.
To extract observable outgoing irradiance from viewpoint directions, we model outgoing radiance Lo
as a function of incident irradiance Li. Specifically, the radiance captured by cameras equals Lo(ωo)
at point x is:
I(x) = Lo(ωo) =
Z
Ω+ Li(ωi) · R(ωi, ωo, ∇h(x)) · (ωi · n)dωi −LA(ωo)
(22)
The Ω+ subset ensures surface normals point outward while viewpoints face inward, maintaining
≥90◦between normals and outgoing rays. LA(ωo) represents inward irradiance along normals,
corresponding to the Ω−subset.
For a detailed derivation and illustration of how to model the relationship between the image and the
BRDF, we recommend referring to these original papers [36–38].
3DGS Reparameterization. Assuming Lambertian reflectance [27], we consider only outward
diffuse irradiance contributing to the RGB image. Under this model, complex BRDFs reduce to
spatially varying albedo kD. Simplifying Equation (3) by omitting LA(ωo) and substituting R with
kD gives:
LD =
Z
Ω+ Li(ωi) · kD · (ωi · n)dωi = kD ·
Z
Ω+ Li(ωi) · (ωi · n)dωi
(23)
14

<!-- page 15 -->
To connect height gradients with diffuse irradiance, we reformulate LD as a normal-dependent
function by isolating the normal term n:
LD = kD · n ·
Z
Ω+ Li(ωi) · ωidωi
(24)
Inspired by Physicallly-Based Rendering (PBR) theory [28], we reparameterize the integral using the
Gabor kernel Ga:
Z
Ω+ Li(ωi) · ωidωi = Ga(x; ξ, α) = G2D(x; ξ) · e−i·2π(α·x)
(25)
Here, G2D(x; ξ) =
1
2πξ2 e−∥x∥2/(2ξ2) denotes normalized isotropic Gaussian distribution, where ξ
derives from scalar diffraction models [27]. The parameter α = 2∇h(x)/λ encodes local height
gradients over k × k neighborhoods.
Rather than explicitly solving the Gabor kernel or specifying diffraction models, we utilize deep
networks to approximate this unified representation. To simplify parameters for network learning, we
model the surface color as:
Lo(ωo) = kD · n · Ga
(26)
This formulation enables gradient propagation from surface colors to normals through the Gabor
kernel derivatives:
∂Lo
∂n = kD · Ga + kD · n · ∂Ga
∂∇h · ∂∇h
∂n
(27)
The differentiable nature of the Gabor kernel permits chain rule propagation through MLPs, enabling
direct learning of ∇h(x)-to-normal mappings while satisfying diffusion model requirements for
discrete parameters.
Our reparameterization framework explicitly quantifies optical processes in 3DGS through Gabor
kernel derivatives that enable surface gradient propagation, where the kernel parameters (wavelength
λ and Gaussian scale ξ) are implicitly learned via an MLP, with λ dynamically adjusted by input
image’s frequency-domain features and ξ generated through a surface roughness estimation module.
Specifically, the MLP’s output layer converts geometric features to Gabor parameters via linear
projection while ensuring physical consistency with local surface gradients ∇h(x), avoiding manual
parameter tuning complexity through this unified learning strategy while preserving light transport
physics, outperforming approaches using complex ViT networks.
For a detailed derivation and illustration of how to analyze BRDF with the Gabor kernel via the PBR
theory, we recommend referring to these original papers [39–41].
B
More Experiment Results
As illustrated in Figure 5 and 6, the comparative analysis of rendering outcomes reveals distinct
characteristics across methodologies: Neural radiance fields excel in photorealistic novel view
synthesis for organic shapes like the tiger model, yet struggle with specular highlights on metallic
surfaces as evidenced by the helmet’s inconsistent reflectance. Volumetric approaches demonstrate
robust performance in reconstructing complex topologies such as the rocky structures, though
they introduce over-smoothing artifacts in fine details like the owl’s feather patterns. Explicit
surface reconstruction methods achieve superior geometric fidelity for manufactured objects (e.g., the
teapot’s spout-handle junction), but require dense input views to resolve ambiguities in textureless
regions. While our proposed hybrid representation balances material-aware rendering through the
purple/blue normal map’s continuous curvature transitions and maintains structural integrity across
scale variations, particularly notable in the tomato’s subtle surface undulations and helmet visor’s
sharp edges that other methods either oversmooth or fragment.
15

<!-- page 16 -->
Figure 5: More comparative results with baselines. Images from Google Scanned Objects (GSO) [16]
Figure 6: More comparative results with baselines. Images from Wonder3D [15]
Figure 7 demonstrates our method’s capability to generate multi-view projections for normal es-
timation constraints. While these projections are not optimized for high-fidelity 3D geometry
reconstruction like dedicated neural rendering pipelines, the reconstructed surfaces exhibit sufficient
multi-scale geometric features (e.g., curvature variations and occlusion boundaries) to adequately
serve as conditional inputs for our diffusion-based refinement framework. This strategic trade-off
prioritizes normal estimation accuracy over exhaustive volumetric modeling, aligning with our focus
on extracting perceptually salient geometric priors rather than photometric precision in 3D space.
16

<!-- page 17 -->
Figure 7: Example for multi-view generation. Images from Google Scanned Objects (GSO) [16]
However, our experiments reveal two limitations common to existing normal estimation methods:
1) Under strong specular reflections (e.g., metallic helmet surfaces), the algorithm erroneously
interprets highlight regions as background components, generating spurious normal vectors that
corrupt surface continuity. Due to the Lambertian reflection [27] we used, a consequence of current
photometric constraints failing to decouple diffuse albedo and specular components. 2) In complex
occlusion scenarios (rocky structures with interpenetrating geometries), the absence of explicit
visibility reasoning leads to pathological cases where occluded background regions are incorrectly
incorporated into foreground normal maps. These failure cases, particularly evident in the helmet’s
inconsistent surface orientation near highlight zones and the rocky structure’s phantom geometry
artifacts, highlight critical needs for future work in dynamic specularity-aware separation modules
and hierarchical scene parsing frameworks.
C
Assets License
In Table 3, we list the licenses of all the existing assets, including the code and data we have used in
this work.
Asset
License Link
3D Gaussian Splatting [26]
https://github.com/graphdeco-inria/
gaussian-splatting/blob/main/LICENSE.md
Magic3D [18]
https://github.com/chinhsuanwu/dreamfusionacc/
blob/master/LICENSE
RealFusion [21]
https://github.com/lukemelas/realfusion/blob/
main/LICENSE
SyncDreamer [17]
https://github.com/liuyuan-pal/SyncDreamer/blob/
main/LICENSE
Zero123 [19]
https://github.com/cvlab-columbia/zero123/blob/
main/LICENSE
One-2-3-45 [22]
https://github.com/One-2-3-45/One-2-3-45/blob/
master/LICENSE
Metric3D [23]
https://github.com/YvanYin/Metric3D/blob/main/
LICENSE
Wonder3D [15]
https://github.com/xxlong0/Wonder3D/blob/main/
LICENSE
Objaverse Dataset [2]
https://objaverse.allenai.org/docs/intro
Google Scanned Objects (GSO) [16]
https://research.google/blog/scanned-objects-
by-google-research-a-dataset-of-
3d-scanned-common-household-items
Table 3: Assets and corresponding license links
17

<!-- page 18 -->
NeurIPS Paper Checklist
1. Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The abstract and introduction clearly articulate the key limitations, multi-
view geometric inconsistency and data dependency, of existing diffusion-based methods in
normal estimation from a single image, and present three core innovations of our proposed
framework SINGAD: 1) Physics-driven light-interaction 3DGS reparameterization for multi-
scale geometric consistency. 2) Cross-domain feature fusion with geometric prior constraints.
3) Differentiable reprojection loss for self-supervised optimization. These claims are fully
supported by methodological details and experimental validation. The contributions align
precisely with the paper’s scope as framed in its title, "Self-Supervised Normal Estimation
via 3D Gaussian Splatting Guided Diffusion from a Single Image," establishing a novel
paradigm shift from data-driven learning to physics-aware modeling.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2. Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: This paper discusses the limitations in Section 5, and demonstrates some
failure cases in Appendix B.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that
the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
18

<!-- page 19 -->
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3. Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: In the theory derivation in Sections 3.1, 3.2 and 3.3, we point out the assumption
of Light-Interaction Model, Lambertian surface, 3DGS Reparameterization, Conditional Dif-
fusion Model, and 3D ReProjection Loss for proposed components. Moreover, we provide a
more detailed derivation process for Light-Interaction Model and 3DGS Reparameterization
in Appendix A.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: The paper provides sufficient information for the reproducibility of its main
results. Key reproducibility elements include: 1) Architectural Details. The network
architecture (e.g., cross-domain fusion module with f7×7 convolutions, 5-layer MLP with
ReLU/batch normalization) is explicitly specified, including feature binding mechanisms
(Equation 10-19) in Section 3. 2) Loss Functions. All loss components (geometry, contour,
SSIM) and their mathematical expressions (Eq. 20) are thoroughly defined, even though
exact loss weights may need tuning, the relative balance is contextually inferable from the
problem setup. 3) Datasets Sources. Public datasets (Objaverse, GSO) and preprocessing
steps (geometric dropout) are clearly cited, enabling data acquisition and augmentation
replication in Section 4.1. 4) Training Details. Key hyperparameters (learning rates: 1e-
4/1e-5, cosine annealing schedule) and multi-stage pipeline are documented, with structural
decisions (e.g., freezing in Stage II) ensuring reproducibility in Section 4.1. 5) Metrics and
Evaluation. Metrics and evaluation steps are comprehensively described for cross-checking
results in Section 4.1. While code is not released, the network ’s modular design and reliance
on public datasets ensure that motivated researchers can reconstruct the framework using
standard libraries by filling in implementation gaps.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
19

<!-- page 20 -->
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [No]
Justification: The details for implementation are provided in Section 4.1, and we will release
our code upon publication.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6. Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
20

<!-- page 21 -->
Answer: [Yes]
Justification: The training and testing details are elaborated in Section 4 and Appendix B.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [No]
Justification: The paper does not report error bars, confidence intervals, or statistical signifi-
cance tests for its experimental results. We evaluate normal estimation using three metrics:
Mean Angular Error (MAE), Median Angular Error (MedAE), and accuracy below threshold
θ ∈[11.25◦, 22.5◦, 30◦], just like our baselines do. In addition, the dataset is too large to
report the results of each scenario. We calculated the average of all scenarios in the whole
Google Scanned Objects (GSO) test set, which reflects the overall performance level of the
algorithm in this article.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: We provide the related information in the experiments Section 4.1.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
21

<!-- page 22 -->
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9. Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: The research is carefully conducted with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: The discussion about potential negative social impacts is mentioned in Section
5
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11. Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: This paper aims to support novel normal estimation, which poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
22

<!-- page 23 -->
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12. Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: We include the assets used in our paper (codes and Datasets) in Appendix C
and References.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: This paper does not release new assets during the time of submission.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
23

<!-- page 24 -->
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15. Institutional review board (IRB) approvals or equivalent for research with human
subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with
human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.
16. Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and does not impact the core methodology,
scientific rigorousness, or originality of the research, declaration is not required.
Answer: [NA]
Justification: The LLMs are only used for the grammar and spelling check in this paper.
Guidelines:
• The answer NA means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
24
